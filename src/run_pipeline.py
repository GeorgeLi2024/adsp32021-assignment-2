from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float64
from feast.value_type import ValueType
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
FEAST_REPO = ROOT / "feature_repo"
TRACKING_DB = ROOT / "mlflow.db"
EXPERIMENT_NAME = "ADSP 32021 Assignment 2"
EVENT_TIMESTAMP = "event_timestamp"


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def make_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "outputs" / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_cohort(data_path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    versions = config["feature_versions"]
    feature_columns = list(dict.fromkeys(
        column
        for version in versions.values()
        for column in version["columns"]
    ))
    target = config["target_column"]
    entity_id = config["entity_column"]
    required = [entity_id, target, *feature_columns]

    data = pd.read_csv(data_path)
    input_rows = len(data)
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()

    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    for column in [entity_id, *feature_columns]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    known_label = data[target].isin(config["class_order"])
    unique_id = data[entity_id].notna() & ~data[entity_id].duplicated(keep=False)
    valid_values = pd.Series(True, index=data.index)
    for column in feature_columns:
        low, high = config["valid_ranges"][column]
        valid_values &= data[column].between(low, high)

    eligible = known_label & unique_id & valid_values
    cohort = data.loc[eligible, [entity_id, target, *feature_columns]].copy()
    cohort[entity_id] = cohort[entity_id].astype("int64")
    cohort = cohort.sort_values(entity_id).reset_index(drop=True)

    if cohort.empty:
        raise ValueError("No rows remain after the documented data-quality filters.")
    if cohort[entity_id].duplicated().any():
        raise ValueError("Athlete IDs must be unique in the modeling cohort.")
    if set(cohort[target].unique()) != set(config["class_order"]):
        raise ValueError("Both configured target classes must remain after filtering.")

    stats = {
        "input_rows": input_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "eligible_rows": len(cohort),
        "excluded_rows": input_rows - len(cohort),
        "class_counts": {
            str(key): int(value)
            for key, value in cohort[target].value_counts().sort_index().items()
        },
        "feature_columns_v1": versions["v1"]["columns"],
        "feature_columns_v2": versions["v2"]["columns"],
    }
    return cohort, stats


def write_split(cohort: pd.DataFrame, config: dict, path: Path) -> tuple[list[int], list[int]]:
    entity_id = config["entity_column"]
    target = config["target_column"]
    split = config["split"]
    train_ids, test_ids = train_test_split(
        cohort[entity_id].to_numpy(),
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=cohort[target],
    )
    split_data = pd.concat(
        [
            pd.DataFrame({entity_id: train_ids, "split": "train"}),
            pd.DataFrame({entity_id: test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    split_data.to_csv(path, index=False)
    return train_ids.tolist(), test_ids.tolist()


def create_feature_store(
    cohort: pd.DataFrame,
    config: dict,
    feature_data_path: Path,
) -> tuple[FeatureStore, dict[str, FeatureView]]:
    entity_id = config["entity_column"]
    all_features = list(dict.fromkeys(
        column
        for version in config["feature_versions"].values()
        for column in version["columns"]
    ))

    source_data = cohort[[entity_id, *all_features]].copy()
    source_data[EVENT_TIMESTAMP] = pd.to_datetime(
        config["snapshot_timestamp"], utc=True
    )
    source_data.to_parquet(feature_data_path, index=False)

    athlete = Entity(
        name="athlete",
        join_keys=[entity_id],
        value_type=ValueType.INT64,
        description="One CrossFit athlete identified by the source athlete_id.",
    )
    source = FileSource(
        name="athlete_csv_snapshot",
        path=str(feature_data_path.resolve()),
        file_format=ParquetFormat(),
        timestamp_field=EVENT_TIMESTAMP,
        description="Selected columns from the supplied athletes.csv snapshot.",
    )

    views: dict[str, FeatureView] = {}
    services = []
    for version, definition in config["feature_versions"].items():
        view_name = f"athlete_features_{version}"
        view = FeatureView(
            name=view_name,
            source=source,
            entities=[athlete],
            schema=[Field(name=column, dtype=Float64) for column in definition["columns"]],
            online=True,
            description=f"{version}: {definition['name']}",
            tags={"feature_version": version},
        )
        service = FeatureService(
            name=f"athlete_model_{version}",
            features=[view],
            description=f"Feature service for model comparison {version}.",
            tags={"feature_version": version},
        )
        views[version] = view
        services.append(service)

    store = FeatureStore(repo_path=str(FEAST_REPO))
    store.apply([athlete, *views.values(), *services])
    return store, views


def read_feature_versions(
    store: FeatureStore,
    cohort: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> dict[str, Path]:
    entity_id = config["entity_column"]
    target = config["target_column"]
    entity_df = cohort[[entity_id]].copy()
    entity_df[EVENT_TIMESTAMP] = pd.to_datetime(config["snapshot_timestamp"], utc=True)
    paths: dict[str, Path] = {}

    for version, definition in config["feature_versions"].items():
        service = store.get_feature_service(f"athlete_model_{version}")
        retrieved = store.get_historical_features(
            entity_df=entity_df,
            features=service,
        ).to_df()

        if len(retrieved) != len(cohort):
            raise ValueError(f"Feast {version} read {len(retrieved)} rows, expected {len(cohort)}.")
        if retrieved[entity_id].duplicated().any():
            raise ValueError(f"Feast {version} returned duplicate athlete IDs.")
        if set(retrieved[entity_id]) != set(cohort[entity_id]):
            raise ValueError(f"Feast {version} read does not match the expected athlete IDs.")

        features = definition["columns"]
        expected = cohort.set_index(entity_id)[features].sort_index()
        actual = retrieved.set_index(entity_id)[features].sort_index()
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-8)

        result = retrieved[[entity_id, *features]].merge(
            cohort[[entity_id, target]],
            on=entity_id,
            how="inner",
            validate="one_to_one",
        ).sort_values(entity_id)
        output_path = run_dir / f"features_{version}.parquet"
        result.to_parquet(output_path, index=False)
        paths[version] = output_path

    return paths


def run_mlflow_projects(
    config_path: Path,
    split_path: Path,
    feature_paths: dict[str, Path],
    run_dir: Path,
    config: dict,
) -> list[dict]:
    tracking_uri = f"sqlite:///{TRACKING_DB}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for version, feature_data in feature_paths.items():
        for parameter_set, parameters in config["hyperparameter_sets"].items():
            run_name = f"{version}_{parameter_set}"
            output_dir = run_dir / "model_runs" / run_name
            output_dir.mkdir(parents=True, exist_ok=False)
            submitted = mlflow.projects.run(
                uri=str(ROOT),
                entry_point="train",
                parameters={
                    "feature_data": str(feature_data.resolve()),
                    "split_data": str(split_path.resolve()),
                    "config": str(config_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "feature_version": version,
                    "hyperparameter_set": parameter_set,
                    "run_name": run_name,
                },
                experiment_name=EXPERIMENT_NAME,
                backend="local",
                env_manager="local",
                synchronous=True,
                run_name=run_name,
            )
            tracked = mlflow.get_run(submitted.run_id)
            if tracked.info.status != "FINISHED":
                raise RuntimeError(f"MLflow project {run_name} ended with {tracked.info.status}.")
            results.append(
                {
                    "run_name": run_name,
                    "run_id": tracked.info.run_id,
                    "feature_version": version,
                    "hyperparameter_set": parameter_set,
                    **parameters,
                    **tracked.data.metrics,
                }
            )
    return results


def make_plots(results: pd.DataFrame, output_dir: Path) -> None:
    labels = results["run_name"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["accuracy", "balanced_accuracy", "f1_female", "roc_auc"]
    results.plot(x="run_name", y=metrics, kind="bar", ax=axes[0])
    axes[0].set_title("Held-out classification metrics")
    axes[0].set_xlabel("Feature version and parameter set")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(loc="lower right")

    axes[1].bar(labels, results["estimated_emissions_kg"])
    axes[1].set_title("CodeCarbon estimated emissions")
    axes[1].set_xlabel("Feature version and parameter set")
    axes[1].set_ylabel("kg CO2e (estimated)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for axis, row in zip(axes.flat, results.to_dict("records")):
        confusion = pd.read_csv(output_dir / "model_runs" / row["run_name"] / "confusion_matrix.csv", index_col=0)
        axis.imshow(confusion.to_numpy(), cmap="Blues")
        axis.set_title(row["run_name"])
        axis.set_xticks([0, 1], labels=["Female", "Male"])
        axis.set_yticks([0, 1], labels=["Female", "Male"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, int(confusion.iloc[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Feast features and launch four MLflow Project runs.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "athletes.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = read_config(args.config)
    run_dir = make_run_dir()

    cohort, stats = load_cohort(args.data, config)
    stats["train_test_split"] = config["split"]
    stats["snapshot_timestamp"] = config["snapshot_timestamp"]
    with (run_dir / "cohort_summary.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    split_path = run_dir / "split_ids.csv"
    train_ids, test_ids = write_split(cohort, config, split_path)
    stats["train_rows"] = len(train_ids)
    stats["test_rows"] = len(test_ids)

    feature_data_path = FEAST_REPO / "data" / "athletes_features.parquet"
    feature_data_path.parent.mkdir(parents=True, exist_ok=True)
    store, _ = create_feature_store(cohort, config, feature_data_path)
    feature_paths = read_feature_versions(store, cohort, config, run_dir)

    results = run_mlflow_projects(args.config, split_path, feature_paths, run_dir, config)
    results_df = pd.DataFrame(results)
    results_df.to_csv(run_dir / "results.csv", index=False, float_format="%.12g")
    make_plots(results_df, run_dir)

    stats["train_rows"] = len(train_ids)
    stats["test_rows"] = len(test_ids)
    stats["feast_read_rows"] = {version: len(pd.read_parquet(path)) for version, path in feature_paths.items()}
    with (run_dir / "cohort_summary.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(json.dumps({"run_dir": str(run_dir), "cohort": stats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
