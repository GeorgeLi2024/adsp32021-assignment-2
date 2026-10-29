from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one assignment experiment.")
    parser.add_argument("--feature-data", type=Path, required=True)
    parser.add_argument("--split-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-version", required=True, choices=["v1", "v2"])
    parser.add_argument("--hyperparameter-set", required=True, choices=["shallow", "deeper"])
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entity_id = config["entity_column"]
    target = config["target_column"]
    features = config["feature_versions"][args.feature_version]["columns"]
    parameters = config["hyperparameter_sets"][args.hyperparameter_set]
    feature_data = pd.read_parquet(args.feature_data)
    split_data = pd.read_csv(args.split_data)
    if feature_data[entity_id].duplicated().any():
        raise ValueError("Feature data contains duplicate athlete IDs.")
    if set(feature_data[target].unique()) != set(config["class_order"]):
        raise ValueError("Feature data does not contain exactly the configured target classes.")
    if not set([entity_id, target, *features]).issubset(feature_data.columns):
        raise ValueError("Feature data is missing a configured feature or target column.")

    train_ids = split_data.loc[split_data["split"] == "train", entity_id]
    test_ids = split_data.loc[split_data["split"] == "test", entity_id]
    if len(train_ids) + len(test_ids) != len(feature_data):
        raise ValueError("The shared train/test split does not cover the feature dataset.")
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test IDs overlap.")
    if set(train_ids) | set(test_ids) != set(feature_data[entity_id]):
        raise ValueError("Train/test IDs do not match the Feast feature data.")

    indexed = feature_data.set_index(entity_id)
    train = indexed.loc[train_ids.to_numpy()]
    test = indexed.loc[test_ids.to_numpy()]
    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    model = DecisionTreeClassifier(
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        class_weight=config["model"]["class_weight"],
        random_state=config["model"]["random_state"],
    )
    carbon_path = args.output_dir / "codecarbon.csv"
    tracker = OfflineEmissionsTracker(
        project_name=args.run_name,
        country_iso_code=config["codecarbon"]["country_iso_code"],
        region=config["codecarbon"]["region"],
        tracking_mode=config["codecarbon"]["tracking_mode"],
        measure_power_secs=config["codecarbon"]["measure_power_secs"],
        output_dir=str(args.output_dir),
        output_file=carbon_path.name,
        log_level="warning",
    )

    tracker.start()
    start = time.perf_counter()
    try:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(config["positive_class"])]
    finally:
        training_seconds = time.perf_counter() - start
        emissions_from_tracker = tracker.stop()

    positive = config["positive_class"]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision_female": precision_score(y_test, predictions, pos_label=positive, zero_division=0),
        "recall_female": recall_score(y_test, predictions, pos_label=positive, zero_division=0),
        "f1_female": f1_score(y_test, predictions, pos_label=positive, zero_division=0),
        "roc_auc": roc_auc_score((y_test == positive).astype(int), probabilities),
        "training_seconds": training_seconds,
        "tree_depth": int(model.get_depth()),
        "tree_leaves": int(model.get_n_leaves()),
    }

    if not carbon_path.is_file():
        raise RuntimeError("CodeCarbon did not write its emissions CSV.")
    carbon_record = pd.read_csv(carbon_path).iloc[-1]
    if "emissions" not in carbon_record or "energy_consumed" not in carbon_record or "duration" not in carbon_record:
        raise RuntimeError("CodeCarbon output is missing emissions, energy, or duration fields.")
    emission_value = float(carbon_record["emissions"])
    if emissions_from_tracker is not None and not np.isclose(
        emission_value, float(emissions_from_tracker), rtol=1e-9, atol=0
    ):
        raise RuntimeError("CodeCarbon return value and CSV emissions value disagree.")
    metrics["estimated_emissions_kg"] = emission_value
    metrics["estimated_energy_kwh"] = float(carbon_record["energy_consumed"])
    metrics["codecarbon_duration_seconds"] = float(carbon_record["duration"])

    labels = config["class_order"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    matrix_frame.to_csv(args.output_dir / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=labels)
    axis.set_yticks([0, 1], labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(args.run_name)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    model_path = args.output_dir / "model.joblib"
    joblib.dump(model, model_path)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({key: float(value) for key, value in metrics.items()}, file, indent=2)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "feature_version": args.feature_version,
                "hyperparameter_set": args.hyperparameter_set,
                "target": target,
                "positive_class": positive,
                "model_family": config["model"]["name"],
                "feature_store": "Feast",
                "carbon_estimator": "CodeCarbon OfflineEmissionsTracker",
            }
        )
        mlflow.log_params(
            {
                "feature_version": args.feature_version,
                "hyperparameter_set": args.hyperparameter_set,
                "max_depth": parameters["max_depth"],
                "min_samples_leaf": parameters["min_samples_leaf"],
                "class_weight": config["model"]["class_weight"],
                "split_random_state": config["split"]["random_state"],
                "train_rows": len(train),
                "test_rows": len(test),
                "features": ",".join(features),
                "carbon_country": config["codecarbon"]["country_iso_code"],
                "carbon_region": config["codecarbon"]["region"],
                "carbon_tracking_mode": config["codecarbon"]["tracking_mode"],
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        for artifact in [
            carbon_path,
            args.output_dir / "confusion_matrix.csv",
            args.output_dir / "confusion_matrix.png",
            model_path,
            args.output_dir / "metrics.json",
        ]:
            mlflow.log_artifact(str(artifact))

    print(json.dumps({"run_name": args.run_name, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
