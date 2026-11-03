#!/usr/bin/env python3
"""Build local Feast snapshots for ADSP Assignment 2 training inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SNAPSHOT_TEXT = "2026-10-18T00:00:00Z"
SNAPSHOT = pd.Timestamp(SNAPSHOT_TEXT)
COHORT_REQUIRED = ("athlete_id", "total_lift", "split")
BASE_FEATURES = ("age", "height", "weight", "is_female")
ENGINEERED_FEATURES = (
    "age_squared",
    "height_squared",
    "weight_squared",
    "age_x_weight",
    "age_x_height",
    "weight_x_height",
)
FEATURES_BY_VERSION = {
    "v1": BASE_FEATURES,
    "v2": BASE_FEATURES + ENGINEERED_FEATURES,
}
FEATURE_VIEW_NAMES = {
    "v1": "athlete_features_v1",
    "v2": "athlete_features_v2",
}
FEATURE_SERVICE_NAMES = {
    "v1": "athlete_feature_service_v1",
    "v2": "athlete_feature_service_v2",
}
SPLITS = {"train", "validation", "test"}
MAX_ONLINE_SAMPLE_ROWS = 5


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} has duplicate column names")


def _validate_ids(frames: dict[str, pd.DataFrame]) -> Any:
    """Validate unique IDs, matching key types, and equal key sets without reordering."""
    id_modes: dict[str, str] = {}
    id_sets: dict[str, set[Any]] = {}

    for label, frame in frames.items():
        values = frame["athlete_id"]
        if values.isna().any():
            raise ValueError(f"{label} has missing athlete_id values")

        raw = values.tolist()
        if all(isinstance(value, str) for value in raw):
            if any(not value or value.strip() != value for value in raw):
                raise ValueError(f"{label} athlete_id values must be nonblank strings without surrounding whitespace")
            mode = "string"
        elif all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in raw
        ):
            mode = "int64"
            frame["athlete_id"] = values.astype("int64")
        else:
            raise ValueError(
                f"{label} athlete_id values must consistently be strings or integers; "
                "the pipeline preserves the input key representation"
            )

        if frame["athlete_id"].duplicated().any():
            raise ValueError(f"{label} athlete_id values must be unique")
        id_modes[label] = mode
        id_sets[label] = set(frame["athlete_id"].tolist())

    if len(set(id_modes.values())) != 1:
        raise ValueError("athlete_id types must match across cohort and feature inputs")

    expected_ids = id_sets["cohort"]
    for label in ("features_v1", "features_v2"):
        if id_sets[label] != expected_ids:
            missing = len(expected_ids - id_sets[label])
            extra = len(id_sets[label] - expected_ids)
            raise ValueError(
                f"{label} athlete_id set must match cohort athlete_id values "
                f"(missing={missing}, extra={extra})"
            )
    return next(iter(id_modes.values()))


def _validate_snapshot(frame: pd.DataFrame, label: str) -> None:
    raw_timestamps = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    if raw_timestamps.isna().any():
        raise ValueError(f"{label} event_timestamp contains missing or invalid timestamps")
    if not raw_timestamps.eq(SNAPSHOT).all():
        actual = sorted({timestamp.isoformat() for timestamp in raw_timestamps.unique()})[:3]
        raise ValueError(
            f"{label} event_timestamp mismatch: expected the synthetic UTC snapshot "
            f"{SNAPSHOT_TEXT}, found {actual}"
        )
    frame["event_timestamp"] = raw_timestamps


def _validate_numeric_values(frame: pd.DataFrame, feature_names: tuple[str, ...], label: str) -> None:
    for feature in feature_names:
        raw = frame[feature]
        if raw.isna().any():
            raise ValueError(f"{label} predictor {feature} has missing values")
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{label} predictor {feature} has nonnumeric values")
        values = numeric.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} predictor {feature} must contain only finite values")
        frame[feature] = numeric.astype("float64")


def _validate_gender_feature(frame: pd.DataFrame, label: str) -> None:
    values = frame["is_female"]
    if values.isna().any():
        raise ValueError(f"{label} predictor is_female has missing values")

    valid = values.map(
        lambda value: isinstance(value, (bool, np.bool_))
        or (
            isinstance(value, (int, np.integer, float, np.floating))
            and value in (0, 1)
        )
    )
    if not valid.all():
        raise ValueError(f"{label} predictor is_female must contain only boolean values")
    frame["is_female"] = values.astype(bool)


def _read_inputs(data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any], str]:
    cohort_path = data_dir / "cohort.parquet"
    feature_paths = {
        "v1": data_dir / "features_v1.parquet",
        "v2": data_dir / "features_v2.parquet",
    }
    receipt_path = data_dir.parent / "data_receipt.json"
    for path in (cohort_path, *feature_paths.values(), receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    try:
        cohort = pd.read_parquet(cohort_path)
        features = {version: pd.read_parquet(path) for version, path in feature_paths.items()}
        data_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read the prepared data inputs: {exc}") from exc

    if not isinstance(data_receipt, dict):
        raise ValueError("data_receipt.json must contain a JSON object")
    receipt_timestamp = data_receipt.get("event_timestamp", {}).get("value")
    if receipt_timestamp != SNAPSHOT_TEXT:
        raise ValueError(
            "data_receipt.json event_timestamp.value must be "
            f"{SNAPSHOT_TEXT}; found {receipt_timestamp!r}"
        )
    source_archive_sha256 = data_receipt.get("source", {}).get("archive_sha256")
    if not isinstance(source_archive_sha256, str) or len(source_archive_sha256) != 64:
        raise ValueError("data_receipt.json is missing source.archive_sha256")
    try:
        int(source_archive_sha256, 16)
    except ValueError as exc:
        raise ValueError("data_receipt.json source.archive_sha256 must be a SHA-256 hex digest") from exc

    _require_columns(cohort, COHORT_REQUIRED, "cohort.parquet")
    for version, frame in features.items():
        required = ("athlete_id", "event_timestamp", *FEATURES_BY_VERSION[version])
        _require_columns(frame, required, f"features_{version}.parquet")

    id_mode = _validate_ids(
        {"cohort": cohort, "features_v1": features["v1"], "features_v2": features["v2"]}
    )

    split = cohort["split"]
    if split.isna().any() or not split.map(lambda value: isinstance(value, str) and value in SPLITS).all():
        raise ValueError(f"cohort split values must be one of {sorted(SPLITS)}")

    target = pd.to_numeric(cohort["total_lift"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype="float64")).all():
        raise ValueError("cohort total_lift target must contain only finite values")
    cohort["total_lift"] = target.astype("float64")

    for version, frame in features.items():
        _validate_snapshot(frame, f"features_{version}.parquet")
        _validate_numeric_values(frame, FEATURES_BY_VERSION[version], f"features_{version}.parquet")
        _validate_gender_feature(frame, f"features_{version}.parquet")

    input_receipt = {
        "path": str(receipt_path.resolve()),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "source_archive_sha256": source_archive_sha256.lower(),
    }
    return cohort, features, input_receipt, id_mode


def _build_feature_store(output_dir: Path, data_dir: Path, id_mode: str):
    import yaml
    from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource, ValueType
    from feast.data_format import ParquetFormat
    from feast.types import Bool, Float64

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to protect existing files: {output_dir}"
        )

    registry_path = output_dir / "registry.db"
    online_db_path = output_dir / "online.db"
    configuration = {
        "project": "adsp_assignment2_feature_store",
        "provider": "local",
        "registry": str(registry_path.resolve()),
        "online_store": {"type": "sqlite", "path": str(online_db_path.resolve())},
        "offline_store": {"type": "dask"},
        "batch_engine": {"type": "local"},
    }
    config_path = output_dir / "feature_store.yaml"
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    store = FeatureStore(repo_path=str(output_dir.resolve()))

    entity = Entity(
        name="athlete",
        join_keys=["athlete_id"],
        value_type=ValueType.STRING if id_mode == "string" else ValueType.INT64,
        description="Athlete identifier preserved from the prepared cohort.",
    )
    views = {}
    services = {}
    for version, feature_names in FEATURES_BY_VERSION.items():
        source = FileSource(
            path=str((data_dir / f"features_{version}.parquet").resolve()),
            timestamp_field="event_timestamp",
            file_format=ParquetFormat(),
        )
        schema = [
            Field(name=name, dtype=Bool if name == "is_female" else Float64)
            for name in feature_names
        ]
        view = FeatureView(
            name=FEATURE_VIEW_NAMES[version],
            entities=[entity],
            source=source,
            schema=schema,
            ttl=timedelta(days=1),
            online=True,
            # Historical retrieval reads the configured FileSource. Keep Feast's
            # local materialization from writing back into this prepared input file.
            offline=False,
            description=(
                f"ADSP Assignment 2 {version} athlete predictors. "
                "All rows use a synthetic 2026-10-18 UTC snapshot timestamp."
            ),
        )
        service = FeatureService(
            name=FEATURE_SERVICE_NAMES[version],
            features=[view],
            description=f"Immutable ADSP Assignment 2 {version} feature set.",
        )
        views[version] = view
        services[version] = service

    store.apply([entity, *views.values(), *services.values()])
    return store, views, services


def _assert_retrieved_matches_source(
    retrieved: pd.DataFrame, source: pd.DataFrame, feature_names: tuple[str, ...], version: str
) -> None:
    if len(retrieved) != len(source):
        raise ValueError(
            f"Feast historical retrieval for {version} returned {len(retrieved)} rows; "
            f"expected {len(source)}"
        )
    if retrieved["athlete_id"].duplicated().any():
        raise ValueError(f"Feast historical retrieval for {version} returned duplicate athlete_id values")
    if set(retrieved["athlete_id"].tolist()) != set(source["athlete_id"].tolist()):
        raise ValueError(f"Feast historical retrieval for {version} athlete_id values do not match input")

    actual = retrieved.set_index("athlete_id")
    expected = source.set_index("athlete_id")
    for feature in feature_names:
        values = actual.loc[expected.index, feature]
        if feature == "is_female":
            matches = values.astype(bool).to_numpy() == expected[feature].astype(bool).to_numpy()
        else:
            matches = np.isclose(
                values.to_numpy(dtype="float64"),
                expected[feature].to_numpy(dtype="float64"),
                rtol=0.0,
                atol=0.0,
            )
        if not bool(matches.all()):
            raise ValueError(
                f"Feast historical retrieval for {version} does not match "
                f"the offline source values for {feature}"
            )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Cannot write non-finite values to the feature store receipt")
        return number
    return value


def _input_sha256s(data_dir: Path) -> dict[str, str]:
    paths = {
        "cohort.parquet": data_dir / "cohort.parquet",
        "features_v1.parquet": data_dir / "features_v1.parquet",
        "features_v2.parquet": data_dir / "features_v2.parquet",
        "data_receipt.json": data_dir.parent / "data_receipt.json",
    }
    hashes = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def _verify_online_readback(store, service, frame: pd.DataFrame, source: pd.DataFrame, version: str) -> dict[str, Any]:
    feature_names = FEATURES_BY_VERSION[version]
    sample_ids = frame["athlete_id"].head(MAX_ONLINE_SAMPLE_ROWS).tolist()
    source_by_id = source.set_index("athlete_id")
    samples = []

    for athlete_id in sample_ids:
        response = store.get_online_features(
            features=service,
            entity_rows=[{"athlete_id": athlete_id}],
        ).to_dict()
        expected_response_names = {*feature_names, "athlete_id"}
        if set(response) != expected_response_names:
            raise ValueError(
                f"SQLite online readback for {version} returned feature names "
                f"{sorted(response)}; expected {sorted(expected_response_names)}"
            )
        returned_ids = response["athlete_id"]
        if len(returned_ids) != 1 or returned_ids[0] != athlete_id:
            raise ValueError(
                f"SQLite online readback for {version} returned a different athlete_id "
                f"than requested: {returned_ids!r}"
            )
        values = {}
        for feature in feature_names:
            received = response[feature]
            if len(received) != 1 or received[0] is None:
                raise ValueError(
                    f"SQLite online readback for {version} is missing {feature} "
                    f"for athlete_id={athlete_id!r}"
                )
            actual = received[0]
            expected = source_by_id.at[athlete_id, feature]
            if feature == "is_female":
                matches = bool(actual) == bool(expected)
            else:
                matches = math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=0.0
                )
            if not matches:
                raise ValueError(
                    f"SQLite online readback for {version} does not match the offline "
                    f"source for athlete_id={athlete_id!r}, feature={feature}"
                )
            values[feature] = _json_value(actual)
        samples.append({"athlete_id": _json_value(athlete_id), "features": values})

    return {
        "sample_rows": len(samples),
        "matches_offline_source": True,
        "sample_values": samples,
    }


def run_pipeline(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    input_hashes_before = _input_sha256s(data_dir)
    cohort, features, input_receipt, id_mode = _read_inputs(data_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to protect existing files: {output_dir}"
        )

    store, views, services = _build_feature_store(output_dir, data_dir, id_mode)
    entity_df = cohort[["athlete_id"]].merge(
        features["v1"][["athlete_id", "event_timestamp"]],
        on="athlete_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if entity_df["event_timestamp"].isna().any():
        raise ValueError("Could not assign a synthetic event_timestamp to every cohort athlete_id")

    result_frames: dict[str, pd.DataFrame] = {}
    for version, feature_names in FEATURES_BY_VERSION.items():
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=services[version],
        ).to_df()
        _require_columns(historical, ("athlete_id", *feature_names), f"Feast {version} historical output")
        _assert_retrieved_matches_source(historical, features[version], feature_names, version)
        feature_rows = historical[["athlete_id", *feature_names]]
        training = cohort[["athlete_id", "total_lift", "split"]].merge(
            feature_rows,
            on="athlete_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if training["athlete_id"].tolist() != cohort["athlete_id"].tolist():
            raise ValueError(f"Feast {version} join changed cohort athlete_id ordering")
        if training[list(feature_names)].isna().any().any():
            raise ValueError(f"Feast {version} training output contains missing predictor values")
        for feature in feature_names:
            if feature == "is_female":
                continue
            if not np.isfinite(training[feature].to_numpy(dtype="float64")).all():
                raise ValueError(f"Feast {version} training predictor {feature} is not finite")
        result_frames[version] = training

    snapshot = SNAPSHOT.to_pydatetime()
    store.materialize(
        start_date=snapshot - timedelta(seconds=1),
        end_date=snapshot + timedelta(seconds=1),
        feature_views=[view.name for view in views.values()],
    )

    online_readback = {
        version: _verify_online_readback(
            store, services[version], result_frames[version], features[version], version
        )
        for version in FEATURES_BY_VERSION
    }

    for version, frame in result_frames.items():
        frame.to_parquet(output_dir / f"retrieved_{version}.parquet", index=False)

    input_hashes_after = _input_sha256s(data_dir)
    inputs_unchanged = input_hashes_before == input_hashes_after
    if not inputs_unchanged:
        raise RuntimeError("One or more prepared input files changed during the feature-store run")

    import feast

    receipt = {
        "feast_version": feast.__version__,
        "provider": "local",
        "offline_store": "dask",
        "online_store": "sqlite",
        "registry_path": str((output_dir / "registry.db").resolve()),
        "online_store_path": str((output_dir / "online.db").resolve()),
        "input_data_receipt": input_receipt,
        "input_read_only_verification": {
            "before_sha256": input_hashes_before,
            "after_sha256": input_hashes_after,
            "unchanged": inputs_unchanged,
        },
        "athlete_id_representation": id_mode,
        "snapshot": {
            "event_timestamp": SNAPSHOT_TEXT,
            "semantics": "synthetic UTC snapshot, not historical event or measurement time",
            "historical_leakage_proof": False,
        },
        "feature_views": {
            version: {
                "name": FEATURE_VIEW_NAMES[version],
                "rows": len(features[version]),
                "feature_names": list(FEATURES_BY_VERSION[version]),
            }
            for version in FEATURES_BY_VERSION
        },
        "feature_services": dict(FEATURE_SERVICE_NAMES),
        "retrieved_outputs": {
            version: {
                "path": f"retrieved_{version}.parquet",
                "rows": len(result_frames[version]),
                "columns": list(result_frames[version].columns),
            }
            for version in FEATURES_BY_VERSION
        },
        "online_readback": online_readback,
    }
    (output_dir / "feature_store_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing prepared Parquet inputs")
    parser.add_argument("--output", required=True, type=Path, help="New, empty output directory for local Feast state and retrievals")
    args = parser.parse_args(argv)

    try:
        receipt = run_pipeline(args.data, args.output)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for version, output in receipt["retrieved_outputs"].items():
        print(f"{version}: {output['rows']} rows -> {args.output / output['path']}")
    print(f"receipt: {args.output / 'feature_store_receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
