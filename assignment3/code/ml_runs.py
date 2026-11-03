#!/usr/bin/env python3
"""Run reproducible Part A and FLAML Part B experiments for ADSP Assignment 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import mlflow
import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker, OutputMethod
from codecarbon.core.cpu import DEFAULT_POWER_PER_CORE, TDP
from flaml import AutoML
from flaml.automl.training_log import training_log_reader
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REQUIRED_COLUMNS = ["athlete_id", "total_lift", "split"]
FEATURES_BY_VERSION = {
    "v1": ["age", "height", "weight", "is_female"],
    "v2": [
        "age",
        "height",
        "weight",
        "is_female",
        "age_squared",
        "height_squared",
        "weight_squared",
        "age_x_weight",
        "age_x_height",
        "weight_x_height",
    ],
}
LEARNERS = ["rf", "extra_tree", "lgbm"]
RF_CONFIGURATIONS = [
    {"config_name": "rf_80trees_depth8_leaf5", "n_estimators": 80, "max_depth": 8, "min_samples_leaf": 5},
    {"config_name": "rf_120trees_depth16_leaf10", "n_estimators": 120, "max_depth": 16, "min_samples_leaf": 10},
]
EMISSIONS_NOTE = (
    "Energy and CO2e are CodeCarbon software estimates. CPU power uses the actual "
    "CodeCarbon CPU-load backend and its detected/default TDP model; RAM power uses "
    "CodeCarbon's estimation model. CO2e uses the offline static US average grid mix. "
    "These are not direct watt-meter or Mac-specific measurements. Training estimates "
    "include the CodeCarbon tracking interval and exclude later prediction benchmarking."
)
BENCHMARK_NOTE = (
    "Inference time is the median of five end-to-end local Python model.predict calls "
    "on the same sorted validation batch after one warmup. It includes the model wrapper "
    "and dataframe input overhead and is not a pure algorithm-speed measurement. An H2O "
    "comparison must include and label H2O client overhead."
)


class IncompleteExperiment(RuntimeError):
    """A requested experiment stage could not produce its minimum valid evidence."""

    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def _write_json(path: Path, value: Any) -> None:
    contents = _json_text(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _feature_file_manifest(path: Path, version: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "feature_columns": FEATURES_BY_VERSION[version],
    }


def _prepare_input_frame(frame: pd.DataFrame, version: str, path: Path) -> pd.DataFrame:
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{version} contains duplicate column names: {duplicated}")

    features = FEATURES_BY_VERSION[version]
    expected = REQUIRED_COLUMNS + features
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing:
        raise ValueError(f"{version} is missing required Feast output column(s): {missing}")
    if unexpected:
        raise ValueError(
            f"{version} has unexpected column(s) {unexpected}; only athlete_id, total_lift, "
            f"split, and the {len(features)} allowlisted Feast predictors are accepted"
        )

    prepared = frame[expected].copy()
    if prepared["athlete_id"].isna().any():
        raise ValueError(f"{version} contains a null athlete_id")
    prepared["athlete_id"] = prepared["athlete_id"].astype(str)
    if prepared["athlete_id"].duplicated().any():
        duplicates = prepared.loc[prepared["athlete_id"].duplicated(), "athlete_id"].head(5).tolist()
        raise ValueError(f"{version} contains duplicate athlete_id values: {duplicates}")

    prepared["split"] = prepared["split"].astype(str).str.lower()
    allowed_splits = set(SPLIT_RATIOS)
    observed_splits = set(prepared["split"])
    if observed_splits != allowed_splits:
        raise ValueError(
            f"{version} split labels must be exactly {sorted(allowed_splits)}; "
            f"found {sorted(observed_splits)}"
        )

    numeric_columns = ["total_lift", *features]
    for column in numeric_columns:
        try:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{version}.{column} must be numeric") from exc
        values = prepared[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{version}.{column} contains null or non-finite values")

    if len(prepared) < 20:
        raise ValueError(f"{version} has {len(prepared)} rows; at least 20 are required")
    for split_name, expected_ratio in SPLIT_RATIOS.items():
        count = int((prepared["split"] == split_name).sum())
        if count < (3 if split_name == "train" else 2):
            raise ValueError(f"{version} has too few rows in the {split_name} split")
        observed_ratio = count / len(prepared)
        tolerance = max(0.015, 2 / len(prepared))
        if abs(observed_ratio - expected_ratio) > tolerance:
            raise ValueError(
                f"{version} {split_name} split proportion {observed_ratio:.4f} is not "
                f"consistent with the required {expected_ratio:.2f}"
            )
    return prepared


def load_feature_inputs(features_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    files = {
        "v1": features_dir / "retrieved_v1.parquet",
        "v2": features_dir / "retrieved_v2.parquet",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Feast retrieved Parquet input(s): {missing}")

    frames: dict[str, pd.DataFrame] = {}
    for version, path in files.items():
        frames[version] = _prepare_input_frame(pd.read_parquet(path), version, path)

    first = frames["v1"].sort_values("athlete_id").set_index("athlete_id")
    second = frames["v2"].sort_values("athlete_id").set_index("athlete_id")
    if not first.index.equals(second.index):
        missing_from_v2 = sorted(set(first.index) - set(second.index))[:5]
        missing_from_v1 = sorted(set(second.index) - set(first.index))[:5]
        raise ValueError(
            "v1 and v2 athlete_id populations do not match; "
            f"missing from v2={missing_from_v2}, missing from v1={missing_from_v1}"
        )

    parity_columns = ["total_lift", "split", *FEATURES_BY_VERSION["v1"]]
    for column in parity_columns:
        left = first[column].to_numpy()
        right = second[column].to_numpy()
        if column in {"split"}:
            equal = np.array_equal(left, right)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise ValueError(f"v1 and v2 {column} values do not match for the common split")

    manifest = {
        version: _feature_file_manifest(files[version], version, frames[version])
        for version in ("v1", "v2")
    }
    return frames, manifest


def _initialize_mlflow(output_dir: Path) -> tuple[str, str, str]:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    tracking_uri = f"sqlite:///{(output_dir / 'mlruns.sqlite').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    artifact_root = output_dir / "mlflow-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    experiments: dict[str, str] = {}
    for experiment_name in ("adsp2_part_a", "adsp2_part_b"):
        artifact_uri = (artifact_root / experiment_name).resolve().as_uri()
        existing = mlflow.get_experiment_by_name(experiment_name)
        if existing is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_uri,
                tags={"assignment": "ADSP Assignment 2", "tracking": "local SQLite"},
            )
        else:
            experiment_id = existing.experiment_id
        experiments[experiment_name] = experiment_id
    return tracking_uri, experiments["adsp2_part_a"], experiments["adsp2_part_b"]


def _new_emissions_tracker(stage_name: str, emissions_dir: Path):
    emissions_dir.mkdir(parents=True, exist_ok=True)
    tracker = OfflineEmissionsTracker(
        project_name="adsp-assignment2",
        experiment_name=stage_name,
        country_iso_code="USA",
        country_2letter_iso_code=None,
        region=None,
        cloud_provider=None,
        cloud_region=None,
        force_mode_cpu_load=True,
        tracking_mode="process",
        output_dir=str(emissions_dir.resolve()),
        output_file="codecarbon.csv",
        output_methods=[OutputMethod.CSV],
        output_handlers=[],
        save_to_file=True,
        save_to_api=False,
        save_to_logger=False,
        save_to_prometheus=False,
        save_to_logfire=False,
        emissions_endpoint=None,
        electricitymaps_api_token=None,
    )

    # CodeCarbon suppresses some initialization exceptions. Verify that the
    # requested local-only, process CPU-load estimate actually initialized.
    if getattr(tracker, "_electricitymaps_api_token", None) is not None:
        raise RuntimeError("CodeCarbon has an Electricity Maps token configured; static offline mode required")
    if getattr(tracker, "_force_cpu_power", None) is not None:
        raise RuntimeError("CodeCarbon CPU power override is configured; use its detected TDP estimate")
    if getattr(tracker, "_force_ram_power", None) is not None:
        raise RuntimeError("CodeCarbon RAM power override is configured; use its RAM estimation model")
    if getattr(tracker, "_force_carbon_intensity_g_co2e_kwh", None) is not None:
        raise RuntimeError("CodeCarbon carbon intensity override is configured; use static USA average")
    if getattr(tracker, "_conf", {}).get("force_mode_cpu_load") is not True:
        raise RuntimeError("CodeCarbon did not enable force_mode_cpu_load")
    if getattr(tracker, "_output_methods", None) != [OutputMethod.CSV]:
        raise RuntimeError("CodeCarbon output methods are not restricted to local CSV")
    if any(
        getattr(tracker, key, None) is not expected
        for key, expected in (
            ("_save_to_file", True),
            ("_save_to_api", False),
            ("_save_to_logger", False),
            ("_save_to_prometheus", False),
            ("_save_to_logfire", False),
        )
    ):
        raise RuntimeError("CodeCarbon output flags are not restricted to the local CSV sink")
    handlers = getattr(tracker, "_output_handlers", [])
    if [type(handler).__name__ for handler in handlers] != ["FileOutput"]:
        raise RuntimeError("CodeCarbon has an unexpected output handler; local CSV is the only allowed sink")
    if getattr(tracker, "_emissions_endpoint", None) is not None:
        raise RuntimeError("CodeCarbon emissions endpoint is configured; local offline output is required")
    if any(
        getattr(tracker, attr, None) is not None
        for attr in ("_region", "_cloud_provider", "_cloud_region", "_country_2letter_iso_code")
    ):
        raise RuntimeError("CodeCarbon region or cloud factor is configured; static USA country average required")

    tracker.get_detected_hardware()
    cpu_hardware = next((item for item in tracker._hardware if type(item).__name__ == "CPU"), None)
    if cpu_hardware is None:
        raise RuntimeError("CodeCarbon did not initialize a CPU estimator")
    if getattr(cpu_hardware, "_mode", None) != "cpu_load":
        raise RuntimeError(
            f"CodeCarbon selected CPU backend {getattr(cpu_hardware, '_mode', None)!r}; "
            "the assignment requires the psutil CPU-load estimator"
        )
    if getattr(cpu_hardware, "_tracking_mode", None) != "process":
        raise RuntimeError("CodeCarbon CPU estimator is not tracking this process")
    tdp_watts = float(getattr(cpu_hardware, "_tdp", 0.0) or 0.0)
    if not math.isfinite(tdp_watts) or tdp_watts <= 0:
        raise RuntimeError("CodeCarbon CPU-load/TDP power estimate is unavailable")

    detected_tdp = TDP()
    registry_tdp_watts = detected_tdp._get_cpu_power_from_registry(str(cpu_hardware._model))
    registry_matched = registry_tdp_watts is not None
    logical_cpu_count = int(tracker._conf.get("cpu_count") or 0)

    physical_cpu_count = int(tracker._conf.get("cpu_physical_count") or 0)
    generic_tdp = not registry_matched
    if registry_matched:
        tdp_source = "CodeCarbon detected CPU TDP registry"
    elif getattr(cpu_hardware, "_is_generic_tdp", False):
        tdp_source = "CodeCarbon generic fallback power constant"
    else:
        tdp_source = (
            f"CodeCarbon default fallback of {DEFAULT_POWER_PER_CORE} W per logical CPU thread; "
            "detected model is absent from the TDP registry"
        )
    metadata = {
        "cpu_backend": str(cpu_hardware._mode),
        "cpu_model": str(cpu_hardware._model),
        "cpu_tracking_mode": str(cpu_hardware._tracking_mode),
        "cpu_tdp_watts": tdp_watts,
        "cpu_tdp_per_physical_cpu_watts": (
            tdp_watts / physical_cpu_count if physical_cpu_count > 0 else None
        ),
        "cpu_tdp_per_logical_thread_watts": (
            tdp_watts / logical_cpu_count if logical_cpu_count > 0 else None
        ),
        "cpu_physical_count": physical_cpu_count,
        "cpu_logical_count": logical_cpu_count,
        "cpu_tdp_generic": generic_tdp,
        "cpu_tdp_registry_matched": registry_matched,
        "cpu_tdp_registry_watts": float(registry_tdp_watts) if registry_matched else None,
        "cpu_tdp_default_per_logical_thread_watts": (
            float(DEFAULT_POWER_PER_CORE) if not registry_matched else None
        ),
        "cpu_tdp_source": tdp_source,
        "tracking_mode": "process",
        "country_iso_code": "USA",
        "estimate_label": "software estimate from CPU-load/TDP and RAM models",
        "co2e_label": "offline static US average grid mix; not a measured local grid factor",
    }
    tracker.start()
    if getattr(tracker, "_start_time", None) is None:
        raise RuntimeError("CodeCarbon tracker failed to start; no emissions estimate is available")
    return tracker, metadata


def _run_tracked(
    stage_name: str,
    emissions_dir: Path,
    operation: Callable[[], Any],
) -> tuple[Any, float, dict[str, Any]]:
    tracker, metadata = _new_emissions_tracker(stage_name, emissions_dir)
    start = time.perf_counter()
    try:
        value = operation()
    except BaseException:
        tracker.stop()
        raise
    training_seconds = time.perf_counter() - start
    tracker.stop()
    final_data = getattr(tracker, "final_emissions_data", None)
    if final_data is None:
        raise RuntimeError(f"CodeCarbon produced no final energy/CO2e record for {stage_name}")
    energy_kwh = float(final_data.energy_consumed)
    emissions_kg = float(final_data.emissions)
    duration_seconds = float(final_data.duration)
    if not all(math.isfinite(number) for number in (energy_kwh, emissions_kg, duration_seconds)):
        raise RuntimeError(f"CodeCarbon returned a non-finite estimate for {stage_name}")
    if energy_kwh <= 0 or emissions_kg <= 0 or duration_seconds <= 0:
        raise RuntimeError(
            f"CodeCarbon energy/CO2e estimate is unavailable for {stage_name}: "
            f"energy_kwh={energy_kwh}, emissions_kg={emissions_kg}, duration={duration_seconds}"
        )
    if str(final_data.country_iso_code).upper() != "USA":
        raise RuntimeError(
            f"CodeCarbon used country {final_data.country_iso_code!r}; static USA average required"
        )
    csv_path = emissions_dir / "codecarbon.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"CodeCarbon did not write its local emissions CSV: {csv_path}")
    metadata.update(
        {
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
            "codecarbon_duration_seconds": duration_seconds,
            "codecarbon_cpu_energy_kwh": float(final_data.cpu_energy),
            "codecarbon_ram_energy_kwh": float(final_data.ram_energy),
            "codecarbon_gpu_energy_kwh": float(final_data.gpu_energy),
            "country_iso_code": str(final_data.country_iso_code),
            "country_name": str(final_data.country_name),
            "codecarbon_version": str(final_data.codecarbon_version),
            "codecarbon_csv_path": str(csv_path),
        }
    )
    return value, training_seconds, metadata


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2 or len(predicted) != len(actual):
        raise ValueError("Regression metrics require at least two paired observations")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metrics cannot use non-finite actual or predicted values")
    values = {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Regression metrics produced a non-finite result")
    return values


def _split_rows(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return frame.loc[frame["split"] == split_name].copy()


def _prediction_table(frame: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "athlete_id": frame["athlete_id"].astype(str).to_numpy(),
            "actual_total_lift": frame["total_lift"].to_numpy(dtype=float),
            "prediction": np.asarray(predicted, dtype=float),
        }
    )
    result["residual"] = result["prediction"] - result["actual_total_lift"]
    return result.sort_values("athlete_id", key=lambda column: column.astype(str)).reset_index(drop=True)


def _path_from_output(output_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def _mlflow_log_artifact(path: Path, artifact_path: str, run_id: str | None = None) -> None:
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path, run_id=run_id)
    elif path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path, run_id=run_id)
    else:
        raise FileNotFoundError(f"Expected MLflow artifact does not exist: {path}")


def _mlflow_log_provenance(provenance: dict[str, Any], artifact_file: str = "input_provenance.json") -> None:
    mlflow.log_dict(provenance, artifact_file, run_id=mlflow.active_run().info.run_id)


def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id_sha256": record["benchmark_batch_id_sha256"],
        "row_count": record["benchmark_row_count"],
        "warmups": record["benchmark_warmups"],
        "repeats": record["benchmark_repeats"],
        "median_batch_ms": record["median_batch_ms"],
        "median_per_row_ms": record["median_per_row_ms"],
        "scope": record["benchmark_scope"],
    }


def _part_a(
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    output_dir: Path,
    experiment_id: str,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    part_dir = output_dir / "part_a"
    for subdirectory in ("models", "predictions", "emissions", "plots"):
        (part_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    candidate_records: list[dict[str, Any]] = []
    mlflow_run_ids: list[str] = []
    benchmark_ids = sorted(
        _split_rows(frames["v1"], "validation")["athlete_id"].astype(str).tolist()
    )[:1000]
    benchmark_ids_path = part_dir / "inference_batch_ids.csv"
    pd.DataFrame({"athlete_id": benchmark_ids}).to_csv(benchmark_ids_path, index=False)
    for version in ("v1", "v2"):
        frame = frames[version]
        feature_columns = FEATURES_BY_VERSION[version]
        train_rows = _split_rows(frame, "train")
        validation_rows = _split_rows(frame, "validation")
        X_train = train_rows[feature_columns]
        y_train = train_rows["total_lift"]
        X_validation = validation_rows[feature_columns]
        y_validation = validation_rows["total_lift"].to_numpy(dtype=float)

        for config in RF_CONFIGURATIONS:
            run_id = f"{version}_{config['config_name']}"
            estimator = RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                n_jobs=2,
                random_state=SEED,
            )

            def fit_model(estimator=estimator, X_train=X_train, y_train=y_train):
                estimator.fit(X_train, y_train)
                return estimator

            model, training_seconds, emissions = _run_tracked(
                run_id,
                part_dir / "emissions" / run_id,
                fit_model,
            )
            predicted_validation = np.asarray(model.predict(X_validation), dtype=float)
            metrics = _regression_metrics(y_validation, predicted_validation)
            benchmark_rows = (
                _sort_validation_rows(validation_rows)
                .set_index("athlete_id")
                .loc[benchmark_ids]
            )
            benchmark_prediction_path = part_dir / "predictions" / f"benchmark_{run_id}.csv"
            benchmark = _benchmark_model(
                model,
                benchmark_rows[feature_columns],
                benchmark_ids,
                benchmark_prediction_path,
            )
            model_path = part_dir / "models" / f"{run_id}.joblib"
            joblib.dump(model, model_path)
            validation_prediction_path = part_dir / "predictions" / f"validation_{run_id}.csv"
            validation_prediction = _prediction_table(validation_rows, predicted_validation)
            validation_prediction.to_csv(validation_prediction_path, index=False)

            record = {
                "run_id": run_id,
                "version": version,
                "config_name": config["config_name"],
                "n_estimators": int(config["n_estimators"]),
                "max_depth": int(config["max_depth"]),
                "min_samples_leaf": int(config["min_samples_leaf"]),
                "n_jobs": 2,
                "random_state": SEED,
                "n_train": int(len(train_rows)),
                "n_validation": int(len(validation_rows)),
                "validation_rmse": metrics["rmse"],
                "validation_mae": metrics["mae"],
                "validation_r2": metrics["r2"],
                "training_time_seconds": float(training_seconds),
                **benchmark,
                "energy_kwh": emissions["energy_kwh"],
                "emissions_kg": emissions["emissions_kg"],
                "cpu_backend": emissions["cpu_backend"],
                "cpu_model": emissions["cpu_model"],
                "cpu_tracking_mode": emissions["cpu_tracking_mode"],
                "cpu_tdp_watts": emissions["cpu_tdp_watts"],
                "cpu_tdp_per_physical_cpu_watts": emissions["cpu_tdp_per_physical_cpu_watts"],
                "cpu_tdp_per_logical_thread_watts": emissions["cpu_tdp_per_logical_thread_watts"],
                "cpu_tdp_generic": emissions["cpu_tdp_generic"],
                "cpu_tdp_registry_matched": emissions["cpu_tdp_registry_matched"],
                "cpu_tdp_registry_watts": emissions["cpu_tdp_registry_watts"],
                "cpu_tdp_default_per_logical_thread_watts": emissions["cpu_tdp_default_per_logical_thread_watts"],
                "cpu_tdp_source": emissions["cpu_tdp_source"],
                "tracking_mode": emissions["tracking_mode"],
                "country_iso_code": emissions["country_iso_code"],
                "estimate_label": emissions["estimate_label"],
                "co2e_label": emissions["co2e_label"],
                "codecarbon_duration_seconds": emissions["codecarbon_duration_seconds"],
                "codecarbon_cpu_energy_kwh": emissions["codecarbon_cpu_energy_kwh"],
                "codecarbon_ram_energy_kwh": emissions["codecarbon_ram_energy_kwh"],
                "model_path": _path_from_output(output_dir, model_path),
                "validation_predictions_path": _path_from_output(output_dir, validation_prediction_path),
                "benchmark_predictions_path": _path_from_output(output_dir, benchmark_prediction_path),
                "benchmark_batch_ids_path": _path_from_output(output_dir, benchmark_ids_path),
                "emissions_path": _path_from_output(
                    output_dir, part_dir / "emissions" / run_id / "codecarbon.csv"
                ),
                "_model": model,
                "_validation_rows": validation_rows,
                "_validation_predictions": predicted_validation,
                "_test_frame": frames[version],
            }
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_id,
                tags={"part": "A", "feature_version": version, "selection_metric": "validation_rmse"},
            ) as active_run:
                record["mlflow_run_id"] = active_run.info.run_id
                mlflow.log_params(
                    {
                        "model_family": "RandomForestRegressor",
                        "feature_version": version,
                        "feature_columns": json.dumps(feature_columns),
                        "config_name": config["config_name"],
                        "n_estimators": config["n_estimators"],
                        "max_depth": config["max_depth"],
                        "min_samples_leaf": config["min_samples_leaf"],
                        "n_jobs": 2,
                        "random_state": SEED,
                        "split_seed": SEED,
                        "selection_metric": "validation_rmse",
                        "tracking_backend": "local sqlite",
                        "codecarbon_backend": emissions["cpu_backend"],
                        "codecarbon_tdp_watts": emissions["cpu_tdp_watts"],
                        "benchmark_batch_id_sha256": _sha256_ids(benchmark_ids),
                    },
                    synchronous=True,
                )
                mlflow.log_metrics(
                    {
                        "validation_rmse": metrics["rmse"],
                        "validation_mae": metrics["mae"],
                        "validation_r2": metrics["r2"],
                        "training_time_seconds": float(training_seconds),
                        "median_batch_ms": benchmark["median_batch_ms"],
                        "median_per_row_ms": benchmark["median_per_row_ms"],
                        "energy_kwh": emissions["energy_kwh"],
                        "emissions_kg": emissions["emissions_kg"],
                    },
                    synchronous=True,
                )
                _mlflow_log_provenance({"source_files": manifest, "split_seed": SEED})
                _mlflow_log_artifact(model_path, "models")
                _mlflow_log_artifact(validation_prediction_path, "validation_predictions")
                _mlflow_log_artifact(benchmark_prediction_path, "benchmark_predictions")
                _mlflow_log_artifact(benchmark_ids_path, "benchmark")
                _mlflow_log_artifact(part_dir / "emissions" / run_id, "codecarbon")
            mlflow_run_ids.append(record["mlflow_run_id"])
            candidate_records.append(record)

    metrics_path = part_dir / "metrics.csv"
    public_records = [_public_record(record) for record in candidate_records]
    _write_csv(metrics_path, public_records)

    selected = min(candidate_records, key=lambda row: (row["validation_rmse"], row["run_id"]))
    selected_frame = selected["_test_frame"]
    test_rows = _split_rows(selected_frame, "test")
    selected_model = joblib.load(output_dir / selected["model_path"])
    predicted_test = np.asarray(
        selected_model.predict(test_rows[FEATURES_BY_VERSION[selected["version"]]]),
        dtype=float,
    )
    test_metrics = _regression_metrics(test_rows["total_lift"].to_numpy(dtype=float), predicted_test)
    test_prediction_path = part_dir / "test_predictions.csv"
    _prediction_table(test_rows, predicted_test).to_csv(test_prediction_path, index=False)
    selected_json = {
        "run_id": selected["run_id"],
        "feature_version": selected["version"],
        "selection_basis": "validation_rmse",
        "validation_metrics": {
            "rmse": selected["validation_rmse"],
            "mae": selected["validation_mae"],
            "r2": selected["validation_r2"],
        },
        "benchmark": _benchmark_summary(selected),
        "test_metrics": test_metrics,
        "test_prediction_count": int(len(test_rows)),
        "test_predictions_path": _path_from_output(output_dir, test_prediction_path),
        "model_path": selected["model_path"],
        "test_evaluated_for_selected_model_only": True,
    }
    _write_json(part_dir / "selected_best.json", selected_json)
    _plot_part_a(candidate_records, part_dir / "plots")

    return (
        {
            "status": "complete",
            "candidate_count": len(candidate_records),
            "candidate_metrics_path": _path_from_output(output_dir, metrics_path),
            "common_benchmark_batch_id_sha256": _sha256_ids(benchmark_ids),
            "benchmark": _benchmark_summary(selected),
            "selected_best": selected_json,
        },
        mlflow_run_ids,
        candidate_records,
    )


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _plot_part_a(records: list[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    all_actual = np.concatenate([record["_validation_rows"]["total_lift"].to_numpy() for record in records])
    minimum = float(np.min(all_actual))
    maximum = float(np.max(all_actual))
    for record in records:
        actual = record["_validation_rows"]["total_lift"].to_numpy(dtype=float)
        predicted = record["_validation_predictions"]
        ax.scatter(actual, predicted, alpha=0.65, s=26, label=record["run_id"])
    ax.plot([minimum, maximum], [minimum, maximum], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual total_lift (validation)")
    ax.set_ylabel("Predicted total_lift")
    ax.set_title("Part A: validation predictions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = [
        record["_validation_predictions"] - record["_validation_rows"]["total_lift"].to_numpy(dtype=float)
        for record in records
    ]
    labels = [record["run_id"] for record in records]
    ax.boxplot(residuals, tick_labels=labels, showmeans=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Prediction minus actual total_lift")
    ax.set_title("Part A: validation residuals")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(records))
    labels = [record["run_id"] for record in records]
    axes[0].bar(x, [record["validation_rmse"] for record in records], color="#4267ac")
    axes[0].set_ylabel("Validation RMSE")
    axes[0].set_title("Part A: four-run validation and emissions comparison")
    axes[1].bar(x, [record["emissions_kg"] for record in records], color="#55a868")
    axes[1].set_ylabel("Estimated CO2e (kg)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_xlabel("Random forest configuration")
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_emissions.png", dpi=160)
    plt.close(fig)


def _training_log_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with training_log_reader(str(path)) as reader:
        return [dict(vars(record)) for record in reader.records()]


def _sort_validation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("athlete_id", key=lambda column: column.astype(str)).reset_index(drop=True)


def _benchmark_model(
    model: Any,
    batch_features: pd.DataFrame,
    batch_ids: list[str],
    prediction_path: Path,
) -> dict[str, Any]:
    if not len(batch_features) or len(batch_features) != len(batch_ids):
        raise ValueError("The shared inference benchmark batch must contain validation rows")
    model.predict(batch_features)
    elapsed_ms: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        prediction = np.asarray(model.predict(batch_features), dtype=float)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    if not np.isfinite(prediction).all() or any(value <= 0 for value in elapsed_ms):
        raise RuntimeError("Model prediction or end-to-end inference timing was unavailable")
    benchmark_prediction = pd.DataFrame(
        {"athlete_id": batch_ids, "prediction": prediction}
    )
    benchmark_prediction.to_csv(prediction_path, index=False)
    median_batch_ms = float(statistics.median(elapsed_ms))
    return {
        "benchmark_batch_id_sha256": _sha256_ids(batch_ids),
        "benchmark_row_count": int(len(batch_ids)),
        "benchmark_warmups": 1,
        "benchmark_repeats": 5,
        "median_batch_ms": median_batch_ms,
        "median_per_row_ms": float(median_batch_ms / len(batch_ids)),
        "benchmark_scope": "end-to-end local Python model.predict call; wrapper/dataframe overhead included",
    }


def _run_flaml_mode(
    mode_name: str,
    feature_columns: list[str],
    frame: pd.DataFrame,
    output_dir: Path,
    time_budget: float,
    max_iter: int | None,
    batch_ids: list[str],
) -> dict[str, Any]:
    mode_dir = output_dir / "part_b" / mode_name
    for subdirectory in ("models", "predictions", "emissions"):
        (mode_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    train_rows = _split_rows(frame, "train")
    validation_rows = _split_rows(frame, "validation")
    X_train = train_rows[feature_columns]
    y_train = train_rows["total_lift"]
    X_validation = validation_rows[feature_columns]
    y_validation = validation_rows["total_lift"].to_numpy(dtype=float)
    validation_sorted = _sort_validation_rows(validation_rows)
    batch_frame = validation_sorted.set_index("athlete_id").loc[batch_ids]
    batch_features = batch_frame[feature_columns]
    batch_ids_path = mode_dir / "inference_batch_ids.csv"
    pd.DataFrame({"athlete_id": batch_ids}).to_csv(batch_ids_path, index=False)

    log_path = mode_dir / "training_log.jsonl"
    automl = AutoML()

    def fit_automl():
        fit_kwargs: dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_validation,
            "y_val": validation_rows["total_lift"],
            "task": "regression",
            "metric": "rmse",
            "eval_method": "holdout",
            "estimator_list": LEARNERS,
            "learner_selector": "roundrobin",
            "time_budget": time_budget,
            "sample": False,
            "model_history": True,
            "log_type": "all",
            "retrain_full": False,
            "ensemble": False,
            "n_jobs": 2,
            "seed": SEED,
            "verbose": 0,
            "log_file_name": str(log_path),
        }
        if max_iter is not None:
            fit_kwargs["max_iter"] = max_iter
        automl.fit(**fit_kwargs)
        return automl

    try:
        _, fit_seconds, emissions = _run_tracked(
            f"part_b_{mode_name}",
            mode_dir / "emissions",
            fit_automl,
        )
    except Exception as exc:
        reason = f"FLAML {mode_name} search failed before a complete candidate set: {type(exc).__name__}: {exc}"
        _write_json(
            mode_dir / "status.json",
            {"status": "incomplete", "required_learners": LEARNERS, "reason": reason},
        )
        raise IncompleteExperiment(mode_name, reason) from exc

    trial_records = _training_log_records(log_path)
    _write_csv(mode_dir / "training_log.csv", trial_records)
    candidates: list[dict[str, Any]] = []
    for learner in LEARNERS:
        history_model = automl.best_model_for_estimator(learner)
        records_for_learner = [record for record in trial_records if record.get("learner") == learner]
        if not records_for_learner:
            continue

        best_trial = min(
            records_for_learner,
            key=lambda item: (float(item["validation_loss"]), int(item["record_id"])),
        )
        training_time = float(best_trial["trial_time"])
        if not math.isfinite(training_time) or training_time <= 0:
            raise IncompleteExperiment(
                mode_name,
                f"FLAML {learner} has a retained model but no positive best-trial training time",
            )
        history_error = None
        history_prediction = None
        if history_model is None:
            history_error = "best_model_for_estimator returned no model"
        else:
            try:
                history_features = automl.preprocess(X_validation)
                history_prediction = np.asarray(history_model.predict(history_features), dtype=float)
                history_metrics = _regression_metrics(y_validation, history_prediction)
                if not math.isclose(
                    history_metrics["rmse"],
                    float(best_trial["validation_loss"]),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    history_error = (
                        f"best_model_for_estimator RMSE {history_metrics['rmse']:.8f} does not match "
                        f"best logged trial {best_trial['record_id']} RMSE "
                        f"{float(best_trial['validation_loss']):.8f}"
                    )
            except Exception as exc:
                history_error = f"best_model_for_estimator could not predict: {type(exc).__name__}: {exc}"

        # FLAML 2.7 can clean a SearchState model when a later global best
        # replaces it, even with model_history=True. Rebuild this exact trial
        # through the installed public log API so the saved AutoML object keeps
        # the correct feature transformer and can predict on raw feature rows.
        # Only the original training partition is supplied; record_id fixes the
        # logged best config, and validation/test data are never used to fit it.
        refitter = AutoML(fit_kwargs_by_estimator={learner: {}})
        refit_started = time.perf_counter()
        try:
            refitter.retrain_from_log(
                log_file_name=str(log_path),
                X_train=X_train,
                y_train=y_train,
                task="regression",
                eval_method="holdout",
                record_id=int(best_trial["record_id"]),
                train_full=True,
                n_jobs=2,
            )
        except Exception as exc:
            raise IncompleteExperiment(
                mode_name,
                f"FLAML {learner} public retrain_from_log failed for best trial "
                f"{best_trial['record_id']}: {type(exc).__name__}: {exc}",
            ) from exc
        model_refit_seconds = time.perf_counter() - refit_started
        if refitter.model is None:
            raise IncompleteExperiment(
                mode_name,
                f"FLAML {learner} public retrain_from_log returned no model for best trial "
                f"{best_trial['record_id']}",
            )
        replay_prediction = np.asarray(refitter.predict(X_validation), dtype=float)
        replay_metrics = _regression_metrics(y_validation, replay_prediction)
        if not math.isclose(
            replay_metrics["rmse"],
            float(best_trial["validation_loss"]),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise IncompleteExperiment(
                mode_name,
                f"FLAML {learner} retrain_from_log validation RMSE "
                f"{replay_metrics['rmse']:.8f} does not match best logged trial "
                f"{best_trial['record_id']} RMSE {float(best_trial['validation_loss']):.8f}",
            )
        if history_prediction is not None and history_error is None and not np.allclose(
            history_prediction, replay_prediction, rtol=1e-8, atol=1e-8
        ):
            raise IncompleteExperiment(
                mode_name,
                f"FLAML {learner} best_model_for_estimator and retrain_from_log predictions "
                "do not agree for the same best trial",
            )

        model = refitter
        predicted_validation = history_prediction if history_error is None else replay_prediction
        metrics = _regression_metrics(y_validation, predicted_validation)
        prediction_source = "best_model_for_estimator" if history_error is None else "retrain_from_log"
        model_materialization_source = "retrain_from_log"

        model_path = mode_dir / "models" / f"{learner}.joblib"
        joblib.dump(model, model_path)
        reloaded_model = joblib.load(model_path)
        reloaded_prediction = np.asarray(reloaded_model.predict(X_validation), dtype=float)
        if not np.allclose(predicted_validation, reloaded_prediction, rtol=1e-8, atol=1e-8):
            raise RuntimeError(f"Serialized FLAML {learner} model did not reproduce its validation predictions")

        validation_prediction_path = mode_dir / "predictions" / f"validation_{learner}.csv"
        validation_prediction = _prediction_table(validation_rows, predicted_validation)
        validation_prediction.to_csv(validation_prediction_path, index=False)
        benchmark_prediction_path = mode_dir / "predictions" / f"benchmark_{learner}.csv"
        benchmark = _benchmark_model(
            model,
            batch_features,
            batch_ids,
            benchmark_prediction_path,
        )

        candidate = {
            "feature_mode": mode_name,
            "estimator": learner,
            "feature_columns": json.dumps(feature_columns),
            "n_features": len(feature_columns),
            "n_train": int(len(train_rows)),
            "n_validation": int(len(validation_rows)),
            "validation_rmse": metrics["rmse"],
            "validation_mae": metrics["mae"],
            "validation_r2": metrics["r2"],
            "best_trial_record_id": int(best_trial["record_id"]),
            "training_time_seconds": training_time,
            "model_refit_seconds": float(model_refit_seconds),
            "model_materialization_source": model_materialization_source,
            "validation_prediction_source": prediction_source,
            **benchmark,
            "flaml_fit_wall_time_seconds": float(fit_seconds),
            "model_path": _path_from_output(output_dir, model_path),
            "validation_predictions_path": _path_from_output(output_dir, validation_prediction_path),
            "benchmark_predictions_path": _path_from_output(output_dir, benchmark_prediction_path),
            "_model": model,
            "_feature_columns": list(feature_columns),
            "_validation_rows": validation_rows,
            "_validation_predictions": predicted_validation,
            "_best_trial": best_trial,
        }
        candidates.append(candidate)

    candidate_rows = [_public_record(candidate) for candidate in candidates]
    _write_csv(mode_dir / "candidates.csv", candidate_rows)
    _write_csv(
        mode_dir / "benchmark.csv",
        [
            {
                "feature_mode": candidate["feature_mode"],
                "estimator": candidate["estimator"],
                "benchmark_batch_id_sha256": candidate["benchmark_batch_id_sha256"],
                "benchmark_row_count": candidate["benchmark_row_count"],
                "benchmark_warmups": candidate["benchmark_warmups"],
                "benchmark_repeats": candidate["benchmark_repeats"],
                "median_batch_ms": candidate["median_batch_ms"],
                "median_per_row_ms": candidate["median_per_row_ms"],
                "benchmark_scope": candidate["benchmark_scope"],
                "training_time_seconds": candidate["training_time_seconds"],
            }
            for candidate in candidates
        ],
    )

    if len(candidates) < len(LEARNERS):
        trained = sorted(candidate["estimator"] for candidate in candidates)
        missing = sorted(set(LEARNERS) - set(trained))
        reason = (
            f"FLAML {mode_name} produced {len(candidates)} retained trained learner models; "
            f"three are required. Missing learner(s): {missing}"
        )
        _write_json(
            mode_dir / "status.json",
            {
                "status": "incomplete",
                "required_learners": LEARNERS,
                "trained_learners": trained,
                "reason": reason,
                "training_log_path": _path_from_output(output_dir, log_path),
            },
        )
        raise IncompleteExperiment(mode_name, reason)

    by_rmse = sorted(candidates, key=lambda row: (row["validation_rmse"], row["estimator"]))
    by_speed = sorted(candidates, key=lambda row: (row["median_batch_ms"], row["estimator"]))
    mode_best = by_rmse[0]
    _write_csv(mode_dir / "ranked_by_validation_rmse.csv", [_public_record(row) for row in by_rmse])
    _write_csv(mode_dir / "ranked_by_inference_speed.csv", [_public_record(row) for row in by_speed])
    _write_csv(mode_dir / "top3_by_validation_rmse.csv", [_public_record(row) for row in by_rmse[:3]])
    _write_csv(mode_dir / "top3_by_inference_speed.csv", [_public_record(row) for row in by_speed[:3]])
    mode_summary = {
        "status": "complete",
        "feature_mode": mode_name,
        "feature_columns": feature_columns,
        "candidate_count": len(candidates),
        "trained_learners": [candidate["estimator"] for candidate in candidates],
        "time_budget_seconds": time_budget,
        "max_iter": max_iter,
        "learner_selector": "roundrobin",
        "selection_metric": "validation_rmse",
        "flaml_fit_wall_time_seconds": fit_seconds,
        "energy_kwh": emissions["energy_kwh"],
        "emissions_kg": emissions["emissions_kg"],
        "emissions_excludes": [
            "best-trial model refits",
            "validation scoring and permutation importance",
            "inference benchmarks",
        ],
        "emissions_measurement_scope": (
            "entire FLAML AutoML fit/search for this feature mode; not attributable to individual learners"
        ),
        "codecarbon_estimate": emissions,
        "emissions_csv_path": _path_from_output(
            output_dir, mode_dir / "emissions" / "codecarbon.csv"
        ),
        "model_refit_seconds_total": float(
            sum(candidate["model_refit_seconds"] for candidate in candidates)
        ),
        "model_materialization_sources": {
            candidate["estimator"]: candidate["model_materialization_source"]
            for candidate in candidates
        },
        "benchmark_batch_ids_path": _path_from_output(output_dir, batch_ids_path),
        "best_by_validation_rmse": {
            "estimator": mode_best["estimator"],
            "validation_metrics": {
                "rmse": mode_best["validation_rmse"],
                "mae": mode_best["validation_mae"],
                "r2": mode_best["validation_r2"],
            },
            "training_time_seconds": mode_best["training_time_seconds"],
            "model_refit_seconds": mode_best["model_refit_seconds"],
            "model_materialization_source": mode_best["model_materialization_source"],
            "benchmark": _benchmark_summary(mode_best),
            "model_path": mode_best["model_path"],
        },
    }
    _write_json(mode_dir / "status.json", mode_summary)
    return {
        "mode_name": mode_name,
        "mode_dir": mode_dir,
        "feature_columns": feature_columns,
        "candidates": candidates,
        "trial_records": trial_records,
        "fit_seconds": fit_seconds,
        "emissions": emissions,
        "summary": mode_summary,
    }


def _feature_importance(
    winner: dict[str, Any],
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    part_b_dir = output_dir / "part_b"
    validation_rows = _sort_validation_rows(_split_rows(frame, "validation"))
    sample_rows = validation_rows.iloc[:2000].copy()
    sample_ids = sample_rows["athlete_id"].astype(str).tolist()
    X_sample = sample_rows[winner["_feature_columns"]]
    y_sample = sample_rows["total_lift"].to_numpy(dtype=float)

    importance = permutation_importance(
        winner["_model"],
        X_sample,
        y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=SEED,
        n_jobs=2,
    )
    rows = []
    for index, feature in enumerate(winner["_feature_columns"]):
        repeat_values = [float(value) for value in importance.importances[index, :]]
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(importance.importances_mean[index]),
                "importance_std": float(importance.importances_std[index]),
                "increase_rmse_repeat_1": repeat_values[0],
                "increase_rmse_repeat_2": repeat_values[1],
                "increase_rmse_repeat_3": repeat_values[2],
            }
        )
    rows.sort(key=lambda item: (-item["importance_mean"], item["feature"]))
    top_five = rows[:5]
    top_three = [item["feature"] for item in rows[:3]]
    if len(top_five) != 5 or len(top_three) != 3:
        raise IncompleteExperiment("feature_importance", "Full-feature permutation importance returned fewer than five features")

    sample_sha256 = _sha256_ids(sample_ids)
    _write_csv(part_b_dir / "feature_importance_all.csv", rows)
    _write_csv(part_b_dir / "feature_importance.csv", top_five)
    metadata = {
        "source_mode": winner["feature_mode"],
        "source_estimator": winner["estimator"],
        "source_validation_rmse": winner["validation_rmse"],
        "sample_split": "validation",
        "sample_row_count": len(sample_ids),
        "sample_id_sha256": sample_sha256,
        "sample_id_order": "lexicographically sorted athlete_id, first min(2000,N_validation)",
        "n_repeats": 3,
        "random_state": SEED,
        "scoring": "neg_root_mean_squared_error",
        "importance_direction": "increase_in_rmse",
        "importance_definition": "permuted RMSE minus baseline RMSE; larger is more important",
        "test_split_used": False,
    }
    _write_json(part_b_dir / "feature_importance_metadata.json", metadata)
    selected = {
        "features": top_three,
        "importance_sample_id_sha256": sample_sha256,
        "importance_source_estimator": winner["estimator"],
        "importance_source_mode": winner["feature_mode"],
        "selection_basis": "validation permutation importance; top three by increase in RMSE",
    }
    _write_json(part_b_dir / "selected_top3_features.json", selected)
    return top_three, metadata


def _plot_part_b(winner: dict[str, Any], output_dir: Path) -> None:
    plots_dir = output_dir / "part_b" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    actual = winner["_validation_rows"]["total_lift"].to_numpy(dtype=float)
    predicted = winner["_validation_predictions"]
    minimum = float(min(np.min(actual), np.min(predicted)))
    maximum = float(max(np.max(actual), np.max(predicted)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(actual, predicted, alpha=0.7, s=28, color="#4267ac")
    ax.plot([minimum, maximum], [minimum, maximum], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual total_lift (validation)")
    ax.set_ylabel("Predicted total_lift")
    ax.set_title(f"Part B selected model: {winner['feature_mode']} / {winner['estimator']}")
    fig.tight_layout()
    fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = predicted - actual
    ax.scatter(predicted, residuals, alpha=0.7, s=28, color="#55a868")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted total_lift")
    ax.set_ylabel("Prediction minus actual")
    ax.set_title("Part B selected model validation residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=160)
    plt.close(fig)


def _log_part_b_mode(
    mode: dict[str, Any],
    experiment_id: str,
    manifest: dict[str, Any],
    output_dir: Path,
    feature_importance_paths: list[Path],
) -> str:
    best = min(mode["candidates"], key=lambda row: (row["validation_rmse"], row["estimator"]))
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"part_b_{mode['mode_name']}",
        tags={"part": "B", "feature_mode": mode["mode_name"], "selection_metric": "validation_rmse"},
    ) as active_run:
        mlflow.log_params(
            {
                "feature_mode": mode["mode_name"],
                "feature_columns": json.dumps(mode["feature_columns"]),
                "estimator_list": json.dumps(LEARNERS),
                "learner_selector": "roundrobin",
                "task": "regression",
                "metric": "rmse",
                "eval_method": "holdout",
                "sample": False,
                "model_history": True,
                "log_type": "all",
                "retrain_full": False,
                "ensemble": False,
                "n_jobs": 2,
                "seed": SEED,
                "time_budget_seconds": mode["summary"]["time_budget_seconds"],
                "max_iter": mode["summary"]["max_iter"] if mode["summary"]["max_iter"] is not None else "unset",
                "selection_metric": "validation_rmse",
                "tracking_backend": "local sqlite",
                "codecarbon_backend": mode["emissions"]["cpu_backend"],
                "codecarbon_tdp_watts": mode["emissions"]["cpu_tdp_watts"],
            },
            synchronous=True,
        )
        mlflow.log_metrics(
            {
                "best_validation_rmse": best["validation_rmse"],
                "best_validation_mae": best["validation_mae"],
                "best_validation_r2": best["validation_r2"],
                "flaml_fit_wall_time_seconds": mode["fit_seconds"],
                "model_refit_seconds_total": mode["summary"]["model_refit_seconds_total"],
                "energy_kwh": mode["emissions"]["energy_kwh"],
                "emissions_kg": mode["emissions"]["emissions_kg"],
                "trained_model_count": len(mode["candidates"]),
            },
            synchronous=True,
        )
        _mlflow_log_provenance({"source_files": manifest, "split_seed": SEED})
        _mlflow_log_artifact(mode["mode_dir"], "mode_outputs")
        for path in feature_importance_paths:
            _mlflow_log_artifact(path, "feature_selection")
        return active_run.info.run_id


def _part_b(
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    output_dir: Path,
    experiment_id: str,
    time_budget: float,
    max_iter: int | None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    part_b_dir = output_dir / "part_b"
    part_b_dir.mkdir(parents=True, exist_ok=True)
    frame = frames["v2"]
    validation_ids = sorted(
        _split_rows(frame, "validation")["athlete_id"].astype(str).tolist()
    )
    batch_ids = validation_ids[:1000]

    modes: list[dict[str, Any]] = []
    try:
        full_mode = _run_flaml_mode(
            "full_10",
            FEATURES_BY_VERSION["v2"],
            frame,
            output_dir,
            time_budget,
            max_iter,
            batch_ids,
        )
        modes.append(full_mode)
        full_winner = min(
            full_mode["candidates"],
            key=lambda row: (row["validation_rmse"], row["estimator"]),
        )
        top3_features, importance_meta = _feature_importance(full_winner, frame, output_dir)
        top3_mode = _run_flaml_mode(
            "top3",
            top3_features,
            frame,
            output_dir,
            time_budget,
            max_iter,
            batch_ids,
        )
        modes.append(top3_mode)
    except IncompleteExperiment:
        raise

    all_candidates = [candidate for mode in modes for candidate in mode["candidates"]]
    by_rmse = sorted(all_candidates, key=lambda row: (row["validation_rmse"], row["feature_mode"], row["estimator"]))
    by_speed = sorted(all_candidates, key=lambda row: (row["median_batch_ms"], row["feature_mode"], row["estimator"]))
    _write_csv(part_b_dir / "candidates.csv", [_public_record(candidate) for candidate in all_candidates])
    _write_csv(part_b_dir / "ranked_by_validation_rmse.csv", [_public_record(row) for row in by_rmse])
    _write_csv(part_b_dir / "ranked_by_inference_speed.csv", [_public_record(row) for row in by_speed])
    _write_csv(part_b_dir / "top3_by_validation_rmse.csv", [_public_record(row) for row in by_rmse[:3]])
    _write_csv(part_b_dir / "top3_by_inference_speed.csv", [_public_record(row) for row in by_speed[:3]])

    selected = by_rmse[0]
    selected_test_rows = _split_rows(frame, "test")
    selected_model = joblib.load(output_dir / selected["model_path"])
    predicted_test = np.asarray(
        selected_model.predict(selected_test_rows[selected["_feature_columns"]]),
        dtype=float,
    )
    test_metrics = _regression_metrics(selected_test_rows["total_lift"].to_numpy(dtype=float), predicted_test)
    test_prediction_path = part_b_dir / "test_predictions.csv"
    _prediction_table(selected_test_rows, predicted_test).to_csv(test_prediction_path, index=False)
    selected_json = {
        "feature_mode": selected["feature_mode"],
        "estimator": selected["estimator"],
        "selection_basis": "validation_rmse",
        "validation_metrics": {
            "rmse": selected["validation_rmse"],
            "mae": selected["validation_mae"],
            "r2": selected["validation_r2"],
        },
        "benchmark": _benchmark_summary(selected),
        "training_time_seconds": selected["training_time_seconds"],
        "model_refit_seconds": selected["model_refit_seconds"],
        "model_materialization_source": selected["model_materialization_source"],
        "test_metrics": test_metrics,
        "test_prediction_count": int(len(selected_test_rows)),
        "test_predictions_path": _path_from_output(output_dir, test_prediction_path),
        "model_path": selected["model_path"],
        "test_evaluated_for_selected_model_only": True,
    }
    _write_json(part_b_dir / "selected_best.json", selected_json)
    _plot_part_b(selected, output_dir)

    feature_importance_paths = [
        part_b_dir / "feature_importance_all.csv",
        part_b_dir / "feature_importance.csv",
        part_b_dir / "feature_importance_metadata.json",
        part_b_dir / "selected_top3_features.json",
    ]
    mlflow_run_ids = [
        _log_part_b_mode(mode, experiment_id, manifest, output_dir, feature_importance_paths)
        for mode in modes
    ]
    for mode, run_id in zip(modes, mlflow_run_ids):
        mode["summary"]["mlflow_run_id"] = run_id
        _write_json(mode["mode_dir"] / "status.json", mode["summary"])

    return (
        {
            "status": "complete",
            "candidate_count": len(all_candidates),
            "candidate_metrics_path": _path_from_output(output_dir, part_b_dir / "candidates.csv"),
            "full_feature_winner": {
                "feature_mode": full_winner["feature_mode"],
                "estimator": full_winner["estimator"],
                "validation_rmse": full_winner["validation_rmse"],
            },
            "permutation_importance": importance_meta,
            "selected_top3_features": top3_features,
            "selected_best": selected_json,
            "modes": [mode["summary"] for mode in modes],
            "benchmark_note": BENCHMARK_NOTE,
        },
        mlflow_run_ids,
        all_candidates,
    )


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        existing = list(output_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"Output directory is not empty; refusing to overwrite existing work: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def run_experiments(
    features_dir: Path,
    output_dir: Path,
    time_budget: float = 120.0,
    max_iter: int | None = None,
) -> dict[str, Any]:
    frames, manifest = load_feature_inputs(features_dir)
    _prepare_output(output_dir)
    tracking_uri, part_a_experiment, part_b_experiment = _initialize_mlflow(output_dir)
    split_counts = {
        split_name: int((frames["v1"]["split"] == split_name).sum())
        for split_name in ("train", "validation", "test")
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "seed": SEED,
        "split_seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "split_counts": split_counts,
        "split_source_note": "Use the shared split labels supplied by the Feast retrieval inputs; validate parity and proportions without recomputing membership.",
        "source_features": manifest,
        "emissions_note": EMISSIONS_NOTE,
        "benchmark_note": BENCHMARK_NOTE,
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_run_ids": {"part_a": [], "part_b": []},
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(output_dir / "input_provenance.json", {"source_features": manifest, "split_seed": SEED})
    try:
        part_a_result, part_a_run_ids, _ = _part_a(
            frames,
            manifest,
            output_dir,
            part_a_experiment,
        )
        summary["part_a"] = part_a_result
        summary["common_benchmark_batch_id_sha256"] = part_a_result[
            "common_benchmark_batch_id_sha256"
        ]
        summary["mlflow_run_ids"]["part_a"] = part_a_run_ids
        _write_json(summary_path, summary)
        part_b_result, part_b_run_ids, _ = _part_b(
            frames,
            manifest,
            output_dir,
            part_b_experiment,
            time_budget,
            max_iter,
        )
        summary["part_b"] = part_b_result
        summary["mlflow_run_ids"]["part_b"] = part_b_run_ids
        summary["status"] = "complete"
        _write_json(summary_path, summary)
        return summary
    except IncompleteExperiment as exc:
        summary["status"] = "incomplete"
        summary["incomplete_stage"] = exc.stage
        summary["incomplete_reason"] = exc.reason
        summary["part_b"] = {"status": "incomplete", "stage": exc.stage, "reason": exc.reason}
        _write_json(summary_path, summary)
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(summary_path, summary)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADSP Assignment 2 Part A random forests and Part B FLAML experiments."
    )
    parser.add_argument("--features", type=Path, required=True, help="Directory containing retrieved_v1.parquet and retrieved_v2.parquet")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory for reports and local MLflow data")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=120.0,
        help="FLAML seconds per feature mode (default: 120)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional FLAML iteration cap, bounded to 3..100",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_budget) or args.time_budget <= 0 or args.time_budget > 3600:
        parser.error("--time-budget must be greater than 0 and at most 3600 seconds")
    if args.max_iter is not None and not 3 <= args.max_iter <= 100:
        parser.error("--max-iter must be between 3 and 100 so each learner family can receive a trial")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = run_experiments(
            args.features,
            args.output,
            time_budget=args.time_budget,
            max_iter=args.max_iter,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if summary["status"] != "complete":
        print(
            f"INCOMPLETE: {summary.get('incomplete_reason', summary.get('failure', 'unknown reason'))}",
            file=sys.stderr,
        )
        return 2
    print(f"Completed ADSP Assignment 2 experiments. Results: {args.output.resolve() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
