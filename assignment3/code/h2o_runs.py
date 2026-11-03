#!/usr/bin/env python3
"""Run the Assignment 2 H2O AutoML comparisons on Feast-retrieved features.

This CLI consumes ``retrieved_v2.parquet`` from the local Feast feature store,
trains two H2O AutoML runs (all ten features and the top three selected from the
all-feature validation permutation importance), and evaluates only the
validation-selected model from each run on the untouched test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil


H2O_VERSION = "3.46.0.12"
DEFAULT_JAVA_HOME = "/opt/homebrew/opt/openjdk@17"
TARGET = "total_lift"
ID_COLUMN = "athlete_id"
SPLIT_COLUMN = "split"
INPUT_FILE = "retrieved_v2.parquet"
ALL_FEATURES = [
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
]
ALGORITHMS = ["GLM", "DRF", "GBM"]
SPLITS = ("train", "validation", "test")
MAX_TIMING_BATCH_ROWS = 1000
MAX_IMPORTANCE_SAMPLE_ROWS = 2000
PERMUTATION_SEED = 42
PERMUTATION_REPEATS = 3
TIMING_WARMUP_CALLS = 1
TIMING_REPEATS = 5


class InputError(ValueError):
    """Raised when the Feast retrieval does not match the expected contract."""


def _sha256_strings(values: Sequence[str]) -> str:
    encoded = "\n".join(values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lexicographic_sample(frame: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, list[str]]:
    """Return the first ``limit`` IDs under stable string lexicographic order."""
    ordered = frame.assign(_lex_id=frame[ID_COLUMN].astype(str)).sort_values(
        "_lex_id", kind="mergesort"
    )
    sample = ordered.head(limit).drop(columns=["_lex_id"]).reset_index(drop=True)
    return sample, sample[ID_COLUMN].astype(str).tolist()


def _read_input(features_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not features_dir.is_dir():
        raise InputError(f"features directory does not exist: {features_dir}")
    input_path = features_dir / INPUT_FILE
    if not input_path.is_file():
        raise InputError(f"missing Feast output file: {input_path}")
    try:
        frame = pd.read_parquet(input_path)
    except Exception as exc:
        raise InputError(f"could not read {input_path}: {exc}") from exc

    required = {ID_COLUMN, TARGET, SPLIT_COLUMN, *ALL_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise InputError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise InputError("retrieved_v2.parquet contains no rows")
    if frame[ID_COLUMN].isna().any():
        raise InputError("athlete_id contains null values")
    if frame[ID_COLUMN].duplicated().any():
        raise InputError("athlete_id must be unique in retrieved_v2.parquet")

    normalized_splits = frame[SPLIT_COLUMN].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(normalized_splits).difference(SPLITS))
    if invalid_splits:
        raise InputError(
            "split must contain only train, validation, and test; found: "
            + ", ".join(invalid_splits)
        )
    frame = frame.copy()
    frame[SPLIT_COLUMN] = normalized_splits

    numeric_columns = [TARGET, *ALL_FEATURES]
    for column in numeric_columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise InputError(f"{column} contains non-numeric values")
        finite_values = numeric.dropna().to_numpy(dtype=float)
        if np.isinf(finite_values).any():
            raise InputError(f"{column} contains infinite values")
        frame[column] = numeric.astype(float)

    if frame[TARGET].isna().any():
        raise InputError("total_lift contains null values")

    counts = {split: int((frame[SPLIT_COLUMN] == split).sum()) for split in SPLITS}
    empty_splits = [split for split, count in counts.items() if count == 0]
    if empty_splits:
        raise InputError("empty split(s): " + ", ".join(empty_splits))
    return frame, counts


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise InputError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise InputError(
                f"refusing to overwrite a non-empty output directory: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise RuntimeError("prediction and target arrays have incompatible shapes")
    if not np.isfinite(predicted).all():
        raise RuntimeError("model produced non-finite predictions")
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _prediction_values(model: Any, frame: Any) -> np.ndarray:
    import h2o

    prediction_frame = model.predict(frame)
    try:
        predictions = prediction_frame.as_data_frame(use_pandas=True)
        if "predict" not in predictions.columns:
            raise RuntimeError(
                "H2O regression prediction output is missing its 'predict' column"
            )
        return predictions["predict"].to_numpy(dtype=float)
    finally:
        h2o.remove(prediction_frame)


def _measure_prediction_speed(
    model: Any,
    batch_frame: Any,
    batch_rows: int,
) -> tuple[float, float]:
    for _ in range(TIMING_WARMUP_CALLS):
        warmup = _prediction_values(model, batch_frame)
        if len(warmup) != batch_rows:
            raise RuntimeError("H2O warm-up prediction returned the wrong row count")

    durations_ms: list[float] = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        predictions = _prediction_values(model, batch_frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(predictions) != batch_rows:
            raise RuntimeError("H2O timing prediction returned the wrong row count")
        durations_ms.append(elapsed_ms)

    median_ms = float(np.median(durations_ms))
    return median_ms, median_ms / batch_rows


def _rank_leaderboard(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if len(rows) < len(ALGORITHMS):
        raise RuntimeError(
            f"H2O AutoML trained only {len(rows)} models; at least {len(ALGORITHMS)} are required"
        )
    algorithms = {str(row["algorithm"]).upper() for row in rows}
    missing_algorithms = sorted(set(ALGORITHMS).difference(algorithms))
    if missing_algorithms:
        raise RuntimeError(
            "H2O AutoML did not train the required model families: "
            + ", ".join(missing_algorithms)
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["validation_rmse", "model_id"], kind="mergesort"
    ).reset_index(drop=True)
    result["validation_rank"] = np.arange(1, len(result) + 1)
    speed_order = result.sort_values(
        ["prediction_batch_median_ms", "validation_rmse", "model_id"],
        kind="mergesort",
    )
    speed_rank = {
        model_id: rank
        for rank, model_id in enumerate(speed_order["model_id"].tolist(), start=1)
    }
    result["speed_rank"] = result["model_id"].map(speed_rank).astype(int)
    return result


def _fit_mode(
    *,
    h2o: Any,
    H2OAutoML: Any,
    mode: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    timing_batch_df: pd.DataFrame,
    max_models: int,
    time_budget: int,
    project_prefix: str,
    output_dir: Path,
) -> dict[str, Any]:
    training_frame = h2o.H2OFrame(train_df[[*feature_columns, TARGET]])
    validation_frame = h2o.H2OFrame(validation_df[[*feature_columns, TARGET]])
    timing_frame = h2o.H2OFrame(timing_batch_df[feature_columns])

    automl = H2OAutoML(
        nfolds=0,
        max_models=max_models,
        max_runtime_secs=time_budget,
        seed=42,
        sort_metric="RMSE",
        include_algos=ALGORITHMS,
        project_name=f"{project_prefix}_{mode}",
        verbosity="warn",
    )
    automl.train(
        x=feature_columns,
        y=TARGET,
        training_frame=training_frame,
        validation_frame=validation_frame,
        leaderboard_frame=validation_frame,
    )

    native_leaderboard_frame = h2o.automl.get_leaderboard(
        automl, extra_columns="ALL"
    )
    native_leaderboard = native_leaderboard_frame.as_data_frame(use_pandas=True)
    required_native_columns = {
        "model_id",
        "algo",
        "training_time_ms",
        "predict_time_per_row_ms",
    }
    missing_native = sorted(required_native_columns.difference(native_leaderboard.columns))
    h2o.remove(native_leaderboard_frame)
    if missing_native:
        raise RuntimeError(
            "H2O extended leaderboard is missing columns: "
            + ", ".join(missing_native)
        )

    actual_validation = validation_df[TARGET].to_numpy(dtype=float)
    validation_features = h2o.H2OFrame(validation_df[feature_columns])
    timing_rows = len(timing_batch_df)
    leaderboard_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for native_row in native_leaderboard.to_dict(orient="records"):
        model_id = str(native_row["model_id"])
        model = h2o.get_model(model_id)
        if model is None:
            raise RuntimeError(f"H2O leaderboard references missing model {model_id}")
        models[model_id] = model

        validation_predictions = _prediction_values(model, validation_features)
        validation_rmse = _rmse(actual_validation, validation_predictions)
        batch_median_ms, batch_median_ms_per_row = _measure_prediction_speed(
            model, timing_frame, timing_rows
        )

        training_time_ms = float(native_row["training_time_ms"])
        native_predict_ms_per_row = float(native_row["predict_time_per_row_ms"])
        if not np.isfinite(training_time_ms) or not np.isfinite(native_predict_ms_per_row):
            raise RuntimeError(f"H2O returned non-finite timing values for {model_id}")

        leaderboard_rows.append(
            {
                "model_id": model_id,
                "algorithm": str(native_row["algo"]).upper(),
                "validation_rmse": validation_rmse,
                "training_time_ms": training_time_ms,
                "native_predict_time_per_row_ms": native_predict_ms_per_row,
                "prediction_batch_median_ms": batch_median_ms,
                "prediction_batch_median_ms_per_row": batch_median_ms_per_row,
            }
        )

    h2o.remove(validation_features)
    h2o.remove([training_frame, validation_frame, timing_frame])
    leaderboard = _rank_leaderboard(leaderboard_rows)
    best_model_id = str(leaderboard.iloc[0]["model_id"])
    best_model = models[best_model_id]

    model_dir = output_dir / "models" / mode
    model_dir.mkdir(parents=True, exist_ok=False)
    saved_path = h2o.save_model(model=best_model, path=str(model_dir), force=False)
    saved_path_obj = Path(saved_path).resolve()
    relative_saved_path = saved_path_obj.relative_to(output_dir.resolve()).as_posix()

    leaderboard_path = output_dir / f"leaderboard_{mode}.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    return {
        "mode": mode,
        "features": list(feature_columns),
        "leaderboard": leaderboard,
        "leaderboard_csv": leaderboard_path.name,
        "best_validation_model_id": best_model_id,
        "best_validation_algorithm": str(leaderboard.iloc[0]["algorithm"]),
        "top3_by_validation_model_ids": leaderboard.head(3)["model_id"].astype(str).tolist(),
        "top3_by_prediction_speed_model_ids": leaderboard.sort_values(
            ["speed_rank"], kind="mergesort"
        ).head(3)["model_id"].astype(str).tolist(),
        "models": models,
        "saved_model_path": relative_saved_path,
        "best_model": best_model,
    }


def _validation_permutation_importance(
    *,
    h2o: Any,
    model: Any,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sample, athlete_ids = _lexicographic_sample(
        validation_df, MAX_IMPORTANCE_SAMPLE_ROWS
    )
    sample_features = sample[feature_columns].reset_index(drop=True)
    actual = sample[TARGET].to_numpy(dtype=float)
    sample_hash = _sha256_strings(athlete_ids)

    baseline_frame = h2o.H2OFrame(sample_features)
    try:
        baseline_predictions = _prediction_values(model, baseline_frame)
    finally:
        h2o.remove(baseline_frame)
    baseline_rmse = _rmse(actual, baseline_predictions)

    random_seed = int(
        np.random.RandomState(PERMUTATION_SEED).randint(
            np.iinfo(np.int32).max + 1
        )
    )
    repeat_rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        random_state = np.random.RandomState(random_seed)
        shuffling_indices = np.arange(len(sample_features))
        original_values = sample_features[feature].to_numpy(copy=True)
        permuted_values = original_values.copy()
        permuted_source_indices = np.arange(len(sample_features))
        for repeat_index in range(1, PERMUTATION_REPEATS + 1):
            # Match sklearn's in-place permutation composition with a fresh
            # RandomState per feature and one shared derived seed.
            random_state.shuffle(shuffling_indices)
            permuted_values = permuted_values[shuffling_indices]
            permuted_source_indices = permuted_source_indices[shuffling_indices]
            permuted_features = sample_features.copy()
            permuted_features[feature] = permuted_values
            prediction_frame = h2o.H2OFrame(permuted_features)
            try:
                predictions = _prediction_values(model, prediction_frame)
            finally:
                h2o.remove(prediction_frame)
            permuted_rmse = _rmse(actual, predictions)
            repeat_rows.append(
                {
                    "feature": feature,
                    "repeat": repeat_index,
                    "baseline_validation_rmse": baseline_rmse,
                    "permuted_validation_rmse": permuted_rmse,
                    "rmse_increase": permuted_rmse - baseline_rmse,
                    "permuted_source_order_sha256": _sha256_strings(
                        [athlete_ids[index] for index in permuted_source_indices]
                    ),
                    "validation_sample_rows": len(sample),
                    "sample_athlete_id_sha256": sample_hash,
                    "seed": PERMUTATION_SEED,
                }
            )

    repeats = pd.DataFrame(repeat_rows)
    importance = (
        repeats.groupby("feature", as_index=False)
        .agg(
            baseline_validation_rmse=("baseline_validation_rmse", "first"),
            mean_permuted_validation_rmse=("permuted_validation_rmse", "mean"),
            mean_rmse_increase=("rmse_increase", "mean"),
            std_rmse_increase=("rmse_increase", "std"),
            repeat_count=("repeat", "count"),
            validation_sample_rows=("validation_sample_rows", "first"),
            sample_athlete_id_sha256=("sample_athlete_id_sha256", "first"),
            seed=("seed", "first"),
        )
        .sort_values(
            ["mean_rmse_increase", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    importance["std_rmse_increase"] = importance["std_rmse_increase"].fillna(0.0)
    importance["rank"] = np.arange(1, len(importance) + 1)
    ordered_features = importance["feature"].tolist()
    rank_map = {feature: rank for rank, feature in enumerate(ordered_features, start=1)}
    repeats["rank"] = repeats["feature"].map(rank_map).astype(int)
    repeats = repeats.sort_values(["rank", "repeat"], kind="mergesort").reset_index(
        drop=True
    )
    protocol = {
        "method": "validation_permutation_rmse_increase",
        "partition": "validation",
        "source_model_id": str(model.model_id),
        "sample_rows": len(sample),
        "sample_athlete_id_sha256": sample_hash,
        "seed": PERMUTATION_SEED,
        "repeats": PERMUTATION_REPEATS,
        "score": "mean increase in RMSE over the unpermuted validation sample",
        "uses_sklearn_permutation_composition": True,
        "test_used": False,
    }
    return importance, repeats, sample, protocol


def _predict_test_for_selected_models(
    *,
    h2o: Any,
    runs: dict[str, dict[str, Any]],
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    actual = test_df[TARGET].to_numpy(dtype=float)
    for mode, result in runs.items():
        feature_columns = result["features"]
        test_frame = h2o.H2OFrame(test_df[feature_columns])
        model = result["best_model"]
        try:
            predicted = _prediction_values(model, test_frame)
        finally:
            h2o.remove(test_frame)
        test_rmse = _rmse(actual, predicted)
        test_mae = float(np.mean(np.abs(actual - predicted)))
        metric_rows.append(
            {
                "mode": mode,
                "model_id": result["best_validation_model_id"],
                "algorithm": result["best_validation_algorithm"],
                "test_rows": len(test_df),
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "selected_by": "external validation RMSE",
            }
        )
        for athlete_id, actual_value, predicted_value in zip(
            test_df[ID_COLUMN].tolist(), actual.tolist(), predicted.tolist()
        ):
            prediction_rows.append(
                {
                    "mode": mode,
                    ID_COLUMN: athlete_id,
                    "actual_total_lift": float(actual_value),
                    "predicted_total_lift": float(predicted_value),
                    "model_id": result["best_validation_model_id"],
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    return predictions, metrics


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _reap_orphaned_owned_server(run_name: str, java_home: Path) -> None:
    """Stop only a server process whose unique H2O name matches this run."""
    expected_java = (java_home / "bin" / "java").resolve()
    for process in psutil.process_iter(attrs=["pid", "cmdline", "exe"]):
        try:
            info = process.info
            command = info.get("cmdline") or []
            executable = info.get("exe") or ""
            if run_name not in command or Path(executable).resolve() != expected_java:
                continue
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                continue
            try:
                process.wait(timeout=8)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.is_running():
                raise RuntimeError(
                    f"owned H2O JVM {process.pid} survived startup cleanup"
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _stop_owned_server(server: Any, process: Any, process_start_time: float) -> dict[str, Any]:
    pid = int(process.pid)
    shutdown_error = None
    fallback_used = False
    try:
        server.shutdown()
    except Exception as exc:
        shutdown_error = f"{type(exc).__name__}: {exc}"

    # H2O 3.46.0.12's shutdown implementation can clear its handle after a
    # swallowed final-poll error. Keep the exact Popen object and verify it.
    try:
        if process.poll() is None:
            fallback_used = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.poll() is None:
            raise RuntimeError(f"owned H2O JVM {pid} remained alive after shutdown")
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.poll() is None:
            raise

    result = {
        "name": str(server.name),
        "ip": str(server.ip),
        "port": int(server.port),
        "pid": pid,
        "process_start_time": float(process_start_time),
        "exit_code": int(process.returncode),
        "cleanup_verified": process.poll() is not None,
        "shutdown_fallback_used": fallback_used,
    }
    if shutdown_error:
        result["library_shutdown_error"] = shutdown_error
    if not result["cleanup_verified"]:
        raise RuntimeError(f"owned H2O JVM {pid} did not exit")
    return result


def run_pipeline(
    *,
    features_dir: Path,
    output_dir: Path,
    max_models: int,
    time_budget: int,
    java_home: Path,
) -> dict[str, Any]:
    if max_models < len(ALGORITHMS):
        raise InputError(f"max_models must be at least {len(ALGORITHMS)}")
    if time_budget < 1:
        raise InputError("time_budget must be at least one second")

    java_binary = java_home / "bin" / "java"
    if not java_binary.is_file():
        raise InputError(f"Java home does not contain bin/java: {java_home}")
    frame, split_counts = _read_input(features_dir)
    _prepare_output(output_dir)
    train_df = frame.loc[frame[SPLIT_COLUMN] == "train"].reset_index(drop=True)
    validation_df = frame.loc[frame[SPLIT_COLUMN] == "validation"].reset_index(drop=True)
    test_df = frame.loc[frame[SPLIT_COLUMN] == "test"].reset_index(drop=True)

    timing_batch_df, timing_ids = _lexicographic_sample(
        validation_df, MAX_TIMING_BATCH_ROWS
    )
    timing_hash = _sha256_strings(timing_ids)
    run_name = f"ADSP2_H2O_{uuid.uuid4().hex}"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "input_file": INPUT_FILE,
        "target": TARGET,
        "predictor_policy": "fixed ten Feast features; athlete_id, split, target, and extra fields excluded",
        "all_features": list(ALL_FEATURES),
        "selected_feature_count": 3,
        "partition_counts": split_counts,
        "max_models_per_run": int(max_models),
        "time_budget_seconds_per_run": int(time_budget),
        "runs": {},
        "top5_features": [],
        "selected_top3_features": [],
        "feature_selection": {},
        "prediction_timing": {
            "method": "median H2O client prediction request plus H2OFrame-to-pandas materialization wall time",
            "batch_rows": int(len(timing_batch_df)),
            "athlete_id_sha256": timing_hash,
            "athlete_id_order": "str(athlete_id), ascending lexicographic, stable sort",
            "warmup_calls": TIMING_WARMUP_CALLS,
            "measured_repeats": TIMING_REPEATS,
            "includes_h2o_rest_and_jvm_serialization": True,
            "excludes_jvm_startup": True,
        },
    }

    runtime_dir = output_dir / "h2o_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["H2O_JAVA_HOME"] = str(java_home)
    os.environ["DO_NOT_TRACK"] = "1"
    tempfile.tempdir = str(runtime_dir)

    import h2o
    from h2o.automl import H2OAutoML
    from h2o.backend.server import H2OLocalServer

    if h2o.__version__ != H2O_VERSION:
        raise RuntimeError(f"expected H2O {H2O_VERSION}, found {h2o.__version__}")
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise RuntimeError(f"H2O requires numpy<2; found {np.__version__}")

    server = None
    server_process = None
    connection = None
    process_start_time = None
    server_started_at = time.time()
    success = False
    try:
        server, server_process, process_start_time = _start_server(
            H2OLocalServer,
            run_name=run_name,
            runtime_dir=runtime_dir,
            max_mem_gb=2,
            java_home=java_home,
        )
        connection = h2o.connect(
            server=server,
            strict_version_check=True,
            verbose=False,
            telemetry=False,
        )
        connection_ip = connection.ip()
        connection_port = int(connection.port())
        if connection_ip != server.ip or connection_port != int(server.port):
            raise RuntimeError(
                "H2O connection did not attach to the owned local server: "
                f"connected to {connection_ip}:{connection_port}, "
                f"owned server is {server.ip}:{server.port}"
            )

        all_result = _fit_mode(
            h2o=h2o,
            H2OAutoML=H2OAutoML,
            mode="all_features",
            feature_columns=list(ALL_FEATURES),
            train_df=train_df,
            validation_df=validation_df,
            timing_batch_df=timing_batch_df,
            max_models=max_models,
            time_budget=time_budget,
            project_prefix=run_name,
            output_dir=output_dir,
        )
        all_best_model = all_result["best_model"]
        importance, importance_repeats, importance_sample, feature_protocol = (
            _validation_permutation_importance(
                h2o=h2o,
                model=all_best_model,
                validation_df=validation_df,
                feature_columns=list(ALL_FEATURES),
            )
        )
        importance.to_csv(output_dir / "feature_importance.csv", index=False)
        importance_repeats.to_csv(
            output_dir / "feature_importance_repeats.csv", index=False
        )
        top5_features = importance.head(5)["feature"].astype(str).tolist()
        top3_features = top5_features[:3]
        if len(top3_features) != 3:
            raise RuntimeError("validation importance did not produce three features")

        top3_result = _fit_mode(
            h2o=h2o,
            H2OAutoML=H2OAutoML,
            mode="top3_features",
            feature_columns=top3_features,
            train_df=train_df,
            validation_df=validation_df,
            timing_batch_df=timing_batch_df,
            max_models=max_models,
            time_budget=time_budget,
            project_prefix=run_name,
            output_dir=output_dir,
        )
        runs = {
            "all_features": all_result,
            "top3_features": top3_result,
        }

        # The test frame is constructed only after feature selection and both
        # validation-ranked model sets are complete.
        _, test_metrics = _predict_test_for_selected_models(
            h2o=h2o,
            runs=runs,
            test_df=test_df,
            output_dir=output_dir,
        )

        summary["top5_features"] = top5_features
        summary["selected_top3_features"] = top3_features
        summary["feature_selection"] = {
            **feature_protocol,
            "source_algorithm": all_result["best_validation_algorithm"],
            "selected_features": top3_features,
            "reported_top5_features": top5_features,
        }
        summary["runs"] = {
            mode: {
                "features": result["features"],
                "leaderboard_csv": result["leaderboard_csv"],
                "best_validation_model_id": result["best_validation_model_id"],
                "best_validation_algorithm": result["best_validation_algorithm"],
                "top3_by_validation_model_ids": result[
                    "top3_by_validation_model_ids"
                ],
                "top3_by_prediction_speed_model_ids": result[
                    "top3_by_prediction_speed_model_ids"
                ],
                "saved_model_path": result["saved_model_path"],
                "test_rmse": float(
                    test_metrics.loc[test_metrics["mode"] == mode, "test_rmse"].iloc[0]
                ),
                "test_mae": float(
                    test_metrics.loc[test_metrics["mode"] == mode, "test_mae"].iloc[0]
                ),
            }
            for mode, result in runs.items()
        }
        summary["software"] = {
            "python": sys.version.split()[0],
            "h2o": h2o.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        success = True
    finally:
        connection_close_error = None
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                connection_close_error = f"{type(exc).__name__}: {exc}"
        if server is not None and server_process is not None:
            summary["server"] = _stop_owned_server(
                server, server_process, float(process_start_time)
            )
            if connection_close_error:
                summary["server"]["connection_close_error"] = connection_close_error
        else:
            _reap_orphaned_owned_server(run_name, java_home)
            summary["server"] = {
                "name": run_name,
                "started": False,
                "cleanup_verified": True,
                "started_at_epoch": server_started_at,
            }
        if success:
            _write_json(output_dir / "summary.json", summary)

    return summary


def _start_server(
    server_class: Any,
    *,
    run_name: str,
    runtime_dir: Path,
    max_mem_gb: int,
    java_home: Path,
) -> tuple[Any, Any, float]:
    ice_dir = runtime_dir / "ice"
    logs_dir = runtime_dir / "logs"
    ice_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["H2O_JAVA_HOME"] = str(java_home)
    try:
        server = server_class.start(
            nthreads=2,
            max_mem_size=max_mem_gb * 1024**3,
            port="54321+",
            name=run_name,
            ice_root=str(ice_dir),
            log_dir=str(logs_dir),
            bind_to_localhost=True,
            verbose=False,
        )
    except BaseException:
        _reap_orphaned_owned_server(run_name, java_home)
        raise
    process = getattr(server, "_process", None)
    if process is None:
        _reap_orphaned_owned_server(run_name, java_home)
        raise RuntimeError("H2O local server returned without its process handle")
    try:
        start_time = float(psutil.Process(process.pid).create_time())
    except psutil.Error:
        start_time = time.time()
    return server, process, start_time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="directory containing Feast's retrieved_v2.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty directory for reports, predictions, and selected models",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=6,
        help="H2O AutoML model limit for each feature mode (minimum 3)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=120,
        help="maximum H2O AutoML seconds for each feature mode",
    )
    parser.add_argument(
        "--java-home",
        type=Path,
        default=Path(DEFAULT_JAVA_HOME),
        help="Java home used by the owned local H2O JVM",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(
            features_dir=args.features,
            output_dir=args.output,
            max_models=args.max_models,
            time_budget=args.time_budget,
            java_home=args.java_home,
        )
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"H2O run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
