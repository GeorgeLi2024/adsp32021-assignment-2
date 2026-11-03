#!/usr/bin/env python3
"""Run the four local Assignment 2 stages sequentially and record verified outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_TIMING_ROWS = 1000
MODEL_KEYS = (
    "part_a_selected",
    "flaml_full10",
    "flaml_top3",
    "h2o_all10",
    "h2o_top3",
)
COMPARISON_COLUMNS = [
    "model_key",
    "engine",
    "feature_set",
    "algorithm",
    "model_id",
    "feature_columns",
    "validation_rmse",
    "batch_rows",
    "warmups",
    "repeats",
    "median_batch_ms",
    "median_ms_per_row",
    "batch_id_sha256",
    "timing_scope",
]


class PipelineError(RuntimeError):
    """Raised when a stage result or cross-stage contract cannot be verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"expected {label} receipt was not created: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"could not read {label} receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} receipt must contain a JSON object")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise PipelineError(f"expected {label} artifact was not created: {path}")
    if path.stat().st_size == 0:
        raise PipelineError(f"expected {label} artifact is empty: {path}")
    return path


def _path_inside(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PipelineError(f"{label} path must be relative to its output directory: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PipelineError(f"{label} path escapes its output directory: {relative}")
    return resolved


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise PipelineError(f"{label} is not finite: {value!r}")
    return number


def _sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PipelineError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PipelineError(f"{label} is not a SHA-256 digest") from exc
    return value.lower()


def _normalize_split_counts(actual: Any, label: str) -> dict[str, int]:
    if not isinstance(actual, dict):
        raise PipelineError(f"{label} must be a split-count mapping")
    if set(actual) != {"train", "validation", "test"}:
        raise PipelineError(f"{label} must contain train, validation, and test only")
    normalized: dict[str, int] = {}
    for key in ("train", "validation", "test"):
        try:
            normalized[key] = int(actual[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"{label} is missing a valid {key} count") from exc
        if normalized[key] <= 0:
            raise PipelineError(f"{label} must have a positive {key} count")
    return normalized


def _same_mapping(actual: Any, expected: dict[str, int], label: str) -> dict[str, int]:
    normalized = _normalize_split_counts(actual, label)
    if normalized != expected:
        raise PipelineError(f"{label} mismatch: expected {expected}, found {normalized}")
    return normalized


def _verify_preparation(
    output_dir: Path,
    source_sha256: str,
) -> dict[str, Any]:
    stage_dir = output_dir / "prepared"
    receipt_path = stage_dir / "data_receipt.json"
    receipt = _read_json(receipt_path, "prepared data")
    recorded_source_sha = _sha256_value(
        receipt.get("source", {}).get("archive_sha256"),
        "data_receipt.json source.archive_sha256",
    )
    if recorded_source_sha != source_sha256:
        raise PipelineError(
            "prepared data receipt source hash does not match the input ZIP "
            f"({recorded_source_sha} != {source_sha256})"
        )
    if receipt.get("target", {}).get("formula") != "candj + snatch + deadlift + backsq":
        raise PipelineError("prepared data receipt has an unexpected target formula")
    cohort_rows = int(receipt.get("cohort", {}).get("row_count", 0))
    if cohort_rows <= 0:
        raise PipelineError("prepared data receipt reports an empty cohort")
    split_sizes = _normalize_split_counts(
        receipt.get("split", {}).get("sizes"), "prepared data split sizes"
    )
    if sum(split_sizes.values()) != cohort_rows:
        raise PipelineError("prepared data split sizes do not sum to the cohort row count")

    expected_outputs = {
        "cohort": "data/cohort.parquet",
        "features_v1": "data/features_v1.parquet",
        "features_v2": "data/features_v2.parquet",
    }
    if receipt.get("outputs") != expected_outputs:
        raise PipelineError(f"prepared data output paths do not match the expected interface: {receipt.get('outputs')!r}")
    artifact_hashes: dict[str, str] = {}
    artifact_paths: dict[str, Path] = {}
    for key, relative in expected_outputs.items():
        path = _require_file(stage_dir / relative, f"prepared {key}")
        artifact_paths[key] = path
        artifact_hashes[key] = _sha256(path)
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "receipt_sha256": _sha256(receipt_path),
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "cohort_rows": cohort_rows,
        "split_sizes": split_sizes,
    }


def _verify_feature_store(
    output_dir: Path,
    source_sha256: str,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    stage_dir = output_dir / "feature_store"
    receipt_path = stage_dir / "feature_store_receipt.json"
    receipt = _read_json(receipt_path, "feature store")
    inputs = receipt.get("input_data_receipt", {})
    if _sha256_value(inputs.get("source_archive_sha256"), "feature store source archive hash") != source_sha256:
        raise PipelineError("feature store receipt does not trace to the requested instructor ZIP")
    if inputs.get("sha256") != prepared["receipt_sha256"]:
        raise PipelineError("feature store receipt does not trace to the prepared data receipt")
    read_only = receipt.get("input_read_only_verification", {})
    if read_only.get("unchanged") is not True or read_only.get("before_sha256") != read_only.get("after_sha256"):
        raise PipelineError("feature store did not verify that prepared inputs remained unchanged")

    retrieved = receipt.get("retrieved_outputs")
    if not isinstance(retrieved, dict) or set(retrieved) != {"v1", "v2"}:
        raise PipelineError("feature store receipt must include retrieved v1 and v2 outputs")
    output_hashes: dict[str, str] = {}
    output_paths: dict[str, Path] = {}
    for version in ("v1", "v2"):
        metadata = retrieved[version]
        if int(metadata.get("rows", -1)) != prepared["cohort_rows"]:
            raise PipelineError(f"feature store {version} row count does not match the prepared cohort")
        expected_name = f"retrieved_{version}.parquet"
        if metadata.get("path") != expected_name:
            raise PipelineError(f"feature store {version} output path is unexpected: {metadata.get('path')!r}")
        path = _require_file(_path_inside(stage_dir, expected_name, f"retrieved {version}"), f"retrieved {version}")
        output_paths[version] = path
        output_hashes[version] = _sha256(path)
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "receipt_sha256": _sha256(receipt_path),
        "output_paths": output_paths,
        "output_hashes": output_hashes,
    }


def _verify_benchmark(
    benchmark: Any,
    label: str,
    expected_hash: str,
    expected_rows: int,
) -> dict[str, Any]:
    if not isinstance(benchmark, dict):
        raise PipelineError(f"{label} benchmark receipt is missing")
    batch_hash = _sha256_value(benchmark.get("batch_id_sha256"), f"{label} batch ID hash")
    if batch_hash != expected_hash:
        raise PipelineError(f"{label} used a different prediction batch than the other models")
    try:
        batch_rows = int(benchmark["row_count"])
        warmups = int(benchmark["warmups"])
        repeats = int(benchmark["repeats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"{label} benchmark receipt has invalid batch counts") from exc
    if batch_rows != expected_rows:
        raise PipelineError(f"{label} benchmark used {batch_rows} rows, expected {expected_rows}")
    if warmups != 1 or repeats != 5:
        raise PipelineError(f"{label} benchmark must use one warmup and five measured repeats")
    median_batch_ms = _finite_number(benchmark.get("median_batch_ms"), f"{label} median batch time")
    median_per_row_ms = _finite_number(benchmark.get("median_per_row_ms"), f"{label} median per-row time")
    if median_batch_ms <= 0 or median_per_row_ms <= 0:
        raise PipelineError(f"{label} benchmark timings must be positive")
    scope = benchmark.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise PipelineError(f"{label} benchmark scope is missing")
    return {
        "batch_id_sha256": batch_hash,
        "batch_rows": batch_rows,
        "warmups": warmups,
        "repeats": repeats,
        "median_batch_ms": median_batch_ms,
        "median_ms_per_row": median_per_row_ms,
        "timing_scope": scope,
    }


def _verify_ml_runs(
    output_dir: Path,
    feature_store: dict[str, Any],
    prepared: dict[str, Any],
    ml_time_budget: float,
    ml_max_iter: int | None,
) -> dict[str, Any]:
    stage_dir = output_dir / "ml_runs"
    receipt_path = stage_dir / "summary.json"
    receipt = _read_json(receipt_path, "ML runs")
    if receipt.get("status") != "complete":
        raise PipelineError(f"ML run receipt is not complete: {receipt.get('status')!r}")
    source_features = receipt.get("source_features")
    if not isinstance(source_features, dict):
        raise PipelineError("ML run receipt is missing source feature hashes")
    for version in ("v1", "v2"):
        recorded = source_features.get(version, {}).get("sha256")
        if _sha256_value(recorded, f"ML {version} source feature hash") != feature_store["output_hashes"][version]:
            raise PipelineError(f"ML {version} input hash does not match the Feast retrieval")
    _same_mapping(receipt.get("split_counts"), prepared["split_sizes"], "ML split counts")

    part_a = receipt.get("part_a", {})
    part_b = receipt.get("part_b", {})
    if part_a.get("status") != "complete" or part_b.get("status") != "complete":
        raise PipelineError("ML Part A and Part B must both report complete")
    selected_part_a = part_a.get("selected_best", {})
    if selected_part_a.get("selection_basis") != "validation_rmse":
        raise PipelineError("Part A selected model is not selected by validation RMSE")
    if part_b.get("status") != "complete":
        raise PipelineError("ML Part B is incomplete")
    modes = part_b.get("modes")
    if not isinstance(modes, list):
        raise PipelineError("ML Part B receipt is missing per-mode results")
    modes_by_name = {mode.get("feature_mode"): mode for mode in modes if isinstance(mode, dict)}
    if set(modes_by_name) != {"full_10", "top3"}:
        raise PipelineError(f"ML Part B modes are incomplete: {sorted(modes_by_name)}")
    for mode_name, mode in modes_by_name.items():
        if mode.get("status") != "complete":
            raise PipelineError(f"FLAML {mode_name} mode did not complete")
        try:
            mode_budget = float(mode["time_budget_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"FLAML {mode_name} time budget is missing") from exc
        if not math.isclose(mode_budget, ml_time_budget, rel_tol=0, abs_tol=1e-9):
            raise PipelineError(f"FLAML {mode_name} time budget does not match the requested cap")
        if mode.get("max_iter") != ml_max_iter:
            raise PipelineError(f"FLAML {mode_name} max_iter does not match the requested cap")
        best = mode.get("best_by_validation_rmse")
        if not isinstance(best, dict) or not isinstance(best.get("validation_metrics"), dict):
            raise PipelineError(f"FLAML {mode_name} selected validation model is missing")
        if not best.get("estimator") or not best.get("model_path"):
            raise PipelineError(f"FLAML {mode_name} selected model identity is incomplete")

    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "receipt_sha256": _sha256(receipt_path),
        "modes_by_name": modes_by_name,
    }


def _read_h2o_leaderboard(path: Path, model_id: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"{label} leaderboard was not created: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        matches = [row for row in reader if row.get("model_id") == model_id]
    if len(matches) != 1:
        raise PipelineError(f"{label} leaderboard must contain exactly one selected model row")
    row = matches[0]
    return {
        "validation_rmse": _finite_number(row.get("validation_rmse"), f"{label} validation RMSE"),
        "median_batch_ms": _finite_number(
            row.get("prediction_batch_median_ms"), f"{label} median batch timing"
        ),
        "median_ms_per_row": _finite_number(
            row.get("prediction_batch_median_ms_per_row"), f"{label} median per-row timing"
        ),
        "algorithm": str(row.get("algorithm", "")),
    }


def _verify_h2o_runs(
    output_dir: Path,
    feature_store: dict[str, Any],
    prepared: dict[str, Any],
    h2o_max_models: int,
    h2o_time_budget: int,
) -> dict[str, Any]:
    stage_dir = output_dir / "h2o_runs"
    receipt_path = stage_dir / "summary.json"
    receipt = _read_json(receipt_path, "H2O runs")
    if receipt.get("input_file") != "retrieved_v2.parquet":
        raise PipelineError("H2O receipt reports an unexpected input file")
    if int(receipt.get("max_models_per_run", -1)) != h2o_max_models:
        raise PipelineError("H2O model-count cap was not forwarded accurately")
    if int(receipt.get("time_budget_seconds_per_run", -1)) != h2o_time_budget:
        raise PipelineError("H2O time-budget cap was not forwarded accurately")
    _same_mapping(receipt.get("partition_counts"), prepared["split_sizes"], "H2O partition counts")
    software = receipt.get("software", {})
    if not str(software.get("python", "")).startswith("3.11."):
        raise PipelineError(f"H2O did not run under the supplied Python 3.11 environment: {software.get('python')!r}")
    if software.get("h2o") != "3.46.0.12":
        raise PipelineError(f"H2O runtime version mismatch: {software.get('h2o')!r}")
    numpy_version = str(software.get("numpy", ""))
    try:
        numpy_major = int(numpy_version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise PipelineError(f"H2O receipt has an invalid NumPy version: {numpy_version!r}") from exc
    if numpy_major >= 2:
        raise PipelineError(f"H2O runtime used unsupported NumPy {numpy_version}")

    timing = receipt.get("prediction_timing", {})
    batch_hash = _sha256_value(timing.get("athlete_id_sha256"), "H2O timing batch ID hash")
    expected_rows = min(MAX_TIMING_ROWS, prepared["split_sizes"]["validation"])
    if int(timing.get("batch_rows", -1)) != expected_rows:
        raise PipelineError("H2O timing batch row count does not match the shared validation batch")
    if int(timing.get("warmup_calls", -1)) != 1 or int(timing.get("measured_repeats", -1)) != 5:
        raise PipelineError("H2O timing must use one warmup and five measured repeats")
    if timing.get("includes_h2o_rest_and_jvm_serialization") is not True:
        raise PipelineError("H2O timing receipt does not disclose its client/JVM scope")
    if timing.get("excludes_jvm_startup") is not True:
        raise PipelineError("H2O timing receipt does not disclose JVM startup exclusion")

    runs = receipt.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"all_features", "top3_features"}:
        raise PipelineError("H2O receipt must include all-feature and top-three runs")
    selected_top3 = receipt.get("selected_top3_features")
    if not isinstance(selected_top3, list) or len(selected_top3) != 3:
        raise PipelineError("H2O receipt does not report exactly three selected features")

    leaderboards: dict[str, dict[str, Any]] = {}
    for mode, expected_feature_count in (("all_features", 10), ("top3_features", 3)):
        run = runs[mode]
        features = run.get("features")
        if not isinstance(features, list) or len(features) != expected_feature_count:
            raise PipelineError(f"H2O {mode} feature list has the wrong size")
        if not run.get("best_validation_model_id") or not run.get("best_validation_algorithm"):
            raise PipelineError(f"H2O {mode} selected model identity is incomplete")
        leaderboard_name = run.get("leaderboard_csv")
        if not isinstance(leaderboard_name, str):
            raise PipelineError(f"H2O {mode} leaderboard path is missing")
        leaderboard_path = _path_inside(stage_dir, leaderboard_name, f"H2O {mode} leaderboard")
        selected = _read_h2o_leaderboard(
            leaderboard_path,
            str(run["best_validation_model_id"]),
            f"H2O {mode}",
        )
        if selected["algorithm"].upper() != str(run["best_validation_algorithm"]).upper():
            raise PipelineError(f"H2O {mode} leaderboard algorithm does not match its receipt")
        selected["model_id"] = str(run["best_validation_model_id"])
        selected["features"] = [str(value) for value in features]
        selected["leaderboard_path"] = leaderboard_name
        leaderboards[mode] = selected

    for version in ("v1", "v2"):
        if _sha256(feature_store["output_paths"][version]) != feature_store["output_hashes"][version]:
            raise PipelineError(f"Feast retrieved_{version}.parquet changed before H2O input verification")
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "receipt_sha256": _sha256(receipt_path),
        "batch_hash": batch_hash,
        "leaderboards": leaderboards,
    }


def _comparison_row(
    *,
    model_key: str,
    engine: str,
    feature_set: str,
    algorithm: str,
    model_id: str,
    feature_columns: list[str],
    validation_rmse: Any,
    benchmark: dict[str, Any],
    expected_hash: str,
    expected_rows: int,
) -> dict[str, Any]:
    normalized_benchmark = _verify_benchmark(benchmark, model_key, expected_hash, expected_rows)
    return {
        "model_key": model_key,
        "engine": engine,
        "feature_set": feature_set,
        "algorithm": algorithm,
        "model_id": model_id,
        "feature_columns": json.dumps(feature_columns, separators=(",", ":")),
        "validation_rmse": _finite_number(validation_rmse, f"{model_key} validation RMSE"),
        "batch_rows": normalized_benchmark["batch_rows"],
        "warmups": normalized_benchmark["warmups"],
        "repeats": normalized_benchmark["repeats"],
        "median_batch_ms": normalized_benchmark["median_batch_ms"],
        "median_ms_per_row": normalized_benchmark["median_ms_per_row"],
        "batch_id_sha256": normalized_benchmark["batch_id_sha256"],
        "timing_scope": normalized_benchmark["timing_scope"],
    }


def _build_comparison(
    *,
    output_dir: Path,
    prepared: dict[str, Any],
    feature_store: dict[str, Any],
    ml: dict[str, Any],
    h2o: dict[str, Any],
) -> dict[str, Any]:
    ml_receipt = ml["receipt"]
    h2o_receipt = h2o["receipt"]
    split_counts = prepared["split_sizes"]
    _same_mapping(ml_receipt.get("split_counts"), split_counts, "ML comparison split counts")
    _same_mapping(h2o_receipt.get("partition_counts"), split_counts, "H2O comparison partition counts")
    expected_rows = min(MAX_TIMING_ROWS, split_counts["validation"])
    batch_hash = h2o["batch_hash"]
    part_a = ml_receipt["part_a"]["selected_best"]
    part_a_features = prepared["receipt"].get("features", {}).get(part_a.get("feature_version"))
    if not isinstance(part_a_features, list):
        raise PipelineError("Part A selected feature version is absent from the prepared data receipt")
    rows = [
        _comparison_row(
            model_key="part_a_selected",
            engine="scikit-learn",
            feature_set=str(part_a["feature_version"]),
            algorithm="RandomForestRegressor",
            model_id=str(part_a["run_id"]),
            feature_columns=[str(value) for value in part_a_features],
            validation_rmse=part_a.get("validation_metrics", {}).get("rmse"),
            benchmark=part_a.get("benchmark"),
            expected_hash=batch_hash,
            expected_rows=expected_rows,
        )
    ]

    modes = ml["modes_by_name"]
    for mode_name, model_key in (("full_10", "flaml_full10"), ("top3", "flaml_top3")):
        mode = modes[mode_name]
        best = mode["best_by_validation_rmse"]
        rows.append(
            _comparison_row(
                model_key=model_key,
                engine="FLAML",
                feature_set=mode_name,
                algorithm=str(best["estimator"]),
                model_id=str(best["model_path"]),
                feature_columns=[str(value) for value in mode["feature_columns"]],
                validation_rmse=best.get("validation_metrics", {}).get("rmse"),
                benchmark=best.get("benchmark"),
                expected_hash=batch_hash,
                expected_rows=expected_rows,
            )
        )

    h2o_scope_metadata = h2o_receipt["prediction_timing"]
    h2o_scope = (
        f"{h2o_scope_metadata['method']}; includes H2O REST/JVM serialization; "
        "excludes JVM startup; not a pure algorithm timing"
    )
    for mode, model_key in (("all_features", "h2o_all10"), ("top3_features", "h2o_top3")):
        selected = h2o["leaderboards"][mode]
        timing = {
            "batch_id_sha256": batch_hash,
            "row_count": int(h2o_scope_metadata["batch_rows"]),
            "warmups": int(h2o_scope_metadata["warmup_calls"]),
            "repeats": int(h2o_scope_metadata["measured_repeats"]),
            "median_batch_ms": selected["median_batch_ms"],
            "median_per_row_ms": selected["median_ms_per_row"],
            "scope": h2o_scope,
        }
        rows.append(
            _comparison_row(
                model_key=model_key,
                engine="H2O AutoML",
                feature_set=mode,
                algorithm=selected["algorithm"],
                model_id=selected["model_id"],
                feature_columns=selected["features"],
                validation_rmse=selected["validation_rmse"],
                benchmark=timing,
                expected_hash=batch_hash,
                expected_rows=expected_rows,
            )
        )

    if {row["model_key"] for row in rows} != set(MODEL_KEYS) or len(rows) != len(MODEL_KEYS):
        raise PipelineError("cross-framework comparison does not contain exactly the five selected models")
    csv_path = output_dir / "combined_comparison.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "status": "complete",
        "csv_path": "combined_comparison.csv",
        "csv_sha256": _sha256(csv_path),
        "validation_partition_counts": split_counts,
        "timing_batch": {
            "rows": expected_rows,
            "athlete_id_sha256": batch_hash,
            "warmups": 1,
            "repeats": 5,
            "selection": "first 1,000 validation athlete IDs in stable lexical order, or all validation IDs when fewer",
        },
        "timing_scope_note": (
            "H2O values include client requests, REST/JVM serialization, and H2OFrame-to-pandas materialization; "
            "they exclude JVM startup and are not pure algorithm timings. Python values are model.predict "
            "call timings with the source-reported wrapper/dataframe scope."
        ),
        "models": rows,
    }


def _stage_manifest_record(
    *,
    name: str,
    command: list[str],
    receipt_path: str,
    stdout_path: str,
    stderr_path: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "not_run",
        "command": command,
        "returncode": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "duration_seconds": None,
        "input_hashes": {},
        "receipt": {"path": receipt_path, "verified": False, "sha256": None},
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
    }


def _save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(output_dir / "stage_manifest.json", manifest)


def _verify_final_source_hashes(
    input_path: Path,
    source_sha256: str,
    prepared: dict[str, Any],
    feature_store: dict[str, Any],
) -> dict[str, str]:
    final_hashes = {"instructor_zip": _sha256(input_path)}
    if final_hashes["instructor_zip"] != source_sha256:
        raise PipelineError("instructor ZIP changed during the assignment pipeline")
    final_hashes["data_receipt"] = _sha256(prepared["receipt_path"])
    if final_hashes["data_receipt"] != prepared["receipt_sha256"]:
        raise PipelineError("prepared data receipt changed during the assignment pipeline")
    for key, path in prepared["artifact_paths"].items():
        actual = _sha256(path)
        if actual != prepared["artifact_hashes"][key]:
            raise PipelineError(f"prepared {key} artifact changed during the assignment pipeline")
        final_hashes[f"prepared_{key}"] = actual
    final_hashes["feature_store_receipt"] = _sha256(feature_store["receipt_path"])
    if final_hashes["feature_store_receipt"] != feature_store["receipt_sha256"]:
        raise PipelineError("feature-store receipt changed during the assignment pipeline")
    for version, path in feature_store["output_paths"].items():
        actual = _sha256(path)
        if actual != feature_store["output_hashes"][version]:
            raise PipelineError(f"retrieved_{version}.parquet changed during the assignment pipeline")
        final_hashes[f"retrieved_{version}"] = actual
    return final_hashes


def _unrun_failure(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    stage: dict[str, Any],
    message: str,
    returncode: int,
) -> int:
    stage["status"] = "failed"
    stage["failure"] = message
    manifest["status"] = "failed"
    manifest["failure"] = {"stage": stage["name"], "message": message}
    manifest["finished_at_utc"] = _utc_now()
    if manifest.get("summary", {}).get("model_comparison", {}).get("status") == "pending":
        manifest["summary"]["model_comparison"] = {"status": "not_run", "reason": message}
    _save_manifest(output_dir, manifest)
    print(f"run_assignment: {stage['name']} failed: {message}", file=sys.stderr)
    return returncode if returncode != 0 else 1


def run_assignment(
    *,
    input_path: Path,
    output_dir: Path,
    h2o_python: Path,
    java_home: Path,
    ml_time_budget: float,
    ml_max_iter: int | None,
    h2o_time_budget: int,
    h2o_max_models: int,
) -> int:
    assignment_dir = Path(__file__).resolve().parent
    input_path = input_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().absolute()
    h2o_python = h2o_python.expanduser().absolute()
    java_home = java_home.expanduser().absolute()
    main_python = assignment_dir / ".venv" / "bin" / "python"
    if not input_path.is_file():
        raise PipelineError(f"input ZIP is not a file: {input_path}")
    if not main_python.is_file() or not os.access(main_python, os.X_OK):
        raise PipelineError(f"main Python 3.12 environment is missing or not executable: {main_python}")
    if not h2o_python.is_file() or not os.access(h2o_python, os.X_OK):
        raise PipelineError(f"H2O Python interpreter is missing or not executable: {h2o_python}")
    if not java_home.is_dir() or not (java_home / "bin" / "java").is_file():
        raise PipelineError(f"Java home does not contain bin/java: {java_home}")
    scripts = {
        "prepare_data": assignment_dir / "prepare_data.py",
        "feature_store": assignment_dir / "feature_store_pipeline.py",
        "ml_runs": assignment_dir / "ml_runs.py",
        "h2o_runs": assignment_dir / "h2o_runs.py",
    }
    missing_scripts = [str(path) for path in scripts.values() if not path.is_file()]
    if missing_scripts:
        raise PipelineError(f"required stage script(s) are missing: {missing_scripts}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    source_sha256 = _sha256(input_path)
    prepared_dir = output_dir / "prepared"
    feature_store_dir = output_dir / "feature_store"
    ml_runs_dir = output_dir / "ml_runs"
    h2o_runs_dir = output_dir / "h2o_runs"
    commands = [
        [
            str(main_python), str(scripts["prepare_data"]), "--input", str(input_path),
            "--output", str(prepared_dir),
        ],
        [
            str(main_python), str(scripts["feature_store"]), "--data", str(prepared_dir / "data"),
            "--output", str(feature_store_dir),
        ],
        [
            str(main_python), str(scripts["ml_runs"]), "--features", str(feature_store_dir),
            "--output", str(ml_runs_dir), "--time-budget", str(ml_time_budget),
            *(["--max-iter", str(ml_max_iter)] if ml_max_iter is not None else []),
        ],
        [
            str(h2o_python), str(scripts["h2o_runs"]), "--features", str(feature_store_dir),
            "--output", str(h2o_runs_dir), "--max-models", str(h2o_max_models),
            "--time-budget", str(h2o_time_budget), "--java-home", str(java_home),
        ],
    ]
    stage_data = [
        ("prepare_data", "prepared/data_receipt.json"),
        ("feature_store", "feature_store/feature_store_receipt.json"),
        ("ml_runs", "ml_runs/summary.json"),
        ("h2o_runs", "h2o_runs/summary.json"),
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "input": {"path": str(input_path), "sha256": source_sha256},
        "parameters": {
            "main_python": str(main_python.resolve()),
            "h2o_python": str(h2o_python),
            "java_home": str(java_home),
            "java_home_resolved": str(java_home.resolve()),
            "ml_time_budget_seconds_per_mode": ml_time_budget,
            "ml_max_iter": ml_max_iter,
            "h2o_time_budget_seconds_per_mode": h2o_time_budget,
            "h2o_max_models_per_mode": h2o_max_models,
        },
        "source_hashes": {"instructor_zip_sha256": source_sha256},
        "stages": [],
        "summary": {"model_comparison": {"status": "pending"}},
    }
    for (name, receipt_relative), command in zip(stage_data, commands):
        manifest["stages"].append(
            _stage_manifest_record(
                name=name,
                command=command,
                receipt_path=receipt_relative,
                stdout_path=f"logs/{name}.stdout.log",
                stderr_path=f"logs/{name}.stderr.log",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    _save_manifest(output_dir, manifest)

    prepared: dict[str, Any] | None = None
    feature_store: dict[str, Any] | None = None
    ml: dict[str, Any] | None = None
    h2o: dict[str, Any] | None = None
    for stage, command in zip(manifest["stages"], commands):
        if stage["name"] == "prepare_data":
            stage["input_hashes"] = {"instructor_zip_sha256": source_sha256}
        elif stage["name"] == "feature_store" and prepared is not None:
            stage["input_hashes"] = {
                "data_receipt_sha256": prepared["receipt_sha256"],
                **{f"prepared_{key}_sha256": value for key, value in prepared["artifact_hashes"].items()},
            }
        elif stage["name"] == "ml_runs" and feature_store is not None:
            stage["input_hashes"] = {
                "feature_store_receipt_sha256": feature_store["receipt_sha256"],
                **{f"retrieved_{version}_sha256": value for version, value in feature_store["output_hashes"].items()},
            }
        elif stage["name"] == "h2o_runs" and feature_store is not None:
            stage["input_hashes"] = {
                "feature_store_receipt_sha256": feature_store["receipt_sha256"],
                "retrieved_v2_sha256": feature_store["output_hashes"]["v2"],
            }

        stdout_file = output_dir / stage["stdout_log"]
        stderr_file = output_dir / stage["stderr_log"]
        stage["status"] = "running"
        stage["started_at_utc"] = _utc_now()
        _save_manifest(output_dir, manifest)
        start = time.monotonic()
        try:
            with stdout_file.open("w", encoding="utf-8") as stdout_handle, stderr_file.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=str(assignment_dir),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
        except OSError as exc:
            stage["returncode"] = None
            stage["finished_at_utc"] = _utc_now()
            stage["duration_seconds"] = round(time.monotonic() - start, 6)
            return _unrun_failure(
                output_dir,
                manifest,
                stage=stage,
                message=f"could not start stage process: {type(exc).__name__}: {exc}",
                returncode=1,
            )
        stage["returncode"] = int(completed.returncode)
        stage["finished_at_utc"] = _utc_now()
        stage["duration_seconds"] = round(time.monotonic() - start, 6)
        _save_manifest(output_dir, manifest)
        if completed.returncode != 0:
            stage["receipt"]["exists"] = Path(output_dir / stage["receipt"]["path"]).is_file()
            if stage["receipt"]["exists"]:
                stage["receipt"]["sha256"] = _sha256(output_dir / stage["receipt"]["path"])
            return _unrun_failure(
                output_dir,
                manifest,
                stage=stage,
                message=(
                    f"process exited with return code {completed.returncode}; "
                    f"logs: {stage['stdout_log']} and {stage['stderr_log']}"
                ),
                returncode=1,
            )

        try:
            if stage["name"] == "prepare_data":
                prepared = _verify_preparation(output_dir, source_sha256)
                manifest["source_hashes"]["data_receipt_sha256"] = prepared["receipt_sha256"]
                manifest["source_hashes"]["prepared_artifacts_sha256"] = prepared["artifact_hashes"]
                manifest["summary"]["cohort_rows"] = prepared["cohort_rows"]
                manifest["summary"]["split_counts"] = prepared["split_sizes"]
                receipt_path = prepared["receipt_path"]
                receipt_sha = prepared["receipt_sha256"]
            elif stage["name"] == "feature_store":
                if prepared is None:
                    raise PipelineError("prepared data was not verified before the feature-store stage")
                feature_store = _verify_feature_store(output_dir, source_sha256, prepared)
                manifest["source_hashes"]["feature_store_receipt_sha256"] = feature_store["receipt_sha256"]
                manifest["source_hashes"]["retrieved_features_sha256"] = feature_store["output_hashes"]
                receipt_path = feature_store["receipt_path"]
                receipt_sha = feature_store["receipt_sha256"]
            elif stage["name"] == "ml_runs":
                if prepared is None or feature_store is None:
                    raise PipelineError("prepared/Feast outputs were not verified before the ML stage")
                ml = _verify_ml_runs(
                    output_dir,
                    feature_store,
                    prepared,
                    ml_time_budget,
                    ml_max_iter,
                )
                manifest["source_hashes"]["ml_summary_sha256"] = ml["receipt_sha256"]
                receipt_path = ml["receipt_path"]
                receipt_sha = ml["receipt_sha256"]
            elif stage["name"] == "h2o_runs":
                if prepared is None or feature_store is None or ml is None:
                    raise PipelineError("prepared, Feast, and ML outputs were not verified before H2O")
                h2o = _verify_h2o_runs(
                    output_dir,
                    feature_store,
                    prepared,
                    h2o_max_models,
                    h2o_time_budget,
                )
                manifest["source_hashes"]["h2o_summary_sha256"] = h2o["receipt_sha256"]
                manifest["source_hashes"]["h2o_retrieved_v2_input_sha256"] = feature_store["output_hashes"]["v2"]
                receipt_path = h2o["receipt_path"]
                receipt_sha = h2o["receipt_sha256"]
            else:
                raise PipelineError(f"unknown pipeline stage: {stage['name']}")
        except Exception as exc:
            stage["receipt"]["exists"] = Path(output_dir / stage["receipt"]["path"]).is_file()
            if stage["receipt"]["exists"]:
                stage["receipt"]["sha256"] = _sha256(output_dir / stage["receipt"]["path"])
            return _unrun_failure(
                output_dir,
                manifest,
                stage=stage,
                message=f"receipt verification failed: {type(exc).__name__}: {exc}",
                returncode=1,
            )

        stage["status"] = "complete"
        stage["receipt"]["verified"] = True
        stage["receipt"]["sha256"] = receipt_sha
        _save_manifest(output_dir, manifest)

    if prepared is None or feature_store is None or ml is None or h2o is None:
        raise PipelineError("one or more pipeline stages finished without verified results")

    try:
        manifest["source_hashes"]["final_verified"] = _verify_final_source_hashes(
            input_path,
            source_sha256,
            prepared,
            feature_store,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"stage": "source_integrity", "message": f"{type(exc).__name__}: {exc}"}
        manifest["summary"]["model_comparison"] = {
            "status": "not_run",
            "reason": f"source integrity verification failed: {type(exc).__name__}: {exc}",
        }
        manifest["finished_at_utc"] = _utc_now()
        _save_manifest(output_dir, manifest)
        print(f"run_assignment: source integrity verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _save_manifest(output_dir, manifest)

    try:
        comparison = _build_comparison(
            output_dir=output_dir,
            prepared=prepared,
            feature_store=feature_store,
            ml=ml,
            h2o=h2o,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"stage": "model_comparison", "message": f"{type(exc).__name__}: {exc}"}
        manifest["summary"]["model_comparison"] = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        manifest["finished_at_utc"] = _utc_now()
        _save_manifest(output_dir, manifest)
        print(f"run_assignment: model comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    manifest["summary"]["model_comparison"] = comparison
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _utc_now()
    _save_manifest(output_dir, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "cohort_rows": prepared["cohort_rows"],
                "split_counts": prepared["split_sizes"],
                "comparison_csv": str(output_dir / comparison["csv_path"]),
                "stage_manifest": str(output_dir / "stage_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def _positive_bounded_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 3600:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 3600 seconds")
    return parsed


def _bounded_h2o_time(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer number of seconds") from exc
    if not 1 <= parsed <= 3600:
        raise argparse.ArgumentTypeError("must be between 1 and 3600 seconds per feature mode")
    return parsed


def _bounded_h2o_models(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 3 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 3 and 100 models per feature mode")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="instructor ZIP containing athletes.csv")
    parser.add_argument("--output", required=True, type=Path, help="new output directory for all stages and receipts")
    parser.add_argument("--h2o-python", required=True, type=Path, help="existing Python 3.11 H2O environment interpreter")
    parser.add_argument("--java-home", required=True, type=Path, help="existing Java 17 home passed to H2O")
    parser.add_argument(
        "--ml-time-budget",
        type=_positive_bounded_float,
        default=120.0,
        help="FLAML time budget in seconds per feature mode, 0 < value <= 3600",
    )
    parser.add_argument(
        "--ml-max-iter",
        type=int,
        default=None,
        help="optional FLAML iteration cap, 3..100",
    )
    parser.add_argument(
        "--h2o-time-budget",
        type=_bounded_h2o_time,
        default=120,
        help="H2O AutoML time budget in seconds per feature mode, 1..3600",
    )
    parser.add_argument(
        "--h2o-max-models",
        type=_bounded_h2o_models,
        default=6,
        help="H2O AutoML model cap per feature mode, 3..100",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ml_max_iter is not None and not 3 <= args.ml_max_iter <= 100:
        parser.error("--ml-max-iter must be between 3 and 100")
    try:
        return run_assignment(
            input_path=args.input,
            output_dir=args.output,
            h2o_python=args.h2o_python,
            java_home=args.java_home,
            ml_time_budget=args.ml_time_budget,
            ml_max_iter=args.ml_max_iter,
            h2o_time_budget=args.h2o_time_budget,
            h2o_max_models=args.h2o_max_models,
        )
    except FileExistsError as exc:
        print(f"run_assignment: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"run_assignment: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
