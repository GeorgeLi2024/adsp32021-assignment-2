#!/usr/bin/env python3
"""Refit and benchmark every retained October 18 FLAML trial configuration.

The original trial validation scores are preserved from FLAML's training logs.
This script does not rerun the AutoML search or touch the test split. It fits
each unique logged config on the original training partition and times the
resulting model against the original shared 1,000-row validation batch.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


FEATURE_COLUMNS = {
    "full_10": [
        "age", "height", "weight", "is_female", "age_squared", "height_squared",
        "weight_squared", "age_x_weight", "age_x_height", "weight_x_height",
    ],
    "top3": ["is_female", "weight", "height"],
}
MODE_DIRS = {"full_10": "full_10", "top3": "top3"}
LEARNERS = {"rf", "extra_tree", "lgbm"}
EXPECTED_SPLITS = {"train": 49227, "validation": 10549, "test": 10549}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_config(learner: str, config: dict[str, Any]) -> str:
    return json.dumps({"learner": learner, "config": config}, sort_keys=True, separators=(",", ":"))


def read_jsonl_trials(path: Path) -> list[dict[str, Any]]:
    trials = []
    seen_records: set[int] = set()
    seen_configs: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        # FLAML appends one curr_best_record_id marker after trial rows.
        if "record_id" not in record:
            continue
        required = {"record_id", "learner", "validation_loss", "config", "sample_size"}
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"{path}:{line_number} is missing {missing}")
        record_id = int(record["record_id"])
        learner = str(record["learner"])
        if record_id in seen_records:
            raise ValueError(f"{path}:{line_number} duplicates record_id {record_id}")
        if learner not in LEARNERS:
            raise ValueError(f"{path}:{line_number} has unexpected learner {learner}")
        if not isinstance(record["config"], dict) or not record["config"]:
            raise ValueError(f"{path}:{line_number} has an empty or invalid config")
        score = float(record["validation_loss"])
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line_number} has non-finite validation_loss")
        key = canonical_config(learner, record["config"])
        if key in seen_configs:
            raise ValueError(f"{path}:{line_number} repeats a config; deduplicate before training")
        seen_records.add(record_id)
        seen_configs.add(key)
        trials.append(record)
    if len(trials) != 30:
        raise ValueError(f"{path} has {len(trials)} trials; expected 30")
    return trials


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    run_dir = run_dir.resolve(strict=True)
    manifest_path = run_dir / "stage_manifest.json"
    data_path = run_dir / "prepared/data_receipt.json"
    summary_path = run_dir / "ml_runs/summary.json"
    retrieved_path = run_dir / "feature_store/retrieved_v2.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The source run manifest is not complete")
    if manifest.get("input", {}).get("sha256") != data.get("source", {}).get("archive_sha256"):
        raise ValueError("Source archive hash differs between the run manifest and data receipt")
    if manifest.get("source_hashes", {}).get("final_verified", {}).get("retrieved_v2") != sha256(retrieved_path):
        raise ValueError("Retrieved v2 Parquet hash does not match the completed source run manifest")
    if data.get("features", {}).get("raw_target_contributors_ids_and_names_excluded_from_predictors") is not True:
        raise ValueError("Data receipt does not confirm target contributors are excluded from predictors")
    target = data.get("target", {})
    if target.get("column") != "total_lift" or target.get("formula") != "candj + snatch + deadlift + backsq":
        raise ValueError("The retained target/formula differs from the documented prior choice")
    expected_partitions = data.get("split", {}).get("sizes")
    if expected_partitions != EXPECTED_SPLITS:
        raise ValueError(f"Expected split sizes {EXPECTED_SPLITS}, got {expected_partitions}")
    if summary.get("split_counts") != EXPECTED_SPLITS:
        raise ValueError("The retained FLAML summary has different split counts")
    trials_by_mode = {}
    batch_ids_by_mode = {}
    for mode, subdir in MODE_DIRS.items():
        log_path = run_dir / "ml_runs/part_b" / subdir / "training_log.jsonl"
        candidates = read_jsonl_trials(log_path)
        if any(int(t["sample_size"]) != EXPECTED_SPLITS["train"] for t in candidates):
            raise ValueError(f"{mode} includes a trial that did not use the full training partition")
        trials_by_mode[mode] = candidates
        ids_path = run_dir / "ml_runs/part_b" / subdir / "inference_batch_ids.csv"
        ids = [str(row["athlete_id"]) for row in csv_rows(ids_path)]
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise ValueError(f"{mode} inference batch must contain 1,000 unique validation IDs")
        batch_ids_by_mode[mode] = ids
    if batch_ids_by_mode["full_10"] != batch_ids_by_mode["top3"]:
        raise ValueError("Full10 and top3 modes do not share the exact validation benchmark IDs")
    return manifest, trials_by_mode, {"data": data, "summary": summary, "batch_ids": batch_ids_by_mode["full_10"], "retrieved_path": retrieved_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed FLAML run output root")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output path; existing paths are refused")
    parser.add_argument("--n-jobs", type=int, default=2, help="Threads per refit; defaults to the original run setting")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_jobs > 2:
        raise SystemExit("--n-jobs must be 1 or 2 to match the original two-thread cap")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing output directory: {output}")

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    manifest, trials_by_mode, inputs = validate_inputs(run_dir)
    # Verify the installed package versions before importing the modeling API.
    import importlib.metadata as metadata
    version = metadata.version("flaml")
    if version != "2.7.0":
        raise SystemExit(f"Expected the pinned FLAML 2.7.0, found {version}")

    import numpy as np
    import pandas as pd
    from flaml.automl import AutoML

    frame = pd.read_parquet(inputs["retrieved_path"])
    if frame.columns.duplicated().any():
        raise ValueError("Retrieved v2 source has duplicated columns")
    expected_columns = ["athlete_id", "total_lift", "split", *FEATURE_COLUMNS["full_10"]]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Retrieved v2 columns do not match the recorded schema: {list(frame.columns)}")
    frame = frame.copy()
    frame["athlete_id"] = frame["athlete_id"].astype(str)
    if frame["athlete_id"].duplicated().any():
        raise ValueError("Retrieved v2 contains duplicate athlete IDs")
    if frame["split"].value_counts().to_dict() != EXPECTED_SPLITS:
        raise ValueError("Retrieved v2 partition counts do not match the retained run")

    validation_rows = frame[frame["split"] == "validation"].sort_values("athlete_id", key=lambda s: s.astype(str)).reset_index(drop=True)
    batch_ids = inputs["batch_ids"]
    validation_indexed = validation_rows.set_index("athlete_id")
    if any(i not in validation_indexed.index for i in batch_ids):
        raise ValueError("The shared benchmark batch contains a non-validation athlete ID")
    batch_frame = validation_indexed.loc[batch_ids]

    # Output is created only after all source and schema checks pass.
    output.mkdir(parents=True, mode=0o700)
    progress_path = output / "candidate-refits.jsonl"
    run_started = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "status": "in_progress",
        "started_at_utc": run_started,
        "flaml_version": version,
        "python_version": sys.version.split()[0],
        "n_jobs": args.n_jobs,
        "source_run_started_at_utc": manifest.get("started_at_utc"),
        "source_run_finished_at_utc": manifest.get("finished_at_utc"),
        "source_run_archive_sha256": inputs["data"]["source"]["archive_sha256"],
        "retrieved_v2_sha256": sha256(inputs["retrieved_path"]),
        "feature_modes": {mode: {"trial_count": len(trials), "log_sha256": sha256(run_dir / "ml_runs/part_b" / MODE_DIRS[mode] / "training_log.jsonl")} for mode, trials in trials_by_mode.items()},
        "benchmark_batch_ids_sha256": hashlib.sha256("\n".join(batch_ids).encode("utf-8")).hexdigest(),
        "benchmark_rows": len(batch_ids),
        "warmups": 1,
        "repeats": 5,
        "timing_scope": "end-to-end local Python model.predict call; wrapper/dataframe overhead included",
        "test_split_used_for_fit_or_ranking": False,
    }
    write_json_new(output / "run-manifest.json", run_manifest)
    fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)

    total = sum(len(v) for v in trials_by_mode.values())
    completed = 0
    for mode, subdir in MODE_DIRS.items():
        feature_columns = FEATURE_COLUMNS[mode]
        train_rows = frame[frame["split"] == "train"]
        validation_mode = validation_rows
        x_train = train_rows[feature_columns]
        y_train = train_rows["total_lift"]
        x_validation = validation_mode[feature_columns]
        y_validation = validation_mode["total_lift"].to_numpy(dtype=float)
        x_batch = batch_frame[feature_columns]
        log_path = run_dir / "ml_runs/part_b" / subdir / "training_log.jsonl"

        # Original validation losses define the score rank. Refit every unique
        # logged config on the original training split solely to get a common
        # candidate-level speed measurement.
        for trial in sorted(trials_by_mode[mode], key=lambda r: (float(r["validation_loss"]), int(r["record_id"]))):
            learner = str(trial["learner"])
            record_id = int(trial["record_id"])
            refitter = AutoML(fit_kwargs_by_estimator={learner: {}})
            refit_started = time.perf_counter()
            refitter.retrain_from_log(
                log_file_name=str(log_path),
                X_train=x_train,
                y_train=y_train,
                task="regression",
                eval_method="holdout",
                record_id=record_id,
                train_full=True,
                n_jobs=args.n_jobs,
            )
            fit_seconds = time.perf_counter() - refit_started
            if refitter.model is None:
                raise RuntimeError(f"FLAML returned no refitted model for {mode} trial record {record_id}")

            validation_prediction = np.asarray(refitter.predict(x_validation), dtype=float).reshape(-1)
            if not np.isfinite(validation_prediction).all():
                raise RuntimeError(f"Non-finite validation prediction for {mode} trial record {record_id}")
            errors = validation_prediction - y_validation
            refit_rmse = float(np.sqrt(np.mean(errors ** 2)))
            logged_rmse = float(trial["validation_loss"])
            validation_close = math.isclose(refit_rmse, logged_rmse, rel_tol=1e-5, abs_tol=1e-5)

            # Match the established one-warmup/five-repeat protocol. The exact
            # same 1,000 validation IDs are passed to every candidate.
            warmup = np.asarray(refitter.predict(x_batch), dtype=float).reshape(-1)
            if len(warmup) != len(batch_ids) or not np.isfinite(warmup).all():
                raise RuntimeError(f"Invalid warmup prediction for {mode} trial record {record_id}")
            elapsed_ms = []
            for _ in range(5):
                started = time.perf_counter()
                prediction = np.asarray(refitter.predict(x_batch), dtype=float).reshape(-1)
                elapsed_ms.append((time.perf_counter() - started) * 1000.0)
                if len(prediction) != len(batch_ids) or not np.isfinite(prediction).all():
                    raise RuntimeError(f"Invalid timing prediction for {mode} trial record {record_id}")
            median_batch_ms = float(statistics.median(elapsed_ms))
            if not math.isfinite(median_batch_ms) or median_batch_ms <= 0:
                raise RuntimeError(f"Invalid median latency for {mode} trial record {record_id}")

            append_jsonl(progress_path, {
                "feature_mode": mode,
                "record_id": record_id,
                "learner": learner,
                "original_validation_rmse": logged_rmse,
                "refit_validation_rmse": refit_rmse,
                "refit_validation_matches_original": validation_close,
                "config": trial["config"],
                "sample_size": int(trial["sample_size"]),
                "refit_seconds": fit_seconds,
                "median_batch_ms": median_batch_ms,
                "median_per_row_ms": median_batch_ms / len(batch_ids),
                "batch_ids_sha256": run_manifest["benchmark_batch_ids_sha256"],
                "batch_rows": len(batch_ids),
                "repeats": len(elapsed_ms),
                "warmups": 1,
                "timing_scope": run_manifest["timing_scope"],
                "fit_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows_used": False,
            })
            completed += 1
            print(f"measured {completed}/{total}: {mode} record_id={record_id} latency={median_batch_ms:.4f} ms/1000")

    # Assemble rankings only after every model got a real benchmark.
    measured = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    if len(measured) != total:
        raise RuntimeError(f"Expected {total} candidate measurements, got {len(measured)}")
    with (output / "validation-ranking.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_mode", "validation_rank", "record_id", "learner", "original_validation_rmse", "refit_validation_rmse", "refit_validation_matches_original", "config_json"])
        writer.writeheader()
        for mode in MODE_DIRS:
            rows = sorted((r for r in measured if r["feature_mode"] == mode), key=lambda r: (r["original_validation_rmse"], r["record_id"]))
            for rank, row in enumerate(rows, 1):
                writer.writerow({"feature_mode": mode, "validation_rank": rank, "record_id": row["record_id"], "learner": row["learner"], "original_validation_rmse": row["original_validation_rmse"], "refit_validation_rmse": row["refit_validation_rmse"], "refit_validation_matches_original": row["refit_validation_matches_original"], "config_json": json.dumps(row["config"], sort_keys=True, separators=(",", ":"))})
    with (output / "speed-ranking.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_mode", "speed_rank", "record_id", "learner", "original_validation_rmse", "refit_validation_rmse", "median_batch_ms", "median_per_row_ms", "batch_rows", "repeats", "warmups", "fit_rows", "validation_rows", "test_rows_used", "config_json"])
        writer.writeheader()
        for mode in MODE_DIRS:
            rows = sorted((r for r in measured if r["feature_mode"] == mode), key=lambda r: (r["median_batch_ms"], r["record_id"]))
            for rank, row in enumerate(rows, 1):
                writer.writerow({"feature_mode": mode, "speed_rank": rank, "record_id": row["record_id"], "learner": row["learner"], "original_validation_rmse": row["original_validation_rmse"], "refit_validation_rmse": row["refit_validation_rmse"], "median_batch_ms": row["median_batch_ms"], "median_per_row_ms": row["median_per_row_ms"], "batch_rows": row["batch_rows"], "repeats": row["repeats"], "warmups": row["warmups"], "fit_rows": row["fit_rows"], "validation_rows": row["validation_rows"], "test_rows_used": row["test_rows_used"], "config_json": json.dumps(row["config"], sort_keys=True, separators=(",", ":"))})

    matches = sum(r["refit_validation_matches_original"] for r in measured)
    finish = {
        **run_manifest,
        "status": "complete" if len(measured) == total else "partial",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(measured),
        "expected_candidate_count": total,
        "refit_validation_match_count": matches,
        "refit_validation_mismatch_count": total - matches,
        "validation_rank_source": "original October 18 FLAML training-log validation_loss; per-mode top-three trial ranking preserved",
        "speed_rank_source": "November 3 local refits of each unique trial config; same 1,000 validation IDs, one warmup and five repeats",
        "test_split_used_for_fit_or_ranking": False,
    }
    write_json_new(output / "run-result.json", finish)
    print(json.dumps({"status": finish["status"], "candidate_count": len(measured), "validation_match_count": matches, "output_dir": str(output)}, indent=2))
    return 0 if finish["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
