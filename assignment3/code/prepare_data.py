#!/usr/bin/env python3
"""Prepare the NEW ADSP32021 Assignment 2 study cohort and feature tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = ["age", "height", "weight", "candj", "snatch", "deadlift", "backsq"]
LIFT_COLUMNS = ["candj", "snatch", "deadlift", "backsq"]
LIFT_CAPS = {"candj": 395, "snatch": 496, "deadlift": 1105, "backsq": 1069}
SENTINEL = 8388607.0
EVENT_TIMESTAMP_VALUE = "2026-10-18T00:00:00Z"
EVENT_TIMESTAMP = pd.Timestamp(EVENT_TIMESTAMP_VALUE)
V1_FEATURES = ["age", "height", "weight", "is_female"]
V2_FEATURES = V1_FEATURES + [
    "age_squared",
    "height_squared",
    "weight_squared",
    "age_x_weight",
    "age_x_height",
    "weight_x_height",
]
REQUIRED_COLUMNS = ["athlete_id", "name", "gender", *NUMERIC_COLUMNS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_outputs_absent(output_path: Path) -> None:
    intended_paths = [
        output_path / "data" / "cohort.parquet",
        output_path / "data" / "features_v1.parquet",
        output_path / "data" / "features_v2.parquet",
        output_path / "data_receipt.json",
    ]
    existing = [path for path in intended_paths if path.exists() or path.is_symlink()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output artifact(s): {paths}")


def _read_csv_zip(input_path: Path) -> tuple[pd.DataFrame, str]:
    try:
        with zipfile.ZipFile(input_path, "r") as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if members != ["athletes.csv"]:
                raise ValueError(
                    "input ZIP must contain exactly one file named athletes.csv; "
                    f"found {members!r}"
                )
            with archive.open("athletes.csv", "r") as csv_source:
                frame = pd.read_csv(
                    csv_source,
                    dtype={"athlete_id": "string", "gender": "string"},
                    keep_default_na=True,
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot read input ZIP {input_path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"athletes.csv is missing required columns: {missing!r}")
    if frame.columns.duplicated().any():
        raise ValueError("athletes.csv contains duplicate column names")
    return frame, "athletes.csv"


def _prepare(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw, member_name = _read_csv_zip(input_path)
    row_count = int(len(raw))

    athlete_id = raw["athlete_id"].astype("string").str.strip()
    athlete_id = athlete_id.mask(athlete_id.eq(""), pd.NA)
    gender = raw["gender"].astype("string").str.strip()
    id_missing = athlete_id.isna()
    id_duplicate = athlete_id.notna() & athlete_id.duplicated(keep=False)

    numeric = pd.DataFrame(index=raw.index)
    for column in NUMERIC_COLUMNS:
        numeric[column] = pd.to_numeric(raw[column], errors="coerce")
    finite = pd.DataFrame(
        {column: np.isfinite(numeric[column].to_numpy(dtype="float64")) for column in NUMERIC_COLUMNS},
        index=raw.index,
    )
    numeric_valid = finite.all(axis=1)
    invalid_numeric = ~numeric_valid
    sentinel_numeric = finite & numeric.eq(SENTINEL)
    sentinel_rows = sentinel_numeric.any(axis=1)

    gender_valid = gender.isin(["Male", "Female"])
    age_valid = finite["age"] & numeric["age"].ge(18)
    height_valid = finite["height"] & numeric["height"].gt(48) & numeric["height"].lt(96)
    weight_valid = finite["weight"] & numeric["weight"].gt(0) & numeric["weight"].lt(1500)
    lifts_positive = finite[LIFT_COLUMNS].all(axis=1) & numeric[LIFT_COLUMNS].gt(0).all(axis=1)
    under_caps_by_lift = pd.DataFrame(
        {column: numeric[column].le(cap) for column, cap in LIFT_CAPS.items()},
        index=raw.index,
    )
    over_caps_by_lift = pd.DataFrame(
        {column: numeric[column].gt(cap) for column, cap in LIFT_CAPS.items()},
        index=raw.index,
    )
    lifts_under_caps = finite[LIFT_COLUMNS].all(axis=1) & under_caps_by_lift.all(axis=1)

    # Every row sharing a duplicate nonblank ID is excluded; no arbitrary
    # first-row winner is selected. Reason counts below are independent and
    # may overlap, while excluded_row_count is the exact union.
    eligible = (
        athlete_id.notna()
        & ~id_duplicate
        & numeric_valid
        & ~sentinel_rows
        & gender_valid
        & age_valid
        & height_valid
        & weight_valid
        & lifts_positive
        & lifts_under_caps
    )

    clean = pd.DataFrame({"athlete_id": athlete_id.loc[eligible].astype(str)})
    clean["gender"] = gender.loc[eligible].astype(str)
    for column in NUMERIC_COLUMNS:
        clean[column] = numeric.loc[eligible, column].to_numpy(dtype="float64")
    clean = clean.sort_values("athlete_id", kind="mergesort").reset_index(drop=True)
    clean["total_lift"] = clean[LIFT_COLUMNS].sum(axis=1)

    train_ids, holdout_ids = train_test_split(
        clean["athlete_id"],
        train_size=0.70,
        test_size=0.30,
        random_state=42,
        stratify=clean["gender"],
    )
    gender_by_id = clean.set_index("athlete_id")["gender"]
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.50,
        random_state=42,
        stratify=gender_by_id.loc[holdout_ids.to_list()],
    )
    split_by_id = {
        **dict.fromkeys(train_ids.to_list(), "train"),
        **dict.fromkeys(validation_ids.to_list(), "validation"),
        **dict.fromkeys(test_ids.to_list(), "test"),
    }
    clean["split"] = clean["athlete_id"].map(split_by_id)

    cohort = clean[["athlete_id", "total_lift", "split", "gender"]].copy()
    cohort = cohort.sort_values("athlete_id", kind="mergesort").reset_index(drop=True)

    event_timestamp = pd.Series(EVENT_TIMESTAMP, index=clean.index, dtype="datetime64[ns, UTC]")
    features = pd.DataFrame(
        {
            "athlete_id": clean["athlete_id"],
            "event_timestamp": event_timestamp,
            "age": clean["age"],
            "height": clean["height"],
            "weight": clean["weight"],
            "is_female": clean["gender"].eq("Female"),
        }
    )
    features_v1 = features[["athlete_id", "event_timestamp", *V1_FEATURES]].copy()
    features_v2 = features.copy()
    features_v2["age_squared"] = features_v2["age"] ** 2
    features_v2["height_squared"] = features_v2["height"] ** 2
    features_v2["weight_squared"] = features_v2["weight"] ** 2
    features_v2["age_x_weight"] = features_v2["age"] * features_v2["weight"]
    features_v2["age_x_height"] = features_v2["age"] * features_v2["height"]
    features_v2["weight_x_height"] = features_v2["weight"] * features_v2["height"]
    features_v2 = features_v2[["athlete_id", "event_timestamp", *V2_FEATURES]]

    group_counts = athlete_id.dropna().value_counts()
    duplicate_id_groups = int(group_counts.gt(1).sum())
    criteria_counts = {
        "missing_athlete_id_rows": int(id_missing.sum()),
        "duplicate_athlete_id_rows": int(id_duplicate.sum()),
        "duplicate_athlete_id_groups": duplicate_id_groups,
        "invalid_numeric_rows": int(invalid_numeric.sum()),
        "sentinel_numeric_rows": int(sentinel_rows.sum()),
        "unsupported_gender_rows": int((~gender_valid).sum()),
        "age_below_18_rows": int((finite["age"] & numeric["age"].lt(18)).sum()),
        "height_out_of_range_rows": int((finite["height"] & ~height_valid).sum()),
        "weight_out_of_range_rows": int((finite["weight"] & ~weight_valid).sum()),
        "nonpositive_lift_rows": int(
            (finite[LIFT_COLUMNS].all(axis=1) & numeric[LIFT_COLUMNS].le(0).any(axis=1)).sum()
        ),
        "over_lift_cap_rows": int(
            (finite[LIFT_COLUMNS].all(axis=1) & over_caps_by_lift.any(axis=1)).sum()
        ),
    }
    split_sizes = {
        split: int(cohort["split"].eq(split).sum())
        for split in ("train", "validation", "test")
    }
    gender_counts = {
        gender_name: int(cohort["gender"].eq(gender_name).sum())
        for gender_name in ("Female", "Male")
    }
    receipt = {
        "source": {
            "archive_sha256": _sha256(input_path),
            "member": member_name,
            "row_count": row_count,
            "column_count": int(len(raw.columns)),
            "columns": [str(column) for column in raw.columns],
        },
        "target": {
            "column": "total_lift",
            "formula": "candj + snatch + deadlift + backsq",
            "convention": "study-defined preparation convention; not instructor-defined",
        },
        "cohort": {
            "row_count": int(len(cohort)),
            "excluded_row_count": int(row_count - len(cohort)),
            "gender_counts": gender_counts,
            "selection_rules": [
                "athlete_id is nonblank and unique; all rows in each duplicate-ID group are excluded",
                "all age, height, weight, and four lift fields parse as finite numeric values",
                "numeric sentinel 8388607.0 is excluded",
                "age >= 18",
                "48 < height < 96",
                "0 < weight < 1500",
                "gender is exactly Male or Female",
                "candj, snatch, deadlift, and backsq are each positive",
                "candj <= 395, snatch <= 496, deadlift <= 1105, backsq <= 1069",
                "no additional female-specific deadlift <= 636 cap is applied",
            ],
            "exclusion_counts": criteria_counts,
            "exclusion_counts_may_overlap": True,
            "female_deadlift_636_cap_applied": False,
        },
        "features": {
            "v1": V1_FEATURES,
            "v2": V2_FEATURES,
            "metadata_columns": ["athlete_id", "event_timestamp"],
            "unit_note": "source columns have no explicit unit metadata; raw numeric values are retained without conversion",
            "raw_target_contributors_ids_and_names_excluded_from_predictors": True,
        },
        "split": {
            "method": "two-stage stratified split: 70% train, then split remaining rows equally",
            "stratified_by": "gender",
            "seed": 42,
            "requested_shares": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "sizes": split_sizes,
        },
        "event_timestamp": {
            "column": "event_timestamp",
            "value": EVENT_TIMESTAMP_VALUE,
            "semantics": "synthetic UTC snapshot timestamp assigned to every row; not historical event or measurement time",
        },
        "outputs": {
            "cohort": "data/cohort.parquet",
            "features_v1": "data/features_v1.parquet",
            "features_v2": "data/features_v2.parquet",
        },
    }
    return cohort, features_v1, features_v2, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source ZIP containing athletes.csv")
    parser.add_argument("--output", required=True, type=Path, help="output directory (data/ is created inside)")
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        _ensure_outputs_absent(output_path)
        cohort, features_v1, features_v2, receipt = _prepare(input_path)
        data_path = output_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        cohort.to_parquet(data_path / "cohort.parquet", index=False)
        features_v1.to_parquet(data_path / "features_v1.parquet", index=False)
        features_v2.to_parquet(data_path / "features_v2.parquet", index=False)
        receipt_path = output_path / "data_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"prepare_data: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "cohort_rows": receipt["cohort"]["row_count"],
                "split_sizes": receipt["split"]["sizes"],
                "outputs": receipt["outputs"],
                "data_receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
