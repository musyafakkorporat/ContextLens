from pathlib import Path
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_FILE = Path("data/processed/dataset_clean.csv")
OUTPUT_DIR = Path("data/processed_leakage_free_v2")

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def normalize_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip().casefold()


def build_unique_dataset(df, label_column):
    records = []

    conflict_groups = 0
    unlabeled_groups = 0
    invalid_label_groups = 0

    grouped = df.groupby(
        "_normalized_text",
        sort=False
    )

    for normalized_text, group in grouped:

        values = pd.to_numeric(
            group[label_column],
            errors="coerce"
        ).dropna()

        labels = set(
            values.astype(float).tolist()
        )

        if labels == {0.0, 1.0}:
            conflict_groups += 1
            continue

        if not labels:
            unlabeled_groups += 1
            continue

        if labels not in ({0.0}, {1.0}):
            invalid_label_groups += 1
            continue

        row = group.iloc[0]

        record = {
            "text_id": row["text_id"],
            "text": row["text"],
            label_column: int(next(iter(labels))),
        }

        if "topic" in group.columns:
            record["topic"] = row["topic"]

        records.append(record)

    unique_df = pd.DataFrame(records)

    summary = {
        "unique_text_groups": int(len(grouped)),
        "usable_unique_texts": int(len(unique_df)),
        "conflicting_label_groups": conflict_groups,
        "unlabeled_groups": unlabeled_groups,
        "invalid_label_groups": invalid_label_groups,
        "class_counts": {
            str(int(label)): int(count)
            for label, count in (
                unique_df[label_column].value_counts().items()
            )
        },
    }

    return unique_df, summary


def split_dataset(df, label_column, task_name):
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=df[label_column],
    )

    validation_fraction = (
        VALIDATION_RATIO /
        (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=train_val[label_column],
    )

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    # Pastikan satu teks tidak pernah muncul di dua split.
    normalized_sets = {}

    for split_name, split_df in splits.items():
        normalized_sets[split_name] = set(
            split_df["text"].map(normalize_text)
        )

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = (
            normalized_sets[first] &
            normalized_sets[second]
        )

        if overlap:
            raise RuntimeError(
                f"{task_name}: ditemukan {len(overlap)} "
                f"teks yang tumpang tindih antara "
                f"{first} dan {second}."
            )

    for split_name, split_df in splits.items():
        output_file = (
            OUTPUT_DIR /
            f"{task_name}_{split_name}.csv"
        )

        split_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        print(
            f"{task_name}_{split_name}: "
            f"{len(split_df)} baris"
        )

        print(
            "Distribusi label:",
            split_df[label_column]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    return {
        "split_counts": {
            name: int(len(split_df))
            for name, split_df in splits.items()
        },
        "text_overlap_checks": "PASS",
        "random_state": RANDOM_STATE,
    }


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset sumber tidak ditemukan: {SOURCE_FILE}"
        )

    df = pd.read_csv(
        SOURCE_FILE,
        low_memory=False,
    )

    required_columns = {
        "text_id",
        "text",
        "toxicity_label",
        "polarized_label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom dataset tidak lengkap: {sorted(missing)}"
        )

    df["_normalized_text"] = (
        df["text"].map(normalize_text)
    )

    df = df[
        df["_normalized_text"] != ""
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "source": str(SOURCE_FILE),
        "source_rows_with_nonempty_text": int(len(df)),
        "normalization": (
            "collapse whitespace, strip, casefold"
        ),
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "random_state": RANDOM_STATE,
        "tasks": {},
    }

    for task_name, label_column in [
        ("toxicity", "toxicity_label"),
        ("polarized", "polarized_label"),
    ]:
        print(f"\n=== {task_name.upper()} ===")

        unique_df, task_summary = build_unique_dataset(
            df,
            label_column,
        )

        task_summary["splits"] = split_dataset(
            unique_df,
            label_column,
            task_name,
        )

        summary["tasks"][task_name] = task_summary

    summary_file = OUTPUT_DIR / "audit_summary.json"

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSelesai.")
    print("Dataset baru:", OUTPUT_DIR)
    print("Ringkasan:", summary_file)


if __name__ == "__main__":
    main()
