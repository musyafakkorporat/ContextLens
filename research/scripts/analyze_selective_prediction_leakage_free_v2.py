import json
from pathlib import Path

import pandas as pd


CONFIGS = [
    {
        "name": "toxicity",
        "file": "results/toxicity_calibration_validation_leakage_free_v2.csv",
    },
    {
        "name": "polarization",
        "file": "results/polarization_calibration_validation_leakage_free_v2.csv",
    },
]

TARGET_ERROR = 0.10
OUTPUT_CSV = Path(
    "results/selective_prediction_validation_leakage_free_v2.csv"
)
OUTPUT_JSON = Path(
    "results/selective_prediction_config_leakage_free_v2.json"
)


def analyze(config):
    path = Path(config["file"])

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)

    required = {"distance", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang di {path}: {sorted(missing)}")

    distance = pd.to_numeric(
        df["distance"], errors="raise"
    )
    correct = pd.to_numeric(
        df["correct"], errors="raise"
    )

    if len(df) == 0 or distance.isna().any() or correct.isna().any():
        raise ValueError(f"Data kosong atau tidak valid di {path}")

    if not correct.isin([0, 1]).all():
        raise ValueError(f"Kolom correct harus bernilai 0 atau 1: {path}")

    rows = []

    for step in range(51):
        delta = round(step * 0.01, 2)
        accepted = distance >= delta
        accepted_count = int(accepted.sum())

        if accepted_count == 0:
            continue

        accepted_error = float(
            1.0 - correct[accepted].mean()
        )

        rows.append({
            "model": config["name"],
            "delta": delta,
            "accepted": accepted_count,
            "uncertain": int((~accepted).sum()),
            "total": int(len(df)),
            "coverage": accepted_count / len(df),
            "error_rate": accepted_error,
            "meets_target": accepted_error <= TARGET_ERROR,
        })

    result = pd.DataFrame(rows)
    valid = result[result["meets_target"]]

    if valid.empty:
        print(
            f"\n{config['name']}: tidak ada delta pada grid "
            f"0.00-0.50 yang memenuhi error <= {TARGET_ERROR:.0%}."
        )
        return None, result

    best = valid.sort_values(
        ["coverage", "delta"],
        ascending=[False, True],
    ).iloc[0]

    selected = {
        "delta": float(best["delta"]),
        "coverage": float(best["coverage"]),
        "accepted": int(best["accepted"]),
        "uncertain": int(best["uncertain"]),
        "total": int(best["total"]),
        "accepted_error_rate": float(best["error_rate"]),
        "target_error_rate": TARGET_ERROR,
        "selection_data": "validation",
    }

    print(f"\n=== {config['name'].upper()} ===")
    print(f"Delta terpilih: {selected['delta']:.2f}")
    print(f"Coverage: {selected['coverage']:.2%}")
    print(f"Accepted: {selected['accepted']}/{selected['total']}")
    print(f"Uncertain: {selected['uncertain']}")
    print(
        "Error rate accepted: "
        f"{selected['accepted_error_rate']:.2%}"
    )

    return selected, result


def main():
    all_selected = {}
    all_rows = []

    for config in CONFIGS:
        selected, result = analyze(config)
        all_rows.append(result)

        if selected is not None:
            all_selected[config["name"]] = selected

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    pd.concat(all_rows, ignore_index=True).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    if len(all_selected) == len(CONFIGS):
        OUTPUT_JSON.write_text(
            json.dumps(all_selected, indent=2),
            encoding="utf-8",
        )
        print(f"\nKonfigurasi tersimpan: {OUTPUT_JSON}")
    else:
        print(
            "\nKonfigurasi lengkap belum disimpan karena "
            "setidaknya satu model tidak memenuhi target."
        )

    print(f"Tabel semua delta: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
