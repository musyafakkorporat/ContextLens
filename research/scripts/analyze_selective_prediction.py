import os
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

TOXICITY_FILE = (
    "results/toxicity_calibration_validation.csv"
)

POLARIZATION_FILE = (
    "results/polarization_calibration_validation.csv"
)

RAW_TOXICITY_THRESHOLD = 0.10
POLARIZATION_THRESHOLD = 0.50

# Target error yang ingin dianalisis
TARGET_ERRORS = [
    0.10,
    0.15,
    0.20
]


# =========================
# HELPER
# =========================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrated_threshold(
    raw_threshold,
    temperature
):

    logit = np.log(
        raw_threshold /
        (1 - raw_threshold)
    )

    return sigmoid(
        logit / temperature
    )


def evaluate_selective(
    data,
    threshold,
    model_name
):

    data = data.copy()

    data["distance"] = (
        np.abs(
            data["calibrated_probability"] -
            threshold
        )
    )

    rows = []

    # Delta 0.00 sampai 0.50
    for delta in np.arange(
        0.00,
        0.51,
        0.01
    ):

        accepted = data[
            data["distance"] >= delta
        ]

        if len(accepted) == 0:
            continue

        coverage = (
            len(accepted) /
            len(data)
        )

        error_rate = (
            1 -
            accepted["correct"].mean()
            if "correct" in accepted.columns
            else np.nan
        )

        rows.append({
            "model": model_name,
            "delta": round(float(delta), 2),
            "accepted": len(accepted),
            "total": len(data),
            "coverage": coverage,
            "error_rate": error_rate
        })

    return pd.DataFrame(rows)


def prepare_data(
    path,
    threshold,
    temperature,
    label_column
):

    data = pd.read_csv(path)

    data["prediction"] = (
        data["calibrated_probability"] >=
        threshold
    ).astype(int)

    data["correct"] = (
        data["prediction"] ==
        data[label_column].astype(int)
    )

    return data


# =========================
# LOAD CALIBRATION
# =========================

print("=== LOAD CALIBRATION DATA ===")

toxicity = pd.read_csv(
    TOXICITY_FILE
)

polarization = pd.read_csv(
    POLARIZATION_FILE
)

# Temperature hasil validation
TOXICITY_TEMPERATURE = 1.434442
POLARIZATION_TEMPERATURE = 1.368066


# =========================
# CALIBRATED THRESHOLDS
# =========================

toxicity_threshold = calibrated_threshold(
    RAW_TOXICITY_THRESHOLD,
    TOXICITY_TEMPERATURE
)

polarization_threshold = calibrated_threshold(
    POLARIZATION_THRESHOLD,
    POLARIZATION_TEMPERATURE
)

print(
    "\nCalibrated toxicity threshold:",
    round(toxicity_threshold, 6)
)

print(
    "Calibrated polarization threshold:",
    round(polarization_threshold, 6)
)


# =========================
# PREPARE
# =========================

toxicity["prediction"] = (
    toxicity["calibrated_probability"] >=
    toxicity_threshold
).astype(int)

toxicity["correct"] = (
    toxicity["prediction"] ==
    toxicity["toxicity_label"].astype(int)
)

toxicity["distance"] = (
    np.abs(
        toxicity["calibrated_probability"] -
        toxicity_threshold
    )
)


polarization["prediction"] = (
    polarization["calibrated_probability"] >=
    polarization_threshold
).astype(int)

polarization["correct"] = (
    polarization["prediction"] ==
    polarization["polarized_label"].astype(int)
)

polarization["distance"] = (
    np.abs(
        polarization["calibrated_probability"] -
        polarization_threshold
    )
)


# =========================
# ANALYSIS
# =========================

toxicity_result = evaluate_selective(
    toxicity,
    toxicity_threshold,
    "toxicity"
)

polarization_result = evaluate_selective(
    polarization,
    polarization_threshold,
    "polarization"
)

all_results = pd.concat(
    [
        toxicity_result,
        polarization_result
    ],
    ignore_index=True
)


# =========================
# PRINT SUMMARY
# =========================

for model_name, result in [
    ("TOXICITY", toxicity_result),
    ("POLARIZATION", polarization_result)
]:

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        result[
            [
                "delta",
                "accepted",
                "coverage",
                "error_rate"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nTarget error analysis:")

    for target in TARGET_ERRORS:

        valid = result[
            result["error_rate"] <= target
        ]

        if len(valid) == 0:

            print(
                f"Error <= {target:.0%}: "
                "tidak ada cutoff"
            )

            continue

        # Pilih coverage terbesar
        best = valid.sort_values(
            "coverage",
            ascending=False
        ).iloc[0]

        print(
            f"Error <= {target:.0%}: "
            f"delta={best['delta']:.2f}, "
            f"coverage={best['coverage']:.2%}, "
            f"error={best['error_rate']:.2%}"
        )


# =========================
# SAVE
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

all_results.to_csv(
    "results/selective_prediction_validation.csv",
    index=False
)

print("\nHasil disimpan:")
print(
    "results/selective_prediction_validation.csv"
)