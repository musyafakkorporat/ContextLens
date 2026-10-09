import json
import os

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# CONFIG
# =========================

TOXICITY_MODEL = (
    "results/toxicity_indobertweet_5000/final_model"
)

POLARIZATION_MODEL = (
    "results/polarized_indobertweet_5000/final_model"
)

TOXICITY_FILE = (
    "data/processed/toxicity_test.csv"
)

POLARIZATION_FILE = (
    "data/processed/polarized_test.csv"
)

CALIBRATION_FILE = (
    "results/calibration_results.json"
)

MAX_LENGTH = 128
BATCH_SIZE = 32

RAW_TOXICITY_THRESHOLD = 0.10
RAW_POLARIZATION_THRESHOLD = 0.50

TOXICITY_DELTA = 0.04
POLARIZATION_DELTA = 0.28


# =========================
# LOAD MODEL
# =========================

def load_model(path):

    tokenizer = AutoTokenizer.from_pretrained(
        path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        path
    )

    model.eval()

    return tokenizer, model


# =========================
# LOGITS
# =========================

def get_logits(
    texts,
    tokenizer,
    model
):

    all_logits = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        encoding = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )

        all_logits.append(
            output.logits.cpu()
        )

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r"
        )

    print()

    return torch.cat(
        all_logits,
        dim=0
    )


# =========================
# CALIBRATION
# =========================

def calibrated_probability(
    logits,
    temperature
):

    difference = (
        logits[:, 1] -
        logits[:, 0]
    )

    return torch.sigmoid(
        difference / temperature
    ).numpy()


def calibrated_threshold(
    raw_threshold,
    temperature
):

    raw_logit = np.log(
        raw_threshold /
        (1 - raw_threshold)
    )

    calibrated_logit = (
        raw_logit /
        temperature
    )

    return 1 / (
        1 +
        np.exp(-calibrated_logit)
    )


# =========================
# SELECTIVE EVALUATION
# =========================

def evaluate_selective(
    name,
    data,
    probability,
    label_column,
    calibrated_threshold_value,
    delta
):

    labels = (
        data[label_column]
        .astype(int)
        .to_numpy()
    )

    predictions = (
        probability >=
        calibrated_threshold_value
    ).astype(int)

    distance = np.abs(
        probability -
        calibrated_threshold_value
    )

    uncertain = (
        distance < delta
    )

    accepted = ~uncertain

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Calibrated threshold:",
        round(
            calibrated_threshold_value,
            6
        )
    )

    print(
        "Delta:",
        delta
    )

    print(
        "Total:",
        len(data)
    )

    print(
        "Accepted:",
        int(accepted.sum())
    )

    print(
        "Uncertain:",
        int(uncertain.sum())
    )

    coverage = (
        accepted.mean()
    )

    print(
        "Coverage:",
        round(
            coverage,
            4
        )
    )

    if accepted.sum() > 0:

        accepted_labels = labels[
            accepted
        ]

        accepted_predictions = predictions[
            accepted
        ]

        accepted_accuracy = (
            accuracy_score(
                accepted_labels,
                accepted_predictions
            )
        )

        accepted_error = (
            1 -
            accepted_accuracy
        )

        print(
            "Accepted accuracy:",
            round(
                accepted_accuracy,
                4
            )
        )

        print(
            "Accepted error:",
            round(
                accepted_error,
                4
            )
        )

        print(
            "Accepted precision:",
            round(
                precision_score(
                    accepted_labels,
                    accepted_predictions,
                    zero_division=0
                ),
                4
            )
        )

        print(
            "Accepted recall:",
            round(
                recall_score(
                    accepted_labels,
                    accepted_predictions,
                    zero_division=0
                ),
                4
            )
        )

        print(
            "Accepted F1:",
            round(
                f1_score(
                    accepted_labels,
                    accepted_predictions,
                    zero_division=0
                ),
                4
            )
        )

    else:

        accepted_error = None

    result = data[
        [
            "text_id",
            "text",
            label_column
        ]
    ].copy()

    result["calibrated_probability"] = (
        probability
    )

    result["prediction"] = (
        predictions
    )

    result["distance"] = (
        distance
    )

    result["uncertain"] = (
        uncertain
    )

    result["accepted"] = (
        accepted
    )

    return result


# =========================
# LOAD CALIBRATION
# =========================

with open(
    CALIBRATION_FILE,
    "r",
    encoding="utf-8"
) as file:

    calibration = json.load(
        file
    )


toxicity_temperature = (
    calibration["toxicity"]["temperature"]
)

polarization_temperature = (
    calibration["polarization"]["temperature"]
)


# =========================
# TOXICITY
# =========================

print("=== TOXICITY ===")

toxicity_tokenizer, toxicity_model = (
    load_model(
        TOXICITY_MODEL
    )
)

toxicity_data = pd.read_csv(
    TOXICITY_FILE
)

toxicity_data = toxicity_data[
    toxicity_data["toxicity_label"].notna()
].copy()

toxicity_logits = get_logits(
    toxicity_data["text"].tolist(),
    toxicity_tokenizer,
    toxicity_model
)

toxicity_probability = calibrated_probability(
    toxicity_logits,
    toxicity_temperature
)

toxicity_calibrated_threshold = (
    calibrated_threshold(
        RAW_TOXICITY_THRESHOLD,
        toxicity_temperature
    )
)

toxicity_result = evaluate_selective(
    "TOXICITY",
    toxicity_data,
    toxicity_probability,
    "toxicity_label",
    toxicity_calibrated_threshold,
    TOXICITY_DELTA
)


# =========================
# POLARIZATION
# =========================

print("\n=== POLARIZATION ===")

polarization_tokenizer, polarization_model = (
    load_model(
        POLARIZATION_MODEL
    )
)

polarization_data = pd.read_csv(
    POLARIZATION_FILE
)

polarization_data = polarization_data[
    polarization_data["polarized_label"].notna()
].copy()

polarization_logits = get_logits(
    polarization_data["text"].tolist(),
    polarization_tokenizer,
    polarization_model
)

polarization_probability = calibrated_probability(
    polarization_logits,
    polarization_temperature
)

polarization_calibrated_threshold = (
    calibrated_threshold(
        RAW_POLARIZATION_THRESHOLD,
        polarization_temperature
    )
)

polarization_result = evaluate_selective(
    "POLARIZATION",
    polarization_data,
    polarization_probability,
    "polarized_label",
    polarization_calibrated_threshold,
    POLARIZATION_DELTA
)


# =========================
# SAVE
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

toxicity_result.to_csv(
    "results/toxicity_selective_test.csv",
    index=False
)

polarization_result.to_csv(
    "results/polarization_selective_test.csv",
    index=False
)

print("\n")
print("=" * 70)
print("SELECTIVE PREDICTION TEST SELESAI")
print("=" * 70)

print(
    "Toxicity:",
    "results/toxicity_selective_test.csv"
)

print(
    "Polarization:",
    "results/polarization_selective_test.csv"
)