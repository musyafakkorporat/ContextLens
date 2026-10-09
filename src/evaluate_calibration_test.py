import json
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    brier_score_loss,
    log_loss
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


# =========================
# LOAD
# =========================

with open(
    CALIBRATION_FILE,
    "r",
    encoding="utf-8"
) as file:
    calibration = json.load(file)


def load_model(path):

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.eval()

    return tokenizer, model


def get_logits(
    texts,
    tokenizer,
    model
):

    outputs = []

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

            result = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )

        outputs.append(
            result.logits.cpu()
        )

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r"
        )

    print()

    return torch.cat(
        outputs,
        dim=0
    )


def ece(
    probabilities,
    labels,
    bins=10
):

    edges = np.linspace(
        0,
        1,
        bins + 1
    )

    result = 0.0

    for i in range(bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == bins - 1:

            mask = (
                (probabilities >= lower) &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower) &
                (probabilities < upper)
            )

        if not np.any(mask):
            continue

        confidence = (
            probabilities[mask].mean()
        )

        accuracy = (
            labels[mask].mean()
        )

        result += (
            mask.mean() *
            abs(
                confidence -
                accuracy
            )
        )

    return result


def evaluate(
    name,
    model_path,
    test_file,
    label_column,
    temperature
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    tokenizer, model = load_model(
        model_path
    )

    data = pd.read_csv(
        test_file
    )

    data = data[
        data[label_column].notna()
    ].copy()

    labels = (
        data[label_column]
        .astype(int)
        .to_numpy()
    )

    logits = get_logits(
        data["text"].tolist(),
        tokenizer,
        model
    )

    raw_difference = (
        logits[:, 1] -
        logits[:, 0]
    )

    raw_probability = (
        torch.sigmoid(
            raw_difference
        )
        .numpy()
    )

    calibrated_probability = (
        torch.sigmoid(
            raw_difference /
            temperature
        )
        .numpy()
    )

    raw_brier = brier_score_loss(
        labels,
        raw_probability
    )

    calibrated_brier = brier_score_loss(
        labels,
        calibrated_probability
    )

    raw_logloss = log_loss(
        labels,
        np.column_stack([
            1 - raw_probability,
            raw_probability
        ])
    )

    calibrated_logloss = log_loss(
        labels,
        np.column_stack([
            1 - calibrated_probability,
            calibrated_probability
        ])
    )

    raw_ece = ece(
        raw_probability,
        labels
    )

    calibrated_ece = ece(
        calibrated_probability,
        labels
    )

    print(
        "Temperature:",
        round(temperature, 6)
    )

    print("\nRaw probability:")
    print(
        "Brier :",
        round(raw_brier, 6)
    )
    print(
        "LogLoss:",
        round(raw_logloss, 6)
    )
    print(
        "ECE   :",
        round(raw_ece, 6)
    )

    print("\nCalibrated probability:")
    print(
        "Brier :",
        round(calibrated_brier, 6)
    )
    print(
        "LogLoss:",
        round(calibrated_logloss, 6)
    )
    print(
        "ECE   :",
        round(calibrated_ece, 6)
    )


evaluate(
    "TOXICITY",
    TOXICITY_MODEL,
    TOXICITY_FILE,
    "toxicity_label",
    calibration["toxicity"]["temperature"]
)


evaluate(
    "POLARIZATION",
    POLARIZATION_MODEL,
    POLARIZATION_FILE,
    "polarized_label",
    calibration["polarization"]["temperature"]
)