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
    "data/processed/toxicity_validation.csv"
)

POLARIZATION_FILE = (
    "data/processed/polarized_validation.csv"
)

MAX_LENGTH = 128
BATCH_SIZE = 32

OUTPUT_DIR = "results"

CALIBRATION_BINS = 10


# =========================
# LOAD MODEL
# =========================

def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


# =========================
# GET LOGITS
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

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoding = tokenizer(
            batch_texts,
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
# TEMPERATURE SCALING
# =========================

def fit_temperature(
    logits,
    labels
):

    logits = logits.float()
    labels = labels.long()

    # Untuk binary classification,
    # gunakan selisih logit class 1 dan class 0.
    logit_difference = (
        logits[:, 1] -
        logits[:, 0]
    )

    log_temperature = torch.nn.Parameter(
        torch.zeros(1)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=100
    )

    loss_function = (
        torch.nn.BCEWithLogitsLoss()
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        scaled_logits = (
            logit_difference /
            temperature
        )

        loss = loss_function(
            scaled_logits,
            labels.float()
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    temperature = float(
        torch.exp(
            log_temperature
        ).detach()
    )

    return temperature


# =========================
# CALIBRATED PROBABILITY
# =========================

def calibrated_probability(
    logits,
    temperature
):

    difference = (
        logits[:, 1] -
        logits[:, 0]
    )

    probability = torch.sigmoid(
        difference /
        temperature
    )

    return probability.numpy()


# =========================
# EXPECTED CALIBRATION ERROR
# =========================

def expected_calibration_error(
    probabilities,
    labels,
    bins=10
):

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    ece = 0.0

    for i in range(bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

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

        bin_probability = (
            probabilities[mask].mean()
        )

        bin_accuracy = (
            labels[mask].mean()
        )

        bin_weight = (
            mask.mean()
        )

        ece += (
            bin_weight *
            abs(
                bin_accuracy -
                bin_probability
            )
        )

    return float(ece)


# =========================
# ANALYZE MODEL
# =========================

def analyze_model(
    model_name,
    model_path,
    validation_file,
    label_column
):

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    tokenizer, model = load_model(
        model_path
    )

    data = pd.read_csv(
        validation_file
    )

    data = data[
        data[label_column].notna()
    ].copy()

    print(
        "Validation data:",
        len(data)
    )

    logits = get_logits(
        data["text"].tolist(),
        tokenizer,
        model
    )

    labels = (
        data[label_column]
        .astype(int)
        .to_numpy()
    )

    # -------------------------
    # RAW PROBABILITY
    # -------------------------

    raw_probability = torch.softmax(
        logits,
        dim=1
    )[:, 1].numpy()

    # -------------------------
    # FIT TEMPERATURE
    # -------------------------

    temperature = fit_temperature(
        logits,
        torch.tensor(labels)
    )

    print(
        "Temperature:",
        round(temperature, 6)
    )

    # -------------------------
    # CALIBRATED PROBABILITY
    # -------------------------

    calibrated = calibrated_probability(
        logits,
        temperature
    )

    # -------------------------
    # METRICS
    # -------------------------

    raw_brier = brier_score_loss(
        labels,
        raw_probability
    )

    calibrated_brier = brier_score_loss(
        labels,
        calibrated
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
            1 - calibrated,
            calibrated
        ])
    )

    raw_ece = expected_calibration_error(
        raw_probability,
        labels,
        CALIBRATION_BINS
    )

    calibrated_ece = expected_calibration_error(
        calibrated,
        labels,
        CALIBRATION_BINS
    )

    print("\nRaw probability:")
    print(
        "Brier:",
        round(raw_brier, 6)
    )
    print(
        "LogLoss:",
        round(raw_logloss, 6)
    )
    print(
        "ECE:",
        round(raw_ece, 6)
    )

    print("\nCalibrated probability:")
    print(
        "Brier:",
        round(calibrated_brier, 6)
    )
    print(
        "LogLoss:",
        round(calibrated_logloss, 6)
    )
    print(
        "ECE:",
        round(calibrated_ece, 6)
    )

    # -------------------------
    # SAVE RESULTS
    # -------------------------

    result = data[
        [
            "text_id",
            "text",
            label_column
        ]
    ].copy()

    result["raw_probability"] = (
        raw_probability
    )

    result["calibrated_probability"] = (
        calibrated
    )

    result.to_csv(
        f"{OUTPUT_DIR}/"
        f"{model_name.lower()}_calibration_validation.csv",
        index=False
    )

    return {
        "temperature": temperature,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_logloss": raw_logloss,
        "calibrated_logloss": calibrated_logloss,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece
    }


# =========================
# RUN
# =========================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = {}

results["toxicity"] = analyze_model(
    "toxicity",
    TOXICITY_MODEL,
    TOXICITY_FILE,
    "toxicity_label"
)

results["polarization"] = analyze_model(
    "polarization",
    POLARIZATION_MODEL,
    POLARIZATION_FILE,
    "polarized_label"
)


# =========================
# SAVE TEMPERATURES
# =========================

with open(
    f"{OUTPUT_DIR}/calibration_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

print("\n")
print("=" * 70)
print("CALIBRATION SELESAI")
print("=" * 70)

print(
    "Hasil:",
    "results/calibration_results.json"
)