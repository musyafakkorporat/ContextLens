import os

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import accuracy_score


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

TOXICITY_THRESHOLD = 0.10
POLARIZATION_THRESHOLD = 0.50


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
# BATCH PREDICTION
# =========================

def predict_probability(
    texts,
    tokenizer,
    model
):

    all_probabilities = []

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

        probabilities = torch.softmax(
            output.logits,
            dim=1
        ).cpu().numpy()

        all_probabilities.append(
            probabilities
        )

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r"
        )

    print()

    return np.vstack(
        all_probabilities
    )


# =========================
# DISTANCE ANALYSIS
# =========================

def analyze_distance(
    probabilities,
    labels,
    threshold,
    model_name
):

    p1 = probabilities[:, 1]

    predictions = (
        p1 >= threshold
    ).astype(int)

    distance = np.abs(
        p1 - threshold
    )

    df = pd.DataFrame({
        "probability": p1,
        "label": labels,
        "prediction": predictions,
        "distance": distance
    })

    df["correct"] = (
        df["label"] ==
        df["prediction"]
    )

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        "Overall accuracy:",
        round(
            accuracy_score(
                labels,
                predictions
            ),
            4
        )
    )

    print(
        "Threshold:",
        threshold
    )

    print("\nDistance analysis:")

    bins = [
        0.00,
        0.02,
        0.05,
        0.10,
        0.20,
        1.00
    ]

    labels_text = [
        "0.00 - 0.02",
        "0.02 - 0.05",
        "0.05 - 0.10",
        "0.10 - 0.20",
        "0.20+"
    ]

    for i in range(
        len(bins) - 1
    ):

        lower = bins[i]
        upper = bins[i + 1]

        if i == len(bins) - 2:

            subset = df[
                (df["distance"] >= lower) &
                (df["distance"] <= upper)
            ]

        else:

            subset = df[
                (df["distance"] >= lower) &
                (df["distance"] < upper)
            ]

        if len(subset) == 0:
            continue

        accuracy = subset["correct"].mean()
        error_rate = 1 - accuracy

        print(
            f"{labels_text[i]:>12} | "
            f"n={len(subset):5d} | "
            f"accuracy={accuracy:.4f} | "
            f"error={error_rate:.4f}"
        )

    return df


# =========================
# TOXICITY
# =========================

print("=== LOAD TOXICITY ===")

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

print(
    "Toxicity validation:",
    len(toxicity_data)
)

toxicity_probabilities = predict_probability(
    toxicity_data["text"].tolist(),
    toxicity_tokenizer,
    toxicity_model
)

toxicity_result = analyze_distance(
    toxicity_probabilities,
    toxicity_data["toxicity_label"].astype(int).to_numpy(),
    TOXICITY_THRESHOLD,
    "TOXICITY"
)


# =========================
# POLARIZATION
# =========================

print("\n=== LOAD POLARIZATION ===")

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

print(
    "Polarization validation:",
    len(polarization_data)
)

polarization_probabilities = predict_probability(
    polarization_data["text"].tolist(),
    polarization_tokenizer,
    polarization_model
)

polarization_result = analyze_distance(
    polarization_probabilities,
    polarization_data["polarized_label"].astype(int).to_numpy(),
    POLARIZATION_THRESHOLD,
    "POLARIZATION"
)


# =========================
# SAVE
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

toxicity_result.to_csv(
    "results/toxicity_uncertainty_analysis.csv",
    index=False
)

polarization_result.to_csv(
    "results/polarization_uncertainty_analysis.csv",
    index=False
)

print("\nAnalisis selesai.")
print(
    "Toxicity:",
    "results/toxicity_uncertainty_analysis.csv"
)
print(
    "Polarization:",
    "results/polarization_uncertainty_analysis.csv"
)