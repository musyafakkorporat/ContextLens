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
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# CONFIG
# =========================

TEST_FILE = (
    "data/processed/toxicity_test.csv"
)

MODEL_DIR = (
    "results/toxicity_indobertweet_5000/final_model"
)

OUTPUT_FILE = (
    "results/toxicity_testset_predictions_5000.csv"
)

THRESHOLD = 0.10


# =========================
# LOAD DATA
# =========================

print("=== LOAD TEST SET ===")

df = pd.read_csv(TEST_FILE)

print(
    "Jumlah data:",
    len(df)
)

print(
    "Kolom:",
    df.columns.tolist()
)


# =========================
# LOAD MODEL
# =========================

print("\n=== LOAD MODEL ===")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


# =========================
# PREDICTION
# =========================

def predict(text):

    encoding = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=128,
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
    )[0]

    probability = float(
        probabilities[1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    return prediction, probability


# =========================
# RUN
# =========================

print("\n=== PREDICTION ===")

predictions = []
probabilities = []

for i, text in enumerate(
    df["text"]
):

    prediction, probability = predict(
        text
    )

    predictions.append(
        prediction
    )

    probabilities.append(
        probability
    )

    if (i + 1) % 500 == 0:

        print(
            f"{i + 1}/{len(df)}"
        )


df["predicted_toxicity"] = (
    predictions
)

df["toxicity_probability"] = (
    probabilities
)


# =========================
# EVALUATION
# =========================

y_true = (
    df["toxicity_label"]
    .astype(int)
)

y_pred = (
    df["predicted_toxicity"]
    .astype(int)
)


print("\n")
print("=" * 60)
print("TOXICITY - INDO DISCOURSE TEST SET")
print("=" * 60)

print(
    "Model:",
    MODEL_DIR
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_true,
            y_pred
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall   :",
    round(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "F1       :",
    round(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# =========================
# DISTRIBUTION
# =========================

print("\n=== DISTRIBUTION ===")

print(
    "Actual:"
)

print(
    y_true.value_counts()
    .sort_index()
)

print(
    "\nPredicted:"
)

print(
    y_pred.value_counts()
    .sort_index()
)


# =========================
# SAVE
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nOutput:")

print(
    OUTPUT_FILE
)