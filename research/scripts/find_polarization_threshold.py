import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# =========================
# CONFIG
# =========================

VALIDATION_FILE = (
    "data/processed/polarized_validation.csv"
)

MODEL_DIR = (
    "results/polarized_indobertweet_1000/final_model"
)


# =========================
# LOAD DATA
# =========================

print("=== LOAD VALIDATION SET ===")

df = pd.read_csv(
    VALIDATION_FILE
)

print(
    "Jumlah data:",
    len(df)
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
# GET PROBABILITIES
# =========================

print("\n=== PREDICTION ===")

probabilities = []

for i, text in enumerate(
    df["text"]
):

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

    probs = torch.softmax(
        output.logits,
        dim=1
    )[0]

    probabilities.append(
        float(probs[1])
    )

    if (i + 1) % 500 == 0:

        print(
            f"{i + 1}/{len(df)}"
        )


y_true = (
    df["polarized_label"]
    .astype(int)
    .to_numpy()
)

probabilities = np.array(
    probabilities
)


# =========================
# THRESHOLD SEARCH
# =========================

print("\n")
print("=" * 70)
print("POLARIZATION THRESHOLD SEARCH")
print("=" * 70)

best_threshold = None
best_f1 = -1

results = []

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# =========================
# BEST
# =========================

print("\n")
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"F1        : {best_f1:.4f}"
)


# =========================
# SAVE RESULT
# =========================

output = pd.DataFrame(
    results
)

output.to_csv(
    "results/polarization_threshold_search.csv",
    index=False
)

print(
    "\nOutput:"
)

print(
    "results/polarization_threshold_search.csv"
)