import os

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
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

MODEL_PATH = (
    "results/polarized_indobertweet_1000/final_model"
)

TEST_FILE = (
    "data/processed/polarized_test.csv"
)

OUTPUT_FILE = (
    "results/polarized_testset_predictions_1000.csv"
)

THRESHOLD = 0.50


# =========================
# LOAD DATA
# =========================

print("=== LOAD TEST SET ===")

test = pd.read_csv(TEST_FILE)

test = test[
    ["text_clean", "polarized_label"]
]

print(
    "Jumlah data:",
    len(test)
)


# =========================
# DATASET
# =========================

class TextDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer
    ):

        self.texts = texts.tolist()

        self.labels = (
            labels
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=128
        )

        encoding["labels"] = (
            self.labels[index]
        )

        return {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }


# =========================
# LOAD TOKENIZER
# =========================

print("\n=== LOAD TOKENIZER ===")

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)

print("Tokenizer berhasil dimuat.")


# =========================
# CREATE DATASET
# =========================

test_dataset = TextDataset(
    test["text_clean"],
    test["polarized_label"],
    tokenizer
)

print("\n=== DATASET TEST ===")

print(
    "Test:",
    len(test_dataset)
)


# =========================
# LOAD MODEL
# =========================

print("\n=== LOAD MODEL ===")

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_PATH,
        num_labels=2
    )
)

model.eval()

print("Model berhasil dimuat.")


# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model
)


# =========================
# PREDICTION
# =========================

print("\n=== MULAI EVALUASI ===")

result = trainer.predict(
    test_dataset
)

logits = result.predictions

probabilities = torch.softmax(
    torch.tensor(logits),
    dim=1
).numpy()

polarized_probability = (
    probabilities[:, 1]
)

predictions = (
    polarized_probability
    >= THRESHOLD
).astype(int)

labels = result.label_ids


# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    labels,
    predictions
)


# =========================
# PRINT RESULTS
# =========================

print("\n")
print("=" * 60)

print(
    "POLARIZATION - INDO DISCOURSE TEST SET"
)

print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1       :",
    round(f1, 4)
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Tidak Polarized",
            "Polarized"
        ],
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
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

print(
    "\nPredicted:"
)

print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)


# =========================
# SAVE RESULTS
# =========================

test["predicted_polarization"] = (
    predictions
)

test["polarization_probability"] = (
    polarized_probability
)

os.makedirs(
    "results",
    exist_ok=True
)

test.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nOutput:")

print(
    OUTPUT_FILE
)