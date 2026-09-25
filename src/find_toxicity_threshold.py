import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


MODEL_PATH = (
    "results/toxicity_indobertweet_1000/final_model"
)

VALIDATION_FILE = (
    "data/processed/toxicity_validation.csv"
)


# =========================
# DATASET
# =========================

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts.tolist()
        self.labels = labels.astype(int).tolist()
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

        encoding["labels"] = self.labels[index]

        return {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }


# =========================
# LOAD VALIDATION
# =========================

validation = pd.read_csv(
    VALIDATION_FILE
)

validation = validation[
    ["text_clean", "toxicity_label"]
].reset_index(drop=True)


print("=== VALIDATION ===")
print("Jumlah:", len(validation))

print(
    validation["toxicity_label"]
    .value_counts()
    .sort_index()
)


# =========================
# LOAD MODEL
# =========================

print("\n=== LOAD MODEL ===")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# =========================
# DATASET
# =========================

dataset = TextDataset(
    validation["text_clean"],
    validation["toxicity_label"],
    tokenizer
)


# =========================
# PREDICTION
# =========================

print("\n=== PREDICTION ===")

trainer = Trainer(
    model=model
)

result = trainer.predict(dataset)

probabilities = torch.softmax(
    torch.tensor(result.predictions),
    dim=1
).numpy()

prob_toxic = probabilities[:, 1]

labels = validation[
    "toxicity_label"
].astype(int).values


# =========================
# SEARCH THRESHOLD
# =========================

print("\n=== CEK THRESHOLD ===")

best_threshold = 0
best_f1 = 0

for threshold in np.arange(
    0.05,
    0.51,
    0.01
):

    predictions = (
        prob_toxic >= threshold
    ).astype(int)

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

    print(
        f"Threshold {threshold:.2f} "
        f"| Precision {precision:.4f} "
        f"| Recall {recall:.4f} "
        f"| F1 {f1:.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# =========================
# BEST RESULT
# =========================

print("\n==============================")
print("THRESHOLD TERBAIK")
print("==============================")

print(
    "Threshold:",
    round(best_threshold, 2)
)

print(
    "F1:",
    round(best_f1, 4)
)

best_predictions = (
    prob_toxic >= best_threshold
).astype(int)

print(
    "Jumlah prediksi Toxic:",
    int(best_predictions.sum())
)