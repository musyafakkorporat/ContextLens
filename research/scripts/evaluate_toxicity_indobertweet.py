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


MODEL_PATH = (
    "results/toxicity_indobertweet_1000/final_model"
)

TEST_FILE = "data/processed/toxicity_test.csv"


# =========================
# LOAD TEST DATA
# =========================

test = pd.read_csv(TEST_FILE)

test = test[
    ["text_clean", "toxicity_label"]
]


print("=== DATASET TEST ===")

print("Test:", len(test))

print("\nDistribusi label:")

print(
    test["toxicity_label"]
    .value_counts()
    .sort_index()
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
            labels.astype(int)
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

        encoding["labels"] = self.labels[index]

        return {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }


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

print("Model berhasil dimuat.")


# =========================
# DATASET
# =========================

test_dataset = TextDataset(
    test["text_clean"],
    test["toxicity_label"],
    tokenizer
)


# =========================
# PREDICTION
# =========================

trainer = Trainer(
    model=model
)


print("\n=== MULAI EVALUASI ===")

result = trainer.predict(
    test_dataset
)

probabilities = torch.softmax(
    torch.tensor(result.predictions),
    dim=1
).numpy()

prob_toxic = probabilities[:, 1]

THRESHOLD = 0.06

predictions = (
    prob_toxic >= THRESHOLD
).astype(int)

labels = result.label_ids


# =========================
# METRICS
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
# RESULT
# =========================

print("\n=== HASIL EVALUASI TOXICITY ===")

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
    "F1-score :",
    round(f1, 4)
)


print("\n=== CONFUSION MATRIX ===")

print(cm)


print("\n=== DISTRIBUSI PREDIKSI ===")

print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)


print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Tidak Toxic",
            "Toxic"
        ],
        zero_division=0
    )
)