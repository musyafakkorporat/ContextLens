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

MODEL_PATH = "results/polarized_indobertweet_1000/final_model"
TEST_FILE = "data/processed/polarized_test.csv"


test = pd.read_csv(TEST_FILE)

test = test[["text_clean", "polarized_label"]]


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


print("=== LOAD TOKENIZER ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Tokenizer berhasil dimuat.")


test_dataset = TextDataset(
    test["text_clean"],
    test["polarized_label"],
    tokenizer
)

print("\n=== DATASET TEST ===")
print("Test:", len(test_dataset))


print("\n=== LOAD MODEL ===")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2
)

print("Model berhasil dimuat.")


trainer = Trainer(
    model=model
)


print("\n=== MULAI EVALUASI ===")

result = trainer.predict(test_dataset)

predictions = result.predictions.argmax(axis=1)
labels = result.label_ids


accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, zero_division=0)
recall = recall_score(labels, predictions, zero_division=0)
f1 = f1_score(labels, predictions, zero_division=0)

cm = confusion_matrix(labels, predictions)


print("\n=== HASIL EVALUASI INDoBERTweet ===")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        labels,
        predictions,
        target_names=["Tidak Polarized", "Polarized"],
        zero_division=0
    )
)