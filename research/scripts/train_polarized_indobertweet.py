import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "indolem/indobertweet-base-uncased"

train_file = "data/processed/polarized_train.csv"
validation_file = "data/processed/polarized_validation.csv"

train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)

train = train.sample(n=5000, random_state=42)
validation = validation.sample(n=1000, random_state=42)

train = train[["text_clean", "polarized_label"]]
validation = validation[["text_clean", "polarized_label"]]

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer berhasil dimuat.")

train_dataset = TextDataset(
    train["text_clean"],
    train["polarized_label"],
    tokenizer
)

validation_dataset = TextDataset(
    validation["text_clean"],
    validation["polarized_label"],
    tokenizer
)

print("=== DATASET ===")
print("Train     :", len(train_dataset))
print("Validation:", len(validation_dataset))

print("\n=== LOAD MODEL ===")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("Model berhasil dimuat.")

training_args = TrainingArguments(
    output_dir="results/polarized_indobertweet_weighted_5000",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none"
)

class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        class_weights = torch.tensor(
            [1.0, 5.0],
            device=logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights
        )

        loss = loss_function(
            logits,
            labels
        )

        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)

print("\n=== MULAI TRAINING ===")

trainer.train()

print("\n=== TRAINING SELESAI ===")