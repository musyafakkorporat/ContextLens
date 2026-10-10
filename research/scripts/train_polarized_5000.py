import os

import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.utils.class_weight import compute_class_weight


# =========================
# CONFIG
# =========================

TRAIN_FILE = (
    "data/processed/polarized_train.csv"
)

VALIDATION_FILE = (
    "data/processed/polarized_validation.csv"
)

OUTPUT_DIR = (
    "results/polarized_indobertweet_5000"
)

MODEL_NAME = (
    "indolem/indobertweet-base-uncased"
)

TRAIN_SAMPLE = 5000
VALIDATION_SAMPLE = 1000

MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 2


# =========================
# LOAD DATA
# =========================

print("=== LOAD DATA ===")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train = train[
    ["text_clean", "polarized_label"]
]

validation = validation[
    ["text_clean", "polarized_label"]
]


# =========================
# SAMPLE DATA
# =========================

train = train.sample(
    n=TRAIN_SAMPLE,
    random_state=42
)

validation = validation.sample(
    n=VALIDATION_SAMPLE,
    random_state=42
)

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(validation)
)


# =========================
# CLASS DISTRIBUTION
# =========================

print("\n=== CLASS DISTRIBUTION ===")

print(
    "Train:"
)

print(
    train["polarized_label"]
    .value_counts()
    .sort_index()
)

print(
    "\nValidation:"
)

print(
    validation["polarized_label"]
    .value_counts()
    .sort_index()
)


# =========================
# TOKENIZER
# =========================

print("\n=== LOAD TOKENIZER ===")

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)

print(
    "Tokenizer berhasil dimuat."
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
            labels.astype(int).tolist()
        )

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        encoding["labels"] = (
            self.labels[index]
        )

        return {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }


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


# =========================
# CLASS WEIGHT
# =========================

print("\n=== CLASS WEIGHT ===")

classes = (
    train["polarized_label"]
    .unique()
)

classes.sort()

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train["polarized_label"]
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
)

print(
    "Class weights:",
    class_weights
)


# =========================
# MODEL
# =========================

print("\n=== LOAD MODEL ===")

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
)

print(
    "Model berhasil dimuat."
)


# =========================
# CUSTOM TRAINER
# =========================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(
                logits.device
            )
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="no",

    logging_strategy="steps",

    logging_steps=100,

    report_to="none"
)


# =========================
# TRAINER
# =========================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset
)


# =========================
# TRAIN
# =========================

print("\n=== MULAI TRAINING ===")

trainer.train()


# =========================
# SAVE MODEL
# =========================

FINAL_MODEL_DIR = (
    f"{OUTPUT_DIR}/final_model"
)

os.makedirs(
    FINAL_MODEL_DIR,
    exist_ok=True
)

print("\n=== SAVE MODEL ===")

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)

print(
    "Model disimpan di:",
    FINAL_MODEL_DIR
)