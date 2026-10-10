import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


MODEL_NAME = "indolem/indobertweet-base-uncased"

TRAIN_FILE = "data/processed/toxicity_train.csv"
VALIDATION_FILE = "data/processed/toxicity_validation.csv"

OUTPUT_DIR = "results/toxicity_indobertweet_1000"


# =========================
# LOAD DATA
# =========================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)

train = train.sample(
    n=1000,
    random_state=42
).reset_index(drop=True)

validation = validation.sample(
    n=1000,
    random_state=42
).reset_index(drop=True)

train = train[[
    "text_clean",
    "toxicity_label"
]]

validation = validation[[
    "text_clean",
    "toxicity_label"
]]


print("=== DATA ===")

print("Train:")
print(
    train["toxicity_label"]
    .value_counts()
    .sort_index()
)

print("\nValidation:")
print(
    validation["toxicity_label"]
    .value_counts()
    .sort_index()
)


# =========================
# DATASET
# =========================

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

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
# TOKENIZER
# =========================

print("\n=== LOAD TOKENIZER ===")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

train_dataset = TextDataset(
    train["text_clean"],
    train["toxicity_label"],
    tokenizer
)

validation_dataset = TextDataset(
    validation["text_clean"],
    validation["toxicity_label"],
    tokenizer
)

print("Dataset siap.")


# =========================
# MODEL
# =========================

print("\n=== LOAD MODEL ===")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("Model berhasil dimuat.")


# =========================
# CLASS WEIGHT
# =========================

label_counts = (
    train["toxicity_label"]
    .value_counts()
)

class_weight_0 = (
    len(train)
    / (2 * label_counts[0])
)

class_weight_1 = (
    len(train)
    / (2 * label_counts[1])
)

class_weights = torch.tensor(
    [
        class_weight_0,
        class_weight_1
    ],
    dtype=torch.float
)

print("\n=== CLASS WEIGHT ===")

print(
    "Class 0:",
    round(class_weight_0, 4)
)

print(
    "Class 1:",
    round(class_weight_1, 4)
)


# =========================
# CUSTOM LOSS
# =========================

def weighted_loss(
    outputs,
    labels,
    num_items_in_batch=None
):

    weights = class_weights.to(
        outputs.logits.device
    )

    loss_function = (
        torch.nn.CrossEntropyLoss(
            weight=weights
        )
    )

    loss = loss_function(
        outputs.logits,
        labels
    )

    return loss


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.argmax(
        axis=1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="no",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50,

    report_to="none",

    seed=42,

    data_seed=42
)


# =========================
# TRAINER
# =========================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics,

    compute_loss_func=weighted_loss
)


# =========================
# TRAIN
# =========================

print("\n=== MULAI TRAINING ===")

trainer.train()


# =========================
# SAVE FINAL MODEL
# =========================

print("\n=== SIMPAN MODEL FINAL ===")

FINAL_MODEL_PATH = (
    OUTPUT_DIR + "/final_model"
)

trainer.save_model(
    FINAL_MODEL_PATH
)

tokenizer.save_pretrained(
    FINAL_MODEL_PATH
)

print(
    "Model berhasil disimpan:",
    FINAL_MODEL_PATH
)


print("\n=== TRAINING SELESAI ===")