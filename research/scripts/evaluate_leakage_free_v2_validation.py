from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


CONFIGS = [
    {
        "name": "TOXICITY",
        "model": "results/toxicity_indobertweet_leakage_free_v2/final_model",
        "data": "data/processed_leakage_free_v2/toxicity_validation.csv",
        "label": "toxicity_label",
        "output": "results/toxicity_validation_predictions_leakage_free_v2.csv",
    },
    {
        "name": "POLARIZATION",
        "model": "results/polarized_indobertweet_leakage_free_v2/final_model",
        "data": "data/processed_leakage_free_v2/polarized_validation.csv",
        "label": "polarized_label",
        "output": "results/polarized_validation_predictions_leakage_free_v2.csv",
    },
]

MAX_LENGTH = 128
BATCH_SIZE = 32


def calculate_ece(probabilities, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if mask.any():
            confidence = probabilities[mask].mean()
            accuracy = labels[mask].mean()
            result += mask.mean() * abs(confidence - accuracy)

    return float(result)


def evaluate(config):
    print(f"\n{'=' * 60}")
    print(config["name"])
    print("=" * 60)

    model_path = Path(config["model"])
    data_path = Path(config["data"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    data = pd.read_csv(data_path)
    data = data.dropna(subset=["text", config["label"]]).copy()

    labels = data[config["label"]].astype(int).to_numpy()
    texts = data["text"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )
    model.eval()

    all_differences = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        encoding = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoding)

        differences = (
            output.logits[:, 1] - output.logits[:, 0]
        ).cpu().numpy()

        all_differences.extend(differences.tolist())

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r",
        )

    print()

    differences = np.asarray(all_differences)
    probabilities = 1 / (1 + np.exp(-np.clip(differences, -50, 50)))
    predictions = (differences >= 0).astype(int)

    print("Jumlah validation:", len(labels))
    print("Accuracy:", round(accuracy_score(labels, predictions), 4))
    print(
        "Precision:",
        round(precision_score(labels, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(labels, predictions, zero_division=0), 4),
    )
    print("F1:", round(f1_score(labels, predictions, zero_division=0), 4))
    print("Brier:", round(brier_score_loss(labels, probabilities), 6))
    print("ECE:", round(calculate_ece(probabilities, labels), 6))
    print("Confusion matrix:\n", confusion_matrix(labels, predictions))

    result = pd.DataFrame({
        "text_id": data["text_id"].astype(str).to_numpy(),
        "label": labels,
        "logit_difference": differences,
        "raw_probability": probabilities,
        "raw_prediction": predictions,
    })

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Hasil tersimpan:", output_path)


if __name__ == "__main__":
    for config in CONFIGS:
        evaluate(config)
