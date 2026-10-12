import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)


CALIBRATION_FILE = Path(
    "results/calibration_results_leakage_free_v2.json"
)

SELECTIVE_CONFIG_FILE = Path(
    "results/selective_prediction_config_leakage_free_v2.json"
)

OUTPUT_DIR = Path("results")
BATCH_SIZE = 32
MAX_LENGTH = 128


CONFIGS = [
    {
        "name": "toxicity",
        "display": "TOXICITY",
        "model": "results/toxicity_indobertweet_leakage_free_v2/final_model",
        "test_file": "data/processed_leakage_free_v2/toxicity_test.csv",
        "label_column": "toxicity_label",
        "raw_threshold": 0.10,
    },
    {
        "name": "polarization",
        "display": "POLARIZATION",
        "model": "results/polarized_indobertweet_leakage_free_v2/final_model",
        "test_file": "data/processed_leakage_free_v2/polarized_test.csv",
        "label_column": "polarized_label",
        "raw_threshold": 0.50,
    },
]


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def calculate_ece(probabilities, labels, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
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


def evaluate(config, calibration, selective_config):
    print(f"\n{'=' * 60}")
    print(config["display"])
    print("=" * 60)

    model_path = Path(config["model"])
    test_path = Path(config["test_file"])

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    if not test_path.is_file():
        raise FileNotFoundError(f"Test set tidak ditemukan: {test_path}")

    data = pd.read_csv(test_path)
    data = data.dropna(
        subset=["text_id", "text", config["label_column"]]
    ).copy()

    labels = data[config["label_column"]].astype(int).to_numpy()
    texts = data["text"].astype(str).tolist()

    if len(data) == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Test set kosong atau label tidak valid.")

    temperature = float(calibration[config["name"]]["temperature"])
    delta = float(selective_config[config["name"]]["delta"])
    raw_threshold = float(config["raw_threshold"])

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )
    model.eval()

    logit_differences = []

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

        logit_differences.extend(differences.tolist())

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r",
        )

    print()

    logits = np.asarray(logit_differences, dtype=np.float64)
    raw_probability = sigmoid(logits)
    calibrated_probability = sigmoid(logits / temperature)

    # Pertahankan aturan label aplikasi yang sudah ditetapkan.
    predictions = (raw_probability >= raw_threshold).astype(int)

    # Transformasi threshold ke skala probabilitas terkalibrasi.
    calibrated_threshold = float(
        sigmoid(
            np.log(raw_threshold / (1 - raw_threshold)) / temperature
        )
    )

    distance = np.abs(
        calibrated_probability - calibrated_threshold
    )

    uncertain = distance < delta
    accepted = ~uncertain

    raw_prediction_metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
    }

    calibration_metrics = {
        "raw_brier": float(brier_score_loss(labels, raw_probability)),
        "calibrated_brier": float(
            brier_score_loss(labels, calibrated_probability)
        ),
        "raw_logloss": float(
            log_loss(
                labels,
                np.column_stack([1 - raw_probability, raw_probability]),
                labels=[0, 1],
            )
        ),
        "calibrated_logloss": float(
            log_loss(
                labels,
                np.column_stack([
                    1 - calibrated_probability,
                    calibrated_probability,
                ]),
                labels=[0, 1],
            )
        ),
        "raw_ece": calculate_ece(raw_probability, labels),
        "calibrated_ece": calculate_ece(
            calibrated_probability, labels
        ),
    }

    accepted_count = int(accepted.sum())
    uncertain_count = int(uncertain.sum())

    accepted_error_rate = None
    accepted_accuracy = None

    if accepted_count:
        accepted_accuracy = float(
            accuracy_score(labels[accepted], predictions[accepted])
        )
        accepted_error_rate = 1.0 - accepted_accuracy

    result_summary = {
        "model": config["name"],
        "test_samples": int(len(data)),
        "temperature": temperature,
        "raw_threshold": raw_threshold,
        "calibrated_threshold": calibrated_threshold,
        "delta_selected_on_validation": delta,
        "prediction_metrics": raw_prediction_metrics,
        "calibration_metrics": calibration_metrics,
        "selective_prediction": {
            "accepted": accepted_count,
            "uncertain": uncertain_count,
            "coverage": float(accepted.mean()),
            "accepted_accuracy": accepted_accuracy,
            "accepted_error_rate": accepted_error_rate,
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }

    print("Test samples:", len(data))
    print("Temperature:", temperature)
    print("Delta:", delta)
    print("Prediction metrics:", raw_prediction_metrics)
    print("Calibration metrics:", calibration_metrics)
    print("Selective prediction:", result_summary["selective_prediction"])
    print("Confusion matrix:")
    print(result_summary["confusion_matrix"])

    output = data[["text_id"]].copy()
    output["label"] = labels
    output["raw_probability"] = raw_probability
    output["calibrated_probability"] = calibrated_probability
    output["prediction"] = predictions
    output["distance"] = distance
    output["uncertain"] = uncertain
    output["accepted"] = accepted

    output_path = OUTPUT_DIR / (
        f"{config['name']}_test_evaluation_leakage_free_v2.csv"
    )
    output.to_csv(output_path, index=False)

    print("Prediksi tersimpan:", output_path)

    return result_summary


def main():
    calibration = json.loads(
        CALIBRATION_FILE.read_text(encoding="utf-8")
    )

    selective_config = json.loads(
        SELECTIVE_CONFIG_FILE.read_text(encoding="utf-8")
    )

    summaries = {}

    for config in CONFIGS:
        summaries[config["name"]] = evaluate(
            config,
            calibration,
            selective_config,
        )

    output_path = OUTPUT_DIR / (
        "test_evaluation_summary_leakage_free_v2.json"
    )

    output_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    print(f"\nEvaluasi test selesai: {output_path}")


if __name__ == "__main__":
    main()
