import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import brier_score_loss, log_loss


CONFIGS = [
    {
        "name": "toxicity",
        "file": "results/toxicity_validation_predictions_leakage_free_v2.csv",
        "label_column": "toxicity_label",
        "threshold": 0.10,
        "output": "results/toxicity_calibration_validation_leakage_free_v2.csv",
    },
    {
        "name": "polarization",
        "file": "results/polarized_validation_predictions_leakage_free_v2.csv",
        "label_column": "polarized_label",
        "threshold": 0.50,
        "output": "results/polarization_calibration_validation_leakage_free_v2.csv",
    },
]


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


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


def fit_temperature(logit_differences, labels):
    logits = torch.tensor(
        logit_differences,
        dtype=torch.float64,
    )
    targets = torch.tensor(
        labels,
        dtype=torch.float64,
    )

    log_temperature = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.float64)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=100,
    )

    loss_function = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)
        loss = loss_function(
            logits / temperature,
            targets,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    return float(
        torch.exp(log_temperature).detach().item()
    )


def evaluate(config):
    source = Path(config["file"])

    if not source.exists():
        raise FileNotFoundError(
            f"File validation tidak ditemukan: {source}"
        )

    df = pd.read_csv(source)

    required = {"text_id", "label", "logit_difference"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom tidak tersedia di {source}: {sorted(missing)}"
        )

    labels = df["label"].astype(int).to_numpy()
    logits = df["logit_difference"].astype(float).to_numpy()

    if len(labels) == 0 or not np.isfinite(logits).all():
        raise ValueError("Data validation kosong atau berisi nilai tidak valid.")

    raw_probability = sigmoid(logits)
    temperature = fit_temperature(logits, labels)
    calibrated_probability = sigmoid(logits / temperature)

    raw_brier = float(
        brier_score_loss(labels, raw_probability)
    )
    calibrated_brier = float(
        brier_score_loss(labels, calibrated_probability)
    )

    raw_logloss = float(
        log_loss(
            labels,
            np.column_stack([
                1 - raw_probability,
                raw_probability,
            ]),
            labels=[0, 1],
        )
    )
    calibrated_logloss = float(
        log_loss(
            labels,
            np.column_stack([
                1 - calibrated_probability,
                calibrated_probability,
            ]),
            labels=[0, 1],
        )
    )

    raw_ece = calculate_ece(raw_probability, labels)
    calibrated_ece = calculate_ece(
        calibrated_probability,
        labels,
    )

    threshold = config["threshold"]
    calibrated_threshold = float(
        sigmoid(
            np.log(threshold / (1 - threshold))
            / temperature
        )
    )

    predictions = (
        raw_probability >= threshold
    ).astype(int)

    distance = np.abs(
        calibrated_probability - calibrated_threshold
    )

    result = df.copy()
    result["raw_probability"] = raw_probability
    result["calibrated_probability"] = calibrated_probability
    result["prediction"] = predictions
    result["correct"] = (predictions == labels).astype(int)
    result["calibrated_threshold"] = calibrated_threshold
    result["distance"] = distance

    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    metrics = {
        "temperature": temperature,
        "threshold": threshold,
        "calibrated_threshold": calibrated_threshold,
        "validation_samples": int(len(labels)),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "raw_logloss": raw_logloss,
        "calibrated_logloss": calibrated_logloss,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
    }

    print(f"\n=== {config['name'].upper()} ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"Prediksi tersimpan: {output}")

    return metrics


def main():
    all_metrics = {}

    for config in CONFIGS:
        all_metrics[config["name"]] = evaluate(config)

    result_path = Path(
        "results/calibration_results_leakage_free_v2.json"
    )

    result_path.write_text(
        json.dumps(
            all_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nKalibrasi selesai: {result_path}")


if __name__ == "__main__":
    main()
