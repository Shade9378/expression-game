from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from constants import CLASSES


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if probabilities.shape != (len(labels), len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction matrix")
    predictions = probabilities.argmax(axis=1)
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", labels=range(7), zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class": classification_report(
            labels,
            predictions,
            labels=range(7),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=range(7)).tolist(),
    }


def save_evaluation(
    labels: np.ndarray, probabilities: np.ndarray, sample_ids: list[str], output: Path
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(labels, probabilities)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    frame = pd.DataFrame(probabilities, columns=[f"p_{c}" for c in CLASSES])
    frame.insert(0, "sample_id", sample_ids)
    frame.insert(1, "label", labels)
    frame.insert(2, "prediction", probabilities.argmax(axis=1))
    frame.to_csv(output / "predictions.csv", index=False)
    return metrics
