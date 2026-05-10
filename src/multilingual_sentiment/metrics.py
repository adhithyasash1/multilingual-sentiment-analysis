from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import ID_TO_LABEL


def prediction_ids(predictions: Any) -> np.ndarray:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    return predictions.astype(int)


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    labels = np.asarray(eval_prediction.label_ids)
    preds = prediction_ids(eval_prediction.predictions)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def build_detailed_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    target_names = [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]
    report = classification_report(
        labels,
        predictions,
        labels=sorted(ID_TO_LABEL),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    details: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=sorted(ID_TO_LABEL)).tolist(),
    }

    if languages is not None:
        language_metrics: dict[str, Any] = {}
        language_array = np.asarray(languages)
        for language in sorted(set(language_array)):
            mask = language_array == language
            if not mask.any():
                continue
            language_labels = labels[mask]
            language_preds = predictions[mask]
            language_metrics[str(language)] = {
                "support": int(mask.sum()),
                "accuracy": float(accuracy_score(language_labels, language_preds)),
                "f1_weighted": float(
                    f1_score(language_labels, language_preds, average="weighted", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(language_labels, language_preds, average="macro", zero_division=0)
                ),
            }
        details["by_language"] = language_metrics

    return details


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
