"""Métriques d'évaluation de la baseline."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    expected: list[str],
    predicted: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Calcule les métriques et représentations nécessaires au rapport."""
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(
            precision_score(
                expected, predicted, labels=labels, average="macro", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                expected, predicted, labels=labels, average="macro", zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                expected, predicted, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                expected, predicted, labels=labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                expected, predicted, labels=labels, average="weighted", zero_division=0
            )
        ),
        "classification_report": classification_report(
            expected, predicted, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=labels
        ).tolist(),
    }
