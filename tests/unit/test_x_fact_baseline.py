"""Tests du réentraînement X-FACT sans exécuter l'import réseau."""

import pandas as pd

from app.ml.training.baseline import BaselineConfig, train_baseline
from scripts.train_x_fact_baseline import evaluate_split


def test_x_fact_baseline_uses_binary_classes_and_evaluates_split() -> None:
    dataframe = pd.DataFrame(
        {
            "clean_title": ["", "", "", ""],
            "clean_text": ["alpha", "beta", "alpha", "beta"],
            "label": [
                "PROBABLEMENT_FIABLE",
                "SUSPECT",
                "PROBABLEMENT_FIABLE",
                "SUSPECT",
            ],
        }
    )
    pipeline = train_baseline(
        dataframe.iloc[:2],
        config=BaselineConfig(ngram_range=(1, 1), random_state=42),
    )

    metrics = evaluate_split(pipeline, dataframe.iloc[2:])

    assert set(pipeline.named_steps["classifier"].classes_) == {
        "PROBABLEMENT_FIABLE",
        "SUSPECT",
    }
    assert len(metrics["confusion_matrix"]) == 2
