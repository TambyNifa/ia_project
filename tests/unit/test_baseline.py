"""Tests unitaires de la baseline et du découpage des données."""

import pandas as pd

from app.ml.data.splitting import split_dataset
from app.ml.evaluation.metrics import evaluate_predictions
from app.ml.preprocessing.text import prepare_dataframe
from app.ml.training.baseline import (
    BaselineConfig,
    combine_text,
    load_model,
    save_model,
    train_baseline,
)


def make_dataframe() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(
        ["PROBABLEMENT_FIABLE", "SUSPECT", "DIFFICILE_A_DETERMINER"] * 4
    ):
        rows.append(
            {
                "id": str(index),
                "title": f"Titre {index}",
                "text": f"Texte distinct numéro {index}",
                "label": label,
            }
        )
    return prepare_dataframe(pd.DataFrame(rows))


def test_split_is_reproducible_and_disjoint() -> None:
    split = split_dataset(make_dataframe(), random_state=42)
    repeat = split_dataset(make_dataframe(), random_state=42)

    assert split.stratified is True
    assert set(split.train["id"]).isdisjoint(split.validation["id"])
    assert set(split.train["id"]).isdisjoint(split.test["id"])
    assert split.train["id"].tolist() == repeat.train["id"].tolist()


def test_small_dataset_uses_documented_fallback() -> None:
    dataframe = make_dataframe().iloc[:4]

    split = split_dataset(dataframe, random_state=42)

    assert split.stratified is False
    assert split.warning is not None
    assert len(split.train) == 2


def test_training_prediction_and_model_loading(tmp_path) -> None:
    dataframe = make_dataframe()
    split = split_dataset(dataframe, random_state=42)
    pipeline = train_baseline(
        split.train,
        config=BaselineConfig(ngram_range=(1, 1), random_state=42),
    )
    path = tmp_path / "model.joblib"
    save_model(pipeline, path)
    loaded = load_model(path)

    assert len(loaded.predict(combine_text(split.test))) == len(split.test)


def test_metrics_include_macro_and_weighted_f1() -> None:
    metrics = evaluate_predictions(
        ["PROBABLEMENT_FIABLE", "SUSPECT"],
        ["PROBABLEMENT_FIABLE", "PROBABLEMENT_FIABLE"],
        ["PROBABLEMENT_FIABLE", "SUSPECT", "DIFFICILE_A_DETERMINER"],
    )

    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 3
