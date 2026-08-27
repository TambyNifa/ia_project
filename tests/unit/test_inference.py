"""Tests de l'inférence locale de la baseline."""

import json
from pathlib import Path

import pytest

from app.ml.inference.baseline import BaselinePredictor
from app.ml.training.baseline import BaselineConfig, save_model, train_baseline
from tests.unit.test_baseline import make_dataframe


@pytest.fixture
def predictor(tmp_path: Path) -> BaselinePredictor:
    dataframe = make_dataframe()
    model = train_baseline(
        dataframe.iloc[:9],
        config=BaselineConfig(ngram_range=(1, 1), random_state=42),
    )
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    save_model(model, model_path)
    metadata_path.write_text(
        json.dumps({"version": "test", "dataset_type": "fixture"}),
        encoding="utf-8",
    )
    return BaselinePredictor(model_path, metadata_path)


def test_prediction_result_has_expected_structure(predictor) -> None:
    result = predictor.predict("Titre", "Un texte de test.")

    assert set(result) == {
        "prediction",
        "confidence",
        "probabilities",
        "model_version",
        "dataset_type",
        "disclaimer",
    }
    assert result["prediction"] in result["probabilities"]
    assert result["confidence"] == max(result["probabilities"].values())


def test_prediction_probabilities_sum_to_one(predictor) -> None:
    result = predictor.predict("Titre", "Un texte de test.")

    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_empty_input_is_rejected(predictor) -> None:
    with pytest.raises(ValueError):
        predictor.predict("   ", "<p> </p>")
