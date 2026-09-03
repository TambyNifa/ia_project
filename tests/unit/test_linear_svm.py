"""Tests unitaires du modèle Linear SVM."""

import pandas as pd

from app.core.labels import Label
from app.ml.training.svm import SVMConfig, load_svm_model, save_svm_model, train_svm


def test_linear_svm_trains_predicts_and_exposes_decision_function(tmp_path) -> None:
    dataframe = pd.DataFrame(
        {
            "clean_title": ["", "", "", ""],
            "clean_text": ["alpha", "beta", "alpha", "beta"],
            "label": ["PROBABLEMENT_FIABLE", "SUSPECT"] * 2,
        }
    )
    pipeline = train_svm(dataframe.iloc[:2], SVMConfig(C=1.0, ngram_range=(1, 1)))
    path = tmp_path / "svm.joblib"
    save_svm_model(pipeline, path)
    loaded = load_svm_model(path)

    assert set(loaded.named_steps["classifier"].classes_) == {
        "PROBABLEMENT_FIABLE",
        "SUSPECT",
    }
    assert len(loaded.predict(["alpha"])) == 1
    assert loaded.decision_function(["alpha"]).shape == (1,)


def test_svm_metadata_contract() -> None:
    metadata = {
        "trained_classes": [Label.PROBABLEMENT_FIABLE.name, Label.SUSPECT.name],
        "not_trained_classes": [Label.DIFFICILE_A_DETERMINER.name],
        "metrics_dev": {},
        "metrics_test": {},
        "metrics_fr_zeroshot": {},
    }

    assert metadata["not_trained_classes"] == ["DIFFICILE_A_DETERMINER"]
    assert set(metadata["trained_classes"]) == {"PROBABLEMENT_FIABLE", "SUSPECT"}
