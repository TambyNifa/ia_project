"""Tests unitaires sans dépendances Transformer ni téléchargement."""

from pathlib import Path

import pandas as pd
import pytest

from app.ml.transformers.config import XLMRobertaConfig
from app.ml.transformers.dataset import encode_labels, load_transformer_split


def test_xlm_configuration_is_reproducible() -> None:
    config = XLMRobertaConfig()

    assert config.model_name == "xlm-roberta-base"
    assert config.max_length == 256
    assert config.as_dict()["seed"] == 42


def test_encode_labels_keeps_binary_contract() -> None:
    assert encode_labels(["PROBABLEMENT_FIABLE", "SUSPECT"]) == [0, 1]
    with pytest.raises(ValueError):
        encode_labels(["DIFFICILE_A_DETERMINER"])


def test_load_split_rejects_zero_shot_as_training_input(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "id": ["1"],
            "title": [None],
            "text": ["claim"],
            "language": ["unknown"],
            "source": ["source"],
            "publication_date": [None],
            "label": ["PROBABLEMENT_FIABLE"],
            "created_at": ["2026-01-01"],
        }
    )
    dataframe.to_csv(tmp_path / "fr_zeroshot.csv", index=False)

    assert (
        load_transformer_split("fr_zeroshot", tmp_path).loc[0, "model_text"] == "claim"
    )
    with pytest.raises(ValueError):
        load_transformer_split("training", tmp_path)
