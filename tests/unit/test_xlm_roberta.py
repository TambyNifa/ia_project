"""Tests unitaires sans dépendances Transformer ni téléchargement."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ml.transformers.config import XLMRobertaConfig
from app.ml.transformers.dataset import (
    encode_labels,
    load_transformer_split,
    tokenize_dataframe,
)
from app.ml.transformers.train import compute_classification_metrics


def test_xlm_configuration_is_reproducible() -> None:
    config = XLMRobertaConfig()

    assert config.model_name == "xlm-roberta-base"
    assert config.max_length == 256
    assert config.as_dict()["seed"] == 42


def test_encode_labels_keeps_binary_contract() -> None:
    assert encode_labels(["PROBABLEMENT_FIABLE", "SUSPECT"]) == [0, 1]
    with pytest.raises(ValueError):
        encode_labels(["DIFFICILE_A_DETERMINER"])


def test_tokenizer_uses_text_without_eager_padding() -> None:
    dataframe = pd.DataFrame({"model_text": ["claim one", "claim two"]})
    received = {}

    def tokenizer(texts, **kwargs):
        received["texts"] = texts
        received.update(kwargs)
        return {"input_ids": [[1, 2], [1, 2, 3]]}

    tokens = tokenize_dataframe(dataframe, tokenizer, max_length=256)

    assert received["texts"] == ["claim one", "claim two"]
    assert received["padding"] is False
    assert received["truncation"] is True
    assert received["max_length"] == 256
    assert [len(row) for row in tokens["input_ids"]] == [2, 3]


def test_metrics_accept_eval_prediction_logits_and_tuple() -> None:
    prediction = SimpleNamespace(
        label_ids=[0, 1, 1],
        predictions=(
            [[4.0, 0.1], [0.2, 3.0], [0.1, 2.0]],
            "unused auxiliary prediction",
        ),
    )

    metrics = compute_classification_metrics(prediction)

    assert metrics == {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}


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
