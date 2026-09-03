"""Préparation des données X-FACT pour XLM-RoBERTa."""

from pathlib import Path

import pandas as pd

from app.core.labels import Label
from app.ml.preprocessing.text import prepare_dataframe

TRAINED_LABELS = [Label.PROBABLEMENT_FIABLE.name, Label.SUSPECT.name]


def load_transformer_split(
    split: str, dataset_directory: Path = Path("data/processed/x_fact_v1")
) -> pd.DataFrame:
    """Charge un split autorisé et construit le texte naturel d'entrée."""
    if split not in {"train", "dev", "test", "fr_zeroshot"}:
        raise ValueError(f"Split non autorisé : {split}")
    path = dataset_directory / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split introuvable : {path}")
    dataframe = prepare_dataframe(pd.read_csv(path))
    if not set(dataframe["label"]).issubset(TRAINED_LABELS):
        raise ValueError(f"Labels inattendus dans {split}")
    dataframe["model_text"] = dataframe["clean_text"]
    return dataframe


def encode_labels(labels: list[str]) -> list[int]:
    """Convertit les labels TENA AI vers les identifiants binaires du modèle."""
    mapping = {label: index for index, label in enumerate(TRAINED_LABELS)}
    try:
        return [mapping[label] for label in labels]
    except KeyError as error:
        raise ValueError(f"Label non entraîné : {error.args[0]}") from error


def tokenize_dataframe(dataframe: pd.DataFrame, tokenizer, max_length: int):
    """Tokenise sans modifier le dataframe source."""
    return tokenizer(
        dataframe["model_text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
    )
