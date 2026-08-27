"""Validation explicite des fichiers tabulaires de TENA AI."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.labels import VALID_LABELS, VALID_LANGUAGES

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = (
    "id",
    "title",
    "text",
    "language",
    "source",
    "publication_date",
    "label",
    "created_at",
)
NON_NULLABLE_COLUMNS = ("id", "title", "text", "language", "label", "created_at")


def validate_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Retourne un rapport détaillé sans modifier le dataframe fourni."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        LOGGER.error("Colonnes obligatoires absentes : %s", missing_columns)
        return {
            "total_rows": len(dataframe),
            "valid_rows": 0,
            "invalid_rows": len(dataframe),
            "missing_values": {},
            "duplicate_rows": 0,
            "duplicate_ids": 0,
            "invalid_labels": {},
            "invalid_languages": {},
            "empty_texts": 0,
            "missing_columns": missing_columns,
            "label_distribution": {},
            "language_distribution": {},
        }

    missing_values = dataframe[list(REQUIRED_COLUMNS)].isna().sum()
    missing_values = {
        column: int(count) for column, count in missing_values.items() if count
    }
    empty_text_mask = dataframe["text"].fillna("").astype(str).str.strip().eq("")
    empty_title_mask = dataframe["title"].fillna("").astype(str).str.strip().eq("")
    invalid_labels = (
        dataframe.loc[~dataframe["label"].isin(VALID_LABELS), "label"]
        .value_counts(dropna=False)
        .to_dict()
    )
    invalid_languages = (
        dataframe.loc[~dataframe["language"].isin(VALID_LANGUAGES), "language"]
        .value_counts(dropna=False)
        .to_dict()
    )
    duplicate_rows_mask = dataframe.duplicated(keep=False)
    duplicate_ids_mask = dataframe["id"].duplicated(keep=False)
    invalid_mask = (
        dataframe[list(NON_NULLABLE_COLUMNS)].isna().any(axis=1)
        | empty_text_mask
        | empty_title_mask
        | ~dataframe["label"].isin(VALID_LABELS)
        | ~dataframe["language"].isin(VALID_LANGUAGES)
        | duplicate_ids_mask
    )
    report = {
        "total_rows": len(dataframe),
        "valid_rows": int((~invalid_mask).sum()),
        "invalid_rows": int(invalid_mask.sum()),
        "missing_values": missing_values,
        "duplicate_rows": int(duplicate_rows_mask.sum()),
        "duplicate_ids": int(duplicate_ids_mask.sum()),
        "invalid_labels": {
            str(key): int(value) for key, value in invalid_labels.items()
        },
        "invalid_languages": {
            str(key): int(value) for key, value in invalid_languages.items()
        },
        "empty_texts": int(empty_text_mask.sum()),
        "empty_titles": int(empty_title_mask.sum()),
        "missing_columns": [],
        "label_distribution": {
            str(key): int(value)
            for key, value in dataframe["label"].value_counts(dropna=False).items()
        },
        "language_distribution": {
            str(key): int(value)
            for key, value in dataframe["language"].value_counts(dropna=False).items()
        },
    }
    LOGGER.info(
        "Validation dataset : %s lignes, %s valides, %s invalides",
        report["total_rows"],
        report["valid_rows"],
        report["invalid_rows"],
    )
    return report


def validate_csv(path: Path) -> dict[str, Any]:
    """Charge et valide un CSV, en laissant le fichier source inchangé."""
    LOGGER.info("Validation du fichier dataset : %s", path)
    return validate_dataframe(pd.read_csv(path))


def get_valid_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Retourne une copie des lignes valides après validation explicite."""
    required = list(REQUIRED_COLUMNS)
    if any(column not in dataframe for column in required):
        raise ValueError("Le dataframe ne contient pas toutes les colonnes requises.")
    empty_text_mask = dataframe["text"].fillna("").astype(str).str.strip().eq("")
    empty_title_mask = dataframe["title"].fillna("").astype(str).str.strip().eq("")
    invalid_mask = (
        dataframe[list(NON_NULLABLE_COLUMNS)].isna().any(axis=1)
        | empty_text_mask
        | empty_title_mask
        | ~dataframe["label"].isin(VALID_LABELS)
        | ~dataframe["language"].isin(VALID_LANGUAGES)
        | dataframe["id"].duplicated(keep=False)
    )
    return dataframe.loc[~invalid_mask].copy()
