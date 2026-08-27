"""Nettoyage textuel conservateur pour les données TENA AI."""

import re
import unicodedata
from html import unescape

from app.ml.data.schema import DatasetRecord

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\r\n?|\n")


def clean_text(value: str) -> str:
    """Normalise Unicode, HTML et espaces sans retirer le contenu sémantique."""
    normalized = unicodedata.normalize("NFC", value)
    normalized = unescape(HTML_TAG_PATTERN.sub(" ", normalized)).replace("\xa0", " ")
    normalized = NEWLINE_PATTERN.sub("\n", normalized)
    normalized = "\n".join(
        WHITESPACE_PATTERN.sub(" ", line).strip() for line in normalized.split("\n")
    )
    return normalized.strip()


def prepare_dataframe(dataframe):
    """Retourne une copie enrichie avec clean_title et clean_text."""
    prepared = dataframe.copy()
    prepared["clean_title"] = prepared["title"].fillna("").map(clean_text)
    prepared["clean_text"] = prepared["text"].fillna("").map(clean_text)
    return prepared


def prepare_record(record: DatasetRecord) -> dict[str, object]:
    """Prépare un enregistrement validé sans écraser ses champs originaux."""
    values = record.model_dump()
    values["clean_title"] = clean_text(record.title)
    values["clean_text"] = clean_text(record.text)
    return values
