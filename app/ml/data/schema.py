"""Schéma d'une observation du dataset TENA AI."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.core.labels import VALID_LABELS, VALID_LANGUAGES


class DatasetRecord(BaseModel):
    """Enregistrement strict utilisé par les étapes de préparation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    title: str | None = None
    text: str
    language: str
    source: str | None = None
    publication_date: date | None = None
    label: str
    created_at: datetime

    @field_validator("id", "text", "label", "language")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("La valeur ne peut pas être vide.")
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in VALID_LABELS:
            raise ValueError(f"Label invalide : {value}")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in VALID_LANGUAGES:
            raise ValueError(f"Langue invalide : {value}")
        return value
