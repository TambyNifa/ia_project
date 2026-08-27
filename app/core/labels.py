"""Valeurs de référence partagées par les pipelines TENA AI."""

from enum import IntEnum, StrEnum


class Label(IntEnum):
    """Labels internes utilisés pour l'entraînement."""

    PROBABLEMENT_FIABLE = 0
    SUSPECT = 1
    DIFFICILE_A_DETERMINER = 2


class Language(StrEnum):
    """Langues prises en charge par le dataset."""

    FR = "fr"
    MG = "mg"
    MIXED = "mixed"
    UNKNOWN = "unknown"


VALID_LABELS = {label.name for label in Label}
VALID_LANGUAGES = {language.value for language in Language}
