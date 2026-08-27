"""Tests du preprocessing conservateur et de la détection de langue."""

import pandas as pd

from app.ml.preprocessing.language import LanguageDetector
from app.ml.preprocessing.text import clean_text, prepare_dataframe


def test_clean_text_preserves_punctuation_hashtags_numbers_and_emoji() -> None:
    value = "  <b>Bonjour&nbsp;!</b>\r\n#Info 2026 😊  "

    assert clean_text(value) == "Bonjour !\n#Info 2026 😊"


def test_prepare_dataframe_keeps_original_columns() -> None:
    dataframe = pd.DataFrame([{"title": "<i>Titre</i>", "text": " Texte  brut "}])

    prepared = prepare_dataframe(dataframe)

    assert prepared.loc[0, "text"] == " Texte  brut "
    assert prepared.loc[0, "clean_title"] == "Titre"
    assert prepared.loc[0, "clean_text"] == "Texte brut"


def test_language_detector_handles_supported_languages_and_unknown() -> None:
    detector = LanguageDetector()

    assert detector.detect("Le texte est dans la source") == "fr"
    assert detector.detect("Ny vaovao dia ao amin'ny lahatsoratra") == "mg"
    assert detector.detect("Le vaovao dia dans ny lahatsoratra") == "mixed"
    assert detector.detect("123 qwerty") == "unknown"
