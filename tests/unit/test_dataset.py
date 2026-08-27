"""Tests rapides des règles de qualité du dataset."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.labels import Label
from app.ml.data.schema import DatasetRecord
from app.ml.data.validation import validate_dataframe


def valid_record(**overrides):
    values = {
        "id": "id-1",
        "title": "Titre",
        "text": "Contenu",
        "language": "fr",
        "source": "Source",
        "publication_date": None,
        "label": "PROBABLEMENT_FIABLE",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(overrides)
    return values


def test_schema_accepts_valid_record_and_centralized_label() -> None:
    record = DatasetRecord(**valid_record())

    assert record.label == Label.PROBABLEMENT_FIABLE.name


@pytest.mark.parametrize(
    "field,value",
    [("label", "FAUX"), ("language", "en"), ("text", " ")],
)
def test_schema_rejects_invalid_values(field, value) -> None:
    with pytest.raises(ValidationError):
        DatasetRecord(**valid_record(**{field: value}))


def test_validation_reports_duplicates_empty_text_and_distributions() -> None:
    dataframe = pd.DataFrame(
        [
            valid_record(),
            valid_record(id="id-1"),
            valid_record(id="id-2", text="", language="mg", label="SUSPECT"),
        ]
    )

    report = validate_dataframe(dataframe)

    assert report["total_rows"] == 3
    assert report["duplicate_ids"] == 2
    assert report["empty_texts"] == 1
    assert report["invalid_rows"] == 3
    assert report["label_distribution"]["SUSPECT"] == 1
    assert report["language_distribution"]["mg"] == 1
