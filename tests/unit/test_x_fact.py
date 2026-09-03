"""Tests de l'importateur X-FACT sans téléchargement réseau."""

import pandas as pd

from app.core.labels import Label
from app.ml.data.x_fact import (
    ZERO_SHOT_SPLIT,
    harmonize_split,
    import_sources,
    map_label,
    split_sources,
)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"language": "de", "site": "site", "claim": "A", "label": "true"},
            {"language": "de", "site": "site", "claim": "B", "label": "false"},
            {
                "language": "de",
                "site": "site",
                "claim": "C",
                "label": "mostly true",
            },
        ]
    )


def test_mapping_excludes_unapproved_labels() -> None:
    assert map_label("true") == Label.PROBABLEMENT_FIABLE.name
    assert map_label("false") == Label.SUSPECT.name
    assert map_label("mostly true") is None


def test_harmonization_does_not_invent_title() -> None:
    result = harmonize_split(source_frame(), "train")

    assert result["title"].isna().all()
    assert result.loc[0, "text"] == "A"
    assert pd.isna(result.loc[2, "label"])


def test_import_keeps_zero_shot_separate(tmp_path) -> None:
    frames = {source.name: source_frame() for source in split_sources()}

    def reader(url: str) -> pd.DataFrame:
        return frames[ZERO_SHOT_SPLIT] if "/fr/" in url else frames["train"]

    report = import_sources(split_sources(), tmp_path / "processed", reader=reader)

    assert set(report["splits"]) == {"train", "dev", "test", ZERO_SHOT_SPLIT}
    assert report["splits"][ZERO_SHOT_SPLIT]["output"].endswith("fr_zeroshot.csv")
    assert (tmp_path / "processed" / "fr_zeroshot.csv").exists()
