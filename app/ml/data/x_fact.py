"""Importation et harmonisation reproductibles du dataset X-FACT."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from app.core.labels import Label
from app.ml.data.validation import validate_dataframe

X_FACT_REPOSITORY = "https://github.com/utahnlp/x-fact"
X_FACT_DATASET = "https://huggingface.co/datasets/utahnlp/x-fact"
X_FACT_BASE_URL = "https://huggingface.co/datasets/utahnlp/x-fact/resolve/main"
TRAINING_SPLITS = ("train", "dev", "test")
ZERO_SHOT_SPLIT = "fr_zeroshot"
LABEL_MAPPING = {
    "true": Label.PROBABLEMENT_FIABLE.name,
    "false": Label.SUSPECT.name,
}
REQUIRED_SOURCE_COLUMNS = {"language", "site", "claim", "label"}


@dataclass(frozen=True)
class SplitSource:
    """Source distante d'un split X-FACT."""

    name: str
    url: str
    output_name: str


def split_sources() -> tuple[SplitSource, ...]:
    """Retourne les quatre sources autorisées pour X-FACT V1."""
    return (
        SplitSource(
            "train",
            f"{X_FACT_BASE_URL}/all_languages/train-00000-of-00001.csv",
            "train.csv",
        ),
        SplitSource(
            "dev",
            f"{X_FACT_BASE_URL}/all_languages/dev-00000-of-00001.csv",
            "dev.csv",
        ),
        SplitSource(
            "test",
            f"{X_FACT_BASE_URL}/all_languages/test-00000-of-00001.csv",
            "test.csv",
        ),
        SplitSource(
            ZERO_SHOT_SPLIT,
            f"{X_FACT_BASE_URL}/fr/zeroshot-00000-of-00001.csv",
            "fr_zeroshot.csv",
        ),
    )


def map_label(label: object) -> str | None:
    """Mappe uniquement les deux labels explicitement approuvés."""
    return LABEL_MAPPING.get(str(label).strip().lower())


def source_label_report(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Décrit chaque label original et l'action appliquée."""
    counts = dataframe["label"].value_counts(dropna=False)
    return [
        {
            "original_label": str(original),
            "count": int(count),
            "action": "mapped" if map_label(original) else "excluded",
            "final_label": map_label(original),
            "reason": (
                "Mapping explicitement validé pour X-FACT V1."
                if map_label(original)
                else "Label non approuvé pour éviter un mapping automatique ambigu."
            ),
        }
        for original, count in counts.items()
    ]


def harmonize_split(dataframe: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Convertit un split source en schéma interne sans inventer de titre."""
    missing = REQUIRED_SOURCE_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Colonnes X-FACT absentes : {sorted(missing)}")
    result = pd.DataFrame(
        {
            "id": [f"x-fact-{split_name}-{index}" for index in range(len(dataframe))],
            "title": None,
            "text": dataframe["claim"],
            "language": dataframe["language"].where(
                dataframe["language"].isin({"fr", "mg", "mixed", "unknown"}),
                "unknown",
            ),
            "original_language": dataframe["language"],
            "source": dataframe["site"],
            "publication_date": dataframe.get("claimDate"),
            "label": dataframe["label"].map(map_label),
            "created_at": datetime.now(UTC),
        }
    )
    return result


def import_sources(
    sources: tuple[SplitSource, ...],
    output_directory,
    external_directory=None,
    *,
    reader: Callable[[str], pd.DataFrame] = pd.read_csv,
) -> dict[str, object]:
    """Importe les splits, les sauvegarde séparément et retourne un rapport."""
    output_directory.mkdir(parents=True, exist_ok=True)
    if external_directory is not None:
        external_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source": X_FACT_DATASET,
        "splits": {},
        "errors": [],
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    for source in sources:
        try:
            raw = reader(source.url)
            if external_directory is not None:
                raw.to_csv(external_directory / source.output_name, index=False)
            split_report = {
                "raw_rows": len(raw),
                "languages": sorted(raw["language"].dropna().unique().tolist()),
                "source_label_report": source_label_report(raw),
            }
            harmonized = harmonize_split(raw, source.name)
            mapped = harmonized.loc[harmonized["label"].notna()].copy()
            mapped = mapped.loc[
                mapped["text"].fillna("").astype(str).str.strip().ne("")
            ]
            duplicate_count = int(
                mapped.duplicated(subset=["text", "original_language", "label"]).sum()
            )
            mapped = mapped.drop_duplicates(
                subset=["text", "original_language", "label"]
            )
            validation = validate_dataframe(mapped)
            output_path = output_directory / source.output_name
            mapped.to_csv(output_path, index=False)
            split_report.update(
                {
                    "imported_rows": len(mapped),
                    "excluded_rows": len(harmonized) - len(mapped),
                    "duplicates_removed": duplicate_count,
                    "validation": validation,
                    "output": str(output_path),
                }
            )
            report["splits"][source.name] = split_report
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            report["errors"].append({"split": source.name, "error": str(error)})
    return report
