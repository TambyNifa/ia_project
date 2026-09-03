"""Réentraîne la baseline sur les splits officiels X-FACT V1."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from app.core.labels import Label
from app.core.logging import configure_logging
from app.ml.evaluation.metrics import evaluate_predictions
from app.ml.preprocessing.text import prepare_dataframe
from app.ml.training.baseline import (
    BaselineConfig,
    combine_text,
    save_model,
    train_baseline,
)

LOGGER = logging.getLogger(__name__)
DATASET_DIRECTORY = Path("data/processed/x_fact_v1")
MODEL_PATH = Path("models/tena_ai_x_fact_v1.joblib")
METADATA_PATH = Path("models/tena_ai_x_fact_v1_metadata.json")
TRAINED_LABELS = [
    Label.PROBABLEMENT_FIABLE.name,
    Label.SUSPECT.name,
]


def load_split(name: str) -> pd.DataFrame:
    """Charge et prépare un split X-FACT V1 sans modifier son fichier."""
    path = DATASET_DIRECTORY / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split X-FACT introuvable : {path}")
    return prepare_dataframe(pd.read_csv(path))


def evaluate_split(pipeline, dataframe: pd.DataFrame) -> dict[str, object]:
    """Évalue un split avec uniquement les classes réellement entraînées."""
    expected = dataframe["label"].tolist()
    predicted = pipeline.predict(combine_text(dataframe)).tolist()
    return evaluate_predictions(expected, predicted, TRAINED_LABELS)


def main() -> None:
    configure_logging()
    splits = {
        "train": load_split("train"),
        "dev": load_split("dev"),
        "test": load_split("test"),
        "fr_zeroshot": load_split("fr_zeroshot"),
    }
    for name, dataframe in splits.items():
        LOGGER.info(
            "%s : %s lignes, %s textes vides, %s doublons",
            name,
            len(dataframe),
            int(dataframe["clean_text"].eq("").sum()),
            int(dataframe.duplicated(subset=["text", "language", "label"]).sum()),
        )
    config = BaselineConfig(random_state=42)
    started = perf_counter()
    pipeline = train_baseline(splits["train"], config=config)
    duration = perf_counter() - started
    metrics_dev = evaluate_split(pipeline, splits["dev"])
    metrics_test = evaluate_split(pipeline, splits["test"])
    metrics_zeroshot = evaluate_split(pipeline, splits["fr_zeroshot"])
    save_model(pipeline, MODEL_PATH)

    metadata = {
        "model_name": "tfidf_logistic_regression",
        "model_version": "x_fact_v1",
        "dataset_name": "X-FACT",
        "dataset_version": "x_fact_v1",
        "dataset_type": "real",
        "training_date": datetime.now(UTC).isoformat(),
        "trained_classes": TRAINED_LABELS,
        "not_trained_classes": [Label.DIFFICILE_A_DETERMINER.name],
        "languages": sorted(splits["train"]["original_language"].unique().tolist()),
        "hyperparameters": config.as_dict(),
        "split_sizes": {name: len(dataframe) for name, dataframe in splits.items()},
        "metrics_dev": metrics_dev,
        "metrics_test": metrics_test,
        "metrics_fr_zeroshot": metrics_zeroshot,
        "training_duration_seconds": round(duration, 4),
        "limitations": [
            "X-FACT contient des affirmations vérifiées, pas des articles complets.",
            "Le dataset ne représente pas Madagascar.",
            "Le malgache est absent.",
            "Le modèle est entraîné uniquement sur deux classes.",
            "Les performances peuvent varier selon la langue.",
            "Une bonne performance ne prouve pas une détection universelle des fake news.",
            "Le résultat est une estimation algorithmique, pas une vérification factuelle.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Train size: {len(splits['train'])}")
    print(f"Dev size: {len(splits['dev'])}")
    print(f"Test size: {len(splits['test'])}")
    print(f"French zero-shot size: {len(splits['fr_zeroshot'])}")
    print(f"Class distribution: {splits['train']['label'].value_counts().to_dict()}")
    print(f"Configuration: {config.as_dict()}")
    print(f"Dev metrics: {metrics_dev}")
    print(f"Test metrics: {metrics_test}")
    print(f"French zero-shot metrics: {metrics_zeroshot}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
