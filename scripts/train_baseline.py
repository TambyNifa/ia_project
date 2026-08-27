"""Entraîne et évalue la baseline TF-IDF + Logistic Regression."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from app.core.config import settings
from app.core.labels import VALID_LABELS, Label
from app.core.logging import configure_logging
from app.ml.data.splitting import split_dataset
from app.ml.data.validation import get_valid_rows, validate_dataframe
from app.ml.evaluation.metrics import evaluate_predictions
from app.ml.preprocessing.text import prepare_dataframe
from app.ml.training.baseline import (
    BaselineConfig,
    combine_text,
    save_model,
    train_baseline,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET = Path("data/raw/demo_dataset.csv")


def main() -> None:
    configure_logging()
    dataset_path = DEFAULT_DATASET
    dataframe = pd.read_csv(dataset_path)
    report = validate_dataframe(dataframe)
    valid = prepare_dataframe(get_valid_rows(dataframe))
    if len(valid) < 6:
        LOGGER.warning(
            "Dataset de démonstration insuffisant : résultats uniquement techniques."
        )
    split = split_dataset(valid, random_state=settings.random_state)
    config = BaselineConfig(random_state=settings.random_state)
    started = perf_counter()
    pipeline = train_baseline(split.train, config=config)
    duration = perf_counter() - started

    validation_predictions = pipeline.predict(combine_text(split.validation))
    validation_metrics = evaluate_predictions(
        split.validation["label"].tolist(),
        validation_predictions.tolist(),
        sorted(VALID_LABELS),
    )
    test_predictions = pipeline.predict(combine_text(split.test))
    test_metrics = evaluate_predictions(
        split.test["label"].tolist(),
        test_predictions.tolist(),
        sorted(VALID_LABELS),
    )
    save_model(pipeline, Path(settings.model_path))
    metadata = {
        "model_name": "tfidf_logistic_regression",
        "version": "0.1.0",
        "metric": "macro_f1",
        "dataset_type": "demo",
        "dataset_path": str(dataset_path),
        "dataset_size": len(valid),
        "valid_rows": report["valid_rows"],
        "class_distribution": valid["label"].value_counts().to_dict(),
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "stratified_split": split.stratified,
        "model_configuration": config.as_dict(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_duration_seconds": round(duration, 4),
        "training_date": datetime.now(UTC).isoformat(),
        "label_mapping": {label.name: label.value for label in Label},
        "limitations": [
            "Dataset de démonstration synthétique",
            "Résultats non représentatifs des performances réelles",
            "Le split est non stratifié lorsque le dataset est trop petit",
        ],
    }
    metadata_path = Path(settings.model_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Dataset size: {len(valid)}")
    print(f"Training size: {len(split.train)}")
    print(f"Validation size: {len(split.validation)}")
    print(f"Test size: {len(split.test)}")
    print(f"Class distribution: {metadata['class_distribution']}")
    print(f"Configuration: {config.as_dict()}")
    print(f"Training duration: {duration:.4f}s")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Test metrics (final): {test_metrics}")


if __name__ == "__main__":
    main()
