"""Entraîne et compare la baseline Linear SVM sur X-FACT V1."""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.labels import Label
from app.core.logging import configure_logging
from app.ml.evaluation.metrics import evaluate_predictions
from app.ml.training.baseline import combine_text
from app.ml.training.svm import SVMConfig, save_svm_model, train_svm
from scripts.train_x_fact_baseline import load_split

MODEL_PATH = Path("models/tena_ai_x_fact_v1_linear_svm.joblib")
METADATA_PATH = Path("models/tena_ai_x_fact_v1_linear_svm_metadata.json")
TRAINED_LABELS = [Label.PROBABLEMENT_FIABLE.name, Label.SUSPECT.name]
C_VALUES = (0.1, 1.0, 2.0)


def evaluate_svm(pipeline, dataframe):
    """Évalue un SVM en utilisant ses scores de décision, jamais des probabilités."""
    expected = dataframe["label"].tolist()
    predicted = pipeline.predict(combine_text(dataframe)).tolist()
    return evaluate_predictions(expected, predicted, TRAINED_LABELS)


def main() -> None:
    configure_logging()
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    zero_shot = load_split("fr_zeroshot")
    dev_results = {}
    for c_value in C_VALUES:
        config = SVMConfig(C=c_value)
        candidate = train_svm(train, config)
        dev_results[str(c_value)] = evaluate_svm(candidate, dev)
        print(
            f"C={c_value} -> Accuracy={dev_results[str(c_value)]['accuracy']:.4f}, "
            f"Macro F1={dev_results[str(c_value)]['macro_f1']:.4f}, "
            f"Weighted F1={dev_results[str(c_value)]['weighted_f1']:.4f}"
        )
    best_c = max(C_VALUES, key=lambda value: dev_results[str(value)]["macro_f1"])
    config = SVMConfig(C=best_c)
    pipeline = train_svm(train, config)
    metrics_test = evaluate_svm(pipeline, test)
    metrics_zero_shot = evaluate_svm(pipeline, zero_shot)
    save_svm_model(pipeline, MODEL_PATH)
    logistic_metadata = json.loads(
        Path("models/tena_ai_x_fact_v1_metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "model_name": "tfidf_linear_svm",
        "model_version": "x_fact_v1_linear_svm",
        "algorithm": "LinearSVC",
        "dataset_name": "X-FACT",
        "dataset_version": "x_fact_v1",
        "dataset_type": "real",
        "training_date": datetime.now(UTC).isoformat(),
        "trained_classes": TRAINED_LABELS,
        "not_trained_classes": [Label.DIFFICILE_A_DETERMINER.name],
        "languages": sorted(train["original_language"].unique().tolist()),
        "hyperparameters": config.as_dict(),
        "c_candidates": list(C_VALUES),
        "dev_selection_metric": "macro_f1",
        "dev_results_by_c": dev_results,
        "metrics_dev": dev_results[str(best_c)],
        "metrics_test": metrics_test,
        "metrics_fr_zeroshot": metrics_zero_shot,
        "comparison": {
            "logistic_regression": {
                "test": logistic_metadata["metrics_test"],
                "fr_zeroshot": logistic_metadata["metrics_fr_zeroshot"],
            },
            "linear_svm": {
                "test": metrics_test,
                "fr_zeroshot": metrics_zero_shot,
            },
        },
        "split_sizes": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "fr_zeroshot": len(zero_shot),
        },
        "limitations": [
            "LinearSVC fournit des scores de décision, pas des probabilités.",
            "X-FACT contient des affirmations, pas des articles complets.",
            "Le dataset ne représente pas Madagascar et ne contient pas de malgache.",
            "Le français n'est pas présent dans l'entraînement ; il est évalué séparément.",
            "Le modèle est une classification binaire et ne constitue pas une vérification factuelle.",
        ],
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best C: {best_c}")
    print(f"Test metrics: {metrics_test}")
    print(f"French zero-shot metrics: {metrics_zero_shot}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
