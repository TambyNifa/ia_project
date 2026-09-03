"""Pipeline Linear SVM pour la comparaison des modèles classiques."""

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.ml.training.baseline import combine_text


@dataclass(frozen=True)
class SVMConfig:
    """Hyperparamètres du pipeline Linear SVM."""

    C: float = 1.0
    max_features: int | None = 5000
    min_df: int = 1
    max_df: float = 1.0
    ngram_range: tuple[int, int] = (1, 2)
    class_weight: str | None = "balanced"
    random_state: int = 42

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_svm_pipeline(config: SVMConfig) -> Pipeline:
    """Construit le pipeline TF-IDF + LinearSVC."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    min_df=config.min_df,
                    max_df=config.max_df,
                    ngram_range=config.ngram_range,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=config.C,
                    class_weight=config.class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def train_svm(train_dataframe: pd.DataFrame, config: SVMConfig) -> Pipeline:
    """Entraîne le SVM uniquement sur le dataframe fourni."""
    pipeline = build_svm_pipeline(config)
    pipeline.fit(combine_text(train_dataframe), train_dataframe["label"].tolist())
    return pipeline


def save_svm_model(pipeline: Pipeline, path: Path) -> None:
    """Sauvegarde le pipeline SVM complet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_svm_model(path: Path) -> Pipeline:
    """Charge un pipeline SVM sauvegardé."""
    if not path.exists():
        raise FileNotFoundError(f"Modèle SVM introuvable : {path}")
    return joblib.load(path)
