"""Baseline TF-IDF et régression logistique."""

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineConfig:
    """Hyperparamètres modifiables de la baseline."""

    max_features: int | None = 5000
    min_df: int = 1
    max_df: float = 1.0
    ngram_range: tuple[int, int] = (1, 2)
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    random_state: int = 42

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def combine_text(dataframe: pd.DataFrame) -> list[str]:
    """Combine le titre et le texte nettoyés selon une règle unique."""
    return (
        (
            dataframe["clean_title"].fillna("").astype(str)
            + " "
            + dataframe["clean_text"].fillna("").astype(str)
        )
        .str.strip()
        .tolist()
    )


def build_pipeline(config: BaselineConfig) -> Pipeline:
    """Construit le pipeline vectorisation + classification."""
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
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def train_baseline(
    train_dataframe: pd.DataFrame,
    *,
    config: BaselineConfig,
) -> Pipeline:
    """Entraîne le pipeline exclusivement sur le jeu TRAIN."""
    texts = combine_text(train_dataframe)
    labels = train_dataframe["label"].tolist()
    pipeline = build_pipeline(config)
    pipeline.fit(texts, labels)
    return pipeline


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Sauvegarde le pipeline complet, vectoriseur inclus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    """Charge un pipeline précédemment sauvegardé."""
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
