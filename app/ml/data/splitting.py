"""Découpage reproductible des données de classification."""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Jeux d'entraînement, validation et test."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    stratified: bool
    warning: str | None = None


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    random_state: int = 42,
) -> DatasetSplit:
    """Sépare les données sans recouvrement, avec fallback pour petits jeux."""
    if len(dataframe) < 3:
        raise ValueError("Au moins trois lignes sont nécessaires pour un split.")
    shuffled = dataframe.sample(frac=1, random_state=random_state)
    class_counts = shuffled[label_column].value_counts()
    stratified_possible = len(class_counts) > 1 and class_counts.min() >= 3

    if stratified_possible:
        train, remainder = train_test_split(
            shuffled,
            test_size=0.30,
            random_state=random_state,
            stratify=shuffled[label_column],
        )
        validation, test = train_test_split(
            remainder, test_size=0.50, random_state=random_state, stratify=None
        )
        return DatasetSplit(train, validation, test, stratified=True)

    test_size = max(1, round(len(shuffled) * 0.15))
    validation_size = max(1, round(len(shuffled) * 0.15))
    if test_size + validation_size >= len(shuffled):
        test_size, validation_size = 1, 1
    train, remainder = train_test_split(
        shuffled,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=None,
    )
    validation = remainder.iloc[:validation_size].copy()
    test = remainder.iloc[validation_size:].copy()
    warning = (
        "Split non stratifié utilisé : le dataset est trop petit ou certaines "
        "classes ont moins de trois exemples."
    )
    LOGGER.warning(warning)
    return DatasetSplit(train, validation, test, stratified=False, warning=warning)
