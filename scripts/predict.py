"""Teste localement la baseline TENA AI."""

import argparse
from pathlib import Path

from app.core.config import settings
from app.ml.inference.baseline import BaselinePredictor

EXAMPLES = (
    (
        "Exemple français",
        "Information officielle",
        "Le ministère publie une annonce vérifiable.",
    ),
    (
        "Exemple malgache",
        "Fampitandremana",
        "Ny vaovao dia mbola tsy voamarina.",
    ),
)


def display_result(result: dict[str, object]) -> None:
    """Affiche une prédiction avec ses probabilités et avertissements."""
    print("=" * 30)
    print("RÉSULTAT TENA AI")
    print("=" * 30)
    print(f"\nPrédiction :\n{result['prediction']}")
    print(f"\nConfiance :\n{float(result['confidence']):.2%}")
    print("\nProbabilités :")
    for label, probability in result["probabilities"].items():
        print(f"{label} : {float(probability):.2%}")
    print(f"\nVersion du modèle :\n{result['model_version']}")
    print(f"\nType de dataset :\n{result['dataset_type']}")
    print(f"\nAVERTISSEMENT :\n{result['disclaimer']}")
    if result["dataset_type"] == "demo":
        print(
            "\nLe modèle actuel a été entraîné avec un dataset synthétique de "
            "démonstration très limité. Les résultats sont uniquement techniques "
            "et non représentatifs des performances réelles."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tester la baseline TENA AI.")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Exécuter des exemples de démonstration prédéfinis.",
    )
    args = parser.parse_args()
    predictor = BaselinePredictor(
        Path(settings.model_path), Path(settings.model_metadata_path)
    )
    if args.example:
        for name, title, text in EXAMPLES:
            print(f"\n--- {name} (démonstration) ---")
            display_result(predictor.predict(title, text))
        return
    title = input("Titre : ").strip()
    text = input("Texte : ").strip()
    display_result(predictor.predict(title, text))


if __name__ == "__main__":
    main()
