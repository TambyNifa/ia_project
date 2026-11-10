"""Inférence XLM-RoBERTa sur un checkpoint entraîné."""

import argparse
from pathlib import Path

from app.ml.transformers.predict import predict_xlm_roberta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/tena_ai_xlm_roberta_v1/best_model"),
    )
    args = parser.parse_args()
    title = input("Titre : ")
    text = input("Texte : ")
    print(predict_xlm_roberta(title, text, args.checkpoint))
