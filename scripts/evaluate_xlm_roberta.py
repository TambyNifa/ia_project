"""Évaluation d'un checkpoint XLM-RoBERTa."""

import argparse
from pathlib import Path

from app.ml.transformers.evaluate import evaluate_xlm_roberta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["test", "fr_zeroshot"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/tena_ai_xlm_roberta_v1/best_model"),
    )
    args = parser.parse_args()
    print(evaluate_xlm_roberta(args.checkpoint, args.split))
