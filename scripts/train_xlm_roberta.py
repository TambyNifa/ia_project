"""Entraînement XLM-RoBERTa sur machine GPU."""

import argparse
from pathlib import Path

from app.ml.transformers.config import XLMRobertaConfig
from app.ml.transformers.train import train_xlm_roberta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tena_ai_xlm_roberta_v1"),
    )
    args = parser.parse_args()
    defaults = XLMRobertaConfig()
    config = XLMRobertaConfig(num_epochs=args.epochs or defaults.num_epochs)
    train_xlm_roberta(
        config, args.output_dir, args.max_train_samples, args.max_dev_samples
    )
