"""Évaluation d'un checkpoint XLM-RoBERTa."""

from pathlib import Path

from app.core.labels import Label
from app.ml.evaluation.metrics import evaluate_predictions
from app.ml.transformers.config import require_transformer_dependencies
from app.ml.transformers.dataset import (
    encode_labels,
    load_transformer_split,
    tokenize_dataframe,
)


def evaluate_xlm_roberta(checkpoint: Path, split: str) -> dict[str, object]:
    """Évalue un checkpoint sur test ou fr_zeroshot."""
    torch, _ = require_transformer_dependencies()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dataframe = load_transformer_split(split)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokens = tokenize_dataframe(dataframe, tokenizer, 256)
    with torch.no_grad():
        outputs = model(
            **{key: torch.tensor(value, device=device) for key, value in tokens.items()}
        )
    predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
    labels = encode_labels(dataframe["label"].tolist())
    return evaluate_predictions(
        [
            Label.PROBABLEMENT_FIABLE.name if value == 0 else Label.SUSPECT.name
            for value in labels
        ],
        [
            Label.PROBABLEMENT_FIABLE.name if value == 0 else Label.SUSPECT.name
            for value in predictions
        ],
        [Label.PROBABLEMENT_FIABLE.name, Label.SUSPECT.name],
    )
