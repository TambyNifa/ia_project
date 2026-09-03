"""Inférence XLM-RoBERTa avec probabilités softmax."""

from pathlib import Path

from app.ml.preprocessing.text import clean_text
from app.ml.transformers.config import require_transformer_dependencies


def predict_xlm_roberta(title: str, text: str, checkpoint: Path) -> dict[str, object]:
    """Retourne une prédiction réelle d'un checkpoint entraîné."""
    if not (title.strip() or text.strip()):
        raise ValueError("Le titre ou le texte doit contenir du contenu.")
    torch, _ = require_transformer_dependencies()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(
        f"{clean_text(title)} {clean_text(text)}".strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    labels = ["PROBABLEMENT_FIABLE", "SUSPECT"]
    index = int(torch.tensor(probabilities).argmax())
    return {
        "prediction": labels[index],
        "confidence": float(probabilities[index]),
        "probabilities": dict(zip(labels, probabilities)),
        "model": "XLM-RoBERTa",
        "dataset": "X-FACT V1",
        "disclaimer": "Cette analyse est une estimation algorithmique et ne constitue pas une vérification factuelle.",
    }
