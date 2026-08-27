"""Inférence locale avec la baseline sauvegardée."""

import json
from pathlib import Path
from typing import Any

from app.core.labels import Label
from app.ml.preprocessing.text import clean_text
from app.ml.training.baseline import load_model

DISCLAIMER = (
    "Cette analyse est une estimation algorithmique et ne constitue pas une "
    "vérification factuelle."
)


class BaselinePredictor:
    """Charge une baseline et expose une prédiction locale exploitable."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model = load_model(model_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

    def predict(self, title: str, text: str) -> dict[str, Any]:
        """Prétraite et prédit un titre et un texte non vides."""
        clean_title = clean_text(title)
        clean_body = clean_text(text)
        if not clean_title and not clean_body:
            raise ValueError("Le titre ou le texte doit contenir du contenu.")
        combined_text = f"{clean_title} {clean_body}".strip()
        probabilities = self.model.predict_proba([combined_text])[0]
        classifier = self.model.named_steps["classifier"]
        probability_by_label = {label.name: 0.0 for label in Label}
        for class_name, probability in zip(classifier.classes_, probabilities):
            probability_by_label[str(class_name)] = float(probability)
        prediction = str(self.model.predict([combined_text])[0])
        confidence = probability_by_label[prediction]
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probability_by_label,
            "model_version": self.metadata.get("version", "unknown"),
            "dataset_type": self.metadata.get("dataset_type", "unknown"),
            "disclaimer": DISCLAIMER,
        }
