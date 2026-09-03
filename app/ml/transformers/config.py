"""Configuration reproductible de l'expérience XLM-RoBERTa."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class XLMRobertaConfig:
    """Hyperparamètres et chemins de l'expérience Transformer."""

    model_name: str = "xlm-roberta-base"
    model_version: str = "xlm_roberta_v1"
    dataset_name: str = "X-FACT"
    dataset_version: str = "x_fact_v1"
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42

    def as_dict(self) -> dict[str, object]:
        """Retourne une représentation sérialisable."""
        return asdict(self)


def require_transformer_dependencies() -> tuple[object, object]:
    """Charge les dépendances optionnelles ou explique comment les installer."""
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "XLM-RoBERTa nécessite PyTorch et Transformers. "
            "Installez-les dans un environnement Python 3.11 compatible GPU "
            "avant d'exécuter cette phase."
        ) from error
    return torch, transformers
