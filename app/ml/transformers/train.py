"""Entraînement XLM-RoBERTa optionnel pour un environnement GPU."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.ml.transformers.config import (
    XLMRobertaConfig,
    require_transformer_dependencies,
)
from app.ml.transformers.dataset import (
    encode_labels,
    load_transformer_split,
    tokenize_dataframe,
)


def train_xlm_roberta(
    config: XLMRobertaConfig,
    output_dir: Path,
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
) -> dict[str, object]:
    """Entraîne avec Trainer et retourne les métriques de l'expérience."""
    torch, _ = require_transformer_dependencies()
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    train = load_transformer_split("train")
    dev = load_transformer_split("dev")
    if max_train_samples:
        train = train.head(max_train_samples).copy()
    if max_dev_samples:
        dev = dev.head(max_dev_samples).copy()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "PROBABLEMENT_FIABLE", 1: "SUSPECT"},
        label2id={"PROBABLEMENT_FIABLE": 0, "SUSPECT": 1},
    )
    train_tokens = tokenize_dataframe(train, tokenizer, config.max_length)
    dev_tokens = tokenize_dataframe(dev, tokenizer, config.max_length)

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, labels):
            self.tokens = tokens
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.tokens.items()}
            item["labels"] = self.labels[index]
            return item

    train_dataset = TokenizedDataset(
        train_tokens, encode_labels(train["label"].tolist())
    )
    dev_dataset = TokenizedDataset(dev_tokens, encode_labels(dev["label"].tolist()))

    def compute_metrics(prediction):
        predictions = np.argmax(prediction.predictions, axis=-1)
        labels = prediction.label_ids
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "weighted_f1": f1_score(labels, predictions, average="weighted"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=config.seed,
    )
    started = perf_counter()
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    metadata = {
        **config.as_dict(),
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "trained_classes": ["PROBABLEMENT_FIABLE", "SUSPECT"],
        "not_trained_classes": ["DIFFICILE_A_DETERMINER"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "training_duration_seconds": round(perf_counter() - started, 3),
        "metrics_dev": metrics,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
