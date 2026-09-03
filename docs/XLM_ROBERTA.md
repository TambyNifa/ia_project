# Préparation XLM-RoBERTa

## Objectif

Cette expérimentation compare une représentation Transformer multilingue à
TF-IDF. XLM-RoBERTa peut mieux partager des représentations entre langues,
mais son coût matériel est nettement supérieur et aucune amélioration n'est
garantie.

Le modèle utilisera les affirmations `claim` de X-FACT V1 dans `text`. X-FACT
ne fournit pas de titre d'article : aucun titre ne sera inventé.

## Données et classes

- `train` : entraînement ;
- `dev` : validation et sélection du checkpoint ;
- `test` : évaluation finale ;
- `fr_zeroshot` : évaluation séparée de généralisation en français.

Le modèle est binaire : `PROBABLEMENT_FIABLE` et `SUSPECT`. La classe
`DIFFICILE_A_DETERMINER` n'est pas créée artificiellement.

Le français n'est pas présent dans le train X-FACT V1 et le malgache est
absent. X-FACT ne représente pas Madagascar. Une performance élevée ne
constitue pas une preuve de détection générale des fake news : toute sortie
reste une estimation algorithmique, pas une vérification factuelle.

## Configuration prévue

La configuration par défaut se trouve dans
[`app/ml/transformers/config.py`](../app/ml/transformers/config.py) :

```text
model_name = xlm-roberta-base
max_length = 256
batch_size = 8
learning_rate = 2e-5
num_epochs = 3
weight_decay = 0.01
warmup_ratio = 0.1
seed = 42
```

## Environnement recommandé

- Python 3.11 (plutôt que Python 3.14 pour la compatibilité des frameworks) ;
- GPU CUDA avec au moins 8 Go de VRAM pour un pilote raisonnable ;
- PyTorch 2.2 ou version compatible avec le CUDA du système ;
- Transformers 4.45 ou version compatible ;
- Datasets 2.20 ou version compatible ;
- Accelerate.

Les versions optionnelles sont listées dans
[`requirements-transformers.txt`](../requirements-transformers.txt). Elles ne
sont volontairement pas installées dans l'environnement local actuel.

## Procédure Google Colab (Phase 5C)

1. Dans Colab, choisissez **Runtime > Change runtime type > GPU**.
2. Importez le dépôt dans `/content/IA-project` (clone Git ou téléversement).
3. Exécutez `%cd /content/IA-project`.
4. Installez les dépendances et vérifiez les versions :

```bash
python -m pip install -r requirements-transformers.txt
python -m scripts.setup_transformer_env
```

5. Exécutez le smoke test du notebook : il télécharge `xlm-roberta-base`,
   tokenise deux exemples de `dev` et vérifie un forward pass.
6. Lancez le pilote :

```bash
python -m scripts.train_xlm_roberta --max-train-samples 500 --max-dev-samples 200 --epochs 1
```

Le pilote produit la loss et les métriques de validation (`accuracy`,
`macro_f1`) et vérifie la sauvegarde. Il n'est pas un résultat scientifique.

7. Après validation du pilote, lancez l'entraînement complet :

```bash
python -m scripts.train_xlm_roberta
```

Le meilleur checkpoint est sélectionné exclusivement selon le Macro F1 de
`dev`. Les fichiers sont écrits dans
`models/tena_ai_xlm_roberta_v1/`, notamment `best_model/` et
`training_metrics.json`.

8. Évaluez séparément les deux jeux jamais utilisés pour sélectionner le modèle :

```bash
python -m scripts.evaluate_xlm_roberta test
python -m scripts.evaluate_xlm_roberta fr_zeroshot
```

9. Archivez et téléchargez `models/tena_ai_xlm_roberta_v1/` depuis Colab. Ce
   répertoire contient le modèle, le tokenizer et la configuration ; les
   métriques sont dans `training_metrics.json`.

L'inférence s'effectue après entraînement avec
`python -m scripts.predict_xlm_roberta`. Avant cela, le script échoue
explicitement si le checkpoint n'existe pas et n'affiche aucun résultat fictif.

## État Phase 5A/5B

L'entraînement n'a pas été exécuté localement dans cet environnement en raison
des contraintes matérielles. Aucun poids XLM-RoBERTa n'a été téléchargé.
