# TENA AI

Système d'aide à l'analyse de la fiabilité d'informations potentiellement trompeuses dans le contexte de Madagascar.

> Les résultats sont des estimations algorithmiques et ne constituent pas une vérification factuelle ni une détermination de la vérité.

## Démarrage rapide

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

L'API est disponible sur `http://127.0.0.1:8000` et sa documentation sur `/docs`.

## Structure

- `app/api/` : application FastAPI et routes HTTP
- `app/core/` : configuration et journalisation
- `app/ml/` : prétraitement, entraînement, évaluation et inférence
- `app/database/` : modèles et accès aux données
- `app/schemas/` : schémas Pydantic
- `frontend/` : interface Streamlit
- `data/` : données brutes, traitées et externes
- `models/` : artefacts de modèles entraînés
- `tests/` : tests unitaires et d'intégration
- `scripts/` : scripts d'exécution

## Développement

```powershell
ruff check .
black --check .
pytest
```

Les données et modèles réels seront ajoutés dans les phases suivantes avec leur provenance documentée.

## Validation des données

Le dataset synthétique de test peut être contrôlé avec :

```powershell
python -m scripts.validate_dataset data/raw/demo_dataset.csv
```

La commande affiche un rapport JSON et retourne un code différent de zéro si
des lignes invalides sont détectées.

## Baseline ML

La baseline technique TF-IDF + Logistic Regression se lance avec :

```powershell
python -m scripts.train_baseline
```

Elle utilise une copie des lignes valides du dataset, conserve les jeux de
validation et de test séparés, puis sauvegarde le pipeline complet dans
`models/` avec ses métadonnées. Le dataset de démonstration est trop petit
pour produire des performances représentatives ; les métriques affichées sont
uniquement techniques.
