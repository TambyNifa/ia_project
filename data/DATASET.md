# Dataset TENA AI

## Schéma

Chaque observation contient `id`, `title`, `text`, `language`, `source`,
`publication_date`, `label` et `created_at`. Les champs `title`, `text`,
`language`, `label` et `id` ne peuvent pas être vides. Les dates utilisent le
format ISO 8601 lorsqu'elles sont disponibles.

## Labels

Les labels internes sont centralisés dans
[`app/core/labels.py`](../app/core/labels.py) :

- `PROBABLEMENT_FIABLE` (`0`) : signaux compatibles avec une information
  documentée, sans constituer une preuve ;
- `SUSPECT` (`1`) : signaux nécessitant une vérification approfondie ;
- `DIFFICILE_A_DETERMINER` (`2`) : informations insuffisantes ou contradictoires.

Ces classes sont des estimations algorithmiques et ne déterminent jamais la
vérité d'une information.

## Langues

Les valeurs acceptées sont `fr` (français), `mg` (malgache), `mixed`
(français/malgache) et `unknown`. La détection initiale repose sur quelques
marqueurs lexicaux : elle est volontairement prudente, limitée pour le
malgache et remplaçable par un modèle spécialisé.

## Qualité et reproductibilité

- `raw/` conserve les fichiers originaux sans modification ;
- `external/` contient les données obtenues de sources externes ;
- `interim/` contient les étapes intermédiaires ;
- `processed/` contient les données prêtes pour l'entraînement ;
- toute correction ou exclusion doit être signalée dans le rapport de validation ;
- les doublons d'identifiants, valeurs manquantes, textes vides, langues et
  labels invalides sont rejetés du comptage des lignes valides ;
- le fichier `raw/demo_dataset.csv` est exclusivement synthétique et destiné
  aux tests : il ne représente pas un dataset réel de Madagascar.

Les futurs datasets devront documenter leur URL ou organisme source, leur date
de collecte, leur licence et les éventuelles opérations de transformation.
