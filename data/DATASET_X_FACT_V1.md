# X-FACT V1

## Provenance

- Nom officiel : X-FACT: A New Benchmark Dataset for Multilingual Fact Checking
- Source officielle : <https://huggingface.co/datasets/utahnlp/x-fact>
- Repository : <https://github.com/utahnlp/x-fact>
- Version interne : `x_fact_v1`
- Préparation : générée par `python -m scripts.import_x_fact`
- Type : affirmations vérifiées, et non articles complets
- Date de préparation : enregistrée dans `quality_report.json`

La licence déclarée du dataset est MIT. **La licence déclarée du dataset ne
garantit pas automatiquement les droits de réutilisation des contenus provenant
des sources tierces.** Les liens, extraits de preuves et contenus des sites
d'origine doivent rester soumis à leurs propres conditions.

## Splits

`train`, `dev` et `test` d'`all_languages` sont les seuls splits du pipeline
principal. `fr/zeroshot` est conservé séparément sous le nom `fr_zeroshot` et
n'est jamais utilisé pour l'entraînement.

Le script conserve les CSV sources dans `data/external/x_fact/` et les données
harmonisées dans `data/processed/x_fact_v1/`. Les fichiers volumineux sont
ignorés par Git.

## Schéma et absence de titre

X-FACT fournit notamment `language`, `site`, `claim`, `label`, des preuves et
des liens. Il ne fournit pas de titre d'article. Le champ interne `title` reste donc vide ; aucune valeur n'est inventée.
L'affirmation `claim` est placée dans `text`. Les dates utilisent `claimDate`
lorsqu'elle est disponible et restent manquantes sinon. La colonne
`original_language` conserve la langue X-FACT ; la colonne interne `language`
utilise `unknown` pour les langues non prises en charge par le schéma TENA AI,
sans prétendre les identifier comme français ou malgache.

## Mapping des labels

Seuls les mappings suivants sont approuvés pour V1 :

| Label X-FACT | Label TENA AI | Action |
|---|---|---|
| `true` | `PROBABLEMENT_FIABLE` | importé |
| `false` | `SUSPECT` | importé |

Tous les autres labels (`mostly true`, `partly true/misleading`, etc.) sont
exclus : ils ne sont pas convertis automatiquement car leur interprétation
dans les trois classes TENA AI n'a pas été validée. La classe
`DIFFICILE_A_DETERMINER` n'est pas créée artificiellement.

## Langues et limites

Les langues sont celles déclarées par X-FACT dans chaque ligne. Le français est
présent uniquement dans `fr/zeroshot` et le malgache (`mg`) est absent. X-FACT
ne représente donc pas Madagascar. Le modèle issu de V1 sera un classificateur
expérimental d'affirmations et de contenus textuels multilingues, adaptable
ultérieurement au contexte malgache.

Le rapport JSON détaille les volumes, labels exclus, doublons, textes vides,
erreurs et distributions réelles après import.
