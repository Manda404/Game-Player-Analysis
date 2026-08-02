# 10 — Reproductibilité et artefacts

## Source unique

Le script [`analyze_raw_data_from_scratch.py`](../scripts/analyze_raw_data_from_scratch.py) lit uniquement :

- `data/raw/train.csv` — SHA-256 `66ab317bb5fcc0df0e248127a25159f1dff9c3b8b16058281ecd6107b067f69b` ;
- `data/raw/test.csv` — SHA-256 `4dd388277253326c4155a8c87abac19fa4a70339675af9e4682affe4c1345956`.

Il ne lit aucun notebook, modèle, rapport ou conclusion préexistante et ne modifie jamais `data/raw`.

## Installation

Depuis la racine du projet :

```bash
poetry install --no-interaction
```

Les versions sont verrouillées dans `poetry.lock`.

## Exécution

Analyse complète :

```bash
poetry run python scripts/analyze_raw_data_from_scratch.py --section all
```

Exécution incrémentale, identique à la documentation continue :

```bash
poetry run python scripts/analyze_raw_data_from_scratch.py --section quality
poetry run python scripts/analyze_raw_data_from_scratch.py --section relationships
poetry run python scripts/analyze_raw_data_from_scratch.py --section kpi
```

Tous les résultats sont écrits dans `reports/independent_raw_analysis/` : tables CSV/JSON auditables et quatre figures PNG. Le clustering utilise `random_state=42`; sa stabilité est en outre testée sur cinq autres seeds.

## Correspondance documentation–artefacts

| Document | Principaux artefacts sources |
|---|---|
| `01_data_inventory.md` | lecture directe des deux CSV |
| `02_data_quality_analysis.md` | `data_quality_checks.csv`, `date_quality.json`, `target_quantisation_check.json` |
| `03_univariate_analysis.md` | `numeric_univariate_summary.csv`, `categorical_frequencies.csv` |
| `04_feature_relationships.md` | corrélations, VIF, associations, interaction, dérive |
| `05_target_analysis.md` | résumé/bins cible, relations feature-target, profils par bins et mode |
| `06_kpi_analysis.md` | `kpi_summary.csv`, KPI par mois et mode |
| `07_player_segmentation.md` | sélection k, profils, stabilité, cible et composition par mode |

## Limites de reproductibilité

- Les p-values peuvent varier à la dernière décimale entre versions de SciPy, sans affecter les conclusions.
- Le score silhouette est calculé sur un échantillon déterministe de 5 000 lignes pour maîtriser le coût ; les labels K-means restent déterministes avec les versions verrouillées.
- Si les hashes sources changent, les résultats ne doivent pas être comparés sans versionner le nouveau snapshot.
