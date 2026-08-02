# 00 — Inventaire du projet avant refactoring

## État de sécurisation

Inventaire établi le 2 août 2026 avant toute suppression.

| Contrôle | Résultat | Conséquence |
|---|---|---|
| Dépôt Git | Aucun dossier `.git`; `git status` échoue | Impossible de créer la branche recommandée ou de distinguer fichiers suivis/non suivis |
| Sauvegarde | `dist/pre_refactor_2026-08-02.zip`, archive testée sans erreur | Tous les fichiers de code, tests, notebooks, analyses, rapports, modèles et résultats restent récupérables |
| Données brutes | `train.csv` et `test.csv` présents | Lecture seule pendant le refactoring |
| SHA-256 train | `66ab317bb5fcc0df0e248127a25159f1dff9c3b8b16058281ecd6107b067f69b` | Garde-fou d'intégrité |
| SHA-256 test | `4dd388277253326c4155a8c87abac19fa4a70339675af9e4682affe4c1345956` | Garde-fou d'intégrité |
| Analyse archivée | 10 fichiers dans `docs.old/` | Dossier conservé intact |
| Tests initiaux | 337 réussis, 11 avertissements, couverture 74 % | Baseline fonctionnelle avant simplification |

## Sources obligatoires consultées

- énoncé officiel : `/Users/surelmanda/GameLoft/description_Gameloft_DS_technical_test.pdf`, 6 pages ;
- transcription de contrôle : `/Users/surelmanda/Downloads/Gameloft_Data_Science_Technical_Test_FULL.md` ;
- rapport historique : `Gameloft-Data-Science-Overview/GamePlayerAnalysis-Overview.{tex,pdf}` ;
- `README.md`, `pyproject.toml`, `poetry.lock` et `config/settings.dev.yaml` ;
- 17 notebooks, dont neuf notebooks EDA ;
- 45 fichiers Python sous `src/`, sept scripts et 31 fichiers de tests ;
- 17 analyses actives, 10 analyses archivées et 97 artefacts analytiques sous `reports/` ;
- les 10 artefacts sous `models/`, le CSV de soumission et les deux CSV préparés.

Le PDF du dossier `Gameloft-Data-Science-Overview/` n'est pas l'énoncé : c'est
un rapport produit lors d'une itération précédente. L'énoncé officiel demande
un notebook présentant Data Cleaning, Data Analysis & Visualization, Feature
Engineering et Modeling, avec une forte priorité donnée au raisonnement, aux
commentaires et à la lisibilité.

## Volumétrie du projet

| Zone | Fichiers pertinents | Volume logique | Rôle actuel |
|---|---:|---:|---|
| `src/` | 45 Python | 7 352 lignes | Package en quatre couches de type Clean Architecture |
| `tests/` | 31 Python | 3 567 lignes, 337 tests | Contrats de couches, unités, persistance et ML |
| `scripts/` | 7 Python | 3 376 lignes | Trois pipelines analytiques et quatre points d'entrée ML |
| `notebooks/` | 17 notebooks | 1,7 Mo | Parcours éclaté en huit étapes et neuf EDA |
| `docs/` | 17 Markdown | 2 515 lignes | Deuxième analyse et consolidation |
| `docs.old/` | 10 Markdown | 1 024 lignes | Première analyse, archive immuable |
| `reports/` | 97 fichiers | tables et figures | Preuves produites par trois scripts analytiques |
| `models/` | 10 fichiers | bundle CatBoost et métadonnées | Bundle historique actuellement incohérent |

## Arborescence fonctionnelle actuelle

```text
data/raw                         sources immuables
data/processed                   CSV générés, contrat courant
data/output                      soumission historique
notebooks/                       huit notebooks de workflow
notebooks/eda/                   neuf notebooks exploratoires
src/game_player_analysis/
  application/                   ports et cas d'usage
  domain/                        entités et contrats
  infrastructure/                pandas, ML, artefacts, graphiques
  presentation/                  composition pour notebooks/scripts
  config/                        paramètres Pydantic/YAML
scripts/                         analyse, évaluation, entraînement, inférence
models/                          modèle et preuves historiques
reports/                         sorties d'analyses précédentes
docs/ et docs.old/               analyses actives et archivées
```

## État des résultats et artefacts

| Élément | État vérifié | Risque |
|---|---|---|
| `data/processed/train_processed.csv` | hash `e1438606…`, 39 colonnes | Ne correspond pas au benchmark sauvegardé |
| `models/selected_features.json` | 17 candidates, 16 retenues, inclut `combatActivity` | Contrat courant uniquement |
| `models/feature_schema.json` | Attend `killRankPercentile` | Colonne absente du CSV préparé actuel |
| `models/model_benchmark.json` | CatBoost MAE 0,06037 sur hash `89ba33b9…` | Preuve historique, pas résultat actif |
| `models/model_manifest.json` | Bundle `untuned_benchmark_winner` | Incompatible avec la sélection courante |
| `data/output/submission.csv` | 5 000 lignes historiques | Ne doit pas être présenté comme reproductible depuis l'état courant |

## Décision de sécurisation

Le projet peut être refactorisé sans toucher à `data/raw` ni à `docs.old`.
L'archive de restauration remplace temporairement la protection qu'aurait
fournie une branche Git. Aucun artefact historique ne sera qualifié de
résultat final avant une reproduction sur un contrat unique.
