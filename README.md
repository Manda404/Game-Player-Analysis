# Game Player Analysis

Reproducible Data Science analysis for the Gameloft technical assessment: predict
`winRankPercentage`, a player's team's normalized final ranking after a Battle
Royale match (`0` for last place, `1` for first place).

The primary deliverable is the executed
[`notebooks/game_player_analysis.ipynb`](notebooks/game_player_analysis.ipynb)
notebook. Starting from the raw CSV files, it covers Data Cleaning, Analysis &
Visualization, Feature Engineering, Modeling, interpretability, error analysis,
and inference.

## Verified result

| Model / scenario | MAE | RMSE | R² |
|---|---:|---:|---:|
| CatBoost, final grouped holdout | **0.06080** | 0.08660 | 0.92083 |
| CatBoost, development GroupKFold | 0.06145 ± 0.00109 | 0.08646 | 0.92083 |
| Behaviour without `killRank`, GroupKFold | 0.09266 | 0.12924 | 0.82314 |
| Linear Ridge, GroupKFold | 0.08831 | 0.12225 | 0.84177 |
| Constant median, GroupKFold | 0.26788 | 0.30781 | -0.00294 |

The audited initial comparison uses library-default learning parameters. CatBoost
outperforms XGBoost by 0.00200 MAE on average and in each of the five folds. The
earlier XGBoost winner came from customized initial configurations; replaying
them gives it only a negligible 0.000054 advantage. Eight CatBoost tuning trials
did not improve on the default configuration and were rejected.

## Key considerations

- Each row describes a player, while the target is the team's score repeated for
  the observed players.
- `killRank` is post-match information. Its use means that the final model must
  not be presented as an early-game prediction.
- The primary split is 5-fold GroupKFold on `gameId`, with no raw ID or
  conservative group shared across partitions. A grouped holdout of 9,872 rows,
  frozen before this audit's selection stage, provides the cycle's final
  evaluation.
- A sensitivity study over 3, 5, 7, and 10 folds confirms CatBoost in all 25
  combined folds. Five folds are retained: about 8,026 validation rows per fold;
  seven folds cost 52% more for only 0.000189 nominal MAE improvement.
- The `date` column is officially the match date, yet 100% of valid multi-row
  `gameId`s contain several dates. The pseudo-temporal split is therefore only a
  purged stress test.
- Nearly 98.82% of `(gameId, teamId)` pairs are singletons. Team- and
  lobby-level aggregates were rejected as indefensible.
- Train/test drift is low for measured variables: maximum numeric PSI is 0.00509,
  maximum categorical PSI is 0.00844, and adversarial-validation ROC AUC is
  0.49325. Performance drift remains unknown without test targets.

## Installation

Python 3.11 to 3.14 and Poetry are supported:

```bash
poetry install --with dev
```

The official CSV files are not versioned. Place them here:

```text
data/raw/train.csv
data/raw/test.csv
```

## Reproducing the analysis

Execute the complete notebook:

```bash
poetry run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/game_player_analysis.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Run the command-line pipeline:

```bash
poetry run python scripts/run_analysis.py
```

## Private Streamlit interface

The interface offers an accessible overview of the analysis, private CSV upload,
exploration, validation and SHAP diagnostics, bounded CatBoost tuning, and
prediction export. Uploaded files are never written by the app: they stay in the
Streamlit session memory.

```bash
poetry run streamlit run app/app.py
```

Upload a semicolon-separated official train file containing
`winRankPercentage`, then optionally a same-schema test file without the target.
The model is explicitly post-match: `killRank` is required and must not be
presented as an early-game signal.

An evaluated CatBoost variant is compared with the reference model on the same
grouped holdout. It becomes the active session model only when its MAE is
strictly lower; subsequent predictions then use that variant. This adoption does
not modify the repository or published model, preventing a public-app visitor
from overwriting a reproducible version.

## CI/CD and Streamlit deployment

Continuous integration is defined in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml): every pull request to
`main` and every push to `main` run tests, Black, and Flake8 with Python 3.12 in
GitHub Actions.

Continuous deployment is handled natively by Streamlit Community Cloud once the
app is linked to the repository: new commits to the tracked branch are deployed
automatically. Theme and upload-limit configuration are versioned in
[`.streamlit/config.toml`](.streamlit/config.toml).

To create the app once in Streamlit Community Cloud:

1. In your Streamlit workspace, click **Create app**.
2. Choose `Manda404/Game-Player-Analysis`, branch `main`, and entry point
   `app/app.py`.
3. Keep Python 3.12 and click **Deploy**.

The application requires no secrets. Visitor CSV files remain in memory, so they
are neither committed to GitHub nor sent to an external API. Consult the
[official Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
if you want to select a custom URL.

Run inference from a new official CSV:

```bash
poetry run python scripts/predict_from_csv.py data/raw/test.csv \
  --model-path artifacts/model.joblib \
  --output-path data/output/submission.csv
```

Quality checks:

```bash
poetry run pytest
poetry run black --check app src tests scripts
poetry run flake8 app src tests scripts
```

## Architecture

```text
src/game_player_analysis/
├── data.py             reading, schema, and fingerprints
├── cleaning.py         business sentinels
├── analysis.py         quality, KPIs, and profiles
├── features.py         single 15/16-feature contract
├── validation.py       folds, holdouts, and leakage audits
├── modeling.py         baselines, ensembles, tuning, and bundle
├── evaluation.py       metrics, subgroups, and importance
├── inference.py        raw CSV → validated submission
├── visualization.py    decision-oriented figures, including drift and validation
└── pipeline.py         reproducible orchestration
```

## Outputs

- model and contract: `artifacts/model.joblib`, `model_manifest.json`;
- detailed metrics: `artifacts/metrics/`;
- figures: `artifacts/figures/`;
- tuning decision: `artifacts/metadata/tuning_decision.json`;
- final decision: `artifacts/metadata/final_selection_decision.json`;
- CLI log: `artifacts/logs/analysis.log`;
- predictions: `data/output/submission.csv`.

The [final report](docs/final_report.md) summarizes the conclusions. Initial
audits and the coverage matrix are in [`docs/review/`](docs/review/). The
pre-refactoring state remains available in `dist/pre_refactor_2026-08-02.zip`.
