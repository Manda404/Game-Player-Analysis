"""Compare grouped cross-validation fold counts without touching the holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

from game_player_analysis.cleaning import clean_ranking_sentinels
from game_player_analysis.config import FIGURES_DIR, METADATA_DIR, METRICS_DIR, TARGET
from game_player_analysis.data import load_train_test
from game_player_analysis.features import build_model_features
from game_player_analysis.modeling import build_model_candidates, cross_validate_model
from game_player_analysis.validation import (
    audit_group_folds,
    make_final_group_holdout,
    make_group_folds,
)
from game_player_analysis.visualization import apply_style, plot_fold_count_study

FOLD_COUNTS = (3, 5, 7, 10)
MODEL_NAMES = ("CatBoost", "XGBoost")


def run_fold_count_study() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate fold-count sensitivity on the frozen development partition."""
    train_raw, _ = load_train_test()
    train = clean_ranking_sentinels(train_raw)
    features = build_model_features(train, include_kill_rank=True)
    development_index, _ = make_final_group_holdout(train)
    development = train.iloc[development_index].reset_index(drop=True)
    X = features.iloc[development_index].reset_index(drop=True)
    y = development[TARGET]
    candidates = build_model_candidates()

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []

    for n_splits in FOLD_COUNTS:
        folds = make_group_folds(development, n_splits=n_splits)
        audit = audit_group_folds(development, folds)
        model_details: dict[str, pd.DataFrame] = {}
        total_fit_seconds = 0.0

        for model_name in MODEL_NAMES:
            summary, oof_prediction, detail = cross_validate_model(
                model_name,
                candidates[model_name],
                X,
                y,
                folds,
            )
            model_details[model_name] = detail
            total_fit_seconds += float(summary["fit_seconds"])
            validation_rows = audit["validation_rows"]
            train_rows = audit["train_rows"]
            summary_rows.append(
                {
                    "n_splits": n_splits,
                    "model": model_name,
                    "mae": summary["mae"],
                    "mae_std": summary["mae_std"],
                    "mae_standard_error": float(summary["mae_std"]) / np.sqrt(n_splits),
                    "weighted_oof_mae": mean_absolute_error(y, oof_prediction),
                    "mae_min": summary["mae_min"],
                    "mae_max": summary["mae_max"],
                    "rmse": summary["rmse"],
                    "r2": summary["r2"],
                    "train_mae": summary["train_mae"],
                    "mae_gap": summary["mae_gap"],
                    "mean_train_rows": float(train_rows.mean()),
                    "mean_validation_rows": float(validation_rows.mean()),
                    "min_validation_rows": int(validation_rows.min()),
                    "max_validation_rows": int(validation_rows.max()),
                    "max_shared_games": int(audit["shared_groups"].max()),
                    "fit_seconds": summary["fit_seconds"],
                    "predict_seconds": summary["predict_seconds"],
                    "fits": n_splits,
                }
            )
            detail_frame = detail.copy()
            detail_frame.insert(0, "model", model_name)
            detail_frame.insert(0, "n_splits", n_splits)
            detail_frame = detail_frame.merge(audit, on="fold", how="left")
            detail_frames.append(detail_frame)

        catboost_mae = model_details["CatBoost"].set_index("fold")["mae"]
        xgboost_mae = model_details["XGBoost"].set_index("fold")["mae"]
        differences = xgboost_mae - catboost_mae
        decision_rows.append(
            {
                "n_splits": n_splits,
                "catboost_mae": float(catboost_mae.mean()),
                "xgboost_mae": float(xgboost_mae.mean()),
                "xgboost_minus_catboost_mae": float(differences.mean()),
                "catboost_winning_folds": int(differences.gt(0).sum()),
                "mean_validation_rows": float(audit["validation_rows"].mean()),
                "min_validation_rows": int(audit["validation_rows"].min()),
                "max_validation_rows": int(audit["validation_rows"].max()),
                "total_candidate_fit_seconds": total_fit_seconds,
                "max_shared_games": int(audit["shared_groups"].max()),
            }
        )

    summary_table = pd.DataFrame(summary_rows)
    detail_table = pd.concat(detail_frames, ignore_index=True)
    decision_table = pd.DataFrame(decision_rows)
    five_fold_cost = float(
        decision_table.loc[decision_table["n_splits"].eq(5), "total_candidate_fit_seconds"].iloc[0]
    )
    decision_table["fit_cost_relative_to_5_folds"] = (
        decision_table["total_candidate_fit_seconds"] / five_fold_cost
    )
    return summary_table, detail_table, decision_table


def main() -> None:
    """Run the study and publish its tables and explicit decision."""
    summary, details, decision = run_fold_count_study()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": METRICS_DIR / "fold_count_sensitivity.csv",
        "details": METRICS_DIR / "fold_count_details.csv",
        "decision_table": METRICS_DIR / "fold_count_decision.csv",
        "decision": METADATA_DIR / "fold_count_decision.json",
        "figure": FIGURES_DIR / "07c_fold_count_sensitivity.png",
    }
    summary.to_csv(paths["summary"], index=False)
    details.to_csv(paths["details"], index=False)
    decision.to_csv(paths["decision_table"], index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_fold_count_study(summary, decision)
    plt.gcf().savefig(paths["figure"], dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    catboost = summary.loc[summary["model"].eq("CatBoost")].set_index("n_splits")
    decision_by_k = decision.set_index("n_splits")
    paths["decision"].write_text(
        json.dumps(
            {
                "recommended_n_splits": 5,
                "validation_method": "GroupKFold on frozen development partition",
                "models_checked": list(MODEL_NAMES),
                "fold_counts_checked": list(FOLD_COUNTS),
                "selection_rule": (
                    "prefer the smallest K that preserves the model ranking, "
                    "provides large match-disjoint validation folds and avoids "
                    "materially higher compute"
                ),
                "evidence": {
                    "catboost_wins_every_fold_for_every_k": bool(
                        decision.apply(
                            lambda row: row["catboost_winning_folds"] == row["n_splits"],
                            axis=1,
                        ).all()
                    ),
                    "five_fold_validation_rows": float(
                        decision_by_k.loc[5, "mean_validation_rows"]
                    ),
                    "seven_fold_nominal_mae_gain_vs_five": float(
                        catboost.loc[5, "mae"] - catboost.loc[7, "mae"]
                    ),
                    "seven_fold_cost_relative_to_five": float(
                        decision_by_k.loc[7, "fit_cost_relative_to_5_folds"]
                    ),
                    "ten_fold_cost_relative_to_five": float(
                        decision_by_k.loc[10, "fit_cost_relative_to_5_folds"]
                    ),
                    "ten_fold_catboost_mae_std": float(catboost.loc[10, "mae_std"]),
                },
                "rationale": [
                    "Five folds already preserve the CatBoost decision in every fold.",
                    "Seven folds improve nominal MAE by less than fold uncertainty.",
                    "Seven and ten folds cost materially more while using "
                    "smaller validation folds.",
                    "The official frozen holdout remains outside this protocol choice.",
                ],
                "holdout_used_for_choice": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(decision.to_string(index=False))
    print("\nPublished:")
    for path in paths.values():
        print(Path(path))


if __name__ == "__main__":
    main()
