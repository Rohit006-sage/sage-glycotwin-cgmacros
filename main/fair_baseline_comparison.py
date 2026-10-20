"""Same-fold nested HGB variants and simple glucose forecast baselines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

from revised_evaluation import (
    OUT, DOC_RESULTS, DOC_FIGURES, REDUCED_EHR, balanced_outer_group_splits,
    classify_features, make_clf, make_reg, tune_class, tune_reg, save_svg,
)

SEED = 42
DATA_PATH = OUT / "rich_120m_samples_local.csv"
PRIMARY_PRED_PATH = OUT / "primary_120m_seed_42_oof_local.csv"


def classification_report(frame, score_col):
    y = frame.label.to_numpy(int)
    s = frame[score_col].to_numpy(float)
    pooled = {"pr_auc": float(average_precision_score(y, s)),
              "roc_auc": float(roc_auc_score(y, s))}
    pr, roc = [], []
    for _, p in frame.groupby("participant_id"):
        # Average precision is defined as zero when a participant has no positive labels.
        pr.append(float(average_precision_score(p.label, p[score_col])) if p.label.sum() else 0.0)
        if p.label.nunique() == 2:
            roc.append(float(roc_auc_score(p.label, p[score_col])))
    macro = {"pr_auc": float(np.mean(pr)), "pr_auc_participants": len(pr),
             "roc_auc": float(np.mean(roc)), "roc_auc_participants": len(roc)}
    return {"pooled": pooled, "equal_participant_macro": macro}


def regression_report(frame, pred_col):
    y = frame.target_glucose.to_numpy(float)
    p = frame[pred_col].to_numpy(float)
    by_person = frame.assign(abs_error=np.abs(y-p), squared_error=(y-p)**2).groupby("participant_id")
    return {
        "pooled": {"mae_mg_dl": float(mean_absolute_error(y,p)),
                   "rmse_mg_dl": float(np.sqrt(mean_squared_error(y,p)))},
        "equal_participant_macro": {
            "mae_mg_dl": float(by_person.abs_error.mean().mean()),
            "rmse_mg_dl": float(np.sqrt(by_person.squared_error.mean()).mean()),
        },
    }


def main():
    data = pd.read_csv(DATA_PATH, dtype={"participant_id": str}, parse_dates=["timestamp"])
    primary = pd.read_csv(PRIMARY_PRED_PATH, dtype={"participant_id": str}, parse_dates=["timestamp"])
    all_features, dynamic_features, _ = classify_features(data)
    sensor = list(dynamic_features)
    reduced = [c for c in all_features if c in sensor or c in REDUCED_EHR]
    hba1c = [c for c in all_features if c in sensor or c == "ehr_hba1c_pct"]
    feature_sets = {"full_fused_hgb": all_features,
                    "sensor_only_hgb": sensor,
                    "reduced_five_ehr_hgb": reduced,
                    "hba1c_only_ehr_hgb": hba1c}
    y = data.label.to_numpy(int)
    groups = data.participant_id.to_numpy()
    splits = balanced_outer_group_splits(y, groups, SEED, 5)
    # Ensure the main model and every candidate are scored on the same held-out people.
    for fold, (_, te) in enumerate(splits, 1):
        expected = set(data.iloc[te].participant_id.astype(str))
        actual = set(primary.loc[primary.outer_fold.eq(fold), "participant_id"].astype(str))
        if expected != actual:
            raise ValueError(f"Fold {fold} does not match the repeated-CV primary split")

    rows = data[["participant_id", "timestamp", "label", "target_glucose"]].copy()
    rows = rows.merge(primary[["participant_id", "timestamp", "score", "outer_fold"]],
                      on=["participant_id", "timestamp"], how="left", validate="one_to_one")
    if rows.score.isna().any():
        raise ValueError("Primary held-out predictions do not cover every sample")
    rows["full_fused_hgb"] = rows.score
    rows["full_fused_hgb_regression"] = np.nan
    fold_settings = []
    for fold, (tr, te) in enumerate(splits, 1):
        fold_record = {"fold": fold, "train_participants": int(len(set(groups[tr]))),
                       "test_participants": int(len(set(groups[te]))), "candidate_settings": {}}
        for name, features in feature_sets.items():
            X = data[features].astype(float).reset_index(drop=True)
            if name != "full_fused_hgb":
                params, _, search = tune_class(X.iloc[tr].reset_index(drop=True), y[tr], groups[tr], SEED+fold)
                estimator = make_clf(params).fit(X.iloc[tr], y[tr])
                rows.loc[te, name] = estimator.predict_proba(X.iloc[te])[:, 1]
                fold_record["candidate_settings"][name] = {"classifier_selected_params": params,
                                                             "inner_grouped_ap": search}
        reg_params, reg_search = tune_reg(data[all_features].iloc[tr].astype(float).reset_index(drop=True),
                                          data.iloc[tr].target_glucose.to_numpy(float), groups[tr], SEED+fold)
        regressor = make_reg(reg_params).fit(data[all_features].iloc[tr].astype(float), data.iloc[tr].target_glucose.to_numpy(float))
        rows.loc[te, "full_fused_hgb_regression"] = regressor.predict(data[all_features].iloc[te].astype(float))
        fold_record["candidate_settings"]["full_fused_hgb_regression"] = {"selected_params": reg_params,
                                                                            "inner_grouped_mae": reg_search}
        fold_settings.append(fold_record)
        print(f"Same-fold comparisons: outer fold {fold}/5", flush=True)

    # Persistence and linear trend are deterministic baselines, not fitted models.
    # Missing baseline inputs are replaced by the training-participant median in each fold.
    rows["persistence"] = np.nan
    rows["trend_extrapolation"] = np.nan
    for fold, (tr, te) in enumerate(splits, 1):
        for col, feature in (("persistence", "glucose_now"),
                             ("trend_extrapolation", "linear_forecast_120m")):
            train_values = pd.to_numeric(data.iloc[tr][feature], errors="coerce")
            fallback = float(train_values.median())
            values = pd.to_numeric(data.iloc[te][feature], errors="coerce").fillna(fallback)
            rows.loc[te, col] = values.to_numpy(float)

    model_order = ["full_fused_hgb", "sensor_only_hgb", "reduced_five_ehr_hgb", "hba1c_only_ehr_hgb",
                   "persistence", "trend_extrapolation"]
    results = {name: classification_report(rows, name) for name in model_order}
    regression_results = {
        "full_fused_hgb": regression_report(rows, "full_fused_hgb_regression"),
        "persistence": regression_report(rows, "persistence"),
        "trend_extrapolation": regression_report(rows, "trend_extrapolation"),
    }
    payload = {
        "design": {
            "seed": SEED, "outer_folds": 5, "inner_folds": 3,
            "same_heldout_participants": True,
            "classifier_tuning": "Inner participant-grouped average precision only; all preprocessing/model fits use training participants.",
            "regression_tuning": "The full-fusion regressor is tuned by inner participant-grouped MAE and evaluated on the same outer participant assignments as classification. Regression baselines use the same held-out rows.",
            "baseline_definitions": {
                "persistence": "Future glucose estimate and risk score equal current observed glucose; missing values use the training-participant median.",
                "trend_extrapolation": "Current glucose plus the observed 60-minute slope extrapolated to 120 minutes; missing values use the training-participant median.",
            },
            "selection_caveat": "The reduced-EHR and HbA1c-only candidates were added after earlier cohort results were seen; choosing among them here is exploratory.",
        },
        "samples": int(len(rows)), "participants": int(rows.participant_id.nunique()),
        "events": int(rows.label.sum()), "models": results, "regression_comparison": regression_results,
        "fold_settings": fold_settings,
    }
    local = OUT / "fair_baseline_comparison_local.json"
    local.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    aggregate_path = DOC_RESULTS / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["same_fold_model_and_baseline_comparison"] = {k:v for k,v in payload.items() if k != "fold_settings"}
    aggregate_path.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    save_svg(DOC_FIGURES / "same_fold_model_comparison.svg",
             "Same held-out participants: model and simple-baseline ranking",
             {"Pooled PR-AUC": [results[k]["pooled"]["pr_auc"] for k in model_order],
              "Participant macro PR-AUC": [results[k]["equal_participant_macro"]["pr_auc"] for k in model_order]},
             ["Fused HGB","Sensor HGB","5-variable HGB","HbA1c HGB","Persistence","Trend"], "PR-AUC")
    print(json.dumps({"models": results, "regression_comparison": regression_results,
                      "samples": len(rows), "participants": rows.participant_id.nunique()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
