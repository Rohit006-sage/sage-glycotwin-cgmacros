from __future__ import annotations

"""Phase 2 nested participant-grouped evaluation and ablations."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_config import BOOTSTRAP_SEED, FFILL_TOLERANCE_MINUTES, GLUCOSE_THRESHOLD, RANDOM_SEED
from train_cgmacros_forecast import BIO_PATH, EXTRACT_DIR, load_ehr_features, bio_id_to_folder_id, build_samples

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = RANDOM_SEED
OUTER_FOLDS = 5
INNER_FOLDS = 3
# Compact deterministic coverage of the requested search ranges. max_iter is
# selected by inner participant holdouts, which is grouped validation-based
# early stopping without sklearn's row-random validation_fraction.
HGB_CONFIGS = [
    {"learning_rate": 0.03, "max_iter": 500, "max_leaf_nodes": 7, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 2.0},
    {"learning_rate": 0.10, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 10.0},
    {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 7, "min_samples_leaf": 300, "l2_regularization": 10.0},
    {"learning_rate": 0.10, "max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 150, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_iter": 250, "max_leaf_nodes": 31, "min_samples_leaf": 300, "l2_regularization": 5.0},
]
REDUCED_EHR = ["ehr_age_years", "ehr_gender_female", "ehr_bmi", "ehr_hba1c_pct", "ehr_fasting_glucose_mg_dl"]


def make_classifier(params: dict) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params),
    )


def make_regressor(params: dict) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(random_state=SEED, loss="squared_error", early_stopping=False, **params),
    )


def inner_splits(X, y, groups, seed):
    n_groups = len(np.unique(groups))
    splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(X, y, groups))


def tune_classifier(X, y, groups, seed):
    splits = inner_splits(X, y, groups, seed)
    candidate_results = []
    for params in HGB_CONFIGS:
        scores = []
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            model = make_classifier(params)
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = pred
            scores.append(average_precision_score(y[va], pred))
        candidate_results.append((float(np.mean(scores)), params, oof))
    candidate_results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_params, best_oof = candidate_results[0]
    return best_params, best_oof, [
        {"mean_inner_pr_auc": score, "params": params}
        for score, params, _ in candidate_results
    ]


def tune_regressor(X, y, groups, seed):
    splits = inner_splits(X, (y >= GLUCOSE_THRESHOLD).astype(int), groups, seed)
    candidate_results = []
    for params in HGB_CONFIGS:
        fold_mae = []
        for tr, va in splits:
            model = make_regressor(params)
            model.fit(X.iloc[tr], y[tr])
            fold_mae.append(mean_absolute_error(y[va], model.predict(X.iloc[va])))
        candidate_results.append((float(np.mean(fold_mae)), params))
    candidate_results.sort(key=lambda item: item[0])
    return candidate_results[0][1], [
        {"mean_inner_mae_mg_dl": score, "params": params}
        for score, params in candidate_results
    ]


def choose_threshold(y, scores, min_precision=0.80):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if not len(valid):
        return None, 0.0, 0.0
    best = valid[np.argmax(recall[valid])]
    return float(thresholds[best]), float(precision[best]), float(recall[best])


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None,
        "brier": float(brier_score_loss(y, p)),
    }


def reg_metrics(y, p):
    return {"mae_mg_dl": float(mean_absolute_error(y, p)), "rmse_mg_dl": float(np.sqrt(mean_squared_error(y, p)))}


def operating_metrics(y, p, threshold):
    pred = np.zeros(len(y), dtype=int) if threshold is None else (np.asarray(p) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.sum() else None,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "predicted_positive_samples": int(pred.sum()),
    }


def macro_metrics(frame, score_col, target_col="label"):
    by_person = []
    for _, part in frame.groupby("participant_id"):
        y = part[target_col].to_numpy(dtype=int)
        p = part[score_col].to_numpy(dtype=float)
        by_person.append({
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None,
            "brier": float(brier_score_loss(y, p)),
        })
    result = {}
    for key in ("pr_auc", "roc_auc", "brier"):
        vals = [r[key] for r in by_person if r[key] is not None]
        result[key] = float(np.mean(vals)) if vals else None
        result[key + "_participants"] = len(vals)
    return result


def calibration_bins(y, p, n_bins=10):
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame.p, q=n_bins, duplicates="drop")
    return [{"mean_predicted": float(r.p), "observed_rate": float(r.y), "samples": int(r.n)}
            for _, r in frame.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).iterrows()]


def save_calibration_svg(bins):
    width, height, left, top, right, bottom = 760, 500, 85, 40, 32, 78
    pw, ph = width-left-right, height-top-bottom
    pts = []
    for b in bins:
        x = left + pw*b["mean_predicted"]
        y = top + ph*(1-b["observed_rate"])
        pts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#16745b"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial" font-size="13" fill="#36564b"><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top}" stroke="#c18a36" stroke-dasharray="7 6" stroke-width="2"/>{''.join(pts)}<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#496b5d"/><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#496b5d"/><text x="{left+pw/2}" y="{height-20}" text-anchor="middle">Mean predicted probability (nested held-out folds)</text><text x="20" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 20 {top+ph/2})">Observed event rate</text><text x="{left}" y="24" font-size="17" font-weight="bold">In-fold Platt calibration evaluated out of fold</text></g></svg>'''
    (OUT / "phase2_calibration.svg").write_text(svg, encoding="utf-8")


def add_ehr(samples):
    ehr = load_ehr_features()
    samples = samples.merge(ehr, on="participant_id", how="left", validate="many_to_one")
    return samples


def get_samples_by_age():
    data = {}
    for age in (15, FFILL_TOLERANCE_MINUTES, 60):
        samples, _, _ = build_samples(ffill_tolerance_minutes=age)
        data[age] = add_ehr(samples)
    base = data[FFILL_TOLERANCE_MINUTES][["participant_id", "timestamp", "label", "target_glucose_t_plus_120m"]].reset_index(drop=True)
    for age, frame in data.items():
        check = frame[["participant_id", "timestamp", "label", "target_glucose_t_plus_120m"]].reset_index(drop=True)
        if not base.equals(check):
            raise AssertionError(f"Forward-fill ablation at {age} minutes changed sample identities or targets")
    return data


def run_nested(samples_by_age):
    base = samples_by_age[FFILL_TOLERANCE_MINUTES]
    excluded = {"participant_id", "timestamp", "target_glucose_t_plus_120m", "label"}
    feature_cols = [c for c in base.columns if c not in excluded]
    dynamic_cols = [c for c in feature_cols if not c.startswith("ehr_")]
    fused_cols = feature_cols
    reduced_cols = dynamic_cols + [c for c in REDUCED_EHR if c in base]
    y = base.label.to_numpy(dtype=int)
    y_reg = base.target_glucose_t_plus_120m.to_numpy(dtype=float)
    groups = base.participant_id.to_numpy()
    X30 = base[feature_cols].astype(float)
    Xdynamic = base[dynamic_cols].astype(float)
    Xreduced = base[reduced_cols].astype(float)
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    pred = pd.DataFrame({"participant_id": base.participant_id, "timestamp": base.timestamp,
                         "label": y, "target_glucose_t_plus_120m": y_reg})
    for name in ("hgb_fused", "hgb_sensor_only", "logistic_fused", "logistic_reduced_ehr",
                 "hgb_ffill_15m", "hgb_ffill_60m", "calibrated_hgb_fused", "regressor_hgb", "regressor_persistence"):
        pred[name] = np.nan
    pred["outer_fold"] = -1
    pred["operating_threshold"] = np.nan
    pred["operating_prediction"] = 0
    folds = []
    for fold_no, (tr, te) in enumerate(outer.split(X30, y, groups), start=1):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Outer participant leakage")
        yt, yv = y[tr], y[te]
        gt, gv = groups[tr], groups[te]
        seed = SEED + fold_no
        best, inner_oof, candidates = tune_classifier(X30.iloc[tr].reset_index(drop=True), yt, gt, seed)
        threshold, train_precision, train_recall = choose_threshold(yt, inner_oof)
        pred.loc[te, "outer_fold"] = fold_no
        pred.loc[te, "operating_threshold"] = np.nan if threshold is None else threshold
        model = make_classifier(best).fit(X30.iloc[tr], yt)
        test_score = model.predict_proba(X30.iloc[te])[:, 1]
        pred.loc[te, "operating_prediction"] = 0 if threshold is None else (test_score >= threshold).astype(int)
        pred.loc[te, "hgb_fused"] = test_score

        # Platt sigmoid calibration is fitted only to inner-fold OOF scores from
        # this outer-training partition, then applied to its untouched test fold.
        calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        calibrator.fit(logit(np.clip(inner_oof, 1e-6, 1-1e-6)).reshape(-1, 1), yt)
        calibrated = calibrator.predict_proba(logit(np.clip(test_score, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
        pred.loc[te, "calibrated_hgb_fused"] = calibrated

        sensor = make_classifier(best).fit(Xdynamic.iloc[tr], yt)
        pred.loc[te, "hgb_sensor_only"] = sensor.predict_proba(Xdynamic.iloc[te])[:, 1]
        for age in (15, 60):
            frame_age = samples_by_age[age]
            cols_age = [c for c in frame_age.columns if c not in excluded and not c.startswith("ehr_")]
            model_age = make_classifier(best).fit(frame_age[cols_age].astype(float).iloc[tr], yt)
            pred.loc[te, f"hgb_ffill_{age}m"] = model_age.predict_proba(frame_age[cols_age].astype(float).iloc[te])[:, 1]

        full_log = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
        full_log.fit(X30.iloc[tr], yt)
        pred.loc[te, "logistic_fused"] = full_log.predict_proba(X30.iloc[te])[:, 1]
        reduced_log = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
        reduced_log.fit(Xreduced.iloc[tr], yt)
        pred.loc[te, "logistic_reduced_ehr"] = reduced_log.predict_proba(Xreduced.iloc[te])[:, 1]

        # Regression uses inner grouped validation to select iteration/complexity
        # by MAE, then refits on the full outer-training participants.
        reg_params, reg_candidates = tune_regressor(X30.iloc[tr].reset_index(drop=True), y_reg[tr], gt, seed)
        reg = make_regressor(reg_params).fit(X30.iloc[tr], y_reg[tr])
        pred.loc[te, "regressor_hgb"] = reg.predict(X30.iloc[te])
        pred.loc[te, "regressor_persistence"] = X30.iloc[te]["glucose_now"].fillna(X30.iloc[tr]["glucose_now"].median()).to_numpy()
        folds.append({
            "fold": fold_no,
            "train_participants": int(len(set(gt))), "test_participants": int(len(set(gv))),
            "train_samples": int(len(tr)), "test_samples": int(len(te)), "test_events": int(yv.sum()),
            "test_participant_ids": sorted(map(str, set(gv))),
            "selected_classifier_params": best,
            "inner_classifier_candidates": candidates,
            "threshold_selected_on_inner_oof": {"threshold": threshold, "precision": train_precision, "recall": train_recall},
            "outer_operating_point": operating_metrics(yv, test_score, threshold),
            "outer_metrics": {"hgb_fused": metrics(yv, test_score), "calibrated_hgb_fused": metrics(yv, calibrated),
                              "hgb_sensor_only": metrics(yv, pred.loc[te, "hgb_sensor_only"]),
                              "logistic_fused": metrics(yv, pred.loc[te, "logistic_fused"]),
                              "logistic_reduced_ehr": metrics(yv, pred.loc[te, "logistic_reduced_ehr"]),
                              "hgb_ffill_15m": metrics(yv, pred.loc[te, "hgb_ffill_15m"]),
                              "hgb_ffill_60m": metrics(yv, pred.loc[te, "hgb_ffill_60m"])},
            "selected_regression_params": reg_params,
            "inner_regression_candidates": reg_candidates,
            "outer_regression": {"hgb": reg_metrics(y_reg[te], pred.loc[te, "regressor_hgb"]),
                                 "persistence": reg_metrics(y_reg[te], pred.loc[te, "regressor_persistence"])},
        })
        pred.to_csv(OUT / "phase2_oof_partial.csv", index=False)
        (OUT / "phase2_folds_partial.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
        print(f"Completed nested outer fold {fold_no}/{OUTER_FOLDS}")
    required = [c for c in pred.columns if c not in {"participant_id", "timestamp", "label", "target_glucose_t_plus_120m", "operating_threshold"}]
    if pred[required].isna().any().any():
        raise AssertionError("At least one sample lacks a nested out-of-fold model prediction")
    return pred, folds, {"all": fused_cols, "dynamic": dynamic_cols, "reduced": reduced_cols}


def summarize(pred, folds):
    y = pred.label.to_numpy(dtype=int)
    result = {"pooled": {}, "participant_macro": {}, "outer_fold_macro": {}, "operating_point": {"target_precision": 0.8, "folds": []}, "regression": {}}
    score_cols = [c for c in pred.columns if c in ("hgb_fused", "calibrated_hgb_fused", "hgb_sensor_only", "logistic_fused", "logistic_reduced_ehr", "hgb_ffill_15m", "hgb_ffill_60m")]
    for col in score_cols:
        result["pooled"][col] = metrics(y, pred[col].to_numpy())
        result["participant_macro"][col] = macro_metrics(pred, col)
        fold_results = [metrics(part.label.to_numpy(), part[col].to_numpy()) for _, part in pred.groupby("outer_fold")]
        result["outer_fold_macro"][col] = {
            k: float(np.mean([m[k] for m in fold_results if m[k] is not None]))
            for k in ("pr_auc", "roc_auc", "brier")
        }
    for f in folds:
        result["operating_point"]["folds"].append({"fold": f["fold"], **f["threshold_selected_on_inner_oof"], **f["outer_operating_point"]})
    op_pred = pred.operating_prediction.to_numpy(dtype=int)
    result["operating_point"]["pooled_outer_oof"] = {
        "precision": float(precision_score(y, op_pred, zero_division=0)),
        "recall": float(recall_score(y, op_pred, zero_division=0)),
        "predicted_positive_samples": int(op_pred.sum()),
        "threshold_policy": "Each outer test fold uses its own threshold selected on inner OOF predictions from that fold's training participants.",
    }
    result["regression"] = {
        "hgb_pooled": reg_metrics(pred.target_glucose_t_plus_120m, pred.regressor_hgb),
        "persistence_pooled": reg_metrics(pred.target_glucose_t_plus_120m, pred.regressor_persistence),
        "hgb_participant_macro": {
            "mae_mg_dl": float(pred.assign(ae=(pred.regressor_hgb-pred.target_glucose_t_plus_120m).abs(), se=(pred.regressor_hgb-pred.target_glucose_t_plus_120m)**2).groupby("participant_id").ae.mean().mean()),
            "rmse_mg_dl": float(np.sqrt(pred.assign(se=(pred.regressor_hgb-pred.target_glucose_t_plus_120m)**2).groupby("participant_id").se.mean()).mean()),
        },
        "persistence_participant_macro": {
            "mae_mg_dl": float(pred.assign(ae=(pred.regressor_persistence-pred.target_glucose_t_plus_120m).abs()).groupby("participant_id").ae.mean().mean()),
            "rmse_mg_dl": float(np.sqrt(pred.assign(se=(pred.regressor_persistence-pred.target_glucose_t_plus_120m)**2).groupby("participant_id").se.mean()).mean()),
        },
    }
    return result


def main():
    samples_by_age = get_samples_by_age()
    pred, folds, feature_sets = run_nested(samples_by_age)
    pred_path = OUT / "phase2_oof_predictions.csv"
    pred.to_csv(pred_path, index=False)
    summary = summarize(pred, folds)
    bins = calibration_bins(pred.label.to_numpy(), pred.calibrated_hgb_fused.to_numpy())
    save_calibration_svg(bins)
    payload = {
        "phase": "Phase 2 nested grouped CV",
        "data": {"participants": int(pred.participant_id.nunique()), "samples": int(len(pred)), "events": int(pred.label.sum()),
                 "target": f"CGM >= {GLUCOSE_THRESHOLD:g} mg/dL at +120 minutes", "outer_folds": OUTER_FOLDS,
                 "inner_folds": INNER_FOLDS, "group": "participant_id", "forward_fill_ablation_minutes": [15, 30, 60]},
        "selection": {"classifier_objective": "mean inner-fold average precision", "regression_objective": "mean inner-fold MAE",
                      "parameter_candidates": HGB_CONFIGS, "max_iter_selection": "inner participant validation; row-random sklearn early_stopping disabled"},
        "feature_sets": feature_sets,
        "results": summary,
        "in_fold_calibration": {"method": "Platt sigmoid fitted on inner grouped OOF predictions within each outer training partition; evaluated on outer held-out participants",
                                 "bins": bins, "plot": "phase2_calibration.svg"},
        "folds": folds,
    }
    (OUT / "phase2_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"results": summary, "metrics_path": str(OUT / 'phase2_metrics.json'), "predictions_path": str(pred_path)}, indent=2))
    finalize_saved()


def participant_bootstrap(frame, score_column, iterations=2000):
    ids = sorted(frame.participant_id.astype(str).unique())
    codes = pd.Categorical(frame.participant_id.astype(str), categories=ids).codes
    counts = np.zeros(len(ids), dtype=int)
    y = frame.label.to_numpy(dtype=int)
    p = frame[score_column].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(iterations):
        counts.fill(0)
        np.add.at(counts, rng.integers(0, len(ids), len(ids)), 1)
        w = counts[codes]
        if w.sum() == 0:
            continue
        if np.sum(y*w) > 0:
            values["pr_auc"].append(float(average_precision_score(y, p, sample_weight=w)))
        if np.unique(y[w > 0]).size > 1:
            values["roc_auc"].append(float(roc_auc_score(y, p, sample_weight=w)))
        values["brier"].append(float(brier_score_loss(y, p, sample_weight=w)))
    return {k: {"lower_95": float(np.quantile(v, .025)), "upper_95": float(np.quantile(v, .975)), "valid_replicates": len(v)} for k, v in values.items()}


def finalize_saved():
    path = OUT / "phase2_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    pred = pd.read_csv(OUT / "phase2_oof_predictions.csv")
    pred["participant_id"] = pred.participant_id.astype(str).str.zfill(3)
    bio = pd.read_csv(BIO_PATH, usecols=["subject", "A1c PDL (Lab)"])
    bio["participant_id"] = bio.subject.map(bio_id_to_folder_id)
    bio["hba1c"] = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    bio["stratum"] = np.select([bio.hba1c < 5.7, bio.hba1c.between(5.7, 6.4), bio.hba1c >= 6.5],
                               ["Healthy-range HbA1c", "Prediabetes-range HbA1c", "Diabetes-range HbA1c"], default="Unclassified")
    frame = pred.merge(bio[["participant_id", "hba1c", "stratum"]], on="participant_id", how="left", validate="many_to_one")
    groups = {}
    for name, part in frame.groupby("stratum", sort=False):
        if name == "Unclassified":
            continue
        event_people = part.loc[part.label.eq(1), "participant_id"].nunique()
        groups[name] = {"participants": int(part.participant_id.nunique()), "participants_with_event": int(event_people),
                        "samples": int(len(part)), "events": int(part.label.sum()), "event_prevalence": float(part.label.mean()),
                        "constant_prevalence_pr_baseline": float(part.label.mean()),
                        "hgb_fused_pooled": metrics(part.label, part.hgb_fused),
                        "calibrated_hgb_fused_pooled": metrics(part.label, part.calibrated_hgb_fused)}
    overall_ci = participant_bootstrap(frame, "hgb_fused")
    payload["results"]["participant_bootstrap_95_ci_hgb_fused"] = overall_ci
    weighted = {}
    for model in payload["results"]["pooled"]:
        values = [f["outer_metrics"][model] for f in payload["folds"]]
        weights = np.asarray([f["test_samples"] for f in payload["folds"]], dtype=float)
        weighted[model] = {metric: float(np.average([v[metric] for v in values], weights=weights))
                           for metric in ("pr_auc", "roc_auc", "brier")}
    payload["results"]["outer_fold_sample_weighted"] = weighted
    operating_rows = []
    for f in payload["folds"]:
        part = pred.loc[pred.outer_fold.eq(f["fold"])]
        op = operating_metrics(part.label.to_numpy(), part.hgb_fused.to_numpy(),
                               f["threshold_selected_on_inner_oof"]["threshold"])
        f["outer_operating_point"] = op
        f["test_participant_ids"] = sorted(part.participant_id.unique().tolist())
        operating_rows.append({"fold": f["fold"], "threshold": op["threshold"],
                               "training_validation_precision": (f["threshold_selected_on_inner_oof"]["precision"] if op["threshold"] is not None else None),
                               "training_validation_recall": (f["threshold_selected_on_inner_oof"]["recall"] if op["threshold"] is not None else None),
                               "heldout_precision": op["precision"], "heldout_recall": op["recall"],
                               "heldout_predicted_positive_samples": op["predicted_positive_samples"]})
    payload["results"]["operating_point"]["folds"] = operating_rows
    op_pred = pred.operating_prediction.to_numpy(dtype=int)
    pooled_op = {"precision": float(precision_score(pred.label, op_pred, zero_division=0)) if op_pred.sum() else None,
                 "recall": float(recall_score(pred.label, op_pred, zero_division=0)),
                 "predicted_positive_samples": int(op_pred.sum()),
                 "threshold_policy": "Outer-fold-specific threshold selected on inner OOF scores from training participants."}
    payload["results"]["operating_point"]["pooled_outer_oof"] = pooled_op
    payload["strata"] = {"definition": "Baseline HbA1c categories; analytic groups, not independent chart diagnoses.", "results": groups}
    payload["artifacts"] = {"oof_predictions": "phase2_oof_predictions.csv", "calibration_plot": "phase2_calibration.svg",
                            "fold_details": "embedded in phase2_metrics.json", "participant_bootstraps": 2000}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        finalize_saved()
    else:
        main()
