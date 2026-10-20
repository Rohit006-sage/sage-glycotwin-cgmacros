from __future__ import annotations

"""Revised nested evaluation with temporal features, horizons, repetitions, and personal adaptation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from project_config import BOOTSTRAP_SEED, GLUCOSE_THRESHOLD, RANDOM_SEED, SAMPLE_MINUTES
from train_cgmacros_forecast import (EXTRACT_DIR, load_ehr_features, make_features,
                                     read_participant_file, folder_id_to_bio_id)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed" / "revised_full"
OUT.mkdir(parents=True, exist_ok=True)
DOC_RESULTS = ROOT / "docs" / "results"
DOC_FIGURES = ROOT / "docs" / "figures"
DOC_RESULTS.mkdir(parents=True, exist_ok=True)
DOC_FIGURES.mkdir(parents=True, exist_ok=True)
HORIZONS = (30, 60, 120)
REPEATED_SEEDS = (11, 23, 42, 73, 101)
OUTER_FOLDS = 5
INNER_FOLDS = 3
# Deliberately small, predeclared grid spanning each requested interval.
# Every choice is scored only on inner participant-grouped validation.
GRID = [
    {"learning_rate": .02, "max_iter": 1000, "max_leaf_nodes": 3, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": .05, "max_iter": 600, "max_leaf_nodes": 15, "min_samples_leaf": 500, "l2_regularization": 5.0},
    {"learning_rate": .10, "max_iter": 1000, "max_leaf_nodes": 31, "min_samples_leaf": 1000, "l2_regularization": 10.0},
    {"learning_rate": .05, "max_iter": 300, "max_leaf_nodes": 7, "min_samples_leaf": 1000, "l2_regularization": 0.0},
]
REDUCED_EHR = ["ehr_age_years", "ehr_gender_female", "ehr_bmi", "ehr_hba1c_pct", "ehr_fasting_glucose_mg_dl"]


def richer_features(raw: pd.DataFrame, grid: pd.DatetimeIndex, tolerance: int = 30) -> pd.DataFrame:
    f = make_features(raw, grid, ffill_tolerance_minutes=tolerance).copy()
    obs = raw.reindex(grid, method="ffill", tolerance=pd.Timedelta(minutes=tolerance))
    g = obs["glucose"]
    for minutes in (30, 60, 120):
        points = minutes // SAMPLE_MINUTES + 1
        roll = g.rolling(points, min_periods=1)
        f[f"glucose_mean_{minutes}m"] = roll.mean()
        f[f"glucose_std_{minutes}m"] = roll.std()
        f[f"glucose_min_{minutes}m"] = roll.min()
        f[f"glucose_max_{minutes}m"] = roll.max()
    above_window = 60 // SAMPLE_MINUTES + 1
    f["glucose_minutes_observed_last_60m"] = g.notna().astype(float).rolling(above_window, min_periods=1).sum() * SAMPLE_MINUTES
    f["glucose_minutes_above_140_last_60m"] = g.ge(140).astype(float).where(g.notna()).rolling(above_window, min_periods=1).sum() * SAMPLE_MINUTES
    f["glucose_minutes_above_180_last_60m"] = g.ge(180).astype(float).where(g.notna()).rolling(above_window, min_periods=1).sum() * SAMPLE_MINUTES

    # Exploratory linear-decay carbohydrate-on-board proxy. Each recorded meal
    # carbohydrate value contributes for up to 180 minutes; this is not a
    # validated physiological absorption model.
    cob = np.zeros(len(grid), dtype=float)
    event_carbs = pd.to_numeric(raw["carbs"], errors="coerce").fillna(0.0)
    event_carbs = event_carbs.loc[event_carbs > 0]
    grid_ns = grid.asi8
    for when, grams in event_carbs.items():
        start = int(np.searchsorted(grid_ns, when.value, side="left"))
        stop = int(np.searchsorted(grid_ns, (when + pd.Timedelta(minutes=180)).value, side="right"))
        if start < stop:
            age = (grid_ns[start:stop] - when.value) / 60_000_000_000
            cob[start:stop] += float(grams) * np.clip(1.0 - age / 180.0, 0.0, 1.0)
    f["carbohydrate_on_board_g"] = cob
    return f.replace([np.inf, -np.inf], np.nan)


def make_all_horizon_data():
    frames = {h: [] for h in HORIZONS}
    bio = load_ehr_features()
    paths = sorted(EXTRACT_DIR.glob("CGMacros-*/CGMacros-*.csv"))
    for path in paths:
        pid = folder_id_to_bio_id(path.parent.name)
        raw = read_participant_file(path)
        if raw.empty:
            continue
        start, end = raw.index.min().ceil(f"{SAMPLE_MINUTES}min"), raw.index.max().floor(f"{SAMPLE_MINUTES}min")
        if start > end:
            continue
        grid = pd.date_range(start, end, freq=f"{SAMPLE_MINUTES}min")
        features = richer_features(raw, grid)
        for horizon in HORIZONS:
            future_time = grid + pd.Timedelta(minutes=horizon)
            target = pd.Series(raw["glucose"].reindex(future_time).to_numpy(), index=grid)
            valid = target.notna()
            part = features.loc[valid].copy()
            part["participant_id"] = pid
            part["timestamp"] = grid[valid]
            part["target_glucose"] = target.loc[valid].to_numpy()
            part["label"] = (part.target_glucose >= GLUCOSE_THRESHOLD).astype(int)
            part["horizon_minutes"] = horizon
            frames[horizon].append(part.reset_index(drop=True))
    result = {}
    for h, items in frames.items():
        data = pd.concat(items, ignore_index=True).merge(bio, on="participant_id", how="left", validate="many_to_one")
        result[h] = data
    return result


def classify_features(data: pd.DataFrame):
    cols = [c for c in data.columns if c not in {"participant_id", "timestamp", "target_glucose", "label", "horizon_minutes"}]
    dynamic = [c for c in cols if not c.startswith("ehr_")]
    groups = {
        "glucose_history_and_rolling": [c for c in cols if (c.startswith("glucose_") or c.startswith("linear_forecast")) and "minutes_above" not in c],
        "time_above_range": [c for c in cols if "minutes_above_" in c or c == "glucose_minutes_observed_last_60m"],
        "wearable_activity": [c for c in cols if c.startswith(("heart_rate", "activity_"))],
        "meal_and_carbohydrate": [c for c in cols if c.startswith(("carbs_", "food_calories_", "protein_", "fat_", "fiber_", "meal_", "minutes_since_meal", "carbohydrate_on_board"))],
        "baseline_ehr": [c for c in cols if c.startswith("ehr_")],
        "time_of_day": [c for c in cols if c.startswith("time_of_day_")],
    }
    return cols, dynamic, groups


def make_clf(params):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         HistGradientBoostingClassifier(random_state=RANDOM_SEED, early_stopping=False, **params))


def make_reg(params):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         HistGradientBoostingRegressor(random_state=RANDOM_SEED, loss="squared_error", early_stopping=False, **params))


def group_splits(X, y, groups, seed, n=INNER_FOLDS):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(X, y, groups))


def balanced_outer_group_splits(y, groups, seed, n=OUTER_FOLDS):
    """Seeded participant-level folds balanced on sample count and event count.

    StratifiedGroupKFold can return the same assignment across seeds when large
    participant groups dominate its greedy allocation. This randomized greedy
    allocator makes each requested seed an actual, reproducible assignment.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    stats = []
    for pid in ids:
        ix = np.flatnonzero(groups == pid)
        stats.append((pid, len(ix), int(y[ix].sum())))
    rng = np.random.default_rng(seed)
    # Randomized tie ordering, while placing event-heavy people first.
    jitter = {pid: rng.random() for pid, _, _ in stats}
    stats.sort(key=lambda z: (-z[2] / max(1, z[1]), -z[2], jitter[z[0]]))
    fold_people = [[] for _ in range(n)]
    fold_n = np.zeros(n, dtype=float)
    fold_pos = np.zeros(n, dtype=float)
    target_n, target_pos, target_people = len(y) / n, float(y.sum()) / n, len(ids) / n
    for pid, count, positives in stats:
        costs = []
        for f in range(n):
            nn, pp, gp = fold_n.copy(), fold_pos.copy(), np.array([len(v) for v in fold_people], dtype=float)
            nn[f] += count; pp[f] += positives; gp[f] += 1
            # Scale dimensions so no one unit dominates; tie-break by seeded RNG.
            cost = np.square((nn - target_n) / max(1, target_n)).sum()
            cost += np.square((pp - target_pos) / max(1, target_pos)).sum()
            cost += .25 * np.square((gp - target_people) / max(1, target_people)).sum()
            costs.append(cost)
        best = np.flatnonzero(np.isclose(costs, min(costs), rtol=0, atol=1e-12))
        f = int(rng.choice(best))
        fold_people[f].append(pid); fold_n[f] += count; fold_pos[f] += positives
    splits = []
    for test_people in fold_people:
        test = np.flatnonzero(np.isin(groups, test_people))
        train = np.flatnonzero(~np.isin(groups, test_people))
        if len(np.unique(y[test])) < 2:
            raise ValueError("A participant fold lacks one target class")
        splits.append((train, test))
    return splits


def tune_class(X, y, groups, seed):
    splits = group_splits(X, y, groups, seed)
    ranked = []
    for params in GRID:
        oof = np.full(len(y), np.nan)
        fold_ap = []
        for tr, va in splits:
            model = make_clf(params).fit(X.iloc[tr], y[tr])
            scores = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = scores
            fold_ap.append(average_precision_score(y[va], scores))
        ranked.append((float(np.mean(fold_ap)), params, oof))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1], ranked[0][2], [{"mean_inner_ap": x[0], "params": x[1]} for x in ranked]


def tune_reg(X, y, groups, seed):
    strata = (y >= GLUCOSE_THRESHOLD).astype(int)
    ranked = []
    for params in GRID:
        fold_mae = []
        for tr, va in group_splits(X, strata, groups, seed):
            model = make_reg(params).fit(X.iloc[tr], y[tr])
            fold_mae.append(mean_absolute_error(y[va], model.predict(X.iloc[va])))
        ranked.append((float(np.mean(fold_mae)), params))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1], [{"mean_inner_mae": x[0], "params": x[1]} for x in ranked]


def class_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    return {"pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None,
            "brier": float(brier_score_loss(y, p))}


def participant_macro(df, score):
    rows = []
    for _, p in df.groupby("participant_id"):
        rows.append(class_metrics(p.label, p[score]))
    return {m: float(np.mean([r[m] for r in rows if r[m] is not None])) if any(r[m] is not None for r in rows) else None
            for m in ("pr_auc", "roc_auc", "brier")}


def regression_metrics(y, pred):
    return {"mae_mg_dl": float(mean_absolute_error(y, pred)),
            "rmse_mg_dl": float(np.sqrt(mean_squared_error(y, pred)))}


def choose_precision_threshold(y, score, target):
    prec, rec, thresholds = precision_recall_curve(y, score)
    good = np.flatnonzero(prec[:-1] >= target)
    if not len(good):
        return None
    idx = good[np.argmax(rec[good])]
    return float(thresholds[idx])


def op_stats(df, alert):
    alert = np.asarray(alert, int)
    y = df.label.to_numpy(int)
    nalerts = int(alert.sum())
    caught = df.loc[(y == 1) & (alert == 1), "participant_id"].nunique()
    truepositives = int(((y == 1) & (alert == 1)).sum())
    return {"precision": float(truepositives / nalerts) if nalerts else None,
            "recall": float(truepositives / y.sum()) if y.sum() else None,
            "alerts": nalerts,
            "alerts_per_participant_day": float(nalerts / (len(df) / 96.0)) if len(df) else 0.0,
            "participants_with_true_positive_caught": int(caught)}


def nested_classification(data, feature_cols, seeds, label="main"):
    X = data[feature_cols].astype(float).reset_index(drop=True)
    y = data.label.to_numpy(int)
    groups = data.participant_id.to_numpy()
    seed_results, prediction_sets, picked_settings = [], [], {}
    for seed in seeds:
        oof = pd.DataFrame({"participant_id": data.participant_id, "timestamp": data.timestamp,
                            "label": y, "target_glucose": data.target_glucose})
        oof["score"] = np.nan
        oof["outer_fold"] = -1
        folds = []
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed).split(X, y, groups), 1):
            params, inner_oof, search = tune_class(X.iloc[tr].reset_index(drop=True), y[tr], groups[tr], seed + fold)
            model = make_clf(params).fit(X.iloc[tr], y[tr])
            scores = model.predict_proba(X.iloc[te])[:, 1]
            oof.loc[te, "score"] = scores
            oof.loc[te, "outer_fold"] = fold
            folds.append({"seed": seed, "fold": fold, "outer_test_samples": len(te), "outer_test_participants": len(set(groups[te])),
                          "selected_params": params, "inner_search": search,
                          "edge_hits": edge_hits(params), "test_metrics": class_metrics(y[te], scores)})
        if oof.score.isna().any():
            raise RuntimeError("Nested split did not provide exactly one prediction per sample")
        oof["seed"] = seed
        prediction_sets.append(oof)
        picked_settings[seed] = folds
        seed_results.append({"seed": seed, "pooled": class_metrics(y, oof.score), "participant_macro": participant_macro(oof, "score"),
                             "folds": folds})
        print(f"{label}: completed seed {seed}", flush=True)
    return seed_results, prediction_sets, picked_settings


def edge_hits(params):
    return {"learning_rate_min": params["learning_rate"] == .02,
            "learning_rate_max": params["learning_rate"] == .10,
            "max_iter_max": params["max_iter"] == 1000,
            "max_leaf_nodes_min": params["max_leaf_nodes"] == 3,
            "max_leaf_nodes_max": params["max_leaf_nodes"] == 31,
            "min_samples_leaf_min": params["min_samples_leaf"] == 50,
            "min_samples_leaf_max": params["min_samples_leaf"] == 1000,
            "l2_min": params["l2_regularization"] == 0.0,
            "l2_max": params["l2_regularization"] == 10.0}


def repeated_summary(seed_results):
    metrics = ["pr_auc", "roc_auc"]
    out = {"by_seed": [], "spread": {}}
    for row in seed_results:
        out["by_seed"].append({"seed": row["seed"], "pooled": {m: row["pooled"][m] for m in metrics},
                               "participant_macro": {m: row["participant_macro"][m] for m in metrics}})
    for estimand in ("pooled", "participant_macro"):
        out["spread"][estimand] = {}
        for m in metrics:
            values = np.asarray([x[estimand][m] for x in out["by_seed"]], float)
            out["spread"][estimand][m] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                                           "min": float(values.min()), "max": float(values.max())}
    return out


def summarize_op_by_seed(op_rows):
    out = []
    for seed, df in op_rows.groupby("seed"):
        item = {"seed": int(seed), "policies": {}}
        for policy, part in df.groupby("policy", sort=False):
            vals = {k: [] for k in ("precision", "recall", "alerts", "alerts_per_participant_day", "participants_with_true_positive_caught")}
            for _, row in part.iterrows():
                for k in vals:
                    v = row[k]
                    if pd.notna(v): vals[k].append(float(v))
            # Fold means for rates; alert totals and participant coverage are summed/unioned at seed-level upstream.
            item["policies"][policy] = {k: (float(np.mean(v)) if v else None) for k, v in vals.items()}
        out.append(item)
    return out


def save_svg(path: Path, title: str, series: dict[str, list[float]], labels: list[str], y_label: str):
    w,h,l,t,r,b=820,500,90,55,35,90; pw,ph=w-l-r,h-t-b
    vals=[v for arr in series.values() for v in arr if np.isfinite(v)] or [0,1]
    ymin,ymax=min(vals),max(vals)
    if ymax-ymin < 1e-9: ymax=ymin+1
    def xy(i,v): return l+pw*i/max(1,len(labels)-1), t+ph*(1-(v-ymin)/(ymax-ymin))
    colors=["#147d64","#d68a2d","#5369a8","#b64a52","#6d8b3d","#855b9a"]
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial" fill="#294a40">']
    for j in range(5):
        y=t+ph*j/4
        val=ymax-(ymax-ymin)*j/4
        svg.append(f'<line x1="{l}" y1="{y:.1f}" x2="{l+pw}" y2="{y:.1f}" stroke="#e3e9e6"/><text x="{l-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{val:.3f}</text>')
    for si,(name, values) in enumerate(series.items()):
        points=[]
        for i,v in enumerate(values):
            x,y=xy(i,v); points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[si%len(colors)]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[si%len(colors)]}" stroke-width="2"/>')
    for i,label in enumerate(labels):
        x,_=xy(i,ymin); svg.append(f'<text x="{x:.1f}" y="{t+ph+23}" text-anchor="middle" font-size="12">{label}</text>')
    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{title}</text><text x="22" y="{h/2}" text-anchor="middle" transform="rotate(-90 22 {h/2})" font-size="14">{y_label}</text>')
    for i,name in enumerate(series):
        y=h-42+(i//3)*18; x=l+(i%3)*220
        svg.append(f'<line x1="{x}" y1="{y-4}" x2="{x+22}" y2="{y-4}" stroke="{colors[i%len(colors)]}" stroke-width="3"/><text x="{x+28}" y="{y}" font-size="12">{name}</text>')
    svg.append('</g></svg>')
    path.write_text(''.join(svg),encoding='utf-8')


def save_calibration_svg(y, p, path):
    frame = pd.DataFrame({"observed": y, "predicted": p})
    frame["bin"] = pd.qcut(frame.predicted, q=10, duplicates="drop")
    bins = frame.groupby("bin", observed=True).agg(predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size"))
    width, height, left, top, right, bottom = 760, 500, 85, 42, 35, 75
    plot_w, plot_h = width-left-right, height-top-bottom
    points=[]
    for row in bins.itertuples():
        x=left+plot_w*row.predicted; ycoord=top+plot_h*(1-row.observed)
        points.append(f'<circle cx="{x:.1f}" cy="{ycoord:.1f}" r="6" fill="#147d64"/>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial" fill="#294a40"><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#d68a2d" stroke-dasharray="7 6" stroke-width="2"/>{''.join(points)}<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#496b5d"/><line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#496b5d"/><text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">Mean predicted event probability</text><text x="20" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot_h/2})">Observed event rate</text><text x="{left}" y="24" font-size="17" font-weight="bold">120-minute nested held-out calibration (seed 42)</text></g></svg>'''
    path.write_text(svg, encoding="utf-8")
    return [{"mean_predicted": float(row.predicted), "observed_rate": float(row.observed), "samples": int(row.n)} for row in bins.itertuples()]


def primary_ops(seed_results, pred_sets, precision_target=.40):
    rows=[]
    policies=("top_1_percent","top_5_percent","top_10_percent","precision_40_percent")
    for res, df in zip(seed_results,pred_sets):
        seed=int(res["seed"])
        folds=sorted(df.outer_fold.unique())
        for fold in folds:
            part=df.loc[df.outer_fold.eq(fold)].copy()
            train_people=set(df.loc[df.outer_fold.ne(fold),"participant_id"])
            # For this policy, thresholds must come from an inner OOF model fit
            # on exactly the outer training participants. Re-create the inner OOF
            # scores under the selected fold configuration stored in the seed result.
            fold_info=res["folds"][int(fold)-1]
            params=fold_info["selected_params"]
            train_index=np.flatnonzero(df.outer_fold.ne(fold).to_numpy())
            xcols=fold_info["feature_columns"] if "feature_columns" in fold_info else None
            # thresholds populated by nested_classification_with_ops below
            for policy in policies:
                rows.append({"seed":seed,"fold":int(fold),"policy":policy,"precision":np.nan,"recall":np.nan,
                             "alerts":0,"alerts_per_participant_day":0.0,"participants_with_true_positive_caught":0})
    return rows


def full_nested_primary(data, feature_cols):
    # Primary repeated nested evaluation includes training-only threshold selection
    # and policy evaluation for each outer held-out fold.
    X=data[feature_cols].astype(float).reset_index(drop=True); y=data.label.to_numpy(int); groups=data.participant_id.to_numpy()
    seeds_out=[]; predictions=[]; params_by_seed={}
    for seed in REPEATED_SEEDS:
        oof=pd.DataFrame({"participant_id":data.participant_id,"timestamp":data.timestamp,"label":y,"target_glucose":data.target_glucose})
        oof["score"]=np.nan; oof["outer_fold"]=-1; folds=[]
        for fold,(tr,te) in enumerate(balanced_outer_group_splits(y,groups,seed,n=5),1):
            params,inner_oof,search=tune_class(X.iloc[tr].reset_index(drop=True),y[tr],groups[tr],seed+fold)
            model=make_clf(params).fit(X.iloc[tr],y[tr]); scores=model.predict_proba(X.iloc[te])[:,1]
            oof.loc[te,"score"]=scores; oof.loc[te,"outer_fold"]=fold
            policies={}
            for budget in (.01,.05,.10):
                k=max(1,int(np.ceil(len(inner_oof)*budget)))
                threshold=float(np.partition(inner_oof,-k)[-k])
                pol=f"top_{int(budget*100)}_percent"
                policies[pol]={"threshold":threshold,"training_alert_fraction":float(np.mean(inner_oof>=threshold))}
            t40=choose_precision_threshold(y[tr],inner_oof,.40)
            policies["precision_40_percent"]={"threshold":t40,"training_alert_fraction":float(np.mean(inner_oof>=t40)) if t40 is not None else 0.0,
                                               "training_precision":float(precision_score(y[tr],inner_oof>=t40)) if t40 is not None else None}
            fold_df=oof.iloc[te].copy()
            op={}
            for pol,setup in policies.items():
                alerts=np.zeros(len(te),dtype=int) if setup["threshold"] is None else (scores>=setup["threshold"]).astype(int)
                op[pol]={**op_stats(fold_df,alerts),**setup}
            folds.append({"seed":seed,"fold":fold,"outer_test_samples":len(te),"outer_test_participants":len(set(groups[te])),
                          "test_participant_ids":sorted(map(str,set(groups[te]))),
                          "selected_params":params,"edge_hits":edge_hits(params),"inner_search":search,
                          "outer_metrics":class_metrics(y[te],scores),"operating_points":op,
                          "feature_columns":feature_cols})
        oof["seed"]=seed; predictions.append(oof); params_by_seed[seed]=folds
        seedrow={"seed":seed,"pooled":class_metrics(y,oof.score),"participant_macro":participant_macro(oof,"score"),"folds":folds}
        # Aggregate decision policies across five held-out folds. Alert counts and
        # caught people are recomputed on the joined test predictions, not averaged.
        seedrow["operating_points"]={}
        for pol in ("top_1_percent","top_5_percent","top_10_percent","precision_40_percent"):
            alerts_all=np.zeros(len(oof),dtype=int); thresholds=[]
            for fold in folds:
                te=oof.outer_fold.eq(fold["fold"]).to_numpy(); threshold=fold["operating_points"][pol]["threshold"]
                alerts_all[te]=(oof.loc[te,"score"].to_numpy()>=threshold).astype(int) if threshold is not None else 0
                thresholds.append(threshold)
            seedrow["operating_points"][pol]={**op_stats(oof,alerts_all),"fold_thresholds":thresholds}
        seeds_out.append(seedrow)
        print(f"primary repeated CV seed {seed}",flush=True)
    return seeds_out,predictions,params_by_seed


def evaluate_horizon_nested(data, feature_cols, seed=42, regress=True):
    X=data[feature_cols].astype(float).reset_index(drop=True); y=data.label.to_numpy(int); yreg=data.target_glucose.to_numpy(float); groups=data.participant_id.to_numpy()
    oof=pd.DataFrame({"participant_id":data.participant_id,"timestamp":data.timestamp,"label":y,"target_glucose":yreg})
    oof["score"]=np.nan; oof["regression"]=np.nan; oof["outer_fold"]=-1; folds=[]
    for fold,(tr,te) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups),1):
        params,inner_oof,search=tune_class(X.iloc[tr].reset_index(drop=True),y[tr],groups[tr],seed+fold)
        model=make_clf(params).fit(X.iloc[tr],y[tr]); score=model.predict_proba(X.iloc[te])[:,1]
        oof.loc[te,"score"]=score; oof.loc[te,"outer_fold"]=fold
        regparams,regsearch=tune_reg(X.iloc[tr].reset_index(drop=True),yreg[tr],groups[tr],seed+fold)
        reg=make_reg(regparams).fit(X.iloc[tr],yreg[tr]); rp=reg.predict(X.iloc[te]); oof.loc[te,"regression"]=rp
        folds.append({"fold":fold,"test_participant_ids":sorted(map(str,set(groups[te]))),
                      "classifier_params":params,"classifier_edge_hits":edge_hits(params),"classifier_search":search,
                      "regression_params":regparams,"regression_edge_hits":edge_hits(regparams),"regression_search":regsearch,
                      "classification":class_metrics(y[te],score),"regression":regression_metrics(yreg[te],rp)})
        print(f"horizon {int(data.horizon_minutes.iloc[0])}: fold {fold}/5",flush=True)
    return {"horizon_minutes":int(data.horizon_minutes.iloc[0]),"participants":int(data.participant_id.nunique()),
            "samples":int(len(data)),"events":int(y.sum()),"pooled_classification":class_metrics(y,oof.score),
            "participant_macro_classification":participant_macro(oof,"score"),
            "regression_pooled":regression_metrics(yreg,oof.regression),
            "regression_participant_macro":{"mae_mg_dl":float(oof.assign(ae=(oof.regression-oof.target_glucose).abs()).groupby("participant_id").ae.mean().mean()),
                "rmse_mg_dl":float(np.sqrt(oof.assign(se=(oof.regression-oof.target_glucose)**2).groupby("participant_id").se.mean()).mean())},
            "outer_fold_macro_classification":{k:float(np.mean([f["classification"][k] for f in folds])) for k in ("pr_auc","roc_auc","brier")},
            "folds":folds},oof


def run_group_ablation(data, all_features, picked_folds, seed=42):
    X=data[all_features].astype(float).reset_index(drop=True); y=data.label.to_numpy(int); groups=data.participant_id.to_numpy()
    ysplit=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    splits=list(ysplit.split(X,y,groups))
    group_map=classify_features(data)[2]
    result={}
    for group,removed in group_map.items():
        cols=[c for c in all_features if c not in removed]
        scores=np.full(len(data),np.nan)
        for fold,(tr,te) in enumerate(splits,1):
            params=picked_folds[seed][fold-1]["selected_params"]
            model=make_clf(params).fit(X[cols].iloc[tr],y[tr]);scores[te]=model.predict_proba(X[cols].iloc[te])[:,1]
        tmp=data[["participant_id","label"]].copy();tmp["score"]=scores
        result[group]={"removed_features":removed,"remaining_features":len(cols),"pooled":class_metrics(y,scores),"participant_macro":participant_macro(tmp,"score"),"selection_note":"No retuning after feature removal; each outer fold reuses settings selected using all feature groups inside its training partition."}
    return result


def run_variants(data, base_features, seed=42):
    variants={"reduced_five_ehr":base_features,"hba1c_only":base_features}
    _,dynamic,groups=classify_features(data)
    variants["reduced_five_ehr"]=dynamic+REDUCED_EHR
    variants["hba1c_only"]=dynamic+["ehr_hba1c_pct"]
    out={}
    for name,cols in variants.items():
        rows,preds,picked=nested_classification(data,cols,(seed,),name)
        out[name]={"pooled":rows[0]["pooled"],"participant_macro":rows[0]["participant_macro"],
                   "outer_fold_macro":{k:float(np.mean([f["test_metrics"][k] for f in rows[0]["folds"]])) for k in ("pr_auc","roc_auc","brier")},
                   "selected_settings":rows[0]["folds"]}
        # Save only local predictions under ignored processed data; never copy them to docs.
        preds[0].to_csv(OUT/f"variant_{name}_oof_local.csv",index=False)
        print(f"variant {name} complete",flush=True)
    return out


def edge_summary(picked):
    flat=[f for seed in picked.values() for f in seed]
    keys=list(flat[0]["edge_hits"]) if flat else []
    return {k:{"selected_folds":sum(bool(f["edge_hits"][k]) for f in flat),"total_selected_folds":len(flat)} for k in keys}


def main():
    data_by_h=make_all_horizon_data()
    base=data_by_h[120]; all_features,dynamic,groups=classify_features(base)
    # Protect full-row consistency before any model fitting.
    feature_groups={name:cols for name,cols in groups.items()}
    primary_seeds,primary_oof,picked=full_nested_primary(base,all_features)
    for seed,df in zip(REPEATED_SEEDS,primary_oof):
        df.to_csv(OUT/f"primary_120m_seed_{seed}_oof_local.csv",index=False)
    primary_spread=repeated_summary(primary_seeds)
    primary_seed42=primary_oof[list(REPEATED_SEEDS).index(42)]
    calibration_bins=save_calibration_svg(primary_seed42.label.to_numpy(), primary_seed42.score.to_numpy(), DOC_FIGURES/"calibration_120m.svg")
    # Alert operating characteristics across repeat assignments.
    op_names=("top_1_percent","top_5_percent","top_10_percent","precision_40_percent")
    op_summary={}
    for name in op_names:
        vals=[r["operating_points"][name] for r in primary_seeds]
        op_summary[name]={"mean_precision":float(np.nanmean([v["precision"] if v["precision"] is not None else np.nan for v in vals])),
                          "mean_recall":float(np.nanmean([v["recall"] if v["recall"] is not None else np.nan for v in vals])),
                          "mean_alerts_per_participant_day":float(np.mean([v["alerts_per_participant_day"] for v in vals])),
                          "mean_participants_with_true_positive_caught":float(np.mean([v["participants_with_true_positive_caught"] for v in vals])),
                          "per_seed":vals}
    edge= edge_summary(picked)
    # Compare reduced-feature HGB variants under the same inner/outer participant folds.
    variants=run_variants(base,all_features)
    # Rich feature group ablation uses primary seed's fold-specific selected settings.
    ablation=run_group_ablation(base,all_features,picked,seed=42)
    horizon_results={120:{"primary_repeated":primary_spread}}
    for h in (30,60):
        result,oof=evaluate_horizon_nested(data_by_h[h],all_features,seed=42)
        oof.to_csv(OUT/f"horizon_{h}m_oof_local.csv",index=False)
        horizon_results[h]=result
    # HGB 120m regression nested evaluation is included in the horizon-120 run.
    reg120,reg_oof=evaluate_horizon_nested(base,all_features,seed=42)
    reg_oof.to_csv(OUT/"horizon_120m_regression_local.csv",index=False)
    horizon_results[120]["classification_regression_nested"]={k:v for k,v in reg120.items() if k!="folds"}
    horizon_results[120]["regression_folds"]=reg120["folds"]
    # Save all granular per-row outputs locally; aggregate documentation never contains these files.
    primary_all=pd.concat(primary_oof,ignore_index=True)
    primary_all.to_csv(OUT/"primary_120m_repeated_oof_local.csv",index=False)
    base.to_csv(OUT/"rich_120m_samples_local.csv",index=False)
    payload={"dataset":{"participants":int(base.participant_id.nunique()),"usable_samples_by_horizon":{str(h):len(d) for h,d in data_by_h.items()},
                         "event_counts_by_horizon":{str(h):int(d.label.sum()) for h,d in data_by_h.items()},"features":all_features,
                         "feature_groups":feature_groups},
             "search":{"configuration_candidates":GRID,"objective_classifier":"inner grouped-fold average precision",
                       "objective_regression":"inner grouped-fold MAE","outer_folds":5,"inner_folds":3,
                       "repeat_seeds":list(REPEATED_SEEDS),"selected_edge_counts":edge},
             "primary_120m_repeated_cv":{"seed_results":primary_seeds,"spread":primary_spread,"operating_points":op_summary},
             "calibration_bins_seed_42":calibration_bins,
             "nested_variants":variants,"feature_group_ablation":ablation,"multi_horizon":horizon_results,
             "notes":["All preprocessing is fitted only on each training split.",
                      "Continuous target metrics cover all participants/samples.",
                      "Macro classification averages participants equally; PR-AUC/ROC-AUC omit participants without both label classes as relevant.",
                      "Carbohydrate-on-board is a linear-decay feature proxy, not a validated physiology model.",
                      "Repeated fold seeds change participant assignment, not the cohort."]}
    local_json=OUT/"revised_full_metrics.json"
    local_json.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    # Aggregate-only doc artifact; remove raw fold score search traces and per-sample files.
    aggregate={k:v for k,v in payload.items() if k not in ("dataset",)}
    aggregate["dataset"]={k:v for k,v in payload["dataset"].items() if k not in ("features","feature_groups")}
    (DOC_RESULTS/"aggregate_metrics.json").write_text(json.dumps(aggregate,indent=2,allow_nan=False),encoding="utf-8")
    save_svg(DOC_FIGURES/"repeated_cv_spread.svg","Repeated participant-grouped CV by assignment seed",
             {"pooled PR-AUC":[x["pooled"]["pr_auc"] for x in primary_spread["by_seed"]],
              "participant macro PR-AUC":[x["participant_macro"]["pr_auc"] for x in primary_spread["by_seed"]]},
             [str(s) for s in REPEATED_SEEDS],"PR-AUC")
    save_svg(DOC_FIGURES/"repeated_cv_roc.svg","Pooled vs participant-macro ROC-AUC across fold seeds",
             {"pooled ROC-AUC":[x["pooled"]["roc_auc"] for x in primary_spread["by_seed"]],
              "participant macro ROC-AUC":[x["participant_macro"]["roc_auc"] for x in primary_spread["by_seed"]]},
             [str(s) for s in REPEATED_SEEDS],"ROC-AUC")
    hlist=[30,60,120]
    save_svg(DOC_FIGURES/"multi_horizon.svg","Nested classification performance by forecast horizon",
             {"pooled PR-AUC":[horizon_results[h]["classification_regression_nested"]["pooled_classification"]["pr_auc"] if h==120 else horizon_results[h]["pooled_classification"]["pr_auc"] for h in hlist],
              "participant macro PR-AUC":[horizon_results[h]["classification_regression_nested"]["participant_macro_classification"]["pr_auc"] if h==120 else horizon_results[h]["participant_macro_classification"]["pr_auc"] for h in hlist]},
             [f"{h}m" for h in hlist],"PR-AUC")
    print(f"Saved aggregate results: {DOC_RESULTS/'aggregate_metrics.json'}",flush=True)


if __name__=="__main__":
    main()
