"""Exploratory event-level alert workload from training-selected OOF thresholds."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from revised_evaluation import (OUT, DOC_RESULTS, DOC_FIGURES, balanced_outer_group_splits,
                                make_all_horizon_data, classify_features, tune_class,
                                choose_precision_threshold, save_svg)

SEED = 42
POLICIES = (0.01, 0.05, 0.10)
MERGE_GAP_MINUTES = 30
COOLDOWN_MINUTES = 60


def episodes(rows, horizon):
    targets = pd.to_datetime(rows.timestamp) + pd.Timedelta(minutes=horizon)
    vals = pd.to_numeric(rows.target_glucose, errors="coerce").to_numpy(float)
    highs = [(t, v) for t, v in zip(targets, vals) if np.isfinite(v) and v >= 180]
    out = []
    for ts, value in highs:
        if not out or ts - out[-1]["end"] > pd.Timedelta(minutes=MERGE_GAP_MINUTES):
            out.append({"start": ts, "end": ts, "max": float(value)})
        else:
            out[-1]["end"] = ts
            out[-1]["max"] = max(out[-1]["max"], float(value))
    return out


def analyze(frame, horizon):
    emitted = false = ongoing = detected = total = any_alert = repeated = none = people_detected = 0
    leads, people = [], []
    for pid, p in frame.groupby("participant_id", sort=True):
        p = p.sort_values("timestamp").reset_index(drop=True)
        eps = episodes(p, horizon)
        total += len(eps)
        alerts, last = [], None
        for _, row in p.loc[p.is_alert].iterrows():
            origin = pd.Timestamp(row.timestamp)
            if last is None or origin - last >= pd.Timedelta(minutes=COOLDOWN_MINUTES):
                alerts.append({"origin": origin, "target": origin + pd.Timedelta(minutes=horizon), "score": float(row.score)})
                last = origin
        emitted += len(alerts)
        any_alert += int(bool(alerts)); repeated += int(len(alerts) >= 2); none += int(not alerts)
        used = set()
        person_detected = 0
        for ei, ep in enumerate(eps):
            candidates = [(i, a) for i, a in enumerate(alerts)
                          if a["origin"] < ep["start"] and ep["start"] <= a["target"] <= ep["end"]]
            if candidates:
                i, a = min(candidates, key=lambda item: item[1]["origin"])
                used.add(i); detected += 1
                person_detected += 1
                leads.append((ep["start"] - a["origin"]).total_seconds() / 60)
        people_detected += int(person_detected > 0)
        for i, a in enumerate(alerts):
            target_ep = next((ep for ep in eps if ep["start"] <= a["target"] <= ep["end"]), None)
            if i not in used:
                if target_ep is None or a["origin"] < target_ep["start"]:
                    false += 1
                else:
                    ongoing += 1
        people.append({"participant_id": str(pid), "episodes": len(eps), "alerts_after_cooldown": len(alerts)})
    opportunities_days = len(frame) / 96.0
    return {"observed_glucose_episodes": int(total), "episodes_detected_early": int(detected),
            "episodes_missed": int(total-detected), "event_recall": float(detected/total) if total else None,
            "event_precision_vs_false_alerts": float(detected/(detected+false)) if detected+false else None,
            "emitted_alerts_after_cooldown": int(emitted), "false_alerts_after_grouping": int(false),
            "false_alerts_per_participant_day": float(false/opportunities_days) if opportunities_days else 0.0,
            "alerts_during_already_started_episode": int(ongoing),
            "median_lead_minutes": float(np.median(leads)) if leads else None,
            "mean_lead_minutes": float(np.mean(leads)) if leads else None,
            "people_with_any_alert": int(any_alert), "people_with_repeated_alerts": int(repeated),
            "people_with_at_least_one_episode_detected": int(people_detected),
            "people_with_no_alerts": int(none), "participants": int(frame.participant_id.nunique()),
            "participant_prediction_days_approx": float(opportunities_days),
            "participant_alert_counts_local": people}


def outer_splits(data, horizon):
    y, groups = data.label.to_numpy(int), data.participant_id.to_numpy()
    if horizon == 120:
        return balanced_outer_group_splits(y, groups, SEED, 5)
    X = data.drop(columns=["participant_id", "timestamp", "target_glucose", "label", "horizon_minutes"], errors="ignore")
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))


def main():
    data_by_horizon = make_all_horizon_data()
    results, all_oof = {}, []
    for horizon in (30, 60, 120):
        data = data_by_horizon[horizon].reset_index(drop=True)
        features, _, _ = classify_features(data)
        y, groups = data.label.to_numpy(int), data.participant_id.to_numpy()
        oof_path = OUT / ("primary_120m_seed_42_oof_local.csv" if horizon == 120 else f"horizon_{horizon}m_oof_local.csv")
        oof = pd.read_csv(oof_path, dtype={"participant_id": str}, parse_dates=["timestamp"])
        splits = outer_splits(data, horizon)
        if "outer_fold" not in oof:
            fold_by_person = {}
            for fold, (_, te) in enumerate(splits, 1):
                for pid in groups[te]:
                    fold_by_person[str(pid)] = fold
            oof["outer_fold"] = oof.participant_id.astype(str).map(fold_by_person)
            if oof.outer_fold.isna().any():
                raise ValueError(f"Could not reconstruct outer folds for {horizon}-minute OOF file")
        frame = data[["participant_id", "timestamp", "target_glucose", "label", "horizon_minutes"]].merge(
            oof[["participant_id", "timestamp", "score", "outer_fold"]], on=["participant_id", "timestamp"],
            how="left", validate="one_to_one")
        if frame.score.isna().any():
            raise ValueError(f"OOF predictions do not cover all {horizon}-minute samples")
        fold_thresholds = {}
        for fold, (tr, te) in enumerate(splits, 1):
            expected = set(map(str, groups[te]))
            actual = set(frame.loc[frame.outer_fold.eq(fold), "participant_id"].astype(str))
            if expected != actual:
                raise ValueError(f"Horizon {horizon}, fold {fold}: held-out participants mismatch")
            _, inner_oof, _ = tune_class(data[features].iloc[tr].reset_index(drop=True), y[tr], groups[tr], SEED+fold)
            fold_thresholds[fold] = {}
            for budget in POLICIES:
                k = max(1, int(np.ceil(len(inner_oof)*budget)))
                fold_thresholds[fold][f"top_{int(budget*100)}_percent"] = float(np.partition(inner_oof, -k)[-k])
            p40 = choose_precision_threshold(y[tr], inner_oof, 0.40)
            fold_thresholds[fold]["precision_40_percent"] = float(p40) if p40 is not None else None
            print(f"Episode workload: horizon {horizon} fold {fold}/5", flush=True)
        results[str(horizon)] = {}
        policy_names = [f"top_{int(budget*100)}_percent" for budget in POLICIES] + ["precision_40_percent"]
        for policy in policy_names:
            test = frame.copy()
            test["training_cutoff"] = test.outer_fold.map({f: v[policy] for f, v in fold_thresholds.items()})
            test["is_alert"] = test.score.ge(test.training_cutoff)
            summary = analyze(test, horizon)
            summary["policy"] = (f"Top {policy.split('_')[1]}% cutoff from inner OOF training-participant scores; applied unchanged to held-out participants."
                                  if policy.startswith("top_") else
                                  "Maximum inner-OOF recall at at least 40% training precision; cutoff applied unchanged to held-out participants.")
            summary["outer_fold_training_thresholds"] = [float(fold_thresholds[f][policy]) for f in range(1, 6)]
            results[str(horizon)][policy] = summary
        all_oof.append(frame)
    design = {"thresholds": "Top 1%, 5%, and 10% cutoffs derived from inner participant-grouped OOF predictions on training participants, applied unchanged to held-out outer folds.",
              "episode_rule": f"Observed future CGM readings >=180 mg/dL; merge readings with gaps <= {MERGE_GAP_MINUTES} minutes.",
              "alert_rule": f"Retain first threshold-crossing prediction then suppress alerts for {COOLDOWN_MINUTES} minutes. Both rules are exploratory assumptions requiring clinician input and prospective validation.",
              "detection_rule": "Early detection requires alert time before episode onset and forecast target within that episode; post-onset alerts are counted separately.",
              "denominator_note": "Participant prediction days approximate prediction opportunities divided by 96 at 15-minute cadence; recording gaps mean this is not exact wear time."}
    payload = {"design": design, "by_horizon_and_budget": results}
    (OUT/"alert_episode_metrics_local.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    aggregate_path = DOC_RESULTS/"aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    public = json.loads(json.dumps(payload))
    for policies in public["by_horizon_and_budget"].values():
        for metrics in policies.values():
            counts = metrics.pop("participant_alert_counts_local")
            metrics["participant_alert_count_distribution"] = {
                "zero_alert_people": metrics["people_with_no_alerts"],
                "one_alert_people": sum(x["alerts_after_cooldown"] == 1 for x in counts),
                "two_or_more_alert_people": metrics["people_with_repeated_alerts"]}
    aggregate["episode_level_alert_workload"] = public
    aggregate_path.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    top5 = [results[str(h)]["top_5_percent"] for h in (30, 60, 120)]
    save_svg(DOC_FIGURES/"episode_alert_horizon_comparison.svg", "Exploratory event workload by forecast horizon (top-5% cutoff)",
             {"Episodes detected (%)": [100*x["event_recall"] if x["event_recall"] is not None else 0 for x in top5],
              "False alerts / participant-day": [x["false_alerts_per_participant_day"] for x in top5],
              "Median lead (minutes)": [x["median_lead_minutes"] or 0 for x in top5]},
             ["30m", "60m", "120m"], "Rate / count")
    print(json.dumps({h:{p:{k:v.get(k) for k in ("observed_glucose_episodes","episodes_detected_early","episodes_missed","false_alerts_per_participant_day","median_lead_minutes","people_with_repeated_alerts","people_with_no_alerts")} for p,v in r.items()} for h,r in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
