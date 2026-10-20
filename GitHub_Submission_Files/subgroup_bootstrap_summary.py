"""Aggregate HbA1c-stratum metrics and participant-cluster bootstrap intervals."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from revised_evaluation import OUT, DOC_RESULTS

SEED = 20261005
BOOTSTRAPS = 2000


def summarize(rows):
    y = rows.label.to_numpy(int)
    s = rows.score.to_numpy(float)
    ap = float(average_precision_score(y, s)) if y.sum() else None
    auc = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else None
    person_ap, person_auc = [], []
    for _, p in rows.groupby("participant_id", sort=False):
        py = p.label.to_numpy(int)
        ps = p.score.to_numpy(float)
        if py.sum():
            person_ap.append(float(average_precision_score(py, ps)))
        if np.unique(py).size == 2:
            person_auc.append(float(roc_auc_score(py, ps)))
    return {"pooled_pr_auc": ap, "pooled_roc_auc": auc,
            "participant_macro_pr_auc": float(np.mean(person_ap)) if person_ap else None,
            "participant_macro_pr_auc_n": len(person_ap),
            "participant_macro_roc_auc": float(np.mean(person_auc)) if person_auc else None,
            "participant_macro_roc_auc_n": len(person_auc),
            "pooled_brier": float(brier_score_loss(y, s))}


def main():
    samples = pd.read_csv(OUT/"rich_120m_samples_local.csv", dtype={"participant_id": str})
    oof = pd.read_csv(OUT/"primary_120m_seed_42_oof_local.csv", dtype={"participant_id": str})
    samples["participant_id"] = samples.participant_id.astype(str).str.zfill(3)
    oof["participant_id"] = oof.participant_id.astype(str).str.zfill(3)
    # The local source rows and OOF predictions share participant/time keys.
    frame = samples[["participant_id", "timestamp", "label", "ehr_hba1c_pct"]].merge(
        oof[["participant_id", "timestamp", "score"]], on=["participant_id", "timestamp"],
        how="inner", validate="one_to_one")
    if len(frame) != len(samples):
        raise ValueError("Seed-42 OOF scores do not cover all 120-minute samples")
    strata = {
        "healthy_range_lt_5_7": frame.ehr_hba1c_pct < 5.7,
        "prediabetes_range_5_7_to_lt_6_5": (frame.ehr_hba1c_pct >= 5.7) & (frame.ehr_hba1c_pct < 6.5),
        "diabetes_range_ge_6_5": frame.ehr_hba1c_pct >= 6.5,
        "all_participants": pd.Series(True, index=frame.index),
    }
    rng = np.random.default_rng(SEED)
    results = {}
    for name, mask in strata.items():
        group = frame.loc[mask].copy()
        base = summarize(group)
        participant_rows = {pid: p for pid, p in group.groupby("participant_id", sort=True)}
        ids = np.array(list(participant_rows), dtype=object)
        vals = {key: [] for key in ("pooled_pr_auc", "pooled_roc_auc")}
        for _ in range(BOOTSTRAPS):
            chosen = rng.choice(ids, size=len(ids), replace=True)
            blocks = []
            for draw, pid in enumerate(chosen):
                p = participant_rows[pid].copy()
                p["participant_id"] = f"bootstrap_{draw}"
                blocks.append(p)
            boot = pd.concat(blocks, ignore_index=True)
            by = boot.label.to_numpy(int)
            bs = boot.score.to_numpy(float)
            if by.sum():
                vals["pooled_pr_auc"].append(float(average_precision_score(by, bs)))
            if np.unique(by).size == 2:
                vals["pooled_roc_auc"].append(float(roc_auc_score(by, bs)))
        intervals = {}
        for key, samples_metric in vals.items():
            intervals[key] = {"lower_95": float(np.quantile(samples_metric, .025)),
                              "upper_95": float(np.quantile(samples_metric, .975)),
                              "valid_resamples": len(samples_metric), "requested_resamples": BOOTSTRAPS}
        results[name] = {
            "participants": int(group.participant_id.nunique()),
            "samples": int(len(group)), "events": int(group.label.sum()),
            "event_prevalence_constant_pr_auc_baseline": float(group.label.mean()),
            "participants_with_at_least_one_event": int(group.groupby("participant_id").label.sum().gt(0).sum()),
            "metrics_seed_42": base, "participant_cluster_bootstrap_95_ci": intervals,
        }
        print(f"Bootstrap subgroup {name} complete", flush=True)
    payload = {"seed": SEED, "requested_cluster_bootstraps": BOOTSTRAPS,
               "source": "120-minute seed-42 held-out OOF predictions; resampled participants as clusters.",
               "limitations": "Intervals condition on fixed out-of-fold predictions and do not include model-refit uncertainty. Healthy-range events come from very few participants; interpret subgroup estimates cautiously.",
               "strata": results}
    local_path = OUT/"subgroup_bootstrap_metrics_local.json"
    local_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    public_path = DOC_RESULTS/"subgroup_bootstrap_metrics.json"
    public_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    agg_path = DOC_RESULTS/"aggregate_metrics.json"
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    aggregate["hba1c_strata_seed_42_bootstrap"] = payload
    agg_path.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
