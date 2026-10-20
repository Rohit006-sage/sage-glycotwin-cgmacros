"""Regenerate only the repeated primary grouped-CV aggregate with distinct folds."""
import json
import numpy as np
import pandas as pd

from revised_evaluation import (
    OUT, DOC_RESULTS, DOC_FIGURES, REPEATED_SEEDS, classify_features,
    full_nested_primary, edge_hits, save_svg,
)


def scrub_ids(value):
    if isinstance(value, dict):
        return {k: scrub_ids(v) for k, v in value.items()
                if k not in ("test_participant_ids", "participant_id", "participant_ids")}
    if isinstance(value, list):
        return [scrub_ids(v) for v in value]
    return value


data = pd.read_csv(OUT / "rich_120m_samples_local.csv", dtype={"participant_id": str}, parse_dates=["timestamp"])
features, _, _ = classify_features(data)
seed_results, prediction_sets, _ = full_nested_primary(data, features)
for seed, frame in zip(REPEATED_SEEDS, prediction_sets):
    frame.to_csv(OUT / f"primary_120m_seed_{seed}_oof_local.csv", index=False)
all_predictions = pd.concat(prediction_sets, ignore_index=True)
all_predictions.to_csv(OUT / "primary_120m_repeated_oof_local.csv", index=False)

metrics = ("pr_auc", "roc_auc")
by_seed = [{"seed": row["seed"],
            "pooled": {m: row["pooled"][m] for m in metrics},
            "participant_macro": {m: row["participant_macro"][m] for m in metrics}}
           for row in seed_results]
spread = {kind: {metric: {"mean": float(np.mean([r[kind][metric] for r in by_seed])),
                         "sd": float(np.std([r[kind][metric] for r in by_seed], ddof=1)),
                         "min": float(np.min([r[kind][metric] for r in by_seed])),
                         "max": float(np.max([r[kind][metric] for r in by_seed]))}
                 for metric in metrics}
          for kind in ("pooled", "participant_macro")}
ops = {}
for policy in ("top_1_percent", "top_5_percent", "top_10_percent", "precision_40_percent"):
    rows = [r["operating_points"][policy] for r in seed_results]
    ops[policy] = {
        "mean_precision": float(np.mean([r["precision"] for r in rows if r["precision"] is not None])) if any(r["precision"] is not None for r in rows) else None,
        "mean_recall": float(np.mean([r["recall"] for r in rows if r["recall"] is not None])) if any(r["recall"] is not None for r in rows) else None,
        "mean_alerts": float(np.mean([r["alerts"] for r in rows])),
        "mean_alerts_per_participant_day": float(np.mean([r["alerts_per_participant_day"] for r in rows])),
        "mean_participants_with_true_positive_caught": float(np.mean([r["participants_with_true_positive_caught"] for r in rows])),
        "per_seed": rows,
    }

edge_keys = list(edge_hits(seed_results[0]["folds"][0]["selected_params"]))
edge_counts = {key: {"selected_folds": sum(f["edge_hits"][key] for row in seed_results for f in row["folds"]),
                     "total_selected_folds": sum(len(row["folds"]) for row in seed_results)}
               for key in edge_keys}

path = DOC_RESULTS / "aggregate_metrics.json"
aggregate = json.loads(path.read_text(encoding="utf-8"))
aggregate["primary_120m_repeated_cv"] = {
    "seed_results": seed_results,
    "spread": {"by_seed": by_seed, "spread": spread},
    "operating_points": ops,
}
aggregate["search"]["selected_edge_counts"] = edge_counts
aggregate["search"]["outer_assignment"] = "seeded participant-level greedy balance on sample and event counts"
aggregate = scrub_ids(aggregate)
path.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")

save_svg(DOC_FIGURES / "repeated_cv_spread.svg", "Repeated participant-grouped CV by assignment seed",
         {"pooled PR-AUC": [r["pooled"]["pr_auc"] for r in by_seed],
          "participant macro PR-AUC": [r["participant_macro"]["pr_auc"] for r in by_seed]},
         [str(s) for s in REPEATED_SEEDS], "PR-AUC")
save_svg(DOC_FIGURES / "repeated_cv_roc.svg", "Pooled vs participant-macro ROC-AUC across fold seeds",
         {"pooled ROC-AUC": [r["pooled"]["roc_auc"] for r in by_seed],
          "participant macro ROC-AUC": [r["participant_macro"]["roc_auc"] for r in by_seed]},
         [str(s) for s in REPEATED_SEEDS], "ROC-AUC")
print("Updated repeated-CV aggregate with five distinct participant assignments.", flush=True)
