"""Participant-level subgroup, calibration, and bootstrap evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from project_config import BOOTSTRAP_SEED
from train_cgmacros_forecast import BIO_PATH, OOF_PATH, bio_id_to_folder_id

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_JSON = PROCESSED_DIR / "cgmacros_extended_evaluation.json"
CALIBRATION_SVG = DOCS_DIR / "calibration.svg"
N_BOOTSTRAPS = 2000
SCORE_COLUMN = "score_gradient_boosting_fused"


def metric_set(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | None]:
    if weights is not None:
        keep = weights > 0
        labels, scores, weights = labels[keep], scores[keep], weights[keep]
    if labels.size == 0:
        return {"pr_auc": None, "roc_auc": None, "brier_score": None}
    return {
        "pr_auc": float(average_precision_score(labels, scores, sample_weight=weights))
        if labels.sum() > 0
        else None,
        "roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights))
        if np.unique(labels).size > 1
        else None,
        "brier_score": float(brier_score_loss(labels, scores, sample_weight=weights)),
    }


def bootstrap_ci(
    frame: pd.DataFrame,
    participant_ids: list[str],
    iterations: int = N_BOOTSTRAPS,
) -> dict[str, dict[str, float | None]]:
    """Resample participants (clusters), retaining all their timestamp rows."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, "".join(participant_ids))))
    ids = np.asarray(participant_ids, dtype=str)
    group_codes = pd.Categorical(frame["participant_id"], categories=ids).codes
    y = frame["label"].to_numpy(dtype=int)
    scores = frame[SCORE_COLUMN].to_numpy(dtype=float)
    values: dict[str, list[float]] = {key: [] for key in ("pr_auc", "roc_auc", "brier_score")}
    for _ in range(iterations):
        draws = rng.integers(0, len(ids), size=len(ids))
        counts = np.bincount(draws, minlength=len(ids))
        weights = counts[group_codes]
        metrics = metric_set(y, scores, weights)
        for name, value in metrics.items():
            if value is not None and np.isfinite(value):
                values[name].append(value)
    result = {}
    for name, samples in values.items():
        result[name] = {
            "lower_95": float(np.quantile(samples, 0.025)) if samples else None,
            "upper_95": float(np.quantile(samples, 0.975)) if samples else None,
            "valid_replicates": len(samples),
        }
    return result


def calibration_bins(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    frame = pd.DataFrame({"observed": labels, "predicted": scores})
    frame["bin"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted=("predicted", "mean"),
        event_rate=("observed", "mean"),
        count=("observed", "size"),
    )
    return [
        {
            "mean_predicted": float(row.mean_predicted),
            "event_rate": float(row.event_rate),
            "count": int(row.count),
        }
        for row in grouped.itertuples()
    ]


def write_calibration_svg(bins: list[dict[str, float | int]]) -> None:
    width, height = 760, 500
    left, right, top, bottom = 85, 32, 40, 78
    plot_w, plot_h = width - left - right, height - top - bottom

    def point(x: float, y: float) -> tuple[float, float]:
        return left + plot_w * x, top + plot_h * (1 - y)

    grid = []
    for tick in np.linspace(0, 1, 6):
        x, y = point(float(tick), float(tick))
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e9e5"/>'
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e9e5"/>'
            f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle">{tick:.1f}</text>'
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>'
        )
    points = []
    for item in bins:
        x, y = point(float(item["mean_predicted"]), float(item["event_rate"]))
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#16745b"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/><g font-family="Arial,sans-serif" font-size="13" fill="#36564b">
{''.join(grid)}<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#c18a36" stroke-dasharray="7 6" stroke-width="2"/>
{''.join(points)}<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#496b5d"/><line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#496b5d"/>
<text x="{left + plot_w/2}" y="{height - 20}" text-anchor="middle">Mean predicted event probability</text>
<text x="20" y="{top + plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h/2})">Observed event rate</text>
<text x="{left}" y="24" font-size="17" font-weight="700">Calibration: pooled out-of-fold predictions</text>
</g></svg>'''
    CALIBRATION_SVG.write_text(svg, encoding="utf-8")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(OOF_PATH)
    predictions["participant_id"] = predictions["participant_id"].astype(str).str.zfill(3)
    bio = pd.read_csv(BIO_PATH, usecols=["subject", "A1c PDL (Lab)"])
    bio["participant_id"] = bio["subject"].map(bio_id_to_folder_id)
    bio["hba1c"] = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    bio["glycaemic_group"] = np.select(
        [bio["hba1c"] < 5.7, bio["hba1c"].between(5.7, 6.4, inclusive="both"), bio["hba1c"] >= 6.5],
        ["Healthy-range HbA1c", "Prediabetes-range HbA1c", "Diabetes-range HbA1c"],
        default="Unclassified",
    )
    frame = predictions.merge(
        bio[["participant_id", "hba1c", "glycaemic_group"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.loc[frame[SCORE_COLUMN].notna() & frame["glycaemic_group"].ne("Unclassified")].copy()
    y = frame["label"].to_numpy(dtype=int)
    scores = frame[SCORE_COLUMN].to_numpy(dtype=float)
    participants = sorted(frame["participant_id"].unique())
    overall = {
        "participants": len(participants),
        "samples": int(len(frame)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        **metric_set(y, scores),
        "participant_bootstrap_95_ci": bootstrap_ci(frame, participants),
    }
    groups = {}
    for name, part in frame.groupby("glycaemic_group", sort=False):
        part_ids = sorted(part["participant_id"].unique())
        labels = part["label"].to_numpy(dtype=int)
        values = part[SCORE_COLUMN].to_numpy(dtype=float)
        groups[str(name)] = {
            "participants": len(part_ids),
            "samples": int(len(part)),
            "events": int(labels.sum()),
            "event_rate": float(labels.mean()),
            **metric_set(labels, values),
            "participant_bootstrap_95_ci": bootstrap_ci(part, part_ids),
        }
    bins = calibration_bins(y, scores)
    write_calibration_svg(bins)
    result = {
        "model": "gradient_boosting_fused",
        "prediction": "Libre glucose >= 180 mg/dL at t+120 minutes",
        "grouping": "Baseline HbA1c bins from CGMacros bio.csv: <5.7%, 5.7-6.4%, >=6.5%; these are analytic strata, not independent clinical diagnoses.",
        "evaluation": "Grouped out-of-fold predictions; participant-level bootstrap resamples participants with all their timestamps (2,000 replicates). Intervals are conditional on existing out-of-fold predictions and do not include model-refit uncertainty.",
        "overall": overall,
        "groups": groups,
        "calibration_bins": bins,
        "calibration_plot": str(CALIBRATION_SVG.relative_to(ROOT)).replace("\\", "/"),
        "limitations": [
            "Only 45 participants; subgroup estimates and bootstrap intervals are uncertain.",
            "Calibration uses pooled out-of-fold predictions and has no separate external cohort.",
            "Calibration is descriptive; no post-hoc probability calibration was fitted.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"overall": overall, "groups": groups, "calibration_plot": result["calibration_plot"]}, indent=2))


if __name__ == "__main__":
    main()
