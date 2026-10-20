from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_config import (
    CANDIDATE_THRESHOLDS,
    FIXED_PRECISION,
    FFILL_TOLERANCE_MINUTES,
    GLUCOSE_THRESHOLD,
    HORIZON_MINUTES,
    RANDOM_SEED,
    SAMPLE_MINUTES,
)

ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = ROOT / "data" / "raw" / "cgmacros"
DATASET_DIR = next(
    (path for path in RAW_DATA_ROOT.iterdir() if path.is_dir() and "cgmacros" in path.name.lower()),
    None,
)
if DATASET_DIR is None:
    raise FileNotFoundError(f"No CGMacros dataset found under {RAW_DATA_ROOT}")

EXTRACT_DIR = DATASET_DIR / "extracted" / "CGMacros"
BIO_PATH = EXTRACT_DIR / "bio.csv"
AUDIT_PATH = ROOT / "cgmacros_audit.txt"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = PROCESSED_DIR / "cgmacros_model_metrics.json"
OOF_PATH = PROCESSED_DIR / "cgmacros_forecast_oof.csv"

RANDOM_STATE = RANDOM_SEED


def folder_id_to_bio_id(folder_name: str) -> str:
    """Map CGMacros-NNN folder suffixes to zero-padded numeric bio.subject IDs."""
    suffix = folder_name.removeprefix("CGMacros-")
    return f"{int(suffix):03d}"


def bio_id_to_folder_id(subject_value: object) -> str:
    """Map numeric bio.subject values to their CGMacros-NNN folder IDs."""
    return f"{int(float(subject_value)):03d}"


def optional_numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def read_participant_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Timestamp" not in frame or "Libre GL" not in frame:
        raise ValueError(f"Missing Timestamp or Libre GL in {path}")

    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame = frame.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    frame = frame.drop_duplicates("Timestamp", keep="last").set_index("Timestamp")

    intensity_column = next((name for name in ("METs", "Intensity") if name in frame), None)
    result = pd.DataFrame(index=frame.index)
    result["glucose"] = pd.to_numeric(frame["Libre GL"], errors="coerce")
    result["heart_rate"] = optional_numeric(frame, "HR" if "HR" in frame else None)
    result["activity_level"] = optional_numeric(frame, intensity_column)
    result["activity_calories"] = optional_numeric(
        frame, "Calories (Activity)" if "Calories (Activity)" in frame else None
    )

    for source, target in (
        ("Carbs", "carbs"),
        ("Calories", "food_calories"),
        ("Protein", "protein"),
        ("Fat", "fat"),
        ("Fiber", "fiber"),
    ):
        result[target] = optional_numeric(frame, source if source in frame else None)

    if "Meal Type" in frame:
        meal_text = frame["Meal Type"].fillna("").astype(str).str.strip()
        result["meal_event"] = meal_text.ne("")
    else:
        result["meal_event"] = False
    return result


def make_features(
    raw: pd.DataFrame,
    grid: pd.DatetimeIndex,
    ffill_tolerance_minutes: int = FFILL_TOLERANCE_MINUTES,
) -> pd.DataFrame:
    # Do not carry sensor values across long outages. A 30-minute maximum age
    # is shared across dynamic streams and recorded in project_config.py.
    observed = raw.reindex(
        grid,
        method="ffill",
        tolerance=pd.Timedelta(minutes=ffill_tolerance_minutes),
    )
    features = pd.DataFrame(index=grid)
    glucose = observed["glucose"]
    features["glucose_now"] = glucose
    features["glucose_available_at_t"] = glucose.notna().astype(float)

    for steps, minutes in ((1, 15), (2, 30), (4, 60), (8, 120)):
        features[f"glucose_lag_{minutes}m"] = glucose.shift(steps)
    features["glucose_change_15m"] = glucose - glucose.shift(1)
    features["glucose_change_60m"] = glucose - glucose.shift(4)
    features["glucose_slope_60m_per_min"] = (glucose - glucose.shift(4)) / 60.0
    features["glucose_mean_60m"] = glucose.rolling(5, min_periods=1).mean()
    features["glucose_std_60m"] = glucose.rolling(5, min_periods=2).std()
    features["glucose_mean_120m"] = glucose.rolling(9, min_periods=1).mean()
    features["glucose_max_120m"] = glucose.rolling(9, min_periods=1).max()
    slope = features["glucose_slope_60m_per_min"].fillna(0.0)
    features["glucose_slope_60m_per_min"] = slope
    features["linear_forecast_120m"] = glucose + slope * HORIZON_MINUTES

    for column in ("heart_rate", "activity_level", "activity_calories"):
        features[column] = observed[column]
        features[f"{column}_lag_60m"] = observed[column].shift(4)

    for column in ("carbs", "food_calories", "protein", "fat", "fiber"):
        rolling_total = raw[column].fillna(0).rolling("120min", min_periods=1).sum()
        features[f"{column}_last_120m"] = rolling_total.reindex(grid, method="ffill")

    meal_events = raw["meal_event"].astype(float)
    features["meal_events_last_120m"] = meal_events.rolling("120min", min_periods=1).sum().reindex(
        grid, method="ffill"
    )
    last_meal_time = pd.Series(raw.index, index=raw.index).where(raw["meal_event"]).ffill()
    last_meal_at_sample = last_meal_time.reindex(grid, method="ffill")
    features["minutes_since_meal"] = (
        grid.to_series(index=grid).to_numpy() - last_meal_at_sample.to_numpy()
    ) / np.timedelta64(1, "m")

    hour = grid.hour + grid.minute / 60.0
    features["time_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["time_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return features.replace([np.inf, -np.inf], np.nan)


def make_target(raw_glucose: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Align the label to the exact future CGM value at t + horizon."""
    future_times = grid + pd.Timedelta(minutes=HORIZON_MINUTES)
    return pd.Series(raw_glucose.reindex(future_times).to_numpy(), index=grid)


def build_samples(
    ffill_tolerance_minutes: int = FFILL_TOLERANCE_MINUTES,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    sample_frames = []
    participant_means = []
    threshold_audit = {threshold: {"samples": 0, "positives": 0, "positive_participants": set()} for threshold in CANDIDATE_THRESHOLDS}

    participant_files = sorted(EXTRACT_DIR.glob("CGMacros-*/CGMacros-*.csv"))
    for path in participant_files:
        participant_id = folder_id_to_bio_id(path.parent.name)
        raw = read_participant_file(path)
        raw = raw.loc[~raw.index.isna()]
        if raw.empty:
            continue

        participant_means.append(
            {
                "participant_id": participant_id,
                "mean_glucose": raw["glucose"].mean(),
            }
        )
        start = raw.index.min().ceil(f"{SAMPLE_MINUTES}min")
        end = raw.index.max().floor(f"{SAMPLE_MINUTES}min")
        if start > end:
            continue
        grid = pd.date_range(start=start, end=end, freq=f"{SAMPLE_MINUTES}min")
        future_glucose = make_target(raw["glucose"], grid)

        usable = future_glucose.notna()
        for threshold in CANDIDATE_THRESHOLDS:
            positives = future_glucose[usable] >= threshold
            threshold_audit[threshold]["samples"] += int(usable.sum())
            threshold_audit[threshold]["positives"] += int(positives.sum())
            if positives.any():
                threshold_audit[threshold]["positive_participants"].add(participant_id)

        features = make_features(raw, grid, ffill_tolerance_minutes).loc[usable].copy()
        frame = features
        frame["participant_id"] = participant_id
        frame["timestamp"] = grid[usable]
        frame["target_glucose_t_plus_120m"] = future_glucose[usable].to_numpy()
        frame["label"] = (frame["target_glucose_t_plus_120m"] >= GLUCOSE_THRESHOLD).astype(int)
        sample_frames.append(frame.reset_index(drop=True))

    if not sample_frames:
        raise ValueError("No usable forecast samples were built from the CGMacros files")
    samples = pd.concat(sample_frames, ignore_index=True)
    for values in threshold_audit.values():
        values["positive_participants"] = len(values["positive_participants"])
    audit_summary = {
        "participant_files": len(participant_files),
        "participants_with_samples": int(samples["participant_id"].nunique()),
        "sample_count_at_selected_threshold": len(samples),
        "positive_samples_at_selected_threshold": int(samples["label"].sum()),
        "positive_participants_at_selected_threshold": int(
            samples.loc[samples["label"].eq(1), "participant_id"].nunique()
        ),
        "threshold_candidates": threshold_audit,
    }
    mean_table = pd.DataFrame(participant_means)
    return samples, audit_summary, {"participant_means": mean_table}


def load_ehr_features() -> pd.DataFrame:
    """Return baseline clinical context from bio.csv, keyed by CGMacros participant ID."""
    bio = pd.read_csv(BIO_PATH)
    bio["participant_id"] = bio["subject"].map(bio_id_to_folder_id)
    # These measurements are recorded at study start. Do not use same-day fingersticks,
    # collection times, subject IDs, or ethnicity as model predictors.
    columns = {
        "Age": "ehr_age_years",
        "BMI": "ehr_bmi",
        "A1c PDL (Lab)": "ehr_hba1c_pct",
        "Fasting GLU - PDL (Lab)": "ehr_fasting_glucose_mg_dl",
        "Insulin ": "ehr_fasting_insulin",
        "Triglycerides": "ehr_triglycerides_mg_dl",
        "Cholesterol": "ehr_total_cholesterol_mg_dl",
        "HDL": "ehr_hdl_mg_dl",
        "LDL (Cal)": "ehr_ldl_mg_dl",
        "VLDL (Cal)": "ehr_vldl_mg_dl",
        "Cho/HDL Ratio": "ehr_cholesterol_hdl_ratio",
    }
    missing = sorted(set(columns) - set(bio.columns))
    if missing:
        raise ValueError(f"Missing expected baseline EHR columns in bio.csv: {missing}")
    features = bio[["participant_id", "Gender", *columns]].copy()
    features = features.rename(columns=columns)
    gender = features["Gender"].astype("string").str.strip().str.upper()
    features["ehr_gender_female"] = gender.map({"F": 1.0, "M": 0.0})
    features = features.drop(columns="Gender")
    for column in features.columns:
        if column != "participant_id":
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def join_check(participant_means: pd.DataFrame) -> dict[str, object]:
    bio = pd.read_csv(BIO_PATH)
    bio["participant_id"] = bio["subject"].map(bio_id_to_folder_id)
    joined = participant_means.merge(
        bio[["participant_id", "A1c PDL (Lab)" ]],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    complete = joined[["mean_glucose", "A1c PDL (Lab)"]].dropna()
    correlation = complete["mean_glucose"].corr(complete["A1c PDL (Lab)"])
    folder_ids = {folder_id_to_bio_id(path.parent.name) for path in EXTRACT_DIR.glob("CGMacros-*/CGMacros-*.csv")}
    bio_ids = set(bio["participant_id"])
    return {
        "mapping_rule": "CGMacros folder suffix NNN -> zero-pad numeric bio.csv subject to three digits; e.g. CGMacros-001 <-> subject 1",
        "folder_ids": len(folder_ids),
        "bio_rows": len(bio_ids),
        "matched_ids": len(folder_ids & bio_ids),
        "unmatched_folder_ids": sorted(folder_ids - bio_ids),
        "unmatched_bio_ids": sorted(bio_ids - folder_ids),
        "mean_glucose_hba1c_pearson_r": None if pd.isna(correlation) else float(correlation),
        "correlation_n": len(complete),
    }


def recall_at_fixed_precision(
    labels: np.ndarray, scores: np.ndarray, target_precision: float = FIXED_PRECISION
) -> dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    valid = np.flatnonzero(precision[:-1] >= target_precision)
    if not len(valid):
        return {"recall": 0.0, "threshold": None}
    best = valid[np.argmax(recall[valid])]
    return {"recall": float(recall[best]), "threshold": float(thresholds[best])}


def score_predictions(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    if len(np.unique(labels)) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(labels, scores))
    fixed_precision = recall_at_fixed_precision(labels, scores)
    return {
        "pr_auc_average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": roc_auc,
        "recall_at_80pct_precision": fixed_precision["recall"],
        "recall_at_80pct_precision_score_threshold": fixed_precision["threshold"],
        "brier_score": float(brier_score_loss(labels, scores)),
    }


def make_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for feature_set in ("sensor_only", "fused"):
        models[f"logistic_regression_{feature_set}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
        models[f"gradient_boosting_{feature_set}"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.08,
                        max_leaf_nodes=15,
                        l2_regularization=2.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return models


def run_grouped_cv(samples: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    feature_columns = [
        column
        for column in samples.columns
        if column not in {"participant_id", "timestamp", "target_glucose_t_plus_120m", "label"}
    ]
    dynamic_feature_columns = [column for column in feature_columns if not column.startswith("ehr_")]
    ehr_feature_columns = [column for column in feature_columns if column.startswith("ehr_")]
    if not ehr_feature_columns:
        raise ValueError("Baseline EHR features were not joined to the forecast samples")
    X = samples[feature_columns].astype(float)
    X_dynamic = samples[dynamic_feature_columns].astype(float)
    y = samples["label"].to_numpy(dtype=int)
    groups = samples["participant_id"].to_numpy()
    score_names = (
        "constant_prevalence",
        "persistence",
        "linear_trend",
        "logistic_regression_sensor_only",
        "logistic_regression_fused",
        "gradient_boosting_sensor_only",
        "gradient_boosting_fused",
    )
    oof_scores = {name: np.full(len(samples), np.nan, dtype=float) for name in score_names}
    fold_reports = []

    for fold_number, (train_idx, test_idx) in enumerate(grouped_splits(X, y, groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        if train_groups & test_groups:
            raise AssertionError("Participant leakage across grouped cross-validation fold")
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Fold {fold_number} training split contains only one class")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        fold_scores: dict[str, np.ndarray] = {
            "constant_prevalence": np.full(len(test_idx), float(y_train.mean())),
            "persistence": np.where(
                X_test["glucose_now"].notna(),
                (X_test["glucose_now"].fillna(-np.inf).to_numpy() >= GLUCOSE_THRESHOLD).astype(float),
                0.0,
            ),
            "linear_trend": np.nan_to_num(
                expit((X_test["linear_forecast_120m"].to_numpy() - GLUCOSE_THRESHOLD) / 15.0),
                nan=0.0,
            ),
        }
        for name, estimator in make_models().items():
            source = X_dynamic if name.endswith("sensor_only") else X
            estimator.fit(source.iloc[train_idx], y_train)
            fold_scores[name] = estimator.predict_proba(source.iloc[test_idx])[:, 1]

        for name, scores in fold_scores.items():
            oof_scores[name][test_idx] = scores

        fold_reports.append(
            {
                "fold": fold_number,
                "test_participants": int(len(test_groups)),
                "test_samples": int(len(test_idx)),
                "test_positives": int(y_test.sum()),
                "test_positive_participants": int(pd.Series(groups[test_idx][y_test == 1]).nunique()),
                "models": {
                    name: score_predictions(y_test, scores)
                    for name, scores in fold_scores.items()
                },
            }
        )

    if any(np.isnan(scores).any() for scores in oof_scores.values()):
        raise AssertionError("Some samples did not receive exactly one out-of-fold prediction")

    model_reports = {
        name: {
            **score_predictions(y, scores),
            "fold_metrics": [fold["models"][name] for fold in fold_reports],
        }
        for name, scores in oof_scores.items()
    }
    weighted_fold_metrics = {}
    for name in score_names:
        summary = {}
        for metric in (
            "pr_auc_average_precision",
            "roc_auc",
            "recall_at_80pct_precision",
            "brier_score",
        ):
            values = np.array(
                [fold["models"][name][metric] for fold in fold_reports],
                dtype=float,
            )
            weights = np.array([fold["test_samples"] for fold in fold_reports], dtype=float)
            mean = float(np.average(values, weights=weights))
            std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
            summary[metric] = {"mean": mean, "std": std}
        weighted_fold_metrics[name] = summary
    results = {
        "task": f"Predict Libre glucose >= {GLUCOSE_THRESHOLD:g} mg/dL at t+{HORIZON_MINUTES} minutes",
        "sample_unit": "participant x 15-minute timestamp",
        "features_cutoff": "Dynamic predictors use only time-series data observed at or before timestamp t; fused models additionally use baseline bio.csv clinical context collected at study start",
        "target_threshold_mg_dl": GLUCOSE_THRESHOLD,
        "horizon_minutes": HORIZON_MINUTES,
        "sample_minutes": SAMPLE_MINUTES,
        "validation": "5-fold StratifiedGroupKFold by participant; participant groups do not overlap between train and test",
        "fixed_precision_target": FIXED_PRECISION,
        "recall_at_fixed_precision_note": "Computed from each evaluation fold's precision-recall curve; pooled value uses all out-of-fold scores.",
        "models_in_baseline_order": list(score_names),
        "cross_validation_summary_method": "Sample-weighted mean and standard deviation of within-fold metrics; pooled out-of-fold metrics are also included for reference.",
        "feature_columns": feature_columns,
        "dynamic_feature_columns": dynamic_feature_columns,
        "ehr_feature_columns": ehr_feature_columns,
        "folds": fold_reports,
        "sample_weighted_fold_metrics": weighted_fold_metrics,
        "pooled_out_of_fold_metrics": model_reports,
        "out_of_fold_sample_count": len(samples),
        "participant_count": int(samples["participant_id"].nunique()),
        "positive_sample_count": int(y.sum()),
    }
    prediction_frame = samples[["participant_id", "timestamp", "target_glucose_t_plus_120m", "label"]].copy()
    for name, scores in oof_scores.items():
        prediction_frame[f"score_{name}"] = scores
    return results, prediction_frame


def grouped_splits(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Yield the shared participant-grouped CV splits used by training and tests."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return splitter.split(X, y, groups)


def make_audit_text(sample_audit: dict[str, object], join: dict[str, object]) -> str:
    lines = [
        "CGMacros time-series forecast audit",
        "====================================",
        f"Dataset root: {DATASET_DIR}",
        f"Participant CSV files: {sample_audit['participant_files']}",
        f"Participants with forecast samples: {sample_audit['participants_with_samples']}",
        "",
        "Threshold audit on 15-minute samples with an observed t+120-minute glucose:",
    ]
    for threshold, counts in sample_audit["threshold_candidates"].items():
        prevalence = counts["positives"] / counts["samples"] if counts["samples"] else 0.0
        lines.append(
            f"- {threshold:g} mg/dL: {counts['positives']} / {counts['samples']} positives "
            f"({prevalence:.2%}); {counts['positive_participants']} participants with >=1 positive"
        )
    selected = sample_audit["threshold_candidates"][GLUCOSE_THRESHOLD]
    lines.extend(
        [
            "",
            f"Selected target: Libre glucose at t+{HORIZON_MINUTES} minutes >= {GLUCOSE_THRESHOLD:g} mg/dL.",
            "Target threshold is domain/challenge chosen at 180 mg/dL; cohort event rates are descriptive and were not used to select the threshold.",
            f"Selected-target usable samples: {selected['samples']}; positives: {selected['positives']} ({selected['positives'] / selected['samples']:.2%}).",
            "Rows are sampled every 15 minutes on the wall-clock grid. A sample is retained only when an exact t+120-minute Libre glucose is present.",
            "",
            "Participant ID join check:",
            f"- Mapping rule: {join['mapping_rule']}",
            f"- Matched participant IDs: {join['matched_ids']} (folder IDs={join['folder_ids']}, bio rows={join['bio_rows']})",
            f"- Unmatched folder IDs: {join['unmatched_folder_ids']}",
            f"- Unmatched bio IDs: {join['unmatched_bio_ids']}",
            f"- Pearson correlation, participant mean Libre glucose vs HbA1c: {join['mean_glucose_hba1c_pearson_r']:.4f} (n={join['correlation_n']})",
            "Baseline bio.csv clinical measures are joined as static EHR context for the fused models. They are recorded at study start; future measurements, subject identifiers, same-day fingersticks, collection times, and ethnicity are not predictors.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    samples, sample_audit, join_inputs = build_samples()
    join = join_check(join_inputs["participant_means"])
    ehr = load_ehr_features()
    samples = samples.merge(ehr, on="participant_id", how="left", validate="many_to_one")
    if samples[ehr.columns.drop("participant_id")].isna().all(axis=None):
        raise ValueError("No baseline EHR values were joined to the forecast samples")
    results, predictions = run_grouped_cv(samples)
    results["join_check"] = join
    results["threshold_audit"] = sample_audit["threshold_candidates"]

    AUDIT_PATH.write_text(make_audit_text(sample_audit, join), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(OOF_PATH, index=False)

    print(AUDIT_PATH.read_text(encoding="utf-8"), end="")
    print("\nGrouped stratified 5-fold results")
    print("==================================")
    for fold in results["folds"]:
        print(
            f"Fold {fold['fold']}: participants={fold['test_participants']}, "
            f"samples={fold['test_samples']}, positives={fold['test_positives']}"
        )
    print("\nSample-weighted within-fold metrics (mean and standard deviation)")
    for model_name in results["models_in_baseline_order"]:
        print(f"{model_name}: {json.dumps(results['sample_weighted_fold_metrics'][model_name])}")
    print(f"\nSaved metrics: {METRICS_PATH}")
    print(f"Saved out-of-fold predictions: {OOF_PATH}")


if __name__ == "__main__":
    main()
