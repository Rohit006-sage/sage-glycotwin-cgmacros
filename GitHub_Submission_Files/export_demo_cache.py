"""Export the verified seed-42 120-minute demo model cache.

This is deliberately an export, not a new evaluation: it re-fits only the
five already-selected outer-fold models and refuses to write anything until
both stored held-out prediction columns match.
"""

from __future__ import annotations

import hashlib
import argparse
import importlib.metadata
import json
import platform
import subprocess
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
SEED = 42
HORIZON = 120
OUT = ROOT / "data" / "processed" / "models"
AGGREGATE = ROOT / "docs" / "results" / "aggregate_metrics.json"
STORED_PREDICTIONS = ROOT / "data" / "processed" / "revised_full" / "horizon_120m_regression_local.csv"
MANIFEST = OUT / "seed42_manifest.json"
ATOL = 1e-12
RTOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ("joblib", "numpy", "pandas", "scikit-learn")
    return {name: importlib.metadata.version(name) for name in names}


def selected_fold_settings() -> list[dict]:
    if not AGGREGATE.is_file():
        raise FileNotFoundError(f"Missing selected-settings source: {AGGREGATE}")
    payload = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    try:
        folds = payload["multi_horizon"]["120"]["regression_folds"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "aggregate_metrics.json does not contain multi_horizon.120.regression_folds; "
            "no regressor settings may be invented"
        ) from exc
    if len(folds) != evaluation.OUTER_FOLDS:
        raise RuntimeError(f"Expected {evaluation.OUTER_FOLDS} stored 120-minute fold settings, found {len(folds)}")
    required = ("classifier_params", "regression_params")
    for expected_fold, item in enumerate(folds, 1):
        if item.get("fold") != expected_fold or any(key not in item for key in required):
            raise RuntimeError(f"Incomplete selected settings for 120-minute fold {expected_fold}")
    return folds


def require_stored_predictions() -> pd.DataFrame:
    if not STORED_PREDICTIONS.is_file():
        raise FileNotFoundError(
            "Stored seed-42 held-out predictions are required before export. "
            "Restore the existing selected-evaluation prediction artifact before exporting."
        )
    stored = pd.read_csv(STORED_PREDICTIONS)
    required = {"participant_id", "timestamp", "score", "regression"}
    missing = sorted(required - set(stored.columns))
    if missing:
        raise RuntimeError(f"Stored predictions missing required columns {missing}: {STORED_PREDICTIONS}")
    return stored


def normalized_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    """Canonical matching keys only; never change training data or its row order."""
    required = {"participant_id", "timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing key columns: {missing}")
    people = []
    invalid = 0
    for value in frame["participant_id"]:
        text = str(value).strip()
        if pd.isna(value) or not text:
            invalid += 1
        try:
            numeric = Decimal(text)
            if not numeric.is_finite() or numeric != numeric.to_integral_value():
                invalid += 1
            else:
                text = str(int(numeric))
        except InvalidOperation:
            pass  # Non-numeric participant identifiers retain their spelling.
        people.append(text)
    times = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    invalid += int(times.isna().sum())
    if invalid:
        raise RuntimeError(f"Invalid key fields: count={invalid}; no cache written")
    # Explicit ns conversion: pandas may retain s/ms/us resolution on parsing.
    ticks = times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    return pd.MultiIndex.from_arrays([people, ticks], names=["participant_id", "timestamp"])


def verify_row_identity(data: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Match unique keys, returning references in the original training row order."""
    actual_keys, stored_keys = normalized_keys(data), normalized_keys(stored)
    key_columns = "participant_id,timestamp"
    if len(actual_keys) != len(stored_keys):
        raise RuntimeError(
            f"Key mismatch: key_columns={key_columns}; rebuilt_rows={len(actual_keys)}, "
            f"stored_rows={len(stored_keys)}, missing={max(0, len(actual_keys) - len(stored_keys))}, "
            f"extra={max(0, len(stored_keys) - len(actual_keys))}, first_mismatch_index=NOT_AVAILABLE; "
            "no cache written"
        )
    duplicates = (int(actual_keys.duplicated(keep=False).sum()),
                  int(stored_keys.duplicated(keep=False).sum()))
    if any(duplicates):
        raise RuntimeError(
            f"Duplicate key rows: key_columns={key_columns}; rebuilt={duplicates[0]}, "
            f"stored={duplicates[1]}, first_mismatch_index=NOT_AVAILABLE; no cache written"
        )
    missing = int((stored_keys.get_indexer(actual_keys) < 0).sum())
    extra = int((actual_keys.get_indexer(stored_keys) < 0).sum())
    if missing or extra:
        actual_positions = stored_keys.get_indexer(actual_keys)
        first = int(np.flatnonzero(actual_positions < 0)[0]) if missing else "NOT_AVAILABLE"
        raise RuntimeError(
            f"Key mismatch: key_columns={key_columns}; rebuilt_rows={len(actual_keys)}, "
            f"stored_rows={len(stored_keys)}, missing={missing}, extra={extra}, "
            f"first_mismatch_index={first}; no cache written"
        )
    return stored.iloc[stored_keys.get_indexer(actual_keys)].reset_index(drop=True)


def compare_predictions(actual: dict[str, np.ndarray], stored: pd.DataFrame) -> dict[str, float]:
    differences = {
        name: float(np.max(np.abs(values - stored[column].to_numpy(float))))
        for name, values, column in (
            ("classifier_score", actual["classifier_score"], "score"),
            ("regression_prediction", actual["regression_prediction"], "regression"),
        )
    }
    failures = []
    for name, maximum in differences.items():
        reference = stored["score" if name == "classifier_score" else "regression"].to_numpy(float)
        if not np.allclose(actual[name], reference, atol=ATOL, rtol=RTOL):
            failures.append(f"{name}: mismatched_rows={int((~np.isclose(actual[name], reference, atol=ATOL, rtol=RTOL)).sum())}")
    if failures:
        raise RuntimeError("Stored prediction mismatch; no cache written. " + "; ".join(failures))
    return differences


def validate_output_paths(out: Path, manifest: Path) -> None:
    allowed = (ROOT / "data" / "processed" / "models").resolve()
    for path in (out, manifest):
        resolved = path.resolve()
        if not resolved.is_relative_to(allowed):
            raise RuntimeError("All cache outputs must be under data/processed/models")
        result = subprocess.run(["git", "check-ignore", "-q", "--", str(resolved)],
                                cwd=ROOT, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("Output must be git-ignored; git check-ignore failed")
    if not manifest.resolve().is_relative_to(out.resolve()):
        raise RuntimeError("Manifest must be inside the cache output directory")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=(
        "Export selected multi_horizon.120.regression_folds using StratifiedGroupKFold, "
        "NOT primary balanced folds. No tuning or evaluation metrics are generated."))
    parser.add_argument("--data-dir", type=Path, help="Extracted CGMacros directory containing bio.csv and participant folders")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, help="Manifest path (default: OUT_DIR/seed42_manifest.json)")
    parser.add_argument("--check-only", action="store_true", help="Validate key alignment without fitting or writing caches")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    manifest_path = (args.manifest or out / "seed42_manifest.json").resolve()
    validate_output_paths(out, manifest_path)
    # Import only after argparse: the existing pipeline discovers local data and
    # creates directories at import time. --help must never trigger that work.
    global evaluation
    import revised_evaluation as evaluation
    if args.data_dir is not None:
        import train_cgmacros_forecast as training
        evaluation.EXTRACT_DIR = args.data_dir.resolve()
        training.EXTRACT_DIR = evaluation.EXTRACT_DIR
        training.BIO_PATH = evaluation.EXTRACT_DIR / "bio.csv"
    stored = require_stored_predictions()
    settings = selected_fold_settings()

    # These are the existing pipeline functions; no search or alternate feature
    # construction is permitted here.
    all_horizons = evaluation.make_all_horizon_data()
    data = all_horizons[HORIZON].reset_index(drop=True)
    feature_columns, _, _ = evaluation.classify_features(data)
    stored = verify_row_identity(data, stored)
    if args.check_only:
        print(f"Key alignment verified: rebuilt_rows={len(data)}, stored_rows={len(stored)}, unique_keys={len(data)}")
        print("Selected settings: multi_horizon.120.regression_folds; StratifiedGroupKFold; NOT primary balanced folds")
        return

    X = data[feature_columns].astype(float).reset_index(drop=True)
    y = data["label"].to_numpy(int)
    target = data["target_glucose"].to_numpy(float)
    groups = data["participant_id"].to_numpy()
    actual_classifier = np.full(len(data), np.nan)
    actual_regression = np.full(len(data), np.nan)
    models: list[dict] = []
    participant_fold: dict[str, int] = {}
    fold_params = []

    splitter = StratifiedGroupKFold(
        n_splits=evaluation.OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
        fold_setting = settings[fold - 1]
        classifier_params = fold_setting["classifier_params"]
        regression_params = fold_setting["regression_params"]
        classifier = evaluation.make_clf(classifier_params).fit(X.iloc[train], y[train])
        regressor = evaluation.make_reg(regression_params).fit(X.iloc[train], target[train])
        actual_classifier[test] = classifier.predict_proba(X.iloc[test])[:, 1]
        actual_regression[test] = regressor.predict(X.iloc[test])
        for participant in np.unique(groups[test]):
            participant_fold[str(participant)] = fold
        fold_params.append({
            "fold": fold,
            "classifier_params": classifier_params,
            "regressor_params": regression_params,
        })
        models.append({
            "seed": SEED,
            "fold": fold,
            "feature_columns": feature_columns,
            "classifier": classifier,
            "regressor": regressor,
        })

    if np.isnan(actual_classifier).any() or np.isnan(actual_regression).any():
        raise RuntimeError("Seed-42 outer split did not produce exactly one prediction per sample; no cache written")
    max_differences = compare_predictions(
        {"classifier_score": actual_classifier, "regression_prediction": actual_regression}, stored
    )

    out.mkdir(parents=True, exist_ok=True)
    cache_filenames = []
    for model in models:
        filename = f"seed42_fold_{model['fold']}.joblib"
        joblib.dump(model, out / filename)
        cache_filenames.append(filename)

    config = {
        "seed": SEED,
        "horizon_minutes": HORIZON,
        "outer_folds": evaluation.OUTER_FOLDS,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold(shuffle=True, random_state=42)",
        "fold_params": fold_params,
        "feature_columns": feature_columns,
    }
    source_paths = [AGGREGATE, ROOT / "revised_evaluation.py", ROOT / "project_config.py", ROOT / "train_cgmacros_forecast.py"]
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "participant_fold": dict(sorted(participant_fold.items())),
        "per_fold_selected_params": fold_params,
        "package_versions": {"python": platform.python_version(), **package_versions()},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "feature_columns": feature_columns,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "cache_filenames": cache_filenames,
        "cache_hashes": {filename: sha256(out / filename) for filename in cache_filenames},
        "selected_settings_source": "multi_horizon.120.regression_folds",
        "splitter": config["splitter"],
        "comparison": {
            "stored_predictions": str(STORED_PREDICTIONS.relative_to(ROOT)),
            "atol": ATOL,
            "rtol": RTOL,
            "max_abs_differences": max_differences,
        },
        "runtime_note": "Exact export runtime was not recorded. Existing revised evaluation documentation reports approximately 57 minutes for the complete evaluation.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(models)} verified seed-42 fold caches and {manifest_path}")
    print("Command: python export_demo_cache.py")
    print("Exact export runtime was not recorded; existing full evaluation documentation reports approximately 57 minutes total.")


if __name__ == "__main__":
    main()
