"""Inference-only research prototype for local retrospective replay.

No training, tuning, metric generation, or OOF-score fallback is performed.
Only the revised_evaluation.richer_features + complete baseline EHR schema is
supported, for a pre-existing fitted classifier/regressor pair at 30/60/120m.

Cache contract (export_demo_cache.py): --manifest is JSON in
``data/processed/models/seed42_manifest.json`` and each participant is routed to
``data/processed/models/seed42_fold_<fold>.joblib``.  A fold cache has exactly
the keys ``seed``, ``fold``, ``feature_columns``, ``classifier``, and
``regressor``.  The manifest contains the participant-to-fold map and exporter
provenance.  Loading is inference-only; this module never fits a model.

The older single-cache contract is still accepted when explicitly supplied:
  schema_version: 1
  cache_sha256: SHA-256 of the serialized cache
  serialization: "pickle" or "joblib"
  feature_pipeline: "revised_evaluation.richer_features+baseline_ehr"
  feature_columns: ordered list of ALL dynamic then baseline EHR feature names
  horizon_minutes: 30, 60, or 120
  glucose_threshold_mg_dl: 180.0
  sample_minutes: 15
  ffill_tolerance_minutes: 30
  source_sha256: mapping for train_cgmacros_forecast.py, revised_evaluation.py,
                 project_config.py, each to its exact source SHA-256
  versions: exact python, numpy, pandas, scikit-learn, joblib versions
  provenance: nonempty strings for run_id, classifier_id, regressor_id,
              training_data_reference, split_description
Both objects must expose matching feature_names_in_ and n_features_in_; the
classifier classes_ must be {0,1}. A manifest is a producer assertion, not proof
of held-out status or authenticity. Inspect its provenance before trusting it.
--trust-cache explicitly permits arbitrary code execution inherent in loading
pickle/joblib. Hash/version checks do not make untrusted serialization safe.
The default cache is produced by ``export_demo_cache.py``.  Never train to
create a cache during inference.

The original feature modules discover raw files/create directories on import.
To support explicit local input paths without those side effects, this adapter
loads only their named, unmodified function definitions into isolated modules.
It does not execute their training functions or top-level discovery code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import pickle
import platform
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

from project_config import FFILL_TOLERANCE_MINUTES, GLUCOSE_THRESHOLD, HORIZON_MINUTES, SAMPLE_MINUTES

ROOT = Path(__file__).resolve().parent
BANNER = "Research prototype — retrospective replay only; not for clinical use."
NEEDS_CACHE = ("A verified fitted model cache and matching manifest are required. "
               "Training is disabled; saved held-out scores are not fresh inference.")
SOURCES = ("train_cgmacros_forecast.py", "revised_evaluation.py", "project_config.py")


class DemoError(ValueError):
    """A user-facing failure with no patient contents in the message."""


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "joblib")}}


def _definitions(filename, names, namespace):
    """Import exact existing function bodies, without module-level I/O or fitting."""
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {node.name for node in nodes} != set(names):
        raise DemoError("Existing feature functions are unavailable or changed.")
    module = ModuleType("_predict_demo_" + path.stem)
    module.__dict__.update(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def load_feature_functions(bio_path):
    base = _definitions("train_cgmacros_forecast.py", {
        "optional_numeric", "read_participant_file", "make_features",
        "bio_id_to_folder_id", "load_ehr_features"}, {
        "np": np, "pd": pd, "Path": Path, "BIO_PATH": Path(bio_path),
        "FFILL_TOLERANCE_MINUTES": FFILL_TOLERANCE_MINUTES,
        "HORIZON_MINUTES": HORIZON_MINUTES})
    revised = _definitions("revised_evaluation.py", {"richer_features"}, {
        "np": np, "pd": pd, "SAMPLE_MINUTES": SAMPLE_MINUTES,
        "make_features": base.make_features})
    return SimpleNamespace(read_participant_file=base.read_participant_file,
                           load_ehr_features=base.load_ehr_features,
                           richer_features=revised.richer_features)


def participant_id(value):
    if not re.fullmatch(r"(?:CGMacros-)?[0-9]{1,3}", value):
        raise DemoError("Participant must be a numeric ID (1–3 digits) or CGMacros-NNN.")
    return value.removeprefix("CGMacros-").zfill(3)


def parse_timestamp(value):
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        raise DemoError("Timestamp must be a valid local ISO date and time.") from None
    if pd.isna(stamp) or stamp.tzinfo is not None:
        raise DemoError("Use a timezone-naive local timestamp matching the dataset clock.")
    if stamp != stamp.floor(f"{SAMPLE_MINUTES}min"):
        raise DemoError(f"Timestamp must be on the existing {SAMPLE_MINUTES}-minute wall-clock grid.")
    return stamp


def discover_data_dir():
    raw = ROOT / "data" / "raw" / "cgmacros"
    candidates = sorted(raw.glob("*/extracted/CGMacros"))
    if len(candidates) != 1:
        raise DemoError("Local dataset is missing or ambiguous; provide --data-dir pointing to "
                        "the extracted CGMacros directory containing bio.csv. No download is performed.")
    return candidates[0]


def build_feature_row(data_dir, participant, timestamp):
    directory = Path(data_dir)
    path = directory / f"CGMacros-{participant}" / f"CGMacros-{participant}.csv"
    bio = directory / "bio.csv"
    if not path.is_file() or not bio.is_file():
        raise DemoError("Missing local participant CSV or bio.csv. Provide the extracted "
                        "CGMacros directory via --data-dir; no existing command creates these raw inputs.")
    functions = load_feature_functions(bio)
    try:
        raw = functions.read_participant_file(path)
        if raw.index.tz is not None:
            raise DemoError("Input timestamps must use the timezone-naive dataset clock.")
        if raw.empty or timestamp < raw.index.min() or timestamp > raw.index.max():
            raise DemoError("Requested timestamp is outside the available participant recording.")
        # Cut off future rows BEFORE computing any feature. Preserve the original
        # full-history grid origin and rolling/missingness semantics exactly.
        raw = raw.loc[raw.index <= timestamp]
        grid = pd.date_range(raw.index.min().ceil(f"{SAMPLE_MINUTES}min"), timestamp,
                             freq=f"{SAMPLE_MINUTES}min")
        dynamic = functions.richer_features(raw, grid).iloc[[-1]].copy()
        ehr = functions.load_ehr_features()
        if ehr.participant_id.duplicated().any():
            raise DemoError("Baseline EHR participant keys are not unique.")
        static = ehr.loc[ehr.participant_id.eq(participant)].drop(columns="participant_id")
        if len(static) != 1 or static.isna().all(axis=None):
            raise DemoError("Participant has no usable baseline EHR record.")
        for column in static:
            dynamic[column] = static.iloc[0][column]
        return dynamic.astype(float)
    except DemoError:
        raise
    except Exception:
        raise DemoError("Could not ingest local inputs: check timestamp, sensor and baseline EHR schemas.") from None


def _validate_feature_objects(objects, columns, fold=None):
    if set(objects) != {"seed", "fold", "feature_columns", "classifier", "regressor"}:
        raise DemoError("Fold cache must contain exactly seed, fold, feature_columns, classifier and regressor.")
    if objects["seed"] != 42 or (fold is not None and objects["fold"] != fold):
        raise DemoError("Fold cache seed/fold does not match the manifest.")
    if objects["feature_columns"] != columns:
        raise DemoError("Fold cache feature_columns do not match the manifest.")
    for role, method in (("classifier", "predict_proba"), ("regressor", "predict")):
        obj = objects[role]
        if (not callable(getattr(obj, method, None)) or
                list(getattr(obj, "feature_names_in_", [])) != columns or
                getattr(obj, "n_features_in_", None) != len(columns)):
            raise DemoError("Fitted object feature schema does not match the cache manifest.")
    if set(objects["classifier"].classes_) != {0, 1} or len(objects["classifier"].classes_) != 2:
        raise DemoError("Classifier must have binary classes 0 and 1 for the recorded threshold.")


def _runtime_manifest_checks(manifest, manifest_path):
    required = ("schema_version", "seed", "participant_fold", "per_fold_selected_params",
                "package_versions", "created_at", "config_hash", "feature_columns",
                "source_hashes", "cache_filenames", "comparison")
    if any(key not in manifest for key in required):
        raise DemoError("Manifest is missing exporter schema/provenance fields.")
    if manifest["schema_version"] != 1 or manifest["seed"] != 42:
        raise DemoError("Manifest schema or seed does not match this runtime.")
    columns = manifest["feature_columns"]
    if (not isinstance(columns, list) or not columns or
            any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns)):
        raise DemoError("Cache requires unique ordered feature_columns.")
    if not isinstance(manifest["participant_fold"], dict) or not manifest["cache_filenames"]:
        raise DemoError("Manifest requires participant_fold and cache_filenames.")
    if not isinstance(manifest["package_versions"], dict):
        raise DemoError("Manifest package_versions are invalid.")
    expected_versions = {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}}
    if manifest["package_versions"] != expected_versions:
        raise DemoError("Cache package versions do not match this runtime.")
    if not isinstance(manifest["source_hashes"], dict) or not isinstance(manifest["config_hash"], str) or not manifest["config_hash"]:
        raise DemoError("Manifest source/config provenance is invalid.")
    # Validate hashes for sources present in this checkout.  Synthetic fixtures
    # may provide only exporter metadata, so absent paths are not invented.
    for relative, digest in manifest["source_hashes"].items():
        path = ROOT / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise DemoError("Cache source provenance does not match this runtime.")


def load_exported_cache(manifest_path, cache_dir=None, participant=None, trusted=False):
    """Load and validate the exporter manifest and the participant's fold cache."""
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else manifest_path.parent
    if not manifest_path.is_file():
        raise DemoError("Fitted cache manifest is missing. " + NEEDS_CACHE)
    if not trusted:
        raise DemoError("Loading joblib requires --trust-cache after verifying the cache's origin; "
                        "deserialization can execute arbitrary code.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _runtime_manifest_checks(manifest, manifest_path)
        if participant is None:
            raise DemoError("A participant is required to select its held-out fold cache.")
        participant = str(participant)
        if participant not in manifest["participant_fold"]:
            raise DemoError("Participant is not present in manifest participant_fold.")
        fold = manifest["participant_fold"][participant]
        if not isinstance(fold, int) or fold < 1:
            raise DemoError("Manifest participant_fold contains an invalid fold.")
        filenames = manifest["cache_filenames"]
        filename = (filenames[fold - 1] if isinstance(filenames, list) and len(filenames) >= fold
                    else filenames.get(str(fold), filenames.get(fold)) if isinstance(filenames, dict) else None)
        if not filename:
            raise DemoError("Manifest has no cache filename for the participant's fold.")
        cache_path = cache_dir / filename
        if not cache_path.is_file():
            raise DemoError("Participant fold cache is missing.")
        # Newer exporters may include hashes keyed by filename (or by fold).
        # The current exporter does not, so validation remains conditional.
        hashes = manifest.get("cache_hashes", manifest.get("cache_sha256"))
        expected_hash = None
        if isinstance(hashes, dict):
            expected_hash = hashes.get(filename, hashes.get(str(fold), hashes.get(fold)))
        if expected_hash and hashlib.sha256(cache_path.read_bytes()).hexdigest() != expected_hash:
            raise DemoError("Fold cache SHA-256 does not match the manifest.")
        import joblib
        objects = joblib.load(cache_path)
        _validate_feature_objects(objects, manifest["feature_columns"], fold)
        return objects, manifest, fold
    except DemoError:
        raise
    except Exception:
        raise DemoError("Could not load fitted fold cache/manifest; check its format and dependencies.") from None


def load_cache(cache_path, manifest_path, trusted=False):
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    if not cache_path.is_file() or not manifest_path.is_file():
        raise DemoError("Fitted cache or manifest is missing. " + NEEDS_CACHE)
    if not trusted:
        raise DemoError("Loading pickle/joblib requires --trust-cache after verifying the cache's "
                        "origin; deserialization can execute arbitrary code.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"schema_version": 1,
                    "feature_pipeline": "revised_evaluation.richer_features+baseline_ehr",
                    "glucose_threshold_mg_dl": GLUCOSE_THRESHOLD,
                    "sample_minutes": SAMPLE_MINUTES,
                    "ffill_tolerance_minutes": FFILL_TOLERANCE_MINUTES,
                    "source_sha256": source_hashes(), "versions": runtime_versions()}
        if any(manifest.get(k) != v for k, v in required.items()):
            raise DemoError("Cache source/configuration/version provenance does not match this runtime.")
        if manifest.get("horizon_minutes") not in (30, 60, 120):
            raise DemoError("Cache horizon must be an existing 30, 60 or 120-minute task.")
        provenance = manifest.get("provenance", {})
        if any(not isinstance(provenance.get(k), str) or not provenance[k].strip()
               for k in ("run_id", "classifier_id", "regressor_id", "training_data_reference", "split_description")):
            raise DemoError("Cache requires explicit classifier/regressor training and split provenance.")
        columns = manifest.get("feature_columns")
        if (not isinstance(columns, list) or not columns or
                any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns)):
            raise DemoError("Cache requires unique ordered feature_columns.")
        payload = cache_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.get("cache_sha256"):
            raise DemoError("Cache SHA-256 does not match the manifest.")
        if manifest.get("serialization") == "pickle":
            objects = pickle.loads(payload)
        elif manifest.get("serialization") == "joblib":
            import joblib
            objects = joblib.load(io.BytesIO(payload))
        else:
            raise DemoError("Cache serialization must be pickle or joblib.")
        # Legacy single-cache files contain only classifier/regressor.
        for role, method in (("classifier", "predict_proba"), ("regressor", "predict")):
            obj = objects[role]
            if (not callable(getattr(obj, method, None)) or list(getattr(obj, "feature_names_in_", [])) != columns or
                    getattr(obj, "n_features_in_", None) != len(columns)):
                raise DemoError("Fitted object feature schema does not match the cache manifest.")
        if set(objects["classifier"].classes_) != {0, 1} or len(objects["classifier"].classes_) != 2:
            raise DemoError("Classifier must have binary classes 0 and 1 for the recorded threshold.")
        return objects, manifest
    except DemoError:
        raise
    except Exception:
        raise DemoError("Could not load fitted cache/manifest; check its format, fitted state and dependencies. "
                        + NEEDS_CACHE) from None


def predict(objects, manifest, row, participant, timestamp, fold=None):
    if list(row.columns) != manifest["feature_columns"]:
        raise DemoError("Ingested feature schema/order differs from the fitted cache; no features were altered.")
    try:
        probabilities = np.asarray(objects["classifier"].predict_proba(row), dtype=float)
        estimate = np.asarray(objects["regressor"].predict(row), dtype=float)
        if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or (probabilities > 1).any() or
                not np.isclose(probabilities.sum(), 1) or estimate.shape != (1,) or
                not np.isfinite(estimate).all()):
            raise DemoError("Fitted objects returned invalid probability/regression outputs.")
        probability = float(probabilities[0, list(objects["classifier"].classes_).index(1)])
    except DemoError:
        raise
    except Exception:
        raise DemoError("Existing fitted inference failed; check fitted state and preprocessing compatibility. "
                        "No training was attempted.") from None
    def value(column):
        v = row.iloc[0][column]
        return None if pd.isna(v) else float(v)
    slope = value("glucose_slope_60m_per_min")
    # The original pipeline fills unavailable slopes with zero. Do not describe
    # that fallback as an observed stable trend.
    trend = "unavailable"
    if value("glucose_now") is not None and value("glucose_lag_60m") is not None:
        trend = "rising" if slope > 0 else "falling" if slope < 0 else "stable"
    selected_fold = fold if fold is not None else manifest.get("fold")
    note = (f"Fold {selected_fold} selected for participant {participant}; "
            "retrospective held-out replay only.")
    return {"notice": BANNER, "note": note,
            "mode": "existing_fitted_inference_on_retrospective_inputs",
            "participant_id": participant, "timestamp": timestamp.isoformat(),
            "horizon_minutes": manifest.get("horizon_minutes", 120),
            "glucose_threshold_mg_dl": GLUCOSE_THRESHOLD,
            "probability_glucose_at_or_above_threshold": probability,
            "regression_glucose_mg_dl": float(estimate[0]),
            "provenance": manifest.get("provenance", {
                "seed": manifest.get("seed"), "fold": selected_fold,
                "source_hashes": manifest.get("source_hashes", {}),
                "config_hash": manifest.get("config_hash")}),
            "validation_note": "Cache provenance is producer-declared; this inference is not evidence of held-out performance.",
            "context": {"trend": trend, **{c: value(c) for c in (
                "glucose_now", "glucose_slope_60m_per_min", "carbs_last_120m",
                "meal_events_last_120m", "minutes_since_meal", "carbohydrate_on_board_g",
                "heart_rate", "activity_level", "activity_calories")}},
            "missing_feature_count": int(row.isna().sum().sum())}


def format_result(result):
    return json.dumps(result, indent=2, allow_nan=False)


class DemoParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse can echo bad values, so use a generic input error instead.
        self.exit(2, BANNER + "\nInvalid CLI arguments; use --help for the supported syntax.\n")


def main(argv=None):
    parser = DemoParser(description=BANNER, epilog=NEEDS_CACHE)
    parser.add_argument("--participant", required=True, help="Local numeric participant ID")
    parser.add_argument("--timestamp", required=True, help="Local ISO timestamp on a 15-minute grid")
    parser.add_argument("--data-dir", type=Path, help="Extracted CGMacros folder containing bio.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/models/seed42_manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/models",
                        help="Directory containing seed42_fold_<fold>.joblib files")
    # Kept for callers of the original demo; an explicitly supplied --cache
    # uses the legacy single-cache loader below.
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trust-cache", action="store_true", help="Permit trusted pickle/joblib code execution")
    args = parser.parse_args(argv)
    try:
        participant = participant_id(args.participant)
        timestamp = parse_timestamp(args.timestamp)
        if args.cache is not None:
            objects, manifest = load_cache(args.cache, args.manifest, args.trust_cache)
            fold = manifest.get("fold")
        else:
            objects, manifest, fold = load_exported_cache(args.manifest, args.cache_dir,
                                                          participant, args.trust_cache)
        row = build_feature_row(args.data_dir or discover_data_dir(), participant, timestamp)
        print(format_result(predict(objects, manifest, row, participant, timestamp, fold)))
        return 0
    except DemoError as exc:
        print(f"{BANNER}\nError: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print(f"{BANNER}\nError: local inference could not complete; check inputs and dependencies. "
              "No training was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
