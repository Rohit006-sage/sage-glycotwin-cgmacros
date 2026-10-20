"""Synthetic inference-only checks; no model fitting or real participant data."""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict_demo as demo


class FakeClassifier:
    def __init__(self, columns):
        self.feature_names_in_ = np.array(columns)
        self.n_features_in_ = len(columns)
        self.classes_ = np.array([1, 0])

    def predict_proba(self, row):
        return np.array([[0.7, 0.3]])


class FakeRegressor:
    def __init__(self, columns):
        self.feature_names_in_ = np.array(columns)
        self.n_features_in_ = len(columns)

    def predict(self, row):
        return np.array([155.0])


@pytest.fixture
def local_inputs(tmp_path):
    directory = tmp_path / "CGMacros-999"
    directory.mkdir()
    path = directory / "CGMacros-999.csv"
    pd.DataFrame({"Timestamp": pd.date_range("2000-01-01", periods=14, freq="15min"),
                  "Libre GL": np.arange(14) + 100, "HR": 70, "METs": 1.5,
                  "Carbs": [20] + [0] * 13, "Meal Type": ["meal"] + [""] * 13}).to_csv(path, index=False)
    pd.DataFrame({"subject": [999], "Gender": ["F"], **{name: [10] for name in (
        "Age", "BMI", "A1c PDL (Lab)", "Fasting GLU - PDL (Lab)", "Insulin ",
        "Triglycerides", "Cholesterol", "HDL", "LDL (Cal)", "VLDL (Cal)", "Cho/HDL Ratio")}}).to_csv(tmp_path / "bio.csv", index=False)
    return tmp_path


@pytest.fixture
def row(local_inputs):
    return demo.build_feature_row(local_inputs, "999", demo.parse_timestamp("2000-01-01T02:00"))


def cache_files(tmp_path, row, serialization="pickle"):
    objects = {"classifier": FakeClassifier(list(row)), "regressor": FakeRegressor(list(row))}
    path = tmp_path / "cache.bin"
    if serialization == "pickle":
        path.write_bytes(pickle.dumps(objects))
    else:
        import joblib
        joblib.dump(objects, path)
    manifest = {"schema_version": 1, "serialization": serialization,
                "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "feature_pipeline": "revised_evaluation.richer_features+baseline_ehr",
                "feature_columns": list(row), "horizon_minutes": 120,
                "glucose_threshold_mg_dl": 180.0, "sample_minutes": 15,
                "ffill_tolerance_minutes": 30, "source_sha256": demo.source_hashes(),
                "versions": demo.runtime_versions(), "provenance": {
                    key: "synthetic-fixture" for key in ("run_id", "classifier_id", "regressor_id",
                    "training_data_reference", "split_description")}}
    meta = tmp_path / "manifest.json"
    meta.write_text(json.dumps(manifest), encoding="utf-8")
    return path, meta


def exported_cache_files(tmp_path, row):
    """Synthetic exporter-shaped fixture; it never reads a CGMacros dataset."""
    import joblib

    models = []
    for fold in (1, 2):
        path = tmp_path / f"seed42_fold_{fold}.joblib"
        joblib.dump({"seed": 42, "fold": fold, "feature_columns": list(row),
                     "classifier": FakeClassifier(list(row)),
                     "regressor": FakeRegressor(list(row))}, path)
        models.append(path.name)
    manifest = {"schema_version": 1, "seed": 42, "participant_fold": {"999": 2},
                "per_fold_selected_params": [{"fold": 1}, {"fold": 2}],
                "package_versions": demo.runtime_versions(), "created_at": "synthetic",
                "config_hash": "synthetic-config", "feature_columns": list(row),
                "source_hashes": {}, "cache_filenames": models, "comparison": {}}
    meta = tmp_path / "seed42_manifest.json"
    meta.write_text(json.dumps(manifest), encoding="utf-8")
    return meta, tmp_path


def test_original_features_static_and_future_cutoff(local_inputs, row):
    assert row.iloc[0].glucose_now == 108
    assert row.iloc[0].glucose_lag_120m == 100
    assert row.iloc[0].ehr_age_years == 10
    assert row.iloc[0].ehr_gender_female == 1
    path = local_inputs / "CGMacros-999/CGMacros-999.csv"
    raw = pd.read_csv(path)
    raw.loc[9:, ["Libre GL", "Carbs"]] = 9999
    raw.to_csv(path, index=False)
    pd.testing.assert_frame_equal(row, demo.build_feature_row(local_inputs, "999", row.index[0]))


@pytest.mark.parametrize("serialization", ["pickle", "joblib"])
def test_cache_inference_and_json(tmp_path, row, serialization):
    paths = cache_files(tmp_path, row, serialization)
    objects, manifest = demo.load_cache(*paths, trusted=True)
    result = demo.predict(objects, manifest, row, "999", row.index[0])
    result = json.loads(demo.format_result(result))
    assert result["probability_glucose_at_or_above_threshold"] == 0.7
    assert result["regression_glucose_mg_dl"] == 155
    assert result["context"]["trend"] == "rising"
    assert result["context"]["activity_calories"] is None
    assert result["notice"] == demo.BANNER


def test_trust_required_before_deserialization(tmp_path, row, monkeypatch):
    paths = cache_files(tmp_path, row)
    monkeypatch.setattr(pickle, "loads", lambda *_: pytest.fail("must not deserialize"))
    with pytest.raises(demo.DemoError, match="trust-cache"):
        demo.load_cache(*paths)


@pytest.mark.parametrize("key,value", [("cache_sha256", "bad"), ("source_sha256", {}),
    ("versions", {}), ("provenance", {}), ("feature_columns", ["wrong"]),
    ("horizon_minutes", 90), ("serialization", "unknown")])
def test_invalid_cache_contract(tmp_path, row, key, value):
    path, meta = cache_files(tmp_path, row)
    manifest = json.loads(meta.read_text())
    manifest[key] = value
    meta.write_text(json.dumps(manifest))
    with pytest.raises(demo.DemoError):
        demo.load_cache(path, meta, trusted=True)


def test_schema_order_rejected(tmp_path, row):
    objects, manifest = demo.load_cache(*cache_files(tmp_path, row), trusted=True)
    with pytest.raises(demo.DemoError, match="schema/order"):
        demo.predict(objects, manifest, row[row.columns[::-1]], "999", row.index[0])


def test_bad_outputs_and_unfitted_errors(tmp_path, row, monkeypatch):
    objects, manifest = demo.load_cache(*cache_files(tmp_path, row), trusted=True)
    monkeypatch.setattr(objects["classifier"], "predict_proba", lambda _: [[2, -1]])
    with pytest.raises(demo.DemoError, match="invalid probability"):
        demo.predict(objects, manifest, row, "999", row.index[0])
    def fail(_):
        raise RuntimeError("synthetic private contents must not escape")
    monkeypatch.setattr(objects["classifier"], "predict_proba", fail)
    with pytest.raises(demo.DemoError, match="Existing fitted inference failed") as exc:
        demo.predict(objects, manifest, row, "999", row.index[0])
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("stamp", ["bad", "NaT", "2000-01-01T00:01", "2000-01-01T00:00Z"])
def test_timestamp_errors(stamp):
    with pytest.raises(demo.DemoError):
        demo.parse_timestamp(stamp)


def test_missing_and_malformed_inputs(tmp_path, local_inputs):
    with pytest.raises(demo.DemoError, match="Missing local"):
        demo.build_feature_row(tmp_path, "998", pd.Timestamp("2000-01-01"))
    (local_inputs / "bio.csv").write_text("unexpected\n1\n")
    with pytest.raises(demo.DemoError, match="schemas"):
        demo.build_feature_row(local_inputs, "999", pd.Timestamp("2000-01-01T02:00"))


def test_cli_missing_cache_and_banner(tmp_path, capsys):
    assert demo.main(["--participant", "999", "--timestamp", "2000-01-01T02:00",
                      "--cache", str(tmp_path / "absent"), "--manifest", str(tmp_path / "absent.json")]) == 2
    error = capsys.readouterr().err
    assert demo.BANNER in error
    assert "Training is disabled" in error
    assert "verified fitted model cache" in error


def test_cli_success(local_inputs, row, capsys):
    path, meta = cache_files(local_inputs, row)
    assert demo.main(["--participant", "999", "--timestamp", "2000-01-01T02:00",
                      "--data-dir", str(local_inputs), "--cache", str(path),
                      "--manifest", str(meta), "--trust-cache"]) == 0
    assert json.loads(capsys.readouterr().out)["regression_glucose_mg_dl"] == 155


def test_exported_manifest_routes_participant_to_fold(tmp_path, row, monkeypatch, capsys):
    manifest, cache_dir = exported_cache_files(tmp_path, row)
    monkeypatch.setattr(demo, "build_feature_row", lambda *_: row)
    assert demo.main(["--participant", "999", "--timestamp", "2000-01-01T02:00",
                      "--manifest", str(manifest), "--cache-dir", str(cache_dir),
                      "--trust-cache"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "Fold 2" in result["note"]
    assert "retrospective held-out replay" in result["note"]


def test_exported_manifest_missing_participant_and_fold(tmp_path, row):
    manifest, cache_dir = exported_cache_files(tmp_path, row)
    with pytest.raises(demo.DemoError, match="not present"):
        demo.load_exported_cache(manifest, cache_dir, "998", trusted=True)
    payload = json.loads(manifest.read_text())
    payload["participant_fold"] = {"999": 3}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(demo.DemoError, match="filename"):
        demo.load_exported_cache(manifest, cache_dir, "999", trusted=True)


def test_help_and_argument_errors_have_banner(capsys):
    with pytest.raises(SystemExit) as exc:
        demo.main(["--help"])
    assert exc.value.code == 0 and demo.BANNER in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        demo.main([])
    assert exc.value.code == 2 and demo.BANNER in capsys.readouterr().err


def test_feature_loading_has_no_raw_discovery_or_training(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("feature import must not discover raw data, create directories or train")
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    import sklearn.ensemble
    import sklearn.linear_model
    import sklearn.pipeline
    monkeypatch.setattr(sklearn.ensemble.HistGradientBoostingClassifier, "fit", forbidden)
    monkeypatch.setattr(sklearn.ensemble.HistGradientBoostingRegressor, "fit", forbidden)
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", forbidden)
    functions = demo.load_feature_functions(tmp_path / "absent.csv")
    assert callable(functions.richer_features)
    assert "run_grouped_cv" not in functions.richer_features.__globals__
