"""Synthetic dashboard contract checks; no CGMacros participant data."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pandas as pd
import pytest

import serve_cgmacros_dashboard as dashboard


# The optional multi-horizon source uses its own StratifiedGroupKFold assignment, so its folds
# deliberately differ from the primary source's folds (1, 2) to catch provenance mix-ups.
PRIMARY_FOLDS = {"001": 1, "002": 2}
OPTIONAL_FOLDS = {"001": 3, "002": 4}


def write_artifacts(
    directory: Path, *, missing_target: bool = False, optional: bool = True, optional_fold_column: bool = True
) -> None:
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    sample_rows = []
    prediction_rows = []
    optional_rows = []
    for participant, fold in (("001", 1), ("002", 2)):
        for index, timestamp in enumerate(timestamps):
            target = 190.0 if index == 2 else 120.0
            if missing_target and participant == "001" and index == 2:
                target = None
            sample_rows.append({
                "participant_id": participant,
                "timestamp": timestamp.isoformat(),
                "glucose_now": 100.0 + index,
                "heart_rate": 70.0 if index != 1 else None,
                "activity_level": 1.5,
                "carbs_last_120m": 0.0,
                "meal_events_last_120m": 0.0,
                "target_glucose": target,
                "label": None if target is None else int(target >= 180),
                "horizon_minutes": 120,
                "ehr_age_years": 40.0,
                "ehr_bmi": 25.0,
            })
            prediction_rows.append({
                "participant_id": participant,
                "timestamp": timestamp.isoformat(),
                "score": 0.1 + index / 10 + (0.05 if participant == "002" else 0),
                "outer_fold": fold,
            })
            optional_rows.append({
                "participant_id": participant,
                "timestamp": timestamp.isoformat(),
                "score": 0.2 + index / 10,
                "regression": 150.0 + index,
                **({"outer_fold": OPTIONAL_FOLDS[participant]} if optional_fold_column else {}),
            })
    pd.DataFrame(sample_rows).to_csv(directory / "rich_120m_samples_local.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(directory / "primary_120m_seed_42_oof_local.csv", index=False)
    if optional:
        pd.DataFrame(optional_rows).to_csv(directory / "horizon_120m_regression_local.csv", index=False)


@pytest.fixture
def artifacts(tmp_path):
    write_artifacts(tmp_path)
    return tmp_path


def test_index_is_metadata_only_and_replay_hides_then_reveals_exact_outcome(artifacts):
    data = dashboard.build_dashboard_data(artifacts)
    index = dashboard.dashboard_index(data)
    assert index["sources"][1]["available"] is True
    assert "rows" not in index["participants"][0]
    assert index["participants"][0]["overview"]["feature_count"] == 7

    timestamp = "2025-01-01T00:00:00"
    hidden = dashboard.replay_snapshot(data, "001", timestamp, "primary", 0)
    assert hidden["outcome"]["status"] == "hidden"
    assert hidden["outcome"]["glucose_mg_dl"] is None
    assert all(pd.Timestamp(row["timestamp"]) <= pd.Timestamp(hidden["replay_clock"]) for row in hidden["timeline"])
    assert hidden["forecast"]["glucose_mg_dl"] is None

    revealed = dashboard.replay_snapshot(data, "001", timestamp, "multi_horizon", 120)
    assert revealed["outcome"]["status"] == "available"
    assert revealed["outcome"]["glucose_mg_dl"] == 120.0
    assert revealed["forecast"]["glucose_mg_dl"] == 150.0
    assert revealed["outcome"]["absolute_error_mg_dl"] == 30.0
    assert revealed["outcome"]["brier_error"] == pytest.approx(0.04)
    assert revealed["forecast"]["source"] == "multi_horizon"


def test_personalization_message_describes_completed_audit_without_claiming_ui_support():
    assert "purged" in dashboard.PERSONALIZATION["reason"]
    assert "remains withheld" in dashboard.PERSONALIZATION["reason"]
    assert "pending a purged-boundary rerun" not in dashboard.PERSONALIZATION["reason"]


def test_worklist_uses_exact_shared_timestamp_and_missingness_preserves_zero(artifacts):
    data = dashboard.build_dashboard_data(artifacts)
    response = dashboard.replay_snapshot(data, "001", "2025-01-01T00:15:00", "primary", 0)
    assert [item["id"] for item in response["worklist"]] == ["002", "001"]
    assert response["context"]["carbs_2h"] == 0.0
    assert response["quality"]["streams"]["heart_rate"] == "missing"
    assert response["quality"]["streams"]["carbs_2h"] == "available"
    assert response["outcome"]["status"] == "hidden"


def test_missing_exact_future_outcome_is_explicitly_unavailable(tmp_path):
    write_artifacts(tmp_path, missing_target=True)
    data = dashboard.build_dashboard_data(tmp_path)
    response = dashboard.replay_snapshot(data, "001", "2025-01-01T00:30:00", "primary", 120)
    assert response["outcome"]["status"] == "unavailable"
    assert response["outcome"]["glucose_mg_dl"] is None
    assert "exact t+120" in response["outcome"]["reason"]


def test_invalid_or_partial_artifacts_fail_without_participant_values(tmp_path):
    write_artifacts(tmp_path)
    predictions = pd.read_csv(tmp_path / "primary_120m_seed_42_oof_local.csv").iloc[:-1]
    predictions.to_csv(tmp_path / "primary_120m_seed_42_oof_local.csv", index=False)
    with pytest.raises(dashboard.DashboardError, match="cover") as error:
        dashboard.build_dashboard_data(tmp_path)
    assert "001" not in str(error.value)


def test_http_routes_serve_index_replay_and_sanitized_errors(artifacts):
    data = dashboard.build_dashboard_data(artifacts)
    dashboard.DashboardHandler.replay_data = data
    dashboard.DashboardHandler.data_json = json.dumps(dashboard.dashboard_index(data)).encode()
    server = dashboard.HTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/replay?participant=001&timestamp=2025-01-01T00:00:00&elapsed=0")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["outcome"]["status"] == "hidden"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        connection.close()

        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/replay?participant=bad&timestamp=private")
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 400
        assert "bad" not in body and "private" not in body
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_optional_source_reports_its_own_fold_not_the_primary_fold(artifacts):
    data = dashboard.build_dashboard_data(artifacts)
    stamp = "2025-01-01T00:15:00"
    for participant in ("001", "002"):
        primary = dashboard.replay_snapshot(data, participant, stamp, "primary", 0)
        optional = dashboard.replay_snapshot(data, participant, stamp, "multi_horizon", 0)
        assert primary["forecast"]["fold"] == PRIMARY_FOLDS[participant]
        assert optional["forecast"]["fold"] == OPTIONAL_FOLDS[participant]
        assert optional["forecast"]["fold"] != primary["forecast"]["fold"]


def test_optional_fold_falls_back_to_source_metadata_when_column_is_absent(tmp_path):
    write_artifacts(tmp_path, optional_fold_column=False)
    metrics = {"multi_horizon": {"120": {"regression_folds": [
        {"fold": 3, "test_participant_ids": ["001"]},
        {"fold": 4, "test_participant_ids": ["CGMacros-002"]},
    ]}}}
    (tmp_path / "revised_full_metrics.json").write_text(json.dumps(metrics))
    data = dashboard.build_dashboard_data(tmp_path)
    assert dashboard.replay_snapshot(data, "001", "2025-01-01T00:00:00", "multi_horizon", 0)["forecast"]["fold"] == 3
    assert dashboard.replay_snapshot(data, "002", "2025-01-01T00:00:00", "multi_horizon", 0)["forecast"]["fold"] == 4
    assert dashboard.replay_snapshot(data, "002", "2025-01-01T00:00:00", "primary", 0)["forecast"]["fold"] == 2


def test_optional_source_fails_closed_without_verifiable_fold(tmp_path):
    write_artifacts(tmp_path, optional_fold_column=False)
    data = dashboard.build_dashboard_data(tmp_path)
    assert data["optional"] is None
    assert "fold metadata" in data["optional_reason"]
    assert dashboard.dashboard_index(data)["sources"][1]["available"] is False
    with pytest.raises(dashboard.DashboardError):
        dashboard.replay_snapshot(data, "001", "2025-01-01T00:00:00", "multi_horizon", 0)


def test_optional_source_rejects_inconsistent_participant_folds(tmp_path):
    write_artifacts(tmp_path)
    path = tmp_path / "horizon_120m_regression_local.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[0], "outer_fold"] = 5  # participant 001 now appears in two folds
    frame.to_csv(path, index=False)
    data = dashboard.build_dashboard_data(tmp_path)
    assert data["optional"] is None
    assert "inconsistent" in data["optional_reason"]
