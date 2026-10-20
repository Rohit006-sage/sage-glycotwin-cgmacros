"""Synthetic alignment and CLI checks; no CGMacros data or model files."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_demo_cache as exporter


def frame():
    return pd.DataFrame({
        "participant_id": ["001", "002", "001"],
        "timestamp": [
            "2025-01-01T00:00:00",
            "2025-01-01T00:15:00",
            "2025-01-01T00:15:00",
        ],
        "score": [0.1, 0.2, 0.3],
        "regression": [100.0, 110.0, 120.0],
    })


def test_key_alignment_accepts_shuffled_rows_and_normalizes_ids():
    actual = frame().drop(columns=["score", "regression"])
    stored = frame().iloc[[2, 0, 1]].copy()
    stored["participant_id"] = stored["participant_id"].astype(int)
    aligned = exporter.verify_row_identity(actual, stored)
    assert aligned["participant_id"].astype(str).tolist() == ["1", "2", "1"]
    assert aligned["timestamp"].tolist() == actual["timestamp"].tolist()


def test_key_alignment_rejects_real_mismatch_without_values_in_error():
    actual = frame().drop(columns=["score", "regression"])
    stored = frame().iloc[[2, 0, 1]].copy()
    stored.loc[1, "timestamp"] = "2025-01-01T00:30:00"
    with pytest.raises(RuntimeError) as exc:
        exporter.verify_row_identity(actual, stored)
    message = str(exc.value)
    assert "key_columns=participant_id,timestamp" in message
    assert "first_mismatch_index=" in message
    assert "001" not in message
    assert "2025-01-01" not in message


def test_help_exits_without_importing_evaluation_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "OUT", tmp_path / "models")
    output = io.StringIO()
    with contextlib.redirect_stdout(output), pytest.raises(SystemExit) as raised:
        exporter.main(["--help"])
    assert raised.value.code == 0
    assert "--data-dir" in output.getvalue()
    assert not (tmp_path / "models").exists()
