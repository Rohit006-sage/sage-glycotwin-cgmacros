from __future__ import annotations

"""Serve the local, inference-free GlycoTwin retrospective replay.

The dashboard consumes already-generated, git-ignored replay artifacts.  It does
not discover raw data, fit models, deserialize model files, or generate metrics.
"""

import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed" / "revised_full"
PAGE = ROOT / "cgmacros_dashboard.html"
SCRIPT = ROOT / "cgmacros_replay.js"
HORIZON_MINUTES = 120
SAMPLE_MINUTES = 15
THRESHOLD_MG_DL = 180.0
NOTICE = "Research prototype — retrospective replay only; not for clinical use."
PERSONALIZATION = {
    "status": "unavailable",
    "reason": (
        "Aggregate personalization was rerun with a label-availability-aware, purged "
        "72-hour boundary. Individual paired personalization replay remains withheld "
        "until that comparison is validated for this interactive UI."
    ),
}
STREAM_COLUMNS = ("glucose", "heart_rate", "activity", "carbs_2h", "meal_events_2h")
SAMPLE_REQUIRED = ("participant_id", "timestamp", "label", "target_glucose")


class DashboardError(ValueError):
    """Expected, sanitized dashboard-data or request failure."""


def clean_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _artifact(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise DashboardError("Required dashboard replay artifact is missing.")
    return path


def _keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"participant_id", "timestamp"}
    if not required.issubset(frame.columns):
        raise DashboardError(f"{name} schema is invalid.")

    ids = frame["participant_id"].astype("string").str.strip()
    extracted = ids.str.extract(r"(?:CGMacros-)?(\d{1,3})$", expand=False)
    if extracted.isna().any():
        raise DashboardError("Dashboard participant identifiers are invalid.")
    frame = frame.copy()
    frame["participant_id"] = extracted.astype(str).str.zfill(3)
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    except Exception:
        raise DashboardError("Dashboard timestamps are invalid.") from None

    timezone = getattr(frame["timestamp"].dt, "tz", None)
    if (
        timezone is not None
        or (frame["timestamp"].dt.minute % SAMPLE_MINUTES != 0).any()
        or frame["timestamp"].dt.second.ne(0).any()
        or frame["timestamp"].dt.microsecond.ne(0).any()
    ):
        raise DashboardError("Dashboard timestamps must be timezone-naive 15-minute wall-clock values.")
    if frame.duplicated(["participant_id", "timestamp"]).any():
        raise DashboardError("Dashboard replay keys are not unique.")
    return frame


def _finite_score(frame: pd.DataFrame, column: str, message: str) -> None:
    if column not in frame or frame[column].isna().any():
        raise DashboardError(message)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or (~numeric.between(0, 1)).any():
        raise DashboardError(message)
    frame[column] = numeric


def _validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if not set(SAMPLE_REQUIRED).issubset(samples.columns):
        raise DashboardError("Samples schema is invalid.")
    labels = pd.to_numeric(samples["label"], errors="coerce")
    labeled = labels.notna()
    if labeled.any() and not labels.loc[labeled].isin([0, 1]).all():
        raise DashboardError("Sample labels are invalid.")
    samples = samples.copy()
    samples["label"] = labels
    targets = pd.to_numeric(samples["target_glucose"], errors="coerce")
    if targets.notna().any():
        if samples.loc[targets.notna() & labels.isna(), "label"].any():
            raise DashboardError("Labeled replay rows require an exact future glucose value.")
        expected = (targets >= THRESHOLD_MG_DL).astype("Int64")
        comparable = targets.notna() & labels.notna()
        if not (labels.loc[comparable].astype(int) == expected.loc[comparable].astype(int)).all():
            raise DashboardError("Replay labels do not match the recorded 180 mg/dL target definition.")
    if "horizon_minutes" in samples:
        horizons = pd.to_numeric(samples["horizon_minutes"], errors="coerce")
        if horizons.isna().any() or not horizons.eq(HORIZON_MINUTES).all():
            raise DashboardError("Replay samples are not for the 120-minute horizon.")
    return samples


def _validate_primary(samples: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = _keys(predictions, "Primary predictions")
    if not {"score", "outer_fold"}.issubset(predictions.columns):
        raise DashboardError("Primary prediction schema is invalid.")
    _finite_score(predictions, "score", "Primary scores are invalid.")
    if "seed" in predictions:
        seeds = pd.to_numeric(predictions["seed"], errors="coerce")
        if seeds.isna().any() or not seeds.eq(42).all():
            raise DashboardError("Primary prediction provenance is not seed 42.")
    folds = pd.to_numeric(predictions["outer_fold"], errors="coerce")
    if folds.isna().any() or (folds < 1).any() or (folds % 1 != 0).any():
        raise DashboardError("Primary fold assignments are invalid.")
    predictions = predictions.copy()
    predictions["outer_fold"] = folds.astype(int)
    merged = samples.merge(
        predictions[["participant_id", "timestamp", "score", "outer_fold"]],
        on=["participant_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if merged["score"].isna().any():
        raise DashboardError("Primary predictions do not cover samples.")
    if merged.groupby("participant_id")["outer_fold"].nunique().gt(1).any():
        raise DashboardError("Primary fold assignments are inconsistent.")
    return merged.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)


def _normalize_participant_id(value: object) -> str | None:
    match = re.search(r"(?:CGMacros-)?(\d{1,3})$", str(value).strip())
    return match.group(1).zfill(3) if match else None


def _folds_from_metrics(path: Path) -> dict[str, int] | None:
    """Map participant -> held-out fold using the multi-horizon source's own metadata."""
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fold_records = metrics["multi_horizon"]["120"]["regression_folds"]
        mapping: dict[str, int] = {}
        for record in fold_records:
            fold = int(record["fold"])
            for raw_id in record["test_participant_ids"]:
                participant_id = _normalize_participant_id(raw_id)
                if participant_id is None or mapping.setdefault(participant_id, fold) != fold:
                    return None
        return mapping
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _validate_optional(
    samples: pd.DataFrame, path: Path, metrics_path: Path | None = None
) -> tuple[pd.DataFrame | None, str]:
    """Load the paired 120-minute SGKF demo source, failing closed."""
    if not path.is_file():
        return None, "Optional multi-horizon regression artifact is not present."
    try:
        optional = _keys(pd.read_csv(path), "Multi-horizon predictions")
        if not {"score", "regression"}.issubset(optional.columns):
            return None, "Optional multi-horizon artifact has no paired classifier and regression outputs."
        _finite_score(optional, "score", "Optional multi-horizon scores are invalid.")
        optional["regression"] = pd.to_numeric(optional["regression"], errors="coerce")
        if optional["regression"].isna().any() or not optional["regression"].map(math.isfinite).all():
            return None, "Optional multi-horizon regression outputs are invalid."
        # The fold must come from this source's own split: an explicit column, else the
        # source's recorded fold metadata. Never reuse the primary classifier's fold.
        if "outer_fold" in optional.columns:
            folds = pd.to_numeric(optional["outer_fold"], errors="coerce")
            if folds.isna().any() or (folds < 1).any() or (folds % 1 != 0).any():
                return None, "Optional multi-horizon fold assignments are invalid."
            optional["outer_fold"] = folds.astype(int)
        else:
            mapping = _folds_from_metrics(metrics_path) if metrics_path is not None else None
            mapped = optional["participant_id"].map(mapping) if mapping else None
            if mapped is None or mapped.isna().any():
                return None, "Optional multi-horizon artifact has no verifiable fold metadata."
            optional["outer_fold"] = mapped.astype(int)
        if optional.groupby("participant_id")["outer_fold"].nunique().gt(1).any():
            return None, "Optional multi-horizon fold assignments are inconsistent."
        keys = ["participant_id", "timestamp"]
        if not optional[keys].merge(samples[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").all():
            return None, "Optional multi-horizon artifact does not cover the replay sample keys."
        merged = samples[keys + ["label", "target_glucose"]].merge(
            optional[keys + ["score", "regression", "outer_fold"]], on=keys, how="inner", validate="one_to_one"
        )
        if len(merged) != len(samples):
            return None, "Optional multi-horizon artifact does not cover every replay sample."
        return merged, ""
    except DashboardError as exc:
        return None, str(exc)
    except Exception:
        return None, "Optional multi-horizon artifact could not be validated."


def _profile(row: pd.Series) -> dict[str, float | None]:
    return {key: clean_number(row[key]) for key in row.index if key.startswith("ehr_")}


def _row_payload(row: pd.Series) -> dict[str, object]:
    return {
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "glucose": clean_number(row.get("glucose_now")),
        "heart_rate": clean_number(row.get("heart_rate")),
        "activity": clean_number(row.get("activity_level")),
        "carbs_2h": clean_number(row.get("carbs_last_120m")),
        "meal_events_2h": clean_number(row.get("meal_events_last_120m")),
        "score": clean_number(row.get("score")),
        "label": clean_number(row.get("label")),
        "target_glucose": clean_number(row.get("target_glucose")),
        "fold": int(row["outer_fold"]),
    }


def build_dashboard_data(out_dir: Path | str | None = None) -> dict[str, object]:
    directory = Path(out_dir) if out_dir is not None else OUT
    samples = _validate_samples(_keys(pd.read_csv(_artifact(directory, "rich_120m_samples_local.csv")), "Samples"))
    primary = _keys(pd.read_csv(_artifact(directory, "primary_120m_seed_42_oof_local.csv")), "Primary predictions")
    replay = _validate_primary(samples, primary)
    optional, optional_reason = _validate_optional(
        samples, directory / "horizon_120m_regression_local.csv", directory / "revised_full_metrics.json"
    )

    dynamic_columns = [
        column for column in samples.columns
        if column not in {"participant_id", "timestamp", "target_glucose", "label", "horizon_minutes"}
        and not column.startswith("ehr_")
    ]
    participants: list[dict[str, object]] = []
    for participant_id, group in replay.groupby("participant_id", sort=True):
        group = group.sort_values("timestamp")
        participants.append({
            "id": participant_id,
            "profile": _profile(group.iloc[0]),
            "overview": {
                "start": pd.Timestamp(group.timestamp.iloc[0]).isoformat(),
                "end": pd.Timestamp(group.timestamp.iloc[-1]).isoformat(),
                "eligible_observations": int(len(group)),
                "feature_count": int(len(dynamic_columns) + len(_profile(group.iloc[0]))),
            },
            "rows": [_row_payload(row) for _, row in group.iterrows()],
        })

    return {
        "participants": participants,
        "primary": replay,
        "optional": optional,
        "optional_reason": optional_reason,
        "metadata": {
            "notice": NOTICE,
            "evaluation": "Seed-42 participant-grouped held-out replay; primary source is not the headline five-assignment summary.",
            "source": "CGMacros v1.0.0, PhysioNet; time-shifted research dataset",
            "dynamic_feature_count": len(dynamic_columns),
        },
    }


def dashboard_index(data: dict[str, object]) -> dict[str, object]:
    participants = []
    for participant in data["participants"]:  # type: ignore[index]
        participants.append({
            "id": participant["id"],
            "profile": participant["profile"],
            "overview": participant["overview"],
            "timestamps": [row["timestamp"] for row in participant["rows"]],
        })
    return {
        "notice": NOTICE,
        "evaluation": data["metadata"]["evaluation"],  # type: ignore[index]
        "source": data["metadata"]["source"],  # type: ignore[index]
        "personalization": PERSONALIZATION,
        "sources": [
            {"id": "primary", "label": "Primary seed-42 OOF", "available": True},
            {
                "id": "multi_horizon",
                "label": "Seed-42 multi-horizon paired regression",
                "available": data["optional"] is not None,
                "reason": "" if data["optional"] is not None else data["optional_reason"],
            },
        ],
        "participants": participants,
    }


def _find_participant(data: dict[str, object], participant: str) -> dict[str, object]:
    normalized = str(participant).zfill(3)
    for item in data["participants"]:  # type: ignore[index]
        if item["id"] == normalized:
            return item
    raise DashboardError("Unknown participant.")


def _find_row(participant: dict[str, object], timestamp: pd.Timestamp) -> dict[str, object]:
    iso = timestamp.isoformat()
    for row in participant["rows"]:  # type: ignore[index]
        if row["timestamp"] == iso:
            return row
    raise DashboardError("Unknown forecast timestamp.")


def _worklist(data: dict[str, object], timestamp: pd.Timestamp, source: str) -> list[dict[str, object]]:
    rows = []
    optional = data["optional"]
    for participant in data["participants"]:  # type: ignore[index]
        row = next((item for item in participant["rows"] if item["timestamp"] == timestamp.isoformat()), None)
        if row is None:
            rows.append({"id": participant["id"], "timestamp": None, "score": None, "glucose": None})
            continue
        if source == "multi_horizon" and optional is not None:
            match = optional[(optional["participant_id"] == participant["id"]) & (optional["timestamp"] == timestamp)]
            score = clean_number(match.iloc[0]["score"]) if len(match) else None
        else:
            score = row["score"]
        rows.append({"id": participant["id"], "timestamp": row["timestamp"], "score": score, "glucose": row["glucose"]})
    return sorted(rows, key=lambda item: item["score"] if item["score"] is not None else -1, reverse=True)


def replay_snapshot(
    data: dict[str, object],
    participant: str,
    timestamp: str,
    source: str = "primary",
    elapsed_minutes: int = 0,
) -> dict[str, object]:
    if source not in {"primary", "multi_horizon"} or elapsed_minutes not in {0, HORIZON_MINUTES}:
        raise DashboardError("Invalid replay selection.")
    if source == "multi_horizon" and data["optional"] is None:
        raise DashboardError("Requested replay source is unavailable.")
    try:
        stamp = pd.Timestamp(timestamp)
    except Exception:
        raise DashboardError("Invalid replay selection.") from None
    if stamp.tzinfo is not None or stamp.minute % SAMPLE_MINUTES or stamp.second or stamp.microsecond:
        raise DashboardError("Invalid replay selection.")

    selected = _find_participant(data, participant)
    row = _find_row(selected, stamp)
    target = stamp + pd.Timedelta(minutes=HORIZON_MINUTES)
    clock = stamp + pd.Timedelta(minutes=elapsed_minutes)
    optional = data["optional"]
    optional_match = None
    if source == "multi_horizon" and optional is not None:
        optional_match = optional[
            (optional["participant_id"] == selected["id"]) & (optional["timestamp"] == stamp)
        ]
    forecast_score = clean_number(optional_match.iloc[0]["score"]) if optional_match is not None and len(optional_match) else row["score"]
    forecast_glucose = clean_number(optional_match.iloc[0]["regression"]) if optional_match is not None and len(optional_match) else None
    forecast_fold = int(optional_match.iloc[0]["outer_fold"]) if optional_match is not None and len(optional_match) else row["fold"]
    streams = {key: ("available" if row.get(key) is not None else "missing") for key in STREAM_COLUMNS}
    target_glucose = row.get("target_glucose")
    outcome_available = elapsed_minutes == HORIZON_MINUTES and target_glucose is not None
    event = int(target_glucose >= THRESHOLD_MG_DL) if outcome_available else None
    absolute_error = abs(forecast_glucose - target_glucose) if outcome_available and forecast_glucose is not None else None
    brier_error = (forecast_score - event) ** 2 if outcome_available and forecast_score is not None else None
    timeline = [
        {"timestamp": item["timestamp"], **{key: item.get(key) for key in STREAM_COLUMNS}}
        for item in selected["rows"]  # type: ignore[index]
        if stamp - pd.Timedelta(hours=24) <= pd.Timestamp(item["timestamp"]) <= clock
    ]
    return {
        "notice": NOTICE,
        "participant_id": selected["id"],
        "forecast_timestamp": stamp.isoformat(),
        "target_timestamp": target.isoformat(),
        "replay_clock": clock.isoformat(),
        "horizon_minutes": HORIZON_MINUTES,
        "threshold_mg_dl": THRESHOLD_MG_DL,
        "forecast": {
            "score": forecast_score,
            "glucose_mg_dl": forecast_glucose,
            "fold": forecast_fold,
            "source": source,
            "provenance": "Seed-42 multi-horizon SGKF demo replay source" if source == "multi_horizon" else "Seed-42 primary participant-grouped OOF replay source",
            "calibration": "Uncalibrated model score; not externally validated",
        },
        "context": {key: row.get(key) for key in STREAM_COLUMNS},
        "quality": {
            "missing_feature_count": sum(value == "missing" for value in streams.values()),
            "feature_count": len(STREAM_COLUMNS),
            "streams": streams,
            "note": "Aligned model-input snapshots; sensor values may carry forward up to 30 minutes; carbs and meal counts are rolling 2-hour summaries, not exact meal events. A zero summary is not proof that the meal stream was complete.",
        },
        "timeline": timeline,
        "outcome": {
            "status": "available" if outcome_available else ("unavailable" if elapsed_minutes == HORIZON_MINUTES else "hidden"),
            "glucose_mg_dl": target_glucose if outcome_available else None,
            "event": event,
            "absolute_error_mg_dl": absolute_error,
            "brier_error": brier_error,
            "reason": "Recorded exact t+120 outcome." if outcome_available else (
                "No exact t+120 CGM outcome is available for this replay timestamp." if elapsed_minutes == HORIZON_MINUTES else "Outcome is hidden until explicit t+120 advance."
            ),
        },
        "worklist": _worklist(data, stamp, source),
        "personalization": PERSONALIZATION,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    replay_data: dict[str, object] | None = None
    data_json: bytes = b"{}"

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/data":
                self._send(self.data_json, "application/json; charset=utf-8")
                return
            if parsed.path == "/api/replay":
                if self.replay_data is None:
                    raise DashboardError("Dashboard replay data is unavailable.")
                query = parse_qs(parsed.query)
                try:
                    elapsed = int(query.get("elapsed", ["0"])[0])
                except (TypeError, ValueError):
                    raise DashboardError("Invalid replay selection.") from None
                result = replay_snapshot(
                    self.replay_data,
                    query.get("participant", [""])[0],
                    query.get("timestamp", [""])[0],
                    query.get("source", ["primary"])[0],
                    elapsed,
                )
                self._send(json.dumps(result, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")
                return
            if parsed.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/cgmacros_replay.js":
                self._send(SCRIPT.read_bytes(), "text/javascript; charset=utf-8")
                return
            self.send_error(404)
        except (DashboardError, KeyError, OSError, ValueError):
            self.send_error(400, "Invalid dashboard request")

    def log_message(self, *_args: object) -> None:
        # Query strings contain participant/time selections; do not log them.
        return


def main() -> None:
    if not PAGE.is_file() or not SCRIPT.is_file():
        print("Dashboard page or script is missing; the dashboard was not started.", file=sys.stderr)
        raise SystemExit(1)
    try:
        data = build_dashboard_data()
    except DashboardError as exc:
        print(
            "Dashboard replay outputs are missing or invalid; the dashboard was not started. "
            "Run the documented local evaluation command after obtaining CGMacros.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    DashboardHandler.replay_data = data
    DashboardHandler.data_json = json.dumps(dashboard_index(data), allow_nan=False).encode("utf-8")
    print("Opening local-only CGMacros replay dashboard at http://127.0.0.1:8765")
    HTTPServer(("127.0.0.1", 8765), DashboardHandler).serve_forever()


if __name__ == "__main__":
    main()
