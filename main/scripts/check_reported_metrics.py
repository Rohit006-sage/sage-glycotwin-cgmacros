"""Fail when public report artifacts drift from aggregate_metrics.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results/aggregate_metrics.json"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))[
        "within_person_chronological_validation"
    ]
    strategies = data["classification_on_eligible_participants"]
    expected = {
        "population": (
            strategies["population_only"]["pooled"]["pr_auc"],
            strategies["population_only"]["equal_participant_macro"]["pr_auc"],
        ),
        "adaptation": (
            strategies["population_plus_early_adaptation"]["pooled"]["pr_auc"],
            strategies["population_plus_early_adaptation"]["equal_participant_macro"]["pr_auc"],
        ),
        "calibration": (
            strategies["population_plus_early_calibration"]["pooled"]["pr_auc"],
            strategies["population_plus_early_calibration"]["equal_participant_macro"]["pr_auc"],
        ),
        "person_only": (
            strategies["per_person_only"]["pooled"]["pr_auc"],
            strategies["per_person_only"]["equal_participant_macro"]["pr_auc"],
        ),
    }
    regression = data["regression_on_all_participants"]
    mae = {
        name: values["pooled"]["mae_mg_dl"]
        for name, values in regression.items()
    }
    docs = [
        ROOT / "README.md",
        ROOT / "docs/EVALUATION.md",
        ROOT / "docs/MODEL_CARD.md",
        ROOT / "docs/JURY_QA.md",
        ROOT / "docs/VIDEO_SCRIPT.md",
        ROOT / "docs/update_presentation.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    stale = ("18.07", "0.535", "0.544", "0.351", "0.151")
    found_stale = [value for value in stale if value in joined]
    if found_stale:
        raise SystemExit(f"Stale personalization values in text reports: {found_stale}")
    for name, values in expected.items():
        for value in values:
            if _fmt(value) not in joined:
                raise SystemExit(f"Missing aggregate {name} PR-AUC value {_fmt(value)}")
    for value in (mae["population_only"], mae["population_plus_early_adaptation"]):
        rendered = f"{value:.2f}"
        if rendered not in joined:
            raise SystemExit(f"Missing aggregate regression MAE {rendered}")

    deck_path = ROOT / "docs/GlycoTwin_CGMacros_Prototype.pptx"
    with ZipFile(deck_path) as pptx:
        slides = [
            pptx.read(name)
            for name in pptx.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    text = " ".join(
        node.text or ""
        for root in [ElementTree.fromstring(blob) for blob in slides]
        for node in root.findall(".//a:t", ns)
    )
    for value in stale:
        if value in text or value.replace("0.", ".") in text:
            raise SystemExit(f"Stale personalization value {value} remains in the presentation")
    for value in ("0.538", "0.523", "0.251", "0.340", "0.140", "18.10"):
        if value not in text and value.replace("0.", ".") not in text:
            raise SystemExit(f"Presentation is missing canonical value {value}")
    print("Reported personalization metrics match docs/results/aggregate_metrics.json.")


if __name__ == "__main__":
    main()
