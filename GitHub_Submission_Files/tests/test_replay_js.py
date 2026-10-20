"""Pure timeline helper checks for cgmacros_replay.js, run in Node (no browser, no data)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "cgmacros_replay.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


def run_node(expression: str):
    program = f"const m = require({json.dumps(str(SCRIPT))}); console.log(JSON.stringify({expression}));"
    done = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(done.stdout)


def stamp(minutes: int) -> str:
    return f"new Date(Date.UTC(2025, 0, 1, 0, {minutes})).toISOString()"


def test_nulls_are_gaps_not_zeros_and_real_zero_is_kept():
    rows = f"[{{timestamp: {stamp(0)}, v: 100}}, {{timestamp: {stamp(15)}, v: null}}, {{timestamp: {stamp(30)}, v: 110}}, {{timestamp: {stamp(45)}, v: 0}}, {{timestamp: {stamp(60)}, v: ''}}]"
    segments = run_node(f"m.segmentSeries({rows}, 'v').map(s => s.map(p => p.value))")
    assert segments == [[100], [110, 0]]


def test_lines_break_across_recording_gaps():
    rows = f"[{{timestamp: {stamp(0)}, v: 1}}, {{timestamp: {stamp(15)}, v: 2}}, {{timestamp: {stamp(120)}, v: 3}}, {{timestamp: {stamp(135)}, v: 4}}]"
    segments = run_node(f"m.segmentSeries({rows}, 'v').map(s => s.map(p => p.value))")
    assert segments == [[1, 2], [3, 4]]
    # A normal 15-minute step is never treated as a gap.
    assert run_node("m.GAP_MS") > 15 * 60 * 1000


def test_finite_or_null_handles_missing_and_nonfinite_values():
    assert run_node("[m.finiteOrNull(null), m.finiteOrNull(undefined), m.finiteOrNull(''), m.finiteOrNull('x'), m.finiteOrNull(0), m.finiteOrNull('7.5')]") == [None, None, None, None, 0, 7.5]
