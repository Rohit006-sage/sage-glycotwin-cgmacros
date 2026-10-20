"""Purged-boundary checks for within-person adaptation; synthetic timestamps only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptation_split import split_adaptation

HORIZON = pd.Timedelta(minutes=120)


def grid(days: int = 5) -> pd.Series:
    return pd.Series(pd.date_range("2025-01-01", periods=days * 96, freq="15min"))


def check_invariants(ts: pd.Series, result) -> None:
    ts = ts.reset_index(drop=True)
    assert not (result.train & result.test).any()
    assert not (result.train & result.purged).any()
    assert not (result.test & result.purged).any()
    assert (result.train | result.purged | result.test).all()
    # No training label is observed after the cutoff; no test row precedes it.
    assert (ts[result.train] + HORIZON <= result.cutoff).all()
    assert (ts[result.test] >= result.cutoff).all()
    # Purged rows are forecast before the cutoff but their labels arrive after it.
    assert (ts[result.purged] < result.cutoff).all()
    assert (ts[result.purged] + HORIZON > result.cutoff).all()


def test_label_availability_boundary_is_purged_and_inclusive():
    ts = grid()
    result = split_adaptation(ts, 72, 120)
    assert result.cutoff == ts.min() + pd.Timedelta(hours=72)
    check_invariants(ts, result)
    # Rows 105..15 minutes before the cutoff have labels that arrive after it.
    assert int(result.purged.sum()) == 7
    # The last training row's label is observed exactly at the cutoff.
    assert (ts[result.train].max() + HORIZON) == result.cutoff
    # The old timestamp-only split would have trained on exactly these leaking rows.
    old_train = (ts < result.cutoff).to_numpy()
    assert int((old_train & ~result.train).sum()) == int(result.purged.sum())


def test_boundary_holds_with_recording_gaps_and_unsorted_input():
    ts = grid()
    kept = ts.drop(ts.index[250:262]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    result = split_adaptation(kept, 72, 120)
    check_invariants(kept, result)
    assert result.train.any() and result.test.any()


def test_shorter_horizon_purges_fewer_rows_and_empty_input_is_safe():
    ts = grid()
    assert int(split_adaptation(ts, 72, 30).purged.sum()) == 1
    empty = split_adaptation(pd.Series([], dtype="datetime64[ns]"))
    assert empty.cutoff is None and not empty.train.any() and not empty.test.any()
    assert np.asarray(empty.purged).size == 0
