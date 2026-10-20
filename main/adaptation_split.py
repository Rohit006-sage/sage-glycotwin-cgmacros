"""Label-availability-aware chronological split for within-person adaptation.

A row stamped ``t`` has its label observed at ``t + horizon``. A row may enter the
adaptation (training) set only if that label is already known at the adaptation
cutoff. Rows whose forecast time precedes the cutoff but whose label arrives after
it are purged: they appear in neither training nor test data.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class AdaptationSplit(NamedTuple):
    cutoff: pd.Timestamp | None
    train: np.ndarray   # boolean mask, label observed by the cutoff
    purged: np.ndarray  # boolean mask, forecast before cutoff but label after it
    test: np.ndarray    # boolean mask, forecast at or after the cutoff


def split_adaptation(timestamps, adaptation_hours: int = 72, horizon_minutes: int = 120) -> AdaptationSplit:
    ts = pd.Series(pd.to_datetime(pd.Series(timestamps))).reset_index(drop=True)
    if ts.empty:
        empty = np.zeros(0, dtype=bool)
        return AdaptationSplit(None, empty, empty, empty)
    cutoff = ts.min() + pd.Timedelta(hours=adaptation_hours)
    label_time = ts + pd.Timedelta(minutes=horizon_minutes)
    train = (label_time <= cutoff).to_numpy()
    test = (ts >= cutoff).to_numpy()
    purged = ~train & ~test
    return AdaptationSplit(cutoff, train, purged, test)
