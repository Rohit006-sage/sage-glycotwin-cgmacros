"""Small deterministic glucose ODE for research prototyping.

States use glucose mg/dL, gut carbohydrate grams, insulin action mg/dL/min,
and cumulative absorbed carbohydrate grams. Meal carbohydrates enter the fast
gut compartment; absorption is split across two transit compartments. The
latent insulin-action state is a phenomenological feedback state, not measured
insulin or a medication effect.

All tunable quantities have explicit bounds in ModelParameters. HOMA-IR is
used only to seed a bounded relative insulin-sensitivity prior. The equations
are deliberately simple and are not suitable for dosing or clinical advice.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    # Time constants/rates: minutes or min^-1. Glucose rates: mg/dL/min.
    gut_transit_min: float = 18.0       # [5, 60] min
    gut_absorb_min: float = 35.0        # [10, 120] min
    glucose_clearance_min: float = 0.006  # [0.001, 0.03] min^-1
    glucose_per_gram: float = 0.75      # [0.1, 2.0] mg/dL per absorbed gram (effective volume)
    insulin_feedback_min: float = 0.015 # [0.001, 0.08] min^-1 per mg/dL deviation
    insulin_decay_min: float = 90.0     # [20, 300] min
    insulin_sensitivity: float = 1.0    # [0.25, 2.5], relative
    activity_uptake: float = 0.015      # [0, 0.08] mg/dL/min per normalized activity
    activity_sensitivity: float = 0.5  # [0, 2], relative to activity covariate

    def __post_init__(self) -> None:
        bounds = {
            "gut_transit_min": (5, 60), "gut_absorb_min": (10, 120),
            "glucose_clearance_min": (0.001, 0.03), "glucose_per_gram": (0.1, 2.0),
            "insulin_feedback_min": (0.001, 0.08), "insulin_decay_min": (20, 300),
            "insulin_sensitivity": (0.25, 2.5), "activity_uptake": (0, 0.08),
            "activity_sensitivity": (0, 2),
        }
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be within [{lo}, {hi}]")


@dataclass(frozen=True)
class State:
    glucose_mg_dl: float
    gut_fast_g: float = 0.0
    gut_slow_g: float = 0.0
    insulin_action_mg_dl: float = 0.0
    absorbed_g: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.glucose_mg_dl, self.gut_fast_g, self.gut_slow_g,
                         self.insulin_action_mg_dl, self.absorbed_g], dtype=float)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "State":
        return cls(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))


def homa_ir(fasting_glucose_mg_dl: float, fasting_insulin_uU_ml: float) -> float:
    """Return HOMA-IR using glucose in mg/dL and insulin in uU/mL."""
    if fasting_glucose_mg_dl <= 0 or fasting_insulin_uU_ml <= 0:
        raise ValueError("fasting glucose and insulin must be positive")
    return float(fasting_glucose_mg_dl * fasting_insulin_uU_ml / 405.0)


def sensitivity_prior(fasting_glucose_mg_dl: float, fasting_insulin_uU_ml: float) -> float:
    """Map HOMA-IR to bounded relative sensitivity; cohort fit is deferred."""
    h = homa_ir(fasting_glucose_mg_dl, fasting_insulin_uU_ml)
    # Reference HOMA-IR=2; conservative clipping prevents extreme priors.
    return float(np.clip(2.0 / max(h, 1e-6), 0.25, 2.5))


def _derivative(x: np.ndarray, baseline: float, activity: float, p: ModelParameters) -> np.ndarray:
    g, fast, slow, action, absorbed = x
    transit = fast / p.gut_transit_min
    uptake = slow / p.gut_absorb_min
    # Effective carbohydrate appearance in mg/dL/min.
    appearance = p.glucose_per_gram * uptake
    # Basal production balances first-order clearance at baseline glucose.
    egp = p.glucose_clearance_min * baseline
    clearance = p.glucose_clearance_min * max(g, 0.0)
    endogenous_feedback = max(g - baseline, 0.0) * p.insulin_feedback_min
    insulin_effect = p.insulin_sensitivity * p.insulin_feedback_min * max(action, 0.0)
    activity_effect = (p.activity_uptake * p.activity_sensitivity *
                       max(activity, 0.0) * p.glucose_clearance_min * max(g, 0.0))
    return np.array([egp + appearance - clearance - insulin_effect - activity_effect,
                     -transit, transit - uptake, endogenous_feedback - action / p.insulin_decay_min,
                     uptake], dtype=float)


def step(state: State, dt_min: float, *, baseline_mg_dl: float,
         activity: float = 0.0, p: ModelParameters = ModelParameters()) -> State:
    """Advance one RK4 step. activity=1 denotes a moderate normalized bout."""
    if dt_min <= 0 or baseline_mg_dl <= 0 or not math.isfinite(activity):
        raise ValueError("dt and baseline must be positive; activity must be finite")
    x = state.as_array()
    k1 = _derivative(x, baseline_mg_dl, activity, p)
    k2 = _derivative(x + dt_min*k1/2, baseline_mg_dl, activity, p)
    k3 = _derivative(x + dt_min*k2/2, baseline_mg_dl, activity, p)
    k4 = _derivative(x + dt_min*k3, baseline_mg_dl, activity, p)
    y = x + dt_min*(k1 + 2*k2 + 2*k3 + k4)/6
    # Numerical guard for nonnegative compartments/states.
    y[0] = max(0.0, y[0]); y[1:4] = np.maximum(0.0, y[1:4]); y[4] = max(0.0, y[4])
    return State.from_array(y)


def simulate(initial_glucose_mg_dl: float | State, minutes: int, *, baseline_mg_dl: float | None = None,
             meal: float | None = None, activity: float | Iterable[float] | None = None,
             p: ModelParameters = ModelParameters(), dt_min: float = 1.0) -> list[State]:
    """Simulate from a state; meal is grams at t=0, activity is normalized per step.

    ``simulate(state, meal=None, activity=None)`` is exposed via accepting a
    State as the first argument; this supports later dashboard what-if calls.
    """
    if isinstance(initial_glucose_mg_dl, State):
        initial = initial_glucose_mg_dl
        g0 = initial.glucose_mg_dl
    else:
        g0 = float(initial_glucose_mg_dl)
        initial = State(g0)
    if minutes < 0 or dt_min <= 0:
        raise ValueError("minutes must be nonnegative and dt_min positive")
    baseline = float(baseline_mg_dl if baseline_mg_dl is not None else g0)
    if meal is not None and meal < 0:
        raise ValueError("meal carbohydrate must be nonnegative")
    if meal:
        initial = State(initial.glucose_mg_dl, initial.gut_fast_g + float(meal),
                         initial.gut_slow_g, initial.insulin_action_mg_dl, initial.absorbed_g)
    n = int(math.ceil(minutes / dt_min))
    if activity is None or np.isscalar(activity):
        acts = [float(activity or 0.0)] * n
    else:
        acts = list(activity)
        if len(acts) < n:
            acts.extend([0.0] * (n - len(acts)))
    states = [initial]
    for i in range(n):
        h = min(dt_min, minutes - i*dt_min)
        if h <= 0: break
        states.append(step(states[-1], h, baseline_mg_dl=baseline, activity=acts[i], p=p))
    return states
