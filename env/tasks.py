"""
Trajectory-aware task graders for FinLearn Tutor.
Loaded by OpenEnv validator as: env.tasks
"""

from __future__ import annotations

import math
from typing import Dict

from env.metrics import (
    clamp_score,
    compute_drawdown,
    compute_regime_adaptation,
    compute_returns,
    compute_trade_efficiency,
    compute_volatility,
    normalize_growth,
    normalize_inverse,
)
from env.models import Observation


def _safe_score(score: float) -> float:
    """Always returns a float strictly between 0.01 and 0.99."""
    try:
        v = float(score)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        return round(max(0.01, min(0.99, v)), 2)
    except Exception:
        return 0.50


TASK_CONFIGS = {
    "task1": {
        "weights": {"growth": 0.20, "risk_control": 0.35, "stability": 0.25, "decision_quality": 0.20},
        "targets": {"growth": 0.04, "drawdown_cap": 0.10, "vol_cap": 0.025, "trade_cap": 8},
    },
    "task2": {
        "weights": {"growth": 0.40, "risk_control": 0.20, "stability": 0.20, "decision_quality": 0.20},
        "targets": {"growth": 0.10, "drawdown_cap": 0.18, "vol_cap": 0.035, "trade_cap": 12},
    },
    "task3": {
        "weights": {"growth": 0.50, "risk_control": 0.15, "stability": 0.10, "decision_quality": 0.25},
        "targets": {"growth": 0.18, "drawdown_cap": 0.28, "vol_cap": 0.05, "trade_cap": 16},
    },
}


def _as_state_dict(final_state):
    return final_state.model_dump() if isinstance(final_state, Observation) else final_state


def _compute_metrics(final_state, initial_value=1000.0, trajectory=None):
    state = _as_state_dict(final_state)
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    ph = trajectory.get("portfolio_history") or [initial_value, state["portfolio_value"]]
    actions = trajectory.get("action_history", [])
    steps = trajectory.get("step_records", [])

    returns = compute_returns(ph)
    growth = (ph[-1] - ph[0]) / max(ph[0], 1e-9)
    drawdown = compute_drawdown(ph)
    volatility = compute_volatility(returns)
    trade_count = sum(1 for a in actions if a not in (0, 8))
    trade_eff = compute_trade_efficiency(actions, ph)
    regime_adap = compute_regime_adaptation(steps)
    dq = clamp_score((trade_eff + regime_adap) / 2.0)
    return growth, drawdown, volatility, trade_count, trade_eff, regime_adap, dq


def _score(final_state, initial_value=1000.0, trajectory=None, task_key="task2"):
    try:
        growth, drawdown, volatility, trade_count, trade_eff, regime_adap, dq = \
            _compute_metrics(final_state, initial_value, trajectory)
        w = TASK_CONFIGS[task_key]["weights"]
        t = TASK_CONFIGS[task_key]["targets"]

        gs   = normalize_growth(growth, t["growth"])
        rcs  = normalize_inverse(drawdown, t["drawdown_cap"])
        ss   = normalize_inverse(volatility, t["vol_cap"])
        ts   = normalize_inverse(trade_count, t["trade_cap"])
        dqs  = clamp_score(dq * 0.7 + ts * 0.3)

        raw = clamp_score(gs * w["growth"] + rcs * w["risk_control"] +
                          ss * w["stability"] + dqs * w["decision_quality"])
        return _safe_score(raw)
    except Exception:
        return 0.05


def grade_task1(final_state, initial_value=1000.0, trajectory=None):
    try:
        return _safe_score(_score(final_state, initial_value, trajectory, "task1"))
    except Exception:
        return 0.05


def grade_task2(final_state, initial_value=1000.0, trajectory=None):
    try:
        return _safe_score(_score(final_state, initial_value, trajectory, "task2"))
    except Exception:
        return 0.05


def grade_task3(final_state, initial_value=1000.0, trajectory=None):
    try:
        return _safe_score(_score(final_state, initial_value, trajectory, "task3"))
    except Exception:
        return 0.05


def run_all_tasks(final_state, initial_value=1000.0, trajectory=None):
    s1 = grade_task1(final_state, initial_value, trajectory)
    s2 = grade_task2(final_state, initial_value, trajectory)
    s3 = grade_task3(final_state, initial_value, trajectory)
    overall = _safe_score((s1 + s2 + s3) / 3.0)
    return {
        "task1": s1,
        "task2": s2,
        "task3": s3,
        "task1_capital_preservation": s1,
        "task2_balanced_growth": s2,
        "task3_aggressive_optimization": s3,
        "overall_score": overall,
    }


TASKS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
