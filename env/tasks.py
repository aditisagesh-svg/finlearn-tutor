"""
Trajectory-aware task graders for FinLearn Tutor.
Imported by OpenEnv validator as: env.tasks
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ── Safe imports with fallback stubs so graders never crash on import ─────────
try:
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
    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False
    Observation = None  # type: ignore

    def clamp_score(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def compute_returns(ph: list) -> list:
        if len(ph) < 2:
            return [0.0]
        return [(ph[i] - ph[i - 1]) / max(abs(ph[i - 1]), 1e-9) for i in range(1, len(ph))]

    def compute_drawdown(ph: list) -> float:
        if not ph:
            return 0.0
        peak = ph[0]
        dd = 0.0
        for v in ph:
            if v > peak:
                peak = v
            dd = max(dd, (peak - v) / max(abs(peak), 1e-9))
        return dd

    def compute_volatility(returns: list) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(var)

    def compute_trade_efficiency(actions: list, ph: list) -> float:
        return 0.5

    def compute_regime_adaptation(steps: list) -> float:
        return 0.5

    def normalize_growth(growth: float, target: float) -> float:
        if target <= 0:
            return 0.5
        return min(1.0, max(0.0, growth / target))

    def normalize_inverse(value: float, cap: float) -> float:
        if cap <= 0:
            return 0.5
        return max(0.0, 1.0 - value / cap)


# ── Safety clamp ──────────────────────────────────────────────────────────────

def _safe(x: float) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        return round(max(0.01, min(0.99, v)), 2)
    except Exception:
        return 0.50


# ── Task configs ──────────────────────────────────────────────────────────────

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


# ── Internal scoring ──────────────────────────────────────────────────────────

def _state_dict(final_state: Any) -> Dict:
    if hasattr(final_state, "model_dump"):
        return final_state.model_dump()
    if isinstance(final_state, dict):
        return final_state
    return {}


def _compute(final_state: Any, initial_value: float = 1000.0, trajectory: Optional[Dict] = None) -> Dict:
    state = _state_dict(final_state)
    traj = trajectory if isinstance(trajectory, dict) else {}
    ph = traj.get("portfolio_history") or [initial_value, state.get("portfolio_value", initial_value)]
    actions = traj.get("action_history", [])
    steps = traj.get("step_records", [])

    returns = compute_returns(ph)
    growth = (ph[-1] - ph[0]) / max(abs(ph[0]), 1e-9)
    drawdown = compute_drawdown(ph)
    volatility = compute_volatility(returns)
    trade_count = sum(1 for a in actions if a not in (0, 8))
    trade_eff = compute_trade_efficiency(actions, ph)
    regime = compute_regime_adaptation(steps)
    dq = clamp_score((trade_eff + regime) / 2.0)

    return {
        "growth": growth,
        "drawdown": drawdown,
        "volatility": volatility,
        "trade_count": trade_count,
        "trade_eff": trade_eff,
        "regime": regime,
        "dq": dq,
    }


def _score_task(final_state: Any, initial_value: float, trajectory: Optional[Dict], key: str) -> float:
    try:
        m = _compute(final_state, initial_value, trajectory)
        w = TASK_CONFIGS[key]["weights"]
        t = TASK_CONFIGS[key]["targets"]

        gs  = normalize_growth(m["growth"], t["growth"])
        rcs = normalize_inverse(m["drawdown"], t["drawdown_cap"])
        ss  = normalize_inverse(m["volatility"], t["vol_cap"])
        ts  = normalize_inverse(m["trade_count"], t["trade_cap"])
        dqs = clamp_score(m["dq"] * 0.7 + ts * 0.3)

        raw = clamp_score(
            gs  * w["growth"]
            + rcs * w["risk_control"]
            + ss  * w["stability"]
            + dqs * w["decision_quality"]
        )
        return _safe(raw)
    except Exception:
        return 0.05


# ── Public graders (validator imports these as env.tasks:grade_taskN) ─────────

def grade_task1(final_state: Any, initial_value: float = 1000.0, trajectory: Optional[Dict] = None) -> float:
    try:
        return _safe(_score_task(final_state, initial_value, trajectory, "task1"))
    except Exception:
        return 0.05


def grade_task2(final_state: Any, initial_value: float = 1000.0, trajectory: Optional[Dict] = None) -> float:
    try:
        return _safe(_score_task(final_state, initial_value, trajectory, "task2"))
    except Exception:
        return 0.05


def grade_task3(final_state: Any, initial_value: float = 1000.0, trajectory: Optional[Dict] = None) -> float:
    try:
        return _safe(_score_task(final_state, initial_value, trajectory, "task3"))
    except Exception:
        return 0.05


# ── Aggregate ─────────────────────────────────────────────────────────────────

def run_all_tasks(
    final_state: Any,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> Dict:
    s1 = grade_task1(final_state, initial_value, trajectory)
    s2 = grade_task2(final_state, initial_value, trajectory)
    s3 = grade_task3(final_state, initial_value, trajectory)
    overall = _safe((s1 + s2 + s3) / 3.0)
    return {
        "capital_preservation": s1,
        "balanced_growth": s2,
        "aggressive_optimization": s3,
        "task1": s1,
        "task2": s2,
        "task3": s3,
        "overall_score": overall,
    }


# ── Registry ──────────────────────────────────────────────────────────────────

TASKS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "capital_preservation": grade_task1,
    "balanced_growth": grade_task2,
    "aggressive_optimization": grade_task3,
}
