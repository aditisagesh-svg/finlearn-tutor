"""
Trajectory-aware task graders for FinLearn Tutor.
ALL scores, rewards, and intermediate values are clamped to (0.01, 0.99).
Zero is never returned anywhere.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ── Crash-proof imports with fallback stubs ───────────────────────────────────
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
except Exception:
    Observation = None  # type: ignore

    def clamp_score(x: float) -> float:          # type: ignore
        return float(max(0.01, min(0.99, x)))

    def compute_returns(ph: list) -> list:        # type: ignore
        if len(ph) < 2:
            return [0.01]
        return [(ph[i] - ph[i-1]) / max(abs(ph[i-1]), 1e-9) for i in range(1, len(ph))]

    def compute_drawdown(ph: list) -> float:      # type: ignore
        if not ph:
            return 0.01
        peak, dd = ph[0], 0.0
        for v in ph:
            if v > peak:
                peak = v
            dd = max(dd, (peak - v) / max(abs(peak), 1e-9))
        return max(dd, 0.0)

    def compute_volatility(returns: list) -> float:  # type: ignore
        if len(returns) < 2:
            return 0.001          # never exactly 0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns)) or 0.001

    def compute_trade_efficiency(actions: list, ph: list) -> float:  # type: ignore
        return 0.50

    def compute_regime_adaptation(steps: list) -> float:            # type: ignore
        return 0.50

    def normalize_growth(growth: float, target: float) -> float:    # type: ignore
        if target <= 0:
            return 0.50
        return min(1.0, max(0.0, growth / target))

    def normalize_inverse(value: float, cap: float) -> float:       # type: ignore
        if cap <= 0:
            return 0.50
        return max(0.0, 1.0 - value / cap)


# ── Master clamp — applied at EVERY exit point ────────────────────────────────

def _safe(x: Any) -> float:
    """
    Clamps any value to strictly (0.01, 0.99).
    Handles nan, inf, None, non-numeric. Never raises. Never returns 0.0 or 1.0.
    """
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        return round(max(0.01, min(0.99, v)), 2)
    except Exception:
        return 0.50


def _safe_reward(x: Any) -> float:
    """Same as _safe but named for reward context — clamps step rewards too."""
    return _safe(x)


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

# Backward-compat aliases used by legacy code
TASK_CONFIGS["task1_capital_preservation"]    = TASK_CONFIGS["task1"]
TASK_CONFIGS["task2_balanced_growth"]         = TASK_CONFIGS["task2"]
TASK_CONFIGS["task3_aggressive_optimization"] = TASK_CONFIGS["task3"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_dict(final_state: Any) -> Dict:
    try:
        if hasattr(final_state, "model_dump"):
            return final_state.model_dump()
        if isinstance(final_state, dict):
            return final_state
    except Exception:
        pass
    return {}


def _safe_ph(portfolio_history: Any, initial_value: float, portfolio_now: float) -> List[float]:
    """
    Ensure portfolio_history is a list with >= 2 numeric elements.
    A single-element list → compute_returns returns [] → volatility = 0.0 → bug.
    """
    if isinstance(portfolio_history, list) and len(portfolio_history) >= 2:
        return [float(v) for v in portfolio_history]
    # Fallback: two-point history
    return [float(initial_value), float(portfolio_now)]


def _build_metrics(final_state: Any, initial_value: float, trajectory: Optional[Dict]) -> Dict:
    state = _as_dict(final_state)
    traj  = trajectory if isinstance(trajectory, dict) else {}

    portfolio_now = float(state.get("portfolio_value") or initial_value)
    ph = _safe_ph(traj.get("portfolio_history"), initial_value, portfolio_now)

    actions = traj.get("action_history", [])
    steps   = traj.get("step_records",   [])

    returns    = compute_returns(ph)
    # Guard: returns must be non-empty so volatility is never exactly 0
    if not returns:
        returns = [0.001]

    growth     = (ph[-1] - ph[0]) / max(abs(ph[0]), 1e-9)
    drawdown   = compute_drawdown(ph)
    volatility = compute_volatility(returns) or 0.001   # never exactly 0

    trade_count = sum(1 for a in actions if a not in (0, 8))
    trade_eff   = compute_trade_efficiency(actions, ph)
    regime      = compute_regime_adaptation(steps)

    # Clamp intermediate values — none can be exactly 0.0 or 1.0
    dq = _safe(clamp_score((_safe(trade_eff) + _safe(regime)) / 2.0))

    return {
        "growth":      growth,
        "drawdown":    max(drawdown, 0.0),
        "volatility":  volatility,
        "trade_count": trade_count,
        "trade_eff":   _safe(trade_eff),
        "regime":      _safe(regime),
        "dq":          dq,
    }


def _score_task(final_state: Any, initial_value: float, trajectory: Optional[Dict], key: str) -> float:
    """Score one task. Returns _safe() float. Never raises."""
    try:
        m = _build_metrics(final_state, initial_value, trajectory)
        w = TASK_CONFIGS[key]["weights"]
        t = TASK_CONFIGS[key]["targets"]

        # Clamp each intermediate normalize output — prevents 0.0 / 1.0 from bleeding in
        gs  = _safe(normalize_growth(m["growth"],      t["growth"]))
        rcs = _safe(normalize_inverse(m["drawdown"],   t["drawdown_cap"]))
        ss  = _safe(normalize_inverse(m["volatility"], t["vol_cap"]))
        ts  = _safe(normalize_inverse(m["trade_count"],t["trade_cap"]))
        dqs = _safe(clamp_score(m["dq"] * 0.7 + ts * 0.3))

        raw = (
            gs  * w["growth"]
            + rcs * w["risk_control"]
            + ss  * w["stability"]
            + dqs * w["decision_quality"]
        )
        return _safe(raw)
    except Exception:
        return 0.05


# ── Public graders ────────────────────────────────────────────────────────────

def grade_task1(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Capital Preservation. Always returns float in (0.01, 0.99)."""
    try:
        return _safe(_score_task(final_state or {}, initial_value, trajectory, "task1"))
    except Exception:
        return 0.05


def grade_task2(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Balanced Growth. Always returns float in (0.01, 0.99)."""
    try:
        return _safe(_score_task(final_state or {}, initial_value, trajectory, "task2"))
    except Exception:
        return 0.05


def grade_task3(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Aggressive Optimization. Always returns float in (0.01, 0.99)."""
    try:
        return _safe(_score_task(final_state or {}, initial_value, trajectory, "task3"))
    except Exception:
        return 0.05


# ── Aggregate — all 3 tasks in one call, each independently guarded ───────────

def run_all_tasks(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> Dict:
    try:
        s1 = grade_task1(final_state, initial_value, trajectory)
    except Exception:
        s1 = 0.05

    try:
        s2 = grade_task2(final_state, initial_value, trajectory)
    except Exception:
        s2 = 0.05

    try:
        s3 = grade_task3(final_state, initial_value, trajectory)
    except Exception:
        s3 = 0.05

    overall = _safe((s1 + s2 + s3) / 3.0)

    return {
        # Primary keys — match openenv.yaml task IDs exactly
        "task1":         s1,
        "task2":         s2,
        "task3":         s3,
        "overall_score": overall,
        # Legacy aliases for backward compat
        "task1_capital_preservation":    s1,
        "task2_balanced_growth":         s2,
        "task3_aggressive_optimization": s3,
    }


# ── Registry ──────────────────────────────────────────────────────────────────

TASKS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}