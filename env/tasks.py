"""
Trajectory-aware task graders for FinLearn Tutor.

Mirrors the exact pattern of passing OpenEnv submissions:
- _EPS constant for open-interval clamping
- clamp_score() called on every return path
- format_score() for clean int/float output
- Named grade_* functions per scoring component
- ScoreBreakdown dict with per-component scores
- Partial credit on intermediate metrics
- All scores strictly in (0, 1) — never 0.0 or 1.0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

# ── Crash-proof imports with fallback stubs ───────────────────────────────────
try:
    from env.metrics import (
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

    def compute_returns(ph: list) -> list:          # type: ignore
        if len(ph) < 2:
            return [0.001]
        return [(ph[i] - ph[i-1]) / max(abs(ph[i-1]), 1e-9) for i in range(1, len(ph))]

    def compute_drawdown(ph: list) -> float:        # type: ignore
        if not ph:
            return 0.001
        peak, dd = ph[0], 0.0
        for v in ph:
            if v > peak:
                peak = v
            dd = max(dd, (peak - v) / max(abs(peak), 1e-9))
        return dd or 0.001

    def compute_volatility(returns: list) -> float: # type: ignore
        if len(returns) < 2:
            return 0.001
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns)) or 0.001

    def compute_trade_efficiency(a: list, ph: list) -> float:  # type: ignore
        return 0.50

    def compute_regime_adaptation(s: list) -> float:           # type: ignore
        return 0.50

    def normalize_growth(growth: float, target: float) -> float:   # type: ignore
        return min(1.0, max(0.0, growth / target)) if target > 0 else 0.50

    def normalize_inverse(value: float, cap: float) -> float:      # type: ignore
        return max(0.0, 1.0 - value / cap) if cap > 0 else 0.50


# ── Core clamping — exact pattern from passing reference ─────────────────────

_EPS = 0.01  # keeps scores strictly inside (0, 1)


def clamp_score(score: float) -> float:
    """Clamp to open interval (0, 1). Never returns 0.0 or 1.0."""
    try:
        v = float(score)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        if v <= 0.0:
            return _EPS          # 0.0  → 0.01
        if v >= 1.0:
            return 1.0 - _EPS   # 1.0  → 0.99
        return v
    except Exception:
        return 0.50


def format_score(score: float) -> Union[int, float]:
    """Return int for whole numbers, float otherwise. Mirrors reference exactly."""
    try:
        return int(score) if score == int(score) else round(float(score), 4)
    except Exception:
        return round(float(score), 4)


def safe(score: float) -> float:
    """Round to 4dp then clamp. Applied at every public exit point."""
    return clamp_score(round(float(score), 4))


# ── Task configs ──────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task1": {
        "label": "Capital Preservation",
        "difficulty": "easy",
        "scoring": {
            "growth_weight":   0.20,
            "risk_weight":     0.35,
            "stability_weight":0.25,
            "quality_weight":  0.20,
        },
        "targets": {"growth": 0.04, "drawdown_cap": 0.10, "vol_cap": 0.025, "trade_cap": 8},
    },
    "task2": {
        "label": "Balanced Growth",
        "difficulty": "medium",
        "scoring": {
            "growth_weight":   0.40,
            "risk_weight":     0.20,
            "stability_weight":0.20,
            "quality_weight":  0.20,
        },
        "targets": {"growth": 0.10, "drawdown_cap": 0.18, "vol_cap": 0.035, "trade_cap": 12},
    },
    "task3": {
        "label": "Aggressive Optimization",
        "difficulty": "hard",
        "scoring": {
            "growth_weight":   0.50,
            "risk_weight":     0.15,
            "stability_weight":0.10,
            "quality_weight":  0.25,
        },
        "targets": {"growth": 0.18, "drawdown_cap": 0.28, "vol_cap": 0.05, "trade_cap": 16},
    },
}

# Backward-compat aliases
TASK_CONFIGS["task1_capital_preservation"]    = TASK_CONFIGS["task1"]
TASK_CONFIGS["task2_balanced_growth"]         = TASK_CONFIGS["task2"]
TASK_CONFIGS["task3_aggressive_optimization"] = TASK_CONFIGS["task3"]


# ── Named grade_* functions — one per scoring component ──────────────────────

def grade_growth(growth: float, target: float) -> float:
    """Score portfolio growth. Partial credit for partial progress."""
    if target <= 0:
        return clamp_score(0.50)
    ratio = growth / target
    if ratio >= 1.0:
        return clamp_score(1.0)
    elif ratio >= 0.5:
        return clamp_score(0.5 + 0.5 * (ratio - 0.5) / 0.5)  # 0.50–0.99
    elif ratio > 0:
        return clamp_score(0.5 * ratio / 0.5)                  # 0.01–0.49
    else:
        return clamp_score(0.0)   # negative growth → _EPS


def grade_risk(drawdown: float, drawdown_cap: float) -> float:
    """Score drawdown control. Less drawdown = higher score."""
    if drawdown_cap <= 0:
        return clamp_score(0.50)
    ratio = drawdown / drawdown_cap
    if ratio <= 0:
        return clamp_score(1.0)
    elif ratio <= 1.0:
        return clamp_score(1.0 - ratio)   # linear: 0 dd → 0.99, cap dd → _EPS
    else:
        return clamp_score(0.0)           # exceeded cap → _EPS


def grade_stability(volatility: float, vol_cap: float) -> float:
    """Score return stability. Lower volatility = higher score."""
    if vol_cap <= 0:
        return clamp_score(0.50)
    ratio = volatility / vol_cap
    if ratio <= 0:
        return clamp_score(1.0)
    elif ratio <= 1.0:
        return clamp_score(1.0 - ratio)
    else:
        return clamp_score(0.0)


def grade_trade_quality(trade_count: int, trade_cap: int,
                        trade_eff: float, regime: float) -> float:
    """Score trade decision quality with partial credit."""
    # Efficiency component (0–0.5): penalise overtrading linearly
    if trade_cap <= 0:
        eff_component = 0.50
    elif trade_count <= trade_cap:
        eff_component = 0.50
    else:
        excess = trade_count - trade_cap
        eff_component = clamp_score(0.5 * max(0.0, 1.0 - excess / max(trade_cap, 1)))

    # Regime adaptation component (0–0.5)
    regime_component = clamp_score(regime) * 0.5

    return clamp_score(eff_component + regime_component)


# ── Episode metric extraction ─────────────────────────────────────────────────

def _as_dict(final_state: Any) -> Dict:
    try:
        if hasattr(final_state, "model_dump"):
            return final_state.model_dump()
        if isinstance(final_state, dict):
            return final_state
    except Exception:
        pass
    return {}


def _build_metrics(final_state: Any, initial_value: float,
                   trajectory: Optional[Dict]) -> Dict:
    state = _as_dict(final_state)
    traj  = trajectory if isinstance(trajectory, dict) else {}

    portfolio_now = float(state.get("portfolio_value") or initial_value)
    ph_raw = traj.get("portfolio_history")
    ph = (
        [float(v) for v in ph_raw]
        if isinstance(ph_raw, list) and len(ph_raw) >= 2
        else [float(initial_value), portfolio_now]
    )

    actions = traj.get("action_history", [])
    steps   = traj.get("step_records",   [])

    returns    = compute_returns(ph) or [0.001]
    growth     = (ph[-1] - ph[0]) / max(abs(ph[0]), 1e-9)
    drawdown   = compute_drawdown(ph) or 0.001
    volatility = compute_volatility(returns) or 0.001
    trade_count = sum(1 for a in actions if a not in (0, 8))
    trade_eff   = float(compute_trade_efficiency(actions, ph)) or 0.50
    regime      = float(compute_regime_adaptation(steps)) or 0.50

    return {
        "growth":      growth,
        "drawdown":    drawdown,
        "volatility":  volatility,
        "trade_count": trade_count,
        "trade_eff":   trade_eff,
        "regime":      regime,
    }


# ── Compute full score breakdown for one task ─────────────────────────────────

def compute_task_score(
    task_key: str,
    final_state: Any,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> Dict:
    """
    Compute per-component and total score for one task.
    Mirrors reference project's compute_task_score().
    Every field clamped with format_score(round(clamp_score(x), 4)).
    """
    cfg     = TASK_CONFIGS[task_key]
    scoring = cfg["scoring"]
    targets = cfg["targets"]

    m = _build_metrics(final_state, initial_value, trajectory)

    gs  = grade_growth(m["growth"],      targets["growth"])
    rs  = grade_risk(m["drawdown"],      targets["drawdown_cap"])
    ss  = grade_stability(m["volatility"], targets["vol_cap"])
    qs  = grade_trade_quality(
            m["trade_count"], targets["trade_cap"],
            m["trade_eff"],   m["regime"])

    total = (
        scoring["growth_weight"]    * gs
        + scoring["risk_weight"]    * rs
        + scoring["stability_weight"] * ss
        + scoring["quality_weight"] * qs
    )

    # Mirror reference exactly: format_score(round(clamp_score(x), 4))
    return {
        "growth_score":    format_score(round(clamp_score(gs),    4)),
        "risk_score":      format_score(round(clamp_score(rs),    4)),
        "stability_score": format_score(round(clamp_score(ss),    4)),
        "quality_score":   format_score(round(clamp_score(qs),    4)),
        "total_score":     format_score(round(clamp_score(min(total, 1.0)), 4)),
    }


# ── Public graders ────────────────────────────────────────────────────────────

def _resolve_final_state(candidate: Any) -> Any:
    """
    Normalize grader input so both validator env objects and plain state dicts work.
    """
    try:
        if hasattr(candidate, "get_state"):
            return candidate.get_state()
        if hasattr(candidate, "state"):
            state = candidate.state()
            if hasattr(state, "model_dump"):
                return state.model_dump()
            return state
        if hasattr(candidate, "model_dump"):
            return candidate.model_dump()
        if isinstance(candidate, dict):
            return candidate
    except Exception:
        pass
    return {}


def _grade_task(
    task_key: str,
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    try:
        bd = compute_task_score(task_key, _resolve_final_state(final_state), initial_value, trajectory)
        return safe(bd["total_score"])
    except Exception:
        return _EPS * 5   # 0.05


class Task1Grader:
    def grade(self, env, *args, **kwargs) -> float:
        final_state = kwargs.pop("final_state", env)
        initial_value = kwargs.pop("initial_value", args[0] if args else 1000.0)
        trajectory = kwargs.pop("trajectory", args[1] if len(args) > 1 else None)
        return _grade_task("task1", _resolve_final_state(final_state), initial_value, trajectory)


class Task2Grader:
    def grade(self, env, *args, **kwargs) -> float:
        final_state = kwargs.pop("final_state", env)
        initial_value = kwargs.pop("initial_value", args[0] if args else 1000.0)
        trajectory = kwargs.pop("trajectory", args[1] if len(args) > 1 else None)
        return _grade_task("task2", _resolve_final_state(final_state), initial_value, trajectory)


class Task3Grader:
    def grade(self, env, *args, **kwargs) -> float:
        final_state = kwargs.pop("final_state", env)
        initial_value = kwargs.pop("initial_value", args[0] if args else 1000.0)
        trajectory = kwargs.pop("trajectory", args[1] if len(args) > 1 else None)
        return _grade_task("task3", _resolve_final_state(final_state), initial_value, trajectory)


def grade_task1(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Capital Preservation. Returns float in (0.01, 0.99)."""
    return _grade_task("task1", final_state, initial_value, trajectory)


def grade_task2(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Balanced Growth. Returns float in (0.01, 0.99)."""
    return _grade_task("task2", final_state, initial_value, trajectory)


def grade_task3(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> float:
    """Aggressive Optimization. Returns float in (0.01, 0.99)."""
    return _grade_task("task3", final_state, initial_value, trajectory)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def run_all_tasks(
    final_state: Any = None,
    initial_value: float = 1000.0,
    trajectory: Optional[Dict] = None,
) -> Dict:
    """Run all 3 graders independently. Each failure falls back to 0.05."""
    try:
        s1 = grade_task1(final_state, initial_value, trajectory)
    except Exception:
        s1 = _EPS * 5

    try:
        s2 = grade_task2(final_state, initial_value, trajectory)
    except Exception:
        s2 = _EPS * 5

    try:
        s3 = grade_task3(final_state, initial_value, trajectory)
    except Exception:
        s3 = _EPS * 5

    overall = safe((s1 + s2 + s3) / 3.0)

    return {
        "task1":         s1,
        "task2":         s2,
        "task3":         s3,
        "overall_score": overall,
        # Backward-compat aliases
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

TASK_GRADER_CLASSES = {
    "task1": Task1Grader,
    "task2": Task2Grader,
    "task3": Task3Grader,
}

TASK_REGISTRY = {
    "task1": {
        "name": "Capital Preservation",
        "description": "Minimize drawdown and protect capital over 30 steps.",
        "difficulty": "easy",
        "grader": grade_task1,
    },
    "task2": {
        "name": "Balanced Growth",
        "description": "Achieve stable returns while maintaining diversification.",
        "difficulty": "medium",
        "grader": grade_task2,
    },
    "task3": {
        "name": "Aggressive Optimization",
        "description": "Maximize returns while controlling volatility.",
        "difficulty": "hard",
        "grader": grade_task3,
    },
}
