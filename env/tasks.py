"""
Trajectory-aware task graders for FinLearn Tutor.
This file is loaded by the OpenEnv validator as: env.tasks
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


# ── Safety clamp: enforces 0 < score < 1 strictly ────────────────────────────

def _safe_score(score: float) -> float:
    try:
        value = float(score)
        if math.isnan(value) or math.isinf(value):
            return 0.50
        return round(max(0.01, min(0.99, value)), 2)
    except Exception:
        return 0.50


# ── Task configuration ────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task1_capital_preservation": {
        "label": "Capital Preservation",
        "weights": {"growth": 0.20, "risk_control": 0.35, "stability": 0.25, "decision_quality": 0.20},
        "targets": {"growth": 0.04, "drawdown_cap": 0.10, "vol_cap": 0.025, "trade_cap": 8},
    },
    "task2_balanced_growth": {
        "label": "Balanced Growth",
        "weights": {"growth": 0.40, "risk_control": 0.20, "stability": 0.20, "decision_quality": 0.20},
        "targets": {"growth": 0.10, "drawdown_cap": 0.18, "vol_cap": 0.035, "trade_cap": 12},
    },
    "task3_aggressive_optimization": {
        "label": "Aggressive Optimization",
        "weights": {"growth": 0.50, "risk_control": 0.15, "stability": 0.10, "decision_quality": 0.25},
        "targets": {"growth": 0.18, "drawdown_cap": 0.28, "vol_cap": 0.05, "trade_cap": 16},
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _as_state_dict(final_state: Observation | Dict) -> Dict:
    return final_state.model_dump() if isinstance(final_state, Observation) else final_state


def build_episode_context(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> Dict:
    state = _as_state_dict(final_state)
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    portfolio_history = trajectory.get("portfolio_history")
    if not portfolio_history:
        portfolio_history = [initial_value, state["portfolio_value"]]
    actions = trajectory.get("action_history", [])
    steps = trajectory.get("step_records", [])

    returns = compute_returns(portfolio_history)
    growth = (portfolio_history[-1] - portfolio_history[0]) / max(portfolio_history[0], 1e-9)
    drawdown = compute_drawdown(portfolio_history)
    volatility = compute_volatility(returns)
    trade_count = sum(1 for action in actions if action not in (0, 8))
    trade_efficiency = compute_trade_efficiency(actions, portfolio_history)
    regime_adaptation = compute_regime_adaptation(steps)

    return {
        "growth": growth,
        "drawdown": drawdown,
        "volatility": volatility,
        "trade_count": trade_count,
        "trade_efficiency": trade_efficiency,
        "regime_adaptation": regime_adaptation,
        "returns": returns,
        "portfolio_history": portfolio_history,
        "decision_quality": clamp_score((trade_efficiency + regime_adaptation) / 2.0),
    }


def score_trajectory(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
    weights: Dict[str, float] | None = None,
    targets: Dict[str, float] | None = None,
) -> Dict[str, float]:
    metrics = build_episode_context(
        final_state, initial_value=initial_value, trajectory=trajectory
    )
    weights = weights or TASK_CONFIGS["task2_balanced_growth"]["weights"]
    targets = targets or TASK_CONFIGS["task2_balanced_growth"]["targets"]

    growth_score        = normalize_growth(metrics["growth"], targets["growth"])
    risk_control_score  = normalize_inverse(metrics["drawdown"], targets["drawdown_cap"])
    stability_score     = normalize_inverse(metrics["volatility"], targets["vol_cap"])
    trade_score         = normalize_inverse(metrics["trade_count"], targets["trade_cap"])
    decision_quality_score = clamp_score(
        (metrics["decision_quality"] * 0.7) + (trade_score * 0.3)
    )

    raw = clamp_score(
        growth_score       * weights["growth"]
        + risk_control_score * weights["risk_control"]
        + stability_score    * weights["stability"]
        + decision_quality_score * weights["decision_quality"]
    )

    return {
        "score":                 _safe_score(raw),
        "growth_score":          _safe_score(growth_score),
        "risk_control_score":    _safe_score(risk_control_score),
        "stability_score":       _safe_score(stability_score),
        "decision_quality_score": _safe_score(decision_quality_score),
        "portfolio_growth":      round(metrics["growth"], 4),
        "maximum_drawdown":      round(metrics["drawdown"], 4),
        "portfolio_volatility":  round(metrics["volatility"], 4),
        "trade_count":           metrics["trade_count"],
        "trade_efficiency":      round(metrics["trade_efficiency"], 4),
        "regime_adaptation":     round(metrics["regime_adaptation"], 4),
    }


# ── Public graders (imported by validator as env.tasks:grade_taskN) ───────────

def grade_task1(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> float:
    """Capital Preservation — score strictly in (0.01, 0.99)."""
    try:
        raw = score_trajectory(
            final_state,
            initial_value=initial_value,
            trajectory=trajectory,
            weights=TASK_CONFIGS["task1_capital_preservation"]["weights"],
            targets=TASK_CONFIGS["task1_capital_preservation"]["targets"],
        )["score"]
        return _safe_score(raw)
    except Exception:
        return 0.05


def grade_task2(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> float:
    """Balanced Growth — score strictly in (0.01, 0.99)."""
    try:
        raw = score_trajectory(
            final_state,
            initial_value=initial_value,
            trajectory=trajectory,
            weights=TASK_CONFIGS["task2_balanced_growth"]["weights"],
            targets=TASK_CONFIGS["task2_balanced_growth"]["targets"],
        )["score"]
        return _safe_score(raw)
    except Exception:
        return 0.05


def grade_task3(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> float:
    """Aggressive Optimization — score strictly in (0.01, 0.99)."""
    try:
        raw = score_trajectory(
            final_state,
            initial_value=initial_value,
            trajectory=trajectory,
            weights=TASK_CONFIGS["task3_aggressive_optimization"]["weights"],
            targets=TASK_CONFIGS["task3_aggressive_optimization"]["targets"],
        )["score"]
        return _safe_score(raw)
    except Exception:
        return 0.05


# ── Aggregate runner — output keys must match validator's expected structure ──

def run_all_tasks(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> Dict:
    """
    Returns a flat dict with top-level score keys that the validator reads directly.

    Required structure (validator checks these exact top-level keys):
        {
            "capital_preservation":    float,   # task1 score
            "balanced_growth":         float,   # task2 score
            "aggressive_optimization": float,   # task3 score
            "overall_score":           float,
        }
    """
    try:
        t1 = score_trajectory(
            final_state, initial_value=initial_value, trajectory=trajectory,
            weights=TASK_CONFIGS["task1_capital_preservation"]["weights"],
            targets=TASK_CONFIGS["task1_capital_preservation"]["targets"],
        )
    except Exception:
        t1 = {"score": 0.05}

    try:
        t2 = score_trajectory(
            final_state, initial_value=initial_value, trajectory=trajectory,
            weights=TASK_CONFIGS["task2_balanced_growth"]["weights"],
            targets=TASK_CONFIGS["task2_balanced_growth"]["targets"],
        )
    except Exception:
        t2 = {"score": 0.05}

    try:
        t3 = score_trajectory(
            final_state, initial_value=initial_value, trajectory=trajectory,
            weights=TASK_CONFIGS["task3_aggressive_optimization"]["weights"],
            targets=TASK_CONFIGS["task3_aggressive_optimization"]["targets"],
        )
    except Exception:
        t3 = {"score": 0.05}

    s1 = _safe_score(t1["score"])
    s2 = _safe_score(t2["score"])
    s3 = _safe_score(t3["score"])
    overall = _safe_score((s1 + s2 + s3) / 3.0)

    return {
        # ── Flat top-level keys the validator reads ──────────────────────────
        "capital_preservation":    s1,
        "balanced_growth":         s2,
        "aggressive_optimization": s3,
        "overall_score":           overall,
        # ── Legacy keys kept for inference.py compatibility ──────────────────
        "task1_capital_preservation": s1,
        "task2_balanced_growth":      s2,
        "task3_aggressive_optimization": s3,
    }


# ── Task registry (used by any framework that discovers graders via TASKS) ────

TASKS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}