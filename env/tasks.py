"""
Trajectory-aware task graders for FinLearn Tutor.
Loaded by validator as: env.tasks
"""
from __future__ import annotations
import math
from typing import Any, Callable, Dict

from env.metrics import (
    clamp_score, compute_drawdown, compute_regime_adaptation,
    compute_returns, compute_trade_efficiency, compute_volatility,
    normalize_growth, normalize_inverse,
)
from env.models import Observation


def _safe_score(score: float) -> float:
    try:
        v = float(score)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        return round(max(0.01, min(0.99, v)), 2)
    except Exception:
        return 0.50


TASK_CONFIGS = {
    "task1_capital_preservation": {
        "weights": {"growth": 0.20, "risk_control": 0.35, "stability": 0.25, "decision_quality": 0.20},
        "targets": {"growth": 0.04, "drawdown_cap": 0.10, "vol_cap": 0.025, "trade_cap": 8},
    },
    "task2_balanced_growth": {
        "weights": {"growth": 0.40, "risk_control": 0.20, "stability": 0.20, "decision_quality": 0.20},
        "targets": {"growth": 0.10, "drawdown_cap": 0.18, "vol_cap": 0.035, "trade_cap": 12},
    },
    "task3_aggressive_optimization": {
        "weights": {"growth": 0.50, "risk_control": 0.15, "stability": 0.10, "decision_quality": 0.25},
        "targets": {"growth": 0.18, "drawdown_cap": 0.28, "vol_cap": 0.05, "trade_cap": 16},
    },
}


def _as_state_dict(final_state):
    return final_state.model_dump() if isinstance(final_state, Observation) else final_state


def build_episode_context(final_state, initial_value=1000.0, trajectory=None):
    state = _as_state_dict(final_state)
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    portfolio_history = trajectory.get("portfolio_history") or [initial_value, state.get("portfolio_value", initial_value)]
    actions = trajectory.get("action_history", [])
    steps = trajectory.get("step_records", [])
    returns = compute_returns(portfolio_history)
    growth = (portfolio_history[-1] - portfolio_history[0]) / max(portfolio_history[0], 1e-9)
    return {
        "growth": growth,
        "drawdown": compute_drawdown(portfolio_history),
        "volatility": compute_volatility(returns),
        "trade_count": sum(1 for a in actions if a not in (0, 8)),
        "trade_efficiency": compute_trade_efficiency(actions, portfolio_history),
        "regime_adaptation": compute_regime_adaptation(steps),
        "decision_quality": clamp_score((compute_trade_efficiency(actions, portfolio_history) + compute_regime_adaptation(steps)) / 2.0),
    }


def score_trajectory(final_state, initial_value=1000.0, trajectory=None, weights=None, targets=None):
    metrics = build_episode_context(final_state, initial_value=initial_value, trajectory=trajectory)
    weights = weights or TASK_CONFIGS["task2_balanced_growth"]["weights"]
    targets = targets or TASK_CONFIGS["task2_balanced_growth"]["targets"]
    growth_score       = normalize_growth(metrics["growth"], targets["growth"])
    risk_control_score = normalize_inverse(metrics["drawdown"], targets["drawdown_cap"])
    stability_score    = normalize_inverse(metrics["volatility"], targets["vol_cap"])
    trade_score        = normalize_inverse(metrics["trade_count"], targets["trade_cap"])
    dq_score           = clamp_score((metrics["decision_quality"] * 0.7) + (trade_score * 0.3))
    raw = clamp_score(
        growth_score       * weights["growth"]
        + risk_control_score * weights["risk_control"]
        + stability_score    * weights["stability"]
        + dq_score           * weights["decision_quality"]
    )
    return {"score": _safe_score(raw)}


def grade_task1(final_state, initial_value=1000.0, trajectory=None) -> float:
    try:
        return _safe_score(score_trajectory(final_state, initial_value, trajectory,
            weights=TASK_CONFIGS["task1_capital_preservation"]["weights"],
            targets=TASK_CONFIGS["task1_capital_preservation"]["targets"])["score"])
    except Exception:
        return 0.05


def grade_task2(final_state, initial_value=1000.0, trajectory=None) -> float:
    try:
        return _safe_score(score_trajectory(final_state, initial_value, trajectory,
            weights=TASK_CONFIGS["task2_balanced_growth"]["weights"],
            targets=TASK_CONFIGS["task2_balanced_growth"]["targets"])["score"])
    except Exception:
        return 0.05


def grade_task3(final_state, initial_value=1000.0, trajectory=None) -> float:
    try:
        return _safe_score(score_trajectory(final_state, initial_value, trajectory,
            weights=TASK_CONFIGS["task3_aggressive_optimization"]["weights"],
            targets=TASK_CONFIGS["task3_aggressive_optimization"]["targets"])["score"])
    except Exception:
        return 0.05


def run_all_tasks(final_state, initial_value=1000.0, trajectory=None) -> Dict:
    s1 = grade_task1(final_state, initial_value, trajectory)
    s2 = grade_task2(final_state, initial_value, trajectory)
    s3 = grade_task3(final_state, initial_value, trajectory)
    overall = _safe_score((s1 + s2 + s3) / 3.0)
    return {
        "capital_preservation":    s1,
        "balanced_growth":         s2,
        "aggressive_optimization": s3,
        "overall_score":           overall,
        "task1_capital_preservation":    s1,
        "task2_balanced_growth":         s2,
        "task3_aggressive_optimization": s3,
    }




class Task:
    """Task definition for FinLearn validator."""

    def __init__(self, name: str, grader: Callable[[Any, Any, Any], float]):
        self.name = name
        self.grader = grader


TASKS = [
    Task(name="task1_capital_preservation", grader=grade_task1),
    Task(name="task2_balanced_growth", grader=grade_task2),
    Task(name="task3_aggressive_optimization", grader=grade_task3),
]
