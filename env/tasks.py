"""
Trajectory-aware task graders for FinLearn Tutor.
"""

from __future__ import annotations

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
    """Clamp to strictly (0.01, 0.99) and round to 2dp. Enforces 0 < score < 1."""
    return round(max(0.01, min(0.99, float(score))), 2)


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

def _as_state_dict(final_state: Observation | Dict) -> Dict:
    return final_state.model_dump() if isinstance(final_state, Observation) else final_state


def build_episode_context(
    final_state: Observation | Dict,
    initial_value: float = 1000.0,
    trajectory: Dict | None = None,
) -> Dict:
    state = _as_state_dict(final_state)
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    portfolio_history = trajectory.get("portfolio_history", [initial_value, state["portfolio_value"]])
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
    metrics = build_episode_context(final_state, initial_value=initial_value, trajectory=trajectory)
    weights = weights or TASK_CONFIGS["task2_balanced_growth"]["weights"]
    targets = targets or TASK_CONFIGS["task2_balanced_growth"]["targets"]

    growth_score = normalize_growth(metrics["growth"], targets["growth"])
    risk_control_score = normalize_inverse(metrics["drawdown"], targets["drawdown_cap"])
    stability_score = normalize_inverse(metrics["volatility"], targets["vol_cap"])
    trade_score = normalize_inverse(metrics["trade_count"], targets["trade_cap"])
    decision_quality_score = clamp_score((metrics["decision_quality"] * 0.7) + (trade_score * 0.3))

    score = clamp_score(
        growth_score * weights["growth"]
        + risk_control_score * weights["risk_control"]
        + stability_score * weights["stability"]
        + decision_quality_score * weights["decision_quality"]
    )

    return {
        "score": _safe_score(score),
        "growth_score": _safe_score(growth_score),
        "risk_control_score": _safe_score(risk_control_score),
        "stability_score": _safe_score(stability_score),
        "decision_quality_score": _safe_score(decision_quality_score),
        "portfolio_growth": round(metrics["growth"], 4),
        "maximum_drawdown": round(metrics["drawdown"], 4),
        "portfolio_volatility": round(metrics["volatility"], 4),
        "trade_count": metrics["trade_count"],
        "trade_efficiency": round(metrics["trade_efficiency"], 4),
        "regime_adaptation": round(metrics["regime_adaptation"], 4),
    }


def grade_task1(final_state: Observation | Dict, initial_value: float = 1000.0, trajectory: Dict | None = None) -> float:
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


def grade_task2(final_state: Observation | Dict, initial_value: float = 1000.0, trajectory: Dict | None = None) -> float:
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


def grade_task3(final_state: Observation | Dict, initial_value: float = 1000.0, trajectory: Dict | None = None) -> float:
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


def run_all_tasks(final_state: Observation | Dict, initial_value: float = 1000.0, trajectory: Dict | None = None) -> Dict:
    preservation = score_trajectory(
        final_state,
        initial_value=initial_value,
        trajectory=trajectory,
        weights=TASK_CONFIGS["task1_capital_preservation"]["weights"],
        targets=TASK_CONFIGS["task1_capital_preservation"]["targets"],
    )
    balanced = score_trajectory(
        final_state,
        initial_value=initial_value,
        trajectory=trajectory,
        weights=TASK_CONFIGS["task2_balanced_growth"]["weights"],
        targets=TASK_CONFIGS["task2_balanced_growth"]["targets"],
    )
    aggressive = score_trajectory(
        final_state,
        initial_value=initial_value,
        trajectory=trajectory,
        weights=TASK_CONFIGS["task3_aggressive_optimization"]["weights"],
        targets=TASK_CONFIGS["task3_aggressive_optimization"]["targets"],
    )
    overall = _safe_score((preservation["score"] + balanced["score"] + aggressive["score"]) / 3)
    return {
        "task1_capital_preservation": _safe_score(preservation["score"]),
        "task2_balanced_growth": _safe_score(balanced["score"]),
        "task3_aggressive_optimization": _safe_score(aggressive["score"]),
        "overall_score": overall,
        "benchmark_breakdown": {
            "capital_preservation": preservation,
            "balanced_growth": balanced,
            "aggressive_optimization": aggressive,
        },
    }


TASKS = {
    "task1_capital_preservation": grade_task1,
    "task2_balanced_growth": grade_task2,
    "task3_aggressive_optimization": grade_task3,
}
