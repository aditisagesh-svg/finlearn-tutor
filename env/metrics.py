"""
Trajectory-aware evaluation metrics for FinLearn Tutor.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List


def _strict_score(value: float) -> float:
    return round(max(0.01, min(0.99, float(value))), 2)


def compute_returns(portfolio_history: List[float]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(portfolio_history)):
        prev_value = max(portfolio_history[idx - 1], 1e-9)
        returns.append((portfolio_history[idx] - portfolio_history[idx - 1]) / prev_value)
    return returns


def compute_drawdown(portfolio_history: List[float]) -> float:
    if not portfolio_history:
        return 0.01

    peak = portfolio_history[0]
    max_drawdown = 0.0
    for value in portfolio_history:
        peak = max(peak, value)
        drawdown = (peak - value) / max(peak, 1e-9)
        max_drawdown = max(max_drawdown, drawdown)
    return _strict_score(max_drawdown)


def compute_volatility(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.01
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return _strict_score(sqrt(variance))


def compute_trade_efficiency(actions: List[int], portfolio_history: List[float]) -> float:
    trade_steps = [
        step_idx for step_idx, action in enumerate(actions, start=1) if action not in (0, 8)
    ]
    if not trade_steps:
        return 0.5

    improvements = 0.0
    for step_idx in trade_steps:
        if step_idx >= len(portfolio_history):
            continue
        delta = portfolio_history[step_idx] - portfolio_history[step_idx - 1]
        improvements += 1.0 if delta > 0 else 0.25 if delta == 0 else 0.0
    return _strict_score(min(improvements / len(trade_steps), 1.0))


def compute_regime_adaptation(trajectory: Iterable[Dict]) -> float:
    scores: List[float] = []
    for record in trajectory:
        action = record.get("action_id", 0)
        regime = record.get("regime", "sideways")
        risk_level = record.get("risk_level", "moderate")
        trend_strength = record.get("best_trend", 0.0)
        sold_risk = action in (4, 5, 6, 7)
        added_risk = action in (1, 2, 3)

        if regime == "bull":
            score = 1.0 if added_risk else 0.7 if action == 0 else 0.55
        elif regime == "bear":
            score = 1.0 if sold_risk else 0.65 if action == 0 else 0.3
        elif regime == "high_volatility":
            score = 1.0 if sold_risk else 0.8 if action == 0 else 0.35
        else:
            score = 1.0 if action in (0, 7) else 0.65

        if trend_strength > 0.006 and added_risk and regime != "bear":
            score = min(1.0, score + 0.05)
        if risk_level == "high" and added_risk:
            score = max(0.0, score - 0.1)
        scores.append(score)

    if not scores:
        return 0.5
    return _strict_score(sum(scores) / len(scores))


def normalize_growth(growth: float, target_growth: float) -> float:
    if target_growth <= 0:
        return 0.99 if growth > 0 else 0.01
    return _strict_score(growth / target_growth)


def normalize_inverse(raw_value: float, cap: float) -> float:
    if cap <= 0:
        return 0.99
    return _strict_score(1.0 - (raw_value / cap))


def clamp_score(value: float) -> float:
    return _strict_score(value)
