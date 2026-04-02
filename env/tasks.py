"""
tasks.py — Task graders for FinLearn Tutor

Each task returns a score from 0.0 to 1.0.
"""

from typing import Dict

from env.models import Observation


def _as_state_dict(final_state: Observation | Dict) -> Dict:
    return final_state.model_dump() if isinstance(final_state, Observation) else final_state


def grade_task1(final_state: Observation | Dict, initial_value: float = 1000.0) -> float:
    """
    TASK 1 (Easy): Avoid Losses
    Score 1.0 if portfolio_value >= initial, scaled down proportionally otherwise.
    """
    state = _as_state_dict(final_state)
    final_value = state["portfolio_value"]
    if final_value >= initial_value:
        return 1.0
    loss_pct = (initial_value - final_value) / initial_value
    score = max(0.0, 1.0 - loss_pct * 5)   # lose 20% → score 0
    return round(score, 4)


def grade_task2(final_state: Observation | Dict) -> float:
    """
    TASK 2 (Medium): Maintain Diversification
    Score based on how evenly holdings are spread across 3 stocks.
    Full score = 1 share of each stock at minimum.
    """
    state = _as_state_dict(final_state)
    holdings = state["holdings"]
    prices = state["prices"]

    stock_values = {s: holdings[s] * prices[s] for s in holdings}
    total = sum(stock_values.values())

    if total == 0:
        return 0.0   # No stocks held at all

    stocks_held = sum(1 for v in stock_values.values() if v > 0)
    if stocks_held == 1:
        base = 0.2
    elif stocks_held == 2:
        base = 0.6
    else:
        base = 1.0

    # Additional bonus for balance (low concentration)
    weights = [v / total for v in stock_values.values()]
    max_weight = max(weights)
    balance_bonus = 1.0 - max_weight  # 0 if all in one, 0.67 if perfectly split

    score = base * 0.7 + balance_bonus * 0.3
    return round(min(1.0, score), 4)


def grade_task3(final_state: Observation | Dict, initial_value: float = 1000.0) -> float:
    """
    TASK 3 (Hard): Maximize Returns with Low Risk
    Reward = growth * (1 - max_concentration_penalty)
    """
    state = _as_state_dict(final_state)
    final_value = state["portfolio_value"]
    growth = (final_value - initial_value) / initial_value  # e.g. 0.15 = 15% gain

    holdings = state["holdings"]
    prices = state["prices"]
    stock_values = {s: holdings[s] * prices[s] for s in holdings}
    total = sum(stock_values.values())

    concentration_penalty = 0.0
    if total > 0:
        max_weight = max(v / total for v in stock_values.values())
        concentration_penalty = max(0.0, max_weight - 0.5)  # penalty if >50% in one stock

    growth_score = min(1.0, max(0.0, growth * 5))         # 20% growth → full score
    risk_multiplier = 1.0 - concentration_penalty

    score = growth_score * risk_multiplier
    return round(score, 4)


def run_all_tasks(final_state: Observation | Dict, initial_value: float = 1000.0) -> Dict:
    """Run all three task graders and return a summary."""
    t1 = grade_task1(final_state, initial_value)
    t2 = grade_task2(final_state)
    t3 = grade_task3(final_state, initial_value)
    overall = round((t1 + t2 + t3) / 3, 4)
    return {
        "task1_avoid_losses":           t1,
        "task2_diversification":        t2,
        "task3_returns_with_low_risk":  t3,
        "overall_score":                overall,
    }
