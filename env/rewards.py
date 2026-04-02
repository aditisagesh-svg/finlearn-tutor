"""
rewards.py — Reward function for FinLearn Tutor
"""

from typing import Dict

from env.models import Reward


def calculate_reward(
    prev_portfolio_value: float,
    curr_portfolio_value: float,
    holdings: Dict[str, int],
    prices: Dict[str, float],
    action: int,
    trade_count: int,
) -> Reward:
    """
    Reward = portfolio growth + diversification bonus
             - risk penalty - overtrading penalty
    """
    # 1. Portfolio growth
    growth = (curr_portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1)
    growth_reward = growth * 10.0

    # 2. Diversification bonus — reward holding multiple stocks
    stocks_held = sum(1 for qty in holdings.values() if qty > 0)
    diversification_bonus = stocks_held * 0.05

    # 3. Risk penalty — penalise when one stock is >70% of portfolio
    stock_values = {s: holdings[s] * prices[s] for s in holdings}
    total_stock_value = sum(stock_values.values())
    concentration_penalty = 0.0
    if total_stock_value > 0:
        for val in stock_values.values():
            weight = val / total_stock_value
            if weight > 0.70:
                concentration_penalty = 0.2

    # 4. Overtrading penalty
    overtrading_penalty = 0.05 if trade_count > 10 else 0.0

    reward_value = growth_reward + diversification_bonus - concentration_penalty - overtrading_penalty
    return Reward(
        value=round(reward_value, 4),
        portfolio_growth=round(growth_reward, 4),
        diversification_bonus=round(diversification_bonus, 4),
        concentration_penalty=round(concentration_penalty, 4),
        overtrading_penalty=round(overtrading_penalty, 4),
    )
