"""
rewards.py — Reward function for FinLearn Tutor

UPGRADED:
  - Profile alignment reward / penalty
  - Transaction cost deduction per trade
  - Risk-awareness penalty (portfolio volatility for low-risk users)
  All existing logic preserved; new terms are purely additive.
"""

from typing import Dict

from env.models import Reward

TRANSACTION_COST = 1.0   # flat fee per executed trade


def calculate_reward(
    prev_portfolio_value: float,
    curr_portfolio_value: float,
    holdings: Dict[str, int],
    prices: Dict[str, float],
    action: int,
    trade_count: int,
    # ── NEW optional kwargs (backward-compatible) ────────────────────────────
    volatility: Dict[str, float] = None,
    profile: Dict = None,
    trade_executed: bool = False,
) -> Reward:
    """
    Reward = portfolio growth + diversification bonus
             - concentration penalty - overtrading penalty
             - transaction cost (if trade executed)
             - profile alignment penalty
             - risk-awareness penalty
    """
    # ── 1. Portfolio growth (unchanged) ──────────────────────────────────────
    growth = (curr_portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1)
    growth_reward = growth * 10.0

    # ── 2. Diversification bonus (unchanged) ─────────────────────────────────
    stocks_held = sum(1 for qty in holdings.values() if qty > 0)
    diversification_bonus = stocks_held * 0.05

    # ── 3. Concentration penalty (unchanged) ─────────────────────────────────
    stock_values = {s: holdings[s] * prices[s] for s in holdings}
    total_stock_value = sum(stock_values.values())
    concentration_penalty = 0.0
    if total_stock_value > 0:
        for val in stock_values.values():
            weight = val / total_stock_value
            if weight > 0.70:
                concentration_penalty = 0.2

    # ── 4. Overtrading penalty (unchanged) ───────────────────────────────────
    overtrading_penalty = 0.05 if trade_count > 10 else 0.0

    # ── 5. NEW: Transaction cost ──────────────────────────────────────────────
    tx_cost = TRANSACTION_COST / max(prev_portfolio_value, 1) if trade_executed else 0.0

    # ── 6. NEW: Profile alignment penalty ────────────────────────────────────
    profile_penalty = 0.0
    if profile is not None and volatility is not None:
        risk_appetite = profile.get("risk_appetite", "medium")
        horizon       = profile.get("investment_horizon", "long")
        goal          = profile.get("goal", "wealth_growth")

        # Weighted portfolio volatility
        total_pv = curr_portfolio_value
        if total_pv > 0 and total_stock_value > 0:
            port_vol = sum(
                (holdings[s] * prices[s] / total_pv) * volatility.get(s, 0.02)
                for s in holdings
            )
        else:
            port_vol = 0.0

        # a) Low-risk user + high portfolio volatility → penalty
        if risk_appetite == "low" and port_vol > 0.025:
            profile_penalty += 0.10

        # b) Long-horizon user + overtrading → penalty
        if horizon == "long" and trade_count > 8:
            profile_penalty += 0.05

        # c) Capital preservation goal + losses → extra penalty
        if goal == "capital_preservation" and growth < -0.005:
            profile_penalty += 0.15

    # ── Assemble final reward ────────────────────────────────────────────────
    reward_value = (
        growth_reward
        + diversification_bonus
        - concentration_penalty
        - overtrading_penalty
        - tx_cost
        - profile_penalty
    )
    return Reward(
        value=round(reward_value, 4),
        portfolio_growth=round(growth_reward, 4),
        diversification_bonus=round(diversification_bonus, 4),
        concentration_penalty=round(concentration_penalty, 4),
        overtrading_penalty=round(overtrading_penalty, 4),
        transaction_cost=round(tx_cost, 4),
        profile_penalty=round(profile_penalty, 4),
    )
