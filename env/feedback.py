"""
feedback.py — Explainable AI feedback generator for FinLearn Tutor
"""

from typing import Dict


ACTION_NAMES = {
    0: "HOLD",
    1: "BUY_ALPHA",
    2: "BUY_BETA",
    3: "BUY_GAMMA",
    4: "SELL_ALPHA",
    5: "SELL_BETA",
    6: "SELL_GAMMA",
    7: "REBALANCE",
    8: "REQUEST_HINT",
}

STOCK_MAP = {
    1: "ALPHA", 2: "BETA", 3: "GAMMA",
    4: "ALPHA", 5: "BETA", 6: "GAMMA",
}


def generate_feedback(
    action: int,
    state: Dict,
    reward: float,
) -> Dict[str, str]:
    """
    Returns structured explainable feedback for the taken action.
    """
    action_name = ACTION_NAMES.get(action, "UNKNOWN")
    trends = state["trends"]
    volatility = state["volatility"]
    holdings = state["holdings"]
    prices = state["prices"]
    cash = state["cash_balance"]

    reasoning = _build_reasoning(action, trends, volatility, holdings, prices, cash, reward)
    insight = _build_insight(action, trends, volatility)
    suggestion = _build_suggestion(state)

    return {
        "action":     action_name,
        "reasoning":  reasoning,
        "insight":    insight,
        "suggestion": suggestion,
    }


def _build_reasoning(action, trends, volatility, holdings, prices, cash, reward):
    stock = STOCK_MAP.get(action)

    if action == 0:  # HOLD
        return (
            "You chose to hold your position. "
            f"Current reward: {reward:+.2f}. "
            "Holding is rational when market signals are mixed or when avoiding unnecessary transaction costs."
        )

    if action in (1, 2, 3):  # BUY
        t = trends[stock]
        v = volatility[stock]
        if t > 0.003:
            quality = "a strong positive trend"
        elif t > 0:
            quality = "a mildly positive trend"
        else:
            quality = "a negative or flat trend — this may be premature"
        return (
            f"You bought {stock} (trend={t:+.4f}, volatility={v:.4f}). "
            f"{stock} is showing {quality}. "
            f"Reward: {reward:+.2f}."
        )

    if action in (4, 5, 6):  # SELL
        t = trends[stock]
        qty = holdings.get(stock, 0)
        if qty == 0:
            return f"You tried to sell {stock} but held none. No trade executed."
        if t < -0.002:
            quality = "a negative trend — selling is a reasonable exit"
        else:
            quality = "a positive trend — selling here may forgo future gains"
        return (
            f"You sold {stock} (trend={t:+.4f}, qty held={qty}). "
            f"{stock} shows {quality}. "
            f"Reward: {reward:+.2f}."
        )

    if action == 7:  # REBALANCE
        return (
            "You rebalanced your portfolio by selling the most concentrated position "
            "and moving funds to cash. Rebalancing helps manage risk by avoiding over-exposure."
        )

    if action == 8:  # HINT
        return "You requested a hint. A tutor tip has been generated based on current market conditions."

    return "Action executed."


def _build_insight(action, trends, volatility):
    concepts = {
        0: "📘 Concept: Patience — Sometimes the best trade is no trade. "
           "Avoiding unnecessary fees and emotional decisions is a hallmark of disciplined investing.",
        1: "📘 Concept: Momentum Investing — Buying assets showing upward price trends, "
           "expecting the trend to continue in the short run.",
        2: "📘 Concept: Momentum Investing — same principle applies to BETA.",
        3: "📘 Concept: Momentum Investing — same principle applies to GAMMA.",
        4: "📘 Concept: Stop-Loss / Profit Taking — Selling to exit a deteriorating position "
           "or lock in gains before a reversal.",
        5: "📘 Concept: Stop-Loss / Profit Taking — same principle for BETA.",
        6: "📘 Concept: Stop-Loss / Profit Taking — same principle for GAMMA.",
        7: "📘 Concept: Portfolio Rebalancing — Periodically resetting asset weights to target "
           "allocations reduces drift risk and enforces buy-low/sell-high discipline.",
        8: "📘 Concept: Guided Learning — Requesting hints simulates consulting a financial advisor "
           "before making a decision. Use hints wisely; over-reliance limits learning.",
    }
    return concepts.get(action, "📘 No concept available.")


def _build_suggestion(state):
    trends = state["trends"]
    volatility = state["volatility"]
    holdings = state["holdings"]
    cash = state["cash_balance"]

    tips = []

    # Suggest buying trending stocks if cash available
    for stock, t in trends.items():
        v = volatility[stock]
        if t > 0.004 and v < 0.03 and cash > 50:
            tips.append(f"Consider buying {stock} — positive trend ({t:+.4f}) with manageable risk.")

    # Suggest selling declining stocks
    for stock, t in trends.items():
        if t < -0.004 and holdings.get(stock, 0) > 0:
            tips.append(f"Consider selling {stock} — negative trend ({t:+.4f}), limit downside.")

    # Suggest rebalancing if concentrated
    stock_values = {s: holdings[s] * state["prices"][s] for s in holdings}
    total = sum(stock_values.values())
    if total > 0:
        for s, v in stock_values.items():
            if v / total > 0.70:
                tips.append(f"Portfolio is concentrated in {s} ({v/total:.0%}). Consider rebalancing.")

    if not tips:
        tips.append("Market signals are mixed. HOLD and monitor trends for another step.")

    return " | ".join(tips)
