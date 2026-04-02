"""
environment.py — FinLearn Tutor core RL environment
"""

from typing import Dict, Tuple

from env.feedback import generate_feedback
from env.market import Market, STOCKS
from env.models import Action, Observation, Reward
from env.rewards import calculate_reward

INITIAL_CASH = 1000.0
TRADE_AMOUNT = 100.0


class FinLearnEnv:
    """
    RL-style financial learning environment.

    Actions:
        0  = HOLD
        1  = BUY_ALPHA
        2  = BUY_BETA
        3  = BUY_GAMMA
        4  = SELL_ALPHA
        5  = SELL_BETA
        6  = SELL_GAMMA
        7  = REBALANCE
        8  = REQUEST_HINT
    """

    BUY_ACTIONS = {1: "ALPHA", 2: "BETA", 3: "GAMMA"}
    SELL_ACTIONS = {4: "ALPHA", 5: "BETA", 6: "GAMMA"}

    def __init__(self, max_steps: int = 30, seed: int = 42):
        self.max_steps = max_steps
        self.market = Market(seed=seed)
        self.last_hint: str = ""
        self.reset()

    def reset(self) -> Observation:
        """Reset environment to the initial typed observation."""
        self.market.reset()
        self.cash_balance = INITIAL_CASH
        self.holdings = {s: 0 for s in STOCKS}
        self.step_count = 0
        self.trade_count = 0
        self.learning_score = 0.0
        self.total_reward = 0.0
        self.last_hint = ""
        return self.state()

    def state(self) -> Observation:
        """Return the current typed observation."""
        snap = self.market.get_snapshot()
        return Observation(
            cash_balance=round(self.cash_balance, 2),
            holdings=dict(self.holdings),
            prices=snap["prices"],
            trends=snap["trends"],
            volatility=snap["volatility"],
            portfolio_value=round(self._portfolio_value(snap["prices"]), 2),
            step=self.step_count,
            learning_score=self.learning_score,
        )

    def get_state(self) -> Dict:
        """Backward-compatible dict view for legacy callers."""
        return self.state().model_dump()

    def step(self, action: Action | int) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one step.
        Returns: (observation, reward, done, info)
        """
        normalized_action = action.action_id if isinstance(action, Action) else int(action)
        prices_before = dict(self.market.prices)
        prev_value = self._portfolio_value(prices_before)

        self._execute_action(normalized_action, prices_before)

        new_prices = self.market.step()
        self.step_count += 1
        curr_value = self._portfolio_value(new_prices)

        reward = calculate_reward(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            holdings=self.holdings,
            prices=new_prices,
            action=normalized_action,
            trade_count=self.trade_count,
        )
        self.total_reward += reward.value
        self.learning_score = round(min(1.0, max(0.0, self.total_reward / 5.0)), 4)

        observation = self.state()
        done = self.step_count >= self.max_steps
        info = generate_feedback(normalized_action, observation.model_dump(), reward.value)
        if self.last_hint:
            info["hint"] = self.last_hint

        return observation, reward, done, info

    def hint(self) -> str:
        """Return a plain-language hint based on current market state."""
        snap = self.market.get_snapshot()
        lines = ["Tutor Hint:"]
        for stock in STOCKS:
            trend = snap["trends"][stock]
            volatility = snap["volatility"][stock]
            if trend > 0.004 and volatility < 0.03:
                lines.append(f"- {stock}: Strong uptrend with low risk. Consider buying.")
            elif trend > 0:
                lines.append(f"- {stock}: Mild uptrend. Hold or add a small position.")
            elif trend < -0.004:
                lines.append(f"- {stock}: Downtrend. Consider selling or avoiding.")
            else:
                lines.append(f"- {stock}: Flat or mixed signal. Hold.")
        return "\n".join(lines)

    def _portfolio_value(self, prices: Dict[str, float]) -> float:
        stock_value = sum(self.holdings[symbol] * prices[symbol] for symbol in STOCKS)
        return self.cash_balance + stock_value

    def _execute_action(self, action: int, prices: Dict[str, float]) -> None:
        self.last_hint = ""

        if action in self.BUY_ACTIONS:
            stock = self.BUY_ACTIONS[action]
            price = prices[stock]
            shares = int(TRADE_AMOUNT / price)
            cost = shares * price
            if shares > 0 and self.cash_balance >= cost:
                self.cash_balance -= cost
                self.holdings[stock] += shares
                self.trade_count += 1
            return

        if action in self.SELL_ACTIONS:
            stock = self.SELL_ACTIONS[action]
            price = prices[stock]
            shares = min(self.holdings[stock], int(TRADE_AMOUNT / price))
            if shares > 0:
                self.cash_balance += shares * price
                self.holdings[stock] -= shares
                self.trade_count += 1
            return

        if action == 7:
            self._rebalance(prices)
            return

        if action == 8:
            self.last_hint = self.hint()

    def _rebalance(self, prices: Dict[str, float]) -> None:
        """Sell half of the most concentrated holding back to cash."""
        stock_values = {symbol: self.holdings[symbol] * prices[symbol] for symbol in STOCKS}
        if max(stock_values.values()) == 0:
            return
        top_stock = max(stock_values, key=lambda symbol: stock_values[symbol])
        sell_shares = self.holdings[top_stock] // 2
        if sell_shares > 0:
            self.cash_balance += sell_shares * prices[top_stock]
            self.holdings[top_stock] -= sell_shares
            self.trade_count += 1
