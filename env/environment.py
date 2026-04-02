"""
environment.py — FinLearn Tutor core RL environment

Enhanced with:
  - user profiles
  - market regimes
  - risk metrics
  - action reasons
  - risk-aware rebalancing
"""

from typing import Dict, Tuple

from env.feedback import generate_feedback
from env.market import Market, STOCKS
from env.models import Action, Observation, Reward
from env.rewards import calculate_reward

INITIAL_CASH = 1000.0
TRADE_AMOUNT = 100.0

_PROFILES = [
    {"risk_appetite": "low", "investment_horizon": "long", "goal": "capital_preservation"},
    {"risk_appetite": "medium", "investment_horizon": "long", "goal": "wealth_growth"},
    {"risk_appetite": "high", "investment_horizon": "short", "goal": "wealth_growth"},
    {"risk_appetite": "low", "investment_horizon": "short", "goal": "capital_preservation"},
    {"risk_appetite": "medium", "investment_horizon": "short", "goal": "wealth_growth"},
    {"risk_appetite": "high", "investment_horizon": "long", "goal": "wealth_growth"},
]


def _profile_for_seed(seed: int) -> Dict[str, str]:
    return dict(_PROFILES[seed % len(_PROFILES)])


class FinLearnEnv:
    BUY_ACTIONS = {1: "ALPHA", 2: "BETA", 3: "GAMMA"}
    SELL_ACTIONS = {4: "ALPHA", 5: "BETA", 6: "GAMMA"}

    def __init__(self, max_steps: int = 30, seed: int = 42):
        self.max_steps = max_steps
        self.seed = seed
        self.market = Market(seed=seed)
        self.profile = _profile_for_seed(seed)
        self.last_hint = ""
        self.reset()

    def reset(self) -> Observation:
        self.market.reset()
        self.cash_balance = INITIAL_CASH
        self.holdings = {s: 0 for s in STOCKS}
        self.step_count = 0
        self.trade_count = 0
        self.learning_score = 0.0
        self.total_reward = 0.0
        self.last_hint = ""
        self._peak_value = INITIAL_CASH
        self._max_drawdown = 0.0
        return self.state()

    def state(self) -> Observation:
        snap = self.market.get_snapshot()
        portfolio_value = self._portfolio_value(snap["prices"])
        portfolio_volatility, concentration_score = self._risk_metrics(
            snap["prices"], snap["volatility"]
        )
        return Observation(
            cash_balance=round(self.cash_balance, 2),
            holdings=dict(self.holdings),
            prices=snap["prices"],
            trends=snap["trends"],
            volatility=snap["volatility"],
            portfolio_value=round(portfolio_value, 2),
            step=self.step_count,
            learning_score=self.learning_score,
            risk_appetite=self.profile["risk_appetite"],
            investment_horizon=self.profile["investment_horizon"],
            goal=self.profile["goal"],
            market_regime=self.market.regime,
            portfolio_volatility=round(portfolio_volatility, 6),
            concentration_score=round(concentration_score, 4),
            max_drawdown=round(self._max_drawdown, 4),
        )

    def get_state(self) -> Dict:
        return self.state().model_dump()

    def step(self, action: Action | int) -> Tuple[Observation, Reward, bool, Dict]:
        normalized_action = action.action_id if isinstance(action, Action) else int(action)
        prices_before = dict(self.market.prices)
        prev_value = self._portfolio_value(prices_before)

        trade_executed = self._execute_action(normalized_action, prices_before)

        new_prices = self.market.step()
        self.step_count += 1
        curr_value = self._portfolio_value(new_prices)

        if curr_value > self._peak_value:
            self._peak_value = curr_value
        drawdown = (self._peak_value - curr_value) / max(self._peak_value, 1)
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        reward = calculate_reward(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            holdings=self.holdings,
            prices=new_prices,
            action=normalized_action,
            trade_count=self.trade_count,
            volatility=self.market.volatility,
            profile=self.profile,
            trade_executed=trade_executed,
        )
        self.total_reward += reward.value
        self.learning_score = round(min(1.0, max(0.0, self.total_reward / 5.0)), 4)

        observation = self.state()
        done = self.step_count >= self.max_steps
        info = generate_feedback(normalized_action, observation.model_dump(), reward.value)
        info["reason"] = _build_reason(normalized_action, self.market, self.profile)
        if self.last_hint:
            info["hint"] = self.last_hint

        return observation, reward, done, info

    def hint(self) -> str:
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
        return self.cash_balance + sum(self.holdings[s] * prices[s] for s in STOCKS)

    def _execute_action(self, action: int, prices: Dict[str, float]) -> bool:
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
                return True
            return False

        if action in self.SELL_ACTIONS:
            stock = self.SELL_ACTIONS[action]
            price = prices[stock]
            shares = min(self.holdings[stock], int(TRADE_AMOUNT / price))
            if shares > 0:
                self.cash_balance += shares * price
                self.holdings[stock] -= shares
                self.trade_count += 1
                return True
            return False

        if action == 7:
            return self._rebalance(prices)

        if action == 8:
            self.last_hint = self.hint()

        return False

    def _rebalance(self, prices: Dict[str, float]) -> bool:
        stock_values = {s: self.holdings[s] * prices[s] for s in STOCKS}
        total_stock = sum(stock_values.values())
        if total_stock == 0:
            return False

        vols = self.market.volatility
        inv_vols = {s: 1.0 / max(vols[s], 0.001) for s in STOCKS}
        inv_vol_sum = sum(inv_vols.values())
        targets = {s: inv_vols[s] / inv_vol_sum for s in STOCKS}
        portfolio_value = self._portfolio_value(prices)
        current_weights = {
            s: stock_values[s] / max(portfolio_value, 1) for s in STOCKS
        }
        overweight = {s: current_weights[s] - targets[s] for s in STOCKS}
        top_stock = max(overweight, key=lambda s: overweight[s])

        if overweight[top_stock] <= 0:
            return False

        sell_shares = self.holdings[top_stock] // 2
        if sell_shares > 0:
            self.cash_balance += sell_shares * prices[top_stock]
            self.holdings[top_stock] -= sell_shares
            self.trade_count += 1
            return True
        return False

    def _risk_metrics(self, prices: Dict[str, float], volatility: Dict[str, float]) -> Tuple[float, float]:
        portfolio_value = self._portfolio_value(prices)
        if portfolio_value == 0:
            return 0.0, 0.0

        stock_values = {s: self.holdings[s] * prices[s] for s in STOCKS}
        total_stock = sum(stock_values.values())

        portfolio_volatility = 0.0
        if total_stock > 0:
            portfolio_volatility = sum(
                (stock_values[s] / portfolio_value) * volatility.get(s, 0.02)
                for s in STOCKS
            )

        concentration_score = 0.0
        if total_stock > 0:
            concentration_score = max(stock_values[s] / total_stock for s in STOCKS)

        return portfolio_volatility, concentration_score


_BUY_STOCK = {1: "ALPHA", 2: "BETA", 3: "GAMMA"}
_SELL_STOCK = {4: "ALPHA", 5: "BETA", 6: "GAMMA"}


def _build_reason(action: int, market: Market, profile: Dict[str, str]) -> str:
    trends = market.trends
    regime = market.regime

    if action in _BUY_STOCK:
        stock = _BUY_STOCK[action]
        trend = trends[stock]
        direction = "upward" if trend > 0 else "flat/downward"
        return f"Buying {stock} due to {direction} trend ({trend:+.4f}) in a {regime} market regime."

    if action in _SELL_STOCK:
        stock = _SELL_STOCK[action]
        trend = trends[stock]
        return f"Selling {stock} to reduce exposure (trend={trend:+.4f}, regime={regime})."

    if action == 7:
        risk_appetite = profile.get("risk_appetite", "medium")
        return f"Rebalancing toward risk-aware equal-weight allocation (profile: {risk_appetite} risk)."

    if action == 8:
        return "Requesting hint — consulting AI tutor for market guidance."

    return f"Holding due to uncertain market conditions (regime={regime}, mixed signals)."
