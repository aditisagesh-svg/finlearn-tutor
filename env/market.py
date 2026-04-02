"""
Deterministic market simulator with regimes, macro events, and external signals.
"""

from __future__ import annotations

import random
from typing import Dict


STOCKS = ["ALPHA", "BETA", "GAMMA"]
ASSET_SECTORS = {
    "ALPHA": "tech",
    "BETA": "defensive",
    "GAMMA": "commodities",
}

INITIAL_PRICES = {
    "ALPHA": 100.0,
    "BETA": 150.0,
    "GAMMA": 80.0,
}

INITIAL_TRENDS = {
    "ALPHA": 0.0045,
    "BETA": 0.0025,
    "GAMMA": 0.0035,
}

INITIAL_VOLATILITY = {
    "ALPHA": 0.022,
    "BETA": 0.014,
    "GAMMA": 0.018,
}

REGIME_PARAMS = {
    "bull": {"trend_bias": 0.0035, "vol_scale": 0.85, "asset_bias": {"ALPHA": 0.0015, "BETA": 0.0005, "GAMMA": 0.001}},
    "bear": {"trend_bias": -0.0045, "vol_scale": 1.15, "asset_bias": {"ALPHA": -0.001, "BETA": 0.0008, "GAMMA": -0.0002}},
    "sideways": {"trend_bias": 0.0, "vol_scale": 0.75, "asset_bias": {"ALPHA": 0.0, "BETA": 0.0002, "GAMMA": 0.0}},
    "high_volatility": {"trend_bias": -0.001, "vol_scale": 1.7, "asset_bias": {"ALPHA": -0.0005, "BETA": 0.0002, "GAMMA": 0.0004}},
}

EVENT_TEMPLATES = {
    "interest_rate_hike": {
        "duration": 4,
        "trend_shift": {"ALPHA": -0.006, "BETA": 0.003, "GAMMA": -0.001},
        "vol_shift": {"ALPHA": 0.012, "BETA": 0.005, "GAMMA": 0.006},
        "signals": [
            {"signal": "Central bank raises rates, growth assets face pressure", "impact": "negative", "sector": "tech"},
            {"signal": "Income-producing assets look steadier after the hike", "impact": "positive", "sector": "defensive"},
        ],
    },
    "market_crash": {
        "duration": 3,
        "trend_shift": {"ALPHA": -0.012, "BETA": -0.007, "GAMMA": -0.005},
        "vol_shift": {"ALPHA": 0.02, "BETA": 0.016, "GAMMA": 0.014},
        "signals": [
            {"signal": "Risk assets are repricing sharply lower", "impact": "negative", "sector": "tech"},
            {"signal": "Capital preservation matters more than chasing upside", "impact": "negative", "sector": "all"},
        ],
    },
    "tech_bubble": {
        "duration": 5,
        "trend_shift": {"ALPHA": 0.009, "BETA": -0.001, "GAMMA": 0.001},
        "vol_shift": {"ALPHA": 0.018, "BETA": 0.003, "GAMMA": 0.004},
        "signals": [
            {"signal": "Tech leadership is strong but valuations are stretched", "impact": "mixed", "sector": "tech"},
            {"signal": "Momentum is rewarding concentrated bets in growth sectors", "impact": "positive", "sector": "tech"},
        ],
    },
    "inflation_spike": {
        "duration": 4,
        "trend_shift": {"ALPHA": -0.003, "BETA": -0.001, "GAMMA": 0.007},
        "vol_shift": {"ALPHA": 0.008, "BETA": 0.006, "GAMMA": 0.01},
        "signals": [
            {"signal": "Inflation expectations are rising, real assets may benefit", "impact": "positive", "sector": "commodities"},
            {"signal": "High input costs are slowing tech sector earnings", "impact": "negative", "sector": "tech"},
        ],
    },
}


class Market:
    """Simulates a deterministic 3-asset market with evolving context."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._regime_cycle = self._build_regime_cycle(seed)
        self._event_schedule = self._build_event_schedule(seed)
        self.reset()

    def reset(self):
        self.prices: Dict[str, float] = dict(INITIAL_PRICES)
        self.trends: Dict[str, float] = dict(INITIAL_TRENDS)
        self.volatility: Dict[str, float] = dict(INITIAL_VOLATILITY)
        self.step_count = 0
        self.regime = "sideways"
        self.market_event = "none"
        self.external_signal = {"signal": "Market is waiting for a catalyst", "impact": "neutral", "sector": "all"}
        self.risk_level = "moderate"
        self._refresh_context()

    def _build_regime_cycle(self, seed: int) -> list[str]:
        base_cycle = [
            "bull",
            "bull",
            "sideways",
            "high_volatility",
            "bear",
            "bear",
            "sideways",
            "bull",
        ]
        offset = seed % len(base_cycle)
        return base_cycle[offset:] + base_cycle[:offset]

    def _build_event_schedule(self, seed: int) -> Dict[int, str]:
        event_order = [
            "interest_rate_hike",
            "tech_bubble",
            "inflation_spike",
            "market_crash",
        ]
        offset = seed % len(event_order)
        rotated = event_order[offset:] + event_order[:offset]
        starts = [2, 7, 13, 19]
        return {start: rotated[idx] for idx, start in enumerate(starts)}

    def _active_event(self, step: int) -> str | None:
        for start, event_name in self._event_schedule.items():
            duration = EVENT_TEMPLATES[event_name]["duration"]
            if start <= step < start + duration:
                return event_name
        return None

    def _current_signal(self, step: int, event_name: str | None) -> Dict[str, str]:
        if event_name:
            event = EVENT_TEMPLATES[event_name]
            signal_idx = min(step - next(start for start, name in self._event_schedule.items() if name == event_name), len(event["signals"]) - 1)
            return dict(event["signals"][signal_idx])

        neutral_signals = [
            {"signal": "Breadth is improving across risky assets", "impact": "positive", "sector": "all"},
            {"signal": "Leadership is narrow, diversification still matters", "impact": "mixed", "sector": "all"},
            {"signal": "Markets are range-bound and selective", "impact": "neutral", "sector": "all"},
            {"signal": "Volatility is elevated, position sizing matters", "impact": "negative", "sector": "all"},
        ]
        return dict(neutral_signals[step % len(neutral_signals)])

    def _refresh_context(self) -> None:
        self.regime = self._regime_cycle[self.step_count % len(self._regime_cycle)]
        active_event = self._active_event(self.step_count)
        self.market_event = active_event or "none"
        self.external_signal = self._current_signal(self.step_count, active_event)

        avg_vol = sum(self.volatility.values()) / len(self.volatility)
        if self.regime == "high_volatility" or self.market_event == "market_crash" or avg_vol >= 0.04:
            self.risk_level = "high"
        elif self.regime == "bull" and avg_vol < 0.025:
            self.risk_level = "low"
        else:
            self.risk_level = "moderate"

    def step(self) -> Dict[str, float]:
        regime_params = REGIME_PARAMS[self.regime]
        event_params = EVENT_TEMPLATES.get(self.market_event)

        for stock in STOCKS:
            trend_bias = regime_params["trend_bias"] + regime_params["asset_bias"].get(stock, 0.0)
            vol_scale = regime_params["vol_scale"]
            event_trend = event_params["trend_shift"].get(stock, 0.0) if event_params else 0.0
            event_vol = event_params["vol_shift"].get(stock, 0.0) if event_params else 0.0

            current_vol = max(0.006, INITIAL_VOLATILITY[stock] * vol_scale + event_vol)
            self.volatility[stock] = round(current_vol, 4)

            shock = self.rng.gauss(0, current_vol)
            if self.market_event == "market_crash":
                shock -= 0.04
            elif self.market_event == "tech_bubble" and stock == "ALPHA":
                shock += 0.008

            effective_trend = self.trends[stock] + trend_bias + event_trend
            new_price = self.prices[stock] * (1 + effective_trend + shock)
            self.prices[stock] = round(max(new_price, 5.0), 2)

            drift = self.rng.gauss(0, 0.0012)
            self.trends[stock] = max(-0.03, min(0.03, self.trends[stock] * 0.55 + trend_bias + event_trend * 0.5 + drift))

        self.step_count += 1
        self._refresh_context()
        return dict(self.prices)

    def get_snapshot(self) -> Dict:
        return {
            "prices": dict(self.prices),
            "trends": dict(self.trends),
            "volatility": dict(self.volatility),
            "market_event": self.market_event,
            "external_signal": dict(self.external_signal),
            "risk_level": self.risk_level,
            "market_regime": self.regime,
        }
