"""
market.py — Synthetic stock market simulation for FinLearn Tutor
"""

import random
from typing import List, Dict


STOCKS = ["ALPHA", "BETA", "GAMMA"]

INITIAL_PRICES = {
    "ALPHA": 100.0,
    "BETA":  150.0,
    "GAMMA": 80.0,
}

INITIAL_TRENDS = {
    "ALPHA":  0.005,
    "BETA":  -0.003,
    "GAMMA":  0.008,
}

INITIAL_VOLATILITY = {
    "ALPHA": 0.02,
    "BETA":  0.04,
    "GAMMA": 0.015,
}


class Market:
    """Simulates a simple 3-stock synthetic market."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.prices: Dict[str, float] = dict(INITIAL_PRICES)
        self.trends: Dict[str, float] = dict(INITIAL_TRENDS)
        self.volatility: Dict[str, float] = dict(INITIAL_VOLATILITY)
        self.step_count: int = 0

    def step(self) -> Dict[str, float]:
        """Advance market by one tick. Returns new prices."""
        for stock in STOCKS:
            noise = self.rng.gauss(0, self.volatility[stock])
            self.prices[stock] = round(
                self.prices[stock] * (1 + self.trends[stock] + noise), 2
            )
            # Slowly drift trend to simulate regime changes
            self.trends[stock] += self.rng.gauss(0, 0.001)
            self.trends[stock] = max(-0.02, min(0.02, self.trends[stock]))

        self.step_count += 1
        return dict(self.prices)

    def get_snapshot(self) -> Dict:
        return {
            "prices":     dict(self.prices),
            "trends":     dict(self.trends),
            "volatility": dict(self.volatility),
        }
