"""
market.py — Synthetic stock market simulation for FinLearn Tutor

UPGRADED: Market regime system (bull / bear / crash / sideways)
          Deterministic regime selection based on seed.
          All existing public signatures preserved.
"""

import random
from typing import Dict


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

# ── Regime definitions ────────────────────────────────────────────────────────
# Each regime modifies trend drift and noise scale deterministically.
REGIME_PARAMS = {
    "bull":     {"trend_bias": +0.004, "vol_scale": 0.8,  "crash_step": None},
    "bear":     {"trend_bias": -0.004, "vol_scale": 1.0,  "crash_step": None},
    "crash":    {"trend_bias": -0.002, "vol_scale": 2.5,  "crash_step": 3},
    "sideways": {"trend_bias":  0.000, "vol_scale": 0.5,  "crash_step": None},
}

_REGIMES = ["bull", "bear", "sideways", "crash", "bull", "sideways", "bear", "bull"]


def _regime_for_seed(seed: int) -> str:
    """Deterministically pick a regime from the seed."""
    return _REGIMES[seed % len(_REGIMES)]


class Market:
    """Simulates a 3-stock synthetic market with deterministic regime behaviour."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        # ── NEW: regime ──────────────────────────────────────────────────────
        self.regime: str = _regime_for_seed(seed)
        self._regime_params = REGIME_PARAMS[self.regime]
        self.reset()

    def reset(self):
        self.prices: Dict[str, float] = dict(INITIAL_PRICES)
        self.trends: Dict[str, float] = dict(INITIAL_TRENDS)
        self.volatility: Dict[str, float] = dict(INITIAL_VOLATILITY)
        self.step_count: int = 0

    def step(self) -> Dict[str, float]:
        """Advance market by one tick. Returns new prices."""
        bias      = self._regime_params["trend_bias"]
        vol_scale = self._regime_params["vol_scale"]
        crash_at  = self._regime_params["crash_step"]

        for stock in STOCKS:
            eff_vol = self.volatility[stock] * vol_scale
            noise   = self.rng.gauss(0, eff_vol)

            # Crash regime: apply a sharp drop at the designated step
            if crash_at is not None and self.step_count == crash_at:
                crash_shock = -0.08  # deterministic 8% crash shock
                noise += crash_shock

            effective_trend = self.trends[stock] + bias
            self.prices[stock] = round(
                self.prices[stock] * (1 + effective_trend + noise), 2
            )

            # Trend drift (regime-biased)
            self.trends[stock] += self.rng.gauss(0, 0.001) + bias * 0.1
            self.trends[stock] = max(-0.02, min(0.02, self.trends[stock]))

        self.step_count += 1
        return dict(self.prices)

    def get_snapshot(self) -> Dict:
        return {
            "prices":     dict(self.prices),
            "trends":     dict(self.trends),
            "volatility": dict(self.volatility),
        }