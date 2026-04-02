"""
Typed models for the FinLearn OpenEnv interface.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionType(IntEnum):
    HOLD = 0
    BUY_ALPHA = 1
    BUY_BETA = 2
    BUY_GAMMA = 3
    SELL_ALPHA = 4
    SELL_BETA = 5
    SELL_GAMMA = 6
    REBALANCE = 7
    REQUEST_HINT = 8


ACTION_NAMES = {action.value: action.name for action in ActionType}


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    cash_balance: float
    holdings: Dict[str, int]
    prices: Dict[str, float]
    trends: Dict[str, float]
    volatility: Dict[str, float]
    portfolio_value: float
    step: int
    learning_score: float = Field(ge=0.0, le=1.0)
    risk_appetite: str | None = None
    investment_horizon: str | None = None
    goal: str | None = None
    market_regime: str | None = None
    portfolio_volatility: float | None = None
    concentration_score: float | None = None
    max_drawdown: float | None = None


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: int = Field(ge=0, le=8)

    @field_validator("action_id")
    @classmethod
    def validate_action_id(cls, value: int) -> int:
        if value not in ACTION_NAMES:
            raise ValueError(f"Unsupported action_id: {value}")
        return value

    @property
    def name(self) -> str:
        return ACTION_NAMES[self.action_id]


class Reward(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float
    portfolio_growth: float
    diversification_bonus: float
    concentration_penalty: float
    overtrading_penalty: float
    transaction_cost: float = 0.0
    profile_penalty: float = 0.0


class StepInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: str
    reasoning: str
    insight: str
    suggestion: str
