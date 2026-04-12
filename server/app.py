"""
OpenEnv-compliant FastAPI server for FinLearn Tutor.
Exposes: /health, /tasks, /reset, /run, /grade, /api/simulation
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import FinLearnEnv
from env.models import Action
from env.tasks import TASK_REGISTRY, run_all_tasks

app = FastAPI(title="FinLearn Tutor API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, FinLearnEnv] = {}
_completed_sessions: Dict[str, dict] = {}
_env_lock = Lock()

TASKS = [
    {
        "task_id": task_id,
        "name": meta["name"],
        "description": meta["description"],
        "difficulty": meta["difficulty"],
    }
    for task_id, meta in TASK_REGISTRY.items()
]


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "finlearn-tutor"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "finlearn-tutor", "version": "1.1.0"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": TASKS}


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: Optional[int] = 42
    max_steps: Optional[int] = 30


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    max_steps = req.max_steps if req.max_steps is not None else 30
    seed = req.seed if req.seed is not None else 42
    env = FinLearnEnv(max_steps=max_steps, seed=seed)
    with _env_lock:
        _envs[req.task_id] = env
        _completed_sessions.pop(req.task_id, None)

    observation = env.state()
    return {
        "task_id": req.task_id,
        "observation": observation.model_dump(),
        "done": False,
        "info": {"step": 0, "max_steps": max_steps},
        "config": {
            "seed": seed,
            "max_steps": max_steps,
            "initial_cash": 1000.0,
            "trade_amount": 100.0,
        },
    }


class RunRequest(BaseModel):
    task_id: str = "task1"
    action: Optional[int] = None
    action_id: Optional[int] = None

    def resolved_action(self) -> int:
        if self.action is not None:
            return self.action
        if self.action_id is not None:
            return self.action_id
        raise ValueError("action is required")


@app.post("/run")
def run_step(req: RunRequest) -> dict:
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    with _env_lock:
        env = _envs.get(req.task_id)

    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task_id '{req.task_id}'. Call /reset first.",
        )

    try:
        action = req.resolved_action()
        observation, reward, done, info = env.step(Action(action_id=action))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    score = None
    if done:
        grader = TASK_REGISTRY[req.task_id]["grader"]
        result = grader(observation, trajectory=env.get_episode_summary())
        score = round(max(0.0, min(float(result), 1.0)), 4)
        with _env_lock:
            _completed_sessions[req.task_id] = {
                "observation": observation.model_dump(),
                "trajectory": env.get_episode_summary(),
            }
            _envs.pop(req.task_id, None)

    return {
        "task_id": req.task_id,
        "observation": observation.model_dump(),
        "reward": reward.value,
        "done": done,
        "score": score,
        "info": info,
    }


@app.post("/grade")
def grade(req: ResetRequest) -> dict:
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    with _env_lock:
        env = _envs.get(req.task_id)
        completed = _completed_sessions.get(req.task_id)

    observation = None
    trajectory = None
    if env is not None:
        observation = env.state()
        trajectory = env.get_episode_summary()
    elif completed is not None:
        observation = completed["observation"]
        trajectory = completed["trajectory"]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"No active or completed session for task_id '{req.task_id}'. Call /reset first.",
        )

    grader = TASK_REGISTRY[req.task_id]["grader"]
    score = max(0.0, min(1.0, float(grader(observation, trajectory=trajectory))))
    return {"task_id": req.task_id, "score": round(score, 4)}


def _choose_dashboard_action(state: dict) -> int:
    trends = state["trends"]
    volatility = state["volatility"]
    holdings = state["holdings"]
    prices = state["prices"]
    cash = state["cash_balance"]
    market_regime = state.get("market_regime", "sideways")
    risk_level = state.get("risk_level", "moderate")
    market_event = state.get("market_event", "none")

    stock_values = {symbol: holdings[symbol] * prices[symbol] for symbol in holdings}
    portfolio_value = state["portfolio_value"]

    if risk_level == "high" or market_event == "market_crash":
        for symbol, trend in trends.items():
            if trend < 0 and holdings.get(symbol, 0) > 0:
                return {"ALPHA": 4, "BETA": 5, "GAMMA": 6}[symbol]
        return 7 if sum(stock_values.values()) > 0 else 0

    if portfolio_value > 0 and sum(stock_values.values()) > 0:
        for symbol, value in stock_values.items():
            if value / portfolio_value > 0.70:
                return 7

    for symbol in ["ALPHA", "GAMMA", "BETA"]:
        trend = trends[symbol]
        vol = volatility[symbol]
        if trend > 0.002 and vol <= 0.035 and cash > 50:
            weight = stock_values.get(symbol, 0) / portfolio_value if portfolio_value > 0 else 0
            if weight < 0.50:
                return {"ALPHA": 1, "BETA": 2, "GAMMA": 3}[symbol]

    if market_regime == "bear":
        for symbol in ["ALPHA", "GAMMA", "BETA"]:
            if trends[symbol] < 0 and holdings.get(symbol, 0) > 0:
                return {"ALPHA": 4, "BETA": 5, "GAMMA": 6}[symbol]
        return 0

    for symbol, trend in trends.items():
        if trend < -0.002 and holdings.get(symbol, 0) > 0:
            return {"ALPHA": 4, "BETA": 5, "GAMMA": 6}[symbol]

    best_buy = None
    best_score = float("-inf")
    for symbol, trend in trends.items():
        vol = volatility[symbol]
        if vol > 0.035:
            continue
        if trend > 0.002:
            score = trend - vol
            if score > best_score:
                best_score = score
                best_buy = symbol

    if best_buy and cash > 50:
        return {"ALPHA": 1, "BETA": 2, "GAMMA": 3}[best_buy]

    return 0


@app.get("/api/simulation")
def api_simulation(max_steps: int = 20, seed: int = 42) -> dict:
    env = FinLearnEnv(max_steps=max_steps, seed=seed)
    observation = env.reset()
    initial_value = observation.portfolio_value

    steps: list[dict] = []
    done = False
    while not done:
        action_id = _choose_dashboard_action(observation.model_dump())
        observation, reward, done, info = env.step(Action(action_id=action_id))
        steps.append(
            {
                "step": observation.step,
                "action": Action(action_id=action_id).name,
                "reward": reward.value,
                "portfolio_value": observation.portfolio_value,
                "cash_balance": observation.cash_balance,
                "holdings": observation.holdings,
                "prices": observation.prices,
                "volatility": observation.volatility,
                "learning_score": observation.learning_score,
                "reasoning": info.get("reasoning", ""),
                "concept": info.get("insight", ""),
                "suggestion": info.get("suggestion", ""),
                "market_regime": observation.market_regime,
                "max_drawdown": observation.max_drawdown,
                "concentration_score": observation.concentration_score,
                "portfolio_volatility": observation.portfolio_volatility,
            }
        )

    task_scores = run_all_tasks(
        final_state=observation,
        initial_value=initial_value,
        trajectory=env.get_episode_summary(),
    )

    return {
        "initial_value": initial_value,
        "steps": steps,
        "task_scores": task_scores,
        "final_state": {
            "market_regime": observation.market_regime,
            "max_drawdown": observation.max_drawdown,
            "concentration_score": observation.concentration_score,
        },
    }
