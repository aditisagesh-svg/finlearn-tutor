"""
OpenEnv-compliant FastAPI server for FinLearn Tutor.
Exposes: /health, /tasks, /reset, /run
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import FinLearnEnv
from env.models import Action
from env.tasks import grade_task1, grade_task2, grade_task3

app = FastAPI(title="FinLearn Tutor API")

_envs: Dict[str, FinLearnEnv] = {}
_env_lock = Lock()

TASKS = [
    {
        "task_id": "task1",
        "name": "Capital Preservation",
        "difficulty": "easy",
        "description": "Protect capital through drawdowns. Minimize max drawdown and volatility.",
        "max_steps": 30,
    },
    {
        "task_id": "task2",
        "name": "Balanced Growth",
        "difficulty": "medium",
        "description": "Achieve stable growth with diversification across all three assets.",
        "max_steps": 30,
    },
    {
        "task_id": "task3",
        "name": "Aggressive Optimization",
        "difficulty": "hard",
        "description": "Maximize portfolio returns while keeping drawdown below threshold.",
        "max_steps": 30,
    },
]

GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


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
    if req.task_id not in GRADERS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

    max_steps = req.max_steps if req.max_steps is not None else 30
    seed = req.seed if req.seed is not None else 42
    env = FinLearnEnv(max_steps=max_steps, seed=seed)
    with _env_lock:
        _envs[req.task_id] = env

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
    if req.task_id not in GRADERS:
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
        grader = GRADERS[req.task_id]
        result = grader(observation, trajectory=env.get_episode_summary())
        score = round(max(0.0, min(float(result), 1.0)), 4)
        with _env_lock:
            _envs.pop(req.task_id, None)

    return {
        "task_id": req.task_id,
        "observation": observation.model_dump(),
        "reward": reward.value,
        "done": done,
        "score": score,
        "info": info,
    }
