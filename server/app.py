"""
OpenEnv-compliant FastAPI server for FinLearn Tutor.
Exposes: /health, /tasks, /reset, /run
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import FinLearnEnv
from env.models import Action
from env.tasks import TASK_REGISTRY

app = FastAPI(title="FinLearn Tutor API")
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
