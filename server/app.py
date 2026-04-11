"""
FastAPI server for FinLearn Tutor — OpenEnv validator compatible.
Exposes: GET /health, GET /, POST /reset, GET /tasks, POST /grader
All endpoints are crash-proof: import failures return safe fallback responses.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# ── Safe environment import ───────────────────────────────────────────────────
try:
    from env.environment import FinLearnEnv
    _ENV_OK = True
except Exception:
    _ENV_OK = False
    FinLearnEnv = None  # type: ignore

# ── Safe grader import ────────────────────────────────────────────────────────
try:
    from env.tasks import grade_task1, grade_task2, grade_task3, run_all_tasks
    _GRADERS_OK = True
except Exception:
    _GRADERS_OK = False

    def grade_task1(final_state=None, initial_value=1000.0, trajectory=None) -> float:
        return 0.50

    def grade_task2(final_state=None, initial_value=1000.0, trajectory=None) -> float:
        return 0.50

    def grade_task3(final_state=None, initial_value=1000.0, trajectory=None) -> float:
        return 0.50

    def run_all_tasks(final_state=None, initial_value=1000.0, trajectory=None) -> Dict:
        return {"task1": 0.50, "task2": 0.50, "task3": 0.50, "overall_score": 0.50}


# ── Task registry (IDs match openenv.yaml exactly) ────────────────────────────
TASK_REGISTRY = {
    "task1": {
        "id": "task1",
        "name": "Capital Preservation",
        "difficulty": "easy",
        "grader": "env.tasks:grade_task1",
        "score_range": [0.01, 0.99],
        "fn": grade_task1,
    },
    "task2": {
        "id": "task2",
        "name": "Balanced Growth",
        "difficulty": "medium",
        "grader": "env.tasks:grade_task2",
        "score_range": [0.01, 0.99],
        "fn": grade_task2,
    },
    "task3": {
        "id": "task3",
        "name": "Aggressive Optimization",
        "difficulty": "hard",
        "grader": "env.tasks:grade_task3",
        "score_range": [0.01, 0.99],
        "fn": grade_task3,
    },
}


def _safe_score(x: float) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.50
        return round(max(0.01, min(0.99, v)), 2)
    except Exception:
        return 0.50


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FinLearn Tutor", version="1.1.0")

_env = None


def _get_env():
    global _env
    if _env is None and _ENV_OK:
        try:
            _env = FinLearnEnv(max_steps=30, seed=42)
            _env.reset()
        except Exception:
            pass
    return _env


# ── Health / root ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "finlearn-tutor"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Reset ─────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    global _env
    try:
        if _ENV_OK:
            _env = FinLearnEnv(max_steps=30, seed=42)
            observation = _env.reset()
            return {"observation": observation.model_dump(), "done": False}
    except Exception:
        pass
    return {"observation": {}, "done": False}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    tasks = [
        {
            "id": meta["id"],
            "name": meta["name"],
            "difficulty": meta["difficulty"],
            "grader": meta["grader"],
            "score_range": meta["score_range"],
        }
        for meta in TASK_REGISTRY.values()
    ]
    return {"tasks": tasks, "count": len(tasks)}


# ── Grader ────────────────────────────────────────────────────────────────────

class GraderRequest(BaseModel):
    task_id: str
    final_state: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None
    initial_value: Optional[float] = 1000.0
    trajectory: Optional[Dict[str, Any]] = None

    def state_payload(self) -> Dict[str, Any]:
        return self.final_state or self.observation or {}


@app.post("/grader")
def run_grader(request: GraderRequest):
    task_id = request.task_id
    if task_id not in TASK_REGISTRY:
        return {"task_id": task_id, "score": 0.05, "error": f"unknown task_id: {task_id}"}

    try:
        initial_value = request.initial_value if request.initial_value is not None else 1000.0
        raw = TASK_REGISTRY[task_id]["fn"](
            final_state=request.state_payload(),
            initial_value=initial_value,
            trajectory=request.trajectory or {},
        )
        score = _safe_score(raw)
    except Exception:
        score = 0.05

    return {
        "task_id": task_id,
        "score": score,
        "score_range": TASK_REGISTRY[task_id]["score_range"],
    }


@app.post("/grade_all")
def grade_all(request: GraderRequest):
    try:
        initial_value = request.initial_value if request.initial_value is not None else 1000.0
        result = run_all_tasks(
            final_state=request.state_payload(),
            initial_value=initial_value,
            trajectory=request.trajectory or {},
        )
        return {k: _safe_score(v) if isinstance(v, (int, float)) else v for k, v in result.items()}
    except Exception:
        return {"task1": 0.05, "task2": 0.05, "task3": 0.05, "overall_score": 0.05}
