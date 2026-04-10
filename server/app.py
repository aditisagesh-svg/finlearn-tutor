"""
Minimal HTTP server for Hugging Face Space validator compatibility.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

from env.environment import FinLearnEnv
from env.tasks import run_all_tasks
from env.models import Observation
from inference import choose_action

app = FastAPI(title="FinLearn Tutor API")
env = FinLearnEnv()
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIST), name="frontend-static")

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


def _render_frontend_index() -> HTMLResponse | FileResponse | dict:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        return {"status": "ok", "service": "finlearn-tutor"}

    html = index_file.read_text(encoding="utf-8")
    bootstrap = simulation(max_steps=20, seed=42)
    payload = json.dumps(bootstrap)
    script = f'<script>window.__FINLEARN_BOOTSTRAP__ = {payload};</script>'
    return HTMLResponse(html.replace("</head>", f"{script}</head>"))


@app.get("/")
def healthcheck():
    return _render_frontend_index()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "finlearn-tutor"}


@app.get("/favicon.ico")
def favicon():
    path = FRONTEND_DIST / "favicon.ico"
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/robots.txt")
def robots():
    path = FRONTEND_DIST / "robots.txt"
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/placeholder.svg")
def placeholder():
    path = FRONTEND_DIST / "placeholder.svg"
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.post("/reset")
def reset() -> dict:
    global env
    env = FinLearnEnv(
        max_steps=getattr(env, "max_steps", 30),
        seed=getattr(env, "seed", 42),
    )
    observation = env.state()
    return {
        "observation": observation.model_dump(),
        "done": False,
    }


@app.get("/api/simulation")
def simulation(max_steps: int = 20, seed: int = 42) -> dict:
    dashboard_env = FinLearnEnv(max_steps=max_steps, seed=seed)
    observation = dashboard_env.reset()
    initial_value = observation.portfolio_value

    steps = []
    done = False

    while not done:
        action = choose_action(observation.model_dump())
        observation, reward, done, info = dashboard_env.step(action)

        steps.append(
            {
                "step": observation.step,
                "action": info["action"],
                "reward": reward.value,
                "portfolio_value": observation.portfolio_value,
                "cash_balance": observation.cash_balance,
                "holdings": observation.holdings,
                "prices": observation.prices,
                "trends": observation.trends,
                "volatility": observation.volatility,
                "learning_score": observation.learning_score,
                "reasoning": info["reasoning"],
                "concept": info["insight"],
                "suggestion": info["suggestion"],
                "market_regime": observation.market_regime,
                "market_event": observation.market_event,
                "risk_level": observation.risk_level,
                "reasoning_hint": observation.reasoning_hint,
                "last_action_feedback": observation.last_action_feedback,
                "external_signal": observation.external_signal,
                "max_drawdown": observation.max_drawdown,
                "concentration_score": observation.concentration_score,
                "portfolio_volatility": observation.portfolio_volatility,
            }
        )

    task_scores = run_all_tasks(
        observation,
        initial_value,
        trajectory=dashboard_env.get_episode_summary(),
    )

    return {
        "initial_value": initial_value,
        "steps": steps,
        "task_scores": task_scores,
        "final_state": observation.model_dump(),
        "trajectory": dashboard_env.get_episode_summary(),
    }


@app.post("/run")
def run() -> dict:
    result = simulation()
    task_scores = result.get("task_scores", {})
    return {
        "task_scores": {
            "capital_preservation": float(task_scores.get("capital_preservation", 0.5)),
            "balanced_growth": float(task_scores.get("balanced_growth", 0.5)),
            "aggressive_optimization": float(task_scores.get("aggressive_optimization", 0.5)),
        }
    }


@app.post("/simulate")
def simulate() -> dict:
    return run()


@app.get("/tasks")
def get_tasks() -> list:
    """
    Validator discovery endpoint.
    Must return ≥ 3 tasks with score_range strictly inside (0, 1).
    """
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "Capital Preservation",
                "difficulty": "easy",
                "score_range": [0.01, 0.99],
            },
            {
                "id": "task2",
                "name": "Balanced Growth",
                "difficulty": "medium",
                "score_range": [0.01, 0.99],
            },
            {
                "id": "task3",
                "name": "Aggressive Optimization",
                "difficulty": "hard",
                "score_range": [0.01, 0.99],
            },
        ]
    }


class GraderRequest(BaseModel):
    task_id: str
    observation: dict
    initial_value: float = 1000.0
    trajectory: dict = {}


@app.post("/grader")
async def grader(request: GraderRequest):
    """
    Validator grading endpoint.
    Accepts {task_id, observation, initial_value, trajectory}
    Returns {task_id, score} with score strictly in (0.01, 0.99).
    """
    import math

    def _safe(x: float) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return 0.50
            return round(max(0.01, min(0.99, v)), 2)
        except Exception:
            return 0.50

    from env.tasks import grade_task1, grade_task2, grade_task3

    graders = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
    }
    fn = graders.get(request.task_id)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {request.task_id}")

    obs = Observation(**request.observation)
    score = _safe(fn(obs))
    return {"task_id": request.task_id, "score": score}


@app.get("/{full_path:path}")
def frontend_fallback(full_path: str):
    requested_file = FRONTEND_DIST / full_path
    if full_path and requested_file.exists() and requested_file.is_file():
        return FileResponse(requested_file)

    response = _render_frontend_index()
    if isinstance(response, dict):
        raise HTTPException(status_code=404, detail="Frontend not built")
    return response


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
