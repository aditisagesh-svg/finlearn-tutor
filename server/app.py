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

from env.environment import FinLearnEnv
from env.market import STOCKS
from env.tasks import run_all_tasks
from inference import choose_action, build_openai_client, ping_llm_proxy

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
    def _fallback_observation() -> dict:
        return {
            "cash_balance": 1000.0,
            "holdings": {stock: 0 for stock in STOCKS},
            "prices": {"ALPHA": 100.0, "BETA": 150.0, "GAMMA": 80.0},
            "trends": {"ALPHA": 0.0, "BETA": 0.0, "GAMMA": 0.0},
            "volatility": {"ALPHA": 0.02, "BETA": 0.02, "GAMMA": 0.02},
            "portfolio_value": 1000.0,
            "step": 0,
            "learning_score": 0.0,
            "risk_appetite": "medium",
            "investment_horizon": "long",
            "goal": "balanced_growth",
            "investor_profile": "balanced",
            "market_regime": "sideways",
            "market_event": "none",
            "external_signal": {"signal": "Market is waiting for a catalyst", "impact": "neutral", "sector": "all"},
            "portfolio_volatility": 0.0,
            "concentration_score": 0.0,
            "max_drawdown": 0.0,
            "risk_level": "moderate",
            "reasoning_hint": "Market conditions are neutral; begin with disciplined, profile-aligned decisions.",
            "last_action_feedback": "Episode reset. Build a strategy that matches the investor profile and market regime.",
        }

    try:
        global env
        env = FinLearnEnv(
            max_steps=getattr(env, "max_steps", 30),
            seed=getattr(env, "seed", 42),
        )
        api_base_url = os.getenv("API_BASE_URL")
        api_key = os.getenv("API_KEY")
        if api_base_url and api_key:
            try:
                ping_llm_proxy(build_openai_client())
            except Exception:
                # Reset must still succeed even if the proxy ping fails.
                pass
        observation = env.state()
        return {
            "observation": observation.model_dump(),
            "done": False,
        }
    except Exception:
        try:
            env = FinLearnEnv(max_steps=30, seed=42)
            observation = env.state()
            return {
                "observation": observation.model_dump(),
                "done": False,
            }
        except Exception:
            return {
                "observation": _fallback_observation(),
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
    port = int(os.environ["PORT"])
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
