"""
Minimal HTTP server for Hugging Face Space validator compatibility.
"""

from fastapi import FastAPI

from env.environment import FinLearnEnv

app = FastAPI(title="FinLearn Tutor API")
env = FinLearnEnv()


@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok", "service": "finlearn-tutor"}


@app.post("/reset")
def reset() -> dict:
    global env
    env = FinLearnEnv(max_steps=env.max_steps, seed=env.seed)
    observation = env.state()
    return {
        "observation": observation.model_dump(),
        "done": False,
    }
