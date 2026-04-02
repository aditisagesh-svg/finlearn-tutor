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
    env = FinLearnEnv(
        max_steps=getattr(env, "max_steps", 30),
        seed=getattr(env, "seed", 42),
    )
    observation = env.state()
    return {
        "observation": observation.model_dump(),
        "done": False,
    }
