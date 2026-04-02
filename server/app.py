"""
Minimal HTTP server for Hugging Face Space validator compatibility.
"""

from fastapi import FastAPI
import uvicorn

from env.environment import FinLearnEnv

app = FastAPI(title="FinLearn Tutor API")
env = FinLearnEnv()


@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok", "service": "finlearn-tutor"}


@app.post("/reset")
def reset() -> dict:
    observation = env.reset()
    return {
        "observation": observation.model_dump(),
        "done": False,
    }


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
