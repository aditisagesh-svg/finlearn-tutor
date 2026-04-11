"""
Compatibility wrapper that re-exports the canonical FastAPI app.

Keep all HTTP logic in ``server/app.py`` so the deployment has one source of
truth. This file only exists for older entrypoints that still import
``server.py`` directly.
"""

from server.app import app


def main() -> None:
    import os

    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
