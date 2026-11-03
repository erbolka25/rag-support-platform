"""Entry point: uvicorn support_copilot.main:app"""

from __future__ import annotations

import logging

from support_copilot.api.app import create_app
from support_copilot.config import get_settings
from support_copilot.rag import build_rag_service

settings = get_settings()
logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
app = create_app(settings, build_rag_service(settings))


def run() -> None:
    import uvicorn

    uvicorn.run("support_copilot.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
