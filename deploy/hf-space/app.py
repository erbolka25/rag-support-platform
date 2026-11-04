"""Hugging Face Space entry point (Gradio SDK): the copilot's FastAPI app and web UI on :7860.

The Space runs ``python app.py``; uvicorn serves support_copilot.main:app directly, so the
home page is the copilot's own UI (Swagger at /docs). rag_core and the copilot are installed
from requirements.txt at a pinned commit; the demo knowledge base is not part of those
packages, so it is downloaded from the same commit at startup.

Everything runs on CPU. ``spaces`` is deliberately not imported and nothing is decorated with
@spaces.GPU: on ZeroGPU hardware ``import spaces`` patches torch for CUDA and fails with the
CPU-only torch build, and a Space without @spaces.GPU functions starts normally.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "erbolka25/rag-support-platform"
DEMO_SRC = "apps/support_copilot/demo"
DEMO_FILES = [
    "kb/delivery.txt",
    "kb/payment.txt",
    "kb/returns.txt",
    "kb/warranty.txt",
    "shop_tickets.json",
]


def pinned_sha(requirements: Path = HERE / "requirements.txt") -> str:
    """The commit requirements.txt installs the copilot from: one place to bump."""
    found = set(re.findall(rf"{REPO}@([0-9a-f]{{40}})", requirements.read_text(encoding="utf-8")))
    if len(found) != 1:
        raise RuntimeError(f"requirements.txt must pin {REPO} to exactly one commit: {found}")
    return found.pop()


def fetch_demo_data(sha: str, dest: Path, opener=urllib.request.urlopen) -> Path:
    """Download the demo knowledge base and tickets at ``sha`` into ``dest`` (once)."""
    for name in DEMO_FILES:
        target = dest / name
        if target.exists():
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{DEMO_SRC}/{name}"
        with opener(url, timeout=30) as response:
            data = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return dest


def configure(demo_dir: Path, data_dir: Path) -> None:
    """Demo defaults; Space secrets (OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL) are already in
    the environment, and any variable set in the Space settings wins."""
    defaults = {
        "DATA_DIR": str(data_dir),
        "COMPANY_NAME": "TechNomad",
        "LLM_PROVIDER": "openai",
        "LLM_MAX_TOKENS": "1500",
        "RATE_LIMIT_PER_HOUR": "20",
        "DEMO_KB_DIR": str(demo_dir / "kb"),
        "DEMO_TICKETS_FILE": str(demo_dir / "shop_tickets.json"),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    configure(fetch_demo_data(pinned_sha(), HERE / "demo"), HERE / "data")

    import uvicorn
    from support_copilot.main import app  # reads the settings above, loads the demo data

    # One process: the per-IP limiter lives in memory. The Spaces proxy is the only way in,
    # so its X-Forwarded-For is trusted for the client address.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
