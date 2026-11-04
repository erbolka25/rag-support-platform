"""Hugging Face Space entry point (Gradio SDK, ZeroGPU): the copilot's FastAPI app and UI on :7860.

The Space runs ``python app.py``; uvicorn serves support_copilot.main:app directly, so the
home page is the copilot's own UI (Swagger at /docs). rag_core and the copilot are installed
from requirements.txt at a pinned commit; the demo knowledge base is not part of those
packages, so it is downloaded from the same commit at startup.

ZeroGPU: ``spaces`` must be imported before torch, and the Space must report at least one
@spaces.GPU function at startup or it is stopped. ``spaces`` sends that report from a hook on
gradio's ``Blocks.launch``, which never runs here (no gradio UI), so main() sends it itself.
The GPU is never used: ``gpu_warmup`` is a no-op and the embedding model runs on CPU.
"""

# spaces first, before anything that may import torch.
try:
    import spaces
except ImportError:  # local runs and tests: no ZeroGPU
    spaces = None

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


def _gpu(func):
    return spaces.GPU(func) if spaces is not None else func


@_gpu
def gpu_warmup() -> None:
    """Never called: it only satisfies ZeroGPU's "at least one @spaces.GPU function"."""


def on_zero_gpu() -> bool:
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")


def report_zerogpu_startup() -> None:
    """Tell ZeroGPU the app started and which @spaces.GPU functions it has.

    ``spaces.zero.startup`` exists in both spaces 0.51 (as a gradio launch hook) and 0.52+
    (also called by ZeroGPUMiddleware on lifespan startup).
    """
    if spaces is None or not on_zero_gpu():
        return
    startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if startup is None:
        raise RuntimeError("This spaces version has no spaces.zero.startup()")
    startup()


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

    report_zerogpu_startup()
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
