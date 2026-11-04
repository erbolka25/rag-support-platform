"""Per-IP rate limit for the expensive endpoints (LLM drafts, uploads) of a public demo.

Self-contained on purpose: importing anything from ``rag_core.api`` runs that package's
``__init__``, which builds rag_core's own web app (a second RagService, its login and
session warnings) that the copilot never uses.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import HTTPException, Request

WINDOW_S = 3600.0


class RateLimiter:
    """At most ``limit`` hits per ``window_s`` seconds for each key, in process memory."""

    def __init__(
        self, limit: int, window_s: float = WINDOW_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int:
        """Record a hit. Returns 0 if allowed, else seconds until the next allowed hit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window_s - now))
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle keys so memory stays bounded
                for k in [k for k, v in self._hits.items() if v[-1] <= now - self.window_s]:
                    del self._hits[k]
            return 0


def client_ip(request: Request) -> str:
    # X-Forwarded-For is client-controlled, so it is not read here; behind a reverse proxy
    # (Hugging Face Spaces) run uvicorn with --proxy-headers so request.client is the client.
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Dependency: 429 once this IP spent its hourly budget; no-op without a limiter."""
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is None:
        return
    wait = limiter.hit(client_ip(request))
    if wait > 0:
        raise HTTPException(
            429,
            detail={
                "code": "rate_limited",
                "message": f"Demo limit: {limiter.limit} requests per hour. "
                f"Try again in {math.ceil(wait / 60)} min.",
            },
            headers={"Retry-After": str(wait)},
        )
