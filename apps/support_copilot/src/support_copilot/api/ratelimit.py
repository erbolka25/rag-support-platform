"""Per-IP rate limit for the expensive endpoints (LLM drafts, uploads) of a public demo.

The limiter itself is rag_core's; create_app puts one on app.state when
RATE_LIMIT_PER_HOUR > 0. The client address is rag_core's ``client_ip``: behind a proxy
(Hugging Face Spaces) run uvicorn with --proxy-headers so it is the real one.
"""

from __future__ import annotations

import math

from fastapi import HTTPException, Request
from rag_core.api.auth import client_ip
from rag_core.api.ratelimit import RateLimiter

WINDOW_S = 3600.0


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
