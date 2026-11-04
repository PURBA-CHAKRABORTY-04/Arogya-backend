"""
auth/rate_limit.py
-------------------
IP-based sliding-window rate limiter.

CURRENT IMPLEMENTATION: in-memory defaultdict (single-process only).

⚠️  PRODUCTION LIMITATION (Issue #3):
    This store resets on every server restart and is NOT shared across
    Uvicorn workers or multiple instances. With Render auto-scaling or
    multiple workers (uvicorn --workers N), each process has its own
    counter, making the effective limit N× higher than intended.

    MIGRATION PATH: Replace the defaultdict store with Redis using
    `slowapi` + `limits` with a Redis backend:

        from slowapi import Limiter
        from slowapi.util import get_remote_address
        from limits.storage import RedisStorage

        limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379"),
        )

    Until Redis is available, this is acceptable for low-traffic deployments
    with a single worker. Set UVICORN_WORKERS=1 in your deployment config
    to at least guarantee correctness until the migration is done.

Usage:
    from auth.rate_limit import RateLimiter
    forgot_password_limiter = RateLimiter(max_calls=5, window_seconds=900)

    @router.post("/forgot-password")
    def forgot_password(request: Request, ...):
        forgot_password_limiter.check(request)
        ...
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import HTTPException, Request, status


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter keyed by client IP.

    :param max_calls:       Maximum allowed calls within the window.
    :param window_seconds:  Length of the rolling time window in seconds.
    """

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls      = max_calls
        self.window         = timedelta(seconds=window_seconds)
        self._store: dict[str, list[datetime]] = defaultdict(list)
        self._lock          = Lock()

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract the real client IP, honouring X-Forwarded-For when behind a
        reverse proxy / Render's load balancer.
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """
        Check the rate limit for the request's IP.
        Raises HTTP 429 if the limit is exceeded.
        Call this at the top of any endpoint that should be rate-limited.
        """
        ip  = self._get_client_ip(request)
        now = datetime.now(timezone.utc)
        cutoff = now - self.window

        with self._lock:
            # Prune timestamps older than the window
            self._store[ip] = [t for t in self._store[ip] if t > cutoff]
            call_count = len(self._store[ip])

            if call_count >= self.max_calls:
                retry_after = int(self.window.total_seconds())
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Too many requests. You may try again after "
                        f"{self.window.seconds // 60} minutes."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

            self._store[ip].append(now)


# ── Pre-built limiter instances ────────────────────────────────────────────────

# Forgot-password: 5 requests per 15 minutes per IP
forgot_password_limiter = RateLimiter(max_calls=5, window_seconds=900)

# OTP verify: 10 attempts per 15 minutes per IP (brute-force protection)
verify_otp_limiter = RateLimiter(max_calls=10, window_seconds=900)
