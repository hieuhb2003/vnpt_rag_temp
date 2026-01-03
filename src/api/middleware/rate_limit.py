# =============================================================================
# Rate Limiting Middleware - Token bucket rate limiting
# =============================================================================
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json

from src.storage.cache import cache_store
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting per client IP.

    Uses Redis to track token buckets for each client IP.
    Tokens replenish over time (requests_per_minute per minute).
    Burst allows limited bursts up to burst_size tokens.

    Algorithm:
    - Each client has a bucket of tokens
    - Request consumes 1 token
    - Tokens replenish at rate: rpm / 60 per second
    - Bucket max size: burst_size
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize RateLimitMiddleware.

        Args:
            app: ASGI application
            requests_per_minute: Sustained rate limit
            burst_size: Maximum burst tokens
        """
        super().__init__(app)
        self.rpm = requests_per_minute
        self.burst = burst_size

        logger.info(
            "Rate limit middleware initialized",
            rpm=requests_per_minute,
            burst=burst_size
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Skips rate limiting for health endpoints.
        """
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/live", "/api/health", "/api/ready", "/api/live"]:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if not await self._check_rate_limit(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.

        Checks X-Forwarded-For header for proxied requests,
        falls back to direct client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has available tokens.

        Uses token bucket algorithm with Redis storage.

        Args:
            client_id: Client identifier (IP address)

        Returns:
            True if request should be allowed, False if rate limited
        """
        key = f"ratelimit:{client_id}"
        now = time.time()

        try:
            data = await cache_store.redis.get(key)
            if data:
                bucket = json.loads(data)
                tokens = bucket["tokens"]
                last_update = bucket["last_update"]

                # Replenish tokens based on elapsed time
                elapsed = now - last_update
                refill_rate = self.rpm / 60  # tokens per second
                new_tokens = min(self.burst, tokens + elapsed * refill_rate)
            else:
                # First request from this client
                new_tokens = self.burst

            # Check if we have enough tokens
            if new_tokens < 1:
                return False

            # Consume one token and update
            await cache_store.redis.setex(
                key,
                60,  # 60 second TTL
                json.dumps({
                    "tokens": new_tokens - 1,
                    "last_update": now
                })
            )
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}", exc_info=True)
            # Fail open - allow request if Redis is down
            return True
