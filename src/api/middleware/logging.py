# =============================================================================
# Request Logging Middleware - Log all HTTP requests
# =============================================================================
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all HTTP requests with timing information.

    Adds:
    - X-Request-ID header to response
    - X-Response-Time header with duration in ms
    - Structured logging for each request
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request with logging.

        Generates unique request ID, logs start/end,
        and adds timing headers to response.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Store request_id in state for access in endpoints
        request.state.request_id = request_id

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log completion
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": f"{duration_ms:.2f}",
                    "method": request.method,
                    "path": request.url.path
                }
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": f"{duration_ms:.2f}",
                    "method": request.method,
                    "path": request.url.path
                }
            )
            raise
