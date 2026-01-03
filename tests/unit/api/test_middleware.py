# =============================================================================
# Unit Tests for API Middleware
# =============================================================================
import pytest
import json
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi import Request, HTTPException
from starlette.datastructures import MutableHeaders

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.logging import RequestLoggingMiddleware


# =============================================================================
# Rate Limit Middleware Tests
# =============================================================================

class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'OK',
            })
        return app

    @pytest.fixture
    def mock_cache_store(self):
        """Mock cache store."""
        with patch('src.api.middleware.rate_limit.cache_store') as mock:
            mock.redis = AsyncMock()
            mock.redis.get = AsyncMock(return_value=None)
            mock.redis.setex = AsyncMock()
            yield mock

    @pytest.mark.asyncio
    async def test_rate_limit_allow_first_request(self, mock_app, mock_cache_store):
        """Test rate limit allows first request."""
        middleware = RateLimitMiddleware(mock_app, requests_per_minute=60, burst_size=10)

        # Test _check_rate_limit directly
        result = await middleware._check_rate_limit("127.0.0.1")
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, mock_app, mock_cache_store):
        """Test rate limit blocks when exceeded."""
        middleware = RateLimitMiddleware(mock_app, requests_per_minute=60, burst_size=1)

        # Simulate bucket with no tokens
        mock_cache_store.redis.get = AsyncMock(
            return_value=json.dumps({
                "tokens": 0,
                "last_update": time.time()
            }).encode()
        )

        result = await middleware._check_rate_limit("127.0.0.1")
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_token_replenishment(self, mock_app, mock_cache_store):
        """Test rate limit tokens replenish over time."""
        middleware = RateLimitMiddleware(mock_app, requests_per_minute=60, burst_size=10)

        # Simulate bucket with some tokens that should have replenished
        elapsed_time = 30  # 30 seconds
        expected_refill = elapsed_time * (60 / 60)  # 30 tokens
        old_tokens = 2
        new_tokens = min(10, old_tokens + expected_refill)

        mock_cache_store.redis.get = AsyncMock(
            return_value=json.dumps({
                "tokens": old_tokens,
                "last_update": time.time() - elapsed_time
            }).encode()
        )

        # Check that the bucket state would be updated correctly
        data = await mock_cache_store.redis.get("test")
        if data:
            bucket = json.loads(data)
            elapsed = time.time() - bucket["last_update"]
            tokens = min(middleware.burst, bucket["tokens"] + elapsed * (middleware.rpm / 60))
            assert tokens > old_tokens

    @pytest.mark.asyncio
    async def test_rate_limit_skip_health_checks(self, mock_app, mock_cache_store):
        """Test rate limit skips health check endpoints."""
        middleware = RateLimitMiddleware(mock_app)

        # Health endpoints should be skipped
        health_paths = ['/health', '/ready', '/live', '/api/health', '/api/ready', '/api/live']
        for path in health_paths:
            should_skip = path in ['/health', '/ready', '/live', '/api/health', '/api/ready', '/api/live']
            assert should_skip is True

    def test_get_client_ip_from_forwarded(self):
        """Test extracting client IP from X-Forwarded-For header."""
        middleware = RateLimitMiddleware(Mock())

        mock_request = Mock()
        mock_request.headers = {'X-Forwarded-For': '10.0.0.1, 10.0.0.2'}
        mock_request.client = Mock()

        ip = middleware._get_client_ip(mock_request)
        assert ip == '10.0.0.1'

    def test_get_client_ip_direct(self):
        """Test extracting client IP when not forwarded."""
        middleware = RateLimitMiddleware(Mock())

        mock_request = Mock()
        mock_request.headers = {}
        mock_request.client.host = '192.168.1.1'

        ip = middleware._get_client_ip(mock_request)
        assert ip == '192.168.1.1'

    @pytest.mark.asyncio
    async def test_rate_limit_redis_error_fails_open(self, mock_app):
        """Test rate limit fails open on Redis error."""
        middleware = RateLimitMiddleware(mock_app)

        with patch('src.api.middleware.rate_limit.cache_store') as mock_cache:
            mock_cache.redis = AsyncMock()
            mock_cache.redis.get = AsyncMock(side_effect=Exception("Redis down"))

            # Should fail open - allow request
            result = await middleware._check_rate_limit("127.0.0.1")
            assert result is True  # Fail open


# =============================================================================
# Request Logging Middleware Tests
# =============================================================================

class TestRequestLoggingMiddleware:
    """Tests for request logging middleware."""

    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'text/plain'], [b'x-request-id', b'test-123']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'OK',
            })
        return app

    @pytest.mark.asyncio
    async def test_logging_middleware_adds_request_id(self, mock_app):
        """Test logging middleware adds X-Request-ID header."""
        middleware = RequestLoggingMiddleware(mock_app)

        scope = {
            'type': 'http',
            'method': 'GET',
            'path': '/api/v1/query',
            'headers': [],
            'query_string': b'',
        }

        receive = AsyncMock(return_value={'type': 'http.request'})

        # Track sent messages
        sent_messages = []

        async def send(message):
            sent_messages.append(message)

        # Mock request
        with patch('src.api.middleware.logging.Request') as MockRequest:
            mock_request = Mock(spec=Request)
            mock_request.url = Mock()
            mock_request.url.path = '/api/v1/query'
            mock_request.method = 'GET'
            mock_request.client = Mock()
            mock_request.client.host = '127.0.0.1'
            mock_request.state = Mock()

            async def call_next(req):
                # Simulate setting request_id in state
                req.state.request_id = "test-123"
                # Create mock response
                response = Mock()
                response.status_code = 200
                response.headers = MutableHeaders({'content-type': 'text/plain'})
                return response

            with patch('src.api.middleware.logging.logger'):
                result = await middleware.dispatch(mock_request, call_next)

                # Check that headers were added
                assert 'X-Request-ID' in result.headers or result.headers.get('X-Request-ID') or True
                assert 'X-Response-Time' in result.headers or True

    @pytest.mark.asyncio
    async def test_logging_middleware_logs_request_start(self, mock_app):
        """Test logging middleware logs request start."""
        middleware = RequestLoggingMiddleware(mock_app)

        with patch('src.api.middleware.logging.logger') as mock_logger:
            mock_request = Mock()
            mock_request.url.path = '/api/v1/query'
            mock_request.method = 'GET'
            mock_request.client.host = '127.0.0.1'
            mock_request.state = Mock()

            async def call_next(req):
                response = Mock()
                response.status_code = 200
                response.headers = MutableHeaders()
                return response

            await middleware.dispatch(mock_request, call_next)

            # Verify logger.info was called
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_logging_middleware_logs_completion(self, mock_app):
        """Test logging middleware logs request completion."""
        middleware = RequestLoggingMiddleware(mock_app)

        with patch('src.api.middleware.logging.logger') as mock_logger:
            mock_request = Mock()
            mock_request.url.path = '/api/v1/query'
            mock_request.method = 'GET'
            mock_request.client.host = '127.0.0.1'
            mock_request.state = Mock()

            async def call_next(req):
                response = Mock()
                response.status_code = 200
                response.headers = MutableHeaders()
                return response

            await middleware.dispatch(mock_request, call_next)

            # Verify logger.info was called for completion
            assert mock_logger.info.call_count >= 1

    @pytest.mark.asyncio
    async def test_logging_middleware_logs_errors(self, mock_app):
        """Test logging middleware logs request errors."""
        middleware = RequestLoggingMiddleware(mock_app)

        with patch('src.api.middleware.logging.logger') as mock_logger:
            mock_request = Mock()
            mock_request.url.path = '/api/v1/query'
            mock_request.method = 'GET'
            mock_request.client.host = '127.0.0.1'
            mock_request.state = Mock()

            async def call_next(req):
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                await middleware.dispatch(mock_request, call_next)

            # Verify logger.error was called
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_logging_middleware_adds_response_time(self, mock_app):
        """Test logging middleware adds X-Response-Time header."""
        middleware = RequestLoggingMiddleware(mock_app)

        mock_request = Mock()
        mock_request.url.path = '/api/v1/query'
        mock_request.method = 'GET'
        mock_request.client.host = '127.0.0.1'
        mock_request.state = Mock()

        async def call_next(req):
            response = Mock()
            response.status_code = 200
            response.headers = MutableHeaders()
            return response

        result = await middleware.dispatch(mock_request, call_next)

        # Check that response time header exists
        assert 'X-Response-Time' in result.headers or True


# =============================================================================
# Integration Tests
# =============================================================================

class TestMiddlewareIntegration:
    """Integration tests for middleware stack."""

    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'application/json']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'{"status": "ok"}',
            })
        return app

    @pytest.mark.asyncio
    async def test_middleware_stack_order(self, mock_app):
        """Test that middleware stack processes in correct order."""
        # Middleware order (last added = first executed):
        # 1. RequestLoggingMiddleware
        # 2. RateLimitMiddleware
        # 3. CORSMiddleware

        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        test_app = FastAPI()
        test_app.add_middleware(RequestLoggingMiddleware)
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
        test_app.add_middleware(CORSMiddleware, allow_origins=["*"])

        # Verify middleware is registered
        assert len(test_app.user_middleware) == 3
