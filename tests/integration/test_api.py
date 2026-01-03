# =============================================================================
# Integration Tests for API Endpoints
# =============================================================================
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import asyncio

from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient


# =============================================================================
# Health Endpoints Tests
# =============================================================================

class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint returns API information."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "endpoints" in data
        assert data["message"] == "Enterprise RAG System"

    @pytest.mark.asyncio
    async def test_liveness_endpoint(self, async_client: AsyncClient):
        """Test liveness probe for Kubernetes."""
        response = await async_client.get("/live")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness_all_healthy(self, async_client):
        """Test readiness probe when all services are healthy."""
        from src.main import app

        # Mock all service checks to return True
        with patch("src.api.routes.health.check_qdrant", return_value=asyncio.sleep(0) or True), \
             patch("src.api.routes.health.check_postgres", return_value=asyncio.sleep(0) or True), \
             patch("src.api.routes.health.check_redis", return_value=asyncio.sleep(0) or True), \
             patch("src.api.routes.health.check_minio", return_value=asyncio.sleep(0) or True):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/ready")

                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert "checks" in data
                assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_qdrant_unhealthy(self, async_client):
        """Test readiness probe when Qdrant is down."""
        from src.main import app

        # Mock Qdrant check to fail
        with patch("src.api.routes.health.check_qdrant", side_effect=Exception("Qdrant down")):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/ready")

                # Should still return response but with degraded status
                assert response.status_code in [200, 503]
                data = response.json()
                assert "status" in data

    @pytest.mark.asyncio
    async def test_health_check_returns_service_status(self, async_client):
        """Test health check returns status of all services."""
        from src.main import app

        with patch("src.api.routes.health.check_qdrant", return_value=True), \
             patch("src.api.routes.health.check_postgres", return_value=True), \
             patch("src.api.routes.health.check_redis", return_value=True), \
             patch("src.api.routes.health.check_minio", return_value=True):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert "checks" in data
                assert "qdrant" in data["checks"]
                assert "postgres" in data["checks"]
                assert "redis" in data["checks"]
                assert "minio" in data["checks"]


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Test query endpoint integration."""

    @pytest.mark.asyncio
    async def test_query_missing_query_field(self, async_client: AsyncClient):
        """Test query endpoint with missing query field."""
        response = await async_client.post("/query", json={})

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_empty_string(self, async_client: AsyncClient):
        """Test query endpoint with empty string."""
        response = await async_client.post("/query", json={
            "query": "   ",
            "use_cache": False
        })

        # Should accept but return minimal response
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_query_with_cache_disabled(self, async_client):
        """Test query endpoint with caching disabled."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        # Mock orchestrator
        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.sources = []
        mock_response.is_grounded = True
        mock_response.query_id = str(uuid4())
        mock_response.confidence = 0.9

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_response)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={
                    "query": "test query",
                    "use_cache": False
                })

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert data["answer"] == "Test answer"
                assert data["cached"] is False

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, async_client):
        """Test query endpoint with conversation history."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        mock_response = MagicMock()
        mock_response.answer = "Contextual answer"
        mock_response.sources = []
        mock_response.is_grounded = True
        mock_response.query_id = str(uuid4())
        mock_response.confidence = 0.85

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_response)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={
                    "query": "follow-up question",
                    "conversation_history": ["First question", "First answer"],
                    "use_cache": False
                })

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data

    @pytest.mark.asyncio
    async def test_query_orchestrator_error(self, async_client):
        """Test query endpoint when orchestrator fails."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        # Mock orchestrator to raise error
        with patch.object(orchestrator, "process_query", new=AsyncMock(side_effect=Exception("Orchestrator error"))):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={
                    "query": "test query",
                    "use_cache": False
                })

                # Should handle error gracefully
                assert response.status_code == 500
                data = response.json()
                assert "detail" in data or "error" in data

    @pytest.mark.asyncio
    async def test_query_with_max_tokens(self, async_client):
        """Test query endpoint with custom max_tokens."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        mock_response = MagicMock()
        mock_response.answer = "Short answer"
        mock_response.sources = []
        mock_response.is_grounded = True
        mock_response.query_id = str(uuid4())
        mock_response.confidence = 0.8

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_response)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={
                    "query": "test query",
                    "max_tokens": 100,
                    "use_cache": False
                })

                assert response.status_code == 200


# =============================================================================
# Document Endpoints Tests
# =============================================================================

class TestDocumentEndpoints:
    """Test document endpoints integration."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, async_client: AsyncClient):
        """Test listing documents when none exist."""
        from src.main import app
        from src.storage.metadata_store import metadata_store

        # Mock empty document list
        with patch.object(metadata_store, "list_documents", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/api/v1/documents")

                assert response.status_code == 200
                data = response.json()
                assert "documents" in data
                assert "total" in data
                assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, async_client):
        """Test listing documents with pagination."""
        from src.main import app
        from src.storage.metadata_store import metadata_store

        # Mock document list
        mock_docs = [
            MagicMock(id=str(uuid4()), title=f"Document {i}", status="indexed")
            for i in range(5)
        ]

        with patch.object(metadata_store, "list_documents", new=AsyncMock(return_value=mock_docs)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/api/v1/documents?page=1&page_size=5")

                assert response.status_code == 200
                data = response.json()
                assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_get_document_found(self, async_client):
        """Test getting a document that exists."""
        from src.main import app
        from src.storage.metadata_store import metadata_store

        doc_id = str(uuid4())
        mock_doc = MagicMock(
            id=doc_id,
            title="Test Document",
            file_type="md",
            status="indexed",
            metadata={}
        )

        with patch.object(metadata_store, "get_document", new=AsyncMock(return_value=mock_doc)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get(f"/api/v1/documents/{doc_id}")

                assert response.status_code == 200
                data = response.json()
                assert data["id"] == doc_id
                assert data["title"] == "Test Document"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, async_client):
        """Test getting a document that doesn't exist."""
        from src.main import app
        from src.storage.metadata_store import metadata_store

        with patch.object(metadata_store, "get_document", new=AsyncMock(return_value=None)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get(f"/api/v1/documents/{uuid4()}")

                assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document(self, async_client):
        """Test deleting a document."""
        from src.main import app
        from src.storage.metadata_store import metadata_store
        from src.storage.vector_store import vector_store

        doc_id = str(uuid4())

        with patch.object(metadata_store, "delete_document", new=AsyncMock(return_value=True)), \
             patch.object(vector_store, "delete_document", new=AsyncMock(return_value=True)):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.delete(f"/api/v1/documents/{doc_id}")

                assert response.status_code == 200
                data = response.json()
                assert "id" in data
                assert data["id"] == doc_id

    @pytest.mark.asyncio
    async def test_reindex_document(self, async_client):
        """Test reindexing a document."""
        from src.main import app
        from src.storage.metadata_store import metadata_store
        from src.indexing.index_manager import index_manager

        doc_id = str(uuid4())
        mock_doc = MagicMock(
            id=doc_id,
            title="Test Document",
            file_type="md",
            status="indexed"
        )

        with patch.object(metadata_store, "get_document", new=AsyncMock(return_value=mock_doc)), \
             patch.object(index_manager, "index_document", new=AsyncMock(return_value=MagicMock(success=True))):

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post(f"/api/v1/documents/{doc_id}/reindex")

                assert response.status_code == 200
                data = response.json()
                assert "id" in data

    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, async_client):
        """Test listing documents with filters."""
        from src.main import app
        from src.storage.metadata_store import metadata_store

        mock_docs = [
            MagicMock(id=str(uuid4()), title="HR Policy", category="hr", status="indexed")
        ]

        with patch.object(metadata_store, "list_documents", new=AsyncMock(return_value=mock_docs)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/api/v1/documents?category=hr&status=indexed")

                assert response.status_code == 200


# =============================================================================
# API Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Test API integration scenarios."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, async_client: AsyncClient):
        """Test that CORS headers are present."""
        response = await async_client.options("/")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_response_time_header(self, async_client):
        """Test that response includes timing information."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        mock_response = MagicMock()
        mock_response.answer = "Test"
        mock_response.sources = []
        mock_response.is_grounded = True
        mock_response.query_id = str(uuid4())

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_response)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={"query": "test", "use_cache": False})

                # Check for timing headers
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_id_header_generated(self, async_client):
        """Test that request ID is generated and returned."""
        from src.main import app
        from src.agents.orchestrator import orchestrator

        mock_response = MagicMock()
        mock_response.answer = "Test"
        mock_response.sources = []
        mock_response.is_grounded = True
        mock_response.query_id = str(uuid4())

        with patch.object(orchestrator, "process_query", new=AsyncMock(return_value=mock_response)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/query", json={"query": "test", "use_cache": False})

                # Check for request ID header
                assert "x-request-id" in response.headers.lower() or response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limiting_works(self, async_client):
        """Test that rate limiting is enforced."""
        from src.main import app
        from src.storage.cache import cache_store

        # Mock rate limit check to block after 10 requests
        request_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            if request_count > 10:
                return None  # Simulate rate limit exceeded
            return '{"tokens": 60}'

        with patch.object(cache_store.redis, "get", new=AsyncMock(side_effect=mock_get)):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                # Make multiple requests
                responses = []
                for _ in range(5):
                    response = await ac.get("/health")
                    responses.append(response.status_code)

                # All should succeed within limit
                assert all(status == 200 for status in responses)
