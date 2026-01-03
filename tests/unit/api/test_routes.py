# =============================================================================
# Unit Tests for API Routes
# =============================================================================
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks

from src.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    with patch('src.api.routes.query.orchestrator') as mock:
        mock.process_query = AsyncMock(return_value={
            "query_id": "test-query-id",
            "answer": "Test answer",
            "citations": [],
            "metadata": {"complexity": "simple"},
            "verification": {"is_grounded": True},
            "processing_time_ms": 100.0
        })
        yield mock


@pytest.fixture
def mock_cache_store():
    """Mock cache store."""
    with patch('src.api.routes.query.cache_store') as mock:
        mock.get_semantic_cache = AsyncMock(return_value=None)
        mock.set_semantic_cache = AsyncMock()
        mock.redis = AsyncMock()
        mock.redis.ping = AsyncMock(return_value=True)
        yield mock


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    with patch('src.api.routes.query.get_embedder') as mock:
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock.return_value = embedder
        yield mock


# =============================================================================
# Health Route Tests
# =============================================================================

class TestHealthRoutes:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns system info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Enterprise RAG System"
        assert data["version"] == "0.1.0"
        assert "docs" in data
        assert "endpoints" in data

    @patch('src.api.routes.health.document_store')
    @patch('src.api.routes.health.cache_store')
    @patch('src.api.routes.health.metadata_store')
    @patch('src.api.routes.health.vector_store')
    def test_health_check_all_healthy(
        self, mock_vec, mock_meta, mock_cache, mock_doc_store, client
    ):
        """Test health check when all services are healthy."""
        # Mock vector store
        mock_vec.client.get_collections = AsyncMock()
        mock_vec.client = Mock()

        # Mock metadata store session
        class MockSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def execute(self, text):
                class MockResult:
                    scalar = AsyncMock(return_value=1)
                return MockResult()

        mock_meta.session = Mock(return_value=MockSession())
        mock_meta.engine = Mock()

        # Mock cache store
        mock_cache.redis = Mock()
        mock_cache.redis.ping = AsyncMock()

        # Mock document store
        mock_doc_store.client = Mock()
        mock_doc_store._loop = Mock()
        mock_doc_store._loop.run_in_executor = AsyncMock(return_value=True)
        mock_doc_store.bucket_name = "documents"

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert len(data["services"]) == 4

    @patch('src.api.routes.health.vector_store')
    def test_health_check_qdrant_unhealthy(self, mock_vec, client):
        """Test health check when Qdrant is unhealthy."""
        mock_vec.client.get_collections = AsyncMock(side_effect=Exception("Connection error"))
        mock_vec.client = Mock()

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        qdrant = next(s for s in data["services"] if s["name"] == "qdrant")
        assert qdrant["status"] == "unhealthy"

    def test_live_endpoint(self, client):
        """Test liveness probe."""
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json() == {"alive": True}

    @patch('src.api.routes.health.vector_store')
    @patch('src.api.routes.health.metadata_store')
    @patch('src.api.routes.health.cache_store')
    @patch('src.api.routes.health.document_store')
    def test_ready_endpoint_all_connected(
        self, mock_doc_store, mock_cache, mock_meta, mock_vec, client
    ):
        """Test readiness probe when all services connected."""
        mock_vec.client = Mock()
        mock_meta.engine = Mock()
        mock_cache.redis = Mock()
        mock_doc_store.client = Mock()

        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True}

    @patch('src.api.routes.health.document_store')
    @patch('src.api.routes.health.cache_store')
    @patch('src.api.routes.health.metadata_store')
    @patch('src.api.routes.health.vector_store')
    def test_ready_endpoint_not_ready(self, mock_vec, mock_meta, mock_cache, mock_doc_store, client):
        """Test readiness probe when services not connected."""
        mock_vec.client = None
        mock_meta.engine = Mock()
        mock_cache.redis = Mock()
        mock_doc_store.client = Mock()

        response = client.get("/ready")
        assert response.status_code == 503


# =============================================================================
# Query Route Tests
# =============================================================================

class TestQueryRoutes:
    """Tests for query processing endpoint."""

    def test_query_endpoint_missing_query(self, client):
        """Test query endpoint with missing query."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query."""
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422

    @patch('src.api.routes.query.get_embedder')
    @patch('src.api.routes.query.cache_store')
    @patch('src.api.routes.query.orchestrator')
    def test_query_cache_hit(
        self, mock_orch, mock_cache, mock_embedder, client
    ):
        """Test query endpoint with cache hit."""
        # Setup mocks
        mock_embedder.return_value.embed = AsyncMock(return_value=[0.1, 0.2])
        mock_cache.get_semantic_cache = AsyncMock(return_value={
            "query_id": "cached-id",
            "answer": "Cached answer",
            "citations": [],
            "metadata": {},
            "verification": {},
            "processing_time_ms": 50.0
        })

        response = client.post("/api/v1/query", json={
            "query": "test question",
            "use_cache": True
        })

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert data["answer"] == "Cached answer"
        mock_orch.process_query.assert_not_called()

    @patch('src.api.routes.query.get_embedder')
    @patch('src.api.routes.query.cache_store')
    @patch('src.api.routes.query.orchestrator')
    def test_query_cache_miss(
        self, mock_orch, mock_cache, mock_embedder, client
    ):
        """Test query endpoint with cache miss."""
        # Setup mocks
        mock_embedder.return_value.embed = AsyncMock(return_value=[0.1, 0.2])
        mock_cache.get_semantic_cache = AsyncMock(return_value=None)
        mock_orch.process_query = AsyncMock(return_value={
            "query_id": "new-id",
            "answer": "New answer",
            "citations": [],
            "metadata": {"complexity": "simple"},
            "verification": {"is_grounded": True},
            "processing_time_ms": 100.0
        })
        mock_cache.set_semantic_cache = AsyncMock()

        response = client.post("/api/v1/query", json={
            "query": "test question",
            "use_cache": True
        })

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is False
        assert data["answer"] == "New answer"
        mock_orch.process_query.assert_awaited_once()

    @patch('src.api.routes.query.get_embedder')
    @patch('src.api.routes.query.cache_store')
    @patch('src.api.routes.query.orchestrator')
    def test_query_with_conversation_id(self, mock_orch, mock_cache, mock_embedder, client):
        """Test query with conversation ID."""
        mock_embedder.return_value.embed = AsyncMock(return_value=[0.1, 0.2])
        mock_cache.get_semantic_cache = AsyncMock(return_value=None)
        mock_orch.process_query = AsyncMock(return_value={
            "query_id": "test-id",
            "answer": "Test answer",
            "citations": [],
            "metadata": {},
            "verification": {},
            "processing_time_ms": 100.0
        })
        mock_cache.set_semantic_cache = AsyncMock()

        response = client.post("/api/v1/query", json={
            "query": "test question",
            "conversation_id": "conv-123",
            "use_cache": False
        })

        assert response.status_code == 200
        # Check that orchestrator was called
        assert mock_orch.process_query.called

    @patch('src.api.routes.query.orchestrator')
    def test_query_orchestrator_error(self, mock_orch, client):
        """Test query endpoint when orchestrator fails."""
        mock_orch.process_query = AsyncMock(side_effect=Exception("Processing failed"))

        response = client.post("/api/v1/query", json={
            "query": "test question",
            "use_cache": False
        })

        assert response.status_code == 500
        assert "Processing failed" in response.json()["detail"]


# =============================================================================
# Documents Route Tests
# =============================================================================

class TestDocumentsRoutes:
    """Tests for document management endpoints."""

    @patch('src.api.routes.documents.metadata_store')
    def test_list_documents_empty(self, mock_meta, client):
        """Test listing documents when empty."""
        mock_meta.list_documents = AsyncMock(return_value=[])
        mock_meta.count_documents = AsyncMock(return_value=0)

        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    @patch('src.api.routes.documents.metadata_store')
    def test_list_documents_with_data(self, mock_meta, client):
        """Test listing documents with data."""
        from datetime import datetime
        from uuid import uuid4

        doc_id = str(uuid4())
        now = datetime.utcnow()

        # Create mock document
        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.title = "Test Document"
        mock_doc.file_type = "pdf"
        mock_doc.status.value = "indexed"
        mock_doc.created_at = now
        mock_doc.updated_at = now

        mock_meta.list_documents = AsyncMock(return_value=[mock_doc])
        mock_meta.count_documents = AsyncMock(return_value=1)
        mock_meta.get_sections_by_document = AsyncMock(return_value=[])

        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["documents"]) == 1
        assert data["documents"][0]["title"] == "Test Document"

    @patch('src.api.routes.documents.metadata_store')
    def test_list_documents_with_filters(self, mock_meta, client):
        """Test listing documents with filters."""
        mock_meta.list_documents = AsyncMock(return_value=[])
        mock_meta.count_documents = AsyncMock(return_value=0)

        response = client.get("/api/v1/documents?status=indexed&limit=10&offset=5")

        assert response.status_code == 200
        mock_meta.list_documents.assert_awaited_with(
            limit=10,
            offset=5,
            status="indexed"
        )

    @patch('src.api.routes.documents.metadata_store')
    def test_get_document_found(self, mock_meta, client):
        """Test getting a document that exists."""
        from uuid import uuid4
        from datetime import datetime

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.title = "Test Doc"
        mock_doc.file_type = "pdf"
        mock_doc.status.value = "indexed"
        mock_doc.created_at = datetime.utcnow()
        mock_doc.updated_at = datetime.utcnow()

        mock_meta.get_document = AsyncMock(return_value=mock_doc)
        mock_meta.get_sections_by_document = AsyncMock(return_value=[])

        response = client.get(f"/api/v1/documents/{mock_doc.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Doc"

    @patch('src.api.routes.documents.metadata_store')
    def test_get_document_not_found(self, mock_meta, client):
        """Test getting a document that doesn't exist."""
        from uuid import uuid4

        mock_meta.get_document = AsyncMock(return_value=None)

        doc_id = uuid4()
        response = client.get(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 404

    @patch('src.api.routes.documents.metadata_store')
    def test_get_document_invalid_id(self, mock_meta, client):
        """Test getting document with invalid ID."""
        response = client.get("/api/v1/documents/invalid-uuid")

        assert response.status_code == 400

    @patch('src.api.routes.documents.metadata_store')
    def test_delete_document(self, mock_meta, client):
        """Test deleting a document."""
        from uuid import uuid4

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_meta.get_document = AsyncMock(return_value=mock_doc)

        with patch('src.api.routes.documents.index_manager') as mock_index:
            mock_index.delete_document_index = AsyncMock(return_value=True)

            response = client.delete(f"/api/v1/documents/{mock_doc.id}")

            assert response.status_code == 200
            assert "deletion started" in response.json()["message"]

    @patch('src.api.routes.documents.index_manager')
    @patch('src.api.routes.documents.document_store')
    @patch('src.api.routes.documents.metadata_store')
    @patch('src.api.routes.documents.tree_builder')
    @patch('src.api.routes.documents.chunker')
    @patch('src.api.routes.documents.parser_factory')
    def test_document_upload(self, mock_parser, mock_chunker, mock_tree, mock_meta, mock_doc_store, mock_index, client):
        """Test document upload."""
        from uuid import uuid4

        # Create mock file
        file_content = b"test content"

        # Mock document store upload
        mock_doc_store.upload_bytes = AsyncMock(return_value="documents/test.txt")

        # Mock metadata store create_document
        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_meta.create_document = AsyncMock()

        # Mock parser
        mock_parsed = Mock()
        mock_parsed.title = "test.txt"
        mock_parsed.sections = []
        mock_parser.parse = AsyncMock(return_value=mock_parsed)

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
            data={"category": "test", "tags": "tag1,tag2"}
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "pending"


# =============================================================================
# Query Health Endpoint Tests
# =============================================================================

class TestQueryHealth:
    """Tests for query health endpoint."""

    @patch('src.api.routes.query.orchestrator')
    def test_query_health(self, mock_orch, client):
        """Test query health endpoint."""
        # Mock orchestrator
        mock_orch_instance = Mock()
        mock_orch_instance.app = Mock()
        mock_orch.__or__ = Mock(return_value=mock_orch_instance)

        response = client.get("/api/v1/query/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDocumentsHealth:
    """Tests for documents health endpoint."""

    @patch('src.api.routes.documents.metadata_store')
    def test_documents_health(self, mock_meta, client):
        """Test documents health endpoint."""
        # Mock metadata store health check
        mock_meta.health_check = AsyncMock(return_value={"status": "healthy"})

        response = client.get("/api/v1/documents/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
