# =============================================================================
# Pytest Configuration and Fixtures
# =============================================================================
import pytest
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.storage import init_storage, close_storage
from src.utils.logging import configure_logging


# =============================================================================
# Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Configure logging for tests."""
    configure_logging(debug=False)


# =============================================================================
# Storage Fixtures
# =============================================================================

@pytest.fixture(scope="function")
async def storage():
    """Initialize and cleanup storage for tests."""
    await init_storage()
    yield
    await close_storage()


@pytest.fixture(scope="function")
async def clean_storage(storage):
    """Clean all collections before each test."""
    from src.storage import vector_store, metadata_store

    # Clear vector collections
    await vector_store.clear_collection("documents")
    await vector_store.clear_collection("sections")
    await vector_store.clear_collection("chunks")

    # Clear database tables
    async with metadata_store.session() as session:
        from src.storage.models import ChunkORM, SectionORM, DocumentORM
        from sqlalchemy import delete

        await session.execute(delete(ChunkORM))
        await session.execute(delete(SectionORM))
        await session.execute(delete(DocumentORM))
        await session.commit()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.config.settings import Settings
    return Settings(
        debug=True,
        llm_provider="openai",
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        embedding_model="text-embedding-3-small",
        qdrant_host="localhost",
        qdrant_port=6333,
        postgres_host="localhost",
        postgres_port=5432,
        redis_host="localhost",
        redis_port=6379,
        minio_host="localhost",
        minio_port=9000
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(
        content='{"rewritten_query": "test query", "keywords": ["test"], "query_type": "factoid", "confidence": 0.9}'
    ))
    return llm


@pytest.fixture
def mock_vector_store():
    """Mock Qdrant vector store."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[
        MagicMock(
            id="chunk-1",
            score=0.95,
            payload={
                "content": "Test content",
                "document_id": str(uuid4()),
                "section_id": str(uuid4())
            }
        )
    ])
    store.hybrid_search = AsyncMock(return_value=[
        {
            "id": "chunk-1",
            "score": 0.95,
            "payload": {
                "content": "Test content",
                "document_id": str(uuid4())
            }
        }
    ])
    return store


@pytest.fixture
def mock_metadata_store():
    """Mock PostgreSQL metadata store."""
    store = AsyncMock()
    store.get_document = AsyncMock(return_value=MagicMock(
        id=uuid4(),
        title="Test Document",
        file_type="md",
        status="indexed"
    ))
    return store


@pytest.fixture
def mock_cache():
    """Mock Redis cache."""
    cache = AsyncMock()
    cache.get_embedding = AsyncMock(return_value=None)
    cache.set_embedding = AsyncMock()
    cache.get_semantic_response = AsyncMock(return_value=None)
    return cache


@pytest.fixture
def mock_embedder():
    """Mock embedding model."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return embedder


# =============================================================================
# Test Client Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create test client."""
    from src.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client() -> AsyncGenerator:
    """Create async test client."""
    from src.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Sample Document Fixtures
# =============================================================================

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    from src.models.document import Document, DocumentMetadata, DocumentStatus
    return Document(
        id=uuid4(),
        title="Test Document",
        file_path="documents/test.md",
        file_type="md",
        metadata=DocumentMetadata(category="test", tags=["unit-test"]),
        status=DocumentStatus.PENDING
    )


@pytest.fixture
def sample_sections():
    """Sample sections for testing."""
    from src.models.document import Section
    doc_id = uuid4()
    return [
        Section(
            id=uuid4(),
            document_id=doc_id,
            heading="Introduction",
            level=1,
            section_path="1",
            content="This is the introduction.",
            position=0
        ),
        Section(
            id=uuid4(),
            document_id=doc_id,
            heading="Details",
            level=2,
            section_path="1.1",
            content="This is the detail section.",
            position=1
        )
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from src.models.document import Chunk
    doc_id = uuid4()
    return [
        Chunk(
            id=uuid4(),
            document_id=doc_id,
            content="This is chunk 1 content.",
            token_count=10,
            position=0
        ),
        Chunk(
            id=uuid4(),
            document_id=doc_id,
            content="This is chunk 2 content.",
            token_count=12,
            position=1
        )
    ]


# =============================================================================
# Sample Content Fixtures
# =============================================================================

@pytest.fixture
def sample_markdown_vi():
    """Sample Vietnamese markdown document."""
    return '''# Hướng Dẫn Sử Dụng

Chào mừng bạn đến với hệ thống.

## Cài Đặt

Hãy cài đặt Python 3.11.

### Bước 1

Tải Python về máy.

## Sử Dụng

Chạy lệnh để bắt đầu.
'''


@pytest.fixture
def sample_markdown_en():
    """Sample English markdown document."""
    return '''# Product Documentation

Welcome to our product.

## Getting Started

Follow these steps.

### Installation

Download the installer.

## Features

Our product includes:
- Fast search
- Real-time updates
'''


@pytest.fixture
def sample_html():
    """Sample HTML document."""
    return '''<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>Introduction paragraph.</p>
    <h2>Section One</h2>
    <p>Content for section one.</p>
    <h3>Subsection</h3>
    <p>More details here.</p>
</body>
</html>'''
