# =============================================================================
# Pytest Configuration and Fixtures
# =============================================================================
import pytest
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from src.storage import init_storage, close_storage
from src.utils.logging import configure_logging


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Configure logging for tests."""
    configure_logging(debug=False)


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
        await session.execute(ChunkORM.__table__.delete())
        await session.execute(SectionORM.__table__.delete())
        await session.execute(DocumentORM.__table__.delete())


# Sample document fixtures
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
