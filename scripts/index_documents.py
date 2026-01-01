#!/usr/bin/env python
"""
Bulk index documents from a directory.

Usage:
    python scripts/index_documents.py --path /data/docs --workers 4
    python scripts/index_documents.py --path ./sample_docs --provider mock
"""
import asyncio
import argparse
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from src.storage import init_storage, close_storage, document_store, metadata_store
from src.indexing.document_parser import DocumentParserFactory
from src.indexing.tree_builder import tree_builder
from src.indexing.chunker import chunker
from src.indexing.index_manager import index_manager
from src.models.document import Document, DocumentMetadata, DocumentStatus
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# Supported file extensions
SUPPORTED_EXTENSIONS = [".md", ".markdown", ".html", ".htm", ".pdf", ".docx", ".doc"]


async def index_file(
    file_path: Path,
    provider: str = "mock"
) -> Tuple[bool, str]:
    """
    Index a single file.

    Args:
        file_path: Path to the file
        provider: Embedding provider to use

    Returns:
        Tuple of (success, message)
    """
    try:
        # Read file content
        content = file_path.read_bytes()

        # Upload to document store
        object_name = f"documents/{file_path.name}"
        await document_store.upload_bytes(content, object_name, len(content))
        logger.debug(f"Uploaded to document store: {object_name}")

        # Create document record
        doc = Document(
            title=file_path.stem,
            file_path=object_name,
            file_type=file_path.suffix[1:],
            metadata=DocumentMetadata(
                source=file_path.as_posix(),
                language="vi",  # Default to Vietnamese
            )
        )
        await metadata_store.create_document(doc)
        logger.debug(f"Created document record: {doc.id}")

        # Parse document
        parsed = await DocumentParserFactory.parse(str(file_path), content)
        logger.info(f"Parsed document: {parsed.title}, sections={len(parsed.sections)}, language={parsed.language}")

        # Build tree structure
        sections = tree_builder.build_tree(parsed.sections, doc.id)
        sections = tree_builder.generate_section_summaries(sections)
        await metadata_store.create_sections(sections)
        logger.debug(f"Created {len(sections)} sections in database")

        # Chunk document
        chunks = chunker.chunk_document(sections, doc.id)
        await metadata_store.create_chunks(chunks)
        logger.debug(f"Created {len(chunks)} chunks in database")

        # Index vectors
        result = await index_manager.index_document(doc.id, parsed, sections, chunks, provider=provider)

        if result.success:
            logger.info(f"✓ Indexed: {file_path.name} ({result.sections_indexed} sections, {result.chunks_indexed} chunks)")
            return True, f"Indexed: {file_path.name}"
        else:
            logger.error(f"✗ Failed to index: {file_path.name} - {result.error}")
            return False, f"Failed: {result.error}"

    except Exception as e:
        logger.error(f"✗ Error indexing {file_path.name}: {e}")
        return False, f"Error: {str(e)}"


async def index_directory(
    path: Path,
    workers: int = 4,
    provider: str = "mock",
    pattern: str = "**/*"
) -> dict:
    """
    Index all supported files in a directory.

    Args:
        path: Directory path containing documents
        workers: Number of parallel workers (for future use)
        provider: Embedding provider
        pattern: Glob pattern for finding files

    Returns:
        Dictionary with indexing statistics
    """
    configure_logging(debug=True)
    logger.info(f"Starting bulk indexing from: {path}")

    # Initialize storage
    await init_storage()

    # Initialize vector collections
    from src.storage import vector_store
    await vector_store.initialize_collections()

    stats = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    try:
        # Find all supported files
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.glob(f"{pattern}{ext}"))

        # Remove duplicates and sort
        files = sorted(set(files))

        stats["total"] = len(files)
        logger.info(f"Found {len(files)} files to index")

        if not files:
            logger.warning(f"No supported files found in {path}")
            return stats

        # Index files sequentially (can be parallelized later)
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")

            success, message = await index_file(file_path, provider)

            if success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(message)

        # Get final index stats
        index_stats = await index_manager.get_index_stats()
        logger.info("Final index statistics:", **index_stats)

    finally:
        # Cleanup
        await close_storage()

    return stats


async def create_sample_documents(path: Path) -> None:
    """Create sample markdown documents for testing."""
    logger.info(f"Creating sample documents in: {path}")

    path.mkdir(parents=True, exist_ok=True)

    # Sample Vietnamese document
    vi_doc = """# Hướng Dẫn Sử Dụng Hệ Thống

Chào mừng bạn đến với hệ thống RAG của chúng tôi.

## Tính Năng Chính

Hệ thống của chúng tôi cung cấp các tính năng:

1. **Tìm kiếm thông minh**: Sử dụng AI để tìm kiếm thông tin
2. **Hỗ trợ đa ngôn ngữ**: Việt Nam và Anh
3. **Hiệu suất cao**: Xử lý hàng triệu tài liệu

### Cài Đặt

Để cài đặt hệ thống, bạn cần:

- Python 3.11 trở lên
- Docker và Docker Compose
- API keys cho LLM services

## Hỗ Trợ

Nếu cần hỗ trợ, vui lòng liên hệ qua email.
"""

    # Sample English document
    en_doc = """# Product Documentation

Welcome to our product documentation.

## Getting Started

This guide will help you get started with our product.

### Installation

Follow these steps to install:

1. Download the installer
2. Run the installation script
3. Configure your settings

### Configuration

Edit the config file to customize your setup.

## Features

Our product includes:

- Fast search
- Real-time updates
- Secure data storage

## Support

For support, please contact our team.
"""

    # Sample technical document
    tech_doc = """# API Reference

## Authentication

All API requests require authentication.

### API Key

Generate an API key from the dashboard.

## Endpoints

### Documents

#### List Documents

GET /api/v1/documents

Returns a list of all documents.

#### Create Document

POST /api/v1/documents

Creates a new document.

### Search

#### Search Documents

POST /api/v1/search

Search for documents by content.
"""

    samples = {
        "vi_guide.md": vi_doc,
        "en_guide.md": en_doc,
        "api_reference.md": tech_doc,
    }

    for filename, content in samples.items():
        file_path = path / filename
        file_path.write_text(content, encoding="utf-8")
        logger.debug(f"Created: {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk index documents for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all documents in a directory
  python scripts/index_documents.py --path ./docs

  # Create and index sample documents
  python scripts/index_documents.py --path ./sample_docs --create-samples

  # Use specific embedding provider
  python scripts/index_documents.py --path ./docs --provider openai
        """
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory containing documents to index"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["openai", "local", "mock"],
        help="Embedding provider (default: mock)"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample documents before indexing"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*",
        help="Glob pattern for finding files (default: **/*)"
    )

    args = parser.parse_args()

    path = Path(args.path)

    # Create samples if requested
    if args.create_samples:
        asyncio.run(create_sample_documents(path))

    # Run indexing
    stats = asyncio.run(index_directory(
        path=path,
        workers=args.workers,
        provider=args.provider,
        pattern=args.pattern
    ))

    # Print results
    print("\n" + "=" * 50)
    print("Indexing Results")
    print("=" * 50)
    print(f"Total files:     {stats['total']}")
    print(f"Successful:     {stats['successful']}")
    print(f"Failed:         {stats['failed']}")

    if stats['errors']:
        print("\nErrors:")
        for error in stats['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
