# =============================================================================
# Enterprise RAG System - Main Application
# =============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import get_settings
from src.utils.logging import configure_logging, get_logger
from src.storage import init_storage, close_storage, vector_store, metadata_store, cache_store, document_store

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    configure_logging(settings.debug)
    logger.info("Starting RAG System", version="0.1.0")

    # Initialize storage connections
    await init_storage()

    yield

    # Cleanup connections
    await close_storage()
    logger.info("Shutting down RAG System")


app = FastAPI(
    title="Enterprise RAG System",
    description="Multi-agent RAG with tree-structured document indexing",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint with storage status."""
    # Vector store status
    vector_store_status = "unknown"
    collections = []
    try:
        collections_response = await vector_store.client.get_collections()
        existing = {c.name for c in collections_response.collections}

        for name, collection_name in vector_store.COLLECTIONS.items():
            if collection_name in existing:
                try:
                    count_info = await vector_store.client.count(collection_name)
                    collections.append({
                        "name": name,
                        "collection": collection_name,
                        "points": count_info.count,
                        "status": "ok"
                    })
                except Exception as e:
                    collections.append({"name": name, "collection": collection_name, "status": "error", "error": str(e)})
            else:
                collections.append({"name": name, "collection": collection_name, "status": "missing"})

        vector_store_status = "healthy"
    except Exception as e:
        vector_store_status = f"error: {str(e)}"
        collections = []

    # Metadata store status
    metadata_store_status = "unknown"
    metadata_stats = {}
    try:
        metadata_stats = await metadata_store.health_check()
        metadata_store_status = metadata_stats.pop("status", "healthy")
    except Exception as e:
        metadata_store_status = f"error: {str(e)}"

    # Cache store status
    cache_store_status = "unknown"
    cache_stats = {}
    try:
        cache_stats = await cache_store.health_check()
        cache_store_status = cache_stats.pop("status", "healthy")
    except Exception as e:
        cache_store_status = f"error: {str(e)}"

    # Document store status
    document_store_status = "unknown"
    document_stats = {}
    try:
        document_stats = await document_store.health_check()
        document_store_status = document_stats.pop("status", "healthy")
    except Exception as e:
        document_store_status = f"error: {str(e)}"

    # Overall health
    is_healthy = all([
        vector_store_status == "healthy",
        metadata_store_status == "healthy",
        cache_store_status == "healthy",
        document_store_status == "healthy",
    ])

    return {
        "status": "healthy" if is_healthy else "degraded",
        "version": "0.1.0",
        "storage": {
            "vector_store": {
                "status": vector_store_status,
                "collections": collections
            },
            "metadata_store": {
                "status": metadata_store_status,
                **metadata_stats
            },
            "cache_store": {
                "status": cache_store_status,
                **cache_stats
            },
            "document_store": {
                "status": document_store_status,
                **document_stats
            }
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise RAG System",
        "docs": "/docs",
        "version": "0.1.0"
    }


# Include routers (will be implemented in TODO-06)
# from src.api.routes import query, documents, health
# app.include_router(query.router, prefix="/api/v1", tags=["query"])
# app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
# app.include_router(health.router, prefix="/api/v1", tags=["health"])
