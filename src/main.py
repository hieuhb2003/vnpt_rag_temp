# =============================================================================
# Enterprise RAG System - Main Application
# =============================================================================
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from src.config.settings import get_settings
from src.utils.logging import configure_logging, get_logger
from src.storage import init_storage, close_storage

# API Routes
from src.api.routes import query, documents, health
from src.api.routes.query import QueryInput, process_query

# Middleware
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware

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

# ============================================================================
# Middleware (order matters - last added = first executed)
# ============================================================================

# Request logging - logs all requests with timing
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting - token bucket algorithm (60 req/min, burst 10)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=60,
    burst_size=10
)

# CORS - allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Routes
# ============================================================================

# Health check endpoints (no prefix for k8s compatibility)
app.include_router(health.router)

# Query endpoint at root level (for convenience)
app.include_router(query.router, prefix="/api/v1")

# Documents endpoints
app.include_router(documents.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise RAG System",
        "docs": "/docs",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "live": "/live",
            "query": "/query",  # Root level query endpoint
            "api": {
                "query": "/api/v1/query",
                "documents": "/api/v1/documents"
            }
        }
    }


# Root level query endpoint (convenience alias)
@app.post("/query")
async def query_root(input: QueryInput, background_tasks: BackgroundTasks):
    """
    Process a query (root level endpoint).

    This is a convenience alias for /api/v1/query.
    """
    return await process_query(input, background_tasks)
