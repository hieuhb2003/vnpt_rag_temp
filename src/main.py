# =============================================================================
# Enterprise RAG System - Main Application
# =============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import get_settings
from src.utils.logging import configure_logging, get_logger
from src.storage import init_storage, close_storage, vector_store, metadata_store, cache_store, document_store

# API Routes
from src.api.routes import query, documents, health

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

# API v1 routes
app.include_router(query.router, prefix="/api/v1")
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
            "api": {
                "query": "/api/v1/query",
                "documents": "/api/v1/documents"
            }
        }
    }
