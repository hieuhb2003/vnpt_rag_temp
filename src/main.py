# =============================================================================
# Enterprise RAG System - Main Application
# =============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import get_settings
from src.utils.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    configure_logging(settings.debug)
    logger.info("Starting RAG System", version="0.1.0")

    # Initialize connections here (will be implemented in TODO-02)
    yield

    # Cleanup connections here
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
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


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
