# =============================================================================
# API Routes Package
# =============================================================================
from src.api.routes.query import router as query_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router

__all__ = [
    "query",
    "documents",
    "health",
    "query_router",
    "documents_router",
    "health_router",
]
