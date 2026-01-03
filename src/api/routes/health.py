# =============================================================================
# Health Route - System health check endpoints
# =============================================================================
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import time

from src.storage import vector_store, metadata_store, cache_store, document_store
from src.utils.logging import get_logger
from sqlalchemy import text

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


class ServiceStatus(BaseModel):
    """Health status for a single service."""
    name: str
    status: str
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall system health response."""
    status: str
    version: str
    services: list[ServiceStatus]


async def check_qdrant() -> bool:
    """Check Qdrant vector store health."""
    await vector_store.client.get_collections()


async def check_postgres() -> bool:
    """Check PostgreSQL metadata store health."""
    async with metadata_store.session() as session:
        await session.execute(text("SELECT 1"))


async def check_redis() -> bool:
    """Check Redis cache store health."""
    await cache_store.redis.ping()


async def check_minio() -> bool:
    """Check MinIO document store health."""
    # bucket_exists is sync, wrap in run_in_executor
    loop = document_store._loop
    await loop.run_in_executor(
        None,
        document_store.client.bucket_exists,
        document_store.bucket_name
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of all system components.

    Returns status for each service:
    - qdrant: Vector database
    - postgresql: Metadata database
    - redis: Cache
    - minio: Document storage
    """
    services = []
    overall_healthy = True

    # Check each service
    for name, check_func in [
        ("qdrant", check_qdrant),
        ("postgresql", check_postgres),
        ("redis", check_redis),
        ("minio", check_minio)
    ]:
        try:
            start = time.time()
            await check_func()
            latency = (time.time() - start) * 1000
            services.append(ServiceStatus(
                name=name,
                status="healthy",
                latency_ms=round(latency, 2)
            ))
        except Exception as e:
            overall_healthy = False
            services.append(ServiceStatus(
                name=name,
                status="unhealthy",
                message=str(e)
            ))

    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        version="0.1.0",
        services=services
    )


@router.get("/ready")
async def readiness():
    """
    Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic.
    This checks if all storage connections are active.
    """
    # Check if all services are connected
    all_ready = all([
        vector_store.client is not None,
        metadata_store.engine is not None,
        cache_store.redis is not None,
        document_store.client is not None
    ])

    if all_ready:
        return {"ready": True}
    else:
        from fastapi import status
        raise status.HTTP_503_SERVICE_UNAVAILABLE


@router.get("/live")
async def liveness():
    """
    Kubernetes liveness probe.

    Returns 200 if the service is alive.
    This is a simple check that the process is running.
    """
    return {"alive": True}


@router.get("/health/detailed")
async def detailed_health():
    """
    Detailed health check with additional information.

    Includes collection stats, cache stats, and storage metrics.
    """
    health_data = await health_check()

    # Get additional stats
    detailed_info = {}

    try:
        # Vector store stats
        collections = await vector_store.client.get_collections()
        detailed_info["vector_store"] = {
            "collections": [
                {
                    "name": c.name,
                    "points_count": (await vector_store.client.count(c.name)).count
                }
                for c in collections.collections
            ]
        }
    except Exception as e:
        detailed_info["vector_store"] = {"error": str(e)}

    try:
        # Metadata store stats
        detailed_info["metadata_store"] = await metadata_store.health_check()
    except Exception as e:
        detailed_info["metadata_store"] = {"error": str(e)}

    try:
        # Cache store stats
        detailed_info["cache_store"] = await cache_store.health_check()
    except Exception as e:
        detailed_info["cache_store"] = {"error": str(e)}

    try:
        # Document store stats
        detailed_info["document_store"] = await document_store.health_check()
    except Exception as e:
        detailed_info["document_store"] = {"error": str(e)}

    return {
        "status": health_data.status,
        "version": health_data.version,
        "services": health_data.services,
        "detailed": detailed_info
    }
