# =============================================================================
# Storage Package
# =============================================================================
from src.storage.vector_store import vector_store
from src.storage.metadata_store import metadata_store
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def init_storage():
    """Initialize all storage connections."""
    logger.info("Initializing storage connections...")

    await vector_store.connect()
    await vector_store.initialize_collections()

    await metadata_store.connect()
    await metadata_store.init_tables()

    logger.info("All storage connections initialized")


async def close_storage():
    """Close all storage connections."""
    logger.info("Closing storage connections...")

    await vector_store.disconnect()
    await metadata_store.disconnect()

    logger.info("All storage connections closed")


__all__ = [
    "vector_store",
    "metadata_store",
    "init_storage",
    "close_storage",
]
