# =============================================================================
# Document Store - MinIO/S3 Object Storage
# =============================================================================
import asyncio
import os
from typing import Optional, BinaryIO
from pathlib import Path
from datetime import datetime, timedelta

from minio import Minio
from minio.deleteobjects import DeleteObject
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentStore:
    """Async wrapper around MinIO/S3 client for document storage."""

    def __init__(self):
        self.client: Optional[Minio] = None
        self.bucket_name = settings.minio_bucket
        self._loop = None

    async def connect(self):
        """Initialize MinIO client."""
        self._loop = asyncio.get_event_loop()
        self.client = Minio(
            endpoint=f"{settings.minio_host}:{settings.minio_port}",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,  # Use HTTP for local development
        )
        # Ensure bucket exists
        await self._ensure_bucket()
        logger.info(
            "Connected to MinIO",
            host=settings.minio_host,
            port=settings.minio_port,
            bucket=self.bucket_name,
        )

    async def disconnect(self):
        """Close MinIO connection."""
        # MinIO client doesn't have explicit close method
        logger.info("Disconnected from MinIO")

    async def _ensure_bucket(self):
        """Ensure bucket exists, create if not."""
        try:
            exists = await self._run_in_executor(
                self.client.bucket_exists, self.bucket_name
            )
            if not exists:
                await self._run_in_executor(
                    self.client.make_bucket, self.bucket_name
                )
                logger.info("Created MinIO bucket", bucket=self.bucket_name)
        except Exception as e:
            logger.error("Failed to ensure bucket", bucket=self.bucket_name, error=str(e))
            raise

    def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous MinIO operation in thread pool."""
        if not self._loop:
            self._loop = asyncio.get_event_loop()
        # Use functools.partial to handle keyword arguments
        from functools import partial
        if kwargs:
            func = partial(func, **kwargs)
        return self._loop.run_in_executor(None, func, *args)

    # =========================================================================
    # File Upload Operations
    # =========================================================================

    async def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload a file to MinIO.

        Args:
            file_path: Local path to file
            object_name: S3 object name (default: basename of file_path)
            metadata: Object metadata

        Returns:
            The object name
        """
        if not object_name:
            object_name = os.path.basename(file_path)

        # Add timestamp prefix to avoid overwrites
        timestamp = datetime.utcnow().strftime("%Y%m%d/%H%M%S")
        object_name = f"{timestamp}/{object_name}"

        try:
            await self._run_in_executor(
                self.client.fput_object,
                self.bucket_name,
                object_name,
                file_path,
                metadata=metadata,
            )
            logger.info(
                "Uploaded file",
                bucket=self.bucket_name,
                object=object_name,
                size=os.path.getsize(file_path),
            )
            return object_name
        except Exception as e:
            logger.error("Failed to upload file", file=file_path, error=str(e))
            raise

    async def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        length: int,
        metadata: Optional[dict] = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload bytes to MinIO.

        Args:
            data: Bytes data to upload
            object_name: S3 object name
            length: Length of data
            metadata: Object metadata
            content_type: MIME type

        Returns:
            The object name
        """
        try:
            from io import BytesIO

            stream = BytesIO(data)
            await self._run_in_executor(
                self.client.put_object,
                self.bucket_name,
                object_name,
                stream,
                length,
                content_type=content_type,
                metadata=metadata,
            )
            logger.info(
                "Uploaded bytes",
                bucket=self.bucket_name,
                object=object_name,
                size=length,
            )
            return object_name
        except Exception as e:
            logger.error("Failed to upload bytes", object=object_name, error=str(e))
            raise

    async def upload_fileobj(
        self,
        fileobj: BinaryIO,
        object_name: str,
        length: int,
        metadata: Optional[dict] = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload a file-like object to MinIO.

        Args:
            fileobj: File-like object to upload
            object_name: S3 object name
            length: Length of data
            metadata: Object metadata
            content_type: MIME type

        Returns:
            The object name
        """
        try:
            await self._run_in_executor(
                self.client.put_object,
                self.bucket_name,
                object_name,
                fileobj,
                length,
                content_type=content_type,
                metadata=metadata,
            )
            logger.info(
                "Uploaded file object",
                bucket=self.bucket_name,
                object=object_name,
                size=length,
            )
            return object_name
        except Exception as e:
            logger.error("Failed to upload file object", object=object_name, error=str(e))
            raise

    # =========================================================================
    # File Download Operations
    # =========================================================================

    async def download_file(self, object_name: str, file_path: str) -> str:
        """
        Download a file from MinIO.

        Args:
            object_name: S3 object name
            file_path: Local destination path

        Returns:
            The file path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

            await self._run_in_executor(
                self.client.fget_object,
                self.bucket_name,
                object_name,
                file_path,
            )
            logger.info(
                "Downloaded file",
                bucket=self.bucket_name,
                object=object_name,
                path=file_path,
            )
            return file_path
        except Exception as e:
            logger.error("Failed to download file", object=object_name, error=str(e))
            raise

    async def download_bytes(self, object_name: str) -> bytes:
        """
        Download bytes from MinIO.

        Args:
            object_name: S3 object name

        Returns:
            Bytes data
        """
        try:
            response = await self._run_in_executor(
                self.client.get_object,
                self.bucket_name,
                object_name,
            )
            data = response.read()
            logger.info(
                "Downloaded bytes",
                bucket=self.bucket_name,
                object=object_name,
                size=len(data),
            )
            return data
        except Exception as e:
            logger.error("Failed to download bytes", object=object_name, error=str(e))
            raise

    async def get_file_url(
        self,
        object_name: str,
        expires: int = 3600,
    ) -> str:
        """
        Get presigned URL for file download.

        Args:
            object_name: S3 object name
            expires: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        try:
            url = await self._run_in_executor(
                self.client.presigned_get_object,
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expires),
            )
            logger.debug(
                "Generated presigned URL",
                bucket=self.bucket_name,
                object=object_name,
                expires=expires,
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned URL", object=object_name, error=str(e))
            raise

    # =========================================================================
    # File Management Operations
    # =========================================================================

    async def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO.

        Args:
            object_name: S3 object name

        Returns:
            True if deleted
        """
        try:
            await self._run_in_executor(
                self.client.remove_object,
                self.bucket_name,
                object_name,
            )
            logger.info(
                "Deleted file",
                bucket=self.bucket_name,
                object=object_name,
            )
            return True
        except Exception as e:
            logger.error("Failed to delete file", object=object_name, error=str(e))
            return False

    async def delete_files(self, object_names: list[str]) -> int:
        """
        Delete multiple files from MinIO.

        Args:
            object_names: List of S3 object names

        Returns:
            Number of files deleted
        """
        try:
            delete_objects = [DeleteObject(name) for name in object_names]
            errors = await self._run_in_executor(
                self.client.remove_objects,
                self.bucket_name,
                delete_objects,
            )
            deleted_count = len(object_names) - len(list(errors))
            logger.info(
                "Deleted files",
                bucket=self.bucket_name,
                count=deleted_count,
                errors=len(list(errors)),
            )
            return deleted_count
        except Exception as e:
            logger.error("Failed to delete files", error=str(e))
            return 0

    async def file_exists(self, object_name: str) -> bool:
        """
        Check if file exists in MinIO.

        Args:
            object_name: S3 object name

        Returns:
            True if exists
        """
        try:
            await self._run_in_executor(
                self.client.stat_object,
                self.bucket_name,
                object_name,
            )
            return True
        except Exception:
            return False

    async def get_file_info(self, object_name: str) -> Optional[dict]:
        """
        Get file metadata from MinIO.

        Args:
            object_name: S3 object name

        Returns:
            File metadata dict or None
        """
        try:
            stat = await self._run_in_executor(
                self.client.stat_object,
                self.bucket_name,
                object_name,
            )
            return {
                "name": object_name,
                "size": stat.size,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type,
                "etag": stat.etag,
            }
        except Exception as e:
            logger.warning("Failed to get file info", object=object_name, error=str(e))
            return None

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int = 1000,
    ) -> list[dict]:
        """
        List files in MinIO bucket.

        Args:
            prefix: Object name prefix
            recursive: List recursively
            limit: Maximum number of objects

        Returns:
            List of file info dicts
        """
        try:
            objects = await self._run_in_executor(
                self.client.list_objects,
                self.bucket_name,
                prefix=prefix,
                recursive=recursive,
            )
            files = []
            for i, obj in enumerate(objects):
                if i >= limit:
                    break
                files.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                })
            logger.info(
                "Listed files",
                bucket=self.bucket_name,
                prefix=prefix,
                count=len(files),
            )
            return files
        except Exception as e:
            logger.error("Failed to list files", error=str(e))
            return []

    async def get_bucket_size(self) -> int:
        """Get total size of all objects in bucket."""
        try:
            objects = await self._run_in_executor(
                self.client.list_objects,
                self.bucket_name,
                recursive=True,
            )
            total_size = sum(obj.size for obj in objects)
            return total_size
        except Exception as e:
            logger.error("Failed to get bucket size", error=str(e))
            return 0

    async def clear_bucket(self) -> int:
        """
        Delete all objects from bucket.

        Returns:
            Number of objects deleted
        """
        try:
            objects = await self._run_in_executor(
                self.client.list_objects,
                self.bucket_name,
                recursive=True,
            )
            object_names = [obj.object_name for obj in objects]
            if object_names:
                return await self.delete_files(object_names)
            return 0
        except Exception as e:
            logger.error("Failed to clear bucket", error=str(e))
            return 0

    async def health_check(self) -> dict:
        """Check document store health and return statistics."""
        try:
            # Check bucket exists
            exists = await self._run_in_executor(
                self.client.bucket_exists,
                self.bucket_name,
            )
            if not exists:
                return {
                    "status": "error",
                    "error": f"Bucket {self.bucket_name} does not exist",
                }

            # Count objects
            objects = await self._run_in_executor(
                self.client.list_objects,
                self.bucket_name,
                recursive=True,
            )
            object_count = len(list(objects))

            # Get bucket size
            total_size = await self.get_bucket_size()

            return {
                "status": "healthy",
                "bucket": self.bucket_name,
                "object_count": object_count,
                "total_size": total_size,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Singleton instance
document_store = DocumentStore()
