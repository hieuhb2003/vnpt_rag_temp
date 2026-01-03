# =============================================================================
# Documents Route - Document management and indexing
# =============================================================================
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from uuid import UUID

from src.storage import metadata_store, document_store
from src.indexing.document_parser import parser_factory, ParsedDocument
from src.indexing.tree_builder import tree_builder
from src.indexing.chunker import chunker
from src.indexing.index_manager import index_manager
from src.models.document import Document, DocumentMetadata, DocumentStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    title: str
    status: str
    message: str


class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    title: str
    file_type: str
    status: str
    section_count: int
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    """Response for document list."""
    documents: list[DocumentInfo]
    total: int
    limit: int
    offset: int


async def index_document_task(doc_id: UUID, content: bytes, filename: str):
    """
    Background task to index a document.

    Process:
    1. Update status to "processing"
    2. Parse document
    3. Build tree structure
    4. Create chunks
    5. Store sections and chunks
    6. Index in vector store
    7. Update status to "indexed" or "failed"
    """
    try:
        logger.info(f"Starting background indexing for {doc_id}")

        await metadata_store.update_document_status(doc_id, "processing")

        # Parse document
        parsed: ParsedDocument = await parser_factory.parse(filename, content)

        # Build tree structure
        sections = tree_builder.build_tree(parsed.sections, doc_id)
        await metadata_store.create_sections(sections)

        # Create chunks
        chunks = chunker.chunk_document(sections, doc_id)
        await metadata_store.create_chunks(chunks)

        # Index in vector store
        result = await index_manager.index_document(
            doc_id, parsed, sections, chunks
        )

        if result.success:
            await metadata_store.update_document_status(doc_id, "indexed")
            logger.info(
                f"Indexing complete for {doc_id}",
                sections=result.sections_indexed,
                chunks=result.chunks_indexed
            )
        else:
            await metadata_store.update_document_status(doc_id, "failed")
            logger.error(f"Indexing failed for {doc_id}: {result.error}")

    except Exception as e:
        logger.error(f"Indexing failed for {doc_id}: {e}", exc_info=True)
        await metadata_store.update_document_status(doc_id, "failed")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """
    Upload and index a new document.

    Process:
    1. Upload file to MinIO
    2. Create document record in database
    3. Start background indexing task
    """
    logger.info(f"Document upload: {file.filename}")

    try:
        content = await file.read()

        # Upload to MinIO
        object_name = f"documents/{file.filename}"
        await document_store.upload_bytes(
            data=content,
            object_name=object_name,
            length=len(content),
            content_type=file.content_type or "application/octet-stream"
        )

        # Create document record
        doc = Document(
            title=file.filename,
            file_path=object_name,
            file_type=file.filename.split(".")[-1] if "." in file.filename else "unknown",
            metadata=DocumentMetadata(
                category=category,
                tags=tags.split(",") if tags else []
            ),
            status=DocumentStatus.PENDING
        )
        await metadata_store.create_document(doc)

        # Start background indexing
        background_tasks.add_task(
            index_document_task,
            doc.id,
            content,
            file.filename
        )

        return DocumentUploadResponse(
            document_id=str(doc.id),
            title=doc.title,
            status="pending",
            message="Document uploaded. Indexing started in background."
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all indexed documents with optional filters."""
    docs = await metadata_store.list_documents(
        limit=limit,
        offset=offset,
        status=status
    )
    total = await metadata_store.count_documents(status=status)

    documents = []
    for d in docs:
        sections = await metadata_store.get_sections_by_document(d.id)
        documents.append(DocumentInfo(
            id=str(d.id),
            title=d.title,
            file_type=d.file_type,
            status=d.status.value,
            section_count=len(sections),
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat()
        ))

    return DocumentListResponse(
        documents=documents,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Get document details by ID."""
    try:
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await metadata_store.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = await metadata_store.get_sections_by_document(doc_uuid)

    return DocumentInfo(
        id=str(doc.id),
        title=doc.title,
        file_type=doc.file_type,
        status=doc.status.value,
        section_count=len(sections),
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat()
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    """
    Delete a document and all its indexed data.

    Process:
    1. Verify document exists
    2. Start background deletion of vector indices
    3. Return immediately
    """
    try:
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await metadata_store.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Start background deletion
    background_tasks.add_task(index_manager.delete_document_index, doc_uuid)

    logger.info(f"Document deletion started: {doc_id}")
    return {"message": "Document deletion started", "document_id": doc_id}


@router.post("/{doc_id}/reindex")
async def reindex_document(doc_id: str, background_tasks: BackgroundTasks):
    """
    Trigger reindexing of a document.

    Process:
    1. Verify document exists
    2. Download file from MinIO
    3. Start background reindexing task
    """
    try:
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await metadata_store.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        content = await document_store.download_file(doc.file_path)
        filename = doc.file_path.split("/")[-1]

        # Start background indexing
        background_tasks.add_task(
            index_document_task,
            doc_uuid,
            content,
            filename
        )

        logger.info(f"Reindexing started: {doc_id}")
        return {"message": "Reindexing started", "document_id": doc_id}

    except Exception as e:
        logger.error(f"Failed to download document for reindexing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download document: {str(e)}"
        )


@router.get("/{doc_id}/sections")
async def get_document_sections(doc_id: str):
    """Get all sections for a document."""
    try:
        doc_uuid = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = await metadata_store.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = await metadata_store.get_sections_by_document(doc_uuid)

    return {
        "document_id": doc_id,
        "sections": [
            {
                "id": str(s.id),
                "heading": s.heading,
                "level": s.level,
                "section_path": s.section_path,
                "summary": s.summary
            }
            for s in sections
        ]
    }


@router.get("/health")
async def documents_health():
    """Health check for documents endpoint."""
    return {"status": "healthy", "service": "documents"}
