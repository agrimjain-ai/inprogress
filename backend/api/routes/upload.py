"""
FastAPI upload endpoint for document ingestion
Handles file upload, validation, and triggers the full ingestion pipeline
"""

import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import psycopg2
from typing import Optional

from backend.ingestion.pipeline import ingest_document_pipeline
from backend.api.schemas import DocumentUploadResponse, ErrorResponse
from backend.core.config import get_settings
from backend.core.database import get_postgres_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file from FastAPI
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} are supported"
        )
    
    # Check filename
    if not file.filename or len(file.filename) == 0:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    logger.info(f"File validation passed: {file.filename}")


def _save_temp_file(file: UploadFile, upload_dir: str) -> str:
    """
    Save uploaded file temporarily to disk
    
    Args:
        file: Uploaded file
        upload_dir: Directory to save to
        
    Returns:
        Path to saved file
        
    Raises:
        HTTPException: If save fails
    """
    try:
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            contents = file.file.read()
            if len(contents) > MAX_FILE_SIZE:
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            f.write(contents)
        
        logger.info(f"File saved: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")


def _get_document_id(filename: str) -> Optional[int]:
    """
    Query PostgreSQL to get document ID after ingestion
    
    Args:
        filename: Name of the document
        
    Returns:
        Document ID or None if not found
    """
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM documents WHERE filename = %s ORDER BY uploaded_at DESC LIMIT 1",
            (filename,)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to retrieve document ID: {str(e)}")
        return None


def _get_chunk_count(doc_id: int) -> int:
    """
    Query PostgreSQL to count chunks for a document
    
    Args:
        doc_id: Document ID
        
    Returns:
        Number of chunks
    """
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (doc_id,))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to count chunks: {str(e)}")
        return 0


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Upload and ingest a document (PDF or DOCX)
    
    This endpoint:
    1. Validates the file (extension, size)
    2. Saves it temporarily
    3. Parses the document
    4. Chunks the content
    5. Extracts metadata
    6. Generates embeddings
    7. Stores in PostgreSQL, Neo4j, and Qdrant
    8. Returns document ID and ingestion status
    
    Args:
        file: PDF or DOCX file to upload
        
    Returns:
        DocumentUploadResponse with doc_id, filename, doc_type, chunks_created, status
        
    Raises:
        HTTPException: 400 if file invalid, 413 if too large, 500 if processing fails
    """
    logger.info(f"Upload started: {file.filename}")
    
    try:
        # 1. Validate file
        _validate_file(file)
        
        # 2. Save temporarily
        settings = get_settings()
        temp_dir = getattr(settings, "TEMP_UPLOAD_DIR", "/app/temp_uploads")
        file_path = _save_temp_file(file, temp_dir)
        
        # 3. Run ingestion pipeline
        logger.info(f"Starting ingestion pipeline for: {file.filename}")
        doc_id = ingest_document_pipeline(file_path, file.filename)
        logger.info(f"Ingestion completed. Document ID: {doc_id}")
        
        # 4. Query results from PostgreSQL
        chunk_count = _get_chunk_count(doc_id)
        
        # 5. Get document metadata
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_type, status FROM documents WHERE id = %s",
            (doc_id,)
        )
        doc_type, status = cur.fetchone()
        cur.close()
        conn.close()
        
        # 6. Clean up temp file
        try:
            os.remove(file_path)
            logger.info(f"Temp file cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file: {str(e)}")
        
        # 7. Return response
        response = DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            doc_type=doc_type,
            chunks_created=chunk_count,
            status=status,
            message="Document ingested successfully"
        )
        logger.info(f"Upload successful: {file.filename} (ID: {doc_id})")
        return response
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/documents", response_model=list)
async def list_documents():
    """
    List all uploaded documents
    
    Returns:
        List of DocumentListResponse objects
    """
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                d.id,
                d.filename,
                d.doc_type,
                d.uploaded_at,
                d.status,
                COUNT(c.id) as chunks_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.uploaded_at DESC
        """)
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        documents = [
            {
                "doc_id": row[0],
                "filename": row[1],
                "doc_type": row[2],
                "uploaded_at": row[3],
                "status": row[4],
                "chunks_count": row[5] or 0
            }
            for row in results
        ]
        
        logger.info(f"Listed {len(documents)} documents")
        return documents
    
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete a document and all its chunks from PostgreSQL
    Note: Qdrant and Neo4j records still exist but are unreachable
    
    Args:
        doc_id: Document ID to delete
        
    Returns:
        Confirmation message
    """
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        
        # Delete chunks first (foreign key constraint)
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        
        # Delete document
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info(f"Document deleted: {doc_id}")
        return {"message": f"Document {doc_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")