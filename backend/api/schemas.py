"""
Pydantic schemas for GST Copilot API
Handles request/response validation and serialization
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload and ingestion"""
    doc_id: int = Field(..., description="Unique document ID in PostgreSQL")
    filename: str = Field(..., description="Original filename")
    doc_type: str = Field(..., description="Detected document type (judgment/circular/notification/act/unknown)")
    chunks_created: int = Field(..., description="Number of chunks created")
    status: str = Field(..., description="Status of ingestion (completed/processing/failed)")
    message: str = Field(..., description="Human-readable status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": 1,
                "filename": "sample.pdf",
                "doc_type": "judgment",
                "chunks_created": 42,
                "status": "completed",
                "message": "Document ingested successfully"
            }
        }


class DocumentMetadataResponse(BaseModel):
    """Extracted metadata from document"""
    doc_type: str
    court_name: Optional[str] = None
    case_number: Optional[str] = None
    date: Optional[str] = None
    circular_number: Optional[str] = None
    notification_number: Optional[str] = None
    parties: List[str] = Field(default_factory=list)
    section_refs: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_type": "judgment",
                "court_name": "Supreme Court of India",
                "case_number": "W.P. No. 12345/2023",
                "date": "15-06-2023",
                "circular_number": None,
                "notification_number": None,
                "parties": ["Union of India", "ABC Ltd"],
                "section_refs": ["Section 16(4)", "Rule 89(1)"]
            }
        }


class DocumentListResponse(BaseModel):
    """Response for listing uploaded documents"""
    doc_id: int
    filename: str
    doc_type: str
    uploaded_at: datetime
    status: str
    chunks_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": 1,
                "filename": "judgment_2023.pdf",
                "doc_type": "judgment",
                "uploaded_at": "2024-01-15T10:30:00",
                "status": "completed",
                "chunks_count": 42
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid file type",
                "detail": "Only PDF and DOCX files are supported",
                "status_code": 400
            }
        }