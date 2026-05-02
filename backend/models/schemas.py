from pydantic import BaseModel
from datetime import datetime


class QueryRequest(BaseModel):
    """Request body for RAG query."""
    question: str
    top_k: int | None = None


class SourceChunk(BaseModel):
    """A single source chunk returned with the answer."""
    text: str
    document_name: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    """Response body for RAG query."""
    answer: str
    sources: list[SourceChunk]
    model: str


class UploadResponse(BaseModel):
    """Response body after uploading a PDF."""
    document_id: str
    filename: str
    num_chunks: int
    message: str


class DocumentInfo(BaseModel):
    """Info about an indexed document."""
    document_id: str
    filename: str
    num_chunks: int
    uploaded_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    qdrant_connected: bool
    embedding_model_loaded: bool
