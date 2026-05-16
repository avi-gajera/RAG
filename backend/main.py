from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DocumentInfo,
    HealthResponse,
    SourceChunk,
)
from backend.rag import pdf_processor, vector_store, groq_client, embeddings
import uuid
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with Groq, Qdrant, and FastAPI",
    version="1.0.0",
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    try:
        vector_store.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning(f"Qdrant not available at startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all services."""
    return HealthResponse(
        status="healthy",
        qdrant_connected=vector_store.is_connected(),
        embedding_model_loaded=embeddings.is_model_loaded(),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # Read file contents
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Save file to disk
        settings = get_settings()
        file_path = os.path.join(settings.upload_dir, f"{document_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Process PDF: extract text and chunk
        chunks = pdf_processor.process_pdf(file_bytes)

        # Index chunks into Qdrant
        num_chunks = vector_store.add_document(document_id, file.filename, chunks)

        logger.info(f"Successfully uploaded and indexed '{file.filename}' ({num_chunks} chunks)")

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            num_chunks=num_chunks,
            message=f"Successfully indexed {num_chunks} chunks from '{file.filename}'",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query indexed documents using RAG."""
    settings = get_settings()

    try:
        # Search for relevant chunks
        results = vector_store.search(request.question, request.top_k)

        if not results:
            # No documents indexed or no relevant results
            answer = groq_client.generate_answer_no_context(request.question)
            return QueryResponse(
                answer=answer,
                sources=[],
                model=settings.groq_model,
            )

        # Generate answer using Groq with context
        answer = groq_client.generate_answer(request.question, results)

        sources = [
            SourceChunk(
                text=r["text"][:500],  # Truncate for response
                document_name=r["document_name"],
                chunk_index=r["chunk_index"],
                score=r["score"],
            )
            for r in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=settings.groq_model,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    try:
        docs = vector_store.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks from the index."""
    try:
        # Delete from Qdrant
        vector_store.delete_document(document_id)

        # Delete file from disk
        settings = get_settings()
        for f in os.listdir(settings.upload_dir):
            if f.startswith(document_id):
                os.remove(os.path.join(settings.upload_dir, f))
                break

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
