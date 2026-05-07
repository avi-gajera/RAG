from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from backend.config import get_settings
from backend.rag.embeddings import embed_texts, embed_query
import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    return _client


def ensure_collection():
    """Create the collection if it doesn't exist."""
    settings = get_settings()
    client = _get_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in collections:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.collection_name}")
    else:
        logger.info(f"Collection already exists: {settings.collection_name}")


def add_document(document_id: str, filename: str, chunks: list[str]) -> int:
    """Embed and store document chunks in Qdrant."""
    settings = get_settings()
    client = _get_client()
    ensure_collection()

    # Generate embeddings for all chunks
    embeddings = embed_texts(chunks)

    # Create points with metadata
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.collection_name, points=batch)

    logger.info(f"Indexed {len(points)} chunks for document '{filename}' (id={document_id})")
    return len(points)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Search for similar chunks using a text query."""
    settings = get_settings()
    client = _get_client()
    ensure_collection()

    k = top_k or settings.top_k
    query_embedding = embed_query(query)

    results = client.query_points(
        collection_name=settings.collection_name,
        query=query_embedding,
        limit=k,
        with_payload=True,
    )

    return [
        {
            "text": point.payload.get("text", ""),
            "document_name": point.payload.get("filename", "unknown"),
            "chunk_index": point.payload.get("chunk_index", 0),
            "score": point.score,
        }
        for point in results.points
    ]


def list_documents() -> list[dict]:
    """List all unique documents stored in the collection."""
    settings = get_settings()
    client = _get_client()
    ensure_collection()

    # Scroll through all points and collect unique documents
    documents = {}
    offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=settings.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = scroll_result

        for point in points:
            doc_id = point.payload.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": point.payload.get("filename", "unknown"),
                    "num_chunks": 0,
                    "uploaded_at": point.payload.get("uploaded_at", ""),
                }
            if doc_id:
                documents[doc_id]["num_chunks"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return list(documents.values())


def delete_document(document_id: str) -> bool:
    """Delete all chunks belonging to a document."""
    settings = get_settings()
    client = _get_client()

    client.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted document: {document_id}")
    return True


def is_connected() -> bool:
    """Check if Qdrant is reachable."""
    try:
        client = _get_client()
        client.get_collections()
        return True
    except Exception:
        return False
