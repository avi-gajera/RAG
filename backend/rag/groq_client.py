from groq import Groq
from backend.config import get_settings
import logging

logger = logging.getLogger(__name__)


def _build_rag_prompt(question: str, context_chunks: list[dict]) -> list[dict]:
    """Build the message list for RAG with context."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['document_name']}, Chunk {chunk['chunk_index']}]\n{chunk['text']}"
        for chunk in context_chunks
    )

    system_message = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Use the context below to answer the user's question accurately. "
        "If the context doesn't contain enough information to answer the question, "
        "say so clearly and provide what you can based on the available context.\n\n"
        "CONTEXT:\n"
        f"{context_text}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Groq with RAG context."""
    settings = get_settings()

    client = Groq(api_key=settings.groq_api_key)
    messages = _build_rag_prompt(question, context_chunks)

    logger.info(f"Sending query to Groq ({settings.groq_model}) with {len(context_chunks)} context chunks")

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=4096,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Collect streamed response
    answer_parts = []
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            answer_parts.append(content)

    answer = "".join(answer_parts)
    logger.info(f"Generated answer: {len(answer)} characters")
    return answer


def generate_answer_no_context(question: str) -> str:
    """Generate an answer using Groq without any RAG context (fallback)."""
    settings = get_settings()

    client = Groq(api_key=settings.groq_api_key)

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. No documents have been uploaded yet, so answer based on your general knowledge.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_completion_tokens=4096,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer_parts = []
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            answer_parts.append(content)

    return "".join(answer_parts)
