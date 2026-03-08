import json
import os

import numpy as np
import openai
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from sample_data import VECTOR_STORE_CHUNKS

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

mcp = FastMCP("VectorStore RAG Server")

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

chunks: list[dict] = []
embeddings: np.ndarray | None = None


def _get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _initialize_store() -> None:
    global chunks, embeddings
    chunks = VECTOR_STORE_CHUNKS
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _get_embeddings(texts)


_initialize_store()


@mcp.tool()
def semantic_search(query: str, top_k: int = 5) -> str:
    """Search the vector store for chunks semantically similar to the query.

    Args:
        query: Natural language search query.
        top_k: Number of top results to return (default 5).

    Returns:
        JSON string containing ranked results with scores, text, and metadata.
    """
    if embeddings is None or len(chunks) == 0:
        return json.dumps({"error": "Vector store is not initialized."})

    top_k = min(top_k, len(chunks))
    query_embedding = _get_embeddings([query])
    similarities = _cosine_similarity(query_embedding, embeddings).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        chunk = chunks[idx]
        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(similarities[idx]),
        })

    return json.dumps(results, indent=2)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Retrieve a specific document chunk by its ID.

    Args:
        doc_id: The unique identifier of the chunk (e.g. 'chunk_01').

    Returns:
        JSON string with the document's text and metadata, or an error message.
    """
    for chunk in chunks:
        if chunk["id"] == doc_id:
            return json.dumps({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }, indent=2)

    return json.dumps({"error": f"Document '{doc_id}' not found."})


@mcp.tool()
def list_collections() -> str:
    """List available data collections and summary statistics.

    Returns:
        JSON string describing available collections with chunk count,
        sources, and sections.
    """
    if not chunks:
        return json.dumps({"error": "No collections loaded."})

    sources = set()
    sections = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if "source" in meta:
            sources.add(meta["source"])
        if "section" in meta:
            sections.add(meta["section"])

    return json.dumps({
        "collection": "vector_store_chunks",
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "sources": sorted(sources),
        "sections": sorted(sections),
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
