"""Embedding-based quality gate for evaluating fused retrieval context."""

from __future__ import annotations

import numpy as np
import openai
from dotenv import load_dotenv

from models import FusedContext, QualityAssessment

load_dotenv()

_client = openai.AsyncOpenAI()
_EMBED_MODEL = "text-embedding-3-small"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


async def _embed(texts: list[str]) -> list[list[float]]:
    response = await _client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


async def evaluate_quality(
    fused_context: FusedContext,
    query: str,
    threshold: float = 0.45,
) -> QualityAssessment:
    top_items = fused_context.ranked_items[:5]
    context_text = "\n\n".join(item.content for item in top_items)

    embeddings = await _embed([query, context_text])
    query_vec = np.array(embeddings[0])
    context_vec = np.array(embeddings[1])

    score = _cosine_similarity(query_vec, context_vec)
    sufficient = score >= threshold

    if sufficient:
        reason = (
            f"Context is relevant (cosine similarity {score:.3f} >= {threshold}). "
            f"Top {len(top_items)} items cover the query adequately."
        )
    else:
        reason = (
            f"Context lacks sufficient relevance (cosine similarity {score:.3f} < {threshold}). "
            f"Retrieved items may not address the query directly."
        )

    return QualityAssessment(
        sufficient=sufficient,
        relevance_score=score,
        reason=reason,
    )
