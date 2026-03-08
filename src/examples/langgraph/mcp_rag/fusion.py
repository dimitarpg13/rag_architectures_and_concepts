"""Reciprocal Rank Fusion for merging retrieval results from heterogeneous sources."""

from __future__ import annotations

from collections import defaultdict

from models import FusedContext, RetrievalResult, RetrievedItem


def rrf_fuse(results: list[RetrievalResult], k: int = 60) -> FusedContext:
    rrf_scores: dict[str, float] = defaultdict(float)
    item_map: dict[str, RetrievedItem] = {}

    for result in results:
        sorted_items = sorted(result.items, key=lambda it: it.score, reverse=True)
        for rank, item in enumerate(sorted_items, start=1):
            rrf_scores[item.source_id] += 1.0 / (k + rank)
            if item.source_id not in item_map:
                item_map[item.source_id] = item

    ranked_ids = sorted(rrf_scores, key=lambda sid: rrf_scores[sid], reverse=True)
    ranked_items = [item_map[sid] for sid in ranked_ids]

    top_scores = [rrf_scores[sid] for sid in ranked_ids[: min(5, len(ranked_ids))]]
    overall_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    source_distribution: dict[str, int] = defaultdict(int)
    for item in ranked_items:
        source_distribution[item.source] += 1

    total_tokens = sum(len(item.content) // 4 for item in ranked_items)

    return FusedContext(
        ranked_items=ranked_items,
        overall_score=overall_score,
        source_distribution=dict(source_distribution),
        total_tokens=total_tokens,
    )


def weighted_fuse(
    results: list[RetrievalResult],
    weights: dict[str, float],
    k: int = 60,
) -> FusedContext:
    rrf_scores: dict[str, float] = defaultdict(float)
    item_map: dict[str, RetrievedItem] = {}

    for result in results:
        weight = weights.get(result.source, 1.0)
        sorted_items = sorted(result.items, key=lambda it: it.score, reverse=True)
        for rank, item in enumerate(sorted_items, start=1):
            rrf_scores[item.source_id] += weight / (k + rank)
            if item.source_id not in item_map:
                item_map[item.source_id] = item

    ranked_ids = sorted(rrf_scores, key=lambda sid: rrf_scores[sid], reverse=True)
    ranked_items = [item_map[sid] for sid in ranked_ids]

    top_scores = [rrf_scores[sid] for sid in ranked_ids[: min(5, len(ranked_ids))]]
    overall_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    source_distribution: dict[str, int] = defaultdict(int)
    for item in ranked_items:
        source_distribution[item.source] += 1

    total_tokens = sum(len(item.content) // 4 for item in ranked_items)

    return FusedContext(
        ranked_items=ranked_items,
        overall_score=overall_score,
        source_distribution=dict(source_distribution),
        total_tokens=total_tokens,
    )
