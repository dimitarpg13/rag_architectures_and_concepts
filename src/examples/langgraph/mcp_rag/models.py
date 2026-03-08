"""Shared data models for the MCP-based distributed RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from typing_extensions import TypedDict


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class QueryClassification:
    query_type: QueryType
    domains: list[str]
    complexity: ComplexityLevel
    requires_multi_hop: bool
    extracted_entities: list[str] = field(default_factory=list)


@dataclass
class ServerCall:
    server_name: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ExecutionPlan:
    parallel_groups: list[list[ServerCall]]
    timeout_ms: int = 30_000


@dataclass
class RetrievedItem:
    content: str
    score: float
    source: str
    source_id: str
    item_type: str  # "chunk", "triple", "row"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    source: str
    items: list[RetrievedItem]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedContext:
    ranked_items: list[RetrievedItem]
    overall_score: float
    source_distribution: dict[str, int]
    total_tokens: int = 0


@dataclass
class QualityAssessment:
    sufficient: bool
    relevance_score: float
    reason: str


class OrchestratorState(TypedDict, total=False):
    """LangGraph state flowing through the orchestration graph."""
    query: str
    classification: dict  # serialized QueryClassification
    execution_plan: dict  # serialized ExecutionPlan
    retrieval_results: list[dict]  # serialized list[RetrievalResult]
    fused_context: dict  # serialized FusedContext
    quality_assessment: dict  # serialized QualityAssessment
    iteration_count: int
    final_response: str
    error: str
