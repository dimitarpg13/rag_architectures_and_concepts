"""LangGraph orchestrator for MCP-based distributed RAG.

Builds and runs a StateGraph that:
  1. Classifies the query
  2. Plans which MCP servers to call
  3. Fans out retrieval in parallel via MCP tool calls
  4. Fuses results with Reciprocal Rank Fusion
  5. Evaluates quality and optionally reformulates
  6. Generates a final answer with citations
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path

import openai
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fusion import rrf_fuse
from models import (
    ComplexityLevel,
    ExecutionPlan,
    FusedContext,
    OrchestratorState,
    QualityAssessment,
    QueryClassification,
    QueryType,
    RetrievalResult,
    RetrievedItem,
    ServerCall,
)
from quality_gate import evaluate_quality

load_dotenv()

MAX_REFINEMENT_ITERATIONS = 2

SERVER_DIR = Path(__file__).parent

SERVER_CONFIGS = {
    "vector_store": {
        "script": "vector_store_server.py",
        "capabilities": ["semantic_search", "factual", "analytical"],
        "description": "Semantic search over text chunks about LLM agents",
    },
    "knowledge_graph": {
        "script": "knowledge_graph_server.py",
        "capabilities": ["graph_query", "relationships", "entities"],
        "description": "Entity-relationship graph of LLM agent concepts",
    },
    "structured_db": {
        "script": "structured_db_server.py",
        "capabilities": ["sql_query", "structured", "papers", "techniques"],
        "description": "Structured database of AI techniques and papers",
    },
}


# ---------------------------------------------------------------------------
# MCP client management
# ---------------------------------------------------------------------------

class MCPClientPool:
    """Manages MCP client sessions to multiple servers."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, list[dict]] = {}

    async def connect(self, name: str, script_path: str) -> None:
        env = {**os.environ}
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[script_path],
            env=env,
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        tools_response = await session.list_tools()
        self._sessions[name] = session
        self._tools[name] = [
            {"name": t.name, "description": t.description}
            for t in tools_response.tools
        ]

    async def connect_all(self) -> None:
        for name, config in SERVER_CONFIGS.items():
            script = str(SERVER_DIR / config["script"])
            await self.connect(name, script)

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        text_parts = [
            block.text for block in result.content
            if hasattr(block, "text")
        ]
        raw = "\n".join(text_parts)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_text": raw}

    def get_tools(self, server_name: str) -> list[dict]:
        return self._tools.get(server_name, [])

    async def close(self) -> None:
        await self._exit_stack.aclose()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

_llm_client = openai.AsyncOpenAI()
_pool: MCPClientPool | None = None


async def _llm_call(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    response = await _llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


async def classify_query(state: OrchestratorState) -> dict:
    query = state["query"]

    system = (
        "You are a query classifier for a RAG system with three knowledge sources:\n"
        "1. vector_store: text chunks about LLM agents (good for factual/explanatory queries)\n"
        "2. knowledge_graph: entity-relationship triples (good for relationship/connection queries)\n"
        "3. structured_db: database of techniques and papers (good for structured/comparative queries)\n\n"
        "Classify the query and respond ONLY with valid JSON (no markdown):\n"
        '{"query_type": "factual|analytical|multi_hop|comparative",\n'
        ' "domains": ["vector_store", "knowledge_graph", "structured_db"],\n'
        ' "complexity": "simple|moderate|complex",\n'
        ' "requires_multi_hop": true/false,\n'
        ' "extracted_entities": ["entity1", "entity2"]}\n\n'
        "Include only relevant domains. Most queries need 1-2 sources."
    )

    raw = await _llm_call(system, query)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "query_type": "factual",
            "domains": ["vector_store"],
            "complexity": "simple",
            "requires_multi_hop": False,
            "extracted_entities": [],
        }

    classification = QueryClassification(
        query_type=QueryType(data.get("query_type", "factual")),
        domains=data.get("domains", ["vector_store"]),
        complexity=ComplexityLevel(data.get("complexity", "simple")),
        requires_multi_hop=data.get("requires_multi_hop", False),
        extracted_entities=data.get("extracted_entities", []),
    )

    return {"classification": asdict(classification), "iteration_count": 0}


async def plan_routes(state: OrchestratorState) -> dict:
    classification = state["classification"]
    domains = classification["domains"]
    entities = classification.get("extracted_entities", [])
    query = state["query"]

    parallel_group: list[dict] = []

    for domain in domains:
        if domain == "vector_store":
            parallel_group.append(asdict(ServerCall(
                server_name="vector_store",
                tool_name="semantic_search",
                arguments={"query": query, "top_k": 5},
            )))
        elif domain == "knowledge_graph":
            entity = entities[0] if entities else query.split()[0]
            parallel_group.append(asdict(ServerCall(
                server_name="knowledge_graph",
                tool_name="graph_query",
                arguments={"entity": entity, "depth": 2},
            )))
        elif domain == "structured_db":
            keyword = entities[0] if entities else query.split()[-1]
            parallel_group.append(asdict(ServerCall(
                server_name="structured_db",
                tool_name="search_techniques",
                arguments={"keyword": keyword},
            )))

    plan = ExecutionPlan(
        parallel_groups=[parallel_group] if parallel_group else [],
        timeout_ms=30_000,
    )
    return {"execution_plan": asdict(plan)}


async def retrieve_parallel(state: OrchestratorState) -> dict:
    assert _pool is not None
    plan = state["execution_plan"]
    all_results: list[dict] = state.get("retrieval_results", [])

    for group in plan["parallel_groups"]:
        tasks = []
        for call in group:
            tasks.append(_execute_mcp_call(call))
        group_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in group_results:
            if isinstance(result, RetrievalResult):
                all_results.append(asdict(result))

    return {"retrieval_results": all_results}


async def _execute_mcp_call(call: dict) -> RetrievalResult:
    assert _pool is not None
    server = call["server_name"]
    tool = call["tool_name"]
    args = call["arguments"]

    start = time.monotonic()
    raw = await _pool.call_tool(server, tool, args)
    latency = (time.monotonic() - start) * 1000

    items = _parse_retrieval_result(server, tool, raw)

    return RetrievalResult(
        source=server,
        items=items,
        latency_ms=latency,
        metadata={"tool": tool, "arguments": args},
    )


def _parse_retrieval_result(source: str, tool: str, raw: dict | list) -> list[RetrievedItem]:
    items: list[RetrievedItem] = []

    if isinstance(raw, dict) and "error" in raw:
        return items

    if tool == "semantic_search" and isinstance(raw, list):
        for entry in raw:
            items.append(RetrievedItem(
                content=entry.get("text", ""),
                score=float(entry.get("score", 0.0)),
                source=source,
                source_id=entry.get("id", ""),
                item_type="chunk",
                metadata=entry.get("metadata", {}),
            ))

    elif tool == "graph_query" and isinstance(raw, dict):
        triples = raw.get("triples", [])
        for i, triple in enumerate(triples):
            text = f"{triple['subject']} --[{triple['predicate']}]--> {triple['object']}"
            items.append(RetrievedItem(
                content=text,
                score=1.0 / (i + 1),
                source=source,
                source_id=f"triple_{i}",
                item_type="triple",
                metadata=triple,
            ))

    elif tool == "search_techniques" and isinstance(raw, list):
        for i, row in enumerate(raw):
            text = (
                f"{row['name']} ({row['year']}, {row['authors']}): "
                f"{row['description']} Paper: {row['paper']}"
            )
            items.append(RetrievedItem(
                content=text,
                score=1.0 / (i + 1),
                source=source,
                source_id=f"technique_{row.get('name', i)}",
                item_type="row",
                metadata=row,
            ))

    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            items.append(RetrievedItem(
                content=json.dumps(entry) if isinstance(entry, dict) else str(entry),
                score=1.0 / (i + 1),
                source=source,
                source_id=f"{source}_{i}",
                item_type="unknown",
            ))

    return items


async def fuse_results(state: OrchestratorState) -> dict:
    raw_results = state.get("retrieval_results", [])
    results = []
    for r in raw_results:
        items = [RetrievedItem(**it) for it in r["items"]]
        results.append(RetrievalResult(
            source=r["source"],
            items=items,
            latency_ms=r.get("latency_ms", 0),
            metadata=r.get("metadata", {}),
        ))

    fused = rrf_fuse(results)
    return {"fused_context": asdict(fused)}


async def check_quality(state: OrchestratorState) -> dict:
    raw_fused = state["fused_context"]
    fused = FusedContext(
        ranked_items=[RetrievedItem(**it) for it in raw_fused["ranked_items"]],
        overall_score=raw_fused["overall_score"],
        source_distribution=raw_fused["source_distribution"],
        total_tokens=raw_fused.get("total_tokens", 0),
    )
    assessment = await evaluate_quality(fused, state["query"])
    return {"quality_assessment": asdict(assessment)}


def should_reformulate(state: OrchestratorState) -> str:
    assessment = state.get("quality_assessment", {})
    iteration = state.get("iteration_count", 0)
    if assessment.get("sufficient", True) or iteration >= MAX_REFINEMENT_ITERATIONS:
        return "generate"
    return "reformulate"


async def reformulate_query(state: OrchestratorState) -> dict:
    query = state["query"]
    fused = state.get("fused_context", {})
    top_items = fused.get("ranked_items", [])[:3]
    partial_context = "\n".join(it.get("content", "") for it in top_items)

    system = (
        "You are a query reformulation expert. The original query did not retrieve "
        "sufficient context. Rewrite the query to be more specific, using any "
        "partial context provided. Return ONLY the reformulated query string."
    )
    user = f"Original query: {query}\n\nPartial context:\n{partial_context}"

    reformulated = await _llm_call(system, user)
    iteration = state.get("iteration_count", 0) + 1

    return {
        "query": reformulated.strip(),
        "iteration_count": iteration,
        "retrieval_results": [],
    }


async def generate_response(state: OrchestratorState) -> dict:
    query = state["query"]
    fused = state.get("fused_context", {})
    ranked_items = fused.get("ranked_items", [])[:8]

    context_parts = []
    sources_seen: set[str] = set()
    for item in ranked_items:
        context_parts.append(f"[{item['source']}] {item['content']}")
        sources_seen.add(item["source"])

    context_text = "\n\n".join(context_parts)

    system = (
        "You are a knowledgeable assistant. Answer the question using the provided "
        "context from multiple sources. Cite the source (vector_store, knowledge_graph, "
        "or structured_db) when referencing information. If the context is insufficient, "
        "say so honestly. Be concise but thorough."
    )
    user = f"Question: {query}\n\nContext:\n{context_text}"

    answer = await _llm_call(system, user)
    return {"final_response": answer}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify_query)
    graph.add_node("plan", plan_routes)
    graph.add_node("retrieve", retrieve_parallel)
    graph.add_node("fuse", fuse_results)
    graph.add_node("quality_check", check_quality)
    graph.add_node("reformulate", reformulate_query)
    graph.add_node("generate", generate_response)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", "quality_check")
    graph.add_conditional_edges("quality_check", should_reformulate, {
        "generate": "generate",
        "reformulate": "reformulate",
    })
    graph.add_edge("reformulate", "plan")
    graph.add_edge("generate", END)

    return graph


async def create_orchestrator() -> tuple:
    """Initialize MCP clients and compile the orchestration graph.

    Returns (compiled_graph, pool) — caller is responsible for closing the pool.
    """
    global _pool
    pool = MCPClientPool()
    await pool.connect_all()
    _pool = pool

    graph = build_graph()
    compiled = graph.compile()
    return compiled, pool
