import json
from collections import defaultdict, deque

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from sample_data import KNOWLEDGE_GRAPH_TRIPLES

load_dotenv()

mcp = FastMCP("KnowledgeGraph RAG Server")

_triples: list[dict] = list(KNOWLEDGE_GRAPH_TRIPLES)
_outgoing: dict[str, list[int]] = defaultdict(list)
_incoming: dict[str, list[int]] = defaultdict(list)
_entities: set[str] = set()

for _idx, _t in enumerate(_triples):
    _outgoing[_t["subject"]].append(_idx)
    _incoming[_t["object"]].append(_idx)
    _entities.add(_t["subject"])
    _entities.add(_t["object"])


def _triple_indices_for(entity: str) -> list[int]:
    seen: set[int] = set()
    for idx in _outgoing.get(entity, []):
        seen.add(idx)
    for idx in _incoming.get(entity, []):
        seen.add(idx)
    return sorted(seen)


@mcp.tool()
def graph_query(entity: str, depth: int = 1) -> str:
    """Return all triples reachable from *entity* within *depth* hops (BFS)."""
    if entity not in _entities:
        return json.dumps({"error": f"Entity '{entity}' not found in the knowledge graph."})

    visited_entities: set[str] = set()
    result_indices: set[int] = set()
    queue: deque[tuple[str, int]] = deque([(entity, 0)])
    visited_entities.add(entity)

    while queue:
        current, current_depth = queue.popleft()
        for idx in _triple_indices_for(current):
            result_indices.add(idx)
            if current_depth < depth:
                triple = _triples[idx]
                for neighbor in (triple["subject"], triple["object"]):
                    if neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        queue.append((neighbor, current_depth + 1))

    return json.dumps(
        {"entity": entity, "depth": depth, "triples": [_triples[i] for i in sorted(result_indices)]},
        indent=2,
    )


@mcp.tool()
def find_path(source: str, target: str) -> str:
    """Find the shortest path between *source* and *target* entities (BFS)."""
    for label, name in [("Source", source), ("Target", target)]:
        if name not in _entities:
            return json.dumps({"error": f"{label} entity '{name}' not found in the knowledge graph."})

    if source == target:
        return json.dumps({"source": source, "target": target, "path": []})

    visited: set[str] = {source}
    queue: deque[tuple[str, list[int]]] = deque([(source, [])])

    while queue:
        current, path_indices = queue.popleft()
        for idx in _triple_indices_for(current):
            triple = _triples[idx]
            neighbor = triple["object"] if triple["subject"] == current else triple["subject"]
            if neighbor in visited:
                continue
            new_path = path_indices + [idx]
            if neighbor == target:
                return json.dumps(
                    {
                        "source": source,
                        "target": target,
                        "path": [_triples[i] for i in new_path],
                    },
                    indent=2,
                )
            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return json.dumps({"error": f"No path found between '{source}' and '{target}'."})


@mcp.tool()
def search_entities(keyword: str) -> str:
    """Case-insensitive search for entities whose name contains *keyword*."""
    keyword_lower = keyword.lower()
    matches = sorted(e for e in _entities if keyword_lower in e.lower())

    results = []
    for entity in matches:
        results.append({
            "entity": entity,
            "relationships": [_triples[i] for i in _triple_indices_for(entity)],
        })

    return json.dumps(results, indent=2)


@mcp.tool()
def list_entities() -> str:
    """Return all unique entities in the knowledge graph."""
    return json.dumps(sorted(_entities), indent=2)


if __name__ == "__main__":
    mcp.run()
