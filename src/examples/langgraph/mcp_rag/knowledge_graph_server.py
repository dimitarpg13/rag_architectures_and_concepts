"""MCP server wrapping a Microsoft GraphRAG knowledge graph.

Expects a pre-built GraphRAG index under ./graphrag_data/ (created by graphrag_setup.py).
Exposes GraphRAG local and global search as MCP tools, plus direct entity/relationship
browsing via the parquet output files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("KnowledgeGraph RAG Server")

GRAPHRAG_DIR = Path(__file__).parent / "graphrag_data"
OUTPUT_DIR = GRAPHRAG_DIR / "output"

_entities_df: pd.DataFrame | None = None
_relationships_df: pd.DataFrame | None = None


def _load_parquet_data() -> None:
    global _entities_df, _relationships_df

    entities_file = OUTPUT_DIR / "entities.parquet"
    rels_file = OUTPUT_DIR / "relationships.parquet"

    if entities_file.exists():
        _entities_df = pd.read_parquet(entities_file)
    if rels_file.exists():
        _relationships_df = pd.read_parquet(rels_file)


_load_parquet_data()


def _entity_name_col() -> str:
    if _entities_df is not None and "title" in _entities_df.columns:
        return "title"
    return "name"


def _run_graphrag_query(query: str, method: str) -> str:
    result = subprocess.run(
        [
            sys.executable, "-m", "graphrag", "query",
            "--root", str(GRAPHRAG_DIR),
            "--method", method,
            "--query", query,
        ],
        capture_output=True,
        text=True,
        cwd=str(GRAPHRAG_DIR),
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return f"Query failed: {result.stderr.strip()}"


@mcp.tool()
def graph_search(query: str, method: str = "local") -> str:
    """Search the GraphRAG knowledge graph using natural language.

    Args:
        query: Natural language question to answer from the knowledge graph.
        method: Search method — "local" for specific entity questions,
                "global" for broad synthesis across the corpus.

    Returns:
        The GraphRAG-generated answer as a text string.
    """
    if method not in ("local", "global"):
        return json.dumps({"error": f"Invalid method '{method}'. Use 'local' or 'global'."})

    if not (OUTPUT_DIR / "entities.parquet").exists():
        return json.dumps({"error": "GraphRAG index not found. Run graphrag_setup.py first."})

    answer = _run_graphrag_query(query, method)
    return json.dumps({"query": query, "method": method, "answer": answer})


@mcp.tool()
def list_entities() -> str:
    """Return all entities extracted by GraphRAG with their types and descriptions."""
    if _entities_df is None:
        return json.dumps({"error": "No entities data. Run graphrag_setup.py first."})

    name_col = _entity_name_col()
    cols = [c for c in [name_col, "type", "description"] if c in _entities_df.columns]
    records = _entities_df[cols].to_dict(orient="records")
    return json.dumps(records, indent=2, default=str)


@mcp.tool()
def search_entities(keyword: str) -> str:
    """Case-insensitive search for entities whose name or description contains the keyword."""
    if _entities_df is None:
        return json.dumps({"error": "No entities data. Run graphrag_setup.py first."})

    name_col = _entity_name_col()
    kw = keyword.lower()
    mask = _entities_df[name_col].str.lower().str.contains(kw, na=False)
    if "description" in _entities_df.columns:
        mask = mask | _entities_df["description"].str.lower().str.contains(kw, na=False)

    matched = _entities_df[mask]
    cols = [c for c in [name_col, "type", "description"] if c in matched.columns]
    records = matched[cols].to_dict(orient="records")

    if _relationships_df is not None:
        for rec in records:
            entity = rec.get(name_col, "")
            rels = _relationships_df[
                (_relationships_df["source"].str.lower() == entity.lower())
                | (_relationships_df["target"].str.lower() == entity.lower())
            ]
            rel_cols = [c for c in ["source", "target", "description"] if c in rels.columns]
            rec["relationships"] = rels[rel_cols].to_dict(orient="records")

    return json.dumps(records, indent=2, default=str)


@mcp.tool()
def get_relationships(entity: str) -> str:
    """Get all relationships involving the given entity."""
    if _relationships_df is None:
        return json.dumps({"error": "No relationships data. Run graphrag_setup.py first."})

    entity_lower = entity.lower()
    mask = (
        _relationships_df["source"].str.lower().str.contains(entity_lower, na=False)
        | _relationships_df["target"].str.lower().str.contains(entity_lower, na=False)
    )
    matched = _relationships_df[mask]
    cols = [c for c in ["source", "target", "description", "weight"] if c in matched.columns]
    records = matched[cols].to_dict(orient="records")

    if not records:
        return json.dumps({"error": f"No relationships found for entity '{entity}'."})

    return json.dumps({"entity": entity, "relationships": records}, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
