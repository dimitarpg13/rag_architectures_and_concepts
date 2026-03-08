import json
import re
import sqlite3

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from sample_data import STRUCTURED_DB_RECORDS

load_dotenv()

mcp = FastMCP("StructuredDB RAG Server")

DB_COLUMNS = [
    ("name", "TEXT"),
    ("category", "TEXT"),
    ("year", "INTEGER"),
    ("authors", "TEXT"),
    ("paper", "TEXT"),
    ("description", "TEXT"),
]

_UNSAFE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    col_defs = ", ".join(f"{name} {typ}" for name, typ in DB_COLUMNS)
    conn.execute(f"CREATE TABLE techniques ({col_defs})")
    placeholders = ", ".join("?" for _ in DB_COLUMNS)
    col_names = [name for name, _ in DB_COLUMNS]
    for record in STRUCTURED_DB_RECORDS:
        values = tuple(record[col] for col in col_names)
        conn.execute(f"INSERT INTO techniques ({', '.join(col_names)}) VALUES ({placeholders})", values)
    conn.commit()
    return conn


db = _init_db()


@mcp.tool()
def sql_query(query: str) -> str:
    """Execute a read-only SQL SELECT query against the techniques table.

    Only SELECT statements are allowed. Returns JSON array of matching rows.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed."})
    if _UNSAFE_PATTERN.search(stripped):
        return json.dumps({"error": "Mutating statements are not allowed."})
    try:
        rows = db.execute(stripped).fetchall()
        return json.dumps([dict(row) for row in rows])
    except sqlite3.Error as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def search_techniques(
    keyword: str,
    category: str = "",
    year_from: int = 0,
    year_to: int = 9999,
) -> str:
    """Search techniques with optional filters.

    Keyword is matched against name, description, and authors (case-insensitive).
    Optionally filter by category and/or year range.
    """
    clauses = [
        "(name LIKE ? OR description LIKE ? OR authors LIKE ?)",
        "year >= ?",
        "year <= ?",
    ]
    kw_param = f"%{keyword}%"
    params: list = [kw_param, kw_param, kw_param, year_from, year_to]

    if category:
        clauses.append("category = ?")
        params.append(category)

    where = " AND ".join(clauses)
    rows = db.execute(f"SELECT * FROM techniques WHERE {where}", params).fetchall()
    return json.dumps([dict(row) for row in rows])


@mcp.tool()
def get_schema() -> str:
    """Return the schema of the techniques table as JSON."""
    schema = [{"column": name, "type": typ} for name, typ in DB_COLUMNS]
    return json.dumps(schema)


@mcp.tool()
def list_categories() -> str:
    """Return a JSON list of distinct technique categories."""
    rows = db.execute("SELECT DISTINCT category FROM techniques ORDER BY category").fetchall()
    return json.dumps([row["category"] for row in rows])


if __name__ == "__main__":
    mcp.run()
