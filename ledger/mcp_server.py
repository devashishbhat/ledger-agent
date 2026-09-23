"""
ledger/mcp_server.py — Stage 10: expose Ledger's search as an MCP server.

Any MCP host (Claude Desktop, Claude Code, Cursor, VS Code, your own agent)
can connect to this and use these tools without knowing anything about
our code.

Run modes:
    uv run python -m ledger.mcp_server                 # stdio (for Claude Desktop)
    uv run python -m ledger.mcp_server --http          # Streamable HTTP on port 8765

IMPORTANT for stdio mode: stdout is the communication channel with the
host. Never print() in code this server calls — a stray print corrupts
the protocol. Log to stderr instead.
"""
import sys

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from ledger import config
from ledger.retrieval import get_retriever

server = MCPServer(
    name="ledger-filings",
    instructions=("Search SEC Form 10-K annual reports. Call list_filings first to see which "
                  "companies and fiscal years are available, then search_filings for facts."),
)

# Annotations tell the host what a tool does, so it can decide how careful
# to be. All our tools only READ data, never change anything.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                            idempotent_hint=True, open_world_hint=False)


@server.tool(annotations=READ_ONLY)
def list_filings() -> dict:
    """List the companies (by ticker) and fiscal years available to search."""
    retriever = get_retriever()
    names = retriever.company_names()
    return {"filings": [{"ticker": t, "company": names[t], "fiscal_years": years}
                        for t, years in retriever.available_filings().items()]}


@server.tool(annotations=READ_ONLY)
def search_filings(query: str, ticker: str | None = None,
                   fiscal_year: int | None = None, k: int = 5) -> dict:
    """Search 10-K filings and return the best-matching passages.

    Use one search per fact. Pass ticker (e.g. "AAPL") and fiscal_year
    (e.g. 2024) whenever you know them: results are much more precise.
    Each passage includes its exact location in the source document.
    """
    k = max(1, min(k, 10))       # never let a caller request 1,000 passages
    results = get_retriever().search(
        query,
        tickers=[ticker.upper()] if ticker else None,
        years=[fiscal_year] if fiscal_year else None,
        k=k,
    )
    return {"passages": [
        {"chunk_id": c["chunk_id"], "ticker": c["ticker"], "fiscal_year": c["fiscal_year"],
         "section": c["section"], "text": c["text"], "source_url": c["source_url"],
         "char_start": c["char_start"], "char_end": c["char_end"]}
        for c in results
    ]}


@server.tool(annotations=READ_ONLY)
def get_passage(chunk_id: str) -> dict:
    """Fetch one passage by its chunk_id (as returned by search_filings)."""
    chunk = get_retriever().by_id.get(chunk_id)
    if chunk is None:
        return {"error": f"No passage with id {chunk_id!r}."}   # an error, as data
    return chunk


if __name__ == "__main__":
    # Load the search models now, so the first tool call isn't slow.
    print("Loading search index...", file=sys.stderr)
    get_retriever()
    if "--http" in sys.argv:
        server.run(transport="streamable-http", port=8765)
    else:
        server.run()     # stdio
