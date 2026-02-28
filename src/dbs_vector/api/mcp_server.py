from mcp.server.fastmcp import FastMCP

from dbs_vector.api.state import _services

# For SSE we do not use the lifespan here, it is driven by main.py
mcp = FastMCP("dbs-vector")


@mcp.tool()
async def search_documents(query: str, limit: int = 5, source_filter: str | None = None) -> str:
    """
    Search indexed codebase documents (Markdown, Python, etc.) via semantic vector search.

    Args:
        query: The semantic search query or concept you are looking for.
        limit: Maximum number of results to return.
        source_filter: Optional file path or pattern to restrict the search.
    """
    service = _services.get("md")
    if not service:
        return "Error: Document search service ('md' engine) is not initialized."

    try:
        # execute_query is synchronous, but we can call it directly since MCP runs locally
        results = service.execute_query(
            query=query,
            source_filter=source_filter,
            limit=limit,
            extra_filters={},
        )

        if not results:
            return f"No results found for query: '{query}'"

        output = [f"Found {len(results)} results for '{query}':\n"]
        for res in results:
            dist_str = f"{res.distance:.4f}" if res.distance is not None else "N/A (FTS)"
            chunk = res.chunk
            output.append(
                f"--- Result (Score: {dist_str}) ---\n"
                f"Source: {chunk.source}\n"
                f"Content:\n{chunk.text}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search execution failed: {e}"


@mcp.tool()
async def search_sql_logs(
    query: str, limit: int = 5, source_filter: str | None = None, min_time: float | None = None
) -> str:
    """
    Search indexed SQL query logs via semantic vector search.

    Args:
        query: The semantic search query, e.g. 'find user by email' or partial SQL.
        limit: Maximum number of results to return.
        source_filter: Optional database name to restrict the search.
        min_time: Minimum execution time in milliseconds.
    """
    service = _services.get("sql")
    if not service:
        return "Error: SQL search service ('sql' engine) is not initialized."

    extra_filters = {}
    if min_time is not None:
        extra_filters["min_time"] = min_time

    try:
        results = service.execute_query(
            query=query,
            source_filter=source_filter,
            limit=limit,
            extra_filters=extra_filters,
        )

        if not results:
            return f"No results found for query: '{query}'"

        output = [f"Found {len(results)} results for '{query}':\n"]
        for res in results:
            dist_str = f"{res.distance:.4f}" if res.distance is not None else "N/A (FTS)"
            chunk = res.chunk
            output.append(
                f"--- Result (Score: {dist_str}) ---\n"
                f"Source Database: {chunk.source}\n"
                f"Execution Time: {chunk.execution_time_ms}ms (Calls: {chunk.calls})\n"
                f"SQL Query:\n{chunk.raw_query}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search execution failed: {e}"
