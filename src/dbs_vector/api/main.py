import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dbs_vector.cli import _build_dependencies
from dbs_vector.config import settings
from dbs_vector.core.models import SearchResult, SqlSearchResult
from dbs_vector.services.search import SearchService

# Global service instances holding the initialized models and databases
_services: dict[str, SearchService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events for the API."""
    global _services
    print("\\n[Startup] Initializing MLX Embedders and LanceDB connections...")

    try:
        # Dynamically load all configured engines
        for engine_name in settings.engines.keys():
            print(f"  -> Loading Engine ({engine_name})...")
            deps = _build_dependencies(engine_name)
            _services[engine_name] = SearchService(deps.embedder, deps.store)

        print("[Startup] API is ready to accept concurrent requests.")
    except Exception as e:
        print(f"[Startup] Failed to initialize search services: {e}")
        raise

    yield

    print("\\n[Shutdown] Cleaning up resources...")
    _services.clear()


app = FastAPI(
    title="dbs-vector Search API",
    description="Async API for high-performance Arrow-native local codebase search.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    """Schema for a standard document search request."""

    query: str = Field(..., description="The semantic search query.")
    limit: int = Field(5, ge=1, le=100, description="Maximum number of results to return.")
    source_filter: str | None = Field(None, description="Optional path/file to filter the search.")


class SqlSearchRequest(BaseModel):
    """Schema for an SQL search request."""

    query: str = Field(..., description="The semantic SQL search query.")
    limit: int = Field(5, ge=1, le=100, description="Maximum number of results to return.")
    source_filter: str | None = Field(None, description="Optional database to filter the search.")
    min_time: float | None = Field(None, description="Minimum execution time in ms.")


class SearchResponse(BaseModel):
    """Schema for returning standard search results."""

    query: str
    results: list[SearchResult]


class SqlSearchResponse(BaseModel):
    """Schema for returning SQL search results."""

    query: str
    results: list[SqlSearchResult]


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    if not _services:
        raise HTTPException(status_code=503, detail="Search service initializing or failed")

    status_dict = {"status": "healthy"}
    for engine_name, config in settings.engines.items():
        status_dict[f"{engine_name}_model"] = config.model_name

    return status_dict


@app.post("/search/md", response_model=SearchResponse)
async def search_md(request: SearchRequest) -> SearchResponse:
    """Executes a hybrid search asynchronously for documents."""
    service = _services.get("md")
    if not service:
        raise HTTPException(status_code=503, detail="Document search service is not initialized.")

    try:
        results = await asyncio.to_thread(
            service.execute_query,
            request.query,
            request.source_filter,
            request.limit,
            extra_filters={},
        )
        return SearchResponse(query=request.query, results=results)  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {e}") from e


@app.post("/search/sql", response_model=SqlSearchResponse)
async def search_sql(request: SqlSearchRequest) -> SqlSearchResponse:
    """Executes a hybrid search asynchronously for SQL queries."""
    service = _services.get("sql")
    if not service:
        raise HTTPException(status_code=503, detail="SQL search service is not initialized.")

    extra_filters = {}
    if request.min_time is not None:
        extra_filters["min_time"] = request.min_time

    try:
        results = await asyncio.to_thread(
            service.execute_query,
            request.query,
            request.source_filter,
            request.limit,
            extra_filters=extra_filters,
        )
        return SqlSearchResponse(query=request.query, results=results)  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {e}") from e
