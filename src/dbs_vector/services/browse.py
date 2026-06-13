"""BrowseService — read-only analytical SQL over a SQL engine's table.

Execution core shared by the CLI (raw SQL passthrough, uncapped) and the MCP
(structured params compiled to SQL, capped). polars.SQLContext runs the SQL
over the Arrow read; filter VALUES on the MCP path are applied as bound polars
expressions, never interpolated (see build_and_run). No embedder, no semantic
ranking, no read-only guard (polars frames are in-memory; mutation is impossible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from dbs_vector.core.ports import IVectorStore


class BrowseError(Exception):
    """Raised when a browse SQL statement fails to execute."""


class BrowseValidationError(BrowseError):
    """Raised by the MCP structured builder on bad params. Safe to disclose:
    carries no infrastructure detail, so it is returned to the LLM verbatim."""


@dataclass
class BrowseResult:
    rows: list[dict[str, Any]]
    columns: list[str]
    total_matching: int
    grouped: bool
    limit_applied: bool


class BrowseService:
    def __init__(self, store: IVectorStore, frame_alias: str) -> None:
        self.store = store
        self.frame_alias = frame_alias

    # --- shared executor -------------------------------------------------
    def _frames(self) -> dict[str, pl.DataFrame]:
        df = pl.from_arrow(self.store.scan())
        if isinstance(df, pl.Series):  # pragma: no cover - from_arrow returns DataFrame for tables
            df = df.to_frame()
        return {self.frame_alias: df, "t": df, "t_by_table": df.explode("tables")}

    @staticmethod
    def _execute(sql: str, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
        try:
            return pl.SQLContext(frames=frames).execute(sql, eager=True)
        except Exception as e:  # surface polars message; caller decides disclosure
            raise BrowseError(str(e)) from e

    # --- CLI path: raw SQL, uncapped ------------------------------------
    def run_sql(self, sql: str) -> BrowseResult:
        result = self._execute(sql, self._frames())
        return BrowseResult(
            rows=result.to_dicts(),
            columns=result.columns,
            total_matching=result.height,
            grouped=False,
            limit_applied=False,
        )


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def result_to_json(result: BrowseResult) -> str:
    """Serialize all rows as JSON; datetimes → ISO 8601."""
    return json.dumps(result.rows, indent=2, ensure_ascii=False, default=_json_default)
