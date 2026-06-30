"""Unit tests for core.models helpers."""

from dbs_vector.core.models import (
    _clean_table_name,
    _normalize_table_name,
    sql_chunk_from_record,
)


class TestCleanTableName:
    def test_strips_double_quotes_preserves_case(self):
        assert _clean_table_name('"OrderShipment"') == "OrderShipment"

    def test_keeps_schema_prefix_and_case(self):
        assert _clean_table_name("TryOTODyn.OrderShipment") == "TryOTODyn.OrderShipment"

    def test_strips_quotes_around_schema_qualified(self):
        assert _clean_table_name('"TryOTODyn"."OrderShipment"') == "TryOTODyn.OrderShipment"

    def test_strips_backticks(self):
        assert _clean_table_name("`OrderShipment`") == "OrderShipment"

    def test_strips_whitespace(self):
        assert _clean_table_name("  TryOTODyn.OrderShipment  ") == "TryOTODyn.OrderShipment"

    def test_empty_string_returns_empty(self):
        assert _clean_table_name("") == ""


class TestNormalizeTableName:
    """Unchanged: query-time normalization (lowercase + schema-strip)."""

    def test_strips_schema_and_lowercases(self):
        assert _normalize_table_name("TryOTODyn.OrderShipment") == "ordershipment"

    def test_strips_quotes(self):
        assert _normalize_table_name('"OrderShipment"') == "ordershipment"

    def test_idempotent(self):
        assert _normalize_table_name("ordershipment") == "ordershipment"

    def test_only_dot_segments_returns_last(self):
        assert _normalize_table_name("a.b.c") == "c"


class TestSqlChunkFromRecord:
    def _base_record(self, **overrides):
        record = {"id": "fp1", "text": "SELECT 1", "source": "db", "tables": []}
        record.update(overrides)
        return record

    def test_tables_keep_original_case_and_schema(self):
        record = self._base_record(tables=['"TryOTODyn.MagentoOrders"', "Clients"])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == ["TryOTODyn.MagentoOrders", "Clients"]

    def test_empty_tables_list_preserved(self):
        chunk = sql_chunk_from_record(self._base_record(tables=[]))
        assert chunk.tables == []

    def test_null_tables_becomes_empty_list(self):
        chunk = sql_chunk_from_record(self._base_record(tables=None))
        assert chunk.tables == []

    def test_dedup_is_exact_not_case_or_schema_folded(self):
        record = self._base_record(tables=["MagentoOrders", "TryOTODyn.MagentoOrders"])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == ["MagentoOrders", "TryOTODyn.MagentoOrders"]

    def test_exact_duplicates_are_dropped(self):
        record = self._base_record(tables=["MagentoOrders", '"MagentoOrders"'])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == ["MagentoOrders"]
