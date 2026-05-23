"""Unit tests for core.models helpers."""

from dbs_vector.core.models import _normalize_table_name, sql_chunk_from_record


class TestNormalizeTableName:
    def test_strips_double_quotes(self):
        assert _normalize_table_name('"OrderShipment"') == "ordershipment"

    def test_strips_schema_prefix(self):
        assert _normalize_table_name("TryOTODyn.OrderShipment") == "ordershipment"

    def test_strips_quoted_schema_qualified(self):
        assert _normalize_table_name('"TryOTODyn"."OrderShipment"') == "ordershipment"

    def test_lowercases(self):
        assert _normalize_table_name("OrderShipment") == "ordershipment"

    def test_already_normalized_is_idempotent(self):
        assert _normalize_table_name("ordershipment") == "ordershipment"

    def test_strips_whitespace(self):
        assert _normalize_table_name("  OrderShipment  ") == "ordershipment"

    def test_handles_backticks(self):
        assert _normalize_table_name("`OrderShipment`") == "ordershipment"

    def test_empty_string_returns_empty(self):
        assert _normalize_table_name("") == ""

    def test_only_dot_segments_returns_last(self):
        assert _normalize_table_name("a.b.c") == "c"


class TestSqlChunkFromRecord:
    def _base_record(self, **overrides):
        record = {
            "id": "fp1",
            "text": "SELECT 1",
            "source": "db",
            "tables": [],
        }
        record.update(overrides)
        return record

    def test_tables_are_normalized(self):
        record = self._base_record(tables=['"TryOTODyn.OrderShipment"', "Clients"])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == ["ordershipment", "clients"]

    def test_empty_tables_list_preserved(self):
        record = self._base_record(tables=[])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == []

    def test_null_tables_becomes_empty_list(self):
        record = self._base_record(tables=None)
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == []

    def test_duplicates_after_normalization_are_dropped(self):
        record = self._base_record(tables=['"OrderShipment"', "TryOTODyn.OrderShipment"])
        chunk = sql_chunk_from_record(record)
        assert chunk.tables == ["ordershipment"]
