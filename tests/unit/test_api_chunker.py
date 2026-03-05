from unittest.mock import MagicMock, patch

from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.api import ApiChunker

BASE_URL = "http://test-api.internal/api/v1"
API_KEY = "sk-test"

_RECORD = {
    "id": "fp1",
    "text": "SELECT * FROM users",
    "raw_query": "SELECT * FROM users WHERE id=1",
    "source": "db1",
    "execution_time_ms": 150.5,
    "calls": 3,
    "tables": ["users"],
    "latest_ts": "2024-01-15T10:30:00Z",
    "user": "admin",
    "host": "localhost",
    "rows_sent": 10,
    "rows_examined": 100,
    "lock_time_sec": 0.01,
}


def _doc() -> Document:
    return Document(filepath=BASE_URL, content="", content_hash="api-chunker")


def _mock_get_response(
    data: list, has_more: bool = False, next_cursor: str | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    payload: dict = {"data": data, "has_more": has_more}
    if next_cursor is not None:
        payload["next_cursor"] = next_cursor
    resp.json.return_value = payload
    return resp


def _mock_post_response(columns: list[str], rows: list[list]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"columns": columns, "rows": rows}
    return resp


# ---------------------------------------------------------------------------
# Paginated GET tests
# ---------------------------------------------------------------------------


def test_paginated_single_page():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)
    mock_resp = _mock_get_response([_RECORD], has_more=False)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert len(chunks) == 1
    c = chunks[0]
    assert c.id == "fp1"
    assert c.text == "SELECT * FROM users"
    assert c.source == "db1"
    assert c.execution_time_ms == 150.5
    assert c.calls == 3
    assert c.tables == ["users"]
    assert c.user == "admin"
    assert c.host == "localhost"
    assert c.rows_sent == 10
    assert c.rows_examined == 100
    assert c.lock_time_sec == 0.01
    assert len(c.content_hash) == 16


def test_paginated_two_pages():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY, page_size=1)

    record2 = {**_RECORD, "id": "fp2", "text": "SELECT * FROM orders", "source": "db2"}

    page1 = _mock_get_response([_RECORD], has_more=True, next_cursor="cursor-abc")
    page2 = _mock_get_response([record2], has_more=False)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.side_effect = [page1, page2]

        chunks = list(chunker.process(_doc()))

    assert len(chunks) == 2
    assert chunks[0].id == "fp1"
    assert chunks[1].id == "fp2"

    # Second call should include cursor param
    _, second_kwargs = ctx.get.call_args_list[1]
    assert second_kwargs["params"]["cursor"] == "cursor-abc"


def test_paginated_empty_response():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)
    mock_resp = _mock_get_response([], has_more=False)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert chunks == []


# ---------------------------------------------------------------------------
# Custom query (POST) tests
# ---------------------------------------------------------------------------


def test_custom_query_success():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY, custom_query="SELECT 1")
    columns = ["id", "text", "source", "execution_time_ms", "calls", "latest_ts"]
    rows = [["fp1", "SELECT * FROM users", "db1", 99.9, 2, "2024-01-15T10:30:00Z"]]
    mock_resp = _mock_post_response(columns, rows)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert len(chunks) == 1
    assert chunks[0].id == "fp1"
    assert chunks[0].execution_time_ms == 99.9
    assert chunks[0].calls == 2


def test_custom_query_missing_required_fields():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY, custom_query="SELECT 1")
    # Row missing 'text'
    columns = ["id", "source", "latest_ts"]
    rows = [["fp1", "db1", "2024-01-15T10:30:00Z"]]
    mock_resp = _mock_post_response(columns, rows)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert chunks == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_http_error_401():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert chunks == []


def test_connection_error():
    import httpx

    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.side_effect = httpx.ConnectError("refused")

        chunks = list(chunker.process(_doc()))

    assert chunks == []


def test_missing_httpx_import():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)

    with patch.dict("sys.modules", {"httpx": None}):
        chunks = list(chunker.process(_doc()))

    assert chunks == []


# ---------------------------------------------------------------------------
# Nullable / optional fields
# ---------------------------------------------------------------------------


def test_nullable_fields_none():
    record = {
        "id": "fp3",
        "text": "SELECT 1",
        "source": "db3",
        "latest_ts": "2024-01-15T10:30:00Z",
    }
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY)
    mock_resp = _mock_get_response([record], has_more=False)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        chunks = list(chunker.process(_doc()))

    assert len(chunks) == 1
    c = chunks[0]
    assert c.user is None
    assert c.host is None
    assert c.rows_sent is None
    assert c.rows_examined is None
    assert c.lock_time_sec is None
    assert c.execution_time_ms == 0.0
    assert c.calls == 1
    assert c.tables == []


# ---------------------------------------------------------------------------
# database param forwarding
# ---------------------------------------------------------------------------


def test_database_param_sent():
    chunker = ApiChunker(base_url=BASE_URL, api_key=API_KEY, database="prod_db")
    mock_resp = _mock_get_response([], has_more=False)

    with patch("httpx.Client") as MockClient:
        ctx = MockClient.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp

        list(chunker.process(_doc()))

    _, call_kwargs = ctx.get.call_args
    assert call_kwargs["params"]["database"] == "prod_db"
