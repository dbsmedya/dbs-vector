# tests/unit/test_server_watch_lifecycle.py
"""The watcher lifecycle inside start_stdio_server().

A watcher that fails partway through start() must still be torn down —
otherwise already-started observer threads leak for the process lifetime.
"""

from unittest.mock import MagicMock

import pytest

from dbs_vector.mcp import server as server_mod


@pytest.fixture
def stub_registration(monkeypatch):
    """Neutralize everything start_stdio_server does except watcher handling."""
    for name in (
        "initialize_services",
        "register_search_tools",
        "register_read_tools",
        "register_browse_tools",
        "register_triage_tools",
        "register_discovery_tool",
    ):
        monkeypatch.setattr(server_mod, name, MagicMock())
    monkeypatch.setattr(server_mod.mcp, "run", MagicMock())


def test_watcher_is_stopped_when_start_raises(stub_registration, monkeypatch):
    watcher = MagicMock()
    watcher.start.side_effect = RuntimeError("observer scheduling blew up")
    monkeypatch.setattr(server_mod, "build_watcher_service", lambda: watcher)

    with pytest.raises(RuntimeError, match="observer scheduling blew up"):
        server_mod.start_stdio_server()

    watcher.stop.assert_called_once()


def test_watcher_is_stopped_on_the_normal_path(stub_registration, monkeypatch):
    watcher = MagicMock()
    monkeypatch.setattr(server_mod, "build_watcher_service", lambda: watcher)

    server_mod.start_stdio_server()

    watcher.start.assert_called_once()
    watcher.stop.assert_called_once()


def test_no_watcher_configured_is_a_clean_no_op(stub_registration, monkeypatch):
    monkeypatch.setattr(server_mod, "build_watcher_service", lambda: None)

    server_mod.start_stdio_server()  # must not raise
