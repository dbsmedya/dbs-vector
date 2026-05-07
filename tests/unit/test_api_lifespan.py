# tests/unit/test_api_lifespan.py
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_lifespan_loads_config_before_initialize_services(monkeypatch, tmp_path):
    """Lifespan must call load_settings(validate=True) before initialize_services."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("system:\n  db_path: \"./lance\"\n")
    monkeypatch.setenv("DBS_CONFIG_FILE", str(yaml_path))

    call_order: list[str] = []

    def fake_load_settings(config_file, validate=False):
        call_order.append(f"load_settings(validate={validate})")
        s = MagicMock()
        s.db_path = "./lance"
        s.engines = {}
        s.profiles = {}
        s.memory_budget_gb = None
        s.nprobes = 20
        s.log_level = "INFO"
        s.log_serialize = False
        return s

    def fake_initialize_services():
        call_order.append("initialize_services")

    with (
        patch("dbs_vector.api.main.load_settings", side_effect=fake_load_settings),
        patch(
            "dbs_vector.api.main.initialize_services",
            side_effect=fake_initialize_services,
        ),
        patch("dbs_vector.api.main.mcp") as fake_mcp,
    ):
        fake_session_manager = MagicMock()

        class FakeAsyncContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                return None

        fake_session_manager.run.return_value = FakeAsyncContext()
        fake_mcp.session_manager = fake_session_manager

        from dbs_vector.api.main import lifespan

        async with lifespan(MagicMock()):
            pass

    assert call_order == ["load_settings(validate=True)", "initialize_services"]
