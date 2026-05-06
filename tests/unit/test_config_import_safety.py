import importlib
import sys
from unittest.mock import patch


def test_importing_config_does_not_open_files(tmp_path, monkeypatch):
    """Module import must not perform any file I/O — neither config.yaml nor .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DBS_LOG_LEVEL=DEBUG\n")  # tempt pydantic-settings
    (tmp_path / "config.yaml").write_text("system: : :\nbroken: :")  # tempt yaml.safe_load
    sys.modules.pop("dbs_vector.config", None)
    with patch("pathlib.Path.open") as mock_open, patch("builtins.open") as builtin_open:
        importlib.import_module("dbs_vector.config")
    assert mock_open.call_count == 0, f"Path.open called: {mock_open.call_args_list}"
    assert builtin_open.call_count == 0, f"builtins.open called: {builtin_open.call_args_list}"


def test_module_singleton_is_default_settings():
    """settings = Settings(_env_file=None) at module bottom — empty defaults, no I/O."""
    sys.modules.pop("dbs_vector.config", None)
    config = importlib.import_module("dbs_vector.config")
    assert config.settings.engines == {}
    assert config.settings.profiles == {}
    assert config.settings.memory_budget_gb is None


def test_load_settings_default_does_not_validate(tmp_path):
    """load_settings(path) without validate=True must not call _validate_config."""
    sys.modules.pop("dbs_vector.config", None)
    from dbs_vector.config import load_settings

    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")
    s = load_settings(str(yaml_path))
    assert s.engines == {}
