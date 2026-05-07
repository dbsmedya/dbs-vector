import importlib
import sys
from unittest.mock import patch


def test_importing_config_does_not_open_files(tmp_path, monkeypatch):
    """Module import must not cause dbs_vector.config to open config.yaml or .env.

    NOTE: We filter mock calls to only paths matching config.yaml / .env
    because patching builtins.open / Path.open catches all transitive imports
    (importlib.metadata plugin scans, package-metadata reads). We only care
    about THIS module's file I/O contract, not the runtime's.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DBS_LOG_LEVEL=DEBUG\n")  # tempt pydantic-settings
    (tmp_path / "config.yaml").write_text("system: : :\nbroken: :")  # tempt yaml.safe_load
    sys.modules.pop("dbs_vector.config", None)

    def _is_config_target(call) -> bool:
        # Each call_args is roughly call(path, *args, **kwargs); coerce to str
        # then look for our two targets. Catches absolute and relative paths.
        rendered = str(call)
        return (
            "config.yaml" in rendered
            or rendered.endswith(".env'")
            or "/.env" in rendered
            or "\\.env" in rendered
        )

    with patch("pathlib.Path.open") as mock_open, patch("builtins.open") as builtin_open:
        importlib.import_module("dbs_vector.config")

    config_opens = [c for c in mock_open.call_args_list if _is_config_target(c)]
    builtin_config_opens = [c for c in builtin_open.call_args_list if _is_config_target(c)]
    assert config_opens == [], f"Path.open touched config files: {config_opens}"
    assert builtin_config_opens == [], f"builtins.open touched config files: {builtin_config_opens}"


def test_importing_config_does_not_honor_dot_env(tmp_path, monkeypatch):
    """If we cd into a directory with a .env that overrides DBS_LOG_LEVEL,
    importing dbs_vector.config must NOT pick that up — the singleton uses
    Settings(_env_file=None)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DBS_LOG_LEVEL=DEBUG\n")
    sys.modules.pop("dbs_vector.config", None)
    config = importlib.import_module("dbs_vector.config")
    assert config.settings.log_level == "INFO", (
        f"Singleton honored .env (log_level={config.settings.log_level}); "
        f"_env_file=None must skip .env loading."
    )


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


def test_importing_config_does_not_load_mcp_modules():
    """A fresh interpreter importing only `dbs_vector.config` must not
    transitively load any `dbs_vector.mcp.*` modules. config.py validation
    imports `dbs_vector.core.families` (lightweight, no presentation
    coupling); it must NEVER reach into the mcp/ tree.

    Implemented via subprocess so the test is independent of pytest
    collection order — other tests in the suite eagerly import mcp/
    modules, which would contaminate an in-process sys.modules check.
    """
    import subprocess
    import sys

    code = (
        "import dbs_vector.config; "
        "import sys; "
        "leaked = sorted(m for m in sys.modules if m.startswith('dbs_vector.mcp')); "
        "assert leaked == [], f'config.py leaked: {leaked}'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
