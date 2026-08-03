import json

import pytest
import yaml

from dbs_vector.services.initializer import run_init
from dbs_vector.services.initializer.render import _CASE_INSENSITIVE_FS

BUDGET = 21.0
PATH_PROMPT = "Directory to index (blank when done)"


def _answers(tmp_path, **overrides):
    """One full interview. The path prompt takes a LIST because the wizard
    loops on it until it receives a blank."""
    answers = {
        "Engine name": "docs",
        "Embedding model": "granite-r2",
        "Chunk granularity": "medium",
        PATH_PROMPT: [str(tmp_path / "notes")],
        "Where should LanceDB store its tables?": str(tmp_path / "lancedb"),
        "Where is dbs-vector installed?": str(tmp_path / "repo"),
        "Write config to": str(tmp_path / "config.yaml"),
        "Write MCP config to": str(tmp_path / ".mcp.json"),
    }
    answers.update(overrides)
    return answers


def _prepare(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "pyproject.toml").write_text(
        '[project]\nname = "dbs-vector"\n', encoding="utf-8"
    )


def test_writes_both_files(tmp_path, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)
    assert result.config_path.exists()
    assert result.mcp_path.exists()


def test_generated_config_parses_and_carries_the_engine(tmp_path, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)
    config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert list(config["engines"]) == ["docs"]
    assert config["engines"]["docs"]["tuning_profile"] == "docs-medium"


def test_kind_question_is_skipped_while_one_kind_is_registered(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path))
    run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert "What are you indexing?" not in io.asked


def test_prefix_questions_are_skipped_for_a_model_without_prefixes(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path, **{"Embedding model": "granite-r2"}))
    run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert "Passage prefix" not in io.asked


def test_prefix_questions_are_asked_for_a_prefixed_model(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(
        _answers(tmp_path, **{"Embedding model": "gemma-bf16", "Chunk granularity": "small"})
    )
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert "Passage prefix" in io.asked
    config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert config["engines"]["docs"]["passage_prefix"] == "title: none | text: "


def test_gemma_is_not_offered_the_large_tier(tmp_path, scripted_io):
    """2048 cap cannot hold 2048 + 256 headroom, so ask_choice must reject it."""
    _prepare(tmp_path)
    io = scripted_io(
        _answers(tmp_path, **{"Embedding model": "gemma-bf16", "Chunk granularity": "large"})
    )
    with pytest.raises(ValueError, match="not in"):
        run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)


def test_db_path_is_absolute_in_the_generated_config(tmp_path, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)
    config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert config["system"]["db_path"].startswith("/")


def test_mcp_preserves_an_existing_server(tmp_path, scripted_io):
    _prepare(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers": {"other": {"command": "x"}}}', encoding="utf-8"
    )
    io = scripted_io(
        _answers(
            tmp_path,
            **{f"{tmp_path / '.mcp.json'} exists. What should init do?": "overwrite"},
        )
    )
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    merged = json.loads(result.mcp_path.read_text(encoding="utf-8"))
    assert merged["mcpServers"]["other"] == {"command": "x"}
    assert "dbs-vector" in merged["mcpServers"]
    assert result.mcp_backup is not None


def test_an_unreadable_mcp_file_offers_a_new_destination(tmp_path, scripted_io):
    """The §9 recovery path. It must be REACHABLE: reading before choosing a
    destination made it dead code in the first draft."""
    _prepare(tmp_path)
    (tmp_path / ".mcp.json").write_text("{ not json", encoding="utf-8")
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                f"{tmp_path / '.mcp.json'} could not be read. What should init do?": "new-file",
                "New filename": "mcp.new.json",
            },
        )
    )
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert result.mcp_path == tmp_path / "mcp.new.json"
    # The unreadable original is left exactly as it was.
    assert (tmp_path / ".mcp.json").read_text(encoding="utf-8") == "{ not json"
    assert any("unreadable" in note for note in result.notes)


def test_aborting_on_an_unreadable_mcp_file_writes_nothing(tmp_path, scripted_io):
    _prepare(tmp_path)
    (tmp_path / ".mcp.json").write_text("{ not json", encoding="utf-8")
    io = scripted_io(
        _answers(
            tmp_path,
            **{f"{tmp_path / '.mcp.json'} could not be read. What should init do?": "abort"},
        )
    )
    with pytest.raises(ValueError, match="could not be parsed"):
        run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert not (tmp_path / "config.yaml").exists()


def test_nothing_is_written_when_a_prompt_raises(tmp_path, scripted_io):
    """All writes happen after the last question.

    Relative paths used to be the trigger here, but Task 13 Part C made them
    valid (resolved against cwd). Watch-without-paths is still a genuine
    prompt-time ValueError from DocumentKind.ask, so it still proves the
    guarantee: nothing is written once a prompt raises.
    """
    _prepare(tmp_path)
    io = scripted_io(
        _answers(tmp_path, **{PATH_PROMPT: [], "Watch these paths for changes?": True})
    )
    with pytest.raises(ValueError):
        run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert not (tmp_path / "config.yaml").exists()
    assert not (tmp_path / ".mcp.json").exists()


def test_both_destinations_cannot_resolve_to_one_file(tmp_path, scripted_io):
    """Regression: answering both prompts with the same absent filename used
    to yield two plans for one path, and the MCP commit overwrote the YAML."""
    _prepare(tmp_path)  # without this, _ask_install_dir fails before the
    #                     collision logic is ever reached
    shared = tmp_path / "shared.json"
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                "Write config to": str(shared),
                "Write MCP config to": str(shared),
                "New filename": "mcp.json",
            },
        )
    )
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    assert result.config_path != result.mcp_path
    assert result.config_path == shared
    assert result.mcp_path == tmp_path / "mcp.json"
    # The config survived: it is YAML, not the MCP JSON.
    assert yaml.safe_load(result.config_path.read_text(encoding="utf-8"))["engines"]
    assert json.loads(result.mcp_path.read_text(encoding="utf-8"))["mcpServers"]


def test_the_mcp_file_cannot_be_routed_onto_the_config_backup(tmp_path, scripted_io):
    """Cross-artifact regression, end to end: overwrite config.yaml (which
    plans config.yaml.bak), then name that .bak as the MCP destination. The
    saved original must survive and result.config_backup must point at it."""
    _prepare(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("ORIGINAL CONFIG", encoding="utf-8")
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                "Write MCP config to": str(tmp_path / "config.yaml.bak"),
                f"{config} exists. What should init do?": "overwrite",
                "New filename": "mcp.json",
            },
        )
    )

    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    assert result.config_backup == tmp_path / "config.yaml.bak"
    assert result.config_backup.read_text(encoding="utf-8") == "ORIGINAL CONFIG"
    assert result.mcp_path == tmp_path / "mcp.json"
    assert json.loads(result.mcp_path.read_text(encoding="utf-8"))["mcpServers"]


def test_the_mcp_backup_cannot_land_on_the_generated_config(tmp_path, scripted_io):
    """Inverse cross-artifact regression, end to end: write the config to the
    absent .mcp.json.bak, then overwrite an existing .mcp.json. The MCP backup
    must step aside to .mcp.json.bak.1 and the generated YAML must survive."""
    _prepare(tmp_path)
    mcp = tmp_path / ".mcp.json"
    mcp.write_text('{"mcpServers": {"other": {"command": "x"}}}', encoding="utf-8")
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                "Write config to": str(tmp_path / ".mcp.json.bak"),
                f"{mcp} exists. What should init do?": "overwrite",
            },
        )
    )

    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    assert result.config_path == tmp_path / ".mcp.json.bak"
    assert result.mcp_backup == tmp_path / ".mcp.json.bak.1"
    # The generated config survived: it is YAML, not the old MCP JSON.
    assert yaml.safe_load(result.config_path.read_text(encoding="utf-8"))["engines"]
    # And the old MCP file was still preserved, just one slot over.
    assert json.loads(result.mcp_backup.read_text(encoding="utf-8"))["mcpServers"]["other"]


def test_no_stray_backup_when_a_later_question_fails(tmp_path, scripted_io):
    """Regression: planning used to back up the config BEFORE asking the MCP
    collision question, so failing there left a .bak behind."""
    _prepare(tmp_path)
    (tmp_path / "config.yaml").write_text("original", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text("{ not json", encoding="utf-8")
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                f"{tmp_path / '.mcp.json'} could not be read. What should init do?": "abort",
                f"{tmp_path / 'config.yaml'} exists. What should init do?": "overwrite",
            },
        )
    )
    with pytest.raises(ValueError):
        run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert not (tmp_path / "config.yaml.bak").exists()
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == "original"


def test_missing_path_warns_but_proceeds(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path, **{PATH_PROMPT: [str(tmp_path / "absent")]}))
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert result.config_path.exists()
    assert any("absent" in note for note in result.notes)


def test_batch_downshift_is_reported(tmp_path, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=0.4)
    assert any("batch" in note.lower() for note in result.notes)


def test_a_budget_too_small_for_the_tier_refuses_before_writing(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path, **{"Chunk granularity": "large"}))
    with pytest.raises(ValueError, match="prefix headroom"):
        run_init(io, cwd=tmp_path, memory_budget_gb=0.02)
    assert not (tmp_path / "config.yaml").exists()


def test_an_illegal_engine_name_re_prompts_rather_than_aborting(tmp_path, scripted_io):
    """Spec §7 row 1: re-ask on reject. A typo must not discard the interview."""
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path, **{"Engine name": ["Bad Name", "docs"]}))

    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert list(config["engines"]) == ["docs"]
    assert any("must match" in line for line in io.echoed)


def test_engine_name_with_dashes_produces_safe_table_and_workflow(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path, **{"Engine name": "my-docs"}))
    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert config["engines"]["my-docs"]["table_name"] == "my_docs_vault"


def test_engine_name_prompt_explains_the_mcp_tool_naming(tmp_path, scripted_io):
    _prepare(tmp_path)
    io = scripted_io(_answers(tmp_path))
    run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)
    assert any("search_docs" in line for line in io.echoed)


@pytest.mark.skipif(not _CASE_INSENSITIVE_FS, reason="case-sensitive filesystem")
def test_case_variant_destinations_cannot_destroy_the_config(tmp_path, scripted_io):
    """End-to-end reproduction of the reported blocker: both destinations
    named the same file in different case; the MCP commit overwrote the YAML
    while InitResult still reported two distinct paths."""
    _prepare(tmp_path)
    io = scripted_io(
        _answers(
            tmp_path,
            **{
                "Write config to": str(tmp_path / "config.yaml"),
                "Write MCP config to": str(tmp_path / "CONFIG.YAML"),
                "New filename": "mcp.json",
            },
        )
    )

    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    assert result.mcp_path == tmp_path / "mcp.json"
    assert yaml.safe_load(result.config_path.read_text(encoding="utf-8"))["engines"]
    assert json.loads(result.mcp_path.read_text(encoding="utf-8"))["mcpServers"]


def test_installed_mode_skips_the_install_dir_question(tmp_path, scripted_io, monkeypatch):
    """A PyPI install has no checkout, so the question is meaningless - not
    merely hard to answer. It must not be asked, and init must still finish."""
    import dbs_vector.services.initializer as init_mod

    _prepare(tmp_path)
    monkeypatch.setattr(init_mod, "detect_install_mode", lambda: ("installed", None))
    io = scripted_io(_answers(tmp_path))

    result = run_init(io, cwd=tmp_path, memory_budget_gb=BUDGET)

    assert "Where is dbs-vector installed?" not in io.asked
    merged = json.loads(result.mcp_path.read_text(encoding="utf-8"))
    assert merged["mcpServers"]["dbs-vector"]["command"] == "dbs-vector"
