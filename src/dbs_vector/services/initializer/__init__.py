"""Interactive project initialization.

Orchestration only: every decision lives in a registry, every question goes
through PromptIO, and every file operation lives in render.py. See
docs/README_CONFIGURATION.md for the user-facing workflow.
"""

from pathlib import Path

from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.core.naming import ENGINE_NAME_PATTERN
from dbs_vector.core.profile_math import ProfileTierRegistry, fitting_tiers
from dbs_vector.services.initializer.answers import InitAnswers, InitResult
from dbs_vector.services.initializer.io import PromptIO
from dbs_vector.services.initializer.kinds import EngineKindRegistry
from dbs_vector.services.initializer.render import (
    DestinationPlanner,
    build_config_dict,
    commit_plan,
    detect_install_mode,
    dump_config_yaml,
    dump_mcp_json,
    is_dbs_vector_checkout,
    merge_mcp_config,
    read_mcp_config,
    validate_rendered_config,
)

__all__ = ["run_init"]


def _ask_engine_name(io: PromptIO) -> str:
    """Re-prompt on a bad name (spec §7 row 1).

    A typo must not discard the interview - the same reasoning as the
    destination re-prompt. This is why a scripted test that supplies a single
    bad name must supply a good one after it; ScriptedIO's repeat guard turns
    a missing follow-up into a clear failure rather than a hang.
    """
    while True:
        name = io.ask_text("Engine name", default="md").strip()
        if ENGINE_NAME_PATTERN.match(name):
            return name
        io.echo(
            f"  '{name}' must match {ENGINE_NAME_PATTERN.pattern} (lowercase "
            f"letters, digits, dash, underscore; starting with a letter or "
            f"digit). It becomes an MCP tool name."
        )


def _ask_model(io: PromptIO) -> str:
    options = []
    for key in ModelRegistry.keys():
        contract = ModelRegistry.get(key)
        options.append(
            (
                key,
                f"{contract.model_name} "
                f"({contract.model_max_token_length}-token context, "
                f"{contract.vector_dimension}-dim)",
            )
        )
    default = options[-1][0]
    return io.ask_choice("Embedding model", options, default=default)


def _ask_kind(io: PromptIO) -> str:
    kinds = EngineKindRegistry.values()
    if len(kinds) == 1:
        return kinds[0].key
    return io.ask_choice(
        "What are you indexing?",
        [(k.key, k.label) for k in kinds],
        default=kinds[0].key,
    )


def _ask_tier(io: PromptIO, model_key: str) -> str:
    contract = ModelRegistry.get(model_key)
    available = fitting_tiers(contract)
    if not available:
        smallest = ProfileTierRegistry.values()[0]
        raise ValueError(
            f"Model '{model_key}' has a {contract.model_max_token_length}-token "
            f"context; the smallest granularity needs {smallest.chunk_max_tokens}. "
            f"Choose another model."
        )
    keys = [t.key for t in available]
    default = "medium" if "medium" in keys else keys[0]
    return io.ask_choice(
        "Chunk granularity", [(t.key, t.label) for t in available], default=default
    )


def _ask_install_dir(io: PromptIO, detected_root: Path) -> str:
    raw = io.ask_text("Where is dbs-vector installed?", default=str(detected_root)).strip()
    install = Path(raw).expanduser().resolve()
    if not is_dbs_vector_checkout(install):
        raise ValueError(
            f"{install} does not look like a dbs-vector checkout "
            f"(no dbs-vector pyproject.toml). The MCP server is launched from there."
        )
    return str(install)


def run_init(
    io: PromptIO,
    cwd: Path,
    memory_budget_gb: float | None = None,
) -> InitResult:
    """Run the interview and write both files. Nothing is written until every
    question has been answered and the rendered config has validated."""
    from dbs_vector.infrastructure.hardware import resolve_memory_budget_gb

    notes: list[str] = []

    engine_name = _ask_engine_name(io)
    model_key = _ask_model(io)
    kind_key = _ask_kind(io)
    tier_key = _ask_tier(io, model_key)

    kind = EngineKindRegistry.get(kind_key)
    contract = ModelRegistry.get(model_key)
    kind_answers = kind.ask(io)

    for path in kind_answers.paths:
        if not Path(path).is_dir():
            notes.append(f"Path {path} does not exist yet - it will be skipped until it does.")

    passage_prefix = ""
    query_prefix = ""
    if contract.default_passage_prefix or contract.default_query_prefix:
        passage_prefix = io.ask_text("Passage prefix", default=contract.default_passage_prefix)
        query_prefix = io.ask_text("Query prefix", default=contract.default_query_prefix)

    db_raw = io.ask_text(
        "Where should LanceDB store its tables?",
        default=str(cwd / "lancedb_dbs_vector"),
    ).strip()
    db_path = str(Path(db_raw).expanduser().resolve())

    mode, detected_root = detect_install_mode()
    if mode == "checkout":
        assert detected_root is not None  # guaranteed by detect_install_mode()
        install_dir: str | None = _ask_install_dir(io, detected_root)
    else:
        install_dir = None
        notes.append("dbs-vector is installed on PATH; the MCP entry launches it directly.")

    config_target = Path(
        io.ask_text("Write config to", default=str(cwd / "config.yaml")).strip()
    ).expanduser()
    mcp_target = Path(
        io.ask_text("Write MCP config to", default=str(cwd / ".mcp.json")).strip()
    ).expanduser()

    budget_gb = memory_budget_gb if memory_budget_gb is not None else resolve_memory_budget_gb(None)
    profile = kind.build_profile(contract, tier_key, budget_gb)
    tier = ProfileTierRegistry.get(tier_key)
    if profile["batch_size"] < tier.batch_size:
        notes.append(
            f"Granularity '{tier_key}' prefers batch {tier.batch_size}; your "
            f"{budget_gb:.1f} GB budget allows {profile['batch_size']} - using "
            f"{profile['batch_size']}."
        )

    answers = InitAnswers(
        engine_name=engine_name,
        model_key=model_key,
        kind_key=kind_key,
        tier_key=tier_key,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
        db_path=db_path,
        install_dir=install_dir,
        config_path=str(config_target),
        mcp_path=str(mcp_target),
        kind=kind_answers,
    )

    config = build_config_dict(answers, profile, contract, kind)
    config_text = dump_config_yaml(config, model_key=model_key, budget_gb=budget_gb)
    # The guarantee: refuse to write anything the next command would reject.
    validate_rendered_config(config_text)

    # --- Decide everything. Still nothing on disk. -----------------------
    # An unreadable .mcp.json must not abort the wizard, and must not be
    # overwritten blind: init cannot see what servers it would destroy.
    mcp_readable = True
    try:
        existing_mcp = read_mcp_config(mcp_target)
    except ValueError as exc:
        recovery = io.ask_choice(
            f"{mcp_target} could not be read. What should init do?",
            [
                ("new-file", "Write a fresh MCP config to a different file"),
                ("abort", "Stop and change nothing"),
            ],
            default="new-file",
        )
        if recovery == "abort":
            raise
        notes.append(f"{mcp_target} was unreadable ({exc}); wrote a fresh file instead.")
        existing_mcp = None
        mcp_readable = False

    # The planner accumulates each plan's claims() - primary AND backup - so
    # no destination can be handed out twice. Adding a third artifact is one
    # more planner.plan() call and cannot forget the earlier reservations.
    planner = DestinationPlanner(io)
    config_plan = planner.plan(config_target, alt_name="config.1.yaml")
    mcp_plan = planner.plan(mcp_target, alt_name=".mcp.1.json", allow_overwrite=mcp_readable)

    mcp_text = dump_mcp_json(
        merge_mcp_config(existing_mcp, install_dir, str(config_plan.path.resolve()))
    )

    # --- Act. Every question is answered and the config has validated. ---
    config_backup = commit_plan(config_plan, config_text)
    mcp_backup = commit_plan(mcp_plan, mcp_text)

    return InitResult(
        engine_name=engine_name,
        config_path=config_plan.path,
        mcp_path=mcp_plan.path,
        config_backup=config_backup,
        mcp_backup=mcp_backup,
        notes=notes,
        used_checkout=(mode == "checkout"),
    )
