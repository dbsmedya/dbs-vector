import unicodedata

import pytest

from dbs_vector.services.initializer.render import (
    _CASE_INSENSITIVE_FS,
    DestinationPlan,
    DestinationPlanner,
    commit_plan,
    plan_destination,
    reservation_key,
    write_atomic,
)


def test_absent_target_needs_no_question(tmp_path, scripted_io):
    target = tmp_path / "config.yaml"
    io = scripted_io({})
    plan = plan_destination(io, target, alt_name="config.1.yaml")
    assert plan == DestinationPlan(path=target, backup_from=None)
    assert io.asked == []


def test_planning_writes_nothing(tmp_path, scripted_io):
    """The §9 guarantee: planning is pure. Only commit_plan touches disk."""
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    io = scripted_io({f"{target} exists. What should init do?": "overwrite"})

    plan = plan_destination(io, target, alt_name="config.1.yaml")

    assert plan == DestinationPlan(
        path=target, backup_from=target, backup_to=tmp_path / "config.yaml.bak"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]
    assert target.read_text(encoding="utf-8") == "original"


def test_the_backup_destination_is_fixed_at_plan_time(tmp_path, scripted_io):
    """It must be knowable before any write, so it can be reserved."""
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "config.yaml.bak").write_text("older", encoding="utf-8")
    io = scripted_io({f"{target} exists. What should init do?": "overwrite"})

    plan = plan_destination(io, target, alt_name="config.1.yaml")

    assert plan.backup_to == tmp_path / "config.yaml.bak.1"
    assert plan.claims() == (target, tmp_path / "config.yaml.bak.1")


def test_claims_is_just_the_path_when_no_backup_is_planned(tmp_path):
    plan = DestinationPlan(path=tmp_path / "a.yaml", backup_from=None)
    assert plan.claims() == (tmp_path / "a.yaml",)


def test_commit_backs_up_then_writes(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    plan = DestinationPlan(path=target, backup_from=target, backup_to=tmp_path / "config.yaml.bak")

    backup = commit_plan(plan, "new content")

    assert backup == tmp_path / "config.yaml.bak"
    assert backup.read_text(encoding="utf-8") == "original"
    assert target.read_text(encoding="utf-8") == "new content"


def test_commit_backup_does_not_clobber_an_existing_bak(tmp_path, scripted_io):
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "config.yaml.bak").write_text("older", encoding="utf-8")
    io = scripted_io({f"{target} exists. What should init do?": "overwrite"})

    backup = commit_plan(plan_destination(io, target, alt_name="config.1.yaml"), "new")

    assert backup == tmp_path / "config.yaml.bak.1"
    assert (tmp_path / "config.yaml.bak").read_text(encoding="utf-8") == "older"


def test_a_reserved_backup_path_is_refused_as_a_destination(tmp_path, scripted_io):
    """Cross-artifact regression: reserving only the primary path let the
    second artifact be routed onto the FIRST artifact's backup, destroying the
    saved original while InitResult still reported it as preserved."""
    config = tmp_path / "config.yaml"
    config.write_text("original", encoding="utf-8")
    io_cfg = scripted_io({f"{config} exists. What should init do?": "overwrite"})
    config_plan = plan_destination(io_cfg, config, alt_name="config.1.yaml")
    assert config_plan.backup_to == tmp_path / "config.yaml.bak"

    # The user now names that very .bak as the MCP destination. It does not
    # exist yet, so only `reserved` can catch it.
    io_mcp = scripted_io({"New filename": "mcp.json"})
    mcp_plan = plan_destination(
        io_mcp,
        tmp_path / "config.yaml.bak",
        alt_name=".mcp.1.json",
        reserved=config_plan.claims(),
    )

    assert mcp_plan.path == tmp_path / "mcp.json"
    assert any("already being written by this run" in line for line in io_mcp.echoed)


def test_new_name_branch_uses_the_default(tmp_path, scripted_io):
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    io = scripted_io({f"{target} exists. What should init do?": "new-name"})

    plan = plan_destination(io, target, alt_name="config.1.yaml")

    assert plan.path == tmp_path / "config.1.yaml"
    assert plan.backup_from is None


def test_new_name_branch_accepts_a_user_supplied_name(tmp_path, scripted_io):
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    io = scripted_io(
        {
            f"{target} exists. What should init do?": "new-name",
            "New filename": "mine.yaml",
        }
    )

    plan = plan_destination(io, target, alt_name="config.1.yaml")

    assert plan.path == tmp_path / "mine.yaml"


def test_a_taken_new_name_re_prompts_rather_than_failing(tmp_path, scripted_io):
    """§9 says re-prompt. Raising would discard the whole interview because
    the user typed one filename that happened to exist."""
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "taken.yaml").write_text("taken", encoding="utf-8")
    io = scripted_io(
        {
            f"{target} exists. What should init do?": "new-name",
            "New filename": ["taken.yaml", "free.yaml"],
        }
    )

    plan = plan_destination(io, target, alt_name="config.1.yaml")

    assert plan.path == tmp_path / "free.yaml"
    assert any("taken.yaml" in line for line in io.echoed)


def test_a_reserved_path_is_never_planned_even_when_absent(tmp_path, scripted_io):
    """Regression: two plans for one path meant the second commit silently
    destroyed the first file. An absent target is normally accepted without a
    question - being reserved must override that."""
    target = tmp_path / "shared.json"
    io = scripted_io({"New filename": "mcp.json"})

    plan = plan_destination(
        io, target, alt_name=".mcp.1.json", reserved=(tmp_path / "shared.json",)
    )

    assert plan.path == tmp_path / "mcp.json"
    assert any("already being written by this run" in line for line in io.echoed)


def test_a_reserved_path_is_not_offered_for_overwrite(tmp_path, scripted_io):
    """Overwriting a path this run already claimed is never a resolution."""
    target = tmp_path / "shared.json"
    target.write_text("existing", encoding="utf-8")
    io = scripted_io({"New filename": "mcp.json"})

    plan = plan_destination(io, target, alt_name=".mcp.1.json", reserved=(target,))

    assert plan.path == tmp_path / "mcp.json"
    assert f"{target} exists. What should init do?" not in io.asked


def test_a_reserved_new_name_re_prompts(tmp_path, scripted_io):
    target = tmp_path / ".mcp.json"
    target.write_text("{}", encoding="utf-8")
    io = scripted_io(
        {
            f"{target} exists. What should init do?": "new-name",
            "New filename": ["config.yaml", "mcp.new.json"],
        }
    )

    plan = plan_destination(
        io, target, alt_name=".mcp.1.json", reserved=(tmp_path / "config.yaml",)
    )

    assert plan.path == tmp_path / "mcp.new.json"


def test_a_backup_never_lands_on_a_reserved_path(tmp_path, scripted_io):
    """The inverse cross-artifact case: a LATER plan's backup must not land on
    an EARLIER plan's primary output. The backup name is derived from the
    target, so `reserved` is the only thing that can steer it - the candidate
    does not exist yet, so the existence check passes."""
    mcp = tmp_path / ".mcp.json"
    mcp.write_text('{"OLD": "MCP"}', encoding="utf-8")

    # Earlier plan claims the absent .mcp.json.bak as its own output.
    config_plan = plan_destination(
        scripted_io({}), tmp_path / ".mcp.json.bak", alt_name="config.1.yaml"
    )
    assert config_plan.path == tmp_path / ".mcp.json.bak"

    io = scripted_io({f"{mcp} exists. What should init do?": "overwrite"})
    mcp_plan = plan_destination(io, mcp, alt_name=".mcp.1.json", reserved=config_plan.claims())

    assert mcp_plan.path == mcp
    assert mcp_plan.backup_to == tmp_path / ".mcp.json.bak.1"


def test_backup_skips_both_existing_and_reserved_candidates(tmp_path, scripted_io):
    """.bak taken on disk AND .bak.1 reserved -> .bak.2."""
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "config.yaml.bak").write_text("older", encoding="utf-8")
    io = scripted_io({f"{target} exists. What should init do?": "overwrite"})

    plan = plan_destination(
        io, target, alt_name="config.1.yaml", reserved=(tmp_path / "config.yaml.bak.1",)
    )

    assert plan.backup_to == tmp_path / "config.yaml.bak.2"


def test_allow_overwrite_false_skips_the_choice(tmp_path, scripted_io):
    """Used when the existing file is unreadable: overwriting it blind would
    destroy servers init could not parse."""
    target = tmp_path / ".mcp.json"
    target.write_text("{ broken", encoding="utf-8")
    io = scripted_io({"New filename": "mcp.new.json"})

    plan = plan_destination(io, target, alt_name=".mcp.1.json", allow_overwrite=False)

    assert plan.path == tmp_path / "mcp.new.json"
    assert plan.backup_from is None
    assert f"{target} exists. What should init do?" not in io.asked


def test_planner_reserves_each_plans_claims(tmp_path, scripted_io):
    target = tmp_path / "config.yaml"
    target.write_text("original", encoding="utf-8")
    planner = DestinationPlanner(
        scripted_io({f"{target} exists. What should init do?": "overwrite"})
    )

    plan = planner.plan(target, alt_name="config.1.yaml")

    assert planner.reserved == {
        reservation_key(target),
        reservation_key(tmp_path / "config.yaml.bak"),
    }
    assert plan.backup_to == tmp_path / "config.yaml.bak"


def test_planner_starts_with_nothing_reserved(scripted_io):
    assert DestinationPlanner(scripted_io({})).reserved == frozenset()


def test_planner_feeds_earlier_claims_into_later_plans(tmp_path, scripted_io):
    """Two artifacts: the second must not be handed the first's path."""
    planner = DestinationPlanner(scripted_io({"New filename": "mcp.json"}))

    first = planner.plan(tmp_path / "shared.json", alt_name="config.1.yaml")
    second = planner.plan(tmp_path / "shared.json", alt_name=".mcp.1.json")

    assert first.path == tmp_path / "shared.json"
    assert second.path == tmp_path / "mcp.json"


def test_planner_survives_a_third_artifact(tmp_path, scripted_io):
    """The case hand-threading would get wrong: a third plan must avoid BOTH
    earlier plans' claims, not just the most recent one."""
    planner = DestinationPlanner(scripted_io({"New filename": ["second.json", "third.json"]}))

    first = planner.plan(tmp_path / "a.json", alt_name="a.1.json")
    second = planner.plan(tmp_path / "a.json", alt_name="a.2.json")
    third = planner.plan(tmp_path / "a.json", alt_name="a.3.json")

    paths = [first.path, second.path, third.path]
    assert paths == [tmp_path / "a.json", tmp_path / "second.json", tmp_path / "third.json"]
    assert len(set(paths)) == 3


def test_planner_reserved_is_read_only(tmp_path, scripted_io):
    """Callers must not be able to drop a reservation."""
    planner = DestinationPlanner(scripted_io({}))
    planner.plan(tmp_path / "a.json", alt_name="a.1.json")
    snapshot = planner.reserved

    with pytest.raises(AttributeError):
        snapshot.add(tmp_path / "b.json")  # type: ignore[attr-defined]

    assert planner.reserved == snapshot


def test_write_atomic_creates_the_file(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    write_atomic(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_write_atomic_leaves_no_temp_files(tmp_path):
    target = tmp_path / "out.txt"
    write_atomic(target, "hello")
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_write_atomic_replaces_existing_content(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    write_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_reservation_key_nfc_normalizes(tmp_path):
    """macOS stores decomposed forms; the same name typed two ways is one file."""
    composed = tmp_path / unicodedata.normalize("NFC", "café.yaml")
    decomposed = tmp_path / unicodedata.normalize("NFD", "café.yaml")
    assert reservation_key(composed) == reservation_key(decomposed)


@pytest.mark.skipif(not _CASE_INSENSITIVE_FS, reason="case-sensitive filesystem")
def test_reservation_key_folds_case_where_the_filesystem_does(tmp_path):
    assert reservation_key(tmp_path / "config.yaml") == reservation_key(tmp_path / "CONFIG.YAML")


@pytest.mark.skipif(not _CASE_INSENSITIVE_FS, reason="case-sensitive filesystem")
def test_case_variant_destinations_collide(tmp_path, scripted_io):
    """Regression: config.yaml and CONFIG.YAML are ONE file here. resolve()
    reports them as distinct, so only a normalized key catches it. Neither
    exists yet, so the existence check cannot help."""
    planner = DestinationPlanner(scripted_io({"New filename": "mcp.json"}))

    first = planner.plan(tmp_path / "config.yaml", alt_name="config.1.yaml")
    second = planner.plan(tmp_path / "CONFIG.YAML", alt_name=".mcp.1.json")

    assert first.path == tmp_path / "config.yaml"
    assert second.path == tmp_path / "mcp.json"


@pytest.mark.skipif(not _CASE_INSENSITIVE_FS, reason="case-sensitive filesystem")
def test_a_case_variant_backup_is_also_refused(tmp_path, scripted_io):
    """The same flaw in the backup arm: a later plan's backup must not land on
    an earlier plan's primary output spelled with different case."""
    mcp = tmp_path / ".mcp.json"
    mcp.write_text('{"OLD": "MCP"}', encoding="utf-8")
    planner = DestinationPlanner(scripted_io({f"{mcp} exists. What should init do?": "overwrite"}))

    first = planner.plan(tmp_path / ".MCP.JSON.BAK", alt_name="config.1.yaml")
    second = planner.plan(mcp, alt_name=".mcp.1.json")

    assert first.path == tmp_path / ".MCP.JSON.BAK"
    assert second.backup_to != tmp_path / ".mcp.json.bak"
