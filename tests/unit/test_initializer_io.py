import pytest

from dbs_vector.services.initializer.io import PromptIO


def test_scripted_io_satisfies_the_protocol(scripted_io):
    io: PromptIO = scripted_io({})
    assert io is not None


def test_scripted_io_returns_scripted_answers(scripted_io):
    io = scripted_io({"engine_name": "docs", "watch": True})
    assert io.ask_text("engine_name", default="md") == "docs"
    assert io.ask_bool("watch", default=False) is True


def test_scripted_io_falls_back_to_default_when_unscripted(scripted_io):
    io = scripted_io({})
    assert io.ask_text("engine_name", default="md") == "md"
    assert io.ask_bool("watch", default=False) is False
    assert io.ask_choice("model", [("a", "A"), ("b", "B")], default="b") == "b"
    assert io.ask_multi("filters", ["x", "y"], default=["x"]) == ["x"]


def test_a_scalar_answer_repeats(scripted_io):
    io = scripted_io({"name": "docs"})
    assert io.ask_text("name", default="") == "docs"
    assert io.ask_text("name", default="") == "docs"


def test_a_list_answer_is_consumed_sequentially(scripted_io):
    """Required by looping prompts: the path loop ends on a blank answer."""
    io = scripted_io({"path": ["/a", "/b"]})
    assert io.ask_text("path", default="") == "/a"
    assert io.ask_text("path", default="") == "/b"
    assert io.ask_text("path", default="") == ""  # exhausted -> default


def test_ask_multi_treats_a_list_as_the_literal_answer(scripted_io):
    io = scripted_io({"filters": ["x", "y"]})
    assert io.ask_multi("filters", ["x", "y", "z"], default=[]) == ["x", "y"]
    assert io.ask_multi("filters", ["x", "y", "z"], default=[]) == ["x", "y"]


def test_repeating_one_prompt_forever_raises_instead_of_hanging(scripted_io):
    """A scalar answer to a looping prompt is a test bug; fail loudly rather
    than spin until the CI job times out."""
    io = scripted_io({"path": "/never-blank"})
    with pytest.raises(AssertionError, match="asked 100 times"):
        for _ in range(200):
            io.ask_text("path", default="")


def test_scripted_io_records_echoes(scripted_io):
    io = scripted_io({})
    io.echo("hello")
    assert io.echoed == ["hello"]


def test_scripted_io_rejects_a_choice_outside_the_options(scripted_io):
    """Guards against a test scripting an answer the real UI could not produce."""
    io = scripted_io({"model": "nonexistent"})
    with pytest.raises(ValueError, match="nonexistent"):
        io.ask_choice("model", [("a", "A")], default="a")
