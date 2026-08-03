"""Pytest configuration and fixtures."""

import pytest
from loguru import logger


@pytest.fixture
def caplog(caplog):
    """Enable Loguru logging to be captured by pytest's caplog fixture."""
    import logging

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def temp_model_registry():
    """Snapshot and restore ModelRegistry so a test may register synthetic models.

    Required: tests/unit/test_model_registry.py asserts the exact key list
    ['gemma-bf16', 'granite-r2'], so a leaked registration fails an unrelated
    test. ModelRegistry has no _reset_for_testing(), hence the manual snapshot.
    """
    from dbs_vector.core.model_registry import ModelRegistry

    snapshot = dict(ModelRegistry._models)
    yield ModelRegistry
    ModelRegistry._models.clear()
    ModelRegistry._models.update(snapshot)


_MAX_REPEATS = 100


class ScriptedIO:
    """A PromptIO double driven by a dict keyed by prompt string.

    An unscripted prompt returns its default, so a test scripts only the
    answers it cares about.

    A LIST value is consumed one element per call for ask_text / ask_choice /
    ask_bool, falling back to the default once exhausted. Looping prompts
    REQUIRE this: DocumentKind.ask() reads paths until it gets a blank, so a
    scalar answer would repeat forever. ask_multi is the exception - a list
    there is the literal answer.
    """

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = dict(answers)
        self._queues: dict[str, list[object]] = {}
        self._counts: dict[str, int] = {}
        self.echoed: list[str] = []
        self.asked: list[str] = []

    def _record(self, prompt: str) -> None:
        self.asked.append(prompt)
        self._counts[prompt] = self._counts.get(prompt, 0) + 1
        # A scalar answer to a looping prompt would spin until the CI job
        # times out. Fail loudly instead.
        assert self._counts[prompt] <= _MAX_REPEATS, (
            f"Prompt {prompt!r} asked {_MAX_REPEATS} times - the wizard is "
            f"looping. Script a list value so the answer sequence terminates."
        )

    def _scalar(self, prompt: str, default: object) -> object:
        self._record(prompt)
        if prompt not in self.answers:
            return default
        value = self.answers[prompt]
        if isinstance(value, list):
            queue = self._queues.setdefault(prompt, list(value))
            return queue.pop(0) if queue else default
        return value

    def echo(self, message: str) -> None:
        self.echoed.append(message)

    def ask_text(self, prompt: str, default: str = "") -> str:
        return str(self._scalar(prompt, default))

    def ask_choice(self, prompt: str, options: list[tuple[str, str]], default: str) -> str:
        chosen = str(self._scalar(prompt, default))
        valid = [key for key, _ in options]
        if chosen not in valid:
            raise ValueError(f"Scripted answer '{chosen}' for '{prompt}' not in {valid}")
        return chosen

    def ask_multi(self, prompt: str, options: list[str], default: list[str]) -> list[str]:
        self._record(prompt)
        chosen = self.answers.get(prompt, default)
        assert isinstance(chosen, list)
        unknown = [c for c in chosen if c not in options]
        if unknown:
            raise ValueError(f"Scripted answers {unknown} for '{prompt}' not in {options}")
        return list(chosen)

    def ask_bool(self, prompt: str, default: bool) -> bool:
        return bool(self._scalar(prompt, default))


@pytest.fixture
def scripted_io():
    """Factory for ScriptedIO. Available to tests/unit and tests/integration
    alike - neither directory is a package, so a module-level import of a
    helper would not resolve from both."""
    return ScriptedIO
