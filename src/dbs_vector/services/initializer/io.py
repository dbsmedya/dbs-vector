"""The prompting boundary.

Every question the wizard asks goes through this Protocol, so the entire
interview is exercisable from a test without a terminal. Typer implements it
in cli.py; tests use a scripted fake. Nothing in this package imports typer.
"""

from typing import Protocol


class PromptIO(Protocol):
    """One-way output plus four question shapes.

    `prompt` doubles as a STABLE KEY: implementations may key scripted
    answers off it, so changing a prompt string is a test-visible change.
    """

    def echo(self, message: str) -> None:
        """Emit a line of output. Never a question."""
        ...

    def ask_text(self, prompt: str, default: str = "") -> str: ...

    def ask_choice(self, prompt: str, options: list[tuple[str, str]], default: str) -> str:
        """`options` are (key, label) pairs. Returns the chosen key."""
        ...

    def ask_multi(self, prompt: str, options: list[str], default: list[str]) -> list[str]: ...

    def ask_bool(self, prompt: str, default: bool) -> bool: ...
