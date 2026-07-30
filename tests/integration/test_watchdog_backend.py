"""Slow-marked smoke test against the REAL watchdog observer.

Every other watcher test dispatches synthetic events — this one proves the
infrastructure adapter actually translates FSEvents into our port's vocabulary.

    RUN_SLOW_TESTS=1 uv run pytest tests/integration/test_watchdog_backend.py -v
"""

from __future__ import annotations

import os
import threading
import time

import pytest

slow = pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1",
    reason="Real filesystem observer; set RUN_SLOW_TESTS=1 to enable.",
)


@slow
@pytest.mark.slow
def test_watchdog_backend_reports_a_created_file(tmp_path):
    from dbs_vector.infrastructure.watch.watchdog_backend import WatchdogBackend

    seen: list[tuple[str, str, bool, str | None]] = []
    got_event = threading.Event()

    def on_event(path: str, kind: str, is_directory: bool, dest: str | None) -> None:
        seen.append((path, kind, is_directory, dest))
        got_event.set()

    backend = WatchdogBackend()
    backend.start([str(tmp_path)], on_event)
    try:
        time.sleep(0.5)  # let the observer settle
        (tmp_path / "new.md").write_text("# new")
        assert got_event.wait(timeout=10.0), "no filesystem event within 10s"
    finally:
        backend.stop()

    kinds = {kind for _path, kind, _is_dir, _dest in seen}
    assert kinds <= {"created", "modified", "deleted", "moved"}
    assert any(path.endswith("new.md") for path, _k, _d, _dest in seen)
