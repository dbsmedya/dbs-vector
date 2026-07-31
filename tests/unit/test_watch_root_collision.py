"""One watcher per directory per process — the failure must never be silent.

watchdog's FSEvents backend registers watches by path process-wide and refuses
a second Observer on the same path. It raises that refusal on the emitter
thread, after `Observer.start()` has already returned, so nothing observes it
and the second engine goes unwatched with no error anywhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dbs_vector.infrastructure.watch import watchdog_backend
from dbs_vector.infrastructure.watch.watchdog_backend import (
    WatchdogBackend,
    roots_already_watched,
)
from dbs_vector.services.watcher import WatchedEngine, WatcherService


@pytest.fixture(autouse=True)
def _clean_registry():
    watchdog_backend._SCHEDULED_ROOTS.clear()
    yield
    watchdog_backend._SCHEDULED_ROOTS.clear()


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "vault"
    d.mkdir()
    return str(d)


class TestBackendRegistry:
    def test_second_backend_on_the_same_root_raises_on_the_calling_thread(self, root):
        first = WatchdogBackend(owner="md")
        first.start([root], lambda *a: None)
        try:
            with pytest.raises(RuntimeError, match="already watches"):
                WatchdogBackend(owner="md-granite").start([root], lambda *a: None)
        finally:
            first.stop()

    def test_the_error_names_the_engine_holding_the_root(self, root):
        first = WatchdogBackend(owner="md")
        first.start([root], lambda *a: None)
        try:
            with pytest.raises(RuntimeError, match="already watched by 'md'"):
                WatchdogBackend(owner="md-granite").start([root], lambda *a: None)
        finally:
            first.stop()

    def test_stop_releases_the_root_for_a_later_backend(self, root):
        first = WatchdogBackend(owner="md")
        first.start([root], lambda *a: None)
        first.stop()

        second = WatchdogBackend(owner="md-granite")
        second.start([root], lambda *a: None)  # must not raise
        second.stop()
        assert not roots_already_watched([root])

    def test_a_rejected_claim_leaves_no_partial_registration(self, tmp_path, root):
        """A backend claiming [free, taken] must not keep the free one."""
        free = tmp_path / "other"
        free.mkdir()
        holder = WatchdogBackend(owner="md")
        holder.start([root], lambda *a: None)
        try:
            with pytest.raises(RuntimeError):
                WatchdogBackend(owner="md-granite").start([str(free), root], lambda *a: None)
            assert str(free) not in roots_already_watched([str(free)])
        finally:
            holder.stop()

    def test_distinct_roots_coexist(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        first, second = WatchdogBackend(owner="md"), WatchdogBackend(owner="md-granite")
        first.start([str(a)], lambda *a_: None)
        second.start([str(b)], lambda *a_: None)
        first.stop()
        second.stop()

    def test_equivalent_spellings_of_one_root_collide(self, tmp_path, root):
        """`/vault` and `/vault/../vault` are the same directory to FSEvents."""
        first = WatchdogBackend(owner="md")
        first.start([root], lambda *a: None)
        try:
            with pytest.raises(RuntimeError, match="already watches"):
                WatchdogBackend(owner="md-granite").start(
                    [str(tmp_path / "vault" / ".." / "vault")], lambda *a: None
                )
        finally:
            first.stop()


def _engine(name: str, roots: list[str], backend) -> WatchedEngine:
    path_filter = MagicMock()
    path_filter.active_roots.return_value = [__import__("pathlib").Path(r) for r in roots]
    return WatchedEngine(
        name=name,
        ingestion=MagicMock(),
        path_filter=path_filter,
        store=MagicMock(),
        debounce_seconds=3.0,
        backend=backend,
    )


class TestWatcherServiceReporting:
    def test_shared_roots_are_reported_before_any_backend_starts(self, root, caplog):
        service = WatcherService(
            {
                "md": _engine("md", [root], MagicMock()),
                "md-granite": _engine("md-granite", [root], MagicMock()),
            }
        )
        with caplog.at_level("ERROR"):
            service.start()
        service.stop()

        text = caplog.text
        assert "md" in text and "md-granite" in text
        assert "Only 'md' will receive events" in text

    def test_a_failing_backend_is_reported_as_a_drifting_index(self, root, caplog):
        failing = MagicMock()
        failing.start.side_effect = RuntimeError("this process already watches …")
        service = WatcherService({"md-granite": _engine("md-granite", [root], failing)})

        with caplog.at_level("ERROR"):
            service.start()
        service.stop()

        text = caplog.text
        assert "is NOT being watched" in text
        # The old wording claimed "search is unaffected", which is exactly the
        # comfortable falsehood this test exists to prevent.
        assert "search is unaffected" not in text
        assert "stale" in text

    def test_one_failing_backend_does_not_stop_the_worker(self, root):
        failing = MagicMock()
        failing.start.side_effect = RuntimeError("boom")
        service = WatcherService({"md": _engine("md", [root], failing)})
        service.start()
        try:
            assert service._worker is not None
            assert service._worker.is_alive()
        finally:
            service.stop()
