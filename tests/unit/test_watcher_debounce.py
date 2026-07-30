# tests/unit/test_watcher_debounce.py
from unittest.mock import MagicMock

import pytest

from dbs_vector.services.path_filter import PathFilter
from dbs_vector.services.watcher import WatchedEngine, WatcherService


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    (root / "a.md").write_text("# a")
    return root


def _engine(name, vault, debounce=3.0):
    pf = PathFilter(
        roots=[str(vault)],
        extensions=[".md"],
        ignore_patterns=[".#*", "*~", "*.tmp", ".DS_Store"],
    )
    ingestion = MagicMock()
    ingestion.reconcile.return_value = (0, 0)  # real signature: (pruned, upserted)
    return WatchedEngine(
        name=name,
        ingestion=ingestion,
        path_filter=pf,
        store=MagicMock(),
        debounce_seconds=debounce,
        backend=MagicMock(),
    )


@pytest.fixture
def service(vault):
    clock = FakeClock()
    engines = {"md": _engine("md", vault), "md-granite": _engine("md-granite", vault)}
    svc = WatcherService(engines, clock=clock, fts_interval_seconds=60.0)
    return svc, clock, engines


class TestDebounceWindow:
    def test_action_does_not_fire_before_the_window_elapses(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "a.md"), "modified")

        clock.advance(2.0)
        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_not_called()

    def test_action_fires_once_the_window_elapses(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "a.md"), "modified")

        clock.advance(3.0)
        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_called_once_with(str(vault / "a.md"))

    def test_a_later_event_resets_the_window(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "a.md"), "modified")
        clock.advance(2.0)
        svc.handle_event("md", str(vault / "a.md"), "modified")
        clock.advance(2.0)

        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_not_called()

    def test_zero_debounce_fires_immediately(self, vault):
        clock = FakeClock()
        engines = {"md": _engine("md", vault, debounce=0.0)}
        svc = WatcherService(engines, clock=clock)
        svc.handle_event("md", str(vault / "a.md"), "created")

        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_called_once()


class TestLastStateWins:
    def test_modify_then_delete_removes(self, service, vault):
        svc, clock, engines = service
        path = str(vault / "a.md")
        svc.handle_event("md", path, "modified")
        svc.handle_event("md", path, "deleted")
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.remove_source.assert_called_once_with(path)
        engines["md"].ingestion.upsert_file.assert_not_called()

    def test_delete_then_create_upserts(self, service, vault):
        svc, clock, engines = service
        path = str(vault / "a.md")
        svc.handle_event("md", path, "deleted")
        svc.handle_event("md", path, "created")
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_called_once_with(path)
        engines["md"].ingestion.remove_source.assert_not_called()

    def test_move_produces_two_entries(self, service, vault):
        svc, clock, engines = service
        old = str(vault / "a.md")
        new = str(vault / "b.md")
        svc.handle_event("md", old, "moved", dest=new)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.remove_source.assert_called_once_with(old)
        engines["md"].ingestion.upsert_file.assert_called_once_with(new)


class TestKeying:
    def test_engines_do_not_swallow_each_others_events(self, service, vault):
        svc, clock, engines = service
        path = str(vault / "a.md")
        svc.handle_event("md", path, "modified")
        svc.handle_event("md-granite", path, "modified")
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_called_once_with(path)
        engines["md-granite"].ingestion.upsert_file.assert_called_once_with(path)


class TestFiltering:
    def test_ignored_file_never_schedules_an_upsert(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "scratch.tmp"), "created")
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.upsert_file.assert_not_called()

    def test_delete_outside_active_roots_is_ignored(self, service):
        svc, clock, engines = service
        svc.handle_event("md", "/somewhere/else/x.md", "deleted")
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.remove_source.assert_not_called()


class TestReconcileScheduling:
    def test_directory_event_schedules_a_reconcile(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()

    def test_reconcile_absorbs_that_engines_pending_file_actions(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "a.md"), "modified")
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()
        engines["md"].ingestion.upsert_file.assert_not_called()

    def test_file_events_never_reset_the_reconcile_window(self, service, vault):
        """Editing must not be able to starve a pending reconcile."""
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        for _ in range(4):
            clock.advance(1.0)
            svc.handle_event("md", str(vault / "a.md"), "modified")

        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()

    def test_only_one_pending_reconcile_per_engine(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "s1"), "created", is_directory=True)
        svc.handle_event("md", str(vault / "s2"), "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()

    def test_directory_event_outside_active_roots_is_ignored(self, service):
        svc, clock, engines = service
        svc.handle_event("md", "/elsewhere/sub", "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].ingestion.reconcile.assert_not_called()

    def test_changed_reconcile_rebuilds_indices(self, service, vault):
        svc, clock, engines = service
        engines["md"].ingestion.reconcile.return_value = (1, 2)
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].store.create_indices.assert_called_once()
        engines["md"].store.compact.assert_called_once()

    def test_no_op_reconcile_skips_index_work(self, service, vault):
        svc, clock, engines = service  # reconcile returns (0, 0) by default
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        clock.advance(3.0)

        svc.run_once()

        engines["md"].store.create_indices.assert_not_called()


class TestFtsTimer:
    def test_dirty_engine_is_refreshed_after_the_interval(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "a.md"), "modified")
        clock.advance(3.0)
        svc.run_once()

        engines["md"].store.refresh_fts.assert_not_called()

        clock.advance(60.0)
        svc.run_once()

        engines["md"].store.refresh_fts.assert_called_once()

    def test_clean_engine_is_not_refreshed(self, service):
        svc, clock, engines = service

        clock.advance(60.0)
        svc.run_once()

        engines["md"].store.refresh_fts.assert_not_called()
        engines["md-granite"].store.refresh_fts.assert_not_called()


class TestFailureIsolation:
    def test_a_failing_action_does_not_stop_the_others(self, service, vault):
        svc, clock, engines = service
        engines["md"].ingestion.upsert_file.side_effect = RuntimeError("boom")
        svc.handle_event("md", str(vault / "a.md"), "modified")
        svc.handle_event("md-granite", str(vault / "a.md"), "modified")
        clock.advance(3.0)

        svc.run_once()  # must not raise

        engines["md-granite"].ingestion.upsert_file.assert_called_once()


class TestReconcileStarvation:
    """Real FSEvents emits a directory `modified` for the PARENT whenever a
    child file changes (verified against the watchdog backend on macOS). Those
    incidental events must never defer an already-scheduled reconcile — the
    spec and docs/README_WATCH.md both promise editing cannot starve it."""

    def test_directory_event_does_not_extend_an_already_pending_reconcile(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)  # due at 1003

        clock.advance(1.0)
        svc.handle_event("md", str(vault / "sub"), "modified", is_directory=True)
        clock.advance(2.0)  # t = 1003: the ORIGINAL window has elapsed
        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()

    def test_editing_files_cannot_starve_a_pending_reconcile(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)

        for _ in range(5):
            clock.advance(1.0)
            svc.handle_event("md", str(vault / "a.md"), "modified")
            # The FSEvents side effect of writing a child file:
            svc.handle_event("md", str(vault), "modified", is_directory=True)
        svc.run_once()

        engines["md"].ingestion.reconcile.assert_called_once()

    def test_a_fresh_directory_event_schedules_again_after_a_reconcile_runs(self, service, vault):
        svc, clock, engines = service
        svc.handle_event("md", str(vault / "sub"), "created", is_directory=True)
        clock.advance(3.0)
        svc.run_once()

        svc.handle_event("md", str(vault / "sub2"), "created", is_directory=True)
        clock.advance(3.0)
        svc.run_once()

        assert engines["md"].ingestion.reconcile.call_count == 2


class TestStartFailureContainment:
    """A backend that refuses to start must not take down MCP startup for
    every other engine (spec: failures log and never kill the MCP server)."""

    def test_a_failing_backend_does_not_propagate(self, vault):
        clock = FakeClock()
        broken, healthy = _engine("md", vault), _engine("md-granite", vault)
        broken.backend.start.side_effect = PermissionError("root not readable")
        svc = WatcherService({"md": broken, "md-granite": healthy}, clock=clock)

        svc.start()  # must not raise
        try:
            healthy.backend.start.assert_called_once()
        finally:
            svc.stop()

    def test_the_worker_still_starts_when_a_backend_fails(self, vault):
        clock = FakeClock()
        broken = _engine("md", vault)
        broken.backend.start.side_effect = PermissionError("root not readable")
        svc = WatcherService({"md": broken}, clock=clock)

        svc.start()
        try:
            assert svc._worker is not None
        finally:
            svc.stop()


class TestShutdownRace:
    """stop() must never write the store while the worker thread is still
    running — that would break the one-writer invariant."""

    def test_index_maintenance_is_skipped_when_the_worker_is_still_busy(self, vault, monkeypatch):
        import threading as _threading

        from dbs_vector.services import watcher as watcher_module

        monkeypatch.setattr(watcher_module, "_WORKER_JOIN_TIMEOUT", 0.2)

        engine = _engine("md", vault, debounce=0.0)
        entered = _threading.Event()
        blocked = _threading.Event()

        def _blocking_reconcile():
            entered.set()
            blocked.wait(30)
            return (1, 1)

        engine.ingestion.reconcile.side_effect = _blocking_reconcile

        svc = WatcherService({"md": engine})  # real clock: start() seeds an immediate reconcile
        svc.start()
        try:
            assert entered.wait(5.0), "worker never entered reconcile"
            # Without this the stop() loop has nothing to iterate and the test
            # would pass even with the race present.
            svc._dirty.add("md")

            svc.stop()

            engine.store.create_indices.assert_not_called()
            engine.store.compact.assert_not_called()
        finally:
            blocked.set()

    def test_index_maintenance_still_runs_when_the_worker_has_exited(self, vault, monkeypatch):
        from dbs_vector.services import watcher as watcher_module

        monkeypatch.setattr(watcher_module, "_WORKER_JOIN_TIMEOUT", 5.0)

        engine = _engine("md", vault)
        svc = WatcherService({"md": engine})
        svc.start()
        svc._dirty.add("md")  # as any prior per-file write would have left it

        svc.stop()

        engine.store.create_indices.assert_called_once()
        engine.store.compact.assert_called_once()
