"""Watch scheduling: debounce map + one worker thread + a 60s FTS timer.

Pure services layer — it consumes the `IWatchBackend` port and never imports
watchdog. Tests dispatch synthetic events into `handle_event()` and drive
`run_once()` with a fake clock; no real observers, no real timers.

Concurrency: exactly ONE worker thread performs every LanceDB write, so
writes are serialized by construction. `MLXEmbedder`'s per-model lock covers
cross-thread embeds; searches may queue behind at most one watcher embed batch
during a large reconcile. Accepted.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from dbs_vector.core.ports import IWatchBackend
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.path_filter import PathFilter

_UPSERT = "upsert"
_REMOVE = "remove"

# How long stop() waits for the worker to finish its current action.
_WORKER_JOIN_TIMEOUT = 5.0


@dataclass
class WatchedEngine:
    """One engine's watch stack. The backend is per-engine because the port's
    callback carries no engine field — identity comes from the closure."""

    name: str
    ingestion: IngestionService
    path_filter: PathFilter
    store: Any  # IVectorStore
    debounce_seconds: float
    backend: IWatchBackend


@dataclass
class _Pending:
    kind: str  # _UPSERT | _REMOVE
    due: float


@dataclass
class _State:
    """Everything the worker consumes. Guarded by WatcherService._cond."""

    pending: dict[tuple[str, str], _Pending] = field(default_factory=dict)
    reconcile_due: dict[str, float] = field(default_factory=dict)


class WatcherService:
    def __init__(
        self,
        engines: dict[str, WatchedEngine],
        clock: Callable[[], float] = time.monotonic,
        fts_interval_seconds: float = 60.0,
    ) -> None:
        self._engines = engines
        self._clock = clock
        self._fts_interval = fts_interval_seconds
        self._state = _State()
        self._cond = threading.Condition()
        self._stopping = False
        self._worker: threading.Thread | None = None
        # Engines written since their last FTS refresh. Only the worker thread
        # mutates this, so it needs no lock.
        self._dirty: set[str] = set()
        self._next_fts = self._clock() + self._fts_interval

    # ---- scheduling (called from backend threads) ----------------------

    def handle_event(
        self,
        engine: str,
        path: str,
        kind: str,
        is_directory: bool = False,
        dest: str | None = None,
    ) -> None:
        """Translate one filesystem event into at most two pending actions.

        Coalescing is last-state-wins: one pending action per (engine, path)
        key — path-only keying would let md and md-granite swallow each other's
        events — and a later event replaces the earlier one AND resets its
        window.
        """
        watched = self._engines.get(engine)
        if watched is None:
            return
        now = self._clock()
        pf = watched.path_filter

        with self._cond:
            if is_directory:
                # Directory events bypass the PathFilter (they have no
                # extension); only active-root membership is checked. A
                # spurious reconcile is a cheap no-op. FSEvents child delivery
                # is unreliable — never enumerate children from the event.
                if pf.root_for(path) is not None or (dest and pf.root_for(dest) is not None):
                    # setdefault, NOT assignment: a pending reconcile keeps its
                    # original due time. Real FSEvents emits a directory
                    # `modified` for the PARENT whenever a child file changes,
                    # so assignment would let ordinary editing defer the
                    # reconcile forever — and would also push back the
                    # immediate startup reconcile that start() seeds.
                    self._state.reconcile_due.setdefault(engine, now + watched.debounce_seconds)
            elif kind == "moved":
                # Same root gate as plain deletes: never touch unmanaged sources.
                if pf.root_for(path) is not None:
                    self._schedule(engine, path, _REMOVE, now, watched)
                if dest and pf.is_ingestable(dest):
                    self._schedule(engine, dest, _UPSERT, now, watched)
            elif kind == "deleted":
                # Deletes never touch sources outside the active roots.
                if pf.root_for(path) is not None:
                    self._schedule(engine, path, _REMOVE, now, watched)
            elif pf.is_ingestable(path):  # created / modified
                self._schedule(engine, path, _UPSERT, now, watched)
            self._cond.notify()

    def _schedule(
        self, engine: str, path: str, kind: str, now: float, watched: WatchedEngine
    ) -> None:
        self._state.pending[(engine, str(path))] = _Pending(
            kind=kind, due=now + watched.debounce_seconds
        )

    # ---- execution (worker thread; tests call run_once directly) --------

    def run_once(self, now: float | None = None) -> None:
        """Execute everything due at `now`. Never raises."""
        now = self._clock() if now is None else now

        with self._cond:
            reconciles = [name for name, due in self._state.reconcile_due.items() if due <= now]
            for name in reconciles:
                del self._state.reconcile_due[name]
                # A reconcile walks live disk, so it absorbs that engine's
                # pending per-file actions.
                for key in [k for k in self._state.pending if k[0] == name]:
                    del self._state.pending[key]

            actions = [(key, p.kind) for key, p in self._state.pending.items() if p.due <= now]
            for key, _kind in actions:
                del self._state.pending[key]

            fts_due = now >= self._next_fts
            if fts_due:
                self._next_fts = now + self._fts_interval

        for name in reconciles:
            self._guard(f"reconcile {name}", self._do_reconcile, name)
        for (name, path), kind in actions:
            self._guard(f"{kind} {path}", self._do_action, name, path, kind)
        if fts_due:
            for name in sorted(self._dirty):
                self._guard(f"refresh_fts {name}", self._do_refresh_fts, name)
            self._dirty.clear()

    def _guard(self, label: str, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 — the worker must never die
            logger.warning("Watch action failed ({}): {}", label, e)

    def _do_action(self, engine: str, path: str, kind: str) -> None:
        watched = self._engines[engine]
        if kind == _UPSERT:
            watched.ingestion.upsert_file(path)
        else:
            watched.ingestion.remove_source(path)
        self._dirty.add(engine)

    def _do_reconcile(self, engine: str) -> None:
        watched = self._engines[engine]
        pruned, upserted = watched.ingestion.reconcile()
        if not (pruned or upserted):
            return
        # create_indices() rebuilds the FTS index too, so the engine is clean.
        watched.store.create_indices()
        watched.store.compact()
        self._dirty.discard(engine)

    def _do_refresh_fts(self, engine: str) -> None:
        self._engines[engine].store.refresh_fts()

    # ---- lifecycle ------------------------------------------------------

    def _warn_on_shared_roots(self) -> None:
        """Name the engines that will lose a root before any backend starts.

        One watcher per directory per process is a watchdog/FSEvents limit. The
        config is deliberately NOT rejected for it: watching several engines
        over one corpus is a legitimate thing to want (A/B-ing two embedding
        models over the same vault), and the day it is supported no one should
        have to edit a config that was correct all along. So the config stays
        valid and the runtime states plainly what it could not honour.
        """
        owner: dict[str, str] = {}
        for watched in self._engines.values():
            for root in watched.path_filter.active_roots():
                key = str(root)
                first = owner.setdefault(key, watched.name)
                if first != watched.name:
                    logger.error(
                        "Engines '{}' and '{}' both watch {}. Only '{}' will "
                        "receive events — one watcher per directory per process "
                        "is a watchdog/FSEvents limit. '{}' will go stale until "
                        "re-ingested; watch one engine and rebuild the other "
                        "when you switch. See docs/README_WATCH.md.",
                        first,
                        watched.name,
                        key,
                        first,
                        watched.name,
                    )

    def start(self) -> None:
        """Start the backends and the worker. Startup reconciliation is
        MANDATORY and runs on the worker, so MCP answers requests immediately."""
        now = self._clock()
        with self._cond:
            for watched in self._engines.values():
                self._state.reconcile_due[watched.name] = now
        self._next_fts = now + self._fts_interval

        self._warn_on_shared_roots()

        for watched in self._engines.values():
            roots = [str(r) for r in watched.path_filter.active_roots()]
            if not roots:
                logger.warning("Engine '{}' has no active roots; not watching", watched.name)
                continue
            try:
                watched.backend.start(roots, self._callback_for(watched.name))
            except Exception as e:  # noqa: BLE001 — one bad root must not kill MCP startup
                # ERROR, not warning: this engine's index now drifts from disk
                # for the whole run. Searches keep answering, which is exactly
                # what makes it dangerous — the results are quietly stale, not
                # obviously broken.
                logger.error(
                    "Engine '{}' is NOT being watched this run: {} Its index "
                    "will drift from disk as files change; searches keep "
                    "answering from increasingly stale rows. Re-ingest it with "
                    "`--rebuild --force` to resync.",
                    watched.name,
                    e,
                )

        self._worker = threading.Thread(target=self._loop, name="dbs-vector-watcher", daemon=True)
        self._worker.start()

    def _callback_for(self, engine: str) -> Callable[[str, str, bool, str | None], None]:
        def _on_event(path: str, kind: str, is_directory: bool, dest: str | None) -> None:
            self.handle_event(engine, path, kind, is_directory, dest)

        return _on_event

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._stopping:
                    return
                timeout = self._next_wake(self._clock())
                if timeout > 0:
                    self._cond.wait(timeout=timeout)
                if self._stopping:
                    return
            self.run_once()

    def _next_wake(self, now: float) -> float:
        """Seconds until the earliest pending due time, capped at the FTS interval."""
        dues = [p.due for p in self._state.pending.values()]
        dues.extend(self._state.reconcile_due.values())
        dues.append(self._next_fts)
        return max(0.0, min(min(dues) - now, self._fts_interval))

    def stop(self) -> None:
        """Stop the backends and the worker. NO drain ceremony — anything
        pending is picked up by the next startup reconciliation."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()

        for watched in self._engines.values():
            try:
                watched.backend.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("Backend stop failed for '{}': {}", watched.name, e)

        if self._worker is not None:
            self._worker.join(timeout=_WORKER_JOIN_TIMEOUT)
            if self._worker.is_alive():
                # The join TIMED OUT — the worker is still running, not gone.
                # Writing the store from this thread now would put two writers
                # on the same table. Skip it: whatever is left dirty is picked
                # up by the next startup reconciliation.
                logger.warning(
                    "Watcher worker still busy at shutdown; skipping index "
                    "maintenance (the next startup reconciliation rebuilds it)"
                )
                return
            self._worker = None

        for name in sorted(self._dirty):
            watched = self._engines[name]
            try:
                watched.store.create_indices()
                watched.store.compact()
            except Exception as e:  # noqa: BLE001
                logger.warning("Shutdown index maintenance failed for '{}': {}", name, e)
        self._dirty.clear()
        logger.info("Watcher stopped")
