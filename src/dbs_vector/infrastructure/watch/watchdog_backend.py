"""The ONLY module allowed to import `watchdog` (FSEvents on macOS)."""

import threading
from pathlib import Path
from typing import Any

from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dbs_vector.core.ports import WatchCallback

# watchdog also emits "opened" / "closed" / "closed_no_write"; those carry no
# state change we act on, so they are dropped here rather than downstream.
_KINDS = {"created", "modified", "deleted", "moved"}

# Roots scheduled by any backend in THIS process, and the engine that holds
# each one. watchdog's FSEvents backend registers watches by path
# process-wide, so a second Observer scheduling the same path is refused with
# `RuntimeError: ... it is already scheduled` — raised on the emitter thread,
# AFTER Observer.start() has already returned. Nothing observes that, so the
# second engine goes silently unwatched and its index drifts from disk with no
# error anywhere. This registry moves the collision onto the CALLING thread,
# where WatcherService can see it and say so.
#
# Per-process by design: two separate `dbs-vector mcp` processes watching the
# same directory is fine and must keep working.
_SCHEDULED_ROOTS: dict[str, str] = {}
_REGISTRY_LOCK = threading.Lock()


def _normalized(root: str) -> str:
    return str(Path(root).resolve())


def roots_already_watched(roots: list[str]) -> dict[str, str]:
    """Which of `roots` this process already watches, mapped to their owner."""
    with _REGISTRY_LOCK:
        return {r: _SCHEDULED_ROOTS[n] for r in roots if (n := _normalized(r)) in _SCHEDULED_ROOTS}


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


class _Handler(FileSystemEventHandler):
    def __init__(self, on_event: WatchCallback) -> None:
        self._on_event = on_event

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in _KINDS:
            return
        src = _as_str(event.src_path)
        if src is None:
            return
        try:
            self._on_event(
                src,
                event.event_type,
                bool(event.is_directory),
                _as_str(getattr(event, "dest_path", None)) or None,
            )
        except Exception as e:  # noqa: BLE001 — an observer thread must never die
            logger.warning("Watch callback failed for {}: {}", src, e)


class WatchdogBackend:
    """`IWatchBackend` over watchdog's recursive Observer."""

    def __init__(self, owner: str = "?") -> None:
        self._observer: Any = None
        self._owner = owner
        self._claimed: list[str] = []

    def start(self, roots: list[str], on_event: WatchCallback) -> None:
        # Claim every root up front: a partial claim would leave this backend
        # watching some of its roots and silently missing the rest.
        with _REGISTRY_LOCK:
            taken = {
                r: _SCHEDULED_ROOTS[n] for r in roots if (n := _normalized(r)) in _SCHEDULED_ROOTS
            }
            if taken:
                detail = "; ".join(f"{r} (already watched by '{o}')" for r, o in taken.items())
                raise RuntimeError(
                    f"this process already watches {detail}. One watcher per "
                    f"directory per process is a watchdog/FSEvents limit, not a "
                    f"dbs-vector policy — see docs/README_WATCH.md."
                )
            for root in roots:
                _SCHEDULED_ROOTS[_normalized(root)] = self._owner
            self._claimed = list(roots)

        try:
            observer = Observer()
            observer.daemon = True
            handler = _Handler(on_event)
            for root in roots:
                observer.schedule(handler, root, recursive=True)
                logger.info("Watching {}", root)
            observer.start()
            self._observer = observer
        except Exception:
            self._release()
            raise

    def _release(self) -> None:
        with _REGISTRY_LOCK:
            for root in self._claimed:
                _SCHEDULED_ROOTS.pop(_normalized(root), None)
            self._claimed = []

    def stop(self) -> None:
        if self._observer is None:
            self._release()
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=5.0)
        except Exception as e:  # noqa: BLE001 — shutdown must not raise
            logger.warning("Watch backend shutdown failed: {}", e)
        finally:
            self._observer = None
            self._release()
