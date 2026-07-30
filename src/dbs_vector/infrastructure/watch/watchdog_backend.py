"""The ONLY module allowed to import `watchdog` (FSEvents on macOS)."""

from typing import Any

from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dbs_vector.core.ports import WatchCallback

# watchdog also emits "opened" / "closed" / "closed_no_write"; those carry no
# state change we act on, so they are dropped here rather than downstream.
_KINDS = {"created", "modified", "deleted", "moved"}


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

    def __init__(self) -> None:
        self._observer: Any = None

    def start(self, roots: list[str], on_event: WatchCallback) -> None:
        observer = Observer()
        observer.daemon = True
        handler = _Handler(on_event)
        for root in roots:
            observer.schedule(handler, root, recursive=True)
            logger.info("Watching {}", root)
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=5.0)
        except Exception as e:  # noqa: BLE001 — shutdown must not raise
            logger.warning("Watch backend shutdown failed: {}", e)
        finally:
            self._observer = None
