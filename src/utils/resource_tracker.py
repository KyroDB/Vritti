from __future__ import annotations

import atexit
import logging
import threading

logger = logging.getLogger(__name__)
_SEMAPHORE_NAMES: set[str] = set()
_LOCK = threading.Lock()
_ORIGINAL_REGISTER = None
_ORIGINAL_UNREGISTER = None


def cleanup_tracked_semaphores() -> None:
    try:
        import _multiprocessing
    except Exception:
        return

    with _LOCK:
        names = list(_SEMAPHORE_NAMES)
        _SEMAPHORE_NAMES.clear()

    for name in names:
        try:
            _multiprocessing.sem_unlink(name)
        except Exception:
            pass
        try:
            if _ORIGINAL_UNREGISTER is not None:
                _ORIGINAL_UNREGISTER(name, "semaphore")
        except Exception:
            pass


def install_resource_tracker_cleanup() -> None:
    """
    Patch multiprocessing.resource_tracker to clean leaked semaphores on shutdown.

    This targets a known issue where some third-party libraries (e.g., PyTorch)
    may register semaphores without unregistering them, producing warnings at
    process exit. We track semaphore registrations and ensure they are cleaned
    up before the resource tracker finalizes.
    """
    try:
        import multiprocessing.resource_tracker as resource_tracker
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Resource tracker patch skipped: {e}")
        return

    with _LOCK:
        if getattr(resource_tracker, "_vritti_semaphore_patch", False):
            return

        global _ORIGINAL_REGISTER, _ORIGINAL_UNREGISTER
        _ORIGINAL_REGISTER = resource_tracker.register
        _ORIGINAL_UNREGISTER = resource_tracker.unregister

        def _register(name: str, rtype: str) -> None:
            if rtype == "semaphore":
                with _LOCK:
                    _SEMAPHORE_NAMES.add(name)
            _ORIGINAL_REGISTER(name, rtype)

        def _unregister(name: str, rtype: str) -> None:
            if rtype == "semaphore":
                with _LOCK:
                    _SEMAPHORE_NAMES.discard(name)
            _ORIGINAL_UNREGISTER(name, rtype)

        resource_tracker.register = _register  # type: ignore[assignment]
        resource_tracker.unregister = _unregister  # type: ignore[assignment]
        resource_tracker._vritti_semaphore_patch = True
        atexit.register(cleanup_tracked_semaphores)
