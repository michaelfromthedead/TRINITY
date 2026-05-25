"""Application lifecycle management."""
from enum import Enum, auto
from typing import Callable, List
import logging
import threading

logger = logging.getLogger(__name__)


class AppState(Enum):
    """Application state."""
    RUNNING = auto()
    PAUSED = auto()
    BACKGROUND = auto()
    SUSPENDED = auto()
    SHUTTING_DOWN = auto()


class AppLifecycle:
    """Application lifecycle manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize lifecycle manager."""
        if self._initialized:
            return

        self._state = AppState.RUNNING
        self._callbacks: List[Callable[[AppState], None]] = []
        self._state_lock = threading.Lock()
        self._initialized = True

    @property
    def current_state(self) -> AppState:
        """Get current application state."""
        with self._state_lock:
            return self._state

    def pause(self) -> None:
        """Pause application."""
        self._transition_to(AppState.PAUSED)

    def resume(self) -> None:
        """Resume application."""
        self._transition_to(AppState.RUNNING)

    def suspend(self) -> None:
        """Suspend application."""
        self._transition_to(AppState.SUSPENDED)

    def shutdown(self) -> None:
        """Shutdown application."""
        self._transition_to(AppState.SHUTTING_DOWN)

    def on_state_change(self, callback: Callable[[AppState], None]) -> None:
        """
        Register state change callback.

        Args:
            callback: Function to call when state changes
        """
        with self._state_lock:
            self._callbacks.append(callback)

    def _transition_to(self, new_state: AppState) -> None:
        """Transition to new state and notify callbacks."""
        callbacks = []
        with self._state_lock:
            if self._state != new_state:
                self._state = new_state
                # Call callbacks outside lock to avoid deadlock
                callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.exception("Callback error in lifecycle state transition")
