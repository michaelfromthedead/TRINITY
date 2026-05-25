"""
Reload Callbacks - Pre/post reload hooks and callback management.

Provides a flexible callback system for hot-reload events,
allowing modules to prepare for and respond to reloads.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set


class ReloadPhase(Enum):
    """Phases of the reload process."""

    # Before any reload processing
    PRE_RELOAD = auto()

    # After state has been preserved but before module reload
    STATE_PRESERVED = auto()

    # After module has been reloaded but before state restore
    MODULE_RELOADED = auto()

    # After state has been restored
    STATE_RESTORED = auto()

    # After all reload processing is complete
    POST_RELOAD = auto()

    # When a reload error occurs
    RELOAD_ERROR = auto()

    # When a reload is cancelled/aborted
    RELOAD_CANCELLED = auto()


class CallbackPriority(Enum):
    """Priority levels for callback execution order."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class ReloadContext:
    """Context information passed to reload callbacks."""

    phase: ReloadPhase
    module_name: str
    timestamp: float = field(default_factory=time.time)
    classes_affected: List[str] = field(default_factory=list)
    instances_affected: int = 0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # State that can be modified by callbacks
    abort: bool = False
    abort_reason: Optional[str] = None


# Type alias for callbacks
ReloadCallback = Callable[[ReloadContext], None]


@dataclass
class CallbackRegistration:
    """Registration info for a callback."""

    callback: ReloadCallback
    priority: CallbackPriority
    phases: Set[ReloadPhase]
    module_filter: Optional[str] = None  # Only trigger for specific modules
    once: bool = False  # Unregister after first invocation
    enabled: bool = True

    def matches(self, ctx: ReloadContext) -> bool:
        """Check if this callback should be invoked for the given context."""
        if not self.enabled:
            return False
        if ctx.phase not in self.phases:
            return False
        if self.module_filter and ctx.module_name != self.module_filter:
            return False
        return True


class ReloadCallbacks:
    """
    Manages callbacks for hot-reload events.

    Features:
    - Multiple callback phases
    - Priority-based execution order
    - Module filtering
    - One-shot callbacks
    - Abort capability
    - Thread-safe invocation
    """

    def __init__(self):
        """Initialize the callback manager."""
        self._callbacks: List[CallbackRegistration] = []
        self._lock = threading.RLock()
        self._invocation_count = 0

    @property
    def callback_count(self) -> int:
        """Number of registered callbacks."""
        with self._lock:
            return len(self._callbacks)

    @property
    def invocation_count(self) -> int:
        """Total number of callback invocations."""
        return self._invocation_count

    def register(
        self,
        callback: ReloadCallback,
        phases: Optional[Set[ReloadPhase]] = None,
        priority: CallbackPriority = CallbackPriority.NORMAL,
        module_filter: Optional[str] = None,
        once: bool = False,
    ) -> int:
        """
        Register a callback for reload events.

        Args:
            callback: Function to call on reload events.
            phases: Set of phases to trigger on (default: all phases).
            priority: Execution priority.
            module_filter: Only trigger for specific module.
            once: Unregister after first invocation.

        Returns:
            Registration ID for later unregistration.
        """
        if phases is None:
            phases = set(ReloadPhase)

        registration = CallbackRegistration(
            callback=callback,
            priority=priority,
            phases=phases,
            module_filter=module_filter,
            once=once,
        )

        with self._lock:
            self._callbacks.append(registration)
            # Sort by priority
            self._callbacks.sort(key=lambda r: r.priority.value)
            return id(registration)

    def unregister(self, callback_or_id: Any) -> bool:
        """
        Unregister a callback.

        Args:
            callback_or_id: Callback function or registration ID.

        Returns:
            True if callback was found and removed.
        """
        with self._lock:
            for i, reg in enumerate(self._callbacks):
                if reg.callback is callback_or_id or id(reg) == callback_or_id:
                    del self._callbacks[i]
                    return True
        return False

    def enable(self, callback_or_id: Any) -> bool:
        """Enable a previously disabled callback."""
        return self._set_enabled(callback_or_id, True)

    def disable(self, callback_or_id: Any) -> bool:
        """Disable a callback without removing it."""
        return self._set_enabled(callback_or_id, False)

    def _set_enabled(self, callback_or_id: Any, enabled: bool) -> bool:
        """Set enabled state for a callback."""
        with self._lock:
            for reg in self._callbacks:
                if reg.callback is callback_or_id or id(reg) == callback_or_id:
                    reg.enabled = enabled
                    return True
        return False

    def invoke(self, ctx: ReloadContext) -> ReloadContext:
        """
        Invoke all matching callbacks for a context.

        Args:
            ctx: Reload context with current state.

        Returns:
            Updated context (may have abort flag set).
        """
        with self._lock:
            matching = [reg for reg in self._callbacks if reg.matches(ctx)]

        to_remove = []

        for reg in matching:
            if ctx.abort:
                break

            try:
                reg.callback(ctx)
                self._invocation_count += 1

                if reg.once:
                    to_remove.append(reg)

            except Exception as e:
                # Store error but continue with other callbacks
                if ctx.error is None:
                    ctx.error = e

        # Remove one-shot callbacks
        if to_remove:
            with self._lock:
                for reg in to_remove:
                    if reg in self._callbacks:
                        self._callbacks.remove(reg)

        return ctx

    def clear(self) -> int:
        """
        Remove all registered callbacks.

        Returns:
            Number of callbacks removed.
        """
        with self._lock:
            count = len(self._callbacks)
            self._callbacks.clear()
            return count

    # Convenience decorators
    def on_pre_reload(
        self,
        priority: CallbackPriority = CallbackPriority.NORMAL,
        module_filter: Optional[str] = None,
    ) -> Callable[[ReloadCallback], ReloadCallback]:
        """Decorator to register a pre-reload callback."""
        def decorator(fn: ReloadCallback) -> ReloadCallback:
            self.register(
                fn,
                phases={ReloadPhase.PRE_RELOAD},
                priority=priority,
                module_filter=module_filter,
            )
            return fn
        return decorator

    def on_post_reload(
        self,
        priority: CallbackPriority = CallbackPriority.NORMAL,
        module_filter: Optional[str] = None,
    ) -> Callable[[ReloadCallback], ReloadCallback]:
        """Decorator to register a post-reload callback."""
        def decorator(fn: ReloadCallback) -> ReloadCallback:
            self.register(
                fn,
                phases={ReloadPhase.POST_RELOAD},
                priority=priority,
                module_filter=module_filter,
            )
            return fn
        return decorator

    def on_reload_error(
        self,
        priority: CallbackPriority = CallbackPriority.NORMAL,
        module_filter: Optional[str] = None,
    ) -> Callable[[ReloadCallback], ReloadCallback]:
        """Decorator to register an error callback."""
        def decorator(fn: ReloadCallback) -> ReloadCallback:
            self.register(
                fn,
                phases={ReloadPhase.RELOAD_ERROR},
                priority=priority,
                module_filter=module_filter,
            )
            return fn
        return decorator


# Global callback manager
_global_callbacks: Optional[ReloadCallbacks] = None


def get_reload_callbacks() -> ReloadCallbacks:
    """Get the global ReloadCallbacks instance."""
    global _global_callbacks
    if _global_callbacks is None:
        _global_callbacks = ReloadCallbacks()
    return _global_callbacks


# Convenience functions using global instance
def on_pre_reload(
    callback: ReloadCallback,
    priority: CallbackPriority = CallbackPriority.NORMAL,
    module_filter: Optional[str] = None,
) -> int:
    """Register a pre-reload callback."""
    return get_reload_callbacks().register(
        callback,
        phases={ReloadPhase.PRE_RELOAD},
        priority=priority,
        module_filter=module_filter,
    )


def on_post_reload(
    callback: ReloadCallback,
    priority: CallbackPriority = CallbackPriority.NORMAL,
    module_filter: Optional[str] = None,
) -> int:
    """Register a post-reload callback."""
    return get_reload_callbacks().register(
        callback,
        phases={ReloadPhase.POST_RELOAD},
        priority=priority,
        module_filter=module_filter,
    )


__all__ = [
    "ReloadPhase",
    "CallbackPriority",
    "ReloadContext",
    "ReloadCallback",
    "CallbackRegistration",
    "ReloadCallbacks",
    "get_reload_callbacks",
    "on_pre_reload",
    "on_post_reload",
]
