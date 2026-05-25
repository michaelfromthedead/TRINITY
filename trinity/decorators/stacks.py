from __future__ import annotations

import functools
import warnings
from typing import Callable

__all__ = ["Stack", "stack", "parameterized_stack", "_noop"]


def _validate_stack_combination(decorators: tuple[Callable, ...]) -> None:
    """Check for known anti-pattern combinations in a stack."""
    names: set[str] = set()
    for d in decorators:
        name = getattr(d, "_decorator_name", None)
        if name is None:
            name = getattr(d, "__name__", None) or getattr(
                d, "__qualname__", ""
            )
        if name:
            names.add(name)

    # Hard errors — contradictory combinations
    if "parallel" in names and "exclusive" in names:
        raise ValueError(
            "Stack contains both @parallel and @exclusive which are contradictory"
        )

    if "tag" in names and "serializable" in names and len(names) == 2:
        raise ValueError(
            "@tag combined with @serializable: tags have no data to serialize"
        )

    # Warnings — likely mistakes
    if "networked" in names and "track_changes" not in names:
        warnings.warn(
            "@networked without @track_changes: delta sync requires change tracking",
            UserWarning,
            stacklevel=3,
        )

    if "reloadable" in names and "build_only" not in names:
        warnings.warn(
            "@reloadable without @build_only: hot reload should be dev-only",
            UserWarning,
            stacklevel=3,
        )


class Stack:
    """A composable group of decorators acting as a single decorator."""

    def __init__(self, *decorators: Callable, name: str | None = None) -> None:
        self._decorators = decorators
        self._name = name

    @property
    def decorators(self) -> tuple[Callable, ...]:
        return self._decorators

    def __call__(self, cls):
        _validate_stack_combination(self._decorators)
        for decorator in reversed(self._decorators):
            result = decorator(cls)
            if result is None:
                raise TypeError(
                    f"Decorator {getattr(decorator, '__name__', repr(decorator))} "
                    f"returned None when decorating {cls!r}"
                )
            cls = result
        return cls

    def __add__(self, other: Stack) -> Stack:
        if not isinstance(other, Stack):
            return NotImplemented
        return Stack(
            *self._decorators,
            *other._decorators,
            name=f"{self._name}+{other._name}" if self._name and other._name else None,
        )

    def expand(self) -> list[str]:
        """Return decorator names for introspection."""
        return [
            getattr(d, "__name__", None) or getattr(d, "__qualname__", repr(d))
            for d in self._decorators
        ]

    def __repr__(self) -> str:
        label = self._name or "Stack"
        return f"{label}({len(self._decorators)} decorators)"

    def __len__(self) -> int:
        return len(self._decorators)


def stack(*decorators: Callable, name: str | None = None) -> Stack:
    """Convenience constructor for Stack."""
    return Stack(*decorators, name=name)


def parameterized_stack(fn: Callable[..., Stack]) -> Callable[..., Stack]:
    """Decorator for creating parameterized stacks.

    Validates that the wrapped function returns a Stack instance.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Stack:
        result = fn(*args, **kwargs)
        if not isinstance(result, Stack):
            raise TypeError(
                f"{fn.__name__} must return a Stack, got {type(result).__name__}"
            )
        return result

    wrapper._is_parameterized_stack = True
    return wrapper


def _noop(cls):
    """No-op decorator for conditional stacks."""
    return cls
