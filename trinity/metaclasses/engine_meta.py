"""
EngineMeta - Base metaclass for all engine types.

Provides common infrastructure:
- Debug/introspection support
- Common validation infrastructure
- Global type registry for debugging
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from trinity.decorators.ops import Op, Step


class EngineMeta(type):
    """
    Base metaclass for all engine types.

    Provides:
    - Common __repr__ for engine types
    - Debug introspection hooks
    - Validation framework
    - Global registry of all engine types (for debugging)

    All other engine metaclasses inherit from this to avoid metaclass conflicts.
    """

    # Global registry of all engine types (for debugging/introspection)
    _all_engine_types: ClassVar[dict[str, type]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Base class names to skip in registration
    _BASE_CLASS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "EngineBase",
            "Component",
            "System",
            "Resource",
            "Event",
            "Asset",
            "Protocol",
            "State",
        }
    )

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> EngineMeta:
        """Create a new engine type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Initialize metaclass steps recording
        cls._metaclass_steps = []

        # Register for introspection (skip base classes)
        if name not in mcs._BASE_CLASS_NAMES:
            qualified_name = f"{cls.__module__}.{name}"
            with mcs._lock:
                mcs._all_engine_types[qualified_name] = cls
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "engine_types", "name": qualified_name})
            )

        return cls

    def __repr__(cls) -> str:
        """Clean repr for engine types."""
        meta_name = type(cls).__name__
        # Remove 'Meta' suffix for cleaner display
        if meta_name.endswith("Meta"):
            kind = meta_name[:-4]
        else:
            kind = meta_name
        return f"<{kind} '{cls.__name__}'>"

    @classmethod
    def get_all_types(mcs) -> dict[str, type]:
        """
        Debug: Get all registered engine types.

        Returns:
            Dict mapping qualified name to type.
        """
        with mcs._lock:
            return dict(mcs._all_engine_types)

    @classmethod
    def get_types_by_metaclass(mcs, metaclass: type) -> dict[str, type]:
        """
        Debug: Get all types using a specific metaclass.

        Args:
            metaclass: The metaclass to filter by.

        Returns:
            Dict mapping qualified name to type.
        """
        with mcs._lock:
            return {
                name: cls
                for name, cls in mcs._all_engine_types.items()
                if isinstance(cls, metaclass)
            }

    @classmethod
    def clear_registry(mcs) -> None:
        """
        Clear the type registry. Useful for testing.
        """
        with mcs._lock:
            mcs._all_engine_types.clear()
