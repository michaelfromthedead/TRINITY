"""
Hot-Reload System - Core hot-reload functionality with @reloadable decorator.

Provides the main HotReloader class and @reloadable decorator for marking
classes that can be safely hot-reloaded at runtime.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from foundation import schema_hash, to_dict, from_dict, mirror


T = TypeVar("T")

# Module-level registry of reloadable classes
_reloadable_registry: Dict[str, Type] = {}
_instance_registry: Dict[str, List[weakref.ref]] = {}


class ReloadError(Exception):
    """Base exception for hot-reload errors."""
    pass


class SchemaBreakingChangeError(ReloadError):
    """Raised when a schema change would break existing instances."""

    def __init__(
        self,
        class_name: str,
        old_hash: str,
        new_hash: str,
        breaking_changes: List[str],
    ):
        self.class_name = class_name
        self.old_hash = old_hash
        self.new_hash = new_hash
        self.breaking_changes = breaking_changes
        super().__init__(
            f"Breaking schema change in {class_name}: {', '.join(breaking_changes)}"
        )


@dataclass
class ReloadableClass:
    """Metadata for a reloadable class."""

    cls: Type
    module_name: str
    class_name: str
    schema_hash: str
    preserve_state: bool = True
    allow_schema_changes: bool = False
    migration_fn: Optional[Callable[[Dict[str, Any], str, str], Dict[str, Any]]] = None

    @property
    def full_name(self) -> str:
        return f"{self.module_name}.{self.class_name}"


def reloadable(
    preserve_state: bool = True,
    allow_schema_changes: bool = False,
    migration: Optional[Callable[[Dict[str, Any], str, str], Dict[str, Any]]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to mark a class as safe for hot-reloading.

    Args:
        preserve_state: If True, preserve instance state across reloads.
        allow_schema_changes: If True, allow schema changes without migration.
        migration: Optional migration function for schema changes.
            Signature: (state_dict, old_hash, new_hash) -> new_state_dict

    Returns:
        Decorated class with hot-reload metadata.

    Example:
        @reloadable(preserve_state=True)
        class MyComponent:
            def __init__(self):
                self.health = 100
                self.position = (0, 0, 0)
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Store original __init__ for state restoration
        original_init = cls.__init__

        def tracked_init(self: T, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            # Register instance for tracking
            full_name = f"{cls.__module__}.{cls.__name__}"
            if full_name not in _instance_registry:
                _instance_registry[full_name] = []
            _instance_registry[full_name].append(weakref.ref(self))

        cls.__init__ = tracked_init

        # Add reloadable metadata
        cls.__reloadable__ = True
        cls.__preserve_state__ = preserve_state
        cls.__allow_schema_changes__ = allow_schema_changes
        cls.__migration_fn__ = migration
        cls.__schema_hash__ = schema_hash(cls)

        # Register in global registry
        full_name = f"{cls.__module__}.{cls.__name__}"
        _reloadable_registry[full_name] = cls

        return cls

    return decorator


@dataclass
class ReloadResult:
    """Result of a reload operation."""

    success: bool
    module_name: str
    reloaded_classes: List[str] = field(default_factory=list)
    preserved_instances: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_time: float = 0.0


class HotReloader:
    """
    Main hot-reload manager.

    Coordinates module reloading, state preservation, and instance updates.
    Integrates with Foundation's Serializer for state management.
    """

    def __init__(
        self,
        on_reload_start: Optional[Callable[[str], None]] = None,
        on_reload_complete: Optional[Callable[[ReloadResult], None]] = None,
        on_reload_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        Initialize the hot reloader.

        Args:
            on_reload_start: Callback when reload begins.
            on_reload_complete: Callback when reload completes.
            on_reload_error: Callback when reload fails.
        """
        self._on_reload_start = on_reload_start
        self._on_reload_complete = on_reload_complete
        self._on_reload_error = on_reload_error
        self._lock = threading.RLock()
        self._preserved_states: Dict[int, Dict[str, Any]] = {}
        self._reload_count = 0
        self._last_reload_time = 0.0
        self._enabled = True

    @property
    def reload_count(self) -> int:
        """Number of successful reloads."""
        return self._reload_count

    @property
    def last_reload_time(self) -> float:
        """Timestamp of last reload."""
        return self._last_reload_time

    @property
    def enabled(self) -> bool:
        """Whether hot-reload is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def get_reloadable_classes(self) -> Dict[str, ReloadableClass]:
        """Get all registered reloadable classes."""
        result = {}
        for full_name, cls in _reloadable_registry.items():
            result[full_name] = ReloadableClass(
                cls=cls,
                module_name=cls.__module__,
                class_name=cls.__name__,
                schema_hash=getattr(cls, "__schema_hash__", schema_hash(cls)),
                preserve_state=getattr(cls, "__preserve_state__", True),
                allow_schema_changes=getattr(cls, "__allow_schema_changes__", False),
                migration_fn=getattr(cls, "__migration_fn__", None),
            )
        return result

    def get_instances(self, class_name: str) -> List[Any]:
        """Get all live instances of a reloadable class."""
        instances = []
        for ref in _instance_registry.get(class_name, []):
            obj = ref()
            if obj is not None:
                instances.append(obj)
        return instances

    def preserve_state(self, obj: Any) -> Dict[str, Any]:
        """
        Preserve the state of an object using Foundation's Serializer.

        Args:
            obj: The object to preserve.

        Returns:
            Dictionary containing the serialized state.
        """
        try:
            return to_dict(obj, include_schema_hash=True)
        except Exception:
            # Fallback to mirror-based preservation
            m = mirror(obj)
            return {name: m.get(name) for name in m.fields}

    def restore_state(
        self,
        obj: Any,
        state: Dict[str, Any],
        old_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
    ) -> None:
        """
        Restore state to an object.

        Args:
            obj: The object to restore.
            state: The state dictionary.
            old_hash: Previous schema hash.
            new_hash: Current schema hash.
        """
        cls = type(obj)
        migration_fn = getattr(cls, "__migration_fn__", None)

        # Apply migration if schema changed and migration is available
        if old_hash and new_hash and old_hash != new_hash and migration_fn:
            state = migration_fn(state, old_hash, new_hash)

        # Restore fields using mirror
        m = mirror(obj)
        current_fields = set(m.fields.keys())

        for name, value in state.items():
            if name.startswith("__"):
                continue
            if name in current_fields:
                try:
                    m.set(name, value)
                except (AttributeError, TypeError):
                    pass

    def reload_module(self, module_name: str) -> ReloadResult:
        """
        Reload a Python module and update all reloadable instances.

        Args:
            module_name: Name of the module to reload.

        Returns:
            ReloadResult with details of the operation.
        """
        if not self._enabled:
            return ReloadResult(
                success=False,
                module_name=module_name,
                errors=["Hot-reload is disabled"],
            )

        start_time = time.time()
        result = ReloadResult(success=True, module_name=module_name)

        with self._lock:
            try:
                if self._on_reload_start:
                    self._on_reload_start(module_name)

                # Get module from sys.modules
                if module_name not in sys.modules:
                    result.success = False
                    result.errors.append(f"Module {module_name} not loaded")
                    return result

                module = sys.modules[module_name]

                # Collect classes and instances to preserve
                classes_to_reload = {}
                instances_to_preserve: Dict[str, List[tuple]] = {}

                for full_name, cls in _reloadable_registry.items():
                    if cls.__module__ == module_name:
                        old_hash = getattr(cls, "__schema_hash__", "")
                        classes_to_reload[full_name] = {
                            "cls": cls,
                            "old_hash": old_hash,
                        }

                        # Preserve instance states
                        instances = self.get_instances(full_name)
                        if instances:
                            instances_to_preserve[full_name] = [
                                (id(inst), self.preserve_state(inst))
                                for inst in instances
                            ]
                            result.preserved_instances += len(instances)

                # Perform the reload
                importlib.reload(module)

                # Update classes and restore states
                for full_name, info in classes_to_reload.items():
                    old_hash = info["old_hash"]

                    # Get the new class
                    if full_name in _reloadable_registry:
                        new_cls = _reloadable_registry[full_name]
                        new_hash = getattr(new_cls, "__schema_hash__", schema_hash(new_cls))

                        # Check for schema changes
                        if old_hash != new_hash:
                            allow_changes = getattr(new_cls, "__allow_schema_changes__", False)
                            has_migration = getattr(new_cls, "__migration_fn__", None) is not None

                            if not allow_changes and not has_migration:
                                # This would break instances - we'll handle it gracefully
                                result.errors.append(
                                    f"Schema changed for {full_name} without migration"
                                )

                        # Restore instance states
                        if full_name in instances_to_preserve:
                            for inst_id, state in instances_to_preserve[full_name]:
                                # Find instance by id (may not work if instance was recreated)
                                for inst in self.get_instances(full_name):
                                    if id(inst) == inst_id:
                                        self.restore_state(inst, state, old_hash, new_hash)
                                        break

                        result.reloaded_classes.append(full_name)

                self._reload_count += 1
                self._last_reload_time = time.time()

            except Exception as e:
                result.success = False
                result.errors.append(str(e))
                if self._on_reload_error:
                    self._on_reload_error(module_name, e)

            finally:
                result.elapsed_time = time.time() - start_time
                if self._on_reload_complete:
                    self._on_reload_complete(result)

        return result

    def reload_class(self, full_name: str) -> ReloadResult:
        """
        Reload a specific class by reloading its module.

        Args:
            full_name: Full class name (module.ClassName).

        Returns:
            ReloadResult with details.
        """
        if full_name not in _reloadable_registry:
            return ReloadResult(
                success=False,
                module_name="",
                errors=[f"Class {full_name} not in reloadable registry"],
            )

        cls = _reloadable_registry[full_name]
        return self.reload_module(cls.__module__)

    def check_schema_compatibility(
        self,
        old_cls: Type,
        new_cls: Type,
    ) -> List[str]:
        """
        Check if two class versions are schema-compatible.

        Args:
            old_cls: Previous class version.
            new_cls: New class version.

        Returns:
            List of incompatibility descriptions (empty if compatible).
        """
        from engine.tooling.hotreload.schema_hash import SchemaHasher

        hasher = SchemaHasher()
        comparison = hasher.compare_schemas(old_cls, new_cls)

        breaking_changes = []
        for change in comparison.changes:
            if change.is_breaking:
                breaking_changes.append(change.description)

        return breaking_changes

    def clear_registry(self) -> None:
        """Clear all registrations (for testing)."""
        _reloadable_registry.clear()
        _instance_registry.clear()
        self._preserved_states.clear()
        self._reload_count = 0


# Singleton instance
_hot_reloader: Optional[HotReloader] = None


def get_hot_reloader() -> HotReloader:
    """Get the global HotReloader instance."""
    global _hot_reloader
    if _hot_reloader is None:
        _hot_reloader = HotReloader()
    return _hot_reloader


__all__ = [
    "reloadable",
    "ReloadError",
    "SchemaBreakingChangeError",
    "ReloadableClass",
    "ReloadResult",
    "HotReloader",
    "get_hot_reloader",
]
