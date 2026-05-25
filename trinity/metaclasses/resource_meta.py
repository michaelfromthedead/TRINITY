"""
ResourceMeta - Metaclass for global singleton resources.

Ensures single instance and provides global access pattern.
Resources are global state containers accessed by systems.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar, Optional

from trinity.constants import DEFAULT_RESOURCE_PRIORITY
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta


class ResourceMeta(EngineMeta):
    """
    Metaclass for global singleton resources.

    Created classes will:
    - Be registered in the resource registry
    - Have at most one instance (singleton pattern)
    - Be accessible globally via the registry
    - Support initialization ordering via priorities

    Optional class attributes (set by decorators):
    - _resource_priority: int (initialization order, lower = earlier)
    - _resource_dependencies: tuple[type, ...] (other resources needed first)

    Attached attributes:
    - _resource_id: int (unique identifier)
    - _resource_name: str (qualified name)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _instances: ClassVar[dict[int, Any]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialization_lock: ClassVar[threading.RLock] = threading.RLock()

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> ResourceMeta:
        """Create a new resource type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Resource class
        if name == "Resource":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._resource_id = mcs._next_id
            mcs._next_id += 1
            cls._resource_name = f"{cls.__module__}.{name}"

            # === 2. SET DEFAULTS ===
            if not hasattr(cls, "_resource_priority"):
                cls._resource_priority = DEFAULT_RESOURCE_PRIORITY  # Default middle priority
            if not hasattr(cls, "_resource_dependencies"):
                cls._resource_dependencies = ()
            if not hasattr(cls, "_resource_lazy"):
                cls._resource_lazy = False

            # === 3. RECORD TAG STEPS ===
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "resource_id", "value": cls._resource_id}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "resource_name", "value": cls._resource_name}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "resource_priority", "value": cls._resource_priority}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "resource_lazy", "value": cls._resource_lazy}))

            # === 4. REGISTER ===
            mcs._registry[cls._resource_id] = cls

            # === 5. RECORD REGISTER STEP ===
            cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "resource_registry", "id": cls._resource_id}))

            # === 6. RECORD HOOK STEP ===
            cls._metaclass_steps.append(Step(Op.HOOK, {"event": "on_create", "callback": "singleton_enforce"}))

        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Control instantiation - enforce singleton pattern.

        Returns the existing instance if one exists, otherwise creates a new one.
        """
        with ResourceMeta._initialization_lock:
            if cls._resource_id in ResourceMeta._instances:
                # Return existing instance
                if args or kwargs:
                    raise TypeError(
                        f"{cls.__name__} is a singleton resource. "
                        f"Use {cls.__name__}() without arguments to get the instance, "
                        f"or use get_instance() class method."
                    )
                return ResourceMeta._instances[cls._resource_id]

            # Create the singleton instance
            instance = super().__call__(*args, **kwargs)
            ResourceMeta._instances[cls._resource_id] = instance

            # Call post-init hook if present
            if hasattr(instance, "_on_resource_created"):
                instance._on_resource_created()

            return instance

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, resource_id: int) -> Optional[type]:
        """Get resource class by ID."""
        return mcs._registry.get(resource_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get resource class by qualified name."""
        for res_id, res_cls in mcs._registry.items():
            if res_cls._resource_name == name:
                return res_cls
        return None

    @classmethod
    def all_resources(mcs) -> list[type]:
        """Get all registered resource classes."""
        return list(mcs._registry.values())

    @classmethod
    def get_instance(mcs, resource_cls: type) -> Optional[Any]:
        """
        Get the singleton instance of a resource.

        Args:
            resource_cls: The resource class to get the instance of.

        Returns:
            The instance if it exists, None otherwise.

        Raises:
            TypeError: If resource_cls is not a resource type.
        """
        if not hasattr(resource_cls, "_resource_id"):
            raise TypeError(f"{resource_cls} is not a resource type")
        return mcs._instances.get(resource_cls._resource_id)

    @classmethod
    def has_instance(mcs, resource_cls: type) -> bool:
        """Check if a resource has been instantiated."""
        if not hasattr(resource_cls, "_resource_id"):
            return False
        return resource_cls._resource_id in mcs._instances

    @classmethod
    def initialize_all(mcs) -> None:
        """
        Initialize all resources in dependency/priority order.

        This creates instances of all registered resources that haven't
        been instantiated yet. Skips resources marked as lazy.

        Raises:
            RuntimeError: If circular dependencies detected or initialization fails
        """
        # Get all uninitialized non-lazy resources
        uninitialized = [
            cls
            for cls in mcs._registry.values()
            if cls._resource_id not in mcs._instances
            and not getattr(cls, "_resource_lazy", False)
        ]

        if not uninitialized:
            return

        # Topological sort by dependencies, then by priority
        initialized_ids = set(mcs._instances.keys())
        failed_resources = []

        while uninitialized:
            # Find resources whose dependencies are satisfied
            ready = []
            for cls in uninitialized:
                deps = getattr(cls, "_resource_dependencies", ())
                deps_satisfied = all(
                    hasattr(dep, "_resource_id") and dep._resource_id in initialized_ids
                    for dep in deps
                )
                if deps_satisfied:
                    ready.append(cls)

            if not ready:
                # Circular dependency or unsatisfied dependencies
                remaining = [cls.__name__ for cls in uninitialized]
                raise RuntimeError(
                    f"Cannot initialize resources - circular or unsatisfied dependencies: {remaining}"
                )

            # Sort by priority and initialize
            ready.sort(key=lambda cls: cls._resource_priority)

            for cls in ready:
                try:
                    cls()  # Trigger instantiation
                    initialized_ids.add(cls._resource_id)
                    uninitialized.remove(cls)
                except Exception as e:
                    # Track failed resource and continue
                    failed_resources.append((cls.__name__, str(e)))
                    uninitialized.remove(cls)

        # If any resources failed to initialize, raise detailed error
        if failed_resources:
            error_msg = "Failed to initialize resources:\n" + "\n".join(
                f"  - {name}: {error}" for name, error in failed_resources
            )
            raise RuntimeError(error_msg)

    @classmethod
    def shutdown_all(mcs) -> None:
        """
        Shutdown all resources in reverse initialization order.

        Calls shutdown() method on each resource if it exists.
        Logs errors but continues shutdown process for remaining resources.
        """
        # Get instances in reverse order of creation (by ID)
        sorted_ids = sorted(mcs._instances.keys(), reverse=True)
        shutdown_errors = []

        for resource_id in sorted_ids:
            instance = mcs._instances.get(resource_id)
            if instance is not None:
                # Call shutdown hook if present
                if hasattr(instance, "shutdown"):
                    try:
                        instance.shutdown()
                    except Exception as e:
                        # Track error but continue shutdown
                        resource_name = type(instance).__name__
                        shutdown_errors.append((resource_name, str(e)))
                        import warnings
                        warnings.warn(
                            f"Error shutting down {resource_name}: {e}",
                            RuntimeWarning,
                            stacklevel=2
                        )

        # Clear all instances even if some shutdowns failed
        mcs._instances.clear()

        # Log summary if there were errors
        if shutdown_errors:
            import warnings
            error_summary = "Some resources failed to shutdown cleanly:\n" + "\n".join(
                f"  - {name}: {error}" for name, error in shutdown_errors
            )
            warnings.warn(error_summary, RuntimeWarning, stacklevel=2)

    @classmethod
    def reset_instance(mcs, resource_cls: type) -> None:
        """
        Remove a resource instance, allowing it to be re-created.

        Useful for testing or resource hot-reloading.
        """
        if not hasattr(resource_cls, "_resource_id"):
            raise TypeError(f"{resource_cls} is not a resource type")

        resource_id = resource_cls._resource_id

        with mcs._initialization_lock:
            if resource_id in mcs._instances:
                instance = mcs._instances[resource_id]
                if hasattr(instance, "shutdown"):
                    instance.shutdown()
                del mcs._instances[resource_id]

    @classmethod
    def get_or_create(mcs, resource_cls: type) -> Any:
        """
        Get the singleton instance of a resource, creating it if necessary.

        Useful for lazy resources that are created on first access.
        Validates dependencies are satisfied before creating the resource.

        Args:
            resource_cls: The resource class to get or create

        Returns:
            The resource instance

        Raises:
            TypeError: If resource_cls is not a resource type
            RuntimeError: If resource dependencies are not satisfied
        """
        if not hasattr(resource_cls, "_resource_id"):
            raise TypeError(f"{resource_cls} is not a resource type")

        with mcs._initialization_lock:
            resource_id = resource_cls._resource_id

            # Return existing instance if available
            if resource_id in mcs._instances:
                return mcs._instances[resource_id]

            # Validate dependencies are satisfied before creating
            deps = getattr(resource_cls, "_resource_dependencies", ())
            for dep in deps:
                if not hasattr(dep, "_resource_id"):
                    raise RuntimeError(
                        f"{resource_cls.__name__}: Dependency {dep} is not a valid resource type"
                    )
                if dep._resource_id not in mcs._instances:
                    raise RuntimeError(
                        f"{resource_cls.__name__}: Dependency {dep.__name__} must be initialized first"
                    )

            # Create new instance (may raise exception)
            try:
                instance = resource_cls()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create resource {resource_cls.__name__}: {e}"
                ) from e

            return instance

    @classmethod
    def is_lazy(mcs, resource_cls: type) -> bool:
        """
        Check if a resource is marked as lazy.

        Args:
            resource_cls: The resource class to check

        Returns:
            True if the resource is lazy, False otherwise

        Raises:
            TypeError: If resource_cls is not a resource type
        """
        if not hasattr(resource_cls, "_resource_id"):
            raise TypeError(f"{resource_cls.__name__} is not a resource type")

        return getattr(resource_cls, "_resource_lazy", False)

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the resource registry and all instances. Useful for testing."""
        with mcs._lock:
            mcs.shutdown_all()
            mcs._registry.clear()
            mcs._next_id = 1
        super().clear_registry()
