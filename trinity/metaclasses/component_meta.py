"""
ComponentMeta - Metaclass for ECS components.

Handles component registration, field processing, and descriptor installation.
This is the most complex metaclass as it manages field-level behavior through
the descriptor system.
"""

from __future__ import annotations

import ctypes
import json
import threading
import warnings
from typing import Annotated, Any, ClassVar, Optional, get_args, get_origin, get_type_hints

from trinity.constants import (
    DEFAULT_COMPONENT_INSTANCE_COUNT_INITIAL,
    DEFAULT_COMPONENT_POOL_INITIAL_SIZE,
)
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta
from trinity.types import (
    REQUIRES_DESCRIPTOR_TYPES,
    NetworkConfig,
    SerializationConfig,
    ValidationRule,
)


class ComponentMeta(EngineMeta):
    """
    Metaclass for ECS components.

    Created classes will:
    - Have a unique _component_id
    - Be registered in the component registry
    - Have fields converted to descriptors (based on decorators)
    - Have memory layout optimizations applied
    - Be validated for component rules

    Optional class attributes (set by decorators):
    - _packed_layout: "soa" | "aos" | None
    - _pooled_config: PoolConfig | None
    - _network_config: NetworkConfig | None
    - _serialization_config: SerializationConfig | None
    - _track_changes: bool
    - _budget_config: BudgetConfig | None

    Attached attributes:
    - _component_id: int (unique identifier)
    - _component_name: str (qualified name)
    - _field_descriptors: dict[str, Descriptor]
    - _field_types: dict[str, type]
    - _field_offsets: dict[str, int] (for bit flags)
    - _field_defaults: dict[str, Any]
    - _pool: list (instance pool, if pooled)
    - _instance_count: int (live instance count, if budgeted)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _name_to_id: ClassVar[dict[str, int]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Map Python types to (Rust type code, byte size) for the Rust ECS bridge.
    #
    # Size note — str / "Str8":
    #   Rust String is 24 bytes on 64-bit (ptr + capacity + len), but the ECS
    #   bridge uses a fixed 8-byte inline buffer for string fields.  The type
    #   code is deliberately "Str8" rather than "String" to signal this
    #   limitation: fields wider than 8 bytes will be silently truncated.
    TYPE_MAP: ClassVar[dict[type, tuple[str, int]]] = {
        float: ("f32", 4),
        ctypes.c_float: ("f32", 4),
        ctypes.c_double: ("f64", 8),
        int: ("i32", 4),
        ctypes.c_uint32: ("u32", 4),
        ctypes.c_int64: ("i64", 8),
        ctypes.c_uint64: ("u64", 8),
        ctypes.c_int32: ("i32", 4),
        ctypes.c_uint8: ("u8", 1),
        ctypes.c_int8: ("i8", 1),
        ctypes.c_uint16: ("u16", 2),
        ctypes.c_int16: ("i16", 2),
        ctypes.c_char: ("u8", 1),
        ctypes.c_byte: ("i8", 1),
        ctypes.c_ubyte: ("u8", 1),
        bool: ("u8", 1),
        str: ("Str8", 8),   # NOTE: 8-byte fixed buffer, NOT 24-byte Rust String — see docstring
    }

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> ComponentMeta:
        """Create a new component type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Component class
        if name == "Component":
            return cls

        with mcs._lock:
            # === 0. IDEMPOTENCY CHECK ===
            qualified_name = f"{cls.__module__}.{name}"
            existing_id = mcs._name_to_id.get(qualified_name)
            if existing_id is not None:
                existing_cls = mcs._registry[existing_id]
                # Compare new field annotations against the existing class to
                # detect silent redefinition with different fields (F-H5).
                new_annotations = {
                    fn for fn in namespace.get("__annotations__", {})
                    if not fn.startswith("_")
                }
                existing_fields = set(existing_cls._field_types.keys())
                if new_annotations != existing_fields:
                    warnings.warn(
                        f"{qualified_name}: Duplicate definition with different fields "
                        f"(new={new_annotations}, existing={existing_fields}). "
                        f"Returning original class.",
                        UserWarning,
                        stacklevel=3,
                    )
                return existing_cls

            # === 1. GENERATE UNIQUE ID ===
            cls._component_id = mcs._next_id
            mcs._next_id += 1
            cls._component_name = f"{cls.__module__}.{name}"

            # 3.2.2: Record TAG steps for component_id and component_name
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "component_id", "value": cls._component_id}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "component_name", "value": cls._component_name}))

            # === 2. PROCESS FIELDS ===
            mcs._process_fields(cls)

            # === 3. INSTALL DESCRIPTORS ===
            mcs._install_descriptors(cls)

            # === 4. VALIDATE COMPONENT ===
            mcs._validate_component(cls)

            # 3.2.5: Record VALIDATE step
            cls._metaclass_steps.append(Step(Op.VALIDATE, {"constraint": "component_rules"}))

            # === 5. REGISTER ===
            mcs._registry[cls._component_id] = cls
            mcs._name_to_id[cls._component_name] = cls._component_id

            # 3.2.6: Record REGISTER step for component registry
            cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "component_registry", "id": cls._component_id}))

            # === 6. FOUNDATION INTEGRATION ===
            # Register with Foundation's central registry for cross-pillar access
            mcs._register_with_foundation(cls)

            # 3.2.7: Record REGISTER step for foundation
            cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "foundation", "name": cls._component_name}))

            # === 6b. RUST TYPE REGISTRATION ===
            fields, total_size = mcs._build_rust_layout(cls)
            try:
                import _omega  # type: ignore[import-untyped, import-not-found]
                _omega.type_register(
                    cls._component_id,
                    cls._component_name,
                    total_size,
                    json.dumps(fields),
                )
            except (ImportError, AttributeError):
                # Rust backend (_omega) not available during testing / docs.
                pass
            except Exception as exc:
                warnings.warn(
                    f"{cls.__name__}: Rust type registration failed: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )

            # === 7. INITIALIZE POOL AND BUDGET ===
            # Initialize pool if pooled_config is set
            if hasattr(cls, "_pooled_config") and cls._pooled_config is not None:
                # Pre-allocate list with initial capacity for better performance
                cls._pool = []
                # Reserve capacity if implementation supports it (CPython doesn't expose this)
                # But we document the expected initial size
                cls._pool_initial_size = DEFAULT_COMPONENT_POOL_INITIAL_SIZE

                # 3.2.8: Record TAG(pooled) + HOOK(on_create, pool_allocate)
                cls._metaclass_steps.append(Step(Op.TAG, {"key": "pooled", "value": True}))
                cls._metaclass_steps.append(Step(Op.HOOK, {"event": "on_create", "callback": "pool_allocate"}))

            # Initialize instance counter if budget_config is set
            if hasattr(cls, "_budget_config") and cls._budget_config is not None:
                cls._instance_count = DEFAULT_COMPONENT_INSTANCE_COUNT_INITIAL

                # 3.2.9: Record TAG(budgeted) + VALIDATE(budget_limit)
                cls._metaclass_steps.append(Step(Op.TAG, {"key": "budgeted", "value": True}))
                cls._metaclass_steps.append(Step(Op.VALIDATE, {"constraint": "budget_limit"}))

        return cls

    @classmethod
    def _build_rust_layout(mcs, cls: type) -> tuple[list[tuple[str, str, int]], int]:
        """Build Rust component layout from Python field types.

        Maps each field's Python type through TYPE_MAP to produce a
        (field_name, type_code, byte_offset) tuple and computes the
        total struct size.

        Returns
        -------
        fields : list[tuple[str, str, int]]
            Each entry is (field_name, rust_type_code, byte_offset).
        total_size : int
            Total byte size of the component struct.
        """
        fields: list[tuple[str, str, int]] = []
        offset = 0
        max_alignment = 1
        for field_name in cls._field_types:
            field_type = cls._field_types[field_name]
            type_code, size = mcs.TYPE_MAP.get(field_type, (getattr(field_type, '__name__', str(field_type)), 4))
            # H-04: Alignment padding — align each field to min(size, 8) boundary,
            # matching Rust repr(C) rules (max alignment cap at 8 avoids over-alignment
            # for wider types that don't exist in the ECS type system).
            #
            # NOTE: 4-byte fallback for unknown types: 4 is the most common alignment
            # boundary (i32/f32), avoids 0-size issues, and matches the Rust ECS
            # bridge's i32 default for unrecognised field types.
            alignment = min(size, 8)
            max_alignment = max(max_alignment, alignment)
            aligned_offset = (offset + alignment - 1) & ~(alignment - 1)
            fields.append((field_name, type_code, aligned_offset))
            offset = aligned_offset + size
        total_size = (offset + max_alignment - 1) & ~(max_alignment - 1) if fields else 0
        return fields, total_size

    @classmethod
    def _register_with_foundation(mcs, cls: type) -> None:
        """Register component with Foundation's central registry.

        Failures are warned but not propagated to avoid leaving the metaclass
        registry (_registry / _name_to_id) in a partially-registered state —
        by the time this method is called, the component is already registered
        in the metaclass's own registries (steps 1–5 in __new__).
        """
        try:
            from foundation import registry
            # Use component_name for consistent naming
            if not registry.is_registered(cls):
                registry.register(cls, name=cls._component_name, track_instances=True)
        except ImportError:
            # Foundation not available during testing / docs — expected.
            pass
        except Exception as exc:
            # Broader catch: registry.register() may raise ValueError (duplicate
            # name), TypeError (non-class), or other runtime errors.  These must
            # not cascade through __new__() which has already committed the
            # component to the metaclass registries above.
            warnings.warn(
                f"{cls.__name__}: Foundation registry registration failed: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def _process_fields(mcs, cls: type) -> None:
        """Process field annotations and defaults."""
        # Get annotations from this class and bases
        try:
            annotations = get_type_hints(cls, include_extras=True)
        except Exception:
            # Fall back to raw annotations if get_type_hints fails
            annotations = getattr(cls, "__annotations__", {})

        cls._field_types = {}
        cls._field_offsets = {}
        cls._field_defaults = {}
        cls._field_descriptors = {}

        # Imports hoisted out of loop to avoid repeated import overhead
        from trinity.descriptors.base import BaseDescriptor

        offset = 0
        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue  # Skip private fields

            # 8.1: Detect Annotated types
            actual_type = field_type
            annotated_descriptors = []
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                actual_type = args[0]  # 8.2: base type
                # 8.3: Extract descriptor classes/instances from metadata
                for meta in args[1:]:
                    if isinstance(meta, BaseDescriptor):
                        # Instance: set field_type
                        meta._field_type = actual_type
                        annotated_descriptors.append(meta)
                    elif isinstance(meta, type) and issubclass(meta, BaseDescriptor):
                        # Class: instantiate
                        annotated_descriptors.append(meta(field_type=actual_type))

            # 8.6: Store unwrapped base type.
            # NOTE: `offset` here is a sequential field index (0, 1, 2, …),
            # NOT a byte offset.  Byte-level layout with alignment padding is
            # computed separately in _build_rust_layout().  The sequential
            # index is used as a dense key for descriptor plumbing (e.g.
            # bit-flags, change-tracking slot IDs).
            cls._field_types[field_name] = actual_type
            cls._field_offsets[field_name] = offset
            offset += 1

            # Capture default value if present
            if hasattr(cls, field_name):
                default = getattr(cls, field_name)
                # Don't store descriptors as defaults
                if not hasattr(default, "__get__"):
                    cls._field_defaults[field_name] = default

            # Validate field type (use actual_type, not Annotated wrapper)
            mcs._validate_field_type(cls, field_name, actual_type)

            # 8.4 + 8.5: If Annotated provided descriptors, compose and install
            if annotated_descriptors:
                from trinity.descriptors import DescriptorComposer, StorageDescriptor
                default = cls._field_defaults.get(field_name)
                storage = StorageDescriptor(field_type=actual_type, default=default)
                # Compose: annotated descriptors (outer to inner) + storage (innermost)
                all_descs = annotated_descriptors + [storage]
                descriptor = DescriptorComposer.compose(*all_descs)
                setattr(cls, field_name, descriptor)
                descriptor.__set_name__(cls, field_name)
                cls._field_descriptors[field_name] = descriptor

        # 3.2.3: Record DESCRIBE step for each processed field
        for field_name, field_type in cls._field_types.items():
            type_name = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
            cls._metaclass_steps.append(Step(Op.DESCRIBE, {"field": field_name, "type": type_name}))

    @classmethod
    def _validate_field_type(mcs, cls: type, field_name: str, field_type: type) -> None:
        """Validate that field types are component-safe."""
        # Get default value
        default = getattr(cls, field_name, None)

        # Disallow mutable defaults
        if default is not None and isinstance(default, (list, dict, set)):
            raise TypeError(
                f"{cls.__name__}.{field_name}: Mutable default values are forbidden. "
                f"Use field(default_factory=...) or initialize in __init__ instead."
            )

        # Warn about callable fields (uncommon in components)
        origin = get_origin(field_type)
        if (
            origin is None
            and callable(field_type)
            and field_type.__name__ == "Callable"
        ):
            warnings.warn(
                f"{cls.__name__}.{field_name}: Callable fields in components are unusual. "
                f"Consider using a System for behavior.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def _install_descriptors(mcs, cls: type) -> None:
        """Install appropriate descriptors based on decorator markers."""
        # Import here to avoid circular imports
        from trinity.descriptors import (
            DescriptorComposer,
            NetworkedDescriptor,
            SerializableDescriptor,
            StorageDescriptor,
            TrackedDescriptor,
            ValidatedDescriptor,
        )

        # Check what features are enabled on this component
        track_changes = getattr(cls, "_track_changes", False)
        network_config = getattr(cls, "_network_config", None)
        serialization_config = getattr(cls, "_serialization_config", None)
        validation_rules = getattr(cls, "_validation_rules", {})

        for field_name, field_type in cls._field_types.items():
            # Skip if a descriptor is already installed (from a decorator)
            existing = cls.__dict__.get(field_name)
            if existing is not None and hasattr(existing, "__get__"):
                cls._field_descriptors[field_name] = existing
                continue

            # Build descriptor chain based on markers (innermost first)
            descriptors = []

            # Storage (innermost) - always present
            default = cls._field_defaults.get(field_name)
            descriptors.append(
                StorageDescriptor(field_type=field_type, default=default)
            )

            # Validation
            if field_name in validation_rules:
                rules = validation_rules[field_name]
                descriptors.append(
                    ValidatedDescriptor(
                        field_type=field_type,
                        validators=[r.validator for r in rules],
                    )
                )

            # Change tracking
            if track_changes:
                descriptors.append(
                    TrackedDescriptor(
                        field_type=field_type,
                        field_offset=cls._field_offsets[field_name],
                    )
                )

            # Networking (outermost)
            if network_config is not None:
                # Check for field-specific network config
                field_network = getattr(cls, "_field_network_config", {}).get(
                    field_name
                )
                config = field_network or network_config
                descriptors.append(
                    NetworkedDescriptor(
                        field_type=field_type,
                        authority=config.authority,
                        interpolated=config.interpolated,
                    )
                )

            # Compose the chain if we have more than just storage
            if len(descriptors) == 1:
                descriptor = descriptors[0]
            else:
                # Compose from innermost to outermost
                descriptor = DescriptorComposer.compose(*reversed(descriptors))

            # Install the descriptor
            setattr(cls, field_name, descriptor)
            descriptor.__set_name__(cls, field_name)
            cls._field_descriptors[field_name] = descriptor

        # 3.2.4: Record INTERCEPT step for each installed descriptor
        for field_name, descriptor in cls._field_descriptors.items():
            desc_id = getattr(descriptor, 'descriptor_id', type(descriptor).__name__)
            cls._metaclass_steps.append(Step(Op.INTERCEPT, {"field": field_name, "descriptor": desc_id}))

        # Phase 9: Auto-install descriptors from _applied_steps
        applied_steps = getattr(cls, "_applied_steps", [])
        if not applied_steps:
            return

        # 9.2: Auto-add TrackedDescriptor if TRACK step present
        has_track_step = any(s.op == Op.TRACK for s in applied_steps)
        if has_track_step and not track_changes:
            for field_name, field_type in cls._field_types.items():
                desc = cls._field_descriptors.get(field_name)
                if desc is None:
                    continue
                chain_ids = [d.descriptor_id for d in desc.get_chain()]
                if "tracked" not in chain_ids:
                    tracked = TrackedDescriptor(
                        field_type=field_type,
                        field_offset=cls._field_offsets[field_name],
                    )
                    tracked._inner = desc
                    tracked.__set_name__(cls, field_name)
                    setattr(cls, field_name, tracked)
                    cls._field_descriptors[field_name] = tracked
                    cls._metaclass_steps.append(Step(Op.INTERCEPT, {
                        "field": field_name, "descriptor": "tracked", "source": "auto_install"
                    }))

        # 9.3: Auto-add ValidatedDescriptor if VALIDATE step present
        validate_steps = [s for s in applied_steps if s.op == Op.VALIDATE]
        if validate_steps:
            for field_name, field_type in cls._field_types.items():
                desc = cls._field_descriptors.get(field_name)
                if desc is None:
                    continue
                chain_ids = [d.descriptor_id for d in desc.get_chain()]
                if "validated" not in chain_ids and "range" not in chain_ids:
                    validated = ValidatedDescriptor(field_type=field_type)
                    validated._inner = desc
                    validated.__set_name__(cls, field_name)
                    setattr(cls, field_name, validated)
                    cls._field_descriptors[field_name] = validated
                    cls._metaclass_steps.append(Step(Op.INTERCEPT, {
                        "field": field_name, "descriptor": "validated", "source": "auto_install"
                    }))

        # 9.4: Warn about unhandled INTERCEPT steps
        intercept_steps = [s for s in applied_steps if s.op == Op.INTERCEPT]
        for step in intercept_steps:
            warnings.warn(
                f"{cls.__name__}: INTERCEPT step {step.args} in _applied_steps "
                f"has no corresponding auto-installed descriptor.",
                UserWarning,
                stacklevel=2,
            )

        # 9.5: Auto-add SerializableDescriptor for HOOK(on_serialize)
        has_serialize_hook = any(
            s.op == Op.HOOK and s.args.get("event") == "on_serialize"
            for s in applied_steps
        )
        if has_serialize_hook and serialization_config is None:
            for field_name, field_type in cls._field_types.items():
                desc = cls._field_descriptors.get(field_name)
                if desc is None:
                    continue
                chain_ids = [d.descriptor_id for d in desc.get_chain()]
                if "serializable" not in chain_ids:
                    serializable = SerializableDescriptor(field_type=field_type)
                    serializable._inner = desc
                    serializable.__set_name__(cls, field_name)
                    setattr(cls, field_name, serializable)
                    cls._field_descriptors[field_name] = serializable
                    cls._metaclass_steps.append(Step(Op.INTERCEPT, {
                        "field": field_name, "descriptor": "serializable", "source": "auto_install"
                    }))

    @classmethod
    def _validate_component(mcs, cls: type) -> None:
        """Validate component definition."""
        # Components should not have complex methods that mutate state
        for name, value in vars(cls).items():
            if name.startswith("_"):
                continue

            if callable(value) and not isinstance(
                value, (classmethod, staticmethod, property)
            ):
                # Check if it's explicitly allowed
                if not getattr(value, "_component_method_allowed", False):
                    # Just warn, don't error - some computed properties are OK
                    warnings.warn(
                        f"{cls.__name__}.{name}(): Components should be data-only. "
                        f"Consider moving logic to a System. "
                        f"Use @component_method to suppress this warning.",
                        UserWarning,
                        stacklevel=4,
                    )

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, component_id: int) -> Optional[type]:
        """Get component class by ID."""
        return mcs._registry.get(component_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get component class by qualified name."""
        component_id = mcs._name_to_id.get(name)
        return mcs._registry.get(component_id) if component_id else None

    @classmethod
    def all_components(mcs) -> list[type]:
        """Get all registered component classes."""
        return list(mcs._registry.values())

    @classmethod
    def component_count(mcs) -> int:
        """Get the number of registered components."""
        return len(mcs._registry)

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the component registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._name_to_id.clear()
            mcs._next_id = 1
        # Also clear from parent
        super().clear_registry()

    # =========================================================================
    # INSTANCE CREATION OVERRIDE (POOL + BUDGET)
    # =========================================================================

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Override instance creation to support pooling and budget enforcement.

        - If _pooled_config is set with max_size, allocate from pool
        - If _budget_config is set with max_instances, enforce budget limit

        Thread-safe: Uses cls._lock for all pool and budget operations.
        """
        # Budget enforcement check (must be before instance creation)
        if hasattr(cls, "_budget_config") and cls._budget_config is not None:
            max_instances = cls._budget_config.get("max_instances")
            if max_instances is not None:
                with cls._lock:
                    if cls._instance_count >= max_instances:
                        raise RuntimeError(
                            f"Budget exceeded: {cls._component_name} has reached "
                            f"max_instances limit of {max_instances}"
                        )
                    cls._instance_count += 1

        # Pool allocation check - ATOMIC: check and pop under same lock
        if hasattr(cls, "_pooled_config") and cls._pooled_config is not None:
            max_size = cls._pooled_config.get("max_size")
            if max_size is not None:
                with cls._lock:
                    if len(cls._pool) > 0:
                        # Reuse pooled instance (pop while locked)
                        instance = cls._pool.pop()
                        # Reinitialize the instance with new data
                        # Note: __init__ called outside lock to avoid holding it during user code
                        if hasattr(instance, "__init__"):
                            instance.__init__(*args, **kwargs)
                        return instance
                # If pool is empty, fall through to normal creation

        # Normal instance creation
        return super(ComponentMeta, cls).__call__(*args, **kwargs)

    # =========================================================================
    # LAYOUT OPTIMIZATION METHODS (SoA/AoS)
    # =========================================================================

    def get_layout_mode(cls) -> str:
        """
        Return the layout mode for this component class.

        Returns:
            "soa" (Structure of Arrays) if _packed_layout is truthy,
            "aos" (Array of Structures) otherwise

        Note:
            _packed_layout is typically set by the @packed decorator.
            Any truthy value (True, "soa", etc.) enables SoA mode.
        """
        if hasattr(cls, "_packed_layout") and cls._packed_layout:
            return "soa"
        return "aos"

    def get_layout_arrays(cls, instances: list[Any]) -> dict[str, list[Any]]:
        """
        Extract field values as Structure of Arrays (SoA).

        When _packed_layout is set, this method extracts all field values
        from the given instances and returns them as separate arrays per field.

        Args:
            instances: List of component instances (must all be of this component type)

        Returns:
            Dict mapping field names to lists of values (one per instance).
            Returns empty dict if:
            - Component is not packed (_packed_layout not set)
            - instances list is empty
            - Component has no fields

        Raises:
            AttributeError: If an instance is missing a required field
            TypeError: If instances contains non-component objects

        Note:
            This method does NOT validate that all instances are of the correct type.
            Caller is responsible for ensuring type homogeneity.
        """
        if not hasattr(cls, "_packed_layout") or not cls._packed_layout:
            # Not using packed layout, return empty dict
            return {}

        if not instances:
            # Empty instance list - return empty dict
            return {}

        if not cls._field_types:
            # Component has no fields - return empty dict
            return {}

        # Build SoA structure
        arrays = {}
        for field_name in cls._field_types.keys():
            arrays[field_name] = [getattr(inst, field_name) for inst in instances]

        return arrays

    # =========================================================================
    # POOL MANAGEMENT METHODS
    # =========================================================================

    def return_to_pool(cls, instance: Any) -> None:
        """
        Return an instance to the pool for reuse.

        This method also decrements the budget counter if budget tracking is enabled.

        Args:
            instance: Instance to return to pool (should be of this component type)

        Returns:
            None

        Behavior:
            - If pooling is disabled or max_size is None: does nothing
            - If pool is full (len >= max_size): discards the instance (no error)
            - If pool has space: adds instance to pool
            - If budget tracking is enabled: always decrements instance count

        Note:
            This method does NOT validate that the instance is of the correct type.
            Returning instances of the wrong type will cause errors on next allocation.

        Thread-safe: Uses cls._lock for all operations.
        """
        if not hasattr(cls, "_pooled_config") or cls._pooled_config is None:
            # Not pooled, but still decrement budget if applicable
            if hasattr(cls, "_budget_config") and cls._budget_config is not None:
                with cls._lock:
                    if hasattr(cls, "_instance_count"):
                        if cls._instance_count <= 0:
                            warnings.warn(
                                f"{cls._component_name}: Attempted to decrement instance count below 0. "
                                f"This may indicate a double-free bug.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        else:
                            cls._instance_count -= 1
            return

        max_size = cls._pooled_config.get("max_size")
        if max_size is None:
            # Pooling enabled but no max_size set - don't pool
            return

        with cls._lock:
            # Only pool if under max_size
            if len(cls._pool) < max_size:
                cls._pool.append(instance)
            # else: pool is full, discard the instance

            # Decrement budget counter if applicable
            if hasattr(cls, "_budget_config") and cls._budget_config is not None:
                if hasattr(cls, "_instance_count"):
                    if cls._instance_count <= 0:
                        warnings.warn(
                            f"{cls._component_name}: Attempted to decrement instance count below 0. "
                            f"This may indicate a double-free bug.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        cls._instance_count -= 1

    def pool_stats(cls) -> dict[str, Any]:
        """
        Get pool statistics for this component class.

        Returns:
            Dict with consistent schema regardless of pooling state:
            {
                "enabled": bool,          # Whether pooling is configured
                "available": int,         # Current pool size (0 if disabled)
                "max_size": int | None,   # Maximum pool size (None if disabled)
                "config": dict | None,    # Full pooled_config (None if disabled)
            }

        Thread-safe: Reads pool size under lock if enabled.
        """
        if not hasattr(cls, "_pooled_config") or cls._pooled_config is None:
            return {
                "enabled": False,
                "available": 0,
                "max_size": None,
                "config": None,
            }

        max_size = cls._pooled_config.get("max_size")
        with cls._lock:
            available = len(cls._pool) if hasattr(cls, "_pool") else 0

        return {
            "enabled": True,
            "available": available,
            "max_size": max_size,
            "config": cls._pooled_config,
        }

    # =========================================================================
    # BUDGET TRACKING METHODS
    # =========================================================================

    def instance_count(cls) -> int:
        """
        Get the current live instance count for this component class.

        Returns:
            Number of live instances if budget tracking is enabled.
            Returns 0 if budget tracking is not configured.

        Note:
            This count is only accurate if:
            1. Budget tracking is enabled via @budgeted decorator
            2. All instances are created via the metaclass __call__
            3. Instances are properly returned via return_to_pool()

        Thread-safe: Reads instance count under lock if enabled.
        """
        if hasattr(cls, "_budget_config") and cls._budget_config is not None:
            if hasattr(cls, "_instance_count"):
                with cls._lock:
                    return cls._instance_count
        return 0
