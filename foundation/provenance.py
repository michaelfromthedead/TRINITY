"""
Computed Provenance - Track why computed values have their current values.

Part of Core Foundation Layer 0. Provides lightweight provenance tracking
for @computed fields only, enabling debugging and explainability of
derived values in the simulation.

Key features:
- Records what method computed a value
- Captures the simulation tick when computed
- Tracks input summaries for debugging
- Automatic read tracking from descriptors
- Derivation tree queries for full traceability
- Minimal overhead via context variables
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from foundation.eventlog import get_current_tick

T = TypeVar('T')


@dataclass
class ReadRecord:
    """
    Records a single field read during a computation.

    Attributes:
        obj_id: The id() of the object that was read from.
        obj_type: The type name of the object.
        field: The field name that was read.
        value: The value that was read.
    """
    obj_id: int
    obj_type: str
    field: str
    value: Any


@dataclass
class DerivationNode:
    """
    A node in the derivation tree showing data lineage.

    Represents one step in a computation's data flow, showing
    what field was read and from where, with children representing
    further reads that contributed to this value.

    Attributes:
        field: The field name that was read.
        value: The value at this node.
        source_obj_id: The id() of the source object (if from another object).
        source_obj_type: The type name of the source object.
        children: Child nodes representing nested reads.
    """
    field: str
    value: Any
    source_obj_id: Optional[int] = None
    source_obj_type: Optional[str] = None
    children: list["DerivationNode"] = field(default_factory=list)


@dataclass
class ComputedProvenance:
    """
    Records provenance for a computed value.

    Attributes:
        value: The computed result.
        computed_by: Method name that computed this (e.g., "Player.threat_level").
        tick: Simulation tick when computed.
        input_summary: Snapshot of inputs at computation time.
        reads: List of field reads captured automatically during computation.
    """
    value: Any
    computed_by: str
    tick: int
    input_summary: dict[str, Any] = field(default_factory=dict)
    reads: list[ReadRecord] = field(default_factory=list)


# Registry of provenance data: (obj_id, field_name) -> ComputedProvenance
_provenance_registry: dict[tuple[int, str], ComputedProvenance] = {}

# Context var to track current computation for input capture
_capturing_inputs: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    'capturing_inputs', default=None
)

# Context var to capture field reads during computation
_current_reads: ContextVar[Optional[list[ReadRecord]]] = ContextVar(
    'current_reads', default=None
)


def track_provenance(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to track provenance for computed methods.

    This decorator wraps a method to automatically record:
    - The computed value
    - Which method computed it
    - The simulation tick
    - Any inputs recorded via record_input()
    - Any field reads captured automatically from descriptors

    Usage:
        @computed
        @track_provenance
        def threat_level(self) -> float:
            nearby = world.query(Enemy).near(self, 10)
            record_input("nearby_enemies", [e.id for e in nearby])
            return sum(e.damage for e in nearby)

    Args:
        fn: The method to wrap.

    Returns:
        Wrapped method that records provenance.
    """
    @wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        # Start capturing inputs
        inputs: dict[str, Any] = {}
        input_token = _capturing_inputs.set(inputs)

        # Start capturing reads
        reads: list[ReadRecord] = []
        reads_token = _current_reads.set(reads)

        try:
            result = fn(self, *args, **kwargs)

            # Store provenance
            obj_id = id(self)
            field_name = fn.__name__
            prov = ComputedProvenance(
                value=result,
                computed_by=f"{type(self).__name__}.{field_name}",
                tick=get_current_tick(),
                input_summary=inputs.copy(),
                reads=reads.copy()
            )
            _provenance_registry[(obj_id, field_name)] = prov

            return result
        finally:
            _capturing_inputs.reset(input_token)
            _current_reads.reset(reads_token)

    return wrapper


def record_input(name: str, value: Any) -> None:
    """
    Record an input value for current computation.

    Call this from within @track_provenance decorated methods to
    capture what inputs influenced the computed result.

    Args:
        name: Descriptive name for the input (e.g., "nearby_enemies").
        value: The input value to record.

    Example:
        @track_provenance
        def threat_level(self) -> float:
            enemies = self.get_nearby_enemies()
            record_input("enemy_count", len(enemies))
            record_input("enemy_ids", [e.id for e in enemies])
            return sum(e.damage for e in enemies)
    """
    inputs = _capturing_inputs.get()
    if inputs is not None:
        inputs[name] = value


def record_read(obj: Any, field: str, value: Any) -> None:
    """
    Record a field read during a provenance-tracked computation.

    This is called automatically by Trinity descriptors when fields
    are accessed during a @track_provenance computation. Users typically
    don't need to call this directly.

    Args:
        obj: The object whose field was read.
        field: The field name that was read.
        value: The value that was read.
    """
    reads = _current_reads.get()
    if reads is not None:
        reads.append(ReadRecord(
            obj_id=id(obj),
            obj_type=type(obj).__name__,
            field=field,
            value=value
        ))


def get_current_reads_collector() -> Optional[list[ReadRecord]]:
    """
    Get the current reads collector for provenance tracking.

    This is used by descriptors to check if provenance tracking
    is active and record reads accordingly.

    Returns:
        The current reads list if provenance is being tracked, None otherwise.
    """
    return _current_reads.get()


def provenance(obj: Any, field: str) -> Optional[ComputedProvenance]:
    """
    Query provenance for a computed field.

    Args:
        obj: The object to query.
        field: The field name (e.g., "threat_level").

    Returns:
        ComputedProvenance if tracked, None otherwise.

    Example:
        >>> prov = provenance(player, "threat_level")
        >>> prov
        ComputedProvenance(
            value=75,
            computed_by="Player.threat_level",
            tick=5030,
            input_summary={"nearby_enemies": [3, 7], "total_damage": 75}
        )
    """
    return _provenance_registry.get((id(obj), field))


def clear_provenance() -> None:
    """Clear all stored provenance data."""
    _provenance_registry.clear()


def all_provenance() -> dict[tuple[int, str], ComputedProvenance]:
    """
    Get all stored provenance (for debugging/testing).

    Returns:
        Copy of the provenance registry mapping
        (object_id, field_name) -> ComputedProvenance.
    """
    return _provenance_registry.copy()


def derivation_tree(
    obj: Any,
    field: str,
    _visited: Optional[set[tuple[int, str]]] = None
) -> Optional[DerivationNode]:
    """
    Build a derivation tree showing how a computed value was derived.

    The derivation tree traces the data lineage of a computed field,
    showing all the field reads that contributed to its value.
    Handles deep nesting and prevents infinite recursion via cycle detection.

    Args:
        obj: The object to query.
        field: The field name (e.g., "threat_level").
        _visited: Internal parameter for cycle detection.

    Returns:
        DerivationNode tree if provenance exists, None otherwise.

    Example:
        >>> tree = derivation_tree(player, "threat_level")
        >>> tree
        DerivationNode(
            field='threat_level',
            value=75,
            children=[
                DerivationNode(field='damage', value=50, source_obj_type='Enemy', ...),
                DerivationNode(field='damage', value=25, source_obj_type='Enemy', ...)
            ]
        )
    """
    prov = provenance(obj, field)
    if prov is None:
        return None

    # Initialize visited set for cycle detection
    if _visited is None:
        _visited = set()

    # Check for cycles
    key = (id(obj), field)
    if key in _visited:
        return None  # Already visited, prevent infinite recursion
    _visited.add(key)

    # Build root node
    root = DerivationNode(
        field=field,
        value=prov.value,
        source_obj_id=id(obj),
        source_obj_type=type(obj).__name__
    )

    # Add children from recorded reads (with full recursion)
    for read in prov.reads:
        child = DerivationNode(
            field=read.field,
            value=read.value,
            source_obj_id=read.obj_id,
            source_obj_type=read.obj_type
        )

        # Check if this read has its own provenance (nested computation)
        nested_prov_key = (read.obj_id, read.field)
        if nested_prov_key in _provenance_registry:
            # Recursively build subtree for nested provenance
            nested_prov = _provenance_registry[nested_prov_key]
            for nested_read in nested_prov.reads:
                # Check for cycle before recursing
                nested_key = (nested_read.obj_id, nested_read.field)
                if nested_key not in _visited:
                    # Check if this nested read has its own provenance
                    if nested_key in _provenance_registry:
                        # Recursively get the full subtree
                        # Create a dummy object reference for the lookup
                        class _ObjRef:
                            pass
                        ref = _ObjRef()
                        object.__setattr__(ref, '__dict__', {})
                        # We can't easily reconstruct the object, so build node directly
                        subtree = _build_derivation_subtree(
                            nested_key, _visited
                        )
                        if subtree is not None:
                            child.children.append(subtree)
                        else:
                            # No deeper provenance, just add the read
                            grandchild = DerivationNode(
                                field=nested_read.field,
                                value=nested_read.value,
                                source_obj_id=nested_read.obj_id,
                                source_obj_type=nested_read.obj_type
                            )
                            child.children.append(grandchild)
                    else:
                        # No provenance for this read, add as leaf
                        grandchild = DerivationNode(
                            field=nested_read.field,
                            value=nested_read.value,
                            source_obj_id=nested_read.obj_id,
                            source_obj_type=nested_read.obj_type
                        )
                        child.children.append(grandchild)

        root.children.append(child)

    return root


def _build_derivation_subtree(
    key: tuple[int, str],
    visited: set[tuple[int, str]]
) -> Optional[DerivationNode]:
    """
    Build a derivation subtree from a provenance registry key.

    Internal helper for recursive tree building.

    Args:
        key: Tuple of (obj_id, field_name) to look up.
        visited: Set of already-visited keys for cycle detection.

    Returns:
        DerivationNode subtree or None if not found/cycle detected.
    """
    if key in visited:
        return None
    if key not in _provenance_registry:
        return None

    visited.add(key)
    prov = _provenance_registry[key]

    node = DerivationNode(
        field=key[1],
        value=prov.value,
        source_obj_id=key[0],
        source_obj_type=prov.computed_by.split('.')[0] if '.' in prov.computed_by else "Unknown"
    )

    # Recursively add children
    for read in prov.reads:
        child_key = (read.obj_id, read.field)
        if child_key in _provenance_registry and child_key not in visited:
            subtree = _build_derivation_subtree(child_key, visited)
            if subtree is not None:
                node.children.append(subtree)
            else:
                # Just add as leaf
                child = DerivationNode(
                    field=read.field,
                    value=read.value,
                    source_obj_id=read.obj_id,
                    source_obj_type=read.obj_type
                )
                node.children.append(child)
        else:
            # No deeper provenance or cycle, add as leaf
            child = DerivationNode(
                field=read.field,
                value=read.value,
                source_obj_id=read.obj_id,
                source_obj_type=read.obj_type
            )
            node.children.append(child)

    return node


__all__ = [
    "ReadRecord",
    "DerivationNode",
    "ComputedProvenance",
    "track_provenance",
    "record_input",
    "record_read",
    "get_current_reads_collector",
    "provenance",
    "clear_provenance",
    "all_provenance",
    "derivation_tree",
]
