"""Animation Session Persistence Integration (T-AN-9.11).

This module provides session persistence for animation state, allowing animation
state machines, blend parameters, and IK settings to persist across sessions
while keeping transient runtime data (bone transforms, playback state) ephemeral.

Key Features:
- AnimationSessionData: Serializable container for persistent animation state
- save_animation_state(): Extract persistent state from an animated entity
- restore_animation_state(): Apply saved state to an entity with partial restore support
- @serializable decorator integration for Foundation Serializer
- Version compatibility for forward/backward migration

What Persists (PERSIST):
- State machine current state (state_machine.current_state)
- Blend parameters (blend_params)
- IK enabled flags (ik_enabled per chain)
- Graph parameters (graph_params)

What Does NOT Persist (TRANSIENT):
- Bone transforms (computed at runtime)
- Clip playback time (clip_time, normalized_time)
- Active transitions (interpolated_pose)
- Cached poses
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING

from foundation import (
    to_dict,
    from_dict,
    register_type,
    schema_hash,
    mirror,
)
from engine.animation.graph.animation_graph import (
    GraphParameter,
    ParameterType,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity
    from engine.animation.graph.state_machine import StateMachine
    from engine.animation.graph.animation_graph import AnimationGraph


__all__ = [
    "AnimationSessionData",
    "StateMachineState",
    "BlendParameterState",
    "IKChainState",
    "GraphParameterState",
    "AnimationSessionError",
    "RestoreResult",
    "save_animation_state",
    "restore_animation_state",
    "serializable",
    "transient",
    "SESSION_VERSION",
]


logger = logging.getLogger(__name__)


# Current session format version for migration support
SESSION_VERSION = 1


# =============================================================================
# EXCEPTIONS
# =============================================================================


class AnimationSessionError(Exception):
    """Base exception for animation session operations."""


class StateMachineNotFoundError(AnimationSessionError):
    """Raised when a state machine referenced in session data doesn't exist."""

    def __init__(self, machine_id: str) -> None:
        self.machine_id = machine_id
        super().__init__(f"State machine not found: {machine_id}")


class StateNotFoundError(AnimationSessionError):
    """Raised when a state referenced in session data doesn't exist."""

    def __init__(self, state_name: str, machine_id: str) -> None:
        self.state_name = state_name
        self.machine_id = machine_id
        super().__init__(f"State '{state_name}' not found in machine '{machine_id}'")


class VersionMismatchError(AnimationSessionError):
    """Raised when session data version is incompatible."""

    def __init__(self, stored_version: int, current_version: int) -> None:
        self.stored_version = stored_version
        self.current_version = current_version
        super().__init__(
            f"Session version mismatch: stored={stored_version}, current={current_version}"
        )


# =============================================================================
# DECORATOR FOR TRANSIENT FIELDS
# =============================================================================


def transient(cls_or_field: Any = None) -> Any:
    """Decorator to mark a class or field as transient (not persisted).

    Can be used as a class decorator or field metadata.

    Usage:
        @transient
        class BoneTransformCache:
            ...

        @dataclass
        class AnimationComponent:
            pose: Pose = field(metadata={"transient": True})

    Args:
        cls_or_field: The class or field to mark as transient.

    Returns:
        Decorated class or field with transient metadata.
    """
    if cls_or_field is None:
        # Used as @transient() with parentheses
        return lambda x: _mark_transient(x)
    return _mark_transient(cls_or_field)


def _mark_transient(obj: Any) -> Any:
    """Internal helper to mark an object as transient."""
    if isinstance(obj, type):
        obj._transient = True
    elif hasattr(obj, "metadata"):
        obj.metadata = {**obj.metadata, "transient": True}
    else:
        # For simple objects, attach metadata
        obj._transient = True
    return obj


def is_transient(obj: Any) -> bool:
    """Check if an object or field is marked as transient.

    Args:
        obj: Object or field to check.

    Returns:
        True if marked transient, False otherwise.
    """
    if hasattr(obj, "_transient"):
        return obj._transient
    if hasattr(obj, "metadata"):
        return obj.metadata.get("transient", False)
    return False


# =============================================================================
# DECORATOR FOR SERIALIZABLE CLASSES
# =============================================================================


def serializable(
    name: Optional[str] = None,
    version: int = 1,
    exclude_fields: Optional[Set[str]] = None,
) -> Callable[[Type], Type]:
    """Decorator to mark a class as serializable for session persistence.

    Registers the class with Foundation's Serializer and adds metadata
    for version tracking and field exclusion.

    Usage:
        @serializable(version=1)
        class MyAnimationData:
            ...

    Args:
        name: Optional custom type name for registration.
        version: Schema version for migration support.
        exclude_fields: Fields to exclude from serialization.

    Returns:
        Decorated class registered with Foundation Serializer.
    """
    def decorator(cls: Type) -> Type:
        # Register with Foundation
        type_name = name or f"{cls.__module__}.{cls.__name__}"
        register_type(cls, type_name)

        # Add serialization metadata
        cls._serializable = True
        cls._serializable_version = version
        cls._serializable_exclude = exclude_fields or set()

        # Override __before_serialize__ to handle exclusions
        original_before = getattr(cls, "__before_serialize__", None)

        def __before_serialize__(self: Any) -> None:
            if original_before:
                original_before(self)
            # Mark excluded fields as transient temporarily
            for field_name in cls._serializable_exclude:
                if hasattr(self, field_name):
                    setattr(self, f"_transient_{field_name}", getattr(self, field_name))

        cls.__before_serialize__ = __before_serialize__
        return cls

    return decorator


# =============================================================================
# STATE DATA CLASSES
# =============================================================================


@dataclass
class StateMachineState:
    """Persistent state for a single state machine.

    Captures the current state name (not the runtime AnimationState object)
    so it can be restored after session reload.
    """

    machine_id: str
    """Unique identifier for the state machine."""

    current_state: Optional[str]
    """Name of the current active state (None if not started)."""

    initial_state: Optional[str]
    """Name of the initial/default state."""

    # Not persisted: active transition, transition progress, runtime state objects


@dataclass
class BlendParameterState:
    """Persistent state for a blend parameter.

    Stores the parameter value with type information for validation
    during restore.
    """

    name: str
    """Parameter name."""

    value: Any
    """Current parameter value."""

    param_type: str
    """Type of parameter (FLOAT, INT, BOOL, etc.)."""

    min_value: Optional[float] = None
    """Minimum value constraint (for numeric types)."""

    max_value: Optional[float] = None
    """Maximum value constraint (for numeric types)."""

    @classmethod
    def from_graph_parameter(cls, param: GraphParameter) -> "BlendParameterState":
        """Create BlendParameterState from a GraphParameter.

        Args:
            param: The GraphParameter to snapshot.

        Returns:
            BlendParameterState capturing the parameter's current value.
        """
        return cls(
            name=param.name,
            value=param._value,  # Access internal value, not property
            param_type=param.param_type.name,
            min_value=param.min_value,
            max_value=param.max_value,
        )

    def apply_to(self, param: GraphParameter) -> bool:
        """Apply this state to a GraphParameter.

        Args:
            param: Target parameter to update.

        Returns:
            True if applied successfully, False on type mismatch.
        """
        # Skip triggers - they are transient by design
        if param.param_type == ParameterType.TRIGGER:
            return True

        try:
            param.value = self.value
            return True
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to restore parameter '{self.name}': {e}"
            )
            return False


@dataclass
class IKChainState:
    """Persistent state for an IK chain.

    Only persists enable/disable flag and weight, not runtime
    goal positions or solved poses.
    """

    chain_id: str
    """Unique identifier for the IK chain."""

    enabled: bool
    """Whether IK is enabled for this chain."""

    weight: float = 1.0
    """Blend weight for IK (0-1)."""

    # Not persisted: goal positions, solved transforms, interpolation state


@dataclass
class GraphParameterState:
    """Persistent state for animation graph parameters.

    Captures all non-trigger graph parameters for session restore.
    """

    parameters: Dict[str, BlendParameterState] = field(default_factory=dict)
    """Map of parameter name to state."""

    @classmethod
    def from_graph(cls, params: Dict[str, GraphParameter]) -> "GraphParameterState":
        """Create GraphParameterState from a parameter dictionary.

        Args:
            params: Dictionary of GraphParameters.

        Returns:
            GraphParameterState capturing all non-trigger parameters.
        """
        result = cls()
        for name, param in params.items():
            # Skip triggers - they reset each frame
            if param.param_type != ParameterType.TRIGGER:
                result.parameters[name] = BlendParameterState.from_graph_parameter(param)
        return result

    def apply_to(self, params: Dict[str, GraphParameter]) -> int:
        """Apply this state to a parameter dictionary.

        Args:
            params: Target parameter dictionary.

        Returns:
            Number of parameters successfully restored.
        """
        restored = 0
        for name, state in self.parameters.items():
            if name in params:
                if state.apply_to(params[name]):
                    restored += 1
            else:
                logger.debug(f"Parameter '{name}' not found in graph, skipping")
        return restored


# =============================================================================
# ANIMATION SESSION DATA
# =============================================================================


@serializable(version=SESSION_VERSION)
@dataclass
class AnimationSessionData:
    """Complete persistent animation state for an entity.

    This is the top-level container for all animation state that should
    persist across sessions. It includes state machines, blend parameters,
    IK settings, and graph parameters.

    Transient data (bone transforms, clip times, poses) is NOT included
    as it is recomputed at runtime.
    """

    entity_id: Optional[str] = None
    """Entity identifier this data belongs to (optional, for lookup)."""

    session_version: int = SESSION_VERSION
    """Version of the session format for migration."""

    timestamp: float = field(default_factory=time.time)
    """When this state was captured."""

    # Persistent state components
    state_machines: Dict[str, StateMachineState] = field(default_factory=dict)
    """State machine states indexed by machine ID."""

    blend_params: Dict[str, BlendParameterState] = field(default_factory=dict)
    """Blend parameters indexed by name."""

    ik_chains: Dict[str, IKChainState] = field(default_factory=dict)
    """IK chain states indexed by chain ID."""

    graph_params: Optional[GraphParameterState] = None
    """Animation graph parameter state."""

    # Metadata
    schema_hash_stored: Optional[str] = None
    """Schema hash at save time for migration detection."""

    def __post_init__(self) -> None:
        """Compute schema hash on initialization."""
        if self.schema_hash_stored is None:
            self.schema_hash_stored = schema_hash(AnimationSessionData)

    def is_empty(self) -> bool:
        """Check if this session data contains any state.

        Returns:
            True if no state has been captured.
        """
        return (
            not self.state_machines
            and not self.blend_params
            and not self.ik_chains
            and self.graph_params is None
        )

    def get_state_machine_names(self) -> List[str]:
        """Get names of all stored state machines.

        Returns:
            List of state machine IDs.
        """
        return list(self.state_machines.keys())

    def get_blend_param_names(self) -> List[str]:
        """Get names of all stored blend parameters.

        Returns:
            List of blend parameter names.
        """
        return list(self.blend_params.keys())

    def get_ik_chain_ids(self) -> List[str]:
        """Get IDs of all stored IK chains.

        Returns:
            List of IK chain IDs.
        """
        return list(self.ik_chains.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary using Foundation Serializer.

        Returns:
            Dictionary representation suitable for JSON/storage.
        """
        return to_dict(self, include_schema_hash=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimationSessionData":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            AnimationSessionData instance.

        Raises:
            VersionMismatchError: If version is incompatible and no migration exists.
        """
        # Check version before deserializing
        stored_version = data.get("session_version", 1)
        if stored_version > SESSION_VERSION:
            raise VersionMismatchError(stored_version, SESSION_VERSION)

        # Apply migrations if needed
        if stored_version < SESSION_VERSION:
            data = _migrate_session_data(data, stored_version, SESSION_VERSION)

        return from_dict(data)


# =============================================================================
# RESTORE RESULT
# =============================================================================


@dataclass
class RestoreResult:
    """Result of a restore operation with details on what was restored.

    Provides fine-grained feedback on which components were restored,
    skipped, or failed.
    """

    success: bool
    """Overall success (True if at least partial restore succeeded)."""

    state_machines_restored: int = 0
    """Number of state machines successfully restored."""

    state_machines_failed: int = 0
    """Number of state machines that failed to restore."""

    state_machines_missing: int = 0
    """Number of state machines in data but not found in entity."""

    blend_params_restored: int = 0
    """Number of blend parameters successfully restored."""

    blend_params_failed: int = 0
    """Number of blend parameters that failed to restore."""

    blend_params_missing: int = 0
    """Number of blend parameters in data but not found in entity."""

    ik_chains_restored: int = 0
    """Number of IK chains successfully restored."""

    ik_chains_failed: int = 0
    """Number of IK chains that failed to restore."""

    ik_chains_missing: int = 0
    """Number of IK chains in data but not found in entity."""

    graph_params_restored: int = 0
    """Number of graph parameters successfully restored."""

    errors: List[str] = field(default_factory=list)
    """List of error messages encountered."""

    warnings: List[str] = field(default_factory=list)
    """List of warning messages (non-fatal issues)."""

    @property
    def total_restored(self) -> int:
        """Total number of components restored."""
        return (
            self.state_machines_restored
            + self.blend_params_restored
            + self.ik_chains_restored
            + self.graph_params_restored
        )

    @property
    def total_failed(self) -> int:
        """Total number of components that failed to restore."""
        return (
            self.state_machines_failed
            + self.blend_params_failed
            + self.ik_chains_failed
        )

    @property
    def total_missing(self) -> int:
        """Total number of components not found in entity."""
        return (
            self.state_machines_missing
            + self.blend_params_missing
            + self.ik_chains_missing
        )

    @property
    def is_partial(self) -> bool:
        """Check if this was a partial restore (some failures/missing)."""
        return self.total_failed > 0 or self.total_missing > 0


# =============================================================================
# SESSION MIGRATION
# =============================================================================


def _migrate_session_data(
    data: Dict[str, Any],
    from_version: int,
    to_version: int,
) -> Dict[str, Any]:
    """Migrate session data between versions.

    Args:
        data: Session data dictionary to migrate.
        from_version: Source version.
        to_version: Target version.

    Returns:
        Migrated session data.
    """
    logger.info(f"Migrating session data from v{from_version} to v{to_version}")

    # Apply migrations sequentially
    current = data.copy()
    version = from_version

    while version < to_version:
        migration_fn = _MIGRATIONS.get((version, version + 1))
        if migration_fn:
            current = migration_fn(current)
        version += 1

    current["session_version"] = to_version
    return current


# Migration functions: (from_version, to_version) -> migration_fn
_MIGRATIONS: Dict[Tuple[int, int], Callable[[Dict], Dict]] = {
    # Add migration functions here as versions evolve
    # Example: (1, 2): _migrate_v1_to_v2,
}


# =============================================================================
# SAVE/RESTORE FUNCTIONS
# =============================================================================


def save_animation_state(
    entity: Any,
    include_state_machines: bool = True,
    include_blend_params: bool = True,
    include_ik: bool = True,
    include_graph_params: bool = True,
) -> AnimationSessionData:
    """Extract persistent animation state from an entity.

    This function traverses an entity's animation components and extracts
    all persistent state into an AnimationSessionData object. Transient
    data (bone transforms, clip times, poses) is NOT captured.

    Args:
        entity: Entity to extract state from. Expected to have animation
                components accessible via attributes or component system.
        include_state_machines: Whether to save state machine states.
        include_blend_params: Whether to save blend parameters.
        include_ik: Whether to save IK chain states.
        include_graph_params: Whether to save graph parameters.

    Returns:
        AnimationSessionData containing all persistent state.

    Example:
        session_data = save_animation_state(player_entity)
        serialized = session_data.to_dict()
        # Save to file or database
    """
    session = AnimationSessionData(
        entity_id=_get_entity_id(entity),
        timestamp=time.time(),
    )

    # Extract state machines
    if include_state_machines:
        _save_state_machines(entity, session)

    # Extract blend parameters
    if include_blend_params:
        _save_blend_params(entity, session)

    # Extract IK chain states
    if include_ik:
        _save_ik_chains(entity, session)

    # Extract graph parameters
    if include_graph_params:
        _save_graph_params(entity, session)

    logger.debug(
        f"Saved animation state for entity '{session.entity_id}': "
        f"{len(session.state_machines)} state machines, "
        f"{len(session.blend_params)} blend params, "
        f"{len(session.ik_chains)} IK chains"
    )

    return session


def restore_animation_state(
    entity: Any,
    session: AnimationSessionData,
    restore_state_machines: bool = True,
    restore_blend_params: bool = True,
    restore_ik: bool = True,
    restore_graph_params: bool = True,
    strict: bool = False,
) -> RestoreResult:
    """Apply saved animation state to an entity.

    This function takes an AnimationSessionData object and applies it to
    an entity, restoring state machine states, blend parameters, and IK
    settings. Handles partial restore gracefully when components are
    missing from the entity.

    Args:
        entity: Entity to restore state to.
        session: AnimationSessionData containing saved state.
        restore_state_machines: Whether to restore state machine states.
        restore_blend_params: Whether to restore blend parameters.
        restore_ik: Whether to restore IK chain states.
        restore_graph_params: Whether to restore graph parameters.
        strict: If True, fail on any missing component. If False, skip
                missing components and continue.

    Returns:
        RestoreResult with details on what was restored.

    Raises:
        AnimationSessionError: If strict=True and components are missing.

    Example:
        result = restore_animation_state(player_entity, session_data)
        if result.is_partial:
            logger.warning(f"Partial restore: {result.warnings}")
    """
    result = RestoreResult(success=True)

    try:
        # Restore state machines
        if restore_state_machines:
            _restore_state_machines(entity, session, result, strict)

        # Restore blend parameters
        if restore_blend_params:
            _restore_blend_params(entity, session, result, strict)

        # Restore IK chain states
        if restore_ik:
            _restore_ik_chains(entity, session, result, strict)

        # Restore graph parameters
        if restore_graph_params:
            _restore_graph_params(entity, session, result, strict)

    except AnimationSessionError as e:
        result.success = False
        result.errors.append(str(e))
        if strict:
            raise

    # Mark success if at least something was restored
    if result.total_restored == 0 and not session.is_empty():
        result.success = False
        if not result.errors:
            result.errors.append("No components were restored")

    logger.debug(
        f"Restored animation state for entity: "
        f"{result.total_restored} restored, "
        f"{result.total_failed} failed, "
        f"{result.total_missing} missing"
    )

    return result


# =============================================================================
# INTERNAL SAVE HELPERS
# =============================================================================


def _get_entity_id(entity: Any) -> Optional[str]:
    """Extract entity ID from various entity representations."""
    if hasattr(entity, "entity_id"):
        return str(entity.entity_id)
    if hasattr(entity, "id"):
        return str(entity.id)
    if hasattr(entity, "name"):
        return str(entity.name)
    return None


def _save_state_machines(entity: Any, session: AnimationSessionData) -> None:
    """Extract state machine states from entity."""
    # Try different component access patterns
    state_machines = _get_state_machines(entity)

    for machine_id, machine in state_machines.items():
        if machine is None:
            continue

        # Safely get current state name
        current_state_name = None
        if hasattr(machine, "current_state_name"):
            current_state_name = machine.current_state_name
        elif hasattr(machine, "current_state") and machine.current_state is not None:
            if hasattr(machine.current_state, "name"):
                current_state_name = machine.current_state.name

        # Safely get initial state
        initial_state = None
        if hasattr(machine, "_initial_state"):
            initial_state = machine._initial_state

        session.state_machines[machine_id] = StateMachineState(
            machine_id=machine_id,
            current_state=current_state_name,
            initial_state=initial_state,
        )


def _get_state_machines(entity: Any) -> Dict[str, Any]:
    """Get state machines from entity using various access patterns."""
    machines = {}

    # Pattern 1: Direct attribute
    if hasattr(entity, "state_machine"):
        sm = entity.state_machine
        if sm is not None:
            machines["default"] = sm

    # Pattern 2: Animation component
    if hasattr(entity, "animation"):
        anim = entity.animation
        if anim is not None:
            if hasattr(anim, "state_machine") and anim.state_machine is not None:
                machines["default"] = anim.state_machine
            if hasattr(anim, "state_machines") and anim.state_machines:
                machines.update(anim.state_machines)

    # Pattern 3: Animation graph with state machines
    if hasattr(entity, "animation_graph"):
        graph = entity.animation_graph
        if graph is not None:
            if hasattr(graph, "get_state_machines"):
                result = graph.get_state_machines()
                if result:
                    machines.update(result)
            elif hasattr(graph, "state_machine") and graph.state_machine is not None:
                machines["default"] = graph.state_machine

    # Pattern 4: Component system (ECS)
    if hasattr(entity, "get_component"):
        anim_comp = entity.get_component("AnimationComponent")
        if anim_comp and hasattr(anim_comp, "state_machines") and anim_comp.state_machines:
            machines.update(anim_comp.state_machines)

    return machines


def _save_blend_params(entity: Any, session: AnimationSessionData) -> None:
    """Extract blend parameters from entity."""
    params = _get_blend_params(entity)

    for name, param in params.items():
        # Skip triggers - they are transient
        if isinstance(param, GraphParameter) and param.param_type == ParameterType.TRIGGER:
            continue
        session.blend_params[name] = BlendParameterState.from_graph_parameter(param)


def _get_blend_params(entity: Any) -> Dict[str, GraphParameter]:
    """Get blend parameters from entity using various access patterns."""
    params = {}

    # Pattern 1: Direct attribute
    if hasattr(entity, "blend_params"):
        params.update(entity.blend_params)

    # Pattern 2: Animation component
    if hasattr(entity, "animation"):
        anim = entity.animation
        if hasattr(anim, "parameters"):
            params.update(anim.parameters)
        if hasattr(anim, "blend_parameters"):
            params.update(anim.blend_parameters)

    # Pattern 3: Animation graph context
    if hasattr(entity, "animation_context"):
        ctx = entity.animation_context
        if hasattr(ctx, "parameters"):
            params.update(ctx.parameters)

    # Pattern 4: Graph parameters
    if hasattr(entity, "animation_graph"):
        graph = entity.animation_graph
        if hasattr(graph, "parameters"):
            params.update(graph.parameters)

    # Pattern 5: Component system
    if hasattr(entity, "get_component"):
        anim_comp = entity.get_component("AnimationComponent")
        if anim_comp and hasattr(anim_comp, "parameters"):
            params.update(anim_comp.parameters)

    return params


def _save_ik_chains(entity: Any, session: AnimationSessionData) -> None:
    """Extract IK chain states from entity."""
    chains = _get_ik_chains(entity)

    for chain_id, chain in chains.items():
        enabled = True
        weight = 1.0

        # Extract enabled state
        if hasattr(chain, "enabled"):
            enabled = chain.enabled
        elif hasattr(chain, "is_enabled"):
            enabled = chain.is_enabled()

        # Extract weight
        if hasattr(chain, "weight"):
            weight = chain.weight
        elif hasattr(chain, "blend_weight"):
            weight = chain.blend_weight

        session.ik_chains[chain_id] = IKChainState(
            chain_id=chain_id,
            enabled=enabled,
            weight=weight,
        )


def _get_ik_chains(entity: Any) -> Dict[str, Any]:
    """Get IK chains from entity using various access patterns."""
    chains = {}

    # Pattern 1: Direct attribute
    if hasattr(entity, "ik_chains"):
        chains.update(entity.ik_chains)

    # Pattern 2: IK component
    if hasattr(entity, "ik"):
        ik = entity.ik
        if hasattr(ik, "chains"):
            chains.update(ik.chains)
        if hasattr(ik, "get_all_chains"):
            chains.update(ik.get_all_chains())

    # Pattern 3: Animation component with IK
    if hasattr(entity, "animation"):
        anim = entity.animation
        if hasattr(anim, "ik_chains"):
            chains.update(anim.ik_chains)
        if hasattr(anim, "ik_system"):
            ik_sys = anim.ik_system
            if hasattr(ik_sys, "chains"):
                chains.update(ik_sys.chains)

    # Pattern 4: Component system
    if hasattr(entity, "get_component"):
        ik_comp = entity.get_component("IKComponent")
        if ik_comp and hasattr(ik_comp, "chains"):
            chains.update(ik_comp.chains)

    return chains


def _save_graph_params(entity: Any, session: AnimationSessionData) -> None:
    """Extract animation graph parameters from entity."""
    params = _get_graph_params(entity)
    if params:
        session.graph_params = GraphParameterState.from_graph(params)


def _get_graph_params(entity: Any) -> Dict[str, GraphParameter]:
    """Get animation graph parameters from entity."""
    # Reuse blend params accessor - graph params are typically the same
    return _get_blend_params(entity)


# =============================================================================
# INTERNAL RESTORE HELPERS
# =============================================================================


def _restore_state_machines(
    entity: Any,
    session: AnimationSessionData,
    result: RestoreResult,
    strict: bool,
) -> None:
    """Restore state machine states to entity."""
    machines = _get_state_machines(entity)

    for machine_id, sm_state in session.state_machines.items():
        if machine_id not in machines:
            result.state_machines_missing += 1
            msg = f"State machine '{machine_id}' not found in entity"
            result.warnings.append(msg)
            if strict:
                raise StateMachineNotFoundError(machine_id)
            continue

        machine = machines[machine_id]

        # Skip if no current state to restore
        if sm_state.current_state is None:
            result.state_machines_restored += 1
            continue

        # Check if state exists in machine
        if hasattr(machine, "get_state"):
            state = machine.get_state(sm_state.current_state)
            if state is None:
                result.state_machines_failed += 1
                msg = f"State '{sm_state.current_state}' not found in '{machine_id}'"
                result.warnings.append(msg)
                if strict:
                    raise StateNotFoundError(sm_state.current_state, machine_id)
                continue

        # Force state transition (immediate, no blend)
        try:
            if hasattr(machine, "force_state"):
                # Create a minimal context for the transition
                from engine.animation.graph.animation_graph import GraphContext
                ctx = GraphContext()
                machine.force_state(sm_state.current_state, ctx, immediate=True)
            elif hasattr(machine, "_current_state"):
                # Direct assignment fallback
                if hasattr(machine, "states"):
                    machine._current_state = machine.states.get(sm_state.current_state)

            result.state_machines_restored += 1
        except Exception as e:
            result.state_machines_failed += 1
            result.errors.append(f"Failed to restore state machine '{machine_id}': {e}")


def _restore_blend_params(
    entity: Any,
    session: AnimationSessionData,
    result: RestoreResult,
    strict: bool,
) -> None:
    """Restore blend parameters to entity."""
    params = _get_blend_params(entity)

    for name, param_state in session.blend_params.items():
        if name not in params:
            result.blend_params_missing += 1
            result.warnings.append(f"Blend parameter '{name}' not found in entity")
            if strict:
                raise AnimationSessionError(f"Blend parameter '{name}' not found")
            continue

        if param_state.apply_to(params[name]):
            result.blend_params_restored += 1
        else:
            result.blend_params_failed += 1


def _restore_ik_chains(
    entity: Any,
    session: AnimationSessionData,
    result: RestoreResult,
    strict: bool,
) -> None:
    """Restore IK chain states to entity."""
    chains = _get_ik_chains(entity)

    for chain_id, chain_state in session.ik_chains.items():
        if chain_id not in chains:
            result.ik_chains_missing += 1
            result.warnings.append(f"IK chain '{chain_id}' not found in entity")
            if strict:
                raise AnimationSessionError(f"IK chain '{chain_id}' not found")
            continue

        chain = chains[chain_id]

        try:
            # Restore enabled state
            restored_enabled = False
            if hasattr(chain, "set_enabled"):
                chain.set_enabled(chain_state.enabled)
                restored_enabled = True
            elif hasattr(chain, "enabled"):
                chain.enabled = chain_state.enabled
                restored_enabled = True

            # Restore weight
            restored_weight = False
            if hasattr(chain, "set_weight"):
                chain.set_weight(chain_state.weight)
                restored_weight = True
            elif hasattr(chain, "weight"):
                chain.weight = chain_state.weight
                restored_weight = True
            elif hasattr(chain, "blend_weight"):
                chain.blend_weight = chain_state.weight
                restored_weight = True

            if restored_enabled or restored_weight:
                result.ik_chains_restored += 1
            else:
                result.ik_chains_failed += 1
                result.warnings.append(f"IK chain '{chain_id}' has no settable properties")
        except Exception as e:
            result.ik_chains_failed += 1
            result.errors.append(f"Failed to restore IK chain '{chain_id}': {e}")


def _restore_graph_params(
    entity: Any,
    session: AnimationSessionData,
    result: RestoreResult,
    strict: bool,
) -> None:
    """Restore animation graph parameters to entity."""
    if session.graph_params is None:
        return

    params = _get_graph_params(entity)
    if not params:
        if strict and session.graph_params.parameters:
            raise AnimationSessionError("No graph parameters found in entity")
        return

    restored = session.graph_params.apply_to(params)
    result.graph_params_restored = restored
