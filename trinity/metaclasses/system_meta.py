"""
SystemMeta - Metaclass for ECS systems.

Handles system registration, dependency analysis, and phase ordering.
Systems are functions or classes that operate on components.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, ClassVar, Optional

from trinity.constants import DEFAULT_SYSTEM_PRIORITY
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta
from trinity.types import SystemPhase


class SystemMeta(EngineMeta):
    """
    Metaclass for ECS systems.

    Created classes/functions will:
    - Be registered in the system registry
    - Have dependencies analyzed
    - Be assigned to execution phases
    - Be validated for correct component access

    Required class attributes (set by decorators):
    - _system_phase: SystemPhase (execution phase)

    Optional class attributes (set by decorators):
    - _reads: tuple[type, ...] (components read)
    - _writes: tuple[type, ...] (components written)
    - _resources: tuple[type, ...] (resources accessed)
    - _parallel_config: ParallelConfig | None
    - _fixed_timestep: float | None (for @fixed)
    - _throttle_config: ThrottleConfig | None
    - _profile_config: ProfileConfig | None
    - _exclusive: bool (if True, runs alone)
    - _priority: int (ordering within phase)

    Attached attributes:
    - _system_id: int (unique identifier)
    - _system_name: str (qualified name)
    - _dependencies: set[int] (system IDs this depends on)
    - _can_parallelize: bool (safe to run in parallel)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _phases: ClassVar[dict[SystemPhase, list[int]]] = {
        phase: [] for phase in SystemPhase
    }
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> SystemMeta:
        """Create a new system type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base System class
        if name == "System":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._system_id = mcs._next_id
            mcs._next_id += 1
            cls._system_name = f"{cls.__module__}.{name}"

            # 3.3.2: Record TAG steps for system_id and system_name
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "system_id", "value": cls._system_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "system_name", "value": cls._system_name})
            )

            # === 2. SET DEFAULTS ===
            if not hasattr(cls, "_system_phase"):
                cls._system_phase = SystemPhase.UPDATE
            if not hasattr(cls, "_reads"):
                cls._reads = ()
            if not hasattr(cls, "_writes"):
                cls._writes = ()
            if not hasattr(cls, "_resources"):
                cls._resources = ()
            if not hasattr(cls, "_system_resources"):
                # Extract resource class names for conflict detection
                cls._system_resources = tuple(
                    res.__name__ if isinstance(res, type) else str(res)
                    for res in cls._resources
                )
            if not hasattr(cls, "_exclusive"):
                cls._exclusive = False
            if not hasattr(cls, "_priority"):
                cls._priority = DEFAULT_SYSTEM_PRIORITY

            # 3.3.3: Record TAG steps for defaults
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "system_phase", "value": cls._system_phase})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "reads", "value": cls._reads})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "writes", "value": cls._writes})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "exclusive", "value": cls._exclusive})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "priority", "value": cls._priority})
            )

            # === 3. VALIDATE DECLARATIONS ===
            mcs._validate_declarations(cls)

            # 3.3.4: Record VALIDATE step
            cls._metaclass_steps.append(
                Step(Op.VALIDATE, {"constraint": "system_declarations"})
            )

            # === 4. ANALYZE DEPENDENCIES ===
            cls._dependencies = mcs._analyze_dependencies(cls)
            cls._can_parallelize = mcs._check_parallelization(cls)

            # 3.3.5: Record DESCRIBE step with dependencies and can_parallelize
            cls._metaclass_steps.append(
                Step(Op.DESCRIBE, {
                    "dependencies": cls._dependencies,
                    "can_parallelize": cls._can_parallelize,
                })
            )

            # === 5. REGISTER ===
            mcs._registry[cls._system_id] = cls
            mcs._phases[cls._system_phase].append(cls._system_id)

            # 3.3.6: Record REGISTER step
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {
                    "registry": "system_registry",
                    "id": cls._system_id,
                    "phase": cls._system_phase,
                })
            )

        return cls

    @classmethod
    def _validate_declarations(mcs, cls: type) -> None:
        """Validate @reads/@writes declarations."""
        reads = cls._reads
        writes = cls._writes

        # Check that all declared types are components
        for comp_type in reads + writes:
            if not isinstance(comp_type, type):
                raise TypeError(
                    f"{cls.__name__}: @reads/@writes must reference component types, "
                    f"got {comp_type!r}"
                )
            # Safely check for _component_id attribute
            try:
                if not hasattr(comp_type, "_component_id"):
                    raise TypeError(
                        f"{cls.__name__}: @reads/@writes must reference component types, "
                        f"'{comp_type.__name__}' is not a component"
                    )
            except AttributeError:
                raise TypeError(
                    f"{cls.__name__}: @reads/@writes must reference component types, "
                    f"got invalid type {comp_type!r}"
                )

        # Warn if no declarations (prevents parallelization)
        if not reads and not writes:
            warnings.warn(
                f"{cls.__name__}: System has no @reads or @writes declarations. "
                f"This prevents automatic parallelization and dependency analysis.",
                UserWarning,
                stacklevel=4,
            )

        # Validate execute method exists
        if not hasattr(cls, "execute") and not hasattr(cls, "__call__"):
            warnings.warn(
                f"{cls.__name__}: System should have an 'execute' method or be callable.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def _analyze_dependencies(mcs, cls: type) -> set[int]:
        """
        Determine which systems this one depends on.

        A system depends on another if the other writes components that this one reads.
        """
        dependencies = set()

        reads = set(getattr(cls, "_reads", ()))

        # Check all previously registered systems
        for system_id, system_cls in mcs._registry.items():
            if system_cls is cls:
                continue

            # Only consider systems in the same phase
            if getattr(system_cls, "_system_phase", None) != cls._system_phase:
                continue

            other_writes = set(getattr(system_cls, "_writes", ()))

            # If they write something we read, we depend on them
            if other_writes & reads:
                dependencies.add(system_id)

        return dependencies

    @classmethod
    def _check_parallelization(mcs, cls: type) -> bool:
        """Check if system can run in parallel with others."""
        # Exclusive systems cannot parallelize
        if getattr(cls, "_exclusive", False):
            return False

        reads = set(getattr(cls, "_reads", ()))
        writes = set(getattr(cls, "_writes", ()))

        # Undeclared access = assume not safe
        if not reads and not writes:
            return False

        # Systems that only read can always parallelize with each other
        if not writes:
            return True

        # Systems that write need to be checked against others
        # (actual conflict detection happens in scheduler)
        return True

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, system_id: int) -> Optional[type]:
        """Get system class by ID."""
        return mcs._registry.get(system_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get system class by qualified name."""
        for sys_id, sys_cls in mcs._registry.items():
            if sys_cls._system_name == name:
                return sys_cls
        return None

    @classmethod
    def all_systems(mcs) -> list[type]:
        """Get all registered system classes."""
        return list(mcs._registry.values())

    @classmethod
    def get_phase_systems(mcs, phase: SystemPhase) -> list[type]:
        """Get all systems in a given phase."""
        system_ids = mcs._phases.get(phase, [])
        return [mcs._registry[sid] for sid in system_ids if sid in mcs._registry]

    @classmethod
    def get_phase_order(mcs, phase: SystemPhase) -> list[type]:
        """
        Get systems in a phase, topologically sorted by dependencies.

        Returns systems in the order they should execute.
        """
        systems = mcs.get_phase_systems(phase)

        if not systems:
            return []

        # Build adjacency list and in-degree count
        in_degree = {sys._system_id: 0 for sys in systems}
        dependents = {sys._system_id: [] for sys in systems}

        for sys in systems:
            for dep_id in sys._dependencies:
                if dep_id in in_degree:
                    in_degree[sys._system_id] += 1
                    dependents[dep_id].append(sys._system_id)

        # Kahn's algorithm for topological sort
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        # Sort by priority for deterministic ordering among independent systems
        queue.sort(key=lambda sid: mcs._registry[sid]._priority)

        result = []
        while queue:
            # Pop the highest priority system with no dependencies
            current = queue.pop(0)
            result.append(mcs._registry[current])

            for dep_id in dependents[current]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    # Insert maintaining priority order
                    priority = mcs._registry[dep_id]._priority
                    insert_pos = 0
                    for i, sid in enumerate(queue):
                        if mcs._registry[sid]._priority > priority:
                            break
                        insert_pos = i + 1
                    queue.insert(insert_pos, dep_id)

        # Check for cycles
        if len(result) != len(systems):
            cycle_systems = [
                s for s in systems if s._system_id not in {r._system_id for r in result}
            ]
            cycle_names = [s.__name__ for s in cycle_systems]
            raise RuntimeError(
                f"Circular dependency detected among systems: {cycle_names}"
            )

        return result

    @classmethod
    def get_parallel_groups(mcs, phase: SystemPhase) -> list[list[type]]:
        """
        Get groups of systems that can run in parallel.

        Returns a list of groups, where systems within a group can run in parallel,
        but groups must be executed sequentially.

        Checks both component (reads/writes) and resource conflicts.
        """
        ordered = mcs.get_phase_order(phase)

        if not ordered:
            return []

        groups = []
        current_group = []
        current_writes = set()
        current_resources = set()

        for system in ordered:
            sys_reads = set(getattr(system, "_reads", ()))
            sys_writes = set(getattr(system, "_writes", ()))
            sys_resources = set(getattr(system, "_system_resources", ()))

            # Check if this system conflicts with current group
            conflicts = False

            if not system._can_parallelize:
                conflicts = True
            elif sys_reads & current_writes:
                # Reads something being written by current group
                conflicts = True
            elif sys_writes & current_writes:
                # Writes something being written by current group
                conflicts = True
            elif sys_resources & current_resources:
                # Shares a resource with current group
                conflicts = True

            if conflicts and current_group:
                # Start a new group
                groups.append(current_group)
                current_group = []
                current_writes = set()
                current_resources = set()

            current_group.append(system)
            current_writes |= sys_writes
            current_resources |= sys_resources

        if current_group:
            groups.append(current_group)

        return groups

    @classmethod
    def hot_reload(mcs, old_cls: type, new_cls: type) -> type:
        """
        Replace a system in the registry with a new version.

        Hot-swaps a system class while maintaining the same name and ID.
        Re-runs dependency analysis and updates all references.

        Args:
            old_cls: The existing system class to replace
            new_cls: The new system class with updated implementation

        Returns:
            The new class, now registered

        Raises:
            ValueError: If old_cls is not registered or names don't match
            TypeError: If new_cls is not a proper system
        """
        with mcs._lock:
            # Validate old class is registered
            if not hasattr(old_cls, "_system_id"):
                raise ValueError(f"{old_cls.__name__} is not a registered system")

            old_id = old_cls._system_id
            if old_id not in mcs._registry:
                raise ValueError(f"{old_cls.__name__} (ID {old_id}) not in registry")

            # Validate that registry entry matches old_cls
            if mcs._registry.get(old_id) is not old_cls:
                raise ValueError(
                    f"{old_cls.__name__} (ID {old_id}) registry mismatch - "
                    f"another system may have replaced it"
                )

            # Validate names match
            if old_cls.__name__ != new_cls.__name__:
                raise ValueError(
                    f"System names must match: {old_cls.__name__} != {new_cls.__name__}"
                )

            # Validate new_cls has required attributes/methods
            if not hasattr(new_cls, "execute") and not hasattr(new_cls, "__call__"):
                raise TypeError(
                    f"{new_cls.__name__}: New system must have an 'execute' method or be callable"
                )

            # Copy ID and name from old to new
            new_cls._system_id = old_id
            new_cls._system_name = old_cls._system_name

            # Set defaults for new class
            if not hasattr(new_cls, "_system_phase"):
                new_cls._system_phase = getattr(old_cls, "_system_phase", SystemPhase.UPDATE)
            if not hasattr(new_cls, "_reads"):
                new_cls._reads = ()
            if not hasattr(new_cls, "_writes"):
                new_cls._writes = ()
            if not hasattr(new_cls, "_resources"):
                new_cls._resources = ()
            if not hasattr(new_cls, "_system_resources"):
                new_cls._system_resources = tuple(
                    res.__name__ if isinstance(res, type) else str(res)
                    for res in new_cls._resources
                )
            if not hasattr(new_cls, "_exclusive"):
                new_cls._exclusive = False
            if not hasattr(new_cls, "_priority"):
                new_cls._priority = getattr(old_cls, "_priority", DEFAULT_SYSTEM_PRIORITY)

            # Validate declarations (may raise TypeError)
            try:
                mcs._validate_declarations(new_cls)
            except (TypeError, AttributeError) as e:
                raise TypeError(f"Hot reload validation failed: {e}")

            # Re-analyze dependencies for the new class
            new_cls._dependencies = mcs._analyze_dependencies(new_cls)
            new_cls._can_parallelize = mcs._check_parallelization(new_cls)

            # Update registry
            mcs._registry[old_id] = new_cls

            # Update phase registry if phase changed
            old_phase = old_cls._system_phase
            new_phase = new_cls._system_phase
            if old_phase != new_phase:
                if old_id in mcs._phases.get(old_phase, []):
                    mcs._phases[old_phase].remove(old_id)
                if old_id not in mcs._phases.get(new_phase, []):
                    mcs._phases[new_phase].append(old_id)

            # Re-analyze dependencies for all systems that might depend on this one
            for system in mcs.get_phase_systems(new_phase):
                if system._system_id != old_id:
                    system._dependencies = mcs._analyze_dependencies(system)

            return new_cls

    @classmethod
    def reload_system(mcs, name: str) -> Optional[type]:
        """
        Re-validate and refresh dependency analysis for an existing system.

        Args:
            name: The qualified name of the system to reload

        Returns:
            The reloaded system class, or None if not found

        Raises:
            RuntimeError: If validation fails
        """
        with mcs._lock:
            system_cls = mcs.get_by_name(name)
            if system_cls is None:
                return None

            # Verify system is still in registry
            if not hasattr(system_cls, "_system_id"):
                raise RuntimeError(f"System {name} is missing _system_id attribute")

            system_id = system_cls._system_id
            if system_id not in mcs._registry:
                raise RuntimeError(f"System {name} (ID {system_id}) not found in registry")

            # Re-validate (may raise TypeError)
            try:
                mcs._validate_declarations(system_cls)
            except (TypeError, AttributeError) as e:
                raise RuntimeError(f"Reload validation failed for {name}: {e}")

            # Re-analyze dependencies
            system_cls._dependencies = mcs._analyze_dependencies(system_cls)
            system_cls._can_parallelize = mcs._check_parallelization(system_cls)

            # Re-analyze dependencies for other systems in the same phase
            phase = system_cls._system_phase
            for other_system in mcs.get_phase_systems(phase):
                if other_system._system_id != system_cls._system_id:
                    other_system._dependencies = mcs._analyze_dependencies(other_system)

            return system_cls

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the system registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._phases = {phase: [] for phase in SystemPhase}
            mcs._next_id = 1
        super().clear_registry()
