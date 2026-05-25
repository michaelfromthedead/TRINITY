"""
Decorator Registry for the Trinity Pattern.

Provides centralized registration, validation, and introspection of all
decorators in the system. The registry ensures decorators are applied
in the correct tier order and validates composition rules.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "Tier",
    "DecoratorSpec",
    "StackSpec",
    "DecoratorValidationError",
    "DecoratorRegistry",
    "registry",
    "DecoratorInfo",
    "get_decorator_chain",
    "has_decorator",
    "inspect_decorated",
    "track_decorator",
    "validate_decorator_requirements",
]


class Tier(IntEnum):
    """
    Decorator tier levels (0-53).

    Decorators must be applied in tier order (lower tiers first / innermost).
    The tier system ensures that foundation decorators are processed before
    decorators that depend on them.
    """

    # Foundation (Tiers 0-9)
    COMPILATION = 0  # @native, @ffi, @target, @unsafe, @backend, @capability, @platform
    ECS_CORE = 1  # @component, @tag, @resource, @event, @system, @query, @bundle, @relation, @derived
    MEMORY = 2  # @pooled, @packed, @aligned, @arena, @flyweight, @intern, @generations, @copy_on_write, @inline_array, @budget, @allocator, @atomic
    SCHEDULING = 3  # @phase, @parallel, @exclusive, @after, @before, @run_if, @fixed, @job, @async_system, @throttle, @deferred, @chain
    DATA_FLOW = 4  # @serializable, @networked, @snapshot, @versioned
    GPU = 5  # @gpu_buffer, @gpu_kernel, @gpu_struct, @bind_group, @dispatch, @shader, @render_pass, @async_compute
    DEV = 6  # @profile, @gpu_profile, @trace, @reloadable, @editor, @test, @bench, @invariant, @deprecated
    LIFECYCLE = 7  # @on_add, @on_remove, @on_change, @on_spawn, @on_despawn
    ASSETS = 8  # @asset, @preload, @cook, @residency, @import_settings
    AI_GENERATION = (
        9  # @example, @constraints, @stub, @pattern, @complexity, @generates, @pure
    )

    # Engine Systems (Tiers 10-21)
    DEBUG_SAFETY = 10  # @reads, @writes, @trace_stack
    CHANGE_DETECTION = 11  # @track_changes
    STATE_MACHINE = 12  # @state_machine, @on_enter, @on_exit
    INPUT = 13  # @input_action, @input_axis
    AUDIO = 14  # @sound, @audio_bus, @spatial
    UI = 15  # @widget, @layout
    SPATIAL = 16  # @spatial, @partitioned
    ANIMATION = 17  # @tween, @blend_tree
    TRANSACTIONS = 18  # @transactional, @undoable
    NETWORK_RPC = 19  # @rpc
    PREFABS = 20  # @prefab, @extends
    COMPOSITION = 21  # @composite, @alias

    # Extended Core (Tiers 22-41)
    LOCALIZATION = 22  # @localized, @plural, @rtl
    ACCESSIBILITY = 23  # @accessible
    LOD_STREAMING = 24  # @lod, @streamable, @chunk
    TIME = 25  # @time_scale, @pausable, @rewindable, @deterministic
    REPLAY = 26  # @recorded, @replay_authority, @keyframe
    DEBUG_CHEAT = 27  # @cheat, @debug_draw, @inspector
    ACHIEVEMENTS = 28  # @achievement, @progress, @stat
    ANALYTICS = 29  # @telemetry, @funnel, @heatmap
    MODDING = 30  # @mod, @requires, @conflicts, @provides, @replaces, @mod_extends, @patch, @load_order, @moddable
    SECURITY = 31  # @server_authoritative, @validated
    PLATFORM = 32  # @battery_aware (note: @platform is Tier 0)
    SAVE_SYSTEM = 33  # @save_slot, @atomic_save, @cloud_sync
    NARRATIVE = 34  # @dialogue, @conversation
    CINEMATICS = 35  # @cutscene, @camera_track
    GAME_AI = 36  # @behavior_tree, @utility_ai
    PROCEDURAL = 37  # @seeded, @procedural, @constraint
    ECONOMY = 38  # @currency, @transaction
    SOCIAL = 39  # @social, @leaderboard
    ERROR_HANDLING = 40  # @crash_safe, @recoverable
    BUILD_DEPLOY = 41  # @build_only, @feature_flag

    # Graphics & Physics (Tiers 42-45)
    RENDERING = 42  # @gi_contributor, @shadow_caster, @material_domain
    DESTRUCTION = 43  # @destructible, @damage_type
    IK_PROCEDURAL = 44  # @ik_chain, @ragdoll, @motion
    PARTICLES_VFX = 45  # @particle_emitter, @trail

    # Advanced Systems (Tiers 46-52)
    PHYSICS_SIM = 46  # @simulation_domain, @substep, @continuous_collision, @buoyancy, @wind_affected
    GAMEPLAY = 47  # @ability, @buff, @quest
    WORLD_BUILDING = 48  # @foliage_type, @trigger_volume
    AUDIO_EXTENDED = 49  # @dsp_node, @sidechain, @music_stem, @reverb_zone
    NETWORK_EXTENDED = (
        50  # @interest, @bandwidth_priority, @snapshot_interpolation, @server_reconcile
    )
    DEBUG_EXTENDED = 51  # @automation_test
    CRAFTING = 52  # @recipe, @loot_table, @ingredient
    BRIDGES_CACHING = 53  # @cached, @lazy, @batch, @async_load, @diff, @priority, @retry, @throttle_network, @observable


@dataclass
class DecoratorSpec:
    """
    Specification for a registered decorator.

    Attributes:
        name: The decorator name (e.g., 'component', 'native').
        tier: The tier this decorator belongs to.
        func: The decorator implementation function.
        requires: Decorator names that must be present on the target.
        excludes: Decorator names that cannot be present on the target.
        unique: Whether this decorator can only be applied once per target.
        foundation: Whether this is a foundation decorator that others depend on.
        doc: Documentation string for the decorator.
        target_types: Valid target types ('class', 'function', 'method', 'any').
    """

    name: str
    tier: Tier
    func: Callable[..., Any]
    requires: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    unique: bool = False
    foundation: bool = False
    doc: str = ""
    target_types: tuple[str, ...] = ("any",)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DecoratorSpec):
            return self.name == other.name
        return False


@dataclass
class StackSpec:
    """Specification for a registered decorator stack."""

    name: str
    stack_fn: Callable
    decorators: list[str]  # decorator names in order
    domain: str
    doc: str
    parameterized: bool


class DecoratorValidationError(TypeError):
    """Raised when decorator validation fails."""

    pass


class DecoratorRegistry:
    """
    Central registry for all Trinity decorators.

    This singleton registry:
    - Stores all decorator specifications
    - Validates decorator composition rules
    - Provides introspection APIs
    - Ensures thread-safe registration

    Usage:
        # Register a decorator
        @registry.register(
            name='native',
            tier=Tier.COMPILATION,
            foundation=True
        )
        def native(backend='cython', nogil=False):
            def decorator(target):
                ...
            return decorator

        # Query decorators
        registry.get('native')  # -> DecoratorSpec
        registry.by_tier(Tier.COMPILATION)  # -> list[DecoratorSpec]
    """

    _instance: Optional[DecoratorRegistry] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> DecoratorRegistry:
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._decorators: dict[str, DecoratorSpec] = {}
        self._by_tier: dict[Tier, list[DecoratorSpec]] = {tier: [] for tier in Tier}
        self._stacks: dict[str, StackSpec] = {}
        self._reg_lock = threading.Lock()
        self._initialized = True

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        name: str,
        tier: Tier,
        requires: tuple[str, ...] = (),
        excludes: tuple[str, ...] = (),
        unique: bool = False,
        foundation: bool = False,
        doc: str = "",
        target_types: tuple[str, ...] = ("any",),
    ) -> Callable[[F], F]:
        """
        Decorator to register a decorator function.

        Args:
            name: The decorator name.
            tier: The tier this decorator belongs to.
            requires: Required decorators that must be present.
            excludes: Decorators that cannot be present.
            unique: Whether only one instance is allowed.
            foundation: Whether this is a foundation decorator.
            doc: Documentation string.
            target_types: Valid targets ('class', 'function', 'method', 'any').

        Returns:
            The registered decorator function.

        Example:
            @registry.register(
                name='component',
                tier=Tier.ECS_CORE,
                foundation=True,
                unique=True,
                target_types=('class',)
            )
            def component(name=None):
                ...
        """

        def registrar(func: F) -> F:
            spec = DecoratorSpec(
                name=name,
                tier=tier,
                func=func,
                requires=requires,
                excludes=excludes,
                unique=unique,
                foundation=foundation,
                doc=doc or func.__doc__ or "",
                target_types=target_types,
            )

            with self._reg_lock:
                if name in self._decorators:
                    raise ValueError(f"Decorator '{name}' is already registered")

                self._decorators[name] = spec
                self._by_tier[tier].append(spec)

            return func

        return registrar

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get(self, name: str) -> Optional[DecoratorSpec]:
        """Get decorator spec by name."""
        return self._decorators.get(name)

    def by_tier(self, tier: Tier) -> list[DecoratorSpec]:
        """Get all decorators in a tier."""
        return self._by_tier.get(tier, []).copy()

    def all(self) -> dict[str, DecoratorSpec]:
        """Get all registered decorators."""
        return self._decorators.copy()

    def names(self) -> list[str]:
        """Get all registered decorator names."""
        return list(self._decorators.keys())

    def count(self) -> int:
        """Get total number of registered decorators."""
        return len(self._decorators)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_application(
        self,
        decorator_name: str,
        target: Any,
        existing_decorators: list[str],
    ) -> None:
        """
        Validate that a decorator can be applied to a target.

        Args:
            decorator_name: Name of decorator being applied.
            target: The function or class being decorated.
            existing_decorators: List of already-applied decorator names.

        Raises:
            DecoratorValidationError: If validation fails.
        """
        spec = self._decorators.get(decorator_name)
        if spec is None:
            # Allow unregistered decorators (for flexibility)
            return

        # Check uniqueness
        if spec.unique and decorator_name in existing_decorators:
            raise DecoratorValidationError(
                f"@{decorator_name} can only be applied once per target"
            )

        # Check required dependencies
        for required in spec.requires:
            if required not in existing_decorators:
                raise DecoratorValidationError(
                    f"@{decorator_name} requires @{required} to be applied first"
                )

        # Check exclusions
        for excluded in spec.excludes:
            if excluded in existing_decorators:
                raise DecoratorValidationError(
                    f"@{decorator_name} cannot be combined with @{excluded}"
                )

        # Check target type
        if "any" not in spec.target_types:
            is_class = isinstance(target, type)
            is_function = callable(target) and not is_class

            if is_class and "class" not in spec.target_types:
                raise DecoratorValidationError(
                    f"@{decorator_name} cannot be applied to classes"
                )
            if is_function and "function" not in spec.target_types:
                raise DecoratorValidationError(
                    f"@{decorator_name} cannot be applied to functions"
                )

    def validate_stack(self, target: Any) -> list[str]:
        """
        Validate the complete decorator stack on a target.

        Args:
            target: The decorated function or class.

        Returns:
            List of validation warnings (empty if all OK).

        Raises:
            DecoratorValidationError: If validation fails.
        """
        warnings_list: list[str] = []

        applied = getattr(target, "_applied_decorators", [])
        if not applied:
            return warnings_list

        # Check tier ordering (lower tiers should be applied first / be innermost)
        tiers_seen: list[tuple[str, Tier]] = []
        for dec_name in applied:
            spec = self._decorators.get(dec_name)
            if spec:
                tiers_seen.append((dec_name, spec.tier))

        # Decorators are applied bottom-up, so the list should be in ascending tier order
        for i in range(len(tiers_seen) - 1):
            name1, tier1 = tiers_seen[i]
            name2, tier2 = tiers_seen[i + 1]
            if tier2 < tier1:
                warnings_list.append(
                    f"Decorator order warning: @{name2} (Tier {tier2.value}) "
                    f"should be applied before @{name1} (Tier {tier1.value})"
                )

        return warnings_list

    def can_stack(self, decorator_a: str, decorator_b: str) -> bool:
        """
        Check if two decorators can be stacked together.

        Args:
            decorator_a: First decorator name.
            decorator_b: Second decorator name.

        Returns:
            True if decorators can be combined.
        """
        spec_a = self._decorators.get(decorator_a)
        spec_b = self._decorators.get(decorator_b)

        if spec_a is None or spec_b is None:
            return True  # Unknown decorators assumed OK

        # Check mutual exclusions
        if decorator_b in spec_a.excludes or decorator_a in spec_b.excludes:
            return False

        return True

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def info(self, name: str) -> dict[str, Any]:
        """
        Get detailed information about a decorator.

        Args:
            name: Decorator name.

        Returns:
            Dictionary with decorator details.
        """
        spec = self._decorators.get(name)
        if spec is None:
            return {"error": f"Decorator '{name}' not found"}

        return {
            "name": spec.name,
            "tier": spec.tier.name,
            "tier_value": spec.tier.value,
            "requires": spec.requires,
            "excludes": spec.excludes,
            "unique": spec.unique,
            "foundation": spec.foundation,
            "doc": spec.doc,
            "target_types": spec.target_types,
        }

    def dump(self) -> str:
        """
        Generate a formatted dump of all registered decorators.

        Returns:
            Formatted string representation.
        """
        lines = ["=" * 60]
        lines.append("DECORATOR REGISTRY")
        lines.append("=" * 60)
        lines.append(f"Total decorators: {self.count()}")
        lines.append("")

        for tier in Tier:
            specs = self._by_tier[tier]
            if specs:
                lines.append(f"--- Tier {tier.value}: {tier.name} ---")
                for spec in specs:
                    marker = "[F]" if spec.foundation else "   "
                    lines.append(f"  {marker} @{spec.name}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # =========================================================================
    # TESTING/RESET
    # =========================================================================

    # =========================================================================
    # STACK REGISTRY
    # =========================================================================

    def register_stack(self, name: str, stack_fn: Any, domain: str, doc: str = "") -> None:
        """Register a built-in stack.

        Args:
            name: Unique non-empty stack name.
            stack_fn: Callable that returns a Stack.
            domain: Domain category for the stack.
            doc: Optional documentation string.

        Raises:
            ValueError: If *name* is empty or already registered.
        """
        if not name or not name.strip():
            raise ValueError("Stack name must be a non-empty string")

        if name in self._stacks:
            raise ValueError(f"Stack '{name}' is already registered")

        from trinity.decorators.stacks import Stack

        parameterized = getattr(stack_fn, '_is_parameterized_stack', False)
        # Try to get decorator names by calling with no args if not parameterized
        decorators: list[str] = []
        if not parameterized:
            try:
                result = stack_fn()
                if isinstance(result, Stack):
                    decorators = [d.__name__ if hasattr(d, '__name__') else str(d) for d in result.decorators]
            except TypeError:
                # stack_fn requires args -- treat as parameterized
                parameterized = True
            except Exception:
                pass
        spec = StackSpec(
            name=name,
            stack_fn=stack_fn,
            decorators=decorators,
            domain=domain,
            doc=doc or getattr(stack_fn, '__doc__', '') or '',
            parameterized=parameterized,
        )
        self._stacks[name] = spec

    def get_stack(self, name: str) -> StackSpec:
        """Get a stack spec by name."""
        if name not in self._stacks:
            raise KeyError(f"Stack '{name}' not registered")
        return self._stacks[name]

    def all_stacks(self) -> list[StackSpec]:
        """Return all registered stacks."""
        return list(self._stacks.values())

    def expand_stack(self, name: str) -> list[str]:
        """Return decorator names for a stack."""
        spec = self.get_stack(name)
        return list(spec.decorators)

    # =========================================================================
    # TESTING/RESET
    # =========================================================================

    def clear(self) -> None:
        """Clear all registered decorators and stacks. Useful for testing."""
        with self._reg_lock:
            self._decorators.clear()
            self._by_tier = {tier: [] for tier in Tier}
            self._stacks.clear()


# Global singleton instance
registry = DecoratorRegistry()


# =============================================================================
# INTROSPECTION UTILITIES
# =============================================================================


@dataclass
class DecoratorInfo:
    """
    Information about decorators applied to a target.

    Returned by inspect_decorated().

    Attributes:
        decorators: List of applied decorator names in order.
        attributes: Dictionary of all decorator-set attributes.
        tier: Highest tier level of applied decorators.
        interactions: List of interaction notes.
    """

    decorators: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    tier: int = -1
    interactions: list[str] = field(default_factory=list)


def get_decorator_chain(target: Any) -> list[str]:
    """
    Get the ordered list of decorators applied to a target.

    Args:
        target: A decorated function or class.

    Returns:
        List of decorator names in application order.
    """
    return list(getattr(target, "_applied_decorators", []))


def has_decorator(target: Any, decorator_name: str) -> bool:
    """
    Check if a target has a specific decorator applied.

    Args:
        target: A decorated function or class.
        decorator_name: The decorator name to check for.

    Returns:
        True if the decorator is applied.
    """
    applied = getattr(target, "_applied_decorators", [])
    return decorator_name in applied


def inspect_decorated(target: Any) -> DecoratorInfo:
    """
    Get all decorator information for a function or class.

    This function collects all decorator-related metadata from a target,
    including applied decorators, attached attributes, and tier information.

    Args:
        target: A decorated function or class.

    Returns:
        DecoratorInfo with decorators, attributes, tier, and interactions.

    Example:
        @parallel(chunk_size=64)
        @fixed(hz=60)
        def my_system(): ...

        info = inspect_decorated(my_system)
        print(info.decorators)   # ['parallel', 'fixed']
        print(info.attributes)   # {'_parallel': True, '_parallel_chunk_size': 64, ...}
    """
    info = DecoratorInfo()

    # Get applied decorators
    info.decorators = get_decorator_chain(target)

    # Collect all decorator-related attributes
    for attr_name in dir(target):
        if attr_name.startswith("_") and not attr_name.startswith("__"):
            try:
                value = getattr(target, attr_name)
                # Skip methods and callables
                if not callable(value):
                    info.attributes[attr_name] = value
            except AttributeError:
                pass

    # Determine highest tier
    for dec_name in info.decorators:
        spec = registry.get(dec_name)
        if spec is not None:
            info.tier = max(info.tier, spec.tier.value)

    # Collect interaction notes
    for dec_name in info.decorators:
        spec = registry.get(dec_name)
        if spec:
            for excluded in spec.excludes:
                info.interactions.append(f"@{dec_name} excludes @{excluded}")
            for required in spec.requires:
                info.interactions.append(f"@{dec_name} requires @{required}")

    return info


def track_decorator(target: Any, decorator_name: str) -> None:
    """
    Track that a decorator has been applied to a target.

    This is called by decorators to record their application for
    later introspection and validation.

    Args:
        target: The decorated function or class.
        decorator_name: The name of the decorator being applied.
    """
    if not hasattr(target, "_applied_decorators"):
        target._applied_decorators = []
    target._applied_decorators.append(decorator_name)


def validate_decorator_requirements(target: Any, decorator_name: str) -> None:
    """
    Validate that required decorators are present and excluded ones are not.

    This function checks the decorator composition rules defined in the
    registry and raises TypeError if any rules are violated.

    Args:
        target: The target being decorated.
        decorator_name: The decorator being applied.

    Raises:
        TypeError: If composition rules are violated.
    """
    applied = getattr(target, "_applied_decorators", [])
    registry.validate_application(decorator_name, target, applied)
