"""
Memory decorators — built from Ops.

Memory layout and lifecycle decorators that control how components
and resources are allocated, stored, and managed in memory.

Every decorator here is a named list of Steps, created by make_decorator.

Decorators:
    @pooled         - Pre-allocate and reuse memory
    @packed         - Control memory layout (AoS/SoA/hybrid)
    @aligned        - Memory alignment for SIMD/cache
    @arena          - Allocate from named arena
    @flyweight      - Shared immutable data
    @intern         - String interning
    @generations    - Generational indices for entity IDs
    @copy_on_write  - Lazy copying on mutation
    @inline_array   - Fixed-size arrays without heap
    @budget         - Memory budget tracking
    @allocator      - Memory allocation strategy
    @atomic         - Atomic operations marker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
)

from trinity.constants import (
    CACHE_LINE_BYTES,
    DEFAULT_POOL_GROW_FACTOR,
    DEFAULT_POOL_SIZE,
    MEMORY_WARN_THRESHOLD,
)
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")
ClassT = TypeVar("ClassT", bound=type)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class PoolConfig:
    """Configuration for object pooling."""

    initial_size: int = DEFAULT_POOL_SIZE
    grow_factor: float = DEFAULT_POOL_GROW_FACTOR
    max_size: Optional[int] = None


@dataclass
class PackedConfig:
    """Configuration for memory packing layout."""

    layout: Literal["aos", "soa", "hybrid"] = "soa"


@dataclass
class AlignedConfig:
    """Configuration for memory alignment."""

    bytes: int = CACHE_LINE_BYTES


@dataclass
class ArenaConfig:
    """Configuration for arena allocation."""

    name: str = "default"


@dataclass
class BudgetConfig:
    """Configuration for memory budget tracking."""

    category: str
    max_bytes: Optional[int] = None
    warn_at: float = MEMORY_WARN_THRESHOLD


@dataclass
class AllocatorConfig:
    """Configuration for memory allocator strategy."""

    type: Literal["linear", "pool", "stack", "buddy", "tlsf"]
    size: int
    thread_safe: bool = False


@dataclass
class InlineArrayConfig:
    """Configuration for inline arrays."""

    size: int


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _pooled_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(
            Op.TAG,
            {
                "key": "pool",
                "value": {
                    "initial_size": params.get("initial_size", DEFAULT_POOL_SIZE),
                    "grow_factor": params.get("grow_factor", DEFAULT_POOL_GROW_FACTOR),
                    "max_size": params.get("max_size"),
                },
            },
        ),
        Step(Op.HOOK, {"event": "on_create"}),
        Step(Op.HOOK, {"event": "on_destroy"}),
        Step(Op.REGISTER, {"registry": "PoolManager"}),
    ]


def _packed_steps(params: dict[str, Any]) -> list[Step]:
    layout = params.get("layout", "soa")
    return [
        Step(Op.TAG, {"key": "memory", "value": {"layout": layout}}),
    ]


def _aligned_steps(params: dict[str, Any]) -> list[Step]:
    bytes_val = params.get("bytes", CACHE_LINE_BYTES)
    return [
        Step(Op.TAG, {"key": "memory", "value": {"alignment": bytes_val}}),
    ]


def _arena_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "memory", "value": {"allocator": "arena"}}),
        Step(Op.HOOK, {"event": "on_create"}),
    ]


def _flyweight_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "memory", "value": {"shared": True}}),
        Step(Op.INTERCEPT, {"get": "lookup"}),
        Step(Op.REGISTER, {"registry": "FlyweightCache"}),
    ]


def _intern_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "memory", "value": {"interned": True}}),
        Step(Op.INTERCEPT, {"get": "intern_lookup"}),
    ]


def _generations_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "memory", "value": {"generational": True}}),
        Step(Op.TRACK, {}),
    ]


def _copy_on_write_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "memory", "value": {"cow": True}}),
        Step(Op.INTERCEPT, {"set": "copy_then_write"}),
    ]


def _inline_array_steps(params: dict[str, Any]) -> list[Step]:
    size = params.get("size", 0)
    return [
        Step(Op.TAG, {"key": "memory", "value": {"inline": True, "size": size}}),
    ]


def _budget_steps(params: dict[str, Any]) -> list[Step]:
    category = params.get("category", "")
    max_bytes = params.get("max_bytes")
    return [
        Step(
            Op.TAG,
            {"key": "resource", "value": {"budget": category, "max_bytes": max_bytes}},
        ),
        Step(Op.VALIDATE, {"constraint": "budget_limit"}),
    ]


def _allocator_steps(params: dict[str, Any]) -> list[Step]:
    alloc_type = params.get("type", "linear")
    return [
        Step(Op.TAG, {"key": "memory", "value": {"allocator": alloc_type}}),
    ]


def _atomic_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.INTERCEPT, {"set": "atomic", "get": "atomic"}),
        Step(Op.TAG, {"key": "thread_safe", "value": True}),
    ]


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_aligned(bytes: int = CACHE_LINE_BYTES, **_: Any) -> None:
    if bytes <= 0 or (bytes & (bytes - 1)) != 0:
        raise ValueError(f"@aligned bytes must be a positive power of 2, got {bytes}")


def _validate_inline_array(size: int = 0, **_: Any) -> None:
    if size <= 0:
        raise ValueError(f"@inline_array size must be positive, got {size}")


def _validate_budget(
    category: str = "",
    warn_at: float = MEMORY_WARN_THRESHOLD,
    **_: Any,
) -> None:
    if warn_at < 0.0 or warn_at > 1.0:
        raise ValueError(f"@budget warn_at must be 0.0-1.0, got {warn_at}")


def _validate_allocator(size: int = 0, **_: Any) -> None:
    if size <= 0:
        raise ValueError(f"@allocator size must be positive, got {size}")


# =============================================================================
# AFTER-STEPS (domain behavior)
# =============================================================================


def _after_pooled(target: Any, params: dict[str, Any]) -> Any:
    initial_size = params.get("initial_size", DEFAULT_POOL_SIZE)
    grow_factor = params.get("grow_factor", DEFAULT_POOL_GROW_FACTOR)
    max_size = params.get("max_size")

    target._pooled = True
    target._pool_config = PoolConfig(
        initial_size=initial_size,
        grow_factor=grow_factor,
        max_size=max_size,
    )

    if not hasattr(target, "release"):

        def release(self: Any) -> None:
            """Return this instance to the pool for reuse."""
            pool = getattr(type(self), "_pool", None)
            if pool is not None:
                pool.release(self)

        target.release = release  # type: ignore

    return None


def _after_packed(target: Any, params: dict[str, Any]) -> Any:
    layout = params.get("layout", "soa")
    target._packed = True
    target._packed_layout = layout
    target._packed_config = PackedConfig(layout=layout)
    return None


def _after_aligned(target: Any, params: dict[str, Any]) -> Any:
    bytes_val = params.get("bytes", CACHE_LINE_BYTES)
    target._aligned = True
    target._aligned_bytes = bytes_val
    target._aligned_config = AlignedConfig(bytes=bytes_val)
    return None


def _after_arena(target: Any, params: dict[str, Any]) -> Any:
    name = params.get("name", "default")
    target._arena = True
    target._arena_name = name
    target._arena_config = ArenaConfig(name=name)
    return None


def _after_flyweight(target: Any, params: dict[str, Any]) -> Any:
    target._flyweight = True
    target._flyweight_registry: dict[int, Any] = {}
    target._flyweight_next_id = 0

    original_init = getattr(target, "__init__", None)
    has_custom_init = original_init is not None and original_init is not object.__init__

    def flyweight_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if has_custom_init:
            original_init(self, *args, **kwargs)
        cls_type = type(self)
        flyweight_id = cls_type._flyweight_next_id
        cls_type._flyweight_next_id += 1
        self._flyweight_id = flyweight_id
        cls_type._flyweight_registry[flyweight_id] = self

    target.__init__ = flyweight_init  # type: ignore

    @classmethod
    def get_by_id(cls_inner: type, flyweight_id: int) -> Any:
        return cls_inner._flyweight_registry.get(flyweight_id)

    target.get_by_id = get_by_id  # type: ignore

    def unregister(self: Any) -> None:
        cls_type = type(self)
        flyweight_id = getattr(self, "_flyweight_id", None)
        if flyweight_id is not None and flyweight_id in cls_type._flyweight_registry:
            del cls_type._flyweight_registry[flyweight_id]

    target.unregister = unregister  # type: ignore
    return None


def _after_intern(target: Any, params: dict[str, Any]) -> Any:
    target._intern = True
    target._intern_table: dict[str, str] = {}

    @classmethod
    def intern_string(cls_inner: type, s: str) -> str:
        if s not in cls_inner._intern_table:
            cls_inner._intern_table[s] = s
        return cls_inner._intern_table[s]

    target.intern_string = intern_string  # type: ignore
    return None


def _after_generations(target: Any, params: dict[str, Any]) -> Any:
    target._generations = True
    target._generation_counters: list[int] = []

    def is_generation_valid(self: Any, index: int, generation: int) -> bool:
        counters = type(self)._generation_counters
        if index < 0 or index >= len(counters):
            return False
        return counters[index] == generation

    target.is_generation_valid = is_generation_valid
    return None


def _after_copy_on_write(target: Any, params: dict[str, Any]) -> Any:
    target._copy_on_write = True

    original_setattr = target.__setattr__ if hasattr(target, "__setattr__") else None

    def cow_setattr(self: Any, name: str, value: Any) -> None:
        if getattr(self, "_cow_shared", False) and not name.startswith("_cow"):
            self._cow_shared = False
            source = getattr(self, "_cow_source", None)
            if source is not None:
                for attr_name in dir(source):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(source, attr_name)
                            if not callable(attr_value):
                                object.__setattr__(self, attr_name, attr_value)
                        except AttributeError:
                            pass
                self._cow_source = None

        if original_setattr:
            original_setattr(self, name, value)
        else:
            object.__setattr__(self, name, value)

    target.__setattr__ = cow_setattr  # type: ignore

    def cow_clone(self: Any) -> Any:
        clone = object.__new__(type(self))
        clone._cow_shared = True
        clone._cow_source = self
        return clone

    target.cow_clone = cow_clone
    return None


def _after_inline_array(target: Any, params: dict[str, Any]) -> Any:
    size = params.get("size", 0)
    target._inline_array = True
    target._inline_array_size = size
    target._inline_array_config = InlineArrayConfig(size=size)
    return None


def _after_budget(target: Any, params: dict[str, Any]) -> Any:
    category = params.get("category", "")
    max_bytes = params.get("max_bytes")
    warn_at = params.get("warn_at", MEMORY_WARN_THRESHOLD)
    target._budget = True
    target._budget_category = category
    target._budget_max_bytes = max_bytes
    target._budget_warn_at = warn_at
    target._budget_config = BudgetConfig(
        category=category,
        max_bytes=max_bytes,
        warn_at=warn_at,
    )
    return None


def _after_allocator(target: Any, params: dict[str, Any]) -> Any:
    alloc_type = params.get("type", "linear")
    size = params.get("size", 0)
    thread_safe = params.get("thread_safe", False)
    target._allocator = True
    target._allocator_type = alloc_type
    target._allocator_size = size
    target._allocator_thread_safe = thread_safe
    target._allocator_config = AllocatorConfig(
        type=alloc_type,
        size=size,
        thread_safe=thread_safe,
    )
    return None


def _after_atomic(target: Any, params: dict[str, Any]) -> Any:
    import threading

    target._atomic = True
    target._atomic_lock = threading.RLock()

    def atomic_load(self: Any) -> Any:
        with type(self)._atomic_lock:
            return getattr(self, "value", None)

    def atomic_store(self: Any, value: Any) -> None:
        with type(self)._atomic_lock:
            self.value = value

    def atomic_exchange(self: Any, value: Any) -> Any:
        with type(self)._atomic_lock:
            old = getattr(self, "value", None)
            self.value = value
            return old

    def compare_exchange(self: Any, expected: Any, desired: Any) -> bool:
        with type(self)._atomic_lock:
            current = getattr(self, "value", None)
            if current == expected:
                self.value = desired
                return True
            return False

    def fetch_add(self: Any, delta: int) -> int:
        with type(self)._atomic_lock:
            old = getattr(self, "value", 0)
            self.value = old + delta
            return old

    def fetch_sub(self: Any, delta: int) -> int:
        with type(self)._atomic_lock:
            old = getattr(self, "value", 0)
            self.value = old - delta
            return old

    target.atomic_load = atomic_load
    target.atomic_store = atomic_store
    target.atomic_exchange = atomic_exchange
    target.compare_exchange = compare_exchange
    target.fetch_add = fetch_add
    target.fetch_sub = fetch_sub
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


pooled = make_decorator(
    name="pooled",
    steps=_pooled_steps,
    doc="Pre-allocate and reuse memory for component instances.",
    after_steps=_after_pooled,
)

packed = make_decorator(
    name="packed",
    steps=_packed_steps,
    doc="Control memory layout for cache-efficient iteration.",
    after_steps=_after_packed,
)

aligned = make_decorator(
    name="aligned",
    steps=_aligned_steps,
    doc="Specify memory alignment for SIMD and cache efficiency.",
    validate=_validate_aligned,
    after_steps=_after_aligned,
)

arena = make_decorator(
    name="arena",
    steps=_arena_steps,
    doc="Allocate from a named memory arena for bulk deallocation.",
    after_steps=_after_arena,
)

flyweight = make_decorator(
    name="flyweight",
    steps=_flyweight_steps,
    doc="Enable flyweight pattern for shared immutable data.",
    after_steps=_after_flyweight,
)

intern = make_decorator(
    name="intern",
    steps=_intern_steps,
    doc="Enable string interning for memory-efficient string storage.",
    after_steps=_after_intern,
)

generations = make_decorator(
    name="generations",
    steps=_generations_steps,
    doc="Enable generational indices for safe entity ID handling.",
    after_steps=_after_generations,
)

copy_on_write = make_decorator(
    name="copy_on_write",
    steps=_copy_on_write_steps,
    doc="Enable copy-on-write semantics for efficient cloning.",
    after_steps=_after_copy_on_write,
)

inline_array = make_decorator(
    name="inline_array",
    steps=_inline_array_steps,
    doc="Specify fixed-size inline arrays without heap allocation.",
    validate=_validate_inline_array,
    after_steps=_after_inline_array,
)

budget = make_decorator(
    name="budget",
    steps=_budget_steps,
    doc="Track memory usage against a budget for certification compliance.",
    validate=_validate_budget,
    after_steps=_after_budget,
)

allocator = make_decorator(
    name="allocator",
    steps=_allocator_steps,
    doc="Specify memory allocation strategy for custom allocators.",
    validate=_validate_allocator,
    after_steps=_after_allocator,
)

atomic = make_decorator(
    name="atomic",
    steps=_atomic_steps,
    doc="Mark data structure as requiring atomic operations.",
    after_steps=_after_atomic,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, dict[str, Any]]] = [
    ("pooled", pooled, {"unique": True, "requires": ("component",)}),
    ("packed", packed, {"unique": True}),
    ("aligned", aligned, {"unique": True}),
    ("arena", arena, {"unique": True}),
    ("flyweight", flyweight, {"unique": True}),
    ("intern", intern, {"unique": True}),
    ("generations", generations, {"unique": True}),
    ("copy_on_write", copy_on_write, {"unique": True}),
    ("inline_array", inline_array, {"unique": True}),
    ("budget", budget, {}),  # Can have multiple budgets
    ("allocator", allocator, {"unique": True}),
    ("atomic", atomic, {"unique": True}),
]

for _name, _func, _extra in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.MEMORY,
            func=_func,
            unique=_extra.get("unique", False),
            requires=_extra.get("requires", ()),
            doc=getattr(_func, "__doc__", ""),
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.MEMORY].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration dataclasses
    "PoolConfig",
    "PackedConfig",
    "AlignedConfig",
    "ArenaConfig",
    "BudgetConfig",
    "AllocatorConfig",
    "InlineArrayConfig",
    # Decorators
    "pooled",
    "packed",
    "aligned",
    "arena",
    "flyweight",
    "intern",
    "generations",
    "copy_on_write",
    "inline_array",
    "budget",
    "allocator",
    "atomic",
]
