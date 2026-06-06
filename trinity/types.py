"""
Type definitions and configuration classes for the Trinity Pattern.

These types define the configuration structures that decorators set
and metaclasses read during class creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Generic, Optional, TypeVar

from trinity.constants import (
    FIXED16_SCALE,
    FIXED16_SHIFT,
    FIXED32_SCALE,
    FIXED32_SHIFT,
    TYPES_POOL_INITIAL_SIZE,
    TYPES_POOL_MAX_SIZE,
    DEFAULT_POOL_GROW_FACTOR,
    DEFAULT_MAX_INSTANCES,
    PARALLEL_BATCH_SIZE,
    PARALLEL_MIN_ENTITIES,
)

T = TypeVar("T")


# =============================================================================
# FIXED-POINT TYPES - Full arithmetic implementation for deterministic math
# =============================================================================

# Re-export constants for backward compatibility
__all__ = [
    "FIXED16_SHIFT",
    "FIXED16_SCALE",
    "FIXED32_SHIFT",
    "FIXED32_SCALE",
    "Fixed16",
    "Fixed32",
    "SystemPhase",
    "PoolConfig",
    "NetworkConfig",
    "SerializationConfig",
    "BudgetConfig",
    "ParallelConfig",
    "ThrottleConfig",
    "ProfileConfig",
    "CachePolicy",
    "ValidationRule",
    "FieldConfig",
    "SIMULATION_SAFE_TYPES",
    "REQUIRES_DESCRIPTOR_TYPES",
    "QualityTier",
    "Tier",
    "DecoratorSpec",
]


class Fixed16:
    """
    Q8.8 fixed-point number for deterministic math.

    Uses 16 bits total: 8 bits for integer part, 8 bits for fractional part.
    Range: -128.0 to 127.99609375 with precision of 1/256 (~0.0039).

    All arithmetic operations are deterministic across platforms.
    """

    __slots__ = ("_value",)

    def __init__(self, value: float | int = 0):
        if isinstance(value, Fixed16):
            self._value = value._value
        elif isinstance(value, int):
            self._value = value << FIXED16_SHIFT
        else:
            self._value = int(value * FIXED16_SCALE)

    @classmethod
    def from_raw(cls, raw_value: int) -> "Fixed16":
        """Create Fixed16 from raw internal representation."""
        result = cls.__new__(cls)
        result._value = raw_value
        return result

    @property
    def as_float(self) -> float:
        """Convert to floating point (may lose precision info)."""
        return self._value / FIXED16_SCALE

    @property
    def as_int(self) -> int:
        """Get integer part (truncates toward zero)."""
        if self._value >= 0:
            return self._value >> FIXED16_SHIFT
        else:
            return -(-self._value >> FIXED16_SHIFT)

    @property
    def raw(self) -> int:
        """Get raw internal representation."""
        return self._value

    def __add__(self, other: "Fixed16 | int | float") -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, Fixed16):
            result._value = self._value + other._value
        elif isinstance(other, int):
            result._value = self._value + (other << FIXED16_SHIFT)
        else:
            result._value = self._value + int(other * FIXED16_SCALE)
        return result

    def __radd__(self, other: int | float) -> "Fixed16":
        return self.__add__(other)

    def __sub__(self, other: "Fixed16 | int | float") -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, Fixed16):
            result._value = self._value - other._value
        elif isinstance(other, int):
            result._value = self._value - (other << FIXED16_SHIFT)
        else:
            result._value = self._value - int(other * FIXED16_SCALE)
        return result

    def __rsub__(self, other: int | float) -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, int):
            result._value = (other << FIXED16_SHIFT) - self._value
        else:
            result._value = int(other * FIXED16_SCALE) - self._value
        return result

    def __mul__(self, other: "Fixed16 | int | float") -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, Fixed16):
            # Multiply raw values then shift back to maintain scale
            result._value = (self._value * other._value) >> FIXED16_SHIFT
        elif isinstance(other, int):
            result._value = self._value * other
        else:
            result._value = int(self._value * other)
        return result

    def __rmul__(self, other: int | float) -> "Fixed16":
        return self.__mul__(other)

    def __truediv__(self, other: "Fixed16 | int | float") -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, Fixed16):
            if other._value == 0:
                raise ZeroDivisionError("Fixed16 division by zero")
            # Shift numerator up before division to maintain precision
            result._value = (self._value << FIXED16_SHIFT) // other._value
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Fixed16 division by zero")
            result._value = self._value // other
        else:
            if other == 0.0:
                raise ZeroDivisionError("Fixed16 division by zero")
            result._value = int(self._value / other)
        return result

    def __rtruediv__(self, other: int | float) -> "Fixed16":
        if self._value == 0:
            raise ZeroDivisionError("Fixed16 division by zero")
        result = Fixed16.__new__(Fixed16)
        if isinstance(other, int):
            result._value = ((other << FIXED16_SHIFT) << FIXED16_SHIFT) // self._value
        else:
            result._value = int((other * FIXED16_SCALE * FIXED16_SCALE) / self._value)
        return result

    def __floordiv__(self, other: "Fixed16 | int | float") -> "Fixed16":
        """Floor division - result is still Fixed16 but truncated to integer part."""
        result = self.__truediv__(other)
        # Truncate to integer part
        if result._value >= 0:
            result._value = (result._value >> FIXED16_SHIFT) << FIXED16_SHIFT
        else:
            result._value = -((-result._value >> FIXED16_SHIFT) << FIXED16_SHIFT)
        return result

    def __mod__(self, other: "Fixed16 | int | float") -> "Fixed16":
        """Modulo operation."""
        if isinstance(other, Fixed16):
            other_raw = other._value
        elif isinstance(other, int):
            other_raw = other << FIXED16_SHIFT
        else:
            other_raw = int(other * FIXED16_SCALE)

        if other_raw == 0:
            raise ZeroDivisionError("Fixed16 modulo by zero")

        result = Fixed16.__new__(Fixed16)
        result._value = self._value % other_raw
        return result

    def __neg__(self) -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        result._value = -self._value
        return result

    def __pos__(self) -> "Fixed16":
        return self

    def __abs__(self) -> "Fixed16":
        result = Fixed16.__new__(Fixed16)
        result._value = abs(self._value)
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fixed16):
            return self._value == other._value
        elif isinstance(other, (int, float)):
            if isinstance(other, int):
                return self._value == (other << FIXED16_SHIFT)
            return self._value == int(other * FIXED16_SCALE)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: "Fixed16 | int | float") -> bool:
        if isinstance(other, Fixed16):
            return self._value < other._value
        elif isinstance(other, int):
            return self._value < (other << FIXED16_SHIFT)
        return self._value < int(other * FIXED16_SCALE)

    def __le__(self, other: "Fixed16 | int | float") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Fixed16 | int | float") -> bool:
        return not self <= other

    def __ge__(self, other: "Fixed16 | int | float") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"Fixed16({self.as_float:.4f})"

    def __str__(self) -> str:
        return f"{self.as_float:.4f}"

    def __bool__(self) -> bool:
        return self._value != 0

    def __int__(self) -> int:
        return self.as_int

    def __float__(self) -> float:
        return self.as_float


class Fixed32:
    """
    Q16.16 fixed-point number for deterministic math.

    Uses 32 bits total: 16 bits for integer part, 16 bits for fractional part.
    Range: -32768.0 to 32767.999984741 with precision of 1/65536 (~0.000015).

    All arithmetic operations are deterministic across platforms.
    """

    __slots__ = ("_value",)

    def __init__(self, value: float | int = 0):
        if isinstance(value, Fixed32):
            self._value = value._value
        elif isinstance(value, Fixed16):
            # Convert Fixed16 to Fixed32 (shift up by difference)
            self._value = value._value << (FIXED32_SHIFT - FIXED16_SHIFT)
        elif isinstance(value, int):
            self._value = value << FIXED32_SHIFT
        else:
            self._value = int(value * FIXED32_SCALE)

    @classmethod
    def from_raw(cls, raw_value: int) -> "Fixed32":
        """Create Fixed32 from raw internal representation."""
        result = cls.__new__(cls)
        result._value = raw_value
        return result

    @classmethod
    def from_fixed16(cls, f16: Fixed16) -> "Fixed32":
        """Convert Fixed16 to Fixed32 with higher precision."""
        result = cls.__new__(cls)
        result._value = f16._value << (FIXED32_SHIFT - FIXED16_SHIFT)
        return result

    @property
    def as_float(self) -> float:
        """Convert to floating point (may lose precision info)."""
        return self._value / FIXED32_SCALE

    @property
    def as_int(self) -> int:
        """Get integer part (truncates toward zero)."""
        if self._value >= 0:
            return self._value >> FIXED32_SHIFT
        else:
            return -(-self._value >> FIXED32_SHIFT)

    @property
    def raw(self) -> int:
        """Get raw internal representation."""
        return self._value

    def to_fixed16(self) -> Fixed16:
        """Convert to Fixed16 (loses precision)."""
        return Fixed16.from_raw(self._value >> (FIXED32_SHIFT - FIXED16_SHIFT))

    def __add__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, Fixed32):
            result._value = self._value + other._value
        elif isinstance(other, Fixed16):
            result._value = self._value + (other._value << (FIXED32_SHIFT - FIXED16_SHIFT))
        elif isinstance(other, int):
            result._value = self._value + (other << FIXED32_SHIFT)
        else:
            result._value = self._value + int(other * FIXED32_SCALE)
        return result

    def __radd__(self, other: int | float) -> "Fixed32":
        return self.__add__(other)

    def __sub__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, Fixed32):
            result._value = self._value - other._value
        elif isinstance(other, Fixed16):
            result._value = self._value - (other._value << (FIXED32_SHIFT - FIXED16_SHIFT))
        elif isinstance(other, int):
            result._value = self._value - (other << FIXED32_SHIFT)
        else:
            result._value = self._value - int(other * FIXED32_SCALE)
        return result

    def __rsub__(self, other: int | float) -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, int):
            result._value = (other << FIXED32_SHIFT) - self._value
        else:
            result._value = int(other * FIXED32_SCALE) - self._value
        return result

    def __mul__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, Fixed32):
            # Multiply raw values then shift back to maintain scale
            result._value = (self._value * other._value) >> FIXED32_SHIFT
        elif isinstance(other, Fixed16):
            # Convert Fixed16 scale to Fixed32 scale during multiplication
            result._value = (self._value * other._value) >> FIXED16_SHIFT
        elif isinstance(other, int):
            result._value = self._value * other
        else:
            result._value = int(self._value * other)
        return result

    def __rmul__(self, other: int | float) -> "Fixed32":
        return self.__mul__(other)

    def __truediv__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, Fixed32):
            if other._value == 0:
                raise ZeroDivisionError("Fixed32 division by zero")
            # Shift numerator up before division to maintain precision
            result._value = (self._value << FIXED32_SHIFT) // other._value
        elif isinstance(other, Fixed16):
            if other._value == 0:
                raise ZeroDivisionError("Fixed32 division by zero")
            # Account for Fixed16's different scale
            result._value = (self._value << FIXED16_SHIFT) // other._value
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Fixed32 division by zero")
            result._value = self._value // other
        else:
            if other == 0.0:
                raise ZeroDivisionError("Fixed32 division by zero")
            result._value = int(self._value / other)
        return result

    def __rtruediv__(self, other: int | float) -> "Fixed32":
        if self._value == 0:
            raise ZeroDivisionError("Fixed32 division by zero")
        result = Fixed32.__new__(Fixed32)
        if isinstance(other, int):
            result._value = ((other << FIXED32_SHIFT) << FIXED32_SHIFT) // self._value
        else:
            result._value = int((other * FIXED32_SCALE * FIXED32_SCALE) / self._value)
        return result

    def __floordiv__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        """Floor division - result is still Fixed32 but truncated to integer part."""
        result = self.__truediv__(other)
        # Truncate to integer part
        if result._value >= 0:
            result._value = (result._value >> FIXED32_SHIFT) << FIXED32_SHIFT
        else:
            result._value = -((-result._value >> FIXED32_SHIFT) << FIXED32_SHIFT)
        return result

    def __mod__(self, other: "Fixed32 | Fixed16 | int | float") -> "Fixed32":
        """Modulo operation."""
        if isinstance(other, Fixed32):
            other_raw = other._value
        elif isinstance(other, Fixed16):
            other_raw = other._value << (FIXED32_SHIFT - FIXED16_SHIFT)
        elif isinstance(other, int):
            other_raw = other << FIXED32_SHIFT
        else:
            other_raw = int(other * FIXED32_SCALE)

        if other_raw == 0:
            raise ZeroDivisionError("Fixed32 modulo by zero")

        result = Fixed32.__new__(Fixed32)
        result._value = self._value % other_raw
        return result

    def __neg__(self) -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        result._value = -self._value
        return result

    def __pos__(self) -> "Fixed32":
        return self

    def __abs__(self) -> "Fixed32":
        result = Fixed32.__new__(Fixed32)
        result._value = abs(self._value)
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fixed32):
            return self._value == other._value
        elif isinstance(other, Fixed16):
            return self._value == (other._value << (FIXED32_SHIFT - FIXED16_SHIFT))
        elif isinstance(other, (int, float)):
            if isinstance(other, int):
                return self._value == (other << FIXED32_SHIFT)
            return self._value == int(other * FIXED32_SCALE)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: "Fixed32 | Fixed16 | int | float") -> bool:
        if isinstance(other, Fixed32):
            return self._value < other._value
        elif isinstance(other, Fixed16):
            return self._value < (other._value << (FIXED32_SHIFT - FIXED16_SHIFT))
        elif isinstance(other, int):
            return self._value < (other << FIXED32_SHIFT)
        return self._value < int(other * FIXED32_SCALE)

    def __le__(self, other: "Fixed32 | Fixed16 | int | float") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Fixed32 | Fixed16 | int | float") -> bool:
        return not self <= other

    def __ge__(self, other: "Fixed32 | Fixed16 | int | float") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"Fixed32({self.as_float:.6f})"

    def __str__(self) -> str:
        return f"{self.as_float:.6f}"

    def __bool__(self) -> bool:
        return self._value != 0

    def __int__(self) -> int:
        return self.as_int

    def __float__(self) -> float:
        return self.as_float


# =============================================================================
# DETERMINISTIC RNG (PCG)
# =============================================================================


class PCG64:
    """
    Permuted Congruential Generator (PCG) for deterministic random numbers.

    PCG-XSH-RR 64/32 variant: 64-bit state, 32-bit output.
    Same seed always produces same sequence. Suitable for lockstep simulation.
    """

    __slots__ = ("_state", "_increment")

    MULTIPLIER: int = 6364136223846793005
    MASK64: int = 0xFFFFFFFFFFFFFFFF
    MASK32: int = 0xFFFFFFFF

    def __init__(self, seed: int = 0, stream: int = 1):
        """
        Initialize PCG64 with seed and stream.

        Args:
            seed: Initial seed value (any 64-bit integer).
            stream: Stream/sequence selector. Different streams produce
                    independent sequences. Must be convertible to odd number.
        """
        self._increment = ((stream << 1) | 1) & self.MASK64
        self._state = 0
        self._advance()
        self._state = (self._state + seed) & self.MASK64
        self._advance()

    @classmethod
    def from_seeds(cls, *seeds: int) -> "PCG64":
        """
        Create RNG from multiple seeds via seed chaining.

        Combines seeds deterministically for reproducible seeding.
        """
        combined = 0
        for i, seed in enumerate(seeds):
            combined ^= (seed * (2654435761 + i)) & cls.MASK64
            combined = ((combined << 31) | (combined >> 33)) & cls.MASK64
        return cls(combined)

    def _advance(self) -> None:
        """Advance internal state by one step."""
        self._state = (self._state * self.MULTIPLIER + self._increment) & self.MASK64

    def _output(self) -> int:
        """Produce 32-bit output from current state using XSH-RR permutation."""
        xorshifted = (((self._state >> 18) ^ self._state) >> 27) & self.MASK32
        rot = (self._state >> 59) & 0x1F
        return ((xorshifted >> rot) | (xorshifted << (32 - rot))) & self.MASK32

    def next_u32(self) -> int:
        """Generate next 32-bit unsigned integer."""
        result = self._output()
        self._advance()
        return result

    def next_u64(self) -> int:
        """Generate next 64-bit unsigned integer."""
        high = self.next_u32()
        low = self.next_u32()
        return (high << 32) | low

    def next_float(self) -> float:
        """Generate next float in [0.0, 1.0)."""
        return self.next_u32() / 4294967296.0

    def next_float64(self) -> float:
        """Generate next double-precision float in [0.0, 1.0)."""
        return self.next_u64() / 18446744073709551616.0

    def next_int(self, low: int, high: int) -> int:
        """
        Generate integer in [low, high] inclusive.

        Uses rejection sampling for uniform distribution.
        """
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        if low == high:
            return low

        range_size = high - low + 1
        threshold = (self.MASK32 + 1 - range_size) % range_size

        while True:
            r = self.next_u32()
            if r >= threshold:
                return low + (r % range_size)

    def next_bool(self) -> bool:
        """Generate random boolean."""
        return (self.next_u32() & 1) == 1

    def next_fixed16(self) -> Fixed16:
        """Generate random Fixed16 in [0, 1)."""
        raw = self.next_u32() & 0xFF
        return Fixed16.from_raw(raw)

    def next_fixed32(self) -> Fixed32:
        """Generate random Fixed32 in [0, 1)."""
        raw = self.next_u32() & 0xFFFF
        return Fixed32.from_raw(raw)

    def shuffle(self, items: list) -> None:
        """Fisher-Yates shuffle in place."""
        for i in range(len(items) - 1, 0, -1):
            j = self.next_int(0, i)
            items[i], items[j] = items[j], items[i]

    def choice(self, items: list):
        """Select random item from non-empty list."""
        if not items:
            raise ValueError("Cannot choose from empty list")
        return items[self.next_int(0, len(items) - 1)]

    def fork(self, stream_id: int = 0) -> "PCG64":
        """
        Create independent child RNG for parallel streams.

        The child RNG is seeded from current state + stream_id,
        producing a deterministic but independent sequence.
        """
        return PCG64(self._state ^ (stream_id * 2685821657736338717), stream_id + 1)

    @property
    def state(self) -> tuple[int, int]:
        """Get current state for serialization."""
        return (self._state, self._increment)

    @classmethod
    def from_state(cls, state: tuple[int, int]) -> "PCG64":
        """Restore RNG from saved state."""
        rng = cls.__new__(cls)
        rng._state = state[0]
        rng._increment = state[1]
        return rng

    def jump(self, delta: int) -> None:
        """Advance state by delta steps (for parallel partitioning)."""
        acc_mult = 1
        acc_plus = 0
        cur_mult = self.MULTIPLIER
        cur_plus = self._increment

        while delta > 0:
            if delta & 1:
                acc_mult = (acc_mult * cur_mult) & self.MASK64
                acc_plus = (acc_plus * cur_mult + cur_plus) & self.MASK64
            cur_plus = ((cur_mult + 1) * cur_plus) & self.MASK64
            cur_mult = (cur_mult * cur_mult) & self.MASK64
            delta >>= 1

        self._state = (acc_mult * self._state + acc_plus) & self.MASK64

    def __repr__(self) -> str:
        return f"PCG64(state=0x{self._state:016x}, inc=0x{self._increment:016x})"


# =============================================================================
# PHASE ENUM
# =============================================================================


class SystemPhase(IntEnum):
    """
    13-phase execution schedule for deterministic simulation.

    Phases execute in strict order, with each phase completing before
    the next begins. This ensures reproducible tick execution.
    """

    PRE_INPUT = 0       # Poll input devices, prepare input state
    INPUT = 1           # Process input events, update input commands
    POST_INPUT = 2      # React to input (UI, menus)

    PRE_PHYSICS = 3     # Prepare physics state (apply forces, constraints)
    PHYSICS = 4         # Physics simulation step
    POST_PHYSICS = 5    # React to physics (collision response, triggers)

    PRE_UPDATE = 6      # Prepare game state update
    UPDATE = 7          # Main game logic (AI, gameplay)
    POST_UPDATE = 8     # React to game state changes (events, observers)

    PRE_RENDER = 9      # Cull, sort, prepare render state
    RENDER = 10         # Submit render commands
    POST_RENDER = 11    # Post-process, present, debug overlays

    LATE = 12           # Late-frame cleanup, network sync, profiling


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class PoolConfig:
    """Configuration for object pooling."""

    initial_size: int = TYPES_POOL_INITIAL_SIZE
    max_size: int = TYPES_POOL_MAX_SIZE
    grow_factor: float = DEFAULT_POOL_GROW_FACTOR


@dataclass
class NetworkConfig:
    """Configuration for network replication."""

    authority: str = "server"  # "server", "client", "owner"
    interpolated: bool = False
    priority: int = 1
    update_frequency: int = 0  # 0 = every tick


@dataclass
class SerializationConfig:
    """Configuration for serialization."""

    format: str = "binary"  # "binary", "json"
    version: int = 1
    include_defaults: bool = False


@dataclass
class BudgetConfig:
    """Configuration for performance budgets."""

    max_instances: int = DEFAULT_MAX_INSTANCES
    max_memory_bytes: int = 0  # 0 = unlimited
    priority: int = 1


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    batch_size: int = PARALLEL_BATCH_SIZE
    min_entities: int = PARALLEL_MIN_ENTITIES  # Don't parallelize below this


@dataclass
class ThrottleConfig:
    """Configuration for system throttling."""

    max_frequency: float = 0  # Hz, 0 = every tick
    skip_if_budget_exceeded: bool = False


@dataclass
class ProfileConfig:
    """Configuration for profiling."""

    enabled: bool = True
    track_memory: bool = False
    track_cache_misses: bool = False


@dataclass
class CachePolicy:
    """Asset caching policy."""

    max_memory_bytes: int = 0
    ttl_seconds: float = 0  # 0 = forever
    preload: bool = False


@dataclass
class ValidationRule:
    """A field validation rule."""

    validator: Callable[[Any], bool]
    error_message: str


@dataclass
class FieldConfig:
    """Complete configuration for a component field."""

    name: str
    field_type: type
    default: Any = None
    default_factory: Optional[Callable[[], Any]] = None
    validation_rules: list[ValidationRule] = field(default_factory=list)
    network_config: Optional[NetworkConfig] = None
    serialization_config: Optional[SerializationConfig] = None
    track_changes: bool = False
    transient: bool = False  # Don't serialize/replicate


# =============================================================================
# COMPONENT FIELD TYPES (for simulation boundary enforcement)
# =============================================================================

# Types allowed in simulation components
SIMULATION_SAFE_TYPES: frozenset[type] = frozenset(
    {
        bool,
        int,
        Fixed16,
        Fixed32,
        str,  # Immutable, so OK
        bytes,  # Immutable
    }
)

# Types that require special handling
REQUIRES_DESCRIPTOR_TYPES: frozenset[type] = frozenset(
    {
        list,
        dict,
        set,
    }
)


# =============================================================================
# QUALITY TIER SYSTEM (T-CC-0.1)
# =============================================================================


class QualityTier(IntEnum):
    """
    Quality tier for rendering feature selection.

    Each tier defines a feature set appropriate for different hardware
    capabilities. The engine selects a tier at startup based on GPU
    capabilities and can dynamically adjust based on frame budget.
    """

    LOW = 0  # Mobile/GLES 3.1: Forward+, 256MB budget, 8 lights, no RT
    MEDIUM = 1  # Integrated GPU: Clustered forward, 512MB, 64 lights, VSM shadows
    HIGH = 2  # Discrete GPU: Full deferred, 1GB, 256 lights, RT shadows
    ULTRA = 3  # High-end: All features, unlimited budget, RT GI

    @property
    def score(self) -> float:
        """Numeric score for tier ordering (0.0 to 1.0)."""
        return self.value / 3.0

    @classmethod
    def from_score(cls, score: float) -> "QualityTier":
        """Select tier based on capability score."""
        if score < 0.25:
            return cls.LOW
        elif score < 0.5:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.ULTRA

    def meets_requirement(self, required: "QualityTier") -> bool:
        """Check if this tier meets or exceeds a requirement."""
        return self.value >= required.value


# =============================================================================
# TIER SYSTEM
# =============================================================================


class Tier(IntEnum):
    """Decorator tier levels for ordering validation."""

    FOUNDATION = 0  # @component, @system, @resource, @event
    DETERMINISM = 1  # @simulation, @presentation
    IDENTITY = 2  # @registered, @singleton, @pooled
    SERIALIZATION = 3  # @serializable, @json_schema, @binary_format
    NETWORKING = 4  # @networked, @server_authoritative, @client_predicted
    PERSISTENCE = 5  # @persistent, @cached, @transient
    LIFECYCLE = 6  # @on_create, @on_destroy, @pooled_lifecycle
    BEHAVIOR = 7  # @track_changes, @observable, @validated
    OPTIMIZATION = 8  # @cached_properties, @lazy_init, @precomputed


@dataclass
class DecoratorSpec:
    """Specification for a registered decorator."""

    name: str
    tier: Tier
    requires: tuple[str, ...] = ()
    requires_before: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    unique: bool = False
    foundation: bool = False

    def __hash__(self) -> int:
        return hash(self.name)
