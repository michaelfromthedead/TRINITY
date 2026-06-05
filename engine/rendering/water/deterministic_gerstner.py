"""Deterministic Gerstner wave simulation using Fixed32 arithmetic.

This module provides bit-exact Gerstner wave simulation using Q16.16 fixed-point
arithmetic for lockstep networking and replay verification.

Task: T-CC-2.2 - Apply Fixed32 to water simulation (S12) Gerstner parameters.

Features:
- All wave parameters stored as Fixed32
- Deterministic sin/cos via Taylor series with Fixed32
- Time accumulator in Fixed32 for frame-independent animation
- Two runs produce bit-identical wave surfaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from trinity.constants import FIXED32_SCALE, FIXED32_SHIFT
from trinity.decorators import component
from trinity.types import Fixed32, PCG64


# =============================================================================
# FIXED-POINT TRIGONOMETRY (Taylor Series with Range Reduction)
# =============================================================================

# Precomputed constants for Fixed32 trig (Q16.16 format)
_PI_RAW = 205887  # pi * 65536 = 3.14159265... * 65536
_TWO_PI_RAW = 411775  # 2 * pi * 65536
_HALF_PI_RAW = 102944  # pi/2 * 65536
_QUARTER_PI_RAW = 51472  # pi/4 * 65536

# Fixed32 constants
FIXED32_ZERO = Fixed32.from_raw(0)
FIXED32_ONE = Fixed32.from_raw(FIXED32_SCALE)
FIXED32_HALF = Fixed32.from_raw(FIXED32_SCALE // 2)
FIXED32_PI = Fixed32.from_raw(_PI_RAW)
FIXED32_TWO_PI = Fixed32.from_raw(_TWO_PI_RAW)
FIXED32_HALF_PI = Fixed32.from_raw(_HALF_PI_RAW)


def _normalize_angle(angle: Fixed32) -> Fixed32:
    """Normalize angle to [-pi, pi] range using Fixed32 modular arithmetic."""
    raw = angle.raw

    # Use integer modular arithmetic for determinism
    two_pi = _TWO_PI_RAW

    # Normalize to [0, 2*pi)
    raw = raw % two_pi
    if raw < 0:
        raw += two_pi

    # Shift to [-pi, pi]
    if raw > _PI_RAW:
        raw -= two_pi

    return Fixed32.from_raw(raw)


def _sin_taylor(x_raw: int) -> int:
    """
    Compute sine using 9th-order Taylor series for x in [-pi/2, pi/2].

    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9!

    Higher order and better coefficients for improved accuracy.
    """
    # x^2
    x2_raw = (x_raw * x_raw) >> FIXED32_SHIFT
    # x^3
    x3_raw = (x2_raw * x_raw) >> FIXED32_SHIFT
    # x^5
    x5_raw = (x3_raw * x2_raw) >> FIXED32_SHIFT
    # x^7
    x7_raw = (x5_raw * x2_raw) >> FIXED32_SHIFT
    # x^9
    x9_raw = (x7_raw * x2_raw) >> FIXED32_SHIFT

    # More accurate coefficients (rounded nearest)
    # 1/6 * 65536 = 10922.67 -> 10923
    # 1/120 * 65536 = 546.13 -> 546
    # 1/5040 * 65536 = 13.00 -> 13
    # 1/362880 * 65536 = 0.18 -> 0 (negligible but helps slightly)
    term1 = x_raw  # x
    term2 = (x3_raw * 10923) >> FIXED32_SHIFT  # x^3/6
    term3 = (x5_raw * 546) >> FIXED32_SHIFT  # x^5/120
    term4 = (x7_raw * 13) >> FIXED32_SHIFT  # x^7/5040

    return term1 - term2 + term3 - term4


def _cos_taylor(x_raw: int) -> int:
    """
    Compute cosine using 8th-order Taylor series for x in [-pi/2, pi/2].

    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!
    """
    # x^2
    x2_raw = (x_raw * x_raw) >> FIXED32_SHIFT
    # x^4
    x4_raw = (x2_raw * x2_raw) >> FIXED32_SHIFT
    # x^6
    x6_raw = (x4_raw * x2_raw) >> FIXED32_SHIFT
    # x^8
    x8_raw = (x6_raw * x2_raw) >> FIXED32_SHIFT

    # Coefficients:
    # 1/2 * 65536 = 32768
    # 1/24 * 65536 = 2730.67 -> 2731
    # 1/720 * 65536 = 91.02 -> 91
    # 1/40320 * 65536 = 1.63 -> 2
    term0 = FIXED32_SCALE  # 1
    term1 = x2_raw >> 1  # x^2/2
    term2 = (x4_raw * 2731) >> FIXED32_SHIFT  # x^4/24
    term3 = (x6_raw * 91) >> FIXED32_SHIFT  # x^6/720
    term4 = (x8_raw * 2) >> FIXED32_SHIFT  # x^8/40320

    return term0 - term1 + term2 - term3 + term4


def fixed32_sin(angle: Fixed32) -> Fixed32:
    """
    Compute sine using Taylor series with quadrant range reduction.

    Uses the identities:
    - sin(x) = sin(pi - x) for x in [pi/2, pi]
    - sin(x) = -sin(-x) (odd function)

    Reduces angle to [-pi/2, pi/2] where Taylor series converges well.
    Determinism: Bit-exact across all platforms.
    """
    # Normalize to [-pi, pi]
    x = _normalize_angle(angle)
    x_raw = x.raw

    # Reduce to [-pi/2, pi/2] using symmetry
    if x_raw > _HALF_PI_RAW:
        # x in (pi/2, pi]: sin(x) = sin(pi - x)
        x_raw = _PI_RAW - x_raw
    elif x_raw < -_HALF_PI_RAW:
        # x in [-pi, -pi/2): sin(x) = sin(-pi - x) = -sin(pi + x)
        # Since sin is odd and sin(x + pi) = -sin(x)
        # sin(x) for x < -pi/2: x is in third quadrant
        # sin(x) = -sin(x + pi) and x + pi is in first quadrant
        # Equivalently: sin(x) = sin(-pi - x) where -pi - x is in [0, pi/2]
        x_raw = -_PI_RAW - x_raw

    result = _sin_taylor(x_raw)
    return Fixed32.from_raw(result)


def fixed32_cos(angle: Fixed32) -> Fixed32:
    """
    Compute cosine using Taylor series with quadrant range reduction.

    Uses the identities:
    - cos(x + pi) = -cos(x)
    - cos(x) = cos(-x)
    - cos(pi/2 - x) = sin(x)

    Reduces angle to [-pi/2, pi/2] where Taylor series converges well.
    Determinism: Bit-exact across all platforms.
    """
    # Normalize to [-pi, pi]
    x = _normalize_angle(angle)
    x_raw = x.raw

    # cos is even: cos(-x) = cos(x)
    if x_raw < 0:
        x_raw = -x_raw

    # Now x_raw is in [0, pi]
    # Reduce to [0, pi/2] using symmetry
    # cos(x) = -cos(pi - x) for x in [pi/2, pi]
    negate = False

    if x_raw > _HALF_PI_RAW:
        x_raw = _PI_RAW - x_raw
        negate = True

    result = _cos_taylor(x_raw)
    if negate:
        result = -result

    return Fixed32.from_raw(result)


def fixed32_sincos(angle: Fixed32) -> Tuple[Fixed32, Fixed32]:
    """Compute both sin and cos efficiently."""
    return (fixed32_sin(angle), fixed32_cos(angle))


# =============================================================================
# FIXED32 VECTOR TYPES
# =============================================================================


@dataclass(frozen=True, slots=True)
class Fixed32Vec2:
    """2D vector with Fixed32 components for deterministic math."""

    x: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    y: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)

    @classmethod
    def from_floats(cls, x: float, y: float) -> "Fixed32Vec2":
        """Create vector from float components."""
        return cls(Fixed32(x), Fixed32(y))

    @classmethod
    def from_raw(cls, x_raw: int, y_raw: int) -> "Fixed32Vec2":
        """Create vector from raw Fixed32 values."""
        return cls(Fixed32.from_raw(x_raw), Fixed32.from_raw(y_raw))

    def dot(self, other: "Fixed32Vec2") -> Fixed32:
        """Dot product of two vectors."""
        return self.x * other.x + self.y * other.y

    def length_squared(self) -> Fixed32:
        """Squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y

    def __add__(self, other: "Fixed32Vec2") -> "Fixed32Vec2":
        return Fixed32Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Fixed32Vec2") -> "Fixed32Vec2":
        return Fixed32Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Fixed32) -> "Fixed32Vec2":
        return Fixed32Vec2(self.x * scalar, self.y * scalar)

    def __neg__(self) -> "Fixed32Vec2":
        return Fixed32Vec2(-self.x, -self.y)

    def as_tuple(self) -> Tuple[float, float]:
        """Convert to float tuple for rendering."""
        return (self.x.as_float, self.y.as_float)


@dataclass(frozen=True, slots=True)
class Fixed32Vec3:
    """3D vector with Fixed32 components for deterministic math."""

    x: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    y: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    z: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)

    @classmethod
    def from_floats(cls, x: float, y: float, z: float) -> "Fixed32Vec3":
        """Create vector from float components."""
        return cls(Fixed32(x), Fixed32(y), Fixed32(z))

    @classmethod
    def from_raw(cls, x_raw: int, y_raw: int, z_raw: int) -> "Fixed32Vec3":
        """Create vector from raw Fixed32 values."""
        return cls(
            Fixed32.from_raw(x_raw),
            Fixed32.from_raw(y_raw),
            Fixed32.from_raw(z_raw),
        )

    def dot(self, other: "Fixed32Vec3") -> Fixed32:
        """Dot product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other: "Fixed32Vec3") -> "Fixed32Vec3":
        return Fixed32Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Fixed32Vec3") -> "Fixed32Vec3":
        return Fixed32Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Fixed32) -> "Fixed32Vec3":
        return Fixed32Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __neg__(self) -> "Fixed32Vec3":
        return Fixed32Vec3(-self.x, -self.y, -self.z)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to float tuple for rendering."""
        return (self.x.as_float, self.y.as_float, self.z.as_float)


# =============================================================================
# GERSTNER WAVE PARAMETERS
# =============================================================================


@dataclass(slots=True)
class Fixed32WaveParams:
    """
    Gerstner wave parameters in Fixed32 for deterministic simulation.

    A Gerstner wave is defined by:
    - amplitude (A): Wave height from trough to crest / 2
    - frequency (omega): Angular frequency = 2*pi / wavelength
    - phase (phi): Initial phase offset
    - direction (D): Normalized wave direction vector
    - steepness (Q): Controls wave sharpness (0 = sinusoidal, 1 = breaking)

    Gerstner wave equation:
        x' = x - Q*A*D.x * sin(D.dot(x,z)*omega - t*omega + phi)
        y  = A * cos(D.dot(x,z)*omega - t*omega + phi)
        z' = z - Q*A*D.z * sin(D.dot(x,z)*omega - t*omega + phi)
    """

    amplitude: Fixed32 = field(default_factory=lambda: Fixed32(0.5))
    frequency: Fixed32 = field(default_factory=lambda: Fixed32(1.0))
    phase: Fixed32 = field(default_factory=lambda: FIXED32_ZERO)
    direction: Fixed32Vec2 = field(
        default_factory=lambda: Fixed32Vec2.from_floats(1.0, 0.0)
    )
    steepness: Fixed32 = field(default_factory=lambda: Fixed32(0.5))

    @classmethod
    def from_floats(
        cls,
        amplitude: float = 0.5,
        wavelength: float = 10.0,
        phase: float = 0.0,
        direction_x: float = 1.0,
        direction_z: float = 0.0,
        steepness: float = 0.5,
    ) -> "Fixed32WaveParams":
        """
        Create wave parameters from float values.

        Args:
            amplitude: Wave height (half peak-to-trough distance).
            wavelength: Distance between wave crests.
            phase: Initial phase offset in radians.
            direction_x: X component of wave direction.
            direction_z: Z component of wave direction.
            steepness: Wave steepness factor (0-1).

        Returns:
            Fixed32WaveParams with converted values.
        """
        import math

        # Compute angular frequency: omega = 2*pi / wavelength
        omega = 2.0 * math.pi / wavelength

        # Normalize direction
        dir_len = math.sqrt(direction_x * direction_x + direction_z * direction_z)
        if dir_len > 0.0001:
            direction_x /= dir_len
            direction_z /= dir_len

        return cls(
            amplitude=Fixed32(amplitude),
            frequency=Fixed32(omega),
            phase=Fixed32(phase),
            direction=Fixed32Vec2.from_floats(direction_x, direction_z),
            steepness=Fixed32(max(0.0, min(1.0, steepness))),
        )

    @classmethod
    def from_random(cls, rng: PCG64, max_amplitude: float = 1.0) -> "Fixed32WaveParams":
        """
        Generate random wave parameters using deterministic RNG.

        Args:
            rng: PCG64 random generator for deterministic results.
            max_amplitude: Maximum wave amplitude.

        Returns:
            Randomly generated wave parameters.
        """
        import math

        # Random amplitude in [0.1, max_amplitude]
        amplitude = 0.1 + rng.next_float() * (max_amplitude - 0.1)

        # Random wavelength in [5, 50]
        wavelength = 5.0 + rng.next_float() * 45.0

        # Random phase in [0, 2*pi]
        phase = rng.next_float() * 2.0 * math.pi

        # Random direction
        angle = rng.next_float() * 2.0 * math.pi
        direction_x = math.cos(angle)
        direction_z = math.sin(angle)

        # Random steepness in [0.2, 0.8]
        steepness = 0.2 + rng.next_float() * 0.6

        return cls.from_floats(
            amplitude=amplitude,
            wavelength=wavelength,
            phase=phase,
            direction_x=direction_x,
            direction_z=direction_z,
            steepness=steepness,
        )

    def get_checksum(self) -> int:
        """Compute deterministic checksum for verification."""
        h = 17
        h = h * 31 + self.amplitude.raw
        h = h * 31 + self.frequency.raw
        h = h * 31 + self.phase.raw
        h = h * 31 + self.direction.x.raw
        h = h * 31 + self.direction.y.raw
        h = h * 31 + self.steepness.raw
        return h & 0xFFFFFFFF


# =============================================================================
# GERSTNER WAVE COMPUTATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class GerstnerWaveResult:
    """Result of Gerstner wave computation at a point."""

    displacement: Fixed32Vec3  # World-space displacement
    height: Fixed32  # Y displacement only (shortcut)
    phase_angle: Fixed32  # Current phase for this point


def compute_gerstner_wave(
    params: Fixed32WaveParams,
    position: Fixed32Vec2,
    time: Fixed32,
) -> GerstnerWaveResult:
    """
    Compute Gerstner wave displacement at a point.

    Args:
        params: Wave parameters.
        position: World XZ position to sample.
        time: Current simulation time.

    Returns:
        GerstnerWaveResult with displacement and height.
    """
    # Phase angle: D.dot(P) * omega - t * omega + phi
    dot_product = params.direction.dot(position)
    phase = dot_product * params.frequency - time * params.frequency + params.phase

    # Compute sin/cos for displacement
    sin_phase, cos_phase = fixed32_sincos(phase)

    # Height displacement: y = A * cos(phase)
    height = params.amplitude * cos_phase

    # Horizontal displacement: -Q * A * D * sin(phase)
    q_a = params.steepness * params.amplitude
    horizontal_factor = q_a * sin_phase

    dx = -params.direction.x * horizontal_factor
    dz = -params.direction.y * horizontal_factor  # direction.y is Z

    displacement = Fixed32Vec3(dx, height, dz)

    return GerstnerWaveResult(
        displacement=displacement,
        height=height,
        phase_angle=phase,
    )


# =============================================================================
# DETERMINISTIC GERSTNER WAVE SIMULATOR
# =============================================================================


@component
class DeterministicGerstnerWave:
    """
    Deterministic multi-wave Gerstner ocean simulation.

    Computes superposition of multiple Gerstner waves using Fixed32
    arithmetic for bit-exact results across platforms.

    Example:
        sim = DeterministicGerstnerWave(seed=42, wave_count=4)
        sim.initialize()

        # Update per frame
        sim.advance_time(Fixed32(0.016))  # 16ms

        # Sample wave height
        pos = Fixed32Vec2.from_floats(10.0, 20.0)
        height = sim.sample_height(pos)
        displacement = sim.sample_displacement(pos)

    Attributes:
        wave_count: Number of wave components.
        seed: RNG seed for reproducible wave generation.
    """

    _component_name: str = "DeterministicGerstnerWave"

    def __init__(
        self,
        seed: int = 0,
        wave_count: int = 4,
        max_amplitude: float = 1.0,
        custom_waves: Optional[List[Fixed32WaveParams]] = None,
    ) -> None:
        """
        Initialize Gerstner wave simulator.

        Args:
            seed: RNG seed for wave generation.
            wave_count: Number of waves to generate (ignored if custom_waves).
            max_amplitude: Maximum amplitude for random waves.
            custom_waves: Optional explicit wave parameters.
        """
        self._seed = seed
        self._wave_count = wave_count
        self._max_amplitude = max_amplitude
        self._custom_waves = custom_waves

        self._waves: List[Fixed32WaveParams] = []
        self._time: Fixed32 = FIXED32_ZERO
        self._initialized = False
        self._tick_count = 0

    @property
    def seed(self) -> int:
        """Get RNG seed."""
        return self._seed

    @property
    def wave_count(self) -> int:
        """Get number of wave components."""
        return len(self._waves)

    @property
    def waves(self) -> List[Fixed32WaveParams]:
        """Get wave parameters (copy)."""
        return list(self._waves)

    @property
    def current_time(self) -> Fixed32:
        """Get current simulation time."""
        return self._time

    @property
    def tick_count(self) -> int:
        """Get number of time advances."""
        return self._tick_count

    @property
    def is_initialized(self) -> bool:
        """Check if simulator has been initialized."""
        return self._initialized

    def initialize(self) -> None:
        """
        Initialize wave simulation.

        Generates wave parameters from seed or uses custom waves.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("DeterministicGerstnerWave already initialized")

        if self._custom_waves is not None:
            self._waves = list(self._custom_waves)
        else:
            # Generate waves from seed
            rng = PCG64(self._seed)
            self._waves = [
                Fixed32WaveParams.from_random(rng, self._max_amplitude)
                for _ in range(self._wave_count)
            ]

        self._time = FIXED32_ZERO
        self._tick_count = 0
        self._initialized = True

    def advance_time(self, delta: Fixed32) -> None:
        """
        Advance simulation time.

        Args:
            delta: Time step in Fixed32 (typically frame time).

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeterministicGerstnerWave not initialized")

        self._time = self._time + delta
        self._tick_count += 1

    def set_time(self, time: Fixed32) -> None:
        """
        Set absolute simulation time.

        Args:
            time: New simulation time.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeterministicGerstnerWave not initialized")

        self._time = time

    def sample_height(self, position: Fixed32Vec2) -> Fixed32:
        """
        Sample wave height at a position.

        Args:
            position: World XZ position.

        Returns:
            Total wave height from all wave components.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeterministicGerstnerWave not initialized")

        total_height = FIXED32_ZERO
        for wave in self._waves:
            result = compute_gerstner_wave(wave, position, self._time)
            total_height = total_height + result.height

        return total_height

    def sample_displacement(self, position: Fixed32Vec2) -> Fixed32Vec3:
        """
        Sample full 3D displacement at a position.

        Args:
            position: World XZ position.

        Returns:
            Total displacement vector from all wave components.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeterministicGerstnerWave not initialized")

        total = Fixed32Vec3(FIXED32_ZERO, FIXED32_ZERO, FIXED32_ZERO)
        for wave in self._waves:
            result = compute_gerstner_wave(wave, position, self._time)
            total = total + result.displacement

        return total

    def sample_height_grid(
        self,
        origin: Fixed32Vec2,
        step: Fixed32,
        width: int,
        height: int,
    ) -> List[List[Fixed32]]:
        """
        Sample wave heights on a grid.

        Args:
            origin: Grid origin (corner) in world space.
            step: Grid cell size.
            width: Number of columns.
            height: Number of rows.

        Returns:
            2D list of heights [row][col].

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError("DeterministicGerstnerWave not initialized")

        grid: List[List[Fixed32]] = []
        for row in range(height):
            grid_row: List[Fixed32] = []
            for col in range(width):
                pos = Fixed32Vec2(
                    origin.x + step * col,
                    origin.y + step * row,
                )
                grid_row.append(self.sample_height(pos))
            grid.append(grid_row)

        return grid

    def get_state_checksum(self) -> int:
        """
        Compute checksum of current simulation state.

        Used for replay verification and desynch detection.

        Returns:
            32-bit checksum of simulation state.
        """
        h = 17
        h = h * 31 + self._seed
        h = h * 31 + len(self._waves)
        h = h * 31 + self._time.raw
        h = h * 31 + self._tick_count

        for wave in self._waves:
            h = h * 31 + wave.get_checksum()
            h = h & 0xFFFFFFFF

        return h & 0xFFFFFFFF

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._time = FIXED32_ZERO
        self._tick_count = 0

    def destroy(self) -> None:
        """Release simulator resources."""
        self._waves.clear()
        self._time = FIXED32_ZERO
        self._tick_count = 0
        self._initialized = False


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "FIXED32_ZERO",
    "FIXED32_ONE",
    "FIXED32_HALF",
    "FIXED32_PI",
    "FIXED32_TWO_PI",
    "FIXED32_HALF_PI",
    # Trigonometry
    "fixed32_sin",
    "fixed32_cos",
    "fixed32_sincos",
    # Vector types
    "Fixed32Vec2",
    "Fixed32Vec3",
    # Wave parameters
    "Fixed32WaveParams",
    "GerstnerWaveResult",
    # Computation
    "compute_gerstner_wave",
    # Simulator
    "DeterministicGerstnerWave",
]
