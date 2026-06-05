"""
Terrain SDF Functions for Demoscene Rendering (T-DEMO-4.1 and T-DEMO-4.2)

This module provides procedural terrain signed distance functions optimized for
SDF-based demoscene rendering with both CPU-side evaluation and WGSL code generation.

T-DEMO-4.1: Heightmap Terrain SDF
- FBM-based height function with configurable octaves
- SDF returning signed distance to terrain surface
- Ground plane at y=0, terrain height above

T-DEMO-4.2: Ridged Noise Terrain
- Sharp valleys, smooth ridges (ridged multifractal)
- Configurable ridge sharpness and gain
- Realistic mountain terrain appearance

Both terrain types follow Trinity patterns with Mirror/Tracker integration
for dirty tracking and cache invalidation.

References:
- Inigo Quilez -- Terrain: https://iquilezles.org/articles/terrainmarching/
- Inigo Quilez -- fBM: https://iquilezles.org/articles/fbm/
- Musgrave -- Ridged Multifractal: https://www.classes.cs.uchicago.edu/archive/2015/fall/23700-1/final-project/MussavesMaxWorly.pdf
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Dict, FrozenSet, Generator, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Core classes
    "TerrainSDF",
    "HeightmapTerrainSDF",
    "RidgedTerrainSDF",
    # Configuration
    "HeightmapConfig",
    "RidgedConfig",
    # Helpers
    "Vec3",
    "TerrainMirror",
    "TerrainTracker",
    # WGSL generation
    "generate_heightmap_terrain_wgsl",
    "generate_ridged_terrain_wgsl",
    "generate_all_terrain_wgsl",
    # Noise functions
    "fbm_2d",
    "ridged_fbm_2d",
    "hash21",
    "value_noise_2d",
]

# Default FBM parameters
DEFAULT_OCTAVES: int = 6
DEFAULT_LACUNARITY: float = 2.0
DEFAULT_GAIN: float = 0.5
DEFAULT_AMPLITUDE: float = 1.0
DEFAULT_FREQUENCY: float = 1.0

# Default ridged parameters
DEFAULT_RIDGE_SHARPNESS: float = 2.0
DEFAULT_RIDGE_OFFSET: float = 1.0

# Epsilon for floating-point comparisons
EPSILON: float = 1e-8


# =============================================================================
# Vec3 Helper (Immutable 3D Vector)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vec3:
    """Immutable 3D vector for terrain evaluation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vec3":
        """Create Vec3 from tuple."""
        return cls(float(t[0]), float(t[1]), float(t[2]))

    @classmethod
    def from_scalar(cls, s: float) -> "Vec3":
        """Create Vec3 with all components equal."""
        return cls(s, s, s)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return as tuple."""
        return (self.x, self.y, self.z)

    def xz(self) -> Tuple[float, float]:
        """Return XZ components as 2D tuple (for heightmap sampling)."""
        return (self.x, self.z)

    def length(self) -> float:
        """Compute vector length."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def length_xz(self) -> float:
        """Compute XZ planar length."""
        return math.sqrt(self.x ** 2 + self.z ** 2)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def to_wgsl(self) -> str:
        """Generate WGSL vec3 literal."""
        return f"vec3<f32>({self.x}, {self.y}, {self.z})"


# =============================================================================
# Noise Functions (Python Reference Implementation)
# =============================================================================

def fract(x: float) -> float:
    """Fractional part (WGSL fract equivalent)."""
    return x - math.floor(x)


def smoothstep_quintic(t: float) -> float:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3 (C2 continuous)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


def hash21(p: Tuple[float, float]) -> float:
    """2D hash: maps (x, y) to pseudo-random float in [0, 1)."""
    qx = fract(p[0] * 0.1031)
    qy = fract(p[1] * 0.1030)
    d = qx * (qy + 33.33) + qy * (qx + 33.33)
    qx += d
    qy += d
    return fract(qx * qy)


def value_noise_2d(p: Tuple[float, float]) -> float:
    """2D value noise: maps (x, y) to smooth noise in [-1, 1]."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    fx = p[0] - ix
    fy = p[1] - iy

    ux = smoothstep_quintic(fx)
    uy = smoothstep_quintic(fy)

    # Hash at four corners, remap to [-1, 1]
    a = hash21((ix, iy)) * 2.0 - 1.0
    b = hash21((ix + 1.0, iy)) * 2.0 - 1.0
    c = hash21((ix, iy + 1.0)) * 2.0 - 1.0
    d = hash21((ix + 1.0, iy + 1.0)) * 2.0 - 1.0

    # Bilinear interpolation
    vx0 = lerp(a, b, ux)
    vx1 = lerp(c, d, ux)
    return lerp(vx0, vx1, uy)


def fbm_2d(
    p: Tuple[float, float],
    octaves: int = DEFAULT_OCTAVES,
    lacunarity: float = DEFAULT_LACUNARITY,
    gain: float = DEFAULT_GAIN,
) -> float:
    """
    2D Fractal Brownian Motion (FBM) using value noise.

    Sums multiple octaves of noise with increasing frequency and
    decreasing amplitude to create natural-looking terrain.

    Args:
        p: 2D coordinate (x, z) for terrain sampling
        octaves: Number of noise layers (4-8 typical for terrain)
        lacunarity: Frequency multiplier per octave (typically 2.0)
        gain: Amplitude multiplier per octave (typically 0.5)

    Returns:
        FBM noise value normalized to [-1, 1]
    """
    if octaves <= 0:
        return 0.0

    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * value_noise_2d((p[0] * frequency, p[1] * frequency))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < EPSILON:
        return 0.0
    return value / max_amplitude


def ridged_fbm_2d(
    p: Tuple[float, float],
    octaves: int = DEFAULT_OCTAVES,
    lacunarity: float = DEFAULT_LACUNARITY,
    gain: float = DEFAULT_GAIN,
    ridge_sharpness: float = DEFAULT_RIDGE_SHARPNESS,
    ridge_offset: float = DEFAULT_RIDGE_OFFSET,
) -> float:
    """
    2D Ridged Multifractal Noise.

    Creates sharp ridges with smooth valleys - ideal for mountain terrain.
    Uses absolute value inversion with sharpness control and gain weighting.

    Args:
        p: 2D coordinate (x, z) for terrain sampling
        octaves: Number of noise layers
        lacunarity: Frequency multiplier per octave
        gain: Amplitude multiplier per octave (also affects ridge weight)
        ridge_sharpness: Controls valley depth (higher = sharper ridges)
        ridge_offset: Base offset for ridge calculation

    Returns:
        Ridged noise value in [0, 1]
    """
    if octaves <= 0:
        return 0.0

    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    weight = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        # Sample base noise
        noise = value_noise_2d((p[0] * frequency, p[1] * frequency))

        # Ridge transformation: invert absolute value
        # This creates sharp peaks where noise crosses zero
        ridge = ridge_offset - abs(noise)

        # Apply sharpness (power function for steeper ridges)
        if ridge > 0:
            ridge = ridge ** ridge_sharpness
        else:
            ridge = 0.0

        # Weight by previous octave's value for multifractal effect
        ridge *= weight
        weight = max(0.0, min(1.0, ridge * gain))

        value += amplitude * ridge
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < EPSILON:
        return 0.0

    # Normalize to [0, 1]
    return max(0.0, min(1.0, value / max_amplitude))


# =============================================================================
# TerrainMirror - Introspection for Terrain SDFs
# =============================================================================

class TerrainMirror:
    """
    Mirror provides introspection for Terrain SDF nodes.

    Following the Trinity pattern, Mirror allows:
    - Field enumeration
    - Type inspection
    - Value access
    - Metadata retrieval
    """

    __slots__ = ("_terrain",)

    def __init__(self, terrain: "TerrainSDF") -> None:
        self._terrain = terrain

    @property
    def terrain_type(self) -> str:
        """Return the terrain type name."""
        return type(self._terrain).__name__

    @property
    def config(self) -> Any:
        """Return terrain configuration."""
        return self._terrain.config

    @property
    def fields(self) -> Dict[str, Any]:
        """Return all configuration fields as dict."""
        config = self._terrain.config
        if hasattr(config, "__dataclass_fields__"):
            return {
                name: getattr(config, name)
                for name in config.__dataclass_fields__
            }
        return {}

    @property
    def is_dirty(self) -> bool:
        """Check if terrain has been modified."""
        return self._terrain._dirty

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return terrain metadata."""
        return {
            "terrain_type": self.terrain_type,
            "is_dirty": self.is_dirty,
            "version": self._terrain._version,
        }

    def __repr__(self) -> str:
        return f"<TerrainMirror for {self.terrain_type}>"


# =============================================================================
# TerrainTracker - Dirty Tracking for Terrain SDFs
# =============================================================================

class TerrainTracker:
    """
    Tracker provides dirty tracking for Terrain SDF nodes.

    Following the Trinity pattern, Tracker enables:
    - Change detection
    - Cache invalidation
    - Versioning
    """

    __slots__ = ("_terrain",)

    def __init__(self, terrain: "TerrainSDF") -> None:
        self._terrain = terrain

    @property
    def is_dirty(self) -> bool:
        """Check if terrain is dirty."""
        return self._terrain._dirty

    @property
    def version(self) -> int:
        """Return terrain version (increments on each change)."""
        return self._terrain._version

    def mark_dirty(self) -> None:
        """Mark terrain as dirty."""
        self._terrain._dirty = True
        self._terrain._version += 1

    def clear(self) -> None:
        """Clear dirty flag."""
        self._terrain._dirty = False

    def __repr__(self) -> str:
        return f"<TerrainTracker dirty={self.is_dirty} version={self.version}>"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class HeightmapConfig:
    """Configuration for FBM heightmap terrain."""
    octaves: int = DEFAULT_OCTAVES
    lacunarity: float = DEFAULT_LACUNARITY
    gain: float = DEFAULT_GAIN
    amplitude: float = DEFAULT_AMPLITUDE
    frequency: float = DEFAULT_FREQUENCY
    ground_level: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.octaves > 16:
            raise ValueError(f"octaves must be <= 16, got {self.octaves}")
        if self.lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if self.gain <= 0 or self.gain > 1:
            raise ValueError(f"gain must be in (0, 1], got {self.gain}")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {self.amplitude}")
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")


@dataclass
class RidgedConfig:
    """Configuration for ridged multifractal terrain."""
    octaves: int = DEFAULT_OCTAVES
    lacunarity: float = DEFAULT_LACUNARITY
    gain: float = DEFAULT_GAIN
    amplitude: float = DEFAULT_AMPLITUDE
    frequency: float = DEFAULT_FREQUENCY
    ground_level: float = 0.0
    ridge_sharpness: float = DEFAULT_RIDGE_SHARPNESS
    ridge_offset: float = DEFAULT_RIDGE_OFFSET

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.octaves > 16:
            raise ValueError(f"octaves must be <= 16, got {self.octaves}")
        if self.lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if self.gain <= 0 or self.gain > 1:
            raise ValueError(f"gain must be in (0, 1], got {self.gain}")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {self.amplitude}")
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.ridge_sharpness <= 0:
            raise ValueError(f"ridge_sharpness must be > 0, got {self.ridge_sharpness}")
        if self.ridge_offset <= 0:
            raise ValueError(f"ridge_offset must be > 0, got {self.ridge_offset}")


# =============================================================================
# TerrainSDF - Abstract Base Class
# =============================================================================

class TerrainSDF(ABC):
    """
    Abstract base class for terrain signed distance functions.

    Terrain SDFs represent procedural terrain as a heightfield
    converted to a signed distance field for ray marching.

    All implementations follow Trinity patterns with Mirror/Tracker.
    """

    _instance_counter: ClassVar[int] = 0
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        "_terrain_id", "_dirty", "_version", "_mirror", "_tracker", "_config"
    )

    def __init__(self, config: Any) -> None:
        """Initialize terrain with configuration."""
        with TerrainSDF._counter_lock:
            TerrainSDF._instance_counter += 1
            self._terrain_id = TerrainSDF._instance_counter

        self._config = config
        self._dirty: bool = True
        self._version: int = 0
        self._mirror: Optional[TerrainMirror] = None
        self._tracker: Optional[TerrainTracker] = None

    @property
    def config(self) -> Any:
        """Return terrain configuration."""
        return self._config

    @property
    def mirror(self) -> TerrainMirror:
        """Get Mirror instance for introspection."""
        if self._mirror is None:
            self._mirror = TerrainMirror(self)
        return self._mirror

    @property
    def tracker(self) -> TerrainTracker:
        """Get Tracker instance for dirty tracking."""
        if self._tracker is None:
            self._tracker = TerrainTracker(self)
        return self._tracker

    @abstractmethod
    def height(self, x: float, z: float) -> float:
        """
        Compute terrain height at (x, z) position.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Terrain height above ground level
        """
        pass

    @abstractmethod
    def sdf(self, p: Vec3) -> float:
        """
        Compute signed distance to terrain surface.

        Args:
            p: 3D position to evaluate

        Returns:
            Signed distance (negative inside terrain, positive above)
        """
        pass

    def sdf_tuple(self, p: Tuple[float, float, float]) -> float:
        """Convenience method for tuple input."""
        return self.sdf(Vec3.from_tuple(p))

    @abstractmethod
    def to_wgsl(self, function_name: str = "terrain_sdf") -> str:
        """
        Generate WGSL shader code for this terrain.

        Args:
            function_name: Name for the generated WGSL function

        Returns:
            WGSL source code string
        """
        pass

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration and mark dirty."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        self.tracker.mark_dirty()


# =============================================================================
# T-DEMO-4.1: HeightmapTerrainSDF
# =============================================================================

class HeightmapTerrainSDF(TerrainSDF):
    """
    Heightmap terrain SDF using FBM noise for height generation.

    Creates natural-looking terrain with configurable detail through
    octave-based fractal Brownian motion. The SDF is computed as the
    signed distance from a point to the terrain surface.

    The terrain surface is defined as y = ground_level + height(x, z)
    where height is computed from FBM noise scaled by amplitude.

    Attributes:
        config: HeightmapConfig with terrain parameters
    """

    def __init__(self, config: Optional[HeightmapConfig] = None) -> None:
        """
        Initialize heightmap terrain.

        Args:
            config: Terrain configuration (uses defaults if None)
        """
        if config is None:
            config = HeightmapConfig()
        super().__init__(config)

    @property
    def config(self) -> HeightmapConfig:
        """Return typed configuration."""
        return self._config

    def height(self, x: float, z: float) -> float:
        """
        Compute terrain height at (x, z) using FBM.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Terrain height (before adding ground_level)
        """
        cfg = self.config

        # Apply frequency scaling
        px = x * cfg.frequency
        pz = z * cfg.frequency

        # Compute FBM noise
        noise = fbm_2d(
            (px, pz),
            octaves=cfg.octaves,
            lacunarity=cfg.lacunarity,
            gain=cfg.gain,
        )

        # FBM returns [-1, 1], remap to [0, 1] and scale by amplitude
        height = (noise * 0.5 + 0.5) * cfg.amplitude

        return height

    def sdf(self, p: Vec3) -> float:
        """
        Compute signed distance to terrain surface.

        The SDF is approximated as the vertical distance to the heightfield.
        This is exact for vertical rays but an approximation for angled rays.
        A more accurate approach would use gradient-based distance estimation.

        Args:
            p: 3D position to evaluate

        Returns:
            Signed distance: negative below terrain, positive above
        """
        terrain_height = self.config.ground_level + self.height(p.x, p.z)
        return p.y - terrain_height

    def to_wgsl(self, function_name: str = "heightmap_terrain_sdf") -> str:
        """
        Generate WGSL shader code.

        Args:
            function_name: Name for the SDF function

        Returns:
            WGSL source code
        """
        cfg = self.config
        return f"""
// Heightmap Terrain SDF (T-DEMO-4.1)
// Generated with: octaves={cfg.octaves}, lacunarity={cfg.lacunarity}, gain={cfg.gain}
// amplitude={cfg.amplitude}, frequency={cfg.frequency}, ground_level={cfg.ground_level}

fn heightmap_terrain_height(p: vec2<f32>) -> f32 {{
    let scaled_p = p * {cfg.frequency};
    var value: f32 = 0.0;
    var amplitude: f32 = 1.0;
    var frequency: f32 = 1.0;
    var max_amplitude: f32 = 0.0;

    for (var i: u32 = 0u; i < {cfg.octaves}u; i = i + 1u) {{
        value = value + amplitude * value_noise_2d(scaled_p * frequency);
        max_amplitude = max_amplitude + amplitude;
        frequency = frequency * {cfg.lacunarity};
        amplitude = amplitude * {cfg.gain};
    }}

    let noise = value / max_amplitude;
    return (noise * 0.5 + 0.5) * {cfg.amplitude};
}}

fn {function_name}(p: vec3<f32>) -> f32 {{
    let terrain_height = {cfg.ground_level} + heightmap_terrain_height(p.xz);
    return p.y - terrain_height;
}}
"""


# =============================================================================
# T-DEMO-4.2: RidgedTerrainSDF
# =============================================================================

class RidgedTerrainSDF(TerrainSDF):
    """
    Ridged multifractal terrain SDF for mountain-like terrain.

    Creates terrain with sharp ridges and smooth valleys using
    ridged multifractal noise. This is ideal for mountain ranges
    and dramatic landscapes.

    The ridged effect is created by taking the absolute value of
    noise, inverting it, and optionally applying a power function
    for sharpness control.

    Attributes:
        config: RidgedConfig with terrain parameters
    """

    def __init__(self, config: Optional[RidgedConfig] = None) -> None:
        """
        Initialize ridged terrain.

        Args:
            config: Terrain configuration (uses defaults if None)
        """
        if config is None:
            config = RidgedConfig()
        super().__init__(config)

    @property
    def config(self) -> RidgedConfig:
        """Return typed configuration."""
        return self._config

    def height(self, x: float, z: float) -> float:
        """
        Compute terrain height at (x, z) using ridged multifractal.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Terrain height (before adding ground_level)
        """
        cfg = self.config

        # Apply frequency scaling
        px = x * cfg.frequency
        pz = z * cfg.frequency

        # Compute ridged noise
        noise = ridged_fbm_2d(
            (px, pz),
            octaves=cfg.octaves,
            lacunarity=cfg.lacunarity,
            gain=cfg.gain,
            ridge_sharpness=cfg.ridge_sharpness,
            ridge_offset=cfg.ridge_offset,
        )

        # Ridged noise returns [0, 1], scale by amplitude
        return noise * cfg.amplitude

    def sdf(self, p: Vec3) -> float:
        """
        Compute signed distance to ridged terrain surface.

        Args:
            p: 3D position to evaluate

        Returns:
            Signed distance: negative below terrain, positive above
        """
        terrain_height = self.config.ground_level + self.height(p.x, p.z)
        return p.y - terrain_height

    def to_wgsl(self, function_name: str = "ridged_terrain_sdf") -> str:
        """
        Generate WGSL shader code.

        Args:
            function_name: Name for the SDF function

        Returns:
            WGSL source code
        """
        cfg = self.config
        return f"""
// Ridged Terrain SDF (T-DEMO-4.2)
// Generated with: octaves={cfg.octaves}, lacunarity={cfg.lacunarity}, gain={cfg.gain}
// amplitude={cfg.amplitude}, frequency={cfg.frequency}, ground_level={cfg.ground_level}
// ridge_sharpness={cfg.ridge_sharpness}, ridge_offset={cfg.ridge_offset}

fn ridged_terrain_height(p: vec2<f32>) -> f32 {{
    let scaled_p = p * {cfg.frequency};
    var value: f32 = 0.0;
    var amplitude: f32 = 1.0;
    var frequency: f32 = 1.0;
    var weight: f32 = 1.0;
    var max_amplitude: f32 = 0.0;

    for (var i: u32 = 0u; i < {cfg.octaves}u; i = i + 1u) {{
        let noise = value_noise_2d(scaled_p * frequency);

        // Ridge transformation
        var ridge = {cfg.ridge_offset} - abs(noise);
        if (ridge > 0.0) {{
            ridge = pow(ridge, {cfg.ridge_sharpness});
        }} else {{
            ridge = 0.0;
        }}

        // Multifractal weighting
        ridge = ridge * weight;
        weight = clamp(ridge * {cfg.gain}, 0.0, 1.0);

        value = value + amplitude * ridge;
        max_amplitude = max_amplitude + amplitude;
        frequency = frequency * {cfg.lacunarity};
        amplitude = amplitude * {cfg.gain};
    }}

    return clamp(value / max_amplitude, 0.0, 1.0) * {cfg.amplitude};
}}

fn {function_name}(p: vec3<f32>) -> f32 {{
    let terrain_height = {cfg.ground_level} + ridged_terrain_height(p.xz);
    return p.y - terrain_height;
}}
"""


# =============================================================================
# WGSL Code Generation
# =============================================================================

def generate_heightmap_terrain_wgsl() -> str:
    """Generate WGSL code for heightmap terrain SDF with default parameters."""
    terrain = HeightmapTerrainSDF()
    return terrain.to_wgsl()


def generate_ridged_terrain_wgsl() -> str:
    """Generate WGSL code for ridged terrain SDF with default parameters."""
    terrain = RidgedTerrainSDF()
    return terrain.to_wgsl()


def generate_all_terrain_wgsl() -> str:
    """Generate combined WGSL code for all terrain types."""
    return f"""
// =============================================================================
// Terrain SDF Functions (T-DEMO-4.1 and T-DEMO-4.2)
// =============================================================================

{generate_heightmap_terrain_wgsl()}

{generate_ridged_terrain_wgsl()}
"""


# =============================================================================
# Factory Functions
# =============================================================================

def create_heightmap_terrain(
    octaves: int = DEFAULT_OCTAVES,
    amplitude: float = DEFAULT_AMPLITUDE,
    frequency: float = DEFAULT_FREQUENCY,
    lacunarity: float = DEFAULT_LACUNARITY,
    gain: float = DEFAULT_GAIN,
    ground_level: float = 0.0,
) -> HeightmapTerrainSDF:
    """
    Create a heightmap terrain with custom parameters.

    Args:
        octaves: Number of FBM layers (4-8 typical)
        amplitude: Maximum terrain height
        frequency: Base noise frequency
        lacunarity: Frequency multiplier per octave
        gain: Amplitude multiplier per octave
        ground_level: Base ground height

    Returns:
        Configured HeightmapTerrainSDF instance
    """
    config = HeightmapConfig(
        octaves=octaves,
        amplitude=amplitude,
        frequency=frequency,
        lacunarity=lacunarity,
        gain=gain,
        ground_level=ground_level,
    )
    return HeightmapTerrainSDF(config)


def create_ridged_terrain(
    octaves: int = DEFAULT_OCTAVES,
    amplitude: float = DEFAULT_AMPLITUDE,
    frequency: float = DEFAULT_FREQUENCY,
    lacunarity: float = DEFAULT_LACUNARITY,
    gain: float = DEFAULT_GAIN,
    ground_level: float = 0.0,
    ridge_sharpness: float = DEFAULT_RIDGE_SHARPNESS,
    ridge_offset: float = DEFAULT_RIDGE_OFFSET,
) -> RidgedTerrainSDF:
    """
    Create a ridged terrain with custom parameters.

    Args:
        octaves: Number of noise layers
        amplitude: Maximum terrain height
        frequency: Base noise frequency
        lacunarity: Frequency multiplier per octave
        gain: Amplitude multiplier per octave
        ground_level: Base ground height
        ridge_sharpness: Controls valley sharpness (higher = sharper)
        ridge_offset: Base offset for ridge calculation

    Returns:
        Configured RidgedTerrainSDF instance
    """
    config = RidgedConfig(
        octaves=octaves,
        amplitude=amplitude,
        frequency=frequency,
        lacunarity=lacunarity,
        gain=gain,
        ground_level=ground_level,
        ridge_sharpness=ridge_sharpness,
        ridge_offset=ridge_offset,
    )
    return RidgedTerrainSDF(config)
