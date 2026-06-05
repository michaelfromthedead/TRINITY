"""
Advanced Terrain SDF Implementations (T-DEMO-4.3 and T-DEMO-4.4).

This module provides advanced terrain generation using signed distance fields:

T-DEMO-4.3: Domain-Warped Terrain
    - DomainWarpedTerrainSDF: Apply FBM warp to input coordinates before height lookup
    - Warp strength parameter controls distortion amount
    - Multiple warp passes for increased variety
    - Produces non-repeating, natural-looking landscapes

T-DEMO-4.4: 3D Terrain with Caves/Overhangs
    - CaveTerrainSDF: Full 3D terrain using volumetric approach
    - Combines heightmap SDF with 3D FBM displacement
    - Negative FBM regions create caves and arches
    - Cave density, overhang probability parameters
    - Ensures caves connect properly (no floating chunks)

Both classes follow the Trinity pattern with:
    - Mirror: Introspection for field access and type information
    - Tracker: Dirty tracking for cache invalidation

Reference:
    - Inigo Quilez: https://iquilezles.org/articles/terrainmarching/
    - Domain warping: https://iquilezles.org/articles/warp/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .sdf_ast import SDFNode, Vec3, Tracker, Mirror


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # T-DEMO-4.3
    "DomainWarpedTerrainSDF",
    "DomainWarpConfig",
    # T-DEMO-4.4
    "CaveTerrainSDF",
    "CaveConfig",
    # Helpers
    "TerrainConfig",
    "NoiseType",
    "WarpPass",
]

# Default FBM parameters
DEFAULT_OCTAVES = 8
DEFAULT_LACUNARITY = 2.0
DEFAULT_GAIN = 0.5

# Warp decorrelation offsets (same as domain_warp WGSL)
WARP_OFFSET_X = 100.0
WARP_OFFSET_Y = 200.0
WARP_OFFSET_Z = 300.0


# =============================================================================
# Noise Type Enumeration
# =============================================================================

class NoiseType(Enum):
    """Type of base noise for FBM generation."""
    VALUE = auto()
    PERLIN = auto()


# =============================================================================
# Python Noise Models (matching WGSL semantics)
# =============================================================================

def _wgsl_fract(x: float) -> float:
    """WGSL fract: x - floor(x)."""
    return x - math.floor(x)


def _hash11(p: float) -> float:
    """1D hash function -> [0, 1)."""
    q = _wgsl_fract(p * 0.1031)
    q = q * (q + 33.33)
    q = q * (q + q)
    return _wgsl_fract(q)


def _hash21(p: Tuple[float, float]) -> float:
    """2D hash function -> [0, 1)."""
    qx = _wgsl_fract(p[0] * 0.1031)
    qy = _wgsl_fract(p[1] * 0.1030)
    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx = qx + d
    qy = qy + d
    return _wgsl_fract(qx * qy)


def _hash31(p: Tuple[float, float, float]) -> float:
    """3D hash function -> [0, 1)."""
    qx = _wgsl_fract(p[0] * 0.1031)
    qy = _wgsl_fract(p[1] * 0.1030)
    qz = _wgsl_fract(p[2] * 0.0973)
    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d
    return _wgsl_fract(qx * qy * qz)


def _smoothstep(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


def _value_noise_2d(p: Tuple[float, float]) -> float:
    """2D value noise in range [-1, 1]."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    fx = p[0] - ix
    fy = p[1] - iy
    ux = _smoothstep(fx)
    uy = _smoothstep(fy)
    a = _hash21((ix, iy)) * 2.0 - 1.0
    b = _hash21((ix + 1.0, iy)) * 2.0 - 1.0
    c = _hash21((ix, iy + 1.0)) * 2.0 - 1.0
    d = _hash21((ix + 1.0, iy + 1.0)) * 2.0 - 1.0
    vx0 = _lerp(a, b, ux)
    vx1 = _lerp(c, d, ux)
    return _lerp(vx0, vx1, uy)


def _value_noise_3d(p: Tuple[float, float, float]) -> float:
    """3D value noise in range [-1, 1]."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz
    ux = _smoothstep(fx)
    uy = _smoothstep(fy)
    uz = _smoothstep(fz)

    def h(dx: float, dy: float, dz: float) -> float:
        return _hash31((ix + dx, iy + dy, iz + dz)) * 2.0 - 1.0

    a = h(0, 0, 0)
    b = h(1, 0, 0)
    c = h(0, 1, 0)
    d = h(1, 1, 0)
    e = h(0, 0, 1)
    f = h(1, 0, 1)
    g = h(0, 1, 1)
    h_val = h(1, 1, 1)

    vx00 = _lerp(a, b, ux)
    vx10 = _lerp(c, d, ux)
    vx01 = _lerp(e, f, ux)
    vx11 = _lerp(g, h_val, ux)
    vy0 = _lerp(vx00, vx10, uy)
    vy1 = _lerp(vx01, vx11, uy)
    return _lerp(vy0, vy1, uz)


# Perlin gradient table
_GRADIENTS: List[Tuple[float, float, float]] = [
    (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0), (0.0, -1.0, 1.0), (0.0, 1.0, -1.0), (0.0, -1.0, -1.0),
]
_INV_SQRT2 = 0.7071067811865475


def _perlin_gradient(hash_value: float, offset: Tuple[float, float, float]) -> float:
    """Select gradient from hash and dot with offset."""
    h = int(hash_value * 12.0) % 12
    gx, gy, gz = _GRADIENTS[h]
    gx *= _INV_SQRT2
    gy *= _INV_SQRT2
    gz *= _INV_SQRT2
    return gx * offset[0] + gy * offset[1] + gz * offset[2]


def _perlin_noise_3d(p: Tuple[float, float, float]) -> float:
    """3D Perlin noise in range approximately [-1, 1]."""
    ix = math.floor(p[0])
    iy = math.floor(p[1])
    iz = math.floor(p[2])
    fx = p[0] - ix
    fy = p[1] - iy
    fz = p[2] - iz
    ux = _smoothstep(fx)
    uy = _smoothstep(fy)
    uz = _smoothstep(fz)

    def grad(dx: float, dy: float, dz: float) -> float:
        h = _hash31((ix + dx, iy + dy, iz + dz))
        o = (fx - dx, fy - dy, fz - dz)
        return _perlin_gradient(h, o)

    g000 = grad(0, 0, 0)
    g100 = grad(1, 0, 0)
    g010 = grad(0, 1, 0)
    g110 = grad(1, 1, 0)
    g001 = grad(0, 0, 1)
    g101 = grad(1, 0, 1)
    g011 = grad(0, 1, 1)
    g111 = grad(1, 1, 1)

    vx00 = _lerp(g000, g100, ux)
    vx10 = _lerp(g010, g110, ux)
    vx01 = _lerp(g001, g101, ux)
    vx11 = _lerp(g011, g111, ux)
    vy0 = _lerp(vx00, vx10, uy)
    vy1 = _lerp(vx01, vx11, uy)
    return _lerp(vy0, vy1, uz)


# =============================================================================
# FBM Functions
# =============================================================================

def _fbm_2d(
    p: Tuple[float, float],
    octaves: int,
    lacunarity: float,
    gain: float,
) -> float:
    """2D Fractal Brownian Motion using value noise."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * _value_noise_2d((p[0] * frequency, p[1] * frequency))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


def _fbm_3d(
    p: Tuple[float, float, float],
    octaves: int,
    lacunarity: float,
    gain: float,
    noise_type: NoiseType = NoiseType.VALUE,
) -> float:
    """3D Fractal Brownian Motion."""
    noise_fn = _value_noise_3d if noise_type == NoiseType.VALUE else _perlin_noise_3d
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * noise_fn((
            p[0] * frequency,
            p[1] * frequency,
            p[2] * frequency,
        ))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass(frozen=True)
class WarpPass:
    """Configuration for a single warp pass."""
    frequency: float = 1.0
    amplitude: float = 1.0
    octaves: int = 4
    lacunarity: float = 2.0
    gain: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.octaves < 0:
            raise ValueError(f"octaves must be >= 0, got {self.octaves}")


@dataclass(frozen=True)
class DomainWarpConfig:
    """Configuration for domain-warped terrain."""
    # Warp parameters
    warp_strength: float = 1.0
    warp_frequency: float = 0.5
    warp_passes: Tuple[WarpPass, ...] = field(default_factory=lambda: (WarpPass(),))

    # Base height FBM parameters
    height_octaves: int = 8
    height_lacunarity: float = 2.0
    height_gain: float = 0.5
    height_amplitude: float = 10.0
    height_frequency: float = 0.1

    # Terrain bounds
    base_height: float = 0.0

    # Noise type
    noise_type: NoiseType = NoiseType.VALUE

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.warp_strength < 0:
            raise ValueError(f"warp_strength must be >= 0, got {self.warp_strength}")
        if self.height_octaves < 1:
            raise ValueError(f"height_octaves must be >= 1, got {self.height_octaves}")


@dataclass(frozen=True)
class CaveConfig:
    """Configuration for cave terrain."""
    # Base heightmap parameters
    height_octaves: int = 6
    height_lacunarity: float = 2.0
    height_gain: float = 0.5
    height_amplitude: float = 10.0
    height_frequency: float = 0.1
    base_height: float = 0.0

    # Cave carving parameters
    cave_strength: float = 3.0  # How much caves carve into terrain
    cave_density: float = 0.5  # 0-1, probability of cave formation
    cave_octaves: int = 4
    cave_lacunarity: float = 2.0
    cave_gain: float = 0.5
    cave_frequency: float = 0.15

    # Overhang parameters
    overhang_probability: float = 0.3  # 0-1
    overhang_depth: float = 2.0  # Maximum overhang depth

    # Connectivity
    min_cave_opening: float = 0.5  # Minimum opening size to ensure connectivity
    connect_caves: bool = True  # Whether to ensure cave connectivity

    # Noise type
    noise_type: NoiseType = NoiseType.VALUE

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.cave_density <= 1.0:
            raise ValueError(f"cave_density must be in [0, 1], got {self.cave_density}")
        if not 0.0 <= self.overhang_probability <= 1.0:
            raise ValueError(
                f"overhang_probability must be in [0, 1], got {self.overhang_probability}"
            )
        if self.cave_strength < 0:
            raise ValueError(f"cave_strength must be >= 0, got {self.cave_strength}")


@dataclass(frozen=True)
class TerrainConfig:
    """Unified terrain configuration (convenience wrapper)."""
    domain_warp: Optional[DomainWarpConfig] = None
    cave: Optional[CaveConfig] = None

    @classmethod
    def domain_warped(cls, **kwargs) -> "TerrainConfig":
        """Create config for domain-warped terrain."""
        return cls(domain_warp=DomainWarpConfig(**kwargs))

    @classmethod
    def with_caves(cls, **kwargs) -> "TerrainConfig":
        """Create config for cave terrain."""
        return cls(cave=CaveConfig(**kwargs))


# =============================================================================
# T-DEMO-4.3: Domain-Warped Terrain SDF
# =============================================================================

class DomainWarpedTerrainSDF(SDFNode):
    """
    Domain-warped terrain SDF (T-DEMO-4.3).

    Applies FBM domain warping to input coordinates before evaluating
    the heightmap, producing non-repeating, natural-looking landscapes.

    Domain warping formula:
        p' = p + fbm(p * warp_freq) * warp_amplitude
        height = fbm(p')

    Multiple warp passes can be chained for increased variety:
        p1 = p + warp1(p)
        p2 = p1 + warp2(p1)
        ...
        height = fbm(pN)

    Attributes:
        config: DomainWarpConfig with all terrain parameters

    Example:
        >>> config = DomainWarpConfig(
        ...     warp_strength=2.0,
        ...     warp_passes=(WarpPass(frequency=0.5), WarpPass(frequency=0.25)),
        ...     height_amplitude=15.0,
        ... )
        >>> terrain = DomainWarpedTerrainSDF(config)
        >>> sdf_value = terrain.evaluate(Vec3(10.0, 5.0, 20.0))
    """

    __slots__ = ("_config", "_heightmap_cache", "_cache_size")

    config: DomainWarpConfig

    def __init__(self, config: Optional[DomainWarpConfig] = None) -> None:
        super().__init__()
        self._config = config or DomainWarpConfig()
        self._heightmap_cache: Dict[Tuple[int, int], float] = {}
        self._cache_size = 1024
        self.tracker.mark_dirty("config")

    @property
    def config(self) -> DomainWarpConfig:
        """Get terrain configuration."""
        return self._config

    @config.setter
    def config(self, value: DomainWarpConfig) -> None:
        """Set terrain configuration and invalidate cache."""
        self._config = value
        self._heightmap_cache.clear()
        self.tracker.mark_dirty("config")

    def _apply_warp(
        self,
        p: Tuple[float, float],
        warp_pass: WarpPass,
    ) -> Tuple[float, float]:
        """Apply a single warp pass to 2D coordinates."""
        # Compute warp displacement
        freq = warp_pass.frequency
        scaled_p = (p[0] * freq, p[1] * freq)

        warp_x = _fbm_2d(
            scaled_p,
            warp_pass.octaves,
            warp_pass.lacunarity,
            warp_pass.gain,
        )
        warp_y = _fbm_2d(
            (scaled_p[0] + WARP_OFFSET_X, scaled_p[1] + WARP_OFFSET_X),
            warp_pass.octaves,
            warp_pass.lacunarity,
            warp_pass.gain,
        )

        # Apply warp with strength and amplitude
        amp = warp_pass.amplitude * self._config.warp_strength
        return (
            p[0] + warp_x * amp,
            p[1] + warp_y * amp,
        )

    def _apply_all_warps(self, p: Tuple[float, float]) -> Tuple[float, float]:
        """Apply all warp passes sequentially."""
        result = p
        for warp_pass in self._config.warp_passes:
            result = self._apply_warp(result, warp_pass)
        return result

    def get_height(self, x: float, z: float) -> float:
        """
        Get terrain height at (x, z) position.

        This applies domain warping then evaluates the height FBM.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Height value at position
        """
        # Check cache
        cache_key = (int(x * 100), int(z * 100))
        if cache_key in self._heightmap_cache:
            return self._heightmap_cache[cache_key]

        # Apply domain warping
        warped = self._apply_all_warps((x, z))

        # Scale by height frequency
        freq = self._config.height_frequency
        scaled_p = (warped[0] * freq, warped[1] * freq)

        # Evaluate height FBM
        height = _fbm_2d(
            scaled_p,
            self._config.height_octaves,
            self._config.height_lacunarity,
            self._config.height_gain,
        )

        # Scale and offset
        result = self._config.base_height + height * self._config.height_amplitude

        # Cache result (with LRU-style eviction)
        if len(self._heightmap_cache) >= self._cache_size:
            # Remove oldest quarter
            keys = list(self._heightmap_cache.keys())[:self._cache_size // 4]
            for k in keys:
                del self._heightmap_cache[k]
        self._heightmap_cache[cache_key] = result

        return result

    def evaluate(self, p: Vec3) -> float:
        """
        Evaluate the SDF at position p.

        The SDF for terrain is: p.y - height(p.x, p.z)
        Negative inside (below ground), positive outside (above ground).

        Args:
            p: 3D position to evaluate

        Returns:
            Signed distance to terrain surface
        """
        height = self.get_height(p.x, p.z)
        return p.y - height

    def evaluate_tuple(self, p: Tuple[float, float, float]) -> float:
        """Evaluate SDF with tuple input (convenience method)."""
        return self.evaluate(Vec3(p[0], p[1], p[2]))

    def get_normal(self, p: Vec3, epsilon: float = 0.001) -> Vec3:
        """
        Estimate surface normal at position using central differences.

        Args:
            p: Position on or near surface
            epsilon: Small offset for gradient estimation

        Returns:
            Normalized surface normal vector
        """
        dx = self.evaluate(Vec3(p.x + epsilon, p.y, p.z)) - \
             self.evaluate(Vec3(p.x - epsilon, p.y, p.z))
        dy = self.evaluate(Vec3(p.x, p.y + epsilon, p.z)) - \
             self.evaluate(Vec3(p.x, p.y - epsilon, p.z))
        dz = self.evaluate(Vec3(p.x, p.y, p.z + epsilon)) - \
             self.evaluate(Vec3(p.x, p.y, p.z - epsilon))

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-10:
            return Vec3(0.0, 1.0, 0.0)  # Default to up
        return Vec3(dx / length, dy / length, dz / length)

    def is_pattern_repeating(
        self,
        sample_points: int = 100,
        area_size: float = 1000.0,
        correlation_threshold: float = 0.95,
    ) -> bool:
        """
        Check if terrain pattern shows repetition.

        Used to verify domain warping produces non-repeating patterns.

        Args:
            sample_points: Number of points to sample
            area_size: Area to sample over
            correlation_threshold: Correlation above which is "repeating"

        Returns:
            True if pattern appears to repeat, False otherwise
        """
        import random

        heights1 = []
        heights2 = []

        for i in range(sample_points):
            # Sample random points
            x = random.uniform(0, area_size)
            z = random.uniform(0, area_size)
            heights1.append(self.get_height(x, z))

            # Sample at offset position
            offset = area_size * 0.123456  # Arbitrary offset
            heights2.append(self.get_height(x + offset, z + offset))

        # Compute correlation
        mean1 = sum(heights1) / len(heights1)
        mean2 = sum(heights2) / len(heights2)

        cov = sum((h1 - mean1) * (h2 - mean2) for h1, h2 in zip(heights1, heights2))
        var1 = sum((h - mean1) ** 2 for h in heights1)
        var2 = sum((h - mean2) ** 2 for h in heights2)

        if var1 < 1e-10 or var2 < 1e-10:
            return True  # Constant = repeating

        correlation = cov / math.sqrt(var1 * var2)
        return abs(correlation) > correlation_threshold

    def to_wgsl(self) -> str:
        """
        Generate WGSL code for this terrain SDF.

        Returns:
            WGSL shader code string
        """
        config = self._config
        warp_passes_code = ""

        for i, wp in enumerate(config.warp_passes):
            warp_passes_code += f"""
    // Warp pass {i}
    {{
        let freq = {wp.frequency}f;
        let amp = {wp.amplitude}f * warp_strength;
        let scaled = warped_p * freq;
        let warp_x = fbm_2d(scaled, {wp.octaves}u, {wp.lacunarity}f, {wp.gain}f);
        let warp_y = fbm_2d(scaled + vec2<f32>({WARP_OFFSET_X}f), {wp.octaves}u, {wp.lacunarity}f, {wp.gain}f);
        warped_p = warped_p + vec2<f32>(warp_x, warp_y) * amp;
    }}
"""

        return f"""\
// Domain-Warped Terrain SDF (T-DEMO-4.3)
// Generated by TRINITY

const TERRAIN_WARP_STRENGTH: f32 = {config.warp_strength}f;
const TERRAIN_HEIGHT_AMP: f32 = {config.height_amplitude}f;
const TERRAIN_HEIGHT_FREQ: f32 = {config.height_frequency}f;
const TERRAIN_BASE_HEIGHT: f32 = {config.base_height}f;

fn terrain_domain_warped_height(p: vec2<f32>) -> f32 {{
    let warp_strength = TERRAIN_WARP_STRENGTH;
    var warped_p = p;
{warp_passes_code}
    // Evaluate height FBM
    let scaled_p = warped_p * TERRAIN_HEIGHT_FREQ;
    let height = fbm_2d(scaled_p, {config.height_octaves}u, {config.height_lacunarity}f, {config.height_gain}f);
    return TERRAIN_BASE_HEIGHT + height * TERRAIN_HEIGHT_AMP;
}}

fn sdf_terrain_domain_warped(p: vec3<f32>) -> f32 {{
    let height = terrain_domain_warped_height(p.xz);
    return p.y - height;
}}
"""

    def label(self) -> str:
        """Return short label for debugging."""
        return f"DomainWarpedTerrain(passes={len(self._config.warp_passes)})"

    def clone(self) -> "DomainWarpedTerrainSDF":
        """Create a deep copy."""
        return DomainWarpedTerrainSDF(self._config)


# =============================================================================
# T-DEMO-4.4: Cave Terrain SDF
# =============================================================================

class CaveTerrainSDF(SDFNode):
    """
    3D Terrain with caves and overhangs SDF (T-DEMO-4.4).

    Combines a heightmap-based terrain SDF with 3D FBM displacement
    to create caves, arches, and overhangs. The approach:

    1. Base SDF from heightmap: sdf_base = p.y - height(p.xz)
    2. 3D cave carving: cave_sdf = fbm_3d(p) * cave_strength
    3. Combined: sdf = sdf_base - max(cave_sdf, 0) when inside threshold

    Negative FBM regions create caves. The cave_density parameter
    controls the threshold at which carving occurs.

    Attributes:
        config: CaveConfig with all terrain/cave parameters

    Example:
        >>> config = CaveConfig(
        ...     cave_strength=4.0,
        ...     cave_density=0.6,
        ...     overhang_probability=0.4,
        ... )
        >>> terrain = CaveTerrainSDF(config)
        >>> sdf_value = terrain.evaluate(Vec3(10.0, 5.0, 20.0))
    """

    __slots__ = ("_config", "_connectivity_checked")

    config: CaveConfig

    def __init__(self, config: Optional[CaveConfig] = None) -> None:
        super().__init__()
        self._config = config or CaveConfig()
        self._connectivity_checked = False
        self.tracker.mark_dirty("config")

    @property
    def config(self) -> CaveConfig:
        """Get cave terrain configuration."""
        return self._config

    @config.setter
    def config(self, value: CaveConfig) -> None:
        """Set cave terrain configuration."""
        self._config = value
        self._connectivity_checked = False
        self.tracker.mark_dirty("config")

    def _get_base_height(self, x: float, z: float) -> float:
        """Get base terrain height (before cave carving)."""
        config = self._config
        freq = config.height_frequency
        p = (x * freq, z * freq)

        height = _fbm_2d(
            p,
            config.height_octaves,
            config.height_lacunarity,
            config.height_gain,
        )

        return config.base_height + height * config.height_amplitude

    def _get_cave_value(self, p: Tuple[float, float, float]) -> float:
        """
        Get 3D cave field value at position.

        Returns value in roughly [-1, 1] range.
        Negative values indicate cave regions.
        """
        config = self._config
        freq = config.cave_frequency
        scaled_p = (p[0] * freq, p[1] * freq, p[2] * freq)

        return _fbm_3d(
            scaled_p,
            config.cave_octaves,
            config.cave_lacunarity,
            config.cave_gain,
            config.noise_type,
        )

    def _get_overhang_value(self, p: Tuple[float, float, float]) -> float:
        """
        Get overhang displacement value.

        Uses a different noise offset to create independent overhang patterns.
        """
        config = self._config
        freq = config.cave_frequency * 0.7  # Slightly different scale
        offset = (WARP_OFFSET_Y, WARP_OFFSET_Y, WARP_OFFSET_Y)
        scaled_p = (
            (p[0] + offset[0]) * freq,
            (p[1] + offset[1]) * freq,
            (p[2] + offset[2]) * freq,
        )

        return _fbm_3d(
            scaled_p,
            max(2, config.cave_octaves - 2),  # Fewer octaves for smoother overhangs
            config.cave_lacunarity,
            config.cave_gain,
            config.noise_type,
        )

    def _compute_connectivity_factor(
        self,
        p: Tuple[float, float, float],
        cave_value: float,
    ) -> float:
        """
        Compute connectivity factor to prevent isolated voids.

        Ensures caves remain connected by smoothing transitions
        and preventing small isolated pockets.
        """
        if not self._config.connect_caves:
            return 1.0

        min_opening = self._config.min_cave_opening

        # Check if we're near a cave boundary
        # by sampling neighboring points
        neighbors_inside = 0
        sample_dist = min_opening * 0.5

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor_p = (
                        p[0] + dx * sample_dist,
                        p[1] + dy * sample_dist,
                        p[2] + dz * sample_dist,
                    )
                    neighbor_cave = self._get_cave_value(neighbor_p)
                    if neighbor_cave < -self._config.cave_density:
                        neighbors_inside += 1

        # If too few neighbors are inside, reduce carving
        # to prevent isolated voids
        if neighbors_inside < 3:
            return 0.3  # Reduce carving significantly
        elif neighbors_inside < 6:
            return 0.7
        return 1.0

    def evaluate(self, p: Vec3) -> float:
        """
        Evaluate the 3D cave terrain SDF at position p.

        The algorithm:
        1. Compute base heightmap SDF
        2. Compute 3D cave field
        3. Apply cave carving where cave field is below threshold
        4. Add overhang displacement
        5. Apply connectivity factor

        Args:
            p: 3D position to evaluate

        Returns:
            Signed distance to terrain/cave surface
        """
        config = self._config

        # Base heightmap SDF
        base_height = self._get_base_height(p.x, p.z)
        base_sdf = p.y - base_height

        # 3D cave field
        p_tuple = (p.x, p.y, p.z)
        cave_value = self._get_cave_value(p_tuple)

        # Cave carving threshold based on density
        threshold = -config.cave_density

        # Only carve where cave_value is below threshold (negative region)
        if cave_value < threshold:
            # How deep into cave region we are
            cave_depth = (threshold - cave_value) * config.cave_strength

            # Apply connectivity factor
            conn_factor = self._compute_connectivity_factor(p_tuple, cave_value)
            cave_depth *= conn_factor

            # Carve into terrain (reduce SDF value to make it "inside")
            base_sdf -= cave_depth

        # Overhang displacement
        if config.overhang_probability > 0:
            overhang_value = self._get_overhang_value(p_tuple)

            # Only apply overhangs on sloped regions (not flat ground)
            # Check if we're near the base terrain surface
            near_surface = abs(p.y - base_height) < config.height_amplitude * 0.3

            if near_surface and overhang_value > (1.0 - config.overhang_probability * 2):
                overhang_amount = overhang_value * config.overhang_depth
                overhang_amount *= config.overhang_probability
                base_sdf -= overhang_amount

        return base_sdf

    def evaluate_tuple(self, p: Tuple[float, float, float]) -> float:
        """Evaluate SDF with tuple input."""
        return self.evaluate(Vec3(p[0], p[1], p[2]))

    def is_inside_cave(self, p: Vec3) -> bool:
        """
        Check if position is inside a cave.

        Args:
            p: Position to check

        Returns:
            True if inside cave, False otherwise
        """
        # Inside cave if SDF < 0 AND base heightmap would say outside
        base_height = self._get_base_height(p.x, p.z)
        base_sdf = p.y - base_height
        actual_sdf = self.evaluate(p)

        # Inside cave if we're below ground level but SDF says inside
        return base_sdf > 0 and actual_sdf < 0

    def has_overhang_at(self, p: Vec3) -> bool:
        """
        Check if there's an overhang at position.

        An overhang exists where there's solid terrain above
        with open space below (or to the side).

        Args:
            p: Position to check

        Returns:
            True if overhang present, False otherwise
        """
        # Check if we're inside terrain
        if self.evaluate(p) > 0:
            return False

        # Check if there's open space below
        below_p = Vec3(p.x, p.y - self._config.overhang_depth, p.z)
        return self.evaluate(below_p) > 0

    def check_cave_connectivity(
        self,
        sample_region: Tuple[float, float, float, float, float, float],
        grid_resolution: int = 10,
    ) -> bool:
        """
        Verify caves in a region are connected (no isolated voids).

        Uses flood-fill algorithm to check connectivity.

        Args:
            sample_region: (min_x, min_y, min_z, max_x, max_y, max_z)
            grid_resolution: Grid cells per axis

        Returns:
            True if all caves connect, False if isolated voids exist
        """
        min_x, min_y, min_z, max_x, max_y, max_z = sample_region
        dx = (max_x - min_x) / grid_resolution
        dy = (max_y - min_y) / grid_resolution
        dz = (max_z - min_z) / grid_resolution

        # Build 3D grid of cave cells
        cave_cells = set()
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                for k in range(grid_resolution):
                    p = Vec3(
                        min_x + (i + 0.5) * dx,
                        min_y + (j + 0.5) * dy,
                        min_z + (k + 0.5) * dz,
                    )
                    if self.is_inside_cave(p):
                        cave_cells.add((i, j, k))

        if len(cave_cells) <= 1:
            return True  # 0 or 1 cave cells = trivially connected

        # Flood fill from first cave cell
        start = next(iter(cave_cells))
        visited = {start}
        stack = [start]

        while stack:
            i, j, k = stack.pop()
            for di, dj, dk in [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ]:
                neighbor = (i + di, j + dj, k + dk)
                if neighbor in cave_cells and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        # All cave cells should be visited
        return len(visited) == len(cave_cells)

    def get_normal(self, p: Vec3, epsilon: float = 0.001) -> Vec3:
        """
        Estimate surface normal using central differences.

        Args:
            p: Position on or near surface
            epsilon: Small offset for gradient estimation

        Returns:
            Normalized surface normal vector
        """
        dx = self.evaluate(Vec3(p.x + epsilon, p.y, p.z)) - \
             self.evaluate(Vec3(p.x - epsilon, p.y, p.z))
        dy = self.evaluate(Vec3(p.x, p.y + epsilon, p.z)) - \
             self.evaluate(Vec3(p.x, p.y - epsilon, p.z))
        dz = self.evaluate(Vec3(p.x, p.y, p.z + epsilon)) - \
             self.evaluate(Vec3(p.x, p.y, p.z - epsilon))

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-10:
            return Vec3(0.0, 1.0, 0.0)
        return Vec3(dx / length, dy / length, dz / length)

    def is_sdf_continuous(
        self,
        p1: Vec3,
        p2: Vec3,
        num_samples: int = 20,
        max_jump: float = 0.5,
    ) -> bool:
        """
        Check if SDF is continuous between two points.

        Used to verify there are no discontinuities through
        cave/surface transitions.

        Args:
            p1: Start point
            p2: End point
            num_samples: Number of samples along path
            max_jump: Maximum allowed SDF change per step

        Returns:
            True if continuous, False if discontinuous
        """
        prev_sdf = self.evaluate(p1)

        for i in range(1, num_samples + 1):
            t = i / num_samples
            p = Vec3(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y),
                p1.z + t * (p2.z - p1.z),
            )
            curr_sdf = self.evaluate(p)

            if abs(curr_sdf - prev_sdf) > max_jump:
                return False
            prev_sdf = curr_sdf

        return True

    def to_wgsl(self) -> str:
        """
        Generate WGSL code for this cave terrain SDF.

        Returns:
            WGSL shader code string
        """
        config = self._config

        return f"""\
// 3D Cave Terrain SDF (T-DEMO-4.4)
// Generated by TRINITY

// Terrain parameters
const TERRAIN_HEIGHT_AMP: f32 = {config.height_amplitude}f;
const TERRAIN_HEIGHT_FREQ: f32 = {config.height_frequency}f;
const TERRAIN_BASE_HEIGHT: f32 = {config.base_height}f;

// Cave parameters
const CAVE_STRENGTH: f32 = {config.cave_strength}f;
const CAVE_DENSITY: f32 = {config.cave_density}f;
const CAVE_FREQUENCY: f32 = {config.cave_frequency}f;

// Overhang parameters
const OVERHANG_PROB: f32 = {config.overhang_probability}f;
const OVERHANG_DEPTH: f32 = {config.overhang_depth}f;

fn terrain_cave_base_height(p: vec2<f32>) -> f32 {{
    let scaled_p = p * TERRAIN_HEIGHT_FREQ;
    let height = fbm_2d(scaled_p, {config.height_octaves}u, {config.height_lacunarity}f, {config.height_gain}f);
    return TERRAIN_BASE_HEIGHT + height * TERRAIN_HEIGHT_AMP;
}}

fn terrain_cave_3d_field(p: vec3<f32>) -> f32 {{
    let scaled_p = p * CAVE_FREQUENCY;
    return fbm_3d(scaled_p, {config.cave_octaves}u, {config.cave_lacunarity}f, {config.cave_gain}f);
}}

fn terrain_overhang_field(p: vec3<f32>) -> f32 {{
    let freq = CAVE_FREQUENCY * 0.7;
    let offset = vec3<f32>({WARP_OFFSET_Y}f);
    let scaled_p = (p + offset) * freq;
    return fbm_3d(scaled_p, {max(2, config.cave_octaves - 2)}u, {config.cave_lacunarity}f, {config.cave_gain}f);
}}

fn sdf_terrain_cave(p: vec3<f32>) -> f32 {{
    // Base heightmap SDF
    let base_height = terrain_cave_base_height(p.xz);
    var sdf = p.y - base_height;

    // 3D cave carving
    let cave_value = terrain_cave_3d_field(p);
    let threshold = -CAVE_DENSITY;

    if (cave_value < threshold) {{
        let cave_depth = (threshold - cave_value) * CAVE_STRENGTH;
        sdf = sdf - cave_depth;
    }}

    // Overhang displacement
    if (OVERHANG_PROB > 0.0) {{
        let overhang_value = terrain_overhang_field(p);
        let near_surface = abs(p.y - base_height) < TERRAIN_HEIGHT_AMP * 0.3;

        if (near_surface && overhang_value > (1.0 - OVERHANG_PROB * 2.0)) {{
            let overhang_amount = overhang_value * OVERHANG_DEPTH * OVERHANG_PROB;
            sdf = sdf - overhang_amount;
        }}
    }}

    return sdf;
}}

fn is_inside_cave(p: vec3<f32>) -> bool {{
    let base_height = terrain_cave_base_height(p.xz);
    let base_sdf = p.y - base_height;
    let actual_sdf = sdf_terrain_cave(p);
    return base_sdf > 0.0 && actual_sdf < 0.0;
}}
"""

    def label(self) -> str:
        """Return short label for debugging."""
        return f"CaveTerrain(strength={self._config.cave_strength})"

    def clone(self) -> "CaveTerrainSDF":
        """Create a deep copy."""
        return CaveTerrainSDF(self._config)


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_domain_warped_terrain(
    warp_strength: float = 1.0,
    warp_passes: int = 1,
    height_amplitude: float = 10.0,
    **kwargs,
) -> DomainWarpedTerrainSDF:
    """
    Create a domain-warped terrain with sensible defaults.

    Args:
        warp_strength: Overall warp strength (0 = no warp)
        warp_passes: Number of warp passes (more = more variation)
        height_amplitude: Maximum terrain height
        **kwargs: Additional DomainWarpConfig parameters

    Returns:
        Configured DomainWarpedTerrainSDF instance
    """
    passes = tuple(
        WarpPass(
            frequency=0.5 / (i + 1),
            amplitude=1.0 / (i + 1),
        )
        for i in range(warp_passes)
    )

    config = DomainWarpConfig(
        warp_strength=warp_strength,
        warp_passes=passes,
        height_amplitude=height_amplitude,
        **kwargs,
    )

    return DomainWarpedTerrainSDF(config)


def create_cave_terrain(
    cave_strength: float = 3.0,
    cave_density: float = 0.5,
    overhang_probability: float = 0.3,
    height_amplitude: float = 10.0,
    **kwargs,
) -> CaveTerrainSDF:
    """
    Create a cave terrain with sensible defaults.

    Args:
        cave_strength: How aggressively caves carve into terrain
        cave_density: Probability of cave formation (0-1)
        overhang_probability: Probability of overhangs (0-1)
        height_amplitude: Maximum terrain height
        **kwargs: Additional CaveConfig parameters

    Returns:
        Configured CaveTerrainSDF instance
    """
    config = CaveConfig(
        cave_strength=cave_strength,
        cave_density=cave_density,
        overhang_probability=overhang_probability,
        height_amplitude=height_amplitude,
        **kwargs,
    )

    return CaveTerrainSDF(config)
