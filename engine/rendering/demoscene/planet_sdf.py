"""
PlanetSDF - Spherical Terrain SDF Generator (T-DEMO-4.9).

Implements procedural planetary terrain using signed distance fields:
- Spherical noise displacement for topography
- Ocean level clipping
- Continent masks with land/water distribution
- Mountain regions with variable amplitude
- Crater impacts via sphere subtraction
- Optional atmosphere shell

The SDF is evaluated by:
1. Converting cartesian to spherical coordinates (theta, phi)
2. Sampling FBM noise at spherical coords for terrain height
3. Applying radial displacement from the base sphere
4. Optionally clipping below ocean level

WGSL codegen produces efficient GPU-side evaluation.

Usage:
    >>> planet = PlanetSDF(planet_radius=1.0, terrain_amplitude=0.1)
    >>> sdf_value = planet.evaluate(position)
    >>> wgsl_code = planet.to_wgsl()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, FrozenSet, Generator, List, Optional, Tuple, Any

from .sdf_ast import SDFNode, SDFNodeMeta, Mirror, Tracker, Vec3


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    "PlanetSDF",
    "CraterConfig",
    "PlanetConfig",
    "SphericalCoord",
]

# Mathematical constants
PI = math.pi
TWO_PI = 2.0 * math.pi
HALF_PI = 0.5 * math.pi
INV_PI = 1.0 / math.pi
INV_TWO_PI = 1.0 / TWO_PI

# Default noise parameters
DEFAULT_OCTAVES = 6
DEFAULT_FREQUENCY = 2.0
DEFAULT_LACUNARITY = 2.0
DEFAULT_PERSISTENCE = 0.5


# =============================================================================
# Spherical Coordinate Helper
# =============================================================================

@dataclass(frozen=True, slots=True)
class SphericalCoord:
    """
    Spherical coordinates (r, theta, phi).

    theta: azimuthal angle in xz-plane from +x axis [0, 2*pi)
    phi: polar angle from xz-plane (latitude) [-pi/2, pi/2]
    r: radial distance from origin
    """
    r: float
    theta: float  # azimuth
    phi: float    # elevation/latitude

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float) -> "SphericalCoord":
        """
        Convert cartesian (x, y, z) to spherical (r, theta, phi).

        theta = atan2(z, x)  -- azimuth in xz-plane
        phi = asin(y / r)    -- elevation from xz-plane
        """
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-10:
            return cls(0.0, 0.0, 0.0)

        # Azimuth angle in xz-plane
        theta = math.atan2(z, x)

        # Elevation from xz-plane (latitude)
        # Clamp y/r to [-1, 1] to avoid domain errors at poles
        y_over_r = max(-1.0, min(1.0, y / r))
        phi = math.asin(y_over_r)

        return cls(r, theta, phi)

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert back to cartesian coordinates."""
        cos_phi = math.cos(self.phi)
        x = self.r * cos_phi * math.cos(self.theta)
        y = self.r * math.sin(self.phi)
        z = self.r * cos_phi * math.sin(self.theta)
        return (x, y, z)

    def normalized_uv(self) -> Tuple[float, float]:
        """
        Convert to normalized UV coordinates [0, 1].

        u: longitude mapped from [-pi, pi] to [0, 1]
        v: latitude mapped from [-pi/2, pi/2] to [0, 1]
        """
        u = (self.theta + PI) * INV_TWO_PI
        v = (self.phi + HALF_PI) * INV_PI
        return (u, v)


# =============================================================================
# Crater Configuration
# =============================================================================

@dataclass
class CraterConfig:
    """Configuration for a single impact crater."""
    center: Vec3
    radius: float
    depth: float = 0.5  # As fraction of radius
    rim_height: float = 0.1  # Raised rim as fraction of radius

    def __post_init__(self) -> None:
        """Validate crater parameters."""
        if self.radius <= 0:
            raise ValueError(f"Crater radius must be positive, got {self.radius}")
        if self.depth < 0 or self.depth > 1:
            raise ValueError(f"Crater depth must be in [0, 1], got {self.depth}")
        if self.rim_height < 0:
            raise ValueError(f"Crater rim height must be non-negative, got {self.rim_height}")


# =============================================================================
# Planet Configuration
# =============================================================================

@dataclass
class PlanetConfig:
    """
    Complete configuration for planetary terrain generation.

    All parameters can be modified at runtime with dirty tracking.
    """
    # Core geometry
    planet_radius: float = 1.0
    ocean_level: float = 0.0  # Height below which is ocean (relative to radius)
    terrain_amplitude: float = 0.1  # Max terrain height as fraction of radius

    # Noise parameters
    noise_octaves: int = DEFAULT_OCTAVES
    noise_frequency: float = DEFAULT_FREQUENCY
    noise_lacunarity: float = DEFAULT_LACUNARITY
    noise_persistence: float = DEFAULT_PERSISTENCE
    noise_seed: int = 0

    # Craters
    craters: List[CraterConfig] = field(default_factory=list)
    crater_count: int = 0  # For procedural crater generation
    crater_radius_range: Tuple[float, float] = (0.05, 0.2)

    # Atmosphere
    atmosphere_thickness: float = 0.0  # 0 means no atmosphere
    atmosphere_density: float = 0.5

    # Continent mask
    continent_frequency: float = 1.0
    continent_threshold: float = 0.0  # Values above this are land

    # Mountain regions
    mountain_frequency: float = 3.0
    mountain_threshold: float = 0.5
    mountain_amplitude_scale: float = 2.0  # Multiplier for terrain in mountain regions

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.planet_radius <= 0:
            raise ValueError(f"Planet radius must be positive, got {self.planet_radius}")
        if self.terrain_amplitude < 0:
            raise ValueError(f"Terrain amplitude must be non-negative, got {self.terrain_amplitude}")
        if self.noise_octaves < 1:
            raise ValueError(f"Noise octaves must be >= 1, got {self.noise_octaves}")
        if self.noise_lacunarity <= 0:
            raise ValueError(f"Noise lacunarity must be positive, got {self.noise_lacunarity}")
        if not 0 < self.noise_persistence <= 1:
            raise ValueError(f"Noise persistence must be in (0, 1], got {self.noise_persistence}")
        if self.atmosphere_thickness < 0:
            raise ValueError(f"Atmosphere thickness must be non-negative, got {self.atmosphere_thickness}")


# =============================================================================
# Noise Functions (Python model of WGSL implementation)
# =============================================================================

def _wgsl_fract(x: float) -> float:
    """WGSL fract function: x - floor(x)."""
    return x - math.floor(x)


def _hash21(x: float, y: float) -> float:
    """
    2D hash function returning value in [0, 1).

    Matches WGSL hash21 implementation.
    """
    qx = _wgsl_fract(x * 0.1031)
    qy = _wgsl_fract(y * 0.1030)
    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx = qx + d
    qy = qy + d
    return _wgsl_fract(qx * qy)


def _hash31(x: float, y: float, z: float) -> float:
    """3D hash function returning value in [0, 1)."""
    qx = _wgsl_fract(x * 0.1031)
    qy = _wgsl_fract(y * 0.1030)
    qz = _wgsl_fract(z * 0.0973)
    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d
    return _wgsl_fract(qx * qy * qz)


def _smoothstep(t: float) -> float:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def _value_noise_2d(x: float, y: float) -> float:
    """
    2D value noise with smooth interpolation.

    Returns value in [-1, 1].
    """
    # Integer grid coordinates
    ix = math.floor(x)
    iy = math.floor(y)

    # Fractional parts
    fx = x - ix
    fy = y - iy

    # Smoothstep
    ux = _smoothstep(fx)
    uy = _smoothstep(fy)

    # Hash at 4 corners, scaled to [-1, 1]
    n00 = _hash21(ix, iy) * 2.0 - 1.0
    n10 = _hash21(ix + 1, iy) * 2.0 - 1.0
    n01 = _hash21(ix, iy + 1) * 2.0 - 1.0
    n11 = _hash21(ix + 1, iy + 1) * 2.0 - 1.0

    # Bilinear interpolation
    nx0 = _lerp(n00, n10, ux)
    nx1 = _lerp(n01, n11, ux)
    return _lerp(nx0, nx1, uy)


def _fbm_2d(
    x: float,
    y: float,
    octaves: int,
    frequency: float,
    lacunarity: float,
    persistence: float,
) -> float:
    """
    Fractal Brownian Motion (FBM) using 2D value noise.

    Args:
        x, y: Sample coordinates
        octaves: Number of noise layers
        frequency: Initial frequency
        lacunarity: Frequency multiplier per octave
        persistence: Amplitude multiplier per octave (gain)

    Returns:
        Normalized noise value approximately in [-1, 1]
    """
    if octaves < 1:
        return 0.0

    value = 0.0
    amplitude = 1.0
    freq = frequency
    total_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * _value_noise_2d(x * freq, y * freq)
        total_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    # Normalize by total amplitude
    if total_amplitude > 0:
        value /= total_amplitude

    return value


def _fbm_spherical(
    theta: float,
    phi: float,
    octaves: int,
    frequency: float,
    lacunarity: float,
    persistence: float,
    seed: int = 0,
) -> float:
    """
    FBM noise sampled in spherical coordinates.

    Maps spherical coords to 2D noise space while handling
    polar singularities.
    """
    # Map spherical to 2D with seam handling
    # Use 3D noise internally to avoid seams at theta=+-pi
    cos_phi = math.cos(phi)
    nx = cos_phi * math.cos(theta)
    ny = math.sin(phi)
    nz = cos_phi * math.sin(theta)

    # Sample 3D noise at surface of unit sphere
    # This avoids seams that would occur with 2D UV mapping
    return _fbm_3d(
        nx + seed * 0.1234,
        ny + seed * 0.5678,
        nz + seed * 0.9012,
        octaves,
        frequency,
        lacunarity,
        persistence,
    )


def _value_noise_3d(x: float, y: float, z: float) -> float:
    """3D value noise with smooth interpolation."""
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)

    fx = x - ix
    fy = y - iy
    fz = z - iz

    ux = _smoothstep(fx)
    uy = _smoothstep(fy)
    uz = _smoothstep(fz)

    # Hash at 8 corners
    n000 = _hash31(ix, iy, iz) * 2.0 - 1.0
    n100 = _hash31(ix + 1, iy, iz) * 2.0 - 1.0
    n010 = _hash31(ix, iy + 1, iz) * 2.0 - 1.0
    n110 = _hash31(ix + 1, iy + 1, iz) * 2.0 - 1.0
    n001 = _hash31(ix, iy, iz + 1) * 2.0 - 1.0
    n101 = _hash31(ix + 1, iy, iz + 1) * 2.0 - 1.0
    n011 = _hash31(ix, iy + 1, iz + 1) * 2.0 - 1.0
    n111 = _hash31(ix + 1, iy + 1, iz + 1) * 2.0 - 1.0

    # Trilinear interpolation
    nx00 = _lerp(n000, n100, ux)
    nx10 = _lerp(n010, n110, ux)
    nx01 = _lerp(n001, n101, ux)
    nx11 = _lerp(n011, n111, ux)

    nxy0 = _lerp(nx00, nx10, uy)
    nxy1 = _lerp(nx01, nx11, uy)

    return _lerp(nxy0, nxy1, uz)


def _fbm_3d(
    x: float,
    y: float,
    z: float,
    octaves: int,
    frequency: float,
    lacunarity: float,
    persistence: float,
) -> float:
    """3D Fractal Brownian Motion."""
    if octaves < 1:
        return 0.0

    value = 0.0
    amplitude = 1.0
    freq = frequency
    total_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * _value_noise_3d(x * freq, y * freq, z * freq)
        total_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    if total_amplitude > 0:
        value /= total_amplitude

    return value


# =============================================================================
# PlanetSDF - Main Class
# =============================================================================

class PlanetSDF(SDFNode):
    """
    Planetary terrain SDF with spherical noise displacement.

    Implements a planet as a base sphere with FBM noise displacement
    applied radially. Supports:
    - Ocean level clipping
    - Continent/water masks
    - Mountain regions with higher amplitude
    - Impact craters
    - Atmosphere shell

    Example:
        >>> planet = PlanetSDF(planet_radius=1.0, terrain_amplitude=0.1)
        >>> sdf = planet.evaluate(Vec3(1.5, 0.0, 0.0))
        >>> wgsl = planet.to_wgsl()
    """

    __slots__ = (
        "_config",
        "_cached_craters",
    )

    # Type annotations for metaclass
    config: PlanetConfig

    def __init__(
        self,
        config: Optional[PlanetConfig] = None,
        *,
        # Convenience parameters (override config if provided)
        planet_radius: Optional[float] = None,
        ocean_level: Optional[float] = None,
        terrain_amplitude: Optional[float] = None,
        noise_octaves: Optional[int] = None,
        noise_frequency: Optional[float] = None,
        noise_lacunarity: Optional[float] = None,
        noise_persistence: Optional[float] = None,
        noise_seed: Optional[int] = None,
        crater_count: Optional[int] = None,
        crater_radius_range: Optional[Tuple[float, float]] = None,
        atmosphere_thickness: Optional[float] = None,
    ) -> None:
        """
        Initialize PlanetSDF.

        Args:
            config: Full configuration object
            planet_radius: Base sphere radius
            ocean_level: Height below which is ocean
            terrain_amplitude: Max terrain displacement
            noise_octaves: FBM octave count
            noise_frequency: Initial noise frequency
            noise_lacunarity: Frequency multiplier per octave
            noise_persistence: Amplitude multiplier per octave
            noise_seed: Random seed for noise
            crater_count: Number of procedural craters
            crater_radius_range: (min, max) crater radius
            atmosphere_thickness: Thickness of atmosphere shell
        """
        super().__init__()

        # Start with default or provided config
        if config is not None:
            self._config = config
        else:
            self._config = PlanetConfig()

        # Apply any override parameters
        if planet_radius is not None:
            self._config.planet_radius = planet_radius
        if ocean_level is not None:
            self._config.ocean_level = ocean_level
        if terrain_amplitude is not None:
            self._config.terrain_amplitude = terrain_amplitude
        if noise_octaves is not None:
            self._config.noise_octaves = noise_octaves
        if noise_frequency is not None:
            self._config.noise_frequency = noise_frequency
        if noise_lacunarity is not None:
            self._config.noise_lacunarity = noise_lacunarity
        if noise_persistence is not None:
            self._config.noise_persistence = noise_persistence
        if noise_seed is not None:
            self._config.noise_seed = noise_seed
        if crater_count is not None:
            self._config.crater_count = crater_count
        if crater_radius_range is not None:
            self._config.crater_radius_range = crater_radius_range
        if atmosphere_thickness is not None:
            self._config.atmosphere_thickness = atmosphere_thickness

        # Generate procedural craters if requested
        self._cached_craters: List[CraterConfig] = list(self._config.craters)
        if self._config.crater_count > 0:
            self._generate_craters()

        # Mark all fields dirty initially
        self.tracker.mark_dirty("config")

    @property
    def config(self) -> PlanetConfig:
        """Get planet configuration."""
        return self._config

    @config.setter
    def config(self, value: PlanetConfig) -> None:
        """Set planet configuration with dirty tracking."""
        self._config = value
        self._cached_craters = list(value.craters)
        if value.crater_count > 0:
            self._generate_craters()
        self.tracker.mark_dirty("config")

    @property
    def planet_radius(self) -> float:
        """Get planet radius."""
        return self._config.planet_radius

    @planet_radius.setter
    def planet_radius(self, value: float) -> None:
        """Set planet radius."""
        if value <= 0:
            raise ValueError(f"Planet radius must be positive, got {value}")
        self._config.planet_radius = value
        self.tracker.mark_dirty("config")

    @property
    def ocean_level(self) -> float:
        """Get ocean level."""
        return self._config.ocean_level

    @ocean_level.setter
    def ocean_level(self, value: float) -> None:
        """Set ocean level."""
        self._config.ocean_level = value
        self.tracker.mark_dirty("config")

    @property
    def terrain_amplitude(self) -> float:
        """Get terrain amplitude."""
        return self._config.terrain_amplitude

    @terrain_amplitude.setter
    def terrain_amplitude(self, value: float) -> None:
        """Set terrain amplitude."""
        if value < 0:
            raise ValueError(f"Terrain amplitude must be non-negative, got {value}")
        self._config.terrain_amplitude = value
        self.tracker.mark_dirty("config")

    @property
    def craters(self) -> List[CraterConfig]:
        """Get all craters (configured + generated)."""
        return self._cached_craters

    def _generate_craters(self) -> None:
        """Generate procedural craters using deterministic random placement."""
        seed = self._config.noise_seed
        count = self._config.crater_count
        min_r, max_r = self._config.crater_radius_range

        for i in range(count):
            # Deterministic pseudo-random placement
            hash_val = _hash31(seed + i * 17.0, seed * 0.7 + i, seed * 0.3 + i * 2)

            # Random direction on sphere
            u = _hash31(seed + i * 31, i * 0.5, seed) * 2.0 - 1.0
            v = _hash31(i * 0.3, seed + i * 23, i) * TWO_PI

            # Convert to cartesian on unit sphere
            cos_u = math.sqrt(1.0 - u * u)
            cx = cos_u * math.cos(v)
            cy = u
            cz = cos_u * math.sin(v)

            # Scale to planet surface
            r = self._config.planet_radius
            center = Vec3(cx * r, cy * r, cz * r)

            # Random radius within range
            crater_radius = min_r + hash_val * (max_r - min_r)

            self._cached_craters.append(CraterConfig(
                center=center,
                radius=crater_radius,
                depth=0.3 + hash_val * 0.4,
                rim_height=0.05 + hash_val * 0.1,
            ))

    def add_crater(self, crater: CraterConfig) -> None:
        """Add a crater to the planet."""
        self._cached_craters.append(crater)
        self.tracker.mark_dirty("config")

    def clear_craters(self) -> None:
        """Remove all craters."""
        self._cached_craters.clear()
        self.tracker.mark_dirty("config")

    def _sample_terrain_height(self, theta: float, phi: float) -> float:
        """
        Sample terrain height at spherical coordinates.

        Returns height offset from base radius.
        """
        cfg = self._config

        # Base FBM terrain
        height = _fbm_spherical(
            theta,
            phi,
            cfg.noise_octaves,
            cfg.noise_frequency,
            cfg.noise_lacunarity,
            cfg.noise_persistence,
            cfg.noise_seed,
        )

        # Apply amplitude
        height *= cfg.terrain_amplitude * cfg.planet_radius

        # Mountain region modulation
        if cfg.mountain_amplitude_scale != 1.0:
            mountain_noise = _fbm_spherical(
                theta,
                phi,
                max(1, cfg.noise_octaves // 2),
                cfg.mountain_frequency,
                cfg.noise_lacunarity,
                cfg.noise_persistence,
                cfg.noise_seed + 1000,
            )
            if mountain_noise > cfg.mountain_threshold:
                mountain_factor = (mountain_noise - cfg.mountain_threshold) / (1.0 - cfg.mountain_threshold)
                height *= 1.0 + mountain_factor * (cfg.mountain_amplitude_scale - 1.0)

        # Ocean clipping
        if height < cfg.ocean_level:
            height = cfg.ocean_level

        return height

    def _evaluate_crater(self, px: float, py: float, pz: float, crater: CraterConfig) -> float:
        """
        Evaluate crater SDF contribution.

        Uses smooth subtraction to create bowl-shaped craters with rims.
        """
        # Distance from crater center
        dx = px - crater.center.x
        dy = py - crater.center.y
        dz = pz - crater.center.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Crater is a sphere subtraction
        crater_depth = crater.radius * crater.depth
        crater_sdf = dist - crater.radius

        # Rim displacement (raised ring around crater)
        if crater.rim_height > 0:
            rim_dist = abs(dist - crater.radius)
            rim_width = crater.radius * 0.3
            if rim_dist < rim_width:
                rim_factor = 1.0 - rim_dist / rim_width
                rim_factor = rim_factor * rim_factor  # Smooth falloff
                # This will be applied as height offset

        return crater_sdf

    def evaluate(self, p: Vec3) -> float:
        """
        Evaluate the planet SDF at position p.

        Args:
            p: Query position in world space

        Returns:
            Signed distance to planet surface (negative inside)
        """
        return self.evaluate_xyz(p.x, p.y, p.z)

    def evaluate_xyz(self, px: float, py: float, pz: float) -> float:
        """
        Evaluate the planet SDF at position (px, py, pz).

        More efficient than evaluate() for bulk evaluation.
        """
        cfg = self._config

        # Convert to spherical coordinates
        spherical = SphericalCoord.from_cartesian(px, py, pz)
        r = spherical.r
        theta = spherical.theta
        phi = spherical.phi

        # Handle center singularity
        if r < 1e-10:
            return -cfg.planet_radius

        # Sample terrain height
        terrain_height = self._sample_terrain_height(theta, phi)

        # Base sphere SDF with terrain displacement
        effective_radius = cfg.planet_radius + terrain_height
        planet_sdf = r - effective_radius

        # Apply craters via smooth min
        for crater in self._cached_craters:
            crater_sdf = self._evaluate_crater(px, py, pz, crater)
            # Smooth subtraction: removes crater volume from planet
            k = crater.radius * 0.2  # Smoothing factor
            h = max(0.0, min(1.0, 0.5 - 0.5 * (crater_sdf + planet_sdf) / k))
            planet_sdf = _lerp(planet_sdf, -crater_sdf, h) + k * h * (1.0 - h)

        # Continent mask (optional: return different value for water)
        # This doesn't affect SDF but could be used for materials

        return planet_sdf

    def evaluate_atmosphere(self, px: float, py: float, pz: float) -> float:
        """
        Evaluate atmosphere shell SDF.

        Returns distance to outer atmosphere boundary.
        """
        cfg = self._config
        if cfg.atmosphere_thickness <= 0:
            return float('inf')

        r = math.sqrt(px * px + py * py + pz * pz)
        atmo_radius = cfg.planet_radius + cfg.terrain_amplitude + cfg.atmosphere_thickness
        return r - atmo_radius

    def is_ocean(self, theta: float, phi: float) -> bool:
        """
        Check if position at (theta, phi) is below ocean level.

        Useful for material assignment.
        """
        cfg = self._config
        raw_height = _fbm_spherical(
            theta,
            phi,
            cfg.noise_octaves,
            cfg.noise_frequency,
            cfg.noise_lacunarity,
            cfg.noise_persistence,
            cfg.noise_seed,
        ) * cfg.terrain_amplitude * cfg.planet_radius
        return raw_height < cfg.ocean_level

    def is_continent(self, theta: float, phi: float) -> bool:
        """
        Check if position is on a continent (land mass).

        Uses separate noise for continent distribution.
        """
        cfg = self._config
        continent_noise = _fbm_spherical(
            theta,
            phi,
            2,  # Low frequency for large continents
            cfg.continent_frequency,
            2.0,
            0.5,
            cfg.noise_seed + 500,
        )
        return continent_noise > cfg.continent_threshold

    def sample_normal(self, p: Vec3, epsilon: float = 0.001) -> Vec3:
        """
        Compute surface normal at position using central differences.

        Args:
            p: Query position
            epsilon: Sampling distance for gradient

        Returns:
            Normalized surface normal vector
        """
        px, py, pz = p.x, p.y, p.z

        # Central difference gradient
        dx = self.evaluate_xyz(px + epsilon, py, pz) - self.evaluate_xyz(px - epsilon, py, pz)
        dy = self.evaluate_xyz(px, py + epsilon, pz) - self.evaluate_xyz(px, py - epsilon, pz)
        dz = self.evaluate_xyz(px, py, pz + epsilon) - self.evaluate_xyz(px, py, pz - epsilon)

        # Normalize
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-10:
            return Vec3(0.0, 1.0, 0.0)  # Default up

        return Vec3(dx / length, dy / length, dz / length)

    def label(self) -> str:
        """Return node label for debugging."""
        cfg = self._config
        return f"PlanetSDF(r={cfg.planet_radius}, amp={cfg.terrain_amplitude})"

    def clone(self) -> "PlanetSDF":
        """Create a deep copy."""
        import copy
        return PlanetSDF(config=copy.deepcopy(self._config))

    def to_wgsl(self, function_name: str = "sdf_planet") -> str:
        """
        Generate WGSL code for GPU-side evaluation.

        Args:
            function_name: Name for the generated function

        Returns:
            WGSL source code string
        """
        cfg = self._config

        # Build crater array literal
        crater_code = ""
        if self._cached_craters:
            crater_entries = []
            for c in self._cached_craters:
                crater_entries.append(
                    f"    Crater(vec3<f32>({c.center.x}, {c.center.y}, {c.center.z}), "
                    f"{c.radius}, {c.depth}, {c.rim_height})"
                )
            crater_array = ",\n".join(crater_entries)
            crater_code = f"""
const CRATERS: array<Crater, {len(self._cached_craters)}> = array<Crater, {len(self._cached_craters)}>(
{crater_array}
);
"""

        wgsl = f"""\
// Auto-generated by PlanetSDF.to_wgsl() (T-DEMO-4.9)
// Planet Configuration:
//   Radius: {cfg.planet_radius}
//   Terrain Amplitude: {cfg.terrain_amplitude}
//   Ocean Level: {cfg.ocean_level}
//   Noise Octaves: {cfg.noise_octaves}
//   Noise Frequency: {cfg.noise_frequency}

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const HALF_PI: f32 = 1.57079632679;

// Planet parameters
const PLANET_RADIUS: f32 = {cfg.planet_radius};
const TERRAIN_AMPLITUDE: f32 = {cfg.terrain_amplitude};
const OCEAN_LEVEL: f32 = {cfg.ocean_level};
const NOISE_OCTAVES: i32 = {cfg.noise_octaves};
const NOISE_FREQUENCY: f32 = {cfg.noise_frequency};
const NOISE_LACUNARITY: f32 = {cfg.noise_lacunarity};
const NOISE_PERSISTENCE: f32 = {cfg.noise_persistence};
const NOISE_SEED: f32 = {float(cfg.noise_seed)};
const MOUNTAIN_THRESHOLD: f32 = {cfg.mountain_threshold};
const MOUNTAIN_AMPLITUDE_SCALE: f32 = {cfg.mountain_amplitude_scale};
const MOUNTAIN_FREQUENCY: f32 = {cfg.mountain_frequency};

struct Crater {{
    center: vec3<f32>,
    radius: f32,
    depth: f32,
    rim_height: f32,
}};
{crater_code}
const CRATER_COUNT: i32 = {len(self._cached_craters)};

// Hash functions
fn hash21(p: vec2<f32>) -> f32 {{
    var q = fract(p * vec2<f32>(0.1031, 0.1030));
    let d = q.x * (q.x + 33.33) + q.y * (q.y + 33.33);
    q = q + d;
    return fract(q.x * q.y);
}}

fn hash31(p: vec3<f32>) -> f32 {{
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    let d = q.x * (q.x + 33.33) + q.y * (q.y + 33.33) + q.z * (q.z + 33.33);
    q = q + d;
    return fract(q.x * q.y * q.z);
}}

// Smoothstep for noise interpolation
fn smoothstep_fade(t: f32) -> f32 {{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}}

// 3D Value noise
fn value_noise_3d(p: vec3<f32>) -> f32 {{
    let i = floor(p);
    let f = p - i;

    let u = vec3<f32>(smoothstep_fade(f.x), smoothstep_fade(f.y), smoothstep_fade(f.z));

    let n000 = hash31(i) * 2.0 - 1.0;
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0)) * 2.0 - 1.0;
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0)) * 2.0 - 1.0;
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0)) * 2.0 - 1.0;
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0)) * 2.0 - 1.0;
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0)) * 2.0 - 1.0;
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0)) * 2.0 - 1.0;
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0)) * 2.0 - 1.0;

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z);
}}

// FBM noise sampled at spherical surface position
fn fbm_spherical(p: vec3<f32>, octaves: i32, frequency: f32, lacunarity: f32, persistence: f32, seed: f32) -> f32 {{
    if (octaves < 1) {{
        return 0.0;
    }}

    // Sample 3D noise at unit sphere surface to avoid seams
    let n = normalize(p);
    let offset = vec3<f32>(seed * 0.1234, seed * 0.5678, seed * 0.9012);

    var value: f32 = 0.0;
    var amplitude: f32 = 1.0;
    var freq: f32 = frequency;
    var total_amplitude: f32 = 0.0;

    for (var i: i32 = 0; i < octaves; i = i + 1) {{
        value = value + amplitude * value_noise_3d((n + offset) * freq);
        total_amplitude = total_amplitude + amplitude;
        amplitude = amplitude * persistence;
        freq = freq * lacunarity;
    }}

    if (total_amplitude > 0.0) {{
        value = value / total_amplitude;
    }}

    return value;
}}

// Sample terrain height at position on sphere
fn sample_terrain_height(p: vec3<f32>) -> f32 {{
    // Base terrain
    var height = fbm_spherical(p, NOISE_OCTAVES, NOISE_FREQUENCY, NOISE_LACUNARITY, NOISE_PERSISTENCE, NOISE_SEED);
    height = height * TERRAIN_AMPLITUDE * PLANET_RADIUS;

    // Mountain modulation
    if (MOUNTAIN_AMPLITUDE_SCALE != 1.0) {{
        let mountain_octaves = max(1, NOISE_OCTAVES / 2);
        let mountain_noise = fbm_spherical(p, mountain_octaves, MOUNTAIN_FREQUENCY, NOISE_LACUNARITY, NOISE_PERSISTENCE, NOISE_SEED + 1000.0);
        if (mountain_noise > MOUNTAIN_THRESHOLD) {{
            let mountain_factor = (mountain_noise - MOUNTAIN_THRESHOLD) / (1.0 - MOUNTAIN_THRESHOLD);
            height = height * (1.0 + mountain_factor * (MOUNTAIN_AMPLITUDE_SCALE - 1.0));
        }}
    }}

    // Ocean clipping
    if (height < OCEAN_LEVEL) {{
        height = OCEAN_LEVEL;
    }}

    return height;
}}

// Evaluate crater contribution
fn evaluate_crater(p: vec3<f32>, crater: Crater) -> f32 {{
    let d = p - crater.center;
    let dist = length(d);
    return dist - crater.radius;
}}

// Main planet SDF function
fn {function_name}(p: vec3<f32>) -> f32 {{
    let r = length(p);

    // Handle center singularity
    if (r < 1e-10) {{
        return -PLANET_RADIUS;
    }}

    // Sample terrain height
    let terrain_height = sample_terrain_height(p);

    // Base sphere with terrain displacement
    let effective_radius = PLANET_RADIUS + terrain_height;
    var sdf = r - effective_radius;

    // Apply craters
    for (var i: i32 = 0; i < CRATER_COUNT; i = i + 1) {{
        let crater_sdf = evaluate_crater(p, CRATERS[i]);
        let k = CRATERS[i].radius * 0.2;
        let h = clamp(0.5 - 0.5 * (crater_sdf + sdf) / k, 0.0, 1.0);
        sdf = mix(sdf, -crater_sdf, h) + k * h * (1.0 - h);
    }}

    return sdf;
}}

// Surface normal via central differences
fn {function_name}_normal(p: vec3<f32>) -> vec3<f32> {{
    let e = 0.001;
    let dx = {function_name}(p + vec3<f32>(e, 0.0, 0.0)) - {function_name}(p - vec3<f32>(e, 0.0, 0.0));
    let dy = {function_name}(p + vec3<f32>(0.0, e, 0.0)) - {function_name}(p - vec3<f32>(0.0, e, 0.0));
    let dz = {function_name}(p + vec3<f32>(0.0, 0.0, e)) - {function_name}(p - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(dx, dy, dz));
}}
"""

        # Add atmosphere if configured
        if cfg.atmosphere_thickness > 0:
            atmo_radius = cfg.planet_radius + cfg.terrain_amplitude + cfg.atmosphere_thickness
            wgsl += f"""
// Atmosphere shell SDF
const ATMOSPHERE_RADIUS: f32 = {atmo_radius};
const ATMOSPHERE_DENSITY: f32 = {cfg.atmosphere_density};

fn {function_name}_atmosphere(p: vec3<f32>) -> f32 {{
    return length(p) - ATMOSPHERE_RADIUS;
}}

// Atmosphere density falloff (for volumetric rendering)
fn {function_name}_atmosphere_density(p: vec3<f32>) -> f32 {{
    let r = length(p);
    if (r > ATMOSPHERE_RADIUS) {{
        return 0.0;
    }}
    let surface_r = PLANET_RADIUS + sample_terrain_height(p);
    if (r < surface_r) {{
        return 0.0;
    }}
    // Linear falloff from surface to edge
    let t = (r - surface_r) / (ATMOSPHERE_RADIUS - surface_r);
    return ATMOSPHERE_DENSITY * (1.0 - t);
}}
"""

        return wgsl
