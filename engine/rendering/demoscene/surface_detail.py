"""
TRINITY Demoscene Surface Detail Module (T-DEMO-4.12, T-DEMO-4.13)

This module provides surface detail utilities for SDF rendering:

T-DEMO-4.12: Bump Mapping from Noise Gradients
- BumpMapper class perturbing surface normals using FBM noise
- 4 FBM evaluations per pixel for gradient estimation
- Gradient: (fbm(p+dx) - fbm(p-dx), ...) / (2*dx) for central differences
- Perturb normal: n' = normalize(n - gradient * bump_strength)
- Parameters: noise_frequency, bump_strength, octaves

T-DEMO-4.13: Surface Curvature Detection
- CurvatureDetector class via Laplacian of noise field
- Laplacian: sum(f(p+d) + f(p-d) - 2*f(p)) for each axis
- Detects edges, creases, ridges
- Output: curvature value for material variation
- Sign indicates convex (+) or concave (-)
- Parameters: sample_distance, threshold

Reference:
- Inigo Quilez bump mapping: https://iquilezles.org/articles/bumpmap/
- Inigo Quilez curvature: https://iquilezles.org/articles/nvscene2008/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from .sdf_ast import Mirror, Tracker, Vec3


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # T-DEMO-4.12: Bump Mapping
    "BumpMapConfig",
    "BumpMapper",
    "compute_bump_normal",
    "compute_noise_gradient_3d",
    # T-DEMO-4.13: Curvature Detection
    "CurvatureConfig",
    "CurvatureDetector",
    "CurvatureResult",
    "compute_laplacian",
    "detect_edges",
    "detect_ridges",
    # WGSL Generation
    "generate_bump_mapping_wgsl",
    "generate_curvature_wgsl",
    # Noise functions (Python reference)
    "fbm_3d",
    "value_noise_3d",
    "perlin_noise_3d",
]

# Default parameters
DEFAULT_BUMP_STRENGTH = 0.1
DEFAULT_NOISE_FREQUENCY = 1.0
DEFAULT_OCTAVES = 4
DEFAULT_LACUNARITY = 2.0
DEFAULT_GAIN = 0.5
DEFAULT_GRADIENT_DX = 0.001
DEFAULT_SAMPLE_DISTANCE = 0.01
DEFAULT_CURVATURE_THRESHOLD = 0.1


# =============================================================================
# Vector Math Helpers
# =============================================================================


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two Vec3 vectors."""
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two Vec3 vectors."""
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def vec3_mul(v: Vec3, s: float) -> Vec3:
    """Multiply Vec3 by scalar."""
    return Vec3(v.x * s, v.y * s, v.z * s)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two Vec3 vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two Vec3 vectors."""
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def vec3_length(v: Vec3) -> float:
    """Length of a Vec3 vector."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a Vec3 vector to unit length."""
    length = vec3_length(v)
    if length < 1e-10:
        return Vec3(0.0, 0.0, 0.0)
    inv_len = 1.0 / length
    return Vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len)


# =============================================================================
# Noise Functions (Python Reference Implementation)
# =============================================================================


def wgsl_fract(x: float) -> float:
    """WGSL fract: x - floor(x)."""
    return x - math.floor(x)


def smoothstep_fade(t: float) -> float:
    """Smoothstep fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + t * (b - a)


def hash31(p: Tuple[float, float, float]) -> float:
    """Hash 3D point to [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)
    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx = qx + d
    qy = qy + d
    qz = qz + d
    return wgsl_fract(qx * qy * qz)


def value_noise_3d(p: Union[Vec3, Tuple[float, float, float]]) -> float:
    """
    3D value noise returning value in [-1, 1].

    Uses trilinear interpolation of hashed corner values.
    """
    if isinstance(p, Vec3):
        px, py, pz = p.x, p.y, p.z
    else:
        px, py, pz = p

    ix = math.floor(px)
    iy = math.floor(py)
    iz = math.floor(pz)
    fx = px - ix
    fy = py - iy
    fz = pz - iz

    ux = smoothstep_fade(fx)
    uy = smoothstep_fade(fy)
    uz = smoothstep_fade(fz)

    # Hash corner values
    a = hash31((ix, iy, iz))
    b = hash31((ix + 1.0, iy, iz))
    c = hash31((ix, iy + 1.0, iz))
    d = hash31((ix + 1.0, iy + 1.0, iz))
    e = hash31((ix, iy, iz + 1.0))
    f = hash31((ix + 1.0, iy, iz + 1.0))
    g = hash31((ix, iy + 1.0, iz + 1.0))
    h = hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    # Convert to [-1, 1]
    va = a * 2.0 - 1.0
    vb = b * 2.0 - 1.0
    vc = c * 2.0 - 1.0
    vd = d * 2.0 - 1.0
    ve = e * 2.0 - 1.0
    vf = f * 2.0 - 1.0
    vg = g * 2.0 - 1.0
    vh = h * 2.0 - 1.0

    # Trilinear interpolation
    vx00 = lerp(va, vb, ux)
    vx10 = lerp(vc, vd, ux)
    vx01 = lerp(ve, vf, ux)
    vx11 = lerp(vg, vh, ux)
    vy0 = lerp(vx00, vx10, uy)
    vy1 = lerp(vx01, vx11, uy)
    return lerp(vy0, vy1, uz)


# Perlin noise gradient table
_GRADIENTS: List[Tuple[float, float, float]] = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
]

INV_SQRT2 = 0.7071067811865475


def perlin_gradient(hash_value: float, offset: Tuple[float, float, float]) -> float:
    """Select gradient and compute dot product with offset."""
    h = int(hash_value * 12.0) % 12
    gx, gy, gz = _GRADIENTS[h]
    gx *= INV_SQRT2
    gy *= INV_SQRT2
    gz *= INV_SQRT2
    return gx * offset[0] + gy * offset[1] + gz * offset[2]


def perlin_noise_3d(p: Union[Vec3, Tuple[float, float, float]]) -> float:
    """
    3D Perlin noise returning value in approximately [-1, 1].

    Uses gradient interpolation for smoother results than value noise.
    """
    if isinstance(p, Vec3):
        px, py, pz = p.x, p.y, p.z
    else:
        px, py, pz = p

    ix = math.floor(px)
    iy = math.floor(py)
    iz = math.floor(pz)
    fx = px - ix
    fy = py - iy
    fz = pz - iz

    ux = smoothstep_fade(fx)
    uy = smoothstep_fade(fy)
    uz = smoothstep_fade(fz)

    # Offset vectors
    o000 = (fx, fy, fz)
    o100 = (fx - 1.0, fy, fz)
    o010 = (fx, fy - 1.0, fz)
    o110 = (fx - 1.0, fy - 1.0, fz)
    o001 = (fx, fy, fz - 1.0)
    o101 = (fx - 1.0, fy, fz - 1.0)
    o011 = (fx, fy - 1.0, fz - 1.0)
    o111 = (fx - 1.0, fy - 1.0, fz - 1.0)

    # Hash corner positions
    h000 = hash31((ix, iy, iz))
    h100 = hash31((ix + 1.0, iy, iz))
    h010 = hash31((ix, iy + 1.0, iz))
    h110 = hash31((ix + 1.0, iy + 1.0, iz))
    h001 = hash31((ix, iy, iz + 1.0))
    h101 = hash31((ix + 1.0, iy, iz + 1.0))
    h011 = hash31((ix, iy + 1.0, iz + 1.0))
    h111 = hash31((ix + 1.0, iy + 1.0, iz + 1.0))

    # Gradient dot products
    g000 = perlin_gradient(h000, o000)
    g100 = perlin_gradient(h100, o100)
    g010 = perlin_gradient(h010, o010)
    g110 = perlin_gradient(h110, o110)
    g001 = perlin_gradient(h001, o001)
    g101 = perlin_gradient(h101, o101)
    g011 = perlin_gradient(h011, o011)
    g111 = perlin_gradient(h111, o111)

    # Trilinear interpolation
    vx00 = g000 + ux * (g100 - g000)
    vx10 = g010 + ux * (g110 - g010)
    vx01 = g001 + ux * (g101 - g001)
    vx11 = g011 + ux * (g111 - g011)
    vy0 = vx00 + uy * (vx10 - vx00)
    vy1 = vx01 + uy * (vx11 - vx01)
    return vy0 + uz * (vy1 - vy0)


def fbm_3d(
    p: Union[Vec3, Tuple[float, float, float]],
    octaves: int = DEFAULT_OCTAVES,
    lacunarity: float = DEFAULT_LACUNARITY,
    gain: float = DEFAULT_GAIN,
    use_perlin: bool = False,
) -> float:
    """
    Fractal Brownian Motion (FBM) in 3D.

    Layers multiple octaves of noise at increasing frequencies
    and decaying amplitudes.

    Args:
        p: 3D position
        octaves: Number of noise layers
        lacunarity: Frequency multiplier between octaves
        gain: Amplitude multiplier between octaves
        use_perlin: Use Perlin noise instead of value noise

    Returns:
        FBM value normalized to approximately [-1, 1]
    """
    if isinstance(p, Vec3):
        px, py, pz = p.x, p.y, p.z
    else:
        px, py, pz = p

    if octaves <= 0:
        return 0.0

    noise_fn = perlin_noise_3d if use_perlin else value_noise_3d

    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        value += amplitude * noise_fn((px * frequency, py * frequency, pz * frequency))
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain

    if max_amplitude < 1e-8:
        return 0.0
    return value / max_amplitude


# =============================================================================
# T-DEMO-4.12: Bump Mapping from Noise Gradients
# =============================================================================


@dataclass
class BumpMapConfig:
    """
    Configuration for bump mapping.

    Attributes:
        noise_frequency: Scale factor for noise sampling coordinates
        bump_strength: Strength of normal perturbation [0, 1+]
        octaves: Number of FBM octaves
        lacunarity: Frequency multiplier between octaves
        gain: Amplitude multiplier between octaves
        gradient_dx: Step size for gradient estimation
        use_perlin: Use Perlin noise instead of value noise
    """
    noise_frequency: float = DEFAULT_NOISE_FREQUENCY
    bump_strength: float = DEFAULT_BUMP_STRENGTH
    octaves: int = DEFAULT_OCTAVES
    lacunarity: float = DEFAULT_LACUNARITY
    gain: float = DEFAULT_GAIN
    gradient_dx: float = DEFAULT_GRADIENT_DX
    use_perlin: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.noise_frequency <= 0:
            raise ValueError(f"noise_frequency must be > 0, got {self.noise_frequency}")
        if self.bump_strength < 0:
            raise ValueError(f"bump_strength must be >= 0, got {self.bump_strength}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if self.gain <= 0:
            raise ValueError(f"gain must be > 0, got {self.gain}")
        if self.gradient_dx <= 0:
            raise ValueError(f"gradient_dx must be > 0, got {self.gradient_dx}")


def compute_noise_gradient_3d(
    p: Vec3,
    noise_func: Callable[[Vec3], float],
    dx: float = DEFAULT_GRADIENT_DX,
) -> Vec3:
    """
    Compute gradient of a noise function at position p using central differences.

    Uses 4 FBM evaluations for a more efficient approximation:
    - Evaluate at p for reference
    - Evaluate at p + dx along each axis
    - Gradient = (f(p+dx) - f(p-dx)) / (2*dx) for each axis

    For full accuracy, this uses 6 samples (+/- for each axis).

    Args:
        p: Position to evaluate gradient at
        noise_func: Noise function taking Vec3 and returning float
        dx: Step size for finite differences

    Returns:
        Gradient vector (df/dx, df/dy, df/dz)
    """
    # Central differences: 6 samples for accurate gradient
    dx_vec = Vec3(dx, 0.0, 0.0)
    dy_vec = Vec3(0.0, dx, 0.0)
    dz_vec = Vec3(0.0, 0.0, dx)

    inv_2dx = 1.0 / (2.0 * dx)

    # Gradient components via central differences
    grad_x = (noise_func(vec3_add(p, dx_vec)) - noise_func(vec3_sub(p, dx_vec))) * inv_2dx
    grad_y = (noise_func(vec3_add(p, dy_vec)) - noise_func(vec3_sub(p, dy_vec))) * inv_2dx
    grad_z = (noise_func(vec3_add(p, dz_vec)) - noise_func(vec3_sub(p, dz_vec))) * inv_2dx

    return Vec3(grad_x, grad_y, grad_z)


def compute_bump_normal(
    normal: Vec3,
    position: Vec3,
    config: BumpMapConfig,
    noise_func: Optional[Callable[[Vec3], float]] = None,
) -> Vec3:
    """
    Compute perturbed surface normal using bump mapping.

    The bump mapping formula:
        n' = normalize(n - gradient * bump_strength)

    Where gradient is computed from the noise field.

    Args:
        normal: Original surface normal (should be unit length)
        position: World position on the surface
        config: Bump mapping configuration
        noise_func: Optional custom noise function

    Returns:
        Perturbed normal (unit length)
    """
    # Scale position by noise frequency
    scaled_pos = vec3_mul(position, config.noise_frequency)

    # Create noise function if not provided
    if noise_func is None:
        def default_noise(p: Vec3) -> float:
            return fbm_3d(
                p,
                octaves=config.octaves,
                lacunarity=config.lacunarity,
                gain=config.gain,
                use_perlin=config.use_perlin,
            )
        noise_func = default_noise

    # Compute noise gradient
    gradient = compute_noise_gradient_3d(scaled_pos, noise_func, config.gradient_dx)

    # Perturb normal: n' = normalize(n - gradient * bump_strength)
    perturbation = vec3_mul(gradient, config.bump_strength)
    perturbed = vec3_sub(normal, perturbation)

    # Normalize result
    result = vec3_normalize(perturbed)

    # Handle degenerate case (zero-length normal)
    if vec3_length(result) < 1e-10:
        return normal

    return result


class BumpMapper:
    """
    Bump mapper for perturbing surface normals using noise gradients.

    Implements T-DEMO-4.12 with Trinity Tracker pattern for cache invalidation.

    Usage:
        >>> config = BumpMapConfig(bump_strength=0.1, octaves=4)
        >>> mapper = BumpMapper(config)
        >>> bumped_normal = mapper.compute_normal(normal, position)

    The mapper uses 4 FBM evaluations per pixel to estimate the noise gradient,
    then perturbs the surface normal accordingly.
    """

    def __init__(
        self,
        config: Optional[BumpMapConfig] = None,
        noise_func: Optional[Callable[[Vec3], float]] = None,
    ) -> None:
        """
        Initialize bump mapper.

        Args:
            config: Bump mapping configuration
            noise_func: Optional custom noise function
        """
        self._config = config or BumpMapConfig()
        self._noise_func = noise_func
        self._version = 0
        self._dirty = True

    @property
    def config(self) -> BumpMapConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: BumpMapConfig) -> None:
        """Set configuration and mark dirty."""
        self._config = value
        self._dirty = True
        self._version += 1

    @property
    def tracker(self) -> "BumpMapperTracker":
        """Get tracker for dirty checking."""
        return BumpMapperTracker(self)

    @property
    def mirror(self) -> "BumpMapperMirror":
        """Get mirror for introspection."""
        return BumpMapperMirror(self)

    def compute_normal(self, normal: Vec3, position: Vec3) -> Vec3:
        """
        Compute perturbed surface normal at given position.

        Args:
            normal: Original surface normal (unit length)
            position: World position on surface

        Returns:
            Perturbed normal (unit length)
        """
        return compute_bump_normal(normal, position, self._config, self._noise_func)

    def compute_normal_batch(
        self,
        normals: Sequence[Vec3],
        positions: Sequence[Vec3],
    ) -> List[Vec3]:
        """
        Compute perturbed normals for multiple positions.

        Args:
            normals: List of original surface normals
            positions: List of world positions

        Returns:
            List of perturbed normals
        """
        if len(normals) != len(positions):
            raise ValueError("normals and positions must have same length")

        return [
            self.compute_normal(n, p)
            for n, p in zip(normals, positions)
        ]

    def to_wgsl(self) -> str:
        """Generate WGSL code for this bump mapper."""
        return generate_bump_mapping_wgsl(self._config)


class BumpMapperTracker:
    """Tracker for BumpMapper dirty state."""

    __slots__ = ("_mapper",)

    def __init__(self, mapper: BumpMapper) -> None:
        self._mapper = mapper

    @property
    def is_dirty(self) -> bool:
        return self._mapper._dirty

    @property
    def version(self) -> int:
        return self._mapper._version

    def clear(self) -> None:
        self._mapper._dirty = False


class BumpMapperMirror:
    """Mirror for BumpMapper introspection."""

    __slots__ = ("_mapper",)

    def __init__(self, mapper: BumpMapper) -> None:
        self._mapper = mapper

    @property
    def config(self) -> BumpMapConfig:
        return self._mapper._config

    @property
    def fields(self) -> Dict[str, Any]:
        return {
            "noise_frequency": self._mapper._config.noise_frequency,
            "bump_strength": self._mapper._config.bump_strength,
            "octaves": self._mapper._config.octaves,
            "lacunarity": self._mapper._config.lacunarity,
            "gain": self._mapper._config.gain,
            "gradient_dx": self._mapper._config.gradient_dx,
            "use_perlin": self._mapper._config.use_perlin,
        }


# =============================================================================
# T-DEMO-4.13: Surface Curvature Detection
# =============================================================================


class CurvatureType(Enum):
    """Type of curvature detected."""
    CONVEX = auto()   # Positive curvature (bulge outward)
    CONCAVE = auto()  # Negative curvature (indent inward)
    FLAT = auto()     # Near-zero curvature
    EDGE = auto()     # Sharp edge (high curvature magnitude)
    RIDGE = auto()    # Ridge or crease


@dataclass
class CurvatureResult:
    """
    Result of curvature detection.

    Attributes:
        value: Raw curvature value (Laplacian)
        magnitude: Absolute curvature magnitude
        curvature_type: Classified curvature type
        is_edge: True if edge detected
        is_ridge: True if ridge detected
    """
    value: float
    magnitude: float
    curvature_type: CurvatureType
    is_edge: bool = False
    is_ridge: bool = False

    @classmethod
    def from_laplacian(
        cls,
        laplacian: float,
        edge_threshold: float = DEFAULT_CURVATURE_THRESHOLD,
        ridge_threshold: float = DEFAULT_CURVATURE_THRESHOLD * 0.5,
    ) -> "CurvatureResult":
        """Create result from Laplacian value."""
        magnitude = abs(laplacian)
        is_edge = magnitude > edge_threshold
        is_ridge = magnitude > ridge_threshold and magnitude <= edge_threshold

        if magnitude < 1e-6:
            curvature_type = CurvatureType.FLAT
        elif is_edge:
            curvature_type = CurvatureType.EDGE
        elif is_ridge:
            curvature_type = CurvatureType.RIDGE
        elif laplacian > 0:
            curvature_type = CurvatureType.CONVEX
        else:
            curvature_type = CurvatureType.CONCAVE

        return cls(
            value=laplacian,
            magnitude=magnitude,
            curvature_type=curvature_type,
            is_edge=is_edge,
            is_ridge=is_ridge,
        )


@dataclass
class CurvatureConfig:
    """
    Configuration for curvature detection.

    Attributes:
        sample_distance: Distance between sample points
        edge_threshold: Threshold for edge detection
        ridge_threshold: Threshold for ridge detection
        noise_frequency: Frequency scale for noise field
        octaves: FBM octaves for noise
        lacunarity: FBM lacunarity
        gain: FBM gain
        use_perlin: Use Perlin noise
    """
    sample_distance: float = DEFAULT_SAMPLE_DISTANCE
    edge_threshold: float = DEFAULT_CURVATURE_THRESHOLD
    ridge_threshold: float = DEFAULT_CURVATURE_THRESHOLD * 0.5
    noise_frequency: float = DEFAULT_NOISE_FREQUENCY
    octaves: int = DEFAULT_OCTAVES
    lacunarity: float = DEFAULT_LACUNARITY
    gain: float = DEFAULT_GAIN
    use_perlin: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_distance <= 0:
            raise ValueError(f"sample_distance must be > 0, got {self.sample_distance}")
        if self.edge_threshold < 0:
            raise ValueError(f"edge_threshold must be >= 0, got {self.edge_threshold}")
        if self.ridge_threshold < 0:
            raise ValueError(f"ridge_threshold must be >= 0, got {self.ridge_threshold}")
        if self.noise_frequency <= 0:
            raise ValueError(f"noise_frequency must be > 0, got {self.noise_frequency}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if self.gain <= 0:
            raise ValueError(f"gain must be > 0, got {self.gain}")


def compute_laplacian(
    p: Vec3,
    noise_func: Callable[[Vec3], float],
    sample_distance: float = DEFAULT_SAMPLE_DISTANCE,
) -> float:
    """
    Compute Laplacian of noise field at position p.

    The Laplacian is the sum of second derivatives:
        Laplacian(f) = d2f/dx2 + d2f/dy2 + d2f/dz2

    Using central differences for second derivative:
        d2f/dx2 = (f(p+dx) + f(p-dx) - 2*f(p)) / dx^2

    Total: 6 samples for +/- on each axis plus center.

    Args:
        p: Position to evaluate
        noise_func: Noise function
        sample_distance: Distance between samples

    Returns:
        Laplacian value (positive = convex, negative = concave)
    """
    h = sample_distance
    h2 = h * h

    # Center value
    center = noise_func(p)

    # Axis offsets
    dx = Vec3(h, 0.0, 0.0)
    dy = Vec3(0.0, h, 0.0)
    dz = Vec3(0.0, 0.0, h)

    # Sample along each axis (6 samples total)
    fx_plus = noise_func(vec3_add(p, dx))
    fx_minus = noise_func(vec3_sub(p, dx))
    fy_plus = noise_func(vec3_add(p, dy))
    fy_minus = noise_func(vec3_sub(p, dy))
    fz_plus = noise_func(vec3_add(p, dz))
    fz_minus = noise_func(vec3_sub(p, dz))

    # Second derivatives via central differences
    d2x = (fx_plus + fx_minus - 2.0 * center) / h2
    d2y = (fy_plus + fy_minus - 2.0 * center) / h2
    d2z = (fz_plus + fz_minus - 2.0 * center) / h2

    # Laplacian is sum of second derivatives
    return d2x + d2y + d2z


def detect_edges(
    p: Vec3,
    noise_func: Callable[[Vec3], float],
    config: CurvatureConfig,
) -> bool:
    """
    Detect if position is on an edge (high curvature).

    Args:
        p: Position to check
        noise_func: Noise function
        config: Curvature configuration

    Returns:
        True if edge detected
    """
    laplacian = compute_laplacian(p, noise_func, config.sample_distance)
    return abs(laplacian) > config.edge_threshold


def detect_ridges(
    p: Vec3,
    noise_func: Callable[[Vec3], float],
    config: CurvatureConfig,
) -> bool:
    """
    Detect if position is on a ridge (moderate curvature).

    Args:
        p: Position to check
        noise_func: Noise function
        config: Curvature configuration

    Returns:
        True if ridge detected
    """
    laplacian = compute_laplacian(p, noise_func, config.sample_distance)
    magnitude = abs(laplacian)
    return magnitude > config.ridge_threshold and magnitude <= config.edge_threshold


class CurvatureDetector:
    """
    Curvature detector using Laplacian of noise field.

    Implements T-DEMO-4.13 with Trinity Tracker pattern for cache invalidation.

    Usage:
        >>> config = CurvatureConfig(sample_distance=0.01)
        >>> detector = CurvatureDetector(config)
        >>> result = detector.detect(position)
        >>> if result.is_edge:
        ...     # Apply edge material variation

    The detector uses 6 samples (+/- on each axis) to compute the Laplacian,
    which indicates surface curvature (convex vs concave).
    """

    def __init__(
        self,
        config: Optional[CurvatureConfig] = None,
        noise_func: Optional[Callable[[Vec3], float]] = None,
    ) -> None:
        """
        Initialize curvature detector.

        Args:
            config: Curvature detection configuration
            noise_func: Optional custom noise function
        """
        self._config = config or CurvatureConfig()
        self._noise_func = noise_func
        self._version = 0
        self._dirty = True

    @property
    def config(self) -> CurvatureConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: CurvatureConfig) -> None:
        """Set configuration and mark dirty."""
        self._config = value
        self._dirty = True
        self._version += 1

    @property
    def tracker(self) -> "CurvatureDetectorTracker":
        """Get tracker for dirty checking."""
        return CurvatureDetectorTracker(self)

    @property
    def mirror(self) -> "CurvatureDetectorMirror":
        """Get mirror for introspection."""
        return CurvatureDetectorMirror(self)

    def _get_noise_func(self) -> Callable[[Vec3], float]:
        """Get noise function, creating default if needed."""
        if self._noise_func is not None:
            return self._noise_func

        def default_noise(p: Vec3) -> float:
            scaled = vec3_mul(p, self._config.noise_frequency)
            return fbm_3d(
                scaled,
                octaves=self._config.octaves,
                lacunarity=self._config.lacunarity,
                gain=self._config.gain,
                use_perlin=self._config.use_perlin,
            )
        return default_noise

    def detect(self, position: Vec3) -> CurvatureResult:
        """
        Detect curvature at given position.

        Args:
            position: World position to analyze

        Returns:
            CurvatureResult with curvature value and classification
        """
        noise_func = self._get_noise_func()
        laplacian = compute_laplacian(position, noise_func, self._config.sample_distance)

        return CurvatureResult.from_laplacian(
            laplacian,
            edge_threshold=self._config.edge_threshold,
            ridge_threshold=self._config.ridge_threshold,
        )

    def detect_batch(self, positions: Sequence[Vec3]) -> List[CurvatureResult]:
        """
        Detect curvature at multiple positions.

        Args:
            positions: List of world positions

        Returns:
            List of CurvatureResult objects
        """
        return [self.detect(p) for p in positions]

    def is_edge(self, position: Vec3) -> bool:
        """Check if position is on an edge."""
        return self.detect(position).is_edge

    def is_ridge(self, position: Vec3) -> bool:
        """Check if position is on a ridge."""
        return self.detect(position).is_ridge

    def is_convex(self, position: Vec3) -> bool:
        """Check if position is on a convex surface."""
        return self.detect(position).curvature_type == CurvatureType.CONVEX

    def is_concave(self, position: Vec3) -> bool:
        """Check if position is on a concave surface."""
        return self.detect(position).curvature_type == CurvatureType.CONCAVE

    def get_curvature_value(self, position: Vec3) -> float:
        """Get raw curvature value (Laplacian)."""
        noise_func = self._get_noise_func()
        return compute_laplacian(position, noise_func, self._config.sample_distance)

    def to_wgsl(self) -> str:
        """Generate WGSL code for this curvature detector."""
        return generate_curvature_wgsl(self._config)


class CurvatureDetectorTracker:
    """Tracker for CurvatureDetector dirty state."""

    __slots__ = ("_detector",)

    def __init__(self, detector: CurvatureDetector) -> None:
        self._detector = detector

    @property
    def is_dirty(self) -> bool:
        return self._detector._dirty

    @property
    def version(self) -> int:
        return self._detector._version

    def clear(self) -> None:
        self._detector._dirty = False


class CurvatureDetectorMirror:
    """Mirror for CurvatureDetector introspection."""

    __slots__ = ("_detector",)

    def __init__(self, detector: CurvatureDetector) -> None:
        self._detector = detector

    @property
    def config(self) -> CurvatureConfig:
        return self._detector._config

    @property
    def fields(self) -> Dict[str, Any]:
        return {
            "sample_distance": self._detector._config.sample_distance,
            "edge_threshold": self._detector._config.edge_threshold,
            "ridge_threshold": self._detector._config.ridge_threshold,
            "noise_frequency": self._detector._config.noise_frequency,
            "octaves": self._detector._config.octaves,
            "lacunarity": self._detector._config.lacunarity,
            "gain": self._detector._config.gain,
            "use_perlin": self._detector._config.use_perlin,
        }


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_bump_mapping_wgsl(config: Optional[BumpMapConfig] = None) -> str:
    """
    Generate WGSL code for bump mapping.

    Args:
        config: Bump mapping configuration (uses defaults if None)

    Returns:
        WGSL shader code string
    """
    cfg = config or BumpMapConfig()

    return f'''// T-DEMO-4.12: Bump Mapping from Noise Gradients
// Auto-generated by surface_detail.py

// Configuration
const BUMP_NOISE_FREQUENCY: f32 = {cfg.noise_frequency:.6f};
const BUMP_STRENGTH: f32 = {cfg.bump_strength:.6f};
const BUMP_OCTAVES: u32 = {cfg.octaves}u;
const BUMP_LACUNARITY: f32 = {cfg.lacunarity:.6f};
const BUMP_GAIN: f32 = {cfg.gain:.6f};
const BUMP_GRADIENT_DX: f32 = {cfg.gradient_dx:.8f};

// Hash function for noise
fn hash31(p: vec3<f32>) -> f32 {{
    var q = vec3<f32>(
        fract(p.x * 0.1031),
        fract(p.y * 0.1030),
        fract(p.z * 0.0973)
    );
    let d = q.x * (q.x + 33.33) + q.y * (q.y + 33.33) + q.z * (q.z + 33.33);
    q = q + vec3<f32>(d);
    return fract(q.x * q.y * q.z);
}}

// Smoothstep fade curve
fn smoothstep_fade(t: f32) -> f32 {{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}}

// 3D value noise [-1, 1]
fn value_noise_3d(p: vec3<f32>) -> f32 {{
    let i = floor(p);
    let f = p - i;
    let u = vec3<f32>(
        smoothstep_fade(f.x),
        smoothstep_fade(f.y),
        smoothstep_fade(f.z)
    );

    let a = hash31(i + vec3<f32>(0.0, 0.0, 0.0)) * 2.0 - 1.0;
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0)) * 2.0 - 1.0;
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0)) * 2.0 - 1.0;
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0)) * 2.0 - 1.0;
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0)) * 2.0 - 1.0;
    let f_val = hash31(i + vec3<f32>(1.0, 0.0, 1.0)) * 2.0 - 1.0;
    let g = hash31(i + vec3<f32>(0.0, 1.0, 1.0)) * 2.0 - 1.0;
    let h = hash31(i + vec3<f32>(1.0, 1.0, 1.0)) * 2.0 - 1.0;

    return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, f_val, u.x), mix(g, h, u.x), u.y),
        u.z
    );
}}

// FBM (Fractal Brownian Motion)
fn fbm_3d(p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32) -> f32 {{
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amplitude = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {{
        value += amplitude * value_noise_3d(p * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }}

    return select(value / max_amplitude, 0.0, max_amplitude < 1e-8);
}}

// Compute noise gradient using central differences
fn compute_noise_gradient(p: vec3<f32>) -> vec3<f32> {{
    let dx = vec3<f32>(BUMP_GRADIENT_DX, 0.0, 0.0);
    let dy = vec3<f32>(0.0, BUMP_GRADIENT_DX, 0.0);
    let dz = vec3<f32>(0.0, 0.0, BUMP_GRADIENT_DX);

    let inv_2dx = 1.0 / (2.0 * BUMP_GRADIENT_DX);

    let grad_x = (fbm_3d(p + dx, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)
                - fbm_3d(p - dx, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)) * inv_2dx;
    let grad_y = (fbm_3d(p + dy, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)
                - fbm_3d(p - dy, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)) * inv_2dx;
    let grad_z = (fbm_3d(p + dz, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)
                - fbm_3d(p - dz, BUMP_OCTAVES, BUMP_LACUNARITY, BUMP_GAIN)) * inv_2dx;

    return vec3<f32>(grad_x, grad_y, grad_z);
}}

// Compute bump-mapped normal
fn compute_bump_normal(normal: vec3<f32>, position: vec3<f32>) -> vec3<f32> {{
    let scaled_pos = position * BUMP_NOISE_FREQUENCY;
    let gradient = compute_noise_gradient(scaled_pos);
    let perturbed = normal - gradient * BUMP_STRENGTH;

    let len = length(perturbed);
    if (len < 1e-10) {{
        return normal;
    }}
    return perturbed / len;
}}
'''


def generate_curvature_wgsl(config: Optional[CurvatureConfig] = None) -> str:
    """
    Generate WGSL code for curvature detection.

    Args:
        config: Curvature configuration (uses defaults if None)

    Returns:
        WGSL shader code string
    """
    cfg = config or CurvatureConfig()

    return f'''// T-DEMO-4.13: Surface Curvature Detection
// Auto-generated by surface_detail.py

// Configuration
const CURVATURE_SAMPLE_DISTANCE: f32 = {cfg.sample_distance:.8f};
const CURVATURE_EDGE_THRESHOLD: f32 = {cfg.edge_threshold:.6f};
const CURVATURE_RIDGE_THRESHOLD: f32 = {cfg.ridge_threshold:.6f};
const CURVATURE_NOISE_FREQUENCY: f32 = {cfg.noise_frequency:.6f};
const CURVATURE_OCTAVES: u32 = {cfg.octaves}u;
const CURVATURE_LACUNARITY: f32 = {cfg.lacunarity:.6f};
const CURVATURE_GAIN: f32 = {cfg.gain:.6f};

// Curvature type constants
const CURVATURE_TYPE_CONVEX: u32 = 0u;
const CURVATURE_TYPE_CONCAVE: u32 = 1u;
const CURVATURE_TYPE_FLAT: u32 = 2u;
const CURVATURE_TYPE_EDGE: u32 = 3u;
const CURVATURE_TYPE_RIDGE: u32 = 4u;

// Curvature result structure
struct CurvatureResult {{
    value: f32,
    magnitude: f32,
    curvature_type: u32,
    is_edge: bool,
    is_ridge: bool,
}};

// Compute Laplacian of noise field (uses 6 samples)
fn compute_laplacian(p: vec3<f32>) -> f32 {{
    let h = CURVATURE_SAMPLE_DISTANCE;
    let h2 = h * h;

    let scaled_p = p * CURVATURE_NOISE_FREQUENCY;

    // Center value
    let center = fbm_3d(scaled_p, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);

    // Axis offsets
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);

    // Sample along each axis
    let fx_plus = fbm_3d(scaled_p + dx, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);
    let fx_minus = fbm_3d(scaled_p - dx, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);
    let fy_plus = fbm_3d(scaled_p + dy, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);
    let fy_minus = fbm_3d(scaled_p - dy, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);
    let fz_plus = fbm_3d(scaled_p + dz, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);
    let fz_minus = fbm_3d(scaled_p - dz, CURVATURE_OCTAVES, CURVATURE_LACUNARITY, CURVATURE_GAIN);

    // Second derivatives
    let d2x = (fx_plus + fx_minus - 2.0 * center) / h2;
    let d2y = (fy_plus + fy_minus - 2.0 * center) / h2;
    let d2z = (fz_plus + fz_minus - 2.0 * center) / h2;

    return d2x + d2y + d2z;
}}

// Detect curvature at position
fn detect_curvature(position: vec3<f32>) -> CurvatureResult {{
    let laplacian = compute_laplacian(position);
    let magnitude = abs(laplacian);

    let is_edge = magnitude > CURVATURE_EDGE_THRESHOLD;
    let is_ridge = magnitude > CURVATURE_RIDGE_THRESHOLD && magnitude <= CURVATURE_EDGE_THRESHOLD;

    var curvature_type: u32;
    if (magnitude < 1e-6) {{
        curvature_type = CURVATURE_TYPE_FLAT;
    }} else if (is_edge) {{
        curvature_type = CURVATURE_TYPE_EDGE;
    }} else if (is_ridge) {{
        curvature_type = CURVATURE_TYPE_RIDGE;
    }} else if (laplacian > 0.0) {{
        curvature_type = CURVATURE_TYPE_CONVEX;
    }} else {{
        curvature_type = CURVATURE_TYPE_CONCAVE;
    }}

    return CurvatureResult(laplacian, magnitude, curvature_type, is_edge, is_ridge);
}}

// Check if position is on an edge
fn is_edge(position: vec3<f32>) -> bool {{
    let laplacian = compute_laplacian(position);
    return abs(laplacian) > CURVATURE_EDGE_THRESHOLD;
}}

// Check if position is on a ridge
fn is_ridge(position: vec3<f32>) -> bool {{
    let laplacian = compute_laplacian(position);
    let magnitude = abs(laplacian);
    return magnitude > CURVATURE_RIDGE_THRESHOLD && magnitude <= CURVATURE_EDGE_THRESHOLD;
}}

// Check if surface is convex at position
fn is_convex(position: vec3<f32>) -> bool {{
    let laplacian = compute_laplacian(position);
    return laplacian > 0.0;
}}

// Check if surface is concave at position
fn is_concave(position: vec3<f32>) -> bool {{
    let laplacian = compute_laplacian(position);
    return laplacian < 0.0;
}}
'''
