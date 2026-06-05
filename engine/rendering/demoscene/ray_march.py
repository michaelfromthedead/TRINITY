"""
TRINITY Demoscene Ray Marching Module (T-DEMO-3.2, T-DEMO-3.3, T-DEMO-3.4)

This module provides ray marching utilities for SDF rendering:

T-DEMO-3.2: Ray Marching Loop (Sphere Tracing)
- Sphere tracing algorithm with configurable max_steps and max_distance
- Returns hit position, normal, material_id, steps_taken
- HitResult dataclass with full hit information

T-DEMO-3.3: Perceptual Termination Criterion
- Adaptive epsilon scaling based on distance and FOV
- Distant objects terminate with larger epsilon (performance optimization)
- Maintains consistent visual quality at all distances

T-DEMO-3.4: Normal Estimation (Central Differences)
- 6-point central differences stencil for accurate normals
- WGSL code generation for GPU-side normal estimation
- Unit-length normals pointing outward from surface

Reference:
- Inigo Quilez normal estimation: https://iquilezles.org/articles/normalsSDF/
- Inigo Quilez ray marching: https://iquilezles.org/articles/distfunctions/
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

from .sdf_ast import SceneNode, SDFNode, Vec3


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # T-DEMO-3.2: Ray Marching Loop
    "HitResult",
    "MarchResultType",
    "SphereTracer",
    "march_ray",
    # T-DEMO-3.3: Perceptual Termination
    "epsilon_at_distance",
    "PerceptualEpsilonConfig",
    "RayMarchConfig",
    "RayMarcher",
    "RayMarchResult",
    # T-DEMO-3.4: Normal Estimation
    "estimate_normal",
    "NormalEstimationConfig",
    "NormalEstimator",
    # WGSL Generation
    "generate_epsilon_wgsl",
    "generate_normal_estimation_wgsl",
    "generate_ray_march_wgsl",
    "generate_ray_march_struct_wgsl",
]

# =============================================================================
# T-DEMO-3.2: Ray Marching Loop (Sphere Tracing)
# =============================================================================


class MarchResultType(Enum):
    """Result type for ray marching termination."""
    HIT = auto()        # Ray hit a surface (SDF < epsilon)
    MISS = auto()       # Ray exceeded max_distance without hit
    MAX_STEPS = auto()  # Ray exceeded max_steps without resolution


@dataclass
class HitResult:
    """
    Result of ray marching through an SDF scene (T-DEMO-3.2).

    This dataclass captures the full result of sphere tracing:
    - Whether a surface was hit
    - The world-space position of the hit
    - The distance from ray origin to hit point
    - Number of marching iterations taken
    - Material ID at the hit point

    Attributes:
        hit: True if the ray hit a surface (SDF < epsilon).
        position: Hit position in world space (valid only if hit=True).
        distance: Distance from ray origin to hit point.
        steps: Number of marching steps taken.
        material_id: Material index at the hit point.
        result_type: Detailed termination reason.

    Example:
        >>> result = HitResult(
        ...     hit=True,
        ...     position=Vec3(0.0, 0.0, 4.0),
        ...     distance=4.0,
        ...     steps=42,
        ...     material_id=0,
        ... )
        >>> result.hit
        True
        >>> result.distance
        4.0
    """
    hit: bool
    position: Vec3
    distance: float
    steps: int
    material_id: int
    result_type: MarchResultType = MarchResultType.MISS

    @classmethod
    def surface_hit(
        cls,
        position: Vec3,
        distance: float,
        steps: int,
        material_id: int = 0,
    ) -> "HitResult":
        """Create a successful hit result."""
        return cls(
            hit=True,
            position=position,
            distance=distance,
            steps=steps,
            material_id=material_id,
            result_type=MarchResultType.HIT,
        )

    @classmethod
    def miss(
        cls,
        distance: float,
        steps: int,
        result_type: MarchResultType = MarchResultType.MISS,
    ) -> "HitResult":
        """Create a miss result (no surface hit)."""
        return cls(
            hit=False,
            position=Vec3(0.0, 0.0, 0.0),
            distance=distance,
            steps=steps,
            material_id=0,
            result_type=result_type,
        )

    def __repr__(self) -> str:
        if self.hit:
            pos = self.position
            return (
                f"HitResult(hit=True, pos=({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}), "
                f"dist={self.distance:.4f}, steps={self.steps}, mat={self.material_id})"
            )
        return (
            f"HitResult(hit=False, dist={self.distance:.4f}, "
            f"steps={self.steps}, type={self.result_type.name})"
        )


# Type alias for SDF function that returns (distance, material_id)
SDFFuncWithMaterial = Callable[[Vec3], Tuple[float, int]]


class SphereTracer:
    """
    Sphere tracing ray marcher for SDF scenes (T-DEMO-3.2).

    Sphere tracing uses the SDF value at each step as a safe step distance.
    Since the SDF returns the minimum distance to any surface, stepping
    by that amount is guaranteed not to miss any surfaces.

    The algorithm terminates when:
      1. SDF value < epsilon: Surface hit detected
      2. Total distance > max_distance: Ray escaped the scene
      3. Step count > max_steps: Maximum iterations reached

    Attributes:
        max_steps: Maximum number of marching iterations.
        max_distance: Maximum ray travel distance.
        epsilon: Surface hit threshold.

    Example:
        >>> tracer = SphereTracer(max_steps=64, max_distance=50.0)
        >>> def sphere_sdf(p):
        ...     return (p.length() - 1.0, 0)
        >>> result = tracer.march(Vec3(0, 0, 5), Vec3(0, 0, -1), sphere_sdf)
        >>> result.hit
        True
    """

    def __init__(
        self,
        max_steps: int = 256,
        max_distance: float = 100.0,
        epsilon: float = 0.001,
    ) -> None:
        """Initialize sphere tracer with marching parameters."""
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.max_steps = max_steps
        self.max_distance = max_distance
        self.epsilon = epsilon
        self._step_history: List[float] = []

    def march(
        self,
        origin: Vec3,
        direction: Vec3,
        sdf: SDFFuncWithMaterial,
    ) -> HitResult:
        """
        March a ray through the SDF scene using sphere tracing.

        Args:
            origin: Ray origin in world space.
            direction: Ray direction (should be normalized).
            sdf: SDF function returning (distance, material_id).

        Returns:
            HitResult with hit information.
        """
        # Normalize direction
        dir_len = direction.length()
        if dir_len < 1e-10:
            return HitResult.miss(0.0, 0, MarchResultType.MISS)
        direction = direction.normalized()

        # Clear step history
        self._step_history.clear()

        t: float = 0.0  # Distance traveled along ray
        material_id: int = 0

        for step in range(self.max_steps):
            # Current position along ray
            p = Vec3(
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t,
            )

            # Evaluate SDF
            d, material_id = sdf(p)
            self._step_history.append(d)

            # Check for surface hit
            if d < self.epsilon:
                return HitResult.surface_hit(
                    position=p,
                    distance=t,
                    steps=step + 1,
                    material_id=material_id,
                )

            # Advance along ray
            t += d

            # Check for escape
            if t > self.max_distance:
                return HitResult.miss(t, step + 1, MarchResultType.MISS)

        # Max steps reached
        return HitResult.miss(t, self.max_steps, MarchResultType.MAX_STEPS)

    def march_with_normal(
        self,
        origin: Vec3,
        direction: Vec3,
        sdf: SDFFuncWithMaterial,
        normal_epsilon: float = 0.0001,
    ) -> Tuple[HitResult, Optional[Vec3]]:
        """
        March ray and compute surface normal at hit point.

        Args:
            origin: Ray origin.
            direction: Ray direction.
            sdf: SDF function.
            normal_epsilon: Epsilon for normal computation.

        Returns:
            Tuple of (HitResult, normal_or_None).
        """
        result = self.march(origin, direction, sdf)

        if not result.hit:
            return (result, None)

        # Compute normal using central differences
        p = result.position
        dx = sdf(Vec3(p.x + normal_epsilon, p.y, p.z))[0] - sdf(Vec3(p.x - normal_epsilon, p.y, p.z))[0]
        dy = sdf(Vec3(p.x, p.y + normal_epsilon, p.z))[0] - sdf(Vec3(p.x, p.y - normal_epsilon, p.z))[0]
        dz = sdf(Vec3(p.x, p.y, p.z + normal_epsilon))[0] - sdf(Vec3(p.x, p.y, p.z - normal_epsilon))[0]

        normal = Vec3(dx, dy, dz).normalized()
        return (result, normal)

    @property
    def step_history(self) -> List[float]:
        """Get SDF values from the last march (for debugging)."""
        return self._step_history.copy()


def march_ray(
    origin: Vec3,
    direction: Vec3,
    sdf: SDFFuncWithMaterial,
    max_steps: int = 256,
    max_distance: float = 100.0,
    epsilon: float = 0.001,
) -> HitResult:
    """
    Convenience function for single ray marching.

    Args:
        origin: Ray origin in world space.
        direction: Ray direction (will be normalized).
        sdf: SDF function returning (distance, material_id).
        max_steps: Maximum iterations.
        max_distance: Maximum ray distance.
        epsilon: Surface hit threshold.

    Returns:
        HitResult with hit information.

    Example:
        >>> def sphere_sdf(p):
        ...     return (p.length() - 1.0, 0)
        >>> result = march_ray(Vec3(0, 0, 5), Vec3(0, 0, -1), sphere_sdf)
        >>> result.hit
        True
    """
    tracer = SphereTracer(max_steps, max_distance, epsilon)
    return tracer.march(origin, direction, sdf)


def generate_ray_march_struct_wgsl() -> str:
    """
    Generate WGSL struct definitions for ray marching (T-DEMO-3.2).

    Returns:
        WGSL code for RayHit struct definition.
    """
    return """\
/// Result of ray marching through the scene (T-DEMO-3.2).
struct RayHit {
    /// True if the ray hit a surface.
    hit: bool,
    /// Hit position in world space (valid only if hit=true).
    position: vec3<f32>,
    /// Distance from ray origin to hit point.
    distance: f32,
    /// Material ID at the hit point.
    material_id: u32,
    /// Number of marching steps taken.
    steps: u32,
}

/// Creates a miss result.
fn ray_hit_miss(steps: u32) -> RayHit {
    return RayHit(false, vec3<f32>(0.0), 0.0, 0u, steps);
}

/// Creates a hit result.
fn ray_hit_surface(
    position: vec3<f32>,
    distance: f32,
    material_id: u32,
    steps: u32,
) -> RayHit {
    return RayHit(true, position, distance, material_id, steps);
}
"""


# Default epsilon for surface detection
DEFAULT_EPSILON = 0.001

# Default pixel scale factor for perceptual epsilon
DEFAULT_PIXEL_SCALE = 0.5

# Default FOV in radians (60 degrees)
DEFAULT_FOV = math.radians(60.0)

# Maximum epsilon to prevent too-early termination
MAX_EPSILON = 0.1

# Minimum epsilon to ensure some precision
MIN_EPSILON = 1e-6


# =============================================================================
# T-DEMO-3.3: Perceptual Termination Criterion
# =============================================================================

def epsilon_at_distance(
    base_epsilon: float,
    distance: float,
    fov: float,
    pixel_scale: float = DEFAULT_PIXEL_SCALE,
    min_epsilon: float = MIN_EPSILON,
    max_epsilon: float = MAX_EPSILON,
) -> float:
    """
    Calculate adaptive epsilon based on distance for perceptual termination.

    The epsilon scales with distance to maintain consistent visual quality:
    objects far from the camera can use a larger epsilon without visible
    quality loss, improving performance by reducing ray march steps.

    Formula: epsilon = base_epsilon * (1.0 + distance * tan(fov/2) * pixel_scale)

    Args:
        base_epsilon: Base epsilon value for close objects
        distance: Current ray distance from camera
        fov: Field of view in radians
        pixel_scale: Scale factor for pixel-based termination (default 0.5)
        min_epsilon: Minimum allowed epsilon value
        max_epsilon: Maximum allowed epsilon value

    Returns:
        Scaled epsilon appropriate for the given distance

    Example:
        >>> eps = epsilon_at_distance(0.001, 10.0, math.radians(60))
        >>> eps > 0.001  # Epsilon increases with distance
        True

    Reference:
        Based on Inigo Quilez's adaptive epsilon technique for
        maintaining consistent visual quality in ray marching.
    """
    if distance < 0.0:
        raise ValueError(f"Distance must be non-negative, got {distance}")
    if base_epsilon <= 0.0:
        raise ValueError(f"Base epsilon must be positive, got {base_epsilon}")
    if fov <= 0.0 or fov >= math.pi:
        raise ValueError(f"FOV must be in (0, pi) radians, got {fov}")
    if pixel_scale < 0.0:
        raise ValueError(f"Pixel scale must be non-negative, got {pixel_scale}")

    # Calculate tangent of half FOV
    tan_half_fov = math.tan(fov / 2.0)

    # Scale epsilon with distance
    # The factor (1 + distance * tan(fov/2) * pixel_scale) grows linearly with distance
    scale_factor = 1.0 + distance * tan_half_fov * pixel_scale

    # Apply scaling
    scaled_epsilon = base_epsilon * scale_factor

    # Clamp to valid range
    return max(min_epsilon, min(scaled_epsilon, max_epsilon))


@dataclass(frozen=True)
class PerceptualEpsilonConfig:
    """Configuration for perceptual epsilon scaling.

    Attributes:
        base_epsilon: Base epsilon for near objects
        fov: Field of view in radians
        pixel_scale: Scale factor for distance-based scaling
        min_epsilon: Minimum epsilon clamp
        max_epsilon: Maximum epsilon clamp
    """
    base_epsilon: float = DEFAULT_EPSILON
    fov: float = DEFAULT_FOV
    pixel_scale: float = DEFAULT_PIXEL_SCALE
    min_epsilon: float = MIN_EPSILON
    max_epsilon: float = MAX_EPSILON

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.base_epsilon <= 0.0:
            raise ValueError(f"base_epsilon must be positive, got {self.base_epsilon}")
        if self.fov <= 0.0 or self.fov >= math.pi:
            raise ValueError(f"fov must be in (0, pi), got {self.fov}")
        if self.pixel_scale < 0.0:
            raise ValueError(f"pixel_scale must be non-negative, got {self.pixel_scale}")
        if self.min_epsilon <= 0.0:
            raise ValueError(f"min_epsilon must be positive, got {self.min_epsilon}")
        if self.max_epsilon <= self.min_epsilon:
            raise ValueError(
                f"max_epsilon ({self.max_epsilon}) must be > min_epsilon ({self.min_epsilon})"
            )

    def compute(self, distance: float) -> float:
        """Compute epsilon for a given distance."""
        return epsilon_at_distance(
            self.base_epsilon,
            distance,
            self.fov,
            self.pixel_scale,
            self.min_epsilon,
            self.max_epsilon,
        )


# =============================================================================
# T-DEMO-3.4: Normal Estimation (Central Differences)
# =============================================================================

# Type alias for SDF evaluation function
SDFFunc = Callable[[Vec3], float]


def estimate_normal(
    p: Vec3,
    sdf: SDFFunc,
    epsilon: float = DEFAULT_EPSILON,
) -> Vec3:
    """
    Estimate surface normal using 6-point central differences.

    Uses the gradient of the SDF to compute the surface normal:
    n = normalize(vec3(
        sdf(p+ex) - sdf(p-ex),
        sdf(p+ey) - sdf(p-ey),
        sdf(p+ez) - sdf(p-ez)
    ))

    Args:
        p: Surface position to estimate normal at
        sdf: Signed distance function callable
        epsilon: Sampling offset for central differences

    Returns:
        Unit-length normal vector pointing outward from surface

    Raises:
        ValueError: If epsilon is not positive
        RuntimeError: If normal computation results in zero-length vector

    Example:
        >>> def sphere_sdf(p: Vec3) -> float:
        ...     return p.length() - 1.0
        >>> normal = estimate_normal(Vec3(1.0, 0.0, 0.0), sphere_sdf)
        >>> abs(normal.x - 1.0) < 1e-5
        True

    Reference:
        Inigo Quilez: https://iquilezles.org/articles/normalsSDF/
    """
    if epsilon <= 0.0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")

    # 6-point central differences stencil
    dx = sdf(Vec3(p.x + epsilon, p.y, p.z)) - sdf(Vec3(p.x - epsilon, p.y, p.z))
    dy = sdf(Vec3(p.x, p.y + epsilon, p.z)) - sdf(Vec3(p.x, p.y - epsilon, p.z))
    dz = sdf(Vec3(p.x, p.y, p.z + epsilon)) - sdf(Vec3(p.x, p.y, p.z - epsilon))

    normal = Vec3(dx, dy, dz)
    length = normal.length()

    if length < 1e-10:
        raise RuntimeError(
            f"Zero-length normal at position {p}. "
            "This may indicate a degenerate SDF or sampling point."
        )

    return normal.normalized()


@dataclass
class NormalEstimationConfig:
    """Configuration for normal estimation.

    Attributes:
        epsilon: Sampling offset for central differences
        use_tetrahedron: If True, use 4-point tetrahedron stencil (faster)
    """
    epsilon: float = DEFAULT_EPSILON
    use_tetrahedron: bool = False

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


class NormalEstimator:
    """Estimates surface normals for SDF scenes.

    Provides both standard 6-point central differences and
    optional 4-point tetrahedron stencil for faster computation.

    Example:
        >>> estimator = NormalEstimator(epsilon=0.001)
        >>> def sphere_sdf(p: Vec3) -> float:
        ...     return p.length() - 1.0
        >>> normal = estimator.estimate(Vec3(1.0, 0.0, 0.0), sphere_sdf)
    """

    def __init__(self, config: Optional[NormalEstimationConfig] = None) -> None:
        """Initialize with configuration."""
        self.config = config or NormalEstimationConfig()

    @property
    def epsilon(self) -> float:
        """Get the sampling epsilon."""
        return self.config.epsilon

    def estimate(self, p: Vec3, sdf: SDFFunc) -> Vec3:
        """Estimate normal at position p using configured method."""
        if self.config.use_tetrahedron:
            return self._estimate_tetrahedron(p, sdf)
        return estimate_normal(p, sdf, self.config.epsilon)

    def _estimate_tetrahedron(self, p: Vec3, sdf: SDFFunc) -> Vec3:
        """
        4-point tetrahedron stencil normal estimation.

        Uses vertices of a tetrahedron for faster computation
        (4 samples instead of 6), with slight reduction in accuracy.
        """
        e = self.config.epsilon

        # Tetrahedron vertices (normalized to unit length, scaled by epsilon)
        k = e * 0.5773502691896258  # 1/sqrt(3)

        # Sample at tetrahedron vertices
        s1 = sdf(Vec3(p.x + k, p.y + k, p.z + k))
        s2 = sdf(Vec3(p.x + k, p.y - k, p.z - k))
        s3 = sdf(Vec3(p.x - k, p.y + k, p.z - k))
        s4 = sdf(Vec3(p.x - k, p.y - k, p.z + k))

        # Compute gradient
        nx = s1 + s2 - s3 - s4
        ny = s1 - s2 + s3 - s4
        nz = s1 - s2 - s3 + s4

        normal = Vec3(nx, ny, nz)
        length = normal.length()

        if length < 1e-10:
            raise RuntimeError(f"Zero-length normal at position {p}")

        return normal.normalized()


# =============================================================================
# Ray Marching Integration
# =============================================================================

@dataclass
class RayMarchConfig:
    """Configuration for ray marching.

    Attributes:
        max_steps: Maximum number of marching steps
        max_distance: Maximum ray travel distance
        base_epsilon: Base surface detection epsilon
        use_perceptual_epsilon: Enable distance-based epsilon scaling
        perceptual_config: Configuration for perceptual epsilon
        normal_config: Configuration for normal estimation
    """
    max_steps: int = 256
    max_distance: float = 100.0
    base_epsilon: float = DEFAULT_EPSILON
    use_perceptual_epsilon: bool = True
    perceptual_config: Optional[PerceptualEpsilonConfig] = None
    normal_config: Optional[NormalEstimationConfig] = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")
        if self.base_epsilon <= 0.0:
            raise ValueError(f"base_epsilon must be positive, got {self.base_epsilon}")

        # Initialize sub-configs if not provided
        if self.perceptual_config is None:
            self.perceptual_config = PerceptualEpsilonConfig(
                base_epsilon=self.base_epsilon
            )
        if self.normal_config is None:
            self.normal_config = NormalEstimationConfig(
                epsilon=self.base_epsilon
            )

    def get_epsilon(self, distance: float) -> float:
        """Get epsilon for a given distance."""
        if self.use_perceptual_epsilon and self.perceptual_config:
            return self.perceptual_config.compute(distance)
        return self.base_epsilon


@dataclass
class RayMarchResult:
    """Result of a ray march operation.

    Attributes:
        hit: Whether a surface was hit
        position: Hit position (if hit)
        normal: Surface normal at hit (if hit)
        distance: Ray travel distance
        steps: Number of steps taken
        material_id: Material ID at hit (if applicable)
        epsilon_used: Final epsilon value used for termination
    """
    hit: bool = False
    position: Optional[Vec3] = None
    normal: Optional[Vec3] = None
    distance: float = 0.0
    steps: int = 0
    material_id: int = 0
    epsilon_used: float = DEFAULT_EPSILON

    @property
    def converged(self) -> bool:
        """True if march terminated due to hitting a surface."""
        return self.hit

    @property
    def exhausted(self) -> bool:
        """True if march exhausted max steps without hitting."""
        return not self.hit and self.steps > 0


class RayMarcher:
    """
    CPU-side ray marcher for SDF scenes with perceptual termination.

    Implements sphere tracing with:
    - Adaptive epsilon scaling based on distance
    - Central differences normal estimation
    - Configurable termination criteria

    Example:
        >>> marcher = RayMarcher(config=RayMarchConfig(max_steps=128))
        >>> scene = create_scene()  # Your scene creation
        >>> result = marcher.march(origin, direction, scene)
        >>> if result.hit:
        ...     print(f"Hit at {result.position}, normal {result.normal}")
    """

    def __init__(self, config: Optional[RayMarchConfig] = None) -> None:
        """Initialize ray marcher with configuration."""
        self.config = config or RayMarchConfig()
        self._normal_estimator = NormalEstimator(self.config.normal_config)

    @property
    def max_steps(self) -> int:
        """Maximum number of ray march steps."""
        return self.config.max_steps

    @property
    def max_distance(self) -> float:
        """Maximum ray travel distance."""
        return self.config.max_distance

    def march(
        self,
        origin: Vec3,
        direction: Vec3,
        sdf: SDFFunc,
        *,
        initial_distance: float = 0.0,
    ) -> RayMarchResult:
        """
        March a ray through the SDF until hit or termination.

        Args:
            origin: Ray origin
            direction: Ray direction (should be normalized)
            sdf: Signed distance function callable
            initial_distance: Starting distance along ray (for secondary rays)

        Returns:
            RayMarchResult with hit information
        """
        # Normalize direction
        dir_len = direction.length()
        if dir_len < 1e-10:
            return RayMarchResult(hit=False, steps=0)
        direction = direction.normalized()

        t = initial_distance
        steps = 0
        epsilon_used = self.config.base_epsilon

        for step in range(self.config.max_steps):
            steps = step + 1

            # Current position
            p = Vec3(
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t,
            )

            # Evaluate SDF
            d = sdf(p)

            # Get perceptual epsilon
            epsilon_used = self.config.get_epsilon(t)

            # Check for hit
            if d < epsilon_used:
                # Estimate normal
                try:
                    normal = self._normal_estimator.estimate(p, sdf)
                except RuntimeError:
                    normal = Vec3(0.0, 1.0, 0.0)  # Fallback normal

                return RayMarchResult(
                    hit=True,
                    position=p,
                    normal=normal,
                    distance=t,
                    steps=steps,
                    epsilon_used=epsilon_used,
                )

            # Advance ray
            t += d

            # Check for max distance
            if t > self.config.max_distance:
                return RayMarchResult(
                    hit=False,
                    distance=t,
                    steps=steps,
                    epsilon_used=epsilon_used,
                )

        # Exhausted steps
        return RayMarchResult(
            hit=False,
            distance=t,
            steps=steps,
            epsilon_used=epsilon_used,
        )

    def march_scene(
        self,
        origin: Vec3,
        direction: Vec3,
        scene: SceneNode,
    ) -> RayMarchResult:
        """
        March a ray through a SceneNode.

        This is a convenience method that creates an SDF evaluator
        from a SceneNode. For repeated evaluations, prefer using
        march() with a cached SDF function.

        Args:
            origin: Ray origin
            direction: Ray direction
            scene: SceneNode to evaluate

        Returns:
            RayMarchResult with hit information
        """
        from .sdf_codegen import create_sdf_evaluator

        sdf = create_sdf_evaluator(scene)
        return self.march(origin, direction, sdf)


# =============================================================================
# WGSL Code Generation
# =============================================================================

def generate_epsilon_wgsl(config: Optional[PerceptualEpsilonConfig] = None) -> str:
    """
    Generate WGSL function for perceptual epsilon calculation.

    Args:
        config: Optional configuration for default values

    Returns:
        WGSL function source code
    """
    cfg = config or PerceptualEpsilonConfig()

    return f"""\
/// Computes adaptive epsilon based on distance for perceptual termination.
/// Distant rays can use larger epsilon without visible quality loss.
///   base_epsilon -- base epsilon for near objects
///   distance     -- current ray distance from camera
///   fov          -- field of view in radians
///   pixel_scale  -- scale factor for pixel-based termination
///   returns      -- scaled epsilon for the given distance
fn epsilon_at_distance(
    base_epsilon: f32,
    distance: f32,
    fov: f32,
    pixel_scale: f32,
) -> f32 {{
    let tan_half_fov = tan(fov * 0.5);
    let scale_factor = 1.0 + distance * tan_half_fov * pixel_scale;
    let scaled = base_epsilon * scale_factor;
    return clamp(scaled, {cfg.min_epsilon}, {cfg.max_epsilon});
}}

/// Convenience function using default FOV and pixel scale.
fn perceptual_epsilon(base_epsilon: f32, distance: f32) -> f32 {{
    return epsilon_at_distance(
        base_epsilon,
        distance,
        {cfg.fov},  // FOV in radians
        {cfg.pixel_scale}   // pixel scale
    );
}}
"""


def generate_normal_estimation_wgsl(use_tetrahedron: bool = False) -> str:
    """
    Generate WGSL function for normal estimation.

    Args:
        use_tetrahedron: If True, generate 4-point tetrahedron stencil

    Returns:
        WGSL function source code
    """
    if use_tetrahedron:
        return """\
/// Estimates surface normal using 4-point tetrahedron stencil.
/// Faster than 6-point but slightly less accurate.
///   p       -- surface position
///   epsilon -- sampling offset
///   returns -- normalized surface normal
fn estimate_normal(p: vec3<f32>, epsilon: f32) -> vec3<f32> {
    // Tetrahedron vertices scaled by epsilon
    let k = epsilon * 0.5773502691896258;  // 1/sqrt(3)

    // Sample at tetrahedron vertices
    let s1 = scene_sdf(p + vec3<f32>(k, k, k)).x;
    let s2 = scene_sdf(p + vec3<f32>(k, -k, -k)).x;
    let s3 = scene_sdf(p + vec3<f32>(-k, k, -k)).x;
    let s4 = scene_sdf(p + vec3<f32>(-k, -k, k)).x;

    // Compute gradient from tetrahedron samples
    let n = vec3<f32>(
        s1 + s2 - s3 - s4,
        s1 - s2 + s3 - s4,
        s1 - s2 - s3 + s4
    );

    return normalize(n);
}
"""
    else:
        return """\
/// Estimates surface normal using 6-point central differences.
/// Formula: n = normalize(vec3(
///     sdf(p+ex) - sdf(p-ex),
///     sdf(p+ey) - sdf(p-ey),
///     sdf(p+ez) - sdf(p-ez)
/// ))
///   p       -- surface position
///   epsilon -- sampling offset
///   returns -- normalized surface normal
fn estimate_normal(p: vec3<f32>, epsilon: f32) -> vec3<f32> {
    let e = vec2<f32>(epsilon, 0.0);
    let n = vec3<f32>(
        scene_sdf(p + e.xyy).x - scene_sdf(p - e.xyy).x,
        scene_sdf(p + e.yxy).x - scene_sdf(p - e.yxy).x,
        scene_sdf(p + e.yyx).x - scene_sdf(p - e.yyx).x
    );
    return normalize(n);
}
"""


def generate_ray_march_wgsl(
    config: Optional[RayMarchConfig] = None,
    use_perceptual_epsilon: bool = True,
) -> str:
    """
    Generate WGSL ray marching function with perceptual termination.

    Args:
        config: Ray march configuration
        use_perceptual_epsilon: Whether to use distance-based epsilon

    Returns:
        WGSL function source code
    """
    cfg = config or RayMarchConfig()

    if use_perceptual_epsilon:
        epsilon_call = "perceptual_epsilon(uniforms.epsilon, t)"
    else:
        epsilon_call = "uniforms.epsilon"

    return f"""\
/// Result of ray marching.
struct RayHit {{
    hit: bool,
    position: vec3<f32>,
    distance: f32,
    material_id: u32,
    steps: u32,
    epsilon_used: f32,
}}

/// Marches a ray through the scene using sphere tracing with perceptual termination.
/// Distant rays use larger epsilon for early termination without quality loss.
///   ro         -- ray origin
///   rd         -- ray direction (normalized)
///   max_steps  -- maximum number of marching steps
///   max_dist   -- maximum ray distance
///   epsilon    -- base surface hit threshold
///   returns    -- RayHit with hit info
fn ray_march_perceptual(
    ro: vec3<f32>,
    rd: vec3<f32>,
    max_steps: u32,
    max_dist: f32,
    base_epsilon: f32,
) -> RayHit {{
    var t: f32 = 0.0;
    var hit = RayHit(false, vec3<f32>(0.0), 0.0, 0u, 0u, base_epsilon);
    var current_epsilon: f32 = base_epsilon;

    for (var i = 0u; i < max_steps; i = i + 1u) {{
        let p = ro + rd * t;
        let result = scene_sdf(p);
        let d = result.x;
        let mat_id = u32(result.y);

        // Compute perceptual epsilon based on distance
        current_epsilon = {epsilon_call};

        if (d < current_epsilon) {{
            hit.hit = true;
            hit.position = p;
            hit.distance = t;
            hit.material_id = mat_id;
            hit.steps = i;
            hit.epsilon_used = current_epsilon;
            return hit;
        }}

        t = t + d;

        if (t > max_dist) {{
            hit.steps = i;
            hit.epsilon_used = current_epsilon;
            return hit;
        }}
    }}

    hit.steps = max_steps;
    hit.epsilon_used = current_epsilon;
    return hit;
}}

/// Standard ray march without perceptual epsilon (for comparison/debugging).
fn ray_march(
    ro: vec3<f32>,
    rd: vec3<f32>,
    max_steps: u32,
    max_dist: f32,
    epsilon: f32,
) -> RayHit {{
    var t: f32 = 0.0;
    var hit = RayHit(false, vec3<f32>(0.0), 0.0, 0u, 0u, epsilon);

    for (var i = 0u; i < max_steps; i = i + 1u) {{
        let p = ro + rd * t;
        let result = scene_sdf(p);
        let d = result.x;
        let mat_id = u32(result.y);

        if (d < epsilon) {{
            hit.hit = true;
            hit.position = p;
            hit.distance = t;
            hit.material_id = mat_id;
            hit.steps = i;
            hit.epsilon_used = epsilon;
            return hit;
        }}

        t = t + d;

        if (t > max_dist) {{
            hit.steps = i;
            return hit;
        }}
    }}

    hit.steps = max_steps;
    return hit;
}}
"""


# =============================================================================
# SDF Primitives for Testing (Reference Implementations)
# =============================================================================

def sdf_sphere(p: Vec3, radius: float = 1.0) -> float:
    """Signed distance to a sphere centered at origin."""
    return p.length() - radius


def sdf_box(p: Vec3, half_extents: Vec3) -> float:
    """Signed distance to an axis-aligned box centered at origin."""
    qx = abs(p.x) - half_extents.x
    qy = abs(p.y) - half_extents.y
    qz = abs(p.z) - half_extents.z

    # Distance outside box
    outside = Vec3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)).length()

    # Distance inside box (negative)
    inside = min(max(qx, max(qy, qz)), 0.0)

    return outside + inside


def sdf_plane(p: Vec3, normal: Vec3, distance: float = 0.0) -> float:
    """Signed distance to an infinite plane."""
    n = normal.normalized()
    return p.x * n.x + p.y * n.y + p.z * n.z + distance


def sdf_torus(p: Vec3, major_radius: float, minor_radius: float) -> float:
    """Signed distance to a torus centered at origin, axis along Y."""
    # Project to XZ plane and get distance to ring
    q_xz = math.sqrt(p.x * p.x + p.z * p.z) - major_radius
    q = Vec3(q_xz, p.y, 0.0)
    return q.length() - minor_radius


def sdf_cylinder(p: Vec3, radius: float, height: float) -> float:
    """Signed distance to a capped cylinder along Y axis."""
    # Distance in XZ plane
    d_xz = math.sqrt(p.x * p.x + p.z * p.z) - radius

    # Distance along Y
    d_y = abs(p.y) - height * 0.5

    # 2D SDF for capped shape
    dx = max(d_xz, 0.0)
    dy = max(d_y, 0.0)
    outside = math.sqrt(dx * dx + dy * dy)
    inside = min(max(d_xz, d_y), 0.0)

    return outside + inside
