"""
TRINITY Demoscene Adaptive Ray Marching Module (T-DEMO-8.4)

This module provides importance-driven SDF evaluation with adaptive step counts
based on scene complexity:

1. Gradient Magnitude Analysis
   - Measure SDF gradient magnitude at each step
   - High gradient = simple geometry (large steps)
   - Low gradient = detail/edge (small steps)

2. Adaptive Step Count
   - Simple regions: 20-40 steps
   - Complex regions: 80-128 steps

3. Per-Pixel Complexity Estimation
   - First-pass complexity map
   - Guide second-pass step allocation
   - Amortize cost across frame

Implementation Details:
  - Gradient: |grad_SDF| = length(gradient(p))
  - Step scale: step *= lerp(0.5, 2.0, gradient_mag)
  - Complexity heuristic from noise frequency

Reference:
  - Inigo Quilez gradient techniques: https://iquilezles.org/articles/normalsSDF/
  - Enhanced sphere tracing: https://iquilezles.org/articles/raymarchingdf/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from .sdf_ast import Vec3


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Complexity analysis
    "ComplexityLevel",
    "ComplexityEstimate",
    "GradientAnalyzer",
    # Adaptive marching
    "AdaptiveMarchConfig",
    "AdaptiveMarchResult",
    "AdaptiveMarcher",
    # Step scaling
    "StepScaler",
    "GradientBasedScaler",
    "DistanceBasedScaler",
    "CombinedScaler",
    # Complexity map
    "ComplexityMap",
    "ComplexityMapConfig",
    "ComplexityMapGenerator",
    # WGSL generation
    "generate_gradient_magnitude_wgsl",
    "generate_adaptive_march_wgsl",
    "generate_complexity_map_wgsl",
    "generate_step_scaler_wgsl",
    # Convenience functions
    "estimate_complexity",
    "compute_gradient_magnitude",
    "adaptive_march_ray",
    "create_adaptive_marcher",
]

# Default step counts for different complexity levels
MIN_STEPS_SIMPLE = 20
MAX_STEPS_SIMPLE = 40
MIN_STEPS_COMPLEX = 80
MAX_STEPS_COMPLEX = 128

# Gradient magnitude thresholds
GRADIENT_LOW_THRESHOLD = 0.3   # Below this = complex (detail/edge)
GRADIENT_HIGH_THRESHOLD = 0.9  # Above this = simple (smooth surface)

# Step scaling bounds
MIN_STEP_SCALE = 0.5   # Minimum step scale factor
MAX_STEP_SCALE = 2.0   # Maximum step scale factor

# Default epsilon for gradient sampling
DEFAULT_GRADIENT_EPSILON = 0.0001

# Complexity map default resolution
DEFAULT_COMPLEXITY_RESOLUTION = 64


# =============================================================================
# Complexity Level Enumeration
# =============================================================================

class ComplexityLevel(Enum):
    """
    Scene complexity classification based on gradient analysis.

    Used to determine adaptive step counts for ray marching.
    """
    SIMPLE = auto()      # Smooth surfaces, large step size
    MODERATE = auto()    # Mixed detail, medium step size
    COMPLEX = auto()     # Fine detail/edges, small step size
    EXTREME = auto()     # Very fine detail, minimum step size

    @classmethod
    def from_gradient(cls, gradient_magnitude: float) -> "ComplexityLevel":
        """
        Classify complexity based on gradient magnitude.

        For SDFs, gradient magnitude ~= 1.0 means well-behaved (simple).
        Deviation from 1.0 in either direction indicates complexity.

        Args:
            gradient_magnitude: Raw gradient magnitude (ideal = 1.0)

        Returns:
            ComplexityLevel classification
        """
        # Ideal gradient is 1.0; measure deviation from ideal
        deviation = abs(gradient_magnitude - 1.0)

        if deviation < 0.1:  # Very close to ideal
            return cls.SIMPLE
        elif deviation < 0.3:  # Moderate deviation
            return cls.MODERATE
        elif deviation < 0.6:  # Significant deviation
            return cls.COMPLEX
        else:  # Large deviation
            return cls.EXTREME

    def get_step_range(self) -> Tuple[int, int]:
        """Get recommended step count range for this complexity level."""
        if self == ComplexityLevel.SIMPLE:
            return (MIN_STEPS_SIMPLE, MAX_STEPS_SIMPLE)
        elif self == ComplexityLevel.MODERATE:
            return (40, 64)
        elif self == ComplexityLevel.COMPLEX:
            return (MIN_STEPS_COMPLEX, MAX_STEPS_COMPLEX)
        else:  # EXTREME
            return (128, 256)

    def get_step_scale(self) -> float:
        """Get step scale factor for this complexity level."""
        if self == ComplexityLevel.SIMPLE:
            return MAX_STEP_SCALE
        elif self == ComplexityLevel.MODERATE:
            return 1.5
        elif self == ComplexityLevel.COMPLEX:
            return 0.75
        else:  # EXTREME
            return MIN_STEP_SCALE


# =============================================================================
# Complexity Estimation
# =============================================================================

@dataclass
class ComplexityEstimate:
    """
    Result of complexity analysis at a point or region.

    Attributes:
        level: Classified complexity level
        gradient_magnitude: Raw gradient magnitude [0, inf)
        normalized_gradient: Gradient normalized to [0, 1] range
        recommended_steps: Recommended step count for this region
        step_scale: Recommended step scale factor
        confidence: Confidence in the estimate (0-1)
    """
    level: ComplexityLevel
    gradient_magnitude: float
    normalized_gradient: float
    recommended_steps: int
    step_scale: float
    confidence: float = 1.0

    @classmethod
    def simple(cls) -> "ComplexityEstimate":
        """Create estimate for simple region."""
        return cls(
            level=ComplexityLevel.SIMPLE,
            gradient_magnitude=1.0,
            normalized_gradient=1.0,
            recommended_steps=MIN_STEPS_SIMPLE,
            step_scale=MAX_STEP_SCALE,
        )

    @classmethod
    def complex(cls) -> "ComplexityEstimate":
        """Create estimate for complex region."""
        return cls(
            level=ComplexityLevel.COMPLEX,
            gradient_magnitude=0.2,
            normalized_gradient=0.2,
            recommended_steps=MAX_STEPS_COMPLEX,
            step_scale=MIN_STEP_SCALE,
        )

    @classmethod
    def from_gradient(cls, gradient_magnitude: float) -> "ComplexityEstimate":
        """
        Create complexity estimate from gradient magnitude.

        Args:
            gradient_magnitude: Raw SDF gradient magnitude (ideal = 1.0)

        Returns:
            ComplexityEstimate with all fields populated
        """
        # Measure how close to ideal gradient (1.0)
        # 1.0 = ideal/simple, deviation from 1.0 = complex
        deviation = abs(gradient_magnitude - 1.0)
        # Normalize: 0 = ideal (simple), 1 = highly deviated (complex)
        normalized = 1.0 - min(deviation, 1.0)  # Invert so higher = simpler

        level = ComplexityLevel.from_gradient(gradient_magnitude)
        min_steps, max_steps = level.get_step_range()

        # Interpolate within step range based on how close to ideal
        # Higher normalized = simpler = fewer steps
        t = normalized  # 0 = complex, 1 = simple
        recommended_steps = int(max_steps - t * (max_steps - min_steps))

        return cls(
            level=level,
            gradient_magnitude=gradient_magnitude,
            normalized_gradient=normalized,
            recommended_steps=recommended_steps,
            step_scale=level.get_step_scale(),
        )


# Type alias for SDF function
SDFFunc = Callable[[Vec3], float]


class GradientAnalyzer:
    """
    Analyzes SDF gradient magnitude for complexity estimation.

    Uses central differences to compute the gradient vector and its
    magnitude. The gradient magnitude indicates how "well-behaved"
    the SDF is at a given point:

    - Magnitude ~= 1.0: Perfect SDF, simple geometry
    - Magnitude < 1.0: Near detail/edge, needs smaller steps
    - Magnitude > 1.0: Lipschitz violation, needs careful handling

    Example:
        >>> analyzer = GradientAnalyzer(epsilon=0.0001)
        >>> def sphere_sdf(p): return p.length() - 1.0
        >>> mag = analyzer.compute_magnitude(Vec3(1.0, 0.0, 0.0), sphere_sdf)
        >>> abs(mag - 1.0) < 0.01
        True
    """

    def __init__(self, epsilon: float = DEFAULT_GRADIENT_EPSILON) -> None:
        """
        Initialize gradient analyzer.

        Args:
            epsilon: Sampling distance for central differences

        Raises:
            ValueError: If epsilon is not positive
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon

    def compute_gradient(self, p: Vec3, sdf: SDFFunc) -> Vec3:
        """
        Compute SDF gradient vector using central differences.

        Args:
            p: Point to evaluate gradient at
            sdf: Signed distance function

        Returns:
            Gradient vector (not normalized)
        """
        e = self.epsilon

        # Central differences
        dx = sdf(Vec3(p.x + e, p.y, p.z)) - sdf(Vec3(p.x - e, p.y, p.z))
        dy = sdf(Vec3(p.x, p.y + e, p.z)) - sdf(Vec3(p.x, p.y - e, p.z))
        dz = sdf(Vec3(p.x, p.y, p.z + e)) - sdf(Vec3(p.x, p.y, p.z - e))

        # Scale by 1/(2*epsilon) for proper gradient
        scale = 0.5 / e
        return Vec3(dx * scale, dy * scale, dz * scale)

    def compute_magnitude(self, p: Vec3, sdf: SDFFunc) -> float:
        """
        Compute gradient magnitude at a point.

        For a true SDF, the gradient magnitude should be exactly 1.0
        everywhere. Deviations indicate:
        - < 1.0: Near edges or fine detail
        - > 1.0: Lipschitz condition violation

        Args:
            p: Point to evaluate gradient at
            sdf: Signed distance function

        Returns:
            Gradient magnitude |grad_SDF|
        """
        grad = self.compute_gradient(p, sdf)
        return grad.length()

    def estimate_complexity(self, p: Vec3, sdf: SDFFunc) -> ComplexityEstimate:
        """
        Estimate scene complexity at a point.

        Args:
            p: Point to analyze
            sdf: Signed distance function

        Returns:
            ComplexityEstimate with full analysis
        """
        magnitude = self.compute_magnitude(p, sdf)
        return ComplexityEstimate.from_gradient(magnitude)

    def compute_magnitude_tetrahedron(self, p: Vec3, sdf: SDFFunc) -> float:
        """
        Compute gradient magnitude using 4-point tetrahedron stencil.

        Faster than 6-point central differences (4 vs 6 samples)
        with slight accuracy reduction.

        Args:
            p: Point to evaluate gradient at
            sdf: Signed distance function

        Returns:
            Gradient magnitude
        """
        e = self.epsilon
        k = e * 0.5773502691896258  # 1/sqrt(3)

        s1 = sdf(Vec3(p.x + k, p.y + k, p.z + k))
        s2 = sdf(Vec3(p.x + k, p.y - k, p.z - k))
        s3 = sdf(Vec3(p.x - k, p.y + k, p.z - k))
        s4 = sdf(Vec3(p.x - k, p.y - k, p.z + k))

        nx = s1 + s2 - s3 - s4
        ny = s1 - s2 + s3 - s4
        nz = s1 - s2 - s3 + s4

        # Scale for proper gradient magnitude
        scale = 0.5 / e
        return math.sqrt(nx*nx + ny*ny + nz*nz) * scale


def compute_gradient_magnitude(
    p: Vec3,
    sdf: SDFFunc,
    epsilon: float = DEFAULT_GRADIENT_EPSILON,
) -> float:
    """
    Convenience function to compute gradient magnitude.

    Args:
        p: Point to evaluate
        sdf: Signed distance function
        epsilon: Sampling distance

    Returns:
        Gradient magnitude |grad_SDF|
    """
    analyzer = GradientAnalyzer(epsilon)
    return analyzer.compute_magnitude(p, sdf)


def estimate_complexity(
    p: Vec3,
    sdf: SDFFunc,
    epsilon: float = DEFAULT_GRADIENT_EPSILON,
) -> ComplexityEstimate:
    """
    Convenience function to estimate complexity at a point.

    Args:
        p: Point to analyze
        sdf: Signed distance function
        epsilon: Sampling distance

    Returns:
        ComplexityEstimate
    """
    analyzer = GradientAnalyzer(epsilon)
    return analyzer.estimate_complexity(p, sdf)


# =============================================================================
# Step Scaling
# =============================================================================

class StepScaler:
    """
    Base class for step size scaling strategies.

    Step scalers modify the ray march step size based on various criteria
    to optimize performance while maintaining quality.
    """

    def scale(
        self,
        base_step: float,
        position: Vec3,
        sdf: SDFFunc,
        distance_traveled: float,
    ) -> float:
        """
        Compute scaled step size.

        Args:
            base_step: Base step size (SDF value)
            position: Current position along ray
            sdf: Signed distance function
            distance_traveled: Distance traveled so far

        Returns:
            Scaled step size
        """
        raise NotImplementedError


class GradientBasedScaler(StepScaler):
    """
    Scale step size based on gradient magnitude.

    High gradient (magnitude ~= 1) indicates simple geometry where
    large steps are safe. Low gradient indicates detail/edges where
    smaller steps are needed.

    Formula: step *= lerp(min_scale, max_scale, gradient_mag)

    Example:
        >>> scaler = GradientBasedScaler()
        >>> def sphere_sdf(p): return p.length() - 1.0
        >>> # At sphere surface, gradient is well-behaved
        >>> scale = scaler.scale(0.1, Vec3(1.0, 0.0, 0.0), sphere_sdf, 0.0)
        >>> scale >= 0.1  # Should scale up or maintain
        True
    """

    def __init__(
        self,
        min_scale: float = MIN_STEP_SCALE,
        max_scale: float = MAX_STEP_SCALE,
        gradient_epsilon: float = DEFAULT_GRADIENT_EPSILON,
    ) -> None:
        """
        Initialize gradient-based scaler.

        Args:
            min_scale: Minimum scale factor (for complex regions)
            max_scale: Maximum scale factor (for simple regions)
            gradient_epsilon: Epsilon for gradient computation

        Raises:
            ValueError: If scales are invalid
        """
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}")
        if max_scale <= min_scale:
            raise ValueError(
                f"max_scale ({max_scale}) must be > min_scale ({min_scale})"
            )

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.analyzer = GradientAnalyzer(gradient_epsilon)

    def scale(
        self,
        base_step: float,
        position: Vec3,
        sdf: SDFFunc,
        distance_traveled: float,
    ) -> float:
        """Compute step scaled by gradient magnitude."""
        # Compute gradient magnitude
        grad_mag = self.analyzer.compute_magnitude(position, sdf)

        # Normalize to [0, 1] range (ideal SDF has grad_mag = 1)
        # Values > 1 are clamped, values < 1 reduce step size
        normalized = min(grad_mag, 1.0)

        # Lerp between min and max scale
        t = normalized
        scale = self.min_scale + t * (self.max_scale - self.min_scale)

        return base_step * scale


class DistanceBasedScaler(StepScaler):
    """
    Scale step size based on distance from camera.

    Far objects can use larger steps without visible quality loss
    (similar to perceptual epsilon but for step size).

    Formula: step *= 1.0 + distance * scale_rate
    """

    def __init__(
        self,
        scale_rate: float = 0.01,
        max_scale: float = MAX_STEP_SCALE,
    ) -> None:
        """
        Initialize distance-based scaler.

        Args:
            scale_rate: How much to increase scale per unit distance
            max_scale: Maximum scale factor
        """
        if scale_rate < 0:
            raise ValueError(f"scale_rate must be non-negative, got {scale_rate}")
        if max_scale <= 0:
            raise ValueError(f"max_scale must be positive, got {max_scale}")

        self.scale_rate = scale_rate
        self.max_scale = max_scale

    def scale(
        self,
        base_step: float,
        position: Vec3,
        sdf: SDFFunc,
        distance_traveled: float,
    ) -> float:
        """Compute step scaled by distance."""
        scale = 1.0 + distance_traveled * self.scale_rate
        scale = min(scale, self.max_scale)
        return base_step * scale


class CombinedScaler(StepScaler):
    """
    Combine multiple scaling strategies.

    Uses the minimum of all scaler outputs to be conservative
    (never take a step larger than any scaler recommends).

    Example:
        >>> scaler = CombinedScaler([
        ...     GradientBasedScaler(),
        ...     DistanceBasedScaler(),
        ... ])
    """

    def __init__(
        self,
        scalers: List[StepScaler],
        mode: str = "min",
    ) -> None:
        """
        Initialize combined scaler.

        Args:
            scalers: List of scalers to combine
            mode: Combination mode ("min", "max", "average")

        Raises:
            ValueError: If scalers is empty or mode is invalid
        """
        if not scalers:
            raise ValueError("scalers cannot be empty")
        if mode not in ("min", "max", "average"):
            raise ValueError(f"mode must be min/max/average, got {mode}")

        self.scalers = scalers
        self.mode = mode

    def scale(
        self,
        base_step: float,
        position: Vec3,
        sdf: SDFFunc,
        distance_traveled: float,
    ) -> float:
        """Compute combined scaled step."""
        results = [
            scaler.scale(base_step, position, sdf, distance_traveled)
            for scaler in self.scalers
        ]

        if self.mode == "min":
            return min(results)
        elif self.mode == "max":
            return max(results)
        else:  # average
            return sum(results) / len(results)


# =============================================================================
# Adaptive Ray Marching
# =============================================================================

@dataclass
class AdaptiveMarchConfig:
    """
    Configuration for adaptive ray marching.

    Attributes:
        base_max_steps: Base maximum step count
        min_steps: Minimum steps for simple regions
        max_steps: Maximum steps for complex regions
        base_epsilon: Base surface hit threshold
        use_gradient_scaling: Enable gradient-based step scaling
        use_distance_scaling: Enable distance-based step scaling
        gradient_epsilon: Epsilon for gradient computation
        min_step_scale: Minimum step scale factor
        max_step_scale: Maximum step scale factor
        max_distance: Maximum ray travel distance
    """
    base_max_steps: int = 64
    min_steps: int = MIN_STEPS_SIMPLE
    max_steps: int = MAX_STEPS_COMPLEX
    base_epsilon: float = 0.001
    use_gradient_scaling: bool = True
    use_distance_scaling: bool = True
    gradient_epsilon: float = DEFAULT_GRADIENT_EPSILON
    min_step_scale: float = MIN_STEP_SCALE
    max_step_scale: float = MAX_STEP_SCALE
    max_distance: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.base_max_steps <= 0:
            raise ValueError(f"base_max_steps must be positive, got {self.base_max_steps}")
        if self.min_steps <= 0:
            raise ValueError(f"min_steps must be positive, got {self.min_steps}")
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})"
            )
        if self.base_epsilon <= 0:
            raise ValueError(f"base_epsilon must be positive, got {self.base_epsilon}")
        if self.max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")


@dataclass
class AdaptiveMarchResult:
    """
    Result of adaptive ray marching.

    Attributes:
        hit: Whether a surface was hit
        position: Hit position (if hit)
        distance: Ray travel distance
        steps: Number of steps taken
        steps_saved: Estimated steps saved vs fixed step count
        avg_step_scale: Average step scale factor used
        complexity: Estimated complexity at hit point
        gradient_magnitude: Final gradient magnitude
        epsilon_used: Final epsilon value used
    """
    hit: bool = False
    position: Optional[Vec3] = None
    distance: float = 0.0
    steps: int = 0
    steps_saved: int = 0
    avg_step_scale: float = 1.0
    complexity: Optional[ComplexityEstimate] = None
    gradient_magnitude: float = 1.0
    epsilon_used: float = 0.001

    @property
    def efficiency(self) -> float:
        """
        Compute efficiency ratio (steps saved / total steps).

        Returns:
            Efficiency as percentage [0, 100]
        """
        if self.steps == 0:
            return 0.0
        return (self.steps_saved / (self.steps + self.steps_saved)) * 100.0


class AdaptiveMarcher:
    """
    Adaptive ray marcher with gradient-based step scaling.

    This marcher adjusts step sizes based on local scene complexity:
    - Simple regions (high gradient): Use larger steps
    - Complex regions (low gradient): Use smaller steps

    This improves performance in simple areas while maintaining
    quality in detailed regions.

    Example:
        >>> config = AdaptiveMarchConfig(use_gradient_scaling=True)
        >>> marcher = AdaptiveMarcher(config)
        >>> def sphere_sdf(p): return p.length() - 1.0
        >>> result = marcher.march(Vec3(0, 0, 5), Vec3(0, 0, -1), sphere_sdf)
        >>> result.hit
        True
    """

    def __init__(self, config: Optional[AdaptiveMarchConfig] = None) -> None:
        """
        Initialize adaptive marcher.

        Args:
            config: Marching configuration
        """
        self.config = config or AdaptiveMarchConfig()

        # Build scaler chain
        scalers: List[StepScaler] = []
        if self.config.use_gradient_scaling:
            scalers.append(GradientBasedScaler(
                min_scale=self.config.min_step_scale,
                max_scale=self.config.max_step_scale,
                gradient_epsilon=self.config.gradient_epsilon,
            ))
        if self.config.use_distance_scaling:
            scalers.append(DistanceBasedScaler(
                max_scale=self.config.max_step_scale,
            ))

        self._scaler: Optional[StepScaler] = None
        if scalers:
            self._scaler = CombinedScaler(scalers, mode="min") if len(scalers) > 1 else scalers[0]

        self._gradient_analyzer = GradientAnalyzer(self.config.gradient_epsilon)
        self._step_scales: List[float] = []

    def march(
        self,
        origin: Vec3,
        direction: Vec3,
        sdf: SDFFunc,
    ) -> AdaptiveMarchResult:
        """
        March ray with adaptive step sizing.

        Args:
            origin: Ray origin
            direction: Ray direction (will be normalized)
            sdf: Signed distance function

        Returns:
            AdaptiveMarchResult with hit information and efficiency metrics
        """
        # Normalize direction
        dir_len = direction.length()
        if dir_len < 1e-10:
            return AdaptiveMarchResult(hit=False, steps=0)
        direction = direction.normalized()

        # Clear step history
        self._step_scales.clear()

        t: float = 0.0
        steps = 0
        total_base_steps = 0

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

            # Check for surface hit
            if d < self.config.base_epsilon:
                # Compute final gradient
                grad_mag = self._gradient_analyzer.compute_magnitude(p, sdf)
                complexity = ComplexityEstimate.from_gradient(grad_mag)

                # Compute efficiency
                avg_scale = sum(self._step_scales) / len(self._step_scales) if self._step_scales else 1.0
                steps_saved = max(0, int(total_base_steps - steps))

                return AdaptiveMarchResult(
                    hit=True,
                    position=p,
                    distance=t,
                    steps=steps,
                    steps_saved=steps_saved,
                    avg_step_scale=avg_scale,
                    complexity=complexity,
                    gradient_magnitude=grad_mag,
                    epsilon_used=self.config.base_epsilon,
                )

            # Apply step scaling
            if self._scaler:
                scaled_d = self._scaler.scale(d, p, sdf, t)
                scale = scaled_d / d if d > 0 else 1.0
            else:
                scaled_d = d
                scale = 1.0

            self._step_scales.append(scale)
            total_base_steps += 1.0 / scale if scale > 0 else 1.0

            # Advance ray
            t += scaled_d

            # Check for max distance
            if t > self.config.max_distance:
                avg_scale = sum(self._step_scales) / len(self._step_scales) if self._step_scales else 1.0
                return AdaptiveMarchResult(
                    hit=False,
                    distance=t,
                    steps=steps,
                    avg_step_scale=avg_scale,
                )

        # Max steps exhausted
        avg_scale = sum(self._step_scales) / len(self._step_scales) if self._step_scales else 1.0
        return AdaptiveMarchResult(
            hit=False,
            distance=t,
            steps=steps,
            avg_step_scale=avg_scale,
        )

    @property
    def step_scales(self) -> List[float]:
        """Get step scale factors from last march."""
        return self._step_scales.copy()


def adaptive_march_ray(
    origin: Vec3,
    direction: Vec3,
    sdf: SDFFunc,
    config: Optional[AdaptiveMarchConfig] = None,
) -> AdaptiveMarchResult:
    """
    Convenience function for single adaptive ray march.

    Args:
        origin: Ray origin
        direction: Ray direction
        sdf: Signed distance function
        config: Optional configuration

    Returns:
        AdaptiveMarchResult
    """
    marcher = AdaptiveMarcher(config)
    return marcher.march(origin, direction, sdf)


def create_adaptive_marcher(
    use_gradient_scaling: bool = True,
    use_distance_scaling: bool = True,
    max_steps: int = MAX_STEPS_COMPLEX,
) -> AdaptiveMarcher:
    """
    Factory function to create an adaptive marcher.

    Args:
        use_gradient_scaling: Enable gradient-based step scaling
        use_distance_scaling: Enable distance-based step scaling
        max_steps: Maximum step count

    Returns:
        Configured AdaptiveMarcher
    """
    config = AdaptiveMarchConfig(
        use_gradient_scaling=use_gradient_scaling,
        use_distance_scaling=use_distance_scaling,
        max_steps=max_steps,
    )
    return AdaptiveMarcher(config)


# =============================================================================
# Complexity Map
# =============================================================================

@dataclass
class ComplexityMapConfig:
    """
    Configuration for complexity map generation.

    Attributes:
        resolution: Map resolution (width and height)
        sample_count: Number of samples per pixel for averaging
        depth_range: Range of depths to sample (near, far)
    """
    resolution: int = DEFAULT_COMPLEXITY_RESOLUTION
    sample_count: int = 4
    depth_range: Tuple[float, float] = (0.1, 50.0)

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {self.sample_count}")


class ComplexityMap:
    """
    Per-pixel complexity map for guiding adaptive ray marching.

    The complexity map stores estimated scene complexity at each pixel,
    allowing the second render pass to allocate step counts based on
    the first pass analysis.

    Attributes:
        width: Map width
        height: Map height
        data: Complexity estimates per pixel
    """

    def __init__(self, width: int, height: int) -> None:
        """
        Initialize complexity map.

        Args:
            width: Map width
            height: Map height
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"dimensions must be positive, got {width}x{height}")

        self.width = width
        self.height = height
        self.data: List[List[ComplexityEstimate]] = [
            [ComplexityEstimate.simple() for _ in range(width)]
            for _ in range(height)
        ]

    def get(self, x: int, y: int) -> ComplexityEstimate:
        """Get complexity at pixel (x, y)."""
        return self.data[y % self.height][x % self.width]

    def set(self, x: int, y: int, estimate: ComplexityEstimate) -> None:
        """Set complexity at pixel (x, y)."""
        self.data[y % self.height][x % self.width] = estimate

    def get_recommended_steps(self, x: int, y: int) -> int:
        """Get recommended step count for pixel."""
        return self.get(x, y).recommended_steps

    def get_step_scale(self, x: int, y: int) -> float:
        """Get step scale for pixel."""
        return self.get(x, y).step_scale

    def average_complexity(self) -> float:
        """Compute average complexity across map."""
        total = 0.0
        for row in self.data:
            for estimate in row:
                total += estimate.normalized_gradient
        return total / (self.width * self.height)

    def average_recommended_steps(self) -> float:
        """Compute average recommended steps across map."""
        total = 0
        for row in self.data:
            for estimate in row:
                total += estimate.recommended_steps
        return total / (self.width * self.height)


class ComplexityMapGenerator:
    """
    Generates per-pixel complexity maps from first-pass ray marching.

    The generator performs a quick first-pass render with reduced
    step counts to estimate scene complexity, then stores the results
    in a complexity map for use in the full render.

    Example:
        >>> config = ComplexityMapConfig(resolution=64)
        >>> generator = ComplexityMapGenerator(config)
        >>> complexity_map = generator.generate(
        ...     camera_origin=Vec3(0, 0, 5),
        ...     ray_generator=make_rays,
        ...     sdf=scene_sdf,
        ... )
    """

    def __init__(self, config: Optional[ComplexityMapConfig] = None) -> None:
        """Initialize generator with configuration."""
        self.config = config or ComplexityMapConfig()
        self._analyzer = GradientAnalyzer()

    def generate(
        self,
        camera_origin: Vec3,
        ray_generator: Callable[[int, int], Vec3],
        sdf: SDFFunc,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> ComplexityMap:
        """
        Generate complexity map from scene.

        Args:
            camera_origin: Camera position
            ray_generator: Function (x, y) -> ray_direction
            sdf: Signed distance function
            width: Output width (defaults to config resolution)
            height: Output height (defaults to config resolution)

        Returns:
            ComplexityMap with per-pixel estimates
        """
        w = width or self.config.resolution
        h = height or self.config.resolution

        complexity_map = ComplexityMap(w, h)

        for y in range(h):
            for x in range(w):
                direction = ray_generator(x, y)
                estimate = self._sample_complexity(
                    camera_origin,
                    direction,
                    sdf,
                )
                complexity_map.set(x, y, estimate)

        return complexity_map

    def _sample_complexity(
        self,
        origin: Vec3,
        direction: Vec3,
        sdf: SDFFunc,
    ) -> ComplexityEstimate:
        """
        Sample complexity along a ray.

        Marches a few steps to find a representative point
        and estimates complexity there.
        """
        dir_len = direction.length()
        if dir_len < 1e-10:
            return ComplexityEstimate.simple()
        direction = direction.normalized()

        near, far = self.config.depth_range
        total_gradient = 0.0
        samples = 0

        # Sample at multiple depths
        for i in range(self.config.sample_count):
            t = near + (far - near) * (i + 0.5) / self.config.sample_count
            p = Vec3(
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t,
            )

            # Only sample if we're near geometry
            d = sdf(p)
            if abs(d) < 5.0:  # Within reasonable distance
                grad_mag = self._analyzer.compute_magnitude(p, sdf)
                total_gradient += grad_mag
                samples += 1

        if samples == 0:
            return ComplexityEstimate.simple()

        avg_gradient = total_gradient / samples
        return ComplexityEstimate.from_gradient(avg_gradient)


# =============================================================================
# WGSL Code Generation
# =============================================================================

def generate_gradient_magnitude_wgsl(use_tetrahedron: bool = False) -> str:
    """
    Generate WGSL function for gradient magnitude computation.

    Args:
        use_tetrahedron: Use 4-point tetrahedron stencil (faster)

    Returns:
        WGSL function source code
    """
    if use_tetrahedron:
        return """\
/// Computes SDF gradient magnitude using 4-point tetrahedron stencil.
/// For a true SDF, magnitude should be ~1.0 everywhere.
///   p       -- position to evaluate
///   epsilon -- sampling distance
///   returns -- gradient magnitude |grad_SDF|
fn gradient_magnitude_tetrahedron(p: vec3<f32>, epsilon: f32) -> f32 {
    let k = epsilon * 0.5773502691896258;  // 1/sqrt(3)

    let s1 = scene_sdf(p + vec3<f32>(k, k, k)).x;
    let s2 = scene_sdf(p + vec3<f32>(k, -k, -k)).x;
    let s3 = scene_sdf(p + vec3<f32>(-k, k, -k)).x;
    let s4 = scene_sdf(p + vec3<f32>(-k, -k, k)).x;

    let nx = s1 + s2 - s3 - s4;
    let ny = s1 - s2 + s3 - s4;
    let nz = s1 - s2 - s3 + s4;

    let scale = 0.5 / epsilon;
    return length(vec3<f32>(nx, ny, nz)) * scale;
}
"""
    else:
        return """\
/// Computes SDF gradient magnitude using 6-point central differences.
/// For a true SDF, magnitude should be ~1.0 everywhere.
///   p       -- position to evaluate
///   epsilon -- sampling distance
///   returns -- gradient magnitude |grad_SDF|
fn gradient_magnitude(p: vec3<f32>, epsilon: f32) -> f32 {
    let e = vec2<f32>(epsilon, 0.0);

    let dx = scene_sdf(p + e.xyy).x - scene_sdf(p - e.xyy).x;
    let dy = scene_sdf(p + e.yxy).x - scene_sdf(p - e.yxy).x;
    let dz = scene_sdf(p + e.yyx).x - scene_sdf(p - e.yyx).x;

    let scale = 0.5 / epsilon;
    return length(vec3<f32>(dx, dy, dz)) * scale;
}
"""


def generate_step_scaler_wgsl() -> str:
    """
    Generate WGSL functions for step scaling.

    Returns:
        WGSL source code for gradient and distance based step scaling
    """
    return """\
/// Step scaling parameters.
struct StepScaleParams {
    min_scale: f32,      // Minimum scale factor (for complex regions)
    max_scale: f32,      // Maximum scale factor (for simple regions)
    distance_rate: f32,  // Distance-based scaling rate
    gradient_eps: f32,   // Epsilon for gradient computation
}

/// Default step scaling parameters.
const DEFAULT_STEP_PARAMS: StepScaleParams = StepScaleParams(
    0.5,     // min_scale
    2.0,     // max_scale
    0.01,    // distance_rate
    0.0001   // gradient_eps
);

/// Scales step size based on gradient magnitude.
/// High gradient = simple geometry = larger steps.
///   base_step     -- SDF value (base step size)
///   p             -- current position
///   params        -- scaling parameters
///   returns       -- scaled step size
fn scale_step_gradient(base_step: f32, p: vec3<f32>, params: StepScaleParams) -> f32 {
    let grad_mag = gradient_magnitude(p, params.gradient_eps);
    let normalized = min(grad_mag, 1.0);
    let scale = mix(params.min_scale, params.max_scale, normalized);
    return base_step * scale;
}

/// Scales step size based on distance from camera.
/// Far objects can use larger steps.
///   base_step -- SDF value
///   distance  -- distance traveled so far
///   params    -- scaling parameters
///   returns   -- scaled step size
fn scale_step_distance(base_step: f32, distance: f32, params: StepScaleParams) -> f32 {
    let scale = min(1.0 + distance * params.distance_rate, params.max_scale);
    return base_step * scale;
}

/// Combined step scaling (gradient + distance).
///   base_step -- SDF value
///   p         -- current position
///   distance  -- distance traveled
///   params    -- scaling parameters
///   returns   -- scaled step size (conservative minimum)
fn scale_step_combined(
    base_step: f32,
    p: vec3<f32>,
    distance: f32,
    params: StepScaleParams,
) -> f32 {
    let grad_scaled = scale_step_gradient(base_step, p, params);
    let dist_scaled = scale_step_distance(base_step, distance, params);
    return min(grad_scaled, dist_scaled);
}
"""


def generate_adaptive_march_wgsl(
    use_gradient_scaling: bool = True,
    use_distance_scaling: bool = True,
) -> str:
    """
    Generate WGSL adaptive ray marching function.

    Args:
        use_gradient_scaling: Include gradient-based scaling
        use_distance_scaling: Include distance-based scaling

    Returns:
        WGSL function source code
    """
    # Build scaling call based on options
    if use_gradient_scaling and use_distance_scaling:
        scale_call = "scale_step_combined(d, p, t, step_params)"
    elif use_gradient_scaling:
        scale_call = "scale_step_gradient(d, p, step_params)"
    elif use_distance_scaling:
        scale_call = "scale_step_distance(d, t, step_params)"
    else:
        scale_call = "d"  # No scaling

    return f"""\
/// Result of adaptive ray marching.
struct AdaptiveRayHit {{
    hit: bool,
    position: vec3<f32>,
    distance: f32,
    steps: u32,
    avg_step_scale: f32,
    gradient_magnitude: f32,
}}

/// Adaptive ray marching with gradient-based step scaling.
/// Uses larger steps in simple regions, smaller steps near detail.
///   ro          -- ray origin
///   rd          -- ray direction (normalized)
///   max_steps   -- maximum step count
///   max_dist    -- maximum ray distance
///   epsilon     -- surface hit threshold
///   step_params -- step scaling parameters
///   returns     -- AdaptiveRayHit with efficiency metrics
fn adaptive_ray_march(
    ro: vec3<f32>,
    rd: vec3<f32>,
    max_steps: u32,
    max_dist: f32,
    epsilon: f32,
    step_params: StepScaleParams,
) -> AdaptiveRayHit {{
    var t: f32 = 0.0;
    var total_scale: f32 = 0.0;
    var hit = AdaptiveRayHit(
        false, vec3<f32>(0.0), 0.0, 0u, 1.0, 1.0
    );

    for (var i = 0u; i < max_steps; i = i + 1u) {{
        let p = ro + rd * t;
        let d = scene_sdf(p).x;

        if (d < epsilon) {{
            hit.hit = true;
            hit.position = p;
            hit.distance = t;
            hit.steps = i;
            hit.avg_step_scale = total_scale / f32(i + 1u);
            hit.gradient_magnitude = gradient_magnitude(p, step_params.gradient_eps);
            return hit;
        }}

        // Apply adaptive step scaling
        let scaled_d = {scale_call};
        total_scale = total_scale + scaled_d / d;

        t = t + scaled_d;

        if (t > max_dist) {{
            hit.steps = i;
            hit.avg_step_scale = total_scale / f32(i + 1u);
            return hit;
        }}
    }}

    hit.steps = max_steps;
    hit.avg_step_scale = total_scale / f32(max_steps);
    return hit;
}}

/// Simplified adaptive march with default parameters.
fn adaptive_ray_march_simple(
    ro: vec3<f32>,
    rd: vec3<f32>,
    max_steps: u32,
    max_dist: f32,
    epsilon: f32,
) -> AdaptiveRayHit {{
    return adaptive_ray_march(ro, rd, max_steps, max_dist, epsilon, DEFAULT_STEP_PARAMS);
}}
"""


def generate_complexity_map_wgsl(resolution: int = DEFAULT_COMPLEXITY_RESOLUTION) -> str:
    """
    Generate WGSL for complexity map sampling and usage.

    Args:
        resolution: Complexity map resolution

    Returns:
        WGSL source code
    """
    return f"""\
/// Complexity map texture binding.
@group(0) @binding(2) var complexity_map: texture_2d<f32>;
@group(0) @binding(3) var complexity_sampler: sampler;

/// Complexity map resolution.
const COMPLEXITY_MAP_RESOLUTION: u32 = {resolution}u;

/// Sample complexity map at screen coordinate.
///   uv -- screen UV [0, 1]
///   returns -- estimated complexity [0, 1] where 0=complex, 1=simple
fn sample_complexity(uv: vec2<f32>) -> f32 {{
    return textureSample(complexity_map, complexity_sampler, uv).r;
}}

/// Get recommended step count based on complexity.
///   complexity  -- sampled complexity value
///   min_steps   -- minimum steps (for simple regions)
///   max_steps   -- maximum steps (for complex regions)
///   returns     -- interpolated step count
fn get_adaptive_step_count(
    complexity: f32,
    min_steps: u32,
    max_steps: u32,
) -> u32 {{
    // Complexity is inverted: 0 = complex = more steps
    let t = 1.0 - complexity;
    return u32(mix(f32(min_steps), f32(max_steps), t));
}}

/// Generate complexity for a pixel during first pass.
/// Samples gradient magnitude at multiple depths.
///   ro        -- ray origin
///   rd        -- ray direction
///   near_far  -- depth range (near, far)
///   samples   -- number of depth samples
///   returns   -- complexity estimate [0, 1]
fn estimate_pixel_complexity(
    ro: vec3<f32>,
    rd: vec3<f32>,
    near_far: vec2<f32>,
    samples: u32,
) -> f32 {{
    var total_gradient: f32 = 0.0;
    var valid_samples: u32 = 0u;

    for (var i = 0u; i < samples; i = i + 1u) {{
        let t = near_far.x + (near_far.y - near_far.x) * (f32(i) + 0.5) / f32(samples);
        let p = ro + rd * t;

        let d = abs(scene_sdf(p).x);
        if (d < 5.0) {{
            let grad_mag = gradient_magnitude(p, 0.0001);
            total_gradient = total_gradient + grad_mag;
            valid_samples = valid_samples + 1u;
        }}
    }}

    if (valid_samples == 0u) {{
        return 1.0;  // Simple (no geometry found)
    }}

    let avg_gradient = total_gradient / f32(valid_samples);
    return clamp(avg_gradient, 0.0, 1.0);
}}
"""


# =============================================================================
# Full WGSL Pipeline Generation
# =============================================================================

def generate_full_adaptive_pipeline_wgsl(
    use_gradient_scaling: bool = True,
    use_distance_scaling: bool = True,
    use_complexity_map: bool = True,
    complexity_resolution: int = DEFAULT_COMPLEXITY_RESOLUTION,
) -> str:
    """
    Generate complete WGSL code for adaptive ray marching pipeline.

    Args:
        use_gradient_scaling: Enable gradient-based step scaling
        use_distance_scaling: Enable distance-based step scaling
        use_complexity_map: Enable complexity map pre-pass
        complexity_resolution: Resolution for complexity map

    Returns:
        Complete WGSL source code
    """
    parts = [
        "// T-DEMO-8.4: Importance-Driven SDF Evaluation",
        "// Adaptive step counts based on scene complexity",
        "",
        generate_gradient_magnitude_wgsl(),
        "",
        generate_step_scaler_wgsl(),
        "",
        generate_adaptive_march_wgsl(use_gradient_scaling, use_distance_scaling),
    ]

    if use_complexity_map:
        parts.extend([
            "",
            generate_complexity_map_wgsl(complexity_resolution),
        ])

    return "\n".join(parts)
