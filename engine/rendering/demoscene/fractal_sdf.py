"""
TRINITY Fractal SDF Module (T-DEMO-4.10 and T-DEMO-4.11)

This module implements advanced fractal Signed Distance Fields:
- T-DEMO-4.10: Mandelbulb SDF (3D Mandelbrot set)
- T-DEMO-4.11: KIFS (Kaleidoscopic Iterated Function System)

The Mandelbulb uses spherical coordinates to compute the 3D Mandelbrot set:
    z^n + c in spherical form where:
    - theta = atan2(y, x)  (azimuth)
    - phi = asin(z / r)    (inclination)
    - r^n * (sin(n*phi)*cos(n*theta), sin(n*phi)*sin(n*theta), cos(n*phi))

KIFS uses iterative folding and scaling for complex fractal geometry:
    - Fold operations: abs(p.x) for symmetry across planes
    - Scale and translate per iteration
    - Distance scaling compensation: d / scale^iterations

Reference: Inigo Quilez - Distance Functions
    https://iquilezles.org/articles/mandelbulb/
    https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Union
from abc import ABC, abstractmethod

from .sdf_ast import SDFNode, SDFNodeMeta, Vec3, Axis, Tracker


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Mandelbulb (T-DEMO-4.10)
    "MandelbulbSDF",
    "MandelbulbConfig",
    "mandelbulb_distance",
    "mandelbulb_distance_estimator",
    # KIFS (T-DEMO-4.11)
    "KIFSSDF",
    "KIFSConfig",
    "KIFSFoldType",
    "kifs_fold_abs",
    "kifs_fold_menger",
    "kifs_fold_sierpinski",
    "kifs_distance",
    # WGSL generation
    "generate_mandelbulb_wgsl",
    "generate_kifs_wgsl",
    "MANDELBULB_WGSL_FUNCTION",
    "KIFS_WGSL_FUNCTION",
    # Utilities
    "FractalSDFNode",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
]

# Default parameters
DEFAULT_MANDELBULB_POWER = 8.0
DEFAULT_MANDELBULB_ITERATIONS = 15
DEFAULT_MANDELBULB_BAILOUT = 2.0

DEFAULT_KIFS_ITERATIONS = 8
DEFAULT_KIFS_SCALE = 2.0
DEFAULT_KIFS_FOLD_COUNT = 3


# =============================================================================
# Spherical Coordinate Utilities
# =============================================================================

def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        Tuple of (r, theta, phi) where:
        - r: radius (distance from origin)
        - theta: azimuth angle (in XY plane, from X axis)
        - phi: inclination angle (from XY plane)
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-10:
        return (0.0, 0.0, 0.0)

    theta = math.atan2(y, x)
    # phi = asin(z / r) for inclination from XY plane
    phi = math.asin(max(-1.0, min(1.0, z / r)))

    return (r, theta, phi)


def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        r: radius
        theta: azimuth angle
        phi: inclination angle

    Returns:
        Tuple of (x, y, z) Cartesian coordinates
    """
    cos_phi = math.cos(phi)
    x = r * cos_phi * math.cos(theta)
    y = r * cos_phi * math.sin(theta)
    z = r * math.sin(phi)
    return (x, y, z)


def spherical_power(
    x: float, y: float, z: float, power: float
) -> Tuple[float, float, float]:
    """
    Raise a 3D point to a power using spherical coordinates.

    This is the core operation for the Mandelbulb:
    z^n in spherical form = r^n * (sin(n*phi)*cos(n*theta), sin(n*phi)*sin(n*theta), cos(n*phi))

    Args:
        x, y, z: Cartesian coordinates of the point
        power: The exponent n

    Returns:
        Tuple of (x', y', z') - the point raised to power n
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)

    if r < 1e-10:
        return (0.0, 0.0, 0.0)

    # Power formula in spherical coordinates
    new_r = r ** power
    new_theta = theta * power
    new_phi = phi * power

    # Convert back to Cartesian
    # Note: The Mandelbulb formula uses a slightly different convention
    # where we compute sin/cos of (n * phi) for the vertical component
    sin_nphi = math.sin(new_phi)
    cos_nphi = math.cos(new_phi)

    new_x = new_r * sin_nphi * math.cos(new_theta)
    new_y = new_r * sin_nphi * math.sin(new_theta)
    new_z = new_r * cos_nphi

    return (new_x, new_y, new_z)


# =============================================================================
# Mandelbulb SDF (T-DEMO-4.10)
# =============================================================================

@dataclass
class MandelbulbConfig:
    """Configuration for Mandelbulb SDF."""

    power: float = DEFAULT_MANDELBULB_POWER
    iterations: int = DEFAULT_MANDELBULB_ITERATIONS
    bailout: float = DEFAULT_MANDELBULB_BAILOUT

    def __post_init__(self):
        """Validate configuration."""
        if self.power < 1.0:
            raise ValueError(f"Power must be >= 1.0, got {self.power}")
        if self.iterations < 1:
            raise ValueError(f"Iterations must be >= 1, got {self.iterations}")
        if self.bailout <= 0.0:
            raise ValueError(f"Bailout must be > 0.0, got {self.bailout}")


def mandelbulb_iterate(
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    power: float,
) -> Tuple[float, float, float, float, float]:
    """
    Perform one iteration of the Mandelbulb formula: z = z^n + c

    Args:
        px, py, pz: Current z value
        cx, cy, cz: The c constant (typically the starting point)
        power: The exponent n

    Returns:
        Tuple of (new_x, new_y, new_z, r, dr_multiplier) where:
        - new_x, new_y, new_z: The new z value
        - r: The radius before the power operation
        - dr_multiplier: The derivative multiplier for distance estimation
    """
    r = math.sqrt(px * px + py * py + pz * pz)

    if r < 1e-10:
        return (cx, cy, cz, r, 1.0)

    # Spherical coordinates
    theta = math.atan2(py, px)
    phi = math.acos(max(-1.0, min(1.0, pz / r)))

    # Power formula
    new_r = r ** power
    new_theta = theta * power
    new_phi = phi * power

    # Convert back to Cartesian and add c
    sin_phi = math.sin(new_phi)
    new_x = new_r * sin_phi * math.cos(new_theta) + cx
    new_y = new_r * sin_phi * math.sin(new_theta) + cy
    new_z = new_r * math.cos(new_phi) + cz

    # Derivative multiplier: d/dr (r^n) = n * r^(n-1)
    dr_multiplier = power * (r ** (power - 1.0))

    return (new_x, new_y, new_z, r, dr_multiplier)


def mandelbulb_distance_estimator(
    x: float, y: float, z: float,
    power: float = DEFAULT_MANDELBULB_POWER,
    iterations: int = DEFAULT_MANDELBULB_ITERATIONS,
    bailout: float = DEFAULT_MANDELBULB_BAILOUT,
) -> float:
    """
    Compute the distance estimate to the Mandelbulb surface.

    Uses the formula: 0.5 * log(r) * r / dr
    where dr is the running derivative estimate.

    Args:
        x, y, z: Point to evaluate
        power: The Mandelbulb exponent (default: 8.0 for classic Mandelbulb)
        iterations: Maximum iteration count
        bailout: Escape radius

    Returns:
        Estimated distance to the Mandelbulb surface
    """
    # Initial values: z = 0, c = p
    zx, zy, zz = 0.0, 0.0, 0.0
    cx, cy, cz = x, y, z

    # Running derivative estimate
    dr = 1.0

    bailout_sq = bailout * bailout
    r = 0.0

    for _ in range(iterations):
        r = math.sqrt(zx * zx + zy * zy + zz * zz)

        if r > bailout:
            break

        # Spherical coordinates
        theta = math.atan2(zy, zx)
        phi = math.acos(max(-1.0, min(1.0, zz / r))) if r > 1e-10 else 0.0

        # Update derivative: dr = n * r^(n-1) * dr + 1
        dr = power * (r ** (power - 1.0)) * dr + 1.0

        # Power formula: z = z^n + c
        new_r = r ** power
        new_theta = theta * power
        new_phi = phi * power

        sin_phi = math.sin(new_phi)
        zx = new_r * sin_phi * math.cos(new_theta) + cx
        zy = new_r * sin_phi * math.sin(new_theta) + cy
        zz = new_r * math.cos(new_phi) + cz

    # Distance estimator formula: 0.5 * log(r) * r / dr
    if r < 1e-10 or dr < 1e-10:
        return 0.0

    return 0.5 * math.log(r) * r / dr


def mandelbulb_distance(
    point: Tuple[float, float, float],
    config: Optional[MandelbulbConfig] = None,
) -> float:
    """
    Compute distance to Mandelbulb surface.

    Args:
        point: (x, y, z) tuple
        config: Mandelbulb configuration (uses defaults if None)

    Returns:
        Estimated distance to surface
    """
    config = config or MandelbulbConfig()
    return mandelbulb_distance_estimator(
        point[0], point[1], point[2],
        power=config.power,
        iterations=config.iterations,
        bailout=config.bailout,
    )


class MandelbulbSDF(SDFNode, metaclass=SDFNodeMeta):
    """
    Mandelbulb SDF node implementing the 3D Mandelbrot set.

    The Mandelbulb is computed using spherical coordinates:
    z = z^n + c where the power operation is:
    r^n * (sin(n*phi)*cos(n*theta), sin(n*phi)*sin(n*theta), cos(n*phi))

    Distance estimation uses: 0.5 * log(r) * r / dr

    Attributes:
        power: The exponent n (default: 8.0 for classic Mandelbulb)
        iterations: Maximum iteration count
        bailout: Escape radius for iteration termination
        position: Center position of the Mandelbulb
    """

    __slots__ = ("power", "iterations", "bailout", "position")

    power: float
    iterations: int
    bailout: float
    position: Vec3

    def __init__(
        self,
        power: float = DEFAULT_MANDELBULB_POWER,
        iterations: int = DEFAULT_MANDELBULB_ITERATIONS,
        bailout: float = DEFAULT_MANDELBULB_BAILOUT,
        position: Optional[Vec3] = None,
    ) -> None:
        super().__init__()
        self.power = power
        self.iterations = iterations
        self.bailout = bailout
        self.position = position or Vec3(0.0, 0.0, 0.0)
        self.tracker.mark_dirty("power")
        self.tracker.mark_dirty("iterations")
        self.tracker.mark_dirty("bailout")
        self.tracker.mark_dirty("position")

    def evaluate(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the distance to the Mandelbulb at point (x, y, z).

        Args:
            x, y, z: Point coordinates

        Returns:
            Estimated distance to surface
        """
        # Transform to local coordinates
        local_x = x - self.position.x
        local_y = y - self.position.y
        local_z = z - self.position.z

        return mandelbulb_distance_estimator(
            local_x, local_y, local_z,
            power=self.power,
            iterations=self.iterations,
            bailout=self.bailout,
        )

    def evaluate_point(self, point: Tuple[float, float, float]) -> float:
        """Evaluate distance at a point tuple."""
        return self.evaluate(point[0], point[1], point[2])

    @property
    def wgsl_function(self) -> str:
        """Return the WGSL function name."""
        return "sdf_mandelbulb"

    def label(self) -> str:
        """Return a human-readable label."""
        return f"Mandelbulb(n={self.power}, iter={self.iterations})"

    def clone(self) -> "MandelbulbSDF":
        """Create a copy of this node."""
        return MandelbulbSDF(
            power=self.power,
            iterations=self.iterations,
            bailout=self.bailout,
            position=self.position,
        )

    def children(self) -> Tuple[SDFNode, ...]:
        """Return child nodes (none for primitive)."""
        return ()

    def to_wgsl(self) -> str:
        """Generate WGSL code for this Mandelbulb."""
        return generate_mandelbulb_wgsl(
            power=self.power,
            iterations=self.iterations,
            bailout=self.bailout,
        )

    def get_config(self) -> MandelbulbConfig:
        """Return a configuration object for this Mandelbulb."""
        return MandelbulbConfig(
            power=self.power,
            iterations=self.iterations,
            bailout=self.bailout,
        )


# =============================================================================
# KIFS (Kaleidoscopic Iterated Function System) - T-DEMO-4.11
# =============================================================================

class KIFSFoldType:
    """Types of KIFS fold operations."""

    ABS = "abs"  # Simple absolute value fold: p.x = abs(p.x)
    MENGER = "menger"  # Menger sponge fold pattern
    SIERPINSKI = "sierpinski"  # Sierpinski tetrahedron fold
    BOX = "box"  # Box fold: conditional reflection
    SPHERE = "sphere"  # Sphere fold: inversion


@dataclass
class KIFSConfig:
    """Configuration for KIFS SDF."""

    iterations: int = DEFAULT_KIFS_ITERATIONS
    scale: float = DEFAULT_KIFS_SCALE
    fold_count: int = DEFAULT_KIFS_FOLD_COUNT
    translation: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    fold_type: str = KIFSFoldType.ABS
    min_radius: float = 0.5  # For sphere fold
    fixed_radius: float = 1.0  # For sphere fold

    def __post_init__(self):
        """Validate configuration."""
        if self.iterations < 1:
            raise ValueError(f"Iterations must be >= 1, got {self.iterations}")
        if self.scale <= 0.0:
            raise ValueError(f"Scale must be > 0.0, got {self.scale}")
        if self.fold_count < 1:
            raise ValueError(f"Fold count must be >= 1, got {self.fold_count}")


def kifs_fold_abs(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Apply absolute value fold for symmetry.

    This creates kaleidoscopic symmetry by reflecting coordinates
    across the coordinate planes.

    Args:
        x, y, z: Input coordinates

    Returns:
        Folded coordinates (abs(x), abs(y), abs(z))
    """
    return (abs(x), abs(y), abs(z))


def kifs_fold_menger(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Apply Menger sponge fold pattern.

    The Menger fold sorts coordinates and applies conditional swaps
    to create the characteristic cross pattern.

    Args:
        x, y, z: Input coordinates

    Returns:
        Folded coordinates following Menger pattern
    """
    # Sort by absolute value to ensure correct folding
    ax, ay, az = abs(x), abs(y), abs(z)

    # Swap to ensure proper ordering for Menger fold
    if ax < ay:
        ax, ay = ay, ax
        x, y = y, x
    if ay < az:
        ay, az = az, ay
        y, z = z, y
    if ax < ay:
        ax, ay = ay, ax
        x, y = y, x

    return (x, y, z)


def kifs_fold_sierpinski(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Apply Sierpinski tetrahedron fold.

    This fold creates the characteristic tetrahedral symmetry
    of the Sierpinski gasket in 3D.

    Args:
        x, y, z: Input coordinates

    Returns:
        Folded coordinates following Sierpinski pattern
    """
    # Fold across planes to create tetrahedral symmetry
    if x + y < 0.0:
        x, y = -y, -x
    if x + z < 0.0:
        x, z = -z, -x
    if y + z < 0.0:
        y, z = -z, -y

    return (x, y, z)


def kifs_fold_box(
    x: float, y: float, z: float,
    fold_limit: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Apply box fold for fractal generation.

    Points outside the fold limit are reflected back inside.

    Args:
        x, y, z: Input coordinates
        fold_limit: The folding boundary

    Returns:
        Folded coordinates
    """
    if x > fold_limit:
        x = 2.0 * fold_limit - x
    elif x < -fold_limit:
        x = -2.0 * fold_limit - x

    if y > fold_limit:
        y = 2.0 * fold_limit - y
    elif y < -fold_limit:
        y = -2.0 * fold_limit - y

    if z > fold_limit:
        z = 2.0 * fold_limit - z
    elif z < -fold_limit:
        z = -2.0 * fold_limit - z

    return (x, y, z)


def kifs_fold_sphere(
    x: float, y: float, z: float,
    min_radius: float = 0.5,
    fixed_radius: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Apply sphere fold (inversion).

    Points inside min_radius are scaled up, points between
    min_radius and fixed_radius are inverted.

    Args:
        x, y, z: Input coordinates
        min_radius: Inner radius for scaling
        fixed_radius: Outer radius for inversion

    Returns:
        Tuple of (folded_x, folded_y, folded_z, scale_factor)
    """
    r2 = x * x + y * y + z * z
    min_r2 = min_radius * min_radius
    fixed_r2 = fixed_radius * fixed_radius

    if r2 < min_r2:
        # Scale up if inside minimum radius
        scale = fixed_r2 / min_r2
        return (x * scale, y * scale, z * scale, scale)
    elif r2 < fixed_r2:
        # Inversion between min and fixed radius
        scale = fixed_r2 / r2
        return (x * scale, y * scale, z * scale, scale)
    else:
        # No change outside fixed radius
        return (x, y, z, 1.0)


def kifs_iteration(
    x: float, y: float, z: float,
    scale: float,
    tx: float, ty: float, tz: float,
    fold_type: str = KIFSFoldType.ABS,
    fold_count: int = 1,
    min_radius: float = 0.5,
    fixed_radius: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Perform one KIFS iteration: fold, scale, translate.

    Args:
        x, y, z: Input coordinates
        scale: Scaling factor for this iteration
        tx, ty, tz: Translation vector
        fold_type: Type of fold operation
        fold_count: Number of times to apply the fold
        min_radius: Minimum radius for sphere fold
        fixed_radius: Fixed radius for sphere fold

    Returns:
        Tuple of (new_x, new_y, new_z, accumulated_scale)
    """
    acc_scale = 1.0

    # Apply folds
    for _ in range(fold_count):
        if fold_type == KIFSFoldType.ABS:
            x, y, z = kifs_fold_abs(x, y, z)
        elif fold_type == KIFSFoldType.MENGER:
            x, y, z = kifs_fold_menger(x, y, z)
        elif fold_type == KIFSFoldType.SIERPINSKI:
            x, y, z = kifs_fold_sierpinski(x, y, z)
        elif fold_type == KIFSFoldType.BOX:
            x, y, z = kifs_fold_box(x, y, z)
        elif fold_type == KIFSFoldType.SPHERE:
            x, y, z, s = kifs_fold_sphere(x, y, z, min_radius, fixed_radius)
            acc_scale *= s

    # Scale
    x *= scale
    y *= scale
    z *= scale
    acc_scale *= scale

    # Translate (scale - 1 factor for proper IFS behavior)
    offset_factor = scale - 1.0
    x -= tx * offset_factor
    y -= ty * offset_factor
    z -= tz * offset_factor

    return (x, y, z, acc_scale)


def kifs_distance(
    x: float, y: float, z: float,
    iterations: int = DEFAULT_KIFS_ITERATIONS,
    scale: float = DEFAULT_KIFS_SCALE,
    translation: Optional[Vec3] = None,
    fold_count: int = DEFAULT_KIFS_FOLD_COUNT,
    fold_type: str = KIFSFoldType.ABS,
    base_distance: Optional[Callable[[float, float, float], float]] = None,
    min_radius: float = 0.5,
    fixed_radius: float = 1.0,
) -> float:
    """
    Compute distance to KIFS fractal surface.

    The distance is computed by iteratively folding space, then
    evaluating a base SDF and compensating for scale.

    Distance compensation: d / scale^iterations

    Args:
        x, y, z: Point to evaluate
        iterations: Number of KIFS iterations
        scale: Scaling factor per iteration
        translation: Translation vector per iteration
        fold_count: Number of folds per iteration
        fold_type: Type of fold operation
        base_distance: Base SDF to evaluate (default: box)
        min_radius: Minimum radius for sphere fold
        fixed_radius: Fixed radius for sphere fold

    Returns:
        Estimated distance to the KIFS fractal surface
    """
    translation = translation or Vec3(1.0, 1.0, 1.0)
    tx, ty, tz = translation.x, translation.y, translation.z

    # Track accumulated scale for distance compensation
    accumulated_scale = 1.0

    # Apply KIFS iterations
    for _ in range(iterations):
        x, y, z, iter_scale = kifs_iteration(
            x, y, z, scale, tx, ty, tz,
            fold_type=fold_type,
            fold_count=fold_count,
            min_radius=min_radius,
            fixed_radius=fixed_radius,
        )
        accumulated_scale *= iter_scale

    # Evaluate base SDF (default: unit sphere)
    if base_distance is not None:
        base_d = base_distance(x, y, z)
    else:
        # Default: unit sphere
        base_d = math.sqrt(x * x + y * y + z * z) - 1.0

    # Distance scaling compensation: d / scale^iterations
    return base_d / accumulated_scale


class KIFSSDF(SDFNode, metaclass=SDFNodeMeta):
    """
    KIFS (Kaleidoscopic Iterated Function System) SDF node.

    Creates fractal geometry through iterative folding and scaling:
    1. Fold operations: abs(p.x), mirror across planes
    2. Scale by a factor each iteration
    3. Translate to create offset
    4. Distance scaling compensation: d / scale^iterations

    Attributes:
        iterations: Number of KIFS iterations
        scale: Scaling factor per iteration
        fold_count: Number of fold operations per iteration
        translation: Translation vector per iteration
        fold_type: Type of fold operation (abs, menger, sierpinski, box, sphere)
        position: Center position
        min_radius: Minimum radius for sphere fold
        fixed_radius: Fixed radius for sphere fold
    """

    __slots__ = (
        "iterations", "scale", "fold_count", "translation",
        "fold_type", "position", "min_radius", "fixed_radius",
        "_base_sdf",
    )

    iterations: int
    scale: float
    fold_count: int
    translation: Vec3
    fold_type: str
    position: Vec3
    min_radius: float
    fixed_radius: float
    _base_sdf: Optional[Callable[[float, float, float], float]]

    def __init__(
        self,
        iterations: int = DEFAULT_KIFS_ITERATIONS,
        scale: float = DEFAULT_KIFS_SCALE,
        fold_count: int = DEFAULT_KIFS_FOLD_COUNT,
        translation: Optional[Vec3] = None,
        fold_type: str = KIFSFoldType.ABS,
        position: Optional[Vec3] = None,
        min_radius: float = 0.5,
        fixed_radius: float = 1.0,
        base_sdf: Optional[Callable[[float, float, float], float]] = None,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.scale = scale
        self.fold_count = fold_count
        self.translation = translation or Vec3(1.0, 1.0, 1.0)
        self.fold_type = fold_type
        self.position = position or Vec3(0.0, 0.0, 0.0)
        self.min_radius = min_radius
        self.fixed_radius = fixed_radius
        self._base_sdf = base_sdf

        # Mark all fields dirty
        self.tracker.mark_dirty("iterations")
        self.tracker.mark_dirty("scale")
        self.tracker.mark_dirty("fold_count")
        self.tracker.mark_dirty("translation")
        self.tracker.mark_dirty("fold_type")
        self.tracker.mark_dirty("position")
        self.tracker.mark_dirty("min_radius")
        self.tracker.mark_dirty("fixed_radius")

    def evaluate(self, x: float, y: float, z: float) -> float:
        """
        Evaluate the distance to the KIFS fractal at point (x, y, z).

        Args:
            x, y, z: Point coordinates

        Returns:
            Estimated distance to surface
        """
        # Transform to local coordinates
        local_x = x - self.position.x
        local_y = y - self.position.y
        local_z = z - self.position.z

        return kifs_distance(
            local_x, local_y, local_z,
            iterations=self.iterations,
            scale=self.scale,
            translation=self.translation,
            fold_count=self.fold_count,
            fold_type=self.fold_type,
            base_distance=self._base_sdf,
            min_radius=self.min_radius,
            fixed_radius=self.fixed_radius,
        )

    def evaluate_point(self, point: Tuple[float, float, float]) -> float:
        """Evaluate distance at a point tuple."""
        return self.evaluate(point[0], point[1], point[2])

    def set_base_sdf(
        self, sdf: Callable[[float, float, float], float]
    ) -> "KIFSSDF":
        """
        Set the base SDF to evaluate after folding.

        Args:
            sdf: A function (x, y, z) -> distance

        Returns:
            Self for chaining
        """
        self._base_sdf = sdf
        self.tracker.mark_dirty("_base_sdf")
        return self

    @property
    def wgsl_function(self) -> str:
        """Return the WGSL function name."""
        return "sdf_kifs"

    def label(self) -> str:
        """Return a human-readable label."""
        return f"KIFS(iter={self.iterations}, scale={self.scale}, folds={self.fold_count})"

    def clone(self) -> "KIFSSDF":
        """Create a copy of this node."""
        return KIFSSDF(
            iterations=self.iterations,
            scale=self.scale,
            fold_count=self.fold_count,
            translation=self.translation,
            fold_type=self.fold_type,
            position=self.position,
            min_radius=self.min_radius,
            fixed_radius=self.fixed_radius,
            base_sdf=self._base_sdf,
        )

    def children(self) -> Tuple[SDFNode, ...]:
        """Return child nodes (none for primitive)."""
        return ()

    def to_wgsl(self) -> str:
        """Generate WGSL code for this KIFS."""
        return generate_kifs_wgsl(
            iterations=self.iterations,
            scale=self.scale,
            fold_count=self.fold_count,
            translation=self.translation,
            fold_type=self.fold_type,
            min_radius=self.min_radius,
            fixed_radius=self.fixed_radius,
        )

    def get_config(self) -> KIFSConfig:
        """Return a configuration object for this KIFS."""
        return KIFSConfig(
            iterations=self.iterations,
            scale=self.scale,
            fold_count=self.fold_count,
            translation=self.translation,
            fold_type=self.fold_type,
            min_radius=self.min_radius,
            fixed_radius=self.fixed_radius,
        )


# =============================================================================
# Base class for fractal SDF nodes
# =============================================================================

class FractalSDFNode(SDFNode, metaclass=SDFNodeMeta):
    """
    Base class for fractal SDF nodes.

    Provides common functionality for fractal distance fields
    including iteration tracking and distance compensation.
    """

    __slots__ = ("_accumulated_scale",)

    def __init__(self) -> None:
        super().__init__()
        self._accumulated_scale = 1.0

    @abstractmethod
    def evaluate(self, x: float, y: float, z: float) -> float:
        """Evaluate the SDF at the given point."""
        pass

    @property
    def accumulated_scale(self) -> float:
        """Return the accumulated scale from iterations."""
        return self._accumulated_scale


# =============================================================================
# WGSL Code Generation
# =============================================================================

MANDELBULB_WGSL_FUNCTION = """\
fn sdf_mandelbulb(p: vec3<f32>, power: f32, iterations: i32, bailout: f32) -> f32 {
    var z = vec3<f32>(0.0, 0.0, 0.0);
    let c = p;
    var dr = 1.0;
    var r = 0.0;

    for (var i = 0; i < iterations; i = i + 1) {
        r = length(z);
        if (r > bailout) {
            break;
        }

        // Spherical coordinates
        let theta = atan2(z.y, z.x);
        let phi = acos(clamp(z.z / max(r, 1e-10), -1.0, 1.0));

        // Update derivative
        dr = power * pow(r, power - 1.0) * dr + 1.0;

        // Power formula: z = z^n + c
        let new_r = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        let sin_phi = sin(new_phi);
        z = vec3<f32>(
            new_r * sin_phi * cos(new_theta),
            new_r * sin_phi * sin(new_theta),
            new_r * cos(new_phi)
        ) + c;
    }

    // Distance estimator: 0.5 * log(r) * r / dr
    return 0.5 * log(max(r, 1e-10)) * r / max(dr, 1e-10);
}"""

KIFS_FOLD_ABS_WGSL = """\
fn kifs_fold_abs(p: vec3<f32>) -> vec3<f32> {
    return abs(p);
}"""

KIFS_FOLD_SIERPINSKI_WGSL = """\
fn kifs_fold_sierpinski(p: vec3<f32>) -> vec3<f32> {
    var q = p;
    if (q.x + q.y < 0.0) {
        let tmp = q.x;
        q.x = -q.y;
        q.y = -tmp;
    }
    if (q.x + q.z < 0.0) {
        let tmp = q.x;
        q.x = -q.z;
        q.z = -tmp;
    }
    if (q.y + q.z < 0.0) {
        let tmp = q.y;
        q.y = -q.z;
        q.z = -tmp;
    }
    return q;
}"""

KIFS_FOLD_BOX_WGSL = """\
fn kifs_fold_box(p: vec3<f32>, fold_limit: f32) -> vec3<f32> {
    var q = p;
    if (q.x > fold_limit) {
        q.x = 2.0 * fold_limit - q.x;
    } else if (q.x < -fold_limit) {
        q.x = -2.0 * fold_limit - q.x;
    }
    if (q.y > fold_limit) {
        q.y = 2.0 * fold_limit - q.y;
    } else if (q.y < -fold_limit) {
        q.y = -2.0 * fold_limit - q.y;
    }
    if (q.z > fold_limit) {
        q.z = 2.0 * fold_limit - q.z;
    } else if (q.z < -fold_limit) {
        q.z = -2.0 * fold_limit - q.z;
    }
    return q;
}"""

KIFS_FOLD_SPHERE_WGSL = """\
fn kifs_fold_sphere(p: vec3<f32>, min_radius: f32, fixed_radius: f32) -> vec4<f32> {
    let r2 = dot(p, p);
    let min_r2 = min_radius * min_radius;
    let fixed_r2 = fixed_radius * fixed_radius;

    if (r2 < min_r2) {
        let scale = fixed_r2 / min_r2;
        return vec4<f32>(p * scale, scale);
    } else if (r2 < fixed_r2) {
        let scale = fixed_r2 / r2;
        return vec4<f32>(p * scale, scale);
    }
    return vec4<f32>(p, 1.0);
}"""

KIFS_WGSL_FUNCTION = """\
fn sdf_kifs(p: vec3<f32>, iterations: i32, scale: f32, translation: vec3<f32>) -> f32 {
    var q = p;
    var accumulated_scale = 1.0;

    for (var i = 0; i < iterations; i = i + 1) {
        // Absolute value fold for symmetry
        q = abs(q);

        // Scale
        q = q * scale;
        accumulated_scale = accumulated_scale * scale;

        // Translate
        let offset = translation * (scale - 1.0);
        q = q - offset;
    }

    // Base SDF: unit sphere
    let base_d = length(q) - 1.0;

    // Distance compensation
    return base_d / accumulated_scale;
}"""


def generate_mandelbulb_wgsl(
    power: float = DEFAULT_MANDELBULB_POWER,
    iterations: int = DEFAULT_MANDELBULB_ITERATIONS,
    bailout: float = DEFAULT_MANDELBULB_BAILOUT,
    function_name: str = "sdf_mandelbulb",
) -> str:
    """
    Generate WGSL code for Mandelbulb SDF.

    Args:
        power: Mandelbulb power
        iterations: Maximum iterations
        bailout: Escape radius
        function_name: Name for the generated function

    Returns:
        WGSL code string
    """
    return f"""\
fn {function_name}(p: vec3<f32>) -> f32 {{
    let power = {power};
    let iterations = {iterations};
    let bailout = {bailout};

    var z = vec3<f32>(0.0, 0.0, 0.0);
    let c = p;
    var dr = 1.0;
    var r = 0.0;

    for (var i = 0; i < iterations; i = i + 1) {{
        r = length(z);
        if (r > bailout) {{
            break;
        }}

        // Spherical coordinates
        let theta = atan2(z.y, z.x);
        let phi = acos(clamp(z.z / max(r, 1e-10), -1.0, 1.0));

        // Update derivative
        dr = power * pow(r, power - 1.0) * dr + 1.0;

        // Power formula: z = z^n + c
        let new_r = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        let sin_phi = sin(new_phi);
        z = vec3<f32>(
            new_r * sin_phi * cos(new_theta),
            new_r * sin_phi * sin(new_theta),
            new_r * cos(new_phi)
        ) + c;
    }}

    // Distance estimator: 0.5 * log(r) * r / dr
    return 0.5 * log(max(r, 1e-10)) * r / max(dr, 1e-10);
}}"""


def generate_kifs_wgsl(
    iterations: int = DEFAULT_KIFS_ITERATIONS,
    scale: float = DEFAULT_KIFS_SCALE,
    fold_count: int = DEFAULT_KIFS_FOLD_COUNT,
    translation: Optional[Vec3] = None,
    fold_type: str = KIFSFoldType.ABS,
    min_radius: float = 0.5,
    fixed_radius: float = 1.0,
    function_name: str = "sdf_kifs",
) -> str:
    """
    Generate WGSL code for KIFS SDF.

    Args:
        iterations: Number of KIFS iterations
        scale: Scaling factor
        fold_count: Number of folds per iteration
        translation: Translation vector
        fold_type: Type of fold operation
        min_radius: Minimum radius for sphere fold
        fixed_radius: Fixed radius for sphere fold
        function_name: Name for the generated function

    Returns:
        WGSL code string
    """
    translation = translation or Vec3(1.0, 1.0, 1.0)

    # Generate fold code based on type
    if fold_type == KIFSFoldType.ABS:
        fold_code = "q = abs(q);"
    elif fold_type == KIFSFoldType.SIERPINSKI:
        fold_code = """\
        // Sierpinski fold
        if (q.x + q.y < 0.0) {
            let tmp = q.x;
            q.x = -q.y;
            q.y = -tmp;
        }
        if (q.x + q.z < 0.0) {
            let tmp = q.x;
            q.x = -q.z;
            q.z = -tmp;
        }
        if (q.y + q.z < 0.0) {
            let tmp = q.y;
            q.y = -q.z;
            q.z = -tmp;
        }"""
    elif fold_type == KIFSFoldType.BOX:
        fold_code = """\
        // Box fold
        let fold_limit = 1.0;
        if (q.x > fold_limit) { q.x = 2.0 * fold_limit - q.x; }
        else if (q.x < -fold_limit) { q.x = -2.0 * fold_limit - q.x; }
        if (q.y > fold_limit) { q.y = 2.0 * fold_limit - q.y; }
        else if (q.y < -fold_limit) { q.y = -2.0 * fold_limit - q.y; }
        if (q.z > fold_limit) { q.z = 2.0 * fold_limit - q.z; }
        else if (q.z < -fold_limit) { q.z = -2.0 * fold_limit - q.z; }"""
    elif fold_type == KIFSFoldType.SPHERE:
        fold_code = f"""\
        // Sphere fold
        let r2 = dot(q, q);
        let min_r2 = {min_radius} * {min_radius};
        let fixed_r2 = {fixed_radius} * {fixed_radius};
        if (r2 < min_r2) {{
            let s = fixed_r2 / min_r2;
            q = q * s;
            accumulated_scale = accumulated_scale * s;
        }} else if (r2 < fixed_r2) {{
            let s = fixed_r2 / r2;
            q = q * s;
            accumulated_scale = accumulated_scale * s;
        }}"""
    else:
        fold_code = "q = abs(q);"

    # Build fold loop if fold_count > 1
    if fold_count > 1:
        fold_section = f"""\
        for (var f = 0; f < {fold_count}; f = f + 1) {{
            {fold_code}
        }}"""
    else:
        fold_section = fold_code

    return f"""\
fn {function_name}(p: vec3<f32>) -> f32 {{
    let iterations = {iterations};
    let scale = {scale}f;
    let translation = vec3<f32>({translation.x}, {translation.y}, {translation.z});

    var q = p;
    var accumulated_scale = 1.0;

    for (var i = 0; i < iterations; i = i + 1) {{
        // Fold operations
        {fold_section}

        // Scale
        q = q * scale;
        accumulated_scale = accumulated_scale * scale;

        // Translate
        let offset = translation * (scale - 1.0);
        q = q - offset;
    }}

    // Base SDF: unit sphere
    let base_d = length(q) - 1.0;

    // Distance compensation: d / scale^iterations
    return base_d / accumulated_scale;
}}"""


# =============================================================================
# Helper functions for testing
# =============================================================================

def sdf_sphere(x: float, y: float, z: float, radius: float = 1.0) -> float:
    """Simple sphere SDF for testing."""
    return math.sqrt(x * x + y * y + z * z) - radius


def sdf_box(
    x: float, y: float, z: float,
    bx: float = 1.0, by: float = 1.0, bz: float = 1.0,
) -> float:
    """Simple box SDF for testing."""
    qx = abs(x) - bx
    qy = abs(y) - by
    qz = abs(z) - bz

    outside = math.sqrt(
        max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2
    )
    inside = min(max(qx, max(qy, qz)), 0.0)

    return outside + inside
