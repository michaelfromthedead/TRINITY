"""
TRINITY Vegetation SDF Library (T-DEMO-4.5 and T-DEMO-4.6)

This module provides Signed Distance Field implementations for vegetation:

T-DEMO-4.5: Tree SDF
    - TreeSDF class combining primitives for tree shape
    - Trunk: cylinder or tapered cone
    - Canopy: union of spheres or ellipsoids
    - Branches: optional capsules connecting trunk to canopy
    - Parameters: trunk_height, trunk_radius, canopy_spheres, branch_count

T-DEMO-4.6: Infinite Forest via Domain Repetition
    - ForestSDF class using domain repetition
    - Per-cell pseudo-random tree variation:
        - Tree type selection
        - Height/width variation
        - Position jitter within cell
        - Rotation variation
    - Cell hash for deterministic randomness

Reference:
    - Inigo Quilez -- Distance Functions
    - Domain repetition: p_cell = fract(p / cell_size) * cell_size - cell_size * 0.5
    - Cell ID hash: hash = floor(p / cell_size)
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from .sdf_ast import (
    SDFNode,
    Vec3,
    Mirror,
    Tracker,
)


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Enums
    "TrunkType",
    "CanopyType",
    # Core SDF Classes (T-DEMO-4.5)
    "TreeSDF",
    "TreeConfig",
    "BranchConfig",
    # Forest SDF Classes (T-DEMO-4.6)
    "ForestSDF",
    "ForestConfig",
    "TreeVariation",
    # Hash functions
    "cell_hash",
    "cell_hash_float",
    "hash_to_float",
    # WGSL generation
    "generate_tree_wgsl",
    "generate_forest_wgsl",
    # Python SDF functions (for testing)
    "sdf_tree",
    "sdf_forest",
]


# =============================================================================
# Tree Component Enumerations
# =============================================================================

class TrunkType(Enum):
    """Type of tree trunk geometry."""
    CYLINDER = auto()
    TAPERED_CONE = auto()

    def to_wgsl(self) -> str:
        """Convert to WGSL constant name."""
        return f"TRUNK_TYPE_{self.name}"


class CanopyType(Enum):
    """Type of tree canopy geometry."""
    SPHERES = auto()
    ELLIPSOIDS = auto()

    def to_wgsl(self) -> str:
        """Convert to WGSL constant name."""
        return f"CANOPY_TYPE_{self.name}"


# =============================================================================
# Branch Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class BranchConfig:
    """Configuration for tree branches.

    Branches are modeled as capsules connecting the trunk to canopy spheres.

    Attributes:
        count: Number of branches (0 for no branches)
        radius: Radius of branch capsules
        attachment_height: Height on trunk where branches start (0-1 normalized)
        spread_angle: Maximum angle from vertical in radians
    """
    count: int = 0
    radius: float = 0.05
    attachment_height: float = 0.6
    spread_angle: float = 0.7854  # 45 degrees

    def __post_init__(self) -> None:
        """Validate branch configuration."""
        if self.count < 0:
            raise ValueError(f"Branch count must be non-negative, got {self.count}")
        if self.radius <= 0:
            raise ValueError(f"Branch radius must be positive, got {self.radius}")
        if not 0.0 <= self.attachment_height <= 1.0:
            raise ValueError(f"Attachment height must be in [0, 1], got {self.attachment_height}")
        if self.spread_angle < 0:
            raise ValueError(f"Spread angle must be non-negative, got {self.spread_angle}")


# =============================================================================
# Tree Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class TreeConfig:
    """Configuration for a tree SDF.

    Attributes:
        trunk_height: Height of the trunk
        trunk_radius: Base radius of the trunk
        trunk_type: Type of trunk geometry (cylinder or tapered cone)
        trunk_taper: For tapered cones, ratio of top to bottom radius (0-1)
        canopy_type: Type of canopy geometry (spheres or ellipsoids)
        canopy_spheres: Number of spheres/ellipsoids in canopy
        canopy_radius: Base radius of canopy elements
        canopy_height_offset: Height offset of canopy center above trunk
        canopy_spread: Horizontal spread of canopy elements
        branches: Branch configuration
        smooth_k: Smoothing factor for smooth union operations
    """
    trunk_height: float = 2.0
    trunk_radius: float = 0.15
    trunk_type: TrunkType = TrunkType.CYLINDER
    trunk_taper: float = 0.7
    canopy_type: CanopyType = CanopyType.SPHERES
    canopy_spheres: int = 5
    canopy_radius: float = 0.8
    canopy_height_offset: float = 0.3
    canopy_spread: float = 0.6
    branches: BranchConfig = field(default_factory=BranchConfig)
    smooth_k: float = 0.15

    def __post_init__(self) -> None:
        """Validate tree configuration."""
        if self.trunk_height <= 0:
            raise ValueError(f"Trunk height must be positive, got {self.trunk_height}")
        if self.trunk_radius <= 0:
            raise ValueError(f"Trunk radius must be positive, got {self.trunk_radius}")
        if not 0.0 < self.trunk_taper <= 1.0:
            raise ValueError(f"Trunk taper must be in (0, 1], got {self.trunk_taper}")
        if self.canopy_spheres < 1:
            raise ValueError(f"Canopy spheres must be at least 1, got {self.canopy_spheres}")
        if self.canopy_radius <= 0:
            raise ValueError(f"Canopy radius must be positive, got {self.canopy_radius}")
        if self.smooth_k < 0:
            raise ValueError(f"Smooth k must be non-negative, got {self.smooth_k}")


# =============================================================================
# Python SDF Primitives (for testing)
# =============================================================================

def sdf_sphere(p: Tuple[float, float, float], radius: float) -> float:
    """Signed distance to a sphere centered at origin."""
    length = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    return length - radius


def sdf_cylinder(p: Tuple[float, float, float], radius: float, half_height: float) -> float:
    """Signed distance to a capped cylinder along Y axis."""
    dx = math.sqrt(p[0] ** 2 + p[2] ** 2) - radius
    dy = abs(p[1]) - half_height
    outside = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2)
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def sdf_cone(
    p: Tuple[float, float, float],
    bottom_radius: float,
    top_radius: float,
    half_height: float,
) -> float:
    """Signed distance to a tapered cone (frustum) along Y axis.

    The cone is centered at origin with:
    - Bottom at y = -half_height with radius bottom_radius
    - Top at y = +half_height with radius top_radius
    """
    # Radial distance in xz plane
    r = math.sqrt(p[0] ** 2 + p[2] ** 2)

    # Linearly interpolate radius at current height
    t = (p[1] + half_height) / (2.0 * half_height) if half_height > 0 else 0.5
    t = max(0.0, min(1.0, t))
    expected_r = bottom_radius + t * (top_radius - bottom_radius)

    # Distance to cone surface (approximation)
    dr = r - expected_r
    dy = abs(p[1]) - half_height

    if dy > 0:
        # Outside height bounds
        if dr > 0:
            return math.sqrt(dr ** 2 + dy ** 2)
        else:
            return dy
    else:
        # Inside height bounds
        if dr > 0:
            return dr
        else:
            # Inside cone - return negative of minimum distance
            return max(dr, dy)


def sdf_capsule(
    p: Tuple[float, float, float],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    radius: float,
) -> float:
    """Signed distance to a capsule from point a to point b."""
    pa = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    ba = (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    dot_pa_ba = pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]
    dot_ba_ba = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]

    if dot_ba_ba < 1e-10:
        # Degenerate capsule (a == b), just use sphere
        return math.sqrt(pa[0] ** 2 + pa[1] ** 2 + pa[2] ** 2) - radius

    h = max(0.0, min(1.0, dot_pa_ba / dot_ba_ba))

    dx = pa[0] - ba[0] * h
    dy = pa[1] - ba[1] * h
    dz = pa[2] - ba[2] * h

    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) - radius


def sdf_ellipsoid(p: Tuple[float, float, float], radii: Tuple[float, float, float]) -> float:
    """Signed distance to an ellipsoid (approximation)."""
    # Normalize to unit sphere space
    px = p[0] / radii[0] if radii[0] > 0 else 0
    py = p[1] / radii[1] if radii[1] > 0 else 0
    pz = p[2] / radii[2] if radii[2] > 0 else 0
    k0 = math.sqrt(px ** 2 + py ** 2 + pz ** 2)

    px2 = p[0] / (radii[0] ** 2) if radii[0] > 0 else 0
    py2 = p[1] / (radii[1] ** 2) if radii[1] > 0 else 0
    pz2 = p[2] / (radii[2] ** 2) if radii[2] > 0 else 0
    k1 = math.sqrt(px2 ** 2 + py2 ** 2 + pz2 ** 2)

    if k1 < 1e-10:
        return -min(radii)

    return k0 * (k0 - 1.0) / k1


def smooth_union(d1: float, d2: float, k: float) -> float:
    """Smooth union of two signed distances."""
    # Handle infinity cases (initial accumulator)
    if d1 == float("inf"):
        return d2
    if d2 == float("inf"):
        return d1
    if k <= 0:
        return min(d1, d2)
    h = max(0.0, min(1.0, 0.5 + 0.5 * (d2 - d1) / k))
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


# =============================================================================
# Hash Functions for Deterministic Randomness
# =============================================================================

def cell_hash(cell_id: Tuple[int, int, int]) -> int:
    """Compute a deterministic hash from a 3D cell ID.

    Uses a simple but effective hash combining technique.

    Args:
        cell_id: Integer coordinates of the cell (ix, iy, iz)

    Returns:
        A 32-bit unsigned integer hash value
    """
    # Constants from xxHash / FNV-inspired mixing
    PRIME1 = 0x9E3779B9  # 2654435769
    PRIME2 = 0x85EBCA6B  # 2246822507
    PRIME3 = 0xC2B2AE35  # 3266489909

    x = cell_id[0] & 0xFFFFFFFF
    y = cell_id[1] & 0xFFFFFFFF
    z = cell_id[2] & 0xFFFFFFFF

    # Mix the coordinates
    h = (x * PRIME1) & 0xFFFFFFFF
    h = ((h ^ (y * PRIME2)) * PRIME3) & 0xFFFFFFFF
    h = ((h ^ (z * PRIME1)) * PRIME2) & 0xFFFFFFFF

    # Final avalanche
    h = ((h ^ (h >> 16)) * PRIME3) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF

    return h


def hash_to_float(h: int) -> float:
    """Convert a hash value to a float in [0, 1)."""
    # Use 0x80000000 as divisor to ensure result is strictly < 1.0
    return (h & 0x7FFFFFFF) / 0x80000000


def cell_hash_float(cell_id: Tuple[int, int, int], channel: int = 0) -> float:
    """Compute a float hash in [0, 1) from cell ID and channel.

    Different channels give independent pseudo-random values for the same cell.

    Args:
        cell_id: Integer coordinates of the cell
        channel: Channel index for multiple independent values

    Returns:
        Float in [0, 1)
    """
    modified_id = (cell_id[0] + channel * 1337, cell_id[1], cell_id[2])
    return hash_to_float(cell_hash(modified_id))


# =============================================================================
# TreeSDF - Tree Signed Distance Function (T-DEMO-4.5)
# =============================================================================

class TreeSDF:
    """Signed Distance Function for a procedural tree.

    Combines primitives to create a tree shape:
    - Trunk: cylinder or tapered cone
    - Canopy: smooth union of spheres or ellipsoids
    - Branches: optional capsules connecting trunk to canopy

    The tree is centered at the origin with:
    - Base of trunk at y=0
    - Tree grows upward along +Y axis

    Attributes:
        config: Tree configuration parameters
        position: Position offset (Vec3)
    """

    __slots__ = ("config", "position", "_mirror", "_tracker", "_node_id", "_dirty_fields", "_version")

    _instance_counter: ClassVar[int] = 0
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: Optional[TreeConfig] = None,
        position: Optional[Vec3] = None,
    ) -> None:
        """Initialize tree SDF.

        Args:
            config: Tree configuration. Uses defaults if None.
            position: Position offset. Uses origin if None.
        """
        with TreeSDF._counter_lock:
            TreeSDF._instance_counter += 1
            self._node_id = TreeSDF._instance_counter

        self.config = config or TreeConfig()
        self.position = position or Vec3()
        self._dirty_fields: set = {"config", "position"}
        self._version: int = 1
        self._mirror: Optional[Mirror] = None
        self._tracker: Optional[Tracker] = None

    @property
    def mirror(self) -> Mirror:
        """Get Mirror instance for introspection."""
        if self._mirror is None:
            self._mirror = _TreeMirror(self)
        return self._mirror

    @property
    def tracker(self) -> Tracker:
        """Get Tracker instance for dirty tracking."""
        if self._tracker is None:
            self._tracker = _TreeTracker(self)
        return self._tracker

    def evaluate(self, p: Tuple[float, float, float]) -> float:
        """Evaluate the signed distance at point p.

        Args:
            p: Query point in 3D space

        Returns:
            Signed distance to tree surface (negative inside)
        """
        # Translate point to tree-local coordinates
        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return sdf_tree(local_p, self.config)

    def evaluate_trunk(self, p: Tuple[float, float, float]) -> float:
        """Evaluate only the trunk SDF.

        Args:
            p: Query point in 3D space

        Returns:
            Signed distance to trunk surface
        """
        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return _sdf_trunk(local_p, self.config)

    def evaluate_canopy(self, p: Tuple[float, float, float]) -> float:
        """Evaluate only the canopy SDF.

        Args:
            p: Query point in 3D space

        Returns:
            Signed distance to canopy surface
        """
        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return _sdf_canopy(local_p, self.config)

    def evaluate_branches(self, p: Tuple[float, float, float]) -> float:
        """Evaluate only the branches SDF.

        Args:
            p: Query point in 3D space

        Returns:
            Signed distance to branches surface (inf if no branches)
        """
        if self.config.branches.count == 0:
            return float("inf")

        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return _sdf_branches(local_p, self.config)

    def get_canopy_sphere_positions(self) -> List[Tuple[float, float, float]]:
        """Get the positions of canopy spheres.

        Returns:
            List of (x, y, z) positions for each canopy sphere
        """
        return _get_canopy_positions(self.config)

    def get_branch_endpoints(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get the endpoints of each branch capsule.

        Returns:
            List of ((ax, ay, az), (bx, by, bz)) endpoint pairs
        """
        return _get_branch_endpoints(self.config)

    def to_wgsl(self, name: str = "tree") -> str:
        """Generate WGSL code for this tree SDF.

        Args:
            name: Function name prefix

        Returns:
            WGSL source code for the tree SDF
        """
        return generate_tree_wgsl(self.config, name)

    def label(self) -> str:
        """Return a short descriptive label."""
        return f"TreeSDF(h={self.config.trunk_height}, canopy={self.config.canopy_spheres})"

    def __repr__(self) -> str:
        return f"<{self.label()} id={self._node_id}>"


# =============================================================================
# Tree SDF Implementation Functions
# =============================================================================

def _sdf_trunk(p: Tuple[float, float, float], config: TreeConfig) -> float:
    """Evaluate trunk SDF.

    The trunk base is at y=0, top at y=trunk_height.
    We shift the center to half-height.
    """
    half_h = config.trunk_height / 2.0
    # Shift point so trunk center is at origin
    shifted_p = (p[0], p[1] - half_h, p[2])

    if config.trunk_type == TrunkType.CYLINDER:
        return sdf_cylinder(shifted_p, config.trunk_radius, half_h)
    else:
        # Tapered cone
        top_radius = config.trunk_radius * config.trunk_taper
        return sdf_cone(shifted_p, config.trunk_radius, top_radius, half_h)


def _get_canopy_positions(config: TreeConfig) -> List[Tuple[float, float, float]]:
    """Calculate positions for canopy spheres/ellipsoids."""
    positions = []
    canopy_center_y = config.trunk_height + config.canopy_height_offset

    if config.canopy_spheres == 1:
        # Single centered sphere
        positions.append((0.0, canopy_center_y, 0.0))
    else:
        # Central sphere plus ring around it
        positions.append((0.0, canopy_center_y, 0.0))

        remaining = config.canopy_spheres - 1
        for i in range(remaining):
            angle = 2.0 * math.pi * i / remaining
            x = config.canopy_spread * math.cos(angle)
            z = config.canopy_spread * math.sin(angle)
            # Vary height slightly based on position
            y_offset = 0.1 * math.sin(angle * 2)
            positions.append((x, canopy_center_y + y_offset, z))

    return positions


def _sdf_canopy(p: Tuple[float, float, float], config: TreeConfig) -> float:
    """Evaluate canopy SDF (smooth union of spheres/ellipsoids)."""
    positions = _get_canopy_positions(config)

    d = float("inf")
    for pos in positions:
        local_p = (p[0] - pos[0], p[1] - pos[1], p[2] - pos[2])

        if config.canopy_type == CanopyType.SPHERES:
            sphere_d = sdf_sphere(local_p, config.canopy_radius)
        else:
            # Ellipsoid - flattened vertically
            radii = (config.canopy_radius, config.canopy_radius * 0.7, config.canopy_radius)
            sphere_d = sdf_ellipsoid(local_p, radii)

        d = smooth_union(d, sphere_d, config.smooth_k)

    return d


def _get_branch_endpoints(config: TreeConfig) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Calculate branch capsule endpoints."""
    if config.branches.count == 0:
        return []

    endpoints = []
    attachment_y = config.trunk_height * config.branches.attachment_height
    canopy_positions = _get_canopy_positions(config)

    # Skip central canopy sphere if present
    canopy_targets = canopy_positions[1:] if len(canopy_positions) > 1 else canopy_positions

    for i in range(config.branches.count):
        # Distribute branches around trunk
        angle = 2.0 * math.pi * i / config.branches.count

        # Start point on trunk
        start_x = config.trunk_radius * 0.9 * math.cos(angle)
        start_z = config.trunk_radius * 0.9 * math.sin(angle)
        start = (start_x, attachment_y, start_z)

        # End point towards a canopy sphere (cycle through targets)
        if canopy_targets:
            target_idx = i % len(canopy_targets)
            target = canopy_targets[target_idx]
            # End point is partway towards the canopy sphere
            t = 0.7
            end = (
                start[0] + t * (target[0] - start[0]),
                start[1] + t * (target[1] - start[1]),
                start[2] + t * (target[2] - start[2]),
            )
        else:
            # No canopy targets, extend outward and upward
            spread = config.trunk_height * 0.3
            end = (
                spread * math.cos(angle),
                attachment_y + config.trunk_height * 0.4,
                spread * math.sin(angle),
            )

        endpoints.append((start, end))

    return endpoints


def _sdf_branches(p: Tuple[float, float, float], config: TreeConfig) -> float:
    """Evaluate branches SDF (union of capsules)."""
    endpoints = _get_branch_endpoints(config)

    if not endpoints:
        return float("inf")

    d = float("inf")
    for start, end in endpoints:
        capsule_d = sdf_capsule(p, start, end, config.branches.radius)
        d = min(d, capsule_d)

    return d


def sdf_tree(p: Tuple[float, float, float], config: TreeConfig) -> float:
    """Evaluate the complete tree SDF.

    Combines trunk, canopy, and optional branches using smooth union.

    Args:
        p: Query point (tree-local coordinates, base at origin)
        config: Tree configuration

    Returns:
        Signed distance to tree surface
    """
    trunk_d = _sdf_trunk(p, config)
    canopy_d = _sdf_canopy(p, config)

    # Combine trunk and canopy with smooth union
    d = smooth_union(trunk_d, canopy_d, config.smooth_k)

    # Add branches if present
    if config.branches.count > 0:
        branches_d = _sdf_branches(p, config)
        d = smooth_union(d, branches_d, config.smooth_k * 0.5)

    return d


# =============================================================================
# Forest Configuration (T-DEMO-4.6)
# =============================================================================

@dataclass(frozen=True, slots=True)
class TreeVariation:
    """Variation parameters for per-cell tree generation.

    Attributes:
        height_min: Minimum trunk height multiplier
        height_max: Maximum trunk height multiplier
        width_min: Minimum trunk width multiplier
        width_max: Maximum trunk width multiplier
        canopy_min: Minimum canopy sphere count
        canopy_max: Maximum canopy sphere count
        position_jitter: Maximum position jitter within cell (0-0.5)
        rotation_enabled: Whether to apply random rotation
    """
    height_min: float = 0.7
    height_max: float = 1.3
    width_min: float = 0.8
    width_max: float = 1.2
    canopy_min: int = 3
    canopy_max: int = 7
    position_jitter: float = 0.3
    rotation_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate variation parameters."""
        if self.height_min <= 0 or self.height_max < self.height_min:
            raise ValueError(f"Invalid height range: [{self.height_min}, {self.height_max}]")
        if self.width_min <= 0 or self.width_max < self.width_min:
            raise ValueError(f"Invalid width range: [{self.width_min}, {self.width_max}]")
        if self.canopy_min < 1 or self.canopy_max < self.canopy_min:
            raise ValueError(f"Invalid canopy range: [{self.canopy_min}, {self.canopy_max}]")
        if not 0.0 <= self.position_jitter <= 0.5:
            raise ValueError(f"Position jitter must be in [0, 0.5], got {self.position_jitter}")


@dataclass(frozen=True, slots=True)
class ForestConfig:
    """Configuration for an infinite forest.

    Attributes:
        cell_size: Size of each cell in world units (x, y, z)
        base_tree: Base tree configuration to vary
        variation: Per-cell variation parameters
        density: Probability of tree in each cell (0-1)
    """
    cell_size: Tuple[float, float, float] = (8.0, 8.0, 8.0)
    base_tree: TreeConfig = field(default_factory=TreeConfig)
    variation: TreeVariation = field(default_factory=TreeVariation)
    density: float = 0.8

    def __post_init__(self) -> None:
        """Validate forest configuration."""
        if any(s <= 0 for s in self.cell_size):
            raise ValueError(f"Cell size must be positive, got {self.cell_size}")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError(f"Density must be in [0, 1], got {self.density}")


# =============================================================================
# ForestSDF - Infinite Forest via Domain Repetition (T-DEMO-4.6)
# =============================================================================

class ForestSDF:
    """Signed Distance Function for an infinite forest.

    Uses domain repetition to create an infinite grid of trees with
    per-cell pseudo-random variation for natural appearance.

    Each cell can have:
    - Different tree height/width
    - Different canopy configuration
    - Position jitter within the cell
    - Random rotation

    The hash function ensures deterministic results: the same cell ID
    always produces the same tree variation.

    Attributes:
        config: Forest configuration parameters
        position: Position offset (Vec3)
    """

    __slots__ = ("config", "position", "_mirror", "_tracker", "_node_id", "_dirty_fields", "_version")

    _instance_counter: ClassVar[int] = 0
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: Optional[ForestConfig] = None,
        position: Optional[Vec3] = None,
    ) -> None:
        """Initialize forest SDF.

        Args:
            config: Forest configuration. Uses defaults if None.
            position: Position offset. Uses origin if None.
        """
        with ForestSDF._counter_lock:
            ForestSDF._instance_counter += 1
            self._node_id = ForestSDF._instance_counter

        self.config = config or ForestConfig()
        self.position = position or Vec3()
        self._dirty_fields: set = {"config", "position"}
        self._version: int = 1
        self._mirror: Optional[Mirror] = None
        self._tracker: Optional[Tracker] = None

    @property
    def mirror(self) -> Mirror:
        """Get Mirror instance for introspection."""
        if self._mirror is None:
            self._mirror = _ForestMirror(self)
        return self._mirror

    @property
    def tracker(self) -> Tracker:
        """Get Tracker instance for dirty tracking."""
        if self._tracker is None:
            self._tracker = _ForestTracker(self)
        return self._tracker

    def evaluate(self, p: Tuple[float, float, float]) -> float:
        """Evaluate the signed distance at point p.

        Uses domain repetition with per-cell variation.

        Args:
            p: Query point in 3D space

        Returns:
            Signed distance to nearest tree surface
        """
        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return sdf_forest(local_p, self.config)

    def get_cell_id(self, p: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Get the cell ID for a point.

        Args:
            p: Query point in 3D space

        Returns:
            Integer cell coordinates (ix, iy, iz)
        """
        local_p = (
            p[0] - self.position.x,
            p[1] - self.position.y,
            p[2] - self.position.z,
        )
        return _get_cell_id(local_p, self.config.cell_size)

    def get_cell_tree_config(self, cell_id: Tuple[int, int, int]) -> Optional[TreeConfig]:
        """Get the tree configuration for a specific cell.

        Args:
            cell_id: Integer cell coordinates

        Returns:
            TreeConfig for the cell, or None if no tree in this cell
        """
        return _get_cell_tree_config(cell_id, self.config)

    def get_cell_tree_position(self, cell_id: Tuple[int, int, int]) -> Optional[Tuple[float, float, float]]:
        """Get the tree position within a specific cell.

        Args:
            cell_id: Integer cell coordinates

        Returns:
            (x, y, z) position, or None if no tree in this cell
        """
        return _get_cell_tree_position(cell_id, self.config)

    def has_tree_in_cell(self, cell_id: Tuple[int, int, int]) -> bool:
        """Check if a cell contains a tree.

        Args:
            cell_id: Integer cell coordinates

        Returns:
            True if the cell has a tree
        """
        density_hash = cell_hash_float(cell_id, channel=0)
        return density_hash < self.config.density

    def to_wgsl(self, name: str = "forest") -> str:
        """Generate WGSL code for this forest SDF.

        Args:
            name: Function name prefix

        Returns:
            WGSL source code for the forest SDF
        """
        return generate_forest_wgsl(self.config, name)

    def label(self) -> str:
        """Return a short descriptive label."""
        return f"ForestSDF(cell={self.config.cell_size}, density={self.config.density})"

    def __repr__(self) -> str:
        return f"<{self.label()} id={self._node_id}>"


# =============================================================================
# Forest SDF Implementation Functions
# =============================================================================

def _get_cell_id(p: Tuple[float, float, float], cell_size: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Get integer cell ID from world position."""
    ix = int(math.floor(p[0] / cell_size[0]))
    iy = int(math.floor(p[1] / cell_size[1]))
    iz = int(math.floor(p[2] / cell_size[2]))
    return (ix, iy, iz)


def _get_cell_center(cell_id: Tuple[int, int, int], cell_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Get world position of cell center."""
    return (
        (cell_id[0] + 0.5) * cell_size[0],
        (cell_id[1] + 0.5) * cell_size[1],
        (cell_id[2] + 0.5) * cell_size[2],
    )


def _get_cell_tree_position(cell_id: Tuple[int, int, int], config: ForestConfig) -> Optional[Tuple[float, float, float]]:
    """Get the tree position within a cell."""
    # Check density
    if cell_hash_float(cell_id, channel=0) >= config.density:
        return None

    center = _get_cell_center(cell_id, config.cell_size)

    # Apply position jitter
    jitter = config.variation.position_jitter
    jitter_x = (cell_hash_float(cell_id, channel=4) - 0.5) * 2.0 * jitter * config.cell_size[0]
    jitter_z = (cell_hash_float(cell_id, channel=5) - 0.5) * 2.0 * jitter * config.cell_size[2]

    # Tree base is at ground level (y=0 for the cell)
    return (
        center[0] + jitter_x,
        cell_id[1] * config.cell_size[1],  # Ground level for this cell layer
        center[2] + jitter_z,
    )


def _get_cell_tree_config(cell_id: Tuple[int, int, int], config: ForestConfig) -> Optional[TreeConfig]:
    """Generate tree configuration for a specific cell."""
    # Check density
    if cell_hash_float(cell_id, channel=0) >= config.density:
        return None

    base = config.base_tree
    var = config.variation

    # Height variation
    height_t = cell_hash_float(cell_id, channel=1)
    height_mult = var.height_min + height_t * (var.height_max - var.height_min)

    # Width variation
    width_t = cell_hash_float(cell_id, channel=2)
    width_mult = var.width_min + width_t * (var.width_max - var.width_min)

    # Canopy count variation
    canopy_t = cell_hash_float(cell_id, channel=3)
    canopy_count = var.canopy_min + int(canopy_t * (var.canopy_max - var.canopy_min + 1))
    canopy_count = min(canopy_count, var.canopy_max)

    return TreeConfig(
        trunk_height=base.trunk_height * height_mult,
        trunk_radius=base.trunk_radius * width_mult,
        trunk_type=base.trunk_type,
        trunk_taper=base.trunk_taper,
        canopy_type=base.canopy_type,
        canopy_spheres=canopy_count,
        canopy_radius=base.canopy_radius * width_mult,
        canopy_height_offset=base.canopy_height_offset * height_mult,
        canopy_spread=base.canopy_spread * width_mult,
        branches=base.branches,
        smooth_k=base.smooth_k,
    )


def _apply_rotation(p: Tuple[float, float, float], angle: float) -> Tuple[float, float, float]:
    """Apply Y-axis rotation to a point."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        p[0] * cos_a - p[2] * sin_a,
        p[1],
        p[0] * sin_a + p[2] * cos_a,
    )


def sdf_forest(p: Tuple[float, float, float], config: ForestConfig) -> float:
    """Evaluate the forest SDF at a point.

    Uses domain repetition to check nearby cells and find the minimum
    distance to any tree.

    Args:
        p: Query point in forest-local coordinates
        config: Forest configuration

    Returns:
        Signed distance to nearest tree
    """
    cell_id = _get_cell_id(p, config.cell_size)

    # Check 3x3x3 neighborhood for trees
    d_min = float("inf")

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                neighbor_id = (cell_id[0] + dx, cell_id[1] + dy, cell_id[2] + dz)

                tree_config = _get_cell_tree_config(neighbor_id, config)
                if tree_config is None:
                    continue

                tree_pos = _get_cell_tree_position(neighbor_id, config)
                if tree_pos is None:
                    continue

                # Transform point to tree-local coordinates
                local_p = (
                    p[0] - tree_pos[0],
                    p[1] - tree_pos[1],
                    p[2] - tree_pos[2],
                )

                # Apply rotation if enabled
                if config.variation.rotation_enabled:
                    rotation_angle = cell_hash_float(neighbor_id, channel=6) * 2.0 * math.pi
                    local_p = _apply_rotation(local_p, rotation_angle)

                # Evaluate tree SDF
                tree_d = sdf_tree(local_p, tree_config)
                d_min = min(d_min, tree_d)

    return d_min


# =============================================================================
# WGSL Code Generation
# =============================================================================

TREE_SDF_HEADER = """\
// SPDX-License-Identifier: MIT
//
// Auto-generated by vegetation_sdf.py (T-DEMO-4.5)
// Tree SDF combining trunk, canopy, and branches
//

"""

FOREST_SDF_HEADER = """\
// SPDX-License-Identifier: MIT
//
// Auto-generated by vegetation_sdf.py (T-DEMO-4.6)
// Infinite Forest via Domain Repetition
//

"""

# Common SDF primitives for trees
TREE_PRIMITIVES_WGSL = """\
// Sphere SDF
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Cylinder SDF (Y-axis aligned)
fn sdf_cylinder(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// Tapered cone SDF (frustum, Y-axis aligned)
fn sdf_cone_tapered(p: vec3<f32>, r_bottom: f32, r_top: f32, h: f32) -> f32 {
    let half_h = h * 0.5;
    let t = clamp((p.y + half_h) / h, 0.0, 1.0);
    let r_expected = mix(r_bottom, r_top, t);
    let r_actual = length(p.xz);
    let dr = r_actual - r_expected;
    let dy = abs(p.y) - half_h;
    if (dy > 0.0) {
        if (dr > 0.0) {
            return sqrt(dr * dr + dy * dy);
        }
        return dy;
    }
    if (dr > 0.0) {
        return dr;
    }
    return max(dr, dy);
}

// Capsule SDF
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Ellipsoid SDF (approximation)
fn sdf_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

// Smooth union
fn sdf_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}
"""

# Hash functions for forest
FOREST_HASH_WGSL = """\
// Cell hash function (xxHash-inspired)
fn cell_hash(cell: vec3<i32>) -> u32 {
    let PRIME1: u32 = 0x9E3779B9u;
    let PRIME2: u32 = 0x85EBCA6Bu;
    let PRIME3: u32 = 0xC2B2AE35u;

    var h = u32(cell.x) * PRIME1;
    h = (h ^ (u32(cell.y) * PRIME2)) * PRIME3;
    h = (h ^ (u32(cell.z) * PRIME1)) * PRIME2;
    h = (h ^ (h >> 16u)) * PRIME3;
    h = h ^ (h >> 13u);
    return h;
}

// Hash to float in [0, 1)
fn hash_to_float(h: u32) -> f32 {
    return f32(h & 0x7FFFFFFFu) / f32(0x7FFFFFFFu);
}

// Cell hash with channel for multiple independent values
fn cell_hash_float(cell: vec3<i32>, channel: i32) -> f32 {
    let modified = vec3<i32>(cell.x + channel * 1337, cell.y, cell.z);
    return hash_to_float(cell_hash(modified));
}

// Get cell ID from world position
fn get_cell_id(p: vec3<f32>, cell_size: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p / cell_size));
}

// Y-axis rotation
fn rotate_y(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
}
"""


def generate_tree_wgsl(config: TreeConfig, name: str = "tree") -> str:
    """Generate WGSL code for a tree SDF.

    Args:
        config: Tree configuration
        name: Function name prefix

    Returns:
        WGSL source code
    """
    lines = [TREE_SDF_HEADER, TREE_PRIMITIVES_WGSL, ""]

    # Generate tree constants
    lines.append(f"// Tree parameters for {name}")
    lines.append(f"const {name.upper()}_TRUNK_HEIGHT: f32 = {config.trunk_height};")
    lines.append(f"const {name.upper()}_TRUNK_RADIUS: f32 = {config.trunk_radius};")
    lines.append(f"const {name.upper()}_TRUNK_TAPER: f32 = {config.trunk_taper};")
    lines.append(f"const {name.upper()}_CANOPY_RADIUS: f32 = {config.canopy_radius};")
    lines.append(f"const {name.upper()}_CANOPY_SPHERES: u32 = {config.canopy_spheres}u;")
    lines.append(f"const {name.upper()}_CANOPY_HEIGHT_OFFSET: f32 = {config.canopy_height_offset};")
    lines.append(f"const {name.upper()}_CANOPY_SPREAD: f32 = {config.canopy_spread};")
    lines.append(f"const {name.upper()}_SMOOTH_K: f32 = {config.smooth_k};")
    lines.append("")

    # Trunk function
    if config.trunk_type == TrunkType.CYLINDER:
        lines.append(f"fn sdf_{name}_trunk(p: vec3<f32>) -> f32 {{")
        lines.append(f"    let half_h = {name.upper()}_TRUNK_HEIGHT * 0.5;")
        lines.append(f"    let shifted = vec3<f32>(p.x, p.y - half_h, p.z);")
        lines.append(f"    return sdf_cylinder(shifted, {name.upper()}_TRUNK_RADIUS, half_h);")
        lines.append("}")
    else:
        lines.append(f"fn sdf_{name}_trunk(p: vec3<f32>) -> f32 {{")
        lines.append(f"    let half_h = {name.upper()}_TRUNK_HEIGHT * 0.5;")
        lines.append(f"    let shifted = vec3<f32>(p.x, p.y - half_h, p.z);")
        lines.append(f"    let top_r = {name.upper()}_TRUNK_RADIUS * {name.upper()}_TRUNK_TAPER;")
        lines.append(f"    return sdf_cone_tapered(shifted, {name.upper()}_TRUNK_RADIUS, top_r, {name.upper()}_TRUNK_HEIGHT);")
        lines.append("}")
    lines.append("")

    # Canopy function
    lines.append(f"fn sdf_{name}_canopy(p: vec3<f32>) -> f32 {{")
    lines.append(f"    let center_y = {name.upper()}_TRUNK_HEIGHT + {name.upper()}_CANOPY_HEIGHT_OFFSET;")
    lines.append(f"    var d: f32 = 1e10;")
    lines.append("")
    lines.append(f"    // Central sphere")
    lines.append(f"    d = sdf_sphere(p - vec3<f32>(0.0, center_y, 0.0), {name.upper()}_CANOPY_RADIUS);")
    lines.append("")
    lines.append(f"    // Ring of spheres")
    lines.append(f"    let n = {name.upper()}_CANOPY_SPHERES - 1u;")
    lines.append(f"    for (var i: u32 = 0u; i < n; i++) {{")
    lines.append(f"        let angle = f32(i) * 6.28318530718 / f32(n);")
    lines.append(f"        let x = {name.upper()}_CANOPY_SPREAD * cos(angle);")
    lines.append(f"        let z = {name.upper()}_CANOPY_SPREAD * sin(angle);")
    lines.append(f"        let y_offset = 0.1 * sin(angle * 2.0);")
    if config.canopy_type == CanopyType.SPHERES:
        lines.append(f"        let sphere_d = sdf_sphere(p - vec3<f32>(x, center_y + y_offset, z), {name.upper()}_CANOPY_RADIUS);")
    else:
        lines.append(f"        let radii = vec3<f32>({name.upper()}_CANOPY_RADIUS, {name.upper()}_CANOPY_RADIUS * 0.7, {name.upper()}_CANOPY_RADIUS);")
        lines.append(f"        let sphere_d = sdf_ellipsoid(p - vec3<f32>(x, center_y + y_offset, z), radii);")
    lines.append(f"        d = sdf_smooth_union(d, sphere_d, {name.upper()}_SMOOTH_K);")
    lines.append(f"    }}")
    lines.append(f"    return d;")
    lines.append("}")
    lines.append("")

    # Branches function (if applicable)
    if config.branches.count > 0:
        lines.append(f"fn sdf_{name}_branches(p: vec3<f32>) -> f32 {{")
        lines.append(f"    var d: f32 = 1e10;")
        lines.append(f"    let attachment_y = {name.upper()}_TRUNK_HEIGHT * {config.branches.attachment_height};")
        for i, (start, end) in enumerate(_get_branch_endpoints(config)):
            lines.append(f"    // Branch {i}")
            lines.append(f"    d = min(d, sdf_capsule(p, vec3<f32>({start[0]}, {start[1]}, {start[2]}), vec3<f32>({end[0]}, {end[1]}, {end[2]}), {config.branches.radius}));")
        lines.append(f"    return d;")
        lines.append("}")
        lines.append("")

    # Main tree function
    lines.append(f"fn sdf_{name}(p: vec3<f32>) -> f32 {{")
    lines.append(f"    let trunk_d = sdf_{name}_trunk(p);")
    lines.append(f"    let canopy_d = sdf_{name}_canopy(p);")
    lines.append(f"    var d = sdf_smooth_union(trunk_d, canopy_d, {name.upper()}_SMOOTH_K);")
    if config.branches.count > 0:
        lines.append(f"    let branches_d = sdf_{name}_branches(p);")
        lines.append(f"    d = sdf_smooth_union(d, branches_d, {name.upper()}_SMOOTH_K * 0.5);")
    lines.append(f"    return d;")
    lines.append("}")

    return "\n".join(lines)


def generate_forest_wgsl(config: ForestConfig, name: str = "forest") -> str:
    """Generate WGSL code for an infinite forest SDF.

    Args:
        config: Forest configuration
        name: Function name prefix

    Returns:
        WGSL source code
    """
    lines = [FOREST_SDF_HEADER, TREE_PRIMITIVES_WGSL, FOREST_HASH_WGSL, ""]

    # Forest constants
    lines.append(f"// Forest parameters for {name}")
    lines.append(f"const {name.upper()}_CELL_SIZE: vec3<f32> = vec3<f32>({config.cell_size[0]}, {config.cell_size[1]}, {config.cell_size[2]});")
    lines.append(f"const {name.upper()}_DENSITY: f32 = {config.density};")
    lines.append("")

    # Variation constants
    var = config.variation
    lines.append(f"// Variation parameters")
    lines.append(f"const {name.upper()}_HEIGHT_MIN: f32 = {var.height_min};")
    lines.append(f"const {name.upper()}_HEIGHT_MAX: f32 = {var.height_max};")
    lines.append(f"const {name.upper()}_WIDTH_MIN: f32 = {var.width_min};")
    lines.append(f"const {name.upper()}_WIDTH_MAX: f32 = {var.width_max};")
    lines.append(f"const {name.upper()}_CANOPY_MIN: u32 = {var.canopy_min}u;")
    lines.append(f"const {name.upper()}_CANOPY_MAX: u32 = {var.canopy_max}u;")
    lines.append(f"const {name.upper()}_JITTER: f32 = {var.position_jitter};")
    lines.append(f"const {name.upper()}_ROTATION_ENABLED: bool = {'true' if var.rotation_enabled else 'false'};")
    lines.append("")

    # Base tree constants
    base = config.base_tree
    lines.append(f"// Base tree parameters")
    lines.append(f"const BASE_TRUNK_HEIGHT: f32 = {base.trunk_height};")
    lines.append(f"const BASE_TRUNK_RADIUS: f32 = {base.trunk_radius};")
    lines.append(f"const BASE_TRUNK_TAPER: f32 = {base.trunk_taper};")
    lines.append(f"const BASE_CANOPY_RADIUS: f32 = {base.canopy_radius};")
    lines.append(f"const BASE_CANOPY_HEIGHT_OFFSET: f32 = {base.canopy_height_offset};")
    lines.append(f"const BASE_CANOPY_SPREAD: f32 = {base.canopy_spread};")
    lines.append(f"const BASE_SMOOTH_K: f32 = {base.smooth_k};")
    lines.append("")

    # Per-cell tree evaluation
    lines.append(f"// Evaluate tree in a specific cell")
    lines.append(f"fn sdf_{name}_cell_tree(p: vec3<f32>, cell: vec3<i32>) -> f32 {{")
    lines.append(f"    // Check density")
    lines.append(f"    if (cell_hash_float(cell, 0) >= {name.upper()}_DENSITY) {{")
    lines.append(f"        return 1e10;")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    // Get variation multipliers")
    lines.append(f"    let height_t = cell_hash_float(cell, 1);")
    lines.append(f"    let width_t = cell_hash_float(cell, 2);")
    lines.append(f"    let height_mult = mix({name.upper()}_HEIGHT_MIN, {name.upper()}_HEIGHT_MAX, height_t);")
    lines.append(f"    let width_mult = mix({name.upper()}_WIDTH_MIN, {name.upper()}_WIDTH_MAX, width_t);")
    lines.append(f"")
    lines.append(f"    // Tree parameters for this cell")
    lines.append(f"    let trunk_height = BASE_TRUNK_HEIGHT * height_mult;")
    lines.append(f"    let trunk_radius = BASE_TRUNK_RADIUS * width_mult;")
    lines.append(f"    let canopy_radius = BASE_CANOPY_RADIUS * width_mult;")
    lines.append(f"    let canopy_spread = BASE_CANOPY_SPREAD * width_mult;")
    lines.append(f"    let canopy_height_offset = BASE_CANOPY_HEIGHT_OFFSET * height_mult;")
    lines.append(f"")
    lines.append(f"    // Calculate tree position with jitter")
    lines.append(f"    let cell_center = (vec3<f32>(cell) + 0.5) * {name.upper()}_CELL_SIZE;")
    lines.append(f"    let jitter_x = (cell_hash_float(cell, 4) - 0.5) * 2.0 * {name.upper()}_JITTER * {name.upper()}_CELL_SIZE.x;")
    lines.append(f"    let jitter_z = (cell_hash_float(cell, 5) - 0.5) * 2.0 * {name.upper()}_JITTER * {name.upper()}_CELL_SIZE.z;")
    lines.append(f"    let tree_pos = vec3<f32>(cell_center.x + jitter_x, f32(cell.y) * {name.upper()}_CELL_SIZE.y, cell_center.z + jitter_z);")
    lines.append(f"")
    lines.append(f"    // Transform to tree-local coordinates")
    lines.append(f"    var local_p = p - tree_pos;")
    lines.append(f"    if ({name.upper()}_ROTATION_ENABLED) {{")
    lines.append(f"        let rot_angle = cell_hash_float(cell, 6) * 6.28318530718;")
    lines.append(f"        local_p = rotate_y(local_p, rot_angle);")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    // Evaluate trunk")
    lines.append(f"    let half_h = trunk_height * 0.5;")
    lines.append(f"    let shifted = vec3<f32>(local_p.x, local_p.y - half_h, local_p.z);")

    if base.trunk_type == TrunkType.CYLINDER:
        lines.append(f"    let trunk_d = sdf_cylinder(shifted, trunk_radius, half_h);")
    else:
        lines.append(f"    let top_r = trunk_radius * BASE_TRUNK_TAPER;")
        lines.append(f"    let trunk_d = sdf_cone_tapered(shifted, trunk_radius, top_r, trunk_height);")

    lines.append(f"")
    lines.append(f"    // Evaluate canopy")
    lines.append(f"    let center_y = trunk_height + canopy_height_offset;")
    lines.append(f"    let canopy_t = cell_hash_float(cell, 3);")
    lines.append(f"    let n_spheres = {name.upper()}_CANOPY_MIN + u32(canopy_t * f32({name.upper()}_CANOPY_MAX - {name.upper()}_CANOPY_MIN + 1u));")
    lines.append(f"")
    lines.append(f"    var canopy_d = sdf_sphere(local_p - vec3<f32>(0.0, center_y, 0.0), canopy_radius);")
    lines.append(f"    let n_ring = n_spheres - 1u;")
    lines.append(f"    for (var i: u32 = 0u; i < n_ring; i++) {{")
    lines.append(f"        let angle = f32(i) * 6.28318530718 / f32(n_ring);")
    lines.append(f"        let x = canopy_spread * cos(angle);")
    lines.append(f"        let z = canopy_spread * sin(angle);")
    lines.append(f"        let y_off = 0.1 * sin(angle * 2.0);")

    if base.canopy_type == CanopyType.SPHERES:
        lines.append(f"        let sphere_d = sdf_sphere(local_p - vec3<f32>(x, center_y + y_off, z), canopy_radius);")
    else:
        lines.append(f"        let radii = vec3<f32>(canopy_radius, canopy_radius * 0.7, canopy_radius);")
        lines.append(f"        let sphere_d = sdf_ellipsoid(local_p - vec3<f32>(x, center_y + y_off, z), radii);")

    lines.append(f"        canopy_d = sdf_smooth_union(canopy_d, sphere_d, BASE_SMOOTH_K);")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    // Combine trunk and canopy")
    lines.append(f"    return sdf_smooth_union(trunk_d, canopy_d, BASE_SMOOTH_K);")
    lines.append(f"}}")
    lines.append("")

    # Main forest function
    lines.append(f"// Main forest SDF")
    lines.append(f"fn sdf_{name}(p: vec3<f32>) -> f32 {{")
    lines.append(f"    let cell = get_cell_id(p, {name.upper()}_CELL_SIZE);")
    lines.append(f"    var d: f32 = 1e10;")
    lines.append(f"")
    lines.append(f"    // Check 3x3x3 neighborhood")
    lines.append(f"    for (var dx: i32 = -1; dx <= 1; dx++) {{")
    lines.append(f"        for (var dy: i32 = -1; dy <= 1; dy++) {{")
    lines.append(f"            for (var dz: i32 = -1; dz <= 1; dz++) {{")
    lines.append(f"                let neighbor = cell + vec3<i32>(dx, dy, dz);")
    lines.append(f"                let tree_d = sdf_{name}_cell_tree(p, neighbor);")
    lines.append(f"                d = min(d, tree_d);")
    lines.append(f"            }}")
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append(f"")
    lines.append(f"    return d;")
    lines.append(f"}}")

    return "\n".join(lines)


# =============================================================================
# Mirror/Tracker Helpers (Trinity Pattern)
# =============================================================================

class _TreeMirror:
    """Mirror implementation for TreeSDF."""

    __slots__ = ("_node",)

    def __init__(self, node: TreeSDF) -> None:
        self._node = node

    @property
    def node_type(self) -> str:
        return "TreeSDF"

    @property
    def node_id(self) -> int:
        return self._node._node_id

    @property
    def fields(self) -> Dict[str, Any]:
        return {
            "config": self._node.config,
            "position": self._node.position,
        }

    @property
    def is_dirty(self) -> bool:
        return bool(self._node._dirty_fields)

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        return frozenset(self._node._dirty_fields)

    def __repr__(self) -> str:
        return f"<Mirror for TreeSDF#{self.node_id}>"


class _TreeTracker:
    """Tracker implementation for TreeSDF."""

    __slots__ = ("_node",)

    def __init__(self, node: TreeSDF) -> None:
        self._node = node

    @property
    def is_dirty(self) -> bool:
        return bool(self._node._dirty_fields)

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        return frozenset(self._node._dirty_fields)

    @property
    def version(self) -> int:
        return self._node._version

    def mark_dirty(self, field_name: str) -> None:
        self._node._dirty_fields.add(field_name)
        self._node._version += 1

    def clear(self) -> None:
        self._node._dirty_fields.clear()

    def __repr__(self) -> str:
        return f"<Tracker dirty={len(self._node._dirty_fields)} version={self.version}>"


class _ForestMirror:
    """Mirror implementation for ForestSDF."""

    __slots__ = ("_node",)

    def __init__(self, node: ForestSDF) -> None:
        self._node = node

    @property
    def node_type(self) -> str:
        return "ForestSDF"

    @property
    def node_id(self) -> int:
        return self._node._node_id

    @property
    def fields(self) -> Dict[str, Any]:
        return {
            "config": self._node.config,
            "position": self._node.position,
        }

    @property
    def is_dirty(self) -> bool:
        return bool(self._node._dirty_fields)

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        return frozenset(self._node._dirty_fields)

    def __repr__(self) -> str:
        return f"<Mirror for ForestSDF#{self.node_id}>"


class _ForestTracker:
    """Tracker implementation for ForestSDF."""

    __slots__ = ("_node",)

    def __init__(self, node: ForestSDF) -> None:
        self._node = node

    @property
    def is_dirty(self) -> bool:
        return bool(self._node._dirty_fields)

    @property
    def dirty_fields(self) -> FrozenSet[str]:
        return frozenset(self._node._dirty_fields)

    @property
    def version(self) -> int:
        return self._node._version

    def mark_dirty(self, field_name: str) -> None:
        self._node._dirty_fields.add(field_name)
        self._node._version += 1

    def clear(self) -> None:
        self._node._dirty_fields.clear()

    def __repr__(self) -> str:
        return f"<Tracker dirty={len(self._node._dirty_fields)} version={self.version}>"
