"""Adaptive DDGI Probe Placement Prototype (T-GIR-P11.1).

This module provides a CPU-side prototype for variance-based adaptive
probe placement using an octree data structure.

Features:
    - AdaptiveProbeGrid: Octree-based adaptive probe volume
    - Variance computation for subdivision decisions
    - Temporal hysteresis for stable transitions
    - GPU-compatible linearization

This is a research prototype for validation before GPU implementation.

References:
    - docs/RUST_DOCS/GAPSET_6_GI_REFLECTIONS/ADAPTIVE_DDGI_RESEARCH.md
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, List, Optional, Tuple

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AdaptiveDDGIConfig:
    """Configuration for adaptive DDGI probe placement.

    Attributes:
        base_dimensions: Initial coarse grid dimensions (x, y, z)
        base_spacing: Spacing between probes at base level (meters)
        max_depth: Maximum subdivision depth (0 = no subdivision)
        base_variance_threshold: Variance threshold at depth 0
        depth_falloff: Threshold multiplier per depth level
        subdivide_hysteresis_frames: Frames above threshold before subdivide
        merge_hysteresis_frames: Frames below threshold before merge
        probe_fade_frames: Frames to fade in new probes
        max_probes: Hard limit on total probe count
    """

    base_dimensions: Tuple[int, int, int] = (16, 16, 4)
    base_spacing: float = 4.0
    max_depth: int = 4
    base_variance_threshold: float = 0.05
    depth_falloff: float = 0.7
    subdivide_hysteresis_frames: int = 8
    merge_hysteresis_frames: int = 30
    probe_fade_frames: int = 16
    max_probes: int = 32768

    def threshold_at_depth(self, depth: int) -> float:
        """Compute variance threshold at given depth."""
        return self.base_variance_threshold * (self.depth_falloff ** depth)

    def merge_threshold_at_depth(self, depth: int) -> float:
        """Compute merge threshold (half of parent's subdivision threshold)."""
        if depth == 0:
            return 0.0
        return self.threshold_at_depth(depth - 1) * 0.5

    @staticmethod
    def low() -> AdaptiveDDGIConfig:
        """Low quality preset."""
        return AdaptiveDDGIConfig(
            base_dimensions=(8, 8, 2),
            base_spacing=6.0,
            max_depth=2,
            max_probes=4096,
        )

    @staticmethod
    def medium() -> AdaptiveDDGIConfig:
        """Medium quality preset."""
        return AdaptiveDDGIConfig(
            base_dimensions=(12, 12, 3),
            base_spacing=5.0,
            max_depth=3,
            max_probes=16384,
        )

    @staticmethod
    def high() -> AdaptiveDDGIConfig:
        """High quality preset."""
        return AdaptiveDDGIConfig(
            base_dimensions=(16, 16, 4),
            base_spacing=4.0,
            max_depth=4,
            max_probes=32768,
        )


# ============================================================================
# Probe State
# ============================================================================


class AdaptiveProbeState(Enum):
    """State of an adaptive probe for temporal blending."""

    INACTIVE = auto()     # Not active
    FADING_IN = auto()    # Newly created, blending in
    ACTIVE = auto()       # Fully active
    FADING_OUT = auto()   # Being removed, blending out


@dataclass
class AdaptiveProbe:
    """Adaptive probe with temporal state.

    Attributes:
        position: World-space probe position
        irradiance: Current irradiance (RGB, used for variance)
        state: Temporal state for blending
        blend_weight: Current blend weight [0, 1]
        _fade_progress: Internal fade progress counter
    """

    position: Vec3 = field(default_factory=Vec3.zero)
    irradiance: Vec3 = field(default_factory=Vec3.zero)
    state: AdaptiveProbeState = AdaptiveProbeState.ACTIVE
    blend_weight: float = 1.0
    _fade_progress: int = 0

    def update_fade(self, fade_frames: int) -> None:
        """Update fade state for one frame."""
        if self.state == AdaptiveProbeState.FADING_IN:
            self._fade_progress += 1
            self.blend_weight = min(1.0, self._fade_progress / fade_frames)
            if self.blend_weight >= 1.0:
                self.state = AdaptiveProbeState.ACTIVE

        elif self.state == AdaptiveProbeState.FADING_OUT:
            self._fade_progress -= 1
            self.blend_weight = max(0.0, self._fade_progress / fade_frames)
            if self.blend_weight <= 0.0:
                self.state = AdaptiveProbeState.INACTIVE

    def start_fade_in(self, fade_frames: int) -> None:
        """Start fading in this probe."""
        self.state = AdaptiveProbeState.FADING_IN
        self._fade_progress = 0
        self.blend_weight = 0.0

    def start_fade_out(self) -> None:
        """Start fading out this probe."""
        self.state = AdaptiveProbeState.FADING_OUT
        # Keep current fade progress for smooth transition

    def luminance(self) -> float:
        """Compute luminance from irradiance."""
        return (
            0.2126 * self.irradiance.x +
            0.7152 * self.irradiance.y +
            0.0722 * self.irradiance.z
        )


# ============================================================================
# Octree Node
# ============================================================================


@dataclass
class AdaptiveProbeNode:
    """Octree node for adaptive probe placement.

    Each node represents a cubic region of space. Leaf nodes contain
    8 corner probes. Internal nodes have up to 8 children.

    Attributes:
        bounds: Axis-aligned bounding box of this node
        depth: Depth in the octree (0 = root)
        children: List of 8 child nodes (None if leaf)
        probes: 8 corner probes (only for leaf nodes)
        variance: Current variance of this cell
        frames_above_threshold: Consecutive frames above subdivide threshold
        frames_below_threshold: Consecutive frames below merge threshold
    """

    bounds: AABB
    depth: int = 0
    children: Optional[List[Optional[AdaptiveProbeNode]]] = None
    probes: Optional[List[AdaptiveProbe]] = None
    variance: float = 0.0
    frames_above_threshold: int = 0
    frames_below_threshold: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.children is None

    def child_bounds(self, child_index: int) -> AABB:
        """Compute bounds for a child octant.

        Args:
            child_index: Octant index (0-7), bits encode (z, y, x)

        Returns:
            AABB for the child octant
        """
        # Parent bounds
        p_min = self.bounds.min
        p_max = self.bounds.max
        center = self.bounds.center

        # Child bounds based on octant
        # Bit 0 (x): 0 = lower half, 1 = upper half
        # Bit 1 (y): 0 = lower half, 1 = upper half
        # Bit 2 (z): 0 = lower half, 1 = upper half

        child_min_x = center.x if (child_index & 1) else p_min.x
        child_max_x = p_max.x if (child_index & 1) else center.x

        child_min_y = center.y if (child_index & 2) else p_min.y
        child_max_y = p_max.y if (child_index & 2) else center.y

        child_min_z = center.z if (child_index & 4) else p_min.z
        child_max_z = p_max.z if (child_index & 4) else center.z

        return AABB(
            Vec3(child_min_x, child_min_y, child_min_z),
            Vec3(child_max_x, child_max_y, child_max_z),
        )

    def create_corner_probes(self) -> List[AdaptiveProbe]:
        """Create 8 probes at the corners of this node's bounds."""
        probes = []
        min_p = self.bounds.min
        max_p = self.bounds.max

        # 8 corners: iterate over binary combinations
        for i in range(8):
            x = max_p.x if (i & 1) else min_p.x
            y = max_p.y if (i & 2) else min_p.y
            z = max_p.z if (i & 4) else min_p.z

            probe = AdaptiveProbe(position=Vec3(x, y, z))
            probes.append(probe)

        return probes

    def compute_variance(self) -> float:
        """Compute irradiance variance across corner probes.

        Returns:
            Variance of probe luminances
        """
        if self.probes is None or len(self.probes) == 0:
            return 0.0

        # Compute mean luminance
        luminances = [p.luminance() for p in self.probes]
        mean = sum(luminances) / len(luminances)

        # Compute variance
        variance = sum((l - mean) ** 2 for l in luminances) / len(luminances)

        return variance

    def iter_probes(self) -> Iterator[AdaptiveProbe]:
        """Iterate over all probes in this node and descendants."""
        if self.is_leaf():
            if self.probes:
                yield from self.probes
        else:
            if self.children:
                for child in self.children:
                    if child is not None:
                        yield from child.iter_probes()

    def total_probe_count(self) -> int:
        """Count total probes in this subtree."""
        return sum(1 for _ in self.iter_probes())


# ============================================================================
# Adaptive Probe Grid
# ============================================================================


class AdaptiveProbeGrid:
    """Octree-based adaptive probe grid.

    This class manages the adaptive probe placement system including:
    - Octree construction and traversal
    - Variance-based subdivision decisions
    - Temporal stability through hysteresis
    - GPU-compatible data export

    Attributes:
        config: Adaptive configuration
        root: Root node of the octree
        origin: World-space origin (minimum corner)
        frame_index: Current frame for temporal tracking
    """

    def __init__(
        self,
        config: AdaptiveDDGIConfig,
        bounds: AABB,
    ) -> None:
        """Initialize the adaptive grid.

        Args:
            config: Configuration for adaptive behavior
            bounds: World-space bounds for the entire grid
        """
        self.config = config
        self.origin = bounds.min
        self.frame_index = 0

        # Create root node
        self.root = AdaptiveProbeNode(bounds=bounds, depth=0)
        self.root.probes = self.root.create_corner_probes()

        # Statistics
        self._last_subdivision_count = 0
        self._last_merge_count = 0

    def total_probes(self) -> int:
        """Count total probes in the grid."""
        return self.root.total_probe_count()

    def iter_probes(self) -> Iterator[AdaptiveProbe]:
        """Iterate over all probes in the grid."""
        return self.root.iter_probes()

    def iter_leaf_nodes(self) -> Iterator[AdaptiveProbeNode]:
        """Iterate over all leaf nodes in the octree."""
        stack = [self.root]

        while stack:
            node = stack.pop()
            if node.is_leaf():
                yield node
            elif node.children:
                for child in node.children:
                    if child is not None:
                        stack.append(child)

    def update(
        self,
        irradiance_sampler: Callable[[Vec3], Vec3],
    ) -> None:
        """Update the grid for one frame.

        Args:
            irradiance_sampler: Function to sample irradiance at a position
        """
        self.frame_index += 1
        self._last_subdivision_count = 0
        self._last_merge_count = 0

        # Update probe irradiance
        for probe in self.iter_probes():
            probe.irradiance = irradiance_sampler(probe.position)
            probe.update_fade(self.config.probe_fade_frames)

        # Update variance and subdivision state
        self._update_node_recursive(self.root, irradiance_sampler)

    def _update_node_recursive(
        self,
        node: AdaptiveProbeNode,
        irradiance_sampler: Callable[[Vec3], Vec3],
    ) -> float:
        """Recursively update a node and its children.

        Args:
            node: Node to update
            irradiance_sampler: Irradiance sampling function

        Returns:
            Maximum variance in this subtree
        """
        if node.is_leaf():
            # Compute variance for leaf
            node.variance = node.compute_variance()

            # Check subdivision
            threshold = self.config.threshold_at_depth(node.depth)

            if node.variance > threshold:
                node.frames_above_threshold += 1
                node.frames_below_threshold = 0

                if self._should_subdivide(node):
                    self._subdivide_node(node, irradiance_sampler)
                    self._last_subdivision_count += 1
            else:
                node.frames_above_threshold = 0
                merge_threshold = self.config.merge_threshold_at_depth(node.depth)

                if node.variance < merge_threshold:
                    node.frames_below_threshold += 1
                else:
                    node.frames_below_threshold = 0

            return node.variance

        else:
            # Recurse into children and compute max variance
            max_child_variance = 0.0
            all_children_below_merge = True

            merge_threshold = self.config.merge_threshold_at_depth(node.depth + 1)

            for i, child in enumerate(node.children):
                if child is not None:
                    child_var = self._update_node_recursive(child, irradiance_sampler)
                    max_child_variance = max(max_child_variance, child_var)

                    if child_var >= merge_threshold or not child.is_leaf():
                        all_children_below_merge = False

            # Check merge
            if all_children_below_merge:
                node.frames_below_threshold += 1

                if self._should_merge(node):
                    self._merge_node(node)
                    self._last_merge_count += 1
            else:
                node.frames_below_threshold = 0

            node.variance = max_child_variance
            return max_child_variance

    def _should_subdivide(self, node: AdaptiveProbeNode) -> bool:
        """Check if a node should subdivide."""
        if node.depth >= self.config.max_depth:
            return False

        if self.total_probes() >= self.config.max_probes:
            return False

        if node.frames_above_threshold < self.config.subdivide_hysteresis_frames:
            return False

        return True

    def _should_merge(self, node: AdaptiveProbeNode) -> bool:
        """Check if a node's children should merge."""
        if node.depth == 0:
            return False  # Can't merge root

        if node.is_leaf():
            return False  # Already a leaf

        if node.children is None:
            return False

        if node.frames_below_threshold < self.config.merge_hysteresis_frames:
            return False

        # All children must be leaves
        for child in node.children:
            if child is not None and not child.is_leaf():
                return False

        return True

    def _subdivide_node(
        self,
        node: AdaptiveProbeNode,
        irradiance_sampler: Callable[[Vec3], Vec3],
    ) -> None:
        """Subdivide a leaf node into 8 children."""
        if not node.is_leaf():
            return

        # Create children
        node.children = [None] * 8

        for i in range(8):
            child_bounds = node.child_bounds(i)
            child = AdaptiveProbeNode(
                bounds=child_bounds,
                depth=node.depth + 1,
            )

            # Create probes and initialize from parent
            child.probes = child.create_corner_probes()

            for probe in child.probes:
                # Initialize irradiance from sampler
                probe.irradiance = irradiance_sampler(probe.position)
                probe.start_fade_in(self.config.probe_fade_frames)

            node.children[i] = child

        # Start fading out parent probes
        if node.probes:
            for probe in node.probes:
                probe.start_fade_out()

            # Keep parent probes during fade (will be removed after fade completes)
            # For simplicity in prototype, clear immediately
            node.probes = None

        node.frames_above_threshold = 0

    def _merge_node(self, node: AdaptiveProbeNode) -> None:
        """Merge children back into parent."""
        if node.children is None:
            return

        # Collect child probe positions for averaging
        all_child_probes = []
        for child in node.children:
            if child is not None and child.probes:
                all_child_probes.extend(child.probes)

        # Create new probes at node corners
        node.probes = node.create_corner_probes()

        # Initialize new probes with averaged irradiance from children
        if all_child_probes:
            avg_irradiance = Vec3.zero()
            for probe in all_child_probes:
                avg_irradiance = avg_irradiance + probe.irradiance
            avg_irradiance = avg_irradiance * (1.0 / len(all_child_probes))

            for probe in node.probes:
                probe.irradiance = avg_irradiance
                probe.start_fade_in(self.config.probe_fade_frames)

        # Remove children
        node.children = None
        node.frames_below_threshold = 0

    def sample_irradiance(
        self,
        world_pos: Vec3,
        normal: Vec3,
    ) -> Vec3:
        """Sample irradiance at a world position.

        Uses octree traversal to find the deepest node containing the
        position, then performs trilinear interpolation.

        Args:
            world_pos: World-space position
            normal: Surface normal for weighting

        Returns:
            Interpolated irradiance
        """
        # Find deepest node containing position
        node = self._find_deepest_node(world_pos)

        if node is None or node.probes is None:
            return Vec3.zero()

        return self._sample_node_probes(world_pos, normal, node)

    def _find_deepest_node(self, world_pos: Vec3) -> Optional[AdaptiveProbeNode]:
        """Find the deepest octree node containing a position."""
        node = self.root

        if not node.bounds.contains(world_pos):
            return None

        while not node.is_leaf() and node.children:
            # Find child octant
            center = node.bounds.center
            octant = 0

            if world_pos.x > center.x:
                octant |= 1
            if world_pos.y > center.y:
                octant |= 2
            if world_pos.z > center.z:
                octant |= 4

            child = node.children[octant]

            if child is None:
                # No child for this octant, use parent
                break

            node = child

        return node

    def _sample_node_probes(
        self,
        world_pos: Vec3,
        normal: Vec3,
        node: AdaptiveProbeNode,
    ) -> Vec3:
        """Sample probes within a leaf node using trilinear interpolation."""
        if node.probes is None or len(node.probes) != 8:
            return Vec3.zero()

        # Compute local coordinates [0, 1]
        local = world_pos - node.bounds.min
        size = node.bounds.max - node.bounds.min

        fx = max(0.0, min(1.0, local.x / size.x if size.x > 0 else 0))
        fy = max(0.0, min(1.0, local.y / size.y if size.y > 0 else 0))
        fz = max(0.0, min(1.0, local.z / size.z if size.z > 0 else 0))

        # Trilinear interpolation with normal-based weighting
        total = Vec3.zero()
        total_weight = 0.0

        for i, probe in enumerate(node.probes):
            # Trilinear weight
            cx = fx if (i & 1) else (1.0 - fx)
            cy = fy if (i & 2) else (1.0 - fy)
            cz = fz if (i & 4) else (1.0 - fz)
            weight = cx * cy * cz * probe.blend_weight

            # Normal weight (backface rejection)
            to_probe = (probe.position - world_pos).normalized()
            normal_weight = max(0.0001, normal.dot(-to_probe))
            weight *= normal_weight

            if weight > 0:
                total = total + probe.irradiance * weight
                total_weight += weight

        if total_weight > 0:
            return total * (1.0 / total_weight)

        return Vec3.zero()

    def get_bounds(self) -> AABB:
        """Get the overall bounds of the grid."""
        return self.root.bounds

    def get_statistics(self) -> dict:
        """Get grid statistics for debugging."""
        total_probes = self.total_probes()
        leaf_count = sum(1 for _ in self.iter_leaf_nodes())
        max_depth = 0

        for node in self.iter_leaf_nodes():
            max_depth = max(max_depth, node.depth)

        return {
            "total_probes": total_probes,
            "leaf_nodes": leaf_count,
            "max_depth_used": max_depth,
            "max_depth_allowed": self.config.max_depth,
            "frame_index": self.frame_index,
            "last_subdivisions": self._last_subdivision_count,
            "last_merges": self._last_merge_count,
        }

    # ========================================================================
    # GPU Export
    # ========================================================================

    def build_linearized_octree(self) -> Tuple[bytes, bytes, bytes]:
        """Build GPU-uploadable linearized octree data.

        Returns:
            Tuple of (nodes_buffer, probes_buffer, header_buffer)
        """
        nodes_data = []
        probes_data = []
        probe_index = 0

        # BFS traversal for breadth-first layout
        queue = [self.root]
        node_index_map = {}  # node -> index

        while queue:
            node = queue.pop(0)
            node_idx = len(nodes_data)
            node_index_map[id(node)] = node_idx

            # Pack node
            child_mask = 0
            first_child_or_probe = 0

            if node.is_leaf():
                # Leaf: record probe base index
                first_child_or_probe = probe_index

                if node.probes:
                    for probe in node.probes:
                        probes_data.append(self._pack_probe(probe))
                        probe_index += 1
            else:
                # Internal: will fill first_child later
                if node.children:
                    for i, child in enumerate(node.children):
                        if child is not None:
                            child_mask |= (1 << i)
                            queue.append(child)

                first_child_or_probe = len(nodes_data) + len(queue) - sum(
                    1 for c in (node.children or []) if c is not None
                ) + 1

            nodes_data.append(self._pack_node(
                node, child_mask, first_child_or_probe
            ))

        # Build buffers
        nodes_buffer = b"".join(nodes_data)
        probes_buffer = b"".join(probes_data)

        # Header: node_count, probe_count, max_depth, _pad
        header_buffer = struct.pack(
            "<IIII",
            len(nodes_data),
            probe_index,
            self.config.max_depth,
            0,
        )

        return nodes_buffer, probes_buffer, header_buffer

    def _pack_node(
        self,
        node: AdaptiveProbeNode,
        child_mask: int,
        first_child_or_probe: int,
    ) -> bytes:
        """Pack an octree node for GPU upload (32 bytes)."""
        return struct.pack(
            "<3ff3fIIII",
            # bounds_min (vec3<f32>)
            node.bounds.min.x, node.bounds.min.y, node.bounds.min.z,
            # _pad0 (f32)
            0.0,
            # bounds_max (vec3<f32>)
            node.bounds.max.x, node.bounds.max.y, node.bounds.max.z,
            # child_mask (u32)
            child_mask,
            # first_child_or_probe (u32)
            first_child_or_probe,
            # depth (u32)
            node.depth,
            # _pad1 (u32)
            0,
        )

    def _pack_probe(self, probe: AdaptiveProbe) -> bytes:
        """Pack a probe for GPU upload (32 bytes)."""
        return struct.pack(
            "<3ff3ff",
            # position (vec3<f32>)
            probe.position.x, probe.position.y, probe.position.z,
            # blend_weight (f32)
            probe.blend_weight,
            # irradiance (vec3<f32>)
            probe.irradiance.x, probe.irradiance.y, probe.irradiance.z,
            # _pad (f32)
            0.0,
        )


# ============================================================================
# Visualization Helpers
# ============================================================================


def visualize_octree_bounds(grid: AdaptiveProbeGrid) -> List[Tuple[AABB, int]]:
    """Generate visualization data for octree node bounds.

    Args:
        grid: Adaptive grid to visualize

    Returns:
        List of (bounds, depth) tuples for each leaf node
    """
    result = []

    for node in grid.iter_leaf_nodes():
        result.append((node.bounds, node.depth))

    return result


def visualize_variance_heatmap(
    grid: AdaptiveProbeGrid,
) -> List[Tuple[Vec3, float]]:
    """Generate variance heatmap data.

    Args:
        grid: Adaptive grid to visualize

    Returns:
        List of (center_position, variance) tuples for each leaf node
    """
    result = []

    for node in grid.iter_leaf_nodes():
        center = node.bounds.center
        result.append((center, node.variance))

    return result


def visualize_probe_density(
    grid: AdaptiveProbeGrid,
    sample_resolution: int = 16,
) -> List[List[List[float]]]:
    """Generate 3D density grid for visualization.

    Args:
        grid: Adaptive grid to analyze
        sample_resolution: Resolution of output density grid

    Returns:
        3D list of probe density values [z][y][x]
    """
    bounds = grid.get_bounds()
    size = bounds.max - bounds.min

    density = []

    for iz in range(sample_resolution):
        plane = []
        for iy in range(sample_resolution):
            row = []
            for ix in range(sample_resolution):
                # Sample position
                fx = (ix + 0.5) / sample_resolution
                fy = (iy + 0.5) / sample_resolution
                fz = (iz + 0.5) / sample_resolution

                pos = Vec3(
                    bounds.min.x + fx * size.x,
                    bounds.min.y + fy * size.y,
                    bounds.min.z + fz * size.z,
                )

                # Find containing node
                node = grid._find_deepest_node(pos)

                if node is None:
                    row.append(0.0)
                else:
                    # Density is inversely proportional to node volume
                    node_volume = (
                        (node.bounds.max.x - node.bounds.min.x) *
                        (node.bounds.max.y - node.bounds.min.y) *
                        (node.bounds.max.z - node.bounds.min.z)
                    )
                    base_volume = (
                        size.x / grid.config.base_dimensions[0] *
                        size.y / grid.config.base_dimensions[1] *
                        size.z / grid.config.base_dimensions[2]
                    )
                    density_value = base_volume / max(node_volume, 0.001)
                    row.append(density_value)

            plane.append(row)
        density.append(plane)

    return density


# ============================================================================
# Test Scenes
# ============================================================================


def create_indoor_corridor_sampler() -> Callable[[Vec3], Vec3]:
    """Create irradiance sampler for indoor corridor test scene.

    Returns:
        Function that samples irradiance at a position
    """
    # Doorway positions (high variance expected)
    doorways = [
        Vec3(-40, 3, 0),
        Vec3(0, 3, 0),
        Vec3(38, 3, 0),
    ]

    def sampler(pos: Vec3) -> Vec3:
        # Base indoor lighting (dim)
        base = Vec3(0.02, 0.02, 0.03)

        # Check proximity to doorways (light portals)
        for door_pos in doorways:
            dist = (pos - door_pos).length()
            if dist < 10.0:
                # Bright near doorway
                factor = 1.0 - (dist / 10.0)
                sun = Vec3(1.0, 0.9, 0.7) * factor * 0.5
                base = base + sun

        return base

    return sampler


def create_outdoor_terrain_sampler() -> Callable[[Vec3], Vec3]:
    """Create irradiance sampler for outdoor terrain test scene.

    Returns:
        Function that samples uniform sky lighting
    """
    # Uniform sky lighting (low variance expected)
    sky_color = Vec3(0.3, 0.4, 0.6)

    def sampler(pos: Vec3) -> Vec3:
        # Height-based attenuation
        height_factor = max(0.0, 1.0 - pos.y / 100.0)
        return sky_color * (0.5 + 0.5 * height_factor)

    return sampler


def create_mixed_scene_sampler() -> Callable[[Vec3], Vec3]:
    """Create irradiance sampler for mixed interior/exterior test scene.

    Returns:
        Function with variable lighting based on position
    """
    building_bounds = AABB(Vec3(-20, 0, -20), Vec3(20, 10, 20))

    # Window positions
    windows = [
        (Vec3(-20, 5, 0), Vec3(1, 0, 0)),   # West wall, facing in
        (Vec3(20, 5, 0), Vec3(-1, 0, 0)),   # East wall, facing in
    ]

    def sampler(pos: Vec3) -> Vec3:
        if not building_bounds.contains(pos):
            # Exterior: uniform sky
            return Vec3(0.4, 0.5, 0.7)

        # Interior: dark base
        base = Vec3(0.02, 0.02, 0.03)

        # Light from windows
        for window_pos, window_dir in windows:
            to_pos = pos - window_pos
            dist = to_pos.length()

            if dist < 20.0:
                # Directional falloff
                to_pos_norm = to_pos.normalized()
                dot = max(0.0, to_pos_norm.dot(window_dir))

                if dot > 0.3:
                    factor = (1.0 - dist / 20.0) * (dot ** 2)
                    sun = Vec3(0.8, 0.7, 0.5) * factor
                    base = base + sun

        return base

    return sampler
