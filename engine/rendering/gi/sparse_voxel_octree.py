"""Sparse Voxel Octree Implementation for TRINITY GI (T-GIR-P11.2).

This module provides a Sparse Voxel Octree (SVO) implementation for memory-efficient
voxel storage, following Crassin 2011 / Laine 2010 methodology.

Features:
    - SVONode: Octree node with 8 child pointers and leaf/branch data
    - SVOBuilder: Construct SVO from dense VoxelGrid
    - SVOCompressor: Prune uniform regions, merge similar children
    - SVOMipGenerator: Build mip chain within SVO structure
    - SVOTraversal: Ray-octree intersection for cone tracing
    - SVOSerializer: Linearized array format for GPU upload
    - MemoryProfiler: Compare dense vs SVO memory usage

Memory Targets:
    - Dense 256^3: 64 MB (RGBA16F)
    - SVO 256^3: Target <13 MB (5-10x compression)

References:
    - Crassin et al., "GigaVoxels: Ray-Guided Streaming for Efficient and
      Detailed Voxel Rendering", I3D 2009
    - Laine & Karras, "Efficient Sparse Voxel Octrees", I3D 2010
    - Crassin, "Interactive Indirect Illumination Using Voxel Cone Tracing",
      Graphics Interface 2011
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional, Callable, Any

import numpy as np
from numpy.typing import NDArray

from engine.rendering.gi.voxelization import VoxelGrid, Voxel, VoxelResolution
from engine.rendering.gi.voxel_mipchain import VoxelData


# ============================================================================
# Constants
# ============================================================================

# Octree child count
OCTREE_CHILDREN = 8

# Minimum SVO resolution
MIN_SVO_RESOLUTION = 2

# Default similarity threshold for compression
DEFAULT_SIMILARITY_THRESHOLD = 0.05

# Default opacity threshold for empty detection
DEFAULT_OPACITY_THRESHOLD = 0.001

# Memory sizes (bytes)
DENSE_VOXEL_BYTES = 8  # RGBA16F
SVO_NODE_BYTES_BASE = 40  # Approximate Python object overhead
SVO_NODE_BYTES_OPTIMIZED = 16  # GPU-optimized packed format

# GPU serialization constants
GPU_CHILD_MASK_OFFSET = 0
GPU_CHILD_PTR_OFFSET = 4
GPU_DATA_OFFSET = 8
GPU_NODE_SIZE = 32  # Bytes per serialized node


# ============================================================================
# Enums
# ============================================================================


class NodeType(Enum):
    """Type of SVO node."""

    EMPTY = auto()      # Node has no geometry
    LEAF = auto()       # Node contains voxel data (terminal)
    BRANCH = auto()     # Node has children (internal)


class TraversalMode(Enum):
    """Ray traversal mode."""

    FRONT_TO_BACK = auto()   # Traverse front to back (for opacity compositing)
    BACK_TO_FRONT = auto()   # Traverse back to front (for transparency)
    ALL = auto()             # Visit all intersecting nodes


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class SVOVoxelData:
    """Voxel data stored in SVO nodes.

    Stores radiance and opacity for rendering, with additional
    metadata for filtering.

    Attributes:
        radiance: RGB radiance values (linear HDR)
        opacity: Opacity value (0.0 = empty, 1.0 = opaque)
        normal: Average normal direction (optional)
        variance: Variance for adaptive sampling (optional)
    """

    radiance: NDArray[np.float32]  # Shape (3,)
    opacity: float
    normal: Optional[NDArray[np.float32]] = None  # Shape (3,)
    variance: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate data."""
        self.radiance = np.asarray(self.radiance, dtype=np.float32)
        if self.radiance.shape != (3,):
            raise ValueError(f"Expected radiance shape (3,), got {self.radiance.shape}")
        if self.normal is not None:
            self.normal = np.asarray(self.normal, dtype=np.float32)
            if self.normal.shape != (3,):
                raise ValueError(f"Expected normal shape (3,), got {self.normal.shape}")

    @classmethod
    def empty(cls) -> SVOVoxelData:
        """Create empty voxel data."""
        return cls(np.zeros(3, dtype=np.float32), 0.0)

    @classmethod
    def from_voxel(cls, voxel: Voxel) -> SVOVoxelData:
        """Create from voxelization.Voxel."""
        radiance = np.array([voxel.albedo.x, voxel.albedo.y, voxel.albedo.z],
                          dtype=np.float32)
        normal = np.array([voxel.normal.x, voxel.normal.y, voxel.normal.z],
                         dtype=np.float32) if voxel.hit_count > 0 else None
        return cls(radiance, voxel.albedo.w, normal)

    @classmethod
    def from_voxel_data(cls, voxel_data: VoxelData) -> SVOVoxelData:
        """Create from voxel_mipchain.VoxelData."""
        return cls(voxel_data.radiance.copy(), voxel_data.opacity)

    def to_rgba(self) -> NDArray[np.float32]:
        """Convert to RGBA array."""
        return np.array([*self.radiance, self.opacity], dtype=np.float32)

    def is_empty(self, threshold: float = DEFAULT_OPACITY_THRESHOLD) -> bool:
        """Check if voxel is effectively empty."""
        return self.opacity < threshold

    def luminance(self) -> float:
        """Compute radiance luminance."""
        return float(
            0.2126 * self.radiance[0] +
            0.7152 * self.radiance[1] +
            0.0722 * self.radiance[2]
        )

    def similarity(self, other: SVOVoxelData) -> float:
        """Compute similarity to another voxel (0-1, 1=identical)."""
        # Compare RGBA values
        diff = np.abs(self.to_rgba() - other.to_rgba())
        max_diff = float(np.max(diff))
        return 1.0 - min(max_diff, 1.0)

    @staticmethod
    def average(voxels: list[SVOVoxelData]) -> SVOVoxelData:
        """Compute weighted average of voxels."""
        if not voxels:
            return SVOVoxelData.empty()

        total_opacity = sum(v.opacity for v in voxels)
        if total_opacity < 1e-6:
            return SVOVoxelData.empty()

        # Weighted average by opacity
        weighted_radiance = np.zeros(3, dtype=np.float32)
        weighted_normal: Optional[NDArray[np.float32]] = None
        normal_count = 0

        for v in voxels:
            weight = v.opacity / total_opacity
            weighted_radiance += v.radiance * weight
            if v.normal is not None:
                if weighted_normal is None:
                    weighted_normal = np.zeros(3, dtype=np.float32)
                weighted_normal += v.normal * weight
                normal_count += 1

        avg_opacity = total_opacity / len(voxels)

        # Normalize normal if present
        if weighted_normal is not None and normal_count > 0:
            length = np.linalg.norm(weighted_normal)
            if length > 1e-6:
                weighted_normal = weighted_normal / length

        return SVOVoxelData(weighted_radiance, avg_opacity, weighted_normal)


# ============================================================================
# SVONode
# ============================================================================


class SVONode:
    """Sparse Voxel Octree node.

    Each node can be:
        - Empty: No geometry in this region
        - Leaf: Contains voxel data (terminal node)
        - Branch: Has up to 8 children (internal node)

    Child indexing follows Morton order (Z-order curve):
        Index | x | y | z
        ------|---|---|---
          0   | 0 | 0 | 0
          1   | 1 | 0 | 0
          2   | 0 | 1 | 0
          3   | 1 | 1 | 0
          4   | 0 | 0 | 1
          5   | 1 | 0 | 1
          6   | 0 | 1 | 1
          7   | 1 | 1 | 1

    Attributes:
        node_type: Type of this node
        children: List of 8 child nodes (or None for empty slots)
        data: Voxel data for leaf nodes or averaged data for branches
        level: Depth level in the octree (0 = root)
        bounds: (min_coord, max_coord) tuple in grid space
    """

    __slots__ = ("node_type", "children", "data", "level", "bounds", "_child_mask")

    def __init__(
        self,
        node_type: NodeType = NodeType.EMPTY,
        level: int = 0,
        bounds: Optional[tuple[tuple[int, int, int], tuple[int, int, int]]] = None,
    ) -> None:
        """Initialize SVO node.

        Args:
            node_type: Initial node type
            level: Depth level in octree
            bounds: Grid-space bounds ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        self.node_type = node_type
        self.children: list[Optional[SVONode]] = [None] * OCTREE_CHILDREN
        self.data: Optional[SVOVoxelData] = None
        self.level = level
        self.bounds = bounds
        self._child_mask = 0  # Bitmask of non-empty children

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return self.node_type == NodeType.LEAF

    def is_empty(self) -> bool:
        """Check if node is empty."""
        return self.node_type == NodeType.EMPTY

    def is_branch(self) -> bool:
        """Check if node is a branch."""
        return self.node_type == NodeType.BRANCH

    def get_child(self, index: int) -> Optional[SVONode]:
        """Get child at index.

        Args:
            index: Child index (0-7)

        Returns:
            Child node or None if empty
        """
        if not 0 <= index < OCTREE_CHILDREN:
            raise IndexError(f"Child index {index} out of range [0, 7]")
        return self.children[index]

    def set_child(self, index: int, child: Optional[SVONode]) -> None:
        """Set child at index.

        Args:
            index: Child index (0-7)
            child: Child node or None
        """
        if not 0 <= index < OCTREE_CHILDREN:
            raise IndexError(f"Child index {index} out of range [0, 7]")
        self.children[index] = child

        # Update child mask
        if child is not None and not child.is_empty():
            self._child_mask |= (1 << index)
        else:
            self._child_mask &= ~(1 << index)

        # Update node type
        if self._child_mask != 0:
            self.node_type = NodeType.BRANCH
        elif self.data is not None and not self.data.is_empty():
            self.node_type = NodeType.LEAF
        else:
            self.node_type = NodeType.EMPTY

    def get_child_mask(self) -> int:
        """Get bitmask of non-empty children."""
        return self._child_mask

    def child_count(self) -> int:
        """Count non-empty children."""
        return bin(self._child_mask).count("1")

    def set_data(self, data: SVOVoxelData) -> None:
        """Set voxel data.

        Args:
            data: Voxel data to store
        """
        self.data = data
        if data.is_empty() and self._child_mask == 0:
            self.node_type = NodeType.EMPTY
        elif self._child_mask == 0:
            self.node_type = NodeType.LEAF

    def get_center(self) -> Optional[tuple[float, float, float]]:
        """Get center of node bounds."""
        if self.bounds is None:
            return None
        min_b, max_b = self.bounds
        return (
            (min_b[0] + max_b[0]) / 2.0,
            (min_b[1] + max_b[1]) / 2.0,
            (min_b[2] + max_b[2]) / 2.0,
        )

    def get_size(self) -> Optional[int]:
        """Get node size (side length in voxels)."""
        if self.bounds is None:
            return None
        min_b, max_b = self.bounds
        return max_b[0] - min_b[0]

    def iter_children(self) -> Iterator[tuple[int, SVONode]]:
        """Iterate over non-empty children.

        Yields:
            Tuples of (index, child_node)
        """
        for i in range(OCTREE_CHILDREN):
            if self.children[i] is not None and not self.children[i].is_empty():
                yield (i, self.children[i])

    def compute_averaged_data(self) -> SVOVoxelData:
        """Compute averaged data from children."""
        child_data = []
        for _, child in self.iter_children():
            if child.data is not None:
                child_data.append(child.data)
        if child_data:
            return SVOVoxelData.average(child_data)
        return SVOVoxelData.empty()

    @staticmethod
    def child_index(x: int, y: int, z: int) -> int:
        """Get child index from local coordinates.

        Args:
            x, y, z: Local coordinates (0 or 1)

        Returns:
            Child index (0-7)
        """
        return (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

    @staticmethod
    def child_offset(index: int) -> tuple[int, int, int]:
        """Get local offset from child index.

        Args:
            index: Child index (0-7)

        Returns:
            Tuple (dx, dy, dz) where each is 0 or 1
        """
        return (index & 1, (index >> 1) & 1, (index >> 2) & 1)


# ============================================================================
# SVOBuilder
# ============================================================================


@dataclass
class SVOBuildConfig:
    """Configuration for SVO construction.

    Attributes:
        opacity_threshold: Minimum opacity to consider non-empty
        min_level: Minimum subdivision level
        max_level: Maximum subdivision level
        prune_empty: Prune empty subtrees during construction
    """

    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD
    min_level: int = 0
    max_level: int = 8
    prune_empty: bool = True


@dataclass
class SVOBuildStats:
    """Statistics from SVO construction.

    Attributes:
        total_nodes: Total nodes created
        leaf_nodes: Number of leaf nodes
        branch_nodes: Number of branch nodes
        empty_nodes: Number of empty nodes
        pruned_nodes: Number of nodes pruned
        max_depth: Maximum depth reached
        build_time_ms: Construction time in milliseconds
    """

    total_nodes: int = 0
    leaf_nodes: int = 0
    branch_nodes: int = 0
    empty_nodes: int = 0
    pruned_nodes: int = 0
    max_depth: int = 0
    build_time_ms: float = 0.0


class SVOBuilder:
    """Builds Sparse Voxel Octree from dense VoxelGrid.

    Implements top-down subdivision with empty subtree pruning.
    The algorithm recursively subdivides the volume until reaching
    leaf voxels or detecting uniform/empty regions.

    Usage:
        builder = SVOBuilder(config)
        root = builder.build_from_dense(voxel_grid)
        stats = builder.get_stats()
    """

    __slots__ = ("config", "_stats", "_grid", "_resolution")

    def __init__(self, config: Optional[SVOBuildConfig] = None) -> None:
        """Initialize builder.

        Args:
            config: Build configuration
        """
        self.config = config or SVOBuildConfig()
        self._stats = SVOBuildStats()
        self._grid: Optional[VoxelGrid] = None
        self._resolution = 0

    def build_from_dense(self, grid: VoxelGrid) -> SVONode:
        """Build SVO from dense voxel grid.

        Args:
            grid: Dense VoxelGrid from voxelization

        Returns:
            Root SVONode of the octree
        """
        start_time = time.perf_counter()

        self._grid = grid
        self._resolution = grid.resolution
        self._stats = SVOBuildStats()

        # Compute max level from resolution
        max_level = int(math.log2(self._resolution))
        self.config.max_level = min(self.config.max_level, max_level)

        # Create root node
        bounds = ((0, 0, 0), (self._resolution, self._resolution, self._resolution))
        root = self._subdivide_node(0, bounds)

        self._stats.build_time_ms = (time.perf_counter() - start_time) * 1000.0
        return root

    def _subdivide_node(
        self,
        level: int,
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> SVONode:
        """Recursively subdivide to create octree.

        Args:
            level: Current depth level
            bounds: Current bounds in grid coordinates

        Returns:
            SVONode for this region
        """
        min_b, max_b = bounds
        size = max_b[0] - min_b[0]

        node = SVONode(level=level, bounds=bounds)
        self._stats.total_nodes += 1
        self._stats.max_depth = max(self._stats.max_depth, level)

        # Check if this is a leaf level (single voxel)
        if size == 1:
            return self._create_leaf_node(node, min_b)

        # Check if we should subdivide or summarize
        if level >= self.config.max_level or not self.should_subdivide(bounds):
            # Create leaf with averaged data
            avg_data = self._average_region(bounds)
            node.set_data(avg_data)
            if avg_data.is_empty(self.config.opacity_threshold):
                self._stats.empty_nodes += 1
            else:
                self._stats.leaf_nodes += 1
            return node

        # Subdivide into 8 children
        half_size = size // 2
        all_empty = True

        for i in range(OCTREE_CHILDREN):
            dx, dy, dz = SVONode.child_offset(i)
            child_min = (
                min_b[0] + dx * half_size,
                min_b[1] + dy * half_size,
                min_b[2] + dz * half_size,
            )
            child_max = (
                child_min[0] + half_size,
                child_min[1] + half_size,
                child_min[2] + half_size,
            )
            child_bounds = (child_min, child_max)

            child = self._subdivide_node(level + 1, child_bounds)

            if self.config.prune_empty and child.is_empty():
                self._stats.pruned_nodes += 1
            else:
                node.set_child(i, child)
                all_empty = False

        # If all children are empty, mark node as empty
        if all_empty:
            node.node_type = NodeType.EMPTY
            self._stats.empty_nodes += 1
        else:
            # Compute averaged data for branch node
            node.set_data(node.compute_averaged_data())
            self._stats.branch_nodes += 1

        return node

    def _create_leaf_node(
        self,
        node: SVONode,
        coords: tuple[int, int, int],
    ) -> SVONode:
        """Create leaf node from single voxel.

        Args:
            node: Node to populate
            coords: Voxel coordinates

        Returns:
            Populated leaf node
        """
        x, y, z = coords
        voxel = self._grid.get_voxel(x, y, z)

        if voxel.is_empty() or voxel.albedo.w < self.config.opacity_threshold:
            node.node_type = NodeType.EMPTY
            self._stats.empty_nodes += 1
        else:
            data = SVOVoxelData.from_voxel(voxel)
            node.set_data(data)
            node.node_type = NodeType.LEAF
            self._stats.leaf_nodes += 1

        return node

    def _average_region(
        self,
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> SVOVoxelData:
        """Compute average voxel data for a region.

        Args:
            bounds: Region bounds

        Returns:
            Averaged SVOVoxelData
        """
        min_b, max_b = bounds
        voxels = []

        for z in range(min_b[2], max_b[2]):
            for y in range(min_b[1], max_b[1]):
                for x in range(min_b[0], max_b[0]):
                    voxel = self._grid.get_voxel(x, y, z)
                    if not voxel.is_empty():
                        voxels.append(SVOVoxelData.from_voxel(voxel))

        return SVOVoxelData.average(voxels)

    def should_subdivide(
        self,
        bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> bool:
        """Determine if a region should be subdivided.

        Subdivides if:
            - Region contains non-uniform data
            - Region is not all empty
            - Region has significant opacity variation

        Args:
            bounds: Region bounds

        Returns:
            True if region should be subdivided
        """
        min_b, max_b = bounds
        size = max_b[0] - min_b[0]

        # Always subdivide at coarse levels
        if size > 4:
            return True

        # Sample region to check uniformity
        samples = []
        sample_step = max(1, size // 2)

        for z in range(min_b[2], max_b[2], sample_step):
            for y in range(min_b[1], max_b[1], sample_step):
                for x in range(min_b[0], max_b[0], sample_step):
                    voxel = self._grid.get_voxel(x, y, z)
                    samples.append(voxel.albedo.w)

        if not samples:
            return False

        # Check opacity variance
        min_opacity = min(samples)
        max_opacity = max(samples)

        # Subdivide if there's significant variation
        return (max_opacity - min_opacity) > 0.1

    def get_stats(self) -> SVOBuildStats:
        """Get construction statistics."""
        return self._stats


# ============================================================================
# SVOCompressor
# ============================================================================


@dataclass
class SVOCompressionConfig:
    """Configuration for SVO compression.

    Attributes:
        merge_threshold: Similarity threshold for merging children
        prune_uniform: Prune uniform regions to single node
        min_merge_level: Minimum level for merging
    """

    merge_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    prune_uniform: bool = True
    min_merge_level: int = 2


@dataclass
class SVOCompressionStats:
    """Statistics from SVO compression.

    Attributes:
        original_nodes: Node count before compression
        compressed_nodes: Node count after compression
        merged_regions: Number of uniform regions merged
        compression_ratio: Ratio of compressed to original
        compression_time_ms: Compression time
    """

    original_nodes: int = 0
    compressed_nodes: int = 0
    merged_regions: int = 0
    compression_ratio: float = 1.0
    compression_time_ms: float = 0.0


class SVOCompressor:
    """Compresses SVO by pruning uniform regions.

    Compression strategies:
        1. Prune uniform regions (all children have same value)
        2. Merge similar children (within threshold)
        3. Collapse sparse branches

    Usage:
        compressor = SVOCompressor(config)
        compressed_root = compressor.compress(root)
        stats = compressor.get_stats()
    """

    __slots__ = ("config", "_stats")

    def __init__(self, config: Optional[SVOCompressionConfig] = None) -> None:
        """Initialize compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or SVOCompressionConfig()
        self._stats = SVOCompressionStats()

    def compress(self, root: SVONode) -> SVONode:
        """Compress SVO by pruning uniform regions.

        Args:
            root: Root node to compress

        Returns:
            Compressed root node (may be same reference if no changes)
        """
        start_time = time.perf_counter()

        # Count original nodes
        self._stats.original_nodes = self._count_nodes(root)
        self._stats.merged_regions = 0

        # Compress bottom-up
        compressed = self._compress_node(root)

        # Count compressed nodes
        self._stats.compressed_nodes = self._count_nodes(compressed)

        # Compute ratio
        if self._stats.original_nodes > 0:
            self._stats.compression_ratio = (
                self._stats.compressed_nodes / self._stats.original_nodes
            )

        self._stats.compression_time_ms = (time.perf_counter() - start_time) * 1000.0
        return compressed

    def _compress_node(self, node: SVONode) -> SVONode:
        """Recursively compress a node.

        Args:
            node: Node to compress

        Returns:
            Compressed node
        """
        if node.is_empty() or node.is_leaf():
            return node

        # First, compress all children
        for i, child in node.iter_children():
            compressed_child = self._compress_node(child)
            node.set_child(i, compressed_child)

        # Check if all children are uniform
        if self.config.prune_uniform and node.level >= self.config.min_merge_level:
            if self._is_uniform(node):
                # Convert to leaf with averaged data
                averaged = node.compute_averaged_data()
                leaf = SVONode(NodeType.LEAF, node.level, node.bounds)
                leaf.set_data(averaged)
                self._stats.merged_regions += 1
                return leaf

        # Check if children can be merged
        if self._can_merge_children(node):
            averaged = node.compute_averaged_data()
            leaf = SVONode(NodeType.LEAF, node.level, node.bounds)
            leaf.set_data(averaged)
            self._stats.merged_regions += 1
            return leaf

        return node

    def _is_uniform(self, node: SVONode) -> bool:
        """Check if all children are uniform (same value).

        Args:
            node: Branch node to check

        Returns:
            True if all children have similar values
        """
        if node.is_empty() or node.is_leaf():
            return True

        child_data = []
        for _, child in node.iter_children():
            if child.data is not None:
                child_data.append(child.data)

        if len(child_data) < 2:
            return True

        # Compare all pairs
        reference = child_data[0]
        for data in child_data[1:]:
            if reference.similarity(data) < (1.0 - self.config.merge_threshold):
                return False

        return True

    def _can_merge_children(self, node: SVONode) -> bool:
        """Check if children can be merged based on similarity.

        Args:
            node: Branch node to check

        Returns:
            True if children are similar enough to merge
        """
        if node.child_count() == 0:
            return True

        # All children must be leaves or empty
        for _, child in node.iter_children():
            if child.is_branch():
                return False

        return self._is_uniform(node)

    def _count_nodes(self, node: SVONode) -> int:
        """Count total nodes in tree.

        Args:
            node: Root node

        Returns:
            Total node count
        """
        count = 1
        for _, child in node.iter_children():
            count += self._count_nodes(child)
        return count

    def get_compression_ratio(self) -> float:
        """Get compression ratio (compressed/original)."""
        return self._stats.compression_ratio

    def get_stats(self) -> SVOCompressionStats:
        """Get compression statistics."""
        return self._stats


# ============================================================================
# SVOMipGenerator
# ============================================================================


@dataclass
class SVOMipConfig:
    """Configuration for mip chain generation.

    Attributes:
        alpha_weighted: Use alpha-weighted averaging
        store_variance: Store variance per node
    """

    alpha_weighted: bool = True
    store_variance: bool = False


class SVOMipGenerator:
    """Generates mip chain within SVO structure.

    The mip chain is implicit in the SVO hierarchy:
        - Leaf nodes at level N are mip 0
        - Parent nodes at level N-1 are mip 1
        - Root node is highest mip level

    Each branch node stores averaged data from its children,
    which can be used for level-of-detail sampling.

    Usage:
        generator = SVOMipGenerator(config)
        generator.generate_mips(root)
    """

    __slots__ = ("config",)

    def __init__(self, config: Optional[SVOMipConfig] = None) -> None:
        """Initialize mip generator.

        Args:
            config: Mip generation configuration
        """
        self.config = config or SVOMipConfig()

    def generate_mips(self, root: SVONode) -> None:
        """Generate mip data for all branch nodes.

        Traverses bottom-up, computing averaged data for each branch
        from its children.

        Args:
            root: Root node of SVO
        """
        self._generate_mips_recursive(root)

    def _generate_mips_recursive(self, node: SVONode) -> Optional[SVOVoxelData]:
        """Recursively generate mip data.

        Args:
            node: Current node

        Returns:
            Averaged data for this node
        """
        if node.is_empty():
            return None

        if node.is_leaf():
            return node.data

        # Process all children first
        child_data = []
        for i in range(OCTREE_CHILDREN):
            child = node.get_child(i)
            if child is not None:
                data = self._generate_mips_recursive(child)
                if data is not None:
                    child_data.append(data)

        # Compute averaged data
        if child_data:
            if self.config.alpha_weighted:
                averaged = SVOVoxelData.average(child_data)
            else:
                # Simple average
                radiance = np.mean([d.radiance for d in child_data], axis=0)
                opacity = np.mean([d.opacity for d in child_data])
                averaged = SVOVoxelData(radiance.astype(np.float32), float(opacity))

            node.set_data(averaged)
            return averaged

        return None

    def get_mip_level(self, node: SVONode, max_level: int) -> int:
        """Get effective mip level of a node.

        Args:
            node: Node to query
            max_level: Maximum level in the tree

        Returns:
            Mip level (0 = finest, max = coarsest)
        """
        return max_level - node.level


# ============================================================================
# SVOTraversal
# ============================================================================


@dataclass
class RayHit:
    """Ray-octree intersection result.

    Attributes:
        hit: Whether ray hit geometry
        t_near: Near intersection distance
        t_far: Far intersection distance
        node: Hit node
        position: Hit position in world space
        data: Voxel data at hit
    """

    hit: bool
    t_near: float = 0.0
    t_far: float = float("inf")
    node: Optional[SVONode] = None
    position: Optional[tuple[float, float, float]] = None
    data: Optional[SVOVoxelData] = None


@dataclass
class ConeTraceResult:
    """Result of voxel cone tracing.

    Attributes:
        accumulated_radiance: Accumulated radiance along cone
        accumulated_opacity: Accumulated opacity
        steps: Number of steps taken
        max_distance: Maximum distance traced
    """

    accumulated_radiance: NDArray[np.float32]
    accumulated_opacity: float
    steps: int
    max_distance: float

    @classmethod
    def empty(cls) -> ConeTraceResult:
        """Create empty result."""
        return cls(np.zeros(3, dtype=np.float32), 0.0, 0, 0.0)


class SVOTraversal:
    """Ray-octree traversal for cone tracing.

    Implements stack-based iterative traversal following
    Laine & Karras 2010 methodology.

    Usage:
        traversal = SVOTraversal(root, voxel_size)
        result = traversal.traverse_ray(origin, direction)
        cone = traversal.trace_cone(origin, direction, aperture)
    """

    __slots__ = ("root", "voxel_size", "max_level", "resolution")

    def __init__(
        self,
        root: SVONode,
        voxel_size: float,
        resolution: int,
    ) -> None:
        """Initialize traversal.

        Args:
            root: Root node of SVO
            voxel_size: Size of one voxel in world units
            resolution: Grid resolution
        """
        self.root = root
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.max_level = int(math.log2(resolution))

    def traverse_ray(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = float("inf"),
    ) -> RayHit:
        """Traverse ray through octree.

        Args:
            origin: Ray origin in voxel space
            direction: Ray direction (normalized)
            max_distance: Maximum traversal distance

        Returns:
            RayHit result
        """
        # Normalize direction
        dx, dy, dz = direction
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return RayHit(False)
        dx, dy, dz = dx / length, dy / length, dz / length

        # Initialize ray
        ox, oy, oz = origin

        # Check bounds intersection
        t_near, t_far = self._ray_box_intersection(
            origin, (dx, dy, dz),
            (0.0, 0.0, 0.0),
            (float(self.resolution), float(self.resolution), float(self.resolution)),
        )

        if t_near >= t_far or t_far < 0:
            return RayHit(False)

        t_near = max(0.0, t_near)
        t_far = min(t_far, max_distance)

        # Stack-based traversal
        result = self._traverse_recursive(
            self.root,
            origin, (dx, dy, dz),
            t_near, t_far,
        )

        return result

    def _traverse_recursive(
        self,
        node: SVONode,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        t_near: float,
        t_far: float,
    ) -> RayHit:
        """Recursive traversal helper.

        Args:
            node: Current node
            origin: Ray origin
            direction: Ray direction
            t_near: Entry distance
            t_far: Exit distance

        Returns:
            RayHit result
        """
        if node.is_empty():
            return RayHit(False)

        if node.bounds is None:
            return RayHit(False)

        # Check if ray intersects this node's bounds
        min_b, max_b = node.bounds
        box_min = (float(min_b[0]), float(min_b[1]), float(min_b[2]))
        box_max = (float(max_b[0]), float(max_b[1]), float(max_b[2]))

        node_t_near, node_t_far = self._ray_box_intersection(
            origin, direction, box_min, box_max
        )

        if node_t_near >= node_t_far or node_t_far < t_near or node_t_near > t_far:
            return RayHit(False)

        # Clamp to search range
        node_t_near = max(node_t_near, t_near)
        node_t_far = min(node_t_far, t_far)

        # If leaf, return hit
        if node.is_leaf():
            ox, oy, oz = origin
            dx, dy, dz = direction
            hit_pos = (
                ox + dx * node_t_near,
                oy + dy * node_t_near,
                oz + dz * node_t_near,
            )
            return RayHit(
                hit=True,
                t_near=node_t_near,
                t_far=node_t_far,
                node=node,
                position=hit_pos,
                data=node.data,
            )

        # Traverse children front to back
        best_hit = RayHit(False)
        for _, child in node.iter_children():
            child_hit = self._traverse_recursive(
                child, origin, direction, node_t_near, node_t_far
            )
            if child_hit.hit:
                if not best_hit.hit or child_hit.t_near < best_hit.t_near:
                    best_hit = child_hit
                    # Update far distance to avoid unnecessary traversal
                    node_t_far = min(node_t_far, child_hit.t_near)

        return best_hit

    def find_leaf(
        self,
        x: float, y: float, z: float,
    ) -> Optional[SVONode]:
        """Find leaf node containing position.

        Args:
            x, y, z: Position in voxel space

        Returns:
            Leaf node or None if empty
        """
        return self._find_leaf_recursive(self.root, x, y, z)

    def _find_leaf_recursive(
        self,
        node: SVONode,
        x: float, y: float, z: float,
    ) -> Optional[SVONode]:
        """Recursive leaf finder.

        Args:
            node: Current node
            x, y, z: Position

        Returns:
            Leaf node or None
        """
        if node.is_empty():
            return None

        if node.bounds is None:
            return None

        # Check if position is in bounds
        min_b, max_b = node.bounds
        if not (min_b[0] <= x < max_b[0] and
                min_b[1] <= y < max_b[1] and
                min_b[2] <= z < max_b[2]):
            return None

        if node.is_leaf():
            return node

        # Find child containing position
        size = (max_b[0] - min_b[0]) // 2
        cx = int((x - min_b[0]) / size) & 1
        cy = int((y - min_b[1]) / size) & 1
        cz = int((z - min_b[2]) / size) & 1
        child_idx = SVONode.child_index(cx, cy, cz)

        child = node.get_child(child_idx)
        if child is None:
            return None

        return self._find_leaf_recursive(child, x, y, z)

    def sample_at_position(
        self,
        x: float, y: float, z: float,
        level: int = 0,
    ) -> Optional[SVOVoxelData]:
        """Sample voxel data at position with level-of-detail.

        Args:
            x, y, z: Position in voxel space
            level: LOD level (0 = finest)

        Returns:
            Sampled voxel data or None
        """
        target_level = self.max_level - level
        return self._sample_recursive(self.root, x, y, z, target_level)

    def _sample_recursive(
        self,
        node: SVONode,
        x: float, y: float, z: float,
        target_level: int,
    ) -> Optional[SVOVoxelData]:
        """Recursive sampling helper.

        Args:
            node: Current node
            x, y, z: Position
            target_level: Target tree level

        Returns:
            Sampled data or None
        """
        if node.is_empty():
            return None

        if node.bounds is None:
            return None

        # Check bounds
        min_b, max_b = node.bounds
        if not (min_b[0] <= x < max_b[0] and
                min_b[1] <= y < max_b[1] and
                min_b[2] <= z < max_b[2]):
            return None

        # Return data if at target level or leaf
        if node.level >= target_level or node.is_leaf():
            return node.data

        # Continue to children
        size = (max_b[0] - min_b[0]) // 2
        if size < 1:
            return node.data

        cx = int((x - min_b[0]) / size) & 1
        cy = int((y - min_b[1]) / size) & 1
        cz = int((z - min_b[2]) / size) & 1
        child_idx = SVONode.child_index(cx, cy, cz)

        child = node.get_child(child_idx)
        if child is None:
            return node.data

        return self._sample_recursive(child, x, y, z, target_level)

    def trace_cone(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        aperture: float,
        max_distance: float = 100.0,
        max_steps: int = 64,
    ) -> ConeTraceResult:
        """Trace cone through octree for GI.

        Implements voxel cone tracing (Crassin 2011) using
        the SVO's hierarchical structure for LOD selection.

        Args:
            origin: Cone origin in voxel space
            direction: Cone direction (normalized)
            aperture: Cone aperture angle (radians)
            max_distance: Maximum trace distance
            max_steps: Maximum number of steps

        Returns:
            ConeTraceResult with accumulated radiance
        """
        # Normalize direction
        dx, dy, dz = direction
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return ConeTraceResult.empty()
        dx, dy, dz = dx / length, dy / length, dz / length

        # Initialize accumulators
        accumulated_radiance = np.zeros(3, dtype=np.float32)
        accumulated_opacity = 0.0

        # Start tracing
        t = 1.0  # Start offset to avoid self-intersection
        step_count = 0
        tan_half_aperture = math.tan(aperture * 0.5)

        while t < max_distance and accumulated_opacity < 0.99 and step_count < max_steps:
            # Current position
            px = origin[0] + dx * t
            py = origin[1] + dy * t
            pz = origin[2] + dz * t

            # Check bounds
            if not (0 <= px < self.resolution and
                    0 <= py < self.resolution and
                    0 <= pz < self.resolution):
                break

            # Compute cone diameter at current distance
            diameter = 2.0 * t * tan_half_aperture

            # Select mip level based on diameter
            level = max(0, min(self.max_level, int(math.log2(max(1.0, diameter)))))

            # Sample at appropriate level
            sample = self.sample_at_position(px, py, pz, level)

            if sample is not None and sample.opacity > 0.001:
                # Front-to-back compositing
                alpha = sample.opacity * (1.0 - accumulated_opacity)
                accumulated_radiance += sample.radiance * alpha
                accumulated_opacity += alpha

            # Step based on diameter (adaptive stepping)
            step_size = max(1.0, diameter * 0.5)
            t += step_size
            step_count += 1

        return ConeTraceResult(
            accumulated_radiance=accumulated_radiance,
            accumulated_opacity=accumulated_opacity,
            steps=step_count,
            max_distance=t,
        )

    def _ray_box_intersection(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Compute ray-box intersection distances.

        Args:
            origin: Ray origin
            direction: Ray direction
            box_min: Box minimum corner
            box_max: Box maximum corner

        Returns:
            Tuple of (t_near, t_far) or (inf, -inf) if no hit
        """
        t_min = float("-inf")
        t_max = float("inf")

        for i in range(3):
            o = (origin[0], origin[1], origin[2])[i]
            d = (direction[0], direction[1], direction[2])[i]
            b_min = (box_min[0], box_min[1], box_min[2])[i]
            b_max = (box_max[0], box_max[1], box_max[2])[i]

            if abs(d) < 1e-8:
                if o < b_min or o > b_max:
                    return (float("inf"), float("-inf"))
            else:
                t1 = (b_min - o) / d
                t2 = (b_max - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

        return (t_min, t_max)


# ============================================================================
# SVOSerializer
# ============================================================================


@dataclass
class SerializedSVO:
    """Serialized SVO data for GPU upload.

    The serialization uses a pointer-less encoding with relative
    offsets for efficient GPU traversal.

    Node Layout (32 bytes):
        Offset 0-3:   Child mask (8 bits used)
        Offset 4-7:   First child offset (relative to this node)
        Offset 8-15:  Radiance RGB (3x float16) + padding
        Offset 16-19: Opacity (float16) + padding
        Offset 20-31: Reserved / Normal data

    Attributes:
        nodes: Packed node data
        node_count: Number of nodes
        root_offset: Offset of root node
        memory_bytes: Total memory size
    """

    nodes: bytes
    node_count: int
    root_offset: int
    memory_bytes: int


class SVOSerializer:
    """Serializes SVO for GPU upload.

    Converts pointer-based octree to linearized array format
    with relative offsets for efficient GPU traversal.

    Usage:
        serializer = SVOSerializer()
        serialized = serializer.serialize(root)
        buffer = serialized.nodes

        # Round-trip
        restored = serializer.deserialize(serialized)
    """

    __slots__ = ("_node_list", "_node_offsets")

    def __init__(self) -> None:
        """Initialize serializer."""
        self._node_list: list[SVONode] = []
        self._node_offsets: dict[int, int] = {}

    def serialize(self, root: SVONode) -> SerializedSVO:
        """Serialize SVO to GPU-uploadable format.

        Args:
            root: Root node of SVO

        Returns:
            SerializedSVO containing packed data
        """
        self._node_list = []
        self._node_offsets = {}

        # Collect nodes in breadth-first order
        self._collect_nodes(root)

        # Compute offsets
        for i, node in enumerate(self._node_list):
            self._node_offsets[id(node)] = i

        # Pack nodes
        packed_data = bytearray()
        for node in self._node_list:
            packed_data.extend(self._pack_node(node))

        return SerializedSVO(
            nodes=bytes(packed_data),
            node_count=len(self._node_list),
            root_offset=0,
            memory_bytes=len(packed_data),
        )

    def _collect_nodes(self, node: SVONode) -> None:
        """Collect nodes in BFS order.

        Args:
            node: Starting node
        """
        queue = [node]

        while queue:
            current = queue.pop(0)
            self._node_list.append(current)

            for _, child in current.iter_children():
                queue.append(child)

    def _pack_node(self, node: SVONode) -> bytes:
        """Pack a single node to bytes.

        Args:
            node: Node to pack

        Returns:
            32-byte packed representation
        """
        data = bytearray(GPU_NODE_SIZE)

        # Child mask (byte 0)
        data[0] = node.get_child_mask()

        # Node type (byte 1)
        data[1] = node.node_type.value

        # Level (byte 2)
        data[2] = node.level

        # Reserved (byte 3)
        data[3] = 0

        # First child offset (bytes 4-7)
        first_child_offset = 0
        for i in range(OCTREE_CHILDREN):
            child = node.get_child(i)
            if child is not None:
                child_idx = self._node_offsets.get(id(child), 0)
                first_child_offset = child_idx
                break
        struct.pack_into("<I", data, 4, first_child_offset)

        # Voxel data (bytes 8-23)
        if node.data is not None:
            # Radiance as float16 (6 bytes)
            r16 = np.float16(node.data.radiance[0])
            g16 = np.float16(node.data.radiance[1])
            b16 = np.float16(node.data.radiance[2])
            struct.pack_into("<eee", data, 8, r16, g16, b16)

            # Opacity as float16 (2 bytes)
            a16 = np.float16(node.data.opacity)
            struct.pack_into("<e", data, 14, a16)

            # Normal as int8 (3 bytes) if present
            if node.data.normal is not None:
                nx = int(node.data.normal[0] * 127)
                ny = int(node.data.normal[1] * 127)
                nz = int(node.data.normal[2] * 127)
                data[16] = nx & 0xFF
                data[17] = ny & 0xFF
                data[18] = nz & 0xFF

        return bytes(data)

    def deserialize(self, serialized: SerializedSVO) -> SVONode:
        """Deserialize SVO from packed format.

        Args:
            serialized: SerializedSVO data

        Returns:
            Reconstructed root node
        """
        if serialized.node_count == 0:
            return SVONode(NodeType.EMPTY)

        # Create all nodes first
        nodes: list[SVONode] = []
        data = serialized.nodes

        for i in range(serialized.node_count):
            offset = i * GPU_NODE_SIZE
            node_data = data[offset:offset + GPU_NODE_SIZE]
            node = self._unpack_node(node_data, i)
            nodes.append(node)

        # Link children
        for i, node in enumerate(nodes):
            offset = i * GPU_NODE_SIZE
            node_data = data[offset:offset + GPU_NODE_SIZE]

            child_mask = node_data[0]
            first_child = struct.unpack_from("<I", node_data, 4)[0]

            child_idx = first_child
            for c in range(OCTREE_CHILDREN):
                if child_mask & (1 << c):
                    if child_idx < len(nodes):
                        node.set_child(c, nodes[child_idx])
                        child_idx += 1

        return nodes[serialized.root_offset]

    def _unpack_node(self, data: bytes, index: int) -> SVONode:
        """Unpack a single node from bytes.

        Args:
            data: 32-byte packed data
            index: Node index

        Returns:
            Unpacked SVONode
        """
        node_type = NodeType(data[1])
        level = data[2]

        node = SVONode(node_type, level)

        # Unpack voxel data
        r16, g16, b16 = struct.unpack_from("<eee", data, 8)
        a16 = struct.unpack_from("<e", data, 14)[0]

        radiance = np.array([float(r16), float(g16), float(b16)], dtype=np.float32)
        opacity = float(a16)

        # Unpack normal
        nx = (data[16] if data[16] < 128 else data[16] - 256) / 127.0
        ny = (data[17] if data[17] < 128 else data[17] - 256) / 127.0
        nz = (data[18] if data[18] < 128 else data[18] - 256) / 127.0
        normal = np.array([nx, ny, nz], dtype=np.float32)

        voxel_data = SVOVoxelData(radiance, opacity, normal)
        node.set_data(voxel_data)

        return node

    def get_gpu_buffer(self, serialized: SerializedSVO) -> bytes:
        """Get raw GPU buffer data.

        Args:
            serialized: SerializedSVO

        Returns:
            Raw bytes for GPU upload
        """
        return serialized.nodes


# ============================================================================
# MemoryProfiler
# ============================================================================


@dataclass
class MemoryProfile:
    """Memory usage profile.

    Attributes:
        dense_bytes: Memory for dense representation
        svo_bytes: Memory for SVO representation
        compression_ratio: SVO / Dense ratio
        node_count: Number of SVO nodes
        leaf_count: Number of leaf nodes
        fill_ratio: Filled voxels / total voxels
    """

    dense_bytes: int
    svo_bytes: int
    compression_ratio: float
    node_count: int
    leaf_count: int
    fill_ratio: float

    @property
    def savings_ratio(self) -> float:
        """Get memory savings ratio (Dense / SVO)."""
        if self.svo_bytes == 0:
            return float("inf")
        return self.dense_bytes / self.svo_bytes


@dataclass
class SceneProfile:
    """Profile for a specific scene type.

    Attributes:
        name: Scene name
        description: Scene description
        memory: Memory profile
        build_time_ms: Build time in milliseconds
    """

    name: str
    description: str
    memory: MemoryProfile
    build_time_ms: float


class MemoryProfiler:
    """Profiles memory usage comparing dense vs SVO.

    Usage:
        profiler = MemoryProfiler()
        profile = profiler.profile(svo_root, resolution)
        report = profiler.generate_report()
    """

    __slots__ = ("_profiles",)

    def __init__(self) -> None:
        """Initialize profiler."""
        self._profiles: list[SceneProfile] = []

    def profile(
        self,
        root: SVONode,
        resolution: int,
        name: str = "unnamed",
        description: str = "",
    ) -> MemoryProfile:
        """Profile SVO memory usage.

        Args:
            root: SVO root node
            resolution: Original dense resolution
            name: Profile name
            description: Profile description

        Returns:
            MemoryProfile with usage statistics
        """
        # Dense memory: RGBA16F = 8 bytes per voxel
        dense_bytes = resolution ** 3 * DENSE_VOXEL_BYTES

        # Count nodes
        node_count, leaf_count, fill_count = self._count_nodes_detailed(root)

        # SVO memory estimate (GPU-optimized format)
        svo_bytes = node_count * GPU_NODE_SIZE

        # Fill ratio
        total_voxels = resolution ** 3
        fill_ratio = fill_count / total_voxels if total_voxels > 0 else 0.0

        # Compression ratio
        compression_ratio = svo_bytes / dense_bytes if dense_bytes > 0 else 0.0

        profile = MemoryProfile(
            dense_bytes=dense_bytes,
            svo_bytes=svo_bytes,
            compression_ratio=compression_ratio,
            node_count=node_count,
            leaf_count=leaf_count,
            fill_ratio=fill_ratio,
        )

        return profile

    def _count_nodes_detailed(
        self,
        node: SVONode,
    ) -> tuple[int, int, int]:
        """Count nodes with detailed breakdown.

        Args:
            node: Root node

        Returns:
            Tuple of (total_nodes, leaf_nodes, filled_voxels)
        """
        total = 1
        leaves = 0
        filled = 0

        if node.is_leaf():
            leaves = 1
            if node.data is not None and not node.data.is_empty():
                filled = 1
        elif node.is_branch():
            for _, child in node.iter_children():
                c_total, c_leaves, c_filled = self._count_nodes_detailed(child)
                total += c_total
                leaves += c_leaves
                filled += c_filled

        return (total, leaves, filled)

    def compare(
        self,
        svo_root: SVONode,
        dense_grid: VoxelGrid,
    ) -> dict[str, Any]:
        """Compare SVO and dense representations.

        Args:
            svo_root: SVO root
            dense_grid: Original dense grid

        Returns:
            Comparison dictionary
        """
        resolution = dense_grid.resolution
        profile = self.profile(svo_root, resolution)

        dense_filled = dense_grid.count_filled_voxels()
        dense_total = dense_grid.total_voxels

        return {
            "resolution": resolution,
            "dense_memory_mb": profile.dense_bytes / (1024 * 1024),
            "svo_memory_mb": profile.svo_bytes / (1024 * 1024),
            "compression_ratio": profile.compression_ratio,
            "savings_ratio": profile.savings_ratio,
            "dense_fill_ratio": dense_filled / dense_total if dense_total > 0 else 0,
            "svo_fill_ratio": profile.fill_ratio,
            "node_count": profile.node_count,
            "leaf_count": profile.leaf_count,
            "target_met": profile.savings_ratio >= 5.0,  # Target: 5-10x savings
        }

    def generate_report(self, profiles: Optional[list[SceneProfile]] = None) -> str:
        """Generate memory comparison report.

        Args:
            profiles: Scene profiles to include

        Returns:
            Formatted report string
        """
        profiles = profiles or self._profiles

        report = []
        report.append("=" * 60)
        report.append("SVO Memory Profiling Report")
        report.append("=" * 60)
        report.append("")

        for scene in profiles:
            report.append(f"Scene: {scene.name}")
            report.append(f"  Description: {scene.description}")
            report.append(f"  Dense Memory: {scene.memory.dense_bytes / (1024*1024):.2f} MB")
            report.append(f"  SVO Memory: {scene.memory.svo_bytes / (1024*1024):.2f} MB")
            report.append(f"  Savings: {scene.memory.savings_ratio:.1f}x")
            report.append(f"  Compression: {scene.memory.compression_ratio:.2%}")
            report.append(f"  Fill Ratio: {scene.memory.fill_ratio:.2%}")
            report.append(f"  Nodes: {scene.memory.node_count}")
            report.append(f"  Leaves: {scene.memory.leaf_count}")
            report.append(f"  Build Time: {scene.build_time_ms:.2f} ms")
            target_met = scene.memory.savings_ratio >= 5.0
            report.append(f"  Target Met (5x+): {'YES' if target_met else 'NO'}")
            report.append("")

        # Summary
        if profiles:
            avg_savings = sum(p.memory.savings_ratio for p in profiles) / len(profiles)
            report.append("-" * 40)
            report.append(f"Average Savings: {avg_savings:.1f}x")
            target_met = all(p.memory.savings_ratio >= 5.0 for p in profiles)
            report.append(f"All Targets Met: {'YES' if target_met else 'NO'}")

        report.append("=" * 60)

        return "\n".join(report)


# ============================================================================
# Scene Generators for Testing
# ============================================================================


def create_empty_room_scene(resolution: int) -> VoxelGrid:
    """Create empty room scene for testing.

    A room with walls but empty interior - high compression expected.

    Args:
        resolution: Grid resolution

    Returns:
        VoxelGrid with empty room
    """
    from engine.core.math.geometry import AABB
    from engine.core.math.vec import Vec3, Vec4

    bounds = AABB(Vec3(0, 0, 0), Vec3(float(resolution), float(resolution), float(resolution)))
    grid = VoxelGrid(resolution, bounds)

    wall_thickness = max(1, resolution // 32)

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                # Check if on boundary
                on_wall = (
                    x < wall_thickness or x >= resolution - wall_thickness or
                    y < wall_thickness or y >= resolution - wall_thickness or
                    z < wall_thickness or z >= resolution - wall_thickness
                )
                if on_wall:
                    voxel = grid.get_voxel(x, y, z)
                    voxel.accumulate(
                        Vec4(0.7, 0.7, 0.7, 1.0),  # Gray wall
                        Vec3(0, 0, 0),
                        Vec3(0, 1, 0),
                    )

    grid.finalize()
    return grid


def create_furnished_room_scene(resolution: int) -> VoxelGrid:
    """Create furnished room scene for testing.

    A room with furniture - medium compression expected.

    Args:
        resolution: Grid resolution

    Returns:
        VoxelGrid with furnished room
    """
    from engine.core.math.geometry import AABB
    from engine.core.math.vec import Vec3, Vec4

    bounds = AABB(Vec3(0, 0, 0), Vec3(float(resolution), float(resolution), float(resolution)))
    grid = VoxelGrid(resolution, bounds)

    # Scale factor
    scale = resolution / 64.0

    # Walls
    wall_t = int(max(1, 2 * scale))
    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                on_wall = (
                    x < wall_t or x >= resolution - wall_t or
                    y < wall_t or  # Floor
                    z < wall_t or z >= resolution - wall_t
                )
                if on_wall:
                    voxel = grid.get_voxel(x, y, z)
                    voxel.accumulate(Vec4(0.6, 0.6, 0.65, 1.0), Vec3(0, 0, 0), Vec3(0, 1, 0))

    # Table (centered box)
    table_x = int(resolution * 0.3)
    table_z = int(resolution * 0.3)
    table_w = int(20 * scale)
    table_d = int(12 * scale)
    table_h = int(8 * scale)
    table_y = wall_t

    for z in range(table_z, table_z + table_d):
        for y in range(table_y, table_y + table_h):
            for x in range(table_x, table_x + table_w):
                if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                    voxel = grid.get_voxel(x, y, z)
                    voxel.accumulate(Vec4(0.5, 0.3, 0.1, 1.0), Vec3(0, 0, 0), Vec3(0, 1, 0))

    # Chair (smaller box)
    chair_x = int(resolution * 0.5)
    chair_z = int(resolution * 0.5)
    chair_w = int(6 * scale)
    chair_d = int(6 * scale)
    chair_h = int(10 * scale)

    for z in range(chair_z, chair_z + chair_d):
        for y in range(table_y, table_y + chair_h):
            for x in range(chair_x, chair_x + chair_w):
                if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                    voxel = grid.get_voxel(x, y, z)
                    voxel.accumulate(Vec4(0.4, 0.2, 0.1, 1.0), Vec3(0, 0, 0), Vec3(0, 1, 0))

    grid.finalize()
    return grid


def create_forest_scene(resolution: int) -> VoxelGrid:
    """Create forest scene for testing.

    Sparse trees with lots of empty space - high compression expected.

    Args:
        resolution: Grid resolution

    Returns:
        VoxelGrid with forest
    """
    import random
    from engine.core.math.geometry import AABB
    from engine.core.math.vec import Vec3, Vec4

    bounds = AABB(Vec3(0, 0, 0), Vec3(float(resolution), float(resolution), float(resolution)))
    grid = VoxelGrid(resolution, bounds)

    random.seed(42)  # Reproducible

    # Ground
    ground_h = max(1, resolution // 32)
    for z in range(resolution):
        for y in range(ground_h):
            for x in range(resolution):
                voxel = grid.get_voxel(x, y, z)
                voxel.accumulate(Vec4(0.2, 0.4, 0.1, 1.0), Vec3(0, 0, 0), Vec3(0, 1, 0))

    # Trees (sparse cylinders with spherical tops)
    num_trees = max(5, resolution // 16)
    tree_radius = max(2, resolution // 32)
    tree_height = max(10, resolution // 4)

    for _ in range(num_trees):
        tx = random.randint(tree_radius * 2, resolution - tree_radius * 2)
        tz = random.randint(tree_radius * 2, resolution - tree_radius * 2)

        # Trunk
        trunk_r = max(1, tree_radius // 3)
        for y in range(ground_h, ground_h + tree_height):
            for dz in range(-trunk_r, trunk_r + 1):
                for dx in range(-trunk_r, trunk_r + 1):
                    if dx * dx + dz * dz <= trunk_r * trunk_r:
                        x, z = tx + dx, tz + dz
                        if 0 <= x < resolution and 0 <= z < resolution and 0 <= y < resolution:
                            voxel = grid.get_voxel(x, y, z)
                            voxel.accumulate(Vec4(0.4, 0.25, 0.1, 1.0), Vec3(0, 0, 0), Vec3(0, 1, 0))

        # Foliage (sphere)
        foliage_y = ground_h + tree_height
        foliage_r = tree_radius
        for dz in range(-foliage_r, foliage_r + 1):
            for dy in range(-foliage_r // 2, foliage_r + 1):
                for dx in range(-foliage_r, foliage_r + 1):
                    if dx * dx + dy * dy + dz * dz <= foliage_r * foliage_r:
                        x = tx + dx
                        y = foliage_y + dy
                        z = tz + dz
                        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                            voxel = grid.get_voxel(x, y, z)
                            g = 0.3 + random.random() * 0.3
                            voxel.accumulate(Vec4(0.1, g, 0.05, 0.9), Vec3(0, 0, 0), Vec3(0, 1, 0))

    grid.finalize()
    return grid


# ============================================================================
# WGSL Shader Generation
# ============================================================================


def generate_svo_traversal_wgsl() -> str:
    """Generate WGSL shader for SVO traversal.

    Returns:
        WGSL compute shader code
    """
    return '''// SVO Traversal Compute Shader
// Based on Laine & Karras 2010, "Efficient Sparse Voxel Octrees"

// Node structure (32 bytes, matches SVOSerializer format)
struct SVONode {
    child_mask: u32,     // Bits 0-7: child presence
    first_child: u32,    // Index of first child
    radiance: vec3<f32>, // RGB radiance
    opacity: f32,        // Opacity
    normal: vec3<f32>,   // Normal direction
    _pad: f32,
}

// Traversal uniforms
struct TraversalUniforms {
    root_index: u32,
    max_level: u32,
    voxel_size: f32,
    resolution: f32,
}

// Ray structure
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

// Hit result
struct HitResult {
    hit: u32,
    t_near: f32,
    t_far: f32,
    node_index: u32,
    radiance: vec3<f32>,
    opacity: f32,
}

// Bindings
@group(0) @binding(0) var<storage, read> nodes: array<SVONode>;
@group(0) @binding(1) var<uniform> uniforms: TraversalUniforms;
@group(0) @binding(2) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(3) var<storage, read_write> results: array<HitResult>;

// Constants
const OCTREE_CHILDREN: u32 = 8u;
const STACK_SIZE: u32 = 24u;  // Max depth

// Child offset from index
fn child_offset(index: u32) -> vec3<u32> {
    return vec3<u32>(
        index & 1u,
        (index >> 1u) & 1u,
        (index >> 2u) & 1u
    );
}

// Ray-AABB intersection
fn ray_box_intersect(
    ray_origin: vec3<f32>,
    ray_dir_inv: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>
) -> vec2<f32> {
    let t1 = (box_min - ray_origin) * ray_dir_inv;
    let t2 = (box_max - ray_origin) * ray_dir_inv;
    let t_min = min(t1, t2);
    let t_max = max(t1, t2);
    return vec2<f32>(
        max(max(t_min.x, t_min.y), t_min.z),
        min(min(t_max.x, t_max.y), t_max.z)
    );
}

// Stack entry for iterative traversal
struct StackEntry {
    node_index: u32,
    t_near: f32,
    t_far: f32,
    box_min: vec3<f32>,
    box_max: vec3<f32>,
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_index = gid.x;
    if ray_index >= arrayLength(&rays) {
        return;
    }

    let ray = rays[ray_index];

    // Initialize result
    var result: HitResult;
    result.hit = 0u;
    result.t_near = 1e30;
    result.t_far = 0.0;
    result.node_index = 0u;
    result.radiance = vec3<f32>(0.0);
    result.opacity = 0.0;

    // Compute ray direction inverse (for AABB test)
    let dir_inv = 1.0 / ray.direction;

    // Root bounds
    let root_min = vec3<f32>(0.0);
    let root_max = vec3<f32>(uniforms.resolution);

    // Check root intersection
    let root_t = ray_box_intersect(ray.origin, dir_inv, root_min, root_max);
    if root_t.x >= root_t.y || root_t.y < 0.0 {
        results[ray_index] = result;
        return;
    }

    // Iterative traversal with stack
    var stack: array<StackEntry, STACK_SIZE>;
    var stack_ptr: u32 = 0u;

    // Push root
    stack[0].node_index = uniforms.root_index;
    stack[0].t_near = max(0.0, root_t.x);
    stack[0].t_far = root_t.y;
    stack[0].box_min = root_min;
    stack[0].box_max = root_max;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let entry = stack[stack_ptr];

        // Skip if behind current hit
        if entry.t_near > result.t_near {
            continue;
        }

        let node = nodes[entry.node_index];

        // Check if leaf (no children)
        if node.child_mask == 0u {
            // Leaf node - check for hit
            if node.opacity > 0.001 {
                result.hit = 1u;
                result.t_near = entry.t_near;
                result.t_far = entry.t_far;
                result.node_index = entry.node_index;
                result.radiance = node.radiance;
                result.opacity = node.opacity;
            }
            continue;
        }

        // Branch node - push children
        let half_size = (entry.box_max - entry.box_min) * 0.5;

        for (var i = 0u; i < OCTREE_CHILDREN; i++) {
            if (node.child_mask & (1u << i)) == 0u {
                continue;
            }

            let offset = child_offset(i);
            let child_min = entry.box_min + vec3<f32>(offset) * half_size;
            let child_max = child_min + half_size;

            let child_t = ray_box_intersect(ray.origin, dir_inv, child_min, child_max);

            if child_t.x < child_t.y && child_t.y > 0.0 {
                let clamped_near = max(entry.t_near, child_t.x);
                let clamped_far = min(entry.t_far, child_t.y);

                if clamped_near < result.t_near && stack_ptr < STACK_SIZE {
                    // Count children before this one to find child index
                    var child_index = node.first_child;
                    for (var j = 0u; j < i; j++) {
                        if (node.child_mask & (1u << j)) != 0u {
                            child_index += 1u;
                        }
                    }

                    stack[stack_ptr].node_index = child_index;
                    stack[stack_ptr].t_near = clamped_near;
                    stack[stack_ptr].t_far = clamped_far;
                    stack[stack_ptr].box_min = child_min;
                    stack[stack_ptr].box_max = child_max;
                    stack_ptr += 1u;
                }
            }
        }
    }

    results[ray_index] = result;
}
'''


def generate_svo_cone_trace_wgsl() -> str:
    """Generate WGSL shader for SVO cone tracing.

    Returns:
        WGSL compute shader code
    """
    return '''// SVO Cone Tracing Compute Shader
// Based on Crassin 2011, "Interactive Indirect Illumination Using Voxel Cone Tracing"

struct SVONode {
    child_mask: u32,
    first_child: u32,
    radiance: vec3<f32>,
    opacity: f32,
    normal: vec3<f32>,
    _pad: f32,
}

struct ConeTraceUniforms {
    root_index: u32,
    max_level: u32,
    voxel_size: f32,
    resolution: f32,
    max_distance: f32,
    max_steps: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Cone {
    origin: vec3<f32>,
    direction: vec3<f32>,
    aperture: f32,  // Half-angle tangent
}

struct ConeResult {
    radiance: vec3<f32>,
    opacity: f32,
    steps: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<SVONode>;
@group(0) @binding(1) var<uniform> uniforms: ConeTraceUniforms;
@group(0) @binding(2) var<storage, read> cones: array<Cone>;
@group(0) @binding(3) var<storage, read_write> results: array<ConeResult>;

// Sample SVO at position with LOD
fn sample_svo(pos: vec3<f32>, lod: f32) -> vec4<f32> {
    // Find node at appropriate level
    let target_level = u32(uniforms.max_level) - u32(clamp(lod, 0.0, f32(uniforms.max_level)));

    var node_index = uniforms.root_index;
    var box_min = vec3<f32>(0.0);
    var box_max = vec3<f32>(uniforms.resolution);
    var level = 0u;

    while level < target_level {
        let node = nodes[node_index];

        if node.child_mask == 0u {
            // Leaf - return data
            return vec4<f32>(node.radiance, node.opacity);
        }

        // Find child containing position
        let half_size = (box_max - box_min) * 0.5;
        let center = box_min + half_size;
        let offset = vec3<u32>(
            select(0u, 1u, pos.x >= center.x),
            select(0u, 1u, pos.y >= center.y),
            select(0u, 1u, pos.z >= center.z)
        );
        let child_idx = offset.x | (offset.y << 1u) | (offset.z << 2u);

        if (node.child_mask & (1u << child_idx)) == 0u {
            // Child doesn't exist - return current node data
            return vec4<f32>(node.radiance, node.opacity);
        }

        // Compute child node index
        var child_node_index = node.first_child;
        for (var i = 0u; i < child_idx; i++) {
            if (node.child_mask & (1u << i)) != 0u {
                child_node_index += 1u;
            }
        }

        // Update bounds
        box_min = box_min + vec3<f32>(offset) * half_size;
        box_max = box_min + half_size;
        node_index = child_node_index;
        level += 1u;
    }

    let node = nodes[node_index];
    return vec4<f32>(node.radiance, node.opacity);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cone_index = gid.x;
    if cone_index >= arrayLength(&cones) {
        return;
    }

    let cone = cones[cone_index];

    // Initialize result
    var result: ConeResult;
    result.radiance = vec3<f32>(0.0);
    result.opacity = 0.0;
    result.steps = 0u;

    // Trace cone
    var t = 1.0;  // Start offset
    let res = uniforms.resolution;

    while t < uniforms.max_distance && result.opacity < 0.99 && result.steps < uniforms.max_steps {
        // Current position
        let pos = cone.origin + cone.direction * t;

        // Bounds check
        if any(pos < vec3<f32>(0.0)) || any(pos >= vec3<f32>(res)) {
            break;
        }

        // Cone diameter at current distance
        let diameter = 2.0 * t * cone.aperture;

        // Select LOD based on diameter
        let lod = log2(max(1.0, diameter / uniforms.voxel_size));

        // Sample
        let sample = sample_svo(pos, lod);

        if sample.a > 0.001 {
            // Front-to-back compositing
            let alpha = sample.a * (1.0 - result.opacity);
            result.radiance += sample.rgb * alpha;
            result.opacity += alpha;
        }

        // Adaptive step size
        let step_size = max(1.0, diameter * 0.5);
        t += step_size;
        result.steps += 1u;
    }

    results[cone_index] = result;
}
'''


# ============================================================================
# Utility Functions
# ============================================================================


def build_svo_from_grid(
    grid: VoxelGrid,
    compress: bool = True,
) -> tuple[SVONode, SVOBuildStats, Optional[SVOCompressionStats]]:
    """Convenience function to build SVO from dense grid.

    Args:
        grid: Dense VoxelGrid
        compress: Whether to compress after building

    Returns:
        Tuple of (root_node, build_stats, compression_stats)
    """
    builder = SVOBuilder()
    root = builder.build_from_dense(grid)
    build_stats = builder.get_stats()

    compression_stats = None
    if compress:
        compressor = SVOCompressor()
        root = compressor.compress(root)
        compression_stats = compressor.get_stats()

        # Generate mip data
        mip_gen = SVOMipGenerator()
        mip_gen.generate_mips(root)

    return (root, build_stats, compression_stats)


def evaluate_svo_compression(
    resolution: int = 256,
) -> dict[str, Any]:
    """Evaluate SVO compression on test scenes.

    Args:
        resolution: Grid resolution to test

    Returns:
        Evaluation results dictionary
    """
    profiler = MemoryProfiler()
    scenes = []

    # Test empty room
    start = time.perf_counter()
    empty_grid = create_empty_room_scene(resolution)
    empty_root, empty_build, _ = build_svo_from_grid(empty_grid)
    empty_time = (time.perf_counter() - start) * 1000
    empty_profile = profiler.profile(empty_root, resolution, "Empty Room", "Walls only, empty interior")
    scenes.append(SceneProfile("Empty Room", "Walls only", empty_profile, empty_time))

    # Test furnished room
    start = time.perf_counter()
    furnished_grid = create_furnished_room_scene(resolution)
    furnished_root, furnished_build, _ = build_svo_from_grid(furnished_grid)
    furnished_time = (time.perf_counter() - start) * 1000
    furnished_profile = profiler.profile(furnished_root, resolution, "Furnished Room", "Room with furniture")
    scenes.append(SceneProfile("Furnished Room", "Room with furniture", furnished_profile, furnished_time))

    # Test forest
    start = time.perf_counter()
    forest_grid = create_forest_scene(resolution)
    forest_root, forest_build, _ = build_svo_from_grid(forest_grid)
    forest_time = (time.perf_counter() - start) * 1000
    forest_profile = profiler.profile(forest_root, resolution, "Forest", "Sparse trees")
    scenes.append(SceneProfile("Forest", "Sparse trees", forest_profile, forest_time))

    # Generate report
    report = profiler.generate_report(scenes)

    # Summary
    avg_savings = sum(s.memory.savings_ratio for s in scenes) / len(scenes)
    all_targets_met = all(s.memory.savings_ratio >= 5.0 for s in scenes)

    return {
        "resolution": resolution,
        "scenes": scenes,
        "report": report,
        "average_savings": avg_savings,
        "targets_met": all_targets_met,
        "recommendation": "GO" if all_targets_met else "NO-GO",
    }
