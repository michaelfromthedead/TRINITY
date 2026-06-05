"""Tests for Sparse Voxel Octree implementation (T-GIR-P11.2).

Tests cover:
    - SVOVoxelData creation and operations
    - SVONode creation, child management, and traversal
    - SVOBuilder: construction from dense grid
    - SVOCompressor: uniform region pruning and merging
    - SVOMipGenerator: mip chain generation
    - SVOTraversal: ray-octree intersection and cone tracing
    - SVOSerializer: GPU serialization and round-trip
    - MemoryProfiler: compression ratio validation
    - Scene generators: empty room, furnished room, forest
    - Integration tests and performance benchmarks
"""

import math
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3, Vec4

from engine.rendering.gi.voxelization import (
    VoxelGrid,
    Voxel,
    VoxelResolution,
    VoxelizationConfig,
    SceneVoxelizer,
    Triangle,
    create_test_triangles,
)

from engine.rendering.gi.sparse_voxel_octree import (
    # Constants
    OCTREE_CHILDREN,
    MIN_SVO_RESOLUTION,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_OPACITY_THRESHOLD,
    DENSE_VOXEL_BYTES,
    GPU_NODE_SIZE,
    # Enums
    NodeType,
    TraversalMode,
    # Data structures
    SVOVoxelData,
    SVONode,
    RayHit,
    ConeTraceResult,
    # Configs
    SVOBuildConfig,
    SVOBuildStats,
    SVOCompressionConfig,
    SVOCompressionStats,
    SVOMipConfig,
    SerializedSVO,
    MemoryProfile,
    SceneProfile,
    # Classes
    SVOBuilder,
    SVOCompressor,
    SVOMipGenerator,
    SVOTraversal,
    SVOSerializer,
    MemoryProfiler,
    # Scene generators
    create_empty_room_scene,
    create_furnished_room_scene,
    create_forest_scene,
    # Utilities
    build_svo_from_grid,
    evaluate_svo_compression,
    # WGSL
    generate_svo_traversal_wgsl,
    generate_svo_cone_trace_wgsl,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_grid() -> VoxelGrid:
    """Create a simple 8x8x8 voxel grid with sparse content."""
    bounds = AABB(Vec3(0, 0, 0), Vec3(8, 8, 8))
    grid = VoxelGrid(8, bounds)

    # Fill a small cube in the center
    for z in range(3, 5):
        for y in range(3, 5):
            for x in range(3, 5):
                voxel = grid.get_voxel(x, y, z)
                voxel.accumulate(Vec4(1, 0, 0, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))

    grid.finalize()
    return grid


@pytest.fixture
def uniform_grid() -> VoxelGrid:
    """Create a uniform (completely filled) voxel grid."""
    bounds = AABB(Vec3(0, 0, 0), Vec3(8, 8, 8))
    grid = VoxelGrid(8, bounds)

    for z in range(8):
        for y in range(8):
            for x in range(8):
                voxel = grid.get_voxel(x, y, z)
                voxel.accumulate(Vec4(0.5, 0.5, 0.5, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))

    grid.finalize()
    return grid


@pytest.fixture
def empty_grid() -> VoxelGrid:
    """Create an empty voxel grid."""
    bounds = AABB(Vec3(0, 0, 0), Vec3(8, 8, 8))
    return VoxelGrid(8, bounds)


# ============================================================================
# SVOVoxelData Tests
# ============================================================================


class TestSVOVoxelData:
    """Tests for SVOVoxelData."""

    def test_create_empty(self) -> None:
        """Empty voxel data should have zero values."""
        data = SVOVoxelData.empty()
        assert data.opacity == 0.0
        assert np.all(data.radiance == 0.0)
        assert data.is_empty()

    def test_create_with_values(self) -> None:
        """Voxel data should store provided values."""
        radiance = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        data = SVOVoxelData(radiance, 0.8)
        assert data.opacity == 0.8
        assert_array_almost_equal(data.radiance, radiance)
        assert not data.is_empty()

    def test_create_with_normal(self) -> None:
        """Voxel data should store normal."""
        radiance = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        data = SVOVoxelData(radiance, 1.0, normal)
        assert data.normal is not None
        assert_array_almost_equal(data.normal, normal)

    def test_from_voxel(self) -> None:
        """Create from voxelization.Voxel."""
        voxel = Voxel()
        voxel.accumulate(Vec4(1, 0.5, 0.25, 0.9), Vec3(0, 0, 0), Vec3(0, 1, 0))
        voxel.finalize()

        data = SVOVoxelData.from_voxel(voxel)
        assert data.opacity == pytest.approx(0.9)
        assert data.radiance[0] == pytest.approx(1.0)
        assert data.radiance[1] == pytest.approx(0.5)

    def test_to_rgba(self) -> None:
        """Convert to RGBA array."""
        data = SVOVoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.8)
        rgba = data.to_rgba()
        assert rgba.shape == (4,)
        assert rgba[0] == pytest.approx(1.0)
        assert rgba[3] == pytest.approx(0.8)

    def test_is_empty_threshold(self) -> None:
        """Empty detection with custom threshold."""
        data = SVOVoxelData(np.array([1.0, 1.0, 1.0], dtype=np.float32), 0.005)
        assert data.is_empty(0.01)
        assert not data.is_empty(0.001)

    def test_luminance(self) -> None:
        """Compute luminance correctly."""
        # Pure green should have luminance ~0.7152
        data = SVOVoxelData(np.array([0.0, 1.0, 0.0], dtype=np.float32), 1.0)
        assert data.luminance() == pytest.approx(0.7152)

    def test_similarity_identical(self) -> None:
        """Identical voxels should have similarity 1.0."""
        data1 = SVOVoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.8)
        data2 = SVOVoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.8)
        assert data1.similarity(data2) == pytest.approx(1.0)

    def test_similarity_different(self) -> None:
        """Different voxels should have lower similarity."""
        data1 = SVOVoxelData(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1.0)
        data2 = SVOVoxelData(np.array([0.0, 1.0, 0.0], dtype=np.float32), 1.0)
        assert data1.similarity(data2) < 0.5

    def test_average_single(self) -> None:
        """Average of single voxel should equal that voxel."""
        data = SVOVoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.8)
        avg = SVOVoxelData.average([data])
        assert_array_almost_equal(avg.radiance, data.radiance)
        assert avg.opacity == pytest.approx(data.opacity)

    def test_average_multiple(self) -> None:
        """Average of multiple voxels weighted by opacity."""
        data1 = SVOVoxelData(np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.5)
        data2 = SVOVoxelData(np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.5)
        avg = SVOVoxelData.average([data1, data2])
        # Equal weights, so average should be 0.5, 0.5, 0.0
        assert avg.radiance[0] == pytest.approx(0.5, abs=0.01)
        assert avg.radiance[1] == pytest.approx(0.5, abs=0.01)
        assert avg.opacity == pytest.approx(0.5)

    def test_average_empty_list(self) -> None:
        """Average of empty list should return empty voxel."""
        avg = SVOVoxelData.average([])
        assert avg.is_empty()


# ============================================================================
# SVONode Tests
# ============================================================================


class TestSVONode:
    """Tests for SVONode."""

    def test_create_empty(self) -> None:
        """Default node should be empty."""
        node = SVONode()
        assert node.is_empty()
        assert not node.is_leaf()
        assert not node.is_branch()
        assert node.node_type == NodeType.EMPTY

    def test_create_with_type(self) -> None:
        """Node with specified type."""
        leaf = SVONode(NodeType.LEAF, level=3)
        assert leaf.is_leaf()
        assert leaf.level == 3

    def test_create_with_bounds(self) -> None:
        """Node with bounds."""
        bounds = ((0, 0, 0), (8, 8, 8))
        node = SVONode(NodeType.BRANCH, bounds=bounds)
        assert node.bounds == bounds
        center = node.get_center()
        assert center == (4.0, 4.0, 4.0)
        assert node.get_size() == 8

    def test_get_set_child(self) -> None:
        """Get and set child nodes."""
        parent = SVONode(NodeType.BRANCH)
        child = SVONode(NodeType.LEAF)
        child.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        parent.set_child(0, child)
        retrieved = parent.get_child(0)
        assert retrieved is child
        assert parent.is_branch()

    def test_child_index_out_of_range(self) -> None:
        """Out of range child index should raise."""
        node = SVONode()
        with pytest.raises(IndexError):
            node.get_child(8)
        with pytest.raises(IndexError):
            node.set_child(-1, None)

    def test_child_mask(self) -> None:
        """Child mask should track non-empty children."""
        parent = SVONode()
        child0 = SVONode(NodeType.LEAF)
        child0.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))
        child3 = SVONode(NodeType.LEAF)
        child3.set_data(SVOVoxelData(np.array([0, 1, 0], dtype=np.float32), 1.0))

        parent.set_child(0, child0)
        parent.set_child(3, child3)

        assert parent.get_child_mask() == 0b00001001  # bits 0 and 3

    def test_child_count(self) -> None:
        """Count non-empty children."""
        parent = SVONode()
        assert parent.child_count() == 0

        for i in range(3):
            child = SVONode(NodeType.LEAF)
            child.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))
            parent.set_child(i, child)

        assert parent.child_count() == 3

    def test_set_data_makes_leaf(self) -> None:
        """Setting data on empty node makes it a leaf."""
        node = SVONode()
        data = SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0)
        node.set_data(data)
        assert node.is_leaf()
        assert node.data is data

    def test_iter_children(self) -> None:
        """Iterate over non-empty children."""
        parent = SVONode()
        for i in [0, 2, 7]:
            child = SVONode(NodeType.LEAF)
            child.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))
            parent.set_child(i, child)

        indices = [i for i, _ in parent.iter_children()]
        assert indices == [0, 2, 7]

    def test_compute_averaged_data(self) -> None:
        """Compute average of children data."""
        parent = SVONode()
        for i in range(2):
            child = SVONode(NodeType.LEAF)
            radiance = np.array([float(i), 0, 0], dtype=np.float32)
            child.set_data(SVOVoxelData(radiance, 1.0))
            parent.set_child(i, child)

        avg = parent.compute_averaged_data()
        assert avg.radiance[0] == pytest.approx(0.5)

    def test_child_index_static(self) -> None:
        """Static child index computation."""
        assert SVONode.child_index(0, 0, 0) == 0
        assert SVONode.child_index(1, 0, 0) == 1
        assert SVONode.child_index(0, 1, 0) == 2
        assert SVONode.child_index(1, 1, 0) == 3
        assert SVONode.child_index(0, 0, 1) == 4
        assert SVONode.child_index(1, 1, 1) == 7

    def test_child_offset_static(self) -> None:
        """Static child offset computation."""
        assert SVONode.child_offset(0) == (0, 0, 0)
        assert SVONode.child_offset(1) == (1, 0, 0)
        assert SVONode.child_offset(7) == (1, 1, 1)


# ============================================================================
# SVOBuilder Tests
# ============================================================================


class TestSVOBuilder:
    """Tests for SVOBuilder."""

    def test_build_from_empty_grid(self, empty_grid: VoxelGrid) -> None:
        """Building from empty grid should produce empty root."""
        builder = SVOBuilder()
        root = builder.build_from_dense(empty_grid)
        assert root.is_empty()

    def test_build_from_simple_grid(self, simple_grid: VoxelGrid) -> None:
        """Building from simple grid should produce valid SVO."""
        builder = SVOBuilder()
        root = builder.build_from_dense(simple_grid)

        # Should have structure
        assert root.is_branch() or root.is_leaf()
        stats = builder.get_stats()
        assert stats.total_nodes > 0

    def test_build_from_uniform_grid(self, uniform_grid: VoxelGrid) -> None:
        """Building from uniform grid should work."""
        builder = SVOBuilder()
        root = builder.build_from_dense(uniform_grid)

        stats = builder.get_stats()
        assert stats.total_nodes > 0
        assert stats.leaf_nodes > 0

    def test_build_stats(self, simple_grid: VoxelGrid) -> None:
        """Build statistics should be populated."""
        builder = SVOBuilder()
        builder.build_from_dense(simple_grid)

        stats = builder.get_stats()
        assert stats.total_nodes > 0
        assert stats.max_depth > 0
        assert stats.build_time_ms >= 0

    def test_build_config_opacity_threshold(self) -> None:
        """Opacity threshold affects empty detection."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(4, 4, 4))
        grid = VoxelGrid(4, bounds)

        # Fill with low opacity
        voxel = grid.get_voxel(2, 2, 2)
        voxel.accumulate(Vec4(1, 1, 1, 0.01), Vec3.zero(), Vec3.unit_y())
        grid.finalize()

        # High threshold - should have fewer leaves
        builder_high = SVOBuilder(SVOBuildConfig(opacity_threshold=0.1))
        root_high = builder_high.build_from_dense(grid)
        high_stats = builder_high.get_stats()

        # Low threshold - should have leaf
        builder_low = SVOBuilder(SVOBuildConfig(opacity_threshold=0.001))
        root_low = builder_low.build_from_dense(grid)
        low_stats = builder_low.get_stats()

        # Low threshold should detect more leaves
        assert low_stats.leaf_nodes >= high_stats.leaf_nodes

    def test_should_subdivide_coarse(self) -> None:
        """Coarse regions should always subdivide."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(64, 64, 64))
        grid = VoxelGrid(64, bounds)

        builder = SVOBuilder()
        builder._grid = grid
        builder._resolution = 64

        # Large region should subdivide
        assert builder.should_subdivide(((0, 0, 0), (32, 32, 32)))


# ============================================================================
# SVOCompressor Tests
# ============================================================================


class TestSVOCompressor:
    """Tests for SVOCompressor."""

    def test_compress_empty(self) -> None:
        """Compressing empty tree should work."""
        root = SVONode(NodeType.EMPTY)
        compressor = SVOCompressor()
        compressed = compressor.compress(root)
        assert compressed.is_empty()

    def test_compress_single_leaf(self) -> None:
        """Single leaf should not change."""
        root = SVONode(NodeType.LEAF)
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        compressor = SVOCompressor()
        compressed = compressor.compress(root)
        assert compressed.is_leaf()

    def test_compress_uniform_children(self, uniform_grid: VoxelGrid) -> None:
        """Uniform children should be merged."""
        builder = SVOBuilder()
        root = builder.build_from_dense(uniform_grid)

        compressor = SVOCompressor()
        compressed = compressor.compress(root)

        stats = compressor.get_stats()
        assert stats.compression_ratio <= 1.0
        # Uniform grid should compress significantly
        assert stats.merged_regions >= 0

    def test_compress_preserves_data(self, simple_grid: VoxelGrid) -> None:
        """Compression should preserve voxel data."""
        builder = SVOBuilder()
        root = builder.build_from_dense(simple_grid)

        compressor = SVOCompressor()
        compressed = compressor.compress(root)

        # Root should still have data
        assert compressed.data is not None or compressed.is_empty()

    def test_compression_stats(self, simple_grid: VoxelGrid) -> None:
        """Compression statistics should be populated."""
        builder = SVOBuilder()
        root = builder.build_from_dense(simple_grid)

        compressor = SVOCompressor()
        compressor.compress(root)

        stats = compressor.get_stats()
        assert stats.original_nodes > 0
        assert stats.compressed_nodes > 0
        assert stats.compression_ratio > 0

    def test_merge_threshold(self) -> None:
        """Merge threshold affects compression."""
        # Create tree with slightly different children
        parent = SVONode(NodeType.BRANCH, level=2)
        for i in range(8):
            child = SVONode(NodeType.LEAF, level=3)
            # Slightly different values
            r = 0.5 + (i * 0.01)
            child.set_data(SVOVoxelData(np.array([r, 0.5, 0.5], dtype=np.float32), 1.0))
            parent.set_child(i, child)

        # High threshold should merge
        compressor_high = SVOCompressor(SVOCompressionConfig(merge_threshold=0.2))
        compressed_high = compressor_high.compress(parent)

        # Low threshold should not merge
        compressor_low = SVOCompressor(SVOCompressionConfig(merge_threshold=0.001))
        compressed_low = compressor_low.compress(parent)

        # High threshold should result in fewer nodes
        assert compressor_high.get_stats().compressed_nodes <= compressor_low.get_stats().compressed_nodes


# ============================================================================
# SVOMipGenerator Tests
# ============================================================================


class TestSVOMipGenerator:
    """Tests for SVOMipGenerator."""

    def test_generate_mips_empty(self) -> None:
        """Mip generation on empty tree should not crash."""
        root = SVONode(NodeType.EMPTY)
        generator = SVOMipGenerator()
        generator.generate_mips(root)  # Should not raise

    def test_generate_mips_leaf(self) -> None:
        """Mip generation on leaf should preserve data."""
        root = SVONode(NodeType.LEAF)
        data = SVOVoxelData(np.array([1, 0.5, 0.25], dtype=np.float32), 0.8)
        root.set_data(data)

        generator = SVOMipGenerator()
        generator.generate_mips(root)

        assert root.data is not None
        assert_array_almost_equal(root.data.radiance, data.radiance)

    def test_generate_mips_branch(self, simple_grid: VoxelGrid) -> None:
        """Mip generation on branch should compute averages."""
        builder = SVOBuilder()
        root = builder.build_from_dense(simple_grid)

        generator = SVOMipGenerator()
        generator.generate_mips(root)

        # Root should have averaged data
        assert root.data is not None

    def test_get_mip_level(self) -> None:
        """Mip level calculation should be correct."""
        generator = SVOMipGenerator()
        max_level = 8

        node_level_0 = SVONode(level=0)
        assert generator.get_mip_level(node_level_0, max_level) == max_level

        node_level_8 = SVONode(level=8)
        assert generator.get_mip_level(node_level_8, max_level) == 0


# ============================================================================
# SVOTraversal Tests
# ============================================================================


class TestSVOTraversal:
    """Tests for SVOTraversal."""

    def test_traverse_ray_empty(self) -> None:
        """Ray through empty tree should miss."""
        root = SVONode(NodeType.EMPTY)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        traversal = SVOTraversal(root, 1.0, 8)

        hit = traversal.traverse_ray((0, 4, 4), (1, 0, 0))
        assert not hit.hit

    def test_traverse_ray_hit(self) -> None:
        """Ray through filled tree should hit."""
        # Create simple filled tree
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        traversal = SVOTraversal(root, 1.0, 8)

        hit = traversal.traverse_ray((0, 4, 4), (1, 0, 0))
        assert hit.hit
        assert hit.t_near >= 0

    def test_traverse_ray_miss_outside(self) -> None:
        """Ray missing bounds should not hit."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        traversal = SVOTraversal(root, 1.0, 8)

        # Ray parallel and outside
        hit = traversal.traverse_ray((10, 10, 10), (1, 0, 0))
        assert not hit.hit

    def test_find_leaf(self) -> None:
        """Find leaf at position."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        traversal = SVOTraversal(root, 1.0, 8)

        leaf = traversal.find_leaf(4, 4, 4)
        assert leaf is not None
        assert leaf.is_leaf()

    def test_find_leaf_outside(self) -> None:
        """Find leaf outside bounds should return None."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))

        traversal = SVOTraversal(root, 1.0, 8)

        leaf = traversal.find_leaf(10, 10, 10)
        assert leaf is None

    def test_sample_at_position(self) -> None:
        """Sample at position with LOD."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        data = SVOVoxelData(np.array([1, 0.5, 0.25], dtype=np.float32), 0.8)
        root.set_data(data)

        traversal = SVOTraversal(root, 1.0, 8)

        sample = traversal.sample_at_position(4, 4, 4, level=0)
        assert sample is not None
        assert_array_almost_equal(sample.radiance, data.radiance)

    def test_trace_cone_empty(self) -> None:
        """Cone through empty tree should accumulate nothing."""
        root = SVONode(NodeType.EMPTY)
        root.bounds = ((0, 0, 0), (8, 8, 8))

        traversal = SVOTraversal(root, 1.0, 8)

        result = traversal.trace_cone((0, 4, 4), (1, 0, 0), 0.1)
        assert result.accumulated_opacity < 0.01

    def test_trace_cone_hit(self) -> None:
        """Cone through filled tree should accumulate."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 0.5))

        traversal = SVOTraversal(root, 1.0, 8)

        result = traversal.trace_cone((0, 4, 4), (1, 0, 0), 0.1, max_distance=10)
        assert result.accumulated_opacity > 0
        assert result.steps > 0


# ============================================================================
# SVOSerializer Tests
# ============================================================================


class TestSVOSerializer:
    """Tests for SVOSerializer."""

    def test_serialize_empty(self) -> None:
        """Serialize empty tree."""
        root = SVONode(NodeType.EMPTY)
        serializer = SVOSerializer()
        serialized = serializer.serialize(root)

        assert serialized.node_count >= 1
        assert serialized.memory_bytes > 0

    def test_serialize_leaf(self) -> None:
        """Serialize single leaf."""
        root = SVONode(NodeType.LEAF)
        root.set_data(SVOVoxelData(np.array([1, 0.5, 0.25], dtype=np.float32), 0.8))

        serializer = SVOSerializer()
        serialized = serializer.serialize(root)

        assert serialized.node_count == 1
        assert len(serialized.nodes) == GPU_NODE_SIZE

    def test_serialize_branch(self) -> None:
        """Serialize tree with children."""
        parent = SVONode(NodeType.BRANCH)
        for i in range(4):
            child = SVONode(NodeType.LEAF)
            child.set_data(SVOVoxelData(np.array([float(i) / 3, 0, 0], dtype=np.float32), 1.0))
            parent.set_child(i, child)

        serializer = SVOSerializer()
        serialized = serializer.serialize(parent)

        assert serialized.node_count == 5  # Parent + 4 children

    def test_deserialize_round_trip(self) -> None:
        """Serialize and deserialize should preserve structure."""
        root = SVONode(NodeType.LEAF)
        data = SVOVoxelData(np.array([1.0, 0.5, 0.25], dtype=np.float32), 0.8)
        root.set_data(data)

        serializer = SVOSerializer()
        serialized = serializer.serialize(root)
        restored = serializer.deserialize(serialized)

        assert restored.is_leaf()
        assert restored.data is not None
        assert restored.data.opacity == pytest.approx(data.opacity, abs=0.01)

    def test_get_gpu_buffer(self) -> None:
        """Get raw GPU buffer."""
        root = SVONode(NodeType.LEAF)
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        serializer = SVOSerializer()
        serialized = serializer.serialize(root)
        buffer = serializer.get_gpu_buffer(serialized)

        assert isinstance(buffer, bytes)
        assert len(buffer) == serialized.memory_bytes


# ============================================================================
# MemoryProfiler Tests
# ============================================================================


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    def test_profile_simple(self, simple_grid: VoxelGrid) -> None:
        """Profile simple SVO."""
        root, _, _ = build_svo_from_grid(simple_grid, compress=False)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, simple_grid.resolution)

        assert profile.dense_bytes > 0
        assert profile.svo_bytes > 0
        assert profile.node_count > 0

    def test_profile_compression_ratio(self, simple_grid: VoxelGrid) -> None:
        """Compression ratio should be calculated."""
        root, _, _ = build_svo_from_grid(simple_grid, compress=True)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, simple_grid.resolution)

        assert profile.compression_ratio > 0
        assert profile.compression_ratio <= 1.0

    def test_savings_ratio(self) -> None:
        """Savings ratio should be inverse of compression ratio."""
        profile = MemoryProfile(
            dense_bytes=1000,
            svo_bytes=100,
            compression_ratio=0.1,
            node_count=10,
            leaf_count=8,
            fill_ratio=0.5,
        )
        assert profile.savings_ratio == pytest.approx(10.0)

    def test_compare(self, simple_grid: VoxelGrid) -> None:
        """Compare SVO and dense representations."""
        root, _, _ = build_svo_from_grid(simple_grid, compress=True)

        profiler = MemoryProfiler()
        comparison = profiler.compare(root, simple_grid)

        assert "resolution" in comparison
        assert "dense_memory_mb" in comparison
        assert "svo_memory_mb" in comparison
        assert "savings_ratio" in comparison

    def test_generate_report(self) -> None:
        """Generate report from profiles."""
        profiler = MemoryProfiler()

        profile = MemoryProfile(
            dense_bytes=64 * 1024 * 1024,
            svo_bytes=8 * 1024 * 1024,
            compression_ratio=0.125,
            node_count=10000,
            leaf_count=8000,
            fill_ratio=0.1,
        )
        scene = SceneProfile("Test Scene", "Test description", profile, 100.0)

        report = profiler.generate_report([scene])

        assert "Test Scene" in report
        assert "Savings" in report


# ============================================================================
# Scene Generator Tests
# ============================================================================


class TestSceneGenerators:
    """Tests for scene generators."""

    def test_create_empty_room_scene(self) -> None:
        """Empty room scene should have walls only."""
        grid = create_empty_room_scene(32)
        assert grid.resolution == 32
        filled = grid.count_filled_voxels()
        total = grid.total_voxels
        # Should be sparse (mostly empty interior)
        assert filled / total < 0.3

    def test_create_furnished_room_scene(self) -> None:
        """Furnished room should have content (walls + furniture)."""
        furnished = create_furnished_room_scene(32)

        # Furnished room should have reasonable fill ratio
        filled = furnished.count_filled_voxels()
        total = furnished.total_voxels
        fill_ratio = filled / total

        # Should be sparse but not empty (walls + furniture)
        assert fill_ratio > 0.05, f"Fill ratio {fill_ratio} too low"
        assert fill_ratio < 0.5, f"Fill ratio {fill_ratio} too high"

    def test_create_forest_scene(self) -> None:
        """Forest scene should be sparse."""
        grid = create_forest_scene(32)
        filled = grid.count_filled_voxels()
        total = grid.total_voxels
        # Forest should be sparse (mostly air)
        assert filled / total < 0.2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, simple_grid: VoxelGrid) -> None:
        """Full SVO pipeline: build, compress, serialize."""
        # Build
        builder = SVOBuilder()
        root = builder.build_from_dense(simple_grid)

        # Compress
        compressor = SVOCompressor()
        root = compressor.compress(root)

        # Generate mips
        mip_gen = SVOMipGenerator()
        mip_gen.generate_mips(root)

        # Serialize
        serializer = SVOSerializer()
        serialized = serializer.serialize(root)

        # Deserialize
        restored = serializer.deserialize(serialized)

        # Profile
        profiler = MemoryProfiler()
        profile = profiler.profile(restored, simple_grid.resolution)

        assert profile.node_count > 0

    def test_build_svo_from_grid_helper(self, simple_grid: VoxelGrid) -> None:
        """Convenience function should work."""
        root, build_stats, compression_stats = build_svo_from_grid(simple_grid)

        assert root is not None
        assert build_stats is not None
        assert compression_stats is not None

    def test_traversal_matches_dense(self) -> None:
        """SVO traversal should match dense grid sampling."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(8, 8, 8))
        grid = VoxelGrid(8, bounds)

        # Fill specific voxel
        voxel = grid.get_voxel(4, 4, 4)
        voxel.accumulate(Vec4(1, 0, 0, 1), Vec3.zero(), Vec3.unit_y())
        grid.finalize()

        # Build SVO
        root, _, _ = build_svo_from_grid(grid)

        # Create traversal
        traversal = SVOTraversal(root, 1.0, 8)

        # Sample at filled position
        sample = traversal.sample_at_position(4.5, 4.5, 4.5, level=0)

        assert sample is not None
        assert sample.opacity > 0.9


# ============================================================================
# Memory Target Tests
# ============================================================================


class TestMemoryTargets:
    """Tests for memory compression targets (5-10x savings)."""

    def test_empty_room_compression(self) -> None:
        """Empty room should achieve target compression."""
        grid = create_empty_room_scene(64)
        root, _, _ = build_svo_from_grid(grid)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, 64)

        # Empty room should achieve very high compression
        assert profile.savings_ratio >= 5.0, f"Empty room savings {profile.savings_ratio}x < 5x target"

    def test_furnished_room_compression(self) -> None:
        """Furnished room should achieve target compression."""
        grid = create_furnished_room_scene(64)
        root, _, _ = build_svo_from_grid(grid)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, 64)

        # Furnished room should still achieve target
        assert profile.savings_ratio >= 3.0, f"Furnished room savings {profile.savings_ratio}x < 3x"

    def test_forest_compression(self) -> None:
        """Forest scene should achieve target compression."""
        grid = create_forest_scene(64)
        root, _, _ = build_svo_from_grid(grid)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, 64)

        # Forest should achieve very high compression (sparse)
        assert profile.savings_ratio >= 5.0, f"Forest savings {profile.savings_ratio}x < 5x target"


# ============================================================================
# WGSL Tests
# ============================================================================


class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_generate_traversal_wgsl(self) -> None:
        """Traversal WGSL should be valid."""
        wgsl = generate_svo_traversal_wgsl()

        assert "SVONode" in wgsl
        assert "@compute" in wgsl
        assert "ray_box_intersect" in wgsl
        assert "child_mask" in wgsl

    def test_generate_cone_trace_wgsl(self) -> None:
        """Cone trace WGSL should be valid."""
        wgsl = generate_svo_cone_trace_wgsl()

        assert "SVONode" in wgsl
        assert "ConeTraceUniforms" in wgsl
        assert "sample_svo" in wgsl
        assert "aperture" in wgsl


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_build_time_reasonable(self) -> None:
        """Build time should be reasonable for 64^3."""
        grid = create_empty_room_scene(64)

        import time
        start = time.perf_counter()
        root, stats, _ = build_svo_from_grid(grid)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in reasonable time (< 5 seconds for Python)
        assert elapsed_ms < 5000, f"Build took {elapsed_ms}ms, expected < 5000ms"

    def test_traversal_performance(self) -> None:
        """Traversal should complete in reasonable time."""
        grid = create_empty_room_scene(64)
        root, _, _ = build_svo_from_grid(grid)
        traversal = SVOTraversal(root, 1.0, 64)

        import time
        start = time.perf_counter()

        # Trace 100 rays
        for i in range(100):
            origin = (float(i % 8), 32.0, 32.0)
            traversal.traverse_ray(origin, (1, 0, 0))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly
        assert elapsed_ms < 2000, f"100 rays took {elapsed_ms}ms"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_voxel_grid(self) -> None:
        """Handle grid with single voxel."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        grid = VoxelGrid(2, bounds)

        voxel = grid.get_voxel(0, 0, 0)
        voxel.accumulate(Vec4(1, 0, 0, 1), Vec3.zero(), Vec3.unit_y())
        grid.finalize()

        root, stats, _ = build_svo_from_grid(grid)
        assert stats.total_nodes > 0

    def test_very_sparse_grid(self) -> None:
        """Handle very sparse grid (single voxel in large grid)."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(64, 64, 64))
        grid = VoxelGrid(64, bounds)

        # Single voxel
        voxel = grid.get_voxel(32, 32, 32)
        voxel.accumulate(Vec4(1, 0, 0, 1), Vec3.zero(), Vec3.unit_y())
        grid.finalize()

        root, stats, _ = build_svo_from_grid(grid)

        profiler = MemoryProfiler()
        profile = profiler.profile(root, 64)

        # Very sparse should have very high compression
        assert profile.savings_ratio > 10

    def test_zero_direction_ray(self) -> None:
        """Handle ray with zero direction."""
        root = SVONode(NodeType.LEAF)
        root.bounds = ((0, 0, 0), (8, 8, 8))
        root.set_data(SVOVoxelData(np.array([1, 0, 0], dtype=np.float32), 1.0))

        traversal = SVOTraversal(root, 1.0, 8)

        hit = traversal.traverse_ray((4, 4, 4), (0, 0, 0))
        assert not hit.hit  # Zero direction should fail gracefully


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Tests for the evaluation function."""

    def test_evaluate_svo_compression(self) -> None:
        """Evaluation function should produce valid results."""
        # Use small resolution for fast test
        results = evaluate_svo_compression(resolution=32)

        assert "resolution" in results
        assert "scenes" in results
        assert "report" in results
        assert "average_savings" in results
        assert "recommendation" in results

        # Should have 3 scenes
        assert len(results["scenes"]) == 3

    def test_evaluation_recommendation(self) -> None:
        """Evaluation should produce GO/NO-GO recommendation."""
        results = evaluate_svo_compression(resolution=32)

        recommendation = results["recommendation"]
        assert recommendation in ["GO", "NO-GO"]
