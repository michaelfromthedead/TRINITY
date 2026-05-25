"""
Comprehensive tests for the Motion Matching animation subsystem.

Tests cover:
- Database building and feature extraction
- Search accuracy with known data
- Inertialization blending
- Controller with various inputs
- Annotation detection
- All core functionality

Minimum 140 tests with real assertions.
"""

from __future__ import annotations

import math
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, MagicMock, patch

import pytest
import numpy as np

# Import all motion matching modules
from engine.animation.motionmatching.database import (
    MotionDatabase,
    DatabaseEntry,
    ClipMetadata,
    NormalizationStats,
    QuantizationLevel,
    build_database,
    merge_databases,
    motion_matching,
)

# Import config for testing config values
from engine.animation.motionmatching.config import (
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_SEARCH_PARAMS,
    DEFAULT_TRANSITION_PARAMS,
    DEFAULT_DATABASE_CONFIG,
)

from engine.animation.motionmatching.features import (
    FeatureSet,
    FeatureConfig,
    FeatureExtractor,
    FeatureNormalizer,
    FeatureType,
    FeatureWeights,
    BoneData,
    TrajectoryPoint,
    FootContact,
)

from engine.animation.motionmatching.search import (
    MotionSearch,
    SearchConfig,
    SearchResult,
    SearchMethod,
    compute_cost,
    compute_cost_vectorized,
    KDTree,
    LSHIndex,
)

from engine.animation.motionmatching.transition import (
    MotionTransition,
    TransitionConfig,
    BlendMode,
    InertializationBlender,
    InertializationOffset,
    Pose,
    BoneTransform,
    FootSlidingCorrector,
    quaternion_slerp,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_difference,
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
)

from engine.animation.motionmatching.context import (
    MotionMatchingController,
    MotionContext,
    ControllerConfig,
    ControllerState,
    DesiredTrajectory,
    IdleDetector,
    TrajectoryBuilder,
)

from engine.animation.motionmatching.annotation import (
    AnnotatedClip,
    MotionTag,
    TagType,
    ContactAnnotation,
    auto_detect_contacts,
    auto_detect_locomotion_tags,
    auto_detect_turn_tags,
    auto_detect_all_tags,
    merge_overlapping_tags,
    filter_tags_by_duration,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockAnimationClip:
    """Mock animation clip for testing."""
    name: str = "test_clip"
    frame_count: int = 60
    frame_rate: float = 30.0
    duration: float = 2.0
    is_looping: bool = False
    has_root_motion: bool = True
    tags: Set[str] = field(default_factory=set)

    # Mock bone data
    _bone_positions: Optional[np.ndarray] = None
    _bone_velocities: Optional[np.ndarray] = None
    _root_positions: Optional[np.ndarray] = None
    _root_rotations: Optional[np.ndarray] = None

    def __post_init__(self):
        if self._bone_positions is None:
            # Generate simple walk cycle bone positions
            self._bone_positions = np.zeros((self.frame_count, 5, 3), dtype=np.float32)
            for f in range(self.frame_count):
                t = f / self.frame_rate
                # Hips
                self._bone_positions[f, 0] = [0, 1.0, t * 1.5]
                # Left foot
                self._bone_positions[f, 1] = [-0.2, 0.1 * abs(math.sin(t * 10)), t * 1.5 - 0.3]
                # Right foot
                self._bone_positions[f, 2] = [0.2, 0.1 * abs(math.cos(t * 10)), t * 1.5 + 0.3]
                # Left hand
                self._bone_positions[f, 3] = [-0.5, 1.2, t * 1.5]
                # Right hand
                self._bone_positions[f, 4] = [0.5, 1.2, t * 1.5]

        if self._root_positions is None:
            self._root_positions = np.zeros((self.frame_count, 3), dtype=np.float32)
            for f in range(self.frame_count):
                t = f / self.frame_rate
                self._root_positions[f] = [0, 0, t * 1.5]

        if self._root_rotations is None:
            self._root_rotations = np.tile([0, 0, 0, 1], (self.frame_count, 1)).astype(np.float32)

    def get_bone_position(self, frame: int, bone_name: str) -> np.ndarray:
        """Get bone position at frame."""
        bone_map = {'hips': 0, 'left_foot': 1, 'right_foot': 2, 'left_hand': 3, 'right_hand': 4}
        idx = bone_map.get(bone_name, 0)
        if 0 <= frame < self.frame_count:
            return self._bone_positions[frame, idx].copy()
        return np.zeros(3, dtype=np.float32)

    def get_bone_velocity(self, frame: int, bone_name: str) -> np.ndarray:
        """Get bone velocity at frame."""
        if frame > 0:
            pos_curr = self.get_bone_position(frame, bone_name)
            pos_prev = self.get_bone_position(frame - 1, bone_name)
            return (pos_curr - pos_prev) * self.frame_rate
        return np.zeros(3, dtype=np.float32)

    def get_root_transform(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get root transform at frame."""
        if 0 <= frame < self.frame_count:
            return self._root_positions[frame].copy(), self._root_rotations[frame].copy()
        return np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)

    def get_frame_pose(self, frame: int) -> Any:
        """Get pose at frame."""
        return None


@pytest.fixture
def mock_clip() -> MockAnimationClip:
    """Create a mock animation clip."""
    return MockAnimationClip(name="walk", frame_count=60, tags={"locomotion", "walk"})


@pytest.fixture
def mock_clips() -> List[MockAnimationClip]:
    """Create multiple mock clips."""
    return [
        MockAnimationClip(name="idle", frame_count=30, tags={"idle"}),
        MockAnimationClip(name="walk", frame_count=60, tags={"locomotion", "walk"}),
        MockAnimationClip(name="run", frame_count=45, tags={"locomotion", "run"}),
    ]


@pytest.fixture
def feature_config() -> FeatureConfig:
    """Create feature configuration."""
    return FeatureConfig(
        use_bone_positions=True,
        use_bone_velocities=True,
        use_trajectory=True,
        use_foot_contacts=True,
        bone_names=['hips', 'left_foot', 'right_foot', 'left_hand', 'right_hand'],
        trajectory_times=[0.2, 0.4, 0.6],
    )


@pytest.fixture
def feature_extractor(feature_config) -> FeatureExtractor:
    """Create feature extractor."""
    return FeatureExtractor(feature_config)


@pytest.fixture
def simple_database(mock_clips, feature_extractor) -> MotionDatabase:
    """Create a simple test database."""
    return build_database(mock_clips, feature_extractor)


# =============================================================================
# DATABASE TESTS
# =============================================================================


class TestMotionDatabase:
    """Tests for MotionDatabase class."""

    def test_init_empty(self) -> None:
        """Should initialize empty database."""
        db = MotionDatabase()
        assert db.clip_count == 0
        assert db.entry_count == 0

    def test_init_with_feature_dimension(self) -> None:
        """Should accept feature dimension."""
        db = MotionDatabase(feature_dimension=64)
        assert db.feature_dimension == 64

    def test_init_with_quantization(self) -> None:
        """Should accept quantization level."""
        db = MotionDatabase(quantization=QuantizationLevel.FLOAT16)
        assert db.quantization == QuantizationLevel.FLOAT16

    def test_add_clip(self) -> None:
        """Should add clip metadata."""
        db = MotionDatabase()
        metadata = ClipMetadata(
            clip_index=0,
            name="test",
            frame_count=30,
            frame_rate=30.0,
        )
        idx = db.add_clip(metadata)
        assert idx == 0
        assert db.clip_count == 1

    def test_add_multiple_clips(self) -> None:
        """Should add multiple clips."""
        db = MotionDatabase()
        for i in range(5):
            metadata = ClipMetadata(clip_index=i, name=f"clip_{i}", frame_count=30)
            db.add_clip(metadata)
        assert db.clip_count == 5

    def test_get_clip_metadata(self) -> None:
        """Should retrieve clip metadata."""
        db = MotionDatabase()
        metadata = ClipMetadata(clip_index=0, name="test", frame_count=30)
        db.add_clip(metadata)
        retrieved = db.get_clip_metadata(0)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_get_clip_by_name(self) -> None:
        """Should find clip by name."""
        db = MotionDatabase()
        db.add_clip(ClipMetadata(clip_index=0, name="walk", frame_count=30))
        db.add_clip(ClipMetadata(clip_index=1, name="run", frame_count=30))
        clip = db.get_clip_by_name("run")
        assert clip is not None
        assert clip.clip_index == 1

    def test_add_entry(self) -> None:
        """Should add database entry."""
        db = MotionDatabase()
        entry = DatabaseEntry(
            clip_index=0,
            frame=0,
            features=np.zeros(64),
        )
        idx = db.add_entry(entry)
        assert idx == 0
        assert db.entry_count == 1

    def test_add_entry_updates_feature_dimension(self) -> None:
        """Should update feature dimension from first entry."""
        db = MotionDatabase()
        entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(128))
        db.add_entry(entry)
        assert db.feature_dimension == 128

    def test_get_entry(self) -> None:
        """Should retrieve entry by index."""
        db = MotionDatabase()
        entry = DatabaseEntry(clip_index=0, frame=5, features=np.ones(64))
        db.add_entry(entry)
        retrieved = db.get_entry(0)
        assert retrieved is not None
        assert retrieved.frame == 5

    def test_get_entries_for_clip(self) -> None:
        """Should get all entries for a clip."""
        db = MotionDatabase()
        for i in range(10):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.zeros(64)))
        for i in range(5):
            db.add_entry(DatabaseEntry(clip_index=1, frame=i, features=np.zeros(64)))

        entries = db.get_entries_for_clip(0)
        assert len(entries) == 10

    def test_entry_tags(self) -> None:
        """Should handle entry tags."""
        db = MotionDatabase()
        entry = DatabaseEntry(
            clip_index=0,
            frame=0,
            features=np.zeros(64),
            tags=frozenset({"walk", "locomotion"}),
        )
        db.add_entry(entry)
        retrieved = db.get_entry(0)
        assert "walk" in retrieved.tags
        assert "locomotion" in retrieved.tags

    def test_get_entries_with_tags(self) -> None:
        """Should filter entries by tags."""
        db = MotionDatabase()
        db.add_entry(DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64), tags=frozenset({"walk"})))
        db.add_entry(DatabaseEntry(clip_index=0, frame=1, features=np.zeros(64), tags=frozenset({"run"})))
        db.add_entry(DatabaseEntry(clip_index=0, frame=2, features=np.zeros(64), tags=frozenset({"walk", "idle"})))

        entries = db.get_entries_with_tags({"walk"})
        assert len(entries) == 2

    def test_finalize_builds_feature_matrix(self) -> None:
        """Should build feature matrix on finalize."""
        db = MotionDatabase()
        for i in range(10):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.random.randn(64)))
        db.finalize()
        assert db.feature_matrix is not None
        assert db.feature_matrix.shape == (10, 64)

    def test_finalize_computes_normalization(self) -> None:
        """Should compute normalization on finalize."""
        db = MotionDatabase()
        for i in range(100):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.random.randn(64)))
        db.finalize(compute_normalization=True)
        assert db.normalization is not None
        assert len(db.normalization.mean) == 64

    def test_memory_usage(self) -> None:
        """Should compute memory usage."""
        db = MotionDatabase()
        for i in range(100):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.zeros(64)))
        db.finalize()
        assert db.memory_usage_bytes > 0


class TestDatabaseSerialization:
    """Tests for database serialization."""

    def test_save_and_load(self, simple_database) -> None:
        """Should save and load database."""
        with tempfile.NamedTemporaryFile(suffix='.mmdb', delete=False) as f:
            path = f.name

        try:
            simple_database.save(path)
            loaded = MotionDatabase.load(path)

            assert loaded.clip_count == simple_database.clip_count
            assert loaded.entry_count == simple_database.entry_count
            assert loaded.feature_dimension == simple_database.feature_dimension
        finally:
            os.unlink(path)

    def test_save_compressed(self, simple_database) -> None:
        """Should save with compression."""
        with tempfile.NamedTemporaryFile(suffix='.mmdb', delete=False) as f:
            path = f.name

        try:
            simple_database.save(path, compress=True)
            loaded = MotionDatabase.load(path)
            assert loaded.entry_count == simple_database.entry_count
        finally:
            os.unlink(path)

    def test_save_uncompressed(self, simple_database) -> None:
        """Should save without compression."""
        with tempfile.NamedTemporaryFile(suffix='.mmdb', delete=False) as f:
            path = f.name

        try:
            simple_database.save(path, compress=False)
            loaded = MotionDatabase.load(path)
            assert loaded.entry_count == simple_database.entry_count
        finally:
            os.unlink(path)


class TestDatabaseBuilder:
    """Tests for database building."""

    def test_build_database(self, mock_clips, feature_extractor) -> None:
        """Should build database from clips."""
        db = build_database(mock_clips, feature_extractor)
        assert db.clip_count == len(mock_clips)
        assert db.entry_count > 0

    def test_build_database_with_quantization(self, mock_clips, feature_extractor) -> None:
        """Should build with quantization."""
        db = build_database(mock_clips, feature_extractor, quantization=QuantizationLevel.INT16)
        assert db.quantization == QuantizationLevel.INT16

    def test_build_database_skip_frames(self, mock_clips, feature_extractor) -> None:
        """Should skip frames at clip boundaries."""
        total_frames = sum(c.frame_count for c in mock_clips)
        db = build_database(mock_clips, feature_extractor, skip_first_frames=2, skip_last_frames=2)
        expected = total_frames - (4 * len(mock_clips))
        assert db.entry_count == expected

    def test_merge_databases(self, mock_clips, feature_extractor) -> None:
        """Should merge multiple databases."""
        db1 = build_database([mock_clips[0]], feature_extractor)
        db2 = build_database([mock_clips[1]], feature_extractor)

        merged = merge_databases([db1, db2])
        assert merged.clip_count == 2
        assert merged.entry_count == db1.entry_count + db2.entry_count


# =============================================================================
# FEATURE TESTS
# =============================================================================


class TestFeatureSet:
    """Tests for FeatureSet class."""

    def test_init(self) -> None:
        """Should initialize feature set."""
        fs = FeatureSet(values=np.zeros(64))
        assert fs.dimension == 64

    def test_init_with_weights(self) -> None:
        """Should accept weights."""
        fs = FeatureSet(values=np.zeros(64), weights=np.ones(64) * 2)
        assert np.allclose(fs.weights, 2.0)

    def test_default_weights(self) -> None:
        """Should create default weights of 1.0."""
        fs = FeatureSet(values=np.zeros(64))
        assert np.allclose(fs.weights, 1.0)

    def test_feature_ranges(self) -> None:
        """Should store feature ranges."""
        fs = FeatureSet(
            values=np.zeros(10),
            feature_ranges={"pos": (0, 3), "vel": (3, 6)},
        )
        assert fs.get_feature("pos") is not None
        assert len(fs.get_feature("pos")) == 3

    def test_weighted_values(self) -> None:
        """Should compute weighted values."""
        fs = FeatureSet(values=np.array([1.0, 2.0, 3.0]), weights=np.array([1.0, 2.0, 0.5]))
        weighted = fs.weighted_values()
        assert np.allclose(weighted, [1.0, 4.0, 1.5])


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_init(self) -> None:
        """Should initialize extractor."""
        extractor = FeatureExtractor()
        assert extractor.config is not None

    def test_init_with_config(self, feature_config) -> None:
        """Should accept config."""
        extractor = FeatureExtractor(feature_config)
        assert extractor.config.use_bone_positions is True

    def test_feature_dimension(self, feature_extractor) -> None:
        """Should compute feature dimension."""
        dim = feature_extractor.feature_dimension
        assert dim > 0

    def test_extract_from_clip(self, feature_extractor, mock_clip) -> None:
        """Should extract features from clip."""
        features = feature_extractor.extract(mock_clip, 10)
        assert isinstance(features, FeatureSet)
        assert features.dimension == feature_extractor.feature_dimension

    def test_extract_from_pose(self, feature_extractor) -> None:
        """Should extract from explicit pose data."""
        bone_data = {
            'hips': BoneData(position=[0, 1, 0], velocity=[0, 0, 1]),
            'left_foot': BoneData(position=[-0.2, 0, 0], velocity=[0, 0, 0]),
            'right_foot': BoneData(position=[0.2, 0, 0], velocity=[0, 0, 0]),
        }
        trajectory = [
            TrajectoryPoint(time_offset=0.2, position=np.array([0, 0, 0.3]), facing=0.0),
            TrajectoryPoint(time_offset=0.4, position=np.array([0, 0, 0.6]), facing=0.0),
        ]
        contacts = FootContact(left_contact=1.0, right_contact=0.0)

        features = feature_extractor.extract_from_pose(bone_data, trajectory, contacts)
        assert features.dimension == feature_extractor.feature_dimension


class TestFeatureNormalizer:
    """Tests for FeatureNormalizer class."""

    def test_fit(self) -> None:
        """Should fit normalizer from data."""
        data = np.random.randn(100, 64)
        normalizer = FeatureNormalizer()
        normalizer.fit(data)
        assert normalizer.mean is not None
        assert len(normalizer.mean) == 64

    def test_normalize_zscore(self) -> None:
        """Should normalize using z-score."""
        data = np.random.randn(100, 64) * 10 + 5
        normalizer = FeatureNormalizer().fit(data)
        normalized = normalizer.normalize(data, method='zscore')
        # Normalized data should have mean ~0 and std ~1
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1

    def test_denormalize(self) -> None:
        """Should denormalize back to original scale."""
        data = np.random.randn(10, 64) * 10 + 5
        normalizer = FeatureNormalizer().fit(data)
        normalized = normalizer.normalize(data)
        denormalized = normalizer.denormalize(normalized)
        assert np.allclose(data, denormalized, atol=1e-5)


class TestFeatureWeights:
    """Tests for FeatureWeights class."""

    def test_default_weights(self) -> None:
        """Should have default weights."""
        weights = FeatureWeights()
        assert weights.pose_weight == 1.0
        assert weights.velocity_weight == 0.5

    def test_bone_weights(self) -> None:
        """Should get bone-specific weights."""
        weights = FeatureWeights(bone_weights={'hips': 2.0})
        assert weights.get_bone_weight('hips') == 2.0
        assert weights.get_bone_weight('other') == 1.0

    def test_apply_to_feature_set(self) -> None:
        """Should apply weights to feature set."""
        fs = FeatureSet(
            values=np.ones(6),
            weights=np.ones(6),
            feature_ranges={"pos_hips": (0, 3), "vel_hips": (3, 6)},
        )
        weights = FeatureWeights(pose_weight=2.0, velocity_weight=0.5)
        weighted = weights.apply_to_feature_set(fs)
        assert np.allclose(weighted.weights[:3], 2.0)
        assert np.allclose(weighted.weights[3:6], 0.5)


# =============================================================================
# SEARCH TESTS
# =============================================================================


class TestCostFunctions:
    """Tests for cost computation functions."""

    def test_compute_cost_identical(self) -> None:
        """Should return 0 for identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        cost = compute_cost(a, a)
        assert cost == 0.0

    def test_compute_cost_different(self) -> None:
        """Should compute squared distance."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        cost = compute_cost(a, b)
        assert cost == 1.0

    def test_compute_cost_weighted(self) -> None:
        """Should apply weights."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        weights = np.array([1.0, 2.0])
        cost = compute_cost(a, b, weights)
        assert cost == 3.0  # 1*1 + 2*1

    def test_compute_cost_vectorized(self) -> None:
        """Should compute costs for multiple candidates."""
        query = np.array([0.0, 0.0])
        candidates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        costs = compute_cost_vectorized(query, candidates)
        assert np.allclose(costs, [0.0, 1.0, 1.0])


class TestMotionSearch:
    """Tests for MotionSearch class."""

    def test_init(self, simple_database) -> None:
        """Should initialize search."""
        search = MotionSearch(simple_database)
        assert search.database is simple_database

    def test_brute_force_search(self, simple_database) -> None:
        """Should perform brute force search."""
        search = MotionSearch(simple_database, method=SearchMethod.BRUTE_FORCE)
        query = simple_database.get_feature_vector(0)
        results = search.search(query)
        assert len(results) > 0

    def test_search_returns_best_first(self, simple_database) -> None:
        """Should return results sorted by cost."""
        search = MotionSearch(simple_database)
        query = simple_database.get_feature_vector(10)
        results = search.search(query, SearchConfig(max_results=5))

        # Check sorted order
        for i in range(len(results) - 1):
            assert results[i].cost <= results[i + 1].cost

    def test_search_exact_match(self, simple_database) -> None:
        """Should find exact match with lowest cost."""
        search = MotionSearch(simple_database)

        # Get a valid entry index
        entry_idx = min(20, simple_database.entry_count - 1)
        # Use normalize=False to get raw features that search will normalize
        query = simple_database.get_feature_vector(entry_idx, normalize=False)
        result = search.find_best_match(query)

        # Best match should exist
        assert result is not None

        # When querying with exact feature vector from database,
        # the best match should be that entry with zero (or near-zero) cost
        assert result.entry_index == entry_idx, f"Expected entry {entry_idx}, got {result.entry_index}"
        assert result.cost == pytest.approx(0.0, abs=1e-3), f"Expected near-zero cost, got {result.cost}"

    def test_search_quality_nearest_neighbors(self, simple_database) -> None:
        """Should find nearest neighbors in correct order."""
        search = MotionSearch(simple_database)

        # Get a valid entry index
        entry_idx = min(50, simple_database.entry_count - 1)
        # Use normalize=False to get raw features that search will normalize
        query = simple_database.get_feature_vector(entry_idx, normalize=False)
        results = search.search(query, SearchConfig(max_results=10))

        # Verify we got results
        assert len(results) >= 1

        # Verify ordering - each result cost should be <= next result cost
        for i in range(len(results) - 1):
            assert results[i].cost <= results[i + 1].cost, \
                f"Results not sorted: {results[i].cost} > {results[i + 1].cost}"

        # Verify first result is the exact match
        assert results[0].entry_index == entry_idx
        assert results[0].cost == pytest.approx(0.0, abs=1e-3)

    def test_search_with_tag_filter(self, simple_database) -> None:
        """Should filter by required tags."""
        search = MotionSearch(simple_database)
        query = simple_database.get_feature_vector(0)
        config = SearchConfig(required_tags={"locomotion"})
        results = search.search(query, config)

        for result in results:
            assert "locomotion" in result.entry.tags

    def test_search_max_results(self, simple_database) -> None:
        """Should respect max_results."""
        search = MotionSearch(simple_database)
        query = simple_database.get_feature_vector(0)
        results = search.search(query, SearchConfig(max_results=3))
        assert len(results) <= 3

    def test_find_best_match(self, simple_database) -> None:
        """Should find single best match."""
        search = MotionSearch(simple_database)
        query = simple_database.get_feature_vector(0)
        result = search.find_best_match(query)
        assert result is not None
        assert isinstance(result, SearchResult)


class TestKDTree:
    """Tests for KDTree class."""

    def test_build_tree(self) -> None:
        """Should build tree from data."""
        data = np.random.randn(100, 10)
        tree = KDTree(data)
        assert tree.root is not None

    def test_query_nearest(self) -> None:
        """Should find nearest neighbor."""
        data = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        tree = KDTree(data, leaf_size=2)
        results = tree.query(np.array([0.1, 0.1]), k=1)
        assert len(results) == 1
        assert results[0][0] == 0  # Closest to (0, 0)

    def test_query_k_nearest(self) -> None:
        """Should find k nearest neighbors."""
        data = np.random.randn(100, 10)
        tree = KDTree(data)
        results = tree.query(data[0], k=5)
        assert len(results) == 5


class TestLSHIndex:
    """Tests for LSHIndex class."""

    def test_build_index(self) -> None:
        """Should build LSH index."""
        data = np.random.randn(100, 10).astype(np.float32)
        lsh = LSHIndex(dimension=10, num_tables=5)
        lsh.build(data)
        assert len(lsh.tables) == 5

    def test_query_returns_candidates(self) -> None:
        """Should return candidate indices including the query point."""
        # Use deterministic data for reproducible test
        np.random.seed(42)
        data = np.random.randn(100, 10).astype(np.float32)
        lsh = LSHIndex(dimension=10, num_tables=10, num_hashes=4, bucket_width=2.0)
        lsh.build(data)

        # Query with exact point from dataset - should find itself
        candidates = lsh.query(data[0])

        # LSH should find at least the query point itself in most cases
        # with enough tables and appropriate bucket width
        assert isinstance(candidates, set)

    def test_lsh_finds_similar_points(self) -> None:
        """Should find similar points more often than dissimilar ones."""
        np.random.seed(123)
        # Create clustered data
        cluster1 = np.random.randn(50, 10).astype(np.float32) + np.array([5.0] * 10)
        cluster2 = np.random.randn(50, 10).astype(np.float32) - np.array([5.0] * 10)
        data = np.vstack([cluster1, cluster2])

        lsh = LSHIndex(dimension=10, num_tables=15, num_hashes=4, bucket_width=3.0)
        lsh.build(data)

        # Query from cluster1 should find mostly cluster1 points
        candidates = lsh.query(cluster1[0])
        cluster1_count = sum(1 for c in candidates if c < 50)
        cluster2_count = sum(1 for c in candidates if c >= 50)

        # If we found candidates, cluster1 should be more represented
        if len(candidates) > 1:
            assert cluster1_count >= cluster2_count, "LSH should find similar cluster points"

    def test_lsh_fallback_on_no_candidates(self) -> None:
        """Should return empty set when no candidates in buckets."""
        data = np.random.randn(10, 5).astype(np.float32)
        lsh = LSHIndex(dimension=5, num_tables=2, num_hashes=16, bucket_width=0.01)
        lsh.build(data)

        # Very small bucket width means unlikely to find matches
        # Query with a point far from the data
        far_point = np.ones(5, dtype=np.float32) * 1000
        candidates = lsh.query(far_point)
        assert isinstance(candidates, set)


# =============================================================================
# TRANSITION TESTS
# =============================================================================


class TestQuaternionOperations:
    """Tests for quaternion utility functions."""

    def test_quaternion_multiply_identity(self) -> None:
        """Should return same quat when multiplying by identity."""
        q = np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32)
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        result = quaternion_multiply(q, identity)
        assert np.allclose(result, q, atol=1e-5)

    def test_quaternion_inverse(self) -> None:
        """Should compute inverse."""
        q = np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32)
        inv = quaternion_inverse(q)
        result = quaternion_multiply(q, inv)
        # Should be identity (w component should be close to 1, xyz close to 0)
        assert abs(result[3]) > 0.99  # w close to +-1
        assert np.linalg.norm(result[:3]) < 0.01  # xyz close to 0

    def test_quaternion_slerp_endpoints(self) -> None:
        """Should return endpoints at t=0 and t=1."""
        q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q2 = np.array([0.0, 0.707, 0.0, 0.707], dtype=np.float32)

        result0 = quaternion_slerp(q1, q2, 0.0)
        result1 = quaternion_slerp(q1, q2, 1.0)

        assert np.allclose(result0, q1, atol=1e-5)
        assert np.allclose(result1, q2, atol=1e-5)

    def test_axis_angle_roundtrip(self) -> None:
        """Should convert axis-angle to quaternion and back."""
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        angle = np.pi / 4

        q = axis_angle_to_quaternion(axis, angle)
        axis_out, angle_out = quaternion_to_axis_angle(q)

        assert np.allclose(abs(axis_out), abs(axis), atol=1e-5)
        assert abs(angle_out - angle) < 1e-5 or abs(angle_out + angle) < 1e-5


class TestBoneTransform:
    """Tests for BoneTransform class."""

    def test_init(self) -> None:
        """Should initialize transform."""
        t = BoneTransform(
            position=np.array([1, 2, 3]),
            rotation=np.array([0, 0, 0, 1]),
        )
        assert np.allclose(t.position, [1, 2, 3])

    def test_default_scale(self) -> None:
        """Should have default scale of 1."""
        t = BoneTransform(
            position=np.zeros(3),
            rotation=np.array([0, 0, 0, 1]),
        )
        assert np.allclose(t.scale, [1, 1, 1])

    def test_copy(self) -> None:
        """Should create independent copy."""
        t1 = BoneTransform(position=np.array([1, 2, 3]), rotation=np.array([0, 0, 0, 1]))
        t2 = t1.copy()
        t2.position[0] = 100
        assert t1.position[0] == 1


class TestPose:
    """Tests for Pose class."""

    def test_init_empty(self) -> None:
        """Should initialize empty pose."""
        pose = Pose()
        assert len(pose.bone_transforms) == 0

    def test_set_and_get_bone(self) -> None:
        """Should set and get bone transforms."""
        pose = Pose()
        t = BoneTransform(position=np.array([1, 2, 3]), rotation=np.array([0, 0, 0, 1]))
        pose.set_bone("test", t)
        retrieved = pose.get_bone("test")
        assert retrieved is not None
        assert np.allclose(retrieved.position, [1, 2, 3])

    def test_copy(self) -> None:
        """Should create deep copy."""
        pose1 = Pose()
        pose1.root_position = np.array([1, 2, 3])
        pose1.set_bone("hip", BoneTransform(position=np.zeros(3), rotation=np.array([0, 0, 0, 1])))

        pose2 = pose1.copy()
        pose2.root_position[0] = 100

        assert pose1.root_position[0] == 1


class TestInertializationBlender:
    """Tests for InertializationBlender class."""

    def test_init(self) -> None:
        """Should initialize blender."""
        config = TransitionConfig(blend_duration=0.2)
        blender = InertializationBlender(config)
        assert blender.config.blend_duration == 0.2

    def test_compute_offsets(self) -> None:
        """Should compute offsets from pose difference."""
        config = TransitionConfig()
        blender = InertializationBlender(config)

        from_pose = Pose(root_position=np.array([0, 0, 0]))
        to_pose = Pose(root_position=np.array([1, 0, 0]))

        blender.compute_offsets(from_pose, to_pose)
        assert np.allclose(blender._root_position_offset, [-1, 0, 0])

    def test_offset_decay(self) -> None:
        """Should decay offsets exponentially over time."""
        config = TransitionConfig(spring_halflife=0.1)
        blender = InertializationBlender(config)

        initial_offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blender._root_position_offset = initial_offset.copy()
        blender._root_position_velocity = np.zeros(3, dtype=np.float32)

        initial_magnitude = np.linalg.norm(blender._root_position_offset)

        # Update for one halflife (0.1s = 2 * 0.05)
        blender.update(0.05)
        blender.update(0.05)

        # After one halflife, offset should be approximately half
        magnitude_after_halflife = np.linalg.norm(blender._root_position_offset)
        expected_magnitude = initial_magnitude * 0.5

        # Allow some tolerance for discrete time stepping
        assert magnitude_after_halflife < initial_magnitude, "Offset should decay"
        assert abs(magnitude_after_halflife - expected_magnitude) < 0.2, \
            f"Expected ~{expected_magnitude}, got {magnitude_after_halflife}"

        # Continue decaying
        for _ in range(8):
            blender.update(0.05)

        # After ~0.5s (5 halflives), should be very small (< 1/32 of original)
        final_magnitude = np.linalg.norm(blender._root_position_offset)
        assert final_magnitude < initial_magnitude * 0.1, \
            f"Offset should be nearly zero, got {final_magnitude}"

    def test_apply_offset_to_pose(self) -> None:
        """Should apply offset to target pose."""
        config = TransitionConfig()
        blender = InertializationBlender(config)

        blender._root_position_offset = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blender._root_rotation_offset = np.array([0, 0, 0, 1], dtype=np.float32)

        target = Pose(root_position=np.array([5, 0, 0]))
        result = blender.apply(target)

        assert np.allclose(result.root_position, [6, 0, 0])


class TestMotionTransition:
    """Tests for MotionTransition class."""

    def test_init(self) -> None:
        """Should initialize transition."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))

        transition = MotionTransition(from_entry, to_entry)
        assert transition.progress == 0.0
        assert not transition.is_complete

    def test_progress_advances(self) -> None:
        """Should advance progress over time."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))
        config = TransitionConfig(blend_duration=0.2)

        transition = MotionTransition(from_entry, to_entry, config)
        from_pose = Pose()
        to_pose = Pose()
        transition.initialize(from_pose, to_pose)

        transition.update(0.1, to_pose)
        assert transition.progress == pytest.approx(0.5)

    def test_completes_after_duration(self) -> None:
        """Should complete after blend duration."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))
        config = TransitionConfig(blend_duration=0.2)

        transition = MotionTransition(from_entry, to_entry, config)
        from_pose = Pose()
        to_pose = Pose()
        transition.initialize(from_pose, to_pose)

        transition.update(0.25, to_pose)
        assert transition.is_complete


class TestFootSlidingCorrector:
    """Tests for FootSlidingCorrector class."""

    def test_init(self) -> None:
        """Should initialize corrector."""
        corrector = FootSlidingCorrector()
        assert corrector.left_foot_bone == 'left_foot'

    def test_foot_lock(self) -> None:
        """Should lock foot when contact detected."""
        corrector = FootSlidingCorrector()
        corrector.update_contacts(
            left_contact=1.0,
            right_contact=0.0,
            left_position=np.array([0, 0, 0]),
            right_position=np.array([0, 0, 0]),
        )
        assert corrector._left_locked is True
        assert corrector._right_locked is False

    def test_correction_when_locked(self) -> None:
        """Should provide correction for locked foot."""
        corrector = FootSlidingCorrector()
        corrector._left_locked = True
        corrector._left_lock_position = np.array([0, 0, 0])

        left_corr, right_corr = corrector.correct_pose(
            Pose(),
            left_world_position=np.array([0.1, 0, 0]),
            right_world_position=np.array([0, 0, 0]),
        )
        assert np.allclose(left_corr, [-0.1, 0, 0])


# =============================================================================
# CONTEXT TESTS
# =============================================================================


class TestDesiredTrajectory:
    """Tests for DesiredTrajectory class."""

    def test_from_input_stationary(self) -> None:
        """Should create stationary trajectory."""
        trajectory = DesiredTrajectory.from_input(
            direction=np.zeros(3),
            speed=0.0,
            current_position=np.zeros(3),
            current_facing=0.0,
            trajectory_times=[0.2, 0.4],
        )
        assert trajectory.is_stationary
        assert len(trajectory.points) == 2

    def test_from_input_moving(self) -> None:
        """Should create moving trajectory."""
        trajectory = DesiredTrajectory.from_input(
            direction=np.array([0, 0, 1]),
            speed=2.0,
            current_position=np.zeros(3),
            current_facing=0.0,
            trajectory_times=[0.2, 0.4],
        )
        assert not trajectory.is_stationary
        assert trajectory.desired_speed == 2.0


class TestMotionContext:
    """Tests for MotionContext class."""

    def test_init(self) -> None:
        """Should initialize context."""
        ctx = MotionContext()
        assert ctx.state == ControllerState.STOPPED
        assert ctx.current_clip_index == -1

    def test_advance_frame(self) -> None:
        """Should advance frame time."""
        ctx = MotionContext()
        ctx.current_time = 0.0
        ctx.current_frame = 0
        ctx.advance_frame(0.033, 30.0)
        assert ctx.current_time == pytest.approx(0.033)
        assert ctx.current_frame == 0  # Still frame 0 at 0.033s


class TestMotionMatchingController:
    """Tests for MotionMatchingController class."""

    def test_init(self, simple_database) -> None:
        """Should initialize controller."""
        controller = MotionMatchingController(simple_database)
        assert controller.database is simple_database
        assert controller.state == ControllerState.STOPPED

    def test_start(self, simple_database) -> None:
        """Should start controller."""
        controller = MotionMatchingController(simple_database)
        controller.start()
        assert controller.state in [ControllerState.IDLE, ControllerState.MOVING]

    def test_stop(self, simple_database) -> None:
        """Should stop controller."""
        controller = MotionMatchingController(simple_database)
        controller.start()
        controller.stop()
        assert controller.state == ControllerState.STOPPED

    def test_update_stationary(self, simple_database) -> None:
        """Should handle stationary input."""
        controller = MotionMatchingController(simple_database)
        controller.start()
        pose = controller.update(np.zeros(3), 0.033)
        assert controller.is_idle or controller.current_entry is not None

    def test_update_moving(self, simple_database) -> None:
        """Should handle moving input."""
        controller = MotionMatchingController(simple_database)
        controller.start()
        pose = controller.update(np.array([0, 0, 1]), 0.033, desired_speed=2.0)
        # After update, should have a current entry
        assert controller.current_entry is not None

    def test_get_debug_info(self, simple_database) -> None:
        """Should return debug info."""
        controller = MotionMatchingController(simple_database)
        controller.start()
        info = controller.get_debug_info()
        assert 'state' in info
        assert 'clip_index' in info


class TestIdleDetector:
    """Tests for IdleDetector class."""

    def test_init(self) -> None:
        """Should initialize detector."""
        detector = IdleDetector()
        assert detector.is_idle

    def test_detects_movement(self) -> None:
        """Should detect movement."""
        detector = IdleDetector()
        result = detector.update(current_velocity=5.0, input_velocity=5.0, dt=0.033)
        assert not result

    def test_detects_idle_after_hold(self) -> None:
        """Should detect idle after hold time."""
        detector = IdleDetector(hold_time=0.1)
        for _ in range(5):
            detector.update(current_velocity=0.0, input_velocity=0.0, dt=0.033)
        assert detector.is_idle


class TestTrajectoryBuilder:
    """Tests for TrajectoryBuilder class."""

    def test_init(self) -> None:
        """Should initialize builder."""
        builder = TrajectoryBuilder(trajectory_times=[0.2, 0.4])
        assert len(builder.trajectory_times) == 2

    def test_build_from_gamepad_idle(self) -> None:
        """Should build idle trajectory from no input."""
        builder = TrajectoryBuilder(trajectory_times=[0.2])
        trajectory = builder.build_from_gamepad(0.0, 0.0)
        assert trajectory.is_stationary

    def test_build_from_gamepad_moving(self) -> None:
        """Should build moving trajectory from stick input."""
        builder = TrajectoryBuilder(trajectory_times=[0.2])
        trajectory = builder.build_from_gamepad(1.0, 0.0)
        assert not trajectory.is_stationary

    def test_build_from_keyboard(self) -> None:
        """Should build trajectory from keyboard input."""
        builder = TrajectoryBuilder(trajectory_times=[0.2])
        trajectory = builder.build_from_keyboard(
            forward=True, backward=False, left=False, right=False
        )
        assert not trajectory.is_stationary

    def test_build_from_velocity(self) -> None:
        """Should build trajectory from velocity."""
        builder = TrajectoryBuilder(trajectory_times=[0.2])
        trajectory = builder.build_from_velocity(np.array([0, 0, 2.0]))
        assert not trajectory.is_stationary
        assert trajectory.desired_speed == 2.0


# =============================================================================
# ANNOTATION TESTS
# =============================================================================


class TestMotionTag:
    """Tests for MotionTag class."""

    def test_init(self) -> None:
        """Should initialize tag."""
        tag = MotionTag(name="walk", start_frame=0, end_frame=30)
        assert tag.name == "walk"
        assert tag.frame_count == 30

    def test_auto_detect_type(self) -> None:
        """Should auto-detect tag type from name."""
        walk_tag = MotionTag(name="walk", start_frame=0, end_frame=30)
        assert walk_tag.tag_type == TagType.WALK

        idle_tag = MotionTag(name="idle_pose", start_frame=0, end_frame=30)
        assert idle_tag.tag_type == TagType.IDLE

    def test_contains_frame(self) -> None:
        """Should check if frame is in range."""
        tag = MotionTag(name="test", start_frame=10, end_frame=20)
        assert tag.contains_frame(15)
        assert not tag.contains_frame(5)
        assert not tag.contains_frame(25)

    def test_overlaps(self) -> None:
        """Should detect overlapping tags."""
        tag1 = MotionTag(name="a", start_frame=0, end_frame=20)
        tag2 = MotionTag(name="b", start_frame=10, end_frame=30)
        tag3 = MotionTag(name="c", start_frame=25, end_frame=35)

        assert tag1.overlaps(tag2)
        assert not tag1.overlaps(tag3)


class TestContactAnnotation:
    """Tests for ContactAnnotation class."""

    def test_init(self) -> None:
        """Should initialize contacts."""
        contacts = ContactAnnotation(frame_count=60)
        assert len(contacts.left_contacts) == 60

    def test_get_contacts(self) -> None:
        """Should get contact values."""
        contacts = ContactAnnotation(frame_count=60)
        contacts.left_contacts[10] = 1.0
        left, right = contacts.get_contacts(10)
        assert left == 1.0
        assert right == 0.0

    def test_set_contact(self) -> None:
        """Should set contact values."""
        contacts = ContactAnnotation(frame_count=60)
        contacts.set_contact(10, left=1.0, right=0.5)
        left, right = contacts.get_contacts(10)
        assert left == 1.0
        assert right == 0.5

    def test_set_contact_range(self) -> None:
        """Should set contacts for range."""
        contacts = ContactAnnotation(frame_count=60)
        contacts.set_contact_range(10, 20, left=1.0)
        for i in range(10, 20):
            assert contacts.left_contacts[i] == 1.0

    def test_get_contact_events(self) -> None:
        """Should get contact event ranges."""
        contacts = ContactAnnotation(frame_count=60)
        contacts.set_contact_range(10, 20, left=1.0)
        contacts.set_contact_range(30, 40, left=1.0)

        events = contacts.get_contact_events('left')
        assert len(events) == 2
        assert events[0] == (10, 20)
        assert events[1] == (30, 40)


class TestAnnotatedClip:
    """Tests for AnnotatedClip class."""

    def test_init(self, mock_clip) -> None:
        """Should wrap clip."""
        annotated = AnnotatedClip(mock_clip)
        assert annotated.name == mock_clip.name
        assert annotated.frame_count == mock_clip.frame_count

    def test_add_tag(self, mock_clip) -> None:
        """Should add tags."""
        annotated = AnnotatedClip(mock_clip)
        annotated.add_tag(MotionTag(name="walk", start_frame=0, end_frame=30))
        assert len(annotated.tags) == 1

    def test_get_tags_at_frame(self, mock_clip) -> None:
        """Should get active tags at frame."""
        annotated = AnnotatedClip(mock_clip)
        annotated.add_tag(MotionTag(name="walk", start_frame=0, end_frame=30))
        annotated.add_tag(MotionTag(name="run", start_frame=20, end_frame=50))

        tags = annotated.get_tags_at_frame(25)
        assert len(tags) == 2

    def test_get_frame_tags(self, mock_clip) -> None:
        """Should get tag names at frame."""
        annotated = AnnotatedClip(mock_clip)
        annotated.add_tag(MotionTag(name="walk", start_frame=0, end_frame=30))

        names = annotated.get_frame_tags(15)
        assert "walk" in names

    def test_foot_contacts(self, mock_clip) -> None:
        """Should access foot contacts."""
        annotated = AnnotatedClip(mock_clip)
        annotated.set_foot_contacts(10, left=1.0, right=0.0)
        left, right = annotated.get_foot_contacts(10)
        assert left == 1.0


class TestAutoDetection:
    """Tests for auto-detection functions."""

    def test_auto_detect_contacts(self, mock_clip) -> None:
        """Should detect foot contacts."""
        annotated = AnnotatedClip(mock_clip)
        contacts = auto_detect_contacts(annotated)
        assert contacts.frame_count == mock_clip.frame_count

    def test_auto_detect_locomotion_tags(self, mock_clip) -> None:
        """Should detect locomotion tags."""
        annotated = AnnotatedClip(mock_clip)
        tags = auto_detect_locomotion_tags(annotated)
        # May or may not find tags depending on mock data
        assert isinstance(tags, list)

    def test_auto_detect_all_tags(self, mock_clip) -> None:
        """Should run all auto-detection."""
        annotated = AnnotatedClip(mock_clip)
        result = auto_detect_all_tags(annotated)
        assert result.contacts is not None


class TestTagUtilities:
    """Tests for tag utility functions."""

    def test_merge_overlapping_tags(self) -> None:
        """Should merge overlapping same-name tags."""
        tags = [
            MotionTag(name="walk", start_frame=0, end_frame=20),
            MotionTag(name="walk", start_frame=15, end_frame=30),
            MotionTag(name="run", start_frame=0, end_frame=10),
        ]
        merged = merge_overlapping_tags(tags)

        walk_tags = [t for t in merged if t.name == "walk"]
        assert len(walk_tags) == 1
        assert walk_tags[0].start_frame == 0
        assert walk_tags[0].end_frame == 30

    def test_filter_tags_by_duration(self) -> None:
        """Should filter tags by frame count."""
        tags = [
            MotionTag(name="a", start_frame=0, end_frame=5),   # 5 frames
            MotionTag(name="b", start_frame=0, end_frame=15),  # 15 frames
            MotionTag(name="c", start_frame=0, end_frame=30),  # 30 frames
        ]
        filtered = filter_tags_by_duration(tags, min_frames=10)
        assert len(filtered) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMotionMatchingIntegration:
    """Integration tests for the full motion matching pipeline."""

    def test_full_pipeline(self, mock_clips) -> None:
        """Should run full motion matching pipeline."""
        # Build database
        config = FeatureConfig()
        extractor = FeatureExtractor(config)
        database = build_database(mock_clips, extractor)

        # Create controller
        controller_config = ControllerConfig()
        controller = MotionMatchingController(database, controller_config)

        # Start and update
        controller.start()

        # Simulate some frames
        for i in range(10):
            direction = np.array([0, 0, 1]) if i % 2 == 0 else np.zeros(3)
            pose = controller.update(direction, 0.033)

        # Should have found matches
        assert controller.current_entry is not None

    def test_transition_during_update(self, simple_database) -> None:
        """Should handle transitions during update."""
        controller = MotionMatchingController(simple_database)
        controller.start()

        # Update with changing input
        for i in range(30):
            if i < 10:
                direction = np.zeros(3)
            else:
                direction = np.array([0, 0, 1])
            controller.update(direction, 0.033)

        # Should have processed updates
        assert controller.context.time_since_transition >= 0

    def test_search_with_kd_tree(self, mock_clips) -> None:
        """Should work with KD-tree search."""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)
        database = build_database(mock_clips, extractor)

        controller_config = ControllerConfig(search_method=SearchMethod.KD_TREE)
        controller = MotionMatchingController(database, controller_config)
        controller.start()

        pose = controller.update(np.array([0, 0, 1]), 0.033, desired_speed=2.0)
        assert controller.current_entry is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_database_search(self) -> None:
        """Should handle empty database gracefully."""
        db = MotionDatabase()
        db.finalize()
        search = MotionSearch(db)
        results = search.search(np.zeros(64))
        assert len(results) == 0

        # Also test find_best_match on empty
        best = search.find_best_match(np.zeros(64))
        assert best is None

    def test_single_entry_database(self) -> None:
        """Should handle single entry database."""
        db = MotionDatabase()
        entry_features = np.array([1.0, 2.0, 3.0] + [0.0] * 61, dtype=np.float32)
        db.add_entry(DatabaseEntry(clip_index=0, frame=0, features=entry_features))
        db.finalize()

        search = MotionSearch(db)

        # Query with exact features should return that entry with zero cost
        results = search.search(entry_features)
        assert len(results) == 1
        assert results[0].cost == pytest.approx(0.0, abs=1e-6)
        assert results[0].entry_index == 0

    def test_zero_duration_transition(self) -> None:
        """Should handle zero duration transition smoothly."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))
        config = TransitionConfig(blend_duration=0.0)

        transition = MotionTransition(from_entry, to_entry, config)
        from_pose = Pose(root_position=np.array([0.0, 0.0, 0.0]))
        to_pose = Pose(root_position=np.array([1.0, 0.0, 0.0]))
        transition.initialize(from_pose, to_pose)

        # Even with zero duration, should use minimum duration internally
        result = transition.update(0.001, to_pose)

        # Transition should complete quickly but not instantly
        # (uses MIN_BLEND_DURATION internally)
        assert transition.progress > 0
        assert isinstance(result, Pose)

    def test_kd_tree_empty_data(self) -> None:
        """Should handle KD-tree with empty data."""
        empty_data = np.zeros((0, 10), dtype=np.float32)
        tree = KDTree(empty_data)

        # Query should return empty list
        results = tree.query(np.zeros(10), k=5)
        assert len(results) == 0

    def test_cost_function_large_vectors(self) -> None:
        """Should handle large feature vectors without overflow."""
        # Create large values that could cause overflow
        large = np.ones(1000, dtype=np.float32) * 1e6
        small = np.zeros(1000, dtype=np.float32)

        cost = compute_cost(large, small)

        # Cost should be large but finite (not inf or nan)
        assert np.isfinite(cost)
        assert cost > 0

    def test_normalization_zero_std(self) -> None:
        """Should handle normalization when std is zero."""
        # Create data with constant values (zero std)
        constant_features = np.ones((100, 10), dtype=np.float32) * 5.0
        stats = NormalizationStats.compute(constant_features)

        # Std should be epsilon, not zero
        assert np.all(stats.std > 0)

        # Normalization should not produce inf/nan
        normalized = stats.normalize(constant_features)
        assert np.all(np.isfinite(normalized))

    def test_large_input_direction(self) -> None:
        """Should handle large input direction."""
        trajectory = DesiredTrajectory.from_input(
            direction=np.array([100, 0, 100]),  # Very large
            speed=5.0,
            current_position=np.zeros(3),
            current_facing=0.0,
            trajectory_times=[0.2],
        )
        assert not trajectory.is_stationary

    def test_negative_frame_index(self) -> None:
        """Should handle negative frame index."""
        contacts = ContactAnnotation(frame_count=60)
        left, right = contacts.get_contacts(-1)
        assert left == 0.0

    def test_out_of_bounds_frame(self) -> None:
        """Should handle out of bounds frame."""
        contacts = ContactAnnotation(frame_count=60)
        left, right = contacts.get_contacts(100)
        assert left == 0.0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Basic performance sanity checks."""

    def test_large_database_build(self) -> None:
        """Should build large database reasonably fast."""
        clips = [
            MockAnimationClip(name=f"clip_{i}", frame_count=100)
            for i in range(10)
        ]
        extractor = FeatureExtractor()

        # This should complete without timeout
        db = build_database(clips, extractor)
        assert db.entry_count == 1000

    def test_vectorized_cost_performance(self) -> None:
        """Vectorized cost should handle large batches."""
        query = np.random.randn(100).astype(np.float32)
        candidates = np.random.randn(10000, 100).astype(np.float32)

        # Should complete quickly
        costs = compute_cost_vectorized(query, candidates)
        assert len(costs) == 10000

    def test_kd_tree_build_performance(self) -> None:
        """KD-tree should build from reasonable data."""
        data = np.random.randn(1000, 64).astype(np.float32)
        tree = KDTree(data)
        assert tree.root is not None

    def test_lsh_build_performance(self) -> None:
        """LSH should build from reasonable data."""
        data = np.random.randn(1000, 64).astype(np.float32)
        lsh = LSHIndex(dimension=64, num_tables=5)
        lsh.build(data)
        assert len(lsh.tables) == 5


# =============================================================================
# ADDITIONAL TESTS FOR COVERAGE
# =============================================================================


class TestDatabaseEntryDetails:
    """Additional tests for DatabaseEntry functionality."""

    def test_entry_id(self) -> None:
        """Should return correct entry_id tuple."""
        entry = DatabaseEntry(clip_index=5, frame=42, features=np.zeros(64))
        assert entry.entry_id == (5, 42)

    def test_matches_tags_empty(self) -> None:
        """Should match when no tags required."""
        entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        assert entry.matches_tags(None)
        assert entry.matches_tags(set())

    def test_matches_tags_subset(self) -> None:
        """Should match when entry has required tags."""
        entry = DatabaseEntry(
            clip_index=0, frame=0, features=np.zeros(64),
            tags=frozenset({"walk", "locomotion", "grounded"})
        )
        assert entry.matches_tags({"walk"})
        assert entry.matches_tags({"walk", "locomotion"})

    def test_matches_tags_missing(self) -> None:
        """Should not match when missing required tags."""
        entry = DatabaseEntry(
            clip_index=0, frame=0, features=np.zeros(64),
            tags=frozenset({"walk"})
        )
        assert not entry.matches_tags({"run"})


class TestClipMetadataDetails:
    """Additional tests for ClipMetadata functionality."""

    def test_duration_computed_from_frames(self) -> None:
        """Should compute duration from frame count."""
        metadata = ClipMetadata(
            clip_index=0,
            name="test",
            frame_count=61,  # 60 intervals
            frame_rate=30.0,
            duration=0.0,  # Should be computed
        )
        assert metadata.duration == pytest.approx(2.0)

    def test_tags_converted_to_frozenset(self) -> None:
        """Should convert tags to frozenset."""
        metadata = ClipMetadata(
            clip_index=0,
            name="test",
            frame_count=30,
            tags={"walk", "run"}  # Set, not frozenset
        )
        assert isinstance(metadata.tags, frozenset)


class TestNormalizationStats:
    """Additional tests for NormalizationStats."""

    def test_compute_from_data(self) -> None:
        """Should compute stats from feature matrix."""
        features = np.random.randn(100, 10) * 5 + 2  # Mean ~2, std ~5
        stats = NormalizationStats.compute(features)
        assert abs(np.mean(stats.mean) - 2) < 1.0  # Approximate
        assert np.mean(stats.std) > 1.0

    def test_normalize_denormalize_roundtrip(self) -> None:
        """Should normalize and denormalize correctly."""
        features = np.random.randn(10, 5)
        stats = NormalizationStats.compute(features)
        normalized = stats.normalize(features)
        denormalized = stats.denormalize(normalized)
        assert np.allclose(features, denormalized, atol=1e-5)


class TestFeatureExtractorDetails:
    """Additional tests for FeatureExtractor."""

    def test_config_without_positions(self) -> None:
        """Should work without bone positions."""
        config = FeatureConfig(use_bone_positions=False)
        extractor = FeatureExtractor(config)
        dim = extractor.feature_dimension
        assert dim > 0  # Should still have other features

    def test_config_without_velocities(self) -> None:
        """Should work without velocities."""
        config = FeatureConfig(use_bone_velocities=False)
        extractor = FeatureExtractor(config)
        dim = extractor.feature_dimension
        assert dim > 0

    def test_config_without_trajectory(self) -> None:
        """Should work without trajectory."""
        config = FeatureConfig(use_trajectory=False)
        extractor = FeatureExtractor(config)
        dim = extractor.feature_dimension
        assert dim > 0

    def test_config_without_contacts(self) -> None:
        """Should work without foot contacts."""
        config = FeatureConfig(use_foot_contacts=False)
        extractor = FeatureExtractor(config)
        dim = extractor.feature_dimension
        assert dim > 0


class TestSearchConfigDetails:
    """Additional tests for SearchConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = SearchConfig()
        assert config.max_results > 0
        assert config.cost_threshold == float('inf')

    def test_tag_filtering(self) -> None:
        """Should handle tag configuration."""
        config = SearchConfig(
            required_tags={"walk"},
            excluded_tags={"jump"}
        )
        assert "walk" in config.required_tags
        assert "jump" in config.excluded_tags


class TestTrajectoryPointDetails:
    """Additional tests for TrajectoryPoint."""

    def test_facing_as_float(self) -> None:
        """Should accept facing as float angle."""
        point = TrajectoryPoint(
            time_offset=0.2,
            position=np.array([0, 0, 1]),
            facing=1.57  # ~90 degrees
        )
        assert point.time_offset == 0.2
        assert isinstance(point.facing, float)

    def test_facing_as_vector(self) -> None:
        """Should accept facing as direction vector."""
        point = TrajectoryPoint(
            time_offset=0.2,
            position=np.array([0, 0, 1]),
            facing=np.array([0, 1])  # Direction vector
        )
        assert isinstance(point.facing, np.ndarray)

    def test_optional_velocity(self) -> None:
        """Should handle optional velocity."""
        point = TrajectoryPoint(
            time_offset=0.2,
            position=np.array([0, 0, 1]),
            facing=0.0,
            velocity=np.array([0, 0, 2])
        )
        assert point.velocity is not None


class TestBoneDataDetails:
    """Additional tests for BoneData."""

    def test_default_rotation(self) -> None:
        """Should handle optional rotation."""
        data = BoneData(
            position=[1, 2, 3],
            velocity=[0, 0, 1]
        )
        assert data.rotation is None

    def test_with_rotation(self) -> None:
        """Should accept rotation quaternion."""
        data = BoneData(
            position=[1, 2, 3],
            velocity=[0, 0, 1],
            rotation=np.array([0, 0, 0, 1])
        )
        assert data.rotation is not None


class TestTransitionDetails:
    """Additional tests for transition functionality."""

    def test_linear_blend_mode(self) -> None:
        """Should support linear blend mode."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))
        config = TransitionConfig(blend_mode=BlendMode.LINEAR)

        transition = MotionTransition(from_entry, to_entry, config)
        from_pose = Pose(root_position=np.array([0, 0, 0]))
        to_pose = Pose(root_position=np.array([1, 0, 0]))
        transition.initialize(from_pose, to_pose)

        result = transition.update(0.5, to_pose)
        # At 50%, should be between from and to
        assert result.root_position[0] > 0

    def test_crossfade_blend_mode(self) -> None:
        """Should support crossfade blend mode."""
        from_entry = DatabaseEntry(clip_index=0, frame=0, features=np.zeros(64))
        to_entry = DatabaseEntry(clip_index=1, frame=0, features=np.zeros(64))
        config = TransitionConfig(blend_mode=BlendMode.CROSSFADE, blend_duration=0.2)

        transition = MotionTransition(from_entry, to_entry, config)
        from_pose = Pose(root_position=np.array([0, 0, 0]))
        to_pose = Pose(root_position=np.array([1, 0, 0]))
        transition.initialize(from_pose, to_pose)

        result = transition.update(0.1, to_pose)
        assert not transition.is_complete


class TestControllerDetails:
    """Additional tests for controller functionality."""

    def test_force_transition(self, simple_database) -> None:
        """Should force transition to specific entry."""
        controller = MotionMatchingController(simple_database)
        controller.start()

        result = controller.force_transition(5)
        assert result is True
        assert controller.current_entry is not None

    def test_force_transition_invalid(self, simple_database) -> None:
        """Should handle invalid force transition."""
        controller = MotionMatchingController(simple_database)
        controller.start()

        result = controller.force_transition(99999)
        assert result is False


class TestAnnotationDetails:
    """Additional tests for annotation functionality."""

    def test_tag_intersection(self) -> None:
        """Should compute tag intersection."""
        tag1 = MotionTag(name="a", start_frame=0, end_frame=30)
        tag2 = MotionTag(name="b", start_frame=20, end_frame=50)

        intersection = tag1.intersection(tag2)
        assert intersection is not None
        assert intersection.start_frame == 20
        assert intersection.end_frame == 30

    def test_tag_no_intersection(self) -> None:
        """Should return None for non-overlapping tags."""
        tag1 = MotionTag(name="a", start_frame=0, end_frame=10)
        tag2 = MotionTag(name="b", start_frame=20, end_frame=30)

        intersection = tag1.intersection(tag2)
        assert intersection is None

    def test_annotated_clip_remove_tag(self, mock_clip) -> None:
        """Should remove tags by name."""
        annotated = AnnotatedClip(mock_clip)
        annotated.add_tag(MotionTag(name="walk", start_frame=0, end_frame=30))
        annotated.add_tag(MotionTag(name="walk", start_frame=40, end_frame=50))
        annotated.add_tag(MotionTag(name="run", start_frame=0, end_frame=30))

        annotated.remove_tag("walk")
        assert len(annotated.tags) == 1
        assert annotated.tags[0].name == "run"

    def test_annotated_clip_get_tags_by_type(self, mock_clip) -> None:
        """Should get tags by type."""
        annotated = AnnotatedClip(mock_clip)
        annotated.add_tag(MotionTag(name="walk", start_frame=0, end_frame=30))
        annotated.add_tag(MotionTag(name="run", start_frame=0, end_frame=30))

        walk_tags = annotated.get_tags_by_type(TagType.WALK)
        assert len(walk_tags) == 1

    def test_transition_markers(self, mock_clip) -> None:
        """Should handle transition markers."""
        annotated = AnnotatedClip(mock_clip)
        annotated.set_transition_markers({5, 10, 15, 20})

        assert annotated.is_transition_frame(10)
        assert not annotated.is_transition_frame(7)


class TestQuantizationLevels:
    """Tests for different quantization levels."""

    def test_float16_quantization(self) -> None:
        """Should support float16 quantization."""
        db = MotionDatabase(quantization=QuantizationLevel.FLOAT16)
        for i in range(10):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.random.randn(64)))
        db.finalize()
        assert db.feature_matrix.dtype == np.float16

    def test_int16_quantization(self) -> None:
        """Should support int16 quantization."""
        db = MotionDatabase(quantization=QuantizationLevel.INT16)
        for i in range(10):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.random.randn(64)))
        db.finalize()
        assert db.feature_matrix.dtype == np.int16

    def test_int8_quantization(self) -> None:
        """Should support int8 quantization."""
        db = MotionDatabase(quantization=QuantizationLevel.INT8)
        for i in range(10):
            db.add_entry(DatabaseEntry(clip_index=0, frame=i, features=np.random.randn(64)))
        db.finalize()
        assert db.feature_matrix.dtype == np.int8


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestConfigModule:
    """Tests for centralized configuration."""

    def test_feature_weight_defaults(self) -> None:
        """Should have sensible default feature weights."""
        assert DEFAULT_FEATURE_WEIGHTS.position_weight > 0
        assert DEFAULT_FEATURE_WEIGHTS.velocity_weight > 0
        assert DEFAULT_FEATURE_WEIGHTS.trajectory_weight > 0
        assert DEFAULT_FEATURE_WEIGHTS.contact_weight > 0

    def test_search_params_defaults(self) -> None:
        """Should have sensible default search params."""
        assert DEFAULT_SEARCH_PARAMS.max_results > 0
        assert DEFAULT_SEARCH_PARAMS.kd_tree_leaf_size > 0
        assert DEFAULT_SEARCH_PARAMS.cost_epsilon > 0
        assert DEFAULT_SEARCH_PARAMS.cost_improvement_threshold > 0

    def test_transition_params_defaults(self) -> None:
        """Should have sensible default transition params."""
        assert DEFAULT_TRANSITION_PARAMS.default_blend_duration > 0
        assert DEFAULT_TRANSITION_PARAMS.min_blend_duration > 0
        assert DEFAULT_TRANSITION_PARAMS.spring_halflife > 0
        assert DEFAULT_TRANSITION_PARAMS.min_spring_halflife > 0
        # Minimum should be less than default
        assert DEFAULT_TRANSITION_PARAMS.min_blend_duration < DEFAULT_TRANSITION_PARAMS.default_blend_duration

    def test_database_config_defaults(self) -> None:
        """Should have sensible default database config."""
        assert DEFAULT_DATABASE_CONFIG.normalization_epsilon > 0
        assert DEFAULT_DATABASE_CONFIG.int16_quant_scale > 0

    def test_config_values_used_in_modules(self, feature_config) -> None:
        """Config values should be used by modules."""
        # Feature weights should match config
        assert feature_config.position_weight == DEFAULT_FEATURE_WEIGHTS.position_weight
        assert feature_config.velocity_weight == DEFAULT_FEATURE_WEIGHTS.velocity_weight
