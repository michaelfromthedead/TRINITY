"""
Whitebox tests for engine.simulation.collision.config module.

Tests configuration constants, quality presets, and CollisionConfig dataclass.
"""

import pytest
from engine.simulation.collision.config import (
    BROADPHASE_MARGIN,
    CONTACT_TOLERANCE,
    MAX_CONTACT_POINTS,
    CCD_THRESHOLD_VELOCITY,
    MAX_CCD_ITERATIONS,
    SPATIAL_HASH_CELL_SIZE,
    GJK_MAX_ITERATIONS,
    EPA_MAX_ITERATIONS,
    EPA_TOLERANCE,
    CONTACT_MAX_AGE,
    WARM_START_FACTOR,
    NUMERICAL_EPSILON,
    PARALLEL_THRESHOLD,
    CCD_SAFETY_FACTOR,
    OCTREE_MAX_DEPTH,
    OCTREE_MAX_OBJECTS_PER_LEAF,
    BVH_REBALANCE_THRESHOLD,
    SAT_EDGE_BIAS,
    CCD_TIME_STEP_FRACTION,
    CCD_SPECULATIVE_MARGIN,
    CCD_MIN_TOI,
    CONTACT_MATCH_THRESHOLD,
    NUM_COLLISION_LAYERS,
    INITIAL_PAIR_CAPACITY,
    SAP_PRIMARY_AXIS,
    NARROWPHASE_BATCH_SIZE,
    CollisionQuality,
    CollisionConfig,
    DEFAULT_CONFIG,
)


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_broadphase_margin_positive(self):
        """Broadphase margin must be positive."""
        assert BROADPHASE_MARGIN > 0

    def test_contact_tolerance_positive(self):
        """Contact tolerance must be positive."""
        assert CONTACT_TOLERANCE > 0

    def test_max_contact_points_reasonable(self):
        """Max contact points must be in reasonable range."""
        assert 1 <= MAX_CONTACT_POINTS <= 16

    def test_ccd_threshold_velocity_positive(self):
        """CCD threshold velocity must be positive."""
        assert CCD_THRESHOLD_VELOCITY > 0

    def test_max_ccd_iterations_positive(self):
        """Max CCD iterations must be positive."""
        assert MAX_CCD_ITERATIONS > 0

    def test_spatial_hash_cell_size_positive(self):
        """Spatial hash cell size must be positive."""
        assert SPATIAL_HASH_CELL_SIZE > 0

    def test_gjk_max_iterations_positive(self):
        """GJK max iterations must be positive."""
        assert GJK_MAX_ITERATIONS > 0

    def test_epa_max_iterations_positive(self):
        """EPA max iterations must be positive."""
        assert EPA_MAX_ITERATIONS > 0

    def test_epa_tolerance_positive(self):
        """EPA tolerance must be positive."""
        assert EPA_TOLERANCE > 0

    def test_contact_max_age_positive(self):
        """Contact max age must be positive."""
        assert CONTACT_MAX_AGE > 0

    def test_warm_start_factor_range(self):
        """Warm start factor must be in [0, 1]."""
        assert 0 <= WARM_START_FACTOR <= 1

    def test_numerical_epsilon_very_small(self):
        """Numerical epsilon must be very small."""
        assert 0 < NUMERICAL_EPSILON < 1e-5

    def test_parallel_threshold_range(self):
        """Parallel threshold must be in (0, 1)."""
        assert 0 < PARALLEL_THRESHOLD < 1

    def test_ccd_safety_factor_range(self):
        """CCD safety factor must be in (0, 1]."""
        assert 0 < CCD_SAFETY_FACTOR <= 1

    def test_octree_max_depth_positive(self):
        """Octree max depth must be positive."""
        assert OCTREE_MAX_DEPTH > 0

    def test_octree_max_objects_per_leaf_positive(self):
        """Octree max objects per leaf must be positive."""
        assert OCTREE_MAX_OBJECTS_PER_LEAF > 0

    def test_bvh_rebalance_threshold_range(self):
        """BVH rebalance threshold must be in (0, 1)."""
        assert 0 < BVH_REBALANCE_THRESHOLD < 1

    def test_sat_edge_bias_very_small(self):
        """SAT edge bias must be very small."""
        assert 0 < SAT_EDGE_BIAS < 0.001

    def test_ccd_time_step_fraction_range(self):
        """CCD time step fraction must be in (0, 1)."""
        assert 0 < CCD_TIME_STEP_FRACTION < 1

    def test_ccd_speculative_margin_positive(self):
        """CCD speculative margin must be positive."""
        assert CCD_SPECULATIVE_MARGIN > 0

    def test_ccd_min_toi_positive(self):
        """CCD min TOI must be positive."""
        assert CCD_MIN_TOI > 0

    def test_contact_match_threshold_positive(self):
        """Contact match threshold must be positive."""
        assert CONTACT_MATCH_THRESHOLD > 0

    def test_num_collision_layers_32(self):
        """Number of collision layers must be 32."""
        assert NUM_COLLISION_LAYERS == 32

    def test_initial_pair_capacity_positive(self):
        """Initial pair capacity must be positive."""
        assert INITIAL_PAIR_CAPACITY > 0

    def test_sap_primary_axis_valid(self):
        """SAP primary axis must be 0, 1, or 2."""
        assert SAP_PRIMARY_AXIS in (0, 1, 2)

    def test_narrowphase_batch_size_positive(self):
        """Narrowphase batch size must be positive."""
        assert NARROWPHASE_BATCH_SIZE > 0


class TestCollisionQuality:
    """Tests for CollisionQuality enum."""

    def test_all_quality_levels_exist(self):
        """All quality levels must be defined."""
        assert hasattr(CollisionQuality, "LOW")
        assert hasattr(CollisionQuality, "MEDIUM")
        assert hasattr(CollisionQuality, "HIGH")
        assert hasattr(CollisionQuality, "ULTRA")

    def test_quality_levels_unique(self):
        """Quality levels must have unique values."""
        values = [q.value for q in CollisionQuality]
        assert len(values) == len(set(values))


class TestCollisionConfig:
    """Tests for CollisionConfig dataclass."""

    def test_default_config_creation(self):
        """Default config should use default values."""
        config = CollisionConfig()
        assert config.broadphase_margin == BROADPHASE_MARGIN
        assert config.contact_tolerance == CONTACT_TOLERANCE
        assert config.max_contact_points == MAX_CONTACT_POINTS

    def test_custom_config_creation(self):
        """Custom config should use specified values."""
        config = CollisionConfig(
            broadphase_margin=0.1,
            contact_tolerance=0.02,
            max_contact_points=8,
        )
        assert config.broadphase_margin == 0.1
        assert config.contact_tolerance == 0.02
        assert config.max_contact_points == 8

    def test_config_is_frozen(self):
        """Config should be immutable."""
        config = CollisionConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.broadphase_margin = 0.5

    def test_from_quality_low(self):
        """LOW quality config should have relaxed values."""
        config = CollisionConfig.from_quality(CollisionQuality.LOW)
        assert config.broadphase_margin > BROADPHASE_MARGIN
        assert config.contact_tolerance > CONTACT_TOLERANCE
        assert config.max_contact_points < MAX_CONTACT_POINTS
        assert config.gjk_max_iterations < GJK_MAX_ITERATIONS

    def test_from_quality_medium(self):
        """MEDIUM quality config should use defaults."""
        config = CollisionConfig.from_quality(CollisionQuality.MEDIUM)
        assert config.broadphase_margin == BROADPHASE_MARGIN
        assert config.contact_tolerance == CONTACT_TOLERANCE

    def test_from_quality_high(self):
        """HIGH quality config should have tighter values."""
        config = CollisionConfig.from_quality(CollisionQuality.HIGH)
        assert config.broadphase_margin < BROADPHASE_MARGIN
        assert config.contact_tolerance < CONTACT_TOLERANCE
        assert config.gjk_max_iterations > GJK_MAX_ITERATIONS

    def test_from_quality_ultra(self):
        """ULTRA quality config should have most precise values."""
        config = CollisionConfig.from_quality(CollisionQuality.ULTRA)
        assert config.broadphase_margin < 0.02
        assert config.contact_tolerance < 0.005
        assert config.gjk_max_iterations >= 256


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG instance."""

    def test_default_config_exists(self):
        """DEFAULT_CONFIG must be defined."""
        assert DEFAULT_CONFIG is not None

    def test_default_config_is_collision_config(self):
        """DEFAULT_CONFIG must be a CollisionConfig instance."""
        assert isinstance(DEFAULT_CONFIG, CollisionConfig)

    def test_default_config_uses_defaults(self):
        """DEFAULT_CONFIG should use default values."""
        assert DEFAULT_CONFIG.broadphase_margin == BROADPHASE_MARGIN
        assert DEFAULT_CONFIG.gjk_max_iterations == GJK_MAX_ITERATIONS


class TestConfigConsistency:
    """Tests for configuration consistency."""

    def test_contact_tolerance_less_than_broadphase_margin(self):
        """Contact tolerance should be smaller than broadphase margin."""
        assert CONTACT_TOLERANCE < BROADPHASE_MARGIN

    def test_epa_tolerance_less_than_contact_tolerance(self):
        """EPA tolerance should be smaller than contact tolerance."""
        assert EPA_TOLERANCE < CONTACT_TOLERANCE

    def test_contact_match_threshold_greater_than_contact_tolerance(self):
        """Contact match threshold should be >= contact tolerance."""
        assert CONTACT_MATCH_THRESHOLD >= CONTACT_TOLERANCE

    def test_ccd_min_toi_less_than_time_step_fraction(self):
        """CCD min TOI should be less than time step fraction."""
        assert CCD_MIN_TOI < CCD_TIME_STEP_FRACTION
