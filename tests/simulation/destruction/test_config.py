"""
Tests for Destruction System Configuration.

Whitebox tests for config.py including:
- Configuration constants validation
- FractureConfig dataclass validation
- DebrisConfig dataclass validation
- DamageConfig dataclass validation
- SupportConfig dataclass validation
- DestructionSystemConfig initialization
"""

import pytest

from engine.simulation.destruction.config import (
    # Constants
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    MIN_VORONOI_SITES,
    MAX_VORONOI_SITES,
    DEFAULT_VORONOI_SITES,
    DEBRIS_LIFETIME,
    DEBRIS_MIN_LIFETIME,
    DEBRIS_MAX_LIFETIME,
    MAX_ACTIVE_DEBRIS,
    DAMAGE_PROPAGATION_FACTOR,
    SUPPORT_STRESS_THRESHOLD,
    # Enums
    FracturePattern,
    DebrisState,
    SupportType,
    # Config dataclasses
    FractureConfig,
    DebrisConfig,
    DamageConfig,
    SupportConfig,
    DestructionSystemConfig,
    DEFAULT_CONFIG,
)


class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_fracture_seed_is_deterministic(self):
        """Verify default seed is a known value for reproducibility."""
        assert DEFAULT_FRACTURE_SEED == 42

    def test_chunk_volume_is_positive(self):
        """Verify minimum chunk volume is positive."""
        assert MIN_CHUNK_VOLUME > 0

    def test_max_chunks_is_reasonable(self):
        """Verify max chunks limit is reasonable."""
        assert MAX_CHUNKS_PER_OBJECT > 0
        assert MAX_CHUNKS_PER_OBJECT <= 256

    def test_voronoi_sites_range_is_valid(self):
        """Verify Voronoi site limits are valid."""
        assert MIN_VORONOI_SITES >= 4  # Need at least 4 for 3D
        assert MAX_VORONOI_SITES > MIN_VORONOI_SITES
        assert MIN_VORONOI_SITES <= DEFAULT_VORONOI_SITES <= MAX_VORONOI_SITES

    def test_debris_lifetime_range(self):
        """Verify debris lifetime bounds are valid."""
        assert DEBRIS_MIN_LIFETIME > 0
        assert DEBRIS_MAX_LIFETIME > DEBRIS_MIN_LIFETIME
        assert DEBRIS_MIN_LIFETIME <= DEBRIS_LIFETIME <= DEBRIS_MAX_LIFETIME

    def test_max_active_debris_is_reasonable(self):
        """Verify debris limit is reasonable for performance."""
        assert MAX_ACTIVE_DEBRIS > 0
        assert MAX_ACTIVE_DEBRIS <= 10000

    def test_damage_propagation_factor_in_range(self):
        """Verify damage propagation is normalized."""
        assert 0.0 <= DAMAGE_PROPAGATION_FACTOR <= 1.0

    def test_stress_threshold_is_positive(self):
        """Verify stress threshold is positive."""
        assert SUPPORT_STRESS_THRESHOLD > 0


class TestFracturePatternEnum:
    """Tests for FracturePattern enumeration."""

    def test_all_patterns_defined(self):
        """Verify all expected patterns exist."""
        assert hasattr(FracturePattern, 'VORONOI')
        assert hasattr(FracturePattern, 'RADIAL')
        assert hasattr(FracturePattern, 'SLICE')
        assert hasattr(FracturePattern, 'CUSTOM')

    def test_patterns_are_unique(self):
        """Verify pattern values are unique."""
        values = [p.value for p in FracturePattern]
        assert len(values) == len(set(values))


class TestDebrisStateEnum:
    """Tests for DebrisState enumeration."""

    def test_all_states_defined(self):
        """Verify all expected states exist."""
        assert hasattr(DebrisState, 'ACTIVE')
        assert hasattr(DebrisState, 'SLEEPING')
        assert hasattr(DebrisState, 'PENDING_CLEANUP')
        assert hasattr(DebrisState, 'POOLED')

    def test_states_are_unique(self):
        """Verify state values are unique."""
        values = [s.value for s in DebrisState]
        assert len(values) == len(set(values))


class TestSupportTypeEnum:
    """Tests for SupportType enumeration."""

    def test_all_types_defined(self):
        """Verify all support types exist."""
        assert hasattr(SupportType, 'FIXED')
        assert hasattr(SupportType, 'STRUCTURAL')
        assert hasattr(SupportType, 'TEMPORARY')


class TestFractureConfig:
    """Tests for FractureConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = FractureConfig()
        assert config.pattern == FracturePattern.VORONOI
        assert config.seed == DEFAULT_FRACTURE_SEED
        assert config.max_chunks == MAX_CHUNKS_PER_OBJECT
        assert config.min_chunk_volume == MIN_CHUNK_VOLUME
        assert config.preserve_surface is True
        assert config.generate_interior_uvs is True

    def test_custom_values(self):
        """Verify custom configuration values."""
        config = FractureConfig(
            pattern=FracturePattern.RADIAL,
            seed=123,
            max_chunks=32,
            num_sites=20
        )
        assert config.pattern == FracturePattern.RADIAL
        assert config.seed == 123
        assert config.max_chunks == 32
        assert config.num_sites == 20

    def test_max_chunks_validation(self):
        """Verify max_chunks must be >= 1."""
        with pytest.raises(ValueError, match="max_chunks must be >= 1"):
            FractureConfig(max_chunks=0)

        with pytest.raises(ValueError, match="max_chunks must be >= 1"):
            FractureConfig(max_chunks=-1)

    def test_min_chunk_volume_validation(self):
        """Verify min_chunk_volume must be > 0."""
        with pytest.raises(ValueError, match="min_chunk_volume must be > 0"):
            FractureConfig(min_chunk_volume=0)

        with pytest.raises(ValueError, match="min_chunk_volume must be > 0"):
            FractureConfig(min_chunk_volume=-0.1)

    def test_num_sites_validation(self):
        """Verify num_sites must be >= MIN_VORONOI_SITES."""
        with pytest.raises(ValueError, match=f"num_sites must be >= {MIN_VORONOI_SITES}"):
            FractureConfig(num_sites=2)

    def test_num_slices_validation(self):
        """Verify num_slices must meet minimum."""
        with pytest.raises(ValueError, match="num_slices must be >= 4"):
            FractureConfig(num_slices=2)

    def test_config_is_frozen(self):
        """Verify config is immutable after creation."""
        config = FractureConfig()
        with pytest.raises(AttributeError):
            config.seed = 999


class TestDebrisConfig:
    """Tests for DebrisConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = DebrisConfig()
        assert config.lifetime == DEBRIS_LIFETIME
        assert config.max_active == MAX_ACTIVE_DEBRIS

    def test_lifetime_min_validation(self):
        """Verify lifetime must be >= minimum."""
        with pytest.raises(ValueError, match=f"lifetime must be >= {DEBRIS_MIN_LIFETIME}"):
            DebrisConfig(lifetime=0.1)

    def test_lifetime_max_validation(self):
        """Verify lifetime must be <= maximum."""
        with pytest.raises(ValueError, match=f"lifetime must be <= {DEBRIS_MAX_LIFETIME}"):
            DebrisConfig(lifetime=1000.0)

    def test_max_active_validation(self):
        """Verify max_active must be >= 1."""
        with pytest.raises(ValueError, match="max_active must be >= 1"):
            DebrisConfig(max_active=0)

    def test_config_is_frozen(self):
        """Verify config is immutable."""
        config = DebrisConfig()
        with pytest.raises(AttributeError):
            config.lifetime = 5.0


class TestDamageConfig:
    """Tests for DamageConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = DamageConfig()
        assert config.propagation_factor == DAMAGE_PROPAGATION_FACTOR
        assert config.min_threshold >= 0
        assert config.accumulation_rate >= 0
        assert config.decay_rate >= 0

    def test_propagation_factor_validation_too_low(self):
        """Verify propagation_factor must be >= 0."""
        with pytest.raises(ValueError, match="propagation_factor must be in"):
            DamageConfig(propagation_factor=-0.1)

    def test_propagation_factor_validation_too_high(self):
        """Verify propagation_factor must be <= 1."""
        with pytest.raises(ValueError, match="propagation_factor must be in"):
            DamageConfig(propagation_factor=1.5)

    def test_min_threshold_validation(self):
        """Verify min_threshold must be >= 0."""
        with pytest.raises(ValueError, match="min_threshold must be >= 0"):
            DamageConfig(min_threshold=-1.0)

    def test_edge_propagation_factor_values(self):
        """Verify edge values are accepted."""
        config_zero = DamageConfig(propagation_factor=0.0)
        assert config_zero.propagation_factor == 0.0

        config_one = DamageConfig(propagation_factor=1.0)
        assert config_one.propagation_factor == 1.0


class TestSupportConfig:
    """Tests for SupportConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = SupportConfig()
        assert config.stress_threshold == SUPPORT_STRESS_THRESHOLD
        assert config.max_connections >= 1
        assert config.min_contact_area > 0
        assert 0 < config.propagation_rate <= 1

    def test_stress_threshold_validation(self):
        """Verify stress_threshold must be > 0."""
        with pytest.raises(ValueError, match="stress_threshold must be > 0"):
            SupportConfig(stress_threshold=0)

        with pytest.raises(ValueError, match="stress_threshold must be > 0"):
            SupportConfig(stress_threshold=-100)

    def test_max_connections_validation(self):
        """Verify max_connections must be >= 1."""
        with pytest.raises(ValueError, match="max_connections must be >= 1"):
            SupportConfig(max_connections=0)


class TestDestructionSystemConfig:
    """Tests for DestructionSystemConfig master config."""

    def test_default_initialization(self):
        """Verify default sub-configs are created."""
        config = DestructionSystemConfig()
        assert config.fracture is not None
        assert config.debris is not None
        assert config.damage is not None
        assert config.support is not None
        assert isinstance(config.fracture, FractureConfig)
        assert isinstance(config.debris, DebrisConfig)
        assert isinstance(config.damage, DamageConfig)
        assert isinstance(config.support, SupportConfig)

    def test_custom_sub_configs(self):
        """Verify custom sub-configs are preserved."""
        custom_fracture = FractureConfig(seed=999)
        config = DestructionSystemConfig(fracture=custom_fracture)
        assert config.fracture.seed == 999

    def test_default_config_singleton_like(self):
        """Verify DEFAULT_CONFIG is valid."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, DestructionSystemConfig)
        assert DEFAULT_CONFIG.fracture is not None
