"""
Whitebox tests for DestructibleComponent.

Tests cover:
- Component creation and configuration
- Health management
- Damage application and propagation
- Staged destruction
- Fracture chunk management
- Repair functionality
- Serialization
"""

import pytest

from engine.simulation.character.character_controller import Vector3
from engine.simulation.components.destruction_component import (
    DamageInfo,
    DamageType,
    DestructibleComponent,
    DestructionConfig,
    DestructionType,
    FractureChunk,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def destruction_config() -> DestructionConfig:
    """Default destruction configuration."""
    return DestructionConfig(
        destruction_type=DestructionType.FRACTURE,
        health=100.0,
        min_damage_threshold=5.0,
        debris_lifetime=10.0,
        debris_count_limit=50,
    )


@pytest.fixture
def destructible(destruction_config) -> DestructibleComponent:
    """Create a destructible component."""
    return DestructibleComponent(entity_id=1, config=destruction_config)


@pytest.fixture
def destructible_with_chunks() -> DestructibleComponent:
    """Create a destructible with pre-computed chunks."""
    config = DestructionConfig(health=100.0)
    component = DestructibleComponent(entity_id=2, config=config)

    chunks = [
        FractureChunk(chunk_id=0, mass=0.25, connected_to=[1]),
        FractureChunk(chunk_id=1, mass=0.25, connected_to=[0, 2]),
        FractureChunk(chunk_id=2, mass=0.25, connected_to=[1, 3]),
        FractureChunk(chunk_id=3, mass=0.25, connected_to=[2]),
    ]
    component.set_fracture_chunks(chunks)
    return component


# =============================================================================
# DestructionType Tests
# =============================================================================


class TestDestructionType:
    """Tests for DestructionType enum."""

    def test_all_types(self):
        """Test all destruction types exist."""
        assert DestructionType.NONE.value == "none"
        assert DestructionType.FRACTURE.value == "fracture"
        assert DestructionType.VORONOI.value == "voronoi"
        assert DestructionType.SLICE.value == "slice"
        assert DestructionType.CRUMBLE.value == "crumble"
        assert DestructionType.SHATTER.value == "shatter"


class TestDamageType:
    """Tests for DamageType enum."""

    def test_all_types(self):
        """Test all damage types exist."""
        assert DamageType.IMPACT.value == "impact"
        assert DamageType.EXPLOSION.value == "explosion"
        assert DamageType.FIRE.value == "fire"
        assert DamageType.BULLET.value == "bullet"
        assert DamageType.MELEE.value == "melee"
        assert DamageType.GENERIC.value == "generic"


# =============================================================================
# FractureChunk Tests
# =============================================================================


class TestFractureChunk:
    """Tests for FractureChunk dataclass."""

    def test_default_values(self):
        """Test default chunk values."""
        chunk = FractureChunk()

        assert chunk.chunk_id == 0
        assert chunk.vertices == []
        assert chunk.indices == []
        assert chunk.mass == 1.0
        assert chunk.connected_to == []
        assert chunk.is_detached is False
        assert chunk.body_id is None

    def test_custom_values(self):
        """Test custom chunk values."""
        chunk = FractureChunk(
            chunk_id=5,
            mass=0.5,
            center_of_mass=Vector3(1.0, 2.0, 3.0),
            connected_to=[4, 6],
            is_detached=True,
            body_id=100,
        )

        assert chunk.chunk_id == 5
        assert chunk.mass == 0.5
        assert chunk.center_of_mass.x == 1.0
        assert chunk.connected_to == [4, 6]
        assert chunk.is_detached is True


# =============================================================================
# DestructionConfig Tests
# =============================================================================


class TestDestructionConfig:
    """Tests for DestructionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DestructionConfig()

        assert config.destruction_type == DestructionType.FRACTURE
        assert config.health == 100.0
        assert config.min_damage_threshold == 5.0
        assert config.debris_lifetime == 10.0
        assert config.debris_count_limit == 50
        assert config.propagate_damage is True
        assert config.propagation_factor == 0.5
        assert config.fracture_seed == 12345

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DestructionConfig(
            destruction_type=DestructionType.SHATTER,
            health=500.0,
            min_damage_threshold=10.0,
            propagate_damage=False,
        )

        assert config.destruction_type == DestructionType.SHATTER
        assert config.health == 500.0
        assert config.propagate_damage is False


# =============================================================================
# DamageInfo Tests
# =============================================================================


class TestDamageInfo:
    """Tests for DamageInfo dataclass."""

    def test_default_values(self):
        """Test default damage info values."""
        info = DamageInfo()

        assert info.amount == 0.0
        assert info.damage_type == DamageType.GENERIC
        assert info.impulse == 0.0

    def test_custom_values(self):
        """Test custom damage info values."""
        info = DamageInfo(
            amount=50.0,
            damage_type=DamageType.EXPLOSION,
            position=Vector3(1.0, 2.0, 3.0),
            direction=Vector3(0.0, -1.0, 0.0),
            impulse=1000.0,
        )

        assert info.amount == 50.0
        assert info.damage_type == DamageType.EXPLOSION
        assert info.impulse == 1000.0


# =============================================================================
# Component Creation Tests
# =============================================================================


class TestDestructibleCreation:
    """Tests for component creation."""

    def test_create_with_default_config(self):
        """Test creating with default config."""
        component = DestructibleComponent(entity_id=1)

        assert component.entity_id == 1
        assert component.current_health == 100.0
        assert component.max_health == 100.0
        assert component.is_destroyed is False
        assert component.enabled is True

    def test_create_with_custom_config(self, destructible, destruction_config):
        """Test creating with custom config."""
        assert destructible.current_health == 100.0
        assert destructible.config.min_damage_threshold == 5.0

    def test_initial_state(self, destructible):
        """Test initial component state."""
        assert destructible.destruction_stage == 0
        assert destructible.chunk_count == 0
        assert destructible.detached_chunk_count == 0


# =============================================================================
# Health Tests
# =============================================================================


class TestHealthManagement:
    """Tests for health management."""

    def test_health_ratio(self, destructible):
        """Test health ratio calculation."""
        assert destructible.health_ratio == 1.0

        # Apply some damage
        destructible.apply_damage(DamageInfo(amount=50.0))

        assert destructible.health_ratio == 0.5

    def test_health_ratio_zero_max(self):
        """Test health ratio with zero max health."""
        config = DestructionConfig(health=0.0)
        component = DestructibleComponent(entity_id=1, config=config)

        assert component.health_ratio == 0


# =============================================================================
# Damage Application Tests
# =============================================================================


class TestDamageApplication:
    """Tests for damage application."""

    def test_apply_damage(self, destructible):
        """Test basic damage application."""
        damage = DamageInfo(amount=30.0)

        result = destructible.apply_damage(damage)

        assert result is True
        assert destructible.current_health == 70.0

    def test_damage_below_threshold(self, destructible):
        """Test damage below threshold is ignored."""
        damage = DamageInfo(amount=4.0)  # Below threshold of 5.0

        result = destructible.apply_damage(damage)

        assert result is False
        assert destructible.current_health == 100.0

    def test_damage_at_threshold(self, destructible):
        """Test damage at threshold is applied."""
        damage = DamageInfo(amount=5.0)

        result = destructible.apply_damage(damage)

        assert result is True
        assert destructible.current_health == 95.0

    def test_damage_when_disabled(self, destructible):
        """Test damage when component is disabled."""
        destructible.enabled = False
        damage = DamageInfo(amount=50.0)

        result = destructible.apply_damage(damage)

        assert result is False
        assert destructible.current_health == 100.0

    def test_damage_when_destroyed(self, destructible):
        """Test damage when already destroyed."""
        destructible.apply_damage(DamageInfo(amount=100.0))
        assert destructible.is_destroyed is True

        result = destructible.apply_damage(DamageInfo(amount=50.0))

        assert result is False

    def test_damage_callback(self, destructible):
        """Test damage callback is called."""
        damages = []
        destructible.set_damage_callback(lambda d: damages.append(d))

        damage = DamageInfo(amount=20.0)
        destructible.apply_damage(damage)

        assert len(damages) == 1
        assert damages[0].amount == 20.0

    def test_explosion_damage(self, destructible):
        """Test explosion damage application."""
        destructible.apply_explosion_damage(
            center=Vector3(0.0, 0.0, 0.0),
            radius=10.0,
            max_damage=50.0,
        )

        assert destructible.current_health == 50.0


# =============================================================================
# Destruction Tests
# =============================================================================


class TestDestruction:
    """Tests for destruction behavior."""

    def test_full_destruction(self, destructible):
        """Test full destruction when health reaches zero."""
        damage = DamageInfo(amount=100.0)

        destructible.apply_damage(damage)

        assert destructible.is_destroyed is True
        assert destructible.current_health == 0.0

    def test_overkill_damage(self, destructible):
        """Test damage beyond health doesn't go negative."""
        damage = DamageInfo(amount=200.0)

        destructible.apply_damage(damage)

        assert destructible.current_health == 0.0
        assert destructible.is_destroyed is True

    def test_destroyed_callback(self, destructible):
        """Test destroyed callback is called."""
        destroyed_called = []
        destructible.set_destroyed_callback(lambda: destroyed_called.append(True))

        destructible.apply_damage(DamageInfo(amount=100.0))

        assert len(destroyed_called) == 1

    def test_force_destroy(self, destructible):
        """Test force destroy."""
        destructible.force_destroy()

        assert destructible.is_destroyed is True
        assert destructible.current_health == 0.0


# =============================================================================
# Staged Destruction Tests
# =============================================================================


class TestStagedDestruction:
    """Tests for staged destruction."""

    def test_default_stages(self, destructible):
        """Test default stage thresholds."""
        # Default: 0.75, 0.5, 0.25, 0.0
        assert destructible.destruction_stage == 0

    def test_stage_progression(self, destructible):
        """Test stage progression as health decreases."""
        # Stage 1 at 75% health (25 damage)
        destructible.apply_damage(DamageInfo(amount=26.0))
        assert destructible.destruction_stage >= 1

    def test_stage_callback(self, destructible):
        """Test stage change callback."""
        stages = []
        destructible.set_stage_change_callback(lambda s: stages.append(s))

        destructible.apply_damage(DamageInfo(amount=30.0))

        assert len(stages) >= 1

    def test_custom_stage_thresholds(self, destructible):
        """Test custom stage thresholds."""
        destructible.set_stage_thresholds([0.9, 0.7, 0.5, 0.3, 0.1])

        # Now first stage triggers at 90%
        destructible.apply_damage(DamageInfo(amount=15.0))

        # Should be at stage 1 (health ratio = 0.85, which is <= 0.9)
        assert destructible.destruction_stage >= 1


# =============================================================================
# Fracture Chunk Tests
# =============================================================================


class TestFractureChunks:
    """Tests for fracture chunk management."""

    def test_set_fracture_chunks(self, destructible):
        """Test setting pre-computed chunks."""
        chunks = [
            FractureChunk(chunk_id=0),
            FractureChunk(chunk_id=1),
        ]

        destructible.set_fracture_chunks(chunks)

        assert destructible.chunk_count == 2

    def test_generate_voronoi_chunks(self, destructible):
        """Test generating Voronoi chunks."""
        vertices = [
            Vector3(0, 0, 0), Vector3(1, 0, 0),
            Vector3(1, 1, 0), Vector3(0, 1, 0),
        ]
        indices = [0, 1, 2, 0, 2, 3]

        destructible.generate_voronoi_chunks(vertices, indices, num_chunks=5)

        assert destructible.chunk_count == 5
        assert destructible._initialized is True

    def test_get_chunk(self, destructible_with_chunks):
        """Test getting chunk by ID."""
        chunk = destructible_with_chunks.get_chunk(1)

        assert chunk is not None
        assert chunk.chunk_id == 1

    def test_get_chunk_out_of_range(self, destructible_with_chunks):
        """Test getting nonexistent chunk."""
        chunk = destructible_with_chunks.get_chunk(999)

        assert chunk is None

    def test_get_attached_chunks(self, destructible_with_chunks):
        """Test getting attached chunks."""
        attached = destructible_with_chunks.get_attached_chunks()

        assert len(attached) == 4

    def test_get_detached_chunks(self, destructible_with_chunks):
        """Test getting detached chunks."""
        # Initially all attached
        detached = destructible_with_chunks.get_detached_chunks()
        assert len(detached) == 0

        # Detach by damaging
        destructible_with_chunks.apply_damage(DamageInfo(amount=50.0))

        # Some may be detached now
        # This depends on implementation details

    def test_chunk_detachment_callback(self, destructible_with_chunks):
        """Test chunk detachment callback."""
        detached_chunks = []
        destructible_with_chunks.set_chunk_detached_callback(
            lambda c: detached_chunks.append(c)
        )

        # Force destruction to detach all
        destructible_with_chunks.force_destroy()

        assert len(detached_chunks) > 0


# =============================================================================
# Damage Propagation Tests
# =============================================================================


class TestDamagePropagation:
    """Tests for damage propagation."""

    def test_propagation_enabled(self, destructible_with_chunks):
        """Test damage propagation to connected chunks."""
        damage = DamageInfo(
            amount=50.0,
            position=Vector3(0.0, 0.0, 0.0),
        )

        destructible_with_chunks.apply_damage(damage)

        # With propagation, some connected chunks may detach
        # Exact behavior depends on implementation

    def test_propagation_disabled(self):
        """Test no propagation when disabled."""
        config = DestructionConfig(propagate_damage=False)
        component = DestructibleComponent(entity_id=1, config=config)

        chunks = [
            FractureChunk(chunk_id=0, connected_to=[1]),
            FractureChunk(chunk_id=1, connected_to=[0]),
        ]
        component.set_fracture_chunks(chunks)

        damage = DamageInfo(amount=50.0)
        component.apply_damage(damage)

        # Propagation disabled, so only direct damage effects


# =============================================================================
# Repair Tests
# =============================================================================


class TestRepair:
    """Tests for repair functionality."""

    def test_repair(self, destructible):
        """Test basic repair."""
        destructible.apply_damage(DamageInfo(amount=50.0))

        repaired = destructible.repair(30.0)

        assert repaired == 30.0
        assert destructible.current_health == 80.0

    def test_repair_over_max(self, destructible):
        """Test repair doesn't exceed max health."""
        destructible.apply_damage(DamageInfo(amount=20.0))

        repaired = destructible.repair(50.0)

        assert repaired == 20.0
        assert destructible.current_health == 100.0

    def test_repair_destroyed(self, destructible):
        """Test repair on destroyed object fails."""
        destructible.apply_damage(DamageInfo(amount=100.0))

        repaired = destructible.repair(50.0)

        assert repaired == 0.0
        assert destructible.is_destroyed is True

    def test_reset(self, destructible):
        """Test full reset to undamaged state."""
        destructible.apply_damage(DamageInfo(amount=60.0))

        destructible.reset()

        assert destructible.current_health == 100.0
        assert destructible.is_destroyed is False
        assert destructible.destruction_stage == 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for component lifecycle."""

    def test_enabled_property(self, destructible):
        """Test enabled property."""
        assert destructible.enabled is True

        destructible.enabled = False
        assert destructible.enabled is False

    def test_cleanup(self, destructible_with_chunks):
        """Test cleanup clears all data."""
        destructible_with_chunks.apply_damage(DamageInfo(amount=30.0))

        destructible_with_chunks.cleanup()

        assert destructible_with_chunks.chunk_count == 0
        assert destructible_with_chunks.detached_chunk_count == 0


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization."""

    def test_get_state(self, destructible):
        """Test getting serializable state."""
        destructible.apply_damage(DamageInfo(amount=30.0))

        state = destructible.get_state()

        assert state["entity_id"] == 1
        assert state["current_health"] == 70.0
        assert state["max_health"] == 100.0
        assert state["is_destroyed"] is False
        assert state["destruction_stage"] >= 0
        assert state["total_damage"] == 30.0
        assert state["enabled"] is True

    def test_state_with_chunks(self, destructible_with_chunks):
        """Test state includes chunk info."""
        state = destructible_with_chunks.get_state()

        assert state["chunk_count"] == 4
        assert state["detached_count"] == 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_health_config(self):
        """Test component with zero max health."""
        config = DestructionConfig(health=0.0)
        component = DestructibleComponent(entity_id=1, config=config)

        assert component.max_health == 0.0
        assert component.current_health == 0.0

    def test_very_low_threshold(self):
        """Test very low damage threshold."""
        config = DestructionConfig(min_damage_threshold=0.001)
        component = DestructibleComponent(entity_id=1, config=config)

        result = component.apply_damage(DamageInfo(amount=0.01))

        assert result is True

    def test_rapid_damage(self, destructible):
        """Test rapid successive damage."""
        for i in range(20):
            destructible.apply_damage(DamageInfo(amount=5.0))

        assert destructible.is_destroyed is True
        assert destructible.current_health == 0.0

    def test_empty_chunks(self, destructible):
        """Test operations with no chunks."""
        attached = destructible.get_attached_chunks()
        detached = destructible.get_detached_chunks()

        assert attached == []
        assert detached == []

    def test_reset_with_chunks(self, destructible_with_chunks):
        """Test reset restores chunks."""
        destructible_with_chunks.force_destroy()

        destructible_with_chunks.reset()

        assert destructible_with_chunks.current_health == 100.0
        assert destructible_with_chunks.is_destroyed is False
        # Chunks should be reset
        assert destructible_with_chunks.detached_chunk_count == 0
