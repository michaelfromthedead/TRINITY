"""
Whitebox tests for engine.simulation.collision.collision_filter module.

Tests collision layer system, masks, filters, and filter manager:
- CollisionLayer
- CollisionMask
- CollisionFilter
- should_collide
- CollisionFilterManager
- FilterPresets
"""

import pytest
from engine.simulation.collision.collision_filter import (
    CollisionLayer,
    CollisionMask,
    CollisionFilter,
    should_collide,
    create_layer_matrix,
    CollisionFilterManager,
    FilterPresets,
)


class TestCollisionLayer:
    """Tests for CollisionLayer IntFlag enum."""

    def test_none_is_zero(self):
        """NONE layer should be 0."""
        assert CollisionLayer.NONE == 0

    def test_default_is_bit_0(self):
        """DEFAULT layer should be bit 0."""
        assert CollisionLayer.DEFAULT == 1

    def test_layers_are_power_of_two(self):
        """Individual layers should be powers of two."""
        individual_layers = [
            CollisionLayer.DEFAULT,
            CollisionLayer.STATIC,
            CollisionLayer.DYNAMIC,
            CollisionLayer.KINEMATIC,
            CollisionLayer.TRIGGER,
            CollisionLayer.PROJECTILE,
            CollisionLayer.DEBRIS,
            CollisionLayer.SENSOR,
            CollisionLayer.PLAYER,
            CollisionLayer.NPC,
            CollisionLayer.ENEMY,
            CollisionLayer.VEHICLE,
            CollisionLayer.TERRAIN,
            CollisionLayer.WATER,
            CollisionLayer.CLIMBABLE,
            CollisionLayer.DESTRUCTIBLE,
        ]
        for layer in individual_layers:
            # Power of 2 check: only one bit set
            assert layer & (layer - 1) == 0
            assert layer > 0

    def test_all_layer(self):
        """ALL layer should have all bits set."""
        assert CollisionLayer.ALL == 0xFFFFFFFF

    def test_layer_combinations(self):
        """Layer combinations should work with bitwise ops."""
        combined = CollisionLayer.PLAYER | CollisionLayer.NPC
        assert combined & CollisionLayer.PLAYER
        assert combined & CollisionLayer.NPC
        assert not (combined & CollisionLayer.ENEMY)

    def test_custom_layers_exist(self):
        """Custom layers should exist."""
        assert hasattr(CollisionLayer, "CUSTOM_1")
        assert hasattr(CollisionLayer, "CUSTOM_16")


class TestCollisionMask:
    """Tests for CollisionMask class."""

    def test_default_is_all(self):
        """Default mask should collide with all."""
        mask = CollisionMask()
        assert mask.value == CollisionLayer.ALL

    def test_from_layers(self):
        """from_layers should create mask from multiple layers."""
        mask = CollisionMask.from_layers(
            CollisionLayer.PLAYER,
            CollisionLayer.ENEMY,
        )
        assert mask.includes(CollisionLayer.PLAYER)
        assert mask.includes(CollisionLayer.ENEMY)
        assert not mask.includes(CollisionLayer.STATIC)

    def test_all_except(self):
        """all_except should exclude specified layers."""
        mask = CollisionMask.all_except(CollisionLayer.TRIGGER)
        assert not mask.includes(CollisionLayer.TRIGGER)
        assert mask.includes(CollisionLayer.PLAYER)

    def test_none(self):
        """none should collide with nothing."""
        mask = CollisionMask.none()
        assert mask.value == 0

    def test_includes(self):
        """includes should check layer presence."""
        mask = CollisionMask.from_layers(CollisionLayer.PLAYER)
        assert mask.includes(CollisionLayer.PLAYER)
        assert not mask.includes(CollisionLayer.ENEMY)

    def test_add(self):
        """add should add layer to mask."""
        mask = CollisionMask.none()
        new_mask = mask.add(CollisionLayer.PLAYER)
        assert new_mask.includes(CollisionLayer.PLAYER)
        # Original should be unchanged
        assert not mask.includes(CollisionLayer.PLAYER)

    def test_remove(self):
        """remove should remove layer from mask."""
        mask = CollisionMask()
        new_mask = mask.remove(CollisionLayer.PLAYER)
        assert not new_mask.includes(CollisionLayer.PLAYER)

    def test_toggle(self):
        """toggle should flip layer bit."""
        mask = CollisionMask.none()
        mask = mask.toggle(CollisionLayer.PLAYER)
        assert mask.includes(CollisionLayer.PLAYER)
        mask = mask.toggle(CollisionLayer.PLAYER)
        assert not mask.includes(CollisionLayer.PLAYER)

    def test_intersects(self):
        """intersects should check for common layers."""
        mask_a = CollisionMask.from_layers(CollisionLayer.PLAYER, CollisionLayer.NPC)
        mask_b = CollisionMask.from_layers(CollisionLayer.NPC, CollisionLayer.ENEMY)
        assert mask_a.intersects(mask_b)

    def test_intersects_no_overlap(self):
        """intersects should return False for no overlap."""
        mask_a = CollisionMask.from_layers(CollisionLayer.PLAYER)
        mask_b = CollisionMask.from_layers(CollisionLayer.ENEMY)
        assert not mask_a.intersects(mask_b)

    def test_bitwise_and(self):
        """Bitwise AND should work."""
        mask_a = CollisionMask.from_layers(CollisionLayer.PLAYER, CollisionLayer.NPC)
        mask_b = CollisionMask.from_layers(CollisionLayer.NPC, CollisionLayer.ENEMY)
        result = mask_a & mask_b
        assert result.includes(CollisionLayer.NPC)
        assert not result.includes(CollisionLayer.PLAYER)

    def test_bitwise_or(self):
        """Bitwise OR should work."""
        mask_a = CollisionMask.from_layers(CollisionLayer.PLAYER)
        mask_b = CollisionMask.from_layers(CollisionLayer.ENEMY)
        result = mask_a | mask_b
        assert result.includes(CollisionLayer.PLAYER)
        assert result.includes(CollisionLayer.ENEMY)

    def test_bitwise_invert(self):
        """Bitwise invert should work."""
        mask = CollisionMask.from_layers(CollisionLayer.PLAYER)
        inverted = ~mask
        assert not inverted.includes(CollisionLayer.PLAYER)
        assert inverted.includes(CollisionLayer.ENEMY)


class TestCollisionFilter:
    """Tests for CollisionFilter class."""

    def test_default_construction(self):
        """Default filter should use DEFAULT layer and ALL mask."""
        filter_ = CollisionFilter()
        assert filter_.category == CollisionLayer.DEFAULT
        assert filter_.mask.value == CollisionLayer.ALL
        assert filter_.group == 0

    def test_int_mask_conversion(self):
        """Integer mask should be converted to CollisionMask."""
        filter_ = CollisionFilter(mask=CollisionLayer.PLAYER)
        assert isinstance(filter_.mask, CollisionMask)

    def test_static_preset(self):
        """static preset should create static filter."""
        filter_ = CollisionFilter.static()
        assert filter_.category == CollisionLayer.STATIC
        assert filter_.mask.includes(CollisionLayer.DYNAMIC)

    def test_dynamic_preset(self):
        """dynamic preset should create dynamic filter."""
        filter_ = CollisionFilter.dynamic()
        assert filter_.category == CollisionLayer.DYNAMIC

    def test_kinematic_preset(self):
        """kinematic preset should create kinematic filter."""
        filter_ = CollisionFilter.kinematic()
        assert filter_.category == CollisionLayer.KINEMATIC

    def test_trigger_preset(self):
        """trigger preset should create trigger filter."""
        filter_ = CollisionFilter.trigger()
        assert filter_.category == CollisionLayer.TRIGGER
        assert filter_.mask.includes(CollisionLayer.PLAYER)

    def test_projectile_preset(self):
        """projectile preset should create projectile filter."""
        filter_ = CollisionFilter.projectile()
        assert filter_.category == CollisionLayer.PROJECTILE
        assert filter_.mask.includes(CollisionLayer.STATIC)

    def test_player_preset(self):
        """player preset should create player filter."""
        filter_ = CollisionFilter.player(group=1)
        assert filter_.category == CollisionLayer.PLAYER
        assert not filter_.mask.includes(CollisionLayer.PLAYER)  # No self-collision
        assert filter_.group == 1

    def test_npc_preset(self):
        """npc preset should create NPC filter."""
        filter_ = CollisionFilter.npc(group=2)
        assert filter_.category == CollisionLayer.NPC
        assert filter_.group == 2

    def test_enemy_preset(self):
        """enemy preset should create enemy filter."""
        filter_ = CollisionFilter.enemy(group=3)
        assert filter_.category == CollisionLayer.ENEMY
        assert filter_.group == 3

    def test_debris_preset(self):
        """debris preset should only collide with world."""
        filter_ = CollisionFilter.debris()
        assert filter_.category == CollisionLayer.DEBRIS
        assert filter_.mask.includes(CollisionLayer.STATIC)
        assert not filter_.mask.includes(CollisionLayer.PLAYER)

    def test_sensor_preset(self):
        """sensor preset should detect dynamic objects."""
        filter_ = CollisionFilter.sensor()
        assert filter_.category == CollisionLayer.SENSOR


class TestShouldCollide:
    """Tests for should_collide function."""

    def test_default_filters_collide(self):
        """Default filters should collide."""
        filter_a = CollisionFilter()
        filter_b = CollisionFilter()
        assert should_collide(filter_a, filter_b)

    def test_same_group_no_collision(self):
        """Objects in same non-zero group should not collide."""
        filter_a = CollisionFilter(group=1)
        filter_b = CollisionFilter(group=1)
        assert not should_collide(filter_a, filter_b)

    def test_different_groups_collide(self):
        """Objects in different groups should collide."""
        filter_a = CollisionFilter(group=1)
        filter_b = CollisionFilter(group=2)
        assert should_collide(filter_a, filter_b)

    def test_zero_group_collides(self):
        """Group 0 should always collide."""
        filter_a = CollisionFilter(group=0)
        filter_b = CollisionFilter(group=0)
        assert should_collide(filter_a, filter_b)

    def test_mask_rejects_category(self):
        """Filter should reject if mask doesn't include category."""
        filter_a = CollisionFilter(
            category=CollisionLayer.PLAYER,
            mask=CollisionMask.from_layers(CollisionLayer.STATIC),
        )
        filter_b = CollisionFilter(category=CollisionLayer.ENEMY)
        assert not should_collide(filter_a, filter_b)

    def test_both_must_accept(self):
        """Both filters must accept each other."""
        filter_a = CollisionFilter(
            category=CollisionLayer.PLAYER,
            mask=CollisionMask(CollisionLayer.ALL),
        )
        filter_b = CollisionFilter(
            category=CollisionLayer.ENEMY,
            mask=CollisionMask.from_layers(CollisionLayer.STATIC),  # Doesn't accept PLAYER
        )
        assert not should_collide(filter_a, filter_b)


class TestCreateLayerMatrix:
    """Tests for create_layer_matrix function."""

    def test_creates_32x32_matrix(self):
        """Matrix should be 32x32."""
        matrix = create_layer_matrix()
        assert len(matrix) == 32
        assert all(len(row) == 32 for row in matrix)

    def test_all_true_by_default(self):
        """All entries should be True by default."""
        matrix = create_layer_matrix()
        assert all(all(cell for cell in row) for row in matrix)


class TestCollisionFilterManager:
    """Tests for CollisionFilterManager class."""

    def test_construction(self):
        """Manager should be constructed correctly."""
        manager = CollisionFilterManager()
        assert manager is not None

    def test_set_and_get_filter(self):
        """set_filter and get_filter should work."""
        manager = CollisionFilterManager()
        filter_ = CollisionFilter.player()
        manager.set_filter(1, filter_)
        retrieved = manager.get_filter(1)
        assert retrieved.category == CollisionLayer.PLAYER

    def test_get_filter_default(self):
        """get_filter should return default for unknown object."""
        manager = CollisionFilterManager()
        filter_ = manager.get_filter(999)
        assert filter_.category == CollisionLayer.DEFAULT

    def test_remove_filter(self):
        """remove_filter should remove filter."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter.player())
        assert manager.remove_filter(1)
        # Should return default now
        assert manager.get_filter(1).category == CollisionLayer.DEFAULT

    def test_remove_filter_nonexistent(self):
        """remove_filter should return False for nonexistent."""
        manager = CollisionFilterManager()
        assert not manager.remove_filter(999)

    def test_set_layer_collision(self):
        """set_layer_collision should modify matrix."""
        manager = CollisionFilterManager()
        # Disable PLAYER-ENEMY collision
        manager.set_layer_collision(
            CollisionLayer.PLAYER,
            CollisionLayer.ENEMY,
            False,
        )
        assert not manager.get_layer_collision(
            CollisionLayer.PLAYER,
            CollisionLayer.ENEMY,
        )

    def test_layer_collision_symmetric(self):
        """Layer collision should be symmetric."""
        manager = CollisionFilterManager()
        manager.set_layer_collision(
            CollisionLayer.PLAYER,
            CollisionLayer.ENEMY,
            False,
        )
        assert not manager.get_layer_collision(
            CollisionLayer.ENEMY,
            CollisionLayer.PLAYER,
        )

    def test_add_callback(self):
        """add_callback should add custom filter."""
        manager = CollisionFilterManager()
        # Filter that blocks all collisions with object 5
        manager.add_callback(lambda a, b: a != 5 and b != 5)
        manager.set_filter(1, CollisionFilter())
        manager.set_filter(5, CollisionFilter())
        assert not manager.should_collide(1, 5)
        assert manager.should_collide(1, 2)

    def test_remove_callback(self):
        """remove_callback should remove custom filter."""
        manager = CollisionFilterManager()
        callback = lambda a, b: a != 5 and b != 5
        manager.add_callback(callback)
        assert manager.remove_callback(callback)
        # Now collisions should work
        assert manager.should_collide(1, 5)

    def test_should_collide_uses_filters(self):
        """should_collide should use object filters."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter.player(group=1))
        manager.set_filter(2, CollisionFilter.player(group=1))  # Same group
        assert not manager.should_collide(1, 2)

    def test_should_collide_uses_layer_matrix(self):
        """should_collide should use layer matrix."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter(category=CollisionLayer.PLAYER))
        manager.set_filter(2, CollisionFilter(category=CollisionLayer.ENEMY))
        manager.set_layer_collision(CollisionLayer.PLAYER, CollisionLayer.ENEMY, False)
        assert not manager.should_collide(1, 2)

    def test_clear(self):
        """clear should reset manager."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter.player())
        manager.add_callback(lambda a, b: False)
        manager.clear()
        # Should use default filter now
        assert manager.get_filter(1).category == CollisionLayer.DEFAULT

    def test_default_trigger_no_collision(self):
        """Triggers should not physically collide by default."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter(category=CollisionLayer.TRIGGER))
        manager.set_filter(2, CollisionFilter(category=CollisionLayer.PLAYER))
        # Trigger layer collision is disabled by default
        assert not manager.get_layer_collision(
            CollisionLayer.TRIGGER,
            CollisionLayer.PLAYER,
        )


class TestFilterPresets:
    """Tests for FilterPresets class."""

    def test_fps_game_preset(self):
        """FPS game preset should have expected filters."""
        presets = FilterPresets.fps_game()
        assert "world" in presets
        assert "player" in presets
        assert "enemy" in presets
        assert "bullet" in presets
        assert "pickup" in presets

    def test_platformer_preset(self):
        """Platformer preset should have expected filters."""
        presets = FilterPresets.platformer()
        assert "terrain" in presets
        assert "player" in presets
        assert "platform" in presets
        assert "collectible" in presets

    def test_racing_preset(self):
        """Racing preset should have expected filters."""
        presets = FilterPresets.racing()
        assert "track" in presets
        assert "vehicle" in presets
        assert "barrier" in presets
        assert "checkpoint" in presets


class TestCollisionFilterEdgeCases:
    """Edge case tests for collision filtering."""

    def test_empty_mask(self):
        """Empty mask should collide with nothing."""
        filter_a = CollisionFilter(mask=CollisionMask.none())
        filter_b = CollisionFilter()
        assert not should_collide(filter_a, filter_b)

    def test_multiple_categories(self):
        """Object can belong to multiple categories."""
        filter_ = CollisionFilter(
            category=CollisionLayer.PLAYER | CollisionLayer.KINEMATIC,
        )
        other = CollisionFilter(
            mask=CollisionMask.from_layers(CollisionLayer.KINEMATIC),
        )
        assert should_collide(filter_, other)

    def test_self_collision_same_object(self):
        """Same object should use filter rules."""
        # If object is in a group with itself, it won't self-collide
        filter_ = CollisionFilter(group=1)
        # "Collision" with itself would be blocked by same group
        assert not should_collide(filter_, filter_)

    def test_high_layer_numbers(self):
        """High layer numbers (CUSTOM_16) should work."""
        filter_a = CollisionFilter(category=CollisionLayer.CUSTOM_16)
        filter_b = CollisionFilter(
            mask=CollisionMask.from_layers(CollisionLayer.CUSTOM_16),
        )
        assert should_collide(filter_a, filter_b)
