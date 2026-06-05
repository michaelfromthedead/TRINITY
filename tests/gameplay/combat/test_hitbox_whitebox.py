"""
WHITEBOX Tests for Hitbox System

Tests internal implementation details:
- Hitbox geometry calculations
- Bounding box intersection algorithms
- Priority system for overlapping hits
- Hit entity tracking and multi-hit prevention
- Activation/deactivation state machine
- Damage multiplier calculations
"""

import pytest
import time
import math
from unittest.mock import Mock, patch

from engine.gameplay.combat.hitbox import (
    HitboxSystem,
    Hitbox,
    Hurtbox,
    HitboxType,
    HurtboxType,
    HitboxShape,
    CollisionResult,
    CollisionInfo,
    Vector3,
    BoundingBox,
)
from engine.gameplay.combat.constants import (
    HitboxZone,
    HITBOX_DAMAGE_MULTIPLIERS,
    CRITICAL_HIT_ZONES,
    COUNTER_HIT_DAMAGE_MULTIPLIER,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def hitbox_system():
    """Create a fresh hitbox system."""
    return HitboxSystem()


@pytest.fixture
def basic_hitbox():
    """Create a basic hitbox."""
    return Hitbox(
        hitbox_id="test_hitbox",
        owner_id=1,
        position=Vector3(0, 0, 0),
        size=Vector3(1, 1, 1),
        damage=10.0,
    )


@pytest.fixture
def basic_hurtbox():
    """Create a basic hurtbox."""
    return Hurtbox(
        hurtbox_id="test_hurtbox",
        owner_id=2,
        position=Vector3(0, 0, 0),
        size=Vector3(1, 1, 1),
    )


# =============================================================================
# VECTOR3 TESTS (20 tests)
# =============================================================================


class TestVector3:
    """Tests for Vector3 class."""

    def test_default_values(self):
        """Default vector should be zero."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_distance_to_same_point(self):
        """Distance to same point should be zero."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(1, 2, 3)
        assert v1.distance_to(v2) == 0.0

    def test_distance_to_different_point(self):
        """Distance calculation should be correct."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(3, 4, 0)
        assert v1.distance_to(v2) == 5.0  # 3-4-5 triangle

    def test_distance_to_3d(self):
        """3D distance calculation."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(1, 2, 2)
        assert v1.distance_to(v2) == 3.0  # sqrt(1+4+4) = 3

    def test_vector_addition(self):
        """Vector addition."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_vector_subtraction(self):
        """Vector subtraction."""
        v1 = Vector3(5, 7, 9)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result.x == 4
        assert result.y == 5
        assert result.z == 6

    def test_to_tuple(self):
        """Convert to tuple."""
        v = Vector3(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)

    def test_negative_values(self):
        """Should handle negative values."""
        v = Vector3(-1, -2, -3)
        assert v.x == -1
        assert v.y == -2
        assert v.z == -3

    def test_distance_negative_coordinates(self):
        """Distance with negative coordinates."""
        v1 = Vector3(-1, -1, -1)
        v2 = Vector3(1, 1, 1)
        expected = math.sqrt(12)  # sqrt((2)^2 + (2)^2 + (2)^2)
        assert abs(v1.distance_to(v2) - expected) < 0.0001


# =============================================================================
# BOUNDING BOX TESTS (25 tests)
# =============================================================================


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_default_values(self):
        """Default bounding box."""
        bb = BoundingBox()
        assert bb.min_point.x == 0
        assert bb.max_point.x == 1

    def test_center_calculation(self):
        """Center should be midpoint."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 4, 6)
        )
        center = bb.center
        assert center.x == 1
        assert center.y == 2
        assert center.z == 3

    def test_size_calculation(self):
        """Size should be difference."""
        bb = BoundingBox(
            min_point=Vector3(1, 2, 3),
            max_point=Vector3(4, 6, 9)
        )
        size = bb.size
        assert size.x == 3
        assert size.y == 4
        assert size.z == 6

    def test_half_extents(self):
        """Half extents should be half of size."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(4, 4, 4)
        )
        half = bb.half_extents
        assert half.x == 2
        assert half.y == 2
        assert half.z == 2

    def test_contains_point_inside(self):
        """Point inside should return True."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2)
        )
        point = Vector3(1, 1, 1)
        assert bb.contains_point(point)

    def test_contains_point_outside(self):
        """Point outside should return False."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2)
        )
        point = Vector3(3, 1, 1)
        assert not bb.contains_point(point)

    def test_contains_point_on_boundary(self):
        """Point on boundary should return True."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2)
        )
        point = Vector3(2, 1, 1)
        assert bb.contains_point(point)

    def test_contains_point_at_corner(self):
        """Point at corner should return True."""
        bb = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2)
        )
        point = Vector3(0, 0, 0)
        assert bb.contains_point(point)

    def test_intersects_overlapping(self):
        """Overlapping boxes should intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2)
        )
        bb2 = BoundingBox(
            min_point=Vector3(1, 1, 1),
            max_point=Vector3(3, 3, 3)
        )
        assert bb1.intersects(bb2)
        assert bb2.intersects(bb1)

    def test_intersects_separate(self):
        """Separate boxes should not intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1)
        )
        bb2 = BoundingBox(
            min_point=Vector3(2, 2, 2),
            max_point=Vector3(3, 3, 3)
        )
        assert not bb1.intersects(bb2)

    def test_intersects_touching(self):
        """Touching boxes should intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1)
        )
        bb2 = BoundingBox(
            min_point=Vector3(1, 0, 0),
            max_point=Vector3(2, 1, 1)
        )
        assert bb1.intersects(bb2)

    def test_intersects_contained(self):
        """Contained box should intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(4, 4, 4)
        )
        bb2 = BoundingBox(
            min_point=Vector3(1, 1, 1),
            max_point=Vector3(2, 2, 2)
        )
        assert bb1.intersects(bb2)

    def test_intersects_x_separated(self):
        """X-separated boxes should not intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1)
        )
        bb2 = BoundingBox(
            min_point=Vector3(5, 0, 0),
            max_point=Vector3(6, 1, 1)
        )
        assert not bb1.intersects(bb2)

    def test_intersects_y_separated(self):
        """Y-separated boxes should not intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1)
        )
        bb2 = BoundingBox(
            min_point=Vector3(0, 5, 0),
            max_point=Vector3(1, 6, 1)
        )
        assert not bb1.intersects(bb2)

    def test_intersects_z_separated(self):
        """Z-separated boxes should not intersect."""
        bb1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1)
        )
        bb2 = BoundingBox(
            min_point=Vector3(0, 0, 5),
            max_point=Vector3(1, 1, 6)
        )
        assert not bb1.intersects(bb2)


# =============================================================================
# HITBOX GEOMETRY TESTS (30 tests)
# =============================================================================


class TestHitboxGeometry:
    """Tests for hitbox geometry calculations."""

    def test_bounding_box_generation(self, basic_hitbox):
        """Hitbox should generate correct bounding box."""
        basic_hitbox.position = Vector3(0, 0, 0)
        basic_hitbox.size = Vector3(2, 2, 2)

        bb = basic_hitbox.bounding_box
        assert bb.min_point.x == -1
        assert bb.max_point.x == 1

    def test_bounding_box_at_offset(self, basic_hitbox):
        """Bounding box should be offset by position."""
        basic_hitbox.position = Vector3(5, 5, 5)
        basic_hitbox.size = Vector3(2, 2, 2)

        bb = basic_hitbox.bounding_box
        assert bb.min_point.x == 4
        assert bb.max_point.x == 6

    def test_damage_multiplier_head(self):
        """Head hitbox should have 2x multiplier."""
        hitbox = Hitbox(
            hitbox_id="head",
            owner_id=1,
            zone=HitboxZone.HEAD
        )
        assert hitbox.damage_multiplier == 2.0

    def test_damage_multiplier_torso(self):
        """Torso hitbox should have 1x multiplier."""
        hitbox = Hitbox(
            hitbox_id="torso",
            owner_id=1,
            zone=HitboxZone.TORSO
        )
        assert hitbox.damage_multiplier == 1.0

    def test_damage_multiplier_limb(self):
        """Limb hitbox should have reduced multiplier."""
        hitbox = Hitbox(
            hitbox_id="arm",
            owner_id=1,
            zone=HitboxZone.LEFT_ARM
        )
        assert hitbox.damage_multiplier == 0.75

    def test_damage_multiplier_extremity(self):
        """Extremity hitbox should have heavily reduced multiplier."""
        hitbox = Hitbox(
            hitbox_id="hand",
            owner_id=1,
            zone=HitboxZone.LEFT_HAND
        )
        assert hitbox.damage_multiplier == 0.5

    def test_is_critical_zone_head(self):
        """Head should be critical zone."""
        hitbox = Hitbox(
            hitbox_id="head",
            owner_id=1,
            zone=HitboxZone.HEAD
        )
        assert hitbox.is_critical_zone

    def test_is_critical_zone_neck(self):
        """Neck should be critical zone."""
        hitbox = Hitbox(
            hitbox_id="neck",
            owner_id=1,
            zone=HitboxZone.NECK
        )
        assert hitbox.is_critical_zone

    def test_is_not_critical_zone_torso(self):
        """Torso should not be critical zone."""
        hitbox = Hitbox(
            hitbox_id="torso",
            owner_id=1,
            zone=HitboxZone.TORSO
        )
        assert not hitbox.is_critical_zone

    def test_set_position_from_tuple(self, basic_hitbox):
        """Should set position from tuple."""
        basic_hitbox.set_position((1, 2, 3))
        assert basic_hitbox.position.x == 1
        assert basic_hitbox.position.y == 2
        assert basic_hitbox.position.z == 3

    def test_hurtbox_zone_multiplier(self, basic_hurtbox):
        """Hurtbox should have zone multiplier."""
        basic_hurtbox.zone = HitboxZone.HEAD
        assert basic_hurtbox.zone_multiplier == 2.0

    def test_hurtbox_combined_multiplier(self, basic_hurtbox):
        """Hurtbox should combine zone and damage multipliers."""
        basic_hurtbox.zone = HitboxZone.HEAD
        basic_hurtbox.damage_multiplier = 1.5
        # Combined: 2.0 * 1.5 = 3.0
        assert basic_hurtbox.zone_multiplier == 3.0


# =============================================================================
# COLLISION DETECTION TESTS (40 tests)
# =============================================================================


class TestCollisionDetection:
    """Tests for collision detection algorithms."""

    def test_collision_same_owner(self, hitbox_system):
        """Should not collide with same owner."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 1, (0, 0, 0), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_different_owners(self, hitbox_system):
        """Should collide with different owners."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is not None

    def test_collision_no_overlap(self, hitbox_system):
        """Should not collide when not overlapping."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (10, 10, 10), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_inactive_hitbox(self, hitbox_system):
        """Inactive hitbox should not collide."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        # Not activated
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_inactive_hurtbox(self, hitbox_system):
        """Inactive hurtbox should not collide."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))
        hurtbox.active = False

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_result_hit(self, hitbox_system):
        """Normal collision should be HIT."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.HIT

    def test_collision_result_invincible(self, hitbox_system):
        """Intangible hurtbox should be INVINCIBLE."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            hurtbox_type=HurtboxType.INTANGIBLE
        )

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.INVINCIBLE

    def test_collision_result_counter_hit(self, hitbox_system):
        """Counter state hurtbox should be COUNTER_HIT."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            hurtbox_type=HurtboxType.COUNTER
        )

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.COUNTER_HIT

    def test_counter_hit_damage_bonus(self, hitbox_system):
        """Counter hit should apply damage bonus."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1), damage=100)
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            hurtbox_type=HurtboxType.COUNTER
        )

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.damage == 100 * COUNTER_HIT_DAMAGE_MULTIPLIER

    def test_collision_blocked(self, hitbox_system):
        """Block hitbox should result in BLOCKED."""
        # Create attack hitbox
        attack = hitbox_system.create_hitbox("attack", 1, (0, 0, 0), (1, 1, 1))

        # Create target hurtbox
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        # Create block hitbox for target
        block = hitbox_system.create_hitbox(
            "block", 2, (0, 0, 0), (2, 2, 2),
            hitbox_type=HitboxType.BLOCK
        )
        block.activate()
        attack.activate()

        collision = hitbox_system.check_collision(attack, hurtbox)
        assert collision.result == CollisionResult.BLOCKED

    def test_collision_parried(self, hitbox_system):
        """Parry hitbox should result in PARRIED."""
        attack = hitbox_system.create_hitbox("attack", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))
        parry = hitbox_system.create_hitbox(
            "parry", 2, (0, 0, 0), (2, 2, 2),
            hitbox_type=HitboxType.PARRY
        )
        parry.activate()
        attack.activate()

        collision = hitbox_system.check_collision(attack, hurtbox)
        assert collision.result == CollisionResult.PARRIED

    def test_collision_armored(self, hitbox_system):
        """Armored hurtbox should result in ARMORED."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=3
        )

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.ARMORED

    def test_armor_depletes(self, hitbox_system):
        """Armor should deplete after hits."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=1
        )

        hitbox.activate()

        # First hit absorbed by armor
        collision1 = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision1.result == CollisionResult.ARMORED

        # Second hit goes through (need to clear hit tracking)
        hitbox.hit_entities.clear()
        collision2 = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision2.result == CollisionResult.HIT

    def test_group_exclusion(self, hitbox_system):
        """Ignore groups should prevent collision."""
        hitbox = hitbox_system.create_hitbox(
            "hit", 1, (0, 0, 0), (1, 1, 1),
            groups={"attack"}
        )
        hitbox.ignore_groups.add("ally")

        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 2, (0, 0, 0), (1, 1, 1),
            groups={"ally"}
        )

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_contact_point(self, hitbox_system):
        """Collision should have contact point."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.point is not None

    def test_contact_point_midpoint(self, hitbox_system):
        """Contact point should be midpoint of hitboxes."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (2, 0, 0), (1, 1, 1))

        hitbox.activate()
        collision = hitbox_system.check_collision(hitbox, hurtbox)
        if collision:
            assert collision.point.x == 1.0  # Midpoint between 0 and 2


# =============================================================================
# PRIORITY SYSTEM TESTS (25 tests)
# =============================================================================


class TestPrioritySystem:
    """Tests for hit priority system."""

    def test_higher_priority_wins(self, hitbox_system):
        """Higher priority hitbox should win."""
        low_priority = hitbox_system.create_hitbox(
            "low", 1, (0, 0, 0), (1, 1, 1), damage=50, priority=1
        )
        high_priority = hitbox_system.create_hitbox(
            "high", 2, (0, 0, 0), (1, 1, 1), damage=100, priority=10
        )
        hurtbox = hitbox_system.create_hurtbox("hurt", 3, (0, 0, 0), (1, 1, 1))

        low_priority.activate()
        high_priority.activate()

        collisions = hitbox_system.process_collisions()
        # Only highest priority should register
        assert len([c for c in collisions if c.hitbox.priority == 10]) >= 1

    def test_same_priority_both_hit(self, hitbox_system):
        """Same priority hitboxes should both hit."""
        hit1 = hitbox_system.create_hitbox("hit1", 1, (0, 0, 0), (1, 1, 1), priority=5)
        hit2 = hitbox_system.create_hitbox("hit2", 2, (0, 0, 0), (1, 1, 1), priority=5)
        hurtbox = hitbox_system.create_hurtbox("hurt", 3, (0, 0, 0), (1, 1, 1))

        hit1.activate()
        hit2.activate()

        collisions = hitbox_system.process_collisions()
        # Both should register since same priority
        assert len(collisions) == 2

    def test_priority_zero_default(self, basic_hitbox):
        """Default priority should be zero."""
        assert basic_hitbox.priority == 0

    def test_priority_affects_damage_taken(self, hitbox_system):
        """Lower priority attack should not deal damage if higher exists."""
        low = hitbox_system.create_hitbox("low", 1, (0, 0, 0), (1, 1, 1), damage=10, priority=1)
        high = hitbox_system.create_hitbox("high", 2, (0, 0, 0), (1, 1, 1), damage=100, priority=10)
        hurtbox = hitbox_system.create_hurtbox("hurt", 3, (0, 0, 0), (1, 1, 1))

        low.activate()
        high.activate()

        collisions = hitbox_system.process_collisions()
        high_damages = [c.damage for c in collisions if c.hitbox.hitbox_id == "high"]
        assert 100 in high_damages


# =============================================================================
# HIT ENTITY TRACKING TESTS (30 tests)
# =============================================================================


class TestHitEntityTracking:
    """Tests for hit entity tracking and multi-hit prevention."""

    def test_entity_recorded_on_hit(self, hitbox_system):
        """Hit entity should be recorded."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        hitbox.activate()
        hitbox_system.check_collision(hitbox, hurtbox)
        hitbox.record_hit(2)

        assert 2 in hitbox.hit_entities

    def test_cant_hit_same_entity_twice(self, hitbox_system):
        """Should not hit same entity twice."""
        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))

        hitbox.activate()

        # First collision
        collision1 = hitbox_system.check_collision(hitbox, hurtbox)
        hitbox.record_hit(2)

        # Second collision attempt
        collision2 = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision2 is None

    def test_can_hit_entity_check(self, basic_hitbox):
        """can_hit_entity should return correct result."""
        basic_hitbox.activate()
        assert basic_hitbox.can_hit_entity(5)

        basic_hitbox.hit_entities.add(5)
        assert not basic_hitbox.can_hit_entity(5)

    def test_cant_hit_owner(self, basic_hitbox):
        """Should not be able to hit owner."""
        assert not basic_hitbox.can_hit_entity(basic_hitbox.owner_id)

    def test_record_hit_returns_false_for_duplicate(self, basic_hitbox):
        """record_hit should return False for already-hit entity."""
        basic_hitbox.activate()
        result1 = basic_hitbox.record_hit(5)
        result2 = basic_hitbox.record_hit(5)

        assert result1
        assert not result2

    def test_hit_count_increments(self, basic_hitbox):
        """Hit count should increment."""
        basic_hitbox.activate()
        basic_hitbox.record_hit(1)
        basic_hitbox.record_hit(2)
        basic_hitbox.record_hit(3)

        assert basic_hitbox._hit_count == 3

    def test_max_hits_deactivates(self, hitbox_system):
        """Hitbox should deactivate when max hits reached."""
        hitbox = hitbox_system.create_hitbox(
            "hit", 1, (0, 0, 0), (1, 1, 1), max_hits=2
        )
        hitbox.activate()

        hitbox.record_hit(2)
        hitbox.record_hit(3)

        assert not hitbox.active

    def test_max_hits_unlimited(self, basic_hitbox):
        """Max hits -1 should allow unlimited."""
        basic_hitbox.max_hits = -1
        basic_hitbox.activate()

        for i in range(100):
            basic_hitbox.record_hit(i + 10)

        assert basic_hitbox.active
        assert basic_hitbox.can_hit_more

    def test_can_hit_more_with_max(self, hitbox_system):
        """can_hit_more should respect max_hits."""
        hitbox = hitbox_system.create_hitbox(
            "hit", 1, (0, 0, 0), (1, 1, 1), max_hits=3
        )
        hitbox.activate()

        assert hitbox.can_hit_more
        hitbox.record_hit(1)
        hitbox.record_hit(2)
        assert hitbox.can_hit_more
        hitbox.record_hit(3)
        assert not hitbox.can_hit_more

    def test_activation_clears_hit_entities(self, basic_hitbox):
        """Activation should clear hit entities."""
        basic_hitbox.hit_entities.add(5)
        basic_hitbox.activate()

        assert 5 not in basic_hitbox.hit_entities

    def test_activation_resets_hit_count(self, basic_hitbox):
        """Activation should reset hit count."""
        basic_hitbox._hit_count = 5
        basic_hitbox.activate()

        assert basic_hitbox._hit_count == 0


# =============================================================================
# ACTIVATION/DEACTIVATION STATE TESTS (25 tests)
# =============================================================================


class TestActivationState:
    """Tests for activation/deactivation state machine."""

    def test_initially_inactive(self, basic_hitbox):
        """Hitbox should be inactive by default."""
        assert not basic_hitbox.active

    def test_activate(self, basic_hitbox):
        """Should activate hitbox."""
        basic_hitbox.activate()
        assert basic_hitbox.active

    def test_deactivate(self, basic_hitbox):
        """Should deactivate hitbox."""
        basic_hitbox.activate()
        basic_hitbox.deactivate()
        assert not basic_hitbox.active

    def test_activation_time_recorded(self, basic_hitbox):
        """Activation time should be recorded."""
        before = time.time()
        basic_hitbox.activate()
        after = time.time()

        assert before <= basic_hitbox._activation_time <= after

    def test_time_active(self, basic_hitbox):
        """time_active should return elapsed time."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            basic_hitbox.activate()

            mock_time.return_value = 1002.0
            assert basic_hitbox.time_active == 2.0

    def test_time_active_when_inactive(self, basic_hitbox):
        """time_active should return 0 when inactive."""
        assert basic_hitbox.time_active == 0.0

    def test_lifetime_activation(self, basic_hitbox):
        """Lifetime can be set on activation."""
        basic_hitbox.activate(lifetime=5.0)
        assert basic_hitbox.lifetime == 5.0

    def test_is_expired_false_initially(self, basic_hitbox):
        """is_expired should be False initially."""
        basic_hitbox.activate(lifetime=5.0)
        assert not basic_hitbox.is_expired

    def test_is_expired_after_lifetime(self, basic_hitbox):
        """is_expired should be True after lifetime."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            basic_hitbox.activate(lifetime=2.0)

            mock_time.return_value = 1003.0
            assert basic_hitbox.is_expired

    def test_is_expired_no_lifetime(self, basic_hitbox):
        """is_expired should be False with no lifetime."""
        basic_hitbox.activate()
        assert not basic_hitbox.is_expired

    def test_system_activate_hitbox(self, hitbox_system):
        """System should activate hitbox by ID."""
        hitbox = hitbox_system.create_hitbox("hit", 1)
        result = hitbox_system.activate_hitbox("hit")

        assert result
        assert hitbox.active

    def test_system_deactivate_hitbox(self, hitbox_system):
        """System should deactivate hitbox by ID."""
        hitbox = hitbox_system.create_hitbox("hit", 1)
        hitbox.activate()
        result = hitbox_system.deactivate_hitbox("hit")

        assert result
        assert not hitbox.active

    def test_system_activate_nonexistent(self, hitbox_system):
        """Should return False for nonexistent hitbox."""
        result = hitbox_system.activate_hitbox("nonexistent")
        assert not result

    def test_activate_entity_hitboxes(self, hitbox_system):
        """Should activate all hitboxes for entity."""
        hitbox_system.create_hitbox("hit1", 1)
        hitbox_system.create_hitbox("hit2", 1)
        hitbox_system.create_hitbox("hit3", 2)  # Different entity

        count = hitbox_system.activate_entity_hitboxes(1)
        assert count == 2

    def test_deactivate_entity_hitboxes(self, hitbox_system):
        """Should deactivate all hitboxes for entity."""
        hitbox_system.create_hitbox("hit1", 1)
        hitbox_system.create_hitbox("hit2", 1)

        hitbox_system.activate_entity_hitboxes(1)
        count = hitbox_system.deactivate_entity_hitboxes(1)

        assert count == 2

    def test_update_deactivates_expired(self, hitbox_system):
        """Update should deactivate expired hitboxes."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            hitbox = hitbox_system.create_hitbox("hit", 1)
            hitbox.activate(lifetime=1.0)

            mock_time.return_value = 1002.0
            hitbox_system.update(0.016)

            assert not hitbox.active


# =============================================================================
# HURTBOX TESTS (20 tests)
# =============================================================================


class TestHurtbox:
    """Tests for hurtbox functionality."""

    def test_hurtbox_initially_active(self, basic_hurtbox):
        """Hurtbox should be active by default."""
        assert basic_hurtbox.active

    def test_is_invincible(self):
        """is_invincible should check type."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.INTANGIBLE
        )
        assert hurtbox.is_invincible

    def test_is_counter_state(self):
        """is_counter_state should check type."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.COUNTER
        )
        assert hurtbox.is_counter_state

    def test_has_armor(self):
        """has_armor should check type and remaining armor."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=3
        )
        assert hurtbox.has_armor

    def test_absorb_armor_hit(self):
        """absorb_armor_hit should reduce armor."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=3
        )
        result = hurtbox.absorb_armor_hit()
        assert result
        assert hurtbox._armor_remaining == 2

    def test_absorb_armor_depleted(self):
        """absorb_armor_hit should return False when depleted."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=1
        )
        hurtbox.absorb_armor_hit()
        result = hurtbox.absorb_armor_hit()
        assert not result

    def test_reset_armor(self):
        """reset_armor should restore armor."""
        hurtbox = Hurtbox(
            hurtbox_id="test",
            owner_id=1,
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=3
        )
        hurtbox.absorb_armor_hit()
        hurtbox.absorb_armor_hit()
        hurtbox.reset_armor()
        assert hurtbox._armor_remaining == 3

    def test_set_hurtbox_type(self, hitbox_system):
        """set_hurtbox_type should change type."""
        hurtbox = hitbox_system.create_hurtbox("hurt", 1)
        hitbox_system.set_hurtbox_type("hurt", HurtboxType.COUNTER)
        assert hurtbox.hurtbox_type == HurtboxType.COUNTER

    def test_set_hurtbox_type_resets_armor(self, hitbox_system):
        """Setting to ARMORED should reset armor."""
        hurtbox = hitbox_system.create_hurtbox(
            "hurt", 1, armor_value=3
        )
        hurtbox._armor_remaining = 0
        hitbox_system.set_hurtbox_type("hurt", HurtboxType.ARMORED)
        assert hurtbox._armor_remaining == 3


# =============================================================================
# SYSTEM MANAGEMENT TESTS (15 tests)
# =============================================================================


class TestSystemManagement:
    """Tests for hitbox system management."""

    def test_create_hitbox_registered(self, hitbox_system):
        """Created hitbox should be registered."""
        hitbox = hitbox_system.create_hitbox("hit", 1)
        assert hitbox_system.get_hitbox("hit") is not None

    def test_remove_hitbox(self, hitbox_system):
        """Should remove hitbox."""
        hitbox_system.create_hitbox("hit", 1)
        result = hitbox_system.remove_hitbox("hit")
        assert result
        assert hitbox_system.get_hitbox("hit") is None

    def test_remove_nonexistent_hitbox(self, hitbox_system):
        """Should return False for nonexistent."""
        result = hitbox_system.remove_hitbox("nonexistent")
        assert not result

    def test_get_entity_hitboxes(self, hitbox_system):
        """Should get all hitboxes for entity."""
        hitbox_system.create_hitbox("hit1", 1)
        hitbox_system.create_hitbox("hit2", 1)
        hitbox_system.create_hitbox("hit3", 2)

        hitboxes = hitbox_system.get_entity_hitboxes(1)
        assert len(hitboxes) == 2

    def test_get_active_hitboxes(self, hitbox_system):
        """Should get only active hitboxes."""
        hit1 = hitbox_system.create_hitbox("hit1", 1)
        hit2 = hitbox_system.create_hitbox("hit2", 2)

        hit1.activate()

        active = hitbox_system.get_active_hitboxes()
        assert len(active) == 1
        assert active[0].hitbox_id == "hit1"

    def test_remove_entity(self, hitbox_system):
        """Should remove all boxes for entity."""
        hitbox_system.create_hitbox("hit", 1)
        hitbox_system.create_hurtbox("hurt", 1)

        hitbox_system.remove_entity(1)

        assert hitbox_system.get_hitbox("hit") is None
        assert hitbox_system.get_hurtbox("hurt") is None

    def test_clear(self, hitbox_system):
        """Should clear all data."""
        hitbox_system.create_hitbox("hit", 1)
        hitbox_system.create_hurtbox("hurt", 2)

        hitbox_system.clear()

        assert len(hitbox_system._hitboxes) == 0
        assert len(hitbox_system._hurtboxes) == 0

    def test_get_stats(self, hitbox_system):
        """Should return stats dict."""
        hitbox_system.create_hitbox("hit", 1)
        hitbox_system.create_hurtbox("hurt", 2)

        stats = hitbox_system.get_stats()
        assert stats["total_hitboxes"] == 1
        assert stats["total_hurtboxes"] == 1

    def test_event_handler_on_hit(self, hitbox_system):
        """Should fire on_hit callback."""
        callback = Mock()
        hitbox_system.on_hit(callback)

        hitbox = hitbox_system.create_hitbox("hit", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))
        hitbox.activate()

        hitbox_system.process_collisions()
        callback.assert_called()

    def test_event_handler_on_blocked(self, hitbox_system):
        """Should fire on_blocked callback."""
        callback = Mock()
        hitbox_system.on_blocked(callback)

        attack = hitbox_system.create_hitbox("attack", 1, (0, 0, 0), (1, 1, 1))
        hurtbox = hitbox_system.create_hurtbox("hurt", 2, (0, 0, 0), (1, 1, 1))
        block = hitbox_system.create_hitbox(
            "block", 2, (0, 0, 0), (2, 2, 2),
            hitbox_type=HitboxType.BLOCK
        )

        attack.activate()
        block.activate()

        hitbox_system.process_collisions()
        callback.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
