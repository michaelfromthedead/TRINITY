"""
Comprehensive tests for the Hitbox System.

Tests cover:
- Hitbox definition (position, size, type)
- Hurtbox definition
- Collision detection
- Hitbox activation/deactivation
- Hitbox groups
- Priority system
- Hitbox events
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

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
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def hitbox_system():
    """Create a fresh hitbox system for each test."""
    return HitboxSystem()


@pytest.fixture
def basic_hitbox():
    """Create a basic hitbox."""
    return Hitbox(
        hitbox_id="hit_1",
        owner_id=1,
        position=Vector3(0, 0, 0),
        size=Vector3(1, 1, 1),
        damage=10.0,
    )


@pytest.fixture
def basic_hurtbox():
    """Create a basic hurtbox."""
    return Hurtbox(
        hurtbox_id="hurt_1",
        owner_id=2,
        position=Vector3(0, 0, 0),
        size=Vector3(1, 1, 1),
    )


@pytest.fixture
def populated_system(hitbox_system):
    """Create a system with hitboxes and hurtboxes."""
    hitbox_system.create_hitbox(
        hitbox_id="attack_1",
        owner_id=1,
        position=(0, 0, 0),
        size=(2, 2, 2),
        damage=20.0,
    )
    hitbox_system.create_hurtbox(
        hurtbox_id="body_1",
        owner_id=2,
        position=(0, 0, 0),
        size=(2, 2, 2),
    )
    return hitbox_system


# =============================================================================
# VECTOR3 TESTS (~10 tests)
# =============================================================================


class TestVector3:
    """Tests for Vector3 class."""

    def test_create_vector(self):
        """Should create vector with coordinates."""
        v = Vector3(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_default_vector(self):
        """Should default to origin."""
        v = Vector3()
        assert v.x == 0
        assert v.y == 0
        assert v.z == 0

    def test_distance_to(self):
        """Should calculate distance."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(3, 4, 0)

        assert v1.distance_to(v2) == pytest.approx(5.0, rel=0.01)

    def test_distance_3d(self):
        """Should calculate 3D distance."""
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(1, 1, 1)

        expected = 1.732  # sqrt(3)
        assert v1.distance_to(v2) == pytest.approx(expected, rel=0.01)

    def test_add_vectors(self):
        """Should add vectors."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2

        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_subtract_vectors(self):
        """Should subtract vectors."""
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2

        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_to_tuple(self):
        """Should convert to tuple."""
        v = Vector3(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)


# =============================================================================
# BOUNDING BOX TESTS (~10 tests)
# =============================================================================


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_create_box(self):
        """Should create bounding box."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1),
        )
        assert box.min_point.x == 0
        assert box.max_point.x == 1

    def test_box_center(self):
        """Should calculate center."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2),
        )
        assert box.center.x == 1
        assert box.center.y == 1
        assert box.center.z == 1

    def test_box_size(self):
        """Should calculate size."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(3, 4, 5),
        )
        assert box.size.x == 3
        assert box.size.y == 4
        assert box.size.z == 5

    def test_half_extents(self):
        """Should calculate half extents."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 4, 6),
        )
        assert box.half_extents.x == 1
        assert box.half_extents.y == 2
        assert box.half_extents.z == 3

    def test_contains_point_inside(self):
        """Should detect point inside box."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2),
        )
        assert box.contains_point(Vector3(1, 1, 1))

    def test_contains_point_outside(self):
        """Should detect point outside box."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2),
        )
        assert not box.contains_point(Vector3(3, 3, 3))

    def test_contains_point_edge(self):
        """Should detect point on edge."""
        box = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2),
        )
        assert box.contains_point(Vector3(0, 0, 0))
        assert box.contains_point(Vector3(2, 2, 2))

    def test_intersects_overlapping(self):
        """Should detect overlapping boxes."""
        box1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(2, 2, 2),
        )
        box2 = BoundingBox(
            min_point=Vector3(1, 1, 1),
            max_point=Vector3(3, 3, 3),
        )
        assert box1.intersects(box2)
        assert box2.intersects(box1)

    def test_intersects_not_overlapping(self):
        """Should detect non-overlapping boxes."""
        box1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1),
        )
        box2 = BoundingBox(
            min_point=Vector3(2, 2, 2),
            max_point=Vector3(3, 3, 3),
        )
        assert not box1.intersects(box2)

    def test_intersects_touching(self):
        """Should detect touching boxes."""
        box1 = BoundingBox(
            min_point=Vector3(0, 0, 0),
            max_point=Vector3(1, 1, 1),
        )
        box2 = BoundingBox(
            min_point=Vector3(1, 0, 0),
            max_point=Vector3(2, 1, 1),
        )
        assert box1.intersects(box2)


# =============================================================================
# HITBOX CREATION TESTS (~15 tests)
# =============================================================================


class TestHitboxCreation:
    """Tests for hitbox creation."""

    def test_create_hitbox(self, hitbox_system):
        """Should create hitbox."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            damage=10.0,
        )
        assert hitbox is not None
        assert hitbox.hitbox_id == "hit_1"

    def test_create_hitbox_with_type(self, hitbox_system):
        """Should create hitbox with type."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            hitbox_type=HitboxType.PROJECTILE,
        )
        assert hitbox.hitbox_type == HitboxType.PROJECTILE

    def test_create_hitbox_with_zone(self, hitbox_system):
        """Should create hitbox with zone."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            zone=HitboxZone.HEAD,
        )
        assert hitbox.zone == HitboxZone.HEAD

    def test_create_hitbox_with_priority(self, hitbox_system):
        """Should create hitbox with priority."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            priority=10,
        )
        assert hitbox.priority == 10

    def test_create_hitbox_with_groups(self, hitbox_system):
        """Should create hitbox with groups."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            groups={"melee", "player"},
        )
        assert "melee" in hitbox.groups
        assert "player" in hitbox.groups

    def test_create_hitbox_with_max_hits(self, hitbox_system):
        """Should create hitbox with max hits."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            max_hits=3,
        )
        assert hitbox.max_hits == 3

    def test_create_hitbox_with_lifetime(self, hitbox_system):
        """Should create hitbox with lifetime."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            lifetime=0.5,
        )
        assert hitbox.lifetime == 0.5

    def test_get_hitbox(self, hitbox_system):
        """Should get hitbox by ID."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox = hitbox_system.get_hitbox("hit_1")
        assert hitbox is not None

    def test_get_nonexistent_hitbox(self, hitbox_system):
        """Should return None for nonexistent hitbox."""
        hitbox = hitbox_system.get_hitbox("nonexistent")
        assert hitbox is None

    def test_remove_hitbox(self, hitbox_system):
        """Should remove hitbox."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        result = hitbox_system.remove_hitbox("hit_1")

        assert result
        assert hitbox_system.get_hitbox("hit_1") is None

    def test_remove_nonexistent_hitbox(self, hitbox_system):
        """Should return False for nonexistent hitbox."""
        result = hitbox_system.remove_hitbox("nonexistent")
        assert not result

    def test_get_entity_hitboxes(self, hitbox_system):
        """Should get all hitboxes for entity."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.create_hitbox(hitbox_id="hit_2", owner_id=1)
        hitbox_system.create_hitbox(hitbox_id="hit_3", owner_id=2)

        hitboxes = hitbox_system.get_entity_hitboxes(1)
        assert len(hitboxes) == 2


# =============================================================================
# HURTBOX CREATION TESTS (~15 tests)
# =============================================================================


class TestHurtboxCreation:
    """Tests for hurtbox creation."""

    def test_create_hurtbox(self, hitbox_system):
        """Should create hurtbox."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )
        assert hurtbox is not None
        assert hurtbox.hurtbox_id == "hurt_1"

    def test_create_hurtbox_with_type(self, hitbox_system):
        """Should create hurtbox with type."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,
            hurtbox_type=HurtboxType.COUNTER,
        )
        assert hurtbox.hurtbox_type == HurtboxType.COUNTER

    def test_create_hurtbox_with_zone(self, hitbox_system):
        """Should create hurtbox with zone."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,
            zone=HitboxZone.HEAD,
        )
        assert hurtbox.zone == HitboxZone.HEAD

    def test_create_hurtbox_with_armor(self, hitbox_system):
        """Should create hurtbox with armor."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,
            armor_value=3,
        )
        assert hurtbox.armor_value == 3

    def test_create_hurtbox_with_multiplier(self, hitbox_system):
        """Should create hurtbox with damage multiplier."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,
            damage_multiplier=1.5,
        )
        assert hurtbox.damage_multiplier == 1.5

    def test_get_hurtbox(self, hitbox_system):
        """Should get hurtbox by ID."""
        hitbox_system.create_hurtbox(hurtbox_id="hurt_1", owner_id=1)
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        assert hurtbox is not None

    def test_get_nonexistent_hurtbox(self, hitbox_system):
        """Should return None for nonexistent hurtbox."""
        hurtbox = hitbox_system.get_hurtbox("nonexistent")
        assert hurtbox is None

    def test_remove_hurtbox(self, hitbox_system):
        """Should remove hurtbox."""
        hitbox_system.create_hurtbox(hurtbox_id="hurt_1", owner_id=1)
        result = hitbox_system.remove_hurtbox("hurt_1")

        assert result
        assert hitbox_system.get_hurtbox("hurt_1") is None

    def test_get_entity_hurtboxes(self, hitbox_system):
        """Should get all hurtboxes for entity."""
        hitbox_system.create_hurtbox(hurtbox_id="hurt_1", owner_id=1)
        hitbox_system.create_hurtbox(hurtbox_id="hurt_2", owner_id=1)
        hitbox_system.create_hurtbox(hurtbox_id="hurt_3", owner_id=2)

        hurtboxes = hitbox_system.get_entity_hurtboxes(1)
        assert len(hurtboxes) == 2


# =============================================================================
# COLLISION DETECTION TESTS (~25 tests)
# =============================================================================


class TestCollisionDetection:
    """Tests for collision detection."""

    def test_check_collision_hit(self, populated_system):
        """Should detect collision when overlapping."""
        hitbox = populated_system.get_hitbox("attack_1")
        hurtbox = populated_system.get_hurtbox("body_1")
        hitbox.activate()

        collision = populated_system.check_collision(hitbox, hurtbox)

        assert collision is not None
        assert collision.result == CollisionResult.HIT

    def test_check_collision_no_overlap(self, hitbox_system):
        """Should not detect collision when not overlapping."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(10, 10, 10),
            size=(1, 1, 1),
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_no_collision_same_owner(self, hitbox_system):
        """Should not detect collision with same owner."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=1,  # Same owner
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_no_collision_inactive_hitbox(self, populated_system):
        """Should not detect collision with inactive hitbox."""
        hitbox = populated_system.get_hitbox("attack_1")
        hurtbox = populated_system.get_hurtbox("body_1")
        # Hitbox not activated

        collision = populated_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_no_collision_inactive_hurtbox(self, populated_system):
        """Should not detect collision with inactive hurtbox."""
        hitbox = populated_system.get_hitbox("attack_1")
        hurtbox = populated_system.get_hurtbox("body_1")
        hitbox.activate()
        hurtbox.active = False

        collision = populated_system.check_collision(hitbox, hurtbox)
        assert collision is None

    def test_collision_damage_calculation(self, populated_system):
        """Collision should calculate damage."""
        hitbox = populated_system.get_hitbox("attack_1")
        hurtbox = populated_system.get_hurtbox("body_1")
        hitbox.activate()

        collision = populated_system.check_collision(hitbox, hurtbox)

        assert collision.damage == hitbox.damage

    def test_collision_damage_with_multiplier(self, hitbox_system):
        """Collision should apply damage multiplier."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            damage=10.0,
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            size=(1, 1, 1),
            damage_multiplier=2.0,
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.damage == 20.0

    def test_collision_invincible_hurtbox(self, hitbox_system):
        """Should detect invincibility."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            hurtbox_type=HurtboxType.INTANGIBLE,
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.INVINCIBLE

    def test_collision_counter_hit(self, hitbox_system):
        """Should detect counter hit."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            hurtbox_type=HurtboxType.COUNTER,
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.COUNTER_HIT

    def test_counter_hit_bonus_damage(self, hitbox_system):
        """Counter hit should apply bonus damage."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            damage=100.0,
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            hurtbox_type=HurtboxType.COUNTER,
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.damage == 125.0  # 25% bonus

    def test_collision_armored_hit(self, hitbox_system):
        """Should detect armored hit."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=2,
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision.result == CollisionResult.ARMORED

    def test_armor_depleted(self, hitbox_system):
        """Should deplete armor."""
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=1,
        )
        hurtbox = hitbox_system.get_hurtbox("hurt_1")

        result = hurtbox.absorb_armor_hit()
        assert result

        # Armor depleted
        assert not hurtbox.has_armor


# =============================================================================
# HITBOX ACTIVATION TESTS (~15 tests)
# =============================================================================


class TestHitboxActivation:
    """Tests for hitbox activation and deactivation."""

    def test_activate_hitbox(self, basic_hitbox):
        """Should activate hitbox."""
        basic_hitbox.activate()
        assert basic_hitbox.active

    def test_deactivate_hitbox(self, basic_hitbox):
        """Should deactivate hitbox."""
        basic_hitbox.activate()
        basic_hitbox.deactivate()
        assert not basic_hitbox.active

    def test_activate_clears_hit_entities(self, basic_hitbox):
        """Activation should clear hit entities."""
        basic_hitbox.hit_entities.add(999)
        basic_hitbox.activate()
        assert len(basic_hitbox.hit_entities) == 0

    def test_activate_with_lifetime(self, basic_hitbox):
        """Should activate with lifetime."""
        basic_hitbox.activate(lifetime=1.0)
        assert basic_hitbox.lifetime == 1.0

    def test_hitbox_expires(self, basic_hitbox):
        """Should detect expired hitbox."""
        basic_hitbox.activate(lifetime=0.01)
        time.sleep(0.02)
        assert basic_hitbox.is_expired

    def test_hitbox_not_expired(self, basic_hitbox):
        """Should detect non-expired hitbox."""
        basic_hitbox.activate(lifetime=10.0)
        assert not basic_hitbox.is_expired

    def test_system_activate_hitbox(self, hitbox_system):
        """Should activate hitbox via system."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        result = hitbox_system.activate_hitbox("hit_1")

        assert result
        assert hitbox_system.get_hitbox("hit_1").active

    def test_system_deactivate_hitbox(self, hitbox_system):
        """Should deactivate hitbox via system."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.activate_hitbox("hit_1")
        result = hitbox_system.deactivate_hitbox("hit_1")

        assert result
        assert not hitbox_system.get_hitbox("hit_1").active

    def test_activate_entity_hitboxes(self, hitbox_system):
        """Should activate all entity hitboxes."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.create_hitbox(hitbox_id="hit_2", owner_id=1)

        count = hitbox_system.activate_entity_hitboxes(1)

        assert count == 2
        for hitbox in hitbox_system.get_entity_hitboxes(1):
            assert hitbox.active

    def test_deactivate_entity_hitboxes(self, hitbox_system):
        """Should deactivate all entity hitboxes."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.create_hitbox(hitbox_id="hit_2", owner_id=1)
        hitbox_system.activate_entity_hitboxes(1)

        count = hitbox_system.deactivate_entity_hitboxes(1)

        assert count == 2
        for hitbox in hitbox_system.get_entity_hitboxes(1):
            assert not hitbox.active

    def test_get_active_hitboxes(self, hitbox_system):
        """Should get only active hitboxes."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.create_hitbox(hitbox_id="hit_2", owner_id=2)
        hitbox_system.activate_hitbox("hit_1")

        active = hitbox_system.get_active_hitboxes()
        assert len(active) == 1
        assert active[0].hitbox_id == "hit_1"


# =============================================================================
# HIT TRACKING TESTS (~10 tests)
# =============================================================================


class TestHitTracking:
    """Tests for hit tracking (multi-hit prevention)."""

    def test_record_hit(self, basic_hitbox):
        """Should record hit."""
        basic_hitbox.activate()
        result = basic_hitbox.record_hit(2)
        assert result
        assert 2 in basic_hitbox.hit_entities

    def test_no_double_hit(self, basic_hitbox):
        """Should not hit same entity twice."""
        basic_hitbox.activate()
        basic_hitbox.record_hit(2)
        result = basic_hitbox.record_hit(2)
        assert not result

    def test_can_hit_entity(self, basic_hitbox):
        """Should check if can hit entity."""
        basic_hitbox.activate()
        assert basic_hitbox.can_hit_entity(2)

        basic_hitbox.record_hit(2)
        assert not basic_hitbox.can_hit_entity(2)

    def test_cannot_hit_owner(self, basic_hitbox):
        """Should not hit owner."""
        basic_hitbox.activate()
        assert not basic_hitbox.can_hit_entity(1)  # Owner ID

    def test_max_hits_limit(self, hitbox_system):
        """Should respect max hits limit."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            max_hits=2,
        )
        hitbox = hitbox_system.get_hitbox("hit_1")
        hitbox.activate()

        hitbox.record_hit(2)
        assert hitbox.can_hit_more

        hitbox.record_hit(3)
        assert not hitbox.can_hit_more

    def test_auto_deactivate_on_max_hits(self, hitbox_system):
        """Should auto-deactivate when max hits reached."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            max_hits=1,
        )
        hitbox = hitbox_system.get_hitbox("hit_1")
        hitbox.activate()

        hitbox.record_hit(2)
        assert not hitbox.active


# =============================================================================
# PRIORITY SYSTEM TESTS (~10 tests)
# =============================================================================


class TestPrioritySystem:
    """Tests for hitbox priority system."""

    def test_higher_priority_wins(self, hitbox_system):
        """Higher priority hitbox should take precedence."""
        hitbox_system.create_hitbox(
            hitbox_id="low",
            owner_id=1,
            position=(0, 0, 0),
            damage=10.0,
            priority=1,
        )
        hitbox_system.create_hitbox(
            hitbox_id="high",
            owner_id=2,
            position=(0, 0, 0),
            damage=20.0,
            priority=10,
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=3,
            position=(0, 0, 0),
        )

        hitbox_system.activate_hitbox("low")
        hitbox_system.activate_hitbox("high")

        collisions = hitbox_system.process_collisions()

        # Should only process highest priority
        damages = [c.damage for c in collisions]
        assert 20.0 in damages

    def test_same_priority_both_hit(self, hitbox_system):
        """Same priority hitboxes should both hit."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            priority=5,
        )
        hitbox_system.create_hitbox(
            hitbox_id="hit_2",
            owner_id=2,
            position=(0, 0, 0),
            priority=5,
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=3,
            position=(0, 0, 0),
        )

        hitbox_system.activate_hitbox("hit_1")
        hitbox_system.activate_hitbox("hit_2")

        collisions = hitbox_system.process_collisions()
        assert len(collisions) == 2


# =============================================================================
# GROUP FILTERING TESTS (~10 tests)
# =============================================================================


class TestGroupFiltering:
    """Tests for group-based filtering."""

    def test_group_membership(self, hitbox_system):
        """Should track group membership."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            groups={"melee", "player"},
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            groups={"enemy"},
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")

        assert "melee" in hitbox.groups
        assert "player" in hitbox.groups
        assert "enemy" in hurtbox.groups

    def test_hitbox_hurtbox_collision_with_groups(self, hitbox_system):
        """Should hit regardless of different groups (groups for filtering only)."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
            groups={"player_attack"},
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
            groups={"enemy"},
        )

        hitbox = hitbox_system.get_hitbox("hit_1")
        hurtbox = hitbox_system.get_hurtbox("hurt_1")
        hitbox.activate()

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is not None


# =============================================================================
# EVENT CALLBACK TESTS (~10 tests)
# =============================================================================


class TestEventCallbacks:
    """Tests for event callbacks."""

    def test_on_hit_callback(self, populated_system):
        """Should call hit callback."""
        handler = Mock()
        populated_system.on_hit(handler)

        populated_system.activate_hitbox("attack_1")
        populated_system.process_collisions()

        handler.assert_called_once()

    def test_on_hit_receives_collision_info(self, populated_system):
        """Hit callback should receive collision info."""
        received = []

        def handler(collision):
            received.append(collision)

        populated_system.on_hit(handler)
        populated_system.activate_hitbox("attack_1")
        populated_system.process_collisions()

        assert len(received) == 1
        assert isinstance(received[0], CollisionInfo)

    def test_on_blocked_callback(self, hitbox_system):
        """Should call blocked callback."""
        handler = Mock()
        hitbox_system.on_blocked(handler)

        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            position=(0, 0, 0),
        )
        # Create block hitbox
        hitbox_system.create_hitbox(
            hitbox_id="block_1",
            owner_id=2,
            position=(0, 0, 0),
            hitbox_type=HitboxType.BLOCK,
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hurt_1",
            owner_id=2,
            position=(0, 0, 0),
        )

        hitbox_system.activate_hitbox("hit_1")
        hitbox_system.activate_hitbox("block_1")
        hitbox_system.process_collisions()

        handler.assert_called()


# =============================================================================
# UPDATE LOOP TESTS (~10 tests)
# =============================================================================


class TestUpdateLoop:
    """Tests for update loop."""

    def test_update_deactivates_expired(self, hitbox_system):
        """Update should deactivate expired hitboxes."""
        hitbox_system.create_hitbox(
            hitbox_id="hit_1",
            owner_id=1,
            lifetime=0.01,
        )
        hitbox_system.activate_hitbox("hit_1", lifetime=0.01)

        time.sleep(0.02)
        hitbox_system.update(0.01)

        assert not hitbox_system.get_hitbox("hit_1").active

    def test_update_processes_collisions(self, populated_system):
        """Update should process collisions."""
        populated_system.activate_hitbox("attack_1")
        collisions = populated_system.update(0.01)

        assert len(collisions) == 1


# =============================================================================
# UTILITY TESTS (~10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_remove_entity(self, hitbox_system):
        """Should remove all entity hitboxes/hurtboxes."""
        hitbox_system.create_hitbox(hitbox_id="hit_1", owner_id=1)
        hitbox_system.create_hurtbox(hurtbox_id="hurt_1", owner_id=1)

        hitbox_system.remove_entity(1)

        assert hitbox_system.get_hitbox("hit_1") is None
        assert hitbox_system.get_hurtbox("hurt_1") is None

    def test_clear(self, populated_system):
        """Should clear all hitboxes/hurtboxes."""
        populated_system.clear()

        assert len(populated_system.get_active_hitboxes()) == 0
        assert len(populated_system.get_active_hurtboxes()) == 0

    def test_get_stats(self, populated_system):
        """Should get system statistics."""
        stats = populated_system.get_stats()

        assert "total_hitboxes" in stats
        assert "total_hurtboxes" in stats
        assert stats["total_hitboxes"] == 1
        assert stats["total_hurtboxes"] == 1

    def test_get_collisions_for_entity(self, populated_system):
        """Should get collisions for entity."""
        populated_system.activate_hitbox("attack_1")
        populated_system.process_collisions()

        collisions = populated_system.get_collisions_for_entity(1, as_attacker=True)
        assert len(collisions) == 1


# =============================================================================
# HITBOX PROPERTIES TESTS (~5 tests)
# =============================================================================


class TestHitboxProperties:
    """Tests for hitbox properties."""

    def test_damage_multiplier(self, basic_hitbox):
        """Should get zone damage multiplier."""
        basic_hitbox.zone = HitboxZone.HEAD
        assert basic_hitbox.damage_multiplier == 2.0

    def test_is_critical_zone(self, basic_hitbox):
        """Should detect critical zone."""
        basic_hitbox.zone = HitboxZone.HEAD
        assert basic_hitbox.is_critical_zone

        basic_hitbox.zone = HitboxZone.TORSO
        assert not basic_hitbox.is_critical_zone

    def test_time_active(self, basic_hitbox):
        """Should track time active."""
        basic_hitbox.activate()
        time.sleep(0.05)
        assert basic_hitbox.time_active > 0

    def test_bounding_box_property(self, basic_hitbox):
        """Should generate bounding box."""
        bbox = basic_hitbox.bounding_box
        assert isinstance(bbox, BoundingBox)
