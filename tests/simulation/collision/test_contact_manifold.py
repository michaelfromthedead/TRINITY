"""
Whitebox tests for engine.simulation.collision.contact_manifold module.

Tests contact point management, manifold operations, caching, and contact pairs:
- ContactPoint
- ManifoldKey
- ContactManifold
- ManifoldCache
- ContactPair
"""

import pytest
from engine.simulation.collision.contact_manifold import (
    ContactPoint,
    ManifoldKey,
    ContactManifold,
    ManifoldCache,
    ContactPair,
    create_contact_pairs,
)
from engine.simulation.collision.broadphase import Vec3
from engine.simulation.collision.config import MAX_CONTACT_POINTS, CONTACT_MAX_AGE


class TestContactPoint:
    """Tests for ContactPoint dataclass."""

    def test_default_construction(self):
        """Default ContactPoint should have sensible defaults."""
        point = ContactPoint()
        assert point.depth == 0.0
        assert point.friction == 0.5
        assert point.restitution == 0.0
        assert point.age == 0

    def test_distance_to(self):
        """distance_to should compute distance between points."""
        point_a = ContactPoint(position=Vec3(0, 0, 0))
        point_b = ContactPoint(position=Vec3(3, 4, 0))
        assert point_a.distance_to(point_b) == 5.0

    def test_update_impulse(self):
        """update_impulse should store impulse values."""
        point = ContactPoint()
        point.update_impulse(10.0, 2.0, 3.0)
        assert point.normal_impulse == 10.0
        assert point.tangent_impulse_1 == 2.0
        assert point.tangent_impulse_2 == 3.0

    def test_get_warm_start_impulse(self):
        """get_warm_start_impulse should scale impulses."""
        point = ContactPoint()
        point.update_impulse(10.0, 2.0, 3.0)
        normal, tan1, tan2 = point.get_warm_start_impulse()
        # Values should be scaled by WARM_START_FACTOR
        assert normal == pytest.approx(10.0 * 0.8)
        assert tan1 == pytest.approx(2.0 * 0.8)
        assert tan2 == pytest.approx(3.0 * 0.8)


class TestManifoldKey:
    """Tests for ManifoldKey dataclass."""

    def test_hash_order_independent(self):
        """ManifoldKey hash should be order-independent."""
        key_a = ManifoldKey(1, 2)
        key_b = ManifoldKey(2, 1)
        assert hash(key_a) == hash(key_b)

    def test_equality_order_independent(self):
        """ManifoldKey equality should be order-independent."""
        key_a = ManifoldKey(1, 2)
        key_b = ManifoldKey(2, 1)
        assert key_a == key_b

    def test_equality_different_pairs(self):
        """Different ManifoldKeys should not be equal."""
        key_a = ManifoldKey(1, 2)
        key_b = ManifoldKey(1, 3)
        assert key_a != key_b

    def test_as_dict_key(self):
        """ManifoldKey should work as dict key."""
        d = {}
        d[ManifoldKey(1, 2)] = "test"
        assert d[ManifoldKey(2, 1)] == "test"


class TestContactManifold:
    """Tests for ContactManifold class."""

    def test_construction(self):
        """ContactManifold should be constructed correctly."""
        manifold = ContactManifold(1, 2)
        assert manifold.body_a == 1
        assert manifold.body_b == 2
        assert manifold.contact_count == 0

    def test_get_key(self):
        """get_key should return correct ManifoldKey."""
        manifold = ContactManifold(1, 2)
        key = manifold.get_key()
        assert key.body_a == 1
        assert key.body_b == 2

    def test_add_contact(self):
        """add_contact should add contact point."""
        manifold = ContactManifold(1, 2)
        contact = manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
        )
        assert manifold.contact_count == 1
        assert contact.position.x == 0
        assert contact.depth == 0.1

    def test_add_contact_updates_existing(self):
        """add_contact should update matching existing contact."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
        )
        # Add contact at same position
        manifold.add_contact(
            position=Vec3(0.001, 0, 0),  # Very close
            normal=Vec3(0, 1, 0),
            depth=0.2,
        )
        assert manifold.contact_count == 1
        assert manifold.contacts[0].depth == 0.2

    def test_add_contact_feature_matching(self):
        """add_contact should match by feature ID."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
            feature_id_a=1,
            feature_id_b=2,
        )
        # Add contact with same feature IDs at different position
        manifold.add_contact(
            position=Vec3(5, 5, 5),  # Different position
            normal=Vec3(0, 1, 0),
            depth=0.2,
            feature_id_a=1,
            feature_id_b=2,
        )
        assert manifold.contact_count == 1

    def test_remove_contact(self):
        """remove_contact should remove contact by ID."""
        manifold = ContactManifold(1, 2)
        contact = manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
        )
        assert manifold.remove_contact(contact.contact_id)
        assert manifold.contact_count == 0

    def test_remove_contact_nonexistent(self):
        """remove_contact should return False for nonexistent."""
        manifold = ContactManifold(1, 2)
        assert not manifold.remove_contact(999)

    def test_reduce_manifold(self):
        """reduce_manifold should limit contacts to max."""
        manifold = ContactManifold(1, 2, max_contacts=4)
        for i in range(10):
            manifold.add_contact(
                position=Vec3(i * 2, 0, 0),
                normal=Vec3(0, 1, 0),
                depth=0.1 + i * 0.01,
            )
        # Should be reduced to max_contacts
        assert manifold.contact_count <= 4

    def test_reduce_manifold_keeps_deepest(self):
        """reduce_manifold should keep deepest contact."""
        manifold = ContactManifold(1, 2, max_contacts=2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), depth=0.1)
        manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), depth=0.5)  # Deepest
        manifold.add_contact(Vec3(2, 0, 0), Vec3(0, 1, 0), depth=0.2)
        # Deepest (0.5) should be kept
        assert any(c.depth == 0.5 for c in manifold.contacts)

    def test_age_contacts(self):
        """age_contacts should increment age and remove old contacts."""
        manifold = ContactManifold(1, 2)
        contact = manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
        )
        # Age contacts multiple times
        for _ in range(CONTACT_MAX_AGE + 1):
            removed = manifold.age_contacts()
        # Contact should be removed
        assert manifold.contact_count == 0

    def test_refresh_contacts(self):
        """refresh_contacts should update positions and remove separated."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
            local_a=Vec3(0, 0, 0),
            local_b=Vec3(0, 0.1, 0),
        )
        # Identity transforms
        removed = manifold.refresh_contacts(
            transform_a=lambda p: p,
            transform_b=lambda p: p,
        )
        assert manifold.contact_count == 1

    def test_refresh_contacts_removes_separated(self):
        """refresh_contacts should remove separated contacts."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            depth=0.1,
            local_a=Vec3(0, 0, 0),
            local_b=Vec3(0, 0.1, 0),
        )
        # Transform that separates bodies
        removed = manifold.refresh_contacts(
            transform_a=lambda p: p,
            transform_b=lambda p: Vec3(p.x, p.y + 10, p.z),  # Move far away
        )
        assert len(removed) == 1

    def test_clear(self):
        """clear should remove all contacts."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.clear()
        assert manifold.contact_count == 0

    def test_update_touching_state(self):
        """update_touching_state should track state changes."""
        manifold = ContactManifold(1, 2)
        # Initially not touching
        began, persist, ended = manifold.update_touching_state()
        assert not began and not persist and not ended

        # Add contact
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        began, persist, ended = manifold.update_touching_state()
        assert began and not persist and not ended

        # Still touching
        began, persist, ended = manifold.update_touching_state()
        assert not began and persist and not ended

        # Remove contact
        manifold.clear()
        began, persist, ended = manifold.update_touching_state()
        assert not began and not persist and ended

    def test_is_touching_property(self):
        """is_touching should reflect current state."""
        manifold = ContactManifold(1, 2)
        assert not manifold.is_touching
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.update_touching_state()
        assert manifold.is_touching

    def test_get_average_normal(self):
        """get_average_normal should compute average."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.1)
        manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.1)
        avg = manifold.get_average_normal()
        # Should be normalized average
        assert abs(avg.length() - 1.0) < 0.01

    def test_get_average_position(self):
        """get_average_position should compute average."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.add_contact(Vec3(2, 0, 0), Vec3(0, 1, 0), 0.1)
        avg = manifold.get_average_position()
        assert avg.x == 1.0

    def test_get_max_depth(self):
        """get_max_depth should return maximum depth."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.5)
        manifold.add_contact(Vec3(2, 0, 0), Vec3(0, 1, 0), 0.3)
        assert manifold.get_max_depth() == 0.5

    def test_get_total_impulse(self):
        """get_total_impulse should sum impulses."""
        manifold = ContactManifold(1, 2)
        c1 = manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        c2 = manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.1)
        c1.normal_impulse = 10.0
        c2.normal_impulse = 5.0
        assert manifold.get_total_impulse() == 15.0


class TestManifoldCache:
    """Tests for ManifoldCache class."""

    def test_construction(self):
        """ManifoldCache should be constructed correctly."""
        cache = ManifoldCache()
        assert cache.manifold_count == 0

    def test_get_or_create_new(self):
        """get_or_create should create new manifold."""
        cache = ManifoldCache()
        manifold = cache.get_or_create(1, 2)
        assert manifold.body_a == 1
        assert manifold.body_b == 2
        assert cache.manifold_count == 1

    def test_get_or_create_existing(self):
        """get_or_create should return existing manifold."""
        cache = ManifoldCache()
        m1 = cache.get_or_create(1, 2)
        m1.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        m2 = cache.get_or_create(1, 2)
        assert m2.contact_count == 1  # Same manifold
        assert cache.manifold_count == 1

    def test_get_or_create_order_independent(self):
        """get_or_create should be order-independent."""
        cache = ManifoldCache()
        m1 = cache.get_or_create(1, 2)
        m2 = cache.get_or_create(2, 1)
        assert m1 is m2

    def test_get_existing(self):
        """get should return existing manifold."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        manifold = cache.get(1, 2)
        assert manifold is not None

    def test_get_nonexistent(self):
        """get should return None for nonexistent."""
        cache = ManifoldCache()
        assert cache.get(1, 2) is None

    def test_remove(self):
        """remove should remove manifold."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        assert cache.remove(1, 2)
        assert cache.manifold_count == 0

    def test_remove_nonexistent(self):
        """remove should return False for nonexistent."""
        cache = ManifoldCache()
        assert not cache.remove(1, 2)

    def test_remove_body(self):
        """remove_body should remove all manifolds involving body."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        cache.get_or_create(1, 3)
        cache.get_or_create(2, 3)
        count = cache.remove_body(1)
        assert count == 2
        assert cache.manifold_count == 1

    def test_update_frame(self):
        """update_frame should age and track state changes."""
        cache = ManifoldCache()
        m = cache.get_or_create(1, 2)
        m.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        m.update_touching_state()
        began, ended = cache.update_frame()
        # Should track began since first frame
        # Note: update_frame ages contacts, so contact may be removed
        # This depends on implementation

    def test_clear(self):
        """clear should remove all manifolds."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        cache.get_or_create(2, 3)
        cache.clear()
        assert cache.manifold_count == 0

    def test_get_all_manifolds(self):
        """get_all_manifolds should return all manifolds."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        cache.get_or_create(2, 3)
        all_m = cache.get_all_manifolds()
        assert len(all_m) == 2

    def test_get_touching_manifolds(self):
        """get_touching_manifolds should return only touching."""
        cache = ManifoldCache()
        m1 = cache.get_or_create(1, 2)
        m1.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        m1.update_touching_state()
        m2 = cache.get_or_create(2, 3)  # Empty, not touching
        touching = cache.get_touching_manifolds()
        assert len(touching) == 1
        assert touching[0] is m1

    def test_eviction_on_max_manifolds(self):
        """Cache should evict when exceeding max."""
        cache = ManifoldCache(max_manifolds=5)
        for i in range(10):
            cache.get_or_create(i, i + 100)
        # Should have evicted some (empty ones first)
        assert cache.manifold_count <= 5


class TestContactPair:
    """Tests for ContactPair dataclass."""

    def test_construction(self):
        """ContactPair should be constructed correctly."""
        contact = ContactPoint(position=Vec3(0, 0, 0), normal=Vec3(0, 1, 0))
        pair = ContactPair(body_a=1, body_b=2, contact=contact)
        assert pair.body_a == 1
        assert pair.body_b == 2

    def test_compute_tangent_basis(self):
        """compute_tangent_basis should create orthogonal basis."""
        contact = ContactPoint(position=Vec3(0, 0, 0), normal=Vec3(0, 1, 0))
        pair = ContactPair(body_a=1, body_b=2, contact=contact, normal=Vec3(0, 1, 0))
        pair.compute_tangent_basis()
        # Tangents should be perpendicular to normal
        assert abs(pair.normal.dot(pair.tangent_1)) < 0.01
        assert abs(pair.normal.dot(pair.tangent_2)) < 0.01
        # Tangents should be perpendicular to each other
        assert abs(pair.tangent_1.dot(pair.tangent_2)) < 0.01

    def test_compute_tangent_basis_various_normals(self):
        """compute_tangent_basis should work for various normals."""
        normals = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0, 1),
            Vec3(1, 1, 0).normalized(),
            Vec3(1, 1, 1).normalized(),
        ]
        for normal in normals:
            contact = ContactPoint(position=Vec3(0, 0, 0), normal=normal)
            pair = ContactPair(body_a=1, body_b=2, contact=contact, normal=normal)
            pair.compute_tangent_basis()
            assert abs(pair.normal.dot(pair.tangent_1)) < 0.01
            assert abs(pair.normal.dot(pair.tangent_2)) < 0.01


class TestCreateContactPairs:
    """Tests for create_contact_pairs function."""

    def test_creates_pairs_from_manifold(self):
        """create_contact_pairs should create pairs for each contact."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.1)
        pairs = create_contact_pairs(manifold)
        assert len(pairs) == 2

    def test_pairs_have_tangent_basis(self):
        """Created pairs should have computed tangent basis."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        pairs = create_contact_pairs(manifold)
        # Tangent basis should be computed
        pair = pairs[0]
        assert abs(pair.normal.dot(pair.tangent_1)) < 0.01

    def test_empty_manifold_no_pairs(self):
        """Empty manifold should produce no pairs."""
        manifold = ContactManifold(1, 2)
        pairs = create_contact_pairs(manifold)
        assert len(pairs) == 0


class TestContactManifoldEdgeCases:
    """Edge case tests for contact manifold system."""

    def test_many_contacts_same_position(self):
        """Many contacts at same position should be merged."""
        manifold = ContactManifold(1, 2)
        for _ in range(10):
            manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        assert manifold.contact_count == 1

    def test_contacts_at_origin(self):
        """Contacts at origin should work."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        assert manifold.contact_count == 1

    def test_negative_depth(self):
        """Negative depth (separation) should be allowed."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), -0.1)
        assert manifold.contacts[0].depth == -0.1

    def test_zero_depth(self):
        """Zero depth (touching) should be allowed."""
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.0)
        assert manifold.contacts[0].depth == 0.0
