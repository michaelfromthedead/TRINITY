"""Tests for PhysicsAsset."""
import pytest

from engine.resource.types.physics_asset import PhysicsAsset, ColliderType


def _phys(**kw):
    defaults = dict(
        asset_id=80, name="box_col", path="/p.phys", size_bytes=64,
        collider_type=ColliderType.BOX, dimensions=(1.0, 1.0, 1.0), mass=10.0,
    )
    defaults.update(kw)
    return PhysicsAsset(**defaults)


class TestPhysicsAsset:
    def test_creation(self):
        p = _phys()
        assert p.collider_type is ColliderType.BOX
        assert p.dimensions == (1.0, 1.0, 1.0)
        assert p.mass == pytest.approx(10.0)

    def test_static_body(self):
        p = _phys(is_static=True, mass=0.0)
        assert p.is_static is True

    def test_default_friction_restitution(self):
        p = _phys()
        assert p.friction == pytest.approx(0.5)
        assert p.restitution == pytest.approx(0.3)

    def test_custom_friction_restitution(self):
        p = _phys(friction=0.8, restitution=0.1)
        assert p.friction == pytest.approx(0.8)
        assert p.restitution == pytest.approx(0.1)

    def test_collider_types_count(self):
        for name in ("BOX", "SPHERE", "CAPSULE", "MESH", "CONVEX"):
            assert hasattr(ColliderType, name), f"ColliderType.{name} missing"

    def test_load_unload(self):
        p = _phys()
        assert not p.is_loaded()
        p.load(b"")
        assert p.is_loaded()
        p.unload()
        assert not p.is_loaded()
