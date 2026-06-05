"""
Whitebox tests for collider components.

Tests cover:
- ColliderComponent base class
- SphereCollider
- BoxCollider
- CapsuleCollider
- MeshCollider
- PhysicsMaterial
- Bounds calculations
- Volume and inertia calculations
"""

import math
import pytest

from engine.simulation.character.character_controller import (
    Quaternion,
    Transform,
    Vector3,
)
from engine.simulation.components.collider_components import (
    BoxCollider,
    CapsuleCollider,
    ColliderComponent,
    ColliderType,
    MeshCollider,
    PhysicsMaterial,
    SphereCollider,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sphere_collider() -> SphereCollider:
    """Create a sphere collider."""
    return SphereCollider(entity_id=1, radius=1.0)


@pytest.fixture
def box_collider() -> BoxCollider:
    """Create a box collider."""
    return BoxCollider(
        entity_id=2,
        half_extents=Vector3(1.0, 1.0, 1.0),
    )


@pytest.fixture
def capsule_collider() -> CapsuleCollider:
    """Create a capsule collider."""
    return CapsuleCollider(
        entity_id=3,
        radius=0.5,
        height=2.0,
    )


@pytest.fixture
def mesh_collider() -> MeshCollider:
    """Create a mesh collider with simple cube."""
    vertices = [
        Vector3(-1, -1, -1), Vector3(1, -1, -1),
        Vector3(1, 1, -1), Vector3(-1, 1, -1),
        Vector3(-1, -1, 1), Vector3(1, -1, 1),
        Vector3(1, 1, 1), Vector3(-1, 1, 1),
    ]
    indices = [
        0, 1, 2, 2, 3, 0,  # Front
        4, 5, 6, 6, 7, 4,  # Back
        0, 4, 7, 7, 3, 0,  # Left
        1, 5, 6, 6, 2, 1,  # Right
        3, 2, 6, 6, 7, 3,  # Top
        0, 1, 5, 5, 4, 0,  # Bottom
    ]
    return MeshCollider(
        entity_id=4,
        vertices=vertices,
        indices=indices,
    )


@pytest.fixture
def identity_transform() -> Transform:
    """Identity transform."""
    return Transform()


# =============================================================================
# PhysicsMaterial Tests
# =============================================================================


class TestPhysicsMaterial:
    """Tests for PhysicsMaterial dataclass."""

    def test_default_values(self):
        """Test default material values."""
        mat = PhysicsMaterial()

        assert mat.friction == 0.5
        assert mat.restitution == 0.0
        assert mat.friction_combine == "average"
        assert mat.restitution_combine == "average"

    def test_custom_values(self):
        """Test custom material values."""
        mat = PhysicsMaterial(
            friction=0.8,
            restitution=0.5,
            friction_combine="minimum",
            restitution_combine="maximum",
        )

        assert mat.friction == 0.8
        assert mat.restitution == 0.5
        assert mat.friction_combine == "minimum"
        assert mat.restitution_combine == "maximum"

    def test_zero_friction(self):
        """Test ice-like material."""
        mat = PhysicsMaterial(friction=0.0)
        assert mat.friction == 0.0

    def test_bouncy_material(self):
        """Test bouncy material."""
        mat = PhysicsMaterial(restitution=1.0)
        assert mat.restitution == 1.0


# =============================================================================
# ColliderType Tests
# =============================================================================


class TestColliderType:
    """Tests for ColliderType enum."""

    def test_all_types_exist(self):
        """Test all collider types exist."""
        assert ColliderType.SPHERE.value == "sphere"
        assert ColliderType.BOX.value == "box"
        assert ColliderType.CAPSULE.value == "capsule"
        assert ColliderType.MESH.value == "mesh"
        assert ColliderType.CONVEX_HULL.value == "convex_hull"
        assert ColliderType.TERRAIN.value == "terrain"
        assert ColliderType.COMPOUND.value == "compound"


# =============================================================================
# SphereCollider Tests
# =============================================================================


class TestSphereCollider:
    """Tests for SphereCollider."""

    def test_creation(self, sphere_collider):
        """Test sphere collider creation."""
        assert sphere_collider.entity_id == 1
        assert sphere_collider.radius == 1.0
        assert sphere_collider.collider_type == ColliderType.SPHERE

    def test_radius_property(self, sphere_collider):
        """Test radius getter and setter."""
        sphere_collider.radius = 2.5
        assert sphere_collider.radius == 2.5

    def test_radius_clamped_to_minimum(self, sphere_collider):
        """Test radius is clamped to minimum value."""
        sphere_collider.radius = 0.0
        assert sphere_collider.radius == 0.001

        sphere_collider.radius = -1.0
        assert sphere_collider.radius == 0.001

    def test_default_trigger(self):
        """Test default is_trigger is False."""
        collider = SphereCollider(entity_id=1)
        assert collider.is_trigger is False

    def test_trigger_collider(self):
        """Test trigger collider."""
        collider = SphereCollider(entity_id=1, is_trigger=True)
        assert collider.is_trigger is True

    def test_volume(self, sphere_collider):
        """Test sphere volume calculation."""
        # V = (4/3) * pi * r^3
        expected = (4.0 / 3.0) * math.pi * 1.0**3
        assert abs(sphere_collider.get_volume() - expected) < 0.001

    def test_volume_large_radius(self):
        """Test volume with large radius."""
        collider = SphereCollider(entity_id=1, radius=10.0)
        expected = (4.0 / 3.0) * math.pi * 10.0**3
        assert abs(collider.get_volume() - expected) < 0.1

    def test_inertia_tensor(self, sphere_collider):
        """Test inertia tensor calculation."""
        mass = 10.0
        inertia = sphere_collider.get_inertia_tensor(mass)

        # I = (2/5) * m * r^2 for solid sphere
        expected = (2.0 / 5.0) * mass * 1.0**2
        assert abs(inertia.x - expected) < 0.001
        assert abs(inertia.y - expected) < 0.001
        assert abs(inertia.z - expected) < 0.001

    def test_bounds(self, sphere_collider, identity_transform):
        """Test bounds calculation."""
        bounds = sphere_collider.get_bounds(identity_transform)
        min_pt, max_pt = bounds

        assert min_pt.x == -1.0
        assert min_pt.y == -1.0
        assert min_pt.z == -1.0
        assert max_pt.x == 1.0
        assert max_pt.y == 1.0
        assert max_pt.z == 1.0

    def test_bounds_with_offset(self, sphere_collider):
        """Test bounds with local offset."""
        sphere_collider.local_position = Vector3(5.0, 0.0, 0.0)
        transform = Transform(position=Vector3(0.0, 0.0, 0.0))

        bounds = sphere_collider.get_bounds(transform)
        min_pt, max_pt = bounds

        assert min_pt.x == 4.0
        assert max_pt.x == 6.0

    def test_get_state(self, sphere_collider):
        """Test serialization."""
        sphere_collider.radius = 2.5
        state = sphere_collider.get_state()

        assert state["type"] == "sphere"
        assert state["radius"] == 2.5
        assert state["is_trigger"] is False


# =============================================================================
# BoxCollider Tests
# =============================================================================


class TestBoxCollider:
    """Tests for BoxCollider."""

    def test_creation(self, box_collider):
        """Test box collider creation."""
        assert box_collider.entity_id == 2
        assert box_collider.half_extents.x == 1.0
        assert box_collider.collider_type == ColliderType.BOX

    def test_default_half_extents(self):
        """Test default half extents."""
        collider = BoxCollider(entity_id=1)
        assert collider.half_extents.x == 0.5
        assert collider.half_extents.y == 0.5
        assert collider.half_extents.z == 0.5

    def test_half_extents_setter(self, box_collider):
        """Test setting half extents."""
        box_collider.half_extents = Vector3(2.0, 3.0, 4.0)
        assert box_collider.half_extents.x == 2.0
        assert box_collider.half_extents.y == 3.0
        assert box_collider.half_extents.z == 4.0

    def test_half_extents_clamped(self, box_collider):
        """Test half extents are clamped to minimum."""
        box_collider.half_extents = Vector3(0.0, -1.0, 0.5)
        assert box_collider.half_extents.x == 0.001
        assert box_collider.half_extents.y == 0.001
        assert box_collider.half_extents.z == 0.5

    def test_size_property(self, box_collider):
        """Test full size property."""
        box_collider.half_extents = Vector3(1.0, 2.0, 3.0)

        size = box_collider.size
        assert size.x == 2.0
        assert size.y == 4.0
        assert size.z == 6.0

    def test_size_setter(self, box_collider):
        """Test setting size updates half extents."""
        box_collider.size = Vector3(4.0, 6.0, 8.0)

        assert box_collider.half_extents.x == 2.0
        assert box_collider.half_extents.y == 3.0
        assert box_collider.half_extents.z == 4.0

    def test_volume(self, box_collider):
        """Test box volume calculation."""
        # V = 8 * hx * hy * hz = full width * height * depth
        expected = 8.0 * 1.0 * 1.0 * 1.0
        assert box_collider.get_volume() == expected

    def test_volume_rectangular(self):
        """Test volume of rectangular box."""
        collider = BoxCollider(
            entity_id=1,
            half_extents=Vector3(1.0, 2.0, 3.0),
        )
        expected = 8.0 * 1.0 * 2.0 * 3.0
        assert collider.get_volume() == expected

    def test_inertia_tensor(self, box_collider):
        """Test inertia tensor calculation."""
        mass = 12.0
        inertia = box_collider.get_inertia_tensor(mass)

        # I_xx = (m/3) * (hy^2 + hz^2), etc.
        factor = mass / 3.0
        expected_x = factor * (1.0**2 + 1.0**2)
        assert abs(inertia.x - expected_x) < 0.001

    def test_bounds(self, box_collider, identity_transform):
        """Test bounds calculation."""
        bounds = box_collider.get_bounds(identity_transform)
        min_pt, max_pt = bounds

        assert min_pt.x == -1.0
        assert max_pt.x == 1.0

    def test_get_state(self, box_collider):
        """Test serialization."""
        state = box_collider.get_state()

        assert state["type"] == "box"
        assert state["half_extents"] == (1.0, 1.0, 1.0)


# =============================================================================
# CapsuleCollider Tests
# =============================================================================


class TestCapsuleCollider:
    """Tests for CapsuleCollider."""

    def test_creation(self, capsule_collider):
        """Test capsule collider creation."""
        assert capsule_collider.entity_id == 3
        assert capsule_collider.radius == 0.5
        assert capsule_collider.height == 2.0
        assert capsule_collider.collider_type == ColliderType.CAPSULE

    def test_default_values(self):
        """Test default capsule values."""
        collider = CapsuleCollider(entity_id=1)
        assert collider.radius == 0.35
        assert collider.height == 1.8

    def test_radius_setter(self, capsule_collider):
        """Test setting radius."""
        capsule_collider.radius = 1.0
        assert capsule_collider.radius == 1.0

    def test_radius_clamped(self, capsule_collider):
        """Test radius is clamped."""
        capsule_collider.radius = 0.0
        assert capsule_collider.radius == 0.001

    def test_height_setter(self, capsule_collider):
        """Test setting height."""
        capsule_collider.height = 3.0
        assert capsule_collider.height == 3.0

    def test_height_minimum(self, capsule_collider):
        """Test height is clamped to at least 2*radius."""
        capsule_collider.radius = 0.5
        capsule_collider.height = 0.5  # Less than 2*radius

        # Should be at least 2*radius + 0.001
        assert capsule_collider.height >= 2 * capsule_collider.radius

    def test_cylinder_height(self, capsule_collider):
        """Test cylinder height (excluding caps)."""
        # height = 2.0, radius = 0.5
        # cylinder_height = height - 2*radius = 2.0 - 1.0 = 1.0
        assert capsule_collider.cylinder_height == 1.0

    def test_cylinder_height_all_caps(self):
        """Test cylinder height when it's all caps."""
        collider = CapsuleCollider(entity_id=1, radius=1.0, height=2.0)
        assert collider.cylinder_height == 0.0

    def test_volume(self, capsule_collider):
        """Test capsule volume calculation."""
        r = 0.5
        h = 1.0  # cylinder height

        cylinder = math.pi * r**2 * h
        sphere = (4.0 / 3.0) * math.pi * r**3
        expected = cylinder + sphere

        assert abs(capsule_collider.get_volume() - expected) < 0.001

    def test_inertia_tensor(self, capsule_collider):
        """Test inertia tensor is computed."""
        mass = 10.0
        inertia = capsule_collider.get_inertia_tensor(mass)

        # Just verify it's non-zero and reasonable
        assert inertia.x > 0
        assert inertia.y > 0
        assert inertia.z > 0

    def test_bounds(self, capsule_collider, identity_transform):
        """Test bounds calculation."""
        bounds = capsule_collider.get_bounds(identity_transform)
        min_pt, max_pt = bounds

        # Half height = 1.0
        assert min_pt.y == -1.0
        assert max_pt.y == 1.0
        # Radius = 0.5
        assert min_pt.x == -0.5
        assert max_pt.x == 0.5

    def test_get_state(self, capsule_collider):
        """Test serialization."""
        state = capsule_collider.get_state()

        assert state["type"] == "capsule"
        assert state["radius"] == 0.5
        assert state["height"] == 2.0


# =============================================================================
# MeshCollider Tests
# =============================================================================


class TestMeshCollider:
    """Tests for MeshCollider."""

    def test_creation(self, mesh_collider):
        """Test mesh collider creation."""
        assert mesh_collider.entity_id == 4
        assert mesh_collider.collider_type == ColliderType.MESH
        assert mesh_collider.is_convex is False

    def test_convex_hull_type(self):
        """Test convex mesh type."""
        collider = MeshCollider(entity_id=1, convex=True)
        assert collider.collider_type == ColliderType.CONVEX_HULL
        assert collider.is_convex is True

    def test_vertex_count(self, mesh_collider):
        """Test vertex count property."""
        assert mesh_collider.vertex_count == 8

    def test_triangle_count(self, mesh_collider):
        """Test triangle count property."""
        # 36 indices / 3 = 12 triangles
        assert mesh_collider.triangle_count == 12

    def test_empty_mesh(self):
        """Test empty mesh collider."""
        collider = MeshCollider(entity_id=1)
        assert collider.vertex_count == 0
        assert collider.triangle_count == 0

    def test_set_mesh(self, mesh_collider):
        """Test setting new mesh data."""
        new_verts = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0.5, 1, 0),
        ]
        new_indices = [0, 1, 2]

        mesh_collider.set_mesh(new_verts, new_indices, convex=True)

        assert mesh_collider.vertex_count == 3
        assert mesh_collider.triangle_count == 1
        assert mesh_collider.is_convex is True

    def test_bounds(self, mesh_collider, identity_transform):
        """Test bounds calculation."""
        bounds = mesh_collider.get_bounds(identity_transform)
        min_pt, max_pt = bounds

        # Cube from -1 to 1
        assert min_pt.x == -1.0
        assert min_pt.y == -1.0
        assert min_pt.z == -1.0
        assert max_pt.x == 1.0
        assert max_pt.y == 1.0
        assert max_pt.z == 1.0

    def test_bounds_empty_mesh(self):
        """Test bounds of empty mesh."""
        collider = MeshCollider(entity_id=1)
        transform = Transform()
        bounds = collider.get_bounds(transform)
        min_pt, max_pt = bounds

        # Should be zero
        assert min_pt.x == 0.0
        assert max_pt.x == 0.0

    def test_volume_approximation(self, mesh_collider):
        """Test volume approximation using bounding box."""
        volume = mesh_collider.get_volume()
        # Bounding box is 2x2x2 = 8
        assert volume == 8.0

    def test_inertia_approximation(self, mesh_collider):
        """Test inertia approximation."""
        mass = 10.0
        inertia = mesh_collider.get_inertia_tensor(mass)

        assert inertia.x > 0
        assert inertia.y > 0
        assert inertia.z > 0

    def test_get_state(self, mesh_collider):
        """Test serialization."""
        state = mesh_collider.get_state()

        assert state["type"] == "mesh"
        assert state["convex"] is False
        assert state["vertex_count"] == 8
        assert state["triangle_count"] == 12


# =============================================================================
# Common ColliderComponent Tests
# =============================================================================


class TestColliderComponentCommon:
    """Tests for common ColliderComponent functionality."""

    def test_enabled_property(self, sphere_collider):
        """Test enabled property."""
        assert sphere_collider.enabled is True

        sphere_collider.enabled = False
        assert sphere_collider.enabled is False

    def test_is_trigger_setter(self, sphere_collider):
        """Test setting trigger mode."""
        sphere_collider.is_trigger = True
        assert sphere_collider.is_trigger is True

    def test_material_property(self, sphere_collider):
        """Test material property."""
        assert sphere_collider.material.friction == 0.5

    def test_set_material(self, sphere_collider):
        """Test setting material."""
        new_mat = PhysicsMaterial(friction=0.9, restitution=0.8)
        sphere_collider.set_material(new_mat)

        assert sphere_collider.material.friction == 0.9
        assert sphere_collider.material.restitution == 0.8

    def test_local_position(self, sphere_collider):
        """Test local position offset."""
        sphere_collider.local_position = Vector3(1.0, 2.0, 3.0)

        assert sphere_collider.local_position.x == 1.0
        assert sphere_collider.local_position.y == 2.0
        assert sphere_collider.local_position.z == 3.0

    def test_local_rotation(self, sphere_collider):
        """Test local rotation offset."""
        rotation = Quaternion(0.0, 0.707, 0.0, 0.707)
        sphere_collider.local_rotation = rotation

        assert sphere_collider.local_rotation.y == 0.707

    def test_collision_filter(self, sphere_collider):
        """Test collision layer and mask."""
        sphere_collider.set_collision_filter(layer=2, mask=0x00FF)

        assert sphere_collider._collision_layer == 2
        assert sphere_collider._collision_mask == 0x00FF

    def test_initialize(self, sphere_collider):
        """Test initialization with physics ID."""
        assert sphere_collider.collider_id is None

        sphere_collider.initialize(collider_id=42)

        assert sphere_collider.collider_id == 42

    def test_cleanup(self, sphere_collider):
        """Test cleanup."""
        sphere_collider.initialize(collider_id=42)
        sphere_collider.cleanup()

        assert sphere_collider.collider_id is None

    def test_get_state_common(self, sphere_collider):
        """Test common state fields."""
        sphere_collider.local_position = Vector3(1.0, 2.0, 3.0)
        sphere_collider.is_trigger = True

        state = sphere_collider.get_state()

        assert state["entity_id"] == 1
        assert state["local_position"] == (1.0, 2.0, 3.0)
        assert state["is_trigger"] is True
        assert state["enabled"] is True
        assert "material" in state


# =============================================================================
# Custom Material Integration Tests
# =============================================================================


class TestMaterialIntegration:
    """Tests for material integration with colliders."""

    def test_sphere_with_custom_material(self):
        """Test sphere collider with custom material."""
        mat = PhysicsMaterial(friction=0.1, restitution=0.9)
        collider = SphereCollider(entity_id=1, material=mat)

        assert collider.material.friction == 0.1
        assert collider.material.restitution == 0.9

    def test_box_with_custom_material(self):
        """Test box collider with custom material."""
        mat = PhysicsMaterial(friction=1.0, restitution=0.0)
        collider = BoxCollider(entity_id=1, material=mat)

        assert collider.material.friction == 1.0

    def test_capsule_with_custom_material(self):
        """Test capsule collider with custom material."""
        mat = PhysicsMaterial(friction=0.3)
        collider = CapsuleCollider(entity_id=1, material=mat)

        assert collider.material.friction == 0.3

    def test_mesh_with_custom_material(self):
        """Test mesh collider with custom material."""
        mat = PhysicsMaterial(friction=0.7)
        collider = MeshCollider(entity_id=1, material=mat)

        assert collider.material.friction == 0.7


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_sphere(self):
        """Test very small sphere."""
        collider = SphereCollider(entity_id=1, radius=0.001)
        assert collider.radius == 0.001
        assert collider.get_volume() > 0

    def test_very_large_sphere(self):
        """Test very large sphere."""
        collider = SphereCollider(entity_id=1, radius=1000.0)
        volume = collider.get_volume()
        assert volume > 0

    def test_thin_box(self):
        """Test thin box (plane-like)."""
        collider = BoxCollider(
            entity_id=1,
            half_extents=Vector3(10.0, 0.01, 10.0),
        )
        volume = collider.get_volume()
        assert volume > 0

    def test_long_capsule(self):
        """Test long capsule."""
        collider = CapsuleCollider(entity_id=1, radius=0.1, height=100.0)
        volume = collider.get_volume()
        assert volume > 0

    def test_single_triangle_mesh(self):
        """Test mesh with single triangle."""
        vertices = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(0, 1, 0),
        ]
        indices = [0, 1, 2]
        collider = MeshCollider(entity_id=1, vertices=vertices, indices=indices)

        assert collider.triangle_count == 1
        assert collider.vertex_count == 3

    def test_degenerate_triangle(self):
        """Test mesh with degenerate (zero area) triangle."""
        vertices = [
            Vector3(0, 0, 0),
            Vector3(0, 0, 0),  # Same point
            Vector3(1, 0, 0),
        ]
        indices = [0, 1, 2]
        collider = MeshCollider(entity_id=1, vertices=vertices, indices=indices)

        # Should still create without error
        assert collider.triangle_count == 1

    def test_transformed_bounds(self):
        """Test bounds with non-identity transform."""
        collider = SphereCollider(entity_id=1, radius=1.0)
        transform = Transform(
            position=Vector3(10.0, 20.0, 30.0),
            scale=Vector3(2.0, 2.0, 2.0),
        )

        bounds = collider.get_bounds(transform)
        min_pt, max_pt = bounds

        # Center should be at (10, 20, 30) with radius 1 (scale not applied to bounds in simple impl)
        assert min_pt.x == 9.0
        assert max_pt.x == 11.0
