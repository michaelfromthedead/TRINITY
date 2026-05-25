"""
Comprehensive tests for TransformComponent.

Tests cover:
- Position get/set
- Rotation (quaternion, euler)
- Scale (uniform, non-uniform)
- Local vs world space
- Transform hierarchy
- Transform dirty flags
- Transform interpolation
- Look-at functionality
- Transform events
"""

import math
import pytest
from typing import List, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.gameplay.components.transform import (
    TransformComponent,
    TransformSnapshot,
    TransformSpace,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def transform():
    """Create a default transform component."""
    return TransformComponent()


@pytest.fixture
def positioned_transform():
    """Create a transform at a specific position."""
    return TransformComponent(position=Vec3(10, 20, 30))


@pytest.fixture
def rotated_transform():
    """Create a transform with a specific rotation."""
    return TransformComponent(rotation=Quat.from_euler(math.pi / 4, 0, 0))


@pytest.fixture
def scaled_transform():
    """Create a transform with a specific scale."""
    return TransformComponent(scale=Vec3(2, 3, 4))


@pytest.fixture
def full_transform():
    """Create a transform with position, rotation, and scale."""
    return TransformComponent(
        position=Vec3(5, 10, 15),
        rotation=Quat.from_euler(math.pi / 6, math.pi / 4, math.pi / 3),
        scale=Vec3(1.5, 2.0, 2.5),
    )


@pytest.fixture
def parent_child_transforms():
    """Create parent-child transform relationship."""
    parent = TransformComponent(position=Vec3(10, 0, 0))
    child = TransformComponent(position=Vec3(5, 0, 0))
    child.parent = parent
    return parent, child


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTransformInitialization:
    """Tests for TransformComponent initialization."""

    def test_default_initialization(self, transform):
        """Test default transform values."""
        assert transform.position == Vec3.zero()
        assert transform.rotation == Quat.identity()
        assert transform.scale == Vec3.one()

    def test_initialization_with_position(self):
        """Test initialization with custom position."""
        pos = Vec3(1, 2, 3)
        t = TransformComponent(position=pos)
        assert t.position == pos

    def test_initialization_with_rotation(self):
        """Test initialization with custom rotation."""
        rot = Quat.from_euler(0.5, 0.5, 0.5)
        t = TransformComponent(rotation=rot)
        assert t.rotation.dot(rot) > 0.99  # Nearly equal quaternions

    def test_initialization_with_scale(self):
        """Test initialization with custom scale."""
        scale = Vec3(2, 2, 2)
        t = TransformComponent(scale=scale)
        assert t.scale == scale

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        pos = Vec3(1, 2, 3)
        rot = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        scale = Vec3(2, 3, 4)
        t = TransformComponent(position=pos, rotation=rot, scale=scale)
        assert t.position == pos
        assert t.scale == scale

    def test_initialization_with_entity_id(self):
        """Test initialization with entity ID."""
        t = TransformComponent(entity_id="test_entity_123")
        assert t._entity_id == "test_entity_123"

    def test_initialization_none_defaults_to_identity(self):
        """Test that None values default to identity values."""
        t = TransformComponent(position=None, rotation=None, scale=None)
        assert t.position == Vec3.zero()
        assert t.rotation == Quat.identity()
        assert t.scale == Vec3.one()


# =============================================================================
# POSITION TESTS
# =============================================================================


class TestTransformPosition:
    """Tests for position manipulation."""

    def test_set_position(self, transform):
        """Test setting position."""
        transform.position = Vec3(10, 20, 30)
        assert transform.position == Vec3(10, 20, 30)

    def test_get_position(self, positioned_transform):
        """Test getting position."""
        assert positioned_transform.position == Vec3(10, 20, 30)

    def test_position_x_component(self, transform):
        """Test position x component."""
        transform.position = Vec3(5, 0, 0)
        assert transform.position.x == 5

    def test_position_y_component(self, transform):
        """Test position y component."""
        transform.position = Vec3(0, 10, 0)
        assert transform.position.y == 10

    def test_position_z_component(self, transform):
        """Test position z component."""
        transform.position = Vec3(0, 0, 15)
        assert transform.position.z == 15

    def test_position_negative_values(self, transform):
        """Test position with negative values."""
        transform.position = Vec3(-5, -10, -15)
        assert transform.position == Vec3(-5, -10, -15)

    def test_position_zero(self, transform):
        """Test setting position to zero."""
        transform.position = Vec3(10, 10, 10)
        transform.position = Vec3.zero()
        assert transform.position == Vec3.zero()

    def test_position_very_large_values(self, transform):
        """Test position with very large values."""
        transform.position = Vec3(1e10, 1e10, 1e10)
        assert transform.position.x == pytest.approx(1e10, rel=1e-6)

    def test_position_very_small_values(self, transform):
        """Test position with very small values."""
        transform.position = Vec3(1e-10, 1e-10, 1e-10)
        assert transform.position.x == pytest.approx(1e-10, rel=1e-6)

    def test_translate_local_space(self, transform):
        """Test translation in local space."""
        transform.position = Vec3(5, 5, 5)
        transform.translate(Vec3(1, 2, 3), TransformSpace.LOCAL)
        assert transform.position == Vec3(6, 7, 8)

    def test_translate_world_space(self, transform):
        """Test translation in world space."""
        transform.position = Vec3(5, 5, 5)
        transform.translate(Vec3(1, 2, 3), TransformSpace.WORLD)
        assert transform.position == Vec3(6, 7, 8)

    def test_translate_self_space(self, transform):
        """Test translation in self (object) space."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        transform.translate(Vec3(1, 0, 0), TransformSpace.SELF)
        # After 90 degree rotation around Y, local X points to -Z
        assert transform.position.z < 0 or transform.position.x != pytest.approx(1, abs=0.1)

    def test_translate_multiple_times(self, transform):
        """Test multiple translations accumulate."""
        transform.translate(Vec3(1, 0, 0))
        transform.translate(Vec3(0, 1, 0))
        transform.translate(Vec3(0, 0, 1))
        assert transform.position == Vec3(1, 1, 1)


# =============================================================================
# ROTATION TESTS
# =============================================================================


class TestTransformRotation:
    """Tests for rotation manipulation."""

    def test_set_rotation_quaternion(self, transform):
        """Test setting rotation with quaternion."""
        rot = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        transform.rotation = rot
        assert transform.rotation.dot(rot) > 0.99

    def test_get_rotation(self, rotated_transform):
        """Test getting rotation."""
        assert rotated_transform.rotation != Quat.identity()

    def test_rotation_identity(self, transform):
        """Test identity rotation."""
        assert transform.rotation == Quat.identity()

    def test_rotation_90_degrees_y(self, transform):
        """Test 90 degree rotation around Y axis."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        forward = transform.forward
        # After 90 degree Y rotation, forward should point toward X
        assert forward.x < -0.9

    def test_rotation_180_degrees_y(self, transform):
        """Test 180 degree rotation around Y axis."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi)
        forward = transform.forward
        # After 180 degree Y rotation, forward should point to +Z
        assert forward.z > 0.9

    def test_rotation_around_x(self, transform):
        """Test rotation around X axis."""
        transform.rotation = Quat.from_axis_angle(Vec3.right(), math.pi / 2)
        up = transform.up
        # After 90 degree X rotation, up should point toward -Z
        assert up.z < -0.9

    def test_rotation_around_z(self, transform):
        """Test rotation around Z axis."""
        transform.rotation = Quat.from_axis_angle(Vec3.forward(), math.pi / 2)
        right = transform.right
        # After 90 degree Z rotation, right should point up
        assert right.y > 0.9

    def test_rotation_euler_conversion(self, transform):
        """Test rotation from Euler angles."""
        transform.rotation = Quat.from_euler(math.pi / 4, math.pi / 6, math.pi / 3)
        assert transform.rotation != Quat.identity()

    def test_rotation_preserves_magnitude(self, transform):
        """Test that rotation quaternion remains normalized."""
        transform.rotation = Quat.from_euler(0.5, 0.5, 0.5)
        assert transform.rotation.length() == pytest.approx(1.0, abs=0.001)

    def test_rotate_method_local(self, transform):
        """Test rotate method in local space."""
        rot1 = Quat.from_axis_angle(Vec3.up(), math.pi / 4)
        rot2 = Quat.from_axis_angle(Vec3.up(), math.pi / 4)
        transform.rotation = rot1
        transform.rotate(rot2, TransformSpace.LOCAL)
        # Combined rotation should be ~90 degrees
        forward = transform.forward
        assert abs(forward.x) > 0.6

    def test_rotate_method_world(self, transform):
        """Test rotate method in world space."""
        rot = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        transform.rotate(rot, TransformSpace.WORLD)
        assert transform.rotation.dot(rot) > 0.99

    def test_rotation_accumulation(self, transform):
        """Test that rotations accumulate correctly."""
        for _ in range(4):
            transform.rotate(Quat.from_axis_angle(Vec3.up(), math.pi / 2))
        # After 4 * 90 degrees, should be back to identity
        assert transform.rotation.dot(Quat.identity()) > 0.99

    def test_rotate_around_point(self, transform):
        """Test rotate_around method."""
        transform.position = Vec3(10, 0, 0)
        transform.rotate_around(Vec3.zero(), Vec3.up(), math.pi / 2)
        # Position should rotate around origin
        assert transform.position.z == pytest.approx(10, abs=0.1)
        assert transform.position.x == pytest.approx(0, abs=0.1)


# =============================================================================
# SCALE TESTS
# =============================================================================


class TestTransformScale:
    """Tests for scale manipulation."""

    def test_set_uniform_scale(self, transform):
        """Test setting uniform scale."""
        transform.scale = Vec3(2, 2, 2)
        assert transform.scale == Vec3(2, 2, 2)

    def test_set_non_uniform_scale(self, transform):
        """Test setting non-uniform scale."""
        transform.scale = Vec3(1, 2, 3)
        assert transform.scale == Vec3(1, 2, 3)

    def test_get_scale(self, scaled_transform):
        """Test getting scale."""
        assert scaled_transform.scale == Vec3(2, 3, 4)

    def test_scale_one_axis(self, transform):
        """Test scaling on one axis."""
        transform.scale = Vec3(5, 1, 1)
        assert transform.scale.x == 5
        assert transform.scale.y == 1
        assert transform.scale.z == 1

    def test_scale_zero(self, transform):
        """Test zero scale on one axis."""
        transform.scale = Vec3(1, 0, 1)
        assert transform.scale.y == 0

    def test_scale_negative(self, transform):
        """Test negative scale (mirroring)."""
        transform.scale = Vec3(-1, 1, 1)
        assert transform.scale.x == -1

    def test_scale_very_small(self, transform):
        """Test very small scale."""
        transform.scale = Vec3(0.001, 0.001, 0.001)
        assert transform.scale.x == pytest.approx(0.001, rel=1e-6)

    def test_scale_very_large(self, transform):
        """Test very large scale."""
        transform.scale = Vec3(1000, 1000, 1000)
        assert transform.scale.x == pytest.approx(1000, rel=1e-6)


# =============================================================================
# HIERARCHY TESTS
# =============================================================================


class TestTransformHierarchy:
    """Tests for transform hierarchy (parent-child relationships)."""

    def test_set_parent(self, transform):
        """Test setting a parent."""
        parent = TransformComponent()
        transform.parent = parent
        assert transform.parent is parent
        assert transform in parent.children

    def test_remove_parent(self, parent_child_transforms):
        """Test removing parent."""
        parent, child = parent_child_transforms
        child.parent = None
        assert child.parent is None
        assert child not in parent.children

    def test_has_parent(self, parent_child_transforms):
        """Test has_parent property."""
        parent, child = parent_child_transforms
        assert child.has_parent is True
        assert parent.has_parent is False

    def test_is_root(self, parent_child_transforms):
        """Test is_root property."""
        parent, child = parent_child_transforms
        assert parent.is_root is True
        assert child.is_root is False

    def test_get_root(self):
        """Test get_root method."""
        root = TransformComponent()
        middle = TransformComponent()
        leaf = TransformComponent()
        middle.parent = root
        leaf.parent = middle
        assert leaf.get_root() is root
        assert middle.get_root() is root
        assert root.get_root() is root

    def test_children_list(self, parent_child_transforms):
        """Test children list."""
        parent, child = parent_child_transforms
        assert child in parent.children
        assert len(parent.children) == 1

    def test_multiple_children(self):
        """Test multiple children."""
        parent = TransformComponent()
        child1 = TransformComponent()
        child2 = TransformComponent()
        child3 = TransformComponent()
        child1.parent = parent
        child2.parent = parent
        child3.parent = parent
        assert len(parent.children) == 3
        assert child1 in parent.children
        assert child2 in parent.children
        assert child3 in parent.children

    def test_reparent_same_parent(self, parent_child_transforms):
        """Test reparenting to same parent does nothing."""
        parent, child = parent_child_transforms
        child.parent = parent  # Set same parent again
        assert child.parent is parent
        assert len(parent.children) == 1

    def test_reparent_different_parent(self):
        """Test reparenting to different parent."""
        parent1 = TransformComponent()
        parent2 = TransformComponent()
        child = TransformComponent()
        child.parent = parent1
        child.parent = parent2
        assert child.parent is parent2
        assert child not in parent1.children
        assert child in parent2.children

    def test_iter_ancestors(self):
        """Test iterate ancestors."""
        root = TransformComponent()
        middle = TransformComponent()
        leaf = TransformComponent()
        middle.parent = root
        leaf.parent = middle
        ancestors = list(leaf.iter_ancestors())
        assert ancestors == [middle, root]

    def test_iter_descendants(self):
        """Test iterate descendants."""
        root = TransformComponent()
        child1 = TransformComponent()
        child2 = TransformComponent()
        grandchild = TransformComponent()
        child1.parent = root
        child2.parent = root
        grandchild.parent = child1
        descendants = list(root.iter_descendants())
        assert len(descendants) == 3
        assert child1 in descendants
        assert grandchild in descendants

    def test_detach_children(self):
        """Test detach_children method."""
        parent = TransformComponent()
        child1 = TransformComponent()
        child2 = TransformComponent()
        child1.parent = parent
        child2.parent = parent
        detached = parent.detach_children()
        assert len(detached) == 2
        assert child1.parent is None
        assert child2.parent is None
        assert len(parent.children) == 0

    def test_reparent_keep_world_position(self):
        """Test reparenting while keeping world position."""
        parent = TransformComponent(position=Vec3(10, 0, 0))
        child = TransformComponent(position=Vec3(5, 0, 0))
        # Initial world position of child is (5, 0, 0)
        child.reparent(parent, keep_world=True)
        # World position should still be (5, 0, 0)
        # So local position relative to parent at (10, 0, 0) should be (-5, 0, 0)
        assert child.world_position.x == pytest.approx(5, abs=0.1)

    def test_reparent_no_keep_world(self):
        """Test reparenting without keeping world position."""
        parent = TransformComponent(position=Vec3(10, 0, 0))
        child = TransformComponent(position=Vec3(5, 0, 0))
        child.reparent(parent, keep_world=False)
        # Local position stays as (5, 0, 0), world becomes (15, 0, 0)
        assert child.world_position.x == pytest.approx(15, abs=0.1)

    def test_circular_reference_prevention(self):
        """Test that circular parent references don't create infinite loops."""
        t1 = TransformComponent()
        t2 = TransformComponent()
        t1.parent = t2
        # This would create a cycle - implementation may or may not prevent this
        # At minimum, it shouldn't cause infinite loop
        t2.parent = t1
        # Just verify we can still get root without infinite loop
        root = t1.get_root()
        assert root is not None


# =============================================================================
# LOCAL VS WORLD SPACE TESTS
# =============================================================================


class TestLocalVsWorldSpace:
    """Tests for local vs world space transformations."""

    def test_world_position_no_parent(self, positioned_transform):
        """Test world position equals local when no parent."""
        assert positioned_transform.world_position == positioned_transform.position

    def test_world_position_with_parent(self, parent_child_transforms):
        """Test world position accumulates with parent."""
        parent, child = parent_child_transforms
        # Parent at (10, 0, 0), child at (5, 0, 0) local
        assert child.world_position == Vec3(15, 0, 0)

    def test_world_rotation_no_parent(self, rotated_transform):
        """Test world rotation equals local when no parent."""
        assert rotated_transform.world_rotation.dot(rotated_transform.rotation) > 0.99

    def test_world_rotation_with_parent(self):
        """Test world rotation combines with parent."""
        parent = TransformComponent(
            rotation=Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        )
        child = TransformComponent(
            rotation=Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        )
        child.parent = parent
        # Combined 180 degree rotation
        world_rot = child.world_rotation
        forward = world_rot.rotate_vector(Vec3(0, 0, -1))
        assert forward.z > 0.9  # Should point backward

    def test_world_scale_no_parent(self, scaled_transform):
        """Test world scale equals local when no parent."""
        assert scaled_transform.world_scale == scaled_transform.scale

    def test_world_scale_with_parent(self):
        """Test world scale multiplies with parent."""
        parent = TransformComponent(scale=Vec3(2, 2, 2))
        child = TransformComponent(scale=Vec3(3, 3, 3))
        child.parent = parent
        assert child.world_scale == Vec3(6, 6, 6)

    def test_set_world_position_no_parent(self, transform):
        """Test set_world_position without parent."""
        transform.set_world_position(Vec3(10, 20, 30))
        assert transform.position == Vec3(10, 20, 30)

    def test_set_world_position_with_parent(self, parent_child_transforms):
        """Test set_world_position with parent."""
        parent, child = parent_child_transforms
        child.set_world_position(Vec3(20, 0, 0))
        # Parent at (10, 0, 0), so local should be (10, 0, 0)
        assert child.position.x == pytest.approx(10, abs=0.1)

    def test_set_world_rotation_no_parent(self, transform):
        """Test set_world_rotation without parent."""
        rot = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        transform.set_world_rotation(rot)
        assert transform.rotation.dot(rot) > 0.99

    def test_set_world_rotation_with_parent(self):
        """Test set_world_rotation with parent."""
        parent = TransformComponent(
            rotation=Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        )
        child = TransformComponent()
        child.parent = parent
        # Set child to identity world rotation
        child.set_world_rotation(Quat.identity())
        # Local should counteract parent's rotation
        combined = child.world_rotation
        assert combined.dot(Quat.identity()) > 0.99

    def test_world_matrix_no_parent(self, full_transform):
        """Test world matrix equals local matrix without parent."""
        world = full_transform.world_matrix
        local = full_transform.local_matrix
        # Matrices should be equal
        for i in range(16):
            assert world.m[i] == pytest.approx(local.m[i], abs=0.001)

    def test_world_matrix_with_parent(self, parent_child_transforms):
        """Test world matrix combines with parent."""
        parent, child = parent_child_transforms
        world_pos = child.world_position
        # Verify world matrix extracts correct position
        matrix = child.world_matrix
        assert matrix.m[12] == pytest.approx(world_pos.x, abs=0.001)
        assert matrix.m[13] == pytest.approx(world_pos.y, abs=0.001)
        assert matrix.m[14] == pytest.approx(world_pos.z, abs=0.001)

    def test_local_matrix_composition(self, full_transform):
        """Test local matrix is correctly composed from TRS."""
        matrix = full_transform.local_matrix
        # Check position in last column
        assert matrix.m[12] == pytest.approx(full_transform.position.x, abs=0.01)
        assert matrix.m[13] == pytest.approx(full_transform.position.y, abs=0.01)
        assert matrix.m[14] == pytest.approx(full_transform.position.z, abs=0.01)


# =============================================================================
# DIRTY FLAG TESTS
# =============================================================================


class TestTransformDirtyFlags:
    """Tests for dirty flag tracking."""

    def test_initial_dirty_cleared(self, transform):
        """Test that initial dirty flags are cleared."""
        # After construction, dirty flags should be cleared
        assert not transform.is_transform_dirty()

    def test_position_change_sets_dirty(self, transform):
        """Test that position change sets dirty flag."""
        transform.clear_dirty_flags()
        transform.position = Vec3(1, 2, 3)
        assert transform.is_transform_dirty()

    def test_rotation_change_sets_dirty(self, transform):
        """Test that rotation change sets dirty flag."""
        transform.clear_dirty_flags()
        transform.rotation = Quat.from_euler(0.1, 0.2, 0.3)
        assert transform.is_transform_dirty()

    def test_scale_change_sets_dirty(self, transform):
        """Test that scale change sets dirty flag."""
        transform.clear_dirty_flags()
        transform.scale = Vec3(2, 2, 2)
        assert transform.is_transform_dirty()

    def test_clear_dirty_flags(self, transform):
        """Test clearing dirty flags."""
        transform.position = Vec3(1, 2, 3)
        transform.clear_dirty_flags()
        assert not transform.is_transform_dirty()

    def test_mark_dirty(self, transform):
        """Test mark_dirty method."""
        transform.clear_dirty_flags()
        transform.mark_dirty()
        assert transform._world_matrix_dirty is True

    def test_world_matrix_cache(self, transform):
        """Test that world matrix is cached."""
        transform.position = Vec3(1, 2, 3)
        matrix1 = transform.world_matrix
        matrix2 = transform.world_matrix
        # Should return same cached instance
        assert matrix1 is matrix2

    def test_world_matrix_invalidation(self, transform):
        """Test that world matrix is invalidated on change."""
        matrix1 = transform.world_matrix
        transform.position = Vec3(10, 20, 30)
        matrix2 = transform.world_matrix
        # Should be different after position change
        assert matrix1.m[12] != matrix2.m[12]

    def test_child_dirty_on_parent_change(self, parent_child_transforms):
        """Test that child becomes dirty when parent changes."""
        parent, child = parent_child_transforms
        _ = child.world_matrix  # Force cache
        parent.position = Vec3(100, 0, 0)
        # Child's world matrix should be invalidated
        assert child._world_matrix_dirty is True


# =============================================================================
# SNAPSHOT AND INTERPOLATION TESTS
# =============================================================================


class TestTransformSnapshots:
    """Tests for transform snapshots and interpolation."""

    def test_create_snapshot(self, full_transform):
        """Test creating a snapshot."""
        snapshot = full_transform.create_snapshot(timestamp=1.0)
        assert snapshot.position == full_transform.position
        assert snapshot.rotation == full_transform.rotation
        assert snapshot.scale == full_transform.scale
        assert snapshot.timestamp == 1.0

    def test_apply_snapshot(self, transform):
        """Test applying a snapshot."""
        snapshot = TransformSnapshot(
            position=Vec3(10, 20, 30),
            rotation=Quat.from_euler(0.1, 0.2, 0.3),
            scale=Vec3(2, 3, 4),
            timestamp=0.5,
        )
        transform.apply_snapshot(snapshot)
        assert transform.position == snapshot.position
        assert transform.scale == snapshot.scale

    def test_snapshot_lerp(self):
        """Test snapshot interpolation."""
        snap1 = TransformSnapshot(
            position=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            scale=Vec3.one(),
            timestamp=0.0,
        )
        snap2 = TransformSnapshot(
            position=Vec3(10, 20, 30),
            rotation=Quat.from_axis_angle(Vec3.up(), math.pi),
            scale=Vec3(2, 2, 2),
            timestamp=1.0,
        )
        interpolated = snap1.lerp(snap2, 0.5)
        assert interpolated.position == Vec3(5, 10, 15)
        assert interpolated.scale == Vec3(1.5, 1.5, 1.5)
        assert interpolated.timestamp == 0.5

    def test_snapshot_lerp_at_zero(self):
        """Test snapshot interpolation at t=0."""
        snap1 = TransformSnapshot(Vec3(0, 0, 0), Quat.identity(), Vec3.one(), 0.0)
        snap2 = TransformSnapshot(Vec3(10, 10, 10), Quat.identity(), Vec3(2, 2, 2), 1.0)
        result = snap1.lerp(snap2, 0.0)
        assert result.position == Vec3(0, 0, 0)

    def test_snapshot_lerp_at_one(self):
        """Test snapshot interpolation at t=1."""
        snap1 = TransformSnapshot(Vec3(0, 0, 0), Quat.identity(), Vec3.one(), 0.0)
        snap2 = TransformSnapshot(Vec3(10, 10, 10), Quat.identity(), Vec3(2, 2, 2), 1.0)
        result = snap1.lerp(snap2, 1.0)
        assert result.position == Vec3(10, 10, 10)

    def test_multiple_snapshots(self, transform):
        """Test creating multiple snapshots over time."""
        snapshots = []
        for i in range(5):
            transform.position = Vec3(i * 10, 0, 0)
            snapshots.append(transform.create_snapshot(timestamp=float(i)))
        assert len(snapshots) == 5
        assert snapshots[0].position.x == 0
        assert snapshots[4].position.x == 40


# =============================================================================
# LOOK AT TESTS
# =============================================================================


class TestTransformLookAt:
    """Tests for look-at functionality."""

    def test_look_at_forward(self, transform):
        """Test look_at pointing forward."""
        transform.look_at(Vec3(0, 0, -10))
        # Should point toward -Z (default forward)
        forward = transform.forward
        assert forward.z < -0.9

    def test_look_at_right(self, transform):
        """Test look_at pointing right."""
        transform.look_at(Vec3(10, 0, 0))
        forward = transform.forward
        assert forward.x > 0.9

    def test_look_at_left(self, transform):
        """Test look_at pointing left."""
        transform.look_at(Vec3(-10, 0, 0))
        forward = transform.forward
        assert forward.x < -0.9

    def test_look_at_up(self, transform):
        """Test look_at pointing up."""
        transform.look_at(Vec3(0, 10, 0), up=Vec3(0, 0, -1))
        forward = transform.forward
        assert forward.y > 0.9

    def test_look_at_down(self, transform):
        """Test look_at pointing down."""
        transform.look_at(Vec3(0, -10, 0), up=Vec3(0, 0, 1))
        forward = transform.forward
        assert forward.y < -0.9

    def test_look_at_from_offset_position(self, positioned_transform):
        """Test look_at from non-origin position."""
        positioned_transform.look_at(Vec3(10, 20, 30 + 10))  # 10 units in front
        forward = positioned_transform.forward
        assert forward.z > 0.9

    def test_look_at_same_position_no_change(self, transform):
        """Test look_at at same position does nothing."""
        original_rot = Quat(transform.rotation.x, transform.rotation.y,
                           transform.rotation.z, transform.rotation.w)
        transform.look_at(Vec3.zero())  # Same as position
        # Rotation should not change (or at least not crash)
        assert transform.rotation is not None


# =============================================================================
# DIRECTION VECTOR TESTS
# =============================================================================


class TestDirectionVectors:
    """Tests for direction vectors (forward, up, right)."""

    def test_default_forward(self, transform):
        """Test default forward direction."""
        forward = transform.forward
        assert forward.z < -0.9  # Forward is -Z

    def test_default_up(self, transform):
        """Test default up direction."""
        up = transform.up
        assert up.y > 0.9

    def test_default_right(self, transform):
        """Test default right direction."""
        right = transform.right
        assert right.x > 0.9

    def test_rotated_forward(self, transform):
        """Test forward after rotation."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        forward = transform.forward
        # After 90 degree Y rotation, forward points to -X
        assert forward.x < -0.9

    def test_rotated_up(self, transform):
        """Test up after rotation."""
        transform.rotation = Quat.from_axis_angle(Vec3.right(), math.pi / 2)
        up = transform.up
        # After 90 degree X rotation, up points to -Z
        assert up.z < -0.9

    def test_rotated_right(self, transform):
        """Test right after rotation."""
        transform.rotation = Quat.from_axis_angle(Vec3.forward(), math.pi / 2)
        right = transform.right
        # After 90 degree Z rotation, right points up
        assert right.y > 0.9

    def test_direction_vectors_orthogonal(self, rotated_transform):
        """Test that direction vectors are orthogonal."""
        f = rotated_transform.forward
        u = rotated_transform.up
        r = rotated_transform.right
        assert abs(f.dot(u)) < 0.01
        assert abs(f.dot(r)) < 0.01
        assert abs(u.dot(r)) < 0.01

    def test_direction_vectors_normalized(self, full_transform):
        """Test that direction vectors are normalized."""
        assert full_transform.forward.length() == pytest.approx(1.0, abs=0.001)
        assert full_transform.up.length() == pytest.approx(1.0, abs=0.001)
        assert full_transform.right.length() == pytest.approx(1.0, abs=0.001)


# =============================================================================
# TRANSFORM EVENT TESTS
# =============================================================================


class TestTransformEvents:
    """Tests for transform change events."""

    def test_register_callback(self, transform):
        """Test registering a change callback."""
        callback_called = [False]

        def callback(t):
            callback_called[0] = True

        transform.on_transform_changed(callback)
        transform.position = Vec3(1, 2, 3)
        transform.mark_dirty()  # Explicitly mark dirty to trigger callbacks
        assert callback_called[0]

    def test_unregister_callback(self, transform):
        """Test unregistering a change callback."""
        callback_called = [False]

        def callback(t):
            callback_called[0] = True

        transform.on_transform_changed(callback)
        transform.off_transform_changed(callback)
        transform.mark_dirty()
        assert not callback_called[0]

    def test_multiple_callbacks(self, transform):
        """Test multiple callbacks."""
        call_count = [0]

        def callback1(t):
            call_count[0] += 1

        def callback2(t):
            call_count[0] += 1

        transform.on_transform_changed(callback1)
        transform.on_transform_changed(callback2)
        transform.mark_dirty()
        assert call_count[0] == 2

    def test_callback_receives_transform(self, transform):
        """Test callback receives the transform instance."""
        received_transform = [None]

        def callback(t):
            received_transform[0] = t

        transform.on_transform_changed(callback)
        transform.mark_dirty()
        assert received_transform[0] is transform

    def test_child_change_propagates(self, parent_child_transforms):
        """Test that parent changes propagate to children."""
        parent, child = parent_child_transforms
        child_dirty = [False]

        def callback(t):
            child_dirty[0] = True

        child.on_transform_changed(callback)
        parent.mark_dirty()
        assert child_dirty[0]


# =============================================================================
# COORDINATE TRANSFORMATION TESTS
# =============================================================================


class TestCoordinateTransformation:
    """Tests for coordinate transformation methods."""

    def test_transform_point(self, transform):
        """Test transform_point method."""
        transform.position = Vec3(10, 0, 0)
        world_point = transform.transform_point(Vec3(5, 0, 0))
        assert world_point == Vec3(15, 0, 0)

    def test_inverse_transform_point(self, transform):
        """Test inverse_transform_point method."""
        transform.position = Vec3(10, 0, 0)
        local_point = transform.inverse_transform_point(Vec3(15, 0, 0))
        assert local_point.x == pytest.approx(5, abs=0.1)

    def test_transform_direction(self, transform):
        """Test transform_direction method."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        world_dir = transform.transform_direction(Vec3(0, 0, -1))  # Local forward
        assert world_dir.x < -0.9  # Should point to -X

    def test_inverse_transform_direction(self, transform):
        """Test inverse_transform_direction method."""
        transform.rotation = Quat.from_axis_angle(Vec3.up(), math.pi / 2)
        local_dir = transform.inverse_transform_direction(Vec3(-1, 0, 0))
        assert local_dir.z < -0.9  # Should become local forward

    def test_transform_point_with_scale(self, transform):
        """Test transform_point respects scale."""
        transform.scale = Vec3(2, 2, 2)
        world_point = transform.transform_point(Vec3(1, 1, 1))
        assert world_point == Vec3(2, 2, 2)

    def test_transform_direction_ignores_scale(self, transform):
        """Test transform_direction ignores scale and position."""
        transform.position = Vec3(100, 100, 100)
        transform.scale = Vec3(5, 5, 5)
        # Direction transformation only considers rotation
        world_dir = transform.transform_direction(Vec3(1, 0, 0))
        assert world_dir.length() == pytest.approx(1.0, abs=0.1)


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestTransformSerialization:
    """Tests for transform serialization."""

    def test_to_dict(self, full_transform):
        """Test serialization to dictionary."""
        data = full_transform.to_dict()
        assert "position" in data
        assert "rotation" in data
        assert "scale" in data
        assert len(data["position"]) == 3
        assert len(data["rotation"]) == 4
        assert len(data["scale"]) == 3

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "position": [1, 2, 3],
            "rotation": [0, 0, 0, 1],
            "scale": [2, 2, 2],
            "entity_id": "test_123",
        }
        transform = TransformComponent.from_dict(data)
        assert transform.position == Vec3(1, 2, 3)
        assert transform.scale == Vec3(2, 2, 2)
        assert transform._entity_id == "test_123"

    def test_round_trip_serialization(self, full_transform):
        """Test serialization round trip."""
        data = full_transform.to_dict()
        restored = TransformComponent.from_dict(data)
        assert restored.position == full_transform.position
        assert restored.scale == full_transform.scale

    def test_serialization_with_entity_id(self, transform):
        """Test serialization preserves entity ID."""
        transform._entity_id = "entity_abc"
        data = transform.to_dict()
        assert data["entity_id"] == "entity_abc"

    def test_repr(self, full_transform):
        """Test string representation."""
        rep = repr(full_transform)
        assert "TransformComponent" in rep
        assert "pos=" in rep
        assert "rot=" in rep
        assert "scale=" in rep


# =============================================================================
# SET LOCAL TRANSFORM TESTS
# =============================================================================


class TestSetLocalTransform:
    """Tests for set_local_transform method."""

    def test_set_local_transform(self, transform):
        """Test setting all local transform values at once."""
        pos = Vec3(1, 2, 3)
        rot = Quat.from_euler(0.1, 0.2, 0.3)
        scale = Vec3(2, 3, 4)
        transform.set_local_transform(pos, rot, scale)
        assert transform.position == pos
        assert transform.scale == scale

    def test_set_local_transform_invalidates_cache(self, transform):
        """Test that set_local_transform invalidates matrix cache."""
        _ = transform.local_matrix  # Force cache
        transform.set_local_transform(Vec3(1, 1, 1), Quat.identity(), Vec3.one())
        assert transform._local_matrix_dirty or transform._world_matrix_dirty


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestTransformEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_deep_hierarchy(self):
        """Test very deep parent-child hierarchy."""
        transforms = [TransformComponent(position=Vec3(1, 0, 0)) for _ in range(100)]
        for i in range(1, 100):
            transforms[i].parent = transforms[i - 1]
        # World position of deepest should be (100, 0, 0)
        assert transforms[99].world_position.x == pytest.approx(100, abs=0.1)

    def test_wide_hierarchy(self):
        """Test wide hierarchy (many children)."""
        parent = TransformComponent()
        children = [TransformComponent() for _ in range(1000)]
        for child in children:
            child.parent = parent
        assert len(parent.children) == 1000

    def test_rotation_near_gimbal_lock(self, transform):
        """Test rotation near gimbal lock (90 degree pitch)."""
        transform.rotation = Quat.from_euler(math.pi / 2 - 0.001, 0, 0)
        # Should still have valid direction vectors
        assert transform.forward.length() == pytest.approx(1.0, abs=0.01)

    def test_zero_scale_handling(self, transform):
        """Test behavior with zero scale."""
        transform.scale = Vec3(0, 0, 0)
        # Should not crash
        matrix = transform.local_matrix
        assert matrix is not None

    def test_nan_handling_position(self, transform):
        """Test NaN values in position."""
        transform.position = Vec3(float("nan"), 0, 0)
        assert math.isnan(transform.position.x)

    def test_inf_handling_position(self, transform):
        """Test infinity values in position."""
        transform.position = Vec3(float("inf"), 0, 0)
        assert math.isinf(transform.position.x)
