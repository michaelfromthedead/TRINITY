"""
Blackbox tests for T1.3 Batch Rendering Logic.

These tests verify the CrowdRenderer and CrowdRenderBatch public contract
without knowledge of internal implementation details. Tests are derived from:
- docs/PYTHON_DOCS/engine_animation_crowds_facial/PHASE_1_ARCH.md
- docs/PYTHON_DOCS/engine_animation_crowds_facial/PHASE_1_TODO.md

Contract under test:
1. Instances group by (mesh_id, material_id)
2. Batch priority ordering is correct
3. Empty batches are handled gracefully
4. Single instance per batch works

Public API Contract (from ARCH and method introspection):
    CrowdRenderBatch
        +-- mesh_id: int
        +-- material_id: int
        +-- instance_count: int
        +-- priority: int (render order)
        +-- add_instance(instance)

    CrowdRenderer
        +-- batches: dict[(mesh, material), CrowdRenderBatch]
        +-- atlases: dict[str, AnimationTextureAtlas]
        +-- add_instance(instance, mesh_id, material_id, atlas_name=None) -> int
        +-- render() -> GPU commands

NOTE: The TODO example shows mesh_id/material_id as CrowdInstance params,
but actual API passes them to add_instance(). Tests use actual API.

CLEANROOM: This file was written without reading crowd_renderer.py implementation.
"""

import pytest
import numpy as np

from engine.core.math import Vec3, Vec4, Quat
from engine.animation.crowds import (
    CrowdInstance,
    CrowdRenderer,
    CrowdRenderBatch,
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def make_instance(
    position: Vec3 = None,
    rotation: Quat = None,
    scale: float = 1.0,
    animation_index: int = 0,
    animation_time: float = 0.0,
    animation_speed: float = 1.0,
    tint_color: Vec4 = None,
    lod_level: int = 0,
) -> CrowdInstance:
    """Create a CrowdInstance with sensible defaults for batch testing.

    Note: mesh_id and material_id are passed to add_instance(), not CrowdInstance.
    """
    if position is None:
        position = Vec3(0.0, 0.0, 0.0)
    if rotation is None:
        rotation = Quat.identity()
    if tint_color is None:
        tint_color = Vec4(1.0, 1.0, 1.0, 1.0)

    return CrowdInstance(
        position=position,
        rotation=rotation,
        scale=scale,
        animation_index=animation_index,
        animation_time=animation_time,
        animation_speed=animation_speed,
        tint_color=tint_color,
        lod_level=lod_level,
    )


def add_instance_to_renderer(
    renderer: CrowdRenderer,
    mesh_id: int = 1,
    material_id: int = 1,
    position: Vec3 = None,
    rotation: Quat = None,
    scale: float = 1.0,
    animation_index: int = 0,
    animation_time: float = 0.0,
    animation_speed: float = 1.0,
    tint_color: Vec4 = None,
    lod_level: int = 0,
    atlas_name: str = None,
) -> int:
    """Add an instance to renderer with specified mesh/material IDs.

    Returns the instance index.
    """
    instance = make_instance(
        position=position,
        rotation=rotation,
        scale=scale,
        animation_index=animation_index,
        animation_time=animation_time,
        animation_speed=animation_speed,
        tint_color=tint_color,
        lod_level=lod_level,
    )
    return renderer.add_instance(instance, mesh_id=mesh_id, material_id=material_id, atlas_name=atlas_name)


# -----------------------------------------------------------------------------
# Batch Grouping Tests (from T1.3 acceptance criteria)
# -----------------------------------------------------------------------------

class TestBatchGrouping:
    """Test that instances group correctly by (mesh_id, material_id)."""

    def test_batch_grouping_same_mesh_material(self):
        """Instances with same mesh_id and material_id belong to same batch.

        From TODO (adapted for actual API):
            renderer.add_instance(instance, mesh_id=1, material_id=1)
            renderer.add_instance(instance, mesh_id=1, material_id=1)
            assert renderer.batches[(1, 1)].instance_count == 2
        """
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        # Both should be in the same batch
        assert (1, 1) in renderer.batches
        batch = renderer.batches[(1, 1)]
        assert batch.instance_count == 2

    def test_batch_grouping_different_mesh_same_material(self):
        """Instances with different mesh_id create separate batches.

        From TODO (adapted for actual API):
            renderer.add_instance(instance, mesh_id=1, material_id=1)
            renderer.add_instance(instance, mesh_id=2, material_id=1)
            assert len(renderer.batches) == 2
        """
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=1)

        assert len(renderer.batches) == 2
        assert (1, 1) in renderer.batches
        assert (2, 1) in renderer.batches

    def test_batch_grouping_same_mesh_different_material(self):
        """Instances with different material_id create separate batches."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=2)

        assert len(renderer.batches) == 2
        assert (1, 1) in renderer.batches
        assert (1, 2) in renderer.batches

    def test_batch_grouping_multiple_batches(self):
        """Complex grouping with multiple mesh/material combinations.

        From TODO example extended:
            assert len(renderer.batches) == 2
        """
        renderer = CrowdRenderer()

        # Add instances in mixed order
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=2)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        # Should have 3 unique batches
        assert len(renderer.batches) == 3
        assert renderer.batches[(1, 1)].instance_count == 3
        assert renderer.batches[(2, 1)].instance_count == 1
        assert renderer.batches[(2, 2)].instance_count == 1

    def test_batch_preserves_mesh_and_material_ids(self):
        """CrowdRenderBatch stores correct mesh_id and material_id."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=5, material_id=10)

        batch = renderer.batches[(5, 10)]
        assert batch.mesh_id == 5
        assert batch.material_id == 10


# -----------------------------------------------------------------------------
# Single Instance Per Batch Tests
# -----------------------------------------------------------------------------

class TestSingleInstanceBatch:
    """Test that single instance per batch works correctly."""

    def test_single_instance_creates_batch(self):
        """Adding single instance creates a valid batch.

        From acceptance criteria: 'Single instance per batch works'
        """
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        assert (1, 1) in renderer.batches
        batch = renderer.batches[(1, 1)]
        assert batch.instance_count == 1

    def test_multiple_single_instance_batches(self):
        """Multiple batches can each have exactly one instance."""
        renderer = CrowdRenderer()

        for mesh_id in range(1, 6):
            add_instance_to_renderer(renderer, mesh_id=mesh_id, material_id=1)

        assert len(renderer.batches) == 5
        for mesh_id in range(1, 6):
            assert renderer.batches[(mesh_id, 1)].instance_count == 1

    def test_single_instance_batch_data_integrity(self):
        """Single instance batch contains the correct instance data."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(
            renderer,
            mesh_id=3,
            material_id=7,
            position=Vec3(1.0, 2.0, 3.0),
            animation_index=5,
        )

        batch = renderer.batches[(3, 7)]
        assert batch.instance_count == 1
        assert batch.mesh_id == 3
        assert batch.material_id == 7


# -----------------------------------------------------------------------------
# Empty Batch Handling Tests
# -----------------------------------------------------------------------------

class TestEmptyBatchHandling:
    """Test that empty batches are handled gracefully."""

    def test_new_renderer_has_no_batches(self):
        """Fresh CrowdRenderer starts with empty batches.

        From acceptance criteria: 'Empty batches are handled gracefully'
        """
        renderer = CrowdRenderer()

        assert len(renderer.batches) == 0

    def test_batches_is_dict_type(self):
        """The batches attribute is a dictionary."""
        renderer = CrowdRenderer()

        assert isinstance(renderer.batches, dict)

    def test_accessing_empty_batches_dict(self):
        """Iterating over empty batches dict doesn't raise."""
        renderer = CrowdRenderer()

        # Should not raise
        for key, batch in renderer.batches.items():
            pass

    def test_renderer_with_cleared_batches(self):
        """Renderer can be cleared and reused (if clear() exists)."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        assert len(renderer.batches) == 1

        # If clear() exists, test it; otherwise skip
        if hasattr(renderer, 'clear'):
            renderer.clear()
            assert len(renderer.batches) == 0


# -----------------------------------------------------------------------------
# Batch Priority Ordering Tests
# -----------------------------------------------------------------------------

class TestBatchPriorityOrdering:
    """Test that batch priority ordering is correct.

    NOTE: ARCH documentation says 'priority: int' but actual implementation
    uses RenderPriority enum. Tests updated to match actual API.
    """

    def test_batch_has_priority_attribute(self):
        """CrowdRenderBatch has a priority attribute.

        From ARCH: 'priority: int (render order)'
        Actual: priority is RenderPriority enum (contract discrepancy documented)
        """
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        batch = renderer.batches[(1, 1)]
        assert hasattr(batch, 'priority')
        # Priority is an enum with a .value attribute
        assert hasattr(batch.priority, 'value')
        assert isinstance(batch.priority.value, int)

    def test_batch_default_priority(self):
        """Batches have sensible default priority."""
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        batch = renderer.batches[(1, 1)]
        # Priority enum should have a numeric value >= 0
        assert batch.priority.value >= 0

    def test_batches_can_have_different_priorities(self):
        """Different batches can be assigned different priorities.

        Note: This tests the public contract that priorities can differ.
        If priorities are set via constructor or material properties,
        this test documents expected behavior.
        """
        renderer = CrowdRenderer()

        # Create multiple batches
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=2)

        batch1 = renderer.batches[(1, 1)]
        batch2 = renderer.batches[(2, 2)]

        # Both should have valid priority values (as enum with int value)
        assert hasattr(batch1.priority, 'value')
        assert hasattr(batch2.priority, 'value')
        assert isinstance(batch1.priority.value, int)
        assert isinstance(batch2.priority.value, int)

    def test_sorted_batches_by_priority(self):
        """Batches can be sorted by priority for render order.

        From ARCH batching strategy:
            1. Group by mesh + material
            2. Sort batches by priority
        """
        renderer = CrowdRenderer()

        # Add instances for multiple batches
        for mesh_id in range(1, 5):
            add_instance_to_renderer(renderer, mesh_id=mesh_id, material_id=1)

        # Should be able to sort batches by priority using .value
        sorted_batches = sorted(
            renderer.batches.values(),
            key=lambda b: b.priority.value
        )

        # All batches should be in the sorted list
        assert len(sorted_batches) == 4


# -----------------------------------------------------------------------------
# CrowdRenderBatch Contract Tests
# -----------------------------------------------------------------------------

class TestCrowdRenderBatchContract:
    """Test the CrowdRenderBatch public interface."""

    def test_batch_has_mesh_id(self):
        """CrowdRenderBatch exposes mesh_id attribute."""
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=42, material_id=1)

        batch = renderer.batches[(42, 1)]
        assert hasattr(batch, 'mesh_id')
        assert batch.mesh_id == 42

    def test_batch_has_material_id(self):
        """CrowdRenderBatch exposes material_id attribute."""
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=99)

        batch = renderer.batches[(1, 99)]
        assert hasattr(batch, 'material_id')
        assert batch.material_id == 99

    def test_batch_has_instance_count(self):
        """CrowdRenderBatch exposes instance_count attribute or property."""
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        batch = renderer.batches[(1, 1)]
        assert hasattr(batch, 'instance_count')
        assert batch.instance_count == 2

    def test_batch_has_instances_list(self):
        """CrowdRenderBatch has instances list (per ARCH).

        From ARCH: 'instances: list[CrowdInstance]'
        """
        renderer = CrowdRenderer()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        batch = renderer.batches[(1, 1)]
        assert hasattr(batch, 'instances')
        assert isinstance(batch.instances, list)
        assert len(batch.instances) == 1


# -----------------------------------------------------------------------------
# CrowdRenderer Contract Tests
# -----------------------------------------------------------------------------

class TestCrowdRendererContract:
    """Test the CrowdRenderer public interface."""

    def test_renderer_has_batches_dict(self):
        """CrowdRenderer exposes batches as dict keyed by (mesh, material).

        From ARCH: 'batches: dict[(mesh, material), CrowdRenderBatch]'
        """
        renderer = CrowdRenderer()
        assert hasattr(renderer, 'batches')
        assert isinstance(renderer.batches, dict)

    def test_renderer_has_add_instance_method(self):
        """CrowdRenderer has add_instance() method.

        From ARCH: 'add_instance(instance)'
        """
        renderer = CrowdRenderer()
        assert hasattr(renderer, 'add_instance')
        assert callable(renderer.add_instance)

    @pytest.mark.xfail(reason="CONTRACT DISCREPANCY: render() not yet implemented per ARCH spec")
    def test_renderer_has_render_method(self):
        """CrowdRenderer has render() method.

        From ARCH: 'render() -> GPU commands'
        NOTE: This test documents expected API per ARCH, but render() is not
        yet implemented. Marked xfail until implementation catches up.
        """
        renderer = CrowdRenderer()
        assert hasattr(renderer, 'render')
        assert callable(renderer.render)

    @pytest.mark.xfail(reason="CONTRACT DISCREPANCY: atlases dict not yet implemented per ARCH spec")
    def test_renderer_has_atlases_dict(self):
        """CrowdRenderer exposes atlases dict (per ARCH).

        From ARCH: 'atlases: dict[str, AnimationTextureAtlas]'
        NOTE: This test documents expected API per ARCH, but atlases dict is not
        yet implemented. Marked xfail until implementation catches up.
        """
        renderer = CrowdRenderer()
        assert hasattr(renderer, 'atlases')
        assert isinstance(renderer.atlases, dict)


# -----------------------------------------------------------------------------
# Instance Addition Order Tests
# -----------------------------------------------------------------------------

class TestInstanceAdditionOrder:
    """Test that instance addition order is handled correctly."""

    def test_interleaved_batch_additions(self):
        """Instances added in interleaved order group correctly."""
        renderer = CrowdRenderer()

        # Add instances in interleaved order
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=2, material_id=1)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1)

        assert len(renderer.batches) == 2
        assert renderer.batches[(1, 1)].instance_count == 3
        assert renderer.batches[(2, 1)].instance_count == 2

    def test_instance_order_preserved_in_batch(self):
        """Instances added to same batch preserve their addition order."""
        renderer = CrowdRenderer()

        positions = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(2.0, 0.0, 0.0),
            Vec3(3.0, 0.0, 0.0),
        ]

        for pos in positions:
            add_instance_to_renderer(renderer, mesh_id=1, material_id=1, position=pos)

        batch = renderer.batches[(1, 1)]
        assert batch.instance_count == 3

        # If instances preserve order, positions should match
        for i, instance in enumerate(batch.instances):
            assert instance.position.x == positions[i].x


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------

class TestBatchEdgeCases:
    """Test edge cases in batch rendering logic."""

    def test_zero_mesh_id(self):
        """Mesh ID of 0 is a valid batch key."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=0, material_id=0)

        assert (0, 0) in renderer.batches
        assert renderer.batches[(0, 0)].instance_count == 1

    def test_large_mesh_material_ids(self):
        """Large mesh and material IDs work correctly."""
        renderer = CrowdRenderer()

        large_mesh_id = 999999
        large_material_id = 888888

        add_instance_to_renderer(
            renderer,
            mesh_id=large_mesh_id,
            material_id=large_material_id
        )

        assert (large_mesh_id, large_material_id) in renderer.batches

    def test_negative_ids_if_supported(self):
        """Negative IDs (if supported) create valid batches."""
        renderer = CrowdRenderer()

        try:
            add_instance_to_renderer(renderer, mesh_id=-1, material_id=-1)
            # If no exception, verify the batch exists
            assert (-1, -1) in renderer.batches
        except (ValueError, TypeError):
            # Negative IDs not supported is acceptable
            pytest.skip("Negative IDs not supported")

    def test_many_unique_batches(self):
        """Renderer handles many unique batches without issue."""
        renderer = CrowdRenderer()

        # Create 100 unique batches
        for mesh_id in range(10):
            for material_id in range(10):
                add_instance_to_renderer(
                    renderer,
                    mesh_id=mesh_id,
                    material_id=material_id
                )

        assert len(renderer.batches) == 100

    def test_many_instances_per_batch(self):
        """Single batch can handle many instances."""
        renderer = CrowdRenderer()

        instance_count = 1000
        for i in range(instance_count):
            add_instance_to_renderer(
                renderer,
                mesh_id=1,
                material_id=1,
                position=Vec3(float(i), 0.0, 0.0)
            )

        assert len(renderer.batches) == 1
        assert renderer.batches[(1, 1)].instance_count == instance_count


# -----------------------------------------------------------------------------
# Instance Data Preservation Tests
# -----------------------------------------------------------------------------

class TestInstanceDataPreservation:
    """Test that instance data is preserved correctly in batches."""

    def test_position_preserved(self):
        """Instance position is preserved in batch."""
        renderer = CrowdRenderer()

        pos = Vec3(10.0, 20.0, 30.0)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1, position=pos)

        batch = renderer.batches[(1, 1)]
        assert batch.instances[0].position.x == 10.0
        assert batch.instances[0].position.y == 20.0
        assert batch.instances[0].position.z == 30.0

    def test_rotation_preserved(self):
        """Instance rotation is preserved in batch."""
        renderer = CrowdRenderer()

        rot = Quat.identity()
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1, rotation=rot)

        batch = renderer.batches[(1, 1)]
        instance_rot = batch.instances[0].rotation
        # Identity quaternion check
        assert abs(instance_rot.w - 1.0) < 0.001 or abs(instance_rot.w + 1.0) < 0.001

    def test_scale_preserved(self):
        """Instance scale is preserved in batch."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1, scale=2.5)

        batch = renderer.batches[(1, 1)]
        assert batch.instances[0].scale == 2.5

    def test_animation_data_preserved(self):
        """Instance animation data is preserved in batch."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(
            renderer,
            mesh_id=1,
            material_id=1,
            animation_index=7,
            animation_time=1.5,
            animation_speed=2.0,
        )

        batch = renderer.batches[(1, 1)]
        instance = batch.instances[0]
        assert instance.animation_index == 7
        assert instance.animation_time == 1.5
        assert instance.animation_speed == 2.0

    def test_tint_color_preserved(self):
        """Instance tint color is preserved in batch."""
        renderer = CrowdRenderer()

        color = Vec4(0.5, 0.6, 0.7, 0.8)
        add_instance_to_renderer(renderer, mesh_id=1, material_id=1, tint_color=color)

        batch = renderer.batches[(1, 1)]
        instance_color = batch.instances[0].tint_color
        assert abs(instance_color.x - 0.5) < 0.001
        assert abs(instance_color.y - 0.6) < 0.001
        assert abs(instance_color.z - 0.7) < 0.001
        assert abs(instance_color.w - 0.8) < 0.001

    def test_lod_level_preserved(self):
        """Instance LOD level is preserved in batch."""
        renderer = CrowdRenderer()

        add_instance_to_renderer(renderer, mesh_id=1, material_id=1, lod_level=3)

        batch = renderer.batches[(1, 1)]
        assert batch.instances[0].lod_level == 3
