"""Whitebox tests for batch rendering logic in crowd_renderer.py.

Tests internal batch grouping, priority ordering, property access, and edge cases
with FULL source access to verify implementation details.

Task: T1.3 Batch Rendering Logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    CrowdRenderBatch,
    CrowdRenderer,
    InstanceBuffer,
    RenderPriority,
)
from engine.animation.config import CROWD_RENDERER_CONFIG
from engine.core.math import Vec3, Vec4, Quat


# =============================================================================
# CrowdRenderBatch.instance_count Property Tests (Lines 287-290)
# =============================================================================


class TestBatchInstanceCountProperty:
    """Whitebox tests for CrowdRenderBatch.instance_count property."""

    def test_instance_count_empty_batch(self):
        """instance_count returns 0 for empty batch."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)
        assert batch.instance_count == 0
        # Verify property directly reads from instances list
        assert batch.instance_count == len(batch.instances)

    def test_instance_count_single_instance(self):
        """instance_count returns 1 with single instance."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)
        instance = CrowdInstance(position=Vec3(0, 0, 0))
        batch.add_instance(instance)

        assert batch.instance_count == 1
        assert batch.instance_count == len(batch.instances)

    def test_instance_count_multiple_instances(self):
        """instance_count tracks multiple instances."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        for i in range(10):
            instance = CrowdInstance(position=Vec3(float(i), 0, 0))
            batch.add_instance(instance)

        assert batch.instance_count == 10
        assert batch.instance_count == len(batch.instances)

    def test_instance_count_after_removal(self):
        """instance_count decreases after removal."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        instances = []
        for i in range(5):
            inst = CrowdInstance(position=Vec3(float(i), 0, 0))
            batch.add_instance(inst)
            instances.append(inst)

        assert batch.instance_count == 5

        # Remove one
        batch.remove_instance(instances[0].instance_id)
        assert batch.instance_count == 4
        assert batch.instance_count == len(batch.instances)

    def test_instance_count_is_readonly_property(self):
        """instance_count is a read-only property, not a settable attribute."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        # Verify it's a property (read-only)
        with pytest.raises(AttributeError):
            batch.instance_count = 999

    def test_instance_count_includes_invisible(self):
        """instance_count includes invisible instances (counts all, not just visible)."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        # Add visible instance
        visible = CrowdInstance(position=Vec3(0, 0, 0), visible=True)
        batch.add_instance(visible)

        # Add invisible instance
        invisible = CrowdInstance(position=Vec3(1, 0, 0), visible=False)
        batch.add_instance(invisible)

        # instance_count should be 2 (counts all), not 1
        assert batch.instance_count == 2
        assert batch.get_visible_count() == 1


# =============================================================================
# CrowdRenderer.batches Property Tests (Lines 357-365)
# =============================================================================


class TestRendererBatchesProperty:
    """Whitebox tests for CrowdRenderer.batches property exposing dict keyed by (mesh_id, material_id)."""

    def test_batches_property_returns_dict(self):
        """batches property returns the internal dictionary."""
        renderer = CrowdRenderer()
        assert isinstance(renderer.batches, dict)

    def test_batches_empty_initially(self):
        """batches dict is empty on new renderer."""
        renderer = CrowdRenderer()
        assert len(renderer.batches) == 0
        assert renderer.batches == {}

    def test_batches_keyed_by_mesh_material_tuple(self):
        """batches dict is keyed by (mesh_id, material_id) tuples."""
        renderer = CrowdRenderer()

        instance1 = CrowdInstance(position=Vec3(0, 0, 0))
        renderer.add_instance(instance1, mesh_id=10, material_id=20)

        # Verify key is tuple
        assert (10, 20) in renderer.batches
        assert isinstance(list(renderer.batches.keys())[0], tuple)

    def test_batches_multiple_mesh_material_combinations(self):
        """batches creates separate entries for different (mesh_id, material_id) pairs."""
        renderer = CrowdRenderer()

        # Add instances with different mesh/material combinations
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=2)

        assert len(renderer.batches) == 4
        assert (1, 1) in renderer.batches
        assert (1, 2) in renderer.batches
        assert (2, 1) in renderer.batches
        assert (2, 2) in renderer.batches

    def test_batches_same_key_groups_together(self):
        """Multiple instances with same (mesh_id, material_id) go to same batch."""
        renderer = CrowdRenderer()

        # Add 3 instances with same mesh/material
        for i in range(3):
            renderer.add_instance(CrowdInstance(position=Vec3(float(i), 0, 0)), mesh_id=5, material_id=10)

        assert len(renderer.batches) == 1
        assert (5, 10) in renderer.batches
        assert renderer.batches[(5, 10)].instance_count == 3

    def test_batches_is_same_object_as_internal(self):
        """batches property returns the actual internal dict, not a copy."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        # Get reference twice
        batches1 = renderer.batches
        batches2 = renderer.batches

        # Should be same object
        assert batches1 is batches2
        assert batches1 is renderer._batches

    def test_batches_values_are_render_batches(self):
        """batches dict values are CrowdRenderBatch instances."""
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        batch = renderer.batches[(1, 1)]
        assert isinstance(batch, CrowdRenderBatch)
        assert batch.mesh_id == 1
        assert batch.material_id == 1


# =============================================================================
# Batch Grouping by (mesh_id, material_id) Tests (Lines 378/392)
# =============================================================================


class TestBatchGroupingByKey:
    """Whitebox tests for batch grouping via batch_key tuple (mesh_id, material_id)."""

    def test_batch_key_creates_new_batch(self):
        """New batch_key creates a new CrowdRenderBatch."""
        renderer = CrowdRenderer()

        # Add first instance - should create batch
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)

        assert len(renderer._batches) == 1
        batch = renderer._batches[(1, 2)]
        assert batch.mesh_id == 1
        assert batch.material_id == 2

    def test_same_batch_key_reuses_batch(self):
        """Same batch_key adds to existing batch."""
        renderer = CrowdRenderer()

        inst1 = CrowdInstance(position=Vec3(0, 0, 0))
        inst2 = CrowdInstance(position=Vec3(1, 0, 0))

        renderer.add_instance(inst1, mesh_id=1, material_id=2)
        renderer.add_instance(inst2, mesh_id=1, material_id=2)

        # Still only one batch
        assert len(renderer._batches) == 1
        # Both instances in same batch
        assert renderer._batches[(1, 2)].instance_count == 2

    def test_different_mesh_creates_separate_batch(self):
        """Different mesh_id creates separate batch."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)

        assert len(renderer._batches) == 2
        assert (1, 1) in renderer._batches
        assert (2, 1) in renderer._batches

    def test_different_material_creates_separate_batch(self):
        """Different material_id creates separate batch."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)

        assert len(renderer._batches) == 2
        assert (1, 1) in renderer._batches
        assert (1, 2) in renderer._batches

    def test_batch_key_order_matters(self):
        """batch_key (mesh_id, material_id) order is significant."""
        renderer = CrowdRenderer()

        # (mesh=1, material=2) vs (mesh=2, material=1) are different
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)

        assert len(renderer._batches) == 2
        assert (1, 2) in renderer._batches
        assert (2, 1) in renderer._batches
        assert (1, 2) != (2, 1)

    def test_zero_ids_valid_batch_key(self):
        """Zero mesh_id and material_id create valid batch."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=0, material_id=0)

        assert (0, 0) in renderer._batches
        assert renderer._batches[(0, 0)].instance_count == 1

    def test_large_ids_valid_batch_key(self):
        """Large mesh_id and material_id values work correctly."""
        renderer = CrowdRenderer()

        large_mesh = 2**31 - 1  # Max 32-bit signed int
        large_material = 2**31 - 1

        renderer.add_instance(CrowdInstance(), mesh_id=large_mesh, material_id=large_material)

        assert (large_mesh, large_material) in renderer._batches

    def test_batch_stores_correct_mesh_material_ids(self):
        """Created batch has correct mesh_id and material_id fields."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=42, material_id=99)

        batch = renderer._batches[(42, 99)]
        assert batch.mesh_id == 42
        assert batch.material_id == 99


# =============================================================================
# Priority Ordering Tests (Line 491 - prepare_render_data sorting)
# =============================================================================


class TestPriorityOrdering:
    """Whitebox tests for batch priority ordering via prepare_render_data() sorting."""

    def test_prepare_render_data_sorts_by_priority_descending(self):
        """prepare_render_data sorts batches by priority.value descending (highest first)."""
        renderer = CrowdRenderer()

        # Add batches with different priorities
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer._batches[(1, 1)].priority = RenderPriority.LOW  # value=0

        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=2)
        renderer._batches[(2, 2)].priority = RenderPriority.HIGH  # value=2

        renderer.add_instance(CrowdInstance(), mesh_id=3, material_id=3)
        renderer._batches[(3, 3)].priority = RenderPriority.NORMAL  # value=1

        result = renderer.prepare_render_data()

        # Should be sorted: HIGH(2), NORMAL(1), LOW(0)
        assert len(result) == 3
        assert result[0][0].priority == RenderPriority.HIGH
        assert result[1][0].priority == RenderPriority.NORMAL
        assert result[2][0].priority == RenderPriority.LOW

    def test_critical_priority_first(self):
        """CRITICAL priority batches render first."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer._batches[(1, 1)].priority = RenderPriority.NORMAL

        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=2)
        renderer._batches[(2, 2)].priority = RenderPriority.CRITICAL  # value=3

        result = renderer.prepare_render_data()

        assert result[0][0].priority == RenderPriority.CRITICAL

    def test_priority_enum_values(self):
        """Verify RenderPriority enum values for sorting."""
        assert RenderPriority.LOW.value == 0
        assert RenderPriority.NORMAL.value == 1
        assert RenderPriority.HIGH.value == 2
        assert RenderPriority.CRITICAL.value == 3

    def test_same_priority_stable_order(self):
        """Batches with same priority maintain some stable order."""
        renderer = CrowdRenderer()

        # Add multiple batches with same priority
        for i in range(5):
            renderer.add_instance(CrowdInstance(), mesh_id=i, material_id=0)
            renderer._batches[(i, 0)].priority = RenderPriority.NORMAL

        result = renderer.prepare_render_data()

        assert len(result) == 5
        # All have NORMAL priority
        for batch, _ in result:
            assert batch.priority == RenderPriority.NORMAL

    def test_prepare_render_data_returns_tuples(self):
        """prepare_render_data returns list of (batch, instance_buffer) tuples."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        assert len(result) == 1
        batch, buffer = result[0]
        assert isinstance(batch, CrowdRenderBatch)
        assert isinstance(buffer, InstanceBuffer)
        assert buffer is batch.instance_buffer

    def test_priority_change_affects_order(self):
        """Changing batch priority changes render order."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=2)

        renderer._batches[(1, 1)].priority = RenderPriority.LOW
        renderer._batches[(2, 2)].priority = RenderPriority.HIGH

        result1 = renderer.prepare_render_data()
        assert result1[0][0].mesh_id == 2  # HIGH first

        # Change priority
        renderer._batches[(1, 1)].priority = RenderPriority.CRITICAL

        result2 = renderer.prepare_render_data()
        assert result2[0][0].mesh_id == 1  # Now CRITICAL is first


# =============================================================================
# Empty Batch Filtering Tests (Line 487 - get_visible_count() > 0)
# =============================================================================


class TestEmptyBatchFiltering:
    """Whitebox tests for empty batch filtering via get_visible_count() > 0."""

    def test_empty_batch_excluded_from_render(self):
        """Empty batches are excluded from prepare_render_data."""
        renderer = CrowdRenderer()

        # Create batch but don't add instances
        renderer._batches[(1, 1)] = CrowdRenderBatch(mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        assert len(result) == 0

    def test_batch_all_invisible_excluded(self):
        """Batch with all invisible instances is excluded."""
        renderer = CrowdRenderer()

        # Add instance then make it invisible
        inst = CrowdInstance(visible=False)
        renderer.add_instance(inst, mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        # Batch exists but has 0 visible
        assert (1, 1) in renderer._batches
        assert renderer._batches[(1, 1)].get_visible_count() == 0
        assert len(result) == 0

    def test_batch_some_invisible_included(self):
        """Batch with some visible instances is included."""
        renderer = CrowdRenderer()

        # Add visible instance
        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)
        # Add invisible instance
        renderer.add_instance(CrowdInstance(visible=False), mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        # Batch has 1 visible, so included
        assert len(result) == 1
        assert renderer._batches[(1, 1)].get_visible_count() == 1

    def test_batch_visible_false_excluded(self):
        """Batch with visible=False is excluded even with visible instances."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)

        # Set batch visible to False
        renderer._batches[(1, 1)].visible = False

        result = renderer.prepare_render_data()

        # Batch is not visible
        assert len(result) == 0

    def test_filter_preserves_non_empty_batches(self):
        """Non-empty batches with visible instances are preserved."""
        renderer = CrowdRenderer()

        # Add batch with visible instance
        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)
        # Add empty batch
        renderer._batches[(2, 2)] = CrowdRenderBatch(mesh_id=2, material_id=2)

        result = renderer.prepare_render_data()

        assert len(result) == 1
        assert result[0][0].mesh_id == 1

    def test_get_visible_count_implementation(self):
        """Verify get_visible_count sums visible instances."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        # Add mix of visible/invisible
        batch.add_instance(CrowdInstance(visible=True))
        batch.add_instance(CrowdInstance(visible=True))
        batch.add_instance(CrowdInstance(visible=False))
        batch.add_instance(CrowdInstance(visible=True))
        batch.add_instance(CrowdInstance(visible=False))

        assert batch.instance_count == 5
        assert batch.get_visible_count() == 3


# =============================================================================
# Single Instance Edge Case Tests
# =============================================================================


class TestSingleInstanceEdgeCase:
    """Whitebox tests for single instance per batch edge cases."""

    def test_single_instance_batch_renders(self):
        """Single instance batch is included in render data."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        assert len(result) == 1
        batch, buffer = result[0]
        assert batch.instance_count == 1
        assert buffer.instance_count == 1

    def test_single_instance_batch_properties(self):
        """Single instance batch has correct properties."""
        batch = CrowdRenderBatch(mesh_id=42, material_id=99)

        instance = CrowdInstance(
            position=Vec3(1, 2, 3),
            visible=True,
            animation_index=5,
        )
        batch.add_instance(instance)

        assert batch.instance_count == 1
        assert batch.get_visible_count() == 1
        assert batch.mesh_id == 42
        assert batch.material_id == 99
        assert batch.visible is True
        assert batch.priority == RenderPriority.NORMAL

    def test_single_instance_removal_empties_batch(self):
        """Removing single instance empties batch."""
        renderer = CrowdRenderer()

        inst = CrowdInstance(visible=True)
        renderer.add_instance(inst, mesh_id=1, material_id=1)

        # Remove the single instance
        renderer.remove_instance(inst.instance_id)

        # Batch still exists but is empty
        assert (1, 1) in renderer._batches
        assert renderer._batches[(1, 1)].instance_count == 0

        # Should not appear in render data
        result = renderer.prepare_render_data()
        assert len(result) == 0

    def test_single_instance_made_invisible(self):
        """Single instance made invisible excludes batch from render."""
        renderer = CrowdRenderer()

        inst = CrowdInstance(visible=True)
        renderer.add_instance(inst, mesh_id=1, material_id=1)

        # Make invisible
        inst.visible = False

        result = renderer.prepare_render_data()
        assert len(result) == 0

    def test_single_invisible_instance_batch(self):
        """Creating batch with single invisible instance."""
        renderer = CrowdRenderer()

        inst = CrowdInstance(visible=False)
        renderer.add_instance(inst, mesh_id=1, material_id=1)

        assert renderer._batches[(1, 1)].instance_count == 1
        assert renderer._batches[(1, 1)].get_visible_count() == 0

        result = renderer.prepare_render_data()
        assert len(result) == 0


# =============================================================================
# Combined Integration Tests
# =============================================================================


class TestBatchRenderingIntegration:
    """Integration tests combining multiple batch rendering features."""

    def test_multiple_batches_different_priorities(self):
        """Multiple batches with different priorities sorted correctly."""
        renderer = CrowdRenderer()

        # Create 4 batches with all priority levels
        priorities = [
            (RenderPriority.LOW, 1),
            (RenderPriority.NORMAL, 2),
            (RenderPriority.HIGH, 3),
            (RenderPriority.CRITICAL, 4),
        ]

        for priority, mesh_id in priorities:
            renderer.add_instance(CrowdInstance(visible=True), mesh_id=mesh_id, material_id=0)
            renderer._batches[(mesh_id, 0)].priority = priority

        result = renderer.prepare_render_data()

        # Should be sorted: CRITICAL(4), HIGH(3), NORMAL(2), LOW(1)
        assert len(result) == 4
        assert result[0][0].mesh_id == 4  # CRITICAL
        assert result[1][0].mesh_id == 3  # HIGH
        assert result[2][0].mesh_id == 2  # NORMAL
        assert result[3][0].mesh_id == 1  # LOW

    def test_mixed_visible_invisible_batches(self):
        """Mix of visible and invisible batches filtered correctly."""
        renderer = CrowdRenderer()

        # Batch 1: all visible
        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=0)

        # Batch 2: all invisible
        renderer.add_instance(CrowdInstance(visible=False), mesh_id=2, material_id=0)

        # Batch 3: some visible
        renderer.add_instance(CrowdInstance(visible=True), mesh_id=3, material_id=0)
        renderer.add_instance(CrowdInstance(visible=False), mesh_id=3, material_id=0)

        # Batch 4: empty
        renderer._batches[(4, 0)] = CrowdRenderBatch(mesh_id=4, material_id=0)

        result = renderer.prepare_render_data()

        # Only batches 1 and 3 have visible instances
        assert len(result) == 2
        mesh_ids = {r[0].mesh_id for r in result}
        assert mesh_ids == {1, 3}

    def test_batch_grouping_stress(self):
        """Stress test: many instances across many batches."""
        renderer = CrowdRenderer()

        # Create 10 mesh x 10 material = 100 batches
        for mesh_id in range(10):
            for material_id in range(10):
                # Add 5 instances per batch
                for _ in range(5):
                    renderer.add_instance(
                        CrowdInstance(visible=True),
                        mesh_id=mesh_id,
                        material_id=material_id,
                    )

        # Verify batch count
        assert len(renderer.batches) == 100

        # Verify total instances
        assert renderer.total_instance_count == 500

        # Verify all batches render
        result = renderer.prepare_render_data()
        assert len(result) == 100

    def test_instance_count_consistency(self):
        """instance_count stays consistent through various operations."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        # Empty
        assert batch.instance_count == 0

        # Add instances
        ids = []
        for i in range(10):
            inst = CrowdInstance()
            batch.add_instance(inst)
            ids.append(inst.instance_id)
            assert batch.instance_count == i + 1

        # Remove some
        batch.remove_instance(ids[0])
        assert batch.instance_count == 9

        batch.remove_instance(ids[5])
        assert batch.instance_count == 8

        # Remove nonexistent - no change
        batch.remove_instance(999999)
        assert batch.instance_count == 8

    def test_clear_renderer_resets_batches(self):
        """Clearing renderer removes all batches."""
        renderer = CrowdRenderer()

        # Add instances to multiple batches
        for i in range(5):
            renderer.add_instance(CrowdInstance(), mesh_id=i, material_id=i)

        assert len(renderer.batches) == 5

        # Clear
        renderer.clear()

        assert len(renderer.batches) == 0
        assert renderer.total_instance_count == 0
        result = renderer.prepare_render_data()
        assert len(result) == 0


# =============================================================================
# Batch Access Methods Tests
# =============================================================================


class TestBatchAccessMethods:
    """Tests for batch access methods complementing batches property."""

    def test_get_batch_returns_correct_batch(self):
        """get_batch(mesh_id, material_id) returns correct batch."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=10, material_id=20)
        renderer.add_instance(CrowdInstance(), mesh_id=30, material_id=40)

        batch = renderer.get_batch(10, 20)
        assert batch is not None
        assert batch.mesh_id == 10
        assert batch.material_id == 20

    def test_get_batch_nonexistent_returns_none(self):
        """get_batch returns None for nonexistent batch."""
        renderer = CrowdRenderer()

        batch = renderer.get_batch(999, 999)
        assert batch is None

    def test_get_batches_iterator(self):
        """get_batches() yields all batches."""
        renderer = CrowdRenderer()

        for i in range(3):
            renderer.add_instance(CrowdInstance(), mesh_id=i, material_id=0)

        batches = list(renderer.get_batches())
        assert len(batches) == 3
        mesh_ids = {b.mesh_id for b in batches}
        assert mesh_ids == {0, 1, 2}

    def test_batch_count_property(self):
        """batch_count property matches batches dict length."""
        renderer = CrowdRenderer()

        assert renderer.batch_count == 0

        for i in range(5):
            renderer.add_instance(CrowdInstance(), mesh_id=i, material_id=0)
            assert renderer.batch_count == i + 1
            assert renderer.batch_count == len(renderer.batches)


# =============================================================================
# Batch Update and Animation Tests
# =============================================================================


class TestBatchUpdateBehavior:
    """Tests for batch update behavior affecting render state."""

    def test_batch_update_advances_visible_instances(self):
        """Batch.update() advances time for visible instances only."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        visible_inst = CrowdInstance(visible=True, animation_time=0.0, animation_speed=1.0)
        invisible_inst = CrowdInstance(visible=False, animation_time=0.0, animation_speed=1.0)

        batch.add_instance(visible_inst)
        batch.add_instance(invisible_inst)

        batch.update(dt=0.5)

        # Only visible advances
        assert visible_inst.animation_time == 0.5
        assert invisible_inst.animation_time == 0.0

    def test_batch_sort_by_distance_rebuilds_buffer(self):
        """sort_by_distance rebuilds instance buffer."""
        batch = CrowdRenderBatch(mesh_id=1, material_id=1)

        # Add instances at different positions
        for i in range(5):
            batch.add_instance(CrowdInstance(position=Vec3(float(i) * 10, 0, 0)))

        camera_pos = Vec3(0, 0, 0)
        batch.sort_by_distance(camera_pos, front_to_back=True)

        # Buffer should be rebuilt (dirty flag reset in next update)
        assert batch.instance_buffer.dirty is True


# =============================================================================
# Animation Atlas Integration Tests
# =============================================================================


class TestAnimationAtlasIntegration:
    """Tests for animation atlas integration with batches."""

    def test_add_instance_with_atlas_name(self):
        """add_instance with atlas_name attaches atlas to batch."""
        renderer = CrowdRenderer()

        # Mock atlas
        mock_atlas = MagicMock()
        renderer.register_animation_atlas("test_atlas", mock_atlas)

        renderer.add_instance(
            CrowdInstance(),
            mesh_id=1,
            material_id=1,
            atlas_name="test_atlas",
        )

        batch = renderer.get_batch(1, 1)
        assert batch.animation_atlas is mock_atlas

    def test_batch_without_atlas(self):
        """Batch created without atlas_name has None atlas."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        batch = renderer.get_batch(1, 1)
        assert batch.animation_atlas is None


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCasesAndBoundaries:
    """Edge case and boundary condition tests."""

    def test_negative_mesh_material_ids(self):
        """Negative mesh_id and material_id work as batch keys."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=-1, material_id=-1)

        assert (-1, -1) in renderer.batches
        batch = renderer.get_batch(-1, -1)
        assert batch is not None
        assert batch.instance_count == 1

    def test_batch_priority_modification(self):
        """Batch priority can be modified after creation."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        batch = renderer.get_batch(1, 1)
        assert batch.priority == RenderPriority.NORMAL

        batch.priority = RenderPriority.CRITICAL
        assert batch.priority == RenderPriority.CRITICAL

    def test_batch_visibility_modification(self):
        """Batch visibility can be modified after creation."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)

        batch = renderer.get_batch(1, 1)
        assert batch.visible is True

        batch.visible = False
        assert batch.visible is False

        # Should not render
        result = renderer.prepare_render_data()
        assert len(result) == 0

    def test_instance_id_returned_by_add(self):
        """add_instance returns the instance_id."""
        renderer = CrowdRenderer()

        inst = CrowdInstance()
        returned_id = renderer.add_instance(inst, mesh_id=1, material_id=1)

        assert returned_id == inst.instance_id
        assert isinstance(returned_id, int)
        assert returned_id > 0

    def test_remove_instance_across_batches(self):
        """remove_instance searches all batches for instance."""
        renderer = CrowdRenderer()

        inst1 = CrowdInstance()
        inst2 = CrowdInstance()

        renderer.add_instance(inst1, mesh_id=1, material_id=1)
        renderer.add_instance(inst2, mesh_id=2, material_id=2)

        # Remove from second batch
        result = renderer.remove_instance(inst2.instance_id)

        assert result is True
        assert renderer.total_instance_count == 1
        assert renderer.get_batch(2, 2).instance_count == 0

    def test_remove_nonexistent_instance(self):
        """remove_instance returns False for nonexistent instance."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)

        result = renderer.remove_instance(99999999)
        assert result is False
        assert renderer.total_instance_count == 1


# =============================================================================
# Render Data Structure Tests
# =============================================================================


class TestRenderDataStructure:
    """Tests for prepare_render_data return structure."""

    def test_render_data_batch_buffer_pair(self):
        """Each element is (batch, buffer) where buffer is batch.instance_buffer."""
        renderer = CrowdRenderer()

        renderer.add_instance(CrowdInstance(visible=True), mesh_id=1, material_id=1)

        result = renderer.prepare_render_data()

        assert len(result) == 1
        batch, buffer = result[0]

        # Buffer is the batch's instance buffer
        assert buffer is batch.instance_buffer

    def test_render_data_order_deterministic(self):
        """Same priority batches have deterministic order."""
        renderer = CrowdRenderer()

        for i in range(10):
            renderer.add_instance(CrowdInstance(visible=True), mesh_id=i, material_id=0)
            renderer._batches[(i, 0)].priority = RenderPriority.NORMAL

        result1 = renderer.prepare_render_data()
        result2 = renderer.prepare_render_data()

        # Same order both times
        mesh_ids1 = [r[0].mesh_id for r in result1]
        mesh_ids2 = [r[0].mesh_id for r in result2]
        assert mesh_ids1 == mesh_ids2
