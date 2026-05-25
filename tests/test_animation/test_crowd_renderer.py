"""Verification tests for crowd renderer batch logic.

Covers the four acceptance criteria for batch rendering:
1. Instances group by (mesh_id, material_id)
2. Batch priority ordering is correct
3. Empty batches are handled gracefully
4. Single instance per batch works
"""

from engine.animation.crowds.crowd_renderer import (
    CrowdRenderer,
    CrowdInstance,
    RenderPriority,
)


class TestBatchGrouping:
    """Instances group by (mesh_id, material_id)."""

    def test_identical_mesh_material_go_to_same_batch(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        assert renderer.batch_count == 1

    def test_different_mesh_same_material_go_to_different_batches(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)
        assert renderer.batch_count == 2

    def test_same_mesh_different_material_go_to_different_batches(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=2)
        assert renderer.batch_count == 2

    def test_batch_instance_counts_are_correct(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)
        assert renderer.batch_count == 2
        batch_a = renderer.get_batch(1, 1)
        batch_b = renderer.get_batch(2, 1)
        assert batch_a is not None
        assert batch_b is not None
        assert batch_a.instance_buffer.instance_count == 2
        assert batch_b.instance_buffer.instance_count == 1

    def test_total_instance_count_tracks_across_batches(self):
        renderer = CrowdRenderer()
        assert renderer.total_instance_count == 0
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)
        assert renderer.total_instance_count == 3


class TestBatchPriorityOrdering:
    """Batch priority ordering is correct (highest priority first)."""

    def _make_renderer_with_priorities(
        self,
        priorities: list[tuple[int, int, RenderPriority]],
    ) -> CrowdRenderer:
        """Helper: create a renderer with instances at given priorities."""
        renderer = CrowdRenderer()
        for i, (mesh_id, material_id, priority) in enumerate(priorities):
            renderer.add_instance(CrowdInstance(), mesh_id=mesh_id, material_id=material_id)
            renderer.get_batch(mesh_id, material_id).priority = priority
        return renderer

    def test_prepare_render_data_returns_highest_priority_first(self):
        renderer = self._make_renderer_with_priorities([
            (1, 1, RenderPriority.HIGH),
            (2, 1, RenderPriority.CRITICAL),
            (3, 1, RenderPriority.LOW),
        ])
        render_data = renderer.prepare_render_data()
        assert len(render_data) == 3
        assert render_data[0][0].priority == RenderPriority.CRITICAL
        assert render_data[1][0].priority == RenderPriority.HIGH
        assert render_data[2][0].priority == RenderPriority.LOW

    def test_same_priority_preserves_insertion_order(self):
        renderer = self._make_renderer_with_priorities([
            (1, 1, RenderPriority.NORMAL),
            (2, 1, RenderPriority.NORMAL),
            (3, 1, RenderPriority.NORMAL),
        ])
        render_data = renderer.prepare_render_data()
        mesh_ids = [batch[0].mesh_id for batch in render_data]
        assert mesh_ids == [1, 2, 3]

    def test_all_four_priority_levels_sort_correctly(self):
        renderer = self._make_renderer_with_priorities([
            (1, 1, RenderPriority.LOW),
            (2, 1, RenderPriority.NORMAL),
            (3, 1, RenderPriority.HIGH),
            (4, 1, RenderPriority.CRITICAL),
        ])
        render_data = renderer.prepare_render_data()
        priorities = [batch[0].priority for batch in render_data]
        assert priorities == [
            RenderPriority.CRITICAL,
            RenderPriority.HIGH,
            RenderPriority.NORMAL,
            RenderPriority.LOW,
        ]

    def test_invisible_batches_are_excluded(self):
        renderer = self._make_renderer_with_priorities([
            (1, 1, RenderPriority.CRITICAL),
            (2, 1, RenderPriority.NORMAL),
        ])
        renderer.get_batch(1, 1).visible = False
        render_data = renderer.prepare_render_data()
        assert len(render_data) == 1
        assert render_data[0][0].mesh_id == 2


class TestEmptyBatches:
    """Empty batches are handled gracefully."""

    def test_fresh_renderer_has_no_batches(self):
        renderer = CrowdRenderer()
        assert renderer.batch_count == 0
        assert renderer.total_instance_count == 0

    def test_prepare_render_data_returns_empty_list(self):
        renderer = CrowdRenderer()
        render_data = renderer.prepare_render_data()
        assert render_data == []

    def test_add_then_remove_all_instances_is_graceful(self):
        renderer = CrowdRenderer()
        instance = CrowdInstance()
        iid = renderer.add_instance(instance, mesh_id=1, material_id=1)
        renderer.remove_instance(iid)
        assert renderer.batch_count == 1  # batch object persists
        assert renderer.total_instance_count == 0
        render_data = renderer.prepare_render_data()
        assert len(render_data) == 0  # no visible instances

    def test_clear_resets_all_state(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        renderer.add_instance(CrowdInstance(), mesh_id=2, material_id=1)
        renderer.clear()
        assert renderer.batch_count == 0
        assert renderer.total_instance_count == 0
        stats = renderer.get_stats()
        assert stats["total_instances"] == 0

    def test_get_batch_returns_none_for_unknown_key(self):
        renderer = CrowdRenderer()
        assert renderer.get_batch(999, 999) is None

    def test_stats_with_no_instances(self):
        renderer = CrowdRenderer()
        stats = renderer.get_stats()
        assert stats["total_instances"] == 0
        assert stats["visible_instances"] == 0
        assert stats["batch_count"] == 0
        assert stats["total_memory_bytes"] == 0


class TestSingleInstance:
    """Single instance per batch works."""

    def test_add_single_instance_creates_one_batch(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=42)
        assert renderer.batch_count == 1

    def test_single_instance_batch_has_correct_mesh_and_material(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=5, material_id=99)
        batch = renderer.get_batch(5, 99)
        assert batch is not None
        assert batch.mesh_id == 5
        assert batch.material_id == 99

    def test_single_instance_appears_in_instance_buffer(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        batch = renderer.get_batch(1, 1)
        assert batch is not None
        assert batch.instance_buffer.instance_count == 1
        assert len(batch.instances) == 1

    def test_prepare_render_data_includes_single_instance_batch(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        render_data = renderer.prepare_render_data()
        assert len(render_data) == 1
        assert render_data[0][0].mesh_id == 1

    def test_single_instance_stats(self):
        renderer = CrowdRenderer()
        renderer.add_instance(CrowdInstance(), mesh_id=1, material_id=1)
        stats = renderer.get_stats()
        assert stats["total_instances"] == 1
        assert stats["batch_count"] == 1
        assert stats["total_memory_bytes"] > 0
