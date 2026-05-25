"""
Tests for the Barrier Manager.

Tests automatic barrier insertion between passes as specified in
RENDERING_CONTEXT.md Section 6.1 and Section 11.
"""

import pytest

from engine.rendering.framegraph import (
    AccessFlags,
    Barrier,
    BarrierBatch,
    BarrierManager,
    BarrierType,
    ComputePass,
    GraphicsPass,
    PassFlags,
    PipelineStage,
    ResourceFormat,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceStateTracker,
)


# =============================================================================
# ResourceStateTracker
# =============================================================================


class TestResourceStateTracker:
    """Test resource state tracking."""

    def test_initial_state_undefined(self):
        """Test that initial state is undefined."""
        tracker = ResourceStateTracker()
        handle = ResourceHandle()

        state = tracker.get_state(handle)
        assert state == ResourceState.UNDEFINED

    def test_set_and_get_state(self):
        """Test setting and getting resource state."""
        tracker = ResourceStateTracker()
        handle = ResourceHandle()

        tracker.set_state(handle, ResourceState.RENDER_TARGET)
        state = tracker.get_state(handle)

        assert state == ResourceState.RENDER_TARGET

    def test_subresource_state(self):
        """Test tracking subresource states."""
        tracker = ResourceStateTracker()
        handle = ResourceHandle()

        # Set different states for different subresources
        tracker.set_state(handle, ResourceState.SHADER_RESOURCE, subresource=0)
        tracker.set_state(handle, ResourceState.UNORDERED_ACCESS, subresource=1)

        assert tracker.get_state(handle, subresource=0) == ResourceState.SHADER_RESOURCE
        assert tracker.get_state(handle, subresource=1) == ResourceState.UNORDERED_ACCESS

    def test_whole_resource_state_clears_subresources(self):
        """Test that setting whole-resource state clears subresource states."""
        tracker = ResourceStateTracker()
        handle = ResourceHandle()

        tracker.set_state(handle, ResourceState.SHADER_RESOURCE, subresource=0)
        tracker.set_state(handle, ResourceState.RENDER_TARGET)  # whole resource

        # Subresource state should fall back to whole-resource state
        state = tracker.get_state(handle, subresource=0)
        assert state == ResourceState.RENDER_TARGET

    def test_clear(self):
        """Test clearing all states."""
        tracker = ResourceStateTracker()
        handle = ResourceHandle()

        tracker.set_state(handle, ResourceState.RENDER_TARGET)
        tracker.clear()

        state = tracker.get_state(handle)
        assert state == ResourceState.UNDEFINED


# =============================================================================
# Barrier
# =============================================================================


class TestBarrier:
    """Test Barrier dataclass."""

    def test_barrier_creation(self):
        """Test creating a barrier."""
        handle = ResourceHandle()

        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.RENDER_TARGET,
            new_state=ResourceState.SHADER_RESOURCE,
        )

        assert barrier.handle is handle
        assert barrier.barrier_type == BarrierType.TRANSITION
        assert barrier.old_state == ResourceState.RENDER_TARGET
        assert barrier.new_state == ResourceState.SHADER_RESOURCE

    def test_barrier_repr(self):
        """Test barrier string representation."""
        handle = ResourceHandle()
        handle.descriptor = type(
            "Desc", (), {"name": "albedo", "resource_type": None}
        )()
        handle.descriptor.resource_type = None

        # Create descriptor properly
        from engine.rendering.framegraph import ResourceDescriptor

        handle.descriptor = ResourceDescriptor(name="albedo")

        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.RENDER_TARGET,
            new_state=ResourceState.SHADER_RESOURCE,
        )

        repr_str = repr(barrier)
        assert "albedo" in repr_str
        assert "RENDER_TARGET" in repr_str
        assert "SHADER_RESOURCE" in repr_str


# =============================================================================
# BarrierBatch
# =============================================================================


class TestBarrierBatch:
    """Test BarrierBatch functionality."""

    def test_batch_creation(self):
        """Test creating a barrier batch."""
        batch = BarrierBatch(before_pass="Lighting")

        assert batch.before_pass == "Lighting"
        assert batch.is_empty() is True

    def test_add_barriers(self):
        """Test adding barriers to a batch."""
        batch = BarrierBatch()
        handle = ResourceHandle()

        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )

        batch.add(barrier)

        assert batch.is_empty() is False
        assert len(batch.barriers) == 1


# =============================================================================
# BarrierManager
# =============================================================================


class TestBarrierManager:
    """Test BarrierManager functionality."""

    def test_basic_transition_barrier(self):
        """Test generating a basic state transition barrier.

        Per spec Integration Pattern from Section 11:
        "Pass A writes texture as RTV (render target)
         Pass B reads same texture as SRV (shader resource)
         Frame graph automatically inserts:
           Barrier(texture, RENDER_TARGET -> SHADER_RESOURCE) between A and B"
        """
        rm = ResourceManager()
        bm = BarrierManager(rm)

        # Create resource
        handle = rm.create_transient("texture")

        # Create passes
        pass_a = GraphicsPass(name="PassA")
        pass_a.write(handle, ResourceState.RENDER_TARGET)

        pass_b = ComputePass(name="PassB")
        pass_b.read(handle, ResourceState.SHADER_RESOURCE)

        # Analyze passes
        batches = bm.analyze_passes([pass_a, pass_b])

        # Verify barriers
        assert len(batches) == 2

        # First pass: UNDEFINED -> RENDER_TARGET
        batch_a = batches[0]
        assert any(
            b.old_state == ResourceState.UNDEFINED
            and b.new_state == ResourceState.RENDER_TARGET
            for b in batch_a.barriers
        )

        # Second pass: RENDER_TARGET -> SHADER_RESOURCE
        batch_b = batches[1]
        assert any(
            b.old_state == ResourceState.RENDER_TARGET
            and b.new_state == ResourceState.SHADER_RESOURCE
            for b in batch_b.barriers
        )

    def test_no_barrier_same_state(self):
        """Test that no barrier is generated for same state."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("texture")

        # Two passes both reading as shader resource
        pass_a = ComputePass(name="PassA")
        pass_a.read(handle, ResourceState.SHADER_RESOURCE)

        pass_b = ComputePass(name="PassB")
        pass_b.read(handle, ResourceState.SHADER_RESOURCE)

        batches = bm.analyze_passes([pass_a, pass_b])

        # First pass needs transition from UNDEFINED
        # Second pass should not need a barrier (already in correct state)
        assert len(batches) == 2

        # Count SRV->SRV transitions (should be 0)
        srv_to_srv = [
            b
            for batch in batches
            for b in batch.barriers
            if b.old_state == ResourceState.SHADER_RESOURCE
            and b.new_state == ResourceState.SHADER_RESOURCE
        ]
        assert len(srv_to_srv) == 0

    def test_uav_barrier(self):
        """Test UAV barrier for read-after-write hazard."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("buffer")

        # First pass writes UAV
        pass_a = ComputePass(name="PassA")
        pass_a.write(handle, ResourceState.UNORDERED_ACCESS)

        # Second pass reads UAV (RAW hazard)
        pass_b = ComputePass(name="PassB")
        pass_b.read(handle, ResourceState.UNORDERED_ACCESS)

        batches = bm.analyze_passes([pass_a, pass_b])

        # Should have UAV barrier before PassB
        uav_barriers = [
            b
            for batch in batches
            for b in batch.barriers
            if b.barrier_type == BarrierType.UAV
        ]
        assert len(uav_barriers) > 0

    def test_depth_state_transitions(self):
        """Test depth buffer state transitions."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("depth", format=ResourceFormat.D32_FLOAT)

        # Pass A writes depth
        pass_a = GraphicsPass(name="DepthPrepass")
        pass_a.write(handle, ResourceState.DEPTH_WRITE)

        # Pass B reads depth
        pass_b = GraphicsPass(name="Lighting")
        pass_b.read(handle, ResourceState.DEPTH_READ)

        batches = bm.analyze_passes([pass_a, pass_b])

        # Verify DEPTH_WRITE -> DEPTH_READ transition
        transitions = [
            b
            for batch in batches
            for b in batch.barriers
            if b.old_state == ResourceState.DEPTH_WRITE
            and b.new_state == ResourceState.DEPTH_READ
        ]
        assert len(transitions) > 0

    def test_external_resource_initial_state(self):
        """Test that external resources start with their initial state."""
        rm = ResourceManager()

        # Register external with initial state
        handle = rm.register_external(
            "backbuffer",
            gpu_resource=None,
            is_backbuffer=True,
        )
        external = rm.get_external("backbuffer")
        external.current_state = ResourceState.PRESENT

        bm = BarrierManager(rm)

        # Pass writes to backbuffer
        pass_node = GraphicsPass(name="Final")
        pass_node.write(handle, ResourceState.RENDER_TARGET)

        batches = bm.analyze_passes([pass_node])

        # Should have PRESENT -> RENDER_TARGET transition
        transitions = [
            b
            for batch in batches
            for b in batch.barriers
            if b.old_state == ResourceState.PRESENT
            and b.new_state == ResourceState.RENDER_TARGET
        ]
        assert len(transitions) == 1

    def test_prepare_for_present(self):
        """Test preparing backbuffer for presentation."""
        rm = ResourceManager()

        handle = rm.register_external(
            "backbuffer",
            gpu_resource=None,
            is_backbuffer=True,
        )

        bm = BarrierManager(rm)

        # Set current state to render target
        bm._state_tracker.set_state(handle, ResourceState.RENDER_TARGET)

        # Prepare for present
        barrier = bm.prepare_for_present(handle)

        assert barrier is not None
        assert barrier.old_state == ResourceState.RENDER_TARGET
        assert barrier.new_state == ResourceState.PRESENT

    def test_get_final_states(self):
        """Test getting final resource states after analysis."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        h1 = rm.create_transient("tex1")
        h2 = rm.create_transient("tex2")

        pass_node = GraphicsPass(name="Render")
        pass_node.write(h1, ResourceState.RENDER_TARGET)
        pass_node.read(h2, ResourceState.SHADER_RESOURCE)

        bm.analyze_passes([pass_node])

        final_states = bm.get_final_states()

        assert "tex1" in final_states
        assert "tex2" in final_states
        assert final_states["tex1"] == ResourceState.RENDER_TARGET
        assert final_states["tex2"] == ResourceState.SHADER_RESOURCE

    def test_aliasing_barrier(self):
        """Test creating an aliasing barrier."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        old_handle = rm.create_transient("old")
        new_handle = rm.create_transient("new")

        barrier = bm.create_aliasing_barrier(old_handle, new_handle)

        assert barrier.barrier_type == BarrierType.ALIASING
        assert barrier.handle is new_handle

    def test_reset(self):
        """Test resetting the barrier manager."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("test")
        pass_node = GraphicsPass(name="Pass")
        pass_node.write(handle, ResourceState.RENDER_TARGET)

        bm.analyze_passes([pass_node])
        assert len(bm._barrier_batches) > 0

        bm.reset()
        assert len(bm._barrier_batches) == 0


# =============================================================================
# Pipeline Stage and Access Flags
# =============================================================================


class TestPipelineStages:
    """Test pipeline stage mappings."""

    def test_render_target_stage(self):
        """Test render target maps to color attachment output."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("rt")

        pass_node = GraphicsPass(name="Pass")
        pass_node.write(handle, ResourceState.RENDER_TARGET)

        batches = bm.analyze_passes([pass_node])

        # Find the transition barrier
        barrier = batches[0].barriers[0]
        assert barrier.dst_stage == PipelineStage.COLOR_ATTACHMENT_OUTPUT

    def test_shader_resource_stage(self):
        """Test shader resource maps to fragment shader."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("tex")

        pass_node = ComputePass(name="Pass")
        pass_node.read(handle, ResourceState.SHADER_RESOURCE)

        batches = bm.analyze_passes([pass_node])

        barrier = batches[0].barriers[0]
        assert barrier.dst_stage == PipelineStage.FRAGMENT_SHADER

    def test_compute_uav_stage(self):
        """Test UAV maps to compute shader stage."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("buf")

        pass_node = ComputePass(name="Pass")
        pass_node.write(handle, ResourceState.UNORDERED_ACCESS)

        batches = bm.analyze_passes([pass_node])

        barrier = batches[0].barriers[0]
        assert barrier.dst_stage == PipelineStage.COMPUTE_SHADER


# =============================================================================
# Integration Tests
# =============================================================================


class TestBarrierIntegration:
    """Integration tests for barrier management."""

    def test_gbuffer_to_lighting_pipeline(self):
        """Test typical deferred rendering barrier flow.

        GBuffer pass writes multiple RTs, Lighting reads them as SRVs.
        """
        rm = ResourceManager()
        bm = BarrierManager(rm)

        # G-Buffer textures
        albedo = rm.create_transient("albedo")
        normal = rm.create_transient("normal")
        depth = rm.create_transient("depth", format=ResourceFormat.D32_FLOAT)
        hdr = rm.create_transient("hdr")

        # GBuffer pass
        gbuffer = GraphicsPass(name="GBuffer")
        gbuffer.write(albedo, ResourceState.RENDER_TARGET)
        gbuffer.write(normal, ResourceState.RENDER_TARGET)
        gbuffer.write(depth, ResourceState.DEPTH_WRITE)

        # Lighting pass
        lighting = ComputePass(name="Lighting")
        lighting.read(albedo, ResourceState.SHADER_RESOURCE)
        lighting.read(normal, ResourceState.SHADER_RESOURCE)
        lighting.read(depth, ResourceState.SHADER_RESOURCE)
        lighting.write(hdr, ResourceState.UNORDERED_ACCESS)

        batches = bm.analyze_passes([gbuffer, lighting])

        # Verify lighting pass gets barriers for RT -> SRV transitions
        lighting_batch = batches[1]

        rt_to_srv = [
            b
            for b in lighting_batch.barriers
            if b.old_state == ResourceState.RENDER_TARGET
            and b.new_state == ResourceState.SHADER_RESOURCE
        ]
        assert len(rt_to_srv) == 2  # albedo and normal

        depth_transition = [
            b
            for b in lighting_batch.barriers
            if b.old_state == ResourceState.DEPTH_WRITE
            and b.new_state == ResourceState.SHADER_RESOURCE
        ]
        assert len(depth_transition) == 1

    def test_culled_passes_skipped(self):
        """Test that culled passes don't generate barriers."""
        rm = ResourceManager()
        bm = BarrierManager(rm)

        handle = rm.create_transient("tex")

        culled_pass = GraphicsPass(name="Culled")
        culled_pass.write(handle, ResourceState.RENDER_TARGET)
        culled_pass._culled = True

        active_pass = GraphicsPass(name="Active")
        active_pass.write(handle, ResourceState.RENDER_TARGET)

        batches = bm.analyze_passes([culled_pass, active_pass])

        # Only active pass should generate barriers
        assert len(batches) == 1
        assert batches[0].before_pass == "Active"
