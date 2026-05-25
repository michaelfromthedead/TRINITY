"""
Whitebox tests for RHIContext protocol and MockContext implementation.

Tests the rendering context abstraction layer that mediates between the
frame graph and the underlying rendering hardware interface (RHI).

WHITEBOX coverage plan:
  - Path A: RHIContext Protocol structural conformance via runtime_checkable
  - Path B: MockContext satisfies RHIContext (isinstance check)
  - Path C: MockContext initial state -- all call lists empty, counters zero
  - Path D: execute_barriers records barriers, barrier_count sums across calls
  - Path E: allocate_transient records descriptor, returns AllocationHandle
  - Path F: begin_pass / end_pass recording with per-pass tracking
  - Path G: submit_queue records queue type and fences
  - Path H: assert_barriers_executed pass/fail at exact thresholds
  - Path I: assert_queue_submitted pass/fail with/without queue filter
  - Path J: assert_barrier_for_resource pass/fail via barrier.handle.name
  - Path K: assert_pass_sequence pass/fail ordering
  - Path L: reset clears all recorded state
  - Path M: Sequence[Barrier] accepts both list and tuple
  - Path N: FenceOp dataclass defaults and explicit construction
  - Path O: AllocationHandle type identity and arithmetic compatibility
  - Path P: Multiple interleaved call types preserved independently
  - Path Q: Empty barriers and empty fences edge cases
  - Path R: PassNode identity preserved through begin/end cycle
  - Path S: concurrent barrier counts from multiple execute_barriers calls
  - Path T: Barrier with subresource field preserved through MockContext
"""

from __future__ import annotations

from typing import Sequence

import pytest

from engine.rendering.framegraph import (
    AllocationHandle,
    Barrier,
    BarrierType,
    FenceOp,
    GraphicsPass,
    PassNode,
    QueueType,
    ResourceDescriptor,
    ResourceHandle,
    ResourceState,
)
from engine.rendering.framegraph.context import RHIContext
from tests.rendering.framegraph.mock_context import MockContext


# =============================================================================
# Test: FenceOp dataclass
# =============================================================================


class TestFenceOp:
    """Whitebox tests for FenceOp dataclass."""

    def test_default_construction(self):
        """Verify FenceOp defaults: operation='signal', fence_value=0, target_queue=GRAPHICS."""
        op = FenceOp()
        assert op.operation == "signal"
        assert op.fence_value == 0
        assert op.target_queue == QueueType.GRAPHICS

    def test_explicit_construction(self):
        """Verify FenceOp accepts explicit values for all fields."""
        op = FenceOp(
            operation="wait",
            fence_value=42,
            target_queue=QueueType.COMPUTE,
        )
        assert op.operation == "wait"
        assert op.fence_value == 42
        assert op.target_queue == QueueType.COMPUTE

    def test_signal_on_copy_queue(self):
        """Verify FenceOp with signal on COPY queue."""
        op = FenceOp(
            operation="signal",
            fence_value=7,
            target_queue=QueueType.COPY,
        )
        assert op.operation == "signal"
        assert op.fence_value == 7
        assert op.target_queue == QueueType.COPY

    def test_fence_value_is_int(self):
        """Verify fence_value is stored as int, not float."""
        op = FenceOp(fence_value=99)
        assert isinstance(op.fence_value, int)
        assert op.fence_value == 99

    def test_operation_string_type(self):
        """Verify operation is a plain string."""
        op = FenceOp(operation="signal")
        assert isinstance(op.operation, str)

    def test_immutable_defaults(self):
        """Verify default factory fields shared across instances do not alias."""
        op1 = FenceOp()
        op2 = FenceOp()
        assert op1 is not op2
        assert op1.fence_value == 0
        assert op2.fence_value == 0


# =============================================================================
# Test: AllocationHandle
# =============================================================================


class TestAllocationHandle:
    """Whitebox tests for AllocationHandle NewType."""

    def test_is_int(self):
        """AllocationHandle is an int-based type."""
        handle = AllocationHandle(1)
        assert isinstance(handle, int)
        assert handle == 1

    def test_arithmetic_compatible(self):
        """AllocationHandle supports int arithmetic."""
        a = AllocationHandle(5)
        b = a + 3
        assert b == 8
        assert isinstance(b, int)

    def test_zero_is_valid(self):
        """Zero is a valid AllocationHandle value."""
        handle = AllocationHandle(0)
        assert handle == 0
        assert isinstance(handle, int)

    def test_negative_is_valid(self):
        """Negative values are valid allocation handles (protocol doesn't forbid)."""
        handle = AllocationHandle(-1)
        assert handle == -1

    def test_large_value(self):
        """Large integer allocation handles are valid."""
        handle = AllocationHandle(2**31 - 1)
        assert handle == 2**31 - 1


# =============================================================================
# Test: Protocol Structural Conformance
# =============================================================================


class TestProtocolConformance:
    """Whitebox tests for RHIContext protocol structural conformance."""

    def test_protocol_is_runtime_checkable(self):
        """RHIContext is decorated with @runtime_checkable."""
        assert hasattr(RHIContext, "__instancecheck__"), (
            "RHIContext must be runtime_checkable for isinstance checks"
        )

    def test_mock_context_satisfies_protocol(self):
        """MockContext structurally satisfies the RHIContext protocol."""
        ctx = MockContext()
        assert isinstance(ctx, RHIContext), (
            "MockContext must be recognized as RHIContext via isinstance"
        )

    def test_protocol_has_required_methods(self):
        """Verify RHIContext protocol defines all 5 required methods."""
        method_names = {
            "execute_barriers",
            "allocate_transient",
            "begin_pass",
            "end_pass",
            "submit_queue",
        }
        protocol_methods = {
            name
            for name in dir(RHIContext)
            if not name.startswith("_")
        }
        for m in method_names:
            assert m in protocol_methods, (
                f"RHIContext missing required method: {m}"
            )

    def test_mock_context_has_all_protocol_methods(self):
        """MockContext implements all 5 RHIContext protocol methods."""
        for name in [
            "execute_barriers",
            "allocate_transient",
            "begin_pass",
            "end_pass",
            "submit_queue",
        ]:
            assert hasattr(MockContext, name), (
                f"MockContext missing method: {name}"
            )

    def test_protocol_methods_return_none_for_void_methods(self):
        """execute_barriers, begin_pass, end_pass return None."""
        ctx = MockContext()
        assert ctx.execute_barriers([]) is None
        assert ctx.begin_pass(ResourceHandle()) is None
        assert ctx.end_pass(ResourceHandle()) is None
        assert ctx.submit_queue(QueueType.GRAPHICS, []) is None

    def test_allocate_transient_returns_allocationhandle(self):
        """allocate_transient returns an AllocationHandle."""
        ctx = MockContext()
        desc = ResourceDescriptor(name="test")
        result = ctx.allocate_transient(desc)
        # AllocationHandle is a NewType(int), which is a function at runtime
        assert isinstance(result, int), f"Expected int (AllocationHandle), got {type(result)}"

    def test_mock_returns_handle_value_one(self):
        """MockContext always returns AllocationHandle(1)."""
        ctx = MockContext()
        desc1 = ResourceDescriptor(name="a")
        desc2 = ResourceDescriptor(name="b")
        assert ctx.allocate_transient(desc1) == AllocationHandle(1)
        assert ctx.allocate_transient(desc2) == AllocationHandle(1)


# =============================================================================
# Test: MockContext Initial State
# =============================================================================


class TestMockContextInitialState:
    """Whitebox tests for MockContext initial (empty) state."""

    def test_barrier_calls_empty(self):
        ctx = MockContext()
        assert ctx.barrier_calls == []
        assert ctx.barrier_count == 0
        assert ctx.barrier_batch_count == 0

    def test_alloc_calls_empty(self):
        ctx = MockContext()
        assert ctx.alloc_calls == []

    def test_pass_begin_and_end_empty(self):
        ctx = MockContext()
        assert ctx.pass_begin_calls == []
        assert ctx.pass_end_calls == []
        assert ctx.pass_begin_count == 0
        assert ctx.pass_end_count == 0

    def test_submit_calls_empty(self):
        ctx = MockContext()
        assert ctx.submit_calls == []
        assert ctx.submit_count == 0

    def test_barrier_calls_list_is_independent(self):
        """Each MockContext gets its own barrier_calls list (no shared default)."""
        ctx1 = MockContext()
        ctx2 = MockContext()
        ctx1.execute_barriers([])
        assert ctx1.barrier_batch_count == 1
        assert ctx2.barrier_batch_count == 0


# =============================================================================
# Test: execute_barriers
# =============================================================================


class TestExecuteBarriers:
    """Whitebox tests for execute_barriers recording."""

    def test_single_barrier(self):
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])

        assert ctx.barrier_batch_count == 1
        assert ctx.barrier_count == 1
        assert ctx.barrier_calls[0][0] is barrier

    def test_multiple_barriers_one_call(self):
        ctx = MockContext()
        barriers = [
            Barrier(
                handle=ResourceHandle(),
                barrier_type=BarrierType.TRANSITION,
                old_state=ResourceState.UNDEFINED,
                new_state=ResourceState.RENDER_TARGET,
            ),
            Barrier(
                handle=ResourceHandle(),
                barrier_type=BarrierType.UAV,
                old_state=ResourceState.UNORDERED_ACCESS,
                new_state=ResourceState.UNORDERED_ACCESS,
            ),
        ]
        ctx.execute_barriers(barriers)

        assert ctx.barrier_batch_count == 1
        assert ctx.barrier_count == 2

    def test_empty_barriers(self):
        ctx = MockContext()
        ctx.execute_barriers([])

        assert ctx.barrier_batch_count == 1
        assert ctx.barrier_count == 0

    def test_multiple_calls_accumulate(self):
        ctx = MockContext()
        handle = ResourceHandle()
        b1 = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                     old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        b2 = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                     old_state=ResourceState.RENDER_TARGET, new_state=ResourceState.SHADER_RESOURCE)

        ctx.execute_barriers([b1])
        ctx.execute_barriers([b2])

        assert ctx.barrier_batch_count == 2
        assert ctx.barrier_count == 2

    def test_sequence_accepts_tuple(self):
        """execute_barriers accepts any Sequence, including tuple."""
        ctx = MockContext()
        handle = ResourceHandle()
        barriers = (
            Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET),
        )
        ctx.execute_barriers(barriers)

        assert ctx.barrier_batch_count == 1
        assert ctx.barrier_count == 1

    def test_barrier_fields_preserved(self):
        """All fields of the original barrier are preserved through recording."""
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.RENDER_TARGET,
            new_state=ResourceState.SHADER_RESOURCE,
            subresource=2,
        )
        ctx.execute_barriers([barrier])

        recorded = ctx.barrier_calls[0][0]
        assert recorded.handle is handle
        assert recorded.barrier_type == BarrierType.TRANSITION
        assert recorded.old_state == ResourceState.RENDER_TARGET
        assert recorded.new_state == ResourceState.SHADER_RESOURCE
        assert recorded.subresource == 2

    def test_uav_barrier_type_preserved(self):
        """UAV barrier type is recorded correctly."""
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.UAV,
            old_state=ResourceState.UNORDERED_ACCESS,
            new_state=ResourceState.UNORDERED_ACCESS,
        )
        ctx.execute_barriers([barrier])
        recorded = ctx.barrier_calls[0][0]
        assert recorded.barrier_type == BarrierType.UAV

    def test_aliasing_barrier_type_preserved(self):
        """ALIASING barrier type is recorded correctly."""
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.ALIASING,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.UNDEFINED,
        )
        ctx.execute_barriers([barrier])
        recorded = ctx.barrier_calls[0][0]
        assert recorded.barrier_type == BarrierType.ALIASING


# =============================================================================
# Test: allocate_transient
# =============================================================================


class TestAllocateTransient:
    """Whitebox tests for allocate_transient recording."""

    def test_single_allocation(self):
        ctx = MockContext()
        desc = ResourceDescriptor(name="shadow_map", width=1024, height=1024)
        result = ctx.allocate_transient(desc)

        assert len(ctx.alloc_calls) == 1
        assert ctx.alloc_calls[0] is desc
        assert result == AllocationHandle(1)

    def test_multiple_allocations_accumulate(self):
        ctx = MockContext()
        desc_a = ResourceDescriptor(name="albedo")
        desc_b = ResourceDescriptor(name="normal")

        ctx.allocate_transient(desc_a)
        ctx.allocate_transient(desc_b)

        assert len(ctx.alloc_calls) == 2
        assert ctx.alloc_calls[0] is desc_a
        assert ctx.alloc_calls[1] is desc_b

    def test_descriptor_fields_preserved(self):
        """ResourceDescriptor fields are accessible through recorded call."""
        ctx = MockContext()
        desc = ResourceDescriptor(
            name="hdr_buffer",
            resource_type=ResourceDescriptor.__dataclass_fields__["resource_type"].default,
            width=1920,
            height=1080,
            format=ResourceDescriptor.__dataclass_fields__["format"].default,
        )
        ctx.allocate_transient(desc)

        recorded = ctx.alloc_calls[0]
        assert recorded.name == "hdr_buffer"
        assert recorded.width == 1920
        assert recorded.height == 1080

    def test_buffer_descriptor_allocated(self):
        """Buffer-style descriptors (not textures) are recorded."""
        ctx = MockContext()
        desc = ResourceDescriptor(
            name="vertex_buffer",
            is_texture=False,
            buffer_size=65536,
        )
        ctx.allocate_transient(desc)

        recorded = ctx.alloc_calls[0]
        assert recorded.is_texture is False
        assert recorded.buffer_size == 65536


# =============================================================================
# Test: begin_pass / end_pass
# =============================================================================


class TestBeginEndPass:
    """Whitebox tests for begin_pass and end_pass recording."""

    def test_begin_records_pass(self):
        ctx = MockContext()
        node = GraphicsPass(name="GBuffer")
        ctx.begin_pass(node)

        assert ctx.pass_begin_count == 1
        assert ctx.pass_begin_calls[0] is node

    def test_end_records_pass(self):
        ctx = MockContext()
        node = GraphicsPass(name="Lighting")
        ctx.end_pass(node)

        assert ctx.pass_end_count == 1
        assert ctx.pass_end_calls[0] is node

    def test_begin_end_sequence(self):
        ctx = MockContext()
        a = GraphicsPass(name="PassA")
        b = GraphicsPass(name="PassB")

        ctx.begin_pass(a)
        ctx.end_pass(a)
        ctx.begin_pass(b)
        ctx.end_pass(b)

        assert ctx.pass_begin_count == 2
        assert ctx.pass_end_count == 2
        assert ctx.pass_begin_calls[0] is a
        assert ctx.pass_begin_calls[1] is b
        assert ctx.pass_end_calls[0] is a
        assert ctx.pass_end_calls[1] is b

    def test_begin_without_end(self):
        """begin_pass recorded even if end_pass not called."""
        ctx = MockContext()
        node = GraphicsPass(name="Orphaned")
        ctx.begin_pass(node)

        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 0

    def test_multiple_begin_same_pass(self):
        """begin_pass can be called multiple times with same pass."""
        ctx = MockContext()
        node = GraphicsPass(name="Repeated")
        ctx.begin_pass(node)
        ctx.begin_pass(node)

        assert ctx.pass_begin_count == 2
        assert ctx.pass_begin_calls[0] is node
        assert ctx.pass_begin_calls[1] is node

    def test_pass_node_name_preserved(self):
        """PassNode name is accessible through recorded call."""
        ctx = MockContext()
        node = GraphicsPass(name="ShadowMap")
        ctx.begin_pass(node)

        assert ctx.pass_begin_calls[0].name == "ShadowMap"

    def test_pass_node_identity_preserved(self):
        """The exact PassNode object identity is preserved through begin/end cycle."""
        ctx = MockContext()
        node = GraphicsPass(name="IdentityTest")
        ctx.begin_pass(node)
        ctx.end_pass(node)

        assert ctx.pass_begin_calls[0] is ctx.pass_end_calls[0] is node


# =============================================================================
# Test: submit_queue
# =============================================================================


class TestSubmitQueue:
    """Whitebox tests for submit_queue recording."""

    def test_submit_queue_graphics(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])

        assert ctx.submit_count == 1
        queue, fences = ctx.submit_calls[0]
        assert queue == QueueType.GRAPHICS
        assert fences == []

    def test_submit_queue_compute(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.COMPUTE, [])

        queue, _ = ctx.submit_calls[0]
        assert queue == QueueType.COMPUTE

    def test_submit_queue_copy(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.COPY, [])

        queue, _ = ctx.submit_calls[0]
        assert queue == QueueType.COPY

    def test_submit_with_fences(self):
        ctx = MockContext()
        fences = [
            FenceOp(operation="signal", fence_value=1),
            FenceOp(operation="wait", fence_value=2),
        ]
        ctx.submit_queue(QueueType.GRAPHICS, fences)

        queue, recorded_fences = ctx.submit_calls[0]
        assert queue == QueueType.GRAPHICS
        assert len(recorded_fences) == 2
        assert recorded_fences[0].operation == "signal"
        assert recorded_fences[0].fence_value == 1
        assert recorded_fences[1].operation == "wait"
        assert recorded_fences[1].fence_value == 2

    def test_submit_fence_queue_preserved(self):
        """FenceOp.target_queue is preserved through submission."""
        ctx = MockContext()
        fence = FenceOp(
            operation="signal",
            fence_value=5,
            target_queue=QueueType.COMPUTE,
        )
        ctx.submit_queue(QueueType.GRAPHICS, [fence])

        _, recorded_fences = ctx.submit_calls[0]
        assert recorded_fences[0].target_queue == QueueType.COMPUTE

    def test_multiple_submits_accumulate(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.submit_queue(QueueType.COMPUTE, [])
        ctx.submit_queue(QueueType.COPY, [])

        assert ctx.submit_count == 3

    def test_submit_empty_fences(self):
        """submit_queue with empty fence list is recorded."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])

        _, fences = ctx.submit_calls[0]
        assert fences == []

    def test_submit_sequence_accepts_tuple(self):
        """submit_queue accepts tuple of fences."""
        ctx = MockContext()
        fences = (FenceOp(operation="signal", fence_value=10),)
        ctx.submit_queue(QueueType.GRAPHICS, fences)

        assert ctx.submit_count == 1
        _, recorded = ctx.submit_calls[0]
        assert len(recorded) == 1
        assert recorded[0].fence_value == 10


# =============================================================================
# Test: Helper assertions
# =============================================================================


class TestAssertBarriersExecuted:
    """Whitebox tests for assert_barriers_executed helper."""

    def test_passes_when_barriers_present(self):
        ctx = MockContext()
        ctx.execute_barriers([
            Barrier(
                handle=ResourceHandle(),
                barrier_type=BarrierType.TRANSITION,
                old_state=ResourceState.UNDEFINED,
                new_state=ResourceState.RENDER_TARGET,
            )
        ])
        ctx.assert_barriers_executed()

    def test_passes_with_min_count(self):
        ctx = MockContext()
        h = ResourceHandle()
        for _ in range(5):
            ctx.execute_barriers([
                Barrier(handle=h, barrier_type=BarrierType.TRANSITION,
                        old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
            ])
        ctx.assert_barriers_executed(min_count=5)

    def test_fails_when_no_barriers(self):
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_barriers_executed()

    def test_fails_below_min_count(self):
        ctx = MockContext()
        h = ResourceHandle()
        ctx.execute_barriers([
            Barrier(handle=h, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ])
        with pytest.raises(AssertionError):
            ctx.assert_barriers_executed(min_count=5)

    def test_failure_message_mentions_count(self):
        ctx = MockContext()
        try:
            ctx.assert_barriers_executed()
        except AssertionError as e:
            msg = str(e)
            assert "0" in msg or "barrier" in msg


class TestAssertQueueSubmitted:
    """Whitebox tests for assert_queue_submitted helper."""

    def test_passes_when_submitted(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.assert_queue_submitted()

    def test_passes_with_queue_filter(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.COMPUTE, [])
        ctx.assert_queue_submitted(queue=QueueType.COMPUTE)

    def test_passes_with_min_count_on_specific_queue(self):
        ctx = MockContext()
        for _ in range(3):
            ctx.submit_queue(QueueType.COMPUTE, [])
        ctx.assert_queue_submitted(queue=QueueType.COMPUTE, min_count=3)

    def test_fails_wrong_queue(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted(queue=QueueType.COMPUTE)

    def test_fails_too_few_on_queue(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.COMPUTE, [])
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted(queue=QueueType.COMPUTE, min_count=5)

    def test_fails_no_submissions(self):
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted()

    def test_submits_on_multiple_queues_filtered(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.submit_queue(QueueType.COMPUTE, [])
        ctx.submit_queue(QueueType.COPY, [])
        ctx.submit_queue(QueueType.GRAPHICS, [])

        ctx.assert_queue_submitted(queue=QueueType.GRAPHICS, min_count=2)
        ctx.assert_queue_submitted(queue=QueueType.COMPUTE, min_count=1)
        ctx.assert_queue_submitted(queue=QueueType.COPY, min_count=1)


class TestAssertBarrierForResource:
    """Whitebox tests for assert_barrier_for_resource helper."""

    def test_passes_when_resource_barriered(self):
        ctx = MockContext()
        handle = ResourceHandle()
        handle.descriptor = ResourceDescriptor(name="albedo")
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])
        ctx.assert_barrier_for_resource("albedo")

    def test_fails_when_resource_not_barriered(self):
        ctx = MockContext()
        handle = ResourceHandle()
        handle.descriptor = ResourceDescriptor(name="normal")
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])
        with pytest.raises(AssertionError):
            ctx.assert_barrier_for_resource("nonexistent")

    def test_fails_with_no_barriers(self):
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_barrier_for_resource("anything")

    def test_searches_all_batches(self):
        """assert_barrier_for_resource searches across all barrier batches."""
        ctx = MockContext()
        h1 = ResourceHandle()
        h1.descriptor = ResourceDescriptor(name="depth")
        h2 = ResourceHandle()
        h2.descriptor = ResourceDescriptor(name="shadow")

        ctx.execute_barriers([
            Barrier(handle=h1, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.DEPTH_WRITE)
        ])
        ctx.execute_barriers([
            Barrier(handle=h2, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.SHADER_RESOURCE)
        ])

        ctx.assert_barrier_for_resource("depth")
        ctx.assert_barrier_for_resource("shadow")


class TestAssertPassSequence:
    """Whitebox tests for assert_pass_sequence helper."""

    def test_passes_exact_sequence(self):
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))
        ctx.begin_pass(GraphicsPass(name="B"))
        ctx.begin_pass(GraphicsPass(name="C"))

        ctx.assert_pass_sequence(["A", "B", "C"])

    def test_fails_wrong_order(self):
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="B"))
        ctx.begin_pass(GraphicsPass(name="A"))

        with pytest.raises(AssertionError):
            ctx.assert_pass_sequence(["A", "B"])

    def test_fails_short_sequence(self):
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))

        with pytest.raises(AssertionError):
            ctx.assert_pass_sequence(["A", "B"])

    def test_fails_no_passes(self):
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_pass_sequence(["A"])

    def test_empty_sequence_passes_no_passes(self):
        ctx = MockContext()
        ctx.assert_pass_sequence([])

    def test_failure_message_lists_actual(self):
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="ActualPass"))
        try:
            ctx.assert_pass_sequence(["ExpectedPass"])
        except AssertionError as e:
            msg = str(e)
            assert "ActualPass" in msg
            assert "ExpectedPass" in msg


# =============================================================================
# Test: reset
# =============================================================================


class TestReset:
    """Whitebox tests for MockContext.reset()."""

    def test_reset_clears_barriers(self):
        ctx = MockContext()
        ctx.execute_barriers([
            Barrier(handle=ResourceHandle(), barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ])
        ctx.reset()
        assert ctx.barrier_calls == []
        assert ctx.barrier_count == 0
        assert ctx.barrier_batch_count == 0

    def test_reset_clears_allocations(self):
        ctx = MockContext()
        ctx.allocate_transient(ResourceDescriptor(name="test"))
        ctx.reset()
        assert ctx.alloc_calls == []

    def test_reset_clears_pass_calls(self):
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))
        ctx.end_pass(GraphicsPass(name="A"))
        ctx.reset()
        assert ctx.pass_begin_calls == []
        assert ctx.pass_end_calls == []
        assert ctx.pass_begin_count == 0
        assert ctx.pass_end_count == 0

    def test_reset_clears_submits(self):
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.reset()
        assert ctx.submit_calls == []
        assert ctx.submit_count == 0

    def test_reset_allows_reuse(self):
        """After reset, MockContext can be reused for a new frame."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.reset()

        ctx.submit_queue(QueueType.COMPUTE, [])
        assert ctx.submit_count == 1
        queue, _ = ctx.submit_calls[0]
        assert queue == QueueType.COMPUTE

    def test_reset_idempotent(self):
        """Calling reset twice is safe."""
        ctx = MockContext()
        ctx.reset()
        ctx.reset()  # should not raise
        assert ctx.barrier_calls == []
        assert ctx.alloc_calls == []
        assert ctx.pass_begin_calls == []
        assert ctx.pass_end_calls == []
        assert ctx.submit_calls == []


# =============================================================================
# Test: Interleaved calls (integration across all methods)
# =============================================================================


class TestInterleavedCalls:
    """Whitebox tests for interleaved calls across all MockContext methods."""

    def test_all_methods_independent(self):
        """Each recording list is independent and does not interfere."""
        ctx = MockContext()
        h = ResourceHandle()
        h.descriptor = ResourceDescriptor(name="tex")

        barrier = Barrier(handle=h, barrier_type=BarrierType.TRANSITION,
                          old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([barrier])
        ctx.allocate_transient(ResourceDescriptor(name="output"))
        ctx.begin_pass(GraphicsPass(name="Render"))
        ctx.end_pass(GraphicsPass(name="Render"))
        ctx.submit_queue(QueueType.GRAPHICS, [])

        assert ctx.barrier_count == 1
        assert len(ctx.alloc_calls) == 1
        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 1
        assert ctx.submit_count == 1

    def test_frame_lifecycle_sequence(self):
        """Simulate a full frame lifecycle: alloc, barriers, passes, submit."""
        ctx = MockContext()

        # Frame start: allocate resources
        albedo = ResourceDescriptor(name="albedo", width=1920, height=1080)
        depth = ResourceDescriptor(name="depth", width=1920, height=1080,
                                   format=ResourceDescriptor.__dataclass_fields__["format"].default)
        ctx.allocate_transient(albedo)
        ctx.allocate_transient(depth)

        # GBuffer pass
        h_albedo = ResourceHandle()
        h_albedo.descriptor = ResourceDescriptor(name="albedo")
        h_depth = ResourceHandle()
        h_depth.descriptor = ResourceDescriptor(name="depth")

        gbuffer = GraphicsPass(name="GBuffer")
        ctx.begin_pass(gbuffer)
        ctx.execute_barriers([
            Barrier(handle=h_albedo, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET),
            Barrier(handle=h_depth, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.DEPTH_WRITE),
        ])
        ctx.end_pass(gbuffer)

        # Lighting pass
        lighting = GraphicsPass(name="Lighting")
        ctx.begin_pass(lighting)
        ctx.execute_barriers([
            Barrier(handle=h_albedo, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.RENDER_TARGET, new_state=ResourceState.SHADER_RESOURCE),
            Barrier(handle=h_depth, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.DEPTH_WRITE, new_state=ResourceState.SHADER_RESOURCE),
        ])
        ctx.end_pass(lighting)

        # Submit
        ctx.submit_queue(QueueType.GRAPHICS, [
            FenceOp(operation="signal", fence_value=1),
        ])

        # Verify
        assert len(ctx.alloc_calls) == 2
        assert ctx.pass_begin_count == 2
        assert ctx.pass_end_count == 2
        assert ctx.barrier_batch_count == 2
        assert ctx.barrier_count == 4
        assert ctx.submit_count == 1

        ctx.assert_pass_sequence(["GBuffer", "Lighting"])
        ctx.assert_barriers_executed(min_count=4)
        ctx.assert_queue_submitted(min_count=1)

    def test_mixed_queue_submissions(self):
        """Multiple queue submissions with different types."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [
            FenceOp(operation="signal", fence_value=1),
        ])
        ctx.submit_queue(QueueType.COMPUTE, [
            FenceOp(operation="signal", fence_value=2),
            FenceOp(operation="wait", fence_value=1),
        ])
        ctx.submit_queue(QueueType.COPY, [])

        assert ctx.submit_count == 3
        assert ctx.submit_calls[0][0] == QueueType.GRAPHICS
        assert ctx.submit_calls[1][0] == QueueType.COMPUTE
        assert ctx.submit_calls[2][0] == QueueType.COPY


# =============================================================================
# Test: MockContext as a protocol consumer (how frame_graph uses it)
# =============================================================================


class TestMockContextAsProtocol:
    """Tests that validate MockContext works where RHIContext is expected."""

    def test_accepts_mock_context_as_rhi_context(self):
        """Functions expecting RHIContext accept MockContext."""
        def execute_frame(ctx: RHIContext) -> list[str]:
            actions = []
            ctx.execute_barriers([])
            actions.append("barriers")
            ctx.begin_pass(GraphicsPass(name="Test"))
            actions.append("begin")
            ctx.end_pass(GraphicsPass(name="Test"))
            actions.append("end")
            ctx.submit_queue(QueueType.GRAPHICS, [])
            actions.append("submit")
            return actions

        ctx = MockContext()
        result = execute_frame(ctx)
        assert result == ["barriers", "begin", "end", "submit"]

    def test_type_annotation_accepts_mock_context(self):
        """MockContext passes isinstance check for RHIContext."""
        ctx = MockContext()
        assert isinstance(ctx, RHIContext)

    def test_can_be_stored_in_rhi_context_typed_variable(self):
        """MockContext can be assigned to a variable typed as RHIContext."""
        ctx: RHIContext = MockContext()
        assert ctx is not None
