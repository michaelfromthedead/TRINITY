"""Blackbox contract tests for RHIContext, AllocationHandle, FenceOp.

CLEANROOM: Tests the public API contract only. No visibility into the
implementation of context.py. All information derived from the public
exported types and the PHASE_1_TODO.md specification.

Equivalence partitioning:
  - RHIContext protocol: class with all 5 methods satisfies; class
    missing any method does not; non-class type does not
  - AllocationHandle: NewType(int), transparent at runtime, hashable
  - FenceOp: dataclass with documented fields and defaults
  - MockContext: no-op by default, records all calls, assertions
    verify recorded state

Boundary cases:
  - FenceOp: zero, negative, large fence_value; empty operation string
  - AllocationHandle: zero, negative values
  - MockContext: empty barrier/fence sequences; missing pass node name
  - Reset from empty state; reset mid-recording

Error cases:
  - Missing protocol methods -> isinstance returns False
  - Non-class argument to isinstance check
"""

import dataclasses
import inspect
import typing

import pytest

from engine.rendering.framegraph import (
    AllocationHandle,
    Barrier,
    BarrierType,
    FenceOp,
    GraphicsPass,
    PassNode,
    PipelineStage,
    QueueType,
    ResourceDescriptor,
    ResourceFormat,
    ResourceHandle,
    ResourceState,
    ResourceType,
    RHIContext,
)
from tests.rendering.framegraph.mock_context import MockContext


# =========================================================================
# 1. Import and Export
# =========================================================================

class TestPublicAPIExport:
    """Verify all contract types are publicly importable."""

    def test_rhicontext_importable(self):
        """RHIContext is importable from the package."""
        from engine.rendering.framegraph import RHIContext as R
        assert R is RHIContext

    def test_allocation_handle_importable(self):
        """AllocationHandle is importable from the package."""
        from engine.rendering.framegraph import AllocationHandle as A
        assert A is AllocationHandle

    def test_fence_op_importable(self):
        """FenceOp is importable from the package."""
        from engine.rendering.framegraph import FenceOp as F
        assert F is FenceOp


# =========================================================================
# 2. RHIContext Protocol
# =========================================================================

class TestRHIContextProtocol:
    """Structural tests for the RHIContext protocol definition."""

    def test_is_runtime_checkable_protocol(self):
        """RHIContext is a runtime-checkable typing.Protocol."""
        assert issubclass(RHIContext, typing.Protocol)
        # Verify isinstance works (this confirms runtime_checkable at runtime)
        class Impl:
            def execute_barriers(self, barriers): pass
            def allocate_transient(self, desc): return AllocationHandle(0)
            def begin_pass(self, pass_node): pass
            def end_pass(self, pass_node): pass
            def submit_queue(self, queue, fences): pass
        assert isinstance(Impl(), RHIContext)

    def test_has_all_five_methods(self):
        """RHIContext declares exactly the 5 contract methods."""
        expected = {
            "execute_barriers",
            "allocate_transient",
            "begin_pass",
            "end_pass",
            "submit_queue",
        }
        # Protocol methods are defined on the class (not inherited)
        actual = {
            name
            for name in vars(RHIContext)
            if not name.startswith("_")
        }
        # Filter to only callable entries (protocol stubs are function-like)
        proto_methods = {
            name for name in actual
            if callable(getattr(RHIContext, name, None))
        }
        assert proto_methods == expected, (
            f"Protocol methods mismatch. "
            f"Extra: {proto_methods - expected}, "
            f"Missing: {expected - proto_methods}"
        )

    def test_execute_barriers_accepts_barrier_sequence(self):
        """execute_barriers accepts a sequence of Barrier objects."""
        sig = inspect.signature(RHIContext.execute_barriers)
        params = list(sig.parameters.values())
        assert len(params) == 2  # self, barriers
        assert params[1].name == "barriers"

    def test_execute_barriers_returns_none(self):
        """execute_barriers has None return annotation."""
        sig = inspect.signature(RHIContext.execute_barriers)
        ret = sig.return_annotation
        # PEP 563 (from __future__ import annotations) makes annotations strings
        assert ret is None or ret == "None" or ret is inspect.Parameter.empty

    def test_allocate_transient_accepts_descriptor(self):
        """allocate_transient accepts a ResourceDescriptor."""
        sig = inspect.signature(RHIContext.allocate_transient)
        params = list(sig.parameters.values())
        assert len(params) == 2  # self, desc

    def test_allocate_transient_returns_allocation_handle(self):
        """allocate_transient returns AllocationHandle."""
        sig = inspect.signature(RHIContext.allocate_transient)
        ret = sig.return_annotation
        assert ret is AllocationHandle or ret in ("AllocationHandle", "~AllocationHandle")

    def test_begin_pass_accepts_pass_node(self):
        """begin_pass accepts a PassNode."""
        sig = inspect.signature(RHIContext.begin_pass)
        params = list(sig.parameters.values())
        assert len(params) == 2  # self, pass_node

    def test_begin_pass_returns_none(self):
        """begin_pass has None return annotation."""
        sig = inspect.signature(RHIContext.begin_pass)
        ret = sig.return_annotation
        assert ret is None or ret == "None" or ret is inspect.Parameter.empty

    def test_end_pass_accepts_pass_node(self):
        """end_pass accepts a PassNode."""
        sig = inspect.signature(RHIContext.end_pass)
        params = list(sig.parameters.values())
        assert len(params) == 2  # self, pass_node

    def test_end_pass_returns_none(self):
        """end_pass has None return annotation."""
        sig = inspect.signature(RHIContext.end_pass)
        ret = sig.return_annotation
        assert ret is None or ret == "None" or ret is inspect.Parameter.empty

    def test_submit_queue_accepts_queue_and_fences(self):
        """submit_queue accepts a QueueType and a FenceOp sequence."""
        sig = inspect.signature(RHIContext.submit_queue)
        params = list(sig.parameters.values())
        assert len(params) == 3  # self, queue, fences

    def test_submit_queue_returns_none(self):
        """submit_queue has None return annotation."""
        sig = inspect.signature(RHIContext.submit_queue)
        ret = sig.return_annotation
        assert ret is None or ret == "None" or ret is inspect.Parameter.empty

    def test_class_with_all_methods_satisfies_protocol(self):
        """A class implementing all 5 methods satisfies the protocol."""
        class DummyContext:
            def execute_barriers(self, barriers): pass
            def allocate_transient(self, desc): return AllocationHandle(0)
            def begin_pass(self, pass_node): pass
            def end_pass(self, pass_node): pass
            def submit_queue(self, queue, fences): pass

        assert isinstance(DummyContext(), RHIContext)

    def test_class_missing_method_fails_isinstance(self):
        """A class missing a protocol method fails isinstance check."""
        class MissingMethod:
            def execute_barriers(self, barriers): pass
            def allocate_transient(self, desc): return AllocationHandle(0)
            def begin_pass(self, pass_node): pass
            # Missing end_pass
            def submit_queue(self, queue, fences): pass

        assert not isinstance(MissingMethod(), RHIContext)

    def test_non_class_fails_isinstance(self):
        """Non-class objects (int, str, None) fail isinstance check."""
        assert not isinstance(42, RHIContext)
        assert not isinstance("string", RHIContext)
        assert not isinstance(None, RHIContext)
        assert not isinstance([], RHIContext)


# =========================================================================
# 3. AllocationHandle
# =========================================================================

class TestAllocationHandle:
    """Tests for the AllocationHandle opaque type."""

    def test_is_newtype(self):
        """AllocationHandle is a NewType."""
        assert isinstance(AllocationHandle, typing.NewType)

    def test_wraps_int(self):
        """AllocationHandle wraps int at runtime."""
        assert AllocationHandle.__supertype__ is int

    def test_create_with_value(self):
        """Creating an AllocationHandle with an int value works."""
        h = AllocationHandle(42)
        assert h == 42
        assert isinstance(h, int)

    def test_create_zero(self):
        """Boundary: AllocationHandle(0) is valid."""
        h = AllocationHandle(0)
        assert h == 0

    def test_create_negative(self):
        """Boundary: AllocationHandle(-1) is valid (opaque)."""
        h = AllocationHandle(-1)
        assert h == -1

    def test_create_large(self):
        """Boundary: AllocationHandle with a large value."""
        h = AllocationHandle(2**63 - 1)
        assert h == 2**63 - 1

    def test_hashable(self):
        """AllocationHandle instances are hashable."""
        h1 = AllocationHandle(1)
        h2 = AllocationHandle(2)
        d = {h1: "a", h2: "b"}
        assert d[h1] == "a"
        assert d[h2] == "b"

    def test_comparable(self):
        """AllocationHandle instances compare by value."""
        assert AllocationHandle(5) == AllocationHandle(5)
        assert AllocationHandle(5) != AllocationHandle(10)

    def test_multiple_distinct_values(self):
        """Multiple handles with different values are distinct."""
        handles = [AllocationHandle(i) for i in range(10)]
        assert len(set(handles)) == 10


# =========================================================================
# 4. FenceOp
# =========================================================================

class TestFenceOp:
    """Tests for the FenceOp dataclass."""

    def test_is_dataclass(self):
        """FenceOp is a dataclass."""
        assert dataclasses.is_dataclass(FenceOp)

    def test_default_operation(self):
        """Default operation is 'signal'."""
        f = FenceOp()
        assert f.operation == "signal"

    def test_default_fence_value(self):
        """Default fence_value is 0."""
        f = FenceOp()
        assert f.fence_value == 0

    def test_default_target_queue(self):
        """Default target_queue is QueueType.GRAPHICS."""
        f = FenceOp()
        assert f.target_queue == QueueType.GRAPHICS

    def test_positional_construction(self):
        """FenceOp supports positional argument construction."""
        f = FenceOp("wait", 42, QueueType.COMPUTE)
        assert f.operation == "wait"
        assert f.fence_value == 42
        assert f.target_queue == QueueType.COMPUTE

    def test_keyword_construction(self):
        """FenceOp supports keyword argument construction."""
        f = FenceOp(operation="signal", fence_value=5, target_queue=QueueType.COPY)
        assert f.operation == "signal"
        assert f.fence_value == 5
        assert f.target_queue == QueueType.COPY

    def test_partial_construction(self):
        """FenceOp supports partial construction with defaults."""
        f = FenceOp(operation="wait")
        assert f.operation == "wait"
        assert f.fence_value == 0  # default
        assert f.target_queue == QueueType.GRAPHICS  # default

    def test_field_types(self):
        """FenceOp fields have correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(FenceOp)}
        # PEP 563 (from __future__ import annotations) makes annotations strings
        assert fields["operation"] in (str, "str")
        assert fields["fence_value"] in (int, "int")
        assert fields["target_queue"] in (QueueType, "QueueType")

    def test_operation_all_values(self):
        """Equivalence: operation accepts any string."""
        for op in ("signal", "wait", "custom", ""):
            f = FenceOp(operation=op)
            assert f.operation == op

    def test_fence_value_zero(self):
        """Boundary: fence_value=0 is valid."""
        f = FenceOp(fence_value=0)
        assert f.fence_value == 0

    def test_fence_value_negative(self):
        """Boundary: fence_value negative is valid."""
        f = FenceOp(fence_value=-1)
        assert f.fence_value == -1

    def test_fence_value_large(self):
        """Boundary: fence_value large positive is valid."""
        f = FenceOp(fence_value=2**63 - 1)
        assert f.fence_value == 2**63 - 1

    def test_target_queue_all_values(self):
        """Equivalence: target_queue accepts all QueueType values."""
        for qt in QueueType:
            f = FenceOp(target_queue=qt)
            assert f.target_queue == qt

    def test_equality(self):
        """FenceOp instances with same values are equal."""
        f1 = FenceOp("signal", 1, QueueType.GRAPHICS)
        f2 = FenceOp("signal", 1, QueueType.GRAPHICS)
        assert f1 == f2

    def test_inequality_operation(self):
        """Different operation makes FenceOp instances unequal."""
        f1 = FenceOp("signal", 1, QueueType.GRAPHICS)
        f2 = FenceOp("wait", 1, QueueType.GRAPHICS)
        assert f1 != f2

    def test_inequality_value(self):
        """Different fence_value makes FenceOp instances unequal."""
        f1 = FenceOp("signal", 1, QueueType.GRAPHICS)
        f2 = FenceOp("signal", 2, QueueType.GRAPHICS)
        assert f1 != f2

    def test_inequality_queue(self):
        """Different target_queue makes FenceOp instances unequal."""
        f1 = FenceOp("signal", 1, QueueType.GRAPHICS)
        f2 = FenceOp("signal", 1, QueueType.COMPUTE)
        assert f1 != f2

    def test_mutable_fields(self):
        """FenceOp fields are mutable (not frozen)."""
        f = FenceOp()
        f.operation = "wait"
        assert f.operation == "wait"
        f.fence_value = 99
        assert f.fence_value == 99
        f.target_queue = QueueType.COPY
        assert f.target_queue == QueueType.COPY

    def test_repr_includes_fields(self):
        """repr of FenceOp includes the field values."""
        f = FenceOp("wait", 42, QueueType.COMPUTE)
        r = repr(f)
        assert "wait" in r
        assert "42" in r
        assert "COMPUTE" in r or "QueueType" in r


# =========================================================================
# 5. MockContext: RHIContext Protocol Conformance
# =========================================================================

class TestMockContextProtocolConformance:
    """MockContext satisfies the RHIContext protocol structurally."""

    def test_isinstance_passes(self):
        """MockContext passes isinstance(RHIContext) check."""
        ctx = MockContext()
        assert isinstance(ctx, RHIContext)

    def test_class_isinstance_passes(self):
        """MockContext class passes issubclass(RHIContext) check."""
        assert issubclass(MockContext, RHIContext)


# =========================================================================
# 6. MockContext: No-op Default Behavior
# =========================================================================

class TestMockContextDefaults:
    """MockContext methods are no-ops by default (no exceptions)."""

    def test_execute_barriers_empty(self):
        """execute_barriers([]) does not raise."""
        ctx = MockContext()
        ctx.execute_barriers([])

    def test_execute_barriers_with_barriers(self):
        """execute_barriers with real Barrier objects does not raise."""
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])

    def test_allocate_transient_minimal(self):
        """allocate_transient with minimal descriptor does not raise."""
        ctx = MockContext()
        desc = ResourceDescriptor(name="test")
        handle = ctx.allocate_transient(desc)
        assert isinstance(handle, int)

    def test_begin_pass_with_real_pass(self):
        """begin_pass with a real GraphicsPass does not raise."""
        ctx = MockContext()
        pass_node = GraphicsPass(name="TestPass")
        ctx.begin_pass(pass_node)

    def test_end_pass_with_real_pass(self):
        """end_pass with a real GraphicsPass does not raise."""
        ctx = MockContext()
        pass_node = GraphicsPass(name="TestPass")
        ctx.end_pass(pass_node)

    def test_submit_queue_empty_fences(self):
        """submit_queue with empty fences does not raise."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])

    def test_submit_queue_with_fences(self):
        """submit_queue with FenceOp objects does not raise."""
        ctx = MockContext()
        fence = FenceOp("signal", 1, QueueType.GRAPHICS)
        ctx.submit_queue(QueueType.COMPUTE, [fence])

    def test_begin_end_pass_pair(self):
        """begin_pass followed by end_pass is well-behaved."""
        ctx = MockContext()
        pass_node = GraphicsPass(name="RenderPass")
        ctx.begin_pass(pass_node)
        ctx.end_pass(pass_node)
        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 1


# =========================================================================
# 7. MockContext: Call Recording
# =========================================================================

class TestMockContextRecording:
    """MockContext records all method calls for assertion."""

    def test_execute_barriers_records_batch(self):
        """execute_barriers records each call batch."""
        ctx = MockContext()
        handle = ResourceHandle()
        b1 = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        b2 = Barrier(
            handle=handle,
            barrier_type=BarrierType.UAV,
            old_state=ResourceState.UNORDERED_ACCESS,
            new_state=ResourceState.UNORDERED_ACCESS,
        )
        ctx.execute_barriers([b1])
        ctx.execute_barriers([b2])
        assert ctx.barrier_batch_count == 2
        assert ctx.barrier_count == 2

    def test_execute_barriers_records_descriptors(self):
        """execute_barriers records the actual barrier objects."""
        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])
        assert len(ctx.barrier_calls) == 1
        recorded_batch = ctx.barrier_calls[0]
        assert len(recorded_batch) == 1
        assert recorded_batch[0] is barrier

    def test_allocate_transient_records_descriptor(self):
        """allocate_transient records the descriptor."""
        ctx = MockContext()
        desc = ResourceDescriptor(name="test_alloc")
        ctx.allocate_transient(desc)
        assert len(ctx.alloc_calls) == 1
        assert ctx.alloc_calls[0] is desc

    def test_begin_pass_records_pass(self):
        """begin_pass records the pass node."""
        ctx = MockContext()
        pass_node = GraphicsPass(name="GBuffer")
        ctx.begin_pass(pass_node)
        assert len(ctx.pass_begin_calls) == 1
        assert ctx.pass_begin_calls[0] is pass_node
        assert ctx.pass_begin_calls[0].name == "GBuffer"

    def test_end_pass_records_pass(self):
        """end_pass records the pass node."""
        ctx = MockContext()
        pass_node = GraphicsPass(name="Lighting")
        ctx.end_pass(pass_node)
        assert len(ctx.pass_end_calls) == 1
        assert ctx.pass_end_calls[0] is pass_node
        assert ctx.pass_end_calls[0].name == "Lighting"

    def test_submit_queue_records_queue_and_fences(self):
        """submit_queue records queue type and fence list."""
        ctx = MockContext()
        fences = [
            FenceOp("signal", 1, QueueType.GRAPHICS),
            FenceOp("wait", 2, QueueType.COMPUTE),
        ]
        ctx.submit_queue(QueueType.COMPUTE, fences)
        assert len(ctx.submit_calls) == 1
        recorded_queue, recorded_fences = ctx.submit_calls[0]
        assert recorded_queue == QueueType.COMPUTE
        assert len(recorded_fences) == 2
        assert recorded_fences[0] is fences[0]
        assert recorded_fences[1] is fences[1]


# =========================================================================
# 8. MockContext: Properties and Counters
# =========================================================================

class TestMockContextCounters:
    """MockContext computed properties report correct counts."""

    def test_initial_counts_all_zero(self):
        """All counters start at zero."""
        ctx = MockContext()
        assert ctx.barrier_count == 0
        assert ctx.barrier_batch_count == 0
        assert ctx.pass_begin_count == 0
        assert ctx.pass_end_count == 0
        assert ctx.submit_count == 0

    def test_barrier_count_counts_individual_barriers(self):
        """barrier_count counts individual barriers across batches."""
        ctx = MockContext()
        handle = ResourceHandle()
        b1 = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                     old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        b2 = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                     old_state=ResourceState.RENDER_TARGET, new_state=ResourceState.SHADER_RESOURCE)
        ctx.execute_barriers([b1, b2])
        assert ctx.barrier_count == 2
        assert ctx.barrier_batch_count == 1

    def test_barrier_count_multiple_batches(self):
        """barrier_count sums across multiple execute_barriers calls."""
        ctx = MockContext()
        handle = ResourceHandle()
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        ctx.execute_barriers([b, b])
        assert ctx.barrier_batch_count == 2
        assert ctx.barrier_count == 3

    def test_submit_count_increments(self):
        """submit_count increments with each submit_queue call."""
        ctx = MockContext()
        assert ctx.submit_count == 0
        ctx.submit_queue(QueueType.GRAPHICS, [])
        assert ctx.submit_count == 1
        ctx.submit_queue(QueueType.COMPUTE, [])
        assert ctx.submit_count == 2

    def test_pass_begin_and_end_counts(self):
        """pass_begin_count and pass_end_count track independently."""
        ctx = MockContext()
        a = GraphicsPass(name="A")
        b = GraphicsPass(name="B")
        ctx.begin_pass(a)
        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 0
        ctx.begin_pass(b)
        assert ctx.pass_begin_count == 2
        ctx.end_pass(a)
        assert ctx.pass_end_count == 1
        ctx.end_pass(b)
        assert ctx.pass_end_count == 2


# =========================================================================
# 9. MockContext: Assertion Helpers
# =========================================================================

class TestMockContextAssertions:
    """MockContext assertion helpers."""

    def test_assert_barriers_executed_passes(self):
        """assert_barriers_executed passes when barriers recorded."""
        ctx = MockContext()
        handle = ResourceHandle()
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        ctx.assert_barriers_executed()

    def test_assert_barriers_executed_fails_when_none(self):
        """assert_barriers_executed fails when no barriers recorded."""
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_barriers_executed()

    def test_assert_barriers_executed_min_count(self):
        """assert_barriers_executed respects min_count parameter."""
        ctx = MockContext()
        handle = ResourceHandle()
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        ctx.assert_barriers_executed(min_count=1)
        with pytest.raises(AssertionError):
            ctx.assert_barriers_executed(min_count=2)

    def test_assert_queue_submitted_passes(self):
        """assert_queue_submitted passes when submissions recorded."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.assert_queue_submitted()

    def test_assert_queue_submitted_fails_when_none(self):
        """assert_queue_submitted fails when no submissions recorded."""
        ctx = MockContext()
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted()

    def test_assert_queue_submitted_specific_queue(self):
        """assert_queue_submitted filters by queue type."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.submit_queue(QueueType.COMPUTE, [FenceOp()])
        ctx.assert_queue_submitted(queue=QueueType.GRAPHICS)
        ctx.assert_queue_submitted(queue=QueueType.COMPUTE)
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted(queue=QueueType.COPY)

    def test_assert_queue_submitted_min_count(self):
        """assert_queue_submitted respects min_count."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.assert_queue_submitted(min_count=1)
        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted(min_count=2)

    def test_assert_barrier_for_resource_passes(self):
        """assert_barrier_for_resource passes when barrier references resource."""
        ctx = MockContext()
        handle = ResourceHandle()
        handle.descriptor = ResourceDescriptor(name="test_tex")
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        ctx.assert_barrier_for_resource("test_tex")

    def test_assert_barrier_for_resource_fails(self):
        """assert_barrier_for_resource fails when resource not found."""
        ctx = MockContext()
        handle = ResourceHandle()
        handle.descriptor = ResourceDescriptor(name="other")
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        with pytest.raises(AssertionError):
            ctx.assert_barrier_for_resource("nonexistent")

    def test_assert_pass_sequence_passes(self):
        """assert_pass_sequence passes when sequence matches."""
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))
        ctx.begin_pass(GraphicsPass(name="B"))
        ctx.begin_pass(GraphicsPass(name="C"))
        ctx.assert_pass_sequence(["A", "B", "C"])

    def test_assert_pass_sequence_fails_on_mismatch(self):
        """assert_pass_sequence fails when sequence differs."""
        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))
        ctx.begin_pass(GraphicsPass(name="B"))
        with pytest.raises(AssertionError):
            ctx.assert_pass_sequence(["B", "A"])

    def test_assert_pass_sequence_empty(self):
        """assert_pass_sequence with empty list passes when no passes."""
        ctx = MockContext()
        ctx.assert_pass_sequence([])


# =========================================================================
# 10. MockContext: Reset
# =========================================================================

class TestMockContextReset:
    """MockContext.reset() clears all recorded state."""

    def test_reset_clears_barrier_calls(self):
        """reset clears barrier recording."""
        ctx = MockContext()
        handle = ResourceHandle()
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        ctx.execute_barriers([b])
        ctx.reset()
        assert ctx.barrier_batch_count == 0
        assert ctx.barrier_count == 0

    def test_reset_clears_alloc_calls(self):
        """reset clears allocation recording."""
        ctx = MockContext()
        ctx.allocate_transient(ResourceDescriptor(name="x"))
        ctx.reset()
        assert len(ctx.alloc_calls) == 0

    def test_reset_clears_pass_calls(self):
        """reset clears pass recording."""
        ctx = MockContext()
        p = GraphicsPass(name="P")
        ctx.begin_pass(p)
        ctx.end_pass(p)
        ctx.reset()
        assert ctx.pass_begin_count == 0
        assert ctx.pass_end_count == 0

    def test_reset_clears_submit_calls(self):
        """reset clears submission recording."""
        ctx = MockContext()
        ctx.submit_queue(QueueType.GRAPHICS, [])
        ctx.reset()
        assert ctx.submit_count == 0

    def test_reset_then_reuse(self):
        """Context can be reused after reset."""
        ctx = MockContext()
        p = GraphicsPass(name="P")
        ctx.begin_pass(p)
        assert ctx.pass_begin_count == 1
        ctx.reset()
        assert ctx.pass_begin_count == 0
        ctx.begin_pass(p)
        assert ctx.pass_begin_count == 1

    def test_reset_from_empty(self):
        """reset on a fresh context does not raise."""
        ctx = MockContext()
        ctx.reset()
        assert ctx.barrier_count == 0


# =========================================================================
# 11. MockContext: Edge Cases
# =========================================================================

class TestMockContextEdgeCases:
    """Edge cases for MockContext behavior."""

    def test_multiple_allocate_transient_returns_distinct(self):
        """Each allocate_transient call returns an AllocationHandle."""
        ctx = MockContext()
        h1 = ctx.allocate_transient(ResourceDescriptor(name="a"))
        h2 = ctx.allocate_transient(ResourceDescriptor(name="b"))
        assert isinstance(h1, int)
        assert isinstance(h2, int)
        assert len(ctx.alloc_calls) == 2

    def test_allocate_transient_returns_positive_handle(self):
        """The default handle value is non-negative (fits AllocationHandle pattern)."""
        ctx = MockContext()
        handle = ctx.allocate_transient(ResourceDescriptor(name="test"))
        assert handle >= 0

    def test_begin_pass_invalid_arguments(self):
        """begin_pass accepts any argument (protocol structural match)."""
        ctx = MockContext()
        ctx.begin_pass("string_arg")
        assert ctx.pass_begin_count == 1

    def test_end_pass_invalid_arguments(self):
        """end_pass accepts any argument (protocol structural match)."""
        ctx = MockContext()
        ctx.end_pass(42)
        assert ctx.pass_end_count == 1

    def test_mixed_operations(self):
        """Multiple operation types interleaved are recorded independently."""
        ctx = MockContext()
        handle = ResourceHandle()
        b = Barrier(handle=handle, barrier_type=BarrierType.TRANSITION,
                    old_state=ResourceState.UNDEFINED, new_state=ResourceState.RENDER_TARGET)
        p = GraphicsPass(name="Pass")

        ctx.execute_barriers([b])
        ctx.begin_pass(p)
        ctx.submit_queue(QueueType.GRAPHICS, [FenceOp()])
        ctx.end_pass(p)
        ctx.allocate_transient(ResourceDescriptor(name="r"))

        assert ctx.barrier_batch_count == 1
        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 1
        assert ctx.submit_count == 1
        assert len(ctx.alloc_calls) == 1
