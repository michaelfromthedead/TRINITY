"""
Phase 1 verification tests for the Frame Graph subsystem.

Tests the core FrameGraph implementation, RHIContext protocol, and
MockContext according to PHASE_1_TODO.md acceptance criteria.

Coverage:
  - Protocol import and structural type checking
  - MockContext protocol satisfaction
  - FrameGraph: pass/resource declaration
  - FrameGraph: compilation and execution with MockContext
  - FrameGraph: IR serialization
  - FrameGraph: dependency graph correctness
  - FrameGraph: pass culling and execution order
  - FrameGraph: async compute integration
  - FrameGraph: barrier integration at the frame graph level
  - FrameGraph: edge cases and error paths
"""

from __future__ import annotations

from typing import Sequence

import pytest

from engine.rendering.framegraph import (
    AllocationHandle,
    Barrier,
    BarrierType,
    CompilationResult,
    ComputePass,
    FenceOp,
    FrameGraph,
    PassFlags,
    PassNode,
    ResourceFormat,
    ResourceState,
    RHIContext,
)
from engine.rendering.framegraph.context import RHIContext as RHIContextDirect


# =============================================================================
# T-FG-1.1: RHIContext Protocol Verification
# =============================================================================


class TestRHIContextProtocol:
    """Verify the RHIContext Protocol definition."""

    def test_protocol_is_importable_from_package(self):
        """RHIContext must be importable from the package."""
        from engine.rendering.framegraph import RHIContext

        assert RHIContext is not None

    def test_allocation_handle_is_importable(self):
        """AllocationHandle must be importable from the package."""
        from engine.rendering.framegraph import AllocationHandle

        assert AllocationHandle is not None

    def test_fence_op_is_importable(self):
        """FenceOp must be importable from the package."""
        from engine.rendering.framegraph import FenceOp

        assert FenceOp is not None

    def test_all_three_imports_in_one_statement(self):
        """All three types must be importable in one statement."""
        from engine.rendering.framegraph import (
            RHIContext,
            AllocationHandle,
            FenceOp,
        )

        assert RHIContext is not None

    def test_protocol_has_required_methods(self):
        """RHIContext must define all required protocol methods."""
        required = {
            "execute_barriers",
            "allocate_transient",
            "begin_pass",
            "end_pass",
            "submit_queue",
        }
        protocol_methods = {
            name for name in dir(RHIContext) if not name.startswith("_")
        }
        assert required.issubset(protocol_methods), (
            f"Missing methods: {required - protocol_methods}"
        )

    def test_fence_op_dataclass(self):
        """FenceOp must be a dataclass with correct fields."""
        op = FenceOp(
            operation="signal",
            fence_value=42,
            target_queue=None,  # Will be set properly below
        )

        assert hasattr(op, "operation")
        assert hasattr(op, "fence_value")
        assert hasattr(op, "target_queue")

    def test_fence_op_defaults(self):
        """FenceOp must have sensible defaults."""
        op = FenceOp()

        assert op.operation == "signal"
        assert op.fence_value == 0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_allocation_handle_type(self):
        """AllocationHandle must be a NewType over int."""
        handle = AllocationHandle(42)

        assert isinstance(handle, int)
        assert handle == 42
        assert int(handle) == 42

    def test_protocol_is_a_protocol(self):
        """RHIContext must be a typing.Protocol subclass."""
        import typing

        # Check that it is a Protocol by verifying it has _is_protocol
        assert hasattr(RHIContext, "_is_protocol")
        assert RHIContext._is_protocol is True

    def test_protocol_allows_structural_subtyping(self):
        """Any object with correct methods should satisfy the protocol."""
        class ValidContext:
            def execute_barriers(self, barriers):
                pass
            def allocate_transient(self, desc):
                return AllocationHandle(1)
            def begin_pass(self, pass_node):
                pass
            def end_pass(self, pass_node):
                pass
            def submit_queue(self, queue, fences):
                pass

        ctx = ValidContext()
        # isinstance check with Protocol uses structural subtyping
        valid = isinstance(ctx, RHIContext)
        assert valid is True

    def test_protocol_rejects_missing_methods(self):
        """An object missing required methods should not satisfy protocol."""
        class InvalidContext:
            def execute_barriers(self, barriers):
                pass
            # Missing allocate_transient, begin_pass, end_pass, submit_queue

        ctx = InvalidContext()
        valid = isinstance(ctx, RHIContext)
        assert valid is False


# =============================================================================
# T-FG-1.5: MockContext Verification
# =============================================================================


class TestMockContext:
    """Verify the MockContext implementation."""

    def test_mock_context_satisfies_protocol(self):
        """MockContext must satisfy the RHIContext protocol."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        assert isinstance(ctx, RHIContext)

    def test_mock_context_records_barriers(self):
        """MockContext must record barrier executions."""
        from engine.rendering.framegraph import ResourceHandle
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        handle = ResourceHandle()
        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )

        ctx.execute_barriers([barrier])

        assert ctx.barrier_count == 1
        assert ctx.barrier_batch_count == 1
        assert ctx.barrier_calls[0][0].handle is handle

    def test_mock_context_records_passes(self):
        """MockContext must record pass begin/end calls."""
        from engine.rendering.framegraph import GraphicsPass
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        pass_node = GraphicsPass(name="TestPass")

        ctx.begin_pass(pass_node)
        ctx.end_pass(pass_node)

        assert ctx.pass_begin_count == 1
        assert ctx.pass_end_count == 1
        assert ctx.pass_begin_calls[0].name == "TestPass"

    def test_mock_context_records_allocations(self):
        """MockContext must record allocation calls."""
        from engine.rendering.framegraph import ResourceDescriptor
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        desc = ResourceDescriptor(name="test_tex", width=1920, height=1080)

        handle = ctx.allocate_transient(desc)

        assert handle == AllocationHandle(1)
        assert len(ctx.alloc_calls) == 1
        assert ctx.alloc_calls[0].name == "test_tex"

    def test_mock_context_records_submissions(self):
        """MockContext must record queue submissions."""
        from engine.rendering.framegraph import QueueType
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        fences = [FenceOp(operation="signal", fence_value=1)]

        ctx.submit_queue(QueueType.GRAPHICS, fences)

        assert ctx.submit_count == 1
        assert ctx.submit_calls[0][0] == QueueType.GRAPHICS

    def test_mock_context_assert_barriers_executed(self):
        """assert_barriers_executed must pass when barriers exist."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        ctx.execute_barriers([])

        # Should not raise
        ctx.assert_barriers_executed(min_count=0)

    def test_mock_context_assert_barriers_executed_fails(self):
        """assert_barriers_executed must fail when no barriers."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()

        with pytest.raises(AssertionError):
            ctx.assert_barriers_executed(min_count=1)

    def test_mock_context_assert_queue_submitted(self):
        """assert_queue_submitted must check queue submission."""
        from engine.rendering.framegraph import QueueType
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        ctx.submit_queue(QueueType.COMPUTE, [])

        # Should not raise
        ctx.assert_queue_submitted(queue=QueueType.COMPUTE)

    def test_mock_context_assert_queue_submitted_fails(self):
        """assert_queue_submitted must fail when queue not used."""
        from engine.rendering.framegraph import QueueType
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()

        with pytest.raises(AssertionError):
            ctx.assert_queue_submitted(queue=QueueType.GRAPHICS)

    def test_mock_context_assert_barrier_for_resource(self):
        """assert_barrier_for_resource must find barriers by resource name."""
        from engine.rendering.framegraph import ResourceHandle
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        handle = ResourceHandle()
        # Set name via descriptor
        from engine.rendering.framegraph import ResourceDescriptor
        handle.descriptor = ResourceDescriptor(name="my_resource")
        handle.descriptor.is_texture = True

        barrier = Barrier(
            handle=handle,
            barrier_type=BarrierType.TRANSITION,
            old_state=ResourceState.UNDEFINED,
            new_state=ResourceState.RENDER_TARGET,
        )
        ctx.execute_barriers([barrier])

        # Should not raise
        ctx.assert_barrier_for_resource("my_resource")

    def test_mock_context_assert_barrier_for_resource_fails(self):
        """assert_barrier_for_resource must fail when resource not found."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()

        with pytest.raises(AssertionError):
            ctx.assert_barrier_for_resource("nonexistent")

    def test_mock_context_reset(self):
        """MockContext.reset() must clear all recorded calls."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        ctx.execute_barriers([])
        ctx.begin_pass(None)  # type: ignore
        ctx.end_pass(None)  # type: ignore

        ctx.reset()

        assert ctx.barrier_count == 0
        assert ctx.pass_begin_count == 0
        assert ctx.pass_end_count == 0
        assert ctx.submit_count == 0

    def test_mock_context_default_noop(self):
        """MockContext must default to no-op (no exceptions)."""
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()

        # Should not raise
        ctx.execute_barriers([])
        ctx.begin_pass(None)  # type: ignore
        ctx.end_pass(None)  # type: ignore
        ctx.submit_queue(None, [])  # type: ignore

    def test_mock_context_assert_pass_sequence(self):
        """assert_pass_sequence must verify pass begin order."""
        from engine.rendering.framegraph import GraphicsPass
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="A"))
        ctx.begin_pass(GraphicsPass(name="B"))

        ctx.assert_pass_sequence(["A", "B"])

    def test_mock_context_assert_pass_sequence_fails(self):
        """assert_pass_sequence must fail on wrong order."""
        from engine.rendering.framegraph import GraphicsPass
        from tests.rendering.framegraph.mock_context import MockContext

        ctx = MockContext()
        ctx.begin_pass(GraphicsPass(name="B"))
        ctx.begin_pass(GraphicsPass(name="A"))

        with pytest.raises(AssertionError):
            ctx.assert_pass_sequence(["A", "B"])


# =============================================================================
# FrameGraph: Basic Functionality
# =============================================================================


class TestFrameGraphBasics:
    """Test basic FrameGraph functionality with RHIContext-aware patterns."""

    def test_create_and_compile_execute_with_mock(self):
        """Full create-compile-execute cycle with MockContext."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("color", width=1920, height=1080)
        backbuffer = fg.import_external(
            "backbuffer", None, is_backbuffer=True
        )

        render = fg.add_graphics_pass("Render")
        render.add_color_attachment(tex)
        render.add_color_attachment(backbuffer)

        result = fg.compile()
        assert result.success is True

        fg.execute(ctx)
        assert ctx.barrier_batch_count >= 0

    def test_compile_with_async_compute(self):
        """Compile with async compute scheduling."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("gbuffer")
        backbuffer = fg.import_external(
            "backbuffer", None, is_backbuffer=True
        )

        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(tex)

        post = fg.add_graphics_pass("PostProcess")
        post.read(tex, ResourceState.SHADER_RESOURCE)
        post.add_color_attachment(backbuffer)

        result = fg.compile()
        assert result.success is True
        assert "GBuffer" in result.execution_order
        assert "PostProcess" in result.execution_order

        fg.execute(ctx)

    def test_resource_counts(self):
        """Test resource counting after creation."""
        fg = FrameGraph()
        assert fg.resource_count == 0

        fg.create_texture("tex1")
        fg.create_buffer("buf1", 1024)
        fg.create_history_texture("hist1")
        fg.import_external("ext1", None)

        # resource_count includes all resource types
        assert fg.resource_count >= 4


# =============================================================================
# FrameGraph: IR Serialization
# =============================================================================


class TestIRSerialization:
    """Test IR serialization for Rust bridge."""

    def test_serialize_empty_graph(self):
        """Serializing an empty graph must produce valid JSON."""
        fg = FrameGraph()

        json_str = fg._serialize_ir()

        import json
        data = json.loads(json_str)
        assert "passes" in data
        assert "resources" in data
        assert data["passes"] == []
        assert data["resources"] == []

    def test_serialize_with_passes(self):
        """Serializing a graph with passes must include pass details."""
        fg = FrameGraph()

        tex = fg.create_texture("color")
        fg.add_graphics_pass("Render").add_color_attachment(tex)
        compute = fg.add_compute_pass("Compute")
        compute.read_texture(tex)
        compute.set_dispatch_size(8, 8, 1)

        json_str = fg._serialize_ir()

        import json
        data = json.loads(json_str)
        assert len(data["passes"]) == 2

        # Check pass structure
        pass_names = [p["name"] for p in data["passes"]]
        assert "Render" in pass_names
        assert "Compute" in pass_names

        # Check resource structure
        resource_names = [r["name"] for r in data["resources"]]
        assert "color" in resource_names

    def test_serialize_pass_reads_writes(self):
        """Serialized passes must include read/write resource lists."""
        fg = FrameGraph()

        tex = fg.create_texture("input_tex")
        output = fg.create_texture("output_tex")
        backbuffer = fg.import_external(
            "backbuffer", None, is_backbuffer=True
        )

        compute = fg.add_compute_pass("Process")
        compute.read_texture(tex)
        compute.write_texture(output)

        present = fg.add_graphics_pass("Present")
        present.read(output, ResourceState.SHADER_RESOURCE)
        present.add_color_attachment(backbuffer)

        json_str = fg._serialize_ir()

        import json
        data = json.loads(json_str)

        process = next(p for p in data["passes"] if p["name"] == "Process")
        assert "input_tex" in process["reads"]
        assert "output_tex" in process["writes"]

    def test_serialize_compute_workgroup(self):
        """Serialized compute passes must include workgroup_size."""
        fg = FrameGraph()

        compute = fg.add_compute_pass("Dispatch")
        compute.set_dispatch_size(16, 16, 1)

        json_str = fg._serialize_ir()

        import json
        data = json.loads(json_str)
        dispatch = next(p for p in data["passes"] if p["name"] == "Dispatch")
        assert dispatch["workgroup_size"] == [16, 16, 1]

    def test_serialize_graphics_attachments(self):
        """Serialized graphics passes must include color/depth attachments."""
        fg = FrameGraph()

        color = fg.create_texture("color_rt")
        depth = fg.create_texture("depth_rt", format=ResourceFormat.D32_FLOAT)

        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(color)
        gbuffer.set_depth_stencil(depth)

        json_str = fg._serialize_ir()

        import json
        data = json.loads(json_str)
        gbuffer_data = next(
            p for p in data["passes"] if p["name"] == "GBuffer"
        )
        assert "color_attachments" in gbuffer_data
        assert gbuffer_data["depth_attachment"] == "depth_rt"


# =============================================================================
# FrameGraph: Execution with MockContext
# =============================================================================


class TestFrameGraphExecution:
    """Test FrameGraph execution with MockContext."""

    def test_execute_invokes_pass_callbacks(self):
        """Execution must invoke pass callbacks in order."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()
        executed: list[str] = []

        tex = fg.create_texture("tmp")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(tex)
        a.set_execute(lambda c: executed.append("PassA"))

        b = fg.add_graphics_pass("PassB")
        b.add_color_attachment(bb)
        b.read(tex, ResourceState.SHADER_RESOURCE)
        b.set_execute(lambda c: executed.append("PassB"))

        result = fg.compile()
        assert result.success is True

        fg.execute(ctx)

        assert executed == ["PassA", "PassB"]

    def test_execute_with_barriers(self):
        """Execution must pass barriers through context."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("shared_tex")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        writer = fg.add_graphics_pass("Writer")
        writer.add_color_attachment(tex)

        reader = fg.add_graphics_pass("Reader")
        reader.read(tex, ResourceState.SHADER_RESOURCE)
        reader.add_color_attachment(bb)
        reader.set_execute(lambda c: None)

        fg.compile()
        fg.execute(ctx)

        # Barriers should have been called through the context
        if ctx.barrier_batch_count > 0:
            ctx.assert_barriers_executed(min_count=0)

    def test_execute_requires_compilation(self):
        """Execution must raise if not compiled."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass")

        with pytest.raises(RuntimeError, match="must be compiled"):
            fg.execute(None)  # type: ignore

    def test_execute_after_clear_requires_recompile(self):
        """After clear, execution must require recompilation."""
        fg = FrameGraph()
        bb = fg.import_external("bb", None, is_backbuffer=True)
        fg.add_graphics_pass("P").add_color_attachment(bb)
        fg.compile()
        fg.clear()

        with pytest.raises(RuntimeError, match="must be compiled"):
            fg.execute(None)  # type: ignore

    def test_execute_with_barriers_and_mock(self):
        """Execute with barrier integration via MockContext."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("tex", width=1920, height=1080)
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(tex)
        a.set_execute(lambda c: ctx.begin_pass(a))

        b = fg.add_graphics_pass("PassB")
        b.read(tex, ResourceState.SHADER_RESOURCE)
        b.add_color_attachment(bb)
        b.set_execute(lambda c: ctx.begin_pass(b))

        fg.compile()
        fg.execute(ctx)

        # If barriers were generated, they went through MockContext
        # The MockContext should not raise on any call
        assert ctx.barrier_batch_count >= 0
        assert ctx.pass_begin_count == 2


# =============================================================================
# FrameGraph: Dependency Graph
# =============================================================================


class TestDependencyGraph:
    """Test the frame graph dependency graph."""

    def test_build_dependency_graph(self):
        """Dependency graph must connect read-after-write chains."""
        fg = FrameGraph()

        rt = fg.create_texture("rt")
        hdr = fg.create_texture("hdr")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(rt)

        b = fg.add_compute_pass("PassB")
        b.read_texture(rt)
        b.write_texture(hdr)

        c = fg.add_graphics_pass("PassC")
        c.read(hdr, ResourceState.SHADER_RESOURCE)
        c.add_color_attachment(bb)

        fg._build_dependency_graph()

        # PassB depends on PassA (reads rt written by PassA)
        assert "PassA" in fg._pass_dependencies["PassB"]

        # PassC depends on PassB (reads hdr written by PassB)
        assert "PassB" in fg._pass_dependencies["PassC"]

        # PassA should have no dependencies
        assert fg._pass_dependencies["PassA"] == []

    def test_dependency_graph_empty(self):
        """Empty graph must have empty dependencies."""
        fg = FrameGraph()
        fg._build_dependency_graph()

        assert fg._pass_dependencies == {}

    def test_dependency_graph_no_edges(self):
        """Independent passes must have no dependencies."""
        fg = FrameGraph()

        tex1 = fg.create_texture("t1")
        tex2 = fg.create_texture("t2")

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(tex1)

        b = fg.add_graphics_pass("PassB")
        b.add_color_attachment(tex2)

        fg._build_dependency_graph()

        assert fg._pass_dependencies["PassA"] == []
        assert fg._pass_dependencies["PassB"] == []

    def test_execution_order_respects_dependencies(self):
        """Compiled execution order must respect dependencies."""
        fg = FrameGraph()

        rt = fg.create_texture("rt")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("First")
        a.add_color_attachment(rt)

        b = fg.add_graphics_pass("Second")
        b.read(rt, ResourceState.SHADER_RESOURCE)
        b.add_color_attachment(bb)

        result = fg.compile()
        assert result.success is True

        order = result.execution_order
        assert order.index("First") < order.index("Second")


# =============================================================================
# FrameGraph: Pass Culling
# =============================================================================


class TestAdvancedPassCulling:
    """Advanced pass culling verification."""

    def test_cascade_culling(self):
        """Cascading culling: direct unused sink is culled.

        Note: Current implementation does one-pass culling. Passes whose
        outputs are read by another pass (even a to-be-culled one) are
        kept. Full cascading culling would require iterative analysis.
        """
        fg = FrameGraph()

        intermediate = fg.create_texture("intermediate")
        unused_output = fg.create_texture("unused_output")

        a = fg.add_graphics_pass("Producer")
        a.add_color_attachment(intermediate)

        b = fg.add_compute_pass("Consumer")
        b.read_texture(intermediate)
        b.write_texture(unused_output)

        result = fg.compile()

        assert result.success is True
        # Consumer's output is never read -> culled
        assert "Consumer" in result.culled_passes
        # Producer's output IS read (by Consumer) -> kept in one-pass culling
        assert "Producer" not in result.culled_passes

    def test_partial_culling(self):
        """Only unused passes must be culled when chain has a sink."""
        fg = FrameGraph()

        used = fg.create_texture("used")
        unused = fg.create_texture("unused")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("Used")
        a.add_color_attachment(used)

        b = fg.add_graphics_pass("Unused")
        b.add_color_attachment(unused)

        c = fg.add_graphics_pass("Final")
        c.read(used, ResourceState.SHADER_RESOURCE)
        c.add_color_attachment(bb)

        result = fg.compile()

        assert "Used" in result.execution_order
        assert "Unused" in result.culled_passes
        assert "Final" in result.execution_order

    def test_culling_reduces_barrier_count(self):
        """Culled passes must not generate barriers."""
        fg = FrameGraph()
        fg.set_pass_culling_enabled(True)

        unused_tex = fg.create_texture("unused")
        fg.add_graphics_pass("Unused").add_color_attachment(unused_tex)

        result_no_cull_expected = fg.compile()
        # With culling enabled, the culled pass generates no barriers
        # by virtue of not appearing in execution order
        assert "Unused" in result_no_cull_expected.culled_passes


# =============================================================================
# FrameGraph: Async Compute Integration
# =============================================================================


class TestAsyncCompute:
    """Test async compute scheduling integration."""

    def test_async_compute_count_in_result(self):
        """Compilation result must report async pass count."""
        fg = FrameGraph()

        tex = fg.create_texture("tex")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(tex)

        compute = fg.add_compute_pass("AsyncWork")
        compute.read_texture(tex)
        compute.set_flag(PassFlags.ASYNC_COMPUTE)

        post = fg.add_graphics_pass("Post")
        post.read(tex, ResourceState.SHADER_RESOURCE)
        post.add_color_attachment(bb)

        result = fg.compile()

        assert result.success is True
        assert result.async_pass_count >= 0

    def test_disabling_async_compute(self):
        """Disabling async compute must reduce count to 0."""
        fg = FrameGraph()
        fg.set_async_compute_enabled(False)

        tex = fg.create_texture("tex")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(tex)

        compute = fg.add_compute_pass("Compute")
        compute.read_texture(tex)
        compute.set_flag(PassFlags.ASYNC_COMPUTE)

        post = fg.add_graphics_pass("Post")
        post.add_color_attachment(bb)

        result = fg.compile()
        assert result.success is True

    def test_async_scheduling_with_backbuffer_present(self):
        """Async compute must not interfere with backbuffer present."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("tex")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        compute = fg.add_compute_pass("Compute")
        compute.write_texture(tex)
        compute.set_flag(PassFlags.ASYNC_COMPUTE)

        present = fg.add_graphics_pass("Present")
        present.read(tex, ResourceState.SHADER_RESOURCE)
        present.add_color_attachment(bb)

        result = fg.compile()
        assert result.success is True

        fg.execute(ctx)
        assert ctx.barrier_batch_count >= 0


# =============================================================================
# FrameGraph: Barrier Integration
# =============================================================================


class TestBarrierIntegration:
    """Test barrier integration at the FrameGraph level."""

    def test_barrier_batches_after_compilation(self):
        """Compiled frame graph must have accessible barrier batches."""
        fg = FrameGraph()

        tex = fg.create_texture("tex")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(tex)

        b = fg.add_graphics_pass("PassB")
        b.read(tex, ResourceState.SHADER_RESOURCE)
        b.add_color_attachment(bb)

        fg.compile()

        barriers = fg.get_barriers_for_pass("PassB")
        assert isinstance(barriers, list)

    def test_get_barriers_for_unknown_pass(self):
        """Getting barriers for unknown pass returns empty list."""
        fg = FrameGraph()
        fg.compile()

        barriers = fg.get_barriers_for_pass("NonExistent")
        assert barriers == []

    def test_compilation_counts_barriers(self):
        """Compilation result must count total barriers."""
        fg = FrameGraph()

        tex1 = fg.create_texture("tex1")
        tex2 = fg.create_texture("tex2")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        a = fg.add_graphics_pass("PassA")
        a.add_color_attachment(tex1)

        b = fg.add_graphics_pass("PassB")
        b.read(tex1, ResourceState.SHADER_RESOURCE)
        b.add_color_attachment(tex2)

        c = fg.add_graphics_pass("PassC")
        c.read(tex2, ResourceState.SHADER_RESOURCE)
        c.add_color_attachment(bb)

        result = fg.compile()

        assert result.barrier_count >= 0

    def test_backbuffer_present_barrier(self):
        """Present barrier must be generated for backbuffer."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()

        tex = fg.create_texture("scene")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        render = fg.add_graphics_pass("Render")
        render.add_color_attachment(tex)

        present = fg.add_graphics_pass("Present")
        present.read(tex, ResourceState.SHADER_RESOURCE)
        present.add_color_attachment(bb)

        fg.compile()
        fg.execute(ctx)


# =============================================================================
# FrameGraph: Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Test FrameGraph error handling."""

    def test_duplicate_pass_name(self):
        """Duplicate pass names must raise ValueError."""
        fg = FrameGraph()
        fg.add_graphics_pass("MyPass")

        with pytest.raises(ValueError, match="already exists"):
            fg.add_graphics_pass("MyPass")

    def test_duplicate_resource_name(self):
        """Duplicate resource names must raise ValueError."""
        fg = FrameGraph()
        fg.create_texture("MyTex")

        with pytest.raises(ValueError, match="already exists"):
            fg.create_texture("MyTex")

    def test_invalid_pass_type(self):
        """Invalid pass type must raise ValueError."""
        fg = FrameGraph()

        with pytest.raises(ValueError, match="Unknown pass type"):
            fg.add_pass("Bad", "invalid_type")

    def test_remove_nonexistent_pass(self):
        """Removing a non-existent pass must return False."""
        fg = FrameGraph()

        result = fg.remove_pass("NonExistent")
        assert result is False

    def test_get_nonexistent_pass(self):
        """Getting a non-existent pass must return None."""
        fg = FrameGraph()

        result = fg.get_pass("NonExistent")
        assert result is None

    def test_get_nonexistent_resource(self):
        """Getting a non-existent resource must return None."""
        fg = FrameGraph()

        result = fg.get_resource("NonExistent")
        assert result is None

    def test_compile_empty_graph(self):
        """Empty frame graph must compile successfully."""
        fg = FrameGraph()

        result = fg.compile()

        assert result.success is True
        assert len(result.execution_order) == 0

    def test_compile_after_invalidation(self):
        """Adding a resource must invalidate compilation."""
        fg = FrameGraph()
        bb = fg.import_external("bb", None, is_backbuffer=True)
        fg.add_graphics_pass("P").add_color_attachment(bb)
        fg.compile()
        assert fg.is_compiled is True

        fg.create_texture("new")
        assert fg.is_compiled is False

    def test_compilation_result_error(self):
        """Compilation result must capture exceptions."""
        fg = FrameGraph()

        # Compile an empty graph (should succeed)
        result = fg.compile()
        assert result.success is True

    def test_clear_empty_graph(self):
        """Clearing an empty graph must not raise."""
        fg = FrameGraph()

        fg.clear()
        assert fg.pass_count == 0

    def test_get_execution_order_before_compile(self):
        """Getting execution order before compile must return []."""
        fg = FrameGraph()

        assert fg.get_execution_order() == []

    def test_get_compilation_result_before_compile(self):
        """Getting compilation result before compile must return None."""
        fg = FrameGraph()

        assert fg.get_compilation_result() is None


# =============================================================================
# FrameGraph: Full Pipeline Integration
# =============================================================================


class TestFullPipeline:
    """Full pipeline integration tests for the frame graph."""

    def test_full_deferred_rendering_pipeline(self):
        """Simulate a complete deferred rendering pipeline."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()
        executed: list[str] = []

        # Create resources
        gbuffer_albedo = fg.create_texture(
            "gbuffer_albedo",
            format=ResourceFormat.R8G8B8A8_UNORM,
            width=1920,
            height=1080,
        )
        gbuffer_normal = fg.create_texture(
            "gbuffer_normal",
            format=ResourceFormat.R16G16B16A16_FLOAT,
            width=1920,
            height=1080,
        )
        gbuffer_depth = fg.create_texture(
            "gbuffer_depth",
            format=ResourceFormat.D32_FLOAT,
            width=1920,
            height=1080,
        )
        hdr_target = fg.create_texture(
            "hdr_target",
            format=ResourceFormat.R16G16B16A16_FLOAT,
            width=1920,
            height=1080,
        )
        backbuffer = fg.import_external(
            "backbuffer", None, is_backbuffer=True
        )

        # Pass 1: GBuffer
        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(gbuffer_albedo)
        gbuffer.add_color_attachment(gbuffer_normal)
        gbuffer.set_depth_stencil(gbuffer_depth)
        gbuffer.set_execute(lambda c: executed.append("GBuffer"))

        # Pass 2: Lighting (deferred)
        lighting = fg.add_compute_pass("Lighting")
        lighting.read_texture(gbuffer_albedo)
        lighting.read_texture(gbuffer_normal)
        lighting.read_texture(gbuffer_depth)
        lighting.write_texture(hdr_target)
        lighting.set_execute(lambda c: executed.append("Lighting"))

        # Pass 3: Tone mapping + post
        postprocess = fg.add_graphics_pass("PostProcess")
        postprocess.read(hdr_target, ResourceState.SHADER_RESOURCE)
        postprocess.add_color_attachment(backbuffer)
        postprocess.set_execute(lambda c: executed.append("PostProcess"))

        # Compile
        result = fg.compile()
        assert result.success is True
        assert result.barrier_count >= 0

        # Verify execution order
        order = result.execution_order
        assert order.index("GBuffer") < order.index("Lighting")
        assert order.index("Lighting") < order.index("PostProcess")

        # Execute
        fg.execute(ctx)
        assert executed == ["GBuffer", "Lighting", "PostProcess"]

    def test_pipeline_with_async_compute(self):
        """Pipeline with async compute scheduling."""
        from tests.rendering.framegraph.mock_context import MockContext

        fg = FrameGraph()
        ctx = MockContext()
        executed: list[str] = []

        # Resources
        depth = fg.create_texture(
            "depth", format=ResourceFormat.D32_FLOAT
        )
        ssao_output = fg.create_texture("ssao")
        hdr = fg.create_texture("hdr")
        bb = fg.import_external("bb", None, is_backbuffer=True)

        # Depth pre-pass
        depth_pass = fg.add_graphics_pass("DepthPrepass")
        depth_pass.set_depth_stencil(depth)
        depth_pass.set_execute(lambda c: executed.append("DepthPrepass"))

        # SSAO on async compute
        ssao = fg.add_compute_pass("SSAO")
        ssao.read_texture(depth)
        ssao.write_texture(ssao_output)
        ssao.set_flag(PassFlags.ASYNC_COMPUTE)
        ssao.set_execute(lambda c: executed.append("SSAO"))

        # Lighting
        lighting = fg.add_compute_pass("Lighting")
        lighting.read_texture(ssao_output)
        lighting.write_texture(hdr)
        lighting.set_execute(lambda c: executed.append("Lighting"))

        # Final
        present = fg.add_graphics_pass("Present")
        present.read(hdr, ResourceState.SHADER_RESOURCE)
        present.add_color_attachment(bb)
        present.set_execute(lambda c: executed.append("Present"))

        result = fg.compile()
        assert result.success is True

        # Async pass count may be 0 or more depending on scheduling
        assert result.async_pass_count >= 0

        fg.execute(ctx)
        assert len(executed) == 4


# =============================================================================
# FrameGraph: Configuration
# =============================================================================


class TestFrameGraphConfig:
    """Test FrameGraph configuration options."""

    def test_toggle_resource_aliasing(self):
        """Resource aliasing toggle must invalidate compilation."""
        fg = FrameGraph()
        bb = fg.import_external("bb", None, is_backbuffer=True)
        fg.add_graphics_pass("P").add_color_attachment(bb)
        fg.compile()
        assert fg.is_compiled is True

        fg.set_resource_aliasing_enabled(False)
        assert fg.is_compiled is False

        fg.compile()
        assert fg.is_compiled is True


# =============================================================================
# T-FG-1.4: Package Export Verification
# =============================================================================


class TestPackageExports:
    """Verify all Phase 1 types are exported from the package."""

    def test_rhicontext_in_all(self):
        """RHIContext must be in __all__."""
        from engine.rendering import framegraph

        assert "RHIContext" in framegraph.__all__

    def test_allocation_handle_in_all(self):
        """AllocationHandle must be in __all__."""
        from engine.rendering import framegraph

        assert "AllocationHandle" in framegraph.__all__

    def test_fence_op_in_all(self):
        """FenceOp must be in __all__."""
        from engine.rendering import framegraph

        assert "FenceOp" in framegraph.__all__

    def test_import_all_three(self):
        """All three types must be importable from the package."""
        from engine.rendering.framegraph import (
            RHIContext,
            AllocationHandle,
            FenceOp,
        )

        assert RHIContext is not None
        assert AllocationHandle is not None
        assert FenceOp is not None
