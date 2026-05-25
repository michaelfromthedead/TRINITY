"""
Tests for the Frame Graph implementation.

Tests the core FrameGraph class including:
- Pass declaration
- Dependency tracking
- Compilation
- Execution
- Pass culling
"""

import pytest

from engine.rendering.framegraph import (
    CompilationResult,
    ComputePass,
    CopyPass,
    FrameGraph,
    GraphicsPass,
    PassFlags,
    PassType,
    RayTracingPass,
    ResourceFormat,
    ResourceState,
)


# =============================================================================
# Pass Declaration
# =============================================================================


class TestPassDeclaration:
    """Test pass creation and configuration."""

    def test_add_graphics_pass(self):
        """Test adding a graphics pass."""
        fg = FrameGraph()
        pass_node = fg.add_graphics_pass("GBuffer")

        assert pass_node is not None
        assert pass_node.name == "GBuffer"
        assert isinstance(pass_node, GraphicsPass)
        assert pass_node.pass_type == PassType.GRAPHICS

    def test_add_compute_pass(self):
        """Test adding a compute pass."""
        fg = FrameGraph()
        pass_node = fg.add_compute_pass("SSAO")

        assert pass_node is not None
        assert pass_node.name == "SSAO"
        assert isinstance(pass_node, ComputePass)
        assert pass_node.pass_type == PassType.COMPUTE

    def test_add_copy_pass(self):
        """Test adding a copy/transfer pass."""
        fg = FrameGraph()
        pass_node = fg.add_copy_pass("Upload")

        assert pass_node is not None
        assert pass_node.name == "Upload"
        assert isinstance(pass_node, CopyPass)
        assert pass_node.pass_type == PassType.COPY

    def test_add_raytracing_pass(self):
        """Test adding a ray tracing pass."""
        fg = FrameGraph()
        pass_node = fg.add_raytracing_pass("RTReflections")

        assert pass_node is not None
        assert pass_node.name == "RTReflections"
        assert isinstance(pass_node, RayTracingPass)
        assert pass_node.pass_type == PassType.RAY_TRACING

    def test_add_pass_generic(self):
        """Test generic add_pass method."""
        fg = FrameGraph()

        graphics = fg.add_pass("Pass1", "graphics")
        assert graphics.pass_type == PassType.GRAPHICS

        compute = fg.add_pass("Pass2", "compute")
        assert compute.pass_type == PassType.COMPUTE

        copy = fg.add_pass("Pass3", "copy")
        assert copy.pass_type == PassType.COPY

    def test_duplicate_pass_name_fails(self):
        """Test that duplicate pass names are rejected."""
        fg = FrameGraph()
        fg.add_graphics_pass("MyPass")

        with pytest.raises(ValueError, match="already exists"):
            fg.add_graphics_pass("MyPass")

    def test_invalid_pass_type_fails(self):
        """Test that invalid pass types are rejected."""
        fg = FrameGraph()

        with pytest.raises(ValueError, match="Unknown pass type"):
            fg.add_pass("Bad", "invalid_type")

    def test_get_pass(self):
        """Test retrieving a pass by name."""
        fg = FrameGraph()
        created = fg.add_graphics_pass("MyPass")

        retrieved = fg.get_pass("MyPass")
        assert retrieved is created

    def test_get_nonexistent_pass(self):
        """Test retrieving a pass that doesn't exist."""
        fg = FrameGraph()

        result = fg.get_pass("NonExistent")
        assert result is None

    def test_remove_pass(self):
        """Test removing a pass."""
        fg = FrameGraph()
        fg.add_graphics_pass("ToRemove")

        assert fg.remove_pass("ToRemove") is True
        assert fg.get_pass("ToRemove") is None
        assert fg.pass_count == 0

    def test_remove_nonexistent_pass(self):
        """Test removing a pass that doesn't exist."""
        fg = FrameGraph()

        assert fg.remove_pass("NonExistent") is False


# =============================================================================
# Resource Creation
# =============================================================================


class TestResourceCreation:
    """Test resource creation and management."""

    def test_create_texture(self):
        """Test creating a transient texture."""
        fg = FrameGraph()
        handle = fg.create_texture(
            "albedo",
            format=ResourceFormat.R8G8B8A8_UNORM,
            width=1920,
            height=1080,
        )

        assert handle is not None
        assert handle.name == "albedo"

    def test_create_buffer(self):
        """Test creating a transient buffer."""
        fg = FrameGraph()
        handle = fg.create_buffer("indirect_args", size_bytes=1024)

        assert handle is not None
        assert handle.name == "indirect_args"

    def test_create_history_texture(self):
        """Test creating a history texture."""
        fg = FrameGraph()
        handle = fg.create_history_texture(
            "taa_history",
            format=ResourceFormat.R16G16B16A16_FLOAT,
            width=1920,
            height=1080,
            double_buffered=True,
        )

        assert handle is not None
        assert handle.name == "taa_history"

    def test_import_external(self):
        """Test importing an external resource."""
        fg = FrameGraph()
        fake_gpu_resource = object()

        handle = fg.import_external(
            "backbuffer",
            gpu_resource=fake_gpu_resource,
            is_backbuffer=True,
            initial_state=ResourceState.PRESENT,
        )

        assert handle is not None
        assert handle.name == "backbuffer"

    def test_get_resource(self):
        """Test retrieving a resource by name."""
        fg = FrameGraph()
        created = fg.create_texture("myTex")

        retrieved = fg.get_resource("myTex")
        assert retrieved is created

    def test_duplicate_resource_name_fails(self):
        """Test that duplicate resource names are rejected."""
        fg = FrameGraph()
        fg.create_texture("duplicate")

        with pytest.raises(ValueError, match="already exists"):
            fg.create_texture("duplicate")


# =============================================================================
# Compilation
# =============================================================================


class TestCompilation:
    """Test frame graph compilation."""

    def test_basic_compilation(self):
        """Test compiling a simple frame graph."""
        fg = FrameGraph()

        # Create resources
        color = fg.create_texture("color")
        depth = fg.create_texture("depth", format=ResourceFormat.D32_FLOAT)

        # Create pass
        gbuffer = fg.add_graphics_pass("GBuffer")
        gbuffer.add_color_attachment(color)
        gbuffer.set_depth_stencil(depth)

        # Mark as having side effects so it's not culled
        gbuffer.set_flag(PassFlags.SIDE_EFFECTS)

        result = fg.compile()

        assert result.success is True
        assert fg.is_compiled is True
        assert "GBuffer" in result.execution_order

    def test_dependency_order(self):
        """Test that dependencies are respected in execution order."""
        fg = FrameGraph()

        # Create resources
        gbuffer = fg.create_texture("gbuffer")
        hdr = fg.create_texture("hdr")
        backbuffer = fg.import_external("backbuffer", None, is_backbuffer=True)

        # GBuffer pass writes gbuffer
        gbuffer_pass = fg.add_graphics_pass("GBuffer")
        gbuffer_pass.add_color_attachment(gbuffer)

        # Lighting reads gbuffer, writes hdr
        lighting = fg.add_compute_pass("Lighting")
        lighting.read_texture(gbuffer)
        lighting.write_texture(hdr)

        # Post writes backbuffer
        post = fg.add_graphics_pass("PostProcess")
        post.read(hdr, ResourceState.SHADER_RESOURCE)
        post.add_color_attachment(backbuffer)

        result = fg.compile()

        assert result.success is True
        order = result.execution_order
        assert order.index("GBuffer") < order.index("Lighting")
        assert order.index("Lighting") < order.index("PostProcess")

    def test_compilation_invalidation(self):
        """Test that modifications invalidate compilation."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass1").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        assert fg.is_compiled is True

        # Adding a pass invalidates
        fg.add_graphics_pass("Pass2")
        assert fg.is_compiled is False

    def test_compile_empty_graph(self):
        """Test compiling an empty frame graph."""
        fg = FrameGraph()
        result = fg.compile()

        assert result.success is True
        assert len(result.execution_order) == 0


# =============================================================================
# Pass Culling
# =============================================================================


class TestPassCulling:
    """Test unused pass culling (dead code elimination)."""

    def test_cull_unused_pass(self):
        """Test that passes with unused outputs are culled."""
        fg = FrameGraph()

        # Create a pass that writes to nothing used
        unused_tex = fg.create_texture("unused")
        unused_pass = fg.add_graphics_pass("Unused")
        unused_pass.add_color_attachment(unused_tex)

        result = fg.compile()

        assert result.success is True
        assert "Unused" in result.culled_passes
        assert "Unused" not in result.execution_order

    def test_no_cull_flag_preserves_pass(self):
        """Test that NO_CULL flag prevents culling."""
        fg = FrameGraph()

        unused_tex = fg.create_texture("unused")
        preserved = fg.add_graphics_pass("Preserved")
        preserved.add_color_attachment(unused_tex)
        preserved.set_flag(PassFlags.NO_CULL)

        result = fg.compile()

        assert "Preserved" not in result.culled_passes
        assert "Preserved" in result.execution_order

    def test_side_effects_prevents_culling(self):
        """Test that SIDE_EFFECTS flag prevents culling."""
        fg = FrameGraph()

        unused_tex = fg.create_texture("unused")
        effects = fg.add_graphics_pass("Effects")
        effects.add_color_attachment(unused_tex)
        effects.set_flag(PassFlags.SIDE_EFFECTS)

        result = fg.compile()

        assert "Effects" not in result.culled_passes
        assert "Effects" in result.execution_order

    def test_backbuffer_write_prevents_culling(self):
        """Test that writing to backbuffer prevents culling."""
        fg = FrameGraph()

        backbuffer = fg.import_external("backbuffer", None, is_backbuffer=True)
        present = fg.add_graphics_pass("Present")
        present.add_color_attachment(backbuffer)

        result = fg.compile()

        assert "Present" not in result.culled_passes
        assert "Present" in result.execution_order

    def test_disable_culling(self):
        """Test disabling pass culling."""
        fg = FrameGraph()
        fg.set_pass_culling_enabled(False)

        unused_tex = fg.create_texture("unused")
        unused_pass = fg.add_graphics_pass("Unused")
        unused_pass.add_color_attachment(unused_tex)

        result = fg.compile()

        assert "Unused" in result.execution_order
        assert len(result.culled_passes) == 0


# =============================================================================
# Execution
# =============================================================================


class TestExecution:
    """Test frame graph execution."""

    def test_execute_requires_compilation(self):
        """Test that execution requires prior compilation."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)

        with pytest.raises(RuntimeError, match="must be compiled"):
            fg.execute(None)

    def test_execute_calls_pass_callbacks(self):
        """Test that execution calls pass callbacks."""
        fg = FrameGraph()
        executed = []

        pass1 = fg.add_graphics_pass("Pass1")
        pass1.set_flag(PassFlags.SIDE_EFFECTS)
        pass1.set_execute(lambda ctx: executed.append("Pass1"))

        pass2 = fg.add_graphics_pass("Pass2")
        pass2.set_flag(PassFlags.SIDE_EFFECTS)
        pass2.set_execute(lambda ctx: executed.append("Pass2"))

        fg.compile()
        fg.execute(None)

        assert "Pass1" in executed
        assert "Pass2" in executed

    def test_execute_respects_order(self):
        """Test that execution respects compiled order."""
        fg = FrameGraph()
        executed = []

        tex1 = fg.create_texture("tex1")
        tex2 = fg.create_texture("tex2")
        backbuffer = fg.import_external("bb", None, is_backbuffer=True)

        pass1 = fg.add_graphics_pass("First")
        pass1.add_color_attachment(tex1)
        pass1.set_execute(lambda ctx: executed.append("First"))

        pass2 = fg.add_compute_pass("Second")
        pass2.read_texture(tex1)
        pass2.write_texture(tex2)
        pass2.set_execute(lambda ctx: executed.append("Second"))

        pass3 = fg.add_graphics_pass("Third")
        pass3.read(tex2, ResourceState.SHADER_RESOURCE)
        pass3.add_color_attachment(backbuffer)
        pass3.set_execute(lambda ctx: executed.append("Third"))

        fg.compile()
        fg.execute(None)

        assert executed == ["First", "Second", "Third"]


# =============================================================================
# Configuration
# =============================================================================


class TestConfiguration:
    """Test frame graph configuration."""

    def test_async_compute_toggle(self):
        """Test enabling/disabling async compute."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        assert fg.is_compiled is True

        fg.set_async_compute_enabled(False)
        assert fg.is_compiled is False

    def test_pass_culling_toggle(self):
        """Test enabling/disabling pass culling."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        assert fg.is_compiled is True

        fg.set_pass_culling_enabled(False)
        assert fg.is_compiled is False

    def test_resource_aliasing_toggle(self):
        """Test enabling/disabling resource aliasing."""
        fg = FrameGraph()
        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        assert fg.is_compiled is True

        fg.set_resource_aliasing_enabled(False)
        assert fg.is_compiled is False


# =============================================================================
# Introspection
# =============================================================================


class TestIntrospection:
    """Test frame graph introspection methods."""

    def test_pass_count(self):
        """Test getting pass count."""
        fg = FrameGraph()
        assert fg.pass_count == 0

        fg.add_graphics_pass("Pass1")
        assert fg.pass_count == 1

        fg.add_compute_pass("Pass2")
        assert fg.pass_count == 2

    def test_resource_count(self):
        """Test getting resource count."""
        fg = FrameGraph()
        assert fg.resource_count == 0

        fg.create_texture("tex1")
        assert fg.resource_count == 1

        fg.create_buffer("buf1", 1024)
        assert fg.resource_count == 2

    def test_get_pass_names(self):
        """Test getting pass names."""
        fg = FrameGraph()
        fg.add_graphics_pass("A")
        fg.add_compute_pass("B")
        fg.add_copy_pass("C")

        names = fg.get_pass_names()
        assert names == ["A", "B", "C"]

    def test_get_execution_order(self):
        """Test getting execution order."""
        fg = FrameGraph()

        # Not compiled yet
        assert fg.get_execution_order() == []

        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        assert "Pass" in fg.get_execution_order()

    def test_get_compilation_result(self):
        """Test getting compilation result."""
        fg = FrameGraph()

        # Not compiled yet
        assert fg.get_compilation_result() is None

        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        result = fg.compile()

        assert fg.get_compilation_result() is result

    def test_clear(self):
        """Test clearing the frame graph."""
        fg = FrameGraph()
        fg.create_texture("tex")
        fg.add_graphics_pass("Pass").set_flag(PassFlags.SIDE_EFFECTS)
        fg.compile()

        fg.clear()

        assert fg.pass_count == 0
        assert fg.resource_count == 0
        assert fg.is_compiled is False


# =============================================================================
# Integration (matches spec example)
# =============================================================================


# =============================================================================
# Input Validation
# =============================================================================


class TestPassValidation:
    """Test input validation for pass configuration."""

    def test_compute_dispatch_negative_size_fails(self):
        """Test that negative dispatch sizes are rejected."""
        fg = FrameGraph()
        compute = fg.add_compute_pass("Compute")

        import pytest
        with pytest.raises(ValueError, match="must be positive"):
            compute.set_dispatch_size(-1, 10, 1)

    def test_compute_dispatch_zero_size_fails(self):
        """Test that zero dispatch sizes are rejected."""
        fg = FrameGraph()
        compute = fg.add_compute_pass("Compute")

        import pytest
        with pytest.raises(ValueError, match="must be positive"):
            compute.set_dispatch_size(10, 0, 1)

    def test_compute_dispatch_valid_size_succeeds(self):
        """Test that positive dispatch sizes are accepted."""
        fg = FrameGraph()
        compute = fg.add_compute_pass("Compute")

        result = compute.set_dispatch_size(8, 8, 1)
        assert result is compute
        assert compute.dispatch_size == (8, 8, 1)

    def test_raytracing_dispatch_negative_fails(self):
        """Test that negative ray dispatch dimensions are rejected."""
        fg = FrameGraph()
        rt = fg.add_raytracing_pass("RT")

        import pytest
        with pytest.raises(ValueError, match="must be positive"):
            rt.set_dispatch_dimensions(-1920, 1080)

    def test_raytracing_dispatch_valid_succeeds(self):
        """Test that positive ray dispatch dimensions are accepted."""
        fg = FrameGraph()
        rt = fg.add_raytracing_pass("RT")

        result = rt.set_dispatch_dimensions(1920, 1080, 1)
        assert result is rt
        assert rt.dispatch_width == 1920
        assert rt.dispatch_height == 1080


# =============================================================================
# Integration (matches spec example)
# =============================================================================


class TestSpecExample:
    """Test the example from RENDERING_CONTEXT.md Section 6.1."""

    def test_spec_frame_graph_example(self):
        """
        Test the conceptual API example from the spec:

        fg = FrameGraph()
        gbuffer = fg.add_pass("GBuffer", type="graphics")
        gbuffer.write(albedo_rt, normal_rt, depth_rt)

        lighting = fg.add_pass("Lighting", type="compute")
        lighting.read(albedo_rt, normal_rt, depth_rt)
        lighting.write(hdr_target)

        postprocess = fg.add_pass("PostProcess", type="graphics")
        postprocess.read(hdr_target)
        postprocess.write(backbuffer)

        fg.compile()
        fg.execute()
        """
        fg = FrameGraph()

        # Create resources
        albedo_rt = fg.create_texture("albedo_rt")
        normal_rt = fg.create_texture("normal_rt")
        depth_rt = fg.create_texture("depth_rt", format=ResourceFormat.D32_FLOAT)
        hdr_target = fg.create_texture(
            "hdr_target", format=ResourceFormat.R16G16B16A16_FLOAT
        )
        backbuffer = fg.import_external("backbuffer", None, is_backbuffer=True)

        # GBuffer pass
        gbuffer = fg.add_pass("GBuffer", pass_type="graphics")
        gbuffer.write(albedo_rt, ResourceState.RENDER_TARGET)
        gbuffer.write(normal_rt, ResourceState.RENDER_TARGET)
        gbuffer.write(depth_rt, ResourceState.DEPTH_WRITE)

        # Lighting pass
        lighting = fg.add_pass("Lighting", pass_type="compute")
        lighting.read(albedo_rt, ResourceState.SHADER_RESOURCE)
        lighting.read(normal_rt, ResourceState.SHADER_RESOURCE)
        lighting.read(depth_rt, ResourceState.SHADER_RESOURCE)
        lighting.write(hdr_target, ResourceState.UNORDERED_ACCESS)

        # PostProcess pass
        postprocess = fg.add_pass("PostProcess", pass_type="graphics")
        postprocess.read(hdr_target, ResourceState.SHADER_RESOURCE)
        postprocess.write(backbuffer, ResourceState.RENDER_TARGET)

        # Compile and verify
        result = fg.compile()
        assert result.success is True

        # Verify execution order
        order = result.execution_order
        assert order.index("GBuffer") < order.index("Lighting")
        assert order.index("Lighting") < order.index("PostProcess")

        # Verify no passes were culled (all feed into backbuffer)
        assert len(result.culled_passes) == 0
