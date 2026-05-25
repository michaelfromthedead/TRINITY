"""Tests for RHI pipeline state."""
import pytest
from engine.platform.rhi import (
    ShaderDesc, ShaderStage,
    GraphicsPipelineDesc, ComputePipelineDesc, RaytracingPipelineDesc,
    PrimitiveTopology, RasterizerState, DepthStencilState, BlendState,
    FillMode, CullMode, BlendFactor, BlendOp,
    Format, CompareOp, PipelineType,
    NullAdapter, NullDevice, DeviceConfig
)


@pytest.fixture
def device():
    """Create test device."""
    adapter = NullAdapter()
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


def test_shader_desc_creation():
    """Test shader descriptor creation."""
    desc = ShaderDesc(
        stage=ShaderStage.VERTEX,
        source=b"vertex_shader_bytecode",
        entry_point="vs_main"
    )

    assert desc.stage == ShaderStage.VERTEX
    assert desc.source == b"vertex_shader_bytecode"
    assert desc.entry_point == "vs_main"


def test_graphics_pipeline_creation(device):
    """Test graphics pipeline creation."""
    vs_desc = ShaderDesc(
        stage=ShaderStage.VERTEX,
        source=b"vs_code",
        entry_point="main"
    )
    ps_desc = ShaderDesc(
        stage=ShaderStage.PIXEL,
        source=b"ps_code",
        entry_point="main"
    )

    pipeline_desc = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        pixel_shader=ps_desc,
        topology=PrimitiveTopology.TRIANGLE_LIST
    )

    pipeline = device.create_graphics_pipeline(pipeline_desc)

    assert pipeline.is_valid()
    assert pipeline.pipeline_type == PipelineType.GRAPHICS
    assert pipeline.handle > 0


def test_compute_pipeline_creation(device):
    """Test compute pipeline creation."""
    cs_desc = ShaderDesc(
        stage=ShaderStage.COMPUTE,
        source=b"compute_kernel",
        entry_point="cs_main"
    )

    pipeline_desc = ComputePipelineDesc(compute_shader=cs_desc)
    pipeline = device.create_compute_pipeline(pipeline_desc)

    assert pipeline.is_valid()
    assert pipeline.pipeline_type == PipelineType.COMPUTE
    assert pipeline.handle > 0


def test_rasterizer_state():
    """Test rasterizer state configuration."""
    raster_state = RasterizerState(
        fill_mode=FillMode.WIREFRAME,
        cull_mode=CullMode.FRONT,
        front_ccw=True,
        depth_bias=100,
        depth_clip=False
    )

    assert raster_state.fill_mode == FillMode.WIREFRAME
    assert raster_state.cull_mode == CullMode.FRONT
    assert raster_state.front_ccw is True
    assert raster_state.depth_bias == 100
    assert raster_state.depth_clip is False


def test_depth_stencil_state():
    """Test depth-stencil state configuration."""
    depth_state = DepthStencilState(
        depth_test=True,
        depth_write=True,
        depth_func=CompareOp.LESS_EQUAL
    )

    assert depth_state.depth_test is True
    assert depth_state.depth_write is True
    assert depth_state.depth_func == CompareOp.LESS_EQUAL


def test_blend_state():
    """Test blend state configuration."""
    blend_state = BlendState(
        enabled=True,
        src_color=BlendFactor.SRC_ALPHA,
        dst_color=BlendFactor.INV_SRC_ALPHA,
        color_op=BlendOp.ADD,
        src_alpha=BlendFactor.ONE,
        dst_alpha=BlendFactor.ZERO,
        alpha_op=BlendOp.ADD
    )

    assert blend_state.enabled is True
    assert blend_state.src_color == BlendFactor.SRC_ALPHA
    assert blend_state.dst_color == BlendFactor.INV_SRC_ALPHA


def test_graphics_pipeline_with_all_states(device):
    """Test graphics pipeline with custom states."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")

    raster = RasterizerState(
        fill_mode=FillMode.SOLID,
        cull_mode=CullMode.BACK
    )

    depth = DepthStencilState(
        depth_test=True,
        depth_write=True,
        depth_func=CompareOp.LESS
    )

    blend = BlendState(enabled=False)

    pipeline_desc = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        pixel_shader=ps_desc,
        topology=PrimitiveTopology.TRIANGLE_STRIP,
        rasterizer=raster,
        depth_stencil=depth,
        blend=blend,
        render_target_formats=[Format.RGBA8_UNORM],
        depth_format=Format.D32_FLOAT
    )

    pipeline = device.create_graphics_pipeline(pipeline_desc)

    assert pipeline.is_valid()
    assert pipeline.desc.topology == PrimitiveTopology.TRIANGLE_STRIP
    assert pipeline.desc.depth_format == Format.D32_FLOAT


def test_geometry_shader_pipeline(device):
    """Test pipeline with geometry shader."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    gs_desc = ShaderDesc(stage=ShaderStage.GEOMETRY, source=b"gs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")

    pipeline_desc = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        geometry_shader=gs_desc,
        pixel_shader=ps_desc
    )

    pipeline = device.create_graphics_pipeline(pipeline_desc)

    assert pipeline.is_valid()


def test_tessellation_pipeline(device):
    """Test pipeline with tessellation shaders."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    hs_desc = ShaderDesc(stage=ShaderStage.HULL, source=b"hs")
    ds_desc = ShaderDesc(stage=ShaderStage.DOMAIN, source=b"ds")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")

    pipeline_desc = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        hull_shader=hs_desc,
        domain_shader=ds_desc,
        pixel_shader=ps_desc
    )

    pipeline = device.create_graphics_pipeline(pipeline_desc)

    assert pipeline.is_valid()


def test_primitive_topologies(device):
    """Test creating pipelines with different primitive topologies."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")

    # Triangle list
    desc_list = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        pixel_shader=ps_desc,
        topology=PrimitiveTopology.TRIANGLE_LIST
    )
    pipeline_list = device.create_graphics_pipeline(desc_list)
    assert pipeline_list.desc.topology == PrimitiveTopology.TRIANGLE_LIST

    # Triangle strip
    desc_strip = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        pixel_shader=ps_desc,
        topology=PrimitiveTopology.TRIANGLE_STRIP
    )
    pipeline_strip = device.create_graphics_pipeline(desc_strip)
    assert pipeline_strip.desc.topology == PrimitiveTopology.TRIANGLE_STRIP

    # Verify they're different
    assert pipeline_list.desc.topology != pipeline_strip.desc.topology


def test_shader_stages(device):
    """Test creating pipelines with different shader stages."""
    # Vertex + Pixel
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
    graphics_desc = GraphicsPipelineDesc(vertex_shader=vs_desc, pixel_shader=ps_desc)
    graphics_pipeline = device.create_graphics_pipeline(graphics_desc)
    assert graphics_pipeline.is_valid()

    # Compute
    cs_desc = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"cs")
    compute_desc = ComputePipelineDesc(compute_shader=cs_desc)
    compute_pipeline = device.create_compute_pipeline(compute_desc)
    assert compute_pipeline.is_valid()

    # Verify different pipeline types
    assert graphics_pipeline.pipeline_type == PipelineType.GRAPHICS
    assert compute_pipeline.pipeline_type == PipelineType.COMPUTE


def test_raytracing_pipeline_desc():
    """Test raytracing pipeline descriptor."""
    rgen_desc = ShaderDesc(stage=ShaderStage.RAY_GENERATION, source=b"rgen")
    miss_desc = ShaderDesc(stage=ShaderStage.MISS, source=b"miss")
    hit_desc = ShaderDesc(stage=ShaderStage.CLOSEST_HIT, source=b"hit")

    rt_desc = RaytracingPipelineDesc(
        ray_gen_shader=rgen_desc,
        miss_shaders=[miss_desc],
        hit_groups=[{"closest_hit": hit_desc}],
        max_recursion_depth=3
    )

    assert rt_desc.max_recursion_depth == 3
    assert len(rt_desc.miss_shaders) == 1
