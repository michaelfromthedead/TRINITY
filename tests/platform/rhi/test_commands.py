"""Tests for RHI command recording."""
import pytest
from engine.platform.rhi import (
    CommandList, NullCommandList, Queue, QueueType,
    BufferDesc, BufferUsage, TextureDesc, TextureType, TextureUsage,
    Format, ResourceState, SampleCount,
    ShaderDesc, ShaderStage, GraphicsPipelineDesc, ComputePipelineDesc,
    NullAdapter, NullDevice, DeviceConfig, NullFence
)


@pytest.fixture
def device():
    """Create test device."""
    adapter = NullAdapter()
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


@pytest.fixture
def cmd_list():
    """Create test command list."""
    return NullCommandList()


def test_command_list_begin_end(cmd_list):
    """Test command list begin/end."""
    cmd_list.begin()
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 0  # No commands recorded


def test_command_list_barrier(cmd_list, device):
    """Test barrier recording."""
    buffer = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.STORAGE))

    cmd_list.begin()
    cmd_list.barrier(buffer, ResourceState.COMMON, ResourceState.UNORDERED_ACCESS)
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 1
    assert commands[0].type == "barrier"
    assert commands[0].args["resource"] == buffer
    assert commands[0].args["state_before"] == ResourceState.COMMON
    assert commands[0].args["state_after"] == ResourceState.UNORDERED_ACCESS


def test_command_list_render_pass(cmd_list, device):
    """Test render pass recording."""
    rt_desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=1920,
        height=1080,
        usage=TextureUsage.RENDER_TARGET
    )
    render_target = device.create_texture(rt_desc)

    depth_desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.D32_FLOAT,
        width=1920,
        height=1080,
        usage=TextureUsage.DEPTH_STENCIL
    )
    depth_target = device.create_texture(depth_desc)

    cmd_list.begin()
    cmd_list.begin_render_pass(
        render_targets=[render_target],
        depth_target=depth_target,
        clear_color=(0.0, 0.0, 0.0, 1.0),
        clear_depth=1.0
    )
    cmd_list.end_render_pass()
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 2
    assert commands[0].type == "begin_render_pass"
    assert commands[1].type == "end_render_pass"


def test_command_list_draw(cmd_list, device):
    """Test draw command recording."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
    pipeline_desc = GraphicsPipelineDesc(vertex_shader=vs_desc, pixel_shader=ps_desc)
    pipeline = device.create_graphics_pipeline(pipeline_desc)

    vertex_buffer = device.create_buffer(BufferDesc(size=4096, usage=BufferUsage.VERTEX))

    cmd_list.begin()
    cmd_list.set_pipeline(pipeline)
    cmd_list.set_viewport(0, 0, 1920, 1080, 0.0, 1.0)
    cmd_list.set_scissor(0, 0, 1920, 1080)
    cmd_list.set_vertex_buffer(0, vertex_buffer, 0, 32)
    cmd_list.draw(vertex_count=3, instance_count=1, first_vertex=0, first_instance=0)
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 5
    assert commands[0].type == "set_pipeline"
    assert commands[1].type == "set_viewport"
    assert commands[2].type == "set_scissor"
    assert commands[3].type == "set_vertex_buffer"
    assert commands[4].type == "draw"
    assert commands[4].args["vertex_count"] == 3


def test_command_list_draw_indexed(cmd_list, device):
    """Test indexed draw command recording."""
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
    pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
        vertex_shader=vs_desc, pixel_shader=ps_desc
    ))

    vertex_buffer = device.create_buffer(BufferDesc(size=4096, usage=BufferUsage.VERTEX))
    index_buffer = device.create_buffer(BufferDesc(size=2048, usage=BufferUsage.INDEX))

    cmd_list.begin()
    cmd_list.set_pipeline(pipeline)
    cmd_list.set_vertex_buffer(0, vertex_buffer, 0, 32)
    cmd_list.set_index_buffer(index_buffer, 0, Format.R32_UINT)
    cmd_list.draw_indexed(
        index_count=36,
        instance_count=1,
        first_index=0,
        vertex_offset=0,
        first_instance=0
    )
    cmd_list.end()

    commands = cmd_list.recorded_commands
    draw_cmd = commands[-1]
    assert draw_cmd.type == "draw_indexed"
    assert draw_cmd.args["index_count"] == 36


def test_command_list_dispatch(cmd_list, device):
    """Test compute dispatch recording."""
    cs_desc = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"cs")
    pipeline = device.create_compute_pipeline(ComputePipelineDesc(compute_shader=cs_desc))

    cmd_list.begin()
    cmd_list.set_pipeline(pipeline)
    cmd_list.dispatch(x=16, y=16, z=1)
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 2
    assert commands[1].type == "dispatch"
    assert commands[1].args["x"] == 16
    assert commands[1].args["y"] == 16
    assert commands[1].args["z"] == 1


def test_command_list_copy_buffer(cmd_list, device):
    """Test buffer copy recording."""
    src_buffer = device.create_buffer(BufferDesc(size=2048, usage=BufferUsage.COPY_SRC))
    dst_buffer = device.create_buffer(BufferDesc(size=2048, usage=BufferUsage.COPY_DST))

    cmd_list.begin()
    cmd_list.copy_buffer(dst_buffer, 0, src_buffer, 0, 1024)
    cmd_list.end()

    commands = cmd_list.recorded_commands
    assert len(commands) == 1
    assert commands[0].type == "copy_buffer"
    assert commands[0].args["dst"] == dst_buffer
    assert commands[0].args["src"] == src_buffer
    assert commands[0].args["size"] == 1024


def test_queue_submit(device):
    """Test queue submission."""
    queue = device.get_queue(QueueType.GRAPHICS)
    cmd_list = NullCommandList()

    cmd_list.begin()
    cmd_list.dispatch(1, 1, 1)
    cmd_list.end()

    queue.submit([cmd_list])


def test_queue_submit_with_fence(device):
    """Test queue submission with fence signal."""
    queue = device.get_queue(QueueType.GRAPHICS)
    fence = NullFence.create(device, initial=0)
    cmd_list = NullCommandList()

    cmd_list.begin()
    cmd_list.draw(3, 1, 0, 0)
    cmd_list.end()

    initial_value = fence.value
    queue.submit([cmd_list], signal_fence=fence)

    # Fence should be signaled
    assert fence.value > initial_value


def test_queue_wait_and_signal(device):
    """Test queue wait and signal operations."""
    queue = device.get_queue(QueueType.GRAPHICS)
    fence = NullFence.create(device, initial=0)

    queue.signal(fence)
    assert fence.value > 0

    queue.wait(fence)  # Should not hang


def test_multiple_command_lists_submission(device):
    """Test submitting multiple command lists."""
    queue = device.get_queue(QueueType.GRAPHICS)

    cmd_list1 = NullCommandList()
    cmd_list1.begin()
    cmd_list1.draw(3, 1, 0, 0)
    cmd_list1.end()

    cmd_list2 = NullCommandList()
    cmd_list2.begin()
    cmd_list2.dispatch(1, 1, 1)
    cmd_list2.end()

    queue.submit([cmd_list1, cmd_list2])
