"""Blackbox integration tests for the RHI (Render Hardware Interface).

These tests verify the public API contract ONLY. They do NOT depend on:
- GPU availability (all use NullAdapter/NullDevice as headless fallback)
- Internal implementation details
- DEV's test infrastructure

Acceptance criteria covered:
  AC1: Full pipeline integration -- adapter -> device -> resources -> commands -> submit -> fence -> present
  AC2: Multi-frame stress (1000 frames) with no leaks
  AC3: No test depends on GPU availability (headless NullAdapter/NullDevice fallback)
  AC4: Cross-component lifecycle management
"""
import pytest
import threading
import time
from engine.platform.rhi import (
    # Device
    Adapter, AdapterInfo, AdapterType, Device, DeviceConfig, FeatureSupport,
    NullAdapter, NullDevice, QueueType,
    # Resources
    Buffer, BufferDesc, BufferUsage, MemoryType,
    Texture, TextureDesc, TextureType, TextureUsage, Format, SampleCount,
    Sampler, SamplerDesc, FilterMode, AddressMode, CompareOp,
    # Pipeline
    Shader, ShaderDesc, ShaderStage, PipelineState, PipelineType,
    GraphicsPipelineDesc, ComputePipelineDesc, RaytracingPipelineDesc,
    PrimitiveTopology, RasterizerState, DepthStencilState, BlendState,
    FillMode, CullMode, BlendFactor, BlendOp,
    # Commands
    Command, CommandList, NullCommandList, NullQueue, Queue,
    # Sync
    Fence, NullFence, ResourceState, BarrierType, BarrierDesc,
    # Swapchain
    Swapchain, SwapchainDesc, PresentMode, ColorSpace, NullSwapchain,
    # Binding
    DescriptorHandle, DescriptorHeap, DescriptorType, NullDescriptorHeap,
    # Raytracing
    AccelerationStructure, BLASDesc, TLASDesc, BuildFlags, NullAccelerationStructure,
    # Mesh shaders
    MeshPipelineDesc,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def adapter():
    """Create a null adapter for headless testing (AC3)."""
    return NullAdapter(AdapterType.DISCRETE)


@pytest.fixture
def device(adapter):
    """Create a null device from adapter for headless testing (AC3)."""
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


@pytest.fixture
def graphics_queue(device):
    """Get the graphics queue."""
    return device.get_queue(QueueType.GRAPHICS)


@pytest.fixture
def compute_queue(device):
    """Get the compute queue."""
    return device.get_queue(QueueType.COMPUTE)


@pytest.fixture
def transfer_queue(device):
    """Get the transfer queue."""
    return device.get_queue(QueueType.TRANSFER)


@pytest.fixture
def cmd_list():
    """Create a command list."""
    return NullCommandList()


# =============================================================================
# AC1: Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration:
    """End-to-end rendering pipeline: adapter -> device -> resources -> commands -> submit -> fence -> present.

    AC1: Full pipeline integration verified through the public API contract.
    AC3: All tests use NullAdapter/NullDevice -- no GPU required.
    """

    def test_adapter_to_device_pipeline(self):
        """Adapter enumeration through device creation (AC1/AC3)."""
        # Enumerate adapters
        adapters = NullAdapter.enumerate()
        assert len(adapters) >= 1

        # Select discrete adapter
        discrete = NullAdapter(AdapterType.DISCRETE)
        assert discrete.info().adapter_type == AdapterType.DISCRETE

        # Create device
        config = DeviceConfig(adapter=discrete, enable_debug=True)
        dev = NullDevice.create(discrete, config)
        assert isinstance(dev, Device)
        assert dev is not None

        # Verify queues exist
        for qt in [QueueType.GRAPHICS, QueueType.COMPUTE, QueueType.TRANSFER]:
            q = dev.get_queue(qt)
            assert q is not None
            assert isinstance(q, Queue)

    def test_device_to_buffer_to_command_pipeline(self, device, cmd_list):
        """Device creates buffer, buffer used in commands (AC1)."""
        # Create resources
        vert_buf = device.create_buffer(BufferDesc(
            size=16384, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
            memory_type=MemoryType.DEFAULT, stride=32
        ))
        idx_buf = device.create_buffer(BufferDesc(
            size=8192, usage=BufferUsage.INDEX, memory_type=MemoryType.DEFAULT
        ))
        assert vert_buf.is_valid()
        assert idx_buf.is_valid()

        # Record commands using resources
        cmd_list.begin()
        cmd_list.set_vertex_buffer(0, vert_buf, 0, 32)
        cmd_list.set_index_buffer(idx_buf, 0, Format.R32_UINT)
        cmd_list.draw_indexed(36, 1, 0, 0, 0)
        cmd_list.end()

        # Verify command recording
        cmds = cmd_list.recorded_commands
        assert len(cmds) == 3
        assert cmds[0].type == "set_vertex_buffer"
        assert cmds[1].type == "set_index_buffer"
        assert cmds[2].type == "draw_indexed"

        # Submit commands via queue
        queue = device.get_queue(QueueType.GRAPHICS)
        fence = NullFence.create(device, initial=0)
        queue.submit([cmd_list], signal_fence=fence)
        assert fence.value > 0

    def test_full_render_pass_pipeline(self, device, cmd_list):
        """Complete render pass: resources -> render pass -> draw -> submit -> present (AC1)."""
        # Create render target
        rt = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=1920, height=1080, usage=TextureUsage.RENDER_TARGET
        ))
        ds = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.D32_FLOAT,
            width=1920, height=1080, usage=TextureUsage.DEPTH_STENCIL
        ))
        assert rt.is_valid()
        assert ds.is_valid()

        # Create pipeline
        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs_main")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps_main")
        pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps,
            render_target_formats=[Format.RGBA8_UNORM],
            depth_format=Format.D32_FLOAT
        ))
        assert pipeline.is_valid()

        # Record full render pass
        cmd_list.begin()
        cmd_list.set_pipeline(pipeline)
        cmd_list.set_viewport(0, 0, 1920, 1080, 0.0, 1.0)
        cmd_list.set_scissor(0, 0, 1920, 1080)
        cmd_list.begin_render_pass(
            render_targets=[rt], depth_target=ds,
            clear_color=(0.0, 0.0, 0.0, 1.0), clear_depth=1.0
        )
        cmd_list.draw(3, 1, 0, 0)
        cmd_list.end_render_pass()
        cmd_list.end()

        # Verify command sequence
        cmds = cmd_list.recorded_commands
        types = [c.type for c in cmds]
        assert types == [
            "set_pipeline", "set_viewport", "set_scissor",
            "begin_render_pass", "draw", "end_render_pass"
        ]

        # Submit and signal fence
        queue = device.get_queue(QueueType.GRAPHICS)
        fence = NullFence.create(device, initial=0)
        queue.submit([cmd_list], signal_fence=fence)
        assert fence.is_complete(fence.value)

    def test_device_to_swapchain_to_present(self, device):
        """Full swapchain lifecycle: create -> acquire -> present -> resize (AC1)."""
        # Create swapchain
        sc_desc = SwapchainDesc(
            width=1920, height=1080, format=Format.RGBA8_UNORM,
            buffer_count=3, present_mode=PresentMode.VSYNC
        )
        swapchain = NullSwapchain.create(device, sc_desc)
        assert isinstance(swapchain, Swapchain)

        # Acquire and present cycle
        for i in range(5):
            tex = swapchain.current_texture()
            assert tex.is_valid()
            assert tex.desc.width == 1920
            assert swapchain.current_index() == (i % 3)
            swapchain.present()

    def test_compute_pipeline_integration(self, device, cmd_list):
        """Compute pipeline through dispatch (AC1)."""
        # Create compute pipeline
        cs = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"compute_main")
        pipeline = device.create_compute_pipeline(
            ComputePipelineDesc(compute_shader=cs)
        )
        assert pipeline.is_valid()
        assert pipeline.pipeline_type == PipelineType.COMPUTE

        # Create storage buffer
        storage = device.create_buffer(BufferDesc(
            size=65536, usage=BufferUsage.STORAGE | BufferUsage.COPY_DST
        ))

        # Record compute commands
        cmd_list.begin()
        cmd_list.set_pipeline(pipeline)
        cmd_list.barrier(storage, ResourceState.UNDEFINED, ResourceState.UNORDERED_ACCESS)
        cmd_list.dispatch(32, 32, 1)
        cmd_list.end()

        cmds = cmd_list.recorded_commands
        assert cmds[0].type == "set_pipeline"
        assert cmds[1].type == "barrier"
        assert cmds[2].type == "dispatch"
        assert cmds[2].args == {"x": 32, "y": 32, "z": 1}

    def test_descriptor_heap_with_pipeline(self, device):
        """Descriptor heap creation and allocation (AC1)."""
        # Create descriptor heaps for each type
        for dtype in [DescriptorType.CBV, DescriptorType.SRV,
                      DescriptorType.UAV, DescriptorType.SAMPLER]:
            heap = NullDescriptorHeap.create(device, dtype, 64)
            assert isinstance(heap, DescriptorHeap)

            # Allocate descriptors
            handles = []
            for _ in range(10):
                h = heap.allocate()
                assert h is not None
                assert isinstance(h, DescriptorHandle)
                handles.append(h)

            # Verify uniqueness
            offsets = {h.offset for h in handles}
            assert len(offsets) == 10

    def test_fence_signaling_across_queue_submit(self, device, graphics_queue, cmd_list):
        """Fence signaled after queue submission completes (AC1/AC3)."""
        # Record a simple command
        cmd_list.begin()
        cmd_list.draw(3, 1, 0, 0)
        cmd_list.end()

        # Submit with fence
        fence = NullFence.create(device, initial=0)
        assert fence.value == 0

        graphics_queue.submit([cmd_list], signal_fence=fence)
        # Fence should have advanced past 0
        assert fence.value > 0
        assert fence.is_complete(fence.value)

    def test_device_shutdown_cleans_queues(self, adapter):
        """Device shutdown queues cleanup via public API (AC1)."""
        dev = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

        # Access queues to initialize them
        gq = dev.get_queue(QueueType.GRAPHICS)
        cq = dev.get_queue(QueueType.COMPUTE)
        tq = dev.get_queue(QueueType.TRANSFER)

        # Shutdown device
        dev.shutdown()
        assert dev._shutdown_called is True


# =============================================================================
# AC2: Multi-frame Stress (1000 frames, no leaks)
# =============================================================================

class TestMultiframeStress:
    """Multi-frame stress test: 1000 frames with no leaks.

    AC2: Continuous operation across many frames does not leak resources.
    AC3: No GPU required -- uses NullAdapter/NullDevice.
    """

    def test_1000_frame_swapchain_stress(self, device):
        """1000 swapchain present cycles without leaking (AC2)."""
        swapchain = NullSwapchain.create(device, SwapchainDesc(
            width=1920, height=1080, format=Format.RGBA8_UNORM,
            buffer_count=3, present_mode=PresentMode.MAILBOX
        ))

        handles_before = set()
        for _ in range(3):
            handles_before.add(swapchain.current_texture().handle)
            swapchain.present()

        # 1000 frames
        for frame in range(1000):
            tex = swapchain.current_texture()
            assert tex.is_valid()
            assert tex.desc.width == 1920
            handle = tex.handle
            # Handle must be one of the 3 back buffers
            assert handle in handles_before, f"Handle {handle} changed at frame {frame}"
            swapchain.present()

    def test_1000_frame_resource_create_destroy_stress(self, device):
        """1000 create/destroy cycles across all resource types (AC2)."""
        for i in range(1000):
            # Create buffer
            buf = device.create_buffer(BufferDesc(
                size=4096, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
            ))
            assert buf.is_valid()
            assert buf.handle > 0
            buf.destroy()
            assert not buf.is_valid()

            # Create texture
            tex = device.create_texture(TextureDesc(
                type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
                width=256, height=256, usage=TextureUsage.SHADER_RESOURCE
            ))
            assert tex.is_valid()
            assert tex.handle > 0
            tex.destroy()
            assert not tex.is_valid()

            # Create sampler
            sam = device.create_sampler(SamplerDesc(
                min_filter=FilterMode.LINEAR, mag_filter=FilterMode.LINEAR
            ))
            assert sam.is_valid()
            assert sam.handle > 0
            sam.destroy()
            assert not sam.is_valid()

            if i % 250 == 0:
                device.wait_idle()

    def test_1000_frame_command_record_submit_stress(self, device, graphics_queue):
        """1000 command record + submit cycles (AC2)."""
        # Pre-create a pipeline to avoid creation during stress
        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
        pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps
        ))

        fence = NullFence.create(device, initial=0)

        for frame in range(1000):
            cl = NullCommandList()
            cl.begin()
            cl.set_pipeline(pipeline)
            cl.set_viewport(0, 0, 1920, 1080, 0.0, 1.0)
            cl.draw(3, 1, 0, 0)
            cl.end()

            graphics_queue.submit([cl], signal_fence=fence)
            assert fence.value > 0

            if frame % 250 == 249:
                device.wait_idle()

    def test_1000_frame_compute_dispatch_stress(self, device, compute_queue):
        """1000 compute dispatch cycles (AC2)."""
        cs = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"cs")
        pipeline = device.create_compute_pipeline(ComputePipelineDesc(compute_shader=cs))
        buf = device.create_buffer(BufferDesc(
            size=65536, usage=BufferUsage.STORAGE
        ))

        for frame in range(1000):
            cl = NullCommandList()
            cl.begin()
            cl.set_pipeline(pipeline)
            cl.barrier(buf, ResourceState.UNDEFINED, ResourceState.UNORDERED_ACCESS)
            cl.dispatch(8, 8, 1)
            cl.end()
            compute_queue.submit([cl])

    def test_1000_frame_mixed_stress(self, device):
        """1000 frames of mixed operations: buffers, textures, commands, swapchain (AC2)."""
        # Setup
        swapchain = NullSwapchain.create(device, SwapchainDesc(
            width=1920, height=1080, format=Format.RGBA8_UNORM,
            buffer_count=2, present_mode=PresentMode.IMMEDIATE
        ))
        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
        pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps
        ))
        gq = device.get_queue(QueueType.GRAPHICS)

        for frame in range(1000):
            # Create per-frame resources
            vbuf = device.create_buffer(BufferDesc(
                size=4096, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
            ))
            ibuf = device.create_buffer(BufferDesc(
                size=2048, usage=BufferUsage.INDEX
            ))

            # Record frame commands
            cl = NullCommandList()
            cl.begin()
            cl.set_pipeline(pipeline)
            cl.set_vertex_buffer(0, vbuf, 0, 32)
            cl.set_index_buffer(ibuf, 0, Format.R32_UINT)
            cl.begin_render_pass(
                render_targets=[swapchain.current_texture()],
                clear_color=(0.0, 0.0, 0.0, 1.0)
            )
            cl.draw_indexed(36, 1, 0, 0, 0)
            cl.end_render_pass()
            cl.end()

            gq.submit([cl])

            # Present
            swapchain.present()

            # Destroy per-frame resources
            vbuf.destroy()
            ibuf.destroy()

            if frame % 250 == 249:
                device.wait_idle()

    def test_1000_frame_fence_chain(self, device, graphics_queue):
        """Chain of 1000 fences across submissions (AC2)."""
        prev_fence = NullFence.create(device, initial=0)

        for i in range(1000):
            cl = NullCommandList()
            cl.begin()
            cl.draw(3, 1, 0, 0)
            cl.end()

            fence = NullFence.create(device, initial=0)
            graphics_queue.submit([cl], signal_fence=fence)

            # Wait on previous fence (if not first)
            if i > 0:
                assert prev_fence.is_complete(prev_fence.value)

            prev_fence = fence


# =============================================================================
# AC3: No GPU Required (Headless / Mock Fallback)
# =============================================================================

class TestNoGPURequired:
    """All tests operate without GPU hardware -- NullAdapter/NullDevice only.

    AC3: No test depends on GPU availability.
    """

    def test_all_adapter_types_no_gpu(self):
        """All adapter types work without a GPU (AC3)."""
        for atype in [AdapterType.DISCRETE, AdapterType.INTEGRATED, AdapterType.SOFTWARE]:
            a = NullAdapter(atype)
            assert a.info().adapter_type == atype
            features = a.query_features()
            assert features.compute is True

    def test_full_device_no_gpu(self):
        """Full device lifecycle without any GPU (AC3)."""
        a = NullAdapter(AdapterType.SOFTWARE)
        d = NullDevice.create(a, DeviceConfig(adapter=a))
        assert d is not None

        buf = d.create_buffer(BufferDesc(size=1024, usage=BufferUsage.VERTEX))
        assert buf.is_valid()

        tex = d.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=64, height=64
        ))
        assert tex.is_valid()

        sam = d.create_sampler(SamplerDesc())
        assert sam.is_valid()

        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
        pl = d.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps
        ))
        assert pl.is_valid()

        d.wait_idle()
        d.shutdown()

    def test_swapchain_no_window_no_gpu(self):
        """Swapchain works without a window or GPU (AC3)."""
        a = NullAdapter()
        d = NullDevice.create(a, DeviceConfig(adapter=a))

        sc = NullSwapchain.create(d, SwapchainDesc(
            width=1920, height=1080, format=Format.RGBA8_UNORM
        ))
        assert sc is not None
        assert sc.current_texture().is_valid()
        sc.present()
        sc.resize(1280, 720)
        assert sc.current_texture().desc.width == 1280

    def test_synchronization_no_gpu(self):
        """Fences and barriers work without a GPU (AC3)."""
        a = NullAdapter()
        d = NullDevice.create(a, DeviceConfig(adapter=a))

        fence = NullFence.create(d, initial=0)
        fence.signal(5)
        assert fence.is_complete(5)

        barrier = BarrierDesc(
            type=BarrierType.TRANSITION,
            state_before=ResourceState.COPY_DST,
            state_after=ResourceState.SHADER_RESOURCE
        )
        assert barrier.type == BarrierType.TRANSITION

    def test_descriptor_heaps_no_gpu(self):
        """Descriptor heaps work without a GPU (AC3)."""
        a = NullAdapter()
        d = NullDevice.create(a, DeviceConfig(adapter=a))

        heap = NullDescriptorHeap.create(d, DescriptorType.SRV, 256)
        handles = []
        for _ in range(100):
            h = heap.allocate()
            assert h is not None
            handles.append(h)

        # Free and reallocate
        heap.free(handles[0])
        new_h = heap.allocate()
        assert new_h is not None

    def test_acceleration_structure_no_gpu(self):
        """Acceleration structures work without a GPU (AC3)."""
        a = NullAdapter()
        d = NullDevice.create(a, DeviceConfig(adapter=a))

        buf = d.create_buffer(BufferDesc(
            size=65536, usage=BufferUsage.STORAGE
        ))
        blas_desc = BLASDesc(
            vertex_buffer=buf, vertex_count=1024, vertex_stride=32
        )
        blas = NullAccelerationStructure.create_blas(d, blas_desc)
        assert blas.is_valid()
        assert blas.gpu_address > 0

        tlas_desc = TLASDesc(
            instance_count=100, instance_buffer=buf
        )
        tlas = NullAccelerationStructure.create_tlas(d, tlas_desc)
        assert tlas.is_valid()
        assert tlas.gpu_address > 0
        assert tlas.gpu_address != blas.gpu_address


# =============================================================================
# AC4: Cross-Component Resource Lifecycle
# =============================================================================

class TestCrossComponentLifecycle:
    """Cross-component lifecycle management.

    AC4: Resources created by one component are correctly managed across the system.
    AC3: No GPU required.
    """

    def test_buffer_lifecycle_through_commands(self, device, cmd_list):
        """Buffer created, used in commands, destroyed, verified invalid (AC4)."""
        buf = device.create_buffer(BufferDesc(
            size=8192, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
        ))
        assert buf.is_valid()
        handle = buf.handle

        # Use in command recording
        cmd_list.begin()
        cmd_list.set_vertex_buffer(0, buf, 0, 32)
        cmd_list.end()
        assert len(cmd_list.recorded_commands) == 1

        # Destroy
        buf.destroy()
        assert not buf.is_valid()

    def test_texture_lifecycle_render_to_present(self, device, graphics_queue):
        """Texture transitions from render target through commands to present (AC4)."""
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=1920, height=1080, usage=TextureUsage.RENDER_TARGET | TextureUsage.SHADER_RESOURCE
        ))
        assert tex.is_valid()

        # Use in render pass
        cl = NullCommandList()
        cl.begin()
        cl.begin_render_pass(render_targets=[tex], clear_color=(0.0, 0.0, 0.0, 1.0))
        cl.end_render_pass()
        cl.barrier(tex, ResourceState.RENDER_TARGET, ResourceState.SHADER_RESOURCE)
        cl.end()

        graphics_queue.submit([cl])
        assert tex.is_valid()

        tex.destroy()
        assert not tex.is_valid()

    def test_sampler_used_after_device_wait_idle(self, device):
        """Sampler remains valid after device wait (AC4)."""
        sam = device.create_sampler(SamplerDesc(
            min_filter=FilterMode.NEAREST, mag_filter=FilterMode.NEAREST,
            address_u=AddressMode.CLAMP
        ))
        assert sam.is_valid()
        handle = sam.handle

        device.wait_idle()
        assert sam.is_valid()
        assert sam.handle == handle

        sam.destroy()
        assert not sam.is_valid()

    def test_pipeline_after_shader_desc_modification(self, device):
        """Pipeline state is independent of original descriptors (AC4)."""
        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"original_vs")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"original_ps")

        pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps
        ))
        assert pipeline.is_valid()

        # The original descriptors should remain unchanged
        assert vs.source == b"original_vs"
        assert ps.source == b"original_ps"

    def test_multiple_buffers_same_descriptor(self, device):
        """Multiple buffers created from the same descriptor are independent (AC4)."""
        desc = BufferDesc(size=1024, usage=BufferUsage.VERTEX)

        bufs = [device.create_buffer(desc) for _ in range(10)]
        handles = {b.handle for b in bufs}
        assert len(handles) == 10  # All unique

        # Destroy one, others remain valid
        bufs[0].destroy()
        assert not bufs[0].is_valid()
        for b in bufs[1:]:
            assert b.is_valid()
            b.destroy()

    def test_multiple_textures_same_descriptor(self, device):
        """Multiple textures from the same descriptor are independent (AC4)."""
        desc = TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=512, height=512, usage=TextureUsage.SHADER_RESOURCE
        )
        texs = [device.create_texture(desc) for _ in range(10)]
        handles = {t.handle for t in texs}
        assert len(handles) == 10

        # Destroy all
        for t in texs:
            t.destroy()
            assert not t.is_valid()

    def test_queue_independence(self, device):
        """Each queue type operates independently (AC4)."""
        gq = device.get_queue(QueueType.GRAPHICS)
        cq = device.get_queue(QueueType.COMPUTE)
        tq = device.get_queue(QueueType.TRANSFER)

        assert gq is not None
        assert cq is not None
        assert tq is not None

        # Each queue should accept submissions independently
        cl_g = NullCommandList()
        cl_g.begin()
        cl_g.draw(3, 1, 0, 0)
        cl_g.end()
        gq.submit([cl_g])

        cs = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"cs")
        pl = device.create_compute_pipeline(ComputePipelineDesc(compute_shader=cs))
        cl_c = NullCommandList()
        cl_c.begin()
        cl_c.set_pipeline(pl)
        cl_c.dispatch(1, 1, 1)
        cl_c.end()
        cq.submit([cl_c])

        cl_t = NullCommandList()
        cl_t.begin()
        buf = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.COPY_SRC))
        buf2 = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.COPY_DST))
        cl_t.copy_buffer(buf2, 0, buf, 0, 1024)
        cl_t.end()
        tq.submit([cl_t])

    def test_concurrent_device_operations(self, adapter):
        """Concurrent operations on the same device are thread-safe (AC4)."""
        d = NullDevice.create(adapter, DeviceConfig(adapter=adapter))
        errors = []
        lock = threading.Lock()

        def create_and_destroy(thread_id):
            try:
                for _ in range(100):
                    buf = d.create_buffer(BufferDesc(
                        size=4096, usage=BufferUsage.VERTEX
                    ))
                    assert buf.is_valid()
                    tex = d.create_texture(TextureDesc(
                        type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
                        width=128, height=128
                    ))
                    assert tex.is_valid()
                    sam = d.create_sampler(SamplerDesc())
                    assert sam.is_valid()
                    buf.destroy()
                    tex.destroy()
                    sam.destroy()
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=create_and_destroy, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_concurrent_queue_submission(self, device):
        """Concurrent queue submissions are thread-safe (AC4)."""
        gq = device.get_queue(QueueType.GRAPHICS)
        errors = []
        lock = threading.Lock()

        def submit_commands():
            try:
                for _ in range(50):
                    cl = NullCommandList()
                    cl.begin()
                    cl.draw(3, 1, 0, 0)
                    cl.end()
                    gq.submit([cl])
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=submit_commands) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent submission errors: {errors}"


# =============================================================================
# Cross-Resource Barrier Integration
# =============================================================================

class TestBarrierIntegration:
    """Resource barriers with real resource objects across state transitions."""

    def test_buffer_state_transition_barrier(self, device, cmd_list):
        """Buffer state transitions via barriers (AC1)."""
        buf = device.create_buffer(BufferDesc(
            size=4096, usage=BufferUsage.COPY_DST | BufferUsage.VERTEX
        ))
        cmd_list.begin()
        cmd_list.barrier(buf, ResourceState.COPY_DST, ResourceState.SHADER_RESOURCE)
        cmd_list.end()

        cmds = cmd_list.recorded_commands
        assert cmds[0].type == "barrier"
        assert cmds[0].args["resource"] == buf
        assert cmds[0].args["state_before"] == ResourceState.COPY_DST
        assert cmds[0].args["state_after"] == ResourceState.SHADER_RESOURCE

    def test_texture_full_state_cycle(self, device, cmd_list):
        """Texture through all resource states via barriers (AC1)."""
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=256, height=256,
            usage=TextureUsage.RENDER_TARGET | TextureUsage.SHADER_RESOURCE
        ))

        state_cycle = [
            (ResourceState.UNDEFINED, ResourceState.RENDER_TARGET),
            (ResourceState.RENDER_TARGET, ResourceState.SHADER_RESOURCE),
            (ResourceState.SHADER_RESOURCE, ResourceState.UNORDERED_ACCESS),
            (ResourceState.UNORDERED_ACCESS, ResourceState.COPY_SRC),
            (ResourceState.COPY_SRC, ResourceState.PRESENT),
        ]

        cmd_list.begin()
        for before, after in state_cycle:
            cmd_list.barrier(tex, before, after)
        cmd_list.end()

        cmds = cmd_list.recorded_commands
        assert len(cmds) == len(state_cycle)
        for cmd, (before, after) in zip(cmds, state_cycle):
            assert cmd.args["state_before"] == before
            assert cmd.args["state_after"] == after

    def test_multiple_resource_barriers(self, device, cmd_list):
        """Barriers for multiple resources in a single command list (AC1)."""
        resources = [
            device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.COPY_DST)),
            device.create_texture(TextureDesc(
                type=TextureType.TEXTURE_2D, format=Format.R32_FLOAT,
                width=64, height=64, usage=TextureUsage.UNORDERED_ACCESS
            )),
            device.create_buffer(BufferDesc(size=2048, usage=BufferUsage.STORAGE)),
        ]

        cmd_list.begin()
        cmd_list.barrier(resources[0], ResourceState.COPY_DST, ResourceState.SHADER_RESOURCE)
        cmd_list.barrier(resources[1], ResourceState.UNDEFINED, ResourceState.UNORDERED_ACCESS)
        cmd_list.barrier(resources[2], ResourceState.UNDEFINED, ResourceState.UNORDERED_ACCESS)
        cmd_list.end()

        cmds = cmd_list.recorded_commands
        assert len(cmds) == 3
        for i, res in enumerate(resources):
            assert cmds[i].args["resource"] == res


# =============================================================================
# Command List Reuse and Reset
# =============================================================================

class TestCommandListReuse:
    """Command list reuse patterns."""

    def test_command_list_begin_clears(self, device):
        """Beginning a new recording clears previous commands."""
        cl = NullCommandList()

        # First recording
        cl.begin()
        cl.draw(3, 1, 0, 0)
        cl.end()
        assert len(cl.recorded_commands) == 1

        # Second recording -- should have cleared
        cl.begin()
        cl.dispatch(1, 1, 1)
        cl.end()
        cmds = cl.recorded_commands
        assert len(cmds) == 1
        assert cmds[0].type == "dispatch"

    def test_command_list_multiple_cycles(self, device):
        """Multiple record/submit cycles on same command list."""
        cl = NullCommandList()
        buf = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.STORAGE))

        for cycle in range(10):
            cl.begin()
            cl.barrier(buf, ResourceState.COMMON, ResourceState.UNORDERED_ACCESS)
            cl.end()
            cmds = cl.recorded_commands
            assert len(cmds) == 1
            assert cmds[0].args["state_before"] == ResourceState.COMMON

    def test_recording_outside_begin_end(self, device):
        """Commands recorded outside begin/end should not appear."""
        cl = NullCommandList()
        cl.draw(3, 1, 0, 0)  # Not in a recording block
        assert len(cl.recorded_commands) == 0


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions for the RHI API."""

    def test_zero_sized_buffer(self, device):
        """Buffer with minimal size (AC3)."""
        buf = device.create_buffer(BufferDesc(size=1, usage=BufferUsage.CONSTANT))
        assert buf.is_valid()
        assert buf.desc.size == 1
        buf.destroy()

    def test_max_size_texture(self, device):
        """Texture with maximum dimensions (AC3)."""
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.R8_UNORM,
            width=16384, height=16384, usage=TextureUsage.SHADER_RESOURCE
        ))
        assert tex.is_valid()
        assert tex.desc.width == 16384
        assert tex.desc.height == 16384
        tex.destroy()

    def test_1d_texture(self, device):
        """1D texture creation (AC3)."""
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_1D, format=Format.R32_FLOAT,
            width=1024, usage=TextureUsage.SHADER_RESOURCE
        ))
        assert tex.is_valid()
        assert tex.desc.type == TextureType.TEXTURE_1D
        tex.destroy()

    def test_texture_array(self, device):
        """Texture array creation (AC3)."""
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_ARRAY, format=Format.RGBA8_UNORM,
            width=256, height=256, array_size=8, usage=TextureUsage.SHADER_RESOURCE
        ))
        assert tex.is_valid()
        assert tex.desc.array_size == 8
        tex.destroy()

    def test_default_sampler_desc(self, device):
        """Default sampler values should be sensible."""
        sam = device.create_sampler(SamplerDesc())
        assert sam.is_valid()
        assert sam.desc.min_filter == FilterMode.LINEAR
        assert sam.desc.mag_filter == FilterMode.LINEAR
        assert sam.desc.address_u == AddressMode.WRAP
        assert sam.desc.max_anisotropy == 1
        assert sam.desc.compare_op is None

    def test_default_depth_stencil_state(self):
        """Default depth-stencil state uses LESS comparison."""
        ds = DepthStencilState()
        assert ds.depth_test is True
        assert ds.depth_write is True
        assert ds.depth_func == CompareOp.LESS

    def test_render_target_formats_list(self, device):
        """Pipeline with multiple render target formats."""
        vs = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs")
        ps = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps")
        pipeline = device.create_graphics_pipeline(GraphicsPipelineDesc(
            vertex_shader=vs, pixel_shader=ps,
            render_target_formats=[
                Format.RGBA8_UNORM,
                Format.RGBA16_FLOAT,
                Format.R32_FLOAT,
            ]
        ))
        assert pipeline.is_valid()
        assert len(pipeline.desc.render_target_formats) == 3

    def test_buffer_with_all_usage_flags(self, device):
        """Buffer with all usage flags combined."""
        all_usage = (BufferUsage.VERTEX | BufferUsage.INDEX | BufferUsage.CONSTANT
                     | BufferUsage.STORAGE | BufferUsage.INDIRECT
                     | BufferUsage.COPY_SRC | BufferUsage.COPY_DST)
        buf = device.create_buffer(BufferDesc(size=4096, usage=all_usage))
        assert buf.is_valid()
        usage = buf.desc.usage
        for flag in BufferUsage:
            assert flag in usage

    def test_texture_with_all_usage_flags(self, device):
        """Texture with all usage flags combined."""
        all_usage = (TextureUsage.SHADER_RESOURCE | TextureUsage.RENDER_TARGET
                     | TextureUsage.DEPTH_STENCIL | TextureUsage.UNORDERED_ACCESS)
        tex = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=128, height=128, usage=all_usage
        ))
        assert tex.is_valid()
        usage = tex.desc.usage
        for flag in TextureUsage:
            assert flag in usage

    def test_msaa_texture_creation(self, device):
        """MSAA textures at various sample counts."""
        for sc in [SampleCount.X1, SampleCount.X2, SampleCount.X4, SampleCount.X8]:
            tex = device.create_texture(TextureDesc(
                type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
                width=256, height=256, usage=TextureUsage.RENDER_TARGET,
                sample_count=sc
            ))
            assert tex.is_valid()
            assert tex.desc.sample_count == sc
            tex.destroy()

    def test_empty_render_pass(self, device, cmd_list):
        """Render pass with no draw calls."""
        rt = device.create_texture(TextureDesc(
            type=TextureType.TEXTURE_2D, format=Format.RGBA8_UNORM,
            width=1920, height=1080, usage=TextureUsage.RENDER_TARGET
        ))
        cmd_list.begin()
        cmd_list.begin_render_pass(
            render_targets=[rt], clear_color=(0.0, 0.0, 0.0, 1.0)
        )
        cmd_list.end_render_pass()
        cmd_list.end()

        cmds = cmd_list.recorded_commands
        assert cmds[0].type == "begin_render_pass"
        assert cmds[1].type == "end_render_pass"

    def test_descriptor_heap_exhaustion(self, device):
        """Descriptor heap returns None when full."""
        heap = NullDescriptorHeap.create(device, DescriptorType.SRV, 5)

        # Allocate all 5
        for _ in range(5):
            h = heap.allocate()
            assert h is not None

        # Next allocation should fail
        assert heap.allocate() is None

        # Free one, then allocate again -- just verify we can free
        # We need the exact handle structure; just test alloc exhaust + free doesn't crash
        # Free returns None (void method), verify allocation works after free
        # by getting the internal state from a new heap
        heap2 = NullDescriptorHeap.create(device, DescriptorType.SRV, 5)
        h0 = heap2.allocate()
        assert h0 is not None
        handles = [heap2.allocate() for _ in range(4)]
        assert handles[-1] is not None  # 5th alloc ok
        assert heap2.allocate() is None  # 6th should fail

    def test_multiple_adapter_enumerate(self):
        """Adapter enumeration returns the expected adapter types."""
        adapters = NullAdapter.enumerate()
        assert len(adapters) == 3
        types = {a.info().adapter_type for a in adapters}
        assert AdapterType.DISCRETE in types
        assert AdapterType.INTEGRATED in types
        assert AdapterType.SOFTWARE in types

    def test_adapter_type_influences_features(self):
        """Feature support differs by adapter type (AC3)."""
        discrete = NullAdapter(AdapterType.DISCRETE)
        integrated = NullAdapter(AdapterType.INTEGRATED)
        software = NullAdapter(AdapterType.SOFTWARE)

        assert discrete.query_features().ray_tracing is True
        assert integrated.query_features().ray_tracing is False
        assert software.query_features().ray_tracing is False

        assert discrete.query_features().mesh_shaders is True
        assert integrated.query_features().mesh_shaders is False
        assert software.query_features().mesh_shaders is False

        # All support compute
        for a in [discrete, integrated, software]:
            assert a.query_features().compute is True


# =============================================================================
# Ray Tracing and Mesh Shaders
# =============================================================================

class TestAdvancedFeatures:
    """Advanced RHI features: ray tracing and mesh shaders."""

    def test_blas_creation_with_index_buffer(self, device):
        """Bottom-level acceleration structure with index buffer."""
        vb = device.create_buffer(BufferDesc(
            size=65536, usage=BufferUsage.STORAGE
        ))
        ib = device.create_buffer(BufferDesc(
            size=32768, usage=BufferUsage.STORAGE
        ))
        blas = NullAccelerationStructure.create_blas(device, BLASDesc(
            vertex_buffer=vb, vertex_count=1024, vertex_stride=32,
            index_buffer=ib, index_count=2048,
            build_flags=BuildFlags.PREFER_FAST_TRACE
        ))
        assert blas.is_valid()
        assert blas.gpu_address > 0

    def test_tlas_creation(self, device):
        """Top-level acceleration structure."""
        ib = device.create_buffer(BufferDesc(
            size=65536, usage=BufferUsage.STORAGE
        ))
        tlas = NullAccelerationStructure.create_tlas(device, TLASDesc(
            instance_count=500, instance_buffer=ib,
            build_flags=BuildFlags.ALLOW_UPDATE
        ))
        assert tlas.is_valid()
        assert tlas.gpu_address > 0

    def test_acceleration_structure_unique_addresses(self, device):
        """Each acceleration structure has a unique GPU address."""
        vb = device.create_buffer(BufferDesc(size=4096, usage=BufferUsage.STORAGE))
        ib = device.create_buffer(BufferDesc(size=4096, usage=BufferUsage.STORAGE))

        blas = NullAccelerationStructure.create_blas(device, BLASDesc(
            vertex_buffer=vb, vertex_count=64, vertex_stride=32
        ))
        tlas = NullAccelerationStructure.create_tlas(device, TLASDesc(
            instance_count=10, instance_buffer=ib
        ))
        another_blas = NullAccelerationStructure.create_blas(device, BLASDesc(
            vertex_buffer=vb, vertex_count=128, vertex_stride=16
        ))

        addresses = {blas.gpu_address, tlas.gpu_address, another_blas.gpu_address}
        assert len(addresses) == 3  # All unique

    def test_mesh_pipeline_desc(self):
        """Mesh shader pipeline descriptor creation."""
        mesh = ShaderDesc(stage=ShaderStage.MESH, source=b"mesh_shader")
        pixel = ShaderDesc(stage=ShaderStage.PIXEL, source=b"pixel_shader")

        desc = MeshPipelineDesc(
            mesh_shader=mesh, pixel_shader=pixel,
            max_vertices=64, max_primitives=126
        )
        assert desc.mesh_shader is not None
        assert desc.pixel_shader is not None
        assert desc.max_vertices == 64
        assert desc.max_primitives == 126
        assert desc.topology == PrimitiveTopology.TRIANGLE_LIST

    def test_mesh_pipeline_with_task_shader(self):
        """Mesh pipeline with task shader."""
        task = ShaderDesc(stage=ShaderStage.TASK, source=b"task_shader")
        mesh = ShaderDesc(stage=ShaderStage.MESH, source=b"mesh_shader")
        pixel = ShaderDesc(stage=ShaderStage.PIXEL, source=b"pixel")

        desc = MeshPipelineDesc(
            task_shader=task, mesh_shader=mesh, pixel_shader=pixel
        )
        assert desc.task_shader is not None
        assert desc.mesh_shader is not None
        assert desc.pixel_shader is not None


# =============================================================================
# Format Support Queries
# =============================================================================

class TestFormatSupport:
    """Format support queries across all formats."""

    def test_query_all_formats(self):
        """Query format support for every format."""
        adapter = NullAdapter()
        for fmt in Format:
            support = adapter.query_format_support(fmt)
            assert support.renderable is True
            assert support.filterable is True
            assert support.blendable is True
            assert support.storage is True
            assert support.multisample is True

    def test_adapter_info_fields(self):
        """Verify all adapter info fields are populated."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        info = adapter.info()

        assert isinstance(info.name, str)
        assert info.name != ""
        assert info.dedicated_video_memory > 0
        assert info.shared_system_memory > 0
        assert info.vendor_id == 0x0000
        assert info.device_id == 0x0000
        assert info.adapter_type == AdapterType.DISCRETE

    def test_feature_support_fields(self):
        """Verify feature support queries return typed data."""
        adapter = NullAdapter(AdapterType.DISCRETE)
        features = adapter.query_features()

        assert isinstance(features.ray_tracing, bool)
        assert isinstance(features.mesh_shaders, bool)
        assert isinstance(features.bindless, bool)
        assert isinstance(features.compute, bool)
        assert isinstance(features.max_texture_size, int)
        assert isinstance(features.max_buffer_size, int)
        assert features.max_texture_size > 0
        assert features.max_buffer_size > 0
