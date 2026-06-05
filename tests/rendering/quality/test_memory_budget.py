"""Tests for T-CC-0.11: Low-tier GPU memory budget manager."""
import pytest
from engine.rendering.quality.memory_budget import (
    TextureFormat,
    BITS_PER_PIXEL,
    MOBILE_FORMATS,
    MemoryBudgetConfig,
    TextureAllocation,
    RenderTargetAllocation,
    BufferAllocation,
    AllocationResult,
    AllocationResponse,
    MemoryStats,
    LRUEvictionPolicy,
    SizeBasedEvictionPolicy,
    MemoryBudgetManager,
    create_low_tier_budget,
    create_medium_tier_budget,
    create_high_tier_budget,
    estimate_mipmap_memory,
    suggest_texture_size,
    power_of_two_size,
)


class TestTextureFormat:
    """Tests for TextureFormat enum and BPP lookup."""

    def test_all_formats_have_bpp(self):
        for fmt in TextureFormat:
            assert fmt in BITS_PER_PIXEL

    def test_mobile_formats_subset(self):
        assert MOBILE_FORMATS.issubset(set(TextureFormat))

    def test_mobile_formats_are_compressed(self):
        for fmt in MOBILE_FORMATS:
            assert BITS_PER_PIXEL[fmt] <= 8

    def test_uncompressed_rgba8_is_32bpp(self):
        assert BITS_PER_PIXEL[TextureFormat.RGBA8] == 32

    def test_etc2_rgb_is_4bpp(self):
        assert BITS_PER_PIXEL[TextureFormat.ETC2_RGB] == 4

    def test_astc_formats_progressively_smaller(self):
        assert BITS_PER_PIXEL[TextureFormat.ASTC_4x4] > BITS_PER_PIXEL[TextureFormat.ASTC_6x6]
        assert BITS_PER_PIXEL[TextureFormat.ASTC_6x6] > BITS_PER_PIXEL[TextureFormat.ASTC_8x8]


class TestMemoryBudgetConfig:
    """Tests for MemoryBudgetConfig defaults."""

    def test_default_values(self):
        cfg = MemoryBudgetConfig()
        assert cfg.max_gpu_memory_mb == 256
        assert cfg.max_texture_size == 1024
        assert cfg.max_render_target_width == 1280
        assert cfg.max_render_target_height == 720
        assert cfg.max_draw_calls_per_frame == 500
        assert cfg.require_compressed_textures is True

    def test_custom_values(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=512, max_texture_size=2048)
        assert cfg.max_gpu_memory_mb == 512
        assert cfg.max_texture_size == 2048

    def test_allowed_formats_default(self):
        cfg = MemoryBudgetConfig()
        assert TextureFormat.ETC2_RGB in cfg.allowed_formats
        assert TextureFormat.ASTC_4x4 in cfg.allowed_formats
        assert TextureFormat.RGBA8 not in cfg.allowed_formats

    def test_config_is_frozen(self):
        cfg = MemoryBudgetConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.max_gpu_memory_mb = 512


class TestTextureAllocation:
    """Tests for TextureAllocation size calculations."""

    def test_rgba8_1024x1024(self):
        alloc = TextureAllocation("test", 1024, 1024, TextureFormat.RGBA8)
        assert alloc.size_bytes == 1024 * 1024 * 4  # 4 bytes per pixel

    def test_etc2_rgb_1024x1024(self):
        alloc = TextureAllocation("test", 1024, 1024, TextureFormat.ETC2_RGB)
        assert alloc.size_bytes == 1024 * 1024 // 2  # 0.5 bytes per pixel

    def test_mipmap_chain_increases_size(self):
        base = TextureAllocation("base", 1024, 1024, TextureFormat.RGBA8, mip_levels=1)
        mipmapped = TextureAllocation("mip", 1024, 1024, TextureFormat.RGBA8, mip_levels=4)
        assert mipmapped.size_bytes > base.size_bytes

    def test_array_layers_multiply_size(self):
        single = TextureAllocation("single", 256, 256, TextureFormat.RGBA8)
        array = TextureAllocation("array", 256, 256, TextureFormat.RGBA8, array_layers=4)
        assert array.size_bytes == single.size_bytes * 4


class TestRenderTargetAllocation:
    """Tests for RenderTargetAllocation size calculations."""

    def test_hd_rgba8(self):
        rt = RenderTargetAllocation("hd", 1280, 720, TextureFormat.RGBA8)
        assert rt.size_bytes == 1280 * 720 * 4

    def test_msaa_4x_multiplies_size(self):
        single = RenderTargetAllocation("single", 1280, 720, TextureFormat.RGBA8, samples=1)
        msaa = RenderTargetAllocation("msaa", 1280, 720, TextureFormat.RGBA8, samples=4)
        assert msaa.size_bytes == single.size_bytes * 4

    def test_hdr_format_larger(self):
        ldr = RenderTargetAllocation("ldr", 1280, 720, TextureFormat.RGBA8)
        hdr = RenderTargetAllocation("hdr", 1280, 720, TextureFormat.RGBA16F)
        assert hdr.size_bytes == ldr.size_bytes * 2


class TestBufferAllocation:
    """Tests for BufferAllocation."""

    def test_basic_allocation(self):
        buf = BufferAllocation("vbo", 1024 * 1024, "vertex")
        assert buf.size_bytes == 1024 * 1024
        assert buf.usage == "vertex"


class TestMemoryStats:
    """Tests for MemoryStats aggregation."""

    def test_total_bytes(self):
        stats = MemoryStats(
            texture_memory_bytes=100,
            render_target_memory_bytes=200,
            buffer_memory_bytes=50,
        )
        assert stats.total_bytes == 350

    def test_total_mb(self):
        stats = MemoryStats(texture_memory_bytes=1024 * 1024)
        assert stats.total_mb == 1.0


class TestLRUEvictionPolicy:
    """Tests for LRU eviction policy."""

    def test_evicts_oldest_first(self):
        policy = LRUEvictionPolicy()
        policy.record_access("a")
        policy.record_access("b")
        policy.record_access("c")

        textures = {
            "a": TextureAllocation("a", 256, 256, TextureFormat.RGBA8),
            "b": TextureAllocation("b", 256, 256, TextureFormat.RGBA8),
            "c": TextureAllocation("c", 256, 256, TextureFormat.RGBA8),
        }

        evict = policy.select_for_eviction(textures, 256 * 256 * 4)
        assert evict[0] == "a"  # oldest

    def test_access_updates_order(self):
        policy = LRUEvictionPolicy()
        policy.record_access("a")
        policy.record_access("b")
        policy.record_access("a")  # a is now newest

        textures = {
            "a": TextureAllocation("a", 256, 256, TextureFormat.RGBA8),
            "b": TextureAllocation("b", 256, 256, TextureFormat.RGBA8),
        }

        evict = policy.select_for_eviction(textures, 256 * 256 * 4)
        assert evict[0] == "b"  # b is now oldest


class TestSizeBasedEvictionPolicy:
    """Tests for size-based eviction policy."""

    def test_evicts_largest_first(self):
        policy = SizeBasedEvictionPolicy()
        textures = {
            "small": TextureAllocation("small", 64, 64, TextureFormat.RGBA8),
            "medium": TextureAllocation("medium", 256, 256, TextureFormat.RGBA8),
            "large": TextureAllocation("large", 512, 512, TextureFormat.RGBA8),
        }

        evict = policy.select_for_eviction(textures, 256 * 256 * 4)
        assert evict[0] == "large"

    def test_evicts_multiple_to_reach_target(self):
        policy = SizeBasedEvictionPolicy()
        textures = {
            "a": TextureAllocation("a", 256, 256, TextureFormat.RGBA8),
            "b": TextureAllocation("b", 256, 256, TextureFormat.RGBA8),
            "c": TextureAllocation("c", 256, 256, TextureFormat.RGBA8),
        }
        needed = 256 * 256 * 4 * 2  # need 2 textures worth
        evict = policy.select_for_eviction(textures, needed)
        assert len(evict) >= 2


class TestMemoryBudgetManager:
    """Tests for MemoryBudgetManager."""

    def test_default_config(self):
        mgr = MemoryBudgetManager()
        assert mgr.config.max_gpu_memory_mb == 256

    def test_custom_config(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=512)
        mgr = MemoryBudgetManager(cfg)
        assert mgr.config.max_gpu_memory_mb == 512

    def test_initial_stats_zero(self):
        mgr = MemoryBudgetManager()
        assert mgr.stats.total_bytes == 0
        assert mgr.stats.texture_count == 0

    def test_available_bytes_starts_full(self):
        mgr = MemoryBudgetManager()
        assert mgr.available_bytes == 256 * 1024 * 1024

    def test_utilization_starts_zero(self):
        mgr = MemoryBudgetManager()
        assert mgr.utilization_percent == 0.0


class TestTextureValidation:
    """Tests for texture validation."""

    def test_valid_texture_passes(self):
        mgr = MemoryBudgetManager()
        resp = mgr.validate_texture(512, 512, TextureFormat.ETC2_RGBA)
        assert resp.result == AllocationResult.SUCCESS

    def test_oversized_texture_fails(self):
        mgr = MemoryBudgetManager()
        resp = mgr.validate_texture(2048, 2048, TextureFormat.ETC2_RGBA)
        assert resp.result == AllocationResult.EXCEEDS_SIZE_LIMIT
        assert resp.suggested_size == (1024, 1024)

    def test_uncompressed_format_fails(self):
        mgr = MemoryBudgetManager()
        resp = mgr.validate_texture(512, 512, TextureFormat.RGBA8)
        assert resp.result == AllocationResult.INVALID_FORMAT
        assert resp.suggested_format is not None

    def test_count_limit_check(self):
        cfg = MemoryBudgetConfig(max_simultaneous_textures=2)
        mgr = MemoryBudgetManager(cfg)
        mgr.allocate_texture("a", 64, 64, TextureFormat.ETC2_RGBA)
        mgr.allocate_texture("b", 64, 64, TextureFormat.ETC2_RGBA)
        resp = mgr.validate_texture(64, 64, TextureFormat.ETC2_RGBA)
        assert resp.result == AllocationResult.EXCEEDS_COUNT_LIMIT


class TestRenderTargetValidation:
    """Tests for render target validation."""

    def test_valid_rt_passes(self):
        mgr = MemoryBudgetManager()
        resp = mgr.validate_render_target(1280, 720, TextureFormat.RGBA8)
        assert resp.result == AllocationResult.SUCCESS

    def test_oversized_rt_fails(self):
        mgr = MemoryBudgetManager()
        resp = mgr.validate_render_target(1920, 1080, TextureFormat.RGBA8)
        assert resp.result == AllocationResult.EXCEEDS_SIZE_LIMIT
        assert resp.suggested_size == (1280, 720)


class TestTextureAllocationManager:
    """Tests for texture allocation in manager."""

    def test_allocate_texture_success(self):
        mgr = MemoryBudgetManager()
        resp = mgr.allocate_texture("diffuse", 512, 512, TextureFormat.ETC2_RGBA)
        assert resp.result == AllocationResult.SUCCESS
        assert mgr.stats.texture_count == 1

    def test_allocate_updates_memory(self):
        mgr = MemoryBudgetManager()
        mgr.allocate_texture("diffuse", 512, 512, TextureFormat.ETC2_RGBA)
        assert mgr.stats.texture_memory_bytes > 0
        assert mgr.available_bytes < 256 * 1024 * 1024

    def test_free_texture_releases_memory(self):
        mgr = MemoryBudgetManager()
        mgr.allocate_texture("diffuse", 512, 512, TextureFormat.ETC2_RGBA)
        initial = mgr.stats.texture_memory_bytes
        mgr.free_texture("diffuse")
        assert mgr.stats.texture_memory_bytes == 0
        assert mgr.stats.texture_count == 0

    def test_get_texture_returns_allocation(self):
        mgr = MemoryBudgetManager()
        mgr.allocate_texture("diffuse", 512, 512, TextureFormat.ETC2_RGBA)
        alloc = mgr.get_texture("diffuse")
        assert alloc is not None
        assert alloc.width == 512

    def test_get_missing_texture_returns_none(self):
        mgr = MemoryBudgetManager()
        assert mgr.get_texture("missing") is None


class TestRenderTargetAllocationManager:
    """Tests for render target allocation in manager."""

    def test_allocate_rt_success(self):
        mgr = MemoryBudgetManager()
        resp = mgr.allocate_render_target("color", 1280, 720, TextureFormat.RGBA8)
        assert resp.result == AllocationResult.SUCCESS
        assert mgr.stats.render_target_count == 1

    def test_allocate_rt_exceeds_budget(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=1)
        mgr = MemoryBudgetManager(cfg)
        resp = mgr.allocate_render_target("huge", 1280, 720, TextureFormat.RGBA16F)
        assert resp.result == AllocationResult.EXCEEDS_BUDGET


class TestBufferAllocationManager:
    """Tests for buffer allocation in manager."""

    def test_allocate_buffer_success(self):
        mgr = MemoryBudgetManager()
        resp = mgr.allocate_buffer("vbo", 1024)
        assert resp.result == AllocationResult.SUCCESS
        assert mgr.stats.buffer_count == 1

    def test_allocate_buffer_exceeds_budget(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=1)
        mgr = MemoryBudgetManager(cfg)
        resp = mgr.allocate_buffer("huge", 2 * 1024 * 1024)
        assert resp.result == AllocationResult.EXCEEDS_BUDGET


class TestEviction:
    """Tests for automatic eviction."""

    def test_auto_evict_on_allocation(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=1)
        mgr = MemoryBudgetManager(cfg)

        # Fill budget
        for i in range(10):
            mgr.allocate_texture(f"tex{i}", 256, 256, TextureFormat.ETC2_RGBA)

        # This should evict old textures
        resp = mgr.allocate_texture("new", 256, 256, TextureFormat.ETC2_RGBA, auto_evict=True)
        assert resp.result == AllocationResult.SUCCESS
        assert "new" in [name for name in mgr._textures]

    def test_eviction_callback_called(self):
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=1)
        mgr = MemoryBudgetManager(cfg)
        evicted = []
        mgr.register_eviction_callback(lambda name: evicted.append(name))

        # Fill and overflow
        for i in range(20):
            mgr.allocate_texture(f"tex{i}", 256, 256, TextureFormat.ETC2_RGBA)

        assert len(evicted) > 0

    def test_no_evict_flag_respects_budget(self):
        # Use very small budget (64KB) so textures actually exceed it
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=0, max_simultaneous_textures=100)
        # 0 MB means 0 bytes budget
        mgr = MemoryBudgetManager(cfg)

        # First texture should fail since budget is 0
        resp = mgr.allocate_texture("new", 256, 256, TextureFormat.ETC2_RGBA, auto_evict=False)
        assert resp.result == AllocationResult.EXCEEDS_BUDGET


class TestDrawCallBudget:
    """Tests for draw call budget tracking."""

    def test_begin_frame_resets_count(self):
        mgr = MemoryBudgetManager()
        mgr.record_draw_call(100)
        mgr.begin_frame()
        assert mgr.draw_calls_this_frame == 0

    def test_record_draw_call(self):
        mgr = MemoryBudgetManager()
        mgr.begin_frame()
        mgr.record_draw_call(10)
        assert mgr.draw_calls_this_frame == 10

    def test_budget_remaining_decreases(self):
        mgr = MemoryBudgetManager()
        mgr.begin_frame()
        initial = mgr.draw_call_budget_remaining
        mgr.record_draw_call(100)
        assert mgr.draw_call_budget_remaining == initial - 100

    def test_over_budget_returns_false(self):
        cfg = MemoryBudgetConfig(max_draw_calls_per_frame=100)
        mgr = MemoryBudgetManager(cfg)
        mgr.begin_frame()
        assert mgr.record_draw_call(50) is True
        assert mgr.record_draw_call(100) is False


class TestAccessTracking:
    """Tests for texture access tracking."""

    def test_access_texture_updates_lru(self):
        mgr = MemoryBudgetManager()
        mgr.allocate_texture("a", 256, 256, TextureFormat.ETC2_RGBA)
        mgr.allocate_texture("b", 256, 256, TextureFormat.ETC2_RGBA)
        mgr.access_texture("a")
        # a is now most recently used

    def test_access_missing_returns_false(self):
        mgr = MemoryBudgetManager()
        assert mgr.access_texture("missing") is False


class TestClearAll:
    """Tests for clear_all method."""

    def test_clear_all_removes_everything(self):
        mgr = MemoryBudgetManager()
        mgr.allocate_texture("tex", 256, 256, TextureFormat.ETC2_RGBA)
        mgr.allocate_render_target("rt", 640, 480, TextureFormat.RGBA8)
        mgr.allocate_buffer("buf", 1024)
        mgr.record_draw_call(50)

        mgr.clear_all()

        assert mgr.stats.total_bytes == 0
        assert mgr.stats.texture_count == 0
        assert mgr.stats.render_target_count == 0
        assert mgr.stats.buffer_count == 0
        assert mgr.draw_calls_this_frame == 0


class TestFactoryFunctions:
    """Tests for tier factory functions."""

    def test_low_tier_budget(self):
        mgr = create_low_tier_budget()
        assert mgr.config.max_gpu_memory_mb == 256
        assert mgr.config.max_texture_size == 1024
        assert mgr.config.require_compressed_textures is True

    def test_medium_tier_budget(self):
        mgr = create_medium_tier_budget()
        assert mgr.config.max_gpu_memory_mb == 512
        assert mgr.config.max_texture_size == 2048
        assert mgr.config.require_compressed_textures is False

    def test_high_tier_budget(self):
        mgr = create_high_tier_budget()
        assert mgr.config.max_gpu_memory_mb == 2048
        assert mgr.config.max_texture_size == 4096


class TestHelperFunctions:
    """Tests for utility functions."""

    def test_estimate_mipmap_memory(self):
        # Full mip chain should be ~1.33x base size
        base = 1024 * 1024 * 4  # 1024x1024 RGBA8
        mip = estimate_mipmap_memory(1024, 1024, TextureFormat.RGBA8)
        assert mip > base
        assert mip < base * 1.5

    def test_suggest_texture_size_within_limit(self):
        w, h = suggest_texture_size(512, 512, 1024)
        assert w == 512
        assert h == 512

    def test_suggest_texture_size_scales_down(self):
        w, h = suggest_texture_size(2048, 2048, 1024, preserve_aspect=True)
        assert w == 1024
        assert h == 1024

    def test_suggest_texture_size_preserves_aspect(self):
        w, h = suggest_texture_size(2048, 1024, 1024, preserve_aspect=True)
        assert w == 1024
        assert h == 512

    def test_suggest_texture_size_no_aspect(self):
        w, h = suggest_texture_size(2048, 1024, 512, preserve_aspect=False)
        assert w == 512
        assert h == 512

    def test_power_of_two_size(self):
        assert power_of_two_size(1) == 1
        assert power_of_two_size(2) == 2
        assert power_of_two_size(3) == 4
        assert power_of_two_size(5) == 8
        assert power_of_two_size(1000) == 1024


class TestIntegration:
    """Integration tests for memory budget system."""

    def test_mobile_game_scenario(self):
        """Simulate mobile game with tight budget."""
        mgr = create_low_tier_budget()

        # Allocate common textures
        mgr.allocate_texture("diffuse_atlas", 1024, 1024, TextureFormat.ASTC_4x4)
        mgr.allocate_texture("normal_atlas", 1024, 1024, TextureFormat.ETC2_RGB)
        mgr.allocate_texture("ui_atlas", 512, 512, TextureFormat.ASTC_4x4)

        # Allocate render targets
        mgr.allocate_render_target("color", 1280, 720, TextureFormat.RGBA8)
        mgr.allocate_render_target("depth", 1280, 720, TextureFormat.R16F)

        # Should still have headroom
        assert mgr.utilization_percent < 50

    def test_budget_exhaustion_and_recovery(self):
        """Test budget limits and eviction recovery."""
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=2)
        mgr = MemoryBudgetManager(cfg)

        # Fill budget
        allocated = 0
        while True:
            resp = mgr.allocate_texture(
                f"tex{allocated}", 256, 256, TextureFormat.ETC2_RGBA, auto_evict=False
            )
            if resp.result != AllocationResult.SUCCESS:
                break
            allocated += 1

        # Now with eviction, allocation should succeed
        resp = mgr.allocate_texture("final", 256, 256, TextureFormat.ETC2_RGBA, auto_evict=True)
        assert resp.result == AllocationResult.SUCCESS

    def test_frame_simulation(self):
        """Simulate multiple frames with draw call tracking."""
        mgr = create_low_tier_budget()

        for frame in range(3):
            mgr.begin_frame()

            # Simulate draws
            for _ in range(100):
                mgr.record_draw_call()

            assert mgr.draw_calls_this_frame == 100
            assert mgr.draw_call_budget_remaining == 400

    def test_texture_streaming_scenario(self):
        """Simulate texture streaming with eviction for memory budget."""
        cfg = MemoryBudgetConfig(max_gpu_memory_mb=1, max_simultaneous_textures=100)
        mgr = MemoryBudgetManager(cfg)

        # Stream in textures - some will be evicted for memory
        for i in range(20):
            resp = mgr.allocate_texture(
                f"stream{i}", 256, 256, TextureFormat.ETC2_RGBA, auto_evict=True
            )
            assert resp.result == AllocationResult.SUCCESS

        # Should stay within memory budget
        assert mgr.stats.texture_memory_bytes <= 1 * 1024 * 1024
