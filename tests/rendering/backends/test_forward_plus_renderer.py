"""Tests for Forward+ renderer (T-CC-0.10)."""

import pytest

from trinity.types import QualityTier
from engine.rendering.backends.forward_plus_renderer import (
    ForwardPassType,
    ToneMapOperator,
    ForwardPlusConfig,
    LightTile,
    ForwardPlusPass,
    ForwardPlusStats,
    LightData,
    ForwardPlusRenderer,
    create_forward_plus_for_tier,
    get_tier_max_lights,
    MAX_LIGHTS_LOW_TIER,
    MAX_LIGHTS_MEDIUM_TIER,
    MAX_LIGHTS_HIGH_TIER,
    TILE_SIZE,
    MAX_LIGHTS_PER_TILE,
)


class TestForwardPassType:
    """Test ForwardPassType enum."""

    def test_pass_types_exist(self):
        """Test all pass types are defined."""
        assert ForwardPassType.DEPTH_PREPASS is not None
        assert ForwardPassType.LIGHT_CULL is not None
        assert ForwardPassType.FORWARD_SHADE is not None
        assert ForwardPassType.TONEMAP is not None

    def test_pass_types_unique(self):
        """Test pass types have unique values."""
        values = [
            ForwardPassType.DEPTH_PREPASS.value,
            ForwardPassType.LIGHT_CULL.value,
            ForwardPassType.FORWARD_SHADE.value,
            ForwardPassType.TONEMAP.value,
        ]
        assert len(values) == len(set(values))


class TestToneMapOperator:
    """Test ToneMapOperator enum."""

    def test_operators_exist(self):
        """Test all operators are defined."""
        assert ToneMapOperator.REINHARD is not None
        assert ToneMapOperator.REINHARD_EXTENDED is not None
        assert ToneMapOperator.ACES is not None
        assert ToneMapOperator.UNCHARTED2 is not None
        assert ToneMapOperator.NONE is not None


class TestForwardPlusConfig:
    """Test ForwardPlusConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ForwardPlusConfig()
        assert config.max_lights == MAX_LIGHTS_LOW_TIER
        assert config.tile_size == TILE_SIZE
        assert config.max_lights_per_tile == MAX_LIGHTS_PER_TILE
        assert config.enable_depth_prepass is True
        assert config.enable_light_culling is True
        assert config.tonemap_enabled is True
        assert config.tonemap_operator == ToneMapOperator.REINHARD
        assert config.exposure == 1.0
        assert config.quality_tier == QualityTier.LOW

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ForwardPlusConfig(
            max_lights=32,
            tile_size=32,
            max_lights_per_tile=64,
            enable_depth_prepass=False,
            tonemap_operator=ToneMapOperator.ACES,
            exposure=2.0,
            quality_tier=QualityTier.MEDIUM,
        )
        assert config.max_lights == 32
        assert config.tile_size == 32
        assert config.max_lights_per_tile == 64
        assert config.enable_depth_prepass is False
        assert config.tonemap_operator == ToneMapOperator.ACES
        assert config.exposure == 2.0
        assert config.quality_tier == QualityTier.MEDIUM

    def test_config_validation_negative_lights(self):
        """Test config rejects negative max_lights."""
        with pytest.raises(ValueError, match="max_lights must be non-negative"):
            ForwardPlusConfig(max_lights=-1)

    def test_config_validation_zero_tile_size(self):
        """Test config rejects zero tile_size."""
        with pytest.raises(ValueError, match="tile_size must be positive"):
            ForwardPlusConfig(tile_size=0)

    def test_config_validation_zero_lights_per_tile(self):
        """Test config rejects zero max_lights_per_tile."""
        with pytest.raises(ValueError, match="max_lights_per_tile must be positive"):
            ForwardPlusConfig(max_lights_per_tile=0)

    def test_config_validation_negative_exposure(self):
        """Test config rejects non-positive exposure."""
        with pytest.raises(ValueError, match="exposure must be positive"):
            ForwardPlusConfig(exposure=0.0)

        with pytest.raises(ValueError, match="exposure must be positive"):
            ForwardPlusConfig(exposure=-1.0)

    def test_config_allows_zero_lights(self):
        """Test config allows zero max_lights (no lighting scenario)."""
        config = ForwardPlusConfig(max_lights=0)
        assert config.max_lights == 0


class TestLightTile:
    """Test LightTile dataclass."""

    def test_tile_creation(self):
        """Test tile creation with coordinates."""
        tile = LightTile(x=5, y=10)
        assert tile.x == 5
        assert tile.y == 10
        assert tile.light_count == 0
        assert tile.light_indices == []

    def test_tile_clear(self):
        """Test clearing tile data."""
        tile = LightTile(x=0, y=0, light_count=3, light_indices=[0, 1, 2])
        tile.clear()
        assert tile.light_count == 0
        assert tile.light_indices == []

    def test_tile_add_light(self):
        """Test adding lights to tile."""
        tile = LightTile(x=0, y=0)

        result = tile.add_light(0, max_per_tile=4)
        assert result is True
        assert tile.light_count == 1
        assert tile.light_indices == [0]

        result = tile.add_light(1, max_per_tile=4)
        assert result is True
        assert tile.light_count == 2

    def test_tile_add_light_max_reached(self):
        """Test adding light when tile is full."""
        tile = LightTile(x=0, y=0)
        tile.add_light(0, max_per_tile=2)
        tile.add_light(1, max_per_tile=2)

        result = tile.add_light(2, max_per_tile=2)
        assert result is False
        assert tile.light_count == 2


class TestForwardPlusPass:
    """Test ForwardPlusPass dataclass."""

    def test_pass_creation(self):
        """Test pass creation."""
        render_pass = ForwardPlusPass(
            pass_type=ForwardPassType.DEPTH_PREPASS,
            name="depth_prepass",
            enabled=True,
        )
        assert render_pass.pass_type == ForwardPassType.DEPTH_PREPASS
        assert render_pass.name == "depth_prepass"
        assert render_pass.enabled is True

    def test_pass_disabled(self):
        """Test disabled pass."""
        render_pass = ForwardPlusPass(
            pass_type=ForwardPassType.TONEMAP,
            name="tonemap",
            enabled=False,
        )
        assert render_pass.enabled is False


class TestForwardPlusStats:
    """Test ForwardPlusStats dataclass."""

    def test_stats_defaults(self):
        """Test default statistics values."""
        stats = ForwardPlusStats()
        assert stats.depth_prepass_time_ms == 0.0
        assert stats.light_cull_time_ms == 0.0
        assert stats.forward_shade_time_ms == 0.0
        assert stats.tonemap_time_ms == 0.0
        assert stats.total_time_ms == 0.0
        assert stats.visible_lights == 0
        assert stats.tiles_with_lights == 0
        assert stats.total_tiles == 0
        assert stats.draw_calls == 0

    def test_stats_reset(self):
        """Test resetting statistics."""
        stats = ForwardPlusStats(
            depth_prepass_time_ms=1.0,
            light_cull_time_ms=2.0,
            visible_lights=10,
            draw_calls=50,
        )
        stats.reset()
        assert stats.depth_prepass_time_ms == 0.0
        assert stats.visible_lights == 0
        assert stats.draw_calls == 0

    def test_average_lights_per_tile(self):
        """Test average lights per tile calculation."""
        stats = ForwardPlusStats(
            visible_lights=100,
            tiles_with_lights=20,
        )
        assert stats.average_lights_per_tile == 5.0

    def test_average_lights_per_tile_zero(self):
        """Test average with zero tiles."""
        stats = ForwardPlusStats(visible_lights=10, tiles_with_lights=0)
        assert stats.average_lights_per_tile == 0.0


class TestLightData:
    """Test LightData dataclass."""

    def test_light_data_defaults(self):
        """Test default light data."""
        light = LightData()
        assert light.position == (0.0, 0.0, 0.0)
        assert light.radius == 10.0
        assert light.color == (1.0, 1.0, 1.0)
        assert light.intensity == 1.0
        assert light.light_type == 0

    def test_light_data_custom(self):
        """Test custom light data."""
        light = LightData(
            position=(1.0, 2.0, 3.0),
            radius=25.0,
            color=(1.0, 0.5, 0.0),
            intensity=2.0,
            light_type=1,
        )
        assert light.position == (1.0, 2.0, 3.0)
        assert light.radius == 25.0
        assert light.color == (1.0, 0.5, 0.0)
        assert light.intensity == 2.0
        assert light.light_type == 1


class TestForwardPlusRendererCreation:
    """Test ForwardPlusRenderer creation."""

    def test_renderer_default_creation(self):
        """Test default renderer creation."""
        renderer = ForwardPlusRenderer()
        assert renderer.config.quality_tier == QualityTier.LOW
        assert renderer.config.max_lights == MAX_LIGHTS_LOW_TIER
        assert renderer.is_initialized is False

    def test_renderer_with_config(self):
        """Test renderer with custom config."""
        config = ForwardPlusConfig(max_lights=64, quality_tier=QualityTier.MEDIUM)
        renderer = ForwardPlusRenderer(config=config)
        assert renderer.config.max_lights == 64
        assert renderer.config.quality_tier == QualityTier.MEDIUM


class TestForwardPlusRendererInitialization:
    """Test ForwardPlusRenderer initialization."""

    def test_initialize_valid(self):
        """Test valid initialization."""
        renderer = ForwardPlusRenderer()
        result = renderer.initialize(1920, 1080)
        assert result is True
        assert renderer.is_initialized is True
        assert renderer.width == 1920
        assert renderer.height == 1080

    def test_initialize_invalid_width(self):
        """Test initialization with invalid width."""
        renderer = ForwardPlusRenderer()
        result = renderer.initialize(0, 1080)
        assert result is False
        assert renderer.is_initialized is False

    def test_initialize_invalid_height(self):
        """Test initialization with invalid height."""
        renderer = ForwardPlusRenderer()
        result = renderer.initialize(1920, 0)
        assert result is False
        assert renderer.is_initialized is False

    def test_initialize_negative_dimensions(self):
        """Test initialization with negative dimensions."""
        renderer = ForwardPlusRenderer()
        result = renderer.initialize(-100, -100)
        assert result is False

    def test_initialize_double_init(self):
        """Test double initialization raises error."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1920, 1080)
        with pytest.raises(RuntimeError, match="already initialized"):
            renderer.initialize(1920, 1080)


class TestForwardPlusRendererTiles:
    """Test ForwardPlusRenderer tile management."""

    def test_tile_count(self):
        """Test tile count calculation."""
        config = ForwardPlusConfig(tile_size=16)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(1920, 1080)

        tiles_x, tiles_y = renderer.tile_count
        assert tiles_x == 120  # 1920 / 16 = 120
        assert tiles_y == 68   # ceil(1080 / 16) = 68

    def test_tile_count_uninitialized(self):
        """Test tile count before initialization."""
        renderer = ForwardPlusRenderer()
        assert renderer.tile_count == (0, 0)

    def test_tile_count_non_divisible(self):
        """Test tile count with non-divisible dimensions."""
        config = ForwardPlusConfig(tile_size=32)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(100, 100)

        tiles_x, tiles_y = renderer.tile_count
        assert tiles_x == 4  # ceil(100 / 32) = 4
        assert tiles_y == 4

    def test_get_tile(self):
        """Test getting tile by grid coordinates."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)

        tile = renderer.get_tile(0, 0)
        assert tile is not None
        assert tile.x == 0
        assert tile.y == 0

    def test_get_tile_out_of_bounds(self):
        """Test getting tile out of bounds."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)

        tile = renderer.get_tile(1000, 1000)
        assert tile is None

    def test_get_tile_at_pixel(self):
        """Test getting tile at pixel coordinate."""
        config = ForwardPlusConfig(tile_size=16)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(640, 480)

        tile = renderer.get_tile_at_pixel(20, 20)
        assert tile is not None
        assert tile.x == 1  # 20 // 16 = 1
        assert tile.y == 1

    def test_get_tile_at_pixel_out_of_bounds(self):
        """Test getting tile at pixel out of bounds."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)

        tile = renderer.get_tile_at_pixel(1000, 1000)
        assert tile is None

    def test_get_tile_at_pixel_uninitialized(self):
        """Test getting tile at pixel before initialization."""
        renderer = ForwardPlusRenderer()
        tile = renderer.get_tile_at_pixel(100, 100)
        assert tile is None


class TestForwardPlusRendererResize:
    """Test ForwardPlusRenderer resize."""

    def test_resize_after_init(self):
        """Test resizing after initialization."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1920, 1080)
        result = renderer.resize(1280, 720)

        assert result is True
        assert renderer.width == 1280
        assert renderer.height == 720

    def test_resize_before_init(self):
        """Test resizing before initialization (auto-initializes)."""
        renderer = ForwardPlusRenderer()
        result = renderer.resize(1920, 1080)

        assert result is True
        assert renderer.is_initialized is True

    def test_resize_invalid_dimensions(self):
        """Test resizing to invalid dimensions."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1920, 1080)
        result = renderer.resize(0, 720)

        assert result is False
        # Dimensions should not have changed
        assert renderer.width == 1920


class TestForwardPlusRendererPasses:
    """Test ForwardPlusRenderer pass management."""

    def test_get_enabled_passes(self):
        """Test getting enabled passes."""
        renderer = ForwardPlusRenderer()
        passes = renderer.get_enabled_passes()

        # Default: depth, cull, shade, tonemap all enabled
        assert len(passes) == 4

    def test_get_enabled_passes_custom_config(self):
        """Test enabled passes with custom config."""
        config = ForwardPlusConfig(
            enable_depth_prepass=False,
            enable_light_culling=False,
            tonemap_enabled=False,
        )
        renderer = ForwardPlusRenderer(config=config)
        passes = renderer.get_enabled_passes()

        # Only forward shade should be enabled
        assert len(passes) == 1
        assert passes[0].pass_type == ForwardPassType.FORWARD_SHADE

    def test_is_pass_enabled(self):
        """Test checking if pass is enabled."""
        renderer = ForwardPlusRenderer()
        assert renderer.is_pass_enabled(ForwardPassType.DEPTH_PREPASS) is True
        assert renderer.is_pass_enabled(ForwardPassType.FORWARD_SHADE) is True

    def test_set_pass_enabled(self):
        """Test enabling/disabling passes."""
        renderer = ForwardPlusRenderer()

        renderer.set_pass_enabled(ForwardPassType.DEPTH_PREPASS, False)
        assert renderer.is_pass_enabled(ForwardPassType.DEPTH_PREPASS) is False

        renderer.set_pass_enabled(ForwardPassType.DEPTH_PREPASS, True)
        assert renderer.is_pass_enabled(ForwardPassType.DEPTH_PREPASS) is True


class TestForwardPlusRendererLightCulling:
    """Test ForwardPlusRenderer light culling."""

    def test_light_culling_no_lights(self):
        """Test light culling with no lights."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.begin_frame()
        renderer.execute_light_culling([])

        assert renderer.stats.visible_lights == 0
        assert renderer.stats.tiles_with_lights == 0

    def test_light_culling_with_lights(self):
        """Test light culling with lights."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.begin_frame()

        lights = [
            LightData(position=(0.0, 0.0, 0.0)),
            LightData(position=(10.0, 0.0, 0.0)),
        ]
        renderer.execute_light_culling(lights)

        assert renderer.stats.visible_lights == 2
        assert renderer.stats.tiles_with_lights > 0

    def test_light_culling_max_lights(self):
        """Test light culling respects max_lights."""
        config = ForwardPlusConfig(max_lights=2)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(640, 480)
        renderer.begin_frame()

        lights = [LightData() for _ in range(10)]
        renderer.execute_light_culling(lights)

        assert renderer.stats.visible_lights == 2

    def test_light_culling_disabled(self):
        """Test light culling when disabled."""
        config = ForwardPlusConfig(enable_light_culling=False)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(640, 480)
        renderer.begin_frame()

        lights = [LightData() for _ in range(5)]
        renderer.execute_light_culling(lights)

        # Stats should not be updated
        assert renderer.stats.visible_lights == 0


class TestForwardPlusRendererFrameLifecycle:
    """Test ForwardPlusRenderer frame lifecycle."""

    def test_begin_frame(self):
        """Test begin frame."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.begin_frame()

        assert renderer.stats.total_tiles > 0

    def test_begin_frame_not_initialized(self):
        """Test begin frame before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.begin_frame()

    def test_end_frame(self):
        """Test end frame."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.begin_frame()
        renderer.end_frame()

        assert renderer.frame_count == 1

    def test_end_frame_not_initialized(self):
        """Test end frame before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.end_frame()

    def test_full_frame_lifecycle(self):
        """Test full frame lifecycle."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1280, 720)

        # Frame 1
        renderer.begin_frame()
        renderer.execute_depth_prepass([])
        renderer.execute_light_culling([LightData()])
        renderer.execute_forward_shading([])
        renderer.execute_tonemapping()
        renderer.end_frame()

        assert renderer.frame_count == 1
        assert renderer.stats.visible_lights == 1

        # Frame 2
        renderer.begin_frame()
        renderer.end_frame()

        assert renderer.frame_count == 2


class TestForwardPlusRendererTonemap:
    """Test ForwardPlusRenderer tone mapping."""

    def test_set_exposure(self):
        """Test setting exposure."""
        renderer = ForwardPlusRenderer()
        renderer.set_exposure(2.0)
        assert renderer.config.exposure == 2.0

    def test_set_exposure_invalid(self):
        """Test setting invalid exposure."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(ValueError, match="exposure must be positive"):
            renderer.set_exposure(0.0)

        with pytest.raises(ValueError, match="exposure must be positive"):
            renderer.set_exposure(-1.0)

    def test_set_tonemap_operator(self):
        """Test setting tone map operator."""
        renderer = ForwardPlusRenderer()
        renderer.set_tonemap_operator(ToneMapOperator.ACES)
        assert renderer.config.tonemap_operator == ToneMapOperator.ACES


class TestForwardPlusRendererDestroy:
    """Test ForwardPlusRenderer destruction."""

    def test_destroy(self):
        """Test destroying renderer."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.destroy()

        assert renderer.is_initialized is False
        assert renderer.width == 0
        assert renderer.height == 0

    def test_destroy_clears_tiles(self):
        """Test destroy clears tiles."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)
        renderer.destroy()

        assert renderer.tile_count == (0, 0)


class TestCreateForwardPlusForTier:
    """Test create_forward_plus_for_tier factory function."""

    def test_create_low_tier(self):
        """Test creating Low tier renderer."""
        renderer = create_forward_plus_for_tier(QualityTier.LOW)
        assert renderer.config.quality_tier == QualityTier.LOW
        assert renderer.config.max_lights == MAX_LIGHTS_LOW_TIER
        assert renderer.config.tonemap_operator == ToneMapOperator.REINHARD

    def test_create_medium_tier(self):
        """Test creating Medium tier renderer."""
        renderer = create_forward_plus_for_tier(QualityTier.MEDIUM)
        assert renderer.config.quality_tier == QualityTier.MEDIUM
        assert renderer.config.max_lights == MAX_LIGHTS_MEDIUM_TIER
        assert renderer.config.tonemap_operator == ToneMapOperator.ACES

    def test_create_high_tier(self):
        """Test creating High tier renderer."""
        renderer = create_forward_plus_for_tier(QualityTier.HIGH)
        assert renderer.config.quality_tier == QualityTier.HIGH
        assert renderer.config.max_lights == MAX_LIGHTS_HIGH_TIER

    def test_create_ultra_tier(self):
        """Test creating Ultra tier renderer."""
        renderer = create_forward_plus_for_tier(QualityTier.ULTRA)
        assert renderer.config.quality_tier == QualityTier.ULTRA
        assert renderer.config.max_lights == MAX_LIGHTS_HIGH_TIER


class TestGetTierMaxLights:
    """Test get_tier_max_lights helper function."""

    def test_low_tier_max_lights(self):
        """Test Low tier max lights."""
        assert get_tier_max_lights(QualityTier.LOW) == MAX_LIGHTS_LOW_TIER

    def test_medium_tier_max_lights(self):
        """Test Medium tier max lights."""
        assert get_tier_max_lights(QualityTier.MEDIUM) == MAX_LIGHTS_MEDIUM_TIER

    def test_high_tier_max_lights(self):
        """Test High tier max lights."""
        assert get_tier_max_lights(QualityTier.HIGH) == MAX_LIGHTS_HIGH_TIER

    def test_ultra_tier_max_lights(self):
        """Test Ultra tier max lights."""
        assert get_tier_max_lights(QualityTier.ULTRA) == MAX_LIGHTS_HIGH_TIER


class TestForwardPlusRendererExecution:
    """Test ForwardPlusRenderer execution methods."""

    def test_execute_depth_prepass_not_initialized(self):
        """Test depth prepass before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.execute_depth_prepass([])

    def test_execute_light_culling_not_initialized(self):
        """Test light culling before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.execute_light_culling([])

    def test_execute_forward_shading_not_initialized(self):
        """Test forward shading before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.execute_forward_shading([])

    def test_execute_tonemapping_not_initialized(self):
        """Test tonemapping before initialization."""
        renderer = ForwardPlusRenderer()
        with pytest.raises(RuntimeError, match="not initialized"):
            renderer.execute_tonemapping()

    def test_execute_depth_prepass_disabled(self):
        """Test depth prepass when disabled."""
        config = ForwardPlusConfig(enable_depth_prepass=False)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(640, 480)
        renderer.begin_frame()
        renderer.execute_depth_prepass([1, 2, 3])

        # Should not update draw calls
        assert renderer.stats.draw_calls == 0

    def test_execute_tonemapping_disabled(self):
        """Test tonemapping when disabled."""
        config = ForwardPlusConfig(tonemap_enabled=False)
        renderer = ForwardPlusRenderer(config=config)
        renderer.initialize(640, 480)
        renderer.begin_frame()
        renderer.execute_tonemapping()

        # Should not add draw call
        assert renderer.stats.draw_calls == 0


class TestForwardPlusRendererStats:
    """Test ForwardPlusRenderer statistics tracking."""

    def test_stats_after_frame(self):
        """Test statistics after complete frame."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1280, 720)

        lights = [
            LightData(position=(0.0, 0.0, 0.0)),
            LightData(position=(10.0, 0.0, 0.0)),
            LightData(position=(20.0, 0.0, 0.0)),
        ]

        renderer.begin_frame()
        renderer.execute_depth_prepass([1, 2, 3])
        renderer.execute_light_culling(lights)
        renderer.execute_forward_shading([1, 2, 3])
        renderer.execute_tonemapping()
        renderer.end_frame()

        stats = renderer.stats
        assert stats.visible_lights == 3
        assert stats.total_tiles > 0
        assert stats.tiles_with_lights > 0
        assert stats.draw_calls > 0

    def test_stats_reset_on_begin_frame(self):
        """Test stats reset on begin_frame."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(640, 480)

        # First frame
        renderer.begin_frame()
        renderer.execute_light_culling([LightData()])
        renderer.end_frame()

        # Second frame
        renderer.begin_frame()
        # Stats should be reset
        assert renderer.stats.visible_lights == 0


class TestIntegration:
    """Integration tests for Forward+ renderer."""

    def test_complete_render_loop(self):
        """Test complete render loop."""
        renderer = create_forward_plus_for_tier(QualityTier.LOW)
        renderer.initialize(1280, 720)

        # Simulate multiple frames
        for i in range(3):
            lights = [LightData(position=(float(i * 10), 0.0, 0.0))]

            renderer.begin_frame()
            renderer.execute_depth_prepass([])
            renderer.execute_light_culling(lights)
            renderer.execute_forward_shading([])
            renderer.execute_tonemapping()
            renderer.end_frame()

        assert renderer.frame_count == 3

    def test_resize_during_rendering(self):
        """Test resize between frames."""
        renderer = ForwardPlusRenderer()
        renderer.initialize(1280, 720)

        # Frame 1
        renderer.begin_frame()
        renderer.end_frame()

        # Resize
        renderer.resize(1920, 1080)

        # Frame 2
        renderer.begin_frame()
        renderer.end_frame()

        assert renderer.width == 1920
        assert renderer.height == 1080
        assert renderer.frame_count == 2

    def test_quality_tier_affects_culling(self):
        """Test quality tier affects light culling."""
        low_renderer = create_forward_plus_for_tier(QualityTier.LOW)
        medium_renderer = create_forward_plus_for_tier(QualityTier.MEDIUM)

        low_renderer.initialize(640, 480)
        medium_renderer.initialize(640, 480)

        # Create 20 lights
        lights = [LightData() for _ in range(20)]

        # Low tier should cull to 8 lights
        low_renderer.begin_frame()
        low_renderer.execute_light_culling(lights)
        low_renderer.end_frame()

        # Medium tier should allow 32 lights
        medium_renderer.begin_frame()
        medium_renderer.execute_light_culling(lights)
        medium_renderer.end_frame()

        assert low_renderer.stats.visible_lights == 8
        assert medium_renderer.stats.visible_lights == 20
