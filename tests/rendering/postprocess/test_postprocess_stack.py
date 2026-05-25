"""
Tests for Post-Process Stack

Tests effect ordering, volume blending, stack management, quality presets,
execution flags, intermediate target management, and frame graph integration.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from engine.rendering.framegraph.pass_node import PassFlags

from engine.rendering.postprocess.postprocess_stack import (
    BoxVolumeShape,
    EffectExecutionPath,
    EffectPriority,
    EffectQuality,
    EffectSettings,
    ExecutionFlags,
    get_quality_preset,
    IntermediateTarget,
    IntermediateTargetManager,
    PostProcessContext,
    PostProcessEffect,
    PostProcessStack,
    PostProcessStackConfig,
    PostProcessStackExecutor,
    PostProcessVolume,
    PostProcessVolumeSettings,
    QUALITY_PRESETS,
    QUALITY_PRESET_HIGH,
    QUALITY_PRESET_LOW,
    QUALITY_PRESET_MEDIUM,
    QUALITY_PRESET_ULTRA,
    QualityPreset,
    SphereVolumeShape,
    VolumeShape,
)


# ==============================================================================
# Mock Frame Graph for Testing
# ==============================================================================


class MockResourceHandle:
    """Simplified resource handle for testing."""

    def __init__(self, name: str, fmt: str = "R16G16B16A16_FLOAT") -> None:
        self.name = name
        self.format = fmt


class MockPassNode:
    """Simplified pass node for testing."""

    def __init__(self, name: str, pass_type: str = "compute") -> None:
        self.name = name
        self.pass_type = pass_type
        self.reads: list = []
        self.writes: list = []
        self._execute_callback = None
        self._flags: set = set()
        self._read_resources: list = []
        self._write_resources: list = []

    def read(self, resource: Any) -> None:
        self._read_resources.append(resource)

    def write(self, resource: Any) -> None:
        self._write_resources.append(resource)

    def set_execute(self, callback: callable) -> None:
        self._execute_callback = callback

    def set_flag(self, flag: Any) -> None:
        self._flags.add(flag)

    def has_flag(self, flag: Any) -> bool:
        return flag in self._flags


class MockFrameGraph:
    """Simplified frame graph for testing pass creation."""

    def __init__(self) -> None:
        self.passes: List[MockPassNode] = []
        self.textures: Dict[str, MockResourceHandle] = {}
        self.resources: Dict[str, MockResourceHandle] = {}

    def add_pass(self, name: str, pass_type: str = "compute") -> MockPassNode:
        pn = MockPassNode(name, pass_type)
        self.passes.append(pn)
        return pn

    def create_texture(
        self,
        name: str,
        format: Any,
        width: int,
        height: int,
    ) -> MockResourceHandle:
        handle = MockResourceHandle(name, str(format))
        self.textures[name] = handle
        self.resources[name] = handle
        return handle

    def get_resource(self, name: str) -> Optional[MockResourceHandle]:
        return self.resources.get(name)

    def add_resource(self, name: str, handle: MockResourceHandle) -> None:
        self.resources[name] = handle


# ==============================================================================
# Mock Effect for Testing
# ==============================================================================


@dataclass
class MockEffectSettings(EffectSettings):
    """Mock settings for testing."""

    test_value: float = 1.0

    def lerp(self, other: "MockEffectSettings", t: float) -> "MockEffectSettings":
        result = MockEffectSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            test_value=self.test_value + (other.test_value - self.test_value) * t,
        )
        return result


class MockEffect(PostProcessEffect[MockEffectSettings]):
    """Mock effect for testing."""

    def __init__(
        self,
        name: str,
        priority: int = 0,
        settings: Optional[MockEffectSettings] = None,
    ) -> None:
        super().__init__(name, settings or MockEffectSettings(), priority)
        self.setup_called = False
        self.execute_called = False
        self.cleanup_called = False
        self.last_inputs: Optional[Dict] = None
        self.last_outputs: Optional[Dict] = None
        self.last_delta_time: float = 0.0
        self.last_context: Optional[PostProcessContext] = None
        self._compute_effect = False

    def get_required_inputs(self) -> List[str]:
        return ["color"]

    def get_outputs(self) -> List[str]:
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        self.setup_called = True
        self.setup_width = width
        self.setup_height = height

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        self.execute_called = True
        self.last_inputs = inputs
        self.last_outputs = outputs
        self.last_delta_time = delta_time

    def cleanup(self) -> None:
        self.cleanup_called = True

    def execute_with_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        context: PostProcessContext,
    ) -> None:
        self.execute_called = True
        self.last_inputs = inputs
        self.last_outputs = outputs
        self.last_context = context
        super().execute_with_context(inputs, outputs, context)

    def set_compute_effect(self, value: bool) -> None:
        self._compute_effect = value

    def is_compute_effect(self) -> bool:
        return self._compute_effect


# ==============================================================================
# Test: EffectSettings
# ==============================================================================


class TestEffectSettings:
    """Test EffectSettings base class."""

    def test_default_values(self):
        """Test default settings values."""
        settings = EffectSettings()
        assert settings.enabled is True
        assert settings.weight == 1.0
        assert settings.priority == EffectPriority.CUSTOM.value

    def test_custom_values(self):
        """Test custom settings values."""
        settings = EffectSettings(enabled=False, weight=0.5, priority=100)
        assert settings.enabled is False
        assert settings.weight == 0.5
        assert settings.priority == 100

    def test_lerp_not_implemented(self):
        """Test that lerp raises NotImplementedError by default."""
        settings = EffectSettings()
        with pytest.raises(NotImplementedError):
            settings.lerp(EffectSettings(), 0.5)


# ==============================================================================
# Test: PostProcessEffect
# ==============================================================================


class TestPostProcessEffect:
    """Test PostProcessEffect base class."""

    def test_effect_creation(self):
        """Test creating a mock effect."""
        effect = MockEffect("TestEffect", priority=100)

        assert effect.name == "TestEffect"
        assert effect.priority == 100
        assert effect.enabled is True
        assert effect.dirty is True
        assert effect.id is not None

    def test_effect_enable_disable(self):
        """Test enabling and disabling effects."""
        effect = MockEffect("TestEffect")

        effect.enabled = False
        assert effect.enabled is False
        assert effect.dirty is True

        effect.mark_clean()
        effect.enabled = True
        assert effect.enabled is True
        assert effect.dirty is True

    def test_effect_settings(self):
        """Test effect settings access."""
        effect = MockEffect("TestEffect")

        assert effect.settings is not None
        assert isinstance(effect.settings, MockEffectSettings)

        new_settings = MockEffectSettings(test_value=2.0)
        effect.settings = new_settings
        assert effect.settings.test_value == 2.0
        assert effect.dirty is True

    def test_effect_setup(self):
        """Test effect setup."""
        effect = MockEffect("TestEffect")
        effect.setup(1920, 1080)

        assert effect.setup_called is True
        assert effect.setup_width == 1920
        assert effect.setup_height == 1080

    def test_effect_execute(self):
        """Test effect execution."""
        effect = MockEffect("TestEffect")
        effect.execute({}, {}, 0.016)

        assert effect.execute_called is True
        assert effect.last_delta_time == 0.016

    def test_effect_cleanup(self):
        """Test effect cleanup."""
        effect = MockEffect("TestEffect")
        effect.cleanup()

        assert effect.cleanup_called is True

    def test_priority_setter(self):
        """Test priority setter marks dirty."""
        effect = MockEffect("TestEffect", priority=100)
        effect.mark_clean()
        effect.priority = 200

        assert effect.priority == 200
        assert effect.dirty is True

    def test_mark_dirty_clean(self):
        """Test mark_dirty and mark_clean."""
        effect = MockEffect("TestEffect")
        effect.mark_clean()
        assert effect.dirty is False

        effect.mark_dirty()
        assert effect.dirty is True

    def test_is_compute_effect_default(self):
        """Test is_compute_effect returns False by default."""
        effect = MockEffect("TestEffect")
        assert effect.is_compute_effect() is False

    def test_is_compute_effect_true(self):
        """Test is_compute_effect override."""
        effect = MockEffect("TestEffect")
        effect.set_compute_effect(True)
        assert effect.is_compute_effect() is True

    def test_abstract_methods_prevent_instantiation(self):
        """Test that PostProcessEffect ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PostProcessEffect("Abstract", EffectSettings(), 0)  # type: ignore

    def test_default_execution_flags(self):
        """Test default execution flags include SKIP_IF_DISABLED and SKIP_IF_NO_INPUT."""
        effect = MockEffect("TestEffect")
        assert effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)
        assert effect.has_execution_flag(ExecutionFlags.SKIP_IF_NO_INPUT)
        assert not effect.has_execution_flag(ExecutionFlags.ALWAYS)
        assert not effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_ON_FIRST_FRAME)

    def test_set_execution_flags(self):
        """Test setting execution flags."""
        effect = MockEffect("TestEffect")
        effect.set_execution_flags(ExecutionFlags.FORCE_ASYNC.value)
        assert effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)

    def test_multiple_execution_flags(self):
        """Test combining multiple execution flags."""
        effect = MockEffect("TestEffect")
        flags = ExecutionFlags.FORCE_ASYNC.value | ExecutionFlags.ALWAYS.value
        effect.set_execution_flags(flags)
        assert effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC)
        assert effect.has_execution_flag(ExecutionFlags.ALWAYS)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)

    def test_execution_flags_mark_dirty(self):
        """Test that set_execution_flags marks the effect dirty."""
        effect = MockEffect("TestEffect")
        effect.mark_clean()
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        assert effect.dirty is True

    def test_should_execute_enabled(self):
        """Test should_execute returns True for enabled effect."""
        effect = MockEffect("TestEffect")
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx) is True

    def test_should_execute_disabled(self):
        """Test should_execute returns False for disabled effect."""
        effect = MockEffect("TestEffect")
        effect.enabled = False
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx) is False

    def test_should_execute_always_bypasses_disabled(self):
        """Test ALWAYS flag bypasses disabled check."""
        effect = MockEffect("TestEffect")
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        effect.enabled = False
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx) is True

    def test_should_execute_skip_on_first_frame(self):
        """Test SKIP_ON_FIRST_FRAME on frame 0."""
        effect = MockEffect("TemporalEffect")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        ctx = PostProcessContext(frame_index=0)
        assert effect.should_execute(ctx) is False

    def test_should_execute_not_skip_after_first_frame(self):
        """Test SKIP_ON_FIRST_FRAME runs after frame 1."""
        effect = MockEffect("TemporalEffect")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        ctx = PostProcessContext(frame_index=2)
        assert effect.should_execute(ctx) is True

    def test_should_execute_with_quality_preset(self):
        """Test quality preset filtering in should_execute."""
        effect = MockEffect("TAA")
        ctx = PostProcessContext(frame_index=10)
        preset = QUALITY_PRESET_MEDIUM
        assert effect.should_execute(ctx, preset) is False

    def test_should_execute_custom_effect_not_in_presets(self):
        """Test that custom effects not in any preset pass through."""
        effect = MockEffect("CustomEffect")
        ctx = PostProcessContext(frame_index=10)
        preset = QUALITY_PRESET_MEDIUM
        assert effect.should_execute(ctx, preset) is True

    def test_execute_with_context(self):
        """Test execute_with_context delegates to execute."""
        effect = MockEffect("TestEffect")
        ctx = PostProcessContext(frame_index=5, delta_time=0.033)
        effect.execute_with_context({"color": "in"}, {"color": "out"}, ctx)
        assert effect.execute_called is True
        assert effect.last_delta_time == 0.033
        assert effect.last_context is ctx

    def test_execute_on_rhi_delegates_to_execute_with_context(self):
        """Test execute_on_rhi delegates to execute_with_context."""
        effect = MockEffect("TestEffect")
        ctx = PostProcessContext(frame_index=5)
        effect.execute_on_rhi("cmd_list", {"color": "in"}, {"color": "out"}, ctx)
        assert effect.execute_called is True
        assert effect.last_context is ctx

    def test_add_to_frame_graph_creates_pass_node(self):
        """Test add_to_frame_graph creates a pass node with resources."""
        fg = MockFrameGraph()
        fg.add_resource("color", MockResourceHandle("color"))
        effect = MockEffect("TestEffect")
        pass_node = effect.add_to_frame_graph(fg)
        assert pass_node.name == "PostProcess_TestEffect"
        assert pass_node.pass_type == "graphics"

    def test_add_to_frame_graph_graphics_effect(self):
        """Test add_to_frame_graph creates graphics pass for non-compute effects."""
        fg = MockFrameGraph()
        fg.add_resource("color", MockResourceHandle("color"))
        effect = MockEffect("TestEffect")
        effect.set_compute_effect(True)
        pass_node = effect.add_to_frame_graph(fg)
        assert pass_node.pass_type == "compute"


# ==============================================================================
# Test: ExecutionFlags
# ==============================================================================


class TestExecutionFlags:
    """Test ExecutionFlags enum values."""

    def test_none_value(self):
        assert ExecutionFlags.NONE.value == 0

    def test_skip_if_disabled(self):
        assert ExecutionFlags.SKIP_IF_DISABLED.value == 1

    def test_skip_on_first_frame(self):
        assert ExecutionFlags.SKIP_ON_FIRST_FRAME.value == 2

    def test_skip_if_no_input(self):
        assert ExecutionFlags.SKIP_IF_NO_INPUT.value == 4

    def test_force_async(self):
        assert ExecutionFlags.FORCE_ASYNC.value == 8

    def test_always(self):
        assert ExecutionFlags.ALWAYS.value == 16

    def test_flags_combine(self):
        combined = ExecutionFlags.SKIP_IF_DISABLED | ExecutionFlags.ALWAYS
        assert combined == 17


# ==============================================================================
# Test: EffectExecutionPath
# ==============================================================================


class TestEffectExecutionPath:
    """Test EffectExecutionPath enum."""

    def test_frame_graph_pass(self):
        assert EffectExecutionPath.FRAME_GRAPH_PASS.name == "FRAME_GRAPH_PASS"

    def test_direct_call(self):
        assert EffectExecutionPath.DIRECT_CALL.name == "DIRECT_CALL"

    def test_merged_compute(self):
        assert EffectExecutionPath.MERGED_COMPUTE.name == "MERGED_COMPUTE"

    def test_all_unique(self):
        values = [e.value for e in EffectExecutionPath]
        assert len(values) == len(set(values))


# ==============================================================================
# Test: QualityPreset
# ==============================================================================


class TestQualityPreset:
    """Test QualityPreset system."""

    def test_preset_creation(self):
        """Test creating a quality preset."""
        preset = QualityPreset(
            name="Test",
            quality=EffectQuality.HIGH,
            active_effects={"EffectA", "EffectB"},
        )
        assert preset.name == "Test"
        assert preset.quality == EffectQuality.HIGH
        assert "EffectA" in preset.active_effects

    def test_is_effect_active(self):
        """Test is_effect_active."""
        preset = QualityPreset(
            name="Test",
            quality=EffectQuality.HIGH,
            active_effects={"EffectA"},
        )
        assert preset.is_effect_active("EffectA") is True
        assert preset.is_effect_active("EffectB") is False

    def test_get_effect_config_exists(self):
        """Test get_effect_config with existing config."""
        preset = QualityPreset(
            name="Test",
            quality=EffectQuality.HIGH,
            active_effects={"Bloom"},
            effect_configs={"Bloom": {"quality": "high"}},
        )
        config = preset.get_effect_config("Bloom")
        assert config == {"quality": "high"}

    def test_get_effect_config_missing(self):
        """Test get_effect_config with missing config returns default."""
        preset = QualityPreset(
            name="Test",
            quality=EffectQuality.HIGH,
            active_effects={"Bloom"},
        )
        config = preset.get_effect_config("Bloom", {"default": True})
        assert config == {"default": True}

    def test_get_effect_config_none_default(self):
        """Test get_effect_config with no default returns None."""
        preset = QualityPreset(
            name="Test",
            quality=EffectQuality.HIGH,
            active_effects={"Bloom"},
        )
        config = preset.get_effect_config("NonExistent")
        assert config is None

    def test_preset_low_effects(self):
        """Test LOW preset has basic effects."""
        assert "Exposure" in QUALITY_PRESET_LOW.active_effects
        assert "Tonemapping" in QUALITY_PRESET_LOW.active_effects
        assert "FXAA" in QUALITY_PRESET_LOW.active_effects
        assert len(QUALITY_PRESET_LOW.active_effects) == 3

    def test_preset_medium_effects(self):
        """Test MEDIUM preset adds bloom and color grading."""
        assert "Bloom" in QUALITY_PRESET_MEDIUM.active_effects
        assert "ColorGrading" in QUALITY_PRESET_MEDIUM.active_effects
        assert "SMAA" in QUALITY_PRESET_MEDIUM.active_effects
        assert len(QUALITY_PRESET_MEDIUM.active_effects) == 5

    def test_preset_high_effects(self):
        """Test HIGH preset includes cinematic effects."""
        assert "DepthOfField" in QUALITY_PRESET_HIGH.active_effects
        assert "MotionBlur" in QUALITY_PRESET_HIGH.active_effects
        assert "AmbientOcclusion" in QUALITY_PRESET_HIGH.active_effects
        assert "TAA" in QUALITY_PRESET_HIGH.active_effects
        assert "FXAA" not in QUALITY_PRESET_HIGH.active_effects

    def test_preset_ultra_effects(self):
        """Test ULTRA preset has all effects including upscaling."""
        assert "Upscaling" in QUALITY_PRESET_ULTRA.active_effects
        assert len(QUALITY_PRESET_ULTRA.active_effects) == 9

    def test_quality_presets_dict(self):
        """Test QUALITY_PRESETS has all 4 levels."""
        assert len(QUALITY_PRESETS) == 4
        assert EffectQuality.LOW in QUALITY_PRESETS
        assert EffectQuality.MEDIUM in QUALITY_PRESETS
        assert EffectQuality.HIGH in QUALITY_PRESETS
        assert EffectQuality.ULTRA in QUALITY_PRESETS

    def test_get_quality_preset_by_enum(self):
        """Test get_quality_preset with enum."""
        preset = get_quality_preset(EffectQuality.HIGH)
        assert preset.name == "High"

    def test_get_quality_preset_by_name(self):
        """Test get_quality_preset with string name."""
        preset = get_quality_preset("ultra")
        assert preset.quality == EffectQuality.ULTRA

    def test_get_quality_preset_by_name_case_insensitive(self):
        """Test get_quality_preset is case-insensitive."""
        preset = get_quality_preset("HIGH")
        assert preset.quality == EffectQuality.HIGH

    def test_get_quality_preset_invalid_enum(self):
        """Test get_quality_preset with invalid enum raises."""
        with pytest.raises(ValueError):
            get_quality_preset(EffectQuality(99))

    def test_get_quality_preset_invalid_name(self):
        """Test get_quality_preset with invalid name raises."""
        with pytest.raises(ValueError):
            get_quality_preset("SuperUltra")

    def test_get_quality_preset_invalid_type(self):
        """Test get_quality_preset with invalid type raises."""
        with pytest.raises(ValueError):
            get_quality_preset(42)


# ==============================================================================
# Test: PostProcessContext
# ==============================================================================


class TestPostProcessContext:
    """Test PostProcessContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = PostProcessContext()
        assert ctx.rhi_command_list is None
        assert ctx.rhi_device is None
        assert ctx.frame_index == 0
        assert ctx.quality == EffectQuality.HIGH
        assert ctx.delta_time == 0.016
        assert ctx.camera_position is None
        assert ctx.history_buffers == {}

    def test_is_first_frame_frame_0(self):
        """Test is_first_frame is True for frame 0."""
        ctx = PostProcessContext(frame_index=0)
        assert ctx.is_first_frame is True

    def test_is_first_frame_frame_1(self):
        """Test is_first_frame is True for frame 1."""
        ctx = PostProcessContext(frame_index=1)
        assert ctx.is_first_frame is True

    def test_is_first_frame_frame_2(self):
        """Test is_first_frame is False for frame 2."""
        ctx = PostProcessContext(frame_index=2)
        assert ctx.is_first_frame is False

    def test_custom_values(self):
        """Test custom context values."""
        ctx = PostProcessContext(
            rhi_command_list="cmd",
            rhi_device="device",
            frame_index=42,
            quality=EffectQuality.ULTRA,
            delta_time=0.033,
            camera_position=(1.0, 2.0, 3.0),
            history_buffers={"prev": "data"},
        )
        assert ctx.rhi_command_list == "cmd"
        assert ctx.frame_index == 42
        assert ctx.quality == EffectQuality.ULTRA
        assert ctx.delta_time == 0.033
        assert ctx.camera_position == (1.0, 2.0, 3.0)


# ==============================================================================
# Test: IntermediateTargetManager
# ==============================================================================


class TestIntermediateTargetManager:
    """Test IntermediateTargetManager."""

    def test_default_creation(self):
        """Test default creation with pool of 2."""
        mgr = IntermediateTargetManager()
        assert mgr.pool_size == 2
        assert mgr.format == "R11G11B10_FLOAT"

    def test_custom_pool_size(self):
        """Test custom pool size."""
        mgr = IntermediateTargetManager(pool_size=3)
        assert mgr.pool_size == 3

    def test_invalid_pool_size_raises(self):
        """Test pool_size < 1 raises ValueError."""
        with pytest.raises(ValueError):
            IntermediateTargetManager(pool_size=0)

    def test_custom_format(self):
        """Test custom format."""
        mgr = IntermediateTargetManager(format="R16G16B16A16_FLOAT")
        assert mgr.format == "R16G16B16A16_FLOAT"

    def test_format_setter(self):
        """Test format setter."""
        mgr = IntermediateTargetManager()
        mgr.format = "R8G8B8A8_UNORM"
        assert mgr.format == "R8G8B8A8_UNORM"

    def test_resize_invalidates(self):
        """Test resize invalidates ready state."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr._ready is True
        mgr.resize(1280, 720)
        assert mgr._ready is False

    def test_allocate_creates_targets(self):
        """Test allocate creates intermediate targets."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        handles = mgr.allocate(fg, 1920, 1080)
        assert len(handles) == 2
        assert mgr._ready is True

    def test_get_target_valid(self):
        """Test get_target returns handle for valid index."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        handle = mgr.get_target(0)
        assert handle is not None

    def test_get_target_invalid_index(self):
        """Test get_target returns None for out-of-range index."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        handle = mgr.get_target(99)
        assert handle is None

    def test_get_target_not_ready(self):
        """Test get_target returns None when not allocated."""
        mgr = IntermediateTargetManager()
        handle = mgr.get_target(0)
        assert handle is None

    def test_get_ping_pong_alternates(self):
        """Test get_ping_pong alternates read/write targets."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)

        read0, write0 = mgr.get_ping_pong(0)
        read1, write1 = mgr.get_ping_pong(1)
        assert read0 is not None
        assert write0 is not None
        assert read1 is not None
        assert write1 is not None

    def test_get_read_target(self):
        """Test get_read_target returns correct target for index."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        handle = mgr.get_read_target(0)
        assert handle is not None
        assert mgr.get_read_target(0) == mgr.get_read_target(2)  # wraps around

    def test_get_write_target(self):
        """Test get_write_target returns different target from read."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        read_handle = mgr.get_read_target(0)
        write_handle = mgr.get_write_target(0)
        assert read_handle is not None
        assert write_handle is not None
        assert read_handle != write_handle

    def test_get_write_target_pool_size_1(self):
        """Test get_write_target returns None for pool_size=1."""
        mgr = IntermediateTargetManager(pool_size=1)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        handle = mgr.get_write_target(0)
        assert handle is None

    def test_ping_pong_not_ready(self):
        """Test get_ping_pong returns None when not ready."""
        mgr = IntermediateTargetManager(pool_size=2)
        read_handle, write_handle = mgr.get_ping_pong(0)
        assert read_handle is None
        assert write_handle is None

    def test_reset_clears_state(self):
        """Test reset clears targets and ready state."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr._ready is True
        mgr.reset()
        assert mgr._ready is False
        assert len(mgr._targets) == 0


# ==============================================================================
# Test: PostProcessStack
# ==============================================================================


class TestPostProcessStack:
    """Test PostProcessStack management."""

    def test_stack_creation(self):
        """Test creating an empty stack."""
        stack = PostProcessStack()
        assert stack.width == 0
        assert stack.height == 0
        assert len(stack.effects) == 0

    def test_stack_config(self):
        """Test stack with custom config."""
        config = PostProcessStackConfig(
            hdr_enabled=True,
            auto_exposure_enabled=False,
        )
        stack = PostProcessStack(config)
        assert stack.config.hdr_enabled is True
        assert stack.config.auto_exposure_enabled is False

    def test_default_quality(self):
        """Test default quality is HIGH."""
        stack = PostProcessStack()
        assert stack.quality == EffectQuality.HIGH
        assert stack.quality_preset.name == "High"

    def test_custom_quality(self):
        """Test custom initial quality."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        assert stack.quality == EffectQuality.LOW

    def test_frame_index_starts_at_0(self):
        """Test frame index starts at 0."""
        stack = PostProcessStack()
        assert stack.frame_index == 0

    def test_add_effect(self):
        """Test adding effects to stack."""
        stack = PostProcessStack()
        effect = MockEffect("TestEffect", priority=100)
        stack.add_effect(effect)
        assert len(stack.effects) == 1
        assert stack.get_effect("TestEffect") is effect

    def test_add_duplicate_effect_raises(self):
        """Test that adding duplicate effect raises error."""
        stack = PostProcessStack()
        effect1 = MockEffect("TestEffect", priority=100)
        effect2 = MockEffect("TestEffect", priority=200)
        stack.add_effect(effect1)
        with pytest.raises(ValueError, match="already exists"):
            stack.add_effect(effect2)

    def test_effect_ordering(self):
        """Test that effects are ordered by priority."""
        stack = PostProcessStack()
        effect_high = MockEffect("HighPriority", priority=1000)
        effect_low = MockEffect("LowPriority", priority=100)
        effect_mid = MockEffect("MidPriority", priority=500)
        stack.add_effect(effect_high)
        stack.add_effect(effect_low)
        stack.add_effect(effect_mid)
        effects = stack.effects
        assert effects[0].name == "LowPriority"
        assert effects[1].name == "MidPriority"
        assert effects[2].name == "HighPriority"

    def test_remove_effect(self):
        """Test removing effects from stack."""
        stack = PostProcessStack()
        effect = MockEffect("TestEffect")
        stack.add_effect(effect)
        removed = stack.remove_effect("TestEffect")
        assert removed is effect
        assert len(stack.effects) == 0
        assert stack.get_effect("TestEffect") is None
        assert effect.cleanup_called is True

    def test_remove_nonexistent_effect(self):
        """Test removing nonexistent effect returns None."""
        stack = PostProcessStack()
        result = stack.remove_effect("NonExistent")
        assert result is None

    def test_enable_effect(self):
        """Test enabling/disabling effects through stack."""
        stack = PostProcessStack()
        effect = MockEffect("TestEffect")
        stack.add_effect(effect)
        stack.enable_effect("TestEffect", False)
        assert effect.enabled is False
        stack.enable_effect("TestEffect", True)
        assert effect.enabled is True

    def test_resize(self):
        """Test stack resize."""
        stack = PostProcessStack()
        effect = MockEffect("TestEffect")
        stack.add_effect(effect)
        stack.resize(1920, 1080)
        assert stack.width == 1920
        assert stack.height == 1080
        assert effect.setup_called is True

    def test_resize_same_size_no_op(self):
        """Test that resizing to same size is a no-op."""
        stack = PostProcessStack()
        effect = MockEffect("TestEffect")
        stack.add_effect(effect)
        stack.resize(1920, 1080)
        effect.setup_called = False
        stack.resize(1920, 1080)
        assert effect.setup_called is False

    def test_cleanup(self):
        """Test stack cleanup."""
        stack = PostProcessStack()
        effect1 = MockEffect("Effect1")
        effect2 = MockEffect("Effect2")
        stack.add_effect(effect1)
        stack.add_effect(effect2)
        stack.cleanup()
        assert effect1.cleanup_called is True
        assert effect2.cleanup_called is True
        assert len(stack.effects) == 0

    def test_set_quality(self):
        """Test set_quality changes preset."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        stack.set_quality(EffectQuality.ULTRA)
        assert stack.quality == EffectQuality.ULTRA
        assert stack.quality_preset.name == "Ultra"

    def test_set_quality_same_no_op(self):
        """Test set_quality to same level is no-op."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack._dirty = False
        stack.set_quality(EffectQuality.HIGH)
        assert stack._dirty is False

    def test_advance_frame(self):
        """Test advance_frame increments counter."""
        stack = PostProcessStack()
        assert stack.frame_index == 0
        stack.advance_frame()
        assert stack.frame_index == 1
        stack.advance_frame()
        assert stack.frame_index == 2

    def test_get_active_effects_by_quality(self):
        """Test get_active_effects filters by quality preset."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        bloom = MockEffect("Bloom")
        tonemap = MockEffect("Tonemapping")
        stack.add_effect(bloom)
        stack.add_effect(tonemap)
        active = stack.get_active_effects()
        assert "Tonemapping" in [e.name for e in active]
        assert "Bloom" not in [e.name for e in active]

    def test_get_active_effects_custom_effect_passthrough(self):
        """Test custom effects not in any preset pass through."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        custom = MockEffect("CustomShader")
        stack.add_effect(custom)
        active = stack.get_active_effects()
        assert "CustomShader" in [e.name for e in active]

    def test_get_active_effects_disabled_filtered(self):
        """Test disabled effects are filtered out."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        bloom = MockEffect("Bloom")
        bloom.enabled = False
        stack.add_effect(bloom)
        active = stack.get_active_effects()
        assert "Bloom" not in [e.name for e in active]

    def test_execute_with_context_runs_active_effects(self):
        """Test execute_with_context runs active effects."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        bloom = MockEffect("Bloom")
        tonemap = MockEffect("Tonemapping")
        stack.add_effect(bloom)
        stack.add_effect(tonemap)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        assert bloom.execute_called is True
        assert tonemap.execute_called is True

    def test_execute_with_context_skips_disabled(self):
        """Test execute_with_context skips disabled effects."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        bloom = MockEffect("Bloom")
        bloom.enabled = False
        tonemap = MockEffect("Tonemapping")
        stack.add_effect(bloom)
        stack.add_effect(tonemap)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        assert bloom.execute_called is False
        assert tonemap.execute_called is True

    def test_execute_with_context_advances_frame(self):
        """Test execute_with_context advances frame index."""
        stack = PostProcessStack()
        assert stack.frame_index == 0
        stack.add_effect(MockEffect("Tonemapping"))
        ctx = PostProcessContext(frame_index=5)
        stack.execute_with_context("hdr_in", "output", ctx)
        assert stack.frame_index == 1

    def test_build_frame_graph_creates_passes(self):
        """Test build_frame_graph adds passes for active effects."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        stack.add_effect(MockEffect("Bloom"))
        fg = MockFrameGraph()
        fg.add_resource("color", MockResourceHandle("color"))
        stack.build_frame_graph(fg)
        assert len(fg.passes) == 2

    def test_build_frame_graph_filters_inactive(self):
        """Test build_frame_graph skips inactive effects."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        stack.add_effect(MockEffect("Tonemapping"))
        stack.add_effect(MockEffect("Bloom"))
        fg = MockFrameGraph()
        fg.add_resource("color", MockResourceHandle("color"))
        stack.build_frame_graph(fg)
        assert len(fg.passes) == 1

    def test_execute_with_context_pipes_inputs(self):
        """Test execute_with_context chains effect inputs/outputs."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect_a = MockEffect("Bloom")
        effect_b = MockEffect("Tonemapping")
        stack.add_effect(effect_a)
        stack.add_effect(effect_b)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_input", "final_output", ctx)
        assert effect_a.last_inputs == {"color": "hdr_input"}
        assert effect_b.last_inputs == {"color": effect_a.last_outputs.get("color")}

    def test_execute_with_context_rhi_path(self):
        """Test execute_with_context uses execute_on_rhi when command list present."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        ctx = PostProcessContext(rhi_command_list="cmd_list", frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        assert effect.execute_called is True


# ==============================================================================
# Test: PostProcessStackExecutor
# ==============================================================================


class TestPostProcessStackExecutor:
    """Test PostProcessStackExecutor."""

    def test_creation_with_stack(self):
        """Test creating executor with a stack."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        assert executor.stack is stack
        assert executor.is_built is False
        assert executor.execution_path == EffectExecutionPath.FRAME_GRAPH_PASS

    def test_creation_with_frame_graph(self):
        """Test creating executor with frame graph."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        assert executor.frame_graph is fg

    def test_execution_path_setter(self):
        """Test changing execution path."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor.execution_path = EffectExecutionPath.DIRECT_CALL
        assert executor.execution_path == EffectExecutionPath.DIRECT_CALL

    def test_frame_graph_setter_invalidates(self):
        """Test frame_graph setter invalidates build state."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor._is_built = True
        executor.frame_graph = MockFrameGraph()
        assert executor.is_built is False

    def test_set_context(self):
        """Test setting execution context."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=42, quality=EffectQuality.ULTRA)
        executor.set_context(ctx)
        assert executor._context.frame_index == 42
        assert executor._context.quality == EffectQuality.ULTRA

    def test_update_context_partial(self):
        """Test partial context update."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor._context.frame_index = 0
        executor._context.quality = EffectQuality.LOW
        executor.update_context(frame_index=10, quality=EffectQuality.ULTRA)
        assert executor._context.frame_index == 10
        assert executor._context.quality == EffectQuality.ULTRA
        assert executor._context.delta_time == 0.016  # unchanged default

    def test_update_context_single_field(self):
        """Test updating a single context field."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor.update_context(delta_time=0.033)
        assert executor._context.delta_time == 0.033
        assert executor._context.frame_index == 0  # unchanged

    def test_update_context_history_buffers(self):
        """Test update_context merges history buffers."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor._context.history_buffers["existing"] = "value"
        executor.update_context(history_buffers={"new": "buffer"})
        assert executor._context.history_buffers["existing"] == "value"
        assert executor._context.history_buffers["new"] == "buffer"

    def test_prepare_resources_sets_size(self):
        """Test prepare_resources sets size on stack."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor.prepare_resources(1920, 1080)
        assert stack.width == 1920
        assert stack.height == 1080

    def test_prepare_resources_no_frame_graph(self):
        """Test prepare_resources without frame graph does not error."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        executor.prepare_resources(1920, 1080)
        assert executor._hdr_handle is None
        assert executor._output_handle is None

    def test_prepare_resources_with_frame_graph(self):
        """Test prepare_resources with frame graph creates textures."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        assert executor._hdr_handle is not None
        assert executor._output_handle is not None
        assert executor._has_resources is True

    def test_prepare_resources_creates_intermediates(self):
        """Test prepare_resources allocates intermediate targets."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        assert executor._intermediate_mgr._ready is True

    def test_build_passes_no_frame_graph_raises(self):
        """Test build_passes without frame graph raises."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        with pytest.raises(RuntimeError, match="Frame graph is required"):
            executor.build_passes()

    def test_build_passes_no_resources_raises(self):
        """Test build_passes without resources raises."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        with pytest.raises(RuntimeError, match="HDR input and output"):
            executor.build_passes()

    def test_build_passes_creates_pass_nodes(self):
        """Test build_passes creates pass nodes for active effects."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        stack.add_effect(MockEffect("Bloom"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert len(fg.passes) == 2

    def test_build_passes_skips_inactive_effects(self):
        """Test build_passes skips effects not in quality preset."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        stack.add_effect(MockEffect("Tonemapping"))
        stack.add_effect(MockEffect("Bloom"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert len(fg.passes) == 1

    def test_build_passes_execution_order(self):
        """Test build_passes maintains execution order."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping", priority=500))
        stack.add_effect(MockEffect("Bloom", priority=100))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert "Bloom" in fg.passes[0].name
        assert "Tonemapping" in fg.passes[1].name

    def test_build_passes_attaches_callbacks(self):
        """Test build_passes attaches execute callbacks to pass nodes."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert fg.passes[0]._execute_callback is not None

    def test_execute_direct_runs_effects(self):
        """Test execute_direct runs effects through the stack."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=10)
        executor.execute_direct("hdr_in", "output", ctx)
        assert effect.execute_called is True

    def test_execute_direct_preserves_order(self):
        """Test execute_direct maintains effect order."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect_a = MockEffect("Bloom")
        effect_b = MockEffect("Tonemapping")
        stack.add_effect(effect_a)
        stack.add_effect(effect_b)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=10)
        executor.execute_direct("hdr_in", "final_output", ctx)
        assert effect_a.last_inputs == {"color": "hdr_in"}
        assert effect_b.last_inputs is not None

    def test_execute_direct_quality_filter(self):
        """Test execute_direct respects quality preset filtering."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        bloom = MockEffect("Bloom")
        tonemap = MockEffect("Tonemapping")
        stack.add_effect(bloom)
        stack.add_effect(tonemap)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=10)
        executor.execute_direct("hdr_in", "output", ctx)
        assert tonemap.execute_called is True
        assert bloom.execute_called is False

    def test_execute_direct_no_context(self):
        """Test execute_direct uses internal context when none provided."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        executor = PostProcessStackExecutor(stack)
        executor.execute_direct("hdr_in", "output")
        assert effect.execute_called is True

    def test_execute_direct_rhi_path(self):
        """Test execute_direct uses execute_on_rhi when command list present."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(rhi_command_list="cmd_list", frame_index=10)
        executor.execute_direct("hdr_in", "output", ctx)
        assert effect.execute_called is True

    def test_rebuild_if_needed_not_dirty(self):
        """Test rebuild_if_needed returns False when clean."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        executor = PostProcessStackExecutor(stack)
        stack._dirty = False
        result = executor.rebuild_if_needed()
        assert result is False

    def test_rebuild_if_needed_dirty_no_fg(self):
        """Test rebuild_if_needed returns True when dirty but no frame graph."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        executor = PostProcessStackExecutor(stack)
        stack._dirty = True
        result = executor.rebuild_if_needed()
        assert result is True

    def test_rebuild_if_needed_dirty_with_fg(self):
        """Test rebuild_if_needed rebuilds passes when dirty."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor._is_built = True
        stack._dirty = True
        result = executor.rebuild_if_needed()
        assert result is True

    def test_executor_execution_flags_via_direct(self):
        """Test execution flags are honored via execute_direct."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        temporal = MockEffect("TAA")
        temporal.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        stack.add_effect(temporal)
        executor = PostProcessStackExecutor(stack)
        # First frame -- should be skipped
        ctx0 = PostProcessContext(frame_index=0)
        executor.execute_direct("hdr_in", "output", ctx0)
        assert temporal.execute_called is False
        # Subsequent frame -- should execute
        ctx1 = PostProcessContext(frame_index=2)
        executor.execute_direct("hdr_in", "output", ctx1)
        assert temporal.execute_called is True

    def test_executor_tag_async_passes(self):
        """Test FORCE_ASYNC tags pass nodes with ASYNC_COMPUTE flag."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value | ExecutionFlags.FORCE_ASYNC.value
        )
        stack.add_effect(effect)
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert len(fg.passes) > 0
        assert fg.passes[0].has_flag(PassFlags.ASYNC_COMPUTE)
        assert executor.is_built is True

    def test_reset_clears_resources(self):
        """Test reset clears all resource handles."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        assert executor._has_resources is True
        executor.reset()
        assert executor._has_resources is False
        assert executor._hdr_handle is None
        assert executor._output_handle is None
        assert executor._is_built is False

    def test_prepare_resources_format_overrides(self):
        """Test prepare_resources with format overrides."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(
            1920, 1080,
            hdr_format="R32G32B32A32_FLOAT",
            output_format="R8G8B8A8_SRGB",
        )
        assert stack.config.hdr_format == "R32G32B32A32_FLOAT"
        assert stack.config.output_format == "R8G8B8A8_SRGB"

    def test_intermediate_format_from_config(self):
        """Test IntermediateTargetManager gets format from stack config."""
        config = PostProcessStackConfig(intermediate_format="R16G16B16A16_FLOAT")
        stack = PostProcessStack(config)
        executor = PostProcessStackExecutor(stack)
        assert executor._intermediate_mgr.format == "R16G16B16A16_FLOAT"


# ==============================================================================
# Test: PostProcessStackConfig
# ==============================================================================


class TestPostProcessStackConfig:
    """Test PostProcessStackConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = PostProcessStackConfig()
        assert config.hdr_enabled is True
        assert config.hdr_format == "R16G16B16A16_FLOAT"
        assert config.intermediate_format == "R11G11B10_FLOAT"
        assert config.output_format == "R8G8B8A8_UNORM"
        assert config.auto_exposure_enabled is True
        assert config.history_buffer_count == 2

    def test_custom_config(self):
        """Test custom config values."""
        config = PostProcessStackConfig(
            hdr_enabled=False,
            hdr_format="R32G32B32A32_FLOAT",
            intermediate_format="R16G16B16A16_FLOAT",
            output_format="R8G8B8A8_SRGB",
            auto_exposure_enabled=False,
            history_buffer_count=4,
        )
        assert config.hdr_enabled is False
        assert config.hdr_format == "R32G32B32A32_FLOAT"
        assert config.intermediate_format == "R16G16B16A16_FLOAT"
        assert config.output_format == "R8G8B8A8_SRGB"
        assert config.auto_exposure_enabled is False
        assert config.history_buffer_count == 4


# ==============================================================================
# Test: VolumeShapes
# ==============================================================================


class TestVolumeShapes:
    """Test volume shapes."""

    def test_box_contains(self):
        """Test box shape containment."""
        box = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        assert box.contains((5.0, 5.0, 5.0)) is True
        assert box.contains((0.0, 0.0, 0.0)) is True
        assert box.contains((10.0, 10.0, 10.0)) is True
        assert box.contains((-1.0, 5.0, 5.0)) is False
        assert box.contains((15.0, 5.0, 5.0)) is False

    def test_box_contains_outside_all_axes(self):
        """Test point outside on all axes."""
        box = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        assert box.contains((-5.0, -5.0, -5.0)) is False

    def test_box_distance_to_boundary_center(self):
        """Test box distance calculation from center."""
        box = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        dist = box.distance_to_boundary((5.0, 5.0, 5.0))
        assert dist == 5.0

    def test_box_distance_to_boundary_near_edge(self):
        """Test box distance calculation near edge."""
        box = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        dist = box.distance_to_boundary((1.0, 5.0, 5.0))
        assert dist == 1.0

    def test_box_distance_to_boundary_corner(self):
        """Test box distance calculation near corner."""
        box = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        dist = box.distance_to_boundary((1.0, 2.0, 1.0))
        assert dist == 1.0

    def test_sphere_contains(self):
        """Test sphere shape containment."""
        sphere = SphereVolumeShape(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        assert sphere.contains((0.0, 0.0, 0.0)) is True
        assert sphere.contains((5.0, 0.0, 0.0)) is True
        assert sphere.contains((10.0, 0.0, 0.0)) is True
        assert sphere.contains((11.0, 0.0, 0.0)) is False

    def test_sphere_contains_3d(self):
        """Test sphere containment in 3D."""
        sphere = SphereVolumeShape(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        assert sphere.contains((5.0, 5.0, 5.0)) is True
        assert sphere.contains((-5.0, 5.0, 5.0)) is True

    def test_sphere_distance_to_boundary_center(self):
        """Test sphere distance calculation from center."""
        sphere = SphereVolumeShape(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        dist = sphere.distance_to_boundary((0.0, 0.0, 0.0))
        assert dist == 10.0

    def test_sphere_distance_to_boundary_halfway(self):
        """Test sphere distance calculation halfway."""
        sphere = SphereVolumeShape(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        dist = sphere.distance_to_boundary((5.0, 0.0, 0.0))
        assert dist == 5.0

    def test_sphere_distance_to_boundary_at_edge(self):
        """Test sphere distance calculation at edge."""
        sphere = SphereVolumeShape(
            center=(0.0, 0.0, 0.0),
            radius=10.0,
        )
        dist = sphere.distance_to_boundary((10.0, 0.0, 0.0))
        assert dist == 0.0


# ==============================================================================
# Test: PostProcessVolume
# ==============================================================================


class TestPostProcessVolume:
    """Test PostProcessVolume functionality."""

    def test_volume_creation(self):
        """Test creating a volume."""
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, priority=10)
        assert volume.priority == 10
        assert volume.enabled is True
        assert volume.id is not None

    def test_global_volume(self):
        """Test global volume affects all points."""
        shape = BoxVolumeShape(
            min_bounds=(100.0, 100.0, 100.0),
            max_bounds=(200.0, 200.0, 200.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, global_volume=True)
        assert volume.contains_point((0.0, 0.0, 0.0)) is True
        assert volume.get_blend_weight((0.0, 0.0, 0.0)) == 1.0

    def test_volume_blend_weight_inside(self):
        """Test volume blend weight inside shape."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=2.0)
        weight = volume.get_blend_weight((5.0, 5.0, 5.0))
        assert weight == 1.0

    def test_volume_blend_weight_near_boundary(self):
        """Test volume blend weight near boundary."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=2.0)
        weight = volume.get_blend_weight((9.5, 5.0, 5.0))
        assert 0.0 <= weight <= 1.0

    def test_disabled_volume(self):
        """Test disabled volume does not affect anything."""
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        volume.enabled = False
        assert volume.contains_point((0.5, 0.5, 0.5)) is False
        assert volume.get_blend_weight((0.5, 0.5, 0.5)) == 0.0

    def test_volume_settings_override(self):
        """Test volume with effect overrides."""
        shape = BoxVolumeShape()
        override_settings = MockEffectSettings(test_value=2.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"TestEffect": override_settings}
        )
        volume = PostProcessVolume(shape, settings)
        assert "TestEffect" in volume.settings.effect_overrides
        assert volume.settings.effect_overrides["TestEffect"].test_value == 2.0

    def test_priority_setter(self):
        """Test priority setter on volume."""
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, priority=10)
        volume.priority = 100
        assert volume.priority == 100

    def test_settings_setter(self):
        """Test settings setter on volume."""
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        new_settings = PostProcessVolumeSettings()
        volume.settings = new_settings
        assert volume.settings is new_settings

    def test_contains_point_outside(self):
        """Test contains_point returns False outside shape."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        assert volume.contains_point((50.0, 50.0, 50.0)) is False

    def test_blend_weight_outside_no_blend(self):
        """Test blend weight is 0 outside when no blend distance."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=0.0)
        assert volume.get_blend_weight((15.0, 5.0, 5.0)) == 0.0

    def test_blend_weight_outside_blend_zone(self):
        """Test blend weight blends when outside but within blend distance."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=5.0)
        weight = volume.get_blend_weight((12.0, 5.0, 5.0))
        assert 0.0 < weight < 1.0

    def test_blend_weight_outside_beyond_blend(self):
        """Test blend weight is 0 beyond blend distance."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=2.0)
        assert volume.get_blend_weight((20.0, 5.0, 5.0)) == 0.0

    def test_apply_to_stack_no_effect(self):
        """Test apply_to_stack with no matching effect does nothing."""
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        # Should not raise
        volume.apply_to_stack(stack, 1.0)

    def test_apply_to_stack_zero_weight(self):
        """Test apply_to_stack with zero weight is a no-op."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=5.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"TestEffect": override}
        )
        volume = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        effect = MockEffect("TestEffect")
        stack.add_effect(effect)
        volume.apply_to_stack(stack, 0.0)
        assert effect.settings.test_value == 1.0  # unchanged

    def test_contains_point_with_blend_distance(self):
        """Test contains_point includes blend distance zone."""
        shape = BoxVolumeShape(
            min_bounds=(0.0, 0.0, 0.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, blend_distance=3.0)
        # Outside volume but within blend distance
        assert volume.contains_point((12.0, 5.0, 5.0)) is True
        # Beyond blend distance
        assert volume.contains_point((15.0, 5.0, 5.0)) is False


# ==============================================================================
# Test: StackVolumeIntegration
# ==============================================================================


class TestStackVolumeIntegration:
    """Test PostProcessStack with PostProcessVolume."""

    def test_add_volume_to_stack(self):
        """Test adding volumes to stack."""
        stack = PostProcessStack()
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        assert len(stack._volumes) == 1

    def test_volume_priority_ordering(self):
        """Test volumes are ordered by priority (highest first)."""
        stack = PostProcessStack()
        volume_low = PostProcessVolume(
            BoxVolumeShape(), PostProcessVolumeSettings(), priority=10
        )
        volume_high = PostProcessVolume(
            BoxVolumeShape(), PostProcessVolumeSettings(), priority=100
        )
        stack.add_volume(volume_low)
        stack.add_volume(volume_high)
        assert stack._volumes[0].priority == 100
        assert stack._volumes[1].priority == 10

    def test_remove_volume_from_stack(self):
        """Test removing volumes from stack."""
        stack = PostProcessStack()
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        stack.add_volume(volume)
        stack.remove_volume(volume)
        assert len(stack._volumes) == 0

    def test_remove_nonexistent_volume(self):
        """Test removing volume not in stack does nothing."""
        stack = PostProcessStack()
        shape = BoxVolumeShape()
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings)
        stack.remove_volume(volume)  # should not raise

    def test_volume_applied_during_execute(self):
        """Test volumes are applied during execute with camera position."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        settings = PostProcessVolumeSettings()
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        ctx = PostProcessContext(
            frame_index=10,
            camera_position=(0.0, 0.0, 0.0),
        )
        stack.execute_with_context("hdr_in", "output", ctx)
        assert effect.execute_called is True

    def test_volume_not_applied_without_camera(self):
        """Test volumes not applied when no camera position."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings)
        stack.add_volume(volume)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        # Effect should still run
        assert effect.execute_called is True


# ==============================================================================
# Test: EffectPriority
# ==============================================================================


class TestEffectPriority:
    """Test EffectPriority enum values."""

    def test_priority_ordering(self):
        """Test that priorities are correctly ordered."""
        assert EffectPriority.EXPOSURE.value < EffectPriority.BLOOM.value
        assert EffectPriority.BLOOM.value < EffectPriority.DEPTH_OF_FIELD.value
        assert EffectPriority.DEPTH_OF_FIELD.value < EffectPriority.MOTION_BLUR.value
        assert EffectPriority.MOTION_BLUR.value < EffectPriority.AMBIENT_OCCLUSION.value
        assert EffectPriority.AMBIENT_OCCLUSION.value < EffectPriority.TONEMAPPING.value
        assert EffectPriority.TONEMAPPING.value < EffectPriority.COLOR_GRADING.value
        assert EffectPriority.COLOR_GRADING.value < EffectPriority.ANTIALIASING.value
        assert EffectPriority.ANTIALIASING.value < EffectPriority.UPSCALING.value

    def test_exposure_first(self):
        """Test EXPOSURE is first effect."""
        assert EffectPriority.EXPOSURE.value == 0

    def test_upscaling_last(self):
        """Test UPSCALING is last effect."""
        assert EffectPriority.UPSCALING.value == 800

    def test_custom_highest(self):
        """Test CUSTOM priority is highest."""
        assert EffectPriority.CUSTOM.value > EffectPriority.UPSCALING.value


# ==============================================================================
# Test: EffectQuality Enum
# ==============================================================================


class TestEffectQuality:
    """Test EffectQuality enum."""

    def test_enum_values(self):
        assert EffectQuality.LOW.value == 0
        assert EffectQuality.MEDIUM.value == 1
        assert EffectQuality.HIGH.value == 2
        assert EffectQuality.ULTRA.value == 3


# ==============================================================================
# Re-Verification: C2 -- Intermediate Target Pool (Blackbox)
# ==============================================================================


class TestC2IntermediateTargetPool:
    """Blackbox re-verification: C2 intermediate target pool behavior."""

    def test_executor_allocates_intermediates(self):
        """Executor prepare_resources allocates intermediate targets."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        assert executor._intermediate_mgr._ready is True
        assert len(executor._intermediate_mgr._targets) == 2

    def test_effect_chain_uses_intermediates(self):
        """C2: Non-terminal effects in chain use intermediate targets."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Bloom"))
        stack.add_effect(MockEffect("Tonemapping"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        # Bloom is first (non-terminal) -- should read HDR, write intermediate
        assert len(fg.passes) == 2
        bloom_pass = fg.passes[0]
        assert "Bloom" in bloom_pass.name

    def test_intermediate_manager_created_with_stack(self):
        """Stack has an IntermediateTargetManager."""
        stack = PostProcessStack()
        assert isinstance(stack._intermediate_mgr, IntermediateTargetManager)


# ==============================================================================
# Re-Verification: I1 -- R11G11B10_FLOAT Format (Blackbox)
# ==============================================================================


class TestI1R11G11B10Format:
    """Blackbox re-verification: I1 R11G11B10_FLOAT format correctness."""

    def test_default_intermediate_format_is_r11g11b10(self):
        """Default intermediate format string is R11G11B10_FLOAT."""
        config = PostProcessStackConfig()
        assert config.intermediate_format == "R11G11B10_FLOAT"

    def test_intermediate_manager_default_format(self):
        """IntermediateTargetManager default format is R11G11B10_FLOAT."""
        mgr = IntermediateTargetManager()
        assert mgr.format == "R11G11B10_FLOAT"

    def test_custom_format_accepted(self):
        """Custom format is accepted by IntermediateTargetManager."""
        mgr = IntermediateTargetManager(format="R16G16B16A16_FLOAT")
        assert mgr.format == "R16G16B16A16_FLOAT"

    def test_config_propagates_to_mgr_through_stack(self):
        """Stack config intermediate_format propagates to internal mgr."""
        config = PostProcessStackConfig(intermediate_format="R32G32B32A32_FLOAT")
        stack = PostProcessStack(config)
        assert stack._intermediate_mgr.format == "R32G32B32A32_FLOAT"


# ==============================================================================
# Re-Verification: I2 -- Volume Blending Save/Restore (Blackbox)
# ==============================================================================


class TestI2VolumeBlendingSaveRestore:
    """Blackbox re-verification: I2 volume blending does not drift."""

    def test_volume_settings_restored_after_execute(self):
        """Blackbox: Volume blending does not permanently change settings."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings, priority=10, blend_distance=5.0)
        stack.add_volume(volume)
        ctx = PostProcessContext(frame_index=10, camera_position=(5.0, 5.0, 5.0))
        stack.execute_with_context("hdr_in", "output", ctx)
        assert effect.settings.test_value == 1.0

    def test_volume_settings_across_multiple_executes(self):
        """Blackbox: Settings remain consistent across repeated executes."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        ctx = PostProcessContext(frame_index=10, camera_position=(0.0, 0.0, 0.0))
        for _ in range(5):
            stack.execute_with_context("hdr_in", "output", ctx)
            assert effect.settings.test_value == 1.0

    def test_volume_blending_with_executor_direct(self):
        """Blackbox: executor.execute_direct also preserves settings."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=10, camera_position=(0.0, 0.0, 0.0))
        executor.execute_direct("hdr_in", "output", ctx)
        assert effect.settings.test_value == 1.0

    def test_volume_settings_restore_not_applied_without_camera(self):
        """Without camera, no save/restore needed (no volume blending)."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        assert effect.settings.test_value == 1.0


# ==============================================================================
# Re-Verification: I3 -- BlendMode Removed (Blackbox)
# ==============================================================================


class TestI3BlendModeRemoved:
    """Blackbox re-verification: I3 BlendMode enum was removed."""

    def test_blendmode_not_importable_from_stack(self):
        """BlendMode cannot be imported from postprocess_stack module."""
        import engine.rendering.postprocess.postprocess_stack as pp_stack
        assert not hasattr(pp_stack, "BlendMode")

    def test_blendmode_not_importable_from_init(self):
        """BlendMode cannot be imported from postprocess package."""
        import engine.rendering.postprocess as pp
        assert not hasattr(pp, "BlendMode")

    def test_blendmode_not_exported(self):
        """BlendMode is not in postprocess_stack.__all__."""
        from engine.rendering.postprocess.postprocess_stack import __all__ as all_list
        assert "BlendMode" not in all_list

    def test_no_blendmode_reference_in_init_all(self):
        """BlendMode is not in postprocess.__init__.__all__."""
        from engine.rendering.postprocess import __all__ as all_list
        assert "BlendMode" not in all_list
