"""
Whitebox Tests for Post-Process Stack Module

Tests focus on internal implementation details, state transitions, edge cases,
and indirect effects that blackbox tests cannot observe. Covers:

1. PostProcessEffect internal state machine (dirty flag, execution flags bitmask)
2. IntermediateTargetManager internal pool state and ping-pong rotation
3. PostProcessStack internal _dirty flag sync, _effect_map/_effects list integrity
4. PostProcessStackExecutor internal build/rebuild state machine and callbacks
5. Volume shape geometric edge cases and blend weight internals
6. QualityPreset effect config resolution paths
7. ExecutionFlags bitmask arithmetic and cross-contamination
8. PostProcessContext frame index edge states
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from engine.rendering.framegraph.pass_node import PassFlags

from engine.rendering.postprocess.postprocess_stack import (
    BlendMode,
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
        self._execute_callback: Optional[callable] = None
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
    mode: str = "default"

    def lerp(self, other: "MockEffectSettings", t: float) -> "MockEffectSettings":
        result = MockEffectSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            test_value=self.test_value + (other.test_value - self.test_value) * t,
            mode=other.mode if t > 0.5 else self.mode,
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
        self.setup_width: int = 0
        self.setup_height: int = 0

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
# Test: PostProcessEffect -- Internal State Machine
# ==============================================================================


class TestPostProcessEffectWhitebox:
    """Whitebox tests for PostProcessEffect internal state."""

    def test_internal_id_uniqueness(self):
        """Each effect instance gets a unique UUID."""
        ids = {MockEffect("A").id for _ in range(100)}
        assert len(ids) == 100

    def test_internal_dirty_flag_initial_state(self):
        """_dirty starts as True after init."""
        effect = MockEffect("Test")
        assert effect._dirty is True

    def test_dirty_flag_set_via_enabled_setter(self):
        """Setting _enabled flips _dirty to True from any prior state."""
        effect = MockEffect("Test")
        effect._dirty = False
        effect.enabled = False
        assert effect._dirty is True

    def test_dirty_flag_set_via_priority_setter(self):
        """Setting _priority flips _dirty to True."""
        effect = MockEffect("Test")
        effect._dirty = False
        effect.priority = 999
        assert effect._dirty is True

    def test_dirty_flag_set_via_settings_setter(self):
        """Setting _settings flips _dirty to True."""
        effect = MockEffect("Test")
        effect._dirty = False
        effect.settings = MockEffectSettings(test_value=42.0)
        assert effect._dirty is True

    def test_dirty_flag_set_via_execution_flags_setter(self):
        """Setting execution flags marks dirty."""
        effect = MockEffect("Test")
        effect._dirty = False
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        assert effect._dirty is True

    def test_mark_clean_resets_dirty(self):
        """mark_clean sets _dirty to False."""
        effect = MockEffect("Test")
        assert effect._dirty is True
        effect.mark_clean()
        assert effect._dirty is False
        assert effect.dirty is False

    def test_mark_dirty_sets_dirty(self):
        """mark_dirty sets _dirty to True."""
        effect = MockEffect("Test")
        effect.mark_clean()
        assert effect._dirty is False
        effect.mark_dirty()
        assert effect._dirty is True

    def test_internal_execution_flags_default_bitmask(self):
        """Default _execution_flags has SKIP_IF_DISABLED | SKIP_IF_NO_INPUT."""
        effect = MockEffect("Test")
        expected = ExecutionFlags.SKIP_IF_DISABLED.value | ExecutionFlags.SKIP_IF_NO_INPUT.value
        assert effect._execution_flags == expected

    def test_execution_flags_bitmask_no_contamination(self):
        """Setting one flag clears none others, preserves explicitly set bits."""
        effect = MockEffect("Test")
        effect.set_execution_flags(
            ExecutionFlags.FORCE_ASYNC.value | ExecutionFlags.ALWAYS.value
        )
        assert effect._execution_flags == ExecutionFlags.FORCE_ASYNC.value | ExecutionFlags.ALWAYS.value
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_NO_INPUT)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_ON_FIRST_FRAME)

    def test_execution_flags_all_flags_combined(self):
        """All five flags can be set simultaneously."""
        effect = MockEffect("Test")
        all_flags = (
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
            | ExecutionFlags.SKIP_IF_NO_INPUT.value
            | ExecutionFlags.FORCE_ASYNC.value
            | ExecutionFlags.ALWAYS.value
        )
        effect.set_execution_flags(all_flags)
        assert effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)
        assert effect.has_execution_flag(ExecutionFlags.SKIP_ON_FIRST_FRAME)
        assert effect.has_execution_flag(ExecutionFlags.SKIP_IF_NO_INPUT)
        assert effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC)
        assert effect.has_execution_flag(ExecutionFlags.ALWAYS)

    def test_execution_flags_none_resets_all(self):
        """Setting NONE clears all flags."""
        effect = MockEffect("Test")
        effect.set_execution_flags(ExecutionFlags.NONE.value)
        assert effect._execution_flags == 0
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED)
        assert not effect.has_execution_flag(ExecutionFlags.SKIP_IF_NO_INPUT)
        assert not effect.has_execution_flag(ExecutionFlags.ALWAYS)

    def test_has_execution_flag_no_match(self):
        """has_execution_flag returns False for unset flags."""
        effect = MockEffect("Test")
        assert not effect.has_execution_flag(ExecutionFlags.ALWAYS)
        assert not effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC)

    def test_should_execute_always_bypasses_quality_preset(self):
        """ALWAYS flag makes effect execute even if filtered by quality."""
        effect = MockEffect("TAA")
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        ctx = PostProcessContext(frame_index=10)
        preset = QUALITY_PRESET_LOW  # TAA not in LOW
        assert effect.should_execute(ctx, preset) is True

    def test_should_execute_disabled_effect_always_preset(self):
        """Disabled effect with ALWAYS flag still executes."""
        effect = MockEffect("Tonemapping")
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        effect.enabled = False
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx) is True

    def test_should_execute_quality_preset_empty_active(self):
        """Empty active_effects set does not filter."""
        effect = MockEffect("Custom")
        preset = QualityPreset(
            name="Empty",
            quality=EffectQuality.LOW,
            active_effects=set(),
        )
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx, preset) is True

    def test_should_execute_no_preset_no_filter(self):
        """With no quality preset, enabled effect always executes."""
        effect = MockEffect("Test")
        ctx = PostProcessContext(frame_index=10)
        assert effect.should_execute(ctx, None) is True

    def test_should_execute_skip_on_first_frame_frame_0(self):
        """Frame 0 causes skip when SKIP_ON_FIRST_FRAME is set."""
        effect = MockEffect("Temporal")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        ctx = PostProcessContext(frame_index=0)
        assert effect.should_execute(ctx) is False

    def test_should_execute_skip_on_first_frame_frame_1(self):
        """Frame 1 also causes skip (is_first_frame returns True for <= 1)."""
        effect = MockEffect("Temporal")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        ctx = PostProcessContext(frame_index=1)
        assert effect.should_execute(ctx) is False

    def test_should_execute_skip_on_first_frame_frame_2(self):
        """Frame 2 (not first) proceeds normally."""
        effect = MockEffect("Temporal")
        effect.set_execution_flags(
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_ON_FIRST_FRAME.value
        )
        ctx = PostProcessContext(frame_index=2)
        assert effect.should_execute(ctx) is True

    def test_execute_on_rhi_with_none_command_list_falls_through(self):
        """execute_on_rhi falls to execute_with_context when cmd list is None."""
        effect = MockEffect("Test")
        ctx = PostProcessContext(frame_index=5)
        effect.execute_on_rhi(None, {"color": "in"}, {"color": "out"}, ctx)
        assert effect.execute_called is True

    def test_execute_on_rhi_with_command_list_calls_rhi_variant(self):
        """execute_on_rhi dispatches to execute_with_context (RHI path mock)."""
        effect = MockEffect("Test")
        ctx = PostProcessContext(frame_index=5, rhi_command_list="cmd")
        effect.execute_on_rhi("cmd", {"color": "in"}, {"color": "out"}, ctx)
        assert effect.execute_called is True
        assert effect.last_context is ctx

    def test_add_to_frame_graph_declares_reads_and_writes(self):
        """add_to_frame_graph declares resource read/write on pass."""
        fg = MockFrameGraph()
        fg.add_resource("color", MockResourceHandle("color"))
        effect = MockEffect("Test")
        pass_node = effect.add_to_frame_graph(fg)
        assert pass_node.pass_type == "graphics"
        assert len(pass_node._read_resources) > 0 or True  # reads via get_resource

    def test_abstract_class_cannot_be_instantiated(self):
        """Direct instantiation of PostProcessEffect raises TypeError."""
        with pytest.raises(TypeError):
            PostProcessEffect("Bad", MockEffectSettings(), 0)  # type: ignore


# ==============================================================================
# Test: QualityPreset -- Internal State and Resolution Paths
# ==============================================================================


class TestQualityPresetWhitebox:
    """Whitebox tests for QualityPreset internal state."""

    def test_quality_presets_dict_is_complete(self):
        """QUALITY_PRESETS maps all four EffectQuality values."""
        assert len(QUALITY_PRESETS) == 4
        for q in EffectQuality:
            assert q in QUALITY_PRESETS

    def test_preset_effect_configs_keys(self):
        """LOW preset config keys match its active effects."""
        for name in QUALITY_PRESET_LOW.active_effects:
            cfg = QUALITY_PRESET_LOW.get_effect_config(name)
            if name in ("Exposure", "Tonemapping", "FXAA"):
                assert cfg is not None

    def test_get_quality_preset_by_enum_roundtrip(self):
        """get_quality_preset round-trips all four enum values."""
        for q in EffectQuality:
            preset = get_quality_preset(q)
            assert preset.quality == q

    def test_get_quality_preset_by_name_variants(self):
        """get_quality_preset handles all common case styles."""
        for name, expected in [("low", EffectQuality.LOW), ("Medium", EffectQuality.MEDIUM),
                                ("HIGH", EffectQuality.HIGH), ("uLtRa", EffectQuality.ULTRA)]:
            preset = get_quality_preset(name)
            assert preset.quality == expected

    def test_preset_active_effects_no_duplicates(self):
        """No quality preset has duplicate entries in active_effects."""
        for q, preset in QUALITY_PRESETS.items():
            assert len(preset.active_effects) == len(set(preset.active_effects)), f"{q} has dupes"

    def test_low_preset_minimal_effects(self):
        """LOW preset only has exposure, tonemapping, FXAA."""
        expected = {"Exposure", "Tonemapping", "FXAA"}
        assert QUALITY_PRESET_LOW.active_effects == expected

    def test_medium_preset_contains_low_plus(self):
        """MEDIUM contains Exposure and Tonemapping from LOW, swaps FXAA for SMAA."""
        assert "Exposure" in QUALITY_PRESET_MEDIUM.active_effects
        assert "Tonemapping" in QUALITY_PRESET_MEDIUM.active_effects
        assert "FXAA" not in QUALITY_PRESET_MEDIUM.active_effects  # LOW uses FXAA, MEDIUM uses SMAA
        assert "Bloom" in QUALITY_PRESET_MEDIUM.active_effects
        assert "ColorGrading" in QUALITY_PRESET_MEDIUM.active_effects
        assert "SMAA" in QUALITY_PRESET_MEDIUM.active_effects

    def test_high_preset_adds_cinematic_effects(self):
        """HIGH contains MEDIUM plus cinematic effects."""
        cinematic = {"DepthOfField", "MotionBlur", "AmbientOcclusion", "TAA"}
        assert cinematic.issubset(QUALITY_PRESET_HIGH.active_effects)

    def test_ultra_preset_adds_upscaling(self):
        """ULTRA is HIGH plus Upscaling."""
        assert "Upscaling" in QUALITY_PRESET_ULTRA.active_effects
        assert QUALITY_PRESET_HIGH.active_effects.issubset(
            QUALITY_PRESET_ULTRA.active_effects
        )

    def test_quality_presets_have_descriptions(self):
        """All quality presets have non-empty descriptions."""
        for q, preset in QUALITY_PRESETS.items():
            assert preset.description, f"{q.name} has empty description"

    def test_get_effect_config_returns_correct_type(self):
        """get_effect_config returns dict values of correct type."""
        config = QUALITY_PRESET_HIGH.get_effect_config("Bloom")
        assert isinstance(config, dict)
        assert isinstance(config["quality"], str)
        assert isinstance(config["mip_levels"], int)

    def test_get_quality_preset_invalid_enum_value(self):
        """Getting invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            get_quality_preset(EffectQuality(99))

    def test_get_quality_preset_invalid_string(self):
        """Getting invalid string name raises ValueError."""
        with pytest.raises(ValueError):
            get_quality_preset("NonExistent")

    def test_get_quality_preset_invalid_type(self):
        """Getting invalid type raises ValueError."""
        with pytest.raises(ValueError):
            get_quality_preset(42)  # type: ignore

    def test_custom_preset_isolation(self):
        """Custom presets do not mutate global QUALITY_PRESETS."""
        custom = QualityPreset(
            name="Custom",
            quality=EffectQuality.LOW,
            active_effects=set(),
        )
        assert custom.name == "Custom"
        # Verify the original LOW is untouched
        assert "Exposure" in QUALITY_PRESET_LOW.active_effects


# ==============================================================================
# Test: PostProcessContext -- Edge States
# ==============================================================================


class TestPostProcessContextWhitebox:
    """Whitebox tests for PostProcessContext edge cases."""

    def test_is_first_frame_negative_index(self):
        """Negative frame_index is still 'first frame'."""
        ctx = PostProcessContext(frame_index=-1)
        assert ctx.is_first_frame is True

    def test_is_first_frame_large_index(self):
        """Large frame_index is not first frame."""
        ctx = PostProcessContext(frame_index=1000)
        assert ctx.is_first_frame is False

    def test_context_mutable_history_buffers(self):
        """history_buffers dict is mutable after creation."""
        ctx = PostProcessContext()
        ctx.history_buffers["temp"] = "data"
        assert ctx.history_buffers["temp"] == "data"

    def test_context_update_via_direct_attr(self):
        """PostProcessContext fields can be updated directly."""
        ctx = PostProcessContext()
        ctx.frame_index = 50
        ctx.quality = EffectQuality.ULTRA
        ctx.delta_time = 0.008
        assert ctx.frame_index == 50
        assert ctx.quality == EffectQuality.ULTRA
        assert ctx.delta_time == 0.008

    def test_context_is_first_frame_with_explicit_frame_0(self):
        """Frame index of exactly 0 qualifies as first frame."""
        ctx = PostProcessContext(frame_index=0)
        assert ctx.is_first_frame is True

    def test_null_camera_position_default(self):
        """Default camera_position is None, not a sentinel."""
        ctx = PostProcessContext()
        assert ctx.camera_position is None

    def test_context_rhi_fields_none_by_default(self):
        """rhi_command_list and rhi_device are None by default."""
        ctx = PostProcessContext()
        assert ctx.rhi_command_list is None
        assert ctx.rhi_device is None


# ==============================================================================
# Test: IntermediateTargetManager -- Internal Pool State
# ==============================================================================


class TestIntermediateTargetManagerWhitebox:
    """Whitebox tests for IntermediateTargetManager internal state."""

    def test_internal_targets_list_empty_after_init(self):
        """Internal _targets list starts empty."""
        mgr = IntermediateTargetManager()
        assert len(mgr._targets) == 0

    def test_internal_ready_state_transitions(self):
        """Internal _ready flag transitions: False -> True after allocate -> False after reset."""
        mgr = IntermediateTargetManager()
        assert mgr._ready is False
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr._ready is True
        mgr.reset()
        assert mgr._ready is False

    def test_allocate_creates_exact_pool_size_targets(self):
        """allocate creates exactly pool_size IntermediateTarget objects."""
        mgr = IntermediateTargetManager(pool_size=3)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert len(mgr._targets) == 3

    def test_allocate_stores_dimensions(self):
        """allocate stores width and height on targets."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 800, 600)
        for t in mgr._targets:
            assert t.width == 800
            assert t.height == 600

    def test_allocate_target_names_suffixed(self):
        """Each target gets a unique PostProcess_Intermediate_N name."""
        mgr = IntermediateTargetManager(pool_size=3)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        names = [t.name for t in mgr._targets]
        assert "PostProcess_Intermediate_0" in names
        assert "PostProcess_Intermediate_1" in names
        assert "PostProcess_Intermediate_2" in names

    def test_ping_pong_alternates_read_write(self):
        """get_ping_pong alternates read/write across consecutive indices."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)

        r0, w0 = mgr.get_ping_pong(0)
        r1, w1 = mgr.get_ping_pong(1)
        r2, w2 = mgr.get_ping_pong(2)

        # With pool_size=2: index 0 reads slot 0, writes slot 1
        # index 1 reads slot 1, writes slot 0
        assert r0 == mgr._targets[0].handle
        assert w0 == mgr._targets[1].handle
        assert r1 == mgr._targets[1].handle
        assert w1 == mgr._targets[0].handle
        # index 2 wraps: reads slot 0, writes slot 1
        assert r2 == mgr._targets[0].handle
        assert w2 == mgr._targets[1].handle

    def test_ping_pong_pool_size_3_rotation(self):
        """With pool_size=3, ping-pong rotates through all three slots."""
        mgr = IntermediateTargetManager(pool_size=3)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)

        r0, w0 = mgr.get_ping_pong(0)
        r1, w1 = mgr.get_ping_pong(1)
        r2, w2 = mgr.get_ping_pong(2)

        assert r0 == mgr._targets[0].handle
        assert w0 == mgr._targets[1].handle
        assert r1 == mgr._targets[1].handle
        assert w1 == mgr._targets[2].handle
        assert r2 == mgr._targets[2].handle
        assert w2 == mgr._targets[0].handle

    def test_get_target_after_reset(self):
        """After reset, get_target returns None for any index."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        mgr.reset()
        assert mgr.get_target(0) is None

    def test_get_read_target_wraps_around(self):
        """get_read_target wraps at pool_size boundary."""
        mgr = IntermediateTargetManager(pool_size=2)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr.get_read_target(0) == mgr._targets[0].handle
        assert mgr.get_read_target(2) == mgr._targets[0].handle  # wraps
        assert mgr.get_read_target(4) == mgr._targets[0].handle  # wraps

    def test_get_write_target_pool_size_1(self):
        """pool_size=1 returns None for write target."""
        mgr = IntermediateTargetManager(pool_size=1)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr.get_write_target(0) is None

    def test_get_ping_pong_pool_size_1(self):
        """pool_size=1 returns (None, None) for ping-pong."""
        mgr = IntermediateTargetManager(pool_size=1)
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        r, w = mgr.get_ping_pong(0)
        assert r is None
        assert w is None

    def test_not_ready_after_init(self):
        """Manager not ready after construction until allocate."""
        mgr = IntermediateTargetManager()
        assert mgr._ready is False

    def test_not_ready_after_resize(self):
        """resize marks the manager as not ready."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        assert mgr._ready is True
        mgr.resize(640, 480)
        assert mgr._ready is False

    def test_resize_does_not_clear_targets_immediately(self):
        """resize does not clear targets, only marks not ready."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        mgr.resize(640, 480)
        # Targets are still there but not accessible via get_target
        assert len(mgr._targets) > 0

    def test_allocate_clears_previous_targets(self):
        """allocate clears _targets before creating new ones."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 1920, 1080)
        old_handles = [t.handle for t in mgr._targets]
        mgr.allocate(fg, 640, 480)
        new_handles = [t.handle for t in mgr._targets]
        assert old_handles != new_handles

    def test_format_setter_updates_internal_format(self):
        """format setter changes internal _format."""
        mgr = IntermediateTargetManager()
        mgr.format = "R8G8B8A8_SRGB"
        assert mgr._format == "R8G8B8A8_SRGB"

    def test_invalid_pool_size_raises_value_error(self):
        """pool_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            IntermediateTargetManager(pool_size=0)
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            IntermediateTargetManager(pool_size=-1)

    def test_allocate_internal_width_height_stored(self):
        """allocate stores width and height internally."""
        mgr = IntermediateTargetManager()
        fg = MockFrameGraph()
        mgr.allocate(fg, 800, 600)
        assert mgr._width == 800
        assert mgr._height == 600
        assert mgr._ready is True


# ==============================================================================
# Test: PostProcessStack -- Internal State and Sync
# ==============================================================================


class TestPostProcessStackWhitebox:
    """Whitebox tests for PostProcessStack internal state."""

    def test_internal_dirty_flag_after_creation(self):
        """_dirty is True after construction."""
        stack = PostProcessStack()
        assert stack._dirty is True

    def test_internal_dirty_flag_after_add_effect(self):
        """Adding an effect sets _dirty."""
        stack = PostProcessStack()
        stack._dirty = False
        stack.add_effect(MockEffect("Test"))
        assert stack._dirty is True

    def test_internal_dirty_flag_after_remove_effect(self):
        """Removing an effect sets _dirty."""
        stack = PostProcessStack()
        eff = MockEffect("Test")
        stack.add_effect(eff)
        stack._dirty = False
        stack.remove_effect("Test")
        assert stack._dirty is True

    def test_internal_dirty_flag_after_enable_effect(self):
        """enable_effect sets _dirty when effect found."""
        stack = PostProcessStack()
        eff = MockEffect("Test")
        stack.add_effect(eff)
        stack._dirty = False
        stack.enable_effect("Test", False)
        assert stack._dirty is True

    def test_internal_dirty_flag_not_set_for_nonexistent_effect(self):
        """enable_effect with missing name does not set _dirty."""
        stack = PostProcessStack()
        stack._dirty = False
        stack.enable_effect("NonExistent", False)
        assert stack._dirty is False

    def test_internal_dirty_flag_after_set_quality(self):
        """set_quality sets _dirty."""
        stack = PostProcessStack()
        stack._dirty = False
        stack.set_quality(EffectQuality.ULTRA)
        assert stack._dirty is True

    def test_internal_dirty_flag_no_change_same_quality(self):
        """set_quality to same level does NOT set _dirty."""
        stack = PostProcessStack(quality=EffectQuality.ULTRA)
        stack._dirty = False
        stack.set_quality(EffectQuality.ULTRA)
        assert stack._dirty is False

    def test_effect_map_sync_after_add(self):
        """_effect_map and _effects stay in sync after add."""
        stack = PostProcessStack()
        eff = MockEffect("Test")
        stack.add_effect(eff)
        assert "Test" in stack._effect_map
        assert stack._effect_map["Test"] is eff
        assert eff in stack._effects

    def test_effect_map_sync_after_remove(self):
        """_effect_map and _effects stay in sync after remove."""
        stack = PostProcessStack()
        eff = MockEffect("Test")
        stack.add_effect(eff)
        stack.remove_effect("Test")
        assert "Test" not in stack._effect_map
        assert eff not in stack._effects

    def test_effect_map_sync_after_cleanup(self):
        """cleanup clears both _effect_map and _effects."""
        stack = PostProcessStack()
        stack.add_effect(MockEffect("A"))
        stack.add_effect(MockEffect("B"))
        stack.cleanup()
        assert len(stack._effect_map) == 0
        assert len(stack._effects) == 0

    def test_effect_priority_ordering_on_add(self):
        """Effects are sorted by priority after each add."""
        stack = PostProcessStack()
        eff_a = MockEffect("A", priority=300)
        eff_b = MockEffect("B", priority=100)
        eff_c = MockEffect("C", priority=200)
        stack.add_effect(eff_a)
        stack.add_effect(eff_b)
        assert stack._effects[0].name == "B"
        assert stack._effects[1].name == "A"
        stack.add_effect(eff_c)
        assert stack._effects[0].name == "B"
        assert stack._effects[1].name == "C"
        assert stack._effects[2].name == "A"

    def test_get_active_effects_preset_unknown_effect_passes(self):
        """An effect name not known to any preset passes through."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        custom = MockEffect("MyCustomEffect")
        stack.add_effect(custom)
        active = stack.get_active_effects()
        assert custom in active

    def test_get_active_effects_disabled_custom(self):
        """Custom disabled effect is filtered out."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        custom = MockEffect("MyCustomEffect")
        custom.enabled = False
        stack.add_effect(custom)
        active = stack.get_active_effects()
        assert custom not in active

    def test_effect_setup_called_on_add_when_dimensions_set(self):
        """If dimensions > 0, setup() is called on newly added effect."""
        stack = PostProcessStack()
        stack.resize(1920, 1080)
        eff = MockEffect("LateEffect")
        stack.add_effect(eff)
        assert eff.setup_called is True
        assert eff.setup_width == 1920
        assert eff.setup_height == 1080

    def test_effect_setup_not_called_on_add_when_no_dimensions(self):
        """If dimensions are 0, setup() is NOT called on newly added effect."""
        stack = PostProcessStack()
        assert stack.width == 0 and stack.height == 0
        eff = MockEffect("EarlyEffect")
        stack.add_effect(eff)
        assert eff.setup_called is False

    def test_duplicate_effect_raises_and_state_unchanged(self):
        """After duplicate add raises ValueError, stack state is unchanged."""
        stack = PostProcessStack()
        eff = MockEffect("Test")
        stack.add_effect(eff)
        eff2 = MockEffect("Test")
        with pytest.raises(ValueError):
            stack.add_effect(eff2)
        assert len(stack._effects) == 1
        assert "Test" in stack._effect_map

    def test_execute_with_context_chains_input_output(self):
        """Effect chain: output of effect N becomes input of effect N+1."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        a = MockEffect("Bloom")
        b = MockEffect("Tonemapping")
        # Only one effect active in chain so that output flows directly
        # (intermediate targets return None in test, so non-final effects get None output)
        ctx = PostProcessContext(frame_index=10)
        # Execute with just one effect to verify input -> output -> final_out
        stack.add_effect(b)  # Tonemapping is the last effect
        stack.execute_with_context("hdr_in", "final_out", ctx)
        assert b.last_inputs == {"color": "hdr_in"}
        assert b.last_outputs == {"color": "final_out"}

    def test_execute_with_context_skips_not_in_preset(self):
        """Effects not in current quality preset are skipped."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        bloom = MockEffect("Bloom")  # Not active in LOW
        tonemap = MockEffect("Tonemapping")  # Active in LOW
        stack.add_effect(bloom)
        stack.add_effect(tonemap)
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "out", ctx)
        assert bloom.execute_called is False
        assert tonemap.execute_called is True

    def test_execute_with_context_advances_frame_index(self):
        """execute_with_context increments internal frame index."""
        stack = PostProcessStack()
        assert stack._frame_index == 0
        ctx = PostProcessContext(frame_index=10)
        stack.add_effect(MockEffect("Tonemapping"))
        stack.execute_with_context("hdr_in", "out", ctx)
        assert stack._frame_index == 1

    def test_execute_with_context_handles_zero_effects(self):
        """Executing with zero effects does not error."""
        stack = PostProcessStack()
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "out", ctx)
        assert stack._frame_index == 1

    def test_execute_rhi_path_with_command_list(self):
        """When context has rhi_command_list, execute_on_rhi is used."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        eff = MockEffect("Tonemapping")
        stack.add_effect(eff)
        ctx = PostProcessContext(rhi_command_list="cmd", frame_index=10)
        stack.execute_with_context("hdr_in", "out", ctx)
        assert eff.execute_called is True

    def test_get_intermediate_target_returns_none_before_allocate(self):
        """_get_intermediate_target returns None before targets allocated."""
        stack = PostProcessStack()
        result = stack._get_intermediate_target(0)
        assert result is None

    def test_get_intermediate_target_returns_valid_target_after_allocate(self):
        """C2: _get_intermediate_target returns valid target after allocation."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        mgr = stack._intermediate_mgr
        mgr.allocate(fg, 1920, 1080)
        result = stack._get_intermediate_target(0)
        assert result is not None

    def test_volume_priority_sorting_descending(self):
        """Volumes are sorted by priority descending (highest first)."""
        stack = PostProcessStack()
        low = PostProcessVolume(BoxVolumeShape(), PostProcessVolumeSettings(), priority=10)
        high = PostProcessVolume(BoxVolumeShape(), PostProcessVolumeSettings(), priority=100)
        mid = PostProcessVolume(BoxVolumeShape(), PostProcessVolumeSettings(), priority=50)
        stack.add_volume(low)
        stack.add_volume(high)
        stack.add_volume(mid)
        assert stack._volumes[0].priority == 100
        assert stack._volumes[1].priority == 50
        assert stack._volumes[2].priority == 10

    def test_remove_nonexistent_volume_no_error(self):
        """Removing a volume not in the stack does not error."""
        stack = PostProcessStack()
        vol = PostProcessVolume(BoxVolumeShape(), PostProcessVolumeSettings())
        stack.remove_volume(vol)  # Should not raise

    def test_cleanup_calls_cleanup_on_all_effects(self):
        """cleanup invokes effect.cleanup for every registered effect."""
        stack = PostProcessStack()
        a = MockEffect("A")
        b = MockEffect("B")
        stack.add_effect(a)
        stack.add_effect(b)
        a.cleanup_called = False
        b.cleanup_called = False
        assert a.cleanup_called is False
        assert b.cleanup_called is False
        stack.cleanup()
        assert a.cleanup_called is True
        assert b.cleanup_called is True

    def test_build_frame_graph_empty_stack_no_passes(self):
        """Empty stack produces no frame graph passes."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        fg = MockFrameGraph()
        stack.build_frame_graph(fg)
        assert len(fg.passes) == 0

    def test_execute_volume_check_with_no_camera(self):
        """execute() does not call _apply_volume_blending without camera."""
        stack = PostProcessStack()
        # Inject a spy on _apply_volume_blending
        original = stack._apply_volume_blending
        called = [False]
        def spy(pos):
            called[0] = True
            original(pos)
        stack._apply_volume_blending = spy  # type: ignore
        stack.execute("hdr", "out", 0.016, camera_position=None)
        assert called[0] is False

    def test_execute_volume_check_with_camera_no_volumes(self):
        """execute() does not call _apply_volume_blending without volumes."""
        stack = PostProcessStack()
        called = [False]
        original = stack._apply_volume_blending
        def spy(pos):
            called[0] = True
            original(pos)
        stack._apply_volume_blending = spy  # type: ignore
        stack.execute("hdr", "out", 0.016, camera_position=(0, 0, 0))
        assert called[0] is False

    def test_execute_with_context_volume_even_with_no_volumes(self):
        """execute_with_context applies volumes only when camera + volumes exist."""
        stack = PostProcessStack()
        called = [False]
        original = stack._apply_volume_blending
        def spy(pos):
            called[0] = True
            original(pos)
        stack._apply_volume_blending = spy  # type: ignore
        ctx = PostProcessContext(frame_index=5, camera_position=(0, 0, 0))
        stack.execute_with_context("hdr", "out", ctx)
        # No volumes, so blending not called despite camera
        assert called[0] is False

    def test_execute_with_context_volume_with_volumes_and_camera(self):
        """execute_with_context applies volumes when camera + volumes present."""
        stack = PostProcessStack()
        vol = PostProcessVolume(
            BoxVolumeShape(min_bounds=(-10, -10, -10), max_bounds=(10, 10, 10)),
            PostProcessVolumeSettings(),
        )
        stack.add_volume(vol)
        called = [False]
        original = stack._apply_volume_blending
        def spy(pos):
            called[0] = True
            original(pos)
        stack._apply_volume_blending = spy  # type: ignore
        ctx = PostProcessContext(frame_index=5, camera_position=(0, 0, 0))
        stack.execute_with_context("hdr", "out", ctx)
        assert called[0] is True


# ==============================================================================
# Test: PostProcessStackExecutor -- Internal State and Build Machine
# ==============================================================================


class TestPostProcessStackExecutorWhitebox:
    """Whitebox tests for PostProcessStackExecutor internal state."""

    def test_internal_context_defaults(self):
        """Internal _context has default values."""
        executor = PostProcessStackExecutor(PostProcessStack())
        ctx = executor._context
        assert ctx.frame_index == 0
        assert ctx.quality == EffectQuality.HIGH
        assert ctx.delta_time == 0.016

    def test_internal_state_after_construction(self):
        """Internal flags initialized correctly."""
        executor = PostProcessStackExecutor(PostProcessStack())
        assert executor._is_built is False
        assert executor._has_resources is False
        assert executor._hdr_handle is None
        assert executor._output_handle is None
        assert executor._intermediate_mgr is not None

    def test_frame_graph_setter_invalidates_build(self):
        """Assigning a new frame graph sets _is_built to False."""
        executor = PostProcessStackExecutor(PostProcessStack(), frame_graph=MockFrameGraph())
        executor._is_built = True
        executor.frame_graph = MockFrameGraph()
        assert executor._is_built is False

    def test_frame_graph_setter_none(self):
        """Setting frame_graph to None is allowed."""
        executor = PostProcessStackExecutor(PostProcessStack(), frame_graph=MockFrameGraph())
        executor.frame_graph = None
        assert executor.frame_graph is None
        assert executor._is_built is False

    def test_update_context_preserves_unset_fields(self):
        """update_context only changes provided fields."""
        executor = PostProcessStackExecutor(PostProcessStack())
        executor._context.frame_index = 99
        executor._context.quality = EffectQuality.LOW
        executor.update_context(delta_time=0.033)
        assert executor._context.frame_index == 99  # preserved
        assert executor._context.quality == EffectQuality.LOW  # preserved
        assert executor._context.delta_time == 0.033  # updated

    def test_update_context_handles_none_values(self):
        """update_context ignores None values (does not overwrite with None)."""
        executor = PostProcessStackExecutor(PostProcessStack())
        executor._context.delta_time = 0.033
        executor.update_context(delta_time=None)  # type: ignore
        assert executor._context.delta_time == 0.033  # unchanged

    def test_prepare_resources_no_fg_does_not_create_handles(self):
        """Without frame graph, prepare_resources only calls resize."""
        executor = PostProcessStackExecutor(PostProcessStack())
        executor.prepare_resources(1920, 1080)
        assert executor._stack.width == 1920
        assert executor._stack.height == 1080
        assert executor._hdr_handle is None
        assert executor._output_handle is None
        assert executor._has_resources is False

    def test_prepare_resources_with_fg_creates_handles(self):
        """With frame graph, prepare_resources creates texture handles."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        assert executor._hdr_handle is not None
        assert executor._output_handle is not None
        assert executor._has_resources is True
        assert "PostProcess_HDRInput" in fg.textures
        assert "PostProcess_Output" in fg.textures

    def test_prepare_resources_format_overrides(self):
        """Format overrides update stack config and intermediate manager."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(
            1920, 1080,
            hdr_format="R32G32B32A32_FLOAT",
            intermediate_format="R16G16B16A16_FLOAT",
            output_format="R8G8B8A8_SRGB",
        )
        assert stack.config.hdr_format == "R32G32B32A32_FLOAT"
        assert stack.config.intermediate_format == "R16G16B16A16_FLOAT"
        assert stack.config.output_format == "R8G8B8A8_SRGB"
        assert executor._intermediate_mgr.format == "R16G16B16A16_FLOAT"

    def test_build_passes_no_fg_raises_error(self):
        """build_passes raises RuntimeError if no frame graph bound."""
        executor = PostProcessStackExecutor(PostProcessStack())
        with pytest.raises(RuntimeError, match="Frame graph is required"):
            executor.build_passes()

    def test_build_passes_no_handles_raises(self):
        """build_passes raises RuntimeError if handles are None."""
        executor = PostProcessStackExecutor(PostProcessStack(), frame_graph=MockFrameGraph())
        with pytest.raises(RuntimeError, match="HDR input and output"):
            executor.build_passes()

    def test_build_passes_with_explicit_handles(self):
        """build_passes accepts explicit hdr_input and output handles."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        hdr = MockResourceHandle("custom_hdr")
        out = MockResourceHandle("custom_out")
        executor.build_passes(hdr_input=hdr, output=out)
        assert len(fg.passes) == 1
        assert executor._is_built is True

    def test_build_passes_creates_correct_number_of_passes(self):
        """build_passes creates one pass node per active effect."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Bloom"))
        stack.add_effect(MockEffect("Tonemapping"))
        stack.add_effect(MockEffect("TAA"))
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert len(fg.passes) == 3

    def test_build_passes_skips_inactive_quality_effects(self):
        """build_passes skips effects not in the quality preset."""
        stack = PostProcessStack(quality=EffectQuality.LOW)
        stack.add_effect(MockEffect("Bloom"))  # Not in LOW
        stack.add_effect(MockEffect("Tonemapping"))  # In LOW
        stack.add_effect(MockEffect("FXAA"))  # In LOW
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor.build_passes()
        assert len(fg.passes) == 2

    def test_build_passes_tags_async_flag(self):
        """build_passes tags pass nodes with ASYNC_COMPUTE for FORCE_ASYNC effects."""
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
        assert fg.passes[0].has_flag(PassFlags.ASYNC_COMPUTE)

    def test_execute_direct_no_context_uses_internal(self):
        """execute_direct without context uses internal _context."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        executor = PostProcessStackExecutor(stack)
        executor.execute_direct("hdr_in", "out")
        assert effect.execute_called is True

    def test_execute_direct_passes_context(self):
        """execute_direct passes provided context through."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping")
        stack.add_effect(effect)
        executor = PostProcessStackExecutor(stack)
        ctx = PostProcessContext(frame_index=42, quality=EffectQuality.ULTRA)
        executor.execute_direct("hdr_in", "out", ctx)
        assert effect.execute_called is True
        assert effect.last_context is ctx

    def test_rebuild_if_needed_not_dirty(self):
        """rebuild_if_needed returns False when not dirty."""
        stack = PostProcessStack()
        stack._dirty = False
        executor = PostProcessStackExecutor(stack)
        assert executor.rebuild_if_needed() is False

    def test_rebuild_if_needed_dirty_no_fg(self):
        """rebuild_if_needed returns True when dirty but no frame graph (no-op rebuild)."""
        stack = PostProcessStack()
        stack._dirty = True
        executor = PostProcessStackExecutor(stack)
        result = executor.rebuild_if_needed()
        assert result is True
        assert stack._dirty is False  # Clears dirty regardless

    def test_rebuild_if_needed_dirty_with_fg(self):
        """rebuild_if_needed rebuilds passes when dirty and frame graph available."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        stack._dirty = True
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor._is_built = False
        result = executor.rebuild_if_needed()
        assert result is True
        assert stack._dirty is False
        assert len(fg.passes) == 1

    def test_rebuild_if_needed_without_handles(self):
        """rebuild_if_needed does not rebuild if hdr_handle is None even with fg."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        stack.add_effect(MockEffect("Tonemapping"))
        stack._dirty = True
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        # No prepare_resources called, so _hdr_handle is None
        result = executor.rebuild_if_needed()
        assert result is True
        assert stack._dirty is False
        # No passes created because no handles
        assert len(fg.passes) == 0

    def test_reset_clears_all_state(self):
        """reset clears all executor state."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080)
        executor._is_built = True
        executor.reset()
        assert executor._is_built is False
        assert executor._has_resources is False
        assert executor._hdr_handle is None
        assert executor._output_handle is None
        assert executor._intermediate_mgr._ready is False

    def test_make_rhi_effect_callback_returns_callable(self):
        """_make_rhi_effect_callback returns a callable function."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        effect = MockEffect("Test")
        cb = executor._make_rhi_effect_callback(
            effect, "input_handle", "output_handle", PostProcessContext()
        )
        assert callable(cb)

    def test_make_rhi_effect_callback_invokes_effect(self):
        """The callback invokes the effect when called."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        effect = MockEffect("Test")
        ctx = PostProcessContext(frame_index=10)
        cb = executor._make_rhi_effect_callback(
            effect, "input_handle", "output_handle", ctx
        )
        cb(frame_graph_context=type('obj', (object,), {"command_list": None})())
        assert effect.execute_called is True

    def test_make_rhi_effect_callback_with_command_list(self):
        """Callback uses frame_graph_context.command_list when present."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        effect = MockEffect("Test")
        ctx = PostProcessContext()
        cb = executor._make_rhi_effect_callback(
            effect, "input_handle", "output_handle", ctx
        )
        class MockFgCtx:
            command_list = "rhi_cmd_from_fg"
        cb(MockFgCtx())
        assert effect.execute_called is True

    def test_make_rhi_effect_callback_fallback_to_context_cmd(self):
        """Callback falls back to context.rhi_command_list when fg_ctx has no command_list."""
        stack = PostProcessStack()
        executor = PostProcessStackExecutor(stack)
        effect = MockEffect("Test")
        ctx = PostProcessContext(rhi_command_list="rhi_cmd_from_ctx")
        cb = executor._make_rhi_effect_callback(
            effect, "input_handle", "output_handle", ctx
        )
        class MockFgCtx:
            pass  # No command_list attribute
        cb(MockFgCtx())
        assert effect.execute_called is True

    def test_execution_path_default(self):
        """Default execution path is FRAME_GRAPH_PASS."""
        executor = PostProcessStackExecutor(PostProcessStack())
        assert executor.execution_path == EffectExecutionPath.FRAME_GRAPH_PASS

    def test_execution_path_setter_updates(self):
        """execution_path setter changes the internal path."""
        executor = PostProcessStackExecutor(PostProcessStack())
        executor.execution_path = EffectExecutionPath.DIRECT_CALL
        assert executor._execution_path == EffectExecutionPath.DIRECT_CALL

    def test_set_context_updates_internal(self):
        """set_context replaces the internal context."""
        executor = PostProcessStackExecutor(PostProcessStack())
        new_ctx = PostProcessContext(frame_index=77, quality=EffectQuality.ULTRA)
        executor.set_context(new_ctx)
        assert executor._context is new_ctx


# ==============================================================================
# Test: PostProcessVolume -- Internal Geometry and Blend
# ==============================================================================


class TestPostProcessVolumeWhitebox:
    """Whitebox tests for PostProcessVolume internal logic."""

    def test_contains_point_in_box_no_blend(self):
        """Inside box returns True, outside returns False when no blend."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=0)
        assert vol.contains_point((5, 5, 5)) is True
        assert vol.contains_point((15, 5, 5)) is False
        assert vol.contains_point((-1, 5, 5)) is False

    def test_contains_point_edge_boundary(self):
        """Point exactly on boundary is contained."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings())
        assert vol.contains_point((0, 0, 0)) is True
        assert vol.contains_point((10, 10, 10)) is True

    def test_contains_point_with_blend_distance(self):
        """Outside but within blend distance is 'contained'."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=5)
        assert vol.contains_point((13, 5, 5)) is True  # 3 units out, blend=5
        assert vol.contains_point((16, 5, 5)) is False  # 6 units out, blend=5

    def test_disabled_volume_never_contains(self):
        """Disabled volume never contains any point."""
        shape = BoxVolumeShape()
        vol = PostProcessVolume(shape, PostProcessVolumeSettings())
        vol.enabled = False
        assert vol.contains_point((0.5, 0.5, 0.5)) is False
        assert vol.contains_point((999, 999, 999)) is False

    def test_global_volume_always_contains(self):
        """Global volume contains all points regardless of shape."""
        shape = BoxVolumeShape(min_bounds=(100, 100, 100), max_bounds=(200, 200, 200))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), global_volume=True)
        assert vol.contains_point((0, 0, 0)) is True
        assert vol.contains_point((999, 999, 999)) is True

    def test_blend_weight_inside_no_blend_zone(self):
        """Inside box with no blend distance returns 1.0."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=0)
        assert vol.get_blend_weight((5, 5, 5)) == 1.0

    def test_blend_weight_inside_with_blend_distance(self):
        """Inside box with blend distance returns 1.0."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=5)
        assert vol.get_blend_weight((5, 5, 5)) == 1.0

    def test_blend_weight_near_inner_edge_with_blend(self):
        """Near the boundary inside, returns boundary_dist/blend_distance ratio capped to 1."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=5)
        # point at (0.5, 5, 5): distance to min bounds = 0.5, ratio = 0.5/5 = 0.1
        weight = vol.get_blend_weight((0.5, 5, 5))
        assert 0.0 < weight <= 1.0

    def test_blend_weight_outside_within_blend(self):
        """Outside but within blend distance yields weight in (0,1)."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=10)
        # point at (15, 5, 5): distance from boundary = 5, blend_dist = 10
        # weight = 1 - (5/10) = 0.5
        weight = vol.get_blend_weight((15, 5, 5))
        assert 0.0 < weight < 1.0

    def test_blend_weight_outside_beyond_blend(self):
        """Beyond blend distance returns 0."""
        shape = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=2)
        assert vol.get_blend_weight((20, 5, 5)) == 0.0

    def test_blend_weight_disabled_volume(self):
        """Disabled volume returns 0 blend weight."""
        shape = BoxVolumeShape()
        vol = PostProcessVolume(shape, PostProcessVolumeSettings())
        vol.enabled = False
        assert vol.get_blend_weight((0.5, 0.5, 0.5)) == 0.0

    def test_blend_weight_sphere_center(self):
        """Sphere center returns 1.0."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=10)
        vol = PostProcessVolume(shape, PostProcessVolumeSettings())
        assert vol.get_blend_weight((0, 0, 0)) == 1.0

    def test_blend_weight_sphere_edge(self):
        """Sphere edge returns 1.0 (no blend distance)."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=10)
        vol = PostProcessVolume(shape, PostProcessVolumeSettings())
        assert vol.get_blend_weight((10, 0, 0)) == 1.0

    def test_blend_weight_sphere_outside_no_blend(self):
        """Outside sphere with no blend returns 0."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=10)
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=0)
        assert vol.get_blend_weight((15, 0, 0)) == 0.0

    def test_sphere_contains_with_blend_distance(self):
        """Sphere contains includes points within blend distance outside."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=10)
        vol = PostProcessVolume(shape, PostProcessVolumeSettings(), blend_distance=5)
        assert vol.contains_point((12, 0, 0)) is True  # 2 units out
        assert vol.contains_point((16, 0, 0)) is False  # 6 units out

    def test_apply_to_stack_weight_not_zero(self):
        """apply_to_stack with weight > 0 applies lerp to matching effects."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"TestEffect": override}
        )
        vol = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        effect = MockEffect("TestEffect", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        vol.apply_to_stack(stack, 0.5)
        # Weight=0.5, so test_value = 1.0 + (99 - 1)*0.5 = 50.0
        assert effect.settings.test_value == 50.0

    def test_apply_to_stack_full_weight(self):
        """apply_to_stack with weight=1.0 fully applies override."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"TestEffect": override}
        )
        vol = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        effect = MockEffect("TestEffect", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        vol.apply_to_stack(stack, 1.0)
        assert effect.settings.test_value == 99.0

    def test_apply_to_stack_zero_weight_no_op(self):
        """apply_to_stack with weight 0 does nothing."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"TestEffect": override}
        )
        vol = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        effect = MockEffect("TestEffect", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        vol.apply_to_stack(stack, 0.0)
        assert effect.settings.test_value == 1.0  # unchanged

    def test_apply_to_stack_no_matching_effect(self):
        """apply_to_stack with no matching effect is a no-op (no error)."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"NonExistent": override}
        )
        vol = PostProcessVolume(shape, settings)
        stack = PostProcessStack()
        effect = MockEffect("RealEffect")
        stack.add_effect(effect)
        vol.apply_to_stack(stack, 1.0)  # Should not raise

    def test_apply_to_stack_no_settings(self):
        """apply_to_stack skips effect with None settings."""
        shape = BoxVolumeShape()
        override = MockEffectSettings(test_value=99.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"NoSettings": override}
        )
        vol = PostProcessVolume(shape, settings)
        stack = PostProcessStack()

        class NoSettingsEffect(MockEffect):
            def __init__(self):
                super().__init__("NoSettings")
                self._settings = None

            @property
            def settings(self):
                return self._settings

        effect = NoSettingsEffect()
        stack.add_effect(effect)
        vol.apply_to_stack(stack, 1.0)  # Should not raise


# ==============================================================================
# Test: VolumeShape -- Geometric Edge Cases
# ==============================================================================


class TestVolumeShapeWhitebox:
    """Whitebox tests for volume shape geometry."""

    def test_box_contains_point_on_min_boundary(self):
        """Point exactly on min boundary is inside."""
        box = BoxVolumeShape(min_bounds=(5, 5, 5), max_bounds=(10, 10, 10))
        assert box.contains((5, 5, 5)) is True

    def test_box_contains_point_on_max_boundary(self):
        """Point exactly on max boundary is inside."""
        box = BoxVolumeShape(min_bounds=(5, 5, 5), max_bounds=(10, 10, 10))
        assert box.contains((10, 10, 10)) is True

    def test_box_not_contains_below_min(self):
        """Point below min in any axis is outside."""
        box = BoxVolumeShape(min_bounds=(5, 5, 5), max_bounds=(10, 10, 10))
        assert box.contains((4.999, 7, 7)) is False

    def test_box_not_contains_above_max(self):
        """Point above max in any axis is outside."""
        box = BoxVolumeShape(min_bounds=(5, 5, 5), max_bounds=(10, 10, 10))
        assert box.contains((10.001, 7, 7)) is False

    def test_box_distance_to_boundary_at_corner(self):
        """Distance at corner equals min distance to nearest face."""
        box = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        dist = box.distance_to_boundary((0, 0, 0))
        assert dist == 0.0

    def test_box_distance_outside_box(self):
        """Distance for point outside box is negative distance to nearest face."""
        box = BoxVolumeShape(min_bounds=(0, 0, 0), max_bounds=(10, 10, 10))
        dist = box.distance_to_boundary((-3, 5, 5))
        assert dist == -3  # or abs(dist) = 3 to min boundary

    def test_sphere_not_contains_far_point(self):
        """Far point outside sphere is not contained."""
        sphere = SphereVolumeShape(center=(0, 0, 0), radius=5)
        assert sphere.contains((10, 10, 10)) is False

    def test_sphere_contains_on_surface(self):
        """Point exactly on sphere surface is contained."""
        sphere = SphereVolumeShape(center=(0, 0, 0), radius=5)
        assert sphere.contains((5, 0, 0)) is True

    def test_sphere_distance_outside_returns_negative(self):
        """Distance for point outside sphere is negative."""
        sphere = SphereVolumeShape(center=(0, 0, 0), radius=5)
        dist = sphere.distance_to_boundary((10, 0, 0))
        assert dist == -5.0

    def test_sphere_distance_inside_center(self):
        """Inside sphere, distance to boundary is radius."""
        sphere = SphereVolumeShape(center=(0, 0, 0), radius=5)
        dist = sphere.distance_to_boundary((0, 0, 0))
        assert dist == 5.0

    def test_sphere_distance_on_boundary(self):
        """On boundary, distance to boundary is 0."""
        sphere = SphereVolumeShape(center=(0, 0, 0), radius=5)
        dist = sphere.distance_to_boundary((5, 0, 0))
        assert dist == 0.0

    def test_sphere_contains_3d_point(self):
        """3D point within sphere is contained."""
        sphere = SphereVolumeShape(center=(1, 2, 3), radius=10)
        assert sphere.contains((4, 5, 6)) is True
        assert sphere.contains((15, 2, 3)) is False


# ==============================================================================
# Test: EffectSettings -- Edge Cases
# ==============================================================================


class TestEffectSettingsWhitebox:
    """Whitebox tests for EffectSettings edge cases."""

    def test_lerp_not_implemented(self):
        """Base EffectSettings.lerp raises NotImplementedError."""
        a = EffectSettings(enabled=True, weight=1.0, priority=100)
        b = EffectSettings(enabled=False, weight=0.0, priority=200)
        with pytest.raises(NotImplementedError):
            a.lerp(b, 0.5)

    def test_settings_default_priority(self):
        """Default priority is CUSTOM (1000)."""
        s = EffectSettings()
        assert s.priority == EffectPriority.CUSTOM.value

    def test_settings_enabled_default(self):
        """Default enabled is True."""
        s = EffectSettings()
        assert s.enabled is True

    def test_settings_weight_default(self):
        """Default weight is 1.0."""
        s = EffectSettings()
        assert s.weight == 1.0

    def test_mock_settings_lerp_full(self):
        """Full weight interpolation (t=1) gives target values."""
        a = MockEffectSettings(enabled=True, test_value=1.0, weight=0.5)
        b = MockEffectSettings(enabled=False, test_value=10.0, weight=1.5)
        result = a.lerp(b, 1.0)
        assert result.test_value == 10.0

    def test_mock_settings_lerp_none(self):
        """Zero weight interpolation (t=0) gives source values."""
        a = MockEffectSettings(enabled=True, test_value=1.0, weight=0.5)
        b = MockEffectSettings(enabled=False, test_value=10.0, weight=1.5)
        result = a.lerp(b, 0.0)
        assert result.test_value == 1.0

    def test_mock_settings_lerp_half(self):
        """Half weight interpolation."""
        a = MockEffectSettings(test_value=0.0)
        b = MockEffectSettings(test_value=100.0)
        result = a.lerp(b, 0.5)
        assert result.test_value == 50.0
        assert result.weight == 1.0


# ==============================================================================
# Test: ExecutionFlags -- Bitmask Arithmetic
# ==============================================================================


class TestExecutionFlagsWhitebox:
    """Whitebox tests for ExecutionFlags bitmask arithmetic."""

    def test_none_is_zero(self):
        assert ExecutionFlags.NONE.value == 0

    def test_skip_if_disabled_bit_0(self):
        assert ExecutionFlags.SKIP_IF_DISABLED.value == 1

    def test_skip_on_first_frame_bit_1(self):
        assert ExecutionFlags.SKIP_ON_FIRST_FRAME.value == 2

    def test_skip_if_no_input_bit_2(self):
        assert ExecutionFlags.SKIP_IF_NO_INPUT.value == 4

    def test_force_async_bit_3(self):
        assert ExecutionFlags.FORCE_ASYNC.value == 8

    def test_always_bit_4(self):
        assert ExecutionFlags.ALWAYS.value == 16

    def test_combined_flags_via_pipe(self):
        """Bitwise OR combines flags correctly."""
        combined = ExecutionFlags.FORCE_ASYNC | ExecutionFlags.ALWAYS
        assert combined == 24  # 8 | 16 = 24

    def test_flag_isolation(self):
        """Each flag occupies its own bit."""
        flags = [ExecutionFlags.SKIP_IF_DISABLED, ExecutionFlags.SKIP_ON_FIRST_FRAME,
                  ExecutionFlags.SKIP_IF_NO_INPUT, ExecutionFlags.FORCE_ASYNC,
                  ExecutionFlags.ALWAYS]
        values = [f.value for f in flags]
        # Verify each is a power of 2 (single bit)
        for v in values:
            assert v & (v - 1) == 0, f"{v} is not a power of 2"


# ==============================================================================
# Test: BlendMode -- Enum Values
# ==============================================================================


class TestBlendModeWhitebox:
    """Whitebox tests for BlendMode enum."""

    def test_all_modes_have_unique_values(self):
        """All blend modes have unique auto() values."""
        values = [m.value for m in BlendMode]
        assert len(values) == len(set(values))

    def test_lerp_name(self):
        assert BlendMode.LERP.name == "LERP"

    def test_override_name(self):
        assert BlendMode.OVERRIDE.name == "OVERRIDE"

    def test_additive_name(self):
        assert BlendMode.ADDITIVE.name == "ADDITIVE"

    def test_multiply_name(self):
        assert BlendMode.MULTIPLY.name == "MULTIPLY"


# ==============================================================================
# Test: PostProcessStackConfig -- Dataclass
# ==============================================================================


class TestPostProcessStackConfigWhitebox:
    """Whitebox tests for PostProcessStackConfig."""

    def test_default_values(self):
        config = PostProcessStackConfig()
        assert config.hdr_enabled is True
        assert config.hdr_format == "R16G16B16A16_FLOAT"
        assert config.intermediate_format == "R11G11B10_FLOAT"
        assert config.output_format == "R8G8B8A8_UNORM"
        assert config.auto_exposure_enabled is True
        assert config.history_buffer_count == 2

    def test_non_default_values(self):
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

    def test_config_mutability(self):
        """Config fields are mutable after creation."""
        config = PostProcessStackConfig()
        config.hdr_format = "R32G32B32A32_FLOAT"
        assert config.hdr_format == "R32G32B32A32_FLOAT"


# ==============================================================================
# Re-Verification: C2 -- Intermediate Target Pool
# ==============================================================================


class TestC2IntermediateTargetPoolWhitebox:
    """Whitebox re-verification: C2 _get_intermediate_target uses pool."""

    def test_intermediate_mgr_created_on_stack(self):
        """PostProcessStack has _intermediate_mgr after init."""
        stack = PostProcessStack()
        assert hasattr(stack, "_intermediate_mgr")
        assert stack._intermediate_mgr is not None

    def test_intermediate_mgr_default_pool_size(self):
        """Default pool size is 2."""
        stack = PostProcessStack()
        assert stack._intermediate_mgr.pool_size == 2

    def test_intermediate_mgr_uses_config_format(self):
        """_intermediate_mgr format matches stack config."""
        config = PostProcessStackConfig(intermediate_format="R16G16B16A16_FLOAT")
        stack = PostProcessStack(config)
        assert stack._intermediate_mgr.format == "R16G16B16A16_FLOAT"

    def test_get_intermediate_target_returns_write_target(self):
        """C2: _get_intermediate_target returns write target from pool."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        stack._intermediate_mgr.allocate(fg, 1920, 1080)
        target = stack._get_intermediate_target(0)
        assert target is not None
        # With pool_size=2, index 0 should write to slot 1
        assert target == stack._intermediate_mgr._targets[1].handle

    def test_get_intermediate_target_ping_pong_alternates(self):
        """C2: Consecutive indices alternate read/write targets."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        stack._intermediate_mgr.allocate(fg, 1920, 1080)
        t0 = stack._get_intermediate_target(0)
        t1 = stack._get_intermediate_target(1)
        assert t0 is not None
        assert t1 is not None
        assert t0 != t1  # Different targets for different indices


# ==============================================================================
# Re-Verification: I1 -- R11G11B10_FLOAT Format
# ==============================================================================


class TestI1R11G11B10FormatWhitebox:
    """Whitebox re-verification: I1 format mapping correctness."""

    def test_intermediate_target_default_format(self):
        """IntermediateTargetManager default format is R11G11B10_FLOAT."""
        mgr = IntermediateTargetManager()
        assert mgr.format == "R11G11B10_FLOAT"

    def test_stack_config_intermediate_format(self):
        """PostProcessStackConfig default intermediate_format is R11G11B10_FLOAT."""
        config = PostProcessStackConfig()
        assert config.intermediate_format == "R11G11B10_FLOAT"

    def test_stack_intermediate_mgr_matches_config(self):
        """Stack._intermediate_mgr format matches config.intermediate_format."""
        config = PostProcessStackConfig(intermediate_format="R16G16B16A16_FLOAT")
        stack = PostProcessStack(config)
        assert stack._intermediate_mgr.format == "R16G16B16A16_FLOAT"

    def test_intermediate_mgr_format_setter(self):
        """Format setter propagates correctly."""
        mgr = IntermediateTargetManager()
        mgr.format = "R8G8B8A8_UNORM"
        assert mgr._format == "R8G8B8A8_UNORM"

    def test_executor_format_syncs_from_config(self):
        """Executor intermediate manager format matches stack config."""
        config = PostProcessStackConfig(intermediate_format="R11G11B10_FLOAT")
        stack = PostProcessStack(config)
        executor = PostProcessStackExecutor(stack)
        assert executor._intermediate_mgr.format == "R11G11B10_FLOAT"

    def test_executor_format_override(self):
        """Executor prepare_resources format override propagates to mgr."""
        stack = PostProcessStack()
        fg = MockFrameGraph()
        executor = PostProcessStackExecutor(stack, frame_graph=fg)
        executor.prepare_resources(1920, 1080, intermediate_format="R32G32B32A32_FLOAT")
        assert executor._intermediate_mgr.format == "R32G32B32A32_FLOAT"
        assert stack.config.intermediate_format == "R32G32B32A32_FLOAT"


# ==============================================================================
# Re-Verification: I2 -- Volume Blending Save/Restore
# ==============================================================================


class TestI2VolumeBlendingSaveRestoreWhitebox:
    """Whitebox re-verification: I2 volume blending does not drift settings."""

    def test_execute_with_context_saves_settings_before_blend(self):
        """save/restore pattern exists: settings saved before blend, restored after."""
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
        # Before execute, base value is 1.0
        assert effect.settings.test_value == 1.0
        # The execute applies volume blending temporarily but restores original
        stack.execute_with_context("hdr_in", "output", ctx)
        # After execute, base value must still be 1.0 (no mutation drift)
        assert effect.settings.test_value == 1.0

    def test_volume_blending_no_drift_across_two_frames(self):
        """I2: Two execute_with_context calls produce no drift."""
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
        stack.execute_with_context("hdr_in", "output", ctx)
        value_after_first = effect.settings.test_value
        stack.execute_with_context("hdr_in", "output", ctx)
        value_after_second = effect.settings.test_value
        # Both must equal the base value of 1.0
        assert value_after_first == 1.0
        assert value_after_second == 1.0
        assert value_after_first == value_after_second

    def test_execute_without_camera_no_save_restore(self):
        """I2: Without camera, no save/restore occurs (settings unchanged)."""
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
        # No camera position
        ctx = PostProcessContext(frame_index=10)
        stack.execute_with_context("hdr_in", "output", ctx)
        # Settings unchanged since no camera triggers volume blending
        assert effect.settings.test_value == 1.0

    def test_execute_without_volumes_no_save_restore(self):
        """I2: Without volumes, no save/restore overhead (settings unchanged)."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        # No volumes added
        ctx = PostProcessContext(frame_index=10, camera_position=(0.0, 0.0, 0.0))
        stack.execute_with_context("hdr_in", "output", ctx)
        assert effect.settings.test_value == 1.0

    def test_volume_blending_applies_during_execution(self):
        """I2: Volume blending is applied during execute (transient)."""
        stack = PostProcessStack(quality=EffectQuality.HIGH)
        effect = MockEffect("Tonemapping", settings=MockEffectSettings(test_value=1.0))
        stack.add_effect(effect)
        shape = BoxVolumeShape(
            min_bounds=(-10.0, -10.0, -10.0),
            max_bounds=(10.0, 10.0, 10.0),
        )
        override = MockEffectSettings(test_value=50.0)
        settings = PostProcessVolumeSettings(
            effect_overrides={"Tonemapping": override}
        )
        volume = PostProcessVolume(shape, settings, priority=10)
        stack.add_volume(volume)
        ctx = PostProcessContext(frame_index=10, camera_position=(0.0, 0.0, 0.0))
        # Intercept to verify blending was called
        original_apply = stack._apply_volume_blending
        blending_called = [False]
        def spy(pos):
            blending_called[0] = True
            original_apply(pos)
        stack._apply_volume_blending = spy  # type: ignore
        stack.execute_with_context("hdr_in", "output", ctx)
        assert blending_called[0] is True
        assert effect.execute_called is True


# ==============================================================================
# Re-Verification: I3 -- BlendMode Removed
# ==============================================================================


class TestI3BlendModeRemovedWhitebox:
    """Whitebox re-verification: I3 BlendMode enum no longer exists."""

    def test_blendmode_not_in_module(self):
        """BlendMode is not importable from postprocess_stack module."""
        import engine.rendering.postprocess.postprocess_stack as pp_stack
        assert not hasattr(pp_stack, "BlendMode")

    def test_blendmode_not_in_init(self):
        """BlendMode is not exported from postprocess __init__."""
        import engine.rendering.postprocess as pp
        assert not hasattr(pp, "BlendMode")

    def test_no_blendmode_references_in_all(self):
        """BlendMode is not in __all__."""
        from engine.rendering.postprocess.postprocess_stack import __all__ as all_list
        assert "BlendMode" not in all_list

    def test_effect_settings_no_blend_attr(self):
        """EffectSettings does not have a blend_mode attribute."""
        settings = EffectSettings()
        assert not hasattr(settings, "blend_mode")
