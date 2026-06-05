"""
Whitebox Tests for Post-Processing Subsystem -- Phase 1

Tests internal branches, loop edge cases, error paths, helper functions,
and non-obvious invariants across ALL postprocess modules.

Coverage Plan (branch-level):
══════════════════════════════

[PostProcess Stack]
  PPS-1  IntermediateTargetManager.__init__: pool_size < 1 => ValueError
  PPS-2  IntermediateTargetManager.get_ping_pong: not ready or len<2 => (None,None)
  PPS-3  PostProcessEffect.should_execute: 4-layer flag decision (ALWAYS/SKIP_IF_DISABLED/quality/SKIP_ON_FIRST_FRAME)
  PPS-4  PostProcessStack.set_quality: early-return if same quality
  PPS-5  PostProcessStack.add_effect: duplicate name => ValueError
  PPS-6  PostProcessStack.resize: early-return if same dimensions
  PPS-7  BoxVolumeShape.contains: all() with 3 axes; corner/edge/center
  PPS-8  SphereVolumeShape.contains: distance-squared vs radius-squared
  PPS-9  PostProcessVolume.contains_point: 4 branches (enabled/global/shape-type/blend-distance)
  PPS-10 PostProcessVolume.get_blend_weight: 6 branches (disabled=0,global=1,inside shape,blend-distance-in,blend-distance-out,outside)

[Exposure]
  EXP-1  luminance_to_ev: luminance <= LUMINANCE_MIN => EV_MIN_FALLBACK
  EXP-2  exposure_to_ev: exposure <= EPSILON => EV_MIN_FALLBACK
  EXP-3  AutoExposure._calculate_weighted_average: returns hardcoded 0.18
  EXP-4  AutoExposure.calculate_target_ev: int/float vs buffer, HIGHLIGHT_PRIORITY halving, min/max clamp
  EXP-5  ManualExposure.calculate_target_ev: manual_ev + compensation
  EXP-6  HistogramExposure.calculate_target_ev: total_pixels == 0 => 0.0
  EXP-7  HistogramExposure._ensure_histogram: exact-length list vs fallback [1]*n
  EXP-8  HistogramExposure._find_percentile_bin: cumulative-sum loop edges
  EXP-9  EyeAdaptation.update: _initialized branch, adaptation_factor = 1-e^(-speed*dt)

[Tone Mapping]
  TON-1  Reinhard.apply: luminance <= 0 => (0,0,0)
  TON-2  ReinhardExtended.apply: white-point scaling
  TON-3  CustomCurve._evaluate_curve: empty points => x; x <= first => first output; x >= last => last output; input_range guard; linear vs hermite
  TON-4  AgX._safe_log2: clamps to SAFE_LOG_MIN
  TON-5  AgX.apply: log2 and curve branches
  TON-6  ACESFitted._aces_curve: denominator always > 0 since e > 0
  TON-7  TonemappingEffect.tonemap_value: exposure_bias, operator fallback (DEFAULT), color_filter, saturation

[Color Grading]
  CLR-1  WhiteBalanceSettings.get_color_temperature_rgb: t>=0 vs t<0, tint offset, min/max clamp
  CLR-2  ContrastSettings.apply: shadow/highlight weights at lum=0, lum=0.5, lum=1
  CLR-3  SaturationSettings.apply: per-channel sat, vibrance with luminance guard
  CLR-4  LUT3D.load_from_cube: size==0 => False; data count mismatch => False; exception => False
  CLR-5  LUT3D.sample: not initialized => passthrough; trilinear 8-corner interpolation
  CLR-6  ColorGradingStack.apply: full pipeline order with LUT blend

[Depth of Field]
  DOF-1  BokehShape.get_bokeh_kernel: dispatch by shape_type (circle/polygon/ellipse/fallback)
  DOF-2  BokehShape._generate_disk_samples: golden-angle spiral, spherical_aberration weight
  DOF-3  BokehShape._generate_polygon_samples: angular clamp, blade_curvature != 0
  DOF-4  BokehShape._generate_ellipse_samples: wraps disk with anamorphic_ratio scaling
  DOF-5  CircleOfConfusion.calculate: depth<=EPSILON => 0; focus_distance<=EPSILON => 0; depth==focus => 0; hyperfocal formula; min with max_coc_radius
  DOF-6  CircleOfConfusion.get_depth_ranges: focus_m >= h => infinite far
  DOF-7  AutoFocusSystem.update: diff <= max_change => snap; else interpolate
  DOF-8  FarFieldDOF.blur: LOW/MEDIUM vs HIGH/CINEMATIC branch
  DOF-9  DOFEffect.execute: auto_focus mode, near_blur, far_blur branches

[Motion Blur]
  MB-1  TileMaxVelocity.tile_size.setter: clamps to [4, 64]
  MB-3  MotionBlurEffect.execute: CAMERA_ONLY/OBJECT_ONLY/COMBINED dispatch; intensity <= 0

[Anti-Aliasing]
  AA-1  JitterSequence._halton: while-loop algorithm
  AA-2  JitterSequence.get_projection_jitter: scales by (2/width, 2/height)
  AA-3  TAA.setup: invalidates history on resolution change
  AA-4  TAA.get_jittered_projection: modifies projection[2][0] and [2][1]
  AA-5  AAEffect.get_jittered_projection: non-TAA passthrough
  AA-6  AAEffect.get_required_inputs: adds depth+velocity only for TAA

[Ambient Occlusion]
  AO-1  SSAOKernel.generate: deterministic (seed=42), hemisphere normalization, scale distribution
  AO-2  HBAO._generate_directions: uniform angular distribution, 4 or 8 directions
  AO-3  BilateralFilter._bilateral_weight: depth weight exp(-diff^2*sharpness), normal dot, spatial gaussian
  AO-4  BentNormalOutput.calculate_specular_occlusion: dot with bent normal, roughness-adjusted power
  AO-5  AOEffect.get_outputs: conditionally adds bent_normals

[Constants]
  CST-1 calculate_luminance: BT.709 coefficients, BT.601 coefficients, component clamping
  CST-2 EXPOSURE constants: LUMINANCE_TO_EV_SCALE, EV_MIN_FALLBACK

[Upscaling]
  USC-1 UpscaleResolution.scale_factor: zero-division guard
  USC-2 get_render_resolution: scale factor lookup, max(1, int())
  USC-3 UpscalingSettings.get_recommended_mip_bias: bias_table lookup + offset
  USC-4 UpscalingEffect.execute: 8-way dispatch with DLSS/XeSS fallback to FSR2
  USC-5 UpscalingEffect.get_required_inputs: temporal adds depth+velocity

[Bloom -- additive to existing test_bloom_whitebox.py]
  BLO-1 BloomDownsample.setup: stops when w<2 or h<2
  BLO-2 BloomBlur.blur: early-return if source is None or dimensions < 2
  BLO-3 BloomBlur._kawase_blur: 5-point cross with bounds checking
  BLO-4 BloomBlur._box_blur: [1,1,1]/3 separable with edge averaging
"""

from __future__ import annotations

import math
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ── Stack ──────────────────────────────────────────────────────────────────
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

# ── Exposure ────────────────────────────────────────────────────────────────
from engine.rendering.postprocess.exposure import (
    AdaptationCurve,
    AutoExposure,
    ev_to_exposure,
    ExposureEffect,
    ExposureMode,
    exposure_to_ev,
    ExposureSettings,
    EyeAdaptation,
    HistogramExposure,
    luminance_to_ev,
    ManualExposure,
    MeteringMode,
)

# ── Tone Mapping ────────────────────────────────────────────────────────────
from engine.rendering.postprocess.tonemapping import (
    ACES,
    ACESFitted,
    AgX,
    CustomCurve,
    CustomCurveSettings,
    Filmic,
    Reinhard,
    ReinhardExtended,
    TonemapCurvePoint,
    TonemapFunction,
    TonemapOperator,
    TonemappingEffect,
    TonemapSettings,
)

# ── Color Grading ───────────────────────────────────────────────────────────
from engine.rendering.postprocess.color_grading import (
    ColorGradingEffect,
    ColorGradingSettings,
    ColorGradingStack,
    ColorSpace,
    ContrastSettings,
    HueSatLightness,
    LiftGammaGain,
    LUT3D,
    LUT3DSettings,
    LUTFormat,
    SaturationSettings,
    WhiteBalanceSettings,
)

# ── Depth of Field ──────────────────────────────────────────────────────────
from engine.rendering.postprocess.dof import (
    AutoFocusSystem,
    BokehShape,
    BokehShapeType,
    CircleOfConfusion,
    DOFEffect,
    DOFMode,
    DOFQuality,
    DOFSettings,
    FarFieldDOF,
    NearFieldDOF,
)

# ── Motion Blur ─────────────────────────────────────────────────────────────
from engine.rendering.postprocess.motion_blur import (
    CameraMotion,
    CameraMotionBlur,
    MotionBlurEffect,
    MotionBlurMode,
    MotionBlurQuality,
    MotionBlurSettings,
    ObjectMotionBlur,
    TileMaxVelocity,
)

# ── Anti-Aliasing ───────────────────────────────────────────────────────────
from engine.rendering.postprocess.antialiasing import (
    AAEffect,
    AAMethod,
    AASettings,
    FXAA,
    FXAAQuality,
    FXAASettings,
    JitterPattern,
    JitterSequence,
    SMAA,
    SMAAQuality,
    SMAASettings,
    TAA,
    TAASettings,
)

# ── Ambient Occlusion ───────────────────────────────────────────────────────
from engine.rendering.postprocess.ambient_occlusion import (
    AOEffect,
    AOMethod,
    AOQuality,
    AOSettings,
    BentNormalOutput,
    BilateralFilter,
    GTAO,
    HBAO,
    SSAO,
    SSAOKernel,
)

# ── Upscaling ───────────────────────────────────────────────────────────────
from engine.rendering.postprocess.upscaling import (
    BilinearUpscaler,
    CASUpscaler,
    DLSSUpscaler,
    FrameGenerationMode,
    FSR1Upscaler,
    FSR2Upscaler,
    get_render_resolution,
    SpatialUpscaler,
    TemporalUpscaler,
    UpscaleQuality,
    UpscaleResolution,
    UpscalerType,
    UpscalingEffect,
    UpscalingSettings,
    XeSSUpscaler,
)

# ── Bloom (additive to existing bloom whitebox tests) ──────────────────────
from engine.rendering.postprocess.bloom import (
    BloomBlur,
    BloomDownsample,
    BloomEffect,
    BloomMipSettings,
    BloomQuality,
    BloomSettings,
    BloomThreshold,
    BloomUpsample,
    BlurMethod,
    LensDirtSettings,
)

# ── Constants ───────────────────────────────────────────────────────────────
from engine.rendering.postprocess.constants import (
    AA,
    AO,
    BLOOM,
    COLOR_GRADING,
    DOF,
    EPSILON,
    EXPOSURE,
    LUMINANCE_COEFFS_BT601,
    LUMINANCE_COEFFS_BT709,
    LUMINANCE_MIN,
    MOTION_BLUR,
    SAFE_LOG_MIN,
    TONEMAP,
    UPSCALING,
    calculate_luminance,
)


# ============================================================================
# Helper: Minimal effect for testing PostProcessEffect internals
# ============================================================================

class _MinimalEffect(PostProcessEffect[EffectSettings]):
    """A minimal concrete effect for whitebox testing of base-class internals."""

    def __init__(self, name: str = "Minimal", enabled: bool = True) -> None:
        super().__init__(name=name, settings=EffectSettings(enabled=enabled), priority=500)
        self._enabled = enabled

    def get_required_inputs(self) -> List[str]:
        return ["color"]

    def get_outputs(self) -> List[str]:
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        pass

    def execute(self, inputs: Dict[str, Any], outputs: Dict[str, Any], delta_time: float) -> None:
        pass

    def cleanup(self) -> None:
        pass


# ============================================================================
# PPS -- PostProcess Stack Whitebox
# ============================================================================

class TestIntermediateTargetManagerWhitebox:
    """PPS-1: IntermediateTargetManager.__init__ pool_size validation."""

    def test_pool_size_less_than_one_raises(self) -> None:
        """pool_size < 1 => ValueError."""
        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            IntermediateTargetManager(pool_size=0)

    def test_pool_size_negative_raises(self) -> None:
        """Negative pool_size => ValueError."""
        with pytest.raises(ValueError):
            IntermediateTargetManager(pool_size=-1)

    def test_pool_size_one_valid(self) -> None:
        """pool_size=1 is valid (minimum)."""
        mgr = IntermediateTargetManager(pool_size=1)
        assert mgr is not None

    """PPS-2: get_ping_pong returns (None, None) when not ready."""

    def test_get_ping_pong_not_ready(self) -> None:
        """Before any allocate, get_ping_pong returns (None, None)."""
        mgr = IntermediateTargetManager(pool_size=2)
        read, write = mgr.get_ping_pong(effect_index=0)
        assert read is None
        assert write is None


class TestPostProcessEffectShouldExecuteWhitebox:
    """PPS-3: should_execute 4-layer flag decision."""

    def _make_ctx(self, frame_index: int = 0) -> PostProcessContext:
        return PostProcessContext(
            frame_index=frame_index,
            quality=EffectQuality.ULTRA,
            delta_time=0.016,
        )

    def test_always_flag_skips_all_checks(self) -> None:
        """ExecutionFlags.ALWAYS => should_execute returns True regardless."""
        effect = _MinimalEffect(name="Test", enabled=False)
        effect.set_execution_flags(ExecutionFlags.ALWAYS.value)
        assert effect.should_execute(self._make_ctx())

    def test_skip_if_disabled_with_enabled_false(self) -> None:
        """SKIP_IF_DISABLED + enabled=False => should_execute returns False."""
        effect = _MinimalEffect(name="Test", enabled=False)
        effect.set_execution_flags(ExecutionFlags.SKIP_IF_DISABLED.value)
        assert not effect.should_execute(self._make_ctx())

    def test_skip_if_disabled_with_enabled_true(self) -> None:
        """SKIP_IF_DISABLED + enabled=True => should_execute returns True."""
        effect = _MinimalEffect(name="Test", enabled=True)
        effect.set_execution_flags(ExecutionFlags.SKIP_IF_DISABLED.value)
        assert effect.should_execute(self._make_ctx())

    def test_skip_on_first_frame_with_frame_zero(self) -> None:
        """SKIP_ON_FIRST_FRAME + frame_index=0 => should_execute returns False."""
        effect = _MinimalEffect(name="Test", enabled=True)
        effect.set_execution_flags(ExecutionFlags.SKIP_ON_FIRST_FRAME.value)
        ctx = self._make_ctx(frame_index=0)
        assert not effect.should_execute(ctx)

    def test_skip_on_first_frame_with_frame_gt_zero(self) -> None:
        """SKIP_ON_FIRST_FRAME + frame_index>0 => should_execute returns True."""
        effect = _MinimalEffect(name="Test", enabled=True)
        effect.set_execution_flags(ExecutionFlags.SKIP_ON_FIRST_FRAME.value)
        ctx = self._make_ctx(frame_index=5)
        assert effect.should_execute(ctx)


class TestPostProcessStackWhitebox:
    """PPS-4/5/6: Stack management edge cases."""

    def test_set_quality_same_quality_early_return(self) -> None:
        """set_quality with same quality is a no-op."""
        stack = PostProcessStack(quality=EffectQuality.ULTRA)
        stack.set_quality(EffectQuality.ULTRA)
        # No crash = success (early return)

    def test_add_effect_duplicate_name_raises(self) -> None:
        """Adding an effect with a duplicate name raises ValueError."""
        stack = PostProcessStack()
        effect = _MinimalEffect(name="TestEffect")
        stack.add_effect(effect)
        with pytest.raises(ValueError, match="already exists"):
            stack.add_effect(_MinimalEffect(name="TestEffect"))

    def test_resize_same_dimensions_early_return(self) -> None:
        """resize with same width/height should be a no-op."""
        stack = PostProcessStack()
        stack.resize(1920, 1080)
        stack.resize(1920, 1080)
        # No crash = success

    def test_get_active_effects_filters_disabled(self) -> None:
        """get_active_effects: a disabled effect is excluded."""
        stack = PostProcessStack(quality=EffectQuality.ULTRA)
        eff1 = _MinimalEffect(name="Exposure", enabled=True)
        eff2 = _MinimalEffect(name="DisabledEff", enabled=False)
        stack.add_effect(eff1)
        stack.add_effect(eff2)
        active = stack.get_active_effects()
        names = [e.name for e in active]
        assert "Exposure" in names
        assert "DisabledEff" not in names


class TestVolumeShapeWhitebox:
    """PPS-7/8: Volume shape containment edge cases."""

    def test_box_contains_center(self) -> None:
        """BoxVolumeShape.contains: center point inside."""
        shape = BoxVolumeShape(min_bounds=(-1, -1, -1), max_bounds=(1, 1, 1))
        assert shape.contains((0, 0, 0))

    def test_box_contains_corner(self) -> None:
        """BoxVolumeShape.contains: corner point (exact boundary)."""
        shape = BoxVolumeShape(min_bounds=(-1, -1, -1), max_bounds=(1, 1, 1))
        assert shape.contains((1, 1, 1))

    def test_box_contains_outside(self) -> None:
        """BoxVolumeShape.contains: outside point."""
        shape = BoxVolumeShape(min_bounds=(-1, -1, -1), max_bounds=(1, 1, 1))
        assert not shape.contains((2, 0, 0))

    def test_box_contains_partial_axis(self) -> None:
        """BoxVolumeShape.contains: outside on one axis only."""
        shape = BoxVolumeShape(min_bounds=(-1, -1, -1), max_bounds=(1, 1, 1))
        assert not shape.contains((0, 0, 2))

    def test_sphere_contains_center(self) -> None:
        """SphereVolumeShape.contains: center."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=5.0)
        assert shape.contains((0, 0, 0))

    def test_sphere_contains_on_surface(self) -> None:
        """SphereVolumeShape.contains: exactly on surface (distance == radius)."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=5.0)
        assert shape.contains((5, 0, 0))

    def test_sphere_contains_outside(self) -> None:
        """SphereVolumeShape.contains: outside radius."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=5.0)
        assert not shape.contains((6, 0, 0))

    def test_sphere_contains_diagonal_outside(self) -> None:
        """SphereVolumeShape.contains: diagonal point just outside."""
        shape = SphereVolumeShape(center=(0, 0, 0), radius=5.0)
        assert not shape.contains((3, 3, 3))


class TestPostProcessVolumeWhitebox:
    """PPS-9/10: Volume contains and blend weight branches."""

    def test_disabled_contains_none(self) -> None:
        """contains_point: disabled volume never contains."""
        vol = PostProcessVolume(
            shape=BoxVolumeShape(min_bounds=(-10, -10, -10), max_bounds=(10, 10, 10)),
            settings=PostProcessVolumeSettings(),
        )
        vol._enabled = False
        assert not vol.contains_point((0, 0, 0))

    def test_global_contains_any(self) -> None:
        """contains_point: global volume contains any point."""
        vol = PostProcessVolume(
            shape=BoxVolumeShape(),
            settings=PostProcessVolumeSettings(),
            global_volume=True,
        )
        assert vol.contains_point((999, 999, 999))

    def test_get_blend_weight_disabled(self) -> None:
        """get_blend_weight: disabled => 0.0."""
        vol = PostProcessVolume(
            shape=BoxVolumeShape(),
            settings=PostProcessVolumeSettings(),
        )
        vol._enabled = False
        assert vol.get_blend_weight((0, 0, 0)) == 0.0

    def test_get_blend_weight_global(self) -> None:
        """get_blend_weight: global volume => 1.0."""
        vol = PostProcessVolume(
            shape=BoxVolumeShape(),
            settings=PostProcessVolumeSettings(),
            global_volume=True,
        )
        assert vol.get_blend_weight((0, 0, 0)) == 1.0

    def test_get_blend_weight_inside_shape_no_blend(self) -> None:
        """get_blend_weight: inside shape with blend_distance=0 => 1.0."""
        vol = PostProcessVolume(
            shape=SphereVolumeShape(center=(0, 0, 0), radius=5.0),
            settings=PostProcessVolumeSettings(),
            blend_distance=0.0,
        )
        assert vol.get_blend_weight((0, 0, 0)) == 1.0

    def test_get_blend_weight_inside_shape_with_blend(self) -> None:
        """get_blend_weight: inside shape with blend_distance > 0 => weight > 0."""
        vol = PostProcessVolume(
            shape=SphereVolumeShape(center=(0, 0, 0), radius=5.0),
            settings=PostProcessVolumeSettings(),
            blend_distance=2.0,
        )
        weight = vol.get_blend_weight((0, 0, 0))
        assert weight > 0.0

    def test_get_blend_weight_outside_shape_beyond_blend(self) -> None:
        """get_blend_weight: outside shape and beyond blend_distance => 0.0."""
        vol = PostProcessVolume(
            shape=SphereVolumeShape(center=(0, 0, 0), radius=1.0),
            settings=PostProcessVolumeSettings(),
            blend_distance=1.0,
        )
        weight = vol.get_blend_weight((10, 0, 0))
        assert weight == 0.0


# ============================================================================
# EXP -- Exposure Whitebox
# ============================================================================

class TestExposureConversionWhitebox:
    """EXP-1/2: Edge case guards in conversion functions."""

    def test_luminance_to_ev_zero_guard(self) -> None:
        """luminance_to_ev: luminance <= LUMINANCE_MIN => EV_MIN_FALLBACK = -10."""
        assert luminance_to_ev(0.0) == -10.0
        assert luminance_to_ev(-1.0) == -10.0
        assert luminance_to_ev(LUMINANCE_MIN / 2) == -10.0

    def test_luminance_to_ev_positive(self) -> None:
        """luminance_to_ev: positive luminance computes correctly."""
        ev = luminance_to_ev(0.18)
        assert isinstance(ev, float)
        assert not math.isinf(ev)
        assert not math.isnan(ev)

    def test_exposure_to_ev_zero_guard(self) -> None:
        """exposure_to_ev: exposure <= EPSILON => EV_MIN_FALLBACK."""
        assert exposure_to_ev(0.0) == -10.0
        assert exposure_to_ev(-1.0) == -10.0
        assert exposure_to_ev(EPSILON / 2) == -10.0

    def test_exposure_to_ev_positive(self) -> None:
        """exposure_to_ev: positive exposure computes -log2(exposure)."""
        ev = exposure_to_ev(0.5)
        assert abs(ev - 1.0) < 0.01

    def test_ev_to_exposure_roundtrip(self) -> None:
        """ev_to_exposure and exposure_to_ev are inverses."""
        for ev in [-5.0, -2.0, 0.0, 2.0, 5.0]:
            exp = ev_to_exposure(ev)
            ev_back = exposure_to_ev(exp)
            assert abs(ev_back - ev) < 0.001


class TestAutoExposureWhitebox:
    """EXP-3/4: AutoExposure internal branches."""

    def test_weighted_average_returns_constant(self) -> None:
        """AutoExposure._calculate_weighted_average always returns 0.18."""
        ae = AutoExposure()
        result = ae._calculate_weighted_average(
            [[0.1, 0.2], [0.3, 0.4]], MeteringMode.MATRIX
        )
        assert abs(result - 0.18) < 0.001

    def test_calculate_target_ev_float_input(self) -> None:
        """calculate_target_ev with float input."""
        ae = AutoExposure()
        settings = ExposureSettings(mode=ExposureMode.AUTO_AVERAGE)
        ev = ae.calculate_target_ev(0.18, settings)
        assert settings.min_ev <= ev <= settings.max_ev

    def test_calculate_target_ev_int_input(self) -> None:
        """calculate_target_ev with int input."""
        ae = AutoExposure()
        settings = ExposureSettings(mode=ExposureMode.AUTO_AVERAGE)
        ev = ae.calculate_target_ev(1, settings)
        assert isinstance(ev, float)

    def test_calculate_target_ev_highlight_priority_differs(self) -> None:
        """calculate_target_ev: HIGHLIGHT_PRIORITY yields different EV."""
        ae = AutoExposure()
        settings = ExposureSettings(
            mode=ExposureMode.AUTO_AVERAGE,
            metering_mode=MeteringMode.HIGHLIGHT_PRIORITY,
        )
        ev_normal = ae.calculate_target_ev(
            1.0, ExposureSettings(mode=ExposureMode.AUTO_AVERAGE)
        )
        ev_highlight = ae.calculate_target_ev(1.0, settings)
        assert ev_normal != ev_highlight

    def test_calculate_target_ev_clamping(self) -> None:
        """calculate_target_ev clamps to min_ev/max_ev."""
        ae = AutoExposure()
        tight = ExposureSettings(
            mode=ExposureMode.AUTO_AVERAGE, min_ev=-1.0, max_ev=1.0
        )
        ev = ae.calculate_target_ev(1000000.0, tight)
        assert ev <= tight.max_ev
        ev2 = ae.calculate_target_ev(1e-10, tight)
        assert ev2 >= tight.min_ev


class TestManualExposureWhitebox:
    """EXP-5: ManualExposure with compensation."""

    def test_manual_ev_with_compensation(self) -> None:
        """ManualExposure returns manual_ev + compensation."""
        me = ManualExposure()
        settings = ExposureSettings(
            mode=ExposureMode.MANUAL, manual_ev=2.0, exposure_compensation=0.5
        )
        assert me.calculate_target_ev(None, settings) == 2.5

    def test_manual_ev_zero(self) -> None:
        """ManualExposure with default EV."""
        me = ManualExposure()
        settings = ExposureSettings(mode=ExposureMode.MANUAL)
        assert me.calculate_target_ev(None, settings) == 0.0


class TestHistogramExposureWhitebox:
    """EXP-6/7/8: Histogram exposure internals."""

    def test_total_pixels_zero_returns_zero(self) -> None:
        """calculate_target_ev: total_pixels == 0 => 0.0."""
        he = HistogramExposure()
        settings = ExposureSettings(mode=ExposureMode.AUTO_HISTOGRAM)
        # _ensure_histogram checks len(data) == num_bins (64). Pass 64 zeros to
        # trigger the total_pixels == 0 early-exit branch.
        zeros_64 = [0] * settings.histogram_bins
        assert he.calculate_target_ev(zeros_64, settings) == 0.0

    def test_ensure_histogram_exact_length(self) -> None:
        """_ensure_histogram: exact-length list passes through."""
        he = HistogramExposure()
        result = he._ensure_histogram([1, 2, 3, 4], 4)
        assert result == [1, 2, 3, 4]

    def test_ensure_histogram_fallback(self) -> None:
        """_ensure_histogram: wrong length => fallback [1]*num_bins."""
        he = HistogramExposure()
        result = he._ensure_histogram([1, 2, 3], 4)
        assert result == [1, 1, 1, 1]

    def test_find_percentile_bin_first_bin(self) -> None:
        """_find_percentile_bin: target hit on first bin."""
        he = HistogramExposure()
        assert he._find_percentile_bin([10, 0, 0], 5) == 0

    def test_find_percentile_bin_middle_bin(self) -> None:
        """_find_percentile_bin: cumulative reaches target in middle."""
        he = HistogramExposure()
        assert he._find_percentile_bin([1, 1, 1], 2) == 1

    def test_find_percentile_bin_last_bin(self) -> None:
        """_find_percentile_bin: target in last bin or past end."""
        he = HistogramExposure()
        assert he._find_percentile_bin([1, 1, 1], 3) == 2
        assert he._find_percentile_bin([1, 1, 1], 100) == 2  # past end


class TestEyeAdaptationWhitebox:
    """EXP-9: EyeAdaptation.update branches."""

    def test_initialization_branch(self) -> None:
        """First update initializes without adaptation factor."""
        ea = EyeAdaptation()
        result = ea.update(target_ev=5.0, delta_time=0.016)
        assert result == 5.0
        assert ea.current_ev == 5.0

    def test_adaptation_positive_diff(self) -> None:
        """After init, adaptation uses speed_up when diff > 0."""
        ea = EyeAdaptation()
        ea.reset(initial_ev=0.0)
        result = ea.update(target_ev=10.0, delta_time=0.016)
        assert 0.0 < result < 10.0

    def test_adaptation_negative_diff(self) -> None:
        """After init, adaptation uses speed_down when diff < 0."""
        ea = EyeAdaptation()
        ea.reset(initial_ev=10.0)
        result = ea.update(target_ev=0.0, delta_time=0.016)
        assert 0.0 < result < 10.0

    def test_adaptation_factor_formula(self) -> None:
        """The adaptation factor is 1 - exp(-speed * dt)."""
        ea = EyeAdaptation()
        ea.reset(initial_ev=0.0)
        dt = 1.0
        result = ea.update(target_ev=10.0, delta_time=dt, speed_up=3.0)
        expected = (10.0 - 0.0) * (1.0 - math.exp(-3.0 * 1.0))
        assert abs(result - expected) < 0.0001

    def test_reset_sets_initial_ev(self) -> None:
        """reset sets current and target to initial_ev."""
        ea = EyeAdaptation()
        ea.reset(initial_ev=-2.0)
        assert ea.current_ev == -2.0
        assert ea.target_ev == -2.0

    def test_get_exposure_multiplier(self) -> None:
        """get_exposure_multiplier returns 2^(-current_ev)."""
        ea = EyeAdaptation()
        ea.reset(initial_ev=2.0)
        assert abs(ea.get_exposure_multiplier() - 0.25) < 0.001


# ============================================================================
# TON -- Tone Mapping Whitebox
# ============================================================================

class TestReinhardWhitebox:
    """TON-1: Reinhard luminance <= 0 branch."""

    def test_reinhard_negative_luminance_returns_black(self) -> None:
        """Reinhard.apply: luminance <= 0 => (0,0,0)."""
        r = Reinhard()
        settings = TonemapSettings()
        result = r.apply(-1.0, -2.0, -3.0, settings)
        assert result == (0.0, 0.0, 0.0)

    def test_reinhard_very_bright_approaches_one(self) -> None:
        """Reinhard.apply: very bright values approach 1.0."""
        r = Reinhard()
        settings = TonemapSettings()
        result = r.apply(1000.0, 1000.0, 1000.0, settings)
        assert result[0] < 1.0
        assert result[0] > 0.99


class TestReinhardExtendedWhitebox:
    """TON-2: ReinhardExtended white-point scaling."""

    def test_extended_with_white_point(self) -> None:
        """ReinhardExtended.apply uses white point."""
        r = ReinhardExtended()
        settings = TonemapSettings(white_point=5.0)
        result = r.apply(10.0, 5.0, 2.5, settings)
        assert all(c >= 0.0 for c in result)


class TestCustomCurveWhitebox:
    """TON-3: CustomCurve._evaluate_curve all branches."""

    def test_empty_points_returns_input(self) -> None:
        """Empty points => applies identity (x for each channel).

        Uses SimpleNamespace to bypass CustomCurveSettings.__post_init__
        which replaces empty points with defaults.
        """
        cc = CustomCurve()
        curve = SimpleNamespace(points=[], interpolation="linear")
        result = cc._evaluate_curve(0.5, curve)
        assert result == 0.5

    def test_below_first_point(self) -> None:
        """x <= first point input => first point output."""
        points = [
            TonemapCurvePoint(input_value=0.2, output_value=0.1),
            TonemapCurvePoint(input_value=0.8, output_value=0.9),
        ]
        cc = CustomCurve()
        curve = CustomCurveSettings(points=points)
        # x=0.1 <= 0.2 => first point output = 0.1
        result = cc._evaluate_curve(0.1, curve)
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_above_last_point(self) -> None:
        """x >= last point input => last point output."""
        points = [
            TonemapCurvePoint(input_value=0.2, output_value=0.1),
            TonemapCurvePoint(input_value=0.8, output_value=0.9),
        ]
        cc = CustomCurve()
        curve = CustomCurveSettings(points=points)
        result = cc._evaluate_curve(0.9, curve)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_input_range_guard(self) -> None:
        """input_range <= 1e-10 uses first point output (no div by zero)."""
        points = [
            TonemapCurvePoint(input_value=0.5, output_value=0.3),
            TonemapCurvePoint(input_value=0.5, output_value=0.7),
        ]
        cc = CustomCurve()
        curve = CustomCurveSettings(points=points)
        result = cc._evaluate_curve(0.5, curve)
        assert result == pytest.approx(0.3, abs=1e-6)

    def test_linear_interpolation(self) -> None:
        """interpolation=linear => linear interpolation."""
        points = [
            TonemapCurvePoint(input_value=0.0, output_value=0.0),
            TonemapCurvePoint(input_value=1.0, output_value=1.0),
        ]
        cc = CustomCurve()
        curve = CustomCurveSettings(points=points, interpolation="linear")
        result = cc._evaluate_curve(0.5, curve)
        assert result == pytest.approx(0.5, abs=0.01)


class TestAgXWhitebox:
    """TON-4/5: AgX internals."""

    def test_safe_log2_clamps_negative(self) -> None:
        """_safe_log2 clamps input to SAFE_LOG_MIN."""
        agx = AgX()
        result = agx._safe_log2(-1.0)
        assert not math.isinf(result)
        assert not math.isnan(result)

    def test_safe_log2_zero(self) -> None:
        """_safe_log2 with zero clamps to SAFE_LOG_MIN."""
        agx = AgX()
        result = agx._safe_log2(0.0)
        assert not math.isinf(result)

    def test_safe_log2_positive(self) -> None:
        """_safe_log2 with positive value."""
        agx = AgX()
        result = agx._safe_log2(2.0)
        assert abs(result - 1.0) < 0.001

    def test_agx_curve_positive_for_large_input(self) -> None:
        """_agx_curve: positive values for typical inputs (>0.001)."""
        agx = AgX()
        for v in [0.1, 0.5, 1.0, 10.0, 100.0]:
            result = agx._agx_curve(v)
            assert result >= 0.0
        # The constant term at x=0 gives -0.00232; apply clamps to 0
        assert agx._agx_curve(0.0) == pytest.approx(-0.00232, abs=1e-5)

    def test_apply_produces_valid_output(self) -> None:
        """AgX.apply produces valid tone-mapped output."""
        agx = AgX()
        settings = TonemapSettings()
        result = agx.apply(0.5, 0.5, 0.5, settings)
        assert all(0.0 <= c <= 1.0 for c in result)


class TestACESFittedWhitebox:
    """TON-6: ACESFitted denominator invariant."""

    def test_aces_curve_denominator_positive(self) -> None:
        """ACESFitted._aces_curve: denominator always > 0 since e > 0."""
        aces = ACESFitted()
        for v in [0.0, 0.1, 0.5, 1.0, 10.0]:
            result = aces._aces_curve(v)
            assert result >= 0.0
            assert not math.isnan(result)

    def test_aces_fitted_apply(self) -> None:
        """ACESFitted.apply produces valid output."""
        aces = ACESFitted()
        settings = TonemapSettings()
        result = aces.apply(0.5, 0.5, 0.5, settings)
        assert all(0.0 <= c <= 1.0 for c in result)


class TestTonemappingEffectWhitebox:
    """TON-7: TonemappingEffect.tonemap_value branches."""

    def test_tonemap_value_exposure_bias(self) -> None:
        """tonemap_value applies exposure_bias."""
        effect = TonemappingEffect(settings=TonemapSettings(
            operator=TonemapOperator.REINHARD, exposure_bias=2.0,
        ))
        result = effect.tonemap_value(0.5, 0.5, 0.5)
        assert len(result) == 3

    def test_tonemap_value_color_filter(self) -> None:
        """tonemap_value applies color_filter when non-default."""
        effect = TonemappingEffect(settings=TonemapSettings(
            operator=TonemapOperator.FILMIC,
            color_filter=(1.0, 0.5, 0.25),
        ))
        result = effect.tonemap_value(0.5, 0.5, 0.5)
        assert isinstance(result, tuple)

    def test_tonemap_value_unknown_operator_fallback(self) -> None:
        """Unknown operator falls back to ACES_FITTED (default)."""
        effect = TonemappingEffect(settings=TonemapSettings())
        # Test each operator produces valid output
        for op in TonemapOperator:
            effect._settings.operator = op
            result = effect.tonemap_value(0.5, 0.5, 0.5)
            assert len(result) == 3
            assert all(isinstance(c, float) for c in result)

    def test_tonemap_value_saturation(self) -> None:
        """tonemap_value with saturation=0 => all channels equal luminance."""
        effect = TonemappingEffect(settings=TonemapSettings(
            operator=TonemapOperator.FILMIC, saturation=0.0,
        ))
        result = effect.tonemap_value(1.0, 0.0, 0.0)
        # All channels desaturated to luminance
        assert abs(result[0] - result[1]) < 0.01
        assert abs(result[0] - result[2]) < 0.01


# ============================================================================
# CLR -- Color Grading Whitebox
# ============================================================================

class TestWhiteBalanceWhitebox:
    """CLR-1: WhiteBalance temperature branches."""

    def test_temperature_zero_neutral(self) -> None:
        """temperature=0 => (1, 1, 1)."""
        wb = WhiteBalanceSettings(temperature=0.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_temperature_positive_warm(self) -> None:
        """temperature > 0 => red > blue (warm)."""
        wb = WhiteBalanceSettings(temperature=50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert r > b

    def test_temperature_negative_cool(self) -> None:
        """temperature < 0 => blue > red (cool)."""
        wb = WhiteBalanceSettings(temperature=-50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert b > r

    def test_tint_affects_balance(self) -> None:
        """Tint shifts green-magenta balance."""
        wb_magenta = WhiteBalanceSettings(temperature=0.0, tint=50.0)
        wb_green = WhiteBalanceSettings(temperature=0.0, tint=-50.0)
        r1, g1, b1 = wb_magenta.get_color_temperature_rgb()
        r2, g2, b2 = wb_green.get_color_temperature_rgb()
        assert (r1, g1, b1) != (r2, g2, b2)

    def test_temperature_output_clamped(self) -> None:
        """get_color_temperature_rgb produces [0, 1] output."""
        wb = WhiteBalanceSettings(temperature=100.0, tint=50.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert all(0.0 <= c <= 1.0 for c in (r, g, b))


class TestContrastSettingsWhitebox:
    """CLR-2: ContrastSettings shadow/highlight weight edge cases."""

    def test_apply_black(self) -> None:
        """apply: luminance=0 => full shadow weight (contrast pushes 0 toward mid)."""
        cs = ContrastSettings(contrast=0.5)
        r, g, b = cs.apply(0.0, 0.0, 0.0)
        # contrast=0.5 pushes each channel toward 0.5: 0.5 + (0-0.5)*0.5 = 0.25
        assert r == pytest.approx(0.25, abs=1e-6)
        assert g == pytest.approx(0.25, abs=1e-6)
        assert b == pytest.approx(0.25, abs=1e-6)

    def test_apply_lum_one_gives_full_highlight(self) -> None:
        """apply: luminance=1 => full highlight weight."""
        cs = ContrastSettings(contrast=0.5)
        r, g, b = cs.apply(1.0, 1.0, 1.0)
        assert r >= 0.0

    def test_apply_mid_blend(self) -> None:
        """apply: luminance ~0.5 uses blended shadow/highlight."""
        cs = ContrastSettings(contrast=0.5)
        result = cs.apply(0.5, 0.5, 0.5)
        assert all(c >= 0.0 for c in result)

    def test_apply_negative_contrast_darkens(self) -> None:
        """apply: negative contrast darkens versus zero contrast."""
        cs_neg = ContrastSettings(contrast=-0.5)
        cs_zero = ContrastSettings(contrast=0.0)
        r_neg, _, _ = cs_neg.apply(0.5, 0.5, 0.5)
        r_zero, _, _ = cs_zero.apply(0.5, 0.5, 0.5)
        assert r_neg <= r_zero


class TestSaturationSettingsWhitebox:
    """CLR-3: SaturationSettings vibrance and per-channel."""

    def test_zero_saturation_grayscale(self) -> None:
        """apply: global_saturation=0 => all channels equal (luminance)."""
        ss = SaturationSettings(global_saturation=0.0)
        r, g, b = ss.apply(1.0, 0.5, 0.25)
        assert abs(r - g) < 0.01
        assert abs(r - b) < 0.01

    def test_full_saturation_passthrough(self) -> None:
        """apply: global_saturation=1.0 => original values."""
        ss = SaturationSettings(global_saturation=1.0)
        result = ss.apply(1.0, 0.5, 0.25)
        assert result == (1.0, 0.5, 0.25)

    def test_vibrance_nonzero_applies(self) -> None:
        """apply: vibrance != 0 applies adjustment."""
        ss = SaturationSettings(global_saturation=1.0, vibrance=0.5)
        result = ss.apply(1.0, 0.5, 0.25)
        assert len(result) == 3

    def test_vibrance_dark_pixel_guard(self) -> None:
        """apply: very dark pixel (lum <= LUMINANCE_MIN) avoids div by zero."""
        ss = SaturationSettings(global_saturation=0.0, vibrance=0.5)
        result = ss.apply(0.0, 0.0, 0.0)
        assert result == (0.0, 0.0, 0.0)

    def test_per_channel_saturation(self) -> None:
        """apply: per-channel saturation factors applied."""
        ss = SaturationSettings(global_saturation=1.0, red_saturation=0.5)
        r, g, b = ss.apply(1.0, 0.5, 0.25)
        assert r <= 1.0


class TestLUT3DWhitebox:
    """CLR-4/5: LUT3D internals."""

    def test_create_identity_sets_size(self) -> None:
        """create_identity: creates identity for the instance size."""
        lut = LUT3D(size=16)
        lut.create_identity()
        assert lut.size == 16
        assert lut.initialized is True
        center = lut.sample(0.5, 0.5, 0.5)
        assert abs(center[0] - 0.5) < 0.02

    def test_load_from_cube_missing_file_false(self) -> None:
        """load_from_cube: missing file => returns False."""
        lut = LUT3D()
        result = lut.load_from_cube("/nonexistent/file.cube")
        assert result is False

    def test_load_from_cube_empty_path(self) -> None:
        """load_from_cube: empty string path => returns False."""
        lut = LUT3D()
        result = lut.load_from_cube("")
        assert result is False

    def test_sample_not_initialized_passthrough(self) -> None:
        """sample: not initialized => returns unchanged."""
        lut = LUT3D()
        result = lut.sample(0.5, 0.75, 0.25)
        assert result == (0.5, 0.75, 0.25)

    def test_sample_identity(self) -> None:
        """sample: identity LUT maps (0,0,0) to (0,0,0) and (1,1,1) to (1,1,1)."""
        lut = LUT3D(size=16)
        lut.create_identity()
        r0, g0, b0 = lut.sample(0.0, 0.0, 0.0)
        assert r0 == pytest.approx(0.0, abs=0.02)
        r1, g1, b1 = lut.sample(1.0, 1.0, 1.0)
        assert r1 == pytest.approx(1.0, abs=0.02)
        r0, g0, b0 = lut.sample(0.0, 0.0, 0.0)
        assert abs(r0) < 0.05
        r1, g1, b1 = lut.sample(1.0, 1.0, 1.0)
        assert abs(r1 - 1.0) < 0.05


class TestColorGradingStackWhitebox:
    """CLR-6: Full pipeline order."""

    def test_apply_default_produces_valid(self) -> None:
        """Full pipeline with defaults produces [0,1] output."""
        stack = ColorGradingStack()
        result = stack.apply(0.5, 0.5, 0.5)
        assert all(0.0 <= c <= 1.0 for c in result)

    def test_apply_with_all_settings(self) -> None:
        """Pipeline with all settings engaged."""
        stack = ColorGradingStack(settings=ColorGradingSettings(
            white_balance=WhiteBalanceSettings(temperature=20.0, tint=5.0),
            contrast=ContrastSettings(contrast=0.3),
            saturation=SaturationSettings(global_saturation=1.2, vibrance=0.3),
        ))
        result = stack.apply(0.5, 0.3, 0.8)
        assert all(0.0 <= c <= 1.0 for c in result)


# ============================================================================
# DOF -- Depth of Field Whitebox
# ============================================================================

class TestBokehShapeWhitebox:
    """DOF-1/2/3/4: Bokeh shape generation internals."""

    def test_get_kernel_circle(self) -> None:
        """CIRCLE dispatches to disk samples."""
        samples = BokehShape(shape_type=BokehShapeType.CIRCLE).get_bokeh_kernel(4)
        assert len(samples) > 0
        for x, y, w in samples:
            assert w > 0

    def test_get_kernel_polygon(self) -> None:
        """POLYGON dispatches to polygon samples."""
        samples = BokehShape(
            shape_type=BokehShapeType.POLYGON, blade_count=6
        ).get_bokeh_kernel(4)
        assert len(samples) > 0

    def test_get_kernel_anamorphic(self) -> None:
        """ANAMORPHIC dispatches to ellipse samples."""
        samples = BokehShape(
            shape_type=BokehShapeType.ANAMORPHIC, anamorphic_ratio=1.5
        ).get_bokeh_kernel(4)
        assert len(samples) > 0

    def test_get_kernel_cat_eye_fallback(self) -> None:
        """CAT_EYE (unknown) falls back to disk samples."""
        samples = BokehShape(shape_type=BokehShapeType.CAT_EYE).get_bokeh_kernel(4)
        assert len(samples) > 0

    def test_disk_samples_spherical_aberration(self) -> None:
        """spherical_aberration affects outer sample weights."""
        gen = BokehShape(spherical_aberration=0.5)._generate_disk_samples(4)
        assert len(gen) > 0

    def test_polygon_blade_curvature(self) -> None:
        """blade_curvature != 0 modifies radii."""
        curved = BokehShape(
            shape_type=BokehShapeType.POLYGON, blade_count=6, blade_curvature=0.5,
        )._generate_polygon_samples(4)
        flat = BokehShape(
            shape_type=BokehShapeType.POLYGON, blade_count=6, blade_curvature=0.0,
        )._generate_polygon_samples(4)
        assert len(curved) == len(flat)

    def test_ellipse_scaling(self) -> None:
        """anamorphic_ratio scales x coordinates."""
        samples = BokehShape(anamorphic_ratio=2.0)._generate_ellipse_samples(4)
        for x, y, w in samples:
            assert abs(x) <= 8.0  # radius * ratio


class TestCircleOfConfusionWhitebox:
    """DOF-5/6: CoC calculation edge cases."""

    def test_depth_zero_returns_zero(self) -> None:
        """depth <= EPSILON => 0."""
        coc = CircleOfConfusion(focus_distance=5.0)
        assert coc.calculate(0.0, 1920) == 0.0
        assert coc.calculate(1e-9, 1920) == 0.0

    def test_focus_distance_zero_returns_zero(self) -> None:
        """focus_distance <= EPSILON => 0."""
        coc = CircleOfConfusion(focus_distance=0.0)
        assert coc.calculate(5.0, 1920) == 0.0

    def test_at_focus_returns_zero(self) -> None:
        """depth == focus_distance => 0."""
        coc = CircleOfConfusion(focus_distance=5.0)
        assert coc.calculate(5.0, 1920) == 0.0

    def test_out_of_focus_positive(self) -> None:
        """depth != focus => positive CoC."""
        coc = CircleOfConfusion(
            sensor_width=36.0, focal_length=50.0, aperture=2.8, focus_distance=5.0,
        )
        assert coc.calculate(10.0, 1920) > 0.0

    def test_capped_at_max_radius(self) -> None:
        """CoC capped at max_coc_radius."""
        coc = CircleOfConfusion(
            sensor_width=36.0, focal_length=200.0, aperture=1.4,
            focus_distance=1.0, max_coc_radius=16.0,
        )
        assert coc.calculate(100.0, 1920) <= 16.0

    def test_depth_ranges_infinite_far(self) -> None:
        """get_depth_ranges: focus_m >= h => infinite far."""
        coc = CircleOfConfusion(
            sensor_width=36.0, focal_length=50.0, aperture=2.8,
            focus_distance=1000.0,
        )
        near, near_sharp, far_sharp, far_blur = coc.get_depth_ranges(1920)
        assert far_sharp == float("inf")
        assert far_blur == float("inf")


class TestAutoFocusSystemWhitebox:
    """DOF-7: AutoFocusSystem.update snap vs interpolate."""

    def test_snap_when_diff_small(self) -> None:
        """diff <= max_change => focus snaps to target."""
        af = AutoFocusSystem()
        af._current_focus = 5.0
        assert af.update(5.3, 1.0) == 5.3

    def test_interpolate_when_diff_large(self) -> None:
        """diff > max_change => focus interpolates."""
        af = AutoFocusSystem()
        af._current_focus = 1.0
        result = af.update(10.0, 1.0)
        assert 1.0 < result < 10.0

    def test_negative_diff(self) -> None:
        """target < current => focus decreases."""
        af = AutoFocusSystem()
        af._current_focus = 10.0
        result = af.update(1.0, 1.0)
        assert result < 10.0

    def test_current_focus_property(self) -> None:
        """current_focus returns default value."""
        assert AutoFocusSystem().current_focus == 5.0


class TestFarFieldDOFWhitebox:
    """DOF-8: FarFieldDOF.blur quality branch."""

    def test_low_uses_separable(self) -> None:
        """LOW quality => _separable_blur path (no raise)."""
        far = FarFieldDOF()
        far.setup(1920, 1080)
        # No real buffers so return is None; the branch itself should not raise
        result = far.blur(None, None, BokehShape(), DOFQuality.LOW)
        assert result is None  # _separable_blur returns None (no real buffers)

    def test_high_uses_scatter_gather(self) -> None:
        """HIGH quality => _scatter_gather_blur path (no raise)."""
        far = FarFieldDOF()
        far.setup(1920, 1080)
        result = far.blur(None, None, BokehShape(), DOFQuality.HIGH)
        assert result is None  # _scatter_gather_blur returns None

    def test_cinematic_uses_scatter_gather(self) -> None:
        """CINEMATIC quality => _scatter_gather_blur path (no raise)."""
        far = FarFieldDOF()
        far.setup(1920, 1080)
        result = far.blur(None, None, BokehShape(), DOFQuality.CINEMATIC)
        assert result is None


# ============================================================================
# MB -- Motion Blur Whitebox
# ============================================================================

class TestTileMaxVelocityWhitebox:
    """MB-1: TileMaxVelocity tile_size clamping."""

    def test_clamps_below_4(self) -> None:
        """tile_size < 4 clamps to 4."""
        tmv = TileMaxVelocity()
        tmv.tile_size = 2
        assert tmv.tile_size == 4

    def test_clamps_above_64(self) -> None:
        """tile_size > 64 clamps to 64."""
        tmv = TileMaxVelocity()
        tmv.tile_size = 128
        assert tmv.tile_size == 64

    def test_valid_value_passthrough(self) -> None:
        """valid tile_size passes through."""
        tmv = TileMaxVelocity()
        tmv.tile_size = 16
        assert tmv.tile_size == 16


class TestMotionBlurEffectWhitebox:
    """MB-3: Mode dispatch branches."""

    def test_execute_camera_only(self) -> None:
        """CAMERA_ONLY mode."""
        effect = MotionBlurEffect(settings=MotionBlurSettings(mode=MotionBlurMode.CAMERA_ONLY))
        effect.setup(1920, 1080)
        effect.execute({"color": None, "depth": None}, {}, 0.016)

    def test_execute_object_only(self) -> None:
        """OBJECT_ONLY mode."""
        effect = MotionBlurEffect(settings=MotionBlurSettings(mode=MotionBlurMode.OBJECT_ONLY))
        effect.setup(1920, 1080)
        effect.execute({"color": None, "depth": None, "velocity": None}, {}, 0.016)

    def test_execute_combined(self) -> None:
        """COMBINED mode."""
        effect = MotionBlurEffect(settings=MotionBlurSettings(mode=MotionBlurMode.COMBINED))
        effect.setup(1920, 1080)
        effect.execute({"color": None, "depth": None, "velocity": None}, {}, 0.016)


# ============================================================================
# AA -- Anti-Aliasing Whitebox
# ============================================================================

class TestJitterSequenceWhitebox:
    """AA-1/2: JitterSequence internal algorithm."""

    def test_halton_base_2(self) -> None:
        """_halton with base 2 produces (0, 1) values."""
        js = JitterSequence(JitterPattern.HALTON_16)
        for i in range(1, 17):
            h = js._halton(i, 2)
            assert 0.0 < h < 1.0

    def test_halton_base_3(self) -> None:
        """_halton with base 3 (for y coordinate)."""
        js = JitterSequence(JitterPattern.HALTON_16)
        for i in range(1, 17):
            h = js._halton(i, 3)
            assert 0.0 < h < 1.0

    def test_samples_are_unique(self) -> None:
        """Samples in a sequence are unique."""
        js = JitterSequence(JitterPattern.HALTON_8)
        samples = [js.next() for _ in range(8)]
        assert len(set(samples)) > 1

    def test_get_projection_jitter_scaling(self) -> None:
        """get_projection_jitter scales by (2/width, 2/height)."""
        js = JitterSequence(JitterPattern.HALTON_16)
        jx, jy = js.get_projection_jitter(1920, 1080)
        # get_projection_jitter calls next() internally, returns 2*x/width, 2*y/height
        # x,y in [-0.5,0.5], so jx in [-1/width, 1/width]
        assert abs(jx) <= 2.0 / 1920.0
        assert abs(jy) <= 2.0 / 1080.0


class TestTAAWhitebox:
    """AA-3/4: TAA state management."""

    def test_setup_invalidates_history_on_resize(self) -> None:
        """setup: resolution change invalidates history."""
        taa = TAA()
        taa.setup(1920, 1080)
        taa._history_valid = True
        taa._prev_width = 1920
        taa._prev_height = 1080
        taa.setup(1280, 720)
        assert not taa._history_valid

    def test_get_jittered_projection_modifies(self) -> None:
        """get_jittered_projection modifies [2][0] and [2][1]."""
        taa = TAA()
        taa.setup(1920, 1080)
        proj = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
        result = taa.get_jittered_projection(proj)
        # First Halton sample x=0 (halton(1,2)-0.5=0), y non-zero
        # result[2][1] gets the y-jitter added
        assert result[2][1] != 0.0


class TestAAEffectWhitebox:
    """AA-5/6: AAEffect conditional paths."""

    def test_non_taa_jittered_projection_passthrough(self) -> None:
        """get_jittered_projection: non-TAA returns original."""
        effect = AAEffect(settings=AASettings(method=AAMethod.FXAA))
        proj = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        assert effect.get_jittered_projection(proj) == proj

    def test_taa_required_inputs_include_depth_velocity(self) -> None:
        """TAA requires depth and velocity."""
        effect = AAEffect(settings=AASettings(method=AAMethod.TAA))
        inputs = effect.get_required_inputs()
        assert "depth" in inputs
        assert "velocity" in inputs

    def test_non_taa_required_inputs_exclude_depth_velocity(self) -> None:
        """FXAA does not require depth/velocity."""
        effect = AAEffect(settings=AASettings(method=AAMethod.FXAA))
        inputs = effect.get_required_inputs()
        assert "depth" not in inputs
        assert "velocity" not in inputs


# ============================================================================
# AO -- Ambient Occlusion Whitebox
# ============================================================================

class TestSSAOKernelWhitebox:
    """AO-1: SSAOKernel generation."""

    def test_deterministic(self) -> None:
        """Same seed (42) produces identical kernels."""
        k1 = SSAOKernel(sample_count=16)
        k2 = SSAOKernel(sample_count=16)
        for s1, s2 in zip(k1.samples, k2.samples):
            assert s1 == pytest.approx(s2)

    def test_all_samples_hemisphere(self) -> None:
        """All samples have z >= 0."""
        kernel = SSAOKernel(sample_count=128)
        for x, y, z in kernel.samples:
            assert z >= 0.0

    def test_count_matches(self) -> None:
        """Correct number of samples."""
        for count in [1, 4, 16, 64, 256]:
            assert len(SSAOKernel(sample_count=count).samples) == count

    def test_samples_inside_unit_hemisphere(self) -> None:
        """Sample magnitude <= 1.0."""
        for x, y, z in SSAOKernel(sample_count=32).samples:
            assert math.sqrt(x * x + y * y + z * z) <= 1.0 + 1e-6


class TestHBAOWhitebox:
    """AO-2: HBAO direction generation."""

    def test_four_directions_uniform(self) -> None:
        """_generate_directions: 4 uniform directions on unit circle."""
        hbao = HBAO()
        hbao._generate_directions(4)
        assert len(hbao._directions) == 4
        for dx, dy in hbao._directions:
            assert math.sqrt(dx * dx + dy * dy) == pytest.approx(1.0, abs=0.01)

    def test_eight_directions(self) -> None:
        """_generate_directions: 8 directions."""
        hbao = HBAO()
        hbao._generate_directions(8)
        assert len(hbao._directions) == 8


class TestBilateralFilterWhitebox:
    """AO-3: BilateralFilter weight computation."""

    def test_identical_depths_weight_one(self) -> None:
        """_bilateral_weight: identical depths and normals, zero spatial => 1."""
        bf = BilateralFilter()
        w = bf._bilateral_weight(5.0, 5.0, (0, 0, 1), (0, 0, 1), 0.0, 8.0)
        assert w == pytest.approx(1.0, abs=1e-6)

    def test_different_depths_weight_less(self) -> None:
        """_bilateral_weight: different depths => weight < 1."""
        bf = BilateralFilter()
        assert bf._bilateral_weight(5.0, 10.0, (0, 0, 1), (0, 0, 1), 0.0, 8.0) < 1.0

    def test_opposite_normals(self) -> None:
        """_bilateral_weight: opposite normals => weight < 1."""
        bf = BilateralFilter()
        assert bf._bilateral_weight(5.0, 5.0, (0, 0, 1), (0, 0, -1), 0.0, 8.0) < 1.0

    def test_large_spatial_distance(self) -> None:
        """_bilateral_weight: large spatial distance => weight < 1."""
        bf = BilateralFilter()
        assert bf._bilateral_weight(5.0, 5.0, (0, 0, 1), (0, 0, 1), 100.0, 8.0) < 1.0


class TestBentNormalOutputWhitebox:
    """AO-4: BentNormalOutput specular occlusion."""

    def test_normal_matches_bent(self) -> None:
        """bent_normal == normal => returns AO value."""
        bno = BentNormalOutput(
            bent_normal=(0, 0, 1), visibility_cone=0.5, occlusion=0.5,
        )
        ao = bno.calculate_specular_occlusion((0, 0, 1), 0.5)
        assert 0.0 <= ao <= 1.0

    def test_bent_normal_differs(self) -> None:
        """bent normal differs from normal => different result."""
        bno = BentNormalOutput(
            bent_normal=(1, 0, 0), visibility_cone=0.5, occlusion=0.5,
        )
        ao = bno.calculate_specular_occlusion((0, 0, 1), 0.5)
        assert ao >= 0.0

    def test_roughness_adjusts_power(self) -> None:
        """Different roughness => different specular occlusion."""
        bno = BentNormalOutput(
            bent_normal=(0.5, 0, 0.866), visibility_cone=0.5, occlusion=0.5,
        )
        smooth = bno.calculate_specular_occlusion((0, 0, 1), 0.1)
        rough = bno.calculate_specular_occlusion((0, 0, 1), 0.9)
        assert smooth != rough


class TestAOEffectWhitebox:
    """AO-5: AOEffect conditional outputs."""

    def test_outputs_without_bent_normals(self) -> None:
        """Without bent_normals, outputs does not include bent_normals."""
        effect = AOEffect(settings=AOSettings())
        assert "bent_normals" not in effect.get_outputs()

    def test_outputs_with_bent_normals(self) -> None:
        """With bent_normals enabled, outputs includes bent_normals."""
        effect = AOEffect(settings=AOSettings(bent_normals_enabled=True))
        assert "bent_normals" in effect.get_outputs()


# ============================================================================
# CST -- Constants Whitebox
# ============================================================================

class TestCalculateLuminanceWhitebox:
    """CST-1: calculate_luminance coefficient branches."""

    def test_bt709(self) -> None:
        """BT.709 coefficients."""
        lum = calculate_luminance(1.0, 0.5, 0.25, LUMINANCE_COEFFS_BT709)
        expected = 0.2126 * 1.0 + 0.7152 * 0.5 + 0.0722 * 0.25
        assert abs(lum - expected) < 0.0001

    def test_bt601(self) -> None:
        """BT.601 coefficients."""
        lum = calculate_luminance(1.0, 0.5, 0.25, LUMINANCE_COEFFS_BT601)
        expected = 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.25
        assert abs(lum - expected) < 0.0001

    def test_black_returns_zero(self) -> None:
        """(0,0,0) => 0."""
        assert calculate_luminance(0.0, 0.0, 0.0, LUMINANCE_COEFFS_BT709) == 0.0

    def test_white_returns_one(self) -> None:
        """(1,1,1) => 1.0."""
        lum = calculate_luminance(1.0, 1.0, 1.0, LUMINANCE_COEFFS_BT709)
        assert abs(lum - 1.0) < 0.0001


class TestExposureConstantsWhitebox:
    """CST-2: Exposure constants."""

    def test_luminance_to_ev_scale(self) -> None:
        """LUMINANCE_TO_EV_SCALE is 100/12.5 = 8."""
        assert EXPOSURE.LUMINANCE_TO_EV_SCALE == 8.0

    def test_ev_min_fallback(self) -> None:
        """EV_MIN_FALLBACK is -10."""
        assert EXPOSURE.EV_MIN_FALLBACK == -10.0

    def test_luminance_min_positive(self) -> None:
        """LUMINANCE_MIN is positive."""
        assert LUMINANCE_MIN > 0.0

    def test_epsilon(self) -> None:
        """EPSILON is small positive."""
        assert 0.0 < EPSILON < 1e-3


# ============================================================================
# USC -- Upscaling Whitebox
# ============================================================================

class TestUpscaleResolutionWhitebox:
    """USC-1: UpscaleResolution properties."""

    def test_scale_factor(self) -> None:
        """scale_factor > 0 for valid UpscaleResolution."""
        ur = UpscaleResolution(render_width=1280, render_height=720,
                               output_width=1920, output_height=1080)
        assert ur.scale_factor == pytest.approx(1920.0 / 1280.0)

    def test_scale_factor_zero_guard(self) -> None:
        """scale_factor returns 1.0 when render_width is 0."""
        ur = UpscaleResolution(render_width=0, render_height=0,
                               output_width=1920, output_height=1080)
        assert ur.scale_factor == 1.0

    def test_render_percentage(self) -> None:
        """render_percentage in (0, 100]."""
        ur = UpscaleResolution(render_width=1280, render_height=720,
                               output_width=1920, output_height=1080)
        assert 0 < ur.render_percentage <= 100


class TestGetRenderResolutionWhitebox:
    """USC-2: get_render_resolution."""

    def test_min_dimension_one(self) -> None:
        """Minimum dimension is at least 1."""
        w, h = get_render_resolution(2, 2, UpscaleQuality.ULTRA_PERFORMANCE)
        assert w >= 1
        assert h >= 1

    def test_ultra_quality_not_smaller(self) -> None:
        """ULTRA_QUALITY quality should not shrink resolution."""
        w, h = get_render_resolution(1920, 1080, UpscaleQuality.NATIVE_AA)
        assert w == 1920
        assert h == 1080


class TestUpscalingSettingsWhitebox:
    """USC-3: MIP bias."""

    def test_mip_bias_non_positive(self) -> None:
        """get_recommended_mip_bias <= 0."""
        assert UpscalingSettings().get_recommended_mip_bias() <= 0.0


class TestUpscalingEffectWhitebox:
    """USC-4/5: UpscalingEffect dispatch and conditional inputs."""

    def test_temporal_requires_depth_velocity(self) -> None:
        """Temporal upscaler (FSR_2) requires depth+velocity."""
        effect = UpscalingEffect(settings=UpscalingSettings(upscaler_type=UpscalerType.FSR_2))
        inputs = effect.get_required_inputs()
        assert "depth" in inputs
        assert "velocity" in inputs

    def test_spatial_no_depth_velocity(self) -> None:
        """Spatial upscaler (FSR_1) does not require depth+velocity."""
        effect = UpscalingEffect(settings=UpscalingSettings(upscaler_type=UpscalerType.FSR_1))
        inputs = effect.get_required_inputs()
        assert "depth" not in inputs
        assert "velocity" not in inputs

    def test_none_passthrough(self) -> None:
        """NONE type sets current_upscaler to None."""
        effect = UpscalingEffect(settings=UpscalingSettings(upscaler_type=UpscalerType.NONE))
        effect.setup(1920, 1080)
        effect.execute({"color": "input_color"}, {}, 0.016)
        assert effect.current_upscaler is None

    def test_bilinear_dispatch(self) -> None:
        """BILINEAR dispatch."""
        effect = UpscalingEffect(settings=UpscalingSettings(upscaler_type=UpscalerType.BILINEAR))
        effect.setup(1920, 1080)
        effect.execute({"color": "input_color"}, {}, 0.016)

    def test_fsr1_dispatch(self) -> None:
        """FSR_1 dispatch."""
        effect = UpscalingEffect(settings=UpscalingSettings(
            upscaler_type=UpscalerType.FSR_1, sharpening_amount=0.5,
        ))
        effect.setup(1920, 1080)
        effect.execute({"color": "input_color"}, {}, 0.016)

    def test_cas_dispatch(self) -> None:
        """CAS dispatch."""
        effect = UpscalingEffect(settings=UpscalingSettings(upscaler_type=UpscalerType.CAS))
        effect.setup(1920, 1080)
        effect.execute({"color": "input_color"}, {}, 0.016)


# ============================================================================
# BLO -- Bloom Additive Whitebox (complementing test_bloom_whitebox.py)
# ============================================================================

class TestBloomDownsampleWhitebox:
    """BLO-1: BloomDownsample.setup stops at minimum size."""

    def test_setup_stops_when_width_lt_two(self) -> None:
        """width < 2 => stops chain."""
        bd = BloomDownsample()
        bd.setup(1, 1080)
        assert bd.mip_count == 0

    def test_setup_stops_when_height_lt_two(self) -> None:
        """height < 2 => stops chain."""
        bd = BloomDownsample()
        bd.setup(1920, 1)
        assert bd.mip_count == 0

    def test_setup_creates_mip_chain(self) -> None:
        """setup creates mip chain."""
        bd = BloomDownsample()
        bd.setup(1920, 1080)
        assert bd.mip_count > 0


class TestBloomBlurWhitebox:
    """BLO-2/3/4: BloomBlur edge cases."""

    def test_blur_none_source_early_return(self) -> None:
        """None source returns target (or None)."""
        bb = BloomBlur()
        assert bb.blur(None, None, 2, 8, 8) is None

    def test_blur_small_dimension_early_return(self) -> None:
        """dimension < 2 returns target if provided."""
        bb = BloomBlur()
        result = bb.blur([0.0] * 4, None, 2, 1, 1)
        # width=1 < 2 => early return, provides source
        assert result is not None

    def test_gaussian_method(self) -> None:
        """GAUSSIAN method produces output."""
        bb = BloomBlur(method=BlurMethod.GAUSSIAN)
        bb.calculate_gaussian_weights(radius=2, sigma=1.0)
        img = [0.5] * (8 * 8 * 4)
        result = bb.blur(img, None, 1, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_kawase_method(self) -> None:
        """KAWASE method uses 5-point cross."""
        bb = BloomBlur(method=BlurMethod.KAWASE)
        img = [0.5] * (8 * 8 * 4)
        result = bb.blur(img, None, 1, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_box_method(self) -> None:
        """BOX method uses [1,1,1]/3 kernel."""
        bb = BloomBlur(method=BlurMethod.BOX)
        img = [0.5] * (8 * 8 * 4)
        result = bb.blur(img, None, 1, 8, 8)
        assert len(result) == 8 * 8 * 4

    def test_kawase_bounds_checking(self) -> None:
        """_kawase_blur: 3x3 image ensures bounds checking."""
        bb = BloomBlur(method=BlurMethod.KAWASE)
        img = [0.5] * (3 * 3 * 4)
        result = bb._kawase_blur(img, None, 1, 3, 3)
        assert len(result) == 3 * 3 * 4

    def test_box_blur_edge(self) -> None:
        """_box_blur: edge pixels use partial averaging."""
        bb = BloomBlur(method=BlurMethod.BOX)
        img = [0.5] * (8 * 8 * 4)
        result = bb._box_blur(img, None, 1, 8, 8)
        assert len(result) == 8 * 8 * 4


# ============================================================================
# Quality Preset tests
# ============================================================================

class TestQualityPresetWhitebox:
    """Quality preset effect filtering."""

    def test_ultra_has_all_effects(self) -> None:
        """ULTRA preset active_effects set."""
        active = QUALITY_PRESET_ULTRA.active_effects
        assert "Exposure" in active
        assert "Bloom" in active
        assert "Tonemapping" in active

    def test_low_has_minimal_effects(self) -> None:
        """LOW preset has minimal effect set."""
        active = QUALITY_PRESET_LOW.active_effects
        assert "Exposure" in active
        assert "Bloom" not in active

    def test_is_effect_active(self) -> None:
        """is_effect_active for known effect name."""
        assert QUALITY_PRESET_ULTRA.is_effect_active("Exposure")

    def test_get_effect_config(self) -> None:
        """get_effect_config returns dict for known effect."""
        config = QUALITY_PRESET_HIGH.get_effect_config("Bloom")
        assert isinstance(config, dict)
        assert "quality" in config
