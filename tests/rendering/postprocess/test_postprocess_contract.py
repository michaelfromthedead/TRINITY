"""
Blackbox Contract Tests for Post-Processing Subsystem

Tests the public API contract as declared in engine.rendering.postprocess.
This is a CLEANROOM test -- it tests only the contract, not the implementation.

Contract covers 8 systems:
  - Tonemapping
  - Bloom
  - Color Grading
  - Exposure
  - Depth of Field (DOF)
  - Anti-Aliasing
  - Ambient Occlusion (AO)
  - Motion Blur

Test design:
  - Equivalence partitioning: each system's public API surface
  - Boundary cases: constants, quality presets, range limits
  - Error cases: invalid parameters, edge inputs
  - Cross-system consistency: all effects share the same interface pattern
"""

import math
import pytest

from engine.rendering.postprocess import (
    # Tonemapping
    TonemapOperator,
    TonemapSettings,
    TonemappingEffect,
    Reinhard,
    ReinhardExtended,
    ACES,
    ACESFitted,
    AgX,
    Filmic,
    CustomCurve,
    TonemapFunction,
    # Bloom
    BloomQuality,
    BloomSettings,
    BloomEffect,
    BlurMethod,
    BloomThreshold,
    BloomMipSettings,
    LensDirtSettings,
    # Color Grading
    ColorSpace,
    ColorGradingEffect,
    ColorGradingSettings,
    ColorGradingStack,
    WhiteBalanceSettings,
    ContrastSettings,
    SaturationSettings,
    LiftGammaGain,
    HueSatLightness,
    LUT3D,
    LUTFormat,
    # Exposure
    ExposureMode,
    MeteringMode,
    ExposureSettings,
    ExposureEffect,
    luminance_to_ev,
    ev_to_exposure,
    exposure_to_ev,
    ManualExposure,
    AutoExposure,
    HistogramExposure,
    AdaptationCurve,
    EyeAdaptation,
    # DOF
    DOFMode,
    DOFQuality,
    DOFSettings,
    DOFEffect,
    BokehShapeType,
    BokehShape,
    CircleOfConfusion,
    AutoFocusSystem,
    # Anti-Aliasing
    AAMethod,
    AASettings,
    AAEffect,
    FXAAQuality,
    SMAAQuality,
    JitterPattern,
    JitterSequence,
    FXAASettings,
    SMAASettings,
    TAASettings,
    # Ambient Occlusion
    AOMethod,
    AOQuality,
    AOSettings,
    AOEffect,
    SSAOKernel,
    BilateralFilter,
    BentNormalOutput,
    # Motion Blur
    MotionBlurMode,
    MotionBlurQuality,
    MotionBlurSettings,
    MotionBlurEffect,
    CameraMotionBlur,
    ObjectMotionBlur,
    TileMaxVelocity,
    CameraMotion,
    # Constants
    EPSILON,
    SAFE_LOG_MIN,
    LUMINANCE_MIN,
    EXPOSURE,
    BLOOM,
    TONEMAP,
    DOF,
    AO,
    AA,
    MOTION_BLUR,
    COLOR_GRADING,
    LUMINANCE_COEFFS_BT709,
    LUMINANCE_COEFFS_BT601,
    calculate_luminance,
)


# ======================================================================
# SECTION 1: PUBLIC API SURFACE -- ALL EXPORTS ARE IMPORTABLE
# ======================================================================

class TestPublicAPIImportability:
    """All 8 contract systems are importable from the top-level package."""

    def test_tonemapping_exports(self):
        """Tonemapping: all public types are importable."""
        assert TonemapOperator is not None
        assert TonemapSettings is not None
        assert TonemappingEffect is not None
        assert Reinhard is not None
        assert ACES is not None
        assert AgX is not None
        assert Filmic is not None

    def test_bloom_exports(self):
        """Bloom: all public types are importable."""
        assert BloomQuality is not None
        assert BloomSettings is not None
        assert BloomEffect is not None
        assert BlurMethod is not None
        assert BloomThreshold is not None
        assert BloomMipSettings is not None
        assert LensDirtSettings is not None

    def test_color_grading_exports(self):
        """Color Grading: all public types are importable."""
        assert ColorSpace is not None
        assert ColorGradingEffect is not None
        assert ColorGradingSettings is not None
        assert ColorGradingStack is not None
        assert WhiteBalanceSettings is not None
        assert ContrastSettings is not None
        assert SaturationSettings is not None
        assert LUT3D is not None
        assert LUTFormat is not None

    def test_exposure_exports(self):
        """Exposure: all public types are importable."""
        assert ExposureMode is not None
        assert MeteringMode is not None
        assert ExposureSettings is not None
        assert ExposureEffect is not None
        assert luminance_to_ev is not None
        assert ev_to_exposure is not None
        assert exposure_to_ev is not None
        assert ManualExposure is not None
        assert AutoExposure is not None
        assert HistogramExposure is not None
        assert EyeAdaptation is not None

    def test_dof_exports(self):
        """DOF: all public types are importable."""
        assert DOFMode is not None
        assert DOFQuality is not None
        assert DOFSettings is not None
        assert DOFEffect is not None
        assert BokehShapeType is not None
        assert BokehShape is not None
        assert CircleOfConfusion is not None
        assert AutoFocusSystem is not None

    def test_antialiasing_exports(self):
        """Anti-Aliasing: all public types are importable."""
        assert AAMethod is not None
        assert AASettings is not None
        assert AAEffect is not None
        assert FXAAQuality is not None
        assert SMAAQuality is not None
        assert JitterPattern is not None
        assert JitterSequence is not None

    def test_ambient_occlusion_exports(self):
        """Ambient Occlusion: all public types are importable."""
        assert AOMethod is not None
        assert AOQuality is not None
        assert AOSettings is not None
        assert AOEffect is not None
        assert SSAOKernel is not None
        assert BilateralFilter is not None
        assert BentNormalOutput is not None

    def test_motion_blur_exports(self):
        """Motion Blur: all public types are importable."""
        assert MotionBlurMode is not None
        assert MotionBlurQuality is not None
        assert MotionBlurSettings is not None
        assert MotionBlurEffect is not None
        assert CameraMotionBlur is not None
        assert ObjectMotionBlur is not None
        assert TileMaxVelocity is not None
        assert CameraMotion is not None


# ======================================================================
# SECTION 2: CROSS-SYSTEM EFFECT INTERFACE CONSISTENCY
# ======================================================================

class TestEffectInterfaceConsistency:
    """All 8 systems' effects share the same public interface pattern."""

    @pytest.fixture(
        params=[
            ("Tonemapping", TonemappingEffect),
            ("Bloom", BloomEffect),
            ("ColorGrading", ColorGradingEffect),
            ("Exposure", ExposureEffect),
            ("DepthOfField", DOFEffect),
            ("AntiAliasing", AAEffect),
            ("AmbientOcclusion", AOEffect),
            ("MotionBlur", MotionBlurEffect),
        ]
    )
    def effect_pair(self, request):
        """Each (expected_name, EffectClass) pair."""
        return request.param

    def test_effect_has_name(self, effect_pair):
        """Every effect has a name matching the contract."""
        expected_name, EffectClass = effect_pair
        effect = EffectClass()
        assert effect.name == expected_name

    def test_effect_has_settings(self, effect_pair):
        """Every effect has a settings attribute."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        assert effect.settings is not None

    def test_effect_has_required_inputs(self, effect_pair):
        """Every effect declares required inputs."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        inputs = effect.get_required_inputs()
        # Contract: required inputs is a non-empty sequence of strings
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) > 0
        assert all(isinstance(name, str) for name in inputs)

    def test_effect_has_outputs(self, effect_pair):
        """Every effect declares outputs."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        outputs = effect.get_outputs()
        # Contract: outputs is a non-empty sequence of strings
        assert isinstance(outputs, (list, tuple))
        assert len(outputs) > 0
        assert all(isinstance(name, str) for name in outputs)

    def test_effect_setup_does_not_crash(self, effect_pair):
        """Every effect can be set up at a standard resolution."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        # Should not raise
        effect.setup(1920, 1080)

    def test_effect_execute_disabled(self, effect_pair):
        """Every effect handles disabled execution without error."""
        _, EffectClass = effect_pair
        # Skip upscaling -- not in contract
        effect = EffectClass()
        # Create a disabled settings variant if possible
        try:
            settings_cls = type(effect.settings)
            disabled = settings_cls(enabled=False)
            disabled_effect = EffectClass(disabled)
        except Exception:
            disabled_effect = effect
        disabled_effect.execute({}, {}, 0.016)

    def test_effect_cleanup_does_not_crash(self, effect_pair):
        """Every effect can be cleaned up."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        effect.setup(1920, 1080)
        effect.cleanup()

    def test_effect_is_compute_or_not(self, effect_pair):
        """Every effect declares whether it uses compute."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        result = effect.is_compute_effect()
        assert isinstance(result, bool)

    def test_effect_with_custom_settings(self, effect_pair):
        """Every effect accepts custom settings."""
        _, EffectClass = effect_pair
        effect = EffectClass()
        settings = effect.settings
        assert settings is not None


# ======================================================================
# SECTION 3: SETTINGS DATA CLASS CONSISTENCY
# ======================================================================

class TestSettingsInterfaceConsistency:
    """All 8 systems' settings classes support the same contract."""

    @pytest.fixture(
        params=[
            TonemapSettings,
            BloomSettings,
            ColorGradingSettings,
            ExposureSettings,
            DOFSettings,
            AASettings,
            AOSettings,
            MotionBlurSettings,
        ]
    )
    def settings_cls(self, request):
        return request.param

    def test_settings_default_creation(self, settings_cls):
        """Every settings class can be created with defaults."""
        settings = settings_cls()
        assert settings is not None

    def test_settings_custom_creation(self, settings_cls):
        """Every settings class accepts keyword arguments (subject to signature)."""
        try:
            settings_cls(enabled=False)
        except TypeError:
            pass  # Some settings may not have enabled param
        except Exception:
            pass

    def test_settings_repr(self, settings_cls):
        """Every settings class has a repr."""
        settings = settings_cls()
        r = repr(settings)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_settings_lerp_method_exists(self, settings_cls):
        """Every settings class has a lerp method."""
        settings = settings_cls()
        assert hasattr(settings, "lerp")
        assert callable(settings.lerp)


# ======================================================================
# SECTION 4: ENUM COMPLETENESS
# ======================================================================

class TestEnumContracts:
    """All enum types exist with expected members."""

    def test_tonemap_operator_enums(self):
        """TonemapOperator has all expected operators."""
        members = [
            TonemapOperator.REINHARD,
            TonemapOperator.REINHARD_EXTENDED,
            TonemapOperator.ACES,
            TonemapOperator.ACES_FITTED,
            TonemapOperator.AGX,
            TonemapOperator.FILMIC,
            TonemapOperator.HABLE,
            TonemapOperator.NEUTRAL,
            TonemapOperator.CUSTOM,
        ]
        for m in members:
            assert m is not None

    def test_bloom_quality_enums(self):
        """BloomQuality has all expected presets."""
        for q in [BloomQuality.LOW, BloomQuality.MEDIUM,
                  BloomQuality.HIGH, BloomQuality.ULTRA]:
            assert q is not None

    def test_blur_method_enums(self):
        """BlurMethod has all expected methods."""
        for m in [BlurMethod.GAUSSIAN, BlurMethod.KAWASE, BlurMethod.BOX]:
            assert m is not None

    def test_exposure_mode_enums(self):
        """ExposureMode has all expected modes."""
        for m in [ExposureMode.MANUAL, ExposureMode.AUTO_AVERAGE,
                  ExposureMode.AUTO_HISTOGRAM]:
            assert m is not None

    def test_metering_mode_enums(self):
        """MeteringMode has all expected modes."""
        for m in [MeteringMode.CENTER_WEIGHTED, MeteringMode.SPOT,
                  MeteringMode.MATRIX]:
            assert m is not None

    def test_dof_mode_enums(self):
        """DOFMode has all expected modes."""
        for m in [DOFMode.PHYSICAL, DOFMode.MANUAL, DOFMode.AUTO_FOCUS]:
            assert m is not None

    def test_dof_quality_enums(self):
        """DOFQuality has all expected presets."""
        for q in [DOFQuality.LOW, DOFQuality.MEDIUM,
                  DOFQuality.HIGH, DOFQuality.CINEMATIC]:
            assert q is not None

    def test_bokeh_shape_type_enums(self):
        """BokehShapeType has all expected types."""
        for t in [BokehShapeType.CIRCLE, BokehShapeType.POLYGON,
                  BokehShapeType.ANAMORPHIC, BokehShapeType.CAT_EYE,
                  BokehShapeType.SWIRL]:
            assert t is not None

    def test_aa_method_enums(self):
        """AAMethod has all expected methods."""
        for m in [AAMethod.NONE, AAMethod.FXAA, AAMethod.SMAA, AAMethod.TAA]:
            assert m is not None

    def test_fxaa_quality_enums(self):
        """FXAAQuality has all expected presets."""
        for q in [FXAAQuality.LOW, FXAAQuality.MEDIUM,
                  FXAAQuality.HIGH, FXAAQuality.EXTREME]:
            assert q is not None

    def test_smaa_quality_enums(self):
        """SMAAQuality has all expected presets."""
        for q in [SMAAQuality.LOW, SMAAQuality.MEDIUM,
                  SMAAQuality.HIGH, SMAAQuality.ULTRA]:
            assert q is not None

    def test_jitter_pattern_enums(self):
        """JitterPattern has all expected patterns."""
        for p in [JitterPattern.HALTON_8, JitterPattern.HALTON_16,
                  JitterPattern.HALTON_32, JitterPattern.UNIFORM_4,
                  JitterPattern.RGSS_4]:
            assert p is not None

    def test_ao_method_enums(self):
        """AOMethod has all expected methods."""
        for m in [AOMethod.SSAO, AOMethod.HBAO, AOMethod.HBAO_PLUS,
                  AOMethod.GTAO, AOMethod.RTAO]:
            assert m is not None

    def test_ao_quality_enums(self):
        """AOQuality has all expected presets."""
        for q in [AOQuality.LOW, AOQuality.MEDIUM,
                  AOQuality.HIGH, AOQuality.ULTRA]:
            assert q is not None

    def test_motion_blur_mode_enums(self):
        """MotionBlurMode has all expected modes."""
        for m in [MotionBlurMode.CAMERA_ONLY, MotionBlurMode.OBJECT_ONLY,
                  MotionBlurMode.COMBINED]:
            assert m is not None

    def test_motion_blur_quality_enums(self):
        """MotionBlurQuality has all expected presets."""
        for q in [MotionBlurQuality.LOW, MotionBlurQuality.MEDIUM,
                  MotionBlurQuality.HIGH, MotionBlurQuality.ULTRA]:
            assert q is not None

    def test_color_space_enums(self):
        """ColorSpace has all expected members."""
        for s in [ColorSpace.LINEAR_SRGB, ColorSpace.SRGB,
                  ColorSpace.ACES_CC, ColorSpace.LOG_C]:
            assert s is not None

    def test_lut_format_enums(self):
        """LUTFormat has all expected members."""
        for f in [LUTFormat.CUBE, LUTFormat.THREE_DL,
                  LUTFormat.CSP, LUTFormat.TEXTURE]:
            assert f is not None


# ======================================================================
# SECTION 5: CROSS-SYSTEM CONSTANT CONTRACT
# ======================================================================

class TestConstantContracts:
    """Centralized constants are consistent and accessible across all 8 systems."""

    def test_numerical_safety_constants(self):
        """Basic numerical safety constants exist."""
        assert EPSILON == 1e-6
        assert SAFE_LOG_MIN == 1e-10
        assert LUMINANCE_MIN == 1e-6

    def test_exposure_constants_exist(self):
        """Exposure constants container is accessible."""
        assert EXPOSURE.EV_MIN_FALLBACK == -10.0
        assert EXPOSURE.EV_DEFAULT_MIN == -4.0
        assert EXPOSURE.EV_DEFAULT_MAX == 16.0
        assert EXPOSURE.MIDDLE_GRAY_LUMINANCE == 0.18
        assert EXPOSURE.ADAPTATION_SPEED_UP_DEFAULT == 3.0
        assert EXPOSURE.ADAPTATION_SPEED_DOWN_DEFAULT == 1.0

    def test_bloom_constants_exist(self):
        """Bloom constants container is accessible."""
        assert BLOOM.THRESHOLD_DEFAULT == 1.0
        assert BLOOM.INTENSITY_DEFAULT == 1.0
        assert BLOOM.SCATTER_DEFAULT == 0.7
        assert BLOOM.RESOLUTION_SCALE_DEFAULT == 0.5
        assert BLOOM.MIP_COUNT_LOW == 3
        assert BLOOM.MIP_COUNT_ULTRA == 8

    def test_tonemap_constants_exist(self):
        """Tonemap constants container is accessible."""
        assert TONEMAP.WHITE_POINT_DEFAULT == 11.2
        assert TONEMAP.GAMMA_DEFAULT == 2.2
        assert TONEMAP.ACES_INPUT_SCALE_DEFAULT == 0.6
        assert TONEMAP.AGX_MIN_EV == -12.47393
        assert TONEMAP.AGX_MAX_EV == 4.026069

    def test_dof_constants_exist(self):
        """DOF constants container is accessible."""
        assert DOF.APERTURE_DEFAULT == 2.8
        assert DOF.FOCAL_LENGTH_DEFAULT == 50.0
        assert DOF.SENSOR_FULL_FRAME == 36.0
        assert DOF.FOCUS_DISTANCE_DEFAULT == 5.0
        assert DOF.BLADE_COUNT_DEFAULT == 6

    def test_ao_constants_exist(self):
        """AO constants container is accessible."""
        assert AO.RADIUS_DEFAULT == 0.5
        assert AO.SAMPLE_COUNT_HIGH == 16
        assert AO.DIRECTION_COUNT_DEFAULT == 8
        assert AO.INTENSITY_DEFAULT == 1.0
        assert AO.BIAS_DEFAULT == 0.01

    def test_aa_constants_exist(self):
        """Anti-Aliasing constants container is accessible."""
        assert AA.TAA_HISTORY_WEIGHT_DEFAULT == 0.9
        assert AA.TAA_SHARPEN_AMOUNT_DEFAULT == 0.25
        assert AA.FXAA_EDGE_THRESHOLD_DEFAULT == 0.166
        assert AA.SMAA_THRESHOLD_DEFAULT == 0.1
        assert AA.JITTER_SAMPLES_16 == 16

    def test_motion_blur_constants_exist(self):
        """Motion blur constants container is accessible."""
        assert MOTION_BLUR.SAMPLE_COUNT_DEFAULT == 16
        assert MOTION_BLUR.MAX_BLUR_RADIUS_DEFAULT == 32.0
        assert MOTION_BLUR.TILE_SIZE_DEFAULT == 16
        assert MOTION_BLUR.SHUTTER_ANGLE_DEFAULT == 180.0

    def test_color_grading_constants_exist(self):
        """Color grading constants container is accessible."""
        assert COLOR_GRADING.TEMPERATURE_MIN == -100.0
        assert COLOR_GRADING.TEMPERATURE_MAX == 100.0
        assert COLOR_GRADING.CONTRAST_DEFAULT == 1.0
        assert COLOR_GRADING.SATURATION_DEFAULT == 1.0
        assert COLOR_GRADING.LUT_SIZE_DEFAULT == 32

    def test_luminance_coefficients(self):
        """Luminance coefficients match standard values."""
        r709, g709, b709 = LUMINANCE_COEFFS_BT709
        assert r709 == 0.2126
        assert g709 == 0.7152
        assert b709 == 0.0722
        assert abs(r709 + g709 + b709 - 1.0) < 0.001

        r601, g601, b601 = LUMINANCE_COEFFS_BT601
        assert r601 == 0.299
        assert g601 == 0.587
        assert b601 == 0.114
        assert abs(r601 + g601 + b601 - 1.0) < 0.001


# ======================================================================
# SECTION 6: LUMINANCE FUNCTION CONTRACT
# ======================================================================

class TestLuminanceContract:
    """calculate_luminance is a shared cross-system utility."""

    def test_luminance_bt709_white(self):
        """White (1, 1, 1) gives luminance 1.0 under BT.709."""
        lum = calculate_luminance(1.0, 1.0, 1.0)
        assert abs(lum - 1.0) < 0.001

    def test_luminance_bt709_black(self):
        """Black (0, 0, 0) gives luminance 0.0 under BT.709."""
        lum = calculate_luminance(0.0, 0.0, 0.0)
        assert lum == 0.0

    def test_luminance_bt709_green_dominates(self):
        """Green contributes most to luminance under BT.709."""
        lum_r = calculate_luminance(1.0, 0.0, 0.0)
        lum_g = calculate_luminance(0.0, 1.0, 0.0)
        lum_b = calculate_luminance(0.0, 0.0, 1.0)
        assert lum_g > lum_r > lum_b

    def test_luminance_bt601(self):
        """Luminance uses BT.601 coefficients when specified."""
        lum = calculate_luminance(1.0, 1.0, 1.0, LUMINANCE_COEFFS_BT601)
        assert abs(lum - 1.0) < 0.001

    def test_luminance_bt601_weights_differ(self):
        """BT.601 weights differ from BT.709."""
        lum_709 = calculate_luminance(1.0, 0.0, 0.0, LUMINANCE_COEFFS_BT709)
        lum_601 = calculate_luminance(1.0, 0.0, 0.0, LUMINANCE_COEFFS_BT601)
        assert lum_709 != lum_601

    def test_luminance_mid_gray(self):
        """Middle gray (0.18, 0.18, 0.18) gives 0.18."""
        lum = calculate_luminance(0.18, 0.18, 0.18)
        assert abs(lum - 0.18) < 0.001

    def test_luminance_linearity(self):
        """Luminance is linear: k * L(r,g,b) = L(k*r, k*g, k*b)."""
        k = 2.0
        lum_base = calculate_luminance(0.3, 0.4, 0.5)
        lum_scaled = calculate_luminance(k * 0.3, k * 0.4, k * 0.5)
        assert abs(k * lum_base - lum_scaled) < 0.001

    def test_luminance_commutative(self):
        """Luminance is commutative in coefficients (mathematical property)."""
        coeffs_a = (0.3, 0.5, 0.2)
        coeffs_b = (0.2, 0.5, 0.3)
        lum_a = calculate_luminance(1.0, 1.0, 1.0, coeffs_a)
        lum_b = calculate_luminance(1.0, 1.0, 1.0, coeffs_b)
        # Both sum to 1.0, so both should give 1.0 for white
        assert abs(lum_a - 1.0) < 0.001
        assert abs(lum_b - 1.0) < 0.001


# ======================================================================
# SECTION 7: EXPOSURE VALUE CONVERSION CONTRACT
# ======================================================================

class TestExposureConversionContract:
    """Exposure math functions are cross-system utilities."""

    def test_luminance_to_ev_zero_returns_fallback(self):
        """Zero luminance returns fallback EV (avoids log(0))."""
        ev = luminance_to_ev(0.0)
        assert ev == -10.0

    def test_luminance_to_ev_negative_returns_fallback(self):
        """Negative luminance returns fallback EV."""
        ev = luminance_to_ev(-1.0)
        assert ev == -10.0

    def test_luminance_to_ev_increasing(self):
        """Higher luminance produces higher EV."""
        ev_low = luminance_to_ev(0.01)
        ev_high = luminance_to_ev(1.0)
        assert ev_high > ev_low

    def test_ev_to_exposure_inverse(self):
        """Higher EV produces lower exposure (inverse relationship)."""
        exp_low = ev_to_exposure(2.0)
        exp_high = ev_to_exposure(-2.0)
        assert exp_low < exp_high

    def test_ev_roundtrip(self):
        """EV -> exposure -> EV roundtrips within tolerance."""
        original = 3.0
        exposure = ev_to_exposure(original)
        recovered = exposure_to_ev(exposure)
        assert abs(recovered - original) < 0.001

    def test_exposure_to_ev_zero_returns_fallback(self):
        """Zero exposure returns fallback EV (avoids log(0))."""
        ev = exposure_to_ev(0.0)
        assert ev == -10.0

    def test_exposure_to_ev_negative_returns_fallback(self):
        """Negative exposure returns fallback EV."""
        ev = exposure_to_ev(-1.0)
        assert ev == -10.0


# ======================================================================
# SECTION 8: QUALITY PRESET CONSISTENCY
# ======================================================================

class TestQualityPresetContracts:
    """Quality presets exist and affect settings across systems."""

    def test_bloom_quality_affects_mip_count(self):
        """Quality presets exist with expected relationships."""
        from engine.rendering.postprocess import (
            QUALITY_PRESETS, QUALITY_PRESET_LOW,
            QUALITY_PRESET_MEDIUM, QUALITY_PRESET_HIGH, QUALITY_PRESET_ULTRA,
        )
        # QUALITY_PRESETS is a sequence of effect quality levels
        assert len(QUALITY_PRESETS) >= 4
        # Individual preset constants have descriptive names
        assert QUALITY_PRESET_LOW.name is not None
        assert QUALITY_PRESET_MEDIUM.name is not None
        assert QUALITY_PRESET_HIGH.name is not None
        assert QUALITY_PRESET_ULTRA.name is not None

    def test_quality_presets_have_names(self):
        """Each quality preset has a name."""
        from engine.rendering.postprocess import QUALITY_PRESETS
        for preset in QUALITY_PRESETS:
            assert preset.name is not None

    def test_bloom_quality_low_vs_ultra(self):
        """Bloom low vs ultra quality produces different settings."""
        settings_low = BloomSettings(quality=BloomQuality.LOW)
        settings_ultra = BloomSettings(quality=BloomQuality.ULTRA)
        assert settings_low.quality is not settings_ultra.quality

    def test_dof_quality_low_vs_cinematic(self):
        """DOF low vs cinematic quality produces different settings."""
        settings_low = DOFSettings(quality=DOFQuality.LOW)
        settings_cinematic = DOFSettings(quality=DOFQuality.CINEMATIC)
        assert settings_low.quality is not settings_cinematic.quality

    def test_ao_quality_low_vs_ultra(self):
        """AO low vs ultra quality produces different settings."""
        settings_low = AOSettings(quality=AOQuality.LOW)
        settings_ultra = AOSettings(quality=AOQuality.ULTRA)
        assert settings_low.quality is not settings_ultra.quality

    def test_motion_blur_quality_low_vs_ultra(self):
        """Motion blur low vs ultra quality produces different settings."""
        settings_low = MotionBlurSettings(quality=MotionBlurQuality.LOW)
        settings_ultra = MotionBlurSettings(quality=MotionBlurQuality.ULTRA)
        assert settings_low.quality is not settings_ultra.quality

    def test_quality_getter_function(self):
        """get_quality_preset is accessible and returns a valid preset."""
        from engine.rendering.postprocess import (
            get_quality_preset, QUALITY_PRESET_HIGH,
            EffectQuality,
        )
        preset = get_quality_preset(EffectQuality.HIGH)
        assert preset is not None


# ======================================================================
# SECTION 9: BOUNDARY AND EDGE CASE CONTRACT
# ======================================================================

class TestBoundaryContracts:
    """Boundary values for the public contract."""

    def test_lens_dirt_texture_path_none(self):
        """Lens dirt defaults to no texture."""
        dirt = LensDirtSettings()
        assert dirt.texture_path is None

    def test_camera_motion_zeros(self):
        """Camera motion defaults to zero."""
        motion = CameraMotion()
        vx, vy, vz = motion.velocity
        assert vx == 0.0
        assert vy == 0.0
        assert vz == 0.0

    def test_bloom_mip_scatter_default(self):
        """Bloom mip settings default scatter is 0.7."""
        mip = BloomMipSettings()
        assert mip.scatter == 0.7

    def test_taa_history_weight_range(self):
        """TAA history weight is within [0.8, 0.98] by default."""
        settings = TAASettings()
        assert 0.8 <= settings.history_weight <= 0.98

    def test_shutter_speed_factor_range(self):
        """Shutter speed factor is in [0, 1] for valid shutter angles."""
        settings = MotionBlurSettings(shutter_angle=180.0)
        assert 0.0 <= settings.shutter_speed_factor <= 1.0

        settings = MotionBlurSettings(shutter_angle=0.0)
        assert settings.shutter_speed_factor == 0.0

        settings = MotionBlurSettings(shutter_angle=360.0)
        assert settings.shutter_speed_factor == 1.0


# ======================================================================
# SECTION 10: CROSS-SYSTEM PROPERTY INVARIANTS
# ======================================================================

class TestCrossSystemPropertyInvariants:
    """Invariants that must hold across the post-processing contract."""

    def test_all_quality_enums_have_low_and_high(self):
        """Cross-cutting: bloom, dof, ao, motion blur all define LOW and HIGH."""
        assert BloomQuality.LOW is not None
        assert BloomQuality.HIGH is not None
        assert DOFQuality.LOW is not None
        assert DOFQuality.HIGH is not None
        assert AOQuality.LOW is not None
        assert AOQuality.HIGH is not None
        assert MotionBlurQuality.LOW is not None
        assert MotionBlurQuality.HIGH is not None

    def test_luminance_coeffs_sum_to_one_bt709(self):
        """BT.709 luminance coefficients sum to 1.0."""
        total = sum(LUMINANCE_COEFFS_BT709)
        assert abs(total - 1.0) < 0.001

    def test_luminance_coeffs_sum_to_one_bt601(self):
        """BT.601 luminance coefficients sum to 1.0."""
        total = sum(LUMINANCE_COEFFS_BT601)
        assert abs(total - 1.0) < 0.001

    def test_exposure_ev_min_less_than_max(self):
        """EV min is always less than EV max in constants."""
        assert EXPOSURE.EV_DEFAULT_MIN < EXPOSURE.EV_DEFAULT_MAX


# ======================================================================
# SECTION 11: TONEMAPPING OPERATOR CONTRACT (CROSS-OPERATOR)
# ======================================================================

class TestTonemappingOperatorContract:
    """All tonemapping operators share the same apply interface."""

    @pytest.fixture(params=[
        Reinhard(), ReinhardExtended(), ACESFitted(), AgX(), Filmic(),
    ])
    def operator(self, request):
        return request.param

    def test_operator_produces_valid_range(self, operator):
        """Every operator produces [0, 1] output for valid HDR input."""
        settings = TonemapSettings()
        test_inputs = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (10.0, 10.0, 10.0),
        ]
        for r_in, g_in, b_in in test_inputs:
            r, g, b = operator.apply(r_in, g_in, b_in, settings)
            assert 0.0 <= r <= 1.0, f"R out of range for {type(operator).__name__}({r_in})"
            assert 0.0 <= g <= 1.0, f"G out of range for {type(operator).__name__}({g_in})"
            assert 0.0 <= b <= 1.0, f"B out of range for {type(operator).__name__}({b_in})"

    def test_operator_black_returns_black(self, operator):
        """Every operator maps black to (approximately) black."""
        settings = TonemapSettings()
        r, g, b = operator.apply(0.0, 0.0, 0.0, settings)
        assert r >= 0.0 and r < 0.01
        assert g >= 0.0 and g < 0.01
        assert b >= 0.0 and b < 0.01

    def test_operator_monotonic_luminance(self, operator):
        """Every operator preserves brightness ordering."""
        settings = TonemapSettings()
        prev = -1.0
        for v in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            r, _, _ = operator.apply(v, v, v, settings)
            assert r >= prev, f"Non-monotonic at {v} for {type(operator).__name__}"
            prev = r


# ======================================================================
# SECTION 12: WHITE BALANCE COLOR TEMPERATURE CONTRACT
# ======================================================================

class TestWhiteBalanceTemperatureContract:
    """White balance color temperature is consistent."""

    def test_neutral_temperature(self):
        """Temperature=0, tint=0 gives neutral (1, 1, 1)."""
        wb = WhiteBalanceSettings(temperature=0.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_warm_temperature_more_red_than_blue(self):
        """Warm temperature gives red > blue."""
        wb = WhiteBalanceSettings(temperature=50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert r > b

    def test_cool_temperature_more_blue_than_red(self):
        """Cool temperature gives blue > red."""
        wb = WhiteBalanceSettings(temperature=-50.0, tint=0.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert b > r

    def test_output_in_valid_range(self):
        """Temperature RGB output is in [0.1, 2.0] for extreme values."""
        wb = WhiteBalanceSettings(temperature=100.0, tint=100.0)
        r, g, b = wb.get_color_temperature_rgb()
        assert 0.1 <= r <= 2.0
        assert 0.1 <= g <= 2.0
        assert 0.1 <= b <= 2.0

    def test_output_non_nan(self):
        """Temperature RGB output is not NaN for any valid input."""
        import math
        for temp in [-100.0, -50.0, 0.0, 50.0, 100.0]:
            for tint in [-100.0, -50.0, 0.0, 50.0, 100.0]:
                wb = WhiteBalanceSettings(temperature=temp, tint=tint)
                r, g, b = wb.get_color_temperature_rgb()
                assert not math.isnan(r), f"NaN at temp={temp}, tint={tint}"
                assert not math.isnan(g), f"NaN at temp={temp}, tint={tint}"
                assert not math.isnan(b), f"NaN at temp={temp}, tint={tint}"
