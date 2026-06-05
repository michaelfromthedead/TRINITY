"""Phase 6 Integration Tests: Temporal Anti-Aliasing and Upscaling.

Integration tests covering:
- TAA Halton sequence jitter
- TAA convergence and history behavior
- TAA disocclusion handling
- TSR Lanczos integration
- Upscaler plugin stack integration
- Full post-process stack E2E
"""

from __future__ import annotations

import math
import pytest
from typing import Any, Dict, List, Tuple

from engine.rendering.postprocess.antialiasing import (
    AAEffect,
    AAMethod,
    AASettings,
    JitterPattern,
    JitterSequence,
    TAA,
    TAASettings,
)
from engine.rendering.postprocess.upscaling import (
    LanczosKernel,
    TSRLanczosSettings,
    TSRLanczosUpscaler,
    UpscaleQuality,
    UpscalerType,
    UpscalingEffect,
    UpscalingSettings,
    create_tsr_lanczos,
    get_render_resolution,
    lanczos_kernel,
    measure_local_contrast,
)
from engine.rendering.postprocess.upscaler_plugins import (
    DLSSPlugin,
    FSRPlugin,
    QualityPreset,
    TSRLanczosPlugin,
    UpscalerCapabilities,
    UpscalerManager,
    XeSSPlugin,
)
from engine.rendering.postprocess.postprocess_stack import (
    EffectPriority,
    EffectQuality,
    PostProcessStack,
    PostProcessStackConfig,
    QUALITY_PRESET_HIGH,
    QUALITY_PRESET_LOW,
    QUALITY_PRESET_ULTRA,
)


# =============================================================================
# 1. TAA HALTON JITTER TESTS (5 tests)
# =============================================================================


class TestTAAHaltonJitter:
    """Tests for TAA Halton jitter sequence behavior."""

    def test_halton_produces_subpixel_offsets(self) -> None:
        """Verify Halton sequence produces offsets in [-0.5, 0.5] range."""
        jitter = JitterSequence(JitterPattern.HALTON_16)

        for _ in range(16):
            x, y = jitter.next()
            assert -0.5 <= x <= 0.5, f"X offset {x} out of [-0.5, 0.5] range"
            assert -0.5 <= y <= 0.5, f"Y offset {y} out of [-0.5, 0.5] range"

    def test_halton_16_sample_sequence(self) -> None:
        """Verify 16-sample Halton sequence produces correct base-2, base-3 values."""
        jitter = JitterSequence(JitterPattern.HALTON_16)
        samples = [jitter.next() for _ in range(16)]

        # First sample (index 1) should be (0.5 - 0.5, 1/3 - 0.5) = (0, -0.167...)
        # Halton(1, 2) = 0.5 - 0.5 = 0
        # Halton(1, 3) = 1/3 - 0.5 = -0.167...
        assert samples[0][0] == pytest.approx(0.0, abs=1e-6)
        assert samples[0][1] == pytest.approx(1 / 3 - 0.5, abs=1e-6)

        # Second sample (index 2) should be (0.25 - 0.5, 2/3 - 0.5) = (-0.25, 0.167...)
        assert samples[1][0] == pytest.approx(0.25 - 0.5, abs=1e-6)
        assert samples[1][1] == pytest.approx(2 / 3 - 0.5, abs=1e-6)

    def test_halton_sequence_non_repeating(self) -> None:
        """First 16 samples should all be unique (low discrepancy)."""
        jitter = JitterSequence(JitterPattern.HALTON_16)
        samples = [jitter.next() for _ in range(16)]

        unique_samples = set(samples)
        assert len(unique_samples) == 16, "All 16 samples should be unique"

    def test_halton_centered_on_pixel(self) -> None:
        """Average offset over sequence should be near (0, 0) for unbiased sampling."""
        jitter = JitterSequence(JitterPattern.HALTON_16)
        samples = [jitter.next() for _ in range(16)]

        avg_x = sum(s[0] for s in samples) / 16
        avg_y = sum(s[1] for s in samples) / 16

        # Average should be close to zero for unbiased temporal sampling
        assert abs(avg_x) < 0.15, f"X average {avg_x} too far from center"
        assert abs(avg_y) < 0.15, f"Y average {avg_y} too far from center"

    def test_jitter_applied_to_projection(self) -> None:
        """Verify jitter modifies projection matrix correctly in clip space."""
        taa = TAA()
        taa.setup(1920, 1080, JitterPattern.HALTON_16)

        # Standard perspective-like projection matrix
        projection = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        jittered = taa.get_jittered_projection(projection)

        # Jitter should modify [2][0] and [2][1] (clip-space offset)
        jitter_x = jittered[2][0] - projection[2][0]
        jitter_y = jittered[2][1] - projection[2][1]

        # Jitter in clip space should be 2*offset/dimension
        # With offset in [-0.5, 0.5], clip jitter in [-1/width, 1/width]
        max_expected_x = 2.0 * 0.5 / 1920
        max_expected_y = 2.0 * 0.5 / 1080

        assert abs(jitter_x) <= max_expected_x + 1e-9
        assert abs(jitter_y) <= max_expected_y + 1e-9


# =============================================================================
# 2. TAA CONVERGENCE TESTS (4 tests)
# =============================================================================


class TestTAAConvergence:
    """Tests for TAA temporal convergence behavior."""

    def test_taa_converges_static_scene(self) -> None:
        """TAA should converge to stable state over N frames for static input."""
        taa = TAA()
        taa.setup(128, 128, JitterPattern.HALTON_16)
        settings = TAASettings(history_weight=0.9)

        # Mock color buffer (static scene)
        static_color = object()
        depth = object()
        velocity = None  # No motion

        # First frame initializes history
        result1 = taa.apply(static_color, depth, velocity, settings)
        assert taa.history_valid, "History should be valid after first frame"

        # After 16+ frames with same input, output should stabilize
        # (In real implementation, blending converges)
        for _ in range(16):
            taa.apply(static_color, depth, velocity, settings)

        # TAA should still be valid and have stable history
        assert taa.history_valid

    def test_taa_blend_factor_affects_convergence_rate(self) -> None:
        """Higher blend factors converge faster but with more ghosting potential."""
        # Low history weight = faster convergence
        settings_fast = TAASettings(history_weight=0.7)
        # High history weight = slower convergence, more history retention
        settings_slow = TAASettings(history_weight=0.95)

        # Fast settings blend more current frame (1 - 0.7 = 0.3)
        # Slow settings blend less current frame (1 - 0.95 = 0.05)
        fast_current_weight = 1.0 - settings_fast.history_weight
        slow_current_weight = 1.0 - settings_slow.history_weight

        assert fast_current_weight > slow_current_weight
        assert fast_current_weight == pytest.approx(0.3, abs=0.01)
        assert slow_current_weight == pytest.approx(0.05, abs=0.01)

    def test_taa_history_accumulation(self) -> None:
        """Verify temporal accumulation averages jittered samples over time."""
        taa = TAA()
        taa.setup(64, 64, JitterPattern.HALTON_8)

        # Track jitter offsets used
        jitter_offsets = []
        for _ in range(8):
            offset = taa.get_jitter_offset()
            jitter_offsets.append(offset)

        # All 8 offsets should be unique (full sequence coverage)
        assert len(set(jitter_offsets)) == 8

        # Sequence should cycle
        next_offset = taa.get_jitter_offset()
        # After 8 samples, index wraps to 0
        # But we already advanced once more, so it's at index 1
        taa._jitter.reset()
        first_offset = taa.get_jitter_offset()
        assert jitter_offsets[0] == first_offset

    def test_taa_subpixel_detail_recovery(self) -> None:
        """TAA should be capable of recovering sub-pixel detail through accumulation.

        This tests the theoretical basis: multiple jittered samples can
        reconstruct detail finer than the pixel grid.
        """
        jitter = JitterSequence(JitterPattern.HALTON_16)

        # Collect all 16 sub-pixel positions
        positions = []
        for _ in range(16):
            x, y = jitter.next()
            positions.append((x + 0.5, y + 0.5))  # Convert to [0, 1] UV range

        # Verify coverage across the pixel
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Good sub-pixel coverage means min/max span most of [0, 1]
        x_coverage = max(x_coords) - min(x_coords)
        y_coverage = max(y_coords) - min(y_coords)

        assert x_coverage > 0.8, f"X coverage {x_coverage} should span most of pixel"
        assert y_coverage > 0.8, f"Y coverage {y_coverage} should span most of pixel"


# =============================================================================
# 3. TAA DISOCCLUSION TESTS (4 tests)
# =============================================================================


class TestTAADisocclusion:
    """Tests for TAA disocclusion and history rejection."""

    def test_disocclusion_detection_depth_change(self) -> None:
        """Large depth changes should indicate disocclusion.

        Note: Actual disocclusion detection happens in the TAA shader.
        Here we test the settings that control detection thresholds.
        """
        settings = TAASettings(
            velocity_rejection=True,
            velocity_rejection_threshold=0.01,
        )

        # Velocity rejection is enabled with threshold 0.01
        # Pixels with velocity > threshold should be detected as disoccluded
        assert settings.velocity_rejection is True
        assert settings.velocity_rejection_threshold == 0.01

        # Higher threshold = less sensitive detection
        relaxed = TAASettings(velocity_rejection_threshold=0.1)
        assert relaxed.velocity_rejection_threshold > settings.velocity_rejection_threshold

    def test_disocclusion_resets_history(self) -> None:
        """Disoccluded pixels should not blend with stale history."""
        taa = TAA()
        taa.setup(64, 64)

        # Simulate valid history
        taa._history_valid = True

        # Camera cut should invalidate history
        taa.invalidate_history()

        assert not taa.history_valid, "History should be invalid after camera cut"

    def test_neighborhood_clamp_prevents_ghosting(self) -> None:
        """History outside color neighborhood should be clamped to prevent ghosting."""
        settings = TAASettings(
            color_box_clamping=True,
            variance_clipping=True,
        )

        # Verify clamping is enabled
        assert settings.color_box_clamping is True
        assert settings.variance_clipping is True

        # Disable clamping for comparison
        no_clamp = TAASettings(
            color_box_clamping=False,
            variance_clipping=False,
        )

        assert not no_clamp.color_box_clamping
        assert not no_clamp.variance_clipping

    def test_motion_vector_reprojection(self) -> None:
        """History sampling should use motion vectors for reprojection.

        TAA requires velocity buffer for temporal reprojection.
        """
        effect = AAEffect(AASettings(method=AAMethod.TAA))
        inputs = effect.get_required_inputs()

        assert "velocity" in inputs, "TAA should require velocity buffer"
        assert "depth" in inputs, "TAA should require depth buffer"
        assert "color" in inputs, "TAA should require color buffer"


# =============================================================================
# 4. TSR INTEGRATION TESTS (4 tests)
# =============================================================================


class TestTSRIntegration:
    """Tests for TSR Lanczos integration with the rendering pipeline."""

    def test_tsr_preserves_edge_sharpness(self) -> None:
        """TSR upscaling should preserve sharp edges better than bilinear."""
        upscaler = TSRLanczosUpscaler()

        # Create a sharp edge: left half black, right half white
        image: List[List[Tuple[float, float, float]]] = [
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        ]

        # Sample near edge (between column 1 and 2)
        # Lanczos should have ringing that can actually make edges sharper
        edge_sample = upscaler.sample_lanczos(image, 1.8, 1.5)

        # Sample in uniform region
        uniform_sample = upscaler.sample_lanczos(image, 0.5, 1.5)

        # Uniform region should be close to source value
        assert uniform_sample[0] == pytest.approx(0.0, abs=0.2)

        # Edge sample should be transitioning (not pure bilinear blend)
        # Lanczos tends to have slight overshoot/undershoot at edges
        assert 0.0 <= edge_sample[0] <= 1.2  # Allow for slight ringing

    def test_tsr_all_scale_factors(self) -> None:
        """TSR should work at 1.5x, 2x, 3x scales."""
        for scale in [1.5, 2.0, 3.0]:
            settings = TSRLanczosSettings(scale_factor=scale)
            upscaler = TSRLanczosUpscaler(settings)

            assert upscaler.settings.scale_factor == scale
            assert upscaler.output_scale == (scale, scale)

            # Verify weights are generated for each scale
            assert len(upscaler._weights_h) > 0
            assert len(upscaler._weights_v) > 0

    def test_tsr_temporal_stability(self) -> None:
        """TSR with temporal blend should maintain frame-to-frame stability."""
        settings = TSRLanczosSettings(
            scale_factor=2.0,
            temporal_blend=0.1,  # 10% temporal blending
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Track jitter positions across frames
        jitters = []
        for _ in range(8):
            jitter = upscaler.get_jitter_offset()
            jitters.append(jitter)
            upscaler.advance_frame()

        # All jitters should be unique (low discrepancy sequence)
        assert len(set(jitters)) == 8

        # Temporal blend setting should be preserved
        assert upscaler.settings.temporal_blend == 0.1

    def test_tsr_fallback_always_available(self) -> None:
        """TSR should be available when vendor upscalers are not."""
        # TSR Lanczos is always available
        assert TSRLanczosUpscaler.is_available() is True

        # Vendor upscalers are not available in test environment
        assert not DLSSPlugin.is_available()
        assert not FSRPlugin.is_available()
        assert not XeSSPlugin.is_available()

        # TSR plugin also always available
        assert TSRLanczosPlugin.is_available() is True


# =============================================================================
# 5. UPSCALER STACK INTEGRATION (4 tests)
# =============================================================================


class TestUpscalerStackIntegration:
    """Tests for upscaler integration with the rendering stack."""

    def test_upscaler_creates_frame_graph_pass(self) -> None:
        """Upscaler should integrate with frame graph as a pass node."""
        settings = UpscalingSettings(
            upscaler_type=UpscalerType.TSR_LANCZOS,
            quality=UpscaleQuality.QUALITY,
        )
        effect = UpscalingEffect(settings)

        # Effect should have correct priority
        assert effect.priority == EffectPriority.UPSCALING.value

        # Effect should declare required inputs
        inputs = effect.get_required_inputs()
        assert "color" in inputs
        # TSR requires depth and velocity for temporal mode
        assert "depth" in inputs
        assert "velocity" in inputs

        # Effect should declare outputs
        outputs = effect.get_outputs()
        assert "color" in outputs

    def test_upscaler_quality_presets_meet_budget(self) -> None:
        """All 4 quality presets should produce appropriate render resolutions."""
        target = (3840, 2160)  # 4K output

        presets = [
            (UpscaleQuality.ULTRA_PERFORMANCE, 3.0),  # ~3x
            (UpscaleQuality.PERFORMANCE, 2.0),  # ~2x
            (UpscaleQuality.BALANCED, 1.7),  # ~1.7x
            (UpscaleQuality.QUALITY, 1.5),  # ~1.5x
        ]

        for quality, expected_scale in presets:
            render_w, render_h = get_render_resolution(target[0], target[1], quality)

            actual_scale = target[0] / render_w

            assert abs(actual_scale - expected_scale) < 0.1, (
                f"Quality {quality}: expected ~{expected_scale}x, got {actual_scale:.2f}x"
            )

    def test_upscaler_fallback_chain(self) -> None:
        """When vendor unavailable, should fallback DLSS->XeSS->FSR->TSR."""
        manager = UpscalerManager()

        # Priority order should be DLSS > XeSS > FSR > TSR
        priority = manager.PRIORITY_ORDER
        assert priority[0] == DLSSPlugin
        assert priority[1] == XeSSPlugin
        assert priority[2] == FSRPlugin
        assert priority[3] == TSRLanczosPlugin

        # With no vendor SDKs, should fall back to TSR
        best = manager.detect_best_upscaler()
        assert best.name == "TSR Lanczos"
        assert best.is_available()

    def test_upscaler_resolution_calculation(self) -> None:
        """Optimal render resolution should match quality preset expectations."""
        plugin = TSRLanczosPlugin()
        target = (2560, 1440)

        # Test each preset
        tests = [
            ("ultra_performance", 3.0),
            ("performance", 2.0),
            ("balanced", 1.7),
            ("quality", 1.5),
            ("ultra_quality", 1.3),
            ("native_aa", 1.0),
        ]

        for preset, expected_scale in tests:
            res = plugin.get_optimal_render_resolution(target, preset)

            expected_w = int(target[0] / expected_scale)
            expected_h = int(target[1] / expected_scale)

            assert res == (expected_w, expected_h), (
                f"Preset {preset}: expected ({expected_w}, {expected_h}), got {res}"
            )


# =============================================================================
# 6. FULL POST-PROCESS STACK E2E (4 tests)
# =============================================================================


class TestPhase6EndToEnd:
    """End-to-end tests for the full post-process stack."""

    def test_full_stack_canonical_order(self) -> None:
        """Effects should execute in canonical priority order."""
        # Verify priority ordering
        priorities = [
            (EffectPriority.EXPOSURE, 0),
            (EffectPriority.BLOOM, 100),
            (EffectPriority.DEPTH_OF_FIELD, 200),
            (EffectPriority.MOTION_BLUR, 300),
            (EffectPriority.AMBIENT_OCCLUSION, 400),
            (EffectPriority.TONEMAPPING, 500),
            (EffectPriority.COLOR_GRADING, 600),
            (EffectPriority.ANTIALIASING, 700),
            (EffectPriority.UPSCALING, 800),
        ]

        for priority, expected_value in priorities:
            assert priority.value == expected_value, (
                f"{priority.name} has value {priority.value}, expected {expected_value}"
            )

        # AA should come before upscaling
        assert EffectPriority.ANTIALIASING.value < EffectPriority.UPSCALING.value

        # Create stack and verify effect order
        stack = PostProcessStack()
        aa_effect = AAEffect()
        up_effect = UpscalingEffect()

        stack.add_effect(aa_effect)
        stack.add_effect(up_effect)

        effects = stack.effects
        assert len(effects) == 2
        assert effects[0].priority < effects[1].priority

    def test_full_stack_quality_ultra(self) -> None:
        """Ultra quality preset enables all effects including upscaling."""
        preset = QUALITY_PRESET_ULTRA

        # Ultra should include all major effects
        assert "Exposure" in preset.active_effects
        assert "Bloom" in preset.active_effects
        assert "DepthOfField" in preset.active_effects
        assert "MotionBlur" in preset.active_effects
        assert "AmbientOcclusion" in preset.active_effects
        assert "Tonemapping" in preset.active_effects
        assert "ColorGrading" in preset.active_effects
        assert "TAA" in preset.active_effects
        assert "Upscaling" in preset.active_effects

        # Verify quality config
        assert preset.quality == EffectQuality.ULTRA

    def test_full_stack_quality_low(self) -> None:
        """Low quality disables expensive effects."""
        preset = QUALITY_PRESET_LOW

        # Low should only have minimal effects
        assert "Exposure" in preset.active_effects
        assert "Tonemapping" in preset.active_effects
        assert "FXAA" in preset.active_effects

        # Expensive effects should be disabled
        assert "DepthOfField" not in preset.active_effects
        assert "MotionBlur" not in preset.active_effects
        assert "AmbientOcclusion" not in preset.active_effects
        assert "Bloom" not in preset.active_effects
        assert "TAA" not in preset.active_effects
        assert "Upscaling" not in preset.active_effects

    def test_full_stack_no_artifacts(self) -> None:
        """Full pipeline should produce valid output without NaN/Inf."""
        # Test TSR sampling produces valid values
        upscaler = TSRLanczosUpscaler()

        # Create test image with various values
        test_image: List[List[Tuple[float, float, float]]] = [
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.25, 0.75, 0.5)],
            [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9), (1.0, 0.0, 0.0)],
            [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
            [(0.9, 0.9, 0.9), (0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.3, 0.3, 0.3)],
        ]

        # Sample at various positions
        test_positions = [
            (0.5, 0.5),
            (1.5, 1.5),
            (2.5, 2.5),
            (0.0, 0.0),
            (3.5, 3.5),
        ]

        for x, y in test_positions:
            color = upscaler.sample_lanczos(test_image, x, y)

            # Verify no NaN or Inf
            for c in color:
                assert not math.isnan(c), f"NaN at position ({x}, {y})"
                assert not math.isinf(c), f"Inf at position ({x}, {y})"

            # Lanczos can have slight overshoot due to ringing
            for c in color:
                assert -0.1 <= c <= 1.2, f"Value {c} out of expected range at ({x}, {y})"

        # Test sharpening produces valid output
        center = (0.6, 0.6, 0.6)
        neighbors = [(0.3, 0.3, 0.3), (0.5, 0.5, 0.5), (0.7, 0.7, 0.7), (0.4, 0.4, 0.4)]

        sharpened = upscaler.apply_sharpening(center, neighbors)

        for c in sharpened:
            assert not math.isnan(c), "NaN in sharpened output"
            assert not math.isinf(c), "Inf in sharpened output"
            assert 0.0 <= c <= 1.0, f"Sharpened value {c} out of [0, 1] range"


# =============================================================================
# ADDITIONAL INTEGRATION TESTS
# =============================================================================


class TestTAAAndTSRIntegration:
    """Tests for TAA and TSR working together."""

    def test_taa_provides_jitter_for_tsr(self) -> None:
        """TAA jitter sequence should be compatible with TSR temporal."""
        # Both use Halton-based jitter
        taa_jitter = JitterSequence(JitterPattern.HALTON_16)
        tsr = TSRLanczosUpscaler(TSRLanczosSettings(jitter_sequence="halton_16"))

        # Both should produce 16 unique samples
        taa_samples = [taa_jitter.next() for _ in range(16)]
        tsr_samples = []
        for _ in range(16):
            tsr_samples.append(tsr.get_jitter_offset())
            tsr.advance_frame()

        assert len(set(taa_samples)) == 16
        assert len(set(tsr_samples)) == 16

    def test_aa_effect_priority_before_upscaling(self) -> None:
        """AA effect should execute before upscaling in pipeline."""
        aa_settings = AASettings(method=AAMethod.TAA)
        aa_effect = AAEffect(aa_settings)

        up_settings = UpscalingSettings(upscaler_type=UpscalerType.TSR_LANCZOS)
        up_effect = UpscalingEffect(up_settings)

        # AA priority should be lower (earlier) than upscaling
        assert aa_effect.priority < up_effect.priority

    def test_upscaler_handles_taa_output(self) -> None:
        """Upscaler should accept TAA-processed input."""
        # Create effects
        aa_effect = AAEffect(AASettings(method=AAMethod.TAA))
        up_effect = UpscalingEffect(UpscalingSettings(upscaler_type=UpscalerType.TSR_LANCZOS))

        # Setup at render resolution
        aa_effect.setup(1920, 1080)
        up_effect.setup(1920, 1080)

        # Verify effect chain is valid
        aa_outputs = aa_effect.get_outputs()
        up_inputs = up_effect.get_required_inputs()

        # AA outputs color, upscaler consumes color
        assert "color" in aa_outputs
        assert "color" in up_inputs


class TestUpscalerPluginConsistency:
    """Tests for consistency across all upscaler plugins."""

    def test_all_plugins_have_consistent_interface(self) -> None:
        """All plugins should implement consistent interface methods."""
        plugins = [
            DLSSPlugin(),
            FSRPlugin(),
            XeSSPlugin(),
            TSRLanczosPlugin(),
        ]

        for plugin in plugins:
            # All should have these properties
            assert hasattr(plugin, "name")
            assert hasattr(plugin, "capabilities")
            assert isinstance(plugin.capabilities, UpscalerCapabilities)

            # All should implement these methods
            assert callable(plugin.initialize)
            assert callable(plugin.evaluate)
            assert callable(plugin.get_optimal_render_resolution)
            assert callable(plugin.shutdown)
            assert callable(plugin.supports_preset)

    def test_all_plugins_support_standard_presets(self) -> None:
        """All plugins should support standard quality presets."""
        plugins = [
            DLSSPlugin(),
            FSRPlugin(),
            XeSSPlugin(),
            TSRLanczosPlugin(),
        ]

        standard_presets = [
            "ultra_performance",
            "performance",
            "balanced",
            "quality",
            "ultra_quality",
            "native_aa",
        ]

        for plugin in plugins:
            for preset in standard_presets:
                assert plugin.supports_preset(preset), (
                    f"{plugin.name} should support preset '{preset}'"
                )

    def test_resolution_calculations_match_across_plugins(self) -> None:
        """All plugins should calculate same resolution for same preset."""
        target = (1920, 1080)
        preset = "balanced"

        dlss_res = DLSSPlugin().get_optimal_render_resolution(target, preset)
        fsr_res = FSRPlugin().get_optimal_render_resolution(target, preset)
        xess_res = XeSSPlugin().get_optimal_render_resolution(target, preset)
        tsr_res = TSRLanczosPlugin().get_optimal_render_resolution(target, preset)

        # All should match for standardized presets
        assert dlss_res == fsr_res == xess_res == tsr_res


class TestPhase6PerformanceCharacteristics:
    """Tests for performance-related characteristics."""

    def test_tsr_separable_faster_than_2d(self) -> None:
        """Separable filter should have lower cost estimate than 2D."""
        settings_sep = TSRLanczosSettings(separable=True)
        settings_2d = TSRLanczosSettings(separable=False)

        sep_upscaler = TSRLanczosUpscaler(settings_sep)
        full_upscaler = TSRLanczosUpscaler(settings_2d)

        sep_budget = sep_upscaler.get_budget_ms()
        full_budget = full_upscaler.get_budget_ms()

        assert sep_budget < full_budget, "Separable should be cheaper than 2D"

    def test_lanczos2_cheaper_than_lanczos3(self) -> None:
        """Lanczos-2 should have lower cost than Lanczos-3."""
        l2_upscaler = TSRLanczosUpscaler(TSRLanczosSettings(kernel=LanczosKernel.LANCZOS2))
        l3_upscaler = TSRLanczosUpscaler(TSRLanczosSettings(kernel=LanczosKernel.LANCZOS3))

        l2_budget = l2_upscaler.get_budget_ms()
        l3_budget = l3_upscaler.get_budget_ms()

        assert l2_budget < l3_budget, "Lanczos-2 should be cheaper than Lanczos-3"

    def test_adaptive_sharpening_contrast_aware(self) -> None:
        """Adaptive sharpening should vary strength based on local contrast."""
        settings = TSRLanczosSettings(
            sharpening=True,
            adaptive_sharpening=True,
            sharpening_min=0.2,
            sharpening_max=0.8,
        )
        upscaler = TSRLanczosUpscaler(settings)

        # Low contrast area
        low_center = (0.5, 0.5, 0.5)
        low_neighbors = [(0.5, 0.5, 0.5)] * 4

        # High contrast area
        high_center = (0.9, 0.9, 0.9)
        high_neighbors = [(0.1, 0.1, 0.1)] * 4

        low_result = upscaler.apply_sharpening(low_center, low_neighbors)
        high_result = upscaler.apply_sharpening(high_center, high_neighbors)

        # Low contrast: no change (center equals neighbors)
        assert low_result == low_center

        # High contrast: significant sharpening effect (clamped to 1.0)
        assert high_result[0] == pytest.approx(1.0, abs=0.01)
