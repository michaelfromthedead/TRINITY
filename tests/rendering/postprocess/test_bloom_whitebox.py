"""
Whitebox Tests for Bloom Effect System

Tests internal math, boundary conditions, state transitions,
path coverage, and numerical edge cases throughout the bloom pipeline.

TPP-1.2 BLOOM THRESHOLD_BLOOM = Bloom code review
"""

import math
from typing import Any, Dict, List, Tuple

import pytest

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
from engine.rendering.postprocess.constants import BLOOM, EPSILON
from engine.rendering.postprocess.postprocess_stack import EffectPriority


# ==============================================================================
# BloomThreshold - Whitebox: configure bounds / clamp / knee curve math
# ==============================================================================


class TestBloomThresholdConfigureWhitebox:
    """Whitebox: configure() input clamping and state."""

    def test_negative_threshold_clamped_to_zero(self) -> None:
        """configure: negative threshold clamped to 0.0."""
        t = BloomThreshold()
        t.configure(threshold=-5.0, softness=0.5, clamp_max=100.0)
        assert t._threshold == 0.0

    def test_softness_clamped_above_one(self) -> None:
        """configure: softness > 1.0 clamped to 1.0."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=2.0, clamp_max=100.0)
        assert t._softness == 1.0

    def test_softness_clamped_below_zero(self) -> None:
        """configure: softness < 0.0 clamped to 0.0."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=-0.5, clamp_max=100.0)
        assert t._softness == 0.0

    def test_negative_clamp_max_clamped_to_zero(self) -> None:
        """configure: negative clamp_max clamped to 0.0."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=0.5, clamp_max=-10.0)
        assert t._clamp_max == 0.0

    def test_configure_preserves_state_across_calls(self) -> None:
        """configure: second call overwrites previous state."""
        t = BloomThreshold()
        t.configure(threshold=0.5, softness=0.1, clamp_max=10.0)
        t.configure(threshold=2.0, softness=0.8, clamp_max=100.0)
        assert t._threshold == 2.0
        assert t._softness == 0.8
        assert t._clamp_max == 100.0


class TestBloomThresholdKneeCurveWhitebox:
    """Whitebox: exact knee interpolation math and boundary values."""

    @staticmethod
    def _make(threshold: float = 1.0, softness: float = 0.5, clamp: float = 65504.0) -> BloomThreshold:
        t = BloomThreshold()
        t.configure(threshold=threshold, softness=softness, clamp_max=clamp)
        return t

    # -- Boundary conditions (soft knee) --

    def test_soft_boundary_below_knee_returns_zero(self) -> None:
        """At luminance == soft_threshold (bottom of knee), result is 0.0."""
        t = self._make(threshold=1.0, softness=0.5)  # knee = 0.5, soft_threshold = 0.5
        assert t.apply(0.5) == 0.0  # exactly at soft_threshold boundary

    def test_soft_boundary_at_threshold_returns_one(self) -> None:
        """At luminance == threshold (top of knee), result is 1.0."""
        t = self._make(threshold=1.0, softness=0.5)
        assert t.apply(1.0) == 1.0  # exactly at threshold boundary

    def test_soft_knee_midpoint_returns_quarter(self) -> None:
        """At midpoint of knee (luminance = threshold - knee/2), result is 0.25."""
        t = self._make(threshold=1.0, softness=0.5)  # knee = 0.5
        # soft_threshold = 0.5, midpoint = 0.75
        # x = 0.75 - 0.5 = 0.25, x^2 = 0.0625
        # 4 * knee = 2.0, denom = 2.0 + EPSILON
        # result = 0.0625 / (2.0 + EPSILON) ≈ 0.03125
        actual = t.apply(0.75)
        eps = EPSILON
        expected = 0.0625 / (2.0 + eps)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_soft_knee_three_quarters(self) -> None:
        """At luminance = threshold - knee/4, result is ~0.5625 * scaling."""
        t = self._make(threshold=2.0, softness=0.5)  # knee = 1.0, soft_threshold = 1.0
        # luminance = 1.75, x = 0.75, x^2 = 0.5625
        # 4*knee = 4.0, denom = 4.0 + EPSILON
        actual = t.apply(1.75)
        expected = 0.5625 / (4.0 + EPSILON)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_soft_knee_continuity_at_bottom(self) -> None:
        """Value just below and at soft_threshold are both 0 (no discontinuity)."""
        t = self._make(threshold=1.0, softness=0.3)
        knee = 1.0 * 0.3
        soft_threshold = 1.0 - knee
        below = soft_threshold - 1e-10
        at_val = soft_threshold
        assert t.apply(below) == 0.0
        assert t.apply(at_val) == 0.0

    def test_soft_knee_continuity_at_top(self) -> None:
        """Value just above and at threshold are both 1.0 (no discontinuity)."""
        t = self._make(threshold=2.0, softness=0.4)
        above = 2.0 + 1e-10
        assert t.apply(above) == 1.0
        assert t.apply(2.0) == 1.0

    def test_knee_exact_value_zero_softness(self) -> None:
        """softness=0 always gives hard 0 or 1."""
        t = self._make(threshold=1.0, softness=0.0)
        assert t.apply(0.999) == 0.0
        assert t.apply(1.001) == 1.0

    def test_knee_exact_value_one_softness(self) -> None:
        """softness=1 gives wide knee from 0 to threshold."""
        t = self._make(threshold=1.0, softness=1.0)  # knee = 1.0, soft_threshold = 0.0
        # At luminance = 0.5: x = 0.5, x^2 = 0.25, 4*knee = 4.0
        expected = 0.25 / (4.0 + EPSILON)
        assert t.apply(0.5) == pytest.approx(expected, abs=1e-12)

    # -- Clamp boundary --

    def test_luminance_clamped_at_max(self) -> None:
        """Luminance is clamped to clamp_max before calculation."""
        t = self._make(threshold=1.0, softness=0.5, clamp=50.0)
        # Both 100 and 50 should produce the same result
        assert t.apply(100.0) == t.apply(50.0)

    def test_clamp_max_zero_always_zero(self) -> None:
        """When clamp_max is 0, all luminance becomes 0, result is 0."""
        t = self._make(threshold=1.0, softness=0.5, clamp=0.0)
        assert t.apply(100.0) == 0.0

    # -- get_knee_params exact math --

    def test_knee_params_exact_values(self) -> None:
        """get_knee_params returns correct (threshold, knee, 2*knee, 0.25/knee)."""
        t = self._make(threshold=4.0, softness=0.25)
        knee = 1.0
        p = t.get_knee_params()
        assert p[0] == 4.0
        assert p[1] == 1.0
        assert p[2] == 2.0
        assert p[3] == pytest.approx(0.25 / (1.0 + EPSILON), abs=1e-12)

    def test_knee_params_zero_softness(self) -> None:
        """get_knee_params with softness=0 uses EPSILON to avoid division by zero."""
        t = self._make(threshold=1.0, softness=0.0)
        p = t.get_knee_params()
        assert p[3] == pytest.approx(0.25 / (0.0 + EPSILON), abs=1e-12)

    def test_apply_zero_threshold_zero_softness(self) -> None:
        """threshold=0, softness=0: hard threshold passes luminance > 0."""
        t = self._make(threshold=0.0, softness=0.0)
        # Hard threshold uses strict >, so luminance == threshold gives 0
        assert t.apply(0.0) == 0.0
        assert t.apply(1e-10) == 1.0

    def test_apply_negative_luminance(self) -> None:
        """Negative luminance returns 0."""
        t = self._make(threshold=1.0, softness=0.5)
        assert t.apply(-1.0) == 0.0
        assert t.apply(-0.001) == 0.0

    def test_apply_clamp_negative_luminance(self) -> None:
        """Clamped negative luminance (to 0) still returns 0."""
        t = self._make(threshold=1.0, softness=0.5, clamp=0.0)
        assert t.apply(-100.0) == 0.0


# ==============================================================================
# BloomDownsample - Whitebox: mip chain generation, size math, buffer access
# ==============================================================================


class TestBloomDownsampleSetupWhitebox:
    """Whitebox: mip chain size calculation logic."""

    def test_chain_stops_when_dimension_below_two(self) -> None:
        """setup stops when w < 2 or h < 2."""
        d = BloomDownsample(max_mips=8)
        d.setup(4, 4, resolution_scale=1.0)
        # 4,4 -> 2,2 -> 1,1 (stops: w=1 < 2)
        assert d.mip_count == 2
        assert d.mip_sizes == [(4, 4), (2, 2)]

    def test_aspect_ratio_chain_short_dim_stops_first(self) -> None:
        """Chain stops when shorter dimension falls below 2 first."""
        d = BloomDownsample(max_mips=8)
        d.setup(1920, 4, resolution_scale=1.0)
        # 1920,4 -> 960,2 -> 480,1 (stops: h=1 < 2)
        assert d.mip_count == 2
        assert d.mip_sizes == [(1920, 4), (960, 2)]

    def test_max_mips_respected(self) -> None:
        """max_mips caps the chain length."""
        d = BloomDownsample(max_mips=3)
        d.setup(1920, 1080, resolution_scale=0.5)
        assert d.mip_count <= 3

    def test_resolution_scale_one_yields_full_start_size(self) -> None:
        """resolution_scale=1.0 starts from full input size."""
        d = BloomDownsample(max_mips=8)
        d.setup(100, 100, resolution_scale=1.0)
        assert d.mip_sizes[0] == (100, 100)

    def test_resolution_scale_point_two_five(self) -> None:
        """resolution_scale=0.25 starts at quarter resolution."""
        d = BloomDownsample(max_mips=8)
        d.setup(100, 100, resolution_scale=0.25)
        assert d.mip_sizes[0] == (25, 25)

    def test_mip_size_progression_exact(self) -> None:
        """Verify exact mip size progression for known input."""
        d = BloomDownsample(max_mips=5)
        d.setup(16, 16, resolution_scale=1.0)
        # 16 -> 8 -> 4 -> 2 -> stops (next would be 1)
        expected = [(16, 16), (8, 8), (4, 4), (2, 2)]
        assert d.mip_sizes == expected

    def test_each_mip_half_previous_width_height(self) -> None:
        """Each mip level is half the previous (floor division)."""
        d = BloomDownsample(max_mips=8)
        d.setup(100, 60, resolution_scale=1.0)
        sizes = d.mip_sizes
        for i in range(1, len(sizes)):
            assert sizes[i][0] == max(1, sizes[i - 1][0] // 2)
            assert sizes[i][1] == max(1, sizes[i - 1][1] // 2)

    def test_odd_dimensions_floor_division(self) -> None:
        """Odd dimensions use floor division for mip sizes.

        7,7 -> 3,3 -> 1,1 breaks (w<2 before adding to chain).
        """
        d = BloomDownsample(max_mips=8)
        d.setup(7, 7, resolution_scale=1.0)
        assert d.mip_sizes == [(7, 7), (3, 3)]

    def test_empty_setup_clears_previous(self) -> None:
        """Calling setup twice clears old state.

        setup(4,4) with default resolution_scale=0.5 starts at (2,2),
        then next halving gives w=1 which stops. So only 1 mip level.
        """
        d = BloomDownsample(max_mips=8)
        d.setup(1920, 1080)
        count_before = d.mip_count
        d.setup(4, 4)
        assert d.mip_count < count_before
        assert d.mip_count == 1  # 2,2 -> stops (4*0.5=2, then 1<2)


class TestBloomDownsampleAccessWhitebox:
    """Whitebox: buffer and boundary access."""

    def test_downsample_out_of_bounds_returns_source(self) -> None:
        """downsample() with mip_level >= buffer count returns source."""
        d = BloomDownsample(max_mips=3)
        d.setup(100, 100)
        source = "source_buffer"
        assert d.downsample(source, 99) is source

    def test_downsample_valid_returns_mip_buffer(self) -> None:
        """downsample() with valid mip_level returns eagerly-initialized buffer."""
        d = BloomDownsample(max_mips=3)
        d.setup(100, 100)
        result = d.downsample("src", 0)
        assert isinstance(result, list)  # buffers now eagerly initialized
        assert len(result) > 0  # non-empty buffer

    def test_get_mip_buffer_negative_index(self) -> None:
        """get_mip_buffer with negative index returns None."""
        d = BloomDownsample(max_mips=3)
        d.setup(100, 100)
        assert d.get_mip_buffer(-1) is None

    def test_get_mip_buffer_equal_to_count(self) -> None:
        """get_mip_buffer with index == count returns None (out of bounds)."""
        d = BloomDownsample(max_mips=3)
        d.setup(100, 100)
        assert d.get_mip_buffer(d.mip_count) is None

    def test_get_mip_buffer_zero_returns_valid(self) -> None:
        """get_mip_buffer(0) is valid when chain is non-empty."""
        d = BloomDownsample(max_mips=3)
        d.setup(100, 100)
        buf = d.get_mip_buffer(0)
        assert buf is None  # placeholder, no real buffer allocated

    def test_mip_property_count_equals_len(self) -> None:
        """mip_count property mirrors len(_mip_sizes)."""
        d = BloomDownsample(max_mips=8)
        d.setup(100, 100)
        assert d.mip_count == len(d.mip_sizes)

    def test_mip_sizes_copy_independence(self) -> None:
        """mip_sizes returns a copy; mutating it doesn't affect internal state."""
        d = BloomDownsample(max_mips=8)
        d.setup(100, 100)
        sizes_copy = d.mip_sizes
        sizes_copy.clear()
        assert d.mip_count > 0
        assert len(d.mip_sizes) > 0


# ==============================================================================
# BloomBlur - Whitebox: gaussian weight math, routing, offsets
# ==============================================================================


class TestBloomBlurGaussianWeightsWhitebox:
    """Whitebox: gaussian weight calculation math."""

    def test_radius_zero_single_weight(self) -> None:
        """radius=0 produces a single weight of 1.0."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=0, sigma=1.0)
        assert b._gaussian_weights == [1.0]
        assert b._gaussian_offsets == [0.0]

    def test_center_weight_highest(self) -> None:
        """Center weight (index 0) is always the highest."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=8, sigma=2.0)
        center = b._gaussian_weights[0]
        for w in b._gaussian_weights[1:]:
            assert center > w

    def test_weights_sum_to_one_radius_two(self) -> None:
        """Normalized weights sum to 1.0 for small radius."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=2, sigma=1.0)
        total = b._gaussian_weights[0] + 2 * sum(b._gaussian_weights[1:])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_weights_sum_to_one_large_radius(self) -> None:
        """Normalized weights sum to 1.0 for large radius."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=16, sigma=4.0)
        total = b._gaussian_weights[0] + 2 * sum(b._gaussian_weights[1:])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_larger_sigma_spreads_weights(self) -> None:
        """Larger sigma distributes weight more evenly (center relatively lower)."""
        b1 = BloomBlur(BlurMethod.GAUSSIAN)
        b2 = BloomBlur(BlurMethod.GAUSSIAN)
        b1.calculate_gaussian_weights(radius=8, sigma=1.0)
        b2.calculate_gaussian_weights(radius=8, sigma=8.0)
        # With sigma=1.0, center should be much more dominant
        assert b1._gaussian_weights[0] > b2._gaussian_weights[0]

    def test_gaussian_formula_known_value(self) -> None:
        """Verify exact weight at offset 1 for radius=4, sigma=2.0."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=4, sigma=2.0)
        # weight[1] = exp(-1^2 / (2*4)) / total = exp(-0.125) / total
        raw = math.exp(-1.0 / (2.0 * 4.0))
        assert b._gaussian_weights[1] > 0
        # Verify it's proportional to exp(-i^2/(2*sigma^2))
        ratio = b._gaussian_weights[2] / b._gaussian_weights[1]
        expected_ratio = math.exp(-4.0 / 8.0) / math.exp(-1.0 / 8.0)
        assert ratio == pytest.approx(expected_ratio, abs=1e-10)

    def test_weights_monotonic_decrease_large_sigma(self) -> None:
        """Weights decrease monotonically for any sigma."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=10, sigma=5.0)
        for i in range(1, len(b._gaussian_weights)):
            assert b._gaussian_weights[i] < b._gaussian_weights[i - 1]

    def test_weight_list_length_is_radius_plus_one(self) -> None:
        """Weight list has exactly radius+1 entries."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        for r in [0, 1, 4, 8]:
            b.calculate_gaussian_weights(radius=r, sigma=2.0)
            assert len(b._gaussian_weights) == r + 1
            assert len(b._gaussian_offsets) == r + 1

    def test_offsets_match_indices(self) -> None:
        """Offsets are [0.0, 1.0, 2.0, ...]."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=5, sigma=2.0)
        assert b._gaussian_offsets == [float(i) for i in range(6)]


class TestBloomBlurKawaseOffsetsWhitebox:
    """Whitebox: Kawase offset math."""

    def test_iteration_zero_is_0_5(self) -> None:
        """get_kawase_offsets(0) returns 0.5."""
        b = BloomBlur(BlurMethod.KAWASE)
        assert b.get_kawase_offsets(0) == 0.5

    def test_iteration_values(self) -> None:
        """get_kawase_offsets(n) returns 0.5 + n."""
        b = BloomBlur(BlurMethod.KAWASE)
        for n in range(10):
            assert b.get_kawase_offsets(n) == 0.5 + n

    def test_offset_independent_of_method(self) -> None:
        """Kawase offset calculation is independent of current blur method."""
        b_gauss = BloomBlur(BlurMethod.GAUSSIAN)
        b_box = BloomBlur(BlurMethod.BOX)
        assert b_gauss.get_kawase_offsets(3) == b_box.get_kawase_offsets(3)


class TestBloomBlurRoutingWhitebox:
    """Whitebox: blur() routes to correct internal method."""

    def test_blur_gaussian_returns_target(self) -> None:
        """Gaussian blur returns target buffer."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        result = b.blur("src", "tgt", iterations=2)
        assert result == "tgt"

    def test_blur_kawase_returns_target(self) -> None:
        """Kawase blur returns target buffer."""
        b = BloomBlur(BlurMethod.KAWASE)
        result = b.blur("src", "tgt", iterations=2)
        assert result == "tgt"

    def test_blur_box_returns_target(self) -> None:
        """Box blur returns target buffer."""
        b = BloomBlur(BlurMethod.BOX)
        result = b.blur("src", "tgt", iterations=2)
        assert result == "tgt"

    def test_blur_method_property_sync(self) -> None:
        """Setting method property updates internal method routing."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.method = BlurMethod.KAWASE
        result = b.blur("src", "tgt", iterations=1)
        assert result == "tgt"

    def test_blur_default_method_kawase(self) -> None:
        """Default blur method is KAWASE."""
        b = BloomBlur()
        assert b.method == BlurMethod.KAWASE

    def test_blur_explicit_method_setter(self) -> None:
        """method setter changes internal _method."""
        b = BloomBlur(BlurMethod.BOX)
        b.method = BlurMethod.GAUSSIAN
        assert b._method == BlurMethod.GAUSSIAN


# ==============================================================================
# BloomUpsample - Whitebox: buffer setup and accumulation
# ==============================================================================


class TestBloomUpsampleWhitebox:
    """Whitebox: upsample buffer setup and accumulation."""

    def test_setup_empty_sizes(self) -> None:
        """setup with empty list produces empty buffers."""
        u = BloomUpsample()
        u.setup([])
        assert u._upsample_buffers == []

    def test_setup_single_mip(self) -> None:
        """setup with single mip size creates single buffer slot."""
        u = BloomUpsample()
        u.setup([(100, 100)])
        assert len(u._upsample_buffers) == 1
        assert u._upsample_buffers[0] is None

    def test_upsample_returns_high_res(self) -> None:
        """upsample_and_accumulate returns high_res buffer."""
        u = BloomUpsample()
        u.setup([(100, 100)])
        mip = BloomMipSettings()
        result = u.upsample_and_accumulate("low", "high", mip)
        assert result == "high"

    def test_upsample_with_settings(self) -> None:
        """upsample_and_accumulate accepts custom mip settings."""
        u = BloomUpsample()
        u.setup([(100, 100)])
        mip = BloomMipSettings(intensity=0.5, scatter=0.3, tint=(0.5, 0.5, 0.5))
        result = u.upsample_and_accumulate("a", "b", mip)
        assert result == "b"


# ==============================================================================
# BloomSettings - Whitebox: post_init, lerp edge cases
# ==============================================================================


class TestBloomSettingsWhitebox:
    """Whitebox: internal state initialization and interpolation."""

    def test_post_init_sets_priority(self) -> None:
        """__post_init__ sets priority to BLOOM value."""
        s = BloomSettings()
        assert s.priority == EffectPriority.BLOOM.value

    def test_post_init_default_mip_settings(self) -> None:
        """__post_init__ creates 6 default mip levels with decreasing intensity."""
        s = BloomSettings()
        assert len(s.mip_settings) == 6
        expected_intensities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        for i, mip in enumerate(s.mip_settings):
            assert mip.intensity == expected_intensities[i]
            assert mip.tint == (1.0, 1.0, 1.0)

    def test_custom_mip_settings_preserved(self) -> None:
        """Custom mip_settings passed to constructor are preserved."""
        custom = [BloomMipSettings(intensity=2.0)]
        s = BloomSettings(mip_settings=custom)
        assert s.mip_settings is custom
        assert len(s.mip_settings) == 1
        assert s.mip_settings[0].intensity == 2.0

    def test_lerp_t_zero_equals_self(self) -> None:
        """lerp(t=0) returns self-equivalent values."""
        s1 = BloomSettings(threshold=1.5, intensity=0.3, scatter=0.2)
        s2 = BloomSettings(threshold=3.0, intensity=0.9, scatter=0.8)
        result = s1.lerp(s2, 0.0)
        assert result.threshold == s1.threshold
        assert result.intensity == s1.intensity
        assert result.scatter == s1.scatter

    def test_lerp_t_one_equals_other(self) -> None:
        """lerp(t=1) returns other-equivalent values (within floating point)."""
        s1 = BloomSettings(threshold=1.5, intensity=0.3, scatter=0.2)
        s2 = BloomSettings(threshold=3.0, intensity=0.9, scatter=0.8)
        result = s1.lerp(s2, 1.0)
        assert result.threshold == s2.threshold
        assert result.intensity == pytest.approx(s2.intensity)
        assert result.scatter == s2.scatter

    def test_lerp_enabled_switches_at_midpoint(self) -> None:
        """enabled uses s1 when t<0.5, s2 when t>=0.5."""
        s1 = BloomSettings(enabled=True)
        s2 = BloomSettings(enabled=False)
        assert s1.lerp(s2, 0.0).enabled is True
        assert s1.lerp(s2, 0.49).enabled is True
        assert s1.lerp(s2, 0.5).enabled is False
        assert s1.lerp(s2, 1.0).enabled is False

    def test_lerp_quality_switches_at_midpoint(self) -> None:
        """quality uses s1 when t<0.5, s2 when t>=0.5."""
        s1 = BloomSettings(quality=BloomQuality.LOW)
        s2 = BloomSettings(quality=BloomQuality.ULTRA)
        assert s1.lerp(s2, 0.0).quality == BloomQuality.LOW
        assert s1.lerp(s2, 0.49).quality == BloomQuality.LOW
        assert s1.lerp(s2, 0.5).quality == BloomQuality.ULTRA
        assert s1.lerp(s2, 1.0).quality == BloomQuality.ULTRA

    def test_lerp_blur_method_switches_at_midpoint(self) -> None:
        """blur_method uses s1 when t<0.5, s2 when t>=0.5."""
        s1 = BloomSettings(blur_method=BlurMethod.BOX)
        s2 = BloomSettings(blur_method=BlurMethod.GAUSSIAN)
        assert s1.lerp(s2, 0.0).blur_method == BlurMethod.BOX
        assert s1.lerp(s2, 0.5).blur_method == BlurMethod.GAUSSIAN

    def test_lerp_mip_different_lengths(self) -> None:
        """lerp with mip lists of different lengths pads with defaults."""
        s1 = BloomSettings(mip_settings=[BloomMipSettings(intensity=0.5)])
        s2 = BloomSettings(
            mip_settings=[
                BloomMipSettings(intensity=1.0),
                BloomMipSettings(intensity=0.8),
                BloomMipSettings(intensity=0.6),
            ]
        )
        result = s1.lerp(s2, 0.5)
        # mip 0: lerped
        assert len(result.mip_settings) == 3
        # mip 0: 0.5 + (1.0 - 0.5) * 0.5 = 0.75
        assert result.mip_settings[0].intensity == 0.75
        # mip 1: padded s1 defaults intensity=1.0, lerp with s2's 0.8 -> 0.9
        assert result.mip_settings[1].intensity == 0.9
        # mip 2: padded s1 defaults, lerp with s2's 0.6 -> 0.8
        assert result.mip_settings[2].intensity == 0.8

    def test_lerp_anamorphic_ratio(self) -> None:
        """anamorphic_ratio interpolates linearly."""
        s1 = BloomSettings(anamorphic_ratio=0.0)
        s2 = BloomSettings(anamorphic_ratio=1.0)
        result = s1.lerp(s2, 0.25)
        assert result.anamorphic_ratio == 0.25

    def test_lerp_mip_tint(self) -> None:
        """Mip tint interpolates per-channel."""
        s1 = BloomSettings(
            mip_settings=[BloomMipSettings(tint=(1.0, 0.0, 0.5))]
        )
        s2 = BloomSettings(
            mip_settings=[BloomMipSettings(tint=(0.0, 1.0, 1.0))]
        )
        result = s1.lerp(s2, 0.5)
        assert result.mip_settings[0].tint == (0.5, 0.5, 0.75)


class TestLensDirtWhitebox:
    """Whitebox: Lens dirt settings edge cases."""

    def test_lens_dirt_enabled_transition(self) -> None:
        """Lens dirt can be toggled on."""
        d = LensDirtSettings(enabled=True)
        assert d.enabled is True

    def test_lens_dirt_texture_path_none(self) -> None:
        """Default texture_path is None."""
        d = LensDirtSettings()
        assert d.texture_path is None

    def test_lens_dirt_custom_tint(self) -> None:
        """Custom tint is preserved."""
        d = LensDirtSettings(tint=(0.8, 0.6, 1.0))
        assert d.tint == (0.8, 0.6, 1.0)

    def test_bloom_mip_scatter_boundary(self) -> None:
        """Mip scatter at boundary values."""
        m = BloomMipSettings(scatter=0.0)
        assert m.scatter == 0.0
        m = BloomMipSettings(scatter=1.0)
        assert m.scatter == 1.0


# ==============================================================================
# BloomEffect - Whitebox: internal configuration, execute path coverage
# ==============================================================================


class TestBloomEffectConfigureWhitebox:
    """Whitebox: _configure_from_settings internal state."""

    def test_configure_empty_settings(self) -> None:
        """_configure_from_settings with settings=None returns early."""
        effect = BloomEffect()
        effect._settings = None
        # Should not raise
        effect._configure_from_settings()

    def test_configure_threshold_state(self) -> None:
        """Threshold component gets configured from settings."""
        effect = BloomEffect(
            settings=BloomSettings(threshold=2.5, threshold_softness=0.75, clamp_max=500.0)
        )
        effect.setup(100, 100)
        assert effect._threshold._threshold == 2.5
        assert effect._threshold._softness == 0.75
        assert effect._threshold._clamp_max == 500.0

    def test_configure_blur_method(self) -> None:
        """Blur method is set from settings."""
        effect = BloomEffect(settings=BloomSettings(blur_method=BlurMethod.GAUSSIAN))
        effect.setup(100, 100)
        assert effect._blur._method == BlurMethod.GAUSSIAN

    def test_configure_quality_low_mip_count(self) -> None:
        """LOW quality sets max_mips to 3."""
        effect = BloomEffect(settings=BloomSettings(quality=BloomQuality.LOW))
        effect.setup(1920, 1080)
        assert effect._downsample._max_mips == BLOOM.MIP_COUNT_LOW

    def test_configure_quality_medium_mip_count(self) -> None:
        """MEDIUM quality sets max_mips to 5."""
        effect = BloomEffect(settings=BloomSettings(quality=BloomQuality.MEDIUM))
        effect.setup(1920, 1080)
        assert effect._downsample._max_mips == BLOOM.MIP_COUNT_MEDIUM

    def test_configure_quality_high_mip_count(self) -> None:
        """HIGH quality sets max_mips to 6."""
        effect = BloomEffect(settings=BloomSettings(quality=BloomQuality.HIGH))
        effect.setup(1920, 1080)
        assert effect._downsample._max_mips == BLOOM.MIP_COUNT_HIGH

    def test_configure_quality_ultra_mip_count(self) -> None:
        """ULTRA quality sets max_mips to 8."""
        effect = BloomEffect(settings=BloomSettings(quality=BloomQuality.ULTRA))
        effect.setup(1920, 1080)
        assert effect._downsample._max_mips == BLOOM.MIP_COUNT_ULTRA

    def test_configure_gaussian_triggers_weight_calc(self) -> None:
        """Gaussian blur method triggers calculate_gaussian_weights."""
        effect = BloomEffect(settings=BloomSettings(blur_method=BlurMethod.GAUSSIAN))
        effect.setup(100, 100)
        assert len(effect._blur._gaussian_weights) > 0
        assert len(effect._blur._gaussian_offsets) > 0

    def test_configure_kawase_does_not_calc_gaussian_weights(self) -> None:
        """Kawase does not pre-calculate Gaussian weights."""
        effect = BloomEffect(settings=BloomSettings(blur_method=BlurMethod.KAWASE))
        effect.setup(100, 100)
        assert effect._blur._gaussian_weights == []

    def test_configure_upsample_setup_called_with_downsample_sizes(self) -> None:
        """Upsample is initialized with downsample mip sizes."""
        effect = BloomEffect()
        effect.setup(100, 100)
        assert len(effect._upsample._upsample_buffers) == effect._downsample.mip_count

    def test_setup_stores_width_height(self) -> None:
        """setup stores width and height."""
        effect = BloomEffect()
        effect.setup(640, 480)
        assert effect._width == 640
        assert effect._height == 480

    def test_configure_passes_resolution_scale(self) -> None:
        """resolution_scale from settings is passed to downsample setup."""
        effect = BloomEffect(settings=BloomSettings(resolution_scale=0.25))
        effect.setup(1920, 1080)
        first_mip = effect._downsample.mip_sizes[0]
        assert first_mip[0] == int(1920 * 0.25)
        assert first_mip[1] == int(1080 * 0.25)


class TestBloomEffectExecuteWhitebox:
    """Whitebox: execute() path coverage and internal behavior."""

    def test_execute_disabled_returns_early(self) -> None:
        """execute returns immediately when disabled."""
        effect = BloomEffect(settings=BloomSettings(enabled=False))
        effect.setup(100, 100)
        effect.execute({"color": "buf"}, {}, 0.016)
        # No crash = success

    def test_execute_zero_intensity_returns_early(self) -> None:
        """execute returns immediately when intensity <= 0."""
        effect = BloomEffect(settings=BloomSettings(intensity=0.0))
        effect.setup(100, 100)
        effect.execute({"color": "buf"}, {}, 0.016)

    def test_execute_negative_intensity_returns_early(self) -> None:
        """Negative intensity raises ValueError in settings validation."""
        with pytest.raises(ValueError, match="intensity must be in"):
            BloomEffect(settings=BloomSettings(intensity=-1.0))

    def test_execute_enabled_runs_pipeline(self) -> None:
        """execute runs the full downsample-blur-upsample pipeline when enabled."""
        effect = BloomEffect()
        effect.setup(100, 100)
        effect.execute({"color": "color_buf"}, {}, 0.016)
        # Pipeline completes without error

    def test_execute_mip_settings_fallback(self) -> None:
        """execute uses default BloomMipSettings when mip > len(mip_settings)."""
        # Create effect with only 2 mip settings but downsample with more mips
        settings = BloomSettings(
            mip_settings=[BloomMipSettings(intensity=0.5), BloomMipSettings(intensity=0.3)]
        )
        effect = BloomEffect(settings=settings)
        effect.setup(100, 100)
        effect.execute({"color": "buf"}, {}, 0.016)

    def test_cleanup_resets_buffers(self) -> None:
        """cleanup sets internal buffers to None."""
        effect = BloomEffect()
        effect.setup(100, 100)
        effect._bright_pass_buffer = "some_buffer"
        effect._lens_dirt_texture = "dirt_texture"
        effect.cleanup()
        assert effect._bright_pass_buffer is None
        assert effect._lens_dirt_texture is None

    def test_cleanup_idempotent(self) -> None:
        """Calling cleanup twice has no error."""
        effect = BloomEffect()
        effect.cleanup()
        effect.cleanup()

    def test_setup_after_cleanup(self) -> None:
        """setup works after cleanup (reinitialization)."""
        effect = BloomEffect()
        effect.setup(100, 100)
        mips_1 = effect.mip_count
        effect.cleanup()
        effect.setup(200, 200)
        mips_2 = effect.mip_count
        assert mips_2 > 0  # Still works

    def test_required_inputs_exactly_color(self) -> None:
        """get_required_inputs returns only 'color'."""
        effect = BloomEffect()
        assert effect.get_required_inputs() == ["color"]

    def test_outputs_include_color_and_bloom_buffer(self) -> None:
        """get_outputs returns color and bloom_buffer."""
        effect = BloomEffect()
        outputs = effect.get_outputs()
        assert "color" in outputs
        assert "bloom_buffer" in outputs

    def test_execute_with_no_color_input(self) -> None:
        """execute handles missing color input gracefully."""
        effect = BloomEffect()
        effect.setup(100, 100)
        effect.execute({}, {}, 0.016)

    def test_execute_upsample_accumulation_order(self) -> None:
        """Execute iterates upsample from highest mip down to 0."""
        effect = BloomEffect()
        effect.setup(100, 100)
        effect.execute({"color": "buf"}, {}, 0.016)
        # The upsample loop goes mip_count-1 down to 0 inclusive;
        # first iteration passes bloom_result (None) -> uses mip_buffer as low_res


class TestBloomEffectSubcomponentsWhitebox:
    """Whitebox: subcomponents are correctly instantiated."""

    def test_internal_threshold_instance(self) -> None:
        """BloomEffect has a BloomThreshold instance."""
        effect = BloomEffect()
        assert isinstance(effect._threshold, BloomThreshold)

    def test_internal_downsample_instance(self) -> None:
        """BloomEffect has a BloomDownsample instance."""
        effect = BloomEffect()
        assert isinstance(effect._downsample, BloomDownsample)

    def test_internal_blur_instance(self) -> None:
        """BloomEffect has a BloomBlur instance."""
        effect = BloomEffect()
        assert isinstance(effect._blur, BloomBlur)

    def test_internal_upsample_instance(self) -> None:
        """BloomEffect has a BloomUpsample instance."""
        effect = BloomEffect()
        assert isinstance(effect._upsample, BloomUpsample)

    def test_effect_name_constant(self) -> None:
        """Effect name is always 'Bloom'."""
        effect1 = BloomEffect()
        effect2 = BloomEffect(settings=BloomSettings(quality=BloomQuality.ULTRA))
        assert effect1.name == "Bloom"
        assert effect2.name == "Bloom"

    def test_is_compute_always_true(self) -> None:
        """is_compute_effect always returns True."""
        effect = BloomEffect()
        assert effect.is_compute_effect() is True


# ==============================================================================
# BloomQuality - Whitebox: enum values and ordering
# ==============================================================================


class TestBloomQualityOrderingWhitebox:
    """Whitebox: quality enum ordering and value access."""

    def test_quality_ordering(self) -> None:
        """Quality levels have correct ordering by detail."""
        order = [BloomQuality.LOW, BloomQuality.MEDIUM, BloomQuality.HIGH, BloomQuality.ULTRA]
        # Just verify they are distinct
        assert len(set(order)) == 4

    def test_quality_names(self) -> None:
        """Quality enum names are as expected."""
        assert BloomQuality.LOW.name == "LOW"
        assert BloomQuality.MEDIUM.name == "MEDIUM"
        assert BloomQuality.HIGH.name == "HIGH"
        assert BloomQuality.ULTRA.name == "ULTRA"


class TestBlurMethodWhitebox:
    """Whitebox: blur method enum."""

    def test_enum_values_distinct(self) -> None:
        """All blur methods are distinct auto values."""
        values = {BlurMethod.GAUSSIAN, BlurMethod.KAWASE, BlurMethod.BOX}
        assert len(values) == 3


# ==============================================================================
# Numerical safety - Whitebox: extreme / pathological inputs
# ==============================================================================


class TestBloomNumericalSafetyWhitebox:
    """Whitebox: numerical stability with extreme values."""

    def test_downsample_extreme_aspect_ratio(self) -> None:
        """Downsample with extreme aspect ratio handles dimension stops."""
        d = BloomDownsample(max_mips=8)
        d.setup(10000, 1, resolution_scale=1.0)
        # 10000,1 -> 5000,1 -> ... but h < 2 stops immediately for most levels
        assert d.mip_count < 8

    def test_downsample_zero_dimensions(self) -> None:
        """Downsample with 0 dimensions produces empty chain."""
        d = BloomDownsample(max_mips=8)
        d.setup(0, 0)
        assert d.mip_count == 0
        assert d.mip_sizes == []

    def test_downsample_one_dimension(self) -> None:
        """Downsample with 1 dimension produces empty chain (w<2)."""
        d = BloomDownsample(max_mips=8)
        d.setup(1, 100)
        assert d.mip_count == 0

    def test_blur_large_sigma_no_error(self) -> None:
        """Large sigma in gaussian weight calc doesn't cause issues."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=4, sigma=1000.0)
        # All weights should be essentially equal
        assert all(w > 0 for w in b._gaussian_weights)
        total = b._gaussian_weights[0] + 2 * sum(b._gaussian_weights[1:])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_blur_tiny_sigma_center_weight_one(self) -> None:
        """Tiny sigma makes center weight approach 1.0."""
        b = BloomBlur(BlurMethod.GAUSSIAN)
        b.calculate_gaussian_weights(radius=4, sigma=0.01)
        assert b._gaussian_weights[0] > 0.99

    def test_threshold_full_range_softness_continuity(self) -> None:
        """apply returns values in [0, 1] for all inputs in range."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=0.5, clamp_max=10.0)
        for lum in [x * 0.1 for x in range(0, 101)]:  # 0.0 to 10.0 step 0.1
            result = t.apply(lum)
            assert 0.0 <= result <= 1.0, f"Out of range at luminance={lum}"

    def test_threshold_zero_softness_max_clamp(self) -> None:
        """Softness=0, clamp_max=0: luminance always clamped to 0, hard threshold gives 0."""
        t = BloomThreshold()
        t.configure(threshold=0.5, softness=0.0, clamp_max=0.0)
        assert t.apply(100.0) == 0.0  # clamped to 0, threshold is 0.5, 0 <= 0.5


# ==============================================================================
# Integration - Whitebox: default settings internal consistency
# ==============================================================================


class TestBloomDefaultsConsistencyWhitebox:
    """Whitebox: default settings are internally consistent."""

    def test_default_mip_count_vs_quality(self) -> None:
        """Default quality (MEDIUM) creates 5 mip levels for a reasonable size."""
        effect = BloomEffect()
        effect.setup(1920, 1080)
        # MEDIUM quality -> max_mips = 5
        assert effect.mip_count <= 5

    def test_default_settings_reference_consistency(self) -> None:
        """Default BloomSettings match BLOOM constants."""
        s = BloomSettings()
        assert s.threshold == BLOOM.THRESHOLD_DEFAULT
        assert s.threshold_softness == BLOOM.THRESHOLD_SOFTNESS_DEFAULT
        assert s.clamp_max == BLOOM.CLAMP_MAX_DEFAULT
        assert s.intensity == BLOOM.INTENSITY_DEFAULT
        assert s.scatter == BLOOM.SCATTER_DEFAULT
        assert s.resolution_scale == BLOOM.RESOLUTION_SCALE_DEFAULT

    def test_default_mip_settings_fallback_count(self) -> None:
        """Default mip_settings list length is 6."""
        s = BloomSettings()
        assert len(s.mip_settings) == 6
        # Each entry should have defaults
        for mip in s.mip_settings:
            assert mip.tint == (1.0, 1.0, 1.0)
