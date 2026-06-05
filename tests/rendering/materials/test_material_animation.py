"""Tests for Material Animation System (T-MAT-5.5).

This module tests:
- WGSL syntax validation for time.wgsl
- AnimationCurve wave functions
- TimeUniforms serialization
- TimeContext accessors
- MaterialAnimator state management
- Reference value matching for animation functions
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import Tuple

import pytest

from trinity.materials.animation import (
    # WGSL access
    get_time_wgsl,
    TIME_WGSL,
    # Core types
    TimeUniforms,
    TimeContext,
    # Animation curves
    AnimationCurve,
    WaveType,
    EasingType,
    # Animator
    MaterialAnimator,
    AnimationState,
    AnimatedParameter,
    # Reference values
    ANIMATION_REFERENCE_VALUES,
    # Constants
    TAU,
    PI,
)


# =============================================================================
# WGSL Syntax Validation Tests
# =============================================================================


class TestWGSLSyntax:
    """Test WGSL source code validity."""

    def test_time_wgsl_loads(self) -> None:
        """Test that time.wgsl can be loaded."""
        wgsl = get_time_wgsl()
        assert len(wgsl) > 0
        assert "TimeUniforms" in wgsl
        assert "elapsed_seconds" in wgsl
        assert "delta_time" in wgsl
        assert "frame_count" in wgsl

    def test_time_wgsl_file_exists(self) -> None:
        """Test that the WGSL file exists at expected path."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "time.wgsl"
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_time_wgsl_has_required_functions(self) -> None:
        """Test that all required animation functions are present."""
        wgsl = get_time_wgsl()
        required_functions = [
            "fn sin_wave",
            "fn cos_wave",
            "fn sin_wave_01",
            "fn cos_wave_01",
            "fn sawtooth",
            "fn triangle_wave",
            "fn pulse",
            "fn animate_uv_scroll",
            "fn animate_uv_rotate",
            "fn animate_color_pulse",
            "fn animate_emission_flicker",
            "fn animate_emission_pulse",
        ]
        for func in required_functions:
            assert func in wgsl, f"Missing required function: {func}"

    def test_time_wgsl_has_struct(self) -> None:
        """Test that TimeUniforms struct is defined correctly."""
        wgsl = get_time_wgsl()
        assert "struct TimeUniforms" in wgsl
        assert "elapsed_seconds: f32" in wgsl
        assert "delta_time: f32" in wgsl
        assert "frame_count: u32" in wgsl

    def test_time_wgsl_syntax_patterns(self) -> None:
        """Test basic WGSL syntax patterns."""
        wgsl = get_time_wgsl()

        # Check function declarations
        fn_pattern = r"fn\s+\w+\([^)]*\)\s*->\s*\w+"
        assert re.search(fn_pattern, wgsl), "No valid function declarations found"

        # Check for proper type annotations
        assert "f32" in wgsl
        assert "vec2<f32>" in wgsl
        assert "vec3<f32>" in wgsl

    def test_time_wgsl_no_syntax_errors(self) -> None:
        """Test that WGSL has no obvious syntax errors."""
        wgsl = get_time_wgsl()

        # Check balanced braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check balanced parentheses
        open_parens = wgsl.count("(")
        close_parens = wgsl.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"

    def test_time_wgsl_module_constant(self) -> None:
        """Test that TIME_WGSL module constant works."""
        wgsl_str = str(TIME_WGSL)
        assert len(wgsl_str) > 0
        assert "TimeUniforms" in wgsl_str


# =============================================================================
# TimeUniforms Tests
# =============================================================================


class TestTimeUniforms:
    """Test TimeUniforms struct."""

    def test_default_values(self) -> None:
        """Test default initialization."""
        uniforms = TimeUniforms()
        assert uniforms.elapsed_seconds == 0.0
        assert uniforms.delta_time == 0.0
        assert uniforms.frame_count == 0

    def test_update(self) -> None:
        """Test time update."""
        uniforms = TimeUniforms()
        uniforms.update(0.016)  # ~60 FPS frame

        assert abs(uniforms.elapsed_seconds - 0.016) < 1e-6
        assert abs(uniforms.delta_time - 0.016) < 1e-6
        assert uniforms.frame_count == 1

        uniforms.update(0.016)
        assert abs(uniforms.elapsed_seconds - 0.032) < 1e-6
        assert uniforms.frame_count == 2

    def test_reset(self) -> None:
        """Test reset to initial state."""
        uniforms = TimeUniforms()
        uniforms.update(1.0)
        uniforms.update(1.0)

        uniforms.reset()

        assert uniforms.elapsed_seconds == 0.0
        assert uniforms.delta_time == 0.0
        assert uniforms.frame_count == 0

    def test_serialization(self) -> None:
        """Test GPU buffer serialization."""
        uniforms = TimeUniforms(elapsed_seconds=1.5, delta_time=0.016, frame_count=100)
        data = uniforms.to_bytes()

        assert len(data) == 16  # 16-byte aligned

        # Deserialize and verify
        restored = TimeUniforms.from_bytes(data)
        assert abs(restored.elapsed_seconds - 1.5) < 1e-6
        assert abs(restored.delta_time - 0.016) < 1e-6
        assert restored.frame_count == 100

    def test_frame_count_wrapping(self) -> None:
        """Test frame count wraps at 2^32."""
        uniforms = TimeUniforms(frame_count=2**32 - 1)
        uniforms.update(0.016)
        assert uniforms.frame_count == 0  # Wrapped


# =============================================================================
# TimeContext Tests
# =============================================================================


class TestTimeContext:
    """Test TimeContext accessor."""

    def test_time_accessor(self) -> None:
        """Test time() accessor."""
        uniforms = TimeUniforms(elapsed_seconds=5.5)
        ctx = TimeContext(_uniforms=uniforms)

        assert abs(ctx.time() - 5.5) < 1e-6

    def test_delta_accessor(self) -> None:
        """Test delta() accessor."""
        uniforms = TimeUniforms(delta_time=0.016)
        ctx = TimeContext(_uniforms=uniforms)

        assert abs(ctx.delta() - 0.016) < 1e-6

    def test_frame_accessor(self) -> None:
        """Test frame() accessor."""
        uniforms = TimeUniforms(frame_count=42)
        ctx = TimeContext(_uniforms=uniforms)

        assert ctx.frame() == 42

    def test_seconds_mod(self) -> None:
        """Test seconds_mod() for looping animations."""
        uniforms = TimeUniforms(elapsed_seconds=5.5)
        ctx = TimeContext(_uniforms=uniforms)

        assert abs(ctx.seconds_mod(2.0) - 1.5) < 1e-6
        assert abs(ctx.seconds_mod(3.0) - 2.5) < 1e-6
        assert ctx.seconds_mod(0.0) == 0.0  # Edge case

    def test_phase(self) -> None:
        """Test phase() for normalized loop position."""
        uniforms = TimeUniforms(elapsed_seconds=1.5)
        ctx = TimeContext(_uniforms=uniforms)

        assert abs(ctx.phase(2.0) - 0.75) < 1e-6
        assert abs(ctx.phase(1.0) - 0.5) < 1e-6
        assert ctx.phase(0.0) == 0.0  # Edge case


# =============================================================================
# AnimationCurve Wave Function Tests
# =============================================================================


class TestSinWave:
    """Test sine wave functions."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["sin_wave"])
    def test_sin_wave_reference_values(self, ref: dict) -> None:
        """Test sin_wave matches reference values."""
        result = AnimationCurve.sin_wave(ref["t"], ref["frequency"], ref["phase"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"sin_wave(t={ref['t']}, f={ref['frequency']}) = {result}, "
            f"expected {ref['expected']}"
        )

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["sin_wave_01"])
    def test_sin_wave_01_reference_values(self, ref: dict) -> None:
        """Test sin_wave_01 matches reference values."""
        result = AnimationCurve.sin_wave_01(ref["t"], ref["frequency"], ref["phase"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"sin_wave_01(t={ref['t']}, f={ref['frequency']}) = {result}, "
            f"expected {ref['expected']}"
        )

    def test_sin_wave_range(self) -> None:
        """Test sin_wave output is in [-1, 1]."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            for freq in [0.5, 1.0, 2.0, 5.0]:
                result = AnimationCurve.sin_wave(t, freq)
                assert -1.0 <= result <= 1.0

    def test_sin_wave_01_range(self) -> None:
        """Test sin_wave_01 output is in [0, 1]."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            for freq in [0.5, 1.0, 2.0, 5.0]:
                result = AnimationCurve.sin_wave_01(t, freq)
                assert 0.0 <= result <= 1.0

    def test_sin_wave_periodicity(self) -> None:
        """Test sin_wave is periodic."""
        frequency = 2.0
        period = 1.0 / frequency

        for t in [0.1, 0.25, 0.37]:
            v1 = AnimationCurve.sin_wave(t, frequency)
            v2 = AnimationCurve.sin_wave(t + period, frequency)
            assert abs(v1 - v2) < 1e-6


class TestCosWave:
    """Test cosine wave functions."""

    def test_cos_wave_at_zero(self) -> None:
        """Test cos_wave starts at 1."""
        result = AnimationCurve.cos_wave(0.0, 1.0)
        assert abs(result - 1.0) < 1e-6

    def test_cos_wave_phase_shift(self) -> None:
        """Test cos_wave is 90 degrees out of phase with sin_wave."""
        for t in [0.0, 0.1, 0.25, 0.5]:
            sin_val = AnimationCurve.sin_wave(t + 0.25, 1.0)
            cos_val = AnimationCurve.cos_wave(t, 1.0)
            assert abs(sin_val - cos_val) < 1e-6


class TestSawtooth:
    """Test sawtooth wave functions."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["sawtooth"])
    def test_sawtooth_reference_values(self, ref: dict) -> None:
        """Test sawtooth matches reference values."""
        result = AnimationCurve.sawtooth(ref["t"], ref["period"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"sawtooth(t={ref['t']}, period={ref['period']}) = {result}, "
            f"expected {ref['expected']}"
        )

    def test_sawtooth_range(self) -> None:
        """Test sawtooth output is in [0, 1)."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
            result = AnimationCurve.sawtooth(t, 1.0)
            assert 0.0 <= result < 1.0

    def test_sawtooth_reverse(self) -> None:
        """Test sawtooth_reverse is complement of sawtooth."""
        for t in [0.0, 0.25, 0.5, 0.75]:
            saw = AnimationCurve.sawtooth(t, 1.0)
            rev = AnimationCurve.sawtooth_reverse(t, 1.0)
            assert abs(saw + rev - 1.0) < 1e-6

    def test_sawtooth_zero_period(self) -> None:
        """Test sawtooth handles zero period gracefully."""
        result = AnimationCurve.sawtooth(0.5, 0.0)
        assert result == 0.0


class TestTriangleWave:
    """Test triangle wave function."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["triangle_wave"])
    def test_triangle_wave_reference_values(self, ref: dict) -> None:
        """Test triangle_wave matches reference values."""
        result = AnimationCurve.triangle_wave(ref["t"], ref["period"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"triangle_wave(t={ref['t']}, period={ref['period']}) = {result}, "
            f"expected {ref['expected']}"
        )

    def test_triangle_wave_range(self) -> None:
        """Test triangle_wave output is in [0, 1]."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
            result = AnimationCurve.triangle_wave(t, 1.0)
            assert 0.0 <= result <= 1.0

    def test_triangle_wave_symmetry(self) -> None:
        """Test triangle wave is symmetric around peak."""
        v1 = AnimationCurve.triangle_wave(0.25, 1.0)
        v2 = AnimationCurve.triangle_wave(0.75, 1.0)
        assert abs(v1 - v2) < 1e-6


class TestPulse:
    """Test pulse wave functions."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["pulse"])
    def test_pulse_reference_values(self, ref: dict) -> None:
        """Test pulse matches reference values."""
        result = AnimationCurve.pulse(ref["t"], ref["period"], ref["duty"])
        assert abs(result - ref["expected"]) < ref["tolerance"], (
            f"pulse(t={ref['t']}, duty={ref['duty']}) = {result}, "
            f"expected {ref['expected']}"
        )

    def test_pulse_binary_output(self) -> None:
        """Test pulse outputs only 0 or 1."""
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
            result = AnimationCurve.pulse(t, 1.0, 0.5)
            assert result in [0.0, 1.0]

    def test_pulse_duty_cycle(self) -> None:
        """Test pulse respects duty cycle."""
        # 25% duty - should be on for first quarter
        assert AnimationCurve.pulse(0.1, 1.0, 0.25) == 1.0
        assert AnimationCurve.pulse(0.3, 1.0, 0.25) == 0.0
        assert AnimationCurve.pulse(0.7, 1.0, 0.25) == 0.0

    def test_smooth_pulse(self) -> None:
        """Test smooth_pulse transitions smoothly."""
        v1 = AnimationCurve.smooth_pulse(0.0, 1.0, 0.1, 0.1)
        v2 = AnimationCurve.smooth_pulse(0.05, 1.0, 0.1, 0.1)
        v3 = AnimationCurve.smooth_pulse(0.5, 1.0, 0.1, 0.1)

        assert v1 < v2  # Ramping up
        assert abs(v3 - 1.0) < 1e-6  # Fully on


# =============================================================================
# Easing Function Tests
# =============================================================================


class TestEasingFunctions:
    """Test easing functions."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["ease_in_out"])
    def test_ease_in_out_reference_values(self, ref: dict) -> None:
        """Test ease_in_out matches reference values."""
        result = AnimationCurve.ease_in_out(ref["t"])
        assert abs(result - ref["expected"]) < ref["tolerance"]

    def test_ease_in_out_endpoints(self) -> None:
        """Test easing function endpoints."""
        assert abs(AnimationCurve.ease_in_out(0.0) - 0.0) < 1e-6
        assert abs(AnimationCurve.ease_in_out(1.0) - 1.0) < 1e-6

    def test_ease_in_out_monotonic(self) -> None:
        """Test ease_in_out is monotonically increasing."""
        prev = 0.0
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            curr = AnimationCurve.ease_in_out(t)
            assert curr >= prev
            prev = curr

    def test_ease_in(self) -> None:
        """Test ease_in accelerates."""
        slow_start = AnimationCurve.ease_in(0.1)
        mid = AnimationCurve.ease_in(0.5)
        # For ease_in (quadratic), f(0.1) < 0.1 and f(0.5) < 0.5
        assert slow_start < 0.1
        assert mid < 0.5

    def test_ease_out(self) -> None:
        """Test ease_out decelerates."""
        mid = AnimationCurve.ease_out(0.5)
        fast_start = AnimationCurve.ease_out(0.1)
        # For ease_out (quadratic), values are higher than linear
        assert fast_start > 0.1
        assert mid > 0.5

    def test_ease_elastic(self) -> None:
        """Test elastic easing has overshoot."""
        result = AnimationCurve.ease_elastic(0.5)
        # Elastic easing oscillates past 1.0
        assert result > 0.0
        # At t=1.0 should converge to 1.0
        assert abs(AnimationCurve.ease_elastic(1.0) - 1.0) < 1e-6

    def test_ease_bounce(self) -> None:
        """Test bounce easing."""
        result = AnimationCurve.ease_bounce(1.0)
        assert abs(result - 1.0) < 1e-6

        # Mid-range should be less than 1
        mid = AnimationCurve.ease_bounce(0.5)
        assert mid < 1.0


# =============================================================================
# Noise Animation Tests
# =============================================================================


class TestNoiseAnimation:
    """Test noise-based animation."""

    def test_noise_anim_range(self) -> None:
        """Test noise_anim output is in [0, 1]."""
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            result = AnimationCurve.noise_anim(t, 1.0, 1)
            assert 0.0 <= result <= 1.0

    def test_noise_anim_octaves(self) -> None:
        """Test noise_anim with multiple octaves."""
        single = AnimationCurve.noise_anim(0.5, 1.0, 1)
        multi = AnimationCurve.noise_anim(0.5, 1.0, 4)

        # Both should be in valid range
        assert 0.0 <= single <= 1.0
        assert 0.0 <= multi <= 1.0

    def test_flicker_range(self) -> None:
        """Test flicker output is in [0, 1]."""
        for t in [0.0, 0.1, 0.5, 1.0, 2.0]:
            result = AnimationCurve.flicker(t, 0.5, 1.0)
            assert 0.0 <= result <= 1.0

    def test_flicker_intensity(self) -> None:
        """Test flicker respects intensity parameter."""
        # Zero intensity should give constant 1.0
        for t in [0.0, 0.5, 1.0]:
            result = AnimationCurve.flicker(t, 0.0, 1.0)
            assert abs(result - 1.0) < 1e-6


# =============================================================================
# UV Animation Tests
# =============================================================================


class TestUVAnimation:
    """Test UV animation functions."""

    @pytest.mark.parametrize("ref", ANIMATION_REFERENCE_VALUES["uv_scroll"])
    def test_uv_scroll_reference_values(self, ref: dict) -> None:
        """Test animate_uv_scroll matches reference values."""
        result = AnimationCurve.animate_uv_scroll(ref["uv"], ref["speed"], ref["t"])
        assert abs(result[0] - ref["expected"][0]) < ref["tolerance"]
        assert abs(result[1] - ref["expected"][1]) < ref["tolerance"]

    def test_uv_scroll_wrapping(self) -> None:
        """Test UV scrolling wraps at 1.0."""
        result = AnimationCurve.animate_uv_scroll((0.0, 0.0), (2.0, 0.0), 1.0)
        assert abs(result[0] - 0.0) < 1e-6  # Wrapped twice
        assert abs(result[1] - 0.0) < 1e-6

    def test_uv_rotate(self) -> None:
        """Test UV rotation."""
        # 90 degree rotation should swap axes (with sign change)
        speed = math.pi / 2  # 90 degrees per second
        result = AnimationCurve.animate_uv_rotate((1.0, 0.5), speed, 1.0)

        # Point (0.5, 0) relative to center rotates to (0, 0.5) relative to center
        # So (1.0, 0.5) -> (0.5, 1.0)
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 1.0) < 1e-6

    def test_uv_oscillate(self) -> None:
        """Test UV oscillation."""
        # At t=0 (phase=0), sin=0, so no offset
        result = AnimationCurve.animate_uv_oscillate((0.5, 0.5), (0.1, 0.1), 1.0, 0.0)
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 0.5) < 1e-6

        # At t=0.25 (quarter period), sin=1, max offset
        result = AnimationCurve.animate_uv_oscillate((0.5, 0.5), (0.1, 0.1), 1.0, 0.25)
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.6) < 1e-6


# =============================================================================
# Color Animation Tests
# =============================================================================


class TestColorAnimation:
    """Test color animation functions."""

    def test_color_pulse(self) -> None:
        """Test color pulsing."""
        color = (1.0, 0.5, 0.0)

        # At t=0 (sin=0.5 for sin_wave_01), factor = 0.5 * intensity + (1 - intensity)
        # With intensity=0.5: factor = 0.5*0.5 + 0.5 = 0.75
        result = AnimationCurve.animate_color_pulse(color, 1.0, 0.5, 0.0)
        expected_factor = 0.5 * 0.5 + 0.5  # sin_wave_01 at t=0 is 0.5
        assert abs(result[0] - color[0] * expected_factor) < 1e-6

    def test_color_flicker(self) -> None:
        """Test color flickering."""
        color = (1.0, 0.8, 0.2)
        result = AnimationCurve.animate_color_flicker(color, 0.5, 1.0, 0.0)

        # Result should be color * flicker_factor
        assert 0.0 <= result[0] <= color[0]
        assert 0.0 <= result[1] <= color[1]
        assert 0.0 <= result[2] <= color[2]


# =============================================================================
# Emission Animation Tests
# =============================================================================


class TestEmissionAnimation:
    """Test emission animation functions."""

    def test_emission_pulse(self) -> None:
        """Test emission pulsing."""
        emission = (1.0, 0.5, 0.0)
        result = AnimationCurve.animate_emission_pulse(emission, 1.0, 0.2, 1.0, 0.25)

        # At t=0.25, sin_wave_01 = 1.0, so factor = min + (max-min) * 1 = 1.0
        assert abs(result[0] - emission[0] * 1.0) < 1e-6

    def test_emission_flicker(self) -> None:
        """Test emission flickering."""
        emission = (2.0, 1.0, 0.5)
        result = AnimationCurve.animate_emission_flicker(emission, 5.0, 0.5, 0.0)

        # Result should be emission modulated by flicker
        assert result[0] <= emission[0] * 1.1  # Allow small overshoot
        assert result[0] >= emission[0] * 0.4  # Not below min


# =============================================================================
# AnimatedParameter Tests
# =============================================================================


class TestAnimatedParameter:
    """Test AnimatedParameter class."""

    def test_sine_parameter(self) -> None:
        """Test sine wave parameter evaluation."""
        param = AnimatedParameter(
            name="test",
            wave_type=WaveType.SINE,
            frequency=1.0,
            amplitude=0.5,
            offset=0.5,
        )

        # At t=0.25 (sin=1), value = 0.5 + 1.0 * 0.5 = 1.0
        result = param.evaluate(0.25)
        assert abs(result - 1.0) < 1e-6

        # At t=0.75 (sin=-1), value = 0.5 + (-1) * 0.5 = 0.0
        result = param.evaluate(0.75)
        assert abs(result - 0.0) < 1e-6

    def test_sawtooth_parameter(self) -> None:
        """Test sawtooth wave parameter evaluation."""
        param = AnimatedParameter(
            name="test",
            wave_type=WaveType.SAWTOOTH,
            frequency=1.0,
            amplitude=1.0,
            offset=0.0,
        )

        # At t=0.25 (saw=0.25), normalized = 0.25*2-1 = -0.5
        # value = 0.0 + (-0.5) * 1.0 = -0.5
        result = param.evaluate(0.25)
        assert abs(result - (-0.5)) < 1e-6

    def test_pulse_parameter(self) -> None:
        """Test pulse wave parameter evaluation."""
        param = AnimatedParameter(
            name="test",
            wave_type=WaveType.SQUARE,
            frequency=1.0,
            amplitude=0.5,
            offset=0.5,
        )

        # Square wave (50% duty): first half is on (1), second half is off (0)
        # Normalized: on=1, off=-1
        result_on = param.evaluate(0.1)
        result_off = param.evaluate(0.6)

        assert abs(result_on - 1.0) < 1e-6  # 0.5 + 1.0 * 0.5
        assert abs(result_off - 0.0) < 1e-6  # 0.5 + (-1) * 0.5


# =============================================================================
# AnimationState Tests
# =============================================================================


class TestAnimationState:
    """Test AnimationState class."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = AnimationState()
        assert state.playing is True
        assert state.loop is True
        assert state.speed == 1.0
        assert state.duration == 0.0
        assert state.time == 0.0

    def test_update(self) -> None:
        """Test state update."""
        state = AnimationState()
        result = state.update(0.016)

        assert result is True
        assert abs(state.time - 0.016) < 1e-6

    def test_pause(self) -> None:
        """Test pausing does not update time."""
        state = AnimationState()
        state.update(0.5)
        state.playing = False
        state.update(0.5)

        assert abs(state.time - 0.5) < 1e-6

    def test_speed(self) -> None:
        """Test speed multiplier."""
        state = AnimationState(speed=2.0)
        state.update(0.5)

        assert abs(state.time - 1.0) < 1e-6

    def test_duration_loop(self) -> None:
        """Test looping at duration."""
        state = AnimationState(duration=1.0, loop=True)
        state.update(1.5)

        # Should wrap at 1.0
        assert abs(state.time - 0.5) < 1e-6

    def test_duration_no_loop(self) -> None:
        """Test stopping at duration without loop."""
        state = AnimationState(duration=1.0, loop=False)
        result = state.update(1.5)

        assert result is False
        assert state.playing is False

    def test_seek(self) -> None:
        """Test seeking to specific time."""
        state = AnimationState()
        state.update(0.5)
        state.seek(2.0)

        assert abs(state.time - 2.0) < 1e-6

    def test_reset(self) -> None:
        """Test reset to initial state."""
        state = AnimationState()
        state.update(1.0)
        state.reset()

        assert state.time == 0.0
        assert state.playing is True


# =============================================================================
# MaterialAnimator Tests
# =============================================================================


class TestMaterialAnimator:
    """Test MaterialAnimator class."""

    def test_create_animator(self) -> None:
        """Test animator creation."""
        animator = MaterialAnimator()
        assert animator.time == 0.0

    def test_add_parameter(self) -> None:
        """Test adding animated parameters."""
        animator = MaterialAnimator()
        param = AnimatedParameter(name="roughness", wave_type=WaveType.SINE)
        animator.add_parameter(param)

        assert animator.get_parameter("roughness") is param

    def test_remove_parameter(self) -> None:
        """Test removing animated parameters."""
        animator = MaterialAnimator()
        animator.add_parameter(AnimatedParameter(name="test"))
        animator.remove_parameter("test")

        assert animator.get_parameter("test") is None

    def test_get_value(self) -> None:
        """Test getting animated values."""
        animator = MaterialAnimator()
        animator.add_parameter(
            AnimatedParameter(
                name="emissive",
                wave_type=WaveType.SINE,
                frequency=1.0,
                amplitude=1.0,
                offset=0.0,
            )
        )

        # At t=0, sin=0
        assert abs(animator.get_value("emissive") or 0.0) < 1e-6

        animator.update(0.25)
        # At t=0.25, sin=1
        assert abs((animator.get_value("emissive") or 0.0) - 1.0) < 1e-6

    def test_get_all_values(self) -> None:
        """Test getting all animated values."""
        animator = MaterialAnimator()
        animator.add_parameter(AnimatedParameter(name="a", offset=1.0))
        animator.add_parameter(AnimatedParameter(name="b", offset=2.0))

        values = animator.get_all_values()
        assert "a" in values
        assert "b" in values

    def test_play_pause_stop(self) -> None:
        """Test playback control."""
        animator = MaterialAnimator()

        animator.update(0.5)
        animator.pause()
        animator.update(0.5)
        assert abs(animator.time - 0.5) < 1e-6

        animator.play()
        animator.update(0.5)
        assert abs(animator.time - 1.0) < 1e-6

        animator.stop()
        assert animator.time == 0.0

    def test_speed_control(self) -> None:
        """Test speed control."""
        animator = MaterialAnimator()
        animator.set_speed(2.0)
        animator.update(0.5)

        assert abs(animator.time - 1.0) < 1e-6

    def test_time_uniforms(self) -> None:
        """Test time uniforms access."""
        animator = MaterialAnimator()
        animator.update(1.5)

        uniforms = animator.time_uniforms
        assert abs(uniforms.elapsed_seconds - 1.5) < 1e-6

    def test_callback(self) -> None:
        """Test animation callback."""
        animator = MaterialAnimator()
        callback_times = []

        animator.add_callback(lambda t: callback_times.append(t))
        animator.update(0.1)
        animator.update(0.2)

        assert len(callback_times) == 2
        assert abs(callback_times[0] - 0.1) < 1e-6
        assert abs(callback_times[1] - 0.3) < 1e-6


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnimationIntegration:
    """Integration tests for the animation system."""

    def test_sample_animation_sin_wave_uv_offset(self) -> None:
        """Test sample animation: sin wave UV offset.

        This verifies the acceptance criteria for animated UV coordinates.
        """
        animator = MaterialAnimator()
        animator.add_parameter(
            AnimatedParameter(
                name="uv_offset",
                wave_type=WaveType.SINE,
                frequency=0.5,  # 0.5 Hz = 2 second period
                amplitude=0.1,  # Max offset of 0.1
                offset=0.0,
            )
        )

        # Run animation for 1 second (quarter period at 0.5 Hz)
        for _ in range(60):  # 60 frames
            animator.update(1.0 / 60.0)

        # At t=1.0 for 0.5 Hz, we're at phase 0.5, so sin(pi) = 0
        offset = animator.get_value("uv_offset")
        assert offset is not None
        assert abs(offset) < 0.1  # Should be near zero at this phase

        # Continue to quarter period (t=0.5 for full period)
        animator.seek(0.5)
        offset = animator.get_value("uv_offset")
        assert offset is not None
        # At t=0.5 for 0.5 Hz, sin(0.5 * 0.5 * 2pi) = sin(pi/2) = 1
        assert abs(offset - 0.1) < 1e-6

    def test_full_animation_pipeline(self) -> None:
        """Test full animation pipeline from uniforms to parameter evaluation."""
        # Create time uniforms
        uniforms = TimeUniforms()

        # Create time context
        ctx = TimeContext(_uniforms=uniforms)

        # Create animator
        animator = MaterialAnimator()
        animator.add_parameter(
            AnimatedParameter(
                name="emission_intensity",
                wave_type=WaveType.SINE,
                frequency=2.0,
                amplitude=0.5,
                offset=0.5,
            )
        )

        # Simulate 60 FPS for 2 seconds
        dt = 1.0 / 60.0
        for frame in range(120):
            uniforms.update(dt)
            animator.update(dt)

            # Verify context and animator stay in sync
            assert abs(ctx.time() - animator.time) < 1e-6

            # Get emission value
            emission = animator.get_value("emission_intensity")
            assert emission is not None
            assert 0.0 <= emission <= 1.0  # Should stay in valid range

        # Verify final time is approximately 2 seconds
        assert abs(ctx.time() - 2.0) < 0.1

    def test_animated_material_simulation(self) -> None:
        """Simulate an animated material with multiple parameters."""
        animator = MaterialAnimator()

        # Add multiple animated parameters
        animator.add_parameter(
            AnimatedParameter(
                name="uv_scroll_x",
                wave_type=WaveType.SAWTOOTH,
                frequency=0.25,  # Complete scroll every 4 seconds
                amplitude=1.0,
                offset=0.0,
            )
        )
        animator.add_parameter(
            AnimatedParameter(
                name="emission_pulse",
                wave_type=WaveType.SINE,
                frequency=1.0,
                amplitude=0.3,
                offset=0.7,
            )
        )
        animator.add_parameter(
            AnimatedParameter(
                name="roughness_variation",
                wave_type=WaveType.TRIANGLE,
                frequency=0.5,
                amplitude=0.1,
                offset=0.5,
            )
        )

        # Run animation
        animator.update(2.0)

        # Get all values
        values = animator.get_all_values()

        # Verify all parameters have values
        assert "uv_scroll_x" in values
        assert "emission_pulse" in values
        assert "roughness_variation" in values

        # Verify values are in expected ranges
        assert -1.0 <= values["uv_scroll_x"] <= 1.0
        assert 0.4 <= values["emission_pulse"] <= 1.0
        assert 0.4 <= values["roughness_variation"] <= 0.6
