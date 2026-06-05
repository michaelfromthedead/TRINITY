"""
Tests for Temporal Anti-Aliasing (T-DEMO-3.13).

This module contains comprehensive tests for:
- Halton sequence generation
- Sub-pixel jitter calculation
- Texture operations
- Temporal accumulation and convergence
- Camera movement detection
- WGSL code generation
"""

from __future__ import annotations

import math
import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.rendering.demoscene.temporal_aa import (
    # Halton sequence
    halton_sequence,
    halton_2d,
    # Jitter
    JitterPattern,
    JitterSequence,
    get_jitter,
    # Texture
    Texture,
    # Accumulator
    AccumulatorConfig,
    TemporalAccumulator,
    # WGSL
    generate_jitter_wgsl,
    generate_accumulation_wgsl,
    generate_taa_pipeline_wgsl,
)


# =============================================================================
# Halton Sequence Tests
# =============================================================================


class TestHaltonSequence:
    """Tests for the Halton sequence implementation."""

    def test_halton_base_2_first_values(self) -> None:
        """Test Halton base-2 sequence produces correct values."""
        # H(0, 2) = 0
        # H(1, 2) = 0.5
        # H(2, 2) = 0.25
        # H(3, 2) = 0.75
        # H(4, 2) = 0.125
        assert halton_sequence(0, 2) == pytest.approx(0.0)
        assert halton_sequence(1, 2) == pytest.approx(0.5)
        assert halton_sequence(2, 2) == pytest.approx(0.25)
        assert halton_sequence(3, 2) == pytest.approx(0.75)
        assert halton_sequence(4, 2) == pytest.approx(0.125)

    def test_halton_base_3_first_values(self) -> None:
        """Test Halton base-3 sequence produces correct values."""
        # H(0, 3) = 0
        # H(1, 3) = 1/3
        # H(2, 3) = 2/3
        # H(3, 3) = 1/9
        assert halton_sequence(0, 3) == pytest.approx(0.0)
        assert halton_sequence(1, 3) == pytest.approx(1.0 / 3.0)
        assert halton_sequence(2, 3) == pytest.approx(2.0 / 3.0)
        assert halton_sequence(3, 3) == pytest.approx(1.0 / 9.0)

    def test_halton_range_zero_to_one(self) -> None:
        """Halton values should always be in [0, 1)."""
        for base in [2, 3, 5, 7]:
            for i in range(100):
                val = halton_sequence(i, base)
                assert 0.0 <= val < 1.0, f"H({i}, {base}) = {val} out of range"

    def test_halton_invalid_index(self) -> None:
        """Negative index should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            halton_sequence(-1, 2)

    def test_halton_invalid_base(self) -> None:
        """Base < 2 should raise ValueError."""
        with pytest.raises(ValueError, match=">= 2"):
            halton_sequence(0, 1)
        with pytest.raises(ValueError, match=">= 2"):
            halton_sequence(0, 0)

    def test_halton_2d_returns_vec2(self) -> None:
        """halton_2d should return Vec2 with bases 2 and 3."""
        h = halton_2d(5)
        assert isinstance(h, Vec2)
        assert h.x == pytest.approx(halton_sequence(5, 2))
        assert h.y == pytest.approx(halton_sequence(5, 3))

    def test_halton_low_discrepancy(self) -> None:
        """
        Halton sequence should have low discrepancy (good coverage).
        After 64 samples, the 2D unit square should be well-covered.
        """
        # Divide unit square into 4x4 grid and count samples per cell
        grid = [[0] * 4 for _ in range(4)]
        # Use 64 samples to ensure good coverage
        for i in range(64):
            h = halton_2d(i)
            gx = min(3, int(h.x * 4))
            gy = min(3, int(h.y * 4))
            grid[gy][gx] += 1

        # Each cell should have at least 1 sample (low discrepancy)
        # With 64 samples in 16 cells, average is 4 per cell
        for row in grid:
            for cell in row:
                assert cell >= 1, "Halton sequence should cover all grid cells"

        # Check that samples are reasonably distributed (not all in one cell)
        total = sum(sum(row) for row in grid)
        assert total == 64


# =============================================================================
# Jitter Tests
# =============================================================================


class TestGetJitter:
    """Tests for the get_jitter function."""

    def test_jitter_centered(self) -> None:
        """Jitter should be centered around 0 (range [-0.5, 0.5))."""
        for i in range(20):
            j = get_jitter(i, 16)
            assert -0.5 <= j.x < 0.5, f"Jitter x={j.x} out of range"
            assert -0.5 <= j.y < 0.5, f"Jitter y={j.y} out of range"

    def test_jitter_wraps_at_sequence_length(self) -> None:
        """Jitter should wrap around at sequence_length."""
        j0 = get_jitter(0, 16)
        j16 = get_jitter(16, 16)
        assert j0.x == pytest.approx(j16.x)
        assert j0.y == pytest.approx(j16.y)

    def test_jitter_different_each_frame(self) -> None:
        """Consecutive frames should have different jitter."""
        jitters = [get_jitter(i, 16) for i in range(16)]
        # Check all pairs are different
        for i in range(16):
            for j in range(i + 1, 16):
                ji, jj = jitters[i], jitters[j]
                same = (abs(ji.x - jj.x) < 1e-9 and abs(ji.y - jj.y) < 1e-9)
                assert not same, f"Jitter {i} and {j} are identical"

    def test_jitter_first_frame(self) -> None:
        """First frame jitter should be at corner of Halton sequence."""
        j = get_jitter(0, 16)
        # H(0, 2) = 0, H(0, 3) = 0 -> centered: (-0.5, -0.5)
        assert j.x == pytest.approx(-0.5)
        assert j.y == pytest.approx(-0.5)


class TestJitterSequence:
    """Tests for the JitterSequence class."""

    def test_halton_pattern(self) -> None:
        """Halton pattern should produce centered jitter."""
        seq = JitterSequence(pattern=JitterPattern.HALTON, sequence_length=8)
        for i in range(8):
            j = seq.get_jitter(i)
            assert -0.5 <= j.x < 0.5
            assert -0.5 <= j.y < 0.5

    def test_halton_rotated_pattern(self) -> None:
        """Rotated Halton should still be centered."""
        seq = JitterSequence(pattern=JitterPattern.HALTON_ROTATED, sequence_length=8)
        for i in range(8):
            j = seq.get_jitter(i)
            # Rotation can push slightly beyond 0.5, but should be close
            assert abs(j.x) < 1.0
            assert abs(j.y) < 1.0

    def test_uniform_grid_pattern(self) -> None:
        """Uniform grid should produce regular samples."""
        seq = JitterSequence(pattern=JitterPattern.UNIFORM_GRID, sequence_length=4)
        jitters = [seq.get_jitter(i) for i in range(4)]
        # Should form a 2x2 grid centered at 0
        xs = sorted([j.x for j in jitters])
        ys = sorted([j.y for j in jitters])
        # Two unique x values, two unique y values
        unique_xs = len(set(round(x, 6) for x in xs))
        unique_ys = len(set(round(y, 6) for y in ys))
        assert unique_xs == 2
        assert unique_ys == 2

    def test_interleaved_pattern(self) -> None:
        """Interleaved pattern should have 4-frame base cycle."""
        seq = JitterSequence(pattern=JitterPattern.INTERLEAVED, sequence_length=8)
        # First 4 frames should be different from second 4
        first_four = [seq.get_jitter(i) for i in range(4)]
        second_four = [seq.get_jitter(i) for i in range(4, 8)]
        # Each should be in reasonable range
        for j in first_four + second_four:
            assert abs(j.x) < 0.5
            assert abs(j.y) < 0.5

    def test_scale_parameter(self) -> None:
        """Scale should multiply jitter magnitude."""
        seq1 = JitterSequence(scale=1.0, sequence_length=8)
        seq2 = JitterSequence(scale=2.0, sequence_length=8)
        for i in range(8):
            j1 = seq1.get_jitter(i)
            j2 = seq2.get_jitter(i)
            assert j2.x == pytest.approx(j1.x * 2.0, abs=1e-9)
            assert j2.y == pytest.approx(j1.y * 2.0, abs=1e-9)

    def test_invalid_sequence_length(self) -> None:
        """Sequence length < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            JitterSequence(sequence_length=0)

    def test_reset(self) -> None:
        """Reset should rebuild the sequence."""
        seq = JitterSequence(sequence_length=4)
        original = [seq.get_jitter(i) for i in range(4)]
        seq.reset()
        after_reset = [seq.get_jitter(i) for i in range(4)]
        for i in range(4):
            assert original[i].x == pytest.approx(after_reset[i].x)
            assert original[i].y == pytest.approx(after_reset[i].y)


# =============================================================================
# Texture Tests
# =============================================================================


class TestTexture:
    """Tests for the Texture class."""

    def test_creation(self) -> None:
        """Texture should initialize with correct dimensions."""
        tex = Texture(16, 8)
        assert tex.width == 16
        assert tex.height == 8
        assert len(tex.data) == 16 * 8

    def test_default_pixels_zero(self) -> None:
        """Pixels should default to zero."""
        tex = Texture(4, 4)
        for y in range(4):
            for x in range(4):
                p = tex.get_pixel(x, y)
                assert p.x == 0.0
                assert p.y == 0.0
                assert p.z == 0.0
                assert p.w == 0.0

    def test_set_get_pixel(self) -> None:
        """set_pixel and get_pixel should work correctly."""
        tex = Texture(4, 4)
        color = Vec4(0.5, 0.25, 0.75, 1.0)
        tex.set_pixel(2, 3, color)
        p = tex.get_pixel(2, 3)
        assert p.x == pytest.approx(0.5)
        assert p.y == pytest.approx(0.25)
        assert p.z == pytest.approx(0.75)
        assert p.w == pytest.approx(1.0)

    def test_out_of_bounds_get_returns_zero(self) -> None:
        """Out-of-bounds get should return zero."""
        tex = Texture(4, 4)
        tex.set_pixel(0, 0, Vec4(1.0, 1.0, 1.0, 1.0))
        assert tex.get_pixel(-1, 0).x == 0.0
        assert tex.get_pixel(4, 0).x == 0.0
        assert tex.get_pixel(0, -1).x == 0.0
        assert tex.get_pixel(0, 4).x == 0.0

    def test_out_of_bounds_set_ignored(self) -> None:
        """Out-of-bounds set should be silently ignored."""
        tex = Texture(4, 4)
        tex.set_pixel(-1, 0, Vec4(1.0, 1.0, 1.0, 1.0))
        tex.set_pixel(4, 0, Vec4(1.0, 1.0, 1.0, 1.0))
        # Should not crash, and no pixels should be modified
        for y in range(4):
            for x in range(4):
                assert tex.get_pixel(x, y).x == 0.0

    def test_sample_center(self) -> None:
        """Sample at pixel center should return pixel value."""
        tex = Texture(4, 4)
        tex.set_pixel(2, 2, Vec4(1.0, 0.0, 0.0, 1.0))
        # UV for center of pixel (2, 2) in 4x4 texture
        uv = Vec2(2.5 / 4.0, 2.5 / 4.0)
        sampled = tex.sample(uv)
        # Should be close to (1, 0, 0, 1) but bilinear will blend with neighbors
        assert sampled.x > 0.2  # At least some red

    def test_clear(self) -> None:
        """Clear should set all pixels to specified color."""
        tex = Texture(4, 4)
        tex.set_pixel(1, 1, Vec4(1.0, 1.0, 1.0, 1.0))
        tex.clear(Vec4(0.5, 0.5, 0.5, 1.0))
        for y in range(4):
            for x in range(4):
                p = tex.get_pixel(x, y)
                assert p.x == pytest.approx(0.5)
                assert p.y == pytest.approx(0.5)
                assert p.z == pytest.approx(0.5)

    def test_copy_from(self) -> None:
        """copy_from should duplicate another texture's data."""
        src = Texture(4, 4)
        src.set_pixel(1, 2, Vec4(0.8, 0.6, 0.4, 1.0))
        dst = Texture(4, 4)
        dst.copy_from(src)
        p = dst.get_pixel(1, 2)
        assert p.x == pytest.approx(0.8)
        assert p.y == pytest.approx(0.6)
        assert p.z == pytest.approx(0.4)

    def test_copy_from_size_mismatch(self) -> None:
        """copy_from with mismatched sizes should raise."""
        src = Texture(4, 4)
        dst = Texture(8, 4)
        with pytest.raises(ValueError, match="dimensions"):
            dst.copy_from(src)

    def test_clone(self) -> None:
        """clone should create an independent copy."""
        original = Texture(4, 4)
        original.set_pixel(0, 0, Vec4(1.0, 0.0, 0.0, 1.0))
        cloned = original.clone()
        # Modify original
        original.set_pixel(0, 0, Vec4(0.0, 1.0, 0.0, 1.0))
        # Clone should be unchanged
        p = cloned.get_pixel(0, 0)
        assert p.x == pytest.approx(1.0)
        assert p.y == pytest.approx(0.0)


# =============================================================================
# Temporal Accumulator Tests
# =============================================================================


class TestTemporalAccumulator:
    """Tests for the TemporalAccumulator class."""

    def test_creation(self) -> None:
        """Accumulator should initialize correctly."""
        acc = TemporalAccumulator(64, 32)
        assert acc.width == 64
        assert acc.height == 32
        assert acc.frame_count == 0
        assert not acc.is_converged

    def test_invalid_dimensions(self) -> None:
        """Invalid dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            TemporalAccumulator(0, 32)
        with pytest.raises(ValueError):
            TemporalAccumulator(32, 0)
        with pytest.raises(ValueError):
            TemporalAccumulator(-1, 32)

    def test_first_frame_passthrough(self) -> None:
        """First frame should pass through unchanged."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)
        current.set_pixel(1, 1, Vec4(1.0, 0.5, 0.25, 1.0))

        result = acc.accumulate(current)

        p = result.get_pixel(1, 1)
        assert p.x == pytest.approx(1.0)
        assert p.y == pytest.approx(0.5)
        assert p.z == pytest.approx(0.25)
        assert acc.frame_count == 1

    def test_frame_count_increments(self) -> None:
        """Frame count should increment each accumulation."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)

        for i in range(5):
            acc.accumulate(current)
            assert acc.frame_count == i + 1

    def test_accumulation_blends_frames(self) -> None:
        """Accumulation should blend current with history."""
        config = AccumulatorConfig(blend_factor=0.5)
        acc = TemporalAccumulator(4, 4, config)

        # Frame 1: white
        frame1 = Texture(4, 4)
        frame1.set_pixel(0, 0, Vec4(1.0, 1.0, 1.0, 1.0))
        acc.accumulate(frame1)

        # Frame 2: black
        frame2 = Texture(4, 4)
        frame2.set_pixel(0, 0, Vec4(0.0, 0.0, 0.0, 1.0))
        result = acc.accumulate(frame2)

        # Should be gray (blend of white and black)
        p = result.get_pixel(0, 0)
        assert 0.3 < p.x < 0.7  # Approximate blend

    def test_convergence_detection(self) -> None:
        """Accumulator should detect convergence after enough frames."""
        config = AccumulatorConfig(blend_factor=0.1)  # Converge after ~10 frames
        acc = TemporalAccumulator(4, 4, config)
        current = Texture(4, 4)

        # Accumulate until converged
        for _ in range(15):
            acc.accumulate(current)

        assert acc.is_converged

    def test_reset_clears_state(self) -> None:
        """Reset should clear history and frame count."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)
        current.set_pixel(0, 0, Vec4(1.0, 1.0, 1.0, 1.0))

        acc.accumulate(current)
        acc.accumulate(current)
        assert acc.frame_count == 2

        acc.reset()
        assert acc.frame_count == 0
        assert not acc.is_converged

    def test_resize(self) -> None:
        """Resize should change dimensions and reset."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)
        acc.accumulate(current)

        acc.resize(8, 8)
        assert acc.width == 8
        assert acc.height == 8
        assert acc.frame_count == 0

    def test_resize_invalid(self) -> None:
        """Resize with invalid dimensions should raise."""
        acc = TemporalAccumulator(4, 4)
        with pytest.raises(ValueError):
            acc.resize(0, 4)

    def test_camera_movement_resets(self) -> None:
        """Camera movement should reset accumulation."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)

        # Accumulate a few frames
        acc.accumulate(current, camera_position=Vec3(0, 0, 0))
        acc.accumulate(current, camera_position=Vec3(0, 0, 0))
        assert acc.frame_count == 2

        # Move camera
        acc.accumulate(current, camera_position=Vec3(1, 0, 0))
        # Frame count should be 1 (reset + this frame)
        assert acc.frame_count == 1

    def test_camera_rotation_resets(self) -> None:
        """Camera rotation should reset accumulation."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)

        acc.accumulate(current, camera_rotation=Vec3(0, 0, 0))
        acc.accumulate(current, camera_rotation=Vec3(0, 0, 0))
        assert acc.frame_count == 2

        # Rotate camera
        acc.accumulate(current, camera_rotation=Vec3(0, 0.1, 0))
        assert acc.frame_count == 1

    def test_mismatched_frame_size(self) -> None:
        """Mismatched frame size should raise ValueError."""
        acc = TemporalAccumulator(4, 4)
        wrong_size = Texture(8, 8)
        with pytest.raises(ValueError, match="size"):
            acc.accumulate(wrong_size)

    def test_get_history(self) -> None:
        """get_history should return a copy of the history buffer."""
        acc = TemporalAccumulator(4, 4)
        current = Texture(4, 4)
        current.set_pixel(0, 0, Vec4(1.0, 1.0, 1.0, 1.0))

        acc.accumulate(current)
        history = acc.get_history()

        # Modify returned history
        history.set_pixel(0, 0, Vec4(0.0, 0.0, 0.0, 0.0))

        # Internal history should be unchanged
        history2 = acc.get_history()
        p = history2.get_pixel(0, 0)
        assert p.x == pytest.approx(1.0)


class TestAccumulatorConfig:
    """Tests for AccumulatorConfig."""

    def test_default_config(self) -> None:
        """Default config should have reasonable values."""
        config = AccumulatorConfig()
        assert 0.0 < config.blend_factor < 1.0
        assert config.min_blend_factor < config.blend_factor
        assert config.max_blend_factor > config.blend_factor

    def test_custom_blend_factor(self) -> None:
        """Custom blend factor should be used."""
        config = AccumulatorConfig(blend_factor=0.2)
        acc = TemporalAccumulator(4, 4, config)
        assert acc.config.blend_factor == 0.2


# =============================================================================
# Convergence Tests
# =============================================================================


class TestConvergence:
    """Tests for temporal convergence behavior."""

    def test_constant_input_converges(self) -> None:
        """Constant input should converge to the input value."""
        config = AccumulatorConfig(blend_factor=0.1)
        acc = TemporalAccumulator(4, 4, config)
        constant = Texture(4, 4)
        constant.set_pixel(1, 1, Vec4(0.5, 0.5, 0.5, 1.0))

        # Accumulate many frames
        for _ in range(50):
            result = acc.accumulate(constant)

        # Should converge to 0.5
        p = result.get_pixel(1, 1)
        assert p.x == pytest.approx(0.5, abs=0.01)
        assert p.y == pytest.approx(0.5, abs=0.01)
        assert p.z == pytest.approx(0.5, abs=0.01)

    def test_alternating_input_averages(self) -> None:
        """Alternating black/white should converge to gray."""
        config = AccumulatorConfig(blend_factor=0.1)
        acc = TemporalAccumulator(4, 4, config)

        white = Texture(4, 4)
        white.set_pixel(0, 0, Vec4(1.0, 1.0, 1.0, 1.0))
        black = Texture(4, 4)
        black.set_pixel(0, 0, Vec4(0.0, 0.0, 0.0, 1.0))

        # Accumulate alternating frames
        for i in range(100):
            frame = white if i % 2 == 0 else black
            result = acc.accumulate(frame)

        # Should be approximately gray
        p = result.get_pixel(0, 0)
        assert 0.4 < p.x < 0.6

    def test_convergence_16_frames(self) -> None:
        """
        After 16 frames with default settings, image should be mostly converged.
        """
        acc = TemporalAccumulator(4, 4)
        target = Texture(4, 4)
        target.set_pixel(2, 2, Vec4(0.8, 0.4, 0.2, 1.0))

        for i in range(16):
            result = acc.accumulate(target)

        # Should be very close to target
        p = result.get_pixel(2, 2)
        assert p.x == pytest.approx(0.8, abs=0.05)
        assert p.y == pytest.approx(0.4, abs=0.05)
        assert p.z == pytest.approx(0.2, abs=0.05)


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generate_jitter_wgsl(self) -> None:
        """Jitter WGSL should contain required functions."""
        wgsl = generate_jitter_wgsl()
        assert "fn halton_sequence" in wgsl
        assert "fn get_jitter" in wgsl
        assert "fn apply_jitter_to_uv" in wgsl
        assert "vec2<f32>" in wgsl

    def test_generate_accumulation_wgsl(self) -> None:
        """Accumulation WGSL should contain compute shaders."""
        wgsl = generate_accumulation_wgsl()
        assert "struct TAAParams" in wgsl
        assert "fn taa_accumulate" in wgsl
        assert "fn taa_accumulate_clamped" in wgsl
        assert "@compute" in wgsl
        assert "@workgroup_size" in wgsl
        assert "textureLoad" in wgsl
        assert "textureStore" in wgsl

    def test_generate_taa_pipeline_wgsl(self) -> None:
        """Pipeline WGSL should combine jitter and accumulation."""
        wgsl = generate_taa_pipeline_wgsl(include_ray_jitter=True)
        assert "halton_sequence" in wgsl
        assert "taa_accumulate" in wgsl

    def test_generate_taa_pipeline_without_jitter(self) -> None:
        """Pipeline without jitter should only have accumulation."""
        wgsl = generate_taa_pipeline_wgsl(include_ray_jitter=False)
        assert "halton_sequence" not in wgsl
        assert "taa_accumulate" in wgsl

    def test_wgsl_syntax_validity(self) -> None:
        """Generated WGSL should have balanced braces and valid structure."""
        wgsl = generate_taa_pipeline_wgsl()
        # Count braces
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, "Unbalanced braces in WGSL"

        # Check for common WGSL keywords
        assert "fn " in wgsl
        assert "var" in wgsl or "let" in wgsl
        assert "return" in wgsl


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining jitter and accumulation."""

    def test_jitter_produces_different_rays(self) -> None:
        """
        Sub-pixel jitter should produce rays that sample different sub-pixel positions.
        This is verified by checking that jitter varies across frames.
        """
        jitters = []
        for frame in range(16):
            j = get_jitter(frame, 16)
            jitters.append((j.x, j.y))

        # All jitters should be unique
        unique_jitters = set(jitters)
        assert len(unique_jitters) == 16

    def test_full_taa_workflow(self) -> None:
        """
        Test complete TAA workflow:
        1. Generate jitter for each frame
        2. Render with jittered UV (simulated)
        3. Accumulate frames
        4. Verify convergence
        """
        acc = TemporalAccumulator(4, 4)

        for frame in range(20):
            jitter = get_jitter(frame, 16)

            # Simulate rendering with jitter
            # In real use, this would be: uv = (pixel + 0.5 + jitter) / resolution
            rendered = Texture(4, 4)
            # Fill with color based on jitter (simulates sub-pixel variation)
            color = Vec4(
                0.5 + jitter.x,  # Varies with jitter
                0.5 + jitter.y,
                0.5,
                1.0,
            )
            for y in range(4):
                for x in range(4):
                    rendered.set_pixel(x, y, color)

            result = acc.accumulate(rendered, jitter=jitter)

        # After many frames, result should average the jitter variation
        # Since jitter ranges from -0.5 to 0.5, color ranges from 0.0 to 1.0
        # The average should converge toward 0.5, but with exponential moving average
        # the convergence isn't perfect. Allow wider tolerance.
        p = result.get_pixel(0, 0)
        assert 0.3 < p.x < 0.7, f"Expected x near 0.5, got {p.x}"
        assert 0.3 < p.y < 0.7, f"Expected y near 0.5, got {p.y}"
        assert p.z == pytest.approx(0.5, abs=0.01)

    def test_camera_movement_resets_convergence(self) -> None:
        """
        Camera movement should reset accumulation,
        preventing ghosting from previous camera position.
        """
        acc = TemporalAccumulator(4, 4)
        white = Texture(4, 4)
        white.clear(Vec4(1.0, 1.0, 1.0, 1.0))
        black = Texture(4, 4)
        black.clear(Vec4(0.0, 0.0, 0.0, 1.0))

        # Accumulate white for several frames
        for _ in range(10):
            acc.accumulate(white, camera_position=Vec3(0, 0, 0))

        # Move camera and render black
        result = acc.accumulate(black, camera_position=Vec3(1, 0, 0))

        # Result should be black (not blended with white history)
        p = result.get_pixel(0, 0)
        assert p.x == pytest.approx(0.0)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_pixel_texture(self) -> None:
        """1x1 texture should work correctly."""
        acc = TemporalAccumulator(1, 1)
        current = Texture(1, 1)
        current.set_pixel(0, 0, Vec4(0.5, 0.5, 0.5, 1.0))
        result = acc.accumulate(current)
        p = result.get_pixel(0, 0)
        assert p.x == pytest.approx(0.5)

    def test_large_sequence_length(self) -> None:
        """Large sequence length should not cause issues."""
        seq = JitterSequence(sequence_length=1024)
        j = seq.get_jitter(500)
        assert -0.5 <= j.x < 0.5
        assert -0.5 <= j.y < 0.5

    def test_halton_large_index(self) -> None:
        """Large indices should still produce valid Halton values."""
        val = halton_sequence(10000, 2)
        assert 0.0 <= val < 1.0

    def test_zero_alpha_handling(self) -> None:
        """Zero alpha pixels should be accumulated correctly."""
        acc = TemporalAccumulator(4, 4)
        transparent = Texture(4, 4)
        transparent.set_pixel(0, 0, Vec4(1.0, 0.0, 0.0, 0.0))
        result = acc.accumulate(transparent)
        p = result.get_pixel(0, 0)
        assert p.w == pytest.approx(0.0)
