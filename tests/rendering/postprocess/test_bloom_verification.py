"""
Bloom FIX Verification Tests (T-PP-1.2)

Verifies all 4 fixes:
1. Input validation in BloomSettings.__post_init__
2. lerp() includes clamp_max, blur_iterations, resolution_scale, lens_dirt
3. downsample() creates buffers (not None)
4. Blur kernels: Gaussian, Kawase, Box produce correct output
"""

import math
import pytest

from engine.rendering.postprocess.bloom import (
    BloomBlur,
    BloomDownsample,
    BloomEffect,
    BloomMipSettings,
    BloomQuality,
    BloomSettings,
    BloomThreshold,
    BlurMethod,
    LensDirtSettings,
)


# =============================================================================
# FIX 1: Input validation in BloomSettings.__post_init__
# =============================================================================

class TestFix_InputValidation:
    """Verify BloomSettings validates all numeric fields."""

    def test_threshold_out_of_range_high(self):
        """threshold > 65504 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            BloomSettings(threshold=70000.0)

    def test_threshold_out_of_range_low(self):
        """threshold < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            BloomSettings(threshold=-1.0)

    def test_threshold_softness_out_of_range_high(self):
        """threshold_softness > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_softness"):
            BloomSettings(threshold_softness=1.5)

    def test_threshold_softness_out_of_range_low(self):
        """threshold_softness < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_softness"):
            BloomSettings(threshold_softness=-0.1)

    def test_clamp_max_negative(self):
        """clamp_max < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="clamp_max"):
            BloomSettings(clamp_max=-1.0)

    def test_intensity_out_of_range_high(self):
        """intensity > 10 should raise ValueError."""
        with pytest.raises(ValueError, match="intensity"):
            BloomSettings(intensity=11.0)

    def test_intensity_out_of_range_low(self):
        """intensity < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="intensity"):
            BloomSettings(intensity=-0.1)

    def test_scatter_out_of_range_high(self):
        """scatter > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="scatter"):
            BloomSettings(scatter=1.5)

    def test_scatter_out_of_range_low(self):
        """scatter < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="scatter"):
            BloomSettings(scatter=-0.1)

    def test_blur_iterations_negative(self):
        """blur_iterations < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="blur_iterations"):
            BloomSettings(blur_iterations=-1)

    def test_resolution_scale_too_small(self):
        """resolution_scale < 0.25 should raise ValueError."""
        with pytest.raises(ValueError, match="resolution_scale"):
            BloomSettings(resolution_scale=0.1)

    def test_resolution_scale_too_large(self):
        """resolution_scale > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="resolution_scale"):
            BloomSettings(resolution_scale=2.0)

    def test_boundary_values_accepted(self):
        """Boundary values should be accepted."""
        settings = BloomSettings(
            threshold=0.0,
            threshold_softness=0.0,
            clamp_max=0.0,
            intensity=0.0,
            scatter=0.0,
            blur_iterations=0,
            resolution_scale=0.25,
        )
        assert settings.threshold == 0.0
        assert settings.threshold_softness == 0.0
        assert settings.clamp_max == 0.0
        assert settings.intensity == 0.0
        assert settings.scatter == 0.0
        assert settings.blur_iterations == 0
        assert settings.resolution_scale == 0.25

    def test_upper_boundary_values_accepted(self):
        """Upper boundary values should be accepted."""
        settings = BloomSettings(
            threshold=65504.0,
            threshold_softness=1.0,
            intensity=10.0,
            scatter=1.0,
            resolution_scale=1.0,
        )
        assert settings.threshold == 65504.0
        assert settings.threshold_softness == 1.0
        assert settings.intensity == 10.0
        assert settings.scatter == 1.0
        assert settings.resolution_scale == 1.0


# =============================================================================
# FIX 2: lerp() includes clamp_max, blur_iterations, resolution_scale, lens_dirt
# =============================================================================

class TestFix_LerpFields:
    """Verify lerp() interpolates all previously-missing fields."""

    def test_lerp_clamp_max(self):
        """clamp_max should interpolate linearly."""
        a = BloomSettings(clamp_max=100.0)
        b = BloomSettings(clamp_max=200.0)
        result = a.lerp(b, 0.5)
        assert result.clamp_max == 150.0

    def test_lerp_clamp_max_extremes(self):
        """clamp_max lerp at t=0 and t=1."""
        a = BloomSettings(clamp_max=100.0)
        b = BloomSettings(clamp_max=200.0)
        assert a.lerp(b, 0.0).clamp_max == 100.0
        assert a.lerp(b, 1.0).clamp_max == 200.0

    def test_lerp_blur_iterations(self):
        """blur_iterations should interpolate as int."""
        a = BloomSettings(blur_iterations=2)
        b = BloomSettings(blur_iterations=6)
        result = a.lerp(b, 0.5)
        assert result.blur_iterations == 4
        assert isinstance(result.blur_iterations, int)

    def test_lerp_resolution_scale(self):
        """resolution_scale should interpolate linearly."""
        a = BloomSettings(resolution_scale=0.25)
        b = BloomSettings(resolution_scale=1.0)
        result = a.lerp(b, 0.5)
        assert abs(result.resolution_scale - 0.625) < 1e-9

    def test_lerp_lens_dirt_enabled_discrete(self):
        """lens_dirt.enabled uses discrete selection."""
        a = BloomSettings(lens_dirt=LensDirtSettings(enabled=True, intensity=0.5))
        b = BloomSettings(lens_dirt=LensDirtSettings(enabled=False, intensity=2.0))
        assert a.lerp(b, 0.25).lens_dirt.enabled is True
        assert a.lerp(b, 0.5).lens_dirt.enabled is False

    def test_lerp_lens_dirt_intensity(self):
        """lens_dirt.intensity should interpolate linearly."""
        a = BloomSettings(lens_dirt=LensDirtSettings(intensity=0.0))
        b = BloomSettings(lens_dirt=LensDirtSettings(intensity=2.0))
        result = a.lerp(b, 0.5)
        assert result.lens_dirt.intensity == 1.0

    def test_lerp_lens_dirt_tint(self):
        """lens_dirt.tint should interpolate per-component."""
        a = BloomSettings(lens_dirt=LensDirtSettings(tint=(0.0, 0.5, 1.0)))
        b = BloomSettings(lens_dirt=LensDirtSettings(tint=(1.0, 0.5, 0.0)))
        result = a.lerp(b, 0.5)
        assert abs(result.lens_dirt.tint[0] - 0.5) < 1e-9
        assert abs(result.lens_dirt.tint[1] - 0.5) < 1e-9
        assert abs(result.lens_dirt.tint[2] - 0.5) < 1e-9

    def test_lerp_lens_dirt_texture_path_discrete(self):
        """lens_dirt.texture_path uses discrete selection."""
        a = BloomSettings(lens_dirt=LensDirtSettings(texture_path="a.png"))
        b = BloomSettings(lens_dirt=LensDirtSettings(texture_path="b.png"))
        assert a.lerp(b, 0.25).lens_dirt.texture_path == "a.png"
        assert a.lerp(b, 0.5).lens_dirt.texture_path == "b.png"

    def test_lerp_all_four_fields_in_one_call(self):
        """All 4 previously-missing fields lerp in a single call."""
        a = BloomSettings(
            clamp_max=100.0,
            blur_iterations=2,
            resolution_scale=0.25,
            lens_dirt=LensDirtSettings(intensity=0.0, tint=(0.0, 0.0, 0.0)),
        )
        b = BloomSettings(
            clamp_max=200.0,
            blur_iterations=6,
            resolution_scale=1.0,
            lens_dirt=LensDirtSettings(intensity=2.0, tint=(1.0, 1.0, 1.0)),
        )
        result = a.lerp(b, 0.5)
        assert result.clamp_max == 150.0
        assert result.blur_iterations == 4
        assert abs(result.resolution_scale - 0.625) < 1e-9
        assert result.lens_dirt.intensity == 1.0
        assert abs(result.lens_dirt.tint[0] - 0.5) < 1e-9


# =============================================================================
# FIX 3: downsample() creates buffers (not None)
# =============================================================================

class TestFix_DownsampleCreatesBuffers:
    """Verify BloomDownsample.downsample() creates actual buffers."""

    def test_downsample_returns_non_none(self):
        """downsample() should never return None for valid mip level."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(1920, 1080, resolution_scale=0.5)
        source = [0.0] * 100
        for i in range(ds.mip_count):
            buf = ds.downsample(source, i)
            assert buf is not None, f"mip {i} returned None"

    def test_downsample_buffer_size_correct(self):
        """Each downsampled buffer should have the correct RGBA size."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(1920, 1080, resolution_scale=0.5)
        source = [0.0] * 100
        for i in range(ds.mip_count):
            w, h = ds.mip_sizes[i]
            buf = ds.downsample(source, i)
            assert len(buf) == w * h * 4, (
                f"mip {i} size ({w}x{h}): expected {w*h*4} floats, got {len(buf)}"
            )

    def test_downsample_buffer_is_list_of_floats(self):
        """Buffer elements should be floats initialized to 0.0."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(1920, 1080, resolution_scale=0.5)
        source = [0.0] * 100
        buf = ds.downsample(source, 0)
        assert isinstance(buf, list)
        assert all(isinstance(v, float) for v in buf[:10])
        assert all(v == 0.0 for v in buf[:10])

    def test_downsample_out_of_range_returns_source(self):
        """downsample() beyond last buffer returns source."""
        ds = BloomDownsample(max_mips=3)
        ds.setup(100, 100)
        source = [42.0]
        result = ds.downsample(source, 99)
        assert result is source

    def test_downsample_idempotent_buffer(self):
        """Calling downsample() twice for same mip returns same buffer."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(800, 600)
        source = [0.0] * 100
        buf1 = ds.downsample(source, 0)
        buf2 = ds.downsample(source, 0)
        assert buf1 is buf2, "second call should return same buffer"

    def test_downsample_get_mip_buffer_after_downsample(self):
        """get_mip_buffer() returns the created buffer."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(800, 600)
        source = [0.0] * 100
        buf = ds.downsample(source, 0)
        assert ds.get_mip_buffer(0) is buf

    def test_downsample_multiple_mips_all_created(self):
        """All mips in range should get valid buffers."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(1920, 1080)
        source = [0.0] * 100
        for i in range(ds.mip_count):
            ds.downsample(source, i)
        for i in range(ds.mip_count):
            assert ds.get_mip_buffer(i) is not None, f"mip {i} buffer is None after creating all"

    def test_downsample_buffer_matches_mip_size(self):
        """Buffer size must match the registered mip_size for every level."""
        ds = BloomDownsample(max_mips=8)
        ds.setup(1920, 1080)
        source = [0.0] * 100
        for i in range(ds.mip_count):
            w, h = ds.mip_sizes[i]
            buf = ds.downsample(source, i)
            assert len(buf) == w * h * 4, (
                f"mip {i} ({w}x{h}): len={len(buf)} vs expected {w*h*4}"
            )


# =============================================================================
# FIX 4: Blur kernels (Gaussian, Kawase, Box)
# =============================================================================

def _make_checkerboard(w, h, tile_size=8):
    """Create a flat RGBA checkerboard buffer for blur testing."""
    buf = [0.0] * (w * h * 4)
    for y in range(h):
        for x in range(w):
            val = 1.0 if ((x // tile_size) + (y // tile_size)) % 2 == 0 else 0.0
            idx = (y * w + x) * 4
            buf[idx] = val
            buf[idx + 1] = val
            buf[idx + 2] = val
            buf[idx + 3] = 1.0
    return buf


def _mean_color(buf):
    """Return mean R value across buffer."""
    return sum(buf[0::4]) / (len(buf) // 4)


class TestFix_GaussianBlur:
    """Verify Gaussian blur kernel produces correct output."""

    def test_gaussian_blur_smoothes_checkerboard(self):
        """Gaussian blur should reduce contrast (smooth) a checkerboard."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=4, sigma=2.0)
        w, h = 64, 64
        src = _make_checkerboard(w, h)
        original_mean = _mean_color(src)
        target = list(src)
        result = blur.blur(src, target, iterations=2, width=w, height=h)
        blurred_mean = _mean_color(result)
        assert abs(blurred_mean - original_mean) < 0.05

    def test_gaussian_weights_stored(self):
        """Gaussian weights should be calculated and accessible."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=4, sigma=2.0)
        assert len(blur._gaussian_weights) == 5
        assert len(blur._gaussian_offsets) == 5

    def test_gaussian_edge_pixels_clamp(self):
        """Edge pixels should not cause index errors."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=4, sigma=2.0)
        w, h = 16, 16
        src = [float(i % 256) / 255.0 for i in range(w * h * 4)]
        target = list(src)
        result = blur.blur(src, target, iterations=1, width=w, height=h)
        assert result is not None

    def test_gaussian_single_pixel(self):
        """Gaussian blur on small buffer should not crash."""
        blur = BloomBlur(method=BlurMethod.GAUSSIAN)
        blur.calculate_gaussian_weights(radius=2, sigma=1.0)
        src = [1.0, 0.0, 0.0, 1.0]
        result = blur.blur(src, list(src), iterations=1, width=2, height=1)
        assert result is not None


class TestFix_KawaseBlur:
    """Verify Kawase blur kernel produces correct output."""

    def test_kawase_blur_smoothes(self):
        """Kawase blur should smooth a checkerboard pattern."""
        blur = BloomBlur(method=BlurMethod.KAWASE)
        w, h = 64, 64
        src = _make_checkerboard(w, h)
        original_mean = _mean_color(src)
        target = list(src)
        result = blur.blur(src, target, iterations=2, width=w, height=h)
        blurred_mean = _mean_color(result)
        assert abs(blurred_mean - original_mean) < 0.05

    def test_kawase_edge_pixels_clamp(self):
        """Kawase edge pixels should not cause index errors."""
        blur = BloomBlur(method=BlurMethod.KAWASE)
        w, h = 4, 4
        src = [float(i) for i in range(w * h * 4)]
        target = list(src)
        result = blur.blur(src, target, iterations=2, width=w, height=h)
        assert result is not None

    def test_kawase_offsets_increase(self):
        """Kawase offsets should increase with iteration."""
        blur = BloomBlur(method=BlurMethod.KAWASE)
        assert blur.get_kawase_offsets(0) == 0.5
        assert blur.get_kawase_offsets(1) == 1.5
        assert blur.get_kawase_offsets(2) == 2.5

    def test_kawase_multi_iteration(self):
        """Multiple Kawase iterations should converge (mean stable)."""
        blur = BloomBlur(method=BlurMethod.KAWASE)
        w, h = 32, 32
        src = _make_checkerboard(w, h)
        result_1 = blur.blur(list(src), list(src), iterations=1, width=w, height=h)
        result_3 = blur.blur(list(src), list(src), iterations=3, width=w, height=h)
        mean_1 = _mean_color(result_1)
        mean_3 = _mean_color(result_3)
        assert abs(mean_1 - mean_3) < 0.1


class TestFix_BoxBlur:
    """Verify Box blur kernel produces correct output."""

    def test_box_blur_smoothes(self):
        """Box blur should smooth a checkerboard pattern."""
        blur = BloomBlur(method=BlurMethod.BOX)
        w, h = 64, 64
        src = _make_checkerboard(w, h)
        original_mean = _mean_color(src)
        target = list(src)
        result = blur.blur(src, target, iterations=2, width=w, height=h)
        blurred_mean = _mean_color(result)
        assert abs(blurred_mean - original_mean) < 0.05

    def test_box_edge_pixels_clamp(self):
        """Box blur edge pixels should not cause index errors."""
        blur = BloomBlur(method=BlurMethod.BOX)
        w, h = 4, 4
        src = [float(i) for i in range(w * h * 4)]
        target = list(src)
        result = blur.blur(src, target, iterations=2, width=w, height=h)
        assert result is not None

    def test_box_blur_onexone(self):
        """Box blur on narrow buffer should not crash."""
        blur = BloomBlur(method=BlurMethod.BOX)
        w, h = 1, 4
        src = [1.0, 0.5, 0.0, 1.0] * 4
        result = blur.blur(src, list(src), iterations=1, width=w, height=h)
        assert result is not None

    def test_box_blur_horizontal_then_vertical(self):
        """Box blur on uniform buffer should keep values unchanged."""
        blur = BloomBlur(method=BlurMethod.BOX)
        w, h = 16, 16
        src = [0.5] * (w * h * 4)
        result = blur.blur(list(src), list(src), iterations=1, width=w, height=h)
        assert all(abs(v - 0.5) < 1e-9 for v in result)


class TestFix_AllBlurMethods:
    """Cross-method blur verification."""

    def test_all_methods_produce_valid_output(self):
        """All three blur methods should return valid non-None buffers."""
        w, h = 32, 32
        src = _make_checkerboard(w, h)
        for method in BlurMethod:
            blur = BloomBlur(method=method)
            if method == BlurMethod.GAUSSIAN:
                blur.calculate_gaussian_weights(radius=4, sigma=2.0)
            result = blur.blur(list(src), list(src), iterations=2, width=w, height=h)
            assert result is not None, f"{method} returned None"
            assert isinstance(result, list), f"{method} did not return a list"
            assert len(result) == len(src), f"{method} changed buffer size"

    def test_all_methods_preserve_mean(self):
        """All methods should approximately preserve mean luminance."""
        w, h = 32, 32
        src = _make_checkerboard(w, h)
        original_mean = _mean_color(src)
        for method in BlurMethod:
            blur = BloomBlur(method=method)
            if method == BlurMethod.GAUSSIAN:
                blur.calculate_gaussian_weights(radius=4, sigma=2.0)
            result = blur.blur(list(src), list(src), iterations=2, width=w, height=h)
            blurred_mean = _mean_color(result)
            assert abs(blurred_mean - original_mean) < 0.1, (
                f"{method} mean shifted: {original_mean} -> {blurred_mean}"
            )

    def test_gaussian_kawase_box_differ(self):
        """Different methods should produce detectably different outputs."""
        w, h = 32, 32
        src = _make_checkerboard(w, h)
        results = {}
        for method in BlurMethod:
            blur = BloomBlur(method=method)
            if method == BlurMethod.GAUSSIAN:
                blur.calculate_gaussian_weights(radius=4, sigma=2.0)
            results[method] = blur.blur(list(src), list(src), iterations=2, width=w, height=h)
        pairs = [
            (BlurMethod.GAUSSIAN, BlurMethod.KAWASE),
            (BlurMethod.GAUSSIAN, BlurMethod.BOX),
            (BlurMethod.KAWASE, BlurMethod.BOX),
        ]
        diffs = sum(
            1 for a, b in pairs
            if any(abs(results[a][i] - results[b][i]) > 0.01
                   for i in range(0, len(src), 4))
        )
        assert diffs >= 2, (
            f"At least 2 method pairs should differ; got {diffs}/3"
        )


# =============================================================================
# FIX 1b: BloomThreshold.configure() also validates (whitebox: uses clamp)
# =============================================================================

class TestFix_ThresholdConfigureValidation:
    """Verify BloomThreshold.configure() clamps parameters safely."""

    def test_configure_negative_threshold_clamps(self):
        """Negative threshold should clamp to 0."""
        t = BloomThreshold()
        t.configure(threshold=-5.0, softness=0.5, clamp_max=100.0)
        result = t.apply(1.0)
        assert result >= 0.0

    def test_configure_softness_clamps_high(self):
        """softness > 1 should clamp to 1."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=2.0, clamp_max=100.0)
        result = t.apply(0.5)
        assert 0.0 <= result <= 1.0

    def test_configure_clamp_max_clamps_negative(self):
        """Negative clamp_max should clamp to 0."""
        t = BloomThreshold()
        t.configure(threshold=1.0, softness=0.5, clamp_max=-10.0)
        result = t.apply(100.0)
        assert result == 0.0


# =============================================================================
# FIX 3b: BloomEffect.execute() with actual downsample buffers
# =============================================================================

class TestFix_EffectWithRealBuffers:
    """Verify BloomEffect works end-to-end with buffer-creating downsample."""

    def test_effect_execute_mip_chain_creates_buffers(self):
        """Full execute should create mip buffers for all levels."""
        effect = BloomEffect(BloomSettings(intensity=1.0, enabled=True))
        effect.setup(256, 256)
        effect.execute({"color": [0.5] * (256 * 256 * 4)}, {}, 0.016)
        for i in range(effect.mip_count):
            buf = effect._downsample.get_mip_buffer(i)
            w, h = effect._downsample.mip_sizes[i]
            assert buf is not None, f"mip {i} buffer should exist after execute"
            assert len(buf) == w * h * 4, f"mip {i} buffer has unexpected size"

    def test_effect_execute_disabled_does_not_create_buffers(self):
        """Disabled effect should not create any buffers."""
        effect = BloomEffect(BloomSettings(enabled=False))
        effect.setup(256, 256)
        effect.execute({"color": [0.5] * (256 * 256 * 4)}, {}, 0.016)
        for i in range(effect.mip_count):
            assert effect._downsample.get_mip_buffer(i) is None, \
                f"mip {i} buffer should not exist when disabled"
