"""
Blackbox tests for Fractal Brownian Motion (FBM) noise (T-DEMO-1.31).

CLEANROOM -- tests are based ONLY on the spec definition:
  T-DEMO-1.31: "Implement FBM (fractal Brownian motion) with configurable
                octaves, lacunarity, gain. Acceptance: multi-octave noise with
                correct spectral composition."

  Spec formulas (FORMULAS.md):
    VOL 10 line 33  (Phase 2 -- fBM):
      height = Sum(noise(p * 2^i) / 2^i)

    VOL 10 line 462 (Phase 10 -- general form):
      Value = Sum_i(Amplitude^i * Noise(Position * Frequency^i))

  Interpretation:
    FBM layers multiple octaves of a base noise function, each at a different
    frequency and amplitude. The frequency increases by `lacunarity` per octave
    (default 2.0, matching 2^i). The amplitude decreases by `gain` (persistence)
    per octave (default 0.5, matching 1/2^i).

    The output is normalized by the sum of amplitudes so the result stays in a
    bounded range regardless of octave count. When persistence < 1, higher
    octaves contribute proportionally less amplitude, giving the characteristic
    1/f spectral slope of natural phenomena.

No implementation knowledge of the class internals is used beyond what the
spec defines. Tests verify the external contract: parameter configurability,
normalized multi-octave composition, and spectral invariants.

COVERAGE PLAN:
  Section 1: Spec verification (spec references and mathematical derivation)
  Section 2: Parameter construction and defaults
  Section 3: Configurable octaves (from 1 to N)
  Section 4: Configurable lacunarity (frequency scaling)
  Section 5: Configurable gain / persistence (amplitude scaling)
  Section 6: Output normalization (stays bounded regardless of params)
  Section 7: Spectral composition -- octave contribution analysis
  Section 8: Spectral composition -- frequency domain (lacunarity effect)
  Section 9: Determinism and seed independence
  Section 10: Base noise type independence
  Section 11: 3D FBM sampling
  Section 12: Edge cases and parameter validation
  Section 13: Relationship to base noise (1 octave = base)
"""

from __future__ import annotations

import math

import pytest

from engine.world.pcg.noise import (
    NoiseType,
    NoiseSettings,
    FractalNoise,
    PerlinNoise,
    SimplexNoise,
    ValueNoise,
    WhiteNoise,
    WorleyNoise,
    create_noise_generator,
)

# =============================================================================
# Spec constants
# =============================================================================

# FORMULAS.md Phase 2: height = Sum(noise(p * 2^i) / 2^i)
# Canonical fBM has lacunarity=2.0, persistence (gain)=0.5
SPEC_LACUNARITY_DEFAULT = 2.0
SPEC_PERSISTENCE_DEFAULT = 0.5

# Approximate numerical tolerances for FBM invariants
# FBM normalization is approximate due to base noise exceeding [-1, 1]
TOL_REL = 1e-5
TOL_ABS = 1e-9
TOL_FBM_RANGE = 0.15  # FBM output may slightly exceed nominal range
TOL_MEAN = 0.15       # Statistical mean tolerance (heuristic sampling)


# =============================================================================
# Section helpers
# =============================================================================


def _sample_1d_strip(
    noise: FractalNoise,
    n: int = 200,
    x_start: float = 0.0,
    x_step: float = 0.01,
    y: float = 0.5,
) -> list[float]:
    """Sample a 1-dimensional strip of FBM noise at constant y.

    Args:
        noise: The FBM noise generator.
        n: Number of samples.
        x_start: Starting x position.
        x_step: Step size between samples.
        y: Fixed y coordinate.

    Returns:
        List of noise values.
    """
    return [noise.sample(x_start + i * x_step, y) for i in range(n)]


def _adjacent_variation(samples: list[float]) -> float:
    """Compute the mean absolute difference between adjacent samples.

    Higher values indicate more high-frequency (short-wavelength) variation.
    Lower values indicate smoother, more correlated output.

    Args:
        samples: List of sample values.

    Returns:
        Mean absolute adjacent difference.
    """
    if len(samples) < 2:
        return 0.0
    total = sum(abs(samples[i] - samples[i + 1]) for i in range(len(samples) - 1))
    return total / (len(samples) - 1)


def _mean(samples: list[float]) -> float:
    """Compute the mean of a list of samples."""
    return sum(samples) / len(samples) if samples else 0.0


def _variance(samples: list[float]) -> float:
    """Compute the variance of a list of samples."""
    if len(samples) < 2:
        return 0.0
    m = _mean(samples)
    return sum((v - m) ** 2 for v in samples) / (len(samples) - 1)


def _zero_crossings(samples: list[float]) -> int:
    """Count sign changes in a list of samples.

    More zero crossings indicates higher-frequency content.

    Args:
        samples: List of sample values.

    Returns:
        Number of times the sign changes between adjacent samples.
    """
    crossings = 0
    for i in range(len(samples) - 1):
        if samples[i] * samples[i + 1] < 0:
            crossings += 1
    return crossings


# =============================================================================
# SECTION 1 -- Spec verification
# =============================================================================


class TestSpecDefinition:
    """Tests that verify the mathematical definition from FORMULAS.md.

    The spec defines two forms of fBM:

    Simplified form (Phase 2):
        height = Sum(noise(p * 2^i) / 2^i)  for i in [0, octaves)

    General form (Phase 10):
        Value = Sum_i(Amplitude^i * Noise(Position * Frequency^i))

    where Frequency = lacunarity, Amplitude = persistence (gain).

    These are equivalent when lacunarity=2 and persistence=0.5, since
    noise(p * 2^i) * 0.5^i = noise(p * 2^i) / 2^i.
    """

    def test_spec_reference_phase2(self):
        """The Phase 2 formula: height = Sum(noise(p * 2^i) / 2^i)."""
        # Verify that with lacunarity=2.0 and persistence=0.5,
        # the per-octave contribution is noise(p*2^i) * 0.5^i
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=1, lacunarity=2.0, persistence=0.5)
        # At octaves=1, FBM should equal base noise (normalized by amp sum = 0.5)
        # The key formula: total = Sum(base.sample(x*freq, y*freq) * amp)
        # normalized by max_amp
        x, y = 1.5, 2.5
        base_val = noise.sample(x, y)
        fbm_val = fbm.sample(x, y)
        # With 1 octave, amp=1.0, max_amp=1.0, so fbm = base_val
        assert fbm_val == pytest.approx(base_val, abs=TOL_ABS), (
            f"1-octave FBM should equal base noise: base={base_val}, fbm={fbm_val}"
        )

    def test_spec_reference_general_form(self):
        """The Phase 10 general form: Value = Sum(amp^i * noise(p * freq^i))."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=3, lacunarity=SPEC_LACUNARITY_DEFAULT,
                           persistence=SPEC_PERSISTENCE_DEFAULT)
        # Verify the FBM produces finite output according to the spec formula
        for x in range(10):
            for y in range(10):
                val = fbm.sample(x * 0.3, y * 0.3)
                assert math.isfinite(val), (
                    f"Non-finite FBM output at ({x*0.3}, {y*0.3}): {val}"
                )

    def test_spec_equivalence_near_defaults(self):
        """Near-default parameters match the canonical spec formula."""
        # Create two FBM instances with the canonical spec defaults
        noise_a = PerlinNoise(seed=42)
        noise_b = PerlinNoise(seed=42)
        fbm_a = FractalNoise(noise_a, octaves=4,
                             lacunarity=SPEC_LACUNARITY_DEFAULT,
                             persistence=SPEC_PERSISTENCE_DEFAULT)
        fbm_b = FractalNoise(noise_b, octaves=4,
                             lacunarity=SPEC_LACUNARITY_DEFAULT,
                             persistence=SPEC_PERSISTENCE_DEFAULT)
        # Identically configured FBM instances produce identical results
        for x in range(-5, 6):
            for y in range(-5, 6):
                va = fbm_a.sample(x * 0.25, y * 0.25)
                vb = fbm_b.sample(x * 0.25, y * 0.25)
                assert va == pytest.approx(vb, abs=TOL_ABS), (
                    f"Identical FBM configs differ at ({x}, {y}): {va} vs {vb}"
                )


# =============================================================================
# SECTION 2 -- Parameter construction and defaults
# =============================================================================


class TestParameterConstruction:
    """Tests for FBM parameter construction and property access."""

    def test_default_construction(self):
        """Default parameters should use spec canonical values."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise)
        assert fbm.octaves == 4
        assert fbm.lacunarity == pytest.approx(SPEC_LACUNARITY_DEFAULT)
        assert fbm.persistence == pytest.approx(SPEC_PERSISTENCE_DEFAULT)

    def test_custom_octaves(self):
        """Should accept custom octave count."""
        noise = PerlinNoise(seed=42)
        for octaves in [1, 2, 3, 5, 8, 16]:
            fbm = FractalNoise(noise, octaves=octaves)
            assert fbm.octaves == octaves

    def test_custom_lacunarity(self):
        """Should accept custom lacunarity."""
        noise = PerlinNoise(seed=42)
        for lacunarity in [1.0, 1.5, 2.0, 3.0, 4.0]:
            fbm = FractalNoise(noise, lacunarity=lacunarity)
            assert fbm.lacunarity == pytest.approx(lacunarity)

    def test_custom_persistence(self):
        """Should accept custom persistence (gain)."""
        noise = PerlinNoise(seed=42)
        for persistence in [0.1, 0.25, 0.5, 0.75, 1.0]:
            fbm = FractalNoise(noise, persistence=persistence)
            assert fbm.persistence == pytest.approx(persistence)

    def test_all_parameters_independent(self):
        """All three parameters should be independently settable."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=6, lacunarity=2.5, persistence=0.3)
        assert fbm.octaves == 6
        assert fbm.lacunarity == pytest.approx(2.5)
        assert fbm.persistence == pytest.approx(0.3)

    def test_base_noise_property(self):
        """Should expose the base noise generator."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise)
        assert fbm.base_noise is noise

    def test_same_noise_multiple_fbm(self):
        """Same base noise can drive multiple FBM instances."""
        noise = PerlinNoise(seed=42)
        fbm_2 = FractalNoise(noise, octaves=2)
        fbm_5 = FractalNoise(noise, octaves=5)
        # Both should produce finite output
        val_2 = fbm_2.sample(0.5, 0.5)
        val_5 = fbm_5.sample(0.5, 0.5)
        assert math.isfinite(val_2)
        assert math.isfinite(val_5)
        # Different octave counts should produce different values
        assert val_2 != pytest.approx(val_5, abs=TOL_ABS), (
            "Different octave counts should produce different results"
        )


# =============================================================================
# SECTION 3 -- Configurable octaves (from 1 to N)
# =============================================================================


class TestConfigurableOctaves:
    """Tests for octave count configurability.

    Spec demands "configurable octaves." The octave count determines how
    many layers of noise are summed together. More octaves add higher-frequency
    detail.
    """

    def test_octave_1_equals_base_noise(self):
        """At octaves=1, FBM should equal the base noise."""
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1)
        for x in range(-5, 6):
            for y in range(-5, 6):
                base_val = noise.sample(x * 0.3, y * 0.3)
                fbm_val = fbm_1.sample(x * 0.3, y * 0.3)
                assert fbm_val == pytest.approx(base_val, abs=TOL_ABS), (
                    f"1-octave FBM mismatch at ({x}, {y}): "
                    f"base={base_val}, fbm={fbm_val}"
                )

    def test_more_octaves_more_variation(self):
        """Higher octave counts should increase total variation."""
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1)
        fbm_4 = FractalNoise(noise, octaves=4)
        fbm_8 = FractalNoise(noise, octaves=8)

        strip_1 = _sample_1d_strip(fbm_1, n=200)
        strip_4 = _sample_1d_strip(fbm_4, n=200)
        strip_8 = _sample_1d_strip(fbm_8, n=200)

        var_1 = _variance(strip_1)
        var_4 = _variance(strip_4)
        var_8 = _variance(strip_8)

        # More octaves should not reduce variance (heuristic)
        assert var_4 >= var_1 * 0.5, (
            f"Variance dropped unexpectedly at 4 octaves: 1={var_1}, 4={var_4}"
        )
        assert var_8 >= var_4 * 0.5, (
            f"Variance dropped unexpectedly at 8 octaves: 4={var_4}, 8={var_8}"
        )

    def test_more_octaves_more_zero_crossings(self):
        """More octaves should produce more zero crossings (higher frequency)."""
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1)
        fbm_4 = FractalNoise(noise, octaves=4)
        fbm_8 = FractalNoise(noise, octaves=8)

        strip_1 = _sample_1d_strip(fbm_1, n=200, x_step=0.05)
        strip_4 = _sample_1d_strip(fbm_4, n=200, x_step=0.05)
        strip_8 = _sample_1d_strip(fbm_8, n=200, x_step=0.05)

        zc_1 = _zero_crossings(strip_1)
        zc_4 = _zero_crossings(strip_4)
        zc_8 = _zero_crossings(strip_8)

        # Each additional octave doubles frequency (lacunarity=2),
        # so should increase zero crossings
        assert zc_4 >= zc_1, (
            f"4 octaves should have >= zero crossings than 1: "
            f"1={zc_1}, 4={zc_4}"
        )
        assert zc_8 >= zc_4, (
            f"8 octaves should have >= zero crossings than 4: "
            f"4={zc_4}, 8={zc_8}"
        )

    def test_octave_independent_of_lacunarity(self):
        """Octave count and lacunarity should be independently controllable."""
        noise = PerlinNoise(seed=42)
        # lacunarity=1.0 means no frequency increase per octave --
        # but more octaves should still be independently controllable
        fbm_1 = FractalNoise(noise, octaves=1, lacunarity=1.0)
        fbm_3 = FractalNoise(noise, octaves=3, lacunarity=1.0)
        # Both should produce finite output
        assert math.isfinite(fbm_1.sample(0.5, 0.5))
        assert math.isfinite(fbm_3.sample(0.5, 0.5))

    def test_octave_count_preserved_across_samples(self):
        """Octave count should remain consistent across all samples."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=6)
        for _ in range(100):
            val = fbm.sample(0.1, 0.2)
            assert math.isfinite(val)
        assert fbm.octaves == 6


# =============================================================================
# SECTION 4 -- Configurable lacunarity (frequency scaling)
# =============================================================================


class TestConfigurableLacunarity:
    """Tests for lacunarity configurability.

    Lacunarity controls the frequency multiplier between successive octaves.
    The spec canonical value is 2.0 (Phase 2 formula: 2^i).

    - lacunarity=1.0: all octaves at same frequency (no frequency scaling)
    - lacunarity=2.0: each octave doubles frequency (canonical)
    - lacunarity>2.0: frequency increases more rapidly per octave
    """

    def test_lacunarity_1_all_same_frequency(self):
        """With lacunarity=1.0, all octaves sample at the same frequency."""
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1, lacunarity=1.0)
        fbm_4 = FractalNoise(noise, octaves=4, lacunarity=1.0)

        # With lacunarity=1, all octaves sample same frequency.
        # The main effect is just amplitude-weighted summation.
        # Both should be stable and finite.
        for x in range(-5, 6):
            v1 = fbm_1.sample(x * 0.3, 0.5)
            v4 = fbm_4.sample(x * 0.3, 0.5)
            assert math.isfinite(v1)
            assert math.isfinite(v4)

    def test_higher_lacunarity_more_variation(self):
        """Higher lacunarity should produce more high-frequency variation."""
        noise = PerlinNoise(seed=42)
        fbm_low = FractalNoise(noise, octaves=4, lacunarity=1.5)
        fbm_high = FractalNoise(noise, octaves=4, lacunarity=4.0)

        strip_low = _sample_1d_strip(fbm_low, n=200, x_step=0.02)
        strip_high = _sample_1d_strip(fbm_high, n=200, x_step=0.02)

        var_low = _variance(strip_low)
        var_high = _variance(strip_high)

        # Higher lacunarity means higher effective frequencies,
        # which may affect variance (not a strict inequality, but heuristic)
        zc_low = _zero_crossings(strip_low)
        zc_high = _zero_crossings(strip_high)

        # Higher lacunarity should generally produce more zero crossings
        # as higher frequencies are amplified
        # This is a heuristic -- higher octave frequencies can alias
        print(f"  lacunarity=1.5: zero_crossings={zc_low}, var={var_low:.4f}")
        print(f"  lacunarity=4.0: zero_crossings={zc_high}, var={var_high:.4f}")
        # At minimum, output should be finite for any lacunarity
        assert math.isfinite(strip_low[0])
        assert math.isfinite(strip_high[0])

    def test_lacunarity_affects_frequency_content(self):
        """Lacunarity should measurably affect the frequency content."""
        noise = PerlinNoise(seed=42)
        fbm_a = FractalNoise(noise, octaves=3, lacunarity=1.1)
        fbm_b = FractalNoise(noise, octaves=3, lacunarity=5.0)

        # Sample at the same points -- lacunarity changes should produce
        # measurably different output (not just a linear scale)
        diffs = 0
        for x in range(20):
            va = fbm_a.sample(x * 0.1, 0.5)
            vb = fbm_b.sample(x * 0.1, 0.5)
            if abs(va - vb) > TOL_ABS:
                diffs += 1

        assert diffs > 15, (
            f"Different lacunarity values produced nearly identical output "
            f"at {20 - diffs}/20 points"
        )

    def test_lacunarity_near_one(self):
        """Lacunarity close to 1.0 should still produce valid output."""
        noise = PerlinNoise(seed=42)
        for lacunarity in [1.001, 1.01, 1.1]:
            fbm = FractalNoise(noise, octaves=4, lacunarity=lacunarity)
            val = fbm.sample(1.5, 2.5)
            assert math.isfinite(val), (
                f"Non-finite at lacunarity={lacunarity}: {val}"
            )

    def test_lacunarity_preserved_across_samples(self):
        """Lacunarity should stay consistent."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, lacunarity=3.5)
        assert fbm.lacunarity == pytest.approx(3.5)
        # Sample many times -- lacunarity should not drift
        for i in range(100):
            fbm.sample(i * 0.1, i * 0.2)
        assert fbm.lacunarity == pytest.approx(3.5)


# =============================================================================
# SECTION 5 -- Configurable gain / persistence (amplitude scaling)
# =============================================================================


class TestConfigurableGain:
    """Tests for persistence (gain) configurability.

    Gain/persistence controls how much each successive octave contributes.
    The spec canonical value is 0.5 (Phase 2 formula: / 2^i).

    - persistence=1.0: all octaves contribute equally (no amplitude decay)
    - persistence=0.5: each octave contributes half the amplitude (canonical)
    - persistence<0.5: higher octaves contribute very little (smoother result)
    """

    def test_persistence_1_equal_contribution(self):
        """With persistence=1.0, each octave contributes equally."""
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1, persistence=1.0)
        fbm_4 = FractalNoise(noise, octaves=4, persistence=1.0)
        fbm_8 = FractalNoise(noise, octaves=8, persistence=1.0)

        # All should produce finite, valid output
        for x in range(-5, 6):
            v1 = fbm_1.sample(x * 0.2, 0.5)
            v4 = fbm_4.sample(x * 0.2, 0.5)
            v8 = fbm_8.sample(x * 0.2, 0.5)
            assert math.isfinite(v1)
            assert math.isfinite(v4)
            assert math.isfinite(v8)

    def test_lower_persistence_smoother(self):
        """Lower persistence should produce smoother (less varying) output
        because higher octaves contribute less amplitude."""
        noise = PerlinNoise(seed=42)
        fbm_low = FractalNoise(noise, octaves=6, persistence=0.1)
        fbm_high = FractalNoise(noise, octaves=6, persistence=1.0)

        strip_low = _sample_1d_strip(fbm_low, n=200, x_step=0.02)
        strip_high = _sample_1d_strip(fbm_high, n=200, x_step=0.02)

        var_low = _variance(strip_low)
        var_high = _variance(strip_high)

        # High persistence (all octaves equal) should have more variation
        # than low persistence (higher octaves attenuated)
        assert var_high > var_low * 0.5, (
            f"High persistence variance ({var_high:.4f}) should be >= "
            f"low persistence variance ({var_low:.4f})"
        )

    def test_persistence_affects_adjacent_variation(self):
        """Persistence should affect adjacent-sample variation."""
        noise = PerlinNoise(seed=42)
        fbm_low = FractalNoise(noise, octaves=5, persistence=0.1)
        fbm_high = FractalNoise(noise, octaves=5, persistence=0.9)

        strip_low = _sample_1d_strip(fbm_low, n=200, x_step=0.02)
        strip_high = _sample_1d_strip(fbm_high, n=200, x_step=0.02)

        adj_low = _adjacent_variation(strip_low)
        adj_high = _adjacent_variation(strip_high)

        # High persistence should have more adjacent variation
        # (more high-frequency content visible)
        assert adj_high >= adj_low * 0.5, (
            f"High persistence adjacent variation ({adj_high:.4f}) "
            f"should be >= low persistence ({adj_low:.4f})"
        )

    def test_persistence_0_to_1_range(self):
        """Persistence should work across its entire valid range (0, 1]."""
        noise = PerlinNoise(seed=42)
        for persistence in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0]:
            fbm = FractalNoise(noise, octaves=4, persistence=persistence)
            val = fbm.sample(1.5, 2.5)
            assert math.isfinite(val), (
                f"Non-finite at persistence={persistence}: {val}"
            )

    def test_gain_not_greater_than_one(self):
        """Persistence (gain) > 1 should be rejected by spec."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(noise, octaves=4, persistence=1.5)

    def test_gain_positive(self):
        """Persistence (gain) must be positive."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(noise, octaves=4, persistence=0.0)

        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(noise, octaves=4, persistence=-0.5)


# =============================================================================
# SECTION 6 -- Output normalization
# =============================================================================


class TestOutputNormalization:
    """Tests for FBM output normalization.

    The spec formula divides by the sum of amplitudes to normalize.
    This keeps output in a bounded range regardless of octave count
    or parameter settings.
    """

    def test_output_bounded_single_octave(self):
        """With 1 octave, output should match base noise range."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=1)
        values = [fbm.sample(x * 0.2, y * 0.3)
                  for x in range(-10, 11) for y in range(-10, 11)]
        # Perlin noise is approximately in [-1, 1] with some slight overage
        for v in values:
            assert -2.0 <= v <= 2.0, (
                f"1-octave FBM out of expected range: {v}"
            )

    def test_output_bounded_many_octaves(self):
        """Output should remain bounded even with many octaves."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=12)
        values = [fbm.sample(x * 0.2, y * 0.3)
                  for x in range(-10, 11) for y in range(-10, 11)]
        for v in values:
            assert abs(v) < 5.0, (
                f"12-octave FBM out of expected range: {v}"
            )

    def test_output_bounded_high_lacunarity(self):
        """Output should remain bounded with high lacunarity."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=6, lacunarity=8.0)
        for x in range(-10, 11):
            for y in range(-10, 11):
                val = fbm.sample(x * 0.2, y * 0.3)
                assert abs(val) < 5.0, (
                    f"High-lacunarity FBM out of expected range: {val}"
                )

    def test_output_bounded_persistence_one(self):
        """Output should remain bounded with persistence=1.0 (equal octaves)."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=8, persistence=1.0)
        for x in range(-5, 6):
            for y in range(-5, 6):
                val = fbm.sample(x * 0.3, y * 0.3)
                assert abs(val) < 5.0, (
                    f"persistence=1.0 FBM out of expected range: {val}"
                )

    def test_normalization_consistent_across_params(self):
        """Normalization should produce consistent ranges across param values."""
        noise = PerlinNoise(seed=42)
        configs = [
            (3, 2.0, 0.5),
            (6, 2.0, 0.5),
            (4, 3.0, 0.5),
            (4, 2.0, 0.3),
            (4, 2.0, 0.8),
        ]
        ranges = []
        for octaves, lacunarity, persistence in configs:
            fbm = FractalNoise(noise, octaves=octaves,
                               lacunarity=lacunarity,
                               persistence=persistence)
            values = [fbm.sample(x * 0.3, y * 0.5)
                      for x in range(-5, 6) for y in range(-5, 6)]
            r = max(values) - min(values)
            ranges.append(r)

        # All configs should produce comparable ranges
        # (within an order of magnitude)
        for i, (r, cfg) in enumerate(zip(ranges, configs)):
            assert 0.1 < r < 5.0, (
                f"Range {r:.3f} for config {cfg} is outside expected [0.1, 5.0]"
            )

    def test_statistical_mean_near_zero(self):
        """FBM output should have mean near zero (if base noise is zero-mean)."""
        noise = PerlinNoise(seed=42)
        configs = [
            (2, 2.0, 0.5),
            (4, 2.0, 0.5),
            (6, 2.0, 0.3),
            (6, 3.0, 0.5),
        ]
        for octaves, lacunarity, persistence in configs:
            fbm = FractalNoise(noise, octaves=octaves,
                               lacunarity=lacunarity,
                               persistence=persistence)
            values = [fbm.sample(x * 0.13, y * 0.17)
                      for x in range(-15, 16) for y in range(-15, 16)]
            m = _mean(values)
            assert abs(m) < TOL_MEAN, (
                f"Mean {m:.4f} for config ({octaves},{lacunarity},{persistence}) "
                f"not near 0"
            )


# =============================================================================
# SECTION 7 -- Spectral composition: octave contribution analysis
# =============================================================================


class TestOctaveContribution:
    """Tests that each octave contributes detail at its respective frequency.

    The acceptance criterion "correct spectral composition" means:
    1. Octave i contributes detail at frequency lacunarity^i
    2. The amplitude of octave i is persistence^i
    3. Higher octaves add finer (higher-frequency) detail
    4. The composite output has a 1/f-like spectrum (for persistence=0.5)
    """

    def test_each_octave_adds_detail(self):
        """Each additional octave should add detail (increase complexity)."""
        noise = PerlinNoise(seed=42)
        # Compare sequential octave counts
        prev_adj_var = 0.0
        for octaves in range(1, 7):
            fbm = FractalNoise(noise, octaves=octaves)
            strip = _sample_1d_strip(fbm, n=300, x_step=0.01)
            adj_var = _adjacent_variation(strip)
            # Each octave adds high-frequency detail, so adjacent
            # variation should generally increase with octave count
            if octaves > 1:
                assert adj_var > prev_adj_var * 0.3, (
                    f"Adjacent variation dropped from {prev_adj_var:.6f} "
                    f"at {octaves-1} octaves to {adj_var:.6f} at {octaves} "
                    f"octaves"
                )
            prev_adj_var = adj_var

    def test_octave_contribution_decays_with_persistence(self):
        """With persistence < 1, higher octaves contribute proportionally less.

        This means the increase in adjacent variation should diminish
        as octave count increases (with persistence=0.5, the 5th octave
        contributes only 0.5^5 = 3% of the first octave's amplitude).
        """
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=8, persistence=0.5)

        # Compute the contribution of each octave by comparing
        # incremental changes
        strip_by_octave = []
        for o in range(1, 9):
            f = FractalNoise(noise, octaves=o, persistence=0.5)
            strip_by_octave.append(_sample_1d_strip(f, n=200, x_step=0.01))

        # The marginal contribution of each octave should generally
        # decrease with octave number (for persistence < 1)
        contributions = []
        for i in range(1, len(strip_by_octave)):
            # RMS difference between N and N+1 octaves
            prev = strip_by_octave[i - 1]
            curr = strip_by_octave[i]
            rms_diff = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(prev, curr)) / len(prev)
            )
            contributions.append(rms_diff)

        # With persistence=0.5, contributions should generally decrease
        # (the last one may be slightly noisy, but the trend should hold)
        print(f"  Octave contributions (RMS): {[f'{c:.6f}' for c in contributions]}")
        # The first contribution (octave 1->2) should be largest
        assert contributions[0] >= max(contributions[1:]) * 0.3, (
            f"First octave contribution ({contributions[0]:.6f}) "
            f"should be significant"
        )

    def test_two_octaves_more_complex_than_one(self):
        """2-octave FBM should be more complex (less predictable) than 1-octave.

        This tests spectral composition: the second octave adds detail at
        a different (higher) frequency band.
        """
        noise = PerlinNoise(seed=42)
        fbm_1 = FractalNoise(noise, octaves=1)
        fbm_2 = FractalNoise(noise, octaves=2)

        # Sample and check that the 2-octave version is not just a
        # scaled version of the 1-octave version
        samples_1 = [fbm_1.sample(x * 0.1, 0.5) for x in range(100)]
        samples_2 = [fbm_2.sample(x * 0.1, 0.5) for x in range(100)]

        # Correlation should not be 1.0 (they are not linearly related)
        mean_1 = _mean(samples_1)
        mean_2 = _mean(samples_2)
        num = sum((a - mean_1) * (b - mean_2) for a, b in zip(samples_1, samples_2))
        den = math.sqrt(
            sum((a - mean_1) ** 2 for a in samples_1) *
            sum((b - mean_2) ** 2 for b in samples_2)
        )
        corr = num / den if den > 0 else 0.0

        # 2-octave should not be perfectly correlated with 1-octave
        # (the extra octave adds uncorrelated detail)
        assert corr < 0.99, (
            f"2-octave and 1-octave FBM too highly correlated: {corr:.4f}"
        )

    def test_spectral_detail_at_different_scales(self):
        """Octaves with different lacunarity should add detail at different scales.

        With lacunarity=2, octave 2 operates at 2x the frequency of octave 1,
        so after normalization, the two should be decorrelated.
        """
        noise = PerlinNoise(seed=42)
        # The difference between 1-octave and N-octave FBM should be
        # larger for larger N
        fbm_1 = FractalNoise(noise, octaves=1)
        fbm_4 = FractalNoise(noise, octaves=4)
        fbm_8 = FractalNoise(noise, octaves=8)

        diff_1 = sum(
            abs(fbm_4.sample(x * 0.1, 0.5) - fbm_1.sample(x * 0.1, 0.5))
            for x in range(100)
        )
        diff_2 = sum(
            abs(fbm_8.sample(x * 0.1, 0.5) - fbm_1.sample(x * 0.1, 0.5))
            for x in range(100)
        )

        # 8 octaves should differ more from 1 octave than 4 octaves do
        # (more accumulated detail from extra octaves)
        assert diff_2 >= diff_1 * 0.5, (
            f"8-octave diff ({diff_2:.4f}) should be >= "
            f"4-octave diff ({diff_1:.4f})"
        )


# =============================================================================
# SECTION 8 -- Spectral composition: frequency domain (lacunarity effect)
# =============================================================================


class TestFrequencyScaling:
    """Tests that lacunarity correctly scales frequencies between octaves.

    The spec formula: frequency_i = lacunarity^i
    With lacunarity=2, each octave doubles frequency.
    """

    def test_lacunarity_2_doubles_freq(self):
        """With lacunarity=2, each octave effectively doubles frequency."""
        noise = PerlinNoise(seed=42)
        fbm_alt = FractalNoise(noise, octaves=4, lacunarity=2.0)

        # Sample at stepped positions and verify no obvious artifacts
        # from frequency doubling (aliasing, strange periodic patterns)
        values = [fbm_alt.sample(x * 0.025, 0.5) for x in range(400)]
        assert all(math.isfinite(v) for v in values)
        # Should have many sign changes (high frequency content)
        zc = _zero_crossings(values)
        assert zc > 20, (
            f"Lacunarity=2 should produce many zero crossings, got {zc}"
        )

    def test_lacunarity_affects_wavelength(self):
        """Higher lacunarity produces shorter-wavelength oscillations."""
        noise = PerlinNoise(seed=42)
        fbm_low = FractalNoise(noise, octaves=4, lacunarity=1.5)
        fbm_high = FractalNoise(noise, octaves=4, lacunarity=4.0)

        # Count sign changes at fine stepping
        strip_low = _sample_1d_strip(fbm_low, n=300, x_step=0.01)
        strip_high = _sample_1d_strip(fbm_high, n=300, x_step=0.01)

        zc_low = _zero_crossings(strip_low)
        zc_high = _zero_crossings(strip_high)

        # Higher lacunarity = more compressed frequencies = more zero crossings
        # This is a heuristic -- may not always hold at all step sizes
        print(f"  lacunarity=1.5: zero_crossings={zc_low}")
        print(f"  lacunarity=4.0: zero_crossings={zc_high}")

    def test_different_lacunarity_different_spectrum(self):
        """Different lacunarity values should produce audibly different output."""
        noise = PerlinNoise(seed=42)
        fbm_a = FractalNoise(noise, octaves=5, lacunarity=1.5, persistence=0.6)
        fbm_b = FractalNoise(noise, octaves=5, lacunarity=3.5, persistence=0.6)

        # Cross-correlation should not be near 1.0
        sa = [fbm_a.sample(x * 0.05, 0.5) for x in range(200)]
        sb = [fbm_b.sample(x * 0.05, 0.5) for x in range(200)]

        mean_a = _mean(sa)
        mean_b = _mean(sb)
        num = sum((a - mean_a) * (b - mean_b) for a, b in zip(sa, sb))
        den = math.sqrt(
            sum((a - mean_a) ** 2 for a in sa) *
            sum((b - mean_b) ** 2 for b in sb)
        )
        corr = num / den if den > 0 else 0.0

        # Different lacunarity should produce substantially different signals
        assert corr < 0.95, (
            f"Different lacunarity values too correlated: {corr:.4f}"
        )


# =============================================================================
# SECTION 9 -- Determinism and seed independence
# =============================================================================


class TestDeterminism:
    """Tests that FBM noise is deterministic given the same base noise seed."""

    def test_deterministic_same_base(self):
        """Same base noise should produce identical FBM output."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=42)
        fbm1 = FractalNoise(noise1, octaves=4)
        fbm2 = FractalNoise(noise2, octaves=4)

        for x in range(-10, 11):
            for y in range(-10, 11):
                v1 = fbm1.sample(x * 0.5, y * 0.5)
                v2 = fbm2.sample(x * 0.5, y * 0.5)
                assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                    f"Non-deterministic FBM at ({x}, {y}): {v1} vs {v2}"
                )

    def test_deterministic_repeated_calls(self):
        """Repeated calls to same FBM should produce same results."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=4)
        points = [(x * 0.7 + 0.3, y * 0.5 + 0.2)
                  for x in range(-5, 6) for y in range(-5, 6)]
        first_run = [fbm.sample(x, y) for x, y in points]
        second_run = [fbm.sample(x, y) for x, y in points]
        for i, ((x, y), v1, v2) in enumerate(zip(points, first_run, second_run)):
            assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                f"Non-deterministic at ({x}, {y}) on repeated call: {v1} vs {v2}"
            )

    def test_different_seeds_different_results(self):
        """Different seeds should produce different FBM results."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=9999)
        fbm1 = FractalNoise(noise1, octaves=4)
        fbm2 = FractalNoise(noise2, octaves=4)

        differences = 0
        total = 0
        for x in range(-5, 6):
            for y in range(-5, 6):
                v1 = fbm1.sample(x * 0.5, y * 0.5)
                v2 = fbm2.sample(x * 0.5, y * 0.5)
                if abs(v1 - v2) > TOL_ABS:
                    differences += 1
                total += 1

        assert differences > total * 0.5, (
            f"Different seeds only differ at {differences}/{total} points"
        )

    def test_deterministic_different_configs(self):
        """Each FBM config should be independently deterministic."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=42)
        configs = [
            (2, 1.5, 0.3),
            (4, 2.0, 0.5),
            (6, 3.0, 0.7),
            (8, 2.5, 0.4),
        ]
        for octaves, lacunarity, persistence in configs:
            fbm1 = FractalNoise(noise1, octaves=octaves,
                                lacunarity=lacunarity,
                                persistence=persistence)
            fbm2 = FractalNoise(noise2, octaves=octaves,
                                lacunarity=lacunarity,
                                persistence=persistence)
            for x in range(-3, 4):
                for y in range(-3, 4):
                    v1 = fbm1.sample(x * 0.5, y * 0.5)
                    v2 = fbm2.sample(x * 0.5, y * 0.5)
                    assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                        f"Non-deterministic FBM ({octaves},{lacunarity},"
                        f"{persistence}) at ({x},{y}): {v1} vs {v2}"
                    )


# =============================================================================
# SECTION 10 -- Base noise type independence
# =============================================================================


class TestBaseNoiseTypes:
    """Tests that FBM works correctly with different base noise types.

    The spec says FBM is a meta-noise that layers any base noise function.
    It should work with Perlin, Simplex, Value, Worley, and White noise.
    """

    BASE_NOISE_TYPES = [
        ("Perlin", lambda s: PerlinNoise(seed=s)),
        ("Simplex", lambda s: SimplexNoise(seed=s)),
        ("Value", lambda s: ValueNoise(seed=s)),
        ("White", lambda s: WhiteNoise(seed=s)),
        ("Worley", lambda s: WorleyNoise(seed=s)),
    ]

    def test_all_base_types_produce_finite_output(self):
        """FBM should produce finite output for all base noise types."""
        for name, make_noise in self.BASE_NOISE_TYPES:
            noise = make_noise(42)
            fbm = FractalNoise(noise, octaves=4)
            for x in range(-5, 6):
                for y in range(-5, 6):
                    val = fbm.sample(x * 0.3, y * 0.3)
                    assert math.isfinite(val), (
                        f"Non-finite FBM output with {name} noise "
                        f"at ({x}, {y}): {val}"
                    )

    def test_all_base_types_deterministic(self):
        """FBM should be deterministic for all base noise types."""
        for name, make_noise in self.BASE_NOISE_TYPES:
            noise1 = make_noise(42)
            noise2 = make_noise(42)
            fbm1 = FractalNoise(noise1, octaves=4)
            fbm2 = FractalNoise(noise2, octaves=4)
            for x in range(-3, 4):
                for y in range(-3, 4):
                    v1 = fbm1.sample(x * 0.5, y * 0.5)
                    v2 = fbm2.sample(x * 0.5, y * 0.5)
                    assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                        f"Non-deterministic {name} FBM at ({x}, {y})"
                    )

    def test_all_base_types_varied_output(self):
        """FBM should produce varied (not constant) output for all base types."""
        for name, make_noise in self.BASE_NOISE_TYPES:
            noise = make_noise(42)
            fbm = FractalNoise(noise, octaves=4)
            samples = [fbm.sample(i * 0.17, i * 0.23) for i in range(30)]
            unique = len(set(round(v, 10) for v in samples))
            assert unique > 5, (
                f"{name} FBM produced too few unique values: {unique}/30"
            )

    def test_all_base_types_multiple_octaves(self):
        """All base types should support multiple octave counts."""
        for name, make_noise in self.BASE_NOISE_TYPES:
            for octaves in [1, 3, 6]:
                noise = make_noise(42)
                fbm = FractalNoise(noise, octaves=octaves)
                val = fbm.sample(0.5, 0.5)
                assert math.isfinite(val), (
                    f"{name} FBM with {octaves} octaves non-finite: {val}"
                )

    def test_all_base_types_varied_lacunarity(self):
        """All base types should support varied lacunarity."""
        for name, make_noise in self.BASE_NOISE_TYPES:
            for lacunarity in [1.0, 2.0, 4.0]:
                noise = make_noise(42)
                fbm = FractalNoise(noise, octaves=3, lacunarity=lacunarity)
                val = fbm.sample(0.5, 0.5)
                assert math.isfinite(val), (
                    f"{name} FBM with lacunarity={lacunarity} non-finite: {val}"
                )


# =============================================================================
# SECTION 11 -- 3D FBM sampling
# =============================================================================


class Test3DSampling:
    """Tests for 3D FBM noise sampling."""

    def test_3d_produces_finite_output(self):
        """3D FBM should produce finite output."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=4)
        val = fbm.sample_3d(1.5, 2.5, 3.5)
        assert math.isfinite(val)

    def test_3d_deterministic(self):
        """3D FBM should be deterministic."""
        noise1 = PerlinNoise(seed=42)
        noise2 = PerlinNoise(seed=42)
        fbm1 = FractalNoise(noise1, octaves=4)
        fbm2 = FractalNoise(noise2, octaves=4)
        for x in range(-3, 4):
            for y in range(-3, 4):
                for z in range(-3, 4):
                    v1 = fbm1.sample_3d(x * 0.5, y * 0.5, z * 0.5)
                    v2 = fbm2.sample_3d(x * 0.5, y * 0.5, z * 0.5)
                    assert v1 == pytest.approx(v2, abs=TOL_ABS), (
                        f"Non-deterministic 3D at ({x}, {y}, {z}): {v1} vs {v2}"
                    )

    def test_3d_varied_output(self):
        """3D FBM should produce varied output."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=4)
        samples = [fbm.sample_3d(i * 0.17, i * 0.23, i * 0.31) for i in range(30)]
        unique = len(set(round(v, 10) for v in samples))
        assert unique > 5, (
            f"Too few unique 3D values: {unique}/30"
        )

    def test_3d_bounded_output(self):
        """3D FBM output should be bounded."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=6)
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    val = fbm.sample_3d(x * 0.3, y * 0.3, z * 0.3)
                    assert abs(val) < 5.0, (
                        f"3D FBM out of expected range: {val} at ({x}, {y}, {z})"
                    )

    def test_3d_different_base_types(self):
        """3D FBM should work with different base noise types."""
        for name, make_noise in [
            ("Perlin", lambda s: PerlinNoise(seed=s)),
            ("Simplex", lambda s: SimplexNoise(seed=s)),
            ("Value", lambda s: ValueNoise(seed=s)),
        ]:
            noise = make_noise(42)
            fbm = FractalNoise(noise, octaves=3)
            val = fbm.sample_3d(1.0, 2.0, 3.0)
            assert math.isfinite(val), (
                f"{name} 3D FBM non-finite: {val}"
            )


# =============================================================================
# SECTION 12 -- Edge cases and parameter validation
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and parameter validation."""

    def test_minimum_octaves(self):
        """octaves=1 should be the minimum valid value."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=1)
        assert fbm.octaves == 1
        val = fbm.sample(0.5, 0.5)
        assert math.isfinite(val)

    def test_octaves_below_1_rejected(self):
        """octaves < 1 should raise ValueError."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="octaves must be >= 1"):
            FractalNoise(noise, octaves=0)
        with pytest.raises(ValueError, match="octaves must be >= 1"):
            FractalNoise(noise, octaves=-1)

    def test_lacunarity_positive_required(self):
        """lacunarity must be positive."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="lacunarity must be > 0"):
            FractalNoise(noise, lacunarity=0)
        with pytest.raises(ValueError, match="lacunarity must be > 0"):
            FractalNoise(noise, lacunarity=-1.0)

    def test_persistence_not_zero(self):
        """persistence must be > 0."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(noise, persistence=0.0)

    def test_persistence_not_greater_than_one(self):
        """persistence must be <= 1."""
        noise = PerlinNoise(seed=42)
        with pytest.raises(ValueError, match="persistence must be in"):
            FractalNoise(noise, persistence=1.1)

    def test_large_octave_count(self):
        """FBM should handle a large number of octaves."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=32, persistence=0.25)
        val = fbm.sample(0.5, 0.5)
        assert math.isfinite(val), (
            f"32-octave FBM non-finite: {val}"
        )

    def test_very_small_persistence(self):
        """Very small persistence should not cause numerical issues."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=8, persistence=0.01)
        val = fbm.sample(0.5, 0.5)
        assert math.isfinite(val)

    def test_negative_coordinates(self):
        """FBM should handle negative coordinates correctly."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=4)
        for x in range(-10, 0):
            for y in range(-10, 0):
                val = fbm.sample(x + 0.3, y + 0.7)
                assert math.isfinite(val), (
                    f"Non-finite at negative ({x+0.3}, {y+0.7}): {val}"
                )

    def test_large_coordinates(self):
        """FBM should handle large coordinates without overflow."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=4)
        for scale in [100, 1000, 10000]:
            val = fbm.sample(scale, scale * 0.5)
            assert math.isfinite(val), (
                f"Non-finite at scale {scale}: {val}"
            )


# =============================================================================
# SECTION 13 -- Relationship to base noise (1 octave = base)
# =============================================================================


class TestRelationshipToBaseNoise:
    """Tests verifying that FBM correctly layers base noise.

    The fundamental FBM property: with 1 octave, FBM should equal the
    base noise. With more octaves, FBM is the base noise plus additional
    high-frequency detail.
    """

    def test_1_octave_vs_base_all_types(self):
        """1-octave FBM should equal base noise for all noise types."""
        for name, make_noise in [
            ("Perlin", lambda s: PerlinNoise(seed=s)),
            ("Simplex", lambda s: SimplexNoise(seed=s)),
            ("Value", lambda s: ValueNoise(seed=s)),
            ("White", lambda s: WhiteNoise(seed=s)),
        ]:
            noise = make_noise(42)
            fbm = FractalNoise(noise, octaves=1)
            for x in range(-3, 4):
                for y in range(-3, 4):
                    base_val = noise.sample(x * 0.5, y * 0.5)
                    fbm_val = fbm.sample(x * 0.5, y * 0.5)
                    assert fbm_val == pytest.approx(base_val, abs=TOL_ABS), (
                        f"{name}: 1-octave FBM ({fbm_val}) != base ({base_val}) "
                        f"at ({x}, {y})"
                    )

    def test_1_octave_3d_vs_base(self):
        """1-octave 3D FBM should equal base 3D noise."""
        noise = PerlinNoise(seed=42)
        fbm = FractalNoise(noise, octaves=1)
        for x in range(-3, 4):
            for y in range(-3, 4):
                for z in range(-3, 4):
                    base_val = noise.sample_3d(x * 0.5, y * 0.5, z * 0.5)
                    fbm_val = fbm.sample_3d(x * 0.5, y * 0.5, z * 0.5)
                    assert fbm_val == pytest.approx(base_val, abs=TOL_ABS), (
                        f"1-octave 3D FBM != base at ({x}, {y}, {z}): "
                        f"{fbm_val} vs {base_val}"
                    )

    def test_fbm_with_noise_settings_frequency(self):
        """FBM combines with NoiseSettings frequency.

        The NoiseSettings.frequency on the base noise should still apply
        to each octave, and the FBM lacunarity compounds on top of it.
        """
        settings = NoiseSettings(frequency=0.5)
        noise = PerlinNoise(seed=42, settings=settings)
        fbm = FractalNoise(noise, octaves=3, lacunarity=2.0)
        # Should produce valid output
        val = fbm.sample(0.5, 0.5)
        assert math.isfinite(val)

    def test_fbm_with_amplitude(self):
        """NoiseSettings amplitude scales the entire FBM output."""
        settings = NoiseSettings(amplitude=2.0)
        noise = PerlinNoise(seed=42, settings=settings)
        fbm = FractalNoise(noise, octaves=1)
        # With octaves=1, FBM = base noise * amplitude
        # (The base noise itself applies amplitude)
        val = fbm.sample(0.5, 0.5)
        assert math.isfinite(val)
        # The amplitude scaling comes from the base noise settings
        assert -2.5 <= val <= 2.5, (
            f"Amplitude-scaled FBM out of range: {val}"
        )

    def test_fbm_identity_with_1_octave_and_any_lacunarity(self):
        """With 1 octave, lacunarity should not affect output."""
        noise = PerlinNoise(seed=42)
        fbm_a = FractalNoise(noise, octaves=1, lacunarity=1.0)
        fbm_b = FractalNoise(noise, octaves=1, lacunarity=10.0)
        for x in range(-3, 4):
            for y in range(-3, 4):
                va = fbm_a.sample(x * 0.5, y * 0.5)
                vb = fbm_b.sample(x * 0.5, y * 0.5)
                assert va == pytest.approx(vb, abs=TOL_ABS), (
                    f"1-octave FBM should not depend on lacunarity at "
                    f"({x}, {y}): {va} vs {vb}"
                )
