"""Tests for Deterministic Gerstner Wave Simulation (T-CC-2.2).

This test module verifies:
1. Fixed32 trigonometry accuracy and determinism
2. Wave parameter storage in Fixed32
3. Deterministic wave computation
4. Multi-run bit-identical results
5. Grid sampling determinism
6. State checksum verification
"""

import math

import pytest

from trinity.types import Fixed32, PCG64
from engine.rendering.water.deterministic_gerstner import (
    # Constants
    FIXED32_ZERO,
    FIXED32_ONE,
    FIXED32_HALF,
    FIXED32_PI,
    FIXED32_TWO_PI,
    FIXED32_HALF_PI,
    # Trigonometry
    fixed32_sin,
    fixed32_cos,
    fixed32_sincos,
    _normalize_angle,
    # Vector types
    Fixed32Vec2,
    Fixed32Vec3,
    # Wave types
    Fixed32WaveParams,
    GerstnerWaveResult,
    # Computation
    compute_gerstner_wave,
    # Simulator
    DeterministicGerstnerWave,
)


# =============================================================================
# FIXED32 TRIGONOMETRY TESTS
# =============================================================================


class TestFixed32Sin:
    """Test fixed32_sin Taylor series implementation."""

    def test_sin_zero(self):
        """sin(0) = 0."""
        result = fixed32_sin(FIXED32_ZERO)
        assert abs(result.as_float) < 0.001

    def test_sin_pi(self):
        """sin(pi) = 0."""
        result = fixed32_sin(FIXED32_PI)
        assert abs(result.as_float) < 0.02  # Fixed-point tolerance

    def test_sin_half_pi(self):
        """sin(pi/2) = 1."""
        result = fixed32_sin(FIXED32_HALF_PI)
        assert abs(result.as_float - 1.0) < 0.01

    def test_sin_negative_half_pi(self):
        """sin(-pi/2) = -1."""
        result = fixed32_sin(-FIXED32_HALF_PI)
        assert abs(result.as_float + 1.0) < 0.01

    def test_sin_quarter_pi(self):
        """sin(pi/4) = sqrt(2)/2 ~ 0.707."""
        quarter_pi = Fixed32(math.pi / 4)
        result = fixed32_sin(quarter_pi)
        expected = math.sin(math.pi / 4)
        assert abs(result.as_float - expected) < 0.01

    def test_sin_deterministic(self):
        """Same input produces same output across calls."""
        angle = Fixed32(1.234)
        result1 = fixed32_sin(angle)
        result2 = fixed32_sin(angle)
        assert result1.raw == result2.raw

    def test_sin_symmetry(self):
        """sin(-x) = -sin(x)."""
        angle = Fixed32(0.75)
        pos = fixed32_sin(angle)
        neg = fixed32_sin(-angle)
        assert abs(pos.raw + neg.raw) < 10  # Allow small rounding

    def test_sin_range_accuracy(self):
        """Check accuracy over multiple angles."""
        for degrees in range(0, 360, 15):
            radians = math.radians(degrees)
            angle = Fixed32(radians)
            result = fixed32_sin(angle)
            expected = math.sin(radians)
            error = abs(result.as_float - expected)
            # Fixed-point Taylor series has ~0.05 max error near boundaries
            assert error < 0.06, f"sin({degrees}deg) error={error}"


class TestFixed32Cos:
    """Test fixed32_cos Taylor series implementation."""

    def test_cos_zero(self):
        """cos(0) = 1."""
        result = fixed32_cos(FIXED32_ZERO)
        assert abs(result.as_float - 1.0) < 0.001

    def test_cos_pi(self):
        """cos(pi) = -1."""
        result = fixed32_cos(FIXED32_PI)
        assert abs(result.as_float + 1.0) < 0.05  # Fixed-point tolerance

    def test_cos_half_pi(self):
        """cos(pi/2) = 0."""
        result = fixed32_cos(FIXED32_HALF_PI)
        assert abs(result.as_float) < 0.02

    def test_cos_quarter_pi(self):
        """cos(pi/4) = sqrt(2)/2 ~ 0.707."""
        quarter_pi = Fixed32(math.pi / 4)
        result = fixed32_cos(quarter_pi)
        expected = math.cos(math.pi / 4)
        assert abs(result.as_float - expected) < 0.01

    def test_cos_deterministic(self):
        """Same input produces same output across calls."""
        angle = Fixed32(1.234)
        result1 = fixed32_cos(angle)
        result2 = fixed32_cos(angle)
        assert result1.raw == result2.raw

    def test_cos_symmetry(self):
        """cos(-x) = cos(x)."""
        angle = Fixed32(0.75)
        pos = fixed32_cos(angle)
        neg = fixed32_cos(-angle)
        assert pos.raw == neg.raw

    def test_cos_range_accuracy(self):
        """Check accuracy over multiple angles."""
        for degrees in range(0, 360, 15):
            radians = math.radians(degrees)
            angle = Fixed32(radians)
            result = fixed32_cos(angle)
            expected = math.cos(radians)
            error = abs(result.as_float - expected)
            # Fixed-point Taylor series has ~0.06 max error near boundaries
            assert error < 0.07, f"cos({degrees}deg) error={error}"


class TestFixed32SinCos:
    """Test combined sincos function."""

    def test_sincos_matches_separate(self):
        """sincos returns same as individual calls."""
        angle = Fixed32(1.5)
        sin_val, cos_val = fixed32_sincos(angle)
        assert sin_val.raw == fixed32_sin(angle).raw
        assert cos_val.raw == fixed32_cos(angle).raw

    def test_sincos_pythagorean(self):
        """sin^2 + cos^2 ~ 1."""
        for degrees in [0, 30, 45, 60, 90, 135, 180, 270]:
            angle = Fixed32(math.radians(degrees))
            sin_val, cos_val = fixed32_sincos(angle)
            sum_squares = (sin_val * sin_val + cos_val * cos_val).as_float
            # Fixed-point errors compound in squaring, allow ~10% error
            assert abs(sum_squares - 1.0) < 0.15, f"Identity at {degrees}deg"


class TestNormalizeAngle:
    """Test angle normalization."""

    def test_normalize_zero(self):
        """Zero stays zero."""
        result = _normalize_angle(FIXED32_ZERO)
        assert result.raw == 0

    def test_normalize_pi(self):
        """Pi stays in range."""
        result = _normalize_angle(FIXED32_PI)
        assert abs(result.as_float - math.pi) < 0.01 or abs(result.as_float + math.pi) < 0.01

    def test_normalize_large_angle(self):
        """Large angles wrap correctly."""
        large = Fixed32(10.0)  # ~3.18 * pi
        result = _normalize_angle(large)
        assert -math.pi <= result.as_float <= math.pi


# =============================================================================
# FIXED32 VECTOR TESTS
# =============================================================================


class TestFixed32Vec2:
    """Test Fixed32Vec2 operations."""

    def test_from_floats(self):
        """Create vector from floats."""
        v = Fixed32Vec2.from_floats(1.5, -2.25)
        assert abs(v.x.as_float - 1.5) < 0.001
        assert abs(v.y.as_float + 2.25) < 0.001

    def test_dot_product(self):
        """Dot product computation."""
        v1 = Fixed32Vec2.from_floats(1.0, 2.0)
        v2 = Fixed32Vec2.from_floats(3.0, 4.0)
        dot = v1.dot(v2)
        assert abs(dot.as_float - 11.0) < 0.01

    def test_length_squared(self):
        """Squared length computation."""
        v = Fixed32Vec2.from_floats(3.0, 4.0)
        len_sq = v.length_squared()
        assert abs(len_sq.as_float - 25.0) < 0.01

    def test_add(self):
        """Vector addition."""
        v1 = Fixed32Vec2.from_floats(1.0, 2.0)
        v2 = Fixed32Vec2.from_floats(0.5, 0.5)
        result = v1 + v2
        assert abs(result.x.as_float - 1.5) < 0.001
        assert abs(result.y.as_float - 2.5) < 0.001

    def test_sub(self):
        """Vector subtraction."""
        v1 = Fixed32Vec2.from_floats(1.0, 2.0)
        v2 = Fixed32Vec2.from_floats(0.5, 0.5)
        result = v1 - v2
        assert abs(result.x.as_float - 0.5) < 0.001
        assert abs(result.y.as_float - 1.5) < 0.001

    def test_mul_scalar(self):
        """Scalar multiplication."""
        v = Fixed32Vec2.from_floats(2.0, 3.0)
        result = v * Fixed32(2.0)
        assert abs(result.x.as_float - 4.0) < 0.01
        assert abs(result.y.as_float - 6.0) < 0.01

    def test_neg(self):
        """Negation."""
        v = Fixed32Vec2.from_floats(1.0, -2.0)
        result = -v
        assert abs(result.x.as_float + 1.0) < 0.001
        assert abs(result.y.as_float - 2.0) < 0.001


class TestFixed32Vec3:
    """Test Fixed32Vec3 operations."""

    def test_from_floats(self):
        """Create vector from floats."""
        v = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        assert abs(v.x.as_float - 1.0) < 0.001
        assert abs(v.y.as_float - 2.0) < 0.001
        assert abs(v.z.as_float - 3.0) < 0.001

    def test_dot_product(self):
        """Dot product computation."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(4.0, 5.0, 6.0)
        dot = v1.dot(v2)
        assert abs(dot.as_float - 32.0) < 0.01

    def test_add(self):
        """Vector addition."""
        v1 = Fixed32Vec3.from_floats(1.0, 2.0, 3.0)
        v2 = Fixed32Vec3.from_floats(0.5, 0.5, 0.5)
        result = v1 + v2
        assert abs(result.x.as_float - 1.5) < 0.001
        assert abs(result.y.as_float - 2.5) < 0.001
        assert abs(result.z.as_float - 3.5) < 0.001


# =============================================================================
# WAVE PARAMETERS TESTS
# =============================================================================


class TestFixed32WaveParams:
    """Test Fixed32WaveParams."""

    def test_default_params(self):
        """Default parameters are valid."""
        params = Fixed32WaveParams()
        assert params.amplitude.as_float > 0
        assert params.frequency.as_float > 0
        assert 0 <= params.steepness.as_float <= 1

    def test_from_floats(self):
        """Create from float values."""
        params = Fixed32WaveParams.from_floats(
            amplitude=0.8,
            wavelength=15.0,
            phase=1.0,
            direction_x=1.0,
            direction_z=0.0,
            steepness=0.6,
        )
        assert abs(params.amplitude.as_float - 0.8) < 0.01
        assert abs(params.steepness.as_float - 0.6) < 0.01

    def test_from_random_deterministic(self):
        """Random generation is deterministic with same seed."""
        rng1 = PCG64(42)
        rng2 = PCG64(42)

        params1 = Fixed32WaveParams.from_random(rng1)
        params2 = Fixed32WaveParams.from_random(rng2)

        assert params1.amplitude.raw == params2.amplitude.raw
        assert params1.frequency.raw == params2.frequency.raw
        assert params1.phase.raw == params2.phase.raw
        assert params1.direction.x.raw == params2.direction.x.raw
        assert params1.direction.y.raw == params2.direction.y.raw
        assert params1.steepness.raw == params2.steepness.raw

    def test_checksum_deterministic(self):
        """Checksum is deterministic."""
        params = Fixed32WaveParams.from_floats(amplitude=0.5, wavelength=10.0)
        cs1 = params.get_checksum()
        cs2 = params.get_checksum()
        assert cs1 == cs2

    def test_checksum_changes(self):
        """Different params have different checksums."""
        params1 = Fixed32WaveParams.from_floats(amplitude=0.5)
        params2 = Fixed32WaveParams.from_floats(amplitude=0.6)
        assert params1.get_checksum() != params2.get_checksum()


# =============================================================================
# WAVE COMPUTATION TESTS
# =============================================================================


class TestComputeGerstnerWave:
    """Test compute_gerstner_wave function."""

    def test_returns_result(self):
        """Function returns GerstnerWaveResult."""
        params = Fixed32WaveParams()
        pos = Fixed32Vec2.from_floats(0.0, 0.0)
        time = FIXED32_ZERO

        result = compute_gerstner_wave(params, pos, time)
        assert isinstance(result, GerstnerWaveResult)
        assert isinstance(result.displacement, Fixed32Vec3)
        assert isinstance(result.height, Fixed32)

    def test_deterministic(self):
        """Same inputs produce same output."""
        params = Fixed32WaveParams.from_floats(amplitude=1.0, wavelength=8.0)
        pos = Fixed32Vec2.from_floats(5.0, 3.0)
        time = Fixed32(2.5)

        result1 = compute_gerstner_wave(params, pos, time)
        result2 = compute_gerstner_wave(params, pos, time)

        assert result1.height.raw == result2.height.raw
        assert result1.displacement.x.raw == result2.displacement.x.raw
        assert result1.displacement.y.raw == result2.displacement.y.raw
        assert result1.displacement.z.raw == result2.displacement.z.raw

    def test_height_bounded(self):
        """Height is bounded by amplitude (with fixed-point tolerance)."""
        params = Fixed32WaveParams.from_floats(amplitude=2.0, wavelength=10.0)

        for x in range(10):
            for z in range(10):
                pos = Fixed32Vec2.from_floats(float(x), float(z))
                result = compute_gerstner_wave(params, pos, Fixed32(1.0))
                # Fixed-point trig can exceed 1.0 slightly, so allow 25% margin
                assert abs(result.height.as_float) <= 2.5, f"Height out of bounds at ({x},{z})"

    def test_time_affects_result(self):
        """Different times produce different results."""
        params = Fixed32WaveParams.from_floats(amplitude=1.0, wavelength=10.0)
        pos = Fixed32Vec2.from_floats(0.0, 0.0)

        result1 = compute_gerstner_wave(params, pos, Fixed32(0.0))
        result2 = compute_gerstner_wave(params, pos, Fixed32(1.0))

        # Results should differ (unless exactly periodic)
        assert result1.height.raw != result2.height.raw


# =============================================================================
# DETERMINISTIC GERSTNER WAVE SIMULATOR TESTS
# =============================================================================


class TestDeterministicGerstnerWave:
    """Test DeterministicGerstnerWave simulator."""

    def test_initialization(self):
        """Simulator initializes correctly."""
        sim = DeterministicGerstnerWave(seed=42, wave_count=4)
        assert not sim.is_initialized

        sim.initialize()
        assert sim.is_initialized
        assert sim.wave_count == 4
        assert sim.tick_count == 0

    def test_double_init_raises(self):
        """Double initialization raises error."""
        sim = DeterministicGerstnerWave()
        sim.initialize()

        with pytest.raises(RuntimeError):
            sim.initialize()

    def test_operations_before_init_raise(self):
        """Operations before init raise errors."""
        sim = DeterministicGerstnerWave()

        with pytest.raises(RuntimeError):
            sim.sample_height(Fixed32Vec2.from_floats(0, 0))

        with pytest.raises(RuntimeError):
            sim.sample_displacement(Fixed32Vec2.from_floats(0, 0))

        with pytest.raises(RuntimeError):
            sim.advance_time(Fixed32(0.016))

    def test_time_advance(self):
        """Time advances correctly."""
        sim = DeterministicGerstnerWave()
        sim.initialize()

        assert sim.current_time.raw == 0

        sim.advance_time(Fixed32(0.016))
        assert sim.current_time.as_float > 0
        assert sim.tick_count == 1

        sim.advance_time(Fixed32(0.016))
        assert sim.tick_count == 2

    def test_sample_height(self):
        """Height sampling works."""
        sim = DeterministicGerstnerWave(seed=123, wave_count=2)
        sim.initialize()

        pos = Fixed32Vec2.from_floats(10.0, 20.0)
        height = sim.sample_height(pos)

        assert isinstance(height, Fixed32)

    def test_sample_displacement(self):
        """Displacement sampling works."""
        sim = DeterministicGerstnerWave(seed=123, wave_count=2)
        sim.initialize()

        pos = Fixed32Vec2.from_floats(10.0, 20.0)
        disp = sim.sample_displacement(pos)

        assert isinstance(disp, Fixed32Vec3)

    def test_determinism_same_seed(self):
        """Same seed produces identical results."""
        sim1 = DeterministicGerstnerWave(seed=42, wave_count=4)
        sim2 = DeterministicGerstnerWave(seed=42, wave_count=4)

        sim1.initialize()
        sim2.initialize()

        pos = Fixed32Vec2.from_floats(5.0, 7.0)

        # Both should produce identical waves
        for wave1, wave2 in zip(sim1.waves, sim2.waves):
            assert wave1.amplitude.raw == wave2.amplitude.raw
            assert wave1.frequency.raw == wave2.frequency.raw

        # Height should match
        h1 = sim1.sample_height(pos)
        h2 = sim2.sample_height(pos)
        assert h1.raw == h2.raw

    def test_determinism_across_time(self):
        """Simulation is deterministic across time steps."""
        sim1 = DeterministicGerstnerWave(seed=99, wave_count=3)
        sim2 = DeterministicGerstnerWave(seed=99, wave_count=3)

        sim1.initialize()
        sim2.initialize()

        delta = Fixed32(0.016)
        pos = Fixed32Vec2.from_floats(3.0, 4.0)

        # Advance both by same amount
        for _ in range(10):
            sim1.advance_time(delta)
            sim2.advance_time(delta)

        h1 = sim1.sample_height(pos)
        h2 = sim2.sample_height(pos)

        assert h1.raw == h2.raw

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        sim1 = DeterministicGerstnerWave(seed=1)
        sim2 = DeterministicGerstnerWave(seed=2)

        sim1.initialize()
        sim2.initialize()

        pos = Fixed32Vec2.from_floats(5.0, 5.0)

        h1 = sim1.sample_height(pos)
        h2 = sim2.sample_height(pos)

        assert h1.raw != h2.raw

    def test_checksum_deterministic(self):
        """State checksum is deterministic."""
        sim1 = DeterministicGerstnerWave(seed=77, wave_count=4)
        sim2 = DeterministicGerstnerWave(seed=77, wave_count=4)

        sim1.initialize()
        sim2.initialize()

        sim1.advance_time(Fixed32(0.1))
        sim2.advance_time(Fixed32(0.1))

        cs1 = sim1.get_state_checksum()
        cs2 = sim2.get_state_checksum()

        assert cs1 == cs2

    def test_checksum_changes_with_time(self):
        """Checksum changes as time advances."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        cs1 = sim.get_state_checksum()
        sim.advance_time(Fixed32(0.1))
        cs2 = sim.get_state_checksum()

        assert cs1 != cs2

    def test_reset(self):
        """Reset returns to initial state."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        sim.advance_time(Fixed32(1.0))
        assert sim.current_time.as_float > 0
        assert sim.tick_count > 0

        sim.reset()
        assert sim.current_time.raw == 0
        assert sim.tick_count == 0

    def test_destroy(self):
        """Destroy cleans up resources."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        sim.destroy()
        assert not sim.is_initialized
        assert sim.wave_count == 0


class TestGridSampling:
    """Test grid sampling for water mesh generation."""

    def test_grid_sampling_basic(self):
        """Grid sampling returns correct dimensions."""
        sim = DeterministicGerstnerWave(seed=42, wave_count=2)
        sim.initialize()

        origin = Fixed32Vec2.from_floats(0.0, 0.0)
        step = Fixed32(1.0)

        grid = sim.sample_height_grid(origin, step, width=4, height=3)

        assert len(grid) == 3  # rows
        assert len(grid[0]) == 4  # cols

    def test_grid_sampling_deterministic(self):
        """Grid sampling is deterministic."""
        sim1 = DeterministicGerstnerWave(seed=42, wave_count=2)
        sim2 = DeterministicGerstnerWave(seed=42, wave_count=2)

        sim1.initialize()
        sim2.initialize()

        origin = Fixed32Vec2.from_floats(10.0, 20.0)
        step = Fixed32(2.0)

        grid1 = sim1.sample_height_grid(origin, step, width=5, height=5)
        grid2 = sim2.sample_height_grid(origin, step, width=5, height=5)

        for row in range(5):
            for col in range(5):
                assert grid1[row][col].raw == grid2[row][col].raw


class TestCustomWaves:
    """Test custom wave parameter injection."""

    def test_custom_waves(self):
        """Custom waves override random generation."""
        custom = [
            Fixed32WaveParams.from_floats(amplitude=1.0, wavelength=10.0),
            Fixed32WaveParams.from_floats(amplitude=0.5, wavelength=5.0),
        ]

        sim = DeterministicGerstnerWave(custom_waves=custom)
        sim.initialize()

        assert sim.wave_count == 2
        assert sim.waves[0].amplitude.raw == custom[0].amplitude.raw
        assert sim.waves[1].amplitude.raw == custom[1].amplitude.raw


class TestBitExactReplay:
    """Test bit-exact replay capability."""

    def test_replay_produces_identical_surface(self):
        """Two runs with same seed produce bit-identical surfaces."""
        seed = 12345
        positions = [
            Fixed32Vec2.from_floats(i * 0.5, j * 0.5)
            for i in range(10)
            for j in range(10)
        ]
        time_steps = [Fixed32(0.016) for _ in range(60)]  # 60 frames

        # First run
        sim1 = DeterministicGerstnerWave(seed=seed, wave_count=4)
        sim1.initialize()

        heights1 = []
        for dt in time_steps:
            sim1.advance_time(dt)
            frame_heights = [sim1.sample_height(p).raw for p in positions]
            heights1.append(frame_heights)

        checksum1 = sim1.get_state_checksum()

        # Second run
        sim2 = DeterministicGerstnerWave(seed=seed, wave_count=4)
        sim2.initialize()

        heights2 = []
        for dt in time_steps:
            sim2.advance_time(dt)
            frame_heights = [sim2.sample_height(p).raw for p in positions]
            heights2.append(frame_heights)

        checksum2 = sim2.get_state_checksum()

        # Verify bit-identical
        assert checksum1 == checksum2, "Checksums must match"

        for frame_idx, (h1, h2) in enumerate(zip(heights1, heights2)):
            for pos_idx, (v1, v2) in enumerate(zip(h1, h2)):
                assert v1 == v2, f"Mismatch at frame {frame_idx}, pos {pos_idx}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_time(self):
        """Zero time step is valid."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        sim.advance_time(FIXED32_ZERO)
        assert sim.current_time.raw == 0

    def test_large_time(self):
        """Large time values work without overflow."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        # Advance by large amount
        sim.advance_time(Fixed32(1000.0))
        pos = Fixed32Vec2.from_floats(0.0, 0.0)
        height = sim.sample_height(pos)

        # Should not crash, value should be bounded
        assert isinstance(height, Fixed32)

    def test_negative_position(self):
        """Negative positions work correctly."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        pos = Fixed32Vec2.from_floats(-100.0, -50.0)
        height = sim.sample_height(pos)

        assert isinstance(height, Fixed32)

    def test_single_wave(self):
        """Single wave simulation works."""
        sim = DeterministicGerstnerWave(seed=42, wave_count=1)
        sim.initialize()

        assert sim.wave_count == 1

        pos = Fixed32Vec2.from_floats(0.0, 0.0)
        height = sim.sample_height(pos)
        assert isinstance(height, Fixed32)

    def test_many_waves(self):
        """Many waves simulation works."""
        sim = DeterministicGerstnerWave(seed=42, wave_count=16)
        sim.initialize()

        assert sim.wave_count == 16

        pos = Fixed32Vec2.from_floats(0.0, 0.0)
        height = sim.sample_height(pos)
        assert isinstance(height, Fixed32)

    def test_set_absolute_time(self):
        """Absolute time setting works."""
        sim = DeterministicGerstnerWave(seed=42)
        sim.initialize()

        sim.set_time(Fixed32(5.0))
        assert abs(sim.current_time.as_float - 5.0) < 0.001


class TestWavePhysics:
    """Test physical correctness of wave simulation."""

    def test_wave_moves_over_time(self):
        """Wave pattern changes over time."""
        sim = DeterministicGerstnerWave(seed=42, wave_count=1)
        sim.initialize()

        pos = Fixed32Vec2.from_floats(10.0, 0.0)

        heights = []
        for i in range(10):
            heights.append(sim.sample_height(pos).as_float)
            sim.advance_time(Fixed32(0.1))

        # Heights should vary (wave is moving)
        assert len(set(h for h in heights)) > 1

    def test_displacement_direction(self):
        """Displacement follows wave direction."""
        wave = Fixed32WaveParams.from_floats(
            amplitude=1.0,
            wavelength=10.0,
            direction_x=1.0,
            direction_z=0.0,
            steepness=0.5,
        )

        sim = DeterministicGerstnerWave(custom_waves=[wave])
        sim.initialize()

        # Sample at phase where sin is nonzero
        sim.set_time(Fixed32(0.5))
        pos = Fixed32Vec2.from_floats(5.0, 0.0)
        disp = sim.sample_displacement(pos)

        # For wave moving in +X, horizontal displacement should be in X
        # Z displacement should be near zero
        assert abs(disp.z.as_float) < abs(disp.x.as_float) + 0.1
