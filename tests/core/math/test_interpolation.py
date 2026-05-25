"""Tests for interpolation functions."""

import pytest

from engine.core.math.interpolation import (
    lerp, inverse_lerp, remap, clamp, smoothstep, smootherstep,
    in_quad, out_quad, in_out_quad, in_cubic, out_cubic, in_out_cubic,
    SpringDamper,
)


class TestLerp:
    def test_lerp_endpoints(self):
        assert lerp(0, 10, 0) == pytest.approx(0)
        assert lerp(0, 10, 1) == pytest.approx(10)
        assert lerp(0, 10, 0.5) == pytest.approx(5)

    def test_lerp_out_of_range(self):
        """Edge case: t outside [0, 1] extrapolates."""
        assert lerp(0, 10, 2) == pytest.approx(20)
        assert lerp(0, 10, -1) == pytest.approx(-10)

    def test_inverse_lerp(self):
        assert inverse_lerp(0, 10, 5) == pytest.approx(0.5)

    def test_inverse_lerp_reversed(self):
        """Edge case: inverse_lerp with reversed range."""
        assert inverse_lerp(10, 0, 5) == pytest.approx(0.5)

    def test_inverse_lerp_equal_inputs(self):
        """Edge case: inverse_lerp with a == b returns 0."""
        assert inverse_lerp(5, 5, 10) == pytest.approx(0)
        assert inverse_lerp(0, 0, 0) == pytest.approx(0)

    def test_remap(self):
        assert remap(5, 0, 10, 100, 200) == pytest.approx(150)

    def test_remap_out_of_range(self):
        """Edge case: remap with value outside input range extrapolates."""
        assert remap(15, 0, 10, 100, 200) == pytest.approx(250)
        assert remap(-5, 0, 10, 100, 200) == pytest.approx(50)


class TestClamp:
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_clamp_inverted_range(self):
        """Edge case: clamp where lo > hi."""
        assert clamp(5, 10, 0) == 10


class TestSmoothstep:
    def test_boundaries(self):
        assert smoothstep(0, 1, 0) == pytest.approx(0)
        assert smoothstep(0, 1, 1) == pytest.approx(1)
        assert smoothstep(0, 1, 0.5) == pytest.approx(0.5)

    def test_smoothstep_equal_edges(self):
        """Edge case: smoothstep with edge0 == edge1 returns 0."""
        assert smoothstep(1, 1, 0.5) == pytest.approx(0)
        assert smoothstep(0, 0, 5) == pytest.approx(0)

    def test_smootherstep(self):
        assert smootherstep(0, 1, 0) == pytest.approx(0)
        assert smootherstep(0, 1, 1) == pytest.approx(1)

    def test_smootherstep_equal_edges(self):
        """Edge case: smootherstep with edge0 == edge1 returns 0."""
        assert smootherstep(1, 1, 0.5) == pytest.approx(0)
        assert smootherstep(0, 0, 5) == pytest.approx(0)


class TestEasing:
    def test_quad_endpoints(self):
        assert in_quad(0) == pytest.approx(0)
        assert in_quad(1) == pytest.approx(1)
        assert out_quad(0) == pytest.approx(0)
        assert out_quad(1) == pytest.approx(1)
        assert in_out_quad(0) == pytest.approx(0)
        assert in_out_quad(1) == pytest.approx(1)

    def test_quad_midpoint(self):
        """Edge case: in_out_quad at midpoint and branch boundary."""
        assert in_out_quad(0.5) == pytest.approx(0.5)
        assert in_out_quad(0.25) == pytest.approx(0.125)

    def test_cubic_endpoints(self):
        assert in_cubic(0) == pytest.approx(0)
        assert in_cubic(1) == pytest.approx(1)
        assert out_cubic(0) == pytest.approx(0)
        assert out_cubic(1) == pytest.approx(1)
        assert in_out_cubic(0) == pytest.approx(0)
        assert in_out_cubic(1) == pytest.approx(1)

    def test_cubic_midpoint(self):
        """Edge case: in_out_cubic at midpoint and branch boundary."""
        assert in_out_cubic(0.5) == pytest.approx(0.5)
        assert in_out_cubic(0.25) == pytest.approx(0.0625)

    def test_out_quad_midpoint(self):
        """Edge case: out_quad symmetric value."""
        assert out_quad(0.5) == pytest.approx(0.75)


class TestSpringDamper:
    def test_converges_to_target(self):
        s = SpringDamper(position=0, target=10, omega=20)
        for _ in range(1000):
            s.update(0.016)
        assert s.position == pytest.approx(10, abs=0.01)

    def test_initial_state(self):
        s = SpringDamper(position=5, target=5)
        s.update(0.1)
        assert s.position == pytest.approx(5, abs=0.01)

    def test_zero_dt(self):
        """Edge case: update with zero dt does not change position."""
        s = SpringDamper(position=3, velocity=2, target=10)
        pos = s.update(0.0)
        assert pos == pytest.approx(3)

    def test_negative_dt_raises(self):
        """Edge case: negative dt raises ValueError."""
        s = SpringDamper(position=0, target=10)
        with pytest.raises(ValueError, match="dt must be non-negative"):
            s.update(-0.1)

    def test_high_omega_converges_fast(self):
        """Edge case: very stiff spring converges rapidly."""
        s = SpringDamper(position=0, target=10, omega=100)
        for _ in range(100):
            s.update(0.016)
        assert s.position == pytest.approx(10, abs=0.01)

    def test_no_overshoot(self):
        """SpringDamper is critically damped: should not overshoot target."""
        s = SpringDamper(position=0, target=10, omega=5)
        positions = []
        for _ in range(200):
            s.update(0.016)
            positions.append(s.position)
        assert all(p <= 10 + 0.01 for p in positions)
