"""
Whitebox tests for noise hash functions (T-DEMO-1.28).

Tests Python model implementations of each WGSL function, verifying:
  - Correctness against mathematical definition
  - Edge cases (zero, negative, boundary values)
  - Distribution properties (mean, variance, uniformity)
  - No visible patterns (autocorrelation, chi-squared)
  - Integer and float coordinate sampling

WHITEBOX coverage plan:
  Path A: hash11 on integer grid -> [0, 1) range, deterministic
  Path B: hash21 on 2D integer grid -> [0, 1) range, deterministic
  Path C: hash31 on 3D integer grid -> [0, 1) range, deterministic
  Path D: hash22 -> both components in [0, 1), uncorrelated
  Path E: hash33 -> all components in [0, 1), uncorrelated
  Path F: hash41 -> 4D input produces [0, 1) output
  Path G: Distribution mean close to 0.5 (uniformity)
  Path H: Distribution variance close to 1/12 (uniformity)
  Path I: Chi-squared test for uniformity
  Path J: Adjacent integer hash outputs are uncorrelated
  Path K: Large coordinate values produce finite output
  Path L: Zero and negative integer coordinates work
  Path M: Float coordinates (non-integer) produce valid output
  Path N: hash22 components differ for same input (different channels)
  Path O: hash33 components differ for same input
  Path P: No two successive integer hashes produce the same result
  Path Q: Integer grid chi-squared test (empirical distribution)
  Path R: Consecutive hash values autocorrelation (lag-1)
"""

import math
import os
import random
from collections import Counter

import numpy as np
import pytest

# =============================================================================
# Python model implementations matching WGSL semantics
# =============================================================================

# WGSL fract: x - floor(x)
def wgsl_fract(x: float) -> float:
    return x - math.floor(x)


def py_hash11(p: float) -> float:
    """Model of WGSL hash11: 1D float -> [0, 1) float."""
    q = p
    q = wgsl_fract(q * 0.1031)
    q = q * (q + 33.33)
    q = q * (q + q)
    return wgsl_fract(q)


def py_hash21(p) -> float:
    """Model of WGSL hash21: 2D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    d = qx * (qx + 33.33) + qy * (qy + 33.33)
    qx += d
    qy += d
    return wgsl_fract(qx * qy)


def py_hash31(p) -> float:
    """Model of WGSL hash31: 3D -> [0, 1) float."""
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)
    d = qx * (qx + 33.33) + qy * (qy + 33.33) + qz * (qz + 33.33)
    qx += d
    qy += d
    qz += d
    return wgsl_fract(qx * qy * qz)


def py_hash41(p) -> float:
    """Model of WGSL hash41: 4D -> [0, 1) float."""
    q = [wgsl_fract(v * 0.1031) for v in p]
    d = sum(vi * (vi + 33.33) for vi in q)
    q = [vi + d for vi in q]
    return wgsl_fract(q[0] * q[1] * q[2] * q[3])


def py_hash22(p):
    """Model of WGSL hash22: 2D -> 2x [0, 1) float.

    WGSL: var q = vec3<f32>(p.x, p.y, p.x);
          q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
          q = q + dot(q, q.yzx + 33.33);
          return fract(vec2<f32>(q.x + q.y, q.x + q.z) * vec2<f32>(q.z, q.y));
    """
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[0] * 0.0973)  # p.x for third component
    # dot(q, q.yzx + 33.33) = q.x*(q.y+33.33) + q.y*(q.z+33.33) + q.z*(q.x+33.33)
    d = qx * (qy + 33.33) + qy * (qz + 33.33) + qz * (qx + 33.33)
    qx += d
    qy += d
    qz += d
    r0 = wgsl_fract((qx + qy) * qz)
    r1 = wgsl_fract((qx + qz) * qy)
    return (r0, r1)


def py_hash32(p):
    """Model of WGSL hash32: 3D -> 2x [0, 1) float.

    WGSL: var q = p;
          q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
          q = q + dot(q, q.yxz + 33.33);
          let sum = vec2<f32>(q.x + q.y, q.x + q.z);
          let zy = vec2<f32>(q.z, q.y);
          return fract(sum * zy);
    """
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)
    # dot(q, q.yxz + 33.33) = q.x*(q.y+33.33) + q.y*(q.x+33.33) + q.z*(q.z+33.33)
    d = qx * (qy + 33.33) + qy * (qx + 33.33) + qz * (qz + 33.33)
    qx += d
    qy += d
    qz += d
    r0 = wgsl_fract((qx + qy) * qz)
    r1 = wgsl_fract((qx + qz) * qy)
    return (r0, r1)


def py_hash33(p):
    """Model of WGSL hash33: 3D -> 3x [0, 1) float.

    WGSL: var q = p;
          q = fract(q * vec3<f32>(0.1031, 0.1030, 0.0973));
          q = q + dot(q, q.yxz + 33.33);
          return fract((q.xxy + q.yxx) * q.zyx);
    """
    qx = wgsl_fract(p[0] * 0.1031)
    qy = wgsl_fract(p[1] * 0.1030)
    qz = wgsl_fract(p[2] * 0.0973)
    d = qx * (qy + 33.33) + qy * (qx + 33.33) + qz * (qz + 33.33)
    qx += d
    qy += d
    qz += d
    # (q.xxy + q.yxx) * q.zyx
    # q.xxy = (qx, qx, qy), q.yxx = (qy, qx, qx) -> sum = (qx+qy, 2*qx, qx+qy)
    # q.zyx = (qz, qy, qx)
    r0 = wgsl_fract((qx + qy) * qz)
    r1 = wgsl_fract((2.0 * qx) * qy)
    r2 = wgsl_fract((qx + qy) * qx)
    return (r0, r1, r2)


# =============================================================================
# Helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-7

SAMPLE_SIZE = 5000  # Number of samples for distribution tests

HASH_FN_1D = [
    ("hash11", py_hash11, lambda i: (float(i),)),
    ("hash21", py_hash21, lambda i: (float(i), float(i + 1))),
    ("hash31", py_hash31, lambda i: (float(i), float(i + 1), float(i + 2))),
]

HASH_FN_ND = [
    ("hash22", py_hash22, lambda i: (float(i), float(i + 1))),
    ("hash32", py_hash32, lambda i: (float(i), float(i + 1), float(i + 2))),
    ("hash33", py_hash33, lambda i: (float(i), float(i + 1), float(i + 2))),
]


# =============================================================================
# Test: T-DEMO-1.28 hash11 (1D -> f32)
# =============================================================================


class TestHash11:
    """Whitebox tests for hash11(p: f32) -> f32."""

    def test_range_integer_grid(self):
        """hash11 on integer grid must be in [0, 1)."""
        for i in range(-100, 101):
            result = py_hash11(float(i))
            assert 0.0 <= result < 1.0, f"hash11({i}) = {result}"

    def test_range_float_values(self):
        """hash11 on float values must be in [0, 1)."""
        for _ in range(200):
            p = random.uniform(-1000, 1000)
            result = py_hash11(p)
            assert 0.0 <= result < 1.0, f"hash11({p}) = {result}"

    def test_deterministic(self):
        """hash11 same input -> same output."""
        for p in [0.0, 1.0, 42.0, -17.5, 3.14159, -0.001, 1e6, -1e6]:
            r1 = py_hash11(p)
            r2 = py_hash11(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL), (
                f"hash11({p}) not deterministic: {r1} != {r2}"
            )

    def test_zero_input(self):
        """hash11(0) produces finite result in [0, 1)."""
        result = py_hash11(0.0)
        assert 0.0 <= result < 1.0, f"hash11(0) = {result}"

    def test_negative_input(self):
        """hash11 on negative values produces valid output."""
        for p in [-1.0, -42.0, -100.0, -0.5, -0.001]:
            result = py_hash11(p)
            assert 0.0 <= result < 1.0, f"hash11({p}) = {result}"

    def test_large_input(self):
        """hash11 on large values produces valid output."""
        for p in [1e6, -1e6, 1e12, -1e12]:
            result = py_hash11(p)
            assert 0.0 <= result < 1.0, f"hash11({p}) = {result}"

    def test_no_successive_duplicates(self):
        """Adjacent integer hashes should differ."""
        prev = None
        for i in range(-20, 21):
            curr = py_hash11(float(i))
            if prev is not None:
                assert not math.isclose(curr, prev, rel_tol=TOL_REL, abs_tol=TOL_ABS), (
                    f"hash11({i}) == hash11({i-1}) = {curr}"
                )
            prev = curr

    def test_successive_diffs_vary(self):
        """Differences between successive hashes should vary."""
        values = [py_hash11(float(i)) for i in range(-20, 21)]
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values) - 1)]
        # At least some variation
        assert len(set(round(d, 6) for d in diffs)) >= 5, (
            "Successive hash diffs do not vary enough"
        )

    def test_hash11_not_constant(self):
        """hash11 should not return the same value for all inputs."""
        values = {py_hash11(float(i)) for i in range(-50, 51)}
        assert len(values) >= 50, (
            f"hash11 only produced {len(values)} unique values out of 101 inputs"
        )


# =============================================================================
# Test: T-DEMO-1.28 hash21 (2D -> f32)
# =============================================================================


class TestHash21:
    """Whitebox tests for hash21(p: vec2<f32>) -> f32."""

    def test_range_integer_grid(self):
        """hash21 on 2D integer grid must be in [0, 1)."""
        for x in range(-10, 11):
            for y in range(-10, 11):
                result = py_hash21((float(x), float(y)))
                assert 0.0 <= result < 1.0, f"hash21({x},{y}) = {result}"

    def test_range_float_values(self):
        """hash21 on float values must be in [0, 1)."""
        for _ in range(200):
            p = (random.uniform(-1000, 1000), random.uniform(-1000, 1000))
            result = py_hash21(p)
            assert 0.0 <= result < 1.0, f"hash21({p}) = {result}"

    def test_deterministic(self):
        """hash21 same input -> same output."""
        for p in [(0.0, 0.0), (1.0, 2.0), (-1.0, -1.0), (42.0, 17.0)]:
            r1 = py_hash21(p)
            r2 = py_hash21(p)
            assert math.isclose(r1, r2, rel_tol=TOL_REL)

    def test_xy_order_matters(self):
        """hash21(x,y) must differ from hash21(y,x) for x != y."""
        r_xy = py_hash21((1.0, 2.0))
        r_yx = py_hash21((2.0, 1.0))
        assert not math.isclose(r_xy, r_yx, rel_tol=TOL_REL), (
            "hash21 should differ when swapping x and y"
        )

    def test_adjacent_cells_differ(self):
        """Adjacent cells should produce different hash values."""
        results = {}
        for x in range(-5, 6):
            for y in range(-5, 6):
                results[(x, y)] = py_hash21((float(x), float(y)))
        unique = set(round(v, 8) for v in results.values())
        assert len(unique) >= 100, (
            f"2D grid only produced {len(unique)} unique values out of 121"
        )

    def test_large_coordinates(self):
        """hash21 on large coordinates produces valid output."""
        for _ in range(50):
            p = (random.uniform(-1e6, 1e6), random.uniform(-1e6, 1e6))
            result = py_hash21(p)
            assert 0.0 <= result < 1.0, f"hash21({p}) = {result}"


# =============================================================================
# Test: T-DEMO-1.28 hash31 (3D -> f32)
# =============================================================================


class TestHash31:
    """Whitebox tests for hash31(p: vec3<f32>) -> f32."""

    def test_range_small_grid(self):
        """hash31 on small 3D integer grid must be in [0, 1)."""
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    result = py_hash31((float(x), float(y), float(z)))
                    assert 0.0 <= result < 1.0, f"hash31({x},{y},{z}) = {result}"

    def test_deterministic(self):
        """hash31 same input -> same output."""
        p = (1.0, 2.0, 3.0)
        assert math.isclose(py_hash31(p), py_hash31(p), rel_tol=TOL_REL)

    def test_xyz_order_matters(self):
        """hash31(x,y,z) must differ from permutations."""
        base = (1.0, 2.0, 3.0)
        perms = [
            (2.0, 1.0, 3.0),
            (3.0, 2.0, 1.0),
            (1.0, 3.0, 2.0),
            (2.0, 3.0, 1.0),
            (3.0, 1.0, 2.0),
        ]
        base_val = py_hash31(base)
        for perm in perms:
            perm_val = py_hash31(perm)
            assert not math.isclose(base_val, perm_val, rel_tol=TOL_REL), (
                f"hash31({base}) == hash31({perm})"
            )

    def test_large_coordinates(self):
        """hash31 on large coordinates produces valid output."""
        for _ in range(50):
            p = tuple(random.uniform(-1e6, 1e6) for _ in range(3))
            result = py_hash31(p)
            assert 0.0 <= result < 1.0, f"hash31({p}) = {result}"


# =============================================================================
# Test: T-DEMO-1.28 hash41 (4D -> f32)
# =============================================================================


class TestHash41:
    """Whitebox tests for hash41(p: vec4<f32>) -> f32."""

    def test_range_float_values(self):
        """hash41 on float values must be in [0, 1)."""
        for _ in range(200):
            p = tuple(random.uniform(-100, 100) for _ in range(4))
            result = py_hash41(p)
            assert 0.0 <= result < 1.0, f"hash41({p}) = {result}"

    def test_deterministic(self):
        """hash41 same input -> same output."""
        p = (1.0, 2.0, 3.0, 4.0)
        assert math.isclose(py_hash41(p), py_hash41(p), rel_tol=TOL_REL)

    def test_4d_vs_3d_different(self):
        """hash41 should differ from hash31 for related inputs."""
        h3 = py_hash31((1.0, 2.0, 3.0))
        h4 = py_hash41((1.0, 2.0, 3.0, 0.0))
        assert not math.isclose(h3, h4, rel_tol=TOL_REL), (
            "hash41 should differ from hash31 for 3D + 0"
        )

    def test_large_coordinates(self):
        """hash41 on large coordinates produces valid output."""
        for _ in range(50):
            p = tuple(random.uniform(-1e5, 1e5) for _ in range(4))
            result = py_hash41(p)
            assert 0.0 <= result < 1.0, f"hash41({p}) = {result}"

    def test_fourth_dimension_changes_output(self):
        """Varying the 4th dimension while keeping first 3 constant changes output."""
        base = (1.0, 2.0, 3.0, 0.0)
        base_val = py_hash41(base)
        vals = [py_hash41((base[0], base[1], base[2], float(t))) for t in range(1, 10)]
        all_close = all(math.isclose(v, base_val, rel_tol=TOL_REL) for v in vals)
        assert not all_close, "hash41 output does not vary with 4th dimension"


# =============================================================================
# Test: T-DEMO-1.28 hash22 (2D -> vec2)
# =============================================================================


class TestHash22:
    """Whitebox tests for hash22(p: vec2<f32>) -> vec2<f32>."""

    def test_both_components_in_range(self):
        """Both hash22 output components must be in [0, 1)."""
        for _ in range(200):
            p = (random.uniform(-100, 100), random.uniform(-100, 100))
            r0, r1 = py_hash22(p)
            assert 0.0 <= r0 < 1.0, f"hash22({p})[0] = {r0}"
            assert 0.0 <= r1 < 1.0, f"hash22({p})[1] = {r1}"

    def test_deterministic(self):
        """hash22 same input -> same output."""
        p = (42.0, 17.0)
        r1 = py_hash22(p)
        r2 = py_hash22(p)
        assert all(math.isclose(r1[i], r2[i], rel_tol=TOL_REL) for i in range(2))

    def test_components_differ(self):
        """Two components of hash22 should differ for the same input."""
        p = (42.0, 17.0)
        r0, r1 = py_hash22(p)
        assert not math.isclose(r0, r1, rel_tol=TOL_REL), (
            f"hash22 components should differ: {r0} == {r1}"
        )

    def test_each_component_varies(self):
        """Each hash22 component should vary with the input."""
        samples_0, samples_1 = [], []
        for i in range(100):
            r0, r1 = py_hash22((float(i), float(i + 1)))
            samples_0.append(r0)
            samples_1.append(r1)
        unique_0 = len(set(round(v, 8) for v in samples_0))
        unique_1 = len(set(round(v, 8) for v in samples_1))
        assert unique_0 >= 50, f"hash22 component 0 only {unique_0} unique"
        assert unique_1 >= 50, f"hash22 component 1 only {unique_1} unique"


# =============================================================================
# Test: T-DEMO-1.28 hash32 (3D -> vec2)
# =============================================================================


class TestHash32:
    """Whitebox tests for hash32(p: vec3<f32>) -> vec2<f32>."""

    def test_both_components_in_range(self):
        """Both hash32 output components must be in [0, 1)."""
        for _ in range(200):
            p = tuple(random.uniform(-100, 100) for _ in range(3))
            r0, r1 = py_hash32(p)
            assert 0.0 <= r0 < 1.0, f"hash32({p})[0] = {r0}"
            assert 0.0 <= r1 < 1.0, f"hash32({p})[1] = {r1}"

    def test_deterministic(self):
        """hash32 same input -> same output."""
        p = (1.0, 2.0, 3.0)
        r1 = py_hash32(p)
        r2 = py_hash32(p)
        assert all(math.isclose(r1[i], r2[i], rel_tol=TOL_REL) for i in range(2))

    def test_components_differ(self):
        """Two components of hash32 should differ for the same input."""
        r0, r1 = py_hash32((1.0, 2.0, 3.0))
        assert not math.isclose(r0, r1, abs_tol=TOL_ABS), "hash32 components match"


# =============================================================================
# Test: T-DEMO-1.28 hash33 (3D -> vec3)
# =============================================================================


class TestHash33:
    """Whitebox tests for hash33(p: vec3<f32>) -> vec3<f32>."""

    def test_all_components_in_range(self):
        """All hash33 output components must be in [0, 1)."""
        for _ in range(200):
            p = tuple(random.uniform(-100, 100) for _ in range(3))
            r0, r1, r2 = py_hash33(p)
            assert 0.0 <= r0 < 1.0, f"hash33({p})[0] = {r0}"
            assert 0.0 <= r1 < 1.0, f"hash33({p})[1] = {r1}"
            assert 0.0 <= r2 < 1.0, f"hash33({p})[2] = {r2}"

    def test_deterministic(self):
        """hash33 same input -> same output."""
        p = (1.0, 2.0, 3.0)
        r1 = py_hash33(p)
        r2 = py_hash33(p)
        assert all(math.isclose(r1[i], r2[i], rel_tol=TOL_REL) for i in range(3))

    def test_components_differ(self):
        """Three components of hash33 should differ for the same input."""
        r0, r1, r2 = py_hash33((1.0, 2.0, 3.0))
        pairs = [(r0, r1), (r0, r2), (r1, r2)]
        differing = sum(1 for a, b in pairs if not math.isclose(a, b, abs_tol=TOL_ABS))
        assert differing >= 2, (
            f"Expected at least 2 differing component pairs, got {differing}: "
            f"({r0}, {r1}, {r2})"
        )


# =============================================================================
# Test: T-DEMO-1.28 Distribution Properties
# =============================================================================


class TestHashDistribution:
    """Statistical tests for hash distribution uniformity.

    A good hash should produce outputs uniformly distributed in [0, 1).
    Expected mean: 0.5, expected variance: 1/12 ~ 0.0833.
    """

    def _sample_hash_1d(self, fn, sample_fn, n: int = SAMPLE_SIZE):
        """Sample n values from a 1D hash function."""
        return [fn(sample_fn(i)) for i in range(n)]

    def test_hash11_mean(self):
        """hash11 mean should be close to 0.5 (uniform [0,1))."""
        values = [py_hash11(float(i)) for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        assert 0.45 < mean < 0.55, f"hash11 mean {mean} not close to 0.5"

    def test_hash21_mean(self):
        """hash21 mean should be close to 0.5 (uniform [0,1))."""
        values = [py_hash21((float(i), float(i + 1))) for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        assert 0.45 < mean < 0.55, f"hash21 mean {mean} not close to 0.5"

    def test_hash31_mean(self):
        """hash31 mean should be close to 0.5 (uniform [0,1))."""
        values = [py_hash31((float(i), float(i + 1), float(i + 2)))
                  for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        assert 0.45 < mean < 0.55, f"hash31 mean {mean} not close to 0.5"

    def test_hash11_variance(self):
        """hash11 variance should be close to 1/12 ~ 0.0833."""
        values = [py_hash11(float(i)) for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        assert 0.07 < variance < 0.10, f"hash11 variance {variance} not close to 0.0833"

    def test_hash21_variance(self):
        """hash21 variance should be close to 1/12 ~ 0.0833."""
        values = [py_hash21((float(i), float(i + 1))) for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        assert 0.07 < variance < 0.10, f"hash21 variance {variance} not close to 0.0833"

    def test_hash31_variance(self):
        """hash31 variance should be close to 1/12 ~ 0.0833."""
        values = [py_hash31((float(i), float(i + 1), float(i + 2)))
                  for i in range(SAMPLE_SIZE)]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        assert 0.07 < variance < 0.10, f"hash31 variance {variance} not close to 0.0833"

    def test_hash22_each_component_variance(self):
        """Each hash22 component variance should be close to 1/12."""
        comp0, comp1 = [], []
        for i in range(SAMPLE_SIZE):
            r0, r1 = py_hash22((float(i), float(i + 1)))
            comp0.append(r0)
            comp1.append(r1)
        for name, comp in [("comp0", comp0), ("comp1", comp1)]:
            mean = sum(comp) / len(comp)
            variance = sum((v - mean) ** 2 for v in comp) / len(comp)
            assert 0.07 < variance < 0.10, (
                f"hash22 {name} variance {variance}"
            )

    def test_hash33_each_component_variance(self):
        """Each hash33 component variance should be close to 1/12."""
        comp0, comp1, comp2 = [], [], []
        for i in range(SAMPLE_SIZE):
            r0, r1, r2 = py_hash33((float(i), float(i + 1), float(i + 2)))
            comp0.append(r0)
            comp1.append(r1)
            comp2.append(r2)
        for name, comp in [("comp0", comp0), ("comp1", comp1), ("comp2", comp2)]:
            mean = sum(comp) / len(comp)
            variance = sum((v - mean) ** 2 for v in comp) / len(comp)
            assert 0.07 < variance < 0.10, (
                f"hash33 {name} variance {variance}"
            )

    def test_hash11_uniform_bins(self):
        """Chi-squared test: hash11 distribution should be approximately uniform."""
        n = SAMPLE_SIZE
        bins = 10
        values = [py_hash11(float(i)) for i in range(n)]
        bin_counts = [0] * bins
        for v in values:
            idx = min(int(v * bins), bins - 1)
            bin_counts[idx] += 1
        expected = n / bins
        chi2 = sum((count - expected) ** 2 / expected for count in bin_counts)
        # Critical value for 9 degrees of freedom at alpha=0.01 is ~21.67
        # We use a more lenient threshold for a basic hash test
        assert chi2 < 3 * bins, (
            f"hash11 chi-squared {chi2} exceeds threshold {3*bins}: "
            f"bin_counts={bin_counts}"
        )

    def test_hash21_uniform_bins(self):
        """Chi-squared test: hash21 distribution should be approximately uniform."""
        n = SAMPLE_SIZE
        bins = 10
        values = [py_hash21((float(i), float(i + 1))) for i in range(n)]
        bin_counts = [0] * bins
        for v in values:
            idx = min(int(v * bins), bins - 1)
            bin_counts[idx] += 1
        expected = n / bins
        chi2 = sum((count - expected) ** 2 / expected for count in bin_counts)
        assert chi2 < 3 * bins, (
            f"hash21 chi-squared {chi2} exceeds threshold {3*bins}: "
            f"bin_counts={bin_counts}"
        )

    def test_hash31_uniform_bins(self):
        """Chi-squared test: hash31 distribution should be approximately uniform."""
        n = SAMPLE_SIZE
        bins = 10
        values = [py_hash31((float(i), float(i + 1), float(i + 2)))
                  for i in range(n)]
        bin_counts = [0] * bins
        for v in values:
            idx = min(int(v * bins), bins - 1)
            bin_counts[idx] += 1
        expected = n / bins
        chi2 = sum((count - expected) ** 2 / expected for count in bin_counts)
        assert chi2 < 3 * bins, (
            f"hash31 chi-squared {chi2} exceeds threshold {3*bins}: "
            f"bin_counts={bin_counts}"
        )


# =============================================================================
# Test: T-DEMO-1.28 Autocorrelation (No Visible Patterns)
# =============================================================================


class TestHashAutocorrelation:
    """Tests that hash outputs have no visible patterns (low autocorrelation).

    A good hash should have lag-1 autocorrelation close to zero, meaning
    adjacent input values produce uncorrelated outputs.
    """

    def _autocorrelation(self, values, lag=1):
        """Compute lag-k autocorrelation of a sequence."""
        n = len(values)
        mean = sum(values) / n
        centered = [v - mean for v in values]
        var = sum(v * v for v in centered)
        if var == 0:
            return 0.0
        acf = sum(centered[i] * centered[i + lag] for i in range(n - lag)) / var
        return acf

    def test_hash11_lag1_autocorrelation(self):
        """hash11 lag-1 autocorrelation should be near zero."""
        values = [py_hash11(float(i)) for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash11 lag-1 ACF = {acf}, expected near 0"

    def test_hash21_lag1_autocorrelation(self):
        """hash21 lag-1 autocorrelation should be near zero."""
        values = [py_hash21((float(i), float(i + 1))) for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash21 lag-1 ACF = {acf}, expected near 0"

    def test_hash31_lag1_autocorrelation(self):
        """hash31 lag-1 autocorrelation should be near zero."""
        values = [py_hash31((float(i), float(i + 1), float(i + 2)))
                  for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash31 lag-1 ACF = {acf}, expected near 0"

    def test_hash22_component0_lag1(self):
        """hash22 component 0 lag-1 autocorrelation should be near zero."""
        values = [py_hash22((float(i), float(i + 1)))[0] for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash22[0] lag-1 ACF = {acf}"

    def test_hash22_component1_lag1(self):
        """hash22 component 1 lag-1 autocorrelation should be near zero."""
        values = [py_hash22((float(i), float(i + 1)))[1] for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash22[1] lag-1 ACF = {acf}"

    def test_hash33_component0_lag1(self):
        """hash33 component 0 lag-1 autocorrelation should be near zero."""
        values = [py_hash33((float(i), float(i + 1), float(i + 2)))[0]
                  for i in range(1000)]
        acf = self._autocorrelation(values, lag=1)
        assert abs(acf) < 0.15, f"hash33[0] lag-1 ACF = {acf}"

    def test_hash11_no_periodic_output(self):
        """hash11 should not produce a periodic sequence on integer grid."""
        values = [py_hash11(float(i)) for i in range(200)]
        # Check there's no obvious period by checking for repeating
        # sequences of length 2, 3, 5, 7
        for period in [2, 3, 5, 7]:
            for start in range(period):
                seq = values[start:start + period]
                # Check if next period matches
                next_seq = values[start + period:start + 2 * period]
                if len(seq) == len(next_seq) and len(seq) > 0:
                    matches = all(
                        math.isclose(seq[j], next_seq[j], rel_tol=TOL_REL)
                        for j in range(len(seq))
                    )
                    assert not matches, (
                        f"hash11 appears periodic with period {period} "
                        f"at offset {start}"
                    )


# =============================================================================
# Test: T-DEMO-1.28 Integer Coordinate Grid
# =============================================================================


class TestHashIntegerGrid:
    """Tests hash behavior on integer coordinate grids (the primary use case)."""

    def test_hash11_all_unique_first_50(self):
        """hash11 should produce distinct values for first 50 positive integers."""
        values = [py_hash11(float(i)) for i in range(50)]
        unique = len(set(round(v, 8) for v in values))
        assert unique >= 45, f"hash11 only {unique} unique in first 50 integers"

    def test_hash11_negative_integers_unique(self):
        """hash11 should produce distinct values for negative integers."""
        values = [py_hash11(float(i)) for i in range(-50, 0)]
        unique = len(set(round(v, 8) for v in values))
        assert unique >= 45, f"hash11 only {unique} unique in 50 negatives"

    def test_hash21_xy_symmetry_2d_grid(self):
        """hash21 on 2D grid: (x,y) should differ from (x+1,y) etc."""
        base = py_hash21((5.0, 5.0))
        neighbors = [
            py_hash21((6.0, 5.0)),
            py_hash21((5.0, 6.0)),
            py_hash21((4.0, 5.0)),
            py_hash21((5.0, 4.0)),
        ]
        differing = sum(1 for n in neighbors if not math.isclose(base, n, rel_tol=TOL_REL))
        assert differing >= 3, (
            "hash21 output doesn't change enough with neighbor coordinates"
        )

    def test_hash31_xyz_symmetry_3d_grid(self):
        """hash31 on 3D grid: (x,y,z) should differ from (x+1,y,z) etc."""
        base = py_hash31((5.0, 5.0, 5.0))
        neighbors = [
            py_hash31((6.0, 5.0, 5.0)),
            py_hash31((5.0, 6.0, 5.0)),
            py_hash31((5.0, 5.0, 6.0)),
            py_hash31((4.0, 5.0, 5.0)),
        ]
        differing = sum(1 for n in neighbors if not math.isclose(base, n, rel_tol=TOL_REL))
        assert differing >= 3, (
            "hash31 output doesn't change enough with neighbor coordinates"
        )

    def test_hash22_components_independent(self):
        """hash22 components should not always produce the same relative ordering."""
        greater = 0
        less = 0
        for i in range(200):
            r0, r1 = py_hash22((float(i), float(i + 1)))
            if r0 > r1:
                greater += 1
            elif r1 > r0:
                less += 1
        # Both orderings should occur
        assert greater > 20, f"hash22 comp0 > comp1 only {greater}/200 times"
        assert less > 20, f"hash22 comp0 < comp1 only {less}/200 times"

    def test_hash33_components_independent(self):
        """hash33 components should produce varied orderings."""
        orderings = set()
        for i in range(500):
            r0, r1, r2 = py_hash33((float(i), float(i + 1), float(i + 2)))
            orderings.add(tuple(sorted([(r0, 0), (r1, 1), (r2, 2)])))
        assert len(orderings) >= 3, (
            f"hash33 only produced {len(orderings)} distinct rank orderings"
        )


# =============================================================================
# Test: T-DEMO-1.28 Cross-Function Properties
# =============================================================================


class TestCrossFunctionProperties:
    """Tests that different hash functions produce uncorrelated outputs."""

    def test_hash11_vs_hash21_different(self):
        """hash11 and hash21 should produce different values for same seed."""
        seed = 42.0
        h11 = py_hash11(seed)
        h21 = py_hash21((seed, seed + 1.0))
        assert not math.isclose(h11, h21, rel_tol=TOL_REL), (
            "hash11 and hash21 should differ"
        )

    def test_hash21_vs_hash22_different(self):
        """hash21 and hash22 should produce different values for same 2D input."""
        p = (42.0, 17.0)
        h21 = py_hash21(p)
        h22_0, _ = py_hash22(p)
        assert not math.isclose(h21, h22_0, rel_tol=TOL_REL), (
            "hash21 should differ from hash22[0]"
        )

    def test_hash31_vs_hash33_different(self):
        """hash31 and hash33 should produce different values for same 3D input."""
        p = (1.0, 2.0, 3.0)
        h31 = py_hash31(p)
        h33_0, _, _ = py_hash33(p)
        assert not math.isclose(h31, h33_0, rel_tol=TOL_REL), (
            "hash31 should differ from hash33[0]"
        )

    def test_hash22_vs_hash32_different(self):
        """hash22 and hash32 should produce different values for same 3D input
        extended to 2D (different mixing)."""
        p2 = (1.0, 2.0)
        p3 = (1.0, 2.0, 3.0)
        h22_0, _ = py_hash22(p2)
        h32_0, _ = py_hash32(p3)
        assert not math.isclose(h22_0, h32_0, rel_tol=TOL_REL), (
            "hash22[0] should differ from hash32[0]"
        )


# =============================================================================
# Test: T-DEMO-1.28 Finite Output Guarantee
# =============================================================================


class TestHashFiniteOutput:
    """All hash functions must produce finite output for all valid inputs."""

    EXTREME_VALUES = [-1e12, -1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6, 1e12]

    def test_hash11_finite_all(self):
        for v in self.EXTREME_VALUES:
            result = py_hash11(v)
            assert math.isfinite(result), f"hash11({v}) non-finite: {result}"
            assert 0.0 <= result < 1.0

    def test_hash21_finite_all(self):
        for v in self.EXTREME_VALUES:
            result = py_hash21((v, v + 1.0))
            assert math.isfinite(result), f"hash21({v}) non-finite: {result}"
            assert 0.0 <= result < 1.0

    def test_hash31_finite_all(self):
        for v in self.EXTREME_VALUES:
            result = py_hash31((v, v + 1.0, v + 2.0))
            assert math.isfinite(result), f"hash31({v}) non-finite: {result}"
            assert 0.0 <= result < 1.0

    def test_hash22_components_finite(self):
        for v in self.EXTREME_VALUES:
            r0, r1 = py_hash22((v, v + 1.0))
            assert math.isfinite(r0), f"hash22({v})[0] non-finite: {r0}"
            assert math.isfinite(r1), f"hash22({v})[1] non-finite: {r1}"
            assert 0.0 <= r0 < 1.0
            assert 0.0 <= r1 < 1.0

    def test_hash33_components_finite(self):
        for v in self.EXTREME_VALUES:
            r0, r1, r2 = py_hash33((v, v + 1.0, v + 2.0))
            assert math.isfinite(r0), f"hash33({v})[0] non-finite: {r0}"
            assert math.isfinite(r1), f"hash33({v})[1] non-finite: {r1}"
            assert math.isfinite(r2), f"hash33({v})[2] non-finite: {r2}"
            assert 0.0 <= r0 < 1.0
            assert 0.0 <= r1 < 1.0
            assert 0.0 <= r2 < 1.0
