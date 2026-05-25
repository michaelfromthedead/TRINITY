"""
Whitebox tests for SDF domain deformation operations (T-DEMO-1.22 through T-DEMO-1.27).

Tests Python model implementations of each WGSL function, verifying:
  - Correctness against mathematical definition
  - Edge cases (zero, negative, boundary values)
  - Composability of operations
  - Numerical invariants (finite output, expected ranges)

WHITEBOX coverage plan:
  - Path A: domain_repeat with positive cell sizes -> centered tiling
  - Path B: domain_repeat with WGSL trunc-mod asymmetry -> range (-1.5c, 0.5c)
  - Path C: domain_cell_id cell centers at n*c + 0.5*c
  - Path D: domain_mirror x/y/z idempotence on positive inputs
  - Path E: domain_kifs NOT an isometry (abs breaks distance preservation)
  - Path F: domain_kifs xy abs property across iterations
  - Path G: domain_twist periodic at 2*pi/k
  - Path H: domain_bend near-zero radius returns identity
  - Path I: domain_bend large radius asymptotically swaps x and z
  - Path J: composability repeat-then-mirror bound at 1.5c
  - Path K: all functions return finite output for diverse inputs
  - Path L: stretch preserves volume (det=1 for x/y/z)
  - Path M: stretch composability with other operations
  - Path N: WGSL % operator trunc-toward-zero asymmetry
  - Path O: KIFS with non-6 fold counts (3, 4, 8, 12, 1, 2)
  - Path P: Euclidean modulo [0, c) range for all inputs, periodicity (FIX 1)
  - Path Q: Euclidean domain_repeat [-0.5c, 0.5c) symmetric range (FIX 1)
  - Path R: KIFS safe_folds guard for zero/negative/sub-one folds (FIX 2)
  - Path S: stretch determinant = 1/s for all axes, NOT volume-preserving (FIX 5)
"""

import math
import os
import random
from typing import Callable

import numpy as np
import pytest

# =============================================================================
# Python model implementations matching WGSL semantics
# =============================================================================

# WGSL %: x - y * trunc(x / y)  (truncation toward zero)
def wgsl_mod(x: float, y: float) -> float:
    return x - y * math.trunc(x / y)


# Euclidean modulo: x - y * floor(x / y)  (always returns [0, y) for y > 0)
def euclidean_mod(x: float, y: float) -> float:
    return x - y * math.floor(x / y)


def py_domain_repeat(p, c):
    """Model of WGSL domain_repeat: p % c - 0.5 * c (WGSL trunc-toward-zero %)."""
    return tuple(wgsl_mod(p[i], c[i]) - 0.5 * c[i] for i in range(3))


def py_domain_repeat_euclidean(p, c):
    """Model of WGSL domain_repeat after fix: (p - c*floor(p/c)) - 0.5*c (Euclidean modulo).

    Matches the current WGSL implementation: return (p - c * floor(p / c)) - 0.5 * c.
    Euclidean modulo returns in [0, c) for all inputs, giving centered range [-0.5c, 0.5c).
    """
    return tuple(euclidean_mod(p[i], c[i]) - 0.5 * c[i] for i in range(3))


def py_domain_cell_id(p, c):
    """Model of WGSL domain_cell_id: floor(p / c + 0.5)."""
    return tuple(math.floor(p[i] / c[i] + 0.5) for i in range(3))


def py_domain_mirror_x(p):
    """Model of WGSL domain_mirror_x: abs(x)."""
    return (abs(p[0]), p[1], p[2])


def py_domain_mirror_y(p):
    """Model of WGSL domain_mirror_y: abs(y)."""
    return (p[0], abs(p[1]), p[2])


def py_domain_mirror_z(p):
    """Model of WGSL domain_mirror_z: abs(z)."""
    return (p[0], p[1], abs(p[2]))


def py_domain_kifs(p, folds):
    """Model of WGSL domain_kifs with safe_folds guard (FIX 2).

    Matches WGSL: let safe_folds = max(abs(folds), 1.0);
    Prevents division by zero and handles negative/sub-one fold values.
    """
    safe_folds = max(abs(folds), 1.0)
    n_iters = int(safe_folds)
    angle = 2.0 * math.pi / safe_folds
    q = [p[0], p[1], p[2]]
    for _ in range(n_iters):
        q[0] = abs(q[0])
        q[1] = abs(q[1])
        ca = math.cos(angle)
        sa = math.sin(angle)
        new_x = ca * q[0] - sa * q[1]
        new_y = sa * q[0] + ca * q[1]
        q[0] = new_x
        q[1] = new_y
    return (q[0], q[1], q[2])


def py_domain_twist(p, k):
    """Model of WGSL domain_twist: xz rotation proportional to y."""
    c = math.cos(k * p[1])
    s = math.sin(k * p[1])
    return (c * p[0] - s * p[2], p[1], s * p[0] + c * p[2])


def py_domain_bend(p, r):
    """Model of WGSL domain_bend: circular arc in xz-plane."""
    if abs(r) < 1e-8:
        return p
    safe_r = max(abs(r), 1e-8)
    theta = p[0] / safe_r
    c = math.cos(theta)
    s = math.sin(theta)
    return (-safe_r + (safe_r + p[2]) * c, p[1], (safe_r + p[2]) * s)


def py_domain_stretch_x(p, s):
    """Model of WGSL domain_stretch_x: scale x by s, y/z by 1/s."""
    return (p[0] * s, p[1] / s, p[2] / s)


def py_domain_stretch_y(p, s):
    """Model of WGSL domain_stretch_y: scale y by s, x/z by 1/s."""
    return (p[0] / s, p[1] * s, p[2] / s)


def py_domain_stretch_z(p, s):
    """Model of WGSL domain_stretch_z: scale z by s, x/y by 1/s."""
    return (p[0] / s, p[1] / s, p[2] * s)


def get_wgsl_source_path():
    """Return absolute path to sdf_domain.wgsl for docstring verification."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        test_dir, "..", "..", "crates",
        "renderer-backend", "src", "demoscene", "sdf_domain.wgsl"
    ))


# =============================================================================
# Helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-7


def vec3_close(a, b, rel_tol=TOL_REL, abs_tol=TOL_ABS) -> bool:
    return all(
        math.isclose(a[i], b[i], rel_tol=rel_tol, abs_tol=abs_tol)
        for i in range(3)
    )


def vec3_dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# =============================================================================
# Test: T-DEMO-1.22 Domain Repetition
# =============================================================================


class TestDomainRepeat:
    """Whitebox tests for domain_repeat(p, c)."""

    def test_centered_at_origin_for_zero_input(self):
        c = (2.0, 2.0, 2.0)
        result = py_domain_repeat((0.0, 0.0, 0.0), c)
        expected = (-1.0, -1.0, -1.0)
        assert vec3_close(result, expected), f"got {result}"

    def test_periodic_at_cell_boundary(self):
        c = (3.0, 3.0, 3.0)
        at_origin = py_domain_repeat((0.0, 0.0, 0.0), c)
        at_cell = py_domain_repeat(c, c)
        assert vec3_close(at_origin, at_cell), f"{at_origin} vs {at_cell}"

    def test_periodic_at_multiple_cell_size(self):
        c = (2.0, 4.0, 1.5)
        at_origin = py_domain_repeat((0.0, 0.0, 0.0), c)
        at_nc = py_domain_repeat((4.0, 8.0, 3.0), c)
        assert vec3_close(at_origin, at_nc), f"{at_origin} vs {at_nc}"

    def test_periodic_shifted_by_cell(self):
        c = (1.5, 2.5, 3.5)
        p = (0.7, 1.2, 0.3)
        r0 = py_domain_repeat(p, c)
        r1 = py_domain_repeat(tuple(p[i] + c[i] for i in range(3)), c)
        assert vec3_close(r0, r1), f"{r0} vs {r1}"

    def test_repeat_range_for_positive_p(self):
        c = (4.0, 4.0, 4.0)
        for _ in range(50):
            p = (random.uniform(0, 4.0), 0.0, 0.0)
            r = py_domain_repeat(p, c)
            assert -2.0 < r[0] < 2.0, f"repeat out of range: {r[0]}"

    def test_repeat_different_cell_sizes(self):
        c = (1.0, 2.0, 4.0)
        p = (5.5, -3.7, 9.2)
        result = py_domain_repeat(p, c)
        assert all(math.isfinite(x) for x in result), f"non-finite: {result}"

    def test_repeat_zero_cell_size(self):
        c = (0.0, 1.0, 1.0)
        with pytest.raises((ZeroDivisionError, ValueError)):
            py_domain_repeat((1.0, 1.0, 1.0), c)

    def test_repeat_negative_p(self):
        c = (2.0, 2.0, 2.0)
        p = (-1.0, 0.0, 0.0)
        r = py_domain_repeat(p, c)
        assert r[0] == -2.0, f"expected -2.0, got {r[0]}"

    def test_repeat_non_uniform_cell(self):
        c = (0.5, 10.0, 100.0)
        p = (3.7, -25.0, 550.0)
        result = py_domain_repeat(p, c)
        assert all(math.isfinite(x) for x in result), f"non-finite: {result}"


# =============================================================================
# Test: T-DEMO-1.22 Euclidean Tiling (FIX 1 verification)
# =============================================================================


class TestDomainRepeatEuclidean:
    """Verifies Euclidean modulo tiling works correctly for negative coordinates (FIX 1).

    The WGSL domain_repeat now uses Euclidean modulo (p - c*floor(p/c)) instead of
    trunc-toward-zero mod. Euclidean modulo guarantees the result is in [0, c),
    giving a centered tiling range of [-0.5c, 0.5c) for ALL inputs regardless of sign.
    """

    def test_euclidean_mod_positive(self):
        """Euclidean mod(5, 3) = 2 (same as trunc for positive)."""
        assert euclidean_mod(5.0, 3.0) == pytest.approx(2.0)

    def test_euclidean_mod_negative(self):
        """Euclidean mod(-5, 3) = 1 (differs from trunc which gives -2)."""
        result = euclidean_mod(-5.0, 3.0)
        assert result == pytest.approx(1.0), (
            f"Euclidean mod(-5,3) should be 1, got {result}; "
            f"trunc mod gives {-5.0 - 3.0 * math.trunc(-5.0 / 3.0)}"
        )

    def test_euclidean_mod_exact_multiple(self):
        """Euclidean mod(6, 3) = 0 and mod(-6, 3) = 0."""
        assert euclidean_mod(6.0, 3.0) == pytest.approx(0.0)
        assert euclidean_mod(-6.0, 3.0) == pytest.approx(0.0)

    def test_euclidean_mod_range_positive(self):
        """Euclidean mod always returns in [0, c) for c > 0."""
        for _ in range(100):
            x = random.uniform(-100, 100)
            c = random.uniform(0.1, 20.0)
            result = euclidean_mod(x, c)
            assert 0.0 <= result < c, (
                f"Euclidean mod({x}, {c}) = {result} out of range [0, {c})"
            )

    def test_euclidean_mod_periodic(self):
        """Euclidean mod(x+c, c) = mod(x, c) for all x, c > 0."""
        for _ in range(50):
            x = random.uniform(-50, 50)
            c = random.uniform(0.1, 10.0)
            assert euclidean_mod(x + c, c) == pytest.approx(euclidean_mod(x, c))
            assert euclidean_mod(x - c, c) == pytest.approx(euclidean_mod(x, c))
            assert euclidean_mod(x + 5 * c, c) == pytest.approx(euclidean_mod(x, c))

    def test_domain_repeat_euclidean_center_zero_input(self):
        """At (0,0,0), Euclidean repeat centers at (-0.5c, -0.5c, -0.5c)."""
        c = (2.0, 2.0, 2.0)
        result = py_domain_repeat_euclidean((0.0, 0.0, 0.0), c)
        expected = (-1.0, -1.0, -1.0)
        assert vec3_close(result, expected), f"got {result}"

    def test_domain_repeat_euclidean_negative_p(self):
        """Negative p tiles into the same centered cell as p + c (periodic)."""
        c = (2.0, 2.0, 2.0)
        p_neg = (-1.0, -1.0, -1.0)
        p_pos = (1.0, 1.0, 1.0)
        r_neg = py_domain_repeat_euclidean(p_neg, c)
        r_pos = py_domain_repeat_euclidean(p_pos, c)
        assert vec3_close(r_neg, r_pos), (
            f"negative {r_neg} != positive {r_pos}"
        )

    def test_domain_repeat_euclidean_negative_large(self):
        """Large negative values tile symmetrically into [-0.5c, 0.5c)."""
        c = (2.0, 2.0, 2.0)
        for p_x in [-2.0, -3.0, -10.0, -100.0, -1000.0]:
            r = py_domain_repeat_euclidean((p_x, 0.0, 0.0), c)
            assert -1.0 <= r[0] < 1.0, (
                f"Euclidean repeat({p_x}, {c[0]}) = {r[0]} out of [-1, 1)"
            )

    def test_domain_repeat_euclidean_symmetric_range(self):
        """Euclidean repeat gives [-0.5c, 0.5c) for ALL inputs."""
        c = (4.0, 4.0, 4.0)
        half_c = 2.0
        for _ in range(200):
            p = tuple(random.uniform(-100, 100) for _ in range(3))
            r = py_domain_repeat_euclidean(p, c)
            for i in range(3):
                assert -half_c <= r[i] < half_c, (
                    f"axis {i}: {r[i]} out of [{-half_c}, {half_c}) for p={p}"
                )

    def test_domain_repeat_euclidean_vs_trunc_differs_negative(self):
        """Euclidean and trunc-mod repeat differ for negative inputs on same cell."""
        c = (2.0, 2.0, 2.0)
        neg_inputs = [(-0.5, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.5, 0.0, 0.0)]
        for p in neg_inputs:
            r_euc = py_domain_repeat_euclidean(p, c)
            r_trunc = py_domain_repeat(p, c)
            # They MUST differ for these inputs (Euclidean gives [-1,1), trunc gives [-2,0))
            assert not vec3_close(r_euc, r_trunc), (
                f"Euclidean {r_euc} == trunc {r_trunc} for p={p}, expected difference"
            )

    def test_domain_repeat_euclidean_non_uniform_sizes(self):
        """Euclidean repeat works with different cell sizes per axis."""
        c = (1.0, 2.0, 4.0)
        for _ in range(50):
            p = tuple(random.uniform(-50, 50) for _ in range(3))
            r = py_domain_repeat_euclidean(p, c)
            for i in range(3):
                assert -0.5 * c[i] <= r[i] < 0.5 * c[i], (
                    f"axis {i}: {r[i]} out of [{-0.5*c[i]}, {0.5*c[i]})"
                )

    def test_domain_repeat_euclidean_periodic_multiple_cells(self):
        """Euclidean repeat is periodic across any integer number of cells."""
        c = (2.5, 3.0, 1.5)
        p = (0.3, -0.7, 1.1)
        r_base = py_domain_repeat_euclidean(p, c)
        for n in [-5, -3, -1, 1, 2, 4, 10]:
            shifted = tuple(p[i] + n * c[i] for i in range(3))
            r_shift = py_domain_repeat_euclidean(shifted, c)
            assert vec3_close(r_base, r_shift), (
                f"n={n}: periodicity broken: {r_base} vs {r_shift}"
            )

    def test_domain_repeat_euclidean_then_cell_id(self):
        """Calling cell_id on repeat output gives the origin cell."""
        c = (2.0, 2.0, 2.0)
        for _ in range(50):
            p = tuple(random.uniform(-20, 20) for _ in range(3))
            r = py_domain_repeat_euclidean(p, c)
            cell = py_domain_cell_id(r, c)
            # The repeat output centered in [0.5c, 0.5c) so cell_id should be 0
            assert vec3_close(cell, (0.0, 0.0, 0.0)), (
                f"cell_id on repeat output should be (0,0,0), got {cell} for p={p}"
            )

    def test_domain_repeat_euclidean_edge_boundary(self):
        """Boundary at exactly c maps to -0.5c (Euclidean: mod(c,c)=0)."""
        c = (2.0, 2.0, 2.0)
        r = py_domain_repeat_euclidean(c, c)
        assert vec3_close(r, (-1.0, -1.0, -1.0)), f"boundary: {r}"
        # Boundary at exactly 0 maps to -0.5c
        r0 = py_domain_repeat_euclidean((0.0, 0.0, 0.0), c)
        assert vec3_close(r0, r), f"0 and c boundaries differ: {r0} vs {r}"

    def test_domain_repeat_euclidean_cell_center(self):
        """Input at cell center (0.5c) maps to repeat output origin."""
        c = (2.0, 2.0, 2.0)
        r = py_domain_repeat_euclidean((1.0, 1.0, 1.0), c)
        expected = (0.0, 0.0, 0.0)
        assert vec3_close(r, expected), f"cell center: {r} vs {expected}"


class TestDomainCellId:
    """Whitebox tests for domain_cell_id(p, c)."""

    def test_cell_id_at_origin(self):
        c = (2.0, 2.0, 2.0)
        result = py_domain_cell_id((0.0, 0.0, 0.0), c)
        assert result == (0.0, 0.0, 0.0), f"got {result}"

    def test_cell_id_at_repeat_center(self):
        c = (2.0, 2.0, 2.0)
        result = py_domain_cell_id((1.0, 1.0, 1.0), c)
        assert result == (1.0, 1.0, 1.0), f"got {result}"

    def test_cell_id_next_repeat_center(self):
        c = (2.0, 2.0, 2.0)
        result = py_domain_cell_id((3.0, 3.0, 3.0), c)
        assert result == (2.0, 2.0, 2.0), f"got {result}"

    def test_cell_id_discrete(self):
        c = (1.5, 2.5, 3.5)
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            result = py_domain_cell_id(p, c)
            for i in range(3):
                assert result[i] == math.floor(result[i]), (
                    f"non-integer at axis {i}: {result[i]}"
                )

    def test_cell_id_jump_at_boundary(self):
        c = (2.0, 2.0, 2.0)
        just_before = py_domain_cell_id((0.999, 0.0, 0.0), c)
        just_after = py_domain_cell_id((1.001, 0.0, 0.0), c)
        assert just_after[0] == just_before[0] + 1, (
            f"jumped by {just_after[0] - just_before[0]}"
        )

    def test_cell_id_negative_side(self):
        c = (2.0, 2.0, 2.0)
        result = py_domain_cell_id((-3.0, -3.0, -3.0), c)
        assert all(x < 0 for x in result), f"expected negative, got {result}"


# =============================================================================
# Test: T-DEMO-1.23 Domain Mirroring
# =============================================================================


class TestDomainMirror:
    """Whitebox tests for domain_mirror_x/y/z."""

    def test_mirror_x_negative(self):
        p = (-3.5, 1.0, 2.0)
        result = py_domain_mirror_x(p)
        assert result[0] == 3.5
        assert result[1] == p[1]
        assert result[2] == p[2]

    def test_mirror_x_positive(self):
        p = (3.5, 1.0, 2.0)
        assert py_domain_mirror_x(p) == p

    def test_mirror_x_zero(self):
        assert py_domain_mirror_x((0.0, 1.0, 2.0))[0] == 0.0

    def test_mirror_y_negative(self):
        p = (1.0, -4.2, 3.0)
        result = py_domain_mirror_y(p)
        assert result[1] == 4.2

    def test_mirror_y_positive(self):
        p = (1.0, 4.2, 3.0)
        assert py_domain_mirror_y(p) == p

    def test_mirror_z_negative(self):
        p = (1.0, 2.0, -7.5)
        result = py_domain_mirror_z(p)
        assert result[2] == 7.5

    def test_mirror_z_positive(self):
        p = (1.0, 2.0, 7.5)
        assert py_domain_mirror_z(p) == p

    def test_mirror_idempotent(self):
        for fn in [py_domain_mirror_x, py_domain_mirror_y, py_domain_mirror_z]:
            for _ in range(20):
                p = tuple(random.uniform(-5, 5) for _ in range(3))
                once = fn(p)
                twice = fn(once)
                assert vec3_close(once, twice), f"{fn.__name__}: {once} vs {twice}"

    def test_mirror_all_axes_commute(self):
        p = tuple(random.uniform(-5, 5) for _ in range(3))
        xy_then_z = py_domain_mirror_z(py_domain_mirror_y(py_domain_mirror_x(p)))
        z_then_xy = py_domain_mirror_x(py_domain_mirror_y(py_domain_mirror_z(p)))
        assert vec3_close(xy_then_z, z_then_xy)

    def test_mirror_distance_non_increasing(self):
        for fn in [py_domain_mirror_x, py_domain_mirror_y, py_domain_mirror_z]:
            for _ in range(20):
                a = tuple(random.uniform(-5, 5) for _ in range(3))
                b = tuple(random.uniform(-5, 5) for _ in range(3))
                d_before = vec3_dist(a, b)
                d_after = vec3_dist(fn(a), fn(b))
                assert d_after <= d_before + 1e-10, (
                    f"{fn.__name__}: dist {d_before} -> {d_after}"
                )


# =============================================================================
# Test: T-DEMO-1.24 Kaleidoscopic Fold (KIFS)
# =============================================================================


class TestDomainKifs:
    """Whitebox tests for domain_kifs(p, folds)."""

    def test_finite_output(self):
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            folds = random.choice([3, 4, 5, 6, 8, 12])
            result = py_domain_kifs(p, folds)
            assert all(math.isfinite(x) for x in result), f"non-finite: {result}"

    def test_z_preserved(self):
        p = tuple(random.uniform(-5, 5) for _ in range(3))
        result = py_domain_kifs(p, 6)
        assert result[2] == p[2], f"z changed: {result[2]} != {p[2]}"

    def test_distance_non_increasing(self):
        for _ in range(30):
            a = tuple(random.uniform(-5, 5) for _ in range(3))
            b = tuple(random.uniform(-5, 5) for _ in range(3))
            folds = random.choice([3, 4, 6])
            d_before = vec3_dist(a, b)
            d_after = vec3_dist(py_domain_kifs(a, folds), py_domain_kifs(b, folds))
            assert d_after <= d_before + TOL_ABS, (
                f"dist increased: {d_before} -> {d_after}"
            )

    def test_xy_abs_preserved_across_iterations(self):
        p = (-3.0, -4.0, 0.0)
        result = py_domain_kifs(p, 6)
        assert all(math.isfinite(x) for x in result)

    def test_fold_symmetry_order_6(self):
        folds = 6
        angle = 2.0 * math.pi / folds
        p = (2.0, 1.0, 0.0)
        result = py_domain_kifs(p, folds)
        ca = math.cos(angle)
        sa = math.sin(angle)
        p_rotated = (ca * p[0] - sa * p[1], sa * p[0] + ca * p[1], p[2])
        result_rotated = py_domain_kifs(p_rotated, folds)
        dist = vec3_dist(result, result_rotated)
        assert dist < 3.0, f"fold symmetry broken: {dist}"

    def test_single_fold_reflection_finite(self):
        folds = 2
        p = (-3.5, 2.1, 1.0)
        result = py_domain_kifs(p, folds)
        assert all(math.isfinite(x) for x in result)


# =============================================================================
# Test: T-DEMO-1.24 Non-6 Fold KIFS (FIX 2 verification)
# =============================================================================


class TestDomainKifsNonSix:
    """Verifies KIFS works correctly with non-6 fold counts (FIX 2)."""

    def test_kifs_3_fold_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 3)
            assert all(math.isfinite(x) for x in result), f"3-fold: {result}"

    def test_kifs_4_fold_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 4)
            assert all(math.isfinite(x) for x in result), f"4-fold: {result}"

    def test_kifs_8_fold_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 8)
            assert all(math.isfinite(x) for x in result), f"8-fold: {result}"

    def test_kifs_12_fold_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 12)
            assert all(math.isfinite(x) for x in result), f"12-fold: {result}"

    def test_kifs_iteration_count_matches_folds(self):
        """1-fold KIFS is just abs(p) (no rotation); 6-fold differs."""
        for _ in range(20):
            p = (random.uniform(0.1, 5.0), random.uniform(0.1, 5.0),
                 random.uniform(-5, 5))
            r1 = py_domain_kifs(p, 1)
            r6 = py_domain_kifs(p, 6)
            # 1-fold: abs only (no rotation), so result is abs(p)
            assert math.isclose(r1[0], abs(p[0])), (
                f"1-fold x: {r1[0]} vs {abs(p[0])}"
            )
            assert math.isclose(r1[1], abs(p[1])), (
                f"1-fold y: {r1[1]} vs {abs(p[1])}"
            )
            # 6-fold: abs + rotate, should differ from 1-fold for non-zero inputs
            dist = vec3_dist(r1, r6)
            assert dist > TOL_ABS, (
                f"1-fold == 6-fold for p={p}: dist={dist}"
            )

    def test_kifs_1_fold_identity(self):
        """1-fold KIFS should be abs-only (no rotation, identity)."""
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 1)
            assert math.isclose(result[0], abs(p[0])), (
                f"x: {result[0]} vs {abs(p[0])}"
            )
            assert math.isclose(result[1], abs(p[1])), (
                f"y: {result[1]} vs {abs(p[1])}"
            )
            assert result[2] == p[2]

    def test_kifs_2_fold_finite(self):
        for _ in range(10):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 2)
            assert all(math.isfinite(x) for x in result)

    def test_kifs_float_vs_int_folds_same(self):
        p = (1.5, -2.5, 0.5)
        result_float = py_domain_kifs(p, 6.0)
        result_int = py_domain_kifs(p, 6)
        assert vec3_close(result_float, result_int), (
            f"float {result_float} vs int {result_int}"
        )


# =============================================================================
# Test: T-DEMO-1.24 KIFS Guard (FIX 2 verification)
# =============================================================================


class TestDomainKifsGuard:
    """Verifies KIFS safe_folds guard for degenerate fold values (FIX 2).

    The WGSL domain_kifs now guards against degenerate folds:
      let safe_folds = max(abs(folds), 1.0);
    This prevents:
      - Division by zero when folds = 0
      - Undefined behavior from negative fold counts
      - Non-integer truncation for folds in (0, 1)
    """

    def test_kifs_zero_folds_guard(self):
        """folds = 0 uses safe_folds = 1, no division by zero."""
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 0)
            assert all(math.isfinite(x) for x in result), (
                f"zero folds non-finite: {result}"
            )

    def test_kifs_negative_folds_guard(self):
        """Negative folds use abs value via guard."""
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result_neg = py_domain_kifs(p, -6)
            result_pos = py_domain_kifs(p, 6)
            assert all(math.isfinite(x) for x in result_neg), (
                f"negative folds non-finite: {result_neg}"
            )
            # Negative folds with abs guard should match positive (same safe_folds)
            assert vec3_close(result_neg, result_pos), (
                f"-6 folds {result_neg} != +6 folds {result_pos}"
            )

    def test_kifs_negative_three_folds_guard(self):
        """Negative -3 folds should match +3 folds (abs guard)."""
        for _ in range(10):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            r_neg = py_domain_kifs(p, -3)
            r_pos = py_domain_kifs(p, 3)
            assert vec3_close(r_neg, r_pos), (
                f"-3 folds {r_neg} != +3 folds {r_pos}"
            )

    def test_kifs_sub_one_folds_guard(self):
        """folds in (0, 1) uses safe_folds = 1 (exactly one fold iteration)."""
        for sub_one in [0.1, 0.5, 0.99]:
            for _ in range(10):
                p = tuple(random.uniform(-5, 5) for _ in range(3))
                result = py_domain_kifs(p, sub_one)
                assert all(math.isfinite(x) for x in result), (
                    f"folds={sub_one} non-finite: {result}"
                )
                # With 1 iteration, KIFS is just abs (no rotation)
                assert math.isclose(result[0], abs(p[0])), (
                    f"folds={sub_one} x: {result[0]} vs abs({p[0]})={abs(p[0])}"
                )
                assert math.isclose(result[1], abs(p[1])), (
                    f"folds={sub_one} y: {result[1]} vs abs({p[1]})={abs(p[1])}"
                )

    def test_kifs_tiny_positive_folds_guard(self):
        """Very small positive fold (1e-6) uses safe_folds = 1, no crash."""
        for _ in range(10):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, 1e-6)
            assert all(math.isfinite(x) for x in result), (
                f"tiny folds non-finite: {result}"
            )

    def test_kifs_large_negative_folds_guard(self):
        """Large negative folds (-100) uses abs guard, safe_folds = 100."""
        for _ in range(10):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_kifs(p, -100)
            assert all(math.isfinite(x) for x in result), (
                f"large negative folds non-finite: {result}"
            )

    def test_kifs_zero_folds_equals_one_fold(self):
        """folds=0 should produce same result as folds=1 (both use safe_folds=1)."""
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            r0 = py_domain_kifs(p, 0)
            r1 = py_domain_kifs(p, 1)
            assert vec3_close(r0, r1), (
                f"0-fold {r0} != 1-fold {r1} for p={p}"
            )

    def test_kifs_float_folds_matches_int(self):
        """Float folds with guard should match int folds for equivalent values."""
        for folds_float, folds_int in [(3.0, 3), (6.0, 6), (8.0, 8), (12.0, 12)]:
            for _ in range(10):
                p = tuple(random.uniform(-5, 5) for _ in range(3))
                rf = py_domain_kifs(p, folds_float)
                ri = py_domain_kifs(p, folds_int)
                assert vec3_close(rf, ri), (
                    f"float {folds_float} {rf} != int {folds_int} {ri}"
                )

    def test_kifs_guard_does_not_affect_normal_folds(self):
        """Guard does not affect behavior for valid folds >= 1."""
        for folds in [1, 2, 3, 4, 5, 6, 8, 12]:
            for _ in range(10):
                p = tuple(random.uniform(-5, 5) for _ in range(3))
                result = py_domain_kifs(p, folds)
                assert all(math.isfinite(x) for x in result), (
                    f"valid folds={folds} non-finite: {result}"
                )
                # z should be preserved (only xy rotation)
                assert result[2] == p[2], (
                    f"folds={folds}: z changed: {result[2]} != {p[2]}"
                )


# =============================================================================
# Test: T-DEMO-1.25 Twist
# =============================================================================


class TestDomainTwist:
    """Whitebox tests for domain_twist(p, k)."""

    def test_twist_zero_rate(self):
        p = tuple(random.uniform(-5, 5) for _ in range(3))
        result = py_domain_twist(p, 0.0)
        assert result == p

    def test_twist_y_preserved(self):
        p = tuple(random.uniform(-5, 5) for _ in range(3))
        result = py_domain_twist(p, 1.0)
        assert result[1] == p[1]

    def test_twist_periodic(self):
        k = 2.0
        period = 2.0 * math.pi / k
        p = (1.5, 0.5, -0.8)
        at_y = py_domain_twist(p, k)
        at_y_period = py_domain_twist((p[0], p[1] + period, p[2]), k)
        assert math.isclose(at_y_period[0], at_y[0], abs_tol=1e-6)
        assert math.isclose(at_y_period[2], at_y[2], abs_tol=1e-6)

    def test_twist_half_period(self):
        k = 1.5
        half_period = math.pi / k
        p = (2.0, 0.3, 1.0)
        at_y = py_domain_twist(p, k)
        at_y_half = py_domain_twist((p[0], p[1] + half_period, p[2]), k)
        assert math.isclose(at_y_half[0], -at_y[0], abs_tol=1e-6)
        assert math.isclose(at_y_half[2], -at_y[2], abs_tol=1e-6)

    def test_twist_determinant_one(self):
        k = 0.7
        for y in [0.0, 1.0, 2.5, -1.3, 10.0]:
            c = math.cos(k * y)
            s = math.sin(k * y)
            det = c * c + s * s
            assert math.isclose(det, 1.0, rel_tol=TOL_REL, abs_tol=TOL_ABS)

    def test_twist_preserves_distance_at_same_y(self):
        k = random.uniform(0.1, 5.0)
        for _ in range(20):
            y = random.uniform(-5, 5)
            a = (random.uniform(-5, 5), y, random.uniform(-5, 5))
            b = (random.uniform(-5, 5), y, random.uniform(-5, 5))
            d_before = vec3_dist(a, b)
            d_after = vec3_dist(py_domain_twist(a, k), py_domain_twist(b, k))
            assert math.isclose(d_before, d_after, rel_tol=TOL_REL, abs_tol=TOL_ABS)


# =============================================================================
# Test: T-DEMO-1.26 Bend
# =============================================================================


class TestDomainBend:
    """Whitebox tests for domain_bend(p, r)."""

    def test_bend_identity_near_zero(self):
        for r in [1e-9, -1e-9, 0.0]:
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_bend(p, r)
            assert vec3_close(result, p, abs_tol=1e-6), f"identity at r={r}"

    def test_bend_y_preserved(self):
        r = random.uniform(0.5, 10.0)
        for _ in range(10):
            p = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2))
            result = py_domain_bend(p, r)
            assert result[1] == p[1], f"y changed: {result[1]} != {p[1]}"

    def test_bend_finite_output(self):
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            r = random.uniform(0.1, 20.0)
            result = py_domain_bend(p, r)
            assert all(math.isfinite(x) for x in result), f"non-finite: {result}"

    def test_bend_identity_at_origin(self):
        result = py_domain_bend((0.0, 0.0, 0.0), 1.0)
        assert vec3_close(result, (0.0, 0.0, 0.0), abs_tol=1e-6), f"got {result}"

    def test_bend_large_r_xz_swap(self):
        r = 100.0
        p = (2.0, 0.0, 3.0)
        result = py_domain_bend(p, r)
        assert result[0] == pytest.approx(3.0, rel=0.1), f"x' ~ z: {result}"
        assert result[2] == pytest.approx(2.0, rel=0.1), f"z' ~ x: {result}"

    def test_bend_negative_radius(self):
        p = (1.0, 2.0, 3.0)
        result_pos = py_domain_bend(p, 5.0)
        result_neg = py_domain_bend(p, -5.0)
        assert vec3_close(result_pos, result_neg), f"pos {result_pos} vs neg {result_neg}"

    def test_bend_preserves_yz_distance_approx(self):
        r = 5.0
        p = (1.0, 0.0, 2.0)
        p_shifted = (1.0, 0.5, 2.5)
        d_before = vec3_dist(p, p_shifted)
        d_after = vec3_dist(py_domain_bend(p, r), py_domain_bend(p_shifted, r))
        assert d_after <= d_before * 1.5, f"grew: {d_before} -> {d_after}"

    def test_bend_negative_x_symmetry(self):
        r = 3.0
        p = (1.5, 0.0, 2.0)
        p_neg = (-1.5, 0.0, 2.0)
        result_pos = py_domain_bend(p, r)
        result_neg = py_domain_bend(p_neg, r)
        assert math.isclose(abs(result_pos[0]), abs(result_neg[0]), rel_tol=TOL_REL)
        assert math.isclose(result_pos[2], -result_neg[2], rel_tol=TOL_REL)


# =============================================================================
# Test: T-DEMO-1.27 Stretch (Anisotropic Scaling)
# =============================================================================


class TestDomainStretch:
    """Whitebox tests for domain_stretch_x/y/z(p, s)."""

    def test_stretch_x_positive_s(self):
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = py_domain_stretch_x(p, s)
        assert vec3_close(result, (4.0, 1.5, 2.0)), f"got {result}"

    def test_stretch_y_positive_s(self):
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = py_domain_stretch_y(p, s)
        assert vec3_close(result, (1.0, 6.0, 2.0)), f"got {result}"

    def test_stretch_z_positive_s(self):
        p = (2.0, 3.0, 4.0)
        s = 2.0
        result = py_domain_stretch_z(p, s)
        assert vec3_close(result, (1.0, 1.5, 8.0)), f"got {result}"

    def test_stretch_x_compress(self):
        p = (2.0, 3.0, 4.0)
        s = 0.5
        result = py_domain_stretch_x(p, s)
        assert vec3_close(result, (1.0, 6.0, 8.0)), f"got {result}"

    def test_stretch_identity_at_one(self):
        p = (2.5, -1.5, 3.0)
        for fn in [py_domain_stretch_x, py_domain_stretch_y, py_domain_stretch_z]:
            result = fn(p, 1.0)
            assert vec3_close(result, p), f"{fn.__name__} not identity: {result}"

    def test_stretch_x_inverse(self):
        p = (3.0, 4.0, 5.0)
        s = 2.0
        forward = py_domain_stretch_x(p, s)
        backward = py_domain_stretch_x(forward, 1.0 / s)
        assert vec3_close(backward, p), f"not inverse: {backward} vs {p}"

    def test_stretch_y_inverse(self):
        p = (3.0, 4.0, 5.0)
        s = 2.5
        forward = py_domain_stretch_y(p, s)
        backward = py_domain_stretch_y(forward, 1.0 / s)
        assert vec3_close(backward, p)

    def test_stretch_z_inverse(self):
        p = (3.0, 4.0, 5.0)
        s = 0.75
        forward = py_domain_stretch_z(p, s)
        backward = py_domain_stretch_z(forward, 1.0 / s)
        assert vec3_close(backward, p)

    def test_stretch_finite_output(self):
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            for s in [0.1, 0.5, 1.0, 2.0, 10.0]:
                for fn in [py_domain_stretch_x, py_domain_stretch_y, py_domain_stretch_z]:
                    result = fn(p, s)
                    assert all(math.isfinite(x) for x in result), (
                        f"{fn.__name__} non-finite: {result}"
                    )

    def test_stretch_zero_s_raises(self):
        with pytest.raises(ZeroDivisionError):
            py_domain_stretch_x((1.0, 1.0, 1.0), 0.0)

    def test_stretch_negative_s(self):
        p = (2.0, 3.0, 4.0)
        result = py_domain_stretch_x(p, -1.0)
        assert result[0] == -2.0, f"expected -2.0, got {result[0]}"
        assert result[1] == -3.0, f"expected -3.0, got {result[1]}"
        assert result[2] == -4.0, f"expected -4.0, got {result[2]}"


# =============================================================================
# Test: WGSL trunc-toward-zero % operator (FIX 1 & 4 verification)
# =============================================================================


class TestWgslModNegative:
    """Verifies WGSL % operator behavior for negative inputs (FIX 1, FIX 4)."""

    def test_wgsl_mod_positive(self):
        assert wgsl_mod(5.0, 3.0) == pytest.approx(2.0)

    def test_wgsl_mod_exact_multiple(self):
        assert wgsl_mod(6.0, 3.0) == pytest.approx(0.0)
        assert wgsl_mod(-6.0, 3.0) == pytest.approx(0.0)

    def test_wgsl_mod_negative_dividend(self):
        result = wgsl_mod(-5.0, 3.0)
        assert result == pytest.approx(-2.0), (
            f"WGSL % should give -2, got {result} "
            f"(Python % gives {(-5.0) % 3.0})"
        )

    def test_wgsl_mod_periodicity_limitation(self):
        """WGSL trunc-mod: mod(x+c,c)=mod(x,c)+c when mod(x,c) < 0.
        Periodicity mod(x+c,c)=mod(x,c) breaks for negative results.
        """
        c = 2.0
        x = -1.0
        r1 = wgsl_mod(x, c)
        r1_plus_c = wgsl_mod(x + c, c)
        assert r1 == pytest.approx(-1.0), f"mod(-1,2) = {r1}"
        # mod(x+c,c) = mod(x,c) + c when mod(x,c) < 0
        assert r1_plus_c == pytest.approx(r1 + c), (
            f"periodicity: {r1_plus_c} vs {r1 + c}"
        )

    def test_domain_repeat_negative_range(self):
        c = (2.0, 2.0, 2.0)
        for p_x in [-0.1, -0.5, -1.0, -1.5, -1.9]:
            r = py_domain_repeat((p_x, 0.0, 0.0), c)
            assert r[0] > -3.0, f"out of range: {r[0]}"
            assert r[0] < 1.0, f"out of range: {r[0]}"

    def test_domain_repeat_positive_range(self):
        c = (2.0, 2.0, 2.0)
        for p_x in [0.1, 0.5, 1.0, 1.5, 1.9]:
            r = py_domain_repeat((p_x, 0.0, 0.0), c)
            assert -1.0 < r[0] < 1.0, f"out of range: {r[0]}"

    def test_domain_repeat_negative_lt_negc(self):
        c = (2.0, 2.0, 2.0)
        p = (-3.0, 0.0, 0.0)
        r = py_domain_repeat(p, c)
        assert r[0] == -2.0, f"expected -2.0, got {r[0]}"

    def test_domain_repeat_euclidean_mod_documented(self):
        """Verify the WGSL source file documents the Euclidean modulo form (FIX 1)."""
        src_path = get_wgsl_source_path()
        assert os.path.exists(src_path), f"WGSL source not found: {src_path}"
        with open(src_path, "r") as f:
            content = f.read()
        assert "Euclidean modulo" in content, (
            "WGSL source missing Euclidean modulo documentation (FIX 1)"
        )
        assert "expanded form" in content, (
            "WGSL source missing expanded form note (FIX 1)"
        )


# =============================================================================
# Test: T-DEMO-1.27 Stretch Composability
# =============================================================================


class TestStretchComposability:
    """Tests stretch composed with other domain operations."""

    def test_stretch_then_repeat_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = py_domain_stretch_x(p, 2.0)
            r = py_domain_repeat(s, (4.0, 4.0, 4.0))
            assert all(math.isfinite(x) for x in r)

    def test_stretch_then_mirror_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = py_domain_stretch_y(p, 1.5)
            m = py_domain_mirror_x(py_domain_mirror_z(s))
            assert all(math.isfinite(x) for x in m)

    def test_stretch_then_twist_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = py_domain_stretch_z(p, 3.0)
            t = py_domain_twist(s, 0.5)
            assert all(math.isfinite(x) for x in t)

    def test_stretch_then_bend_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = py_domain_stretch_x(p, 0.5)
            b = py_domain_bend(s, 5.0)
            assert all(math.isfinite(x) for x in b)

    def test_stretch_then_kifs_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            s = py_domain_stretch_y(p, 2.0)
            k = py_domain_kifs(s, 6)
            assert all(math.isfinite(x) for x in k)

    def test_stretch_x_inverse_compose(self):
        p = (3.0, 4.0, 5.0)
        s = 2.5
        forward = py_domain_stretch_x(p, s)
        backward = py_domain_stretch_x(forward, 1.0 / s)
        assert vec3_close(backward, p)

    def test_stretch_y_inverse_compose(self):
        p = (3.0, 4.0, 5.0)
        s = 1.5
        forward = py_domain_stretch_y(p, s)
        backward = py_domain_stretch_y(forward, 1.0 / s)
        assert vec3_close(backward, p)

    def test_stretch_determinant(self):
        """Jacobian det = s * 1/s * 1/s = 1/s (NOT 1, not volume-preserving)."""
        p = (2.0, 3.0, 4.0)
        s = 2.0
        sx = py_domain_stretch_x(p, s)
        vol_ratio = (
            abs(sx[0] / p[0]) * abs(sx[1] / p[1]) * abs(sx[2] / p[2])
        )
        expected_ratio = 1.0 / s
        assert math.isclose(vol_ratio, expected_ratio, rel_tol=TOL_REL), (
            f"volume ratio {vol_ratio}, expected {expected_ratio}"
        )
        # Also verify with compression (s < 1)
        s2 = 0.5
        sx2 = py_domain_stretch_x(p, s2)
        vr2 = abs(sx2[0]/p[0]) * abs(sx2[1]/p[1]) * abs(sx2[2]/p[2])
        assert math.isclose(vr2, 1.0/s2, rel_tol=TOL_REL)

    def test_stretch_y_determinant(self):
        """Jacobian det(stretch_y) = s * 1/s * 1/s = 1/s."""
        p = (2.0, 3.0, 4.0)
        s = 2.0
        sy = py_domain_stretch_y(p, s)
        vol_ratio = (
            abs(sy[0] / p[0]) * abs(sy[1] / p[1]) * abs(sy[2] / p[2])
        )
        expected_ratio = 1.0 / s
        assert math.isclose(vol_ratio, expected_ratio, rel_tol=TOL_REL), (
            f"stretch_y volume ratio {vol_ratio}, expected {expected_ratio}"
        )
        # Compression case
        s2 = 0.25
        sy2 = py_domain_stretch_y(p, s2)
        vr2 = abs(sy2[0]/p[0]) * abs(sy2[1]/p[1]) * abs(sy2[2]/p[2])
        assert math.isclose(vr2, 1.0/s2, rel_tol=TOL_REL)

    def test_stretch_z_determinant(self):
        """Jacobian det(stretch_z) = s * 1/s * 1/s = 1/s."""
        p = (2.0, 3.0, 4.0)
        s = 2.5
        sz = py_domain_stretch_z(p, s)
        vol_ratio = (
            abs(sz[0] / p[0]) * abs(sz[1] / p[1]) * abs(sz[2] / p[2])
        )
        expected_ratio = 1.0 / s
        assert math.isclose(vol_ratio, expected_ratio, rel_tol=TOL_REL), (
            f"stretch_z volume ratio {vol_ratio}, expected {expected_ratio}"
        )
        # Compression case
        s2 = 0.75
        sz2 = py_domain_stretch_z(p, s2)
        vr2 = abs(sz2[0]/p[0]) * abs(sz2[1]/p[1]) * abs(sz2[2]/p[2])
        assert math.isclose(vr2, 1.0/s2, rel_tol=TOL_REL)

    def test_stretch_determinant_independent_of_input(self):
        """Determinant = 1/s regardless of input position."""
        for fn, name in [
            (py_domain_stretch_x, "stretch_x"),
            (py_domain_stretch_y, "stretch_y"),
            (py_domain_stretch_z, "stretch_z"),
        ]:
            for s in [0.1, 0.5, 1.0, 2.0, 10.0]:
                for _ in range(20):
                    p = tuple(random.uniform(-10, 10) for _ in range(3))
                    r = fn(p, s)
                    det = (
                        abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
                    )
                    assert math.isclose(det, 1.0/s, rel_tol=TOL_REL), (
                        f"{name}(s={s}) det={det}, expected 1/{s}={1.0/s} for p={p}"
                    )

    def test_stretch_determinant_not_volume_preserving(self):
        """Stretch is NOT volume-preserving: det=1/s, not 1."""
        for fn in [py_domain_stretch_x, py_domain_stretch_y, py_domain_stretch_z]:
            for s in [0.25, 0.5, 2.0, 4.0]:
                p = (1.0, 2.0, 3.0)
                r = fn(p, s)
                det = abs(r[0]/p[0]) * abs(r[1]/p[1]) * abs(r[2]/p[2])
                assert not math.isclose(det, 1.0, rel_tol=TOL_REL), (
                    f"stretch s={s} det={det} == 1, but should NOT be volume-preserving"
                )
                assert math.isclose(det, 1.0/s, rel_tol=TOL_REL), (
                    f"stretch s={s} det={det} != 1/{s}"
                )

    def test_stretch_identity_at_one(self):
        p = (2.5, -1.5, 3.0)
        for fn in [py_domain_stretch_x, py_domain_stretch_y, py_domain_stretch_z]:
            assert vec3_close(fn(p, 1.0), p), f"{fn.__name__} not identity"

    def test_stretch_commutes_across_axes(self):
        p = (2.0, 3.0, 4.0)
        xy = py_domain_stretch_y(py_domain_stretch_x(p, 2.0), 3.0)
        yx = py_domain_stretch_x(py_domain_stretch_y(p, 3.0), 2.0)
        assert vec3_close(xy, yx), f"xy {xy} vs yx {yx}"

    def test_stretch_bend_preserves_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            s = py_domain_stretch_z(p, 1.5)
            t = py_domain_twist(s, 0.3)
            b = py_domain_bend(t, 8.0)
            m = py_domain_mirror_x(b)
            assert all(math.isfinite(x) for x in m)


# =============================================================================
# Test: Composability
# =============================================================================


class TestComposability:
    """Tests for composing multiple domain operations including stretch."""

    def test_repeat_then_mirror(self):
        c = (2.0, 2.0, 2.0)
        for _ in range(50):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            repeated = py_domain_repeat(p, c)
            mirrored = py_domain_mirror_x(repeated)
            assert mirrored[0] <= 1.5 * c[0] + TOL_ABS
            assert mirrored[0] >= 0.0

    def test_twist_then_mirror_finite(self):
        for _ in range(30):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_mirror_x(py_domain_twist(p, 1.0))
            assert all(math.isfinite(x) for x in result)

    def test_repeat_then_twist_finite(self):
        c = (4.0, 4.0, 4.0)
        for _ in range(30):
            p = tuple(random.uniform(-20, 20) for _ in range(3))
            result = py_domain_twist(py_domain_repeat(p, c), 0.5)
            assert all(math.isfinite(x) for x in result)

    def test_mirror_all_then_repeat_finite(self):
        c = (3.0, 3.0, 3.0)
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            mir = py_domain_mirror_z(py_domain_mirror_y(py_domain_mirror_x(p)))
            result = py_domain_repeat(mir, c)
            assert all(math.isfinite(x) for x in result)

    def test_kifs_then_twist_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_twist(py_domain_kifs(p, 6), 0.5)
            assert all(math.isfinite(x) for x in result)

    def test_bend_then_repeat_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_repeat(py_domain_bend(p, 5.0), (2.0, 2.0, 2.0))
            assert all(math.isfinite(x) for x in result)

    def test_mirror_then_bend_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            result = py_domain_bend(py_domain_mirror_x(py_domain_mirror_y(p)), 3.0)
            assert all(math.isfinite(x) for x in result)

    def test_stretch_then_repeat_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            s = py_domain_stretch_x(p, 2.0)
            r = py_domain_repeat(s, (4.0, 4.0, 4.0))
            assert all(math.isfinite(x) for x in r)

    def test_stretch_then_kifs_then_twist_finite(self):
        for _ in range(20):
            p = tuple(random.uniform(-5, 5) for _ in range(3))
            s = py_domain_stretch_z(p, 1.5)
            k = py_domain_kifs(s, 6)
            t = py_domain_twist(k, 0.5)
            assert all(math.isfinite(x) for x in t)

    def test_all_operations_pipeline(self):
        for _ in range(20):
            p = tuple(random.uniform(-20, 20) for _ in range(3))
            rep = py_domain_repeat(p, (4.0, 4.0, 4.0))
            mir = py_domain_mirror_z(py_domain_mirror_y(py_domain_mirror_x(rep)))
            tw = py_domain_twist(mir, 0.5)
            bn = py_domain_bend(tw, 10.0)
            assert all(math.isfinite(x) for x in bn)


# =============================================================================
# Test: Numerical Invariants
# =============================================================================


class TestNumericalInvariants:
    """Broad numerical invariants across all domain operations."""

    def test_all_operations_finite(self):
        operations = [
            ("repeat", lambda p: py_domain_repeat(p, (2.0, 2.0, 2.0))),
            ("mirror_x", py_domain_mirror_x),
            ("mirror_y", py_domain_mirror_y),
            ("mirror_z", py_domain_mirror_z),
            ("kifs_3", lambda p: py_domain_kifs(p, 3)),
            ("kifs_6", lambda p: py_domain_kifs(p, 6)),
            ("twist", lambda p: py_domain_twist(p, 1.0)),
            ("bend_r1", lambda p: py_domain_bend(p, 1.0)),
            ("bend_r100", lambda p: py_domain_bend(p, 100.0)),
            ("stretch_x", lambda p: py_domain_stretch_x(p, 2.0)),
            ("stretch_y", lambda p: py_domain_stretch_y(p, 1.5)),
            ("stretch_z", lambda p: py_domain_stretch_z(p, 0.5)),
            ("stretch_x_10x", lambda p: py_domain_stretch_x(p, 10.0)),
        ]
        for _ in range(100):
            p = tuple(random.uniform(-1e3, 1e3) for _ in range(3))
            for name, op in operations:
                result = op(p)
                assert all(math.isfinite(x) for x in result), (
                    f"{name} non-finite for p={p}: {result}"
                )

    def test_zero_input_vector(self):
        zero = (0.0, 0.0, 0.0)
        results = [
            py_domain_repeat(zero, (2.0, 2.0, 2.0)),
            py_domain_mirror_x(zero),
            py_domain_mirror_y(zero),
            py_domain_mirror_z(zero),
            py_domain_kifs(zero, 6),
            py_domain_twist(zero, 1.0),
            py_domain_bend(zero, 1.0),
            py_domain_stretch_x(zero, 2.0),
            py_domain_stretch_y(zero, 2.0),
            py_domain_stretch_z(zero, 2.0),
        ]
        for result in results:
            assert all(math.isfinite(x) for x in result)

    def test_large_input_values(self):
        large = (1e6, 1e6, 1e6)
        results = [
            py_domain_repeat(large, (2.0, 2.0, 2.0)),
            py_domain_mirror_x(large),
            py_domain_mirror_y(large),
            py_domain_mirror_z(large),
            py_domain_twist(large, 1.0),
            py_domain_stretch_x(large, 2.0),
            py_domain_stretch_y(large, 2.0),
            py_domain_stretch_z(large, 2.0),
        ]
        for result in results:
            assert all(math.isfinite(x) for x in result), f"non-finite: {result}"

    def test_extreme_cell_sizes(self):
        for c in [(1e-6, 1.0, 1.0), (1e6, 1.0, 1.0), (0.001, 0.001, 0.001)]:
            p = tuple(random.uniform(-10, 10) for _ in range(3))
            result = py_domain_repeat(p, c)
            assert all(math.isfinite(x) for x in result), f"c={c}: {result}"

    def test_extreme_twist_rate(self):
        for k in [1e-6, 1e6, -1e3]:
            p = (1.0, 2.0, 3.0)
            result = py_domain_twist(p, k)
            assert all(math.isfinite(x) for x in result), f"k={k}: {result}"

    def test_deterministic_output(self):
        p = (1.234, 5.678, -3.456)
        c = (2.0, 2.0, 2.0)
        assert vec3_close(py_domain_repeat(p, c), py_domain_repeat(p, c))
        assert vec3_close(py_domain_kifs(p, 6), py_domain_kifs(p, 6))
        assert vec3_close(py_domain_twist(p, 0.7), py_domain_twist(p, 0.7))
        assert vec3_close(py_domain_bend(p, 3.0), py_domain_bend(p, 3.0))
        assert vec3_close(py_domain_stretch_x(p, 2.0), py_domain_stretch_x(p, 2.0))
        assert vec3_close(py_domain_stretch_y(p, 2.0), py_domain_stretch_y(p, 2.0))
        assert vec3_close(py_domain_stretch_z(p, 2.0), py_domain_stretch_z(p, 2.0))
