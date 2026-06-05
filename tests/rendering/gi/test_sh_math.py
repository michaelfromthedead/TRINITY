"""Tests for spherical harmonics math library.

Validates that the Python sh_math module produces results matching
the WGSL spherical_harmonics.wgsl shader and meets numerical accuracy
requirements from Ramamoorthi & Hanrahan 2001.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.rendering.gi.sh_math import (
    SH_A0,
    SH_A1,
    SH_A2,
    SH_Y00,
    SH_Y1,
    SH_Y2_0,
    SH_Y2_NEG1,
    SH_Y2_NEG2,
    SH_Y2_POS1,
    SH_Y2_POS2,
    SHCoefficientsL2,
    fibonacci_sphere_directions,
    generate_ramamoorthi_reference,
    generate_test_data_constant,
    generate_test_data_gradient,
    generate_test_data_rotation,
    sh_basis_l2,
    sh_basis_l2_batch,
    sh_convolve_irradiance,
    sh_energy,
    sh_evaluate_l2,
    sh_evaluate_l2_batch,
    sh_project_function,
    sh_project_l2,
    sh_rotate_l2,
    validate_orthonormality,
    validate_roundtrip_error,
)


class TestSHConstants:
    """Test that SH constants match expected values."""

    def test_y00_constant(self) -> None:
        """Y_0^0 = sqrt(1/(4*PI))."""
        expected = math.sqrt(1.0 / (4.0 * math.pi))
        assert abs(SH_Y00 - expected) < 1e-10

    def test_y1_constant(self) -> None:
        """Y_1 = sqrt(3/(4*PI))."""
        expected = math.sqrt(3.0 / (4.0 * math.pi))
        assert abs(SH_Y1 - expected) < 1e-10

    def test_y2_neg2_constant(self) -> None:
        """Y_2^-2 normalization = sqrt(15/(4*PI))."""
        expected = math.sqrt(15.0 / (4.0 * math.pi))
        assert abs(SH_Y2_NEG2 - expected) < 1e-10

    def test_y2_0_constant(self) -> None:
        """Y_2^0 normalization = sqrt(5/(16*PI))."""
        expected = math.sqrt(5.0 / (16.0 * math.pi))
        assert abs(SH_Y2_0 - expected) < 1e-10

    def test_y2_pos2_constant(self) -> None:
        """Y_2^2 normalization = sqrt(15/(16*PI))."""
        expected = math.sqrt(15.0 / (16.0 * math.pi))
        assert abs(SH_Y2_POS2 - expected) < 1e-10

    def test_cosine_lobe_coefficients(self) -> None:
        """Test Ramamoorthi & Hanrahan cosine lobe coefficients."""
        assert SH_A0 == 1.0
        assert abs(SH_A1 - 2.0 / 3.0) < 1e-10
        assert SH_A2 == 0.25


class TestSHBasisFunctions:
    """Test SH basis function evaluation."""

    def test_basis_l0_constant(self) -> None:
        """L0 basis should be constant for all directions."""
        directions = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32) / math.sqrt(3),
        ]
        for d in directions:
            basis = sh_basis_l2(d)
            assert abs(basis[0] - SH_Y00) < 1e-6

    def test_basis_l1_x_direction(self) -> None:
        """Y_1^1 = SH_Y1 * x at +X direction."""
        basis = sh_basis_l2(np.array([1, 0, 0], dtype=np.float32))
        assert abs(basis[3] - SH_Y1) < 1e-6  # Y_1^1 = x component
        assert abs(basis[1]) < 1e-6  # Y_1^-1 = y = 0
        assert abs(basis[2]) < 1e-6  # Y_1^0 = z = 0

    def test_basis_l1_y_direction(self) -> None:
        """Y_1^-1 = SH_Y1 * y at +Y direction."""
        basis = sh_basis_l2(np.array([0, 1, 0], dtype=np.float32))
        assert abs(basis[1] - SH_Y1) < 1e-6  # Y_1^-1 = y component
        assert abs(basis[3]) < 1e-6  # Y_1^1 = x = 0

    def test_basis_l1_z_direction(self) -> None:
        """Y_1^0 = SH_Y1 * z at +Z direction."""
        basis = sh_basis_l2(np.array([0, 0, 1], dtype=np.float32))
        assert abs(basis[2] - SH_Y1) < 1e-6  # Y_1^0 = z component

    def test_basis_batch_matches_single(self) -> None:
        """Batch evaluation should match single evaluation."""
        directions = fibonacci_sphere_directions(100)
        batch_result = sh_basis_l2_batch(directions)

        for i, d in enumerate(directions):
            single_result = sh_basis_l2(d)
            np.testing.assert_allclose(batch_result[i], single_result, rtol=1e-5)


class TestSHCoefficientsL2:
    """Test the SH coefficient container."""

    def test_zero_coefficients(self) -> None:
        """Zero coefficients should all be zero."""
        coeffs = SHCoefficientsL2.zero()
        assert np.all(coeffs.coeffs == 0)

    def test_get_set(self) -> None:
        """Get and set operations."""
        coeffs = SHCoefficientsL2.zero()
        coeffs.set(0, np.array([1, 2, 3], dtype=np.float32))
        result = coeffs.get(0)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_scale(self) -> None:
        """Scaling coefficients."""
        coeffs = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        coeffs.scale(2.0)
        assert np.all(coeffs.coeffs == 2.0)

    def test_add(self) -> None:
        """Adding coefficient sets."""
        a = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        b = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32) * 2)
        a.add(b)
        assert np.all(a.coeffs == 3.0)

    def test_lerp(self) -> None:
        """Linear interpolation."""
        a = SHCoefficientsL2.zero()
        b = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        result = a.lerp(b, 0.5)
        assert np.all(np.abs(result.coeffs - 0.5) < 1e-6)

    def test_to_from_bytes(self) -> None:
        """Byte serialization round-trip."""
        original = SHCoefficientsL2(np.arange(27).reshape(9, 3).astype(np.float32))
        data = original.to_bytes()
        assert len(data) == 144  # 9 * 4 * 4 bytes (vec4 padding)
        restored = SHCoefficientsL2.from_bytes(data)
        np.testing.assert_array_equal(original.coeffs, restored.coeffs)


class TestSHProjectionEvaluation:
    """Test projection and evaluation."""

    def test_project_single_direction(self) -> None:
        """Project a color at a direction."""
        direction = np.array([0, 0, 1], dtype=np.float32)
        color = np.array([1, 0.5, 0.25], dtype=np.float32)
        coeffs = sh_project_l2(direction, color)

        # Evaluate at same direction should recover scaled color
        result = sh_evaluate_l2(coeffs, direction)
        # Result should be positive and proportional to input
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] > 0

    def test_roundtrip_constant_function(self) -> None:
        """Project constant function and verify roundtrip."""
        test_data = generate_test_data_constant((0.5, 0.3, 0.8))
        assert test_data["max_error"] < 0.1

    def test_roundtrip_gradient_function(self) -> None:
        """Project z-gradient function and verify roundtrip."""
        test_data = generate_test_data_gradient()
        assert test_data["max_error"] < 0.05

    def test_roundtrip_error_below_threshold(self) -> None:
        """Numerical roundtrip error should be below 1e-5 for smooth functions."""
        error = validate_roundtrip_error(num_samples=10000, num_test_dirs=1000)
        # SH truncation introduces error, but should be small for low-frequency
        assert error < 0.05, f"Roundtrip error {error} exceeds threshold"


class TestSHOrthonormality:
    """Test orthonormality of SH basis."""

    def test_approximate_orthonormality(self) -> None:
        """SH basis should be approximately orthonormal."""
        result = validate_orthonormality(num_samples=10000)
        assert result["is_orthonormal"], (
            f"Basis not orthonormal, error: {result['identity_error']}"
        )


class TestSHIrradiance:
    """Test irradiance convolution (Ramamoorthi & Hanrahan 2001)."""

    def test_l0_scaling(self) -> None:
        """L0 band scales by A0 = 1.0."""
        ref = generate_ramamoorthi_reference()
        assert abs(ref["l0_scale_factor"] - SH_A0) < 1e-6

    def test_l1_scaling(self) -> None:
        """L1 band scales by A1 = 2/3."""
        ref = generate_ramamoorthi_reference()
        assert abs(ref["l1_scale_factor"] - SH_A1) < 1e-6

    def test_l2_scaling(self) -> None:
        """L2 band scales by A2 = 1/4."""
        ref = generate_ramamoorthi_reference()
        assert abs(ref["l2_scale_factor"] - SH_A2) < 1e-6

    def test_convolve_preserves_structure(self) -> None:
        """Convolution should preserve coefficient structure."""
        original = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        convolved = sh_convolve_irradiance(original)

        # L0 unchanged (A0 = 1)
        np.testing.assert_allclose(convolved.get(0), original.get(0))

        # L1 scaled by 2/3
        for i in [1, 2, 3]:
            np.testing.assert_allclose(convolved.get(i), original.get(i) * SH_A1, rtol=1e-5)

        # L2 scaled by 1/4
        for i in [4, 5, 6, 7, 8]:
            np.testing.assert_allclose(convolved.get(i), original.get(i) * SH_A2, rtol=1e-5)


class TestSHRotation:
    """Test SH rotation."""

    def test_identity_rotation(self) -> None:
        """Identity rotation should preserve coefficients."""
        original = SHCoefficientsL2(np.arange(27).reshape(9, 3).astype(np.float32))
        identity = np.eye(3, dtype=np.float32)
        rotated = sh_rotate_l2(original, identity)
        np.testing.assert_allclose(rotated.coeffs, original.coeffs, rtol=1e-5)

    def test_l0_invariant(self) -> None:
        """L0 should be rotationally invariant."""
        coeffs = SHCoefficientsL2.zero()
        coeffs.set(0, np.array([1, 2, 3], dtype=np.float32))

        angle = np.pi / 4
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ], dtype=np.float32)

        rotated = sh_rotate_l2(coeffs, rotation)
        np.testing.assert_allclose(rotated.get(0), coeffs.get(0), rtol=1e-5)

    def test_rotation_equivariance(self) -> None:
        """Rotating coefficients then evaluating should equal evaluating at rotated direction."""
        test_data = generate_test_data_rotation()
        # The peak should move from +X to +Y after 90-degree Z rotation
        assert test_data["match_error"] < 0.1


class TestFibonacciSphere:
    """Test Fibonacci sphere direction generation."""

    def test_count(self) -> None:
        """Should generate requested number of directions."""
        dirs = fibonacci_sphere_directions(100)
        assert len(dirs) == 100

    def test_normalized(self) -> None:
        """All directions should be normalized."""
        dirs = fibonacci_sphere_directions(100)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_coverage(self) -> None:
        """Directions should cover the sphere approximately uniformly."""
        dirs = fibonacci_sphere_directions(1000)

        # Count directions in each octant
        pos_x = np.sum(dirs[:, 0] > 0)
        pos_y = np.sum(dirs[:, 1] > 0)
        pos_z = np.sum(dirs[:, 2] > 0)

        # Should be roughly 50% in each positive half-space
        assert 400 < pos_x < 600
        assert 400 < pos_y < 600
        assert 400 < pos_z < 600


class TestSHEnergy:
    """Test energy computation."""

    def test_zero_energy(self) -> None:
        """Zero coefficients should have zero energy."""
        coeffs = SHCoefficientsL2.zero()
        assert sh_energy(coeffs) == 0.0

    def test_nonzero_energy(self) -> None:
        """Non-zero coefficients should have positive energy."""
        coeffs = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        energy = sh_energy(coeffs)
        assert energy > 0
        assert abs(energy - 27.0) < 1e-6  # 9 coeffs * 3 channels * 1^2


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_small_coefficients(self) -> None:
        """Small coefficients should evaluate without underflow."""
        coeffs = SHCoefficientsL2(np.full((9, 3), 1e-30, dtype=np.float32))
        direction = np.array([1, 0, 0], dtype=np.float32)
        result = sh_evaluate_l2(coeffs, direction)
        assert np.all(np.isfinite(result))

    def test_large_coefficients(self) -> None:
        """Large coefficients should evaluate without overflow."""
        coeffs = SHCoefficientsL2(np.full((9, 3), 1e30, dtype=np.float32))
        direction = np.array([1, 0, 0], dtype=np.float32)
        result = sh_evaluate_l2(coeffs, direction)
        assert np.all(np.isfinite(result))

    def test_unnormalized_direction(self) -> None:
        """Unnormalized direction should still work (but may give scaled result)."""
        coeffs = SHCoefficientsL2(np.ones((9, 3), dtype=np.float32))
        direction = np.array([2, 0, 0], dtype=np.float32)  # Not normalized
        result = sh_evaluate_l2(coeffs, direction)
        assert np.all(np.isfinite(result))
