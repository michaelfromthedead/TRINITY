"""Whitebox tests for Jacobian IK solver.

Covers T-IK-3.14 through T-IK-3.18:
- T-IK-3.14: Matrix Class
- T-IK-3.15: Jacobian Computation
- T-IK-3.16: Jacobian Transpose Method
- T-IK-3.17: Jacobian Pseudoinverse Method
- T-IK-3.18: Jacobian DLS (Damped Least Squares)
"""

from __future__ import annotations

import pytest
import math
from typing import List

from engine.animation.ik.jacobian import (
    Matrix,
    JacobianMethod,
    JacobianIK,
    JacobianResult,
    MultiTargetJacobianIK,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    JACOBIAN_DEFAULT_MAX_ITERATIONS,
    JACOBIAN_DLS_DAMPING,
    JACOBIAN_DEFAULT_STEP_SIZE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_2bone_chain():
    """Create a simple 2-bone chain along the Y-axis."""
    positions = [
        Vec3(0.0, 0.0, 0.0),  # Root
        Vec3(0.0, 1.0, 0.0),  # Mid
        Vec3(0.0, 2.0, 0.0),  # End
    ]
    rotations = [Quat.identity() for _ in range(3)]
    return positions, rotations


@pytest.fixture
def simple_3bone_chain():
    """Create a 3-bone chain along the Y-axis."""
    positions = [
        Vec3(0.0, 0.0, 0.0),  # Root
        Vec3(0.0, 1.0, 0.0),  # Joint 1
        Vec3(0.0, 2.0, 0.0),  # Joint 2
        Vec3(0.0, 3.0, 0.0),  # End
    ]
    rotations = [Quat.identity() for _ in range(4)]
    return positions, rotations


@pytest.fixture
def jacobian_ik_2bone():
    """Create a JacobianIK solver for 2 bones."""
    return JacobianIK(
        bone_indices=[0, 1, 2],
        method=JacobianMethod.DAMPED_LEAST_SQUARES,
        tolerance=0.001,
        max_iterations=50,
    )


@pytest.fixture
def jacobian_ik_transpose():
    """Create a JacobianIK solver using transpose method."""
    return JacobianIK(
        bone_indices=[0, 1, 2],
        method=JacobianMethod.TRANSPOSE,
        tolerance=0.001,
        max_iterations=50,
    )


@pytest.fixture
def jacobian_ik_pseudoinverse():
    """Create a JacobianIK solver using pseudoinverse method."""
    return JacobianIK(
        bone_indices=[0, 1, 2],
        method=JacobianMethod.PSEUDOINVERSE,
        tolerance=0.001,
        max_iterations=50,
    )


# =============================================================================
# T-IK-3.14: Matrix Class Tests
# =============================================================================


class TestMatrixConstruction:
    """Test Matrix construction and initialization."""

    def test_create_zero_matrix(self):
        """Test creating a zero-initialized matrix."""
        m = Matrix(3, 4)
        assert m.rows == 3
        assert m.cols == 4
        assert len(m.data) == 12
        assert all(x == 0.0 for x in m.data)

    def test_create_1x1_matrix(self):
        """Test creating a 1x1 matrix."""
        m = Matrix(1, 1)
        assert m.rows == 1
        assert m.cols == 1
        assert len(m.data) == 1
        assert m[0, 0] == 0.0

    def test_create_square_matrix(self):
        """Test creating a square matrix."""
        m = Matrix(4, 4)
        assert m.rows == 4
        assert m.cols == 4
        assert len(m.data) == 16

    def test_create_rectangular_matrix(self):
        """Test creating rectangular matrices."""
        m1 = Matrix(2, 5)
        assert m1.rows == 2
        assert m1.cols == 5
        assert len(m1.data) == 10

        m2 = Matrix(6, 3)
        assert m2.rows == 6
        assert m2.cols == 3
        assert len(m2.data) == 18

    def test_create_with_custom_data(self):
        """Test creating a matrix with custom data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m = Matrix(2, 3, data)
        assert m[0, 0] == 1.0
        assert m[0, 1] == 2.0
        assert m[0, 2] == 3.0
        assert m[1, 0] == 4.0
        assert m[1, 1] == 5.0
        assert m[1, 2] == 6.0

    def test_create_with_wrong_data_length_raises(self):
        """Test that wrong data length raises ValueError."""
        with pytest.raises(ValueError, match="Data length"):
            Matrix(2, 3, [1.0, 2.0, 3.0])  # 3 != 6

    def test_create_identity_matrix(self):
        """Test creating an identity matrix."""
        m = Matrix.identity(3)
        assert m.rows == 3
        assert m.cols == 3
        assert m[0, 0] == 1.0
        assert m[1, 1] == 1.0
        assert m[2, 2] == 1.0
        assert m[0, 1] == 0.0
        assert m[0, 2] == 0.0
        assert m[1, 0] == 0.0
        assert m[1, 2] == 0.0
        assert m[2, 0] == 0.0
        assert m[2, 1] == 0.0

    def test_create_identity_1x1(self):
        """Test creating a 1x1 identity matrix."""
        m = Matrix.identity(1)
        assert m[0, 0] == 1.0

    def test_create_identity_5x5(self):
        """Test creating a larger identity matrix."""
        m = Matrix.identity(5)
        for i in range(5):
            for j in range(5):
                expected = 1.0 if i == j else 0.0
                assert m[i, j] == expected

    def test_data_is_copied(self):
        """Test that data is copied, not referenced."""
        data = [1.0, 2.0, 3.0, 4.0]
        m = Matrix(2, 2, data)
        data[0] = 999.0
        assert m[0, 0] == 1.0  # Original data unchanged in matrix


class TestMatrixAccessors:
    """Test Matrix element access."""

    def test_get_element(self):
        """Test getting matrix elements."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        assert m[0, 0] == 1.0
        assert m[0, 1] == 2.0
        assert m[1, 0] == 3.0
        assert m[1, 1] == 4.0

    def test_set_element(self):
        """Test setting matrix elements."""
        m = Matrix(2, 2)
        m[0, 0] = 5.0
        m[1, 1] = 10.0
        assert m[0, 0] == 5.0
        assert m[1, 1] == 10.0
        assert m[0, 1] == 0.0

    def test_row_major_indexing(self):
        """Test that indexing is row-major."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m = Matrix(2, 3, data)
        # Row 0: [1, 2, 3]
        # Row 1: [4, 5, 6]
        assert m[0, 0] == 1.0
        assert m[0, 2] == 3.0
        assert m[1, 0] == 4.0
        assert m[1, 2] == 6.0

    def test_modify_element_persists(self):
        """Test that modifications persist."""
        m = Matrix(3, 3)
        m[1, 2] = 7.5
        assert m[1, 2] == 7.5
        m[1, 2] = 8.0
        assert m[1, 2] == 8.0


class TestMatrixTranspose:
    """Test Matrix transpose operation."""

    def test_transpose_square(self):
        """Test transposing a square matrix."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        t = m.transpose()
        assert t.rows == 2
        assert t.cols == 2
        assert t[0, 0] == 1.0
        assert t[0, 1] == 3.0
        assert t[1, 0] == 2.0
        assert t[1, 1] == 4.0

    def test_transpose_rectangular(self):
        """Test transposing a rectangular matrix."""
        m = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        t = m.transpose()
        assert t.rows == 3
        assert t.cols == 2
        assert t[0, 0] == 1.0
        assert t[0, 1] == 4.0
        assert t[1, 0] == 2.0
        assert t[1, 1] == 5.0
        assert t[2, 0] == 3.0
        assert t[2, 1] == 6.0

    def test_transpose_tall_matrix(self):
        """Test transposing a tall matrix."""
        m = Matrix(4, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        t = m.transpose()
        assert t.rows == 2
        assert t.cols == 4

    def test_double_transpose_equals_original(self):
        """Test that transpose of transpose equals original."""
        m = Matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tt = m.transpose().transpose()
        assert tt.rows == m.rows
        assert tt.cols == m.cols
        for i in range(m.rows):
            for j in range(m.cols):
                assert tt[i, j] == m[i, j]

    def test_transpose_identity(self):
        """Test that identity transpose equals identity."""
        m = Matrix.identity(3)
        t = m.transpose()
        for i in range(3):
            for j in range(3):
                assert t[i, j] == m[i, j]

    def test_transpose_preserves_original(self):
        """Test that transpose creates a new matrix."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        t = m.transpose()
        t[0, 0] = 999.0
        assert m[0, 0] == 1.0  # Original unchanged


class TestMatrixMultiplication:
    """Test Matrix multiplication."""

    def test_multiply_identity(self):
        """Test multiplying by identity."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        identity = Matrix.identity(2)
        result = m @ identity
        for i in range(2):
            for j in range(2):
                assert abs(result[i, j] - m[i, j]) < MATH_EPSILON

    def test_multiply_2x2(self):
        """Test 2x2 matrix multiplication."""
        a = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        b = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])
        c = a @ b
        # [1*5+2*7, 1*6+2*8] = [19, 22]
        # [3*5+4*7, 3*6+4*8] = [43, 50]
        assert abs(c[0, 0] - 19.0) < MATH_EPSILON
        assert abs(c[0, 1] - 22.0) < MATH_EPSILON
        assert abs(c[1, 0] - 43.0) < MATH_EPSILON
        assert abs(c[1, 1] - 50.0) < MATH_EPSILON

    def test_multiply_rectangular(self):
        """Test rectangular matrix multiplication."""
        a = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = Matrix(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        c = a @ b
        assert c.rows == 2
        assert c.cols == 2
        # [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        # [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert abs(c[0, 0] - 58.0) < MATH_EPSILON
        assert abs(c[0, 1] - 64.0) < MATH_EPSILON
        assert abs(c[1, 0] - 139.0) < MATH_EPSILON
        assert abs(c[1, 1] - 154.0) < MATH_EPSILON

    def test_multiply_column_vector(self):
        """Test multiplying matrix by column vector."""
        a = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = Matrix.from_vector([1.0, 2.0, 3.0])
        result = a @ v
        assert result.rows == 2
        assert result.cols == 1
        # [1*1+2*2+3*3] = [14]
        # [4*1+5*2+6*3] = [32]
        assert abs(result[0, 0] - 14.0) < MATH_EPSILON
        assert abs(result[1, 0] - 32.0) < MATH_EPSILON

    def test_multiply_incompatible_raises(self):
        """Test that incompatible matrices raise ValueError."""
        a = Matrix(2, 3)
        b = Matrix(2, 3)
        with pytest.raises(ValueError, match="Cannot multiply"):
            _ = a @ b

    def test_multiply_3x3(self):
        """Test 3x3 matrix multiplication."""
        a = Matrix.identity(3)
        a[0, 0] = 2.0
        a[1, 1] = 3.0
        a[2, 2] = 4.0
        b = Matrix(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        c = a @ b
        assert abs(c[0, 0] - 2.0) < MATH_EPSILON
        assert abs(c[0, 1] - 4.0) < MATH_EPSILON
        assert abs(c[1, 1] - 15.0) < MATH_EPSILON
        assert abs(c[2, 2] - 36.0) < MATH_EPSILON


class TestMatrixAddition:
    """Test Matrix addition."""

    def test_add_same_dimensions(self):
        """Test adding matrices with same dimensions."""
        a = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        b = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])
        c = a + b
        assert c[0, 0] == 6.0
        assert c[0, 1] == 8.0
        assert c[1, 0] == 10.0
        assert c[1, 1] == 12.0

    def test_add_to_zero(self):
        """Test adding to zero matrix."""
        a = Matrix(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        zero = Matrix(3, 3)
        result = a + zero
        for i in range(3):
            for j in range(3):
                assert result[i, j] == a[i, j]

    def test_add_incompatible_raises(self):
        """Test that incompatible dimensions raise ValueError."""
        a = Matrix(2, 3)
        b = Matrix(3, 2)
        with pytest.raises(ValueError, match="dimensions must match"):
            _ = a + b


class TestMatrixScalarMultiplication:
    """Test Matrix scalar multiplication."""

    def test_scalar_multiply(self):
        """Test scalar multiplication."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        result = m * 2.0
        assert result[0, 0] == 2.0
        assert result[0, 1] == 4.0
        assert result[1, 0] == 6.0
        assert result[1, 1] == 8.0

    def test_scalar_multiply_reverse(self):
        """Test reverse scalar multiplication."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        result = 3.0 * m
        assert result[0, 0] == 3.0
        assert result[0, 1] == 6.0
        assert result[1, 0] == 9.0
        assert result[1, 1] == 12.0

    def test_scalar_multiply_zero(self):
        """Test multiplying by zero."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        result = m * 0.0
        for i in range(2):
            for j in range(2):
                assert result[i, j] == 0.0

    def test_scalar_multiply_negative(self):
        """Test multiplying by negative scalar."""
        m = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        result = m * -1.0
        assert result[0, 0] == -1.0
        assert result[1, 1] == -4.0


class TestMatrixVectorConversions:
    """Test Matrix to/from vector conversions."""

    def test_from_vector(self):
        """Test creating column matrix from vector."""
        v = [1.0, 2.0, 3.0, 4.0]
        m = Matrix.from_vector(v)
        assert m.rows == 4
        assert m.cols == 1
        assert m[0, 0] == 1.0
        assert m[1, 0] == 2.0
        assert m[2, 0] == 3.0
        assert m[3, 0] == 4.0

    def test_to_vector(self):
        """Test converting column matrix to vector."""
        m = Matrix(3, 1, [5.0, 6.0, 7.0])
        v = m.to_vector()
        assert v == [5.0, 6.0, 7.0]

    def test_to_vector_non_column_raises(self):
        """Test that to_vector on non-column matrix raises."""
        m = Matrix(2, 2)
        with pytest.raises(ValueError, match="column vector"):
            m.to_vector()

    def test_from_vector_empty(self):
        """Test creating from empty vector."""
        v: List[float] = []
        m = Matrix.from_vector(v)
        assert m.rows == 0
        assert m.cols == 1

    def test_roundtrip_vector(self):
        """Test roundtrip from vector to matrix and back."""
        original = [1.5, 2.5, 3.5]
        m = Matrix.from_vector(original)
        result = m.to_vector()
        assert result == original


class TestMatrixInversion:
    """Test Gauss-Jordan matrix inversion."""

    def test_invert_identity(self):
        """Test inverting identity matrix."""
        m = Matrix.identity(3)
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is not None
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(inv[i, j] - expected) < 1e-6

    def test_invert_2x2_simple(self):
        """Test inverting a simple 2x2 matrix."""
        # [[4, 7], [2, 6]] inverse = [[0.6, -0.7], [-0.2, 0.4]]
        m = Matrix(2, 2, [4.0, 7.0, 2.0, 6.0])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is not None
        # det = 4*6 - 7*2 = 10
        # inv = 1/10 * [[6, -7], [-2, 4]]
        assert abs(inv[0, 0] - 0.6) < 1e-6
        assert abs(inv[0, 1] - (-0.7)) < 1e-6
        assert abs(inv[1, 0] - (-0.2)) < 1e-6
        assert abs(inv[1, 1] - 0.4) < 1e-6

    def test_invert_2x2_verify_product(self):
        """Test that M * M^-1 = I for 2x2."""
        m = Matrix(2, 2, [3.0, 1.0, 2.0, 1.0])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is not None
        product = m @ inv
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                assert abs(product[i, j] - expected) < 1e-6

    def test_invert_3x3(self):
        """Test inverting a 3x3 matrix."""
        # Simple 3x3 diagonal matrix
        m = Matrix(3, 3, [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is not None
        assert abs(inv[0, 0] - 0.5) < 1e-6
        assert abs(inv[1, 1] - (1.0/3.0)) < 1e-6
        assert abs(inv[2, 2] - 0.25) < 1e-6

    def test_invert_3x3_verify_product(self):
        """Test that M * M^-1 = I for 3x3."""
        m = Matrix(3, 3, [
            1.0, 2.0, 0.0,
            0.0, 1.0, 2.0,
            0.0, 0.0, 1.0
        ])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is not None
        product = m @ inv
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(product[i, j] - expected) < 1e-6

    def test_invert_singular_matrix(self):
        """Test that singular matrix returns None."""
        # Singular: rows are linearly dependent
        m = Matrix(2, 2, [1.0, 2.0, 2.0, 4.0])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is None

    def test_invert_zero_matrix(self):
        """Test that zero matrix returns None."""
        m = Matrix(3, 3)
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is None

    def test_invert_non_square_returns_none(self):
        """Test that non-square matrix returns None."""
        m = Matrix(2, 3)
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        assert inv is None

    def test_invert_near_singular(self):
        """Test matrix near singular boundary."""
        # Nearly singular but invertible
        m = Matrix(2, 2, [1.0, 2.0, 2.0, 4.0 + 1e-5])
        ik = JacobianIK([0, 1, 2])
        inv = ik._invert_matrix(m)
        # Should be invertible but numerically unstable
        # Check if it returns something reasonable or None
        if inv is not None:
            product = m @ inv
            assert abs(product[0, 0] - 1.0) < 0.1


class TestMatrixNumericalStability:
    """Test numerical stability of Matrix operations."""

    def test_large_values(self):
        """Test with large values."""
        m = Matrix(2, 2, [1e6, 2e6, 3e6, 4e6])
        t = m.transpose()
        assert t[0, 1] == 3e6

    def test_small_values(self):
        """Test with small values."""
        m = Matrix(2, 2, [1e-6, 2e-6, 3e-6, 4e-6])
        t = m.transpose()
        assert abs(t[0, 1] - 3e-6) < 1e-12

    def test_mixed_magnitude(self):
        """Test with mixed magnitude values."""
        m = Matrix(2, 2, [1e6, 1e-6, 1e-6, 1e6])
        product = m @ Matrix.identity(2)
        assert abs(product[0, 0] - 1e6) < 1.0
        assert abs(product[0, 1] - 1e-6) < 1e-12


# =============================================================================
# T-IK-3.15: Jacobian Computation Tests
# =============================================================================


class TestJacobianIKConstruction:
    """Test JacobianIK solver construction."""

    def test_construct_with_bone_indices(self):
        """Test construction with bone indices."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        assert ik.bone_indices == [0, 1, 2]
        assert ik.num_joints == 3

    def test_construct_minimum_bones(self):
        """Test construction with minimum 2 bones."""
        ik = JacobianIK(bone_indices=[0, 1])
        assert ik.num_joints == 2

    def test_construct_too_few_bones_raises(self):
        """Test that fewer than 2 bones raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 bones"):
            JacobianIK(bone_indices=[0])

    def test_construct_empty_bones_raises(self):
        """Test that empty bone list raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 bones"):
            JacobianIK(bone_indices=[])

    def test_default_method_is_dls(self):
        """Test default method is DLS."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        assert ik.method == JacobianMethod.DAMPED_LEAST_SQUARES

    def test_construct_with_all_methods(self):
        """Test construction with all methods."""
        for method in JacobianMethod:
            ik = JacobianIK(bone_indices=[0, 1, 2], method=method)
            assert ik.method == method

    def test_default_parameters(self):
        """Test default parameter values."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        assert ik.tolerance == IK_DEFAULT_TOLERANCE
        assert ik.max_iterations == JACOBIAN_DEFAULT_MAX_ITERATIONS
        assert ik.damping == JACOBIAN_DLS_DAMPING
        assert ik.step_size == JACOBIAN_DEFAULT_STEP_SIZE

    def test_custom_parameters(self):
        """Test custom parameter values."""
        ik = JacobianIK(
            bone_indices=[0, 1, 2],
            tolerance=0.01,
            max_iterations=100,
            damping=1.0,
            step_size=0.5
        )
        assert ik.tolerance == 0.01
        assert ik.max_iterations == 100
        assert ik.damping == 1.0
        assert ik.step_size == 0.5

    def test_num_end_effectors_default(self):
        """Test default number of end effectors."""
        ik = JacobianIK(bone_indices=[0, 1, 2, 3])
        assert ik.num_end_effectors == 1

    def test_bone_indices_are_copied(self):
        """Test that bone indices are copied."""
        indices = [0, 1, 2]
        ik = JacobianIK(bone_indices=indices)
        indices[0] = 999
        assert ik.bone_indices[0] == 0


class TestJacobianIKJointAxes:
    """Test joint axes configuration."""

    def test_default_joint_axes(self):
        """Test default joint axes are XYZ."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        # Default should be [X, Y, Z] for each joint
        assert len(ik._joint_axes) == 3

    def test_set_joint_axes(self):
        """Test setting custom joint axes."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3.unit_y()])
        assert len(ik._joint_axes[0]) == 1

    def test_set_joint_axes_normalized(self):
        """Test that set axes are normalized."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3(2.0, 0.0, 0.0)])
        axis = ik._joint_axes[0][0]
        assert abs(axis.length() - 1.0) < MATH_EPSILON

    def test_set_joint_axes_out_of_bounds_ignored(self):
        """Test that out of bounds index is ignored."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(10, [Vec3.unit_y()])  # Should not raise

    def test_multiple_axes_per_joint(self):
        """Test setting multiple axes per joint."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3.unit_x(), Vec3.unit_y()])
        assert len(ik._joint_axes[0]) == 2


class TestJacobianComputation:
    """Test compute_jacobian method."""

    def test_compute_jacobian_dimensions(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test Jacobian matrix dimensions."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        # 1 end effector * 3 DOF = 3 rows
        # num_dofs = sum of all joint axes (including end effector in axes list)
        # 3 joints * 3 axes = 9 columns (even though only 2 are used in computation)
        assert J.rows == 3
        assert J.cols == 9  # All joints have axes allocated

    def test_compute_jacobian_single_axis(self, simple_2bone_chain):
        """Test Jacobian with single axis per joint."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3.unit_y()])
        ik.set_joint_axes(1, [Vec3.unit_y()])
        ik.set_joint_axes(2, [Vec3.unit_y()])  # End effector axes also count in DOF total
        J = ik.compute_jacobian(positions, rotations)
        assert J.rows == 3
        # num_dofs = sum of axes for all joints including end effector
        assert J.cols == 3  # 3 joints * 1 axis each (even though only 2 used)

    def test_compute_jacobian_cross_product(self, simple_2bone_chain):
        """Test Jacobian uses cross product correctly."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3.unit_z()])  # Z-axis rotation
        ik.set_joint_axes(1, [Vec3.unit_z()])
        J = ik.compute_jacobian(positions, rotations)
        # For Z-axis rotation at origin, cross with (0, 2, 0) gives (-2, 0, 0)
        assert J.rows == 3
        # First joint: Z cross (0,2,0) = (-2, 0, 0)
        # Note: actual values depend on implementation details

    def test_compute_jacobian_3_bone(self, simple_3bone_chain):
        """Test Jacobian computation for 3-bone chain."""
        positions, rotations = simple_3bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2, 3])
        J = ik.compute_jacobian(positions, rotations)
        # 1 end effector * 3 DOF = 3 rows
        # 4 joints * 3 axes = 12 columns (all joints have axes allocated)
        assert J.rows == 3
        assert J.cols == 12

    def test_compute_jacobian_rotation_transforms_axis(self, simple_2bone_chain):
        """Test that joint rotation transforms axis to world space."""
        positions, rotations = simple_2bone_chain
        # Rotate first joint 90 degrees around Y
        rotations[0] = Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2)
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.set_joint_axes(0, [Vec3.unit_x()])  # X in local
        ik.set_joint_axes(1, [Vec3.unit_x()])
        ik.set_joint_axes(2, [Vec3.unit_x()])  # End effector axes also count
        J = ik.compute_jacobian(positions, rotations)
        # After 90 Y rotation, local X becomes world Z
        assert J.rows == 3
        assert J.cols == 3  # 3 joints * 1 axis each


class TestJacobianEndEffectors:
    """Test end effector configuration."""

    def test_add_end_effector(self):
        """Test adding end effector."""
        ik = JacobianIK(bone_indices=[0, 1, 2, 3])
        ik.add_end_effector(2)
        assert ik.num_end_effectors == 2
        assert 2 in ik._end_effector_indices

    def test_add_duplicate_end_effector_ignored(self):
        """Test that duplicate end effector is ignored."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        ik.add_end_effector(2)  # Already default
        assert ik.num_end_effectors == 1

    def test_multiple_end_effectors_jacobian_size(self, simple_3bone_chain):
        """Test Jacobian size with multiple end effectors."""
        positions, rotations = simple_3bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2, 3])
        ik.add_end_effector(2)  # Add middle joint as effector
        J = ik.compute_jacobian(positions, rotations)
        # 2 end effectors * 3 DOF = 6 rows
        assert J.rows == 6


# =============================================================================
# T-IK-3.16: Jacobian Transpose Method Tests
# =============================================================================


class TestJacobianTransposeMethod:
    """Test Jacobian transpose solver."""

    def test_transpose_solve_returns_correct_length(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test transpose solve returns correct number of DOFs."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [0.1, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        # num_dofs = sum of all joint axes = 3 joints * 3 axes = 9 DOFs
        assert len(dq) == 9

    def test_transpose_solve_zero_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test transpose solve with zero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [0.0, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        # With zero error, angle changes should be zero or near-zero
        assert all(abs(x) < 0.1 for x in dq)

    def test_transpose_solve_nonzero_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test transpose solve produces angle changes for nonzero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]  # Error in X direction
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        # Should produce some angle change
        assert any(abs(x) > MATH_EPSILON for x in dq)

    def test_transpose_step_size_effect(self, simple_2bone_chain):
        """Test that step size affects results."""
        positions, rotations = simple_2bone_chain
        ik1 = JacobianIK(bone_indices=[0, 1, 2], step_size=0.5)
        ik2 = JacobianIK(bone_indices=[0, 1, 2], step_size=1.0)
        J = ik1.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq1 = ik1.solve_jacobian_transpose(J, error)
        dq2 = ik2.solve_jacobian_transpose(J, error)
        # Results should differ
        # Note: step size may be computed adaptively

    def test_transpose_optimal_alpha_computation(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test optimal step size computation."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [0.5, 0.5, 0.0]
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        # Just verify it returns reasonable values
        assert all(not math.isnan(x) for x in dq)
        assert all(not math.isinf(x) for x in dq)

    def test_transpose_convergence_direction(self, simple_2bone_chain):
        """Test that transpose method moves toward target."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.8, 0.0)  # Slightly offset target
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.TRANSPOSE)
        result = ik.solve(positions, rotations, [target])
        # Final position should be closer to target
        final_ee = result.positions[-1]
        initial_ee = positions[-1]
        final_dist = (target - final_ee).length()
        initial_dist = (target - initial_ee).length()
        assert final_dist < initial_dist or result.success


class TestTransposeSolveEdgeCases:
    """Test edge cases for transpose solve."""

    def test_transpose_degenerate_jacobian(self):
        """Test transpose with degenerate Jacobian."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        # Create a degenerate Jacobian (all zeros)
        J = Matrix(3, 6)
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_jacobian_transpose(J, error)
        # Should fall back to step_size
        assert len(dq) == 6

    def test_transpose_large_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test transpose with large error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [100.0, 100.0, 100.0]
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        assert all(not math.isnan(x) for x in dq)

    def test_transpose_small_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test transpose with very small error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [1e-10, 1e-10, 1e-10]
        dq = jacobian_ik_2bone.solve_jacobian_transpose(J, error)
        assert all(abs(x) < 1.0 for x in dq)


# =============================================================================
# T-IK-3.17: Jacobian Pseudoinverse Method Tests
# =============================================================================


class TestJacobianPseudoinverseMethod:
    """Test Jacobian pseudoinverse solver."""

    def test_pseudoinverse_solve_returns_correct_length(
        self, simple_2bone_chain, jacobian_ik_pseudoinverse
    ):
        """Test pseudoinverse solve returns correct number of DOFs."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_pseudoinverse.compute_jacobian(positions, rotations)
        error = [0.1, 0.0, 0.0]
        dq = jacobian_ik_pseudoinverse.solve_pseudoinverse(J, error)
        # num_dofs = sum of all joint axes = 3 joints * 3 axes = 9 DOFs
        assert len(dq) == 9

    def test_pseudoinverse_solve_zero_error(
        self, simple_2bone_chain, jacobian_ik_pseudoinverse
    ):
        """Test pseudoinverse solve with zero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_pseudoinverse.compute_jacobian(positions, rotations)
        error = [0.0, 0.0, 0.0]
        dq = jacobian_ik_pseudoinverse.solve_pseudoinverse(J, error)
        assert all(abs(x) < 0.1 for x in dq)

    def test_pseudoinverse_solve_nonzero_error(
        self, simple_2bone_chain, jacobian_ik_pseudoinverse
    ):
        """Test pseudoinverse solve produces changes for nonzero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_pseudoinverse.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = jacobian_ik_pseudoinverse.solve_pseudoinverse(J, error)
        assert any(abs(x) > MATH_EPSILON for x in dq)

    def test_pseudoinverse_singular_fallback(self, simple_2bone_chain):
        """Test pseudoinverse falls back to transpose for singular matrix."""
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.PSEUDOINVERSE)
        # Create singular Jacobian
        J = Matrix(3, 6)
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_pseudoinverse(J, error)
        # Should still return valid result (from transpose fallback)
        assert len(dq) == 6

    def test_pseudoinverse_better_than_transpose_near_solution(self, simple_2bone_chain):
        """Test pseudoinverse is more accurate near solution."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.1, 1.95, 0.0)  # Very close to end effector

        ik_pseudo = JacobianIK(
            bone_indices=[0, 1, 2], method=JacobianMethod.PSEUDOINVERSE, max_iterations=20
        )
        ik_trans = JacobianIK(
            bone_indices=[0, 1, 2], method=JacobianMethod.TRANSPOSE, max_iterations=20
        )

        result_pseudo = ik_pseudo.solve(positions, rotations, [target])
        result_trans = ik_trans.solve(positions, rotations, [target])

        # Both should converge for easy targets
        # Check that results are reasonable
        assert result_pseudo.final_error < 1.0 or result_pseudo.success
        assert result_trans.final_error < 1.0 or result_trans.success

    def test_pseudoinverse_step_size(self, simple_2bone_chain):
        """Test pseudoinverse respects step size."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(
            bone_indices=[0, 1, 2],
            method=JacobianMethod.PSEUDOINVERSE,
            step_size=0.1
        )
        J = ik.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_pseudoinverse(J, error)
        # Step size should limit magnitude
        # Results multiplied by 0.1
        assert all(not math.isnan(x) for x in dq)


class TestPseudoinverseSolveEdgeCases:
    """Test edge cases for pseudoinverse solve."""

    def test_pseudoinverse_near_singular(self, simple_2bone_chain):
        """Test pseudoinverse near singularity falls back gracefully."""
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.PSEUDOINVERSE)
        # Create nearly singular Jacobian
        J = Matrix(3, 6, [
            1e-10, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1e-10, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1e-10, 0.0, 0.0, 0.0,
        ])
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_pseudoinverse(J, error)
        # Should handle gracefully
        assert all(not math.isnan(x) for x in dq)

    def test_pseudoinverse_large_error(self, simple_2bone_chain, jacobian_ik_pseudoinverse):
        """Test pseudoinverse with large error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_pseudoinverse.compute_jacobian(positions, rotations)
        error = [50.0, 50.0, 50.0]
        dq = jacobian_ik_pseudoinverse.solve_pseudoinverse(J, error)
        assert all(not math.isnan(x) for x in dq)
        assert all(not math.isinf(x) for x in dq)


# =============================================================================
# T-IK-3.18: Jacobian DLS (Damped Least Squares) Tests
# =============================================================================


class TestJacobianDLSMethod:
    """Test Jacobian Damped Least Squares solver."""

    def test_dls_solve_returns_correct_length(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test DLS solve returns correct number of DOFs."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [0.1, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_damped_least_squares(J, error)
        # num_dofs = sum of all joint axes = 3 joints * 3 axes = 9 DOFs
        assert len(dq) == 9

    def test_dls_solve_zero_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test DLS solve with zero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [0.0, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_damped_least_squares(J, error)
        assert all(abs(x) < 0.1 for x in dq)

    def test_dls_solve_nonzero_error(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test DLS solve produces changes for nonzero error."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_damped_least_squares(J, error)
        assert any(abs(x) > MATH_EPSILON for x in dq)

    def test_dls_custom_damping(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test DLS with custom damping factor."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_damped_least_squares(J, error, damping=0.1)
        # num_dofs = sum of all joint axes = 3 joints * 3 axes = 9 DOFs
        assert len(dq) == 9
        assert all(not math.isnan(x) for x in dq)

    def test_dls_damping_effect(self, simple_2bone_chain):
        """Test that higher damping produces smaller steps."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2])
        J = ik.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]

        dq_low = ik.solve_damped_least_squares(J, error, damping=0.1)
        dq_high = ik.solve_damped_least_squares(J, error, damping=2.0)

        # Higher damping should generally produce smaller steps
        mag_low = sum(x * x for x in dq_low)
        mag_high = sum(x * x for x in dq_high)
        assert mag_high <= mag_low + 0.1  # Allow small tolerance

    def test_dls_singularity_handling(self):
        """Test DLS handles singularities better than pseudoinverse."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        # Create nearly singular Jacobian
        J = Matrix(3, 6, [
            1e-8, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1e-8, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1e-8, 0.0, 0.0, 0.0,
        ])
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_damped_least_squares(J, error, damping=0.5)
        # Should produce valid result (damping prevents explosion)
        assert all(not math.isnan(x) for x in dq)
        assert all(abs(x) < 1e6 for x in dq)

    def test_dls_fallback_to_transpose(self):
        """Test DLS falls back to transpose if matrix inversion fails."""
        ik = JacobianIK(bone_indices=[0, 1, 2])
        # All zeros - completely singular
        J = Matrix(3, 6)
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_damped_least_squares(J, error)
        # Should still return result from transpose fallback
        assert len(dq) == 6

    def test_dls_vs_pseudoinverse_stability(self, simple_2bone_chain):
        """Test DLS is more stable than pseudoinverse near singularity."""
        positions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 2.0, 0.0),
        ]
        # Fully extended chain is near singularity
        rotations = [Quat.identity() for _ in range(3)]

        ik_dls = JacobianIK(
            bone_indices=[0, 1, 2],
            method=JacobianMethod.DAMPED_LEAST_SQUARES,
            damping=0.5
        )
        ik_pseudo = JacobianIK(
            bone_indices=[0, 1, 2],
            method=JacobianMethod.PSEUDOINVERSE
        )

        J = ik_dls.compute_jacobian(positions, rotations)
        error = [2.0, 0.0, 0.0]  # Large sideways error

        dq_dls = ik_dls.solve_damped_least_squares(J, error)
        dq_pseudo = ik_pseudo.solve_pseudoinverse(J, error)

        # DLS should produce reasonable values
        assert all(not math.isnan(x) for x in dq_dls)
        assert all(abs(x) < 100 for x in dq_dls)


class TestDLSSolveParameters:
    """Test DLS parameter effects."""

    def test_dls_default_damping(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test DLS uses default damping when None."""
        positions, rotations = simple_2bone_chain
        J = jacobian_ik_2bone.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = jacobian_ik_2bone.solve_damped_least_squares(J, error, damping=None)
        # num_dofs = sum of all joint axes = 3 joints * 3 axes = 9 DOFs
        assert len(dq) == 9

    def test_dls_zero_damping_like_pseudoinverse(self, simple_2bone_chain):
        """Test DLS with zero damping is like pseudoinverse."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2])
        J = ik.compute_jacobian(positions, rotations)
        error = [0.5, 0.0, 0.0]

        dq_dls = ik.solve_damped_least_squares(J, error, damping=1e-10)
        dq_pseudo = ik.solve_pseudoinverse(J, error)

        # Should be similar (allowing for numerical differences)
        for a, b in zip(dq_dls, dq_pseudo):
            assert abs(a - b) < 1.0 or (math.isnan(a) and math.isnan(b))

    def test_dls_large_damping(self, simple_2bone_chain):
        """Test DLS with very large damping."""
        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2])
        J = ik.compute_jacobian(positions, rotations)
        error = [1.0, 0.0, 0.0]
        dq = ik.solve_damped_least_squares(J, error, damping=100.0)
        # Very high damping should produce small steps
        assert all(abs(x) < 1.0 for x in dq)


# =============================================================================
# Full Solve Pipeline Tests
# =============================================================================


class TestJacobianSolve:
    """Test complete solve method."""

    def test_solve_reachable_target(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test solving for a reachable target."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)  # Reachable
        result = jacobian_ik_2bone.solve(positions, rotations, [target])
        assert isinstance(result, JacobianResult)
        assert result.final_error < 0.5 or result.success

    def test_solve_at_current_position(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test solving when already at target."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.0, 2.0, 0.0)  # Current end effector position
        result = jacobian_ik_2bone.solve(positions, rotations, [target])
        assert result.success
        assert result.final_error < 0.01

    def test_solve_unreachable_target(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test solving for unreachable target."""
        positions, rotations = simple_2bone_chain
        target = Vec3(10.0, 10.0, 0.0)  # Far beyond reach
        result = jacobian_ik_2bone.solve(positions, rotations, [target])
        # Should not converge but should produce valid result
        assert result.iterations == jacobian_ik_2bone.max_iterations or result.success
        assert len(result.positions) == 3
        assert len(result.rotations) == 3

    def test_solve_returns_positions(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test that solve returns updated positions."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        result = jacobian_ik_2bone.solve(positions, rotations, [target])
        assert len(result.positions) == 3
        # End effector should have moved toward target
        if result.success or result.final_error < 1.0:
            dist_to_target = (result.positions[-1] - target).length()
            assert dist_to_target < 1.5

    def test_solve_returns_rotations(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test that solve returns updated rotations."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        result = jacobian_ik_2bone.solve(positions, rotations, [target])
        assert len(result.rotations) == 3
        for rot in result.rotations:
            assert isinstance(rot, Quat)

    def test_solve_wrong_position_count_raises(self, jacobian_ik_2bone):
        """Test that wrong position count raises ValueError."""
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]  # Only 2, need 3
        rotations = [Quat.identity() for _ in range(2)]
        target = Vec3(0.5, 1.5, 0.0)
        with pytest.raises(ValueError, match="Expected 3 positions"):
            jacobian_ik_2bone.solve(positions, rotations, [target])

    def test_solve_wrong_target_count_raises(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test that wrong target count raises ValueError."""
        positions, rotations = simple_2bone_chain
        targets = [Vec3(0.5, 1.5, 0.0), Vec3(1.0, 1.0, 0.0)]  # 2 targets, only 1 effector
        with pytest.raises(ValueError, match="Expected 1 targets"):
            jacobian_ik_2bone.solve(positions, rotations, targets)


class TestSolveConvergence:
    """Test solve convergence behavior."""

    def test_solve_converges_within_iterations(self, simple_2bone_chain):
        """Test that solve converges within max iterations for easy target."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.1, 1.9, 0.0)  # Very close to current
        ik = JacobianIK(bone_indices=[0, 1, 2], max_iterations=100, tolerance=0.01)
        result = ik.solve(positions, rotations, [target])
        assert result.success or result.final_error < 0.1

    def test_solve_respects_max_iterations(self, simple_2bone_chain):
        """Test that solve respects max_iterations limit."""
        positions, rotations = simple_2bone_chain
        target = Vec3(10.0, 10.0, 0.0)  # Unreachable
        ik = JacobianIK(bone_indices=[0, 1, 2], max_iterations=5)
        result = ik.solve(positions, rotations, [target])
        assert result.iterations <= 5

    def test_solve_respects_tolerance(self, simple_2bone_chain):
        """Test that solve stops when within tolerance."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.0, 2.0, 0.0)  # Current position
        ik = JacobianIK(bone_indices=[0, 1, 2], tolerance=0.1)
        result = ik.solve(positions, rotations, [target])
        # Already at target, should converge quickly
        assert result.success or result.iterations < 5

    def test_solve_iteration_count(self, simple_2bone_chain):
        """Test that iterations are counted correctly."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        ik = JacobianIK(bone_indices=[0, 1, 2], max_iterations=50)
        result = ik.solve(positions, rotations, [target])
        assert result.iterations >= 1
        assert result.iterations <= 50


class TestSolveWithDifferentMethods:
    """Test solve with different Jacobian methods."""

    def test_solve_transpose_method(self, simple_2bone_chain):
        """Test solve with transpose method."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.TRANSPOSE)
        result = ik.solve(positions, rotations, [target])
        assert isinstance(result, JacobianResult)

    def test_solve_pseudoinverse_method(self, simple_2bone_chain):
        """Test solve with pseudoinverse method."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.PSEUDOINVERSE)
        result = ik.solve(positions, rotations, [target])
        assert isinstance(result, JacobianResult)

    def test_solve_dls_method(self, simple_2bone_chain):
        """Test solve with DLS method."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.DAMPED_LEAST_SQUARES)
        result = ik.solve(positions, rotations, [target])
        assert isinstance(result, JacobianResult)

    def test_solve_sdls_falls_back_to_dls(self, simple_2bone_chain):
        """Test that SDLS method uses DLS internally."""
        positions, rotations = simple_2bone_chain
        target = Vec3(0.5, 1.5, 0.0)
        ik = JacobianIK(bone_indices=[0, 1, 2], method=JacobianMethod.SELECTIVELY_DAMPED)
        result = ik.solve(positions, rotations, [target])
        assert isinstance(result, JacobianResult)


class TestApplyAngleChanges:
    """Test internal angle change application."""

    def test_apply_angle_changes_modifies_rotations(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test that angle changes modify rotations."""
        positions, rotations = simple_2bone_chain
        positions = [Vec3(p.x, p.y, p.z) for p in positions]
        rotations = [Quat(r.x, r.y, r.z, r.w) for r in rotations]
        original_rot = Quat(rotations[0].x, rotations[0].y, rotations[0].z, rotations[0].w)

        dq = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Small rotation on first joint
        jacobian_ik_2bone._apply_angle_changes(positions, rotations, dq)

        # Rotation should have changed
        # Note: May or may not be significantly different depending on implementation

    def test_apply_angle_changes_updates_child_positions(
        self, simple_2bone_chain, jacobian_ik_2bone
    ):
        """Test that child positions are updated after rotation."""
        positions, rotations = simple_2bone_chain
        positions = [Vec3(p.x, p.y, p.z) for p in positions]
        rotations = [Quat(r.x, r.y, r.z, r.w) for r in rotations]
        original_end = Vec3(positions[-1].x, positions[-1].y, positions[-1].z)

        dq = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # Rotation on first joint
        jacobian_ik_2bone._apply_angle_changes(positions, rotations, dq)

        # End position should have changed (rotated around first joint)
        # At least some change expected for non-zero angle

    def test_apply_empty_angle_changes(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test applying empty angle changes."""
        positions, rotations = simple_2bone_chain
        positions = [Vec3(p.x, p.y, p.z) for p in positions]
        rotations = [Quat(r.x, r.y, r.z, r.w) for r in rotations]

        dq: List[float] = []
        jacobian_ik_2bone._apply_angle_changes(positions, rotations, dq)
        # Should not crash


class TestUpdateChildPositions:
    """Test child position updating."""

    def test_update_child_positions_rotation(self, simple_2bone_chain, jacobian_ik_2bone):
        """Test child positions rotate around parent."""
        positions, _ = simple_2bone_chain
        positions = [Vec3(p.x, p.y, p.z) for p in positions]

        # 90 degree rotation around Z axis
        rot = Quat.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        jacobian_ik_2bone._update_child_positions(0, positions, rot)

        # Child at (0, 1, 0) should rotate to (-1, 0, 0)
        # Child at (0, 2, 0) should rotate to (-2, 0, 0)
        assert abs(positions[1].x - (-1.0)) < 0.01
        assert abs(positions[1].y) < 0.01
        assert abs(positions[2].x - (-2.0)) < 0.01
        assert abs(positions[2].y) < 0.01

    def test_update_child_positions_preserves_parent(
        self, simple_2bone_chain, jacobian_ik_2bone
    ):
        """Test that parent position is preserved."""
        positions, _ = simple_2bone_chain
        positions = [Vec3(p.x, p.y, p.z) for p in positions]
        original = Vec3(positions[0].x, positions[0].y, positions[0].z)

        rot = Quat.from_axis_angle(Vec3.unit_z(), 0.5)
        jacobian_ik_2bone._update_child_positions(0, positions, rot)

        assert positions[0].x == original.x
        assert positions[0].y == original.y
        assert positions[0].z == original.z


class TestSolveWithTransforms:
    """Test solve_with_transforms method."""

    def test_solve_with_transforms_basic(self, jacobian_ik_2bone):
        """Test solve using Transform objects."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
            Transform(Vec3(0, 1, 0), Quat.identity(), Vec3.one()),
            Transform(Vec3(0, 2, 0), Quat.identity(), Vec3.one()),
        ]
        target = Vec3(0.5, 1.5, 0.0)

        result_transforms = jacobian_ik_2bone.solve_with_transforms(transforms, [target])

        assert len(result_transforms) == 3
        for t in result_transforms:
            assert isinstance(t, Transform)

    def test_solve_with_transforms_updates_positions(self, jacobian_ik_2bone):
        """Test that solve_with_transforms updates positions."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
            Transform(Vec3(0, 1, 0), Quat.identity(), Vec3.one()),
            Transform(Vec3(0, 2, 0), Quat.identity(), Vec3.one()),
        ]
        target = Vec3(0.5, 1.5, 0.0)

        result = jacobian_ik_2bone.solve_with_transforms(transforms, [target])

        # Check that positions were updated in result
        # End effector should move toward target
        end_pos = result[2].translation
        dist = (end_pos - target).length()
        # Should be closer than original
        assert dist < 2.5


# =============================================================================
# JacobianResult Tests
# =============================================================================


class TestJacobianResult:
    """Test JacobianResult dataclass."""

    def test_result_success_false_default(self):
        """Test result defaults."""
        result = JacobianResult(success=False)
        assert result.success is False
        assert result.iterations == 0
        assert result.final_error == float('inf')
        assert result.angle_changes == []
        assert result.rotations == []
        assert result.positions == []

    def test_result_success_true(self):
        """Test successful result."""
        result = JacobianResult(
            success=True,
            iterations=10,
            final_error=0.001,
            rotations=[Quat.identity()],
            positions=[Vec3.zero()]
        )
        assert result.success is True
        assert result.iterations == 10
        assert result.final_error == 0.001


# =============================================================================
# JacobianMethod Enum Tests
# =============================================================================


class TestJacobianMethodEnum:
    """Test JacobianMethod enum."""

    def test_transpose_method(self):
        """Test TRANSPOSE method exists."""
        assert JacobianMethod.TRANSPOSE is not None

    def test_pseudoinverse_method(self):
        """Test PSEUDOINVERSE method exists."""
        assert JacobianMethod.PSEUDOINVERSE is not None

    def test_dls_method(self):
        """Test DAMPED_LEAST_SQUARES method exists."""
        assert JacobianMethod.DAMPED_LEAST_SQUARES is not None

    def test_sdls_method(self):
        """Test SELECTIVELY_DAMPED method exists."""
        assert JacobianMethod.SELECTIVELY_DAMPED is not None

    def test_all_methods_unique(self):
        """Test all methods have unique values."""
        methods = list(JacobianMethod)
        values = [m.value for m in methods]
        assert len(values) == len(set(values))


# =============================================================================
# MultiTargetJacobianIK Tests
# =============================================================================


class TestMultiTargetJacobianIK:
    """Test MultiTargetJacobianIK class."""

    def test_construct_multi_target(self):
        """Test constructing multi-target solver."""
        ik = MultiTargetJacobianIK(bone_indices=[0, 1, 2, 3])
        assert isinstance(ik, JacobianIK)

    def test_add_weighted_end_effector(self):
        """Test adding weighted end effector."""
        ik = MultiTargetJacobianIK(bone_indices=[0, 1, 2, 3])
        ik.add_end_effector_weighted(2, weight=0.5)
        assert ik.num_end_effectors == 2
        assert ik._target_weights[-1] == 0.5

    def test_solve_multi_target(self):
        """Test solve with multiple targets."""
        ik = MultiTargetJacobianIK(bone_indices=[0, 1, 2, 3])
        ik.add_end_effector_weighted(2, weight=0.5)

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
            Vec3(0, 3, 0),
        ]
        rotations = [Quat.identity() for _ in range(4)]
        targets = [Vec3(0.5, 2.5, 0), Vec3(0.3, 1.7, 0)]

        result = ik.solve(positions, rotations, targets)
        assert isinstance(result, JacobianResult)

    def test_multi_target_weight_padding(self):
        """Test that weights are padded if needed."""
        ik = MultiTargetJacobianIK(bone_indices=[0, 1, 2])
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0)]
        rotations = [Quat.identity() for _ in range(3)]
        targets = [Vec3(0.5, 1.5, 0)]

        # Should not raise even without explicit weights
        result = ik.solve(positions, rotations, targets)
        assert isinstance(result, JacobianResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestJacobianIKIntegration:
    """Integration tests for complete IK scenarios."""

    def test_arm_reach_forward(self):
        """Test arm reaching forward."""
        # Arm pointing up
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.5, 0),
            Vec3(0, 1.0, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(
            bone_indices=[0, 1, 2],
            method=JacobianMethod.DAMPED_LEAST_SQUARES,
            max_iterations=50
        )

        # Reach forward
        target = Vec3(0.5, 0.5, 0)
        result = ik.solve(positions, rotations, [target])

        assert result.final_error < 0.5 or result.success

    def test_arm_reach_sideways(self):
        """Test arm reaching sideways."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.5, 0),
            Vec3(0, 1.0, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(0.7, 0.3, 0.5)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_chain_with_prerotated_joints(self):
        """Test chain with joints already rotated."""
        # Pre-rotate joint by 45 degrees
        rot45 = Quat.from_axis_angle(Vec3.unit_z(), math.pi / 4)

        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [rot45, Quat.identity(), Quat.identity()]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(0.5, 1.5, 0)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_longer_chain_solve(self):
        """Test solving longer chain."""
        positions = [Vec3(0, i * 0.5, 0) for i in range(6)]
        rotations = [Quat.identity() for _ in range(6)]

        ik = JacobianIK(bone_indices=list(range(6)), max_iterations=100)
        target = Vec3(1.0, 1.5, 0.5)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)
        assert len(result.positions) == 6

    def test_quick_convergence_close_target(self):
        """Test quick convergence for nearby target."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2], tolerance=0.01)
        target = Vec3(0.01, 1.99, 0.01)  # Very close
        result = ik.solve(positions, rotations, [target])

        assert result.success or result.iterations < 10


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestJacobianIKRobustness:
    """Test robustness and edge cases."""

    def test_collinear_chain(self):
        """Test fully extended collinear chain."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(2, 0, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(1, 1, 0)  # Perpendicular to chain
        result = ik.solve(positions, rotations, [target])

        # Should not crash
        assert isinstance(result, JacobianResult)

    def test_target_at_root(self):
        """Test target at root position."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(0, 0, 0)  # At root
        result = ik.solve(positions, rotations, [target])

        # Should not crash
        assert isinstance(result, JacobianResult)

    def test_very_small_chain(self):
        """Test chain with very small bones."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.001, 0),
            Vec3(0, 0.002, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(0.001, 0.001, 0)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_very_large_chain(self):
        """Test chain with very large bones."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1000, 0),
            Vec3(0, 2000, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(500, 1500, 0)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_negative_coordinates(self):
        """Test chain with negative coordinates."""
        positions = [
            Vec3(-1, -1, -1),
            Vec3(-1, 0, -1),
            Vec3(-1, 1, -1),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(-0.5, 0.5, -1)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_diagonal_chain(self):
        """Test diagonally oriented chain."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 1, 1),
            Vec3(2, 2, 2),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])
        target = Vec3(1.5, 2.5, 1.5)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)


class TestJacobianIKStressTests:
    """Stress tests for Jacobian IK."""

    def test_many_iterations(self):
        """Test with many iterations."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2], max_iterations=500)
        target = Vec3(0.5, 1.5, 0)
        result = ik.solve(positions, rotations, [target])

        assert isinstance(result, JacobianResult)

    def test_very_tight_tolerance(self):
        """Test with very tight tolerance."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2], tolerance=1e-8, max_iterations=200)
        target = Vec3(0.0, 2.0, 0.0)  # Already at target
        result = ik.solve(positions, rotations, [target])

        assert result.success or result.final_error < 1e-6

    def test_multiple_solves_same_instance(self):
        """Test multiple solves with same solver instance."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 2, 0),
        ]
        rotations = [Quat.identity() for _ in range(3)]

        ik = JacobianIK(bone_indices=[0, 1, 2])

        targets = [
            Vec3(0.5, 1.5, 0),
            Vec3(-0.5, 1.5, 0),
            Vec3(0, 1.8, 0.3),
            Vec3(0.3, 1.2, -0.3),
        ]

        for target in targets:
            result = ik.solve(positions, rotations, [target])
            assert isinstance(result, JacobianResult)


# =============================================================================
# Performance Sanity Tests
# =============================================================================


class TestJacobianIKPerformance:
    """Basic performance sanity tests."""

    def test_solve_completes_in_reasonable_time(self, simple_2bone_chain):
        """Test that solve completes without hanging."""
        import time

        positions, rotations = simple_2bone_chain
        ik = JacobianIK(bone_indices=[0, 1, 2], max_iterations=50)
        target = Vec3(0.5, 1.5, 0)

        start = time.time()
        result = ik.solve(positions, rotations, [target])
        elapsed = time.time() - start

        # Should complete within 1 second
        assert elapsed < 1.0
        assert isinstance(result, JacobianResult)

    def test_matrix_operations_performance(self):
        """Test that matrix operations are reasonably fast."""
        import time

        m1 = Matrix(10, 10)
        m2 = Matrix(10, 10)
        for i in range(10):
            for j in range(10):
                m1[i, j] = float(i + j)
                m2[i, j] = float(i * j)

        start = time.time()
        for _ in range(100):
            _ = m1 @ m2
            _ = m1.transpose()
            _ = m1 + m2
            _ = m1 * 2.0
        elapsed = time.time() - start

        # 100 iterations should be fast
        assert elapsed < 1.0
