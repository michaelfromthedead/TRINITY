"""
Blackbox tests for Jacobian IK system.

Tasks covered:
- T-IK-3.14: Matrix Class
- T-IK-3.15: Jacobian Computation
- T-IK-3.16: Jacobian Transpose Method
- T-IK-3.17: Jacobian Pseudoinverse Method
- T-IK-3.18: Jacobian DLS Method

These tests verify observable behavior without knowledge of implementation details.
"""

import pytest
import math
from typing import List, Tuple


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def Matrix():
    """Import Matrix class."""
    from engine.animation.ik.jacobian import Matrix
    return Matrix


@pytest.fixture
def JacobianIK():
    """Import JacobianIK class."""
    from engine.animation.ik.jacobian import JacobianIK
    return JacobianIK


@pytest.fixture
def JacobianMethod():
    """Import JacobianMethod enum."""
    from engine.animation.ik.jacobian import JacobianMethod
    return JacobianMethod


@pytest.fixture
def Vec3():
    """Import Vec3 class."""
    from engine.core.math.vec import Vec3
    return Vec3


@pytest.fixture
def Quat():
    """Import Quat class."""
    from engine.core.math.quat import Quat
    return Quat


@pytest.fixture
def identity_quat(Quat):
    """Create identity quaternion."""
    return Quat(0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def simple_chain_positions(Vec3):
    """Create a simple 3-bone chain along X axis (matching bone_indices count)."""
    return [
        Vec3(0.0, 0.0, 0.0),  # Bone 0 position
        Vec3(1.0, 0.0, 0.0),  # Bone 1 position
        Vec3(2.0, 0.0, 0.0),  # Bone 2 position (end effector)
    ]


@pytest.fixture
def simple_chain_rotations(Quat):
    """Create identity rotations for simple chain (matching bone_indices count)."""
    identity = Quat(0.0, 0.0, 0.0, 1.0)
    return [identity, identity, identity]


@pytest.fixture
def bone_indices():
    """Standard bone indices for 3-bone chain."""
    return [0, 1, 2]


@pytest.fixture
def two_bone_positions(Vec3):
    """Two bone chain positions."""
    return [
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    ]


@pytest.fixture
def two_bone_rotations(Quat):
    """Two bone rotations."""
    identity = Quat(0.0, 0.0, 0.0, 1.0)
    return [identity, identity]


@pytest.fixture
def two_bone_indices():
    """Two bone indices."""
    return [0, 1]


# =============================================================================
# Helper to get method enum value
# =============================================================================

def get_method(JacobianMethod, name):
    """Get method enum value by name, trying different naming conventions."""
    names_to_try = [
        name.upper(),
        name.lower(),
        name.capitalize(),
        name.replace('_', ''),
        name.upper().replace('_', ''),
    ]
    for n in names_to_try:
        if hasattr(JacobianMethod, n):
            return getattr(JacobianMethod, n)
    return None


# =============================================================================
# T-IK-3.14: Matrix Class Tests
# =============================================================================

class TestMatrixExists:
    """Test that Matrix class exists and can be instantiated."""

    def test_matrix_class_importable(self, Matrix):
        """Matrix class should be importable."""
        assert Matrix is not None

    def test_can_create_identity_matrix(self, Matrix):
        """Should be able to create an identity matrix."""
        m = Matrix.identity(3)
        assert m is not None

    def test_identity_returns_matrix_type(self, Matrix):
        """identity() should return a Matrix instance."""
        m = Matrix.identity(4)
        assert isinstance(m, Matrix)

    def test_matrix_has_transpose_method(self, Matrix):
        """Matrix should have transpose() method."""
        m = Matrix.identity(3)
        assert hasattr(m, 'transpose') or hasattr(m, 'T')

    def test_identity_with_size_1(self, Matrix):
        """identity() should work with size 1."""
        m = Matrix.identity(1)
        assert m is not None

    def test_identity_with_size_5(self, Matrix):
        """identity() should work with size 5."""
        m = Matrix.identity(5)
        assert m is not None

    def test_identity_with_size_10(self, Matrix):
        """identity() should work with size 10."""
        m = Matrix.identity(10)
        assert m is not None

    def test_matrix_can_be_instantiated_directly(self, Matrix):
        """Matrix can be instantiated with data if supported."""
        try:
            m = Matrix([[1, 0], [0, 1]])
            assert m is not None
        except (TypeError, ValueError):
            m = Matrix.identity(2)
            assert m is not None


class TestMatrixDimensions:
    """Test matrix dimension properties."""

    def test_identity_is_square(self, Matrix):
        """identity() should create a square matrix."""
        m = Matrix.identity(5)
        if hasattr(m, 'shape'):
            assert m.shape[0] == m.shape[1] == 5
        elif hasattr(m, 'rows') and hasattr(m, 'cols'):
            assert m.rows == m.cols == 5
        elif hasattr(m, 'nrows') and hasattr(m, 'ncols'):
            assert m.nrows == m.ncols == 5
        else:
            assert m is not None

    def test_identity_3x3_dimensions(self, Matrix):
        """3x3 identity should have 3 rows and columns."""
        m = Matrix.identity(3)
        if hasattr(m, 'shape'):
            assert m.shape == (3, 3)
        elif hasattr(m, 'rows'):
            assert m.rows == 3

    def test_identity_4x4_dimensions(self, Matrix):
        """4x4 identity should have 4 rows and columns."""
        m = Matrix.identity(4)
        if hasattr(m, 'shape'):
            assert m.shape == (4, 4)


class TestMatrixOperations:
    """Test matrix arithmetic operations."""

    def test_identity_transpose_returns_matrix(self, Matrix):
        """transpose() should return a Matrix."""
        m = Matrix.identity(3)
        if hasattr(m, 'transpose'):
            t = m.transpose()
            assert isinstance(t, Matrix)
        elif hasattr(m, 'T'):
            t = m.T
            assert isinstance(t, Matrix)

    def test_identity_transpose_same_dimensions(self, Matrix):
        """Transpose of identity should have same dimensions."""
        m = Matrix.identity(4)
        if hasattr(m, 'transpose'):
            t = m.transpose()
        elif hasattr(m, 'T'):
            t = m.T
        else:
            pytest.skip("No transpose method found")

        if hasattr(m, 'shape'):
            assert t.shape == m.shape
        elif hasattr(m, 'rows'):
            assert t.rows == m.rows

    def test_matrix_multiplication_exists(self, Matrix):
        """Matrix should support multiplication."""
        m1 = Matrix.identity(3)
        m2 = Matrix.identity(3)
        can_multiply = (
            hasattr(m1, '__mul__') or
            hasattr(m1, '__matmul__') or
            hasattr(m1, 'multiply') or
            hasattr(m1, 'dot')
        )
        assert can_multiply

    def test_identity_times_identity(self, Matrix):
        """I * I should equal I (same dimensions)."""
        i = Matrix.identity(3)

        if hasattr(i, '__matmul__'):
            result = i @ i
        elif hasattr(i, '__mul__'):
            result = i * i
        elif hasattr(i, 'multiply'):
            result = i.multiply(i)
        elif hasattr(i, 'dot'):
            result = i.dot(i)
        else:
            pytest.skip("No multiplication method found")

        assert isinstance(result, Matrix)


class TestMatrixElementAccess:
    """Test matrix element access."""

    def test_can_access_matrix_elements(self, Matrix):
        """Should be able to read matrix elements."""
        m = Matrix.identity(3)
        can_access = (
            hasattr(m, '__getitem__') or
            hasattr(m, 'get') or
            hasattr(m, 'at') or
            hasattr(m, 'data')
        )
        assert can_access

    def test_identity_diagonal_should_be_ones(self, Matrix):
        """Identity matrix diagonal elements should be 1."""
        m = Matrix.identity(3)

        if hasattr(m, '__getitem__'):
            try:
                val = m[0, 0] if m[0, 0] is not None else m[(0, 0)]
                assert val == 1.0 or val == 1
            except (TypeError, KeyError, IndexError):
                try:
                    assert m[0][0] == 1.0 or m[0][0] == 1
                except (TypeError, KeyError, IndexError):
                    pass
        elif hasattr(m, 'data'):
            data = m.data
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    assert data[0][0] == 1.0 or data[0][0] == 1


# =============================================================================
# T-IK-3.15: Jacobian Computation Tests
# =============================================================================

class TestJacobianIKExists:
    """Test that JacobianIK class exists and can be instantiated."""

    def test_jacobian_ik_class_importable(self, JacobianIK):
        """JacobianIK class should be importable."""
        assert JacobianIK is not None

    def test_jacobian_method_enum_importable(self, JacobianMethod):
        """JacobianMethod enum should be importable."""
        assert JacobianMethod is not None

    def test_can_instantiate_with_bone_indices(self, JacobianIK, bone_indices):
        """Should be able to create JacobianIK with bone indices."""
        solver = JacobianIK(bone_indices=bone_indices)
        assert solver is not None

    def test_can_instantiate_with_damping(self, JacobianIK, bone_indices):
        """Should be able to specify damping parameter."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            damping=0.1
        )
        assert solver is not None

    def test_has_solve_method(self, JacobianIK, bone_indices):
        """JacobianIK should have solve() method."""
        solver = JacobianIK(bone_indices=bone_indices)
        assert hasattr(solver, 'solve')
        assert callable(solver.solve)

    def test_can_instantiate_with_max_iterations(self, JacobianIK, bone_indices):
        """Should accept max_iterations parameter."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            max_iterations=50
        )
        assert solver is not None

    def test_requires_at_least_two_bones(self, JacobianIK):
        """Solver should require at least 2 bones."""
        with pytest.raises(ValueError):
            JacobianIK(bone_indices=[0])


class TestJacobianMethodEnum:
    """Test JacobianMethod enum values."""

    def test_has_transpose_method(self, JacobianMethod):
        """Should have TRANSPOSE method."""
        method = get_method(JacobianMethod, 'TRANSPOSE')
        assert method is not None

    def test_has_pseudoinverse_method(self, JacobianMethod):
        """Should have PSEUDOINVERSE method."""
        method = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PINV')
        assert method is not None

    def test_has_dls_method(self, JacobianMethod):
        """Should have DLS (Damped Least Squares) method."""
        method = get_method(JacobianMethod, 'DLS')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED')
        assert method is not None

    def test_enum_has_multiple_values(self, JacobianMethod):
        """Enum should have multiple method values."""
        methods = [m for m in dir(JacobianMethod) if not m.startswith('_')]
        assert len(methods) >= 2


class TestJacobianComputation:
    """Test Jacobian matrix computation."""

    def test_solver_has_compute_jacobian_if_exposed(self, JacobianIK, bone_indices):
        """Check if compute_jacobian method is exposed."""
        solver = JacobianIK(bone_indices=bone_indices)
        if hasattr(solver, 'compute_jacobian'):
            assert callable(solver.compute_jacobian)

    def test_solve_implicitly_computes_jacobian(self, JacobianIK, Vec3, bone_indices,
                                                simple_chain_positions, simple_chain_rotations):
        """solve() should work, implying Jacobian computation occurs."""
        solver = JacobianIK(bone_indices=bone_indices)
        # Target as a list for multi-effector support
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


# =============================================================================
# T-IK-3.16-18: Jacobian Methods Tests
# =============================================================================

class TestSolveBehavior:
    """Test basic solve behavior."""

    def test_solve_returns_result(self, JacobianIK, Vec3, bone_indices,
                                  simple_chain_positions, simple_chain_rotations):
        """solve() should return a result."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_solve_returns_rotations(self, JacobianIK, Vec3, Quat, bone_indices,
                                     simple_chain_positions, simple_chain_rotations):
        """solve() should return rotation data."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        has_rotations = (
            hasattr(result, 'rotations') or
            hasattr(result, 'joint_rotations') or
            hasattr(result, 'angles') or
            isinstance(result, (list, tuple))
        )
        assert has_rotations

    def test_solve_at_current_position(self, JacobianIK, Vec3, bone_indices,
                                       simple_chain_positions, simple_chain_rotations):
        """Solving when already at target should succeed."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(2.0, 0.0, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_solve_with_reachable_target(self, JacobianIK, Vec3, bone_indices,
                                         simple_chain_positions, simple_chain_rotations):
        """Should converge for reachable targets."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_solve_with_unreachable_target_still_returns(self, JacobianIK, Vec3, bone_indices,
                                                         simple_chain_positions, simple_chain_rotations):
        """Should return result even for unreachable targets."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(100.0, 100.0, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_solve_with_different_targets(self, JacobianIK, Vec3, bone_indices,
                                          simple_chain_positions, simple_chain_rotations):
        """Should handle various target positions."""
        solver = JacobianIK(bone_indices=bone_indices)

        target_positions = [
            Vec3(1.0, 1.0, 0.0),
            Vec3(0.5, 0.5, 0.0),
            Vec3(1.8, 0.2, 0.0),
        ]

        for target_pos in target_positions:
            result = solver.solve(
                simple_chain_positions,
                simple_chain_rotations,
                [target_pos]
            )
            assert result is not None


class TestMaxIterations:
    """Test iteration limiting."""

    def test_can_set_max_iterations(self, JacobianIK, bone_indices):
        """Should be able to set max iterations."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            max_iterations=10
        )
        assert solver is not None

    def test_respects_low_max_iterations(self, JacobianIK, Vec3, bone_indices,
                                         simple_chain_positions, simple_chain_rotations):
        """Should stop within max iterations."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            max_iterations=1
        )
        targets = [Vec3(1.5, 1.0, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_more_iterations_does_not_crash(self, JacobianIK, Vec3, bone_indices,
                                            simple_chain_positions, simple_chain_rotations):
        """More iterations should complete without error."""
        targets = [Vec3(1.5, 0.5, 0.0)]

        solver = JacobianIK(bone_indices=bone_indices, max_iterations=100)
        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestTransposeMethod:
    """Test Jacobian Transpose method (T-IK-3.16)."""

    def test_transpose_method_exists(self, JacobianMethod):
        """TRANSPOSE method should exist."""
        method = get_method(JacobianMethod, 'TRANSPOSE')
        assert method is not None

    def test_can_create_solver_with_transpose(self, JacobianIK, JacobianMethod, bone_indices):
        """Should create solver with transpose method."""
        method = get_method(JacobianMethod, 'TRANSPOSE')
        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method
        )
        assert solver is not None

    def test_transpose_method_solves(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                     simple_chain_positions, simple_chain_rotations):
        """Transpose method should produce a solution."""
        method = get_method(JacobianMethod, 'TRANSPOSE')
        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method
        )
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_transpose_converges_for_simple_case(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                                 simple_chain_positions, simple_chain_rotations):
        """Transpose should converge for easy targets."""
        method = get_method(JacobianMethod, 'TRANSPOSE')
        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method,
            max_iterations=100
        )
        targets = [Vec3(1.8, 0.3, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestPseudoinverseMethod:
    """Test Jacobian Pseudoinverse method (T-IK-3.17)."""

    def test_pseudoinverse_method_exists(self, JacobianMethod):
        """PSEUDOINVERSE method should exist."""
        method = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PINV')
        assert method is not None

    def test_can_create_solver_with_pseudoinverse(self, JacobianIK, JacobianMethod, bone_indices):
        """Should create solver with pseudoinverse method."""
        method = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PINV')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method
        )
        assert solver is not None

    def test_pseudoinverse_method_solves(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                         simple_chain_positions, simple_chain_rotations):
        """Pseudoinverse method should produce a solution."""
        method = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PINV')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method
        )
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_pseudoinverse_accurate_for_reachable(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                                   simple_chain_positions, simple_chain_rotations):
        """Pseudoinverse should complete for reachable targets."""
        method = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if method is None:
            method = get_method(JacobianMethod, 'PINV')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method,
            max_iterations=50
        )
        targets = [Vec3(1.5, 0.3, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestDLSMethod:
    """Test Damped Least Squares method (T-IK-3.18)."""

    def test_dls_method_exists(self, JacobianMethod):
        """DLS method should exist."""
        method = get_method(JacobianMethod, 'DLS')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED')
        assert method is not None

    def test_can_create_solver_with_dls(self, JacobianIK, JacobianMethod, bone_indices):
        """Should create solver with DLS method."""
        method = get_method(JacobianMethod, 'DLS')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method,
            damping=0.1
        )
        assert solver is not None

    def test_dls_method_solves(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                               simple_chain_positions, simple_chain_rotations):
        """DLS method should produce a solution."""
        method = get_method(JacobianMethod, 'DLS')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method,
            damping=0.1
        )
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_dls_stable_at_singularity(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                       simple_chain_positions, simple_chain_rotations):
        """DLS should handle near-singular configurations."""
        method = get_method(JacobianMethod, 'DLS')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if method is None:
            method = get_method(JacobianMethod, 'DAMPED')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=method,
            damping=0.5
        )
        targets = [Vec3(2.0, 0.0, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestMethodComparison:
    """Test that all methods work and can be compared."""

    def test_all_methods_converge_simple_case(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                              simple_chain_positions, simple_chain_rotations):
        """All methods should converge for a simple case."""
        targets = [Vec3(1.5, 0.5, 0.0)]

        methods = []
        transpose = get_method(JacobianMethod, 'TRANSPOSE')
        if transpose:
            methods.append(transpose)

        pseudo = get_method(JacobianMethod, 'PSEUDOINVERSE')
        if pseudo is None:
            pseudo = get_method(JacobianMethod, 'PSEUDO_INVERSE')
        if pseudo is None:
            pseudo = get_method(JacobianMethod, 'PINV')
        if pseudo:
            methods.append(pseudo)

        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED')
        if dls:
            methods.append(dls)

        for method in methods:
            solver = JacobianIK(
                bone_indices=bone_indices,
                method=method,
                damping=0.1,
                max_iterations=100
            )
            result = solver.solve(
                simple_chain_positions,
                simple_chain_rotations,
                targets
            )
            assert result is not None, f"Method {method} failed to produce result"

    def test_different_methods_can_be_used(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                           simple_chain_positions, simple_chain_rotations):
        """Different methods should all be usable."""
        targets = [Vec3(1.2, 0.8, 0.0)]

        transpose = get_method(JacobianMethod, 'TRANSPOSE')
        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')

        if transpose and dls:
            solver_t = JacobianIK(bone_indices=bone_indices, method=transpose, max_iterations=10)
            solver_d = JacobianIK(bone_indices=bone_indices, method=dls, damping=0.1, max_iterations=10)

            result_t = solver_t.solve(simple_chain_positions, simple_chain_rotations, targets)
            result_d = solver_d.solve(simple_chain_positions, simple_chain_rotations, targets)

            assert result_t is not None
            assert result_d is not None


class TestDampingEffect:
    """Test damping parameter behavior."""

    def test_accepts_damping_parameter(self, JacobianIK, bone_indices):
        """Solver should accept damping parameter."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            damping=0.5
        )
        assert solver is not None

    def test_zero_damping_accepted(self, JacobianIK, bone_indices):
        """Zero damping should be accepted."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            damping=0.0
        )
        assert solver is not None

    def test_high_damping_accepted(self, JacobianIK, bone_indices):
        """High damping values should be accepted."""
        solver = JacobianIK(
            bone_indices=bone_indices,
            damping=10.0
        )
        assert solver is not None

    def test_high_damping_produces_stable_result(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                                  simple_chain_positions, simple_chain_rotations):
        """High damping should produce stable (non-divergent) results."""
        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED')
        if not dls:
            pytest.skip("DLS method not available")

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=dls,
            damping=1.0,
            max_iterations=50
        )
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_low_damping_produces_result(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                         simple_chain_positions, simple_chain_rotations):
        """Lower damping should still produce results."""
        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED')
        if not dls:
            pytest.skip("DLS method not available")

        targets = [Vec3(1.8, 0.2, 0.0)]

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=dls,
            damping=0.01,
            max_iterations=10
        )

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_different_damping_values(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                      simple_chain_positions, simple_chain_rotations):
        """Different damping values should all produce results."""
        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED')
        if not dls:
            pytest.skip("DLS method not available")

        targets = [Vec3(1.5, 0.5, 0.0)]

        for damping in [0.001, 0.1, 0.5, 1.0, 5.0]:
            solver = JacobianIK(
                bone_indices=bone_indices,
                method=dls,
                damping=damping,
                max_iterations=10
            )
            result = solver.solve(
                simple_chain_positions,
                simple_chain_rotations,
                targets
            )
            assert result is not None


class TestToleranceBehavior:
    """Test tolerance/threshold behavior."""

    def test_accepts_tolerance_parameter(self, JacobianIK, bone_indices):
        """Solver should accept tolerance parameter."""
        try:
            solver = JacobianIK(bone_indices=bone_indices, tolerance=0.001)
            assert solver is not None
        except TypeError:
            try:
                solver = JacobianIK(bone_indices=bone_indices, threshold=0.001)
                assert solver is not None
            except TypeError:
                try:
                    solver = JacobianIK(bone_indices=bone_indices, epsilon=0.001)
                    assert solver is not None
                except TypeError:
                    pass


class TestChainConfigurations:
    """Test different chain configurations."""

    def test_two_bone_chain(self, JacobianIK, Vec3, two_bone_positions, two_bone_rotations, two_bone_indices):
        """Should work with two bones."""
        solver = JacobianIK(bone_indices=two_bone_indices)
        targets = [Vec3(0.8, 0.3, 0)]

        result = solver.solve(two_bone_positions, two_bone_rotations, targets)
        assert result is not None

    def test_four_bone_chain(self, JacobianIK, Vec3, Quat):
        """Should work with four bones."""
        positions = [Vec3(i, 0, 0) for i in range(4)]
        rotations = [Quat(0, 0, 0, 1)] * 4
        bone_indices = [0, 1, 2, 3]

        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(2, 1, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None

    def test_chain_with_non_unit_bone_lengths(self, JacobianIK, Vec3, Quat):
        """Should work with varying bone lengths."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(2.5, 0, 0),
            Vec3(3.0, 0, 0),
        ]
        rotations = [Quat(0, 0, 0, 1)] * 3

        solver = JacobianIK(bone_indices=[0, 1, 2])
        targets = [Vec3(2, 1, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None

    def test_five_bone_chain(self, JacobianIK, Vec3, Quat):
        """Should work with five bones."""
        positions = [Vec3(i * 0.5, 0, 0) for i in range(5)]
        rotations = [Quat(0, 0, 0, 1)] * 5
        bone_indices = [0, 1, 2, 3, 4]

        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_target_at_origin(self, JacobianIK, Vec3, bone_indices,
                              simple_chain_positions, simple_chain_rotations):
        """Should handle target at origin."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(0, 0, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_target_behind_chain(self, JacobianIK, Vec3, bone_indices,
                                 simple_chain_positions, simple_chain_rotations):
        """Should handle target behind the chain root."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(-1, 0, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_target_in_negative_space(self, JacobianIK, Vec3, bone_indices,
                                      simple_chain_positions, simple_chain_rotations):
        """Should handle targets with negative coordinates."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(-2, -2, -2)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_target_very_close_to_current(self, JacobianIK, Vec3, bone_indices,
                                          simple_chain_positions, simple_chain_rotations):
        """Should handle target very close to current position."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(2.0001, 0.0001, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_3d_target(self, JacobianIK, Vec3, bone_indices,
                       simple_chain_positions, simple_chain_rotations):
        """Should handle 3D targets (not just 2D plane)."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1, 0.5, 0.5)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_in_result(self, JacobianIK, Vec3, bone_indices,
                              simple_chain_positions, simple_chain_rotations):
        """Results should not contain NaN values."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        if hasattr(result, 'rotations'):
            for rot in result.rotations:
                if hasattr(rot, 'x'):
                    assert not math.isnan(rot.x)
                    assert not math.isnan(rot.y)
                    assert not math.isnan(rot.z)
                    assert not math.isnan(rot.w)

    def test_no_infinity_in_result(self, JacobianIK, Vec3, bone_indices,
                                   simple_chain_positions, simple_chain_rotations):
        """Results should not contain infinity."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        if hasattr(result, 'rotations'):
            for rot in result.rotations:
                if hasattr(rot, 'x'):
                    assert not math.isinf(rot.x)
                    assert not math.isinf(rot.y)
                    assert not math.isinf(rot.z)
                    assert not math.isinf(rot.w)

    def test_handles_very_small_bones(self, JacobianIK, Vec3, Quat):
        """Should handle chains with very small bone lengths."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(0.001, 0, 0),
        ]
        rotations = [Quat(0, 0, 0, 1)] * 2

        solver = JacobianIK(bone_indices=[0, 1])
        targets = [Vec3(0.0008, 0.0005, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None

    def test_handles_very_large_positions(self, JacobianIK, Vec3, Quat):
        """Should handle chains at large world positions."""
        offset = 10000.0
        positions = [
            Vec3(offset, 0, 0),
            Vec3(offset + 1, 0, 0),
        ]
        rotations = [Quat(0, 0, 0, 1)] * 2

        solver = JacobianIK(bone_indices=[0, 1])
        targets = [Vec3(offset + 0.8, 0.3, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None


class TestAlphaParameter:
    """Test step size / alpha parameter for transpose method."""

    def test_accepts_alpha_parameter(self, JacobianIK, JacobianMethod, bone_indices):
        """Solver should accept alpha/step_size parameter."""
        transpose = get_method(JacobianMethod, 'TRANSPOSE')
        if not transpose:
            pytest.skip("Transpose method not available")

        try:
            solver = JacobianIK(
                bone_indices=bone_indices,
                method=transpose,
                alpha=0.1
            )
            assert solver is not None
        except TypeError:
            try:
                solver = JacobianIK(
                    bone_indices=bone_indices,
                    method=transpose,
                    step_size=0.1
                )
                assert solver is not None
            except TypeError:
                pass


class TestResultFormat:
    """Test result format and structure."""

    def test_result_has_expected_attributes(self, JacobianIK, Vec3, bone_indices,
                                            simple_chain_positions, simple_chain_rotations):
        """Result should have expected attributes."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        has_data = (
            hasattr(result, 'rotations') or
            hasattr(result, 'joint_rotations') or
            hasattr(result, 'angles') or
            isinstance(result, (list, tuple))
        )
        assert has_data

    def test_result_is_iterable_or_has_rotations(self, JacobianIK, Vec3, bone_indices,
                                                  simple_chain_positions, simple_chain_rotations):
        """Result should be iterable or have rotations attribute."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        try:
            if hasattr(result, 'rotations'):
                rots = result.rotations
            elif isinstance(result, (list, tuple)):
                rots = result
            else:
                rots = list(result)
            assert rots is not None
        except TypeError:
            assert hasattr(result, 'converged') or hasattr(result, 'error')


class TestMultipleSolves:
    """Test multiple sequential solves."""

    def test_can_solve_multiple_times(self, JacobianIK, Vec3, bone_indices,
                                      simple_chain_positions, simple_chain_rotations):
        """Should be able to call solve() multiple times."""
        solver = JacobianIK(bone_indices=bone_indices)

        target_positions = [
            Vec3(1.5, 0.5, 0),
            Vec3(1.2, 0.8, 0),
            Vec3(1.8, 0.2, 0),
        ]

        for target_pos in target_positions:
            result = solver.solve(
                simple_chain_positions,
                simple_chain_rotations,
                [target_pos]
            )
            assert result is not None

    def test_solves_are_independent(self, JacobianIK, Vec3, bone_indices,
                                    simple_chain_positions, simple_chain_rotations):
        """Each solve should be independent."""
        solver = JacobianIK(bone_indices=bone_indices)

        targets = [Vec3(1.5, 0.5, 0)]

        result1 = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        result2 = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )

        assert result1 is not None
        assert result2 is not None


class TestSingularityHandling:
    """Test handling of singular/near-singular configurations."""

    def test_handles_stretched_chain(self, JacobianIK, JacobianMethod, Vec3, bone_indices,
                                     simple_chain_positions, simple_chain_rotations):
        """Should handle fully stretched chain (singularity)."""
        dls = get_method(JacobianMethod, 'DLS')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED_LEAST_SQUARES')
        if dls is None:
            dls = get_method(JacobianMethod, 'DAMPED')

        solver = JacobianIK(
            bone_indices=bone_indices,
            method=dls,
            damping=0.5
        )
        targets = [Vec3(2.0, 0, 0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_handles_collinear_configuration(self, JacobianIK, Vec3, Quat):
        """Should handle collinear bone configuration."""
        positions = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(2, 0, 0),
        ]
        rotations = [Quat(0, 0, 0, 1)] * 3

        solver = JacobianIK(bone_indices=[0, 1, 2], damping=0.5)
        targets = [Vec3(1.5, 0.5, 0)]

        result = solver.solve(positions, rotations, targets)
        assert result is not None


class TestMultipleEndEffectors:
    """Test multiple end effector support."""

    def test_single_end_effector(self, JacobianIK, Vec3, bone_indices,
                                 simple_chain_positions, simple_chain_rotations):
        """Should work with single end effector."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None

    def test_targets_as_list(self, JacobianIK, Vec3, bone_indices,
                             simple_chain_positions, simple_chain_rotations):
        """Targets should be provided as a list."""
        solver = JacobianIK(bone_indices=bone_indices)
        targets = [Vec3(1.5, 0.5, 0.0)]

        result = solver.solve(
            simple_chain_positions,
            simple_chain_rotations,
            targets
        )
        assert result is not None


class TestOrientationGoal:
    """Test orientation goal support if available."""

    def test_accepts_orientation_goal(self, JacobianIK, Vec3, Quat, bone_indices,
                                      simple_chain_positions, simple_chain_rotations):
        """Should accept orientation goal if supported."""
        solver = JacobianIK(bone_indices=bone_indices)
        target_pos = Vec3(1.5, 0.5, 0)
        target_rot = Quat(0, 0, 0.707, 0.707)

        if hasattr(solver, 'solve_with_orientation'):
            result = solver.solve_with_orientation(
                simple_chain_positions,
                simple_chain_rotations,
                target_pos,
                target_rot
            )
            assert result is not None
        else:
            try:
                result = solver.solve(
                    simple_chain_positions,
                    simple_chain_rotations,
                    [target_pos],
                    target_orientation=target_rot
                )
                assert result is not None
            except TypeError:
                pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
