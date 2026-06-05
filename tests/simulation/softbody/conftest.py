"""Pytest configuration and shared fixtures for softbody tests."""

import pytest
import numpy as np


@pytest.fixture
def simple_tetrahedron():
    """Simple unit tetrahedron with known volume 1/6."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return vertices, indices


@pytest.fixture
def unit_cube():
    """Unit cube vertices (8 corners)."""
    return np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)


@pytest.fixture
def cube_tetrahedralization():
    """Standard 5-tet decomposition of a unit cube."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    tetrahedra = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int32)

    return vertices, tetrahedra


@pytest.fixture
def rotation_90z():
    """90 degree rotation around z-axis."""
    theta = np.pi / 2
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float64)


@pytest.fixture
def uniform_masses_8():
    """Uniform masses for 8 vertices."""
    return np.ones(8, dtype=np.float64)


@pytest.fixture
def uniform_masses_4():
    """Uniform masses for 4 vertices."""
    return np.ones(4, dtype=np.float64)


def assert_vectors_close(actual, expected, atol=1e-6, rtol=1e-5):
    """Assert two vectors are close with better error messages."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = actual - expected
        max_diff = np.max(np.abs(diff))
        raise AssertionError(
            f"Vectors not close.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Max diff: {max_diff}"
        )


def assert_matrix_orthogonal(matrix, atol=1e-6):
    """Assert a matrix is orthogonal (R^T * R = I)."""
    result = matrix.T @ matrix
    identity = np.eye(matrix.shape[0])

    if not np.allclose(result, identity, atol=atol):
        raise AssertionError(
            f"Matrix is not orthogonal.\n"
            f"  R^T * R =\n{result}\n"
            f"  Expected identity."
        )


def assert_positive_definite(matrix, atol=1e-10):
    """Assert a matrix is positive definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)

    if not np.all(eigenvalues > -atol):
        raise AssertionError(
            f"Matrix is not positive definite.\n"
            f"  Eigenvalues: {eigenvalues}"
        )
