"""Spherical Harmonics math library for DDGI.

This module provides NumPy implementations of spherical harmonic functions
that match the WGSL `spherical_harmonics.wgsl` shader library exactly.

The implementation uses 3rd-order (L2) real spherical harmonics with
Condon-Shortley phase convention:
    - L0: 1 coefficient (DC/ambient)
    - L1: 3 coefficients (linear/directional)
    - L2: 5 coefficients (quadratic)

Total: 9 coefficients per color channel, 27 for RGB.

Coefficient ordering:
    Index | l,m    | Basis Function
    ------+--------+------------------
      0   | 0, 0   | Y_0^0  = 0.282095
      1   | 1,-1   | Y_1^-1 = 0.488603 * y
      2   | 1, 0   | Y_1^0  = 0.488603 * z
      3   | 1, 1   | Y_1^1  = 0.488603 * x
      4   | 2,-2   | Y_2^-2 = 1.092548 * xy
      5   | 2,-1   | Y_2^-1 = 1.092548 * yz
      6   | 2, 0   | Y_2^0  = 0.315392 * (3z^2-1)
      7   | 2, 1   | Y_2^1  = 1.092548 * xz
      8   | 2, 2   | Y_2^2  = 0.546274 * (x^2-y^2)

References:
    - Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
      Environment Maps", SIGGRAPH 2001
    - Green, "Spherical Harmonic Lighting: The Gritty Details", GDC 2003
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Constants
# ============================================================================

# SH basis normalization constants
SH_Y00: float = 0.28209479177387814  # sqrt(1/(4*PI))
SH_Y1: float = 0.4886025119029199  # sqrt(3/(4*PI))
SH_Y2_NEG2: float = 1.0925484305920792  # sqrt(15/(4*PI))
SH_Y2_NEG1: float = 1.0925484305920792  # sqrt(15/(4*PI))
SH_Y2_0: float = 0.31539156525252005  # sqrt(5/(16*PI))
SH_Y2_POS1: float = 1.0925484305920792  # sqrt(15/(4*PI))
SH_Y2_POS2: float = 0.5462742152960396  # sqrt(15/(16*PI))

# Cosine lobe convolution coefficients (Ramamoorthi & Hanrahan 2001)
SH_A0: float = 1.0  # PI * (1/PI)
SH_A1: float = 2.0 / 3.0  # 2*PI/3 * (1/PI)
SH_A2: float = 0.25  # PI/4 * (1/PI)


# ============================================================================
# SH Coefficient Container
# ============================================================================


@dataclass
class SHCoefficientsL2:
    """Container for L2 (9-coefficient) spherical harmonic RGB data.

    Attributes:
        coeffs: Shape (9, 3) array of RGB coefficients.
    """

    coeffs: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((9, 3), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=np.float32)
        if self.coeffs.shape != (9, 3):
            raise ValueError(f"Expected shape (9, 3), got {self.coeffs.shape}")

    @classmethod
    def zero(cls) -> SHCoefficientsL2:
        """Create zero-initialized coefficients."""
        return cls(np.zeros((9, 3), dtype=np.float32))

    def get(self, index: int) -> NDArray[np.float32]:
        """Get RGB coefficient at index (0-8)."""
        return self.coeffs[index].copy()

    def set(self, index: int, rgb: NDArray[np.float32] | list[float]) -> None:
        """Set RGB coefficient at index (0-8)."""
        self.coeffs[index] = rgb

    def scale(self, factor: float) -> None:
        """Scale all coefficients in place."""
        self.coeffs *= factor

    def add(self, other: SHCoefficientsL2) -> None:
        """Add another coefficient set in place."""
        self.coeffs += other.coeffs

    def lerp(self, other: SHCoefficientsL2, t: float) -> SHCoefficientsL2:
        """Linear interpolation with another coefficient set."""
        return SHCoefficientsL2((1 - t) * self.coeffs + t * other.coeffs)

    def to_bytes(self) -> bytes:
        """Convert to bytes for GPU upload (144 bytes with padding)."""
        # Add w=0 padding for vec4 alignment
        padded = np.zeros((9, 4), dtype=np.float32)
        padded[:, :3] = self.coeffs
        return padded.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> SHCoefficientsL2:
        """Create from GPU buffer bytes (144 bytes with padding)."""
        padded = np.frombuffer(data, dtype=np.float32).reshape(9, 4)
        return cls(padded[:, :3].copy())


# ============================================================================
# Basis Function Evaluation
# ============================================================================


def sh_basis_l2(direction: NDArray[np.float32]) -> NDArray[np.float32]:
    """Evaluate all 9 SH basis functions at a direction.

    Args:
        direction: Normalized direction vector, shape (3,).

    Returns:
        Array of 9 basis values, shape (9,).
    """
    x, y, z = direction

    return np.array(
        [
            SH_Y00,  # Y_0^0
            SH_Y1 * y,  # Y_1^-1
            SH_Y1 * z,  # Y_1^0
            SH_Y1 * x,  # Y_1^1
            SH_Y2_NEG2 * x * y,  # Y_2^-2
            SH_Y2_NEG1 * y * z,  # Y_2^-1
            SH_Y2_0 * (3 * z * z - 1),  # Y_2^0
            SH_Y2_POS1 * x * z,  # Y_2^1
            SH_Y2_POS2 * (x * x - y * y),  # Y_2^2
        ],
        dtype=np.float32,
    )


def sh_basis_l2_batch(directions: NDArray[np.float32]) -> NDArray[np.float32]:
    """Evaluate all 9 SH basis functions at multiple directions.

    Args:
        directions: Normalized direction vectors, shape (N, 3).

    Returns:
        Array of basis values, shape (N, 9).
    """
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]

    return np.column_stack(
        [
            np.full_like(x, SH_Y00),  # Y_0^0
            SH_Y1 * y,  # Y_1^-1
            SH_Y1 * z,  # Y_1^0
            SH_Y1 * x,  # Y_1^1
            SH_Y2_NEG2 * x * y,  # Y_2^-2
            SH_Y2_NEG1 * y * z,  # Y_2^-1
            SH_Y2_0 * (3 * z * z - 1),  # Y_2^0
            SH_Y2_POS1 * x * z,  # Y_2^1
            SH_Y2_POS2 * (x * x - y * y),  # Y_2^2
        ]
    ).astype(np.float32)


# ============================================================================
# SH Evaluation (Coefficients -> Color)
# ============================================================================


def sh_evaluate_l2(
    coeffs: SHCoefficientsL2 | NDArray[np.float32], direction: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Evaluate SH at a direction using L2 coefficients.

    Args:
        coeffs: SH coefficients (9, 3) or SHCoefficientsL2.
        direction: Normalized direction vector, shape (3,).

    Returns:
        RGB color at the direction, shape (3,).
    """
    if isinstance(coeffs, SHCoefficientsL2):
        c = coeffs.coeffs
    else:
        c = coeffs

    basis = sh_basis_l2(direction)
    # Sum over all basis functions: result[channel] = sum(coeffs[i, channel] * basis[i])
    return np.einsum("i,ic->c", basis, c)


def sh_evaluate_l2_batch(
    coeffs: SHCoefficientsL2 | NDArray[np.float32], directions: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Evaluate SH at multiple directions.

    Args:
        coeffs: SH coefficients (9, 3) or SHCoefficientsL2.
        directions: Normalized directions, shape (N, 3).

    Returns:
        RGB colors at directions, shape (N, 3).
    """
    if isinstance(coeffs, SHCoefficientsL2):
        c = coeffs.coeffs
    else:
        c = coeffs

    basis = sh_basis_l2_batch(directions)  # (N, 9)
    # result[n, c] = sum_i(basis[n, i] * coeffs[i, c])
    return np.einsum("ni,ic->nc", basis, c)


# ============================================================================
# SH Projection (Color -> Coefficients)
# ============================================================================


def sh_project_l2(
    direction: NDArray[np.float32], color: NDArray[np.float32]
) -> SHCoefficientsL2:
    """Project a color sample at a direction into SH coefficients.

    For Monte Carlo integration, multiply the accumulated result by 4*PI/N.

    Args:
        direction: Normalized direction vector, shape (3,).
        color: RGB color at the direction, shape (3,).

    Returns:
        SH coefficients containing the projected sample.
    """
    basis = sh_basis_l2(direction)  # (9,)
    # coeffs[i, c] = basis[i] * color[c]
    coeffs = np.outer(basis, color).astype(np.float32)
    return SHCoefficientsL2(coeffs)


def sh_project_function(
    sample_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    num_samples: int = 10000,
) -> SHCoefficientsL2:
    """Project a function over the sphere into SH coefficients via Monte Carlo.

    Args:
        sample_fn: Function taking direction (3,) and returning RGB color (3,).
        num_samples: Number of Monte Carlo samples.

    Returns:
        SH coefficients representing the function.
    """
    directions = fibonacci_sphere_directions(num_samples)
    coeffs = SHCoefficientsL2.zero()

    for direction in directions:
        color = sample_fn(direction)
        projected = sh_project_l2(direction, color)
        coeffs.add(projected)

    # Scale by solid angle of sphere / number of samples
    coeffs.scale(4 * np.pi / num_samples)
    return coeffs


def sh_project_environment_map(
    env_map: NDArray[np.float32], num_samples: int = 10000
) -> SHCoefficientsL2:
    """Project an equirectangular environment map into SH coefficients.

    Args:
        env_map: Environment map, shape (H, W, 3).
        num_samples: Number of Monte Carlo samples.

    Returns:
        SH coefficients representing the environment.
    """
    h, w = env_map.shape[:2]

    def sample_fn(direction: NDArray[np.float32]) -> NDArray[np.float32]:
        # Convert direction to equirectangular coordinates
        x, y, z = direction
        theta = np.arctan2(x, z)  # azimuth
        phi = np.arcsin(np.clip(y, -1, 1))  # elevation

        # Map to texture coordinates
        u = (theta / np.pi + 1) * 0.5  # [0, 1]
        v = (phi / (np.pi / 2) + 1) * 0.5  # [0, 1]

        # Sample with bilinear interpolation
        px = u * (w - 1)
        py = (1 - v) * (h - 1)

        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = px - x0, py - y0

        c00 = env_map[y0, x0]
        c10 = env_map[y0, x1]
        c01 = env_map[y1, x0]
        c11 = env_map[y1, x1]

        return (
            c00 * (1 - fx) * (1 - fy)
            + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy
            + c11 * fx * fy
        ).astype(np.float32)

    return sh_project_function(sample_fn, num_samples)


# ============================================================================
# Irradiance Convolution
# ============================================================================


def sh_convolve_irradiance(
    coeffs: SHCoefficientsL2 | NDArray[np.float32],
) -> SHCoefficientsL2:
    """Convolve SH coefficients with cosine lobe for irradiance.

    Applies the A_l factors from Ramamoorthi & Hanrahan 2001:
        A_0 = 1.0
        A_1 = 2/3
        A_2 = 1/4

    Args:
        coeffs: Radiance SH coefficients.

    Returns:
        Irradiance SH coefficients.
    """
    if isinstance(coeffs, SHCoefficientsL2):
        c = coeffs.coeffs.copy()
    else:
        c = coeffs.copy()

    result = np.zeros_like(c)

    # L0 band
    result[0] = c[0] * SH_A0

    # L1 band
    result[1:4] = c[1:4] * SH_A1

    # L2 band
    result[4:9] = c[4:9] * SH_A2

    return SHCoefficientsL2(result)


# ============================================================================
# SH Rotation
# ============================================================================


def sh_rotate_l2(
    coeffs: SHCoefficientsL2 | NDArray[np.float32], rotation: NDArray[np.float32]
) -> SHCoefficientsL2:
    """Rotate L2 SH coefficients by a 3x3 rotation matrix.

    L0 is rotationally invariant.
    L1 transforms as a vector.
    L2 transforms via a 5x5 matrix derived from the rotation.

    Args:
        coeffs: SH coefficients to rotate.
        rotation: 3x3 rotation matrix (row-major).

    Returns:
        Rotated SH coefficients.
    """
    if isinstance(coeffs, SHCoefficientsL2):
        c = coeffs.coeffs
    else:
        c = coeffs

    result = np.zeros_like(c)

    # L0: rotationally invariant
    result[0] = c[0]

    # L1: transforms as a vector
    # Coefficient mapping: c1=Y_1^-1=y, c2=Y_1^0=z, c3=Y_1^1=x
    for ch in range(3):
        l1_vec = np.array([c[3, ch], c[1, ch], c[2, ch]])  # [x, y, z] components
        rotated = rotation @ l1_vec
        result[1, ch] = rotated[1]  # Y_1^-1 = y
        result[2, ch] = rotated[2]  # Y_1^0 = z
        result[3, ch] = rotated[0]  # Y_1^1 = x

    # L2: 5x5 rotation matrix
    m = _compute_l2_rotation_matrix(rotation)

    for ch in range(3):
        l2_in = c[4:9, ch]
        result[4:9, ch] = m @ l2_in

    return SHCoefficientsL2(result)


def _compute_l2_rotation_matrix(r: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute the 5x5 rotation matrix for L=2 band.

    Uses the analytical formulas from Green 2003 and Sloan 2008.
    These ensure identity rotation maps to identity matrix.

    Args:
        r: 3x3 rotation matrix.

    Returns:
        5x5 L2 rotation matrix.
    """
    r00, r01, r02 = r[0]
    r10, r11, r12 = r[1]
    r20, r21, r22 = r[2]

    # Helper for accessing rotation matrix elements by index
    def s(a: int, b: int) -> float:
        return r[a, b]

    sqrt3 = np.sqrt(3.0)

    m = np.array(
        [
            # Row 0: Y_2^-2 (xy term) - how original coeffs contribute
            [
                s(0, 0) * s(1, 1) + s(0, 1) * s(1, 0),  # -2 -> -2
                s(0, 1) * s(1, 2) + s(0, 2) * s(1, 1),  # -1 -> -2
                s(0, 2) * s(1, 2) * sqrt3,              # 0 -> -2
                s(0, 0) * s(1, 2) + s(0, 2) * s(1, 0),  # 1 -> -2
                s(0, 0) * s(1, 0) - s(0, 1) * s(1, 1),  # 2 -> -2
            ],
            # Row 1: Y_2^-1 (yz term)
            [
                s(1, 0) * s(2, 1) + s(1, 1) * s(2, 0),
                s(1, 1) * s(2, 2) + s(1, 2) * s(2, 1),
                s(1, 2) * s(2, 2) * sqrt3,
                s(1, 0) * s(2, 2) + s(1, 2) * s(2, 0),
                s(1, 0) * s(2, 0) - s(1, 1) * s(2, 1),
            ],
            # Row 2: Y_2^0 (3z^2-1 term) - diagonal should be 1 for identity
            [
                s(2, 0) * s(2, 1) * 2.0 / sqrt3,
                s(2, 1) * s(2, 2) * 2.0 / sqrt3,
                1.5 * s(2, 2) * s(2, 2) - 0.5,  # This gives 1 when r22=1
                s(2, 0) * s(2, 2) * 2.0 / sqrt3,
                (s(2, 0) * s(2, 0) - s(2, 1) * s(2, 1)) / sqrt3,
            ],
            # Row 3: Y_2^1 (xz term)
            [
                s(0, 0) * s(2, 1) + s(0, 1) * s(2, 0),
                s(0, 1) * s(2, 2) + s(0, 2) * s(2, 1),
                s(0, 2) * s(2, 2) * sqrt3,
                s(0, 0) * s(2, 2) + s(0, 2) * s(2, 0),
                s(0, 0) * s(2, 0) - s(0, 1) * s(2, 1),
            ],
            # Row 4: Y_2^2 (x^2-y^2 term)
            [
                s(0, 0) * s(0, 1) - s(1, 0) * s(1, 1),
                s(0, 1) * s(0, 2) - s(1, 1) * s(1, 2),
                (s(0, 2) * s(0, 2) - s(1, 2) * s(1, 2)) / sqrt3,
                s(0, 0) * s(0, 2) - s(1, 0) * s(1, 2),
                0.5 * (s(0, 0) * s(0, 0) - s(0, 1) * s(0, 1) - s(1, 0) * s(1, 0) + s(1, 1) * s(1, 1)),
            ],
        ],
        dtype=np.float32,
    )

    return m


# ============================================================================
# Utility Functions
# ============================================================================


def fibonacci_sphere_directions(n: int) -> NDArray[np.float32]:
    """Generate approximately uniform directions on sphere using Fibonacci lattice.

    Args:
        n: Number of directions to generate.

    Returns:
        Array of normalized directions, shape (n, 3).
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)

    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(np.clip(1 - 2 * (indices + 0.5) / n, -1, 1))

    directions = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    ).astype(np.float32)

    return directions


def sh_energy(coeffs: SHCoefficientsL2 | NDArray[np.float32]) -> float:
    """Compute approximate energy of SH coefficients (sum of squared magnitudes).

    Args:
        coeffs: SH coefficients.

    Returns:
        Energy (L2 norm squared).
    """
    if isinstance(coeffs, SHCoefficientsL2):
        c = coeffs.coeffs
    else:
        c = coeffs

    return float(np.sum(c * c))


# ============================================================================
# Test Data Generation
# ============================================================================


def generate_test_data_constant(
    color: tuple[float, float, float] = (0.5, 0.3, 0.8),
) -> dict:
    """Generate test data for constant color projection.

    Args:
        color: RGB color to project.

    Returns:
        Dictionary with coefficients and expected evaluation results.
    """
    coeffs = sh_project_function(lambda _: np.array(color, dtype=np.float32), 10000)

    # Test directions
    test_dirs = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        dtype=np.float32,
    )

    results = sh_evaluate_l2_batch(coeffs, test_dirs)

    return {
        "input_color": color,
        "coeffs": coeffs.coeffs.tolist(),
        "test_directions": test_dirs.tolist(),
        "expected_results": results.tolist(),
        "max_error": float(np.max(np.abs(results - np.array(color)))),
    }


def generate_test_data_gradient() -> dict:
    """Generate test data for z-gradient function.

    Returns:
        Dictionary with coefficients and expected evaluation results.
    """

    def gradient_fn(d: NDArray[np.float32]) -> NDArray[np.float32]:
        # Linear gradient along z: [0.5 + 0.5*z, 0.5 + 0.5*z, 0.5 + 0.5*z]
        return np.full(3, 0.5 + 0.5 * d[2], dtype=np.float32)

    coeffs = sh_project_function(gradient_fn, 10000)

    # Test along z-axis
    test_dirs = np.array(
        [[0, 0, 1], [0, 0, -1], [0, 0, 0.5], [1, 0, 0]], dtype=np.float32
    )
    # Normalize
    test_dirs = test_dirs / np.linalg.norm(test_dirs, axis=1, keepdims=True)

    results = sh_evaluate_l2_batch(coeffs, test_dirs)

    # Expected values
    expected = np.array([[0.5 + 0.5 * d[2]] * 3 for d in test_dirs], dtype=np.float32)

    return {
        "coeffs": coeffs.coeffs.tolist(),
        "test_directions": test_dirs.tolist(),
        "expected_results": expected.tolist(),
        "actual_results": results.tolist(),
        "max_error": float(np.max(np.abs(results - expected))),
    }


def generate_test_data_rotation() -> dict:
    """Generate test data for rotation validation.

    Returns:
        Dictionary with coefficients before/after rotation and test results.
    """
    # Create directional light from +X
    light_dir = np.array([1, 0, 0], dtype=np.float32)
    light_color = np.array([1, 1, 1], dtype=np.float32)

    # Project a directional function peaked at +X
    def directional_fn(d: NDArray[np.float32]) -> NDArray[np.float32]:
        dot = max(0, np.dot(d, light_dir))
        return (dot**4 * light_color).astype(np.float32)

    original = sh_project_function(directional_fn, 10000)

    # 90-degree rotation around Z (X -> Y)
    angle = np.pi / 2
    rot_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    rotated = sh_rotate_l2(original, rot_z)

    # Evaluate original at +X and rotated at +Y (should match)
    eval_original = sh_evaluate_l2(original, np.array([1, 0, 0], dtype=np.float32))
    eval_rotated = sh_evaluate_l2(rotated, np.array([0, 1, 0], dtype=np.float32))

    return {
        "original_coeffs": original.coeffs.tolist(),
        "rotation_matrix": rot_z.tolist(),
        "rotated_coeffs": rotated.coeffs.tolist(),
        "eval_original_at_x": eval_original.tolist(),
        "eval_rotated_at_y": eval_rotated.tolist(),
        "match_error": float(np.max(np.abs(eval_original - eval_rotated))),
    }


def generate_ramamoorthi_reference() -> dict:
    """Generate reference data matching Ramamoorthi & Hanrahan 2001 formulas.

    This validates the irradiance convolution against known analytical results.

    Returns:
        Dictionary with reference values and computed results.
    """
    # Reference: for a purely L0 (constant) environment, irradiance = radiance
    l0_only = SHCoefficientsL2.zero()
    l0_only.set(0, np.array([1, 1, 1], dtype=np.float32))

    l0_irradiance = sh_convolve_irradiance(l0_only)

    # Reference: L1 coefficient should scale by 2/3
    l1_only = SHCoefficientsL2.zero()
    l1_only.set(2, np.array([1, 1, 1], dtype=np.float32))  # Z-direction

    l1_irradiance = sh_convolve_irradiance(l1_only)

    # Reference: L2 coefficient should scale by 1/4
    l2_only = SHCoefficientsL2.zero()
    l2_only.set(6, np.array([1, 1, 1], dtype=np.float32))  # 3z^2-1 term

    l2_irradiance = sh_convolve_irradiance(l2_only)

    return {
        "l0_input": l0_only.coeffs.tolist(),
        "l0_output": l0_irradiance.coeffs.tolist(),
        "l0_scale_factor": float(l0_irradiance.get(0)[0] / l0_only.get(0)[0]),
        "expected_l0_scale": SH_A0,
        "l1_input": l1_only.coeffs.tolist(),
        "l1_output": l1_irradiance.coeffs.tolist(),
        "l1_scale_factor": float(l1_irradiance.get(2)[0] / l1_only.get(2)[0]),
        "expected_l1_scale": SH_A1,
        "l2_input": l2_only.coeffs.tolist(),
        "l2_output": l2_irradiance.coeffs.tolist(),
        "l2_scale_factor": float(l2_irradiance.get(6)[0] / l2_only.get(6)[0]),
        "expected_l2_scale": SH_A2,
    }


# ============================================================================
# Validation Functions
# ============================================================================


def validate_roundtrip_error(num_samples: int = 10000, num_test_dirs: int = 1000) -> float:
    """Validate projection + evaluation roundtrip error.

    Projects a smooth function, evaluates at test directions, and computes
    maximum error.

    Args:
        num_samples: Number of projection samples.
        num_test_dirs: Number of test directions.

    Returns:
        Maximum absolute error.
    """

    def test_fn(d: NDArray[np.float32]) -> NDArray[np.float32]:
        # Linear z-gradient
        return np.full(3, 0.5 + 0.5 * d[2], dtype=np.float32)

    coeffs = sh_project_function(test_fn, num_samples)
    test_dirs = fibonacci_sphere_directions(num_test_dirs)

    max_error = 0.0
    for d in test_dirs:
        expected = test_fn(d)
        result = sh_evaluate_l2(coeffs, d)
        error = np.max(np.abs(expected - result))
        max_error = max(max_error, error)

    return max_error


def validate_orthonormality(num_samples: int = 10000) -> dict:
    """Validate approximate orthonormality of SH basis functions.

    Args:
        num_samples: Number of integration samples.

    Returns:
        Dictionary with inner products between basis functions.
    """
    directions = fibonacci_sphere_directions(num_samples)
    basis_values = sh_basis_l2_batch(directions)  # (N, 9)

    scale = 4 * np.pi / num_samples

    # Compute approximate inner products
    inner_products = np.zeros((9, 9), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            inner_products[i, j] = scale * np.sum(basis_values[:, i] * basis_values[:, j])

    # Should be approximately identity matrix
    identity_error = np.max(np.abs(inner_products - np.eye(9)))

    return {
        "inner_products": inner_products.tolist(),
        "identity_error": float(identity_error),
        "is_orthonormal": identity_error < 0.1,
    }
