"""
Test base class (T-1.1) providing assertion helpers for physics tests.

Provides:
  - assertAlmostEqualVec3   -- compare 3D vectors with tolerance
  - assertAlmostEqualMat3   -- compare 3x3 matrices with tolerance
  - assertAlmostEqualQuat   -- compare quaternions with sign-ambiguity handling
  - assertPositiveDefinite  -- verify a 3x3 matrix is positive definite
"""

import math


class PhysicsTestCase:
    """Mixin base class providing assertion helpers for physics simulation tests."""

    # ------------------------------------------------------------------
    # Vec3 helpers
    # ------------------------------------------------------------------
    def assertAlmostEqualVec3(self, actual, expected, places=6, msg=""):
        """Assert that *actual* (Vec3 dataclass / tuple / list) equals *expected* (tuple)."""
        tol = 10.0 ** (-places)
        if hasattr(actual, "x"):                     # Vec3 dataclass (jacobian.py)
            items = (actual.x, actual.y, actual.z)
        elif isinstance(actual, (tuple, list)) and len(actual) == 3:
            items = actual
        else:
            raise TypeError(f"Cannot compare type {type(actual)} as Vec3")

        for i, label in enumerate(("x", "y", "z")):
            diff = abs(items[i] - expected[i])
            if diff > tol:
                raise AssertionError(
                    f"{msg}  Vec3.{label} mismatch: got {items[i]}, "
                    f"expected {expected[i]}, diff {diff:.2e}"
                )

    # ------------------------------------------------------------------
    # Mat3 helpers
    # ------------------------------------------------------------------
    def assertAlmostEqualMat3(self, actual, expected, places=6, msg=""):
        """Assert that a Mat3 dataclass equals a 3x3 nested-tuple."""
        tol = 10.0 ** (-places)
        for i in range(3):
            for j in range(3):
                attr = f"m{i}{j}"
                v = getattr(actual, attr, None)
                if v is None:
                    raise TypeError(f"Mat3 has no attribute {attr}")
                diff = abs(v - expected[i][j])
                if diff > tol:
                    raise AssertionError(
                        f"{msg}  Mat3.{attr} mismatch: got {v}, "
                        f"expected {expected[i][j]}, diff {diff:.2e}"
                    )

    # ------------------------------------------------------------------
    # Quaternion helpers
    # ------------------------------------------------------------------
    def assertAlmostEqualQuat(self, actual, expected, places=6, msg=""):
        """Assert that a Quaternion equals (x,y,z,w) allowing sign ambiguity.

        q and -q represent the same rotation, so both are accepted.
        """
        tol = 10.0 ** (-places)
        if not hasattr(actual, "x"):
            raise TypeError(f"Cannot compare type {type(actual)} as Quaternion")

        vals = (actual.x, actual.y, actual.z, actual.w)
        pos_match = all(abs(v - e) <= tol for v, e in zip(vals, expected))
        neg_match = all(abs(v + e) <= tol for v, e in zip(vals, expected))

        if not pos_match and not neg_match:
            raise AssertionError(
                f"{msg}  Quaternion mismatch: got ({actual.x},{actual.y},{actual.z},{actual.w}), "
                f"expected (+/-){expected}"
            )

    # ------------------------------------------------------------------
    # Positive definiteness
    # ------------------------------------------------------------------
    def assertPositiveDefinite(self, mat3, msg=""):
        """Assert that a Mat3 is symmetric-positive-definite.

        Checks:
          1. Symmetry  (m01==m10, m02==m20, m12==m21)
          2. Leading principal minors > 0
        """
        m = mat3
        prefix = f"{msg}  " if msg else ""

        # Symmetry
        assert abs(m.m01 - m.m10) < 1e-12, f"{prefix}Mat3 not symmetric at m01/m10"
        assert abs(m.m02 - m.m20) < 1e-12, f"{prefix}Mat3 not symmetric at m02/m20"
        assert abs(m.m12 - m.m21) < 1e-12, f"{prefix}Mat3 not symmetric at m12/m21"

        # Leading principal minors
        det1 = m.m00
        assert det1 > 1e-12, f"{prefix}Leading principal minor 1 (m00) = {det1} <= 0"

        det2 = m.m00 * m.m11 - m.m01 * m.m10
        assert det2 > 1e-12, f"{prefix}Leading principal minor 2 = {det2} <= 0"

        det3 = m.determinant()
        assert det3 > 1e-12, f"{prefix}Determinant (minor 3) = {det3} <= 0"
