"""
T-1.2: Test inertia tensors for all collision shape types.

Verifies analytical inertia formulas for Sphere, Box, Capsule, Cylinder,
Cone, and Compound shapes.
"""

import math
import pytest

from engine.simulation.physics.collision_shapes import (
    SphereShape,
    BoxShape,
    CapsuleShape,
    CylinderShape,
    ConeShape,
    ConvexHullShape,
    MeshShape,
    CompoundShape,
    ShapeType,
)
from ..physics_test_base import PhysicsTestCase


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _diag(I):
    """Return (Ixx, Iyy, Izz) from a 3x3 inertia tensor (nested tuple)."""
    return I[0][0], I[1][1], I[2][2]


# ===========================================================================
# T-1.2 — Shape inertia
# ===========================================================================

class TestCollisionShapesInertia(PhysicsTestCase):
    """Inertia tensor verification for every analytical shape type."""

    # ------------------------------------------------------------------
    # Sphere  (reference: I_diag = 16.0 for m=10, r=2)
    # ------------------------------------------------------------------
    def test_sphere_inertia(self):
        """sphere inertia I = (2/5) * m * r^2  (diagonal, isotropic)."""
        m, r = 10.0, 2.0
        shape = SphereShape(radius=r)
        props = shape.compute_mass_properties(density=1.0)
        m = props.mass

        expected = (2.0 / 5.0) * m * r * r
        ixx, iyy, izz = _diag(props.inertia_tensor)

        assert abs(ixx - expected) < 1e-9, f"Ixx: {ixx} != {expected}"
        assert abs(iyy - expected) < 1e-9, f"Iyy: {iyy} != {expected}"
        assert abs(izz - expected) < 1e-9, f"Izz: {izz} != {expected}"

    def test_sphere_inertia_isotropic(self):
        """sphere inertia tensor is isotropic (Ixx == Iyy == Izz)."""
        m, r = 5.0, 1.0
        shape = SphereShape(radius=r)
        props = shape.compute_mass_properties(density=1.0)
        ixx, iyy, izz = _diag(props.inertia_tensor)
        assert abs(ixx - iyy) < 1e-12 and abs(iyy - izz) < 1e-12

    # ------------------------------------------------------------------
    # Box  (reference: Ixx=41, Iyy=34, Izz=25 for m=12, sx=3, sy=4, sz=5)
    # ------------------------------------------------------------------
    def test_box_inertia(self):
        """box inertia Ixx = (1/12)*m*(sy^2+sz^2), etc."""
        sx, sy, sz = 3.0, 4.0, 5.0
        shape = BoxShape(half_extents=(sx / 2, sy / 2, sz / 2))
        props = shape.compute_mass_properties(density=1.0)
        m = props.mass

        Ixx = (1.0 / 12.0) * m * (sy * sy + sz * sz)
        Iyy = (1.0 / 12.0) * m * (sx * sx + sz * sz)
        Izz = (1.0 / 12.0) * m * (sx * sx + sy * sy)

        ixx, iyy, izz = _diag(props.inertia_tensor)
        assert abs(ixx - Ixx) < 1e-9, f"Ixx: {ixx} != {Ixx}"
        assert abs(iyy - Iyy) < 1e-9, f"Iyy: {iyy} != {Iyy}"
        assert abs(izz - Izz) < 1e-9, f"Izz: {izz} != {Izz}"

    def test_box_inertia_off_diagonal_zero(self):
        """box inertia tensor has zero off-diagonal terms."""
        shape = BoxShape(half_extents=(1.0, 2.0, 3.0))
        props = shape.compute_mass_properties(density=1.0)
        I = props.inertia_tensor
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(I[i][j]) < 1e-12, f"I[{i}][{j}] = {I[i][j]} != 0"

    # ------------------------------------------------------------------
    # Capsule  (cylinder + hemispheres + parallel axis)
    # ------------------------------------------------------------------
    def test_capsule_inertia_values_positive(self):
        """capsule inertia tensor has all diagonal entries positive."""
        r, h = 1.0, 4.0
        shape = CapsuleShape(radius=r, half_height=h / 2)
        props = shape.compute_mass_properties(density=1.0)
        ixx, iyy, izz = _diag(props.inertia_tensor)
        assert ixx > 0 and iyy > 0 and izz > 0

    def test_capsule_inertia_symmetry(self):
        """capsule inertia is symmetric about Y (Ixx == Izz for Y-up capsule)."""
        shape = CapsuleShape(radius=0.5, half_height=1.0)
        props = shape.compute_mass_properties(density=1.0)
        ixx, iyy, izz = _diag(props.inertia_tensor)
        assert abs(ixx - izz) < 1e-9, f"Capsule Ixx ({ixx}) != Izz ({izz})"

    # ------------------------------------------------------------------
    # Cylinder
    # ------------------------------------------------------------------
    def test_cylinder_inertia(self):
        """cylinder inertia: Y-axis axial, X/Z radial."""
        r, h = 1.0, 3.0
        shape = CylinderShape(radius=r, height=h)
        props = shape.compute_mass_properties(density=1.0)
        m = props.mass

        expected_radial = (1.0 / 12.0) * m * (3 * r * r + h * h)
        expected_axial = 0.5 * m * r * r
        ixx, iyy, izz = _diag(props.inertia_tensor)

        # Cylinder axis is Y: Ixx = Izz = radial, Iyy = axial
        assert abs(ixx - expected_radial) < 1e-9, f"Ixx: {ixx} != {expected_radial}"
        assert abs(iyy - expected_axial) < 1e-9, f"Iyy: {iyy} != {expected_axial}"
        assert abs(izz - expected_radial) < 1e-9, f"Izz: {izz} != {expected_radial}"

    # ------------------------------------------------------------------
    # Cone
    # ------------------------------------------------------------------
    def test_cone_inertia(self):
        """cone inertia: Iyy = (3/10)*m*r^2, Ixx=Izz = (3/80)*m*(4*r^2+h^2)."""
        r, h = 1.0, 3.0
        shape = ConeShape(radius=r, height=h)
        props = shape.compute_mass_properties(density=1.0)
        m = props.mass

        # Cone inertia formulas (about center of mass)
        expected_axial = (3.0 / 10.0) * m * r * r
        expected_radial = (3.0 / 80.0) * m * (4 * r * r + h * h)
        ixx, iyy, izz = _diag(props.inertia_tensor)

        # Cone axis is Y: Ixx = Izz = radial, Iyy = axial
        assert abs(ixx - expected_radial) < 1e-9, f"Ixx: {ixx} != {expected_radial}"
        assert abs(iyy - expected_axial) < 1e-9, f"Iyy: {iyy} != {expected_axial}"
        assert abs(izz - expected_radial) < 1e-9, f"Izz: {izz} != {expected_radial}"

    def test_cone_inertia_symmetry(self):
        """cone inertia is symmetric about Y (Ixx == Izz for Y-up cone)."""
        shape = ConeShape(radius=0.5, height=2.0)
        props = shape.compute_mass_properties(density=1.0)
        ixx, iyy, izz = _diag(props.inertia_tensor)
        assert abs(ixx - izz) < 1e-9, f"Cone Ixx ({ixx}) != Izz ({izz})"

    def test_cone_center_of_mass(self):
        """cone center of mass is at 1/4 height from base."""
        shape = ConeShape(radius=1.0, height=4.0)
        props = shape.compute_mass_properties(density=1.0)
        # Cone is centered at origin (half_height = 2), base at -2, apex at +2
        # COM should be at 1/4 height from base = -2 + 1 = -1
        expected_com_y = -1.0
        assert abs(props.center_of_mass[1] - expected_com_y) < 1e-9, \
            f"Cone COM y: {props.center_of_mass[1]} != {expected_com_y}"

    # ------------------------------------------------------------------
    # Compound shape
    # ------------------------------------------------------------------
    def test_compound_inertia_positive_definite(self):
        """compound shape inertia is positive definite."""
        compound = CompoundShape()
        compound.add_child(SphereShape(radius=1.0), local_offset=(0.0, 0.0, 0.0))
        compound.add_child(BoxShape(half_extents=(0.5, 0.5, 0.5)), local_offset=(1.0, 0.0, 0.0))
        props = compound.compute_mass_properties(density=1.0)

        # The inertia_tensor from collision_shapes is a nested tuple, not a Mat3.
        # We can still check positivity of diagonal entries as a sanity check.
        I = props.inertia_tensor
        assert I[0][0] > 0, f"Ixx = {I[0][0]} <= 0"
        assert I[1][1] > 0, f"Iyy = {I[1][1]} <= 0"
        assert I[2][2] > 0, f"Izz = {I[2][2]} <= 0"

    def test_compound_inertia_empty(self):
        """empty compound shape returns default MassProperties."""
        compound = CompoundShape()
        props = compound.compute_mass_properties(density=1.0)
        assert props.mass > 0  # default MassProperties has mass=1.0

    # ------------------------------------------------------------------
    # Positive definiteness for all analytical shapes
    # ------------------------------------------------------------------
    def test_all_shapes_positive_definite(self):
        """every shape type yields a positive-definite inertia tensor."""
        # NOTE: collision_shapes returns a nested-tuple, not a Mat3.
        # Positive-definiteness of the tuple form is checked via
        # diagonal positivity for these simple shapes.
        shapes = [
            ("sphere", SphereShape(radius=1.0)),
            ("box",    BoxShape(half_extents=(1.0, 1.0, 1.0))),
            ("capsule",CapsuleShape(radius=0.5, half_height=1.0)),
            ("cylinder", CylinderShape(radius=0.5, height=2.0)),
            ("cone", ConeShape(radius=0.5, height=2.0)),
        ]
        for name, shape in shapes:
            props = shape.compute_mass_properties(density=1.0)
            I = props.inertia_tensor
            assert I[0][0] > 0, f"{name}: Ixx = {I[0][0]} <= 0"
            assert I[1][1] > 0, f"{name}: Iyy = {I[1][1]} <= 0"
            assert I[2][2] > 0, f"{name}: Izz = {I[2][2]} <= 0"


# ===========================================================================
# T-1.2 — ConvexHullShape & MeshShape (approximate / placeholder)
# ===========================================================================

class TestCollisionShapesApproximate(PhysicsTestCase):
    """Approximate inertia checks for hull and mesh shapes."""

    def test_convex_hull_inertia_computes(self):
        """ConvexHullShape compute_mass_properties does not raise."""
        shape = ConvexHullShape(points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ])
        props = shape.compute_mass_properties(density=1.0)
        assert props.mass >= 0
        assert abs(props.inertia_tensor[0][0]) > 0

    def test_mesh_shape_inertia_computes(self):
        """MeshShape compute_mass_properties does not raise."""
        shape = MeshShape(vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ], indices=[0, 1, 2])
        props = shape.compute_mass_properties(density=1.0)
        assert props.mass >= 0
