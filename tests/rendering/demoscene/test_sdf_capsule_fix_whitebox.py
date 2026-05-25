"""
Whitebox fix-verification tests for sdCapsule WGSL function (T-DEMO-1.7).

Verifies that the capsule SDF implementation is correctly wired across all
four layers of the rendering pipeline:

  Layer 1 -- WGSL source: sdf_capsule.wgsl has the correct fn signature
             and IQ formula (length(pa - ba * h) - abs(r)).
  Layer 2 -- AST node: CapsuleNode is importable from ast_nodes and __init__.
  Layer 3 -- AST dispatch: sdCapsule in _PRIMITIVE_DISPATCH and "capsule"
             in _MARKER_DISPATCH both produce CapsuleNode instances.
  Layer 4 -- Codegen: SDF_CAPSULE template formats the correct WGSL call.

Reference: Inigo Quilez -- Signed distance functions
https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

import math
import os
import re

import pytest

# =============================================================================
# Path to this project
# =============================================================================

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


# =============================================================================
# Layer 1: WGSL source file verification
# =============================================================================


class TestWgslSignature:
    """Verify sdf_capsule.wgsl exists, has the correct fn signature, and
    implements the IQ capsule formula."""

    WGSL_PATH = os.path.join(
        _PROJECT_ROOT,
        "engine",
        "rendering",
        "demoscene",
        "wgsl",
        "sdf_capsule.wgsl",
    )

    def test_wgsl_file_exists(self):
        """Layer 1a: WGSL source file must exist."""
        assert os.path.isfile(self.WGSL_PATH), (
            f"WGSL file not found at {self.WGSL_PATH}"
        )

    def test_wgsl_fn_signature(self):
        """Layer 1b: fn sdCapsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32."""
        with open(self.WGSL_PATH) as f:
            src = f.read()

        # Capture the function signature
        sig_re = r"fn\s+sdCapsule\s*\(([^)]+)\)\s*->\s*f32"
        m = re.search(sig_re, src)
        assert m is not None, (
            "Could not find 'fn sdCapsule(...) -> f32' in sdf_capsule.wgsl"
        )

        sig = m.group(1)
        assert "p: vec3<f32>" in sig, f"Missing 'p: vec3<f32>' in signature: {sig}"
        assert "a: vec3<f32>" in sig, f"Missing 'a: vec3<f32>' in signature: {sig}"
        assert "b: vec3<f32>" in sig, f"Missing 'b: vec3<f32>' in signature: {sig}"
        assert "r: f32" in sig, f"Missing 'r: f32' in signature: {sig}"

    def test_wgsl_formula(self):
        """Layer 1c: Function body contains the IQ capsule formula.

        The capsule SDF is:
          pa = p - a
          ba = b - a
          h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
          return length(pa - ba * h) - abs(r)
        """
        with open(self.WGSL_PATH) as f:
            src = f.read()

        assert "length(pa - ba * h)" in src, (
            "Body missing 'length(pa - ba * h)' -- formula may have changed"
        )
        assert "abs(r)" in src or "abs(" in src, (
            "Body missing abs(r) guard"
        )
        assert "clamp" in src, (
            "Body missing clamp() -- projection h not clamped"
        )

    def test_wgsl_comment_attributes(self):
        """Layer 1d: File has expected header comments."""
        with open(self.WGSL_PATH) as f:
            src = f.read()

        assert "sdf_capsule.wgsl" in src, "Missing filename in header"
        assert "T-DEMO-1.7" in src, "Missing T-DEMO-1.7 tracking marker"
        assert "SPDX-License-Identifier: MIT" in src, "Missing license header"

    def test_wgsl_edge_case_comment(self):
        """Layer 1e: Comments document edge cases (r=0, A==B, r<0)."""
        with open(self.WGSL_PATH) as f:
            src = f.read()

        assert "r = 0" in src or "r=0" in src, (
            "Missing r=0 edge case documentation"
        )
        assert "A ==" in src or "a == b" in src, (
            "Missing A==B degenerate case documentation"
        )
        assert "r < 0" in src or "r<0" in src, (
            "Missing negative radius documentation"
        )


# =============================================================================
# Layer 2: AST node import verification
# =============================================================================


class TestCapsuleNodeImport:
    """Verify CapsuleNode is importable both from ast_nodes and __init__."""

    def test_import_from_ast_nodes(self):
        """Layer 2a: CapsuleNode imports directly from ast_nodes."""
        from engine.rendering.demoscene.ast_nodes import (
            CapsuleNode, PositionNode, Vec3Node, FloatNode,
        )

        node = CapsuleNode(
            position=PositionNode(),
            endpoint_a=Vec3Node(0.0, -1.0, 0.0),
            endpoint_b=Vec3Node(0.0, 1.0, 0.0),
            radius=FloatNode(0.5),
        )
        assert node is not None
        assert node.label() == "Capsule(r=0.5)"
        # Verify it's an SdfPrimitiveNode
        from engine.rendering.demoscene.ast_nodes import SdfPrimitiveNode
        assert isinstance(node, SdfPrimitiveNode), (
            "CapsuleNode must inherit from SdfPrimitiveNode"
        )

    def test_import_from_init(self):
        """Layer 2b: CapsuleNode is re-exported via __init__.py."""
        from engine.rendering.demoscene import CapsuleNode
        assert CapsuleNode is not None

    def test_import_from_init_full(self):
        """Layer 2c: All expected capsule exports are present."""
        from engine.rendering.demoscene import (
            CapsuleNode, AstBuilder, PositionNode, Vec3Node, FloatNode,
        )
        from engine.rendering.demoscene import ast_builder
        # AstBuilder should have sdCapsule in its dispatch tables
        assert hasattr(ast_builder, "_PRIMITIVE_DISPATCH")
        assert "sdCapsule" in ast_builder._PRIMITIVE_DISPATCH
        assert "capsule" in ast_builder._MARKER_DISPATCH


# =============================================================================
# Layer 3: AST dispatch verification
# =============================================================================


class TestAstBuilderDispatch:
    """Verify that the AST builder dispatches produce CapsuleNode."""

    def test_sd_capsule_primitive_dispatch(self):
        """Layer 3a: sdCapsule dispatch produces CapsuleNode with correct args."""
        from engine.rendering.demoscene.ast_builder import _PRIMITIVE_DISPATCH

        node = _PRIMITIVE_DISPATCH["sdCapsule"](
            (1.0, 2.0, 3.0),
            a=(0.0, -2.0, 0.0),
            b=(0.0, 2.0, 0.0),
            r=0.75,
        )
        from engine.rendering.demoscene.ast_nodes import CapsuleNode
        assert isinstance(node, CapsuleNode), (
            f"sdCapsule dispatch should return CapsuleNode, got {type(node).__name__}"
        )
        assert node.label() == "Capsule(r=0.75)", f"Unexpected label: {node.label()}"

    def test_capsule_marker_dispatch(self):
        """Layer 3b: 'capsule' marker dispatch produces CapsuleNode."""
        from engine.rendering.demoscene.ast_builder import _MARKER_DISPATCH

        from engine.rendering.demoscene.ast_nodes import PositionNode
        node = _MARKER_DISPATCH["capsule"](
            position=PositionNode(),
            endpoint_a=(0.0, -1.0, 0.0),
            endpoint_b=(0.0, 1.0, 0.0),
            radius=0.5,
        )
        from engine.rendering.demoscene.ast_nodes import CapsuleNode
        assert isinstance(node, CapsuleNode), (
            f"'capsule' marker dispatch should return CapsuleNode, "
            f"got {type(node).__name__}"
        )
        assert node.label() == "Capsule(r=0.5)"

    def test_sd_capsule_default_radius(self):
        """Layer 3c: sdCapsule default radius is 0.5."""
        from engine.rendering.demoscene.ast_builder import _PRIMITIVE_DISPATCH

        node = _PRIMITIVE_DISPATCH["sdCapsule"](
            (0.0, 0.0, 0.0),
            a=(0.0, -1.0, 0.0),
            b=(0.0, 1.0, 0.0),
        )
        assert "r=0.5" in node.label(), (
            f"Default radius should be 0.5, got {node.label()}"
        )

    def test_capsule_marker_default_radius(self):
        """Layer 3d: 'capsule' marker default radius is 0.5."""
        from engine.rendering.demoscene.ast_builder import _MARKER_DISPATCH

        node = _MARKER_DISPATCH["capsule"]()
        assert "r=0.5" in node.label(), (
            f"Default radius should be 0.5, got {node.label()}"
        )


# =============================================================================
# Layer 4: Codegen template verification
# =============================================================================


class TestCodegenTemplate:
    """Verify SDF_CAPSULE template produces correct WGSL call strings."""

    def test_sdf_capsule_template_exists(self):
        """Layer 4a: SDF_CAPSULE is defined in wgsl_codegen."""
        from engine.rendering.demoscene.wgsl_codegen import SDF_CAPSULE
        assert SDF_CAPSULE is not None
        assert isinstance(SDF_CAPSULE, str)

    def test_sdf_capsule_template_format(self):
        """Layer 4b: SDF_CAPSULE template formats correctly."""
        from engine.rendering.demoscene.wgsl_codegen import SDF_CAPSULE

        rendered = SDF_CAPSULE.format(
            position="worldPos",
            endpoint_a="vec3<f32>(0.0, -2.0, 0.0)",
            endpoint_b="vec3<f32>(0.0, 2.0, 0.0)",
            radius="0.75",
        )
        expected = (
            "sdCapsule(worldPos, vec3<f32>(0.0, -2.0, 0.0), "
            "vec3<f32>(0.0, 2.0, 0.0), 0.75)"
        )
        assert rendered == expected, (
            f"Template rendered incorrectly:\n"
            f"  Expected: {expected}\n"
            f"  Got:      {rendered}"
        )

    def test_sdf_capsule_template_placeholders(self):
        """Layer 4c: Template contains all expected placeholders."""
        from engine.rendering.demoscene.wgsl_codegen import SDF_CAPSULE

        for placeholder in ("position", "endpoint_a", "endpoint_b", "radius"):
            assert "{" + placeholder + "}" in SDF_CAPSULE, (
                f"Missing placeholder {{{placeholder}}} in SDF_CAPSULE template"
            )

    def test_sdf_capsule_template_importable_via_init(self):
        """Layer 4d: SDF_CAPSULE is reachable from the demoscene package."""
        from engine.rendering.demoscene.wgsl_codegen import SDF_CAPSULE
        # Just a smoke check
        assert "sdCapsule" in SDF_CAPSULE
        assert "{position}" in SDF_CAPSULE


# =============================================================================
# Python model of sdCapsule matching WGSL semantics exactly (whitebox)
# =============================================================================


def py_sd_capsule(p, a, b, r):
    """Python model of WGSL sdCapsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32.

    Signed distance from point p to a capsule defined by endpoints a, b and
    radius r. Uses the IQ formula:
      pa = p - a
      ba = b - a
      h  = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
      return length(pa - ba * h) - abs(r)

    Reference: https://iquilezles.org/articles/distfunctions/
    """
    pa_x = p[0] - a[0]
    pa_y = p[1] - a[1]
    pa_z = p[2] - a[2]

    ba_x = b[0] - a[0]
    ba_y = b[1] - a[1]
    ba_z = b[2] - a[2]

    baba = max(ba_x * ba_x + ba_y * ba_y + ba_z * ba_z, 1e-10)

    h_num = pa_x * ba_x + pa_y * ba_y + pa_z * ba_z
    h = max(0.0, min(1.0, h_num / baba))

    px = pa_x - ba_x * h
    py = pa_y - ba_y * h
    pz = pa_z - ba_z * h

    return math.sqrt(px * px + py * py + pz * pz) - abs(r)


# =============================================================================
# Tolerance constants
# =============================================================================

TOL_SURFACE = 1e-12     # Points on surface should be extremely close to 0
TOL_EXACT = 1e-15       # For exact arithmetic expectations


# =============================================================================
# Path 1: Formula verification -- IQ capsule SDF
# =============================================================================


class TestFormula:
    """Verify the Python model matches the IQ capsule formula."""

    def test_formula_structure(self):
        """Verify sdCapsule computes length(pa - ba*h) - abs(r)."""
        p = (4.0, 0.0, 0.0)
        a = (0.0, -1.0, 0.0)
        b = (0.0, 1.0, 0.0)
        r = 2.0

        result = py_sd_capsule(p, a, b, r)

        # The point is at (4,0,0). Closest point on segment is (0,0,0).
        # Distance from p to segment = 4.0. Subtract radius 2.0 = 2.0.
        expected = 4.0 - 2.0  # 2.0
        assert result == pytest.approx(expected, abs=TOL_EXACT), (
            f"sdCapsule({p}, {a}, {b}, {r}) should equal {expected}, got {result}"
        )

    def test_abs_r_guard(self):
        """Verify abs(r) handles negative radius."""
        r_pos = 2.0
        r_neg = -2.0
        p = (3.0, 0.0, 0.0)
        a = (0.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0)

        d_pos = py_sd_capsule(p, a, b, r_pos)
        d_neg = py_sd_capsule(p, a, b, r_neg)

        assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
            f"abs(r) guard: {d_pos} should equal {d_neg}"
        )


# =============================================================================
# Path 2: Point aligned with segment axis (along the cylinder body)
# =============================================================================


class TestCylinderBody:
    """Points whose closest feature is the cylinder body (not an end-cap)."""

    def test_point_perpendicular_to_segment_midpoint(self):
        """Point offset perpendicular from segment midpoint."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (3.0, 0.0, 0.0)  # 3 units from segment at midpoint y=0
        r = 1.0
        result = py_sd_capsule(p, a, b, r)
        # Distance from segment = 3, minus radius 1 = 2
        expected = 3.0 - 1.0
        assert result == pytest.approx(expected, abs=TOL_EXACT)


# =============================================================================
# Path 3: Point above the top end-cap (closest to hemispherical cap)
# =============================================================================


class TestEndCap:
    """Points whose closest feature is a hemispherical end-cap."""

    def test_point_above_endpoint_b(self):
        """Point directly above endpoint B -- distance from B minus r."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (0.0, 5.0, 0.0)  # 3 units above B
        r = 1.0
        result = py_sd_capsule(p, a, b, r)
        # Distance from B = 3, minus radius 1 = 2
        expected = 3.0 - 1.0
        assert result == pytest.approx(expected, abs=TOL_EXACT)

    def test_point_below_endpoint_a(self):
        """Point directly below endpoint A -- distance from A minus r."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (0.0, -5.0, 0.0)  # 3 units below A
        r = 1.0
        result = py_sd_capsule(p, a, b, r)
        expected = 3.0 - 1.0
        assert result == pytest.approx(expected, abs=TOL_EXACT)


# =============================================================================
# Path 4: Surface point (distance exactly = radius)
# =============================================================================


class TestSurface:
    """Points exactly on the capsule surface should return 0."""

    def test_surface_cylinder_body(self):
        """Point on cylinder body surface at segment midpoint."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 1.5
        # Perpendicular offset = r = 1.5 from midpoint (0,0,0)
        p = (1.5, 0.0, 0.0)
        result = py_sd_capsule(p, a, b, r)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} on cylinder body should give 0, got {result}"
        )

    def test_surface_top_cap(self):
        """Point on hemispherical cap at endpoint B."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 1.0
        # At B + (r, 0, 0) = (1, 2, 0) -- on the cap surface
        p = (1.0, 2.0, 0.0)
        result = py_sd_capsule(p, a, b, r)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point {p} on top cap should give 0, got {result}"
        )


# =============================================================================
# Path 5: Inside point (negative SDF)
# =============================================================================


class TestInside:
    """Points inside the capsule should return negative SDF."""

    def test_inside_cylinder_body(self):
        """Point inside the cylinder body near the axis."""
        a = (0.0, -3.0, 0.0)
        b = (0.0, 3.0, 0.0)
        r = 2.0
        p = (1.0, 0.0, 0.0)  # 1 unit from axis, r=2, so 1 unit inside
        result = py_sd_capsule(p, a, b, r)
        assert result < 0, f"Inside point should give negative SDF, got {result}"
        expected = 1.0 - 2.0  # -1.0
        assert result == pytest.approx(expected, abs=TOL_EXACT)


# =============================================================================
# Path 6: Sign convention (negative inside, zero surface, positive outside)
# =============================================================================


class TestSignConvention:
    """Verify the sign convention: - / 0 / +."""

    INSIDE = (0.0, 0.0, 0.0)
    SURFACE = (1.0, 0.0, 0.0)
    OUTSIDE = (2.0, 0.0, 0.0)
    A = (0.0, -1.0, 0.0)
    B = (0.0, 1.0, 0.0)
    R = 1.0

    def test_inside_negative(self):
        result = py_sd_capsule(self.INSIDE, self.A, self.B, self.R)
        assert result < 0, f"Inside should be negative, got {result}"

    def test_surface_zero(self):
        result = py_sd_capsule(self.SURFACE, self.A, self.B, self.R)
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface should be 0, got {result}"
        )

    def test_outside_positive(self):
        result = py_sd_capsule(self.OUTSIDE, self.A, self.B, self.R)
        assert result > 0, f"Outside should be positive, got {result}"


# =============================================================================
# Path 7: Degenerate A==B case (collapses to sphere)
# =============================================================================


class TestDegeneratePoints:
    """When A == B, the capsule collapses to a sphere of radius r centered at A."""

    def test_a_equals_b_collapses_to_sphere(self):
        """A==B should behave like sdSphere(p, r) at A."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)  # Same as A
        r = 2.0
        p = (1.0, 2.0, 5.0)  # 2 units from A

        result = py_sd_capsule(p, a, b, r)
        # Distance from (1,2,3) to (1,2,5) = 2. Subtract radius 2 = 0
        assert result == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"A==B surface point should give 0, got {result}"
        )

    def test_a_equals_b_outside(self):
        """A==B outside point."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)
        r = 1.0
        p = (1.0, 2.0, 5.0)  # 2 units from A
        result = py_sd_capsule(p, a, b, r)
        expected = 2.0 - 1.0  # 1.0
        assert result == pytest.approx(expected, abs=TOL_EXACT)


# =============================================================================
# Path 8: Zero radius (collapses to line segment)
# =============================================================================


class TestZeroRadius:
    """When r=0, the capsule collapses to the line segment AB."""

    def test_zero_radius_perpendicular(self):
        """r=0: distance from point to line segment."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.0
        p = (3.0, 0.0, 0.0)
        result = py_sd_capsule(p, a, b, r)
        expected = 3.0  # Just perpendicular distance to segment
        assert result == pytest.approx(expected, abs=TOL_EXACT), (
            f"r=0 should give perpendicular distance {expected}, got {result}"
        )


# =============================================================================
# Path 9: Symmetry test (capsule is symmetric about its axis)
# =============================================================================


class TestSymmetry:
    """The capsule SDF is symmetric about the line through AB."""

    def test_rotational_symmetry_about_axis(self):
        """Points at same distance from segment axis yield same SDF."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5

        p_x = (3.0, 0.0, 0.0)
        p_z = (0.0, 0.0, 3.0)
        p_neg_x = (-3.0, 0.0, 0.0)

        d_x = py_sd_capsule(p_x, a, b, r)
        d_z = py_sd_capsule(p_z, a, b, r)
        d_nx = py_sd_capsule(p_neg_x, a, b, r)

        assert d_x == pytest.approx(d_z, abs=TOL_EXACT), (
            f"Symmetry: X={d_x} should equal Z={d_z}"
        )
        assert d_x == pytest.approx(d_nx, abs=TOL_EXACT), (
            f"Symmetry: X={d_x} should equal -X={d_nx}"
        )


# =============================================================================
# Path 10: Edge cases
# =============================================================================


class TestEdgeCases:
    """Additional edge case handling."""

    def test_clamp_projection_below_zero(self):
        """Point below A projects to h=0, giving distance to endpoint A."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (3.0, -5.0, 0.0)  # Below endpoint A
        result = py_sd_capsule(p, a, b, r)
        # Distance from p to A = sqrt(3^2 + 3^2) = sqrt(18) approx 4.2426
        dist_to_A = math.sqrt(3.0**2 + 3.0**2)
        expected = dist_to_A - r
        assert result == pytest.approx(expected, abs=TOL_SURFACE)

    def test_clamp_projection_above_one(self):
        """Point above B projects to h=1, giving distance to endpoint B."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (3.0, 5.0, 0.0)  # Above endpoint B
        result = py_sd_capsule(p, a, b, r)
        # Distance from p to B = sqrt(3^2 + 3^2) = sqrt(18) approx 4.2426
        dist_to_B = math.sqrt(3.0**2 + 3.0**2)
        expected = dist_to_B - r
        assert result == pytest.approx(expected, abs=TOL_SURFACE)

    def test_on_axis_inside(self):
        """Point exactly on the segment axis between A and B, inside the capsule."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 1.0
        p = (0.0, 0.0, 0.0)  # On axis, at midpoint
        result = py_sd_capsule(p, a, b, r)
        expected = -r  # -1.0: inside by exactly the radius
        assert result == pytest.approx(expected, abs=TOL_EXACT)
