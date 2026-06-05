"""
Tests for SDF WGSL Code Generator (T-DEMO-2.3, T-DEMO-2.5).

This module contains 40+ tests verifying:
- T-DEMO-2.3: WGSL code generation for all 12 primitive types
- T-DEMO-2.5: WGSL code generation for 6 domain operations
- Correct WGSL function call generation
- Proper handling of transformations
- Domain operation chaining
- WGSL output validity

Test Categories:
- TestPrimitiveCodegen: Tests for each of the 12 primitives
- TestDomainOpCodegen: Tests for each of the 6 domain operations
- TestFunctionDefinitions: Tests for WGSL function body correctness
- TestDomainChaining: Tests for nested domain operation generation
- TestTransformations: Tests for translate/rotate/scale wrapping
- TestWGSLValidity: Tests for syntactic validity of generated code
- TestEdgeCases: Tests for edge cases and error handling
"""

from __future__ import annotations

import math
import re

import pytest

from engine.rendering.demoscene.sdf_codegen import (
    WGSLCodegen,
    TransformContext,
    generate_primitive_wgsl,
    generate_domain_op_wgsl,
    generate_scene_sdf,
    get_all_primitive_wgsl,
    get_all_domain_op_wgsl,
    PRIMITIVE_WGSL_FUNCTIONS,
    DOMAIN_OP_WGSL_FUNCTIONS,
    _fmt_float,
    _fmt_vec3,
)
from engine.rendering.demoscene.ast_nodes import (
    Axis,
    BendNode,
    BoxFrameNode,
    BoxNode,
    CapsuleNode,
    ConeNode,
    CylinderNode,
    EllipsoidNode,
    FloatNode,
    KifsNode,
    MirrorNode,
    OctahedronNode,
    PlaneNode,
    PositionNode,
    PyramidNode,
    RepeatNode,
    RoundedBoxNode,
    SceneGraph,
    SphereNode,
    StretchNode,
    TorusNode,
    TwistNode,
    Vec3Node,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def codegen():
    """Fresh WGSLCodegen instance for each test."""
    return WGSLCodegen()


@pytest.fixture
def sphere_node():
    return SphereNode(PositionNode(), FloatNode(1.5))


@pytest.fixture
def box_node():
    return BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))


@pytest.fixture
def torus_node():
    return TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))


@pytest.fixture
def cylinder_node():
    return CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0))


@pytest.fixture
def cone_node():
    return ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0))


@pytest.fixture
def plane_node():
    return PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0))


@pytest.fixture
def capsule_node():
    return CapsuleNode(
        PositionNode(),
        Vec3Node(0.0, 0.0, 0.0),
        Vec3Node(0.0, 1.0, 0.0),
        FloatNode(0.5),
    )


@pytest.fixture
def ellipsoid_node():
    return EllipsoidNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))


@pytest.fixture
def box_frame_node():
    return BoxFrameNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.1))


@pytest.fixture
def rounded_box_node():
    return RoundedBoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0), FloatNode(0.2))


@pytest.fixture
def octahedron_node():
    return OctahedronNode(PositionNode(), FloatNode(1.0))


@pytest.fixture
def pyramid_node():
    return PyramidNode(PositionNode(), FloatNode(1.0))


# =============================================================================
# T-DEMO-2.3: PRIMITIVE CODE GENERATION TESTS
# =============================================================================

class TestPrimitiveCodegen:
    """Tests for WGSL code generation of all 12 primitive types."""

    def test_sphere_generates_sdf_sphere_call(self, codegen, sphere_node):
        """Test: Sphere -> sdf_sphere(p, radius)"""
        result = codegen.generate_primitive(sphere_node)
        assert result == "sdf_sphere(p, 1.5)"

    def test_sphere_with_center_offset(self, codegen, sphere_node):
        """Test: Sphere with center -> sdf_sphere(p - center, radius)"""
        result = codegen.generate_primitive(sphere_node, "p", "center")
        assert result == "sdf_sphere(p - center, 1.5)"

    def test_box_generates_sdf_box_call(self, codegen, box_node):
        """Test: Box -> sdf_box(p, half_extents)"""
        result = codegen.generate_primitive(box_node)
        assert result == "sdf_box(p, vec3<f32>(1.0, 2.0, 3.0))"

    def test_torus_generates_sdf_torus_call(self, codegen, torus_node):
        """Test: Torus -> sdf_torus(p, vec2(major, minor))"""
        result = codegen.generate_primitive(torus_node)
        assert result == "sdf_torus(p, vec2<f32>(2.0, 0.5))"

    def test_cylinder_generates_sdf_cylinder_call(self, codegen, cylinder_node):
        """Test: Cylinder -> sdf_cylinder(p, vec2(radius, half_height))"""
        result = codegen.generate_primitive(cylinder_node)
        assert result == "sdf_cylinder(p, vec2<f32>(1.0, 1.0))"

    def test_cone_generates_sdf_cone_call(self, codegen, cone_node):
        """Test: Cone -> sdf_cone(p, slope, height)"""
        result = codegen.generate_primitive(cone_node)
        assert "sdf_cone(p, vec2<f32>(" in result
        assert "2.0)" in result

    def test_plane_generates_sdf_plane_call(self, codegen, plane_node):
        """Test: Plane -> sdf_plane(p, normal)"""
        result = codegen.generate_primitive(plane_node)
        assert result == "sdf_plane(p, vec4<f32>(0.0, 1.0, 0.0, 0.0))"

    def test_capsule_generates_sdf_capsule_call(self, codegen, capsule_node):
        """Test: Capsule -> sdf_capsule(p, a, b, radius)"""
        result = codegen.generate_primitive(capsule_node)
        assert "sdf_capsule(p, vec3<f32>(0.0, 0.0, 0.0)" in result
        assert "vec3<f32>(0.0, 1.0, 0.0), 0.5)" in result

    def test_ellipsoid_generates_sdf_ellipsoid_call(self, codegen, ellipsoid_node):
        """Test: Ellipsoid -> sdf_ellipsoid(p, radii)"""
        result = codegen.generate_primitive(ellipsoid_node)
        assert result == "sdf_ellipsoid(p, vec3<f32>(1.0, 2.0, 3.0))"

    def test_box_frame_generates_sdf_box_frame_call(self, codegen, box_frame_node):
        """Test: BoxFrame -> sdf_box_frame(p, half_extents, edge_thickness)"""
        result = codegen.generate_primitive(box_frame_node)
        assert result == "sdf_box_frame(p, vec3<f32>(1.0, 1.0, 1.0), 0.1)"

    def test_rounded_box_generates_sdf_rounded_box_call(self, codegen, rounded_box_node):
        """Test: RoundedBox -> sdf_rounded_box(p, half_extents, corner_radius)"""
        result = codegen.generate_primitive(rounded_box_node)
        assert result == "sdf_rounded_box(p, vec3<f32>(1.0, 1.0, 1.0), 0.2)"

    def test_octahedron_generates_sdf_octahedron_call(self, codegen, octahedron_node):
        """Test: Octahedron -> sdf_octahedron(p, scale)"""
        result = codegen.generate_primitive(octahedron_node)
        assert result == "sdf_octahedron(p, 1.0)"

    def test_pyramid_generates_sdf_pyramid_call(self, codegen, pyramid_node):
        """Test: Pyramid -> sdf_pyramid(p, height)"""
        result = codegen.generate_primitive(pyramid_node)
        assert result == "sdf_pyramid(p, 1.0)"


# =============================================================================
# T-DEMO-2.5: DOMAIN OPERATION CODE GENERATION TESTS
# =============================================================================

class TestDomainOpCodegen:
    """Tests for WGSL code generation of all 6 domain operations."""

    def test_repeat_generates_domain_repeat_call(self, codegen):
        """Test: Repeat -> domain_repeat(p, cell_size)"""
        node = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        result = codegen.generate_domain_op(node)
        assert result == "domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0))"

    def test_mirror_x_generates_domain_mirror_x_call(self, codegen):
        """Test: Mirror X -> domain_mirror_x(p)"""
        node = MirrorNode(PositionNode(), Axis.X)
        result = codegen.generate_domain_op(node)
        assert result == "domain_mirror_x(p)"

    def test_mirror_y_generates_domain_mirror_y_call(self, codegen):
        """Test: Mirror Y -> domain_mirror_y(p)"""
        node = MirrorNode(PositionNode(), Axis.Y)
        result = codegen.generate_domain_op(node)
        assert result == "domain_mirror_y(p)"

    def test_mirror_z_generates_domain_mirror_z_call(self, codegen):
        """Test: Mirror Z -> domain_mirror_z(p)"""
        node = MirrorNode(PositionNode(), Axis.Z)
        result = codegen.generate_domain_op(node)
        assert result == "domain_mirror_z(p)"

    def test_kifs_generates_domain_fold_kifs_call(self, codegen):
        """Test: KIFS -> domain_fold_kifs(p, iterations)"""
        node = KifsNode(PositionNode(), FloatNode(6.0))
        result = codegen.generate_domain_op(node)
        assert result == "domain_fold_kifs(p, 6)"

    def test_twist_generates_domain_twist_call(self, codegen):
        """Test: Twist -> domain_twist_y(p, amount)"""
        node = TwistNode(PositionNode(), FloatNode(0.5))
        result = codegen.generate_domain_op(node)
        assert result == "domain_twist_y(p, 0.5)"

    def test_bend_generates_domain_bend_call(self, codegen):
        """Test: Bend -> domain_bend_z(p, radius)"""
        node = BendNode(PositionNode(), FloatNode(10.0))
        result = codegen.generate_domain_op(node)
        assert result == "domain_bend_z(p, 10.0)"

    def test_stretch_x_generates_domain_stretch_x_call(self, codegen):
        """Test: Stretch X -> domain_stretch_x(p, scale)"""
        node = StretchNode(PositionNode(), FloatNode(2.0), Axis.X)
        result = codegen.generate_domain_op(node)
        assert result == "domain_stretch_x(p, 2.0)"

    def test_stretch_y_generates_domain_stretch_y_call(self, codegen):
        """Test: Stretch Y -> domain_stretch_y(p, scale)"""
        node = StretchNode(PositionNode(), FloatNode(2.0), Axis.Y)
        result = codegen.generate_domain_op(node)
        assert result == "domain_stretch_y(p, 2.0)"

    def test_stretch_z_generates_domain_stretch_z_call(self, codegen):
        """Test: Stretch Z -> domain_stretch_z(p, scale)"""
        node = StretchNode(PositionNode(), FloatNode(2.0), Axis.Z)
        result = codegen.generate_domain_op(node)
        assert result == "domain_stretch_z(p, 2.0)"


# =============================================================================
# FUNCTION DEFINITION TESTS
# =============================================================================

class TestFunctionDefinitions:
    """Tests for WGSL function body correctness."""

    def test_sphere_function_contains_length(self, codegen, sphere_node):
        """Test: Sphere WGSL uses length() builtin"""
        codegen.emit_function(SphereNode)
        wgsl = codegen.get_emitted_functions()
        assert "length(p) - r" in wgsl

    def test_box_function_contains_abs(self, codegen, box_node):
        """Test: Box WGSL uses abs() builtin"""
        codegen.emit_function(BoxNode)
        wgsl = codegen.get_emitted_functions()
        assert "abs(p) - b" in wgsl

    def test_torus_function_contains_xz_swizzle(self, codegen, torus_node):
        """Test: Torus WGSL uses p.xz swizzle"""
        codegen.emit_function(TorusNode)
        wgsl = codegen.get_emitted_functions()
        assert "p.xz" in wgsl

    def test_ellipsoid_function_uses_normalized_formula(self, codegen, ellipsoid_node):
        """Test: Ellipsoid WGSL uses k0/k1 normalization"""
        codegen.emit_function(EllipsoidNode)
        wgsl = codegen.get_emitted_functions()
        assert "k0" in wgsl
        assert "k1" in wgsl
        assert "p / r" in wgsl

    def test_octahedron_function_contains_magic_constant(self, codegen, octahedron_node):
        """Test: Octahedron WGSL uses 0.57735027 (1/sqrt(3))"""
        codegen.emit_function(OctahedronNode)
        wgsl = codegen.get_emitted_functions()
        assert "0.57735027" in wgsl

    def test_pyramid_function_contains_select(self, codegen, pyramid_node):
        """Test: Pyramid WGSL uses select() for inside/outside"""
        codegen.emit_function(PyramidNode)
        wgsl = codegen.get_emitted_functions()
        assert "select(" in wgsl

    def test_all_primitive_wgsl_contains_all_functions(self):
        """Test: get_all_primitive_wgsl() includes all 12 primitives"""
        wgsl = get_all_primitive_wgsl()
        assert "sdf_sphere" in wgsl
        assert "sdf_box" in wgsl
        assert "sdf_torus" in wgsl
        assert "sdf_cylinder" in wgsl
        assert "sdf_cone" in wgsl
        assert "sdf_plane" in wgsl
        assert "sdf_capsule" in wgsl
        assert "sdf_ellipsoid" in wgsl
        assert "sdf_box_frame" in wgsl
        assert "sdf_rounded_box" in wgsl
        assert "sdf_octahedron" in wgsl
        assert "sdf_pyramid" in wgsl

    def test_all_domain_op_wgsl_contains_all_operations(self):
        """Test: get_all_domain_op_wgsl() includes all domain ops"""
        wgsl = get_all_domain_op_wgsl()
        assert "domain_repeat" in wgsl
        assert "domain_mirror_x" in wgsl
        assert "domain_mirror_y" in wgsl
        assert "domain_mirror_z" in wgsl
        assert "domain_fold_kifs" in wgsl
        assert "domain_twist_x" in wgsl
        assert "domain_twist_y" in wgsl
        assert "domain_twist_z" in wgsl
        assert "domain_bend_x" in wgsl
        assert "domain_bend_y" in wgsl
        assert "domain_bend_z" in wgsl
        assert "domain_stretch_x" in wgsl
        assert "domain_stretch_y" in wgsl
        assert "domain_stretch_z" in wgsl


# =============================================================================
# DOMAIN CHAINING TESTS
# =============================================================================

class TestDomainChaining:
    """Tests for nested domain operation generation."""

    def test_single_domain_op_chain(self, codegen):
        """Test: Single operation chain"""
        pipeline = (RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        result = codegen.generate_domain_chain(pipeline)
        assert "let p_transformed = domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0));" in result

    def test_empty_pipeline_returns_identity(self, codegen):
        """Test: Empty pipeline -> identity transform"""
        result = codegen.generate_domain_chain(())
        assert result == "let p_transformed = p;"

    def test_two_operation_chain_nests_correctly(self, codegen):
        """Test: Two operations nest outer->inner correctly"""
        pipeline = (
            RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            MirrorNode(PositionNode(), Axis.X),
        )
        result = codegen.generate_domain_chain(pipeline)
        # Pipeline order: Repeat first, then Mirror
        # In WGSL: first op is outermost: repeat(mirror(p))
        assert "domain_repeat(domain_mirror_x(p), vec3<f32>(2.0, 2.0, 2.0))" in result

    def test_three_operation_chain(self, codegen):
        """Test: Three operations chain correctly"""
        pipeline = (
            RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            MirrorNode(PositionNode(), Axis.X),
            TwistNode(PositionNode(), FloatNode(0.5)),
        )
        result = codegen.generate_domain_chain(pipeline)
        # Twist wraps Mirror wraps Repeat
        assert "domain_twist_y(" in result
        assert "domain_mirror_x(" in result
        assert "domain_repeat(" in result


# =============================================================================
# TRANSFORMATION TESTS
# =============================================================================

class TestTransformations:
    """Tests for translate/rotate/scale transformation wrapping."""

    def test_translate_generates_offset_subtraction(self, codegen, sphere_node):
        """Test: Translation generates (p - center)"""
        context = TransformContext(translate=Vec3Node(1.0, 2.0, 3.0))
        result = codegen.generate_primitive_with_transforms(sphere_node, context)
        assert "vec3<f32>(1.0, 2.0, 3.0)" in result

    def test_scale_generates_inverse_multiply(self, codegen, sphere_node):
        """Test: Scale generates inverse multiplication"""
        context = TransformContext(scale=Vec3Node(2.0, 2.0, 2.0))
        result = codegen.generate_primitive_with_transforms(sphere_node, context)
        assert "0.5" in result  # 1/2 = 0.5

    def test_no_transform_generates_direct_call(self, codegen, sphere_node):
        """Test: No transform -> direct primitive call"""
        context = TransformContext()
        result = codegen.generate_primitive_with_transforms(sphere_node, context)
        assert result == "sdf_sphere(p, 1.5)"


# =============================================================================
# WGSL VALIDITY TESTS
# =============================================================================

class TestWGSLValidity:
    """Tests for syntactic validity of generated WGSL code."""

    def test_generated_code_has_no_python_none(self, codegen, sphere_node):
        """Test: No Python 'None' leaks into WGSL"""
        codegen.emit_function(SphereNode)
        wgsl = codegen.get_emitted_functions()
        assert "None" not in wgsl

    def test_generated_code_has_balanced_parentheses(self, codegen, sphere_node):
        """Test: Parentheses are balanced"""
        result = codegen.generate_primitive(sphere_node)
        assert result.count("(") == result.count(")")

    def test_generated_code_has_balanced_angle_brackets(self, codegen, box_node):
        """Test: Angle brackets are balanced (vec3<f32>)"""
        result = codegen.generate_primitive(box_node)
        assert result.count("<") == result.count(">")

    def test_function_definitions_are_valid_wgsl(self, codegen):
        """Test: All function definitions have fn keyword"""
        codegen.emit_function(SphereNode)
        codegen.emit_function(BoxNode)
        wgsl = codegen.get_emitted_functions()
        assert wgsl.count("fn ") >= 2

    def test_function_definitions_have_return_type(self, codegen):
        """Test: Functions have -> f32 return type"""
        codegen.emit_function(SphereNode)
        wgsl = codegen.get_emitted_functions()
        assert "-> f32" in wgsl

    def test_scene_wgsl_is_complete(self, sphere_node):
        """Test: Scene generation produces complete WGSL"""
        graph = SceneGraph(primitives=(sphere_node,), name="test")
        wgsl = generate_scene_sdf(graph, "test")
        assert "// SPDX-License-Identifier: MIT" in wgsl
        assert "fn sdf_sphere" in wgsl
        assert "fn sd_scene_test" in wgsl
        assert "return " in wgsl


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_radius_sphere(self, codegen):
        """Test: Zero radius sphere generates valid code"""
        node = SphereNode(PositionNode(), FloatNode(0.0))
        result = codegen.generate_primitive(node)
        assert result == "sdf_sphere(p, 0.0)"

    def test_negative_values(self, codegen):
        """Test: Negative values are formatted correctly"""
        node = BoxNode(PositionNode(), Vec3Node(-1.0, -2.0, -3.0))
        result = codegen.generate_primitive(node)
        assert "-1.0" in result
        assert "-2.0" in result
        assert "-3.0" in result

    def test_large_values(self, codegen):
        """Test: Large values are handled"""
        node = SphereNode(PositionNode(), FloatNode(1000000.0))
        result = codegen.generate_primitive(node)
        assert "1000000.0" in result

    def test_small_values(self, codegen):
        """Test: Small values preserve precision"""
        node = SphereNode(PositionNode(), FloatNode(0.001))
        result = codegen.generate_primitive(node)
        assert "0.001" in result

    def test_unsupported_node_raises_error(self, codegen):
        """Test: Unsupported node type raises ValueError"""
        class UnknownNode:
            pass
        with pytest.raises(ValueError, match="Unsupported primitive"):
            codegen.generate_primitive(UnknownNode())

    def test_function_not_emitted_twice(self, codegen):
        """Test: Same function is not emitted twice"""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        codegen.emit_function(SphereNode)
        codegen.emit_function(SphereNode)
        wgsl = codegen.get_emitted_functions()
        assert wgsl.count("fn sdf_sphere") == 1

    def test_reset_clears_emitted_functions(self, codegen):
        """Test: reset() clears internal state"""
        codegen.emit_function(SphereNode)
        codegen.reset()
        codegen.emit_function(SphereNode)
        wgsl = codegen.get_emitted_functions()
        assert wgsl.count("fn sdf_sphere") == 1


# =============================================================================
# FORMAT HELPER TESTS
# =============================================================================

class TestFormatHelpers:
    """Tests for _fmt_float and _fmt_vec3 helpers."""

    def test_fmt_float_integer_adds_decimal(self):
        """Test: Integer floats get .0 suffix"""
        assert _fmt_float(1.0) == "1.0"
        assert _fmt_float(0.0) == "0.0"
        assert _fmt_float(-5.0) == "-5.0"

    def test_fmt_float_preserves_decimals(self):
        """Test: Non-integer floats preserved"""
        assert _fmt_float(1.5) == "1.5"
        assert _fmt_float(0.001) == "0.001"

    def test_fmt_vec3_generates_constructor(self):
        """Test: Vec3Node -> vec3<f32>(x, y, z)"""
        v = Vec3Node(1.0, 2.5, -3.0)
        assert _fmt_vec3(v) == "vec3<f32>(1.0, 2.5, -3.0)"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_generate_primitive_wgsl_function(self):
        """Test: generate_primitive_wgsl() convenience function"""
        sphere = SphereNode(PositionNode(), FloatNode(1.5))
        result = generate_primitive_wgsl(sphere)
        assert result == "sdf_sphere(p, 1.5)"

    def test_generate_domain_op_wgsl_function(self):
        """Test: generate_domain_op_wgsl() convenience function"""
        repeat = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        result = generate_domain_op_wgsl(repeat)
        assert result == "domain_repeat(p, vec3<f32>(2.0, 2.0, 2.0))"

    def test_generate_scene_sdf_with_pipeline(self):
        """Test: Scene with domain pipeline generates correct WGSL"""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        pipeline = (RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),)
        graph = SceneGraph(primitives=(sphere,), pipeline=pipeline, name="repeat_demo")
        wgsl = generate_scene_sdf(graph, "repeat_demo")
        assert "domain_repeat" in wgsl
        assert "sdf_sphere" in wgsl
        assert "p_d" in wgsl  # Transformed position variable


# =============================================================================
# MULTI-PRIMITIVE SCENE TESTS
# =============================================================================

class TestMultiPrimitiveScenes:
    """Tests for scenes with multiple primitives."""

    def test_two_primitives_union(self):
        """Test: Two primitives generate min() union"""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        graph = SceneGraph(primitives=(sphere, box), name="union")
        wgsl = generate_scene_sdf(graph, "union")
        assert "let d0 = sdf_sphere" in wgsl
        assert "let d1 = sdf_box" in wgsl
        assert "min(d0, d1)" in wgsl

    def test_three_primitives_nested_min(self):
        """Test: Three primitives generate nested min()"""
        prims = (
            SphereNode(PositionNode(), FloatNode(1.0)),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
        )
        graph = SceneGraph(primitives=prims, name="triple")
        wgsl = generate_scene_sdf(graph, "triple")
        assert "d0" in wgsl
        assert "d1" in wgsl
        assert "d2" in wgsl
        # Should have nested min calls
        assert "min(" in wgsl


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestRegistries:
    """Tests for WGSL function registries."""

    def test_primitive_registry_has_12_entries(self):
        """Test: PRIMITIVE_WGSL_FUNCTIONS has all 12 primitives"""
        assert len(PRIMITIVE_WGSL_FUNCTIONS) == 12

    def test_domain_op_registry_has_14_entries(self):
        """Test: DOMAIN_OP_WGSL_FUNCTIONS has all domain ops"""
        # 1 repeat + 3 mirrors + 1 kifs + 3 twists + 3 bends + 3 stretches = 14
        assert len(DOMAIN_OP_WGSL_FUNCTIONS) == 14

    def test_each_primitive_has_name_and_body(self):
        """Test: Each primitive entry has (name, body) tuple"""
        for node_type, (name, body) in PRIMITIVE_WGSL_FUNCTIONS.items():
            assert isinstance(name, str)
            assert name.startswith("sdf_")
            assert isinstance(body, str)
            assert f"fn {name}" in body
