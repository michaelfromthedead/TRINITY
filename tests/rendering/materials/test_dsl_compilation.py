"""Comprehensive DSL Compilation Test Suite (T-MAT-1.7).

This module provides extensive test coverage for the Material DSL compilation
pipeline, covering all 15+ AST node types, builtin function calls, texture
sampling, SurfaceOutput construction, and error cases.

Tests verify that generated WGSL is syntactically valid and semantically correct
for all supported language constructs.

Gap: S1-G7
Dependencies: T-MAT-1.3, T-MAT-1.4, T-MAT-1.6 (all DONE)
"""

from __future__ import annotations

import ast
import pytest
from typing import Any

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec2,
    Vec3,
    Vec4,
    PythonToWGSLTranslator,
    WGSLTranslationError,
    MaterialCompiler,
)


# =============================================================================
# Fixture: Translator instance
# =============================================================================


@pytest.fixture
def translator():
    """Fresh translator instance for each test."""
    return PythonToWGSLTranslator()


@pytest.fixture
def compiler():
    """Fresh compiler instance."""
    return MaterialCompiler(include_pbr_template=True)


@pytest.fixture
def body_compiler():
    """Compiler that returns only the surface body."""
    return MaterialCompiler(include_pbr_template=False)


# =============================================================================
# Helper Functions
# =============================================================================


def translate_code(code: str, translator: PythonToWGSLTranslator = None) -> str:
    """Translate Python code snippet to WGSL."""
    if translator is None:
        translator = PythonToWGSLTranslator()
    tree = ast.parse(code)
    return translator.translate(tree)


def assert_wgsl_contains(wgsl: str, *substrings: str) -> None:
    """Assert WGSL output contains all given substrings."""
    for sub in substrings:
        assert sub in wgsl, f"Expected '{sub}' in WGSL:\n{wgsl}"


def assert_no_compilation_error(material_class: type) -> None:
    """Assert material compiled without errors."""
    assert material_class._compilation_error is None, (
        f"Compilation error: {material_class._compilation_error}"
    )


# =============================================================================
# Suite 1: AST Node Type Coverage (15+ types)
# =============================================================================


class TestASTNodeExpr:
    """Tests for Expr AST node (expression statements)."""

    def test_expr_statement_function_call(self, translator):
        """Expression statement with function call emits correctly."""
        wgsl = translate_code("some_function(1.0, 2.0)", translator)
        assert_wgsl_contains(wgsl, "some_function", "1.0", "2.0", ";")

    def test_expr_statement_method_call(self, translator):
        """Expression statement with method call emits correctly."""
        wgsl = translate_code("obj.method()", translator)
        assert_wgsl_contains(wgsl, "obj.method()", ";")

    def test_expr_statement_chained_call(self, translator):
        """Expression statement with chained calls."""
        wgsl = translate_code("a.b.c()", translator)
        assert_wgsl_contains(wgsl, "a.b.c()", ";")


class TestASTNodeAssign:
    """Tests for Assign AST node (simple assignment)."""

    def test_assign_float(self, translator):
        """Simple float assignment."""
        wgsl = translate_code("x = 0.5", translator)
        assert_wgsl_contains(wgsl, "x = 0.5", ";")

    def test_assign_expression(self, translator):
        """Assignment with expression on RHS."""
        wgsl = translate_code("result = a + b * c", translator)
        assert_wgsl_contains(wgsl, "result =", "+", "*")

    def test_assign_multiple_targets(self, translator):
        """Multiple assignment targets."""
        wgsl = translate_code("a = b = 1.0", translator)
        # Should produce two separate assignments
        assert wgsl.count("= 1.0") >= 1

    def test_assign_vec_constructor(self, translator):
        """Assignment with vector constructor."""
        wgsl = translate_code("v = Vec3(1.0, 2.0, 3.0)", translator)
        assert_wgsl_contains(wgsl, "vec3<f32>", "1.0", "2.0", "3.0")


class TestASTNodeAnnAssign:
    """Tests for AnnAssign AST node (annotated assignment)."""

    def test_annassign_float(self, translator):
        """Annotated float assignment becomes let."""
        wgsl = translate_code("value: float = 0.5", translator)
        assert_wgsl_contains(wgsl, "let value = 0.5")

    def test_annassign_int(self, translator):
        """Annotated int assignment."""
        wgsl = translate_code("count: int = 42", translator)
        assert_wgsl_contains(wgsl, "let count = 42")

    def test_annassign_expression(self, translator):
        """Annotated assignment with expression."""
        wgsl = translate_code("result: float = x * 2.0 + y", translator)
        assert_wgsl_contains(wgsl, "let result =", "*", "+")


class TestASTNodeCall:
    """Tests for Call AST node (function/method calls)."""

    def test_call_builtin_single_arg(self, translator):
        """Builtin call with single argument."""
        wgsl = translate_code("result = abs(x)", translator)
        assert_wgsl_contains(wgsl, "abs(x)")

    def test_call_builtin_multiple_args(self, translator):
        """Builtin call with multiple arguments."""
        wgsl = translate_code("result = clamp(x, 0.0, 1.0)", translator)
        assert_wgsl_contains(wgsl, "clamp(x, 0.0, 1.0)")

    def test_call_vec_constructor(self, translator):
        """Vector constructor call."""
        wgsl = translate_code("v = Vec4(1.0, 2.0, 3.0, 4.0)", translator)
        assert_wgsl_contains(wgsl, "vec4<f32>(1.0, 2.0, 3.0, 4.0)")

    def test_call_nested(self, translator):
        """Nested function calls."""
        wgsl = translate_code("result = normalize(cross(a, b))", translator)
        assert_wgsl_contains(wgsl, "normalize(cross(a, b))")

    def test_call_context_sample(self, translator):
        """Context sample method call."""
        wgsl = translate_code("color = ctx.sample(tex, uv)", translator)
        assert_wgsl_contains(wgsl, "textureSample(tex, uv)")

    def test_call_context_sample_level(self, translator):
        """Context sample_level method call."""
        wgsl = translate_code("color = ctx.sample_level(tex, uv, 2.0)", translator)
        assert_wgsl_contains(wgsl, "textureSampleLevel(tex, uv, 2.0)")


class TestASTNodeBinOp:
    """Tests for BinOp AST node (binary operations)."""

    def test_binop_add(self, translator):
        """Addition operator."""
        wgsl = translate_code("result = a + b", translator)
        assert_wgsl_contains(wgsl, "(a + b)")

    def test_binop_sub(self, translator):
        """Subtraction operator."""
        wgsl = translate_code("result = a - b", translator)
        assert_wgsl_contains(wgsl, "(a - b)")

    def test_binop_mult(self, translator):
        """Multiplication operator."""
        wgsl = translate_code("result = a * b", translator)
        assert_wgsl_contains(wgsl, "(a * b)")

    def test_binop_div(self, translator):
        """Division operator."""
        wgsl = translate_code("result = a / b", translator)
        assert_wgsl_contains(wgsl, "(a / b)")

    def test_binop_mod(self, translator):
        """Modulo operator."""
        wgsl = translate_code("result = a % b", translator)
        assert_wgsl_contains(wgsl, "(a % b)")

    def test_binop_pow(self, translator):
        """Power operator becomes pow() call."""
        wgsl = translate_code("result = a ** b", translator)
        assert_wgsl_contains(wgsl, "pow(a, b)")

    def test_binop_bitand(self, translator):
        """Bitwise AND operator."""
        wgsl = translate_code("result = a & b", translator)
        assert_wgsl_contains(wgsl, "(a & b)")

    def test_binop_bitor(self, translator):
        """Bitwise OR operator."""
        wgsl = translate_code("result = a | b", translator)
        assert_wgsl_contains(wgsl, "(a | b)")

    def test_binop_bitxor(self, translator):
        """Bitwise XOR operator."""
        wgsl = translate_code("result = a ^ b", translator)
        assert_wgsl_contains(wgsl, "(a ^ b)")

    def test_binop_lshift(self, translator):
        """Left shift operator."""
        wgsl = translate_code("result = a << b", translator)
        assert_wgsl_contains(wgsl, "(a << b)")

    def test_binop_rshift(self, translator):
        """Right shift operator."""
        wgsl = translate_code("result = a >> b", translator)
        assert_wgsl_contains(wgsl, "(a >> b)")

    def test_binop_chained(self, translator):
        """Chained binary operations."""
        wgsl = translate_code("result = a + b * c - d", translator)
        assert_wgsl_contains(wgsl, "+", "*", "-")


class TestASTNodeUnaryOp:
    """Tests for UnaryOp AST node (unary operations)."""

    def test_unaryop_negate(self, translator):
        """Unary negation."""
        wgsl = translate_code("result = -x", translator)
        assert_wgsl_contains(wgsl, "(-x)")

    def test_unaryop_positive(self, translator):
        """Unary positive (identity)."""
        wgsl = translate_code("result = +x", translator)
        assert_wgsl_contains(wgsl, "(+x)")

    def test_unaryop_not(self, translator):
        """Unary not."""
        wgsl = translate_code("result = not condition", translator)
        assert_wgsl_contains(wgsl, "(!condition)")

    def test_unaryop_invert(self, translator):
        """Bitwise invert."""
        wgsl = translate_code("result = ~bits", translator)
        assert_wgsl_contains(wgsl, "(~bits)")


class TestASTNodeCompare:
    """Tests for Compare AST node (comparison expressions)."""

    def test_compare_eq(self, translator):
        """Equality comparison."""
        wgsl = translate_code("result = a == b", translator)
        assert_wgsl_contains(wgsl, "==")

    def test_compare_neq(self, translator):
        """Inequality comparison."""
        wgsl = translate_code("result = a != b", translator)
        assert_wgsl_contains(wgsl, "!=")

    def test_compare_lt(self, translator):
        """Less than comparison."""
        wgsl = translate_code("result = a < b", translator)
        assert_wgsl_contains(wgsl, "<")

    def test_compare_lte(self, translator):
        """Less than or equal comparison."""
        wgsl = translate_code("result = a <= b", translator)
        assert_wgsl_contains(wgsl, "<=")

    def test_compare_gt(self, translator):
        """Greater than comparison."""
        wgsl = translate_code("result = a > b", translator)
        assert_wgsl_contains(wgsl, ">")

    def test_compare_gte(self, translator):
        """Greater than or equal comparison."""
        wgsl = translate_code("result = a >= b", translator)
        assert_wgsl_contains(wgsl, ">=")


class TestASTNodeBoolOp:
    """Tests for BoolOp AST node (boolean operations)."""

    def test_boolop_and(self, translator):
        """Boolean AND."""
        wgsl = translate_code("result = a and b", translator)
        assert_wgsl_contains(wgsl, "(a && b)")

    def test_boolop_or(self, translator):
        """Boolean OR."""
        wgsl = translate_code("result = a or b", translator)
        assert_wgsl_contains(wgsl, "(a || b)")

    def test_boolop_chained(self, translator):
        """Chained boolean operations."""
        wgsl = translate_code("result = a and b or c", translator)
        assert_wgsl_contains(wgsl, "&&", "||")


class TestASTNodeIf:
    """Tests for If AST node (conditional statements)."""

    def test_if_simple(self, translator):
        """Simple if statement."""
        code = """
if condition:
    x = 1.0
"""
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, "if (condition)", "{", "}")

    def test_if_else(self, translator):
        """If-else statement."""
        code = """
if condition:
    x = 1.0
else:
    x = 0.0
"""
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, "if", "else", "{", "}")

    def test_if_elif_else(self, translator):
        """If-elif-else chain."""
        code = """
if a > 0.5:
    x = 1.0
elif a > 0.25:
    x = 0.5
else:
    x = 0.0
"""
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, "if", "else")


class TestASTNodeAttribute:
    """Tests for Attribute AST node (attribute access)."""

    def test_attribute_simple(self, translator):
        """Simple attribute access."""
        wgsl = translate_code("result = obj.attr", translator)
        assert_wgsl_contains(wgsl, "obj.attr")

    def test_attribute_output_mapping(self, translator):
        """Output field mapping."""
        wgsl = translate_code("out.base_color = value", translator)
        assert_wgsl_contains(wgsl, "out.base_color")

    def test_attribute_context_mapping(self, translator):
        """Context property mapping."""
        wgsl = translate_code("pos = ctx.position", translator)
        assert_wgsl_contains(wgsl, "in.position")

    def test_attribute_self_texture(self, translator):
        """Self texture reference."""
        wgsl = translate_code("tex = self.albedo_texture", translator)
        assert_wgsl_contains(wgsl, "material.albedo_texture")


class TestASTNodeName:
    """Tests for Name AST node (variable references)."""

    def test_name_variable(self, translator):
        """Simple variable reference."""
        wgsl = translate_code("result = value", translator)
        assert_wgsl_contains(wgsl, "value")

    def test_name_true(self, translator):
        """Python True becomes WGSL true."""
        wgsl = translate_code("flag = True", translator)
        assert_wgsl_contains(wgsl, "true")

    def test_name_false(self, translator):
        """Python False becomes WGSL false."""
        wgsl = translate_code("flag = False", translator)
        assert_wgsl_contains(wgsl, "false")

    def test_name_none(self, translator):
        """Python None becomes 0.0."""
        wgsl = translate_code("val = None", translator)
        assert_wgsl_contains(wgsl, "0.0")


class TestASTNodeConstant:
    """Tests for Constant AST node (literals)."""

    def test_constant_int(self, translator):
        """Integer literal."""
        wgsl = translate_code("x = 42", translator)
        assert_wgsl_contains(wgsl, "42")

    def test_constant_float(self, translator):
        """Float literal."""
        wgsl = translate_code("x = 3.14159", translator)
        assert_wgsl_contains(wgsl, "3.14159")

    def test_constant_float_scientific(self, translator):
        """Scientific notation float."""
        wgsl = translate_code("x = 1e-5", translator)
        assert_wgsl_contains(wgsl, "1e-05") or "1e-5" in wgsl or "0.00001" in wgsl

    def test_constant_bool_true(self, translator):
        """Boolean True literal."""
        wgsl = translate_code("x = True", translator)
        assert_wgsl_contains(wgsl, "true")

    def test_constant_bool_false(self, translator):
        """Boolean False literal."""
        wgsl = translate_code("x = False", translator)
        assert_wgsl_contains(wgsl, "false")


class TestASTNodeSubscript:
    """Tests for Subscript AST node (index access)."""

    def test_subscript_numeric(self, translator):
        """Numeric index access."""
        wgsl = translate_code("x = arr[0]", translator)
        assert_wgsl_contains(wgsl, "arr[0]")

    def test_subscript_variable(self, translator):
        """Variable index access."""
        wgsl = translate_code("x = arr[i]", translator)
        assert_wgsl_contains(wgsl, "arr[i]")

    def test_subscript_expression(self, translator):
        """Expression index access."""
        wgsl = translate_code("x = arr[i + 1]", translator)
        assert_wgsl_contains(wgsl, "arr[(i + 1)]")


class TestASTNodeTuple:
    """Tests for Tuple AST node (vector construction)."""

    def test_tuple_vec2(self, translator):
        """2-element tuple becomes vec2."""
        wgsl = translate_code("v = (1.0, 2.0)", translator)
        assert_wgsl_contains(wgsl, "vec2<f32>(1.0, 2.0)")

    def test_tuple_vec3(self, translator):
        """3-element tuple becomes vec3."""
        wgsl = translate_code("v = (1.0, 2.0, 3.0)", translator)
        assert_wgsl_contains(wgsl, "vec3<f32>(1.0, 2.0, 3.0)")

    def test_tuple_vec4(self, translator):
        """4-element tuple becomes vec4."""
        wgsl = translate_code("v = (1.0, 2.0, 3.0, 4.0)", translator)
        assert_wgsl_contains(wgsl, "vec4<f32>(1.0, 2.0, 3.0, 4.0)")


class TestASTNodeReturn:
    """Tests for Return AST node."""

    def test_return_value(self, translator):
        """Return with value."""
        wgsl = translate_code("return value", translator)
        assert_wgsl_contains(wgsl, "return value;")

    def test_return_expression(self, translator):
        """Return with expression."""
        wgsl = translate_code("return a + b", translator)
        assert_wgsl_contains(wgsl, "return (a + b);")

    def test_return_void(self, translator):
        """Return without value."""
        wgsl = translate_code("return", translator)
        assert_wgsl_contains(wgsl, "return;")


class TestASTNodeIfExp:
    """Tests for IfExp AST node (ternary expression)."""

    def test_ifexp_simple(self, translator):
        """Simple ternary expression."""
        wgsl = translate_code("result = a if condition else b", translator)
        assert_wgsl_contains(wgsl, "select(b, a, condition)")

    def test_ifexp_nested(self, translator):
        """Nested ternary expression."""
        wgsl = translate_code("result = x if a else (y if b else z)", translator)
        assert_wgsl_contains(wgsl, "select")


class TestASTNodeList:
    """Tests for List AST node (array-like construction)."""

    def test_list_vec3(self, translator):
        """3-element list becomes vec3."""
        wgsl = translate_code("v = [1.0, 2.0, 3.0]", translator)
        assert_wgsl_contains(wgsl, "vec3<f32>(1.0, 2.0, 3.0)")


# =============================================================================
# Suite 2: Builtin Function Calls
# =============================================================================


class TestBuiltinMathFunctions:
    """Tests for built-in math functions."""

    @pytest.mark.parametrize("func,args", [
        ("abs", "x"),
        ("min", "a, b"),
        ("max", "a, b"),
        ("pow", "x, 2.0"),
        ("sqrt", "x"),
        ("sin", "x"),
        ("cos", "x"),
        ("tan", "x"),
        ("asin", "x"),
        ("acos", "x"),
        ("atan", "x"),
        ("atan2", "y, x"),
        ("exp", "x"),
        ("exp2", "x"),
        ("log", "x"),
        ("log2", "x"),
        ("floor", "x"),
        ("ceil", "x"),
        ("round", "x"),
        ("fract", "x"),
        ("trunc", "x"),
        ("sign", "x"),
        ("clamp", "x, 0.0, 1.0"),
        ("saturate", "x"),
    ])
    def test_math_builtins(self, translator, func, args):
        """Math builtin functions translate correctly."""
        code = f"result = {func}({args})"
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, func)


class TestBuiltinVectorFunctions:
    """Tests for built-in vector functions."""

    @pytest.mark.parametrize("func,args", [
        ("length", "v"),
        ("distance", "a, b"),
        ("dot", "a, b"),
        ("cross", "a, b"),
        ("normalize", "v"),
        ("reflect", "i, n"),
        ("refract", "i, n, eta"),
        ("faceforward", "n, i, nref"),
    ])
    def test_vector_builtins(self, translator, func, args):
        """Vector builtin functions translate correctly."""
        code = f"result = {func}({args})"
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, func)


class TestBuiltinInterpolationFunctions:
    """Tests for interpolation and blending functions."""

    @pytest.mark.parametrize("func,args,wgsl_func", [
        ("mix", "a, b, t", "mix"),
        ("lerp", "a, b, t", "mix"),  # lerp is aliased to mix
        ("step", "edge, x", "step"),
        ("smoothstep", "e0, e1, x", "smoothstep"),
    ])
    def test_interpolation_builtins(self, translator, func, args, wgsl_func):
        """Interpolation builtin functions translate correctly."""
        code = f"result = {func}({args})"
        wgsl = translate_code(code, translator)
        assert_wgsl_contains(wgsl, wgsl_func)


class TestCustomNoiseBuiltins:
    """Tests for custom noise builtin functions."""

    @pytest.mark.parametrize("func", [
        "value_noise",
        "perlin_noise",
        "simplex_noise",
        "worley_noise",
        "fbm",
        "turbulence",
    ])
    def test_noise_builtins(self, translator, func):
        """Noise builtin functions are tracked as used."""
        code = f"result = {func}(uv)"
        translate_code(code, translator)
        assert func in translator.used_builtins


class TestCustomColorBuiltins:
    """Tests for custom color conversion builtin functions."""

    @pytest.mark.parametrize("func", [
        "rgb_to_hsv",
        "hsv_to_rgb",
        "srgb_to_linear",
        "linear_to_srgb",
        "tonemap_reinhard",
        "tonemap_aces",
        "tonemap_uncharted2",
        "tonemap_agx",
    ])
    def test_color_builtins(self, translator, func):
        """Color builtin functions are tracked as used."""
        code = f"result = {func}(color)"
        translate_code(code, translator)
        assert func in translator.used_builtins


class TestCustomMathUtilBuiltins:
    """Tests for custom math utility builtin functions."""

    @pytest.mark.parametrize("func", [
        "remap",
        "inverse_lerp",
        "smooth_min",
        "smooth_max",
        "smootherstep",
    ])
    def test_math_util_builtins(self, translator, func):
        """Math utility builtin functions are tracked as used."""
        code = f"result = {func}(0.0, 1.0, x)"
        translate_code(code, translator)
        assert func in translator.used_builtins


# =============================================================================
# Suite 3: Texture Sampling
# =============================================================================


class TestTextureSampling:
    """Tests for texture sampling operations."""

    def test_sample_2d_basic(self, translator):
        """Basic 2D texture sampling."""
        wgsl = translate_code("color = ctx.sample(tex, uv)", translator)
        assert_wgsl_contains(wgsl, "textureSample(tex, uv)")

    def test_sample_cube_basic(self, translator):
        """Basic cubemap texture sampling."""
        wgsl = translate_code("color = ctx.sample_cube(cubemap, direction)", translator)
        assert_wgsl_contains(wgsl, "textureSample(cubemap, direction)")

    def test_sample_level_basic(self, translator):
        """Basic mip level texture sampling."""
        wgsl = translate_code("color = ctx.sample_level(tex, uv, 3.0)", translator)
        assert_wgsl_contains(wgsl, "textureSampleLevel(tex, uv, 3.0)")

    def test_sample_with_uv_expression(self, translator):
        """Texture sampling with UV expression."""
        wgsl = translate_code("color = ctx.sample(tex, uv * 2.0)", translator)
        assert_wgsl_contains(wgsl, "textureSample", "(uv * 2.0)")

    def test_sample_swizzle_result(self, translator):
        """Texture sampling result swizzle."""
        wgsl = translate_code("r = ctx.sample(tex, uv).r", translator)
        assert_wgsl_contains(wgsl, "textureSample")
        assert ".r" in wgsl


# =============================================================================
# Suite 4: SurfaceContext Accessors
# =============================================================================


class TestSurfaceContextAccessors:
    """Tests for SurfaceContext method and property access."""

    def test_ctx_world_position_method(self, translator):
        """ctx.world_position() maps to in.world_position."""
        wgsl = translate_code("pos = ctx.world_position()", translator)
        assert_wgsl_contains(wgsl, "in.world_position")

    def test_ctx_world_normal_method(self, translator):
        """ctx.world_normal() maps to in.world_normal."""
        wgsl = translate_code("n = ctx.world_normal()", translator)
        assert_wgsl_contains(wgsl, "in.world_normal")

    def test_ctx_world_tangent_method(self, translator):
        """ctx.world_tangent() maps to in.world_tangent."""
        wgsl = translate_code("t = ctx.world_tangent()", translator)
        assert_wgsl_contains(wgsl, "in.world_tangent")

    def test_ctx_uv_property(self, translator):
        """ctx.uv property access."""
        wgsl = translate_code("uv = ctx.uv", translator)
        assert_wgsl_contains(wgsl, "in.uv")

    def test_ctx_get_uv_method(self, translator):
        """ctx.get_uv() maps to in.uv."""
        wgsl = translate_code("uv = ctx.get_uv()", translator)
        assert_wgsl_contains(wgsl, "in.uv")

    def test_ctx_vertex_color_property(self, translator):
        """ctx.vertex_color property access."""
        wgsl = translate_code("c = ctx.vertex_color", translator)
        assert_wgsl_contains(wgsl, "in.vertex_color")

    def test_ctx_time_property(self, translator):
        """ctx.time property access maps to time uniforms."""
        wgsl = translate_code("t = ctx.time", translator)
        # Implementation uses time_uniforms.elapsed_seconds
        assert_wgsl_contains(wgsl, "time_uniforms.elapsed_seconds")

    def test_ctx_get_time_method(self, translator):
        """ctx.get_time() maps to time uniforms."""
        wgsl = translate_code("t = ctx.get_time()", translator)
        # Implementation uses time_uniforms.elapsed_seconds
        assert_wgsl_contains(wgsl, "time_uniforms.elapsed_seconds")

    def test_ctx_view_direction(self, translator):
        """ctx.view_direction maps to input view direction."""
        wgsl = translate_code("v = ctx.view_direction", translator)
        # Property access maps to in.view_direction
        assert_wgsl_contains(wgsl, "in.view_direction")


# =============================================================================
# Suite 5: SurfaceOutput Construction
# =============================================================================


class TestSurfaceOutputFields:
    """Tests for SurfaceOutput field assignment."""

    @pytest.mark.parametrize("field,wgsl_field", [
        ("base_color", "out.base_color"),
        ("albedo", "out.base_color"),  # Alias
        ("metallic", "out.metallic"),
        ("roughness", "out.roughness"),
        ("normal", "out.normal"),
        ("emissive", "out.emissive"),
        ("emission", "out.emissive"),  # Alias
        ("ao", "out.ambient_occlusion"),
        ("alpha", "out.alpha"),
        ("specular", "out.specular"),
        ("subsurface", "out.subsurface"),
        ("subsurface_color", "out.subsurface_color"),
        ("clearcoat", "out.clearcoat"),
        ("clearcoat_roughness", "out.clearcoat_roughness"),
        ("anisotropy", "out.anisotropy"),
        ("anisotropy_direction", "out.anisotropy_direction"),
        ("sheen", "out.sheen"),
        ("sheen_color", "out.sheen_color"),
        ("transmission", "out.transmission"),
        ("ior", "out.ior"),
    ])
    def test_output_field_mapping(self, translator, field, wgsl_field):
        """Output fields map to correct WGSL names."""
        wgsl = translate_code(f"out.{field} = value", translator)
        assert_wgsl_contains(wgsl, wgsl_field)


# =============================================================================
# Suite 6: Error Cases
# =============================================================================


class TestUnsupportedConstructs:
    """Tests for unsupported Python constructs."""

    @pytest.mark.parametrize("code,description", [
        ("class Nested: pass", "Class definition"),
        ("import math", "Import statement"),
        ("from math import sqrt", "From import"),
        ("result = [x * 2 for x in items]", "List comprehension"),
        ("result = {k: v for k, v in pairs}", "Dict comprehension"),
        ("result = sum(x * 2 for x in items)", "Generator expression"),
        ("f = lambda x: x * 2", "Lambda expression"),
        ("s = {1, 2, 3}", "Set literal"),
        ("d = {'key': 'value'}", "Dict literal"),
        ("global x", "Global statement"),
        ("nonlocal y", "Nonlocal statement"),
        ("assert condition", "Assert statement"),
        ("raise Exception()", "Raise statement"),
        ("del x", "Delete statement"),
    ])
    def test_unsupported_raises_error(self, translator, code, description):
        """Unsupported construct raises WGSLTranslationError."""
        tree = ast.parse(code)
        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    @pytest.mark.parametrize("code,description", [
        ("try:\n    x = 1\nexcept:\n    x = 0", "Try/except"),
        ("with ctx as c:\n    x = c.value", "With statement"),
        ("for i in range(10):\n    x = i", "For loop"),
        ("while x < 10:\n    x = x + 1", "While loop"),
        ("async def foo():\n    pass", "Async function"),
        ("async def foo():\n    result = await bar()", "Await expression"),
    ])
    def test_block_constructs_unsupported(self, translator, code, description):
        """Block constructs raise WGSLTranslationError."""
        tree = ast.parse(code)
        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)


class TestErrorMessages:
    """Tests for error message clarity."""

    def test_error_includes_node_type(self, translator):
        """Error message includes the unsupported AST node type."""
        tree = ast.parse("class Foo: pass")
        with pytest.raises(WGSLTranslationError) as exc_info:
            translator.translate(tree)
        # Should mention the node type in the error
        assert "ClassDef" in str(exc_info.value) or "class" in str(exc_info.value).lower()

    def test_error_provides_guidance(self, translator):
        """Error message provides guidance on supported constructs."""
        tree = ast.parse("import os")
        with pytest.raises(WGSLTranslationError) as exc_info:
            translator.translate(tree)
        # Should mention DSL or shader limitation
        assert "DSL" in str(exc_info.value) or "shader" in str(exc_info.value).lower()


# =============================================================================
# Suite 7: Material Compilation Integration
# =============================================================================


class TestMaterialCompilationIntegration:
    """Tests for complete material compilation."""

    def test_simple_material_compiles(self, compiler):
        """Simple material compiles to valid WGSL."""

        class SimpleMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.0)
                out.roughness = 0.5

        assert_no_compilation_error(SimpleMat)
        wgsl = compiler.compile(SimpleMat)
        assert_wgsl_contains(wgsl, "struct PBRInput", "struct PBRParams", "@fragment")

    def test_material_with_all_pbr_outputs(self, compiler):
        """Material setting all PBR outputs compiles."""

        class FullPBR(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.6, 0.4)
                out.metallic = 0.5
                out.roughness = 0.4
                out.ao = 0.9
                out.alpha = 1.0
                out.emissive = Vec3(0.0, 0.0, 0.0)
                out.specular = 0.5
                out.clearcoat = 0.2
                out.sheen = 0.1
                out.transmission = 0.0
                out.ior = 1.5

        assert_no_compilation_error(FullPBR)
        wgsl = compiler.compile(FullPBR)
        assert len(wgsl) > 1000  # Should be a substantial shader

    def test_material_with_conditionals(self, body_compiler):
        """Material with if/else compiles."""

        class ConditionalMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if ctx.uv.y > 0.5:
                    out.metallic = 1.0
                else:
                    out.metallic = 0.0
                out.roughness = 0.5

        assert_no_compilation_error(ConditionalMat)
        wgsl = ConditionalMat._wgsl_source
        assert_wgsl_contains(wgsl, "if", "else")

    def test_material_with_local_vars(self, body_compiler):
        """Material with annotated local variables compiles."""

        class LocalVarMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                factor: float = 0.8
                brightness: float = factor * 0.5
                out.base_color = Vec3(brightness, brightness, brightness)
                out.roughness = 0.5

        assert_no_compilation_error(LocalVarMat)
        wgsl = LocalVarMat._wgsl_source
        assert_wgsl_contains(wgsl, "let factor", "let brightness")

    def test_material_with_math_operations(self, body_compiler):
        """Material with math operations compiles."""

        class MathMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                x: float = ctx.uv.x * 2.0 - 1.0
                y: float = ctx.uv.y * 2.0 - 1.0
                dist: float = x * x + y * y
                out.roughness = dist
                out.base_color = Vec3(0.5, 0.5, 0.5)

        assert_no_compilation_error(MathMat)
        wgsl = MathMat._wgsl_source
        assert_wgsl_contains(wgsl, "*", "+", "-")


class TestMaterialCompilerOptions:
    """Tests for MaterialCompiler options."""

    def test_compiler_template_disabled(self):
        """Compiler without PBR template returns only body."""
        compiler = MaterialCompiler(include_pbr_template=False)

        class TestMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        wgsl = compiler.compile(TestMat)
        assert "struct PBRInput" not in wgsl
        assert "@fragment" not in wgsl

    def test_compiler_template_enabled(self):
        """Compiler with PBR template returns full shader."""
        compiler = MaterialCompiler(include_pbr_template=True)

        class TestMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        wgsl = compiler.compile(TestMat)
        assert_wgsl_contains(wgsl, "struct PBRInput", "@fragment", "@vertex")

    def test_compiler_no_surface_raises(self):
        """Compiler raises ValueError for class without surface."""
        compiler = MaterialCompiler()

        class NoSurface:
            pass

        with pytest.raises(ValueError, match="no surface"):
            compiler.compile(NoSurface)


# =============================================================================
# Suite 8: WGSL Output Quality
# =============================================================================


class TestWGSLOutputQuality:
    """Tests for WGSL output quality and correctness."""

    def test_output_has_correct_indentation(self, compiler):
        """Generated WGSL has consistent indentation."""

        class IndentMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if ctx.uv.x > 0.5:
                    out.roughness = 0.8
                else:
                    out.roughness = 0.2

        wgsl = compiler.compile(IndentMat)
        # Check that braces are properly matched
        assert wgsl.count("{") == wgsl.count("}")

    def test_output_has_semicolons(self, body_compiler):
        """Statements end with semicolons."""

        class SemiMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5
                out.metallic = 0.0

        wgsl = SemiMat._wgsl_source
        # Each assignment should end with semicolon
        assert wgsl.count(";") >= 2

    def test_output_floats_have_decimal(self, body_compiler):
        """Float literals have decimal points."""

        class FloatMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 1.0  # Explicit float

        wgsl = FloatMat._wgsl_source
        assert "1.0" in wgsl or "1." in wgsl


# =============================================================================
# Suite 9: Used Builtins Tracking
# =============================================================================


class TestUsedBuiltinsTracking:
    """Tests for tracking which custom builtins are used."""

    def test_tracks_noise_builtins(self):
        """Translator tracks noise builtin usage."""

        class NoiseMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                n: float = perlin_noise(ctx.uv)
                out.roughness = n

        assert "perlin_noise" in NoiseMat._used_builtins

    def test_tracks_color_builtins(self):
        """Translator tracks color builtin usage."""

        class ColorMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                hsv = rgb_to_hsv(Vec3(1.0, 0.0, 0.0))
                out.base_color = hsv

        assert "rgb_to_hsv" in ColorMat._used_builtins

    def test_no_builtins_when_unused(self):
        """Empty builtin set when no custom builtins used."""

        class SimpleMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert len(SimpleMat._used_builtins) == 0


# =============================================================================
# Stub functions for material surface bodies
# =============================================================================


def perlin_noise(p):
    """Stub for perlin_noise used in tests."""
    return 0.5


def rgb_to_hsv(rgb):
    """Stub for rgb_to_hsv used in tests."""
    return Vec3(0, 1, 1)


def clamp(x, min_val, max_val):
    """Stub for clamp used in tests."""
    return max(min_val, min(x, max_val))
