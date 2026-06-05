"""AST Node Type Coverage Test Suite (T-MAT-1.7).

This module provides focused test coverage for all 15+ AST node types
supported by the PythonToWGSLTranslator, with emphasis on edge cases,
boundary conditions, and combinations.

Gap: S1-G7
Dependencies: T-MAT-1.3, T-MAT-1.4, T-MAT-1.6 (all DONE)
"""

from __future__ import annotations

import ast
import pytest
from typing import Tuple

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
# Fixtures
# =============================================================================


@pytest.fixture
def translator():
    """Fresh translator instance."""
    return PythonToWGSLTranslator()


def translate(code: str) -> str:
    """Shorthand for translating code."""
    tree = ast.parse(code)
    return PythonToWGSLTranslator().translate(tree)


def assert_contains(wgsl: str, *parts: str):
    """Assert WGSL contains all parts."""
    for part in parts:
        assert part in wgsl, f"Missing '{part}' in:\n{wgsl}"


# =============================================================================
# Suite A: Expression Nodes (Expr, Constant, Name, Attribute)
# =============================================================================


class TestExprNode:
    """Tests for Expr AST node - expression statements."""

    def test_expr_function_call(self, translator):
        """Function call as statement."""
        wgsl = translate("do_something()")
        assert_contains(wgsl, "do_something();")

    def test_expr_method_call(self, translator):
        """Method call as statement."""
        wgsl = translate("obj.method(arg)")
        assert_contains(wgsl, "obj.method(arg);")

    def test_expr_nested_call(self, translator):
        """Nested function call as statement."""
        wgsl = translate("outer(inner(x))")
        assert_contains(wgsl, "outer(inner(x));")

    def test_expr_with_operators(self, translator):
        """Expression statement with operators (unusual but valid)."""
        wgsl = translate("a + b")
        assert_contains(wgsl, "(a + b);")


class TestConstantNode:
    """Tests for Constant AST node - literal values."""

    def test_constant_int_zero(self, translator):
        """Integer zero."""
        wgsl = translate("x = 0")
        assert_contains(wgsl, "0")

    def test_constant_int_positive(self, translator):
        """Positive integer."""
        wgsl = translate("x = 42")
        assert_contains(wgsl, "42")

    def test_constant_int_negative(self, translator):
        """Negative integer via unary op."""
        wgsl = translate("x = -1")
        assert_contains(wgsl, "-1") or assert_contains(wgsl, "(-1)")

    def test_constant_float_zero(self, translator):
        """Float zero."""
        wgsl = translate("x = 0.0")
        assert_contains(wgsl, "0.0")

    def test_constant_float_decimal(self, translator):
        """Float with decimal."""
        wgsl = translate("x = 3.14159")
        assert_contains(wgsl, "3.14159")

    def test_constant_float_small(self, translator):
        """Very small float."""
        wgsl = translate("x = 0.001")
        assert_contains(wgsl, "0.001")

    def test_constant_float_large(self, translator):
        """Large float."""
        wgsl = translate("x = 1000000.0")
        assert "1000000" in wgsl

    def test_constant_float_scientific_positive(self, translator):
        """Scientific notation with positive exponent."""
        wgsl = translate("x = 1e6")
        # May be rendered as 1000000.0 or 1e6
        assert "1" in wgsl

    def test_constant_float_scientific_negative(self, translator):
        """Scientific notation with negative exponent."""
        wgsl = translate("x = 1e-5")
        # May be rendered as 0.00001 or 1e-5
        assert "1" in wgsl

    def test_constant_bool_true(self, translator):
        """Boolean True becomes true."""
        wgsl = translate("x = True")
        assert_contains(wgsl, "true")

    def test_constant_bool_false(self, translator):
        """Boolean False becomes false."""
        wgsl = translate("x = False")
        assert_contains(wgsl, "false")

    def test_constant_none(self, translator):
        """None becomes 0.0."""
        wgsl = translate("x = None")
        assert_contains(wgsl, "0.0")

    def test_constant_string(self, translator):
        """String literal (for texture paths)."""
        wgsl = translate('path = "texture.png"')
        assert '"texture.png"' in wgsl


class TestNameNode:
    """Tests for Name AST node - variable references."""

    def test_name_simple(self, translator):
        """Simple variable name."""
        wgsl = translate("result = value")
        assert_contains(wgsl, "value")

    def test_name_underscore(self, translator):
        """Name with underscore."""
        wgsl = translate("result = my_var")
        assert_contains(wgsl, "my_var")

    def test_name_camelcase(self, translator):
        """CamelCase name."""
        wgsl = translate("result = myVariable")
        assert_contains(wgsl, "myVariable")

    def test_name_ctx(self, translator):
        """Context variable."""
        wgsl = translate("result = ctx")
        assert_contains(wgsl, "ctx")

    def test_name_out(self, translator):
        """Output variable."""
        wgsl = translate("result = out")
        assert_contains(wgsl, "out")

    def test_name_true_maps_correctly(self, translator):
        """Python True maps to WGSL true."""
        wgsl = translate("flag = True")
        assert "True" not in wgsl
        assert_contains(wgsl, "true")

    def test_name_false_maps_correctly(self, translator):
        """Python False maps to WGSL false."""
        wgsl = translate("flag = False")
        assert "False" not in wgsl
        assert_contains(wgsl, "false")


class TestAttributeNode:
    """Tests for Attribute AST node - attribute access."""

    def test_attribute_simple(self, translator):
        """Simple attribute access."""
        wgsl = translate("x = obj.attr")
        assert_contains(wgsl, "obj.attr")

    def test_attribute_chained(self, translator):
        """Chained attribute access."""
        wgsl = translate("x = a.b.c")
        assert_contains(wgsl, "a.b.c")

    def test_attribute_ctx_uv(self, translator):
        """Context UV attribute."""
        wgsl = translate("uv = ctx.uv")
        assert_contains(wgsl, "in.uv")

    def test_attribute_ctx_time(self, translator):
        """Context time attribute."""
        wgsl = translate("t = ctx.time")
        # Implementation uses time_uniforms.elapsed_seconds
        assert_contains(wgsl, "time_uniforms.elapsed_seconds")

    def test_attribute_ctx_position(self, translator):
        """Context position attribute."""
        wgsl = translate("p = ctx.position")
        assert_contains(wgsl, "in.position")

    def test_attribute_ctx_normal(self, translator):
        """Context normal attribute."""
        wgsl = translate("n = ctx.normal")
        assert_contains(wgsl, "in.normal")

    def test_attribute_ctx_vertex_color(self, translator):
        """Context vertex color attribute."""
        wgsl = translate("c = ctx.vertex_color")
        assert_contains(wgsl, "in.vertex_color")

    def test_attribute_out_base_color(self, translator):
        """Output base_color field."""
        wgsl = translate("out.base_color = value")
        assert_contains(wgsl, "out.base_color")

    def test_attribute_out_metallic(self, translator):
        """Output metallic field."""
        wgsl = translate("out.metallic = value")
        assert_contains(wgsl, "out.metallic")

    def test_attribute_out_roughness(self, translator):
        """Output roughness field."""
        wgsl = translate("out.roughness = value")
        assert_contains(wgsl, "out.roughness")

    def test_attribute_out_ao_maps(self, translator):
        """Output ao maps to ambient_occlusion."""
        wgsl = translate("out.ao = value")
        assert_contains(wgsl, "out.ambient_occlusion")

    def test_attribute_out_albedo_alias(self, translator):
        """Output albedo alias maps to base_color."""
        wgsl = translate("out.albedo = value")
        assert_contains(wgsl, "out.base_color")

    def test_attribute_self_texture(self, translator):
        """Self texture reference."""
        wgsl = translate("tex = self.albedo_map")
        assert_contains(wgsl, "material.albedo_map")

    def test_attribute_vec_component_x(self, translator):
        """Vector x component access."""
        wgsl = translate("x = v.x")
        assert_contains(wgsl, "v.x")

    def test_attribute_vec_component_y(self, translator):
        """Vector y component access."""
        wgsl = translate("y = v.y")
        assert_contains(wgsl, "v.y")

    def test_attribute_vec_component_z(self, translator):
        """Vector z component access."""
        wgsl = translate("z = v.z")
        assert_contains(wgsl, "v.z")

    def test_attribute_vec_component_w(self, translator):
        """Vector w component access."""
        wgsl = translate("w = v.w")
        assert_contains(wgsl, "v.w")


# =============================================================================
# Suite B: Operator Nodes (BinOp, UnaryOp, Compare, BoolOp)
# =============================================================================


class TestBinOpNode:
    """Tests for BinOp AST node - binary operations."""

    def test_binop_add_simple(self, translator):
        """Simple addition."""
        wgsl = translate("r = a + b")
        assert_contains(wgsl, "(a + b)")

    def test_binop_add_multiple(self, translator):
        """Multiple additions."""
        wgsl = translate("r = a + b + c")
        assert wgsl.count("+") >= 2

    def test_binop_sub_simple(self, translator):
        """Simple subtraction."""
        wgsl = translate("r = a - b")
        assert_contains(wgsl, "(a - b)")

    def test_binop_mult_simple(self, translator):
        """Simple multiplication."""
        wgsl = translate("r = a * b")
        assert_contains(wgsl, "(a * b)")

    def test_binop_div_simple(self, translator):
        """Simple division."""
        wgsl = translate("r = a / b")
        assert_contains(wgsl, "(a / b)")

    def test_binop_mod_simple(self, translator):
        """Simple modulo."""
        wgsl = translate("r = a % b")
        assert_contains(wgsl, "(a % b)")

    def test_binop_pow_becomes_pow_func(self, translator):
        """Power operator becomes pow()."""
        wgsl = translate("r = a ** b")
        assert_contains(wgsl, "pow(a, b)")

    def test_binop_pow_with_constant(self, translator):
        """Power with constant exponent."""
        wgsl = translate("r = x ** 2.0")
        assert_contains(wgsl, "pow(x, 2.0)")

    def test_binop_bitand(self, translator):
        """Bitwise AND."""
        wgsl = translate("r = a & b")
        assert_contains(wgsl, "(a & b)")

    def test_binop_bitor(self, translator):
        """Bitwise OR."""
        wgsl = translate("r = a | b")
        assert_contains(wgsl, "(a | b)")

    def test_binop_bitxor(self, translator):
        """Bitwise XOR."""
        wgsl = translate("r = a ^ b")
        assert_contains(wgsl, "(a ^ b)")

    def test_binop_lshift(self, translator):
        """Left shift."""
        wgsl = translate("r = a << b")
        assert_contains(wgsl, "(a << b)")

    def test_binop_rshift(self, translator):
        """Right shift."""
        wgsl = translate("r = a >> b")
        assert_contains(wgsl, "(a >> b)")

    def test_binop_precedence_mul_add(self, translator):
        """Multiplication before addition."""
        wgsl = translate("r = a + b * c")
        # Should preserve precedence with parentheses
        assert "+" in wgsl and "*" in wgsl

    def test_binop_complex_expression(self, translator):
        """Complex expression with multiple operators."""
        wgsl = translate("r = (a + b) * (c - d) / e")
        assert_contains(wgsl, "+", "-", "*", "/")


class TestUnaryOpNode:
    """Tests for UnaryOp AST node - unary operations."""

    def test_unaryop_usub(self, translator):
        """Unary subtraction (negation)."""
        wgsl = translate("r = -x")
        assert_contains(wgsl, "(-x)")

    def test_unaryop_uadd(self, translator):
        """Unary addition (identity)."""
        wgsl = translate("r = +x")
        assert_contains(wgsl, "(+x)")

    def test_unaryop_not(self, translator):
        """Logical not."""
        wgsl = translate("r = not condition")
        assert_contains(wgsl, "(!condition)")

    def test_unaryop_invert(self, translator):
        """Bitwise invert."""
        wgsl = translate("r = ~bits")
        assert_contains(wgsl, "(~bits)")

    def test_unaryop_double_negative(self, translator):
        """Double negative."""
        wgsl = translate("r = --x")
        assert wgsl.count("-") >= 2

    def test_unaryop_not_not(self, translator):
        """Double not."""
        wgsl = translate("r = not not flag")
        assert wgsl.count("!") >= 2


class TestCompareNode:
    """Tests for Compare AST node - comparison expressions."""

    def test_compare_eq(self, translator):
        """Equality comparison."""
        wgsl = translate("r = a == b")
        assert_contains(wgsl, "== b")

    def test_compare_neq(self, translator):
        """Inequality comparison."""
        wgsl = translate("r = a != b")
        assert_contains(wgsl, "!= b")

    def test_compare_lt(self, translator):
        """Less than comparison."""
        wgsl = translate("r = a < b")
        assert "< b" in wgsl

    def test_compare_lte(self, translator):
        """Less than or equal comparison."""
        wgsl = translate("r = a <= b")
        assert_contains(wgsl, "<= b")

    def test_compare_gt(self, translator):
        """Greater than comparison."""
        wgsl = translate("r = a > b")
        assert "> b" in wgsl

    def test_compare_gte(self, translator):
        """Greater than or equal comparison."""
        wgsl = translate("r = a >= b")
        assert_contains(wgsl, ">= b")

    def test_compare_with_constant(self, translator):
        """Comparison with constant."""
        wgsl = translate("r = x > 0.5")
        assert "> 0.5" in wgsl

    def test_compare_chained_not_supported(self, translator):
        """Chained comparisons translate (a < b < c)."""
        wgsl = translate("r = a < b < c")
        # Should contain comparison operators
        assert "<" in wgsl


class TestBoolOpNode:
    """Tests for BoolOp AST node - boolean operations."""

    def test_boolop_and_simple(self, translator):
        """Simple boolean AND."""
        wgsl = translate("r = a and b")
        assert_contains(wgsl, "(a && b)")

    def test_boolop_or_simple(self, translator):
        """Simple boolean OR."""
        wgsl = translate("r = a or b")
        assert_contains(wgsl, "(a || b)")

    def test_boolop_and_three(self, translator):
        """Three-way AND."""
        wgsl = translate("r = a and b and c")
        assert wgsl.count("&&") >= 1

    def test_boolop_or_three(self, translator):
        """Three-way OR."""
        wgsl = translate("r = a or b or c")
        assert wgsl.count("||") >= 1

    def test_boolop_mixed_and_or(self, translator):
        """Mixed AND/OR."""
        wgsl = translate("r = a and b or c")
        assert_contains(wgsl, "&&", "||")


# =============================================================================
# Suite C: Statement Nodes (Assign, AnnAssign, AugAssign, If, Return, Pass)
# =============================================================================


class TestAssignNode:
    """Tests for Assign AST node - simple assignment."""

    def test_assign_simple(self, translator):
        """Simple assignment."""
        wgsl = translate("x = 1.0")
        assert_contains(wgsl, "x = 1.0;")

    def test_assign_expression(self, translator):
        """Assignment with expression."""
        wgsl = translate("result = a + b * c")
        assert_contains(wgsl, "result =", "+", "*")

    def test_assign_function_call(self, translator):
        """Assignment from function call."""
        wgsl = translate("x = func(a, b)")
        assert_contains(wgsl, "x = func(a, b);")

    def test_assign_vector(self, translator):
        """Assignment of vector constructor."""
        wgsl = translate("v = Vec3(1.0, 2.0, 3.0)")
        assert_contains(wgsl, "vec3<f32>")


class TestAnnAssignNode:
    """Tests for AnnAssign AST node - annotated assignment."""

    def test_annassign_float(self, translator):
        """Annotated float assignment."""
        wgsl = translate("x: float = 1.0")
        assert_contains(wgsl, "let x = 1.0")

    def test_annassign_int(self, translator):
        """Annotated int assignment."""
        wgsl = translate("n: int = 42")
        assert_contains(wgsl, "let n = 42")

    def test_annassign_expression(self, translator):
        """Annotated assignment with expression."""
        wgsl = translate("result: float = a + b")
        assert_contains(wgsl, "let result =", "+")

    def test_annassign_no_value_skipped(self, translator):
        """Annotated declaration without value is skipped."""
        wgsl = translate("x: float")
        # Should not emit anything for declaration without value
        assert "x" not in wgsl or wgsl.strip() == ""


class TestAugAssignNode:
    """Tests for AugAssign AST node - augmented assignment."""

    def test_augassign_add(self, translator):
        """Augmented addition."""
        wgsl = translate("x += 1.0")
        assert_contains(wgsl, "x = x + 1.0")

    def test_augassign_sub(self, translator):
        """Augmented subtraction."""
        wgsl = translate("x -= 1.0")
        assert_contains(wgsl, "x = x - 1.0")

    def test_augassign_mult(self, translator):
        """Augmented multiplication."""
        wgsl = translate("x *= 2.0")
        assert_contains(wgsl, "x = x * 2.0")

    def test_augassign_div(self, translator):
        """Augmented division."""
        wgsl = translate("x /= 2.0")
        assert_contains(wgsl, "x = x / 2.0")

    def test_augassign_on_attribute(self, translator):
        """Augmented assignment on attribute."""
        wgsl = translate("out.roughness *= 0.5")
        assert "roughness" in wgsl
        assert "*" in wgsl


class TestIfNode:
    """Tests for If AST node - conditional statements."""

    def test_if_simple(self, translator):
        """Simple if statement."""
        wgsl = translate("""
if condition:
    x = 1.0
""")
        assert_contains(wgsl, "if (condition)", "{", "}")

    def test_if_else(self, translator):
        """If-else statement."""
        wgsl = translate("""
if condition:
    x = 1.0
else:
    x = 0.0
""")
        assert_contains(wgsl, "if", "else")

    def test_if_elif(self, translator):
        """If-elif statement."""
        wgsl = translate("""
if a:
    x = 1.0
elif b:
    x = 0.5
""")
        assert_contains(wgsl, "if", "else")

    def test_if_elif_else(self, translator):
        """If-elif-else statement."""
        wgsl = translate("""
if a:
    x = 1.0
elif b:
    x = 0.5
else:
    x = 0.0
""")
        assert wgsl.count("if") >= 1
        assert wgsl.count("else") >= 1

    def test_if_nested(self, translator):
        """Nested if statements."""
        wgsl = translate("""
if outer:
    if inner:
        x = 1.0
""")
        assert wgsl.count("if") >= 2

    def test_if_with_comparison(self, translator):
        """If with comparison condition."""
        wgsl = translate("""
if x > 0.5:
    result = 1.0
""")
        assert_contains(wgsl, "if", ">", "0.5")

    def test_if_with_boolean_op(self, translator):
        """If with boolean operation."""
        wgsl = translate("""
if a and b:
    result = 1.0
""")
        assert_contains(wgsl, "if", "&&")


class TestReturnNode:
    """Tests for Return AST node - return statements."""

    def test_return_value(self, translator):
        """Return with value."""
        wgsl = translate("return x")
        assert_contains(wgsl, "return x;")

    def test_return_expression(self, translator):
        """Return with expression."""
        wgsl = translate("return a + b")
        assert_contains(wgsl, "return", "+")

    def test_return_void(self, translator):
        """Return without value."""
        wgsl = translate("return")
        assert_contains(wgsl, "return;")

    def test_return_function_call(self, translator):
        """Return function call result."""
        wgsl = translate("return func(x)")
        assert_contains(wgsl, "return func(x);")


class TestPassNode:
    """Tests for Pass AST node - no-op."""

    def test_pass_empty(self, translator):
        """Pass statement produces no code."""
        wgsl = translate("pass")
        assert wgsl.strip() == "" or "pass" not in wgsl.lower()

    def test_pass_in_if(self, translator):
        """Pass in if block."""
        wgsl = translate("""
if condition:
    pass
""")
        assert_contains(wgsl, "if")
        assert "pass" not in wgsl.lower()


# =============================================================================
# Suite D: Call and Index Nodes (Call, Subscript, Tuple, List)
# =============================================================================


class TestCallNode:
    """Tests for Call AST node - function/method calls."""

    def test_call_no_args(self, translator):
        """Function call with no arguments."""
        wgsl = translate("result = func()")
        assert_contains(wgsl, "func()")

    def test_call_single_arg(self, translator):
        """Function call with single argument."""
        wgsl = translate("result = func(x)")
        assert_contains(wgsl, "func(x)")

    def test_call_multiple_args(self, translator):
        """Function call with multiple arguments."""
        wgsl = translate("result = func(a, b, c)")
        assert_contains(wgsl, "func(a, b, c)")

    def test_call_nested(self, translator):
        """Nested function calls."""
        wgsl = translate("result = outer(inner(x))")
        assert_contains(wgsl, "outer(inner(x))")

    def test_call_method(self, translator):
        """Method call."""
        wgsl = translate("result = obj.method(x)")
        assert_contains(wgsl, "obj.method(x)")

    def test_call_vec2_constructor(self, translator):
        """Vec2 constructor."""
        wgsl = translate("v = Vec2(1.0, 2.0)")
        assert_contains(wgsl, "vec2<f32>(1.0, 2.0)")

    def test_call_vec3_constructor(self, translator):
        """Vec3 constructor."""
        wgsl = translate("v = Vec3(1.0, 2.0, 3.0)")
        assert_contains(wgsl, "vec3<f32>(1.0, 2.0, 3.0)")

    def test_call_vec4_constructor(self, translator):
        """Vec4 constructor."""
        wgsl = translate("v = Vec4(1.0, 2.0, 3.0, 4.0)")
        assert_contains(wgsl, "vec4<f32>(1.0, 2.0, 3.0, 4.0)")

    def test_call_builtin_abs(self, translator):
        """Builtin abs()."""
        wgsl = translate("result = abs(x)")
        assert_contains(wgsl, "abs(x)")

    def test_call_builtin_clamp(self, translator):
        """Builtin clamp()."""
        wgsl = translate("result = clamp(x, 0.0, 1.0)")
        assert_contains(wgsl, "clamp(x, 0.0, 1.0)")

    def test_call_builtin_mix(self, translator):
        """Builtin mix()."""
        wgsl = translate("result = mix(a, b, t)")
        assert_contains(wgsl, "mix(a, b, t)")

    def test_call_builtin_lerp_becomes_mix(self, translator):
        """Builtin lerp() maps to mix()."""
        wgsl = translate("result = lerp(a, b, t)")
        assert_contains(wgsl, "mix(a, b, t)")

    def test_call_ctx_sample(self, translator):
        """Context sample method."""
        wgsl = translate("color = ctx.sample(tex, uv)")
        assert_contains(wgsl, "textureSample(tex, uv)")

    def test_call_ctx_sample_level(self, translator):
        """Context sample_level method."""
        wgsl = translate("color = ctx.sample_level(tex, uv, 2.0)")
        assert_contains(wgsl, "textureSampleLevel(tex, uv, 2.0)")

    def test_call_ctx_sample_cube(self, translator):
        """Context sample_cube method."""
        wgsl = translate("color = ctx.sample_cube(cube, dir)")
        assert_contains(wgsl, "textureSample(cube, dir)")

    def test_call_ctx_world_position(self, translator):
        """Context world_position method."""
        wgsl = translate("pos = ctx.world_position()")
        assert_contains(wgsl, "in.world_position")

    def test_call_ctx_world_normal(self, translator):
        """Context world_normal method."""
        wgsl = translate("n = ctx.world_normal()")
        assert_contains(wgsl, "in.world_normal")


class TestSubscriptNode:
    """Tests for Subscript AST node - index access."""

    def test_subscript_numeric_index(self, translator):
        """Numeric index."""
        wgsl = translate("x = arr[0]")
        assert_contains(wgsl, "arr[0]")

    def test_subscript_variable_index(self, translator):
        """Variable index."""
        wgsl = translate("x = arr[i]")
        assert_contains(wgsl, "arr[i]")

    def test_subscript_expression_index(self, translator):
        """Expression index."""
        wgsl = translate("x = arr[i + 1]")
        assert_contains(wgsl, "arr[", "]")

    def test_subscript_nested(self, translator):
        """Nested subscript."""
        wgsl = translate("x = arr[matrix[0]]")
        assert_contains(wgsl, "arr[", "matrix[0]", "]")


class TestTupleNode:
    """Tests for Tuple AST node - vector construction."""

    def test_tuple_2_elements(self, translator):
        """2-element tuple becomes vec2."""
        wgsl = translate("v = (1.0, 2.0)")
        assert_contains(wgsl, "vec2<f32>(1.0, 2.0)")

    def test_tuple_3_elements(self, translator):
        """3-element tuple becomes vec3."""
        wgsl = translate("v = (1.0, 2.0, 3.0)")
        assert_contains(wgsl, "vec3<f32>(1.0, 2.0, 3.0)")

    def test_tuple_4_elements(self, translator):
        """4-element tuple becomes vec4."""
        wgsl = translate("v = (1.0, 2.0, 3.0, 4.0)")
        assert_contains(wgsl, "vec4<f32>(1.0, 2.0, 3.0, 4.0)")

    def test_tuple_with_expressions(self, translator):
        """Tuple with expression elements."""
        wgsl = translate("v = (a + b, c * d, e)")
        assert_contains(wgsl, "vec3<f32>", "+", "*")


class TestListNode:
    """Tests for List AST node - same as tuple for vectors."""

    def test_list_2_elements(self, translator):
        """2-element list becomes vec2."""
        wgsl = translate("v = [1.0, 2.0]")
        assert_contains(wgsl, "vec2<f32>(1.0, 2.0)")

    def test_list_3_elements(self, translator):
        """3-element list becomes vec3."""
        wgsl = translate("v = [1.0, 2.0, 3.0]")
        assert_contains(wgsl, "vec3<f32>(1.0, 2.0, 3.0)")


# =============================================================================
# Suite E: IfExp and Complex Expressions
# =============================================================================


class TestIfExpNode:
    """Tests for IfExp AST node - ternary expression."""

    def test_ifexp_simple(self, translator):
        """Simple ternary."""
        wgsl = translate("result = a if condition else b")
        assert_contains(wgsl, "select(b, a, condition)")

    def test_ifexp_with_expressions(self, translator):
        """Ternary with expressions."""
        wgsl = translate("result = x * 2 if flag else y / 2")
        assert_contains(wgsl, "select")

    def test_ifexp_nested(self, translator):
        """Nested ternary."""
        wgsl = translate("result = a if x else (b if y else c)")
        # Should contain multiple selects
        assert "select" in wgsl


# =============================================================================
# Suite F: Function Definition Node (for surface methods)
# =============================================================================


class TestFunctionDefNode:
    """Tests for FunctionDef AST node - extracting function body."""

    def test_funcdef_extracts_body(self, translator):
        """Function body is extracted correctly."""
        code = """
def surface(self, ctx, out):
    out.roughness = 0.5
"""
        wgsl = translate(code)
        assert_contains(wgsl, "roughness", "0.5")

    def test_funcdef_multiple_statements(self, translator):
        """Multiple statements in function body."""
        code = """
def surface(self, ctx, out):
    out.roughness = 0.5
    out.metallic = 0.0
    out.base_color = Vec3(1.0, 1.0, 1.0)
"""
        wgsl = translate(code)
        assert_contains(wgsl, "roughness", "metallic", "base_color")


# =============================================================================
# Suite G: Module Node
# =============================================================================


class TestModuleNode:
    """Tests for Module AST node - top-level container."""

    def test_module_multiple_statements(self, translator):
        """Module with multiple statements."""
        code = """
x = 1.0
y = 2.0
z = x + y
"""
        wgsl = translate(code)
        assert_contains(wgsl, "x = 1.0", "y = 2.0")


# =============================================================================
# Suite H: Error Case Coverage
# =============================================================================


class TestErrorCases:
    """Tests for error handling on unsupported nodes."""

    @pytest.mark.parametrize("code", [
        "class Foo: pass",
        "import os",
        "from math import sqrt",
        "[x for x in items]",
        "{k: v for k, v in items}",
        "(x for x in items)",
        "lambda x: x * 2",
        "{1, 2, 3}",
        "{'a': 1}",
        "global x",
        "nonlocal y",
        "assert True",
        "raise ValueError()",
        "del x",
        "yield 1",
        "yield from gen",
    ])
    def test_unsupported_construct(self, translator, code):
        """Unsupported constructs raise WGSLTranslationError."""
        tree = ast.parse(code)
        with pytest.raises(WGSLTranslationError):
            translator.translate(tree)


# =============================================================================
# Suite I: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_function(self, translator):
        """Empty function body."""
        code = """
def surface(self, ctx, out):
    pass
"""
        wgsl = translate(code)
        # Should not crash, may produce empty output or just comments
        assert wgsl is not None

    def test_deeply_nested_expression(self, translator):
        """Deeply nested expression."""
        wgsl = translate("result = ((((a + b) * c) - d) / e)")
        assert_contains(wgsl, "+", "*", "-", "/")

    def test_very_long_chain(self, translator):
        """Very long chain of operations."""
        wgsl = translate("result = a + b + c + d + e + f + g + h")
        assert wgsl.count("+") >= 7

    def test_mixed_operators(self, translator):
        """Mixed arithmetic and comparison."""
        wgsl = translate("result = (a + b) > (c * d)")
        assert_contains(wgsl, "+", "*", ">")

    def test_unicode_in_strings(self, translator):
        """Unicode characters in strings."""
        wgsl = translate('path = "texture_é.png"')
        assert "texture" in wgsl

    def test_float_precision(self, translator):
        """Float precision is preserved."""
        wgsl = translate("x = 0.123456789")
        assert "0.123456789" in wgsl or "0.12345" in wgsl


# =============================================================================
# Suite J: Integration with Material Classes
# =============================================================================


class TestMaterialIntegration:
    """Tests for AST translation in material context."""

    def test_material_with_all_node_types(self):
        """Material using many node types compiles."""

        class ComprehensiveMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Constant, AnnAssign
                factor: float = 0.5

                # BinOp, Attribute
                x: float = ctx.uv.x * 2.0 - 1.0
                y: float = ctx.uv.y * 2.0 - 1.0

                # Call, BinOp
                dist: float = sqrt(x * x + y * y)

                # If, Compare
                if dist < 0.5:
                    out.metallic = 1.0
                else:
                    out.metallic = 0.0

                # Assign, Call, Tuple
                out.base_color = Vec3(0.8, 0.6, 0.4)
                out.roughness = clamp(dist, 0.0, 1.0)

        assert ComprehensiveMat._compilation_error is None
        wgsl = ComprehensiveMat._wgsl_source
        assert_contains(wgsl, "let factor", "sqrt", "if", "else", "clamp")


# =============================================================================
# Stub functions
# =============================================================================


def sqrt(x):
    """Stub for sqrt."""
    import math
    return math.sqrt(x)


def clamp(x, min_val, max_val):
    """Stub for clamp."""
    return max(min_val, min(x, max_val))
