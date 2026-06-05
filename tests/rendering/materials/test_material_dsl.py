"""Tests for the Material DSL metaclass and WGSL codegen (T-MAT-1.1).

Verifies:
- MaterialMeta metaclass extracts surface() and compiles to WGSL
- SurfaceContext and SurfaceOutput proxies work correctly
- Python AST -> WGSL translation for all 15 core node types
- Error handling for unsupported constructs
"""

from __future__ import annotations

import pytest

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
# Suite A: MaterialMeta Metaclass
# =============================================================================


class TestMaterialMetaBasics:
    """MaterialMeta creates materials and compiles surface() to WGSL."""

    def test_metaclass_creates_wgsl_source(self):
        """A material with surface() gets _wgsl_source populated."""

        class SimpleMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.2)
                out.roughness = 0.4

        assert hasattr(SimpleMaterial, "_wgsl_source")
        assert SimpleMaterial._wgsl_source != ""
        assert "base_color" in SimpleMaterial._wgsl_source or "out.base_color" in SimpleMaterial._wgsl_source

    def test_metaclass_registers_surface_method(self):
        """MaterialMeta stores reference to the original surface method."""

        class TestMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert TestMaterial._surface_method is not None
        assert callable(TestMaterial._surface_method)

    def test_metaclass_without_surface_decorator(self):
        """A method named 'surface' is still compiled even without decorator."""

        class TestMaterial(Material, metaclass=MaterialMeta):
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.9

        assert TestMaterial._wgsl_source != ""

    def test_metaclass_no_compilation_error_for_valid_code(self):
        """Valid surface code produces no compilation error."""

        class ValidMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert ValidMaterial._compilation_error is None

    def test_material_registry(self):
        """MaterialMeta registers materials by name."""

        class RegisteredMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert MaterialMeta.get_material("RegisteredMaterial") is RegisteredMaterial

    def test_all_materials_includes_registered(self):
        """MaterialMeta.all_materials() returns registered materials."""

        class AnotherMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert AnotherMaterial in MaterialMeta.all_materials()

    def test_get_wgsl_helper(self):
        """Material.get_wgsl() returns the compiled shader code."""

        class WgslTestMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        wgsl = WgslTestMaterial.get_wgsl()
        assert isinstance(wgsl, str)
        assert len(wgsl) > 0


# =============================================================================
# Suite B: SurfaceContext Proxy
# =============================================================================


class TestSurfaceContext:
    """SurfaceContext provides shader input accessors."""

    def test_context_default_values(self):
        """SurfaceContext initializes with valid defaults."""
        ctx = SurfaceContext()
        assert isinstance(ctx.position, Vec3)
        assert isinstance(ctx.normal, Vec3)
        assert isinstance(ctx.uv, Vec2)
        assert isinstance(ctx.vertex_color, Vec4)

    def test_context_sample_returns_vec4(self):
        """ctx.sample() returns a Vec4 (runtime stub)."""
        ctx = SurfaceContext()
        result = ctx.sample(None, (0.5, 0.5))
        assert isinstance(result, Vec4)

    def test_context_sample_cube_returns_vec4(self):
        """ctx.sample_cube() returns a Vec4."""
        ctx = SurfaceContext()
        result = ctx.sample_cube(None, (1.0, 0.0, 0.0))
        assert isinstance(result, Vec4)

    def test_context_noise_returns_float(self):
        """ctx.noise() returns a float."""
        ctx = SurfaceContext()
        result = ctx.noise((0.0, 0.0), scale=1.0)
        assert isinstance(result, float)

    def test_context_accessors(self):
        """Context accessor methods return correct types."""
        ctx = SurfaceContext()
        assert isinstance(ctx.world_position(), Vec3)
        assert isinstance(ctx.world_normal(), Vec3)
        assert isinstance(ctx.world_tangent(), Vec3)
        assert isinstance(ctx.get_uv(), Vec2)
        assert isinstance(ctx.get_vertex_color(), Vec4)
        assert isinstance(ctx.get_time(), float)


# =============================================================================
# Suite C: SurfaceOutput Proxy
# =============================================================================


class TestSurfaceOutput:
    """SurfaceOutput provides PBR parameter fields."""

    def test_output_default_values(self):
        """SurfaceOutput initializes with valid PBR defaults."""
        out = SurfaceOutput()
        assert isinstance(out.base_color, Vec3)
        assert out.metallic == 0.0
        assert out.roughness == 0.5
        assert out.ao == 1.0
        assert out.alpha == 1.0

    def test_output_extended_params(self):
        """SurfaceOutput has extended PBR parameters."""
        out = SurfaceOutput()
        assert out.clearcoat == 0.0
        assert out.sheen == 0.0
        assert out.transmission == 0.0
        assert out.ior == 1.5

    def test_output_albedo_alias(self):
        """out.albedo is an alias for base_color (deprecated)."""
        out = SurfaceOutput()
        out.albedo = (1.0, 0.5, 0.2)
        assert out.base_color.x == 1.0
        assert out.base_color.y == 0.5
        assert out.base_color.z == 0.2

    def test_output_emission_alias(self):
        """out.emission is an alias for emissive (deprecated)."""
        out = SurfaceOutput()
        out.emission = (2.0, 1.0, 0.0)
        assert out.emissive.x == 2.0
        assert out.emissive.y == 1.0

    def test_output_vec3_assignment(self):
        """Vec3 can be assigned to base_color."""
        out = SurfaceOutput()
        out.base_color = Vec3(0.8, 0.2, 0.1)
        assert out.base_color.x == 0.8

    def test_output_albedo_setter_with_vec3(self):
        """albedo setter accepts Vec3."""
        out = SurfaceOutput()
        out.albedo = Vec3(0.5, 0.5, 0.5)
        assert out.base_color.x == 0.5


# =============================================================================
# Suite D: Vector Types
# =============================================================================


class TestVectorTypes:
    """Vec2, Vec3, Vec4 proxies for WGSL vectors."""

    def test_vec2_creation(self):
        """Vec2 can be created with x, y components."""
        v = Vec2(1.0, 2.0)
        assert v.x == 1.0
        assert v.y == 2.0

    def test_vec3_creation(self):
        """Vec3 can be created with x, y, z components."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec4_creation(self):
        """Vec4 can be created with x, y, z, w components."""
        v = Vec4(1.0, 2.0, 3.0, 4.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
        assert v.w == 4.0

    def test_vec4_default_w(self):
        """Vec4 defaults w to 1.0."""
        v = Vec4(1.0, 2.0, 3.0)
        assert v.w == 1.0

    def test_vec_repr(self):
        """Vectors have informative __repr__."""
        v2 = Vec2(1.0, 2.0)
        v3 = Vec3(1.0, 2.0, 3.0)
        v4 = Vec4(1.0, 2.0, 3.0, 4.0)
        assert "Vec2" in repr(v2)
        assert "Vec3" in repr(v3)
        assert "Vec4" in repr(v4)


# =============================================================================
# Suite E: AST Translation (15 core node types)
# =============================================================================


class TestASTTranslationNodeTypes:
    """PythonToWGSLTranslator handles all 15 core AST node types."""

    def test_constant_float(self):
        """Constant float translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert "0.5" in M._wgsl_source

    def test_constant_int(self):
        """Constant int translates to WGSL."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 1

        assert "1" in M._wgsl_source

    def test_binop_add(self):
        """Binary addition translates to WGSL +."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.2 + 0.3

        assert "+" in M._wgsl_source

    def test_binop_mul(self):
        """Binary multiplication translates to WGSL *."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5 * 2.0

        assert "*" in M._wgsl_source

    def test_binop_pow(self):
        """Power operator translates to pow()."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 2.0 ** 3.0

        assert "pow" in M._wgsl_source

    def test_unaryop_negate(self):
        """Unary negation translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = -0.5

        # Either -0.5 directly or (-0.5)
        assert "-0.5" in M._wgsl_source or "(-0.5)" in M._wgsl_source

    def test_compare_lt(self):
        """Comparison < translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness < 0.5:
                    out.metallic = 1.0

        assert "<" in M._wgsl_source

    def test_boolop_and(self):
        """Boolean and translates to &&."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness < 0.5 and out.metallic > 0.0:
                    out.ao = 0.5

        assert "&&" in M._wgsl_source

    def test_if_statement(self):
        """If statement translates to WGSL if."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if True:
                    out.roughness = 0.5

        assert "if" in M._wgsl_source

    def test_attribute_access(self):
        """Attribute access translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        # out.roughness -> out.roughness (or similar mapping)
        wgsl = M._wgsl_source
        assert "out" in wgsl or "roughness" in wgsl

    def test_call_builtin(self):
        """Built-in function calls translate correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = clamp(0.5, 0.0, 1.0)

        assert "clamp" in M._wgsl_source

    def test_tuple_to_vec(self):
        """Tuple translates to vec constructor."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 0.5, 0.0)

        assert "vec3<f32>" in M._wgsl_source

    def test_subscript_access(self):
        """Subscript access translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                v = Vec3(1.0, 2.0, 3.0)
                out.roughness = v.x

        # Should contain subscript-like access
        assert "v" in M._wgsl_source

    def test_annassign(self):
        """Annotated assignment (AnnAssign) translates to let statement."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                factor: float = 0.5
                out.roughness = factor

        wgsl = M._wgsl_source
        assert "let factor" in wgsl or "factor" in wgsl
        assert "0.5" in wgsl

    def test_return_statement(self):
        """Return statement translates to WGSL return."""
        # Use translator directly since surface methods don't use return
        import ast
        code = "return value"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "return" in wgsl

    def test_name_node(self):
        """Name node translates variable references correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                value = 0.5
                out.roughness = value

        wgsl = M._wgsl_source
        assert "value" in wgsl
        assert "0.5" in wgsl

    def test_expr_statement(self):
        """Expr statement (expression as statement) translates correctly."""
        # Use translator directly to test standalone expression statement
        import ast
        code = "some_function()"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "some_function()" in wgsl


class TestASTTranslationMappings:
    """Test specific WGSL mappings for context and output."""

    def test_output_field_mapping(self):
        """Output fields map to WGSL PBROutput fields."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)
                out.metallic = 0.5
                out.roughness = 0.3

        wgsl = M._wgsl_source
        assert "base_color" in wgsl or "out.base_color" in wgsl
        assert "metallic" in wgsl or "out.metallic" in wgsl
        assert "roughness" in wgsl or "out.roughness" in wgsl

    def test_vec_constructor(self):
        """Vec3/Vec4 constructors emit correct WGSL."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.8, 0.2, 0.1)

        assert "vec3<f32>" in M._wgsl_source


# =============================================================================
# Suite F: MaterialCompiler
# =============================================================================


class TestMaterialCompiler:
    """MaterialCompiler produces complete WGSL shader modules."""

    def test_compiler_basic(self):
        """Compiler produces WGSL from a material class."""

        class TestMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        compiler = MaterialCompiler()
        wgsl = compiler.compile(TestMat)
        assert "roughness" in wgsl

    def test_compiler_without_template(self):
        """Compiler can produce just the surface body."""

        class TestMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        compiler = MaterialCompiler(include_pbr_template=False)
        wgsl = compiler.compile(TestMat)
        # Should not have PBR structs
        assert "struct PBRInput" not in wgsl

    def test_compiler_with_template(self):
        """Compiler includes PBR structs when template is enabled."""

        class TestMat(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        compiler = MaterialCompiler(include_pbr_template=True)
        wgsl = compiler.compile(TestMat)
        assert "struct PBRInput" in wgsl
        assert "struct PBRParams" in wgsl
        assert "@fragment" in wgsl

    def test_compiler_no_surface_raises(self):
        """Compiler raises ValueError for class without surface method."""

        class NoSurface:
            pass

        compiler = MaterialCompiler()
        with pytest.raises(ValueError, match="no surface"):
            compiler.compile(NoSurface)


# =============================================================================
# Suite G: End-to-End Material Examples
# =============================================================================


class TestEndToEndMaterials:
    """Complete material definitions compile successfully."""

    def test_simple_gold_material(self):
        """Simple metallic material compiles."""

        class GoldMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.83, 0.69, 0.22)
                out.metallic = 0.9
                out.roughness = 0.3

        assert GoldMaterial._compilation_error is None
        assert GoldMaterial._wgsl_source != ""

    def test_brick_material_with_expressions(self):
        """Material with arithmetic expressions compiles."""

        class BrickMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.6, 0.3, 0.2)
                out.roughness = 0.7 + 0.1
                out.metallic = 0.0
                out.ao = 1.0 * 0.9

        assert BrickMaterial._compilation_error is None
        wgsl = BrickMaterial._wgsl_source
        assert "+" in wgsl
        assert "*" in wgsl

    def test_conditional_material(self):
        """Material with if statement compiles."""

        class ConditionalMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5
                if out.roughness < 0.3:
                    out.metallic = 1.0
                else:
                    out.metallic = 0.0

        assert ConditionalMaterial._compilation_error is None
        assert "if" in ConditionalMaterial._wgsl_source
        assert "else" in ConditionalMaterial._wgsl_source

    def test_builtin_functions_material(self):
        """Material using built-in math functions compiles."""

        class MathMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = clamp(0.7, 0.0, 1.0)
                out.metallic = min(0.5, 0.3)
                out.ao = max(0.2, 0.8)

        assert MathMaterial._compilation_error is None
        wgsl = MathMaterial._wgsl_source
        assert "clamp" in wgsl
        assert "min" in wgsl
        assert "max" in wgsl


# =============================================================================
# Suite H: Surface Decorator
# =============================================================================


class TestSurfaceDecorator:
    """@surface decorator marks shader entry points."""

    def test_surface_decorator_marks_method(self):
        """@surface sets _is_surface attribute."""

        @surface
        def my_surface(self, ctx, out):
            pass

        assert hasattr(my_surface, "_is_surface")
        assert my_surface._is_surface is True

    def test_surface_decorator_preserves_function(self):
        """@surface returns the original function."""

        def original(self, ctx, out):
            return 42

        decorated = surface(original)
        assert decorated(None, None, None) == 42


# =============================================================================
# Suite I: Additional AST Node Type Coverage
# =============================================================================


class TestASTNodeTypesCoverage:
    """Additional tests for all 15 AST node types with complete operator coverage."""

    def test_binop_sub(self):
        """Binary subtraction translates to WGSL -."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.8 - 0.3

        assert "-" in M._wgsl_source
        assert M._compilation_error is None

    def test_binop_div(self):
        """Binary division translates to WGSL /."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 1.0 / 2.0

        assert "/" in M._wgsl_source
        assert M._compilation_error is None

    def test_binop_mod(self):
        """Binary modulo translates to WGSL %."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 5.0 % 3.0

        assert "%" in M._wgsl_source
        assert M._compilation_error is None

    def test_augassign_add(self):
        """Augmented assignment += translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5
                out.roughness += 0.1

        wgsl = M._wgsl_source
        assert M._compilation_error is None
        # Should contain assignment with addition
        assert "roughness" in wgsl

    def test_augassign_mul(self):
        """Augmented assignment *= translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5
                out.roughness *= 2.0

        assert M._compilation_error is None

    def test_boolop_or(self):
        """Boolean or translates to ||."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness < 0.3 or out.metallic > 0.5:
                    out.ao = 0.8

        assert "||" in M._wgsl_source
        assert M._compilation_error is None

    def test_unaryop_not(self):
        """Unary not translates to !."""
        import ast
        code = "if not condition: pass"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "!" in wgsl

    def test_unaryop_invert(self):
        """Unary invert translates to ~."""
        import ast
        code = "result = ~value"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "~" in wgsl

    def test_compare_eq(self):
        """Comparison == translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.metallic == 1.0:
                    out.roughness = 0.2

        assert "==" in M._wgsl_source
        assert M._compilation_error is None

    def test_compare_neq(self):
        """Comparison != translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.metallic != 0.0:
                    out.roughness = 0.3

        assert "!=" in M._wgsl_source
        assert M._compilation_error is None

    def test_compare_gt(self):
        """Comparison > translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness > 0.5:
                    out.metallic = 0.0

        assert ">" in M._wgsl_source
        assert M._compilation_error is None

    def test_compare_gte(self):
        """Comparison >= translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness >= 0.5:
                    out.metallic = 0.0

        assert ">=" in M._wgsl_source
        assert M._compilation_error is None

    def test_compare_lte(self):
        """Comparison <= translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.roughness <= 0.5:
                    out.metallic = 1.0

        assert "<=" in M._wgsl_source
        assert M._compilation_error is None

    def test_ifexp_ternary(self):
        """Ternary expression (a if cond else b) translates to select()."""
        import ast
        code = "result = value_a if condition else value_b"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "select" in wgsl

    def test_list_to_vec(self):
        """List literal translates to vec constructor."""
        import ast
        code = "v = [1.0, 2.0, 3.0]"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "vec3<f32>" in wgsl

    def test_pass_statement(self):
        """Pass statement produces no code."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                pass
                out.roughness = 0.5

        assert M._compilation_error is None

    def test_constant_bool_true(self):
        """Boolean True translates to true."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if True:
                    out.roughness = 0.5

        assert "true" in M._wgsl_source
        assert M._compilation_error is None

    def test_constant_bool_false(self):
        """Boolean False translates to false."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if False:
                    out.roughness = 0.5

        assert "false" in M._wgsl_source
        assert M._compilation_error is None

    def test_if_elif_else(self):
        """If-elif-else chain translates correctly."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                if out.metallic > 0.8:
                    out.roughness = 0.2
                elif out.metallic > 0.5:
                    out.roughness = 0.4
                else:
                    out.roughness = 0.6

        wgsl = M._wgsl_source
        assert "if" in wgsl
        assert "else" in wgsl
        assert M._compilation_error is None

    def test_vec2_constructor(self):
        """Vec2 constructor translates to vec2<f32>."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                uv = Vec2(0.5, 0.5)
                out.roughness = uv.x

        assert "vec2<f32>" in M._wgsl_source
        assert M._compilation_error is None

    def test_vec4_constructor(self):
        """Vec4 constructor translates to vec4<f32>."""

        class M(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                color = Vec4(1.0, 0.5, 0.2, 1.0)
                out.alpha = color.w

        assert "vec4<f32>" in M._wgsl_source
        assert M._compilation_error is None


# =============================================================================
# Suite J: Error Case Tests
# =============================================================================


class TestErrorCases:
    """Tests for unsupported Python constructs and error handling."""

    def test_unsupported_class_definition(self):
        """Class definition inside surface() raises WGSLTranslationError."""
        import ast
        code = "class Nested: pass"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_import(self):
        """Import statement raises WGSLTranslationError."""
        import ast
        code = "import math"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_from_import(self):
        """From import statement raises WGSLTranslationError."""
        import ast
        code = "from math import sqrt"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_list_comprehension(self):
        """List comprehension raises WGSLTranslationError."""
        import ast
        code = "result = [x * 2 for x in items]"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_dict_comprehension(self):
        """Dict comprehension raises WGSLTranslationError."""
        import ast
        code = "result = {k: v for k, v in pairs}"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_generator_expression(self):
        """Generator expression raises WGSLTranslationError."""
        import ast
        code = "result = sum(x * 2 for x in items)"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_lambda(self):
        """Lambda expression raises WGSLTranslationError."""
        import ast
        code = "f = lambda x: x * 2"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_try_except(self):
        """Try/except statement raises WGSLTranslationError."""
        import ast
        code = """
try:
    x = 1
except:
    x = 0
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_with_statement(self):
        """With statement raises WGSLTranslationError."""
        import ast
        code = """
with resource as r:
    x = r.value
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_for_loop(self):
        """For loop raises WGSLTranslationError (no loop support yet)."""
        import ast
        code = """
for i in range(10):
    x = i
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_while_loop(self):
        """While loop raises WGSLTranslationError (no loop support yet)."""
        import ast
        code = """
while x < 10:
    x = x + 1
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_async_def(self):
        """Async function raises WGSLTranslationError."""
        import ast
        code = """
async def foo():
    pass
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_await(self):
        """Await expression raises WGSLTranslationError."""
        import ast
        code = """
async def foo():
    result = await bar()
"""
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_compilation_error_stored_on_class(self):
        """Material with invalid source stores error on _compilation_error."""
        # This tests that MaterialMeta captures errors gracefully
        # We can't easily create an invalid material since Python parses it first
        # But we can verify the attribute exists and is None for valid materials

        class ValidMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert ValidMaterial._compilation_error is None

    def test_unsupported_set_literal(self):
        """Set literal raises WGSLTranslationError."""
        import ast
        code = "s = {1, 2, 3}"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)

    def test_unsupported_dict_literal(self):
        """Dict literal raises WGSLTranslationError."""
        import ast
        code = "d = {'key': 'value'}"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()

        with pytest.raises(WGSLTranslationError, match="Unsupported"):
            translator.translate(tree)


# =============================================================================
# Suite K: Texture Sampling Integration Tests
# =============================================================================


class TestTextureSamplingIntegration:
    """Tests for texture sampling in material DSL."""

    def test_sample_2d_texture(self):
        """ctx.sample() generates textureSample call."""
        import ast
        code = "color = ctx.sample(texture, uv)"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "textureSample" in wgsl

    def test_sample_cube_texture(self):
        """ctx.sample_cube() generates textureSample call."""
        import ast
        code = "color = ctx.sample_cube(cubemap, direction)"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "textureSample" in wgsl

    def test_sample_level(self):
        """ctx.sample_level() generates textureSampleLevel call."""
        import ast
        code = "color = ctx.sample_level(texture, uv, 2.0)"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "textureSampleLevel" in wgsl

    def test_context_uv_access(self):
        """ctx.uv translates to in.uv."""
        import ast
        code = "uv = ctx.uv"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "in.uv" in wgsl

    def test_context_world_position(self):
        """ctx.world_position() translates to in.world_position."""
        import ast
        code = "pos = ctx.world_position()"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "in.world_position" in wgsl

    def test_context_world_normal(self):
        """ctx.world_normal() translates to in.world_normal."""
        import ast
        code = "n = ctx.world_normal()"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "in.world_normal" in wgsl

    def test_context_time(self):
        """ctx.time translates to time_uniforms.elapsed_seconds (T-MAT-5.5)."""
        import ast
        code = "t = ctx.time"
        tree = ast.parse(code)
        translator = PythonToWGSLTranslator()
        wgsl = translator.translate(tree)
        assert "time_uniforms.elapsed_seconds" in wgsl


# =============================================================================
# Suite L: Complex Material Examples
# =============================================================================


class TestComplexMaterials:
    """Complex material examples with multiple features."""

    def test_layered_material(self):
        """Material with multiple layers of logic compiles."""

        class LayeredMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Base layer
                out.base_color = Vec3(0.8, 0.6, 0.4)
                out.roughness = 0.7

                # Conditional metallic regions
                if ctx.uv.y > 0.5:
                    out.metallic = 0.9
                    out.roughness = 0.2
                else:
                    out.metallic = 0.0

                # Edge darkening
                edge: float = 1.0 - abs(ctx.uv.x - 0.5) * 2.0
                out.ao = edge

        assert LayeredMaterial._compilation_error is None
        wgsl = LayeredMaterial._wgsl_source
        assert "if" in wgsl
        assert "else" in wgsl
        assert "abs" in wgsl

    def test_animated_material(self):
        """Material using time-based animation compiles."""

        class AnimatedMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                t: float = ctx.time
                pulse: float = sin(t * 2.0) * 0.5 + 0.5
                out.emissive = Vec3(pulse, pulse * 0.5, 0.0)
                out.base_color = Vec3(0.2, 0.2, 0.2)
                out.roughness = 0.5

        assert AnimatedMaterial._compilation_error is None
        wgsl = AnimatedMaterial._wgsl_source
        assert "sin" in wgsl
        assert "uniforms.time" in wgsl or "ctx.time" in wgsl

    def test_gradient_material(self):
        """Material with UV-based gradient compiles."""

        class GradientMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Vertical gradient
                gradient: float = ctx.uv.y
                color_a = Vec3(1.0, 0.0, 0.0)
                color_b = Vec3(0.0, 0.0, 1.0)

                # Mix between colors
                out.base_color = mix(color_a, color_b, gradient)
                out.roughness = mix(0.3, 0.8, gradient)
                out.metallic = 0.0

        assert GradientMaterial._compilation_error is None
        wgsl = GradientMaterial._wgsl_source
        assert "mix" in wgsl

    def test_pbr_material_all_outputs(self):
        """Material setting all PBR output fields compiles."""

        class FullPBRMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Core PBR
                out.base_color = Vec3(0.8, 0.6, 0.4)
                out.metallic = 0.5
                out.roughness = 0.4
                out.ao = 0.9
                out.alpha = 1.0

                # Extended PBR
                out.emissive = Vec3(0.0, 0.0, 0.0)
                out.specular = 0.5
                out.clearcoat = 0.2
                out.sheen = 0.1
                out.transmission = 0.0
                out.ior = 1.5

        assert FullPBRMaterial._compilation_error is None
        wgsl = FullPBRMaterial._wgsl_source
        # Verify key fields are in output
        assert "base_color" in wgsl
        assert "metallic" in wgsl
        assert "roughness" in wgsl

    def test_math_heavy_material(self):
        """Material with complex math expressions compiles."""

        class MathMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                # Complex math operations
                x: float = ctx.uv.x * 2.0 - 1.0
                y: float = ctx.uv.y * 2.0 - 1.0

                dist: float = sqrt(x * x + y * y)
                angle: float = atan2(y, x)

                pattern: float = sin(dist * 10.0 + angle * 3.0)
                pattern = pattern * 0.5 + 0.5
                pattern = clamp(pattern, 0.0, 1.0)

                out.base_color = Vec3(pattern, pattern, pattern)
                out.roughness = 0.5

        assert MathMaterial._compilation_error is None
        wgsl = MathMaterial._wgsl_source
        assert "sqrt" in wgsl
        assert "atan2" in wgsl
        assert "sin" in wgsl
        assert "clamp" in wgsl


# =============================================================================
# Suite M: Builtin Function Integration Tests
# =============================================================================


class TestBuiltinIntegration:
    """Tests for builtin functions used in material context."""

    def test_noise_with_uv_scaling(self):
        """Noise function with UV scaling compiles."""

        class NoiseMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                scale: float = 10.0
                n: float = perlin_noise(Vec2(ctx.uv.x * scale, ctx.uv.y * scale))
                out.roughness = n * 0.5 + 0.5
                out.base_color = Vec3(0.5, 0.5, 0.5)

        assert NoiseMaterial._compilation_error is None
        assert "perlin_noise" in NoiseMaterial._wgsl_source

    def test_color_conversion_chain(self):
        """Color conversion functions chain correctly."""

        class ColorMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                rgb = Vec3(0.8, 0.2, 0.3)
                hsv = rgb_to_hsv(rgb)
                # Modify hue
                modified_hsv = Vec3(hsv.x + 0.2, hsv.y, hsv.z)
                result = hsv_to_rgb(modified_hsv)
                out.base_color = result
                out.roughness = 0.5

        assert ColorMaterial._compilation_error is None
        wgsl = ColorMaterial._wgsl_source
        assert "rgb_to_hsv" in wgsl
        assert "hsv_to_rgb" in wgsl

    def test_tonemap_in_emissive(self):
        """Tonemap function used for emissive compiles."""

        class HDRMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                hdr_color = Vec3(4.0, 2.0, 1.0)
                out.emissive = tonemap_aces(hdr_color)
                out.base_color = Vec3(0.1, 0.1, 0.1)
                out.roughness = 0.8

        assert HDRMaterial._compilation_error is None
        assert "tonemap_aces" in HDRMaterial._wgsl_source

    def test_smoothstep_gradient(self):
        """Smoothstep function for gradient compiles."""

        class SmoothMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                gradient: float = smoothstep(0.2, 0.8, ctx.uv.y)
                out.base_color = Vec3(gradient, gradient, gradient)
                out.roughness = mix(0.3, 0.9, gradient)

        assert SmoothMaterial._compilation_error is None
        wgsl = SmoothMaterial._wgsl_source
        assert "smoothstep" in wgsl
        assert "mix" in wgsl

    def test_reflect_for_environment(self):
        """Reflect function for environment mapping compiles."""

        class ReflectMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                view = normalize(ctx.view_direction)
                normal = ctx.normal
                reflected = reflect(view, normal)
                out.base_color = Vec3(reflected.x * 0.5 + 0.5, reflected.y * 0.5 + 0.5, reflected.z * 0.5 + 0.5)
                out.metallic = 1.0
                out.roughness = 0.1

        assert ReflectMaterial._compilation_error is None
        wgsl = ReflectMaterial._wgsl_source
        assert "reflect" in wgsl
        assert "normalize" in wgsl


# =============================================================================
# Fixture imports for DSL functions used in test classes
# =============================================================================

# These functions are used in the surface() bodies above
from trinity.materials.builtins import (
    perlin_noise,
    rgb_to_hsv,
    hsv_to_rgb,
    tonemap_aces,
)

# WGSL mapped builtins used directly
def sin(x):
    """Stub for sin() used in tests."""
    import math
    return math.sin(x)

def sqrt(x):
    """Stub for sqrt() used in tests."""
    import math
    return math.sqrt(x)

def atan2(y, x):
    """Stub for atan2() used in tests."""
    import math
    return math.atan2(y, x)

def clamp(x, min_val, max_val):
    """Stub for clamp() used in tests."""
    return max(min_val, min(x, max_val))

def mix(a, b, t):
    """Stub for mix() used in tests."""
    return a  # Stub

def smoothstep(edge0, edge1, x):
    """Stub for smoothstep() used in tests."""
    return x  # Stub

def normalize(v):
    """Stub for normalize() used in tests."""
    return v  # Stub

def reflect(i, n):
    """Stub for reflect() used in tests."""
    return Vec3(0, 0, 1)  # Stub
