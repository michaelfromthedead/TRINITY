"""Tests for material function library and built-in functions.

Tests MaterialFunctionLibrary singleton and all built-in shader
function generators.
"""
import pytest

from engine.rendering.materials.material_functions import (
    MaterialFunctionLibrary,
    create_box_mask_function,
    create_checkerboard_function,
    create_contrast_function,
    create_detail_normal_function,
    create_fresnel_function,
    create_fresnel_schlick_function,
    create_gradient_noise_function,
    create_height_blend_function,
    create_linear_to_srgb_function,
    create_luminance_function,
    create_noise_function,
    create_normal_blend_function,
    create_normal_blend_rnm_function,
    create_parallax_function,
    create_parallax_occlusion_function,
    create_radial_gradient_function,
    create_saturation_function,
    create_sphere_mask_function,
    create_srgb_to_linear_function,
    create_triplanar_function,
    create_voronoi_function,
    create_blend_overlay_function,
    create_blend_soft_light_function,
)
from engine.rendering.materials.material_system import (
    MaterialFunction,
    MaterialParameter,
    ParameterType,
)


class TestMaterialFunctionLibrary:
    """Test MaterialFunctionLibrary singleton."""

    def test_singleton(self):
        """Test that library is a singleton."""
        lib1 = MaterialFunctionLibrary()
        lib2 = MaterialFunctionLibrary()
        assert lib1 is lib2

    def test_get_instance(self):
        """Test get_instance class method."""
        instance = MaterialFunctionLibrary.get_instance()
        assert isinstance(instance, MaterialFunctionLibrary)

    def test_has_builtin_functions(self):
        """Test that built-in functions are pre-loaded."""
        lib = MaterialFunctionLibrary()
        functions = lib.get_all()
        # We expect at least 20+ built-in functions
        assert len(functions) >= 20

    def test_get_known_function(self):
        """Test retrieving a known function by name."""
        lib = MaterialFunctionLibrary()
        fresnel = lib.get("Fresnel")
        assert fresnel is not None
        assert fresnel.name == "Fresnel"

    def test_get_unknown_function(self):
        """Test retrieving an unknown function returns None."""
        lib = MaterialFunctionLibrary()
        result = lib.get("NonExistentFunction")
        assert result is None

    def test_register_custom_function(self):
        """Test registering a custom function."""
        lib = MaterialFunctionLibrary()
        custom = MaterialFunction(
            name="CustomFunc",
            code="float custom() { return 1.0; }",
        )
        lib.register(custom)
        retrieved = lib.get("CustomFunc")
        assert retrieved is not None
        assert retrieved.name == "CustomFunc"

    def test_get_by_category(self):
        """Test filtering functions by category."""
        lib = MaterialFunctionLibrary()
        lighting_funcs = lib.get_by_category("lighting")
        assert len(lighting_funcs) >= 2  # Fresnel + FresnelSchlick
        names = [f.name for f in lighting_funcs]
        assert "Fresnel" in names
        assert "FresnelSchlick" in names

    def test_get_by_category_normals(self):
        """Test filtering normal functions."""
        lib = MaterialFunctionLibrary()
        normal_funcs = lib.get_by_category("normal")
        assert len(normal_funcs) >= 2
        names = [f.name for f in normal_funcs]
        assert "NormalBlend" in names
        assert "NormalBlendRNM" in names

    def test_get_by_category_procedural(self):
        """Test filtering procedural functions."""
        lib = MaterialFunctionLibrary()
        procedural = lib.get_by_category("procedural")
        assert len(procedural) >= 4
        names = [f.name for f in procedural]
        assert "ValueNoise" in names
        assert "Voronoi" in names
        assert "GradientNoise" in names
        assert "Checkerboard" in names

    def test_get_by_category_uv(self):
        """Test filtering UV functions."""
        lib = MaterialFunctionLibrary()
        uv_funcs = lib.get_by_category("uv")
        assert len(uv_funcs) >= 3
        names = [f.name for f in uv_funcs]
        assert "ParallaxOffset" in names
        assert "ParallaxOcclusionMapping" in names
        assert "TriplanarSample" in names

    def test_get_by_category_blending(self):
        """Test filtering blending functions."""
        lib = MaterialFunctionLibrary()
        blend_funcs = lib.get_by_category("blending")
        assert len(blend_funcs) >= 3
        names = [f.name for f in blend_funcs]
        assert "HeightBlend" in names
        assert "BlendOverlay" in names
        assert "BlendSoftLight" in names

    def test_get_by_category_color(self):
        """Test filtering color functions."""
        lib = MaterialFunctionLibrary()
        color_funcs = lib.get_by_category("color")
        assert len(color_funcs) >= 5
        names = [f.name for f in color_funcs]
        assert "SRGBToLinear" in names
        assert "LinearToSRGB" in names
        assert "Luminance" in names
        assert "AdjustSaturation" in names
        assert "AdjustContrast" in names

    def test_get_by_category_masks(self):
        """Test filtering mask functions."""
        lib = MaterialFunctionLibrary()
        mask_funcs = lib.get_by_category("mask")
        assert len(mask_funcs) >= 2
        names = [f.name for f in mask_funcs]
        assert "BoxMask" in names
        assert "SphereMask" in names


class TestFresnelFunctions:
    """Test Fresnel effect functions."""

    def test_fresnel_function_structure(self):
        """Test Fresnel function structure."""
        func = create_fresnel_function()
        assert func.name == "Fresnel"
        assert "Schlick" in func.code

    def test_fresnel_has_inputs(self):
        """Test Fresnel function has correct inputs."""
        func = create_fresnel_function()
        assert len(func.inputs) == 3
        input_names = [i.name for i in func.inputs]
        assert "viewDir" in input_names
        assert "normal" in input_names
        assert "power" in input_names

    def test_fresnel_has_output(self):
        """Test Fresnel function has output."""
        func = create_fresnel_function()
        assert len(func.outputs) == 1
        assert func.outputs[0].name == "result"

    def test_fresnel_power_default(self):
        """Test Fresnel power default value."""
        func = create_fresnel_function()
        power_input = [i for i in func.inputs if i.name == "power"][0]
        assert power_input.default_value == 5.0
        assert power_input.min_value == 0.0
        assert power_input.max_value == 10.0

    def test_fresnel_schlick_function(self):
        """Test Fresnel-Schlick function structure."""
        func = create_fresnel_schlick_function()
        assert func.name == "FresnelSchlick"
        assert len(func.inputs) == 2


class TestNormalBlendFunctions:
    """Test normal blending functions."""

    def test_normal_blend_structure(self):
        """Test normal blend function structure."""
        func = create_normal_blend_function()
        assert func.name == "NormalBlend"
        assert "whiteout" in func.code.lower()

    def test_normal_blend_inputs(self):
        """Test normal blend inputs."""
        func = create_normal_blend_function()
        assert len(func.inputs) == 2
        input_names = [i.name for i in func.inputs]
        assert "n1" in input_names
        assert "n2" in input_names

    def test_normal_blend_rnm(self):
        """Test RNM normal blend structure."""
        func = create_normal_blend_rnm_function()
        assert func.name == "NormalBlendRNM"
        assert "Reoriented" in func.code


class TestParallaxFunctions:
    """Test parallax mapping functions."""

    def test_parallax_offset(self):
        """Test basic parallax offset structure."""
        func = create_parallax_function()
        assert func.name == "ParallaxOffset"
        assert len(func.inputs) == 4
        assert func.outputs[0].param_type == ParameterType.VEC2

    def test_parallax_scale_default(self):
        """Test parallax scale default."""
        func = create_parallax_function()
        scale = [i for i in func.inputs if i.name == "scale"][0]
        assert scale.default_value == 0.05
        assert scale.min_value == 0.0
        assert scale.max_value == 0.5

    def test_parallax_occlusion(self):
        """Test POM function structure."""
        func = create_parallax_occlusion_function()
        assert func.name == "ParallaxOcclusionMapping"
        assert "Parallax Occlusion" in func.description


class TestTriplanarFunction:
    """Test triplanar projection function."""

    def test_triplanar_structure(self):
        """Test triplanar function structure."""
        func = create_triplanar_function()
        assert func.name == "TriplanarSample"
        assert len(func.inputs) == 4
        input_names = [i.name for i in func.inputs]
        assert "worldPos" in input_names
        assert "worldNormal" in input_names
        assert "tiling" in input_names
        assert func.outputs[0].param_type == ParameterType.VEC4


class TestDetailNormalFunction:
    """Test detail normal blending."""

    def test_detail_normal_structure(self):
        """Test detail normal function structure."""
        func = create_detail_normal_function()
        assert func.name == "DetailNormal"
        assert len(func.inputs) == 3
        input_names = [i.name for i in func.inputs]
        assert "baseNormal" in input_names
        assert "detailNormal" in input_names

    def test_detail_normal_strength_default(self):
        """Test detail normal strength default."""
        func = create_detail_normal_function()
        strength = [i for i in func.inputs if i.name == "strength"][0]
        assert strength.default_value == 1.0
        assert strength.min_value == 0.0
        assert strength.max_value == 2.0


class TestHeightBlendFunction:
    """Test height-based blending function."""

    def test_height_blend_structure(self):
        """Test height blend function structure."""
        func = create_height_blend_function()
        assert func.name == "HeightBlend"
        assert len(func.inputs) == 4


class TestColorSpaceFunctions:
    """Test color space conversion functions."""

    def test_srgb_to_linear(self):
        """Test sRGB to linear conversion function."""
        func = create_srgb_to_linear_function()
        assert func.name == "SRGBToLinear"
        assert "pow" in func.code
        assert "2.2" in func.code

    def test_linear_to_srgb(self):
        """Test linear to sRGB conversion function."""
        func = create_linear_to_srgb_function()
        assert func.name == "LinearToSRGB"
        assert "1.0 / 2.2" in func.code or "1/2.2" in func.code


class TestColorAdjustFunctions:
    """Test color adjustment functions."""

    def test_luminance(self):
        """Test luminance function structure."""
        func = create_luminance_function()
        assert func.name == "Luminance"
        assert "Rec. 709" in func.description or "0.2126" in func.code
        assert func.outputs[0].param_type == ParameterType.FLOAT

    def test_saturation(self):
        """Test saturation function structure."""
        func = create_saturation_function()
        assert func.name == "AdjustSaturation"
        assert len(func.inputs) == 2
        saturation = [i for i in func.inputs if i.name == "saturation"][0]
        assert saturation.default_value == 1.0
        assert saturation.min_value == 0.0
        assert saturation.max_value == 3.0

    def test_contrast(self):
        """Test contrast function structure."""
        func = create_contrast_function()
        assert func.name == "AdjustContrast"
        contrast = [i for i in func.inputs if i.name == "contrast"][0]
        assert contrast.default_value == 1.0
        assert contrast.max_value == 3.0


class TestProceduralFunctions:
    """Test procedural noise/pattern functions."""

    def test_value_noise(self):
        """Test value noise function structure."""
        func = create_noise_function()
        assert func.name == "ValueNoise"
        assert len(func.inputs) == 1
        assert func.inputs[0].name == "p"
        assert func.inputs[0].param_type == ParameterType.VEC2
        assert func.outputs[0].param_type == ParameterType.FLOAT

    def test_voronoi(self):
        """Test Voronoi function structure."""
        func = create_voronoi_function()
        assert func.name == "Voronoi"
        assert func.outputs[0].param_type == ParameterType.VEC2

    def test_gradient_noise(self):
        """Test gradient noise function structure."""
        func = create_gradient_noise_function()
        assert func.name == "GradientNoise"
        assert "Quintic" in func.code or "Perlin" in func.description

    def test_checkerboard(self):
        """Test checkerboard function structure."""
        func = create_checkerboard_function()
        assert func.name == "Checkerboard"
        assert len(func.inputs) == 2

    def test_radial_gradient(self):
        """Test radial gradient function structure."""
        func = create_radial_gradient_function()
        assert func.name == "RadialGradient"
        assert len(func.inputs) == 3


class TestMaskFunctions:
    """Test mask functions."""

    def test_box_mask(self):
        """Test box mask function structure."""
        func = create_box_mask_function()
        assert func.name == "BoxMask"
        assert len(func.inputs) == 4
        input_names = [i.name for i in func.inputs]
        assert "pos" in input_names
        assert "boxMin" in input_names
        assert "boxMax" in input_names

    def test_sphere_mask(self):
        """Test sphere mask function structure."""
        func = create_sphere_mask_function()
        assert func.name == "SphereMask"
        assert len(func.inputs) == 4
        input_names = [i.name for i in func.inputs]
        assert "center" in input_names
        assert "radius" in input_names
        assert "hardness" in input_names

    def test_sphere_mask_hardness_default(self):
        """Test sphere mask hardness default."""
        func = create_sphere_mask_function()
        hardness = [i for i in func.inputs if i.name == "hardness"][0]
        assert hardness.default_value == 0.5
        assert hardness.min_value == 0.001
        assert hardness.max_value == 1.0


class TestBlendFunctions:
    """Test blend mode functions."""

    def test_blend_overlay(self):
        """Test overlay blend function."""
        func = create_blend_overlay_function()
        assert func.name == "BlendOverlay"
        assert len(func.inputs) == 2

    def test_blend_soft_light(self):
        """Test soft light blend function."""
        func = create_blend_soft_light_function()
        assert func.name == "BlendSoftLight"
        assert len(func.inputs) == 2
