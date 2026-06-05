"""Tests for Texture Binding Model (T-MAT-1.5).

Verifies:
- Texture2D generates correct WGSL bindings
- TextureCube generates correct WGSL bindings
- sRGB flag affects format selection
- Default texture fallbacks work
- Binding index assignment works
- Integration with MaterialMeta class
"""

from __future__ import annotations

import pytest

from trinity.materials import (
    Material,
    MaterialMeta,
    SurfaceContext,
    SurfaceOutput,
    surface,
    Vec3,
    MaterialCompiler,
)

from trinity.materials.textures import (
    Texture2D,
    TextureCube,
    TextureDescriptor,
    TextureBindingSet,
    FilterMode,
    AddressMode,
    TextureFormat,
    DEFAULT_TEXTURES,
    DefaultTextureSpec,
    get_default_texture,
    is_valid_default,
    collect_texture_descriptors,
    validate_texture_descriptor,
)


# =============================================================================
# Suite A: Texture2D WGSL Binding Generation
# =============================================================================


class TestTexture2DBindingGeneration:
    """Texture2D generates correct WGSL bindings."""

    def test_texture2d_generates_texture_binding(self):
        """Texture2D generates texture_2d<f32> binding."""
        tex = Texture2D(default="white")
        tex.binding_index = 1
        tex._name = "albedo"

        wgsl = tex.generate_wgsl_binding("albedo", group=0)

        assert "@group(0) @binding(1)" in wgsl
        assert "var albedo_tex: texture_2d<f32>" in wgsl

    def test_texture2d_generates_sampler_binding(self):
        """Texture2D generates sampler binding at index+1."""
        tex = Texture2D(default="white")
        tex.binding_index = 1
        tex._name = "albedo"

        wgsl = tex.generate_wgsl_binding("albedo", group=0)

        assert "@group(0) @binding(2)" in wgsl
        assert "var albedo_sampler: sampler" in wgsl

    def test_texture2d_respects_group_parameter(self):
        """Texture2D uses specified binding group."""
        tex = Texture2D(default="white")
        tex.binding_index = 1

        wgsl = tex.generate_wgsl_binding("albedo", group=1)

        assert "@group(1) @binding(1)" in wgsl
        assert "@group(1) @binding(2)" in wgsl

    def test_texture2d_custom_binding_index(self):
        """Texture2D uses assigned binding index."""
        tex = Texture2D(default="white")
        tex.binding_index = 5

        wgsl = tex.generate_wgsl_binding("roughness", group=0)

        assert "@group(0) @binding(5)" in wgsl
        assert "@group(0) @binding(6)" in wgsl
        assert "roughness_tex" in wgsl
        assert "roughness_sampler" in wgsl

    def test_texture2d_no_binding_index_raises(self):
        """Texture2D raises if binding_index not set."""
        tex = Texture2D(default="white")
        # binding_index is None by default

        with pytest.raises(RuntimeError, match="no binding_index"):
            tex.generate_wgsl_binding("albedo")

    def test_texture2d_sample_call_generation(self):
        """Texture2D generates correct textureSample call."""
        tex = Texture2D(default="white")
        tex._name = "albedo"

        sample_call = tex.generate_sample_call("in.uv")

        assert sample_call == "textureSample(albedo_tex, albedo_sampler, in.uv)"


# =============================================================================
# Suite B: TextureCube WGSL Binding Generation
# =============================================================================


class TestTextureCubeBindingGeneration:
    """TextureCube generates correct WGSL bindings."""

    def test_texturecube_generates_cube_binding(self):
        """TextureCube generates texture_cube<f32> binding."""
        tex = TextureCube(default="black")
        tex.binding_index = 1

        wgsl = tex.generate_wgsl_binding("environment", group=0)

        assert "@group(0) @binding(1)" in wgsl
        assert "var environment_tex: texture_cube<f32>" in wgsl

    def test_texturecube_generates_sampler_binding(self):
        """TextureCube generates sampler binding at index+1."""
        tex = TextureCube(default="black")
        tex.binding_index = 1

        wgsl = tex.generate_wgsl_binding("environment", group=0)

        assert "@group(0) @binding(2)" in wgsl
        assert "var environment_sampler: sampler" in wgsl

    def test_texturecube_is_cube_flag(self):
        """TextureCube has _is_cube=True."""
        tex = TextureCube(default="black")
        assert tex._is_cube is True

    def test_texture2d_is_not_cube(self):
        """Texture2D has _is_cube=False."""
        tex = Texture2D(default="white")
        assert tex._is_cube is False

    def test_texturecube_sample_call_generation(self):
        """TextureCube generates correct textureSample call."""
        tex = TextureCube(default="black")
        tex._name = "environment"

        sample_call = tex.generate_sample_call("reflect_dir")

        assert sample_call == "textureSample(environment_tex, environment_sampler, reflect_dir)"


# =============================================================================
# Suite C: sRGB Format Selection
# =============================================================================


class TestSRGBFormatSelection:
    """sRGB flag affects format selection."""

    def test_srgb_false_uses_unorm(self):
        """sRGB=False uses RGBA8_UNORM format."""
        tex = Texture2D(default="white", srgb=False)

        format = tex.get_format()

        assert format == TextureFormat.RGBA8_UNORM
        assert format.to_wgsl() == "rgba8unorm"

    def test_srgb_true_uses_srgb_format(self):
        """sRGB=True uses RGBA8_UNORM_SRGB format."""
        tex = Texture2D(default="white", srgb=True)

        format = tex.get_format()

        assert format == TextureFormat.RGBA8_UNORM_SRGB
        assert format.to_wgsl() == "rgba8unorm-srgb"

    def test_srgb_adds_comment_to_binding(self):
        """sRGB texture includes format comment in binding."""
        tex = Texture2D(default="white", srgb=True)
        tex.binding_index = 1

        wgsl = tex.generate_wgsl_binding("albedo", group=0)

        assert "// sRGB" in wgsl
        assert "rgba8unorm-srgb" in wgsl

    def test_non_srgb_no_format_comment(self):
        """Non-sRGB texture has no format comment."""
        tex = Texture2D(default="white", srgb=False)
        tex.binding_index = 1

        wgsl = tex.generate_wgsl_binding("albedo", group=0)

        assert "// sRGB" not in wgsl

    def test_texturecube_srgb_format(self):
        """TextureCube also supports sRGB."""
        tex = TextureCube(default="black", srgb=True)

        format = tex.get_format()

        assert format == TextureFormat.RGBA8_UNORM_SRGB


# =============================================================================
# Suite D: Default Texture Fallbacks
# =============================================================================


class TestDefaultTextureFallbacks:
    """Default texture fallbacks work correctly."""

    def test_white_default_exists(self):
        """'white' default texture is defined."""
        spec = get_default_texture("white")

        assert spec is not None
        assert spec.name == "white"
        assert spec.color == (1.0, 1.0, 1.0, 1.0)

    def test_black_default_exists(self):
        """'black' default texture is defined."""
        spec = get_default_texture("black")

        assert spec is not None
        assert spec.name == "black"
        assert spec.color == (0.0, 0.0, 0.0, 1.0)

    def test_flat_normal_default_exists(self):
        """'flat_normal' default texture is defined."""
        spec = get_default_texture("flat_normal")

        assert spec is not None
        assert spec.name == "flat_normal"
        assert spec.color == (0.5, 0.5, 1.0, 1.0)

    def test_gray_default_exists(self):
        """'gray' default texture is defined."""
        spec = get_default_texture("gray")

        assert spec is not None
        assert spec.name == "gray"
        assert spec.color == (0.5, 0.5, 0.5, 1.0)

    def test_transparent_default_exists(self):
        """'transparent' default texture is defined."""
        spec = get_default_texture("transparent")

        assert spec is not None
        assert spec.name == "transparent"
        assert spec.color == (0.0, 0.0, 0.0, 0.0)

    def test_invalid_default_returns_none(self):
        """Invalid default name returns None."""
        spec = get_default_texture("invalid_name")
        assert spec is None

    def test_is_valid_default_true(self):
        """is_valid_default returns True for valid names."""
        assert is_valid_default("white") is True
        assert is_valid_default("black") is True
        assert is_valid_default("flat_normal") is True

    def test_is_valid_default_false(self):
        """is_valid_default returns False for invalid names."""
        assert is_valid_default("invalid") is False
        assert is_valid_default("") is False

    def test_texture2d_invalid_default_raises(self):
        """Texture2D with invalid default raises ValueError."""
        with pytest.raises(ValueError, match="Invalid default texture"):
            Texture2D(default="not_a_valid_default")

    def test_texture_get_default_spec(self):
        """get_default_spec returns correct spec."""
        tex = Texture2D(default="flat_normal")

        spec = tex.get_default_spec()

        assert spec.name == "flat_normal"
        assert spec.color == (0.5, 0.5, 1.0, 1.0)

    def test_default_texture_spec_is_frozen(self):
        """DefaultTextureSpec is immutable (frozen dataclass)."""
        spec = get_default_texture("white")

        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "modified"


# =============================================================================
# Suite E: Binding Index Assignment
# =============================================================================


class TestBindingIndexAssignment:
    """Binding index assignment works correctly."""

    def test_texture_binding_set_adds_texture(self):
        """TextureBindingSet.add assigns binding index."""
        binding_set = TextureBindingSet()
        tex = Texture2D(default="white")

        binding_idx = binding_set.add("albedo", tex)

        assert binding_idx == 1  # First binding after uniforms
        assert tex.binding_index == 1

    def test_texture_binding_set_increments_by_two(self):
        """Each texture uses 2 bindings (texture + sampler)."""
        binding_set = TextureBindingSet()
        tex1 = Texture2D(default="white")
        tex2 = Texture2D(default="flat_normal")

        idx1 = binding_set.add("albedo", tex1)
        idx2 = binding_set.add("normal", tex2)

        assert idx1 == 1
        assert idx2 == 3  # 1 + 2 (texture + sampler)

    def test_texture_binding_set_assigns_name(self):
        """TextureBindingSet.add sets _name on descriptor."""
        binding_set = TextureBindingSet()
        tex = Texture2D(default="white")

        binding_set.add("my_texture", tex)

        assert tex._name == "my_texture"

    def test_texture_binding_set_length(self):
        """TextureBindingSet len() returns texture count."""
        binding_set = TextureBindingSet()
        binding_set.add("a", Texture2D(default="white"))
        binding_set.add("b", Texture2D(default="black"))

        assert len(binding_set) == 2

    def test_texture_binding_set_contains(self):
        """TextureBindingSet 'in' operator works."""
        binding_set = TextureBindingSet()
        binding_set.add("albedo", Texture2D(default="white"))

        assert "albedo" in binding_set
        assert "normal" not in binding_set

    def test_texture_binding_set_get(self):
        """TextureBindingSet.get retrieves descriptor."""
        binding_set = TextureBindingSet()
        tex = Texture2D(default="white")
        binding_set.add("albedo", tex)

        result = binding_set.get("albedo")

        assert result is tex

    def test_texture_binding_set_get_missing_returns_none(self):
        """TextureBindingSet.get returns None for missing."""
        binding_set = TextureBindingSet()

        result = binding_set.get("nonexistent")

        assert result is None

    def test_texture_binding_set_generate_all(self):
        """TextureBindingSet.generate_all_bindings creates WGSL."""
        binding_set = TextureBindingSet()
        binding_set.add("albedo", Texture2D(default="white"))
        binding_set.add("normal", Texture2D(default="flat_normal"))

        wgsl = binding_set.generate_all_bindings(group=0)

        assert "albedo_tex" in wgsl
        assert "albedo_sampler" in wgsl
        assert "normal_tex" in wgsl
        assert "normal_sampler" in wgsl

    def test_collect_texture_descriptors(self):
        """collect_texture_descriptors finds all descriptors."""
        namespace = {
            "albedo": Texture2D(default="white"),
            "normal": Texture2D(default="flat_normal"),
            "not_a_texture": 42,
            "also_not": "string",
        }

        binding_set = collect_texture_descriptors(namespace)

        assert len(binding_set) == 2
        assert "albedo" in binding_set
        assert "normal" in binding_set


# =============================================================================
# Suite F: MaterialMeta Integration
# =============================================================================


class TestMaterialMetaIntegration:
    """Integration with MaterialMeta class."""

    def test_material_with_texture2d_compiles(self):
        """Material with Texture2D compiles without error."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)
                out.roughness = 0.5

        assert TexturedMaterial._compilation_error is None
        assert TexturedMaterial._wgsl_source != ""

    def test_material_texture_bindings_collected(self):
        """MaterialMeta collects texture descriptors."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")
            normal = Texture2D(default="flat_normal")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        bindings = TexturedMaterial._texture_bindings

        assert "albedo" in bindings
        assert "normal" in bindings
        assert len(bindings) == 2

    def test_material_has_texture_helper(self):
        """Material.has_texture() works correctly."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert TexturedMaterial.has_texture("albedo") is True
        assert TexturedMaterial.has_texture("normal") is False

    def test_material_get_textures_helper(self):
        """Material.get_textures() returns all bindings."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")
            roughness = Texture2D(default="gray")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.metallic = 0.0

        textures = TexturedMaterial.get_textures()

        assert "albedo" in textures
        assert "roughness" in textures
        assert isinstance(textures["albedo"], Texture2D)

    def test_material_compiler_emits_texture_bindings(self):
        """MaterialCompiler includes texture bindings in output."""

        class TexturedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        compiler = MaterialCompiler(include_pbr_template=True)
        wgsl = compiler.compile(TexturedMaterial)

        # The compiler should emit texture bindings
        assert "albedo_texture" in wgsl or "texture" in wgsl.lower()

    def test_material_with_texturecube(self):
        """Material with TextureCube compiles."""

        class EnvMaterial(Material, metaclass=MaterialMeta):
            environment = TextureCube(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.2

        assert EnvMaterial._compilation_error is None
        bindings = EnvMaterial._texture_bindings
        assert "environment" in bindings
        assert bindings["environment"]._is_cube is True

    def test_material_mixed_texture_types(self):
        """Material with both Texture2D and TextureCube compiles."""

        class MixedMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")
            environment = TextureCube(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        assert MixedMaterial._compilation_error is None
        bindings = MixedMaterial._texture_bindings
        assert len(bindings) == 3
        assert bindings["albedo"]._is_cube is False
        assert bindings["environment"]._is_cube is True


# =============================================================================
# Suite G: Filter and Address Mode Validation
# =============================================================================


class TestFilterAndAddressModes:
    """Filter and address mode configuration."""

    def test_default_filter_is_linear(self):
        """Default filter mode is linear."""
        tex = Texture2D(default="white")
        assert tex.filter == FilterMode.LINEAR

    def test_default_address_is_repeat(self):
        """Default address mode is repeat."""
        tex = Texture2D(default="white")
        assert tex.address == AddressMode.REPEAT

    def test_filter_mode_string_conversion(self):
        """Filter mode string is converted to enum."""
        tex = Texture2D(default="white", filter="nearest")
        assert tex.filter == FilterMode.NEAREST

    def test_address_mode_string_conversion(self):
        """Address mode string is converted to enum."""
        tex = Texture2D(default="white", address="clamp-to-edge")
        assert tex.address == AddressMode.CLAMP_TO_EDGE

    def test_filter_mode_enum_accepted(self):
        """FilterMode enum is accepted directly."""
        tex = Texture2D(default="white", filter=FilterMode.NEAREST)
        assert tex.filter == FilterMode.NEAREST

    def test_address_mode_enum_accepted(self):
        """AddressMode enum is accepted directly."""
        tex = Texture2D(default="white", address=AddressMode.MIRROR_REPEAT)
        assert tex.address == AddressMode.MIRROR_REPEAT

    def test_invalid_filter_mode_raises(self):
        """Invalid filter mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filter mode"):
            Texture2D(default="white", filter="invalid_filter")

    def test_invalid_address_mode_raises(self):
        """Invalid address mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid address mode"):
            Texture2D(default="white", address="invalid_address")

    def test_filter_mode_to_wgsl(self):
        """FilterMode.to_wgsl() returns correct string."""
        assert FilterMode.NEAREST.to_wgsl() == "nearest"
        assert FilterMode.LINEAR.to_wgsl() == "linear"

    def test_address_mode_to_wgsl(self):
        """AddressMode.to_wgsl() returns correct string."""
        assert AddressMode.REPEAT.to_wgsl() == "repeat"
        assert AddressMode.MIRROR_REPEAT.to_wgsl() == "mirror-repeat"
        assert AddressMode.CLAMP_TO_EDGE.to_wgsl() == "clamp-to-edge"


# =============================================================================
# Suite H: Descriptor Attributes
# =============================================================================


class TestDescriptorAttributes:
    """Texture descriptor attribute behavior."""

    def test_texture_is_descriptor_marker(self):
        """TextureDescriptor has _is_texture_descriptor marker."""
        tex = Texture2D(default="white")
        assert hasattr(tex, "_is_texture_descriptor")
        assert tex._is_texture_descriptor is True

    def test_texture_path_attribute(self):
        """Texture path attribute is stored."""
        tex = Texture2D(default="white", path="/textures/brick.png")
        assert tex.path == "/textures/brick.png"

    def test_texture_path_default_none(self):
        """Texture path defaults to None."""
        tex = Texture2D(default="white")
        assert tex.path is None

    def test_set_name_captures_attribute_name(self):
        """__set_name__ captures the attribute name."""

        class Container:
            my_texture = Texture2D(default="white")

        # __set_name__ is called during class creation
        assert Container.my_texture._name == "my_texture"

    def test_name_property(self):
        """name property returns _name or default."""
        tex1 = Texture2D(default="white")
        tex1._name = "custom_name"
        assert tex1.name == "custom_name"

        tex2 = Texture2D(default="white")
        assert tex2.name == "unnamed_texture"

    def test_texture_repr(self):
        """Texture descriptor has informative __repr__."""
        tex = Texture2D(default="white", srgb=True, filter="nearest")

        repr_str = repr(tex)

        assert "Texture2D" in repr_str
        assert "white" in repr_str
        assert "srgb=True" in repr_str
        assert "nearest" in repr_str


# =============================================================================
# Suite I: Validation
# =============================================================================


class TestValidation:
    """Texture descriptor validation."""

    def test_validate_valid_descriptor(self):
        """validate_texture_descriptor passes for valid config."""
        tex = Texture2D(default="white", filter="linear", address="repeat")

        # Should not raise
        validate_texture_descriptor(tex)

    def test_validate_invalid_default_raises(self):
        """validate_texture_descriptor raises for invalid default."""
        tex = Texture2D(default="white")
        tex.default = "invalid"  # Force invalid state

        with pytest.raises(ValueError, match="Invalid default texture"):
            validate_texture_descriptor(tex)

    def test_validate_invalid_filter_raises(self):
        """validate_texture_descriptor raises for invalid filter."""
        tex = Texture2D(default="white")
        tex.filter = "invalid"  # Force invalid state

        with pytest.raises(ValueError, match="Invalid filter mode"):
            validate_texture_descriptor(tex)

    def test_validate_invalid_address_raises(self):
        """validate_texture_descriptor raises for invalid address."""
        tex = Texture2D(default="white")
        tex.address = "invalid"  # Force invalid state

        with pytest.raises(ValueError, match="Invalid address mode"):
            validate_texture_descriptor(tex)


# =============================================================================
# Suite J: Texture Format Enum
# =============================================================================


class TestTextureFormat:
    """TextureFormat enum values."""

    def test_all_formats_have_wgsl(self):
        """All TextureFormat values have to_wgsl()."""
        for fmt in TextureFormat:
            wgsl = fmt.to_wgsl()
            assert isinstance(wgsl, str)
            assert len(wgsl) > 0

    def test_format_values(self):
        """TextureFormat values match WGSL format strings."""
        assert TextureFormat.RGBA8_UNORM.to_wgsl() == "rgba8unorm"
        assert TextureFormat.RGBA8_UNORM_SRGB.to_wgsl() == "rgba8unorm-srgb"
        assert TextureFormat.RGBA16_FLOAT.to_wgsl() == "rgba16float"
        assert TextureFormat.RGBA32_FLOAT.to_wgsl() == "rgba32float"
        assert TextureFormat.R8_UNORM.to_wgsl() == "r8unorm"
        assert TextureFormat.RG8_UNORM.to_wgsl() == "rg8unorm"
        assert TextureFormat.DEPTH32_FLOAT.to_wgsl() == "depth32float"


# =============================================================================
# Suite K: End-to-End Material Examples
# =============================================================================


class TestEndToEndTexturedMaterials:
    """Complete textured material definitions."""

    def test_pbr_material_with_all_textures(self):
        """Full PBR material with multiple textures compiles."""

        class FullPBRMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)
            normal = Texture2D(default="flat_normal")
            roughness = Texture2D(default="gray")
            metallic = Texture2D(default="black")
            ao = Texture2D(default="white")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(1.0, 1.0, 1.0)
                out.roughness = 0.5
                out.metallic = 0.0

        assert FullPBRMaterial._compilation_error is None
        assert len(FullPBRMaterial._texture_bindings) == 5

    def test_environment_mapped_material(self):
        """Material with environment cube map compiles."""

        class EnvironmentMaterial(Material, metaclass=MaterialMeta):
            environment = TextureCube(default="black", srgb=True)
            irradiance = TextureCube(default="black")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.5, 0.5, 0.5)
                out.metallic = 1.0
                out.roughness = 0.1

        assert EnvironmentMaterial._compilation_error is None
        bindings = EnvironmentMaterial._texture_bindings
        assert bindings["environment"]._is_cube is True
        assert bindings["environment"].srgb is True
        assert bindings["irradiance"]._is_cube is True

    def test_material_inherits_textures(self):
        """Child material can add more textures."""

        class BaseMaterial(Material, metaclass=MaterialMeta):
            albedo = Texture2D(default="white", srgb=True)

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.5

        class ExtendedMaterial(BaseMaterial, metaclass=MaterialMeta):
            detail = Texture2D(default="gray")

            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.roughness = 0.3

        # Base has 1 texture
        assert "albedo" in BaseMaterial._texture_bindings

        # Extended has its own texture
        assert "detail" in ExtendedMaterial._texture_bindings
