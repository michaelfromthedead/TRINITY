"""Tests for the PBR metallic-roughness model.

Tests PBRParameters, PBRMaterial, and related validation functions.
"""
import pytest

from engine.core.math.vec import Vec3, Vec4
from engine.rendering.materials.pbr_model import (
    PBRDirtyFlags,
    PBRMaterial,
    PBRParameters,
    PBRTextureSet,
    PBRWorkflow,
    TextureChannel,
    clamp_pbr_parameter,
    validate_pbr_parameter,
)


class TestPBRParameters:
    """Test PBRParameters dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = PBRParameters()
        assert params.base_color == Vec4(1.0, 1.0, 1.0, 1.0)
        assert params.metallic == 0.0
        assert params.roughness == 0.5
        assert params.normal_scale == 1.0
        assert params.ao == 1.0
        assert params.emissive == Vec3(0.0, 0.0, 0.0)

    def test_custom_values(self):
        """Test custom parameter values."""
        params = PBRParameters(
            base_color=Vec4(0.8, 0.2, 0.1, 1.0),
            metallic=0.9,
            roughness=0.1,
            emissive=Vec3(1.0, 0.5, 0.0),
        )
        assert params.metallic == 0.9
        assert params.roughness == 0.1

    def test_validate_valid_params(self):
        """Test validation of valid parameters."""
        params = PBRParameters()
        is_valid, errors = params.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_metallic(self):
        """Test validation catches invalid metallic."""
        params = PBRParameters(metallic=1.5)
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("metallic" in e for e in errors)

    def test_validate_invalid_roughness(self):
        """Test validation catches invalid roughness."""
        params = PBRParameters(roughness=-0.1)
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("roughness" in e for e in errors)

    def test_validate_invalid_base_color(self):
        """Test validation catches invalid base color."""
        params = PBRParameters(base_color=Vec4(1.5, 0.0, 0.0, 1.0))
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("base_color" in e for e in errors)

    def test_validate_invalid_emissive(self):
        """Test validation catches negative emissive."""
        params = PBRParameters(emissive=Vec3(-1.0, 0.0, 0.0))
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("emissive" in e for e in errors)

    def test_clamp_values(self):
        """Test clamping out-of-range values."""
        params = PBRParameters(
            metallic=1.5,
            roughness=-0.5,
            ao=2.0,
        )
        clamped = params.clamp()
        assert clamped.metallic == 1.0
        assert clamped.roughness == 0.0
        assert clamped.ao == 1.0

    def test_to_shader_data(self):
        """Test conversion to shader data format."""
        params = PBRParameters(
            base_color=Vec4(1.0, 0.5, 0.0, 1.0),
            metallic=0.8,
        )
        data = params.to_shader_data()
        assert data["baseColor"] == (1.0, 0.5, 0.0, 1.0)
        assert data["metallic"] == 0.8

    def test_lerp_interpolation(self):
        """Test linear interpolation between parameters."""
        params_a = PBRParameters(
            metallic=0.0,
            roughness=0.0,
        )
        params_b = PBRParameters(
            metallic=1.0,
            roughness=1.0,
        )
        result = params_a.lerp(params_b, 0.5)
        assert abs(result.metallic - 0.5) < 0.001
        assert abs(result.roughness - 0.5) < 0.001


class TestPBRTextureSet:
    """Test PBRTextureSet texture bindings."""

    def test_default_empty(self):
        """Test default empty texture set."""
        textures = PBRTextureSet()
        assert not textures.has_any_texture()
        assert len(textures.get_texture_paths()) == 0

    def test_with_textures(self):
        """Test texture set with assigned textures."""
        textures = PBRTextureSet(
            base_color_map="textures/albedo.png",
            normal_map="textures/normal.png",
        )
        assert textures.has_any_texture()
        paths = textures.get_texture_paths()
        assert len(paths) == 2
        assert "textures/albedo.png" in paths

    def test_channel_configuration(self):
        """Test channel configuration for packed textures."""
        textures = PBRTextureSet(
            metallic_channel=TextureChannel.B,
            roughness_channel=TextureChannel.G,
            ao_channel=TextureChannel.R,
        )
        assert textures.metallic_channel == TextureChannel.B
        assert textures.roughness_channel == TextureChannel.G


class TestPBRDirtyFlags:
    """Test PBRDirtyFlags tracking."""

    def test_initial_all_dirty(self):
        """Test initial state."""
        flags = PBRDirtyFlags()
        assert flags.any_dirty()  # Defaults to all dirty

    def test_clear_all(self):
        """Test clearing all flags."""
        flags = PBRDirtyFlags()
        flags.clear_all()
        assert not flags.any_dirty()

    def test_individual_flags(self):
        """Test individual flag setting."""
        flags = PBRDirtyFlags()
        flags.clear_all()

        flags.metallic = True
        assert flags.metallic
        assert flags.any_dirty()
        assert not flags.roughness


class TestPBRMaterial:
    """Test PBRMaterial component."""

    def test_create_default(self):
        """Test creating default PBR material."""
        mat = PBRMaterial()
        assert mat.name == "PBRMaterial"
        assert mat.metallic == 0.0
        assert mat.roughness == 0.5

    def test_create_with_params(self):
        """Test creating with custom parameters."""
        params = PBRParameters(
            metallic=0.9,
            roughness=0.1,
        )
        mat = PBRMaterial(name="Metal", params=params)
        assert mat.metallic == 0.9
        assert mat.roughness == 0.1

    def test_property_setters_clamp(self):
        """Test that setters clamp values."""
        mat = PBRMaterial()
        mat.metallic = 1.5
        assert mat.metallic == 1.0

        mat.roughness = -0.5
        assert mat.roughness == 0.0

    def test_dirty_flag_on_change(self):
        """Test dirty flags are set on property changes."""
        mat = PBRMaterial()
        mat.dirty.clear_all()

        mat.metallic = 0.5
        assert mat.dirty.metallic

        mat.dirty.clear_all()
        mat.roughness = 0.8
        assert mat.dirty.roughness

    def test_no_dirty_if_same_value(self):
        """Test no dirty flag if value unchanged."""
        mat = PBRMaterial()
        mat.dirty.clear_all()

        original = mat.metallic
        mat.metallic = original
        assert not mat.dirty.metallic

    def test_base_color_setter(self):
        """Test base color property."""
        mat = PBRMaterial()
        mat.dirty.clear_all()

        mat.base_color = Vec4(0.5, 0.5, 0.5, 1.0)
        assert mat.base_color == Vec4(0.5, 0.5, 0.5, 1.0)
        assert mat.dirty.base_color

    def test_emissive_setter(self):
        """Test emissive property."""
        mat = PBRMaterial()
        mat.emissive = Vec3(10.0, 5.0, 0.0)
        assert mat.emissive.x == 10.0

    def test_get_parameters(self):
        """Test getting parameters as dataclass."""
        mat = PBRMaterial()
        mat.metallic = 0.8
        mat.roughness = 0.2
        params = mat.get_parameters()
        assert params.metallic == 0.8
        assert params.roughness == 0.2

    def test_set_parameters(self):
        """Test setting all parameters from dataclass."""
        mat = PBRMaterial()
        params = PBRParameters(
            metallic=0.9,
            roughness=0.1,
            ao=0.8,
        )
        mat.set_parameters(params)
        assert mat.metallic == 0.9
        assert mat.roughness == 0.1
        assert mat.ao == 0.8

    def test_to_shader_data(self):
        """Test shader data conversion."""
        mat = PBRMaterial()
        mat.metallic = 0.5
        data = mat.to_shader_data()
        assert "metallic" in data
        assert data["metallic"] == 0.5

    def test_on_change_callback(self):
        """Test change notification callback."""
        mat = PBRMaterial()
        mat.dirty.clear_all()

        changes = []

        def callback(name, value):
            changes.append((name, value))

        mat.on_change(callback)
        mat.metallic = 0.7
        mat.roughness = 0.3

        assert len(changes) == 2
        assert changes[0] == ("metallic", 0.7)
        assert changes[1] == ("roughness", 0.3)

    def test_clone(self):
        """Test cloning a material."""
        mat = PBRMaterial(name="Original")
        mat.metallic = 0.8
        mat.roughness = 0.2

        clone = mat.clone("Cloned")
        assert clone.name == "Cloned"
        assert clone.metallic == 0.8
        assert clone.roughness == 0.2
        assert clone.material_id != mat.material_id

    def test_workflow_type(self):
        """Test workflow type property."""
        mat = PBRMaterial(workflow=PBRWorkflow.SPECULAR_GLOSSINESS)
        assert mat.workflow == PBRWorkflow.SPECULAR_GLOSSINESS


class TestValidationFunctions:
    """Test standalone validation functions."""

    def test_validate_valid_float(self):
        """Test validating valid float parameter."""
        is_valid, _ = validate_pbr_parameter("metallic", 0.5)
        assert is_valid

    def test_validate_invalid_float(self):
        """Test validating invalid float parameter."""
        is_valid, error = validate_pbr_parameter("metallic", 1.5)
        assert not is_valid
        assert "must be in" in error

    def test_validate_vec4(self):
        """Test validating Vec4 parameter."""
        is_valid, _ = validate_pbr_parameter(
            "base_color",
            Vec4(0.5, 0.5, 0.5, 1.0),
        )
        assert is_valid

    def test_validate_wrong_type(self):
        """Test validating wrong type."""
        is_valid, error = validate_pbr_parameter("metallic", "not a number")
        assert not is_valid
        assert "must be a number" in error

    def test_validate_unknown_parameter(self):
        """Test validating unknown parameter."""
        is_valid, error = validate_pbr_parameter("unknown", 0.5)
        assert not is_valid
        assert "Unknown parameter" in error

    def test_clamp_float(self):
        """Test clamping float parameter."""
        result = clamp_pbr_parameter("metallic", 1.5)
        assert result == 1.0

        result = clamp_pbr_parameter("roughness", -0.5)
        assert result == 0.0

    def test_clamp_vec4(self):
        """Test clamping Vec4 parameter."""
        result = clamp_pbr_parameter(
            "base_color",
            Vec4(1.5, -0.5, 0.5, 2.0),
        )
        assert result == Vec4(1.0, 0.0, 0.5, 1.0)

    def test_clamp_unknown_passthrough(self):
        """Test that unknown parameters pass through unchanged."""
        result = clamp_pbr_parameter("unknown", 999)
        assert result == 999
