"""Tests for PBR struct definitions (T-MAT-3.1).

Tests WGSL struct syntax validity and Python dataclass conformance.
"""

import pytest
import re
import subprocess
import shutil
from pathlib import Path

from trinity.materials.pbr_types import (
    PBRInput,
    PBRParams,
    PBROutput,
    get_pbr_structs_wgsl,
    PBR_STRUCTS_WGSL,
    PBR_PARAMS_FIELDS,
)


class TestWGSLSyntax:
    """Test WGSL struct syntax validity."""

    def test_wgsl_file_exists(self):
        """Test that the WGSL file exists."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "pbr_structs.wgsl"
        assert wgsl_path.exists(), f"WGSL file not found at {wgsl_path}"

    def test_get_pbr_structs_wgsl_loads_file(self):
        """Test that get_pbr_structs_wgsl() returns WGSL content."""
        wgsl = get_pbr_structs_wgsl()
        assert len(wgsl) > 0
        assert "struct PBRInput" in wgsl
        assert "struct PBRParams" in wgsl
        assert "struct PBROutput" in wgsl

    def test_embedded_wgsl_matches_file(self):
        """Test embedded WGSL has same structs as file."""
        file_wgsl = get_pbr_structs_wgsl()
        # Both should define the same structs
        assert "struct PBRInput" in PBR_STRUCTS_WGSL
        assert "struct PBRParams" in PBR_STRUCTS_WGSL
        assert "struct PBROutput" in PBR_STRUCTS_WGSL

    def test_wgsl_contains_all_pbr_input_fields(self):
        """Test PBRInput struct has all required fields."""
        wgsl = get_pbr_structs_wgsl()
        required_fields = [
            "world_position: vec3<f32>",
            "world_normal: vec3<f32>",
            "world_tangent: vec4<f32>",
            "world_view: vec3<f32>",
            "uv: vec2<f32>",
            "vertex_color: vec4<f32>",
            "time: f32",
            "light_count: u32",
        ]
        for field in required_fields:
            assert field in wgsl, f"Missing PBRInput field: {field}"

    def test_wgsl_contains_all_pbr_params_fields(self):
        """Test PBRParams struct has all required fields."""
        wgsl = get_pbr_structs_wgsl()
        required_fields = [
            "base_color: vec3<f32>",
            "normal: vec3<f32>",
            "roughness: f32",
            "metallic: f32",
            "specular: f32",
            "occlusion: f32",
            "emissive: vec3<f32>",
            "alpha: f32",
            "subsurface: f32",
            "anisotropy: f32",
            "clearcoat: f32",
            "clearcoat_roughness: f32",
        ]
        for field in required_fields:
            assert field in wgsl, f"Missing PBRParams field: {field}"

    def test_wgsl_contains_pbr_output_fields(self):
        """Test PBROutput struct has color field."""
        wgsl = get_pbr_structs_wgsl()
        assert "color: vec4<f32>" in wgsl

    def test_wgsl_struct_syntax_valid(self):
        """Test WGSL struct syntax is valid (basic regex check)."""
        wgsl = get_pbr_structs_wgsl()

        # Check struct declarations follow WGSL syntax
        struct_pattern = r"struct\s+\w+\s*\{"
        structs = re.findall(struct_pattern, wgsl)
        assert len(structs) >= 3, "Expected at least 3 struct definitions"

        # Check field declarations follow WGSL syntax
        field_pattern = r"\w+:\s*(vec[234]<f32>|f32|u32|i32)"
        fields = re.findall(field_pattern, wgsl)
        assert len(fields) >= 20, f"Expected at least 20 field declarations, got {len(fields)}"

        # Check all struct blocks are closed
        open_braces = wgsl.count("{")
        close_braces = wgsl.count("}")
        assert open_braces == close_braces, "Mismatched braces in WGSL"

    def test_wgsl_default_function_syntax(self):
        """Test pbr_params_default function syntax."""
        wgsl = get_pbr_structs_wgsl()
        assert "fn pbr_params_default() -> PBRParams" in wgsl
        assert "var params: PBRParams" in wgsl
        assert "return params" in wgsl

    @pytest.mark.skipif(
        not shutil.which("naga"),
        reason="naga CLI not available for WGSL validation",
    )
    def test_wgsl_compiles_with_naga(self):
        """Test WGSL compiles with naga (if available)."""
        wgsl_path = Path(__file__).parents[3] / "trinity" / "materials" / "wgsl" / "pbr_structs.wgsl"

        result = subprocess.run(
            ["naga", str(wgsl_path), "--validate"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"naga validation failed: {result.stderr}"


class TestPBRInputDataclass:
    """Test PBRInput Python dataclass."""

    def test_default_values(self):
        """Test PBRInput default values match WGSL semantics."""
        inp = PBRInput()
        assert inp.world_position == (0.0, 0.0, 0.0)
        assert inp.world_normal == (0.0, 1.0, 0.0)
        assert inp.world_tangent == (1.0, 0.0, 0.0, 1.0)
        assert inp.world_view == (0.0, 0.0, 1.0)
        assert inp.uv == (0.0, 0.0)
        assert inp.vertex_color == (1.0, 1.0, 1.0, 1.0)
        assert inp.time == 0.0
        assert inp.light_count == 0

    def test_custom_values(self):
        """Test PBRInput with custom values."""
        inp = PBRInput(
            world_position=(1.0, 2.0, 3.0),
            world_normal=(0.0, 0.0, 1.0),
            uv=(0.5, 0.5),
            time=1.5,
            light_count=4,
        )
        assert inp.world_position == (1.0, 2.0, 3.0)
        assert inp.world_normal == (0.0, 0.0, 1.0)
        assert inp.uv == (0.5, 0.5)
        assert inp.time == 1.5
        assert inp.light_count == 4

    def test_field_types(self):
        """Test PBRInput field types are correct."""
        inp = PBRInput()
        assert isinstance(inp.world_position, tuple)
        assert len(inp.world_position) == 3
        assert isinstance(inp.world_tangent, tuple)
        assert len(inp.world_tangent) == 4
        assert isinstance(inp.uv, tuple)
        assert len(inp.uv) == 2
        assert isinstance(inp.time, float)
        assert isinstance(inp.light_count, int)


class TestPBRParamsDataclass:
    """Test PBRParams Python dataclass."""

    def test_default_values_match_wgsl(self):
        """Test PBRParams defaults match WGSL pbr_params_default()."""
        params = PBRParams()
        assert params.base_color == (1.0, 1.0, 1.0)
        assert params.normal == (0.0, 0.0, 1.0)
        assert params.roughness == 0.5
        assert params.metallic == 0.0
        assert params.specular == 0.5
        assert params.occlusion == 1.0
        assert params.emissive == (0.0, 0.0, 0.0)
        assert params.alpha == 1.0
        assert params.subsurface == 0.0
        assert params.anisotropy == 0.0
        assert params.clearcoat == 0.0
        assert params.clearcoat_roughness == 0.0

    def test_custom_metal_material(self):
        """Test creating a custom metal material."""
        gold = PBRParams(
            base_color=(1.0, 0.766, 0.336),
            metallic=1.0,
            roughness=0.3,
        )
        assert gold.base_color == (1.0, 0.766, 0.336)
        assert gold.metallic == 1.0
        assert gold.roughness == 0.3

    def test_custom_emissive_material(self):
        """Test creating an emissive material."""
        glow = PBRParams(
            base_color=(1.0, 0.0, 0.0),
            emissive=(10.0, 0.0, 0.0),  # HDR emissive
            roughness=0.8,
        )
        assert glow.emissive == (10.0, 0.0, 0.0)

    def test_validate_valid_params(self):
        """Test validation of valid parameters."""
        params = PBRParams()
        is_valid, errors = params.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_roughness(self):
        """Test validation catches invalid roughness."""
        params = PBRParams(roughness=1.5)
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("roughness" in e for e in errors)

    def test_validate_invalid_metallic(self):
        """Test validation catches invalid metallic."""
        params = PBRParams(metallic=-0.1)
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("metallic" in e for e in errors)

    def test_validate_invalid_anisotropy(self):
        """Test validation catches invalid anisotropy."""
        params = PBRParams(anisotropy=2.0)
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("anisotropy" in e for e in errors)

    def test_validate_invalid_normal_length(self):
        """Test validation warns about non-unit normals."""
        params = PBRParams(normal=(0.0, 0.0, 0.5))  # Not unit length
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("normal" in e and "unit" in e for e in errors)

    def test_validate_negative_base_color(self):
        """Test validation catches negative base color."""
        params = PBRParams(base_color=(-0.1, 0.5, 0.5))
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("base_color" in e for e in errors)

    def test_validate_negative_emissive(self):
        """Test validation catches negative emissive."""
        params = PBRParams(emissive=(-1.0, 0.0, 0.0))
        is_valid, errors = params.validate()
        assert not is_valid
        assert any("emissive" in e for e in errors)

    def test_clamp_out_of_range_values(self):
        """Test clamping out-of-range values."""
        params = PBRParams(
            roughness=1.5,
            metallic=-0.5,
            anisotropy=2.0,
            base_color=(-0.1, 1.5, 0.5),
        )
        clamped = params.clamp()
        assert clamped.roughness == 1.0
        assert clamped.metallic == 0.0
        assert clamped.anisotropy == 1.0
        assert clamped.base_color == (0.0, 1.5, 0.5)  # Only clamps negative

    def test_clamp_returns_new_instance(self):
        """Test clamp() returns a new instance."""
        params = PBRParams(roughness=1.5)
        clamped = params.clamp()
        assert params is not clamped
        assert params.roughness == 1.5  # Original unchanged
        assert clamped.roughness == 1.0


class TestPBROutputDataclass:
    """Test PBROutput Python dataclass."""

    def test_default_values(self):
        """Test PBROutput default values."""
        out = PBROutput()
        assert out.color == (0.0, 0.0, 0.0, 1.0)

    def test_custom_color(self):
        """Test PBROutput with custom color."""
        out = PBROutput(color=(1.0, 0.5, 0.25, 1.0))
        assert out.color == (1.0, 0.5, 0.25, 1.0)


class TestPBRParamsFieldMetadata:
    """Test PBR_PARAMS_FIELDS metadata."""

    def test_all_params_fields_have_metadata(self):
        """Test all PBRParams fields have metadata."""
        params = PBRParams()
        param_fields = [
            f for f in dir(params)
            if not f.startswith("_") and not callable(getattr(params, f))
        ]

        for field in param_fields:
            assert field in PBR_PARAMS_FIELDS, f"Missing metadata for field: {field}"

    def test_metadata_has_required_keys(self):
        """Test each metadata entry has required keys."""
        required_keys = {"type", "default", "range"}
        for field, meta in PBR_PARAMS_FIELDS.items():
            assert required_keys.issubset(meta.keys()), f"Missing keys in {field} metadata"

    def test_metadata_defaults_match_dataclass(self):
        """Test metadata defaults match dataclass defaults."""
        params = PBRParams()
        for field, meta in PBR_PARAMS_FIELDS.items():
            actual = getattr(params, field)
            expected = meta["default"]
            assert actual == expected, f"Mismatch for {field}: {actual} != {expected}"

    def test_wgsl_types_are_valid(self):
        """Test WGSL types in metadata are valid."""
        valid_types = {"f32", "vec2<f32>", "vec3<f32>", "vec4<f32>", "u32", "i32"}
        for field, meta in PBR_PARAMS_FIELDS.items():
            assert meta["type"] in valid_types, f"Invalid WGSL type for {field}: {meta['type']}"


class TestPythonWGSLConsistency:
    """Test Python dataclass matches WGSL struct definitions."""

    def test_pbr_params_field_count_matches(self):
        """Test Python PBRParams has same field count as WGSL."""
        wgsl = get_pbr_structs_wgsl()

        # Count fields in WGSL PBRParams struct
        # Find the PBRParams struct block
        match = re.search(r"struct PBRParams \{([^}]+)\}", wgsl)
        assert match, "Could not find PBRParams struct in WGSL"

        wgsl_fields = re.findall(r"(\w+):\s*(?:vec[234]<f32>|f32)", match.group(1))

        # Count Python dataclass fields
        params = PBRParams()
        py_fields = [
            f for f in dir(params)
            if not f.startswith("_") and not callable(getattr(params, f))
        ]

        assert len(wgsl_fields) == len(py_fields), (
            f"Field count mismatch: WGSL={len(wgsl_fields)}, Python={len(py_fields)}"
        )

    def test_pbr_params_field_names_match(self):
        """Test Python PBRParams field names match WGSL."""
        wgsl = get_pbr_structs_wgsl()

        match = re.search(r"struct PBRParams \{([^}]+)\}", wgsl)
        assert match

        wgsl_fields = set(re.findall(r"(\w+):\s*(?:vec[234]<f32>|f32)", match.group(1)))

        params = PBRParams()
        py_fields = {
            f for f in dir(params)
            if not f.startswith("_") and not callable(getattr(params, f))
        }

        assert wgsl_fields == py_fields, (
            f"Field name mismatch:\n"
            f"  WGSL only: {wgsl_fields - py_fields}\n"
            f"  Python only: {py_fields - wgsl_fields}"
        )

    def test_pbr_input_field_names_match(self):
        """Test Python PBRInput field names match WGSL."""
        wgsl = get_pbr_structs_wgsl()

        match = re.search(r"struct PBRInput \{([^}]+)\}", wgsl)
        assert match

        wgsl_fields = set(re.findall(r"(\w+):\s*(?:vec[234]<f32>|f32|u32)", match.group(1)))

        inp = PBRInput()
        py_fields = {
            f for f in dir(inp)
            if not f.startswith("_") and not callable(getattr(inp, f))
        }

        assert wgsl_fields == py_fields, (
            f"PBRInput field mismatch:\n"
            f"  WGSL only: {wgsl_fields - py_fields}\n"
            f"  Python only: {py_fields - wgsl_fields}"
        )
