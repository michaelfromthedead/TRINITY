"""Tests for ShaderAsset."""
import pytest

from engine.resource.types.shader_asset import ShaderAsset, ShaderStage


def _shader(**kw):
    defaults = dict(
        asset_id=30, name="pbr_vert", path="/s.glsl", size_bytes=2048,
        stage=ShaderStage.VERTEX,
    )
    defaults.update(kw)
    return ShaderAsset(**defaults)


class TestShaderAsset:
    def test_creation(self):
        s = _shader()
        assert s.stage is ShaderStage.VERTEX
        assert s.source_code is None

    def test_compile_stub(self):
        src = "uniform vec3 uColor;\nvoid main(){}"
        s = _shader(source_code=src)
        binary = s.compile()
        assert isinstance(binary, bytes)
        assert s.is_loaded()

    def test_compile_no_source_raises(self):
        s = _shader()
        with pytest.raises(RuntimeError):
            s.compile()

    def test_get_uniforms(self):
        src = "uniform vec3 uColor;\nuniform float uRoughness;\nvoid main(){}"
        s = _shader(source_code=src)
        uniforms = s.get_uniforms()
        assert "uColor" in uniforms
        assert "uRoughness" in uniforms
        assert len(uniforms) == 2

    def test_get_uniforms_empty(self):
        s = _shader()
        assert s.get_uniforms() == []

    def test_stages_count(self):
        for name in ("VERTEX", "FRAGMENT", "COMPUTE", "GEOMETRY", "TESSELLATION"):
            assert hasattr(ShaderStage, name), f"ShaderStage.{name} missing"
