"""Tests for MaterialAsset."""
import pytest

from engine.resource.types.material_asset import MaterialAsset, BlendMode


def _mat(**kw):
    defaults = dict(
        asset_id=20, name="metal", path="/m.mat", size_bytes=128, shader_id=5,
    )
    defaults.update(kw)
    return MaterialAsset(**defaults)


class TestMaterialAsset:
    def test_creation(self):
        m = _mat()
        assert m.shader_id == 5
        assert m.blend_mode is BlendMode.OPAQUE

    def test_default_render_queue(self):
        assert _mat(blend_mode=BlendMode.OPAQUE).render_queue == 2000
        assert _mat(blend_mode=BlendMode.ALPHA_BLEND).render_queue == 3000

    def test_custom_render_queue(self):
        m = _mat(render_queue=9999)
        assert m.render_queue == 9999

    def test_textures(self):
        m = _mat(textures={"albedo": 100, "normal": 101})
        assert m.textures["albedo"] == 100
        m.set_texture("roughness", 102)
        assert m.textures["roughness"] == 102

    def test_parameters(self):
        m = _mat(parameters={"metallic": 0.9})
        assert m.parameters["metallic"] == pytest.approx(0.9)
        m.set_parameter("color", (1.0, 0.0, 0.0))
        assert m.parameters["color"] == (1.0, 0.0, 0.0)

    def test_load_unload(self):
        m = _mat()
        assert not m.is_loaded()
        m.load(b"")
        assert m.is_loaded()
        m.unload()
        assert not m.is_loaded()

    def test_blend_modes_count(self):
        for name in ("OPAQUE", "ALPHA_TEST", "ALPHA_BLEND", "ADDITIVE"):
            assert hasattr(BlendMode, name), f"BlendMode.{name} missing"
