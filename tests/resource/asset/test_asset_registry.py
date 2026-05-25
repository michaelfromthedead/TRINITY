"""Tests for AssetRegistry."""
import pytest

from engine.resource.asset.asset_registry import AssetRegistry, AssetType


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    AssetRegistry.reset()
    yield  # type: ignore[misc]
    AssetRegistry.reset()


class TestAssetRegistry:
    def test_singleton_identity(self) -> None:
        a = AssetRegistry.instance()
        b = AssetRegistry.instance()
        assert a is b

    def test_default_texture_png(self) -> None:
        reg = AssetRegistry.instance()
        assert reg.lookup("hero.png") == AssetType.TEXTURE

    def test_default_texture_jpg(self) -> None:
        assert AssetRegistry.instance().lookup("bg.jpg") == AssetType.TEXTURE

    def test_default_mesh(self) -> None:
        assert AssetRegistry.instance().lookup("model.fbx") == AssetType.MESH

    def test_default_audio(self) -> None:
        assert AssetRegistry.instance().lookup("sound.wav") == AssetType.AUDIO

    def test_default_shader(self) -> None:
        assert AssetRegistry.instance().lookup("vert.glsl") == AssetType.SHADER

    def test_default_data_table(self) -> None:
        assert AssetRegistry.instance().lookup("config.json") == AssetType.DATA_TABLE

    def test_unknown_extension_returns_none(self) -> None:
        assert AssetRegistry.instance().lookup("readme.xyz") is None

    def test_register_custom_extension(self) -> None:
        reg = AssetRegistry.instance()
        reg.register(".custom", AssetType.MESH)
        assert reg.lookup("thing.custom") == AssetType.MESH

    def test_register_without_dot(self) -> None:
        reg = AssetRegistry.instance()
        reg.register("tga", AssetType.TEXTURE)
        assert reg.lookup("image.tga") == AssetType.TEXTURE
