"""Tests for TextureAsset."""
import math

import pytest

from engine.resource.types.texture_asset import TextureAsset, TextureFormat


def _tex(**kw):
    defaults = dict(
        asset_id=1, name="diffuse", path="/t.png", size_bytes=256,
        width=256, height=256, channels=4, fmt=TextureFormat.RGBA8,
    )
    defaults.update(kw)
    return TextureAsset(**defaults)


class TestTextureAsset:
    def test_creation(self):
        t = _tex()
        assert t.width == 256
        assert t.height == 256
        assert t.channels == 4
        assert t.format is TextureFormat.RGBA8

    def test_load_unload(self):
        t = _tex()
        assert not t.is_loaded()
        t.load(b"\x00" * 16)
        assert t.is_loaded()
        assert t.memory_footprint == 16
        t.unload()
        assert not t.is_loaded()
        assert t.memory_footprint == 0

    def test_mip_size_level_0(self):
        t = _tex(width=512, height=256, mip_levels=3)
        assert t.get_mip_size(0) == (512, 256)

    def test_mip_size_level_1(self):
        t = _tex(width=512, height=256, mip_levels=3)
        assert t.get_mip_size(1) == (256, 128)

    def test_mip_size_invalid_level(self):
        t = _tex(mip_levels=1)
        with pytest.raises(ValueError):
            t.get_mip_size(1)

    def test_max_mip_levels(self):
        t = _tex(width=64, height=32)
        expected = int(math.log2(64)) + 1  # 7
        assert t.max_mip_levels == expected

    def test_invalid_mip_levels_raises(self):
        with pytest.raises(ValueError):
            _tex(width=64, height=64, mip_levels=100)

    def test_all_formats_exist(self):
        for name in ("RGBA8", "RGB8", "BC1", "BC3", "BC5", "BC7", "R16F", "RGBA16F", "RGBA32F"):
            assert hasattr(TextureFormat, name), f"TextureFormat.{name} missing"
