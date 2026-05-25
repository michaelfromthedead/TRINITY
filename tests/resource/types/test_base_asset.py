"""Tests for BaseAsset ABC."""
import pytest

from engine.resource.types.base_asset import BaseAsset


class _ConcreteAsset(BaseAsset):
    """Minimal concrete implementation for testing."""
    __slots__ = ("_loaded",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loaded = False

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


def _make(**overrides):
    defaults = dict(asset_id=1, name="test", path="/a.bin", size_bytes=1024, version=2)
    defaults.update(overrides)
    return _ConcreteAsset(**defaults)


class TestBaseAsset:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseAsset(asset_id=1, name="x", path="/x", size_bytes=0)

    def test_properties(self):
        a = _make()
        assert a.asset_id == 1
        assert a.name == "test"
        assert a.path == "/a.bin"
        assert a.size_bytes == 1024
        assert a.version == 2

    def test_load_unload_cycle(self):
        a = _make()
        assert not a.is_loaded()
        a.load(b"data")
        assert a.is_loaded()
        a.unload()
        assert not a.is_loaded()

    def test_memory_footprint_default(self):
        a = _make(size_bytes=4096)
        assert a.memory_footprint == 4096

    def test_repr(self):
        a = _make(asset_id=42, name="hero")
        r = repr(a)
        assert "42" in r
        assert "hero" in r
        assert "loaded=False" in r

    def test_slots_enforced(self):
        a = _make()
        with pytest.raises(AttributeError):
            a.random_attr = True  # type: ignore[attr-defined]
