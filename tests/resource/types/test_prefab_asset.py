"""Tests for PrefabAsset."""
import pytest

from engine.resource.types.prefab_asset import PrefabAsset


def _prefab(**kw):
    defaults = dict(asset_id=60, name="enemy", path="/p.prefab", size_bytes=64)
    defaults.update(kw)
    return PrefabAsset(**defaults)


class TestPrefabAsset:
    def test_creation(self):
        p = _prefab(components=[{"type": "Transform", "x": 0}])
        assert len(p.components) == 1

    def test_children(self):
        child = _prefab(asset_id=61, name="weapon")
        parent = _prefab(children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].name == "weapon"

    def test_add_child(self):
        p = _prefab()
        p.add_child(_prefab(asset_id=62))
        assert len(p.children) == 1

    def test_instantiate(self):
        child = _prefab(asset_id=61, name="child", components=[{"type": "Mesh"}])
        parent = _prefab(components=[{"type": "Transform"}], children=[child])
        result = parent.instantiate()
        assert result["name"] == "enemy"
        assert len(result["components"]) == 1
        assert len(result["children"]) == 1
        assert result["children"][0]["name"] == "child"

    def test_load_unload(self):
        p = _prefab()
        assert not p.is_loaded()
        p.load(b"")
        assert p.is_loaded()
        p.unload()
        assert not p.is_loaded()
