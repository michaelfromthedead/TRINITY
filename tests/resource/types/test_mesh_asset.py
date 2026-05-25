"""Tests for MeshAsset."""
import pytest

from engine.resource.types.mesh_asset import (
    MeshAsset, VertexFormat, SubMesh, BYTES_PER_FLOAT, BYTES_PER_INDEX,
)


def _mesh(**kw):
    defaults = dict(
        asset_id=10, name="cube", path="/m.mesh", size_bytes=4096,
        vertex_count=24, index_count=36, vertex_format=VertexFormat.P3N3T2,
    )
    defaults.update(kw)
    return MeshAsset(**defaults)


class TestMeshAsset:
    def test_creation(self):
        m = _mesh()
        assert m.vertex_count == 24
        assert m.index_count == 36
        assert m.vertex_format is VertexFormat.P3N3T2

    def test_submeshes(self):
        subs = [SubMesh(0, 18, 0), SubMesh(18, 18, 1)]
        m = _mesh(submeshes=subs)
        assert len(m.submeshes) == 2
        assert m.submeshes[0].material_index == 0

    def test_lod_levels(self):
        lod1 = _mesh(asset_id=11, vertex_count=12, index_count=18)
        m = _mesh()
        m.add_lod(lod1)
        assert len(m.lod_levels) == 1
        assert m.lod_levels[0].vertex_count == 12

    def test_memory_footprint(self):
        m = _mesh(vertex_count=100, index_count=300, vertex_format=VertexFormat.P3)
        # 100 verts * 3 floats * 4 bytes + 300 indices * 4 bytes
        expected = 100 * 3 * BYTES_PER_FLOAT + 300 * BYTES_PER_INDEX
        assert m.memory_footprint == expected

    def test_load_unload(self):
        m = _mesh()
        assert not m.is_loaded()
        m.load(b"\x00")
        assert m.is_loaded()
        m.unload()
        assert not m.is_loaded()

    def test_vertex_formats_count(self):
        for name in ("P3", "P3N3", "P3N3T2", "P3N3T2T3"):
            assert hasattr(VertexFormat, name), f"VertexFormat.{name} missing"
