"""Tests for MeshStreamManager."""

from engine.resource.streaming.mesh_streaming import LODStreamRequest, MeshStreamManager


class TestLODStreamRequest:
    def test_default_current_lod(self) -> None:
        req = LODStreamRequest(mesh_id="m", target_lod=2)
        assert req.current_lod == -1


class TestMeshStreamManager:
    def test_unloaded_returns_negative_one(self) -> None:
        mgr = MeshStreamManager()
        assert mgr.get_resident_lod("unknown") == -1

    def test_request_and_update_loads_lod(self) -> None:
        mgr = MeshStreamManager()
        mgr.request_lod("mesh_01", lod_level=2)
        mgr.update()
        assert mgr.get_resident_lod("mesh_01") == 2

    def test_multiple_requests_last_wins(self) -> None:
        mgr = MeshStreamManager()
        mgr.request_lod("mesh_01", lod_level=3)
        mgr.request_lod("mesh_01", lod_level=0)
        mgr.update()
        assert mgr.get_resident_lod("mesh_01") == 0

    def test_request_records_current_lod(self) -> None:
        mgr = MeshStreamManager()
        mgr.request_lod("mesh_01", lod_level=2)
        mgr.update()
        req = mgr.request_lod("mesh_01", lod_level=1)
        assert req.current_lod == 2

    def test_pending_cleared_after_update(self) -> None:
        mgr = MeshStreamManager()
        mgr.request_lod("m1", 0)
        mgr.request_lod("m2", 1)
        mgr.update()
        # Second update should not change anything
        mgr.update()
        assert mgr.get_resident_lod("m1") == 0
        assert mgr.get_resident_lod("m2") == 1
