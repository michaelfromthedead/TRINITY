"""Tests for the virtual geometry system."""

import pytest

from engine.resource.virtualization.virtual_geometry import (
    LOD_DISTANCES,
    Cluster,
    ClusterGroup,
    VirtualGeometrySystem,
)


def _make_cluster(
    cid: int, lod: int = 0, pos: tuple = (0.0, 0.0, 0.0), radius: float = 1.0
) -> Cluster:
    return Cluster(
        cluster_id=cid,
        lod_level=lod,
        vertex_count=100,
        triangle_count=50,
        bounding_sphere=(pos[0], pos[1], pos[2], radius),
    )


class TestCluster:
    def test_defaults(self) -> None:
        c = _make_cluster(1)
        assert c.is_visible is False
        assert c.is_resident is False


class TestClusterGroup:
    def test_get_clusters_at_lod(self) -> None:
        clusters = [_make_cluster(i, lod=i % 3) for i in range(9)]
        group = ClusterGroup(group_id=0, clusters=clusters)
        lod0 = group.get_clusters_at_lod(0)
        assert len(lod0) == 3
        assert all(c.lod_level == 0 for c in lod0)


class TestVirtualGeometrySystem:
    def test_submit_marks_resident(self) -> None:
        vgs = VirtualGeometrySystem()
        clusters = [_make_cluster(i) for i in range(5)]
        vgs.submit_clusters(clusters)
        assert vgs.get_resident_count() == 5

    def test_cull_near_lod0(self) -> None:
        vgs = VirtualGeometrySystem()
        # Cluster at origin, LOD 0 -- camera very close
        vgs.submit_clusters([_make_cluster(0, lod=0, pos=(10.0, 0.0, 0.0))])
        visible = vgs.cull(camera_pos=(0.0, 0.0, 0.0), fov=90.0)
        assert len(visible) == 1
        assert visible[0].is_visible is True

    def test_cull_selects_correct_lod(self) -> None:
        vgs = VirtualGeometrySystem()
        # Place cluster at distance ~150, which is LOD 2 (between 100 and 200)
        c_lod0 = _make_cluster(0, lod=0, pos=(150.0, 0.0, 0.0))
        c_lod2 = _make_cluster(1, lod=2, pos=(150.0, 0.0, 0.0))
        vgs.submit_clusters([c_lod0, c_lod2])
        visible = vgs.cull(camera_pos=(0.0, 0.0, 0.0), fov=90.0)
        assert len(visible) == 1
        assert visible[0].cluster_id == 1
        assert visible[0].lod_level == 2

    def test_cull_far_away_not_visible(self) -> None:
        vgs = VirtualGeometrySystem()
        # Place far beyond max LOD distance + radius
        vgs.submit_clusters([_make_cluster(0, lod=4, pos=(1000.0, 0.0, 0.0), radius=1.0)])
        visible = vgs.cull(camera_pos=(0.0, 0.0, 0.0), fov=90.0)
        assert len(visible) == 0
        assert vgs.get_visible_count() == 0

    def test_visible_count(self) -> None:
        vgs = VirtualGeometrySystem()
        # All at origin, LOD 0, camera at origin => distance ~0 => LOD 0
        clusters = [_make_cluster(i, lod=0, pos=(1.0, 0.0, 0.0)) for i in range(3)]
        vgs.submit_clusters(clusters)
        vgs.cull(camera_pos=(0.0, 0.0, 0.0), fov=90.0)
        assert vgs.get_visible_count() == 3

    def test_lod_distances_constant(self) -> None:
        assert len(LOD_DISTANCES) > 0
        assert all(d > 0 for d in LOD_DISTANCES)
        # Each successive LOD distance must be strictly increasing
        assert all(LOD_DISTANCES[i] < LOD_DISTANCES[i + 1] for i in range(len(LOD_DISTANCES) - 1))
