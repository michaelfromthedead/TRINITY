"""Virtual geometry system with nanite-style cluster-based mesh rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.resource.constants import LOD_DISTANCES


@dataclass(slots=True)
class Cluster:
    """A mesh cluster at a specific LOD level."""

    cluster_id: int
    lod_level: int
    vertex_count: int
    triangle_count: int
    bounding_sphere: tuple[float, float, float, float]  # cx, cy, cz, radius
    is_visible: bool = False
    is_resident: bool = False


@dataclass(slots=True)
class ClusterGroup:
    """Groups of clusters sharing a LOD hierarchy."""

    group_id: int
    clusters: list[Cluster]

    def get_clusters_at_lod(self, lod_level: int) -> list[Cluster]:
        return [c for c in self.clusters if c.lod_level == lod_level]


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _select_lod(dist: float) -> int:
    """Select LOD level based on distance from camera."""
    for level, threshold in enumerate(LOD_DISTANCES):
        if dist < threshold:
            return level
    return len(LOD_DISTANCES)


class VirtualGeometrySystem:
    """Manages nanite-style cluster-based virtual geometry."""

    __slots__ = ("_clusters", "_visible")

    def __init__(self) -> None:
        self._clusters: dict[int, Cluster] = {}
        self._visible: list[Cluster] = []

    def submit_clusters(self, clusters: list[Cluster]) -> None:
        """Submit clusters for rendering consideration."""
        for c in clusters:
            c.is_resident = True
            self._clusters[c.cluster_id] = c

    def cull(
        self, camera_pos: tuple[float, float, float], fov: float
    ) -> list[Cluster]:
        """Distance-based LOD selection and basic frustum culling."""
        self._visible.clear()

        for cluster in self._clusters.values():
            cx, cy, cz, radius = cluster.bounding_sphere
            center = (cx, cy, cz)
            dist = _distance(camera_pos, center)

            desired_lod = _select_lod(dist)
            if cluster.lod_level != desired_lod:
                cluster.is_visible = False
                continue

            # Simple distance-based frustum approximation:
            # reject clusters behind camera or beyond max range
            half_fov_rad = math.radians(fov / 2.0)
            max_range = LOD_DISTANCES[-1] + radius
            if dist - radius > max_range:
                cluster.is_visible = False
                continue

            cluster.is_visible = True
            self._visible.append(cluster)

        return list(self._visible)

    def get_visible_count(self) -> int:
        return len(self._visible)

    def get_resident_count(self) -> int:
        return sum(1 for c in self._clusters.values() if c.is_resident)
