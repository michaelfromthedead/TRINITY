"""Mesh asset type with vertex formats, submeshes, and LOD support."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from engine.resource.constants import BYTES_PER_FLOAT, BYTES_PER_INDEX
from engine.resource.types.base_asset import BaseAsset

__all__ = ["VertexFormat", "SubMesh", "MeshAsset"]


class VertexFormat(Enum):
    """Vertex attribute layouts."""
    P3 = auto()          # position only (3 floats)
    P3N3 = auto()        # position + normal (6 floats)
    P3N3T2 = auto()      # position + normal + texcoord (8 floats)
    P3N3T2T3 = auto()    # position + normal + texcoord + tangent (11 floats)


# Floats per vertex for each format.
_FLOATS_PER_VERTEX: dict[VertexFormat, int] = {
    VertexFormat.P3: 3,
    VertexFormat.P3N3: 6,
    VertexFormat.P3N3T2: 8,
    VertexFormat.P3N3T2T3: 11,
}


@dataclass(frozen=True, slots=True)
class SubMesh:
    """A contiguous range of indices sharing one material."""
    start_index: int
    index_count: int
    material_index: int


class MeshAsset(BaseAsset):
    """A 3-D mesh asset."""

    __slots__ = (
        "_vertex_count", "_index_count", "_vertex_format",
        "_submeshes", "_lod_levels", "_vertex_data", "_index_data",
    )

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        vertex_count: int,
        index_count: int,
        vertex_format: VertexFormat,
        submeshes: list[SubMesh] | None = None,
        lod_levels: list[MeshAsset] | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._vertex_count = vertex_count
        self._index_count = index_count
        self._vertex_format = vertex_format
        self._submeshes: list[SubMesh] = submeshes or []
        self._lod_levels: list[MeshAsset] = lod_levels or []
        self._vertex_data: bytes | None = None
        self._index_data: bytes | None = None

    @property
    def vertex_count(self) -> int:
        return self._vertex_count

    @property
    def index_count(self) -> int:
        return self._index_count

    @property
    def vertex_format(self) -> VertexFormat:
        return self._vertex_format

    @property
    def submeshes(self) -> list[SubMesh]:
        return list(self._submeshes)

    @property
    def lod_levels(self) -> list[MeshAsset]:
        return list(self._lod_levels)

    def add_lod(self, lod: MeshAsset) -> None:
        self._lod_levels.append(lod)

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        """Load vertex data. For full mesh loading, also call set_index_data()."""
        self._vertex_data = data

    def set_index_data(self, data: bytes) -> None:
        """Set index buffer data separately from vertex data."""
        self._index_data = data

    def unload(self) -> None:
        self._vertex_data = None
        self._index_data = None

    def is_loaded(self) -> bool:
        return self._vertex_data is not None

    @property
    def memory_footprint(self) -> int:
        floats = _FLOATS_PER_VERTEX[self._vertex_format]
        vb = self._vertex_count * floats * BYTES_PER_FLOAT
        ib = self._index_count * BYTES_PER_INDEX
        return vb + ib
