"""LOD streaming for meshes."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LODStreamRequest", "MeshStreamManager"]


@dataclass(slots=True)
class LODStreamRequest:
    """Request to stream a specific LOD level for a mesh."""

    mesh_id: str = ""
    target_lod: int = 0
    current_lod: int = -1


class MeshStreamManager:
    """Manages LOD streaming for meshes."""

    __slots__ = ("_resident_lods", "_pending")

    def __init__(self) -> None:
        self._resident_lods: dict[str, int] = {}
        self._pending: list[LODStreamRequest] = []

    def request_lod(self, mesh_id: str, lod_level: int) -> LODStreamRequest:
        """Request streaming of a specific LOD level."""
        current = self._resident_lods.get(mesh_id, -1)
        req = LODStreamRequest(mesh_id=mesh_id, target_lod=lod_level, current_lod=current)
        self._pending.append(req)
        return req

    def get_resident_lod(self, mesh_id: str) -> int:
        """Return the currently resident LOD level, or -1 if not loaded."""
        return self._resident_lods.get(mesh_id, -1)

    def update(self) -> None:
        """Process pending LOD requests."""
        for req in self._pending:
            self._resident_lods[req.mesh_id] = req.target_lod
        self._pending.clear()
