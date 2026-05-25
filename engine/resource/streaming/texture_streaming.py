"""Mip-level streaming for textures."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MipStreamRequest", "TextureStreamManager"]


@dataclass(slots=True)
class MipStreamRequest:
    """Request to stream a specific mip level for a texture."""

    texture_id: str = ""
    target_mip_level: int = 0
    current_mip_level: int = 0

    @property
    def priority(self) -> int:
        """Lower target mip = higher priority (more detail = more important)."""
        return self.target_mip_level


class TextureStreamManager:
    """Manages mip-level streaming for textures."""

    __slots__ = ("_resident_mips", "_pending")

    def __init__(self) -> None:
        self._resident_mips: dict[str, int] = {}
        self._pending: list[MipStreamRequest] = []

    def request_mip(self, texture_id: str, mip_level: int) -> MipStreamRequest:
        """Request streaming of a specific mip level."""
        current = self._resident_mips.get(texture_id, -1)
        req = MipStreamRequest(
            texture_id=texture_id,
            target_mip_level=mip_level,
            current_mip_level=current,
        )
        self._pending.append(req)
        # Sort by priority (lower mip = higher priority).
        self._pending.sort(key=lambda r: r.priority)
        return req

    def get_resident_mip(self, texture_id: str) -> int:
        """Return the currently resident mip level, or -1 if not loaded."""
        return self._resident_mips.get(texture_id, -1)

    def update(self) -> None:
        """Process pending mip requests, keeping the best (lowest) mip per texture."""
        best: dict[str, int] = {}
        for req in self._pending:
            prev = best.get(req.texture_id)
            if prev is None or req.target_mip_level < prev:
                best[req.texture_id] = req.target_mip_level
        for tex_id, mip in best.items():
            self._resident_mips[tex_id] = mip
        self._pending.clear()
