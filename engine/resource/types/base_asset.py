"""Base asset abstract class for all engine resource types."""
from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseAsset"]


class BaseAsset(ABC):
    """Abstract base for all loadable engine assets."""

    __slots__ = ("_asset_id", "_name", "_path", "_size_bytes", "_version")

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        version: int = 1,
    ) -> None:
        self._asset_id = asset_id
        self._name = name
        self._path = path
        self._size_bytes = size_bytes
        self._version = version

    # --- read-only properties ---

    @property
    def asset_id(self) -> int:
        return self._asset_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def version(self) -> int:
        return self._version

    # --- abstract interface ---

    @abstractmethod
    def load(self, data: bytes) -> None:
        """Load asset from raw byte data."""

    @abstractmethod
    def unload(self) -> None:
        """Release loaded data and free memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if asset data is currently in memory."""

    @property
    def memory_footprint(self) -> int:
        """Return estimated runtime memory usage in bytes."""
        return self._size_bytes

    def __repr__(self) -> str:
        loaded = self.is_loaded()
        return (
            f"<{type(self).__name__} id={self._asset_id} "
            f"name={self._name!r} loaded={loaded}>"
        )
