"""Package pipeline — bundles cooked assets into distributable packages."""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.resource.constants import CRC32_MASK


class PackageFormat(Enum):
    """Supported package formats."""

    PAK = "pak"
    ZIP = "zip"
    DIRECTORY = "directory"


@dataclass(slots=True)
class PackageEntry:
    """A single entry within a package."""

    asset_path: str
    offset: int
    size: int
    compressed_size: int
    checksum: int


class PackageManifest:
    """Manifest describing all entries in a package."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, PackageEntry] = {}

    @property
    def entries(self) -> dict[str, PackageEntry]:
        return dict(self._entries)

    @property
    def total_size(self) -> int:
        return sum(e.size for e in self._entries.values())

    @property
    def asset_count(self) -> int:
        return len(self._entries)

    def add_entry(self, entry: PackageEntry) -> None:
        self._entries[entry.asset_path] = entry

    def get_entry(self, path: str) -> Optional[PackageEntry]:
        return self._entries.get(path)


class PackageBuilder:
    """Builds a package from cooked assets."""

    __slots__ = ("_assets",)

    def __init__(self) -> None:
        self._assets: dict[str, bytes] = {}

    def add_asset(self, path: str, data: bytes) -> None:
        """Add an asset to the package."""
        self._assets[path] = data

    def build(self, fmt: PackageFormat, output_path: str) -> PackageManifest:
        """Build the package and return its manifest."""
        manifest = PackageManifest()
        offset = 0

        for path, data in self._assets.items():
            checksum = zlib.crc32(data) & CRC32_MASK
            entry = PackageEntry(
                asset_path=path,
                offset=offset,
                size=len(data),
                compressed_size=len(data),
                checksum=checksum,
            )
            manifest.add_entry(entry)
            offset += len(data)

        return manifest


class PackageReader:
    """Reads assets from a package."""

    __slots__ = ("_manifest", "_data")

    def __init__(self) -> None:
        self._manifest: Optional[PackageManifest] = None
        self._data: bytes = b""

    def open(self, path: str, manifest: PackageManifest, data: bytes) -> PackageManifest:
        """Open a package. For in-memory use, accepts manifest and raw data."""
        self._manifest = manifest
        self._data = data
        return manifest

    def read_asset(self, entry: PackageEntry) -> bytes:
        """Read asset bytes from the package data."""
        return self._data[entry.offset : entry.offset + entry.size]
