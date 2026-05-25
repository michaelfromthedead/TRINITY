"""Game packaging with compression, encryption, and DLC support.

Provides functionality for creating distributable game packages
with support for various compression methods, encryption, and DLC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import io
import json
import os
import struct
import threading
import time
import zlib


class CompressionMethod(Enum):
    """Compression algorithms."""
    NONE = auto()
    ZLIB = auto()
    LZ4 = auto()
    ZSTD = auto()
    LZMA = auto()
    BROTLI = auto()


class EncryptionMethod(Enum):
    """Encryption algorithms."""
    NONE = auto()
    AES_128 = auto()
    AES_256 = auto()
    CHACHA20 = auto()


class PackageType(Enum):
    """Type of package."""
    FULL = auto()          # Complete game package
    PATCH = auto()         # Update patch
    DLC = auto()           # Downloadable content
    MOD = auto()           # User modification
    DEMO = auto()          # Demo version


@dataclass
class FileEntry:
    """Entry in a package file table."""
    path: str
    offset: int
    size: int
    compressed_size: int
    compression: CompressionMethod
    encrypted: bool
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DLCInfo:
    """Information about downloadable content."""
    id: str
    name: str
    version: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    size: int = 0
    release_date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "files": self.files,
            "size": self.size,
            "release_date": self.release_date,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DLCInfo:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            files=data.get("files", []),
            size=data.get("size", 0),
            release_date=data.get("release_date", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PackageManifest:
    """Manifest describing package contents."""
    name: str
    version: str
    package_type: PackageType
    platform: str
    files: List[FileEntry] = field(default_factory=list)
    dlc_info: Optional[DLCInfo] = None
    compression: CompressionMethod = CompressionMethod.ZLIB
    encryption: EncryptionMethod = EncryptionMethod.NONE
    created_at: str = ""
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "name": self.name,
            "version": self.version,
            "package_type": self.package_type.name,
            "platform": self.platform,
            "files": [
                {
                    "path": f.path,
                    "offset": f.offset,
                    "size": f.size,
                    "compressed_size": f.compressed_size,
                    "compression": f.compression.name,
                    "encrypted": f.encrypted,
                    "checksum": f.checksum,
                    "metadata": f.metadata,
                }
                for f in self.files
            ],
            "dlc_info": self.dlc_info.to_dict() if self.dlc_info else None,
            "compression": self.compression.name,
            "encryption": self.encryption.name,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PackageManifest:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        files = [
            FileEntry(
                path=f["path"],
                offset=f["offset"],
                size=f["size"],
                compressed_size=f["compressed_size"],
                compression=CompressionMethod[f["compression"]],
                encrypted=f["encrypted"],
                checksum=f["checksum"],
                metadata=f.get("metadata", {}),
            )
            for f in data.get("files", [])
        ]

        dlc_info = None
        if data.get("dlc_info"):
            dlc_info = DLCInfo.from_dict(data["dlc_info"])

        return cls(
            name=data["name"],
            version=data["version"],
            package_type=PackageType[data["package_type"]],
            platform=data["platform"],
            files=files,
            dlc_info=dlc_info,
            compression=CompressionMethod[data.get("compression", "NONE")],
            encryption=EncryptionMethod[data.get("encryption", "NONE")],
            created_at=data.get("created_at", ""),
            checksum=data.get("checksum", ""),
            metadata=data.get("metadata", {}),
        )

    def get_total_size(self) -> int:
        """Get total uncompressed size."""
        return sum(f.size for f in self.files)

    def get_compressed_size(self) -> int:
        """Get total compressed size."""
        return sum(f.compressed_size for f in self.files)


class Compressor(ABC):
    """Abstract base for compressors."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass


class ZlibCompressor(Compressor):
    """Zlib compression."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class NullCompressor(Compressor):
    """No compression."""

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class PackageEncryption:
    """Handles package encryption."""

    def __init__(self, method: EncryptionMethod = EncryptionMethod.NONE, key: Optional[bytes] = None):
        self.method = method
        self._key = key

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        if self.method == EncryptionMethod.NONE:
            return data

        # Placeholder - in real implementation, use cryptographic libraries
        # This is a simple XOR for demonstration
        if self._key:
            key_bytes = self._key * (len(data) // len(self._key) + 1)
            return bytes(a ^ b for a, b in zip(data, key_bytes[:len(data)]))
        return data

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        # XOR encryption is symmetric
        return self.encrypt(data)

    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        if self.method == EncryptionMethod.AES_128:
            return os.urandom(16)
        elif self.method == EncryptionMethod.AES_256:
            return os.urandom(32)
        elif self.method == EncryptionMethod.CHACHA20:
            return os.urandom(32)
        return b""


class PackageBuilder:
    """Builds game packages."""

    # Package file signature
    SIGNATURE = b"GPKG"
    VERSION = 1

    def __init__(
        self,
        compression: CompressionMethod = CompressionMethod.ZLIB,
        encryption: EncryptionMethod = EncryptionMethod.NONE,
        encryption_key: Optional[bytes] = None
    ):
        self.compression = compression
        self.encryption = PackageEncryption(encryption, encryption_key)
        self._compressor = self._get_compressor(compression)

    def _get_compressor(self, method: CompressionMethod) -> Compressor:
        """Get compressor for the specified method."""
        if method == CompressionMethod.ZLIB:
            return ZlibCompressor()
        # Add other compressors as needed
        return NullCompressor()

    def build(
        self,
        source_dir: str,
        output_path: str,
        manifest: PackageManifest,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Build a package from source files."""
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")

            # Collect all files
            files_to_package = []
            for root, _, files in os.walk(source_dir):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, source_dir)
                    files_to_package.append((full_path, rel_path))

            total_files = len(files_to_package)

            with open(output_path, "wb") as out_file:
                # Write header placeholder
                header_size = self._write_header(out_file, manifest, 0)

                # Write file data
                current_offset = header_size
                file_entries = []

                for idx, (full_path, rel_path) in enumerate(files_to_package):
                    if progress_callback:
                        progress_callback(idx, total_files)

                    entry = self._write_file(out_file, full_path, rel_path, current_offset)
                    file_entries.append(entry)
                    current_offset = entry.offset + entry.compressed_size

                # Update manifest with file entries
                manifest.files = file_entries
                manifest.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Compute package checksum by reopening in read mode
            with open(output_path, "rb") as check_file:
                check_file.seek(header_size)
                hasher = hashlib.sha256()
                while True:
                    chunk = check_file.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
                manifest.checksum = hasher.hexdigest()

            # Rewrite header with final manifest including checksum
            with open(output_path, "r+b") as out_file:
                out_file.seek(0)
                self._write_header(out_file, manifest, current_offset - header_size)

            # Write manifest file alongside package
            manifest_path = output_path + ".manifest"
            with open(manifest_path, "w") as f:
                f.write(manifest.to_json())

            return True

        except Exception as e:
            # Clean up partial file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

    def _write_header(self, out_file: BinaryIO, manifest: PackageManifest, data_size: int) -> int:
        """Write package header."""
        # Signature (4 bytes)
        out_file.write(self.SIGNATURE)

        # Version (4 bytes)
        out_file.write(struct.pack("<I", self.VERSION))

        # Package type (4 bytes)
        out_file.write(struct.pack("<I", manifest.package_type.value))

        # Compression method (4 bytes)
        out_file.write(struct.pack("<I", manifest.compression.value))

        # Encryption method (4 bytes)
        out_file.write(struct.pack("<I", manifest.encryption.value))

        # Data size (8 bytes)
        out_file.write(struct.pack("<Q", data_size))

        # File count (4 bytes)
        out_file.write(struct.pack("<I", len(manifest.files)))

        # Reserved (36 bytes for future use)
        out_file.write(b"\x00" * 36)

        return 64  # Total header size

    def _write_file(
        self,
        out_file: BinaryIO,
        full_path: str,
        rel_path: str,
        offset: int
    ) -> FileEntry:
        """Write a single file to the package."""
        with open(full_path, "rb") as f:
            data = f.read()

        original_size = len(data)

        # Compute checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Compress
        compressed_data = self._compressor.compress(data)
        compressed_size = len(compressed_data)

        # Encrypt if needed
        encrypted = self.encryption.method != EncryptionMethod.NONE
        if encrypted:
            compressed_data = self.encryption.encrypt(compressed_data)

        # Write to output
        out_file.seek(offset)
        out_file.write(compressed_data)

        return FileEntry(
            path=rel_path,
            offset=offset,
            size=original_size,
            compressed_size=compressed_size,
            compression=self.compression,
            encrypted=encrypted,
            checksum=checksum,
        )

    def extract(
        self,
        package_path: str,
        output_dir: str,
        manifest: Optional[PackageManifest] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Extract a package to a directory."""
        try:
            # Load manifest if not provided
            if manifest is None:
                manifest_path = package_path + ".manifest"
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = PackageManifest.from_json(f.read())
                else:
                    raise ValueError("No manifest provided or found")

            os.makedirs(output_dir, exist_ok=True)
            total_files = len(manifest.files)

            with open(package_path, "rb") as pkg_file:
                for idx, entry in enumerate(manifest.files):
                    if progress_callback:
                        progress_callback(idx, total_files)

                    # Read compressed data
                    pkg_file.seek(entry.offset)
                    data = pkg_file.read(entry.compressed_size)

                    # Decrypt if needed
                    if entry.encrypted:
                        data = self.encryption.decrypt(data)

                    # Decompress
                    compressor = self._get_compressor(entry.compression)
                    data = compressor.decompress(data)

                    # Verify checksum
                    if hashlib.sha256(data).hexdigest() != entry.checksum:
                        raise ValueError(f"Checksum mismatch for {entry.path}")

                    # Write to output
                    output_path = os.path.join(output_dir, entry.path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(data)

            return True

        except Exception:
            raise


class DLCManager:
    """Manages downloadable content."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._dlc_registry: Dict[str, DLCInfo] = {}
        self._installed: Set[str] = set()
        self._lock = threading.Lock()

    def register(self, dlc: DLCInfo) -> None:
        """Register DLC in the manager."""
        with self._lock:
            self._dlc_registry[dlc.id] = dlc

    def unregister(self, dlc_id: str) -> bool:
        """Unregister DLC."""
        with self._lock:
            if dlc_id in self._dlc_registry:
                del self._dlc_registry[dlc_id]
                self._installed.discard(dlc_id)
                return True
            return False

    def get_dlc(self, dlc_id: str) -> Optional[DLCInfo]:
        """Get DLC info by ID."""
        return self._dlc_registry.get(dlc_id)

    def get_all_dlc(self) -> List[DLCInfo]:
        """Get all registered DLC."""
        return list(self._dlc_registry.values())

    def get_installed(self) -> List[DLCInfo]:
        """Get all installed DLC."""
        return [self._dlc_registry[dlc_id] for dlc_id in self._installed if dlc_id in self._dlc_registry]

    def is_installed(self, dlc_id: str) -> bool:
        """Check if DLC is installed."""
        return dlc_id in self._installed

    def install(self, dlc_id: str, package_path: str) -> bool:
        """Install DLC from a package."""
        dlc = self._dlc_registry.get(dlc_id)
        if not dlc:
            return False

        # Check dependencies
        for dep_id in dlc.dependencies:
            if not self.is_installed(dep_id):
                raise ValueError(f"Missing dependency: {dep_id}")

        # Extract package
        builder = PackageBuilder()
        output_dir = os.path.join(self.base_dir, "DLC", dlc_id)

        manifest = PackageManifest(
            name=dlc.name,
            version=dlc.version,
            package_type=PackageType.DLC,
            platform="",
            dlc_info=dlc,
        )

        if builder.extract(package_path, output_dir, manifest):
            with self._lock:
                self._installed.add(dlc_id)
            return True

        return False

    def uninstall(self, dlc_id: str) -> bool:
        """Uninstall DLC."""
        if not self.is_installed(dlc_id):
            return False

        # Check if other DLC depends on this
        for dlc in self._dlc_registry.values():
            if dlc.id != dlc_id and dlc_id in dlc.dependencies and dlc.id in self._installed:
                raise ValueError(f"Cannot uninstall: {dlc.id} depends on {dlc_id}")

        # Remove files
        dlc_dir = os.path.join(self.base_dir, "DLC", dlc_id)
        if os.path.exists(dlc_dir):
            import shutil
            shutil.rmtree(dlc_dir)

        with self._lock:
            self._installed.discard(dlc_id)

        return True

    def verify(self, dlc_id: str) -> List[str]:
        """Verify DLC installation integrity."""
        errors = []
        dlc = self._dlc_registry.get(dlc_id)

        if not dlc:
            errors.append(f"DLC not found: {dlc_id}")
            return errors

        if not self.is_installed(dlc_id):
            errors.append(f"DLC not installed: {dlc_id}")
            return errors

        dlc_dir = os.path.join(self.base_dir, "DLC", dlc_id)
        for file_path in dlc.files:
            full_path = os.path.join(dlc_dir, file_path)
            if not os.path.exists(full_path):
                errors.append(f"Missing file: {file_path}")

        return errors

    def save_registry(self, path: str) -> None:
        """Save DLC registry to file."""
        data = {
            "dlc": [dlc.to_dict() for dlc in self._dlc_registry.values()],
            "installed": list(self._installed),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str) -> None:
        """Load DLC registry from file."""
        with open(path, "r") as f:
            data = json.load(f)

        with self._lock:
            self._dlc_registry.clear()
            self._installed.clear()

            for dlc_data in data.get("dlc", []):
                dlc = DLCInfo.from_dict(dlc_data)
                self._dlc_registry[dlc.id] = dlc

            self._installed = set(data.get("installed", []))
