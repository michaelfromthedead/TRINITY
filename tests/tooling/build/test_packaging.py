"""Tests for game packaging with compression, encryption, DLC support."""
import pytest
import os
import tempfile
import shutil
import json
from engine.tooling.build.packaging import (
    CompressionMethod,
    EncryptionMethod,
    PackageType,
    FileEntry,
    DLCInfo,
    PackageManifest,
    PackageBuilder,
    PackageEncryption,
    DLCManager,
    NullCompressor,
    ZlibCompressor,
)


class TestCompressionMethod:
    """Tests for CompressionMethod enum."""

    def test_all_methods_exist(self):
        """Test all compression methods exist."""
        assert CompressionMethod.NONE
        assert CompressionMethod.ZLIB
        assert CompressionMethod.LZ4
        assert CompressionMethod.ZSTD
        assert CompressionMethod.LZMA
        assert CompressionMethod.BROTLI


class TestEncryptionMethod:
    """Tests for EncryptionMethod enum."""

    def test_all_methods_exist(self):
        """Test all encryption methods exist."""
        assert EncryptionMethod.NONE
        assert EncryptionMethod.AES_128
        assert EncryptionMethod.AES_256
        assert EncryptionMethod.CHACHA20


class TestPackageType:
    """Tests for PackageType enum."""

    def test_all_types_exist(self):
        """Test all package types exist."""
        assert PackageType.FULL
        assert PackageType.PATCH
        assert PackageType.DLC
        assert PackageType.MOD
        assert PackageType.DEMO


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_file_entry_creation(self):
        """Test creating file entry."""
        entry = FileEntry(
            path="textures/diffuse.png",
            offset=1024,
            size=4096,
            compressed_size=2048,
            compression=CompressionMethod.ZLIB,
            encrypted=False,
            checksum="abc123",
        )
        assert entry.path == "textures/diffuse.png"
        assert entry.compressed_size < entry.size


class TestDLCInfo:
    """Tests for DLCInfo dataclass."""

    def test_dlc_info_creation(self):
        """Test creating DLC info."""
        dlc = DLCInfo(
            id="dlc_expansion_1",
            name="The Great Expansion",
            version="1.0.0",
            description="Adds new content",
        )
        assert dlc.id == "dlc_expansion_1"
        assert dlc.name == "The Great Expansion"

    def test_to_dict(self):
        """Test converting to dictionary."""
        dlc = DLCInfo(
            id="dlc_1",
            name="DLC 1",
            version="1.0",
            dependencies=["base_game"],
        )
        data = dlc.to_dict()
        assert data["id"] == "dlc_1"
        assert "base_game" in data["dependencies"]

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "dlc_2",
            "name": "DLC 2",
            "version": "2.0",
            "dependencies": [],
            "files": ["content/map.pak"],
            "size": 1024000,
        }
        dlc = DLCInfo.from_dict(data)
        assert dlc.id == "dlc_2"
        assert dlc.size == 1024000


class TestPackageManifest:
    """Tests for PackageManifest dataclass."""

    def test_manifest_creation(self):
        """Test creating package manifest."""
        manifest = PackageManifest(
            name="MyGame",
            version="1.0.0",
            package_type=PackageType.FULL,
            platform="windows",
        )
        assert manifest.name == "MyGame"
        assert manifest.package_type == PackageType.FULL

    def test_to_json(self):
        """Test serializing to JSON."""
        manifest = PackageManifest(
            name="MyGame",
            version="1.0.0",
            package_type=PackageType.FULL,
            platform="windows",
            files=[
                FileEntry(
                    path="data.pak",
                    offset=0,
                    size=1000,
                    compressed_size=500,
                    compression=CompressionMethod.ZLIB,
                    encrypted=False,
                    checksum="abc",
                )
            ],
        )
        json_str = manifest.to_json()
        data = json.loads(json_str)
        assert data["name"] == "MyGame"
        assert len(data["files"]) == 1

    def test_from_json(self):
        """Test deserializing from JSON."""
        json_str = json.dumps({
            "name": "TestGame",
            "version": "2.0",
            "package_type": "DLC",
            "platform": "linux",
            "files": [],
            "compression": "ZLIB",
            "encryption": "NONE",
        })
        manifest = PackageManifest.from_json(json_str)
        assert manifest.name == "TestGame"
        assert manifest.package_type == PackageType.DLC

    def test_get_total_size(self):
        """Test calculating total size."""
        manifest = PackageManifest(
            name="Test",
            version="1.0",
            package_type=PackageType.FULL,
            platform="windows",
            files=[
                FileEntry("a.txt", 0, 100, 50, CompressionMethod.ZLIB, False, ""),
                FileEntry("b.txt", 50, 200, 100, CompressionMethod.ZLIB, False, ""),
            ],
        )
        assert manifest.get_total_size() == 300
        assert manifest.get_compressed_size() == 150


class TestCompressors:
    """Tests for compression classes."""

    def test_null_compressor(self):
        """Test null compressor."""
        compressor = NullCompressor()
        data = b"test data"
        compressed = compressor.compress(data)
        assert compressed == data
        decompressed = compressor.decompress(compressed)
        assert decompressed == data

    def test_zlib_compressor(self):
        """Test zlib compressor."""
        compressor = ZlibCompressor(level=6)
        data = b"test data " * 100  # Repetitive data compresses well
        compressed = compressor.compress(data)
        assert len(compressed) < len(data)
        decompressed = compressor.decompress(compressed)
        assert decompressed == data


class TestPackageEncryption:
    """Tests for PackageEncryption."""

    def test_no_encryption(self):
        """Test with no encryption."""
        encryption = PackageEncryption(EncryptionMethod.NONE)
        data = b"secret data"
        encrypted = encryption.encrypt(data)
        assert encrypted == data

    def test_encryption_with_key(self):
        """Test encryption with key."""
        key = b"test_key_1234567"
        encryption = PackageEncryption(EncryptionMethod.AES_128, key)
        data = b"secret data"
        encrypted = encryption.encrypt(data)
        assert encrypted != data
        decrypted = encryption.decrypt(encrypted)
        assert decrypted == data

    def test_generate_key(self):
        """Test key generation."""
        encryption = PackageEncryption(EncryptionMethod.AES_256)
        key = encryption.generate_key()
        assert len(key) == 32  # 256 bits


class TestPackageBuilder:
    """Tests for PackageBuilder."""

    @pytest.fixture
    def temp_source(self):
        """Create temporary source directory."""
        temp_dir = tempfile.mkdtemp()
        # Create test files
        os.makedirs(os.path.join(temp_dir, "data"))
        with open(os.path.join(temp_dir, "data", "file1.txt"), "w") as f:
            f.write("File 1 content " * 100)
        with open(os.path.join(temp_dir, "data", "file2.txt"), "w") as f:
            f.write("File 2 content " * 100)
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_builder_creation(self):
        """Test creating package builder."""
        builder = PackageBuilder(
            compression=CompressionMethod.ZLIB,
            encryption=EncryptionMethod.NONE,
        )
        assert builder.compression == CompressionMethod.ZLIB

    def test_build_package(self, temp_source):
        """Test building a package."""
        output_dir = tempfile.mkdtemp()
        try:
            builder = PackageBuilder()
            manifest = PackageManifest(
                name="TestPackage",
                version="1.0",
                package_type=PackageType.FULL,
                platform="windows",
            )

            output_path = os.path.join(output_dir, "test.pak")
            result = builder.build(
                temp_source,
                output_path,
                manifest,
            )

            assert result is True
            assert os.path.exists(output_path)
            assert os.path.exists(output_path + ".manifest")
        finally:
            shutil.rmtree(output_dir)

    def test_build_with_compression(self, temp_source):
        """Test building with compression."""
        output_dir = tempfile.mkdtemp()
        try:
            builder = PackageBuilder(compression=CompressionMethod.ZLIB)
            manifest = PackageManifest(
                name="Compressed",
                version="1.0",
                package_type=PackageType.FULL,
                platform="windows",
            )

            output_path = os.path.join(output_dir, "compressed.pak")
            builder.build(temp_source, output_path, manifest)

            # Verify manifest has compressed sizes
            with open(output_path + ".manifest", "r") as f:
                saved_manifest = PackageManifest.from_json(f.read())

            for entry in saved_manifest.files:
                # Compressed size should be less than original for text files
                assert entry.compressed_size <= entry.size
        finally:
            shutil.rmtree(output_dir)

    def test_build_with_encryption(self, temp_source):
        """Test building with encryption."""
        output_dir = tempfile.mkdtemp()
        try:
            key = b"0123456789abcdef"
            builder = PackageBuilder(
                encryption=EncryptionMethod.AES_128,
                encryption_key=key,
            )
            manifest = PackageManifest(
                name="Encrypted",
                version="1.0",
                package_type=PackageType.FULL,
                platform="windows",
            )

            output_path = os.path.join(output_dir, "encrypted.pak")
            builder.build(temp_source, output_path, manifest)

            # Verify files are marked as encrypted
            with open(output_path + ".manifest", "r") as f:
                saved_manifest = PackageManifest.from_json(f.read())

            for entry in saved_manifest.files:
                assert entry.encrypted is True
        finally:
            shutil.rmtree(output_dir)

    def test_progress_callback(self, temp_source):
        """Test progress callback during build."""
        output_dir = tempfile.mkdtemp()
        try:
            builder = PackageBuilder()
            manifest = PackageManifest(
                name="Test",
                version="1.0",
                package_type=PackageType.FULL,
                platform="windows",
            )

            progress = []

            def callback(current, total):
                progress.append((current, total))

            output_path = os.path.join(output_dir, "test.pak")
            builder.build(temp_source, output_path, manifest, progress_callback=callback)

            assert len(progress) > 0
        finally:
            shutil.rmtree(output_dir)


class TestDLCManager:
    """Tests for DLCManager."""

    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_manager_creation(self, temp_base_dir):
        """Test creating DLC manager."""
        manager = DLCManager(temp_base_dir)
        assert manager.base_dir == temp_base_dir

    def test_register_dlc(self, temp_base_dir):
        """Test registering DLC."""
        manager = DLCManager(temp_base_dir)
        dlc = DLCInfo(id="dlc_1", name="DLC 1", version="1.0")
        manager.register(dlc)

        retrieved = manager.get_dlc("dlc_1")
        assert retrieved is not None
        assert retrieved.name == "DLC 1"

    def test_unregister_dlc(self, temp_base_dir):
        """Test unregistering DLC."""
        manager = DLCManager(temp_base_dir)
        dlc = DLCInfo(id="dlc_1", name="DLC 1", version="1.0")
        manager.register(dlc)

        result = manager.unregister("dlc_1")
        assert result is True
        assert manager.get_dlc("dlc_1") is None

    def test_get_all_dlc(self, temp_base_dir):
        """Test getting all DLC."""
        manager = DLCManager(temp_base_dir)
        manager.register(DLCInfo(id="dlc_1", name="DLC 1", version="1.0"))
        manager.register(DLCInfo(id="dlc_2", name="DLC 2", version="1.0"))

        all_dlc = manager.get_all_dlc()
        assert len(all_dlc) == 2

    def test_is_installed(self, temp_base_dir):
        """Test checking installation status."""
        manager = DLCManager(temp_base_dir)
        dlc = DLCInfo(id="dlc_1", name="DLC 1", version="1.0")
        manager.register(dlc)

        assert manager.is_installed("dlc_1") is False

    def test_save_and_load_registry(self, temp_base_dir):
        """Test saving and loading registry."""
        manager = DLCManager(temp_base_dir)
        manager.register(DLCInfo(id="dlc_1", name="DLC 1", version="1.0"))

        registry_path = os.path.join(temp_base_dir, "dlc_registry.json")
        manager.save_registry(registry_path)

        # Create new manager and load
        manager2 = DLCManager(temp_base_dir)
        manager2.load_registry(registry_path)

        assert manager2.get_dlc("dlc_1") is not None
