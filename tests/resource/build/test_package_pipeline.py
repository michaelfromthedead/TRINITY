"""Tests for the package pipeline."""
import zlib

from engine.resource.build.package_pipeline import (
    PackageBuilder,
    PackageEntry,
    PackageFormat,
    PackageManifest,
    PackageReader,
)


class TestPackageManifest:
    def test_empty_manifest(self) -> None:
        m = PackageManifest()
        assert m.asset_count == 0
        assert m.total_size == 0

    def test_add_and_get_entry(self) -> None:
        m = PackageManifest()
        entry = PackageEntry("tex.png", offset=0, size=100, compressed_size=80, checksum=123)
        m.add_entry(entry)
        assert m.asset_count == 1
        assert m.total_size == 100
        assert m.get_entry("tex.png") is entry
        assert m.get_entry("missing") is None


class TestPackageBuilder:
    def test_build_manifest(self) -> None:
        b = PackageBuilder()
        b.add_asset("a.bin", b"hello")
        b.add_asset("b.bin", b"world!")
        manifest = b.build(PackageFormat.PAK, "out.pak")
        assert manifest.asset_count == 2
        assert manifest.total_size == 11

    def test_entry_offsets(self) -> None:
        b = PackageBuilder()
        b.add_asset("first", b"AAA")
        b.add_asset("second", b"BB")
        manifest = b.build(PackageFormat.PAK, "out.pak")
        entries = manifest.entries
        assert entries["first"].offset == 0
        assert entries["second"].offset == 3

    def test_checksums(self) -> None:
        b = PackageBuilder()
        data = b"test_data"
        b.add_asset("file", data)
        manifest = b.build(PackageFormat.ZIP, "out.zip")
        expected = zlib.crc32(data) & 0xFFFFFFFF
        assert manifest.get_entry("file").checksum == expected

    def test_formats(self) -> None:
        for name in ("PAK", "ZIP", "DIRECTORY"):
            assert hasattr(PackageFormat, name), f"PackageFormat.{name} missing"


class TestPackageReader:
    def test_read_asset(self) -> None:
        b = PackageBuilder()
        b.add_asset("x", b"AAAA")
        b.add_asset("y", b"BB")
        manifest = b.build(PackageFormat.PAK, "p.pak")
        raw_data = b"AAAABB"

        reader = PackageReader()
        reader.open("p.pak", manifest, raw_data)
        assert reader.read_asset(manifest.get_entry("x")) == b"AAAA"
        assert reader.read_asset(manifest.get_entry("y")) == b"BB"

    def test_empty_asset(self) -> None:
        b = PackageBuilder()
        b.add_asset("empty", b"")
        manifest = b.build(PackageFormat.PAK, "p.pak")
        reader = PackageReader()
        reader.open("p.pak", manifest, b"")
        assert reader.read_asset(manifest.get_entry("empty")) == b""
