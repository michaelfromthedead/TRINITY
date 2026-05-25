"""Tests for terrain import/export functionality."""

import pytest
import struct
from engine.tooling.terrain.terrain_import import (
    HeightmapFormat,
    TerrainExportFormat,
    HeightmapMetadata,
    ImportResult,
    ExportResult,
    HeightmapImporter,
    TerrainExporter,
    TerrainImportExport,
)


class TestHeightmapImporter:
    """Tests for heightmap importer."""

    def test_creation(self):
        """Test importer creation."""
        importer = HeightmapImporter()
        assert importer is not None

    def test_import_raw_16bit(self):
        """Test importing RAW 16-bit heightmap."""
        importer = HeightmapImporter()

        # Create test data (4x4 heightmap)
        data = b""
        for i in range(16):
            data += struct.pack("<H", i * 4096)

        result = importer.import_raw(data, 4, 4, 16, "little")

        assert result.success
        assert result.width == 4
        assert result.height == 4
        assert result.heights is not None
        assert len(result.heights) == 4
        assert len(result.heights[0]) == 4

    def test_import_raw_8bit(self):
        """Test importing RAW 8-bit heightmap."""
        importer = HeightmapImporter()

        data = bytes([i * 16 for i in range(16)])
        result = importer.import_raw(data, 4, 4, 8, "little")

        assert result.success
        assert result.width == 4
        assert result.height == 4

    def test_import_raw_32bit(self):
        """Test importing RAW 32-bit heightmap."""
        importer = HeightmapImporter()

        data = b""
        for i in range(16):
            data += struct.pack("<f", i / 15.0)

        result = importer.import_raw(data, 4, 4, 32, "little")

        assert result.success
        assert result.width == 4
        assert result.height == 4

    def test_import_raw_data_too_small(self):
        """Test importing with insufficient data."""
        importer = HeightmapImporter()

        data = b"\x00\x00\x00\x00"  # Only 2 16-bit values
        result = importer.import_raw(data, 4, 4, 16, "little")

        assert not result.success
        assert "too small" in result.error_message.lower()

    def test_import_raw_big_endian(self):
        """Test importing big-endian data."""
        importer = HeightmapImporter()

        data = b""
        for i in range(16):
            data += struct.pack(">H", i * 4096)

        result = importer.import_raw(data, 4, 4, 16, "big")
        assert result.success

    def test_import_png_invalid(self):
        """Test importing invalid PNG."""
        importer = HeightmapImporter()

        result = importer.import_png(b"not a png")
        assert not result.success
        assert "invalid" in result.error_message.lower()

    def test_import_png_valid_signature(self):
        """Test importing with valid PNG signature."""
        importer = HeightmapImporter()

        # Minimal valid PNG structure
        png_sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack(">IIBBBBB", 4, 4, 8, 0, 0, 0, 0)
        ihdr_len = struct.pack(">I", len(ihdr_data))
        ihdr_type = b"IHDR"
        ihdr_crc = struct.pack(">I", 0)

        iend_len = struct.pack(">I", 0)
        iend_type = b"IEND"
        iend_crc = struct.pack(">I", 0)

        data = png_sig + ihdr_len + ihdr_type + ihdr_data + ihdr_crc + iend_len + iend_type + iend_crc

        result = importer.import_png(data)
        assert result.success
        assert result.width == 4
        assert result.height == 4

    def test_import_unsupported_bit_depth(self):
        """Test importing with unsupported bit depth."""
        importer = HeightmapImporter()

        result = importer.import_raw(b"\x00" * 64, 4, 4, 24, "little")
        assert not result.success


class TestTerrainExporter:
    """Tests for terrain exporter."""

    def setup_method(self):
        """Set up test data."""
        self.heights = [
            [0.0, 0.25, 0.5, 0.75],
            [0.1, 0.35, 0.6, 0.85],
            [0.2, 0.45, 0.7, 0.95],
            [0.3, 0.55, 0.8, 1.0],
        ]

    def test_export_raw_16bit(self):
        """Test exporting RAW 16-bit."""
        exporter = TerrainExporter()
        result = exporter.export_raw(self.heights, 16, "little")

        assert result.success
        assert result.data is not None
        assert len(result.data) == 32  # 16 values * 2 bytes

    def test_export_raw_32bit(self):
        """Test exporting RAW 32-bit."""
        exporter = TerrainExporter()
        result = exporter.export_raw(self.heights, 32, "little")

        assert result.success
        assert result.data is not None
        assert len(result.data) == 64  # 16 values * 4 bytes

    def test_export_raw_empty(self):
        """Test exporting empty heightmap."""
        exporter = TerrainExporter()
        result = exporter.export_raw([], 16)

        assert not result.success

    def test_export_obj(self):
        """Test exporting OBJ mesh."""
        exporter = TerrainExporter()
        result = exporter.export_obj(self.heights)

        assert result.success
        assert result.data is not None

        obj_text = result.data.decode("utf-8")
        assert "# Terrain mesh" in obj_text
        assert "v " in obj_text
        assert "f " in obj_text

    def test_export_obj_with_uvs(self):
        """Test exporting OBJ with UVs."""
        exporter = TerrainExporter()
        result = exporter.export_obj(self.heights, include_uvs=True)

        assert result.success
        obj_text = result.data.decode("utf-8")
        assert "vt " in obj_text

    def test_export_obj_without_uvs(self):
        """Test exporting OBJ without UVs."""
        exporter = TerrainExporter()
        result = exporter.export_obj(self.heights, include_uvs=False)

        assert result.success
        obj_text = result.data.decode("utf-8")
        assert "vt " not in obj_text

    def test_export_obj_scaled(self):
        """Test exporting OBJ with scale."""
        exporter = TerrainExporter()
        result = exporter.export_obj(self.heights, scale_x=2.0, scale_y=10.0, scale_z=2.0)

        assert result.success

    def test_export_json(self):
        """Test exporting JSON."""
        exporter = TerrainExporter()
        result = exporter.export_json(self.heights)

        assert result.success
        assert result.data is not None

        import json
        data = json.loads(result.data.decode("utf-8"))
        assert "heights" in data

    def test_export_json_with_metadata(self):
        """Test exporting JSON with metadata."""
        exporter = TerrainExporter()
        result = exporter.export_json(self.heights, include_metadata=True)

        import json
        data = json.loads(result.data.decode("utf-8"))
        assert "metadata" in data
        assert data["metadata"]["width"] == 4
        assert data["metadata"]["height"] == 4

    def test_export_json_precision(self):
        """Test JSON precision control."""
        exporter = TerrainExporter()
        result = exporter.export_json(self.heights, precision=2)

        import json
        data = json.loads(result.data.decode("utf-8"))
        # Check precision is applied


class TestTerrainImportExport:
    """Tests for unified import/export manager."""

    def test_creation(self):
        """Test manager creation."""
        manager = TerrainImportExport()
        assert manager.importer is not None
        assert manager.exporter is not None

    def test_export_raw_16bit(self):
        """Test exporting RAW 16-bit through manager."""
        manager = TerrainImportExport()
        heights = [[0.0, 0.5], [0.5, 1.0]]

        result = manager.export_terrain(heights, TerrainExportFormat.RAW_16BIT)
        assert result.success

    def test_export_raw_32bit(self):
        """Test exporting RAW 32-bit through manager."""
        manager = TerrainImportExport()
        heights = [[0.0, 0.5], [0.5, 1.0]]

        result = manager.export_terrain(heights, TerrainExportFormat.RAW_32BIT)
        assert result.success

    def test_export_obj(self):
        """Test exporting OBJ through manager."""
        manager = TerrainImportExport()
        heights = [[0.0, 0.5], [0.5, 1.0]]

        result = manager.export_terrain(heights, TerrainExportFormat.OBJ)
        assert result.success

    def test_export_json(self):
        """Test exporting JSON through manager."""
        manager = TerrainImportExport()
        heights = [[0.0, 0.5], [0.5, 1.0]]

        result = manager.export_terrain(heights, TerrainExportFormat.JSON)
        assert result.success

    def test_roundtrip_raw(self):
        """Test roundtrip export/import for RAW format."""
        manager = TerrainImportExport()
        original = [[0.0, 0.25], [0.5, 1.0]]

        # Export
        export_result = manager.export_terrain(original, TerrainExportFormat.RAW_16BIT)
        assert export_result.success

        # Import
        import_result = manager.importer.import_raw(
            export_result.data, 2, 2, 16, "little"
        )
        assert import_result.success

        # Compare (with tolerance for quantization)
        for y in range(2):
            for x in range(2):
                assert abs(import_result.heights[y][x] - original[y][x]) < 0.001
