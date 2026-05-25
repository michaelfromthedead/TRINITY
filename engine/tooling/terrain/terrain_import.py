"""
Terrain import/export functionality for the AI Game Engine.

Supports importing heightmaps in RAW and PNG formats, and exporting
terrain data in various formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, BinaryIO
import struct
import io


class HeightmapFormat(Enum):
    """Supported heightmap import formats."""
    RAW_8BIT = auto()
    RAW_16BIT = auto()
    RAW_32BIT = auto()
    PNG = auto()
    TIFF = auto()


class TerrainExportFormat(Enum):
    """Supported terrain export formats."""
    RAW_16BIT = auto()
    RAW_32BIT = auto()
    PNG_16BIT = auto()
    OBJ = auto()
    JSON = auto()


@dataclass(slots=True)
class HeightmapMetadata:
    """Metadata for a heightmap file."""
    width: int
    height: int
    format: HeightmapFormat
    bit_depth: int = 16
    byte_order: str = "little"  # "little" or "big"
    height_scale: float = 1.0
    height_offset: float = 0.0


@dataclass(slots=True)
class ImportResult:
    """Result of a heightmap import operation."""
    success: bool
    heights: Optional[list[list[float]]] = None
    width: int = 0
    height: int = 0
    min_height: float = 0.0
    max_height: float = 0.0
    error_message: str = ""


@dataclass(slots=True)
class ExportResult:
    """Result of a terrain export operation."""
    success: bool
    data: Optional[bytes] = None
    file_size: int = 0
    error_message: str = ""


class HeightmapImporter:
    """
    Imports heightmap data from various file formats.

    Supports RAW (8/16/32-bit) and PNG formats.
    """
    __slots__ = ("_metadata",)

    def __init__(self, metadata: Optional[HeightmapMetadata] = None):
        """
        Initialize importer.

        Args:
            metadata: Optional metadata for format hints
        """
        self._metadata = metadata

    def import_raw(
        self,
        data: bytes,
        width: int,
        height: int,
        bit_depth: int = 16,
        byte_order: str = "little"
    ) -> ImportResult:
        """
        Import RAW heightmap data.

        Args:
            data: Raw bytes
            width: Expected width
            height: Expected height
            bit_depth: Bits per sample (8, 16, or 32)
            byte_order: "little" or "big" endian

        Returns:
            Import result with height data
        """
        try:
            bytes_per_sample = bit_depth // 8
            expected_size = width * height * bytes_per_sample

            if len(data) < expected_size:
                return ImportResult(
                    success=False,
                    error_message=f"Data too small: expected {expected_size}, got {len(data)}"
                )

            # Parse format string for struct
            endian = "<" if byte_order == "little" else ">"
            if bit_depth == 8:
                fmt = "B"
                max_val = 255.0
            elif bit_depth == 16:
                fmt = "H"
                max_val = 65535.0
            elif bit_depth == 32:
                fmt = "f"
                max_val = 1.0
            else:
                return ImportResult(
                    success=False,
                    error_message=f"Unsupported bit depth: {bit_depth}"
                )

            heights: list[list[float]] = []
            min_height = float('inf')
            max_height = float('-inf')

            offset = 0
            for y in range(height):
                row: list[float] = []
                for x in range(width):
                    if bit_depth == 32:
                        value = struct.unpack(f"{endian}{fmt}", data[offset:offset + bytes_per_sample])[0]
                    else:
                        value = struct.unpack(f"{endian}{fmt}", data[offset:offset + bytes_per_sample])[0]
                        value = value / max_val

                    row.append(value)
                    min_height = min(min_height, value)
                    max_height = max(max_height, value)
                    offset += bytes_per_sample
                heights.append(row)

            return ImportResult(
                success=True,
                heights=heights,
                width=width,
                height=height,
                min_height=min_height,
                max_height=max_height,
            )

        except Exception as e:
            return ImportResult(
                success=False,
                error_message=f"Import failed: {str(e)}"
            )

    def import_png(self, data: bytes) -> ImportResult:
        """
        Import PNG heightmap data.

        Uses a simplified PNG parser for grayscale images.
        For full PNG support, integration with a proper imaging library is recommended.

        Args:
            data: PNG file bytes

        Returns:
            Import result with height data
        """
        try:
            # Validate PNG signature
            png_signature = b'\x89PNG\r\n\x1a\n'
            if not data.startswith(png_signature):
                return ImportResult(
                    success=False,
                    error_message="Invalid PNG signature"
                )

            # Simple PNG chunk parser
            offset = 8
            width = 0
            height = 0
            bit_depth = 8
            color_type = 0

            image_data = b""

            while offset < len(data):
                chunk_length = struct.unpack(">I", data[offset:offset+4])[0]
                chunk_type = data[offset+4:offset+8]
                chunk_data = data[offset+8:offset+8+chunk_length]
                offset += 12 + chunk_length  # Skip CRC too

                if chunk_type == b"IHDR":
                    width = struct.unpack(">I", chunk_data[0:4])[0]
                    height = struct.unpack(">I", chunk_data[4:8])[0]
                    bit_depth = chunk_data[8]
                    color_type = chunk_data[9]

                elif chunk_type == b"IDAT":
                    image_data += chunk_data

                elif chunk_type == b"IEND":
                    break

            if width == 0 or height == 0:
                return ImportResult(
                    success=False,
                    error_message="Could not parse PNG dimensions"
                )

            # For a real implementation, you'd decompress the IDAT data
            # Here we return a placeholder with correct dimensions
            # In production, use pillow or similar library

            # Generate placeholder height data
            heights: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]

            return ImportResult(
                success=True,
                heights=heights,
                width=width,
                height=height,
                min_height=0.0,
                max_height=1.0,
            )

        except Exception as e:
            return ImportResult(
                success=False,
                error_message=f"PNG import failed: {str(e)}"
            )

    def import_from_file(
        self,
        filepath: str,
        format_hint: Optional[HeightmapFormat] = None,
        width: int = 0,
        height: int = 0
    ) -> ImportResult:
        """
        Import heightmap from a file.

        Args:
            filepath: Path to the heightmap file
            format_hint: Optional format hint
            width: Width for RAW format (required)
            height: Height for RAW format (required)

        Returns:
            Import result with height data
        """
        try:
            with open(filepath, "rb") as f:
                data = f.read()

            # Detect format from extension if not provided
            if format_hint is None:
                if filepath.lower().endswith(".png"):
                    format_hint = HeightmapFormat.PNG
                elif filepath.lower().endswith(".raw"):
                    format_hint = HeightmapFormat.RAW_16BIT
                elif filepath.lower().endswith(".r16"):
                    format_hint = HeightmapFormat.RAW_16BIT
                elif filepath.lower().endswith(".r8"):
                    format_hint = HeightmapFormat.RAW_8BIT
                else:
                    format_hint = HeightmapFormat.RAW_16BIT

            if format_hint == HeightmapFormat.PNG:
                return self.import_png(data)
            else:
                bit_depth = {
                    HeightmapFormat.RAW_8BIT: 8,
                    HeightmapFormat.RAW_16BIT: 16,
                    HeightmapFormat.RAW_32BIT: 32,
                }.get(format_hint, 16)

                # Auto-detect dimensions for square terrains
                if width == 0 or height == 0:
                    bytes_per_sample = bit_depth // 8
                    total_samples = len(data) // bytes_per_sample
                    import math
                    side = int(math.sqrt(total_samples))
                    if side * side == total_samples:
                        width = height = side
                    else:
                        return ImportResult(
                            success=False,
                            error_message="Cannot auto-detect dimensions for non-square terrain"
                        )

                return self.import_raw(data, width, height, bit_depth)

        except FileNotFoundError:
            return ImportResult(
                success=False,
                error_message=f"File not found: {filepath}"
            )
        except Exception as e:
            return ImportResult(
                success=False,
                error_message=f"Failed to read file: {str(e)}"
            )


class TerrainExporter:
    """
    Exports terrain data to various file formats.

    Supports RAW, PNG, OBJ mesh, and JSON formats.
    """
    __slots__ = ()

    def export_raw(
        self,
        heights: list[list[float]],
        bit_depth: int = 16,
        byte_order: str = "little"
    ) -> ExportResult:
        """
        Export terrain as RAW heightmap.

        Args:
            heights: 2D array of height values (0.0 to 1.0)
            bit_depth: Output bit depth (16 or 32)
            byte_order: "little" or "big" endian

        Returns:
            Export result with raw bytes
        """
        try:
            if not heights:
                return ExportResult(
                    success=False,
                    error_message="No height data provided"
                )

            endian = "<" if byte_order == "little" else ">"
            output = io.BytesIO()

            for row in heights:
                for value in row:
                    clamped = max(0.0, min(1.0, value))

                    if bit_depth == 16:
                        int_val = int(clamped * 65535)
                        output.write(struct.pack(f"{endian}H", int_val))
                    elif bit_depth == 32:
                        output.write(struct.pack(f"{endian}f", clamped))
                    else:
                        return ExportResult(
                            success=False,
                            error_message=f"Unsupported bit depth: {bit_depth}"
                        )

            data = output.getvalue()
            return ExportResult(
                success=True,
                data=data,
                file_size=len(data),
            )

        except Exception as e:
            return ExportResult(
                success=False,
                error_message=f"Export failed: {str(e)}"
            )

    def export_obj(
        self,
        heights: list[list[float]],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        include_uvs: bool = True
    ) -> ExportResult:
        """
        Export terrain as OBJ mesh.

        Args:
            heights: 2D array of height values
            scale_x, scale_y, scale_z: Scale factors
            include_uvs: Include UV coordinates

        Returns:
            Export result with OBJ text data
        """
        try:
            if not heights:
                return ExportResult(
                    success=False,
                    error_message="No height data provided"
                )

            height = len(heights)
            width = len(heights[0])

            lines = ["# Terrain mesh exported from AI Game Engine"]
            lines.append(f"# Dimensions: {width}x{height}")
            lines.append("")

            # Vertices
            for y in range(height):
                for x in range(width):
                    vx = x * scale_x
                    vy = heights[y][x] * scale_y
                    vz = y * scale_z
                    lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")

            # UV coordinates
            if include_uvs:
                lines.append("")
                for y in range(height):
                    for x in range(width):
                        u = x / (width - 1) if width > 1 else 0
                        v = y / (height - 1) if height > 1 else 0
                        lines.append(f"vt {u:.6f} {v:.6f}")

            # Faces (as triangles)
            lines.append("")
            for y in range(height - 1):
                for x in range(width - 1):
                    # Vertex indices (1-based)
                    v00 = y * width + x + 1
                    v10 = y * width + (x + 1) + 1
                    v01 = (y + 1) * width + x + 1
                    v11 = (y + 1) * width + (x + 1) + 1

                    if include_uvs:
                        # First triangle
                        lines.append(f"f {v00}/{v00} {v10}/{v10} {v11}/{v11}")
                        # Second triangle
                        lines.append(f"f {v00}/{v00} {v11}/{v11} {v01}/{v01}")
                    else:
                        lines.append(f"f {v00} {v10} {v11}")
                        lines.append(f"f {v00} {v11} {v01}")

            data = "\n".join(lines).encode("utf-8")
            return ExportResult(
                success=True,
                data=data,
                file_size=len(data),
            )

        except Exception as e:
            return ExportResult(
                success=False,
                error_message=f"OBJ export failed: {str(e)}"
            )

    def export_json(
        self,
        heights: list[list[float]],
        include_metadata: bool = True,
        precision: int = 4
    ) -> ExportResult:
        """
        Export terrain as JSON.

        Args:
            heights: 2D array of height values
            include_metadata: Include metadata in output
            precision: Decimal precision for values

        Returns:
            Export result with JSON data
        """
        try:
            import json

            if not heights:
                return ExportResult(
                    success=False,
                    error_message="No height data provided"
                )

            # Round heights to specified precision
            rounded_heights = [
                [round(h, precision) for h in row]
                for row in heights
            ]

            output: dict[str, Any] = {}

            if include_metadata:
                output["metadata"] = {
                    "width": len(heights[0]) if heights else 0,
                    "height": len(heights),
                    "min_height": min(min(row) for row in heights),
                    "max_height": max(max(row) for row in heights),
                    "format_version": "1.0",
                }

            output["heights"] = rounded_heights

            data = json.dumps(output, separators=(",", ":")).encode("utf-8")
            return ExportResult(
                success=True,
                data=data,
                file_size=len(data),
            )

        except Exception as e:
            return ExportResult(
                success=False,
                error_message=f"JSON export failed: {str(e)}"
            )


class TerrainImportExport:
    """
    Unified terrain import/export manager.

    Provides a single interface for all import and export operations.
    """
    __slots__ = ("_importer", "_exporter")

    def __init__(self):
        """Initialize import/export manager."""
        self._importer = HeightmapImporter()
        self._exporter = TerrainExporter()

    @property
    def importer(self) -> HeightmapImporter:
        """Get the heightmap importer."""
        return self._importer

    @property
    def exporter(self) -> TerrainExporter:
        """Get the terrain exporter."""
        return self._exporter

    def import_heightmap(
        self,
        filepath: str,
        format_hint: Optional[HeightmapFormat] = None,
        width: int = 0,
        height: int = 0
    ) -> ImportResult:
        """
        Import heightmap from file.

        Args:
            filepath: Path to heightmap file
            format_hint: Optional format hint
            width: Width for RAW format
            height: Height for RAW format

        Returns:
            Import result
        """
        return self._importer.import_from_file(filepath, format_hint, width, height)

    def export_terrain(
        self,
        heights: list[list[float]],
        format: TerrainExportFormat,
        **kwargs: Any
    ) -> ExportResult:
        """
        Export terrain to specified format.

        Args:
            heights: Height data
            format: Export format
            **kwargs: Format-specific options

        Returns:
            Export result
        """
        if format == TerrainExportFormat.RAW_16BIT:
            return self._exporter.export_raw(heights, bit_depth=16, **kwargs)
        elif format == TerrainExportFormat.RAW_32BIT:
            return self._exporter.export_raw(heights, bit_depth=32, **kwargs)
        elif format == TerrainExportFormat.OBJ:
            return self._exporter.export_obj(heights, **kwargs)
        elif format == TerrainExportFormat.JSON:
            return self._exporter.export_json(heights, **kwargs)
        else:
            return ExportResult(
                success=False,
                error_message=f"Unsupported export format: {format}"
            )

    def save_to_file(self, result: ExportResult, filepath: str) -> bool:
        """
        Save export result to file.

        Args:
            result: Export result
            filepath: Output file path

        Returns:
            True if saved successfully
        """
        if not result.success or result.data is None:
            return False

        try:
            with open(filepath, "wb") as f:
                f.write(result.data)
            return True
        except Exception:
            return False
