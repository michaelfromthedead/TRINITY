"""Baked Probe Capture System.

Implements offline cubemap capture and BC6H compression for probe-based GI:
- Offline cubemap rendering (6 faces at 90 FOV)
- BC6H compression for HDR cubemaps
- KTX2 storage format
- Mip chain generation
- Pre-filtered roughness levels
- Asset pipeline integration

Reference: RENDERING_CONTEXT.md Section 6.4
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from engine.core.math.geometry import AABB
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec3, Vec4

if TYPE_CHECKING:
    from engine.resource.types.base_asset import BaseAsset


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class BakedProbeConstants:
    """Constants for baked probe capture."""
    # Cubemap face count
    FACE_COUNT: int = 6
    # Field of view for cubemap capture (90 degrees)
    CUBEMAP_FOV_DEGREES: float = 90.0
    CUBEMAP_FOV_RADIANS: float = math.pi / 2.0
    # Default resolution
    DEFAULT_RESOLUTION: int = 256
    # Maximum resolution
    MAX_RESOLUTION: int = 4096
    # BC6H block size
    BC6H_BLOCK_SIZE: int = 4
    # BC6H bytes per block
    BC6H_BYTES_PER_BLOCK: int = 16
    # Default roughness levels
    DEFAULT_ROUGHNESS_LEVELS: int = 8
    # KTX2 magic number
    KTX2_MAGIC: bytes = b'\xABKTX 20\xBB\r\n\x1A\n'
    # Minimum blend distance to avoid division by zero
    MIN_BLEND_DISTANCE: float = 0.001
    # HDR range for BC6H
    HDR_MAX_VALUE: float = 65504.0  # Half-float max
    # Compression quality levels
    QUALITY_FAST: int = 0
    QUALITY_BALANCED: int = 1
    QUALITY_HIGH: int = 2


class CubemapFace(Enum):
    """Cubemap face identifiers."""
    POSITIVE_X = 0  # Right
    NEGATIVE_X = 1  # Left
    POSITIVE_Y = 2  # Up
    NEGATIVE_Y = 3  # Down
    POSITIVE_Z = 4  # Front
    NEGATIVE_Z = 5  # Back


# Face direction vectors and up vectors for view matrix construction
CUBEMAP_FACE_DIRECTIONS: dict[CubemapFace, tuple[Vec3, Vec3]] = {
    CubemapFace.POSITIVE_X: (Vec3(1, 0, 0), Vec3(0, -1, 0)),
    CubemapFace.NEGATIVE_X: (Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    CubemapFace.POSITIVE_Y: (Vec3(0, 1, 0), Vec3(0, 0, 1)),
    CubemapFace.NEGATIVE_Y: (Vec3(0, -1, 0), Vec3(0, 0, -1)),
    CubemapFace.POSITIVE_Z: (Vec3(0, 0, 1), Vec3(0, -1, 0)),
    CubemapFace.NEGATIVE_Z: (Vec3(0, 0, -1), Vec3(0, -1, 0)),
}


class CompressionQuality(Enum):
    """BC6H compression quality levels."""
    FAST = BakedProbeConstants.QUALITY_FAST
    BALANCED = BakedProbeConstants.QUALITY_BALANCED
    HIGH = BakedProbeConstants.QUALITY_HIGH


class FilterMode(Enum):
    """Pre-filter modes for roughness mips."""
    BOX = auto()
    GAUSSIAN = auto()
    IMPORTANCE_SAMPLED = auto()  # GGX importance sampling


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class HDRPixel:
    """High dynamic range pixel value (RGB).

    Attributes:
        r: Red channel (float, can exceed 1.0)
        g: Green channel (float, can exceed 1.0)
        b: Blue channel (float, can exceed 1.0)
    """
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def __add__(self, other: HDRPixel) -> HDRPixel:
        return HDRPixel(self.r + other.r, self.g + other.g, self.b + other.b)

    def __mul__(self, scalar: float) -> HDRPixel:
        return HDRPixel(self.r * scalar, self.g * scalar, self.b * scalar)

    def __rmul__(self, scalar: float) -> HDRPixel:
        return self * scalar

    def clamp_hdr(self) -> HDRPixel:
        """Clamp to valid HDR range."""
        max_val = BakedProbeConstants.HDR_MAX_VALUE
        return HDRPixel(
            max(0.0, min(self.r, max_val)),
            max(0.0, min(self.g, max_val)),
            max(0.0, min(self.b, max_val)),
        )

    def luminance(self) -> float:
        """Compute perceived luminance."""
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    def to_vec3(self) -> Vec3:
        """Convert to Vec3."""
        return Vec3(self.r, self.g, self.b)

    @staticmethod
    def from_vec3(v: Vec3) -> HDRPixel:
        """Create from Vec3."""
        return HDRPixel(v.x, v.y, v.z)


@dataclass
class CubemapFaceData:
    """Raw pixel data for a single cubemap face.

    Attributes:
        face: Which face this data represents
        resolution: Width/height of the face (square)
        pixels: Row-major HDR pixel data
    """
    face: CubemapFace
    resolution: int
    pixels: list[HDRPixel] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected = self.resolution * self.resolution
        if not self.pixels:
            self.pixels = [HDRPixel() for _ in range(expected)]
        elif len(self.pixels) != expected:
            raise ValueError(
                f"Expected {expected} pixels, got {len(self.pixels)}"
            )

    def get_pixel(self, x: int, y: int) -> HDRPixel:
        """Get pixel at (x, y) coordinates."""
        if not (0 <= x < self.resolution and 0 <= y < self.resolution):
            raise IndexError(f"Pixel ({x}, {y}) out of bounds")
        return self.pixels[y * self.resolution + x]

    def set_pixel(self, x: int, y: int, pixel: HDRPixel) -> None:
        """Set pixel at (x, y) coordinates."""
        if not (0 <= x < self.resolution and 0 <= y < self.resolution):
            raise IndexError(f"Pixel ({x}, {y}) out of bounds")
        self.pixels[y * self.resolution + x] = pixel

    def sample_bilinear(self, u: float, v: float) -> HDRPixel:
        """Bilinear sample at normalized coordinates [0, 1]."""
        x = u * (self.resolution - 1)
        y = v * (self.resolution - 1)

        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, self.resolution - 1)
        y1 = min(y0 + 1, self.resolution - 1)

        fx = x - x0
        fy = y - y0

        p00 = self.get_pixel(x0, y0)
        p10 = self.get_pixel(x1, y0)
        p01 = self.get_pixel(x0, y1)
        p11 = self.get_pixel(x1, y1)

        top = p00 * (1 - fx) + p10 * fx
        bottom = p01 * (1 - fx) + p11 * fx
        return top * (1 - fy) + bottom * fy


@dataclass
class CubemapData:
    """Complete cubemap with all 6 faces.

    Attributes:
        resolution: Resolution of each face
        faces: List of 6 face data structures
    """
    resolution: int
    faces: list[CubemapFaceData] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.faces:
            self.faces = [
                CubemapFaceData(face=CubemapFace(i), resolution=self.resolution)
                for i in range(BakedProbeConstants.FACE_COUNT)
            ]
        elif len(self.faces) != BakedProbeConstants.FACE_COUNT:
            raise ValueError(
                f"Cubemap must have exactly 6 faces, got {len(self.faces)}"
            )

    def get_face(self, face: CubemapFace) -> CubemapFaceData:
        """Get data for a specific face."""
        return self.faces[face.value]

    def sample_direction(self, direction: Vec3) -> HDRPixel:
        """Sample the cubemap in a direction."""
        d = direction.normalized()
        abs_x, abs_y, abs_z = abs(d.x), abs(d.y), abs(d.z)

        # Determine which face and compute UV coordinates
        if abs_x >= abs_y and abs_x >= abs_z:
            if d.x > 0:
                face = CubemapFace.POSITIVE_X
                u = (-d.z / abs_x + 1) / 2
                v = (-d.y / abs_x + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_X
                u = (d.z / abs_x + 1) / 2
                v = (-d.y / abs_x + 1) / 2
        elif abs_y >= abs_x and abs_y >= abs_z:
            if d.y > 0:
                face = CubemapFace.POSITIVE_Y
                u = (d.x / abs_y + 1) / 2
                v = (d.z / abs_y + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_Y
                u = (d.x / abs_y + 1) / 2
                v = (-d.z / abs_y + 1) / 2
        else:
            if d.z > 0:
                face = CubemapFace.POSITIVE_Z
                u = (d.x / abs_z + 1) / 2
                v = (-d.y / abs_z + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_Z
                u = (-d.x / abs_z + 1) / 2
                v = (-d.y / abs_z + 1) / 2

        return self.get_face(face).sample_bilinear(u, v)


@dataclass
class MipLevel:
    """Single mip level of a cubemap.

    Attributes:
        level: Mip level index (0 = full resolution)
        resolution: Resolution at this mip level
        cubemap: Cubemap data at this resolution
        roughness: Associated roughness value (for pre-filtered)
    """
    level: int
    resolution: int
    cubemap: CubemapData
    roughness: float = 0.0


@dataclass
class CubemapMipChain:
    """Complete mip chain for a cubemap.

    Attributes:
        base_resolution: Resolution of level 0
        mip_count: Number of mip levels
        mips: List of mip level data
        is_prefiltered: Whether mips represent roughness levels
    """
    base_resolution: int
    mip_count: int
    mips: list[MipLevel] = field(default_factory=list)
    is_prefiltered: bool = False

    @property
    def max_mip_levels(self) -> int:
        """Maximum possible mip levels for base resolution."""
        return int(math.log2(self.base_resolution)) + 1

    def get_mip(self, level: int) -> Optional[MipLevel]:
        """Get a specific mip level."""
        if 0 <= level < len(self.mips):
            return self.mips[level]
        return None

    def sample_roughness(self, direction: Vec3, roughness: float) -> HDRPixel:
        """Sample with roughness-based mip selection."""
        if not self.mips:
            return HDRPixel()

        if not self.is_prefiltered:
            # Use standard mip selection
            mip_level = roughness * (len(self.mips) - 1)
        else:
            # Pre-filtered: roughness directly maps to mip level
            mip_level = roughness * (len(self.mips) - 1)

        # Trilinear filtering between mip levels
        lower = int(mip_level)
        upper = min(lower + 1, len(self.mips) - 1)
        frac = mip_level - lower

        lower_sample = self.mips[lower].cubemap.sample_direction(direction)
        if lower == upper:
            return lower_sample

        upper_sample = self.mips[upper].cubemap.sample_direction(direction)
        return lower_sample * (1 - frac) + upper_sample * frac


# -----------------------------------------------------------------------------
# BC6H Compression
# -----------------------------------------------------------------------------

@dataclass
class BC6HBlock:
    """A single 4x4 BC6H compressed block.

    BC6H stores HDR data in 128 bits (16 bytes).

    Attributes:
        data: 16 bytes of compressed block data
    """
    data: bytes = field(default_factory=lambda: bytes(16))

    def __post_init__(self) -> None:
        if len(self.data) != BakedProbeConstants.BC6H_BYTES_PER_BLOCK:
            raise ValueError(
                f"BC6H block must be 16 bytes, got {len(self.data)}"
            )


class BC6HCompressor:
    """BC6H texture compression for HDR cubemaps.

    BC6H is a block-based HDR compression format that compresses
    4x4 pixel blocks into 128 bits while preserving HDR range.
    """

    def __init__(self, quality: CompressionQuality = CompressionQuality.BALANCED) -> None:
        """Initialize the compressor.

        Args:
            quality: Compression quality level
        """
        self._quality = quality
        self._block_size = BakedProbeConstants.BC6H_BLOCK_SIZE

    @property
    def quality(self) -> CompressionQuality:
        """Get compression quality."""
        return self._quality

    @quality.setter
    def quality(self, value: CompressionQuality) -> None:
        """Set compression quality."""
        self._quality = value

    def compress_face(self, face_data: CubemapFaceData) -> bytes:
        """Compress a single cubemap face.

        Args:
            face_data: HDR pixel data for the face

        Returns:
            Compressed BC6H data
        """
        res = face_data.resolution

        # Handle sub-block resolutions by padding
        if res < self._block_size:
            padded_face = self._pad_to_block_size(face_data)
            return self._compress_padded_face(padded_face, res)

        if res % self._block_size != 0:
            raise ValueError(
                f"Resolution must be multiple of {self._block_size}, got {res}"
            )

        blocks_per_row = res // self._block_size
        total_blocks = blocks_per_row * blocks_per_row
        compressed = bytearray(total_blocks * BakedProbeConstants.BC6H_BYTES_PER_BLOCK)

        block_idx = 0
        for by in range(0, res, self._block_size):
            for bx in range(0, res, self._block_size):
                # Extract 4x4 block
                block_pixels = []
                for py in range(self._block_size):
                    for px in range(self._block_size):
                        block_pixels.append(face_data.get_pixel(bx + px, by + py))

                # Compress block
                block_data = self._compress_block(block_pixels)

                # Store compressed block
                offset = block_idx * BakedProbeConstants.BC6H_BYTES_PER_BLOCK
                compressed[offset:offset + BakedProbeConstants.BC6H_BYTES_PER_BLOCK] = block_data
                block_idx += 1

        return bytes(compressed)

    def compress_cubemap(self, cubemap: CubemapData) -> list[bytes]:
        """Compress all faces of a cubemap.

        Args:
            cubemap: Complete cubemap data

        Returns:
            List of 6 compressed face data
        """
        return [self.compress_face(face) for face in cubemap.faces]

    def decompress_face(self, data: bytes, resolution: int, face: CubemapFace) -> CubemapFaceData:
        """Decompress BC6H data to a cubemap face.

        Args:
            data: Compressed BC6H data
            resolution: Face resolution
            face: Which face this represents

        Returns:
            Decompressed face data
        """
        # Handle sub-block resolutions
        if resolution < self._block_size:
            return self._decompress_sub_block_face(data, resolution, face)

        blocks_per_row = resolution // self._block_size
        total_blocks = blocks_per_row * blocks_per_row
        expected_size = total_blocks * BakedProbeConstants.BC6H_BYTES_PER_BLOCK

        if len(data) != expected_size:
            raise ValueError(
                f"Expected {expected_size} bytes, got {len(data)}"
            )

        face_data = CubemapFaceData(face=face, resolution=resolution)

        block_idx = 0
        for by in range(0, resolution, self._block_size):
            for bx in range(0, resolution, self._block_size):
                offset = block_idx * BakedProbeConstants.BC6H_BYTES_PER_BLOCK
                block_bytes = data[offset:offset + BakedProbeConstants.BC6H_BYTES_PER_BLOCK]

                # Decompress block
                block_pixels = self._decompress_block(block_bytes)

                # Store pixels
                for py in range(self._block_size):
                    for px in range(self._block_size):
                        face_data.set_pixel(
                            bx + px, by + py,
                            block_pixels[py * self._block_size + px]
                        )
                block_idx += 1

        return face_data

    def _decompress_sub_block_face(
        self,
        data: bytes,
        resolution: int,
        face: CubemapFace,
    ) -> CubemapFaceData:
        """Decompress a sub-block resolution face.

        Args:
            data: Compressed single block
            resolution: Original resolution (< block_size)
            face: Which face this represents

        Returns:
            Decompressed face at original resolution
        """
        if len(data) != BakedProbeConstants.BC6H_BYTES_PER_BLOCK:
            raise ValueError(
                f"Expected {BakedProbeConstants.BC6H_BYTES_PER_BLOCK} bytes, got {len(data)}"
            )

        block_pixels = self._decompress_block(data)
        face_data = CubemapFaceData(face=face, resolution=resolution)

        for y in range(resolution):
            for x in range(resolution):
                face_data.set_pixel(x, y, block_pixels[y * self._block_size + x])

        return face_data

    def _compress_block(self, pixels: list[HDRPixel]) -> bytes:
        """Compress a 4x4 block to BC6H format.

        This is a simplified BC6H encoder. In production, use a dedicated
        library like bc6h-enc for optimal quality.

        Args:
            pixels: 16 HDR pixels in row-major order

        Returns:
            16 bytes of compressed data
        """
        if len(pixels) != 16:
            raise ValueError(f"Expected 16 pixels, got {len(pixels)}")

        # Compute block bounds
        min_r = min_g = min_b = float('inf')
        max_r = max_g = max_b = float('-inf')

        for p in pixels:
            clamped = p.clamp_hdr()
            min_r = min(min_r, clamped.r)
            min_g = min(min_g, clamped.g)
            min_b = min(min_b, clamped.b)
            max_r = max(max_r, clamped.r)
            max_g = max(max_g, clamped.g)
            max_b = max(max_b, clamped.b)

        # Convert to half-float representation
        def float_to_half_bytes(f: float) -> int:
            """Convert float to 16-bit half representation."""
            f = max(0.0, min(f, BakedProbeConstants.HDR_MAX_VALUE))
            if f == 0.0:
                return 0
            # Simplified half-float encoding
            sign = 0
            exp = int(math.log2(max(f, 1e-10))) + 15
            exp = max(1, min(exp, 30))
            mantissa = int((f / (2 ** (exp - 15))) * 1024) & 0x3FF
            return (sign << 15) | (exp << 10) | mantissa

        # Encode endpoints as half floats
        ep0_r = float_to_half_bytes(min_r)
        ep0_g = float_to_half_bytes(min_g)
        ep0_b = float_to_half_bytes(min_b)
        ep1_r = float_to_half_bytes(max_r)
        ep1_g = float_to_half_bytes(max_g)
        ep1_b = float_to_half_bytes(max_b)

        # Compute indices for each pixel (4 bits each = 64 bits total for 16 pixels)
        indices = 0
        range_r = max(max_r - min_r, 1e-6)
        range_g = max(max_g - min_g, 1e-6)
        range_b = max(max_b - min_b, 1e-6)

        for i, p in enumerate(pixels):
            clamped = p.clamp_hdr()
            # Compute interpolation factor
            t_r = (clamped.r - min_r) / range_r if range_r > 0 else 0
            t_g = (clamped.g - min_g) / range_g if range_g > 0 else 0
            t_b = (clamped.b - min_b) / range_b if range_b > 0 else 0
            t = (t_r + t_g + t_b) / 3.0
            idx = int(t * 15 + 0.5)
            idx = max(0, min(15, idx))
            indices |= (idx << (i * 4))

        # Pack into 16 bytes
        # Mode 0 format (simplified):
        # Bytes 0-1: mode + endpoint 0 R
        # Bytes 2-3: endpoint 0 G
        # Bytes 4-5: endpoint 0 B
        # Bytes 6-7: endpoint 1 R
        # Bytes 8-9: endpoint 1 G
        # Bytes 10-11: endpoint 1 B
        # Bytes 12-15: indices
        result = bytearray(16)
        struct.pack_into('<H', result, 0, ep0_r)
        struct.pack_into('<H', result, 2, ep0_g)
        struct.pack_into('<H', result, 4, ep0_b)
        struct.pack_into('<H', result, 6, ep1_r)
        struct.pack_into('<H', result, 8, ep1_g)
        struct.pack_into('<H', result, 10, ep1_b)
        struct.pack_into('<I', result, 12, indices & 0xFFFFFFFF)

        return bytes(result)

    def _decompress_block(self, data: bytes) -> list[HDRPixel]:
        """Decompress a BC6H block to 16 HDR pixels.

        Args:
            data: 16 bytes of compressed data

        Returns:
            16 HDR pixels
        """
        if len(data) != 16:
            raise ValueError(f"Expected 16 bytes, got {len(data)}")

        def half_bytes_to_float(h: int) -> float:
            """Convert 16-bit half to float."""
            if h == 0:
                return 0.0
            exp = (h >> 10) & 0x1F
            mantissa = h & 0x3FF
            return (1.0 + mantissa / 1024.0) * (2 ** (exp - 15))

        # Unpack endpoints
        ep0_r = half_bytes_to_float(struct.unpack_from('<H', data, 0)[0])
        ep0_g = half_bytes_to_float(struct.unpack_from('<H', data, 2)[0])
        ep0_b = half_bytes_to_float(struct.unpack_from('<H', data, 4)[0])
        ep1_r = half_bytes_to_float(struct.unpack_from('<H', data, 6)[0])
        ep1_g = half_bytes_to_float(struct.unpack_from('<H', data, 8)[0])
        ep1_b = half_bytes_to_float(struct.unpack_from('<H', data, 10)[0])

        indices = struct.unpack_from('<I', data, 12)[0]

        # Reconstruct pixels
        pixels = []
        for i in range(16):
            idx = (indices >> (i * 4)) & 0xF
            t = idx / 15.0

            r = ep0_r * (1 - t) + ep1_r * t
            g = ep0_g * (1 - t) + ep1_g * t
            b = ep0_b * (1 - t) + ep1_b * t

            pixels.append(HDRPixel(r, g, b))

        return pixels

    def _pad_to_block_size(self, face_data: CubemapFaceData) -> CubemapFaceData:
        """Pad a sub-block face to block size.

        Args:
            face_data: Face with resolution < block_size

        Returns:
            Padded face at block_size resolution
        """
        padded = CubemapFaceData(face=face_data.face, resolution=self._block_size)
        src_res = face_data.resolution

        for y in range(self._block_size):
            for x in range(self._block_size):
                # Map to source pixel (clamp to edge)
                src_x = min(x, src_res - 1)
                src_y = min(y, src_res - 1)
                padded.set_pixel(x, y, face_data.get_pixel(src_x, src_y))

        return padded

    def _compress_padded_face(self, padded_face: CubemapFaceData, original_res: int) -> bytes:
        """Compress a padded face, storing original resolution info.

        Args:
            padded_face: Face padded to block_size
            original_res: Original resolution before padding

        Returns:
            Compressed data with single block
        """
        # Extract 4x4 block
        block_pixels = []
        for py in range(self._block_size):
            for px in range(self._block_size):
                block_pixels.append(padded_face.get_pixel(px, py))

        return self._compress_block(block_pixels)

    def estimate_compressed_size(self, resolution: int, face_count: int = 6) -> int:
        """Estimate compressed size for a cubemap.

        Args:
            resolution: Face resolution
            face_count: Number of faces (default 6)

        Returns:
            Estimated size in bytes
        """
        # Handle sub-block resolutions
        effective_res = max(resolution, self._block_size)
        blocks_per_face = (effective_res // self._block_size) ** 2
        bytes_per_face = blocks_per_face * BakedProbeConstants.BC6H_BYTES_PER_BLOCK
        return bytes_per_face * face_count


# -----------------------------------------------------------------------------
# KTX2 Format
# -----------------------------------------------------------------------------

class KTX2Format(Enum):
    """Vulkan/GL texture formats."""
    VK_FORMAT_BC6H_UFLOAT_BLOCK = 143
    VK_FORMAT_BC6H_SFLOAT_BLOCK = 144


@dataclass
class KTX2Header:
    """KTX2 file header structure.

    Attributes:
        format: Vulkan format enum
        type_size: Type size for format
        width: Texture width
        height: Texture height
        depth: Texture depth (1 for cubemap)
        layer_count: Array layers (0 for non-array)
        face_count: Cubemap faces (6 for cubemap)
        level_count: Mip levels
        supercompression: Supercompression scheme
    """
    format: KTX2Format = KTX2Format.VK_FORMAT_BC6H_UFLOAT_BLOCK
    type_size: int = 1
    width: int = 256
    height: int = 256
    depth: int = 0
    layer_count: int = 0
    face_count: int = 6
    level_count: int = 1
    supercompression: int = 0  # 0 = none, 1 = BasisLZ, 2 = Zstd, 3 = Zlib


class KTX2Writer:
    """Writer for KTX2 container format.

    KTX2 is the Khronos Texture format version 2, supporting
    compressed and HDR textures with full mip chains.
    """

    def __init__(self) -> None:
        """Initialize KTX2 writer."""
        self._magic = BakedProbeConstants.KTX2_MAGIC

    def write(
        self,
        mip_chain: CubemapMipChain,
        compressed_faces: list[list[bytes]],
        output_path: Path,
        supercompress: bool = True,
    ) -> int:
        """Write a compressed cubemap to KTX2 format.

        Args:
            mip_chain: Mip chain metadata
            compressed_faces: List of [mip][face] compressed data
            output_path: Output file path
            supercompress: Whether to apply supercompression (zlib)

        Returns:
            Written file size in bytes
        """
        header = KTX2Header(
            width=mip_chain.base_resolution,
            height=mip_chain.base_resolution,
            level_count=len(compressed_faces),
            supercompression=3 if supercompress else 0,
        )

        with open(output_path, 'wb') as f:
            # Write magic
            f.write(self._magic)

            # Write header (simplified)
            f.write(struct.pack('<I', header.format.value))  # vkFormat
            f.write(struct.pack('<I', header.type_size))     # typeSize
            f.write(struct.pack('<I', header.width))         # pixelWidth
            f.write(struct.pack('<I', header.height))        # pixelHeight
            f.write(struct.pack('<I', header.depth))         # pixelDepth
            f.write(struct.pack('<I', header.layer_count))   # layerCount
            f.write(struct.pack('<I', header.face_count))    # faceCount
            f.write(struct.pack('<I', header.level_count))   # levelCount
            f.write(struct.pack('<I', header.supercompression))  # supercompression

            # Write level index (offset and size for each mip)
            level_index_offset = f.tell()
            # Reserve space for level index (3 uint64 per level)
            f.write(bytes(24 * header.level_count))

            # Write mip data
            level_offsets = []
            for mip_idx, mip_faces in enumerate(compressed_faces):
                level_start = f.tell()

                # Concatenate all faces for this mip
                mip_data = b''.join(mip_faces)

                if supercompress:
                    mip_data = zlib.compress(mip_data, level=6)

                f.write(mip_data)
                level_offsets.append((level_start, len(mip_data)))

            # Go back and fill in level index
            end_pos = f.tell()
            f.seek(level_index_offset)
            for offset, size in level_offsets:
                f.write(struct.pack('<Q', offset))  # byteOffset
                f.write(struct.pack('<Q', size))    # byteLength
                f.write(struct.pack('<Q', size if not supercompress else 0))  # uncompressedLength

            return end_pos

    def write_to_bytes(
        self,
        mip_chain: CubemapMipChain,
        compressed_faces: list[list[bytes]],
        supercompress: bool = True,
    ) -> bytes:
        """Write KTX2 to bytes buffer.

        Args:
            mip_chain: Mip chain metadata
            compressed_faces: Compressed face data per mip
            supercompress: Apply supercompression

        Returns:
            KTX2 file as bytes
        """
        buffer = io.BytesIO()

        header = KTX2Header(
            width=mip_chain.base_resolution,
            height=mip_chain.base_resolution,
            level_count=len(compressed_faces),
            supercompression=3 if supercompress else 0,
        )

        # Write magic
        buffer.write(self._magic)

        # Write header
        buffer.write(struct.pack('<I', header.format.value))
        buffer.write(struct.pack('<I', header.type_size))
        buffer.write(struct.pack('<I', header.width))
        buffer.write(struct.pack('<I', header.height))
        buffer.write(struct.pack('<I', header.depth))
        buffer.write(struct.pack('<I', header.layer_count))
        buffer.write(struct.pack('<I', header.face_count))
        buffer.write(struct.pack('<I', header.level_count))
        buffer.write(struct.pack('<I', header.supercompression))

        # Reserve space for level index
        level_index_offset = buffer.tell()
        buffer.write(bytes(24 * header.level_count))

        # Write mip data
        level_offsets = []
        for mip_faces in compressed_faces:
            level_start = buffer.tell()
            mip_data = b''.join(mip_faces)
            if supercompress:
                mip_data = zlib.compress(mip_data, level=6)
            buffer.write(mip_data)
            level_offsets.append((level_start, len(mip_data)))

        # Fill in level index
        buffer.seek(level_index_offset)
        for offset, size in level_offsets:
            buffer.write(struct.pack('<Q', offset))
            buffer.write(struct.pack('<Q', size))
            buffer.write(struct.pack('<Q', size if not supercompress else 0))

        return buffer.getvalue()


class KTX2Reader:
    """Reader for KTX2 container format."""

    def __init__(self) -> None:
        """Initialize KTX2 reader."""
        self._magic = BakedProbeConstants.KTX2_MAGIC

    def read(self, path: Path) -> tuple[KTX2Header, list[list[bytes]]]:
        """Read a KTX2 file.

        Args:
            path: Path to KTX2 file

        Returns:
            Tuple of (header, mip data per level per face)
        """
        with open(path, 'rb') as f:
            return self._read_from_stream(f)

    def read_from_bytes(self, data: bytes) -> tuple[KTX2Header, list[list[bytes]]]:
        """Read KTX2 from bytes.

        Args:
            data: KTX2 file bytes

        Returns:
            Tuple of (header, mip data)
        """
        buffer = io.BytesIO(data)
        return self._read_from_stream(buffer)

    def _read_from_stream(self, f: io.IOBase) -> tuple[KTX2Header, list[list[bytes]]]:
        """Read from a stream."""
        # Verify magic
        magic = f.read(12)
        if magic != self._magic:
            raise ValueError("Invalid KTX2 magic number")

        # Read header
        vk_format = struct.unpack('<I', f.read(4))[0]
        type_size = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        depth = struct.unpack('<I', f.read(4))[0]
        layer_count = struct.unpack('<I', f.read(4))[0]
        face_count = struct.unpack('<I', f.read(4))[0]
        level_count = struct.unpack('<I', f.read(4))[0]
        supercompression = struct.unpack('<I', f.read(4))[0]

        header = KTX2Header(
            format=KTX2Format(vk_format),
            type_size=type_size,
            width=width,
            height=height,
            depth=depth,
            layer_count=layer_count,
            face_count=face_count,
            level_count=level_count,
            supercompression=supercompression,
        )

        # Read level index
        level_offsets = []
        for _ in range(level_count):
            offset = struct.unpack('<Q', f.read(8))[0]
            size = struct.unpack('<Q', f.read(8))[0]
            _uncompressed = struct.unpack('<Q', f.read(8))[0]
            level_offsets.append((offset, size))

        # Read mip data
        mip_data = []
        for offset, size in level_offsets:
            f.seek(offset)
            data = f.read(size)

            if supercompression == 3:  # Zlib
                data = zlib.decompress(data)

            # Split into faces
            face_size = len(data) // face_count
            faces = [
                data[i * face_size:(i + 1) * face_size]
                for i in range(face_count)
            ]
            mip_data.append(faces)

        return header, mip_data


# -----------------------------------------------------------------------------
# Cubemap Rendering
# -----------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    """Configuration for cubemap capture.

    Attributes:
        resolution: Resolution of each face
        near_plane: Near clip plane
        far_plane: Far clip plane
        hdr_exposure: HDR exposure multiplier
        include_layers: Layer mask for rendering
        exclude_tags: Tags to exclude from capture
    """
    resolution: int = BakedProbeConstants.DEFAULT_RESOLUTION
    near_plane: float = 0.1
    far_plane: float = 1000.0
    hdr_exposure: float = 1.0
    include_layers: int = 0xFFFFFFFF
    exclude_tags: list[str] = field(default_factory=list)


class CubemapRenderer(ABC):
    """Abstract base for cubemap rendering.

    Subclasses implement actual rendering using the engine's
    render pipeline.
    """

    def __init__(self, config: CaptureConfig) -> None:
        """Initialize renderer.

        Args:
            config: Capture configuration
        """
        self._config = config

    @property
    def config(self) -> CaptureConfig:
        """Get capture config."""
        return self._config

    def capture(self, position: Vec3) -> CubemapData:
        """Capture a complete cubemap at a position.

        Args:
            position: World position to capture from

        Returns:
            Complete cubemap data
        """
        cubemap = CubemapData(resolution=self._config.resolution)

        for face in CubemapFace:
            face_data = self._capture_face(position, face)
            cubemap.faces[face.value] = face_data

        return cubemap

    def get_view_matrix(self, position: Vec3, face: CubemapFace) -> Mat4:
        """Get view matrix for a cubemap face.

        Args:
            position: Camera position
            face: Which face to render

        Returns:
            View matrix for the face
        """
        direction, up = CUBEMAP_FACE_DIRECTIONS[face]
        target = position + direction
        return Mat4.look_at(position, target, up)

    def get_projection_matrix(self) -> Mat4:
        """Get projection matrix for cubemap rendering.

        Returns:
            90-degree FOV projection matrix
        """
        return Mat4.perspective(
            BakedProbeConstants.CUBEMAP_FOV_RADIANS,
            1.0,  # Aspect ratio 1:1 for cubemap faces
            self._config.near_plane,
            self._config.far_plane,
        )

    @abstractmethod
    def _capture_face(self, position: Vec3, face: CubemapFace) -> CubemapFaceData:
        """Capture a single cubemap face.

        Args:
            position: Camera position
            face: Which face to capture

        Returns:
            Captured face data
        """
        ...


class FunctionCubemapRenderer(CubemapRenderer):
    """Cubemap renderer using a sample function.

    Useful for testing or procedural environment capture.
    """

    def __init__(
        self,
        config: CaptureConfig,
        sample_func: Callable[[Vec3, Vec3], Vec3],
    ) -> None:
        """Initialize with a sample function.

        Args:
            config: Capture config
            sample_func: Function(position, direction) -> color
        """
        super().__init__(config)
        self._sample_func = sample_func

    def _capture_face(self, position: Vec3, face: CubemapFace) -> CubemapFaceData:
        """Capture using the sample function."""
        res = self._config.resolution
        face_data = CubemapFaceData(face=face, resolution=res)

        direction, up = CUBEMAP_FACE_DIRECTIONS[face]
        right = direction.cross(up).normalized()

        for y in range(res):
            for x in range(res):
                # Compute direction for this pixel
                u = (x + 0.5) / res * 2.0 - 1.0  # [-1, 1]
                v = (y + 0.5) / res * 2.0 - 1.0  # [-1, 1]

                # Pixel direction in face space
                pixel_dir = (direction + right * u + up * (-v)).normalized()

                # Sample environment
                color = self._sample_func(position, pixel_dir)
                pixel = HDRPixel(color.x, color.y, color.z)
                face_data.set_pixel(x, y, pixel)

        return face_data


# -----------------------------------------------------------------------------
# Mip Chain Generation
# -----------------------------------------------------------------------------

class MipGenerator:
    """Generates mip chains for cubemaps."""

    def __init__(self, filter_mode: FilterMode = FilterMode.BOX) -> None:
        """Initialize mip generator.

        Args:
            filter_mode: Filter mode for downsampling
        """
        self._filter_mode = filter_mode

    @property
    def filter_mode(self) -> FilterMode:
        """Get filter mode."""
        return self._filter_mode

    def generate_mips(
        self,
        base_cubemap: CubemapData,
        mip_count: Optional[int] = None,
    ) -> CubemapMipChain:
        """Generate a mip chain from a base cubemap.

        Args:
            base_cubemap: Full-resolution cubemap
            mip_count: Number of mips (None = all levels)

        Returns:
            Complete mip chain
        """
        res = base_cubemap.resolution
        max_mips = int(math.log2(res)) + 1

        if mip_count is None:
            mip_count = max_mips
        else:
            mip_count = min(mip_count, max_mips)

        mip_chain = CubemapMipChain(
            base_resolution=res,
            mip_count=mip_count,
        )

        # Level 0 is the base
        mip_chain.mips.append(MipLevel(
            level=0,
            resolution=res,
            cubemap=base_cubemap,
            roughness=0.0,
        ))

        # Generate subsequent mips
        current = base_cubemap
        current_res = res

        for level in range(1, mip_count):
            new_res = max(1, current_res // 2)
            new_cubemap = self._downsample_cubemap(current, new_res)

            mip_chain.mips.append(MipLevel(
                level=level,
                resolution=new_res,
                cubemap=new_cubemap,
                roughness=level / (mip_count - 1) if mip_count > 1 else 0.0,
            ))

            current = new_cubemap
            current_res = new_res

        return mip_chain

    def _downsample_cubemap(self, src: CubemapData, new_res: int) -> CubemapData:
        """Downsample a cubemap to a new resolution."""
        result = CubemapData(resolution=new_res)

        for face in CubemapFace:
            src_face = src.get_face(face)
            dst_face = result.get_face(face)

            scale = src_face.resolution / new_res

            for y in range(new_res):
                for x in range(new_res):
                    if self._filter_mode == FilterMode.BOX:
                        pixel = self._box_filter(src_face, x, y, scale)
                    elif self._filter_mode == FilterMode.GAUSSIAN:
                        pixel = self._gaussian_filter(src_face, x, y, scale)
                    else:
                        pixel = self._box_filter(src_face, x, y, scale)

                    dst_face.set_pixel(x, y, pixel)

        return result

    def _box_filter(
        self,
        src: CubemapFaceData,
        x: int,
        y: int,
        scale: float,
    ) -> HDRPixel:
        """Apply box filter for downsampling."""
        src_x = int(x * scale)
        src_y = int(y * scale)
        size = int(scale)

        total = HDRPixel()
        count = 0

        for dy in range(size):
            for dx in range(size):
                px = min(src_x + dx, src.resolution - 1)
                py = min(src_y + dy, src.resolution - 1)
                total = total + src.get_pixel(px, py)
                count += 1

        if count > 0:
            return total * (1.0 / count)
        return total

    def _gaussian_filter(
        self,
        src: CubemapFaceData,
        x: int,
        y: int,
        scale: float,
    ) -> HDRPixel:
        """Apply Gaussian filter for downsampling."""
        # Simplified: use bilinear sampling at center
        u = (x + 0.5) / (src.resolution / scale)
        v = (y + 0.5) / (src.resolution / scale)
        return src.sample_bilinear(u, v)


# -----------------------------------------------------------------------------
# Pre-filtered Environment Maps
# -----------------------------------------------------------------------------

class PrefilteredGenerator:
    """Generates pre-filtered environment maps for roughness-based sampling.

    Pre-filtering convolves the environment map with GGX BRDF for
    different roughness levels, enabling efficient IBL.
    """

    def __init__(
        self,
        sample_count: int = 1024,
        roughness_levels: int = BakedProbeConstants.DEFAULT_ROUGHNESS_LEVELS,
    ) -> None:
        """Initialize pre-filter generator.

        Args:
            sample_count: Samples per pixel for integration
            roughness_levels: Number of roughness mip levels
        """
        self._sample_count = sample_count
        self._roughness_levels = roughness_levels

    @property
    def sample_count(self) -> int:
        """Get sample count."""
        return self._sample_count

    @property
    def roughness_levels(self) -> int:
        """Get roughness level count."""
        return self._roughness_levels

    def generate_prefiltered(self, source: CubemapData) -> CubemapMipChain:
        """Generate pre-filtered mip chain.

        Args:
            source: Source environment cubemap

        Returns:
            Pre-filtered mip chain
        """
        base_res = source.resolution
        mip_chain = CubemapMipChain(
            base_resolution=base_res,
            mip_count=self._roughness_levels,
            is_prefiltered=True,
        )

        for level in range(self._roughness_levels):
            roughness = level / max(1, self._roughness_levels - 1)
            mip_res = max(1, base_res >> level)

            mip_cubemap = self._filter_for_roughness(source, mip_res, roughness)

            mip_chain.mips.append(MipLevel(
                level=level,
                resolution=mip_res,
                cubemap=mip_cubemap,
                roughness=roughness,
            ))

        return mip_chain

    def _filter_for_roughness(
        self,
        source: CubemapData,
        resolution: int,
        roughness: float,
    ) -> CubemapData:
        """Filter cubemap for a specific roughness level."""
        result = CubemapData(resolution=resolution)

        for face in CubemapFace:
            src_face = source.get_face(face)
            dst_face = result.get_face(face)

            direction_base, up = CUBEMAP_FACE_DIRECTIONS[face]
            right = direction_base.cross(up).normalized()

            for y in range(resolution):
                for x in range(resolution):
                    # Compute reflection direction
                    u = (x + 0.5) / resolution * 2.0 - 1.0
                    v = (y + 0.5) / resolution * 2.0 - 1.0
                    N = (direction_base + right * u + up * (-v)).normalized()

                    # Convolve with GGX
                    pixel = self._convolve_ggx(source, N, roughness)
                    dst_face.set_pixel(x, y, pixel)

        return result

    def _convolve_ggx(
        self,
        source: CubemapData,
        N: Vec3,
        roughness: float,
    ) -> HDRPixel:
        """Convolve environment with GGX distribution.

        Args:
            source: Source cubemap
            N: Normal/reflection direction
            roughness: Surface roughness

        Returns:
            Convolved color
        """
        # For roughness = 0, just sample directly
        if roughness < 0.001:
            return source.sample_direction(N)

        total = HDRPixel()
        total_weight = 0.0

        # Use reduced samples for higher roughness
        samples = max(32, self._sample_count // (1 + int(roughness * 4)))

        for i in range(samples):
            # Quasi-random sampling using Hammersley sequence
            xi_x = self._radical_inverse_vdc(i)
            xi_y = i / samples

            # Importance sample GGX
            H = self._importance_sample_ggx(xi_x, xi_y, N, roughness)
            L = N * 2.0 * N.dot(H) - H  # Reflect

            NdotL = max(0.0, N.dot(L))

            if NdotL > 0:
                sample = source.sample_direction(L)
                total = total + sample * NdotL
                total_weight += NdotL

        if total_weight > 0:
            return total * (1.0 / total_weight)
        return source.sample_direction(N)

    def _radical_inverse_vdc(self, bits: int) -> float:
        """Van der Corput sequence for quasi-random sampling."""
        bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
        bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
        bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
        bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
        bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
        return bits * 2.3283064365386963e-10

    def _importance_sample_ggx(
        self,
        xi_x: float,
        xi_y: float,
        N: Vec3,
        roughness: float,
    ) -> Vec3:
        """Importance sample the GGX distribution.

        Args:
            xi_x: Random value [0, 1]
            xi_y: Random value [0, 1]
            N: Surface normal
            roughness: Surface roughness

        Returns:
            Sampled half-vector
        """
        a = roughness * roughness

        phi = 2.0 * math.pi * xi_x
        cos_theta = math.sqrt((1.0 - xi_y) / (1.0 + (a * a - 1.0) * xi_y))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        # Spherical to Cartesian (tangent space)
        H_tangent = Vec3(
            math.cos(phi) * sin_theta,
            math.sin(phi) * sin_theta,
            cos_theta,
        )

        # Tangent to world
        up = Vec3(0, 1, 0) if abs(N.y) < 0.999 else Vec3(1, 0, 0)
        tangent = up.cross(N).normalized()
        bitangent = N.cross(tangent)

        return (
            tangent * H_tangent.x +
            bitangent * H_tangent.y +
            N * H_tangent.z
        ).normalized()


# -----------------------------------------------------------------------------
# Baked Probe Capture
# -----------------------------------------------------------------------------

@dataclass
class BakedProbeConfig:
    """Configuration for baked probe capture.

    Attributes:
        resolution: Cubemap resolution per face
        roughness_levels: Number of pre-filtered mip levels
        compression_quality: BC6H compression quality
        near_plane: Near clip plane
        far_plane: Far clip plane
        sample_count: Samples for pre-filtering
        supercompress: Whether to apply KTX2 supercompression
    """
    resolution: int = BakedProbeConstants.DEFAULT_RESOLUTION
    roughness_levels: int = BakedProbeConstants.DEFAULT_ROUGHNESS_LEVELS
    compression_quality: CompressionQuality = CompressionQuality.BALANCED
    near_plane: float = 0.1
    far_plane: float = 1000.0
    sample_count: int = 512
    supercompress: bool = True


@dataclass
class BakedProbeAsset:
    """A baked environment probe asset.

    Attributes:
        probe_id: Unique identifier
        name: Human-readable name
        position: World position
        bounds: Influence bounds
        resolution: Base resolution
        mip_count: Number of mip levels
        is_prefiltered: Whether mips are pre-filtered
        ktx2_data: Compressed KTX2 data
        loaded: Whether cubemap data is loaded
    """
    probe_id: int
    name: str
    position: Vec3
    bounds: AABB
    resolution: int
    mip_count: int
    is_prefiltered: bool
    ktx2_data: bytes = field(default_factory=bytes, repr=False)
    loaded: bool = False

    # Runtime data (not serialized)
    _mip_chain: Optional[CubemapMipChain] = field(default=None, repr=False)

    def load(self) -> None:
        """Load and decompress the probe data."""
        if self.loaded:
            return

        reader = KTX2Reader()
        header, mip_data = reader.read_from_bytes(self.ktx2_data)

        compressor = BC6HCompressor()
        self._mip_chain = CubemapMipChain(
            base_resolution=self.resolution,
            mip_count=self.mip_count,
            is_prefiltered=self.is_prefiltered,
        )

        for level, face_data_list in enumerate(mip_data):
            mip_res = max(1, self.resolution >> level)
            cubemap = CubemapData(resolution=mip_res)

            for face_idx, compressed_face in enumerate(face_data_list):
                face = CubemapFace(face_idx)
                decompressed = compressor.decompress_face(
                    compressed_face, mip_res, face
                )
                cubemap.faces[face_idx] = decompressed

            roughness = level / max(1, self.mip_count - 1) if self.is_prefiltered else 0.0
            self._mip_chain.mips.append(MipLevel(
                level=level,
                resolution=mip_res,
                cubemap=cubemap,
                roughness=roughness,
            ))

        self.loaded = True

    def unload(self) -> None:
        """Unload runtime data."""
        self._mip_chain = None
        self.loaded = False

    def sample(self, direction: Vec3, roughness: float = 0.0) -> Vec3:
        """Sample the probe in a direction.

        Args:
            direction: Sample direction
            roughness: Surface roughness for mip selection

        Returns:
            Sampled color
        """
        if not self.loaded or self._mip_chain is None:
            return Vec3(0, 0, 0)

        pixel = self._mip_chain.sample_roughness(direction, roughness)
        return pixel.to_vec3()


class BakedProbeCapture:
    """Main interface for baked probe capture.

    Orchestrates cubemap rendering, pre-filtering, compression,
    and KTX2 serialization.
    """

    _id_counter: int = 0

    def __init__(
        self,
        renderer: CubemapRenderer,
        config: BakedProbeConfig,
    ) -> None:
        """Initialize capture system.

        Args:
            renderer: Cubemap rendering backend
            config: Capture configuration
        """
        self._renderer = renderer
        self._config = config
        self._compressor = BC6HCompressor(config.compression_quality)
        self._prefilter = PrefilteredGenerator(
            sample_count=config.sample_count,
            roughness_levels=config.roughness_levels,
        )
        self._ktx2_writer = KTX2Writer()

    @property
    def config(self) -> BakedProbeConfig:
        """Get capture configuration."""
        return self._config

    def capture_probe(
        self,
        name: str,
        position: Vec3,
        bounds: AABB,
        prefilter: bool = True,
    ) -> BakedProbeAsset:
        """Capture and bake a probe at a position.

        Args:
            name: Probe name
            position: World position
            bounds: Influence bounds
            prefilter: Whether to generate pre-filtered mips

        Returns:
            Baked probe asset ready for saving/loading
        """
        # Capture raw cubemap
        cubemap = self._renderer.capture(position)

        # Generate mip chain
        if prefilter:
            mip_chain = self._prefilter.generate_prefiltered(cubemap)
        else:
            generator = MipGenerator()
            mip_chain = generator.generate_mips(cubemap)

        # Compress each mip level
        compressed_mips = []
        for mip in mip_chain.mips:
            compressed_faces = self._compressor.compress_cubemap(mip.cubemap)
            compressed_mips.append(compressed_faces)

        # Write to KTX2
        ktx2_data = self._ktx2_writer.write_to_bytes(
            mip_chain,
            compressed_mips,
            supercompress=self._config.supercompress,
        )

        # Create asset
        BakedProbeCapture._id_counter += 1
        return BakedProbeAsset(
            probe_id=BakedProbeCapture._id_counter,
            name=name,
            position=position,
            bounds=bounds,
            resolution=self._config.resolution,
            mip_count=len(mip_chain.mips),
            is_prefiltered=prefilter,
            ktx2_data=ktx2_data,
            loaded=False,
        )

    def save_probe(self, asset: BakedProbeAsset, path: Path) -> int:
        """Save a baked probe to disk.

        Args:
            asset: Probe asset to save
            path: Output path

        Returns:
            Written file size
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            # Write header
            f.write(b'BKPR')  # Magic
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<I', asset.probe_id))
            f.write(struct.pack('<I', len(asset.name)))
            f.write(asset.name.encode('utf-8'))

            # Position and bounds
            f.write(struct.pack('<fff', asset.position.x, asset.position.y, asset.position.z))
            f.write(struct.pack('<fff', asset.bounds.min.x, asset.bounds.min.y, asset.bounds.min.z))
            f.write(struct.pack('<fff', asset.bounds.max.x, asset.bounds.max.y, asset.bounds.max.z))

            # Properties
            f.write(struct.pack('<I', asset.resolution))
            f.write(struct.pack('<I', asset.mip_count))
            f.write(struct.pack('<B', 1 if asset.is_prefiltered else 0))

            # KTX2 data
            f.write(struct.pack('<I', len(asset.ktx2_data)))
            f.write(asset.ktx2_data)

            return f.tell()

    def load_probe(self, path: Path) -> BakedProbeAsset:
        """Load a baked probe from disk.

        Args:
            path: Path to probe file

        Returns:
            Loaded probe asset
        """
        with open(path, 'rb') as f:
            # Verify magic
            magic = f.read(4)
            if magic != b'BKPR':
                raise ValueError("Invalid baked probe file")

            version = struct.unpack('<I', f.read(4))[0]
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")

            probe_id = struct.unpack('<I', f.read(4))[0]
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            # Position and bounds
            px, py, pz = struct.unpack('<fff', f.read(12))
            position = Vec3(px, py, pz)

            bmin_x, bmin_y, bmin_z = struct.unpack('<fff', f.read(12))
            bmax_x, bmax_y, bmax_z = struct.unpack('<fff', f.read(12))
            bounds = AABB(Vec3(bmin_x, bmin_y, bmin_z), Vec3(bmax_x, bmax_y, bmax_z))

            # Properties
            resolution = struct.unpack('<I', f.read(4))[0]
            mip_count = struct.unpack('<I', f.read(4))[0]
            is_prefiltered = struct.unpack('<B', f.read(1))[0] != 0

            # KTX2 data
            ktx2_size = struct.unpack('<I', f.read(4))[0]
            ktx2_data = f.read(ktx2_size)

            return BakedProbeAsset(
                probe_id=probe_id,
                name=name,
                position=position,
                bounds=bounds,
                resolution=resolution,
                mip_count=mip_count,
                is_prefiltered=is_prefiltered,
                ktx2_data=ktx2_data,
                loaded=False,
            )


# -----------------------------------------------------------------------------
# Probe Manager
# -----------------------------------------------------------------------------

class BakedProbeManager:
    """Manages multiple baked probes.

    Handles probe loading, LOD selection, and blending.
    """

    def __init__(self, max_loaded_probes: int = 32) -> None:
        """Initialize probe manager.

        Args:
            max_loaded_probes: Maximum probes to keep loaded
        """
        self._probes: dict[int, BakedProbeAsset] = {}
        self._max_loaded = max_loaded_probes
        self._loaded_order: list[int] = []

    def add_probe(self, probe: BakedProbeAsset) -> None:
        """Add a probe to the manager.

        Args:
            probe: Probe to add
        """
        self._probes[probe.probe_id] = probe

    def remove_probe(self, probe_id: int) -> None:
        """Remove a probe from the manager.

        Args:
            probe_id: ID of probe to remove
        """
        if probe_id in self._probes:
            self._probes[probe_id].unload()
            del self._probes[probe_id]
            if probe_id in self._loaded_order:
                self._loaded_order.remove(probe_id)

    def get_probe(self, probe_id: int) -> Optional[BakedProbeAsset]:
        """Get a probe by ID.

        Args:
            probe_id: Probe ID

        Returns:
            Probe asset or None
        """
        return self._probes.get(probe_id)

    def load_probe(self, probe_id: int) -> bool:
        """Ensure a probe is loaded.

        Args:
            probe_id: Probe to load

        Returns:
            True if loaded successfully
        """
        probe = self._probes.get(probe_id)
        if probe is None:
            return False

        if probe.loaded:
            return True

        # Evict old probes if necessary
        while len(self._loaded_order) >= self._max_loaded:
            evict_id = self._loaded_order.pop(0)
            if evict_id in self._probes:
                self._probes[evict_id].unload()

        probe.load()
        self._loaded_order.append(probe_id)
        return True

    def find_affecting_probes(
        self,
        point: Vec3,
        max_probes: int = 4,
    ) -> list[tuple[BakedProbeAsset, float]]:
        """Find probes affecting a point.

        Args:
            point: World position
            max_probes: Maximum probes to return

        Returns:
            List of (probe, weight) tuples
        """
        affecting = []

        for probe in self._probes.values():
            if probe.bounds.contains(point):
                # Compute blend weight based on distance to center
                center = probe.bounds.center
                extent = probe.bounds.max - probe.bounds.min
                max_dist = extent.length() * 0.5

                dist = point.distance(center)
                weight = 1.0 - min(dist / max(max_dist, 0.001), 1.0)

                affecting.append((probe, weight))

        # Sort by weight and take top probes
        affecting.sort(key=lambda x: x[1], reverse=True)
        return affecting[:max_probes]

    def sample(
        self,
        point: Vec3,
        direction: Vec3,
        roughness: float = 0.0,
    ) -> Vec3:
        """Sample blended probe lighting at a point.

        Args:
            point: World position
            direction: Sample direction
            roughness: Surface roughness

        Returns:
            Blended probe color
        """
        affecting = self.find_affecting_probes(point)

        if not affecting:
            return Vec3(0, 0, 0)

        result = Vec3(0, 0, 0)
        total_weight = 0.0

        for probe, weight in affecting:
            if not probe.loaded:
                self.load_probe(probe.probe_id)

            if probe.loaded:
                sample = probe.sample(direction, roughness)
                result = result + sample * weight
                total_weight += weight

        if total_weight > 0:
            return result * (1.0 / total_weight)
        return Vec3(0, 0, 0)

    def clear(self) -> None:
        """Unload and remove all probes."""
        for probe in self._probes.values():
            probe.unload()
        self._probes.clear()
        self._loaded_order.clear()
