"""
Heightfield terrain data for the game engine World Layer.

This module provides heightfield-based terrain representation with:
- Configurable precision (16-bit or 32-bit)
- Bilinear interpolation for smooth height queries
- Normal calculation from neighboring heights
- Compression/decompression for storage
- Region sampling for efficient bulk queries

Coordinate System:
- X: Increases to the right (East)
- Z: Increases forward (North)
- Y: Height (Up)
- Origin (0,0) is at the corner of the heightfield
- Sample positions are at integer coordinates
- World position = sample_index * scale
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import math
import struct
import zlib

from .constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_HEIGHT_RANGE,
    DEFAULT_SCALE,
    MIN_RESOLUTION,
    BITS_16_MAX_VALUE,
    HEIGHT_EPSILON,
    NORMAL_EPSILON,
    NORMAL_DELTA_MULTIPLIER,
    GRADIENT_DIVISOR,
    ZLIB_COMPRESSION_LEVEL,
    COMPRESSED_HEADER_SIZE,
)


class HeightfieldPrecision(Enum):
    """Precision modes for heightfield data storage."""
    BITS_16 = auto()  # 65536 height levels - good for most terrain
    BITS_32 = auto()  # Full float precision - for high-detail terrain


@dataclass
class HeightfieldConfig:
    """Configuration for a heightfield.

    Attributes:
        resolution: Number of samples per edge. Default 65 (64+1 for stitching
                   with adjacent patches). Must be >= 2.
        precision: Storage precision. BITS_16 uses 2 bytes per sample,
                  BITS_32 uses 4 bytes per sample.
        height_range: (min, max) heights in world units. Heights are clamped
                     to this range and quantized for BITS_16 precision.
        scale: World units per sample. A scale of 1.0 means samples are
              1 world unit apart.
    """
    resolution: int = DEFAULT_RESOLUTION  # Samples per edge (64+1 for stitching)
    precision: HeightfieldPrecision = HeightfieldPrecision.BITS_16
    height_range: Tuple[float, float] = DEFAULT_HEIGHT_RANGE
    scale: float = DEFAULT_SCALE  # World units per sample

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.resolution < MIN_RESOLUTION:
            raise ValueError("resolution must be >= 2")
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.height_range[0] >= self.height_range[1]:
            raise ValueError("height_range[0] must be < height_range[1]")


class Heightfield:
    """2D heightfield terrain data with bilinear interpolation.

    The heightfield stores a 2D grid of height values. Heights can be
    queried at any (x, z) position using bilinear interpolation for
    smooth results between sample points.

    Sample Layout:
        Sample (i, j) represents height at world position (i * scale, j * scale)
        where i is the column (X) and j is the row (Z).

    Example:
        config = HeightfieldConfig(resolution=65, scale=1.0)
        hf = Heightfield(config)
        hf.set_height_at(10, 10, 25.5)
        height = hf.get_height_at(10.5, 10.5)  # Interpolated
        normal = hf.get_normal_at(10, 10)
    """

    def __init__(self, config: Optional[HeightfieldConfig] = None):
        """Initialize heightfield with given configuration.

        Args:
            config: Heightfield configuration. Uses defaults if None.
        """
        self.config = config or HeightfieldConfig()
        self._data: List[List[float]] = self._create_empty_data()
        self._dirty = True  # Track if data has changed since last compression
        self._cached_bounds: Optional[Tuple[float, float]] = None

    def _create_empty_data(self) -> List[List[float]]:
        """Create an empty 2D height array initialized to 0."""
        res = self.config.resolution
        return [[0.0 for _ in range(res)] for _ in range(res)]

    def _clamp_height(self, height: float) -> float:
        """Clamp height to configured range."""
        min_h, max_h = self.config.height_range
        return max(min_h, min(max_h, height))

    def _is_valid_sample_index(self, x: int, z: int) -> bool:
        """Check if sample indices are within bounds."""
        res = self.config.resolution
        return 0 <= x < res and 0 <= z < res

    def get_height_at(self, x: float, z: float) -> float:
        """Get interpolated height at world position (x, z).

        Uses bilinear interpolation between the four nearest sample points.
        Positions outside the heightfield are clamped to the edge.

        Args:
            x: X position in world units (not sample index)
            z: Z position in world units (not sample index)

        Returns:
            Interpolated height at the given position.
        """
        # Convert world position to sample-space coordinates
        scale = self.config.scale
        sx = x / scale
        sz = z / scale

        res = self.config.resolution

        # Clamp to valid range
        sx = max(0.0, min(res - 1.0, sx))
        sz = max(0.0, min(res - 1.0, sz))

        # Get integer sample indices
        x0 = int(math.floor(sx))
        z0 = int(math.floor(sz))
        x1 = min(x0 + 1, res - 1)
        z1 = min(z0 + 1, res - 1)

        # Get fractional part for interpolation
        fx = sx - x0
        fz = sz - z0

        # Get the four corner heights
        h00 = self._data[z0][x0]
        h10 = self._data[z0][x1]
        h01 = self._data[z1][x0]
        h11 = self._data[z1][x1]

        # Bilinear interpolation
        # First interpolate along X at z0 and z1
        h0 = h00 * (1.0 - fx) + h10 * fx
        h1 = h01 * (1.0 - fx) + h11 * fx

        # Then interpolate along Z
        return h0 * (1.0 - fz) + h1 * fz

    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        """Compute surface normal at world position (x, z).

        Calculates the normal using central differences from neighboring
        height samples. The normal is normalized to unit length.

        Args:
            x: X position in world units
            z: Z position in world units

        Returns:
            Tuple (nx, ny, nz) representing the unit normal vector.
            ny is always positive (pointing upward).
        """
        scale = self.config.scale

        # Use a small delta for gradient calculation
        delta = scale * NORMAL_DELTA_MULTIPLIER

        # Sample heights at neighboring points
        h_left = self.get_height_at(x - delta, z)
        h_right = self.get_height_at(x + delta, z)
        h_back = self.get_height_at(x, z - delta)
        h_front = self.get_height_at(x, z + delta)

        # Calculate gradient
        dx = (h_right - h_left) / (GRADIENT_DIVISOR * delta)
        dz = (h_front - h_back) / (2.0 * delta)

        # Normal is (-dx, 1, -dz) normalized
        # This comes from the cross product of tangent vectors
        nx = -dx
        ny = 1.0
        nz = -dz

        # Normalize with epsilon to avoid division by near-zero
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > NORMAL_EPSILON:
            nx /= length
            ny /= length
            nz /= length

        return (nx, ny, nz)

    def set_height_at(self, x: int, z: int, height: float) -> bool:
        """Set height at sample position (x, z).

        Args:
            x: Sample X index (column)
            z: Sample Z index (row)
            height: Height value to set (will be clamped to height_range)

        Returns:
            True if successful, False if indices out of bounds.
        """
        if not self._is_valid_sample_index(x, z):
            return False

        self._data[z][x] = self._clamp_height(height)
        self._dirty = True
        self._cached_bounds = None
        return True

    def get_raw_height_at(self, x: int, z: int) -> Optional[float]:
        """Get raw height at sample position without interpolation.

        Args:
            x: Sample X index (column)
            z: Sample Z index (row)

        Returns:
            Height value or None if indices out of bounds.
        """
        if not self._is_valid_sample_index(x, z):
            return None
        return self._data[z][x]

    def sample_region(
        self,
        min_x: int,
        min_z: int,
        max_x: int,
        max_z: int
    ) -> List[List[float]]:
        """Sample a rectangular region of heights.

        Args:
            min_x: Minimum X sample index (inclusive)
            min_z: Minimum Z sample index (inclusive)
            max_x: Maximum X sample index (inclusive)
            max_z: Maximum Z sample index (inclusive)

        Returns:
            2D list of heights for the region. Returns empty list if
            region is invalid or out of bounds.
        """
        res = self.config.resolution

        # Clamp to valid range
        min_x = max(0, min(res - 1, min_x))
        min_z = max(0, min(res - 1, min_z))
        max_x = max(0, min(res - 1, max_x))
        max_z = max(0, min(res - 1, max_z))

        if min_x > max_x or min_z > max_z:
            return []

        result = []
        for z in range(min_z, max_z + 1):
            row = []
            for x in range(min_x, max_x + 1):
                row.append(self._data[z][x])
            result.append(row)

        return result

    def import_from_data(self, data: List[List[float]]) -> bool:
        """Import height data from a 2D list.

        The data must match the configured resolution. Heights are
        clamped to the configured height_range.

        Args:
            data: 2D list of heights [z][x]

        Returns:
            True if import successful, False if data dimensions don't match.
        """
        res = self.config.resolution

        if len(data) != res:
            return False

        for row in data:
            if len(row) != res:
                return False

        # Copy and clamp data
        for z in range(res):
            for x in range(res):
                self._data[z][x] = self._clamp_height(data[z][x])

        self._dirty = True
        self._cached_bounds = None
        return True

    def export_to_data(self) -> List[List[float]]:
        """Export height data to a 2D list.

        Returns:
            Copy of the internal height data [z][x].
        """
        return [row[:] for row in self._data]

    def compress(self) -> bytes:
        """Compress heightfield data for storage.

        Uses quantization (for BITS_16) followed by zlib compression.

        Returns:
            Compressed bytes containing:
            - 4 bytes: resolution (uint32)
            - 1 byte: precision flag (0=16bit, 1=32bit)
            - 8 bytes: min_height (float64)
            - 8 bytes: max_height (float64)
            - 8 bytes: scale (float64)
            - N bytes: zlib-compressed height data
        """
        res = self.config.resolution
        precision = self.config.precision
        min_h, max_h = self.config.height_range
        scale = self.config.scale

        # Build header
        header = struct.pack('<I', res)  # resolution
        header += struct.pack('<B', 0 if precision == HeightfieldPrecision.BITS_16 else 1)
        header += struct.pack('<d', min_h)
        header += struct.pack('<d', max_h)
        header += struct.pack('<d', scale)

        # Pack height data
        height_range = max_h - min_h
        raw_data = bytearray()

        for z in range(res):
            for x in range(res):
                h = self._data[z][x]

                if precision == HeightfieldPrecision.BITS_16:
                    # Quantize to 16-bit
                    if height_range > 0:
                        normalized = (h - min_h) / height_range
                        quantized = int(normalized * BITS_16_MAX_VALUE)
                        quantized = max(0, min(BITS_16_MAX_VALUE, quantized))
                    else:
                        quantized = 0
                    raw_data.extend(struct.pack('<H', quantized))
                else:
                    # Full float precision
                    raw_data.extend(struct.pack('<f', h))

        # Compress the height data
        compressed = zlib.compress(bytes(raw_data), level=ZLIB_COMPRESSION_LEVEL)

        return header + compressed

    @classmethod
    def decompress(cls, data: bytes) -> 'Heightfield':
        """Decompress heightfield data from storage.

        Args:
            data: Compressed bytes from compress()

        Returns:
            New Heightfield instance with the decompressed data.

        Raises:
            ValueError: If data is corrupted or too short.
        """
        if len(data) < COMPRESSED_HEADER_SIZE:
            raise ValueError("Data too short for heightfield header")

        # Parse header
        offset = 0
        resolution = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        precision_flag = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        precision = HeightfieldPrecision.BITS_16 if precision_flag == 0 else HeightfieldPrecision.BITS_32

        min_h = struct.unpack_from('<d', data, offset)[0]
        offset += 8

        max_h = struct.unpack_from('<d', data, offset)[0]
        offset += 8

        scale = struct.unpack_from('<d', data, offset)[0]
        offset += 8

        # Decompress height data
        compressed_data = data[offset:]
        try:
            raw_data = zlib.decompress(compressed_data)
        except zlib.error as e:
            raise ValueError(f"Failed to decompress heightfield data: {e}")

        # Create heightfield
        config = HeightfieldConfig(
            resolution=resolution,
            precision=precision,
            height_range=(min_h, max_h),
            scale=scale
        )
        hf = cls(config)

        # Unpack height data
        height_range = max_h - min_h
        data_offset = 0

        for z in range(resolution):
            for x in range(resolution):
                if precision == HeightfieldPrecision.BITS_16:
                    quantized = struct.unpack_from('<H', raw_data, data_offset)[0]
                    data_offset += 2

                    if height_range > 0:
                        h = min_h + (quantized / float(BITS_16_MAX_VALUE)) * height_range
                    else:
                        h = min_h
                else:
                    h = struct.unpack_from('<f', raw_data, data_offset)[0]
                    data_offset += 4

                hf._data[z][x] = h

        hf._dirty = False
        return hf

    def get_bounds(self) -> Tuple[float, float]:
        """Get the actual min and max heights in the data.

        Returns:
            Tuple (min_height, max_height) of actual data values.
        """
        if self._cached_bounds is not None:
            return self._cached_bounds

        min_h = float('inf')
        max_h = float('-inf')

        for row in self._data:
            for h in row:
                min_h = min(min_h, h)
                max_h = max(max_h, h)

        if min_h == float('inf'):
            min_h = 0.0
            max_h = 0.0

        self._cached_bounds = (min_h, max_h)
        return self._cached_bounds

    def get_world_size(self) -> Tuple[float, float]:
        """Get the world-space dimensions of the heightfield.

        Returns:
            Tuple (width_x, depth_z) in world units.
        """
        edge_samples = self.config.resolution - 1
        size = edge_samples * self.config.scale
        return (size, size)

    def fill(self, height: float) -> None:
        """Fill entire heightfield with a constant height.

        Args:
            height: Height value to fill with (clamped to range).
        """
        h = self._clamp_height(height)
        res = self.config.resolution

        for z in range(res):
            for x in range(res):
                self._data[z][x] = h

        self._dirty = True
        self._cached_bounds = (h, h)

    def copy(self) -> 'Heightfield':
        """Create a deep copy of this heightfield.

        Returns:
            New Heightfield with copied configuration and data.
        """
        new_config = HeightfieldConfig(
            resolution=self.config.resolution,
            precision=self.config.precision,
            height_range=self.config.height_range,
            scale=self.config.scale
        )
        new_hf = Heightfield(new_config)
        new_hf._data = [row[:] for row in self._data]
        new_hf._dirty = self._dirty
        new_hf._cached_bounds = self._cached_bounds
        return new_hf

    def __eq__(self, other: object) -> bool:
        """Check equality with another heightfield."""
        if not isinstance(other, Heightfield):
            return False

        if self.config.resolution != other.config.resolution:
            return False
        if self.config.precision != other.config.precision:
            return False
        if self.config.height_range != other.config.height_range:
            return False
        if self.config.scale != other.config.scale:
            return False

        # Compare data with tolerance for floating point
        for z in range(self.config.resolution):
            for x in range(self.config.resolution):
                if abs(self._data[z][x] - other._data[z][x]) > HEIGHT_EPSILON:
                    return False

        return True
