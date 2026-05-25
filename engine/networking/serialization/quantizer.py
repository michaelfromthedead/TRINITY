"""
Numeric quantization utilities for network compression.

Provides functions for compressing floating-point values, 3D vectors,
and quaternions into compact binary representations.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from typing import Tuple, Union

from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Vector3:
    """Simple 3D vector for quantization."""
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


@dataclass
class Quaternion:
    """Quaternion for rotation representation (x, y, z, w)."""
    x: float
    y: float
    z: float
    w: float

    def normalize(self) -> 'Quaternion':
        """Return normalized quaternion."""
        magnitude = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if magnitude < DEFAULT_CONFIG.NORMALIZATION_EPSILON:
            return Quaternion(0.0, 0.0, 0.0, 1.0)
        return Quaternion(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
            self.w / magnitude
        )


def quantize_float(value: float, min_value: float, max_value: float, bits: int) -> int:
    """
    Quantize a float to an integer with specified bit precision.

    Args:
        value: The float value to quantize.
        min_value: Minimum of the expected range.
        max_value: Maximum of the expected range.
        bits: Number of bits for the quantized value (1-32).

    Returns:
        Quantized integer value (0 to 2^bits - 1).

    Raises:
        ValueError: If bits out of range or min >= max.

    Example:
        >>> quantize_float(0.5, 0.0, 1.0, 8)
        128  # Approximately half of 255
    """
    if bits < 1 or bits > 32:
        raise ValueError(f"bits must be 1-32, got {bits}")
    if min_value >= max_value:
        raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

    # Clamp value to range
    value = max(min_value, min(max_value, value))

    # Normalize to [0, 1]
    normalized = (value - min_value) / (max_value - min_value)

    # Scale to bit range
    max_int = (1 << bits) - 1
    quantized = int(round(normalized * max_int))

    return max(0, min(max_int, quantized))


def dequantize_float(quantized: int, min_value: float, max_value: float, bits: int) -> float:
    """
    Dequantize an integer back to a float.

    Args:
        quantized: The quantized integer value.
        min_value: Minimum of the expected range.
        max_value: Maximum of the expected range.
        bits: Number of bits used for quantization (1-32).

    Returns:
        The dequantized float value.

    Raises:
        ValueError: If bits out of range or min >= max.

    Example:
        >>> dequantize_float(128, 0.0, 1.0, 8)
        0.5019...  # Approximately 0.5
    """
    if bits < 1 or bits > 32:
        raise ValueError(f"bits must be 1-32, got {bits}")
    if min_value >= max_value:
        raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

    max_int = (1 << bits) - 1

    # Clamp quantized value
    quantized = max(0, min(max_int, quantized))

    # Normalize and scale
    normalized = quantized / max_int
    return min_value + normalized * (max_value - min_value)


def quantize_vector3(
    vec: Union[Vector3, Tuple[float, float, float]],
    precision: int = DEFAULT_CONFIG.VECTOR_PRECISION_DEFAULT
) -> bytes:
    """
    Quantize a 3D vector to compressed bytes.

    Uses the specified bits per component for compression.
    Default range is -1000 to 1000 for each component.

    Args:
        vec: Vector3 or (x, y, z) tuple.
        precision: Bits per component (8, 12, 16, or 24). Default 16.

    Returns:
        Compressed bytes representation.

    Raises:
        ValueError: If precision is not supported.

    Example:
        >>> data = quantize_vector3((10.5, -20.0, 5.25), precision=16)
        >>> len(data)
        6  # 2 bytes per component
    """
    if precision not in DEFAULT_CONFIG.VECTOR_PRECISIONS:
        raise ValueError(f"precision must be one of {DEFAULT_CONFIG.VECTOR_PRECISIONS}, got {precision}")

    if isinstance(vec, Vector3):
        x, y, z = vec.x, vec.y, vec.z
    else:
        x, y, z = vec

    # Default range for positions
    min_val = DEFAULT_CONFIG.VECTOR_RANGE_MIN
    max_val = DEFAULT_CONFIG.VECTOR_RANGE_MAX

    # Quantize each component
    qx = quantize_float(x, min_val, max_val, precision)
    qy = quantize_float(y, min_val, max_val, precision)
    qz = quantize_float(z, min_val, max_val, precision)

    # Pack based on precision
    if precision == 8:
        return struct.pack('!BBB', qx, qy, qz)
    elif precision == 12:
        # Pack 3x12 bits = 36 bits = 5 bytes (with 4 padding bits)
        combined = (qx << 24) | (qy << 12) | qz
        return combined.to_bytes(5, 'big')
    elif precision == 16:
        return struct.pack('!HHH', qx, qy, qz)
    else:  # 24
        # Pack 3x24 bits = 72 bits = 9 bytes
        result = bytearray(9)
        result[0:3] = qx.to_bytes(3, 'big')
        result[3:6] = qy.to_bytes(3, 'big')
        result[6:9] = qz.to_bytes(3, 'big')
        return bytes(result)


def dequantize_vector3(
    data: bytes,
    precision: int = DEFAULT_CONFIG.VECTOR_PRECISION_DEFAULT
) -> Vector3:
    """
    Dequantize bytes back to a 3D vector.

    Args:
        data: Compressed bytes from quantize_vector3.
        precision: Bits per component (must match quantization).

    Returns:
        Vector3 with dequantized components.

    Raises:
        ValueError: If precision is not supported or data length is wrong.
    """
    if precision not in DEFAULT_CONFIG.VECTOR_PRECISIONS:
        raise ValueError(f"precision must be one of {DEFAULT_CONFIG.VECTOR_PRECISIONS}, got {precision}")

    min_val = DEFAULT_CONFIG.VECTOR_RANGE_MIN
    max_val = DEFAULT_CONFIG.VECTOR_RANGE_MAX

    if precision == 8:
        if len(data) < 3:
            raise ValueError("Need at least 3 bytes for 8-bit precision")
        qx, qy, qz = struct.unpack('!BBB', data[:3])
    elif precision == 12:
        if len(data) < 5:
            raise ValueError("Need at least 5 bytes for 12-bit precision")
        combined = int.from_bytes(data[:5], 'big')
        qx = (combined >> 24) & 0xFFF
        qy = (combined >> 12) & 0xFFF
        qz = combined & 0xFFF
    elif precision == 16:
        if len(data) < 6:
            raise ValueError("Need at least 6 bytes for 16-bit precision")
        qx, qy, qz = struct.unpack('!HHH', data[:6])
    else:  # 24
        if len(data) < 9:
            raise ValueError("Need at least 9 bytes for 24-bit precision")
        qx = int.from_bytes(data[0:3], 'big')
        qy = int.from_bytes(data[3:6], 'big')
        qz = int.from_bytes(data[6:9], 'big')

    return Vector3(
        dequantize_float(qx, min_val, max_val, precision),
        dequantize_float(qy, min_val, max_val, precision),
        dequantize_float(qz, min_val, max_val, precision)
    )


def quantize_quaternion(quat: Union[Quaternion, Tuple[float, float, float, float]]) -> bytes:
    """
    Quantize a quaternion using smallest-three encoding.

    This encoding drops the largest component (which can be reconstructed)
    and encodes the other three with 10 bits each, plus 2 bits to indicate
    which component was dropped. Total: 32 bits = 4 bytes.

    Args:
        quat: Quaternion or (x, y, z, w) tuple.

    Returns:
        4 bytes of compressed quaternion data.

    Example:
        >>> q = Quaternion(0.0, 0.0, 0.0, 1.0)  # Identity
        >>> data = quantize_quaternion(q)
        >>> len(data)
        4
    """
    if isinstance(quat, Quaternion):
        components = [quat.x, quat.y, quat.z, quat.w]
    else:
        components = list(quat)

    # Normalize
    magnitude = math.sqrt(sum(c**2 for c in components))
    if magnitude < DEFAULT_CONFIG.NORMALIZATION_EPSILON:
        components = [0.0, 0.0, 0.0, 1.0]
    else:
        components = [c / magnitude for c in components]

    # Find largest component (by absolute value)
    abs_components = [abs(c) for c in components]
    largest_index = abs_components.index(max(abs_components))

    # If largest is negative, negate entire quaternion (quaternions are double-cover)
    if components[largest_index] < 0:
        components = [-c for c in components]

    # Get the three smaller components
    smaller = [components[i] for i in range(4) if i != largest_index]

    # Quantize each to 10 bits
    # Range is [-1/sqrt(2), 1/sqrt(2)] since the others must be smaller
    min_val = DEFAULT_CONFIG.QUATERNION_COMPONENT_MIN
    max_val = DEFAULT_CONFIG.QUATERNION_COMPONENT_MAX

    q0 = quantize_float(smaller[0], min_val, max_val, 10)
    q1 = quantize_float(smaller[1], min_val, max_val, 10)
    q2 = quantize_float(smaller[2], min_val, max_val, 10)

    # Pack: 2 bits index + 10+10+10 bits = 32 bits
    packed = (largest_index << 30) | (q0 << 20) | (q1 << 10) | q2
    return packed.to_bytes(4, 'big')


def dequantize_quaternion(data: bytes) -> Quaternion:
    """
    Dequantize bytes back to a quaternion.

    Args:
        data: 4 bytes from quantize_quaternion.

    Returns:
        Quaternion with normalized components.

    Raises:
        ValueError: If data is not 4 bytes.
    """
    if len(data) < 4:
        raise ValueError("Need 4 bytes for quaternion")

    packed = int.from_bytes(data[:4], 'big')

    largest_index = (packed >> 30) & 0x3
    q0 = (packed >> 20) & 0x3FF
    q1 = (packed >> 10) & 0x3FF
    q2 = packed & 0x3FF

    # Dequantize
    min_val = DEFAULT_CONFIG.QUATERNION_COMPONENT_MIN
    max_val = DEFAULT_CONFIG.QUATERNION_COMPONENT_MAX

    smaller = [
        dequantize_float(q0, min_val, max_val, 10),
        dequantize_float(q1, min_val, max_val, 10),
        dequantize_float(q2, min_val, max_val, 10),
    ]

    # Reconstruct largest component
    # w^2 = 1 - x^2 - y^2 - z^2
    sum_sq = sum(c**2 for c in smaller)
    largest = math.sqrt(max(0.0, 1.0 - sum_sq))

    # Rebuild quaternion
    components = [0.0, 0.0, 0.0, 0.0]
    small_idx = 0
    for i in range(4):
        if i == largest_index:
            components[i] = largest
        else:
            components[i] = smaller[small_idx]
            small_idx += 1

    return Quaternion(components[0], components[1], components[2], components[3])


def quantize_angle(angle: float, bits: int = 8) -> int:
    """
    Quantize an angle in radians to a compact integer.

    Args:
        angle: Angle in radians.
        bits: Bits for quantization (default 8 = 256 values).

    Returns:
        Quantized angle value.
    """
    # Normalize to [0, 2*pi)
    two_pi = 2.0 * math.pi
    normalized = angle % two_pi
    if normalized < 0:
        normalized += two_pi

    return quantize_float(normalized, 0.0, two_pi, bits)


def dequantize_angle(quantized: int, bits: int = 8) -> float:
    """
    Dequantize an integer back to an angle in radians.

    Args:
        quantized: Quantized angle value.
        bits: Bits used for quantization.

    Returns:
        Angle in radians [0, 2*pi).
    """
    two_pi = 2.0 * math.pi
    return dequantize_float(quantized, 0.0, two_pi, bits)


def quantize_unit_float(value: float, bits: int = 8) -> int:
    """
    Quantize a float in [0, 1] range.

    Args:
        value: Float in [0, 1].
        bits: Bits for quantization.

    Returns:
        Quantized value.
    """
    return quantize_float(value, 0.0, 1.0, bits)


def dequantize_unit_float(quantized: int, bits: int = 8) -> float:
    """
    Dequantize an integer back to a [0, 1] float.

    Args:
        quantized: Quantized value.
        bits: Bits used for quantization.

    Returns:
        Float in [0, 1].
    """
    return dequantize_float(quantized, 0.0, 1.0, bits)


def quantize_signed_unit_float(value: float, bits: int = 8) -> int:
    """
    Quantize a float in [-1, 1] range.

    Args:
        value: Float in [-1, 1].
        bits: Bits for quantization.

    Returns:
        Quantized value.
    """
    return quantize_float(value, -1.0, 1.0, bits)


def dequantize_signed_unit_float(quantized: int, bits: int = 8) -> float:
    """
    Dequantize an integer back to a [-1, 1] float.

    Args:
        quantized: Quantized value.
        bits: Bits used for quantization.

    Returns:
        Float in [-1, 1].
    """
    return dequantize_float(quantized, -1.0, 1.0, bits)
