"""
Bit-level packing and unpacking for efficient network serialization.

Provides BitWriter and BitReader classes for writing and reading
individual bits, enabling compact encoding of game state data.
"""

from __future__ import annotations

import logging
import struct
from typing import List, Optional

from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BitWriter:
    """
    Writes data at the bit level for compact serialization.

    Supports writing individual bits, booleans, bounded integers,
    and compressed floating-point values.

    Example:
        writer = BitWriter()
        writer.write_bool(True)
        writer.write_int(50, 0, 100)  # Uses 7 bits
        writer.write_float_compressed(3.14, 0.0, 10.0, 0.01)
        data = writer.to_bytes()
    """

    def __init__(self, initial_capacity: int = DEFAULT_CONFIG.BIT_WRITER_INITIAL_CAPACITY) -> None:
        """
        Initialize the bit writer.

        Args:
            initial_capacity: Initial buffer size in bytes.
        """
        self._buffer: bytearray = bytearray(initial_capacity)
        self._bit_position: int = 0
        self._byte_position: int = 0

    @property
    def bit_position(self) -> int:
        """Current position in bits."""
        return self._byte_position * 8 + self._bit_position

    @property
    def byte_length(self) -> int:
        """Number of bytes written (rounded up)."""
        return self._byte_position + (1 if self._bit_position > 0 else 0)

    def _ensure_capacity(self, additional_bits: int) -> None:
        """Ensure buffer has capacity for additional bits."""
        needed_bytes = (self.bit_position + additional_bits + 7) // 8
        if needed_bytes > len(self._buffer):
            new_size = max(len(self._buffer) * 2, needed_bytes)
            self._buffer.extend(bytearray(new_size - len(self._buffer)))

    def write_bits(self, value: int, num_bits: int) -> None:
        """
        Write a specific number of bits to the buffer.

        Args:
            value: The value to write (only lowest num_bits are used).
            num_bits: Number of bits to write (1-64).

        Raises:
            ValueError: If num_bits is out of range.
        """
        if num_bits < 1 or num_bits > 64:
            raise ValueError(f"num_bits must be 1-64, got {num_bits}")

        self._ensure_capacity(num_bits)

        # Mask the value to the number of bits
        mask = (1 << num_bits) - 1
        value = value & mask

        # Write bits
        remaining_bits = num_bits
        while remaining_bits > 0:
            # Bits available in current byte
            bits_in_byte = 8 - self._bit_position
            bits_to_write = min(bits_in_byte, remaining_bits)

            # Calculate shift and mask for this chunk
            shift = remaining_bits - bits_to_write
            chunk = (value >> shift) & ((1 << bits_to_write) - 1)

            # Position the chunk in the current byte
            byte_shift = bits_in_byte - bits_to_write
            self._buffer[self._byte_position] |= chunk << byte_shift

            remaining_bits -= bits_to_write
            self._bit_position += bits_to_write

            if self._bit_position >= 8:
                self._bit_position = 0
                self._byte_position += 1
                if self._byte_position < len(self._buffer):
                    self._buffer[self._byte_position] = 0

    def write_bool(self, value: bool) -> None:
        """
        Write a boolean as a single bit.

        Args:
            value: The boolean value to write.
        """
        self.write_bits(1 if value else 0, 1)

    def write_int(self, value: int, min_value: int, max_value: int) -> None:
        """
        Write a bounded integer using the minimum required bits.

        Args:
            value: The value to write (will be clamped to range).
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.

        Raises:
            ValueError: If min_value >= max_value.
        """
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

        # Clamp value to range
        value = max(min_value, min(max_value, value))

        # Calculate bits needed
        range_size = max_value - min_value
        num_bits = range_size.bit_length()

        # Write normalized value
        self.write_bits(value - min_value, num_bits)

    def write_float_compressed(
        self,
        value: float,
        min_value: float,
        max_value: float,
        precision: float
    ) -> None:
        """
        Write a float with specified precision using quantization.

        Args:
            value: The float value to write (will be clamped).
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            precision: Smallest representable difference.

        Raises:
            ValueError: If precision <= 0 or range is invalid.
        """
        if precision <= 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

        # Clamp value
        value = max(min_value, min(max_value, value))

        # Quantize
        range_size = max_value - min_value
        num_steps = int(range_size / precision)
        num_bits = max(1, num_steps.bit_length())

        normalized = (value - min_value) / range_size
        quantized = int(round(normalized * num_steps))
        quantized = max(0, min(num_steps, quantized))

        self.write_bits(quantized, num_bits)

    def write_bytes(self, data: bytes) -> None:
        """
        Write raw bytes, aligning to byte boundary first.

        Args:
            data: Bytes to write.
        """
        self.align_to_byte()
        for byte in data:
            self.write_bits(byte, 8)

    def write_string(self, value: str, max_length: int = DEFAULT_CONFIG.MAX_STRING_LENGTH) -> None:
        """
        Write a UTF-8 string with length prefix.

        Args:
            value: String to write.
            max_length: Maximum string length (default 255).
        """
        encoded = value.encode('utf-8')[:max_length]
        length_bits = max_length.bit_length()
        self.write_bits(len(encoded), length_bits)
        self.write_bytes(encoded)

    def align_to_byte(self) -> None:
        """Advance to the next byte boundary if not already aligned."""
        if self._bit_position > 0:
            self._bit_position = 0
            self._byte_position += 1
            if self._byte_position < len(self._buffer):
                self._buffer[self._byte_position] = 0

    def to_bytes(self) -> bytes:
        """
        Get the written data as bytes.

        Returns:
            The serialized byte data.
        """
        return bytes(self._buffer[:self.byte_length])

    def reset(self) -> None:
        """Reset the writer to empty state."""
        self._buffer = bytearray(len(self._buffer))
        self._bit_position = 0
        self._byte_position = 0


class BitReader:
    """
    Reads data at the bit level for deserialization.

    Provides methods to read individual bits, booleans, bounded integers,
    and compressed floating-point values.

    Example:
        reader = BitReader(data)
        flag = reader.read_bool()
        value = reader.read_int(0, 100)
        position = reader.read_float_compressed(0.0, 10.0, 0.01)
    """

    def __init__(self, data: bytes) -> None:
        """
        Initialize the bit reader with data.

        Args:
            data: The byte data to read from.
        """
        self._buffer: bytes = data
        self._bit_position: int = 0
        self._byte_position: int = 0

    @property
    def bit_position(self) -> int:
        """Current position in bits."""
        return self._byte_position * 8 + self._bit_position

    @property
    def bits_remaining(self) -> int:
        """Number of bits left to read."""
        return len(self._buffer) * 8 - self.bit_position

    def read_bits(self, num_bits: int) -> int:
        """
        Read a specific number of bits from the buffer.

        Args:
            num_bits: Number of bits to read (1-64).

        Returns:
            The read value.

        Raises:
            ValueError: If num_bits is out of range.
            EOFError: If not enough bits remain.
        """
        if num_bits < 1 or num_bits > 64:
            raise ValueError(f"num_bits must be 1-64, got {num_bits}")

        if self.bits_remaining < num_bits:
            raise EOFError(f"Not enough bits: need {num_bits}, have {self.bits_remaining}")

        result = 0
        remaining_bits = num_bits

        while remaining_bits > 0:
            # Bits available in current byte
            bits_in_byte = 8 - self._bit_position
            bits_to_read = min(bits_in_byte, remaining_bits)

            # Extract bits from current byte
            byte_shift = bits_in_byte - bits_to_read
            mask = ((1 << bits_to_read) - 1) << byte_shift
            chunk = (self._buffer[self._byte_position] & mask) >> byte_shift

            # Add to result
            result = (result << bits_to_read) | chunk

            remaining_bits -= bits_to_read
            self._bit_position += bits_to_read

            if self._bit_position >= 8:
                self._bit_position = 0
                self._byte_position += 1

        return result

    def read_bool(self) -> bool:
        """
        Read a boolean from a single bit.

        Returns:
            The boolean value.
        """
        return self.read_bits(1) == 1

    def read_int(self, min_value: int, max_value: int) -> int:
        """
        Read a bounded integer.

        Args:
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.

        Returns:
            The integer value.

        Raises:
            ValueError: If min_value >= max_value.
        """
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

        range_size = max_value - min_value
        num_bits = range_size.bit_length()

        return self.read_bits(num_bits) + min_value

    def read_float_compressed(
        self,
        min_value: float,
        max_value: float,
        precision: float
    ) -> float:
        """
        Read a quantized float value.

        Args:
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            precision: Smallest representable difference.

        Returns:
            The float value (may differ from original by up to precision/2).

        Raises:
            ValueError: If precision <= 0 or range is invalid.
        """
        if precision <= 0:
            raise ValueError(f"precision must be > 0, got {precision}")
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be < max_value ({max_value})")

        range_size = max_value - min_value
        num_steps = int(range_size / precision)
        num_bits = max(1, num_steps.bit_length())

        quantized = self.read_bits(num_bits)
        normalized = quantized / num_steps

        return min_value + normalized * range_size

    def read_bytes(self, count: int) -> bytes:
        """
        Read raw bytes, aligning to byte boundary first.

        Args:
            count: Number of bytes to read.

        Returns:
            The read bytes.
        """
        self.align_to_byte()
        result = bytearray(count)
        for i in range(count):
            result[i] = self.read_bits(8)
        return bytes(result)

    def read_string(self, max_length: int = DEFAULT_CONFIG.MAX_STRING_LENGTH) -> str:
        """
        Read a UTF-8 string with length prefix.

        Args:
            max_length: Maximum string length (default 255).

        Returns:
            The decoded string.
        """
        length_bits = max_length.bit_length()
        length = self.read_bits(length_bits)
        data = self.read_bytes(length)
        return data.decode('utf-8')

    def align_to_byte(self) -> None:
        """Advance to the next byte boundary if not already aligned."""
        if self._bit_position > 0:
            self._bit_position = 0
            self._byte_position += 1

    def peek_bits(self, num_bits: int) -> int:
        """
        Peek at bits without advancing the position.

        Args:
            num_bits: Number of bits to peek.

        Returns:
            The peeked value.
        """
        saved_bit_pos = self._bit_position
        saved_byte_pos = self._byte_position

        try:
            return self.read_bits(num_bits)
        finally:
            self._bit_position = saved_bit_pos
            self._byte_position = saved_byte_pos

    def skip_bits(self, num_bits: int) -> None:
        """
        Skip a number of bits.

        Args:
            num_bits: Number of bits to skip.
        """
        total_bits = self._byte_position * 8 + self._bit_position + num_bits
        self._byte_position = total_bits // 8
        self._bit_position = total_bits % 8

    def reset(self) -> None:
        """Reset to the beginning of the buffer."""
        self._bit_position = 0
        self._byte_position = 0
