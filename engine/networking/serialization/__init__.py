"""
Serialization module for network message encoding/decoding.

Provides bit-level packing, numeric quantization, delta compression,
and high-level message serialization for efficient network transmission.
"""

from .bit_packer import BitWriter, BitReader
from .quantizer import (
    quantize_float,
    dequantize_float,
    quantize_vector3,
    dequantize_vector3,
    quantize_quaternion,
    dequantize_quaternion,
)
from .delta_encoder import DeltaEncoder
from .net_serializer import (
    NetSerializer,
    serialize_message,
    deserialize_message,
    MessageType,
)

__all__ = [
    # Bit packing
    "BitWriter",
    "BitReader",
    # Quantization
    "quantize_float",
    "dequantize_float",
    "quantize_vector3",
    "dequantize_vector3",
    "quantize_quaternion",
    "dequantize_quaternion",
    # Delta compression
    "DeltaEncoder",
    # Message serialization
    "NetSerializer",
    "serialize_message",
    "deserialize_message",
    "MessageType",
]
