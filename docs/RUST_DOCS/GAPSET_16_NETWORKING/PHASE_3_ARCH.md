# Phase 3 Architecture -- Serialization

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/serialization/`

---

## Overview

The serialization layer provides a 4-stage pipeline for converting game state into compact network packets: Schema + Type Registration -> Delta Encoding -> Quantization -> Bit Packing -> Compression.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `net_serializer.py` | 513 | Message type registry, 23 MessageTypes, 20-byte header |
| `delta_encoder.py` | 580 | Field-level delta encoding, baseline management, SnapshotDeltaEncoder |
| `quantizer.py` | 437 | Float/vector3/quaternion precision reduction |
| `bit_packer.py` | 442 | Bit-aligned read/write with compression |

---

## Architecture

### Serialization Pipeline

```
[Game State]
    |
    v
NetSerializer -- message type registry, schema dispatch
    |
    v
DeltaEncoder -- compare with baseline, encode only changed fields
    |
    v
Quantizer -- reduce precision (float32 -> 8/12/16/24-bit)
    |
    v
BitWriter -- bit-aligned packing
    |
    v
zlib compress -- deflate compression
    |
    v
[Network Packet]
```

### NetSerializer (net_serializer.py)

**MessageHeader** (20 bytes):
```
Offset  Size  Field
0       2     message_type (uint16)
2       4     sequence (uint32)
6       4     ack (uint32)
10      4     timestamp (uint32, ms)
14      2     payload_size (uint16)
16      4     checksum (uint32)
```

**23 MessageTypes**: Enum covering join/leave, input, game state, entity updates, RPC, chat, voice, system messages. Custom types can be registered via `register_type()`.

**Type Registry**: Mapping from type_id to serialization/deserialization functions. Supports dynamic registration for extensibility.

### DeltaEncoder (delta_encoder.py)

**DeltaSchema**: Field-level schema with 5 field type IDs:
- INT8, INT16, INT32, FLOAT, VECTOR3

**DeltaEncoder**: Compares current field values against baseline. Outputs bitmask of changed fields followed by their values. Uses zlib compression on the delta blob.

**SnapshotDeltaEncoder**: Accumulates changes across multiple ticks, sends full snapshots periodically (every N ticks), delta-only between snapshots. Manages baseline updates.

**Baseline Tracking**: Per-entity baselines stored by NetGUID. Baselines updated on ack or periodic full snapshot.

### Quantizer (quantizer.py)

| Function | Input | Output | Precision |
|----------|-------|--------|-----------|
| `quantize_float` | float32 | 8/12/16/24-bit uint | Configurable |
| `quantize_vector3` | 3 floats | 3 * N-bit uint | Per-component config |
| `quantize_quaternion` | 4 floats | 3 floats (smallest-three) | ~0.01 deg at 16-bit |
| `quantize_angle` | float rad | uint | 8/16-bit wrap |
| `unit_float` | [0,1] | uint | Full range |
| `signed_unit_float` | [-1,1] | uint | Full range |

**Quaternion Quantization**: Uses smallest-three representation: drops the largest component, encodes sign bit + 3 components at reduced precision. Reconstruction infers the dropped component.

### BitPacker (bit_packer.py)

**BitWriter**: Writes bits MSB-first into a byte buffer. Supports:
- `write_bits(value, bits)` -- arbitrary bit width (1-32)
- `write_int(value, bits)` -- signed integer with zigzag encoding
- `write_float_compressed(value, min, max, bits)` -- ranged float
- `write_bytes(data)` -- length-prefixed byte array
- `write_string(s)` -- length-prefixed UTF-8

**BitReader**: Symmetric read operations. Supports:
- `read_bits(bits)`, `read_int(bits)`, `read_float_compressed(min, max, bits)`
- `read_bytes()`, `read_string()`
- `peek_bits(bits)` -- read without advancing
- `skip(bits)` -- advance without reading

Both use zlib compression on the final buffer (compress on write, decompress on read).

---

## Missing Components

1. **Dedicated test file**: No tests for serialization despite 1,972 lines of code.
2. **LZ4 compression**: TODO specifies LZ4 but code uses zlib. Acceptable trade-off unless performance profiling indicates bottleneck.
3. **Schema versioning**: No protocol version negotiation or backwards compatibility mechanism.

---

## Reality Status

- NetSerializer (23 types, registry, 20-byte header): **[x]** Complete
- DeltaEncoder (schemas, baselines, SnapshotDeltaEncoder): **[x]** Complete
- Quantizer (float/vec3/quaternion): **[x]** Complete
- BitPacker (read/write with compression): **[x]** Complete
- Tests: **[-]** Not implemented

---

*End of PHASE_3_ARCH.md*
