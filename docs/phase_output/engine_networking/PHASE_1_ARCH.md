# PHASE 1 ARCHITECTURE: Transport and Serialization Foundation

## Phase Overview

Phase 1 establishes the foundational networking primitives: UDP transport with multi-channel reliability and binary serialization with bandwidth-optimized encoding. These components underpin all higher-level networking systems.

---

## 1. Transport Layer Architecture

### 1.1 Packet Structure

```
PacketHeader (12 bytes, big-endian):
+----------------+----------------+----------------+----------------+
| packet_type(1) | flags(1)       | sequence(2)    | ack(2)         |
+----------------+----------------+----------------+----------------+
| ack_bits(4)                     | payload_size(2)                 |
+----------------+----------------+---------------------------------+

Payload: 0-1388 bytes (MTU 1400 - header 12)
```

**Packet Types**:

| Type | Value | Purpose |
|------|-------|---------|
| DATA | 0 | Unreliable data |
| ACK | 1 | Acknowledgment |
| NACK | 2 | Negative acknowledgment |
| CONNECT | 10 | Connection request |
| CONNECT_ACK | 11 | Connection accepted |
| DISCONNECT | 12 | Disconnect request |
| DISCONNECT_ACK | 13 | Disconnect confirmed |
| HEARTBEAT | 20 | Keep-alive ping |
| HEARTBEAT_ACK | 21 | Keep-alive response |
| FRAGMENT | 30 | Fragmented packet |
| FRAGMENT_ACK | 31 | Fragment acknowledgment |
| RELIABLE_DATA | 40 | Reliable data packet |
| SEQUENCED_DATA | 41 | Sequenced data packet |

**Packet Flags**:

| Flag | Value | Behavior |
|------|-------|----------|
| COMPRESSED | 0x01 | Payload is zlib-compressed |
| ENCRYPTED | 0x02 | Payload is encrypted (reserved) |
| FRAGMENTED | 0x04 | Packet is a fragment |
| RELIABLE | 0x08 | Requires acknowledgment |
| ORDERED | 0x10 | Must be delivered in order |
| PRIORITY_HIGH | 0x20 | High priority scheduling |
| PRIORITY_LOW | 0x40 | Low priority scheduling |

### 1.2 Channel System

```
ChannelManager
    |
    +-- Channel[0]: UnreliableChannel
    |       - Fire-and-forget delivery
    |       - No ACK tracking
    |       - No retransmission
    |
    +-- Channel[1]: ReliableOrderedChannel
    |       - ACK-based reliability
    |       - Ordering buffer for gaps
    |       - Retransmission with exponential backoff
    |
    +-- Channel[2]: ReliableChannel (unordered)
    |       - ACK-based reliability
    |       - Immediate delivery on receipt
    |       - Retransmission with exponential backoff
    |
    +-- Channel[3]: SequencedChannel
            - No reliability
            - Drops packets older than last received
            - For frequently-updated state
```

**Reliability Algorithm**:

```python
# Pending packet tracking
_pending[sequence] = PendingPacket(data, send_time, retransmit_time, retransmit_count)

# Retransmit timeout calculation
retransmit_time = now + rtt_estimate * 1.5 * (1.5 ** retransmit_count)

# ACK processing (32-bit bitfield)
for i in range(32):
    if ack_bits & (1 << i):
        seq = (ack - 1 - i) & 0xFFFF
        if seq in _pending:
            _pending.pop(seq)
```

**Ordered Delivery**:

```python
# Buffer out-of-order packets
_order_buffer[sequence] = data

# Deliver when contiguous
while _next_deliver_sequence in _order_buffer:
    deliver(_order_buffer.pop(_next_deliver_sequence))
    _next_deliver_sequence = (_next_deliver_sequence + 1) & 0xFFFF
```

### 1.3 Connection State Machine

```
               CONNECT sent
DISCONNECTED ─────────────────> CONNECTING
      ^                              |
      |                              | CONNECT_ACK received
      |       idle/error             v
      +───────────────────────── CONNECTED
      ^                              |
      | DISCONNECT_ACK received      | disconnect()
      |          or timeout          |
DISCONNECTING <──────────────────────+
```

**Heartbeat System**:

- Server sends HEARTBEAT every `heartbeat_interval` seconds
- Client responds with HEARTBEAT_ACK
- If no response within `heartbeat_timeout`, connection marked failed
- Idle connections (no packets for `idle_timeout`) disconnected

### 1.4 UDP Transport

```
UDPTransport
    |
    +-- _socket: UDP socket (non-blocking)
    |
    +-- _connections: dict[address, Connection]
    |
    +-- Rate limiting
    |       - _packets_this_second
    |       - _bytes_this_second
    |       - Reset each second
    |
    +-- Event generation
            - CONNECTED
            - DISCONNECTED
            - DATA_RECEIVED
            - ERROR
```

**Non-blocking receive loop**:

```python
while True:
    readable, _, _ = select.select([socket], [], [], 0)
    if not readable:
        break
    data, address = socket.recvfrom(MTU)
    route_to_connection(data, address)
```

### 1.5 Quality Monitoring

```
QualityMonitor
    |
    +-- RTT tracking (EWMA smoothing)
    |       rtt = (1-alpha)*rtt + alpha*sample
    |       variance = (1-beta)*variance + beta*|sample-rtt|
    |
    +-- Packet loss tracking
    |       loss = (sent - received) / sent
    |
    +-- Bandwidth estimation
    |       bandwidth = bytes_in_window / window_duration
    |
    +-- Quality level classification
            EXCELLENT < GOOD < FAIR < POOR < CRITICAL
```

**Adaptive Settings**:

| Quality | Update Rate | Compression | Interpolation Delay |
|---------|-------------|-------------|---------------------|
| EXCELLENT | 60 Hz | None | 50ms |
| GOOD | 40 Hz | Low | 75ms |
| FAIR | 30 Hz | Medium | 100ms |
| POOR | 20 Hz | High | 150ms |
| CRITICAL | 10 Hz | Maximum | 250ms |

---

## 2. Serialization Architecture

### 2.1 Bit Packer

```
BitWriter
    |
    +-- write_bits(value, num_bits)  # 1-64 bits
    +-- write_bool(value)            # 1 bit
    +-- write_int(value, min, max)   # Bounded integer (ceil(log2(range)) bits)
    +-- write_float_compressed(value, min, max, precision)  # Quantized float
    +-- write_bytes(data)            # Length-prefixed bytes
    +-- write_string(value)          # UTF-8 with length prefix
    +-- align_to_byte()              # Pad to byte boundary
    +-- to_bytes()                   # Final output

BitReader
    |
    +-- read_bits(num_bits)
    +-- read_bool()
    +-- read_int(min, max)
    +-- read_float_compressed(min, max, precision)
    +-- read_bytes(count)
    +-- read_string()
    +-- peek_bits(num_bits)          # Non-consuming read
    +-- skip_bits(num_bits)
```

**Bounded Integer Encoding**:

```python
def write_int(value, min_value, max_value):
    range_size = max_value - min_value + 1
    bits_needed = range_size.bit_length()
    normalized = value - min_value
    write_bits(normalized, bits_needed)
```

### 2.2 Quantization

**Vector3 Encoding**:

| Precision | Bytes | Resolution (1000 unit range) |
|-----------|-------|------------------------------|
| 8-bit | 3 | 7.84 units |
| 12-bit | 5 | 0.49 units |
| 16-bit | 6 | 0.031 units |
| 24-bit | 9 | 0.00012 units |

**Quaternion Smallest-Three Encoding** (4 bytes total):

```
+----------------+----------------+----------------+----------------+
| dropped_idx(2) | component_a(10)| component_b(10)| component_c(10)|
+----------------+----------------+----------------+----------------+

Components clamped to [-1/sqrt(2), 1/sqrt(2)]
Fourth component reconstructed from unit constraint
Sign determined by convention (positive dropped component)
```

### 2.3 Delta Encoding

```
DeltaEncoder
    |
    +-- set_baseline(sequence, state, timestamp)
    |
    +-- encode_delta(current_state, baseline_seq)
    |       - Compare current to baseline
    |       - Emit only changed fields
    |       - Track removed fields
    |
    +-- decode_delta(delta_bytes, baseline_seq)
    |       - Apply changes to baseline copy
    |       - Remove deleted fields
    |
    +-- acknowledge_baseline(sequence)
            - Prune old baselines
```

**Delta Packet Format**:

```
+----------------+------------------+-------------------+
| compression(1) | changed_count(2) | removed_count(2)  |
+----------------+------------------+-------------------+
| field_1: key(UTF-8) + type(4 bits) + value            |
| field_2: ...                                          |
| removed_key_1(UTF-8)                                  |
+-------------------------------------------------------+
```

**Type Tags** (4 bits):

| Tag | Type |
|-----|------|
| 0 | NULL |
| 1 | BOOL |
| 2-5 | INT8/16/32/64 |
| 6 | FLOAT32 |
| 7 | STRING |
| 8 | BYTES |
| 9 | ARRAY |
| 10 | DICT |

### 2.4 Message Framing

**Message Header** (20 bytes):

```
+-------------+----------+-----------+--------------+-------+---------------+
| msg_type(1) | version(1)| sequence(4)| timestamp(8)| flags(2)| size(4)     |
+-------------+----------+-----------+--------------+-------+---------------+
```

**Message Types**:

| Category | Types |
|----------|-------|
| Connection | CONNECT_REQUEST(1), CONNECT_RESPONSE(2), DISCONNECT(3), HEARTBEAT(4/5) |
| State | FULL_STATE(10), DELTA_STATE(11), STATE_ACK(12) |
| Entity | ENTITY_SPAWN(20), ENTITY_DESPAWN(21), ENTITY_UPDATE(22) |
| RPC | RPC_REQUEST(30), RPC_RESPONSE(31), RPC_ERROR(32) |
| Input | INPUT_STATE(40), INPUT_ACK(41) |
| Custom | 100-255 (application-defined) |

---

## 3. Integration Patterns

### Transport-Serialization Integration

```python
# Sending
message = serialize_message(MESSAGE_TYPE_ENTITY_UPDATE, entity_data)
if len(message) > COMPRESS_THRESHOLD:
    message = compress(message)
    flags |= FLAG_COMPRESSED
connection.send(message, channel_type=RELIABLE_ORDERED)

# Receiving
def on_data_received(data, address):
    if header.flags & FLAG_COMPRESSED:
        data = decompress(data)
    msg_type, payload = deserialize_message(data)
    route_to_handler(msg_type, payload)
```

### Quality-Aware Serialization

```python
# Adapt serialization to network quality
settings = quality_adapter.get_settings(quality_monitor.get_level())

if settings.delta_compression:
    data = delta_encoder.encode_delta(state, baseline_seq)
else:
    data = serialize_full_state(state)

if len(data) > COMPRESS_THRESHOLD and settings.compression_level > 0:
    data = compress(data, level=settings.compression_level)
```

---

## 4. Data Flow Summary

```
Application Layer
       |
       v
+------------------+
| NetSerializer    |  Message framing, type routing
+------------------+
       |
       v
+------------------+
| DeltaEncoder     |  State diffing against baseline
+------------------+
       |
       v
+------------------+
| Quantizer        |  Float/vector/quaternion compression
+------------------+
       |
       v
+------------------+
| BitWriter        |  Bit-level packing
+------------------+
       |
       v
+------------------+
| Channel          |  Reliability, ordering
+------------------+
       |
       v
+------------------+
| PacketFragmenter |  MTU splitting
+------------------+
       |
       v
+------------------+
| UDPTransport     |  Socket I/O
+------------------+
       |
       v
    Network
```
