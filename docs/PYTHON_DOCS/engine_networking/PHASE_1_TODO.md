# PHASE 1 TODO: Transport and Serialization Foundation

## Overview

Phase 1 implements the foundational networking primitives. All tasks assume the existing implementation is production-ready; these TODOs focus on testing, verification, and identified gaps.

---

## 1. Transport Layer Tasks

### 1.1 Unit Tests: Packet Primitives

**File**: `tests/blackbox_packet.py`

**Acceptance Criteria**:
- [ ] PacketHeader serialization roundtrip preserves all fields
- [ ] Packet creation with all PacketType values succeeds
- [ ] Sequence wraparound comparison returns correct results:
  - `sequence_greater_than(100, 50)` returns True
  - `sequence_greater_than(50, 100)` returns False
  - `sequence_greater_than(10, 65530)` returns True (wraparound)
  - `sequence_greater_than(65530, 10)` returns False (wraparound)
- [ ] FragmentHeader serialization roundtrip preserves fields
- [ ] Fragmentation splits payload at MTU boundary
- [ ] Reassembly produces original payload from all fragments
- [ ] Reassembly handles out-of-order fragment arrival

---

### 1.2 Unit Tests: Channel Types

**File**: `tests/blackbox_channel.py`

**Acceptance Criteria**:
- [ ] UnreliableChannel `send()` returns packet with correct sequence
- [ ] UnreliableChannel `receive()` delivers payload immediately
- [ ] ReliableChannel tracks pending packets after send
- [ ] ReliableChannel removes pending packet on ACK
- [ ] ReliableChannel retransmits after timeout
- [ ] ReliableChannel exponential backoff increases timeout per retry
- [ ] ReliableChannel deduplicates received packets
- [ ] ReliableOrderedChannel buffers out-of-order packets
- [ ] ReliableOrderedChannel delivers in sequence order
- [ ] SequencedChannel drops packets older than last received
- [ ] SequencedChannel delivers packets newer than last received
- [ ] ChannelManager creates channels by ID and type
- [ ] ChannelManager aggregates statistics across channels

---

### 1.3 Unit Tests: Connection State Machine

**File**: `tests/blackbox_connection.py`

**Acceptance Criteria**:
- [ ] Connection starts in DISCONNECTED state
- [ ] `connect()` transitions to CONNECTING
- [ ] Receiving CONNECT_ACK transitions to CONNECTED
- [ ] `disconnect()` from CONNECTED transitions to DISCONNECTING
- [ ] Timeout in DISCONNECTING transitions to DISCONNECTED
- [ ] Heartbeat sent after heartbeat_interval seconds
- [ ] Missing heartbeat ACK within timeout marks FAILED
- [ ] Idle timeout (no packets) marks FAILED
- [ ] `send()` routes to correct channel by type
- [ ] ACK piggyback info added to outgoing packets

---

### 1.4 Unit Tests: UDP Transport

**File**: `tests/blackbox_udp_transport.py`

**Acceptance Criteria**:
- [ ] `bind()` creates socket with SO_REUSEADDR
- [ ] `bind()` sets socket buffer sizes from config
- [ ] `bind()` sets non-blocking mode when configured
- [ ] `connect()` creates Connection and sends CONNECT
- [ ] `connect()` rejects when at max_connections
- [ ] `send()` routes through Connection channel system
- [ ] `broadcast()` sends to all connections
- [ ] `update()` receives pending packets via select
- [ ] `update()` routes packets to correct Connection
- [ ] `update()` generates CONNECTED event on new connection
- [ ] `update()` generates DISCONNECTED event on connection close
- [ ] Rate limiting rejects sends beyond packets_per_second
- [ ] Rate limiting rejects sends beyond bytes_per_second

---

### 1.5 Unit Tests: Quality Monitoring

**File**: `tests/blackbox_quality.py`

**Acceptance Criteria**:
- [ ] RTT smoothing converges toward sample mean
- [ ] Variance tracking reflects sample spread
- [ ] Packet loss calculation matches (sent - received) / sent
- [ ] Bandwidth estimation sums bytes over window
- [ ] Quality level EXCELLENT for low RTT and low loss
- [ ] Quality level CRITICAL for high RTT or high loss
- [ ] Quality change callback invoked on level transition
- [ ] AdaptiveSettings update_rate decreases with worse quality
- [ ] AdaptiveSettings compression_level increases with worse quality
- [ ] Hysteresis prevents rapid oscillation between levels

---

### 1.6 Gap: Implement Encryption

**File**: `engine/networking/transport/encryption.py` (new)

**Background**: `PacketFlags.ENCRYPTED` is defined but not implemented.

**Acceptance Criteria**:
- [ ] AES-GCM encryption for packet payload
- [ ] Key exchange via ECDH during CONNECT handshake
- [ ] Nonce derived from sequence number (prevents replay)
- [ ] Authentication tag validates integrity
- [ ] Decrypt failure logs warning and drops packet
- [ ] Performance: < 5% overhead on 1KB payloads

---

### 1.7 Gap: Implement Compression

**File**: `engine/networking/transport/packet.py` (modify)

**Background**: `PacketFlags.COMPRESSED` is defined but compression not implemented.

**Acceptance Criteria**:
- [ ] zlib compression for payloads > COMPRESS_THRESHOLD
- [ ] Compression level from AdaptiveSettings
- [ ] FLAG_COMPRESSED set when compression applied
- [ ] Decompression on receive when FLAG_COMPRESSED set
- [ ] Skip compression for already-compressed payload

---

## 2. Serialization Layer Tasks

### 2.1 Unit Tests: Bit Packer

**File**: `tests/blackbox_bit_packer.py`

**Acceptance Criteria**:
- [ ] write_bits/read_bits roundtrip for 1-64 bit values
- [ ] write_bool/read_bool roundtrip for True/False
- [ ] write_int/read_int bounded encoding uses minimum bits
- [ ] write_float_compressed/read_float_compressed within precision
- [ ] write_bytes/read_bytes roundtrip for empty and large data
- [ ] write_string/read_string roundtrip for ASCII and Unicode
- [ ] align_to_byte pads to next byte boundary
- [ ] peek_bits does not advance read position
- [ ] skip_bits advances read position without returning data
- [ ] Buffer expands automatically for large writes

---

### 2.2 Unit Tests: Quantization

**File**: `tests/blackbox_quantizer.py`

**Acceptance Criteria**:
- [ ] quantize_float/dequantize_float roundtrip within precision
- [ ] quantize_vector3/dequantize_vector3 roundtrip for all precisions (8/12/16/24)
- [ ] Vector3 at range boundaries quantizes correctly
- [ ] quantize_quaternion/dequantize_quaternion roundtrip within tolerance
- [ ] Quaternion normalization maintains unit length
- [ ] Smallest-three selects correct component to drop
- [ ] quantize_angle/dequantize_angle roundtrip for 0-360 range
- [ ] Unit float quantization maps [0,1] to [0, 2^bits-1]
- [ ] Signed unit float quantization maps [-1,1] to [0, 2^bits-1]

---

### 2.3 Unit Tests: Delta Encoding

**File**: `tests/blackbox_delta_encoder.py`

**Acceptance Criteria**:
- [ ] set_baseline stores state by sequence
- [ ] encode_delta with no changes produces empty delta
- [ ] encode_delta detects added fields
- [ ] encode_delta detects changed fields
- [ ] encode_delta detects removed fields
- [ ] decode_delta reconstructs original state
- [ ] acknowledge_baseline removes old baselines
- [ ] MAX_BASELINES limit enforced
- [ ] Type tags serialize correctly for all supported types
- [ ] Nested dict/array fields encode recursively
- [ ] Compression applied when delta > COMPRESS_THRESHOLD

---

### 2.4 Unit Tests: Message Framing

**File**: `tests/blackbox_net_serializer.py`

**Acceptance Criteria**:
- [ ] serialize/deserialize roundtrip for all MessageType values
- [ ] Header fields preserved: type, version, sequence, timestamp, flags, size
- [ ] FLAG_COMPRESSED applied when payload > threshold
- [ ] Custom encoder/decoder registration works
- [ ] Value types serialize correctly (NULL, BOOL, INT*, FLOAT, STRING, BYTES, ARRAY, DICT)
- [ ] Nested structures serialize recursively
- [ ] Invalid message type raises ValueError
- [ ] Truncated message raises struct.error

---

### 2.5 Gap: Schema Versioning

**File**: `engine/networking/serialization/delta_encoder.py` (modify)

**Background**: Delta encoding has no schema version for forward compatibility.

**Acceptance Criteria**:
- [ ] Schema version byte prepended to delta packets
- [ ] Version mismatch triggers full state resend request
- [ ] Old clients can decode new schema (ignore unknown fields)
- [ ] New clients can decode old schema (use defaults for missing fields)

---

## 3. Integration Tests

### 3.1 Client-Server Connection

**File**: `tests/integration_transport.py`

**Acceptance Criteria**:
- [ ] Server binds and accepts client connection
- [ ] Client connects and receives CONNECT_ACK
- [ ] Bidirectional unreliable data transmission works
- [ ] Bidirectional reliable data transmission works
- [ ] Graceful disconnect from client side
- [ ] Graceful disconnect from server side
- [ ] Connection timeout when server unreachable
- [ ] Heartbeat keeps idle connection alive

---

### 3.2 Reliable Delivery Under Loss

**File**: `tests/integration_reliability.py`

**Acceptance Criteria**:
- [ ] 10% packet loss: all reliable messages delivered
- [ ] 30% packet loss: all reliable messages delivered
- [ ] Messages delivered in order for ordered channel
- [ ] Retransmission count matches expected for loss rate
- [ ] RTT estimate converges despite loss

---

### 3.3 Fragmented Payload Delivery

**File**: `tests/integration_fragmentation.py`

**Acceptance Criteria**:
- [ ] Payload 2x MTU fragments into 2 packets
- [ ] Payload 10x MTU fragments into 10 packets
- [ ] Out-of-order fragment arrival reassembles correctly
- [ ] Single lost fragment retransmitted and reassembled
- [ ] Fragment timeout cleans up incomplete assemblies

---

### 3.4 Serialization Roundtrip

**File**: `tests/integration_serialization.py`

**Acceptance Criteria**:
- [ ] Entity state serializes, transmits, deserializes correctly
- [ ] Delta-compressed state smaller than full state
- [ ] Quantized positions within tolerance on receive
- [ ] Quantized rotations within tolerance on receive
- [ ] Mixed reliability channels deliver data correctly

---

## 4. Performance Tasks

### 4.1 Benchmark: Packet Throughput

**File**: `benchmarks/transport_throughput.py`

**Acceptance Criteria**:
- [ ] Unreliable channel: > 10,000 packets/second
- [ ] Reliable channel: > 5,000 packets/second
- [ ] Reliable ordered channel: > 3,000 packets/second
- [ ] Fragmented packets: > 1,000 reassemblies/second

---

### 4.2 Benchmark: Serialization Speed

**File**: `benchmarks/serialization_speed.py`

**Acceptance Criteria**:
- [ ] Bit packing: > 100,000 write/read cycles per second
- [ ] Quantization: > 50,000 vector/quaternion roundtrips per second
- [ ] Delta encoding: > 10,000 encode/decode cycles per second
- [ ] Message framing: > 20,000 serialize/deserialize per second

---

### 4.3 Profile Hot Paths

**Acceptance Criteria**:
- [ ] Profile packet send/receive path
- [ ] Profile serialization/deserialization
- [ ] Identify allocations in hot paths
- [ ] Document optimization opportunities for Cython

---

## 5. Documentation Tasks

### 5.1 API Documentation

**Acceptance Criteria**:
- [ ] All public classes have docstrings with Args/Returns/Raises
- [ ] Usage examples for UDPTransport (client and server)
- [ ] Usage examples for ChannelManager
- [ ] Usage examples for NetSerializer
- [ ] Usage examples for DeltaEncoder

---

### 5.2 Architecture Diagram

**Acceptance Criteria**:
- [ ] Data flow diagram from application to network
- [ ] State machine diagram for Connection
- [ ] Reliability algorithm flowchart
- [ ] Serialization layer stack diagram
