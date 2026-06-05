# Engine Networking Transport Investigation

**Path**: `engine/networking/transport/`
**Classification**: REAL (fully implemented)
**Total Lines**: 2,663

## Summary

The transport layer is a **fully implemented** UDP networking stack designed for real-time game networking. It provides packet handling, multi-channel reliability guarantees, connection management, and adaptive quality-of-service (QoS). All components contain complete implementations with proper algorithms for RTT estimation, packet fragmentation/reassembly, sequence number wraparound handling, and network quality adaptation.

---

## File Analysis

### 1. `__init__.py` (69 lines) - REAL

**Purpose**: Module initialization and public API exports.

**Exports**:
- Packet primitives: `Packet`, `PacketHeader`, `PacketType`, `MTU`, `MAX_PAYLOAD_SIZE`
- Channel types: `Channel`, `ChannelType`, `ReliableChannel`, `UnreliableChannel`, `SequencedChannel`, `ReliableOrderedChannel`
- Connection: `Connection`, `ConnectionState`, `ConnectionConfig`, `ConnectionStats`
- Transport: `UDPTransport`, `TransportConfig`, `TransportStats`
- Quality: `QualityLevel`, `QualityMetrics`, `QualityMonitor`, `NetworkQualityAdapter`

---

### 2. `packet.py` (436 lines) - REAL

**Purpose**: Network packet structure and fragmentation.

#### Packet Types (PacketType enum)

| Type | Value | Description |
|------|-------|-------------|
| DATA | 0 | Unreliable data |
| ACK | 1 | Acknowledgment |
| NACK | 2 | Negative acknowledgment |
| CONNECT | 10 | Connection request |
| CONNECT_ACK | 11 | Connection accepted |
| DISCONNECT | 12 | Disconnect request |
| DISCONNECT_ACK | 13 | Disconnect confirmation |
| HEARTBEAT | 20 | Keep-alive ping |
| HEARTBEAT_ACK | 21 | Keep-alive response |
| FRAGMENT | 30 | Fragment packet |
| FRAGMENT_ACK | 31 | Fragment acknowledgment |
| RELIABLE_DATA | 40 | Reliable data packet |
| SEQUENCED_DATA | 41 | Sequenced (latest-only) data |

#### Packet Flags (PacketFlags enum)

| Flag | Value | Description |
|------|-------|-------------|
| NONE | 0x00 | No flags |
| COMPRESSED | 0x01 | Payload is compressed |
| ENCRYPTED | 0x02 | Payload is encrypted |
| FRAGMENTED | 0x04 | Packet is a fragment |
| RELIABLE | 0x08 | Requires acknowledgment |
| ORDERED | 0x10 | Must be delivered in order |
| PRIORITY_HIGH | 0x20 | High priority |
| PRIORITY_LOW | 0x40 | Low priority |

#### PacketHeader (12 bytes)

```
Format (big-endian):
  - packet_type: 1 byte
  - flags: 1 byte
  - sequence: 2 bytes (0-65535)
  - ack: 2 bytes (last received sequence from remote)
  - ack_bits: 4 bytes (bitfield acknowledging 32 previous packets)
  - size: 2 bytes (payload size)
```

#### Key Classes

**PacketHeader**: Header serialization/deserialization with `to_bytes()` and `from_bytes()`.

**Packet**: Full packet with header + payload.
- Factory methods: `create()`, `create_ack()`, `create_heartbeat()`
- Properties: `total_size`, `is_reliable()`, `is_fragmented()`

**FragmentHeader** (4 bytes):
- `fragment_id`: 2 bytes - unique group ID
- `fragment_index`: 1 byte - index in group (0-255)
- `fragment_total`: 1 byte - total fragments (1-255)

**PacketFragmenter**: Handles large payload splitting and reassembly.
- `fragment(payload, sequence)` - splits payload exceeding MTU
- `add_fragment(packet)` - accumulates fragments, returns complete payload when all received
- Stores pending fragments in `_pending_fragments[frag_id][frag_idx]`

#### Utility Functions

```python
sequence_greater_than(s1, s2, max_value=65535)  # Wraparound-aware comparison
sequence_difference(s1, s2, max_value=65535)    # Signed difference with wraparound
```

---

### 3. `channel.py` (577 lines) - REAL

**Purpose**: Multi-channel reliability layer with different delivery guarantees.

#### Channel Types (ChannelType enum)

| Type | Value | Guarantee |
|------|-------|-----------|
| UNRELIABLE | 0 | Fire and forget |
| RELIABLE_UNORDERED | 1 | Guaranteed delivery, any order |
| RELIABLE_ORDERED | 2 | Guaranteed delivery, in order |
| SEQUENCED | 3 | Latest only, drops old |

#### Channel Implementations

**UnreliableChannel**:
- No reliability, no ordering
- `send()`: Creates DATA packet with sequence number
- `receive()`: Immediately delivers payload
- No ACK processing or retransmission

**ReliableChannel** (RELIABLE_UNORDERED):
- ACK-based reliability with retransmission
- RTT estimation using EWMA: `rtt = (1-alpha)*rtt + alpha*sample`
- Retransmit timeout: `rtt * 1.5 * (1.5^retransmit_count)`
- Tracks pending packets in `_pending[sequence]`
- Duplicate detection via `_received` set
- Handles fragmented packets via `PacketFragmenter`

**ReliableOrderedChannel** (extends ReliableChannel):
- Adds ordering buffer `_order_buffer[sequence] = data`
- Tracks `_next_deliver_sequence`
- Buffers out-of-order packets until gaps are filled
- `_deliver_ordered()` delivers contiguous packets in sequence

**SequencedChannel**:
- No reliability, but drops old packets
- Tracks `_last_received_sequence`
- Only delivers if `sequence_greater_than(seq, last_received)`
- Ideal for frequently-updated state (positions, etc.)

#### ACK System

The reliable channels implement a sliding-window ACK system:
- `ack`: Latest received sequence number
- `ack_bits`: 32-bit bitfield acknowledging previous 32 packets
- `get_ack_data()` builds (ack, ack_bits) for outgoing packets
- `process_ack()` removes acknowledged packets from pending

#### ChannelManager

Manages multiple channels per connection:
- Factory method `create_channel(id, type, config)`
- Lookup by ID or type
- Aggregate statistics across all channels
- `update(dt)` collects retransmit packets from all channels

---

### 4. `connection.py` (531 lines) - REAL

**Purpose**: Connection state machine and per-connection management.

#### Connection States (ConnectionState enum)

| State | Value | Description |
|-------|-------|-------------|
| DISCONNECTED | 0 | Not connected |
| CONNECTING | 1 | Handshake in progress |
| CONNECTED | 2 | Fully connected |
| DISCONNECTING | 3 | Graceful shutdown |
| FAILED | 4 | Connection failed |

#### ConnectionConfig

| Setting | Default Source | Description |
|---------|---------------|-------------|
| connect_timeout | DEFAULT_CONFIG | Time to wait for CONNECT_ACK |
| disconnect_timeout | DEFAULT_CONFIG | Time to wait for DISCONNECT_ACK |
| idle_timeout | DEFAULT_CONFIG | Disconnect after no packets |
| heartbeat_interval | DEFAULT_CONFIG | Seconds between heartbeats |
| heartbeat_timeout | DEFAULT_CONFIG | Heartbeat ACK timeout |
| max_pending_packets | DEFAULT_CONFIG | Max unacked packets |
| max_retries | DEFAULT_CONFIG | Retransmit attempts |
| default_channels | True | Create default 4 channels |

#### Default Channels

| Channel ID | Type | Purpose |
|------------|------|---------|
| 0 | UNRELIABLE | Frequent updates (positions) |
| 1 | RELIABLE_ORDERED | Important game events |
| 2 | RELIABLE_UNORDERED | Less critical reliable data |
| 3 | SEQUENCED | State that only needs latest |

#### Connection Flow

**Client initiates**:
```
DISCONNECTED -> connect() -> CONNECTING (sends CONNECT)
              -> receive(CONNECT_ACK) -> CONNECTED
              -> update() timeout -> FAILED
```

**Server accepts**:
```
DISCONNECTED -> receive(CONNECT) -> CONNECTED (sends CONNECT_ACK)
```

**Disconnect**:
```
CONNECTED -> disconnect(reason) -> DISCONNECTING (sends DISCONNECT)
          -> update() timeout -> DISCONNECTED
```

#### Key Methods

- `send(data, channel_type)`: Routes to appropriate channel, adds ACK info
- `receive(packet)`: State machine handling + channel routing
- `update(dt)`: Timeouts, heartbeats, channel retransmissions
- `_add_ack_info(packet)`: Piggybacks ACK data on outgoing packets
- `_process_ack_info(packet)`: Notifies reliable channels of ACKs

#### Statistics Tracking (ConnectionStats)

- `packets_sent`, `packets_received`, `packets_lost`, `packets_retransmitted`
- `bytes_sent`, `bytes_received`
- `rtt`, `rtt_variance`, `jitter`, `packet_loss`
- `send_bandwidth`, `receive_bandwidth`
- `connected_time`, `last_packet_time`, `last_heartbeat_time`

---

### 5. `udp_transport.py` (533 lines) - REAL

**Purpose**: Non-blocking UDP socket wrapper with connection management.

#### TransportConfig

| Setting | Default Source | Description |
|---------|---------------|-------------|
| receive_buffer_size | DEFAULT_CONFIG | Socket SO_RCVBUF |
| send_buffer_size | DEFAULT_CONFIG | Socket SO_SNDBUF |
| non_blocking | True | Non-blocking socket mode |
| max_connections | DEFAULT_CONFIG | Maximum concurrent connections |
| connection_config | ConnectionConfig() | Per-connection settings |
| max_packets_per_second | DEFAULT_CONFIG | Rate limit |
| max_bytes_per_second | DEFAULT_CONFIG | Bandwidth limit |

#### UDPTransport Class

**Server Usage**:
```python
server = UDPTransport()
server.bind("0.0.0.0", 12345)

while running:
    events = server.update(0.016)  # 60 FPS
    for event in events:
        if event.event_type == TransportEvent.DATA_RECEIVED:
            process(event.data)
```

**Client Usage**:
```python
client = UDPTransport()
conn = client.connect("server.example.com", 12345)

while running:
    client.send(data, server_address, reliable=True)
    events = client.update(0.016)
```

#### Key Methods

**bind(host, port)**:
- Creates UDP socket with SO_REUSEADDR
- Sets buffer sizes and non-blocking mode
- Returns True on success

**connect(host, port)**:
- Creates socket if needed (client mode)
- Creates Connection object
- Sends CONNECT packet
- Returns Connection or None (at limit)

**send(data, address, reliable=False)**:
- Routes through Connection's channel system
- Returns True if sent successfully

**broadcast(data, reliable=False)**:
- Sends to all connected clients
- Returns count of successful sends

**update(dt)**:
- Resets rate limit counters each second
- Receives pending packets via `select()`
- Routes packets to connections
- Updates all connections (heartbeats, retransmits)
- Returns list of TransportEventData

#### Event System

| Event | Trigger |
|-------|---------|
| CONNECTED | New connection accepted |
| DISCONNECTED | Connection closed |
| DATA_RECEIVED | Data ready from connection |
| ERROR | Socket or protocol error |

#### Rate Limiting

- `_packets_this_second`: Counter reset each second
- `_bytes_this_second`: Counter reset each second
- `_send_packet()` checks limits before sending
- Returns False if rate limit exceeded

#### Connection Acceptance (Server)

```python
def _handle_connect_request(packet, address):
    if address in connections:
        # Already connected
        return None
    if len(connections) >= max_connections:
        # Send DISCONNECT "Server full"
        return None
    # Create Connection, send CONNECT_ACK
    return TransportEventData(CONNECTED, address)
```

---

### 6. `quality.py` (517 lines) - REAL

**Purpose**: Network quality monitoring and adaptive settings.

#### Quality Levels (QualityLevel enum)

| Level | RTT Threshold | Loss Threshold |
|-------|---------------|----------------|
| EXCELLENT | < QUALITY_RTT_EXCELLENT | < QUALITY_LOSS_EXCELLENT |
| GOOD | < QUALITY_RTT_GOOD | < QUALITY_LOSS_GOOD |
| FAIR | < QUALITY_RTT_FAIR | < QUALITY_LOSS_FAIR |
| POOR | < QUALITY_RTT_POOR | < QUALITY_LOSS_POOR |
| CRITICAL | >= QUALITY_RTT_POOR | >= QUALITY_LOSS_POOR |

#### QualityMetrics

| Metric | Description |
|--------|-------------|
| rtt | Round-trip time (seconds) |
| rtt_variance | RTT variance for jitter |
| jitter | Network jitter (seconds) |
| packet_loss | Loss ratio (0-1) |
| bandwidth_up | Upload estimate (bytes/sec) |
| bandwidth_down | Download estimate (bytes/sec) |
| timestamp | Last update time |

#### QualityMonitor Class

**RTT Tracking**:
- EWMA smoothing: `rtt = (1-alpha)*rtt + alpha*sample`
- Variance tracking for jitter calculation
- Windowed sample history

**Packet Loss Tracking**:
- Counters: `packets_sent`, `packets_received`, `packets_lost`
- Rolling window for recent loss calculation
- `loss = (sent - received) / sent`

**Bandwidth Estimation**:
- Timestamped byte samples
- Pruned to bandwidth_window duration
- `bandwidth = sum(bytes) / window`

**Quality Change Callbacks**:
```python
monitor.on_quality_change(lambda old, new: print(f"Quality: {old} -> {new}"))
```

#### NetworkQualityAdapter Class

Automatically adjusts settings based on quality level.

**AdaptiveSettings**:

| Setting | Description |
|---------|-------------|
| update_rate | Network updates per second |
| compression_level | zlib level (1-9) |
| delta_compression | Enable delta encoding |
| interpolation_delay | Seconds of interpolation buffer |
| extrapolation_limit | Max extrapolation time |
| packet_aggregation | Bundle small packets |
| priority_queue | Prioritize important data |

**Quality Presets** (from DEFAULT_CONFIG):

| Level | Update Rate | Compression | Interpolation | Extrapolation |
|-------|-------------|-------------|---------------|---------------|
| EXCELLENT | High | Low | Short | Short |
| GOOD | Medium-High | Low-Medium | Medium | Medium |
| FAIR | Medium | Medium | Medium-Long | Medium-Long |
| POOR | Lower | Higher | Long | Long |
| CRITICAL | Lowest | Highest | Very Long | Very Long |

**Hysteresis**:
- `hysteresis_threshold`: Time quality must be stable before changing
- `adaptation_delay`: Minimum time between adaptations
- Prevents rapid oscillation between settings

**Bandwidth-Aware Update Rate**:
```python
max_rate = bandwidth_up / ASSUMED_BYTES_PER_UPDATE
rate = min(base_rate, max_rate)
```

**Interpolation Delay Calculation**:
```python
delay = rtt + jitter * 2
delay = clamp(delay, MIN_INTERPOLATION_DELAY, MAX_INTERPOLATION_DELAY)
```

---

## Architecture Overview

```
UDPTransport
    |
    +-- UDP Socket (non-blocking)
    |
    +-- Connection[] (per remote address)
            |
            +-- ChannelManager
            |       |
            |       +-- UnreliableChannel (ID 0)
            |       +-- ReliableOrderedChannel (ID 1)
            |       +-- ReliableChannel (ID 2)
            |       +-- SequencedChannel (ID 3)
            |
            +-- State Machine (DISCONNECTED -> CONNECTING -> CONNECTED)
            |
            +-- Heartbeat Timer
            |
            +-- Statistics (RTT, loss, bandwidth)

QualityMonitor -> QualityMetrics -> NetworkQualityAdapter -> AdaptiveSettings
```

---

## Key Algorithms

### 1. Sequence Number Wraparound

```python
def sequence_greater_than(s1, s2, max_value=65535):
    half = max_value // 2
    return ((s1 > s2) and (s1 - s2 <= half)) or ((s1 < s2) and (s2 - s1 > half))
```

Handles 16-bit sequence wraparound (0 -> 65535 -> 0).

### 2. RTT Estimation (EWMA)

```python
diff = abs(sample - rtt_estimate)
rtt_variance = (1 - beta) * rtt_variance + beta * diff
rtt_estimate = (1 - alpha) * rtt_estimate + alpha * sample
```

Based on TCP's Jacobson/Karels algorithm for stable RTT estimation.

### 3. Retransmission Timeout

```python
retransmit_time = now + rtt_estimate * (1.5 ** retransmit_count)
```

Exponential backoff with RTT-based initial timeout.

### 4. ACK Bitfield

32-bit field acknowledges packets `ack-1` through `ack-32`:
```python
for i in range(32):
    seq = (ack - 1 - i) & 0xFFFF
    if seq in received_sequences:
        ack_bits |= (1 << i)
```

Enables redundant ACK for better loss recovery.

---

## Implementation Evidence

### Real Socket Handling (udp_transport.py:151-170)
```python
self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._config.receive_buffer_size)
self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._config.send_buffer_size)
if self._config.non_blocking:
    self._socket.setblocking(False)
self._socket.bind((host, port))
```

### Non-blocking Receive with select() (udp_transport.py:356-364)
```python
if self._config.non_blocking:
    readable, _, _ = select.select([self._socket], [], [], 0)
    if not readable:
        break
data, address = self._socket.recvfrom(MTU)
```

### Binary Packet Header (packet.py:97-110)
```python
# Format string for struct packing (big-endian)
_FORMAT = '!BBHHIH'  # 12 bytes total

def to_bytes(self) -> bytes:
    return struct.pack(
        self._FORMAT,
        self.packet_type,
        self.flags,
        self.sequence & 0xFFFF,
        self.ack & 0xFFFF,
        self.ack_bits & 0xFFFFFFFF,
        self.size & 0xFFFF
    )
```

### ACK-based Reliability (channel.py:284-309)
```python
def process_ack(self, ack: int, ack_bits: int) -> List[Packet]:
    acked_sequences = []
    if ack in self._pending:
        acked_sequences.append(ack)
    for i in range(32):
        if ack_bits & (1 << i):
            seq = (ack - 1 - i) & 0xFFFF
            if seq in self._pending:
                acked_sequences.append(seq)
    now = time.time()
    for seq in acked_sequences:
        pending = self._pending.pop(seq, None)
        if pending and pending.retransmit_count == 0:
            rtt = now - pending.send_time
            self._update_rtt(rtt)
```

---

## Integration Points

| Dependency | Location |
|------------|----------|
| `engine.networking.config.DEFAULT_CONFIG` | All modules |
| `engine.networking.prediction` | Uses Connection for client prediction |
| `engine.networking.replication` | Uses Channel for state replication |
| `engine.networking.lag_compensation` | Uses RTT for compensation timing |

---

## Testing Requirements

### Unit Tests

1. **Packet Tests**:
   - Header serialization roundtrip
   - Fragment creation and reassembly
   - Sequence wraparound comparison

2. **Channel Tests**:
   - Unreliable send/receive
   - Reliable ACK/retransmit
   - Ordered delivery buffering
   - Sequenced drop-old behavior

3. **Connection Tests**:
   - State machine transitions
   - Heartbeat timeout detection
   - Multi-channel routing

4. **Transport Tests**:
   - Socket bind/close
   - Client connect/disconnect
   - Rate limiting enforcement

5. **Quality Tests**:
   - RTT smoothing accuracy
   - Packet loss calculation
   - Quality level transitions
   - Adaptation hysteresis

### Integration Tests

1. Client-server connection establishment
2. Reliable message delivery under packet loss
3. Quality degradation response
4. Fragmented large payload delivery

---

## Gaps and Recommendations

### Minor Gaps

1. **No encryption implementation**: `PacketFlags.ENCRYPTED` exists but encryption not implemented
2. **No compression implementation**: `PacketFlags.COMPRESSED` exists but compression not implemented
3. **NACK not used**: `PacketType.NACK` defined but not implemented in reliability layer

### Recommendations

1. **Add DTLS or custom encryption** for secure transport
2. **Add zlib/lz4 compression** using `compression_level` from AdaptiveSettings
3. **Consider selective acknowledgment (SACK)** for better loss recovery
4. **Add congestion control** beyond simple rate limiting
5. **Add IPv6 support** (currently IPv4 only)

---

## Conclusion

The transport layer is **production-ready** with a complete implementation of:
- UDP socket management with non-blocking I/O
- Multi-channel reliability (unreliable, reliable, ordered, sequenced)
- Connection state machine with heartbeats
- Packet fragmentation and reassembly
- RTT-based retransmission timing
- Sliding-window ACK system with bitfield
- Quality monitoring and adaptive settings

No stubs or placeholder implementations were found. This is production-quality game networking code implementing a UDP-based reliable transport similar to ENet or RakNet patterns.
