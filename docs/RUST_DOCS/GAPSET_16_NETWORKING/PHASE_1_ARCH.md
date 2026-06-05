# Phase 1 Architecture -- Transport Foundation

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/transport/`

---

## Overview

The transport layer provides low-level network I/O over UDP, packet encoding/decoding, channel-based reliability semantics, connection state management, and adaptive quality monitoring. Five Python files implement the core transport stack: `udp_transport.py`, `connection.py`, `channel.py`, `packet.py`, and `quality.py`.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `udp_transport.py` | 534 | UDP socket management, packet routing, rate limiting |
| `connection.py` | 532 | 5-state connection machine, handshake, heartbeats, ACK processing |
| `channel.py` | 578 | 5 channel types with reliability/ordering semantics |
| `packet.py` | 437 | 12-byte header, 12 packet types, 7 flags, fragmentation |
| `quality.py` | 518 | EWMA RTT/jitter/loss, 5-level adaptive quality |
| `__init__.py` | 69 | Public API exports |

---

## Architecture

### UDPTransport (udp_transport.py)

Single-threaded, non-blocking UDP socket manager using `select.select()` for readable checks. Routes received packets to `Connection` objects keyed by `(host, port)` tuple.

**Key Design Decisions**:
- Uses `select.select()` with 0 timeout rather than `setblocking(True)` to avoid blocking the main loop
- Rate limiting via per-second packet/byte counters reset on interval
- Connection acceptance generates `CONNECT_ACK` response automatically
- Callback-based for `on_connect`, `on_disconnect`, `on_data` events returned via event list

**Flow**:
```
bind() -> update(dt) loop:
  1. _receive_packets() -> select() -> recvfrom() -> parse Packet -> _route_packet() -> Connection.receive()
  2. For each Connection: conn.update(dt) -> send pending packets
  3. Clean up disconnected/failed connections
```

### Connection (connection.py)

5-state finite state machine:

```
DISCONNECTED --connect()--> CONNECTING --CONNECT_ACK--> CONNECTED
CONNECTED --disconnect()--> DISCONNECTING --timeout--> DISCONNECTED
CONNECTING --timeout--> FAILED
CONNECTED --idle timeout--> FAILED
```

**Handshake**: Simple 2-packet exchange (CONNECT / CONNECT_ACK). No cryptographic challenge. No cookie/anti-spoof.

**Heartbeats**: Periodic heartbeat packets with ACK tracking. Heartbeat interval and timeout configurable.

**ACK Processing**: Every outgoing packet carries `ack` (last received sequence) and 32-bit `ack_bits` bitfield. Reliable channels decode this to clear retransmission queues.

**Channel Ownership**: Each Connection creates 4 default channels:
- Channel 0: UNRELIABLE (frequent state updates)
- Channel 1: RELIABLE_ORDERED (important game events)
- Channel 2: RELIABLE_UNORDERED (less critical reliable data)
- Channel 3: SEQUENCED (latest-only state)

### Channel (channel.py)

5 concrete channel types:

| Channel | Delivery | Ordering | Duplicates | Use Case |
|---------|----------|----------|------------|----------|
| UnreliableChannel | Best-effort | None | Possible | Position updates, non-critical state |
| ReliableChannel | Guaranteed | None | Filtered | Individual reliable messages |
| ReliableOrderedChannel | Guaranteed | Strict | Filtered | Game events, chat messages |
| SequencedChannel | Best-effort | Latest only | Filtered | Health, ammo, timers |

**Reliability Mechanism**: Sequence number tracking with ACK bitfield. Unacknowledged packets are stored in a send queue and retransmitted on timeout. Received packets are deduplicated via received sequence set.

### Packet (packet.py)

**PacketHeader** (12 bytes):
```
Offset  Size  Field
0       2     magic (0x5452 "TR")
2       2     sequence (16-bit)
4       2     ack (16-bit)
6       4     ack_bits (32-bit)
10      1     flags bitmask
11      1     packet_type
12+     n     payload
```

**12 Packet Types**: CONNECT, CONNECT_ACK, DISCONNECT, DISCONNECT_ACK, HEARTBEAT, HEARTBEAT_ACK, DATA, RELIABLE_DATA, SEQUENCED_DATA, ACK, FRAGMENT, CUSTOM.

**7 Packet Flags**: RELIABLE, ORDERED, SEQUENCED, FRAGMENT, HEARTBEAT, ACK, CUSTOM_BIT.

**Fragmentation**: Fragmenter splits packets exceeding MTU (1400 bytes) into FRAGMENT packets reassembled by ID and total count.

### Quality (quality.py)

**QualityMetrics**: Tracks RTT (EWMA with configurable alpha), jitter (RTT variance), loss rate (sliding window), bandwidth (bytes/sec).

**QualityMonitor**: Thread-safe monitor that records send/receive events, computes metrics on demand. Uses configurable window sizes and EWMA smoothing factors.

**NetworkQualityAdapter**: Maps metrics to 5 `QualityLevel` values:

| Level | Max Send Rate | Update Freq | Skip Non-Essential |
|-------|--------------|-------------|-------------------|
| EXCELLENT | Unlimited | 60 Hz | No |
| GOOD | 75% | 45 Hz | No |
| FAIR | 50% | 30 Hz | Minor |
| POOR | 25% | 15 Hz | Yes |
| CRITICAL | 10% | 5 Hz | Yes |

Hysteresis prevents oscillation between levels. Bandwidth-aware rate calculation uses current bandwidth measurement.

---

## Missing Components

1. **TCP Transport** (`transport/tcp_transport.py`): Not implemented. Would provide stream-oriented transport for asset downloads, REST API, and non-realtime communication.
2. **NAT Traversal**: No STUN client, hole-punching logic, or TURN relay integration. Peer-to-peer connections behind NAT will fail.
3. **Cryptographic Challenge**: Handshake lacks DTLS or challenge-response. Vulnerable to spoofed CONNECT packets.

---

## Reality Status

- UDPTransport: **[x]** Complete and functional
- Connection: **[x]** Complete with 5-state machine
- Channel: **[x]** Complete with 5 channel types
- Packet: **[x]** Complete with fragmentation
- Quality: **[x]** Complete with 5-level adaptation
- TCP Transport: **[-]** Not implemented
- ProtocolMeta: **[-]** Not integrated

---

*End of PHASE_1_ARCH.md*
