# Phase 2 Architecture -- Connection Management

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/transport/` (connection.py, quality.py)

---

## Overview

Connection management is embedded within the transport layer files (connection.py, quality.py) rather than living in a separate directory. It provides the connection state machine, quality-of-service monitoring, adaptive bandwidth management, and packet reliability tracking.

---

## Architecture

### Connection State Machine

5 explicit states via `ConnectionState(IntEnum)`:

```
DISCONNECTED (0)
    |-- connect() --> CONNECTING (1)
    |                    |-- CONNECT_ACK received --> CONNECTED (2)
    |                    |-- timeout --> FAILED (4)
    |                    |-- disconnect() --> DISCONNECTING (3)
    v
CONNECTED (2)
    |-- disconnect() --> DISCONNECTING (3)
    |-- idle timeout --> FAILED (4)
    |-- remote disconnect --> DISCONNECTED (0)
    v
DISCONNECTING (3)
    |-- timeout --> DISCONNECTED (0)
    |-- remote ack --> DISCONNECTED (0)
    v
FAILED (4) -- (terminal, must reconnect)
```

**FAILED** state: Distinguishes unexpected disconnection (timeout) from intentional close. Connection must be re-created from DISCONNECTED to retry.

### 4-Way Handshake

```
Client                     Server
  |--- CONNECT ------------->|
  |<-- CONNECT_ACK ----------|
  |--- DATA (implicit) ----->|
  |<-- DATA (implicit) ------|
```

The server accepts on CONNECT, immediately sends CONNECT_ACK, then processes data. The client transitions to CONNECTED upon receiving CONNECT_ACK. Both sides fire `on_connected` callback.

### Heartbeat Protocol

Periodic HEARTBEAT packets (configurable interval, default 1s). The receiver responds with HEARTBEAT_ACK. If no packets (data or heartbeat) received within `idle_timeout`, the connection transitions to FAILED. Heartbeat packets carry ACK info for reliable channel processing.

### Sequence Tracking

- `_local_sequence`: 16-bit counter, incremented per outgoing packet
- `_remote_sequence`: Last received sequence from remote
- `_received_sequences`: Set of received sequence numbers for ACK bitfield generation
- `MAX_RECEIVED_SEQUENCES`: Configurable limit (default 1024), old entries pruned

### Quality Monitoring (QualityMonitor)

**EWMA RTT**: Computed on ACK receipt:
```
rtt = alpha * sample_rtt + (1 - alpha) * rtt
alpha = DEFAULT_CONFIG.RTT_SMOOTHING (default 0.125)
```

**Jitter**: RTT variance via EWMA:
```
variance = beta * |sample_rtt - rtt| + (1 - beta) * variance
beta = DEFAULT_CONFIG.JITTER_SMOOTHING (default 0.25)
```

**Loss Rate**: Sliding window of sent/received counts over configurable window (default 100 packets).

**Bandwidth Tracking**: Bytes sent/received per second.

### Adaptive Quality (NetworkQualityAdapter)

5 `QualityLevel` presets (EXCELLENT/GOOD/FAIR/POOR/CRITICAL). Hysteresis prevents oscillation:

```
Transition up: metrics must meet higher level for N consecutive samples
Transition down: metrics degrade immediately
```

**Bandwidth-Aware Rate Calculation**: Computes max send rate from current bandwidth measurement and target utilization percentage.

**Per-Connection Stats** (ConnectionStats dataclass):
- packets_sent/received/lost/retransmitted
- bytes_sent/received
- rtt / rtt_variance / jitter / packet_loss
- send_bandwidth / receive_bandwidth
- connected_time / last_packet_time / last_heartbeat_time

---

## Missing Components

1. **NAT Traversal**: No STUN client, UDP hole-punching, or TURN relay. Required for peer-to-peer behind consumer NAT.
2. **Cryptographic Challenge**: Handshake is vulnerable to reflection attacks. Standard mitigation: cookie challenge (SCTP-style).
3. **Explicit Reconnect**: No sequence number recovery mechanism for reconnecting clients.

---

## Reality Status

- Connection (5-state, handshake, hearbeats, ACK): **[x]** Complete
- QualityMonitor (EWMA RTT/jitter/loss/bandwidth): **[x]** Complete
- NetworkQualityAdapter (5 presets, hysteresis): **[x]** Complete
- ConnectionStats: **[x]** Complete
- NAT Traversal: **[-]** Not implemented
- Crypto challenge in handshake: **[-]** Not implemented

---

*End of PHASE_2_ARCH.md*
