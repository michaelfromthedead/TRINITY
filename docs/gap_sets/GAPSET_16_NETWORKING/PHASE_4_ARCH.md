# Phase 4 Architecture -- Replication System

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/replication/`

---

## Overview

The replication system is the core of the networking layer, responsible for synchronizing entity state from server to clients. It implements a full replication cycle: collect dirty properties, filter by relevancy, prioritize by bandwidth, serialize, and send.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `replication_manager.py` | 748 | Orchestrator: replication cycle, 4 roles, 4 packet types |
| `net_guid.py` | 399 | 32-bit unique ID allocation with authority tracking |
| `actor_channel.py` | 681 | Per-entity, per-client state streaming with 4 states, 6 message types |
| `relevancy.py` | 572 | 6 strategies including CompositeRelevancy |
| `bandwidth.py` | 580 | 5 priority levels, token bucket, anti-starvation |
| `property_replication.py` | 437 | ReplicatedProperty, 7 conditions, 3 notify modes |

---

## Architecture

### Replication Cycle (ReplicationManager)

Per-frame cycle executed by ReplicationManager:

```
1. Collect dirty properties from tracked entities
2. For each client:
   a. Filter entities by relevancy
   b. Prioritize by bandwidth manager
   c. Serialize through actor channels
   d. Queue for transport
3. Send queued packets
```

**ReplicationRole**: 4 roles controlling behavior:
- `SERVER`: Full authority, broadcasts to all clients
- `CLIENT`: Receives updates, sends inputs only
- `AUTONOMOUS`: Local authority with server reconciliation
- `SIMULATED`: Receives interpolated updates

**Packet Types**: 4 replication-specific packet types for state updates, delta, RPC, and control messages.

### NetGUID (net_guid.py)

**32-bit ID format**:
```
Bit 31:     Authority flag (1 = server-authoritative)
Bits 16-30: Client ID (supports ~32K clients)
Bits 0-15:  Object index (64K objects per client)
```

**NetGUIDManager**: Thread-safe (RLock) allocation and release. Tracks max clients and per-client object counts. Supports explicit registration with custom IDs for well-known objects.

### ActorChannel (actor_channel.py)

Per-entity, per-client communication channel with 4 states:
- `INACTIVE`: Not replicating to this client
- `PENDING_SPAWN`: Waiting for spawn confirmation
- `ACTIVE`: Full replication
- `DORMANT`: Replicating only dormancy updates

**6 Message Types**: Spawn, Despawn, Update, DeltaUpdate, RPC, Control.

**State Management**: Tracks last sent state per property, handles reliable delivery of spawn/despawn, manages pending spawn confirmations.

### Relevancy (relevancy.py)

**6 Strategies**:
| Strategy | Behavior |
|----------|----------|
| Always | Always relevant to all clients |
| Owner | Relevant only to owning client |
| Radius | Relevant within distance from client |
| Grid | Relevant within spatial grid cells |
| Custom | User-provided callback function |
| Composite | Combines multiple strategies (logical AND/OR) |

**RelevancyContext**: Carries spatial position, priority value, visibility state, and connection info for evaluation.

### Bandwidth Management (bandwidth.py)

**5 Priority Levels**: CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND.

**TokenBucket**: Per-connection bandwidth budget with configurable fill rate and burst capacity.

**Priority Queue**: Entities queued by priority level, with round-robin within each level.

**Anti-Starvation**: Background priority entities get minimum bandwidth guarantee (configurable percentage). Prevents complete starvation under load.

**SendQueue**: Bounded queue per connection with drop-from-front on overflow (keeps most recent updates).

### Property Replication (property_replication.py)

**ReplicatedProperty**: Typed descriptor with automatic dirty tracking. Supports int, float, bool, string, Vector3, Quaternion, and custom types.

**ReplicationCondition** (7 conditions, inline enum):
| Condition | Description |
|-----------|-------------|
| ALL | Replicate to all clients |
| OWNER_ONLY | Only to owning client |
| SIMULATED_PROXY | Only to simulated proxies |
| AUTONOMOUS_PROXY | Only to autonomous proxies |
| SERVER_ONLY | Never replicate (server internal) |
| DORMANT | Only replicate when dormant |
| NEVER | Never replicate |

**NotifyMode** (3 modes):
- `ON_CHANGE`: Notify when value changes
- `ON_DEMAND`: Notify when requested
- `ALWAYS`: Notify every replication cycle

---

## Missing Components

1. **Foundation decorator integration**: `@networked`, `@replicated`, `@interest` decorators described in NETWORKING_CONTEXT.md are not wired to this Python layer.
2. **__init__.py**: No explicit package exports file (namespace package works).
3. **Dedicated tests**: No test coverage for any of the 6 files (~3,800 LOC untested).

---

## Reality Status

- ReplicationManager (full cycle, 4 roles): **[x]** Complete
- NetGUID (32-bit, authority bit, thread-safe): **[x]** Complete
- ActorChannel (4 states, 6 message types): **[x]** Complete
- Relevancy (6 strategies + Composite): **[x]** Complete
- Bandwidth (5 priorities, token bucket, anti-starvation): **[x]** Complete
- Property Replication (7 conditions, 3 notify modes): **[x]** Complete
- Foundation decorator integration: **[-]** Not implemented

---

*End of PHASE_4_ARCH.md*
