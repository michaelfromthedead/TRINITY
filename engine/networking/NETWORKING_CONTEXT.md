# NETWORKING_CONTEXT.md — Networking Layer

> **Purpose**: Complete implementation reference for the engine/networking/ layer.  
> Read this file and ONLY this file when implementing networking systems.

---

## 1. Architecture Summary

The networking layer provides multiplayer connectivity: transport, replication, RPCs, prediction, lag compensation, security, and social systems. It bridges Trinity's declarative networking decorators/descriptors to wire protocols.

**Core Subsystems (7):**
1. **Transport** — UDP/TCP/custom protocols, channels, reliability, connection lifecycle
2. **Replication** — Property sync (server→client), relevancy filtering, bandwidth management
3. **RPC** — Remote procedure calls (server/client/multicast), reliability, ordering
4. **Prediction** — Client-side prediction, server reconciliation, entity interpolation
5. **Lag Compensation** — Server-side rewind, hitbox timing, view-time reconstruction
6. **Security** — Server authority validation, anti-cheat, input rate limiting
7. **Social** — Matchmaking, lobbies, parties, voice chat, text chat

**Network Models Supported:**
- **Lockstep** — Deterministic sync, frame-based, identical inputs required
- **Snapshot** — Complete world state snapshots with delta compression, interpolation
- **State Synchronization** — Property-level replication + RPCs + relevancy

**Network Topologies:**
- **Client-Server** — Dedicated servers, listen servers, server authority
- **Peer-to-Peer** — Full mesh, host migration, relay-assisted
- **Hybrid** — Distributed authority, regional servers

**Connection Lifecycle:**
```
DISCONNECTED → CONNECTING → CONNECTED → DISCONNECTING → DISCONNECTED
                  │
                  └─ Handshake: SYN → Challenge → Response → ACK
```

**Per-Frame Network Sync Flow:**
```
1. End of frame → flush()
2. Tracker.all_dirty() → all changed objects
3. Filter for _networked entities
4. pop_network_updates() per entity
5. Serialize via _serialize_net()
6. Prioritize by bandwidth_priority + relevancy
7. Send over transport layer (channels)
```

---

## 2. Decorators

### 2.1 Core Network Decorators

#### @networked (data_flow.py)
```python
@networked(
    relevance: str = "global",    # "global" | "spatial" | "owner"
    authority: str = "server",    # "server" | "client" | "owner"
    priority: int = 1,            # update priority (higher = more important)
    unreliable: bool = False,     # UDP-style delivery
    delta: bool = False,          # delta compression enabled
    predicted: bool = False,      # client-side prediction enabled
    interpolated: str = "none",   # "linear" | "hermite" | "none"
)
```
- **Config Dataclass:** `NetworkedConfig`
- **Steps:** TAG(networked=True), TAG(networked_config), REGISTER(data_flow)
- **After-Apply Sets:** `_networked`, `_networked_relevance`, `_networked_authority`, `_networked_priority`, `_networked_unreliable`, `_networked_delta`, `_networked_predicted`, `_networked_interpolated`
- **Methods Added:** `_serialize_net()`, `_deserialize_net()`

#### @serializable (data_flow.py)
```python
@serializable(
    format: str = "binary",  # "binary" | "json" | "msgpack"
    version: int = 1,        # schema version
)
```
- **Config Dataclass:** `SerializableConfig`
- **Steps:** TAG(serializable=True), TAG(serializable_config), REGISTER(data_flow), DESCRIBE()
- **After-Apply Sets:** `_serializable`, `_serializable_format`, `_serializable_version`, `_serializable_fields`
- **Methods Added:** `serialize()`, `deserialize()` (classmethods)
- **Required by:** @snapshot, @versioned, @networked

#### @snapshot (data_flow.py)
```python
@snapshot(
    history_frames: int = 60,  # frames to keep
)
```
- **Config Dataclass:** `SnapshotConfig`
- **Steps:** TAG(snapshot=True), TAG(snapshot_config), REGISTER(data_flow)
- **After-Apply Sets:** `_snapshot`, `_snapshot_history_frames`
- **Methods Added:** `snapshot_save()`, `snapshot_restore(frame)`
- **Requires:** @serializable

#### @versioned (data_flow.py)
```python
@versioned(
    version: int = 1,
    migrations: dict = {},  # {from_version: migration_fn}
)
```
- **Config Dataclass:** `VersionedConfig`
- **Steps:** TAG(versioned=True), TAG(versioned_config), REGISTER(data_flow), VALIDATE(requires_serializable)
- **After-Apply Sets:** `_versioned`, `_versioned_version`, `_versioned_migrations`
- **Requires:** @serializable

#### @rpc (rpc.py)
```python
@rpc(
    authority: str = "server",  # "server" | "client" | "owner"
    reliable: bool = True,       # guaranteed delivery
)
```
- **Steps:** TAG(rpc=True), TAG(rpc_authority), TAG(rpc_reliable), REGISTER(rpc)
- **After-Apply Sets:** `_rpc`, `_rpc_authority`, `_rpc_reliable`
- **Applies to:** Functions (methods), not classes

### 2.2 Extended Network Decorators

#### @interest (network_extended.py)
```python
@interest(
    type: str = "radius",               # "radius" | "grid" | "custom"
    radius: float = 5000.0,             # for radius type
    always_relevant_to_owner: bool = True,
)
```
- **Config Dataclass:** `InterestConfig`
- **Steps:** TAG(interest=True), TAG(interest_config), REGISTER(network_extended)
- **Purpose:** Relevancy filtering — which clients receive updates

#### @bandwidth_priority (network_extended.py)
```python
@bandwidth_priority(
    priority: int = 50,           # higher = more important
    max_bps: Optional[int] = None, # max bits per second
)
```
- **Config Dataclass:** `BandwidthPriorityConfig`
- **Steps:** TAG(bandwidth_priority=True), TAG(bandwidth_priority_config), REGISTER(network_extended)

#### @snapshot_interpolation (network_extended.py)
```python
@snapshot_interpolation(
    buffer_size_ms: float = 100.0,   # interpolation buffer
    interp_delay_ms: float = 100.0,  # interpolation delay
)
```
- **Config Dataclass:** `SnapshotInterpolationConfig`
- **Steps:** TAG(snapshot_interpolation=True), TAG(snapshot_interpolation_config), REGISTER(network_extended)

#### @server_reconcile (network_extended.py)
```python
@server_reconcile(
    max_reconcile_frames: int = 10,  # frames to reconcile
    snap_threshold: float = 0.5,      # snap vs interpolate threshold
)
```
- **Config Dataclass:** `ServerReconcileConfig`
- **Steps:** TAG(server_reconcile=True), TAG(server_reconcile_config), REGISTER(network_extended)

### 2.3 Supporting Decorators (from other modules)

| Decorator | Module | Network Role |
|-----------|--------|--------------|
| @tracked | debug_safety.py | Dirty flags for replication |
| @diff | bridges_caching.py | Delta compression for bandwidth |
| @batch | bridges_caching.py | Coalesce network updates per frame |
| @throttle_network | bridges_caching.py | Rate-limit network updates |
| @server_authoritative | security.py | Server is source of truth |
| @rate_limited | security.py | Per-player command rate limiting |
| @pooled | lifecycle.py | Object pooling for net spawns |
| @packed | ecs_core.py | Memory layout for net serialization |

---

## 3. Metaclasses

### ProtocolMeta
- **Purpose:** Auto-register network protocols, define wire formats
- **Auto-applies:** TAG, REGISTER into protocol registry
- **Key Methods:**
  - `get_wire_format()` → serialization format for the protocol
  - `get_version()` → protocol version number
  - `validate_message(data)` → validate incoming message structure
- **Usage:** Base metaclass for all network protocol classes

### ComponentMeta (networking-relevant)
- **Purpose:** Components with @networked get auto-registered for replication
- **Network-Relevant Features:**
  - Discovers `_networked` flag on components
  - Registers networked components with ReplicationManager
  - Provides `get_networked_fields()` introspection

### EventMeta (networking-relevant)
- **Purpose:** Events with @rpc get auto-registered for RPC dispatch
- **Network-Relevant Features:**
  - Discovers `_rpc` flag on methods
  - Registers RPC methods with RPCManager
  - Provides `get_rpc_methods()` introspection

---

## 4. Descriptors

### NetworkedDescriptor (networking.py)
```python
NetworkedDescriptor(
    authority: str = "server",    # "server" | "client" | "owner"
    interpolated: bool = False,   # enable interpolation
    priority: int = 1,            # network priority
    update_frequency: int = 0,    # ticks between updates (0 = every change)
)
```
- **post_set():** Queues update → `{field, value, old_value, priority}` into `_network_queue`
- **get_metadata():** Returns `{authority, interpolated, priority, update_frequency}`
- **Accepts Inner:** tracked, observable, validated, range, storage
- **Excludes:** transient, local_only

**Network Queue API:**
```python
get_network_queue(obj)      # Get pending updates list
clear_network_queue(obj)    # Clear the queue
pop_network_updates(obj)    # Get and clear (atomic)
```

### InterpolatedDescriptor (networking.py)
```python
InterpolatedDescriptor(mode: str = "linear")  # "linear" | "hermite"
```
- **post_set():** Buffers snapshots (max `INTERPOLATION_BUFFER_SIZE` = 3)
- **get_interpolated(obj, t):** Returns interpolated value (0..1 between last two snapshots)
- **Modes:** Linear (lerp), Hermite (cubic hermite with zero tangents)
- **Non-Numeric Fallback:** Returns latest value

### PredictedDescriptor (networking.py)
```python
PredictedDescriptor(max_history: int = 30)  # DEFAULT_PREDICTION_HISTORY
```
- **post_set():** Accumulates history entry
- **rollback(obj, frames=1):** Rewinds state, trims history
- **get_history(obj):** Returns history list
- **Ring Buffer:** Limits to `max_history` entries

### ThrottledNetworkDescriptor (networking.py)
```python
ThrottledNetworkDescriptor(max_updates_per_second: float = 20.0)
```
- **post_set():** Token bucket — skips if within `min_interval` (1/rate)
- **has_pending(obj):** Check if throttled update waiting
- **flush(obj):** Force-send pending update
- **Min Interval:** `1.0 / max_updates_per_second`

---

## 5. Foundation Integration Points

### Foundation Tracker → Replication
```
NetworkedDescriptor.post_set()
    → Tracker.mark_dirty(entity, field)
    → _network_queue.append({field, value, old_value, priority})

Per-frame:
    Tracker.all_dirty()
    → filter _networked entities
    → pop_network_updates() per entity
    → serialize + send
```

### Foundation EventLog → Rollback
```
EventLog.record(entity, "set_field", {field, old, new})
    → builds causal chain
    → enables rollback via EventLog.events_for_entity()
```

### Foundation Tracker → Reconciliation
```
Server correction received:
    → Tracker.undo() to rollback to last confirmed state
    → Re-apply buffered inputs
    → PredictedDescriptor.rollback(frames)
```

### Foundation Mirror → Net Serialization
```
Mirror.ObjectMirror(entity)
    → enumerate all networked fields
    → schema_hash() for version validation
    → serialize/deserialize for wire format
```

### Foundation DeltaSync → Snapshot Diffing
```
DeltaSync.diff(snapshot_old, snapshot_new)
    → minimal delta payload
    → delta compression for bandwidth efficiency
```

### Foundation Capabilities → Authority
```
Capabilities.has(player, "server_authority")
    → validate RPC permissions
    → validate write authority on networked fields
```

### Foundation Bridge → Network Object Factory
```
Bridge.spawn_networked(prefab, net_guid)
    → creates entity via Registry
    → assigns net GUID
    → opens actor channel
    → sends initial state
```

---

## 6. Architecture Spec Details

### 6.1 Transport Layer

**Channel System:**
| Channel | Reliability | Ordering | Use Case |
|---------|-------------|----------|----------|
| Unreliable | No | No | Position updates, voice |
| Reliable Ordered | Yes | Yes | RPCs, chat, important state |
| Reliable Unordered | Yes | No | Asset loading, bulk data |
| Sequenced | No | Latest-only | Frequent updates (newest wins) |

**Packet Structure:**
```
┌─────────┬──────────┬──────────┬─────────┬──────────┐
│ Header  │ Sequence │ Ack Mask │ Payload │ Checksum │
│ (type,  │ (16-bit) │ (32-bit) │ (var)   │ (CRC32)  │
│  size)  │          │          │         │          │
└─────────┴──────────┴──────────┴─────────┴──────────┘
```

**Packet Priority:** Critical > High > Normal > Low
**Coalescing:** Multiple messages batched per MTU (~1200 bytes), flushed on timer

**Connection Quality Metrics:**
- RTT (round-trip time)
- Jitter (RTT variance)
- Packet Loss (%)
- Bandwidth (bytes/sec)

**Quality Adaptation:**
- Degrade gracefully: reduce update rate → increase compression → drop low-priority

**NAT Traversal:** STUN → Hole Punching → TURN fallback
**NAT Types:** Open, Moderate (cone), Strict (symmetric)

### 6.2 Replication System

**Object Lifecycle:**
```
Create Object → Assign Net GUID → Send Initial State → Open Actor Channel
                                                          │
                                              ┌───────────┼───────────┐
                                              ▼           ▼           ▼
                                          Replicate   Close Channel  Destroy
                                          Properties  (irrelevant)   (despawn)
```

**Ownership Types:**
| Type | Description | Authority |
|------|-------------|-----------|
| Server-Owned | Server controls fully | Server |
| Client-Owned | Client has authority | Owning client |
| Autonomous Proxy | Local client can predict | Client (predicted) |
| Simulated Proxy | Remote clients see interpolated | Server |

**Replication Conditions:**
| Condition | When Sent |
|-----------|-----------|
| Always | Every update tick |
| On Change | Only when value changes |
| Initial Only | Once on spawn |
| Owner Only | Only to owning client |
| Skip Owner | Everyone except owner |
| Custom | User-defined predicate |

**Change Notification Modes:**
| Mode | Behavior |
|------|----------|
| None | No callback on receive |
| Rep Notify | Call OnRep_PropertyName() |
| With Previous | Call OnRep with old value |

### 6.3 RPC System

**RPC Types:**
| Type | Direction | Reliability | Use Case |
|------|-----------|-------------|----------|
| Server RPC | Client → Server | Reliable | Player actions, requests |
| Client RPC | Server → Client | Reliable/Unreliable | Responses, effects |
| Multicast | Server → All Clients | Reliable/Unreliable | Explosions, announcements |

**RPC Validation:**
1. Check authority (is caller allowed?)
2. Check rate limit (not spamming?)
3. Validate parameters (within bounds?)
4. Execute or reject

### 6.4 Prediction & Reconciliation

**Client-Side Prediction Flow:**
```
1. Client presses input
2. Apply input locally (predict)
3. Store input in buffer with sequence number
4. Send input to server
5. Server processes, sends authoritative state
6. Client receives server state:
   a. If matches prediction → discard buffer entries up to confirmed
   b. If mismatch → rollback to server state, re-apply buffered inputs
```

**Smoothing Methods:**
| Method | When | Behavior |
|--------|------|----------|
| Snap | Error > threshold | Teleport to correct position |
| Interpolate | Error < threshold | Smooth blend over frames |
| Threshold-Based | Always | Choose snap/interp by error magnitude |

**Entity Interpolation (non-predicted entities):**
```
Buffer snapshots (INTERPOLATION_BUFFER_SIZE = 3)
Render at time = now - interp_delay_ms
Interpolate between two nearest snapshots
Extrapolate if no new data (with limit)
```

### 6.5 Lag Compensation

**Server-Side Rewind:**
```
1. Client fires weapon at time T_client
2. Server receives at T_server (T_client + RTT/2)
3. Server rewinds world state to T_client
4. Perform hit detection at rewound positions
5. Apply results at current time
6. Maximum rewind: MAX_LAG_COMPENSATION_MS
```

### 6.6 Serialization

**Binary Format (Primary):**
- Custom binary with bit packing
- Delta compression (only changed fields)
- Quantization: Float→Fixed16, Vec3→compressed, Quat→smallest-three

**Compression Pipeline:**
```
Raw State → Delta (changed only) → Quantize → Bit Pack → LZ4 Compress
```

### 6.7 Anti-Cheat

**Server-Side Validation:**
- Input sanity checks (speed, rate, bounds)
- Physics bounds verification
- Rate limiting per player
- Sequence validation

**Detection Methods:**
- Statistical analysis (abnormal accuracy/reaction time)
- Heuristic rules (impossible movements)
- ML anomaly detection

**Response Tiers:**
| Severity | Response |
|----------|----------|
| Low | Warning |
| Medium | Kick |
| High | Temporary Ban |
| Critical | Permanent Ban + Shadow Ban |

### 6.8 Matchmaking

**Flow:**
```
Queue → Match Criteria → Expand Search → Form Match → Select Server → Connect
```

**Skill Systems:** Elo, Glicko-2, TrueSkill, MMR
**Server Selection:** Ping-based, Regional, Load-balanced

**Party System:** Create → Invite → Accept → Queue as group → Leader controls

### 6.9 Voice & Chat

**Voice Pipeline:**
```
Mic → Voice Activity Detection → Noise Reduction → Opus Encode → Transmit
Receive → Jitter Buffer → Decode → 3D Positional Audio → Speaker
```

**Voice Channels:** Team, Squad, Proximity, Global
**Text Chat:** Global, Team, Whisper, System + profanity filter + moderation

---

## 7. Decorator Stacks

### @networked_entity — Basic Replication
```python
from trinity.decorators.builtin_stacks.network import networked_entity

@networked_entity(authority="server", relevance="spatial", priority=10, pool_size=64)
class Player(Component):
    health: Annotated[int, Tracked(), NetworkedDescriptor(authority="server")]
    position: Annotated[FVec3, Tracked(), NetworkedDescriptor(interpolated=True)]
```
**Combines:** @component + @packed(layout="soa") + @pooled + @networked + @serializable + @track_changes

### @bandwidth_efficient — Bandwidth Optimized
```python
from trinity.decorators.builtin_stacks.network import bandwidth_efficient

@bandwidth_efficient(radius=5000, max_updates_per_second=20.0, priority=50)
class DistantNPC(Component):
    position: Annotated[FVec3, NetworkedDescriptor(priority=5)]
    state: Annotated[int, NetworkedDescriptor()]
```
**Combines:** @networked + @diff + @interest + @bandwidth_priority + @throttle_network + @batch

### @predicted_entity — Client-Side Prediction
```python
from trinity.decorators.builtin_stacks.network import predicted_entity

@predicted_entity(history_frames=30, max_reconcile_frames=10, snap_threshold=0.5)
class LocalPlayer(Component):
    position: Annotated[FVec3, PredictedDescriptor(max_history=30), InterpolatedDescriptor("hermite")]
    velocity: Annotated[FVec3, PredictedDescriptor(max_history=30)]
```
**Combines:** @networked(predicted=True, interpolated="hermite") + @snapshot + @server_reconcile + @diff

### @secure_multiplayer — Anti-Cheat Hardened
```python
from trinity.decorators.builtin_stacks.network import secure_multiplayer

@secure_multiplayer(rate_limit=10)
class SecurePlayerActions(System):
    @rpc(authority="server", reliable=True)
    def request_action(self, action_id: int, params: dict): ...
```
**Combines:** @server_authoritative + @validated + @rate_limited

### Compound Stacks (Composable)
```python
# Predicted + Bandwidth Efficient (for player characters in large worlds)
@predicted_entity(history_frames=30)
@bandwidth_efficient(radius=10000)
class OpenWorldPlayer(Component): ...

# Networked + Secure (for authoritative game objects)
@networked_entity(authority="server")
@secure_multiplayer(rate_limit=20)
class AuthoritativeObject(Component): ...
```

---

## 8. TODO Checklist (from GAME_ENGINE_INTEGRATION_TODO.md S11)

### 11.1 Transport
- [ ] Implement UDP transport with reliability layer
- [ ] Implement channel system (reliable-ordered, reliable-unordered, unreliable, sequenced)
- [ ] Implement connection management (handshake, heartbeat, timeout, reconnect)
- [ ] Wire ProtocolMeta → wire format
- [ ] Implement packet coalescing (MTU-aware batching)
- [ ] Implement connection quality metrics (RTT, jitter, packet loss, bandwidth)
- [ ] Implement quality adaptation (degrade gracefully under poor conditions)
- [ ] Implement NAT traversal (STUN, hole punching, TURN fallback)

### 11.2 Replication
- [ ] Implement property replication (server → client)
- [ ] Wire NetworkedDescriptor dirty flags → replication prioritization
- [ ] Implement relevancy / interest management (radius, grid, custom)
- [ ] Wire @networked decorator → replication config (authority, interpolate, priority)
- [ ] Wire Foundation Tracker.all_dirty() → collect dirty networked fields per frame
- [ ] Wire SerializableDescriptor → network serialization format
- [ ] Implement Net GUID assignment and actor channel management
- [ ] Implement replication conditions (always, on_change, initial_only, owner_only, skip_owner)
- [ ] Implement change notification modes (none, rep_notify, with_previous)
- [ ] Implement bandwidth management (per-actor limits, saturation handling)

### 11.3 RPCs
- [ ] Implement RPC system (client→server, server→client, multicast)
- [ ] Wire @rpc decorators → RPC registration and dispatch
- [ ] Implement RPC validation and rate limiting
- [ ] Implement RPC ordering (ordered vs unordered per channel)

### 11.4 Prediction & Reconciliation
- [ ] Implement client-side prediction (predict locally, reconcile on correction)
- [ ] Implement server reconciliation (replay inputs on correction)
- [ ] Implement entity interpolation (buffered, for non-predicted entities)
- [ ] Wire Foundation EventLog → operation history for rollback
- [ ] Wire Foundation Tracker.undo() → state rollback for reconciliation
- [ ] Wire @reconciliation decorator → reconciliation strategy
- [ ] Implement smoothing methods (snap, interpolate, threshold-based)

### 11.5 Lag Compensation
- [ ] Implement lag compensation (server rewinds to client's view time)
- [ ] Wire Foundation snapshots → historical state storage
- [ ] Wire Foundation DeltaSync → efficient snapshot diffing
- [ ] Implement maximum rewind limit (MAX_LAG_COMPENSATION_MS)

### 11.6 Anti-Cheat & Security
- [ ] Implement server authority validation
- [ ] Implement input validation and rate limiting
- [ ] Wire Foundation Capabilities → permission checking for network commands
- [ ] Wire @security decorators → authority rules
- [ ] Implement statistical anomaly detection
- [ ] Implement response tiers (warn, kick, ban, shadow ban)

### 11.7 Social Systems
- [ ] Implement matchmaking (lobby, queue, skill-based)
- [ ] Implement voice chat (or integrate middleware e.g. Vivox)
- [ ] Wire @social decorators → social system integration
- [ ] Implement party system (create, invite, leave, leader)
- [ ] Implement text chat with moderation
- [ ] Implement server browser (list, filter, favorites)

---

## 9. Directory Structure

```
engine/networking/
├── __init__.py                    # Public API exports
├── transport/
│   ├── __init__.py
│   ├── udp_transport.py           # UDP with reliability layer
│   ├── tcp_transport.py           # TCP for non-realtime
│   ├── channel.py                 # Channel types (reliable/unreliable/sequenced)
│   ├── packet.py                  # Packet structure, header, coalescing
│   ├── connection.py              # Connection lifecycle, handshake, heartbeat
│   ├── quality.py                 # RTT, jitter, loss metrics + adaptation
│   └── nat.py                     # NAT traversal (STUN, hole punch, TURN)
├── replication/
│   ├── __init__.py
│   ├── replication_manager.py     # Central replication coordinator
│   ├── net_guid.py                # Network GUID assignment
│   ├── actor_channel.py           # Per-entity replication channel
│   ├── relevancy.py               # Interest management (radius/grid/custom)
│   ├── conditions.py              # Replication conditions (always/on_change/etc.)
│   ├── bandwidth.py               # Bandwidth allocation + prioritization
│   └── property_replication.py    # Per-field change detection + serialization
├── rpc/
│   ├── __init__.py
│   ├── rpc_manager.py             # RPC registration + dispatch
│   ├── rpc_channel.py             # RPC transport channel
│   └── rpc_validation.py          # Authority + rate limit checks
├── prediction/
│   ├── __init__.py
│   ├── client_prediction.py       # Input buffer, predict, store sequence
│   ├── server_reconciliation.py   # Detect mismatch, trigger rollback
│   ├── entity_interpolation.py    # Buffered interpolation for remote entities
│   └── smoothing.py               # Snap / interpolate / threshold methods
├── lag_compensation/
│   ├── __init__.py
│   ├── rewind_manager.py          # Server-side world state rewind
│   ├── hitbox_history.py          # Historical hitbox positions
│   └── view_time.py               # Client view-time reconstruction
├── security/
│   ├── __init__.py
│   ├── authority_validator.py     # Server authority enforcement
│   ├── input_validator.py         # Input sanity + bounds checking
│   ├── rate_limiter.py            # Per-player rate limiting
│   ├── anomaly_detector.py        # Statistical / heuristic cheat detection
│   └── response.py                # Warn / kick / ban / shadow ban
├── social/
│   ├── __init__.py
│   ├── matchmaking.py             # Queue, criteria, skill-based matching
│   ├── lobby.py                   # Lobby creation, join, ready-up
│   ├── party.py                   # Party system (create, invite, leader)
│   ├── server_browser.py          # List, filter, favorites
│   ├── voice_chat.py              # Voice pipeline (capture → encode → transmit)
│   └── text_chat.py               # Text channels + moderation
└── serialization/
    ├── __init__.py
    ├── net_serializer.py          # Binary serialization for wire format
    ├── delta_encoder.py           # Delta compression (changed fields only)
    ├── quantizer.py               # Float→Fixed, Vec→compressed, Quat→smallest-3
    └── bit_packer.py              # Bit-level packing for minimal bandwidth
```

---

## 10. Canonical Usage Examples

### Example 1: Networked Player Component
```python
from typing import Annotated
from trinity.base import Component
from trinity.types import FVec3, FQuat, Fixed32
from trinity.descriptors.networking import (
    NetworkedDescriptor, InterpolatedDescriptor,
    PredictedDescriptor, ThrottledNetworkDescriptor,
)
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.decorators.builtin_stacks.network import networked_entity, predicted_entity

@predicted_entity(history_frames=30, max_reconcile_frames=10, snap_threshold=0.5)
class PlayerMovement(Component):
    """Client-predicted player movement with server reconciliation."""
    
    position: Annotated[FVec3,
        TrackedDescriptor(),
        PredictedDescriptor(max_history=30),
        InterpolatedDescriptor(mode="hermite"),
        NetworkedDescriptor(authority="server", priority=10),
    ]
    
    velocity: Annotated[FVec3,
        TrackedDescriptor(),
        PredictedDescriptor(max_history=30),
        NetworkedDescriptor(authority="server", priority=8),
    ]
    
    rotation: Annotated[FQuat,
        TrackedDescriptor(),
        InterpolatedDescriptor(mode="linear"),
        NetworkedDescriptor(authority="owner", priority=5),
    ]
    
    health: Annotated[Fixed32,
        TrackedDescriptor(),
        NetworkedDescriptor(authority="server", priority=10),
    ]
```

### Example 2: RPC Usage
```python
from trinity.base import System
from trinity.decorators.rpc import rpc

class CombatSystem(System):
    """Server-authoritative combat with client requests via RPC."""
    
    @rpc(authority="server", reliable=True)
    def request_attack(self, attacker_id: int, target_id: int, ability_id: int):
        """Client → Server: Request to perform attack."""
        # Server validates: is attacker alive? In range? Ability off cooldown?
        # If valid, apply damage and multicast result
        pass
    
    @rpc(authority="client", reliable=True)
    def notify_damage(self, target_id: int, damage: int, source_id: int):
        """Server → Client: Notify of damage applied."""
        # Client plays VFX, updates UI
        pass
    
    @rpc(authority="client", reliable=False)
    def multicast_explosion(self, position: tuple, radius: float):
        """Server → All Clients: Visual explosion effect."""
        # All clients play VFX (unreliable — cosmetic only)
        pass
```

### Example 3: Bandwidth-Efficient NPC
```python
from trinity.decorators.builtin_stacks.network import bandwidth_efficient

@bandwidth_efficient(radius=5000, max_updates_per_second=10.0, priority=20)
class DistantNPC(Component):
    """Low-priority NPC that only replicates to nearby players."""
    
    position: Annotated[FVec3,
        TrackedDescriptor(),
        ThrottledNetworkDescriptor(max_updates_per_second=10.0),
        InterpolatedDescriptor(mode="linear"),
        NetworkedDescriptor(authority="server", priority=3),
    ]
    
    ai_state: Annotated[int,
        TrackedDescriptor(),
        NetworkedDescriptor(authority="server", priority=1),
    ]
```

### Example 4: Lag-Compensated Weapon
```python
class WeaponSystem(System):
    """Server-side lag compensation for hit detection."""
    
    @rpc(authority="server", reliable=True)
    def fire_weapon(self, shooter_id: int, aim_origin: tuple, aim_dir: tuple, client_time: float):
        """Client fires weapon — server rewinds and checks hit."""
        # 1. Calculate client's view time
        rtt = self.get_connection(shooter_id).rtt
        rewind_time = client_time  # or server_time - rtt/2
        
        # 2. Rewind world state via lag_compensation
        # rewind_manager.rewind_to(rewind_time)
        
        # 3. Raycast at rewound positions
        # hit = physics.raycast(aim_origin, aim_dir)
        
        # 4. Restore current state
        # rewind_manager.restore()
        
        # 5. Apply damage if hit
        pass
```

### Example 5: Matchmaking Flow
```python
class MatchmakingSystem(System):
    """Skill-based matchmaking with party support."""
    
    def queue_player(self, player_id: int, mode: str = "ranked"):
        # 1. Get player MMR
        # 2. Add to queue with criteria
        # 3. Expand search range over time
        # 4. Form match when criteria met
        # 5. Select server (ping-based)
        # 6. Connect all players
        pass
    
    def queue_party(self, party_leader_id: int, member_ids: list, mode: str = "ranked"):
        # Queue entire party as unit
        # Average MMR for matching
        pass
```

---

## 11. Integration Patterns

### Pattern 1: NetworkedDescriptor → Foundation Tracker → Wire
```python
# WRITE PATH (field change → network queue → transport)
player.health = 50
# 1. NetworkedDescriptor.post_set() fires
# 2. Tracker.mark_dirty(player, "health")
# 3. _network_queue.append({field="health", value=50, old_value=100, priority=10})
# 4. End of frame: flush()
# 5. Tracker.all_dirty() → [player]
# 6. pop_network_updates(player) → [{field, value, old_value, priority}]
# 7. net_serializer.serialize(updates) → bytes
# 8. delta_encoder.encode(bytes) → compressed
# 9. transport.send(channel="reliable_ordered", data=compressed)

# READ PATH (wire → deserialize → apply)
# 1. transport.receive() → compressed bytes
# 2. delta_encoder.decode(compressed) → bytes
# 3. net_serializer.deserialize(bytes) → updates
# 4. For each update: entity.field = value (triggers local descriptors)
# 5. If rep_notify: call OnRep_field(old_value)
```

### Pattern 2: Prediction + Reconciliation Loop
```python
# CLIENT TICK:
# 1. Sample input → input_buffer.push(seq, input)
# 2. Apply input locally (predict): PredictedDescriptor records history
# 3. Send input to server: rpc.send_input(seq, input)

# SERVER TICK:
# 1. Receive client input
# 2. Apply in authoritative simulation
# 3. Send authoritative state + last_processed_seq

# CLIENT RECEIVE SERVER STATE:
# 1. Compare server state vs predicted state at last_processed_seq
# 2. If match: discard confirmed inputs from buffer
# 3. If mismatch:
#    a. PredictedDescriptor.rollback(frames_since_confirmed)
#    b. Apply server state
#    c. Re-apply all unconfirmed inputs from buffer
#    d. Smoothing: snap if error > snap_threshold, else interpolate
```

### Pattern 3: Interest Management Filter
```python
# PER-FRAME REPLICATION:
# For each connected client:
#   1. Get client's interest area (@interest config)
#   2. For each networked entity:
#      a. Check relevancy: distance < radius? Owner? Always relevant?
#      b. If relevant: include in replication set
#      c. If newly relevant: send full initial state
#      d. If newly irrelevant: close actor channel
#   3. Prioritize by bandwidth_priority
#   4. Serialize and send within bandwidth budget
```

### Pattern 4: Foundation Bridge → Network Spawning
```python
# SPAWN NETWORKED ENTITY:
# 1. Server creates entity: entity = Bridge.spawn("Player")
# 2. Assign net GUID: net_guid = NetGUIDManager.assign(entity)
# 3. Registry.register(entity, tags={"networked", "player"})
# 4. For each relevant client:
#    a. Open actor channel
#    b. Send spawn message (prefab_id, net_guid, initial_state)
# 5. Client receives spawn:
#    a. Bridge.spawn("Player") locally
#    b. Apply initial state
#    c. Register local net_guid mapping
```

---

## 12. Quick Reference Tables

### Constants
| Constant | Value | Purpose |
|----------|-------|---------|
| DEFAULT_NETWORK_PRIORITY | 1 | Default field replication priority |
| DEFAULT_UPDATE_FREQUENCY | 0 | 0 = send every change |
| DEFAULT_MAX_UPDATES_PER_SECOND | 20.0 | Throttle rate |
| DEFAULT_PREDICTION_HISTORY | 30 | Frames for rollback |
| INTERPOLATION_BUFFER_SIZE | 3 | Snapshots for interpolation |
| NETWORK_AUTHORITY_SERVER | "server" | Server authority string |
| NETWORK_AUTHORITY_CLIENT | "client" | Client authority string |
| NETWORK_AUTHORITY_OWNER | "owner" | Owner authority string |

### Authority Model
| Authority | Who Writes | Who Reads | Use Case |
|-----------|-----------|-----------|----------|
| server | Server only | All clients | Health, score, world state |
| client | Owning client | Server + others | Input, aim direction |
| owner | Owner only | Server + others | Cosmetics, preferences |

### Descriptor Stacking Order (innermost → outermost)
```
TrackedDescriptor          ← innermost (change detection)
  → PredictedDescriptor    ← history buffer
    → InterpolatedDescriptor ← smoothing
      → ThrottledNetworkDescriptor ← rate limiting
        → NetworkedDescriptor ← outermost (queue for wire)
```

### Stack Quick Reference
| Stack | Combines | Use Case |
|-------|----------|----------|
| @networked_entity | component+packed+pooled+networked+serializable+track_changes | Basic replicated entity |
| @bandwidth_efficient | networked+diff+interest+bandwidth_priority+throttle+batch | Distant/low-priority entities |
| @predicted_entity | networked(predicted)+snapshot+server_reconcile+diff | Player-controlled characters |
| @secure_multiplayer | server_authoritative+validated+rate_limited | Anti-cheat hardened systems |

### Networking Decorators Summary
| Decorator | Module | Key Params | Purpose |
|-----------|--------|------------|---------|
| @networked | data_flow | relevance, authority, priority, delta, predicted, interpolated | Core replication config |
| @serializable | data_flow | format, version | Wire serialization |
| @snapshot | data_flow | history_frames | State history for rollback |
| @versioned | data_flow | version, migrations | Schema migration |
| @rpc | rpc | authority, reliable | Remote procedure calls |
| @interest | network_extended | type, radius, always_relevant_to_owner | Relevancy filtering |
| @bandwidth_priority | network_extended | priority, max_bps | Bandwidth allocation |
| @snapshot_interpolation | network_extended | buffer_size_ms, interp_delay_ms | Interpolation config |
| @server_reconcile | network_extended | max_reconcile_frames, snap_threshold | Reconciliation config |

### Descriptors Summary
| Descriptor | Key Params | Method | Purpose |
|------------|------------|--------|---------|
| NetworkedDescriptor | authority, priority, update_frequency | post_set→queue | Wire replication |
| InterpolatedDescriptor | mode (linear/hermite) | get_interpolated(t) | Smooth between snapshots |
| PredictedDescriptor | max_history | rollback(frames) | Client prediction buffer |
| ThrottledNetworkDescriptor | max_updates_per_second | flush() | Rate-limit updates |

---

*End of NETWORKING_CONTEXT.md — This file is the sole reference for implementing engine/networking/.*
