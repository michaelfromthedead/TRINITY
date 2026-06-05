# Engine Networking Replication Investigation

**Module:** `engine/networking/replication/`  
**Total Lines:** 3,538  
**Classification:** REAL (fully implemented, production-ready)  
**Date:** 2026-05-22

## Overview

The replication system provides a complete Unreal-style network state synchronization framework with property-level replication, relevancy filtering, bandwidth management, and per-entity actor channels. All 7 files contain substantial, working implementations with no stub code.

## File Classification

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `replication_manager.py` | 747 | REAL | Central replication coordinator |
| `actor_channel.py` | 680 | REAL | Per-entity replication streams |
| `bandwidth.py` | 579 | REAL | Bandwidth allocation with anti-starvation |
| `relevancy.py` | 571 | REAL | Interest management system |
| `property_replication.py` | 436 | REAL | Per-property change detection |
| `net_guid.py` | 398 | REAL | Network GUID system |
| `__init__.py` | 127 | REAL | Module exports |

## Architecture Summary

### Replication Graph

```
ReplicationManager (central coordinator)
    |
    +-- NetGUIDManager (unique ID allocation)
    |       +-- Server authority: 0x0000_0000 - 0x7FFF_FFFF
    |       +-- Client authority: 0x8000_0000 - 0xFFFF_FFFF
    |       +-- WeakValueDictionary for entity tracking (GC-safe)
    |       +-- Thread-safe with locks
    |
    +-- RelevancyManager (interest filtering)
    |       +-- RadiusRelevancy (3D distance-based)
    |       +-- GridRelevancy (spatial hash optimization)
    |       +-- OwnerRelevant (owner-only visibility)
    |       +-- CompositeRelevancy (AND/OR logic)
    |       +-- CustomRelevancy (predicate functions)
    |
    +-- BandwidthManager (rate limiting)
    |       +-- Token bucket algorithm
    |       +-- Priority queues per connection
    |       +-- Anti-starvation with time-based boosting
    |
    +-- ReplicatedEntity (per-entity wrapper)
            +-- PropertyReplicationGroup (property batching)
            +-- EntityState (PENDING_SPAWN, ACTIVE, PENDING_DESTROY, DORMANT)
```

### Actor Channel System

```
ActorChannelManager
    |
    +-- Per-entity channels (guid -> {connection_id -> ActorChannel})
    |
    +-- Channel states: CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED
    |
    +-- Message types:
            MSG_SPAWN (0x01): Entity creation with initial state
            MSG_INITIAL_STATE (0x02): Full state transfer
            MSG_DELTA_UPDATE (0x03): Incremental updates
            MSG_RPC (0x04): Remote procedure calls
            MSG_CLOSE (0x05): Channel closure
            MSG_ACK (0x06): Reliable delivery acknowledgment
    |
    +-- Reliable delivery with sequence numbers and retransmission
```

## Relevancy System

### Interest Area Types

| Type | Use Case | Priority Calculation |
|------|----------|---------------------|
| `AlwaysRelevant` | Global state (weather, time) | Constant 1.0 |
| `OwnerRelevant` | Private data (inventory) | 1.0 for owner, 0.0 others |
| `RadiusRelevancy` | Distance-based (players, NPCs) | Linear falloff from falloff_start to radius |
| `GridRelevancy` | Large worlds (spatial hash) | Cell distance-based |
| `CustomRelevancy` | Game-specific rules | User predicate function |
| `CompositeRelevancy` | Combined rules | AND/OR of child areas |

### Relevancy Check Flow

1. Entity-specific area checked first
2. Falls back to viewer-specific area
3. Falls back to default area (RadiusRelevancy)
4. Returns `RelevancyResult(is_relevant, priority, reason)`

### RadiusRelevancy Algorithm

```python
if distance > radius:
    return RelevancyResult(is_relevant=False, priority=0.0)
elif distance <= falloff_start:
    priority = 1.0
else:
    priority = 1.0 - (distance - falloff_start) / (radius - falloff_start)
```

### GridRelevancy Optimization

- Spatial hash grid with configurable cell size
- O(1) cell lookup vs O(n) radius checks
- View distance in cells (default: 3 cells each direction = 7x7x7 cube)
- Entities registered/updated in grid on position change

## Bandwidth Management

### Token Bucket Rate Limiting

```python
BandwidthBudget:
    max_bps: int          # Sustained rate (bits/sec)
    burst_bps: int        # Maximum burst capacity
    current_tokens: float # Available bits
    
    refill():
        elapsed = now - last_update
        current_tokens += elapsed * max_bps
        current_tokens = min(current_tokens, burst_bps)
```

### Priority Queue with Anti-Starvation

```python
effective_priority = base_priority
if starvation_time > MAX_STARVATION_TIME:
    boost = (starvation_time / MAX_STARVATION_TIME) * STARVATION_PRIORITY_BOOST
    effective_priority = min(priority + boost, MAX_PRIORITY_WITH_BOOST)
```

### Entity Priority Levels

| Level | Value | Use Case |
|-------|-------|----------|
| CRITICAL | 100 | Player damage, death |
| HIGH | 75 | Combat events |
| NORMAL | 50 | Standard updates |
| LOW | 25 | Environmental |
| MINIMAL | 10 | Background/cosmetic |

### Allocation Algorithm

1. Refill bandwidth tokens based on elapsed time
2. Sort entities by effective priority (with starvation boost)
3. Greedily select entities that fit within budget
4. Track last_sent_time for anti-starvation
5. Return list of (entity, guid) to send

## Property Sync System

### Replication Conditions

| Condition | Behavior |
|-----------|----------|
| `ALWAYS` | Replicate every tick |
| `ON_CHANGE` | Only when value changes (default) |
| `INITIAL_ONLY` | Once on spawn |
| `OWNER_ONLY` | Only to owning client |
| `SKIP_OWNER` | To everyone except owner |
| `CUSTOM` | User predicate function |

### Change Notification Modes

| Mode | Behavior |
|------|----------|
| `NONE` | No callback |
| `REP_NOTIFY` | Call OnRep with new value |
| `WITH_PREVIOUS` | Call OnRep with old and new values |

### Built-in Serializers

| Type | Format | Size |
|------|--------|------|
| `int` | `<i` (little-endian signed 32-bit) | 4 bytes |
| `float` | `<f` (little-endian float) | 4 bytes |
| `bool` | `<B` (unsigned byte) | 1 byte |
| `str` | `<H` length + UTF-8 bytes | 2 + len bytes |
| `bytes` | `<I` length + raw bytes | 4 + len bytes |

Custom serializers registered via `register_serializer(type_, serialize, deserialize)`.

### Property Extraction

Entities can expose replicated properties via:
1. `__networked_fields__` dict attribute
2. `_replicated_properties` dict attribute

## Packet Protocol

### Packet Types

| Type | Value | Contents |
|------|-------|----------|
| PACKET_SPAWN | 0x01 | GUID (4B) + is_owner (1B) + props_len (2B) + props |
| PACKET_UPDATE | 0x02 | GUID (4B) + props_len (2B) + props |
| PACKET_DESTROY | 0x03 | GUID (4B) |
| PACKET_BATCH | 0x04 | Multiple packets concatenated |

### Channel Message Format

```
Header (7 bytes):
    msg_type (1B) + sequence (4B) + flags (2B)
Payload:
    length (2B) + data (variable)
```

## Key Implementation Details

### NetGUID Format (32-bit)

```
Server GUIDs: 0x0000_0000 - 0x7FFF_FFFF
    Bit 31 = 0
    Bits 0-30: Sequence number

Client GUIDs: 0x8000_0000 - 0xFFFF_FFFF
    Bit 31 = 1
    Bits 16-30: Client ID (0-32767)
    Bits 0-15: Per-client sequence
```

### Thread Safety

- `NetGUIDManager` uses `threading.Lock` for all operations
- `WeakValueDictionary` used for entity tracking (allows GC)
- GUID recycling via `_free_guids` list

### Connection State Tracking

```python
# Per-connection entity visibility
_connection_entity_states: dict[int, dict[int, EntityState]]
# connection_id -> {guid_value -> EntityState}
```

### Reliable Delivery

- Sequence numbers per channel
- Pending ACK tracking with retransmission
- Out-of-order message buffering
- Configurable retransmit timeout (default from config)

## Configuration Integration

All magic numbers imported from `..config.get_config()`:

- Packet type constants
- Actor message types
- Priority levels
- Bandwidth limits
- Starvation thresholds
- Grid/radius defaults
- GUID ranges and masks

## Dependencies

### Internal

- `engine/networking/config.py` - Configuration values

### External (stdlib only)

- `struct` - Binary serialization
- `time` - Timestamps
- `heapq` - Priority queue
- `math` - Distance calculation
- `threading` - Thread safety
- `weakref` - WeakValueDictionary
- `dataclasses` - Data structures
- `enum` - Enumerations
- `typing` - Type hints
- `abc` - Abstract base classes
- `logging` - Logging
- `pickle` - Fallback serialization (for unregistered types)

## Quality Assessment

### Strengths

1. **Complete Unreal-style replication** - Mirrors UE4/5 replication concepts
2. **Modular design** - Clear separation of concerns
3. **Production-ready bandwidth management** - Token bucket with anti-starvation
4. **Flexible relevancy system** - Composable interest areas
5. **Efficient spatial queries** - Grid-based optimization available
6. **Thread-safe GUID management** - Proper locking
7. **Configurable** - All constants from config file
8. **Type-safe** - Full type hints throughout

### Potential Improvements

1. No delta compression for property updates (mentioned in code but not implemented)
2. No snapshot interpolation layer
3. No jitter buffer for update smoothing
4. Grid relevancy could use octree for non-uniform distributions
5. Missing network simulation/lag compensation layer

## Evidence Code Samples

### Delta compression (property_replication.py:271-283)
```python
def serialize_delta(self, baseline: Optional[T] = None) -> Optional[bytes]:
    """Serialize only if changed from baseline."""
    if baseline is not None and self.value == baseline:
        return None
    return self.serialize()
```

### Priority scheduling with anti-starvation (bandwidth.py:376-381)
```python
if starvation > MAX_STARVATION_TIME:
    boost = (starvation / MAX_STARVATION_TIME) * STARVATION_PRIORITY_BOOST
    effective_priority = min(priority + boost, _config.MAX_PRIORITY_WITH_BOOST)
```

### Spatial grid relevancy (relevancy.py:343-367)
```python
def get_entities_near(self, position: tuple[float, float, float]) -> set[Any]:
    center_cell = self._get_cell(position)
    result: set[Any] = set()
    for dx in range(-self.view_distance, self.view_distance + 1):
        for dy in range(-self.view_distance, self.view_distance + 1):
            for dz in range(-self.view_distance, self.view_distance + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                entities_dict = self._grid.get(cell)
                if entities_dict:
                    result.update(entities_dict.values())
    return result
```

### Reliable delivery with retransmission (actor_channel.py:307-325)
```python
def get_retransmit_messages(self, timeout: float = _config.DEFAULT_RETRANSMIT_TIMEOUT) -> list[ChannelMessage]:
    now = time.time()
    retransmit = []
    for seq, msg in list(self._pending_ack.items()):
        if now - msg.timestamp > timeout:
            msg.timestamp = now
            retransmit.append(msg)
    return retransmit
```

## Integration Points

### Expected Entity Interface

```python
class NetworkedEntity:
    __networked_fields__: dict[str, dict]  # Optional
    _replicated_properties: dict[str, ReplicatedProperty]  # Optional
    position: tuple[float, float, float]  # For relevancy
    owner_id: int  # For owner-based filtering
```

### Expected Viewer Interface

```python
class NetworkViewer:
    player_id: int  # Or use id(viewer) as fallback
    position: tuple[float, float, float]  # For relevancy
```

## Usage Example

```python
from engine.networking.replication import (
    ReplicationManager, ReplicationRole, EntityPriority,
    RadiusRelevancy, BandwidthManager
)

# Server setup
manager = ReplicationManager(role=ReplicationRole.SERVER)

# Register entity
guid = manager.register_entity(
    entity=player,
    priority=EntityPriority.HIGH,
    owner_id=player.id,
    relevancy=RadiusRelevancy(radius=1000.0)
)

# Each tick: collect data for each connection
for conn_id, viewer in connections.items():
    data = manager.collect_replication_data(viewer, conn_id)
    send_to_client(conn_id, data)

# Client: apply received data
manager.apply_replication_data(received_data)
```

## Verdict

**REAL IMPLEMENTATION**

This is a complete, production-quality replication system inspired by Unreal Engine's networking model. All core features are fully implemented with proper data structures, algorithms, and edge case handling. The system provides property-level dirty tracking, relevancy filtering, bandwidth management with anti-starvation, reliable delivery with actor channels, and efficient GUID management.
