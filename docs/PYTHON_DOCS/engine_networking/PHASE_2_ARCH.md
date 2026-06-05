# PHASE 2 ARCHITECTURE: Replication and State Synchronization

## Phase Overview

Phase 2 builds the state synchronization layer on top of Phase 1's transport primitives. This includes Unreal-style property replication, relevancy filtering, bandwidth management, and reliable actor channels.

---

## 1. Replication Manager Architecture

### 1.1 Component Overview

```
ReplicationManager (central coordinator)
    |
    +-- NetGUIDManager
    |       - Unique ID allocation
    |       - Server: 0x00000000 - 0x7FFFFFFF
    |       - Client: 0x80000000 - 0xFFFFFFFF
    |       - WeakValueDictionary for GC-safe tracking
    |       - Thread-safe with Lock
    |
    +-- RelevancyManager
    |       - Interest filtering per viewer
    |       - Multiple area types
    |       - Priority calculation
    |
    +-- BandwidthManager
    |       - Token bucket rate limiting
    |       - Priority queue scheduling
    |       - Anti-starvation guarantees
    |
    +-- ReplicatedEntity[]
            - Per-entity wrapper
            - PropertyReplicationGroup
            - State: PENDING_SPAWN, ACTIVE, PENDING_DESTROY, DORMANT
```

### 1.2 Entity Registration Flow

```python
# Server registers entity
guid = manager.register_entity(
    entity=player,
    priority=EntityPriority.HIGH,
    owner_id=player.id,
    relevancy=RadiusRelevancy(radius=1000.0)
)

# Entity marked for replication
replicated = ReplicatedEntity(
    guid=guid,
    entity=player,
    priority=EntityPriority.HIGH,
    state=EntityState.PENDING_SPAWN
)
```

### 1.3 Replication Tick

```python
def collect_replication_data(viewer, connection_id):
    # 1. Get relevant entities for this viewer
    relevant = relevancy_manager.get_relevant_entities(viewer)
    
    # 2. Filter by bandwidth budget
    budgeted = bandwidth_manager.allocate(relevant, connection_id)
    
    # 3. Collect property updates
    updates = []
    for entity, guid in budgeted:
        if entity.state == EntityState.PENDING_SPAWN:
            updates.append(create_spawn_packet(entity))
        elif entity.has_dirty_properties():
            updates.append(create_update_packet(entity))
    
    return batch_updates(updates)
```

---

## 2. Relevancy System Architecture

### 2.1 Interest Area Types

```
InterestArea (abstract base)
    |
    +-- AlwaysRelevant
    |       - Priority: 1.0 always
    |       - Use: global state (weather, time)
    |
    +-- OwnerRelevant
    |       - Priority: 1.0 for owner, 0.0 others
    |       - Use: private data (inventory)
    |
    +-- RadiusRelevancy
    |       - Distance-based with falloff
    |       - Priority: linear interpolation
    |       - Use: players, NPCs, pickups
    |
    +-- GridRelevancy
    |       - Spatial hash optimization
    |       - O(1) cell lookup
    |       - Use: large worlds
    |
    +-- CustomRelevancy
    |       - User predicate function
    |       - Use: game-specific rules
    |
    +-- CompositeRelevancy
            - AND/OR of child areas
            - Use: complex rules
```

### 2.2 Radius Relevancy Algorithm

```python
def check_relevancy(entity_position, viewer_position):
    distance = euclidean_distance(entity_position, viewer_position)
    
    if distance > self.radius:
        return RelevancyResult(is_relevant=False, priority=0.0)
    
    if distance <= self.falloff_start:
        priority = 1.0
    else:
        # Linear falloff from falloff_start to radius
        t = (distance - self.falloff_start) / (self.radius - self.falloff_start)
        priority = 1.0 - t
    
    return RelevancyResult(is_relevant=True, priority=priority)
```

### 2.3 Grid Relevancy Optimization

```python
# Spatial hash grid
class GridRelevancy:
    def __init__(self, cell_size=100.0, view_distance=3):
        self._grid: dict[tuple[int,int,int], dict[int, Entity]] = {}
        self._cell_size = cell_size
        self._view_distance = view_distance  # cells in each direction
    
    def get_entities_near(self, position):
        center_cell = self._get_cell(position)
        result = set()
        
        # Check 7x7x7 cube of cells (view_distance=3)
        for dx in range(-self._view_distance, self._view_distance + 1):
            for dy in range(-self._view_distance, self._view_distance + 1):
                for dz in range(-self._view_distance, self._view_distance + 1):
                    cell = (center_cell[0]+dx, center_cell[1]+dy, center_cell[2]+dz)
                    if cell in self._grid:
                        result.update(self._grid[cell].values())
        
        return result
```

---

## 3. Bandwidth Management Architecture

### 3.1 Token Bucket Rate Limiting

```python
class BandwidthBudget:
    max_bps: int           # Sustained rate (bits/sec)
    burst_bps: int         # Maximum burst capacity
    current_tokens: float  # Available bits
    last_update: float     # Timestamp
    
    def refill(self):
        elapsed = time.time() - self.last_update
        self.current_tokens += elapsed * self.max_bps
        self.current_tokens = min(self.current_tokens, self.burst_bps)
        self.last_update = time.time()
    
    def consume(self, bits):
        if bits > self.current_tokens:
            return False
        self.current_tokens -= bits
        return True
```

### 3.2 Priority Scheduling with Anti-Starvation

```python
class BandwidthManager:
    def allocate(self, entities, connection_id):
        budget = self._budgets[connection_id]
        budget.refill()
        
        # Calculate effective priority with starvation boost
        scored = []
        now = time.time()
        for entity in entities:
            base_priority = entity.priority.value
            starvation_time = now - self._last_sent[entity.guid]
            
            if starvation_time > MAX_STARVATION_TIME:
                boost = (starvation_time / MAX_STARVATION_TIME) * STARVATION_BOOST
                effective_priority = min(base_priority + boost, MAX_PRIORITY)
            else:
                effective_priority = base_priority
            
            scored.append((effective_priority, entity))
        
        # Sort by effective priority (descending)
        scored.sort(key=lambda x: -x[0])
        
        # Greedily select entities that fit
        result = []
        for priority, entity in scored:
            size_bits = estimate_update_size(entity) * 8
            if budget.consume(size_bits):
                result.append(entity)
                self._last_sent[entity.guid] = now
        
        return result
```

### 3.3 Priority Levels

| Level | Value | Use Case |
|-------|-------|----------|
| CRITICAL | 100 | Player damage, death events |
| HIGH | 75 | Combat events, weapon fire |
| NORMAL | 50 | Standard entity updates |
| LOW | 25 | Environmental changes |
| MINIMAL | 10 | Background/cosmetic |

---

## 4. Property Replication Architecture

### 4.1 Replication Conditions

```python
class ReplicationCondition(Enum):
    ALWAYS = auto()        # Replicate every tick
    ON_CHANGE = auto()     # Only when value changes (default)
    INITIAL_ONLY = auto()  # Once on spawn
    OWNER_ONLY = auto()    # Only to owning client
    SKIP_OWNER = auto()    # To everyone except owner
    CUSTOM = auto()        # User predicate function
```

### 4.2 Change Notification Modes

```python
class NotifyMode(Enum):
    NONE = auto()          # No callback
    REP_NOTIFY = auto()    # Call OnRep with new value
    WITH_PREVIOUS = auto() # Call OnRep with old and new values
```

### 4.3 Property Definition

```python
@dataclass
class ReplicatedProperty:
    name: str
    type: Type
    condition: ReplicationCondition
    notify_mode: NotifyMode
    serializer: Optional[Callable]
    deserializer: Optional[Callable]
    validator: Optional[Callable]
    
    _previous_value: Any = field(default=None, repr=False)
    _dirty: bool = field(default=False, repr=False)
```

### 4.4 Entity Interface

```python
class NetworkedEntity:
    # Option 1: Declarative field list
    __networked_fields__ = {
        'position': {'condition': 'ON_CHANGE', 'precision': 16},
        'rotation': {'condition': 'ON_CHANGE'},
        'health': {'condition': 'ON_CHANGE', 'notify': 'REP_NOTIFY'},
        'team_id': {'condition': 'INITIAL_ONLY'},
        'inventory': {'condition': 'OWNER_ONLY'},
    }
    
    # Option 2: Property objects
    _replicated_properties = {
        'position': ReplicatedProperty(name='position', type=Vector3, ...),
    }
```

### 4.5 Built-in Serializers

| Type | Format | Size |
|------|--------|------|
| int | `<i` (signed 32-bit) | 4 bytes |
| float | `<f` (32-bit) | 4 bytes |
| bool | `<B` (unsigned byte) | 1 byte |
| str | `<H` length + UTF-8 | 2 + len |
| bytes | `<I` length + raw | 4 + len |
| Vector3 | quantized | 3-9 bytes |
| Quaternion | smallest-three | 4 bytes |

---

## 5. Actor Channel Architecture

### 5.1 Channel State Machine

```
CLOSED ─────> OPENING ─────> OPEN ─────> CLOSING ─────> CLOSED
         SPAWN sent    INITIAL_STATE    CLOSE sent    ACK/timeout
                           received
```

### 5.2 Message Types

| Type | Value | Contents |
|------|-------|----------|
| MSG_SPAWN | 0x01 | GUID + is_owner + initial props |
| MSG_INITIAL_STATE | 0x02 | Full property state |
| MSG_DELTA_UPDATE | 0x03 | Changed properties only |
| MSG_RPC | 0x04 | RPC call payload |
| MSG_CLOSE | 0x05 | Channel closure |
| MSG_ACK | 0x06 | Reliable acknowledgment |

### 5.3 Channel Message Format

```
Header (7 bytes):
+-------------+------------+-------+
| msg_type(1) | sequence(4)| flags(2)|
+-------------+------------+-------+

Payload:
+-----------+------------------+
| length(2) | data (variable)  |
+-----------+------------------+
```

### 5.4 Reliable Delivery

```python
class ActorChannel:
    _send_sequence: int
    _recv_sequence: int
    _pending_ack: dict[int, ChannelMessage]
    
    def send_reliable(self, data):
        msg = ChannelMessage(
            msg_type=MSG_DELTA_UPDATE,
            sequence=self._send_sequence,
            flags=FLAG_RELIABLE,
            payload=data,
            timestamp=time.time()
        )
        self._pending_ack[self._send_sequence] = msg
        self._send_sequence += 1
        return msg.serialize()
    
    def get_retransmit_messages(self, timeout):
        now = time.time()
        retransmit = []
        for seq, msg in self._pending_ack.items():
            if now - msg.timestamp > timeout:
                msg.timestamp = now  # Update for next retransmit
                retransmit.append(msg)
        return retransmit
```

---

## 6. Packet Protocol

### 6.1 Spawn Packet

```
+----------+-----------+------------+-----------+
| GUID (4) | is_owner(1)| props_len(2)| props     |
+----------+-----------+------------+-----------+
```

### 6.2 Update Packet

```
+----------+------------+-----------+
| GUID (4) | props_len(2)| props     |
+----------+------------+-----------+
```

### 6.3 Destroy Packet

```
+----------+
| GUID (4) |
+----------+
```

### 6.4 Batch Packet

```
+------------+----------+----------+----------+
| count (2)  | packet_1 | packet_2 | ...      |
+------------+----------+----------+----------+
```

---

## 7. Connection State Tracking

```python
# Per-connection entity visibility
_connection_entity_states: dict[int, dict[int, EntityState]]
# connection_id -> {guid_value -> EntityState}

class EntityState(Enum):
    PENDING_SPAWN = auto()    # Spawn packet not yet sent
    ACTIVE = auto()           # Receiving updates
    PENDING_DESTROY = auto()  # Destroy packet not yet sent
    DORMANT = auto()          # Exists but not sending updates
```

---

## 8. Data Flow Summary

```
Game Tick
    |
    v
ReplicationManager.tick()
    |
    +-- For each connection:
    |       |
    |       +-- RelevancyManager.get_relevant_entities(viewer)
    |       |       |
    |       |       +-- RadiusRelevancy.check()
    |       |       +-- GridRelevancy.check()
    |       |       +-- CompositeRelevancy.check()
    |       |
    |       +-- BandwidthManager.allocate(relevant, conn_id)
    |       |       |
    |       |       +-- Token bucket refill
    |       |       +-- Priority sort with anti-starvation
    |       |       +-- Greedy selection
    |       |
    |       +-- For each allocated entity:
    |               |
    |               +-- PropertyReplicationGroup.get_dirty_properties()
    |               +-- Serialize changed properties
    |               +-- ActorChannel.send_reliable()
    |
    +-- NetGUIDManager.cleanup_unreferenced()
```
