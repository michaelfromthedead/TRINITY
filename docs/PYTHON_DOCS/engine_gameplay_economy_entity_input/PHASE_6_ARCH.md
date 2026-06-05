# PHASE 6 ARCHITECTURE: Networking Preparation

## Overview

Add replication infrastructure for future multiplayer support. This phase adds metadata and markers only — actual networking implementation is out of scope.

## Replication Requirements

### State Classification

| Classification | Description | Example |
|----------------|-------------|---------|
| **Replicated** | Server-authoritative, synced to clients | Actor position, health |
| **Local** | Client-only, not replicated | Input state, UI data |
| **Predicted** | Client-predicted, server-corrected | Character movement |
| **Owner-only** | Replicated to owning client only | Inventory contents |

### Authority Model

| Authority | Description | Usage |
|-----------|-------------|-------|
| **Server** | Server owns truth, clients receive | NPC state, world objects |
| **Client** | Client owns truth, server accepts | Input state |
| **Shared** | Both can modify, conflict resolution needed | Crafting queue |

## Components to Annotate

### Economy Module

| Component | Replication | Authority | Delta Hints |
|-----------|-------------|-----------|-------------|
| ItemInstance.quantity | Owner-only | Server | High frequency |
| InventoryContainer.slots | Owner-only | Server | Sparse updates |
| EquipmentContainer.equipped | Replicated | Server | Low frequency |
| CraftingQueue.progress | Owner-only | Server | High frequency |

### Entity Module

| Component | Replication | Authority | Delta Hints |
|-----------|-------------|-----------|-------------|
| Actor.transform | Replicated | Server/Predicted | Very high frequency |
| Character.velocity | Replicated | Server/Predicted | High frequency |
| Character.movement_mode | Replicated | Server | Low frequency |
| Pawn.controller_id | Replicated | Server | On change only |

### Input Module

| Component | Replication | Authority | Delta Hints |
|-----------|-------------|-----------|-------------|
| Input state | Local | Client | Never replicated |
| Action bindings | Local | Client | Never replicated |

## Architecture Decisions

### ADR-NET-1: Replication Decorators

Use decorators to mark replication metadata:
```python
@replicated(authority=Authority.SERVER, scope=Scope.OWNER_ONLY)
class InventoryContainer:
    @replicated_field(delta=DeltaHint.HIGH_FREQUENCY)
    quantity: int
```

### ADR-NET-2: Delta Compression Hints

Provide hints for efficient delta compression:
- **HIGH_FREQUENCY**: Small changes often, use differential encoding
- **LOW_FREQUENCY**: Large changes rarely, use full encoding
- **SPARSE**: Many fields, few change, use field presence bits

### ADR-NET-3: Authority Markers

Mark authority to enable validation:
```python
class Actor:
    @authority(Authority.SERVER)
    def apply_damage(self, amount: int):
        # Server only
        pass
    
    @authority(Authority.CLIENT)
    def send_input(self, input: InputState):
        # Client only
        pass
```

### ADR-NET-4: Replication Conditions

Support conditional replication:
```python
@replicate_if(lambda self: self.is_visible)
transform: Transform
```

### ADR-NET-5: RPC Stubs

Mark methods as RPCs for future implementation:
```python
@server_rpc
def request_equip(self, item_id: str, slot: EquipSlot):
    """Called by client, executed on server."""
    pass

@client_rpc
def notify_equip_result(self, success: bool, reason: str):
    """Called by server, executed on client."""
    pass
```

## Implementation Approach

### Phase 6a: Define Decorators

Create decorator definitions with no runtime behavior:
```python
def replicated(authority=Authority.SERVER, scope=Scope.ALL):
    def decorator(cls):
        cls._replication_meta = ReplicationMeta(authority, scope)
        return cls
    return decorator
```

### Phase 6b: Annotate Classes

Add decorators to all relevant classes.

### Phase 6c: Add RPC Stubs

Add RPC method stubs with clear documentation.

### Phase 6d: Validation Layer

Add authority validation that can be enabled:
```python
if ENABLE_AUTHORITY_CHECKS:
    if not is_server() and method._authority == Authority.SERVER:
        raise AuthorityError("Server-only method called on client")
```

## Files to Create

```
engine/
  networking/
    __init__.py
    decorators.py          # @replicated, @replicated_field, @server_rpc, etc.
    authority.py           # Authority enum, validation
    delta.py               # DeltaHint enum, compression hints
    replication.py         # ReplicationMeta dataclass
```

## Risks

| Risk | Mitigation |
|------|------------|
| Over-engineering | Metadata only, no networking code |
| Breaking existing code | Decorators are no-op by default |
| Inconsistent annotation | Comprehensive checklist |
