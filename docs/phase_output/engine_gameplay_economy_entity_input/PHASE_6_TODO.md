# PHASE 6 TODO: Networking Preparation

## Summary

Add replication metadata and authority markers for future multiplayer.

---

## T-NET-6.1: Networking Decorators

**File**: `engine/networking/decorators.py`

### Tasks

- [ ] Create @replicated class decorator with authority and scope
- [ ] Create @replicated_field property decorator with delta hints
- [ ] Create @server_rpc method decorator
- [ ] Create @client_rpc method decorator
- [ ] Create @replicate_if conditional decorator
- [ ] Create @authority method decorator
- [ ] All decorators are no-op by default (metadata only)

### Acceptance Criteria

- All decorators importable without networking stack
- Decorated classes function identically to undecorated
- Metadata accessible via _replication_meta attribute

---

## T-NET-6.2: Authority Enum and Validation

**File**: `engine/networking/authority.py`

### Tasks

- [ ] Create Authority enum (SERVER, CLIENT, SHARED)
- [ ] Create Scope enum (ALL, OWNER_ONLY, LOCAL)
- [ ] Create DeltaHint enum (HIGH_FREQUENCY, LOW_FREQUENCY, SPARSE)
- [ ] Create is_server() and is_client() stubs (always return True/False)
- [ ] Create authority validation function (disabled by default)
- [ ] Create ENABLE_AUTHORITY_CHECKS flag

### Acceptance Criteria

- Enums have clear docstrings
- Validation disabled by default
- Validation raises AuthorityError when enabled and violated

---

## T-NET-6.3: Economy Replication Annotations

**Files**: `engine/gameplay/economy/*.py`

### Tasks

- [ ] Annotate ItemInstance with @replicated(scope=OWNER_ONLY)
- [ ] Annotate ItemInstance.quantity with @replicated_field(delta=HIGH_FREQUENCY)
- [ ] Annotate InventoryContainer with @replicated(scope=OWNER_ONLY)
- [ ] Annotate EquipmentContainer with @replicated(scope=ALL)
- [ ] Annotate CraftingQueue with @replicated(scope=OWNER_ONLY)
- [ ] Add @server_rpc to equip/unequip methods
- [ ] Add @server_rpc to craft methods

### Acceptance Criteria

- All mutable state annotated
- Authority is SERVER for all economy mutations
- Owner-only scope for private inventory data

---

## T-NET-6.4: Entity Replication Annotations

**Files**: `engine/gameplay/entity/*.py`

### Tasks

- [ ] Annotate Actor with @replicated
- [ ] Annotate Actor.transform with @replicated_field(delta=HIGH_FREQUENCY)
- [ ] Annotate Character.velocity with @replicated_field(delta=HIGH_FREQUENCY)
- [ ] Annotate Character.movement_mode with @replicated_field(delta=LOW_FREQUENCY)
- [ ] Annotate Pawn.controller_id with @replicated_field
- [ ] Add @server_rpc to spawn/destroy methods
- [ ] Add @client_rpc to prediction correction methods

### Acceptance Criteria

- Transform and velocity marked for prediction
- Controller changes marked for immediate sync
- Spawn/destroy are server-authoritative

---

## T-NET-6.5: Input Local Annotation

**Files**: `engine/gameplay/input/*.py`

### Tasks

- [ ] Annotate all input state as LOCAL (not replicated)
- [ ] Document that input is sent via RPC, not state replication
- [ ] Add @authority(CLIENT) to input processing methods

### Acceptance Criteria

- No input state marked as replicated
- Clear documentation on input handling model

---

## T-NET-6.6: RPC Stubs for Common Operations

**Files**: Various

### Tasks

- [ ] Add equip_item_rpc(item_id, slot) -> result
- [ ] Add craft_item_rpc(recipe_id) -> result
- [ ] Add pickup_item_rpc(item_actor_id) -> result
- [ ] Add use_item_rpc(item_id) -> result
- [ ] Add move_to_rpc(position) -> ack
- [ ] Add possess_pawn_rpc(pawn_id) -> result
- [ ] All RPCs have clear docstrings explaining flow

### Acceptance Criteria

- RPCs are stubs with pass body
- Docstrings explain client/server flow
- Return types documented

---

## T-NET-6.7: Replication Metadata Inspection

**File**: `engine/networking/replication.py`

### Tasks

- [ ] Create ReplicationMeta dataclass
- [ ] Create get_replication_meta(cls) function
- [ ] Create get_replicated_fields(obj) function
- [ ] Create get_rpc_methods(cls) function
- [ ] Create validate_authority(obj, method) function

### Acceptance Criteria

- Metadata extractable from annotated classes
- Field list includes delta hints
- RPC list includes direction (server/client)

---

## T-NET-6.8: Documentation

**File**: `engine/networking/README.md` (if requested)

### Tasks

- [ ] Document replication model
- [ ] Document authority model
- [ ] Document RPC conventions
- [ ] Document delta compression hints
- [ ] Provide examples for each pattern

### Acceptance Criteria

- README only created if explicitly requested
- Clear examples for common patterns
