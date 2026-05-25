# PHASE 2 TODO: Replication and State Synchronization

## Overview

Phase 2 implements the state synchronization layer. All tasks assume the existing implementation is production-ready; these TODOs focus on testing, verification, and identified gaps.

---

## 1. NetGUID Manager Tasks

### 1.1 Unit Tests: GUID Allocation

**File**: `tests/blackbox_net_guid.py`

**Acceptance Criteria**:
- [ ] `allocate_server_guid()` returns value in range 0x00000000-0x7FFFFFFF
- [ ] `allocate_client_guid(client_id)` returns value in range 0x80000000-0xFFFFFFFF
- [ ] Client GUID encodes client_id in bits 16-30
- [ ] Consecutive server allocations increment sequence
- [ ] `release_guid(guid)` adds to free list
- [ ] Released GUID reused on next allocation
- [ ] Thread-safe: concurrent allocations don't produce duplicates
- [ ] WeakValueDictionary releases entity references on GC
- [ ] `get_entity(guid)` returns None for released GUID
- [ ] `get_guid(entity)` returns None after entity GC'd

---

### 1.2 Unit Tests: Thread Safety

**File**: `tests/whitebox_net_guid_threading.py`

**Acceptance Criteria**:
- [ ] 10 threads allocating 1000 GUIDs each: no duplicates
- [ ] 10 threads releasing/allocating: no race conditions
- [ ] Lock contention under load: < 1ms average wait time

---

## 2. Relevancy System Tasks

### 2.1 Unit Tests: Radius Relevancy

**File**: `tests/blackbox_relevancy.py`

**Acceptance Criteria**:
- [ ] Entity at distance 0: priority 1.0, is_relevant True
- [ ] Entity at falloff_start: priority 1.0
- [ ] Entity at midpoint (falloff_start + radius)/2: priority 0.5
- [ ] Entity at radius boundary: priority ~0.0
- [ ] Entity beyond radius: is_relevant False, priority 0.0
- [ ] Position update changes relevancy result

---

### 2.2 Unit Tests: Grid Relevancy

**File**: `tests/blackbox_grid_relevancy.py`

**Acceptance Criteria**:
- [ ] Entity registered in correct cell based on position
- [ ] Entity updated to new cell on position change
- [ ] `get_entities_near()` returns entities in view_distance cells
- [ ] Entity outside view_distance not returned
- [ ] Empty cells handled gracefully
- [ ] Cell boundary edge cases correct

---

### 2.3 Unit Tests: Composite Relevancy

**File**: `tests/blackbox_composite_relevancy.py`

**Acceptance Criteria**:
- [ ] AND composite: both children must pass
- [ ] OR composite: either child can pass
- [ ] Priority: AND takes minimum, OR takes maximum
- [ ] Nested composites evaluate correctly
- [ ] Empty composite: AlwaysRelevant fallback

---

### 2.4 Unit Tests: Owner Relevancy

**File**: `tests/blackbox_owner_relevancy.py`

**Acceptance Criteria**:
- [ ] Owner viewer: priority 1.0, is_relevant True
- [ ] Non-owner viewer: priority 0.0, is_relevant False
- [ ] Entity with no owner: always not relevant

---

## 3. Bandwidth Management Tasks

### 3.1 Unit Tests: Token Bucket

**File**: `tests/blackbox_bandwidth.py`

**Acceptance Criteria**:
- [ ] Initial tokens equal burst_bps
- [ ] `refill()` adds tokens based on elapsed time
- [ ] Tokens capped at burst_bps
- [ ] `consume()` returns False when insufficient tokens
- [ ] `consume()` deducts tokens on success
- [ ] Sustained rate enforced over time

---

### 3.2 Unit Tests: Priority Scheduling

**File**: `tests/blackbox_priority_scheduling.py`

**Acceptance Criteria**:
- [ ] CRITICAL priority selected before HIGH
- [ ] HIGH priority selected before NORMAL
- [ ] Equal priority: first-come-first-served
- [ ] Budget exhausted: lower priority entities skipped
- [ ] Entity size considered in budget calculation

---

### 3.3 Unit Tests: Anti-Starvation

**File**: `tests/blackbox_anti_starvation.py`

**Acceptance Criteria**:
- [ ] Entity not sent for MAX_STARVATION_TIME: priority boosted
- [ ] Boost increases linearly with starvation time
- [ ] Boosted priority capped at MAX_PRIORITY_WITH_BOOST
- [ ] Previously starved entity: last_sent updated on send
- [ ] Starvation tracking per-entity per-connection

---

## 4. Property Replication Tasks

### 4.1 Unit Tests: Dirty Tracking

**File**: `tests/blackbox_property_replication.py`

**Acceptance Criteria**:
- [ ] Property marked dirty on value change
- [ ] Property not dirty if value unchanged
- [ ] `clear_dirty()` resets dirty flag
- [ ] `get_dirty_properties()` returns only dirty properties
- [ ] Deep comparison for complex types (dict, list)

---

### 4.2 Unit Tests: Replication Conditions

**File**: `tests/blackbox_replication_conditions.py`

**Acceptance Criteria**:
- [ ] ALWAYS: property included every tick
- [ ] ON_CHANGE: property included only when dirty
- [ ] INITIAL_ONLY: property included only on spawn
- [ ] OWNER_ONLY: property included only for owner connection
- [ ] SKIP_OWNER: property excluded for owner connection
- [ ] CUSTOM: predicate function called with context

---

### 4.3 Unit Tests: Change Notification

**File**: `tests/blackbox_change_notification.py`

**Acceptance Criteria**:
- [ ] NONE: no callback invoked
- [ ] REP_NOTIFY: callback with new value
- [ ] WITH_PREVIOUS: callback with old and new values
- [ ] Callback invoked on receive, not send
- [ ] Callback exception doesn't break replication

---

### 4.4 Unit Tests: Serializers

**File**: `tests/blackbox_property_serializers.py`

**Acceptance Criteria**:
- [ ] int serializer roundtrip
- [ ] float serializer roundtrip
- [ ] bool serializer roundtrip
- [ ] str serializer roundtrip with Unicode
- [ ] bytes serializer roundtrip
- [ ] Custom serializer registration and use
- [ ] Pickle fallback for unregistered types

---

## 5. Actor Channel Tasks

### 5.1 Unit Tests: Channel State Machine

**File**: `tests/blackbox_actor_channel.py`

**Acceptance Criteria**:
- [ ] Channel starts CLOSED
- [ ] Spawn message transitions to OPENING
- [ ] Initial state received transitions to OPEN
- [ ] Close message transitions to CLOSING
- [ ] ACK or timeout transitions to CLOSED
- [ ] State transition callbacks invoked

---

### 5.2 Unit Tests: Reliable Delivery

**File**: `tests/blackbox_channel_reliability.py`

**Acceptance Criteria**:
- [ ] Sent message tracked in pending_ack
- [ ] ACK received removes from pending_ack
- [ ] Timeout triggers retransmit list
- [ ] Retransmit updates timestamp
- [ ] Out-of-order messages buffered
- [ ] Buffer drained when gaps filled

---

### 5.3 Unit Tests: Message Serialization

**File**: `tests/blackbox_channel_messages.py`

**Acceptance Criteria**:
- [ ] MSG_SPAWN serialization roundtrip
- [ ] MSG_INITIAL_STATE serialization roundtrip
- [ ] MSG_DELTA_UPDATE serialization roundtrip
- [ ] MSG_RPC serialization roundtrip
- [ ] MSG_CLOSE serialization roundtrip
- [ ] MSG_ACK serialization roundtrip

---

## 6. Integration Tests

### 6.1 Entity Spawn Replication

**File**: `tests/integration_entity_spawn.py`

**Acceptance Criteria**:
- [ ] Server registers entity
- [ ] Client receives spawn packet
- [ ] Client creates entity with correct GUID
- [ ] Initial properties match server state
- [ ] Owner flag correctly set for owning client

---

### 6.2 Property Update Replication

**File**: `tests/integration_property_update.py`

**Acceptance Criteria**:
- [ ] Server modifies property
- [ ] Property marked dirty
- [ ] Client receives delta update
- [ ] Client property value matches server
- [ ] Change notification callback invoked

---

### 6.3 Entity Destroy Replication

**File**: `tests/integration_entity_destroy.py`

**Acceptance Criteria**:
- [ ] Server unregisters entity
- [ ] Client receives destroy packet
- [ ] Client removes entity
- [ ] GUID released on both sides

---

### 6.4 Relevancy-Based Updates

**File**: `tests/integration_relevancy.py`

**Acceptance Criteria**:
- [ ] Entity enters relevancy range: spawn sent
- [ ] Entity in range: updates sent
- [ ] Entity leaves range: destroy sent
- [ ] Entity re-enters range: spawn sent again
- [ ] Priority affects update frequency

---

### 6.5 Bandwidth-Constrained Replication

**File**: `tests/integration_bandwidth.py`

**Acceptance Criteria**:
- [ ] High-priority entities sent first
- [ ] Low-priority entities deferred when over budget
- [ ] Starved entities eventually sent
- [ ] Bandwidth limits not exceeded

---

## 7. Gap Tasks

### 7.1 Gap: Delta Compression for Properties

**File**: `engine/networking/replication/property_replication.py` (modify)

**Background**: Code mentions delta compression but only baseline diffing exists.

**Acceptance Criteria**:
- [ ] Property values delta-compressed against last acknowledged state
- [ ] Integer delta: XOR or difference encoding
- [ ] Float delta: quantized difference
- [ ] String delta: not delta-compressed (full replacement)
- [ ] Compression ratio tracked in stats

---

### 7.2 Gap: Dormancy System

**File**: `engine/networking/replication/dormancy.py` (new)

**Background**: DORMANT state exists but dormancy logic not implemented.

**Acceptance Criteria**:
- [ ] Entity marked dormant after inactivity period
- [ ] Dormant entities skip replication tick
- [ ] Wake condition: property change or explicit wake
- [ ] Dormancy reduces server CPU for static entities

---

## 8. Performance Tasks

### 8.1 Benchmark: Relevancy Checking

**File**: `benchmarks/relevancy_performance.py`

**Acceptance Criteria**:
- [ ] RadiusRelevancy: > 100,000 checks/second
- [ ] GridRelevancy: > 500,000 checks/second
- [ ] Grid vs Radius: 5x improvement with 1000 entities

---

### 8.2 Benchmark: Property Serialization

**File**: `benchmarks/property_serialization.py`

**Acceptance Criteria**:
- [ ] 100-property entity: < 1ms serialization
- [ ] 10-dirty-property delta: < 0.2ms serialization
- [ ] Quantized vector: < 0.01ms

---

### 8.3 Benchmark: Bandwidth Manager

**File**: `benchmarks/bandwidth_manager.py`

**Acceptance Criteria**:
- [ ] 100 entities, 10 connections: < 1ms allocation
- [ ] Priority sort with 1000 entities: < 5ms
- [ ] Anti-starvation calculation overhead: < 10%

---

## 9. Documentation Tasks

### 9.1 API Documentation

**Acceptance Criteria**:
- [ ] ReplicationManager usage example
- [ ] Relevancy configuration guide
- [ ] Bandwidth tuning guide
- [ ] PropertyReplication decorator usage
- [ ] Actor channel lifecycle diagram

---

### 9.2 Tuning Guide

**Acceptance Criteria**:
- [ ] Bandwidth allocation formulas explained
- [ ] Priority level selection guidelines
- [ ] Relevancy radius sizing recommendations
- [ ] Grid cell size optimization
- [ ] Anti-starvation parameter tuning
