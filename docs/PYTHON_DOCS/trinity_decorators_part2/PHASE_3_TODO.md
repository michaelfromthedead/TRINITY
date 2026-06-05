# PHASE 3 TODO: Assets, Networking & Persistence

## Overview

Validate assets, LOD/streaming, prefabs, networking, save system, transactions, and replay decorators.

---

## T3.1: Validate Assets Decorators

**File**: `trinity/decorators/assets.py`

**Tasks**:
- [ ] Verify `@asset` accepts path and preload settings
- [ ] Verify `@asset` creates AssetConfig dataclass
- [ ] Verify `@preload` marks asset for preloading
- [ ] Verify `@cook` validates platform settings
- [ ] Verify `@cook` creates CookConfig dataclass
- [ ] Verify `@residency` sets memory residency policy
- [ ] Verify `@import_settings` creates ImportSettingsConfig

**Config Dataclass Tests**:
- [ ] AssetConfig is frozen (immutable)
- [ ] CookConfig is frozen
- [ ] ResidencyConfig is frozen
- [ ] ImportSettingsConfig is frozen

**Acceptance Criteria**:
- All 5 decorators produce correct steps
- Config dataclasses are frozen and hashable
- All register in "assets" registry

---

## T3.2: Validate LOD Streaming Decorators

**File**: `trinity/decorators/lod_streaming.py`

**Tasks**:
- [ ] Verify `@lod` validates level >= 0
- [ ] Verify `@lod` validates distance > 0
- [ ] Verify `@streamable` sets streaming thresholds
- [ ] Verify `@chunk` sets world partitioning
- [ ] Verify `@loading_priority` validates against VALID_LOADING_PRIORITIES
- [ ] Verify `@unloadable` marks for memory management

**Valid Priorities**:
- [ ] "critical" - always loaded
- [ ] "high" - prioritized loading
- [ ] "normal" - default
- [ ] "low" - deferred loading
- [ ] "background" - lowest priority

**Acceptance Criteria**:
- All 5 decorators follow 6-part pattern
- LOD level validation catches negative values
- Priority validation produces actionable errors

---

## T3.3: Validate Prefabs Decorators

**File**: `trinity/decorators/prefabs.py`

**Tasks**:
- [ ] Verify `@prefab` marks class as prefab template
- [ ] Verify `@extends` requires parent parameter
- [ ] Verify `@extends` validates parent exists
- [ ] Verify inheritance chain is tracked

**Error Message Test**:
```python
@extends()  # Should raise:
# ValueError: @extends: parent prefab is required
```

**Acceptance Criteria**:
- Both decorators produce correct steps
- Parent validation is enforced
- Inheritance chain queryable via introspection

---

## T3.4: Validate Network Extended Decorators

**File**: `trinity/decorators/network_extended.py`

**Tasks**:
- [ ] Verify `@interest` creates InterestConfig
- [ ] Verify `@interest` validates radius > 0
- [ ] Verify `@bandwidth_priority` creates BandwidthPriorityConfig
- [ ] Verify `@snapshot_interpolation` creates SnapshotInterpolationConfig
- [ ] Verify `@server_reconcile` creates ServerReconcileConfig
- [ ] Verify all configs are frozen dataclasses

**Config Fields**:
- [ ] InterestConfig: radius, priority, relevance_curve
- [ ] BandwidthPriorityConfig: priority, channel
- [ ] SnapshotInterpolationConfig: interpolation_time, extrapolation_time
- [ ] ServerReconcileConfig: rewind_frames, error_threshold

**Acceptance Criteria**:
- All 4 decorators create config dataclasses
- All configs are frozen and hashable
- All register in "network" registry

---

## T3.5: Validate RPC Decorator

**File**: `trinity/decorators/rpc.py`

**Tasks**:
- [ ] Verify `@rpc` validates authority against VALID_AUTHORITIES
- [ ] Verify `@rpc` accepts reliability setting
- [ ] Verify `@rpc` accepts channel configuration
- [ ] Verify function signature preservation

**Valid Authorities**:
- [ ] "server" - server can call
- [ ] "client" - client can call
- [ ] "owner" - owner can call
- [ ] "any" - anyone can call

**Acceptance Criteria**:
- Authority validation produces actionable error
- RPC metadata set on function
- Registers in "network" registry

---

## T3.6: Validate Save System Decorators

**File**: `trinity/decorators/save_system.py`

**Tasks**:
- [ ] Verify `@save_slot` configures save slot
- [ ] Verify `@atomic_save` enables atomic writes
- [ ] Verify `@cloud_sync` validates conflict_resolution
- [ ] Verify `@save_migration` tracks version

**Conflict Resolution Strategies**:
- [ ] "local_wins" - local overwrites remote
- [ ] "remote_wins" - remote overwrites local
- [ ] "merge" - attempt merge
- [ ] "prompt" - prompt user

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Conflict resolution validated against VALID_CONFLICT_RESOLUTION
- All register in "save" registry

---

## T3.7: Validate Transactions Decorators

**File**: `trinity/decorators/transactions.py`

**Tasks**:
- [ ] Verify `@transactional` validates isolation_level
- [ ] Verify `@undoable` enables undo stack

**Valid Isolation Levels**:
- [ ] "read_uncommitted"
- [ ] "read_committed"
- [ ] "repeatable_read"
- [ ] "serializable"

**Acceptance Criteria**:
- Isolation level validated against VALID_ISOLATION_LEVELS
- Both decorators produce correct steps
- Registers in appropriate registry

---

## T3.8: Validate Replay Decorators

**File**: `trinity/decorators/replay.py`

**Tasks**:
- [ ] Verify `@recorded` validates frequency > 0
- [ ] Verify `@replay_authority` validates source
- [ ] Verify `@keyframe` sets keyframe interval

**Valid Replay Sources**:
- [ ] "input" - record inputs only
- [ ] "state" - record state only
- [ ] "both" - record both

**Acceptance Criteria**:
- Frequency validation catches <= 0
- Source validated against VALID_REPLAY_SOURCES
- All register in "replay" registry

---

## Summary

| Task | File | Decorators | Lines |
|------|------|------------|-------|
| T3.1 | assets.py | 5 | 354 |
| T3.2 | lod_streaming.py | 5 | 292 |
| T3.3 | prefabs.py | 2 | 133 |
| T3.4 | network_extended.py | 4 | 331 |
| T3.5 | rpc.py | 1 | 118 |
| T3.6 | save_system.py | 4 | 236 |
| T3.7 | transactions.py | 2 | 137 |
| T3.8 | replay.py | 3 | 171 |

**Total**: 26 decorators, 1,772 lines
