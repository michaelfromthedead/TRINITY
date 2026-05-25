# PHASE 3 ARCHITECTURE: Assets, Networking & Persistence

## Scope

Resource management, networking, and persistence decorators:
- `assets.py` (354 lines) - Tier 8
- `lod_streaming.py` (292 lines)
- `prefabs.py` (133 lines)
- `network_extended.py` (331 lines)
- `rpc.py` (118 lines)
- `save_system.py` (236 lines)
- `transactions.py` (137 lines)
- `replay.py` (171 lines)

## Architecture Pattern: Config Dataclasses

Several files use frozen dataclasses for complex configurations:

```python
@dataclass(frozen=True)
class AssetConfig:
    path: str
    preload: bool = False
    cook_settings: Optional[CookConfig] = None

@dataclass(frozen=True)
class InterestConfig:
    radius: float
    priority: int
    relevance_curve: str
```

Benefits: Type safety, immutability, equality, hashability.

## Component: Assets Decorators

**File**: `trinity/decorators/assets.py` (354 lines, Tier 8)

**Decorators**: `@asset`, `@preload`, `@cook`, `@residency`, `@import_settings`

**Architecture**:
- Full ops + config dataclasses pattern
- AssetConfig, CookConfig, ResidencyConfig, ImportSettingsConfig
- Validates asset paths and cook settings

**Config Dataclasses**:
```python
@dataclass(frozen=True)
class AssetConfig:
    path: str
    preload: bool = False
    async_load: bool = True

@dataclass(frozen=True)
class CookConfig:
    platform: str
    compression: str = "default"
    strip_debug: bool = True
```

## Component: LOD & Streaming Decorators

**File**: `trinity/decorators/lod_streaming.py` (292 lines)

**Decorators**: `@lod`, `@streamable`, `@chunk`, `@loading_priority`, `@unloadable`

**Architecture**:
- LOD level validation (0-N)
- Streaming distance thresholds
- Chunk-based world partitioning
- Loading priority queues

**Validation**:
```python
VALID_LOADING_PRIORITIES = frozenset({"critical", "high", "normal", "low", "background"})

def _validate_lod(level=0, distance=0.0, **_):
    if level < 0:
        raise ValueError(f"@lod: level must be >= 0, got {level}")
```

## Component: Prefabs Decorators

**File**: `trinity/decorators/prefabs.py` (133 lines)

**Decorators**: `@prefab`, `@extends`

**Architecture**:
- Parent prefab validation
- Inheritance chain tracking
- Override detection

**Validation**:
```python
def _validate_extends(parent=None, **_):
    if parent is None:
        raise ValueError("@extends: parent prefab is required")
```

## Component: Network Extended Decorators

**File**: `trinity/decorators/network_extended.py` (331 lines)

**Decorators**: `@interest`, `@bandwidth_priority`, `@snapshot_interpolation`, `@server_reconcile`

**Architecture**:
- Config dataclasses for each decorator
- InterestConfig, BandwidthPriorityConfig
- SnapshotInterpolationConfig, ServerReconcileConfig

**Config Pattern**:
```python
@dataclass(frozen=True)
class InterestConfig:
    radius: float
    priority: int = 0
    relevance_curve: str = "linear"

@dataclass(frozen=True)
class ServerReconcileConfig:
    rewind_frames: int = 10
    error_threshold: float = 0.1
```

## Component: RPC Decorators

**File**: `trinity/decorators/rpc.py` (118 lines)

**Decorators**: `@rpc`

**Architecture**:
- Authority validation (server, client, owner, any)
- Reliability settings
- Channel configuration

**Valid Authorities**:
```python
VALID_AUTHORITIES = frozenset({"server", "client", "owner", "any"})
```

## Component: Save System Decorators

**File**: `trinity/decorators/save_system.py` (236 lines)

**Decorators**: `@save_slot`, `@atomic_save`, `@cloud_sync`, `@save_migration`

**Architecture**:
- Full conflict resolution strategies
- Migration version tracking
- Cloud sync with conflict handling

**Conflict Resolution**:
```python
VALID_CONFLICT_RESOLUTION = frozenset({"local_wins", "remote_wins", "merge", "prompt"})
```

## Component: Transactions Decorators

**File**: `trinity/decorators/transactions.py` (137 lines)

**Decorators**: `@transactional`, `@undoable`

**Architecture**:
- Isolation level validation
- Undo stack management
- Transaction boundaries

**Valid Isolation Levels**:
```python
VALID_ISOLATION_LEVELS = frozenset({"read_uncommitted", "read_committed", "repeatable_read", "serializable"})
```

## Component: Replay Decorators

**File**: `trinity/decorators/replay.py` (171 lines)

**Decorators**: `@recorded`, `@replay_authority`, `@keyframe`

**Architecture**:
- Recording frequency validation
- Authority source validation
- Keyframe interval settings

**Valid Sources**:
```python
VALID_REPLAY_SOURCES = frozenset({"input", "state", "both"})
```

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store asset/network metadata | All |
| `Op.REGISTER` | Register in "assets", "network", "save" | All |
| `Op.TRACK` | Track changes for replication | network_extended.py |
| `Op.VALIDATE` | Runtime validation | transactions.py |

## Key Decisions

1. **Config dataclasses**: Complex decorators use frozen dataclasses for type-safe configuration
2. **Validation constants**: All valid options as frozensets for O(1) lookup
3. **Registry separation**: assets, network, save registries keep domains isolated
4. **Conflict resolution**: Save system handles cloud sync conflicts explicitly
