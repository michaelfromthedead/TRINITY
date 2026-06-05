# PHASE 1 ARCHITECTURE: Descriptor System

## Phase Scope

The descriptor system (`trinity/descriptors/`) implementing composable field-level behaviors.

## Components

### 1.1 BaseDescriptor (base.py - 382 lines)

Core foundation for all descriptors:

- Full Python descriptor protocol (`__get__`, `__set__`, `__delete__`, `__set_name__`)
- Lifecycle hooks (`pre_get`, `post_get`, `pre_set`, `post_set`)
- Read-tracking via `ContextVar` for incremental computation
- Composition metadata (`inner`, `accepts_inner`, `accepts_outer`, `excludes`)
- Foundation integration (provenance tracking)

### 1.2 DescriptorComposer (composer.py - 164 lines)

Safe composition of multiple descriptors:

- Chain validation (exclusions, accepts_inner/outer checks)
- Topological composition from innermost to outermost
- Step collection across the entire chain
- `explain_chain()` for debugging composition

### 1.3 Networking Descriptors (networking.py - 342 lines)

| Descriptor | Purpose |
|------------|---------|
| NetworkedDescriptor | Queue updates for replication with authority rules |
| InterpolatedDescriptor | Linear/Hermite interpolation between network snapshots |
| PredictedDescriptor | Client-side prediction with rollback support |
| ThrottledNetworkDescriptor | Rate limiting with token bucket algorithm |

### 1.4 Tracking Descriptors (tracking.py - 302 lines)

| Descriptor | Purpose |
|------------|---------|
| TrackedDescriptor | Dirty flags (set-based and bitmask tracking) |
| VersionedDescriptor | Per-field version counter |
| DiffDescriptor | Previous value storage with shallow/deep/custom comparison |

Foundation integration for central tracker and EventLog.

### 1.5 Validation Descriptors (validation.py - 296 lines)

| Descriptor | Purpose |
|------------|---------|
| ValidatedDescriptor | Custom validator functions |
| RangeDescriptor | Numeric clamping with raise/clamp modes |
| TypeDescriptor | Runtime type enforcement with optional coercion |
| ChoiceDescriptor | Enum-like value constraints |
| PatternDescriptor | Regex validation for strings |

### 1.6 Persistence Descriptors (persistence.py - 237 lines)

| Descriptor | Purpose |
|------------|---------|
| SerializableDescriptor | Custom encode/decode with format tags |
| TransientDescriptor | Skip field on save/load |
| MigratedDescriptor | Field rename migration across versions |
| EncryptedDescriptor | Encrypt at rest (base64 default, custom fn support) |

### 1.7 RustStorageDescriptor (rust_storage.py - 135 lines)

Bridges Python and Rust component stores:

- Routes reads/writes to Rust via `_omega` module
- Falls back to `__dict__` storage when Rust unavailable
- Type mapping: `float->f32`, `int->i32`, `bool->u8`, `str->string`

### 1.8 Utility Descriptors

| File | Descriptors |
|------|-------------|
| caching.py (159 lines) | CachedDescriptor (TTL-based), ComputedDescriptor (read-only computed) |
| async_descriptors.py (124 lines) | LazyDescriptor (deferred init), AsyncLoadDescriptor (async state machine) |
| atomic.py (116 lines) | AtomicDescriptor (thread-safe with compare_and_swap) |
| rate_limiting.py (121 lines) | RateLimitedDescriptor (throttle writes, raise/drop policy) |
| compressed.py (109 lines) | CompressedDescriptor (zlib/lz4 compression) |
| observable.py (125 lines) | ObservableDescriptor (observer pattern with callbacks) |
| debug.py (144 lines) | ProfiledDescriptor, LoggedDescriptor, WatchedDescriptor |

## Architecture Decisions

### AD-1.1: Composition Over Inheritance

Descriptors compose via wrapping rather than inheritance:

```python
# Composition chain (outermost to innermost):
Networked -> Tracked -> Validated -> Storage
```

Rationale: Each descriptor handles one concern. Composition allows arbitrary combinations without a combinatorial explosion of subclasses.

### AD-1.2: Lifecycle Hooks

All descriptors support `pre_get`, `post_get`, `pre_set`, `post_set` hooks.

Rationale: Cross-cutting concerns (logging, metrics, validation) can be injected without modifying the descriptor itself.

### AD-1.3: ContextVar for Read Tracking

Read tracking uses `ContextVar` rather than thread-local or instance state.

Rationale: `ContextVar` is async-safe and properly scoped to the current execution context, enabling incremental computation patterns.

### AD-1.4: Graceful Rust Fallback

RustStorageDescriptor checks `_omega` availability and falls back to `__dict__`.

Rationale: Development and testing can proceed without Rust compilation. Production uses Rust for performance.

## Data Flow

```
User Code
    |
    v
[Descriptor.__get__ / __set__]
    |
    +-- pre_get/pre_set hook
    |
    +-- Inner descriptor chain (if composed)
    |       |
    |       v
    |   [...recursive...]
    |       |
    |       v
    |   Storage (RustStorage or __dict__)
    |
    +-- post_get/post_set hook
    |
    v
Return to User
```

## Composition Validation Rules

The DescriptorComposer enforces:

1. **Exclusions**: If A excludes B, they cannot be in the same chain
2. **accepts_inner**: A descriptor can declare what types it accepts as inner
3. **accepts_outer**: A descriptor can declare what types it accepts as outer
4. **Topological order**: Chain is built innermost-first
