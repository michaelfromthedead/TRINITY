# PHASE 1 TODO: Descriptor System

## Objective

Verify and validate the descriptor system implementation.

---

## T-1.1: BaseDescriptor Protocol Compliance

**File**: `trinity/descriptors/base.py`

### Tasks
- [ ] Verify `__get__` returns descriptor instance when accessed on class (not instance)
- [ ] Verify `__get__` returns value when accessed on instance
- [ ] Verify `__set__` stores value correctly
- [ ] Verify `__delete__` removes value from storage
- [ ] Verify `__set_name__` captures attribute name and owner class
- [ ] Test lifecycle hooks fire in correct order: pre_get -> get -> post_get
- [ ] Test lifecycle hooks fire in correct order: pre_set -> set -> post_set
- [ ] Verify read tracking via ContextVar populates correctly

### Acceptance Criteria
- All protocol methods work per Python descriptor specification
- Hooks can be overridden in subclasses
- Read tracking captures field access within context

---

## T-1.2: DescriptorComposer Validation

**File**: `trinity/descriptors/composer.py`

### Tasks
- [ ] Test composition of 2 descriptors
- [ ] Test composition of 5+ descriptors
- [ ] Verify exclusion rules are enforced
- [ ] Verify accepts_inner rules are enforced
- [ ] Verify accepts_outer rules are enforced
- [ ] Verify topological order (innermost first)
- [ ] Test `explain_chain()` output format
- [ ] Verify steps are collected from all descriptors in chain

### Acceptance Criteria
- Invalid compositions raise descriptive exceptions
- Valid compositions produce working composite descriptor
- explain_chain provides human-readable output

---

## T-1.3: Networking Descriptors

**File**: `trinity/descriptors/networking.py`

### Tasks
- [ ] Test NetworkedDescriptor queues updates correctly
- [ ] Test authority rules (server/client/owner)
- [ ] Test InterpolatedDescriptor linear interpolation
- [ ] Test InterpolatedDescriptor Hermite interpolation
- [ ] Test PredictedDescriptor prediction
- [ ] Test PredictedDescriptor rollback on server correction
- [ ] Test ThrottledNetworkDescriptor token bucket algorithm
- [ ] Test throttle rate limiting behavior

### Acceptance Criteria
- Updates queue for network replication
- Interpolation produces smooth values between snapshots
- Prediction matches expected behavior with rollback

---

## T-1.4: Tracking Descriptors

**File**: `trinity/descriptors/tracking.py`

### Tasks
- [ ] Test TrackedDescriptor set-based dirty tracking
- [ ] Test TrackedDescriptor bitmask dirty tracking
- [ ] Test VersionedDescriptor increments on change
- [ ] Test VersionedDescriptor does not increment on same-value write
- [ ] Test DiffDescriptor shallow comparison
- [ ] Test DiffDescriptor deep comparison
- [ ] Test DiffDescriptor custom comparison function
- [ ] Verify Foundation tracker integration

### Acceptance Criteria
- Dirty flags set/clear correctly
- Version counter increments only on actual changes
- Previous values captured for diff computation

---

## T-1.5: Validation Descriptors

**File**: `trinity/descriptors/validation.py`

### Tasks
- [ ] Test ValidatedDescriptor with passing validator
- [ ] Test ValidatedDescriptor with failing validator
- [ ] Test RangeDescriptor clamp mode
- [ ] Test RangeDescriptor raise mode
- [ ] Test TypeDescriptor type enforcement
- [ ] Test TypeDescriptor coercion
- [ ] Test ChoiceDescriptor valid value
- [ ] Test ChoiceDescriptor invalid value
- [ ] Test PatternDescriptor matching pattern
- [ ] Test PatternDescriptor non-matching pattern

### Acceptance Criteria
- Invalid values raise appropriate exceptions
- Clamp mode constrains values without exception
- Type coercion converts compatible types

---

## T-1.6: Persistence Descriptors

**File**: `trinity/descriptors/persistence.py`

### Tasks
- [ ] Test SerializableDescriptor encode
- [ ] Test SerializableDescriptor decode
- [ ] Test TransientDescriptor excluded from serialization
- [ ] Test MigratedDescriptor field rename
- [ ] Test EncryptedDescriptor encryption/decryption
- [ ] Test EncryptedDescriptor with custom encrypt function

### Acceptance Criteria
- Serialization round-trips values correctly
- Transient fields not included in serialized output
- Migration handles renamed fields

---

## T-1.7: RustStorageDescriptor

**File**: `trinity/descriptors/rust_storage.py`

### Tasks
- [ ] Test storage via _omega when available
- [ ] Test fallback to __dict__ when _omega unavailable
- [ ] Test float -> f32 mapping
- [ ] Test int -> i32 mapping
- [ ] Test bool -> u8 mapping
- [ ] Test str -> string mapping
- [ ] Verify graceful handling of _omega import failure

### Acceptance Criteria
- Rust storage works when _omega present
- Python fallback works when _omega absent
- Type mapping correct for all supported types

---

## T-1.8: Utility Descriptors

### CachedDescriptor (caching.py)
- [ ] Test TTL expiration
- [ ] Test cache invalidation
- [ ] Test ComputedDescriptor read-only behavior

### AsyncLoadDescriptor (async_descriptors.py)
- [ ] Test lazy initialization
- [ ] Test async loading state machine (pending, loading, loaded, error)

### AtomicDescriptor (atomic.py)
- [ ] Test compare_and_swap success
- [ ] Test compare_and_swap failure (concurrent modification)
- [ ] Test thread safety under concurrent access

### RateLimitedDescriptor (rate_limiting.py)
- [ ] Test raise policy
- [ ] Test drop policy
- [ ] Test rate calculation

### CompressedDescriptor (compressed.py)
- [ ] Test zlib compression/decompression
- [ ] Test lz4 compression/decompression (if available)

### ObservableDescriptor (observable.py)
- [ ] Test observer registration
- [ ] Test observer notification on change
- [ ] Test observer removal

### Debug Descriptors (debug.py)
- [ ] Test ProfiledDescriptor timing capture
- [ ] Test LoggedDescriptor log output
- [ ] Test WatchedDescriptor breakpoint behavior

### Acceptance Criteria
- Each utility descriptor performs its documented function
- Thread-safe descriptors handle concurrent access correctly
- Debug descriptors produce expected output
