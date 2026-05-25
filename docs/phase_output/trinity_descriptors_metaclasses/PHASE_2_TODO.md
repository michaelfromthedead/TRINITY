# PHASE 2 TODO: Metaclass System

## Objective

Verify and validate the metaclass system implementation.

---

## T-2.1: EngineMeta Base

**File**: `trinity/metaclasses/engine_meta.py`

### Tasks
- [ ] Verify classes register in `_engine_types`
- [ ] Verify `_metaclass_steps` list created on each class
- [ ] Verify thread safety with concurrent class creation
- [ ] Test `__repr__` format for engine types
- [ ] Test `clear_registry()` clears `_engine_types`

### Acceptance Criteria
- All engine types trackable via registry
- Step recording infrastructure present on all classes
- Thread-safe under concurrent access

---

## T-2.2: ComponentMeta

**File**: `trinity/metaclasses/component_meta.py`

### Tasks
- [ ] Verify unique `_component_id` generation
- [ ] Test field processing from type hints
- [ ] Test field processing from `Annotated` types
- [ ] Verify descriptor installation from markers
- [ ] Test mutable default detection (list)
- [ ] Test mutable default detection (dict)
- [ ] Test mutable default detection (set)
- [ ] Verify mutable default rejection raises error
- [ ] Test pool management `return_to_pool`
- [ ] Test pool management `pool_stats`
- [ ] Test budget enforcement `max_instances`
- [ ] Test instance count tracking `_instance_count`
- [ ] Test layout optimization `get_layout_arrays`
- [ ] Verify Rust registration via `_omega.type_register`
- [ ] Verify Foundation registry integration
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Component IDs are unique across all components
- Field hints correctly parsed and processed
- Mutable defaults rejected with clear error message
- Pool and budget management functional

---

## T-2.3: SystemMeta

**File**: `trinity/metaclasses/system_meta.py`

### Tasks
- [ ] Test phase assignment via `SystemPhase` enum
- [ ] Test `@reads` decorator parsing
- [ ] Test `@writes` decorator parsing
- [ ] Test dependency analysis from read/write declarations
- [ ] Verify `_can_parallelize` detection
- [ ] Test topological sort `get_phase_order`
- [ ] Test `get_parallel_groups` grouping
- [ ] Test resource conflict detection
- [ ] Test hot reload via `hot_reload`
- [ ] Test `reload_system`
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Execution order respects dependencies
- Parallel groups contain only non-conflicting systems
- Hot reload updates system without restart

---

## T-2.4: StateMeta

**File**: `trinity/metaclasses/state_meta.py`

### Tasks
- [ ] Test per-machine state registration
- [ ] Test `can_transition` validation
- [ ] Test `validate_transitions` batch validation
- [ ] Test invalid transition rejection
- [ ] Test hierarchical state `register_substate`
- [ ] Test `get_substates`
- [ ] Test hierarchy cycle detection
- [ ] Test history `record_transition`
- [ ] Test `get_previous_state`
- [ ] Test enter/exit hooks
- [ ] Test `clear_registry()`

### Acceptance Criteria
- State machines enforce valid transitions
- Hierarchical states work correctly
- Cycles in hierarchy detected and rejected

---

## T-2.5: EventMeta

**File**: `trinity/metaclasses/event_meta.py`

### Tasks
- [ ] Test data-only validation passes for valid events
- [ ] Test data-only validation fails for events with methods
- [ ] Test inheritance tracking `_event_parent_ids`
- [ ] Test channel-based filtering
- [ ] Test event pooling `acquire`
- [ ] Test event pooling `release`
- [ ] Test event pooling `pool_stats`
- [ ] Test serialization
- [ ] Test deserialization
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Events are data-only (methods rejected)
- Pooling reduces allocation
- Serialization round-trips correctly

---

## T-2.6: AssetMeta

**File**: `trinity/metaclasses/asset_meta.py`

### Tasks
- [ ] Test extension-based type mapping
- [ ] Test conflict detection for duplicate extensions
- [ ] Test priority-based loading queue
- [ ] Test async loading
- [ ] Test hot reload file watching `watch`
- [ ] Test `check_changes` detection
- [ ] Test `get_load_order` dependency ordering
- [ ] Test circular dependency detection
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Extensions map to correct asset types
- Conflicts detected with warning
- Load order respects dependencies

---

## T-2.7: ProtocolMeta

**File**: `trinity/metaclasses/protocol_meta.py`

### Tasks
- [ ] Test version validation
- [ ] Test version compatibility checking
- [ ] Test message type registration
- [ ] Test version-specific decoder registration
- [ ] Test `negotiate_version` between client/server
- [ ] Test migration path generation
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Version negotiation finds highest compatible version
- Migration paths generated correctly

---

## T-2.8: ResourceMeta

**File**: `trinity/metaclasses/resource_meta.py`

### Tasks
- [ ] Test singleton pattern enforcement
- [ ] Test second instantiation blocked
- [ ] Test dependency declaration
- [ ] Test `initialize_all` order
- [ ] Test lazy resource support
- [ ] Test shutdown with error handling
- [ ] Test `clear_registry()`

### Acceptance Criteria
- Only one instance per resource type
- Initialization order respects dependencies
- Shutdown completes even if some resources error

---

## T-2.9: Cross-Metaclass Integration

### Tasks
- [ ] Test ComponentMeta installs descriptors from annotations
- [ ] Test SystemMeta can read ComponentMeta field metadata
- [ ] Test EventMeta serialization includes component references
- [ ] Verify all metaclasses share EngineMeta registry
- [ ] Test combined step recording from multiple metaclasses

### Acceptance Criteria
- Metaclasses interoperate correctly
- Step recording captures actions from all layers
