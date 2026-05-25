# PHASE 1 TODO: Audio Mixing Core

**RDC Phase Task Breakdown**
**Phase**: Foundation Mixing Infrastructure

---

## Task 1.1: MixBus Implementation

**File**: `engine/audio/mixing/mix_bus.py`
**Estimated Lines**: ~750

### Subtasks

- [ ] Define `MixBusState` dataclass with volume, filters, mute/solo
- [ ] Implement `MixBus.__init__` with parent/children tracking
- [ ] Implement `add_child()` / `remove_child()` with thread safety
- [ ] Implement `set_volume_db()` / `set_volume_linear()` conversion
- [ ] Implement `read_acc_buffer()` for input accumulation
- [ ] Implement `write_output()` for output accumulation
- [ ] Implement `process_audio()` with volume and filter application
- [ ] Implement `has_effects()` and DSP chain integration
- [ ] Add `_low_pass_filter` and `_high_pass_filter` instances
- [ ] Implement `create_default_hierarchy()` factory function

### Acceptance Criteria

- Volume stored in dB, converted to linear for processing
- Filters only process when enabled
- DSP chain applied after volume, before output
- Thread-safe with `_lock` and `_acc_lock`

---

## Task 1.2: Mixer Core Implementation

**File**: `engine/audio/mixing/mixer.py`
**Estimated Lines**: ~1,100

### Subtasks

- [ ] Define `MixerConfig` dataclass with buffer size, sample rate
- [ ] Implement `Mixer.__init__` with bus registry and master bus
- [ ] Implement `register_bus()` / `unregister_bus()`
- [ ] Implement `get_bus()` lookup by name
- [ ] Implement `_compute_processing_order()` DFS post-order
- [ ] Implement `tick()` 8-stage pipeline:
  - [ ] Stage 1: Compute DFS order
  - [ ] Stage 2: Clear accumulation buffers
  - [ ] Stage 2b: Generate source impulses
  - [ ] Stage 3: Bottom-up processing
  - [ ] Stage 4: PRE_FADER sends
  - [ ] Stage 5: Process through bus
  - [ ] Stage 5b: POST_FADER sends
  - [ ] Stage 6: Ducking (placeholder for Phase 2)
  - [ ] Stage 7: HDR/Sidechain (placeholder for Phase 2)
  - [ ] Stage 8: Hard clip
- [ ] Implement `_hard_clip()` using `np.clip(-1.0, 1.0)`
- [ ] Add thread-safe source impulse queue

### Acceptance Criteria

- DFS order processes children before parents
- Master bus output is final tick result
- Tick returns NumPy array of samples
- Thread-safe bus registration/unregistration

---

## Task 1.3: BusRouter Implementation

**File**: `engine/audio/mixing/bus_routing.py`
**Estimated Lines**: ~490

### Subtasks

- [ ] Define `AuxSendType` enum (PRE_FADER, POST_FADER)
- [ ] Define `AuxSend` dataclass with from_bus, to_bus, type, gain
- [ ] Implement `BusRouter.__init__` with send registry
- [ ] Implement `create_send()` with validation
- [ ] Implement `remove_send()` with cleanup
- [ ] Implement `set_send_gain()` with bounds checking
- [ ] Implement `get_sends_from()` query
- [ ] Implement `get_sends_to()` query
- [ ] Implement `process_sends()` called from Mixer tick stages 4 and 5b

### Acceptance Criteria

- PRE_FADER sends process before bus volume
- POST_FADER sends process after bus volume
- Send gain applied during copy
- No circular sends allowed

---

## Task 1.4: Configuration Module

**File**: `engine/audio/mixing/config.py`
**Estimated Lines**: ~250

### Subtasks

- [ ] Define audio constants (SAMPLE_RATE, BUFFER_SIZE, etc.)
- [ ] Define mixing constants (DEFAULT_VOLUME_DB, etc.)
- [ ] Define bus hierarchy constants (DEFAULT_BUSES list)
- [ ] Implement `db_to_linear()` conversion function
- [ ] Implement `linear_to_db()` conversion function
- [ ] Implement `clamp()` utility function

### Acceptance Criteria

- All constants documented with units
- Conversion functions handle edge cases (0.0, -inf)
- Constants used consistently across modules

---

## Task 1.5: Module Init and Exports

**File**: `engine/audio/mixing/__init__.py`
**Estimated Lines**: ~220

### Subtasks

- [ ] Import and export all public classes
- [ ] Document module-level API
- [ ] Provide `create_mixer()` factory function
- [ ] Provide `create_bus()` factory function

### Acceptance Criteria

- `from engine.audio.mixing import Mixer, MixBus` works
- Factory functions documented with examples

---

## Dependencies

- NumPy for buffer operations
- `engine/audio/dsp/dsp_graph` for effect chains
- `engine/audio/dsp/filters` for LP/HP filters

---

## Verification

1. Unit tests for each class
2. Integration test: create hierarchy, route audio, verify output
3. Thread safety test: concurrent tick and bus modification
4. Performance test: tick latency under load
