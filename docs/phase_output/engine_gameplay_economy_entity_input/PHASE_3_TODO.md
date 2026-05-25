# PHASE 3 TODO: Input Module Testing

## Summary

Test coverage for engine/gameplay/input (~4,064 lines).

---

## T-INP-3.1: Keyboard Device Tests

**File**: `tests/input/test_device_keyboard.py`

### Tasks

- [ ] Test key_down() sets key state to True
- [ ] Test key_up() sets key state to False
- [ ] Test is_key_down() returns correct state
- [ ] Test modifier tracking (Shift, Ctrl, Alt, Meta)
- [ ] Test text buffer accumulates typed characters
- [ ] Test text buffer clears on consume
- [ ] Test multiple keys down simultaneously

### Acceptance Criteria

- All tests pass with `uv run pytest tests/input/test_device_keyboard.py`
- Modifier combinations verified (Ctrl+Shift+A)
- Text buffer encoding verified (UTF-8)

---

## T-INP-3.2: Mouse Device Tests

**File**: `tests/input/test_device_mouse.py`

### Tasks

- [ ] Test position updates correctly
- [ ] Test delta calculated from position change
- [ ] Test delta resets after read
- [ ] Test scroll accumulates correctly
- [ ] Test sensitivity multiplies delta
- [ ] Test capture mode prevents position updates
- [ ] Test button states track correctly

### Acceptance Criteria

- All tests pass
- Delta calculation: delta = (new_pos - old_pos) * sensitivity
- Capture mode: position frozen, delta still calculated

---

## T-INP-3.3: Gamepad Device Tests

**File**: `tests/input/test_device_gamepad.py`

### Tasks

- [ ] Test axis values in range [-1, 1]
- [ ] Test trigger values in range [0, 1]
- [ ] Test button states track correctly
- [ ] Test rumble scheduling sets motor values
- [ ] Test rumble duration expires correctly
- [ ] Test multiple axes independent

### Acceptance Criteria

- All tests pass
- Axis clamping verified at boundaries
- Rumble: verify motor values clear after duration

---

## T-INP-3.4: Motion Device Tests

**File**: `tests/input/test_device_motion.py`

### Tasks

- [ ] Test gyroscope values set correctly
- [ ] Test gyroscope sensitivity multiplies values
- [ ] Test smoothing alpha = 0 passes through unchanged
- [ ] Test smoothing alpha = 1 holds previous value
- [ ] Test smoothing interpolates correctly
- [ ] Test accelerometer values set correctly
- [ ] Test orientation quaternion normalizes

### Acceptance Criteria

- All tests pass
- Smoothing formula verified: out = prev * alpha + new * (1-alpha)
- Quaternion: magnitude = 1.0 after normalization

---

## T-INP-3.5: Hold Trigger Tests

**File**: `tests/input/test_trigger_hold.py`

### Tasks

- [ ] Test initial state is NONE
- [ ] Test press starts STARTED state
- [ ] Test holding transitions to ONGOING
- [ ] Test hold_time accumulates with delta_time
- [ ] Test COMPLETED fires when hold_time >= hold_duration
- [ ] Test release before completion fires CANCELLED
- [ ] Test progress value is hold_time / hold_duration
- [ ] Test state resets after release

### Acceptance Criteria

- All tests pass
- State machine: NONE -> STARTED -> ONGOING -> COMPLETED
- Cancel path: STARTED/ONGOING -> CANCELLED -> NONE
- Progress clamped to [0, 1]

---

## T-INP-3.6: Combo Trigger Tests

**File**: `tests/input/test_trigger_combo.py`

### Tasks

- [ ] Test combo requires inputs in order
- [ ] Test out-of-order input resets combo
- [ ] Test combo timeout resets progress
- [ ] Test COMPLETED fires when sequence complete
- [ ] Test partial sequence tracked correctly
- [ ] Test combo with repeated input (e.g., A, A, B)

### Acceptance Criteria

- All tests pass
- Timeout: combo resets if gap > timeout
- Order: A, B, C must be exactly in that order

---

## T-INP-3.7: Radial Dead Zone Tests

**File**: `tests/input/test_dead_zone_radial.py`

### Tasks

- [ ] Test magnitude < dead_zone -> output (0, 0)
- [ ] Test magnitude = dead_zone -> output (0, 0)
- [ ] Test magnitude > dead_zone -> rescaled output
- [ ] Test magnitude > outer_zone -> normalized to unit circle
- [ ] Test rescaling formula is correct
- [ ] Test zero magnitude does not cause division by zero
- [ ] Test negative values handled correctly

### Acceptance Criteria

- All tests pass
- Rescaling: out_mag = (in_mag - dead_zone) / (outer_zone - dead_zone)
- Zero division: explicit test with x=0, y=0

---

## T-INP-3.8: Response Curve Tests

**File**: `tests/input/test_response_curves.py`

### Tasks

- [ ] Test Linear: output = input
- [ ] Test Power: output = sign(input) * abs(input)^exponent
- [ ] Test Exponential: output = sign(input) * (exp(abs(input)*k) - 1) / (exp(k) - 1)
- [ ] Test S-curve: verify tanh-based mapping with midpoint and steepness
- [ ] Test Step: output snaps to discrete levels
- [ ] Test all curves: input=0 -> output=0
- [ ] Test all curves: input=1 -> output=1
- [ ] Test all curves: input=-1 -> output=-1

### Acceptance Criteria

- All tests pass with pytest.approx(tolerance=1e-6)
- Boundary values exact
- S-curve: verify midpoint behavior

---

## T-INP-3.9: Input Smoother Tests

**File**: `tests/input/test_input_smoother.py`

### Tasks

- [ ] Test MovingAverage with window size 1 -> pass-through
- [ ] Test MovingAverage with window size N -> averages N values
- [ ] Test Exponential with alpha=0 -> pass-through
- [ ] Test Exponential with alpha=1 -> holds previous
- [ ] Test DoubleExponential tracks velocity
- [ ] Test reset() clears history

### Acceptance Criteria

- All tests pass
- Window: verify buffer fills correctly
- Exponential: verify formula matches ADR-INP-5

---

## T-INP-3.10: Device Manager Hot-Plug Tests

**File**: `tests/input/test_device_manager.py`

### Tasks

- [ ] Test register_device() adds device to registry
- [ ] Test unregister_device() removes device
- [ ] Test on_device_connected callback fires
- [ ] Test on_device_disconnected callback fires
- [ ] Test get_device_by_type() returns correct device
- [ ] Test get_all_devices() returns all registered
- [ ] Test duplicate registration raises error
- [ ] Test reset_instance() clears all devices

### Acceptance Criteria

- All tests pass
- Callbacks verified with mock listeners
- Singleton behavior verified
