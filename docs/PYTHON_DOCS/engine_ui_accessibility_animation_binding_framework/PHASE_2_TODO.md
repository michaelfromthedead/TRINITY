# PHASE 2 TODO: Animation Module

## Summary

Verify and test the 5 animation files (~4,233 lines) for correct implementation of state machines, tweening, keyframes, triggers, and easing.

---

## T1: Easing Functions

**File**: `engine/ui/animation/easing.py`

### T1.1: Test Linear Easing
- [ ] f(0) = 0
- [ ] f(1) = 1
- [ ] f(0.5) = 0.5

**Acceptance**: Linear returns input unchanged.

### T1.2: Test Quad Easings
- [ ] ease_in_quad(0.5) = 0.25
- [ ] ease_out_quad(0.5) = 0.75
- [ ] ease_in_out_quad(0.5) = 0.5

**Acceptance**: Quad easings match t^2 formula.

### T1.3: Test Cubic Easings
- [ ] ease_in_cubic(0.5) = 0.125
- [ ] ease_out_cubic(0.5) = 0.875
- [ ] ease_in_out_cubic(0.5) = 0.5

**Acceptance**: Cubic easings match t^3 formula.

### T1.4: Test Bounce Easing
- [ ] Bounce at end (out) creates 4 bounces
- [ ] Bounces decrease in amplitude
- [ ] f(0) = 0, f(1) = 1

**Acceptance**: Bounce physics are correct.

### T1.5: Test Elastic Easing
- [ ] Overshoot at end (out)
- [ ] Oscillation dampens
- [ ] f(0) = 0, f(1) = 1

**Acceptance**: Elastic spring physics are correct.

### T1.6: Test CubicBezier
- [ ] Standard ease (0.25, 0.1, 0.25, 1.0) matches CSS ease
- [ ] ease-in-out (0.42, 0, 0.58, 1) matches CSS
- [ ] Newton-Raphson converges within 8 iterations
- [ ] Handles edge case (vertical tangent)

**Acceptance**: CubicBezier matches CSS timing functions.

---

## T2: Tween System

**File**: `engine/ui/animation/tween.py`

### T2.1: Test Numeric Interpolation
- [ ] Tween from 0 to 100, at t=0.5 → 50
- [ ] Tween from -10 to 10, at t=0.5 → 0
- [ ] Float precision maintained

**Acceptance**: Numeric interpolation is exact.

### T2.2: Test Tuple Interpolation
- [ ] Tween from (0,0) to (100,100), at t=0.5 → (50,50)
- [ ] RGB color tuple interpolation works
- [ ] Different tuple lengths raise error

**Acceptance**: Tuple interpolation element-wise.

### T2.3: Test Dict Interpolation
- [ ] Tween {x:0, y:0} to {x:100, y:100}, at t=0.5 → {x:50, y:50}
- [ ] Nested dicts work recursively
- [ ] Missing keys raise error

**Acceptance**: Dict interpolation key-wise.

### T2.4: Test Delay
- [ ] Tween with delay=1.0 doesn't start until t=1.0
- [ ] on_start called after delay
- [ ] Progress stays at 0 during delay

**Acceptance**: Delay delays the start.

### T2.5: Test Repeat
- [ ] repeat=0: plays once
- [ ] repeat=2: plays 3 times total
- [ ] repeat=-1: plays forever

**Acceptance**: Repeat count is honored.

### T2.6: Test Yoyo
- [ ] yoyo=False: 0→1, 0→1, 0→1
- [ ] yoyo=True: 0→1, 1→0, 0→1

**Acceptance**: Yoyo reverses direction each repeat.

### T2.7: Test Callbacks
- [ ] on_start called once at start
- [ ] on_update called each frame with current value
- [ ] on_complete called once at end

**Acceptance**: All callbacks fire at correct times.

### T2.8: Test TweenSequence
- [ ] Tweens play in order
- [ ] Second tween starts when first completes
- [ ] Total duration is sum of individual durations

**Acceptance**: Sequential execution is correct.

### T2.9: Test TweenGroup
- [ ] All tweens start simultaneously
- [ ] Group completes when longest tween completes
- [ ] Values update independently

**Acceptance**: Parallel execution is correct.

---

## T3: Keyframe Animation

**File**: `engine/ui/animation/keyframe.py`

### T3.1: Test Keyframe Sampling
- [ ] At keyframe time: exact value
- [ ] Between keyframes: interpolated value
- [ ] Before first keyframe: first value
- [ ] After last keyframe: last value

**Acceptance**: Sampling handles all time positions.

### T3.2: Test Multi-Track Animation
- [ ] Multiple tracks update simultaneously
- [ ] Tracks can target different properties
- [ ] Independent easing per track

**Acceptance**: Multi-track animations work correctly.

### T3.3: Test Loop Mode ONCE
- [ ] Animation plays once
- [ ] Stops at end
- [ ] Progress stays at 1.0

**Acceptance**: ONCE mode doesn't repeat.

### T3.4: Test Loop Mode LOOP
- [ ] Animation repeats from start
- [ ] Time wraps (t=1.1 → t=0.1)
- [ ] Plays indefinitely

**Acceptance**: LOOP mode wraps correctly.

### T3.5: Test Loop Mode PING_PONG
- [ ] Forward: 0 → 1
- [ ] Backward: 1 → 0
- [ ] Repeats alternating

**Acceptance**: PING_PONG reverses direction.

### T3.6: Test Seek
- [ ] seek(0.5) jumps to middle
- [ ] seek(0) resets to start
- [ ] seek(1) jumps to end

**Acceptance**: Seek positions animation correctly.

### T3.7: Test Keyframe Insertion
- [ ] Insert keyframe at t=0.5
- [ ] Existing keyframes preserved
- [ ] New keyframe participates in interpolation

**Acceptance**: Dynamic keyframe insertion works.

---

## T4: Animator State Machine

**File**: `engine/ui/animation/animator.py`

### T4.1: Test State Transitions
- [ ] Transition fires when condition is true
- [ ] Transition duration respected
- [ ] Callbacks fire (on_transition_start, on_transition_end)

**Acceptance**: Transitions work correctly.

### T4.2: Test Conditional Transitions
- [ ] Transition only when condition returns true
- [ ] Multiple outgoing transitions: first match wins
- [ ] No match: stay in current state

**Acceptance**: Conditions control transitions.

### T4.3: Test Layer Blending - Override
- [ ] Higher priority layer replaces lower
- [ ] Weight 1.0 = full override
- [ ] Weight 0.0 = no effect

**Acceptance**: Override mode replaces values.

### T4.4: Test Layer Blending - Additive
- [ ] Upper layer adds to lower
- [ ] Weight scales addition
- [ ] Works with multiple layers

**Acceptance**: Additive mode sums values.

### T4.5: Test Layer Blending - Multiply
- [ ] Upper layer scales lower
- [ ] Weight scales the scale factor
- [ ] 1.0 * 1.0 = 1.0 (no change)

**Acceptance**: Multiply mode scales values.

### T4.6: Test Layer Blending - Average
- [ ] Weighted average of layers
- [ ] Equal weights = midpoint
- [ ] Single layer = that layer's value

**Acceptance**: Average mode blends correctly.

### T4.7: Test Parameter-Driven Transitions
- [ ] Set parameter value
- [ ] Transition condition reads parameter
- [ ] Parameter change triggers transition

**Acceptance**: Parameters drive state machine.

---

## T5: Animation Triggers

**File**: `engine/ui/animation/triggers.py`

### T5.1: Test StateTrigger
- [ ] Triggers on widget HOVER state
- [ ] Triggers on widget PRESSED state
- [ ] Triggers on widget FOCUSED state
- [ ] False when state doesn't match

**Acceptance**: State triggers detect widget states.

### T5.2: Test EventTrigger
- [ ] Triggers when event fires
- [ ] Correct event type filter
- [ ] Cleanup when trigger disposed

**Acceptance**: Event triggers respond to events.

### T5.3: Test PropertyTrigger
- [ ] Triggers when property value matches condition
- [ ] Works with equality check
- [ ] Works with custom predicate

**Acceptance**: Property triggers detect value changes.

### T5.4: Test DataTrigger
- [ ] Triggers when bound value equals target
- [ ] Two-way binding updates trigger
- [ ] Works with converters

**Acceptance**: Data triggers integrate with binding.

### T5.5: Test MultiTrigger AND
- [ ] True only when ALL sub-triggers true
- [ ] False if any sub-trigger false

**Acceptance**: AND logic is correct.

### T5.6: Test MultiTrigger OR
- [ ] True if ANY sub-trigger true
- [ ] False only when all false

**Acceptance**: OR logic is correct.

### T5.7: Test MultiTrigger XOR
- [ ] True if EXACTLY ONE sub-trigger true
- [ ] False if zero or multiple true

**Acceptance**: XOR logic is correct.

### T5.8: Test MultiTrigger NAND/NOR
- [ ] NAND = NOT AND
- [ ] NOR = NOT OR

**Acceptance**: Inverted logic is correct.

---

## T6: Integration Tests

### T6.1: Trigger Starts Animation
- [ ] StateTrigger(HOVER) starts fade-in tween
- [ ] EventTrigger(click) starts bounce animation
- [ ] PropertyTrigger starts keyframe animation

**Acceptance**: Triggers correctly start animations.

### T6.2: Animation Respects Motion Preferences
- [ ] REDUCE motion multiplies duration
- [ ] NONE motion skips to end
- [ ] NO_PREFERENCE runs normally

**Acceptance**: Accessibility integration works.

### T6.3: Nested Animations
- [ ] Animator state contains keyframe animation
- [ ] Keyframe animation uses easing functions
- [ ] Layer blending affects keyframe output

**Acceptance**: Animation systems compose correctly.

---

## Completion Criteria

All tasks T1-T6 marked complete with tests passing.
