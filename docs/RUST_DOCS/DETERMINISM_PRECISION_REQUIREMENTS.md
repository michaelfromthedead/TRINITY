# Determinism Precision Requirements by Subsystem

**Task:** T-CC-0.21
**Date:** 2026-05-26
**Status:** GREEN_LIGHT

This document specifies which subsystems require deterministic fixed-point math and at what precision level. Fixed-point math is required for any system that must produce bit-identical results across different platforms, compilers, or floating-point modes.

---

## Fixed-Point Type Summary

| Type | Format | Range | Precision | Use Case |
|------|--------|-------|-----------|----------|
| `Fixed16` | Q8.8 | -128 to ~128 | 1/256 (~0.004) | Fast, small values, angles, normalized |
| `Fixed32` | Q16.16 | -32768 to ~32768 | 1/65536 (~0.00002) | World positions, physics, timestamps |

---

## Subsystem Requirements

### Tier 1: MUST USE Fixed-Point (Lockstep Critical)

These subsystems directly affect game state that must be identical across all clients in a lockstep simulation.

| Subsystem | Precision | Reason |
|-----------|-----------|--------|
| **Physics Simulation** | Fixed32 | Forces, velocities, positions must match exactly |
| **Collision Detection** | Fixed32 | Intersection tests determine gameplay outcomes |
| **Gameplay Logic** | Fixed32 | Health, damage, timers, cooldowns |
| **RNG (PCG64)** | Integer | Already implemented with integer-only operations |
| **Tick Scheduler** | Fixed32 | Frame timing, delta time accumulator |
| **Entity State** | Fixed32 | Position, rotation, scale of deterministic entities |
| **Command Buffer** | N/A | Checksums use integer hashing |

### Tier 2: SHOULD USE Fixed-Point (Desync Risk)

These subsystems can cause subtle desyncs if using floating-point, but desyncs may not be immediately visible.

| Subsystem | Precision | Reason |
|-----------|-----------|--------|
| **AI Navigation** | Fixed32 | Pathfinding decisions must match |
| **Steering Behaviors** | Fixed32 | Velocity calculations affect movement |
| **Animation Blend** | Fixed16 | Blend weights determine pose |
| **Particle Physics** | Fixed16 | Particle positions (if gameplay-relevant) |
| **Procedural Generation** | Fixed32 | World generation must match |
| **Audio Timing** | Fixed32 | Music sync for rhythm games |

### Tier 3: MAY USE Float (Visual Only)

These subsystems produce visual output only and do not affect game state. Floating-point is acceptable.

| Subsystem | Notes |
|-----------|-------|
| **Rendering** | GPU uses float; visual-only |
| **Post-Processing** | Bloom, DOF, color grading |
| **UI Layout** | Screen-space positioning |
| **Camera Smoothing** | Visual interpolation |
| **Particle Visuals** | Non-gameplay particle effects |
| **Audio Mixing** | Volume/panning are visual feedback |
| **Skeletal Animation Display** | Visual bone transforms (gameplay uses Fixed) |

---

## Precision Selection Guide

### Use Fixed16 (Q8.8) When:
- Values are normalized (0.0 to 1.0)
- Values represent percentages or blend weights
- Values are angles in turns (0.0 to 1.0 = 360°)
- Memory is constrained (e.g., large arrays)
- Values stay within -128 to +128 range

**Examples:**
- Animation blend weights
- Health percentage
- Opacity/alpha
- Audio volume (0.0 to 1.0)
- Normalized direction vectors

### Use Fixed32 (Q16.16) When:
- Values represent world-space positions
- Values can exceed ±128
- High precision is needed
- Time values (seconds, milliseconds)
- Physics forces and velocities

**Examples:**
- Entity world position
- Physics velocity vectors
- Accumulated time
- Damage values
- Currency/score

---

## Implementation Notes

### Conversion Guidelines

```python
# Float to Fixed (at system boundary)
position = Fixed32(entity.x)  # Input from editor

# Fixed arithmetic (in simulation)
new_pos = position + velocity * delta_time  # All Fixed32

# Fixed to Float (for rendering)
render_x = position.as_float  # Output to GPU
```

### Checksum Integration

All deterministic state changes should go through `DeterministicCommandBuffer` which automatically checksums operations:

```python
# Correct: Checksum includes the value
buf.set_value(entity, Position, "x", Fixed32(10.5))

# Incorrect: Direct modification bypasses checksum
entity.x = 10.5  # NOT checksummed!
```

### Phase Execution

Systems that modify deterministic state must run in the correct scheduler phase:

| Phase | Determinism Level |
|-------|-------------------|
| PRE_INPUT → POST_INPUT | Deterministic (input buffering) |
| PRE_PHYSICS → POST_PHYSICS | **CRITICAL** (physics sim) |
| PRE_UPDATE → POST_UPDATE | **CRITICAL** (gameplay logic) |
| PRE_RENDER → LATE | Visual only (float OK) |

---

## Subsystem-Specific Details

### S19: Physics

**Required Precision:** Fixed32 for all simulation values

| Component | Type | Notes |
|-----------|------|-------|
| RigidBody.position | Vec3<Fixed32> | World position |
| RigidBody.velocity | Vec3<Fixed32> | Linear velocity |
| RigidBody.angular_velocity | Vec3<Fixed32> | Angular velocity |
| RigidBody.mass | Fixed32 | Mass in kg |
| Force.direction | Vec3<Fixed16> | Normalized |
| Force.magnitude | Fixed32 | Newton force |
| Collision.normal | Vec3<Fixed16> | Normalized |
| Collision.depth | Fixed32 | Penetration depth |

### S17: Gameplay

**Required Precision:** Fixed32 for most, Fixed16 for percentages

| Component | Type | Notes |
|-----------|------|-------|
| Health.current | Fixed32 | Can be fractional |
| Health.max | Fixed32 | Maximum health |
| Damage.amount | Fixed32 | Damage value |
| Timer.remaining | Fixed32 | Seconds |
| Cooldown.progress | Fixed16 | 0.0 to 1.0 |
| StatusEffect.strength | Fixed16 | Percentage |

### S14: Animation

**Required Precision:** Fixed16 for blend, Fixed32 for time

| Component | Type | Notes |
|-----------|------|-------|
| AnimState.time | Fixed32 | Current playback time |
| AnimState.speed | Fixed32 | Playback rate |
| BlendWeight | Fixed16 | Layer blend (0-1) |
| MaskWeight | Fixed16 | Per-bone mask (0-1) |

### S16: Networking

**Required Precision:** Fixed32 for state, Integer for sequence

| Component | Type | Notes |
|-----------|------|-------|
| NetworkTick | u64 | Tick number (integer) |
| Timestamp | Fixed32 | Relative time |
| InputCommand | Bitfield | Integer flags |
| StateSnapshot | Fixed32[] | Serialized state |

---

## Testing Requirements

Each deterministic subsystem must have tests that verify:

1. **Same-seed determinism**: Running twice with same seed produces identical checksums
2. **Cross-platform consistency**: Checksums match on different architectures (via CI)
3. **Replay fidelity**: Recorded input produces identical state on replay
4. **Desync detection**: Intentional corruption is detected within 1 tick

Tests implemented:
- `tests/trinity/test_fixed_point.py` (36 tests)
- `tests/trinity/test_pcg64.py` (45 tests)
- `tests/engine/core/ecs/test_deterministic_buffer.py` (38 tests)
- `tests/engine/core/test_tick_scheduler.py` (37 tests)
- `tests/engine/core/test_replay_verification.py` (12 tests)

**Total:** 168 determinism tests

---

## Migration Path

For existing float-based systems:

1. **Identify** all state that affects gameplay outcomes
2. **Convert** those fields to Fixed16 or Fixed32
3. **Route** all modifications through DeterministicCommandBuffer
4. **Test** with replay verification to confirm determinism
5. **Document** any remaining float fields and justify why they're safe

---

## References

- `trinity/types.py`: Fixed16, Fixed32, PCG64, SystemPhase
- `engine/core/ecs/deterministic_buffer.py`: DeterministicCommandBuffer
- `engine/core/tick_scheduler.py`: TickScheduler, PhaseContext
- GAPSET_20 Phase 0: T-CC-0.14 through T-CC-0.18
