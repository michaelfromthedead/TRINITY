# RECOMMENDATIONS: engine/gameplay/{economy,entity,input}

---

## Rust Bridge Requirements

### High Priority

| Requirement | Module | Rationale |
|-------------|--------|-----------|
| Actor Transform FFI | entity | Actors need GPU-side transform for rendering; position/rotation must sync to Rust renderer every frame |
| Equipment Stats FFI | economy | Combat damage calculations in Rust need equipment modifiers (flat/percent/multiplier) |
| Entity ID Registry | entity | Rust needs to map Python entity IDs to GPU instance indices |

### Medium Priority

| Requirement | Module | Rationale |
|-------------|--------|-----------|
| Input Processing FFI | input | Frame-critical input (dead zones, response curves) may benefit from Rust latency reduction |
| Visibility Flags | entity | Actor visibility/cull state needs bridge for GPU culling |
| Animation State | entity | Character movement state (walking, running, jumping) drives animation in Rust |

### Low Priority

| Requirement | Module | Rationale |
|-------------|--------|-----------|
| Inventory Serialization | economy | Save/load can remain Python-only; no GPU involvement |
| Loot Table Caching | economy | Loot calculations are infrequent; Python suffices |
| Prefab Serialization | entity | Prefab loading is one-time; Python suffices |

---

## Integration Strategy

### Phase 1: Entity Transform Bridge (Week 1-2)

1. Define Rust struct matching Python Transform (position, rotation, scale)
2. Implement PyO3 bindings for bidirectional transform sync
3. Add dirty flag to Python Actor to minimize FFI calls
4. Register entity IDs with Rust TypeRegistry for GPU instance mapping

```rust
// Example Rust side
#[pyclass]
struct TransformBridge {
    position: [f32; 3],
    rotation: [f32; 4],  // Quaternion
    scale: [f32; 3],
}
```

### Phase 2: Equipment Stats Bridge (Week 3)

1. Define Rust struct for EquipmentStats aggregation
2. Implement batch update for all equipped items
3. Expose combined stats to Rust combat system

### Phase 3: Input Processing (Week 4, Optional)

1. Profile Python input latency
2. If > 1ms, port dead zone and response curve to Rust
3. Keep action mapping logic in Python (flexibility over latency)

---

## Testing Strategy

### Unit Tests Required

| Test | Module | Priority |
|------|--------|----------|
| Pity system edge cases | economy/loot | HIGH |
| Stack overflow during merge | economy/inventory | HIGH |
| Prefab inheritance depth limit | entity/prefab | MEDIUM |
| DoubleTap timing edge cases | input/action_mapper | MEDIUM |
| Lifecycle state transition validation | entity/lifecycle | MEDIUM |
| Cross dead zone near-zero values | input/processing | LOW |

### Integration Tests Required

| Test | Modules | Priority |
|------|---------|----------|
| Equipment equip -> stat update -> combat | economy, combat | HIGH |
| Actor spawn -> possession -> input | entity, input | HIGH |
| Loot drop -> inventory add -> weight check | economy | MEDIUM |
| Crafting queue -> progress -> completion | economy | MEDIUM |

### Performance Tests Required

| Test | Target | Threshold |
|------|--------|-----------|
| 1000 actors tick | entity | < 5ms |
| 100 items add to inventory | economy | < 1ms |
| Input processing per frame | input | < 0.5ms |
| Loot table roll (10 entries) | economy | < 0.1ms |

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Transform sync latency | Frame drops, visual stutter | Batch updates, dirty flags, async queue |
| Entity ID collision | Rendering corruption | Use Python UUID -> Rust u64 mapping with collision check |
| Equipment stat desync | Incorrect combat damage | Single source of truth (Python), read-only Rust cache |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Inventory weight overflow | Negative weight display | Add explicit weight cap validation |
| Pity counter overflow | Guaranteed drops stop working | Use u32 counter, document max threshold |
| Input polling race | Missed inputs | Use input buffer with frame timestamp |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prefab inheritance cycles | Stack overflow | Already have depth limit (10) |
| Crafting queue memory | Memory growth | Queue size limit, eviction policy |
| Device hot-plug spam | Event flood | Debounce device events |

---

## Recommended Next Steps

1. **Immediate**: Add entity transform FFI to GAPSET_3_BRIDGE scope
2. **Short-term**: Complete serialization (from_dict) for save/load
3. **Medium-term**: Add unit tests for pity system, stack overflow
4. **Long-term**: Profile input latency, decide on Rust port
5. **Documentation**: Add API docs for external consumers
