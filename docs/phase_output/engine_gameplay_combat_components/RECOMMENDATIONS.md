# RECOMMENDATIONS: engine/gameplay/combat + engine/gameplay/components

---

## Rust Bridge Requirements

### High Priority

| Component | Reason | Complexity | Performance Impact |
|-----------|--------|------------|-------------------|
| **Hitbox Collision** | O(n*m) per frame, called every update | Medium | HIGH |
| **Transform Matrices** | Matrix multiplication in hierarchy walks | Low | HIGH |
| **Stats Computation** | Called frequently during gameplay | Low | MEDIUM |

**Hitbox Collision Details**:
- Current: Python nested loop over active hitboxes/hurtboxes
- Target: Rust BVH or spatial hash with SIMD AABB tests
- Expected Improvement: 10-50x for scenes with 100+ collision volumes

**Transform Matrix Details**:
- Current: Python Mat4 multiplication per hierarchy level
- Target: Rust SIMD-accelerated matrix math
- Expected Improvement: 5-10x for deep hierarchies

### Medium Priority

| Component | Reason | Complexity | Performance Impact |
|-----------|--------|------------|-------------------|
| **Modifier Application** | Hot path during combat | Low | MEDIUM |
| **Spawn Selection** | Distance calculations, sorting | Low | LOW |
| **Leaderboard Sorting** | Frequent during multiplayer | Low | LOW |

### Low Priority

| Component | Reason | Complexity | Performance Impact |
|-----------|--------|------------|-------------------|
| **Team Relationship** | Infrequent lookups | Low | LOW |
| **Death State Machine** | Simple state transitions | Low | LOW |
| **Event Emission** | Python callback pattern sufficient | N/A | N/A |

---

## Integration Strategy

### Phase 1: Type Registry Integration (Week 1-2)

1. Register component data classes with existing type_registry.rs
2. Define Rust equivalents for:
   - `Vec3`, `Quat`, `Mat4` (already exists in omega math)
   - `StatModifier`, `Stat`
   - `Hitbox`, `Hurtbox`, `BoundingBox`
   - `TransformComponent` hierarchy

### Phase 2: Read-Path Optimization (Week 3-4)

1. Implement Rust-backed storage for:
   - Transform world matrices (read frequently)
   - Computed stat values (read frequently)
   - Hitbox/hurtbox positions (read every frame)

2. Keep Python facade for API compatibility:
   ```python
   class TransformComponent:
       @property
       def world_matrix(self) -> Mat4:
           return self._rust_storage.get_world_matrix(self._entity_id)
   ```

### Phase 3: Write-Path Migration (Week 5-6)

1. Migrate dirty tracking to Rust bitfields
2. Implement Rust-side change detection
3. Generate Python events from Rust state changes

### Phase 4: Hot-Path Acceleration (Week 7-8)

1. Implement Rust hitbox collision system
2. Implement Rust modifier computation
3. Benchmark and tune

---

## Testing Strategy

### Unit Tests (Existing Python Tests)

1. Verify all algorithms with known inputs/outputs:
   - AABB intersection edge cases (touching, overlapping, separate)
   - Modifier stacking order (verify OVERRIDE > FLAT > PERCENT_BASE > MULTIPLY > PERCENT_TOTAL)
   - Multi-kill window (boundary conditions at exactly window time)
   - Coyote time (jump at exact threshold)

### Integration Tests

1. **Combat Flow Test**: Player joins -> takes damage -> dies -> respawns -> kills enemy -> wins
2. **Component Lifecycle Test**: Entity creation -> component attach -> update -> dirty tracking -> serialize -> deserialize
3. **Team Test**: IFF checks across team changes, relationship modifications

### Blackbox Tests

| Test | Input | Expected Output |
|------|-------|-----------------|
| Kill attribution | killer_id, victim_id, assists | Score updates, killstreak, multi-kill detection |
| Spawn selection | player position, enemy positions | Spawn point meeting distance requirements |
| Damage calculation | raw damage, resistances, armor, shield | Final damage after all modifiers |
| Modifier expiration | timed modifier, elapsed time | Modifier removed when duration exceeded |

### Whitebox Tests

1. Verify cache invalidation triggers on modifier add/remove
2. Verify dirty flags set on velocity/position changes
3. Verify respawn queue ordering by time
4. Verify priority resolution selects highest priority hitbox

### Performance Tests

1. **Collision Stress Test**: 1000 hitboxes vs 1000 hurtboxes
2. **Modifier Stress Test**: 100 modifiers per stat, 100 stats
3. **Hierarchy Stress Test**: 10-level deep transform hierarchy with 100 children per level

---

## Risk Assessment

### Low Risk

| Area | Risk | Mitigation |
|------|------|------------|
| Python API Compatibility | Breaking changes | Keep Python facade, deprecate gradually |
| Serialization Format | Schema drift | Use type_registry versioning |
| Event Ordering | Race conditions | Document callback ordering guarantees |

### Medium Risk

| Area | Risk | Mitigation |
|------|------|------------|
| Performance Regression | Rust FFI overhead | Batch operations, avoid per-call overhead |
| Memory Ownership | Lifetime issues | Clear ownership rules, prefer copy over borrow |
| Error Handling | Panic vs Exception | Map Rust panics to Python exceptions |

### High Risk

| Area | Risk | Mitigation |
|------|------|------------|
| Concurrent Access | Data races | Single-threaded access from Python, or explicit locking |
| State Synchronization | Python/Rust divergence | Single source of truth (Rust), Python reads only |

---

## Implementation Checklist

- [ ] Type registry entries for component types
- [ ] Rust BoundingBox with SIMD AABB test
- [ ] Rust TransformComponent with Mat4 caching
- [ ] Rust StatsComponent with modifier computation
- [ ] Python bindings via PyO3
- [ ] Blackbox test suite
- [ ] Whitebox test suite
- [ ] Performance benchmark suite
- [ ] Migration documentation
- [ ] API compatibility verification
