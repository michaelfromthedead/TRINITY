# PHASE 4 TODO: Integration Verification

## Objective

Verify cross-module integration between character physics, cloth simulation, and collision detection systems.

---

## Task 1: Implement PhysicsWorld

**File**: `engine/simulation/physics_world.py` (new)

**Description**: Concrete implementation of `PhysicsWorldInterface` connecting broadphase and narrowphase.

**Methods Required**:
- [ ] `__init__(broadphase: Broadphase)`
- [ ] `add_body(body)` - register collision body
- [ ] `remove_body(body)` - deregister collision body
- [ ] `sweep(shape, start, end)` - swept collision query
- [ ] `overlap(shape, position)` - overlap test
- [ ] `raycast(origin, direction, max_distance)` - ray query
- [ ] `step(dt)` - update all bodies

**Acceptance Criteria**:
- [ ] Implements `PhysicsWorldInterface` from `character_controller.py`
- [ ] Uses broadphase for candidate filtering
- [ ] Uses narrowphase for precise collision
- [ ] Returns ContactManifold with contact points

---

## Task 2: Character-Collision Integration Test

**File**: `tests/integration/simulation/test_character_collision.py`

**Description**: Verify character controller integrates with collision system.

**Test Scenarios**:
- [ ] Character walks on flat ground
- [ ] Character stops at wall
- [ ] Character climbs step
- [ ] Character slides down steep slope
- [ ] Character hits ceiling
- [ ] Character collides with moving platform

**Acceptance Criteria**:
- [ ] No penetration through geometry
- [ ] Ground detection works correctly
- [ ] Slope limiting enforced
- [ ] Step climbing works up to configured height

---

## Task 3: Cloth-World Collision Test

**File**: `tests/integration/simulation/test_cloth_world.py`

**Description**: Verify cloth particles collide with static world geometry.

**Test Scenarios**:
- [ ] Cloth drops onto flat plane
- [ ] Cloth drapes over sphere
- [ ] Cloth slides off tilted surface
- [ ] Cloth catches on box corner
- [ ] Cloth wraps around capsule

**Acceptance Criteria**:
- [ ] No penetration through geometry
- [ ] Cloth rests naturally on surfaces
- [ ] Collision response smooth (no jitter)
- [ ] Self-collision prevents fold-through

---

## Task 4: Cloth-Character Collision Test

**File**: `tests/integration/simulation/test_cloth_character.py`

**Description**: Verify cloth collides with character ragdoll bodies.

**Test Scenarios**:
- [ ] Cape attached to character shoulders
- [ ] Cape collides with character back
- [ ] Skirt attached to character waist
- [ ] Skirt collides with character legs
- [ ] Cloth collision during ragdoll fall

**Acceptance Criteria**:
- [ ] Attachment points follow bone transforms
- [ ] Cloth does not penetrate character body
- [ ] Cloth responds to ragdoll motion
- [ ] Performance acceptable with full cloth + ragdoll

---

## Task 5: Define Standard Collision Layers

**File**: `engine/simulation/collision/collision_filter.py` (modify)

**Description**: Add standard layer definitions for integrated physics.

**Layers to Define**:
- [ ] CLOTH (new custom layer)
- [ ] RAGDOLL (new custom layer)
- [ ] Update PLAYER layer documentation
- [ ] Update NPC layer documentation

**Layer Matrix**:
- [ ] CLOTH vs TERRAIN: collide
- [ ] CLOTH vs RAGDOLL (same character): collide
- [ ] CLOTH vs PLAYER (other characters): collide
- [ ] CLOTH vs CLOTH: no collide (use self-collision)
- [ ] RAGDOLL vs TERRAIN: collide
- [ ] RAGDOLL vs PLAYER: collide

**Acceptance Criteria**:
- [ ] Layers documented with intended usage
- [ ] Matrix matches integration requirements
- [ ] FilterPresets updated for simulation

---

## Task 6: Multi-Character Interaction Test

**File**: `tests/integration/simulation/test_multi_character.py`

**Description**: Verify multiple characters interact correctly.

**Test Scenarios**:
- [ ] Two characters walk toward each other
- [ ] Characters collide and stop
- [ ] Character A's cloth does not collide with Character A
- [ ] Character A's cloth collides with Character B
- [ ] Ragdoll collision between two characters

**Acceptance Criteria**:
- [ ] Layer filtering prevents self-collision bugs
- [ ] Cross-character collision works
- [ ] No performance degradation with multiple characters
- [ ] Deterministic results

---

## Task 7: ClothCharacterAttachment System

**File**: `engine/simulation/cloth/cloth_attachment.py` (new)

**Description**: System for attaching cloth particles to character bones.

**Classes**:
- [ ] `AttachmentPoint(bone_name, local_offset, particle_index)`
- [ ] `ClothAttachment(cloth_mesh, attachment_points)`
- [ ] `update_from_skeleton(skeleton_pose)` method

**Acceptance Criteria**:
- [ ] Pinned particles follow bone transforms
- [ ] Local offset applied in bone space
- [ ] Works with both animation and ragdoll poses
- [ ] Multiple attachment points per cloth

---

## Task 8: Full System Integration Test

**File**: `tests/integration/simulation/test_full_system.py`

**Description**: End-to-end test of all simulation systems together.

**Test Scenario**:
1. [ ] Spawn character with attached cloth
2. [ ] Character walks across terrain
3. [ ] Character collides with obstacle
4. [ ] Character activates ragdoll (falls)
5. [ ] Cloth responds to ragdoll motion
6. [ ] Character recovers (active ragdoll balance)
7. [ ] Verify no penetrations throughout

**Acceptance Criteria**:
- [ ] No exceptions during simulation
- [ ] No visual artifacts (penetration, explosion)
- [ ] Performance within 60 FPS target
- [ ] Memory stable (no leaks over 1000 frames)

---

## Task 9: Performance Profiling

**File**: `tests/integration/simulation/test_performance.py`

**Description**: Profile integrated simulation performance.

**Metrics**:
- [ ] Frame time with 1 character + cloth
- [ ] Frame time with 10 characters + cloth
- [ ] Broadphase time percentage
- [ ] Narrowphase time percentage
- [ ] Cloth solve time percentage
- [ ] Memory usage

**Acceptance Criteria**:
- [ ] 60 FPS with 1 character (16ms budget)
- [ ] 30 FPS with 10 characters (33ms budget)
- [ ] Identify bottlenecks for optimization
- [ ] Results logged to benchmark file

---

## Task 10: Determinism Verification

**File**: `tests/integration/simulation/test_determinism.py`

**Description**: Verify simulation produces identical results with same input.

**Test Method**:
1. [ ] Run simulation with fixed seed
2. [ ] Record all particle/body positions at frame 100
3. [ ] Re-run simulation with same seed
4. [ ] Compare positions (must be exactly equal)

**Acceptance Criteria**:
- [ ] Bitwise-identical results across runs
- [ ] No floating-point non-determinism
- [ ] Works across different machines (if no platform-specific math)

---

## Dependencies

- Phase 1: GPU Cloth (for GPU cloth integration testing)
- Phase 2: Math Unification (clean shared types)
- Phase 3: Test Coverage (edge case handling verified)

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: PhysicsWorld | High | 6 hours |
| Task 2: Character-Collision | Medium | 3 hours |
| Task 3: Cloth-World | Medium | 3 hours |
| Task 4: Cloth-Character | High | 4 hours |
| Task 5: Collision Layers | Low | 2 hours |
| Task 6: Multi-Character | Medium | 3 hours |
| Task 7: Attachment System | Medium | 4 hours |
| Task 8: Full System Test | High | 4 hours |
| Task 9: Performance | Medium | 3 hours |
| Task 10: Determinism | Low | 2 hours |
| **Total** | | **34 hours** |
