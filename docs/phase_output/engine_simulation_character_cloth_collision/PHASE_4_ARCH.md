# PHASE 4 ARCHITECTURE: Integration Verification

## Objective

Verify cross-module integration between character physics, cloth simulation, and collision detection systems.

## Integration Points Identified

The investigation identified several integration boundaries that require verification:

### 1. Character Controller + Collision System

The `CharacterController` in `character_controller.py` uses:
- `PhysicsWorldInterface.sweep()` for movement sweeps
- `PhysicsWorldInterface.overlap()` for ground detection
- Broadphase for initial candidate filtering
- Narrowphase for precise contact generation

### 2. Cloth + Collision Shapes

The `ClothCollisionHandler` in `cloth_collision.py` collides cloth particles against:
- `SphereCollider` (character capsule approximation)
- `CapsuleCollider` (limbs)
- `BoxCollider` (props)
- `MeshCollider` (static geometry)
- `SDFCollider` (smooth surfaces)

### 3. Cloth + Ragdoll Bodies

Cloth attached to character (clothing, capes) must:
- Follow ragdoll body transforms
- Collide with ragdoll body parts
- Not penetrate through body on impact

### 4. Collision Layers + All Systems

The `CollisionFilterManager` governs all collision interactions:
- Character vs terrain
- Character vs NPC/enemy
- Cloth vs character
- Cloth vs world geometry

## Architecture Decisions

### ADR-INT-001: PhysicsWorldInterface Implementation

**Decision**: Create concrete `PhysicsWorld` implementing `PhysicsWorldInterface` with actual broadphase/narrowphase.

**Rationale**:
- Character controller requires working collision queries
- Abstract interface exists but no concrete binding observed
- Integration tests need functional world

**Implementation**:
```python
class PhysicsWorld(PhysicsWorldInterface):
    def __init__(self, broadphase: Broadphase):
        self.broadphase = broadphase
        
    def sweep(self, shape, start, end):
        candidates = self.broadphase.query_aabb(swept_aabb)
        return narrowphase_sweep(shape, candidates, start, end)
```

### ADR-INT-002: ClothCharacterAttachment

**Decision**: Define attachment protocol for cloth anchored to character bones.

**Rationale**:
- Clothing simulation requires bone transforms
- Attachment points must update with ragdoll
- Need consistent interface for various cloth types

**Protocol**:
```python
class AttachmentPoint:
    bone_name: str
    local_offset: Vec3
    particle_index: int
```

### ADR-INT-003: Unified Collision Layer Configuration

**Decision**: Define standard collision layers for character+cloth+world interactions.

**Layer Assignments**:
| Layer | Bit | Description |
|-------|-----|-------------|
| PLAYER | 8 | Player character |
| NPC | 9 | Non-player characters |
| ENEMY | 10 | Hostile entities |
| CLOTH | Custom | Character clothing |
| RAGDOLL | Custom | Ragdoll body parts |
| TERRAIN | 12 | World geometry |

**Matrix Configuration**:
- CLOTH collides with: TERRAIN, RAGDOLL (self), PLAYER (others)
- CLOTH does not collide with: Own PLAYER (to avoid self-intersection)
- RAGDOLL collides with: TERRAIN, PLAYER, NPC, ENEMY

### ADR-INT-004: Integration Test Scenarios

**Decision**: Define canonical test scenarios for cross-module verification.

**Scenarios**:
1. **Walking Character**: Character moves, collision resolves, pose updates
2. **Falling Character**: Gravity, ground contact, ragdoll activation
3. **Cloth Drape**: Cloth falls onto static geometry
4. **Cloth-Character Collision**: Cape collides with character body
5. **Multi-Character**: Two characters interact with collision filtering

## System Flow Diagrams

### Character Movement Flow

```
Input Velocity
      |
      v
CharacterController.move_and_slide()
      |
      v
PhysicsWorld.sweep()
      |
      +---> Broadphase.query_aabb()
      |           |
      |           v
      |     Candidate list
      |           |
      |           v
      +---> Narrowphase.collide_shapes()
                  |
                  v
            Contact manifold
                  |
                  v
       Resolve collision
                  |
                  v
       Updated position
```

### Cloth-Character Collision Flow

```
Ragdoll pose
      |
      v
Extract body colliders
      |
      v
ClothCollisionHandler.add_colliders()
      |
      v
ClothSimulation.step()
      |
      +---> PBD integration
      |           |
      |           v
      +---> Constraint projection
      |           |
      |           v
      +---> ClothCollisionHandler.resolve_collisions()
                  |
                  v
            Corrected positions
```

## Integration Test Infrastructure

```
tests/
  integration/
    simulation/
      test_character_collision.py     # Character + broadphase + narrowphase
      test_cloth_world.py             # Cloth + static geometry
      test_cloth_character.py         # Cloth + ragdoll bodies
      test_multi_character.py         # Multiple characters + layers
      test_full_system.py             # All systems together
```

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing PhysicsWorld impl | High | Create minimal implementation for tests |
| Layer configuration conflicts | Medium | Document standard layers, validate on startup |
| Performance at integration | Medium | Profile integrated scenarios, optimize hot paths |
| State sync issues | High | Strict update ordering, integration tests |

## Verification Criteria

### Functional Verification

- [ ] Character moves and stops at obstacles
- [ ] Character does not penetrate geometry
- [ ] Cloth drapes naturally on surfaces
- [ ] Cloth does not penetrate character body
- [ ] Collision layers correctly filter interactions

### Performance Verification

- [ ] 60 FPS with 1 character + cloth
- [ ] 30 FPS with 10 characters + cloth each
- [ ] Memory stable over 1000 frame simulation

### Stability Verification

- [ ] No position explosion over 10000 frames
- [ ] No penetration accumulation over time
- [ ] Deterministic results with fixed seed
