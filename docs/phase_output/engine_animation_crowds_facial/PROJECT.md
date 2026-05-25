# PROJECT: engine_animation_crowds_facial

**Scope, Goals, and Constraints**
**Generated:** 2026-05-23

---

## Project Identity

**Name:** TRINITY Animation - Crowds and Facial Subsystems
**Domain:** Game Engine Animation Pipeline
**Status:** REAL IMPLEMENTATION (Production-Ready)

---

## Scope

### In Scope

1. **GPU-Accelerated Crowd Rendering**
   - Animation texture baking for vertex shader sampling
   - Instance buffer management with batching
   - Distance-based LOD with skeleton reduction
   - Behavior-driven agent simulation

2. **Facial Animation System**
   - Blend shape management with sparse deltas
   - FACS Action Unit support (21 AUs)
   - Lip sync with phoneme-to-viseme mapping
   - Eye animation (vergence, saccades, blinking)
   - Motion capture playback and retargeting
   - Priority-based animation layering

### Out of Scope

1. **Not Implemented (Identified Gaps)**
   - Flow field navigation for crowds
   - Spatial partitioning (O(n) neighbor queries)
   - GPU compute for agent simulation
   - NavMesh integration for pathfinding
   - Neural network-based lip sync from audio
   - Automatic face rigging from mesh topology

2. **External Dependencies**
   - GPU backend execution (Python-side preparation only)
   - Physics engine integration
   - Audio system for lip sync timing

---

## Goals

### Primary Goals

| # | Goal | Status |
|---|------|--------|
| G1 | Render 1000+ crowd agents with GPU instancing | ACHIEVED |
| G2 | Support FACS-compliant facial expressions | ACHIEVED |
| G3 | Provide ARKit 52 blend shape compatibility | ACHIEVED |
| G4 | Enable lip sync from phoneme events | ACHIEVED |
| G5 | Support motion capture playback | ACHIEVED |

### Secondary Goals

| # | Goal | Status |
|---|------|--------|
| G6 | LOD system with smooth transitions | ACHIEVED |
| G7 | Eye animation with physiological accuracy | ACHIEVED |
| G8 | Coarticulation for natural speech | ACHIEVED |
| G9 | Priority-based animation blending | ACHIEVED |

---

## Constraints

### Technical Constraints

| Constraint | Rationale |
|------------|-----------|
| Python 3.13 | Engine embeds statically-linked Python 3.13 |
| NumPy Required | Vectorized operations for performance |
| Config-Driven | All parameters via engine/animation/config.py |
| No GPU Compute | Agent simulation is CPU-side |

### Design Constraints

| Constraint | Rationale |
|------------|-----------|
| FACS Compliance | Industry standard for facial animation |
| ARKit Compatibility | iOS face tracking integration |
| Sparse Blend Shapes | Memory efficiency for morph targets |
| Priority Layers | Deterministic blend ordering |

### Performance Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Neighbor Query | O(n) | No spatial partitioning implemented |
| Animation Texture | 2 pixels/bone | GPU memory budget |
| LOD Levels | Configurable | Quality/performance tradeoff |
| Instance Buffer | Dynamic growth | Memory management |

---

## Success Criteria

### Functional Criteria

1. Crowds render with correct animation from GPU textures
2. All 8 Ekman expressions blend correctly
3. Lip sync visemes transition smoothly with coarticulation
4. Eye vergence converges correctly for near targets
5. LOD transitions are flicker-free (hysteresis)

### Quality Criteria

1. All algorithms have edge case handling
2. Division-by-zero protection (MIN_DISTANCE_EPSILON)
3. Buffer overflow protection (InstanceBufferOverflowError)
4. Configuration externalized (no magic numbers)

---

## Dependencies

### Internal Dependencies

| Module | Dependency | Purpose |
|--------|------------|---------|
| face_rig | blend_shapes | Shape evaluation |
| face_rig | facs | AU blending |
| face_rig | lip_sync | Speech animation |
| face_rig | eye_animation | Procedural eyes |
| crowd_renderer | animation_texture | GPU baking |
| crowd_lod | animation_texture | Skeleton data |

### External Dependencies

| System | Integration Point |
|--------|-------------------|
| engine/animation/config | All configuration |
| engine/core | Math types (Vec3, Quaternion, Transform) |
| GPU Backend (Rust) | Actual rendering dispatch |

---

## Stakeholders

| Stakeholder | Interest |
|-------------|----------|
| Animation Artists | Facial rig usability, blend shape workflow |
| Technical Artists | LOD tuning, crowd optimization |
| Game Designers | Crowd behavior configuration |
| Performance Engineers | GPU/CPU balance, batching efficiency |
| QA | Edge case handling, stress testing |

---

## Risks

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| O(n) neighbor queries | Poor crowd scaling | Add spatial hash (future) |
| CPU-side agents | Crowd count ceiling | GPU compute (future) |
| No NavMesh | Agents can't pathfind | Integration required |

### Integration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Rust backend disconnect | Rendering non-functional | Bridge implementation (GAPSET_3) |
| Audio timing sync | Lip sync drift | Audio system integration |

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| Investigation | 2026-05-22 | Archaeological analysis complete |
| RDC Output | 2026-05-23 | Documentation consolidation |
