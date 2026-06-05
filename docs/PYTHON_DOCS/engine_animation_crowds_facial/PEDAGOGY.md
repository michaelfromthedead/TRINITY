# PEDAGOGY: engine_animation_crowds_facial

**Concept Evolution Log (Append-Only)**
**Generated:** 2026-05-23

---

## Evolution Record

This document tracks concept evolution across source documents in temporal order.

---

### Pass 1: engine_animation_crowds.md

**New Concepts Introduced:**

| Concept | Initial Value | Source Line | Notes |
|---------|--------------|-------------|-------|
| AnimationTexture | GPU texture baking for skeletal animation | L22-24 | First definition |
| CrowdAgent | Full agent model with position, velocity, state | L30 | First definition |
| LODLevel | Per-level LOD configuration | L35-37 | First definition |
| InstanceBuffer | GPU-ready packed buffer | L33 | First definition |
| Steering Avoidance | RVO-style with priority weighting | L55-98 | With code evidence |
| Skeleton Reduction | Importance-based bone culling | L122-147 | With algorithm |
| BehaviorTypes | 5 behaviors (Idle, Walking, Waiting, Fleeing, Formation) | L45-51 | Complete enumeration |

**No Prior Values** - First document processed.

---

### Pass 2: engine_animation_facial.md

**New Concepts Introduced:**

| Concept | Initial Value | Source Line | Notes |
|---------|--------------|-------------|-------|
| BlendShape | Sparse vertex delta representation | L43-52 | With numpy arrays |
| ActionUnit | FACS AU enumeration | L55-62 | 21 AUs with muscle annotations |
| LipSyncController | Phoneme-to-viseme pipeline | L20-21 | First definition |
| EyeController | Vergence, saccades, blinking | L24 | First definition |
| Coarticulation | Anticipation + carryover blending | L66-70 | Algorithm reference |
| Vergence | Eye convergence calculation | L74-80 | With geometric formula |
| FaceRig | Priority-based layer blending | L85-92 | With blend logic |
| ARKit52 | 52 standard blend shapes | L94-101 | Industry compatibility |

**No Overwrites** - Distinct subsystem from Pass 1.

---

### Pass 3: engine_animation_crowds_facial.md

**Concept Updates (Upsert-Overwrite):**

| Concept | Prior Value | New Value | Reason |
|---------|-------------|-----------|--------|
| Crowds Line Count | ~2,241 | 2,237 | Synthesis document uses exact count |
| Facial Line Count | Not specified | 5,233 | Added explicit total |
| AnimationTexture | GPU texture baking | Expanded: 2 pixels per bone encoding | More detail from combined analysis |
| Avoidance Algorithm | RVO-style | Added: Lines 304-345 reference, MIN_DISTANCE_EPSILON | Line-specific evidence |
| FACS | 21 AUs | Added: 18 AU subset detail (upper face, mouth, eye) | Refinement |
| Ekman Expressions | Not enumerated | 8 named: NEUTRAL through CONTEMPT | Complete list |
| CONTEMPT | Not detailed | Asymmetric - unilateral lip corner | Anatomical accuracy note |
| Cross-Module Dependencies | Not present | Full dependency graph | New synthesis |

**New Concepts from Synthesis:**

| Concept | Value | Source | Notes |
|---------|-------|--------|-------|
| Configuration Centralization | All modules use config.py | L223-229 | Architectural pattern |
| Dependency Graph | face_rig -> blend_shapes/eye/facs/lip_sync | L208-220 | Integration view |
| Production Readiness | Both subsystems production-ready | L237-239 | Status assessment |
| Suggested Enhancements | Spatial partitioning, neural lip sync | L241-252 | Gap analysis |
| Testing Priorities | Avoidance edge cases, coarticulation, vergence | L254-257 | QA focus |

---

## Evolution Summary

| Pass | Document | New Concepts | Updated Concepts | Total |
|------|----------|--------------|------------------|-------|
| 1 | engine_animation_crowds.md | 7 | 0 | 7 |
| 2 | engine_animation_facial.md | 8 | 0 | 15 |
| 3 | engine_animation_crowds_facial.md | 5 | 8 | 28 |

**Final Concept Count: 28 distinct concepts**

---

## Cross-References

- No COURT sessions required (no contradictions found)
- All documents consistent in classification (REAL IMPLEMENTATION)
- Line counts refined in synthesis pass but not contradictory
