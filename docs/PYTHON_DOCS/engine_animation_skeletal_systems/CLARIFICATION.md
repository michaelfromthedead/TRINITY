# CLARIFICATION: Animation Skeletal Systems

**Generated:** 2026-05-23
**Subsystem:** engine/animation/skeletal + engine/animation/systems

---

## Philosophical Framing

This document provides the conceptual and pedagogical framing for understanding the Animation Skeletal Systems subsystem within the TRINITY game engine architecture.

---

## 1. Why Skeletal Animation?

### The Problem

Game characters need to move believably. Raw mesh vertex manipulation is impractical for complex characters with hundreds of thousands of vertices. The industry solution is **skeletal animation**: define a simplified bone hierarchy, animate the bones, and mathematically derive vertex positions from bone transforms.

### The TRINITY Approach

TRINITY implements a **layered animation architecture** where:

1. **Data Layer** (`skeletal/`): Pure data structures for skeletons, poses, clips
2. **Logic Layer** (`systems/`): ECS systems that process animation state per-frame
3. **Integration Layer**: Connection to rendering and gameplay

This separation follows the engine's broader **Data-Oriented Design** philosophy where data transformations are explicit and cache-friendly.

---

## 2. Core Abstractions

### 2.1 Skeleton vs. Pose

| Concept | Definition | Mutability |
|---------|------------|------------|
| **Skeleton** | The bone hierarchy structure - which bones exist and how they connect | Immutable after creation |
| **Pose** | The current transforms of all bones in a skeleton | Changes every frame |

A Skeleton is a **type** (shared across all instances of a character). A Pose is an **instance** (unique to each animated entity).

### 2.2 Local vs. Model Space

| Space | Definition | Use Case |
|-------|------------|----------|
| **Local Space** | Transform relative to parent bone | Animation authoring, blending |
| **Model Space** | Transform relative to skeleton root | Skinning, world placement |

Animation clips store **local space** transforms. Skinning requires **model space** matrices. The conversion is a key operation performed during pose evaluation.

### 2.3 Bind Pose vs. Current Pose

| Concept | Definition |
|---------|------------|
| **Bind Pose** | The pose in which the mesh was modeled |
| **Current Pose** | The animated pose for this frame |

Skinning matrices are computed as: `CurrentWorldMatrix * InverseBindMatrix`

This transforms vertices from bind pose space to current pose space.

---

## 3. Animation Data Flow

```
AnimationClip          [stored on disk]
     |
     v
ClipPlayer.sample()    [samples at time t]
     |
     v
Pose (local space)     [bone transforms]
     |
     v
LayeredBlender         [combine multiple poses]
     |
     v
Pose (blended)         [final local transforms]
     |
     v
local_to_model()       [hierarchy multiplication]
     |
     v
Pose (model space)     [world-relative transforms]
     |
     v
compute_skinning()     [multiply by inverse bind]
     |
     v
Mat4[] (GPU)           [upload to vertex shader]
```

This pipeline runs once per animated entity per frame.

---

## 4. Design Decisions

### 4.1 Dual Quaternion Skinning (DQS)

**Why both LBS and DQS?**

Linear Blend Skinning (LBS) is fast but causes **volume loss** at joints (the "candy wrapper" effect). Dual Quaternion Skinning preserves volume but is more expensive.

TRINITY provides both:
- **LBS** for background characters, LOD levels, crowds
- **DQS** for hero characters where quality matters

The `SkinningMethod` enum allows per-mesh selection.

### 4.2 ACL-Style Compression

**Why not just quantize everything to 16-bit?**

Uniform quantization wastes bits on tracks that don't change much while losing precision on tracks that do. ACL (Animation Compression Library) style compression:

- Analyzes each track's range and variance
- Allocates more bits to high-variance tracks
- Uses curve fitting to reduce keyframe count

TRINITY implements this via `compression.py` with configurable error thresholds.

### 4.3 Motion Matching vs. State Machines

**Why implement both?**

| System | Strengths | Weaknesses |
|--------|-----------|------------|
| **State Machines** | Predictable, designer-controlled | Combinatorial explosion, manual tuning |
| **Motion Matching** | Natural transitions, data-driven | Large database, less control |

TRINITY supports both:
- `animation_graph_system.py` for traditional state machines
- `motion_matching_system.py` for physics-driven characters

They can be combined: state machine for high-level states, motion matching within each state.

---

## 5. ECS Integration Philosophy

### Why Systems, Not Methods?

Traditional OOP would put animation logic in `Character.update()`. TRINITY uses ECS systems because:

1. **Cache Efficiency**: Process all IK components together, not scattered across objects
2. **Parallelization**: Independent systems can run on separate threads
3. **Composition**: Mix-and-match capabilities via component presence

### System Responsibilities

| System | Responsibility |
|--------|----------------|
| `skinning_system.py` | Convert poses to GPU matrices |
| `ik_system.py` | Adjust poses for procedural targets |
| `procedural_system.py` | Add secondary motion |
| `facial_system.py` | Facial expression and lip sync |
| `motion_matching_system.py` | Select next animation clip |
| `animation_graph_system.py` | Execute state machine logic |
| `crowd_system.py` | Batch animation for many entities |

---

## 6. Mathematical Foundations

### 6.1 Quaternions

All rotations in TRINITY use **unit quaternions** because:
- No gimbal lock
- Smooth interpolation (SLERP)
- Efficient composition
- Compact representation (4 floats vs 9 for matrix)

**Critical rule**: Quaternions must be normalized after interpolation.

### 6.2 Hermite Splines

Keyframe interpolation uses **Hermite splines** for cubic interpolation:
- Smooth curves through control points
- Tangent control for natural motion
- Local control (changing one keyframe doesn't affect distant curves)

The Hermite basis functions:
```
h00(t) = 2t^3 - 3t^2 + 1
h10(t) = t^3 - 2t^2 + t
h01(t) = -2t^3 + 3t^2
h11(t) = t^3 - t^2
```

### 6.3 Dual Quaternions

A dual quaternion is two quaternions: `dq = (r, d)` where:
- `r` is the rotation quaternion
- `d` encodes translation: `d = 0.5 * t * r` where `t` is a pure quaternion from translation

This representation enables volume-preserving skinning by avoiding the linear matrix interpolation that causes volume loss.

---

## 7. Common Pitfalls

### 7.1 Quaternion Hemisphere

When blending quaternions, `q` and `-q` represent the same rotation but SLERP will take the long path if mixing them. **Antipodality handling** (flipping the sign if dot product is negative) prevents this.

### 7.2 Root Motion Accumulation

Root motion must be **extracted** from the animation (so the clip stays in place) and **applied** to the entity's world transform. Failing to do this causes characters to slide or moonwalk.

### 7.3 Bone Order Dependencies

Skinning matrices must be computed in **hierarchy order** (parents before children) or world transforms will be wrong. TRINITY skeletons store bones in topological order.

---

## 8. Relationship to Engine Architecture

```
+------------------------------------------------------------------+
|                      GAMEPLAY SYSTEMS                             |
|            (Entity, AI, Input, Camera, Abilities)                 |
+------------------------------------------------------------------+
|     WORLD      |     SIMULATION      |      ANIMATION  <--- HERE  |
+------------------------------------------------------------------+
|                         RENDERING                                 |
|       (Frame Graph, GPU-Driven, Lighting, Post-Process)           |
+------------------------------------------------------------------+
```

Animation sits between:
- **Above**: Gameplay systems that request animations (AI, Input, Abilities)
- **Below**: Rendering that consumes skinning matrices

It is a **transformation layer** converting high-level animation commands into low-level GPU data.
