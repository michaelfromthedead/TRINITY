# PROJECT: Animation Skeletal Systems

**Generated:** 2026-05-23
**Subsystem:** engine/animation/skeletal + engine/animation/systems

---

## 1. Scope

The Animation Skeletal Systems subsystem provides the complete skeletal animation pipeline for the TRINITY game engine, encompassing:

### 1.1 In Scope

- **Bone Hierarchy Management**: Skeleton definition, parent-child relationships, bind poses
- **Pose Representation**: Local and model-space transforms, pose blending
- **Animation Clips**: Keyframe curves, playback control, events, root motion
- **Skinning**: Linear Blend Skinning (LBS) and Dual Quaternion Skinning (DQS)
- **Compression**: Quantization, curve fitting, ACL-style adaptive compression
- **Retargeting**: Animation transfer between different skeletons
- **Inverse Kinematics**: Two-Bone, FABRIK, CCD solvers
- **Procedural Animation**: Spring, LookAt, Sway, Breathing controllers
- **Facial Animation**: Emotion expressions, lip sync, eye tracking
- **Motion Matching**: Feature-based animation selection, KNN search
- **Animation Graphs**: State machines, transitions, parameter binding
- **ECS Skinning**: Entity-based skinning pipeline
- **Crowd Animation**: Agent synchronization, LOD, formations

### 1.2 Out of Scope

- GPU compute shaders for skinning (covered by rendering backend)
- Animation asset authoring tools (covered by tooling subsystem)
- Physics-based secondary animation (covered by simulation subsystem)
- Audio system integration (lip sync receives phoneme data but does not process audio directly)

---

## 2. Goals

### 2.1 Primary Goals

| Goal | Success Criteria |
|------|------------------|
| **Complete Animation Pipeline** | Full data flow from clips to poses to skinning matrices |
| **Multiple Skinning Methods** | Support both LBS (performance) and DQS (quality) |
| **Comprehensive IK** | Solve common IK scenarios (limbs, chains, free-form) |
| **Production-Ready Blending** | Override, additive, and multiply modes with bone masks |
| **Motion Matching Ready** | Feature extraction and database search infrastructure |

### 2.2 Secondary Goals

| Goal | Success Criteria |
|------|------------------|
| **Animation Compression** | Reduce memory footprint without visible quality loss |
| **Skeleton Retargeting** | Share animations across characters of different proportions |
| **Procedural Enhancement** | Add secondary motion (breathing, swaying) programmatically |
| **Crowd Scalability** | Animate thousands of entities with LOD-based quality |

### 2.3 Non-Goals

- Real-time motion capture integration (out of scope for this subsystem)
- Machine learning-based animation (no ML infrastructure assumed)
- VR/AR-specific animation handling (covered by XR subsystem)

---

## 3. Constraints

### 3.1 Technical Constraints

| Constraint | Implication |
|------------|-------------|
| **Python Performance** | Hot paths must be optimized or delegated to Rust/GPU |
| **4-Bone Influence Limit** | Skinning uses standard 4-bone-per-vertex maximum |
| **Quaternion Representation** | All rotations use quaternions (no Euler angle issues) |
| **ECS Architecture** | Systems must integrate with engine.core.ecs patterns |

### 3.2 Integration Constraints

| Constraint | Implication |
|------------|-------------|
| **Centralized Config** | All systems reference engine.animation.config |
| **Math Library** | Must use engine.core.math for Vec3, Quat, Mat4, Transform |
| **GPU Data Format** | Skinning matrices must be prepared in GPU-friendly format |

### 3.3 Quality Constraints

| Constraint | Implication |
|------------|-------------|
| **Numerical Stability** | Epsilon handling required for all division and acos operations |
| **Quaternion Normalization** | Must normalize after interpolation operations |
| **Thread Safety** | Systems may be called from ECS job scheduler |

---

## 4. Dependencies

### 4.1 Upstream Dependencies (Required)

| Module | Purpose |
|--------|---------|
| `engine.core.math` | Vec3, Quat, Mat4, Transform types |
| `engine.core.ecs` | Entity, World for ECS integration |
| `engine.animation.config` | Configuration dataclasses |
| `engine.animation.skeletal.constants` | Magic numbers and thresholds |

### 4.2 Downstream Dependents (Consume This Module)

| Module | Usage |
|--------|-------|
| `engine.rendering` | Skinning matrices for mesh rendering |
| `engine.animation.crowds` | Skeletal data for crowd instances |
| `engine.gameplay` | Character animation control |

### 4.3 Peer Dependencies (Horizontal)

| Module | Relationship |
|--------|--------------|
| `engine.animation.graph` | Animation graph execution |
| `engine.animation.motionmatching` | Motion matching database |
| `engine.animation.ik` | Standalone IK module (this duplicates for ECS) |

---

## 5. Architecture Summary

```
+--------------------------------------------+
|          Animation Systems (ECS)           |
|  [IK] [Procedural] [Facial] [Graph] [Crowd]|
+--------------------------------------------+
            |           |           |
            v           v           v
+--------------------------------------------+
|          Skeletal Animation Core           |
|  [Skeleton] [Pose] [Clip] [Blending]       |
|  [Skinning] [Compression] [Retargeting]    |
+--------------------------------------------+
            |
            v
+--------------------------------------------+
|         engine.core.math / ecs             |
+--------------------------------------------+
```

### Data Flow

1. **Clip Playback**: ClipPlayer samples AnimationClip at current time
2. **Pose Generation**: Keyframes interpolated to produce Pose
3. **Blending**: Multiple poses combined via LayeredBlender
4. **IK/Procedural**: Pose modified by IK solvers and procedural controllers
5. **Skinning**: Pose converted to skinning matrices for GPU

---

## 6. Risk Assessment

### 6.1 Known Issues (from Investigation)

| Issue | Severity | Location | Mitigation |
|-------|----------|----------|------------|
| Simplified foot contact preservation | Medium | retargeting.py:611-642 | Full FK pass needed for production |
| Basic lip sync audio analysis | Low | facial_system.py:452-495 | Replace with FFT or ML model |
| Up-vector preservation in rotation | Low | ik_system.py:489-502 | Review edge cases |

### 6.2 Integration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Python performance bottleneck | Medium | High | Profile and offload to Rust |
| GPU buffer format mismatch | Low | High | Validate against renderer requirements |

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Skinning performance | 10,000 entities @ 60fps | Benchmark test |
| IK convergence | <10 iterations for 95% cases | Unit tests |
| Compression ratio | 4:1 average | Compression tests |
| API coverage | 100% of documented components | Test coverage |
