# CLARIFICATION: engine_animation_crowds_facial

**Philosophical Framing and Design Rationale**
**Generated:** 2026-05-23

---

## Design Philosophy

### Core Principle: Data-Oriented Design

The crowds and facial subsystems embody TRINITY's data-oriented philosophy:

1. **Separation of Data and Logic**
   - Crowds: `CrowdAgent` is pure data; behaviors are separate processors
   - Facial: Blend shape weights are data; rig is the evaluator

2. **Cache-Efficient Access Patterns**
   - Animation textures: Sequential pixel access for GPU
   - Instance buffers: Contiguous packed data
   - Sparse blend shapes: Only affected vertices

3. **Stateless Behavior Processing**
   - Behaviors receive context, return decisions
   - No hidden state in behavior objects
   - Enables parallel evaluation (future)

---

## Why These Architectures?

### Crowds: GPU-First Design

**Question:** Why bake animations to textures instead of skeletal animation?

**Answer:** GPU crowd rendering requires different trade-offs than traditional skeletal animation:

1. **Vertex Shader Sampling**
   - Traditional: CPU computes transforms, uploads per-instance
   - Texture: GPU samples texture, no CPU-GPU transfer per frame
   - Result: 10-100x more instances possible

2. **Memory vs Compute Trade-off**
   - Animation textures use GPU memory (static)
   - But save GPU bandwidth (no per-frame uploads)
   - LOD further reduces memory for distant crowds

3. **Instance Homogeneity**
   - All instances share same texture atlas
   - Variation through animation time offset, speed, clip selection
   - Enables massive batching

### Facial: Layer-Based Composition

**Question:** Why priority-based layers instead of a single blend?

**Answer:** Facial animation has natural semantic layers:

1. **Override Semantics**
   - Designer needs explicit control order
   - Lip sync must override idle but not override cutscene
   - Priority makes this explicit and debuggable

2. **Additive vs Replace**
   - Emotions add to neutral
   - Lip sync replaces mouth region
   - Per-layer mode captures intent

3. **Subsystem Independence**
   - Eye controller doesn't know about lip sync
   - Lip sync doesn't know about procedural blinks
   - Layers compose without coupling

---

## Key Design Decisions

### Decision 1: FACS Over Custom Expressions

**Rationale:**
- FACS (Facial Action Coding System) is scientifically validated
- Maps directly to muscle anatomy
- Industry standard (UE, Unity, AAA studios)
- Enables mocap retargeting

**Trade-off:**
- 21 AUs is less intuitive than "smile slider"
- Requires artist training
- But enables any expression, not just presets

### Decision 2: Sparse Blend Shapes

**Rationale:**
- Most blend shapes affect <10% of vertices
- Dense representation wastes memory
- Sparse allows more blend shapes per budget

**Implementation:**
```python
vertex_indices: np.ndarray  # Which vertices move
deltas: np.ndarray          # How they move (N x 3)
```

**Trade-off:**
- Indirection cost on evaluation
- But NumPy handles it efficiently
- And memory savings are 10x+

### Decision 3: RVO-Style Avoidance Over Flow Fields

**Rationale:**
- RVO (Reciprocal Velocity Obstacles) is agent-local
- No global grid to maintain
- Handles dynamic obstacles naturally
- Priority weighting for asymmetric avoidance

**Trade-off:**
- O(n) neighbor queries (current gap)
- But agents avoid each other correctly
- Spatial hash would fix scaling

### Decision 4: Coarticulation for Lip Sync

**Rationale:**
- Human speech naturally blends between sounds
- Instant viseme switching looks robotic
- Coarticulation models anticipation and carryover

**Implementation:**
- Anticipation: Start blending to next phoneme early
- Carryover: Previous phoneme persists briefly
- Result: Natural-looking speech

---

## Architectural Boundaries

### What Crowds Owns
- Agent state and behavior simulation
- Animation texture baking and sampling
- LOD level selection and transitions
- Instance buffer management

### What Crowds Does NOT Own
- Pathfinding (NavMesh integration needed)
- Physical collision (uses avoidance, not physics)
- Actual GPU dispatch (Rust backend)

### What Facial Owns
- Blend shape evaluation
- FACS AU mapping and blending
- Lip sync timeline playback
- Eye procedural animation
- Face rig layer composition

### What Facial Does NOT Own
- Audio analysis for lip sync (receives PhonemeEvents)
- Mesh deformation (outputs weights, not vertices)
- Actual rendering (outputs to blend shape system)

---

## Evolution Rationale

### Why These Subsystems Together?

1. **Both animate characters** - Different scales, same pipeline stage
2. **Both use GPU resources** - Texture atlases, instance data
3. **Both need LOD** - Crowds by distance, facial by camera proximity
4. **Natural partition** - Hero characters (facial) vs background (crowds)

### What's Missing (By Design)

1. **Flow Fields**
   - Useful for massive crowds (10K+)
   - Current target is 1K crowds
   - Can be added without architectural change

2. **Neural Lip Sync**
   - Requires ML inference runtime
   - Current phoneme-based approach works
   - Plugin architecture could add this

3. **GPU Compute Agents**
   - Requires compute shader support
   - Python prototype validates algorithms
   - Rust backend can add compute path

---

## Relationship to TRINITY Pattern

### Metaclass Integration

Both subsystems follow TRINITY patterns:

| Pattern | Crowds Usage | Facial Usage |
|---------|-------------|--------------|
| `@component` | CrowdAgent could be component | Face rig state |
| `@system` | CrowdSimulator as system | FaceRig.update() |
| Config injection | CROWD_BEHAVIOR_CONFIG | FacialConfig |

### Not Yet Migrated

Current code is vanilla Python, not yet decorated with:
- `@component` for agent/rig data
- `@system` for simulation loops
- TrackedDescriptor for dirty tracking

**Reason:** These are engine subsystems, not user-facing components. Migration is a separate phase.

---

## Quality Attributes

### Correctness
- All algorithms have edge case handling
- Division-by-zero protection throughout
- Buffer overflow protection with exceptions

### Performance
- NumPy vectorization for batch operations
- Sparse representations where beneficial
- LOD to reduce work for distant entities

### Maintainability
- Configuration externalized
- Clear module boundaries
- Dependency injection via config

### Extensibility
- Behavior system is pluggable
- Face rig layers are extensible
- ARKit compatibility enables external input
