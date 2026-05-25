# CLARIFICATION: Physics Simulation Architecture

**Domain**: engine/simulation  
**Subsystems**: components, constraints, softbody, vehicles

---

## Philosophical Framing

### What This System Is

The engine/simulation subsystem is a **production-quality physics foundation** built on mathematically rigorous algorithms. It is not a prototype, stub collection, or placeholder system. Every one of the 23 files demonstrates:

1. **Correct Mathematical Models** — Formulas match published physics literature
2. **Complete Algorithm Implementation** — No pass statements, no NotImplementedError patterns
3. **Physical Unit Consistency** — SI units throughout (N, m, kg, rad/s)
4. **Edge Case Handling** — Numerical stability safeguards in place

### Design Philosophy

The system follows a **constraint-based physics** paradigm where:

- **Rigid bodies** are constrained by joints and contacts
- **Soft bodies** are constrained by distance, volume, and material properties
- **Vehicles** are constrained by tire-road interaction and drivetrain dynamics

This unifying constraint abstraction enables:

- Consistent solver infrastructure (Jacobians, effective mass, warm starting)
- Predictable numerical behavior across subsystems
- Clear interfaces for force application and state query

---

## Design Rationale

### Why Position-Based Dynamics for Cloth?

PBD was chosen over force-based methods because:

1. **Unconditional Stability** — No explosion risk regardless of timestep
2. **Direct Control** — Stiffness parameters map intuitively to compliance
3. **Iterative Convergence** — Gauss-Seidel naturally handles constraint systems
4. **GPU Friendliness** — Constraint projection is embarrassingly parallel

The implementation includes inverse mass weighting, which is critical for correct energy distribution when particles have varying masses.

### Why Multiple Soft Body Solvers?

Three distinct soft body approaches serve different use cases:

| Solver | Use Case | Tradeoff |
|--------|----------|----------|
| **PBD (soft_body_pbd.py)** | Cloth, rope, jelly | Fast, stable, less physically accurate |
| **FEM (fem_solver.py)** | Elastic solids, muscle tissue | Accurate, computationally expensive |
| **Shape Matching (shape_matching.py)** | Character flesh, impact deformation | Fast approximation, good visual quality |

This triad covers the fidelity/performance spectrum that different game objects require.

### Why Pacejka Magic Formula?

The Pacejka tire model is the **industry standard** for vehicle simulation because:

1. **Empirical Accuracy** — Formula coefficients are fitted to real tire test data
2. **Combined Slip** — Handles simultaneous braking and cornering
3. **Load Sensitivity** — Correctly models force scaling with vertical load
4. **Proven Track Record** — Used in motorsport simulators worldwide

The implementation includes temperature and wear modeling for extended realism.

### Why Multiple Differential Types?

Different differential behaviors serve distinct driving feel targets:

| Type | Behavior | Use Case |
|------|----------|----------|
| **Open** | Equal torque split, no traction aid | Economy vehicles |
| **Limited Slip (LSD)** | Friction-based bias to slower wheel | Sports cars |
| **Torsen** | Gear-based bias with bias ratio | Performance vehicles |
| **Locked** | 50/50 split always | Off-road, drift vehicles |

Each type has distinct physics that affect vehicle handling character.

### Why Jacobian-Based Constraint Solving?

The Jacobian formulation enables:

1. **Unified Treatment** — All constraints (joints, contacts, limits) share the same solver
2. **Effective Mass** — Correct impulse distribution across linked bodies
3. **Warm Starting** — Reuse previous impulses for faster convergence
4. **Position Correction** — Baumgarte stabilization prevents drift accumulation

This architecture mirrors industry-standard engines (Bullet, PhysX, Box2D).

---

## Key Decisions Explained

### Baumgarte Stabilization vs. Post-Stabilization

The codebase uses **Baumgarte stabilization** (velocity-level bias term) rather than post-stabilization because:

- Simpler implementation within existing velocity solver
- Single-pass correction without separate position phase
- Tunable beta parameter controls position drift vs. energy injection tradeoff

### CFM/ERP Soft Constraints

Soft constraints use the Constraint Force Mixing (CFM) and Error Reduction Parameter (ERP) formulation from ODE physics:

```
gamma = 1.0 / (h * (damping + h * stiffness))
beta = h * stiffness * gamma
```

This converts spring-damper parameters into constraint solver terms, enabling springs/dampers within the unified constraint framework.

### Hill Muscle Model

The Hill-type muscle model was chosen for biomechanics because:

1. **Phenomenological Accuracy** — Captures force-length and force-velocity relationships
2. **Computational Efficiency** — Closed-form equations, no PDE solving
3. **Physiological Basis** — Parameters map to measurable muscle properties
4. **Animation Integration** — Activation signal drives procedural motion

---

## Architectural Boundaries

### What Belongs in engine/simulation

- Physics algorithms (constraint solving, integration, collision response)
- Material models (elastic, plastic, viscous)
- Vehicle dynamics (tire, drivetrain, aerodynamics)
- Soft body deformation (PBD, FEM, shape matching)

### What Does NOT Belong Here

- Rendering code (belongs in renderer-backend)
- Input handling (belongs in engine/input)
- AI/behavior (belongs in engine/ai)
- Audio triggers (belongs in engine/audio)

---

## Integration Points

### With Collision System

- Collider components provide support mapping for GJK/EPA
- Contact constraint receives manifold from narrow phase
- Inertia tensors computed per shape type

### With Renderer

- Deformable mesh provides vertex positions after simulation
- Surface normals recomputed post-deformation
- Collision proxy generated for culling

### With Entity System

- Components attach to entities via component_meta metaclass
- Simulation step iterates all active simulation components
- Sleep/wake state managed by entity activity tracker

---

## Numerical Considerations

### Floating Point Precision

- All computations use 64-bit float (Python default)
- Constraint error tolerance: 1e-6
- Singularity threshold: 1e-9 for matrix inversion

### Stability Safeguards

- Polar decomposition uses SVD fallback for degenerate cases
- Clamped impulse ratios prevent explosion
- Rest length hysteresis prevents oscillation

### Iteration Budgets

| Solver | Typical Iterations | Maximum |
|--------|-------------------|---------|
| Contact constraints | 4-8 | 20 |
| PBD distance | 10-20 | 50 |
| FEM Newton | 3-5 | 10 |
