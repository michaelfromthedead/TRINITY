# PEDAGOGY: engine_gameplay_abilities_ai_camera

**Created**: 2026-05-23
**Purpose**: Concept evolution log (append-only)

---

## Pass 1: engine_gameplay_abilities.md

### Concept: Attribute System
- **Prior Value**: None (new concept)
- **New Value**: Complete attribute system with modifiers following 5-stage order of operations (ADD_BASE -> MULTIPLY_BASE -> ADD_BONUS -> MULTIPLY_BONUS -> OVERRIDE -> Clamp)
- **Reason**: First introduction from abilities investigation

### Concept: Effect Types
- **Prior Value**: None (new concept)
- **New Value**: Four effect types (Instant, Duration, Infinite, Periodic) with full lifecycle management
- **Reason**: First introduction from abilities investigation

### Concept: Targeting System
- **Prior Value**: None (new concept)
- **New Value**: Five targeting modes (Self, Actor, Point, Area, Confirmation) with geometric shape support
- **Reason**: First introduction from abilities investigation

### Concept: Gameplay Tags
- **Prior Value**: None (new concept)
- **New Value**: Hierarchical tag system with wildcards, containers, queries, and LRU-cached registry
- **Reason**: First introduction from abilities investigation

---

## Pass 2: engine_gameplay_ai.md

### Concept: Behavior Tree
- **Prior Value**: None (new concept)
- **New Value**: Complete BT with 14 node types (3 composite, 7 decorator, 4 leaf), depth limiting, debug tracing
- **Reason**: First introduction from AI investigation

### Concept: GOAP
- **Prior Value**: None (new concept)
- **New Value**: A* GOAP planner with WorldState, Goal, GOAPAction, plan caching (100 plans, 5s TTL)
- **Reason**: First introduction from AI investigation

### Concept: Utility AI
- **Prior Value**: None (new concept)
- **New Value**: 8 response curves with compensation factor scoring and momentum for action stability
- **Reason**: First introduction from AI investigation

### Concept: Perception
- **Prior Value**: None (new concept)
- **New Value**: Multi-sense system (6 types) with stimuli aging, decay, and memory persistence
- **Reason**: First introduction from AI investigation

### Concept: Blackboard
- **Prior Value**: None (new concept)
- **New Value**: Key-value store with namespaces, observers, TTL, scopes, typed keys
- **Reason**: First introduction from AI investigation

### Concept: Combat AI
- **Prior Value**: None (new concept)
- **New Value**: 9 behaviors, threat assessment, target priorities, health retreat threshold
- **Reason**: First introduction from AI investigation

---

## Pass 3: engine_gameplay_camera.md

### Concept: Camera Controllers
- **Prior Value**: None (new concept)
- **New Value**: 8 controllers (FirstPerson, ThirdPerson, Orbit, Follow, Free, Cinematic, TopDown, Isometric) with lag formula
- **Reason**: First introduction from camera investigation

### Concept: Camera Collision
- **Prior Value**: None (new concept)
- **New Value**: 5 response modes with 9-ray sphere cast and occlusion fade states
- **Reason**: First introduction from camera investigation

### Concept: Camera Effects
- **Prior Value**: None (new concept)
- **New Value**: 7 shake types + FOV, tilt, DOF, motion blur, vignette effects
- **Reason**: First introduction from camera investigation

### Concept: Blending
- **Prior Value**: None (new concept)
- **New Value**: 12 blend curves including elastic/bounce, blend stack, 7 split-screen layouts
- **Reason**: First introduction from camera investigation

### Concept: Camera Rails
- **Prior Value**: None (new concept)
- **New Value**: 4 spline types with arc-length parameterization, dolly/crane helpers
- **Reason**: First introduction from camera investigation

---

## Pass 4: engine_gameplay_abilities_ai_camera.md

### Concept: Total Line Count
- **Prior Value**: ~14,719 (sum of individual reports)
- **New Value**: ~14,383 (consolidated report)
- **Reason**: Consolidated document provides unified count; difference may be due to overlapping analysis or rounding

### Concept: Classification
- **Prior Value**: Individual "REAL" classifications per subsystem
- **New Value**: Unified "ALL REAL, NO STUBS FOUND" classification
- **Reason**: Consolidated document confirms consistent quality across all three subsystems

### Concept: Dependencies
- **Prior Value**: Scattered mentions in individual docs
- **New Value**: Unified dependency list (Vec3, Quat, Mat4, PhysicsWorld, TransformComponent, constants modules, Actor)
- **Reason**: Consolidated document synthesizes cross-subsystem dependencies

### Concept: Architecture Patterns
- **Prior Value**: Scattered pattern mentions
- **New Value**: Unified pattern catalog (Strategy, Observer, Composite, State Machine, Factory, Builder)
- **Reason**: Consolidated document identifies patterns used across all subsystems
