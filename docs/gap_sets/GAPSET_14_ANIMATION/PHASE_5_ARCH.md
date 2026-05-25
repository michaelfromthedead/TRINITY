# Phase 5: Animation Graph Runtime -- Architecture

## Status: 7 [x] 0 [~] 1 [-]

## Module: `engine/animation/graph/`

### Files
| File | Lines | Purpose |
|------|-------|---------|
| animation_graph.py | 1039 | Graph DAG, metaclass, parameters, context |
| state_machine.py | 828 | State machine with transitions and conditions |
| blend_tree.py | 848 | 1D/2D/Direct parametric blend trees |
| blend_node.py | 775 | 8 node types for animation operations |
| layer.py | 551 | Animation layer stack with bone masks |
| sync.py | 671 | Marker-based animation synchronization |
| config.py | 99 | Graph system configuration |
| __init__.py | 246 | Public API, decorators, type re-exports |

### Architecture

**AnimationGraph** (`animation_graph.py`):
- `GraphNodeMeta`: metaclass for node registration
- `AnimationNode` base: evaluate() -> Pose, connect() interface
- `AnimationGraph`: DAG of nodes, parameters, connections
- `GraphParameter`: float/int/bool/trigger types with range validation
- `GraphContext`: parameter storage, runtime state per instance
- `SubgraphNode`: nested graph composition
- Evaluation: topological sort, dirty-flag pruning

**StateMachine** (`state_machine.py`):
- `AnimationState`: name, clip, speed, looping
- `StateTransition`: from/to, condition, blend_curve, sync_mode, priority
- `TransitionCondition`: 8 comparison ops (==, !=, <, >, <=, >=, AND, OR)
- `BlendCurve`: LINEAR, EASE_IN, EASE_OUT, STEP
- `StateMachine`: state registry, transition evaluation, current state tracking
- `StateMachineBuilder`: fluent builder pattern
- `@state_machine`: decorator for declarative state machine definition

**BlendTree** (`blend_tree.py`):
- `BlendTree1D`: single parameter, N entries, linear interpolation between nearest
- `BlendTree2D`: Cartesian/polar/freeform modes, Delaunay triangulation
- `BlendTreeDirect`: explicit per-entry weights
- `@blend_tree`: decorator for declarative blend tree definition

**BlendNode** (`blend_node.py`):
- `ClipNode`: single animation clip playback
- `BlendNode`: weighted blend between children
- `AdditiveNode`: additive overlay with weight
- `LayerNode`: masked layer application
- `MirrorNode`: L/R bone mirroring with configurable pairs
- `TimeScaleNode`: playback speed multiplier
- `PoseCacheNode`: cached evaluation for shared subtrees
- `SelectNode`: conditional child selection

**Layer** (`layer.py`):
- `LayerBlendMode`: OVERRIDE, ADDITIVE, MASKED_ADDITIVE
- `AnimationLayer`: bone mask, weight, blend mode, source clip
- `LayerStack`: ordered layer collection with accumulation
- `LayerStackBuilder`: fluent API for layer construction
- `BoneMaskPresets`: factory for common masks (upper_body, lower_body, arms, legs, gradient)

**Sync** (`sync.py`):
- `SyncMarker`: labeled time positions
- `SyncMarkerTrack`: ordered marker list per clip
- `SyncGroup`: clips synchronized by marker alignment
- `SyncMode`: NORMALIZED, PHASE, LEADER_FOLLOWER, WEIGHTED
- `EventSynchronizer`: event-aligned cross-clip synchronization

### Missing
- T-AN-5.8: Tests

### Key Design Decisions
- Metaclass-based node registration (GraphNodeMeta) enables extensibility
- BlendTree2D uses Delaunay triangulation for optimal interpolation weights
- State machine supports wildcard (*) transitions via priority system
- Layer stack supports additive blending for facial/procedural layering
- Sync groups support leader-follower mode for natural motion matching
- Dirty-flag evaluation prevents redundant graph traversal
