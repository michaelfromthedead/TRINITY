# CLARIFICATION: engine/gameplay/nav + engine/gameplay/quest

---

## Philosophical Framing

### Why In-House Navigation?

The navigation subsystem implements algorithms that are typically provided by game engines (Recast/Detour, Unity NavMesh). The decision to build custom:

1. **Full Control** — Algorithms can be tuned for specific game requirements
2. **No External Dependencies** — Engine remains self-contained
3. **Educational Value** — Implementation demonstrates mastery of AI navigation
4. **Customization** — Special behaviors (parabolic jumps, timed doors) integrate seamlessly

### Why Graph-Based Dialogue?

The quest subsystem uses a graph structure rather than state machines or scripted sequences:

1. **Non-Linear Narrative** — Graphs naturally represent branching dialogue
2. **Validation** — BFS/cycle detection catches authoring errors
3. **Tooling** — Graph structure maps to visual node editors
4. **Flexibility** — Random nodes, events, and branches mix freely

---

## Design Rationale

### Navigation Design Decisions

#### NavMesh Pipeline (Recast-Style)

The pipeline follows the industry-standard approach pioneered by Recast:

```
Geometry -> Voxelize -> Build Regions -> Trace Contours -> Generate Polygons
```

**Why voxelization?** Converting arbitrary geometry to voxels provides a uniform representation that handles complex shapes, overhangs, and multi-level structures.

**Why flood-fill for regions?** Connected component analysis naturally groups walkable areas, which become NavMesh polygons.

**Why Graham scan?** Convex polygons are required for efficient point-in-polygon tests and funnel algorithm compatibility.

#### Multiple Pathfinding Algorithms

| Algorithm | Best For |
|-----------|----------|
| A* | General-purpose, weighted graphs |
| JPS | Uniform-cost grids with many open areas |
| Theta* | Any-angle paths (smoother than A* post-processing) |
| HPA* | Large maps with hierarchical structure |

**Why all four?** Different game scenarios benefit from different algorithms. A single algorithm forces tradeoffs.

#### RVO + ORCA Layering

RVO (velocity sampling) and ORCA (half-plane solving) represent two generations of collision avoidance:

- **RVO**: Intuitive, robust, but O(n) velocity samples
- **ORCA**: Optimal, but requires linear programming

**Why both?** ORCA is primary; RVO provides fallback when ORCA solver fails (degenerate cases).

#### Steering Behaviors

Craig Reynolds' steering behaviors remain the gold standard for local movement:

- **Seek/Flee** — Basic attraction/repulsion
- **Arrive** — Deceleration on approach
- **Pursue/Evade** — Predictive interception
- **Wander** — Organic random movement
- **Flocking** — Emergent group behavior

**Why all of them?** Different AI archetypes need different behaviors. Guards pursue, animals wander, crowds flock.

### Quest Design Decisions

#### Three Variable Scopes

```
LocalVariableStore  <- Per-conversation, ephemeral
GlobalVariableStore <- Persistent across game
QuestVariableStore  <- Quest-linked, synced bidirectionally
```

**Why three scopes?** Dialogue state (local), world state (global), and quest state (quest) have different lifecycles and persistence requirements.

#### Operator Overloading for Conditions

```python
condition = (HasItem("key") & ~HasQuest("door_opened")) | QuestState("lockpick", "complete")
```

**Why operator overloading?** DSL-like readability for designers. The alternative (nested constructor calls) is unreadable.

#### Execute/Rollback Pattern

```python
class Effect:
    def execute(context) -> bool
    def rollback(context) -> None
```

**Why transactional?** Partial effect execution creates inconsistent state. If giving an item succeeds but starting a quest fails, the item must be taken back.

#### Composite Objectives

```python
CompositeObjective([KillObjective(...), CollectObjective(...)], mode="all")
```

**Why composite?** Quest objectives naturally form trees. "Kill 10 wolves AND collect 5 pelts" is a composite. "Talk to guard OR bribe guard" is also a composite.

#### Flow Patterns

| Pattern | Use Case |
|---------|----------|
| SequentialFlow | Main quest progression |
| ParallelFlow | Side objectives, exploration |
| BranchingFlow | Player choice, moral decisions |
| OptionalFlow | Bonus content, achievements |
| MixedFlow | Complex quests with all patterns |

**Why named patterns?** Explicit flow types make quest structure self-documenting and tool-friendly.

---

## Integration Points

### Navigation <-> Quest

- **ReachObjective** uses pathfinding to verify player proximity
- **EscortObjective** uses steering to control NPC movement
- **TalkObjective** uses NavLinks to position NPCs at interaction points

### Navigation <-> Combat

- **KillObjective** triggers when navigation detects combat proximity
- **Defend objectives** use spatial indexing to track area boundaries

### Quest <-> UI

- **DialogueGraph** provides nodes for UI rendering
- **Objectives** emit events for HUD updates
- **Variables** support observers for UI binding

---

## Classification Confidence

Both subsystems were classified as 100% REAL based on:

1. **Complete algorithm implementations** — No placeholder returns
2. **Working data structures** — Proper methods, not empty classes
3. **Serialization support** — Save/load functionality
4. **Event handling** — Observer patterns, callbacks
5. **Validation logic** — Error detection and handling
6. **Industry-standard patterns** — Recognized algorithms (Graham scan, A*, RVO, ORCA)
7. **No TODO placeholders** in core logic
8. **No NotImplementedError** in critical paths

The code evidence (Graham scan, JPS jump detection, ORCA half-plane computation, operator overloading, transactional rollback) demonstrates working implementations rather than stubs.
