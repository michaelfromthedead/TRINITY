# SUMMARY: trinity/decorators (Part 2)

## Metrics Table

| Metric | Value |
|--------|-------|
| Total Files | 39 |
| Total Lines | ~6,900 |
| Decorators Defined | ~110 |
| Utility Functions | ~10 (stacks + introspection) |
| Config Dataclasses | 9 |
| Validation Functions | 39+ |
| Step Builder Functions | 39+ |
| After-Apply Functions | 39+ |

## File Breakdown by Size

| Size Category | Files | Lines Range |
|---------------|-------|-------------|
| Large (300+) | 4 | 331-354 |
| Medium (200-299) | 14 | 191-292 |
| Small (100-199) | 20 | 99-196 |
| Compact (<100) | 1 | 99 |

## Algorithm Inventory Table

| Algorithm/Pattern | Location | Status | Notes |
|-------------------|----------|--------|-------|
| make_decorator factory | ops.py (referenced) | REAL | Core decorator creation |
| validate_* functions | all files | REAL | Parameter validation |
| *_steps functions | all files | REAL | Op sequence builders |
| *_after_apply functions | all files | REAL | Post-application setup |
| Stack class | stacks.py | REAL | Decorator composition |
| _validate_stack_combination | stacks.py | REAL | Anti-pattern detection |
| primitives() | introspection.py | REAL | Get Steps from target |
| composites() | introspection.py | REAL | Get decorator names |
| chain() | introspection.py | REAL | Human-readable chain |
| compose() | introspection.py | REAL | Anonymous decorator |
| find_decorators() | introspection.py | REAL | Decorator discovery |
| Config dataclasses | network_extended, debug_extended, assets | REAL | Typed configuration |

## Decorator Categories

| Category | File | Count | Examples |
|----------|------|-------|----------|
| Gameplay | gameplay.py | 6 | @ability, @buff, @spawner |
| World Building | world_building.py | 6 | @foliage_type, @water_body |
| Lifecycle | lifecycle.py | 5 | @on_add, @on_remove |
| Assets | assets.py | 5 | @asset, @preload, @cook |
| LOD/Streaming | lod_streaming.py | 5 | @lod, @streamable |
| IK/Procedural | ik_procedural.py | 5 | @ik_chain, @ragdoll |
| Game AI | game_ai.py | 5 | @behavior_tree, @utility_ai |
| Save System | save_system.py | 4 | @save_slot, @cloud_sync |
| Social | social.py | 4 | @social, @leaderboard |
| Economy | economy.py | 4 | @currency, @transaction |
| Localization | localization.py | 4 | @localized, @plural |
| Time | time.py | 4 | @pausable, @rewindable |
| Error Handling | error_handling.py | 4 | @crash_safe, @recoverable |
| Build/Deploy | build_deploy.py | 4 | @build_only, @feature_flag |
| Network Extended | network_extended.py | 4 | @interest, @bandwidth_priority |
| Debug/Safety | debug_safety.py | 4 | @reads, @writes |
| Security | security.py | 4 | @validated, @rate_limited |
| Achievements | achievements.py | 3 | @achievement, @progress |
| State Machine | state_machine.py | 3 | @state_machine, @on_enter |
| Analytics | analytics.py | 3 | @telemetry, @funnel |
| Narrative | narrative.py | 3 | @dialogue, @voice_over |
| Debug/Cheat | debug_cheat.py | 3 | @cheat, @debug_draw |
| Procedural | procedural.py | 3 | @seeded, @procedural |
| Replay | replay.py | 3 | @recorded, @keyframe |
| UI | ui.py | 2 | @widget, @layout |
| Spatial | spatial.py | 2 | @spatial, @partitioned |
| Animation | animation.py | 2 | @tween, @blend_tree |
| Input | input.py | 2 | @input_action, @input_axis |
| Cinematics | cinematics.py | 2 | @cutscene, @camera_track |
| Composition | composition.py | 2 | @composite, @alias |
| Transactions | transactions.py | 2 | @transactional, @undoable |
| Prefabs | prefabs.py | 2 | @prefab, @extends |
| Debug Extended | debug_extended.py | 2 | @network_debug, @automation_test |
| Accessibility | accessibility.py | 1 | @accessible |
| RPC | rpc.py | 1 | @rpc |
| Platform | platform_specifics.py | 1 | @battery_aware |

## Key Evidence Snippets

### Validation Pattern
```python
# From gameplay.py - actionable error messages
if stacking not in VALID_STACKING:
    raise ValueError(
        f"@buff: invalid stacking '{stacking}'. "
        f"Valid stacking modes: {sorted(VALID_STACKING)}"
    )
```

### Step Builder Pattern
```python
# From gameplay.py - Op sequence generation
def _ability_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "ability", "value": True}),
        Step(Op.TAG, {"key": "ability_cost", "value": dict(cost)}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]
```

### Stack Anti-Pattern Detection
```python
# From stacks.py - prevents contradictory decorators
if "parallel" in names and "exclusive" in names:
    raise ValueError(
        "Stack contains both @parallel and @exclusive which are contradictory"
    )
```

### Config Dataclass Pattern
```python
# From network_extended.py - typed configuration
@dataclass(frozen=True)
class InterestConfig:
    radius: float
    priority: int
    update_hz: float
```