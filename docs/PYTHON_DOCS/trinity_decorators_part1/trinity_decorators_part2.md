# Trinity Decorators Archaeological Investigation - Part 2

**Date**: 2026-05-22
**Investigator**: Research Agent
**Scope**: 39 decorator files in `trinity/decorators/`
**Total Lines Analyzed**: ~6,900

## Executive Summary

**Classification: 100% REAL IMPLEMENTATIONS**

All 39 files examined contain fully implemented decorator systems using the ops-based architecture. Every file follows the same robust pattern:
1. Parameter validators with meaningful error messages
2. Step builders producing `Op` sequences (TAG, REGISTER, HOOK, etc.)
3. After-apply functions setting class/function attributes
4. Registry registration with proper `DecoratorSpec`
5. Well-defined exports via `__all__`

Zero stub files. Zero `pass` placeholders. Zero `NotImplementedError` raises.

## File Analysis Table

| File | Lines | Classification | Decorators | Pattern |
|------|-------|----------------|------------|---------|
| assets.py | 354 | REAL | @asset, @preload, @cook, @residency, @import_settings | Full ops+config dataclasses |
| gameplay.py | 349 | REAL | @ability, @buff, @gameplay_tag, @spawner, @interactable, @quest | Full validation chain |
| world_building.py | 339 | REAL | @foliage_type, @procedural_placement, @level_instance, @water_body, @navmesh_modifier, @trigger_volume | Full with VALID_* constants |
| network_extended.py | 331 | REAL | @interest, @bandwidth_priority, @snapshot_interpolation, @server_reconcile | Config dataclasses |
| lod_streaming.py | 292 | REAL | @lod, @streamable, @chunk, @loading_priority, @unloadable | Full validation |
| ik_procedural.py | 288 | REAL | @ik_chain, @ik_goal, @procedural_bone, @motion_matching, @ragdoll | Full with solver enums |
| debug_safety.py | 278 | REAL | @reads, @writes, @trace_stack, @track_changes | Manual + make_decorator mix |
| game_ai.py | 247 | REAL | @behavior_tree, @utility_ai, @blackboard, @ai_debug, @perception | Full ops chain |
| save_system.py | 236 | REAL | @save_slot, @atomic_save, @cloud_sync, @save_migration | Full conflict resolution |
| social.py | 233 | REAL | @social, @leaderboard, @shareable, @presence | Full ops chain |
| economy.py | 228 | REAL | @currency, @transaction, @mtx, @daily_reward | Full validation |
| localization.py | 227 | REAL | @localized, @plural, @rtl_aware, @text_overflow | Full with validate_target_type |
| time.py | 224 | REAL | @time_scale, @pausable, @rewindable, @deterministic | Full with interpolation enums |
| audio.py | 218 | REAL | @sound, @audio_bus, @spatial_audio | Full falloff validation |
| error_handling.py | 216 | REAL | @crash_safe, @recoverable, @error_boundary, @bug_report | Full recovery strategies |
| build_deploy.py | 216 | REAL | @build_only, @strip_in_release, @asset_bundle, @feature_flag | Full ops chain |
| debug_extended.py | 212 | REAL | @network_debug, @automation_test | Full with config dataclasses |
| achievements.py | 197 | REAL | @achievement, @progress, @stat | Full validation chain |
| state_machine.py | 196 | REAL | @state_machine, @on_enter, @on_exit | Full transition validation |
| security.py | 196 | REAL | @server_authoritative, @validated, @rate_limited, @encrypted | Full rate scope validation |
| analytics.py | 195 | REAL | @telemetry, @funnel, @heatmap | Full consent level validation |
| narrative.py | 192 | REAL | @dialogue, @conversation, @voice_over | Full ops chain |
| debug_cheat.py | 191 | REAL | @cheat, @debug_draw, @inspector | Full range validation |
| lifecycle.py | 189 | REAL | @on_add, @on_remove, @on_change, @on_spawn, @on_despawn | Full hook system |
| procedural.py | 179 | REAL | @seeded, @procedural, @constraint | Full seed source validation |
| replay.py | 171 | REAL | @recorded, @replay_authority, @keyframe | Full frequency/source validation |
| ui.py | 165 | REAL | @widget, @layout | Full direction validation |
| spatial.py | 165 | REAL | @spatial, @partitioned | Full structure validation |
| animation.py | 160 | REAL | @tween, @blend_tree | Full easing validation |
| input.py | 159 | REAL | @input_action, @input_axis | Full binding validation |
| cinematics.py | 155 | REAL | @cutscene, @camera_track | Full blend validation |
| composition.py | 138 | REAL | @composite, @alias | Full callable validation |
| transactions.py | 137 | REAL | @transactional, @undoable | Full isolation level validation |
| prefabs.py | 133 | REAL | @prefab, @extends | Full parent validation |
| stacks.py | 122 | REAL | Stack, stack(), parameterized_stack | Stack composition logic |
| accessibility.py | 120 | REAL | @accessible | Full role validation |
| rpc.py | 118 | REAL | @rpc | Full authority validation |
| platform_specifics.py | 111 | REAL | @battery_aware | Full battery mode validation |
| introspection.py | 99 | REAL | primitives(), composites(), chain(), find_decorators(), compose() | Introspection API |

## Key Findings

### 1. Consistent Architecture Pattern

Every decorator file follows the same structural pattern:

```python
# 1. Valid constants (frozenset)
VALID_SOMETHING = frozenset({"option1", "option2"})

# 2. Validators (_validate_decoratorname)
def _validate_decoratorname(**kwargs) -> None:
    # Raise ValueError/TypeError on invalid params
    
# 3. Step builders (_decoratorname_steps)
def _decoratorname_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "...", "value": ...}),
        Step(Op.REGISTER, {"registry": "..."}),
    ]

# 4. After-apply (_after_decoratorname or _decoratorname_after_apply)
def _after_decoratorname(target, params) -> Any:
    target._decoratorname = True
    # Set other attributes
    
# 5. Decorator creation
decoratorname = make_decorator(
    name="decoratorname",
    steps=_decoratorname_steps,
    validate=_validate_decoratorname,
    after_steps=_after_decoratorname,
)

# 6. Registry registration
```

### 2. Op Types Used

The decorators use these primitive operations:
- `Op.TAG` - Store metadata on target
- `Op.REGISTER` - Register in named registry
- `Op.HOOK` - Register lifecycle hooks
- `Op.TRACK` - Enable change tracking
- `Op.VALIDATE` - Runtime validation
- `Op.DESCRIBE` - Documentation generation

### 3. Validation Quality

All validators provide specific, actionable error messages:

```python
# From gameplay.py
if stacking not in VALID_STACKING:
    raise ValueError(
        f"@buff: invalid stacking '{stacking}'. "
        f"Valid stacking modes: {sorted(VALID_STACKING)}"
    )

# From state_machine.py  
if initial not in states_set:
    raise ValueError(
        f"@state_machine: initial state '{initial}' is not in states {sorted(states_set)}"
    )
```

### 4. Notable Special Cases

**stacks.py** - Not a decorator file, but a Stack composition utility:
- `Stack` class for combining decorators
- `_validate_stack_combination` prevents anti-patterns
- Warns about `@networked` without `@track_changes`
- Errors on contradictory `@parallel` + `@exclusive`

**introspection.py** - API for querying decorator metadata:
- `primitives(cls, field)` - Get Steps
- `composites(cls, field)` - Get decorator names
- `chain(cls, field)` - Human-readable chain
- `compose(*steps)` - Create anonymous decorator

**debug_safety.py** - Mixed pattern with both:
- Manual decorator (`@reads`, `@writes`) using `run_steps`
- make_decorator style (`@trace_stack`, `@track_changes`)

### 5. Tier Distribution

The 39 files span tiers 7-51:
- Tier 7: lifecycle
- Tier 8: assets
- Tier 10-11: debug_safety
- Tier 32: platform_specifics
- Tier 33-51: Various game systems

### 6. Config Dataclasses

Several files use frozen dataclasses for configuration:
- `network_extended.py`: InterestConfig, BandwidthPriorityConfig, SnapshotInterpolationConfig, ServerReconcileConfig
- `debug_extended.py`: NetworkDebugConfig, AutomationTestConfig
- `assets.py`: AssetConfig, CookConfig, ResidencyConfig, ImportSettingsConfig

## Evidence Snippets

### Full Implementation Example (gameplay.py)

```python
def _validate_ability(
    cost: dict[str, float] = {},
    cooldown: float = 0.0,
    tags: set[str] = set(),
    blocked_by: set[str] = set(),
    **_: Any
) -> None:
    """Validate @ability parameters."""
    if cooldown < 0:
        raise ValueError(f"@ability: cooldown must be >= 0, got {cooldown}")


def _ability_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ability decorator."""
    cost = params.get("cost", {})
    cooldown = params.get("cooldown", 0.0)
    tags = params.get("tags", set())
    blocked_by = params.get("blocked_by", set())

    return [
        Step(Op.TAG, {"key": "ability", "value": True}),
        Step(Op.TAG, {"key": "ability_cost", "value": dict(cost)}),
        Step(Op.TAG, {"key": "ability_cooldown", "value": cooldown}),
        Step(Op.TAG, {"key": "ability_tags", "value": set(tags)}),
        Step(Op.TAG, {"key": "ability_blocked_by", "value": set(blocked_by)}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _after_ability(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @ability is applied."""
    target._ability = True
    target._ability_cost = dict(params.get("cost", {}))
    target._ability_cooldown = params.get("cooldown", 0.0)
    target._ability_tags = set(params.get("tags", set()))
    target._ability_blocked_by = set(params.get("blocked_by", set()))
    return None


ability = make_decorator(
    name="ability",
    steps=_ability_steps,
    doc="Declare gameplay ability with cost, cooldown, and tags.",
    validate=_validate_ability,
    after_steps=_after_ability,
)
```

### Stack Validation (stacks.py)

```python
def _validate_stack_combination(decorators: tuple[Callable, ...]) -> None:
    """Check for known anti-pattern combinations in a stack."""
    names: set[str] = set()
    for d in decorators:
        name = getattr(d, "_decorator_name", None)
        if name is None:
            name = getattr(d, "__name__", None) or getattr(d, "__qualname__", "")
        if name:
            names.add(name)

    # Hard errors - contradictory combinations
    if "parallel" in names and "exclusive" in names:
        raise ValueError(
            "Stack contains both @parallel and @exclusive which are contradictory"
        )

    # Warnings - likely mistakes
    if "networked" in names and "track_changes" not in names:
        warnings.warn(
            "@networked without @track_changes: delta sync requires change tracking",
            UserWarning,
            stacklevel=3,
        )
```

## Decorator Count Summary

| Category | Count |
|----------|-------|
| Gameplay | 6 |
| World Building | 6 |
| Lifecycle | 5 |
| Assets | 5 |
| LOD/Streaming | 5 |
| IK/Procedural | 5 |
| Save System | 4 |
| Social | 4 |
| Economy | 4 |
| Localization | 4 |
| Time | 4 |
| Error Handling | 4 |
| Build/Deploy | 4 |
| Network Extended | 4 |
| Debug/Safety | 4 |
| Security | 4 |
| Game AI | 5 |
| Achievements | 3 |
| State Machine | 3 |
| Analytics | 3 |
| Narrative | 3 |
| Debug/Cheat | 3 |
| Procedural | 3 |
| Replay | 3 |
| UI | 2 |
| Spatial | 2 |
| Animation | 2 |
| Input | 2 |
| Cinematics | 2 |
| Composition | 2 |
| Transactions | 2 |
| Prefabs | 2 |
| Debug Extended | 2 |
| Accessibility | 1 |
| RPC | 1 |
| Platform | 1 |
| **Total Decorators** | **~110** |

Plus:
- Stack utilities (3 functions)
- Introspection API (7 functions)

## Conclusion

The trinity/decorators directory is a comprehensive, production-quality decorator framework. All 39 files examined contain real implementations with:
- Proper parameter validation
- Step-based ops architecture
- Registry integration
- Clean exports

No stubs or placeholder code found. The system is architecturally complete with ~110 decorators spanning game engine functionality from assets to networking to gameplay systems.
