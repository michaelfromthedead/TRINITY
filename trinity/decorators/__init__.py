"""
Trinity Pattern Decorators.

This package provides the decorator layer of the Trinity Pattern.

The decorator system is built on 7 fundamental primitives (Chomsky Grammar):
    REGISTER   - Add class to a registry
    DESCRIBE   - Extract schema from annotations
    TRACK      - Monitor field changes
    VALIDATE   - Enforce constraints on fields
    INTERCEPT  - Wrap field get/set/delete
    HOOK       - Attach lifecycle callbacks
    TAG        - Attach queryable metadata

All ~40 decorators are compositions of these primitives.
"""

from trinity.decorators.accessibility import (
    accessible,
)
from trinity.decorators.achievements import (
    achievement,
    progress,
    stat,
)
from trinity.decorators.ai_generation import (
    complexity,
    constraints,
    example,
    generates,
    pattern,
    pure,
    stub,
)
from trinity.decorators.analytics import (
    funnel,
    heatmap,
    telemetry,
)
from trinity.decorators.animation import (
    blend_tree,
    tween,
)
from trinity.decorators.assets import (
    AssetConfig,
    CookConfig,
    ImportSettingsConfig,
    ResidencyConfig,
    asset,
    cook,
    import_settings,
    preload,
    residency,
)
from trinity.decorators.audio import (
    audio_bus,
    sound,
    spatial_audio,
)
from trinity.decorators.audio_extended import (
    audio_snapshot,
    dsp_node,
    music_stem,
    music_transition,
    occlusion,
    reverb_zone,
    sidechain,
    voice_priority,
)
from trinity.decorators.base import (
    PlatformUnavailableError,
    attach_attributes,
    check_excluded_decorators,
    check_required_decorators,
    create_unavailable_stub,
    get_applied_decorators,
    get_attribute,
    get_current_arch,
    get_current_platform,
    get_decorator_chain,
    has_decorator,
    inspect_decorated,
    merge_attributes,
    track_decorator,
    validate_parameters,
    validate_target_type,
)
from trinity.decorators.build_deploy import (
    asset_bundle,
    build_only,
    feature_flag,
    strip_in_release,
)
from trinity.decorators.builtin_stacks import (
    bandwidth_efficient,
    complete_ai,
    competitive_entity,
    deterministic_data,
    lod_scalable,
    mmo_entity,
    moddable_content,
    multiplayer_character,
    networked_entity,
    open_world_entity,
    predicted_entity,
    production_component,
    profiled_dev,
    replay_ready,
    safe_system,
    saveable_data,
    secure_multiplayer,
    streaming_chunk,
    versioned_saveable,
)
from trinity.decorators.cinematics import (
    camera_track,
    cutscene,
)
from trinity.decorators.compilation import (
    BackendConfig,
    CapabilityConfig,
    FFIConfig,
    NativeConfig,
    PlatformConfig,
    TargetConfig,
    backend,
    capability,
    ffi,
    native,
    platform,
    target,
    unsafe,
)
from trinity.decorators.composition import (
    alias,
    composite,
)
from trinity.decorators.crafting import (
    crafting_station,
    ingredient,
    loot_table,
    recipe,
    salvage_recipe,
)
from trinity.decorators.data_flow import (
    NetworkedConfig,
    SerializableConfig,
    SnapshotConfig,
    VersionedConfig,
    networked,
    serializable,
    snapshot,
    versioned,
)
from trinity.decorators.debug_cheat import (
    cheat,
    debug_draw,
    inspector,
)
from trinity.decorators.debug_extended import (
    automation_test,
    network_debug,
)
from trinity.decorators.debug_safety import (
    reads,
    trace_stack,
    track_changes,
    writes,
)
from trinity.decorators.destruction import (
    damage_resistance,
    damage_type,
    destructible,
    fracture,
    joint,
    physics_material,
)
from trinity.decorators.dev import (
    ProfileConfig,
    ReloadableConfig,
    bench,
    deprecated,
    editor,
    gpu_profile,
    invariant,
    profile,
    reloadable,
    test,
    trace,
)
from trinity.decorators.economy import (
    currency,
    daily_reward,
    mtx,
    transaction,
)
from trinity.decorators.ecs_core import (
    bundle,
    component,
    derived,
    event,
    query,
    relation,
    resource,
    system,
    tag,
)
from trinity.decorators.error_handling import (
    bug_report,
    crash_safe,
    error_boundary,
    recoverable,
)
from trinity.decorators.game_ai import (
    ai_debug,
    behavior_tree,
    blackboard,
    perception,
    utility_ai,
)
from trinity.decorators.gameplay import (
    ability,
    buff,
    gameplay_tag,
    interactable,
    quest,
    spawner,
)
from trinity.decorators.gpu import (
    VALID_BUFFER_USAGE,
    GpuBufferConfig,
    GpuKernelConfig,
    Mat4,
    RenderPassConfig,
    ShaderConfig,
    Vec2,
    Vec3,
    Vec4,
    WgpuBufferAllocation,
    allocate_wgpu_buffer,
    async_compute,
    bind_group,
    create_wgpu_buffer,
    dispatch,
    gpu_buffer,
    gpu_kernel,
    gpu_struct,
    render_pass,
    shader,
)
from trinity.decorators.ik_procedural import (
    ik_chain,
    ik_goal,
    motion_matching,
    procedural_bone,
    ragdoll,
)
from trinity.decorators.input import (
    input_action,
    input_axis,
)
from trinity.decorators.lifecycle import (
    on_add,
    on_change,
    on_despawn,
    on_remove,
    on_spawn,
)
from trinity.decorators.localization import (
    localized,
    plural,
    rtl_aware,
    text_overflow,
)
from trinity.decorators.lod_streaming import (
    chunk,
    loading_priority,
    lod,
    streamable,
    unloadable,
)
from trinity.decorators.memory import (
    AlignedConfig,
    AllocatorConfig,
    ArenaConfig,
    BudgetConfig,
    InlineArrayConfig,
    PackedConfig,
    PoolConfig,
    aligned,
    allocator,
    arena,
    atomic,
    budget,
    copy_on_write,
    flyweight,
    generations,
    inline_array,
    intern,
    packed,
    pooled,
)
from trinity.decorators.modding import (
    conflicts,
    load_order,
    mod,
    mod_extends,
    moddable,
    patch,
    provides,
    replaces,
    requires,
)
from trinity.decorators.narrative import (
    conversation,
    dialogue,
    voice_over,
)
from trinity.decorators.network_extended import (
    bandwidth_priority,
    interest,
    server_reconcile,
    snapshot_interpolation,
)
from trinity.decorators.particles_vfx import (
    decal,
    gpu_particle,
    particle_emitter,
    particle_module,
    trail,
    vfx_event,
)
from trinity.decorators.physics_sim import (
    buoyancy,
    continuous_collision,
    simulation_domain,
    sleep_threshold,
    solver_hint,
    substep,
    wind_affected,
)
from trinity.decorators.platform_specifics import (
    battery_aware,
)
from trinity.decorators.prefabs import (
    extends,
    prefab,
)
from trinity.decorators.procedural import (
    constraint,
    procedural,
    seeded,
)
from trinity.decorators.registry import (
    DecoratorRegistry,
    DecoratorSpec,
    DecoratorValidationError,
    StackSpec,
    Tier,
    registry,
)
from trinity.decorators.rendering import (
    gi_contributor,
    material_blend,
    material_domain,
    reflection_probe,
    render_layer,
    shadow_caster,
)
from trinity.decorators.replay import (
    keyframe,
    recorded,
    replay_authority,
)
from trinity.decorators.rpc import (
    rpc,
)
from trinity.decorators.save_system import (
    atomic_save,
    cloud_sync,
    save_migration,
    save_slot,
)
from trinity.decorators.scheduling import (
    after,
    async_system,
    before,
    chain,
    deferred,
    exclusive,
    fixed,
    job,
    parallel,
    phase,
    run_if,
    throttle,
)
from trinity.decorators.stacks import (
    Stack,
    parameterized_stack,
    stack,
)
from trinity.decorators.security import (
    encrypted,
    rate_limited,
    server_authoritative,
    validated,
)
from trinity.decorators.social import (
    leaderboard,
    presence,
    shareable,
    social,
)
from trinity.decorators.spatial import (
    partitioned,
    spatial,
)
from trinity.decorators.state_machine import (
    on_enter,
    on_exit,
    state_machine,
)
from trinity.decorators.time import (
    deterministic,
    pausable,
    rewindable,
    time_scale,
)
from trinity.decorators.transactions import (
    transactional,
    undoable,
)
from trinity.decorators.ui import (
    layout,
    widget,
)
from trinity.decorators.bridges_caching import (
    async_load,
    batch,
    cached,
    diff,
    lazy,
    observable,
    priority,
    retry,
    throttle_network,
)
from trinity.decorators.world_building import (
    foliage_type,
    level_instance,
    navmesh_modifier,
    procedural_placement,
    trigger_volume,
    water_body,
)
from trinity.decorators import introspection
from trinity.decorators.introspection import (
    all_rules,
    compose,
    composites,
    find_decorators,
    primitives,
    validate_combination,
)
from trinity.decorators.introspection import chain as introspection_chain

__all__ = [
    # Stacks
    "Stack",
    "stack",
    "parameterized_stack",
    # Registry
    "DecoratorRegistry",
    "DecoratorSpec",
    "DecoratorValidationError",
    "Tier",
    "registry",
    "track_decorator",
    "get_applied_decorators",
    "has_decorator",
    "attach_attributes",
    "get_attribute",
    "merge_attributes",
    "validate_target_type",
    "validate_parameters",
    "check_required_decorators",
    "check_excluded_decorators",
    "inspect_decorated",
    "get_decorator_chain",
    "get_current_platform",
    "get_current_arch",
    "PlatformUnavailableError",
    "create_unavailable_stub",
    "NativeConfig",
    "FFIConfig",
    "TargetConfig",
    "BackendConfig",
    "CapabilityConfig",
    "PlatformConfig",
    "native",
    "ffi",
    "target",
    "unsafe",
    "backend",
    "capability",
    "platform",
    "SerializableConfig",
    "NetworkedConfig",
    "SnapshotConfig",
    "VersionedConfig",
    "serializable",
    "networked",
    "snapshot",
    "versioned",
    "PoolConfig",
    "PackedConfig",
    "AlignedConfig",
    "ArenaConfig",
    "BudgetConfig",
    "AllocatorConfig",
    "InlineArrayConfig",
    "pooled",
    "packed",
    "aligned",
    "arena",
    "flyweight",
    "intern",
    "generations",
    "copy_on_write",
    "inline_array",
    "budget",
    "allocator",
    "atomic",
    "phase",
    "parallel",
    "exclusive",
    "after",
    "before",
    "run_if",
    "fixed",
    "job",
    "async_system",
    "throttle",
    "deferred",
    "chain",
    "component",
    "tag",
    "resource",
    "event",
    "system",
    "query",
    "bundle",
    "relation",
    "derived",
    # Tier 5: GPU
    "VALID_BUFFER_USAGE",
    "GpuBufferConfig",
    "GpuKernelConfig",
    "Mat4",
    "RenderPassConfig",
    "ShaderConfig",
    "Vec2",
    "Vec3",
    "Vec4",
    "WgpuBufferAllocation",
    "allocate_wgpu_buffer",
    "gpu_buffer",
    "gpu_kernel",
    "gpu_struct",
    "bind_group",
    "create_wgpu_buffer",
    "dispatch",
    "shader",
    "render_pass",
    "async_compute",
    # Tier 6: Dev
    "ProfileConfig",
    "ReloadableConfig",
    "profile",
    "gpu_profile",
    "trace",
    "reloadable",
    "editor",
    "test",
    "bench",
    "invariant",
    "deprecated",
    # Tier 7: Lifecycle
    "on_add",
    "on_remove",
    "on_change",
    "on_spawn",
    "on_despawn",
    # Tier 8: Assets
    "AssetConfig",
    "CookConfig",
    "ResidencyConfig",
    "ImportSettingsConfig",
    "asset",
    "preload",
    "cook",
    "residency",
    "import_settings",
    # Tier 9: AI/Generation
    "example",
    "constraints",
    "stub",
    "pattern",
    "complexity",
    "generates",
    "pure",
    # Tier 10: Debug/Safety
    "reads",
    "writes",
    "trace_stack",
    # Tier 11: Change Detection
    "track_changes",
    # Tier 12: State Machine
    "state_machine",
    "on_enter",
    "on_exit",
    # Tier 13: Input
    "input_action",
    "input_axis",
    # Tier 14: Audio
    "sound",
    "audio_bus",
    "spatial_audio",
    # Tier 15: UI
    "widget",
    "layout",
    # Tier 16: Spatial
    "spatial",
    "partitioned",
    # Tier 17: Animation
    "tween",
    "blend_tree",
    # Tier 18: Transactions
    "transactional",
    "undoable",
    # Tier 19: RPC
    "rpc",
    # Tier 20: Prefabs
    "prefab",
    "extends",
    # Tier 21: Composition
    "composite",
    "alias",
    # Tier 22: Localization
    "localized",
    "plural",
    "rtl_aware",
    "text_overflow",
    # Tier 23: Accessibility
    "accessible",
    # Tier 24: LOD/Streaming
    "lod",
    "streamable",
    "chunk",
    "loading_priority",
    "unloadable",
    # Tier 25: Time
    "time_scale",
    "pausable",
    "rewindable",
    "deterministic",
    # Tier 26: Replay
    "recorded",
    "replay_authority",
    "keyframe",
    # Tier 27: Debug/Cheat
    "cheat",
    "debug_draw",
    "inspector",
    # Tier 28: Achievements
    "achievement",
    "progress",
    "stat",
    # Tier 29: Analytics
    "telemetry",
    "funnel",
    "heatmap",
    # Tier 30: Modding
    "mod",
    "requires",
    "conflicts",
    "provides",
    "replaces",
    "mod_extends",
    "patch",
    "load_order",
    "moddable",
    # Tier 31: Security
    "server_authoritative",
    "validated",
    "rate_limited",
    "encrypted",
    # Tier 32: Platform
    "battery_aware",
    # Tier 33: Save System
    "save_slot",
    "atomic_save",
    "cloud_sync",
    "save_migration",
    # Tier 34: Narrative
    "dialogue",
    "conversation",
    "voice_over",
    # Tier 35: Cinematics
    "cutscene",
    "camera_track",
    # Tier 36: Game AI
    "behavior_tree",
    "utility_ai",
    "blackboard",
    "ai_debug",
    "perception",
    # Tier 37: Procedural
    "seeded",
    "procedural",
    "constraint",
    # Tier 38: Economy
    "currency",
    "transaction",
    "mtx",
    "daily_reward",
    # Tier 39: Social
    "social",
    "leaderboard",
    "shareable",
    "presence",
    # Tier 40: Error Handling
    "crash_safe",
    "recoverable",
    "error_boundary",
    "bug_report",
    # Tier 41: Build/Deploy
    "build_only",
    "strip_in_release",
    "asset_bundle",
    "feature_flag",
    # Tier 42: Rendering
    "gi_contributor",
    "shadow_caster",
    "reflection_probe",
    "material_domain",
    "material_blend",
    "render_layer",
    # Tier 43: Destruction
    "destructible",
    "damage_type",
    "damage_resistance",
    "fracture",
    "physics_material",
    "joint",
    # Tier 44: IK/Procedural
    "ik_chain",
    "ik_goal",
    "procedural_bone",
    "motion_matching",
    "ragdoll",
    # Tier 45: Particles/VFX
    "particle_emitter",
    "particle_module",
    "vfx_event",
    "gpu_particle",
    "trail",
    "decal",
    # Tier 46: Physics Sim
    "simulation_domain",
    "substep",
    "solver_hint",
    "sleep_threshold",
    "continuous_collision",
    "buoyancy",
    "wind_affected",
    # Tier 47: Gameplay
    "ability",
    "buff",
    "gameplay_tag",
    "spawner",
    "interactable",
    "quest",
    # Tier 48: World Building
    "foliage_type",
    "procedural_placement",
    "level_instance",
    "water_body",
    "navmesh_modifier",
    "trigger_volume",
    # Tier 49: Audio Extended
    "dsp_node",
    "voice_priority",
    "occlusion",
    "reverb_zone",
    "music_stem",
    "music_transition",
    "audio_snapshot",
    "sidechain",
    # Tier 50: Network Extended
    "interest",
    "bandwidth_priority",
    "snapshot_interpolation",
    "server_reconcile",
    # Tier 51: Debug Extended
    "network_debug",
    "automation_test",
    # Tier 52: Crafting
    "crafting_station",
    "recipe",
    "ingredient",
    "loot_table",
    "salvage_recipe",
    # Tier 53: Bridges & Caching
    "cached",
    "lazy",
    "batch",
    "async_load",
    "diff",
    "priority",
    "retry",
    "throttle_network",
    "observable",
    # Registry
    "StackSpec",
    # Built-in Stacks
    "production_component",
    "safe_system",
    "saveable_data",
    "networked_entity",
    "bandwidth_efficient",
    "predicted_entity",
    "secure_multiplayer",
    "versioned_saveable",
    "replay_ready",
    "deterministic_data",
    "streaming_chunk",
    "lod_scalable",
    "complete_ai",
    "profiled_dev",
    # Composite Stacks
    "multiplayer_character",
    "competitive_entity",
    "open_world_entity",
    "mmo_entity",
    "moddable_content",
    # Introspection API
    "introspection",
    "primitives",
    "composites",
    "introspection_chain",
    "find_decorators",
    "compose",
    "validate_combination",
    "all_rules",
]
