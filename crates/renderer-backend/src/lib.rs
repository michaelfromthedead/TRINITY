//! TRINITY Renderer Backend
//!
//! This crate provides the Rust rendering infrastructure:
//! - Frame graph IR and compiler (`frame_graph`)
//! - GPU-driven resource tables (`gpu_driven`)
//! - Demoscene SDF/noise shaders (`demoscene`)
//! - Memory allocators (`memory`)
//! - ECS component storage (`component_store`, `type_registry`)
//! - RHI wgpu abstractions (`rhi_*`, `renderer`, `pipeline`)
//! - Device management (`device::TrinityInstance`, future: adapter, device, queue)
//! - Resource management (`resources::TrinityBuffer`, future: textures, bind groups)
//! - Job system scaffolding (`job_graph`, `scheduler`, `thread_pool`)
//! - Pipeline caching with LRU eviction (`shader_cache`, `pipeline_table`)
//! - Shader preprocessing (`shader::wgsl_preprocessor`)

// Device management (Phase 1 WGPU infrastructure)
pub mod device;

// Resource management (Phase 2 WGPU infrastructure)
pub mod resources;

// Shader module creation (Phase 2 WGPU infrastructure)
pub mod shaders;

// Render pipeline abstraction (Phase 3 WGPU infrastructure - T-WGPU-P3.1)
pub mod render_pipeline;

// Compute pipeline abstraction (Phase 3 WGPU infrastructure - T-WGPU-P3.9.1)
pub mod compute_pipeline;

// Compute pass creation wrapper (Phase 3 WGPU infrastructure - T-WGPU-P3.9.3)
pub mod compute_pass;

// Compute shader library (Phase 3 WGPU infrastructure - T-WGPU-P3.10.1)
pub mod compute_library;

// Command encoding (Phase 4 WGPU infrastructure - T-WGPU-P4.1.1)
pub mod command_encoder;

// Copy commands (Phase 4 WGPU infrastructure - T-WGPU-P4.2.1)
pub mod copy_commands;

// Clear commands (Phase 4 WGPU infrastructure - T-WGPU-P4.3.2)
pub mod clear_commands;

// Timestamp query pool (Phase 4 WGPU infrastructure - T-WGPU-P4.4.1)
pub mod query_pool;

// Occlusion query pool (Phase 4 WGPU infrastructure - T-WGPU-P4.4.4)
pub mod occlusion_query;

// Pipeline statistics query pool (Phase 4 WGPU infrastructure - T-WGPU-P4.4.5)
pub mod pipeline_statistics;

// Debug group RAII utilities (Phase 4 WGPU infrastructure - T-WGPU-P4.5.1)
pub mod debug_utils;

// Resource label utilities (Phase 4 WGPU infrastructure - T-WGPU-P4.5.3)
pub mod resource_labels;

// Debug markers and labels (Phase 7 WGPU infrastructure - T-WGPU-P7.3.1)
pub mod debug;

// Frame synchronization (Phase 4 WGPU infrastructure - T-WGPU-P4.6.1)
pub mod frame_sync;

// Resource state tracking (Phase 4 WGPU infrastructure - T-WGPU-P4.7.1)
pub mod resource_state;

// Buffer mapping (Phase 4 WGPU infrastructure - T-WGPU-P4.8.1)
pub mod buffer_mapping;

// Presentation module - surface creation and swapchain (Phase 7 WGPU infrastructure - T-WGPU-P7.1.1)
pub mod presentation;

// Backend-specific abstractions (Phase 7 WGPU infrastructure - T-WGPU-P7.2.1)
pub mod backend;

// Core modules
pub mod shader;
pub mod frame_graph;
pub mod gpu_driven;
pub mod demoscene;
pub mod demoscene_render;
pub mod demoscene_framegraph;
pub mod sdf_noise;
pub mod sdf_primitives;
pub mod sdf_combinators;
pub mod sdf_domain_ops;
pub mod skeleton;
pub mod skeleton_simd;
pub mod pose;
pub mod pose_blending;
pub mod ik_two_bone;
pub mod ik_fabrik;
pub mod ik_ccd;
pub mod ik_jacobian;
pub mod ik_fullbody;
pub mod ik_goals;
pub mod lookat_controller;
pub mod skinning;
pub mod skinning_orchestrator;
pub mod animation_clip;
pub mod retargeting;
pub mod root_motion;
pub mod clip_compression;
pub mod clip_player;
pub mod playback_backend;
pub mod animation_events;
pub mod animation_graph;
pub mod motion_matching_db;
pub mod motion_features;
pub mod motion_search;
pub mod inertialization;
pub mod motion_context;
pub mod spring_bone;
pub mod eye_animation;
pub mod state_machine;
pub mod state_machine_system;
pub mod animation_layers;
pub mod blend_tree;
pub mod blend_node;
pub mod sync_groups;
pub mod animation_textures;
pub mod crowd_renderer;
pub mod crowd_shaders;
pub mod impostor_system;
pub mod crowd_lod;
pub mod twist_distribution;
pub mod blend_shapes;
pub mod facs_action_units;
pub mod lip_sync;
pub mod virtual_geometry;
pub mod terrain;
pub mod water;
pub mod shoreline;
pub mod foliage_instancing;
pub mod foliage_wind;

// Memory and ECS
pub mod memory;
pub mod component_store;
pub mod type_registry;
pub mod entity;

// Rendering
pub mod renderer;
pub mod pipeline;
pub mod aerial_perspective;
pub mod cloud_lighting;
pub mod cloud_noise;
pub mod cloud_raymarching;
pub mod cloud_shadows;
pub mod cloud_temporal;
pub mod god_rays;
pub mod weather_map;
pub mod sky_rendering;
pub mod celestial_bodies;
pub mod froxel;
pub mod layered_fog;
pub mod volumetric_fog;
pub mod froxel_composite;
pub mod froxel_lighting;
pub mod rt_capability;
pub mod shadow_fallback;
pub mod light_types;
pub mod light_buffers;
pub mod light_bindings;
pub mod ltc_lut;
pub mod lut_cooking;
pub mod staging;
pub mod lighting_pass;
pub mod lighting_frame_graph;
pub mod culling;
pub mod csm;
pub mod cube_shadow;
pub mod shadow_atlas;
pub mod shadow_request;
pub mod shadow_tile_buffer;
pub mod shadow_dispatch;
pub mod shadow_modulation;
pub mod contact_shadow;
pub mod shadow_flags;
pub mod gc;
pub mod shader_cache;
pub mod pipeline_table;
pub mod post_process;
pub mod particles;
pub mod decals;
pub mod ddgi;
pub mod sh;
pub mod gi;
pub mod gi_light_handoff;
pub mod reflection_buffer;
pub mod headless;
pub mod executor;
pub mod hiz;
pub mod ssr;
pub mod ssgi;
pub mod ssr_linear;
pub mod planar_mirror;
pub mod ray_budget;
pub mod raytracing;
pub mod env_performance;
pub mod mobile_fallback;

// RHI (wgpu abstractions)
pub mod rhi_device;
pub mod rhi_resources;
pub mod rhi_pipeline;
pub mod rhi_commands;
pub mod rhi_swapchain;
pub mod rhi_bind_group;

// Asset pipeline
pub mod asset;

// Infrastructure
pub mod checksum;
pub mod command_buffer;
pub mod material_dep_graph;
pub mod hot_reload;
pub mod param_hot_reload;
pub mod asset_loader;
pub mod gltf;
pub mod lod;
pub mod budget;
pub mod meshlet;
pub mod texture_import;
pub mod cubemap;
pub mod blas;
pub mod blas_pool;
pub mod dynamic_blas;
pub mod mesh_rt_integration;
pub mod tlas;
pub mod scratch_buffer;
pub mod instance_buffer;
pub mod cache_db;
pub mod remote_cache;
pub mod incremental_build;
pub mod priority_queue;
pub mod virtual_texture;
pub mod virtual_texture_atlas;
pub mod virtual_texture_page_table;
pub mod virtual_texture_feedback;
pub mod virtual_texture_streaming;
pub mod preload;
pub mod streaming_heuristics;

// Job system (scaffolding)
pub mod job_graph;
pub mod scheduler;
pub mod thread_pool;
pub mod system_phase;

// Asset streaming (T-AS-5.1)
pub mod streaming;

// GPU profiling (T-TL-1.4)
pub mod gpu_profiler;

// GPU timestamp profiling (T-WGPU-P7.4.1)
pub mod profiling;

// Editor integration
pub mod editor;
pub mod editor_camera;
pub mod hierarchy_panel;
pub mod inspector_panel;
pub mod inspector_diff;
pub mod moldable_views;
pub mod repl_panel;
pub mod selection_state;
pub mod foundation;

// Bridge modules
/// Three-channel bridge protocol for Python-Rust communication (T-TL-1.1).
/// Always available regardless of pyo3 feature.
pub mod bridge_protocol;

/// Bridge endpoint handlers for Python-Rust communication (T-TL-1.2).
/// Implements handlers for all protocol namespaces.
pub mod bridge_handlers;

/// Bridge health monitoring and error handling (T-TL-1.5).
/// Provides timeout tracking, connection health monitoring, and reconnection logic.
pub mod bridge_health;

/// EguiUIContext adapter for Python-Rust UI bridge (T-TL-1.3).
/// Provides UIContext trait mapping to egui::Ui.
pub mod egui_adapter;

/// EguiUIContext input handling for Python-Rust UI bridge (T-TL-1.6).
/// Provides input event mapping from Python to egui's input system.
pub mod egui_input;

/// FlowForge visual node-graph scripting system (T-TL-8.1).
/// Provides node graph editor for visual programming.
pub mod flowforge;

/// FlowForge node executors for runtime execution (T-TL-8.2).
/// Provides 40+ built-in node types across 7 categories.
pub mod flowforge_nodes;

/// FlowForge Python bytecode compiler (T-TL-8.3).
/// Compiles node graphs to executable Python code.
pub mod flowforge_compiler;

/// FlowForge sub-graph macros system (T-TL-8.4).
/// Provides reusable sub-graphs as macro nodes.
pub mod flowforge_macros;

/// AnimationConfig resource for global animation system configuration (T-AN-1.7).
/// Provides configuration for IK chains, motion matching budgets, LOD distances, and skeleton constraints.
pub mod animation_config;

/// Animation asset registration and metadata (T-AN-1.5).
/// Provides asset system integration for skeletons, animation clips, and motion databases.
pub mod animation_assets;

/// Crowd renderer bridge (requires pyo3 for runtime, but tests work without).
#[cfg(any(feature = "pyo3", test))]
pub mod bridge;

/// Python bindings for wgpu types (T-WGPU-P7.6.3).
/// Provides PyBufferDescriptor, PyBufferUsage, PyBufferBindingType, and PyBufferSize.
#[cfg(feature = "pyo3")]
pub mod bindings;

/// Integration tests for animation foundation (T-AN-1.8).
/// Tests skeleton hierarchy, pose blending, clip sampling, and SIMD operations.
#[cfg(test)]
mod animation_tests;

/// Integration tests for IK solvers (T-AN-4.7).
/// Tests two-bone, FABRIK, CCD, Jacobian, and full-body IK.
#[cfg(test)]
mod ik_tests;

/// Integration tests for animation playback system (T-AN-2.7).
/// Tests end-to-end playback, blending, root motion, compression, and error handling.
#[cfg(test)]
mod playback_tests;

/// Integration tests for procedural animation systems (T-AN-7.8).
/// Tests eye animation, spring bones, look-at controllers, and twist distribution.
#[cfg(test)]
mod procedural_tests;

/// Integration tests for motion matching system (T-AN-6.6).
/// Tests end-to-end pipeline, context-driven queries, inertialization,
/// feature matching, performance, and edge cases.
#[cfg(test)]
mod motion_matching_tests;

/// Integration tests for animation graph system (T-AN-5.8).
/// Tests graph construction, evaluation pipeline, parameters, state machines, and performance.
#[cfg(test)]
mod animation_graph_tests;

/// Integration tests for crowd rendering system (T-AN-8.6).
/// Tests full rendering pipeline, LOD system, impostors, instancing, performance, and edge cases.
#[cfg(test)]
mod crowd_tests;
