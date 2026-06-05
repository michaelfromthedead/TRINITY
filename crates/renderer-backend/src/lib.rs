//! TRINITY Renderer Backend
//!
//! This crate provides the Rust rendering infrastructure:
//! - Frame graph IR and compiler (`frame_graph`)
//! - GPU-driven resource tables (`gpu_driven`)
//! - Demoscene SDF/noise shaders (`demoscene`)
//! - Memory allocators (`memory`)
//! - ECS component storage (`component_store`, `type_registry`)
//! - RHI wgpu abstractions (`rhi_*`, `renderer`, `pipeline`)
//! - Job system scaffolding (`job_graph`, `scheduler`, `thread_pool`)

// Core modules
pub mod frame_graph;
pub mod gpu_driven;
pub mod demoscene;

// Memory and ECS
pub mod memory;
pub mod component_store;
pub mod type_registry;
pub mod entity;

// Rendering
pub mod renderer;
pub mod pipeline;
pub mod post_process;
pub mod particles;
pub mod ddgi;

// Compute
pub mod compute_pipeline;
pub mod compute_library;

// RHI (wgpu abstractions)
pub mod rhi_device;
pub mod rhi_resources;
pub mod rhi_pipeline;
pub mod rhi_commands;
pub mod rhi_swapchain;
pub mod rhi_bind_group;

// Infrastructure
pub mod checksum;
pub mod command_buffer;
pub mod material_dep_graph;
pub mod asset_loader;

// Job system (scaffolding)
pub mod job_graph;
pub mod scheduler;
pub mod thread_pool;
pub mod system_phase;

// Editor integration
pub mod editor;

// Profiling and diagnostics
pub mod profiling;
pub mod presentation;

// Debug utilities
pub mod debug;
pub mod debug_utils;
pub mod resource_state;
pub mod compute_pass;
pub mod buffer_mapping;

// Phase 1 exports (modules with inline tests, no external deps)
pub mod terrain;
pub mod virtual_geometry;
pub mod water;
pub mod decals;
pub mod skinning;
pub mod device;
pub mod resources;
pub mod render_pipeline;

// Additional modules (previously internal)
pub mod shaders;
pub mod query_pool;
pub mod frame_sync;
pub mod backend;
pub mod texture_import;
pub mod resource_labels;
pub mod pipeline_statistics;
pub mod occlusion_query;
pub mod gltf;
pub mod culling;
pub mod light_bindings;
pub mod blend_tree;
pub mod copy_commands;

// Animation: skeleton, pose, and IK solvers (T-IK-3.x series)
pub mod skeleton;
pub mod pose;
pub mod ik_goals;
pub mod ik_two_bone;
pub mod ik_ccd;
pub mod ik_fabrik;
pub mod ik_jacobian;
pub mod ik_fullbody;
#[cfg(test)]
mod ik_tests;

// Bridge module requires pyo3 - not built by default
#[cfg(feature = "pyo3")]
pub mod bridge;
