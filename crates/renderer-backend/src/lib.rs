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

// Bridge module requires pyo3 - not built by default
#[cfg(feature = "pyo3")]
pub mod bridge;
