//! Ray tracing infrastructure for TRINITY renderer (GAPSET_9_RAY_TRACING).
//!
//! This module provides hardware-accelerated ray tracing capabilities using
//! inline ray queries. It builds on the existing BLAS infrastructure to
//! provide shadow rays, ambient occlusion, and reflections.
//!
//! # Submodules
//!
//! - [`rt_shadow`]: Shadow ray compute pipeline using ray queries
//! - [`alpha_test`]: Alpha testing support for transparent geometry
//!
//! # Architecture
//!
//! ```text
//! +----------------+     +----------------+     +----------------+
//! |     BLAS       |---->|     TLAS       |---->|   Ray Query    |
//! | (per-mesh AS)  |     | (scene AS)     |     | (inline trace) |
//! +----------------+     +----------------+     +----------------+
//!        ^                      ^                      |
//!        |                      |                      v
//! +----------------+     +----------------+     +----------------+
//! |   Meshlets     |     |  Instances     |     | Shadow Output  |
//! | (geometry)     |     | (transforms)   |     | (per-pixel)    |
//! +----------------+     +----------------+     +----------------+
//! ```
//!
//! # Feature Requirements
//!
//! Ray tracing requires hardware support:
//! - Vulkan: VK_KHR_ray_query
//! - DirectX 12: DXR 1.1 (inline ray tracing)
//! - Metal: Pending (Apple Silicon)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::raytracing::{RTShadowPipeline, ShadowRayParams};
//!
//! // Create pipeline
//! let pipeline = RTShadowPipeline::new(&device);
//!
//! // Set up parameters
//! let params = ShadowRayParams {
//!     inverse_view_proj: camera.inverse_view_proj(),
//!     light_count: lights.len() as u32,
//!     width: 1920,
//!     height: 1080,
//!     bias: 0.001,
//! };
//!
//! // Create bind group and dispatch
//! let bind_group = pipeline.create_bind_group(...);
//! pipeline.dispatch(&mut encoder, &bind_group, width, height);
//! ```

pub mod alpha_test;
pub mod denoiser;
pub mod rt_shadow;

// Re-export commonly used types
pub use alpha_test::{
    AlphaTestParams,
    AlphaTestPipeline,
    DEFAULT_ALPHA_CUTOFF,
};
pub use denoiser::{
    DenoiseParams,
    DenoiserPipeline,
    PingPongBuffers,
    DEFAULT_SIGMA_COLOR,
    DEFAULT_SIGMA_DEPTH,
    DEFAULT_SIGMA_NORMAL,
    MAX_ITERATIONS,
};
pub use rt_shadow::{
    Light,
    LightType,
    RTShadowPipeline,
    ShadowRayParams,
    WORKGROUP_SIZE,
};
