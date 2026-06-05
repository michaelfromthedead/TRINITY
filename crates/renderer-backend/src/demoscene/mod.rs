//! Demoscene WGSL shaders for SDF and noise operations.
//!
//! These shaders are embedded as strings for compile-time validation.

pub mod bootstrap;
pub mod depth_barriers;
pub mod hybrid_depth;
pub mod minimal;
pub mod multipass;
pub mod post_integration;

// Re-exports from hybrid_depth
pub use hybrid_depth::{
    HybridDepthConfig, HybridDepthRenderer, HybridUniforms, DepthBufferBinding,
    DepthCompareResult, HYBRID_DEPTH_SHADER, DEPTH_BUFFER_FORMAT, DEFAULT_NEAR_PLANE,
    DEFAULT_FAR_PLANE, MAX_RAY_MARCH_DIST, DEPTH_EPSILON, ndc_to_linear, linear_to_ndc,
    reverse_z_to_linear, linear_to_reverse_z,
};

// Re-exports from multipass
pub use multipass::{
    MultiPassSdfRenderer, MultiPassUniforms, SdfBlendMode, SdfDepthMode, SdfPassConfig,
    SdfPassType, RenderPassOrder, MULTIPASS_OPAQUE_SHADER, MULTIPASS_TRANSPARENT_SHADER,
    MULTIPASS_WORKGROUP_SIZE, MAX_TRANSPARENT_OBJECTS, OPAQUE_ALPHA_THRESHOLD,
    TRANSPARENT_ALPHA_MIN,
};

// Re-exports from post_integration
pub use post_integration::{
    SdfPostProcessConfig, SdfPostProcessor, PostProcessUniforms, create_sdf_post_process_chain,
    create_sdf_bloom_bright_pass, create_bloom_blur_pass, create_sdf_motion_vector_pass,
    SDF_BLOOM_THRESHOLD, SDF_BLOOM_INTENSITY, SDF_TAA_FEEDBACK, SDF_MOTION_SCALE,
    TONEMAP_SHADER, BLOOM_BRIGHT_SHADER, BLOOM_BLUR_SHADER, TAA_SHADER,
};

// Re-exports from minimal
pub use minimal::{MinimalRenderer, MinimalUniforms, MINIMAL_SHADER};

// Re-exports from depth_barriers
pub use depth_barriers::{
    DepthProjection, DemoResourceState, DemoResourceTransition, DemoBarrierScheduler,
    DemoFrameBarriers, resource_ids,
};

pub const NOISE_HASH: &str = include_str!("noise_hash.wgsl");
pub const NOISE_VALUE: &str = include_str!("noise_value.wgsl");
pub const NOISE_PERLIN: &str = include_str!("noise_perlin.wgsl");
pub const NOISE_FBM: &str = include_str!("noise_fbm.wgsl");
pub const NOISE_RIDGED: &str = include_str!("noise_ridged.wgsl");
pub const NOISE_DOMAIN_WARP: &str = include_str!("noise_domain_warp.wgsl");
pub const SDF_DOMAIN: &str = include_str!("sdf_domain.wgsl");
