//! Demoscene WGSL shaders for SDF and noise operations.
//!
//! These shaders are embedded as strings for compile-time validation.
//!
//! # Compiled Shaders (T-DEMO-5.5)
//!
//! The `DEMO_SCENE_COMPILED` shader is generated at build time from Python
//! scene definitions. See `build.rs` for the compilation pipeline:
//!
//! ```text
//! scenes/demo.py -> scripts/compile_demo.py -> target/generated/demo.wgsl
//! ```
//!
//! The shader is embedded via `include_str!` and contains:
//! - Scene SDF function (`scene_sdf`)
//! - Material lookup (`scene_material`)
//! - PBR lighting
//! - Ray marching compute shader
//!
//! # 4K Mode (T-DEMO-5.7)
//!
//! The `minimal` module provides extreme minimization for 4K demoscene intros:
//! - Single-file structure
//! - Inline shader as literal string
//! - No runtime file I/O
//! - Minimal dependencies
//!
//! ```rust,ignore
//! use renderer_backend::demoscene::minimal::MinimalRenderer;
//!
//! let renderer = MinimalRenderer::new(&device, 800, 600);
//! renderer.render(&device, &queue, elapsed_time);
//! ```
//!
//! # Multi-Pass Rendering (T-DEMO-6.7)
//!
//! The `multipass` module implements opaque and transparent SDF passes:
//! - Opaque pass: writes depth, alpha test
//! - Transparent pass: reads depth, alpha blending
//! - Correct ordering: raster opaque -> SDF opaque -> SDF transparent -> raster transparent
//!
//! ```rust,ignore
//! use renderer_backend::demoscene::multipass::{MultiPassSdfRenderer, RenderPassOrder};
//!
//! let renderer = MultiPassSdfRenderer::new(&device, 1920, 1080);
//! renderer.update(&queue, elapsed_time);
//! renderer.dispatch_all(&mut encoder);
//! ```
//!
//! # Post-Processing Integration (T-DEMO-6.8)
//!
//! The `post_integration` module connects SDF output to the S8 post-processing
//! pipeline: tone mapping (ACES), bloom extraction, and TAA.
//!
//! ```rust,ignore
//! use renderer_backend::demoscene::post_integration::{
//!     SdfPostProcessConfig, create_sdf_post_process_chain
//! };
//!
//! let config = SdfPostProcessConfig::full(1920, 1080);
//! let (passes, resources) = create_sdf_post_process_chain(
//!     PassIndex(0), hdr_input, ldr_output, &config
//! );
//! ```

pub mod bootstrap;
pub mod depth_barriers;
pub mod hybrid_depth;
pub mod minimal;
pub mod multipass;
pub mod post_integration;

pub use bootstrap::{BootstrapError, DemoBootstrap, DemoWindow};
pub use depth_barriers::{
    DepthProjection, DepthReconstructionResult, DemoResourceState,
    DemoResourceTransition, DemoBarrierScheduler, DemoFrameBarriers,
    resource_ids, DEPTH_RECONSTRUCTION_WGSL,
};
pub use hybrid_depth::{
    HybridDepthConfig, HybridDepthRenderer, HybridUniforms, DepthBufferBinding,
    DepthCompareResult, HYBRID_DEPTH_SHADER, DEPTH_BUFFER_FORMAT,
    DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE, MAX_RAY_MARCH_DIST, DEPTH_EPSILON,
    ndc_to_linear, linear_to_ndc, reverse_z_to_linear, linear_to_reverse_z,
};
pub use minimal::{MinimalRenderer, MinimalUniforms, MINIMAL_SHADER};
pub use multipass::{
    MultiPassSdfRenderer, MultiPassUniforms, SdfBlendMode, SdfDepthMode,
    SdfPassConfig, SdfPassType, RenderPassOrder,
    MULTIPASS_OPAQUE_SHADER, MULTIPASS_TRANSPARENT_SHADER,
    MULTIPASS_WORKGROUP_SIZE, MAX_TRANSPARENT_OBJECTS,
    OPAQUE_ALPHA_THRESHOLD, TRANSPARENT_ALPHA_MIN,
};
pub use post_integration::{
    SdfPostProcessConfig, SdfPostProcessor, PostProcessUniforms,
    create_sdf_post_process_chain, create_sdf_bloom_bright_pass,
    create_bloom_blur_pass, create_sdf_motion_vector_pass,
    SDF_BLOOM_THRESHOLD, SDF_BLOOM_INTENSITY, SDF_TAA_FEEDBACK, SDF_MOTION_SCALE,
    TONEMAP_SHADER, BLOOM_BRIGHT_SHADER, BLOOM_BLUR_SHADER, TAA_SHADER,
};

// Static noise shaders
pub const NOISE_HASH: &str = include_str!("noise_hash.wgsl");
pub const NOISE_VALUE: &str = include_str!("noise_value.wgsl");
pub const NOISE_PERLIN: &str = include_str!("noise_perlin.wgsl");
pub const NOISE_FBM: &str = include_str!("noise_fbm.wgsl");
pub const NOISE_RIDGED: &str = include_str!("noise_ridged.wgsl");
pub const NOISE_DOMAIN_WARP: &str = include_str!("noise_domain_warp.wgsl");

// Static SDF shaders
pub const SDF_DOMAIN: &str = include_str!("sdf_domain.wgsl");
pub const SDF_PRIMITIVES: &str = include_str!("sdf_primitives.wgsl");
pub const SDF_COMBINATORS: &str = include_str!("sdf_combinators.wgsl");

// Static demo shader (checked in)
pub const DEMO_SCENE_STATIC: &str = include_str!("demo.wgsl");

/// Compiled demoscene shader (T-DEMO-5.5).
///
/// This shader is generated at build time from `scenes/demo.py` via the
/// Python DSL compiler. If compilation fails, a fallback shader is embedded
/// that displays a magenta checkerboard pattern.
///
/// # Usage
///
/// ```rust,ignore
/// use renderer_backend::demoscene::DEMO_SCENE_COMPILED;
///
/// let shader_source = DEMO_SCENE_COMPILED;
/// // Create shader module with wgpu...
/// ```
///
/// # Build Dependencies
///
/// The shader is recompiled when:
/// - `scenes/demo.py` changes
/// - `scripts/compile_demo.py` changes
/// - Any file in `engine/rendering/demoscene/` changes
#[cfg(feature = "build-compiled-shaders")]
pub const DEMO_SCENE_COMPILED: &str = include_str!(concat!(env!("OUT_DIR"), "/generated/demo.wgsl"));

/// Get the compiled demo scene shader, falling back to static if not available.
///
/// This function returns the build-time compiled shader if available,
/// otherwise falls back to the static demo.wgsl checked into the repo.
pub fn get_demo_scene_shader() -> &'static str {
    #[cfg(feature = "build-compiled-shaders")]
    {
        DEMO_SCENE_COMPILED
    }
    #[cfg(not(feature = "build-compiled-shaders"))]
    {
        DEMO_SCENE_STATIC
    }
}

/// Validate that a WGSL shader contains required demoscene components.
///
/// Checks for:
/// - `@compute` entry point
/// - `fn main(` function
/// - `fn scene_sdf(` function
/// - `fn scene_material(` function
/// - Balanced braces and parentheses
///
/// # Returns
///
/// `Ok(())` if valid, `Err(message)` describing the validation failure.
pub fn validate_demoscene_shader(wgsl: &str) -> Result<(), String> {
    let mut errors = Vec::new();

    // Check for compute entry point
    if !wgsl.contains("@compute") {
        errors.push("Missing @compute entry point");
    }

    if !wgsl.contains("fn main(") {
        errors.push("Missing main() function");
    }

    // Check for scene_sdf function
    if !wgsl.contains("fn scene_sdf(") {
        errors.push("Missing scene_sdf() function");
    }

    // Check for scene_material function
    if !wgsl.contains("fn scene_material(") {
        errors.push("Missing scene_material() function");
    }

    // Check balanced braces
    let open_braces = wgsl.matches('{').count();
    let close_braces = wgsl.matches('}').count();
    if open_braces != close_braces {
        errors.push("Unbalanced braces");
    }

    // Check balanced parentheses
    let open_parens = wgsl.matches('(').count();
    let close_parens = wgsl.matches(')').count();
    if open_parens != close_parens {
        errors.push("Unbalanced parentheses");
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_demo_shader_loads() {
        assert!(!DEMO_SCENE_STATIC.is_empty());
    }

    #[test]
    fn test_static_demo_shader_has_entry_point() {
        assert!(DEMO_SCENE_STATIC.contains("@compute"));
        assert!(DEMO_SCENE_STATIC.contains("fn main("));
    }

    #[test]
    fn test_static_demo_shader_has_scene_sdf() {
        assert!(DEMO_SCENE_STATIC.contains("fn scene_sdf(")
            || DEMO_SCENE_STATIC.contains("fn sd_scene"));
    }

    #[test]
    fn test_get_demo_scene_shader_returns_valid() {
        let shader = get_demo_scene_shader();
        assert!(!shader.is_empty());
    }

    #[test]
    fn test_validate_valid_shader() {
        // Minimal valid shader
        let valid = r#"
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        "#;
        assert!(validate_demoscene_shader(valid).is_ok());
    }

    #[test]
    fn test_validate_missing_compute() {
        let invalid = r#"
            fn main() {}
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        "#;
        let result = validate_demoscene_shader(invalid);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("@compute"));
    }

    #[test]
    fn test_validate_missing_scene_sdf() {
        let invalid = r#"
            @compute @workgroup_size(8, 8, 1)
            fn main() {}
            fn scene_material(id: u32) -> Material { return Material(); }
        "#;
        let result = validate_demoscene_shader(invalid);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("scene_sdf"));
    }

    #[test]
    fn test_validate_unbalanced_braces() {
        let invalid = r#"
            @compute @workgroup_size(8, 8, 1)
            fn main() {
            fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
            fn scene_material(id: u32) -> Material { return Material(); }
        "#;
        let result = validate_demoscene_shader(invalid);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("braces"));
    }
}
