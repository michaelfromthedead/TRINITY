// PBR Struct Definitions for TRINITY Material System
// T-MAT-3.1: Foundational BRDF structs
//
// These structs define the data flow for PBR materials:
//   PBRInput  -> surface() function -> PBRParams -> BRDF evaluation -> PBROutput

/// PBR Input - data from vertex shader and scene uniforms.
/// Passed to the surface() function as read-only context.
struct PBRInput {
    /// World-space position of the fragment
    world_position: vec3<f32>,
    /// World-space normal (interpolated, may not be normalized)
    world_normal: vec3<f32>,
    /// World-space tangent with handedness in w component
    /// Used for tangent-space normal mapping
    world_tangent: vec4<f32>,
    /// Normalized view direction (fragment to camera)
    world_view: vec3<f32>,
    /// Primary UV coordinates
    uv: vec2<f32>,
    /// Vertex color (linear, premultiplied alpha)
    vertex_color: vec4<f32>,
    /// Time in seconds since scene start
    time: f32,
    /// Number of active lights affecting this fragment
    light_count: u32,
}

/// PBR Parameters - material properties output by surface().
/// These define the appearance of the surface at a point.
struct PBRParams {
    /// Base diffuse/albedo color (linear RGB)
    base_color: vec3<f32>,
    /// Tangent-space normal perturbation
    /// Default: vec3(0.0, 0.0, 1.0) = no perturbation
    normal: vec3<f32>,
    /// Roughness: 0.0 = mirror, 1.0 = fully diffuse
    roughness: f32,
    /// Metallic: 0.0 = dielectric, 1.0 = metal
    metallic: f32,
    /// Specular reflectance for dielectrics at normal incidence
    /// Default: 0.5 = 4% reflectance (typical for plastics)
    specular: f32,
    /// Ambient occlusion: 0.0 = fully occluded, 1.0 = no occlusion
    occlusion: f32,
    /// Emissive color (linear RGB, can exceed 1.0 for HDR bloom)
    emissive: vec3<f32>,
    /// Alpha/opacity: 0.0 = transparent, 1.0 = opaque
    alpha: f32,
    /// Subsurface scattering intensity
    /// 0.0 = no SSS, 1.0 = full SSS (for skin, wax, etc.)
    subsurface: f32,
    /// Anisotropic roughness factor
    /// -1.0 to 1.0, affects specular highlights along tangent
    anisotropy: f32,
    /// Clearcoat layer intensity
    /// 0.0 = no clearcoat, 1.0 = full clearcoat (for car paint, etc.)
    clearcoat: f32,
    /// Clearcoat layer roughness
    /// Typically low (0.0-0.3) for glossy clearcoat effects
    clearcoat_roughness: f32,
}

/// PBR Output - final fragment shader output.
/// Produced by BRDF evaluation.
struct PBROutput {
    /// Final fragment color (linear RGBA, pre-multiplied alpha)
    color: vec4<f32>,
    // Future: G-buffer outputs for deferred rendering
    // gbuffer_normal: vec4<f32>,
    // gbuffer_material: vec4<f32>,
}

/// Create default PBR parameters.
/// This produces a white, non-metallic, medium-rough surface.
fn pbr_params_default() -> PBRParams {
    var params: PBRParams;
    params.base_color = vec3<f32>(1.0, 1.0, 1.0);
    params.normal = vec3<f32>(0.0, 0.0, 1.0);
    params.roughness = 0.5;
    params.metallic = 0.0;
    params.specular = 0.5;
    params.occlusion = 1.0;
    params.emissive = vec3<f32>(0.0, 0.0, 0.0);
    params.alpha = 1.0;
    params.subsurface = 0.0;
    params.anisotropy = 0.0;
    params.clearcoat = 0.0;
    params.clearcoat_roughness = 0.0;
    return params;
}
