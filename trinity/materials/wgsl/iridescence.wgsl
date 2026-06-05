// Iridescence Functions for TRINITY Material System
// T-MAT-4.6: Thin-Film Iridescence Implementation
//
// This module implements thin-film interference for iridescent materials:
//   - Air-film Fresnel reflection at the air-coating interface
//   - Film-substrate Fresnel reflection at the coating-base interface
//   - Phase shift computation from film thickness and viewing angle
//   - Wavelength-dependent interference color computation
//   - Final iridescence application to base F0
//
// Physical model based on thin-film interference theory:
//   - Light reflects at both interfaces of the thin film
//   - Interference between reflected rays depends on path difference
//   - Path difference = 2 * n * d * cos(theta_film)
//   - Phase shift = 2 * PI * path_difference / wavelength
//
// References:
//   - Belcour, Barla "A Practical Extension to Microfacet Theory for the
//     Modeling of Varying Iridescence"
//   - Kneiphof et al. "Real-time Rendering of Layered Materials with
//     Anisotropic Normal Distributions"
//   - glTF KHR_materials_iridescence extension specification

// Mathematical constants (should match brdf.wgsl)
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;

// Representative wavelengths for RGB channels (in nanometers)
// Using sRGB primaries' peak wavelengths
const WAVELENGTH_R: f32 = 650.0;  // Red
const WAVELENGTH_G: f32 = 532.0;  // Green
const WAVELENGTH_B: f32 = 450.0;  // Blue

// Quality tier const for enabling/disabling iridescence
// This allows dead-code elimination by naga at compile time
// Set to true for HIGH quality, false for LOW/MEDIUM
const QUALITY_IRIDESCENCE_ENABLED: bool = true;

// ============================================================================
// Iridescence Parameter Struct
// ============================================================================

/// Parameters for thin-film iridescence.
/// These control the appearance of the interference effect.
struct IridescenceParams {
    /// Intensity of the iridescence effect [0,1].
    /// 0 = no iridescence, 1 = full iridescence effect.
    intensity: f32,
    /// Index of refraction of the thin film [1.3-2.0].
    /// Typical values: 1.3 (oil), 1.5 (glass), 1.8 (titanium oxide).
    ior: f32,
    /// Thickness of the thin film in nanometers [100-1000].
    /// Controls the interference pattern spacing.
    /// Lower values = wider color bands, higher values = tighter bands.
    thickness_nm: f32,
}

// ============================================================================
// Fresnel Functions for Interfaces
// ============================================================================

/// Compute refracted angle using Snell's law.
/// @param cos_theta_i: Cosine of incident angle
/// @param eta: Ratio n1/n2 (incident IOR / transmitted IOR)
/// @returns: Cosine of transmitted angle, or negative for TIR
fn snell_cos_theta_t(cos_theta_i: f32, eta: f32) -> f32 {
    let sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_t_sq = eta * eta * sin_theta_i_sq;

    // Check for total internal reflection
    if sin_theta_t_sq >= 1.0 {
        return -1.0;  // TIR indicator
    }

    return sqrt(1.0 - sin_theta_t_sq);
}

/// Fresnel reflectance for dielectric interface (unpolarized light).
/// Uses exact Fresnel equations (not Schlick approximation) for accuracy.
///
/// @param cos_theta_i: Cosine of incident angle
/// @param cos_theta_t: Cosine of transmitted angle
/// @param eta: Ratio n1/n2 (incident IOR / transmitted IOR)
/// @returns: Fresnel reflectance [0,1]
fn fresnel_dielectric(cos_theta_i: f32, cos_theta_t: f32, eta: f32) -> f32 {
    // Handle total internal reflection
    if cos_theta_t < 0.0 {
        return 1.0;
    }

    // Fresnel equations for s and p polarization
    let r_s = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t + EPSILON);
    let r_p = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t + EPSILON);

    // Average of s and p for unpolarized light
    return 0.5 * (r_s * r_s + r_p * r_p);
}

/// Fresnel reflectance at air-film interface.
/// Air (n=1.0) to thin-film (n=ior).
///
/// @param cos_theta: Cosine of viewing angle (dot(V,H) or dot(V,N))
/// @param film_ior: IOR of the thin film coating
/// @returns: Fresnel reflectance at air-film interface
fn fresnel_air_film(cos_theta: f32, film_ior: f32) -> f32 {
    let eta = 1.0 / film_ior;  // Air to film ratio
    let cos_theta_t = snell_cos_theta_t(cos_theta, eta);
    return fresnel_dielectric(cos_theta, cos_theta_t, eta);
}

/// Fresnel reflectance at film-substrate interface.
/// Thin-film (n=film_ior) to substrate (n=substrate_ior).
/// For metals, substrate_ior represents the effective IOR.
///
/// @param cos_theta_film: Cosine of angle inside the film
/// @param film_ior: IOR of the thin film
/// @param substrate_ior: IOR of the substrate (base material)
/// @returns: Fresnel reflectance at film-substrate interface
fn fresnel_film_substrate(cos_theta_film: f32, film_ior: f32, substrate_ior: f32) -> f32 {
    let eta = film_ior / substrate_ior;
    let cos_theta_substrate = snell_cos_theta_t(cos_theta_film, eta);
    return fresnel_dielectric(cos_theta_film, cos_theta_substrate, eta);
}

// ============================================================================
// Phase Computation
// ============================================================================

/// Compute the optical path difference phase shift in radians.
/// The phase determines where constructive/destructive interference occurs.
///
/// OPD = 2 * n * d * cos(theta_film)
/// Phase = 2 * PI * OPD / wavelength
///
/// @param thickness_nm: Film thickness in nanometers
/// @param cos_theta_film: Cosine of angle inside the film
/// @param film_ior: Index of refraction of the film
/// @param wavelength_nm: Wavelength of light in nanometers
/// @returns: Phase shift in radians
fn compute_film_phase(
    thickness_nm: f32,
    cos_theta_film: f32,
    film_ior: f32,
    wavelength_nm: f32
) -> f32 {
    // Optical path difference = 2 * n * d * cos(theta_film)
    let opd = 2.0 * film_ior * thickness_nm * cos_theta_film;
    // Convert to phase shift
    return 2.0 * PI * opd / wavelength_nm;
}

// ============================================================================
// Interference Computation
// ============================================================================

/// Compute interference factor for a single wavelength.
/// Combines reflections from both interfaces with phase-dependent interference.
///
/// Using the thin-film interference formula:
/// R = R1 + R2 + 2*sqrt(R1*R2)*cos(phase + phase_offset)
///
/// where R1, R2 are the Fresnel reflectances at each interface.
///
/// @param R_air_film: Fresnel reflectance at air-film interface
/// @param R_film_sub: Fresnel reflectance at film-substrate interface
/// @param phase: Phase shift from optical path difference
/// @returns: Total interference reflectance [0,1]
fn compute_interference(R_air_film: f32, R_film_sub: f32, phase: f32) -> f32 {
    // Phase shift of PI at one interface (180 degree reflection phase shift
    // when going from lower to higher IOR)
    let phase_total = phase + PI;

    // Interference term
    let interference = 2.0 * sqrt(R_air_film * R_film_sub) * cos(phase_total);

    // Total reflectance with interference
    let R_total = R_air_film + R_film_sub * (1.0 - R_air_film) * (1.0 - R_air_film) + interference;

    return clamp(R_total, 0.0, 1.0);
}

/// Compute interference colors for RGB channels.
/// Evaluates thin-film interference at three wavelengths corresponding to R, G, B.
///
/// @param cos_theta: Cosine of viewing angle
/// @param params: Iridescence parameters (intensity, ior, thickness)
/// @param substrate_ior: IOR of the base material (1.5 for dielectrics, higher for metals)
/// @returns: Interference color as RGB
fn compute_interference_color(
    cos_theta: f32,
    params: IridescenceParams,
    substrate_ior: f32
) -> vec3<f32> {
    // Compute angle inside the film using Snell's law
    let eta_air_film = 1.0 / params.ior;
    let cos_theta_film = snell_cos_theta_t(cos_theta, eta_air_film);

    // Handle total internal reflection (shouldn't happen for air->film with reasonable IOR)
    if cos_theta_film < 0.0 {
        return vec3<f32>(1.0);
    }

    // Fresnel at air-film interface (same for all wavelengths)
    let R_air_film = fresnel_air_film(cos_theta, params.ior);

    // Fresnel at film-substrate interface
    let R_film_substrate = fresnel_film_substrate(cos_theta_film, params.ior, substrate_ior);

    // Compute phase and interference for each wavelength
    let phase_r = compute_film_phase(params.thickness_nm, cos_theta_film, params.ior, WAVELENGTH_R);
    let phase_g = compute_film_phase(params.thickness_nm, cos_theta_film, params.ior, WAVELENGTH_G);
    let phase_b = compute_film_phase(params.thickness_nm, cos_theta_film, params.ior, WAVELENGTH_B);

    let irid_r = compute_interference(R_air_film, R_film_substrate, phase_r);
    let irid_g = compute_interference(R_air_film, R_film_substrate, phase_g);
    let irid_b = compute_interference(R_air_film, R_film_substrate, phase_b);

    return vec3<f32>(irid_r, irid_g, irid_b);
}

// ============================================================================
// Main Iridescence Evaluation
// ============================================================================

/// Evaluate complete iridescence effect.
/// Computes the interference color based on viewing angle and film properties.
///
/// @param cos_theta: Cosine of viewing angle (typically VoH or NoV)
/// @param params: Iridescence parameters
/// @param is_metallic: Whether the substrate is metallic
/// @returns: Iridescence color factor
fn evaluate_iridescence(
    cos_theta: f32,
    params: IridescenceParams,
    is_metallic: bool
) -> vec3<f32> {
    // Quality gating - return neutral if iridescence disabled
    if !QUALITY_IRIDESCENCE_ENABLED {
        return vec3<f32>(1.0);
    }

    // Early exit if no iridescence
    if params.intensity < EPSILON {
        return vec3<f32>(1.0);
    }

    // Estimate substrate IOR
    // Metals have high effective IOR (complex IOR), use 2.5 as approximation
    // Dielectrics typically around 1.5
    let substrate_ior = select(1.5, 2.5, is_metallic);

    // Clamp viewing angle to avoid edge artifacts
    let cos_theta_clamped = max(cos_theta, 0.01);

    // Compute interference colors
    let irid_color = compute_interference_color(cos_theta_clamped, params, substrate_ior);

    return irid_color;
}

/// Apply iridescence effect to base F0.
/// Modulates the specular reflectance with interference colors.
///
/// The final F0 is a blend between the original F0 and the iridescence-modulated F0:
/// F0_final = lerp(F0, F0 * irid_color, intensity)
///
/// For a more physically accurate result, you can replace F0 entirely:
/// F0_final = lerp(F0, irid_color, intensity)
///
/// @param F0: Base Fresnel reflectance at normal incidence
/// @param cos_theta: Cosine of viewing angle
/// @param params: Iridescence parameters
/// @param is_metallic: Whether the surface is metallic
/// @returns: Iridescence-modulated F0
fn apply_iridescence(
    F0: vec3<f32>,
    cos_theta: f32,
    params: IridescenceParams,
    is_metallic: bool
) -> vec3<f32> {
    // Quality gating
    if !QUALITY_IRIDESCENCE_ENABLED {
        return F0;
    }

    // Early exit if no iridescence effect
    if params.intensity < EPSILON {
        return F0;
    }

    // Evaluate iridescence interference
    let irid_color = evaluate_iridescence(cos_theta, params, is_metallic);

    // Blend iridescence with base F0
    // We use a multiplicative blend for metallic (preserves base color character)
    // and an additive/replacement blend for dielectrics (more rainbow effect)
    var F0_irid: vec3<f32>;
    if is_metallic {
        // Metallic: modulate F0 with interference
        F0_irid = F0 * irid_color;
    } else {
        // Dielectric: blend toward interference color
        F0_irid = mix(F0, irid_color, 0.5);
    }

    // Final blend with original F0 based on intensity
    return mix(F0, F0_irid, params.intensity);
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create default iridescence parameters (disabled).
fn iridescence_params_default() -> IridescenceParams {
    var params: IridescenceParams;
    params.intensity = 0.0;
    params.ior = 1.5;
    params.thickness_nm = 400.0;
    return params;
}

/// Create iridescence parameters for common materials.
///
/// @param preset: Material preset name
///   - "soap_bubble": Thin soap film, strong rainbow
///   - "oil_slick": Oil on water, classic iridescence
///   - "beetle": Beetle shell, subtle green-purple
///   - "pearl": Pearl/nacre, subtle pink-green
/// @returns: IridescenceParams for the preset
fn iridescence_preset(preset: u32) -> IridescenceParams {
    var params: IridescenceParams;

    switch preset {
        case 0u: {  // soap_bubble
            params.intensity = 1.0;
            params.ior = 1.33;  // Water-like
            params.thickness_nm = 200.0;  // Very thin
        }
        case 1u: {  // oil_slick
            params.intensity = 0.8;
            params.ior = 1.4;
            params.thickness_nm = 350.0;
        }
        case 2u: {  // beetle
            params.intensity = 0.6;
            params.ior = 1.8;  // Chitin
            params.thickness_nm = 500.0;
        }
        case 3u: {  // pearl
            params.intensity = 0.4;
            params.ior = 1.53;  // Nacre
            params.thickness_nm = 550.0;
        }
        default: {
            params.intensity = 0.0;
            params.ior = 1.5;
            params.thickness_nm = 400.0;
        }
    }

    return params;
}
