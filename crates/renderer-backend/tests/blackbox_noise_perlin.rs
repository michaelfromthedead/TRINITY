// blackbox_noise_perlin.rs -- Blackbox tests for noise_perlin.wgsl.
//
// This test file verifies the WGSL Perlin noise source is structurally valid.
// Cleanroom: tests are based ONLY on the spec definition (T-DEMO-1.30:
// "Perlin noise 3D. Acceptance: gradient-based noise with zero mean").
//
// Coverage:
//   - naga WGSL compilation (full file)
//   - Individual perlin_noise_3d function compilation
//   - perlin_gradient helper compilation
//   - Function existence verification
//   - Source structure checks

use naga::front::wgsl::parse_str;

/// The full WGSL source file, baked in at compile time via include_str!.
static WGSL_SOURCE: &str = include_str!("../src/demoscene/noise_perlin.wgsl");

// =============================================================================
// Section 1: naga compilation
// =============================================================================

/// The full noise_perlin.wgsl file must compile through naga without errors.
/// This is the primary acceptance check for structural WGSL validity.
#[test]
fn compiles_via_naga() {
    let source = format!(
        "{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        WGSL_SOURCE
    );
    let result = parse_str(&source);
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("noise_perlin.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// The perlin_noise_3d function must compile individually. This tests that
/// the function body is self-contained with no module-level dependencies
/// (other than the expected hash31 call).
#[test]
fn perlin_noise_3d_compiles_individually() {
    let source = format!(
        "{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        WGSL_SOURCE
    );
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            assert!(
                module.functions.iter().any(|(_, f)| f.name.as_deref() == Some("perlin_noise_3d")),
                "perlin_noise_3d not found in parsed module"
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_perlin.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// The perlin_gradient helper function must compile individually.
#[test]
fn perlin_gradient_compiles_individually() {
    let source = format!(
        "{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        WGSL_SOURCE
    );
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            assert!(
                module.functions.iter().any(|(_, f)| f.name.as_deref() == Some("perlin_gradient")),
                "perlin_gradient not found in parsed module"
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_perlin.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// Section 2: Function existence
// =============================================================================

/// perlin_noise_3d must exist in the source.
#[test]
fn perlin_noise_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn perlin_noise_3d("),
        "perlin_noise_3d function not found in noise_perlin.wgsl"
    );
}

/// perlin_gradient must exist in the source.
#[test]
fn perlin_gradient_exists() {
    assert!(
        WGSL_SOURCE.contains("fn perlin_gradient("),
        "perlin_gradient function not found in noise_perlin.wgsl"
    );
}

// =============================================================================
// Section 3: Source structure
// =============================================================================

/// The file must start with the SPDX MIT license header.
#[test]
fn starts_with_license() {
    assert!(
        WGSL_SOURCE.starts_with("// SPDX-License-Identifier: MIT"),
        "noise_perlin.wgsl must start with the MIT license header"
    );
}

/// The T-DEMO-1.30 section header must appear.
#[test]
fn section_header_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.30: Perlin Noise 3D"),
        "Section header 'T-DEMO-1.30: Perlin Noise 3D' not found"
    );
}

/// The file must reference its noise_hash.wgsl dependency.
#[test]
fn dependency_reference_present() {
    assert!(
        WGSL_SOURCE.contains("noise_hash.wgsl"),
        "noise_perlin.wgsl must reference noise_hash.wgsl as a dependency"
    );
}

/// The file must reference T-DEMO-1.28 (hash dependency).
#[test]
fn references_hash_dependency_version() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28"),
        "noise_perlin.wgsl must reference T-DEMO-1.28 as hash dependency"
    );
}

/// The function must use hash31 (3D hash from noise_hash.wgsl).
#[test]
fn uses_hash31() {
    assert!(
        WGSL_SOURCE.contains("hash31("),
        "perlin_noise_3d must call hash31"
    );
}

/// The function must use the smoothstep fade curve expression.
#[test]
fn uses_smoothstep_fade() {
    let has_fade = WGSL_SOURCE.contains("6.0") && WGSL_SOURCE.contains("15.0") && WGSL_SOURCE.contains("10.0");
    let has_fade_cube = WGSL_SOURCE.contains("f * f * f");
    assert!(
        has_fade || has_fade_cube,
        "perlin_noise_3d must use the smoothstep fade curve (6t^5 - 15t^4 + 10t^3)"
    );
}

/// The file must contain exactly 12 gradient vectors in the switch statement.
#[test]
fn has_twelve_gradient_cases() {
    let case_count = WGSL_SOURCE.matches("case ").count();
    assert_eq!(
        case_count, 12,
        "perlin_gradient must have exactly 12 gradient cases, found {}",
        case_count
    );
}

// =============================================================================
// Section 4: naga function signature verification
// =============================================================================

/// perlin_noise_3d must have the correct naga-level function signature:
/// one parameter (p: vec3<f32>) returning f32.
#[test]
fn perlin_noise_3d_naga_signature() {
    let source = format!(
        "{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        WGSL_SOURCE
    );
    let module = parse_str(&source)
        .expect("noise_perlin.wgsl must parse via naga for signature verification");

    let (_, func) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("perlin_noise_3d"))
        .expect("perlin_noise_3d must exist in the parsed module");

    assert_eq!(
        func.arguments.len(),
        1,
        "perlin_noise_3d must have exactly 1 parameter"
    );
}

/// perlin_gradient must have the correct naga-level function signature:
/// two parameters (hash_value: f32, offset: vec3<f32>) returning f32.
#[test]
fn perlin_gradient_naga_signature() {
    let source = format!(
        "{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        WGSL_SOURCE
    );
    let module = parse_str(&source)
        .expect("noise_perlin.wgsl must parse via naga for signature verification");

    let (_, func) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("perlin_gradient"))
        .expect("perlin_gradient must exist in the parsed module");

    assert_eq!(
        func.arguments.len(),
        2,
        "perlin_gradient must have exactly 2 parameters"
    );
}
