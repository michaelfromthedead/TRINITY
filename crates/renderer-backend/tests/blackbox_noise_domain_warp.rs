// SPDX-License-Identifier: MIT
//
// blackbox_noise_domain_warp.rs -- Blackbox tests for noise_domain_warp.wgsl.
//
// This test file verifies the WGSL domain warping source is structurally valid.
// Cleanroom: tests are based ONLY on the spec definition (T-DEMO-1.33:
// "FBM-warped FBM with increased visual complexity through non-linear
//  coordinate deformation. WGSL.").
//
// Coverage:
//   - naga WGSL compilation (full file)
//   - Individual domain_warp_2d, domain_warp_3d, domain_warp_perlin_3d
//     function compilation
//   - Function existence verification
//   - Source structure checks
//   - Warp displacement pattern verification
//   - Separate warp vs base parameter sets
//   - FBM dependency references

use naga::front::wgsl::parse_str;

/// The full WGSL source file, baked in at compile time via include_str!.
static WGSL_SOURCE: &str = include_str!("../src/demoscene/noise_domain_warp.wgsl");

// =============================================================================
// Section 1: naga compilation
// =============================================================================

/// Helper: concatenate all dependency WGSL sources for compilation tests.
/// noise_domain_warp.wgsl depends on noise_fbm.wgsl (which itself depends
/// on noise_hash.wgsl, noise_value.wgsl, noise_perlin.wgsl).
fn dependency_source() -> String {
    format!(
        "{}\n\n{}\n\n{}\n\n{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        include_str!("../src/demoscene/noise_value.wgsl"),
        include_str!("../src/demoscene/noise_perlin.wgsl"),
        include_str!("../src/demoscene/noise_fbm.wgsl"),
        WGSL_SOURCE
    )
}

/// The full noise_domain_warp.wgsl file must compile through naga without errors.
/// Requires all dependencies (hash + value + perlin + fbm).
#[test]
fn compiles_via_naga() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("noise_domain_warp.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// domain_warp_2d must compile with all dependencies.
#[test]
fn domain_warp_2d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"domain_warp_2d"),
                "domain_warp_2d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_domain_warp.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// domain_warp_3d must compile with all dependencies.
#[test]
fn domain_warp_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"domain_warp_3d"),
                "domain_warp_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_domain_warp.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// domain_warp_perlin_3d must compile with all dependencies.
#[test]
fn domain_warp_perlin_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"domain_warp_perlin_3d"),
                "domain_warp_perlin_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_domain_warp.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// Section 2: Function existence
// =============================================================================

/// domain_warp_2d must exist in the source.
#[test]
fn domain_warp_2d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_2d("),
        "domain_warp_2d function not found in noise_domain_warp.wgsl"
    );
}

/// domain_warp_3d must exist in the source.
#[test]
fn domain_warp_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_3d("),
        "domain_warp_3d function not found in noise_domain_warp.wgsl"
    );
}

/// domain_warp_perlin_3d must exist in the source.
#[test]
fn domain_warp_perlin_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_perlin_3d("),
        "domain_warp_perlin_3d function not found in noise_domain_warp.wgsl"
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
        "noise_domain_warp.wgsl must start with the MIT license header"
    );
}

/// The T-DEMO-1.33 section headers must appear.
#[test]
fn section_headers_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.33: Domain Warp 2D"),
        "Section header 'T-DEMO-1.33: Domain Warp 2D' not found"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.33: Domain Warp 3D"),
        "Section header 'T-DEMO-1.33: Domain Warp 3D' not found"
    );
}

/// The file must reference its FBM dependency file.
#[test]
fn dependency_references_present() {
    assert!(
        WGSL_SOURCE.contains("noise_fbm.wgsl"),
        "noise_domain_warp.wgsl must reference noise_fbm.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_hash.wgsl"),
        "noise_domain_warp.wgsl must reference noise_hash.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_value.wgsl"),
        "noise_domain_warp.wgsl must reference noise_value.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_perlin.wgsl"),
        "noise_domain_warp.wgsl must reference noise_perlin.wgsl as a dependency"
    );
}

/// The file must reference the dependency versions T-DEMO-1.28 through T-DEMO-1.31.
#[test]
fn references_dependency_versions() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28"),
        "noise_domain_warp.wgsl must reference T-DEMO-1.28 as hash dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.29"),
        "noise_domain_warp.wgsl must reference T-DEMO-1.29 as value noise dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.30"),
        "noise_domain_warp.wgsl must reference T-DEMO-1.30 as Perlin noise dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.31"),
        "noise_domain_warp.wgsl must reference T-DEMO-1.31 as FBM dependency"
    );
}

/// All domain warp functions must call the corresponding FBM function.
#[test]
fn domain_warp_2d_calls_fbm_2d() {
    let warp_call_count = WGSL_SOURCE.matches("fbm_2d(").count();
    assert!(
        warp_call_count >= 3,
        "domain_warp_2d must call fbm_2d at least 3 times (2 warp + 1 base), found {}",
        warp_call_count
    );
}

#[test]
fn domain_warp_3d_calls_fbm_3d() {
    let warp_call_count = WGSL_SOURCE.matches("fbm_3d(").count();
    assert!(
        warp_call_count >= 3,
        "domain_warp_3d must call fbm_3d at least 3 times (3 warp + 1 base), found {}",
        warp_call_count
    );
}

#[test]
fn domain_warp_perlin_3d_calls_fbm_perlin_3d() {
    let warp_call_count = WGSL_SOURCE.matches("fbm_perlin_3d(").count();
    assert!(
        warp_call_count >= 3,
        "domain_warp_perlin_3d must call fbm_perlin_3d at least 3 times (3 warp + 1 base), found {}",
        warp_call_count
    );
}

/// All domain warp functions must use a strength parameter that scales the warp displacement.
#[test]
fn uses_strength_parameter() {
    assert!(
        WGSL_SOURCE.contains("strength"),
        "Domain warp functions must use a 'strength' parameter for warp magnitude"
    );
}

/// All domain warp functions must have separate warp and base parameter sets.
#[test]
fn has_separate_warp_and_base_parameters() {
    let has_warp_octaves = WGSL_SOURCE.contains("warp_octaves");
    let has_warp_lacunarity = WGSL_SOURCE.contains("warp_lacunarity");
    let has_warp_gain = WGSL_SOURCE.contains("warp_gain");
    let has_base_octaves = WGSL_SOURCE.contains("base_octaves");
    let has_base_lacunarity = WGSL_SOURCE.contains("base_lacunarity");
    let has_base_gain = WGSL_SOURCE.contains("base_gain");

    assert!(
        has_warp_octaves && has_warp_lacunarity && has_warp_gain,
        "Domain warp functions must have warp_octaves, warp_lacunarity, and warp_gain parameters"
    );
    assert!(
        has_base_octaves && has_base_lacunarity && has_base_gain,
        "Domain warp functions must have base_octaves, base_lacunarity, and base_gain parameters"
    );
}

/// All domain warp functions must create a warped_p from p + strength * warp_vector.
#[test]
fn uses_warped_p_variable() {
    assert!(
        WGSL_SOURCE.contains("warped_p"),
        "Domain warp functions must compute 'warped_p = p + strength * warp_vec'"
    );
}

/// All domain warp functions must evaluate FBM at the warped position.
#[test]
fn evaluates_fbm_at_warped_position() {
    let at_warped = WGSL_SOURCE.matches("warped_p").count();
    assert!(
        at_warped >= 3,
        "Domain warp functions must pass warped_p to base FBM evaluation; found {} references",
        at_warped
    );
}

/// The file must have three domain warp subsections, each with delimiters.
#[test]
fn all_subsections_have_delimiters() {
    let sections: Vec<&str> = WGSL_SOURCE.split("// =============").collect();
    // Minimum: license header, 2D, 3D value, 3D Perlin = at least 4 sections
    assert!(
        sections.len() >= 4,
        "Expected at least 4 section delimiters in noise_domain_warp.wgsl, found {}",
        sections.len()
    );
}

/// The file must contain exactly 3 function definitions.
#[test]
fn has_correct_number_of_functions() {
    let fn_count = WGSL_SOURCE.matches("fn domain_warp_").count();
    assert_eq!(
        fn_count, 3,
        "Expected exactly 3 domain_warp_ function definitions, found {}",
        fn_count
    );
}

// =============================================================================
// Section 4: Warp displacement verification
// =============================================================================

/// Each 2D warp function must compute a 2-component warp vector.
#[test]
fn warp_2d_uses_two_warp_components() {
    // Extract the domain_warp_2d function body
    let start = WGSL_SOURCE.find("fn domain_warp_2d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    let has_warp_x = body.contains("warp_x");
    let has_warp_y = body.contains("warp_y");

    assert!(
        has_warp_x && has_warp_y,
        "domain_warp_2d body must compute both warp_x and warp_y components"
    );
}

/// Each 3D warp function must compute a 3-component warp vector.
#[test]
fn warp_3d_uses_three_warp_components() {
    let start = WGSL_SOURCE.find("fn domain_warp_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    let has_warp_x = body.contains("warp_x");
    let has_warp_y = body.contains("warp_y");
    let has_warp_z = body.contains("warp_z");

    assert!(
        has_warp_x && has_warp_y && has_warp_z,
        "domain_warp_3d body must compute warp_x, warp_y, and warp_z components"
    );
}

/// Warp components must use different seed offsets for decorrelation.
#[test]
fn warp_components_have_different_seed_offsets() {
    // Check that warp components use different coordinate offsets for decorrelation
    let offset_100_count = WGSL_SOURCE.matches("100.0").count();
    let offset_200_count = WGSL_SOURCE.matches("200.0").count();

    assert!(
        offset_100_count >= 1,
        "Warp components must include a 100.0 offset for decorrelation"
    );
    assert!(
        offset_200_count >= 1,
        "3D warp components must include a 200.0 offset for z-axis decorrelation"
    );
}

/// Each Perlin warp function must compute a 3-component warp vector.
#[test]
fn warp_perlin_3d_uses_three_warp_components() {
    let start = WGSL_SOURCE.find("fn domain_warp_perlin_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    let has_warp_x = body.contains("warp_x");
    let has_warp_y = body.contains("warp_y");
    let has_warp_z = body.contains("warp_z");

    assert!(
        has_warp_x && has_warp_y && has_warp_z,
        "domain_warp_perlin_3d body must compute warp_x, warp_y, and warp_z components"
    );
}

// =============================================================================
// Section 5: Acceptance criteria (T-DEMO-1.33 spec compliance)
// =============================================================================

/// The spec acceptance formula must be referenced in comments.
#[test]
fn acceptance_formula_referenced() {
    let lower = WGSL_SOURCE.to_lowercase();
    assert!(
        lower.contains("fbm-warped") ||
        lower.contains("fbm warped") ||
        lower.contains("domain warp"),
        "Spec acceptance formula 'FBM-warped FBM' must be referenced in comments"
    );
}

/// The "increased visual complexity" property must be documented per the spec.
#[test]
fn increased_complexity_documented() {
    assert!(
        WGSL_SOURCE.contains("Increased visual complexity") ||
        WGSL_SOURCE.contains("increased visual complexity") ||
        WGSL_SOURCE.contains("Increased complexity"),
        "The 'increased visual complexity' property must be documented in noise_domain_warp.wgsl"
    );
}

/// The "organic" or "swirling" visual property must be documented.
/// Domain warping creates organic, swirling deformation patterns.
#[test]
fn organic_swirling_documented() {
    let lower = WGSL_SOURCE.to_lowercase();
    assert!(
        lower.contains("organic") || lower.contains("swirling"),
        "The organic/swirling visual property must be documented in noise_domain_warp.wgsl"
    );
}

/// The "non-linear" deformation property must be documented.
#[test]
fn non_linear_deformation_documented() {
    let lower = WGSL_SOURCE.to_lowercase();
    assert!(
        lower.contains("non-linear"),
        "The non-linear deformation property must be documented in noise_domain_warp.wgsl"
    );
}

/// Deterministic property must be documented.
/// The spec requires: same input + parameters always produces same output.
#[test]
fn deterministic_property_documented() {
    assert!(
        WGSL_SOURCE.contains("Deterministic") || WGSL_SOURCE.contains("deterministic"),
        "The deterministic property must be documented in noise_domain_warp.wgsl"
    );
}

/// The [-1, 1] output range must be documented (inherited from FBM base).
/// Domain warping uses FBM as base, output in same range as FBM.
#[test]
fn output_range_documented() {
    assert!(
        WGSL_SOURCE.contains("[-1, 1]"),
        "The [-1, 1] output range must be documented in noise_domain_warp.wgsl"
    );
}

// =============================================================================
// Section 6: Warp mechanics verification
// =============================================================================

/// Each 2D warp component must receive a seed offset to decorrelate.
/// warp_y = fbm_2d(p + vec2(100.0), ...) to avoid correlated warp axes.
#[test]
fn warp_2d_components_use_seed_offset() {
    // The 2D warp function should have the warp_y call with a seed offset
    let start = WGSL_SOURCE.find("fn domain_warp_2d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("100.0"),
        "domain_warp_2d must use a 100.0 seed offset for warp_y decorrelation"
    );
}

/// Each 3D warp function must pass warped_p to the base FBM.
#[test]
fn warp_3d_passes_warped_p_to_base() {
    let start = WGSL_SOURCE.find("fn domain_warp_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("fbm_3d(warped_p,"),
        "domain_warp_3d must pass warped_p to the base fbm_3d call"
    );
}

/// Each Perlin warp function must pass warped_p to the base Perlin FBM.
#[test]
fn warp_perlin_3d_passes_warped_p_to_base() {
    let start = WGSL_SOURCE.find("fn domain_warp_perlin_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("fbm_perlin_3d(warped_p,"),
        "domain_warp_perlin_3d must pass warped_p to the base fbm_perlin_3d call"
    );
}

// =============================================================================
// Section 7: Type signature verification
// =============================================================================

/// domain_warp_2d must take a vec2<f32> input (2D coordinate).
#[test]
fn domain_warp_2d_has_vec2_input() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_2d(\n    p: vec2<f32>,"),
        "domain_warp_2d must take p: vec2<f32> for 2D input"
    );
}

/// domain_warp_3d must take a vec3<f32> input (3D coordinate).
#[test]
fn domain_warp_3d_has_vec3_input() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_3d(\n    p: vec3<f32>,"),
        "domain_warp_3d must take p: vec3<f32> for 3D input"
    );
}

/// domain_warp_perlin_3d must take a vec3<f32> input (3D coordinate).
#[test]
fn domain_warp_perlin_3d_has_vec3_input() {
    assert!(
        WGSL_SOURCE.contains("fn domain_warp_perlin_3d(\n    p: vec3<f32>,"),
        "domain_warp_perlin_3d must take p: vec3<f32> for 3D input"
    );
}

/// All domain warp functions must return f32.
///
/// Function signatures span multiple lines (one parameter per line), so we
/// check the return type in the lines following the function declaration.
#[test]
fn all_functions_return_f32() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut fn_count = 0u32;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            // Scan forward from the function declaration until we find "-> f32"
            let mut found_return = false;
            for j in i..lines.len() {
                if lines[j].contains("-> f32") {
                    found_return = true;
                    break;
                }
                // Stop if we hit another function declaration
                if j > i && lines[j].trim().starts_with("fn ") {
                    break;
                }
            }
            if found_return {
                fn_count += 1;
            }
        }
    }

    assert_eq!(
        fn_count, 3,
        "All 3 domain_warp_ functions must return f32; found {} with return type",
        fn_count
    );
}

/// All domain warp functions must accept strength: f32 for warp magnitude.
///
/// Function signatures span multiple lines (one parameter per line), so we
/// check for the "strength: f32" parameter in the lines following the
/// function declaration.
#[test]
fn all_functions_accept_f32_strength() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut fn_count = 0u32;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            let mut found_strength = false;
            for j in i..lines.len() {
                if lines[j].contains("strength: f32") {
                    found_strength = true;
                    break;
                }
                if j > i && lines[j].trim().starts_with("fn ") {
                    break;
                }
            }
            if found_strength {
                fn_count += 1;
            }
        }
    }

    assert_eq!(
        fn_count, 3,
        "All 3 domain_warp_ functions must accept strength: f32; found {}",
        fn_count
    );
}

/// warp_octaves must be u32 (unsigned integer octave count).
///
/// Function signatures span multiple lines, so we scan forward from the fn
/// declaration to find the parameter.
#[test]
fn warp_octaves_is_u32() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut fn_count = 0u32;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            let mut found_param = false;
            for j in i..lines.len() {
                if lines[j].contains("warp_octaves: u32") {
                    found_param = true;
                    break;
                }
                if j > i && lines[j].trim().starts_with("fn ") {
                    break;
                }
            }
            if found_param {
                fn_count += 1;
            }
        }
    }

    assert_eq!(
        fn_count, 3,
        "All 3 domain_warp_ functions must accept warp_octaves: u32; found {}",
        fn_count
    );
}

/// base_octaves must be u32 (unsigned integer octave count).
///
/// Function signatures span multiple lines, so we scan forward from the fn
/// declaration to find the parameter.
#[test]
fn base_octaves_is_u32() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut fn_count = 0u32;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            let mut found_param = false;
            for j in i..lines.len() {
                if lines[j].contains("base_octaves: u32") {
                    found_param = true;
                    break;
                }
                if j > i && lines[j].trim().starts_with("fn ") {
                    break;
                }
            }
            if found_param {
                fn_count += 1;
            }
        }
    }

    assert_eq!(
        fn_count, 3,
        "All 3 domain_warp_ functions must accept base_octaves: u32; found {}",
        fn_count
    );
}

// =============================================================================
// Section 8: Individual function structure verification
// =============================================================================

/// domain_warp_2d must compute warp_x and warp_y, create warped_p, and return fbm_2d at warped_p.
#[test]
fn domain_warp_2d_structure_correct() {
    let start = WGSL_SOURCE.find("fn domain_warp_2d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("let warp_x = fbm_2d("),
        "domain_warp_2d body must compute warp_x from fbm_2d"
    );
    assert!(
        body.contains("let warp_y = fbm_2d("),
        "domain_warp_2d body must compute warp_y from fbm_2d"
    );
    assert!(
        body.contains("let warped_p = p + strength * vec2<f32>"),
        "domain_warp_2d body must compute warped_p from p + strength * warp vector"
    );
    assert!(
        body.contains("return fbm_2d(warped_p,"),
        "domain_warp_2d body must return fbm_2d at warped_p"
    );
}

/// domain_warp_3d must compute warp_x, warp_y, and warp_z, then return fbm_3d at warped_p.
#[test]
fn domain_warp_3d_structure_correct() {
    let start = WGSL_SOURCE.find("fn domain_warp_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("let warp_x = fbm_3d("),
        "domain_warp_3d body must compute warp_x from fbm_3d"
    );
    assert!(
        body.contains("let warp_y = fbm_3d("),
        "domain_warp_3d body must compute warp_y from fbm_3d"
    );
    assert!(
        body.contains("let warp_z = fbm_3d("),
        "domain_warp_3d body must compute warp_z from fbm_3d"
    );
    assert!(
        body.contains("let warped_p = p + strength * vec3<f32>"),
        "domain_warp_3d body must compute warped_p from p + strength * 3D warp vector"
    );
    assert!(
        body.contains("return fbm_3d(warped_p,"),
        "domain_warp_3d body must return fbm_3d at warped_p"
    );
}

/// domain_warp_perlin_3d structure must match domain_warp_3d but use Perlin FBM.
#[test]
fn domain_warp_perlin_3d_structure_correct() {
    let start = WGSL_SOURCE.find("fn domain_warp_perlin_3d(").unwrap_or(0);
    let remaining = &WGSL_SOURCE[start..];
    let end = remaining.find("\n}\n").map(|i| start + i + 3).unwrap_or(WGSL_SOURCE.len());
    let body = &WGSL_SOURCE[start..end];

    assert!(
        body.contains("let warp_x = fbm_perlin_3d("),
        "domain_warp_perlin_3d body must compute warp_x from fbm_perlin_3d"
    );
    assert!(
        body.contains("let warp_y = fbm_perlin_3d("),
        "domain_warp_perlin_3d body must compute warp_y from fbm_perlin_3d"
    );
    assert!(
        body.contains("let warp_z = fbm_perlin_3d("),
        "domain_warp_perlin_3d body must compute warp_z from fbm_perlin_3d"
    );
    assert!(
        body.contains("let warped_p = p + strength * vec3<f32>"),
        "domain_warp_perlin_3d body must compute warped_p from p + strength * 3D warp vector"
    );
    assert!(
        body.contains("return fbm_perlin_3d(warped_p,"),
        "domain_warp_perlin_3d body must return fbm_perlin_3d at warped_p"
    );
}

// =============================================================================
// Section 9: Source integrity
// =============================================================================

/// All functions must have doc comments (///) before their definition.
#[test]
fn all_functions_have_doc_comments() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut missing_doc = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            let mut found_doc = false;
            for j in (0..i).rev() {
                let prev = lines[j].trim();
                if prev.starts_with("///") {
                    found_doc = true;
                    break;
                } else if prev.is_empty() || prev.starts_with("//") {
                    continue;
                } else {
                    break;
                }
            }
            if !found_doc {
                missing_doc.push(trimmed);
            }
        }
    }

    assert!(
        missing_doc.is_empty(),
        "Functions missing doc comments: {:?}",
        missing_doc
    );
}

/// The file must contain exactly 3 function definitions total (no extras, no duplicates).
#[test]
fn no_duplicate_function_definitions() {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, usize> = HashMap::new();

    for line in WGSL_SOURCE.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn domain_warp_") {
            let name = trimmed.split('(').next().unwrap_or("");
            *counts.entry(name).or_insert(0) += 1;
        }
    }

    let duplicates: Vec<&&str> = counts.keys()
        .filter(|k| counts[*k] > 1)
        .collect();

    assert!(
        duplicates.is_empty(),
        "Duplicate function definitions found: {:?}",
        duplicates
    );
}

/// The file must not have trailing whitespace (cleanroom quality).
#[test]
fn no_trailing_whitespace() {
    for (i, line) in WGSL_SOURCE.lines().enumerate() {
        let line_num = i + 1;
        assert!(
            !line.ends_with(' ') && !line.ends_with('\t'),
            "Line {} has trailing whitespace: {:?}",
            line_num,
            line
        );
    }
}

/// Every T-DEMO-1.33 annotation must appear -- 1 at file scope plus
/// 3 in section headers just before each function definition.
#[test]
fn all_section_headers_reference_spec() {
    let header_count = WGSL_SOURCE.matches("T-DEMO-1.33:").count();
    assert_eq!(
        header_count, 4,
        "Expected exactly 4 'T-DEMO-1.33:' annotations in noise_domain_warp.wgsl \
         (1 file-scope + 3 section headers), found {}",
        header_count
    );
}

/// Each T-DEMO-1.33 section header is followed by the expected function definition.
/// The 3 headers respectively precede domain_warp_2d, domain_warp_3d,
/// and domain_warp_perlin_3d.
#[test]
fn each_header_precedes_correct_function() {
    let expected_pairs = [
        ("Domain Warp 2D", "fn domain_warp_2d("),
        ("Domain Warp 3D", "fn domain_warp_3d("),
        ("Domain Warp 3D (Perlin", "fn domain_warp_perlin_3d("),
    ];

    for (header_tag, fn_sig) in &expected_pairs {
        let header_pos = WGSL_SOURCE.find(header_tag)
            .unwrap_or_else(|| panic!("Header '{}' not found", header_tag));
        let fn_pos = WGSL_SOURCE.find(fn_sig)
            .unwrap_or_else(|| panic!("Function '{}' not found", fn_sig));
        assert!(
            header_pos < fn_pos,
            "Header '{}' must appear before function '{}'",
            header_tag, fn_sig
        );
        let between = &WGSL_SOURCE[header_pos..fn_pos];
        let intrusions: Vec<&str> = ["fn domain_warp_2d(", "fn domain_warp_3d(",
                                      "fn domain_warp_perlin_3d("]
            .iter()
            .filter(|s| **s != *fn_sig && between.contains(**s))
            .copied()
            .collect();
        assert!(
            intrusions.is_empty(),
            "Header '{}' has intruding function(s) between it and '{}': {:?}",
            header_tag, fn_sig, intrusions
        );
    }
}
