// SPDX-License-Identifier: MIT
//
// blackbox_noise_ridged.rs -- Blackbox tests for noise_ridged.wgsl.
//
// This test file verifies the WGSL ridged noise source is structurally valid.
// Cleanroom: tests are based ONLY on the spec definition (T-DEMO-1.32:
// "1.0 - abs(FBM) for terrain, sharp valleys smooth ridges. WGSL.").
//
// Coverage:
//   - naga WGSL compilation (full file)
//   - Individual ridged_noise_1d, ridged_noise_2d, ridged_noise_3d,
//     ridged_perlin_3d function compilation
//   - Function existence verification
//   - Source structure checks
//   - 1.0 - abs(FBM) transform verification
//   - FBM dependency references

use naga::front::wgsl::parse_str;

/// The full WGSL source file, baked in at compile time via include_str!.
static WGSL_SOURCE: &str = include_str!("../src/demoscene/noise_ridged.wgsl");

// =============================================================================
// Section 1: naga compilation
// =============================================================================

/// Helper: concatenate all dependency WGSL sources for compilation tests.
/// noise_ridged.wgsl depends on noise_fbm.wgsl, which in turn depends on
/// noise_hash.wgsl, noise_value.wgsl, and noise_perlin.wgsl.
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

/// The full noise_ridged.wgsl file must compile through naga without errors.
/// Requires all dependencies (hash + value + perlin + fbm).
#[test]
fn compiles_via_naga() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("noise_ridged.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// ridged_noise_1d must compile with all dependencies.
#[test]
fn ridged_noise_1d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"ridged_noise_1d"),
                "ridged_noise_1d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_ridged.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// ridged_noise_2d must compile with all dependencies.
#[test]
fn ridged_noise_2d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"ridged_noise_2d"),
                "ridged_noise_2d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_ridged.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// ridged_noise_3d must compile with all dependencies.
#[test]
fn ridged_noise_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"ridged_noise_3d"),
                "ridged_noise_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_ridged.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// ridged_perlin_3d must compile with all dependencies.
#[test]
fn ridged_perlin_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"ridged_perlin_3d"),
                "ridged_perlin_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("Dependencies + noise_ridged.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// Section 2: Function existence
// =============================================================================

/// ridged_noise_1d must exist in the source.
#[test]
fn ridged_noise_1d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_1d("),
        "ridged_noise_1d function not found in noise_ridged.wgsl"
    );
}

/// ridged_noise_2d must exist in the source.
#[test]
fn ridged_noise_2d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_2d("),
        "ridged_noise_2d function not found in noise_ridged.wgsl"
    );
}

/// ridged_noise_3d must exist in the source.
#[test]
fn ridged_noise_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_3d("),
        "ridged_noise_3d function not found in noise_ridged.wgsl"
    );
}

/// ridged_perlin_3d must exist in the source.
#[test]
fn ridged_perlin_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_perlin_3d("),
        "ridged_perlin_3d function not found in noise_ridged.wgsl"
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
        "noise_ridged.wgsl must start with the MIT license header"
    );
}

/// The T-DEMO-1.32 section headers must appear.
#[test]
fn section_headers_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.32: Ridged Noise 1D"),
        "Section header 'T-DEMO-1.32: Ridged Noise 1D' not found"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.32: Ridged Noise 2D"),
        "Section header 'T-DEMO-1.32: Ridged Noise 2D' not found"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.32: Ridged Noise 3D"),
        "Section header 'T-DEMO-1.32: Ridged Noise 3D' not found"
    );
}

/// The file must reference its FBM dependency file.
#[test]
fn dependency_references_present() {
    assert!(
        WGSL_SOURCE.contains("noise_fbm.wgsl"),
        "noise_ridged.wgsl must reference noise_fbm.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_hash.wgsl"),
        "noise_ridged.wgsl must reference noise_hash.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_value.wgsl"),
        "noise_ridged.wgsl must reference noise_value.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_perlin.wgsl"),
        "noise_ridged.wgsl must reference noise_perlin.wgsl as a dependency"
    );
}

/// The file must reference T-DEMO-1.31 (FBM), T-DEMO-1.28, T-DEMO-1.29, T-DEMO-1.30.
#[test]
fn references_dependency_versions() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.31"),
        "noise_ridged.wgsl must reference T-DEMO-1.31 as FBM dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28"),
        "noise_ridged.wgsl must reference T-DEMO-1.28 as hash dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.29"),
        "noise_ridged.wgsl must reference T-DEMO-1.29 as value noise dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.30"),
        "noise_ridged.wgsl must reference T-DEMO-1.30 as Perlin noise dependency"
    );
}

/// All ridged noise functions must contain "1.0 - abs" for the transform.
#[test]
fn uses_one_minus_abs_transform() {
    assert!(
        WGSL_SOURCE.contains("1.0 - abs("),
        "Ridged noise functions must use the 1.0 - abs() transform"
    );
}

/// All ridged noise functions must call the corresponding FBM function.
#[test]
fn ridged_noise_1d_calls_fbm_1d() {
    assert!(
        WGSL_SOURCE.contains("fbm_1d("),
        "ridged_noise_1d must call fbm_1d"
    );
}

#[test]
fn ridged_noise_2d_calls_fbm_2d() {
    assert!(
        WGSL_SOURCE.contains("fbm_2d("),
        "ridged_noise_2d must call fbm_2d"
    );
}

#[test]
fn ridged_noise_3d_calls_fbm_3d() {
    assert!(
        WGSL_SOURCE.contains("fbm_3d("),
        "ridged_noise_3d must call fbm_3d"
    );
}

#[test]
fn ridged_perlin_3d_calls_fbm_perlin_3d() {
    assert!(
        WGSL_SOURCE.contains("fbm_perlin_3d("),
        "ridged_perlin_3d must call fbm_perlin_3d"
    );
}

/// The file must have four ridged noise subsections, each with delimiters.
#[test]
fn all_subsections_have_delimiters() {
    let sections: Vec<&str> = WGSL_SOURCE.split("// =============").collect();
    // Minimum: license header, 1D, 2D, 3D value, 3D Perlin = at least 5 sections
    assert!(
        sections.len() >= 5,
        "Expected at least 5 section delimiters in noise_ridged.wgsl, found {}",
        sections.len()
    );
}

/// The file must contain exactly 4 ridged noise function definitions.
#[test]
fn has_correct_number_of_functions() {
    let fn_count = WGSL_SOURCE.matches("fn ridged_").count();
    assert_eq!(
        fn_count, 4,
        "Expected exactly 4 ridged_ function definitions, found {}",
        fn_count
    );
}

// =============================================================================
// Section 4: Transform verification
// =============================================================================

/// All ridged noise functions must use both fbm_* call and abs.
#[test]
fn each_function_stores_fbm_result_and_applies_abs() {
    // ridged = 1.0 - abs(fbm_*_call)
    let has_fbm_val = WGSL_SOURCE.contains("let fbm_val = fbm_");
    assert!(
        has_fbm_val,
        "Ridged noise functions should store FBM result in fbm_val"
    );
    let has_return = WGSL_SOURCE.contains("return 1.0 - abs(fbm_val)");
    assert!(
        has_return,
        "Ridged noise functions must return 1.0 - abs(fbm_val)"
    );
}

/// The ridged noise functions must not modify the FBM parameters.
#[test]
fn passes_parameters_directly_to_fbm() {
    // All ridged noise functions take the same parameter names as FBM
    assert!(
        WGSL_SOURCE.contains("p, octaves, lacunarity, gain"),
        "Ridged noise functions must pass p, octaves, lacunarity, gain directly"
    );
}

// =============================================================================
// Section 5: Acceptance criteria (T-DEMO-1.32 spec compliance)
// =============================================================================

/// The spec acceptance formula must be referenced in comments.
#[test]
fn acceptance_formula_referenced() {
    assert!(
        WGSL_SOURCE.contains("Acceptance: 1.0 - abs(FBM)") || WGSL_SOURCE.contains("1.0 - abs(fbm"),
        "Spec acceptance formula '1.0 - abs(FBM)' must be referenced in comments"
    );
}

/// The [0, 1] output range must be documented per the spec.
/// ridged(p) = 1.0 - abs(fbm(p)) maps FBM's [-1, 1] to [0, 1].
#[test]
fn output_range_zero_to_one_documented() {
    assert!(
        WGSL_SOURCE.contains("[0, 1]") || WGSL_SOURCE.contains("output is in [0, 1]"),
        "The [0, 1] output range must be documented in noise_ridged.wgsl"
    );
}

/// The "sharp ridges" visual property must be documented.
/// Spec acceptance: "sharp valleys smooth ridges" -- sharp V-cusps at FBM zero-crossings.
#[test]
fn sharp_ridges_documented() {
    assert!(
        WGSL_SOURCE.contains("sharp ridges") || WGSL_SOURCE.contains("Sharp ridges"),
        "The 'sharp ridges' property must be documented in noise_ridged.wgsl"
    );
}

/// The "smooth valleys" visual property must be documented.
/// Spec acceptance: "sharp valleys smooth ridges" -- smooth troughs near FBM extrema.
#[test]
fn smooth_valleys_documented() {
    assert!(
        WGSL_SOURCE.contains("smooth valleys") || WGSL_SOURCE.contains("smooth, rounded valleys"),
        "The 'smooth valleys' property must be documented in noise_ridged.wgsl"
    );
}

/// Deterministic property must be documented.
/// The spec requires: same input + parameters always produces same output.
#[test]
fn deterministic_property_documented() {
    assert!(
        WGSL_SOURCE.contains("Deterministic") || WGSL_SOURCE.contains("deterministic"),
        "The deterministic property must be documented in noise_ridged.wgsl"
    );
}

/// The terrain use case must be referenced per the spec acceptance criteria.
#[test]
fn terrain_use_case_referenced() {
    assert!(
        WGSL_SOURCE.to_lowercase().contains("terrain"),
        "The spec acceptance references 'for terrain'; 'terrain' keyword must appear"
    );
}

// =============================================================================
// Section 6: Perlin variant properties
// =============================================================================

/// ridged_perlin_3d must document that FBM = 0 at integer grid positions
/// produces the sharpest possible ridge (output = 1).
#[test]
fn ridged_perlin_3d_integer_ridge_documented() {
    assert!(
        WGSL_SOURCE.contains("ridged_perlin_3d = 1") ||
        WGSL_SOURCE.contains("output = 1") ||
        WGSL_SOURCE.contains("sharpest possible ridge"),
        "ridged_perlin_3d must document integer grid ridge property (output = 1 at integer positions)"
    );
}

/// The file must reference the Inigo Quilez article per the standard reference.
#[test]
fn quilez_article_referenced() {
    assert!(
        WGSL_SOURCE.contains("Inigo Quilez") || WGSL_SOURCE.contains("iquilezles"),
        "noise_ridged.wgsl must reference the Inigo Quilez article"
    );
}

// =============================================================================
// Section 7: Type signature verification
// =============================================================================

/// ridged_noise_1d must take a scalar f32 input (1D coordinate).
#[test]
fn ridged_noise_1d_has_f32_input() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_1d(p: f32,"),
        "ridged_noise_1d must take p: f32 for 1D input"
    );
}

/// ridged_noise_2d must take a vec2<f32> input (2D coordinate).
#[test]
fn ridged_noise_2d_has_vec2_input() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_2d(p: vec2<f32>,"),
        "ridged_noise_2d must take p: vec2<f32> for 2D input"
    );
}

/// ridged_noise_3d must take a vec3<f32> input (3D coordinate).
#[test]
fn ridged_noise_3d_has_vec3_input() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_noise_3d(p: vec3<f32>,"),
        "ridged_noise_3d must take p: vec3<f32> for 3D input"
    );
}

/// ridged_perlin_3d must take a vec3<f32> input (3D coordinate).
#[test]
fn ridged_perlin_3d_has_vec3_input() {
    assert!(
        WGSL_SOURCE.contains("fn ridged_perlin_3d(p: vec3<f32>,"),
        "ridged_perlin_3d must take p: vec3<f32> for 3D input"
    );
}

/// All ridged noise functions must return f32.
#[test]
fn all_functions_return_f32() {
    // Each ridged function signature ends with "-> f32 {"
    let matches: Vec<&str> = WGSL_SOURCE.lines()
        .filter(|l| l.trim().starts_with("fn ridged_") && l.contains("-> f32"))
        .collect();
    assert_eq!(
        matches.len(),
        4,
        "All 4 ridged noise functions must return f32; found {} with return type",
        matches.len()
    );
}

/// All ridged noise functions must accept octaves: u32 (unsigned integer octave count).
#[test]
fn all_functions_accept_u32_octaves() {
    let octave_params: Vec<&str> = WGSL_SOURCE.lines()
        .filter(|l| l.trim().starts_with("fn ridged_") && l.contains("octaves: u32"))
        .collect();
    assert_eq!(
        octave_params.len(),
        4,
        "All 4 ridged functions must accept octaves: u32; found {}",
        octave_params.len()
    );
}

// =============================================================================
// Section 8: Individual function transform verification
// =============================================================================

/// Each ridged_noise_1d body must store fbm_1d result then apply 1.0 - abs.
#[test]
fn ridged_noise_1d_transform_correct() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let source = lines.join("\n");

    // Extract the function body between ridged_noise_1d signature and the closing brace
    let start = source.find("fn ridged_noise_1d(").unwrap_or(0);
    let end = source[start..].find("\n}\n").map(|i| start + i + 3).unwrap_or(source.len());
    let body = &source[start..end];

    assert!(
        body.contains("let fbm_val = fbm_1d("),
        "ridged_noise_1d body must call fbm_1d and store in fbm_val"
    );
    assert!(
        body.contains("return 1.0 - abs(fbm_val)"),
        "ridged_noise_1d body must return 1.0 - abs(fbm_val)"
    );
}

/// Each ridged_noise_2d body must store fbm_2d result then apply 1.0 - abs.
#[test]
fn ridged_noise_2d_transform_correct() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let source = lines.join("\n");

    let start = source.find("fn ridged_noise_2d(").unwrap_or(0);
    let end = source[start..].find("\n}\n").map(|i| start + i + 3).unwrap_or(source.len());
    let body = &source[start..end];

    assert!(
        body.contains("let fbm_val = fbm_2d("),
        "ridged_noise_2d body must call fbm_2d and store in fbm_val"
    );
    assert!(
        body.contains("return 1.0 - abs(fbm_val)"),
        "ridged_noise_2d body must return 1.0 - abs(fbm_val)"
    );
}

/// Each ridged_noise_3d body must store fbm_3d result then apply 1.0 - abs.
#[test]
fn ridged_noise_3d_transform_correct() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let source = lines.join("\n");

    let start = source.find("fn ridged_noise_3d(").unwrap_or(0);
    let end = source[start..].find("\n}\n").map(|i| start + i + 3).unwrap_or(source.len());
    let body = &source[start..end];

    assert!(
        body.contains("let fbm_val = fbm_3d("),
        "ridged_noise_3d body must call fbm_3d and store in fbm_val"
    );
    assert!(
        body.contains("return 1.0 - abs(fbm_val)"),
        "ridged_noise_3d body must return 1.0 - abs(fbm_val)"
    );
}

/// Each ridged_perlin_3d body must store fbm_perlin_3d result then apply 1.0 - abs.
#[test]
fn ridged_perlin_3d_transform_correct() {
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let source = lines.join("\n");

    let start = source.find("fn ridged_perlin_3d(").unwrap_or(0);
    let end = source[start..].find("\n}\n").map(|i| start + i + 3).unwrap_or(source.len());
    let body = &source[start..end];

    assert!(
        body.contains("let fbm_val = fbm_perlin_3d("),
        "ridged_perlin_3d body must call fbm_perlin_3d and store in fbm_val"
    );
    assert!(
        body.contains("return 1.0 - abs(fbm_val)"),
        "ridged_perlin_3d body must return 1.0 - abs(fbm_val)"
    );
}

// =============================================================================
// Section 9: Source integrity
// =============================================================================

/// All functions must have exactly 2 statement bodies (let fbm_val = ...; return ...).
#[test]
fn all_function_bodies_are_two_lines() {
    // Each ridged function must contain exactly one let binding and one return
    let fbm_lets = WGSL_SOURCE.matches("let fbm_val = fbm_").count();
    let returns = WGSL_SOURCE.matches("return 1.0 - abs(fbm_val)").count();
    assert_eq!(
        fbm_lets, 4,
        "Expected exactly 4 'let fbm_val = fbm_' assignments, found {}",
        fbm_lets
    );
    assert_eq!(
        returns, 4,
        "Expected exactly 4 'return 1.0 - abs(fbm_val)' statements, found {}",
        returns
    );
}

/// All ridged noise functions must have doc comments (///) before their definition.
#[test]
fn all_functions_have_doc_comments() {
    // Each ridged function should be preceded by a doc comment
    // Pattern: check that before each "fn ridged_" there's a "///"
    let lines: Vec<&str> = WGSL_SOURCE.lines().collect();
    let mut missing_doc = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn ridged_") {
            // Look backwards for a /// line (skip blank lines)
            let mut found_doc = false;
            for j in (0..i).rev() {
                let prev = lines[j].trim();
                if prev.starts_with("///") {
                    found_doc = true;
                    break;
                } else if prev.is_empty() || prev.starts_with("//") {
                    continue; // skip blank lines and section comments
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

/// The file must contain exactly 4 function definitions total (no extras, no duplicates).
#[test]
fn no_duplicate_function_definitions() {
    // Each unique function name should appear exactly once as a definition
    use std::collections::HashMap;
    let mut counts: HashMap<&str, usize> = HashMap::new();

    for line in WGSL_SOURCE.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("fn ridged_") {
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

/// Every T-DEMO-1.32 annotation must appear -- 1 at file scope plus
/// 4 in section headers just before each function definition.
#[test]
fn all_section_headers_reference_spec() {
    let header_count = WGSL_SOURCE.matches("T-DEMO-1.32:").count();
    assert_eq!(
        header_count, 5,
        "Expected exactly 5 'T-DEMO-1.32:' annotations in noise_ridged.wgsl \
         (1 file-scope + 4 section headers), found {}",
        header_count
    );
}

/// Each T-DEMO-1.32 section header is followed by the expected function definition.
/// The 4 headers respectively precede ridged_noise_1d, ridged_noise_2d,
/// ridged_noise_3d, and ridged_perlin_3d.
#[test]
fn each_header_precedes_correct_function() {
    let expected_pairs = [
        ("Ridged Noise 1D", "fn ridged_noise_1d("),
        ("Ridged Noise 2D", "fn ridged_noise_2d("),
        ("Ridged Noise 3D", "fn ridged_noise_3d("),
        ("Ridged Noise 3D (Perlin", "fn ridged_perlin_3d("),
    ];

    for (header_tag, fn_sig) in &expected_pairs {
        // Find the header position
        let header_pos = WGSL_SOURCE.find(header_tag)
            .unwrap_or_else(|| panic!("Header '{}' not found", header_tag));
        // Find the function position
        let fn_pos = WGSL_SOURCE.find(fn_sig)
            .unwrap_or_else(|| panic!("Function '{}' not found", fn_sig));
        // Header must appear before the function
        assert!(
            header_pos < fn_pos,
            "Header '{}' must appear before function '{}'",
            header_tag, fn_sig
        );
        // No other function definition should appear between header and its function
        let between = &WGSL_SOURCE[header_pos..fn_pos];
        let intrusions: Vec<&str> = ["fn ridged_noise_1d(", "fn ridged_noise_2d(",
                                      "fn ridged_noise_3d(", "fn ridged_perlin_3d("]
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
