// SPDX-License-Identifier: MIT
//
// blackbox_noise_hash.rs -- Blackbox tests for noise_hash.wgsl hash functions.
//
// These tests load the WGSL source as a raw string and verify:
//   - All 7 documented hash functions exist in the parsed module
//   - Compilation via naga (WebGPU shader language parser)
//   - Section headers match the WGSL source organization
//
// These are CONTRACT tests (not unit tests). They validate the WGSL source
// code invariants at the text level, independent of any specific renderer
// implementation.

// =============================================================================
// Test fixture: load WGSL source
// =============================================================================

static WGSL_SOURCE: &str = include_str!("../src/demoscene/noise_hash.wgsl");

// =============================================================================
// SECTION 1 -- Well-formedness & header
// =============================================================================

/// WGSL source should not contain a BOM.
#[test]
fn no_bom() {
    assert!(
        !WGSL_SOURCE.starts_with('\u{feff}'),
        "File must not start with a BOM"
    );
}

/// File starts with the SPDX license header.
#[test]
fn starts_with_license() {
    assert!(
        WGSL_SOURCE.starts_with("// SPDX-License-Identifier: MIT"),
        "File must start with the MIT license header"
    );
}

/// Every line is either: blank, a comment, or a valid WGSL construct.
#[test]
fn no_stray_text() {
    for (i, line) in WGSL_SOURCE.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        assert!(
            trimmed.ends_with(';')
                || trimmed.ends_with('{')
                || trimmed.ends_with('}')
                || trimmed.ends_with(')')
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("var ")
                || trimmed.starts_with("let "),
            "Line {} has unexpected non-comment content: {:?}",
            i + 1,
            trimmed
        );
    }
}

// =============================================================================
// SECTION 2 -- Section headers match WGSL source
// =============================================================================

/// Section header for T-DEMO-1.28 Scalar Hash Functions is present.
#[test]
fn scalar_hash_section_header_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28: Scalar Hash Functions (1D-4D -> f32)"),
        "Scalar hash section header not found"
    );
}

/// Section header for T-DEMO-1.28 Vector Hash Functions is present.
#[test]
fn vector_hash_section_header_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28: Vector Hash Functions (2D-3D -> vec2/vec3)"),
        "Vector hash section header not found"
    );
}

// =============================================================================
// SECTION 3 -- Hash function compilation via naga
// =============================================================================

/// The naga compile test: parse the WGSL source via naga and verify it
/// produces no errors. This catches syntax errors, type mismatches, and
/// unsupported constructs.
#[test]
fn compiles_via_naga() {
    let module = naga::front::wgsl::parse_str(WGSL_SOURCE);
    match module {
        Ok(_) => {} // Success
        Err(err) => {
            panic!("WGSL failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// SECTION 4 -- All 7 hash functions exist in parsed module
// =============================================================================

/// Returns the set of function names defined in the parsed WGSL module.
fn find_hash_function_names(source: &str) -> Vec<String> {
    let module = naga::front::wgsl::parse_str(source)
        .expect("WGSL source must compile for function enumeration");

    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter_map(|(_, func)| func.name.clone())
        .filter(|name| name.starts_with("hash"))
        .collect();
    names.sort();
    names
}

/// All 7 documented hash functions exist in the parsed module.
#[test]
fn contains_all_hash_functions() {
    let expected: Vec<String> = vec![
        "hash11",
        "hash21",
        "hash22",
        "hash31",
        "hash32",
        "hash33",
        "hash41",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let names = find_hash_function_names(WGSL_SOURCE);

    for want in &expected {
        assert!(
            names.contains(want),
            "Expected hash function '{}' not found in parsed module. Found: {:?}",
            want,
            names
        );
    }
}

/// There are exactly 7 hash functions in the parsed module.
#[test]
fn hash_function_count() {
    let names = find_hash_function_names(WGSL_SOURCE);
    assert_eq!(
        names.len(),
        7,
        "Expected exactly 7 hash functions, found {}: {:?}",
        names.len(),
        names
    );
}

// =============================================================================
// SECTION 5 -- Return type verification
// =============================================================================

/// All scalar hash functions (hash11, hash21, hash31, hash41) return f32.
#[test]
fn scalar_hash_functions_return_f32() {
    let fn_re = regex::Regex::new(
        r"(?m)^\s*fn\s+(hash[1-4]1)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]*)"
    ).unwrap();

    for cap in fn_re.captures_iter(WGSL_SOURCE) {
        let fn_name = cap.get(1).unwrap().as_str().to_string();
        let ret = cap.get(2).unwrap().as_str().trim();
        assert_eq!(
            ret, "f32",
            "Hash function '{}' must return 'f32', found '{}'",
            fn_name, ret
        );
    }
}

/// Vector hash functions return the correct vector types:
/// hash22 -> vec2<f32>, hash32 -> vec2<f32>, hash33 -> vec3<f32>.
#[test]
fn vector_hash_functions_return_correct_types() {
    let expected: Vec<(&str, &str)> = vec![
        ("hash22", "vec2<f32>"),
        ("hash32", "vec2<f32>"),
        ("hash33", "vec3<f32>"),
    ];

    let fn_re = regex::Regex::new(
        r"(?m)^\s*fn\s+(hash2[23]|hash3[23])\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]*)"
    ).unwrap();

    for cap in fn_re.captures_iter(WGSL_SOURCE) {
        let fn_name = cap.get(1).unwrap().as_str().to_string();
        let ret = cap.get(2).unwrap().as_str().trim();
        if let Some((_, expected_type)) = expected.iter().find(|(name, _)| *name == fn_name) {
            assert_eq!(
                ret, *expected_type,
                "Hash function '{}' must return '{}', found '{}'",
                fn_name, expected_type, ret
            );
        }
    }
}

// =============================================================================
// SECTION 6 -- Parameter type verification
// =============================================================================

/// Each hash function has the correct parameter types.
#[test]
fn hash_functions_have_correct_param_types() {
    let expected_params: Vec<(&str, &str)> = vec![
        ("hash11", "f32"),
        ("hash21", "vec2<f32>"),
        ("hash22", "vec2<f32>"),
        ("hash31", "vec3<f32>"),
        ("hash32", "vec3<f32>"),
        ("hash33", "vec3<f32>"),
        ("hash41", "vec4<f32>"),
    ];

    let fn_re = regex::Regex::new(
        r"(?m)^\s*fn\s+(hash[1-4][1-4])\s*\(\s*p\s*:\s*([a-zA-Z0-9_<>]*)"
    ).unwrap();

    for cap in fn_re.captures_iter(WGSL_SOURCE) {
        let fn_name = cap.get(1).unwrap().as_str().to_string();
        let param_type = cap.get(2).unwrap().as_str().trim();
        if let Some((_, expected_type)) = expected_params.iter().find(|(name, _)| *name == fn_name) {
            assert_eq!(
                param_type, *expected_type,
                "Hash function '{}' must have parameter type '{}', found '{}'",
                fn_name, expected_type, param_type
            );
        }
    }
}

/// Every hash function has exactly one parameter named 'p'.
#[test]
fn every_hash_function_has_single_p_param() {
    let fn_re = regex::Regex::new(
        r"(?m)^\s*fn\s+(hash[1-4][1-4])\s*\(\s*p\s*:"
    ).unwrap();
    let count = fn_re.captures_iter(WGSL_SOURCE).count();
    assert_eq!(count, 7, "Expected all 7 hash functions to have a single 'p' parameter, found {}", count);
}
