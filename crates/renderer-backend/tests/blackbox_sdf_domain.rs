// SPDX-License-Identifier: MIT
//
// blackbox_sdf_domain.rs -- Blackbox tests for sdf_domain.wgsl domain deformations.
//
// These tests load the WGSL source as a raw string and verify:
//   - All documented functions exist with correct signatures
//   - Compilation via naga (WebGPU shader language parser)
//   - Landing-zone guards prevent undefined behavior
//   - Section headers match the WGSL source organization
//
// These are CONTRACT tests (not unit tests). They validate the WGSL source
// code invariants at the text level, independent of any specific renderer
// implementation.

use regex::Regex;

// =============================================================================
// Test fixture: load WGSL source
// =============================================================================

static WGSL_SOURCE: &str = include_str!("../src/demoscene/sdf_domain.wgsl");

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
/// We do a coarse scan: non-blank, non-comment lines must end with ';', '{', '}',
/// ')', ',', or start with 'fn ', 'var ', 'let ', 'return '.
/// The ',' case handles multi-line function call arguments (e.g. vec3 literals
/// spanning several lines inside a return statement).
/// Lines that are bare expressions (last argument in a multi-line call without
/// trailing comma) are checked for valid WGSL characters only.
#[test]
fn no_stray_text() {
    let valid_wgsl_chars = |c: char| {
        c.is_alphanumeric()
            || c.is_whitespace()
            || "_+-*/=<>!&|^%~().,;:[]@#".contains(c)
    };
    for (i, line) in WGSL_SOURCE.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        // Doc comments (///) or section markers (//=) are already covered above.
        let matched =
            trimmed.ends_with(';')
                || trimmed.ends_with('{')
                || trimmed.ends_with('}')
                || trimmed.ends_with(')')
                || trimmed.ends_with(',')
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("var ")
                || trimmed.starts_with("let ")
                || trimmed.starts_with("return ");
        if !matched {
            // Bare WGSL expression line (e.g., last arg in multi-line fn call).
            // Full syntax validation is delegated to compiles_via_naga.
            assert!(
                trimmed.chars().all(valid_wgsl_chars),
                "Line {} has unexpected non-comment content with invalid characters: {:?}",
                i + 1,
                trimmed
            );
        }
    }
}

// =============================================================================
// SECTION 2 -- Section headers match WGSL source
// =============================================================================

/// Helper: read a `// ==== ... ====` section header.
fn find_section_headers(source: &str) -> Vec<String> {
    let re = Regex::new(r"(?m)^\s*// =====+$").unwrap();
    let mut headers = Vec::new();
    for m in re.find_iter(source) {
        headers.push(m.as_str().trim().to_string());
    }
    headers
}

/// There are exactly 6 section-defining header pairs (start + end) for
/// T-DEMO-1.22 through T-DEMO-1.27. Each section has an opening and closing
/// `// ====...` delimiter line, for 12 delimiter lines total.
#[test]
fn section_header_count() {
    let headers = find_section_headers(WGSL_SOURCE);
    // 12 = 6 demos x 2 (open + close)
    assert!(
        headers.len() >= 12,
        "Expected at least 12 section header delimiters, found {}",
        headers.len()
    );
}

/// Every T-DEMO-1.2x section header appears in order.
#[test]
fn demo_section_headers_present() {
    let sections = [
        "T-DEMO-1.22: Domain Repetition",
        "T-DEMO-1.23: Domain Mirroring",
        "T-DEMO-1.24: Kaleidoscopic Fold (KIFS)",
        "T-DEMO-1.25: Twist",
        "T-DEMO-1.26: Bend",
        "T-DEMO-1.27: Stretch (Anisotropic Scaling)",
    ];
    for section in &sections {
        assert!(
            WGSL_SOURCE.contains(section),
            "Section header '{}' not found in WGSL source",
            section
        );
    }
}

// =============================================================================
// SECTION 3 -- Domain function signature verification
// =============================================================================

/// Helper: extract the domain function name from source.
fn find_domain_names(source: &str) -> Vec<String> {
    let fn_re =
        Regex::new(r"(?m)^\s*fn\s+(domain_[a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap();
    let mut names: Vec<String> = fn_re
        .captures_iter(source)
        .map(|c| c.get(1).unwrap().as_str().to_string())
        .collect();
    names.sort();
    names
}

/// Helper: find the byte span of a function definition in source.
/// Matches from `fn name(` up to and including the opening `{`.
fn find_function_span<'a>(source: &'a str, name: &str) -> Option<std::ops::Range<usize>> {
    let re = Regex::new(&format!(
        r"(?s)\bfn\s+{}[\s\S]*?\{{",
        regex::escape(name)
    ))
    .unwrap();
    re.find(source).map(|m| m.range())
}

/// Helper: extract the function signature + body as a string.
fn extract_signature_body<'a>(source: &'a str, span: std::ops::Range<usize>) -> &'a str {
    &source[span]
}

/// There is at least one domain function.
#[test]
fn find_at_least_one_domain_function() {
    let names = find_domain_names(WGSL_SOURCE);
    assert!(
        !names.is_empty(),
        "Must find at least one domain_ function"
    );
}

/// All documented core domain functions exist.
#[test]
fn contains_core_domain_functions() {
    let expected: Vec<String> = vec![
        "domain_repeat",
        "domain_cell_id",
        "domain_mirror_x",
        "domain_mirror_y",
        "domain_mirror_z",
        "domain_kifs",
        "domain_kifs_compensation",
        "domain_twist",
        "domain_bend",
        "domain_stretch_x",
        "domain_stretch_y",
        "domain_stretch_z",
        "domain_stretch_compensation",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let names = find_domain_names(WGSL_SOURCE);

    for want in &expected {
        assert!(
            names.contains(want),
            "Expected domain function '{}' not found in source. Found: {:?}",
            want,
            names
        );
    }
}

/// Each domain function has at least a vec3<f32> position parameter.
/// Helper/compensation functions (e.g. domain_stretch_compensation,
/// domain_kifs_compensation) are excluded as they take f32, not vec3<f32>.
#[test]
fn domain_functions_have_vec3_position_param() {
    let names = find_domain_names(WGSL_SOURCE);

    for name in &names {
        // Skip compensation helpers which take f32 (not a position vec3)
        if name.contains("compensation") {
            continue;
        }
        let pos = find_function_span(WGSL_SOURCE, name)
            .unwrap_or_else(|| panic!("Function '{}' not found in source", name));

        let sig = extract_signature_body(WGSL_SOURCE, pos);

        // Extract parameter types from the signature
        let params_part = sig
            .split("->")
            .next()
            .expect("Signature should have a parameter list");
        let paren_open = params_part.find('(').expect("Signature should have '('");
        let paren_close = params_part.rfind(')').expect("Signature should have ')'");
        let params_body = &params_part[paren_open + 1..paren_close];

        // Parse parameter declarations: "name: type, ..."
        let actual_types: Vec<String> = if params_body.trim().is_empty() {
            vec![]
        } else {
            let mut types = Vec::new();
            let mut depth_angle = 0u32;
            let mut start = 0usize;
            for (i, ch) in params_body.char_indices() {
                match ch {
                    '<' => depth_angle += 1,
                    '>' if depth_angle > 0 => depth_angle -= 1,
                    ',' if depth_angle == 0 => {
                        let segment = params_body[start..i].trim();
                        if let Some(_colon) = segment.find(':') {
                            let typ = segment.split(':').last().unwrap().trim();
                            types.push(typ.to_string());
                        }
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            let last = params_body[start..].trim();
            if !last.is_empty() {
                if let Some(_colon) = last.find(':') {
                    let typ = last.split(':').last().unwrap().trim();
                    types.push(typ.to_string());
                }
            }
            types
        };

        // Every non-compensation domain function must have at least one
        // parameter of type vec3<f32> (the position).
        let has_vec3 = actual_types.iter().any(|t| t == "vec3<f32>");
        assert!(
            has_vec3,
            "Domain function '{}' must have at least one 'vec3<f32>' parameter. Found types: {:?}",
            name,
            actual_types
        );
    }
}

// =============================================================================
// SECTION 4 -- Return type verification
// =============================================================================

/// Every domain function returns vec3<f32>. Helper/compensation functions that
/// return f32 (e.g. domain_stretch_compensation, domain_kifs_compensation) are excluded.
#[test]
fn all_domain_functions_return_vec3_f32() {
    let fn_re = regex::Regex::new(
        r"(?m)^\s*fn\s+(domain_[a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*->\s*([a-zA-Z0-9_<>]*)"
    ).unwrap();

    let mut found_any = false;
    for cap in fn_re.captures_iter(WGSL_SOURCE) {
        let fn_name = cap.get(1).unwrap().as_str().trim();
        // Skip compensation helpers which return f32 (not a position vector)
        if fn_name.contains("compensation") {
            continue;
        }
        found_any = true;
        let ret = cap.get(2).unwrap().as_str().trim();
        assert_eq!(
            ret, "vec3<f32>",
            "Domain function must return 'vec3<f32>', found '{}' in function '{}'",
            ret, fn_name
        );
    }
    assert!(found_any, "No domain_ function definitions found");
}

// =============================================================================
// SECTION 5 -- Landing-zone guards
// =============================================================================

/// Domain operations that receive a raw f32 parameter must guard against
/// division by zero. Specifically: domain_bend guards 'r' via max(abs(r), 1e-8),
/// domain_kifs guards 'folds' via max(abs(folds), 1.0),
/// and domain_stretch_* guards 's' via select(s, 1e-8, abs(s) < 1e-8).
#[test]
fn division_by_zero_guards() {
    // domain_bend
    assert!(
        WGSL_SOURCE.contains("safe_r = max(abs(r), 1e-8)"),
        "domain_bend must guard r with max(abs(r), 1e-8)"
    );
    // domain_kifs
    assert!(
        WGSL_SOURCE.contains("safe_folds = max(abs(folds), 1.0)"),
        "domain_kifs must guard folds with max(abs(folds), 1.0)"
    );
    // domain_stretch_x / y / z (three occurrences of the safe_s guard)
    let safe_s_count = WGSL_SOURCE.matches("safe_s = select(s, 1e-8, abs(s) < 1e-8)").count();
    assert!(
        safe_s_count >= 3,
        "Expected at least 3 safe_s guards (one per stretch axis), found {}",
        safe_s_count
    );
    // domain_stretch_compensation
    assert!(
        WGSL_SOURCE.contains("safe_s = select(s, 1e-8, abs(s) < 1e-8)"),
        "domain_stretch_compensation must guard s with select(s, 1e-8, abs(s) < 1e-8)"
    );
}

/// The naga compile test: parse the WGSL source via naga and verify it
/// produces no errors. This catches syntax errors, type mismatches, and
/// unsupported constructs (e.g. swizzle assignment).
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
// SECTION 6 -- Compensation functions
// =============================================================================

/// domain_kifs_compensation returns f32 (not vec3<f32>).
#[test]
fn kifs_compensation_returns_f32() {
    let re = regex::Regex::new(
        r"fn\s+domain_kifs_compensation\s*\([^)]*\)\s*->\s*(f32)"
    ).unwrap();
    assert!(
        re.is_match(WGSL_SOURCE),
        "domain_kifs_compensation must return f32"
    );
}

/// domain_stretch_compensation returns f32 (not vec3<f32>).
#[test]
fn stretch_compensation_returns_f32() {
    let re = regex::Regex::new(
        r"fn\s+domain_stretch_compensation\s*\([^)]*\)\s*->\s*(f32)"
    ).unwrap();
    assert!(
        re.is_match(WGSL_SOURCE),
        "domain_stretch_compensation must return f32"
    );
}

/// domain_stretch_compensation computes min(|s|, 1/|s|).
#[test]
fn stretch_compensation_formula() {
    assert!(
        WGSL_SOURCE.contains("min(abs(safe_s), 1.0 / abs(safe_s))"),
        "domain_stretch_compensation must compute min(|s|, 1/|s|)"
    );
}
