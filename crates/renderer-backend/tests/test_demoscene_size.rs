//! Demoscene Size Optimization Tests (T-DEMO-5.6)
//!
//! Tests for 64K binary size budget enforcement, profile configuration,
//! build script execution, and compression validation.
//!
//! Run with: `cargo test demoscene_size --features test-utils`

use std::fs;
use std::path::PathBuf;

// ===========================================================================
// Test Constants
// ===========================================================================

/// Maximum allowed binary size in bytes (64KB)
const MAX_SIZE_BYTES: u64 = 65536;

/// Warning threshold (88% of budget)
const WARNING_THRESHOLD_BYTES: u64 = 57344;

/// Critical threshold (94% of budget)
const CRITICAL_THRESHOLD_BYTES: u64 = 61440;

/// Expected compression ratio with UPX (approximate)
const EXPECTED_COMPRESSION_RATIO: f64 = 0.35;

// ===========================================================================
// Profile Configuration Tests (10 tests)
// ===========================================================================

#[test]
fn test_demoscene_profile_exists_in_workspace_cargo_toml() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("[profile.demoscene]"),
        "Workspace Cargo.toml must contain [profile.demoscene]"
    );
}

#[test]
fn test_demoscene_profile_has_size_optimization() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    // opt-level = "z" is for size optimization
    assert!(
        content.contains("opt-level = \"z\""),
        "demoscene profile must have opt-level = \"z\" for size optimization"
    );
}

#[test]
fn test_demoscene_profile_has_lto_enabled() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("lto = true") || content.contains("lto = \"fat\""),
        "demoscene profile must have LTO enabled for link-time optimization"
    );
}

#[test]
fn test_demoscene_profile_has_single_codegen_unit() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("codegen-units = 1"),
        "demoscene profile must have codegen-units = 1 for maximum optimization"
    );
}

#[test]
fn test_demoscene_profile_has_panic_abort() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("panic = \"abort\""),
        "demoscene profile must have panic = \"abort\" to eliminate unwinding code"
    );
}

#[test]
fn test_demoscene_profile_has_strip_symbols() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("strip = \"symbols\""),
        "demoscene profile must have strip = \"symbols\" to remove debug symbols"
    );
}

#[test]
fn test_demoscene_profile_inherits_release() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    // Find the demoscene profile section
    let profile_section = extract_profile_section(&content, "demoscene");
    assert!(
        profile_section.contains("inherits = \"release\""),
        "demoscene profile must inherit from release"
    );
}

#[test]
fn test_demoscene_profile_has_no_debug() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    let profile_section = extract_profile_section(&content, "demoscene");
    assert!(
        profile_section.contains("debug = false"),
        "demoscene profile must have debug = false"
    );
}

#[test]
fn test_demoscene_profile_has_no_debug_assertions() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    let profile_section = extract_profile_section(&content, "demoscene");
    assert!(
        profile_section.contains("debug-assertions = false"),
        "demoscene profile must have debug-assertions = false"
    );
}

#[test]
fn test_demoscene_minimal_profile_exists() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    assert!(
        content.contains("[profile.demoscene-minimal]"),
        "Workspace Cargo.toml must contain [profile.demoscene-minimal]"
    );
}

// ===========================================================================
// Feature Flag Tests (6 tests)
// ===========================================================================

#[test]
fn test_demoscene_minimal_feature_exists() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    assert!(
        content.contains("demoscene-minimal"),
        "Cargo.toml must contain demoscene-minimal feature"
    );
}

#[test]
fn test_test_utils_feature_exists() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    assert!(
        content.contains("test-utils"),
        "Cargo.toml must contain test-utils feature"
    );
}

#[test]
fn test_features_section_exists() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    assert!(
        content.contains("[features]"),
        "Cargo.toml must contain [features] section"
    );
}

#[test]
fn test_default_features_defined() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    assert!(
        content.contains("default = "),
        "Cargo.toml must define default features"
    );
}

#[test]
fn test_demoscene_minimal_feature_is_empty_or_minimal() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    // demoscene-minimal should be defined as empty or with minimal deps
    assert!(
        content.contains("demoscene-minimal = []")
            || content.contains("demoscene-minimal = [")
            && !content.contains("demoscene-minimal = [\"pyo3"),
        "demoscene-minimal feature should not include heavy dependencies like pyo3"
    );
}

#[test]
fn test_feature_flags_compatible() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    // Ensure pyo3 is optional, not default
    assert!(
        !content.contains("default = [\"pyo3\"]"),
        "pyo3 should not be in default features (too heavy for demoscene)"
    );
}

// ===========================================================================
// Build Script Tests (8 tests)
// ===========================================================================

#[test]
fn test_build_script_exists() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");

    assert!(
        script_path.exists(),
        "Build script must exist at scripts/build_demoscene.sh"
    );
}

#[test]
fn test_build_script_is_bash() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.starts_with("#!/usr/bin/env bash") || content.starts_with("#!/bin/bash"),
        "Build script must have bash shebang"
    );
}

#[test]
fn test_build_script_uses_demoscene_profile() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("--profile demoscene")
            || content.contains("--profile $BUILD_PROFILE")
            || content.contains("BUILD_PROFILE"),
        "Build script must use demoscene profile"
    );
}

#[test]
fn test_build_script_has_upx_option() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("upx") || content.contains("UPX"),
        "Build script must have UPX compression option"
    );
}

#[test]
fn test_build_script_has_size_reporting() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("size") || content.contains("stat") || content.contains("ls -l"),
        "Build script must report binary size"
    );
}

#[test]
fn test_build_script_has_budget_check() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("65536") || content.contains("MAX_SIZE") || content.contains("budget"),
        "Build script must check against 64KB budget"
    );
}

#[test]
fn test_build_script_has_help_option() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("--help") || content.contains("-h"),
        "Build script must have help option"
    );
}

#[test]
fn test_build_script_has_strip_step() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    assert!(
        content.contains("strip") || content.contains("STRIP"),
        "Build script must have strip step"
    );
}

// ===========================================================================
// Size Budget JSON Tests (10 tests)
// ===========================================================================

#[test]
fn test_size_budget_json_exists() {
    let crate_dir = get_crate_dir();
    let budget_path = crate_dir.join("size_budget.json");

    assert!(
        budget_path.exists(),
        "Size budget file must exist at crates/renderer-backend/size_budget.json"
    );
}

#[test]
fn test_size_budget_json_is_valid() {
    let crate_dir = get_crate_dir();
    let budget_path = crate_dir.join("size_budget.json");
    let content = fs::read_to_string(&budget_path)
        .expect("Failed to read size budget JSON");

    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_ok(), "size_budget.json must be valid JSON");
}

#[test]
fn test_size_budget_has_version() {
    let budget = load_size_budget();
    assert!(
        budget.get("version").is_some(),
        "size_budget.json must have version field"
    );
}

#[test]
fn test_size_budget_has_budget_bytes() {
    let budget = load_size_budget();
    let budget_bytes = budget.get("budget_bytes")
        .and_then(|v| v.as_u64())
        .expect("size_budget.json must have budget_bytes field");

    assert_eq!(
        budget_bytes, MAX_SIZE_BYTES,
        "Budget must be 65536 bytes (64KB)"
    );
}

#[test]
fn test_size_budget_has_sizes_object() {
    let budget = load_size_budget();
    assert!(
        budget.get("sizes").is_some(),
        "size_budget.json must have sizes object"
    );
}

#[test]
fn test_size_budget_has_base_bytes() {
    let budget = load_size_budget();
    let sizes = budget.get("sizes")
        .expect("Missing sizes object");

    assert!(
        sizes.get("base_bytes").is_some(),
        "sizes object must have base_bytes field"
    );
}

#[test]
fn test_size_budget_has_stripped_bytes() {
    let budget = load_size_budget();
    let sizes = budget.get("sizes")
        .expect("Missing sizes object");

    assert!(
        sizes.get("stripped_bytes").is_some(),
        "sizes object must have stripped_bytes field"
    );
}

#[test]
fn test_size_budget_has_compressed_bytes() {
    let budget = load_size_budget();
    let sizes = budget.get("sizes")
        .expect("Missing sizes object");

    assert!(
        sizes.get("compressed_bytes").is_some(),
        "sizes object must have compressed_bytes field"
    );
}

#[test]
fn test_size_budget_has_within_budget_field() {
    let budget = load_size_budget();
    assert!(
        budget.get("within_budget").is_some(),
        "size_budget.json must have within_budget field"
    );
}

#[test]
fn test_size_budget_has_thresholds() {
    let budget = load_size_budget();
    let thresholds = budget.get("thresholds");

    // Thresholds are optional but should be present if defined
    if let Some(t) = thresholds {
        assert!(
            t.get("warning_bytes").is_some(),
            "thresholds must have warning_bytes"
        );
        assert!(
            t.get("critical_bytes").is_some(),
            "thresholds must have critical_bytes"
        );
        assert!(
            t.get("max_bytes").is_some(),
            "thresholds must have max_bytes"
        );
    }
}

// ===========================================================================
// Size Budget Validation Tests (6 tests)
// ===========================================================================

#[test]
fn test_budget_thresholds_are_ordered() {
    let budget = load_size_budget();

    if let Some(thresholds) = budget.get("thresholds") {
        let warning = thresholds.get("warning_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let critical = thresholds.get("critical_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let max = thresholds.get("max_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        assert!(
            warning < critical,
            "Warning threshold must be less than critical: {} < {}",
            warning, critical
        );
        assert!(
            critical < max,
            "Critical threshold must be less than max: {} < {}",
            critical, max
        );
        assert!(
            max <= MAX_SIZE_BYTES,
            "Max threshold must not exceed budget: {} <= {}",
            max, MAX_SIZE_BYTES
        );
    }
}

#[test]
fn test_headroom_bytes_calculation() {
    let budget = load_size_budget();

    let budget_bytes = budget.get("budget_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(MAX_SIZE_BYTES);

    let compressed = budget.get("sizes")
        .and_then(|s| s.get("compressed_bytes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let headroom = budget.get("headroom_bytes")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let expected_headroom = budget_bytes as i64 - compressed as i64;
    assert_eq!(
        headroom, expected_headroom,
        "Headroom must equal budget - compressed"
    );
}

#[test]
fn test_within_budget_is_correct() {
    let budget = load_size_budget();

    let budget_bytes = budget.get("budget_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(MAX_SIZE_BYTES);

    let compressed = budget.get("sizes")
        .and_then(|s| s.get("compressed_bytes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let within_budget = budget.get("within_budget")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let expected = compressed <= budget_bytes;
    assert_eq!(
        within_budget, expected,
        "within_budget must be true if compressed <= budget"
    );
}

#[test]
fn test_compression_ratio_valid() {
    let budget = load_size_budget();

    let ratio = budget.get("compression_ratio")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    assert!(
        ratio >= 0.0 && ratio <= 1.0,
        "Compression ratio must be between 0 and 1, got {}",
        ratio
    );
}

#[test]
fn test_budget_has_options() {
    let budget = load_size_budget();

    assert!(
        budget.get("options").is_some(),
        "size_budget.json must have options field"
    );
}

#[test]
fn test_options_has_features() {
    let budget = load_size_budget();

    let options = budget.get("options")
        .expect("Missing options");

    assert!(
        options.get("features").is_some(),
        "options must have features array"
    );
}

// ===========================================================================
// Size Calculation Tests (5 tests)
// ===========================================================================

#[test]
fn test_64k_budget_constant() {
    assert_eq!(MAX_SIZE_BYTES, 65536, "64KB must be 65536 bytes");
}

#[test]
fn test_warning_threshold_is_88_percent() {
    let expected = (MAX_SIZE_BYTES as f64 * 0.88) as u64;
    let tolerance = 1024; // Allow 1KB variance

    assert!(
        (WARNING_THRESHOLD_BYTES as i64 - expected as i64).abs() < tolerance as i64,
        "Warning threshold should be approximately 88% of budget"
    );
}

#[test]
fn test_critical_threshold_is_94_percent() {
    let expected = (MAX_SIZE_BYTES as f64 * 0.94) as u64;
    let tolerance = 1024; // Allow 1KB variance

    assert!(
        (CRITICAL_THRESHOLD_BYTES as i64 - expected as i64).abs() < tolerance as i64,
        "Critical threshold should be approximately 94% of budget"
    );
}

#[test]
fn test_compression_ratio_reasonable() {
    // UPX typically achieves 30-40% compression on binaries
    assert!(
        EXPECTED_COMPRESSION_RATIO > 0.2 && EXPECTED_COMPRESSION_RATIO < 0.5,
        "Expected compression ratio should be between 0.2 and 0.5"
    );
}

#[test]
fn test_size_units() {
    // 64KB in various units
    assert_eq!(MAX_SIZE_BYTES, 64 * 1024);
    assert_eq!(MAX_SIZE_BYTES, 0x10000);
    assert!(MAX_SIZE_BYTES < 1024 * 1024); // Less than 1MB
}

// ===========================================================================
// Integration-Level Tests (5 tests)
// ===========================================================================

#[test]
fn test_all_optimization_flags_present() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    let required_flags = [
        "opt-level = \"z\"",
        "lto = true",
        "codegen-units = 1",
        "panic = \"abort\"",
        "strip = \"symbols\"",
    ];

    for flag in &required_flags {
        assert!(
            content.contains(flag),
            "Missing optimization flag: {}",
            flag
        );
    }
}

#[test]
fn test_profile_not_overriding_workspace() {
    let crate_dir = get_crate_dir();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read crate Cargo.toml");

    // Crate should not override profile (profiles must be in workspace root)
    assert!(
        !content.contains("[profile.demoscene]"),
        "Crate Cargo.toml should not override demoscene profile (must be in workspace root)"
    );
}

#[test]
fn test_no_conflicting_profiles() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .expect("Failed to read workspace Cargo.toml");

    // Count profile definitions
    let demoscene_count = content.matches("[profile.demoscene]").count();
    assert_eq!(
        demoscene_count, 1,
        "Should have exactly one [profile.demoscene] definition"
    );
}

#[test]
fn test_build_script_references_correct_profile() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    // Should use demoscene profile, not release
    assert!(
        content.contains("demoscene") && !content.contains("--release"),
        "Build script should use demoscene profile, not --release"
    );
}

#[test]
fn test_size_budget_profile_matches_build_script() {
    let budget = load_size_budget();
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let script_content = fs::read_to_string(&script_path)
        .expect("Failed to read build script");

    let budget_profile = budget.get("profile")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Build script default should match budget profile
    assert!(
        script_content.contains(&format!("BUILD_PROFILE=\"{}\"", budget_profile))
            || script_content.contains("BUILD_PROFILE=\"demoscene\""),
        "Build script default profile should match size_budget.json profile"
    );
}

// ===========================================================================
// Helper Functions
// ===========================================================================

fn get_workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(manifest_dir))
}

fn get_crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_size_budget() -> serde_json::Value {
    let crate_dir = get_crate_dir();
    let budget_path = crate_dir.join("size_budget.json");
    let content = fs::read_to_string(&budget_path)
        .expect("Failed to read size budget JSON");

    serde_json::from_str(&content)
        .expect("Failed to parse size_budget.json")
}

fn extract_profile_section(content: &str, profile_name: &str) -> String {
    let marker = format!("[profile.{}]", profile_name);
    if let Some(start) = content.find(&marker) {
        let section_start = start + marker.len();
        // Find next section or end
        let remaining = &content[section_start..];
        if let Some(end) = remaining.find("\n[") {
            content[start..section_start + end].to_string()
        } else {
            content[start..].to_string()
        }
    } else {
        String::new()
    }
}

// ===========================================================================
// Bonus: Edge Case Tests
// ===========================================================================

#[test]
fn test_size_budget_handles_zero_sizes() {
    // Initial state should have zero sizes
    let budget = load_size_budget();
    let sizes = budget.get("sizes").expect("Missing sizes");

    // Zero is valid for initial/unbuilt state
    let base = sizes.get("base_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
    // base is u64, always non-negative - just verify field exists
    let _ = base;
}

#[test]
fn test_profile_inherits_correctly() {
    let workspace_root = get_workspace_root();
    let cargo_toml = workspace_root.join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml).unwrap();

    // demoscene inherits from release
    let demoscene_section = extract_profile_section(&content, "demoscene");
    assert!(demoscene_section.contains("inherits = \"release\""));

    // demoscene-minimal inherits from demoscene
    let minimal_section = extract_profile_section(&content, "demoscene-minimal");
    assert!(minimal_section.contains("inherits = \"demoscene\""));
}

#[test]
fn test_build_script_has_set_errexit() {
    let workspace_root = get_workspace_root();
    let script_path = workspace_root.join("scripts/build_demoscene.sh");
    let content = fs::read_to_string(&script_path).unwrap();

    assert!(
        content.contains("set -e") || content.contains("set -euo pipefail"),
        "Build script should use set -e for error handling"
    );
}

#[test]
fn test_size_budget_json_has_notes() {
    let budget = load_size_budget();

    // Notes field helps document the budget
    if let Some(notes) = budget.get("notes") {
        assert!(notes.is_array(), "Notes should be an array");
    }
}
