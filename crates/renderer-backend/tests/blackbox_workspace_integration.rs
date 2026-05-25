// SPDX-License-Identifier: MIT
//
// blackbox_workspace_integration.rs -- Workspace-level blackbox integration tests.
//
// These tests validate the workspace and crate configuration from the outside:
//   - The workspace Cargo.toml is well-formed
//   - Both omega and renderer-backend are declared workspace members
//   - The renderer-backend crate compiles within the workspace
//   - All required dependencies are declared in renderer-backend's Cargo.toml
//   - lib.rs exports the expected public module surface

use std::fs;
use std::process::Command;

// =============================================================================
// Test helpers
// =============================================================================

/// Locate workspace root by searching upward for the root Cargo.toml that
/// contains `[workspace]`.
fn find_workspace_root() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // CARGO_MANIFEST_DIR for integration tests in renderer-backend/tests is:
    //   .../crates/renderer-backend
    // Workspace root is two levels up.
    let candidate = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("renderer-backend should be two levels below workspace root");
    // Verify it contains a Cargo.toml with [workspace]
    let cargo_toml = candidate.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|_| panic!("Expected Cargo.toml at {:?}", cargo_toml));
    assert!(
        contents.contains("[workspace]"),
        "Root Cargo.toml must declare a workspace"
    );
    candidate.to_path_buf()
}

/// The absolute path to the renderer-backend Cargo.toml.
fn renderer_backend_cargo_path() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("Cargo.toml")
}

/// The absolute path to the lib.rs.
fn lib_rs_path() -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("src").join("lib.rs")
}

/// Extract the `[dependencies]` section content, up to the next section
/// header (a line starting with `[` that appears at the beginning of a
/// line, NOT a `[` inside an inline table like `features = ["derive"]`).
fn dependencies_section(contents: &str) -> &str {
    let deps_start = contents
        .find("[dependencies]")
        .expect("Cargo.toml missing [dependencies] section");

    let body = &contents[deps_start..];

    // Find the next line that starts with `[` (a section header), ignoring
    // the `[dependencies]` header itself and any `[` inside values.
    let after_header = &body["[dependencies]".len()..];
    // Look for '\n[' which marks a new section.
    if let Some(pos) = after_header.find("\n[") {
        // Include the newline before the next section for readability.
        let end = "[dependencies]".len() + pos;
        &body[..end]
    } else {
        body
    }
}

/// Assert that a dependency name appears in the `[dependencies]` section.
fn assert_dep_exists(contents: &str, dep_name: &str) {
    let deps = dependencies_section(contents);

    // A dependency line starts with the dep name at the beginning of a line
    // (after trimming).  Check for patterns like:
    //   dep_name = "..."
    //   dep_name = { ... }
    //   dep_name\n (multi-line table)
    let found = deps
        .lines()
        .any(|line| line.trim().starts_with(dep_name) || line.trim() == dep_name);

    assert!(
        found,
        "Dependency '{}' not found in [dependencies] section.\nSection contents:\n{}",
        dep_name,
        deps
    );
}

// =============================================================================
// SECTION 1 -- Workspace Cargo.toml
// =============================================================================

/// Root Cargo.toml must exist and contain `[workspace]`.
#[test]
fn root_has_workspace_declaration() {
    let root = find_workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    assert!(
        cargo_toml.exists(),
        "Root Cargo.toml must exist at {:?}",
        cargo_toml
    );
    let contents = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", cargo_toml));
    assert!(
        contents.contains("[workspace]"),
        "Root Cargo.toml must contain [workspace]"
    );
}

/// Root Cargo.toml declares both `omega` and `crates/renderer-backend` as
/// workspace members.
#[test]
fn workspace_declares_required_members() {
    let root = find_workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", cargo_toml));

    assert!(
        contents.contains("omega"),
        "Workspace members must include 'omega'"
    );
    assert!(
        contents.contains("crates/renderer-backend"),
        "Workspace members must include 'crates/renderer-backend'"
    );
}

/// Root workspace Cargo.toml should use resolver = "2".
#[test]
fn workspace_uses_resolver_2() {
    let root = find_workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", cargo_toml));
    assert!(
        contents.contains("resolver = \"2\""),
        "Workspace should use resolver = \"2\""
    );
}

// =============================================================================
// SECTION 2 -- Crate compilation
// =============================================================================

/// `cargo build --workspace` from the workspace root must succeed.
///
/// This is an expensive test that compiles the entire workspace. It is the
/// most authoritative check that all workspace members are well-formed.
#[test]
fn cargo_build_workspace_succeeds() {
    let root = find_workspace_root();
    let output = Command::new("cargo")
        .args(["build", "--workspace"])
        .current_dir(&root)
        .output()
        .expect("Failed to execute cargo build --workspace");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo build --workspace failed:\n{}",
        stderr
    );
}

/// The omega crate compiles individually (as a workspace member).
#[test]
fn omega_crate_compiles() {
    let root = find_workspace_root();
    let output = Command::new("cargo")
        .args(["check", "-p", "omega"])
        .current_dir(&root)
        .output()
        .expect("Failed to execute cargo check for omega");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo check -p omega failed:\n{}",
        stderr
    );
}

/// The renderer-backend crate compiles individually (as a workspace member).
#[test]
fn renderer_backend_crate_compiles() {
    let root = find_workspace_root();
    let output = Command::new("cargo")
        .args(["check", "-p", "renderer-backend"])
        .current_dir(&root)
        .output()
        .expect("Failed to execute cargo check for renderer-backend");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo check -p renderer-backend failed:\n{}",
        stderr
    );
}

// =============================================================================
// SECTION 3 -- renderer-backend Cargo.toml
// =============================================================================

/// renderer-backend Cargo.toml declares all required crate dependencies:
/// wgpu, bytemuck, crossbeam, parking_lot, slotmap, serde.
#[test]
fn renderer_backend_has_all_required_deps() {
    let path = renderer_backend_cargo_path();
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    for dep in &["wgpu", "bytemuck", "crossbeam", "parking_lot", "slotmap", "serde"] {
        assert_dep_exists(&contents, dep);
    }
}

/// renderer-backend Cargo.toml has serde as a dependency used for
/// deserializing FieldLayout across the PyO3 bridge.
#[test]
fn renderer_backend_has_serde_dep() {
    let path = renderer_backend_cargo_path();
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    let deps = dependencies_section(&contents);

    assert!(
        deps.contains("serde"),
        "serde dependency not found in [dependencies] section"
    );
    assert!(
        deps.contains("derive"),
        "serde dependency must have derive feature enabled"
    );
}

/// renderer-backend Cargo.toml uses edition "2021".
#[test]
fn renderer_backend_has_correct_edition() {
    let path = renderer_backend_cargo_path();
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    assert!(
        contents.contains("edition = \"2021\""),
        "Cargo.toml must have edition = \"2021\""
    );
}

// =============================================================================
// SECTION 4 -- lib.rs public module surface
// =============================================================================

/// lib.rs exports exactly 4 public modules: frame_graph, gpu_driven,
/// type_registry, bridge.
#[test]
fn lib_rs_exports_exactly_4_modules() {
    let path = lib_rs_path();
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    // Count lines matching `pub mod <name>;`
    let pub_mod_lines: Vec<&str> = contents
        .lines()
        .filter(|line| line.trim().starts_with("pub mod ") && line.trim().ends_with(';'))
        .collect();

    assert_eq!(
        pub_mod_lines.len(),
        4,
        "Expected exactly 4 `pub mod` declarations in lib.rs, found {}: {:?}",
        pub_mod_lines.len(),
        pub_mod_lines
    );
}

/// lib.rs exports the `frame_graph` module.
#[test]
fn lib_rs_exports_frame_graph() {
    assert_lib_rs_has_module("frame_graph");
}

/// lib.rs exports the `gpu_driven` module.
#[test]
fn lib_rs_exports_gpu_driven() {
    assert_lib_rs_has_module("gpu_driven");
}

/// lib.rs exports the `type_registry` module.
#[test]
fn lib_rs_exports_type_registry() {
    assert_lib_rs_has_module("type_registry");
}

/// lib.rs exports the `bridge` module.
#[test]
fn lib_rs_exports_bridge() {
    assert_lib_rs_has_module("bridge");
}

fn assert_lib_rs_has_module(module: &str) {
    let path = lib_rs_path();
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));

    let expected = format!("pub mod {};", module);
    assert!(
        contents.contains(&expected),
        "lib.rs must contain '{}' for the {} module",
        expected,
        module
    );
}
