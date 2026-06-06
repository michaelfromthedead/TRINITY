//! Blackbox tests for CLI commands with real projects.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::cli::{cmd_query_needs_testing, cmd_run_stale, CliConfig};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, content).expect("Failed to write file");
}

#[test]
fn test_query_needs_testing_real_project() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
fn helper() -> i32 { 1 }
"#);

    let config = CliConfig::new(root);
    let result = cmd_query_needs_testing(&config);

    assert!(result.success);
    assert!(result.message.contains("Nodes"));
}

#[test]
fn test_query_needs_testing_json() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", "fn test_fn() {}");

    let config = CliConfig::new(root).format(trinity_harness::cli::OutputFormat::Json);
    let result = cmd_query_needs_testing(&config);

    assert!(result.success);
    assert!(result.message.contains("needs_testing") || result.message.contains("\"total\""));
}

#[test]
fn test_run_stale_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    // Empty project - no tests to run
    let config = CliConfig::new(root);
    let result = cmd_run_stale(&config);

    // Should succeed but report no stale tests or scan failure
    assert!(result.message.contains("No stale") || result.message.contains("Scan"));
}

#[test]
fn test_query_with_tests() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn production_code() -> i32 { 42 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_production() { assert_eq!(production_code(), 42); }
"#);

    let config = CliConfig::new(root);
    let result = cmd_query_needs_testing(&config);

    assert!(result.success);
}

#[test]
fn test_cli_config_with_path() {
    let dir = create_test_dir();
    let config = CliConfig::new(dir.path());

    assert_eq!(config.project_root, dir.path());
}
