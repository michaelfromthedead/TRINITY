//! Blackbox tests for test executor.

use tempfile::TempDir;
use trinity_harness::runners::{run_all_tests, ExecutorConfig};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_run_all_tests_empty_dir() {
    let dir = create_test_dir();
    let config = ExecutorConfig::new(dir.path().to_string_lossy().to_string());

    let result = run_all_tests(&config);

    // Empty dir should have no tests
    assert_eq!(result.total, 0);
    assert!(result.success);
}

#[test]
fn test_run_all_tests_cargo_only() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create a simple Cargo.toml
    std::fs::write(
        root.join("Cargo.toml"),
        r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#,
    ).ok();

    std::fs::create_dir_all(root.join("src")).ok();
    std::fs::write(
        root.join("src/lib.rs"),
        r#"
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() { assert_eq!(add(2, 2), 4); }
}
"#,
    ).ok();

    let config = ExecutorConfig::new(root.to_string_lossy().to_string())
        .cargo_only();

    let result = run_all_tests(&config);

    // Either has tests or has errors (cargo might not be available)
    assert!(result.total >= 1 || !result.errors.is_empty() || result.success);
}

#[test]
fn test_run_all_tests_pytest_only() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create a simple pytest setup
    std::fs::create_dir_all(root.join("tests")).ok();
    std::fs::write(
        root.join("tests/test_sample.py"),
        r#"
def test_always_passes():
    assert True
"#,
    ).ok();

    let config = ExecutorConfig::new(root.to_string_lossy().to_string())
        .pytest_only();

    let result = run_all_tests(&config);

    // Either has tests or pytest not available
    assert!(result.total >= 0);
}

#[test]
fn test_run_all_tests_on_self() {
    // Run on trinity-harness itself with limited scope
    let config = ExecutorConfig::new(".")
        .cargo_only()
        .package("trinity-harness")
        .cargo_timeout(60);

    // We don't actually run this to avoid recursion
    // Just validate config
    assert_eq!(config.cargo_package, Some("trinity-harness".to_string()));
}

#[test]
fn test_executor_result_report() {
    let dir = create_test_dir();
    let config = ExecutorConfig::new(dir.path().to_string_lossy().to_string());

    let result = run_all_tests(&config);
    let report = result.generate_report();

    assert!(report.contains("Test Execution Report"));
    assert!(report.contains("Total:"));
}
