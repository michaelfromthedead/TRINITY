//! Whitebox tests for test executor.

use std::time::{Duration, Instant};
use trinity_harness::runners::{
    estimate_duration, should_skip_tests, ExecutorConfig, ExecutorResult,
    CargoTestResult, PytestResult, TestResult, TestOutcome,
};

// ==================== ExecutorConfig ====================

#[test]
fn test_config_default() {
    let config = ExecutorConfig::default();
    assert_eq!(config.project_root, ".");
    // Verify timeouts use constants (not magic numbers)
    assert_eq!(config.cargo_timeout_secs, trinity_harness::constants::DEFAULT_CARGO_TIMEOUT_SECS);
    assert_eq!(config.pytest_timeout_secs, trinity_harness::constants::DEFAULT_PYTEST_TIMEOUT_SECS);
    assert!(config.run_cargo);
    assert!(config.run_pytest);
}

#[test]
fn test_config_uses_constants() {
    // Explicitly verify the config defaults match the defined constants
    let config = ExecutorConfig::default();
    assert_eq!(config.cargo_timeout_secs, 600); // DEFAULT_CARGO_TIMEOUT_SECS
    assert_eq!(config.pytest_timeout_secs, 1800); // DEFAULT_PYTEST_TIMEOUT_SECS
}

#[test]
fn test_config_new() {
    let config = ExecutorConfig::new("/project");
    assert_eq!(config.project_root, "/project");
}

#[test]
fn test_config_builder() {
    let config = ExecutorConfig::new(".")
        .cargo_timeout(300)
        .pytest_timeout(600)
        .package("my-crate")
        .pytest_path("tests/unit");

    assert_eq!(config.cargo_timeout_secs, 300);
    assert_eq!(config.pytest_timeout_secs, 600);
    assert_eq!(config.cargo_package, Some("my-crate".to_string()));
    assert_eq!(config.pytest_path, Some("tests/unit".to_string()));
}

#[test]
fn test_config_cargo_only() {
    let config = ExecutorConfig::new(".").cargo_only();
    assert!(config.run_cargo);
    assert!(!config.run_pytest);
}

#[test]
fn test_config_pytest_only() {
    let config = ExecutorConfig::new(".").pytest_only();
    assert!(!config.run_cargo);
    assert!(config.run_pytest);
}

// ==================== ExecutorResult ====================

#[test]
fn test_result_new() {
    let result = ExecutorResult::new();
    assert_eq!(result.total, 0);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
    assert!(result.all_tests.is_empty());
}

#[test]
fn test_result_merge_cargo() {
    let mut result = ExecutorResult::new();
    
    let mut cargo = CargoTestResult::new();
    cargo.total = 10;
    cargo.passed = 8;
    cargo.failed = 1;
    cargo.ignored = 1;
    cargo.total_duration_ms = 1000;

    result.merge_cargo(cargo);

    assert_eq!(result.total, 10);
    assert_eq!(result.passed, 8);
    assert_eq!(result.failed, 1);
    assert_eq!(result.skipped, 1);
    assert!(result.cargo.is_some());
}

#[test]
fn test_result_merge_pytest() {
    let mut result = ExecutorResult::new();
    
    let mut pytest = PytestResult::new();
    pytest.total = 5;
    pytest.passed = 4;
    pytest.failed = 1;
    pytest.skipped = 0;
    pytest.total_duration_ms = 500;

    result.merge_pytest(pytest);

    assert_eq!(result.total, 5);
    assert_eq!(result.passed, 4);
    assert_eq!(result.failed, 1);
    assert!(result.pytest.is_some());
}

#[test]
fn test_result_finalize_success() {
    let mut result = ExecutorResult::new();
    result.passed = 10;
    result.finalize();

    assert!(result.success);
}

#[test]
fn test_result_finalize_failure() {
    let mut result = ExecutorResult::new();
    result.passed = 8;
    result.failed = 2;
    result.finalize();

    assert!(!result.success);
}

#[test]
fn test_result_finalize_with_errors() {
    let mut result = ExecutorResult::new();
    result.passed = 10;
    result.errors.push("Test error".to_string());
    result.finalize();

    assert!(!result.success);
}

#[test]
fn test_result_pass_rate() {
    let mut result = ExecutorResult::new();
    result.total = 10;
    result.passed = 8;

    assert_eq!(result.pass_rate(), 80.0);
}

#[test]
fn test_result_pass_rate_zero() {
    let result = ExecutorResult::new();
    assert_eq!(result.pass_rate(), 0.0);
}

#[test]
fn test_result_failed_tests() {
    let mut result = ExecutorResult::new();
    result.all_tests = vec![
        TestResult::passed("test_a", 10),
        TestResult::failed("test_b", 20, "error"),
        TestResult::passed("test_c", 10),
    ];

    let failed = result.failed_tests();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].name, "test_b");
}

#[test]
fn test_result_generate_report() {
    let mut result = ExecutorResult::new();
    result.total = 10;
    result.passed = 8;
    result.failed = 2;
    result.duration_ms = 1000;
    result.finalize();

    let report = result.generate_report();

    assert!(report.contains("Test Execution Report"));
    assert!(report.contains("10"));
    assert!(report.contains("80.0%"));
    assert!(report.contains("FAILED"));
}

#[test]
fn test_result_generate_report_success() {
    let mut result = ExecutorResult::new();
    result.total = 10;
    result.passed = 10;
    result.finalize();

    let report = result.generate_report();
    assert!(report.contains("PASSED"));
}

// ==================== Helpers ====================

#[test]
fn test_should_skip_tests_none() {
    let should_skip = should_skip_tests(None, Duration::from_secs(60));
    assert!(!should_skip);
}

#[test]
fn test_should_skip_tests_recent() {
    let last_run = Some(Instant::now());
    let should_skip = should_skip_tests(last_run, Duration::from_secs(60));
    assert!(should_skip);
}

#[test]
fn test_estimate_duration() {
    let duration = estimate_duration(100, 50, 10);
    assert_eq!(duration, Duration::from_millis(1500));
}
