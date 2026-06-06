//! Whitebox tests for cargo test runner.

use trinity_harness::runners::{
    CargoTestConfig, CargoTestResult, TestOutcome, TestResult,
};

// ==================== TestResult ====================

#[test]
fn test_result_passed() {
    let result = TestResult::passed("test_name", 100);
    assert_eq!(result.name, "test_name");
    assert_eq!(result.outcome, TestOutcome::Passed);
    assert_eq!(result.duration_ms, 100);
    assert!(result.message.is_none());
}

#[test]
fn test_result_failed() {
    let result = TestResult::failed("test_fail", 200, "assertion failed");
    assert_eq!(result.name, "test_fail");
    assert_eq!(result.outcome, TestOutcome::Failed);
    assert_eq!(result.duration_ms, 200);
    assert_eq!(result.message, Some("assertion failed".to_string()));
}

#[test]
fn test_result_ignored() {
    let result = TestResult::ignored("test_skip");
    assert_eq!(result.name, "test_skip");
    assert_eq!(result.outcome, TestOutcome::Ignored);
    assert_eq!(result.duration_ms, 0);
}

// ==================== CargoTestResult ====================

#[test]
fn test_cargo_result_new() {
    let result = CargoTestResult::new();
    assert_eq!(result.total, 0);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
    assert!(result.tests.is_empty());
}

#[test]
fn test_cargo_result_add_passed() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::passed("test1", 100));

    assert_eq!(result.total, 1);
    assert_eq!(result.passed, 1);
    assert_eq!(result.failed, 0);
    assert_eq!(result.total_duration_ms, 100);
}

#[test]
fn test_cargo_result_add_failed() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::failed("test1", 100, "error"));

    assert_eq!(result.total, 1);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 1);
}

#[test]
fn test_cargo_result_add_ignored() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::ignored("test1"));

    assert_eq!(result.total, 1);
    assert_eq!(result.ignored, 1);
}

#[test]
fn test_cargo_result_finalize() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::passed("test1", 100));
    result.finalize();

    assert!(result.success);
}

#[test]
fn test_cargo_result_finalize_with_failure() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::passed("test1", 100));
    result.add_result(TestResult::failed("test2", 100, "error"));
    result.finalize();

    assert!(!result.success);
}

#[test]
fn test_cargo_result_by_name() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::passed("test_alpha", 100));
    result.add_result(TestResult::passed("test_beta", 200));

    let found = result.by_name("test_alpha");
    assert!(found.is_some());
    assert_eq!(found.unwrap().duration_ms, 100);

    let not_found = result.by_name("test_gamma");
    assert!(not_found.is_none());
}

#[test]
fn test_cargo_result_multiple() {
    let mut result = CargoTestResult::new();
    result.add_result(TestResult::passed("a", 10));
    result.add_result(TestResult::passed("b", 20));
    result.add_result(TestResult::failed("c", 30, "err"));
    result.add_result(TestResult::ignored("d"));
    result.finalize();

    assert_eq!(result.total, 4);
    assert_eq!(result.passed, 2);
    assert_eq!(result.failed, 1);
    assert_eq!(result.ignored, 1);
    assert_eq!(result.total_duration_ms, 60);
    assert!(!result.success);
}

// ==================== CargoTestConfig ====================

#[test]
fn test_config_default() {
    let config = CargoTestConfig::default();
    assert_eq!(config.working_dir, ".");
    assert!(config.package.is_none());
    assert!(config.test_name.is_none());
    assert_eq!(config.timeout_secs, 600);
}

#[test]
fn test_config_new() {
    let config = CargoTestConfig::new("/path/to/crate");
    assert_eq!(config.working_dir, "/path/to/crate");
}

#[test]
fn test_config_builder() {
    let config = CargoTestConfig::new(".")
        .package("my-crate")
        .test("test_foo")
        .timeout(300);

    assert_eq!(config.package, Some("my-crate".to_string()));
    assert_eq!(config.test_name, Some("test_foo".to_string()));
    assert_eq!(config.timeout_secs, 300);
}

// ==================== TestOutcome ====================

#[test]
fn test_outcome_default() {
    let outcome = TestOutcome::default();
    assert_eq!(outcome, TestOutcome::Unknown);
}

#[test]
fn test_outcome_eq() {
    assert_eq!(TestOutcome::Passed, TestOutcome::Passed);
    assert_ne!(TestOutcome::Passed, TestOutcome::Failed);
}
