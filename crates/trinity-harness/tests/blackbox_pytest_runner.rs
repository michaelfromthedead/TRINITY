//! Blackbox tests for pytest runner.

use trinity_harness::runners::{run_pytest, PytestConfig};

#[test]
fn test_pytest_config() {
    let config = PytestConfig::new("/nonexistent")
        .timeout(60)
        .path("tests/unit");

    assert_eq!(config.working_dir, "/nonexistent");
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.test_path, Some("tests/unit".to_string()));
}

#[test]
fn test_pytest_invalid_dir() {
    let config = PytestConfig::new("/nonexistent/path/to/tests");
    
    // Should fail gracefully
    let result = run_pytest(&config);
    
    match result {
        Ok(r) => {
            assert_eq!(r.total, 0, "No tests in nonexistent dir");
        }
        Err(_) => {
            // Expected - pytest fails in nonexistent dir
        }
    }
}

#[test]
fn test_config_builder_chain() {
    let config = PytestConfig::new(".")
        .path("tests/")
        .test("test_specific")
        .timeout(120);
    
    assert_eq!(config.test_path, Some("tests/".to_string()));
    assert_eq!(config.test_name, Some("test_specific".to_string()));
    assert_eq!(config.timeout_secs, 120);
}

#[test]
fn test_run_pytest_on_project() {
    // Test on the actual project's tests directory
    let config = PytestConfig::new(".")
        .path("tests/")
        .test("NONEXISTENT_TEST_NAME_12345"); // Run nothing

    let result = run_pytest(&config);

    match result {
        Ok(r) => {
            // With a nonexistent filter, should have 0 tests
            // or pytest might not be available
            assert!(r.total == 0 || r.success);
        }
        Err(e) => {
            // OK if pytest not available
            eprintln!("Pytest skipped: {}", e);
        }
    }
}
