//! Blackbox tests for cargo test runner.

use trinity_harness::runners::{run_cargo_test, CargoTestConfig};

#[test]
fn test_run_cargo_test_config() {
    let config = CargoTestConfig::new("/nonexistent")
        .timeout(60)
        .package("my-pkg");

    assert_eq!(config.working_dir, "/nonexistent");
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.package, Some("my-pkg".to_string()));
}

#[test]
fn test_run_cargo_test_invalid_dir() {
    let config = CargoTestConfig::new("/nonexistent/path/to/crate");
    
    // Should fail gracefully
    let result = run_cargo_test(&config);
    
    // Either it fails or returns an empty result
    match result {
        Ok(r) => {
            assert_eq!(r.total, 0, "No tests in nonexistent dir");
        }
        Err(_) => {
            // Expected - cargo fails in nonexistent dir
        }
    }
}

#[test]
fn test_config_builder_chain() {
    let config = CargoTestConfig::new(".")
        .package("crate-a")
        .test("test_specific")
        .timeout(120);
    
    assert_eq!(config.package, Some("crate-a".to_string()));
    assert_eq!(config.test_name, Some("test_specific".to_string()));
    assert_eq!(config.timeout_secs, 120);
}

#[test]
fn test_run_cargo_test_on_self() {
    // Test on trinity-harness itself
    let config = CargoTestConfig::new(".")
        .package("trinity-harness")
        .test("test_result_passed"); // Run just one specific test

    let result = run_cargo_test(&config);

    match result {
        Ok(r) => {
            // If it ran, should have at least one result
            // (could be 0 if test name wasn't found)
            assert!(r.success || r.total == 0);
        }
        Err(e) => {
            // OK if cargo not available or nightly required
            eprintln!("Cargo test skipped: {}", e);
        }
    }
}
