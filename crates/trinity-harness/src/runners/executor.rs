//! Unified test executor for running all project tests.
//!
//! Executes cargo test and pytest, handles timeouts, and aggregates results.

use std::path::Path;
use std::time::{Duration, Instant};

use super::{
    run_cargo_test, run_pytest, CargoTestConfig, CargoTestResult, PytestConfig,
    PytestResult, TestOutcome, TestResult,
};

/// Configuration for running all tests.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Project root directory.
    pub project_root: String,
    /// Timeout for cargo tests in seconds.
    pub cargo_timeout_secs: u64,
    /// Timeout for pytest in seconds.
    pub pytest_timeout_secs: u64,
    /// Whether to run cargo tests.
    pub run_cargo: bool,
    /// Whether to run pytest.
    pub run_pytest: bool,
    /// Specific cargo package to test (optional).
    pub cargo_package: Option<String>,
    /// Specific test name filter (optional).
    pub test_filter: Option<String>,
    /// Specific pytest path (optional).
    pub pytest_path: Option<String>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            project_root: ".".to_string(),
            cargo_timeout_secs: crate::constants::DEFAULT_CARGO_TIMEOUT_SECS,
            pytest_timeout_secs: crate::constants::DEFAULT_PYTEST_TIMEOUT_SECS,
            run_cargo: true,
            run_pytest: true,
            cargo_package: None,
            test_filter: None,
            pytest_path: None,
        }
    }
}

impl ExecutorConfig {
    /// Create a new config for a project.
    pub fn new(project_root: impl Into<String>) -> Self {
        Self {
            project_root: project_root.into(),
            ..Default::default()
        }
    }

    /// Set cargo timeout.
    pub fn cargo_timeout(mut self, secs: u64) -> Self {
        self.cargo_timeout_secs = secs;
        self
    }

    /// Set pytest timeout.
    pub fn pytest_timeout(mut self, secs: u64) -> Self {
        self.pytest_timeout_secs = secs;
        self
    }

    /// Only run cargo tests.
    pub fn cargo_only(mut self) -> Self {
        self.run_cargo = true;
        self.run_pytest = false;
        self
    }

    /// Only run pytest.
    pub fn pytest_only(mut self) -> Self {
        self.run_cargo = false;
        self.run_pytest = true;
        self
    }

    /// Set specific cargo package.
    pub fn package(mut self, pkg: impl Into<String>) -> Self {
        self.cargo_package = Some(pkg.into());
        self
    }

    /// Set pytest path.
    pub fn pytest_path(mut self, path: impl Into<String>) -> Self {
        self.pytest_path = Some(path.into());
        self
    }

    /// Set test name filter (runs only tests matching this pattern).
    pub fn test_filter(mut self, filter: impl Into<String>) -> Self {
        self.test_filter = Some(filter.into());
        self
    }
}

/// Result of running all tests.
#[derive(Debug, Clone, Default)]
pub struct ExecutorResult {
    /// Cargo test results.
    pub cargo: Option<CargoTestResult>,
    /// Pytest results.
    pub pytest: Option<PytestResult>,
    /// Combined test results.
    pub all_tests: Vec<TestResult>,
    /// Total tests run.
    pub total: usize,
    /// Tests passed.
    pub passed: usize,
    /// Tests failed.
    pub failed: usize,
    /// Tests skipped/ignored.
    pub skipped: usize,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Whether all tests passed.
    pub success: bool,
    /// Errors during execution.
    pub errors: Vec<String>,
}

impl ExecutorResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge cargo test results.
    pub fn merge_cargo(&mut self, result: CargoTestResult) {
        self.total += result.total;
        self.passed += result.passed;
        self.failed += result.failed;
        self.skipped += result.ignored;
        self.duration_ms += result.total_duration_ms;
        self.all_tests.extend(result.tests.clone());
        self.cargo = Some(result);
    }

    /// Merge pytest results.
    pub fn merge_pytest(&mut self, result: PytestResult) {
        self.total += result.total;
        self.passed += result.passed;
        self.failed += result.failed;
        self.skipped += result.skipped;
        self.duration_ms += result.total_duration_ms;
        self.all_tests.extend(result.tests.clone());
        self.pytest = Some(result);
    }

    /// Finalize the result.
    pub fn finalize(&mut self) {
        self.success = self.failed == 0 && self.errors.is_empty();
    }

    /// Get pass rate.
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
        }
    }

    /// Get failed tests.
    pub fn failed_tests(&self) -> Vec<&TestResult> {
        self.all_tests
            .iter()
            .filter(|t| t.outcome == TestOutcome::Failed)
            .collect()
    }

    /// Generate a summary report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Test Execution Report ===\n\n");

        if let Some(ref cargo) = self.cargo {
            report.push_str(&format!("Cargo Tests: {}/{} passed\n", cargo.passed, cargo.total));
        }

        if let Some(ref pytest) = self.pytest {
            report.push_str(&format!("Pytest Tests: {}/{} passed\n", pytest.passed, pytest.total));
        }

        report.push_str(&format!("\nTotal: {}/{} passed ({:.1}%)\n", 
            self.passed, self.total, self.pass_rate()));
        report.push_str(&format!("Failed: {}\n", self.failed));
        report.push_str(&format!("Skipped: {}\n", self.skipped));
        report.push_str(&format!("Duration: {:.2}s\n", self.duration_ms as f64 / 1000.0));

        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for err in &self.errors {
                report.push_str(&format!("  - {}\n", err));
            }
        }

        if self.success {
            report.push_str("\nStatus: PASSED ✓\n");
        } else {
            report.push_str("\nStatus: FAILED ✗\n");
        }

        report
    }
}

/// Run all tests in a project.
pub fn run_all_tests(config: &ExecutorConfig) -> ExecutorResult {
    let mut result = ExecutorResult::new();
    let start = Instant::now();

    // Run cargo tests
    if config.run_cargo {
        let has_cargo = Path::new(&config.project_root).join("Cargo.toml").exists();
        if has_cargo {
            let mut cargo_config = CargoTestConfig::new(&config.project_root)
                .timeout(config.cargo_timeout_secs);

            if let Some(ref pkg) = config.cargo_package {
                cargo_config = cargo_config.package(pkg);
            }

            if let Some(ref filter) = config.test_filter {
                cargo_config = cargo_config.test(filter);
            }

            match run_cargo_test(&cargo_config) {
                Ok(cargo_result) => {
                    result.merge_cargo(cargo_result);
                }
                Err(e) => {
                    result.errors.push(format!("Cargo test error: {}", e));
                }
            }
        }
    }

    // Run pytest
    if config.run_pytest {
        let tests_dir = Path::new(&config.project_root).join("tests");
        let has_pytest = tests_dir.exists() || 
            Path::new(&config.project_root).join("pytest.ini").exists() ||
            Path::new(&config.project_root).join("pyproject.toml").exists();

        if has_pytest {
            let mut pytest_config = PytestConfig::new(&config.project_root)
                .timeout(config.pytest_timeout_secs);

            if let Some(ref path) = config.pytest_path {
                pytest_config = pytest_config.path(path);
            }

            match run_pytest(&pytest_config) {
                Ok(pytest_result) => {
                    result.merge_pytest(pytest_result);
                }
                Err(e) => {
                    result.errors.push(format!("Pytest error: {}", e));
                }
            }
        }
    }

    result.duration_ms = start.elapsed().as_millis() as u64;
    result.finalize();

    result
}

/// Check if tests should be skipped due to no changes.
pub fn should_skip_tests(last_run: Option<Instant>, min_interval: Duration) -> bool {
    if let Some(last) = last_run {
        last.elapsed() < min_interval
    } else {
        false
    }
}

/// Estimate test duration based on previous runs.
pub fn estimate_duration(
    cargo_tests: usize,
    pytest_tests: usize,
    avg_ms_per_test: u64,
) -> Duration {
    let total_tests = cargo_tests + pytest_tests;
    Duration::from_millis(total_tests as u64 * avg_ms_per_test)
}
