//! Cargo test runner and result parser.
//!
//! Executes `cargo test` with JSON output and parses results.

use std::process::{Command, Output};

use serde::{Deserialize, Serialize};

/// Configuration for running cargo tests.
#[derive(Debug, Clone)]
pub struct CargoTestConfig {
    /// Working directory (crate root).
    pub working_dir: String,
    /// Specific package to test (optional).
    pub package: Option<String>,
    /// Specific test to run (optional).
    pub test_name: Option<String>,
    /// Maximum timeout in seconds.
    pub timeout_secs: u64,
    /// Additional cargo arguments.
    pub extra_args: Vec<String>,
}

impl Default for CargoTestConfig {
    fn default() -> Self {
        Self {
            working_dir: ".".to_string(),
            package: None,
            test_name: None,
            timeout_secs: crate::constants::DEFAULT_CARGO_TIMEOUT_SECS,
            extra_args: vec![],
        }
    }
}

impl CargoTestConfig {
    /// Create a new config for a specific directory.
    pub fn new(working_dir: impl Into<String>) -> Self {
        Self {
            working_dir: working_dir.into(),
            ..Default::default()
        }
    }

    /// Set the package to test.
    pub fn package(mut self, pkg: impl Into<String>) -> Self {
        self.package = Some(pkg.into());
        self
    }

    /// Set a specific test to run.
    pub fn test(mut self, name: impl Into<String>) -> Self {
        self.test_name = Some(name.into());
        self
    }

    /// Set timeout in seconds.
    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// Outcome of a single test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    /// Test passed.
    Passed,
    /// Test failed.
    Failed,
    /// Test was ignored.
    Ignored,
    /// Test timed out.
    Timeout,
    /// Test outcome unknown.
    Unknown,
}

impl Default for TestOutcome {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Result of a single test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Full test name (e.g., "module::test_name").
    pub name: String,
    /// Test outcome.
    pub outcome: TestOutcome,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Failure message if failed.
    pub message: Option<String>,
}

impl TestResult {
    /// Create a new passed test result.
    pub fn passed(name: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            name: name.into(),
            outcome: TestOutcome::Passed,
            duration_ms,
            message: None,
        }
    }

    /// Create a new failed test result.
    pub fn failed(name: impl Into<String>, duration_ms: u64, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            outcome: TestOutcome::Failed,
            duration_ms,
            message: Some(message.into()),
        }
    }

    /// Create an ignored test result.
    pub fn ignored(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            outcome: TestOutcome::Ignored,
            duration_ms: 0,
            message: None,
        }
    }
}

/// Result of running cargo test.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CargoTestResult {
    /// Individual test results.
    pub tests: Vec<TestResult>,
    /// Total tests run.
    pub total: usize,
    /// Tests passed.
    pub passed: usize,
    /// Tests failed.
    pub failed: usize,
    /// Tests ignored.
    pub ignored: usize,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Whether the run was successful.
    pub success: bool,
    /// Raw output if parsing failed.
    pub raw_output: Option<String>,
}

impl CargoTestResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a test result.
    pub fn add_result(&mut self, result: TestResult) {
        match result.outcome {
            TestOutcome::Passed => self.passed += 1,
            TestOutcome::Failed => self.failed += 1,
            TestOutcome::Ignored => self.ignored += 1,
            _ => {}
        }
        self.total += 1;
        self.total_duration_ms += result.duration_ms;
        self.tests.push(result);
    }

    /// Finalize the result.
    pub fn finalize(&mut self) {
        self.success = self.failed == 0;
    }

    /// Get results by test name.
    pub fn by_name(&self, name: &str) -> Option<&TestResult> {
        self.tests.iter().find(|t| t.name == name)
    }
}

/// Error from running cargo test.
#[derive(Debug, Clone)]
pub struct CargoTestError {
    /// Error message.
    pub message: String,
    /// Exit code if available.
    pub exit_code: Option<i32>,
}

impl std::fmt::Display for CargoTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CargoTestError {}

/// Run cargo test and parse results.
///
/// Uses `cargo test -- -Z unstable-options --format json` for JSON output,
/// falling back to standard output parsing if JSON is unavailable.
pub fn run_cargo_test(config: &CargoTestConfig) -> Result<CargoTestResult, CargoTestError> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test");

    if let Some(ref pkg) = config.package {
        cmd.arg("-p").arg(pkg);
    }

    // Add JSON format args
    cmd.arg("--").arg("--format").arg("json").arg("-Z").arg("unstable-options");

    if let Some(ref test) = config.test_name {
        cmd.arg(test);
    }

    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    cmd.current_dir(&config.working_dir);

    // Run the command
    let output = cmd.output().map_err(|e| CargoTestError {
        message: format!("Failed to execute cargo test: {}", e),
        exit_code: None,
    })?;

    parse_cargo_output(&output)
}

/// Parse cargo test output.
fn parse_cargo_output(output: &Output) -> Result<CargoTestResult, CargoTestError> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut result = CargoTestResult::new();

    // Try to parse JSON output
    for line in stdout.lines() {
        if let Ok(event) = serde_json::from_str::<CargoTestEvent>(line) {
            match event {
                CargoTestEvent::Test { event: test_event } => match test_event {
                    CargoTestEventType::Ok { name, exec_time } => {
                        let duration_ms = parse_duration(&exec_time);
                        result.add_result(TestResult::passed(name, duration_ms));
                    }
                    CargoTestEventType::Failed { name, exec_time, stdout: msg } => {
                        let duration_ms = parse_duration(&exec_time);
                        let message = msg.unwrap_or_default();
                        result.add_result(TestResult::failed(name, duration_ms, message));
                    }
                    CargoTestEventType::Ignored { name } => {
                        result.add_result(TestResult::ignored(name));
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // If no JSON results, try to parse standard output
    if result.tests.is_empty() {
        result = parse_standard_output(&stdout, &stderr);
    }

    result.finalize();

    if !output.status.success() && result.failed == 0 {
        // Command failed but no test failures detected
        result.raw_output = Some(format!("{}\n{}", stdout, stderr));
    }

    Ok(result)
}

/// Parse standard cargo test output (non-JSON).
fn parse_standard_output(stdout: &str, _stderr: &str) -> CargoTestResult {
    let mut result = CargoTestResult::new();

    for line in stdout.lines() {
        let line = line.trim();

        // Match "test module::name ... ok"
        if line.starts_with("test ") && line.contains(" ... ") {
            let parts: Vec<&str> = line.splitn(2, " ... ").collect();
            if parts.len() == 2 {
                let name = parts[0].strip_prefix("test ").unwrap_or(parts[0]).trim();
                let status = parts[1].trim();

                match status {
                    "ok" => result.add_result(TestResult::passed(name, 0)),
                    "FAILED" => result.add_result(TestResult::failed(name, 0, "Test failed")),
                    "ignored" => result.add_result(TestResult::ignored(name)),
                    _ => {}
                }
            }
        }
    }

    result
}

/// Parse duration string like "0.001s".
fn parse_duration(s: &str) -> u64 {
    let s = s.trim_end_matches('s');
    if let Ok(secs) = s.parse::<f64>() {
        (secs * 1000.0) as u64
    } else {
        0
    }
}

// Cargo test JSON event types

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CargoTestEvent {
    Suite { event: SuiteEvent },
    Test { event: CargoTestEventType },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SuiteEvent {
    Started { test_count: usize },
    Ok { passed: usize, failed: usize, ignored: usize },
    Failed { passed: usize, failed: usize, ignored: usize },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum CargoTestEventType {
    Started { name: String },
    Ok {
        name: String,
        #[serde(default)]
        exec_time: String,
    },
    Failed {
        name: String,
        #[serde(default)]
        exec_time: String,
        stdout: Option<String>,
    },
    Ignored { name: String },
    #[serde(other)]
    Other,
}
