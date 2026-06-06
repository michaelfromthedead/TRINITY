//! Pytest runner and result parser.
//!
//! Executes `pytest` with JSON report and parses results.

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::{TestOutcome, TestResult};

/// Configuration for running pytest.
#[derive(Debug, Clone)]
pub struct PytestConfig {
    /// Working directory.
    pub working_dir: String,
    /// Test path or pattern (optional).
    pub test_path: Option<String>,
    /// Specific test to run (optional).
    pub test_name: Option<String>,
    /// Maximum timeout in seconds.
    pub timeout_secs: u64,
    /// Additional pytest arguments.
    pub extra_args: Vec<String>,
}

impl Default for PytestConfig {
    fn default() -> Self {
        Self {
            working_dir: ".".to_string(),
            test_path: None,
            test_name: None,
            timeout_secs: 1800, // 30 minutes
            extra_args: vec![],
        }
    }
}

impl PytestConfig {
    /// Create a new config for a specific directory.
    pub fn new(working_dir: impl Into<String>) -> Self {
        Self {
            working_dir: working_dir.into(),
            ..Default::default()
        }
    }

    /// Set the test path.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.test_path = Some(path.into());
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

/// Result of running pytest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PytestResult {
    /// Individual test results.
    pub tests: Vec<TestResult>,
    /// Total tests run.
    pub total: usize,
    /// Tests passed.
    pub passed: usize,
    /// Tests failed.
    pub failed: usize,
    /// Tests skipped.
    pub skipped: usize,
    /// Tests with errors.
    pub errors: usize,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Whether the run was successful.
    pub success: bool,
    /// Raw output if parsing failed.
    pub raw_output: Option<String>,
}

impl PytestResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a test result.
    pub fn add_result(&mut self, result: TestResult) {
        match result.outcome {
            TestOutcome::Passed => self.passed += 1,
            TestOutcome::Failed => self.failed += 1,
            TestOutcome::Ignored => self.skipped += 1,
            _ => {}
        }
        self.total += 1;
        self.total_duration_ms += result.duration_ms;
        self.tests.push(result);
    }

    /// Finalize the result.
    pub fn finalize(&mut self) {
        self.success = self.failed == 0 && self.errors == 0;
    }

    /// Get results by test name.
    pub fn by_name(&self, name: &str) -> Option<&TestResult> {
        self.tests.iter().find(|t| t.name == name || t.name.ends_with(name))
    }
}

/// Error from running pytest.
#[derive(Debug, Clone)]
pub struct PytestError {
    /// Error message.
    pub message: String,
    /// Exit code if available.
    pub exit_code: Option<i32>,
}

impl std::fmt::Display for PytestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PytestError {}

/// Run pytest and parse results.
///
/// Uses `pytest --json-report` for JSON output if available,
/// falling back to standard output parsing.
pub fn run_pytest(config: &PytestConfig) -> Result<PytestResult, PytestError> {
    // Try with json-report first
    let json_result = run_pytest_json(config);
    if let Ok(result) = json_result {
        if !result.tests.is_empty() {
            return Ok(result);
        }
    }

    // Fall back to standard output
    run_pytest_standard(config)
}

/// Run pytest with JSON report.
fn run_pytest_json(config: &PytestConfig) -> Result<PytestResult, PytestError> {
    let report_path = std::env::temp_dir().join("pytest_report.json");

    let mut cmd = Command::new("pytest");
    cmd.arg("--json-report");
    cmd.arg(format!("--json-report-file={}", report_path.display()));
    cmd.arg("-v");

    if let Some(ref path) = config.test_path {
        cmd.arg(path);
    }

    if let Some(ref test) = config.test_name {
        cmd.arg("-k").arg(test);
    }

    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    cmd.current_dir(&config.working_dir);

    let output = cmd.output().map_err(|e| PytestError {
        message: format!("Failed to execute pytest: {}", e),
        exit_code: None,
    })?;

    // Try to parse JSON report
    if report_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&report_path) {
            let _ = std::fs::remove_file(&report_path);
            return parse_json_report(&content);
        }
    }

    // Parse from stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_pytest_output(&stdout)
}

/// Run pytest with standard output.
fn run_pytest_standard(config: &PytestConfig) -> Result<PytestResult, PytestError> {
    let mut cmd = Command::new("pytest");
    cmd.arg("-v");

    if let Some(ref path) = config.test_path {
        cmd.arg(path);
    }

    if let Some(ref test) = config.test_name {
        cmd.arg("-k").arg(test);
    }

    for arg in &config.extra_args {
        cmd.arg(arg);
    }

    cmd.current_dir(&config.working_dir);

    let output = cmd.output().map_err(|e| PytestError {
        message: format!("Failed to execute pytest: {}", e),
        exit_code: None,
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_pytest_output(&stdout)
}

/// Parse pytest JSON report format.
fn parse_json_report(content: &str) -> Result<PytestResult, PytestError> {
    let report: JsonReport = serde_json::from_str(content).map_err(|e| PytestError {
        message: format!("Failed to parse JSON report: {}", e),
        exit_code: None,
    })?;

    let mut result = PytestResult::new();

    for test in report.tests {
        let duration_ms = (test.call.duration * 1000.0) as u64;
        
        let outcome = match test.outcome.as_str() {
            "passed" => TestOutcome::Passed,
            "failed" => TestOutcome::Failed,
            "skipped" => TestOutcome::Ignored,
            "error" => TestOutcome::Failed,
            _ => TestOutcome::Unknown,
        };

        let test_result = TestResult {
            name: test.nodeid,
            outcome,
            duration_ms,
            message: test.call.longrepr,
        };

        result.add_result(test_result);
    }

    result.total_duration_ms = (report.duration * 1000.0) as u64;
    result.finalize();

    Ok(result)
}

/// Parse standard pytest output.
fn parse_pytest_output(stdout: &str) -> Result<PytestResult, PytestError> {
    let mut result = PytestResult::new();

    for line in stdout.lines() {
        let line = line.trim();

        // Match "tests/test_foo.py::test_bar PASSED" or similar
        if line.contains("::") && (line.ends_with("PASSED") || line.ends_with("FAILED") || line.ends_with("SKIPPED")) {
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() == 2 {
                let status = parts[0];
                let name = parts[1].trim();

                let outcome = match status {
                    "PASSED" => TestOutcome::Passed,
                    "FAILED" => TestOutcome::Failed,
                    "SKIPPED" => TestOutcome::Ignored,
                    _ => TestOutcome::Unknown,
                };

                result.add_result(TestResult {
                    name: name.to_string(),
                    outcome,
                    duration_ms: 0,
                    message: None,
                });
            }
        }
    }

    result.finalize();
    Ok(result)
}

// JSON report structures

#[derive(Debug, Deserialize)]
struct JsonReport {
    duration: f64,
    #[serde(default)]
    tests: Vec<JsonTest>,
}

#[derive(Debug, Deserialize)]
struct JsonTest {
    nodeid: String,
    outcome: String,
    #[serde(default)]
    call: JsonCall,
}

#[derive(Debug, Default, Deserialize)]
struct JsonCall {
    #[serde(default)]
    duration: f64,
    longrepr: Option<String>,
}
