//! Smart test runner with advanced execution strategies.
//!
//! Features:
//! 1. Incremental mode - run one test file at a time, clean between
//! 2. Affected-only - use graph edges to find tests for changed files
//! 3. Priority ordering - run tests for recently changed code first
//! 4. Disk budget - batch tests to fit within disk limit
//! 5. Result caching - skip tests if code unchanged since last GREEN

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use super::{run_cargo_test, CargoTestConfig, CargoTestResult, TestOutcome};
use crate::db::HarnessDb;

/// Configuration for smart test execution.
#[derive(Debug, Clone, Default)]
pub struct SmartTestConfig {
    /// Project root directory.
    pub project_root: String,
    /// Cargo package to test.
    pub package: Option<String>,
    /// Run tests incrementally (one file at a time, clean between).
    pub incremental: bool,
    /// Only run tests affected by changed files.
    pub affected_only: bool,
    /// Changed files (for affected-only mode).
    pub changed_files: Vec<String>,
    /// Prioritize tests by file modification time.
    pub priority_order: bool,
    /// Maximum disk usage in bytes (0 = unlimited).
    pub disk_budget_bytes: u64,
    /// Skip tests for unchanged code (use cached results).
    pub use_cache: bool,
}

impl SmartTestConfig {
    pub fn new(project_root: impl Into<String>) -> Self {
        Self {
            project_root: project_root.into(),
            ..Default::default()
        }
    }

    pub fn package(mut self, pkg: impl Into<String>) -> Self {
        self.package = Some(pkg.into());
        self
    }

    pub fn incremental(mut self) -> Self {
        self.incremental = true;
        self
    }

    pub fn affected_only(mut self, changed: Vec<String>) -> Self {
        self.affected_only = true;
        self.changed_files = changed;
        self
    }

    pub fn priority_order(mut self) -> Self {
        self.priority_order = true;
        self
    }

    pub fn disk_budget(mut self, bytes: u64) -> Self {
        self.disk_budget_bytes = bytes;
        self
    }

    pub fn use_cache(mut self) -> Self {
        self.use_cache = true;
        self
    }
}

/// Result of smart test execution.
#[derive(Debug, Clone, Default)]
pub struct SmartTestResult {
    /// Total tests run.
    pub total_run: usize,
    /// Tests passed.
    pub passed: usize,
    /// Tests failed.
    pub failed: usize,
    /// Tests skipped (cached).
    pub skipped_cached: usize,
    /// Tests skipped (not affected).
    pub skipped_unaffected: usize,
    /// Test files processed.
    pub files_processed: Vec<String>,
    /// Individual test results.
    pub tests: Vec<super::TestResult>,
    /// Errors during execution.
    pub errors: Vec<String>,
    /// Disk cleaned between runs (bytes).
    pub disk_reclaimed: u64,
}

/// Run tests with smart execution strategy.
pub fn run_smart_tests(config: &SmartTestConfig, db: &HarnessDb) -> SmartTestResult {
    let mut result = SmartTestResult::default();

    // Get list of test files
    let test_files = discover_test_files(&config.project_root, config.package.as_deref());

    // Filter by affected-only if enabled
    let test_files = if config.affected_only && !config.changed_files.is_empty() {
        filter_affected_tests(db, &test_files, &config.changed_files)
    } else {
        test_files
    };

    // Sort by priority if enabled
    let test_files = if config.priority_order {
        sort_by_priority(&test_files, &config.changed_files)
    } else {
        test_files
    };

    // Filter by cache if enabled
    let (test_files, cached_count) = if config.use_cache {
        filter_cached_tests(db, &test_files)
    } else {
        (test_files, 0)
    };
    result.skipped_cached = cached_count;

    // Run tests
    if config.incremental {
        run_incremental(config, &test_files, &mut result);
    } else if config.disk_budget_bytes > 0 {
        run_with_budget(config, &test_files, config.disk_budget_bytes, &mut result);
    } else {
        run_batch(config, &test_files, &mut result);
    }

    result
}

/// Discover test files in a project.
fn discover_test_files(project_root: &str, package: Option<&str>) -> Vec<String> {
    let mut test_files = Vec::new();

    // Find test directory
    let tests_dir = if let Some(pkg) = package {
        Path::new(project_root).join("crates").join(pkg).join("tests")
    } else {
        Path::new(project_root).join("tests")
    };

    if tests_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&tests_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    if let Some(stem) = path.file_stem() {
                        test_files.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    test_files
}

/// Filter tests to only those affected by changed files.
fn filter_affected_tests(db: &HarnessDb, test_files: &[String], changed_files: &[String]) -> Vec<String> {
    let conn = db.connection();

    // Find nodes in changed files
    let mut affected_nodes: HashSet<String> = HashSet::new();
    for file in changed_files {
        let query = "SELECT node_id FROM code_nodes WHERE file_path LIKE ?1";
        if let Ok(mut stmt) = conn.prepare(query) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![format!("%{}", file)], |row| {
                row.get::<_, String>(0)
            }) {
                for node_id in rows.flatten() {
                    affected_nodes.insert(node_id);
                }
            }
        }
    }

    // Find tests that cover these nodes via edges
    let mut affected_tests: HashSet<String> = HashSet::new();
    for node_id in &affected_nodes {
        let query = "SELECT from_node FROM code_edges WHERE kind = 'tests' AND to_node = ?1";
        if let Ok(mut stmt) = conn.prepare(query) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![node_id], |row| {
                row.get::<_, String>(0)
            }) {
                for test_node in rows.flatten() {
                    // Extract test file name from node_id (path:line:name)
                    if let Some(file_part) = test_node.split(':').next() {
                        if let Some(file_name) = Path::new(file_part).file_stem() {
                            affected_tests.insert(file_name.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    // Filter test files to only affected ones
    test_files
        .iter()
        .filter(|f| affected_tests.contains(*f) || affected_tests.is_empty())
        .cloned()
        .collect()
}

/// Sort test files by priority (recently changed first).
fn sort_by_priority(test_files: &[String], changed_files: &[String]) -> Vec<String> {
    let changed_set: HashSet<&str> = changed_files.iter().map(|s| s.as_str()).collect();

    let mut files: Vec<_> = test_files.to_vec();
    files.sort_by(|a, b| {
        let a_changed = changed_set.iter().any(|c| c.contains(a));
        let b_changed = changed_set.iter().any(|c| c.contains(b));
        b_changed.cmp(&a_changed) // Changed files first
    });
    files
}

/// Filter out tests where all covered code is still GREEN.
fn filter_cached_tests(db: &HarnessDb, test_files: &[String]) -> (Vec<String>, usize) {
    let conn = db.connection();
    let mut to_run = Vec::new();
    let mut cached = 0;

    for test_file in test_files {
        // Check if any node tested by this file is not GREEN
        let query = r#"
            SELECT COUNT(*) FROM code_nodes n
            JOIN code_edges e ON e.to_node = n.node_id
            WHERE e.kind = 'tests'
            AND e.from_node LIKE ?1
            AND n.current_state != 'tested_green'
        "#;

        let needs_run = conn
            .query_row(query, rusqlite::params![format!("%{}%", test_file)], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(1) > 0;

        if needs_run {
            to_run.push(test_file.clone());
        } else {
            cached += 1;
        }
    }

    (to_run, cached)
}

/// Run tests one file at a time, cleaning between.
fn run_incremental(config: &SmartTestConfig, test_files: &[String], result: &mut SmartTestResult) {
    for test_file in test_files {
        // Run single test file
        let mut cargo_config = CargoTestConfig::new(&config.project_root);
        if let Some(ref pkg) = config.package {
            cargo_config = cargo_config.package(pkg);
        }
        cargo_config = cargo_config.test(test_file);

        match run_cargo_test(&cargo_config) {
            Ok(test_result) => {
                result.total_run += test_result.total;
                result.passed += test_result.passed;
                result.failed += test_result.failed;
                result.tests.extend(test_result.tests);
                result.files_processed.push(test_file.clone());
            }
            Err(e) => {
                result.errors.push(format!("{}: {}", test_file, e));
            }
        }

        // Clean intermediate artifacts
        let cleaned = clean_test_artifacts(&config.project_root, config.package.as_deref());
        result.disk_reclaimed += cleaned;
    }
}

/// Run tests in batches to fit within disk budget.
fn run_with_budget(
    config: &SmartTestConfig,
    test_files: &[String],
    budget_bytes: u64,
    result: &mut SmartTestResult,
) {
    let estimated_per_test = 100 * 1024 * 1024; // ~100MB per test binary
    let batch_size = std::cmp::max(1, (budget_bytes / estimated_per_test) as usize);

    for chunk in test_files.chunks(batch_size) {
        for test_file in chunk {
            let mut cargo_config = CargoTestConfig::new(&config.project_root);
            if let Some(ref pkg) = config.package {
                cargo_config = cargo_config.package(pkg);
            }
            cargo_config = cargo_config.test(test_file);

            match run_cargo_test(&cargo_config) {
                Ok(test_result) => {
                    result.total_run += test_result.total;
                    result.passed += test_result.passed;
                    result.failed += test_result.failed;
                    result.tests.extend(test_result.tests);
                    result.files_processed.push(test_file.clone());
                }
                Err(e) => {
                    result.errors.push(format!("{}: {}", test_file, e));
                }
            }
        }

        // Clean after each batch
        let cleaned = clean_test_artifacts(&config.project_root, config.package.as_deref());
        result.disk_reclaimed += cleaned;
    }
}

/// Run all tests in a single batch.
fn run_batch(config: &SmartTestConfig, test_files: &[String], result: &mut SmartTestResult) {
    for test_file in test_files {
        let mut cargo_config = CargoTestConfig::new(&config.project_root);
        if let Some(ref pkg) = config.package {
            cargo_config = cargo_config.package(pkg);
        }
        cargo_config = cargo_config.test(test_file);

        match run_cargo_test(&cargo_config) {
            Ok(test_result) => {
                result.total_run += test_result.total;
                result.passed += test_result.passed;
                result.failed += test_result.failed;
                result.tests.extend(test_result.tests);
                result.files_processed.push(test_file.clone());
            }
            Err(e) => {
                result.errors.push(format!("{}: {}", test_file, e));
            }
        }
    }
}

/// Clean test artifacts to reclaim disk space.
fn clean_test_artifacts(project_root: &str, package: Option<&str>) -> u64 {
    let mut cmd = Command::new("cargo");
    cmd.arg("clean");

    if let Some(pkg) = package {
        cmd.arg("-p").arg(pkg);
    }

    cmd.current_dir(project_root);

    // Estimate cleaned space (rough)
    let before = get_target_size(project_root);
    let _ = cmd.output();
    let after = get_target_size(project_root);

    before.saturating_sub(after)
}

/// Get approximate size of target directory.
fn get_target_size(project_root: &str) -> u64 {
    let target_dir = Path::new(project_root).join("target");
    if !target_dir.exists() {
        return 0;
    }

    // Quick estimate using du
    let output = Command::new("du")
        .args(["-sb", target_dir.to_str().unwrap_or("target")])
        .output();

    if let Ok(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(size_str) = stdout.split_whitespace().next() {
            return size_str.parse().unwrap_or(0);
        }
    }

    0
}

/// Get changed files from git.
pub fn get_changed_files(project_root: &str) -> Vec<String> {
    let output = Command::new("git")
        .args(["diff", "--name-only", "HEAD"])
        .current_dir(project_root)
        .output();

    if let Ok(output) = output {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    }
}

/// Get available disk space in bytes.
pub fn get_available_disk_space(path: &str) -> u64 {
    let output = Command::new("df")
        .args(["-B1", "--output=avail", path])
        .output();

    if let Ok(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Skip header line
        if let Some(line) = stdout.lines().nth(1) {
            return line.trim().parse().unwrap_or(0);
        }
    }

    0
}
