//! Test mapping for connecting tests to code nodes.
//!
//! Implements convention-based auto-mapping:
//! - Rust blackbox: `tests/test_<name>.rs` → `src/<name>.rs`
//! - Rust unit: `#[test] fn test_<name>()` → `fn <name>()`
//! - Python: `test_<name>.py` → `<name>.py`

use std::collections::{HashMap, HashSet};

use super::{CodeEdge, CodeGraph, EdgeType, NodeId};
use crate::parsers::{Language, UnitType};

/// A mapping from a test to code nodes it tests.
#[derive(Debug, Clone)]
pub struct TestMapping {
    /// The test node.
    pub test_id: NodeId,
    /// The code nodes being tested.
    pub targets: Vec<NodeId>,
    /// How the mapping was determined.
    pub source: MappingSource,
}

/// How a test mapping was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingSource {
    /// Auto-mapped via naming convention.
    Convention,
    /// Explicitly mapped in config file.
    Explicit,
    /// Mapped via inline annotation.
    Annotation,
    /// Unmapped/orphan test.
    Unmapped,
}

/// Statistics from test mapping.
#[derive(Debug, Clone, Default)]
pub struct MappingStats {
    /// Tests processed.
    pub tests_processed: usize,
    /// Tests successfully mapped.
    pub tests_mapped: usize,
    /// Tests left unmapped.
    pub tests_unmapped: usize,
    /// Edges created.
    pub edges_created: usize,
    /// Mappings by source.
    pub by_source: HashMap<MappingSource, usize>,
}

impl MappingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_mapped(&mut self, source: MappingSource) {
        self.tests_processed += 1;
        self.tests_mapped += 1;
        *self.by_source.entry(source).or_insert(0) += 1;
    }

    pub fn record_unmapped(&mut self) {
        self.tests_processed += 1;
        self.tests_unmapped += 1;
        *self.by_source.entry(MappingSource::Unmapped).or_insert(0) += 1;
    }
}

/// An unmapped test with review information.
#[derive(Debug, Clone)]
pub struct UnmappedTest {
    /// The test node ID.
    pub test_id: NodeId,
    /// The test name.
    pub test_name: String,
    /// The file path.
    pub file_path: String,
    /// Suggested targets based on name similarity.
    pub suggestions: Vec<String>,
}

/// Review report for unmapped tests.
#[derive(Debug, Clone, Default)]
pub struct UnmappedReview {
    /// List of unmapped tests.
    pub unmapped: Vec<UnmappedTest>,
    /// Total unmapped count.
    pub total_unmapped: usize,
}

impl UnmappedReview {
    /// Create a new empty review.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any unmapped tests.
    pub fn has_unmapped(&self) -> bool {
        !self.unmapped.is_empty()
    }

    /// Generate a human-readable report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== Unmapped Tests Review ===\n"));
        report.push_str(&format!("Total: {} unmapped tests\n\n", self.total_unmapped));

        if self.unmapped.is_empty() {
            report.push_str("All tests are mapped.\n");
            return report;
        }

        for (i, test) in self.unmapped.iter().enumerate() {
            report.push_str(&format!(
                "{}. {} ({})\n",
                i + 1,
                test.test_name,
                test.file_path
            ));
            if !test.suggestions.is_empty() {
                report.push_str("   Suggestions:\n");
                for suggestion in &test.suggestions {
                    report.push_str(&format!("   - {}\n", suggestion));
                }
            }
            report.push('\n');
        }

        report
    }
}

/// Extract unmapped tests from mapping results.
pub fn extract_unmapped(
    mappings: &[TestMapping],
    graph: &super::CodeGraph,
) -> UnmappedReview {
    let mut review = UnmappedReview::new();

    // Build a name index for suggestions
    let code_names: Vec<&str> = graph
        .nodes()
        .iter()
        .filter(|n| !n.name().starts_with("test_") && !n.name().starts_with("Test"))
        .map(|n| n.name())
        .collect();

    for mapping in mappings {
        if mapping.source != MappingSource::Unmapped {
            continue;
        }

        let Some(node) = graph.nodes().get(mapping.test_id.0) else {
            continue;
        };

        let test_name = node.name().to_string();
        let file_path = node.file_path.clone();

        // Generate suggestions based on name similarity
        let suggestions = suggest_targets(&test_name, &code_names);

        review.unmapped.push(UnmappedTest {
            test_id: mapping.test_id,
            test_name,
            file_path,
            suggestions,
        });
    }

    review.total_unmapped = review.unmapped.len();
    review
}

/// Suggest potential targets based on test name.
fn suggest_targets(test_name: &str, code_names: &[&str]) -> Vec<String> {
    let mut suggestions = Vec::new();

    // Extract base name from test name
    let base_name = test_name
        .strip_prefix("test_")
        .or_else(|| test_name.strip_prefix("Test"))
        .unwrap_or(test_name);

    // Find similar names
    for name in code_names {
        if name.contains(base_name) || base_name.contains(*name) {
            suggestions.push(name.to_string());
        }
    }

    // Limit suggestions
    suggestions.truncate(5);
    suggestions
}

/// Mark specific tests as orphan (acknowledged unmapped).
pub fn mark_as_orphan(
    mappings: &mut [TestMapping],
    test_ids: &[NodeId],
) -> usize {
    let mut count = 0;
    let test_set: std::collections::HashSet<_> = test_ids.iter().collect();

    for mapping in mappings.iter_mut() {
        if test_set.contains(&mapping.test_id) {
            mapping.source = MappingSource::Unmapped;
            mapping.targets.clear();
            count += 1;
        }
    }

    count
}

// ==================== Coverage Report ====================

/// Test coverage report for the codebase.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    /// Total code nodes in the graph.
    pub total_code_nodes: usize,
    /// Code nodes with at least one test.
    pub covered_nodes: usize,
    /// Code nodes with no tests.
    pub uncovered_nodes: usize,
    /// Coverage percentage (0.0 - 100.0).
    pub coverage_percent: f64,
    /// List of uncovered node IDs.
    pub uncovered_ids: Vec<NodeId>,
    /// Coverage by file.
    pub by_file: HashMap<String, FileCoverage>,
}

/// Coverage stats for a single file.
#[derive(Debug, Clone, Default)]
pub struct FileCoverage {
    /// Total code nodes in the file.
    pub total: usize,
    /// Covered code nodes.
    pub covered: usize,
    /// Uncovered code nodes.
    pub uncovered: usize,
    /// Coverage percentage.
    pub percent: f64,
}

impl CoverageReport {
    /// Create a new empty coverage report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any uncovered nodes.
    pub fn has_uncovered(&self) -> bool {
        self.uncovered_nodes > 0
    }

    /// Generate a human-readable coverage summary.
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== Test Coverage Report ===\n\n");
        summary.push_str(&format!(
            "Overall: {:.1}% ({}/{} nodes covered)\n",
            self.coverage_percent, self.covered_nodes, self.total_code_nodes
        ));
        summary.push_str(&format!("Uncovered: {} nodes\n\n", self.uncovered_nodes));

        if !self.by_file.is_empty() {
            summary.push_str("Coverage by file:\n");

            let mut files: Vec<_> = self.by_file.iter().collect();
            files.sort_by(|a, b| a.1.percent.partial_cmp(&b.1.percent).unwrap());

            for (path, cov) in files.iter().take(20) {
                summary.push_str(&format!(
                    "  {}: {:.1}% ({}/{})\n",
                    path, cov.percent, cov.covered, cov.total
                ));
            }

            if files.len() > 20 {
                summary.push_str(&format!("  ... and {} more files\n", files.len() - 20));
            }
        }

        summary
    }
}

/// Generate a coverage report from test mappings.
///
/// This analyzes the graph and mappings to determine which code
/// nodes have tests and which don't.
pub fn generate_coverage_report(
    graph: &super::CodeGraph,
    mappings: &[TestMapping],
) -> CoverageReport {
    let mut report = CoverageReport::new();

    // Build set of covered node IDs
    let covered_set: HashSet<NodeId> = mappings
        .iter()
        .flat_map(|m| m.targets.iter())
        .copied()
        .collect();

    // Find all code nodes (non-test nodes)
    let code_nodes: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|n| !n.name().starts_with("test_") && !n.name().starts_with("Test"))
        .collect();

    report.total_code_nodes = code_nodes.len();

    // Classify each code node
    let mut by_file: HashMap<String, (usize, usize)> = HashMap::new();

    for node in &code_nodes {
        let entry = by_file.entry(node.file_path.clone()).or_insert((0, 0));
        entry.0 += 1; // total

        if covered_set.contains(&node.id) {
            report.covered_nodes += 1;
            entry.1 += 1; // covered
        } else {
            report.uncovered_nodes += 1;
            report.uncovered_ids.push(node.id);
        }
    }

    // Calculate percentages
    if report.total_code_nodes > 0 {
        report.coverage_percent =
            (report.covered_nodes as f64 / report.total_code_nodes as f64) * 100.0;
    }

    // Build file coverage
    for (path, (total, covered)) in by_file {
        let percent = if total > 0 {
            (covered as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        report.by_file.insert(
            path,
            FileCoverage {
                total,
                covered,
                uncovered: total - covered,
                percent,
            },
        );
    }

    report
}

/// Get a list of code nodes with at least one test.
pub fn get_covered_nodes(
    graph: &super::CodeGraph,
    mappings: &[TestMapping],
) -> Vec<NodeId> {
    let covered_set: HashSet<NodeId> = mappings
        .iter()
        .flat_map(|m| m.targets.iter())
        .copied()
        .collect();

    graph
        .nodes()
        .iter()
        .filter(|n| {
            !n.name().starts_with("test_")
                && !n.name().starts_with("Test")
                && covered_set.contains(&n.id)
        })
        .map(|n| n.id)
        .collect()
}

/// Get a list of code nodes with no tests.
pub fn get_uncovered_nodes(
    graph: &super::CodeGraph,
    mappings: &[TestMapping],
) -> Vec<NodeId> {
    let covered_set: HashSet<NodeId> = mappings
        .iter()
        .flat_map(|m| m.targets.iter())
        .copied()
        .collect();

    graph
        .nodes()
        .iter()
        .filter(|n| {
            !n.name().starts_with("test_")
                && !n.name().starts_with("Test")
                && !covered_set.contains(&n.id)
        })
        .map(|n| n.id)
        .collect()
}

/// Auto-mapper using naming conventions.
pub struct ConventionMapper;

impl ConventionMapper {
    pub fn new() -> Self {
        Self
    }

    /// Map tests to code nodes using naming conventions.
    ///
    /// Returns mappings and statistics.
    pub fn map_tests(&self, graph: &CodeGraph) -> (Vec<TestMapping>, MappingStats) {
        let mut mappings = Vec::new();
        let mut stats = MappingStats::new();

        // Build index of code nodes by name
        let mut code_by_name: HashMap<String, Vec<NodeId>> = HashMap::new();
        let mut tests: Vec<NodeId> = Vec::new();

        for node in graph.nodes() {
            let name = node.name().to_string();
            let is_test = self.is_test_node(node);

            if is_test {
                tests.push(node.id);
            } else {
                code_by_name.entry(name).or_default().push(node.id);
            }
        }

        // Map each test
        for test_id in tests {
            let Some(test_node) = graph.nodes().get(test_id.0) else {
                continue;
            };

            let targets = self.find_targets(test_node, &code_by_name);

            if targets.is_empty() {
                stats.record_unmapped();
                mappings.push(TestMapping {
                    test_id,
                    targets: vec![],
                    source: MappingSource::Unmapped,
                });
            } else {
                stats.record_mapped(MappingSource::Convention);
                stats.edges_created += targets.len();
                mappings.push(TestMapping {
                    test_id,
                    targets,
                    source: MappingSource::Convention,
                });
            }
        }

        (mappings, stats)
    }

    /// Check if a node is a test.
    fn is_test_node(&self, node: &super::CodeNode) -> bool {
        let name = node.name();
        let file_path = &node.file_path;

        // Check name patterns
        if name.starts_with("test_") || name.ends_with("_test") {
            return true;
        }

        // Rust-specific prefixes
        if node.language() == Language::Rust {
            if name.starts_with("blackbox_") || name.starts_with("whitebox_") {
                return true;
            }
        }

        // Python test classes
        if node.language() == Language::Python && name.starts_with("Test") {
            return true;
        }

        // Check file path patterns (handle both /tests/ and tests/ at start)
        let in_tests_dir = file_path.contains("/tests/")
            || file_path.starts_with("tests/")
            || file_path.contains("/test_")
            || file_path.starts_with("test_");

        if in_tests_dir && node.unit_type() == UnitType::Function {
            return true;
        }

        false
    }

    /// Find target code nodes for a test using conventions.
    fn find_targets(
        &self,
        test_node: &super::CodeNode,
        code_by_name: &HashMap<String, Vec<NodeId>>,
    ) -> Vec<NodeId> {
        let test_name = test_node.name();
        let mut targets = Vec::new();

        // Extract the target name from test name
        let target_names = self.extract_target_names(test_name, test_node.language());

        for target_name in target_names {
            if let Some(ids) = code_by_name.get(&target_name) {
                targets.extend(ids.iter().copied());
            }
        }

        // Deduplicate
        targets.sort();
        targets.dedup();

        targets
    }

    /// Extract potential target names from a test name.
    fn extract_target_names(&self, test_name: &str, lang: Language) -> Vec<String> {
        let mut names = Vec::new();

        // Pattern: test_<name> → <name>
        if let Some(name) = test_name.strip_prefix("test_") {
            names.push(name.to_string());

            // Also try snake_case variations
            // test_my_function → my_function
            names.push(name.to_string());
        }

        // Pattern: <name>_test → <name>
        if let Some(name) = test_name.strip_suffix("_test") {
            names.push(name.to_string());
        }

        // Pattern: test_<module>_<function> → <function>
        if test_name.starts_with("test_") {
            let parts: Vec<&str> = test_name.strip_prefix("test_").unwrap().split('_').collect();
            if parts.len() >= 2 {
                // Last part might be function name
                names.push(parts.last().unwrap().to_string());
                // Or combined parts
                names.push(parts.join("_"));
            }
        }

        // Rust-specific: blackbox_<name> → <name>
        if lang == Language::Rust {
            if let Some(name) = test_name.strip_prefix("blackbox_") {
                names.push(name.to_string());
            }
            if let Some(name) = test_name.strip_prefix("whitebox_") {
                names.push(name.to_string());
            }
        }

        // Python-specific: TestClass → Class
        if lang == Language::Python && test_name.starts_with("Test") {
            let class_name = test_name.strip_prefix("Test").unwrap();
            names.push(class_name.to_string());
            // Also lowercase
            names.push(class_name.to_lowercase());
        }

        names
    }
}

impl Default for ConventionMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Add "Tests" edge type and create test edges in the graph.
pub fn create_test_edges(graph: &mut CodeGraph, mappings: &[TestMapping]) -> usize {
    let mut count = 0;

    for mapping in mappings {
        if mapping.source == MappingSource::Unmapped {
            continue;
        }

        for target_id in &mapping.targets {
            graph.add_edge(CodeEdge::new(mapping.test_id, *target_id, EdgeType::Tests));
            count += 1;
        }
    }

    count
}

/// Rust-specific test mapper with enhanced conventions.
pub struct RustTestMapper;

impl RustTestMapper {
    pub fn new() -> Self {
        Self
    }

    /// Find blackbox test files in a crate's tests/ directory.
    pub fn find_blackbox_tests(&self, graph: &CodeGraph) -> Vec<NodeId> {
        graph
            .nodes()
            .iter()
            .filter(|n| {
                n.language() == Language::Rust
                    && n.file_path.contains("/tests/")
                    && n.unit_type() == UnitType::Function
                    && (n.name().starts_with("test_") || n.name().starts_with("blackbox_"))
            })
            .map(|n| n.id)
            .collect()
    }

    /// Find unit tests (inline #[test] functions) in source files.
    pub fn find_unit_tests(&self, graph: &CodeGraph) -> Vec<NodeId> {
        graph
            .nodes()
            .iter()
            .filter(|n| {
                n.language() == Language::Rust
                    && !n.file_path.contains("/tests/")
                    && n.unit_type() == UnitType::Function
                    && n.name().starts_with("test_")
            })
            .map(|n| n.id)
            .collect()
    }

    /// Map Rust tests to their targets.
    pub fn map_rust_tests(&self, graph: &CodeGraph) -> (Vec<TestMapping>, MappingStats) {
        let convention = ConventionMapper::new();
        convention.map_tests(graph)
    }
}

impl Default for RustTestMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Python-specific test mapper.
pub struct PythonTestMapper;

impl PythonTestMapper {
    pub fn new() -> Self {
        Self
    }

    /// Find Python test functions.
    pub fn find_python_tests(&self, graph: &CodeGraph) -> Vec<NodeId> {
        graph
            .nodes()
            .iter()
            .filter(|n| {
                n.language() == Language::Python
                    && (n.name().starts_with("test_")
                        || n.name().starts_with("Test")
                        || n.file_path.contains("/test_")
                        || n.file_path.contains("_test.py"))
            })
            .map(|n| n.id)
            .collect()
    }

    /// Map Python tests to their targets.
    pub fn map_python_tests(&self, graph: &CodeGraph) -> (Vec<TestMapping>, MappingStats) {
        let convention = ConventionMapper::new();
        convention.map_tests(graph)
    }
}

impl Default for PythonTestMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Mapper for inline tests (#[test] functions in source files).
///
/// Inline tests are mapped to other code in the same file.
pub struct InlineTestMapper;

impl InlineTestMapper {
    pub fn new() -> Self {
        Self
    }

    /// Find inline tests in source files.
    ///
    /// These are test functions that live alongside production code,
    /// typically in a `#[cfg(test)]` module.
    pub fn find_inline_tests(&self, graph: &CodeGraph) -> Vec<NodeId> {
        graph
            .nodes()
            .iter()
            .filter(|n| {
                // Must be a function starting with test_
                if n.unit_type() != UnitType::Function || !n.name().starts_with("test_") {
                    return false;
                }

                // Must NOT be in a tests/ directory (those are blackbox tests)
                !n.file_path.contains("/tests/")
                    && !n.file_path.starts_with("tests/")
            })
            .map(|n| n.id)
            .collect()
    }

    /// Map inline tests to code in the same file.
    ///
    /// Returns mappings where each inline test maps to all non-test
    /// functions/structs in its containing file.
    pub fn map_inline_tests(&self, graph: &CodeGraph) -> (Vec<TestMapping>, MappingStats) {
        let mut mappings = Vec::new();
        let mut stats = MappingStats::new();

        // Group nodes by file
        let mut nodes_by_file: HashMap<String, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            nodes_by_file
                .entry(node.file_path.clone())
                .or_default()
                .push(node.id);
        }

        // Find inline tests and map to same-file code
        let inline_tests = self.find_inline_tests(graph);

        for test_id in inline_tests {
            let Some(test_node) = graph.nodes().get(test_id.0) else {
                continue;
            };

            let file_path = &test_node.file_path;

            // Get all nodes in the same file
            let Some(file_nodes) = nodes_by_file.get(file_path) else {
                stats.record_unmapped();
                mappings.push(TestMapping {
                    test_id,
                    targets: vec![],
                    source: MappingSource::Unmapped,
                });
                continue;
            };

            // Filter to non-test code in the same file
            let targets: Vec<NodeId> = file_nodes
                .iter()
                .filter(|&&id| {
                    if id == test_id {
                        return false; // Don't map to self
                    }
                    let Some(node) = graph.nodes().get(id.0) else {
                        return false;
                    };
                    // Target non-test functions and structs
                    !node.name().starts_with("test_")
                })
                .copied()
                .collect();

            if targets.is_empty() {
                stats.record_unmapped();
                mappings.push(TestMapping {
                    test_id,
                    targets: vec![],
                    source: MappingSource::Unmapped,
                });
            } else {
                stats.record_mapped(MappingSource::Convention);
                stats.edges_created += targets.len();
                mappings.push(TestMapping {
                    test_id,
                    targets,
                    source: MappingSource::Convention,
                });
            }
        }

        (mappings, stats)
    }
}

impl Default for InlineTestMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Manual Mapping (TOML) ====================

use serde::Deserialize;
use std::path::Path;

/// TOML configuration for explicit test mappings.
///
/// Example `test_mappings.toml`:
/// ```toml
/// [[mappings]]
/// test = "tests/special_test.rs"
/// targets = ["src/special.rs"]
///
/// [[mappings]]
/// test = "tests/integration/*.rs"
/// targets = ["src/core.rs", "src/utils/*.rs"]
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct MappingConfig {
    /// List of explicit mappings.
    #[serde(default)]
    pub mappings: Vec<ExplicitMapping>,
}

/// A single explicit mapping from test to targets.
#[derive(Debug, Clone, Deserialize)]
pub struct ExplicitMapping {
    /// Test file or glob pattern.
    pub test: String,
    /// Target files or glob patterns.
    pub targets: Vec<String>,
}

/// Error type for mapping config operations.
#[derive(Debug)]
pub enum MappingConfigError {
    /// Failed to read the config file.
    IoError(std::io::Error),
    /// Failed to parse TOML.
    ParseError(toml::de::Error),
    /// Invalid glob pattern.
    GlobError(String),
}

impl std::fmt::Display for MappingConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingConfigError::IoError(e) => write!(f, "IO error: {}", e),
            MappingConfigError::ParseError(e) => write!(f, "Parse error: {}", e),
            MappingConfigError::GlobError(e) => write!(f, "Glob error: {}", e),
        }
    }
}

impl std::error::Error for MappingConfigError {}

impl From<std::io::Error> for MappingConfigError {
    fn from(e: std::io::Error) -> Self {
        MappingConfigError::IoError(e)
    }
}

impl From<toml::de::Error> for MappingConfigError {
    fn from(e: toml::de::Error) -> Self {
        MappingConfigError::ParseError(e)
    }
}

impl MappingConfig {
    /// Load mapping config from a TOML file.
    pub fn load(path: &Path) -> Result<Self, MappingConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: MappingConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Parse mapping config from a TOML string.
    pub fn parse(content: &str) -> Result<Self, MappingConfigError> {
        let config: MappingConfig = toml::from_str(content)?;
        Ok(config)
    }

    /// Create an empty config.
    pub fn empty() -> Self {
        Self { mappings: Vec::new() }
    }
}

/// Mapper that uses explicit TOML configuration.
pub struct ExplicitMapper {
    config: MappingConfig,
}

impl ExplicitMapper {
    /// Create a new explicit mapper from config.
    pub fn new(config: MappingConfig) -> Self {
        Self { config }
    }

    /// Load explicit mapper from a TOML file.
    pub fn load(path: &Path) -> Result<Self, MappingConfigError> {
        let config = MappingConfig::load(path)?;
        Ok(Self::new(config))
    }

    /// Apply explicit mappings to the graph.
    ///
    /// Returns mappings and statistics.
    pub fn map_tests(&self, graph: &CodeGraph, root_path: &Path) -> (Vec<TestMapping>, MappingStats) {
        let mut mappings = Vec::new();
        let mut stats = MappingStats::new();

        // Build indexes
        let mut nodes_by_path: HashMap<String, NodeId> = HashMap::new();
        let mut test_nodes: HashMap<String, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let normalized = normalize_path(&node.file_path);
            nodes_by_path.insert(normalized.clone(), node.id);

            // Also track test nodes
            if is_test_path(&node.file_path) || node.name().starts_with("test_") {
                test_nodes.insert(normalized, node.id);
            }
        }

        // Process each explicit mapping
        for explicit in &self.config.mappings {
            let test_pattern = &explicit.test;

            // Find matching test files
            let test_matches = match_glob_pattern(test_pattern, root_path, &test_nodes);

            for test_path in test_matches {
                let Some(test_id) = test_nodes.get(&test_path).copied() else {
                    continue;
                };

                // Find matching target files
                let mut targets = Vec::new();
                for target_pattern in &explicit.targets {
                    let target_matches = match_glob_pattern(target_pattern, root_path, &nodes_by_path);
                    for target_path in target_matches {
                        if let Some(target_id) = nodes_by_path.get(&target_path) {
                            if !targets.contains(target_id) {
                                targets.push(*target_id);
                            }
                        }
                    }
                }

                if targets.is_empty() {
                    stats.record_unmapped();
                    mappings.push(TestMapping {
                        test_id,
                        targets: vec![],
                        source: MappingSource::Unmapped,
                    });
                } else {
                    stats.record_mapped(MappingSource::Explicit);
                    stats.edges_created += targets.len();
                    mappings.push(TestMapping {
                        test_id,
                        targets,
                        source: MappingSource::Explicit,
                    });
                }
            }
        }

        (mappings, stats)
    }
}

/// Normalize a file path for matching.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

/// Check if a path looks like a test file.
fn is_test_path(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("/test_")
        || path.ends_with("_test.rs")
        || path.ends_with("_test.py")
        || path.starts_with("tests/")
        || path.starts_with("test_")
}

/// Match a glob pattern against known paths.
fn match_glob_pattern(
    pattern: &str,
    root_path: &Path,
    known_paths: &HashMap<String, NodeId>,
) -> Vec<String> {
    let mut matches = Vec::new();

    // If pattern contains glob characters, use glob matching
    if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
        let full_pattern = root_path.join(pattern);
        let pattern_str = full_pattern.to_string_lossy();

        if let Ok(paths) = glob::glob(&pattern_str) {
            for entry in paths.flatten() {
                let path_str = entry.to_string_lossy().to_string();
                let normalized = normalize_path(&path_str);

                // Check if this matches a known path
                for known in known_paths.keys() {
                    if known.ends_with(&normalized) || normalized.ends_with(known) {
                        matches.push(known.clone());
                    }
                }
            }
        }
    } else {
        // Exact match
        let normalized = normalize_path(pattern);
        for known in known_paths.keys() {
            if known.ends_with(&normalized) || known == &normalized {
                matches.push(known.clone());
            }
        }
    }

    matches
}

/// Combined mapper that applies both convention and explicit mappings.
pub struct CombinedMapper {
    convention: ConventionMapper,
    explicit: Option<ExplicitMapper>,
}

impl CombinedMapper {
    /// Create a combined mapper with only convention mapping.
    pub fn convention_only() -> Self {
        Self {
            convention: ConventionMapper::new(),
            explicit: None,
        }
    }

    /// Create a combined mapper with both convention and explicit mapping.
    pub fn with_explicit(config: MappingConfig) -> Self {
        Self {
            convention: ConventionMapper::new(),
            explicit: Some(ExplicitMapper::new(config)),
        }
    }

    /// Load explicit mappings from a TOML file.
    pub fn load_explicit(path: &Path) -> Result<Self, MappingConfigError> {
        let config = MappingConfig::load(path)?;
        Ok(Self::with_explicit(config))
    }

    /// Map tests using both convention and explicit mappings.
    ///
    /// Explicit mappings take precedence over convention mappings.
    pub fn map_tests(&self, graph: &CodeGraph, root_path: &Path) -> (Vec<TestMapping>, MappingStats) {
        let mut all_mappings = Vec::new();
        let mut combined_stats = MappingStats::new();
        let mut mapped_tests: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

        // First, apply explicit mappings (higher priority)
        if let Some(explicit) = &self.explicit {
            let (explicit_mappings, explicit_stats) = explicit.map_tests(graph, root_path);

            for mapping in explicit_mappings {
                if mapping.source != MappingSource::Unmapped {
                    mapped_tests.insert(mapping.test_id);
                }
                all_mappings.push(mapping);
            }

            // Add explicit stats
            combined_stats.tests_processed += explicit_stats.tests_processed;
            combined_stats.tests_mapped += explicit_stats.tests_mapped;
            combined_stats.edges_created += explicit_stats.edges_created;
            for (source, count) in explicit_stats.by_source {
                *combined_stats.by_source.entry(source).or_insert(0) += count;
            }
        }

        // Then, apply convention mappings for unmapped tests
        let (convention_mappings, _) = self.convention.map_tests(graph);

        for mapping in convention_mappings {
            // Skip if already mapped explicitly
            if mapped_tests.contains(&mapping.test_id) {
                continue;
            }

            combined_stats.tests_processed += 1;

            if mapping.source == MappingSource::Unmapped {
                combined_stats.tests_unmapped += 1;
                *combined_stats.by_source.entry(MappingSource::Unmapped).or_insert(0) += 1;
            } else {
                combined_stats.tests_mapped += 1;
                combined_stats.edges_created += mapping.targets.len();
                *combined_stats.by_source.entry(mapping.source).or_insert(0) += 1;
            }

            all_mappings.push(mapping);
        }

        (all_mappings, combined_stats)
    }
}
