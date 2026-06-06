//! Result mapping from test results to code graph nodes.
//!
//! Maps test execution results to their target code nodes via Tests edges.

use std::collections::HashMap;

use super::TestResult;
use crate::graph::{CodeGraph, EdgeType, NodeId};

/// Aggregated result for a code node.
#[derive(Debug, Clone)]
pub struct NodeResult {
    /// The code node ID.
    pub node_id: NodeId,
    /// Number of tests that passed.
    pub passed: usize,
    /// Number of tests that failed.
    pub failed: usize,
    /// Number of tests that were skipped.
    pub skipped: usize,
    /// Total test count.
    pub total: usize,
    /// Test names that failed.
    pub failed_tests: Vec<String>,
}

impl NodeResult {
    /// Create a new result for a node.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            passed: 0,
            failed: 0,
            skipped: 0,
            total: 0,
            failed_tests: Vec::new(),
        }
    }

    /// Check if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.total > 0
    }

    /// Check if any tests failed.
    pub fn has_failures(&self) -> bool {
        self.failed > 0
    }

    /// Check if node is untested.
    pub fn is_untested(&self) -> bool {
        self.total == 0
    }

    /// Get pass rate as percentage.
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
        }
    }
}

/// Result of mapping test results to code nodes.
#[derive(Debug, Clone, Default)]
pub struct MappingResult {
    /// Results per code node.
    pub by_node: HashMap<NodeId, NodeResult>,
    /// Tests that couldn't be mapped.
    pub unmapped_tests: Vec<String>,
    /// Total tests processed.
    pub total_tests: usize,
    /// Tests successfully mapped.
    pub mapped_tests: usize,
}

impl MappingResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get result for a specific node.
    pub fn get(&self, node_id: NodeId) -> Option<&NodeResult> {
        self.by_node.get(&node_id)
    }

    /// Get all nodes with failures.
    pub fn failed_nodes(&self) -> Vec<NodeId> {
        self.by_node
            .iter()
            .filter(|(_, r)| r.has_failures())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all nodes with all tests passing.
    pub fn passing_nodes(&self) -> Vec<NodeId> {
        self.by_node
            .iter()
            .filter(|(_, r)| r.all_passed())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get summary stats.
    pub fn summary(&self) -> (usize, usize, usize) {
        let passing = self.passing_nodes().len();
        let failing = self.failed_nodes().len();
        let untested = self.by_node.iter().filter(|(_, r)| r.is_untested()).count();
        (passing, failing, untested)
    }
}

/// Map test results to code nodes via the graph.
///
/// Looks up each test by name in the graph, finds its target nodes
/// via Tests edges, and aggregates results per target.
pub fn map_results(
    graph: &CodeGraph,
    test_results: &[TestResult],
) -> MappingResult {
    let mut result = MappingResult::new();
    result.total_tests = test_results.len();

    // Build test name to node ID index
    let mut test_index: HashMap<String, NodeId> = HashMap::new();
    for node in graph.nodes() {
        if node.name().starts_with("test_") || node.name().starts_with("Test") {
            test_index.insert(node.name().to_string(), node.id);
            // Also index by full path
            let full_name = format!("{}::{}", node.file_path, node.name());
            test_index.insert(full_name, node.id);
        }
    }

    // Build edge index: test node -> target nodes
    let mut test_targets: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for edge in graph.edges() {
        if edge.edge_type == EdgeType::Tests {
            test_targets
                .entry(edge.source)
                .or_default()
                .push(edge.target);
        }
    }

    // Map each test result
    for test in test_results {
        // Try to find the test node
        let test_node_id = find_test_node(&test.name, &test_index);

        if let Some(test_id) = test_node_id {
            // Get target nodes
            if let Some(targets) = test_targets.get(&test_id) {
                result.mapped_tests += 1;

                for &target_id in targets {
                    let node_result = result.by_node
                        .entry(target_id)
                        .or_insert_with(|| NodeResult::new(target_id));

                    node_result.total += 1;
                    match test.outcome {
                        super::TestOutcome::Passed => node_result.passed += 1,
                        super::TestOutcome::Failed => {
                            node_result.failed += 1;
                            node_result.failed_tests.push(test.name.clone());
                        }
                        super::TestOutcome::Ignored => node_result.skipped += 1,
                        _ => {}
                    }
                }
            } else {
                // Test exists but has no targets
                result.unmapped_tests.push(test.name.clone());
            }
        } else {
            // Couldn't find test in graph
            result.unmapped_tests.push(test.name.clone());
        }
    }

    result
}

/// Find a test node by name, trying various matching strategies.
fn find_test_node(
    test_name: &str,
    index: &HashMap<String, NodeId>,
) -> Option<NodeId> {
    // Exact match
    if let Some(&id) = index.get(test_name) {
        return Some(id);
    }

    // Extract just the test function name
    // e.g., "tests/test_foo.py::TestClass::test_method" -> "test_method"
    let parts: Vec<&str> = test_name.rsplit("::").collect();
    if let Some(name) = parts.first() {
        if let Some(&id) = index.get(*name) {
            return Some(id);
        }
    }

    // Try without file path
    // e.g., "module::test_name" -> "test_name"
    if test_name.contains("::") {
        let name = test_name.rsplit("::").next().unwrap_or(test_name);
        if let Some(&id) = index.get(name) {
            return Some(id);
        }
    }

    None
}

/// Get targets for a test node.
pub fn get_test_targets(graph: &CodeGraph, test_id: NodeId) -> Vec<NodeId> {
    graph
        .edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests && e.source == test_id)
        .map(|e| e.target)
        .collect()
}

/// Look up a test node by name.
pub fn lookup_test_node(graph: &CodeGraph, test_name: &str) -> Option<NodeId> {
    for node in graph.nodes() {
        if node.name() == test_name {
            return Some(node.id);
        }
        // Check if name ends with the test name
        if test_name.contains("::") {
            let short_name = test_name.rsplit("::").next().unwrap_or(test_name);
            if node.name() == short_name {
                return Some(node.id);
            }
        }
    }
    None
}
