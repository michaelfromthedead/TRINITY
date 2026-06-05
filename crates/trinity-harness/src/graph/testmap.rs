//! Test mapping for connecting tests to code nodes.
//!
//! Implements convention-based auto-mapping:
//! - Rust blackbox: `tests/test_<name>.rs` → `src/<name>.rs`
//! - Rust unit: `#[test] fn test_<name>()` → `fn <name>()`
//! - Python: `test_<name>.py` → `<name>.py`

use std::collections::HashMap;

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
