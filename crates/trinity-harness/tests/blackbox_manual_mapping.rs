//! Blackbox tests for manual TOML-based test mapping integration.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{
    create_test_edges, CombinedMapper, EdgeType, GraphBuilder, MappingConfig, MappingSource,
};
use trinity_harness::parsers::ParserRegistry;

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, content).expect("Failed to write file");
}

#[test]
fn test_load_mapping_config() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/test_parser.rs"
targets = ["src/parser.rs"]

[[mappings]]
test = "tests/test_utils.rs"
targets = ["src/utils.rs", "src/helpers.rs"]
"#);

    let config = MappingConfig::load(&root.join("test_mappings.toml")).expect("Load failed");

    assert_eq!(config.mappings.len(), 2);
    assert_eq!(config.mappings[0].targets.len(), 1);
    assert_eq!(config.mappings[1].targets.len(), 2);
}

#[test]
fn test_explicit_mapping_integration() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create source files
    write_file(root, "src/special.rs", r#"
fn special_function() -> i32 {
    42
}
"#);

    // Create test files
    write_file(root, "tests/integration_test.rs", r#"
fn test_special() {
    assert_eq!(special_function(), 42);
}
"#);

    // Create mapping config
    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/integration_test.rs"
targets = ["src/special.rs"]
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = MappingConfig::load(&root.join("test_mappings.toml")).expect("Load failed");
    let mapper = CombinedMapper::with_explicit(config);
    let (mappings, stats) = mapper.map_tests(&graph, root);

    assert!(stats.tests_mapped >= 1);
    assert!(stats.by_source.get(&MappingSource::Explicit).copied().unwrap_or(0) >= 1);
}

#[test]
fn test_combined_mapping_integration() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create source files
    write_file(root, "src/explicit_target.rs", r#"
fn explicit_fn() -> i32 {
    1
}
"#);

    write_file(root, "src/convention_target.rs", r#"
fn convention_target() -> i32 {
    2
}
"#);

    // Create test files
    write_file(root, "tests/test_explicit.rs", r#"
fn test_explicit() {
    assert_eq!(explicit_fn(), 1);
}
"#);

    write_file(root, "tests/test_convention_target.rs", r#"
fn test_convention_target() {
    assert_eq!(convention_target(), 2);
}
"#);

    // Explicit mapping for test_explicit -> explicit_target
    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/test_explicit.rs"
targets = ["src/explicit_target.rs"]
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = MappingConfig::load(&root.join("test_mappings.toml")).expect("Load failed");
    let mapper = CombinedMapper::with_explicit(config);
    let (_, stats) = mapper.map_tests(&graph, root);

    // Should have both explicit and convention mappings
    assert!(stats.tests_mapped >= 2);
    assert!(stats.by_source.get(&MappingSource::Explicit).copied().unwrap_or(0) >= 1);
    assert!(stats.by_source.get(&MappingSource::Convention).copied().unwrap_or(0) >= 1);
}

#[test]
fn test_create_edges_with_explicit_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/core.rs", r#"
fn core_function() -> i32 {
    42
}
"#);

    write_file(root, "tests/core_test.rs", r#"
fn test_core() {
    assert_eq!(core_function(), 42);
}
"#);

    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/core_test.rs"
targets = ["src/core.rs"]
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    let config = MappingConfig::load(&root.join("test_mappings.toml")).expect("Load failed");
    let mapper = CombinedMapper::with_explicit(config);
    let (mappings, _) = mapper.map_tests(&graph, root);

    let edge_count = create_test_edges(&mut graph, &mappings);

    assert!(edge_count >= 1);

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty());
}

#[test]
fn test_missing_config_file() {
    let dir = create_test_dir();
    let root = dir.path();

    let result = MappingConfig::load(&root.join("nonexistent.toml"));
    assert!(result.is_err());
}

#[test]
fn test_invalid_toml_file() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "invalid.toml", "this is not { valid toml [");

    let result = MappingConfig::load(&root.join("invalid.toml"));
    assert!(result.is_err());
}

#[test]
fn test_full_pipeline_with_explicit_mappings() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create a complete project structure
    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    42
}

fn process(x: i32) -> i32 {
    x * 2
}
"#);

    write_file(root, "tests/unit_tests.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}

fn test_process() {
    assert_eq!(process(21), 42);
}
"#);

    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/unit_tests.rs"
targets = ["src/lib.rs"]
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full scan
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // Dependency analysis
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Test mapping with explicit config
    let config = MappingConfig::load(&root.join("test_mappings.toml")).expect("Load failed");
    let mapper = CombinedMapper::with_explicit(config);
    let (mappings, stats) = mapper.map_tests(&graph, root);

    let edge_count = create_test_edges(&mut graph, &mappings);

    // Validate
    let result = graph.validate();

    assert!(result.total_nodes >= 2);
    assert!(edge_count >= 1);
    assert!(stats.tests_mapped >= 1);
}

#[test]
fn test_convention_only_fallback() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/helper.rs", r#"
fn helper() -> i32 {
    42
}
"#);

    write_file(root, "tests/test_helper.rs", r#"
fn test_helper() {
    assert_eq!(helper(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    // Use convention-only mapper
    let mapper = CombinedMapper::convention_only();
    let (_, stats) = mapper.map_tests(&graph, root);

    assert!(stats.tests_mapped >= 1);
    assert!(stats.by_source.get(&MappingSource::Convention).copied().unwrap_or(0) >= 1);
}
