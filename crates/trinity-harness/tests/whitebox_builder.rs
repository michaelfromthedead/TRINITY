//! Whitebox tests for trinity-harness graph builder module.
//!
//! WHITEBOX coverage plan:
//!   - GraphBuilder::new() construction with ParserRegistry
//!   - full_scan() on empty directory returns empty graph
//!   - full_scan() on directory with mixed files filters correctly
//!   - Extension filtering (.rs, .py, .wgsl only)
//!   - ScanStats correctness (files_scanned, files_skipped, nodes_per_language)
//!   - Error handling for invalid/nonexistent paths
//!   - ScanStats::new() creates default stats
//!   - ScanStats::record_node() increments correctly
//!   - ScanStats::record_file() increments correctly
//!   - ScanStats::record_skip() increments correctly
//!   - scan_single_file() with supported extensions
//!   - scan_single_file() with unsupported extensions

use std::fs;
use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{CodeGraph, GraphBuilder, ScanStats};
use trinity_harness::parsers::{Language, ParserRegistry};

// =============================================================================
// ScanStats tests
// =============================================================================

#[test]
fn test_scan_stats_new_is_default() {
    let stats = ScanStats::new();
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 0);
    assert_eq!(stats.total_nodes, 0);
    assert!(stats.nodes_per_language.is_empty());
}

#[test]
fn test_scan_stats_default_matches_new() {
    let new_stats = ScanStats::new();
    let default_stats = ScanStats::default();

    assert_eq!(new_stats.files_scanned, default_stats.files_scanned);
    assert_eq!(new_stats.files_skipped, default_stats.files_skipped);
    assert_eq!(new_stats.total_nodes, default_stats.total_nodes);
}

#[test]
fn test_scan_stats_record_node_increments_total() {
    let mut stats = ScanStats::new();

    stats.record_node(Language::Rust);
    assert_eq!(stats.total_nodes, 1);

    stats.record_node(Language::Rust);
    assert_eq!(stats.total_nodes, 2);

    stats.record_node(Language::Python);
    assert_eq!(stats.total_nodes, 3);
}

#[test]
fn test_scan_stats_record_node_tracks_per_language() {
    let mut stats = ScanStats::new();

    stats.record_node(Language::Rust);
    stats.record_node(Language::Rust);
    stats.record_node(Language::Python);
    stats.record_node(Language::Wgsl);
    stats.record_node(Language::Wgsl);
    stats.record_node(Language::Wgsl);

    assert_eq!(stats.nodes_per_language.get(&Language::Rust), Some(&2));
    assert_eq!(stats.nodes_per_language.get(&Language::Python), Some(&1));
    assert_eq!(stats.nodes_per_language.get(&Language::Wgsl), Some(&3));
}

#[test]
fn test_scan_stats_record_file_increments() {
    let mut stats = ScanStats::new();

    stats.record_file();
    assert_eq!(stats.files_scanned, 1);

    stats.record_file();
    stats.record_file();
    assert_eq!(stats.files_scanned, 3);
}

#[test]
fn test_scan_stats_record_skip_increments() {
    let mut stats = ScanStats::new();

    stats.record_skip();
    assert_eq!(stats.files_skipped, 1);

    stats.record_skip();
    stats.record_skip();
    assert_eq!(stats.files_skipped, 3);
}

#[test]
fn test_scan_stats_combined_operations() {
    let mut stats = ScanStats::new();

    // Simulate scanning 3 files, skipping 2
    stats.record_file();
    stats.record_node(Language::Rust);
    stats.record_node(Language::Rust);

    stats.record_skip(); // unsupported file

    stats.record_file();
    stats.record_node(Language::Python);

    stats.record_skip(); // another unsupported file

    stats.record_file();
    stats.record_node(Language::Wgsl);

    assert_eq!(stats.files_scanned, 3);
    assert_eq!(stats.files_skipped, 2);
    assert_eq!(stats.total_nodes, 4);
    assert_eq!(stats.nodes_per_language.len(), 3);
}

// =============================================================================
// GraphBuilder construction tests
// =============================================================================

#[test]
fn test_graph_builder_new_with_registry() {
    let registry = ParserRegistry::new();
    let _builder = GraphBuilder::new(&registry);
    // If we get here without panic, construction succeeded
}

#[test]
fn test_graph_builder_new_with_default_registry() {
    let registry = ParserRegistry::default();
    let _builder = GraphBuilder::new(&registry);
}

// =============================================================================
// full_scan() on empty directory
// =============================================================================

#[test]
fn test_full_scan_empty_directory_returns_empty_graph() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty(), "graph should have no nodes");
    assert!(graph.edges().is_empty(), "graph should have no edges");
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 0);
    assert_eq!(stats.total_nodes, 0);
}

#[test]
fn test_full_scan_directory_with_subdirs_only() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create subdirectories but no files
    fs::create_dir(temp_dir.path().join("subdir1")).unwrap();
    fs::create_dir(temp_dir.path().join("subdir2")).unwrap();
    fs::create_dir(temp_dir.path().join("subdir1/nested")).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 0);
}

// =============================================================================
// full_scan() extension filtering
// =============================================================================

#[test]
fn test_full_scan_filters_rust_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create a Rust file with a function
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn hello() { println!(\"Hello\"); }"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(!graph.nodes().is_empty(), "should have parsed Rust function");
    assert_eq!(stats.files_scanned, 1);
    assert!(stats.nodes_per_language.contains_key(&Language::Rust));
}

#[test]
fn test_full_scan_filters_python_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create a Python file with a function
    fs::write(
        temp_dir.path().join("script.py"),
        "def hello():\n    print('Hello')\n"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(!graph.nodes().is_empty(), "should have parsed Python function");
    assert_eq!(stats.files_scanned, 1);
    assert!(stats.nodes_per_language.contains_key(&Language::Python));
}

#[test]
fn test_full_scan_filters_wgsl_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create a WGSL file with a function
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn vertex_main() -> @location(0) vec4<f32> {\n    return vec4<f32>(0.0);\n}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(!graph.nodes().is_empty(), "should have parsed WGSL function");
    assert_eq!(stats.files_scanned, 1);
    assert!(stats.nodes_per_language.contains_key(&Language::Wgsl));
}

#[test]
fn test_full_scan_skips_unsupported_extensions() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create files with unsupported extensions
    fs::write(temp_dir.path().join("readme.md"), "# Readme").unwrap();
    fs::write(temp_dir.path().join("config.json"), "{}").unwrap();
    fs::write(temp_dir.path().join("style.css"), "body {}").unwrap();
    fs::write(temp_dir.path().join("script.js"), "function foo() {}").unwrap();
    fs::write(temp_dir.path().join("data.txt"), "some data").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty(), "should not have parsed any unsupported files");
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 5, "all 5 unsupported files should be skipped");
}

#[test]
fn test_full_scan_mixed_files_filters_correctly() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create supported files
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn rust_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("script.py"),
        "def python_func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn wgsl_func() {}"
    ).unwrap();

    // Create unsupported files
    fs::write(temp_dir.path().join("readme.md"), "# Readme").unwrap();
    fs::write(temp_dir.path().join("config.toml"), "[package]").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "should scan 3 supported files");
    assert_eq!(stats.files_skipped, 2, "should skip 2 unsupported files");
    assert!(stats.nodes_per_language.contains_key(&Language::Rust));
    assert!(stats.nodes_per_language.contains_key(&Language::Python));
    assert!(stats.nodes_per_language.contains_key(&Language::Wgsl));
    assert!(!graph.nodes().is_empty());
}

// =============================================================================
// full_scan() with nested directories
// =============================================================================

#[test]
fn test_full_scan_recursive_directory_traversal() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create nested directory structure
    fs::create_dir_all(temp_dir.path().join("src/module/submodule")).unwrap();

    fs::write(
        temp_dir.path().join("src/lib.rs"),
        "fn root_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("src/module/mod.rs"),
        "fn module_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("src/module/submodule/deep.rs"),
        "fn deep_func() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "should scan all 3 Rust files recursively");
    assert!(graph.nodes().len() >= 3, "should have at least 3 nodes");
}

// =============================================================================
// ScanStats correctness
// =============================================================================

#[test]
fn test_scan_stats_nodes_per_language_accuracy() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create multiple Rust files with multiple functions each
    fs::write(
        temp_dir.path().join("a.rs"),
        "fn func1() {}\nfn func2() {}\nstruct MyStruct {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("b.py"),
        "def func1():\n    pass\n\ndef func2():\n    pass\n"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Verify total nodes matches sum of per-language counts
    let sum: usize = stats.nodes_per_language.values().sum();
    assert_eq!(stats.total_nodes, sum, "total_nodes should equal sum of per-language counts");
    assert_eq!(stats.total_nodes, graph.nodes().len(), "total_nodes should match graph.nodes().len()");
}

#[test]
fn test_scan_stats_files_scanned_plus_skipped() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // 3 supported, 2 unsupported = 5 total files
    fs::write(temp_dir.path().join("a.rs"), "fn f() {}").unwrap();
    fs::write(temp_dir.path().join("b.py"), "def f():\n    pass\n").unwrap();
    fs::write(temp_dir.path().join("c.wgsl"), "fn f() {}").unwrap();
    fs::write(temp_dir.path().join("d.txt"), "text").unwrap();
    fs::write(temp_dir.path().join("e.md"), "# doc").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned + stats.files_skipped, 5);
    assert_eq!(stats.files_scanned, 3);
    assert_eq!(stats.files_skipped, 2);
}

// =============================================================================
// Error handling for invalid paths
// =============================================================================

#[test]
fn test_full_scan_nonexistent_path_returns_empty() {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // WalkDir will not error on a nonexistent path; it just yields nothing
    // This tests that the implementation handles this gracefully
    let result = builder.full_scan(Path::new("/nonexistent/path/that/does/not/exist"));

    // The implementation returns Ok with empty results, not an error
    let (graph, stats) = result.expect("should not error on nonexistent path");
    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
}

// =============================================================================
// scan_single_file() tests
// =============================================================================

#[test]
fn test_scan_single_file_rust() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let file_path = temp_dir.path().join("single.rs");
    fs::write(&file_path, "fn single_func() {}\nstruct SingleStruct {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut graph = CodeGraph::new();

    let count = builder.scan_single_file(&file_path, &mut graph);

    assert!(count.is_some(), "should return Some for .rs file");
    assert!(count.unwrap() >= 1, "should parse at least one unit");
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_single_file_python() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let file_path = temp_dir.path().join("single.py");
    fs::write(&file_path, "def single_func():\n    pass\n").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut graph = CodeGraph::new();

    let count = builder.scan_single_file(&file_path, &mut graph);

    assert!(count.is_some(), "should return Some for .py file");
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_single_file_wgsl() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let file_path = temp_dir.path().join("single.wgsl");
    fs::write(&file_path, "fn single_shader() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut graph = CodeGraph::new();

    let count = builder.scan_single_file(&file_path, &mut graph);

    assert!(count.is_some(), "should return Some for .wgsl file");
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_single_file_unsupported_returns_none() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let file_path = temp_dir.path().join("file.txt");
    fs::write(&file_path, "some text content").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut graph = CodeGraph::new();

    let count = builder.scan_single_file(&file_path, &mut graph);

    assert!(count.is_none(), "should return None for unsupported extension");
    assert!(graph.nodes().is_empty(), "graph should remain empty");
}

#[test]
fn test_scan_single_file_no_extension_returns_none() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let file_path = temp_dir.path().join("Makefile");
    fs::write(&file_path, "all: build").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut graph = CodeGraph::new();

    let count = builder.scan_single_file(&file_path, &mut graph);

    assert!(count.is_none(), "should return None for file without extension");
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_full_scan_empty_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create empty files with supported extensions
    fs::write(temp_dir.path().join("empty.rs"), "").unwrap();
    fs::write(temp_dir.path().join("empty.py"), "").unwrap();
    fs::write(temp_dir.path().join("empty.wgsl"), "").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Empty files should be scanned but produce no nodes
    assert_eq!(stats.files_scanned, 3, "empty files should still be counted as scanned");
    assert_eq!(stats.total_nodes, 0, "empty files should produce no nodes");
    assert!(graph.nodes().is_empty());
}

#[test]
fn test_full_scan_file_with_only_comments() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("comments.rs"),
        "// This is a comment\n// Another comment\n/* Block comment */"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1);
    // Comments don't produce code units
    assert!(graph.nodes().is_empty() || stats.total_nodes == 0);
}

#[test]
fn test_full_scan_hidden_files_included() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Hidden files (starting with .) should still be scanned if they have valid extensions
    fs::write(
        temp_dir.path().join(".hidden.rs"),
        "fn hidden_func() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // WalkDir by default includes hidden files
    assert!(stats.files_scanned >= 1, "hidden files should be scanned");
}

#[test]
fn test_full_scan_preserves_file_path_in_nodes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::create_dir(temp_dir.path().join("subdir")).unwrap();
    let file_path = temp_dir.path().join("subdir/module.rs");
    fs::write(&file_path, "fn test_func() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(!graph.nodes().is_empty());
    let node = &graph.nodes()[0];
    assert!(
        node.file_path.contains("subdir") && node.file_path.contains("module.rs"),
        "node file_path should contain the full path: got {}",
        node.file_path
    );
}

#[test]
fn test_scan_stats_clone() {
    let mut stats = ScanStats::new();
    stats.record_file();
    stats.record_node(Language::Rust);
    stats.record_skip();

    let cloned = stats.clone();

    assert_eq!(cloned.files_scanned, stats.files_scanned);
    assert_eq!(cloned.files_skipped, stats.files_skipped);
    assert_eq!(cloned.total_nodes, stats.total_nodes);
    assert_eq!(cloned.nodes_per_language, stats.nodes_per_language);
}

#[test]
fn test_scan_stats_debug() {
    let stats = ScanStats::new();
    let debug_str = format!("{:?}", stats);

    assert!(debug_str.contains("ScanStats"));
    assert!(debug_str.contains("files_scanned"));
}
