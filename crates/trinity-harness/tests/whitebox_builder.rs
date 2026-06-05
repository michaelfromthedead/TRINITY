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

// =============================================================================
// scan_rust() tests - T-GRAPH-2.2
// =============================================================================

#[test]
fn test_scan_rust_filters_only_rs_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create files with various extensions
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn rust_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("main.rs"),
        "fn main() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("script.py"),
        "def python_func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn wgsl_func() {}"
    ).unwrap();
    fs::write(temp_dir.path().join("readme.md"), "# Readme").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_rust(temp_dir.path()).expect("scan should succeed");

    // Should only scan .rs files
    assert_eq!(stats.files_scanned, 2, "should scan only 2 Rust files");
    assert_eq!(stats.files_skipped, 3, "should skip 3 non-Rust files");
    assert!(stats.nodes_per_language.contains_key(&Language::Rust));
    assert!(!stats.nodes_per_language.contains_key(&Language::Python));
    assert!(!stats.nodes_per_language.contains_key(&Language::Wgsl));
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_rust_nested_directories() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create nested structure
    fs::create_dir_all(temp_dir.path().join("src/module")).unwrap();

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn root() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("src/main.rs"),
        "fn main() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("src/module/mod.rs"),
        "fn module_fn() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("src/module/helper.py"),
        "def helper():\n    pass\n"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (_graph, stats) = builder.scan_rust(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "should scan all 3 Rust files");
    assert_eq!(stats.files_skipped, 1, "should skip 1 Python file");
}

#[test]
fn test_scan_rust_empty_directory() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_rust(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 0);
}

#[test]
fn test_scan_rust_no_rust_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("script.py"),
        "def func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn shader() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_rust(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 2, "should skip all non-Rust files");
}

// =============================================================================
// scan_python() tests - T-GRAPH-2.2
// =============================================================================

#[test]
fn test_scan_python_filters_only_py_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create files with various extensions
    fs::write(
        temp_dir.path().join("script.py"),
        "def python_func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("module.py"),
        "class MyClass:\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn rust_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn wgsl_func() {}"
    ).unwrap();
    fs::write(temp_dir.path().join("config.json"), "{}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_python(temp_dir.path()).expect("scan should succeed");

    // Should only scan .py files
    assert_eq!(stats.files_scanned, 2, "should scan only 2 Python files");
    assert_eq!(stats.files_skipped, 3, "should skip 3 non-Python files");
    assert!(stats.nodes_per_language.contains_key(&Language::Python));
    assert!(!stats.nodes_per_language.contains_key(&Language::Rust));
    assert!(!stats.nodes_per_language.contains_key(&Language::Wgsl));
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_python_nested_packages() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create Python package structure
    fs::create_dir_all(temp_dir.path().join("mypackage/subpackage")).unwrap();

    fs::write(
        temp_dir.path().join("mypackage/__init__.py"),
        ""
    ).unwrap();
    fs::write(
        temp_dir.path().join("mypackage/module.py"),
        "def module_func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("mypackage/subpackage/__init__.py"),
        ""
    ).unwrap();
    fs::write(
        temp_dir.path().join("mypackage/subpackage/deep.py"),
        "def deep_func():\n    pass\n"
    ).unwrap();
    fs::write(
        temp_dir.path().join("mypackage/helper.rs"),
        "fn helper() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (_graph, stats) = builder.scan_python(temp_dir.path()).expect("scan should succeed");

    // 4 .py files (2 __init__.py + module.py + deep.py)
    assert_eq!(stats.files_scanned, 4, "should scan all 4 Python files");
    assert_eq!(stats.files_skipped, 1, "should skip 1 Rust file");
}

#[test]
fn test_scan_python_no_python_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shader.wgsl"),
        "fn shader() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_python(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 2, "should skip all non-Python files");
}

// =============================================================================
// scan_wgsl() tests - T-GRAPH-2.2
// =============================================================================

#[test]
fn test_scan_wgsl_filters_only_wgsl_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create files with various extensions
    fs::write(
        temp_dir.path().join("vertex.wgsl"),
        "fn vertex_main() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("fragment.wgsl"),
        "fn fragment_main() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn rust_func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("script.py"),
        "def python_func():\n    pass\n"
    ).unwrap();
    fs::write(temp_dir.path().join("shader.glsl"), "void main() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_wgsl(temp_dir.path()).expect("scan should succeed");

    // Should only scan .wgsl files
    assert_eq!(stats.files_scanned, 2, "should scan only 2 WGSL files");
    assert_eq!(stats.files_skipped, 3, "should skip 3 non-WGSL files");
    assert!(stats.nodes_per_language.contains_key(&Language::Wgsl));
    assert!(!stats.nodes_per_language.contains_key(&Language::Rust));
    assert!(!stats.nodes_per_language.contains_key(&Language::Python));
    assert!(!graph.nodes().is_empty());
}

#[test]
fn test_scan_wgsl_shader_directory() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create shader directory structure
    fs::create_dir_all(temp_dir.path().join("shaders/postprocess")).unwrap();

    fs::write(
        temp_dir.path().join("shaders/common.wgsl"),
        "fn common_util() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shaders/pbr.wgsl"),
        "fn pbr_lighting() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shaders/postprocess/bloom.wgsl"),
        "fn bloom_effect() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("shaders/readme.md"),
        "# Shader documentation"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (_graph, stats) = builder.scan_wgsl(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "should scan all 3 WGSL files");
    assert_eq!(stats.files_skipped, 1, "should skip readme.md");
}

#[test]
fn test_scan_wgsl_no_wgsl_files() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn func() {}"
    ).unwrap();
    fs::write(
        temp_dir.path().join("script.py"),
        "def func():\n    pass\n"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, stats) = builder.scan_wgsl(temp_dir.path()).expect("scan should succeed");

    assert!(graph.nodes().is_empty());
    assert_eq!(stats.files_scanned, 0);
    assert_eq!(stats.files_skipped, 2, "should skip all non-WGSL files");
}

// =============================================================================
// scan_language() tests - generic language filtering
// =============================================================================

#[test]
fn test_scan_language_rust_same_as_scan_rust() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("lib.rs"), "fn func() {}").unwrap();
    fs::write(temp_dir.path().join("script.py"), "def f():\n    pass\n").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph1, stats1) = builder.scan_rust(temp_dir.path()).expect("scan_rust should succeed");
    let (graph2, stats2) = builder.scan_language(temp_dir.path(), Language::Rust).expect("scan_language should succeed");

    assert_eq!(stats1.files_scanned, stats2.files_scanned);
    assert_eq!(stats1.files_skipped, stats2.files_skipped);
    assert_eq!(stats1.total_nodes, stats2.total_nodes);
    assert_eq!(graph1.nodes().len(), graph2.nodes().len());
}

#[test]
fn test_scan_language_python_same_as_scan_python() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("lib.rs"), "fn func() {}").unwrap();
    fs::write(temp_dir.path().join("script.py"), "def f():\n    pass\n").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph1, stats1) = builder.scan_python(temp_dir.path()).expect("scan_python should succeed");
    let (graph2, stats2) = builder.scan_language(temp_dir.path(), Language::Python).expect("scan_language should succeed");

    assert_eq!(stats1.files_scanned, stats2.files_scanned);
    assert_eq!(stats1.files_skipped, stats2.files_skipped);
    assert_eq!(stats1.total_nodes, stats2.total_nodes);
    assert_eq!(graph1.nodes().len(), graph2.nodes().len());
}

#[test]
fn test_scan_language_wgsl_same_as_scan_wgsl() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("shader.wgsl"), "fn shader() {}").unwrap();
    fs::write(temp_dir.path().join("lib.rs"), "fn func() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph1, stats1) = builder.scan_wgsl(temp_dir.path()).expect("scan_wgsl should succeed");
    let (graph2, stats2) = builder.scan_language(temp_dir.path(), Language::Wgsl).expect("scan_language should succeed");

    assert_eq!(stats1.files_scanned, stats2.files_scanned);
    assert_eq!(stats1.files_skipped, stats2.files_skipped);
    assert_eq!(stats1.total_nodes, stats2.total_nodes);
    assert_eq!(graph1.nodes().len(), graph2.nodes().len());
}

// =============================================================================
// persist_graph_to_db() tests - T-GRAPH-2.2
// =============================================================================

use trinity_harness::db::HarnessDb;
use trinity_harness::graph::persist_graph_to_db;

#[test]
fn test_persist_graph_to_db_inserts_nodes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    // Create a Rust file
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn func1() {}\nfn func2() {}\nstruct MyStruct {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Open in-memory database
    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");

    // Persist the graph
    let count = persist_graph_to_db(&graph, &db).expect("persist should succeed");

    assert_eq!(count, graph.nodes().len(), "persisted count should match node count");
    assert!(count >= 1, "should have persisted at least one node");
    assert_eq!(count, stats.total_nodes, "persisted count should match stats.total_nodes");
}

#[test]
fn test_persist_graph_to_db_empty_graph() {
    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    let graph = CodeGraph::new();

    let count = persist_graph_to_db(&graph, &db).expect("persist should succeed");

    assert_eq!(count, 0, "persisting empty graph should return 0");
}

#[test]
fn test_persist_graph_to_db_correct_node_data() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("test.rs"),
        "fn test_function() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query the database to verify the data
    let conn = db.connection();
    let mut stmt = conn.prepare("SELECT file_path, language, kind, name FROM code_nodes").unwrap();
    let rows: Vec<(String, String, String, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(!rows.is_empty(), "should have at least one row");

    let (file_path, language, kind, name) = &rows[0];
    assert!(file_path.contains("test.rs"), "file_path should contain test.rs");
    assert_eq!(language, "rust", "language should be rust");
    assert_eq!(kind, "rust_function", "kind should be rust_function");
    assert_eq!(name, "test_function", "name should be test_function");
}

#[test]
fn test_persist_graph_to_db_multiple_languages() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

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

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query to check we have nodes for each language
    let conn = db.connection();

    let rust_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE language = 'rust'", [], |row| row.get(0))
        .unwrap();
    let python_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE language = 'python'", [], |row| row.get(0))
        .unwrap();
    let wgsl_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE language = 'wgsl'", [], |row| row.get(0))
        .unwrap();

    assert!(rust_count >= 1, "should have at least 1 Rust node");
    assert!(python_count >= 1, "should have at least 1 Python node");
    assert!(wgsl_count >= 1, "should have at least 1 WGSL node");
}

#[test]
fn test_persist_graph_to_db_replaces_existing() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn original_func() {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // First scan and persist
    let (graph1, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");
    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph1, &db).expect("first persist should succeed");

    // Modify file and scan again
    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn original_func() {} // modified"
    ).unwrap();
    let (graph2, _) = builder.full_scan(temp_dir.path()).expect("second scan should succeed");

    // Persist again - should replace
    let count = persist_graph_to_db(&graph2, &db).expect("second persist should succeed");

    // The INSERT OR REPLACE should have updated the row
    assert!(count >= 1, "should persist at least one node");

    // Verify we still have only one entry for this function
    let conn = db.connection();
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE name = 'original_func'", [], |row| row.get(0))
        .unwrap();
    assert_eq!(total, 1, "should have exactly one row for original_func after replace");
}

#[test]
fn test_persist_graph_to_db_stores_line_numbers() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "// comment\n// another\nfn func_on_line_three() {\n    println!(\"body\");\n}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query line numbers
    let conn = db.connection();
    let (start_line, end_line): (i64, i64) = conn
        .query_row(
            "SELECT span_start_line, span_end_line FROM code_nodes WHERE name = 'func_on_line_three'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?))
        )
        .expect("should find the function");

    assert!(start_line >= 3, "start_line should be at least 3");
    assert!(end_line >= start_line, "end_line should be >= start_line");
}

#[test]
fn test_persist_graph_to_db_stores_hash() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn hashed_func() { let x = 42; }"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query hash
    let conn = db.connection();
    let hash: String = conn
        .query_row(
            "SELECT hash_full FROM code_nodes WHERE name = 'hashed_func'",
            [],
            |row| row.get(0)
        )
        .expect("should find the function");

    assert!(!hash.is_empty(), "hash should not be empty");
    // Hash should be hex encoded (64 chars for SHA-256)
    assert!(hash.len() >= 32, "hash should be at least 32 chars (hex encoded)");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "hash should be hex encoded");
}

// =============================================================================
// Database roundtrip tests - T-GRAPH-2.2
// =============================================================================

#[test]
fn test_db_roundtrip_persist_then_query_all() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn func1() {}\nstruct Struct1 {}\nenum Enum1 {}"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    let persisted_count = persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query back all nodes
    let conn = db.connection();
    let db_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes", [], |row| row.get(0))
        .unwrap();

    assert_eq!(persisted_count as i64, db_count, "db count should match persisted count");
    assert_eq!(stats.total_nodes as i64, db_count, "db count should match stats.total_nodes");
}

#[test]
fn test_db_roundtrip_query_by_language() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("a.rs"), "fn rs1() {}\nfn rs2() {}").unwrap();
    fs::write(temp_dir.path().join("b.py"), "def py1():\n    pass\n").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query Rust nodes
    let conn = db.connection();
    let rust_nodes: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE language = 'rust'", [], |row| row.get(0))
        .unwrap();
    let python_nodes: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE language = 'python'", [], |row| row.get(0))
        .unwrap();

    assert!(rust_nodes >= 2, "should have at least 2 Rust nodes");
    assert!(python_nodes >= 1, "should have at least 1 Python node");
}

#[test]
fn test_db_roundtrip_query_by_kind() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(
        temp_dir.path().join("lib.rs"),
        "fn func1() {}\nstruct MyStruct {}\nenum MyEnum { A, B }"
    ).unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query by kind
    let conn = db.connection();
    let functions: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE kind = 'rust_function'", [], |row| row.get(0))
        .unwrap();
    let structs: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE kind = 'rust_struct'", [], |row| row.get(0))
        .unwrap();
    let enums: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE kind = 'rust_enum'", [], |row| row.get(0))
        .unwrap();

    assert!(functions >= 1, "should have at least 1 function");
    assert!(structs >= 1, "should have at least 1 struct");
    assert!(enums >= 1, "should have at least 1 enum");
}

#[test]
fn test_db_roundtrip_query_by_file() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::create_dir(temp_dir.path().join("src")).unwrap();
    fs::write(temp_dir.path().join("src/lib.rs"), "fn lib_func() {}").unwrap();
    fs::write(temp_dir.path().join("src/utils.rs"), "fn util_func() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Query by file path pattern
    let conn = db.connection();
    let lib_nodes: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE file_path LIKE '%lib.rs%'", [], |row| row.get(0))
        .unwrap();
    let util_nodes: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_nodes WHERE file_path LIKE '%utils.rs%'", [], |row| row.get(0))
        .unwrap();

    assert!(lib_nodes >= 1, "should have nodes from lib.rs");
    assert!(util_nodes >= 1, "should have nodes from utils.rs");
}

#[test]
fn test_db_roundtrip_default_state() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("lib.rs"), "fn stateful_func() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Verify default state is 'unknown'
    let conn = db.connection();
    let state: String = conn
        .query_row(
            "SELECT current_state FROM code_nodes WHERE name = 'stateful_func'",
            [],
            |row| row.get(0)
        )
        .expect("should find the function");

    assert_eq!(state, "unknown", "default state should be 'unknown'");
}

#[test]
fn test_db_roundtrip_node_id_format() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");

    fs::write(temp_dir.path().join("lib.rs"), "fn my_function() {}").unwrap();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    let db = HarnessDb::open_in_memory().expect("failed to open in-memory db");
    persist_graph_to_db(&graph, &db).expect("persist should succeed");

    // Node ID should be in format: file_path:start_line:name
    let conn = db.connection();
    let node_id: String = conn
        .query_row(
            "SELECT node_id FROM code_nodes WHERE name = 'my_function'",
            [],
            |row| row.get(0)
        )
        .expect("should find the function");

    assert!(node_id.contains("lib.rs"), "node_id should contain file path");
    assert!(node_id.contains("my_function"), "node_id should contain function name");
    let parts: Vec<&str> = node_id.split(':').collect();
    assert!(parts.len() >= 3, "node_id should have at least 3 parts separated by :");
}
