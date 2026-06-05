//! Blackbox tests for GraphBuilder public API
//!
//! CLEANROOM: Tests are written based on the public contract only.
//! The implementation file (graph/builder.rs) was NOT read.
//!
//! Contract under test (from PHASE_2_GRAPH_TODO.md):
//! - GraphBuilder - graph construction utilities
//! - full_scan(path) - scans directories using walkdir
//! - Filter by language extension (.rs, .py, .wgsl)
//! - ScanStats - statistics about the scan
//! - ScanError - error handling
//!
//! Test coverage plan:
//!   - T1: GraphBuilder construction
//!   - T2: full_scan on empty directory
//!   - T3: full_scan on directory with Rust files
//!   - T4: full_scan on directory with Python files
//!   - T5: full_scan on directory with WGSL files
//!   - T6: full_scan filters non-code files
//!   - T7: full_scan with mixed languages
//!   - T8: full_scan with nested directories
//!   - T9: ScanStats accuracy
//!   - T10: full_scan on nonexistent path returns error
//!   - T11: full_scan handles hidden files/directories
//!   - T12: full_scan handles symlinks gracefully
//!   - T13: ScanStats counts by language
//!   - T14: full_scan handles empty files
//!   - T15: full_scan handles files with parse errors gracefully

use std::fs::{self, File};
use std::io::Write;
use tempfile::TempDir;
use trinity_harness::{GraphBuilder, Language, ParserRegistry};

// ============================================================================
// T1: GraphBuilder Construction
// ============================================================================

#[test]
fn blackbox_builder_new_creates_instance() {
    // GraphBuilder requires a ParserRegistry
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    // If this compiles and runs, the constructor works
    drop(builder);
}

// ============================================================================
// T2: full_scan on Empty Directory
// ============================================================================

#[test]
fn blackbox_builder_scan_empty_directory() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let result = builder.full_scan(temp_dir.path());

    assert!(result.is_ok(), "Scanning empty directory should succeed");
    let (graph, stats) = result.unwrap();
    assert_eq!(stats.files_scanned, 0, "Empty directory has no files");
    assert_eq!(stats.total_nodes, 0, "Empty directory has no code units");
    assert_eq!(graph.nodes().len(), 0, "Graph should have no nodes");
}

// ============================================================================
// T3: full_scan on Directory with Rust Files
// ============================================================================

#[test]
fn blackbox_builder_scan_single_rust_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create a simple Rust file
    let rust_file = temp_dir.path().join("lib.rs");
    let mut file = File::create(&rust_file).expect("create file");
    writeln!(file, "fn hello() {{}}").expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    assert!(result.is_ok(), "Scanning Rust file should succeed");
    let (graph, stats) = result.unwrap();
    assert_eq!(stats.files_scanned, 1, "Should scan one file");
    assert!(stats.total_nodes >= 1, "Should find at least one function");
    // Verify Rust nodes exist in the language breakdown
    assert!(stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0) >= 1,
            "Should have at least one Rust node");
    assert!(graph.nodes().len() >= 1, "Graph should have at least one node");
}

#[test]
fn blackbox_builder_scan_multiple_rust_items() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let rust_file = temp_dir.path().join("complex.rs");
    let mut file = File::create(&rust_file).expect("create file");
    writeln!(file, r#"
fn func1() {{}}

struct MyStruct {{
    field: i32,
}}

fn func2() -> i32 {{
    42
}}
"#).expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1);
    // Should find 2 functions and 1 struct = 3 units minimum
    assert!(stats.total_nodes >= 3, "Should find at least 3 code units, found {}", stats.total_nodes);
    assert!(graph.nodes().len() >= 3, "Graph should have at least 3 nodes");
}

// ============================================================================
// T4: full_scan on Directory with Python Files
// ============================================================================

#[test]
fn blackbox_builder_scan_single_python_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let py_file = temp_dir.path().join("main.py");
    let mut file = File::create(&py_file).expect("create file");
    writeln!(file, "def greet():\n    pass").expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan one file");
    assert!(stats.total_nodes >= 1, "Should find at least one function");
    // Verify Python nodes exist in the language breakdown
    assert!(stats.nodes_per_language.get(&Language::Python).copied().unwrap_or(0) >= 1,
            "Should have at least one Python node");
}

#[test]
fn blackbox_builder_scan_python_class() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let py_file = temp_dir.path().join("models.py");
    let mut file = File::create(&py_file).expect("create file");
    writeln!(file, r#"
class User:
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {{self.name}}"

def create_user(name: str) -> User:
    return User(name)
"#).expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1);
    // Should find: 1 class, 2 methods (__init__, greet), 1 function (create_user) = 4 minimum
    // But depending on impl, might only count class + function = 2, or include methods
    assert!(stats.total_nodes >= 2, "Should find at least 2 code units");
}

// ============================================================================
// T5: full_scan on Directory with WGSL Files
// ============================================================================

#[test]
fn blackbox_builder_scan_single_wgsl_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let wgsl_file = temp_dir.path().join("shader.wgsl");
    let mut file = File::create(&wgsl_file).expect("create file");
    writeln!(file, r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {{
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}}
"#).expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan one file");
    assert!(stats.total_nodes >= 1, "Should find at least one entry point/function");
    // Verify WGSL nodes exist in the language breakdown
    assert!(stats.nodes_per_language.get(&Language::Wgsl).copied().unwrap_or(0) >= 1,
            "Should have at least one WGSL node");
}

#[test]
fn blackbox_builder_scan_wgsl_struct_and_function() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let wgsl_file = temp_dir.path().join("types.wgsl");
    let mut file = File::create(&wgsl_file).expect("create file");
    writeln!(file, r#"
struct VertexInput {{
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}}

struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {{
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}}
"#).expect("write content");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1);
    // Should find 2 structs + 1 function = 3 minimum
    assert!(stats.total_nodes >= 3, "Should find at least 3 code units, found {}", stats.total_nodes);
}

// ============================================================================
// T6: full_scan Filters Non-Code Files
// ============================================================================

#[test]
fn blackbox_builder_ignores_txt_files() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create a .txt file (should be ignored)
    let txt_file = temp_dir.path().join("readme.txt");
    File::create(&txt_file).expect("create file");

    // Create a .md file (should be ignored)
    let md_file = temp_dir.path().join("README.md");
    File::create(&md_file).expect("create file");

    // Create a .json file (should be ignored)
    let json_file = temp_dir.path().join("config.json");
    File::create(&json_file).expect("create file");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 0, "Should not scan non-code files");
    assert_eq!(stats.total_nodes, 0, "Should find no code units");
}

#[test]
fn blackbox_builder_scans_only_supported_extensions() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create various files
    File::create(temp_dir.path().join("file.c")).expect("create file");
    File::create(temp_dir.path().join("file.cpp")).expect("create file");
    File::create(temp_dir.path().join("file.java")).expect("create file");
    File::create(temp_dir.path().join("file.js")).expect("create file");
    File::create(temp_dir.path().join("file.ts")).expect("create file");
    File::create(temp_dir.path().join("file.go")).expect("create file");

    // Create one valid Rust file
    let mut rust_file = File::create(temp_dir.path().join("valid.rs")).expect("create file");
    writeln!(rust_file, "fn test() {{}}").expect("write");
    drop(rust_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Should only scan the .rs file
    assert_eq!(stats.files_scanned, 1, "Should only scan .rs file");
    assert!(stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0) >= 1,
            "Should have at least one Rust node");
}

// ============================================================================
// T7: full_scan with Mixed Languages
// ============================================================================

#[test]
fn blackbox_builder_scan_mixed_languages() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create Rust file
    let mut rust_file = File::create(temp_dir.path().join("lib.rs")).expect("create file");
    writeln!(rust_file, "pub fn rust_func() {{}}").expect("write");
    drop(rust_file);

    // Create Python file
    let mut py_file = File::create(temp_dir.path().join("script.py")).expect("create file");
    writeln!(py_file, "def python_func():\n    pass").expect("write");
    drop(py_file);

    // Create WGSL file
    let mut wgsl_file = File::create(temp_dir.path().join("shader.wgsl")).expect("create file");
    writeln!(wgsl_file, "@fragment\nfn frag_main() -> @location(0) vec4<f32> {{\n    return vec4<f32>(1.0);\n}}").expect("write");
    drop(wgsl_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "Should scan all three files");
    // Verify language breakdown
    assert!(stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0) >= 1,
            "Should have at least one Rust node");
    assert!(stats.nodes_per_language.get(&Language::Python).copied().unwrap_or(0) >= 1,
            "Should have at least one Python node");
    assert!(stats.nodes_per_language.get(&Language::Wgsl).copied().unwrap_or(0) >= 1,
            "Should have at least one WGSL node");
    assert!(stats.total_nodes >= 3, "Should find at least 3 code units");
}

// ============================================================================
// T8: full_scan with Nested Directories
// ============================================================================

#[test]
fn blackbox_builder_scan_nested_directories() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create nested structure: src/lib.rs, src/utils/helpers.rs
    let src_dir = temp_dir.path().join("src");
    let utils_dir = src_dir.join("utils");
    fs::create_dir_all(&utils_dir).expect("create dirs");

    let mut lib_file = File::create(src_dir.join("lib.rs")).expect("create file");
    writeln!(lib_file, "pub mod utils;\npub fn main_func() {{}}").expect("write");
    drop(lib_file);

    let mut helpers_file = File::create(utils_dir.join("helpers.rs")).expect("create file");
    writeln!(helpers_file, "pub fn helper_func() {{}}\npub fn another_helper() {{}}").expect("write");
    drop(helpers_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 2, "Should scan both files");
    // Verify Rust nodes from both files
    assert!(stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0) >= 3,
            "Should have at least 3 Rust nodes from both files");
    // Should find functions from both files
    assert!(stats.total_nodes >= 3, "Should find at least 3 functions");
}

#[test]
fn blackbox_builder_scan_deeply_nested() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create deep nesting: a/b/c/d/e/deep.py
    let deep_dir = temp_dir.path().join("a").join("b").join("c").join("d").join("e");
    fs::create_dir_all(&deep_dir).expect("create dirs");

    let mut py_file = File::create(deep_dir.join("deep.py")).expect("create file");
    writeln!(py_file, "def deep_func():\n    pass").expect("write");
    drop(py_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should find deeply nested file");
    assert!(stats.total_nodes >= 1, "Should find the function");
}

// ============================================================================
// T9: ScanStats Accuracy
// ============================================================================

#[test]
fn blackbox_builder_stats_counts_accurate() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create exactly 2 Rust files with known number of items
    let mut file1 = File::create(temp_dir.path().join("mod1.rs")).expect("create file");
    writeln!(file1, "fn a() {{}}\nfn b() {{}}").expect("write");
    drop(file1);

    let mut file2 = File::create(temp_dir.path().join("mod2.rs")).expect("create file");
    writeln!(file2, "struct S {{}}\nfn c() {{}}").expect("write");
    drop(file2);

    // Create 1 Python file
    let mut py_file = File::create(temp_dir.path().join("script.py")).expect("create file");
    writeln!(py_file, "def x():\n    pass").expect("write");
    drop(py_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 3, "Should scan exactly 3 files");
    // Verify language breakdown: 4 Rust items (a, b, S, c) + 1 Python item (x)
    let rust_nodes = stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0);
    let python_nodes = stats.nodes_per_language.get(&Language::Python).copied().unwrap_or(0);
    let wgsl_nodes = stats.nodes_per_language.get(&Language::Wgsl).copied().unwrap_or(0);
    assert!(rust_nodes >= 4, "Should have at least 4 Rust nodes, found {}", rust_nodes);
    assert!(python_nodes >= 1, "Should have at least 1 Python node, found {}", python_nodes);
    assert_eq!(wgsl_nodes, 0, "Should have 0 WGSL nodes");
    // 4 items in Rust (a, b, S, c) + 1 in Python (x) = 5 minimum
    assert!(stats.total_nodes >= 5, "Should have at least 5 units, found {}", stats.total_nodes);
}

#[test]
fn blackbox_builder_stats_zero_when_no_units() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create empty Rust file
    File::create(temp_dir.path().join("empty.rs")).expect("create file");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan the empty file");
    assert_eq!(stats.total_nodes, 0, "Empty file has no code units");
}

// ============================================================================
// T10: full_scan on Nonexistent Path
// ============================================================================

#[test]
fn blackbox_builder_scan_nonexistent_path_graceful_handling() {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(std::path::Path::new("/nonexistent/path/that/does/not/exist"));

    // The implementation may either return an error or gracefully return empty stats
    // Both are valid behaviors - the key is no panic
    match result {
        Ok((graph, stats)) => {
            // Graceful handling: empty results
            assert_eq!(stats.files_scanned, 0, "Nonexistent path has no files");
            assert_eq!(stats.total_nodes, 0, "Nonexistent path has no nodes");
            assert!(graph.nodes().is_empty(), "Nonexistent path has no graph nodes");
        }
        Err(_) => {
            // Error handling: also valid
        }
    }
}

#[test]
fn blackbox_builder_scan_nonexistent_path_no_panic() {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    // The key test: this should not panic, regardless of Ok/Err
    let _result = builder.full_scan(std::path::Path::new("/nonexistent/path"));
}

// ============================================================================
// T11: full_scan Handles Hidden Files/Directories
// ============================================================================

#[test]
fn blackbox_builder_hidden_files_behavior() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create hidden file
    let mut hidden_file = File::create(temp_dir.path().join(".hidden.rs")).expect("create file");
    writeln!(hidden_file, "fn hidden() {{}}").expect("write");
    drop(hidden_file);

    // Create visible file
    let mut visible_file = File::create(temp_dir.path().join("visible.rs")).expect("create file");
    writeln!(visible_file, "fn visible() {{}}").expect("write");
    drop(visible_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Implementation may or may not skip hidden files - just verify no crash
    // and at least the visible file is scanned
    assert!(stats.files_scanned >= 1, "Should scan at least the visible file");
}

#[test]
fn blackbox_builder_hidden_directory_behavior() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create hidden directory with file inside
    let hidden_dir = temp_dir.path().join(".hidden_dir");
    fs::create_dir(&hidden_dir).expect("create hidden dir");
    let mut hidden_file = File::create(hidden_dir.join("inside.rs")).expect("create file");
    writeln!(hidden_file, "fn inside_hidden() {{}}").expect("write");
    drop(hidden_file);

    // Create visible file
    let mut visible_file = File::create(temp_dir.path().join("visible.rs")).expect("create file");
    writeln!(visible_file, "fn visible() {{}}").expect("write");
    drop(visible_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Should at least scan the visible file without crashing
    assert!(stats.files_scanned >= 1, "Should scan at least visible file");
}

// ============================================================================
// T12: full_scan Handles Symlinks Gracefully
// ============================================================================

#[cfg(unix)]
#[test]
fn blackbox_builder_symlink_to_file() {
    use std::os::unix::fs::symlink;

    let temp_dir = TempDir::new().expect("create temp dir");

    // Create actual file
    let mut actual_file = File::create(temp_dir.path().join("actual.rs")).expect("create file");
    writeln!(actual_file, "fn actual() {{}}").expect("write");
    drop(actual_file);

    // Create symlink to the file
    symlink(
        temp_dir.path().join("actual.rs"),
        temp_dir.path().join("link.rs")
    ).expect("create symlink");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    // Should not crash on symlinks
    assert!(result.is_ok(), "Should handle symlinks gracefully");
}

#[cfg(unix)]
#[test]
fn blackbox_builder_broken_symlink() {
    use std::os::unix::fs::symlink;

    let temp_dir = TempDir::new().expect("create temp dir");

    // Create symlink to nonexistent file
    symlink(
        temp_dir.path().join("nonexistent.rs"),
        temp_dir.path().join("broken_link.rs")
    ).expect("create symlink");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    // Should handle broken symlinks gracefully (not crash)
    assert!(result.is_ok(), "Should handle broken symlinks gracefully");
}

// ============================================================================
// T13: ScanStats Counts by Language
// ============================================================================

#[test]
fn blackbox_builder_stats_language_breakdown() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create 3 Rust files
    for i in 0..3 {
        let mut f = File::create(temp_dir.path().join(format!("rust{}.rs", i))).expect("create");
        writeln!(f, "fn func{}() {{}}", i).expect("write");
    }

    // Create 2 Python files
    for i in 0..2 {
        let mut f = File::create(temp_dir.path().join(format!("py{}.py", i))).expect("create");
        writeln!(f, "def func{}():\n    pass", i).expect("write");
    }

    // Create 1 WGSL file
    let mut wgsl = File::create(temp_dir.path().join("shader.wgsl")).expect("create");
    writeln!(wgsl, "@compute @workgroup_size(1)\nfn main() {{}}").expect("write");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Verify language breakdown
    let rust_nodes = stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0);
    let python_nodes = stats.nodes_per_language.get(&Language::Python).copied().unwrap_or(0);
    let wgsl_nodes = stats.nodes_per_language.get(&Language::Wgsl).copied().unwrap_or(0);
    assert!(rust_nodes >= 3, "Should have at least 3 Rust nodes (one per file), found {}", rust_nodes);
    assert!(python_nodes >= 2, "Should have at least 2 Python nodes (one per file), found {}", python_nodes);
    assert!(wgsl_nodes >= 1, "Should have at least 1 WGSL node, found {}", wgsl_nodes);
    assert_eq!(stats.files_scanned, 6, "Should scan 6 files total");
}

// ============================================================================
// T14: full_scan Handles Empty Files
// ============================================================================

#[test]
fn blackbox_builder_empty_rust_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create empty Rust file
    File::create(temp_dir.path().join("empty.rs")).expect("create file");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan empty file");
    assert_eq!(stats.total_nodes, 0, "Empty file has no units");
}

#[test]
fn blackbox_builder_whitespace_only_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("whitespace.rs")).expect("create file");
    writeln!(file, "   \n\n   \n  \t  ").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan whitespace file");
    assert_eq!(stats.total_nodes, 0, "Whitespace file has no units");
}

#[test]
fn blackbox_builder_comment_only_file() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("comments.rs")).expect("create file");
    writeln!(file, "// Just a comment\n// Another comment").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 1, "Should scan comment-only file");
    assert_eq!(stats.total_nodes, 0, "Comment-only file has no units");
}

// ============================================================================
// T15: full_scan Handles Files with Parse Errors Gracefully
// ============================================================================

#[test]
fn blackbox_builder_invalid_rust_syntax() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("invalid.rs")).expect("create file");
    writeln!(file, "fn broken( {{ }}").expect("write"); // Invalid syntax
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    // Should not crash, may either skip the file or return partial results
    // The key is graceful handling
    assert!(result.is_ok(), "Should handle parse errors gracefully");
}

#[test]
fn blackbox_builder_mixed_valid_invalid_files() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create valid file
    let mut valid = File::create(temp_dir.path().join("valid.rs")).expect("create file");
    writeln!(valid, "fn valid_func() {{}}").expect("write");
    drop(valid);

    // Create invalid file
    let mut invalid = File::create(temp_dir.path().join("invalid.rs")).expect("create file");
    writeln!(invalid, "fn broken((( {{").expect("write");
    drop(invalid);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    // Should still scan and extract from the valid file
    assert!(result.is_ok(), "Should handle mixed valid/invalid files");
    let (_graph, stats) = result.unwrap();
    // At minimum, should have scanned files (may skip invalid)
    assert!(stats.files_scanned >= 1, "Should scan at least valid file");
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn blackbox_builder_unicode_filenames() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("测试.rs")).expect("create file");
    writeln!(file, "fn unicode_test() {{}}").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    assert!(result.is_ok(), "Should handle unicode filenames");
}

#[test]
fn blackbox_builder_unicode_content() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("content.rs")).expect("create file");
    writeln!(file, r#"
fn greeting() -> &'static str {{
    "こんにちは世界"
}}
"#).expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert!(stats.total_nodes >= 1, "Should parse file with unicode content");
}

#[test]
fn blackbox_builder_very_long_filename() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create file with long name (but not exceeding OS limits)
    let long_name = format!("{}.rs", "a".repeat(200));
    let mut file = File::create(temp_dir.path().join(&long_name)).expect("create file");
    writeln!(file, "fn long_name() {{}}").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let result = builder.full_scan(temp_dir.path());

    assert!(result.is_ok(), "Should handle long filenames");
}

#[test]
fn blackbox_builder_file_extension_case_sensitivity() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create files with various case extensions
    let mut rs_lower = File::create(temp_dir.path().join("lower.rs")).expect("create");
    writeln!(rs_lower, "fn lower() {{}}").expect("write");

    let mut rs_upper = File::create(temp_dir.path().join("upper.RS")).expect("create");
    writeln!(rs_upper, "fn upper() {{}}").expect("write");

    let mut py_mixed = File::create(temp_dir.path().join("mixed.Py")).expect("create");
    writeln!(py_mixed, "def mixed():\n    pass").expect("write");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Standard behavior: extensions are case-sensitive, so .rs works but .RS may not
    // At minimum, .rs should be scanned
    assert!(stats.files_scanned >= 1, "Should scan at least lowercase extensions");
}

#[test]
fn blackbox_builder_scan_directory_with_many_files() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create 100 files
    for i in 0..100 {
        let mut f = File::create(temp_dir.path().join(format!("file{}.rs", i))).expect("create");
        writeln!(f, "fn func{}() {{}}", i).expect("write");
    }

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (_graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    assert_eq!(stats.files_scanned, 100, "Should scan all 100 files");
    let rust_nodes = stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0);
    assert!(rust_nodes >= 100, "Should have at least 100 Rust nodes, found {}", rust_nodes);
    assert!(stats.total_nodes >= 100, "Should find at least 100 functions");
}

// ============================================================================
// Builder with Graph Access
// ============================================================================

#[test]
fn blackbox_builder_returns_populated_graph_and_stats() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let mut file = File::create(temp_dir.path().join("test.rs")).expect("create file");
    writeln!(file, "struct Data {{}}\nfn process(d: Data) {{}}").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Verify stats are properly populated
    assert!(stats.files_scanned > 0, "Should have scanned files");
    assert!(stats.total_nodes > 0, "Should have found units");
    let rust_nodes = stats.nodes_per_language.get(&Language::Rust).copied().unwrap_or(0);
    assert!(rust_nodes > 0, "Should have Rust nodes");

    // Verify graph is populated
    assert!(!graph.nodes().is_empty(), "Graph should have nodes");
    assert_eq!(graph.nodes().len(), stats.total_nodes, "Graph nodes should match stats.total_nodes");
}

#[test]
fn blackbox_builder_graph_nodes_have_correct_file_paths() {
    let temp_dir = TempDir::new().expect("create temp dir");

    let rust_file = temp_dir.path().join("module.rs");
    let mut file = File::create(&rust_file).expect("create file");
    writeln!(file, "fn test_func() {{}}").expect("write");
    drop(file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Check that nodes have file paths
    for node in graph.nodes() {
        assert!(!node.file_path.is_empty(), "Node should have a file path");
        assert!(node.file_path.ends_with(".rs"), "Rust node should have .rs path");
    }
}

#[test]
fn blackbox_builder_graph_nodes_have_language_info() {
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create mixed language files
    let mut rs_file = File::create(temp_dir.path().join("lib.rs")).expect("create");
    writeln!(rs_file, "fn rust_fn() {{}}").expect("write");
    drop(rs_file);

    let mut py_file = File::create(temp_dir.path().join("script.py")).expect("create");
    writeln!(py_file, "def python_fn():\n    pass").expect("write");
    drop(py_file);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let (graph, _stats) = builder.full_scan(temp_dir.path()).expect("scan should succeed");

    // Check that we have nodes from both languages
    let rust_nodes: Vec<_> = graph.nodes().iter()
        .filter(|n| n.file_path.ends_with(".rs"))
        .collect();
    let python_nodes: Vec<_> = graph.nodes().iter()
        .filter(|n| n.file_path.ends_with(".py"))
        .collect();

    assert!(!rust_nodes.is_empty(), "Should have Rust nodes");
    assert!(!python_nodes.is_empty(), "Should have Python nodes");
}
