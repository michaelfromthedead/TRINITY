//! Whitebox tests for coverage report generation.

use trinity_harness::graph::{
    generate_coverage_report, get_covered_nodes, get_uncovered_nodes, CodeGraph, CodeNode,
    CoverageReport, FileCoverage, MappingSource, NodeId, TestMapping,
};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};

fn empty_hashes() -> ContentHashes {
    ContentHashes {
        full_hash: [0u8; 32],
        signature_hash: [0u8; 32],
        body_hash: [0u8; 32],
        layout_hash: [0u8; 32],
    }
}

fn make_node(
    graph: &mut CodeGraph,
    file: &str,
    name: &str,
    unit_type: UnitType,
    lang: Language,
) -> NodeId {
    let id = NodeId(graph.nodes().len());
    let unit = CodeUnit {
        unit_type,
        name: name.to_string(),
        start_line: 1,
        end_line: 10,
        language: lang,
        hashes: empty_hashes(),
    };
    let node = CodeNode::new(id, file.to_string(), unit);
    graph.add_node(node)
}

// ==================== CoverageReport ====================

#[test]
fn test_coverage_report_new() {
    let report = CoverageReport::new();
    assert_eq!(report.total_code_nodes, 0);
    assert_eq!(report.covered_nodes, 0);
    assert_eq!(report.uncovered_nodes, 0);
    assert_eq!(report.coverage_percent, 0.0);
}

#[test]
fn test_coverage_report_has_uncovered() {
    let mut report = CoverageReport::new();
    assert!(!report.has_uncovered());

    report.uncovered_nodes = 1;
    assert!(report.has_uncovered());
}

#[test]
fn test_coverage_report_generate_summary() {
    let mut report = CoverageReport::new();
    report.total_code_nodes = 10;
    report.covered_nodes = 7;
    report.uncovered_nodes = 3;
    report.coverage_percent = 70.0;

    let summary = report.generate_summary();

    assert!(summary.contains("Test Coverage Report"));
    assert!(summary.contains("70.0%"));
    assert!(summary.contains("7/10"));
    assert!(summary.contains("Uncovered: 3"));
}

// ==================== Generate Coverage Report ====================

#[test]
fn test_generate_coverage_empty() {
    let graph = CodeGraph::new();
    let mappings: Vec<TestMapping> = vec![];

    let report = generate_coverage_report(&graph, &mappings);

    assert_eq!(report.total_code_nodes, 0);
    assert_eq!(report.coverage_percent, 0.0);
}

#[test]
fn test_generate_coverage_all_covered() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_compute", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let report = generate_coverage_report(&graph, &mappings);

    assert_eq!(report.total_code_nodes, 1);
    assert_eq!(report.covered_nodes, 1);
    assert_eq!(report.uncovered_nodes, 0);
    assert_eq!(report.coverage_percent, 100.0);
}

#[test]
fn test_generate_coverage_partial() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/lib.rs", "covered_fn", UnitType::Function, Language::Rust);
    let _code2 = make_node(&mut graph, "src/lib.rs", "uncovered_fn", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_covered", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code1],
        source: MappingSource::Convention,
    }];

    let report = generate_coverage_report(&graph, &mappings);

    assert_eq!(report.total_code_nodes, 2);
    assert_eq!(report.covered_nodes, 1);
    assert_eq!(report.uncovered_nodes, 1);
    assert_eq!(report.coverage_percent, 50.0);
}

#[test]
fn test_generate_coverage_no_tests() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "func_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/lib.rs", "func_b", UnitType::Function, Language::Rust);

    let mappings: Vec<TestMapping> = vec![];

    let report = generate_coverage_report(&graph, &mappings);

    assert_eq!(report.total_code_nodes, 2);
    assert_eq!(report.covered_nodes, 0);
    assert_eq!(report.uncovered_nodes, 2);
    assert_eq!(report.coverage_percent, 0.0);
}

#[test]
fn test_generate_coverage_by_file() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/a.rs", "func_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/b.rs", "func_b", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_a", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code1],
        source: MappingSource::Convention,
    }];

    let report = generate_coverage_report(&graph, &mappings);

    assert!(report.by_file.contains_key("src/a.rs"));
    assert!(report.by_file.contains_key("src/b.rs"));

    let a_cov = &report.by_file["src/a.rs"];
    assert_eq!(a_cov.covered, 1);
    assert_eq!(a_cov.uncovered, 0);
    assert_eq!(a_cov.percent, 100.0);

    let b_cov = &report.by_file["src/b.rs"];
    assert_eq!(b_cov.covered, 0);
    assert_eq!(b_cov.uncovered, 1);
    assert_eq!(b_cov.percent, 0.0);
}

#[test]
fn test_generate_coverage_excludes_tests() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "production_fn", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test.rs", "test_fn", UnitType::Function, Language::Rust);

    let mappings: Vec<TestMapping> = vec![];

    let report = generate_coverage_report(&graph, &mappings);

    // Only counts production code, not tests
    assert_eq!(report.total_code_nodes, 1);
}

// ==================== Get Covered/Uncovered ====================

#[test]
fn test_get_covered_nodes() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/lib.rs", "covered", UnitType::Function, Language::Rust);
    let _code2 = make_node(&mut graph, "src/lib.rs", "uncovered", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_covered", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code1],
        source: MappingSource::Convention,
    }];

    let covered = get_covered_nodes(&graph, &mappings);

    assert_eq!(covered.len(), 1);
    assert!(covered.contains(&code1));
}

#[test]
fn test_get_uncovered_nodes() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/lib.rs", "covered", UnitType::Function, Language::Rust);
    let code2 = make_node(&mut graph, "src/lib.rs", "uncovered", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_covered", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code1],
        source: MappingSource::Convention,
    }];

    let uncovered = get_uncovered_nodes(&graph, &mappings);

    assert_eq!(uncovered.len(), 1);
    assert!(uncovered.contains(&code2));
}

#[test]
fn test_get_uncovered_excludes_tests() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "func", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test.rs", "test_func", UnitType::Function, Language::Rust);

    let mappings: Vec<TestMapping> = vec![];

    let uncovered = get_uncovered_nodes(&graph, &mappings);

    // Should only contain production code
    assert_eq!(uncovered.len(), 1);
}

// ==================== Summary Generation ====================

#[test]
fn test_summary_with_file_coverage() {
    let mut report = CoverageReport::new();
    report.total_code_nodes = 10;
    report.covered_nodes = 6;
    report.uncovered_nodes = 4;
    report.coverage_percent = 60.0;

    report.by_file.insert(
        "src/a.rs".to_string(),
        FileCoverage {
            total: 5,
            covered: 3,
            uncovered: 2,
            percent: 60.0,
        },
    );

    let summary = report.generate_summary();

    assert!(summary.contains("Coverage by file"));
    assert!(summary.contains("src/a.rs"));
    assert!(summary.contains("60.0%"));
}
