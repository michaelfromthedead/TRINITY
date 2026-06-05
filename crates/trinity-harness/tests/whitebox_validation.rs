//! Whitebox tests for test mapping validation.

use trinity_harness::graph::{
    get_orphan_tests, validate_mappings, verify_test_targets, CodeGraph, CodeNode,
    MappingSource, NodeId, TestMapping, TestValidationResult,
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

// ==================== TestValidationResult ====================

#[test]
fn test_validation_result_new() {
    let result = TestValidationResult::new();
    assert!(result.is_valid);
    assert_eq!(result.total_tests, 0);
    assert!(result.orphan_tests.is_empty());
    assert!(result.circular_deps.is_empty());
}

#[test]
fn test_validation_result_has_orphans() {
    let mut result = TestValidationResult::new();
    assert!(!result.has_orphans());

    result.orphan_tests.push(NodeId(0));
    assert!(result.has_orphans());
}

#[test]
fn test_validation_result_has_circular_deps() {
    let mut result = TestValidationResult::new();
    assert!(!result.has_circular_deps());

    result.circular_deps.push((NodeId(0), NodeId(1)));
    assert!(result.has_circular_deps());
}

#[test]
fn test_validation_result_generate_report() {
    let mut result = TestValidationResult::new();
    result.total_tests = 10;
    result.tests_with_targets = 8;
    result.orphan_tests = vec![NodeId(0), NodeId(1)];

    let report = result.generate_report();

    assert!(report.contains("Validation Report"));
    assert!(report.contains("PASSED"));
    assert!(report.contains("Tests validated: 10"));
    assert!(report.contains("Orphan tests: 2"));
}

#[test]
fn test_validation_result_report_failed() {
    let mut result = TestValidationResult::new();
    result.is_valid = false;
    result.errors.push("Test error".to_string());

    let report = result.generate_report();

    assert!(report.contains("FAILED"));
    assert!(report.contains("Test error"));
}

// ==================== Validate Mappings ====================

#[test]
fn test_validate_mappings_empty() {
    let graph = CodeGraph::new();
    let mappings: Vec<TestMapping> = vec![];

    let result = validate_mappings(&graph, &mappings);

    assert!(result.is_valid);
    assert_eq!(result.total_tests, 0);
}

#[test]
fn test_validate_mappings_all_valid() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_compute", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let result = validate_mappings(&graph, &mappings);

    assert!(result.is_valid);
    assert_eq!(result.total_tests, 1);
    assert_eq!(result.tests_with_targets, 1);
    assert!(result.orphan_tests.is_empty());
}

#[test]
fn test_validate_mappings_with_orphans() {
    let mut graph = CodeGraph::new();

    let test_id = make_node(&mut graph, "tests/test.rs", "test_orphan", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![],
        source: MappingSource::Unmapped,
    }];

    let result = validate_mappings(&graph, &mappings);

    // Orphans are warnings, not failures
    assert!(result.is_valid);
    assert_eq!(result.orphan_tests.len(), 1);
    assert!(result.orphan_tests.contains(&test_id));
}

#[test]
fn test_validate_mappings_circular_deps() {
    let mut graph = CodeGraph::new();

    let test1 = make_node(&mut graph, "tests/a.rs", "test_a", UnitType::Function, Language::Rust);
    let test2 = make_node(&mut graph, "tests/b.rs", "test_b", UnitType::Function, Language::Rust);

    // Circular: test1 targets test2
    let mappings = vec![
        TestMapping {
            test_id: test1,
            targets: vec![test2],
            source: MappingSource::Convention,
        },
        TestMapping {
            test_id: test2,
            targets: vec![test1],
            source: MappingSource::Convention,
        },
    ];

    let result = validate_mappings(&graph, &mappings);

    assert!(!result.is_valid);
    assert!(!result.circular_deps.is_empty());
}

// ==================== Verify Test Targets ====================

#[test]
fn test_verify_test_targets_all_valid() {
    let code_id = NodeId(0);
    let test_id = NodeId(1);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let (valid, invalid) = verify_test_targets(&mappings);

    assert_eq!(valid, 1);
    assert!(invalid.is_empty());
}

#[test]
fn test_verify_test_targets_with_invalid() {
    let code_id = NodeId(0);
    let test1 = NodeId(1);
    let test2 = NodeId(2);

    let mappings = vec![
        TestMapping {
            test_id: test1,
            targets: vec![code_id],
            source: MappingSource::Convention,
        },
        TestMapping {
            test_id: test2,
            targets: vec![],
            source: MappingSource::Unmapped,
        },
    ];

    let (valid, invalid) = verify_test_targets(&mappings);

    assert_eq!(valid, 1);
    assert_eq!(invalid.len(), 1);
    assert!(invalid.contains(&test2));
}

// ==================== Get Orphan Tests ====================

#[test]
fn test_get_orphan_tests_none() {
    let code_id = NodeId(0);
    let test_id = NodeId(1);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let orphans = get_orphan_tests(&mappings);

    assert!(orphans.is_empty());
}

#[test]
fn test_get_orphan_tests_some() {
    let test1 = NodeId(0);
    let test2 = NodeId(1);

    let mappings = vec![
        TestMapping {
            test_id: test1,
            targets: vec![NodeId(2)],
            source: MappingSource::Convention,
        },
        TestMapping {
            test_id: test2,
            targets: vec![],
            source: MappingSource::Unmapped,
        },
    ];

    let orphans = get_orphan_tests(&mappings);

    assert_eq!(orphans.len(), 1);
    assert!(orphans.contains(&test2));
}

#[test]
fn test_get_orphan_tests_empty_targets() {
    let test_id = NodeId(0);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![],
        source: MappingSource::Convention,
    }];

    let orphans = get_orphan_tests(&mappings);

    assert_eq!(orphans.len(), 1);
    assert!(orphans.contains(&test_id));
}
