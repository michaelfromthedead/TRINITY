//! Whitebox tests for unmapped test handling.

use trinity_harness::graph::{
    extract_unmapped, mark_as_orphan, CodeGraph, CodeNode, ConventionMapper, MappingSource,
    NodeId, TestMapping, UnmappedReview,
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

// ==================== UnmappedReview ====================

#[test]
fn test_unmapped_review_new() {
    let review = UnmappedReview::new();
    assert!(review.unmapped.is_empty());
    assert_eq!(review.total_unmapped, 0);
}

#[test]
fn test_unmapped_review_has_unmapped() {
    let mut review = UnmappedReview::new();
    assert!(!review.has_unmapped());

    review.total_unmapped = 1;
    review.unmapped.push(trinity_harness::graph::UnmappedTest {
        test_id: NodeId(0),
        test_name: "test_foo".to_string(),
        file_path: "tests/test.rs".to_string(),
        suggestions: vec![],
    });

    assert!(review.has_unmapped());
}

#[test]
fn test_unmapped_review_generate_report_empty() {
    let review = UnmappedReview::new();
    let report = review.generate_report();

    assert!(report.contains("Total: 0 unmapped tests"));
    assert!(report.contains("All tests are mapped"));
}

#[test]
fn test_unmapped_review_generate_report_with_unmapped() {
    let mut review = UnmappedReview::new();
    review.total_unmapped = 1;
    review.unmapped.push(trinity_harness::graph::UnmappedTest {
        test_id: NodeId(0),
        test_name: "test_missing".to_string(),
        file_path: "tests/test.rs".to_string(),
        suggestions: vec!["missing_fn".to_string()],
    });

    let report = review.generate_report();

    assert!(report.contains("Total: 1 unmapped tests"));
    assert!(report.contains("test_missing"));
    assert!(report.contains("tests/test.rs"));
    assert!(report.contains("Suggestions"));
    assert!(report.contains("missing_fn"));
}

// ==================== Extract Unmapped ====================

#[test]
fn test_extract_unmapped_empty() {
    let graph = CodeGraph::new();
    let mappings: Vec<TestMapping> = vec![];

    let review = extract_unmapped(&mappings, &graph);

    assert!(!review.has_unmapped());
    assert_eq!(review.total_unmapped, 0);
}

#[test]
fn test_extract_unmapped_none_unmapped() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_compute", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let review = extract_unmapped(&mappings, &graph);

    assert!(!review.has_unmapped());
}

#[test]
fn test_extract_unmapped_finds_unmapped() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_nonexistent", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![],
        source: MappingSource::Unmapped,
    }];

    let review = extract_unmapped(&mappings, &graph);

    assert!(review.has_unmapped());
    assert_eq!(review.total_unmapped, 1);
    assert_eq!(review.unmapped[0].test_id, test_id);
    assert_eq!(review.unmapped[0].test_name, "test_nonexistent");
}

#[test]
fn test_extract_unmapped_generates_suggestions() {
    let mut graph = CodeGraph::new();

    // Code with similar name
    make_node(&mut graph, "src/lib.rs", "process_data", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_process", UnitType::Function, Language::Rust);

    let mappings = vec![TestMapping {
        test_id,
        targets: vec![],
        source: MappingSource::Unmapped,
    }];

    let review = extract_unmapped(&mappings, &graph);

    assert!(review.has_unmapped());
    // Should suggest process_data since it contains "process"
    assert!(!review.unmapped[0].suggestions.is_empty());
    assert!(review.unmapped[0].suggestions.contains(&"process_data".to_string()));
}

#[test]
fn test_extract_unmapped_multiple() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "func", UnitType::Function, Language::Rust);
    let test1 = make_node(&mut graph, "tests/a.rs", "test_missing_a", UnitType::Function, Language::Rust);
    let test2 = make_node(&mut graph, "tests/b.rs", "test_missing_b", UnitType::Function, Language::Rust);

    let mappings = vec![
        TestMapping {
            test_id: test1,
            targets: vec![],
            source: MappingSource::Unmapped,
        },
        TestMapping {
            test_id: test2,
            targets: vec![],
            source: MappingSource::Unmapped,
        },
    ];

    let review = extract_unmapped(&mappings, &graph);

    assert_eq!(review.total_unmapped, 2);
    assert_eq!(review.unmapped.len(), 2);
}

// ==================== Mark as Orphan ====================

#[test]
fn test_mark_as_orphan() {
    let code_id = NodeId(0);
    let test_id = NodeId(1);

    let mut mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let count = mark_as_orphan(&mut mappings, &[test_id]);

    assert_eq!(count, 1);
    assert_eq!(mappings[0].source, MappingSource::Unmapped);
    assert!(mappings[0].targets.is_empty());
}

#[test]
fn test_mark_as_orphan_selective() {
    let code_id = NodeId(0);
    let test1 = NodeId(1);
    let test2 = NodeId(2);

    let mut mappings = vec![
        TestMapping {
            test_id: test1,
            targets: vec![code_id],
            source: MappingSource::Convention,
        },
        TestMapping {
            test_id: test2,
            targets: vec![code_id],
            source: MappingSource::Convention,
        },
    ];

    // Only mark test1 as orphan
    let count = mark_as_orphan(&mut mappings, &[test1]);

    assert_eq!(count, 1);
    assert_eq!(mappings[0].source, MappingSource::Unmapped);
    assert_eq!(mappings[1].source, MappingSource::Convention);
}

#[test]
fn test_mark_as_orphan_none() {
    let code_id = NodeId(0);
    let test_id = NodeId(1);
    let other_id = NodeId(99);

    let mut mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];

    let count = mark_as_orphan(&mut mappings, &[other_id]);

    assert_eq!(count, 0);
    assert_eq!(mappings[0].source, MappingSource::Convention);
}

// ==================== Integration ====================

#[test]
fn test_full_unmapped_flow() {
    let mut graph = CodeGraph::new();

    // Some code
    make_node(&mut graph, "src/lib.rs", "existing_fn", UnitType::Function, Language::Rust);

    // Mapped test
    make_node(&mut graph, "tests/mapped.rs", "test_existing_fn", UnitType::Function, Language::Rust);

    // Unmapped test
    make_node(&mut graph, "tests/orphan.rs", "test_something_else", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    // Should have some unmapped
    assert!(stats.tests_unmapped >= 1);

    // Extract and review
    let review = extract_unmapped(&mappings, &graph);
    let report = review.generate_report();

    // Report should mention the unmapped test
    assert!(report.contains("test_something_else") || review.total_unmapped >= 1);
}
