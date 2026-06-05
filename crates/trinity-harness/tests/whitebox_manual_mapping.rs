//! Whitebox tests for manual TOML-based test mapping.

use trinity_harness::graph::{
    CombinedMapper, CodeGraph, CodeNode, ExplicitMapper, ExplicitMapping, MappingConfig,
    MappingSource, NodeId,
};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};
use std::path::Path;

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

// ==================== MappingConfig Parsing ====================

#[test]
fn test_parse_empty_config() {
    let toml = "";
    let config = MappingConfig::parse(toml).unwrap();
    assert!(config.mappings.is_empty());
}

#[test]
fn test_parse_single_mapping() {
    let toml = r#"
[[mappings]]
test = "tests/test_parser.rs"
targets = ["src/parser.rs"]
"#;
    let config = MappingConfig::parse(toml).unwrap();
    assert_eq!(config.mappings.len(), 1);
    assert_eq!(config.mappings[0].test, "tests/test_parser.rs");
    assert_eq!(config.mappings[0].targets, vec!["src/parser.rs"]);
}

#[test]
fn test_parse_multiple_mappings() {
    let toml = r#"
[[mappings]]
test = "tests/test_a.rs"
targets = ["src/a.rs"]

[[mappings]]
test = "tests/test_b.rs"
targets = ["src/b.rs", "src/c.rs"]
"#;
    let config = MappingConfig::parse(toml).unwrap();
    assert_eq!(config.mappings.len(), 2);
    assert_eq!(config.mappings[0].targets.len(), 1);
    assert_eq!(config.mappings[1].targets.len(), 2);
}

#[test]
fn test_parse_glob_patterns() {
    let toml = r#"
[[mappings]]
test = "tests/integration/*.rs"
targets = ["src/core/*.rs", "src/utils.rs"]
"#;
    let config = MappingConfig::parse(toml).unwrap();
    assert_eq!(config.mappings.len(), 1);
    assert!(config.mappings[0].test.contains("*"));
    assert!(config.mappings[0].targets[0].contains("*"));
}

#[test]
fn test_parse_invalid_toml() {
    let toml = "invalid { toml [";
    let result = MappingConfig::parse(toml);
    assert!(result.is_err());
}

#[test]
fn test_empty_config() {
    let config = MappingConfig::empty();
    assert!(config.mappings.is_empty());
}

// ==================== ExplicitMapper ====================

#[test]
fn test_explicit_mapper_exact_match() {
    let mut graph = CodeGraph::new();

    // Code node
    let code_id = make_node(&mut graph, "src/parser.rs", "parse", UnitType::Function, Language::Rust);

    // Test node
    let test_id = make_node(&mut graph, "tests/test_parser.rs", "test_parse", UnitType::Function, Language::Rust);

    let config = MappingConfig {
        mappings: vec![ExplicitMapping {
            test: "tests/test_parser.rs".to_string(),
            targets: vec!["src/parser.rs".to_string()],
        }],
    };

    let mapper = ExplicitMapper::new(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    assert_eq!(stats.tests_processed, 1);
    assert_eq!(stats.tests_mapped, 1);

    let mapping = &mappings[0];
    assert_eq!(mapping.test_id, test_id);
    assert!(mapping.targets.contains(&code_id));
    assert_eq!(mapping.source, MappingSource::Explicit);
}

#[test]
fn test_explicit_mapper_multiple_targets() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/a.rs", "func_a", UnitType::Function, Language::Rust);
    let code2 = make_node(&mut graph, "src/b.rs", "func_b", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test_both.rs", "test_all", UnitType::Function, Language::Rust);

    let config = MappingConfig {
        mappings: vec![ExplicitMapping {
            test: "tests/test_both.rs".to_string(),
            targets: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
        }],
    };

    let mapper = ExplicitMapper::new(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    assert_eq!(stats.tests_mapped, 1);
    assert_eq!(stats.edges_created, 2);

    let mapping = &mappings[0];
    assert!(mapping.targets.contains(&code1));
    assert!(mapping.targets.contains(&code2));
}

#[test]
fn test_explicit_mapper_no_match() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/other.rs", "other", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_nonexistent.rs", "test_it", UnitType::Function, Language::Rust);

    let config = MappingConfig {
        mappings: vec![ExplicitMapping {
            test: "tests/test_nonexistent.rs".to_string(),
            targets: vec!["src/doesnt_exist.rs".to_string()],
        }],
    };

    let mapper = ExplicitMapper::new(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    assert_eq!(stats.tests_unmapped, 1);
    assert_eq!(mappings[0].source, MappingSource::Unmapped);
}

#[test]
fn test_explicit_mapper_empty_config() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/parser.rs", "parse", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_parser.rs", "test_parse", UnitType::Function, Language::Rust);

    let config = MappingConfig::empty();
    let mapper = ExplicitMapper::new(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    assert_eq!(stats.tests_processed, 0);
    assert!(mappings.is_empty());
}

// ==================== CombinedMapper ====================

#[test]
fn test_combined_mapper_convention_only() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/helper.rs", "helper", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test_helper.rs", "test_helper", UnitType::Function, Language::Rust);

    let mapper = CombinedMapper::convention_only();
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    assert!(stats.tests_mapped >= 1);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
    assert_eq!(mapping.source, MappingSource::Convention);
}

#[test]
fn test_combined_mapper_explicit_takes_precedence() {
    let mut graph = CodeGraph::new();

    // Code nodes
    let convention_target = make_node(&mut graph, "src/func.rs", "func", UnitType::Function, Language::Rust);
    let explicit_target = make_node(&mut graph, "src/special.rs", "special", UnitType::Function, Language::Rust);

    // Test that would match "func" by convention
    let test_id = make_node(&mut graph, "tests/test_func.rs", "test_func", UnitType::Function, Language::Rust);

    // But we explicitly map it to "special"
    let config = MappingConfig {
        mappings: vec![ExplicitMapping {
            test: "tests/test_func.rs".to_string(),
            targets: vec!["src/special.rs".to_string()],
        }],
    };

    let mapper = CombinedMapper::with_explicit(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    // Should use explicit mapping, not convention
    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert_eq!(mapping.source, MappingSource::Explicit);
    assert!(mapping.targets.contains(&explicit_target));
    assert!(!mapping.targets.contains(&convention_target));
}

#[test]
fn test_combined_mapper_mixed_sources() {
    let mut graph = CodeGraph::new();

    // Code nodes
    let explicit_code = make_node(&mut graph, "src/explicit.rs", "explicit", UnitType::Function, Language::Rust);
    let convention_code = make_node(&mut graph, "src/convention.rs", "convention", UnitType::Function, Language::Rust);

    // Explicitly mapped test
    let explicit_test = make_node(&mut graph, "tests/test_explicit.rs", "test_explicit", UnitType::Function, Language::Rust);

    // Convention-mapped test
    let convention_test = make_node(&mut graph, "tests/test_convention.rs", "test_convention", UnitType::Function, Language::Rust);

    let config = MappingConfig {
        mappings: vec![ExplicitMapping {
            test: "tests/test_explicit.rs".to_string(),
            targets: vec!["src/explicit.rs".to_string()],
        }],
    };

    let mapper = CombinedMapper::with_explicit(config);
    let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));

    // Should have both explicit and convention mappings
    assert!(stats.by_source.get(&MappingSource::Explicit).copied().unwrap_or(0) >= 1);
    assert!(stats.by_source.get(&MappingSource::Convention).copied().unwrap_or(0) >= 1);

    let explicit_mapping = mappings.iter().find(|m| m.test_id == explicit_test).unwrap();
    assert_eq!(explicit_mapping.source, MappingSource::Explicit);

    let convention_mapping = mappings.iter().find(|m| m.test_id == convention_test).unwrap();
    assert_eq!(convention_mapping.source, MappingSource::Convention);
}

#[test]
fn test_combined_mapper_stats_aggregation() {
    let mut graph = CodeGraph::new();

    // 2 explicit mappings
    make_node(&mut graph, "src/a.rs", "a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/b.rs", "b", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_a.rs", "test_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_b.rs", "test_b", UnitType::Function, Language::Rust);

    // 1 convention mapping
    make_node(&mut graph, "src/c.rs", "c", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_c.rs", "test_c", UnitType::Function, Language::Rust);

    let config = MappingConfig {
        mappings: vec![
            ExplicitMapping {
                test: "tests/test_a.rs".to_string(),
                targets: vec!["src/a.rs".to_string()],
            },
            ExplicitMapping {
                test: "tests/test_b.rs".to_string(),
                targets: vec!["src/b.rs".to_string()],
            },
        ],
    };

    let mapper = CombinedMapper::with_explicit(config);
    let (_, stats) = mapper.map_tests(&graph, Path::new("."));

    // 3 tests total
    assert_eq!(stats.tests_processed, 3);
    assert_eq!(stats.tests_mapped, 3);
    assert_eq!(stats.by_source.get(&MappingSource::Explicit), Some(&2));
    assert_eq!(stats.by_source.get(&MappingSource::Convention), Some(&1));
}
