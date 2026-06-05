//! Whitebox tests for RustParser.
//!
//! WHITEBOX coverage plan:
//!   - Path A: empty input -> empty Vec
//!   - Path B: function definition -> CodeUnit with Function type
//!   - Path C: struct definition -> CodeUnit with Struct type
//!   - Path D: impl block with trait -> CodeUnit with Method type
//!   - Path E: impl block without trait -> None (no CodeUnit)
//!   - Path F: module definition -> CodeUnit with Module type
//!   - Path G: invalid syntax -> empty Vec (parse error)
//!   - Path H: multiple items -> multiple CodeUnits

use std::path::Path;
use trinity_harness::parsers::{Language, ParserRegistry, UnitType};

#[test]
fn test_rust_parse_empty_input() {
    // Path A: empty input returns empty Vec
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("test.rs"), "", Language::Rust);
    assert!(units.is_empty(), "empty input should produce no units");
}

#[test]
fn test_rust_parse_function() {
    // Path B: function definition
    let registry = ParserRegistry::new();
    let source = r#"
        fn hello_world() {
            println!("Hello!");
        }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1, "should find one function");
    assert_eq!(units[0].name, "hello_world");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert_eq!(units[0].language, Language::Rust);
}

#[test]
fn test_rust_parse_struct() {
    // Path C: struct definition
    let registry = ParserRegistry::new();
    let source = r#"
        struct Point {
            x: f32,
            y: f32,
        }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1, "should find one struct");
    assert_eq!(units[0].name, "Point");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

#[test]
fn test_rust_parse_impl_with_trait() {
    // Path D: impl block with trait -> Method type
    let registry = ParserRegistry::new();
    let source = r#"
        struct Foo;
        trait Bar {}
        impl Bar for Foo {}
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Should have struct Foo and impl Bar
    let methods: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .collect();
    assert_eq!(methods.len(), 1, "should find one method (trait impl)");
    assert_eq!(methods[0].name, "Bar");
}

#[test]
fn test_rust_parse_impl_without_trait() {
    // Path E: impl block without trait -> no CodeUnit for the impl itself
    let registry = ParserRegistry::new();
    let source = r#"
        struct Foo;
        impl Foo {
            fn new() -> Self {
                Foo
            }
        }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Should only have the struct, not the inherent impl
    // (The impl block without trait returns None in extract_item)
    let structs: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .collect();
    assert_eq!(structs.len(), 1, "should find the struct");
    assert_eq!(structs[0].name, "Foo");

    // No method units for inherent impl
    let methods: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Method)
        .collect();
    assert!(methods.is_empty(), "inherent impl should not produce Method unit");
}

#[test]
fn test_rust_parse_module() {
    // Path F: module definition
    let registry = ParserRegistry::new();
    let source = r#"
        mod inner {
            fn private() {}
        }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let modules: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Module)
        .collect();
    assert_eq!(modules.len(), 1, "should find one module");
    assert_eq!(modules[0].name, "inner");
}

#[test]
fn test_rust_parse_invalid_syntax() {
    // Path G: invalid syntax returns empty Vec
    let registry = ParserRegistry::new();
    let source = r#"
        fn broken(
        // missing closing paren and body
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);
    assert!(units.is_empty(), "invalid syntax should produce no units");
}

#[test]
fn test_rust_parse_multiple_items() {
    // Path H: multiple items
    let registry = ParserRegistry::new();
    let source = r#"
        struct A;
        struct B;
        fn foo() {}
        fn bar() {}
        mod inner {}
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 5, "should find all 5 items");

    let structs = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .count();
    let functions = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .count();
    let modules = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Module)
        .count();

    assert_eq!(structs, 2, "should find 2 structs");
    assert_eq!(functions, 2, "should find 2 functions");
    assert_eq!(modules, 1, "should find 1 module");
}

#[test]
fn test_rust_parse_ignores_other_items() {
    // Items like use, const, static, type alias should be ignored
    let registry = ParserRegistry::new();
    let source = r#"
        use std::io;
        const VALUE: i32 = 42;
        static GLOBAL: &str = "hello";
        type Alias = String;
        enum Color { Red, Green, Blue }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // None of these are handled by extract_item
    assert!(units.is_empty(), "should ignore use/const/static/type/enum");
}

#[test]
fn test_rust_parser_default_impl() {
    // RustParser::default() should work
    use trinity_harness::parsers::RustParser;
    let _parser = RustParser::default();
}
