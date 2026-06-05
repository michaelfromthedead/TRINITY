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
    // Path D: impl block with trait -> Impl type
    let registry = ParserRegistry::new();
    let source = r#"
        struct Foo;
        trait Bar {}
        impl Bar for Foo {}
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Should have struct Foo, trait Bar, and impl Bar for Foo
    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();
    assert_eq!(impls.len(), 1, "should find one impl (trait impl)");
    assert_eq!(impls[0].name, "Bar for Foo");
}

#[test]
fn test_rust_parse_impl_without_trait() {
    // Path E: impl block without trait -> Impl type for inherent impl
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

    // Should have struct Foo and the inherent impl
    let structs: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .collect();
    assert_eq!(structs.len(), 1, "should find the struct");
    assert_eq!(structs[0].name, "Foo");

    // Should have impl unit for inherent impl
    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();
    assert_eq!(impls.len(), 1, "should find inherent impl");
    assert_eq!(impls[0].name, "impl Foo");
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
    // Note: enums ARE extracted now per task T-HARNESS-1.4
    let registry = ParserRegistry::new();
    let source = r#"
        use std::io;
        const VALUE: i32 = 42;
        static GLOBAL: &str = "hello";
        type Alias = String;
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // None of these are handled by extract_item
    assert!(units.is_empty(), "should ignore use/const/static/type");
}

#[test]
fn test_rust_parse_enum() {
    // Enums are now extracted per task T-HARNESS-1.4
    let registry = ParserRegistry::new();
    let source = r#"
        enum Color {
            Red,
            Green,
            Blue,
        }
    "#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1, "should find one enum");
    assert_eq!(units[0].name, "Color");
    assert_eq!(units[0].unit_type, UnitType::Enum);
}

#[test]
fn test_rust_parser_default_impl() {
    // RustParser::default() should work
    use trinity_harness::parsers::RustParser;
    let _parser = RustParser::default();
}

// =============================================================================
// Hash Computation Tests (T-HARNESS-1.4)
// =============================================================================

#[test]
fn test_rust_function_hash_computation() {
    // Verify that function hashes are computed and non-zero
    let registry = ParserRegistry::new();
    let source = r#"
fn compute_sum(a: i32, b: i32) -> i32 {
    a + b
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];

    // Full hash should be non-zero
    assert!(
        unit.hashes.full_hash != [0u8; 32],
        "full_hash should be computed"
    );
    // Signature hash should be non-zero
    assert!(
        unit.hashes.signature_hash != [0u8; 32],
        "signature_hash should be computed"
    );
    // Body hash should be non-zero
    assert!(
        unit.hashes.body_hash != [0u8; 32],
        "body_hash should be computed"
    );
    // Layout hash is not applicable for functions (should be zero)
    assert_eq!(
        unit.hashes.layout_hash,
        [0u8; 32],
        "layout_hash should be zero for functions"
    );
}

#[test]
fn test_rust_function_hash_changes_with_body() {
    // If only the body changes, body_hash and full_hash should change,
    // but signature_hash should remain the same
    let registry = ParserRegistry::new();

    let source1 = r#"
fn compute(x: i32) -> i32 {
    x + 1
}
"#;
    let source2 = r#"
fn compute(x: i32) -> i32 {
    x * 2
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    let h1 = &units1[0].hashes;
    let h2 = &units2[0].hashes;

    // Signature should be the same (same function signature)
    assert_eq!(
        h1.signature_hash, h2.signature_hash,
        "signature_hash should be identical for same signature"
    );

    // Body should differ
    assert_ne!(
        h1.body_hash, h2.body_hash,
        "body_hash should differ when body changes"
    );

    // Full hash should also differ
    assert_ne!(
        h1.full_hash, h2.full_hash,
        "full_hash should differ when body changes"
    );
}

#[test]
fn test_rust_function_hash_changes_with_signature() {
    // If signature changes, signature_hash should change
    let registry = ParserRegistry::new();

    let source1 = r#"
fn compute(x: i32) -> i32 {
    x + 1
}
"#;
    let source2 = r#"
fn compute(x: i64) -> i64 {
    x + 1
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    let h1 = &units1[0].hashes;
    let h2 = &units2[0].hashes;

    // Signature should differ
    assert_ne!(
        h1.signature_hash, h2.signature_hash,
        "signature_hash should differ when parameter type changes"
    );
}

#[test]
fn test_rust_struct_hash_computation() {
    // Verify struct hashes: full_hash, signature_hash, layout_hash
    let registry = ParserRegistry::new();
    let source = r#"
struct Point {
    x: f32,
    y: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];

    assert!(
        unit.hashes.full_hash != [0u8; 32],
        "full_hash should be computed for struct"
    );
    assert!(
        unit.hashes.signature_hash != [0u8; 32],
        "signature_hash should be computed (struct name)"
    );
    // Body hash is not applicable for structs
    assert_eq!(
        unit.hashes.body_hash,
        [0u8; 32],
        "body_hash should be zero for structs"
    );
    // Layout hash should be computed from fields
    assert!(
        unit.hashes.layout_hash != [0u8; 32],
        "layout_hash should be computed for struct with fields"
    );
}

#[test]
fn test_rust_struct_layout_hash_changes_with_fields() {
    // Adding/changing fields should change layout_hash
    let registry = ParserRegistry::new();

    let source1 = r#"
struct Point {
    x: f32,
    y: f32,
}
"#;
    let source2 = r#"
struct Point {
    x: f32,
    y: f32,
    z: f32,
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    let h1 = &units1[0].hashes;
    let h2 = &units2[0].hashes;

    // Signature should be the same (same struct name)
    assert_eq!(
        h1.signature_hash, h2.signature_hash,
        "signature_hash should be same (same struct name)"
    );

    // Layout should differ
    assert_ne!(
        h1.layout_hash, h2.layout_hash,
        "layout_hash should differ when fields change"
    );
}

#[test]
fn test_rust_struct_tuple_fields() {
    // Tuple struct should also compute layout_hash
    let registry = ParserRegistry::new();
    let source = r#"
struct Rgb(u8, u8, u8);
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert_eq!(unit.name, "Rgb");
    assert_eq!(unit.unit_type, UnitType::Struct);

    // Layout hash should be computed from tuple fields
    assert!(
        unit.hashes.layout_hash != [0u8; 32],
        "layout_hash should be computed for tuple struct"
    );
}

#[test]
fn test_rust_struct_unit() {
    // Unit struct has no fields, layout_hash should still be computed (empty)
    let registry = ParserRegistry::new();
    let source = r#"
struct Marker;
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert_eq!(unit.name, "Marker");
    assert_eq!(unit.unit_type, UnitType::Struct);
    // For unit struct, layout is empty string, hash is still computed
    assert!(
        unit.hashes.layout_hash != [0u8; 32],
        "layout_hash should be computed even for unit struct (empty layout)"
    );
}

// =============================================================================
// Line Number Extraction Tests
// =============================================================================

#[test]
fn test_rust_function_line_numbers() {
    let registry = ParserRegistry::new();
    let source = "fn first() {}\n\nfn second() {\n    // body\n}\n";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 2);

    let first = units.iter().find(|u| u.name == "first").unwrap();
    let second = units.iter().find(|u| u.name == "second").unwrap();

    // first() is on line 1
    assert_eq!(first.start_line, 1, "first() should start on line 1");
    assert_eq!(first.end_line, 1, "first() should end on line 1");

    // second() spans lines 3-5
    assert_eq!(second.start_line, 3, "second() should start on line 3");
    assert_eq!(second.end_line, 5, "second() should end on line 5");
}

#[test]
fn test_rust_struct_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Simple;

struct Complex {
    field: i32,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let simple = units.iter().find(|u| u.name == "Simple").unwrap();
    let complex = units.iter().find(|u| u.name == "Complex").unwrap();

    // Simple is on line 2
    assert_eq!(simple.start_line, 2);
    assert_eq!(simple.end_line, 2);

    // Complex spans lines 4-6
    assert_eq!(complex.start_line, 4);
    assert_eq!(complex.end_line, 6);
}

// =============================================================================
// Enum Extraction Tests
// =============================================================================

#[test]
fn test_rust_enum_hash_computation() {
    // Verify enum hashes: full_hash, signature_hash (name), layout_hash (variants)
    let registry = ParserRegistry::new();
    let source = r#"
enum Status {
    Active,
    Inactive,
    Pending,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert_eq!(unit.name, "Status");
    assert_eq!(unit.unit_type, UnitType::Enum);

    assert!(
        unit.hashes.full_hash != [0u8; 32],
        "full_hash should be computed for enum"
    );
    assert!(
        unit.hashes.signature_hash != [0u8; 32],
        "signature_hash should be computed (enum name)"
    );
    // Layout hash should contain variant names
    assert!(
        unit.hashes.layout_hash != [0u8; 32],
        "layout_hash should be computed from variants"
    );
    // Body hash not applicable for enums
    assert_eq!(
        unit.hashes.body_hash,
        [0u8; 32],
        "body_hash should be zero for enums"
    );
}

#[test]
fn test_rust_enum_layout_hash_changes_with_variants() {
    let registry = ParserRegistry::new();

    let source1 = r#"
enum Color {
    Red,
    Green,
}
"#;
    let source2 = r#"
enum Color {
    Red,
    Green,
    Blue,
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    let h1 = &units1[0].hashes;
    let h2 = &units2[0].hashes;

    // Signature same (same enum name)
    assert_eq!(h1.signature_hash, h2.signature_hash);

    // Layout differs (different variants)
    assert_ne!(
        h1.layout_hash, h2.layout_hash,
        "layout_hash should differ when variants change"
    );
}

#[test]
fn test_rust_enum_with_data_variants() {
    // Enum variants with associated data
    let registry = ParserRegistry::new();
    let source = r#"
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert_eq!(unit.name, "Message");
    assert_eq!(unit.unit_type, UnitType::Enum);

    // Should have computed hashes
    assert!(unit.hashes.full_hash != [0u8; 32]);
    assert!(unit.hashes.layout_hash != [0u8; 32]);
}

#[test]
fn test_rust_enum_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"

enum Status {
    Ok,
    Error,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];

    // Enum starts on line 3, ends on line 6
    assert_eq!(unit.start_line, 3);
    assert_eq!(unit.end_line, 6);
}

// =============================================================================
// Impl Block Extraction Tests
// =============================================================================

#[test]
fn test_rust_impl_hash_computation() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Counter;
impl Counter {
    fn new() -> Self {
        Counter
    }
    fn increment(&mut self) {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();
    assert_eq!(impls.len(), 1);

    let impl_unit = impls[0];
    assert_eq!(impl_unit.name, "impl Counter");

    // Full hash computed
    assert!(impl_unit.hashes.full_hash != [0u8; 32]);
    // Signature hash (type name)
    assert!(impl_unit.hashes.signature_hash != [0u8; 32]);
    // Body hash (method names)
    assert!(
        impl_unit.hashes.body_hash != [0u8; 32],
        "body_hash should contain method names hash"
    );
}

#[test]
fn test_rust_impl_body_hash_changes_with_methods() {
    let registry = ParserRegistry::new();

    let source1 = r#"
struct Foo;
impl Foo {
    fn method_a() {}
}
"#;
    let source2 = r#"
struct Foo;
impl Foo {
    fn method_a() {}
    fn method_b() {}
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    let impl1 = units1
        .iter()
        .find(|u| u.unit_type == UnitType::Impl)
        .unwrap();
    let impl2 = units2
        .iter()
        .find(|u| u.unit_type == UnitType::Impl)
        .unwrap();

    // Body hash should differ (different method names)
    assert_ne!(
        impl1.hashes.body_hash, impl2.hashes.body_hash,
        "body_hash should differ when methods change"
    );
}

#[test]
fn test_rust_impl_trait_signature_hash() {
    // Trait impl signature includes "Trait for Type"
    let registry = ParserRegistry::new();
    let source = r#"
struct Foo;
trait Bar {}
impl Bar for Foo {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impl_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Impl)
        .unwrap();
    assert_eq!(impl_unit.name, "Bar for Foo");

    // Signature hash should be computed
    assert!(impl_unit.hashes.signature_hash != [0u8; 32]);
}

#[test]
fn test_rust_impl_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Widget;

impl Widget {
    fn render(&self) {
        // rendering code
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impl_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Impl)
        .unwrap();

    // impl Widget starts on line 4, ends on line 8
    assert_eq!(impl_unit.start_line, 4);
    assert_eq!(impl_unit.end_line, 8);
}

// =============================================================================
// Trait Extraction Tests
// =============================================================================

#[test]
fn test_rust_parse_trait() {
    let registry = ParserRegistry::new();
    let source = r#"
trait Drawable {
    fn draw(&self);
    fn update(&mut self);
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let traits: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Trait)
        .collect();
    assert_eq!(traits.len(), 1);
    assert_eq!(traits[0].name, "Drawable");
}

#[test]
fn test_rust_trait_hash_computation() {
    let registry = ParserRegistry::new();
    let source = r#"
trait Service {
    fn start(&self);
    fn stop(&self);
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let trait_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Trait)
        .unwrap();

    // Full hash
    assert!(trait_unit.hashes.full_hash != [0u8; 32]);
    // Signature hash (trait name)
    assert!(trait_unit.hashes.signature_hash != [0u8; 32]);
    // Body hash (method names)
    assert!(
        trait_unit.hashes.body_hash != [0u8; 32],
        "body_hash should contain method signature names"
    );
    // Layout not applicable
    assert_eq!(trait_unit.hashes.layout_hash, [0u8; 32]);
}

#[test]
fn test_rust_trait_body_hash_changes_with_methods() {
    let registry = ParserRegistry::new();

    let source1 = r#"
trait Api {
    fn get(&self);
}
"#;
    let source2 = r#"
trait Api {
    fn get(&self);
    fn post(&self);
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source2, Language::Rust);

    let t1 = units1
        .iter()
        .find(|u| u.unit_type == UnitType::Trait)
        .unwrap();
    let t2 = units2
        .iter()
        .find(|u| u.unit_type == UnitType::Trait)
        .unwrap();

    // Body hash should differ
    assert_ne!(
        t1.hashes.body_hash, t2.hashes.body_hash,
        "body_hash should differ when trait methods change"
    );

    // Signature should be the same (same trait name)
    assert_eq!(t1.hashes.signature_hash, t2.hashes.signature_hash);
}

#[test]
fn test_rust_trait_with_default_impl() {
    // Traits with default implementations
    let registry = ParserRegistry::new();
    let source = r#"
trait Logger {
    fn log(&self, msg: &str);
    fn debug(&self, msg: &str) {
        self.log(&format!("[DEBUG] {}", msg));
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let trait_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Trait)
        .unwrap();
    assert_eq!(trait_unit.name, "Logger");

    // Body hash should include both methods
    assert!(trait_unit.hashes.body_hash != [0u8; 32]);
}

#[test]
fn test_rust_trait_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"

trait Validator {
    fn validate(&self) -> bool;
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let trait_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Trait)
        .unwrap();

    // Trait starts on line 3, ends on line 5
    assert_eq!(trait_unit.start_line, 3);
    assert_eq!(trait_unit.end_line, 5);
}

// =============================================================================
// Module Extraction Tests
// =============================================================================

#[test]
fn test_rust_module_hash_computation() {
    let registry = ParserRegistry::new();
    let source = r#"
mod utils {
    fn helper() {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let module = units
        .iter()
        .find(|u| u.unit_type == UnitType::Module)
        .unwrap();
    assert_eq!(module.name, "utils");

    // Full hash
    assert!(module.hashes.full_hash != [0u8; 32]);
    // Signature hash (module name)
    assert!(module.hashes.signature_hash != [0u8; 32]);
    // Body hash and layout hash not applicable for modules
    assert_eq!(module.hashes.body_hash, [0u8; 32]);
    assert_eq!(module.hashes.layout_hash, [0u8; 32]);
}

#[test]
fn test_rust_module_declaration_only() {
    // Module declaration without body (external file)
    let registry = ParserRegistry::new();
    let source = "mod external;\n";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let modules: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Module)
        .collect();
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].name, "external");
}

#[test]
fn test_rust_module_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"

mod inner {
    fn private() {}
    fn another() {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let module = units
        .iter()
        .find(|u| u.unit_type == UnitType::Module)
        .unwrap();

    // Module starts on line 3, ends on line 6
    assert_eq!(module.start_line, 3);
    assert_eq!(module.end_line, 6);
}

// =============================================================================
// Nested Items Handling Tests
// =============================================================================

#[test]
fn test_rust_nested_function_in_module() {
    // Note: current implementation does NOT recurse into module content
    // This test documents expected behavior
    let registry = ParserRegistry::new();
    let source = r#"
mod inner {
    fn nested_func() {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Only the module should be extracted, not the nested function
    // (current implementation doesn't recurse into modules)
    let modules: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Module)
        .collect();
    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(modules.len(), 1);
    // Nested functions are NOT extracted (by design - top-level only)
    assert_eq!(
        functions.len(),
        0,
        "nested functions should not be extracted at top level"
    );
}

#[test]
fn test_rust_multiple_impl_blocks_same_type() {
    // Multiple impl blocks for the same type
    let registry = ParserRegistry::new();
    let source = r#"
struct Widget;

impl Widget {
    fn new() -> Self { Widget }
}

impl Widget {
    fn render(&self) {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();

    // Both impl blocks should be extracted
    assert_eq!(impls.len(), 2);

    // Both should have the same name format
    assert!(impls.iter().all(|i| i.name == "impl Widget"));

    // But they should have different full hashes (different content)
    assert_ne!(
        impls[0].hashes.full_hash, impls[1].hashes.full_hash,
        "different impl blocks should have different full_hash"
    );

    // And different body hashes (different methods)
    assert_ne!(
        impls[0].hashes.body_hash, impls[1].hashes.body_hash,
        "different impl blocks should have different body_hash"
    );
}

#[test]
fn test_rust_impl_with_trait_and_inherent() {
    // Both trait impl and inherent impl for same type
    let registry = ParserRegistry::new();
    let source = r#"
struct Data;
trait Processor {}

impl Data {
    fn process(&self) {}
}

impl Processor for Data {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();

    assert_eq!(impls.len(), 2);

    let inherent = impls.iter().find(|i| i.name == "impl Data").unwrap();
    let trait_impl = impls.iter().find(|i| i.name == "Processor for Data").unwrap();

    assert!(inherent.hashes.full_hash != [0u8; 32]);
    assert!(trait_impl.hashes.full_hash != [0u8; 32]);
}

#[test]
fn test_rust_generic_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Container<T> {
    value: T,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert_eq!(unit.name, "Container");
    assert_eq!(unit.unit_type, UnitType::Struct);

    // Should still compute layout hash with generic type
    assert!(unit.hashes.layout_hash != [0u8; 32]);
}

#[test]
fn test_rust_generic_impl() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Wrapper<T>(T);

impl<T> Wrapper<T> {
    fn get(&self) -> &T {
        &self.0
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Impl)
        .collect();

    assert_eq!(impls.len(), 1);
    // Name extraction for generic type
    assert!(impls[0].name.contains("Wrapper"));
}

#[test]
fn test_rust_async_function() {
    let registry = ParserRegistry::new();
    let source = r#"
async fn fetch_data() -> Result<String, Error> {
    Ok("data".to_string())
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "fetch_data");
    assert_eq!(units[0].unit_type, UnitType::Function);

    // Hashes should be computed
    assert!(units[0].hashes.full_hash != [0u8; 32]);
    assert!(units[0].hashes.signature_hash != [0u8; 32]);
}

#[test]
fn test_rust_pub_visibility_modifiers() {
    // Items with different visibility should still be extracted
    let registry = ParserRegistry::new();
    let source = r#"
pub struct Public;
pub(crate) struct CrateVisible;
pub(super) fn super_fn() {}
fn private() {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 4);

    let structs: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .collect();
    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(structs.len(), 2);
    assert_eq!(functions.len(), 2);
}

#[test]
fn test_rust_attributes_on_items() {
    // Items with attributes should be extracted
    let registry = ParserRegistry::new();
    let source = r#"
#[derive(Debug, Clone)]
struct Derived {
    field: i32,
}

#[cfg(test)]
fn test_only() {}

#[inline]
fn fast() {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let structs: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .collect();
    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(structs.len(), 1);
    assert_eq!(functions.len(), 2);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_rust_empty_struct_fields() {
    // Unit struct
    let registry = ParserRegistry::new();
    let source = "struct Empty;";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    // Empty layout should still produce a hash
    assert!(units[0].hashes.layout_hash != [0u8; 32]);
}

#[test]
fn test_rust_empty_enum_no_variants() {
    // Enum with no variants (uncommon but valid)
    let registry = ParserRegistry::new();
    let source = "enum Never {}";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Never");
    // Empty variants list should produce a hash
    assert!(units[0].hashes.layout_hash != [0u8; 32]);
}

#[test]
fn test_rust_empty_impl_block() {
    // Impl block with no methods
    let registry = ParserRegistry::new();
    let source = r#"
struct Empty;
impl Empty {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impl_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Impl)
        .unwrap();

    // Body hash for empty impl should be hash of empty string
    assert!(impl_unit.hashes.body_hash != [0u8; 32]);
}

#[test]
fn test_rust_empty_trait() {
    // Trait with no methods (marker trait)
    let registry = ParserRegistry::new();
    let source = "trait Marker {}";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let trait_unit = &units[0];
    assert_eq!(trait_unit.name, "Marker");
    assert_eq!(trait_unit.unit_type, UnitType::Trait);

    // Empty body hash (hash of empty string)
    assert!(trait_unit.hashes.body_hash != [0u8; 32]);
}

#[test]
fn test_rust_function_with_complex_return_type() {
    let registry = ParserRegistry::new();
    let source = r#"
fn complex() -> Result<Vec<HashMap<String, i32>>, Box<dyn Error>> {
    Ok(vec![])
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "complex");
    assert!(units[0].hashes.signature_hash != [0u8; 32]);
}

#[test]
fn test_rust_hash_determinism() {
    // Same source should always produce the same hash
    let registry = ParserRegistry::new();
    let source = r#"
fn stable() {
    let x = 42;
}
"#;

    let units1 = registry.parse_file(Path::new("test.rs"), source, Language::Rust);
    let units2 = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units1[0].hashes.full_hash, units2[0].hashes.full_hash);
    assert_eq!(
        units1[0].hashes.signature_hash,
        units2[0].hashes.signature_hash
    );
    assert_eq!(units1[0].hashes.body_hash, units2[0].hashes.body_hash);
}
