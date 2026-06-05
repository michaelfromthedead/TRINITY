//! Blackbox tests for RustParser via ParserRegistry.
//!
//! CLEANROOM: Tests are written based on the public contract only.
//! The implementation file (rust.rs) was NOT read.
//!
//! Test coverage plan:
//!   - T1: Basic function extraction
//!   - T2: Struct with fields extraction
//!   - T3: Enum extraction
//!   - T4: Impl block extraction (inherent and trait)
//!   - T5: Module extraction
//!   - T6: Multiple items in single file
//!   - T7: Nested structures
//!   - T8: Hash computation validation
//!   - T9: Line number accuracy
//!   - T10: Edge cases (empty, invalid syntax)
//!   - T11: Real-world code patterns
//!   - T12: Generic types and lifetimes
//!   - T13: Async functions
//!   - T14: Trait definitions
//!   - T15: Complex nested modules

use std::path::Path;
use trinity_harness::{Language, ParserRegistry, UnitType};

// ============================================================================
// T1: Basic Function Extraction
// ============================================================================

#[test]
fn blackbox_rust_simple_function() {
    let registry = ParserRegistry::new();
    let source = "fn simple() {}";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "simple");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert_eq!(units[0].language, Language::Rust);
}

#[test]
fn blackbox_rust_function_with_params() {
    let registry = ParserRegistry::new();
    let source = r#"
fn add(x: i32, y: i32) -> i32 {
    x + y
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "add");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_rust_pub_function() {
    let registry = ParserRegistry::new();
    let source = r#"
pub fn public_api() -> String {
    String::from("hello")
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "public_api");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

// ============================================================================
// T2: Struct Extraction
// ============================================================================

#[test]
fn blackbox_rust_unit_struct() {
    let registry = ParserRegistry::new();
    let source = "struct UnitStruct;";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "UnitStruct");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

#[test]
fn blackbox_rust_tuple_struct() {
    let registry = ParserRegistry::new();
    let source = "struct Pair(i32, i32);";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Pair");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

#[test]
fn blackbox_rust_struct_with_fields() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Person {
    name: String,
    age: u32,
    email: Option<String>,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Person");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

#[test]
fn blackbox_rust_generic_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Container<T> {
    value: T,
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Container");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

// ============================================================================
// T3: Enum Extraction
// ============================================================================

#[test]
fn blackbox_rust_simple_enum() {
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
    assert_eq!(units[0].name, "Status");
    assert_eq!(units[0].unit_type, UnitType::Enum);
}

#[test]
fn blackbox_rust_enum_with_data() {
    let registry = ParserRegistry::new();
    let source = r#"
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(u8, u8, u8),
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Message");
    assert_eq!(units[0].unit_type, UnitType::Enum);
}

#[test]
fn blackbox_rust_generic_enum() {
    let registry = ParserRegistry::new();
    let source = r#"
enum Result<T, E> {
    Ok(T),
    Err(E),
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Result");
    assert_eq!(units[0].unit_type, UnitType::Enum);
}

// ============================================================================
// T4: Impl Block Extraction
// ============================================================================

#[test]
fn blackbox_rust_inherent_impl() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Counter {
    value: i32,
}

impl Counter {
    fn new() -> Self {
        Counter { value: 0 }
    }

    fn increment(&mut self) {
        self.value += 1;
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Should have struct and impl
    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    let impls: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Impl).collect();

    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "Counter");

    assert_eq!(impls.len(), 1);
    assert_eq!(impls[0].name, "impl Counter");
}

#[test]
fn blackbox_rust_trait_impl() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Point {
    x: f64,
    y: f64,
}

trait Display {
    fn display(&self) -> String;
}

impl Display for Point {
    fn display(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Impl).collect();
    assert_eq!(impls.len(), 1);
    assert_eq!(impls[0].name, "Display for Point");
}

#[test]
fn blackbox_rust_multiple_impl_blocks() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Widget;

trait Drawable {
    fn draw(&self);
}

trait Clickable {
    fn click(&self);
}

impl Drawable for Widget {
    fn draw(&self) {}
}

impl Clickable for Widget {
    fn click(&self) {}
}

impl Widget {
    fn new() -> Self { Widget }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let impls: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Impl).collect();
    assert_eq!(impls.len(), 3, "should find 3 impl blocks");

    // Check that both trait impls and inherent impl are captured
    let impl_names: Vec<_> = impls.iter().map(|i| i.name.as_str()).collect();
    assert!(impl_names.contains(&"Drawable for Widget"));
    assert!(impl_names.contains(&"Clickable for Widget"));
    assert!(impl_names.contains(&"impl Widget"));
}

// ============================================================================
// T5: Module Extraction
// ============================================================================

#[test]
fn blackbox_rust_inline_module() {
    let registry = ParserRegistry::new();
    let source = r#"
mod utils {
    pub fn helper() {}
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].name, "utils");
}

#[test]
fn blackbox_rust_multiple_modules() {
    let registry = ParserRegistry::new();
    let source = r#"
mod alpha {
    fn a() {}
}

mod beta {
    fn b() {}
}

mod gamma {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    assert_eq!(modules.len(), 3);

    let names: Vec<_> = modules.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
    assert!(names.contains(&"gamma"));
}

// ============================================================================
// T6: Multiple Items in Single File
// ============================================================================

#[test]
fn blackbox_rust_mixed_items() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Data { value: i32 }

enum Mode { Fast, Slow }

fn process(d: Data, m: Mode) -> i32 {
    match m {
        Mode::Fast => d.value * 2,
        Mode::Slow => d.value,
    }
}

impl Data {
    fn new(v: i32) -> Self { Data { value: v } }
}

mod helpers {
    pub fn double(x: i32) -> i32 { x * 2 }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Count each type
    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let enums = units.iter().filter(|u| u.unit_type == UnitType::Enum).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    let impls = units.iter().filter(|u| u.unit_type == UnitType::Impl).count();
    let modules = units.iter().filter(|u| u.unit_type == UnitType::Module).count();

    assert_eq!(structs, 1, "should find 1 struct");
    assert_eq!(enums, 1, "should find 1 enum");
    assert_eq!(functions, 1, "should find 1 function");
    assert_eq!(impls, 1, "should find 1 impl");
    assert_eq!(modules, 1, "should find 1 module");
}

// ============================================================================
// T7: Nested Structures
// ============================================================================

#[test]
fn blackbox_rust_nested_module_with_items() {
    let registry = ParserRegistry::new();
    let source = r#"
mod outer {
    struct Inner;

    fn inner_fn() {}

    mod nested {
        fn deeply_nested() {}
    }
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    // Outer module should be extracted
    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    assert!(modules.len() >= 1, "should find at least outer module");
    assert!(modules.iter().any(|m| m.name == "outer"));
}

// ============================================================================
// T8: Hash Computation Validation
// ============================================================================

#[test]
fn blackbox_rust_hash_populated() {
    let registry = ParserRegistry::new();
    let source = r#"
fn compute(x: i32) -> i32 {
    x * 2
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);

    // Full hash should be non-zero (not default)
    let hashes = &units[0].hashes;
    assert_ne!(hashes.full_hash, [0u8; 32], "full_hash should be computed");
}

#[test]
fn blackbox_rust_different_functions_different_hashes() {
    let registry = ParserRegistry::new();

    let source1 = "fn foo() { let x = 1; }";
    let source2 = "fn bar() { let y = 2; }";

    let units1 = registry.parse_file(Path::new("a.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("b.rs"), source2, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Different function content should produce different hashes
    assert_ne!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "different functions should have different full hashes"
    );
}

#[test]
fn blackbox_rust_same_function_same_hash() {
    let registry = ParserRegistry::new();

    let source = "fn identical() { println!(\"hello\"); }";

    let units1 = registry.parse_file(Path::new("a.rs"), source, Language::Rust);
    let units2 = registry.parse_file(Path::new("b.rs"), source, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Same function content should produce same hash
    assert_eq!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "identical functions should have same full hash"
    );
}

// ============================================================================
// T9: Line Number Accuracy
// ============================================================================

#[test]
fn blackbox_rust_line_numbers_start_line() {
    let registry = ParserRegistry::new();
    let source = r#"
fn first() {}

fn second() {}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 2);

    let first = units.iter().find(|u| u.name == "first").unwrap();
    let second = units.iter().find(|u| u.name == "second").unwrap();

    // First function starts on line 2 (1-indexed, after empty first line)
    assert!(first.start_line >= 1, "first should start early in file");
    // Second function should start after first
    assert!(second.start_line > first.start_line, "second should start after first");
}

#[test]
fn blackbox_rust_line_numbers_end_after_start() {
    let registry = ParserRegistry::new();
    let source = r#"
fn multiline() {
    let a = 1;
    let b = 2;
    let c = a + b;
    println!("{}", c);
}
"#;
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].end_line >= units[0].start_line,
        "end_line should be >= start_line"
    );
}

// ============================================================================
// T10: Edge Cases
// ============================================================================

#[test]
fn blackbox_rust_empty_source() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.rs"), "", Language::Rust);
    assert!(units.is_empty(), "empty source should produce no units");
}

#[test]
fn blackbox_rust_whitespace_only() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("whitespace.rs"), "   \n\n\t\t\n   ", Language::Rust);
    assert!(units.is_empty(), "whitespace-only source should produce no units");
}

#[test]
fn blackbox_rust_comments_only() {
    let registry = ParserRegistry::new();
    let source = r#"
// This is a comment
/* Block comment */
/// Doc comment
//! Inner doc comment
"#;
    let units = registry.parse_file(Path::new("comments.rs"), source, Language::Rust);
    assert!(units.is_empty(), "comments-only source should produce no units");
}

#[test]
fn blackbox_rust_invalid_syntax() {
    let registry = ParserRegistry::new();
    let source = "fn broken( { let x = ";
    let units = registry.parse_file(Path::new("broken.rs"), source, Language::Rust);
    // Invalid syntax should either return empty or gracefully handle
    // The exact behavior depends on implementation, but it should not panic
    let _ = units; // Just ensure no panic occurred
}

#[test]
fn blackbox_rust_ignores_use_statements() {
    let registry = ParserRegistry::new();
    let source = r#"
use std::collections::HashMap;
use std::io::{self, Read, Write};
"#;
    let units = registry.parse_file(Path::new("imports.rs"), source, Language::Rust);
    assert!(units.is_empty(), "use statements should not produce units");
}

#[test]
fn blackbox_rust_ignores_const_and_static() {
    let registry = ParserRegistry::new();
    let source = r#"
const MAX_SIZE: usize = 100;
static GLOBAL: &str = "value";
static mut MUTABLE: i32 = 0;
"#;
    let units = registry.parse_file(Path::new("constants.rs"), source, Language::Rust);
    assert!(units.is_empty(), "const/static should not produce units");
}

#[test]
fn blackbox_rust_ignores_type_aliases() {
    let registry = ParserRegistry::new();
    let source = r#"
type MyString = String;
type Result<T> = std::result::Result<T, Error>;
"#;
    let units = registry.parse_file(Path::new("aliases.rs"), source, Language::Rust);
    assert!(units.is_empty(), "type aliases should not produce units");
}

// ============================================================================
// T11: Real-World Code Patterns
// ============================================================================

#[test]
fn blackbox_rust_builder_pattern() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Config {
    debug: bool,
    threads: usize,
}

struct ConfigBuilder {
    debug: bool,
    threads: usize,
}

impl ConfigBuilder {
    fn new() -> Self {
        ConfigBuilder {
            debug: false,
            threads: 1,
        }
    }

    fn debug(mut self, value: bool) -> Self {
        self.debug = value;
        self
    }

    fn threads(mut self, count: usize) -> Self {
        self.threads = count;
        self
    }

    fn build(self) -> Config {
        Config {
            debug: self.debug,
            threads: self.threads,
        }
    }
}
"#;
    let units = registry.parse_file(Path::new("builder.rs"), source, Language::Rust);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let impls = units.iter().filter(|u| u.unit_type == UnitType::Impl).count();

    assert_eq!(structs, 2, "should find Config and ConfigBuilder structs");
    assert_eq!(impls, 1, "should find impl ConfigBuilder");
}

#[test]
fn blackbox_rust_error_handling_pattern() {
    let registry = ParserRegistry::new();
    let source = r#"
enum Error {
    NotFound,
    PermissionDenied,
    IoError(String),
}

struct Handler;

impl Handler {
    fn handle(&self) -> Result<(), Error> {
        Ok(())
    }
}
"#;
    let units = registry.parse_file(Path::new("error.rs"), source, Language::Rust);

    assert!(units.iter().any(|u| u.name == "Error" && u.unit_type == UnitType::Enum));
    assert!(units.iter().any(|u| u.name == "Handler" && u.unit_type == UnitType::Struct));
    assert!(units.iter().any(|u| u.unit_type == UnitType::Impl));
}

// ============================================================================
// T12: Generic Types and Lifetimes
// ============================================================================

#[test]
fn blackbox_rust_generic_function() {
    let registry = ParserRegistry::new();
    let source = r#"
fn identity<T>(value: T) -> T {
    value
}
"#;
    let units = registry.parse_file(Path::new("generic.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "identity");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_rust_lifetime_function() {
    let registry = ParserRegistry::new();
    let source = r#"
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
"#;
    let units = registry.parse_file(Path::new("lifetime.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "longest");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_rust_where_clause() {
    let registry = ParserRegistry::new();
    let source = r#"
fn debug_print<T>(value: T)
where
    T: std::fmt::Debug,
{
    println!("{:?}", value);
}
"#;
    let units = registry.parse_file(Path::new("where.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "debug_print");
}

#[test]
fn blackbox_rust_struct_with_lifetime() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Ref<'a, T> {
    data: &'a T,
}
"#;
    let units = registry.parse_file(Path::new("ref.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Ref");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

// ============================================================================
// T13: Async Functions
// ============================================================================

#[test]
fn blackbox_rust_async_function() {
    let registry = ParserRegistry::new();
    let source = r#"
async fn fetch_data() -> String {
    String::from("data")
}
"#;
    let units = registry.parse_file(Path::new("async.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "fetch_data");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_rust_async_method() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Client;

impl Client {
    async fn request(&self) -> Vec<u8> {
        vec![]
    }
}
"#;
    let units = registry.parse_file(Path::new("async_impl.rs"), source, Language::Rust);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let impls = units.iter().filter(|u| u.unit_type == UnitType::Impl).count();

    assert_eq!(structs, 1);
    assert_eq!(impls, 1);
}

// ============================================================================
// T14: Trait Definitions
// ============================================================================

#[test]
fn blackbox_rust_simple_trait() {
    let registry = ParserRegistry::new();
    let source = r#"
trait Drawable {
    fn draw(&self);
}
"#;
    let units = registry.parse_file(Path::new("trait.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Drawable");
    assert_eq!(units[0].unit_type, UnitType::Trait);
}

#[test]
fn blackbox_rust_trait_with_default() {
    let registry = ParserRegistry::new();
    let source = r#"
trait Logger {
    fn log(&self, message: &str);

    fn debug(&self, message: &str) {
        self.log(&format!("[DEBUG] {}", message));
    }
}
"#;
    let units = registry.parse_file(Path::new("trait_default.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Logger");
    assert_eq!(units[0].unit_type, UnitType::Trait);
}

#[test]
fn blackbox_rust_generic_trait() {
    let registry = ParserRegistry::new();
    let source = r#"
trait Iterator<Item> {
    fn next(&mut self) -> Option<Item>;
}
"#;
    let units = registry.parse_file(Path::new("generic_trait.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Iterator");
    assert_eq!(units[0].unit_type, UnitType::Trait);
}

// ============================================================================
// T15: Complex Nested Modules
// ============================================================================

#[test]
fn blackbox_rust_deeply_nested_module() {
    let registry = ParserRegistry::new();
    let source = r#"
mod level1 {
    pub mod level2 {
        pub mod level3 {
            pub fn deep_function() {}
        }
    }
}
"#;
    let units = registry.parse_file(Path::new("nested.rs"), source, Language::Rust);

    // At minimum, the top-level module should be found
    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    assert!(!modules.is_empty(), "should find at least one module");
    assert!(modules.iter().any(|m| m.name == "level1"));
}

#[test]
fn blackbox_rust_module_with_all_item_types() {
    let registry = ParserRegistry::new();
    let source = r#"
mod complete {
    struct Data { value: i32 }

    enum State { On, Off }

    trait Runnable {
        fn run(&self);
    }

    impl Runnable for Data {
        fn run(&self) {}
    }

    fn standalone() {}
}
"#;
    let units = registry.parse_file(Path::new("complete_module.rs"), source, Language::Rust);

    // Verify module exists
    assert!(units.iter().any(|u| u.name == "complete" && u.unit_type == UnitType::Module));
}

// ============================================================================
// Language Detection (via ParserRegistry)
// ============================================================================

#[test]
fn blackbox_rust_language_detection() {
    assert_eq!(
        ParserRegistry::detect_language(Path::new("test.rs")),
        Some(Language::Rust)
    );
    assert_eq!(
        ParserRegistry::detect_language(Path::new("/path/to/module.rs")),
        Some(Language::Rust)
    );
    assert_eq!(
        ParserRegistry::detect_language(Path::new("lib.rs")),
        Some(Language::Rust)
    );
}

#[test]
fn blackbox_rust_language_returned_correct() {
    let registry = ParserRegistry::new();
    let source = "fn test() {}";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].language, Language::Rust);
}

// ============================================================================
// ParserRegistry API Tests
// ============================================================================

#[test]
fn blackbox_rust_registry_default() {
    // ParserRegistry should implement Default
    let registry = ParserRegistry::default();
    let units = registry.parse_file(Path::new("test.rs"), "fn x() {}", Language::Rust);
    assert_eq!(units.len(), 1);
}

#[test]
fn blackbox_rust_parser_reusable() {
    // Same registry should be usable for multiple parse calls
    let registry = ParserRegistry::new();

    let units1 = registry.parse_file(Path::new("a.rs"), "fn a() {}", Language::Rust);
    let units2 = registry.parse_file(Path::new("b.rs"), "fn b() {}", Language::Rust);
    let units3 = registry.parse_file(Path::new("c.rs"), "fn c() {}", Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_eq!(units3.len(), 1);

    assert_eq!(units1[0].name, "a");
    assert_eq!(units2[0].name, "b");
    assert_eq!(units3[0].name, "c");
}

// ============================================================================
// Attribute Handling
// ============================================================================

#[test]
fn blackbox_rust_function_with_attributes() {
    let registry = ParserRegistry::new();
    let source = r#"
#[inline]
#[must_use]
fn fast_compute(x: i32) -> i32 {
    x * 2
}

#[cfg(test)]
fn test_only() {}

#[allow(dead_code)]
fn unused() {}
"#;
    let units = registry.parse_file(Path::new("attrs.rs"), source, Language::Rust);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 3, "should find all 3 functions despite attributes");
}

#[test]
fn blackbox_rust_derive_macro_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
#[derive(Debug, Clone, PartialEq)]
struct Derived {
    field: String,
}
"#;
    let units = registry.parse_file(Path::new("derive.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Derived");
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

// ============================================================================
// Visibility Modifiers
// ============================================================================

#[test]
fn blackbox_rust_various_visibility() {
    let registry = ParserRegistry::new();
    let source = r#"
pub struct Public;
pub(crate) struct CrateVisible;
pub(super) struct SuperVisible;
pub(in crate::module) struct PathVisible;
struct Private;
"#;
    let units = registry.parse_file(Path::new("visibility.rs"), source, Language::Rust);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 5, "should find all structs regardless of visibility");
}

// ============================================================================
// Macro-Related Edge Cases
// ============================================================================

#[test]
fn blackbox_rust_macro_invocations_ignored() {
    let registry = ParserRegistry::new();
    let source = r#"
println!("Hello");
vec![1, 2, 3];
assert_eq!(1, 1);
"#;
    let units = registry.parse_file(Path::new("macros.rs"), source, Language::Rust);

    // Macro invocations are not code units
    assert!(units.is_empty(), "macro invocations should not produce units");
}

#[test]
fn blackbox_rust_function_with_macro_body() {
    let registry = ParserRegistry::new();
    let source = r#"
fn with_macros() {
    println!("debug");
    let v = vec![1, 2, 3];
    assert!(!v.is_empty());
}
"#;
    let units = registry.parse_file(Path::new("macro_body.rs"), source, Language::Rust);

    // Function itself should be extracted, macros in body don't matter
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "with_macros");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

// ============================================================================
// Comprehensive Real-World File
// ============================================================================

#[test]
fn blackbox_rust_realistic_module() {
    let registry = ParserRegistry::new();
    let source = r#"
//! Module documentation

use std::collections::HashMap;

const VERSION: &str = "1.0.0";

/// A configuration struct.
#[derive(Debug, Clone)]
pub struct Config {
    name: String,
    options: HashMap<String, String>,
}

/// Error types for this module.
#[derive(Debug)]
pub enum ConfigError {
    NotFound(String),
    InvalidValue { key: String, value: String },
}

/// Trait for configurable items.
pub trait Configurable {
    fn configure(&mut self, config: &Config) -> Result<(), ConfigError>;
}

impl Config {
    /// Create a new Config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Config {
            name: name.into(),
            options: HashMap::new(),
        }
    }

    /// Set an option.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.options.insert(key.into(), value.into());
    }

    /// Get an option value.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.options.get(key)
    }
}

/// Load configuration from environment.
pub fn load_from_env(prefix: &str) -> Config {
    Config::new(prefix)
}

mod internal {
    fn helper() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Config {
        Config::new("test")
    }
}
"#;
    let units = registry.parse_file(Path::new("realistic.rs"), source, Language::Rust);

    // Verify we find the expected items
    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    let enums: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Enum).collect();
    let traits: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Trait).collect();
    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    let impls: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Impl).collect();
    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();

    // Should find Config struct
    assert!(structs.iter().any(|s| s.name == "Config"), "should find Config struct");

    // Should find ConfigError enum
    assert!(enums.iter().any(|e| e.name == "ConfigError"), "should find ConfigError enum");

    // Should find Configurable trait
    assert!(traits.iter().any(|t| t.name == "Configurable"), "should find Configurable trait");

    // Should find load_from_env function
    assert!(functions.iter().any(|f| f.name == "load_from_env"), "should find load_from_env function");

    // Should find impl Config
    assert!(!impls.is_empty(), "should find impl blocks");

    // Should find modules (internal, tests)
    assert!(modules.len() >= 2, "should find internal and tests modules");
}
