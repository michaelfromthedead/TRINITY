//! WGSL Preprocessor for TRINITY shader compilation.
//!
//! Implements a C-style preprocessor for WGSL shader source files supporting:
//!
//! - `#define NAME [value]` and `#undef NAME`
//! - `#if`, `#elif`, `#else`, `#endif` conditionals with integer expressions
//! - `#ifdef` and `#ifndef` for macro existence checks
//! - `#include "path"` with configurable search paths
//! - `#warning` and `#error` directives
//! - Predefined macros: `TRINITY_VERSION`, `TRINITY_RHI_VULKAN/D3D12/METAL`
//! - Dependency extraction for incremental compilation
//! - Serializable preprocessor state for deterministic recompilation
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::WgslPreprocessor;
//!
//! let mut pp = WgslPreprocessor::new();
//! pp.add_search_path("shaders/");
//! pp.define("ENABLE_SHADOWS", Some("1"));
//!
//! let source = r#"
//! #ifdef ENABLE_SHADOWS
//! @group(0) @binding(0) var shadow_map: texture_depth_2d;
//! #endif
//! "#;
//!
//! let result = pp.preprocess(source, "main.wgsl")?;
//! println!("Processed: {}", result.output);
//! println!("Dependencies: {:?}", result.dependencies);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    /// `#error` directive was encountered.
    ErrorDirective { message: String, file: String, line: usize },
    /// Circular include detected.
    CircularInclude { path: String, include_stack: Vec<String> },
    /// Include file not found.
    IncludeNotFound { path: String, search_paths: Vec<String>, file: String, line: usize },
    /// Invalid preprocessor directive syntax.
    InvalidSyntax { directive: String, message: String, file: String, line: usize },
    /// Unbalanced conditional blocks.
    UnbalancedConditional { directive: String, file: String, line: usize },
    /// Invalid expression in `#if`/`#elif`.
    InvalidExpression { expression: String, message: String, file: String, line: usize },
    /// IO error reading include file.
    IoError { path: String, message: String },
    /// Undefined macro in expression.
    UndefinedMacro { name: String, file: String, line: usize },
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreprocessError::ErrorDirective { message, file, line } => {
                write!(f, "{}:{}: error: {}", file, line, message)
            }
            PreprocessError::CircularInclude { path, include_stack } => {
                write!(
                    f,
                    "circular include detected: {} (include stack: {})",
                    path,
                    include_stack.join(" -> ")
                )
            }
            PreprocessError::IncludeNotFound { path, search_paths, file, line } => {
                write!(
                    f,
                    "{}:{}: include not found: '{}' (searched: {:?})",
                    file, line, path, search_paths
                )
            }
            PreprocessError::InvalidSyntax { directive, message, file, line } => {
                write!(f, "{}:{}: invalid syntax in '{}': {}", file, line, directive, message)
            }
            PreprocessError::UnbalancedConditional { directive, file, line } => {
                write!(f, "{}:{}: unbalanced conditional: {}", file, line, directive)
            }
            PreprocessError::InvalidExpression { expression, message, file, line } => {
                write!(
                    f,
                    "{}:{}: invalid expression '{}': {}",
                    file, line, expression, message
                )
            }
            PreprocessError::IoError { path, message } => {
                write!(f, "IO error reading '{}': {}", path, message)
            }
            PreprocessError::UndefinedMacro { name, file, line } => {
                write!(f, "{}:{}: undefined macro: {}", file, line, name)
            }
        }
    }
}

impl std::error::Error for PreprocessError {}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of preprocessing a WGSL shader.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// The preprocessed WGSL source.
    pub output: String,
    /// List of all included file paths (for dependency tracking).
    pub dependencies: Vec<String>,
    /// Warnings generated during preprocessing.
    pub warnings: Vec<String>,
}

/// RHI backend type for predefined macros.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RhiBackend {
    Vulkan,
    D3D12,
    Metal,
    WebGpu,
}

impl RhiBackend {
    /// Get the macro name for this backend.
    pub fn macro_name(&self) -> &'static str {
        match self {
            RhiBackend::Vulkan => "TRINITY_RHI_VULKAN",
            RhiBackend::D3D12 => "TRINITY_RHI_D3D12",
            RhiBackend::Metal => "TRINITY_RHI_METAL",
            RhiBackend::WebGpu => "TRINITY_RHI_WEBGPU",
        }
    }
}

/// Serializable preprocessor state for deterministic recompilation.
///
/// This captures all the state needed to reproduce a preprocessing run:
/// - User-defined macros
/// - Search paths
/// - Backend selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessorState {
    /// User-defined macros with optional values.
    pub defines: HashMap<String, Option<String>>,
    /// Include search paths.
    pub search_paths: Vec<String>,
    /// Selected RHI backend.
    pub backend: RhiBackend,
    /// TRINITY version (major, minor, patch).
    pub version: (u32, u32, u32),
}

impl Default for PreprocessorState {
    fn default() -> Self {
        Self {
            defines: HashMap::new(),
            search_paths: Vec::new(),
            backend: RhiBackend::Vulkan,
            version: (0, 1, 0),
        }
    }
}

impl PreprocessorState {
    /// Create a new preprocessor state with the given backend.
    pub fn with_backend(backend: RhiBackend) -> Self {
        Self { backend, ..Default::default() }
    }

    /// Serialize to JSON for caching.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Conditional State
// ---------------------------------------------------------------------------

/// State for tracking nested conditional blocks.
#[derive(Debug, Clone)]
struct ConditionalState {
    /// Whether this block is active (should emit output).
    active: bool,
    /// Whether any branch in this conditional chain has been taken.
    any_branch_taken: bool,
    /// Whether we're in an else block (no more elif allowed).
    in_else: bool,
    /// File where this conditional started.
    file: String,
    /// Line where this conditional started.
    start_line: usize,
}

// ---------------------------------------------------------------------------
// Expression Evaluator
// ---------------------------------------------------------------------------

/// Simple expression evaluator for `#if` / `#elif` conditions.
/// Supports: integer literals, defined(NAME), &&, ||, !, ==, !=, <, >, <=, >=, (, )
struct ExpressionEvaluator<'a> {
    defines: &'a HashMap<String, Option<String>>,
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(i64),
    Identifier(String),
    Defined,
    LParen,
    RParen,
    And,
    Or,
    Not,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Plus,
    Minus,
    Eof,
}

impl<'a> ExpressionEvaluator<'a> {
    fn new(defines: &'a HashMap<String, Option<String>>) -> Self {
        Self { defines, tokens: Vec::new(), pos: 0 }
    }

    fn tokenize(&mut self, expr: &str) -> Result<(), String> {
        let mut chars = expr.chars().peekable();
        self.tokens.clear();
        self.pos = 0;

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\t' | '\r' | '\n' => {
                    chars.next();
                }
                '0'..='9' => {
                    let mut num = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    let value = num.parse::<i64>().map_err(|e| e.to_string())?;
                    self.tokens.push(Token::Number(value));
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_alphanumeric() || c == '_' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if ident == "defined" {
                        self.tokens.push(Token::Defined);
                    } else {
                        self.tokens.push(Token::Identifier(ident));
                    }
                }
                '(' => {
                    self.tokens.push(Token::LParen);
                    chars.next();
                }
                ')' => {
                    self.tokens.push(Token::RParen);
                    chars.next();
                }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        self.tokens.push(Token::And);
                    } else {
                        return Err("expected '&&'".to_string());
                    }
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        self.tokens.push(Token::Or);
                    } else {
                        return Err("expected '||'".to_string());
                    }
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Ne);
                    } else {
                        self.tokens.push(Token::Not);
                    }
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Eq);
                    } else {
                        return Err("expected '=='".to_string());
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Le);
                    } else {
                        self.tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Ge);
                    } else {
                        self.tokens.push(Token::Gt);
                    }
                }
                '+' => {
                    self.tokens.push(Token::Plus);
                    chars.next();
                }
                '-' => {
                    self.tokens.push(Token::Minus);
                    chars.next();
                }
                _ => return Err(format!("unexpected character: '{}'", c)),
            }
        }

        self.tokens.push(Token::Eof);
        Ok(())
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn evaluate(&mut self, expr: &str) -> Result<i64, String> {
        self.tokenize(expr)?;
        let result = self.parse_or()?;
        if *self.current() != Token::Eof {
            return Err(format!("unexpected token: {:?}", self.current()));
        }
        Ok(result)
    }

    fn parse_or(&mut self) -> Result<i64, String> {
        let mut left = self.parse_and()?;
        while *self.current() == Token::Or {
            self.advance();
            let right = self.parse_and()?;
            left = if left != 0 || right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<i64, String> {
        let mut left = self.parse_comparison()?;
        while *self.current() == Token::And {
            self.advance();
            let right = self.parse_comparison()?;
            left = if left != 0 && right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<i64, String> {
        let left = self.parse_additive()?;
        match self.current().clone() {
            Token::Eq => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left == right { 1 } else { 0 })
            }
            Token::Ne => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left != right { 1 } else { 0 })
            }
            Token::Lt => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left < right { 1 } else { 0 })
            }
            Token::Gt => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left > right { 1 } else { 0 })
            }
            Token::Le => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left <= right { 1 } else { 0 })
            }
            Token::Ge => {
                self.advance();
                let right = self.parse_additive()?;
                Ok(if left >= right { 1 } else { 0 })
            }
            _ => Ok(left),
        }
    }

    fn parse_additive(&mut self) -> Result<i64, String> {
        let mut left = self.parse_unary()?;
        loop {
            match self.current().clone() {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = left.wrapping_add(right);
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = left.wrapping_sub(right);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<i64, String> {
        match self.current().clone() {
            Token::Not => {
                self.advance();
                let val = self.parse_unary()?;
                Ok(if val == 0 { 1 } else { 0 })
            }
            Token::Minus => {
                self.advance();
                let val = self.parse_unary()?;
                Ok(-val)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<i64, String> {
        match self.current().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(n)
            }
            Token::Defined => {
                self.advance();
                // defined(NAME) or defined NAME
                let has_paren = *self.current() == Token::LParen;
                if has_paren {
                    self.advance();
                }
                let name = match self.current().clone() {
                    Token::Identifier(name) => name,
                    _ => return Err("expected identifier after 'defined'".to_string()),
                };
                self.advance();
                if has_paren {
                    if *self.current() != Token::RParen {
                        return Err("expected ')' after defined(NAME".to_string());
                    }
                    self.advance();
                }
                Ok(if self.defines.contains_key(&name) { 1 } else { 0 })
            }
            Token::Identifier(name) => {
                self.advance();
                // Look up macro value, default to 0 if not defined or no value
                Ok(if let Some(value) = self.defines.get(&name) {
                    if let Some(v) = value {
                        v.parse::<i64>().unwrap_or(0)
                    } else {
                        1 // Defined but no value = 1
                    }
                } else {
                    0 // Not defined = 0
                })
            }
            Token::LParen => {
                self.advance();
                let val = self.parse_or()?;
                if *self.current() != Token::RParen {
                    return Err("expected ')'".to_string());
                }
                self.advance();
                Ok(val)
            }
            _ => Err(format!("unexpected token in expression: {:?}", self.current())),
        }
    }
}

// ---------------------------------------------------------------------------
// WgslPreprocessor
// ---------------------------------------------------------------------------

/// WGSL preprocessor supporting C-style directives.
///
/// # Features
///
/// - `#define NAME [value]` - Define a macro
/// - `#undef NAME` - Undefine a macro
/// - `#ifdef NAME` / `#ifndef NAME` - Conditional based on macro existence
/// - `#if expr` / `#elif expr` / `#else` / `#endif` - General conditionals
/// - `#include "path"` - Include another file
/// - `#warning message` - Emit a warning
/// - `#error message` - Emit an error (stops processing)
///
/// # Predefined Macros
///
/// - `TRINITY_VERSION` - Version as integer (major * 10000 + minor * 100 + patch)
/// - `TRINITY_RHI_VULKAN` - Defined when targeting Vulkan
/// - `TRINITY_RHI_D3D12` - Defined when targeting D3D12
/// - `TRINITY_RHI_METAL` - Defined when targeting Metal
/// - `TRINITY_RHI_WEBGPU` - Defined when targeting WebGPU
pub struct WgslPreprocessor {
    /// User-defined macros.
    defines: HashMap<String, Option<String>>,
    /// Include search paths.
    search_paths: Vec<PathBuf>,
    /// Current RHI backend.
    backend: RhiBackend,
    /// TRINITY version.
    version: (u32, u32, u32),
    /// File reader function (for testing/mocking).
    #[allow(clippy::type_complexity)]
    file_reader: Option<std::sync::Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>>,
}

impl std::fmt::Debug for WgslPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgslPreprocessor")
            .field("defines", &self.defines)
            .field("search_paths", &self.search_paths)
            .field("backend", &self.backend)
            .field("version", &self.version)
            .field("file_reader", &self.file_reader.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl Clone for WgslPreprocessor {
    fn clone(&self) -> Self {
        Self {
            defines: self.defines.clone(),
            search_paths: self.search_paths.clone(),
            backend: self.backend,
            version: self.version,
            file_reader: self.file_reader.clone(),
        }
    }
}

impl Default for WgslPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl WgslPreprocessor {
    /// Create a new preprocessor with default settings.
    pub fn new() -> Self {
        Self {
            defines: HashMap::new(),
            search_paths: Vec::new(),
            backend: RhiBackend::Vulkan,
            version: (0, 1, 0),
            file_reader: None,
        }
    }

    /// Create a preprocessor from serialized state.
    pub fn from_state(state: &PreprocessorState) -> Self {
        let mut pp = Self::new();
        for (k, v) in &state.defines {
            pp.defines.insert(k.clone(), v.clone());
        }
        for path in &state.search_paths {
            pp.search_paths.push(PathBuf::from(path));
        }
        pp.backend = state.backend;
        pp.version = state.version;
        pp
    }

    /// Export current state for serialization.
    pub fn to_state(&self) -> PreprocessorState {
        PreprocessorState {
            defines: self.defines.clone(),
            search_paths: self.search_paths.iter().map(|p| p.to_string_lossy().to_string()).collect(),
            backend: self.backend,
            version: self.version,
        }
    }

    /// Set the RHI backend (affects predefined macros).
    pub fn set_backend(&mut self, backend: RhiBackend) {
        self.backend = backend;
    }

    /// Get the current RHI backend.
    pub fn backend(&self) -> RhiBackend {
        self.backend
    }

    /// Set the TRINITY version.
    pub fn set_version(&mut self, major: u32, minor: u32, patch: u32) {
        self.version = (major, minor, patch);
    }

    /// Define a macro with an optional value.
    pub fn define(&mut self, name: &str, value: Option<&str>) {
        self.defines.insert(name.to_string(), value.map(|s| s.to_string()));
    }

    /// Undefine a macro.
    pub fn undef(&mut self, name: &str) {
        self.defines.remove(name);
    }

    /// Check if a macro is defined.
    pub fn is_defined(&self, name: &str) -> bool {
        self.defines.contains_key(name) || self.is_predefined(name)
    }

    /// Add a search path for includes.
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Set a custom file reader (for testing).
    pub fn set_file_reader<F>(&mut self, reader: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        self.file_reader = Some(std::sync::Arc::new(reader));
    }

    /// Check if a macro is predefined.
    fn is_predefined(&self, name: &str) -> bool {
        name == "TRINITY_VERSION" || name == self.backend.macro_name()
    }

    /// Get the value of a predefined macro.
    fn predefined_value(&self, name: &str) -> Option<String> {
        if name == "TRINITY_VERSION" {
            let (major, minor, patch) = self.version;
            Some(format!("{}", major * 10000 + minor * 100 + patch))
        } else if name == self.backend.macro_name() {
            Some("1".to_string())
        } else {
            None
        }
    }

    /// Get all defined macros including predefined ones.
    fn all_defines(&self) -> HashMap<String, Option<String>> {
        let mut all = self.defines.clone();
        // Add predefined macros
        all.insert("TRINITY_VERSION".to_string(), self.predefined_value("TRINITY_VERSION"));
        all.insert(self.backend.macro_name().to_string(), Some("1".to_string()));
        all
    }

    /// Read a file from the filesystem or custom reader.
    fn read_file(&self, path: &str) -> Result<String, PreprocessError> {
        if let Some(ref reader) = self.file_reader {
            reader(path).map_err(|msg| PreprocessError::IoError {
                path: path.to_string(),
                message: msg,
            })
        } else {
            std::fs::read_to_string(path).map_err(|e| PreprocessError::IoError {
                path: path.to_string(),
                message: e.to_string(),
            })
        }
    }

    /// Resolve an include path using search paths.
    fn resolve_include(
        &self,
        include_path: &str,
        current_file: &str,
        line: usize,
    ) -> Result<String, PreprocessError> {
        // First, try relative to current file
        if let Some(parent) = Path::new(current_file).parent() {
            let relative = parent.join(include_path);
            if self.file_exists(&relative.to_string_lossy()) {
                return Ok(relative.to_string_lossy().to_string());
            }
        }

        // Then try search paths
        for search_path in &self.search_paths {
            let full_path = search_path.join(include_path);
            if self.file_exists(&full_path.to_string_lossy()) {
                return Ok(full_path.to_string_lossy().to_string());
            }
        }

        Err(PreprocessError::IncludeNotFound {
            path: include_path.to_string(),
            search_paths: self.search_paths.iter().map(|p| p.to_string_lossy().to_string()).collect(),
            file: current_file.to_string(),
            line,
        })
    }

    /// Check if a file exists.
    fn file_exists(&self, path: &str) -> bool {
        if let Some(ref reader) = self.file_reader {
            reader(path).is_ok()
        } else {
            Path::new(path).exists()
        }
    }

    /// Preprocess a WGSL source string.
    pub fn preprocess(
        &self,
        source: &str,
        file: &str,
    ) -> Result<PreprocessResult, PreprocessError> {
        let mut context = PreprocessContext {
            preprocessor: self,
            output: String::new(),
            dependencies: Vec::new(),
            warnings: Vec::new(),
            include_stack: vec![file.to_string()],
            defines: self.all_defines(),
            conditional_stack: Vec::new(),
        };

        context.process_source(source, file)?;

        // Check for unbalanced conditionals
        if let Some(cond) = context.conditional_stack.last() {
            return Err(PreprocessError::UnbalancedConditional {
                directive: "#if/#ifdef/#ifndef".to_string(),
                file: cond.file.clone(),
                line: cond.start_line,
            });
        }

        Ok(PreprocessResult {
            output: context.output,
            dependencies: context.dependencies,
            warnings: context.warnings,
        })
    }

    /// Preprocess and perform macro substitution in the output.
    pub fn preprocess_with_substitution(
        &self,
        source: &str,
        file: &str,
    ) -> Result<PreprocessResult, PreprocessError> {
        let mut result = self.preprocess(source, file)?;

        // Perform macro substitution on the output
        let defines = self.all_defines();
        for (name, value) in &defines {
            if let Some(v) = value {
                result.output = result.output.replace(name, v);
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// PreprocessContext
// ---------------------------------------------------------------------------

/// Internal context for preprocessing a file.
struct PreprocessContext<'a> {
    preprocessor: &'a WgslPreprocessor,
    output: String,
    dependencies: Vec<String>,
    warnings: Vec<String>,
    include_stack: Vec<String>,
    defines: HashMap<String, Option<String>>,
    conditional_stack: Vec<ConditionalState>,
}

impl<'a> PreprocessContext<'a> {
    /// Check if output should be emitted (all conditionals are active).
    fn is_active(&self) -> bool {
        self.conditional_stack.iter().all(|c| c.active)
    }

    /// Process a source string.
    fn process_source(&mut self, source: &str, file: &str) -> Result<(), PreprocessError> {
        for (line_num, line) in source.lines().enumerate() {
            let line_num = line_num + 1; // 1-indexed
            let trimmed = line.trim();

            if trimmed.starts_with('#') {
                self.process_directive(trimmed, file, line_num)?;
            } else if self.is_active() {
                // Emit line with macro substitution
                let mut output_line = line.to_string();
                for (name, value) in &self.defines {
                    if let Some(v) = value {
                        // Only substitute whole words
                        output_line = substitute_macro(&output_line, name, v);
                    }
                }
                self.output.push_str(&output_line);
                self.output.push('\n');
            }
        }
        Ok(())
    }

    /// Process a preprocessor directive.
    fn process_directive(
        &mut self,
        line: &str,
        file: &str,
        line_num: usize,
    ) -> Result<(), PreprocessError> {
        let line = line.trim_start_matches('#').trim();

        // Parse directive name
        let (directive, rest) = match line.find(|c: char| c.is_whitespace()) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };

        match directive {
            "define" => {
                if self.is_active() {
                    self.handle_define(rest, file, line_num)?;
                }
            }
            "undef" => {
                if self.is_active() {
                    self.handle_undef(rest, file, line_num)?;
                }
            }
            "ifdef" => {
                self.handle_ifdef(rest, file, line_num, false)?;
            }
            "ifndef" => {
                self.handle_ifdef(rest, file, line_num, true)?;
            }
            "if" => {
                self.handle_if(rest, file, line_num)?;
            }
            "elif" => {
                self.handle_elif(rest, file, line_num)?;
            }
            "else" => {
                self.handle_else(file, line_num)?;
            }
            "endif" => {
                self.handle_endif(file, line_num)?;
            }
            "include" => {
                if self.is_active() {
                    self.handle_include(rest, file, line_num)?;
                }
            }
            "warning" => {
                if self.is_active() {
                    self.warnings.push(format!("{}:{}: warning: {}", file, line_num, rest));
                }
            }
            "error" => {
                if self.is_active() {
                    return Err(PreprocessError::ErrorDirective {
                        message: rest.to_string(),
                        file: file.to_string(),
                        line: line_num,
                    });
                }
            }
            _ => {
                return Err(PreprocessError::InvalidSyntax {
                    directive: format!("#{}", directive),
                    message: "unknown directive".to_string(),
                    file: file.to_string(),
                    line: line_num,
                });
            }
        }

        Ok(())
    }

    fn handle_define(&mut self, rest: &str, file: &str, line: usize) -> Result<(), PreprocessError> {
        let parts: Vec<&str> = rest.splitn(2, char::is_whitespace).collect();
        if parts.is_empty() || parts[0].is_empty() {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#define".to_string(),
                message: "expected macro name".to_string(),
                file: file.to_string(),
                line,
            });
        }

        let name = parts[0].to_string();
        let value = parts.get(1).map(|s| s.trim().to_string());
        self.defines.insert(name, value);
        Ok(())
    }

    fn handle_undef(&mut self, rest: &str, file: &str, line: usize) -> Result<(), PreprocessError> {
        let name = rest.trim();
        if name.is_empty() {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#undef".to_string(),
                message: "expected macro name".to_string(),
                file: file.to_string(),
                line,
            });
        }
        self.defines.remove(name);
        Ok(())
    }

    fn handle_ifdef(
        &mut self,
        rest: &str,
        file: &str,
        line: usize,
        negate: bool,
    ) -> Result<(), PreprocessError> {
        let name = rest.trim();
        if name.is_empty() {
            return Err(PreprocessError::InvalidSyntax {
                directive: if negate { "#ifndef" } else { "#ifdef" }.to_string(),
                message: "expected macro name".to_string(),
                file: file.to_string(),
                line,
            });
        }

        let defined = self.defines.contains_key(name);
        let condition = if negate { !defined } else { defined };
        let active = self.is_active() && condition;

        self.conditional_stack.push(ConditionalState {
            active,
            any_branch_taken: active,
            in_else: false,
            file: file.to_string(),
            start_line: line,
        });

        Ok(())
    }

    fn handle_if(&mut self, rest: &str, file: &str, line: usize) -> Result<(), PreprocessError> {
        let expr = rest.trim();
        if expr.is_empty() {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#if".to_string(),
                message: "expected expression".to_string(),
                file: file.to_string(),
                line,
            });
        }

        let condition = self.evaluate_expression(expr, file, line)?;
        let active = self.is_active() && condition;

        self.conditional_stack.push(ConditionalState {
            active,
            any_branch_taken: active,
            in_else: false,
            file: file.to_string(),
            start_line: line,
        });

        Ok(())
    }

    fn handle_elif(&mut self, rest: &str, file: &str, line: usize) -> Result<(), PreprocessError> {
        let cond = self.conditional_stack.last_mut().ok_or_else(|| {
            PreprocessError::UnbalancedConditional {
                directive: "#elif".to_string(),
                file: file.to_string(),
                line,
            }
        })?;

        if cond.in_else {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#elif".to_string(),
                message: "#elif after #else".to_string(),
                file: file.to_string(),
                line,
            });
        }

        let expr = rest.trim();
        if expr.is_empty() {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#elif".to_string(),
                message: "expected expression".to_string(),
                file: file.to_string(),
                line,
            });
        }

        // Check if parent conditionals are active
        let parent_active = self.conditional_stack.len() <= 1
            || self.conditional_stack[..self.conditional_stack.len() - 1]
                .iter()
                .all(|c| c.active);

        let condition = self.evaluate_expression(expr, file, line)?;

        // Only activate if no branch taken yet and parent is active
        let cond = self.conditional_stack.last_mut().unwrap();
        cond.active = parent_active && !cond.any_branch_taken && condition;
        if cond.active {
            cond.any_branch_taken = true;
        }

        Ok(())
    }

    fn handle_else(&mut self, file: &str, line: usize) -> Result<(), PreprocessError> {
        let cond = self.conditional_stack.last_mut().ok_or_else(|| {
            PreprocessError::UnbalancedConditional {
                directive: "#else".to_string(),
                file: file.to_string(),
                line,
            }
        })?;

        if cond.in_else {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#else".to_string(),
                message: "duplicate #else".to_string(),
                file: file.to_string(),
                line,
            });
        }

        // Check if parent conditionals are active
        let parent_active = self.conditional_stack.len() <= 1
            || self.conditional_stack[..self.conditional_stack.len() - 1]
                .iter()
                .all(|c| c.active);

        let cond = self.conditional_stack.last_mut().unwrap();
        cond.active = parent_active && !cond.any_branch_taken;
        cond.in_else = true;
        if cond.active {
            cond.any_branch_taken = true;
        }

        Ok(())
    }

    fn handle_endif(&mut self, file: &str, line: usize) -> Result<(), PreprocessError> {
        self.conditional_stack.pop().ok_or_else(|| PreprocessError::UnbalancedConditional {
            directive: "#endif".to_string(),
            file: file.to_string(),
            line,
        })?;
        Ok(())
    }

    fn handle_include(
        &mut self,
        rest: &str,
        file: &str,
        line: usize,
    ) -> Result<(), PreprocessError> {
        // Parse include path (supports "path" format)
        let path = rest.trim();
        let path = if path.starts_with('"') && path.ends_with('"') && path.len() > 2 {
            &path[1..path.len() - 1]
        } else {
            return Err(PreprocessError::InvalidSyntax {
                directive: "#include".to_string(),
                message: "expected \"path\" format".to_string(),
                file: file.to_string(),
                line,
            });
        };

        // Resolve the include path
        let resolved = self.preprocessor.resolve_include(path, file, line)?;

        // Check for circular includes
        if self.include_stack.contains(&resolved) {
            return Err(PreprocessError::CircularInclude {
                path: resolved,
                include_stack: self.include_stack.clone(),
            });
        }

        // Track dependency
        if !self.dependencies.contains(&resolved) {
            self.dependencies.push(resolved.clone());
        }

        // Read and process the included file
        let content = self.preprocessor.read_file(&resolved)?;
        self.include_stack.push(resolved.clone());
        self.process_source(&content, &resolved)?;
        self.include_stack.pop();

        Ok(())
    }

    fn evaluate_expression(
        &self,
        expr: &str,
        file: &str,
        line: usize,
    ) -> Result<bool, PreprocessError> {
        let mut evaluator = ExpressionEvaluator::new(&self.defines);
        match evaluator.evaluate(expr) {
            Ok(value) => Ok(value != 0),
            Err(msg) => Err(PreprocessError::InvalidExpression {
                expression: expr.to_string(),
                message: msg,
                file: file.to_string(),
                line,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Substitute a macro in a line, only matching whole words.
fn substitute_macro(line: &str, name: &str, value: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_alphanumeric() || c == '_' {
            // Start of an identifier - collect it
            let mut ident = String::new();
            ident.push(c);
            while let Some(&next) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' {
                    ident.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            // Check if it matches the macro name
            if ident == name {
                result.push_str(value);
            } else {
                result.push_str(&ident);
            }
        } else {
            result.push(c);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Helper to create a preprocessor with in-memory file system
    fn pp_with_files(files: HashMap<&str, &str>) -> WgslPreprocessor {
        let files: HashMap<String, String> = files
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let mut pp = WgslPreprocessor::new();
        pp.set_file_reader(move |path: &str| {
            files.get(path).cloned().ok_or_else(|| format!("file not found: {}", path))
        });
        pp
    }

    // -----------------------------------------------------------------------
    // #define tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_define_simple_replacement() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define PI 3.14159
let x = PI;
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let x = 3.14159;"));
    }

    #[test]
    fn test_define_without_value() {
        let mut pp = WgslPreprocessor::new();
        pp.define("ENABLE_FEATURE", None);
        let source = r#"
#ifdef ENABLE_FEATURE
let enabled = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let enabled = true;"));
    }

    #[test]
    fn test_define_with_value() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define MAX_LIGHTS 16
let lights: array<Light, MAX_LIGHTS>;
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let lights: array<Light, 16>;"));
    }

    #[test]
    fn test_define_in_directive() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define DEBUG 1
#if DEBUG
let debug_mode = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let debug_mode = true;"));
    }

    #[test]
    fn test_undef() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define FEATURE
#undef FEATURE
#ifdef FEATURE
should_not_appear
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("should_not_appear"));
    }

    // -----------------------------------------------------------------------
    // #ifdef / #ifndef tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ifdef_true_path() {
        let mut pp = WgslPreprocessor::new();
        pp.define("ENABLE_SHADOWS", None);
        let source = r#"
#ifdef ENABLE_SHADOWS
let shadow = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let shadow = true;"));
    }

    #[test]
    fn test_ifdef_false_path() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifdef ENABLE_SHADOWS
let shadow = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("let shadow = true;"));
    }

    #[test]
    fn test_ifndef_true() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifndef UNDEFINED_MACRO
let not_defined = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let not_defined = true;"));
    }

    #[test]
    fn test_ifndef_false() {
        let mut pp = WgslPreprocessor::new();
        pp.define("DEFINED_MACRO", None);
        let source = r#"
#ifndef DEFINED_MACRO
should_not_appear
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("should_not_appear"));
    }

    #[test]
    fn test_ifdef_else() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifdef UNDEFINED
let a = 1;
#else
let b = 2;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("let a = 1;"));
        assert!(result.output.contains("let b = 2;"));
    }

    // -----------------------------------------------------------------------
    // Nested conditionals
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_conditionals() {
        let mut pp = WgslPreprocessor::new();
        pp.define("OUTER", None);
        pp.define("INNER", None);
        let source = r#"
#ifdef OUTER
outer_start
#ifdef INNER
inner_content
#endif
outer_end
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("outer_start"));
        assert!(result.output.contains("inner_content"));
        assert!(result.output.contains("outer_end"));
    }

    #[test]
    fn test_nested_conditional_outer_false() {
        let mut pp = WgslPreprocessor::new();
        pp.define("INNER", None);
        let source = r#"
#ifdef OUTER
#ifdef INNER
should_not_appear
#endif
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("should_not_appear"));
    }

    #[test]
    fn test_deeply_nested_conditionals() {
        let mut pp = WgslPreprocessor::new();
        pp.define("A", None);
        pp.define("B", None);
        pp.define("C", None);
        let source = r#"
#ifdef A
level_a
#ifdef B
level_b
#ifdef C
level_c
#endif
#endif
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("level_a"));
        assert!(result.output.contains("level_b"));
        assert!(result.output.contains("level_c"));
    }

    // -----------------------------------------------------------------------
    // #if / #elif / #else
    // -----------------------------------------------------------------------

    #[test]
    fn test_if_true() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#if 1
let x = 1;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let x = 1;"));
    }

    #[test]
    fn test_if_false() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#if 0
should_not_appear
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("should_not_appear"));
    }

    #[test]
    fn test_if_expression_comparison() {
        let mut pp = WgslPreprocessor::new();
        pp.define("VERSION", Some("2"));
        let source = r#"
#if VERSION >= 2
let v2 = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let v2 = true;"));
    }

    #[test]
    fn test_if_defined() {
        let mut pp = WgslPreprocessor::new();
        pp.define("FEATURE", None);
        let source = r#"
#if defined(FEATURE)
let feature = true;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let feature = true;"));
    }

    #[test]
    fn test_elif_chain() {
        let mut pp = WgslPreprocessor::new();
        pp.define("MODE", Some("2"));
        let source = r#"
#if MODE == 1
mode_1
#elif MODE == 2
mode_2
#elif MODE == 3
mode_3
#else
mode_default
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("mode_1"));
        assert!(result.output.contains("mode_2"));
        assert!(!result.output.contains("mode_3"));
        assert!(!result.output.contains("mode_default"));
    }

    #[test]
    fn test_elif_fallthrough_to_else() {
        let mut pp = WgslPreprocessor::new();
        pp.define("MODE", Some("99"));
        let source = r#"
#if MODE == 1
mode_1
#elif MODE == 2
mode_2
#else
mode_default
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(!result.output.contains("mode_1"));
        assert!(!result.output.contains("mode_2"));
        assert!(result.output.contains("mode_default"));
    }

    #[test]
    fn test_if_logical_and() {
        let mut pp = WgslPreprocessor::new();
        pp.define("A", Some("1"));
        pp.define("B", Some("1"));
        let source = r#"
#if A && B
both_true
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("both_true"));
    }

    #[test]
    fn test_if_logical_or() {
        let mut pp = WgslPreprocessor::new();
        pp.define("A", Some("0"));
        pp.define("B", Some("1"));
        let source = r#"
#if A || B
one_true
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("one_true"));
    }

    #[test]
    fn test_if_not() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#if !0
not_zero
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("not_zero"));
    }

    // -----------------------------------------------------------------------
    // #include tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_include_resolution() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"common.wgsl\"\nlet main = 1;"),
            ("common.wgsl", "let common = 2;"),
        ]);
        let pp = pp_with_files(files);

        let result = pp.preprocess("#include \"common.wgsl\"\nlet main = 1;", "main.wgsl").unwrap();
        assert!(result.output.contains("let common = 2;"));
        assert!(result.output.contains("let main = 1;"));
    }

    #[test]
    fn test_include_search_path() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"utils.wgsl\""),
            ("shaders/utils.wgsl", "let util = 1;"),
        ]);
        let mut pp = pp_with_files(files);
        pp.add_search_path("shaders");

        let result = pp.preprocess("#include \"utils.wgsl\"", "main.wgsl").unwrap();
        assert!(result.output.contains("let util = 1;"));
    }

    #[test]
    fn test_circular_include_detection() {
        let files = HashMap::from([
            ("a.wgsl", "#include \"b.wgsl\""),
            ("b.wgsl", "#include \"a.wgsl\""),
        ]);
        let pp = pp_with_files(files);

        let result = pp.preprocess("#include \"b.wgsl\"", "a.wgsl");
        assert!(matches!(result, Err(PreprocessError::CircularInclude { .. })));
    }

    #[test]
    fn test_include_not_found() {
        let pp = WgslPreprocessor::new();
        let result = pp.preprocess("#include \"nonexistent.wgsl\"", "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::IncludeNotFound { .. })));
    }

    #[test]
    fn test_dependency_tracking() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"a.wgsl\"\n#include \"b.wgsl\""),
            ("a.wgsl", "let a = 1;"),
            ("b.wgsl", "#include \"c.wgsl\""),
            ("c.wgsl", "let c = 3;"),
        ]);
        let pp = pp_with_files(files);

        let result = pp.preprocess("#include \"a.wgsl\"\n#include \"b.wgsl\"", "main.wgsl").unwrap();
        assert!(result.dependencies.contains(&"a.wgsl".to_string()));
        assert!(result.dependencies.contains(&"b.wgsl".to_string()));
        assert!(result.dependencies.contains(&"c.wgsl".to_string()));
    }

    #[test]
    fn test_include_with_conditionals() {
        let files = HashMap::from([
            ("main.wgsl", "#define USE_SHADOWS\n#include \"effects.wgsl\""),
            ("effects.wgsl", "#ifdef USE_SHADOWS\nlet shadows = true;\n#endif"),
        ]);
        let pp = pp_with_files(files);

        let result = pp.preprocess("#define USE_SHADOWS\n#include \"effects.wgsl\"", "main.wgsl").unwrap();
        assert!(result.output.contains("let shadows = true;"));
    }

    // -----------------------------------------------------------------------
    // Predefined macros
    // -----------------------------------------------------------------------

    #[test]
    fn test_predefined_trinity_version() {
        let mut pp = WgslPreprocessor::new();
        pp.set_version(1, 2, 3);
        let source = r#"
let version = TRINITY_VERSION;
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        // 1 * 10000 + 2 * 100 + 3 = 10203
        assert!(result.output.contains("let version = 10203;"));
    }

    #[test]
    fn test_predefined_rhi_vulkan() {
        let mut pp = WgslPreprocessor::new();
        pp.set_backend(RhiBackend::Vulkan);
        let source = r#"
#ifdef TRINITY_RHI_VULKAN
let backend = vulkan;
#endif
#ifdef TRINITY_RHI_D3D12
let backend = d3d12;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let backend = vulkan;"));
        assert!(!result.output.contains("let backend = d3d12;"));
    }

    #[test]
    fn test_predefined_rhi_d3d12() {
        let mut pp = WgslPreprocessor::new();
        pp.set_backend(RhiBackend::D3D12);
        let source = r#"
#ifdef TRINITY_RHI_D3D12
let backend = d3d12;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let backend = d3d12;"));
    }

    #[test]
    fn test_predefined_rhi_metal() {
        let mut pp = WgslPreprocessor::new();
        pp.set_backend(RhiBackend::Metal);
        let source = r#"
#ifdef TRINITY_RHI_METAL
let backend = metal;
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let backend = metal;"));
    }

    // -----------------------------------------------------------------------
    // #error / #warning
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_directive() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#error This is an error message
"#;
        let result = pp.preprocess(source, "test.wgsl");
        match result {
            Err(PreprocessError::ErrorDirective { message, .. }) => {
                assert_eq!(message, "This is an error message");
            }
            _ => panic!("expected ErrorDirective"),
        }
    }

    #[test]
    fn test_error_directive_in_false_branch() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#if 0
#error This should not trigger
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(result.is_ok());
    }

    #[test]
    fn test_warning_directive() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#warning This is a warning
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.warnings.len() == 1);
        assert!(result.warnings[0].contains("This is a warning"));
    }

    // -----------------------------------------------------------------------
    // State serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_serialization() {
        let mut pp = WgslPreprocessor::new();
        pp.define("FOO", Some("1"));
        pp.define("BAR", None);
        pp.add_search_path("shaders");
        pp.set_backend(RhiBackend::Metal);
        pp.set_version(2, 1, 0);

        let state = pp.to_state();
        let json = state.to_json().unwrap();
        let restored_state = PreprocessorState::from_json(&json).unwrap();

        assert_eq!(state, restored_state);
    }

    #[test]
    fn test_state_restoration() {
        let mut original = WgslPreprocessor::new();
        original.define("TEST", Some("42"));
        original.set_backend(RhiBackend::D3D12);

        let state = original.to_state();
        let restored = WgslPreprocessor::from_state(&state);

        let source = r#"
let test = TEST;
#ifdef TRINITY_RHI_D3D12
let backend = d3d12;
#endif
"#;
        let result = restored.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let test = 42;"));
        assert!(result.output.contains("let backend = d3d12;"));
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_unbalanced_endif() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::UnbalancedConditional { .. })));
    }

    #[test]
    fn test_missing_endif() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifdef FOO
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::UnbalancedConditional { .. })));
    }

    #[test]
    fn test_invalid_define_syntax() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::InvalidSyntax { .. })));
    }

    #[test]
    fn test_invalid_expression() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#if 1 +
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::InvalidExpression { .. })));
    }

    #[test]
    fn test_duplicate_else() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifdef FOO
#else
#else
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::InvalidSyntax { .. })));
    }

    #[test]
    fn test_elif_after_else() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#ifdef FOO
#else
#elif 1
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl");
        assert!(matches!(result, Err(PreprocessError::InvalidSyntax { .. })));
    }

    // -----------------------------------------------------------------------
    // Complex scenarios
    // -----------------------------------------------------------------------

    #[test]
    fn test_complex_expression() {
        let mut pp = WgslPreprocessor::new();
        pp.define("A", Some("5"));
        pp.define("B", Some("3"));
        let source = r#"
#if (A > B) && (defined(A) || !defined(C))
complex_true
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("complex_true"));
    }

    #[test]
    fn test_macro_substitution_whole_word() {
        let pp = WgslPreprocessor::new();
        let source = r#"
#define FOO 123
let FOO_bar = 1;
let bar_FOO = 2;
let FOO = 3;
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        // FOO_bar and bar_FOO should NOT be substituted
        assert!(result.output.contains("let FOO_bar = 1;"));
        assert!(result.output.contains("let bar_FOO = 2;"));
        // FOO should be substituted
        assert!(result.output.contains("let 123 = 3;"));
    }

    #[test]
    fn test_include_deduplication() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"a.wgsl\"\n#include \"a.wgsl\""),
            ("a.wgsl", "let a = 1;"),
        ]);
        let pp = pp_with_files(files);

        let result = pp.preprocess("#include \"a.wgsl\"\n#include \"a.wgsl\"", "main.wgsl").unwrap();
        // Dependency should only appear once
        assert_eq!(result.dependencies.len(), 1);
        // But content is included twice (standard C preprocessor behavior)
        assert_eq!(result.output.matches("let a = 1;").count(), 2);
    }

    #[test]
    fn test_backend_conditional_metal_fallback() {
        let mut pp = WgslPreprocessor::new();
        pp.set_backend(RhiBackend::WebGpu);
        let source = r#"
#if defined(TRINITY_RHI_VULKAN)
vulkan_code
#elif defined(TRINITY_RHI_D3D12)
d3d12_code
#elif defined(TRINITY_RHI_METAL)
metal_code
#elif defined(TRINITY_RHI_WEBGPU)
webgpu_code
#else
fallback_code
#endif
"#;
        let result = pp.preprocess(source, "test.wgsl").unwrap();
        assert!(result.output.contains("webgpu_code"));
        assert!(!result.output.contains("vulkan_code"));
        assert!(!result.output.contains("d3d12_code"));
        assert!(!result.output.contains("metal_code"));
        assert!(!result.output.contains("fallback_code"));
    }

    #[test]
    fn test_preprocess_with_substitution() {
        let mut pp = WgslPreprocessor::new();
        pp.set_version(1, 0, 0);
        let source = "let version = TRINITY_VERSION;";
        let result = pp.preprocess_with_substitution(source, "test.wgsl").unwrap();
        assert!(result.output.contains("let version = 10000;"));
    }
}
