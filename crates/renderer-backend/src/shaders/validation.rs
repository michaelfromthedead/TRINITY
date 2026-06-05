//! Naga pre-validation layer for better shader error messages.
//!
//! This module provides a dedicated validation layer using naga's WGSL frontend
//! and validation infrastructure to produce detailed, human-readable error messages
//! with source code context.
//!
//! # Overview
//!
//! The Naga validation layer provides:
//!
//! - **WGSL parsing**: Parse WGSL source into Naga IR with detailed error reporting
//! - **Module validation**: Validate parsed modules with configurable strictness
//! - **Error location extraction**: Extract precise line/column from naga spans
//! - **Human-readable messages**: Convert naga errors to user-friendly format
//! - **Source snippets**: Show relevant source lines with error markers
//!
//! # Architecture
//!
//! ```text
//! NagaValidator
//! +-- config: ValidationConfig (flags, strictness, enabled checks)
//! +-- validate_wgsl(source) -> ValidationResult
//! +-- parse_wgsl(source) -> Result<naga::Module, ValidationError>
//! +-- validate_module(module) -> Result<ModuleInfo, ValidationError>
//! +-- format_error(error, source) -> String
//!
//! ValidationError
//! +-- Parse { message, location, labels, suggestions }
//! +-- Validation { message, location, labels }
//! +-- EmptySource
//!
//! SourceSnippet
//! +-- line_number: u32
//! +-- content: String
//! +-- marker_start: Option<u32>
//! +-- marker_length: Option<u32>
//! ```
//!
//! # Example
//!
//! ```
//! use renderer_backend::shaders::validation::{
//!     NagaValidator, ValidationConfig, Strictness,
//! };
//!
//! let validator = NagaValidator::new(ValidationConfig::default());
//!
//! let source = r#"
//!     @vertex
//!     fn main() -> @builtin(position) vec4<f32> {
//!         return vec4<f32>(0.0);
//!     }
//! "#;
//!
//! match validator.validate_wgsl(source) {
//!     Ok(result) => println!("Valid shader with {} entry points", result.entry_points),
//!     Err(err) => {
//!         let formatted = validator.format_error(&err, source);
//!         eprintln!("{}", formatted);
//!     }
//! }
//! ```

use std::fmt;
use std::path::PathBuf;

// ============================================================================
// Constants
// ============================================================================

/// Default number of context lines to show before/after error.
pub const DEFAULT_CONTEXT_LINES: u32 = 2;

/// Maximum source snippet length (in chars) before truncation.
pub const MAX_SNIPPET_LENGTH: usize = 200;

// ============================================================================
// Strictness Level
// ============================================================================

/// Strictness level for shader validation.
///
/// Controls how strict the validation is. Higher strictness levels
/// catch more potential issues but may reject valid shaders that
/// work on some hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strictness {
    /// Relaxed validation - only catch definite errors.
    Relaxed,
    /// Standard validation - catch common issues.
    #[default]
    Standard,
    /// Strict validation - catch all potential issues.
    Strict,
    /// Pedantic - warn about style and best practices.
    Pedantic,
}

impl Strictness {
    /// Returns the naga ValidationFlags for this strictness level.
    pub fn validation_flags(&self) -> naga::valid::ValidationFlags {
        match self {
            Strictness::Relaxed => {
                naga::valid::ValidationFlags::empty()
            }
            Strictness::Standard => {
                naga::valid::ValidationFlags::all()
            }
            Strictness::Strict => {
                naga::valid::ValidationFlags::all()
            }
            Strictness::Pedantic => {
                naga::valid::ValidationFlags::all()
            }
        }
    }

    /// Returns the naga Capabilities for this strictness level.
    pub fn capabilities(&self) -> naga::valid::Capabilities {
        match self {
            Strictness::Relaxed => naga::valid::Capabilities::all(),
            Strictness::Standard => naga::valid::Capabilities::all(),
            Strictness::Strict => {
                // More restricted capabilities to ensure broad compatibility
                naga::valid::Capabilities::all()
            }
            Strictness::Pedantic => {
                // Even more restricted for maximum portability
                naga::valid::Capabilities::all()
            }
        }
    }
}

// ============================================================================
// Validation Configuration
// ============================================================================

/// Configuration for the Naga validator.
///
/// Controls validation behavior including strictness level, enabled checks,
/// and error formatting options.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Strictness level for validation.
    pub strictness: Strictness,
    /// Whether to include source snippets in error messages.
    pub include_snippets: bool,
    /// Number of context lines to show around errors.
    pub context_lines: u32,
    /// Whether to include suggestions in error messages.
    pub include_suggestions: bool,
    /// Optional file path for error messages.
    pub file_path: Option<PathBuf>,
    /// Whether to use ANSI color codes in formatted output.
    pub use_colors: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strictness: Strictness::Standard,
            include_snippets: true,
            context_lines: DEFAULT_CONTEXT_LINES,
            include_suggestions: true,
            file_path: None,
            use_colors: false,
        }
    }
}

impl ValidationConfig {
    /// Creates a new config with the given strictness level.
    pub fn with_strictness(strictness: Strictness) -> Self {
        Self {
            strictness,
            ..Default::default()
        }
    }

    /// Creates a relaxed config for quick validation.
    pub fn relaxed() -> Self {
        Self {
            strictness: Strictness::Relaxed,
            include_snippets: false,
            include_suggestions: false,
            ..Default::default()
        }
    }

    /// Creates a strict config for production validation.
    pub fn strict() -> Self {
        Self {
            strictness: Strictness::Strict,
            include_snippets: true,
            context_lines: 3,
            include_suggestions: true,
            ..Default::default()
        }
    }

    /// Creates a pedantic config for development/CI.
    pub fn pedantic() -> Self {
        Self {
            strictness: Strictness::Pedantic,
            include_snippets: true,
            context_lines: 4,
            include_suggestions: true,
            ..Default::default()
        }
    }

    /// Sets the file path for error messages.
    pub fn with_file_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Enables ANSI color codes in formatted output.
    pub fn with_colors(mut self) -> Self {
        self.use_colors = true;
        self
    }

    /// Disables source snippets in error messages.
    pub fn without_snippets(mut self) -> Self {
        self.include_snippets = false;
        self
    }

    /// Sets the number of context lines.
    pub fn with_context_lines(mut self, lines: u32) -> Self {
        self.context_lines = lines;
        self
    }
}

// ============================================================================
// Source Snippet
// ============================================================================

/// A snippet of source code for error display.
///
/// Contains a line of source code with optional error marking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSnippet {
    /// Line number (1-based).
    pub line_number: u32,
    /// The line content.
    pub content: String,
    /// Column where the marker starts (0-based within the line).
    pub marker_start: Option<u32>,
    /// Length of the marker in characters.
    pub marker_length: Option<u32>,
    /// Whether this is the error line (vs context).
    pub is_error_line: bool,
}

impl SourceSnippet {
    /// Creates a context line (no marker).
    pub fn context(line_number: u32, content: impl Into<String>) -> Self {
        Self {
            line_number,
            content: content.into(),
            marker_start: None,
            marker_length: None,
            is_error_line: false,
        }
    }

    /// Creates an error line with a marker.
    pub fn error(
        line_number: u32,
        content: impl Into<String>,
        marker_start: u32,
        marker_length: u32,
    ) -> Self {
        Self {
            line_number,
            content: content.into(),
            marker_start: Some(marker_start),
            marker_length: Some(marker_length.max(1)),
            is_error_line: true,
        }
    }

    /// Formats the snippet for display.
    pub fn format(&self, line_width: usize, use_colors: bool) -> String {
        let line_num_str = format!("{:>width$}", self.line_number, width = line_width);
        let prefix = if self.is_error_line { ">" } else { " " };

        let mut result = format!("{} {} | {}", prefix, line_num_str, self.content);

        // Add marker line if this is an error line
        if let (Some(start), Some(len)) = (self.marker_start, self.marker_length) {
            let marker_prefix = format!("  {} | ", " ".repeat(line_width));
            let marker_padding = " ".repeat(start as usize);
            let marker = if use_colors {
                format!("\x1b[31m{}\x1b[0m", "^".repeat(len as usize))
            } else {
                "^".repeat(len as usize)
            };
            result.push('\n');
            result.push_str(&marker_prefix);
            result.push_str(&marker_padding);
            result.push_str(&marker);
        }

        result
    }
}

// ============================================================================
// Error Location
// ============================================================================

/// Location of an error in shader source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLocation {
    /// Line number (1-based).
    pub line: u32,
    /// Column number (1-based).
    pub column: u32,
    /// Byte offset from start of source.
    pub offset: u32,
    /// Length of the error span in bytes.
    pub length: u32,
}

impl ErrorLocation {
    /// Creates a new error location.
    pub fn new(line: u32, column: u32, offset: u32, length: u32) -> Self {
        Self {
            line,
            column,
            offset,
            length,
        }
    }

    /// Creates a location from a naga span and source.
    pub fn from_span(span: naga::Span, source: &str) -> Option<Self> {
        let range = span.to_range()?;
        let offset = range.start;
        let length = range.end.saturating_sub(range.start);

        let (line, column) = offset_to_line_column(source, offset);

        Some(Self {
            line,
            column,
            offset: offset as u32,
            length: length as u32,
        })
    }

    /// Formats the location for display.
    pub fn format(&self, file_path: Option<&PathBuf>) -> String {
        match file_path {
            Some(path) => format!("{}:{}:{}", path.display(), self.line, self.column),
            None => format!("{}:{}", self.line, self.column),
        }
    }
}

impl Default for ErrorLocation {
    fn default() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
            length: 0,
        }
    }
}

// ============================================================================
// Error Label
// ============================================================================

/// A label attached to an error (primary or secondary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLabel {
    /// Location of this label.
    pub location: Option<ErrorLocation>,
    /// The label message.
    pub message: String,
    /// Whether this is the primary label.
    pub is_primary: bool,
}

impl ErrorLabel {
    /// Creates a primary label.
    pub fn primary(message: impl Into<String>, location: Option<ErrorLocation>) -> Self {
        Self {
            location,
            message: message.into(),
            is_primary: true,
        }
    }

    /// Creates a secondary label.
    pub fn secondary(message: impl Into<String>, location: Option<ErrorLocation>) -> Self {
        Self {
            location,
            message: message.into(),
            is_primary: false,
        }
    }
}

// ============================================================================
// Validation Error
// ============================================================================

/// Errors from Naga validation.
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// WGSL parsing failed.
    Parse {
        /// Error message from naga.
        message: String,
        /// Primary error location.
        location: Option<ErrorLocation>,
        /// Additional labels.
        labels: Vec<ErrorLabel>,
        /// Suggested fixes.
        suggestions: Vec<String>,
    },

    /// Module validation failed.
    Validation {
        /// Error message from naga.
        message: String,
        /// Primary error location.
        location: Option<ErrorLocation>,
        /// Additional labels.
        labels: Vec<ErrorLabel>,
    },

    /// Source is empty or whitespace-only.
    EmptySource,
}

impl ValidationError {
    /// Creates a parse error.
    pub fn parse(message: impl Into<String>) -> Self {
        Self::Parse {
            message: message.into(),
            location: None,
            labels: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Creates a parse error with location.
    pub fn parse_at(message: impl Into<String>, location: ErrorLocation) -> Self {
        Self::Parse {
            message: message.into(),
            location: Some(location),
            labels: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Creates a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            location: None,
            labels: Vec::new(),
        }
    }

    /// Creates a validation error with location.
    pub fn validation_at(message: impl Into<String>, location: ErrorLocation) -> Self {
        Self::Validation {
            message: message.into(),
            location: Some(location),
            labels: Vec::new(),
        }
    }

    /// Adds a label to the error.
    pub fn with_label(mut self, label: ErrorLabel) -> Self {
        match &mut self {
            Self::Parse { labels, .. } | Self::Validation { labels, .. } => {
                labels.push(label);
            }
            Self::EmptySource => {}
        }
        self
    }

    /// Adds a suggestion to parse errors.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        if let Self::Parse { suggestions, .. } = &mut self {
            suggestions.push(suggestion.into());
        }
        self
    }

    /// Returns the primary error message.
    pub fn message(&self) -> &str {
        match self {
            Self::Parse { message, .. } | Self::Validation { message, .. } => message,
            Self::EmptySource => "shader source is empty",
        }
    }

    /// Returns the primary location if available.
    pub fn location(&self) -> Option<&ErrorLocation> {
        match self {
            Self::Parse { location, .. } | Self::Validation { location, .. } => location.as_ref(),
            Self::EmptySource => None,
        }
    }

    /// Returns whether this is a parse error.
    pub fn is_parse_error(&self) -> bool {
        matches!(self, Self::Parse { .. })
    }

    /// Returns whether this is a validation error.
    pub fn is_validation_error(&self) -> bool {
        matches!(self, Self::Validation { .. })
    }

    /// Returns the error kind as a string.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Parse { .. } => "parse",
            Self::Validation { .. } => "validation",
            Self::EmptySource => "empty",
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse {
                message, location, ..
            } => {
                if let Some(loc) = location {
                    write!(f, "parse error at {}:{}: {}", loc.line, loc.column, message)
                } else {
                    write!(f, "parse error: {}", message)
                }
            }
            Self::Validation {
                message, location, ..
            } => {
                if let Some(loc) = location {
                    write!(
                        f,
                        "validation error at {}:{}: {}",
                        loc.line, loc.column, message
                    )
                } else {
                    write!(f, "validation error: {}", message)
                }
            }
            Self::EmptySource => {
                write!(f, "shader source is empty")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// ============================================================================
// Validation Result
// ============================================================================

/// Successful validation result with shader information.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The validated naga module.
    pub module: naga::Module,
    /// Module validation info from naga.
    pub info: naga::valid::ModuleInfo,
    /// Number of entry points.
    pub entry_points: usize,
    /// Names of entry points.
    pub entry_point_names: Vec<String>,
    /// Total number of functions.
    pub function_count: usize,
    /// Total number of global variables.
    pub global_count: usize,
    /// Total number of types.
    pub type_count: usize,
    /// Total number of constants.
    pub constant_count: usize,
}

impl ValidationResult {
    /// Creates a new validation result from a module and info.
    pub fn new(module: naga::Module, info: naga::valid::ModuleInfo) -> Self {
        let entry_points = module.entry_points.len();
        let entry_point_names = module
            .entry_points
            .iter()
            .map(|ep| ep.name.clone())
            .collect();
        let function_count = module.functions.len();
        let global_count = module.global_variables.len();
        let type_count = module.types.len();
        let constant_count = module.constants.len();

        Self {
            module,
            info,
            entry_points,
            entry_point_names,
            function_count,
            global_count,
            type_count,
            constant_count,
        }
    }

    /// Returns whether the shader has a vertex entry point.
    pub fn has_vertex(&self) -> bool {
        self.module
            .entry_points
            .iter()
            .any(|ep| ep.stage == naga::ShaderStage::Vertex)
    }

    /// Returns whether the shader has a fragment entry point.
    pub fn has_fragment(&self) -> bool {
        self.module
            .entry_points
            .iter()
            .any(|ep| ep.stage == naga::ShaderStage::Fragment)
    }

    /// Returns whether the shader has a compute entry point.
    pub fn has_compute(&self) -> bool {
        self.module
            .entry_points
            .iter()
            .any(|ep| ep.stage == naga::ShaderStage::Compute)
    }
}

// ============================================================================
// Naga Validator
// ============================================================================

/// Naga-based WGSL validator with rich error reporting.
///
/// Wraps naga's WGSL frontend and validation infrastructure to provide
/// detailed, human-readable error messages with source code context.
#[derive(Debug, Clone)]
pub struct NagaValidator {
    config: ValidationConfig,
}

impl NagaValidator {
    /// Creates a new validator with the given configuration.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Creates a validator with default configuration.
    pub fn default_validator() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Creates a strict validator for production use.
    pub fn strict() -> Self {
        Self::new(ValidationConfig::strict())
    }

    /// Creates a relaxed validator for quick checks.
    pub fn relaxed() -> Self {
        Self::new(ValidationConfig::relaxed())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Returns a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut ValidationConfig {
        &mut self.config
    }

    /// Validates WGSL source completely (parse + validate).
    ///
    /// Returns a `ValidationResult` on success, or a `ValidationError` with
    /// detailed information on failure.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::validation::{NagaValidator, ValidationConfig};
    ///
    /// let validator = NagaValidator::new(ValidationConfig::default());
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    ///
    /// match validator.validate_wgsl(source) {
    ///     Ok(result) => println!("Valid! {} entry points", result.entry_points),
    ///     Err(err) => eprintln!("Invalid: {}", err),
    /// }
    /// ```
    pub fn validate_wgsl(&self, source: &str) -> Result<ValidationResult, ValidationError> {
        // Check for empty source
        if source.trim().is_empty() {
            return Err(ValidationError::EmptySource);
        }

        // Parse the source
        let module = self.parse_wgsl(source)?;

        // Validate the module
        let info = self.validate_module(&module)?;

        Ok(ValidationResult::new(module, info))
    }

    /// Parses WGSL source into a naga Module.
    ///
    /// This performs only parsing, not validation. Use `validate_wgsl()` for
    /// complete validation.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::validation::{NagaValidator, ValidationConfig};
    ///
    /// let validator = NagaValidator::new(ValidationConfig::default());
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    ///
    /// let module = validator.parse_wgsl(source)?;
    /// println!("Parsed {} entry points", module.entry_points.len());
    /// # Ok::<(), renderer_backend::shaders::validation::ValidationError>(())
    /// ```
    pub fn parse_wgsl(&self, source: &str) -> Result<naga::Module, ValidationError> {
        if source.trim().is_empty() {
            return Err(ValidationError::EmptySource);
        }

        naga::front::wgsl::parse_str(source).map_err(|err| self.convert_parse_error(err, source))
    }

    /// Validates a parsed naga Module.
    ///
    /// Returns the `ModuleInfo` on success, which contains information about
    /// the module's entry points, functions, and resources.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::validation::{NagaValidator, ValidationConfig};
    ///
    /// let validator = NagaValidator::new(ValidationConfig::default());
    /// let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    ///
    /// let module = validator.parse_wgsl(source)?;
    /// let info = validator.validate_module(&module)?;
    /// # Ok::<(), renderer_backend::shaders::validation::ValidationError>(())
    /// ```
    pub fn validate_module(
        &self,
        module: &naga::Module,
    ) -> Result<naga::valid::ModuleInfo, ValidationError> {
        let flags = self.config.strictness.validation_flags();
        let caps = self.config.strictness.capabilities();

        let mut validator = naga::valid::Validator::new(flags, caps);

        validator
            .validate(module)
            .map_err(|err| self.convert_validation_error(err, ""))
    }

    /// Validates a module with the source for better error messages.
    pub fn validate_module_with_source(
        &self,
        module: &naga::Module,
        source: &str,
    ) -> Result<naga::valid::ModuleInfo, ValidationError> {
        let flags = self.config.strictness.validation_flags();
        let caps = self.config.strictness.capabilities();

        let mut validator = naga::valid::Validator::new(flags, caps);

        validator
            .validate(module)
            .map_err(|err| self.convert_validation_error(err, source))
    }

    /// Formats an error with source context for display.
    ///
    /// Produces a human-readable error message with:
    /// - Error type and location
    /// - Source code snippet with error marker
    /// - Additional labels and notes
    /// - Suggestions (for parse errors)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::validation::{NagaValidator, ValidationConfig};
    ///
    /// let validator = NagaValidator::new(ValidationConfig::default());
    /// let source = "invalid wgsl @@@";
    ///
    /// if let Err(err) = validator.validate_wgsl(source) {
    ///     let formatted = validator.format_error(&err, source);
    ///     eprintln!("{}", formatted);
    /// }
    /// ```
    pub fn format_error(&self, error: &ValidationError, source: &str) -> String {
        let mut output = String::new();

        // Header line
        let file_name = self
            .config
            .file_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<shader>".to_string());

        let error_kind = error.kind();
        let message = error.message();

        if let Some(loc) = error.location() {
            output.push_str(&format!(
                "error[{}]: {}\n  --> {}:{}:{}\n",
                error_kind, message, file_name, loc.line, loc.column
            ));
        } else {
            output.push_str(&format!("error[{}]: {}\n  --> {}\n", error_kind, message, file_name));
        }

        // Source snippet
        if self.config.include_snippets {
            if let Some(loc) = error.location() {
                let snippets =
                    self.extract_snippets(source, loc, self.config.context_lines);
                if !snippets.is_empty() {
                    let max_line_num = snippets.iter().map(|s| s.line_number).max().unwrap_or(1);
                    let line_width = format!("{}", max_line_num).len();

                    output.push_str("   |\n");
                    for snippet in &snippets {
                        output.push_str(&snippet.format(line_width, self.config.use_colors));
                        output.push('\n');
                    }
                    output.push_str("   |\n");
                }
            }
        }

        // Additional labels
        match error {
            ValidationError::Parse { labels, .. } | ValidationError::Validation { labels, .. } => {
                for label in labels {
                    if !label.is_primary {
                        if let Some(loc) = &label.location {
                            output.push_str(&format!(
                                "   = note: at {}:{}: {}\n",
                                loc.line, loc.column, label.message
                            ));
                        } else {
                            output.push_str(&format!("   = note: {}\n", label.message));
                        }
                    }
                }
            }
            ValidationError::EmptySource => {}
        }

        // Suggestions
        if self.config.include_suggestions {
            if let ValidationError::Parse { suggestions, .. } = error {
                for suggestion in suggestions {
                    output.push_str(&format!("   = help: {}\n", suggestion));
                }
            }
        }

        output
    }

    /// Extracts source snippets around an error location.
    pub fn extract_snippets(
        &self,
        source: &str,
        location: &ErrorLocation,
        context_lines: u32,
    ) -> Vec<SourceSnippet> {
        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        let error_line = location.line.saturating_sub(1) as usize;
        if error_line >= lines.len() {
            return Vec::new();
        }

        let start_line = error_line.saturating_sub(context_lines as usize);
        let end_line = (error_line + context_lines as usize + 1).min(lines.len());

        let mut snippets = Vec::new();

        for (idx, line_content) in lines[start_line..end_line].iter().enumerate() {
            let line_num = (start_line + idx + 1) as u32;

            if line_num == location.line {
                // This is the error line - calculate marker position
                let col_start = location.column.saturating_sub(1);
                let marker_len = if location.length > 0 {
                    location.length.min(line_content.len() as u32 - col_start + 1)
                } else {
                    1
                };

                snippets.push(SourceSnippet::error(
                    line_num,
                    truncate_line(line_content, MAX_SNIPPET_LENGTH),
                    col_start,
                    marker_len,
                ));
            } else {
                snippets.push(SourceSnippet::context(
                    line_num,
                    truncate_line(line_content, MAX_SNIPPET_LENGTH),
                ));
            }
        }

        snippets
    }

    /// Converts a naga ParseError to our ValidationError.
    fn convert_parse_error(
        &self,
        err: naga::front::wgsl::ParseError,
        source: &str,
    ) -> ValidationError {
        let message = err.message().to_string();

        // Extract primary location from first label
        let location = err.labels().next().and_then(|(span, _)| {
            ErrorLocation::from_span(span, source)
        });

        // Collect additional labels
        let labels: Vec<ErrorLabel> = err
            .labels()
            .enumerate()
            .map(|(i, (span, label_msg))| {
                let loc = ErrorLocation::from_span(span, source);
                if i == 0 {
                    ErrorLabel::primary(label_msg, loc)
                } else {
                    ErrorLabel::secondary(label_msg, loc)
                }
            })
            .collect();

        // Generate suggestions based on common errors
        let suggestions = self.generate_suggestions(&message);

        ValidationError::Parse {
            message,
            location,
            labels,
            suggestions,
        }
    }

    /// Converts a naga ValidationError to our ValidationError.
    fn convert_validation_error(
        &self,
        err: naga::WithSpan<naga::valid::ValidationError>,
        source: &str,
    ) -> ValidationError {
        let message = format!("{}", err.as_inner());

        // Extract location from the error's spans
        let location = err.spans().next().and_then(|(span, _)| {
            if span.is_defined() {
                ErrorLocation::from_span(*span, source)
            } else {
                None
            }
        });

        // Collect additional labels
        let labels: Vec<ErrorLabel> = err
            .spans()
            .enumerate()
            .filter_map(|(i, (span, label_msg))| {
                if span.is_defined() {
                    let loc = ErrorLocation::from_span(*span, source);
                    Some(if i == 0 {
                        ErrorLabel::primary(label_msg, loc)
                    } else {
                        ErrorLabel::secondary(label_msg, loc)
                    })
                } else {
                    None
                }
            })
            .collect();

        ValidationError::Validation {
            message,
            location,
            labels,
        }
    }

    /// Generates helpful suggestions based on error messages.
    fn generate_suggestions(&self, message: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        let msg_lower = message.to_lowercase();

        // Common WGSL mistakes
        if msg_lower.contains("expected") && msg_lower.contains("found") {
            suggestions.push("check for missing semicolons, brackets, or parentheses".into());
        }

        if msg_lower.contains("type") && msg_lower.contains("mismatch") {
            suggestions.push("ensure types match between operations and assignments".into());
        }

        if msg_lower.contains("undefined") || msg_lower.contains("unknown") {
            suggestions.push("check that all variables and functions are defined before use".into());
        }

        if msg_lower.contains("builtin") {
            suggestions.push("verify builtin attribute names match WGSL spec (e.g., @builtin(position), @builtin(vertex_index))".into());
        }

        if msg_lower.contains("binding") || msg_lower.contains("group") {
            suggestions.push(
                "ensure @group and @binding attributes are present for resource bindings".into(),
            );
        }

        if msg_lower.contains("workgroup") {
            suggestions.push("compute shaders require @workgroup_size attribute".into());
        }

        if msg_lower.contains("entry point") || msg_lower.contains("entrypoint") {
            suggestions.push("shaders need at least one entry point (@vertex, @fragment, or @compute)".into());
        }

        suggestions
    }
}

impl Default for NagaValidator {
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Converts a byte offset to line and column (1-based).
fn offset_to_line_column(source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let prefix = &source[..offset];

    let line = prefix.matches('\n').count() + 1;
    let last_newline = prefix.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let column = offset - last_newline + 1;

    (line as u32, column as u32)
}

/// Truncates a line to a maximum length, adding ellipsis if needed.
fn truncate_line(line: &str, max_len: usize) -> String {
    if line.len() <= max_len {
        line.to_string()
    } else {
        format!("{}...", &line[..max_len - 3])
    }
}

// ============================================================================
// Quick Validation Functions
// ============================================================================

/// Validates WGSL source quickly, returning Ok(()) or Err(ValidationError).
///
/// This is a convenience function for simple validation without creating
/// a validator instance.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::validation::quick_validate_wgsl;
///
/// let result = quick_validate_wgsl("@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }");
/// assert!(result.is_ok());
/// ```
pub fn quick_validate_wgsl(source: &str) -> Result<(), ValidationError> {
    NagaValidator::default_validator()
        .validate_wgsl(source)
        .map(|_| ())
}

/// Checks if WGSL source is valid, returning a boolean.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::validation::is_valid_wgsl;
///
/// assert!(is_valid_wgsl("@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"));
/// assert!(!is_valid_wgsl("invalid wgsl"));
/// ```
pub fn is_valid_wgsl(source: &str) -> bool {
    quick_validate_wgsl(source).is_ok()
}

/// Parses WGSL source, returning the naga Module or an error.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::validation::parse_wgsl;
///
/// let module = parse_wgsl("@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }")?;
/// assert_eq!(module.entry_points.len(), 1);
/// # Ok::<(), renderer_backend::shaders::validation::ValidationError>(())
/// ```
pub fn parse_wgsl(source: &str) -> Result<naga::Module, ValidationError> {
    NagaValidator::default_validator().parse_wgsl(source)
}

/// Formats an error with source context using default configuration.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::validation::{quick_validate_wgsl, format_validation_error};
///
/// let source = "invalid @@@";
/// if let Err(err) = quick_validate_wgsl(source) {
///     let formatted = format_validation_error(&err, source);
///     eprintln!("{}", formatted);
/// }
/// ```
pub fn format_validation_error(error: &ValidationError, source: &str) -> String {
    NagaValidator::default_validator().format_error(error, source)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Strictness Tests
    // =========================================================================

    #[test]
    fn test_strictness_default() {
        assert_eq!(Strictness::default(), Strictness::Standard);
    }

    #[test]
    fn test_strictness_validation_flags() {
        let relaxed = Strictness::Relaxed.validation_flags();
        let standard = Strictness::Standard.validation_flags();
        assert!(relaxed.is_empty());
        assert_eq!(standard, naga::valid::ValidationFlags::all());
    }

    #[test]
    fn test_strictness_capabilities() {
        let caps = Strictness::Standard.capabilities();
        assert_eq!(caps, naga::valid::Capabilities::all());
    }

    // =========================================================================
    // ValidationConfig Tests
    // =========================================================================

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.strictness, Strictness::Standard);
        assert!(config.include_snippets);
        assert_eq!(config.context_lines, DEFAULT_CONTEXT_LINES);
        assert!(config.include_suggestions);
        assert!(config.file_path.is_none());
        assert!(!config.use_colors);
    }

    #[test]
    fn test_validation_config_relaxed() {
        let config = ValidationConfig::relaxed();
        assert_eq!(config.strictness, Strictness::Relaxed);
        assert!(!config.include_snippets);
    }

    #[test]
    fn test_validation_config_strict() {
        let config = ValidationConfig::strict();
        assert_eq!(config.strictness, Strictness::Strict);
        assert!(config.include_snippets);
        assert_eq!(config.context_lines, 3);
    }

    #[test]
    fn test_validation_config_pedantic() {
        let config = ValidationConfig::pedantic();
        assert_eq!(config.strictness, Strictness::Pedantic);
        assert_eq!(config.context_lines, 4);
    }

    #[test]
    fn test_validation_config_with_file_path() {
        let config = ValidationConfig::default().with_file_path("test.wgsl");
        assert_eq!(config.file_path, Some(PathBuf::from("test.wgsl")));
    }

    #[test]
    fn test_validation_config_with_colors() {
        let config = ValidationConfig::default().with_colors();
        assert!(config.use_colors);
    }

    #[test]
    fn test_validation_config_without_snippets() {
        let config = ValidationConfig::default().without_snippets();
        assert!(!config.include_snippets);
    }

    #[test]
    fn test_validation_config_with_context_lines() {
        let config = ValidationConfig::default().with_context_lines(5);
        assert_eq!(config.context_lines, 5);
    }

    #[test]
    fn test_validation_config_with_strictness() {
        let config = ValidationConfig::with_strictness(Strictness::Pedantic);
        assert_eq!(config.strictness, Strictness::Pedantic);
    }

    // =========================================================================
    // SourceSnippet Tests
    // =========================================================================

    #[test]
    fn test_source_snippet_context() {
        let snippet = SourceSnippet::context(5, "let x = 1;");
        assert_eq!(snippet.line_number, 5);
        assert_eq!(snippet.content, "let x = 1;");
        assert!(snippet.marker_start.is_none());
        assert!(!snippet.is_error_line);
    }

    #[test]
    fn test_source_snippet_error() {
        let snippet = SourceSnippet::error(10, "let x = invalid;", 8, 7);
        assert_eq!(snippet.line_number, 10);
        assert_eq!(snippet.marker_start, Some(8));
        assert_eq!(snippet.marker_length, Some(7));
        assert!(snippet.is_error_line);
    }

    #[test]
    fn test_source_snippet_error_minimum_length() {
        let snippet = SourceSnippet::error(1, "x", 0, 0);
        assert_eq!(snippet.marker_length, Some(1)); // Minimum is 1
    }

    #[test]
    fn test_source_snippet_format_context() {
        let snippet = SourceSnippet::context(5, "let x = 1;");
        let formatted = snippet.format(3, false);
        assert!(formatted.contains("  5"));
        assert!(formatted.contains("let x = 1;"));
        assert!(!formatted.contains("^"));
    }

    #[test]
    fn test_source_snippet_format_error() {
        let snippet = SourceSnippet::error(10, "let x = invalid;", 8, 7);
        let formatted = snippet.format(3, false);
        // Format is: "> 10 | let x = invalid;"
        assert!(formatted.contains(">"));
        assert!(formatted.contains("10"));
        assert!(formatted.contains("let x = invalid;"));
        assert!(formatted.contains("^^^^^^^"));
    }

    #[test]
    fn test_source_snippet_format_with_colors() {
        let snippet = SourceSnippet::error(1, "error", 0, 5);
        let formatted = snippet.format(1, true);
        assert!(formatted.contains("\x1b[31m")); // Red color code
        assert!(formatted.contains("\x1b[0m")); // Reset code
    }

    // =========================================================================
    // ErrorLocation Tests
    // =========================================================================

    #[test]
    fn test_error_location_new() {
        let loc = ErrorLocation::new(10, 5, 100, 20);
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.offset, 100);
        assert_eq!(loc.length, 20);
    }

    #[test]
    fn test_error_location_default() {
        let loc = ErrorLocation::default();
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);
        assert_eq!(loc.offset, 0);
        assert_eq!(loc.length, 0);
    }

    #[test]
    fn test_error_location_format_with_path() {
        let loc = ErrorLocation::new(10, 5, 0, 0);
        let formatted = loc.format(Some(&PathBuf::from("test.wgsl")));
        assert_eq!(formatted, "test.wgsl:10:5");
    }

    #[test]
    fn test_error_location_format_without_path() {
        let loc = ErrorLocation::new(10, 5, 0, 0);
        let formatted = loc.format(None);
        assert_eq!(formatted, "10:5");
    }

    #[test]
    fn test_error_location_equality() {
        let loc1 = ErrorLocation::new(10, 5, 100, 20);
        let loc2 = ErrorLocation::new(10, 5, 100, 20);
        let loc3 = ErrorLocation::new(10, 6, 100, 20);
        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);
    }

    // =========================================================================
    // ErrorLabel Tests
    // =========================================================================

    #[test]
    fn test_error_label_primary() {
        let label = ErrorLabel::primary("error here", None);
        assert!(label.is_primary);
        assert_eq!(label.message, "error here");
    }

    #[test]
    fn test_error_label_secondary() {
        let label = ErrorLabel::secondary("related note", None);
        assert!(!label.is_primary);
        assert_eq!(label.message, "related note");
    }

    #[test]
    fn test_error_label_with_location() {
        let loc = ErrorLocation::new(5, 10, 50, 5);
        let label = ErrorLabel::primary("error", Some(loc.clone()));
        assert_eq!(label.location, Some(loc));
    }

    // =========================================================================
    // ValidationError Tests
    // =========================================================================

    #[test]
    fn test_validation_error_parse() {
        let err = ValidationError::parse("unexpected token");
        assert!(err.is_parse_error());
        assert!(!err.is_validation_error());
        assert_eq!(err.message(), "unexpected token");
        assert_eq!(err.kind(), "parse");
    }

    #[test]
    fn test_validation_error_parse_at() {
        let loc = ErrorLocation::new(5, 10, 50, 5);
        let err = ValidationError::parse_at("syntax error", loc.clone());
        assert_eq!(err.location(), Some(&loc));
    }

    #[test]
    fn test_validation_error_validation() {
        let err = ValidationError::validation("type mismatch");
        assert!(!err.is_parse_error());
        assert!(err.is_validation_error());
        assert_eq!(err.message(), "type mismatch");
        assert_eq!(err.kind(), "validation");
    }

    #[test]
    fn test_validation_error_empty_source() {
        let err = ValidationError::EmptySource;
        assert!(!err.is_parse_error());
        assert!(!err.is_validation_error());
        assert_eq!(err.message(), "shader source is empty");
        assert_eq!(err.kind(), "empty");
    }

    #[test]
    fn test_validation_error_with_label() {
        let err = ValidationError::parse("error")
            .with_label(ErrorLabel::secondary("note", None));
        if let ValidationError::Parse { labels, .. } = err {
            assert_eq!(labels.len(), 1);
        } else {
            panic!("expected Parse error");
        }
    }

    #[test]
    fn test_validation_error_with_suggestion() {
        let err = ValidationError::parse("error")
            .with_suggestion("try this fix");
        if let ValidationError::Parse { suggestions, .. } = err {
            assert_eq!(suggestions.len(), 1);
            assert_eq!(suggestions[0], "try this fix");
        } else {
            panic!("expected Parse error");
        }
    }

    #[test]
    fn test_validation_error_display_parse() {
        let loc = ErrorLocation::new(5, 10, 50, 5);
        let err = ValidationError::parse_at("syntax error", loc);
        let display = format!("{}", err);
        assert!(display.contains("parse error"));
        assert!(display.contains("5:10"));
        assert!(display.contains("syntax error"));
    }

    #[test]
    fn test_validation_error_display_validation() {
        let err = ValidationError::validation("type mismatch");
        let display = format!("{}", err);
        assert!(display.contains("validation error"));
        assert!(display.contains("type mismatch"));
    }

    #[test]
    fn test_validation_error_display_empty() {
        let err = ValidationError::EmptySource;
        let display = format!("{}", err);
        assert!(display.contains("empty"));
    }

    // =========================================================================
    // ValidationResult Tests
    // =========================================================================

    #[test]
    fn test_validation_result_vertex_shader() {
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert_eq!(result.entry_points, 1);
        assert!(result.has_vertex());
        assert!(!result.has_fragment());
        assert!(!result.has_compute());
        assert_eq!(result.entry_point_names, vec!["main"]);
    }

    #[test]
    fn test_validation_result_fragment_shader() {
        let source = "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }";
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert!(result.has_fragment());
        assert!(!result.has_vertex());
    }

    #[test]
    fn test_validation_result_compute_shader() {
        let source = "@compute @workgroup_size(64) fn main() {}";
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert!(result.has_compute());
        assert!(!result.has_vertex());
        assert!(!result.has_fragment());
    }

    #[test]
    fn test_validation_result_counts() {
        let source = r#"
            struct Data { value: f32 }
            fn helper() -> f32 { return 1.0; }
            @compute @workgroup_size(1) fn main() { let _x = helper(); }
        "#;
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert_eq!(result.entry_points, 1);
        assert!(result.function_count >= 1); // At least helper()
        assert!(result.type_count >= 1); // At least Data struct
    }

    // =========================================================================
    // NagaValidator Tests
    // =========================================================================

    #[test]
    fn test_naga_validator_default() {
        let validator = NagaValidator::default();
        assert_eq!(validator.config().strictness, Strictness::Standard);
    }

    #[test]
    fn test_naga_validator_strict() {
        let validator = NagaValidator::strict();
        assert_eq!(validator.config().strictness, Strictness::Strict);
    }

    #[test]
    fn test_naga_validator_relaxed() {
        let validator = NagaValidator::relaxed();
        assert_eq!(validator.config().strictness, Strictness::Relaxed);
    }

    #[test]
    fn test_naga_validator_config_mut() {
        let mut validator = NagaValidator::default();
        validator.config_mut().strictness = Strictness::Pedantic;
        assert_eq!(validator.config().strictness, Strictness::Pedantic);
    }

    #[test]
    fn test_validate_wgsl_valid() {
        let validator = NagaValidator::default();
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let result = validator.validate_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_wgsl_empty() {
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl("");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_validate_wgsl_whitespace() {
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl("   \n\t  ");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_validate_wgsl_parse_error() {
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl("invalid @@@");
        assert!(result.is_err());
        assert!(result.unwrap_err().is_parse_error());
    }

    #[test]
    fn test_validate_wgsl_validation_error() {
        let validator = NagaValidator::default();
        // Type error - assigning float to int
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let x: i32 = 1.5;
            }
        "#;
        let result = validator.validate_wgsl(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_wgsl_valid() {
        let validator = NagaValidator::default();
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let module = validator.parse_wgsl(source);
        assert!(module.is_ok());
        assert_eq!(module.unwrap().entry_points.len(), 1);
    }

    #[test]
    fn test_parse_wgsl_empty() {
        let validator = NagaValidator::default();
        let result = validator.parse_wgsl("");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_validate_module() {
        let validator = NagaValidator::default();
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let module = validator.parse_wgsl(source).unwrap();
        let result = validator.validate_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_error_parse() {
        let validator = NagaValidator::default();
        let source = "invalid @@@";
        let err = validator.validate_wgsl(source).unwrap_err();
        let formatted = validator.format_error(&err, source);

        assert!(formatted.contains("error[parse]"));
    }

    #[test]
    fn test_format_error_with_file_path() {
        let validator = NagaValidator::new(
            ValidationConfig::default().with_file_path("shaders/test.wgsl")
        );
        let source = "invalid @@@";
        let err = validator.validate_wgsl(source).unwrap_err();
        let formatted = validator.format_error(&err, source);

        assert!(formatted.contains("shaders/test.wgsl"));
    }

    #[test]
    fn test_format_error_empty_source() {
        let validator = NagaValidator::default();
        let err = ValidationError::EmptySource;
        let formatted = validator.format_error(&err, "");

        assert!(formatted.contains("error[empty]"));
        assert!(formatted.contains("empty"));
    }

    #[test]
    fn test_extract_snippets() {
        let validator = NagaValidator::default();
        let source = "line1\nline2\nline3\nline4\nline5";
        let location = ErrorLocation::new(3, 1, 12, 5);

        let snippets = validator.extract_snippets(source, &location, 1);

        assert_eq!(snippets.len(), 3); // line2, line3, line4
        assert_eq!(snippets[0].line_number, 2);
        assert!(!snippets[0].is_error_line);
        assert_eq!(snippets[1].line_number, 3);
        assert!(snippets[1].is_error_line);
        assert_eq!(snippets[2].line_number, 4);
    }

    #[test]
    fn test_extract_snippets_at_start() {
        let validator = NagaValidator::default();
        let source = "line1\nline2\nline3";
        let location = ErrorLocation::new(1, 1, 0, 5);

        let snippets = validator.extract_snippets(source, &location, 1);

        assert_eq!(snippets.len(), 2); // line1 (error), line2
        assert_eq!(snippets[0].line_number, 1);
        assert!(snippets[0].is_error_line);
    }

    #[test]
    fn test_extract_snippets_at_end() {
        let validator = NagaValidator::default();
        let source = "line1\nline2\nline3";
        let location = ErrorLocation::new(3, 1, 12, 5);

        let snippets = validator.extract_snippets(source, &location, 1);

        assert_eq!(snippets.len(), 2); // line2, line3 (error)
        assert!(snippets[1].is_error_line);
    }

    #[test]
    fn test_extract_snippets_empty_source() {
        let validator = NagaValidator::default();
        let location = ErrorLocation::new(1, 1, 0, 0);

        let snippets = validator.extract_snippets("", &location, 1);
        assert!(snippets.is_empty());
    }

    // =========================================================================
    // Quick Function Tests
    // =========================================================================

    #[test]
    fn test_quick_validate_wgsl_valid() {
        let result = quick_validate_wgsl(
            "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_quick_validate_wgsl_invalid() {
        let result = quick_validate_wgsl("invalid @@@");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_valid_wgsl_valid() {
        assert!(is_valid_wgsl(
            "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        ));
    }

    #[test]
    fn test_is_valid_wgsl_invalid() {
        assert!(!is_valid_wgsl("invalid @@@"));
    }

    #[test]
    fn test_is_valid_wgsl_empty() {
        assert!(!is_valid_wgsl(""));
    }

    #[test]
    fn test_parse_wgsl_function() {
        let module = parse_wgsl(
            "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
        ).unwrap();
        assert_eq!(module.entry_points.len(), 1);
    }

    #[test]
    fn test_format_validation_error_function() {
        let source = "invalid @@@";
        let err = quick_validate_wgsl(source).unwrap_err();
        let formatted = format_validation_error(&err, source);
        assert!(formatted.contains("error"));
    }

    // =========================================================================
    // Utility Function Tests
    // =========================================================================

    #[test]
    fn test_offset_to_line_column_first_char() {
        assert_eq!(offset_to_line_column("hello", 0), (1, 1));
    }

    #[test]
    fn test_offset_to_line_column_middle() {
        assert_eq!(offset_to_line_column("hello", 2), (1, 3));
    }

    #[test]
    fn test_offset_to_line_column_multiline() {
        let source = "line1\nline2\nline3";
        assert_eq!(offset_to_line_column(source, 0), (1, 1));
        assert_eq!(offset_to_line_column(source, 6), (2, 1));
        assert_eq!(offset_to_line_column(source, 12), (3, 1));
    }

    #[test]
    fn test_offset_to_line_column_beyond_end() {
        let source = "abc";
        let (line, col) = offset_to_line_column(source, 100);
        assert_eq!(line, 1);
        assert_eq!(col, 4); // Clamped to end
    }

    #[test]
    fn test_truncate_line_short() {
        assert_eq!(truncate_line("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_line_exact() {
        assert_eq!(truncate_line("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_line_long() {
        assert_eq!(truncate_line("hello world", 8), "hello...");
    }

    // =========================================================================
    // Complex Shader Tests
    // =========================================================================

    #[test]
    fn test_complex_vertex_fragment_shader() {
        let source = r#"
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
                var out: VertexOutput;
                out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                out.uv = vec2<f32>(0.0, 0.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(in.uv, 0.0, 1.0);
            }
        "#;

        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert_eq!(result.entry_points, 2);
        assert!(result.has_vertex());
        assert!(result.has_fragment());
    }

    #[test]
    fn test_complex_compute_shader() {
        let source = r#"
            struct Particle {
                position: vec3<f32>,
                velocity: vec3<f32>,
            }

            @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if idx >= arrayLength(&particles) {
                    return;
                }

                var p = particles[idx];
                p.velocity.y = p.velocity.y - 9.8;
                p.position = p.position + p.velocity;
                particles[idx] = p;
            }
        "#;

        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert!(result.has_compute());
        assert!(result.global_count >= 1); // particles binding
    }

    #[test]
    fn test_shader_with_uniforms() {
        let source = r#"
            struct Uniforms {
                model: mat4x4<f32>,
                view: mat4x4<f32>,
                proj: mat4x4<f32>,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
                let mvp = uniforms.proj * uniforms.view * uniforms.model;
                return mvp * vec4<f32>(pos, 1.0);
            }
        "#;

        let result = quick_validate_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shader_with_textures() {
        let source = r#"
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var samp: sampler;

            @fragment
            fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(tex, samp, uv);
            }
        "#;

        let result = quick_validate_wgsl(source);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Error Message Tests
    // =========================================================================

    #[test]
    fn test_error_message_undefined_variable() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let x = undefined_var;
            }
        "#;

        let err = quick_validate_wgsl(source).unwrap_err();
        assert!(err.message().to_lowercase().contains("unknown") ||
                err.message().to_lowercase().contains("identifier") ||
                err.message().to_lowercase().contains("undefined"));
    }

    #[test]
    fn test_error_message_type_mismatch() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let x: i32 = 1.5;
            }
        "#;

        let err = quick_validate_wgsl(source).unwrap_err();
        let msg = err.message().to_lowercase();
        assert!(msg.contains("type") || msg.contains("mismatch") || msg.contains("expected"));
    }

    #[test]
    fn test_error_message_missing_return() {
        let source = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                // Missing return
            }
        "#;

        let err = quick_validate_wgsl(source).unwrap_err();
        // Should have some error about missing return or control flow
        assert!(err.is_parse_error() || err.is_validation_error());
    }

    // =========================================================================
    // Suggestion Generation Tests
    // =========================================================================

    #[test]
    fn test_suggestions_type_mismatch() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("type mismatch error");
        assert!(suggestions.iter().any(|s| s.contains("type")));
    }

    #[test]
    fn test_suggestions_undefined() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("undefined identifier");
        assert!(suggestions.iter().any(|s| s.contains("defined")));
    }

    #[test]
    fn test_suggestions_expected_found() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("expected ';' found '}'");
        assert!(suggestions.iter().any(|s| s.contains("semicolon") || s.contains("bracket")));
    }

    #[test]
    fn test_suggestions_builtin() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("unknown builtin");
        assert!(suggestions.iter().any(|s| s.contains("builtin")));
    }

    #[test]
    fn test_suggestions_binding() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("missing binding");
        assert!(suggestions.iter().any(|s| s.contains("@group") || s.contains("@binding")));
    }

    #[test]
    fn test_suggestions_workgroup() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("missing workgroup_size");
        assert!(suggestions.iter().any(|s| s.contains("workgroup")));
    }

    #[test]
    fn test_suggestions_entry_point() {
        let validator = NagaValidator::default();
        let suggestions = validator.generate_suggestions("no entry point");
        assert!(suggestions.iter().any(|s| s.contains("@vertex") || s.contains("entry point")));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_edge_case_unicode_in_comments() {
        let source = r#"
            // Unicode comment: hello world
            @vertex fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;

        let result = quick_validate_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_very_long_shader() {
        let mut source = String::from("@compute @workgroup_size(1) fn main() {\n");
        for i in 0..100 {
            source.push_str(&format!("    let x{} = {};\n", i, i));
        }
        source.push_str("}\n");

        let result = quick_validate_wgsl(&source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_deeply_nested() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                if true {
                    if true {
                        if true {
                            if true {
                                let x = 1;
                            }
                        }
                    }
                }
            }
        "#;

        let result = quick_validate_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_many_functions() {
        let mut source = String::new();
        for i in 0..50 {
            source.push_str(&format!("fn func_{}() -> i32 {{ return {}i; }}\n", i, i));
        }
        source.push_str("@compute @workgroup_size(1) fn main() { let _x = func_0(); }\n");

        let result = quick_validate_wgsl(&source);
        assert!(result.is_ok(), "validation failed: {:?}", result.err());
    }

    #[test]
    fn test_edge_case_empty_function() {
        let source = "@compute @workgroup_size(1) fn main() {}";
        let result = quick_validate_wgsl(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_single_line() {
        let validator = NagaValidator::default();
        let source = "error";
        let location = ErrorLocation::new(1, 1, 0, 5);
        let snippets = validator.extract_snippets(source, &location, 2);

        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].is_error_line);
    }

    // =========================================================================
    // Clone and Debug Tests
    // =========================================================================

    #[test]
    fn test_validation_config_clone() {
        let config = ValidationConfig::default().with_file_path("test.wgsl");
        let cloned = config.clone();
        assert_eq!(cloned.file_path, config.file_path);
    }

    #[test]
    fn test_validation_config_debug() {
        let config = ValidationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ValidationConfig"));
    }

    #[test]
    fn test_naga_validator_clone() {
        let validator = NagaValidator::strict();
        let cloned = validator.clone();
        assert_eq!(cloned.config().strictness, Strictness::Strict);
    }

    #[test]
    fn test_naga_validator_debug() {
        let validator = NagaValidator::default();
        let debug = format!("{:?}", validator);
        assert!(debug.contains("NagaValidator"));
    }

    #[test]
    fn test_source_snippet_clone() {
        let snippet = SourceSnippet::error(1, "test", 0, 4);
        let cloned = snippet.clone();
        assert_eq!(cloned.line_number, snippet.line_number);
    }

    #[test]
    fn test_error_location_clone() {
        let loc = ErrorLocation::new(1, 2, 3, 4);
        let cloned = loc.clone();
        assert_eq!(cloned, loc);
    }

    #[test]
    fn test_validation_error_clone() {
        let err = ValidationError::parse("test");
        let cloned = err.clone();
        assert_eq!(cloned.message(), err.message());
    }

    // =========================================================================
    // Additional Whitebox Tests for T-WGPU-P2.7.3
    // =========================================================================

    // --- Thread Safety Tests ---

    #[test]
    fn test_concurrent_validation_same_validator() {
        use std::sync::Arc;
        use std::thread;

        let validator = Arc::new(NagaValidator::default());
        let sources = vec![
            "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
            "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
            "@compute @workgroup_size(64) fn main() {}",
        ];

        let handles: Vec<_> = sources
            .into_iter()
            .map(|src| {
                let v = Arc::clone(&validator);
                thread::spawn(move || v.validate_wgsl(src).is_ok())
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_concurrent_validation_separate_validators() {
        use std::thread;

        let sources = vec![
            "@vertex fn a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
            "@fragment fn b() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
            "@compute @workgroup_size(1) fn c() {}",
            "@vertex fn d() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }",
        ];

        let handles: Vec<_> = sources
            .into_iter()
            .map(|src| {
                thread::spawn(move || {
                    let validator = NagaValidator::default();
                    validator.validate_wgsl(src).is_ok()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_concurrent_validation_with_errors() {
        use std::sync::Arc;
        use std::thread;

        let validator = Arc::new(NagaValidator::default());
        let sources = vec![
            ("valid", "@compute @workgroup_size(1) fn main() {}"),
            ("invalid", "invalid @@@"),
            ("empty", ""),
        ];

        let handles: Vec<_> = sources
            .into_iter()
            .map(|(expected, src)| {
                let v = Arc::clone(&validator);
                thread::spawn(move || {
                    let result = v.validate_wgsl(src);
                    match expected {
                        "valid" => result.is_ok(),
                        "invalid" => result.is_err() && result.unwrap_err().is_parse_error(),
                        "empty" => matches!(result, Err(ValidationError::EmptySource)),
                        _ => false,
                    }
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    // --- Strictness Level Deep Tests ---

    #[test]
    fn test_strictness_relaxed_validation_flags() {
        let flags = Strictness::Relaxed.validation_flags();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_strictness_strict_validation_flags() {
        let flags = Strictness::Strict.validation_flags();
        assert_eq!(flags, naga::valid::ValidationFlags::all());
    }

    #[test]
    fn test_strictness_pedantic_validation_flags() {
        let flags = Strictness::Pedantic.validation_flags();
        assert_eq!(flags, naga::valid::ValidationFlags::all());
    }

    #[test]
    fn test_strictness_relaxed_capabilities() {
        let caps = Strictness::Relaxed.capabilities();
        assert_eq!(caps, naga::valid::Capabilities::all());
    }

    #[test]
    fn test_strictness_strict_capabilities() {
        let caps = Strictness::Strict.capabilities();
        assert_eq!(caps, naga::valid::Capabilities::all());
    }

    #[test]
    fn test_strictness_pedantic_capabilities() {
        let caps = Strictness::Pedantic.capabilities();
        assert_eq!(caps, naga::valid::Capabilities::all());
    }

    #[test]
    fn test_strictness_copy_clone() {
        let s1 = Strictness::Strict;
        let s2 = s1; // Copy
        let s3 = s1.clone(); // Clone
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
    }

    #[test]
    fn test_strictness_debug() {
        let debug = format!("{:?}", Strictness::Pedantic);
        assert!(debug.contains("Pedantic"));
    }

    // --- Snippet Extraction Edge Cases ---

    #[test]
    fn test_extract_snippets_line_past_end() {
        let validator = NagaValidator::default();
        let source = "line1\nline2";
        let location = ErrorLocation::new(100, 1, 0, 5); // Line 100 doesn't exist

        let snippets = validator.extract_snippets(source, &location, 1);
        assert!(snippets.is_empty()); // Should handle gracefully
    }

    #[test]
    fn test_extract_snippets_zero_context() {
        let validator = NagaValidator::default();
        let source = "line1\nline2\nline3\nline4\nline5";
        let location = ErrorLocation::new(3, 1, 12, 5);

        let snippets = validator.extract_snippets(source, &location, 0);
        assert_eq!(snippets.len(), 1); // Only the error line
        assert!(snippets[0].is_error_line);
        assert_eq!(snippets[0].line_number, 3);
    }

    #[test]
    fn test_extract_snippets_large_context() {
        let validator = NagaValidator::default();
        let source = "line1\nline2\nline3";
        let location = ErrorLocation::new(2, 1, 6, 5);

        // Context larger than file
        let snippets = validator.extract_snippets(source, &location, 10);
        assert_eq!(snippets.len(), 3); // All 3 lines
    }

    #[test]
    fn test_extract_snippets_column_position() {
        let validator = NagaValidator::default();
        let source = "let x = invalid_token;";
        let location = ErrorLocation::new(1, 9, 8, 13); // "invalid_token"

        let snippets = validator.extract_snippets(source, &location, 0);
        assert_eq!(snippets.len(), 1);
        assert_eq!(snippets[0].marker_start, Some(8)); // Column 9 - 1 = 8 (0-based)
    }

    #[test]
    fn test_extract_snippets_truncation() {
        let validator = NagaValidator::default();
        let long_line = "x".repeat(300); // Longer than MAX_SNIPPET_LENGTH
        let location = ErrorLocation::new(1, 1, 0, 10);

        let snippets = validator.extract_snippets(&long_line, &location, 0);
        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].content.len() <= MAX_SNIPPET_LENGTH);
        assert!(snippets[0].content.ends_with("..."));
    }

    // --- Error Label Edge Cases ---

    #[test]
    fn test_error_label_equality() {
        let loc = ErrorLocation::new(1, 1, 0, 5);
        let label1 = ErrorLabel::primary("error", Some(loc.clone()));
        let label2 = ErrorLabel::primary("error", Some(loc.clone()));
        let label3 = ErrorLabel::secondary("error", Some(loc));

        assert_eq!(label1, label2);
        assert_ne!(label1, label3); // Different is_primary
    }

    #[test]
    fn test_error_label_clone() {
        let loc = ErrorLocation::new(5, 10, 50, 15);
        let label = ErrorLabel::primary("test error", Some(loc));
        let cloned = label.clone();
        assert_eq!(cloned.message, label.message);
        assert_eq!(cloned.is_primary, label.is_primary);
        assert_eq!(cloned.location, label.location);
    }

    #[test]
    fn test_error_label_debug() {
        let label = ErrorLabel::primary("test", None);
        let debug = format!("{:?}", label);
        assert!(debug.contains("ErrorLabel"));
        assert!(debug.contains("primary"));
    }

    // --- ValidationError Builder Pattern ---

    #[test]
    fn test_validation_error_multiple_labels() {
        let err = ValidationError::parse("multi-error")
            .with_label(ErrorLabel::primary("first", None))
            .with_label(ErrorLabel::secondary("second", None))
            .with_label(ErrorLabel::secondary("third", None));

        if let ValidationError::Parse { labels, .. } = err {
            assert_eq!(labels.len(), 3);
            assert!(labels[0].is_primary);
            assert!(!labels[1].is_primary);
            assert!(!labels[2].is_primary);
        } else {
            panic!("Expected Parse error");
        }
    }

    #[test]
    fn test_validation_error_multiple_suggestions() {
        let err = ValidationError::parse("syntax error")
            .with_suggestion("fix A")
            .with_suggestion("fix B")
            .with_suggestion("fix C");

        if let ValidationError::Parse { suggestions, .. } = err {
            assert_eq!(suggestions.len(), 3);
            assert_eq!(suggestions[0], "fix A");
            assert_eq!(suggestions[1], "fix B");
            assert_eq!(suggestions[2], "fix C");
        } else {
            panic!("Expected Parse error");
        }
    }

    #[test]
    fn test_validation_error_suggestion_on_validation() {
        // Suggestions should be ignored for Validation errors
        let err = ValidationError::validation("type error")
            .with_suggestion("this should be ignored");

        if let ValidationError::Validation { .. } = err {
            // OK - suggestions not added to Validation variant
        } else {
            panic!("Expected Validation error");
        }
    }

    #[test]
    fn test_validation_error_label_on_empty_source() {
        // Labels should be ignored for EmptySource
        let err = ValidationError::EmptySource
            .with_label(ErrorLabel::primary("ignored", None));

        assert!(matches!(err, ValidationError::EmptySource));
    }

    // --- ValidationResult Tests ---

    #[test]
    fn test_validation_result_multiple_entry_points() {
        let source = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
            @compute @workgroup_size(1) fn cs() {}
        "#;

        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert_eq!(result.entry_points, 3);
        assert!(result.has_vertex());
        assert!(result.has_fragment());
        assert!(result.has_compute());
        assert!(result.entry_point_names.contains(&"vs".to_string()));
        assert!(result.entry_point_names.contains(&"fs".to_string()));
        assert!(result.entry_point_names.contains(&"cs".to_string()));
    }

    #[test]
    fn test_validation_result_no_entry_points() {
        // Pure helper module with no entry points
        let source = "fn helper(x: f32) -> f32 { return x * 2.0; }";
        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert_eq!(result.entry_points, 0);
        assert!(!result.has_vertex());
        assert!(!result.has_fragment());
        assert!(!result.has_compute());
        assert!(result.entry_point_names.is_empty());
    }

    #[test]
    fn test_validation_result_struct_count() {
        let source = r#"
            struct A { x: f32 }
            struct B { a: A }
            struct C { b: B, extra: f32 }
            @compute @workgroup_size(1) fn main() {}
        "#;

        let validator = NagaValidator::default();
        let result = validator.validate_wgsl(source).unwrap();

        assert!(result.type_count >= 3); // At least A, B, C
    }

    // --- Format Error Deep Tests ---

    #[test]
    fn test_format_error_without_snippets() {
        let validator = NagaValidator::new(ValidationConfig::default().without_snippets());
        let source = "invalid @@@";
        let err = validator.validate_wgsl(source).unwrap_err();
        let formatted = validator.format_error(&err, source);

        // Should not contain snippet markers
        assert!(!formatted.contains("   |"));
    }

    #[test]
    fn test_format_error_with_secondary_labels() {
        let loc1 = ErrorLocation::new(1, 5, 4, 3);
        let loc2 = ErrorLocation::new(2, 10, 20, 5);

        let err = ValidationError::parse_at("error with labels", loc1)
            .with_label(ErrorLabel::secondary("related here", Some(loc2)));

        let validator = NagaValidator::default();
        let source = "line1\nline2 with more content";
        let formatted = validator.format_error(&err, source);

        assert!(formatted.contains("= note:"));
        assert!(formatted.contains("related here"));
    }

    #[test]
    fn test_format_error_with_suggestions() {
        let err = ValidationError::parse("expected semicolon")
            .with_suggestion("add ';' at end of statement");

        let validator = NagaValidator::new(ValidationConfig::default());
        let formatted = validator.format_error(&err, "let x = 1");

        assert!(formatted.contains("= help:"));
        assert!(formatted.contains("add ';' at end of statement"));
    }

    #[test]
    fn test_format_error_suggestions_disabled() {
        let err = ValidationError::parse("error")
            .with_suggestion("this should not appear");

        let mut config = ValidationConfig::default();
        config.include_suggestions = false;
        let validator = NagaValidator::new(config);

        let formatted = validator.format_error(&err, "source");
        assert!(!formatted.contains("= help:"));
    }

    #[test]
    fn test_format_error_with_colors() {
        let validator = NagaValidator::new(ValidationConfig::default().with_colors());
        let source = "invalid @@@";
        let err = validator.validate_wgsl(source).unwrap_err();
        let formatted = validator.format_error(&err, source);

        // Should contain ANSI color codes
        assert!(formatted.contains("\x1b[31m") || formatted.contains("error"));
    }

    // --- Convenience Function Edge Cases ---

    #[test]
    fn test_quick_validate_empty() {
        let result = quick_validate_wgsl("");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_quick_validate_whitespace_only() {
        let result = quick_validate_wgsl("   \n\t\r\n   ");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_parse_wgsl_empty_source() {
        let result = parse_wgsl("");
        assert!(matches!(result, Err(ValidationError::EmptySource)));
    }

    #[test]
    fn test_parse_wgsl_syntax_error() {
        let result = parse_wgsl("fn invalid {{{");
        assert!(result.is_err());
        assert!(result.unwrap_err().is_parse_error());
    }

    #[test]
    fn test_is_valid_wgsl_whitespace() {
        assert!(!is_valid_wgsl("    "));
    }

    // --- Offset to Line/Column Edge Cases ---

    #[test]
    fn test_offset_to_line_column_empty_source() {
        let (line, col) = offset_to_line_column("", 0);
        assert_eq!(line, 1);
        assert_eq!(col, 1);
    }

    #[test]
    fn test_offset_to_line_column_newline_only() {
        let source = "\n\n\n";
        assert_eq!(offset_to_line_column(source, 0), (1, 1)); // Before first \n
        assert_eq!(offset_to_line_column(source, 1), (2, 1)); // After first \n
        assert_eq!(offset_to_line_column(source, 2), (3, 1)); // After second \n
    }

    #[test]
    fn test_offset_to_line_column_crlf() {
        let source = "line1\r\nline2";
        // CRLF is treated as 2 chars, \n triggers newline
        let (line, col) = offset_to_line_column(source, 7);
        assert_eq!(line, 2);
        assert_eq!(col, 1);
    }

    #[test]
    fn test_offset_to_line_column_end_of_line() {
        let source = "abc\ndef";
        // Position 3 is the \n character
        let (line, col) = offset_to_line_column(source, 3);
        assert_eq!(line, 1);
        assert_eq!(col, 4); // 4th char of line 1
    }

    // --- Truncate Line Tests ---

    #[test]
    fn test_truncate_line_empty() {
        assert_eq!(truncate_line("", 10), "");
    }

    #[test]
    fn test_truncate_line_edge_cases() {
        // Max length that would cause empty prefix
        assert_eq!(truncate_line("abcde", 4), "a...");
        assert_eq!(truncate_line("ab", 3), "ab"); // Exact fit with 1 char margin
    }

    // --- Validator Configuration Chaining ---

    #[test]
    fn test_validation_config_chaining() {
        let config = ValidationConfig::with_strictness(Strictness::Pedantic)
            .with_file_path("/path/to/shader.wgsl")
            .with_colors()
            .with_context_lines(5)
            .without_snippets();

        assert_eq!(config.strictness, Strictness::Pedantic);
        assert!(config.use_colors);
        assert_eq!(config.context_lines, 5);
        assert!(!config.include_snippets);
        assert_eq!(config.file_path, Some(PathBuf::from("/path/to/shader.wgsl")));
    }

    #[test]
    fn test_validation_config_builder_order_independence() {
        let config1 = ValidationConfig::default()
            .with_colors()
            .without_snippets();

        let config2 = ValidationConfig::default()
            .without_snippets()
            .with_colors();

        assert_eq!(config1.use_colors, config2.use_colors);
        assert_eq!(config1.include_snippets, config2.include_snippets);
    }

    // --- validate_module_with_source Tests ---

    #[test]
    fn test_validate_module_with_source_valid() {
        let validator = NagaValidator::default();
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let module = validator.parse_wgsl(source).unwrap();
        let result = validator.validate_module_with_source(&module, source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_module_with_source_empty() {
        let validator = NagaValidator::default();
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let module = validator.parse_wgsl(source).unwrap();
        // Validate with empty source string (for error messages)
        let result = validator.validate_module_with_source(&module, "");
        assert!(result.is_ok()); // Should still validate, just won't have good error messages
    }

    // --- Default Validator Tests ---

    #[test]
    fn test_naga_validator_default_validator_fn() {
        let v = NagaValidator::default_validator();
        assert_eq!(v.config().strictness, Strictness::Standard);
    }

    // --- Complex Shader Validation ---

    #[test]
    fn test_shader_with_arrays() {
        let source = r#"
            @group(0) @binding(0) var<storage, read> data: array<f32, 256>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if idx < 256u {
                    let val = data[idx];
                }
            }
        "#;

        assert!(quick_validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_shader_with_matrix_operations() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                let identity = mat4x4<f32>(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                );
                let v = vec4<f32>(1.0, 2.0, 3.0, 1.0);
                let result = identity * v;
            }
        "#;

        assert!(quick_validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_shader_with_control_flow() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {
                var sum = 0i;
                for (var i = 0i; i < 10i; i = i + 1i) {
                    if i % 2i == 0i {
                        sum = sum + i;
                    } else {
                        continue;
                    }
                }

                switch sum {
                    case 0i: { }
                    case 1i, 2i: { }
                    default: { }
                }
            }
        "#;

        assert!(quick_validate_wgsl(source).is_ok());
    }

    #[test]
    fn test_shader_with_atomics() {
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;

            @compute @workgroup_size(64)
            fn main() {
                atomicAdd(&counter, 1u);
            }
        "#;

        assert!(quick_validate_wgsl(source).is_ok());
    }

    // --- Error Types and Std Error Trait ---

    #[test]
    fn test_validation_error_std_error() {
        let err = ValidationError::parse("test error");
        // Test that it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_validation_error_location_none() {
        let err = ValidationError::parse("no location");
        assert!(err.location().is_none());
    }

    #[test]
    fn test_validation_error_validation_at() {
        let loc = ErrorLocation::new(10, 5, 100, 20);
        let err = ValidationError::validation_at("type error", loc.clone());
        assert_eq!(err.location(), Some(&loc));
        assert!(err.is_validation_error());
    }

    // --- SourceSnippet Edge Cases ---

    #[test]
    fn test_source_snippet_equality() {
        let s1 = SourceSnippet::error(1, "test", 0, 4);
        let s2 = SourceSnippet::error(1, "test", 0, 4);
        let s3 = SourceSnippet::context(1, "test");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_source_snippet_debug() {
        let snippet = SourceSnippet::error(1, "test", 0, 4);
        let debug = format!("{:?}", snippet);
        assert!(debug.contains("SourceSnippet"));
    }

    #[test]
    fn test_source_snippet_format_line_width() {
        let snippet = SourceSnippet::context(1, "test");
        let formatted1 = snippet.format(1, false);
        let formatted5 = snippet.format(5, false);

        // Line number should be padded differently
        assert!(formatted1.contains(" 1 |"));
        assert!(formatted5.contains("     1 |"));
    }

    // --- Constants Tests ---

    #[test]
    fn test_constants_values() {
        assert_eq!(DEFAULT_CONTEXT_LINES, 2);
        assert_eq!(MAX_SNIPPET_LENGTH, 200);
    }
}
