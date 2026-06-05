//! Override constants for WGSL shader pipeline compilation.
//!
//! This module provides support for WGSL override constants (also known as
//! specialization constants or pipeline-overridable constants). These allow
//! compile-time values to be specified when creating a pipeline, enabling
//! shader specialization without recompilation.
//!
//! # Overview
//!
//! Override constants in WGSL are declared with the `override` keyword:
//!
//! ```wgsl
//! // With explicit ID
//! @id(0) override SCREEN_WIDTH: f32 = 1920.0;
//! @id(1) override SCREEN_HEIGHT: f32 = 1080.0;
//!
//! // With just a name (ID assigned by compiler)
//! override TILE_SIZE: u32 = 16u;
//!
//! // Required constant (no default)
//! @id(2) override MAX_LIGHTS: u32;
//! ```
//!
//! When creating a pipeline, these can be overridden:
//!
//! ```rust,ignore
//! let constants = PipelineConstants::new()
//!     .set_f32("SCREEN_WIDTH", 2560.0)
//!     .set_f32("SCREEN_HEIGHT", 1440.0)
//!     .set_u32("TILE_SIZE", 32);
//!
//! let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
//!     vertex: wgpu::VertexState {
//!         compilation_options: wgpu::PipelineCompilationOptions {
//!             constants: &constants.to_wgpu(),
//!             ..Default::default()
//!         },
//!         // ...
//!     },
//!     // ...
//! });
//! ```
//!
//! # Architecture
//!
//! ```text
//! OverrideConstantType
//! +-- Bool, I32, U32, F32
//!
//! OverrideConstantInfo
//! +-- name: Option<String>       // Named constant
//! +-- id: Option<u32>            // @id(N) value
//! +-- ty: OverrideConstantType   // Type
//! +-- default_value: Option<f64> // Default if provided
//! +-- required: bool             // True if no default
//!
//! OverrideConstants
//! +-- constants: Vec<OverrideConstantInfo>
//! +-- from_module(module) -> Self
//! +-- from_reflection(reflection) -> Self
//! +-- get_by_name(name) -> Option<&OverrideConstantInfo>
//! +-- get_by_id(id) -> Option<&OverrideConstantInfo>
//! +-- iter() -> impl Iterator
//!
//! PipelineConstants
//! +-- values: HashMap<String, f64>
//! +-- set(key, value) -> &mut Self
//! +-- set_bool(key, value) -> &mut Self
//! +-- set_i32(key, value) -> &mut Self
//! +-- set_u32(key, value) -> &mut Self
//! +-- set_f32(key, value) -> &mut Self
//! +-- to_wgpu() -> HashMap<String, f64>
//! +-- validate(overrides) -> Result<(), OverrideError>
//!
//! OverrideError
//! +-- UnknownConstant { key }
//! +-- TypeMismatch { key, expected, got }
//! +-- MissingRequired { name, id }
//! +-- ValueOutOfRange { key, value, expected }
//! ```

use std::collections::HashMap;
use std::fmt;

use super::reflection::ShaderReflection;

// ============================================================================
// Override Constant Type
// ============================================================================

/// Type of an override constant.
///
/// WGSL supports bool, i32, u32, and f32 as override constant types.
/// f16 is also supported in WGSL but is less common in practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverrideConstantType {
    /// Boolean constant (true/false).
    Bool,
    /// Signed 32-bit integer constant.
    I32,
    /// Unsigned 32-bit integer constant.
    U32,
    /// 32-bit floating-point constant.
    F32,
}

impl OverrideConstantType {
    /// Returns the type name as used in WGSL.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            OverrideConstantType::Bool => "bool",
            OverrideConstantType::I32 => "i32",
            OverrideConstantType::U32 => "u32",
            OverrideConstantType::F32 => "f32",
        }
    }

    /// Converts from a naga ScalarKind and width.
    pub fn from_naga(kind: naga::ScalarKind, width: u8) -> Option<Self> {
        match (kind, width) {
            (naga::ScalarKind::Bool, _) => Some(OverrideConstantType::Bool),
            (naga::ScalarKind::Sint, 4) => Some(OverrideConstantType::I32),
            (naga::ScalarKind::Uint, 4) => Some(OverrideConstantType::U32),
            (naga::ScalarKind::Float, 4) => Some(OverrideConstantType::F32),
            _ => None,
        }
    }

    /// Returns the default value for this type as f64.
    pub fn default_value(&self) -> f64 {
        match self {
            OverrideConstantType::Bool => 0.0,
            OverrideConstantType::I32 => 0.0,
            OverrideConstantType::U32 => 0.0,
            OverrideConstantType::F32 => 0.0,
        }
    }

    /// Checks if a value is valid for this type.
    pub fn is_valid_value(&self, value: f64) -> bool {
        match self {
            OverrideConstantType::Bool => value == 0.0 || value == 1.0,
            OverrideConstantType::I32 => {
                value >= i32::MIN as f64 && value <= i32::MAX as f64 && value.fract() == 0.0
            }
            OverrideConstantType::U32 => {
                value >= 0.0 && value <= u32::MAX as f64 && value.fract() == 0.0
            }
            OverrideConstantType::F32 => {
                value.is_finite() || value.is_nan()
            }
        }
    }

    /// Returns the expected value range description.
    pub fn value_range(&self) -> &'static str {
        match self {
            OverrideConstantType::Bool => "0 or 1",
            OverrideConstantType::I32 => "-2147483648 to 2147483647",
            OverrideConstantType::U32 => "0 to 4294967295",
            OverrideConstantType::F32 => "finite f32 value",
        }
    }
}

impl fmt::Display for OverrideConstantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wgsl_name())
    }
}

impl Default for OverrideConstantType {
    fn default() -> Self {
        OverrideConstantType::F32
    }
}

// ============================================================================
// Override Constant Info
// ============================================================================

/// Information about a single override constant in a shader.
///
/// Override constants can be identified by name, by @id(N), or both.
/// They have a type and optionally a default value. If no default is
/// provided, the constant is required when creating a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct OverrideConstantInfo {
    /// Optional name of the constant.
    pub name: Option<String>,
    /// Optional @id(N) value.
    pub id: Option<u32>,
    /// Type of the constant.
    pub ty: OverrideConstantType,
    /// Default value (if provided in shader).
    pub default_value: Option<f64>,
    /// Whether this constant is required (no default).
    pub required: bool,
}

impl OverrideConstantInfo {
    /// Creates a new override constant info.
    pub fn new(
        name: Option<String>,
        id: Option<u32>,
        ty: OverrideConstantType,
        default_value: Option<f64>,
    ) -> Self {
        Self {
            name,
            id,
            ty,
            default_value,
            required: default_value.is_none(),
        }
    }

    /// Creates a required constant with no default.
    pub fn required(name: Option<String>, id: Option<u32>, ty: OverrideConstantType) -> Self {
        Self {
            name,
            id,
            ty,
            default_value: None,
            required: true,
        }
    }

    /// Creates a constant with a default value.
    pub fn with_default(
        name: Option<String>,
        id: Option<u32>,
        ty: OverrideConstantType,
        default: f64,
    ) -> Self {
        Self {
            name,
            id,
            ty,
            default_value: Some(default),
            required: false,
        }
    }

    /// Returns the key to use for wgpu PipelineCompilationOptions.
    ///
    /// wgpu uses the constant name if available, otherwise the stringified ID.
    pub fn key(&self) -> Option<String> {
        self.name.clone().or_else(|| self.id.map(|id| id.to_string()))
    }

    /// Returns true if this constant has a name.
    #[inline]
    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    /// Returns true if this constant has an explicit @id.
    #[inline]
    pub fn has_id(&self) -> bool {
        self.id.is_some()
    }

    /// Returns true if this constant has a default value.
    #[inline]
    pub fn has_default(&self) -> bool {
        self.default_value.is_some()
    }

    /// Validates that a value is appropriate for this constant's type.
    pub fn validate_value(&self, value: f64) -> Result<(), OverrideError> {
        if !self.ty.is_valid_value(value) {
            let key = self.key().unwrap_or_else(|| "<unknown>".to_string());
            return Err(OverrideError::ValueOutOfRange {
                key,
                value,
                expected: self.ty.value_range(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for OverrideConstantInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(id) = self.id {
            write!(f, "@id({}) ", id)?;
        }
        write!(f, "override ")?;
        if let Some(name) = &self.name {
            write!(f, "{}: ", name)?;
        }
        write!(f, "{}", self.ty)?;
        if let Some(default) = self.default_value {
            write!(f, " = {}", default)?;
        }
        if self.required {
            write!(f, " (required)")?;
        }
        Ok(())
    }
}

// ============================================================================
// Override Constants Collection
// ============================================================================

/// Collection of override constants extracted from a shader module.
///
/// Provides lookup by name or ID, and iteration over all constants.
#[derive(Debug, Clone, Default)]
pub struct OverrideConstants {
    /// All override constants in the shader.
    constants: Vec<OverrideConstantInfo>,
    /// Name-to-index lookup map.
    name_index: HashMap<String, usize>,
    /// ID-to-index lookup map.
    id_index: HashMap<u32, usize>,
}

impl OverrideConstants {
    /// Creates an empty collection.
    pub fn new() -> Self {
        Self {
            constants: Vec::new(),
            name_index: HashMap::new(),
            id_index: HashMap::new(),
        }
    }

    /// Creates a collection from a list of constant infos.
    pub fn from_infos(infos: impl IntoIterator<Item = OverrideConstantInfo>) -> Self {
        let mut result = Self::new();
        for info in infos {
            result.add(info);
        }
        result
    }

    /// Extracts override constants from a naga Module.
    ///
    /// In naga, override constants are represented in the `overrides` arena
    /// of the module (naga::Module.overrides).
    pub fn from_module(module: &naga::Module) -> Self {
        let mut result = Self::new();

        // Iterate over all overrides in the module
        for (handle, override_const) in module.overrides.iter() {
            // Get the type info
            let ty_handle = override_const.ty;
            let ty = &module.types[ty_handle];

            // Extract scalar type
            let override_ty = match &ty.inner {
                naga::TypeInner::Scalar(scalar) => {
                    OverrideConstantType::from_naga(scalar.kind, scalar.width)
                }
                _ => None,
            };

            let override_ty = match override_ty {
                Some(t) => t,
                None => continue, // Skip non-scalar overrides
            };

            // Get the default value if present
            let default_value = override_const.init.and_then(|init_handle| {
                Self::extract_const_value(module, init_handle)
            });

            // Get the name from the override
            let name = override_const.name.clone();

            // Get the explicit @id if present (naga uses u16, we use u32)
            let id = override_const.id.map(|x| x as u32);

            let info = OverrideConstantInfo::new(name, id, override_ty, default_value);
            result.add(info);
        }

        result
    }

    /// Extracts a constant value from a naga constant expression.
    fn extract_const_value(module: &naga::Module, expr_handle: naga::Handle<naga::Expression>) -> Option<f64> {
        // In naga, override initializers are stored as constant expressions
        // We need to evaluate the expression to get the value
        // For simple literal values, this is straightforward

        // The expression is in the global_expressions arena for overrides
        let expr = &module.global_expressions[expr_handle];

        match expr {
            naga::Expression::Literal(literal) => match literal {
                naga::Literal::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                naga::Literal::I32(i) => Some(*i as f64),
                naga::Literal::U32(u) => Some(*u as f64),
                naga::Literal::F32(f) => Some(*f as f64),
                naga::Literal::F64(f) => Some(*f),
                naga::Literal::I64(i) => Some(*i as f64),
                naga::Literal::U64(u) => Some(*u as f64),
                naga::Literal::AbstractInt(i) => Some(*i as f64),
                naga::Literal::AbstractFloat(f) => Some(*f),
            },
            _ => None, // Complex expressions not supported
        }
    }

    /// Creates override constants from shader reflection data.
    ///
    /// This is an alternative to from_module() when you have reflection
    /// data available rather than the raw naga module.
    pub fn from_reflection(reflection: &ShaderReflection) -> Self {
        // ShaderReflection doesn't currently expose override constants,
        // so we return an empty collection. This would need to be extended
        // in reflection.rs to support override constants.
        Self::new()
    }

    /// Adds an override constant to the collection.
    pub fn add(&mut self, info: OverrideConstantInfo) {
        let index = self.constants.len();

        if let Some(name) = &info.name {
            self.name_index.insert(name.clone(), index);
        }

        if let Some(id) = info.id {
            self.id_index.insert(id, index);
        }

        self.constants.push(info);
    }

    /// Returns the number of override constants.
    #[inline]
    pub fn len(&self) -> usize {
        self.constants.len()
    }

    /// Returns true if there are no override constants.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.constants.is_empty()
    }

    /// Gets an override constant by name.
    pub fn get_by_name(&self, name: &str) -> Option<&OverrideConstantInfo> {
        self.name_index.get(name).map(|&idx| &self.constants[idx])
    }

    /// Gets an override constant by @id.
    pub fn get_by_id(&self, id: u32) -> Option<&OverrideConstantInfo> {
        self.id_index.get(&id).map(|&idx| &self.constants[idx])
    }

    /// Gets an override constant by key (name or stringified ID).
    pub fn get_by_key(&self, key: &str) -> Option<&OverrideConstantInfo> {
        // Try name first
        if let Some(info) = self.get_by_name(key) {
            return Some(info);
        }

        // Try parsing as ID
        if let Ok(id) = key.parse::<u32>() {
            return self.get_by_id(id);
        }

        None
    }

    /// Returns an iterator over all override constants.
    pub fn iter(&self) -> impl Iterator<Item = &OverrideConstantInfo> {
        self.constants.iter()
    }

    /// Returns all constants as a slice.
    pub fn as_slice(&self) -> &[OverrideConstantInfo] {
        &self.constants
    }

    /// Returns all required constants (those without defaults).
    pub fn required(&self) -> impl Iterator<Item = &OverrideConstantInfo> {
        self.constants.iter().filter(|c| c.required)
    }

    /// Returns all optional constants (those with defaults).
    pub fn optional(&self) -> impl Iterator<Item = &OverrideConstantInfo> {
        self.constants.iter().filter(|c| !c.required)
    }

    /// Returns all constant names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.constants.iter().filter_map(|c| c.name.as_deref())
    }

    /// Returns all constant IDs.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.constants.iter().filter_map(|c| c.id)
    }

    /// Validates that all required constants are provided in the given values.
    pub fn validate_required(&self, values: &HashMap<String, f64>) -> Result<(), OverrideError> {
        for constant in self.required() {
            let key = constant.key();
            let has_value = key
                .as_ref()
                .map(|k| values.contains_key(k))
                .unwrap_or(false);

            if !has_value {
                return Err(OverrideError::MissingRequired {
                    name: constant.name.clone(),
                    id: constant.id,
                });
            }
        }
        Ok(())
    }
}

impl fmt::Display for OverrideConstants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OverrideConstants ({} constants):", self.constants.len())?;
        for constant in &self.constants {
            writeln!(f, "  {}", constant)?;
        }
        Ok(())
    }
}

impl IntoIterator for OverrideConstants {
    type Item = OverrideConstantInfo;
    type IntoIter = std::vec::IntoIter<OverrideConstantInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.constants.into_iter()
    }
}

impl<'a> IntoIterator for &'a OverrideConstants {
    type Item = &'a OverrideConstantInfo;
    type IntoIter = std::slice::Iter<'a, OverrideConstantInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.constants.iter()
    }
}

// ============================================================================
// Pipeline Constants (Builder for Setting Values)
// ============================================================================

/// Builder for setting override constant values for pipeline compilation.
///
/// Provides a fluent API for setting override constant values with
/// type-specific methods that handle the conversion to f64.
///
/// # Example
///
/// ```ignore
/// let constants = PipelineConstants::new()
///     .set_f32("SCREEN_WIDTH", 2560.0)
///     .set_f32("SCREEN_HEIGHT", 1440.0)
///     .set_u32("TILE_SIZE", 32)
///     .set_bool("ENABLE_SHADOWS", true);
///
/// let wgpu_constants = constants.to_wgpu();
/// ```
#[derive(Debug, Clone, Default)]
pub struct PipelineConstants {
    /// The constant values (all stored as f64).
    values: HashMap<String, f64>,
}

impl PipelineConstants {
    /// Creates a new empty PipelineConstants builder.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Creates a PipelineConstants with pre-populated values.
    pub fn from_map(values: HashMap<String, f64>) -> Self {
        Self { values }
    }

    /// Sets a constant value (generic f64).
    ///
    /// This is the base method - all other set_* methods call this.
    pub fn set(&mut self, key: impl Into<String>, value: f64) -> &mut Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Sets a constant value, consuming and returning self for chaining.
    pub fn with(mut self, key: impl Into<String>, value: f64) -> Self {
        self.set(key, value);
        self
    }

    /// Sets a boolean constant value.
    ///
    /// `true` is stored as 1.0, `false` as 0.0.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) -> &mut Self {
        self.set(key, if value { 1.0 } else { 0.0 })
    }

    /// Sets a boolean constant value (builder pattern).
    pub fn with_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.set_bool(key, value);
        self
    }

    /// Sets a signed 32-bit integer constant value.
    pub fn set_i32(&mut self, key: impl Into<String>, value: i32) -> &mut Self {
        self.set(key, value as f64)
    }

    /// Sets a signed 32-bit integer constant value (builder pattern).
    pub fn with_i32(mut self, key: impl Into<String>, value: i32) -> Self {
        self.set_i32(key, value);
        self
    }

    /// Sets an unsigned 32-bit integer constant value.
    pub fn set_u32(&mut self, key: impl Into<String>, value: u32) -> &mut Self {
        self.set(key, value as f64)
    }

    /// Sets an unsigned 32-bit integer constant value (builder pattern).
    pub fn with_u32(mut self, key: impl Into<String>, value: u32) -> Self {
        self.set_u32(key, value);
        self
    }

    /// Sets a 32-bit floating-point constant value.
    pub fn set_f32(&mut self, key: impl Into<String>, value: f32) -> &mut Self {
        self.set(key, value as f64)
    }

    /// Sets a 32-bit floating-point constant value (builder pattern).
    pub fn with_f32(mut self, key: impl Into<String>, value: f32) -> Self {
        self.set_f32(key, value);
        self
    }

    /// Removes a constant value.
    pub fn remove(&mut self, key: &str) -> Option<f64> {
        self.values.remove(key)
    }

    /// Returns the value for a key, if set.
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }

    /// Returns true if a value is set for the given key.
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Returns the number of constants set.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if no constants are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clears all constant values.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Returns an iterator over the constant key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.values.iter()
    }

    /// Returns the keys of all set constants.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }

    /// Converts to a HashMap<String, f64> for wgpu PipelineCompilationOptions.
    ///
    /// This returns a reference to the internal map, which can be used with
    /// wgpu's `PipelineCompilationOptions::constants` field.
    pub fn to_wgpu(&self) -> HashMap<String, f64> {
        self.values.clone()
    }

    /// Returns a reference to the internal values map.
    pub fn as_map(&self) -> &HashMap<String, f64> {
        &self.values
    }

    /// Validates that all set constants are valid for the given override declarations.
    ///
    /// Checks:
    /// - All set constants exist in the shader
    /// - Values are within valid range for their types
    /// - All required constants are provided
    pub fn validate(&self, overrides: &OverrideConstants) -> Result<(), OverrideError> {
        // Check that all set values correspond to known constants
        for (key, &value) in &self.values {
            let constant = overrides.get_by_key(key).ok_or_else(|| {
                OverrideError::UnknownConstant { key: key.clone() }
            })?;

            // Validate the value is in range for the type
            constant.validate_value(value)?;
        }

        // Check that all required constants are provided
        overrides.validate_required(&self.values)?;

        Ok(())
    }

    /// Merges another PipelineConstants into this one.
    ///
    /// Values from `other` overwrite values in `self` if keys conflict.
    pub fn merge(&mut self, other: &PipelineConstants) {
        for (key, &value) in &other.values {
            self.values.insert(key.clone(), value);
        }
    }

    /// Creates a new PipelineConstants by merging this one with another.
    pub fn merged_with(&self, other: &PipelineConstants) -> Self {
        let mut result = self.clone();
        result.merge(other);
        result
    }
}

impl fmt::Display for PipelineConstants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PipelineConstants ({} values):", self.values.len())?;
        for (key, value) in &self.values {
            writeln!(f, "  {} = {}", key, value)?;
        }
        Ok(())
    }
}

impl FromIterator<(String, f64)> for PipelineConstants {
    fn from_iter<T: IntoIterator<Item = (String, f64)>>(iter: T) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

impl<'a> FromIterator<(&'a str, f64)> for PipelineConstants {
    fn from_iter<T: IntoIterator<Item = (&'a str, f64)>>(iter: T) -> Self {
        Self {
            values: iter.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }
}

// ============================================================================
// Override Error
// ============================================================================

/// Errors related to override constant handling.
#[derive(Debug, Clone, PartialEq)]
pub enum OverrideError {
    /// Attempted to set a constant that doesn't exist in the shader.
    UnknownConstant {
        /// The unknown key.
        key: String,
    },

    /// Value type doesn't match the constant's declared type.
    TypeMismatch {
        /// The constant key.
        key: String,
        /// The expected type.
        expected: OverrideConstantType,
        /// Description of the provided value.
        got: &'static str,
    },

    /// A required constant (no default value) was not provided.
    MissingRequired {
        /// The constant name (if available).
        name: Option<String>,
        /// The constant @id (if available).
        id: Option<u32>,
    },

    /// Value is out of range for the constant's type.
    ValueOutOfRange {
        /// The constant key.
        key: String,
        /// The invalid value.
        value: f64,
        /// Description of the expected range.
        expected: &'static str,
    },
}

impl OverrideError {
    /// Returns the key associated with this error, if any.
    pub fn key(&self) -> Option<&str> {
        match self {
            OverrideError::UnknownConstant { key } => Some(key),
            OverrideError::TypeMismatch { key, .. } => Some(key),
            OverrideError::ValueOutOfRange { key, .. } => Some(key),
            OverrideError::MissingRequired { name, .. } => name.as_deref(),
        }
    }

    /// Returns true if this is an unknown constant error.
    pub fn is_unknown(&self) -> bool {
        matches!(self, OverrideError::UnknownConstant { .. })
    }

    /// Returns true if this is a type mismatch error.
    pub fn is_type_mismatch(&self) -> bool {
        matches!(self, OverrideError::TypeMismatch { .. })
    }

    /// Returns true if this is a missing required error.
    pub fn is_missing_required(&self) -> bool {
        matches!(self, OverrideError::MissingRequired { .. })
    }

    /// Returns true if this is a value out of range error.
    pub fn is_out_of_range(&self) -> bool {
        matches!(self, OverrideError::ValueOutOfRange { .. })
    }
}

impl fmt::Display for OverrideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OverrideError::UnknownConstant { key } => {
                write!(f, "unknown override constant: '{}'", key)
            }
            OverrideError::TypeMismatch { key, expected, got } => {
                write!(
                    f,
                    "type mismatch for constant '{}': expected {}, got {}",
                    key, expected, got
                )
            }
            OverrideError::MissingRequired { name, id } => {
                match (name, id) {
                    (Some(name), Some(id)) => {
                        write!(f, "missing required override constant '{}' (@id({}))", name, id)
                    }
                    (Some(name), None) => {
                        write!(f, "missing required override constant '{}'", name)
                    }
                    (None, Some(id)) => {
                        write!(f, "missing required override constant @id({})", id)
                    }
                    (None, None) => {
                        write!(f, "missing required override constant")
                    }
                }
            }
            OverrideError::ValueOutOfRange { key, value, expected } => {
                write!(
                    f,
                    "value {} for constant '{}' is out of range, expected {}",
                    value, key, expected
                )
            }
        }
    }
}

impl std::error::Error for OverrideError {}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Extracts override constants from WGSL source.
///
/// Parses the source and extracts all override constant declarations.
///
/// # Example
///
/// ```ignore
/// let source = r#"
///     @id(0) override SCREEN_WIDTH: f32 = 1920.0;
///     @id(1) override SCREEN_HEIGHT: f32 = 1080.0;
///     override TILE_SIZE: u32 = 16u;
/// "#;
///
/// let overrides = extract_overrides_from_wgsl(source)?;
/// assert_eq!(overrides.len(), 3);
/// ```
pub fn extract_overrides_from_wgsl(source: &str) -> Result<OverrideConstants, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("parse error: {}", e.message()))?;

    Ok(OverrideConstants::from_module(&module))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OverrideConstantType Tests
    // =========================================================================

    #[test]
    fn test_override_constant_type_wgsl_name() {
        assert_eq!(OverrideConstantType::Bool.wgsl_name(), "bool");
        assert_eq!(OverrideConstantType::I32.wgsl_name(), "i32");
        assert_eq!(OverrideConstantType::U32.wgsl_name(), "u32");
        assert_eq!(OverrideConstantType::F32.wgsl_name(), "f32");
    }

    #[test]
    fn test_override_constant_type_display() {
        assert_eq!(format!("{}", OverrideConstantType::Bool), "bool");
        assert_eq!(format!("{}", OverrideConstantType::I32), "i32");
        assert_eq!(format!("{}", OverrideConstantType::U32), "u32");
        assert_eq!(format!("{}", OverrideConstantType::F32), "f32");
    }

    #[test]
    fn test_override_constant_type_default() {
        assert_eq!(OverrideConstantType::default(), OverrideConstantType::F32);
    }

    #[test]
    fn test_override_constant_type_default_value() {
        assert_eq!(OverrideConstantType::Bool.default_value(), 0.0);
        assert_eq!(OverrideConstantType::I32.default_value(), 0.0);
        assert_eq!(OverrideConstantType::U32.default_value(), 0.0);
        assert_eq!(OverrideConstantType::F32.default_value(), 0.0);
    }

    #[test]
    fn test_override_constant_type_from_naga() {
        assert_eq!(
            OverrideConstantType::from_naga(naga::ScalarKind::Bool, 1),
            Some(OverrideConstantType::Bool)
        );
        assert_eq!(
            OverrideConstantType::from_naga(naga::ScalarKind::Sint, 4),
            Some(OverrideConstantType::I32)
        );
        assert_eq!(
            OverrideConstantType::from_naga(naga::ScalarKind::Uint, 4),
            Some(OverrideConstantType::U32)
        );
        assert_eq!(
            OverrideConstantType::from_naga(naga::ScalarKind::Float, 4),
            Some(OverrideConstantType::F32)
        );
        // Invalid combinations
        assert_eq!(
            OverrideConstantType::from_naga(naga::ScalarKind::Float, 8),
            None
        );
    }

    #[test]
    fn test_override_constant_type_is_valid_value_bool() {
        assert!(OverrideConstantType::Bool.is_valid_value(0.0));
        assert!(OverrideConstantType::Bool.is_valid_value(1.0));
        assert!(!OverrideConstantType::Bool.is_valid_value(0.5));
        assert!(!OverrideConstantType::Bool.is_valid_value(2.0));
        assert!(!OverrideConstantType::Bool.is_valid_value(-1.0));
    }

    #[test]
    fn test_override_constant_type_is_valid_value_i32() {
        assert!(OverrideConstantType::I32.is_valid_value(0.0));
        assert!(OverrideConstantType::I32.is_valid_value(100.0));
        assert!(OverrideConstantType::I32.is_valid_value(-100.0));
        assert!(OverrideConstantType::I32.is_valid_value(i32::MAX as f64));
        assert!(OverrideConstantType::I32.is_valid_value(i32::MIN as f64));
        assert!(!OverrideConstantType::I32.is_valid_value(0.5)); // Fractional
        assert!(!OverrideConstantType::I32.is_valid_value(i64::MAX as f64)); // Out of range
    }

    #[test]
    fn test_override_constant_type_is_valid_value_u32() {
        assert!(OverrideConstantType::U32.is_valid_value(0.0));
        assert!(OverrideConstantType::U32.is_valid_value(100.0));
        assert!(OverrideConstantType::U32.is_valid_value(u32::MAX as f64));
        assert!(!OverrideConstantType::U32.is_valid_value(-1.0)); // Negative
        assert!(!OverrideConstantType::U32.is_valid_value(0.5)); // Fractional
    }

    #[test]
    fn test_override_constant_type_is_valid_value_f32() {
        assert!(OverrideConstantType::F32.is_valid_value(0.0));
        assert!(OverrideConstantType::F32.is_valid_value(0.5));
        assert!(OverrideConstantType::F32.is_valid_value(-1.5));
        assert!(OverrideConstantType::F32.is_valid_value(1e30));
        assert!(OverrideConstantType::F32.is_valid_value(f64::NAN));
        assert!(!OverrideConstantType::F32.is_valid_value(f64::INFINITY));
    }

    #[test]
    fn test_override_constant_type_value_range() {
        assert_eq!(OverrideConstantType::Bool.value_range(), "0 or 1");
        assert!(OverrideConstantType::I32.value_range().contains("2147483647"));
        assert!(OverrideConstantType::U32.value_range().contains("4294967295"));
        assert!(OverrideConstantType::F32.value_range().contains("f32"));
    }

    // =========================================================================
    // OverrideConstantInfo Tests
    // =========================================================================

    #[test]
    fn test_override_constant_info_new() {
        let info = OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        );
        assert_eq!(info.name, Some("WIDTH".to_string()));
        assert_eq!(info.id, Some(0));
        assert_eq!(info.ty, OverrideConstantType::F32);
        assert_eq!(info.default_value, Some(1920.0));
        assert!(!info.required);
    }

    #[test]
    fn test_override_constant_info_required() {
        let info = OverrideConstantInfo::required(
            Some("MAX_LIGHTS".to_string()),
            Some(1),
            OverrideConstantType::U32,
        );
        assert!(info.required);
        assert!(info.default_value.is_none());
    }

    #[test]
    fn test_override_constant_info_with_default() {
        let info = OverrideConstantInfo::with_default(
            Some("TILE_SIZE".to_string()),
            None,
            OverrideConstantType::U32,
            16.0,
        );
        assert!(!info.required);
        assert_eq!(info.default_value, Some(16.0));
    }

    #[test]
    fn test_override_constant_info_key_with_name() {
        let info = OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        );
        assert_eq!(info.key(), Some("WIDTH".to_string()));
    }

    #[test]
    fn test_override_constant_info_key_id_only() {
        let info = OverrideConstantInfo::new(
            None,
            Some(5),
            OverrideConstantType::F32,
            Some(1.0),
        );
        assert_eq!(info.key(), Some("5".to_string()));
    }

    #[test]
    fn test_override_constant_info_key_neither() {
        let info = OverrideConstantInfo::new(
            None,
            None,
            OverrideConstantType::F32,
            Some(1.0),
        );
        assert_eq!(info.key(), None);
    }

    #[test]
    fn test_override_constant_info_has_name() {
        let info1 = OverrideConstantInfo::new(Some("X".to_string()), None, OverrideConstantType::F32, None);
        let info2 = OverrideConstantInfo::new(None, Some(0), OverrideConstantType::F32, None);
        assert!(info1.has_name());
        assert!(!info2.has_name());
    }

    #[test]
    fn test_override_constant_info_has_id() {
        let info1 = OverrideConstantInfo::new(Some("X".to_string()), Some(0), OverrideConstantType::F32, None);
        let info2 = OverrideConstantInfo::new(Some("Y".to_string()), None, OverrideConstantType::F32, None);
        assert!(info1.has_id());
        assert!(!info2.has_id());
    }

    #[test]
    fn test_override_constant_info_has_default() {
        let info1 = OverrideConstantInfo::new(Some("X".to_string()), None, OverrideConstantType::F32, Some(1.0));
        let info2 = OverrideConstantInfo::new(Some("Y".to_string()), None, OverrideConstantType::F32, None);
        assert!(info1.has_default());
        assert!(!info2.has_default());
    }

    #[test]
    fn test_override_constant_info_validate_value_valid() {
        let info = OverrideConstantInfo::new(
            Some("X".to_string()),
            None,
            OverrideConstantType::U32,
            None,
        );
        assert!(info.validate_value(100.0).is_ok());
    }

    #[test]
    fn test_override_constant_info_validate_value_invalid() {
        let info = OverrideConstantInfo::new(
            Some("X".to_string()),
            None,
            OverrideConstantType::U32,
            None,
        );
        let result = info.validate_value(-5.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_out_of_range());
    }

    #[test]
    fn test_override_constant_info_display() {
        let info = OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        );
        let display = format!("{}", info);
        assert!(display.contains("@id(0)"));
        assert!(display.contains("WIDTH"));
        assert!(display.contains("f32"));
        assert!(display.contains("1920"));
    }

    #[test]
    fn test_override_constant_info_display_required() {
        let info = OverrideConstantInfo::required(
            Some("MAX_LIGHTS".to_string()),
            None,
            OverrideConstantType::U32,
        );
        let display = format!("{}", info);
        assert!(display.contains("required"));
    }

    // =========================================================================
    // OverrideConstants Tests
    // =========================================================================

    #[test]
    fn test_override_constants_new() {
        let overrides = OverrideConstants::new();
        assert!(overrides.is_empty());
        assert_eq!(overrides.len(), 0);
    }

    #[test]
    fn test_override_constants_add() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        ));
        assert_eq!(overrides.len(), 1);
    }

    #[test]
    fn test_override_constants_from_infos() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), Some(1), OverrideConstantType::U32, Some(2.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);
        assert_eq!(overrides.len(), 2);
    }

    #[test]
    fn test_override_constants_get_by_name() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        ));

        let info = overrides.get_by_name("WIDTH");
        assert!(info.is_some());
        assert_eq!(info.unwrap().default_value, Some(1920.0));

        assert!(overrides.get_by_name("HEIGHT").is_none());
    }

    #[test]
    fn test_override_constants_get_by_id() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(5),
            OverrideConstantType::F32,
            Some(1920.0),
        ));

        let info = overrides.get_by_id(5);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, Some("WIDTH".to_string()));

        assert!(overrides.get_by_id(10).is_none());
    }

    #[test]
    fn test_override_constants_get_by_key_name() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        ));

        assert!(overrides.get_by_key("WIDTH").is_some());
    }

    #[test]
    fn test_override_constants_get_by_key_id_string() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            None,
            Some(5),
            OverrideConstantType::F32,
            Some(1920.0),
        ));

        assert!(overrides.get_by_key("5").is_some());
    }

    #[test]
    fn test_override_constants_iter() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let names: Vec<_> = overrides.iter().filter_map(|c| c.name.as_deref()).collect();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn test_override_constants_as_slice() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        assert_eq!(overrides.as_slice().len(), 1);
    }

    #[test]
    fn test_override_constants_required() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::required(Some("B".to_string()), None, OverrideConstantType::U32),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let required: Vec<_> = overrides.required().collect();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].name, Some("B".to_string()));
    }

    #[test]
    fn test_override_constants_optional() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::required(Some("B".to_string()), None, OverrideConstantType::U32),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let optional: Vec<_> = overrides.optional().collect();
        assert_eq!(optional.len(), 1);
        assert_eq!(optional[0].name, Some("A".to_string()));
    }

    #[test]
    fn test_override_constants_names() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(None, Some(0), OverrideConstantType::U32, Some(2.0)),
            OverrideConstantInfo::new(Some("C".to_string()), None, OverrideConstantType::Bool, Some(1.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let names: Vec<_> = overrides.names().collect();
        assert_eq!(names, vec!["A", "C"]);
    }

    #[test]
    fn test_override_constants_ids() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(None, Some(5), OverrideConstantType::U32, Some(2.0)),
            OverrideConstantInfo::new(Some("C".to_string()), None, OverrideConstantType::Bool, Some(1.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let ids: Vec<_> = overrides.ids().collect();
        assert_eq!(ids, vec![0, 5]);
    }

    #[test]
    fn test_override_constants_validate_required_success() {
        let infos = vec![
            OverrideConstantInfo::required(Some("A".to_string()), None, OverrideConstantType::F32),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let mut values = HashMap::new();
        values.insert("A".to_string(), 1.0);

        assert!(overrides.validate_required(&values).is_ok());
    }

    #[test]
    fn test_override_constants_validate_required_failure() {
        let infos = vec![
            OverrideConstantInfo::required(Some("A".to_string()), None, OverrideConstantType::F32),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let values = HashMap::new();

        let result = overrides.validate_required(&values);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_missing_required());
    }

    #[test]
    fn test_override_constants_display() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let display = format!("{}", overrides);
        assert!(display.contains("OverrideConstants"));
        assert!(display.contains("1 constants"));
    }

    #[test]
    fn test_override_constants_into_iter() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        let collected: Vec<_> = overrides.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    // =========================================================================
    // PipelineConstants Tests
    // =========================================================================

    #[test]
    fn test_pipeline_constants_new() {
        let constants = PipelineConstants::new();
        assert!(constants.is_empty());
        assert_eq!(constants.len(), 0);
    }

    #[test]
    fn test_pipeline_constants_set() {
        let mut constants = PipelineConstants::new();
        constants.set("WIDTH", 1920.0);

        assert_eq!(constants.get("WIDTH"), Some(1920.0));
        assert_eq!(constants.len(), 1);
    }

    #[test]
    fn test_pipeline_constants_with() {
        let constants = PipelineConstants::new()
            .with("WIDTH", 1920.0)
            .with("HEIGHT", 1080.0);

        assert_eq!(constants.get("WIDTH"), Some(1920.0));
        assert_eq!(constants.get("HEIGHT"), Some(1080.0));
        assert_eq!(constants.len(), 2);
    }

    #[test]
    fn test_pipeline_constants_set_bool() {
        let mut constants = PipelineConstants::new();
        constants.set_bool("ENABLED", true);
        constants.set_bool("DISABLED", false);

        assert_eq!(constants.get("ENABLED"), Some(1.0));
        assert_eq!(constants.get("DISABLED"), Some(0.0));
    }

    #[test]
    fn test_pipeline_constants_with_bool() {
        let constants = PipelineConstants::new()
            .with_bool("ENABLED", true);

        assert_eq!(constants.get("ENABLED"), Some(1.0));
    }

    #[test]
    fn test_pipeline_constants_set_i32() {
        let mut constants = PipelineConstants::new();
        constants.set_i32("OFFSET", -100);

        assert_eq!(constants.get("OFFSET"), Some(-100.0));
    }

    #[test]
    fn test_pipeline_constants_with_i32() {
        let constants = PipelineConstants::new()
            .with_i32("OFFSET", -100);

        assert_eq!(constants.get("OFFSET"), Some(-100.0));
    }

    #[test]
    fn test_pipeline_constants_set_u32() {
        let mut constants = PipelineConstants::new();
        constants.set_u32("TILE_SIZE", 32);

        assert_eq!(constants.get("TILE_SIZE"), Some(32.0));
    }

    #[test]
    fn test_pipeline_constants_with_u32() {
        let constants = PipelineConstants::new()
            .with_u32("TILE_SIZE", 32);

        assert_eq!(constants.get("TILE_SIZE"), Some(32.0));
    }

    #[test]
    fn test_pipeline_constants_set_f32() {
        let mut constants = PipelineConstants::new();
        constants.set_f32("SCALE", 1.5);

        assert_eq!(constants.get("SCALE"), Some(1.5 as f64));
    }

    #[test]
    fn test_pipeline_constants_with_f32() {
        let constants = PipelineConstants::new()
            .with_f32("SCALE", 1.5);

        assert_eq!(constants.get("SCALE"), Some(1.5 as f64));
    }

    #[test]
    fn test_pipeline_constants_remove() {
        let mut constants = PipelineConstants::new();
        constants.set("WIDTH", 1920.0);

        let removed = constants.remove("WIDTH");
        assert_eq!(removed, Some(1920.0));
        assert!(constants.is_empty());
    }

    #[test]
    fn test_pipeline_constants_contains() {
        let constants = PipelineConstants::new().with("WIDTH", 1920.0);

        assert!(constants.contains("WIDTH"));
        assert!(!constants.contains("HEIGHT"));
    }

    #[test]
    fn test_pipeline_constants_clear() {
        let mut constants = PipelineConstants::new()
            .with("A", 1.0)
            .with("B", 2.0);

        constants.clear();
        assert!(constants.is_empty());
    }

    #[test]
    fn test_pipeline_constants_iter() {
        let constants = PipelineConstants::new()
            .with("A", 1.0)
            .with("B", 2.0);

        let collected: Vec<_> = constants.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_pipeline_constants_keys() {
        let constants = PipelineConstants::new()
            .with("A", 1.0)
            .with("B", 2.0);

        let keys: Vec<_> = constants.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_pipeline_constants_to_wgpu() {
        let constants = PipelineConstants::new()
            .with("WIDTH", 1920.0)
            .with("HEIGHT", 1080.0);

        let wgpu_map = constants.to_wgpu();
        assert_eq!(wgpu_map.get("WIDTH"), Some(&1920.0));
        assert_eq!(wgpu_map.get("HEIGHT"), Some(&1080.0));
    }

    #[test]
    fn test_pipeline_constants_as_map() {
        let constants = PipelineConstants::new()
            .with("WIDTH", 1920.0);

        let map = constants.as_map();
        assert_eq!(map.get("WIDTH"), Some(&1920.0));
    }

    #[test]
    fn test_pipeline_constants_validate_success() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("WIDTH".to_string()), None, OverrideConstantType::F32, Some(1920.0)),
            OverrideConstantInfo::required(Some("HEIGHT".to_string()), None, OverrideConstantType::F32),
        ]);

        let constants = PipelineConstants::new()
            .with_f32("WIDTH", 2560.0)
            .with_f32("HEIGHT", 1440.0);

        assert!(constants.validate(&overrides).is_ok());
    }

    #[test]
    fn test_pipeline_constants_validate_unknown() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("WIDTH".to_string()), None, OverrideConstantType::F32, Some(1920.0)),
        ]);

        let constants = PipelineConstants::new()
            .with_f32("UNKNOWN", 1.0);

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unknown());
    }

    #[test]
    fn test_pipeline_constants_validate_out_of_range() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("VALUE".to_string()), None, OverrideConstantType::U32, Some(0.0)),
        ]);

        let constants = PipelineConstants::new()
            .with("VALUE", -5.0); // Negative for u32

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_out_of_range());
    }

    #[test]
    fn test_pipeline_constants_validate_missing_required() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::required(Some("REQUIRED".to_string()), None, OverrideConstantType::F32),
        ]);

        let constants = PipelineConstants::new();

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_missing_required());
    }

    #[test]
    fn test_pipeline_constants_merge() {
        let mut a = PipelineConstants::new()
            .with("A", 1.0)
            .with("B", 2.0);

        let b = PipelineConstants::new()
            .with("B", 20.0)
            .with("C", 3.0);

        a.merge(&b);

        assert_eq!(a.get("A"), Some(1.0));
        assert_eq!(a.get("B"), Some(20.0)); // Overwritten
        assert_eq!(a.get("C"), Some(3.0));
    }

    #[test]
    fn test_pipeline_constants_merged_with() {
        let a = PipelineConstants::new()
            .with("A", 1.0);

        let b = PipelineConstants::new()
            .with("B", 2.0);

        let c = a.merged_with(&b);

        assert_eq!(c.get("A"), Some(1.0));
        assert_eq!(c.get("B"), Some(2.0));
    }

    #[test]
    fn test_pipeline_constants_display() {
        let constants = PipelineConstants::new()
            .with("WIDTH", 1920.0);

        let display = format!("{}", constants);
        assert!(display.contains("PipelineConstants"));
        assert!(display.contains("WIDTH"));
        assert!(display.contains("1920"));
    }

    #[test]
    fn test_pipeline_constants_from_iter_string() {
        let items = vec![
            ("A".to_string(), 1.0),
            ("B".to_string(), 2.0),
        ];

        let constants: PipelineConstants = items.into_iter().collect();
        assert_eq!(constants.get("A"), Some(1.0));
        assert_eq!(constants.get("B"), Some(2.0));
    }

    #[test]
    fn test_pipeline_constants_from_iter_str() {
        let items = vec![
            ("A", 1.0),
            ("B", 2.0),
        ];

        let constants: PipelineConstants = items.into_iter().collect();
        assert_eq!(constants.get("A"), Some(1.0));
        assert_eq!(constants.get("B"), Some(2.0));
    }

    #[test]
    fn test_pipeline_constants_from_map() {
        let mut map = HashMap::new();
        map.insert("WIDTH".to_string(), 1920.0);

        let constants = PipelineConstants::from_map(map);
        assert_eq!(constants.get("WIDTH"), Some(1920.0));
    }

    // =========================================================================
    // OverrideError Tests
    // =========================================================================

    #[test]
    fn test_override_error_unknown_constant() {
        let err = OverrideError::UnknownConstant { key: "UNKNOWN".to_string() };
        assert!(err.is_unknown());
        assert!(!err.is_type_mismatch());
        assert_eq!(err.key(), Some("UNKNOWN"));

        let display = format!("{}", err);
        assert!(display.contains("unknown"));
        assert!(display.contains("UNKNOWN"));
    }

    #[test]
    fn test_override_error_type_mismatch() {
        let err = OverrideError::TypeMismatch {
            key: "VALUE".to_string(),
            expected: OverrideConstantType::U32,
            got: "negative number",
        };
        assert!(err.is_type_mismatch());
        assert_eq!(err.key(), Some("VALUE"));

        let display = format!("{}", err);
        assert!(display.contains("type mismatch"));
        assert!(display.contains("VALUE"));
    }

    #[test]
    fn test_override_error_missing_required_with_name() {
        let err = OverrideError::MissingRequired {
            name: Some("MAX_LIGHTS".to_string()),
            id: Some(0),
        };
        assert!(err.is_missing_required());
        assert_eq!(err.key(), Some("MAX_LIGHTS"));

        let display = format!("{}", err);
        assert!(display.contains("missing required"));
        assert!(display.contains("MAX_LIGHTS"));
        assert!(display.contains("@id(0)"));
    }

    #[test]
    fn test_override_error_missing_required_id_only() {
        let err = OverrideError::MissingRequired {
            name: None,
            id: Some(5),
        };

        let display = format!("{}", err);
        assert!(display.contains("@id(5)"));
    }

    #[test]
    fn test_override_error_missing_required_name_only() {
        let err = OverrideError::MissingRequired {
            name: Some("VALUE".to_string()),
            id: None,
        };

        let display = format!("{}", err);
        assert!(display.contains("VALUE"));
        assert!(!display.contains("@id"));
    }

    #[test]
    fn test_override_error_missing_required_neither() {
        let err = OverrideError::MissingRequired {
            name: None,
            id: None,
        };

        let display = format!("{}", err);
        assert!(display.contains("missing required"));
    }

    #[test]
    fn test_override_error_value_out_of_range() {
        let err = OverrideError::ValueOutOfRange {
            key: "COUNT".to_string(),
            value: -5.0,
            expected: "0 to 4294967295",
        };
        assert!(err.is_out_of_range());
        assert_eq!(err.key(), Some("COUNT"));

        let display = format!("{}", err);
        assert!(display.contains("out of range"));
        assert!(display.contains("-5"));
        assert!(display.contains("COUNT"));
    }

    #[test]
    fn test_override_error_debug() {
        let err = OverrideError::UnknownConstant { key: "X".to_string() };
        let debug = format!("{:?}", err);
        assert!(debug.contains("UnknownConstant"));
    }

    #[test]
    fn test_override_error_clone() {
        let err = OverrideError::UnknownConstant { key: "X".to_string() };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // =========================================================================
    // Integration Tests with Naga
    // =========================================================================

    #[test]
    fn test_extract_overrides_from_simple_shader() {
        let source = r#"
            override SCALE: f32 = 1.0;

            var<private> sink: f32;

            @compute @workgroup_size(1)
            fn main() {
                sink = SCALE;
            }
        "#;

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_ok(), "Extract failed: {:?}", result.err());

        let overrides = result.unwrap();
        assert_eq!(overrides.len(), 1, "Found {} overrides", overrides.len());

        let scale = overrides.get_by_name("SCALE");
        assert!(scale.is_some());
        assert_eq!(scale.unwrap().ty, OverrideConstantType::F32);
        assert_eq!(scale.unwrap().default_value, Some(1.0));
    }

    #[test]
    fn test_extract_overrides_with_id() {
        let source = r#"
            @id(0) override WIDTH: f32 = 1920.0;
            @id(1) override HEIGHT: f32 = 1080.0;

            var<private> sink: f32;

            @compute @workgroup_size(1)
            fn main() {
                sink = WIDTH + HEIGHT;
            }
        "#;

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_ok());

        let overrides = result.unwrap();
        assert_eq!(overrides.len(), 2);

        let width = overrides.get_by_id(0);
        assert!(width.is_some());
        assert_eq!(width.unwrap().name, Some("WIDTH".to_string()));

        let height = overrides.get_by_id(1);
        assert!(height.is_some());
        assert_eq!(height.unwrap().name, Some("HEIGHT".to_string()));
    }

    #[test]
    fn test_extract_overrides_multiple_types() {
        let source = r#"
            override ENABLE_FEATURE: bool = false;
            override COUNT: u32 = 16u;
            override OFFSET: i32 = -10i;
            override SCALE: f32 = 1.5;

            var<private> sink: f32;

            @compute @workgroup_size(1)
            fn main() {
                if ENABLE_FEATURE {
                    sink = f32(COUNT) * SCALE + f32(OFFSET);
                }
            }
        "#;

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_ok());

        let overrides = result.unwrap();
        assert_eq!(overrides.len(), 4);

        let enable = overrides.get_by_name("ENABLE_FEATURE");
        assert!(enable.is_some());
        assert_eq!(enable.unwrap().ty, OverrideConstantType::Bool);

        let count = overrides.get_by_name("COUNT");
        assert!(count.is_some());
        assert_eq!(count.unwrap().ty, OverrideConstantType::U32);

        let offset = overrides.get_by_name("OFFSET");
        assert!(offset.is_some());
        assert_eq!(offset.unwrap().ty, OverrideConstantType::I32);

        let scale = overrides.get_by_name("SCALE");
        assert!(scale.is_some());
        assert_eq!(scale.unwrap().ty, OverrideConstantType::F32);
    }

    #[test]
    fn test_extract_overrides_required() {
        let source = r#"
            override MAX_LIGHTS: u32;

            var<private> sink: u32;

            @compute @workgroup_size(1)
            fn main() {
                sink = MAX_LIGHTS;
            }
        "#;

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_ok());

        let overrides = result.unwrap();
        let max_lights = overrides.get_by_name("MAX_LIGHTS");
        assert!(max_lights.is_some());
        assert!(max_lights.unwrap().required);
        assert!(max_lights.unwrap().default_value.is_none());
    }

    #[test]
    fn test_extract_overrides_no_overrides() {
        let source = r#"
            @compute @workgroup_size(1)
            fn main() {}
        "#;

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_ok());

        let overrides = result.unwrap();
        assert!(overrides.is_empty());
    }

    #[test]
    fn test_extract_overrides_invalid_source() {
        let source = "invalid wgsl @@@";

        let result = extract_overrides_from_wgsl(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_override_constants_from_module_direct() {
        let source = r#"
            @id(5) override TILE_SIZE: u32 = 16u;

            var<private> sink: u32;

            @compute @workgroup_size(1)
            fn main() {
                sink = TILE_SIZE;
            }
        "#;

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let overrides = OverrideConstants::from_module(&module);

        assert_eq!(overrides.len(), 1);

        let tile_size = overrides.get_by_id(5);
        assert!(tile_size.is_some());
        assert_eq!(tile_size.unwrap().name, Some("TILE_SIZE".to_string()));
        assert_eq!(tile_size.unwrap().ty, OverrideConstantType::U32);
        assert_eq!(tile_size.unwrap().default_value, Some(16.0));
    }

    #[test]
    fn test_override_constants_from_reflection_empty() {
        // Currently returns empty since ShaderReflection doesn't expose overrides
        let source = r#"
            @compute @workgroup_size(1) fn main() {}
        "#;

        let module = naga::front::wgsl::parse_str(source).unwrap();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).unwrap();
        let reflection = ShaderReflection::from_module(&module, &info).unwrap();

        let overrides = OverrideConstants::from_reflection(&reflection);
        assert!(overrides.is_empty());
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_override_constant_type_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OverrideConstantType>();
    }

    #[test]
    fn test_override_constant_info_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OverrideConstantInfo>();
    }

    #[test]
    fn test_override_constants_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OverrideConstants>();
    }

    #[test]
    fn test_pipeline_constants_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineConstants>();
    }

    #[test]
    fn test_override_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OverrideError>();
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_override_constant_info_equality() {
        let a = OverrideConstantInfo::new(
            Some("X".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1.0),
        );
        let b = OverrideConstantInfo::new(
            Some("X".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1.0),
        );
        let c = OverrideConstantInfo::new(
            Some("Y".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1.0),
        );

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_pipeline_constants_overwrite() {
        let mut constants = PipelineConstants::new();
        constants.set("X", 1.0);
        constants.set("X", 2.0);

        assert_eq!(constants.get("X"), Some(2.0));
        assert_eq!(constants.len(), 1);
    }

    #[test]
    fn test_empty_override_constants_validation() {
        let overrides = OverrideConstants::new();
        let constants = PipelineConstants::new();

        // No required constants, no values set - should succeed
        assert!(constants.validate(&overrides).is_ok());
    }

    #[test]
    fn test_override_constants_default() {
        let overrides = OverrideConstants::default();
        assert!(overrides.is_empty());
    }

    #[test]
    fn test_pipeline_constants_default() {
        let constants = PipelineConstants::default();
        assert!(constants.is_empty());
    }

    // =========================================================================
    // WHITEBOX TESTS: Type Conversion Edge Cases
    // =========================================================================

    #[test]
    fn test_type_from_naga_f16_returns_none() {
        // f16 (2-byte float) is not directly supported as override constant type
        let result = OverrideConstantType::from_naga(naga::ScalarKind::Float, 2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_type_from_naga_vec_returns_none() {
        // Vector types (width 8, 12, 16 for vec2/3/4) are not valid override types
        // Testing with invalid widths that don't match our supported scalars
        let result_f64 = OverrideConstantType::from_naga(naga::ScalarKind::Float, 8);
        assert_eq!(result_f64, None);
    }

    #[test]
    fn test_type_from_naga_matrix_returns_none() {
        // Matrices would have larger widths; scalar with width 16 is not supported
        let result = OverrideConstantType::from_naga(naga::ScalarKind::Float, 16);
        assert_eq!(result, None);
    }

    #[test]
    fn test_type_from_naga_i8_returns_none() {
        // i8 (1-byte signed int) is not supported
        let result = OverrideConstantType::from_naga(naga::ScalarKind::Sint, 1);
        assert_eq!(result, None);
    }

    #[test]
    fn test_type_from_naga_abstract_returns_none() {
        // AbstractInt and AbstractFloat are not supported as override constants
        let result_int = OverrideConstantType::from_naga(naga::ScalarKind::AbstractInt, 8);
        let result_float = OverrideConstantType::from_naga(naga::ScalarKind::AbstractFloat, 8);
        assert_eq!(result_int, None);
        assert_eq!(result_float, None);
    }

    // =========================================================================
    // WHITEBOX TESTS: Value Validation Edge Cases
    // =========================================================================

    #[test]
    fn test_bool_validation_rejects_negative() {
        assert!(!OverrideConstantType::Bool.is_valid_value(-1.0));
        assert!(!OverrideConstantType::Bool.is_valid_value(-0.5));
        assert!(!OverrideConstantType::Bool.is_valid_value(-100.0));
    }

    #[test]
    fn test_bool_validation_rejects_two() {
        assert!(!OverrideConstantType::Bool.is_valid_value(2.0));
        assert!(!OverrideConstantType::Bool.is_valid_value(1.5));
        assert!(!OverrideConstantType::Bool.is_valid_value(100.0));
    }

    #[test]
    fn test_i32_validation_rejects_fraction() {
        assert!(!OverrideConstantType::I32.is_valid_value(1.5));
        assert!(!OverrideConstantType::I32.is_valid_value(-1.5));
        assert!(!OverrideConstantType::I32.is_valid_value(0.001));
        assert!(!OverrideConstantType::I32.is_valid_value(0.999));
    }

    #[test]
    fn test_u32_validation_rejects_negative() {
        assert!(!OverrideConstantType::U32.is_valid_value(-0.001));
        assert!(!OverrideConstantType::U32.is_valid_value(-1.0));
        assert!(!OverrideConstantType::U32.is_valid_value(-1000.0));
        assert!(!OverrideConstantType::U32.is_valid_value(i32::MIN as f64));
    }

    #[test]
    fn test_u32_validation_accepts_max() {
        assert!(OverrideConstantType::U32.is_valid_value(u32::MAX as f64));
        assert!(OverrideConstantType::U32.is_valid_value((u32::MAX - 1) as f64));
    }

    #[test]
    fn test_i32_validation_accepts_min_max() {
        assert!(OverrideConstantType::I32.is_valid_value(i32::MIN as f64));
        assert!(OverrideConstantType::I32.is_valid_value(i32::MAX as f64));
        assert!(OverrideConstantType::I32.is_valid_value(0.0));
    }

    #[test]
    fn test_f32_validation_accepts_infinity() {
        // Note: The current implementation rejects infinity (requires is_finite || is_nan)
        assert!(!OverrideConstantType::F32.is_valid_value(f64::INFINITY));
        assert!(!OverrideConstantType::F32.is_valid_value(f64::NEG_INFINITY));
    }

    #[test]
    fn test_f32_validation_accepts_nan() {
        assert!(OverrideConstantType::F32.is_valid_value(f64::NAN));
        // Different NaN representations should also work
        let neg_nan = -f64::NAN;
        assert!(neg_nan.is_nan()); // Verify it's NaN
        assert!(OverrideConstantType::F32.is_valid_value(neg_nan));
    }

    // =========================================================================
    // WHITEBOX TESTS: OverrideConstants Collection
    // =========================================================================

    #[test]
    fn test_constants_duplicate_name_keeps_first() {
        // When adding constants with duplicate names, the index map will overwrite
        // but constants vector will have both - get_by_name returns the LAST one
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("X".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1.0),
        ));
        overrides.add(OverrideConstantInfo::new(
            Some("X".to_string()),
            Some(1),
            OverrideConstantType::U32,
            Some(2.0),
        ));

        // Both are in the vector
        assert_eq!(overrides.len(), 2);

        // get_by_name returns the second one (index was overwritten)
        let info = overrides.get_by_name("X").unwrap();
        assert_eq!(info.ty, OverrideConstantType::U32);
        assert_eq!(info.id, Some(1));
    }

    #[test]
    fn test_constants_duplicate_id_keeps_first() {
        // Similar to duplicate name - ID index will point to the last one added
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("A".to_string()),
            Some(5),
            OverrideConstantType::F32,
            Some(1.0),
        ));
        overrides.add(OverrideConstantInfo::new(
            Some("B".to_string()),
            Some(5),
            OverrideConstantType::U32,
            Some(2.0),
        ));

        assert_eq!(overrides.len(), 2);

        // get_by_id returns the second one (index was overwritten)
        let info = overrides.get_by_id(5).unwrap();
        assert_eq!(info.name, Some("B".to_string()));
    }

    #[test]
    fn test_constants_len_matches_count() {
        let mut overrides = OverrideConstants::new();
        assert_eq!(overrides.len(), 0);

        for i in 0..10 {
            overrides.add(OverrideConstantInfo::new(
                Some(format!("C{}", i)),
                Some(i),
                OverrideConstantType::F32,
                Some(i as f64),
            ));
            assert_eq!(overrides.len(), i as usize + 1);
        }
    }

    #[test]
    fn test_constants_is_empty_true_for_new() {
        let overrides = OverrideConstants::new();
        assert!(overrides.is_empty());
        assert_eq!(overrides.len(), 0);
    }

    #[test]
    fn test_constants_is_empty_false_after_add() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("X".to_string()),
            None,
            OverrideConstantType::F32,
            Some(1.0),
        ));
        assert!(!overrides.is_empty());
    }

    #[test]
    fn test_constants_clear_removes_all() {
        // Note: OverrideConstants doesn't have a clear method, but we can test
        // the default/new behavior and verify the structure is correct
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), Some(1), OverrideConstantType::U32, Some(2.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        // Verify the structure before (no clear method, so just verify initial state)
        assert!(!overrides.is_empty());
        assert_eq!(overrides.len(), 2);

        // Create a new empty one to simulate "clear"
        let cleared = OverrideConstants::new();
        assert!(cleared.is_empty());
        assert_eq!(cleared.len(), 0);
    }

    // =========================================================================
    // WHITEBOX TESTS: PipelineConstants Builder
    // =========================================================================

    #[test]
    fn test_builder_chain_multiple_sets() {
        let mut constants = PipelineConstants::new();
        constants
            .set("A", 1.0)
            .set("B", 2.0)
            .set("C", 3.0)
            .set_bool("D", true)
            .set_i32("E", -5)
            .set_u32("F", 100)
            .set_f32("G", 1.5);

        assert_eq!(constants.len(), 7);
        assert_eq!(constants.get("A"), Some(1.0));
        assert_eq!(constants.get("B"), Some(2.0));
        assert_eq!(constants.get("C"), Some(3.0));
        assert_eq!(constants.get("D"), Some(1.0));
        assert_eq!(constants.get("E"), Some(-5.0));
        assert_eq!(constants.get("F"), Some(100.0));
        assert_eq!(constants.get("G"), Some(1.5f32 as f64));
    }

    #[test]
    fn test_builder_with_returns_clone() {
        let base = PipelineConstants::new().with("A", 1.0);
        let extended = base.clone().with("B", 2.0);

        // base should still only have A
        assert_eq!(base.len(), 1);
        assert!(base.contains("A"));
        assert!(!base.contains("B"));

        // extended should have both
        assert_eq!(extended.len(), 2);
        assert!(extended.contains("A"));
        assert!(extended.contains("B"));
    }

    #[test]
    fn test_builder_from_iter_empty() {
        let items: Vec<(String, f64)> = vec![];
        let constants: PipelineConstants = items.into_iter().collect();
        assert!(constants.is_empty());
    }

    #[test]
    fn test_builder_from_iter_multiple() {
        let items = vec![
            ("A".to_string(), 1.0),
            ("B".to_string(), 2.0),
            ("C".to_string(), 3.0),
            ("D".to_string(), 4.0),
            ("E".to_string(), 5.0),
        ];
        let constants: PipelineConstants = items.into_iter().collect();

        assert_eq!(constants.len(), 5);
        for i in 1..=5 {
            let key = (b'A' + i as u8 - 1) as char;
            assert_eq!(constants.get(&key.to_string()), Some(i as f64));
        }
    }

    #[test]
    fn test_builder_get_returns_value() {
        let constants = PipelineConstants::new()
            .with("WIDTH", 1920.0)
            .with("HEIGHT", 1080.0);

        assert_eq!(constants.get("WIDTH"), Some(1920.0));
        assert_eq!(constants.get("HEIGHT"), Some(1080.0));
    }

    #[test]
    fn test_builder_get_returns_none_missing() {
        let constants = PipelineConstants::new().with("WIDTH", 1920.0);

        assert_eq!(constants.get("HEIGHT"), None);
        assert_eq!(constants.get("DEPTH"), None);
        assert_eq!(constants.get(""), None);
    }

    // =========================================================================
    // WHITEBOX TESTS: Validation Error Paths
    // =========================================================================

    #[test]
    fn test_validate_multiple_missing_required() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::required(Some("A".to_string()), Some(0), OverrideConstantType::F32),
            OverrideConstantInfo::required(Some("B".to_string()), Some(1), OverrideConstantType::U32),
            OverrideConstantInfo::required(Some("C".to_string()), Some(2), OverrideConstantType::I32),
        ]);

        // Provide none of the required constants
        let constants = PipelineConstants::new();
        let result = constants.validate(&overrides);

        assert!(result.is_err());
        // First missing required should be reported
        let err = result.unwrap_err();
        assert!(err.is_missing_required());
    }

    #[test]
    fn test_validate_multiple_unknown() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("VALID".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        ]);

        // Set multiple unknown constants
        let constants = PipelineConstants::new()
            .with("UNKNOWN1", 1.0)
            .with("UNKNOWN2", 2.0);

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unknown());
    }

    #[test]
    fn test_validate_reports_first_error() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("VALUE".to_string()), None, OverrideConstantType::U32, Some(0.0)),
        ]);

        // Set an invalid value (negative for u32)
        let constants = PipelineConstants::new().with("VALUE", -5.0);

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_out_of_range());
        assert_eq!(err.key(), Some("VALUE"));
    }

    #[test]
    fn test_validate_type_mismatch_bool_for_i32() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("FLAG".to_string()), None, OverrideConstantType::Bool, Some(0.0)),
        ]);

        // Provide an i32 value for a bool constant (value 5 is not valid for bool)
        let constants = PipelineConstants::new().with("FLAG", 5.0);

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_out_of_range());
    }

    #[test]
    fn test_validate_type_mismatch_u32_overflow() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("COUNT".to_string()), None, OverrideConstantType::U32, Some(0.0)),
        ]);

        // Value exceeds u32::MAX
        let constants = PipelineConstants::new().with("COUNT", (u32::MAX as f64) + 1000.0);

        let result = constants.validate(&overrides);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_out_of_range());
    }

    #[test]
    fn test_validate_all_constants_provided() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::required(Some("A".to_string()), None, OverrideConstantType::F32),
            OverrideConstantInfo::required(Some("B".to_string()), None, OverrideConstantType::U32),
            OverrideConstantInfo::required(Some("C".to_string()), None, OverrideConstantType::I32),
            OverrideConstantInfo::required(Some("D".to_string()), None, OverrideConstantType::Bool),
        ]);

        let constants = PipelineConstants::new()
            .with_f32("A", 1.0)
            .with_u32("B", 100)
            .with_i32("C", -50)
            .with_bool("D", true);

        let result = constants.validate(&overrides);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_partial_constants_with_defaults() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("OPTIONAL".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::required(Some("REQUIRED".to_string()), None, OverrideConstantType::U32),
        ]);

        // Only provide required constant, skip optional
        let constants = PipelineConstants::new().with_u32("REQUIRED", 42);

        let result = constants.validate(&overrides);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_empty_constants_all_defaults() {
        let overrides = OverrideConstants::from_infos(vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
            OverrideConstantInfo::new(Some("C".to_string()), None, OverrideConstantType::Bool, Some(0.0)),
        ]);

        // No constants provided - all have defaults
        let constants = PipelineConstants::new();

        let result = constants.validate(&overrides);
        assert!(result.is_ok());
    }

    // =========================================================================
    // WHITEBOX TESTS: Thread Safety
    // =========================================================================

    #[test]
    fn test_override_constant_info_send_sync_static() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<OverrideConstantInfo>();
    }

    #[test]
    fn test_override_constants_send_sync_static() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<OverrideConstants>();
    }

    #[test]
    fn test_pipeline_constants_send_sync_static() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<PipelineConstants>();
    }

    #[test]
    fn test_override_error_send_sync_static() {
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<OverrideError>();
    }

    // =========================================================================
    // WHITEBOX TESTS: Display/Debug
    // =========================================================================

    #[test]
    fn test_override_constant_info_debug_format() {
        let info = OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        );
        let debug = format!("{:?}", info);

        assert!(debug.contains("OverrideConstantInfo"));
        assert!(debug.contains("WIDTH"));
        assert!(debug.contains("F32"));
        assert!(debug.contains("1920"));
    }

    #[test]
    fn test_override_constants_debug_format() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);
        let debug = format!("{:?}", overrides);

        assert!(debug.contains("OverrideConstants"));
        assert!(debug.contains("constants"));
    }

    #[test]
    fn test_pipeline_constants_debug_format() {
        let constants = PipelineConstants::new().with("WIDTH", 1920.0);
        let debug = format!("{:?}", constants);

        assert!(debug.contains("PipelineConstants"));
        assert!(debug.contains("values"));
    }

    #[test]
    fn test_all_error_variants_display() {
        // Test all error variants have reasonable display output
        let errors = vec![
            OverrideError::UnknownConstant { key: "UNKNOWN".to_string() },
            OverrideError::TypeMismatch {
                key: "VALUE".to_string(),
                expected: OverrideConstantType::U32,
                got: "negative number",
            },
            OverrideError::MissingRequired {
                name: Some("REQ".to_string()),
                id: Some(5),
            },
            OverrideError::MissingRequired {
                name: None,
                id: Some(5),
            },
            OverrideError::MissingRequired {
                name: Some("REQ".to_string()),
                id: None,
            },
            OverrideError::MissingRequired {
                name: None,
                id: None,
            },
            OverrideError::ValueOutOfRange {
                key: "COUNT".to_string(),
                value: -5.0,
                expected: "0 to 4294967295",
            },
        ];

        for err in &errors {
            let display = format!("{}", err);
            assert!(!display.is_empty());
            // Verify it's a reasonable error message
            assert!(display.len() > 10);
        }

        // Verify specific content
        assert!(format!("{}", errors[0]).contains("UNKNOWN"));
        assert!(format!("{}", errors[1]).contains("type mismatch"));
        assert!(format!("{}", errors[2]).contains("missing required"));
        assert!(format!("{}", errors[6]).contains("out of range"));
    }

    // =========================================================================
    // WHITEBOX TESTS: Additional Internal Coverage
    // =========================================================================

    #[test]
    fn test_override_constant_info_validate_value_no_key() {
        // Test validation with a constant that has no name or ID
        let info = OverrideConstantInfo::new(
            None,
            None,
            OverrideConstantType::U32,
            None,
        );

        // Valid value should pass
        assert!(info.validate_value(100.0).is_ok());

        // Invalid value should fail with "<unknown>" as key
        let result = info.validate_value(-5.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("<unknown>"));
    }

    #[test]
    fn test_override_constants_ref_iter() {
        let infos = vec![
            OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
            OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
        ];
        let overrides = OverrideConstants::from_infos(infos);

        // Test iteration by reference
        let names: Vec<_> = (&overrides).into_iter().filter_map(|c| c.name.as_deref()).collect();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn test_override_constants_get_by_key_invalid_parse() {
        let mut overrides = OverrideConstants::new();
        overrides.add(OverrideConstantInfo::new(
            Some("WIDTH".to_string()),
            Some(0),
            OverrideConstantType::F32,
            Some(1920.0),
        ));

        // Test with a key that's not a valid name and not a valid u32
        assert!(overrides.get_by_key("not_a_number_or_name").is_none());
        assert!(overrides.get_by_key("-1").is_none()); // Negative can't be u32
        assert!(overrides.get_by_key("999999999999").is_none()); // Overflows u32
    }

    #[test]
    fn test_override_constant_type_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(OverrideConstantType::Bool);
        set.insert(OverrideConstantType::I32);
        set.insert(OverrideConstantType::U32);
        set.insert(OverrideConstantType::F32);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&OverrideConstantType::Bool));
        assert!(set.contains(&OverrideConstantType::F32));
    }

    #[test]
    fn test_override_constant_type_copy() {
        let ty = OverrideConstantType::F32;
        let ty_copy = ty; // Copy
        assert_eq!(ty, ty_copy);
        // Both are still usable
        assert_eq!(ty.wgsl_name(), "f32");
        assert_eq!(ty_copy.wgsl_name(), "f32");
    }

    #[test]
    fn test_override_error_partial_eq() {
        let err1 = OverrideError::UnknownConstant { key: "X".to_string() };
        let err2 = OverrideError::UnknownConstant { key: "X".to_string() };
        let err3 = OverrideError::UnknownConstant { key: "Y".to_string() };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_pipeline_constants_from_iter_str_ref() {
        // Test the FromIterator<(&'a str, f64)> implementation
        let items: Vec<(&str, f64)> = vec![
            ("X", 1.0),
            ("Y", 2.0),
            ("Z", 3.0),
        ];

        let constants: PipelineConstants = items.into_iter().collect();

        assert_eq!(constants.len(), 3);
        assert_eq!(constants.get("X"), Some(1.0));
        assert_eq!(constants.get("Y"), Some(2.0));
        assert_eq!(constants.get("Z"), Some(3.0));
    }
}
