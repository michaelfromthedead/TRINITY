//! Provenance Chain System for TRINITY asset pipeline (T-AS-4.7).
//!
//! Provides complete asset provenance tracking for reproducibility and debugging:
//!
//! - **Provenance Records**: Source hash, import timestamp, tool version,
//!   processing parameters, intermediate hashes, cook config, dependencies
//! - **Content-Addressed Storage**: Provenance stored in ContentStore as
//!   `provenance/<asset_guid>/` tree with sentinel markers
//! - **Incremental Rebuild**: Skip processing if source hash + params unchanged
//! - **Reproducibility**: Same source hash + same params = same output
//! - **Debug Tracing**: Identify which processing step introduced artifacts
//! - **ContentDiffer Compatible**: Diff provenance trees for change tracking
//!
//! # Architecture
//!
//! ```text
//! ProvenanceChain
//!   |-- source_hash         (input content hash)
//!   |-- import_timestamp    (when imported)
//!   |-- tool_version        (importer version)
//!   |-- processing_params   (all processing settings)
//!   |-- steps[]             (ProcessingStep chain)
//!   |     |-- step_name
//!   |     |-- input_hash
//!   |     |-- output_hash
//!   |     |-- params_hash
//!   |     |-- duration_ms
//!   |-- cook_config         (platform/build config)
//!   |-- dependencies[]      (referenced asset GUIDs)
//!   |-- final_output_hash   (cooked asset hash)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::provenance::{ProvenanceChain, ProcessingStep};
//!
//! // Start provenance tracking for an asset
//! let mut chain = ProvenanceChain::new(source_hash, "texture_importer", "1.2.0");
//!
//! // Record processing steps
//! chain.add_step(ProcessingStep::new("decode_png", source_hash, decoded_hash));
//! chain.add_step(ProcessingStep::new("generate_mipmaps", decoded_hash, mipmap_hash));
//! chain.add_step(ProcessingStep::new("compress_bc7", mipmap_hash, final_hash));
//!
//! // Set cook config and finalize
//! chain.set_cook_config(CookConfig::for_platform(Platform::Windows, Quality::High));
//! chain.finalize(final_hash);
//!
//! // Store in ContentStore
//! let store = FileContentStore::new("/assets/provenance", Default::default())?;
//! let tree = chain.to_tree();
//! store.put_tree(&asset_guid, &tree)?;
//!
//! // Check for incremental rebuild
//! if chain.needs_rebuild(&previous_chain) {
//!     // Re-process the asset
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during provenance operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    /// Missing required field in provenance record.
    MissingField(&'static str),
    /// Invalid provenance data format.
    InvalidFormat(String),
    /// Hash chain verification failed.
    ChainVerificationFailed {
        step_name: String,
        expected: ContentHash,
        actual: ContentHash,
    },
    /// Cyclic dependency detected.
    CyclicDependency(String),
    /// Serialization/deserialization error.
    SerializationError(String),
    /// I/O error.
    IoError(String),
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "missing required field: {}", field),
            Self::InvalidFormat(msg) => write!(f, "invalid provenance format: {}", msg),
            Self::ChainVerificationFailed { step_name, expected, actual } => {
                write!(
                    f,
                    "chain verification failed at '{}': expected {}, got {}",
                    step_name, expected, actual
                )
            }
            Self::CyclicDependency(asset) => write!(f, "cyclic dependency detected: {}", asset),
            Self::SerializationError(msg) => write!(f, "serialization error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for ProvenanceError {}

impl From<io::Error> for ProvenanceError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

/// Result type for provenance operations.
pub type ProvenanceResult<T> = Result<T, ProvenanceError>;

// ---------------------------------------------------------------------------
// Timestamp Utilities
// ---------------------------------------------------------------------------

/// Get current timestamp in milliseconds since UNIX epoch.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Format timestamp as ISO 8601 string.
fn format_timestamp(timestamp_ms: u64) -> String {
    // Simple ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ
    let secs = timestamp_ms / 1000;
    let millis = timestamp_ms % 1000;

    // Calculate date/time components (simplified, assumes UTC)
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Approximate year/month/day calculation from days since epoch
    let mut year = 1970;
    let mut remaining_days = days as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let (month, day) = day_of_year_to_month_day(remaining_days as u32 + 1, is_leap_year(year));

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn day_of_year_to_month_day(day_of_year: u32, leap: bool) -> (u32, u32) {
    let days_in_months = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut remaining = day_of_year;
    for (i, &days) in days_in_months.iter().enumerate() {
        if remaining <= days {
            return ((i + 1) as u32, remaining);
        }
        remaining -= days;
    }
    (12, 31) // Fallback
}

// ---------------------------------------------------------------------------
// Processing Parameters
// ---------------------------------------------------------------------------

/// A single processing parameter (key-value pair).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingParam {
    /// Parameter name.
    pub name: String,
    /// Parameter value (serialized as string).
    pub value: String,
}

impl ProcessingParam {
    /// Create a new processing parameter.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Create from a boolean value.
    pub fn from_bool(name: impl Into<String>, value: bool) -> Self {
        Self::new(name, if value { "true" } else { "false" })
    }

    /// Create from an integer value.
    pub fn from_int(name: impl Into<String>, value: i64) -> Self {
        Self::new(name, value.to_string())
    }

    /// Create from a float value.
    pub fn from_float(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, format!("{:.6}", value))
    }

    /// Parse value as boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self.value.as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        }
    }

    /// Parse value as integer.
    pub fn as_int(&self) -> Option<i64> {
        self.value.parse().ok()
    }

    /// Parse value as float.
    pub fn as_float(&self) -> Option<f64> {
        self.value.parse().ok()
    }
}

/// Collection of processing parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessingParams {
    /// Parameters in insertion order.
    params: Vec<ProcessingParam>,
}

impl ProcessingParams {
    /// Create an empty parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter.
    pub fn add(&mut self, param: ProcessingParam) {
        // Remove existing param with same name
        self.params.retain(|p| p.name != param.name);
        self.params.push(param);
    }

    /// Add a string parameter.
    pub fn add_str(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.add(ProcessingParam::new(name, value));
    }

    /// Add a boolean parameter.
    pub fn add_bool(&mut self, name: impl Into<String>, value: bool) {
        self.add(ProcessingParam::from_bool(name, value));
    }

    /// Add an integer parameter.
    pub fn add_int(&mut self, name: impl Into<String>, value: i64) {
        self.add(ProcessingParam::from_int(name, value));
    }

    /// Add a float parameter.
    pub fn add_float(&mut self, name: impl Into<String>, value: f64) {
        self.add(ProcessingParam::from_float(name, value));
    }

    /// Get a parameter by name.
    pub fn get(&self, name: &str) -> Option<&ProcessingParam> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Get a parameter value by name.
    pub fn get_value(&self, name: &str) -> Option<&str> {
        self.get(name).map(|p| p.value.as_str())
    }

    /// Check if parameter set is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get number of parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Iterate over parameters.
    pub fn iter(&self) -> impl Iterator<Item = &ProcessingParam> {
        self.params.iter()
    }

    /// Compute a deterministic hash of all parameters.
    ///
    /// Parameters are sorted by name before hashing for determinism.
    pub fn compute_hash(&self) -> ContentHash {
        let mut sorted: Vec<_> = self.params.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));

        let mut data = Vec::new();
        for param in sorted {
            data.extend_from_slice(param.name.as_bytes());
            data.push(0); // Separator
            data.extend_from_slice(param.value.as_bytes());
            data.push(0); // Separator
        }

        ContentHash::from_bytes(&data)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Write parameter count
        data.extend_from_slice(&(self.params.len() as u32).to_le_bytes());

        for param in &self.params {
            // Write name length and name
            data.extend_from_slice(&(param.name.len() as u32).to_le_bytes());
            data.extend_from_slice(param.name.as_bytes());

            // Write value length and value
            data.extend_from_slice(&(param.value.len() as u32).to_le_bytes());
            data.extend_from_slice(param.value.as_bytes());
        }

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> ProvenanceResult<Self> {
        if data.len() < 4 {
            return Err(ProvenanceError::InvalidFormat("params too short".into()));
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut params = Vec::with_capacity(count);
        let mut pos = 4;

        for _ in 0..count {
            // Read name
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated param name length".into()));
            }
            let name_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + name_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated param name".into()));
            }
            let name = String::from_utf8(data[pos..pos + name_len].to_vec())
                .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in param name".into()))?;
            pos += name_len;

            // Read value
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated param value length".into()));
            }
            let value_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + value_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated param value".into()));
            }
            let value = String::from_utf8(data[pos..pos + value_len].to_vec())
                .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in param value".into()))?;
            pos += value_len;

            params.push(ProcessingParam { name, value });
        }

        Ok(Self { params })
    }
}

// ---------------------------------------------------------------------------
// Processing Step
// ---------------------------------------------------------------------------

/// A single step in the processing chain.
///
/// Each step records:
/// - The step name (e.g., "decode_png", "generate_mipmaps")
/// - Input and output content hashes
/// - Processing parameters for this step
/// - Duration (for profiling)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingStep {
    /// Step name/identifier.
    pub name: String,
    /// Hash of input content.
    pub input_hash: ContentHash,
    /// Hash of output content.
    pub output_hash: ContentHash,
    /// Parameters used in this step.
    pub params: ProcessingParams,
    /// Processing duration in milliseconds.
    pub duration_ms: u64,
    /// Optional error message if step failed.
    pub error: Option<String>,
}

impl ProcessingStep {
    /// Create a new processing step.
    pub fn new(
        name: impl Into<String>,
        input_hash: ContentHash,
        output_hash: ContentHash,
    ) -> Self {
        Self {
            name: name.into(),
            input_hash,
            output_hash,
            params: ProcessingParams::new(),
            duration_ms: 0,
            error: None,
        }
    }

    /// Create a step with parameters.
    pub fn with_params(mut self, params: ProcessingParams) -> Self {
        self.params = params;
        self
    }

    /// Set the processing duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Mark step as failed with error.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Check if this step succeeded.
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Compute a hash of this step's configuration (for rebuild detection).
    pub fn config_hash(&self) -> ContentHash {
        let mut data = Vec::new();
        data.extend_from_slice(self.name.as_bytes());
        data.push(0);
        data.extend_from_slice(self.input_hash.as_bytes());
        data.extend_from_slice(&self.params.compute_hash().into_bytes());
        ContentHash::from_bytes(&data)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Name
        data.extend_from_slice(&(self.name.len() as u32).to_le_bytes());
        data.extend_from_slice(self.name.as_bytes());

        // Hashes
        data.extend_from_slice(self.input_hash.as_bytes());
        data.extend_from_slice(self.output_hash.as_bytes());

        // Params
        let params_bytes = self.params.to_bytes();
        data.extend_from_slice(&(params_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&params_bytes);

        // Duration
        data.extend_from_slice(&self.duration_ms.to_le_bytes());

        // Error
        match &self.error {
            Some(e) => {
                data.push(1);
                data.extend_from_slice(&(e.len() as u32).to_le_bytes());
                data.extend_from_slice(e.as_bytes());
            }
            None => {
                data.push(0);
            }
        }

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> ProvenanceResult<Self> {
        if data.len() < 4 {
            return Err(ProvenanceError::InvalidFormat("step too short".into()));
        }

        let mut pos = 0;

        // Name
        let name_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + name_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated step name".into()));
        }
        let name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in step name".into()))?;
        pos += name_len;

        // Hashes
        if pos + 64 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated step hashes".into()));
        }
        let mut input_bytes = [0u8; 32];
        input_bytes.copy_from_slice(&data[pos..pos + 32]);
        let input_hash = ContentHash::from_raw(input_bytes);
        pos += 32;

        let mut output_bytes = [0u8; 32];
        output_bytes.copy_from_slice(&data[pos..pos + 32]);
        let output_hash = ContentHash::from_raw(output_bytes);
        pos += 32;

        // Params
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated params length".into()));
        }
        let params_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + params_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated params data".into()));
        }
        let params = ProcessingParams::from_bytes(&data[pos..pos + params_len])?;
        pos += params_len;

        // Duration
        if pos + 8 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated duration".into()));
        }
        let duration_ms = u64::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]);
        pos += 8;

        // Error
        if pos >= data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated error flag".into()));
        }
        let has_error = data[pos] != 0;
        pos += 1;

        let error = if has_error {
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated error length".into()));
            }
            let error_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + error_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated error message".into()));
            }
            let error_msg = String::from_utf8(data[pos..pos + error_len].to_vec())
                .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in error".into()))?;
            Some(error_msg)
        } else {
            None
        };

        Ok(Self {
            name,
            input_hash,
            output_hash,
            params,
            duration_ms,
            error,
        })
    }
}

// ---------------------------------------------------------------------------
// Cook Configuration
// ---------------------------------------------------------------------------

/// Target platform for cooking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    /// Windows desktop.
    Windows,
    /// Linux desktop.
    Linux,
    /// macOS desktop.
    MacOS,
    /// iOS mobile.
    IOS,
    /// Android mobile.
    Android,
    /// Web (WebGPU/WebGL).
    Web,
    /// Generic/unknown platform.
    Generic,
}

impl Platform {
    /// Get platform name as string.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Windows => "windows",
            Self::Linux => "linux",
            Self::MacOS => "macos",
            Self::IOS => "ios",
            Self::Android => "android",
            Self::Web => "web",
            Self::Generic => "generic",
        }
    }

    /// Parse platform from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "windows" => Some(Self::Windows),
            "linux" => Some(Self::Linux),
            "macos" => Some(Self::MacOS),
            "ios" => Some(Self::IOS),
            "android" => Some(Self::Android),
            "web" => Some(Self::Web),
            "generic" => Some(Self::Generic),
            _ => None,
        }
    }

    /// Get platform as byte for serialization.
    pub const fn to_byte(&self) -> u8 {
        match self {
            Self::Windows => 1,
            Self::Linux => 2,
            Self::MacOS => 3,
            Self::IOS => 4,
            Self::Android => 5,
            Self::Web => 6,
            Self::Generic => 0,
        }
    }

    /// Parse platform from byte.
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::Windows,
            2 => Self::Linux,
            3 => Self::MacOS,
            4 => Self::IOS,
            5 => Self::Android,
            6 => Self::Web,
            _ => Self::Generic,
        }
    }
}

impl Default for Platform {
    fn default() -> Self {
        Self::Generic
    }
}

/// Quality level for cooking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityLevel {
    /// Low quality (fast, small).
    Low,
    /// Medium quality (balanced).
    Medium,
    /// High quality (slow, large).
    High,
    /// Ultra quality (maximum).
    Ultra,
}

impl QualityLevel {
    /// Get quality level name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    /// Parse from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" | "max" => Some(Self::Ultra),
            _ => None,
        }
    }

    /// Get as byte.
    pub const fn to_byte(&self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Ultra => 4,
        }
    }

    /// Parse from byte.
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            4 => Self::Ultra,
            _ => Self::Medium,
        }
    }
}

impl Default for QualityLevel {
    fn default() -> Self {
        Self::Medium
    }
}

/// Cook configuration for asset processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CookConfig {
    /// Target platform.
    pub platform: Platform,
    /// Quality level.
    pub quality: QualityLevel,
    /// Debug mode (preserves more data, less optimization).
    pub debug: bool,
    /// Additional platform-specific settings.
    pub platform_params: ProcessingParams,
}

impl CookConfig {
    /// Create a new cook config for a platform.
    pub fn for_platform(platform: Platform, quality: QualityLevel) -> Self {
        Self {
            platform,
            quality,
            debug: false,
            platform_params: ProcessingParams::new(),
        }
    }

    /// Create a debug cook config.
    pub fn debug(platform: Platform) -> Self {
        Self {
            platform,
            quality: QualityLevel::High,
            debug: true,
            platform_params: ProcessingParams::new(),
        }
    }

    /// Add a platform parameter.
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.platform_params.add_str(name, value);
        self
    }

    /// Compute a deterministic hash of this config.
    pub fn compute_hash(&self) -> ContentHash {
        let mut data = Vec::new();
        data.push(self.platform.to_byte());
        data.push(self.quality.to_byte());
        data.push(if self.debug { 1 } else { 0 });
        data.extend_from_slice(&self.platform_params.compute_hash().into_bytes());
        ContentHash::from_bytes(&data)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(self.platform.to_byte());
        data.push(self.quality.to_byte());
        data.push(if self.debug { 1 } else { 0 });

        let params_bytes = self.platform_params.to_bytes();
        data.extend_from_slice(&(params_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&params_bytes);

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> ProvenanceResult<Self> {
        if data.len() < 7 {
            return Err(ProvenanceError::InvalidFormat("cook config too short".into()));
        }

        let platform = Platform::from_byte(data[0]);
        let quality = QualityLevel::from_byte(data[1]);
        let debug = data[2] != 0;

        let params_len = u32::from_le_bytes([data[3], data[4], data[5], data[6]]) as usize;
        if data.len() < 7 + params_len {
            return Err(ProvenanceError::InvalidFormat("truncated cook config params".into()));
        }
        let platform_params = ProcessingParams::from_bytes(&data[7..7 + params_len])?;

        Ok(Self {
            platform,
            quality,
            debug,
            platform_params,
        })
    }
}

impl Default for CookConfig {
    fn default() -> Self {
        Self::for_platform(Platform::Generic, QualityLevel::Medium)
    }
}

// ---------------------------------------------------------------------------
// Dependency Reference
// ---------------------------------------------------------------------------

/// Reference to a dependent asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DependencyRef {
    /// Asset GUID of the dependency.
    pub asset_guid: String,
    /// Content hash of the dependency at import time.
    pub content_hash: ContentHash,
    /// Type of dependency.
    pub dep_type: DependencyType,
}

/// Type of dependency relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyType {
    /// Direct reference (e.g., texture used by material).
    Direct,
    /// Indirect reference (e.g., included header file).
    Indirect,
    /// Runtime reference (resolved at runtime).
    Runtime,
}

impl DependencyType {
    /// Get as byte.
    pub const fn to_byte(&self) -> u8 {
        match self {
            Self::Direct => 1,
            Self::Indirect => 2,
            Self::Runtime => 3,
        }
    }

    /// Parse from byte.
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::Direct,
            2 => Self::Indirect,
            3 => Self::Runtime,
            _ => Self::Direct,
        }
    }
}

impl Default for DependencyType {
    fn default() -> Self {
        Self::Direct
    }
}

impl DependencyRef {
    /// Create a new dependency reference.
    pub fn new(
        asset_guid: impl Into<String>,
        content_hash: ContentHash,
        dep_type: DependencyType,
    ) -> Self {
        Self {
            asset_guid: asset_guid.into(),
            content_hash,
            dep_type,
        }
    }

    /// Create a direct dependency.
    pub fn direct(asset_guid: impl Into<String>, content_hash: ContentHash) -> Self {
        Self::new(asset_guid, content_hash, DependencyType::Direct)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend_from_slice(&(self.asset_guid.len() as u32).to_le_bytes());
        data.extend_from_slice(self.asset_guid.as_bytes());
        data.extend_from_slice(self.content_hash.as_bytes());
        data.push(self.dep_type.to_byte());

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> ProvenanceResult<Self> {
        if data.len() < 4 {
            return Err(ProvenanceError::InvalidFormat("dependency too short".into()));
        }

        let guid_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut pos = 4;

        if pos + guid_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated dependency guid".into()));
        }
        let asset_guid = String::from_utf8(data[pos..pos + guid_len].to_vec())
            .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in guid".into()))?;
        pos += guid_len;

        if pos + 33 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated dependency hash".into()));
        }
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(&data[pos..pos + 32]);
        let content_hash = ContentHash::from_raw(hash_bytes);
        pos += 32;

        let dep_type = DependencyType::from_byte(data[pos]);

        Ok(Self {
            asset_guid,
            content_hash,
            dep_type,
        })
    }
}

// ---------------------------------------------------------------------------
// Provenance Chain
// ---------------------------------------------------------------------------

/// Sentinel marker for provenance tree nodes.
pub const PROVENANCE_SENTINEL: u32 = 0x50524F56; // "PROV" in big-endian

/// Version of the provenance format.
pub const PROVENANCE_VERSION: u32 = 1;

/// Complete provenance chain for an asset.
///
/// Tracks all processing steps from source to final cooked asset,
/// enabling incremental rebuilds and reproducibility verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceChain {
    /// Hash of the original source content.
    pub source_hash: ContentHash,
    /// Import timestamp (ms since UNIX epoch).
    pub import_timestamp_ms: u64,
    /// Importer tool name.
    pub tool_name: String,
    /// Importer tool version.
    pub tool_version: String,
    /// Global processing parameters (apply to all steps).
    pub global_params: ProcessingParams,
    /// Ordered list of processing steps.
    pub steps: Vec<ProcessingStep>,
    /// Cook configuration.
    pub cook_config: CookConfig,
    /// Dependencies on other assets.
    pub dependencies: Vec<DependencyRef>,
    /// Hash of final output content.
    pub final_output_hash: Option<ContentHash>,
    /// Whether processing completed successfully.
    pub success: bool,
    /// Optional error message if processing failed.
    pub error_message: Option<String>,
}

impl ProvenanceChain {
    /// Create a new provenance chain.
    pub fn new(
        source_hash: ContentHash,
        tool_name: impl Into<String>,
        tool_version: impl Into<String>,
    ) -> Self {
        Self {
            source_hash,
            import_timestamp_ms: current_timestamp_ms(),
            tool_name: tool_name.into(),
            tool_version: tool_version.into(),
            global_params: ProcessingParams::new(),
            steps: Vec::new(),
            cook_config: CookConfig::default(),
            dependencies: Vec::new(),
            final_output_hash: None,
            success: false,
            error_message: None,
        }
    }

    /// Set the import timestamp.
    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.import_timestamp_ms = timestamp_ms;
        self
    }

    /// Set global parameters.
    pub fn with_global_params(mut self, params: ProcessingParams) -> Self {
        self.global_params = params;
        self
    }

    /// Set cook configuration.
    pub fn set_cook_config(&mut self, config: CookConfig) {
        self.cook_config = config;
    }

    /// Add a processing step.
    pub fn add_step(&mut self, step: ProcessingStep) {
        self.steps.push(step);
    }

    /// Add a dependency.
    pub fn add_dependency(&mut self, dep: DependencyRef) {
        self.dependencies.push(dep);
    }

    /// Finalize the chain with the final output hash.
    pub fn finalize(&mut self, output_hash: ContentHash) {
        self.final_output_hash = Some(output_hash);
        self.success = true;
    }

    /// Mark the chain as failed.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.success = false;
        self.error_message = Some(error.into());
    }

    /// Check if processing completed successfully.
    pub fn is_success(&self) -> bool {
        self.success && self.final_output_hash.is_some()
    }

    /// Get the formatted import timestamp.
    pub fn import_timestamp_formatted(&self) -> String {
        format_timestamp(self.import_timestamp_ms)
    }

    /// Get total processing duration across all steps.
    pub fn total_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.duration_ms).sum()
    }

    /// Get the last output hash in the chain.
    pub fn last_output_hash(&self) -> Option<ContentHash> {
        self.steps.last().map(|s| s.output_hash)
    }

    /// Verify the hash chain integrity.
    ///
    /// Checks that each step's input hash matches the previous step's output hash.
    pub fn verify_chain(&self) -> ProvenanceResult<()> {
        let mut expected_input = self.source_hash;

        for step in &self.steps {
            if step.input_hash != expected_input {
                return Err(ProvenanceError::ChainVerificationFailed {
                    step_name: step.name.clone(),
                    expected: expected_input,
                    actual: step.input_hash,
                });
            }
            expected_input = step.output_hash;
        }

        // Verify final output matches last step
        if let (Some(final_hash), Some(last_step)) = (&self.final_output_hash, self.steps.last()) {
            if *final_hash != last_step.output_hash {
                return Err(ProvenanceError::ChainVerificationFailed {
                    step_name: "final_output".to_string(),
                    expected: last_step.output_hash,
                    actual: *final_hash,
                });
            }
        }

        Ok(())
    }

    /// Compute a configuration hash for rebuild detection.
    ///
    /// Includes: source hash, tool version, global params, cook config, dependencies.
    pub fn config_hash(&self) -> ContentHash {
        let mut data = Vec::new();

        // Source
        data.extend_from_slice(self.source_hash.as_bytes());

        // Tool
        data.extend_from_slice(self.tool_name.as_bytes());
        data.push(0);
        data.extend_from_slice(self.tool_version.as_bytes());
        data.push(0);

        // Global params
        data.extend_from_slice(&self.global_params.compute_hash().into_bytes());

        // Cook config
        data.extend_from_slice(&self.cook_config.compute_hash().into_bytes());

        // Dependencies (sorted by GUID for determinism)
        let mut deps: Vec<_> = self.dependencies.iter().collect();
        deps.sort_by(|a, b| a.asset_guid.cmp(&b.asset_guid));
        for dep in deps {
            data.extend_from_slice(dep.asset_guid.as_bytes());
            data.extend_from_slice(dep.content_hash.as_bytes());
        }

        ContentHash::from_bytes(&data)
    }

    /// Check if this chain requires a rebuild compared to another.
    ///
    /// Returns true if source, params, config, or dependencies differ.
    pub fn needs_rebuild(&self, other: &ProvenanceChain) -> bool {
        // Check source hash
        if self.source_hash != other.source_hash {
            return true;
        }

        // Check tool version
        if self.tool_name != other.tool_name || self.tool_version != other.tool_version {
            return true;
        }

        // Check config hash (includes params, cook config, deps)
        if self.config_hash() != other.config_hash() {
            return true;
        }

        false
    }

    /// Find the step where output first differs from another chain.
    ///
    /// Useful for debugging which processing step introduced a difference.
    pub fn find_divergence(&self, other: &ProvenanceChain) -> Option<(usize, &str)> {
        if self.source_hash != other.source_hash {
            return Some((0, "source"));
        }

        for (i, (a, b)) in self.steps.iter().zip(other.steps.iter()).enumerate() {
            if a.output_hash != b.output_hash {
                return Some((i + 1, &a.name));
            }
        }

        if self.steps.len() != other.steps.len() {
            return Some((self.steps.len().min(other.steps.len()), "step_count"));
        }

        None
    }

    /// Serialize to bytes with sentinel markers.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header: sentinel + version
        data.extend_from_slice(&PROVENANCE_SENTINEL.to_le_bytes());
        data.extend_from_slice(&PROVENANCE_VERSION.to_le_bytes());

        // Source hash
        data.extend_from_slice(self.source_hash.as_bytes());

        // Timestamp
        data.extend_from_slice(&self.import_timestamp_ms.to_le_bytes());

        // Tool name
        data.extend_from_slice(&(self.tool_name.len() as u32).to_le_bytes());
        data.extend_from_slice(self.tool_name.as_bytes());

        // Tool version
        data.extend_from_slice(&(self.tool_version.len() as u32).to_le_bytes());
        data.extend_from_slice(self.tool_version.as_bytes());

        // Global params
        let params_bytes = self.global_params.to_bytes();
        data.extend_from_slice(&(params_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&params_bytes);

        // Steps
        data.extend_from_slice(&(self.steps.len() as u32).to_le_bytes());
        for step in &self.steps {
            let step_bytes = step.to_bytes();
            data.extend_from_slice(&(step_bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(&step_bytes);
        }

        // Cook config
        let config_bytes = self.cook_config.to_bytes();
        data.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&config_bytes);

        // Dependencies
        data.extend_from_slice(&(self.dependencies.len() as u32).to_le_bytes());
        for dep in &self.dependencies {
            let dep_bytes = dep.to_bytes();
            data.extend_from_slice(&(dep_bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(&dep_bytes);
        }

        // Final output hash
        match &self.final_output_hash {
            Some(hash) => {
                data.push(1);
                data.extend_from_slice(hash.as_bytes());
            }
            None => {
                data.push(0);
            }
        }

        // Success flag
        data.push(if self.success { 1 } else { 0 });

        // Error message
        match &self.error_message {
            Some(msg) => {
                data.push(1);
                data.extend_from_slice(&(msg.len() as u32).to_le_bytes());
                data.extend_from_slice(msg.as_bytes());
            }
            None => {
                data.push(0);
            }
        }

        // Footer sentinel
        data.extend_from_slice(&PROVENANCE_SENTINEL.to_le_bytes());

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> ProvenanceResult<Self> {
        if data.len() < 12 {
            return Err(ProvenanceError::InvalidFormat("provenance data too short".into()));
        }

        let mut pos = 0;

        // Header sentinel
        let sentinel = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if sentinel != PROVENANCE_SENTINEL {
            return Err(ProvenanceError::InvalidFormat(format!(
                "invalid sentinel: expected {:08x}, got {:08x}",
                PROVENANCE_SENTINEL, sentinel
            )));
        }
        pos += 4;

        // Version
        let version = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        if version != PROVENANCE_VERSION {
            return Err(ProvenanceError::InvalidFormat(format!(
                "unsupported version: {}",
                version
            )));
        }
        pos += 4;

        // Source hash
        if pos + 32 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated source hash".into()));
        }
        let mut source_bytes = [0u8; 32];
        source_bytes.copy_from_slice(&data[pos..pos + 32]);
        let source_hash = ContentHash::from_raw(source_bytes);
        pos += 32;

        // Timestamp
        if pos + 8 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated timestamp".into()));
        }
        let import_timestamp_ms = u64::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]);
        pos += 8;

        // Tool name
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated tool name length".into()));
        }
        let name_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + name_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated tool name".into()));
        }
        let tool_name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in tool name".into()))?;
        pos += name_len;

        // Tool version
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated tool version length".into()));
        }
        let version_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + version_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated tool version".into()));
        }
        let tool_version = String::from_utf8(data[pos..pos + version_len].to_vec())
            .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in tool version".into()))?;
        pos += version_len;

        // Global params
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated global params length".into()));
        }
        let params_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + params_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated global params".into()));
        }
        let global_params = ProcessingParams::from_bytes(&data[pos..pos + params_len])?;
        pos += params_len;

        // Steps
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated steps count".into()));
        }
        let steps_count = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        let mut steps = Vec::with_capacity(steps_count);
        for _ in 0..steps_count {
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated step length".into()));
            }
            let step_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + step_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated step data".into()));
            }
            let step = ProcessingStep::from_bytes(&data[pos..pos + step_len])?;
            steps.push(step);
            pos += step_len;
        }

        // Cook config
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated cook config length".into()));
        }
        let config_len = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        if pos + config_len > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated cook config".into()));
        }
        let cook_config = CookConfig::from_bytes(&data[pos..pos + config_len])?;
        pos += config_len;

        // Dependencies
        if pos + 4 > data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated deps count".into()));
        }
        let deps_count = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        ]) as usize;
        pos += 4;

        let mut dependencies = Vec::with_capacity(deps_count);
        for _ in 0..deps_count {
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated dep length".into()));
            }
            let dep_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + dep_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated dep data".into()));
            }
            let dep = DependencyRef::from_bytes(&data[pos..pos + dep_len])?;
            dependencies.push(dep);
            pos += dep_len;
        }

        // Final output hash
        if pos >= data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated final hash flag".into()));
        }
        let has_final = data[pos] != 0;
        pos += 1;

        let final_output_hash = if has_final {
            if pos + 32 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated final hash".into()));
            }
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&data[pos..pos + 32]);
            pos += 32;
            Some(ContentHash::from_raw(hash_bytes))
        } else {
            None
        };

        // Success flag
        if pos >= data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated success flag".into()));
        }
        let success = data[pos] != 0;
        pos += 1;

        // Error message
        if pos >= data.len() {
            return Err(ProvenanceError::InvalidFormat("truncated error flag".into()));
        }
        let has_error = data[pos] != 0;
        pos += 1;

        let error_message = if has_error {
            if pos + 4 > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated error length".into()));
            }
            let error_len = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            ]) as usize;
            pos += 4;

            if pos + error_len > data.len() {
                return Err(ProvenanceError::InvalidFormat("truncated error message".into()));
            }
            let msg = String::from_utf8(data[pos..pos + error_len].to_vec())
                .map_err(|_| ProvenanceError::InvalidFormat("invalid UTF-8 in error".into()))?;
            pos += error_len;
            Some(msg)
        } else {
            None
        };

        // Footer sentinel (optional check)
        if pos + 4 <= data.len() {
            let footer = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            if footer != PROVENANCE_SENTINEL {
                return Err(ProvenanceError::InvalidFormat("invalid footer sentinel".into()));
            }
        }

        Ok(Self {
            source_hash,
            import_timestamp_ms,
            tool_name,
            tool_version,
            global_params,
            steps,
            cook_config,
            dependencies,
            final_output_hash,
            success,
            error_message,
        })
    }
}

// ---------------------------------------------------------------------------
// Provenance Tree (for ContentStore storage)
// ---------------------------------------------------------------------------

/// A tree node in the provenance storage structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceTreeNode {
    /// Blob node with raw data.
    Blob {
        name: String,
        data: Vec<u8>,
        hash: ContentHash,
    },
    /// Directory node with children.
    Directory {
        name: String,
        children: Vec<ProvenanceTreeNode>,
    },
}

impl ProvenanceTreeNode {
    /// Create a blob node.
    pub fn blob(name: impl Into<String>, data: Vec<u8>) -> Self {
        let hash = ContentHash::from_bytes(&data);
        Self::Blob {
            name: name.into(),
            data,
            hash,
        }
    }

    /// Create a directory node.
    pub fn directory(name: impl Into<String>, children: Vec<ProvenanceTreeNode>) -> Self {
        Self::Directory {
            name: name.into(),
            children,
        }
    }

    /// Get the node name.
    pub fn name(&self) -> &str {
        match self {
            Self::Blob { name, .. } => name,
            Self::Directory { name, .. } => name,
        }
    }

    /// Check if this is a blob node.
    pub fn is_blob(&self) -> bool {
        matches!(self, Self::Blob { .. })
    }

    /// Check if this is a directory node.
    pub fn is_directory(&self) -> bool {
        matches!(self, Self::Directory { .. })
    }

    /// Compute the tree hash (for diffing).
    pub fn compute_hash(&self) -> ContentHash {
        match self {
            Self::Blob { hash, .. } => *hash,
            Self::Directory { name, children } => {
                let mut data = Vec::new();
                data.extend_from_slice(name.as_bytes());
                data.push(0);

                for child in children {
                    data.extend_from_slice(child.name().as_bytes());
                    data.push(0);
                    data.extend_from_slice(child.compute_hash().as_bytes());
                }

                ContentHash::from_bytes(&data)
            }
        }
    }
}

/// Convert a ProvenanceChain to a tree structure for storage.
pub fn chain_to_tree(chain: &ProvenanceChain, asset_guid: &str) -> ProvenanceTreeNode {
    let mut children = Vec::new();

    // Main provenance blob
    children.push(ProvenanceTreeNode::blob("provenance.bin", chain.to_bytes()));

    // Individual step blobs (for partial updates)
    let mut step_children = Vec::new();
    for (i, step) in chain.steps.iter().enumerate() {
        step_children.push(ProvenanceTreeNode::blob(
            format!("step_{:03}.bin", i),
            step.to_bytes(),
        ));
    }
    if !step_children.is_empty() {
        children.push(ProvenanceTreeNode::directory("steps", step_children));
    }

    // Dependencies (for quick lookup)
    let mut dep_children = Vec::new();
    for (i, dep) in chain.dependencies.iter().enumerate() {
        dep_children.push(ProvenanceTreeNode::blob(
            format!("dep_{:03}.bin", i),
            dep.to_bytes(),
        ));
    }
    if !dep_children.is_empty() {
        children.push(ProvenanceTreeNode::directory("dependencies", dep_children));
    }

    ProvenanceTreeNode::directory(format!("provenance/{}", asset_guid), children)
}

// ---------------------------------------------------------------------------
// Provenance Differ (ContentDiffer compatible)
// ---------------------------------------------------------------------------

/// Diff result for provenance chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceDiff {
    /// Whether source content changed.
    pub source_changed: bool,
    /// Whether tool version changed.
    pub tool_changed: bool,
    /// Whether global params changed.
    pub params_changed: bool,
    /// Whether cook config changed.
    pub config_changed: bool,
    /// Steps that were added.
    pub steps_added: Vec<String>,
    /// Steps that were removed.
    pub steps_removed: Vec<String>,
    /// Steps that were modified.
    pub steps_modified: Vec<String>,
    /// Dependencies that were added.
    pub deps_added: Vec<String>,
    /// Dependencies that were removed.
    pub deps_removed: Vec<String>,
    /// Dependencies that were modified.
    pub deps_modified: Vec<String>,
}

impl ProvenanceDiff {
    /// Check if there are any changes.
    pub fn is_empty(&self) -> bool {
        !self.source_changed
            && !self.tool_changed
            && !self.params_changed
            && !self.config_changed
            && self.steps_added.is_empty()
            && self.steps_removed.is_empty()
            && self.steps_modified.is_empty()
            && self.deps_added.is_empty()
            && self.deps_removed.is_empty()
            && self.deps_modified.is_empty()
    }

    /// Check if rebuild is required.
    pub fn requires_rebuild(&self) -> bool {
        self.source_changed
            || self.tool_changed
            || self.params_changed
            || self.config_changed
            || !self.deps_modified.is_empty()
    }
}

/// Compute the difference between two provenance chains.
pub fn diff_provenance(old: &ProvenanceChain, new: &ProvenanceChain) -> ProvenanceDiff {
    let source_changed = old.source_hash != new.source_hash;
    let tool_changed = old.tool_name != new.tool_name || old.tool_version != new.tool_version;
    let params_changed = old.global_params.compute_hash() != new.global_params.compute_hash();
    let config_changed = old.cook_config.compute_hash() != new.cook_config.compute_hash();

    // Compare steps
    let old_steps: HashMap<_, _> = old.steps.iter().map(|s| (&s.name, s)).collect();
    let new_steps: HashMap<_, _> = new.steps.iter().map(|s| (&s.name, s)).collect();

    let mut steps_added = Vec::new();
    let mut steps_removed = Vec::new();
    let mut steps_modified = Vec::new();

    for (name, step) in &new_steps {
        match old_steps.get(name) {
            Some(old_step) => {
                if step.output_hash != old_step.output_hash
                    || step.params.compute_hash() != old_step.params.compute_hash()
                {
                    steps_modified.push((*name).clone());
                }
            }
            None => {
                steps_added.push((*name).clone());
            }
        }
    }

    for name in old_steps.keys() {
        if !new_steps.contains_key(name) {
            steps_removed.push((*name).clone());
        }
    }

    // Compare dependencies
    let old_deps: HashMap<_, _> = old.dependencies.iter().map(|d| (&d.asset_guid, d)).collect();
    let new_deps: HashMap<_, _> = new.dependencies.iter().map(|d| (&d.asset_guid, d)).collect();

    let mut deps_added = Vec::new();
    let mut deps_removed = Vec::new();
    let mut deps_modified = Vec::new();

    for (guid, dep) in &new_deps {
        match old_deps.get(guid) {
            Some(old_dep) => {
                if dep.content_hash != old_dep.content_hash {
                    deps_modified.push((*guid).clone());
                }
            }
            None => {
                deps_added.push((*guid).clone());
            }
        }
    }

    for guid in old_deps.keys() {
        if !new_deps.contains_key(guid) {
            deps_removed.push((*guid).clone());
        }
    }

    ProvenanceDiff {
        source_changed,
        tool_changed,
        params_changed,
        config_changed,
        steps_added,
        steps_removed,
        steps_modified,
        deps_added,
        deps_removed,
        deps_modified,
    }
}

// ---------------------------------------------------------------------------
// Incremental Rebuild Support
// ---------------------------------------------------------------------------

/// Check if an asset needs rebuilding based on provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildDecision {
    /// Whether rebuild is needed.
    pub needs_rebuild: bool,
    /// Reason for rebuild decision.
    pub reason: RebuildReason,
    /// Step to resume from (if incremental rebuild possible).
    pub resume_from_step: Option<usize>,
}

/// Reason for rebuild decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildReason {
    /// Source content unchanged, no rebuild needed.
    SourceUnchanged,
    /// Source content changed.
    SourceChanged,
    /// Tool version changed.
    ToolChanged,
    /// Processing parameters changed.
    ParamsChanged,
    /// Cook configuration changed.
    ConfigChanged,
    /// Dependency changed.
    DependencyChanged(String),
    /// No previous provenance exists.
    NoPreviousProvenance,
    /// Previous processing failed.
    PreviousFailed,
}

impl RebuildDecision {
    /// Compute rebuild decision by comparing provenances.
    pub fn compute(
        current_source_hash: ContentHash,
        current_deps: &[DependencyRef],
        previous: Option<&ProvenanceChain>,
        tool_name: &str,
        tool_version: &str,
        params: &ProcessingParams,
        config: &CookConfig,
    ) -> Self {
        let Some(prev) = previous else {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::NoPreviousProvenance,
                resume_from_step: None,
            };
        };

        // Check if previous processing failed
        if !prev.success {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::PreviousFailed,
                resume_from_step: None,
            };
        }

        // Check source hash
        if current_source_hash != prev.source_hash {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::SourceChanged,
                resume_from_step: None,
            };
        }

        // Check tool version
        if tool_name != prev.tool_name || tool_version != prev.tool_version {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::ToolChanged,
                resume_from_step: None,
            };
        }

        // Check params
        if params.compute_hash() != prev.global_params.compute_hash() {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::ParamsChanged,
                resume_from_step: None,
            };
        }

        // Check config
        if config.compute_hash() != prev.cook_config.compute_hash() {
            return Self {
                needs_rebuild: true,
                reason: RebuildReason::ConfigChanged,
                resume_from_step: None,
            };
        }

        // Check dependencies
        let prev_deps: HashMap<_, _> = prev.dependencies.iter()
            .map(|d| (&d.asset_guid, &d.content_hash))
            .collect();

        for dep in current_deps {
            match prev_deps.get(&dep.asset_guid) {
                Some(&prev_hash) if *prev_hash != dep.content_hash => {
                    return Self {
                        needs_rebuild: true,
                        reason: RebuildReason::DependencyChanged(dep.asset_guid.clone()),
                        resume_from_step: None,
                    };
                }
                None => {
                    return Self {
                        needs_rebuild: true,
                        reason: RebuildReason::DependencyChanged(dep.asset_guid.clone()),
                        resume_from_step: None,
                    };
                }
                _ => {}
            }
        }

        // No rebuild needed
        Self {
            needs_rebuild: false,
            reason: RebuildReason::SourceUnchanged,
            resume_from_step: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Timestamp utilities tests
    // ========================================================================

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp(0);
        assert!(ts.starts_with("1970-01-01"));
        assert!(ts.ends_with("Z"));
    }

    #[test]
    fn test_format_timestamp_recent() {
        let ts = current_timestamp_ms();
        let formatted = format_timestamp(ts);
        assert!(formatted.contains("T"));
        assert!(formatted.ends_with("Z"));
    }

    #[test]
    fn test_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2023));
        assert!(!is_leap_year(1900));
    }

    // ========================================================================
    // ProcessingParam tests
    // ========================================================================

    #[test]
    fn test_processing_param_new() {
        let param = ProcessingParam::new("quality", "high");
        assert_eq!(param.name, "quality");
        assert_eq!(param.value, "high");
    }

    #[test]
    fn test_processing_param_from_bool() {
        let param = ProcessingParam::from_bool("enabled", true);
        assert_eq!(param.as_bool(), Some(true));

        let param2 = ProcessingParam::from_bool("disabled", false);
        assert_eq!(param2.as_bool(), Some(false));
    }

    #[test]
    fn test_processing_param_from_int() {
        let param = ProcessingParam::from_int("count", 42);
        assert_eq!(param.as_int(), Some(42));
    }

    #[test]
    fn test_processing_param_from_float() {
        let param = ProcessingParam::from_float("scale", 1.5);
        let value = param.as_float().unwrap();
        assert!((value - 1.5).abs() < 0.001);
    }

    // ========================================================================
    // ProcessingParams tests
    // ========================================================================

    #[test]
    fn test_processing_params_empty() {
        let params = ProcessingParams::new();
        assert!(params.is_empty());
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_processing_params_add() {
        let mut params = ProcessingParams::new();
        params.add_str("key1", "value1");
        params.add_int("key2", 123);

        assert_eq!(params.len(), 2);
        assert_eq!(params.get_value("key1"), Some("value1"));
        assert_eq!(params.get("key2").unwrap().as_int(), Some(123));
    }

    #[test]
    fn test_processing_params_replace() {
        let mut params = ProcessingParams::new();
        params.add_str("key", "old");
        params.add_str("key", "new");

        assert_eq!(params.len(), 1);
        assert_eq!(params.get_value("key"), Some("new"));
    }

    #[test]
    fn test_processing_params_hash_deterministic() {
        let mut params1 = ProcessingParams::new();
        params1.add_str("b", "2");
        params1.add_str("a", "1");

        let mut params2 = ProcessingParams::new();
        params2.add_str("a", "1");
        params2.add_str("b", "2");

        // Should produce same hash regardless of insertion order
        assert_eq!(params1.compute_hash(), params2.compute_hash());
    }

    #[test]
    fn test_processing_params_serialization() {
        let mut params = ProcessingParams::new();
        params.add_str("name", "test");
        params.add_int("value", 42);

        let bytes = params.to_bytes();
        let restored = ProcessingParams::from_bytes(&bytes).unwrap();

        assert_eq!(params, restored);
    }

    // ========================================================================
    // ProcessingStep tests
    // ========================================================================

    #[test]
    fn test_processing_step_new() {
        let input = ContentHash::from_bytes(b"input");
        let output = ContentHash::from_bytes(b"output");

        let step = ProcessingStep::new("decode", input, output);

        assert_eq!(step.name, "decode");
        assert_eq!(step.input_hash, input);
        assert_eq!(step.output_hash, output);
        assert!(step.is_success());
    }

    #[test]
    fn test_processing_step_with_error() {
        let input = ContentHash::from_bytes(b"input");
        let output = ContentHash::zero();

        let step = ProcessingStep::new("decode", input, output)
            .with_error("Failed to decode");

        assert!(!step.is_success());
        assert_eq!(step.error, Some("Failed to decode".to_string()));
    }

    #[test]
    fn test_processing_step_serialization() {
        let input = ContentHash::from_bytes(b"input");
        let output = ContentHash::from_bytes(b"output");

        let mut params = ProcessingParams::new();
        params.add_str("quality", "high");

        let step = ProcessingStep::new("encode", input, output)
            .with_params(params)
            .with_duration(100);

        let bytes = step.to_bytes();
        let restored = ProcessingStep::from_bytes(&bytes).unwrap();

        assert_eq!(step, restored);
    }

    #[test]
    fn test_processing_step_config_hash() {
        let input = ContentHash::from_bytes(b"input");
        let output = ContentHash::from_bytes(b"output");

        let step1 = ProcessingStep::new("encode", input, output);
        let step2 = ProcessingStep::new("encode", input, output);

        assert_eq!(step1.config_hash(), step2.config_hash());
    }

    // ========================================================================
    // Platform and QualityLevel tests
    // ========================================================================

    #[test]
    fn test_platform_roundtrip() {
        for platform in [
            Platform::Windows,
            Platform::Linux,
            Platform::MacOS,
            Platform::IOS,
            Platform::Android,
            Platform::Web,
            Platform::Generic,
        ] {
            assert_eq!(Platform::from_byte(platform.to_byte()), platform);
            assert_eq!(Platform::from_name(platform.name()), Some(platform));
        }
    }

    #[test]
    fn test_quality_level_roundtrip() {
        for quality in [
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ] {
            assert_eq!(QualityLevel::from_byte(quality.to_byte()), quality);
            assert_eq!(QualityLevel::from_name(quality.name()), Some(quality));
        }
    }

    // ========================================================================
    // CookConfig tests
    // ========================================================================

    #[test]
    fn test_cook_config_default() {
        let config = CookConfig::default();
        assert_eq!(config.platform, Platform::Generic);
        assert_eq!(config.quality, QualityLevel::Medium);
        assert!(!config.debug);
    }

    #[test]
    fn test_cook_config_for_platform() {
        let config = CookConfig::for_platform(Platform::Windows, QualityLevel::High);
        assert_eq!(config.platform, Platform::Windows);
        assert_eq!(config.quality, QualityLevel::High);
    }

    #[test]
    fn test_cook_config_serialization() {
        let config = CookConfig::for_platform(Platform::Linux, QualityLevel::Ultra)
            .with_param("compression", "zstd");

        let bytes = config.to_bytes();
        let restored = CookConfig::from_bytes(&bytes).unwrap();

        assert_eq!(config, restored);
    }

    #[test]
    fn test_cook_config_hash() {
        let config1 = CookConfig::for_platform(Platform::Windows, QualityLevel::High);
        let config2 = CookConfig::for_platform(Platform::Windows, QualityLevel::High);
        let config3 = CookConfig::for_platform(Platform::Linux, QualityLevel::High);

        assert_eq!(config1.compute_hash(), config2.compute_hash());
        assert_ne!(config1.compute_hash(), config3.compute_hash());
    }

    // ========================================================================
    // DependencyRef tests
    // ========================================================================

    #[test]
    fn test_dependency_ref_new() {
        let hash = ContentHash::from_bytes(b"texture");
        let dep = DependencyRef::new("texture_001", hash, DependencyType::Direct);

        assert_eq!(dep.asset_guid, "texture_001");
        assert_eq!(dep.content_hash, hash);
        assert_eq!(dep.dep_type, DependencyType::Direct);
    }

    #[test]
    fn test_dependency_ref_direct() {
        let hash = ContentHash::from_bytes(b"mesh");
        let dep = DependencyRef::direct("mesh_001", hash);

        assert_eq!(dep.dep_type, DependencyType::Direct);
    }

    #[test]
    fn test_dependency_ref_serialization() {
        let hash = ContentHash::from_bytes(b"shader");
        let dep = DependencyRef::new("shader_001", hash, DependencyType::Indirect);

        let bytes = dep.to_bytes();
        let restored = DependencyRef::from_bytes(&bytes).unwrap();

        assert_eq!(dep, restored);
    }

    // ========================================================================
    // ProvenanceChain tests
    // ========================================================================

    #[test]
    fn test_provenance_chain_new() {
        let source = ContentHash::from_bytes(b"source");
        let chain = ProvenanceChain::new(source, "texture_importer", "1.0.0");

        assert_eq!(chain.source_hash, source);
        assert_eq!(chain.tool_name, "texture_importer");
        assert_eq!(chain.tool_version, "1.0.0");
        assert!(!chain.success);
    }

    #[test]
    fn test_provenance_chain_add_steps() {
        let source = ContentHash::from_bytes(b"source");
        let decoded = ContentHash::from_bytes(b"decoded");
        let compressed = ContentHash::from_bytes(b"compressed");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("decode", source, decoded));
        chain.add_step(ProcessingStep::new("compress", decoded, compressed));

        assert_eq!(chain.steps.len(), 2);
    }

    #[test]
    fn test_provenance_chain_finalize() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("process", source, output));
        chain.finalize(output);

        assert!(chain.is_success());
        assert_eq!(chain.final_output_hash, Some(output));
    }

    #[test]
    fn test_provenance_chain_fail() {
        let source = ContentHash::from_bytes(b"source");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.fail("Processing error");

        assert!(!chain.is_success());
        assert_eq!(chain.error_message, Some("Processing error".to_string()));
    }

    #[test]
    fn test_provenance_chain_verify_chain_valid() {
        let source = ContentHash::from_bytes(b"source");
        let step1_out = ContentHash::from_bytes(b"step1");
        let step2_out = ContentHash::from_bytes(b"step2");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("step1", source, step1_out));
        chain.add_step(ProcessingStep::new("step2", step1_out, step2_out));
        chain.finalize(step2_out);

        assert!(chain.verify_chain().is_ok());
    }

    #[test]
    fn test_provenance_chain_verify_chain_invalid() {
        let source = ContentHash::from_bytes(b"source");
        let step1_out = ContentHash::from_bytes(b"step1");
        let wrong_input = ContentHash::from_bytes(b"wrong");
        let step2_out = ContentHash::from_bytes(b"step2");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("step1", source, step1_out));
        chain.add_step(ProcessingStep::new("step2", wrong_input, step2_out));

        assert!(chain.verify_chain().is_err());
    }

    #[test]
    fn test_provenance_chain_serialization() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0")
            .with_timestamp(1234567890);
        chain.global_params.add_str("quality", "high");
        chain.set_cook_config(CookConfig::for_platform(Platform::Windows, QualityLevel::High));
        chain.add_step(ProcessingStep::new("process", source, output).with_duration(50));
        chain.add_dependency(DependencyRef::direct("dep1", ContentHash::from_bytes(b"dep")));
        chain.finalize(output);

        let bytes = chain.to_bytes();
        let restored = ProvenanceChain::from_bytes(&bytes).unwrap();

        assert_eq!(chain, restored);
    }

    #[test]
    fn test_provenance_chain_needs_rebuild_source_changed() {
        let source1 = ContentHash::from_bytes(b"source1");
        let source2 = ContentHash::from_bytes(b"source2");
        let output = ContentHash::from_bytes(b"output");

        let mut chain1 = ProvenanceChain::new(source1, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("process", source1, output));
        chain1.finalize(output);

        let mut chain2 = ProvenanceChain::new(source2, "importer", "1.0.0");
        chain2.add_step(ProcessingStep::new("process", source2, output));
        chain2.finalize(output);

        assert!(chain2.needs_rebuild(&chain1));
    }

    #[test]
    fn test_provenance_chain_needs_rebuild_same() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");

        let mut chain1 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("process", source, output));
        chain1.finalize(output);

        let mut chain2 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain2.add_step(ProcessingStep::new("process", source, output));
        chain2.finalize(output);

        assert!(!chain2.needs_rebuild(&chain1));
    }

    #[test]
    fn test_provenance_chain_find_divergence() {
        let source = ContentHash::from_bytes(b"source");
        let step1 = ContentHash::from_bytes(b"step1");
        let step2a = ContentHash::from_bytes(b"step2a");
        let step2b = ContentHash::from_bytes(b"step2b");

        let mut chain1 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("step1", source, step1));
        chain1.add_step(ProcessingStep::new("step2", step1, step2a));

        let mut chain2 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain2.add_step(ProcessingStep::new("step1", source, step1));
        chain2.add_step(ProcessingStep::new("step2", step1, step2b));

        let divergence = chain1.find_divergence(&chain2);
        assert_eq!(divergence, Some((2, "step2")));
    }

    #[test]
    fn test_provenance_chain_total_duration() {
        let source = ContentHash::from_bytes(b"source");
        let step1 = ContentHash::from_bytes(b"step1");
        let step2 = ContentHash::from_bytes(b"step2");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("step1", source, step1).with_duration(100));
        chain.add_step(ProcessingStep::new("step2", step1, step2).with_duration(200));

        assert_eq!(chain.total_duration_ms(), 300);
    }

    // ========================================================================
    // ProvenanceTreeNode tests
    // ========================================================================

    #[test]
    fn test_provenance_tree_node_blob() {
        let node = ProvenanceTreeNode::blob("test.bin", vec![1, 2, 3]);

        assert!(node.is_blob());
        assert!(!node.is_directory());
        assert_eq!(node.name(), "test.bin");
    }

    #[test]
    fn test_provenance_tree_node_directory() {
        let child = ProvenanceTreeNode::blob("child.bin", vec![1, 2, 3]);
        let node = ProvenanceTreeNode::directory("parent", vec![child]);

        assert!(node.is_directory());
        assert!(!node.is_blob());
        assert_eq!(node.name(), "parent");
    }

    #[test]
    fn test_provenance_tree_hash() {
        let node1 = ProvenanceTreeNode::blob("test.bin", vec![1, 2, 3]);
        let node2 = ProvenanceTreeNode::blob("test.bin", vec![1, 2, 3]);
        let node3 = ProvenanceTreeNode::blob("test.bin", vec![1, 2, 4]);

        assert_eq!(node1.compute_hash(), node2.compute_hash());
        assert_ne!(node1.compute_hash(), node3.compute_hash());
    }

    #[test]
    fn test_chain_to_tree() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");

        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");
        chain.add_step(ProcessingStep::new("process", source, output));
        chain.finalize(output);

        let tree = chain_to_tree(&chain, "asset_001");

        assert!(tree.is_directory());
        assert_eq!(tree.name(), "provenance/asset_001");
    }

    // ========================================================================
    // ProvenanceDiff tests
    // ========================================================================

    #[test]
    fn test_provenance_diff_empty() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");

        let mut chain1 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("process", source, output));
        chain1.finalize(output);

        let chain2 = chain1.clone();

        let diff = diff_provenance(&chain1, &chain2);
        assert!(diff.is_empty());
        assert!(!diff.requires_rebuild());
    }

    #[test]
    fn test_provenance_diff_source_changed() {
        let source1 = ContentHash::from_bytes(b"source1");
        let source2 = ContentHash::from_bytes(b"source2");
        let output = ContentHash::from_bytes(b"output");

        let mut chain1 = ProvenanceChain::new(source1, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("process", source1, output));
        chain1.finalize(output);

        let mut chain2 = ProvenanceChain::new(source2, "importer", "1.0.0");
        chain2.add_step(ProcessingStep::new("process", source2, output));
        chain2.finalize(output);

        let diff = diff_provenance(&chain1, &chain2);
        assert!(diff.source_changed);
        assert!(diff.requires_rebuild());
    }

    #[test]
    fn test_provenance_diff_step_added() {
        let source = ContentHash::from_bytes(b"source");
        let step1 = ContentHash::from_bytes(b"step1");
        let step2 = ContentHash::from_bytes(b"step2");

        let mut chain1 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain1.add_step(ProcessingStep::new("step1", source, step1));
        chain1.finalize(step1);

        let mut chain2 = ProvenanceChain::new(source, "importer", "1.0.0");
        chain2.add_step(ProcessingStep::new("step1", source, step1));
        chain2.add_step(ProcessingStep::new("step2", step1, step2));
        chain2.finalize(step2);

        let diff = diff_provenance(&chain1, &chain2);
        assert!(diff.steps_added.contains(&"step2".to_string()));
    }

    // ========================================================================
    // RebuildDecision tests
    // ========================================================================

    #[test]
    fn test_rebuild_decision_no_previous() {
        let source = ContentHash::from_bytes(b"source");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let decision = RebuildDecision::compute(
            source,
            &[],
            None,
            "importer",
            "1.0.0",
            &params,
            &config,
        );

        assert!(decision.needs_rebuild);
        assert_eq!(decision.reason, RebuildReason::NoPreviousProvenance);
    }

    #[test]
    fn test_rebuild_decision_source_unchanged() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let mut prev = ProvenanceChain::new(source, "importer", "1.0.0");
        prev.add_step(ProcessingStep::new("process", source, output));
        prev.finalize(output);

        let decision = RebuildDecision::compute(
            source,
            &[],
            Some(&prev),
            "importer",
            "1.0.0",
            &params,
            &config,
        );

        assert!(!decision.needs_rebuild);
        assert_eq!(decision.reason, RebuildReason::SourceUnchanged);
    }

    #[test]
    fn test_rebuild_decision_source_changed() {
        let source1 = ContentHash::from_bytes(b"source1");
        let source2 = ContentHash::from_bytes(b"source2");
        let output = ContentHash::from_bytes(b"output");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let mut prev = ProvenanceChain::new(source1, "importer", "1.0.0");
        prev.add_step(ProcessingStep::new("process", source1, output));
        prev.finalize(output);

        let decision = RebuildDecision::compute(
            source2,
            &[],
            Some(&prev),
            "importer",
            "1.0.0",
            &params,
            &config,
        );

        assert!(decision.needs_rebuild);
        assert_eq!(decision.reason, RebuildReason::SourceChanged);
    }

    #[test]
    fn test_rebuild_decision_tool_changed() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let mut prev = ProvenanceChain::new(source, "importer", "1.0.0");
        prev.add_step(ProcessingStep::new("process", source, output));
        prev.finalize(output);

        let decision = RebuildDecision::compute(
            source,
            &[],
            Some(&prev),
            "importer",
            "2.0.0", // Version changed
            &params,
            &config,
        );

        assert!(decision.needs_rebuild);
        assert_eq!(decision.reason, RebuildReason::ToolChanged);
    }

    #[test]
    fn test_rebuild_decision_dependency_changed() {
        let source = ContentHash::from_bytes(b"source");
        let output = ContentHash::from_bytes(b"output");
        let dep_hash1 = ContentHash::from_bytes(b"dep1");
        let dep_hash2 = ContentHash::from_bytes(b"dep2");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let mut prev = ProvenanceChain::new(source, "importer", "1.0.0");
        prev.add_step(ProcessingStep::new("process", source, output));
        prev.add_dependency(DependencyRef::direct("texture_001", dep_hash1));
        prev.finalize(output);

        let current_deps = vec![DependencyRef::direct("texture_001", dep_hash2)];

        let decision = RebuildDecision::compute(
            source,
            &current_deps,
            Some(&prev),
            "importer",
            "1.0.0",
            &params,
            &config,
        );

        assert!(decision.needs_rebuild);
        assert!(matches!(decision.reason, RebuildReason::DependencyChanged(_)));
    }

    #[test]
    fn test_rebuild_decision_previous_failed() {
        let source = ContentHash::from_bytes(b"source");
        let params = ProcessingParams::new();
        let config = CookConfig::default();

        let mut prev = ProvenanceChain::new(source, "importer", "1.0.0");
        prev.fail("Previous failure");

        let decision = RebuildDecision::compute(
            source,
            &[],
            Some(&prev),
            "importer",
            "1.0.0",
            &params,
            &config,
        );

        assert!(decision.needs_rebuild);
        assert_eq!(decision.reason, RebuildReason::PreviousFailed);
    }

    // ========================================================================
    // Sentinel and version tests
    // ========================================================================

    #[test]
    fn test_provenance_sentinel() {
        // "PROV" = 0x50 0x52 0x4F 0x56 in ASCII
        assert_eq!(PROVENANCE_SENTINEL, 0x50524F56);
    }

    #[test]
    fn test_provenance_version() {
        assert_eq!(PROVENANCE_VERSION, 1);
    }

    #[test]
    fn test_provenance_invalid_sentinel() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // Wrong sentinel

        let result = ProvenanceChain::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_provenance_invalid_version() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&PROVENANCE_SENTINEL.to_le_bytes());
        data[4..8].copy_from_slice(&99u32.to_le_bytes()); // Wrong version

        let result = ProvenanceChain::from_bytes(&data);
        assert!(result.is_err());
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_empty_provenance_chain() {
        let source = ContentHash::from_bytes(b"source");
        let chain = ProvenanceChain::new(source, "importer", "1.0.0");

        // Empty chain (no steps) should still serialize/deserialize
        let bytes = chain.to_bytes();
        let restored = ProvenanceChain::from_bytes(&bytes).unwrap();

        assert_eq!(chain.source_hash, restored.source_hash);
        assert!(restored.steps.is_empty());
    }

    #[test]
    fn test_provenance_chain_many_steps() {
        let source = ContentHash::from_bytes(b"source");
        let mut chain = ProvenanceChain::new(source, "importer", "1.0.0");

        let mut prev_hash = source;
        for i in 0..100 {
            let next_hash = ContentHash::from_bytes(format!("step_{}", i).as_bytes());
            chain.add_step(ProcessingStep::new(format!("step_{}", i), prev_hash, next_hash));
            prev_hash = next_hash;
        }
        chain.finalize(prev_hash);

        let bytes = chain.to_bytes();
        let restored = ProvenanceChain::from_bytes(&bytes).unwrap();

        assert_eq!(chain.steps.len(), 100);
        assert_eq!(restored.steps.len(), 100);
    }

    #[test]
    fn test_provenance_chain_unicode() {
        let source = ContentHash::from_bytes(b"source");

        let mut chain = ProvenanceChain::new(source, "importer_unicode", "1.0.0");
        chain.global_params.add_str("description", "Test with unicode: ");
        chain.fail("Error: Failed to process file");

        let bytes = chain.to_bytes();
        let restored = ProvenanceChain::from_bytes(&bytes).unwrap();

        assert!(restored.global_params.get_value("description").unwrap().contains("unicode"));
    }
}
