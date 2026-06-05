//! Python bindings for GPU error handling and propagation (T-WGPU-P7.6.9).
//!
//! This module provides Python-accessible error types and handlers for GPU operations,
//! enabling Python code to catch, inspect, and handle GPU errors gracefully.
//!
//! # Feature Gate
//!
//! All types are gated behind the `pyo3` feature flag:
//!
//! ```toml
//! [features]
//! pyo3 = ["dep:pyo3"]
//! ```
//!
//! # Example (Python)
//!
//! ```python
//! from trinity_renderer.bindings import (
//!     PyGpuError, PyErrorCategory, PyResult, PyErrorHandler, PyValidationReport
//! )
//!
//! # Check error properties
//! error = PyGpuError.out_of_memory(1024 * 1024, 512 * 1024)
//! print(f"Error: {error}")
//! print(f"Recoverable: {error.is_recoverable()}")
//! print(f"Category: {error.category()}")
//! print(f"Code: {error.error_code()}")
//!
//! # Use error handler with callback
//! handler = PyErrorHandler()
//! handler.set_callback(lambda e: print(f"GPU Error: {e}"))
//! handler.handle(error)
//! print(f"Has errors: {handler.has_errors()}")
//!
//! # Create validation reports
//! report = PyValidationReport()
//! report.add_error(PyGpuError.validation("Invalid buffer size"))
//! report.add_warning("Consider using power-of-two sizes")
//! print(report.format_report())
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyMemoryError};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// PyErrorCategory
// ---------------------------------------------------------------------------

/// Error category enumeration for classifying GPU errors.
///
/// Categories help organize errors by their source and nature,
/// enabling category-based error handling strategies.
#[pyclass(name = "ErrorCategory")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyErrorCategory {
    /// Input validation failures.
    Validation,
    /// Memory allocation or access errors.
    Memory,
    /// Device-level errors (lost, disconnected).
    Device,
    /// Shader compilation or linking errors.
    Shader,
    /// Pipeline creation or configuration errors.
    Pipeline,
    /// Resource creation or management errors.
    Resource,
    /// Unsupported feature or capability.
    Feature,
    /// Hardware or API limit exceeded.
    Limit,
}

#[pymethods]
impl PyErrorCategory {
    /// Returns the canonical name of this error category.
    pub fn name(&self) -> &str {
        match self {
            Self::Validation => "Validation",
            Self::Memory => "Memory",
            Self::Device => "Device",
            Self::Shader => "Shader",
            Self::Pipeline => "Pipeline",
            Self::Resource => "Resource",
            Self::Feature => "Feature",
            Self::Limit => "Limit",
        }
    }

    /// Returns a description of this error category.
    pub fn description(&self) -> &str {
        match self {
            Self::Validation => "Input validation failure",
            Self::Memory => "Memory allocation or access error",
            Self::Device => "GPU device error",
            Self::Shader => "Shader compilation error",
            Self::Pipeline => "Pipeline creation error",
            Self::Resource => "Resource management error",
            Self::Feature => "Unsupported feature",
            Self::Limit => "Hardware limit exceeded",
        }
    }

    /// Returns true if errors in this category are typically recoverable.
    pub fn typically_recoverable(&self) -> bool {
        match self {
            Self::Validation => true,  // Can fix inputs
            Self::Memory => true,      // Can free memory and retry
            Self::Device => false,     // Device lost is usually fatal
            Self::Shader => true,      // Can fix shader and retry
            Self::Pipeline => true,    // Can reconfigure pipeline
            Self::Resource => true,    // Can release resources
            Self::Feature => false,    // Hardware doesn't support feature
            Self::Limit => true,       // Can reduce resource usage
        }
    }

    fn __repr__(&self) -> String {
        format!("ErrorCategory.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __hash__(&self) -> u64 {
        *self as u64
    }

    fn __eq__(&self, other: &Self) -> bool {
        *self == *other
    }

    fn __ne__(&self, other: &Self) -> bool {
        *self != *other
    }
}

impl Default for PyErrorCategory {
    fn default() -> Self {
        Self::Validation
    }
}

// ---------------------------------------------------------------------------
// PyGpuError
// ---------------------------------------------------------------------------

/// GPU error type with detailed information.
///
/// Represents various GPU-related errors with contextual information
/// for debugging and error recovery.
#[pyclass(name = "GpuError")]
#[derive(Clone, Debug)]
pub struct PyGpuError {
    variant: GpuErrorVariant,
}

/// Internal error variant representation.
#[derive(Clone, Debug)]
enum GpuErrorVariant {
    ValidationError { message: String },
    OutOfMemory { requested: u64, available: u64 },
    DeviceLost { reason: String },
    ShaderCompilation { shader: String, errors: Vec<String> },
    PipelineCreation { pipeline: String, message: String },
    ResourceCreation { resource_type: String, message: String },
    FeatureNotSupported { feature: String },
    LimitExceeded { limit: String, requested: u64, max: u64 },
}

#[pymethods]
impl PyGpuError {
    // --- Constructors for each error variant ---

    /// Creates a validation error.
    ///
    /// # Arguments
    /// * `message` - Description of the validation failure
    #[staticmethod]
    pub fn validation(message: String) -> Self {
        Self {
            variant: GpuErrorVariant::ValidationError { message },
        }
    }

    /// Creates an out-of-memory error.
    ///
    /// # Arguments
    /// * `requested` - Bytes requested for allocation
    /// * `available` - Bytes actually available
    #[staticmethod]
    pub fn out_of_memory(requested: u64, available: u64) -> Self {
        Self {
            variant: GpuErrorVariant::OutOfMemory { requested, available },
        }
    }

    /// Creates a device lost error.
    ///
    /// # Arguments
    /// * `reason` - Reason for device loss (e.g., "timeout", "driver crash")
    #[staticmethod]
    pub fn device_lost(reason: String) -> Self {
        Self {
            variant: GpuErrorVariant::DeviceLost { reason },
        }
    }

    /// Creates a shader compilation error.
    ///
    /// # Arguments
    /// * `shader` - Name or path of the shader
    /// * `errors` - List of compilation error messages
    #[staticmethod]
    pub fn shader_compilation(shader: String, errors: Vec<String>) -> Self {
        Self {
            variant: GpuErrorVariant::ShaderCompilation { shader, errors },
        }
    }

    /// Creates a pipeline creation error.
    ///
    /// # Arguments
    /// * `pipeline` - Name of the pipeline
    /// * `message` - Error description
    #[staticmethod]
    pub fn pipeline_creation(pipeline: String, message: String) -> Self {
        Self {
            variant: GpuErrorVariant::PipelineCreation { pipeline, message },
        }
    }

    /// Creates a resource creation error.
    ///
    /// # Arguments
    /// * `resource_type` - Type of resource (e.g., "Buffer", "Texture")
    /// * `message` - Error description
    #[staticmethod]
    pub fn resource_creation(resource_type: String, message: String) -> Self {
        Self {
            variant: GpuErrorVariant::ResourceCreation { resource_type, message },
        }
    }

    /// Creates a feature not supported error.
    ///
    /// # Arguments
    /// * `feature` - Name of the unsupported feature
    #[staticmethod]
    pub fn feature_not_supported(feature: String) -> Self {
        Self {
            variant: GpuErrorVariant::FeatureNotSupported { feature },
        }
    }

    /// Creates a limit exceeded error.
    ///
    /// # Arguments
    /// * `limit` - Name of the limit (e.g., "maxTextureSize")
    /// * `requested` - Value that was requested
    /// * `max` - Maximum allowed value
    #[staticmethod]
    pub fn limit_exceeded(limit: String, requested: u64, max: u64) -> Self {
        Self {
            variant: GpuErrorVariant::LimitExceeded { limit, requested, max },
        }
    }

    /// Returns true if this error is potentially recoverable.
    ///
    /// Non-recoverable errors include device loss and unsupported features.
    pub fn is_recoverable(&self) -> bool {
        match &self.variant {
            GpuErrorVariant::ValidationError { .. } => true,
            GpuErrorVariant::OutOfMemory { .. } => true,
            GpuErrorVariant::DeviceLost { .. } => false,
            GpuErrorVariant::ShaderCompilation { .. } => true,
            GpuErrorVariant::PipelineCreation { .. } => true,
            GpuErrorVariant::ResourceCreation { .. } => true,
            GpuErrorVariant::FeatureNotSupported { .. } => false,
            GpuErrorVariant::LimitExceeded { .. } => true,
        }
    }

    /// Returns a unique error code for this error type.
    ///
    /// Error codes are stable and can be used for programmatic error handling.
    pub fn error_code(&self) -> u32 {
        match &self.variant {
            GpuErrorVariant::ValidationError { .. } => 1000,
            GpuErrorVariant::OutOfMemory { .. } => 2000,
            GpuErrorVariant::DeviceLost { .. } => 3000,
            GpuErrorVariant::ShaderCompilation { .. } => 4000,
            GpuErrorVariant::PipelineCreation { .. } => 5000,
            GpuErrorVariant::ResourceCreation { .. } => 6000,
            GpuErrorVariant::FeatureNotSupported { .. } => 7000,
            GpuErrorVariant::LimitExceeded { .. } => 8000,
        }
    }

    /// Returns the error category for this error.
    pub fn category(&self) -> PyErrorCategory {
        match &self.variant {
            GpuErrorVariant::ValidationError { .. } => PyErrorCategory::Validation,
            GpuErrorVariant::OutOfMemory { .. } => PyErrorCategory::Memory,
            GpuErrorVariant::DeviceLost { .. } => PyErrorCategory::Device,
            GpuErrorVariant::ShaderCompilation { .. } => PyErrorCategory::Shader,
            GpuErrorVariant::PipelineCreation { .. } => PyErrorCategory::Pipeline,
            GpuErrorVariant::ResourceCreation { .. } => PyErrorCategory::Resource,
            GpuErrorVariant::FeatureNotSupported { .. } => PyErrorCategory::Feature,
            GpuErrorVariant::LimitExceeded { .. } => PyErrorCategory::Limit,
        }
    }

    /// Returns the variant name of this error.
    pub fn variant_name(&self) -> &str {
        match &self.variant {
            GpuErrorVariant::ValidationError { .. } => "ValidationError",
            GpuErrorVariant::OutOfMemory { .. } => "OutOfMemory",
            GpuErrorVariant::DeviceLost { .. } => "DeviceLost",
            GpuErrorVariant::ShaderCompilation { .. } => "ShaderCompilation",
            GpuErrorVariant::PipelineCreation { .. } => "PipelineCreation",
            GpuErrorVariant::ResourceCreation { .. } => "ResourceCreation",
            GpuErrorVariant::FeatureNotSupported { .. } => "FeatureNotSupported",
            GpuErrorVariant::LimitExceeded { .. } => "LimitExceeded",
        }
    }

    /// Returns the error message.
    pub fn message(&self) -> String {
        match &self.variant {
            GpuErrorVariant::ValidationError { message } => message.clone(),
            GpuErrorVariant::OutOfMemory { requested, available } => {
                format!("Out of memory: requested {} bytes, {} available", requested, available)
            }
            GpuErrorVariant::DeviceLost { reason } => {
                format!("Device lost: {}", reason)
            }
            GpuErrorVariant::ShaderCompilation { shader, errors } => {
                format!("Shader '{}' compilation failed: {}", shader, errors.join("; "))
            }
            GpuErrorVariant::PipelineCreation { pipeline, message } => {
                format!("Pipeline '{}' creation failed: {}", pipeline, message)
            }
            GpuErrorVariant::ResourceCreation { resource_type, message } => {
                format!("{} creation failed: {}", resource_type, message)
            }
            GpuErrorVariant::FeatureNotSupported { feature } => {
                format!("Feature '{}' not supported", feature)
            }
            GpuErrorVariant::LimitExceeded { limit, requested, max } => {
                format!("Limit '{}' exceeded: requested {}, max {}", limit, requested, max)
            }
        }
    }

    /// Returns detailed information for specific error types.
    ///
    /// For OutOfMemory, returns (requested, available).
    /// For LimitExceeded, returns (limit, requested, max).
    /// For ShaderCompilation, returns list of errors.
    /// For other types, returns None.
    pub fn details(&self, py: Python<'_>) -> PyObject {
        match &self.variant {
            GpuErrorVariant::OutOfMemory { requested, available } => {
                (requested, available).into_pyobject(py).unwrap().into_any().unbind()
            }
            GpuErrorVariant::LimitExceeded { limit, requested, max } => {
                (limit.clone(), requested, max).into_pyobject(py).unwrap().into_any().unbind()
            }
            GpuErrorVariant::ShaderCompilation { shader, errors } => {
                (shader.clone(), errors.clone()).into_pyobject(py).unwrap().into_any().unbind()
            }
            GpuErrorVariant::PipelineCreation { pipeline, message } => {
                (pipeline.clone(), message.clone()).into_pyobject(py).unwrap().into_any().unbind()
            }
            GpuErrorVariant::ResourceCreation { resource_type, message } => {
                (resource_type.clone(), message.clone()).into_pyobject(py).unwrap().into_any().unbind()
            }
            _ => py.None(),
        }
    }

    fn __repr__(&self) -> String {
        format!("GpuError.{}({})", self.variant_name(), self.message())
    }

    fn __str__(&self) -> String {
        self.message()
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.error_code().hash(&mut hasher);
        self.message().hash(&mut hasher);
        hasher.finish()
    }
}

impl PyGpuError {
    /// Converts this error to a PyErr for raising in Python.
    pub fn to_py_err(&self) -> PyErr {
        match &self.variant {
            GpuErrorVariant::OutOfMemory { .. } => {
                PyMemoryError::new_err(self.message())
            }
            GpuErrorVariant::ValidationError { .. } => {
                PyValueError::new_err(self.message())
            }
            _ => {
                PyRuntimeError::new_err(self.message())
            }
        }
    }

    /// Creates a PyGpuError from a Python exception.
    pub fn from_py_err(err: &PyErr, py: Python<'_>) -> Self {
        let message = err.value(py).to_string();

        if err.is_instance_of::<PyMemoryError>(py) {
            Self::out_of_memory(0, 0)
        } else if err.is_instance_of::<PyValueError>(py) {
            Self::validation(message)
        } else {
            Self::validation(message)
        }
    }
}

// ---------------------------------------------------------------------------
// PyResult<T> - Result wrapper
// ---------------------------------------------------------------------------

/// Python-exposed result type wrapping a value or GPU error.
///
/// Provides Rust-like Result semantics in Python for GPU operations.
#[pyclass(name = "GpuResult")]
#[derive(Clone, Debug)]
pub struct PyGpuResult {
    value: Option<PyObject>,
    error: Option<PyGpuError>,
}

#[pymethods]
impl PyGpuResult {
    /// Creates a successful result with the given value.
    #[staticmethod]
    pub fn ok(py: Python<'_>, value: PyObject) -> Self {
        let _ = py; // Unused but keeps API consistent
        Self {
            value: Some(value),
            error: None,
        }
    }

    /// Creates a failed result with the given error.
    #[staticmethod]
    pub fn err(error: PyGpuError) -> Self {
        Self {
            value: None,
            error: Some(error),
        }
    }

    /// Returns the success value, or None if this is an error.
    pub fn ok_value(&self, py: Python<'_>) -> PyObject {
        match &self.value {
            Some(v) => v.clone_ref(py),
            None => py.None(),
        }
    }

    /// Returns the error, or None if this is a success.
    pub fn err_value(&self) -> Option<PyGpuError> {
        self.error.clone()
    }

    /// Returns true if this result is successful.
    pub fn is_ok(&self) -> bool {
        self.value.is_some()
    }

    /// Returns true if this result is an error.
    pub fn is_err(&self) -> bool {
        self.error.is_some()
    }

    /// Returns the value if successful, otherwise returns the default.
    ///
    /// # Arguments
    /// * `default` - Value to return if this is an error
    pub fn unwrap_or(&self, py: Python<'_>, default: PyObject) -> PyObject {
        match &self.value {
            Some(v) => v.clone_ref(py),
            None => default,
        }
    }

    /// Returns the value if successful, otherwise raises the error.
    ///
    /// # Raises
    /// * `RuntimeError` - If this result contains an error
    pub fn unwrap(&self, py: Python<'_>) -> PyResult<PyObject> {
        match (&self.value, &self.error) {
            (Some(v), _) => Ok(v.clone_ref(py)),
            (None, Some(e)) => Err(e.to_py_err()),
            (None, None) => Err(PyRuntimeError::new_err("Result contains neither value nor error")),
        }
    }

    /// Applies a function to the success value.
    ///
    /// If this is an error, returns the error unchanged.
    ///
    /// # Arguments
    /// * `func` - Function to apply to the success value
    pub fn map(&self, py: Python<'_>, func: PyObject) -> PyResult<Self> {
        match (&self.value, &self.error) {
            (Some(v), _) => {
                let result = func.call1(py, (v.clone_ref(py),))?;
                Ok(Self {
                    value: Some(result),
                    error: None,
                })
            }
            (None, Some(e)) => Ok(Self {
                value: None,
                error: Some(e.clone()),
            }),
            (None, None) => Err(PyRuntimeError::new_err("Invalid result state")),
        }
    }

    /// Applies a function that returns a GpuResult to the success value.
    ///
    /// If this is an error, returns the error unchanged.
    /// If the function returns an error, returns that error.
    ///
    /// # Arguments
    /// * `func` - Function returning a GpuResult
    pub fn and_then(&self, py: Python<'_>, func: PyObject) -> PyResult<Self> {
        match (&self.value, &self.error) {
            (Some(v), _) => {
                let result = func.call1(py, (v.clone_ref(py),))?;
                // Try to extract as PyGpuResult
                if let Ok(gpu_result) = result.extract::<PyGpuResult>(py) {
                    Ok(gpu_result)
                } else {
                    // If not a GpuResult, wrap the value
                    Ok(Self {
                        value: Some(result),
                        error: None,
                    })
                }
            }
            (None, Some(e)) => Ok(Self {
                value: None,
                error: Some(e.clone()),
            }),
            (None, None) => Err(PyRuntimeError::new_err("Invalid result state")),
        }
    }

    /// Applies a function to the error value.
    ///
    /// If this is a success, returns the success unchanged.
    ///
    /// # Arguments
    /// * `func` - Function to apply to the error
    pub fn map_err(&self, py: Python<'_>, func: PyObject) -> PyResult<Self> {
        match (&self.value, &self.error) {
            (Some(v), _) => Ok(Self {
                value: Some(v.clone_ref(py)),
                error: None,
            }),
            (None, Some(e)) => {
                let result = func.call1(py, (e.clone(),))?;
                if let Ok(new_error) = result.extract::<PyGpuError>(py) {
                    Ok(Self {
                        value: None,
                        error: Some(new_error),
                    })
                } else {
                    Ok(Self {
                        value: None,
                        error: Some(e.clone()),
                    })
                }
            }
            (None, None) => Err(PyRuntimeError::new_err("Invalid result state")),
        }
    }

    fn __repr__(&self) -> String {
        match (&self.value, &self.error) {
            (Some(_), _) => "GpuResult.Ok(...)".to_string(),
            (None, Some(e)) => format!("GpuResult.Err({})", e.message()),
            (None, None) => "GpuResult.Invalid".to_string(),
        }
    }

    fn __bool__(&self) -> bool {
        self.is_ok()
    }
}

// ---------------------------------------------------------------------------
// PyErrorHandler
// ---------------------------------------------------------------------------

/// Error handler with callback support.
///
/// Collects errors and optionally invokes callbacks when errors occur.
/// Thread-safe for use across multiple GPU operations.
#[pyclass(name = "ErrorHandler")]
#[derive(Clone)]
pub struct PyErrorHandler {
    errors: Arc<Mutex<Vec<PyGpuError>>>,
    callback: Arc<Mutex<Option<PyObject>>>,
}

#[pymethods]
impl PyErrorHandler {
    /// Creates a new error handler.
    #[new]
    pub fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
            callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Handles an error by storing it and invoking the callback if set.
    ///
    /// # Arguments
    /// * `error` - The GPU error to handle
    pub fn handle(&self, py: Python<'_>, error: PyGpuError) -> PyResult<()> {
        // Store the error
        {
            let mut errors = self.errors.lock().unwrap();
            errors.push(error.clone());
        }

        // Invoke callback if set
        let callback = {
            let cb = self.callback.lock().unwrap();
            cb.clone()
        };

        if let Some(cb) = callback {
            cb.call1(py, (error,))?;
        }

        Ok(())
    }

    /// Returns true if any errors have been recorded.
    pub fn has_errors(&self) -> bool {
        let errors = self.errors.lock().unwrap();
        !errors.is_empty()
    }

    /// Returns the number of recorded errors.
    pub fn error_count(&self) -> usize {
        let errors = self.errors.lock().unwrap();
        errors.len()
    }

    /// Returns the most recent error, if any.
    pub fn last_error(&self) -> Option<PyGpuError> {
        let errors = self.errors.lock().unwrap();
        errors.last().cloned()
    }

    /// Returns all recorded errors.
    pub fn errors(&self) -> Vec<PyGpuError> {
        let errors = self.errors.lock().unwrap();
        errors.clone()
    }

    /// Returns all errors of a specific category.
    pub fn errors_by_category(&self, category: PyErrorCategory) -> Vec<PyGpuError> {
        let errors = self.errors.lock().unwrap();
        errors.iter()
            .filter(|e| e.category() == category)
            .cloned()
            .collect()
    }

    /// Returns only recoverable errors.
    pub fn recoverable_errors(&self) -> Vec<PyGpuError> {
        let errors = self.errors.lock().unwrap();
        errors.iter()
            .filter(|e| e.is_recoverable())
            .cloned()
            .collect()
    }

    /// Returns only non-recoverable (fatal) errors.
    pub fn fatal_errors(&self) -> Vec<PyGpuError> {
        let errors = self.errors.lock().unwrap();
        errors.iter()
            .filter(|e| !e.is_recoverable())
            .cloned()
            .collect()
    }

    /// Clears all recorded errors.
    pub fn clear(&self) {
        let mut errors = self.errors.lock().unwrap();
        errors.clear();
    }

    /// Sets the error callback function.
    ///
    /// The callback is invoked with each error when `handle()` is called.
    ///
    /// # Arguments
    /// * `callback` - Function taking a GpuError argument
    pub fn set_callback(&self, callback: PyObject) {
        let mut cb = self.callback.lock().unwrap();
        *cb = Some(callback);
    }

    /// Clears the error callback.
    pub fn clear_callback(&self) {
        let mut cb = self.callback.lock().unwrap();
        *cb = None;
    }

    /// Returns true if a callback is set.
    pub fn has_callback(&self) -> bool {
        let cb = self.callback.lock().unwrap();
        cb.is_some()
    }

    fn __repr__(&self) -> String {
        let count = self.error_count();
        let has_cb = self.has_callback();
        format!("ErrorHandler(errors={}, callback={})", count, has_cb)
    }

    fn __len__(&self) -> usize {
        self.error_count()
    }

    fn __bool__(&self) -> bool {
        !self.has_errors()
    }
}

impl Default for PyErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PyValidationReport
// ---------------------------------------------------------------------------

/// Validation report collecting errors and warnings.
///
/// Used to aggregate multiple validation issues from GPU operations
/// before presenting them to the user.
#[pyclass(name = "ValidationReport")]
#[derive(Clone, Debug)]
pub struct PyValidationReport {
    errors: Vec<PyGpuError>,
    warnings: Vec<String>,
}

#[pymethods]
impl PyValidationReport {
    /// Creates a new empty validation report.
    #[new]
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Creates a validation report with the given errors and warnings.
    #[staticmethod]
    pub fn with_issues(errors: Vec<PyGpuError>, warnings: Vec<String>) -> Self {
        Self { errors, warnings }
    }

    /// Adds an error to the report.
    pub fn add_error(&mut self, error: PyGpuError) {
        self.errors.push(error);
    }

    /// Adds a warning message to the report.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Returns true if there are no errors (warnings are allowed).
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns true if there are no errors or warnings.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }

    /// Returns the number of errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Returns the number of warnings.
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// Returns all errors.
    #[getter]
    pub fn errors(&self) -> Vec<PyGpuError> {
        self.errors.clone()
    }

    /// Returns all warnings.
    #[getter]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Returns errors of a specific category.
    pub fn errors_by_category(&self, category: PyErrorCategory) -> Vec<PyGpuError> {
        self.errors.iter()
            .filter(|e| e.category() == category)
            .cloned()
            .collect()
    }

    /// Formats the report as a human-readable string.
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        if self.is_clean() {
            report.push_str("Validation: PASSED\n");
            return report;
        }

        if self.is_valid() {
            report.push_str("Validation: PASSED (with warnings)\n");
        } else {
            report.push_str("Validation: FAILED\n");
        }

        report.push('\n');

        if !self.errors.is_empty() {
            report.push_str(&format!("Errors ({}):\n", self.errors.len()));
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("  {}. [{}] {}\n",
                    i + 1,
                    error.category().name(),
                    error.message()
                ));
            }
        }

        if !self.warnings.is_empty() {
            if !self.errors.is_empty() {
                report.push('\n');
            }
            report.push_str(&format!("Warnings ({}):\n", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, warning));
            }
        }

        report
    }

    /// Merges another report into this one.
    pub fn merge(&mut self, other: &PyValidationReport) {
        self.errors.extend(other.errors.clone());
        self.warnings.extend(other.warnings.clone());
    }

    /// Clears all errors and warnings.
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }

    /// Raises an exception if the report contains any errors.
    ///
    /// # Raises
    /// * `RuntimeError` - If there are validation errors
    pub fn raise_if_invalid(&self) -> PyResult<()> {
        if !self.is_valid() {
            Err(PyRuntimeError::new_err(self.format_report()))
        } else {
            Ok(())
        }
    }

    fn __repr__(&self) -> String {
        format!("ValidationReport(errors={}, warnings={})",
            self.errors.len(),
            self.warnings.len()
        )
    }

    fn __str__(&self) -> String {
        self.format_report()
    }

    fn __bool__(&self) -> bool {
        self.is_valid()
    }

    fn __len__(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }
}

impl Default for PyValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Module Registration
// ---------------------------------------------------------------------------

/// Registers error types with a Python module.
pub fn register_module(
    py: Python<'_>,
    parent: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    parent.add_class::<PyErrorCategory>()?;
    parent.add_class::<PyGpuError>()?;
    parent.add_class::<PyGpuResult>()?;
    parent.add_class::<PyErrorHandler>()?;
    parent.add_class::<PyValidationReport>()?;

    // Add convenience aliases
    parent.setattr("ErrorCategory", py.get_type::<PyErrorCategory>())?;
    parent.setattr("GpuError", py.get_type::<PyGpuError>())?;
    parent.setattr("GpuResult", py.get_type::<PyGpuResult>())?;
    parent.setattr("ErrorHandler", py.get_type::<PyErrorHandler>())?;
    parent.setattr("ValidationReport", py.get_type::<PyValidationReport>())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- PyErrorCategory Tests ---

    #[test]
    fn test_error_category_names() {
        assert_eq!(PyErrorCategory::Validation.name(), "Validation");
        assert_eq!(PyErrorCategory::Memory.name(), "Memory");
        assert_eq!(PyErrorCategory::Device.name(), "Device");
        assert_eq!(PyErrorCategory::Shader.name(), "Shader");
        assert_eq!(PyErrorCategory::Pipeline.name(), "Pipeline");
        assert_eq!(PyErrorCategory::Resource.name(), "Resource");
        assert_eq!(PyErrorCategory::Feature.name(), "Feature");
        assert_eq!(PyErrorCategory::Limit.name(), "Limit");
    }

    #[test]
    fn test_error_category_recoverability() {
        assert!(PyErrorCategory::Validation.typically_recoverable());
        assert!(PyErrorCategory::Memory.typically_recoverable());
        assert!(!PyErrorCategory::Device.typically_recoverable());
        assert!(PyErrorCategory::Shader.typically_recoverable());
        assert!(PyErrorCategory::Pipeline.typically_recoverable());
        assert!(PyErrorCategory::Resource.typically_recoverable());
        assert!(!PyErrorCategory::Feature.typically_recoverable());
        assert!(PyErrorCategory::Limit.typically_recoverable());
    }

    // --- PyGpuError Tests ---

    #[test]
    fn test_validation_error() {
        let error = PyGpuError::validation("Invalid buffer size".to_string());
        assert_eq!(error.variant_name(), "ValidationError");
        assert_eq!(error.category(), PyErrorCategory::Validation);
        assert_eq!(error.error_code(), 1000);
        assert!(error.is_recoverable());
        assert!(error.message().contains("Invalid buffer size"));
    }

    #[test]
    fn test_out_of_memory_error() {
        let error = PyGpuError::out_of_memory(1024 * 1024, 512 * 1024);
        assert_eq!(error.variant_name(), "OutOfMemory");
        assert_eq!(error.category(), PyErrorCategory::Memory);
        assert_eq!(error.error_code(), 2000);
        assert!(error.is_recoverable());
        assert!(error.message().contains("1048576"));
        assert!(error.message().contains("524288"));
    }

    #[test]
    fn test_device_lost_error() {
        let error = PyGpuError::device_lost("Driver timeout".to_string());
        assert_eq!(error.variant_name(), "DeviceLost");
        assert_eq!(error.category(), PyErrorCategory::Device);
        assert_eq!(error.error_code(), 3000);
        assert!(!error.is_recoverable()); // Device lost is NOT recoverable
        assert!(error.message().contains("Driver timeout"));
    }

    #[test]
    fn test_shader_compilation_error() {
        let errors = vec![
            "Line 10: undefined variable 'foo'".to_string(),
            "Line 15: type mismatch".to_string(),
        ];
        let error = PyGpuError::shader_compilation("main.wgsl".to_string(), errors);
        assert_eq!(error.variant_name(), "ShaderCompilation");
        assert_eq!(error.category(), PyErrorCategory::Shader);
        assert_eq!(error.error_code(), 4000);
        assert!(error.is_recoverable());
        assert!(error.message().contains("main.wgsl"));
        assert!(error.message().contains("undefined variable"));
    }

    #[test]
    fn test_pipeline_creation_error() {
        let error = PyGpuError::pipeline_creation(
            "pbr_pipeline".to_string(),
            "Vertex shader missing position output".to_string(),
        );
        assert_eq!(error.variant_name(), "PipelineCreation");
        assert_eq!(error.category(), PyErrorCategory::Pipeline);
        assert_eq!(error.error_code(), 5000);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_resource_creation_error() {
        let error = PyGpuError::resource_creation(
            "Texture".to_string(),
            "Dimensions exceed maximum".to_string(),
        );
        assert_eq!(error.variant_name(), "ResourceCreation");
        assert_eq!(error.category(), PyErrorCategory::Resource);
        assert_eq!(error.error_code(), 6000);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_feature_not_supported_error() {
        let error = PyGpuError::feature_not_supported("raytracing".to_string());
        assert_eq!(error.variant_name(), "FeatureNotSupported");
        assert_eq!(error.category(), PyErrorCategory::Feature);
        assert_eq!(error.error_code(), 7000);
        assert!(!error.is_recoverable()); // Feature not supported is NOT recoverable
        assert!(error.message().contains("raytracing"));
    }

    #[test]
    fn test_limit_exceeded_error() {
        let error = PyGpuError::limit_exceeded(
            "maxTextureSize".to_string(),
            16384,
            8192,
        );
        assert_eq!(error.variant_name(), "LimitExceeded");
        assert_eq!(error.category(), PyErrorCategory::Limit);
        assert_eq!(error.error_code(), 8000);
        assert!(error.is_recoverable());
        assert!(error.message().contains("16384"));
        assert!(error.message().contains("8192"));
    }

    #[test]
    fn test_error_codes_are_unique() {
        let errors = vec![
            PyGpuError::validation("test".to_string()),
            PyGpuError::out_of_memory(0, 0),
            PyGpuError::device_lost("test".to_string()),
            PyGpuError::shader_compilation("test".to_string(), vec![]),
            PyGpuError::pipeline_creation("test".to_string(), "test".to_string()),
            PyGpuError::resource_creation("test".to_string(), "test".to_string()),
            PyGpuError::feature_not_supported("test".to_string()),
            PyGpuError::limit_exceeded("test".to_string(), 0, 0),
        ];

        let codes: Vec<u32> = errors.iter().map(|e| e.error_code()).collect();
        let mut unique_codes = codes.clone();
        unique_codes.sort();
        unique_codes.dedup();

        assert_eq!(codes.len(), unique_codes.len(), "Error codes must be unique");
    }

    // --- PyErrorHandler Tests ---

    #[test]
    fn test_error_handler_creation() {
        let handler = PyErrorHandler::new();
        assert!(!handler.has_errors());
        assert_eq!(handler.error_count(), 0);
        assert!(handler.last_error().is_none());
    }

    #[test]
    fn test_error_handler_stores_errors() {
        let handler = PyErrorHandler::new();

        // Manually store errors (bypassing Python callback)
        {
            let mut errors = handler.errors.lock().unwrap();
            errors.push(PyGpuError::validation("Error 1".to_string()));
            errors.push(PyGpuError::out_of_memory(1024, 512));
        }

        assert!(handler.has_errors());
        assert_eq!(handler.error_count(), 2);

        let last = handler.last_error().unwrap();
        assert_eq!(last.variant_name(), "OutOfMemory");
    }

    #[test]
    fn test_error_handler_clear() {
        let handler = PyErrorHandler::new();

        {
            let mut errors = handler.errors.lock().unwrap();
            errors.push(PyGpuError::validation("Error".to_string()));
        }

        assert!(handler.has_errors());
        handler.clear();
        assert!(!handler.has_errors());
        assert_eq!(handler.error_count(), 0);
    }

    #[test]
    fn test_error_handler_filter_by_category() {
        let handler = PyErrorHandler::new();

        {
            let mut errors = handler.errors.lock().unwrap();
            errors.push(PyGpuError::validation("Val 1".to_string()));
            errors.push(PyGpuError::out_of_memory(1024, 512));
            errors.push(PyGpuError::validation("Val 2".to_string()));
            errors.push(PyGpuError::device_lost("Lost".to_string()));
        }

        let validation_errors = handler.errors_by_category(PyErrorCategory::Validation);
        assert_eq!(validation_errors.len(), 2);

        let memory_errors = handler.errors_by_category(PyErrorCategory::Memory);
        assert_eq!(memory_errors.len(), 1);

        let device_errors = handler.errors_by_category(PyErrorCategory::Device);
        assert_eq!(device_errors.len(), 1);
    }

    #[test]
    fn test_error_handler_recoverable_filter() {
        let handler = PyErrorHandler::new();

        {
            let mut errors = handler.errors.lock().unwrap();
            errors.push(PyGpuError::validation("Recoverable".to_string()));
            errors.push(PyGpuError::device_lost("Fatal".to_string()));
            errors.push(PyGpuError::feature_not_supported("Also fatal".to_string()));
            errors.push(PyGpuError::out_of_memory(1024, 512));
        }

        let recoverable = handler.recoverable_errors();
        assert_eq!(recoverable.len(), 2);

        let fatal = handler.fatal_errors();
        assert_eq!(fatal.len(), 2);
    }

    // --- PyValidationReport Tests ---

    #[test]
    fn test_validation_report_creation() {
        let report = PyValidationReport::new();
        assert!(report.is_valid());
        assert!(report.is_clean());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_validation_report_with_errors() {
        let mut report = PyValidationReport::new();
        report.add_error(PyGpuError::validation("Error 1".to_string()));
        report.add_error(PyGpuError::validation("Error 2".to_string()));

        assert!(!report.is_valid());
        assert!(!report.is_clean());
        assert_eq!(report.error_count(), 2);
    }

    #[test]
    fn test_validation_report_with_warnings() {
        let mut report = PyValidationReport::new();
        report.add_warning("Warning 1".to_string());
        report.add_warning("Warning 2".to_string());

        assert!(report.is_valid()); // Warnings don't invalidate
        assert!(!report.is_clean()); // But not clean
        assert_eq!(report.warning_count(), 2);
    }

    #[test]
    fn test_validation_report_format() {
        let mut report = PyValidationReport::new();
        report.add_error(PyGpuError::validation("Missing field".to_string()));
        report.add_warning("Consider using smaller buffer".to_string());

        let formatted = report.format_report();
        assert!(formatted.contains("FAILED"));
        assert!(formatted.contains("Missing field"));
        assert!(formatted.contains("smaller buffer"));
    }

    #[test]
    fn test_validation_report_merge() {
        let mut report1 = PyValidationReport::new();
        report1.add_error(PyGpuError::validation("Error 1".to_string()));
        report1.add_warning("Warning 1".to_string());

        let mut report2 = PyValidationReport::new();
        report2.add_error(PyGpuError::validation("Error 2".to_string()));
        report2.add_warning("Warning 2".to_string());

        report1.merge(&report2);

        assert_eq!(report1.error_count(), 2);
        assert_eq!(report1.warning_count(), 2);
    }

    #[test]
    fn test_validation_report_clear() {
        let mut report = PyValidationReport::new();
        report.add_error(PyGpuError::validation("Error".to_string()));
        report.add_warning("Warning".to_string());

        report.clear();

        assert!(report.is_clean());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_validation_report_errors_by_category() {
        let mut report = PyValidationReport::new();
        report.add_error(PyGpuError::validation("Val".to_string()));
        report.add_error(PyGpuError::out_of_memory(1024, 512));
        report.add_error(PyGpuError::validation("Val 2".to_string()));

        let validation = report.errors_by_category(PyErrorCategory::Validation);
        assert_eq!(validation.len(), 2);

        let memory = report.errors_by_category(PyErrorCategory::Memory);
        assert_eq!(memory.len(), 1);
    }

    // --- PyGpuResult Tests ---

    #[test]
    fn test_gpu_result_is_ok() {
        let result = PyGpuResult {
            value: Some(pyo3::Python::with_gil(|py| py.None())),
            error: None,
        };
        assert!(result.is_ok());
        assert!(!result.is_err());
    }

    #[test]
    fn test_gpu_result_is_err() {
        let result = PyGpuResult {
            value: None,
            error: Some(PyGpuError::validation("Error".to_string())),
        };
        assert!(!result.is_ok());
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_result_err_value() {
        let error = PyGpuError::validation("Test error".to_string());
        let result = PyGpuResult {
            value: None,
            error: Some(error),
        };

        let err = result.err_value().unwrap();
        assert_eq!(err.variant_name(), "ValidationError");
    }

    // --- Integration Tests ---

    #[test]
    fn test_error_handler_with_report() {
        let handler = PyErrorHandler::new();
        let mut report = PyValidationReport::new();

        // Add some errors to handler
        {
            let mut errors = handler.errors.lock().unwrap();
            errors.push(PyGpuError::validation("Error 1".to_string()));
            errors.push(PyGpuError::shader_compilation(
                "test.wgsl".to_string(),
                vec!["Syntax error".to_string()],
            ));
        }

        // Transfer to report
        for error in handler.errors() {
            report.add_error(error);
        }

        assert_eq!(report.error_count(), 2);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_all_error_categories_covered() {
        // Ensure every error type maps to a valid category
        let errors = vec![
            PyGpuError::validation("".to_string()),
            PyGpuError::out_of_memory(0, 0),
            PyGpuError::device_lost("".to_string()),
            PyGpuError::shader_compilation("".to_string(), vec![]),
            PyGpuError::pipeline_creation("".to_string(), "".to_string()),
            PyGpuError::resource_creation("".to_string(), "".to_string()),
            PyGpuError::feature_not_supported("".to_string()),
            PyGpuError::limit_exceeded("".to_string(), 0, 0),
        ];

        let categories: Vec<PyErrorCategory> = errors.iter().map(|e| e.category()).collect();

        // All 8 error types should map to 8 distinct categories
        assert_eq!(categories.len(), 8);

        // Check each maps correctly
        assert_eq!(errors[0].category(), PyErrorCategory::Validation);
        assert_eq!(errors[1].category(), PyErrorCategory::Memory);
        assert_eq!(errors[2].category(), PyErrorCategory::Device);
        assert_eq!(errors[3].category(), PyErrorCategory::Shader);
        assert_eq!(errors[4].category(), PyErrorCategory::Pipeline);
        assert_eq!(errors[5].category(), PyErrorCategory::Resource);
        assert_eq!(errors[6].category(), PyErrorCategory::Feature);
        assert_eq!(errors[7].category(), PyErrorCategory::Limit);
    }
}
