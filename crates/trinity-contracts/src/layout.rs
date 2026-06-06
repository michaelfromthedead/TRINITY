//! Layout contracts for struct size and alignment verification.
//!
//! Provides compile-time checks for struct layouts, useful for
//! GPU data structures that must match WGSL mirrors.

use std::collections::HashMap;

/// Layout specification for a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutSpec {
    /// Expected size in bytes.
    pub size: Option<usize>,
    /// Expected alignment in bytes.
    pub align: Option<usize>,
    /// Packed layout (no padding).
    pub packed: bool,
}

impl LayoutSpec {
    /// Create a new layout spec.
    pub fn new() -> Self {
        Self {
            size: None,
            align: None,
            packed: false,
        }
    }

    /// Set expected size.
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set expected alignment.
    pub fn align(mut self, align: usize) -> Self {
        self.align = Some(align);
        self
    }

    /// Set packed layout.
    pub fn packed(mut self) -> Self {
        self.packed = true;
        self
    }

    /// Check if size matches.
    pub fn check_size(&self, actual: usize) -> bool {
        self.size.map_or(true, |expected| expected == actual)
    }

    /// Check if alignment matches.
    pub fn check_align(&self, actual: usize) -> bool {
        self.align.map_or(true, |expected| expected == actual)
    }

    /// Check both size and alignment.
    pub fn check(&self, actual_size: usize, actual_align: usize) -> LayoutResult {
        let size_ok = self.check_size(actual_size);
        let align_ok = self.check_align(actual_align);

        if size_ok && align_ok {
            LayoutResult::Ok
        } else {
            LayoutResult::Mismatch {
                expected_size: self.size,
                actual_size,
                expected_align: self.align,
                actual_align,
            }
        }
    }
}

impl Default for LayoutSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of layout verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutResult {
    /// Layout matches.
    Ok,
    /// Layout mismatch.
    Mismatch {
        expected_size: Option<usize>,
        actual_size: usize,
        expected_align: Option<usize>,
        actual_align: usize,
    },
}

impl LayoutResult {
    /// Check if layout is ok.
    pub fn is_ok(&self) -> bool {
        matches!(self, LayoutResult::Ok)
    }

    /// Get error message if mismatch.
    pub fn error_message(&self) -> Option<String> {
        match self {
            LayoutResult::Ok => None,
            LayoutResult::Mismatch {
                expected_size,
                actual_size,
                expected_align,
                actual_align,
            } => {
                let mut msg = String::new();
                if let Some(exp) = expected_size {
                    if *exp != *actual_size {
                        msg.push_str(&format!(
                            "size mismatch: expected {}, got {}",
                            exp, actual_size
                        ));
                    }
                }
                if let Some(exp) = expected_align {
                    if *exp != *actual_align {
                        if !msg.is_empty() {
                            msg.push_str("; ");
                        }
                        msg.push_str(&format!(
                            "alignment mismatch: expected {}, got {}",
                            exp, actual_align
                        ));
                    }
                }
                Some(msg)
            }
        }
    }
}

/// WGSL type mirror mapping.
#[derive(Debug, Clone)]
pub struct WgslMirror {
    /// Rust type name.
    pub rust_type: String,
    /// WGSL type name.
    pub wgsl_type: String,
    /// Expected layout.
    pub layout: LayoutSpec,
}

impl WgslMirror {
    /// Create a new mirror mapping.
    pub fn new(rust_type: impl Into<String>, wgsl_type: impl Into<String>) -> Self {
        Self {
            rust_type: rust_type.into(),
            wgsl_type: wgsl_type.into(),
            layout: LayoutSpec::new(),
        }
    }

    /// Set layout.
    pub fn layout(mut self, layout: LayoutSpec) -> Self {
        self.layout = layout;
        self
    }

    /// Verify layout matches.
    pub fn verify(&self, actual_size: usize, actual_align: usize) -> LayoutResult {
        self.layout.check(actual_size, actual_align)
    }
}

/// Registry of WGSL mirror types.
#[derive(Debug, Default)]
pub struct MirrorRegistry {
    /// Mirrors by Rust type name.
    mirrors: HashMap<String, WgslMirror>,
}

impl MirrorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a mirror.
    pub fn register(&mut self, mirror: WgslMirror) {
        self.mirrors.insert(mirror.rust_type.clone(), mirror);
    }

    /// Get a mirror by Rust type.
    pub fn get(&self, rust_type: &str) -> Option<&WgslMirror> {
        self.mirrors.get(rust_type)
    }

    /// List all registered types.
    pub fn types(&self) -> Vec<&str> {
        self.mirrors.keys().map(|s| s.as_str()).collect()
    }

    /// Get count.
    pub fn len(&self) -> usize {
        self.mirrors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.mirrors.is_empty()
    }

    /// Verify all mirrors against actual layouts.
    pub fn verify_all<F>(&self, get_layout: F) -> Vec<LayoutError>
    where
        F: Fn(&str) -> Option<(usize, usize)>,
    {
        let mut errors = Vec::new();

        for (name, mirror) in &self.mirrors {
            if let Some((size, align)) = get_layout(name) {
                let result = mirror.verify(size, align);
                if let Some(msg) = result.error_message() {
                    errors.push(LayoutError {
                        rust_type: name.clone(),
                        wgsl_type: mirror.wgsl_type.clone(),
                        message: msg,
                    });
                }
            }
        }

        errors
    }
}

/// Layout verification error.
#[derive(Debug, Clone)]
pub struct LayoutError {
    /// Rust type name.
    pub rust_type: String,
    /// WGSL type name.
    pub wgsl_type: String,
    /// Error message.
    pub message: String,
}

/// Common GPU struct sizes for std140/std430.
pub mod gpu_sizes {
    /// vec2<f32> size
    pub const VEC2_F32: usize = 8;
    /// vec3<f32> size (padded to 16)
    pub const VEC3_F32: usize = 12;
    /// vec4<f32> size
    pub const VEC4_F32: usize = 16;
    /// mat4x4<f32> size
    pub const MAT4X4_F32: usize = 64;

    /// std140 alignment for vec3
    pub const STD140_VEC3_ALIGN: usize = 16;
    /// std430 alignment for vec3
    pub const STD430_VEC3_ALIGN: usize = 16;
}

/// Verify a type's layout at compile time.
#[macro_export]
macro_rules! assert_layout {
    ($ty:ty, size = $size:expr) => {
        const _: () = {
            assert!(
                std::mem::size_of::<$ty>() == $size,
                concat!("size mismatch for ", stringify!($ty))
            );
        };
    };
    ($ty:ty, align = $align:expr) => {
        const _: () = {
            assert!(
                std::mem::align_of::<$ty>() == $align,
                concat!("alignment mismatch for ", stringify!($ty))
            );
        };
    };
    ($ty:ty, size = $size:expr, align = $align:expr) => {
        const _: () = {
            assert!(
                std::mem::size_of::<$ty>() == $size,
                concat!("size mismatch for ", stringify!($ty))
            );
            assert!(
                std::mem::align_of::<$ty>() == $align,
                concat!("alignment mismatch for ", stringify!($ty))
            );
        };
    };
}

/// Check layout at runtime.
pub fn check_layout<T>(spec: &LayoutSpec) -> LayoutResult {
    let actual_size = std::mem::size_of::<T>();
    let actual_align = std::mem::align_of::<T>();
    spec.check(actual_size, actual_align)
}

/// Get layout of a type.
pub fn get_layout<T>() -> (usize, usize) {
    (std::mem::size_of::<T>(), std::mem::align_of::<T>())
}
