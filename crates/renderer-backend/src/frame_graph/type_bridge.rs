//! Type channel: PyO3 bridge for ECS component type registration.
//!
//! Provides [`type_register`] — a PyO3-exported function that accepts
//! component metadata from Python's `ComponentMeta.__new__()` and populates
//! the shared Rust [`TypeRegistry`].
//!
//! # Type Channel (T-FG-7.1)
//!
//! The bridge accepts:
//!
//! | Parameter | Python type | Rust type | Description |
//! |-----------|-------------|-----------|-------------|
//! | `component_id` | `int` | `u32` | Unique component type identifier |
//! | `component_name` | `str` | `String` | Human-readable component name |
//! | `field_layouts` | `list[tuple[str, str, int]]` | `Vec<(String, String, usize)>` | `(name, type_code, offset)` per field |
//! | `flags` | `int` | `u32` | Bitfield for component properties |
//!
//! The total component `size` is computed automatically from the field
//! layouts as `max(offset + type_size)` across all fields, so callers do
//! not need to pass it explicitly.
//!
//! # Global registry
//!
//! A single global [`TypeRegistry`] instance (backed by
//! `parking_lot::RwLock`) is lazily initialised on the first call and shared
//! across all subsequent calls.  This is thread-safe: readers never block
//! each other, and writers get exclusive access.
//!
//! [`TypeRegistry`]: crate::type_registry::TypeRegistry

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::type_registry::{ComponentTypeInfo, FieldLayout, TypeRegistry};
use std::sync::Arc;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

static GLOBAL_TYPE_REGISTRY: OnceLock<Arc<TypeRegistry>> = OnceLock::new();

/// Returns a reference to the global `TypeRegistry`, initialising it on the
/// first call.
fn global_registry() -> &'static Arc<TypeRegistry> {
    GLOBAL_TYPE_REGISTRY.get_or_init(|| Arc::new(TypeRegistry::new()))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a `type_code` string to its byte size.
///
/// Covers the common types used by the Python ECS descriptor layer.
/// Unknown type codes default to 4 bytes (reasonable for f32/i32/u32).
fn type_code_size(type_code: &str) -> usize {
    match type_code {
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" => 8,
        "f16" | "u16" | "i16" => 2,
        "u8" | "i8" | "bool" => 1,
        _ => {
            // Unknown type code — assume 4 bytes as a safe default for
            // GPU-oriented components (most fields are f32 or i32).
            4
        }
    }
}

/// Compute the total byte size of a component from its field layouts.
///
/// The result is `max(field.offset + type_code_size(field.type_code))`
/// across all fields, or 0 when there are no fields.
fn compute_component_size(fields: &[FieldLayout]) -> usize {
    fields
        .iter()
        .map(|f| f.offset + type_code_size(&f.type_code))
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Validation (pure-Rust, testable without Python runtime)
// ---------------------------------------------------------------------------

/// Validate field layout tuples.  Returns `Ok(())` or a descriptive error
/// string.
///
/// This pure-Rust function can be unit-tested without a Python runtime.
/// Checks:
/// - `component_name` must not be empty.
/// - Each field tuple must have a non-empty `name` and non-empty `type_code`.
pub fn validate_field_layouts(
    component_name: &str,
    field_layouts: &[(String, String, usize)],
) -> Result<(), String> {
    if component_name.is_empty() {
        return Err("component_name must not be empty".into());
    }
    for (idx, (name, type_code, _)) in field_layouts.iter().enumerate() {
        if name.is_empty() {
            return Err(format!("field at index {idx} has an empty name"));
        }
        if type_code.is_empty() {
            return Err(format!("field '{name}' has an empty type_code"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Register a component type in the global [`TypeRegistry`].
///
/// Called from Python `ComponentMeta.__new__()` when a new ECS component
/// class is defined.
///
/// # Arguments
///
/// * `component_id` — Unique numeric identifier for the component type.
/// * `component_name` — Human-readable name (e.g. `"Transform"`).
/// * `field_layouts` — Sequence of `(name, type_code, offset)` tuples
///   describing each field of the component.  The total byte `size` of the
///   component is derived from the maximum `offset + type_size` across all
///   fields.
/// * `flags` — Bitfield for component properties (reserved for future use;
///   pass 0).
///
/// # Errors
///
/// Returns [`PyValueError`] if the component name is empty or if any field
/// layout tuple has an empty field name or type code.
///
/// # Example (Python)
///
/// ```python
/// type_register(
///     1,
///     "Transform",
///     [("position", "f32", 0), ("rotation", "f32", 12)],
///     0,
/// )
/// ```
#[pyfunction]
pub fn type_register(
    component_id: u32,
    component_name: String,
    field_layouts: Vec<(String, String, usize)>,
    flags: u32,
) -> PyResult<()> {
    // Validate input; delegate to pure-Rust function for testability.
    validate_field_layouts(&component_name, &field_layouts)
        .map_err(|msg| PyValueError::new_err(msg))?;

    // Convert raw tuples into FieldLayout records (already validated above).
    let fields: Vec<FieldLayout> = field_layouts
        .into_iter()
        .map(|(name, type_code, offset)| FieldLayout {
            name,
            type_code,
            offset,
        })
        .collect();

    // Compute total byte size from the field layout list.
    let size = compute_component_size(&fields);

    let info = ComponentTypeInfo {
        id: component_id,
        name: component_name,
        size,
        fields,
        flags,
        archetype_id: None,
    };

    global_registry().register(info);
    Ok(())
}

// ---------------------------------------------------------------------------
// Module initialisation for PyO3
// ---------------------------------------------------------------------------

/// Initialise the `type_bridge` submodule within the frame_graph PyO3 module.
///
/// This function is called from the parent `#[pymodule]` to register the
/// `type_register` function so that Python code can import it.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(type_register, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a fresh (non-global) registry for testing, so that tests do
    /// not interfere with each other or with the global instance.
    fn test_registry() -> TypeRegistry {
        TypeRegistry::new()
    }

    // -- type_code_size -------------------------------------------------------

    #[test]
    fn test_type_code_size_known_types() {
        assert_eq!(type_code_size("f32"), 4);
        assert_eq!(type_code_size("i32"), 4);
        assert_eq!(type_code_size("u32"), 4);
        assert_eq!(type_code_size("f64"), 8);
        assert_eq!(type_code_size("i64"), 8);
        assert_eq!(type_code_size("u64"), 8);
        assert_eq!(type_code_size("u16"), 2);
        assert_eq!(type_code_size("i16"), 2);
        assert_eq!(type_code_size("f16"), 2);
        assert_eq!(type_code_size("u8"), 1);
        assert_eq!(type_code_size("i8"), 1);
        assert_eq!(type_code_size("bool"), 1);
    }

    #[test]
    fn test_type_code_size_unknown_defaults_to_four() {
        // Unknown type codes default to 4 bytes.
        assert_eq!(type_code_size("vec3"), 4);
        assert_eq!(type_code_size("mat4"), 4);
        assert_eq!(type_code_size(""), 4);
    }

    // -- compute_component_size -----------------------------------------------

    #[test]
    fn test_compute_size_empty_fields_is_zero() {
        assert_eq!(compute_component_size(&[]), 0);
    }

    #[test]
    fn test_compute_size_single_field() {
        let fields = vec![FieldLayout {
            name: "x".into(),
            type_code: "f32".into(),
            offset: 0,
        }];
        assert_eq!(compute_component_size(&fields), 4);
    }

    #[test]
    fn test_compute_size_multiple_fields() {
        let fields = vec![
            FieldLayout {
                name: "x".into(),
                type_code: "f32".into(),
                offset: 0,
            },
            FieldLayout {
                name: "y".into(),
                type_code: "f32".into(),
                offset: 4,
            },
            FieldLayout {
                name: "z".into(),
                type_code: "f32".into(),
                offset: 8,
            },
        ];
        assert_eq!(compute_component_size(&fields), 12);
    }

    #[test]
    fn test_compute_size_uses_max_offset_plus_type_size() {
        let fields = vec![
            FieldLayout {
                name: "header".into(),
                type_code: "u64".into(),
                offset: 0,
            },
            FieldLayout {
                name: "payload".into(),
                type_code: "u8".into(),
                offset: 100,
            },
        ];
        // max(0+8, 100+1) = 101
        assert_eq!(compute_component_size(&fields), 101);
    }

    #[test]
    fn test_compute_size_non_contiguous_offsets() {
        let fields = vec![
            FieldLayout {
                name: "a".into(),
                type_code: "f32".into(),
                offset: 0,
            },
            FieldLayout {
                name: "b".into(),
                type_code: "f32".into(),
                offset: 16, // gap at offset 8-15
            },
        ];
        // max(0+4, 16+4) = 20
        assert_eq!(compute_component_size(&fields), 20);
    }

    // -- Registration (uses a local registry, not the global) -----------------

    #[test]
    fn test_register_component_via_local_registry() {
        let registry = test_registry();
        let fields = vec![FieldLayout {
            name: "value".into(),
            type_code: "f32".into(),
            offset: 0,
        }];
        let size = compute_component_size(&fields);

        registry.register(ComponentTypeInfo {
            id: 10,
            name: "TestComponent".into(),
            size,
            fields,
            flags: 0,
            archetype_id: None,
        });

        let retrieved = registry.get(10).expect("component should be registered");
        assert_eq!(retrieved.id, 10);
        assert_eq!(retrieved.name, "TestComponent");
        assert_eq!(retrieved.size, 4);
        assert_eq!(retrieved.flags, 0);
        assert_eq!(retrieved.fields.len(), 1);
        assert_eq!(retrieved.fields[0].name, "value");
        assert!(retrieved.archetype_id.is_none());
    }

    #[test]
    fn test_register_component_with_flags() {
        let registry = test_registry();
        let fields = vec![
            FieldLayout {
                name: "x".into(),
                type_code: "f32".into(),
                offset: 0,
            },
            FieldLayout {
                name: "y".into(),
                type_code: "f32".into(),
                offset: 4,
            },
        ];
        let size = compute_component_size(&fields);

        registry.register(ComponentTypeInfo {
            id: 20,
            name: "Position".into(),
            size,
            fields,
            flags: 0b0001, // arbitrary flag
            archetype_id: None,
        });

        let retrieved = registry.get(20).expect("component should be registered");
        assert_eq!(retrieved.flags, 0b0001);
    }

    #[test]
    fn test_register_component_with_no_fields() {
        let registry = test_registry();
        registry.register(ComponentTypeInfo {
            id: 30,
            name: "Tag".into(),
            size: 0,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });

        let retrieved = registry.get(30).expect("component should be registered");
        assert_eq!(retrieved.id, 30);
        assert_eq!(retrieved.name, "Tag");
        assert_eq!(retrieved.size, 0);
        assert!(retrieved.fields.is_empty());
    }

    #[test]
    fn test_register_multiple_components_unique_ids() {
        let registry = test_registry();

        for id in 100..105u32 {
            registry.register(ComponentTypeInfo {
                id,
                name: format!("Comp{id}"),
                size: 4,
                fields: vec![],
                flags: 0,
                archetype_id: None,
            });
        }

        assert_eq!(registry.len(), 5);
        let ids = registry.ids();
        for id in 100..105u32 {
            assert!(ids.contains(&id), "id {id} should be registered");
        }
    }

    #[test]
    fn test_register_overwrites_existing() {
        let registry = test_registry();

        registry.register(ComponentTypeInfo {
            id: 99,
            name: "OldName".into(),
            size: 4,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });

        assert_eq!(registry.get(99).unwrap().name, "OldName");

        // Overwrite with different data.
        registry.register(ComponentTypeInfo {
            id: 99,
            name: "NewName".into(),
            size: 8,
            fields: vec![],
            flags: 0b0010,
            archetype_id: None,
        });

        let retrieved = registry.get(99).unwrap();
        assert_eq!(retrieved.name, "NewName");
        assert_eq!(retrieved.size, 8);
        assert_eq!(retrieved.flags, 0b0010);
    }

    // -- validate_field_layouts (pure-Rust, no Python runtime) ----------------

    #[test]
    fn test_validate_field_layouts_empty_component_name() {
        let fields = vec![("pos".into(), "f32".into(), 0usize)];
        let result = validate_field_layouts("", &fields);
        assert!(result.is_err(), "empty component_name should fail");
        assert_eq!(
            result.unwrap_err(),
            "component_name must not be empty",
        );
    }

    #[test]
    fn test_validate_field_layouts_empty_field_name() {
        let fields = vec![("".into(), "f32".into(), 0usize)];
        let result = validate_field_layouts("Transform", &fields);
        assert!(result.is_err(), "empty field name should fail");
        assert_eq!(
            result.unwrap_err(),
            "field at index 0 has an empty name",
        );
    }

    #[test]
    fn test_validate_field_layouts_empty_type_code() {
        let fields = vec![("pos".into(), "".into(), 0usize)];
        let result = validate_field_layouts("Transform", &fields);
        assert!(result.is_err(), "empty type_code should fail");
        assert_eq!(
            result.unwrap_err(),
            "field 'pos' has an empty type_code",
        );
    }

    #[test]
    fn test_validate_field_layouts_success() {
        let fields = vec![("x".into(), "f32".into(), 0usize)];
        let result = validate_field_layouts("Transform", &fields);
        assert!(result.is_ok(), "valid layout should pass");
    }

    // -- Field layout ordering is preserved -----------------------------------

    #[test]
    fn test_field_layout_order_preserved() {
        let registry = test_registry();
        let fields = vec![
            FieldLayout {
                name: "z".into(),
                type_code: "f32".into(),
                offset: 8,
            },
            FieldLayout {
                name: "y".into(),
                type_code: "f32".into(),
                offset: 4,
            },
            FieldLayout {
                name: "x".into(),
                type_code: "f32".into(),
                offset: 0,
            },
        ];
        let size = compute_component_size(&fields);
        assert_eq!(size, 12); // max(8+4, 4+4, 0+4) = 12

        registry.register(ComponentTypeInfo {
            id: 60,
            name: "OutOfOrder".into(),
            size,
            fields,
            flags: 0,
            archetype_id: None,
        });

        let info = registry.get(60).unwrap();
        assert_eq!(info.fields[0].name, "z");
        assert_eq!(info.fields[1].name, "y");
        assert_eq!(info.fields[2].name, "x");
    }

    // -- Global registry smoke test -------------------------------------------

    #[test]
    fn test_global_registry_is_some() {
        // Access the global registry — it should initialise without panicking.
        let _reg = global_registry();
        // Just verifying no crash.
    }

}
