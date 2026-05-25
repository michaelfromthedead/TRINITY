// Omega -- PyO3 bridge to expose TypeRegistry and ComponentStore to Python
// as the `_omega` module.
//
// Functions here are called from trinity/metaclasses/component_meta.py and
// trinity/descriptors/rust_storage.py.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use renderer_backend::component_store;
use renderer_backend::frame_graph;
use renderer_backend::type_registry::{FieldLayout, TypeRegistry};
use std::sync::{Arc, Mutex, OnceLock};

static REGISTRY: OnceLock<Arc<TypeRegistry>> = OnceLock::new();

fn get_registry() -> &'static Arc<TypeRegistry> {
    REGISTRY.get_or_init(|| Arc::new(TypeRegistry::new()))
}

/// Register a component type with the global type registry.
///
/// `fields_json` is a JSON-encoded `Vec<FieldLayout>` array.
#[pyfunction]
pub fn type_register(component_id: u32, name: String, total_size: usize, fields_json: String) -> PyResult<()> {
    let fields: Vec<FieldLayout> =
        serde_json::from_str(&fields_json).map_err(|e| PyValueError::new_err(e.to_string()))?;

    get_registry().register(renderer_backend::type_registry::ComponentTypeInfo {
        id: component_id,
        name,
        size: total_size,
        fields,
        flags: 0,
        archetype_id: None,
    });
    Ok(())
}

/// Return a list of all registered component types as `(id, name, size)` tuples.
#[pyfunction]
pub fn type_list() -> Vec<(u32, String, usize)> {
    get_registry().type_list()
}

/// Initialise the global ECS component store.
///
/// Must be called once (and only once) before any read / write / delete
/// operations.  Grabs a clone of the shared type registry so the store can
/// resolve component sizes at runtime.
#[pyfunction]
pub fn initialize_component_store() -> PyResult<()> {
    let registry = get_registry().clone();
    component_store::initialize_component_store(registry);
    Ok(())
}

// ---------------------------------------------------------------------------
// Data Channel -- component-level read / write / delete
// ---------------------------------------------------------------------------

/// Read a single field from a component on an entity and return the decoded
/// Python value.
///
/// Parameters
/// ----------
/// entity_id : int
///     ECS entity identifier.
/// component_id : int
///     Registered component type ID.
/// offset : int
///     Byte offset of the field within the component struct.
/// field_type : str
///     Rust type-code string (`"f32"`, `"i32"`, `"u8"`, or `"string"`)
///     controlling how the raw bytes are decoded.
///
/// Returns
/// -------
/// Python `float`, `int`, `str`, or `None` when the entity / component /
/// field does not exist.
#[pyfunction]
pub fn component_read(
    entity_id: u64,
    component_id: u32,
    offset: usize,
    field_type: String,
    py: Python<'_>,
) -> PyObject {
    let store = component_store::global_component_store().read();

    // Determine how many bytes to read for this field type.
    let size: Option<usize> = match field_type.as_str() {
        "f32" | "i32" => Some(4),
        "u8" => Some(1),
        "string" => {
            // Variable-length: read from offset to end of the component.
            get_registry().get(component_id).map(|info| {
                if offset < info.size {
                    info.size - offset
                } else {
                    0
                }
            })
        }
        _ => None,
    };

    let size = match size {
        Some(s) if s > 0 => s,
        _ => return py.None(),
    };

    match store.read_field(entity_id, component_id, offset, size) {
        Some(bytes) if !bytes.is_empty() => match field_type.as_str() {
            "f32" if bytes.len() >= 4 => {
                let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                val.into_py(py)
            }
            "i32" if bytes.len() >= 4 => {
                let val = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                val.into_py(py)
            }
            "u8" => bytes[0].into_py(py),
            "string" => {
                // Stop at the first null byte (C-string convention used by
                // the SoA column writers).
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                String::from_utf8_lossy(&bytes[..end]).into_py(py)
            }
            _ => py.None(),
        },
        _ => py.None(),
    }
}

/// Write a Python value into a component field on an entity.
///
/// The value is auto-serialised to raw bytes:
///
/// | Python type | Bytes written |
/// |-------------|---------------|
/// | `bool`      | 1 (`u8`)      |
/// | `int`       | 4 (`i32` LE)  |
/// | `float`     | 4 (`f32` LE)  |
/// | `str`       | UTF-8 bytes   |
/// | `bytes`     | as-is         |
///
/// Returns `Ok(())` even when the entity or component does not exist (the
/// underlying store is a silent no-op for unknown keys).
#[pyfunction]
pub fn component_write(
    entity_id: u64,
    component_id: u32,
    offset: usize,
    value: PyObject,
    py: Python<'_>,
) -> PyResult<()> {
    let data = pyobject_to_bytes(&value, py)?;
    component_store::global_component_store()
        .write()
        .write_field(entity_id, component_id, offset, &data);
    Ok(())
}

/// Zero out a component field on an entity (logical delete).
///
/// The field byte-size is resolved from the `TypeRegistry` so the correct
/// number of zero bytes is written.
///
/// Errors
/// ------
/// PyValueError
///     When `component_id` is not registered or no field exists at `offset`.
#[pyfunction]
pub fn component_delete(
    entity_id: u64,
    component_id: u32,
    offset: usize,
) -> PyResult<()> {
    let info = get_registry()
        .get(component_id)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown component type {}", component_id)))?;

    let size = field_size_from_registry(&info, offset)
        .ok_or_else(|| PyValueError::new_err(format!("No field at offset {} in component {}", offset, component_id)))?;

    let zeros = vec![0u8; size];
    component_store::global_component_store()
        .write()
        .write_field(entity_id, component_id, offset, &zeros);
    Ok(())
}

#[pyfunction]
pub fn frame_graph_execute(json: String) -> PyResult<String> {
    let (passes, resources) = frame_graph::deserialize_from_json(&json)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let result = frame_graph::execute(passes, resources)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(serde_json::to_string(&result).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a Python object to raw bytes for storage in the component store.
///
/// Check order: `bool` *before* `i32` because `bool` is a subclass of `int`
/// in Python and `extract::<i32>` would happily consume a `True` value.
fn pyobject_to_bytes(value: &PyObject, py: Python<'_>) -> PyResult<Vec<u8>> {
    if let Ok(b) = value.extract::<bool>(py) {
        return Ok(vec![b as u8]);
    }
    if let Ok(i) = value.extract::<i32>(py) {
        return Ok(i.to_le_bytes().to_vec());
    }
    if let Ok(f) = value.extract::<f64>(py) {
        return Ok((f as f32).to_le_bytes().to_vec());
    }
    if let Ok(s) = value.extract::<String>(py) {
        return Ok(s.into_bytes());
    }
    if let Ok(b) = value.extract::<Vec<u8>>(py) {
        return Ok(b);
    }
    Err(PyValueError::new_err(format!(
        "Unsupported Python type for component_write: {}",
        value.as_ref(py).get_type().name()?
    )))
}

/// Determine the byte size of a field at `offset` from the component's
/// `FieldLayout` entries.
///
/// For fixed-width type codes (`"f32"`, `"i32"`, `"u8"`, `"Fixed16"`,
/// `"Fixed32"`) the known width is returned.  For variable-width types the
/// gap to the next field (or the remaining component bytes for the last
/// field) is used.
fn field_size_from_registry(
    info: &renderer_backend::type_registry::ComponentTypeInfo,
    offset: usize,
) -> Option<usize> {
    let pos = info.fields.iter().position(|f| f.offset == offset)?;
    let current = &info.fields[pos];

    match current.type_code.as_str() {
        "f32" => return Some(4),
        "i32" => return Some(4),
        "u8" => return Some(1),
        "Fixed16" => return Some(16),
        "Fixed32" => return Some(32),
        _ => { /* fall through to offset-based sizing */ }
    }

    if let Some(next) = info.fields.get(pos + 1) {
        Some(next.offset - offset)
    } else {
        Some(info.size - offset)
    }
}

// ---------------------------------------------------------------------------
// Command Channel — Python→Renderer control functions
// ---------------------------------------------------------------------------

static RENDERER: OnceLock<Mutex<Option<renderer_backend::renderer::Renderer>>> = OnceLock::new();

fn get_renderer_lock() -> &'static Mutex<Option<renderer_backend::renderer::Renderer>> {
    RENDERER.get_or_init(|| Mutex::new(None))
}

#[pyfunction]
pub fn renderer_init() -> PyResult<()> {
    let _lock = get_renderer_lock();
    Ok(())
}

#[pyfunction]
pub fn renderer_resize(width: u32, height: u32) -> PyResult<()> {
    let mut guard = get_renderer_lock().lock().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })?;
    if let Some(ref mut renderer) = *guard {
        renderer.resize(width, height);
    }
    Ok(())
}

#[pyfunction]
pub fn renderer_screenshot(_path: String) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
pub fn material_compile(_material_id: u32, _wgsl_body: String) -> PyResult<String> {
    Ok(format!("// compiled material {}", _material_id))
}

#[pyfunction]
pub fn renderer_recompile_materials(_ids: Vec<u32>) -> PyResult<()> {
    Ok(())
}

/// Return the IDs of every alive entity in the global component store.
#[pyfunction]
pub fn editor_list_entities() -> PyResult<Vec<u64>> {
    let store = component_store::global_component_store().read();
    Ok(store.entity_index.keys().copied().collect())
}

#[pyfunction]
pub fn renderer_shutdown() -> PyResult<()> {
    let mut guard = get_renderer_lock().lock().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })?;
    *guard = None;
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 module registration
// ---------------------------------------------------------------------------

/// The `_omega` Python module exported by this crate.
#[pymodule]
fn _omega(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(type_register, m)?)?;
    m.add_function(wrap_pyfunction!(type_list, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_component_store, m)?)?;
    m.add_function(wrap_pyfunction!(component_read, m)?)?;
    m.add_function(wrap_pyfunction!(component_write, m)?)?;
    m.add_function(wrap_pyfunction!(component_delete, m)?)?;
    m.add_function(wrap_pyfunction!(frame_graph_execute, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_init, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_resize, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_screenshot, m)?)?;
    m.add_function(wrap_pyfunction!(material_compile, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_recompile_materials, m)?)?;
    m.add_function(wrap_pyfunction!(editor_list_entities, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_shutdown, m)?)?;
    Ok(())
}
