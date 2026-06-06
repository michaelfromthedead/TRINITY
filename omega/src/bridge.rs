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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Global entity ID counter for generating unique entity IDs.
/// Starts at 1 so that 0 can be reserved as an invalid/null entity.
static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1);

static REGISTRY: OnceLock<Arc<TypeRegistry>> = OnceLock::new();

fn get_registry() -> &'static Arc<TypeRegistry> {
    REGISTRY.get_or_init(|| Arc::new(TypeRegistry::new()))
}

/// Register a component type with the global type registry.
///
/// Called from Python `ComponentMeta.__new__()` when a new ECS component
/// class is defined.
///
/// # Arguments
///
/// * `component_id` — Unique numeric identifier for the component type.
/// * `component_name` — Human-readable name (e.g. `"Transform"`).
/// * `field_layouts` — Sequence of `(name, type_code, offset)` tuples
///   describing each field of the component. The total byte `size` of the
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
    // Validate input
    if component_name.is_empty() {
        return Err(PyValueError::new_err("component_name must not be empty"));
    }
    for (idx, (name, type_code, _)) in field_layouts.iter().enumerate() {
        if name.is_empty() {
            return Err(PyValueError::new_err(format!(
                "field at index {idx} has an empty name"
            )));
        }
        if type_code.is_empty() {
            return Err(PyValueError::new_err(format!(
                "field '{name}' has an empty type_code"
            )));
        }
    }

    // Convert raw tuples into FieldLayout records
    let fields: Vec<FieldLayout> = field_layouts
        .into_iter()
        .map(|(name, type_code, offset)| FieldLayout {
            name,
            type_code,
            offset,
        })
        .collect();

    // Compute total byte size from the field layout list
    let size = compute_component_size(&fields);

    get_registry().register(renderer_backend::type_registry::ComponentTypeInfo {
        id: component_id,
        name: component_name,
        size,
        fields,
        flags,
        archetype_id: None,
    });
    Ok(())
}

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
        "Fixed16" => 2,
        "Fixed32" => 4,
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

/// Compile a frame graph from separate passes and resources JSON arrays.
///
/// This function provides a more granular API than `frame_graph_execute`, allowing
/// callers to pass passes and resources as separate JSON arrays rather than a
/// single combined object.
///
/// # Arguments
///
/// * `passes_json` - JSON array of pass definitions. Each pass object should have:
///   - `name` (string): Pass identifier
///   - `pass_type` (string): "Graphics", "Compute", "Copy", or "RayTracing"
///   - `reads` (array of strings): Resource names this pass reads
///   - `writes` (array of strings): Resource names this pass writes
///   - Additional type-specific fields (e.g., `workgroup_size` for Compute)
///
/// * `resources_json` - JSON array of resource definitions. Each resource should have:
///   - `name` (string): Resource identifier
///   - `resource_type` (string): "Texture2D", "Texture3D", "Buffer", etc.
///   - `width`, `height`, `depth` (int): Dimensions
///   - `format` (string): Format string (e.g., "rgba8unorm")
///   - `is_transient` (bool): Whether the resource is transient
///
/// # Returns
///
/// A JSON string containing the compilation result with:
/// - `success` (bool): Whether compilation succeeded
/// - `num_passes` (int): Number of passes after dead pass elimination
/// - `num_resources` (int): Number of resources
/// - `num_edges` (int): Number of dependency edges
/// - `num_barriers` (int): Number of pipeline barriers
/// - `execution_order` (array of ints): Topologically sorted pass indices
/// - `passes` (array): Compiled pass information
/// - `resources` (array): Resource information
/// - `barriers` (array): Barrier information
/// - `cull_stats` (object): Dead pass elimination statistics
/// - `async_passes` (array): Passes eligible for async compute
/// - `parallel_regions` (array): Groups of passes that can execute in parallel
///
/// # Errors
///
/// Returns `PyRuntimeError` if:
/// - JSON parsing fails
/// - Pass references unknown resource names
/// - Frame graph has a dependency cycle
///
/// # Example (Python)
///
/// ```python
/// import json
/// from _omega import frame_graph_compile
///
/// passes = json.dumps([
///     {
///         "name": "gbuffer",
///         "pass_type": "Graphics",
///         "reads": [],
///         "writes": ["albedo", "normal", "depth"]
///     },
///     {
///         "name": "lighting",
///         "pass_type": "Compute",
///         "reads": ["albedo", "normal", "depth"],
///         "writes": ["hdr_color"],
///         "workgroup_size": [8, 8, 1]
///     }
/// ])
///
/// resources = json.dumps([
///     {"name": "albedo", "resource_type": "Texture2D", "width": 1920, "height": 1080, "format": "rgba8unorm"},
///     {"name": "normal", "resource_type": "Texture2D", "width": 1920, "height": 1080, "format": "rgba16f"},
///     {"name": "depth", "resource_type": "Texture2D", "width": 1920, "height": 1080, "format": "depth32float"},
///     {"name": "hdr_color", "resource_type": "Texture2D", "width": 1920, "height": 1080, "format": "rgba16f"}
/// ])
///
/// result = frame_graph_compile(passes, resources)
/// data = json.loads(result)
/// print(f"Compiled {data['num_passes']} passes with {data['num_barriers']} barriers")
/// ```
#[pyfunction]
pub fn frame_graph_compile(passes_json: &str, resources_json: &str) -> PyResult<String> {
    // Build a combined JSON object for deserialize_from_json
    let passes_value: serde_json::Value = serde_json::from_str(passes_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid passes JSON: {e}")))?;
    let resources_value: serde_json::Value = serde_json::from_str(resources_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid resources JSON: {e}")))?;

    // Validate that both are arrays
    if !passes_value.is_array() {
        return Err(PyValueError::new_err(
            "passes_json must be a JSON array",
        ));
    }
    if !resources_value.is_array() {
        return Err(PyValueError::new_err(
            "resources_json must be a JSON array",
        ));
    }

    // Combine into the expected format for deserialize_from_json
    let combined = serde_json::json!({
        "passes": passes_value,
        "resources": resources_value
    });

    let combined_str = serde_json::to_string(&combined)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("JSON serialization error: {e}")))?;

    // Parse into IR types
    let (passes, resources) = frame_graph::deserialize_from_json(&combined_str)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Compile the frame graph
    let compiled = frame_graph::CompiledFrameGraph::compile(passes, resources)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Serialize the full compilation result using emit_bridge_json
    let result_json = compiled.emit_bridge_json();

    serde_json::to_string(&result_json)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to serialize result: {e}")))
}

// ---------------------------------------------------------------------------
// World Operations -- entity spawn / despawn / query
// ---------------------------------------------------------------------------

/// Spawn a new entity with the specified component types.
///
/// Creates a new entity in the component store with zero-initialized data
/// for each component type. The entity is assigned a unique ID atomically.
///
/// Parameters
/// ----------
/// component_ids : list[int]
///     List of component type IDs that this entity will have. All component
///     types must be registered via `type_register()` before spawning.
///
/// Returns
/// -------
/// int
///     The unique entity ID assigned to the new entity.
///
/// Raises
/// ------
/// PyValueError
///     If any component_id is not registered in the type registry.
///
/// Example (Python)
/// ----------------
/// ```python
/// # Spawn entity with Transform (id=1) and Velocity (id=2) components
/// entity_id = world_spawn([1, 2])
/// ```
#[pyfunction]
pub fn world_spawn(component_ids: Vec<u32>) -> PyResult<u64> {
    // Validate all component IDs are registered
    let registry = get_registry();
    for &cid in &component_ids {
        if registry.get(cid).is_none() {
            return Err(PyValueError::new_err(format!(
                "Unknown component type ID: {}. Register it with type_register() first.",
                cid
            )));
        }
    }

    // Generate unique entity ID
    let entity_id = NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed);

    // Prepare zero-initialized component data
    let component_data: Vec<(u32, Vec<u8>)> = component_ids
        .iter()
        .filter_map(|&cid| {
            registry.get(cid).map(|info| (cid, vec![0u8; info.size]))
        })
        .collect();

    // Spawn in the component store
    component_store::global_component_store()
        .write()
        .spawn(entity_id, &component_ids, &component_data);

    Ok(entity_id)
}

/// Remove an entity from the component store.
///
/// Frees the entity's storage row for reuse by future spawns. This operation
/// is idempotent: despawning a non-existent or already-despawned entity
/// returns `False` without error.
///
/// Parameters
/// ----------
/// entity_id : int
///     The entity ID to despawn.
///
/// Returns
/// -------
/// bool
///     `True` if the entity existed and was removed, `False` otherwise.
///
/// Example (Python)
/// ----------------
/// ```python
/// entity_id = world_spawn([1, 2])
/// removed = world_despawn(entity_id)  # True
/// removed = world_despawn(entity_id)  # False (already removed)
/// ```
#[pyfunction]
pub fn world_despawn(entity_id: u64) -> PyResult<bool> {
    let mut store = component_store::global_component_store().write();

    // Check if entity exists before despawning
    let exists = store.entity_index.contains_key(&entity_id);

    if exists {
        store.despawn(entity_id);
    }

    Ok(exists)
}

/// Query entities that have all the specified component types.
///
/// Returns a list of entity IDs for all alive entities that possess every
/// component type in the `required_components` list. Despawned entities
/// are excluded from results.
///
/// Parameters
/// ----------
/// required_components : list[int]
///     List of component type IDs that entities must have. An empty list
///     returns all alive entities.
///
/// Returns
/// -------
/// list[int]
///     Entity IDs of all matching entities.
///
/// Example (Python)
/// ----------------
/// ```python
/// # Find all entities with both Transform (id=1) and Velocity (id=2)
/// moving_entities = world_query([1, 2])
///
/// # Find all entities with just Transform
/// all_transforms = world_query([1])
///
/// # Find all alive entities
/// all_entities = world_query([])
/// ```
#[pyfunction]
pub fn world_query(required_components: Vec<u32>) -> PyResult<Vec<u64>> {
    let store = component_store::global_component_store().read();
    Ok(store.query(&required_components))
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
    m.add_function(wrap_pyfunction!(frame_graph_compile, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_init, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_resize, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_screenshot, m)?)?;
    m.add_function(wrap_pyfunction!(material_compile, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_recompile_materials, m)?)?;
    m.add_function(wrap_pyfunction!(editor_list_entities, m)?)?;
    m.add_function(wrap_pyfunction!(renderer_shutdown, m)?)?;
    // World operations (T-FG-7.4)
    m.add_function(wrap_pyfunction!(world_spawn, m)?)?;
    m.add_function(wrap_pyfunction!(world_despawn, m)?)?;
    m.add_function(wrap_pyfunction!(world_query, m)?)?;
    Ok(())
}
