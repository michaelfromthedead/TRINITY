# RECOMMENDATIONS: engine/rendering/materials

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **Shader Compilation via naga** | Replace placeholder `_compile_internal()` with real SPIRV/WGSL output | Medium |
| **PBR Parameter Upload** | Zero-copy GPU upload of `PBRParameters` via PyO3 | Low |
| **Material Dirty Sync** | Batch sync of dirty materials to GPU each frame | Low |
| **WGSL Code Generator** | `GraphCompiler` currently emits GLSL only; wgpu needs WGSL | Medium |

### Medium Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **Shader Reflection** | Extract uniforms, samplers, bindings from compiled shaders | Medium |
| **Texture Table Integration** | Connect `PBRTextureSet` to bindless texture system | Medium |
| **Material Instance Buffers** | GPU-side storage for instanced material parameters | Medium |
| **Hot-Reload Signal** | Notify Rust when material files change | Low |

### Low Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **Advanced Model Shaders** | Generate WGSL for SSS, clearcoat, anisotropy, etc. | High |
| **Procedural Material Runtime** | Execute noise/pattern functions on GPU | High |
| **Raytracing Material Hit Shaders** | Extend material graph to RT pipeline | High |

## Integration Strategy

### Phase 1: Compilation Bridge (Week 1-2)
```rust
// crates/renderer-backend/src/materials/compiler.rs

use naga::{front::glsl, back::wgsl, valid::Validator};

pub fn compile_glsl_to_wgsl(glsl_source: &str, stage: ShaderStage) -> Result<String, CompileError> {
    let module = glsl::Frontend::default()
        .parse(&glsl::Options::from(stage), glsl_source)?;
    
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)?;
    
    let wgsl = wgsl::write_string(&module, &info, WriterFlags::empty())?;
    Ok(wgsl)
}

// PyO3 binding
#[pyfunction]
fn py_compile_glsl_to_wgsl(source: &str, stage: &str) -> PyResult<String> {
    compile_glsl_to_wgsl(source, stage.parse()?)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
```

### Phase 2: Material Data Bridge (Week 2-3)
```rust
// crates/renderer-backend/src/materials/bridge.rs

#[pyclass]
pub struct MaterialBridge {
    materials: HashMap<u64, GpuMaterial>,
    dirty_queue: Vec<u64>,
}

#[pymethods]
impl MaterialBridge {
    fn upload_pbr(&mut self, id: u64, params: &PBRParamsView) -> PyResult<()> {
        let gpu_params = GpuPBRParams {
            base_color: params.base_color.into(),
            metallic: params.metallic,
            roughness: params.roughness,
            // ...
        };
        self.materials.insert(id, GpuMaterial::PBR(gpu_params));
        self.dirty_queue.push(id);
        Ok(())
    }
    
    fn sync_dirty(&mut self, queue: &wgpu::Queue) -> PyResult<usize> {
        let count = self.dirty_queue.len();
        for id in self.dirty_queue.drain(..) {
            if let Some(mat) = self.materials.get(&id) {
                mat.upload(queue);
            }
        }
        Ok(count)
    }
}
```

### Phase 3: Graph Compiler Extension (Week 3-4)
Extend `GraphCompiler` in Python to emit WGSL alongside GLSL:

```python
class GraphCompiler:
    def compile(self, graph: MaterialGraph, language: ShaderLanguage = ShaderLanguage.GLSL) -> str:
        if language == ShaderLanguage.WGSL:
            return self._compile_wgsl(graph)
        return self._compile_glsl(graph)
    
    def _compile_wgsl(self, graph: MaterialGraph) -> str:
        # Generate WGSL syntax directly
        # OR call Rust naga bridge on GLSL output
        glsl = self._compile_glsl(graph)
        return rust_bridge.compile_glsl_to_wgsl(glsl, "fragment")
```

## Testing Strategy

### Unit Tests

| Test Suite | Coverage Target | Notes |
|------------|-----------------|-------|
| Material Graph Validation | All node types | Cycle detection, type checking |
| Code Generation | GLSL + WGSL output | Compare against golden files |
| PBR Parameter Clamping | Edge cases | Min/max ranges, NaN handling |
| Dirty Flag Propagation | All flag combinations | Instance-to-template tracking |
| Permutation Key Hashing | Conflict detection | Feature combination validity |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_material_graph_to_wgsl` | Graph compile -> WGSL -> naga validation |
| `test_pbr_upload_roundtrip` | Python params -> Rust bridge -> GPU buffer |
| `test_material_hot_reload` | File change -> dirty flag -> re-upload |
| `test_permutation_pso_cache` | Multiple variants -> cache hits |

### Visual Tests

| Test | Description |
|------|-------------|
| `visual_pbr_sphere` | Render PBR sphere with metallic/roughness gradients |
| `visual_sss_skin` | Subsurface scattering on skin profile |
| `visual_clearcoat_car` | Clearcoat on metallic car paint |
| `visual_anisotropic_hair` | Anisotropic highlights on hair strands |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **naga incompatibility** | Low | High | Pin naga version, test all GLSL features |
| **GLSL dialect differences** | Medium | Medium | Lint GLSL output for compatibility |
| **Python-Rust data mismatch** | Medium | Medium | Code-gen POD structs from shared schema |
| **Hot-reload race conditions** | Low | Medium | Use file locks, version tokens |
| **Shader compilation latency** | Medium | Low | Background compilation, PSO pre-warming |

### Dependency Risks

| Dependency | Risk | Notes |
|------------|------|-------|
| naga | Low | Well-maintained, gfx-rs ecosystem |
| wgpu | Low | Active development, stable API |
| PyO3 | Low | Mature, widely used |
| glslang (if used) | Medium | Large dependency, consider naga-only |

## Success Criteria

### Minimum Viable Bridge
1. `GraphCompiler` produces valid WGSL via naga transpilation
2. `PBRParameters` upload to GPU with < 1ms latency
3. Dirty flag sync batches 100+ materials per frame
4. Hot-reload triggers shader recompilation within 100ms

### Full Integration
1. All 20+ material functions work in WGSL
2. Advanced shading models render correctly
3. Permutation variants compile to separate PSOs
4. No Python GIL contention on GPU upload path
