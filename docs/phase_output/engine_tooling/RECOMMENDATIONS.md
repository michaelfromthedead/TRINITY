# RECOMMENDATIONS: engine_tooling

## Rust Bridge Requirements

### High Priority

#### 1. Material Compiler WGSL Output
**Current State**: `material_compiler.py` generates HLSL/GLSL shaders
**Requirement**: Add WGSL generation backend for `renderer-backend`
**Implementation**:
```python
class MaterialCompiler:
    def compile(self, graph, target: str = "wgsl") -> str:
        if target == "wgsl":
            return self._compile_wgsl(graph)
        elif target == "hlsl":
            return self._compile_hlsl(graph)
```
**Effort**: 2-3 days
**Impact**: Real-time material preview in Rust renderer

#### 2. GPU Profiler Integration
**Current State**: `gpu_profiler.py` has placeholder timing
**Requirement**: Connect to wgpu timestamp queries
**Implementation**:
- Add PyO3 bindings for wgpu profiler
- Map Python profiler markers to GPU query pools
- Stream results back to Python profiler UI
**Effort**: 3-4 days
**Impact**: Accurate GPU performance metrics

#### 3. Asset Binary Format
**Current State**: Asset processor uses Python-internal formats
**Requirement**: Output Rust-loadable binary format
**Implementation**:
- Define shared binary format specification
- Add serialization to asset processor
- Match Rust asset loader expectations
**Effort**: 2-3 days
**Impact**: Single asset pipeline

### Medium Priority

#### 4. Visual Script WASM Compilation
**Current State**: Python bytecode interpreter
**Requirement**: Optional WASM output for Rust runtime
**Implementation**:
- Add WASM target to blueprint compiler
- Map opcodes to WASM instructions
- Generate WASM module with exports
**Effort**: 5-7 days
**Impact**: Game logic at native speed

#### 5. Editor Viewport Bridge
**Current State**: Placeholder/CPU preview rendering
**Requirement**: Route viewport rendering to Rust renderer
**Implementation**:
- Create Python wrapper for Rust render target
- Forward camera/transform data
- Display rendered frame in Python UI
**Effort**: 4-5 days
**Impact**: WYSIWYG editing

#### 6. Hot Reload Cross-Language Signal
**Current State**: Python-only schema change detection
**Requirement**: Notify Rust when Python components reload
**Implementation**:
- Add FFI callback for reload events
- Include type hash in notification
- Rust invalidates cached references
**Effort**: 1-2 days
**Impact**: Seamless cross-language iteration

### Low Priority

#### 7. VCS Large File Optimization
**Current State**: Pure Python file operations
**Requirement**: Rust acceleration for large repo operations
**Implementation**:
- Move file hashing to Rust
- Parallel diff generation
- Async file status checks
**Effort**: 2-3 days
**Impact**: Better UX for large projects

#### 8. Terrain Data Transfer
**Current State**: Python heightmaps stay in Python
**Requirement**: Efficient transfer to Rust terrain system
**Implementation**:
- Shared memory buffer for heightmap
- Delta updates for erosion changes
- LOD data format agreement
**Effort**: 2-3 days
**Impact**: Live terrain editing preview

## Integration Strategy

### Phase 1: Data Format Alignment (Week 1)
1. Define shared binary format for assets
2. Add WGSL output to material compiler
3. Document type mappings (Python ↔ Rust)

### Phase 2: Profiler Bridge (Week 2)
1. Add PyO3 bindings for GPU profiler
2. Wire up timestamp queries
3. Unify profiler UI for CPU+GPU

### Phase 3: Rendering Bridge (Week 3-4)
1. Create viewport render target wrapper
2. Implement camera/transform forwarding
3. Display Rust-rendered frames in Python

### Phase 4: Advanced Features (Week 5+)
1. Visual script WASM compilation
2. Hot reload cross-language signaling
3. Terrain data streaming

## Testing Strategy

### Unit Tests
- Each Python module already has test coverage
- Add tests for bridge code (PyO3 bindings)
- Test format serialization round-trips

### Integration Tests
```
test_material_wgsl_output.py
  - Compile material graph to WGSL
  - Validate shader compiles with naga
  - Compare rendering output

test_profiler_gpu_integration.py
  - Start GPU profile session
  - Render test scene
  - Verify GPU timing data received

test_asset_format_roundtrip.py
  - Process asset in Python
  - Load in Rust
  - Verify data integrity
```

### End-to-End Tests
- Open editor, modify material, see result in Rust renderer
- Profile frame with both CPU and GPU data
- Import asset, cook for platform, load in game

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WGSL compatibility issues | Medium | High | Use naga for validation |
| GPU profiler overhead | Low | Medium | Conditional enabling |
| Format version mismatch | Medium | High | Version header, migration |
| Memory ownership confusion | Medium | High | Clear FFI ownership rules |
| Hot reload race conditions | Low | Medium | Lock during reload |

### Critical Risks
1. **WGSL generation bugs**: Material shaders are complex; edge cases in codegen could cause subtle rendering bugs
   - Mitigation: Comprehensive test suite with reference images

2. **Profiler data correlation**: Matching Python and Rust profiler data requires careful timestamp synchronization
   - Mitigation: Use single clock source, handle latency

### Manageable Risks
- Format changes: Version headers allow graceful migration
- Performance: Bridge overhead minimal for infrequent operations
- Complexity: Well-defined interfaces limit scope

## Success Criteria

### Phase 1 Complete
- [ ] Material graph compiles to valid WGSL
- [ ] WGSL compiles with naga without errors
- [ ] Asset format documented and implemented

### Phase 2 Complete
- [ ] GPU profiler shows accurate frame times
- [ ] CPU+GPU data unified in profiler UI
- [ ] No measurable performance regression

### Phase 3 Complete
- [ ] Editor viewport renders via Rust
- [ ] Camera controls work smoothly
- [ ] Material changes update in real-time

### Full Integration Complete
- [ ] Visual scripts run in WASM
- [ ] Hot reload works across language boundary
- [ ] All 20 subsystems have Rust bridges where needed
