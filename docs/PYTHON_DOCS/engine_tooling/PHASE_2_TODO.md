# PHASE 2 TODO: Asset Pipeline and Build System

## Overview

Phase 2 implements the asset pipeline from import through packaging. This phase creates the infrastructure for cooking assets to platform-specific formats and building distributable packages.

---

## 1. Build System

### 1.1 Cook Pipeline
- [ ] **T1.1.1**: Implement TextureCooker with actual image processing
  - Acceptance: PNG/TGA/EXR converted to BC1-BC7/ASTC/GNF
  - Acceptance: Mipmaps generated correctly
  - Integration: Requires image processing library (PIL/wand)
  - File: `engine/tooling/build/cook_system.py`

- [ ] **T1.1.2**: Implement MeshCooker with format parsing
  - Acceptance: FBX/OBJ/glTF loaded and converted
  - Acceptance: LOD generation functional
  - Integration: Requires mesh parsing library
  
- [ ] **T1.1.3**: Implement AudioCooker with encoding
  - Acceptance: WAV converted to platform formats
  - Acceptance: Quality settings respected
  - Integration: Requires audio encoding library

- [ ] **T1.1.4**: Implement ShaderCooker with compilation
  - Acceptance: HLSL compiled to DXIL/DXBC
  - Acceptance: Cross-compilation to SPIRV/Metal
  - Integration: Requires shader compiler (dxc, spirv-cross)

### 1.2 Build Pipeline
- [ ] **T1.2.1**: Test parallel execution scaling
  - Acceptance: Near-linear scaling to 8 workers
  - Acceptance: No race conditions in concurrent builds
  - File: `engine/tooling/build/build_pipeline.py`

- [ ] **T1.2.2**: Implement build cancellation
  - Acceptance: In-flight stages can be cancelled
  - Acceptance: Partial results cleaned up

- [ ] **T1.2.3**: Add build progress callbacks
  - Acceptance: Per-stage progress reported
  - Acceptance: ETA calculation functional

### 1.3 Build Cache
- [ ] **T1.3.1**: Test incremental build correctness
  - Acceptance: Modified files rebuilt
  - Acceptance: Unmodified files served from cache
  - File: `engine/tooling/build/build_cache.py`

- [ ] **T1.3.2**: Implement dependency invalidation
  - Acceptance: Header changes invalidate dependents
  - Acceptance: Cascading invalidation works

- [ ] **T1.3.3**: Add cache statistics
  - Acceptance: Hit/miss rate tracked
  - Acceptance: Cache size monitoring

### 1.4 Build Targets
- [ ] **T1.4.1**: Verify platform compiler flags
  - Acceptance: MSVC flags correct for Windows
  - Acceptance: Clang flags correct for Apple
  - Acceptance: GCC flags correct for Linux
  - File: `engine/tooling/build/build_targets.py`

- [ ] **T1.4.2**: Test console platform defines
  - Acceptance: PS5 defines (__PROSPERO__)
  - Acceptance: Xbox defines (_GAMING_XBOX_SCARLETT)
  - Acceptance: Switch defines (NN_NINTENDO_SDK)

### 1.5 Build Reports
- [ ] **T1.5.1**: Implement HTML report generation
  - Acceptance: CSS-styled report with timing breakdown
  - Acceptance: Expandable warning/error sections
  - File: `engine/tooling/build/build_report.py`

- [ ] **T1.5.2**: Add slowest stages analysis
  - Acceptance: Top 10 bottlenecks identified
  - Acceptance: Optimization suggestions

---

## 2. Packaging System

### 2.1 Package Builder
- [ ] **T2.1.1**: Implement package creation
  - Acceptance: Files compressed and packaged
  - Acceptance: Manifest written correctly
  - File: `engine/tooling/build/packaging.py`

- [ ] **T2.1.2**: Add package verification
  - Acceptance: Checksums validated on load
  - Acceptance: Corruption detected

- [ ] **T2.1.3**: Implement package extraction
  - Acceptance: Files extracted to destination
  - Acceptance: Directory structure preserved

### 2.2 Compression
- [ ] **T2.2.1**: Test compression levels
  - Acceptance: Level 1-9 produce expected ratios
  - Acceptance: Decompression produces original data
  
- [ ] **T2.2.2**: Add LZ4 compression option
  - Acceptance: Fast compression mode available
  - Acceptance: Better decompression speed

- [ ] **T2.2.3**: Add Zstd compression option
  - Acceptance: Better compression ratio than zlib
  - Acceptance: Streaming decompression

### 2.3 DLC System
- [ ] **T2.3.1**: Implement DLC dependency resolution
  - Acceptance: Required DLC checked before install
  - Acceptance: Circular dependencies prevented
  
- [ ] **T2.3.2**: Add DLC verification
  - Acceptance: Content integrity checked
  - Acceptance: Version compatibility checked

- [ ] **T2.3.3**: Implement DLC uninstall
  - Acceptance: Files removed cleanly
  - Acceptance: Dependent DLC handled

---

## 3. Asset Tools

### 3.1 Content Browser
- [ ] **T3.1.1**: Implement directory scanning
  - Acceptance: Recursive scanning with filters
  - Acceptance: Performance acceptable for 100k+ files
  - File: `engine/tooling/assettools/content_browser.py`

- [ ] **T3.1.2**: Add file watching for updates
  - Acceptance: New/modified files detected
  - Acceptance: Deleted files removed from view

- [ ] **T3.1.3**: Implement drag-drop operations
  - Acceptance: Files can be dragged to other views
  - Acceptance: Payload data correct

### 3.2 Import Pipeline
- [ ] **T3.2.1**: Implement format detection
  - Acceptance: File extension mapping works
  - Acceptance: Magic number detection for ambiguous files
  - File: `engine/tooling/assettools/import_pipeline.py`

- [ ] **T3.2.2**: Add import preset management
  - Acceptance: Presets saved and loaded
  - Acceptance: Default presets per asset type

- [ ] **T3.2.3**: Integrate with actual importers
  - Acceptance: FBX import working
  - Acceptance: Texture import working
  - Acceptance: Audio import working

### 3.3 Asset Search
- [ ] **T3.3.1**: Test query parsing
  - Acceptance: field:value syntax works
  - Acceptance: Operators (>, <, ~=) work
  - File: `engine/tooling/assettools/search.py`

- [ ] **T3.3.2**: Verify index performance
  - Acceptance: Search < 100ms for 100k assets
  - Acceptance: Index update < 10ms

- [ ] **T3.3.3**: Implement saved searches
  - Acceptance: Searches saved to disk
  - Acceptance: Usage tracking works

### 3.4 Asset Validation
- [ ] **T3.4.1**: Implement texture validation rules
  - Acceptance: Power-of-2 check
  - Acceptance: Max size check
  - Acceptance: Format compatibility check
  - File: `engine/tooling/assettools/asset_validation.py`

- [ ] **T3.4.2**: Implement mesh validation rules
  - Acceptance: Triangle count check
  - Acceptance: UV coverage check
  - Acceptance: Bone limit check

- [ ] **T3.4.3**: Add auto-fix support
  - Acceptance: Texture resize auto-fix
  - Acceptance: Naming convention auto-fix

### 3.5 Reference Manager
- [ ] **T3.5.1**: Test cycle detection
  - Acceptance: Circular references detected
  - Acceptance: Cycle path reported
  - File: `engine/tooling/assettools/reference_manager.py`

- [ ] **T3.5.2**: Implement reference redirection
  - Acceptance: Moved assets redirect correctly
  - Acceptance: Broken references fixable

### 3.6 Thumbnail Generator
- [ ] **T3.6.1**: Integrate image generation
  - Acceptance: Textures render as thumbnails
  - Acceptance: Meshes render as 3D previews
  - File: `engine/tooling/assettools/thumbnail_generator.py`

- [ ] **T3.6.2**: Test cache management
  - Acceptance: LRU eviction works
  - Acceptance: Content-hash invalidation works

---

## 4. Automation Framework

### 4.1 Commandlets
- [ ] **T4.1.1**: Implement CookCommandlet
  - Acceptance: Assets cooked per platform
  - Acceptance: Progress reporting
  - File: `engine/tooling/automation/commandlets.py`

- [ ] **T4.1.2**: Implement TestCommandlet
  - Acceptance: Tests discovered and run
  - Acceptance: Results reported

- [ ] **T4.1.3**: Implement ValidateCommandlet
  - Acceptance: Assets validated with rules
  - Acceptance: Summary report generated

### 4.2 CI Integration
- [ ] **T4.2.1**: Test Jenkins integration
  - Acceptance: Build triggered via API
  - Acceptance: Results fetched
  - Acceptance: Artifacts uploaded
  - File: `engine/tooling/automation/ci_integration.py`

- [ ] **T4.2.2**: Test GitHub Actions integration
  - Acceptance: Workflow dispatched
  - Acceptance: Check runs created
  - Acceptance: Commit status updated

- [ ] **T4.2.3**: Implement environment detection
  - Acceptance: CI detected from env vars
  - Acceptance: Correct provider instantiated

### 4.3 Build Agents
- [ ] **T4.3.1**: Test capability matching
  - Acceptance: Jobs assigned to capable agents
  - Acceptance: Missing capabilities rejected
  - File: `engine/tooling/automation/build_agents.py`

- [ ] **T4.3.2**: Implement job queueing
  - Acceptance: Priority ordering respected
  - Acceptance: Timeout handling works

- [ ] **T4.3.3**: Add agent health monitoring
  - Acceptance: Heartbeat tracking
  - Acceptance: Stale agents marked offline

### 4.4 Automated Testing
- [ ] **T4.4.1**: Implement bot behaviors
  - Acceptance: RandomWalk navigates level
  - Acceptance: Exploration covers map
  - Acceptance: Combat engages enemies
  - File: `engine/tooling/automation/automated_testing.py`

- [ ] **T4.4.2**: Test playtest recording
  - Acceptance: Events captured with timestamps
  - Acceptance: JSON export works

- [ ] **T4.4.3**: Implement playtest reporting
  - Acceptance: Metrics summarized
  - Acceptance: Anomalies highlighted

---

## Integration Tests

### I1. Full Asset Import
- [ ] **I1.1**: Import FBX with textures
  - Steps: Import FBX, import referenced textures, verify references
  - Acceptance: Complete asset with materials

### I2. Incremental Build
- [ ] **I2.1**: Build, modify, rebuild
  - Steps: Full build, modify one asset, rebuild
  - Acceptance: Only modified asset rebuilt

### I3. CI Pipeline
- [ ] **I3.1**: Simulate CI run
  - Steps: Run cook, test, package, report
  - Acceptance: Full pipeline completes

### I4. Package Round-Trip
- [ ] **I4.1**: Create and extract package
  - Steps: Build package, extract, verify contents
  - Acceptance: All files intact

---

## Performance Targets

| Metric | Target | Test Method |
|--------|--------|-------------|
| Texture cook | < 500ms per 4K texture | Benchmark BC7 compression |
| Mesh cook | < 1s per 100K triangle mesh | Benchmark FBX processing |
| Cache lookup | < 10ms | Benchmark 10000 lookups |
| Search query | < 100ms | Benchmark with 100K assets |
| Package creation | > 50MB/s | Benchmark with 10GB content |

---

## Dependencies

### Required Before Phase 2
- Phase 1 complete (logging, undo, console)
- Foundation ContentStore protocol
- Platform file system APIs

### External Libraries (Integration Points)
- PIL/Pillow for image processing
- FBX SDK or alternative for mesh import
- Audio encoding libraries
- Shader compilers (dxc, spirv-cross)

### Blocks Phase 3+
- Visual tools need asset import working
- Animation tools need mesh/skeleton import
- Material editor needs texture import
