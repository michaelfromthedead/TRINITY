# PHASE 2 ARCHITECTURE: Asset Pipeline and Build System

## Phase Overview

Phase 2 implements the complete asset pipeline from import through packaging, including the build system, asset management, and automation infrastructure.

## Components

### 1. Build System (engine/tooling/build/)

**Purpose**: Multi-platform asset cooking and compilation

**Architecture**:
```
BuildPipeline
    |
    +-- BuildConfig
    |       +-- CompilerSettings (optimization, debug, defines)
    |       +-- LinkerSettings (static, LTO, stripping)
    |       +-- AssetSettings (textures, compression, mipmaps)
    |
    +-- BuildGraph (DAG)
    |       +-- BuildStage (name, execute, dependencies, priority)
    |       +-- topological_sort() (Kahn's algorithm)
    |       +-- get_ready_stages() (parallel scheduling)
    |
    +-- BuildExecutor
    |       |-- SequentialBuildExecutor
    |       +-- ParallelBuildExecutor (ThreadPoolExecutor)
    |
    +-- CookPipeline
    |       +-- CookRegistry
    |       +-- AssetCooker (ABC)
    |               |-- TextureCooker (ASTC, GNF, DDS)
    |               |-- MeshCooker (LOD support)
    |               |-- AudioCooker (AT9, XMA2, Opus)
    |               +-- ShaderCooker (DXIL, Metal, SPIRV)
    |
    +-- BuildCache
    |       +-- ContentHash (SHA-256)
    |       +-- CacheEntry (dependencies, hit count)
    |       +-- BuildCacheBackend
    |               |-- FilesystemCache
    |               +-- MemoryCache (LRU)
    |
    +-- BuildReport
            +-- BuildTiming (hierarchical)
            +-- BuildMessage (severity, source location)
            +-- ReportFormatter
                    |-- TextReportFormatter
                    |-- JSONReportFormatter
                    +-- HTMLReportFormatter
```

**Platform Targets**:
| Platform | Architecture | Graphics APIs |
|----------|--------------|---------------|
| Windows | x64 | DX12, DX11, Vulkan |
| Linux | x64 | Vulkan, OpenGL |
| macOS | ARM64/x64/Universal | Metal, Vulkan |
| Android | ARM64/ARM32 | Vulkan, OpenGL ES |
| iOS | ARM64 | Metal |
| PS5 | x64 | GNM |
| Xbox Series | x64 | DX12 |
| Switch | ARM64 | NVN, Vulkan |

**Configuration Presets**:
- DEBUG: Full symbols, no optimization, cheats enabled
- DEVELOPMENT: Standard optimization, symbols, dev features
- SHIPPING: Aggressive optimization, LTO, stripped
- TEST: Unit testing defines enabled
- PROFILE: Optimized with profiling instrumentation

### 2. Packaging System (engine/tooling/build/packaging.py)

**Purpose**: Game distribution with compression and encryption

**Architecture**:
```
PackageBuilder
    |
    +-- PackageManifest
    |       +-- FileEntry (path, offset, sizes, checksum)
    |       +-- DLCInfo (dependencies, files)
    |
    +-- Compressor
    |       |-- ZlibCompressor
    |       |-- LZ4Compressor (planned)
    |       +-- ZstdCompressor (planned)
    |
    +-- PackageEncryption
    |       +-- KeyGenerator (AES-128, AES-256, ChaCha20)
    |
    +-- DLCManager
            +-- register(), install(), uninstall()
            +-- dependency resolution
```

**Package Binary Format**:
```
Header (64 bytes):
  Signature: "GPKG" (4 bytes)
  Version: uint32
  Package type: uint32 (FULL, PATCH, DLC, MOD, DEMO)
  Compression: uint32
  Encryption: uint32
  Data size: uint64
  File count: uint32
  Reserved: 36 bytes

[Compressed/Encrypted file data]
[File entry table]
```

### 3. Asset Tools (engine/tooling/assettools/)

**Purpose**: Content management and processing

**Architecture**:
```
ContentBrowser
    |
    +-- Directory navigation
    +-- Multi-criteria filtering
    +-- Sorting (name, date, size, type)
    +-- Selection management
    +-- Favorites and history
    +-- Drag-drop payload

ImportPipeline
    |
    +-- Format detection
    +-- Format-specific settings
    |       +-- FBXImportSettings
    |       +-- TextureImportSettings
    |       +-- AudioImportSettings
    +-- Preset management
    +-- ContentStore integration (deduplication)
    +-- Provenance tracking

AssetProcessor
    |
    +-- ThreadPoolExecutor
    +-- Pipeline operations
    |       +-- compress, convert, resize
    |       +-- optimize_mesh, convert_audio
    +-- Task tracking (status, progress, cancellation)

AssetSearch
    |
    +-- Query parsing (field:value syntax)
    +-- 14 operators (EQUALS, CONTAINS, MATCHES, GT, LT, IN, etc.)
    +-- Inverted index
    +-- Saved searches
    +-- Pagination

AssetValidator
    |
    +-- ValidationRule (ABC)
    |       +-- TextureValidationRule
    |       +-- MeshValidationRule
    |       +-- MaterialValidationRule
    |       +-- NamingConventionRule
    +-- Validation profiles
    +-- Result caching

ReferenceManager
    |
    +-- Reference graph (bidirectional)
    +-- Cycle detection
    +-- Broken reference suggestions
    +-- Reference redirects

CollectionManager
    |
    +-- Manual collections
    +-- Smart collections (query-based)
    +-- Nested hierarchies

ThumbnailGenerator
    |
    +-- Priority queue
    +-- LRU cache
    +-- Thread pool
```

### 4. Automation Framework (engine/tooling/automation/)

**Purpose**: Build automation and CI/CD integration

**Architecture**:
```
CommandletRunner
    |
    +-- Commandlet (ABC)
            |-- CookCommandlet (shaders, textures, meshes, audio)
            |-- BuildCommandlet (clean, generate, compile, link)
            |-- TestCommandlet (discovery, execution, reporting)
            |-- ValidateCommandlet (textures, meshes, references)
            |-- CleanCommandlet
            +-- PackageCommandlet

CIProvider (ABC)
    |
    +-- JenkinsIntegration (job API, artifacts)
    +-- GitHubActionsIntegration (workflow dispatches, checks)
    +-- TeamCityIntegration (REST API, service messages)

BuildAgentPool
    |
    +-- BuildAgent (capabilities, tags, heartbeat)
    +-- BuildJob (priority, timeout, status)
    +-- Capability-based matching
    +-- Priority queue dispatch

AutomationTestRunner
    |
    +-- @automation_test decorator
    +-- @automation_step decorator
    +-- Timeout enforcement (threading)
    +-- Automatic retry
    +-- Screenshot on failure

PlaytestSession
    |
    +-- BotController
    |       +-- BotBehavior (ABC)
    |               |-- RandomWalkBehavior
    |               |-- ExplorationBehavior
    |               |-- CombatBehavior
    |               +-- ScriptedBehavior
    +-- PlaytestRecorder
    +-- PlaytestReporter
```

## Data Flow

### Asset Import Flow
```
Source File
    -> ImportPipeline.detect_format()
    -> ImportPipeline.get_settings()
    -> AssetProcessor.process()
    -> ContentStore.store() (deduplicate)
    -> Provenance.record()
    -> ThumbnailGenerator.queue()
    -> AssetSearch.index()
```

### Build Pipeline Flow
```
BuildConfig.create_from_preset()
    -> BuildGraph.add_stage() (for each asset type)
    -> BuildGraph.validate() (check cycles, missing deps)
    -> BuildGraph.topological_sort()
    -> ParallelBuildExecutor.execute()
        -> For each ready stage:
            -> CookRegistry.get_cooker(extension)
            -> BuildCache.check(content_hash)
            -> If cache miss:
                -> AssetCooker.cook()
                -> BuildCache.store()
            -> BuildReport.record_timing()
    -> PackageBuilder.build()
    -> BuildReport.generate()
```

### CI/CD Flow
```
CI Trigger (push, PR)
    -> CIProvider.detect_environment()
    -> CommandletRunner.run("build")
    -> CommandletRunner.run("test")
    -> TestCommandlet.discover_tests()
    -> TestCommandlet.run_tests()
    -> CIProvider.publish_test_results()
    -> CommandletRunner.run("package")
    -> CIProvider.upload_artifacts()
    -> CIProvider.update_commit_status()
```

## Integration Points

### Foundation Integration
- Provenance tracking for asset lineage
- ContentStore protocol for deduplication

### Platform Integration
- Platform-specific cookers for each target
- SDK version awareness
- Compiler/linker flag generation

### VCS Integration
- Asset locking during import
- Reference tracking for VCS operations

## Thread Safety

| Component | Lock Strategy |
|-----------|---------------|
| CookRegistry | RLock (registration) |
| BuildCache | File-based locking |
| AssetSearch | Index lock per operation |
| ThumbnailGenerator | Queue lock |
| BuildAgentPool | RLock (agent/job management) |

## Configuration

### Build Configuration
```python
BuildConfiguration(
    preset=ConfigurationPreset.DEVELOPMENT,
    target=WindowsTarget(),
    compiler=CompilerSettings(
        optimization=OptimizationLevel.SPEED,
        debug=DebugLevel.STANDARD,
        defines=["DEVELOPMENT=1"],
    ),
    linker=LinkerSettings(
        static_linking=False,
        lto=False,
    ),
    asset=AssetSettings(
        texture_format="BC7",
        max_texture_size=4096,
        compression_level=6,
    ),
)
```

### Cache Configuration
```python
BuildCache(
    backend=FilesystemCache(cache_dir=".build_cache"),
    max_age_days=30,
    max_size_mb=10000,
)
```

### Agent Pool Configuration
```python
BuildAgentPool(
    name="default",
    max_jobs_per_agent=4,
    job_timeout_seconds=3600,
    capabilities_required={AgentCapability.WINDOWS, AgentCapability.GPU},
)
```

## Testing Strategy

### Unit Tests
- Content hash consistency
- Topological sort correctness
- Cache hit/miss logic
- Query parsing

### Integration Tests
- Full asset import cycle
- Build with cache invalidation
- Package creation and verification

### Performance Tests
- Parallel build scaling
- Cache lookup latency
- Large asset processing
