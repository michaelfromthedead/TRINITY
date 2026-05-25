# PHASE 4 TODO: VFX Graph System

## Tasks

### 4.1 Validate Graph Compilation Flow
**File**: `vfx_graph.py` (lines 830-870)
**Acceptance**:
- Dirty flag checked before compilation
- Cached emitter returned when not dirty
- Module categorization called first
- EmitterConfig extracted from VFXEmitterModule

### 4.2 Validate Module Categorization
**File**: `vfx_graph.py`
**Acceptance**:
- All VFXNodes converted to ParticleModules
- Modules sorted into spawn/update/render lists
- Stage property queried correctly
- Unknown node types handled gracefully

### 4.3 Validate VFXEmitterModule Conversion
**File**: `vfx_graph.py`
**Acceptance**:
- Capacity parameter transferred
- Duration parameter transferred
- Loop flag transferred
- Prewarm time transferred

### 4.4 Validate VFXSpawnNode Conversion
**File**: `vfx_graph.py`
**Acceptance**:
- Shape type maps to correct ShapeEmitter
- Rate converts to RateEmitter
- Burst converts to BurstEmitter
- Spawn parameters preserved

### 4.5 Validate VFXForceNode Conversion
**File**: `vfx_graph.py`
**Acceptance**:
- Gravity type maps to GravityModule
- Wind type maps to WindModule
- Turbulence type maps to TurbulenceModule
- Vortex type maps to VortexModule
- Force parameters preserved

### 4.6 Validate VFXAttributeNode Conversion
**File**: `vfx_graph.py`
**Acceptance**:
- Size over life converts correctly
- Color over life converts correctly
- Rotation converts correctly
- Easing curves preserved

### 4.7 Validate VFXRenderNode Conversion
**File**: `vfx_graph.py`
**Acceptance**:
- Billboard type maps to BillboardRenderer
- Mesh type maps to MeshParticleRenderer
- Alignment mode preserved
- Stretch settings preserved

### 4.8 Validate Connection Type Checking
**File**: `vfx_graph.py`
**Acceptance**:
- Vec3 to Vec3 connections allowed
- Float to float connections allowed
- Type mismatch produces error
- Error message indicates which nodes/ports

### 4.9 Validate DAG Enforcement
**File**: `vfx_graph.py`
**Acceptance**:
- Cycle detection runs at compile time
- Cycles produce compilation error
- Error message indicates cycle location
- Valid DAGs compile successfully

### 4.10 Validate Dirty Flag Behavior
**File**: `vfx_graph.py`
**Acceptance**:
- Adding node sets dirty flag
- Removing node sets dirty flag
- Adding connection sets dirty flag
- Changing parameter sets dirty flag
- Compilation clears dirty flag

### 4.11 Validate Empty Graph Handling
**File**: `vfx_graph.py`
**Acceptance**:
- Empty graph compiles without error
- Produces emitter with default config
- No modules added to empty emitter

### 4.12 Validate Multiple Emitter Modules
**File**: `vfx_graph.py`
**Acceptance**:
- Only first VFXEmitterModule used for config
- Warning if multiple emitter modules present
- Consistent behavior across compilations

### 4.13 Validate Module Ordering Within Stage
**File**: `vfx_graph.py`
**Acceptance**:
- Modules within stage maintain graph order
- Deterministic ordering across compilations
- Order affects simulation result as expected

### 4.14 Validate Compiled Emitter Functionality
**File**: `vfx_graph.py`
**Acceptance**:
- Compiled emitter spawns particles
- All modules execute correctly
- Visual output matches graph configuration
- Performance comparable to hand-coded emitter

### 4.15 Validate Graph Serialization (If Present)
**File**: `vfx_graph.py`
**Acceptance**:
- Graph saves to file format
- Graph loads from file format
- Round-trip preserves all data
- Compilation after load produces same emitter
