# PHASE 3 TODO: Trail Renderer and Decal System

## Trail Renderer Tasks

### 3.1 Validate Ring Buffer Operations
**File**: `trail_renderer.py`
**Acceptance**:
- Insert at head wraps correctly at capacity
- Remove from tail wraps correctly
- Count calculation handles wrap-around
- Full buffer behavior correct (overwrite or reject)

### 3.2 Validate Catmull-Rom Tangent Calculation
**File**: `trail_renderer.py` (lines 498-556)
**Acceptance**:
- Interior points use `(P[i+1] - P[i-1]) / 2`
- First point uses forward difference
- Last point uses backward difference
- Zero-length segments handled gracefully

### 3.3 Validate Right Vector Computation
**File**: `trail_renderer.py`
**Acceptance**:
- Cross product with up hint produces perpendicular
- Handles degenerate case where tangent parallel to up
- Right vector normalized correctly

### 3.4 Validate Ribbon Edge Positions
**File**: `trail_renderer.py`
**Acceptance**:
- Left edge at `position + right * (width / 2)`
- Right edge at `position - right * (width / 2)`
- Width interpolation over lifetime correct
- Color interpolation over lifetime correct

### 3.5 Validate Quad Generation
**File**: `trail_renderer.py`
**Acceptance**:
- Quads connect consecutive point pairs
- Winding order correct for backface culling
- No degenerate triangles (zero area)
- Vertex positions match edge calculations

### 3.6 Validate UV Coordinate Generation
**File**: `trail_renderer.py`
**Acceptance**:
- U coordinate: 0 for left edge, 1 for right edge
- V coordinate: normalized position along trail (0 = head, 1 = tail)
- UVs suitable for texture scrolling effects

### 3.7 Validate Cap Geometry
**File**: `trail_renderer.py`
**Acceptance**:
- Start cap closes trail beginning
- End cap closes trail ending
- Cap shape matches trail width
- No T-junctions with ribbon mesh

### 3.8 Validate Trail Fade
**File**: `trail_renderer.py`
**Acceptance**:
- Alpha fades based on point age
- Fade curve configurable (linear, ease)
- Oldest points fully transparent

## Decal System Tasks

### 3.9 Validate Box Projection
**File**: `decal_system.py`
**Acceptance**:
- World position reconstructed from depth buffer
- Inside-box test correct for oriented boxes
- Local UV coordinates computed correctly
- Handles boxes at arbitrary orientations

### 3.10 Validate Atlas Shelf Packing
**File**: `decal_system.py` (lines 568-632)
**Acceptance**:
- Textures placed left-to-right on shelf
- New shelf started when width exceeded
- Shelf height tracks tallest texture in row
- Padding applied correctly between textures

### 3.11 Validate Atlas Region Lookup
**File**: `decal_system.py`
**Acceptance**:
- Texture ID maps to correct atlas region
- UV coordinates transformed to atlas space
- Padding excluded from sampled region

### 3.12 Validate G-Buffer Blending
**File**: `decal_system.py`
**Acceptance**:
- Albedo blended correctly with alpha
- Normal blending uses tangent space transform
- Roughness/metallic modified per decal settings
- Blend mode (alpha, additive, multiply) respected

### 3.13 Validate Decal Sorting
**File**: `decal_system.py`
**Acceptance**:
- Back-to-front depth sort for transparency
- Priority override for explicit ordering
- Age-based tiebreaker

### 3.14 Validate Decal Fade
**File**: `decal_system.py`
**Acceptance**:
- Distance fade based on camera proximity
- Age fade over decal lifetime
- Combined fade multiplied correctly

### 3.15 Validate Decal Culling
**File**: `decal_system.py`
**Acceptance**:
- Frustum culling for offscreen decals
- Occlusion culling if available
- LOD distance culling

### 3.16 Validate Atlas Resize
**File**: `decal_system.py`
**Acceptance**:
- Atlas grows when capacity exceeded
- Existing regions preserved
- UV remapping after resize (if needed)
