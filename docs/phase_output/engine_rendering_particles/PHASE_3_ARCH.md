# PHASE 3 ARCHITECTURE: Trail Renderer and Decal System

## Overview
The rendering subsystems for trail effects and deferred decals are fully implemented. This phase validates correctness and edge cases.

## Trail Renderer Architecture

### Core Components
- **TrailRenderer**: Main controller, ring buffer of trail points
- **TrailPoint**: Position, width, color, tangent, right vector
- **Ribbon Mesh Generator**: Converts points to renderable geometry

### Ring Buffer Design
Trail points stored in circular buffer:
```
head -> newest point
tail -> oldest point
capacity -> maximum points

Insert: write at head, increment head % capacity
Remove: increment tail % capacity
Count: (head - tail + capacity) % capacity
```

### Catmull-Rom Tangent Calculation
From `trail_renderer.py` (lines 498-556):
```python
# For point i, tangent = (P[i+1] - P[i-1]) / 2
# Boundary cases use forward/backward difference
tangent = (points[i+1].position - points[i-1].position) * 0.5
```

Provides C1 continuity (smooth first derivative) across control points.

### Ribbon Mesh Generation
For each point:
1. Compute tangent via Catmull-Rom
2. Compute right vector: `right = normalize(cross(tangent, up_hint))`
3. Compute edge positions:
   - `left_pos = position + right * (width / 2)`
   - `right_pos = position - right * (width / 2)`
4. Generate quad between consecutive point pairs
5. UV coordinates: u = edge (0 or 1), v = normalized age

### Cap Generation
Trail endpoints get special cap geometry:
- Start cap: single triangle or rounded fan
- End cap: mirrors start cap
- Prevents visual gaps at trail termination

## Decal System Architecture

### Core Components
- **DecalSystem**: Manages decal instances and rendering
- **Decal**: Box projection, material, fade parameters
- **DecalAtlas**: Texture atlas with shelf packing
- **G-Buffer Integration**: Deferred rendering pipeline

### Box Projection
Decals project onto scene geometry via oriented bounding box:
```
1. Rasterize decal box in screen space
2. For each pixel, reconstruct world position from depth
3. Check if world position inside decal box
4. Transform to decal local space for UV lookup
5. Blend decal material with G-Buffer
```

### Atlas Shelf Packing
From `decal_system.py` (lines 568-632):
```python
# Shelf algorithm:
# 1. Try fit on current shelf (same row)
# 2. If no fit, start new shelf below
# 3. Track shelf height as max texture height in row

if current_shelf_x + padded_width <= atlas_width:
    # Fits on current shelf
    allocate at (current_shelf_x, current_shelf_y)
    current_shelf_x += padded_width
else:
    # Start new shelf
    current_shelf_y += current_shelf_height
    current_shelf_x = 0
    current_shelf_height = padded_height
```

### G-Buffer Modification
Decals can modify:
- Albedo (base color)
- Normal (via tangent space blending)
- Roughness/Metallic
- Emission

### Sorting
Decals sorted by:
1. Depth (back to front for correct blending)
2. Priority (user-defined for overlaps)
3. Age (newer decals on top)

## Decisions

### ADR-TRAIL-001: Ring Buffer for Points
- **Context**: Trail points added/removed frequently
- **Decision**: Circular buffer with head/tail pointers
- **Consequence**: O(1) add/remove, fixed memory footprint

### ADR-TRAIL-002: Catmull-Rom over Bezier
- **Context**: Need smooth curves through control points
- **Decision**: Catmull-Rom splines (interpolating)
- **Consequence**: Curve passes through all points, easier authoring

### ADR-DECAL-001: Deferred over Forward
- **Context**: Decal count potentially high
- **Decision**: Deferred G-Buffer decals
- **Consequence**: Decals independent of scene complexity

### ADR-DECAL-002: Shelf Packing for Atlas
- **Context**: Runtime texture atlas allocation
- **Decision**: Shelf packing (NFDH variant)
- **Consequence**: Fast O(n) packing, ~80-90% utilization
