# XR Spatial Module Investigation

**Path**: `engine/xr/spatial/`  
**Date**: 2026-05-22  
**Investigator**: Research Agent

## Summary

The XR spatial module provides comprehensive AR spatial awareness capabilities across 6 Python modules totaling approximately 4,500 lines of code. All modules are **REAL implementations** with complete business logic, not stubs.

## File Inventory

| File | Lines | Classification | Description |
|------|-------|----------------|-------------|
| `__init__.py` | 178 | REAL | Module exports, comprehensive docstring with usage examples |
| `anchor.py` | 683 | REAL | Spatial anchors for world-locked AR content |
| `plane_detection.py` | 818 | REAL | Surface detection (floors, walls, tables) |
| `mesh_mapping.py` | 824 | REAL | Real-time spatial mesh reconstruction |
| `scene_understanding.py` | 772 | REAL | Semantic scene analysis and labeling |
| `image_tracking.py` | 786 | REAL | 2D image marker tracking |
| `object_tracking.py` | 812 | REAL | 3D object tracking |

**Total**: ~4,873 lines

## Classification Rationale: REAL (Not Stubs)

All modules exhibit:
- Complete class implementations with `__slots__` optimization
- Full business logic in methods (not `pass` or `raise NotImplementedError`)
- Proper state management with private attributes
- Callback/event systems for async notifications
- Mathematical algorithms (ray-triangle intersection, Moller-Trumbore, shoelace formula)
- Configuration dataclasses with sensible defaults
- Comprehensive docstrings with parameter/return documentation

## Module Analysis

### 1. Spatial Anchors (`anchor.py`)

**Classes**:
- `AnchorType` (Enum): LOCAL, PERSISTENT, CLOUD
- `AnchorTrackingState` (Enum): UNKNOWN, TRACKING, LIMITED, PAUSED, LOST, NOT_TRACKING
- `AnchorPersistenceState` (Enum): State machine for cloud anchor operations
- `AnchorPose` (dataclass): Position + orientation + timestamp
- `CloudAnchorConfig` (dataclass): Cloud sharing configuration
- `SpatialAnchor`: Core anchor with tracking, persistence, callbacks
- `AnchorManager`: Lifecycle management, spatial queries

**Capabilities**:
- Local session-only anchors
- Persistent device-stored anchors (restored across sessions)
- Cloud-shared anchors for multi-user AR
- Confidence decay for degraded tracking
- Entity attachment system
- Spatial queries (`get_anchors_near`)

**Integration Points**: References `engine.core.math.vec`, `engine.core.math.quat`, `engine.xr.config.XR_CONFIG`

### 2. Plane Detection (`plane_detection.py`)

**Classes**:
- `PlaneType` (Enum): FLOOR, CEILING, WALL, TABLE, SEAT, DOOR, WINDOW
- `PlaneOrientation` (Enum): HORIZONTAL_UP/DOWN, VERTICAL, ARBITRARY
- `PlaneTrackingState` (Enum): Detection lifecycle
- `PlaneBounds`: 2D polygon with ray-casting point containment
- `PlaneGeometry`: Center, normal, orientation, boundary
- `DetectedPlane`: Full plane with tracking, callbacks, coordinate transforms
- `PlaneDetector`: Multi-plane management, raycasting, placement helpers

**Capabilities**:
- Semantic plane classification (floor vs table vs wall)
- Boundary polygon with area computation (shoelace formula)
- Plane merging/subsumption
- Ray-plane intersection for hit testing
- `find_placement_surface()` for AR content placement
- World-to-local coordinate transforms

### 3. Mesh Mapping (`mesh_mapping.py`)

**Classes**:
- `MeshUpdateMode` (Enum): NONE, FULL, INCREMENTAL, ADAPTIVE
- `MeshLODLevel` (Enum): LOW, MEDIUM, HIGH, ULTRA
- `MeshClassification` (Enum): Per-vertex semantic labels
- `MeshVertex`, `MeshTriangle`, `MeshBounds`, `MeshBlock`: Geometry primitives
- `SpatialMesh`: Full mesh with LOD, versioning, raycast
- `SpatialMeshManager`: Lifecycle, cleanup, physics/occlusion integration

**Capabilities**:
- Block-based mesh chunking for efficient updates
- Moller-Trumbore ray-triangle intersection
- Ray-AABB bounds test for culling
- Distance-based block cleanup
- Mesh optimization (degenerate triangle removal)
- Physics and occlusion mesh extraction

### 4. Scene Understanding (`scene_understanding.py`)

**Classes**:
- `SemanticLabel` (Enum): 19 labels (FLOOR, WALL, TABLE, CHAIR, BED, PERSON, PET, etc.)
- `RoomType` (Enum): LIVING_ROOM, BEDROOM, KITCHEN, BATHROOM, OFFICE, etc.
- `OcclusionMode` (Enum): NONE, DEPTH_BASED, MESH_BASED, HUMAN_ONLY, FULL
- `SemanticRegion`, `RoomBounds`, `SceneObject`, `HumanSegment`: Scene elements
- `LightEstimate`: Ambient + directional + spherical harmonics
- `SceneUnderstanding`: Full scene management with ML-like heuristics

**Capabilities**:
- Room classification from detected objects
- Semantic region labeling
- Human segmentation and occlusion detection
- Light estimation (ambient + main directional)
- Placement helpers: `find_floor_position()`, `find_wall_position()`, `find_table_position()`
- Scene object tracking (movable, interactable flags)

### 5. Image Tracking (`image_tracking.py`)

**Classes**:
- `ImageTrackingState` (Enum): NONE, DETECTING, TRACKING, LIMITED, EXTENDED, LOST
- `TrackingMode` (Enum): CONTINUOUS, ONCE, ADAPTIVE
- `ar_trackable` (decorator): Mark classes as AR trackable
- `ImageReference`: Reference image database entry
- `ImageTargetPose`, `ImageTarget`: Tracked marker with pose
- `ImageTracker`: Database + target lifecycle management

**Capabilities**:
- Reference image database with physical dimensions
- Multi-image tracking (configurable max)
- Extended tracking when image not visible
- Tracking timeout management
- Corner position extraction for alignment
- Coordinate space transforms

### 6. Object Tracking (`object_tracking.py`)

**Classes**:
- `ObjectTrackingState`, `ObjectTrackingQuality` (Enums)
- `ObjectBounds`: 3D OBB with containment test
- `ObjectReference`: Reference 3D object with feature data
- `ObjectPose`: Position + orientation + scale
- `TrackedObject`: Full 3D object tracking
- `ObjectTracker`: Database + lifecycle management

**Capabilities**:
- 3D object recognition and pose estimation
- Scale-aware tracking
- World-space bounding box computation
- 8-corner extraction for visualization
- Object occlusion support
- Point containment testing

## Room-Scale Support

**Full room-scale support via**:
- `RoomBounds` class with floor_level, ceiling_level, wall_count
- `classify_room()` heuristic classification
- Plane detection for floor/wall boundaries
- Spatial mesh for environment geometry

## Boundary System

**Guardian/boundary equivalent**:
- `PlaneBounds` polygon boundaries on detected surfaces
- `MeshBounds` AABB for spatial mesh regions
- `RoomBounds` for room-level constraints
- No explicit "guardian" system but plane detection provides boundary awareness

## Anchor System

**Comprehensive anchoring**:
- Local (session-only)
- Persistent (device storage)
- Cloud (multi-user sharing)
- Confidence tracking and decay
- Entity attachment for content placement

## Mesh Reconstruction

**Real-time mesh mapping**:
- Block-based chunking
- LOD levels (LOW to ULTRA)
- Per-vertex classification
- Incremental/adaptive updates
- Ray-mesh intersection
- Physics/occlusion mesh extraction

## Dependencies

All modules depend on:
- `engine.core.math.vec` (Vec2, Vec3)
- `engine.core.math.quat` (Quat)
- `engine.xr.config.XR_CONFIG` (configuration constants)

## Missing/Placeholder Elements

1. **Cloud anchor actual networking**: `save_to_cloud()` and `resolve_cloud_anchor()` set state but no actual HTTP/cloud calls
2. **Native platform handles**: `_native_handle` attributes present but unused (ARKit/ARCore integration point)
3. **Sensor data processing**: `update()` methods are hooks for actual depth/camera data
4. **ML inference**: Scene understanding uses heuristics, not actual ML models

## Architecture Quality

- Clean separation of concerns (data classes, entities, managers)
- Consistent callback pattern across all modules
- Proper use of `__slots__` for memory efficiency
- Type hints throughout
- Comprehensive docstrings

## Recommendations

1. **Integration**: Modules are ready for platform-specific AR backend implementation
2. **Testing**: Mathematical algorithms (ray intersection, polygon area) should have unit tests
3. **Performance**: Consider spatial indexing (octree/BVH) for mesh raycasting at scale
4. **Cloud**: Implement actual cloud anchor service integration when needed
