# PHASE 7 ARCHITECTURE: Spatial AR Features

## Phase Overview

Phase 7 implements the spatial awareness features required for AR and mixed reality applications. This phase covers spatial anchors, plane detection, mesh mapping, scene understanding, and image/object tracking. These features enable virtual content to interact meaningfully with the physical environment.

## Architectural Decisions

### ADR-XR-060: Spatial Anchor Persistence Model

**Context**: AR content must persist across sessions and potentially share across users.

**Decision**: Implement three anchor persistence levels:
1. **LOCAL**: Session-only, lost on app close
2. **PERSISTENT**: Device-stored, restored on restart
3. **CLOUD**: Cloud-stored, shareable across users/devices

**Consequences**:
- Local anchors have zero latency
- Persistent anchors survive app restarts
- Cloud anchors enable shared AR experiences

### ADR-XR-061: Anchor Tracking Quality Model

**Context**: Anchor tracking degrades when visual features change or device moves.

**Decision**: Track confidence and state per anchor:
- **States**: TRACKING, LIMITED, PAUSED, LOST, NOT_TRACKING
- **Confidence**: 0-1 float, decays over time if not reinforced
- **Callbacks**: Notify on state/confidence changes

**Consequences**:
- Application knows when anchor is unreliable
- Confidence decay encourages re-localization
- Callbacks enable graceful degradation

### ADR-XR-062: Plane Detection Pipeline

**Context**: AR needs to detect horizontal and vertical surfaces for content placement.

**Decision**: Implement hierarchical plane classification:
1. **Orientation**: HORIZONTAL_UP, HORIZONTAL_DOWN, VERTICAL, ARBITRARY
2. **Semantic**: FLOOR, CEILING, WALL, TABLE, SEAT, DOOR, WINDOW

With configurable detection mode:
- **HORIZONTAL_ONLY**: Floors and tables (faster)
- **VERTICAL_ONLY**: Walls (faster)
- **ALL**: Both orientations (complete)

**Consequences**:
- Semantic labels enable intelligent placement
- Mode selection trades coverage for speed
- Plane merging reduces duplicates

### ADR-XR-063: Mesh Mapping Architecture

**Context**: Real-time environment reconstruction requires efficient updates.

**Decision**: Implement block-based mesh chunking:
- **Block size**: 1m^3 chunks for efficient updates
- **LOD levels**: LOW, MEDIUM, HIGH, ULTRA based on distance
- **Update modes**: FULL, INCREMENTAL, ADAPTIVE

**Consequences**:
- Only changed blocks updated per frame
- Distant geometry uses lower LOD
- Adaptive mode balances quality vs performance

### ADR-XR-064: Scene Understanding Heuristics

**Context**: ML-based scene understanding may not be available on all platforms.

**Decision**: Implement heuristic-based scene classification:
1. Detect objects from mesh segments and planes
2. Classify rooms by object composition
3. Estimate lighting from camera/environment

With hooks for ML override when available.

**Consequences**:
- Works without ML models
- ML can improve accuracy when available
- Room classification enables context-aware behavior

### ADR-XR-065: Image Tracking Strategy

**Context**: 2D image markers enable AR content anchoring.

**Decision**: Implement reference image database with:
- **Physical dimensions**: Known size for scale
- **Tracking modes**: CONTINUOUS, ONCE, ADAPTIVE
- **Extended tracking**: Maintain pose when image not visible

**Consequences**:
- Multiple images trackable simultaneously
- Physical size enables world-scale positioning
- Extended tracking prevents pop-in on occlusion

### ADR-XR-066: Object Tracking Strategy

**Context**: 3D object recognition enables interaction with physical objects.

**Decision**: Implement reference object database with:
- **Feature data**: 3D point cloud or mesh
- **Scale awareness**: Match tracked object scale to reference
- **Occlusion support**: Objects can occlude virtual content

**Consequences**:
- Real objects become interactive
- Scale matching handles size variation
- Occlusion improves realism

## Component Specifications

### Spatial Anchor System

```
SpatialAnchor (Component)
├── Identity
│   ├── anchor_id: str (UUID)
│   ├── anchor_type: AnchorType (LOCAL, PERSISTENT, CLOUD)
│   └── name: Optional[str]
├── Pose
│   ├── position: Vec3
│   ├── orientation: Quat
│   ├── timestamp: float
│   └── get_pose() -> AnchorPose
├── Tracking
│   ├── tracking_state: AnchorTrackingState
│   ├── confidence: float (0-1)
│   ├── last_tracked_time: float
│   └── confidence_decay_rate: float
├── Persistence
│   ├── persistence_state: AnchorPersistenceState
│   ├── save_to_device() -> bool
│   ├── save_to_cloud(config) -> bool
│   └── cloud_anchor_id: Optional[str]
├── Attachment
│   ├── attached_entities: List[EntityID]
│   ├── attach_entity(entity_id) -> None
│   └── detach_entity(entity_id) -> None
└── Callbacks
    ├── on_tracking_state_changed: Callable
    ├── on_confidence_changed: Callable
    └── on_persistence_completed: Callable

AnchorManager (Singleton)
├── Lifecycle
│   ├── create_anchor(position, orientation, type) -> SpatialAnchor
│   ├── destroy_anchor(anchor_id) -> bool
│   └── update(delta_time) -> None (decay confidence)
├── Persistence
│   ├── load_persistent_anchors() -> List[SpatialAnchor]
│   ├── resolve_cloud_anchor(cloud_id) -> SpatialAnchor
│   └── clear_persistent_anchors() -> None
├── Queries
│   ├── get_anchor(anchor_id) -> Optional[SpatialAnchor]
│   ├── get_anchors_near(position, radius) -> List[SpatialAnchor]
│   └── get_all_anchors() -> List[SpatialAnchor]
└── Integration
    └── update_from_runtime(anchor_updates) -> None

AnchorType Enum
├── LOCAL
├── PERSISTENT
└── CLOUD

AnchorTrackingState Enum
├── UNKNOWN
├── TRACKING
├── LIMITED
├── PAUSED
├── LOST
└── NOT_TRACKING

CloudAnchorConfig
├── expire_days: int
├── privacy: CloudPrivacy
└── allowed_users: List[str]
```

### Plane Detection System

```
DetectedPlane (Component)
├── Identity
│   ├── plane_id: str (UUID)
│   ├── plane_type: PlaneType
│   └── plane_orientation: PlaneOrientation
├── Geometry
│   ├── center: Vec3
│   ├── normal: Vec3
│   ├── bounds: PlaneBounds (2D polygon)
│   ├── width: float
│   ├── height: float
│   └── area: float (calculated via shoelace)
├── Tracking
│   ├── tracking_state: PlaneTrackingState
│   ├── last_updated_time: float
│   └── is_subsumed: bool
├── Transforms
│   ├── world_to_local(point) -> Vec3
│   └── local_to_world(point) -> Vec3
├── Hit Testing
│   ├── contains_point(point) -> bool
│   └── ray_intersect(origin, direction) -> Optional[Vec3]
└── Callbacks
    ├── on_geometry_updated: Callable
    ├── on_tracking_state_changed: Callable
    └── on_merged: Callable

PlaneDetector (Singleton)
├── Configuration
│   ├── detection_mode: PlaneDetectionMode
│   ├── merge_threshold: float
│   └── min_area: float
├── Detection
│   ├── start_detection() -> None
│   ├── stop_detection() -> None
│   ├── is_detecting: bool
│   └── update_from_runtime(plane_data) -> None
├── Queries
│   ├── get_plane(plane_id) -> Optional[DetectedPlane]
│   ├── get_planes_by_type(type) -> List[DetectedPlane]
│   ├── get_all_planes() -> List[DetectedPlane]
│   └── raycast(origin, direction) -> Optional[PlaneHit]
├── Placement
│   ├── find_placement_surface(position, radius) -> Optional[DetectedPlane]
│   └── get_floor_plane() -> Optional[DetectedPlane]
└── Callbacks
    ├── on_plane_added: Callable
    ├── on_plane_updated: Callable
    └── on_plane_removed: Callable

PlaneType Enum
├── FLOOR
├── CEILING
├── WALL
├── TABLE
├── SEAT
├── DOOR
└── WINDOW

PlaneOrientation Enum
├── HORIZONTAL_UP
├── HORIZONTAL_DOWN
├── VERTICAL
└── ARBITRARY

PlaneBounds
├── vertices: List[Vec2] (local 2D polygon)
├── get_area() -> float (shoelace formula)
├── contains_point(local_point) -> bool (ray casting)
└── get_corners() -> List[Vec3] (world space)
```

### Mesh Mapping System

```
SpatialMesh (Component)
├── Identity
│   ├── mesh_id: str (UUID)
│   └── version: int
├── Geometry
│   ├── vertices: List[MeshVertex]
│   ├── triangles: List[MeshTriangle]
│   ├── bounds: MeshBounds (AABB)
│   └── vertex_count: int
├── Classification
│   ├── vertex_classifications: List[MeshClassification]
│   └── has_classification: bool
├── LOD
│   ├── lod_level: MeshLODLevel
│   └── lod_distance: float
├── Operations
│   ├── raycast(origin, direction) -> Optional[MeshHit]
│   ├── optimize() -> None (remove degenerates)
│   └── extract_physics_mesh() -> MeshData
└── Updates
    ├── update_from_block(block_data) -> None
    └── is_dirty: bool

SpatialMeshManager (Singleton)
├── Configuration
│   ├── update_mode: MeshUpdateMode
│   ├── max_distance: float
│   ├── lod_distances: Dict[MeshLODLevel, float]
│   └── block_size: float (default 1m)
├── Lifecycle
│   ├── start_mapping() -> None
│   ├── stop_mapping() -> None
│   ├── is_mapping: bool
│   └── update(observer_position) -> None
├── Queries
│   ├── get_mesh(mesh_id) -> Optional[SpatialMesh]
│   ├── get_meshes_in_bounds(bounds) -> List[SpatialMesh]
│   ├── get_all_meshes() -> List[SpatialMesh]
│   └── raycast(origin, direction) -> Optional[MeshHit]
├── Cleanup
│   ├── cleanup_distant_blocks(observer_position) -> None
│   └── clear_all_meshes() -> None
├── Export
│   ├── extract_physics_mesh(bounds) -> MeshData
│   └── extract_occlusion_mesh() -> MeshData
└── Callbacks
    ├── on_mesh_added: Callable
    ├── on_mesh_updated: Callable
    └── on_mesh_removed: Callable

MeshUpdateMode Enum
├── NONE
├── FULL (re-mesh entire volume)
├── INCREMENTAL (update changed blocks)
└── ADAPTIVE (quality based on distance)

MeshLODLevel Enum
├── LOW (distant)
├── MEDIUM
├── HIGH
└── ULTRA (close)

MeshClassification Enum
├── UNKNOWN
├── WALL
├── FLOOR
├── CEILING
├── TABLE
├── SEAT
└── UNCLASSIFIED
```

### Scene Understanding System

```
SceneUnderstanding (Singleton)
├── Room Classification
│   ├── room_type: RoomType
│   ├── room_bounds: RoomBounds
│   ├── classify_room() -> RoomType
│   └── get_room_confidence() -> float
├── Semantic Regions
│   ├── regions: List[SemanticRegion]
│   ├── get_region_at(position) -> Optional[SemanticRegion]
│   └── get_regions_by_label(label) -> List[SemanticRegion]
├── Scene Objects
│   ├── objects: List[SceneObject]
│   ├── get_object(object_id) -> Optional[SceneObject]
│   └── get_objects_by_label(label) -> List[SceneObject]
├── Human Segmentation
│   ├── humans: List[HumanSegment]
│   ├── is_human_at(position) -> bool
│   └── get_human_occlusion_mask() -> Texture
├── Light Estimation
│   ├── light_estimate: LightEstimate
│   ├── ambient_intensity: float
│   ├── main_light_direction: Vec3
│   └── spherical_harmonics: List[float]
├── Placement Helpers
│   ├── find_floor_position(screen_point) -> Optional[Vec3]
│   ├── find_wall_position(screen_point) -> Optional[Vec3]
│   └── find_table_position(screen_point) -> Optional[Vec3]
└── Integration
    └── update_from_runtime(scene_data) -> None

RoomType Enum
├── LIVING_ROOM
├── BEDROOM
├── KITCHEN
├── BATHROOM
├── OFFICE
├── HALLWAY
├── OUTDOOR
└── UNKNOWN

SemanticRegion
├── region_id: str
├── label: SemanticLabel
├── bounds: AABB
├── confidence: float
└── center: Vec3

SemanticLabel Enum (19 labels)
├── FLOOR, WALL, CEILING, DOOR, WINDOW
├── TABLE, CHAIR, BED, COUCH, DESK
├── SHELF, CABINET, APPLIANCE
├── PLANT, SCREEN, LIGHT
├── PERSON, PET
└── UNKNOWN

SceneObject
├── object_id: str
├── label: SemanticLabel
├── position: Vec3
├── bounds: OBB
├── is_movable: bool
└── is_interactable: bool

LightEstimate
├── ambient_intensity: float
├── ambient_color: Color
├── main_light_direction: Vec3
├── main_light_intensity: float
├── main_light_color: Color
└── spherical_harmonics: List[float] (9 coefficients)
```

### Image Tracking System

```
ImageTarget (Component)
├── Reference
│   ├── reference_id: str
│   ├── reference_image: ImageReference
│   └── physical_size: Vec2 (meters)
├── Tracking
│   ├── tracking_state: ImageTrackingState
│   ├── tracking_mode: TrackingMode
│   ├── is_tracking: bool
│   └── tracking_timeout: float
├── Pose
│   ├── position: Vec3
│   ├── orientation: Quat
│   ├── last_seen_time: float
│   └── get_corner_positions() -> List[Vec3]
├── Extended Tracking
│   ├── extended_tracking_enabled: bool
│   ├── extended_tracking_timeout: float
│   └── is_extended_tracking: bool
└── Callbacks
    ├── on_tracking_started: Callable
    ├── on_tracking_updated: Callable
    └── on_tracking_lost: Callable

ImageTracker (Singleton)
├── Database
│   ├── add_reference(image, physical_size) -> str
│   ├── remove_reference(reference_id) -> bool
│   ├── get_reference(reference_id) -> Optional[ImageReference]
│   └── get_all_references() -> List[ImageReference]
├── Configuration
│   ├── max_tracked_images: int
│   ├── tracking_mode: TrackingMode
│   └── extended_tracking_timeout: float
├── Lifecycle
│   ├── start_tracking() -> None
│   ├── stop_tracking() -> None
│   ├── is_tracking: bool
│   └── update_from_runtime(tracking_data) -> None
├── Queries
│   ├── get_target(target_id) -> Optional[ImageTarget]
│   ├── get_active_targets() -> List[ImageTarget]
│   └── get_targets_for_reference(reference_id) -> List[ImageTarget]
└── Callbacks
    ├── on_target_found: Callable
    ├── on_target_updated: Callable
    └── on_target_lost: Callable

ImageReference
├── reference_id: str
├── image_data: bytes
├── width_pixels: int
├── height_pixels: int
├── physical_width: float
├── physical_height: float
└── feature_count: int

TrackingMode Enum
├── CONTINUOUS (track every frame)
├── ONCE (stop after first detection)
└── ADAPTIVE (reduce updates when stable)

ImageTrackingState Enum
├── NONE
├── DETECTING
├── TRACKING
├── LIMITED
├── EXTENDED
└── LOST

@ar_trackable Decorator
└── Marks classes as AR trackable content
```

### Object Tracking System

```
TrackedObject (Component)
├── Reference
│   ├── reference_id: str
│   ├── reference_object: ObjectReference
│   └── expected_scale: float
├── Tracking
│   ├── tracking_state: ObjectTrackingState
│   ├── tracking_quality: ObjectTrackingQuality
│   └── is_tracking: bool
├── Pose
│   ├── position: Vec3
│   ├── orientation: Quat
│   ├── scale: float
│   └── get_world_bounds() -> ObjectBounds
├── Occlusion
│   ├── occlusion_enabled: bool
│   └── get_occlusion_mesh() -> MeshData
└── Callbacks
    ├── on_tracking_started: Callable
    ├── on_tracking_updated: Callable
    └── on_tracking_lost: Callable

ObjectTracker (Singleton)
├── Database
│   ├── add_reference(object_data) -> str
│   ├── remove_reference(reference_id) -> bool
│   └── get_all_references() -> List[ObjectReference]
├── Configuration
│   ├── max_tracked_objects: int
│   └── scale_tolerance: float
├── Lifecycle
│   ├── start_tracking() -> None
│   ├── stop_tracking() -> None
│   └── update_from_runtime(tracking_data) -> None
├── Queries
│   ├── get_object(object_id) -> Optional[TrackedObject]
│   └── get_active_objects() -> List[TrackedObject]
└── Callbacks
    ├── on_object_found: Callable
    └── on_object_lost: Callable

ObjectReference
├── reference_id: str
├── feature_data: bytes (3D point cloud)
├── bounds: ObjectBounds
├── nominal_scale: float
└── feature_count: int

ObjectBounds
├── center: Vec3
├── half_extents: Vec3
├── orientation: Quat
├── get_corners() -> List[Vec3]
└── contains_point(point) -> bool
```

## Integration Points

### Dependencies (Incoming)
- Phase 1: Runtime provides spatial tracking data
- Renderer: Occlusion mesh, light estimation
- `engine.core.math`: Vec3, Quat, Transform

### Dependents (Outgoing)
- Application: Consumes anchors, planes, scene objects
- Physics: Mesh mapping for collision
- Rendering: Light estimation, human occlusion

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      XR Runtime                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Depth   │  │  Camera  │  │   IMU    │  │   ML     │    │
│  │  Sensor  │  │  Feed    │  │  Data    │  │  Models  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼────────────┼────────────┼─────────────┘
        │             │            │            │
        ▼             ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Spatial Features                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │    Anchor     │  │     Plane     │  │     Mesh      │    │
│  │   Manager     │  │   Detector    │  │    Mapper     │    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘    │
│          │                  │                  │             │
│  ┌───────┴──────────────────┴──────────────────┴───────┐    │
│  │                 Scene Understanding                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │    │
│  │  │  Room   │  │Semantic │  │  Light  │  │ Human   │ │    │
│  │  │  Class  │  │ Regions │  │  Est.   │  │ Segment │ │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐                       │
│  │    Image      │  │    Object     │                       │
│  │   Tracker     │  │    Tracker    │                       │
│  └───────────────┘  └───────────────┘                       │
└──────────────────────────────────────────────────────────────┘
        │                       │
        ▼                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      Application                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Content   │  │  Physics    │  │  Rendering  │          │
│  │  Placement  │  │  Collision  │  │  Occlusion  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## Performance Requirements

| Component | Update Rate | CPU Budget |
|-----------|-------------|------------|
| Anchor Update | 60 Hz | <0.5ms |
| Plane Detection | 30 Hz | <2ms |
| Mesh Mapping | 15 Hz | <5ms |
| Scene Understanding | 10 Hz | <5ms |
| Image Tracking | 30 Hz | <2ms |
| Object Tracking | 30 Hz | <3ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Anchor drift over time | High | Medium | Confidence decay, re-localization |
| Plane detection noise | Medium | Medium | Minimum area threshold, smoothing |
| Mesh gaps/holes | Medium | Medium | Fill algorithm, LOD distance |
| Cloud anchor latency | High | Medium | Async resolve, local fallback |
