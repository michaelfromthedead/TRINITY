"""Spatial Understanding (AR) module for XR.

Provides AR spatial understanding capabilities including:
- Spatial anchors (local, persistent, cloud)
- Plane detection (floor, ceiling, wall, table, seat)
- Real-time mesh mapping and spatial reconstruction
- Semantic scene understanding
- Image target tracking
- 3D object tracking

Example usage:
    from engine.xr.spatial import (
        SpatialAnchor, AnchorManager, AnchorType,
        PlaneDetector, PlaneType,
        SpatialMeshManager,
        SceneUnderstanding,
        ImageTracker, ImageTarget,
        ObjectTracker, TrackedObject,
    )

    # Create anchor manager
    anchor_manager = AnchorManager()
    anchor = anchor_manager.create_anchor(
        position=Vec3(0, 0, -1),
        orientation=Quat.identity(),
        anchor_type=AnchorType.PERSISTENT,
    )

    # Start plane detection
    detector = PlaneDetector()
    detector.start()
    floor_planes = detector.get_floor_planes()

    # Use scene understanding
    scene = SceneUnderstanding()
    scene.start()
    room = scene.classify_room()
"""

# Anchor module
from engine.xr.spatial.anchor import (
    AnchorManager,
    AnchorPersistenceState,
    AnchorPose,
    AnchorTrackingState,
    AnchorType,
    CloudAnchorConfig,
    SpatialAnchor,
    spatial_anchor,
)

# Plane detection module
from engine.xr.spatial.plane_detection import (
    DetectedPlane,
    PlaneBounds,
    PlaneAlignment,
    PlaneDetectionConfig,
    PlaneDetector,
    PlaneGeometry,
    PlaneOrientation,
    PlaneTrackingState,
    PlaneType,
)

# Mesh mapping module
from engine.xr.spatial.mesh_mapping import (
    MeshBlock,
    MeshBounds,
    MeshClassification,
    MeshLODLevel,
    MeshManagerConfig,
    MeshTriangle,
    MeshUpdateMode,
    MeshVertex,
    SpatialMesh,
    SpatialMeshManager,
)

# Scene understanding module
from engine.xr.spatial.scene_understanding import (
    HumanSegment,
    LightEstimate,
    OcclusionMode,
    RoomBounds,
    RoomType,
    SceneObject,
    SceneUnderstanding,
    SceneUnderstandingConfig,
    SemanticLabel,
    SemanticRegion,
)

# Image tracking module
from engine.xr.spatial.image_tracking import (
    ImageReference,
    ImageTarget,
    ImageTargetPose,
    ImageTracker,
    ImageTrackerConfig,
    ImageTrackingState,
    TrackingMode,
    ar_trackable,
)

# Object tracking module
from engine.xr.spatial.object_tracking import (
    ObjectBounds,
    ObjectPose,
    ObjectReference,
    ObjectTracker,
    ObjectTrackerConfig,
    ObjectTrackingQuality,
    ObjectTrackingState,
    TrackedObject,
)

__all__ = [
    # Anchor
    "AnchorManager",
    "AnchorPersistenceState",
    "AnchorPose",
    "AnchorTrackingState",
    "AnchorType",
    "CloudAnchorConfig",
    "SpatialAnchor",
    "spatial_anchor",
    # Plane Detection
    "DetectedPlane",
    "PlaneBounds",
    "PlaneAlignment",
    "PlaneDetectionConfig",
    "PlaneDetector",
    "PlaneGeometry",
    "PlaneOrientation",
    "PlaneTrackingState",
    "PlaneType",
    # Mesh Mapping
    "MeshBlock",
    "MeshBounds",
    "MeshClassification",
    "MeshLODLevel",
    "MeshManagerConfig",
    "MeshTriangle",
    "MeshUpdateMode",
    "MeshVertex",
    "SpatialMesh",
    "SpatialMeshManager",
    # Scene Understanding
    "HumanSegment",
    "LightEstimate",
    "OcclusionMode",
    "RoomBounds",
    "RoomType",
    "SceneObject",
    "SceneUnderstanding",
    "SceneUnderstandingConfig",
    "SemanticLabel",
    "SemanticRegion",
    # Image Tracking
    "ImageReference",
    "ImageTarget",
    "ImageTargetPose",
    "ImageTracker",
    "ImageTrackerConfig",
    "ImageTrackingState",
    "TrackingMode",
    "ar_trackable",
    # Object Tracking
    "ObjectBounds",
    "ObjectPose",
    "ObjectReference",
    "ObjectTracker",
    "ObjectTrackerConfig",
    "ObjectTrackingQuality",
    "ObjectTrackingState",
    "TrackedObject",
]
