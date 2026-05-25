"""
Motion Matching Animation Subsystem.

This package provides a complete motion matching implementation for
high-quality character animation with minimal authoring.

Motion matching is a data-driven animation technique that searches a
database of animation poses to find the best match for the current
character state and desired trajectory.

Modules:
    database: Motion database storage and serialization
    features: Feature extraction for pose/trajectory matching
    search: Database search algorithms (brute force, KD-tree, LSH)
    transition: Inertialization-based smooth transitions
    context: Runtime controller and state management
    annotation: Clip tagging and contact detection

Basic Usage:
    from engine.animation.motionmatching import (
        MotionDatabase,
        MotionMatchingController,
        FeatureExtractor,
        build_database,
    )

    # Build database from animation clips
    extractor = FeatureExtractor()
    database = build_database(clips, extractor)

    # Create controller
    controller = MotionMatchingController(database)
    controller.start()

    # Update each frame
    pose = controller.update(input_direction, dt)

Architecture:
    The motion matching system follows a standard pipeline:

    1. Feature Extraction: Extract pose features (bone positions,
       velocities) and trajectory features (future positions, facings)

    2. Database Search: Find the best matching frame in the database
       using weighted cost function

    3. Transition: Smoothly blend from current pose to new match using
       inertialization (spring-based blending)

    4. Output: Final pose for skinning/rendering

Performance:
    - Brute force search: O(n) - good for small databases
    - KD-tree search: O(log n) average - good for medium databases
    - LSH search: O(1) average - good for large databases (approximate)

References:
    - "Motion Matching and The Road to Next-Gen Animation" (GDC 2016)
    - "Inertialization: High-Performance Animation Transitions" (GDC 2018)
"""

from engine.animation.motionmatching.database import (
    MotionDatabase,
    DatabaseEntry,
    ClipMetadata,
    NormalizationStats,
    QuantizationLevel,
    build_database,
    merge_databases,
    motion_matching,
)

from engine.animation.motionmatching.features import (
    FeatureSet,
    FeatureConfig,
    FeatureExtractor,
    FeatureNormalizer,
    FeatureType,
    FeatureWeights,
    BoneData,
    TrajectoryPoint,
    FootContact,
)

from engine.animation.motionmatching.search import (
    MotionSearch,
    SearchConfig,
    SearchResult,
    SearchMethod,
    compute_cost,
    compute_cost_vectorized,
    KDTree,
    LSHIndex,
)

from engine.animation.motionmatching.transition import (
    MotionTransition,
    TransitionConfig,
    BlendMode,
    InertializationBlender,
    InertializationOffset,
    Pose,
    BoneTransform,
    FootSlidingCorrector,
    quaternion_slerp,
    quaternion_multiply,
    quaternion_inverse,
)

from engine.animation.motionmatching.context import (
    MotionMatchingController,
    MotionContext,
    ControllerConfig,
    ControllerState,
    DesiredTrajectory,
    IdleDetector,
    TrajectoryBuilder,
)

from engine.animation.motionmatching.annotation import (
    AnnotatedClip,
    MotionTag,
    TagType,
    ContactAnnotation,
    auto_detect_contacts,
    auto_detect_locomotion_tags,
    auto_detect_turn_tags,
    auto_detect_all_tags,
    merge_overlapping_tags,
    filter_tags_by_duration,
)

__all__ = [
    # Database
    'MotionDatabase',
    'DatabaseEntry',
    'ClipMetadata',
    'NormalizationStats',
    'QuantizationLevel',
    'build_database',
    'merge_databases',
    'motion_matching',

    # Features
    'FeatureSet',
    'FeatureConfig',
    'FeatureExtractor',
    'FeatureNormalizer',
    'FeatureType',
    'FeatureWeights',
    'BoneData',
    'TrajectoryPoint',
    'FootContact',

    # Search
    'MotionSearch',
    'SearchConfig',
    'SearchResult',
    'SearchMethod',
    'compute_cost',
    'compute_cost_vectorized',
    'KDTree',
    'LSHIndex',

    # Transition
    'MotionTransition',
    'TransitionConfig',
    'BlendMode',
    'InertializationBlender',
    'InertializationOffset',
    'Pose',
    'BoneTransform',
    'FootSlidingCorrector',
    'quaternion_slerp',
    'quaternion_multiply',
    'quaternion_inverse',

    # Context
    'MotionMatchingController',
    'MotionContext',
    'ControllerConfig',
    'ControllerState',
    'DesiredTrajectory',
    'IdleDetector',
    'TrajectoryBuilder',

    # Annotation
    'AnnotatedClip',
    'MotionTag',
    'TagType',
    'ContactAnnotation',
    'auto_detect_contacts',
    'auto_detect_locomotion_tags',
    'auto_detect_turn_tags',
    'auto_detect_all_tags',
    'merge_overlapping_tags',
    'filter_tags_by_duration',
]
