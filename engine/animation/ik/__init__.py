"""Inverse Kinematics subsystem for the Animation Layer.

This module provides various IK solvers for skeletal animation:

- **TwoBoneIK**: Analytical two-bone solver using law of cosines (arms/legs)
- **FABRIK**: Forward And Backward Reaching IK for chains of any length
- **CCD**: Cyclic Coordinate Descent for constrained chains
- **JacobianIK**: Jacobian-based methods (transpose, pseudoinverse, DLS)
- **FullBodyIK**: Multi-effector full body solver with balance
- **FootPlacement**: Terrain-adaptive foot IK

Example usage:

    from engine.animation.ik import TwoBoneIK, FABRIKChain, PositionGoal

    # Two-bone IK for arm
    arm_ik = TwoBoneIK(
        root_bone=shoulder_idx,
        mid_bone=elbow_idx,
        end_bone=wrist_idx
    )
    result = arm_ik.solve(shoulder_tf, elbow_tf, wrist_tf, target_pos)

    # FABRIK for spine
    spine_ik = FABRIKChain(
        bone_indices=[pelvis, spine1, spine2, chest, neck, head],
        tolerance=0.001,
        max_iterations=10
    )
    result = spine_ik.solve(positions, target)

    # Full body IK
    fbik = FullBodyIK(skeleton_mapping, tolerance=0.001)
    goals = [
        FullBodyIKGoal(hand_idx, target_pos, chain_type="left_arm"),
        FullBodyIKGoal(foot_idx, ground_pos, chain_type="left_leg"),
    ]
    result = fbik.solve(transforms, goals)
"""

# IK Goal definitions
from engine.animation.ik.ik_goal import (
    IKGoalType,
    IKGoal,
    PositionGoal,
    RotationGoal,
    LookAtGoal,
    PositionRotationGoal,
    PoleVectorGoal,
    ChainGoal,
    CenterOfMassGoal,
    IKGoalBlender,
    ik_goal,
    ik_chain,
)

# Two-bone analytical IK
from engine.animation.ik.two_bone import (
    TwoBoneIK,
    TwoBoneIKResult,
    TwoBoneIKConstraint,
)

# FABRIK solver
from engine.animation.ik.fabrik import (
    FABRIKChain,
    FABRIKResult,
    FABRIKMultiChain,
    JointConstraint,
    JointConstraintType,
)

# CCD solver
from engine.animation.ik.ccd import (
    CCDSolver,
    CCDResult,
    CCDRotationOrder,
    RotationLimit,
    CCDSolverWithWeights,
    ConstrainedCCDSolver,
)

# Joint limits (for CCD and other solvers)
from engine.animation.ik.joint_limits import (
    EulerOrder,
    JointLimit,
    EulerLimit,
    SwingTwistLimit,
    HingeLimit,
    quat_to_euler,
    euler_to_quat,
    create_elbow_limit,
    create_knee_limit,
    create_shoulder_limit,
    create_hip_limit,
)

# Jacobian-based IK
from engine.animation.ik.jacobian import (
    JacobianIK,
    JacobianResult,
    JacobianMethod,
    MultiTargetJacobianIK,
    Matrix,
)

# Full body IK
from engine.animation.ik.fullbody import (
    FullBodyIK,
    FullBodyIKGoal,
    FullBodyIKResult,
    SkeletonMapping,
    BodyPart,
    LookAtSolver,
)

# Foot placement
from engine.animation.ik.foot_placement import (
    FootPlacement,
    FootPlacementResult,
    FootData,
    FootState,
    FootPlacementAnimated,
    MultiLegFootPlacement,
    RaycastCallback,
)

# Configuration constants
from engine.animation.ik.config import (
    # Tolerances
    IK_DEFAULT_TOLERANCE,
    IK_TOLERANCE_TIGHT,
    IK_TOLERANCE_LOOSE,
    IK_ROTATION_TOLERANCE,
    # Iteration limits
    FABRIK_DEFAULT_MAX_ITERATIONS,
    CCD_DEFAULT_MAX_ITERATIONS,
    JACOBIAN_DEFAULT_MAX_ITERATIONS,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
    # Damping
    JACOBIAN_DLS_DAMPING,
    CCD_DEFAULT_DAMPING,
    # Soft IK
    SOFT_IK_DEFAULT_RATIO,
    SOFT_IK_DEFAULT_BLEND,
    # Foot placement
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
    FOOT_PLACEMENT_MAX_PELVIS_DROP,
    # Joint limits
    JOINT_MIN_BEND_ANGLE,
    JOINT_MAX_BEND_ANGLE,
    JOINT_DEFAULT_CONE_ANGLE,
    LOOK_AT_MAX_ANGLE,
)

__all__ = [
    # Goal types
    "IKGoalType",
    "IKGoal",
    "PositionGoal",
    "RotationGoal",
    "LookAtGoal",
    "PositionRotationGoal",
    "PoleVectorGoal",
    "ChainGoal",
    "CenterOfMassGoal",
    "IKGoalBlender",
    "ik_goal",
    "ik_chain",
    # Two-bone
    "TwoBoneIK",
    "TwoBoneIKResult",
    "TwoBoneIKConstraint",
    # FABRIK
    "FABRIKChain",
    "FABRIKResult",
    "FABRIKMultiChain",
    "JointConstraint",
    "JointConstraintType",
    # CCD
    "CCDSolver",
    "CCDResult",
    "CCDRotationOrder",
    "RotationLimit",
    "CCDSolverWithWeights",
    "ConstrainedCCDSolver",
    # Joint limits
    "EulerOrder",
    "JointLimit",
    "EulerLimit",
    "SwingTwistLimit",
    "HingeLimit",
    "quat_to_euler",
    "euler_to_quat",
    "create_elbow_limit",
    "create_knee_limit",
    "create_shoulder_limit",
    "create_hip_limit",
    # Jacobian
    "JacobianIK",
    "JacobianResult",
    "JacobianMethod",
    "MultiTargetJacobianIK",
    "Matrix",
    # Full body
    "FullBodyIK",
    "FullBodyIKGoal",
    "FullBodyIKResult",
    "SkeletonMapping",
    "BodyPart",
    "LookAtSolver",
    # Foot placement
    "FootPlacement",
    "FootPlacementResult",
    "FootData",
    "FootState",
    "FootPlacementAnimated",
    "MultiLegFootPlacement",
    "RaycastCallback",
    # Config constants
    "IK_DEFAULT_TOLERANCE",
    "IK_TOLERANCE_TIGHT",
    "IK_TOLERANCE_LOOSE",
    "IK_ROTATION_TOLERANCE",
    "FABRIK_DEFAULT_MAX_ITERATIONS",
    "CCD_DEFAULT_MAX_ITERATIONS",
    "JACOBIAN_DEFAULT_MAX_ITERATIONS",
    "FULLBODY_DEFAULT_MAX_ITERATIONS",
    "JACOBIAN_DLS_DAMPING",
    "CCD_DEFAULT_DAMPING",
    "SOFT_IK_DEFAULT_RATIO",
    "SOFT_IK_DEFAULT_BLEND",
    "FOOT_PLACEMENT_RAY_LENGTH",
    "FOOT_PLACEMENT_FOOT_HEIGHT",
    "FOOT_PLACEMENT_BLEND_SPEED",
    "FOOT_PLACEMENT_MAX_PELVIS_DROP",
    "JOINT_MIN_BEND_ANGLE",
    "JOINT_MAX_BEND_ANGLE",
    "JOINT_DEFAULT_CONE_ANGLE",
    "LOOK_AT_MAX_ANGLE",
]
