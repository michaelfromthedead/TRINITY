"""XR Avatars module.

Provides full-body avatar representation for XR with inverse kinematics,
hand animation, face tracking, and calibration.

Key components:
- XRAvatar: Main avatar component with IK targets
- IKSolver: Abstract IK solver with FABRIK, CCD, TwoBone implementations
- AvatarHand: Hand/finger animation from controller or tracking
- FaceTracking: Face expression and lip sync
- AvatarCalibration: Player dimension calibration

Example usage:
    from engine.xr.avatars import XRAvatar, AvatarCalibration, AvatarHand

    # Create avatar with default height
    avatar = XRAvatar(player_height=1.8)

    # Calibrate from HMD position
    calibration = AvatarCalibration()
    calibration.quick_calibrate(hmd_position, left_hand, right_hand)

    # Update avatar from HMD and controller poses
    avatar.update_from_hmd(hmd_pos, hmd_rot)
    avatar.update_from_controllers(left_pos, left_rot, right_pos, right_rot)
    avatar.estimate_body()

    # Animate hands
    left_hand = AvatarHand("left")
    left_hand.update_from_controller(trigger_value, grip_value)
    left_hand.update(delta_time)
"""

from engine.xr.avatars.avatar import (
    AvatarVisibility,
    DisplayMode,
    IKTarget,
    PersonalSpace,
    XRAvatar,
    xr_avatar,
    xr_ik_target,
)

from engine.xr.avatars.ik_solver import (
    CCDSolver,
    FABRIKSolver,
    IKChain,
    IKJoint,
    IKSolver,
    IKSolverType,
    TwoBoneSolver,
    create_solver,
)

from engine.xr.avatars.hand_animator import (
    AvatarHand,
    FingerCurl,
    FingerName,
    HandPose,
    HandPoseType,
    PoseLibrary,
)

from engine.xr.avatars.face_tracking import (
    BlendShapeController,
    BlendShapeType,
    ExpressionType,
    EyeGazeData,
    FaceDrivingMode,
    FaceTracking,
    LipSyncData,
)

from engine.xr.avatars.calibration import (
    AvatarCalibration,
    CalibrationData,
    CalibrationState,
    CalibrationStep,
)

__all__ = [
    # Avatar
    "AvatarVisibility",
    "DisplayMode",
    "IKTarget",
    "PersonalSpace",
    "XRAvatar",
    "xr_avatar",
    "xr_ik_target",

    # IK Solver
    "CCDSolver",
    "FABRIKSolver",
    "IKChain",
    "IKJoint",
    "IKSolver",
    "IKSolverType",
    "TwoBoneSolver",
    "create_solver",

    # Hand Animation
    "AvatarHand",
    "FingerCurl",
    "FingerName",
    "HandPose",
    "HandPoseType",
    "PoseLibrary",

    # Face Tracking
    "BlendShapeController",
    "BlendShapeType",
    "ExpressionType",
    "EyeGazeData",
    "FaceDrivingMode",
    "FaceTracking",
    "LipSyncData",

    # Calibration
    "AvatarCalibration",
    "CalibrationData",
    "CalibrationState",
    "CalibrationStep",
]
