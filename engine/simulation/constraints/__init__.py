"""
Constraints Module for AI Game Engine.

This module provides various joint and constraint types for physics simulation:
- Fixed joints (welding)
- Hinge/Revolute joints
- Slider/Prismatic joints
- Ball/Spherical joints
- Spring joints
- Distance joints
- D6 (configurable) joints
- Contact constraints

Also includes:
- Motor helpers
- Limit helpers
- Joint base class
"""

from .joint_base import (
    Joint,
    JointState,
    JointBreakEvent,
)
from .joint_fixed import FixedJoint
from .joint_hinge import HingeJoint
from .joint_slider import SliderJoint
from .joint_ball import BallJoint
from .joint_spring import SpringJoint
from .joint_distance import DistanceJoint
from .joint_d6 import D6Joint, D6MotionType, D6Axis
from .joint_motors import Motor, MotorMode, compute_motor_impulse
from .joint_limits import (
    LinearLimit,
    AngularLimit,
    compute_limit_impulse,
    LimitState,
)
from .contact_constraint import (
    ContactConstraint,
    ContactPoint,
    ContactManifold,
    compute_contact_jacobian,
)

__all__ = [
    # Base
    "Joint",
    "JointState",
    "JointBreakEvent",
    # Joints
    "FixedJoint",
    "HingeJoint",
    "SliderJoint",
    "BallJoint",
    "SpringJoint",
    "DistanceJoint",
    "D6Joint",
    "D6MotionType",
    "D6Axis",
    # Motors
    "Motor",
    "MotorMode",
    "compute_motor_impulse",
    # Limits
    "LinearLimit",
    "AngularLimit",
    "compute_limit_impulse",
    "LimitState",
    # Contacts
    "ContactConstraint",
    "ContactPoint",
    "ContactManifold",
    "compute_contact_jacobian",
]
