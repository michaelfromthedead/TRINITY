"""IK chain setup, solver configuration, and effectors.

Provides tools for setting up inverse kinematics chains, configuring
IK solvers, and managing effectors and constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.core.math import Quat, Transform, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class IKChainType(Enum):
    """Types of IK chains."""

    LIMB = auto()       # Two-bone limb (arm, leg)
    SPINE = auto()      # Spine/tail chain
    CHAIN = auto()      # General multi-bone chain
    FULL_BODY = auto()  # Full body IK


class IKSolverType(Enum):
    """Types of IK solvers."""

    TWO_BONE = auto()   # Analytical two-bone solver
    FABRIK = auto()     # FABRIK iterative solver
    CCD = auto()        # Cyclic Coordinate Descent
    JACOBIAN = auto()   # Jacobian-based solver
    FULL_BODY = auto()  # Full body solver


class IKConstraintType(Enum):
    """Types of IK constraints."""

    NONE = auto()           # No constraint
    HINGE = auto()          # Single-axis rotation (knee, elbow)
    BALL_SOCKET = auto()    # Cone constraint
    TWIST_LIMIT = auto()    # Twist rotation limit
    ANGLE_LIMIT = auto()    # General angle limit


# =============================================================================
# IK BONE
# =============================================================================


@dataclass
class IKBone:
    """A bone in an IK chain.

    Attributes:
        bone_name: Name of the bone
        bone_index: Index in skeleton
        length: Bone length
        constraint_type: Type of constraint
        constraint_axis: Axis for hinge constraint
        min_angle: Minimum angle (radians)
        max_angle: Maximum angle (radians)
        stiffness: Bone stiffness (0-1)
    """

    bone_name: str
    bone_index: int
    length: float = 0.0
    constraint_type: IKConstraintType = IKConstraintType.NONE
    constraint_axis: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))
    min_angle: float = -math.pi
    max_angle: float = math.pi
    stiffness: float = 0.0

    def __post_init__(self) -> None:
        if not self.bone_name:
            raise ValueError("Bone name cannot be empty")
        if self.bone_index < 0:
            raise ValueError(f"Bone index must be >= 0, got {self.bone_index}")

    def is_within_limits(self, angle: float) -> bool:
        """Check if angle is within limits."""
        return self.min_angle <= angle <= self.max_angle

    def clamp_angle(self, angle: float) -> float:
        """Clamp angle to limits."""
        return max(self.min_angle, min(self.max_angle, angle))


# =============================================================================
# IK EFFECTOR
# =============================================================================


@dataclass
class IKEffector:
    """An IK end effector.

    Attributes:
        name: Effector name
        bone_name: Bone this effector is attached to
        position_weight: Weight for position solving (0-1)
        rotation_weight: Weight for rotation solving (0-1)
        target_position: Target world position
        target_rotation: Target world rotation
        offset: Local offset from bone
    """

    name: str
    bone_name: str
    position_weight: float = 1.0
    rotation_weight: float = 0.0
    target_position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    target_rotation: Quat = field(default_factory=Quat.identity)
    offset: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Effector name cannot be empty")
        if not self.bone_name:
            raise ValueError("Bone name cannot be empty")

    def set_target(self, position: Vec3, rotation: Optional[Quat] = None) -> None:
        """Set target position and optional rotation."""
        self.target_position = position
        if rotation is not None:
            self.target_rotation = rotation


# =============================================================================
# IK POLE VECTOR
# =============================================================================


@dataclass
class IKPoleVector:
    """Pole vector for controlling IK plane orientation.

    Attributes:
        name: Pole vector name
        position: World position of pole vector
        weight: Influence weight (0-1)
    """

    name: str
    position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Pole vector name cannot be empty")


# =============================================================================
# IK CONSTRAINT
# =============================================================================


@dataclass
class IKConstraint:
    """A constraint applied to IK solving.

    Attributes:
        constraint_type: Type of constraint
        bone_name: Bone to constrain
        axis: Constraint axis
        min_value: Minimum value
        max_value: Maximum value
        enabled: Whether constraint is active
    """

    constraint_type: IKConstraintType
    bone_name: str
    axis: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    min_value: float = -math.pi
    max_value: float = math.pi
    enabled: bool = True

    def apply(self, rotation: Quat, reference_axis: Vec3) -> Quat:
        """Apply constraint to a rotation.

        Args:
            rotation: Input rotation
            reference_axis: Reference axis for constraint

        Returns:
            Constrained rotation
        """
        if not self.enabled:
            return rotation

        if self.constraint_type == IKConstraintType.NONE:
            return rotation

        if self.constraint_type == IKConstraintType.HINGE:
            # Project rotation onto hinge axis
            return self._apply_hinge(rotation, reference_axis)

        if self.constraint_type == IKConstraintType.BALL_SOCKET:
            # Constrain to cone
            return self._apply_cone(rotation, reference_axis)

        if self.constraint_type == IKConstraintType.ANGLE_LIMIT:
            # Clamp rotation angle
            return self._apply_angle_limit(rotation)

        return rotation

    def _apply_hinge(self, rotation: Quat, reference_axis: Vec3) -> Quat:
        """Apply hinge constraint."""
        # Get rotation around hinge axis
        axis_dot = abs(rotation.x * self.axis.x +
                       rotation.y * self.axis.y +
                       rotation.z * self.axis.z)

        if axis_dot < 1e-6:
            return rotation

        # Extract angle around axis
        angle = 2 * math.acos(max(-1, min(1, rotation.w)))
        angle = max(self.min_value, min(self.max_value, angle))

        return Quat.from_axis_angle(self.axis, angle)

    def _apply_cone(self, rotation: Quat, reference_axis: Vec3) -> Quat:
        """Apply cone constraint."""
        # Get rotated axis
        rotated = rotation.rotate_vector(reference_axis)

        # Check angle from constraint axis
        dot = rotated.dot(self.axis)
        angle = math.acos(max(-1, min(1, dot)))

        if angle <= self.max_value:
            return rotation

        # Clamp to cone surface
        cross = self.axis.cross(rotated)
        if cross.length_squared() < 1e-6:
            return rotation

        cross = cross.normalized()
        return Quat.from_axis_angle(cross, self.max_value)

    def _apply_angle_limit(self, rotation: Quat) -> Quat:
        """Apply general angle limit."""
        angle = 2 * math.acos(max(-1, min(1, rotation.w)))

        if angle <= self.max_value:
            return rotation

        # Scale rotation to max angle
        scale = self.max_value / angle if angle > 0 else 0
        axis = Vec3(rotation.x, rotation.y, rotation.z)
        if axis.length_squared() > 0:
            axis = axis.normalized()
            return Quat.from_axis_angle(axis, self.max_value)

        return Quat.identity()


# =============================================================================
# SOLVER CONFIGURATIONS
# =============================================================================


@dataclass
class IKSolverConfig:
    """Base configuration for IK solvers.

    Attributes:
        solver_type: Type of solver
        iterations: Maximum iterations
        tolerance: Convergence tolerance
        damping: Damping factor for stability
    """

    solver_type: IKSolverType
    iterations: int = 10
    tolerance: float = 0.001
    damping: float = 0.1


@dataclass
class TwoBoneSolverConfig(IKSolverConfig):
    """Configuration for two-bone IK solver.

    Attributes:
        allow_twist: Whether to allow twist around bone axis
        use_pole_vector: Whether to use pole vector
        maintain_bone_lengths: Whether to preserve bone lengths
    """

    allow_twist: bool = True
    use_pole_vector: bool = True
    maintain_bone_lengths: bool = True

    def __post_init__(self) -> None:
        self.solver_type = IKSolverType.TWO_BONE
        self.iterations = 1  # Analytical solver


@dataclass
class FABRIKSolverConfig(IKSolverConfig):
    """Configuration for FABRIK solver.

    Attributes:
        use_constraints: Whether to apply joint constraints
        blend_to_source: Blend factor with source pose
    """

    use_constraints: bool = True
    blend_to_source: float = 0.0

    def __post_init__(self) -> None:
        self.solver_type = IKSolverType.FABRIK


@dataclass
class CCDSolverConfig(IKSolverConfig):
    """Configuration for CCD solver.

    Attributes:
        limit_rotation: Maximum rotation per iteration
        joint_weights: Per-joint influence weights
    """

    limit_rotation: float = math.pi / 4  # 45 degrees
    joint_weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.solver_type = IKSolverType.CCD


@dataclass
class FullBodySolverConfig(IKSolverConfig):
    """Configuration for full-body IK solver.

    Attributes:
        root_motion_weight: Weight for root motion
        maintain_center_of_mass: Whether to maintain CoM
        chain_configs: Per-chain configurations
    """

    root_motion_weight: float = 1.0
    maintain_center_of_mass: bool = True
    chain_configs: Dict[str, IKSolverConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.solver_type = IKSolverType.FULL_BODY


# =============================================================================
# IK CHAIN
# =============================================================================


class IKChain:
    """An inverse kinematics chain.

    An IK chain defines a sequence of bones from root to end effector
    that can be solved using various IK algorithms.

    Attributes:
        name: Chain name
        chain_type: Type of chain
        bones: List of bones in chain
        effector: End effector
    """

    def __init__(
        self,
        name: str,
        chain_type: IKChainType = IKChainType.CHAIN,
    ) -> None:
        if not name:
            raise ValueError("Chain name cannot be empty")

        self._name = name
        self._chain_type = chain_type
        self._bones: List[IKBone] = []
        self._effector: Optional[IKEffector] = None
        self._pole_vector: Optional[IKPoleVector] = None
        self._solver_config: IKSolverConfig = IKSolverConfig(IKSolverType.FABRIK)
        self._enabled = True

    @property
    def name(self) -> str:
        """Get chain name."""
        return self._name

    @property
    def chain_type(self) -> IKChainType:
        """Get chain type."""
        return self._chain_type

    @property
    def bones(self) -> List[IKBone]:
        """Get bones in chain."""
        return list(self._bones)

    @property
    def bone_count(self) -> int:
        """Get number of bones."""
        return len(self._bones)

    @property
    def effector(self) -> Optional[IKEffector]:
        """Get end effector."""
        return self._effector

    @property
    def pole_vector(self) -> Optional[IKPoleVector]:
        """Get pole vector."""
        return self._pole_vector

    @property
    def solver_config(self) -> IKSolverConfig:
        """Get solver configuration."""
        return self._solver_config

    @property
    def enabled(self) -> bool:
        """Check if chain is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._enabled = value

    @property
    def total_length(self) -> float:
        """Get total chain length."""
        return sum(bone.length for bone in self._bones)

    def add_bone(self, bone: IKBone) -> None:
        """Add a bone to the chain."""
        self._bones.append(bone)

    def remove_bone(self, bone_name: str) -> bool:
        """Remove a bone from the chain."""
        for i, bone in enumerate(self._bones):
            if bone.bone_name == bone_name:
                self._bones.pop(i)
                return True
        return False

    def get_bone(self, bone_name: str) -> Optional[IKBone]:
        """Get bone by name."""
        for bone in self._bones:
            if bone.bone_name == bone_name:
                return bone
        return None

    def set_effector(self, effector: IKEffector) -> None:
        """Set the end effector."""
        self._effector = effector

    def set_pole_vector(self, pole_vector: IKPoleVector) -> None:
        """Set the pole vector."""
        self._pole_vector = pole_vector

    def set_solver_config(self, config: IKSolverConfig) -> None:
        """Set solver configuration."""
        self._solver_config = config

    def set_bone_constraint(
        self,
        bone_name: str,
        constraint_type: IKConstraintType,
        axis: Optional[Vec3] = None,
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
    ) -> bool:
        """Set constraint for a bone."""
        bone = self.get_bone(bone_name)
        if bone is None:
            return False

        bone.constraint_type = constraint_type
        if axis is not None:
            bone.constraint_axis = axis
        bone.min_angle = min_angle
        bone.max_angle = max_angle
        return True

    def is_valid(self) -> bool:
        """Check if chain is valid."""
        if len(self._bones) < 2:
            return False
        if self._effector is None:
            return False
        return True

    def get_bone_indices(self) -> List[int]:
        """Get bone indices for solver."""
        return [bone.bone_index for bone in self._bones]


# =============================================================================
# IK PREVIEW
# =============================================================================


class IKPreview:
    """Preview settings for IK visualization."""

    def __init__(self) -> None:
        self.show_chains = True
        self.show_effectors = True
        self.show_pole_vectors = True
        self.show_constraints = True
        self.show_bone_lengths = False
        self.chain_color = (100, 200, 100)
        self.effector_color = (200, 100, 100)
        self.pole_color = (100, 100, 200)
        self.constraint_color = (200, 200, 100)
        self.effector_size = 0.1
        self.pole_size = 0.05


# =============================================================================
# IK SETUP EDITOR
# =============================================================================


class IKSetupEditor:
    """Editor for IK setup.

    Provides functionality for creating and configuring IK chains,
    effectors, and constraints.
    """

    def __init__(self) -> None:
        self._chains: Dict[str, IKChain] = {}
        self._preview = IKPreview()
        self._selected_chain: Optional[str] = None
        self._selected_bone: Optional[str] = None
        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def chains(self) -> List[IKChain]:
        """Get all chains."""
        return list(self._chains.values())

    @property
    def chain_count(self) -> int:
        """Get number of chains."""
        return len(self._chains)

    @property
    def preview(self) -> IKPreview:
        """Get preview settings."""
        return self._preview

    @property
    def selected_chain(self) -> Optional[str]:
        """Get selected chain name."""
        return self._selected_chain

    @property
    def selected_bone(self) -> Optional[str]:
        """Get selected bone name."""
        return self._selected_bone

    def create_chain(
        self,
        name: str,
        chain_type: IKChainType = IKChainType.CHAIN,
    ) -> IKChain:
        """Create a new IK chain."""
        if name in self._chains:
            raise ValueError(f"Chain '{name}' already exists")

        chain = IKChain(name, chain_type)
        self._chains[name] = chain
        self._notify_change()
        return chain

    def remove_chain(self, name: str) -> bool:
        """Remove an IK chain."""
        if name not in self._chains:
            return False

        del self._chains[name]
        if self._selected_chain == name:
            self._selected_chain = None
        self._notify_change()
        return True

    def get_chain(self, name: str) -> Optional[IKChain]:
        """Get chain by name."""
        return self._chains.get(name)

    def rename_chain(self, old_name: str, new_name: str) -> bool:
        """Rename a chain."""
        if old_name not in self._chains:
            return False
        if new_name in self._chains:
            return False

        chain = self._chains.pop(old_name)
        chain._name = new_name
        self._chains[new_name] = chain

        if self._selected_chain == old_name:
            self._selected_chain = new_name

        self._notify_change()
        return True

    def select_chain(self, name: Optional[str]) -> None:
        """Select a chain."""
        if name is None or name in self._chains:
            self._selected_chain = name
            self._selected_bone = None

    def select_bone(self, bone_name: Optional[str]) -> None:
        """Select a bone within the selected chain."""
        self._selected_bone = bone_name

    def add_bone_to_chain(
        self,
        chain_name: str,
        bone_name: str,
        bone_index: int,
        length: float = 0.0,
    ) -> bool:
        """Add a bone to a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        bone = IKBone(
            bone_name=bone_name,
            bone_index=bone_index,
            length=length,
        )
        chain.add_bone(bone)
        self._notify_change()
        return True

    def remove_bone_from_chain(self, chain_name: str, bone_name: str) -> bool:
        """Remove a bone from a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        if chain.remove_bone(bone_name):
            if self._selected_bone == bone_name:
                self._selected_bone = None
            self._notify_change()
            return True
        return False

    def set_effector(
        self,
        chain_name: str,
        bone_name: str,
        position_weight: float = 1.0,
        rotation_weight: float = 0.0,
    ) -> bool:
        """Set effector for a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        effector = IKEffector(
            name=f"{chain_name}_effector",
            bone_name=bone_name,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )
        chain.set_effector(effector)
        self._notify_change()
        return True

    def set_pole_vector(
        self,
        chain_name: str,
        position: Vec3,
        weight: float = 1.0,
    ) -> bool:
        """Set pole vector for a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        pole = IKPoleVector(
            name=f"{chain_name}_pole",
            position=position,
            weight=weight,
        )
        chain.set_pole_vector(pole)
        self._notify_change()
        return True

    def set_bone_constraint(
        self,
        chain_name: str,
        bone_name: str,
        constraint_type: IKConstraintType,
        axis: Optional[Vec3] = None,
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
    ) -> bool:
        """Set constraint for a bone in a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        if chain.set_bone_constraint(
            bone_name,
            constraint_type,
            axis,
            min_angle,
            max_angle,
        ):
            self._notify_change()
            return True
        return False

    def configure_solver(
        self,
        chain_name: str,
        solver_config: IKSolverConfig,
    ) -> bool:
        """Configure solver for a chain."""
        chain = self.get_chain(chain_name)
        if chain is None:
            return False

        chain.set_solver_config(solver_config)
        self._notify_change()
        return True

    def create_limb_ik(
        self,
        name: str,
        upper_bone: str,
        upper_index: int,
        upper_length: float,
        lower_bone: str,
        lower_index: int,
        lower_length: float,
        end_bone: str,
        end_index: int,
        pole_position: Optional[Vec3] = None,
    ) -> IKChain:
        """Create a limb IK setup (arm or leg)."""
        chain = self.create_chain(name, IKChainType.LIMB)

        # Add bones
        chain.add_bone(IKBone(
            bone_name=upper_bone,
            bone_index=upper_index,
            length=upper_length,
            constraint_type=IKConstraintType.BALL_SOCKET,
            max_angle=math.pi * 0.8,
        ))

        chain.add_bone(IKBone(
            bone_name=lower_bone,
            bone_index=lower_index,
            length=lower_length,
            constraint_type=IKConstraintType.HINGE,
            min_angle=0,
            max_angle=math.pi * 0.9,
        ))

        chain.add_bone(IKBone(
            bone_name=end_bone,
            bone_index=end_index,
            length=0,
        ))

        # Set effector
        chain.set_effector(IKEffector(
            name=f"{name}_effector",
            bone_name=end_bone,
        ))

        # Set pole vector
        if pole_position:
            chain.set_pole_vector(IKPoleVector(
                name=f"{name}_pole",
                position=pole_position,
            ))

        # Configure two-bone solver
        chain.set_solver_config(TwoBoneSolverConfig())

        self._notify_change()
        return chain

    def create_spine_ik(
        self,
        name: str,
        bones: List[Tuple[str, int, float]],  # (name, index, length)
    ) -> IKChain:
        """Create a spine IK setup."""
        if len(bones) < 3:
            raise ValueError("Spine chain requires at least 3 bones")

        chain = self.create_chain(name, IKChainType.SPINE)

        # Add bones
        for bone_name, bone_index, length in bones:
            chain.add_bone(IKBone(
                bone_name=bone_name,
                bone_index=bone_index,
                length=length,
                constraint_type=IKConstraintType.BALL_SOCKET,
                max_angle=math.pi / 6,  # 30 degrees
            ))

        # Set effector on last bone
        last_bone = bones[-1][0]
        chain.set_effector(IKEffector(
            name=f"{name}_effector",
            bone_name=last_bone,
        ))

        # Configure FABRIK solver
        chain.set_solver_config(FABRIKSolverConfig(
            iterations=15,
            tolerance=0.001,
        ))

        self._notify_change()
        return chain

    def validate(self) -> List[str]:
        """Validate all IK setups."""
        errors = []

        for name, chain in self._chains.items():
            if not chain.is_valid():
                if chain.bone_count < 2:
                    errors.append(f"Chain '{name}' has fewer than 2 bones")
                if chain.effector is None:
                    errors.append(f"Chain '{name}' has no effector")

            # Check for duplicate bone indices
            indices = chain.get_bone_indices()
            if len(indices) != len(set(indices)):
                errors.append(f"Chain '{name}' has duplicate bone indices")

        return errors

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "IKChainType",
    "IKSolverType",
    "IKConstraintType",
    "IKBone",
    "IKEffector",
    "IKPoleVector",
    "IKConstraint",
    "IKSolverConfig",
    "TwoBoneSolverConfig",
    "FABRIKSolverConfig",
    "CCDSolverConfig",
    "FullBodySolverConfig",
    "IKChain",
    "IKPreview",
    "IKSetupEditor",
]
