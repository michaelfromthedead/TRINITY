"""
Body Flags Module

Defines configuration flags for rigid body behavior and constraints.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import IntFlag, auto


class BodyFlagBits(IntFlag):
    """
    Bit flags for efficient body state storage.
    Can be combined with bitwise OR.
    """
    NONE = 0
    USE_GRAVITY = auto()
    ENABLE_CCD = auto()
    LOCK_POSITION_X = auto()
    LOCK_POSITION_Y = auto()
    LOCK_POSITION_Z = auto()
    LOCK_ROTATION_X = auto()
    LOCK_ROTATION_Y = auto()
    LOCK_ROTATION_Z = auto()
    IS_TRIGGER = auto()
    ENABLE_GYROSCOPIC = auto()
    IS_SLEEPING = auto()
    DISABLE_DEACTIVATION = auto()
    KINEMATIC_OBJECT = auto()
    STATIC_OBJECT = auto()
    CHARACTER_OBJECT = auto()
    DISABLE_WORLD_GRAVITY = auto()
    ENABLE_CONTACT_CALLBACK = auto()
    ENABLE_COLLISION_CALLBACK = auto()
    CUSTOM_MATERIAL_CALLBACK = auto()

    # Composite flags
    LOCK_POSITION_ALL = LOCK_POSITION_X | LOCK_POSITION_Y | LOCK_POSITION_Z
    LOCK_ROTATION_ALL = LOCK_ROTATION_X | LOCK_ROTATION_Y | LOCK_ROTATION_Z
    LOCK_ALL = LOCK_POSITION_ALL | LOCK_ROTATION_ALL


@dataclass
class BodyFlags:
    """
    Configuration flags for rigid body behavior.

    This class provides a convenient interface for managing body flags
    while maintaining efficient bit-level storage internally.

    Attributes:
        use_gravity: Whether gravity affects this body
        enable_ccd: Enable continuous collision detection
        lock_position_x: Lock position on X axis
        lock_position_y: Lock position on Y axis
        lock_position_z: Lock position on Z axis
        lock_rotation_x: Lock rotation around X axis
        lock_rotation_y: Lock rotation around Y axis
        lock_rotation_z: Lock rotation around Z axis
        is_trigger: Body acts as trigger (no physics response)
        enable_gyroscopic: Enable gyroscopic forces for rotation
        is_sleeping: Body is currently sleeping
        disable_deactivation: Prevent body from sleeping
        enable_contact_callback: Fire callbacks on contact
        enable_collision_callback: Fire callbacks on collision
    """

    use_gravity: bool = True
    enable_ccd: bool = False
    lock_position_x: bool = False
    lock_position_y: bool = False
    lock_position_z: bool = False
    lock_rotation_x: bool = False
    lock_rotation_y: bool = False
    lock_rotation_z: bool = False
    is_trigger: bool = False
    enable_gyroscopic: bool = True
    is_sleeping: bool = False
    disable_deactivation: bool = False
    enable_contact_callback: bool = False
    enable_collision_callback: bool = False
    custom_material_callback: bool = False

    def __post_init__(self):
        """Initialize internal bit flags."""
        self._update_bits()

    def _update_bits(self) -> None:
        """Update internal bit representation from boolean flags."""
        self._bits = BodyFlagBits.NONE

        if self.use_gravity:
            self._bits |= BodyFlagBits.USE_GRAVITY
        if self.enable_ccd:
            self._bits |= BodyFlagBits.ENABLE_CCD
        if self.lock_position_x:
            self._bits |= BodyFlagBits.LOCK_POSITION_X
        if self.lock_position_y:
            self._bits |= BodyFlagBits.LOCK_POSITION_Y
        if self.lock_position_z:
            self._bits |= BodyFlagBits.LOCK_POSITION_Z
        if self.lock_rotation_x:
            self._bits |= BodyFlagBits.LOCK_ROTATION_X
        if self.lock_rotation_y:
            self._bits |= BodyFlagBits.LOCK_ROTATION_Y
        if self.lock_rotation_z:
            self._bits |= BodyFlagBits.LOCK_ROTATION_Z
        if self.is_trigger:
            self._bits |= BodyFlagBits.IS_TRIGGER
        if self.enable_gyroscopic:
            self._bits |= BodyFlagBits.ENABLE_GYROSCOPIC
        if self.is_sleeping:
            self._bits |= BodyFlagBits.IS_SLEEPING
        if self.disable_deactivation:
            self._bits |= BodyFlagBits.DISABLE_DEACTIVATION
        if self.enable_contact_callback:
            self._bits |= BodyFlagBits.ENABLE_CONTACT_CALLBACK
        if self.enable_collision_callback:
            self._bits |= BodyFlagBits.ENABLE_COLLISION_CALLBACK
        if self.custom_material_callback:
            self._bits |= BodyFlagBits.CUSTOM_MATERIAL_CALLBACK

    @property
    def bits(self) -> BodyFlagBits:
        """Get the internal bit representation."""
        return self._bits

    @bits.setter
    def bits(self, value: BodyFlagBits) -> None:
        """Set flags from bit representation."""
        self._bits = value
        self._sync_from_bits()

    def _sync_from_bits(self) -> None:
        """Synchronize boolean flags from internal bits."""
        self.use_gravity = bool(self._bits & BodyFlagBits.USE_GRAVITY)
        self.enable_ccd = bool(self._bits & BodyFlagBits.ENABLE_CCD)
        self.lock_position_x = bool(self._bits & BodyFlagBits.LOCK_POSITION_X)
        self.lock_position_y = bool(self._bits & BodyFlagBits.LOCK_POSITION_Y)
        self.lock_position_z = bool(self._bits & BodyFlagBits.LOCK_POSITION_Z)
        self.lock_rotation_x = bool(self._bits & BodyFlagBits.LOCK_ROTATION_X)
        self.lock_rotation_y = bool(self._bits & BodyFlagBits.LOCK_ROTATION_Y)
        self.lock_rotation_z = bool(self._bits & BodyFlagBits.LOCK_ROTATION_Z)
        self.is_trigger = bool(self._bits & BodyFlagBits.IS_TRIGGER)
        self.enable_gyroscopic = bool(self._bits & BodyFlagBits.ENABLE_GYROSCOPIC)
        self.is_sleeping = bool(self._bits & BodyFlagBits.IS_SLEEPING)
        self.disable_deactivation = bool(self._bits & BodyFlagBits.DISABLE_DEACTIVATION)
        self.enable_contact_callback = bool(self._bits & BodyFlagBits.ENABLE_CONTACT_CALLBACK)
        self.enable_collision_callback = bool(self._bits & BodyFlagBits.ENABLE_COLLISION_CALLBACK)
        self.custom_material_callback = bool(self._bits & BodyFlagBits.CUSTOM_MATERIAL_CALLBACK)

    def set_flag(self, flag: BodyFlagBits, value: bool = True) -> None:
        """Set a specific flag bit."""
        if value:
            self._bits |= flag
        else:
            self._bits &= ~flag
        self._sync_from_bits()

    def get_flag(self, flag: BodyFlagBits) -> bool:
        """Get a specific flag bit."""
        return bool(self._bits & flag)

    def toggle_flag(self, flag: BodyFlagBits) -> None:
        """Toggle a specific flag bit."""
        self._bits ^= flag
        self._sync_from_bits()

    def clear_all(self) -> None:
        """Clear all flags."""
        self._bits = BodyFlagBits.NONE
        self._sync_from_bits()

    @property
    def lock_position_all(self) -> bool:
        """Check if all position axes are locked."""
        return self.lock_position_x and self.lock_position_y and self.lock_position_z

    @lock_position_all.setter
    def lock_position_all(self, value: bool) -> None:
        """Lock or unlock all position axes."""
        self.lock_position_x = value
        self.lock_position_y = value
        self.lock_position_z = value
        self._update_bits()

    @property
    def lock_rotation_all(self) -> bool:
        """Check if all rotation axes are locked."""
        return self.lock_rotation_x and self.lock_rotation_y and self.lock_rotation_z

    @lock_rotation_all.setter
    def lock_rotation_all(self, value: bool) -> None:
        """Lock or unlock all rotation axes."""
        self.lock_rotation_x = value
        self.lock_rotation_y = value
        self.lock_rotation_z = value
        self._update_bits()

    @property
    def has_position_lock(self) -> bool:
        """Check if any position axis is locked."""
        return self.lock_position_x or self.lock_position_y or self.lock_position_z

    @property
    def has_rotation_lock(self) -> bool:
        """Check if any rotation axis is locked."""
        return self.lock_rotation_x or self.lock_rotation_y or self.lock_rotation_z

    @property
    def is_fully_locked(self) -> bool:
        """Check if body is fully constrained."""
        return self.lock_position_all and self.lock_rotation_all

    @property
    def can_sleep(self) -> bool:
        """Check if body is allowed to sleep."""
        return not self.disable_deactivation and not self.is_trigger

    def get_position_lock_mask(self) -> tuple[float, float, float]:
        """
        Get position lock as a multiplier mask.

        Returns:
            Tuple of (x, y, z) where 0.0 = locked, 1.0 = unlocked
        """
        return (
            0.0 if self.lock_position_x else 1.0,
            0.0 if self.lock_position_y else 1.0,
            0.0 if self.lock_position_z else 1.0,
        )

    def get_rotation_lock_mask(self) -> tuple[float, float, float]:
        """
        Get rotation lock as a multiplier mask.

        Returns:
            Tuple of (x, y, z) where 0.0 = locked, 1.0 = unlocked
        """
        return (
            0.0 if self.lock_rotation_x else 1.0,
            0.0 if self.lock_rotation_y else 1.0,
            0.0 if self.lock_rotation_z else 1.0,
        )

    def copy(self) -> 'BodyFlags':
        """Create a copy of these flags."""
        flags = BodyFlags(
            use_gravity=self.use_gravity,
            enable_ccd=self.enable_ccd,
            lock_position_x=self.lock_position_x,
            lock_position_y=self.lock_position_y,
            lock_position_z=self.lock_position_z,
            lock_rotation_x=self.lock_rotation_x,
            lock_rotation_y=self.lock_rotation_y,
            lock_rotation_z=self.lock_rotation_z,
            is_trigger=self.is_trigger,
            enable_gyroscopic=self.enable_gyroscopic,
            is_sleeping=self.is_sleeping,
            disable_deactivation=self.disable_deactivation,
            enable_contact_callback=self.enable_contact_callback,
            enable_collision_callback=self.enable_collision_callback,
            custom_material_callback=self.custom_material_callback,
        )
        return flags

    @classmethod
    def from_bits(cls, bits: BodyFlagBits) -> 'BodyFlags':
        """Create BodyFlags from bit representation."""
        flags = cls()
        flags.bits = bits
        return flags

    @classmethod
    def static_body(cls) -> 'BodyFlags':
        """Create flags for a static body."""
        return cls(
            use_gravity=False,
            lock_position_x=True,
            lock_position_y=True,
            lock_position_z=True,
            lock_rotation_x=True,
            lock_rotation_y=True,
            lock_rotation_z=True,
            disable_deactivation=True,
        )

    @classmethod
    def kinematic_body(cls) -> 'BodyFlags':
        """Create flags for a kinematic body."""
        return cls(
            use_gravity=False,
            disable_deactivation=True,
        )

    @classmethod
    def dynamic_body(cls) -> 'BodyFlags':
        """Create flags for a dynamic body."""
        return cls(
            use_gravity=True,
            enable_gyroscopic=True,
        )

    @classmethod
    def trigger_volume(cls) -> 'BodyFlags':
        """Create flags for a trigger volume."""
        return cls(
            use_gravity=False,
            is_trigger=True,
            lock_position_x=True,
            lock_position_y=True,
            lock_position_z=True,
            lock_rotation_x=True,
            lock_rotation_y=True,
            lock_rotation_z=True,
            disable_deactivation=True,
            enable_collision_callback=True,
        )

    @classmethod
    def character_body(cls) -> 'BodyFlags':
        """Create flags for a character controller body."""
        return cls(
            use_gravity=True,
            enable_ccd=True,
            lock_rotation_x=True,
            lock_rotation_y=False,  # Allow Y rotation for turning
            lock_rotation_z=True,
            enable_gyroscopic=False,
            disable_deactivation=True,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BodyFlags):
            return NotImplemented
        return self._bits == other._bits

    def __hash__(self) -> int:
        return hash(self._bits)

    def __repr__(self) -> str:
        flags = []
        if self.use_gravity:
            flags.append("GRAVITY")
        if self.enable_ccd:
            flags.append("CCD")
        if self.lock_position_x:
            flags.append("LOCK_PX")
        if self.lock_position_y:
            flags.append("LOCK_PY")
        if self.lock_position_z:
            flags.append("LOCK_PZ")
        if self.lock_rotation_x:
            flags.append("LOCK_RX")
        if self.lock_rotation_y:
            flags.append("LOCK_RY")
        if self.lock_rotation_z:
            flags.append("LOCK_RZ")
        if self.is_trigger:
            flags.append("TRIGGER")
        if self.enable_gyroscopic:
            flags.append("GYRO")
        if self.is_sleeping:
            flags.append("SLEEPING")
        if self.disable_deactivation:
            flags.append("NO_SLEEP")
        return f"BodyFlags({' | '.join(flags) if flags else 'NONE'})"
