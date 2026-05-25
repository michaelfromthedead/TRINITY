"""
Drivetrain simulation components.

This module provides engine, transmission, differential, and clutch
simulation for vehicle drivetrain physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from .config import (
    ENGINE_IDLE_RPM,
    ENGINE_MAX_RPM,
    ENGINE_REDLINE_RPM,
    ENGINE_INERTIA,
    ENGINE_FRICTION,
    DEFAULT_MAX_TORQUE,
    DEFAULT_GEAR_RATIOS,
    DEFAULT_FINAL_DRIVE,
    SHIFT_TIME,
    CLUTCH_ENGAGEMENT_RATE,
    LSD_PRELOAD,
    LSD_POWER_RATIO,
    LSD_COAST_RATIO,
)


class DiffType(Enum):
    """Types of differentials."""

    OPEN = auto()           # Standard open differential
    LIMITED_SLIP = auto()   # Limited slip (clutch-type)
    LOCKED = auto()         # Fully locked (spool)
    TORSEN = auto()         # Torsen (torque-sensing)
    ELECTRONIC = auto()     # Electronically controlled


class DrivetrainLayout(Enum):
    """Drivetrain layouts."""

    FWD = auto()    # Front-wheel drive
    RWD = auto()    # Rear-wheel drive
    AWD = auto()    # All-wheel drive
    FOURWD = auto() # Four-wheel drive (part-time)


@dataclass
class EngineState:
    """Current engine state."""

    rpm: float = ENGINE_IDLE_RPM
    throttle: float = 0.0        # 0.0 to 1.0
    torque_output: float = 0.0
    power_output: float = 0.0    # Watts
    fuel_consumption: float = 0.0
    is_running: bool = True
    is_revlimited: bool = False


class Engine:
    """
    Internal combustion engine simulation.

    Provides torque curve, inertia, and friction modeling.
    """

    def __init__(
        self,
        idle_rpm: int = ENGINE_IDLE_RPM,
        max_rpm: int = ENGINE_MAX_RPM,
        redline_rpm: int = ENGINE_REDLINE_RPM,
        max_torque: float = DEFAULT_MAX_TORQUE,
        torque_curve: Optional[Dict[int, float]] = None,
        inertia: float = ENGINE_INERTIA,
        friction: float = ENGINE_FRICTION,
    ):
        """
        Initialize engine.

        Args:
            idle_rpm: Idle RPM.
            max_rpm: Maximum RPM (hard limit).
            redline_rpm: Redline RPM (soft limit).
            max_torque: Maximum torque output.
            torque_curve: RPM -> torque multiplier dict. Uses default if None.
            inertia: Rotational inertia (kg*m^2).
            friction: Friction coefficient (N*m / RPM).
        """
        self._idle_rpm = idle_rpm
        self._max_rpm = max_rpm
        self._redline_rpm = redline_rpm
        self._max_torque = max_torque
        self._inertia = inertia
        self._friction = friction

        # Build torque curve
        if torque_curve is not None:
            self._torque_curve = torque_curve
        else:
            self._torque_curve = self._default_torque_curve()

        # State
        self._state = EngineState()
        self._angular_velocity = self._rpm_to_rad_s(idle_rpm)

    def _default_torque_curve(self) -> Dict[int, float]:
        """Generate default torque curve."""
        # Typical naturally-aspirated engine curve
        return {
            1000: 0.60,
            2000: 0.75,
            3000: 0.88,
            4000: 0.95,
            4500: 1.00,  # Peak torque
            5000: 0.98,
            5500: 0.94,
            6000: 0.88,
            6500: 0.78,
            7000: 0.65,
        }

    def _rpm_to_rad_s(self, rpm: float) -> float:
        """Convert RPM to radians per second."""
        return rpm * math.pi / 30.0

    def _rad_s_to_rpm(self, rad_s: float) -> float:
        """Convert radians per second to RPM."""
        return rad_s * 30.0 / math.pi

    @property
    def rpm(self) -> float:
        """Current engine RPM."""
        return self._state.rpm

    @property
    def idle_rpm(self) -> int:
        """Idle RPM."""
        return self._idle_rpm

    @property
    def max_rpm(self) -> int:
        """Maximum RPM."""
        return self._max_rpm

    @property
    def redline_rpm(self) -> int:
        """Redline RPM."""
        return self._redline_rpm

    @property
    def max_torque(self) -> float:
        """Maximum torque."""
        return self._max_torque

    @property
    def inertia(self) -> float:
        """Rotational inertia."""
        return self._inertia

    @property
    def state(self) -> EngineState:
        """Current state."""
        return self._state

    @property
    def angular_velocity(self) -> float:
        """Angular velocity (rad/s)."""
        return self._angular_velocity

    def get_torque_multiplier(self, rpm: float) -> float:
        """
        Get torque multiplier at given RPM from curve.

        Interpolates between defined points using linear interpolation.

        Args:
            rpm: Engine RPM.

        Returns:
            Torque multiplier (0.0 to 1.0).
        """
        if not self._torque_curve:
            return 1.0

        rpms = sorted(self._torque_curve.keys())

        # Handle single-point curve
        if len(rpms) == 1:
            return self._torque_curve[rpms[0]]

        # Clamp to curve bounds
        if rpm <= rpms[0]:
            return self._torque_curve[rpms[0]]
        if rpm >= rpms[-1]:
            return self._torque_curve[rpms[-1]]

        # Find surrounding points and interpolate
        for i in range(len(rpms) - 1):
            if rpms[i] <= rpm <= rpms[i + 1]:
                # Guard against division by zero (duplicate RPM points)
                rpm_range = rpms[i + 1] - rpms[i]
                if rpm_range <= 0:
                    return self._torque_curve[rpms[i]]

                t = (rpm - rpms[i]) / rpm_range
                v0 = self._torque_curve[rpms[i]]
                v1 = self._torque_curve[rpms[i + 1]]
                return v0 + t * (v1 - v0)

        return 1.0

    def compute_torque(self, throttle: float, rpm: float) -> float:
        """
        Compute engine torque at given throttle and RPM.

        Args:
            throttle: Throttle position (0.0 to 1.0).
            rpm: Engine RPM.

        Returns:
            Engine torque (N*m).
        """
        # Clamp throttle
        throttle = max(0.0, min(1.0, throttle))

        # Get torque curve multiplier
        multiplier = self.get_torque_multiplier(rpm)

        # Base torque
        torque = self._max_torque * multiplier * throttle

        # Engine braking at closed throttle
        if throttle < 0.1:
            braking_factor = (0.1 - throttle) / 0.1
            torque -= self._max_torque * 0.1 * braking_factor

        # Friction loss
        friction_torque = self._friction * rpm

        return torque - friction_torque

    def update(self, throttle: float, load_torque: float, dt: float) -> float:
        """
        Update engine state.

        Args:
            throttle: Throttle input (0.0 to 1.0).
            load_torque: Torque load from drivetrain.
            dt: Delta time.

        Returns:
            Torque output.
        """
        if not self._state.is_running:
            self._state.rpm = 0
            self._state.torque_output = 0
            return 0.0

        # Get engine torque
        engine_torque = self.compute_torque(throttle, self._state.rpm)

        # Net torque
        net_torque = engine_torque - load_torque

        # Angular acceleration
        angular_accel = net_torque / self._inertia

        # Update angular velocity
        self._angular_velocity += angular_accel * dt

        # Convert to RPM
        new_rpm = self._rad_s_to_rpm(self._angular_velocity)

        # Rev limiter
        self._state.is_revlimited = new_rpm >= self._redline_rpm
        if new_rpm > self._max_rpm:
            new_rpm = self._max_rpm
            self._angular_velocity = self._rpm_to_rad_s(new_rpm)

        # Idle governor
        if new_rpm < self._idle_rpm and throttle < 0.1:
            new_rpm = self._idle_rpm
            self._angular_velocity = self._rpm_to_rad_s(new_rpm)

        # Update state
        self._state.rpm = new_rpm
        self._state.throttle = throttle
        self._state.torque_output = engine_torque
        self._state.power_output = engine_torque * self._angular_velocity

        return engine_torque

    def start(self) -> None:
        """Start the engine."""
        self._state.is_running = True
        self._state.rpm = self._idle_rpm
        self._angular_velocity = self._rpm_to_rad_s(self._idle_rpm)

    def stop(self) -> None:
        """Stop the engine."""
        self._state.is_running = False
        self._state.rpm = 0
        self._angular_velocity = 0


@dataclass
class TransmissionState:
    """Current transmission state."""

    current_gear: int = 1  # 0 = neutral, -1 = reverse, 1+ = forward gears
    gear_ratio: float = 0.0
    output_torque: float = 0.0
    is_shifting: bool = False
    shift_progress: float = 0.0  # 0 to 1 during shift


class Transmission:
    """
    Manual or automatic transmission simulation.

    Handles gear ratios, shifting, and power interruption.
    """

    def __init__(
        self,
        gear_ratios: tuple = DEFAULT_GEAR_RATIOS,
        final_drive: float = DEFAULT_FINAL_DRIVE,
        shift_time: float = SHIFT_TIME,
        auto_mode: bool = False,
        upshift_rpm: Optional[float] = None,
        downshift_rpm: Optional[float] = None,
    ):
        """
        Initialize transmission.

        Args:
            gear_ratios: Tuple of gear ratios (R, N, 1, 2, 3, ...).
            final_drive: Final drive ratio.
            shift_time: Time for gear change.
            auto_mode: Enable automatic shifting.
            upshift_rpm: RPM to upshift (auto mode).
            downshift_rpm: RPM to downshift (auto mode).
        """
        self._gear_ratios = gear_ratios
        self._final_drive = final_drive
        self._shift_time = shift_time
        self._auto_mode = auto_mode
        self._upshift_rpm = upshift_rpm or 6000.0
        self._downshift_rpm = downshift_rpm or 2500.0

        # State
        self._state = TransmissionState()
        self._state.current_gear = 1  # Start in 1st
        self._state.gear_ratio = self._get_total_ratio(1)

        # Shift timer
        self._shift_timer = 0.0
        self._target_gear = 1

    @property
    def current_gear(self) -> int:
        """Current gear (-1 = R, 0 = N, 1+ = forward)."""
        return self._state.current_gear

    @property
    def gear_ratio(self) -> float:
        """Current total gear ratio."""
        return self._state.gear_ratio

    @property
    def final_drive(self) -> float:
        """Final drive ratio."""
        return self._final_drive

    @property
    def gear_count(self) -> int:
        """Number of forward gears."""
        return len(self._gear_ratios) - 2  # Exclude R and N

    @property
    def state(self) -> TransmissionState:
        """Current state."""
        return self._state

    @property
    def is_shifting(self) -> bool:
        """Whether currently shifting."""
        return self._state.is_shifting

    def _gear_to_index(self, gear: int) -> int:
        """Convert gear number to ratios index."""
        if gear == -1:  # Reverse
            return 0
        elif gear == 0:  # Neutral
            return 1
        else:  # Forward gears
            return gear + 1

    def _get_gear_ratio(self, gear: int) -> float:
        """Get gear ratio for specified gear."""
        index = self._gear_to_index(gear)
        if 0 <= index < len(self._gear_ratios):
            return self._gear_ratios[index]
        return 0.0

    def _get_total_ratio(self, gear: int) -> float:
        """Get total ratio (gear * final drive)."""
        return self._get_gear_ratio(gear) * self._final_drive

    def shift(self, gear: int) -> bool:
        """
        Request gear change.

        Args:
            gear: Target gear.

        Returns:
            True if shift initiated.
        """
        # Validate gear
        max_gear = self.gear_count
        if gear < -1 or gear > max_gear:
            return False

        # Already in gear or shifting
        if gear == self._state.current_gear or self._state.is_shifting:
            return False

        # Initiate shift
        self._state.is_shifting = True
        self._shift_timer = 0.0
        self._target_gear = gear

        return True

    def shift_up(self) -> bool:
        """Shift up one gear."""
        if self._state.current_gear >= self.gear_count:
            return False
        return self.shift(self._state.current_gear + 1)

    def shift_down(self) -> bool:
        """Shift down one gear."""
        if self._state.current_gear <= -1:
            return False
        return self.shift(self._state.current_gear - 1)

    def update(
        self,
        input_torque: float,
        input_rpm: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Update transmission state.

        Args:
            input_torque: Torque from engine/clutch.
            input_rpm: Input shaft RPM.
            dt: Delta time.

        Returns:
            Tuple of (output_torque, output_rpm).
        """
        # Handle shifting
        if self._state.is_shifting:
            self._shift_timer += dt
            self._state.shift_progress = self._shift_timer / self._shift_time

            if self._shift_timer >= self._shift_time:
                # Shift complete
                self._state.current_gear = self._target_gear
                self._state.gear_ratio = self._get_total_ratio(self._target_gear)
                self._state.is_shifting = False
                self._state.shift_progress = 0.0

            # No torque during shift
            self._state.output_torque = 0.0
            return (0.0, input_rpm)

        # Auto shifting
        if self._auto_mode and not self._state.is_shifting:
            if input_rpm >= self._upshift_rpm and self._state.current_gear > 0:
                self.shift_up()
            elif input_rpm <= self._downshift_rpm and self._state.current_gear > 1:
                self.shift_down()

        # Calculate output
        ratio = self._state.gear_ratio
        if abs(ratio) < 0.001:  # Neutral
            self._state.output_torque = 0.0
            return (0.0, 0.0)

        output_torque = input_torque * ratio
        output_rpm = input_rpm / abs(ratio)

        self._state.output_torque = output_torque
        return (output_torque, output_rpm)


@dataclass
class ClutchState:
    """Current clutch state."""

    engagement: float = 1.0      # 0 = disengaged, 1 = engaged
    slip: float = 0.0            # Slip velocity
    torque_transfer: float = 0.0
    is_slipping: bool = False


class Clutch:
    """
    Clutch simulation.

    Models engagement, slip, and torque transfer.
    """

    def __init__(
        self,
        max_torque: float = 500.0,
        engagement_rate: float = CLUTCH_ENGAGEMENT_RATE,
        inertia: float = 0.02,
    ):
        """
        Initialize clutch.

        Args:
            max_torque: Maximum torque capacity.
            engagement_rate: Rate of engagement (1/s).
            inertia: Clutch disc inertia.
        """
        self._max_torque = max_torque
        self._engagement_rate = engagement_rate
        self._inertia = inertia

        self._state = ClutchState()
        self._target_engagement = 1.0

    @property
    def engagement(self) -> float:
        """Current engagement (0-1)."""
        return self._state.engagement

    @property
    def is_engaged(self) -> bool:
        """Whether clutch is fully engaged."""
        return self._state.engagement >= 0.99

    @property
    def is_slipping(self) -> bool:
        """Whether clutch is slipping."""
        return self._state.is_slipping

    @property
    def state(self) -> ClutchState:
        """Current state."""
        return self._state

    def set_engagement(self, engagement: float) -> None:
        """
        Set target engagement.

        Args:
            engagement: Target engagement (0-1).
        """
        self._target_engagement = max(0.0, min(1.0, engagement))

    def disengage(self) -> None:
        """Fully disengage clutch."""
        self._target_engagement = 0.0

    def engage(self) -> None:
        """Fully engage clutch."""
        self._target_engagement = 1.0

    def update(
        self,
        engine_torque: float,
        engine_rpm: float,
        transmission_rpm: float,
        dt: float,
    ) -> float:
        """
        Update clutch state and calculate torque transfer.

        Args:
            engine_torque: Engine output torque.
            engine_rpm: Engine RPM.
            transmission_rpm: Transmission input RPM.
            dt: Delta time.

        Returns:
            Torque transferred to transmission.
        """
        # Smoothly move engagement
        engagement_diff = self._target_engagement - self._state.engagement
        max_change = self._engagement_rate * dt
        if abs(engagement_diff) <= max_change:
            self._state.engagement = self._target_engagement
        else:
            self._state.engagement += math.copysign(max_change, engagement_diff)

        # Calculate slip
        slip_velocity = (engine_rpm - transmission_rpm) * math.pi / 30.0
        self._state.slip = slip_velocity

        # Maximum transferable torque
        max_transfer = self._max_torque * self._state.engagement

        # Check for slip
        if abs(engine_torque) > max_transfer:
            # Clutch slipping
            self._state.is_slipping = True
            transfer = math.copysign(max_transfer, engine_torque)
        else:
            # Clutch locked (or nearly so)
            self._state.is_slipping = abs(slip_velocity) > 1.0
            transfer = engine_torque * self._state.engagement

        self._state.torque_transfer = transfer
        return transfer


class Differential:
    """
    Differential simulation.

    Models open, limited-slip, and locked differentials.
    """

    def __init__(
        self,
        diff_type: DiffType = DiffType.OPEN,
        preload: float = LSD_PRELOAD,
        power_ratio: float = LSD_POWER_RATIO,
        coast_ratio: float = LSD_COAST_RATIO,
        bias_ratio: float = 3.0,  # For Torsen
    ):
        """
        Initialize differential.

        Args:
            diff_type: Type of differential.
            preload: LSD preload torque.
            power_ratio: LSD power locking ratio.
            coast_ratio: LSD coast locking ratio.
            bias_ratio: Torsen bias ratio.
        """
        self._type = diff_type
        self._preload = preload
        self._power_ratio = power_ratio
        self._coast_ratio = coast_ratio
        self._bias_ratio = bias_ratio

    @property
    def diff_type(self) -> DiffType:
        """Differential type."""
        return self._type

    @diff_type.setter
    def diff_type(self, value: DiffType) -> None:
        """Set differential type."""
        self._type = value

    def torque_split(
        self,
        input_torque: float,
        left_speed: float,
        right_speed: float,
    ) -> Tuple[float, float]:
        """
        Calculate torque split to left and right outputs.

        Args:
            input_torque: Total input torque.
            left_speed: Left output angular velocity.
            right_speed: Right output angular velocity.

        Returns:
            Tuple of (left_torque, right_torque).
        """
        if self._type == DiffType.LOCKED:
            return self._locked_split(input_torque)
        elif self._type == DiffType.OPEN:
            return self._open_split(input_torque)
        elif self._type == DiffType.LIMITED_SLIP:
            return self._lsd_split(input_torque, left_speed, right_speed)
        elif self._type == DiffType.TORSEN:
            return self._torsen_split(input_torque, left_speed, right_speed)
        else:
            return self._open_split(input_torque)

    def _locked_split(self, input_torque: float) -> Tuple[float, float]:
        """Locked differential - equal torque split."""
        half_torque = input_torque / 2.0
        return (half_torque, half_torque)

    def _open_split(
        self,
        input_torque: float,
        left_speed: float = 0.0,
        right_speed: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Open differential - equal torque distribution.

        An open diff always splits torque 50/50, but the wheel with less
        grip will spin faster. The torque is limited by the wheel with
        least traction, so both wheels get equal torque but may have
        different speeds.
        """
        half_torque = input_torque / 2.0
        return (half_torque, half_torque)

    def _lsd_split(
        self,
        input_torque: float,
        left_speed: float,
        right_speed: float,
    ) -> Tuple[float, float]:
        """
        Limited slip differential with clutch packs.

        Transfers torque based on speed difference.
        """
        # Speed difference
        speed_diff = abs(left_speed - right_speed)

        # Determine if accelerating or coasting
        is_power = input_torque > 0
        lock_ratio = self._power_ratio if is_power else self._coast_ratio

        # Locking torque from clutches
        lock_torque = self._preload + abs(input_torque) * lock_ratio

        # Transfer torque from fast wheel to slow wheel
        if left_speed > right_speed:
            # Left is faster, transfer to right
            transfer = min(lock_torque, abs(input_torque) / 2)
            if speed_diff > 0.1:  # Only transfer if significant difference
                left_torque = input_torque / 2 - transfer * 0.5
                right_torque = input_torque / 2 + transfer * 0.5
            else:
                left_torque = right_torque = input_torque / 2
        else:
            # Right is faster, transfer to left
            transfer = min(lock_torque, abs(input_torque) / 2)
            if speed_diff > 0.1:
                left_torque = input_torque / 2 + transfer * 0.5
                right_torque = input_torque / 2 - transfer * 0.5
            else:
                left_torque = right_torque = input_torque / 2

        return (left_torque, right_torque)

    def _torsen_split(
        self,
        input_torque: float,
        left_speed: float,
        right_speed: float,
    ) -> Tuple[float, float]:
        """
        Torsen differential - torque-biasing based on gear friction.

        Can send up to bias_ratio times more torque to slower wheel.
        """
        speed_diff = left_speed - right_speed

        if abs(speed_diff) < 0.1:
            # No speed difference - equal split
            return (input_torque / 2, input_torque / 2)

        # Bias torque toward slower wheel
        max_bias = (self._bias_ratio - 1) / (self._bias_ratio + 1)

        if speed_diff > 0:
            # Left faster - bias to right
            bias = min(max_bias, speed_diff / 10.0)
            left_torque = input_torque * (0.5 - bias)
            right_torque = input_torque * (0.5 + bias)
        else:
            # Right faster - bias to left
            bias = min(max_bias, -speed_diff / 10.0)
            left_torque = input_torque * (0.5 + bias)
            right_torque = input_torque * (0.5 - bias)

        return (left_torque, right_torque)


class Drivetrain:
    """
    Complete drivetrain assembly.

    Combines engine, clutch, transmission, and differential(s).
    """

    def __init__(
        self,
        layout: DrivetrainLayout = DrivetrainLayout.RWD,
        engine: Optional[Engine] = None,
        transmission: Optional[Transmission] = None,
        clutch: Optional[Clutch] = None,
        front_diff: Optional[Differential] = None,
        rear_diff: Optional[Differential] = None,
        center_diff: Optional[Differential] = None,
        awd_torque_split: float = 0.5,  # Front bias for AWD
    ):
        """
        Initialize drivetrain.

        Args:
            layout: Drivetrain layout.
            engine: Engine instance (creates default if None).
            transmission: Transmission instance.
            clutch: Clutch instance.
            front_diff: Front differential.
            rear_diff: Rear differential.
            center_diff: Center differential (AWD).
            awd_torque_split: AWD front/rear split (0.5 = 50/50).
        """
        self._layout = layout

        # Create default components if not provided
        self._engine = engine or Engine()
        self._transmission = transmission or Transmission()
        self._clutch = clutch or Clutch()

        # Differentials based on layout
        if layout in (DrivetrainLayout.FWD, DrivetrainLayout.AWD, DrivetrainLayout.FOURWD):
            self._front_diff = front_diff or Differential(DiffType.OPEN)
        else:
            self._front_diff = None

        if layout in (DrivetrainLayout.RWD, DrivetrainLayout.AWD, DrivetrainLayout.FOURWD):
            self._rear_diff = rear_diff or Differential(DiffType.OPEN)
        else:
            self._rear_diff = None

        if layout == DrivetrainLayout.AWD:
            self._center_diff = center_diff or Differential(DiffType.LIMITED_SLIP)
        else:
            self._center_diff = None

        self._awd_torque_split = awd_torque_split

    @property
    def layout(self) -> DrivetrainLayout:
        """Drivetrain layout."""
        return self._layout

    @property
    def engine(self) -> Engine:
        """Engine."""
        return self._engine

    @property
    def transmission(self) -> Transmission:
        """Transmission."""
        return self._transmission

    @property
    def clutch(self) -> Clutch:
        """Clutch."""
        return self._clutch

    @property
    def current_gear(self) -> int:
        """Current gear."""
        return self._transmission.current_gear

    @property
    def engine_rpm(self) -> float:
        """Engine RPM."""
        return self._engine.rpm

    def update(
        self,
        throttle: float,
        wheel_speeds: Tuple[float, float, float, float],
        dt: float,
    ) -> Tuple[float, float, float, float]:
        """
        Update drivetrain and compute wheel torques.

        Args:
            throttle: Throttle input (0-1).
            wheel_speeds: (FL, FR, RL, RR) angular velocities.
            dt: Delta time.

        Returns:
            (FL, FR, RL, RR) torques.
        """
        fl_speed, fr_speed, rl_speed, rr_speed = wheel_speeds

        # Calculate transmission input speed from driven wheels
        if self._layout == DrivetrainLayout.FWD:
            avg_speed = (fl_speed + fr_speed) / 2
        elif self._layout == DrivetrainLayout.RWD:
            avg_speed = (rl_speed + rr_speed) / 2
        else:  # AWD/4WD
            avg_speed = (fl_speed + fr_speed + rl_speed + rr_speed) / 4

        # Convert wheel speed to transmission input RPM
        trans_ratio = self._transmission.gear_ratio
        if abs(trans_ratio) > 0.001:
            trans_rpm = abs(avg_speed) * 30.0 / math.pi * abs(trans_ratio)
        else:
            trans_rpm = self._engine.rpm  # In neutral

        # Update clutch
        clutch_torque = self._clutch.update(
            self._engine.state.torque_output,
            self._engine.rpm,
            trans_rpm,
            dt,
        )

        # Update engine with load
        engine_torque = self._engine.update(throttle, clutch_torque * 0.5, dt)

        # Update transmission
        trans_torque, trans_output_rpm = self._transmission.update(
            clutch_torque,
            self._engine.rpm,
            dt,
        )

        # Distribute torque through differentials
        fl_torque = fr_torque = rl_torque = rr_torque = 0.0

        if self._layout == DrivetrainLayout.FWD:
            fl_torque, fr_torque = self._front_diff.torque_split(
                trans_torque, fl_speed, fr_speed
            )
        elif self._layout == DrivetrainLayout.RWD:
            rl_torque, rr_torque = self._rear_diff.torque_split(
                trans_torque, rl_speed, rr_speed
            )
        elif self._layout == DrivetrainLayout.AWD:
            # Split through center diff
            front_torque, rear_torque = self._center_diff.torque_split(
                trans_torque,
                (fl_speed + fr_speed) / 2,
                (rl_speed + rr_speed) / 2,
            )
            # Apply fixed bias
            front_torque = trans_torque * self._awd_torque_split
            rear_torque = trans_torque * (1 - self._awd_torque_split)

            # Split to individual wheels
            fl_torque, fr_torque = self._front_diff.torque_split(
                front_torque, fl_speed, fr_speed
            )
            rl_torque, rr_torque = self._rear_diff.torque_split(
                rear_torque, rl_speed, rr_speed
            )
        elif self._layout == DrivetrainLayout.FOURWD:
            # Simple 50/50 split, no center diff slip
            fl_torque, fr_torque = self._front_diff.torque_split(
                trans_torque / 2, fl_speed, fr_speed
            )
            rl_torque, rr_torque = self._rear_diff.torque_split(
                trans_torque / 2, rl_speed, rr_speed
            )

        return (fl_torque, fr_torque, rl_torque, rr_torque)

    def shift_up(self) -> bool:
        """Shift up."""
        return self._transmission.shift_up()

    def shift_down(self) -> bool:
        """Shift down."""
        return self._transmission.shift_down()

    def shift_to(self, gear: int) -> bool:
        """Shift to specific gear."""
        return self._transmission.shift(gear)
