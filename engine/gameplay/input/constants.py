"""Constants for the gameplay input system.

This module contains all magic numbers and default values for input processing,
dead zones, smoothing, combo timing, and other input-related configurations.
"""

from __future__ import annotations

# =============================================================================
# Dead Zone Constants
# =============================================================================

# Default dead zone for analog stick input (0.0 to 1.0)
DEFAULT_DEAD_ZONE = 0.15

# Radial dead zone for combined X/Y stick input
DEFAULT_RADIAL_DEAD_ZONE = 0.15

# Dead zone for analog triggers
DEFAULT_TRIGGER_DEAD_ZONE = 0.1

# Outer dead zone (for snap-to-max behavior)
DEFAULT_OUTER_DEAD_ZONE = 0.95

# =============================================================================
# Response Curve Constants
# =============================================================================

# Default exponent for power curves (1.0 = linear, 2.0 = quadratic, etc.)
DEFAULT_RESPONSE_EXPONENT = 1.0

# S-curve inflection point (0.5 = centered)
DEFAULT_SCURVE_MIDPOINT = 0.5

# S-curve steepness factor
DEFAULT_SCURVE_STEEPNESS = 2.0

# Maximum allowed response exponent to prevent extreme sensitivity
MAX_RESPONSE_EXPONENT = 5.0

# =============================================================================
# Smoothing Constants
# =============================================================================

# Default smoothing factor (0.0 = no smoothing, 1.0 = full smoothing)
DEFAULT_SMOOTHING_FACTOR = 0.0

# Maximum smoothing factor to prevent excessive input lag
MAX_SMOOTHING_FACTOR = 0.9

# Smoothing sample window size for averaging
DEFAULT_SMOOTHING_WINDOW = 3

# Exponential smoothing alpha (higher = less smoothing)
DEFAULT_SMOOTHING_ALPHA = 0.5

# =============================================================================
# Combo Detection Constants
# =============================================================================

# Default time window for combo inputs (seconds)
DEFAULT_COMBO_WINDOW = 0.5

# Maximum time between inputs in a combo sequence (seconds)
DEFAULT_COMBO_INPUT_TIMEOUT = 0.3

# Minimum time for a hold input to register (seconds)
DEFAULT_HOLD_THRESHOLD = 0.2

# Maximum time for a tap input (quick press) (seconds)
DEFAULT_TAP_THRESHOLD = 0.15

# Double-tap detection window (seconds)
DEFAULT_DOUBLE_TAP_WINDOW = 0.25

# Maximum combo sequence length
MAX_COMBO_LENGTH = 16

# Input buffer size (number of inputs to remember)
DEFAULT_INPUT_BUFFER_SIZE = 32

# Buffer entry lifetime (seconds)
DEFAULT_BUFFER_LIFETIME = 1.0

# =============================================================================
# Context Constants
# =============================================================================

# Maximum context stack depth
MAX_CONTEXT_STACK_DEPTH = 16

# Default context priority
DEFAULT_CONTEXT_PRIORITY = 0

# Modal context priority (highest)
MODAL_CONTEXT_PRIORITY = 1000

# =============================================================================
# Device Constants
# =============================================================================

# Default poll rate for devices (Hz)
DEFAULT_DEVICE_POLL_RATE = 1000

# Hot-plug detection interval (seconds)
DEFAULT_HOTPLUG_INTERVAL = 1.0

# Maximum number of simultaneous devices per type
MAX_DEVICES_PER_TYPE = 8

# Maximum touch points for touch devices
MAX_TOUCH_POINTS = 10

# =============================================================================
# Axis Constants
# =============================================================================

# Default axis sensitivity multiplier
DEFAULT_AXIS_SENSITIVITY = 1.0

# Maximum axis sensitivity
MAX_AXIS_SENSITIVITY = 10.0

# Minimum axis sensitivity
MIN_AXIS_SENSITIVITY = 0.1

# Default axis acceleration
DEFAULT_AXIS_ACCELERATION = 0.0

# =============================================================================
# Motion Sensor Constants
# =============================================================================

# Gyroscope sensitivity default (degrees per second)
DEFAULT_GYRO_SENSITIVITY = 1.0

# Accelerometer range (G-force)
DEFAULT_ACCELEROMETER_RANGE = 2.0

# Motion data smoothing
DEFAULT_MOTION_SMOOTHING = 0.1

# =============================================================================
# Rebinding Constants
# =============================================================================

# Default conflict resolution mode
CONFLICT_RESOLUTION_REJECT = "reject"
CONFLICT_RESOLUTION_SWAP = "swap"
CONFLICT_RESOLUTION_DUPLICATE = "duplicate"

# Reserved bindings that cannot be rebound
RESERVED_BINDINGS = frozenset({
    "escape",  # Always used for menu/cancel
})

# Maximum number of bindings per action
MAX_BINDINGS_PER_ACTION = 4

# =============================================================================
# Timing Constants
# =============================================================================

# Minimum delta time to prevent division by zero
MIN_DELTA_TIME = 0.0001

# Maximum delta time to prevent huge jumps
MAX_DELTA_TIME = 0.1

# Input timestamp precision (seconds)
INPUT_TIMESTAMP_PRECISION = 0.001

# =============================================================================
# Mouse Constants
# =============================================================================

# Minimum mouse sensitivity
MIN_MOUSE_SENSITIVITY = 0.1

# Maximum mouse sensitivity
MAX_MOUSE_SENSITIVITY = 10.0

# Default mouse sensitivity
DEFAULT_MOUSE_SENSITIVITY = 1.0

# =============================================================================
# Quaternion Constants
# =============================================================================

# Minimum quaternion length for normalization (avoids division by near-zero)
MIN_QUATERNION_LENGTH = 0.0001

# =============================================================================
# Gravity Constants
# =============================================================================

# Standard gravity in m/s^2
STANDARD_GRAVITY = 9.81
