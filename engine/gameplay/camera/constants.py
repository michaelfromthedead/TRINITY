"""Camera System Constants.

This module contains all magic numbers and default values for the camera system,
including FOV settings, distances, collision parameters, shake effects, blending
durations, and rail configurations.
"""

from __future__ import annotations

import math

# =============================================================================
# Field of View Constants
# =============================================================================

# Default field of view in degrees
DEFAULT_FOV: float = 90.0

# Minimum allowed field of view in degrees
MIN_FOV: float = 60.0

# Maximum allowed field of view in degrees
MAX_FOV: float = 120.0

# FOV for zoomed/ADS view
ZOOMED_FOV: float = 45.0

# Sprint FOV increase
SPRINT_FOV_INCREASE: float = 10.0

# FOV transition speed (degrees per second)
FOV_TRANSITION_SPEED: float = 180.0

# =============================================================================
# Camera Distance Constants
# =============================================================================

# Default camera distance for third-person view
DEFAULT_CAMERA_DISTANCE: float = 300.0

# Minimum camera distance
MIN_CAMERA_DISTANCE: float = 100.0

# Maximum camera distance
MAX_CAMERA_DISTANCE: float = 1000.0

# Orbit camera default distance
DEFAULT_ORBIT_DISTANCE: float = 500.0

# Orbit minimum distance (zoom in limit)
MIN_ORBIT_DISTANCE: float = 50.0

# Orbit maximum distance (zoom out limit)
MAX_ORBIT_DISTANCE: float = 2000.0

# Zoom speed (units per second)
DEFAULT_ZOOM_SPEED: float = 200.0

# Scroll wheel zoom sensitivity
SCROLL_ZOOM_SENSITIVITY: float = 50.0

# =============================================================================
# Third-Person Camera Constants
# =============================================================================

# Boom arm length default
DEFAULT_BOOM_ARM_LENGTH: float = 300.0

# Socket offset from character (local space)
DEFAULT_SOCKET_OFFSET_X: float = 0.0
DEFAULT_SOCKET_OFFSET_Y: float = 60.0  # Above character center
DEFAULT_SOCKET_OFFSET_Z: float = 0.0

# Target offset (where camera looks relative to character)
DEFAULT_TARGET_OFFSET_X: float = 0.0
DEFAULT_TARGET_OFFSET_Y: float = 50.0  # Slightly above character
DEFAULT_TARGET_OFFSET_Z: float = 0.0

# Camera lag speed (0 = instant, lower = smoother)
DEFAULT_CAMERA_LAG_SPEED: float = 10.0

# Camera rotation lag speed
DEFAULT_ROTATION_LAG_SPEED: float = 15.0

# Maximum lag distance
MAX_LAG_DISTANCE: float = 200.0

# Lag recovery speed when lag exceeds max
LAG_RECOVERY_SPEED: float = 50.0

# =============================================================================
# First-Person Camera Constants
# =============================================================================

# Eye height from character origin
DEFAULT_EYE_HEIGHT: float = 170.0

# Crouching eye height
CROUCH_EYE_HEIGHT: float = 100.0

# Head bob amplitude
DEFAULT_HEAD_BOB_AMPLITUDE: float = 2.0

# Head bob frequency (cycles per step)
DEFAULT_HEAD_BOB_FREQUENCY: float = 2.0

# Head bob sway amplitude (side to side)
DEFAULT_HEAD_BOB_SWAY: float = 1.0

# Weapon sway amplitude
WEAPON_SWAY_AMPLITUDE: float = 0.5

# Landing impact scale
LANDING_IMPACT_SCALE: float = 5.0

# =============================================================================
# Collision Constants
# =============================================================================

# Collision probe radius for sphere casting
COLLISION_PROBE_RADIUS: float = 10.0

# Minimum safe distance from collision surface
MIN_COLLISION_DISTANCE: float = 5.0

# Collision pull-in speed (units per second)
COLLISION_PULL_IN_SPEED: float = 500.0

# Collision push-out speed
COLLISION_PUSH_OUT_SPEED: float = 200.0

# Collision detection layer mask (default: world geometry)
DEFAULT_COLLISION_MASK: int = 1

# Maximum number of collision probes per frame
MAX_COLLISION_PROBES: int = 8

# Collision interpolation smoothness
COLLISION_INTERP_SPEED: float = 20.0

# =============================================================================
# Camera Shake Constants
# =============================================================================

# Shake trauma decay rate (per second)
SHAKE_DECAY_RATE: float = 0.9

# Trauma exponent for non-linear shake intensity
SHAKE_TRAUMA_EXPONENT: float = 2.0

# Maximum trauma value
MAX_TRAUMA: float = 1.0

# Default shake frequency (Hz)
DEFAULT_SHAKE_FREQUENCY: float = 25.0

# Default shake amplitude (translation)
DEFAULT_SHAKE_AMPLITUDE_TRANSLATION: float = 5.0

# Default shake amplitude (rotation in degrees)
DEFAULT_SHAKE_AMPLITUDE_ROTATION: float = 2.0

# Perlin noise octaves for shake
SHAKE_NOISE_OCTAVES: int = 3

# Perlin noise persistence
SHAKE_NOISE_PERSISTENCE: float = 0.5

# Explosion shake trauma
EXPLOSION_SHAKE_TRAUMA: float = 0.8

# Damage shake trauma
DAMAGE_SHAKE_TRAUMA: float = 0.3

# Footstep shake trauma
FOOTSTEP_SHAKE_TRAUMA: float = 0.05

# =============================================================================
# Blend Duration Constants
# =============================================================================

# Instant cut (no blend)
BLEND_DURATION_CUT: float = 0.0

# Fast ease transition
BLEND_DURATION_FAST: float = 0.25

# Standard ease transition
BLEND_DURATION_EASE: float = 0.5

# Smooth cinematic transition
BLEND_DURATION_SMOOTH: float = 1.0

# Long cinematic transition
BLEND_DURATION_LONG: float = 2.0

# Death camera blend duration
DEATH_CAMERA_BLEND: float = 1.5

# Vehicle enter/exit blend
VEHICLE_CAMERA_BLEND: float = 0.75

# ADS (aim down sights) blend
ADS_BLEND_DURATION: float = 0.2

# =============================================================================
# Rail and Path Constants
# =============================================================================

# Default spline resolution (points per unit length)
RAIL_SPLINE_RESOLUTION: int = 100

# Minimum rail segment length
MIN_RAIL_SEGMENT_LENGTH: float = 10.0

# Maximum rail points
MAX_RAIL_POINTS: int = 256

# Catmull-Rom spline tension
DEFAULT_SPLINE_TENSION: float = 0.5

# Rail follower default speed
DEFAULT_RAIL_SPEED: float = 100.0

# Dolly default speed
DEFAULT_DOLLY_SPEED: float = 50.0

# Crane arc angle range (degrees)
DEFAULT_CRANE_ARC_ANGLE: float = 90.0

# Crane arm length
DEFAULT_CRANE_ARM_LENGTH: float = 500.0

# =============================================================================
# View Angle Limits
# =============================================================================

# Pitch limits (looking up/down) in degrees
MIN_PITCH_ANGLE: float = -89.0
MAX_PITCH_ANGLE: float = 89.0

# Orbit pitch limits
MIN_ORBIT_PITCH: float = -80.0
MAX_ORBIT_PITCH: float = 80.0

# Top-down camera pitch (fixed)
TOP_DOWN_PITCH: float = -90.0

# Isometric camera pitch (fixed)
ISOMETRIC_PITCH: float = -45.0

# Isometric camera rotation snap (degrees)
ISOMETRIC_ROTATION_SNAP: float = 45.0

# =============================================================================
# Mouse/Input Sensitivity
# =============================================================================

# Default mouse sensitivity
DEFAULT_MOUSE_SENSITIVITY: float = 0.15

# Minimum mouse sensitivity
MIN_MOUSE_SENSITIVITY: float = 0.01

# Maximum mouse sensitivity
MAX_MOUSE_SENSITIVITY: float = 2.0

# ADS sensitivity multiplier
ADS_SENSITIVITY_MULTIPLIER: float = 0.5

# Controller look sensitivity
DEFAULT_CONTROLLER_SENSITIVITY: float = 2.0

# Orbit rotation speed (degrees per second)
DEFAULT_ORBIT_ROTATION_SPEED: float = 180.0

# Free camera move speed
DEFAULT_FREE_CAM_SPEED: float = 500.0

# Free camera fast speed multiplier
FREE_CAM_FAST_MULTIPLIER: float = 3.0

# Free camera slow speed multiplier
FREE_CAM_SLOW_MULTIPLIER: float = 0.25

# =============================================================================
# Depth of Field Constants
# =============================================================================

# Default focus distance
DEFAULT_FOCUS_DISTANCE: float = 500.0

# Default aperture (f-stop)
DEFAULT_APERTURE: float = 2.8

# Default focal length (mm)
DEFAULT_FOCAL_LENGTH: float = 35.0

# Minimum focus distance
MIN_FOCUS_DISTANCE: float = 10.0

# Maximum focus distance
MAX_FOCUS_DISTANCE: float = 10000.0

# Auto-focus transition speed
AUTO_FOCUS_SPEED: float = 5.0

# =============================================================================
# Motion Blur Constants
# =============================================================================

# Default motion blur intensity
DEFAULT_MOTION_BLUR_INTENSITY: float = 0.5

# Maximum motion blur amount
MAX_MOTION_BLUR: float = 1.0

# Velocity threshold for motion blur activation
MOTION_BLUR_VELOCITY_THRESHOLD: float = 10.0

# Motion blur sample count
MOTION_BLUR_SAMPLES: int = 8

# =============================================================================
# Vignette Constants
# =============================================================================

# Default vignette intensity
DEFAULT_VIGNETTE_INTENSITY: float = 0.0

# Damage vignette intensity
DAMAGE_VIGNETTE_INTENSITY: float = 0.4

# Low health vignette intensity
LOW_HEALTH_VIGNETTE_INTENSITY: float = 0.6

# Vignette feather amount
DEFAULT_VIGNETTE_FEATHER: float = 0.4

# Vignette transition speed
VIGNETTE_TRANSITION_SPEED: float = 3.0

# =============================================================================
# Priority and Layer Constants
# =============================================================================

# Default camera priority
DEFAULT_CAMERA_PRIORITY: int = 0

# Cinematic camera priority (overrides gameplay)
CINEMATIC_CAMERA_PRIORITY: int = 100

# Cutscene camera priority
CUTSCENE_CAMERA_PRIORITY: int = 200

# Debug camera priority (highest)
DEBUG_CAMERA_PRIORITY: int = 1000

# =============================================================================
# Near/Far Plane Constants
# =============================================================================

# Default near clipping plane
DEFAULT_NEAR_PLANE: float = 0.1

# Default far clipping plane
DEFAULT_FAR_PLANE: float = 10000.0

# Minimum near plane
MIN_NEAR_PLANE: float = 0.01

# Maximum far plane
MAX_FAR_PLANE: float = 100000.0

# =============================================================================
# Timing Constants
# =============================================================================

# Minimum delta time to prevent division by zero
MIN_DELTA_TIME: float = 0.0001

# Maximum delta time to prevent huge jumps
MAX_DELTA_TIME: float = 0.1

# Camera update tick rate (Hz)
CAMERA_UPDATE_RATE: int = 60

# Interpolation history buffer size
INTERPOLATION_BUFFER_SIZE: int = 16

# =============================================================================
# Mathematical Constants
# =============================================================================

# Degrees to radians conversion
DEG_TO_RAD: float = math.pi / 180.0

# Radians to degrees conversion
RAD_TO_DEG: float = 180.0 / math.pi

# Epsilon for floating point comparisons
CAMERA_EPSILON: float = 1e-6

# Small angle threshold (radians)
SMALL_ANGLE_THRESHOLD: float = 0.001

# =============================================================================
# Trigger Volume Constants
# =============================================================================

# Default trigger check frequency
TRIGGER_CHECK_INTERVAL: float = 0.1

# Trigger hysteresis distance (prevents flickering)
TRIGGER_HYSTERESIS: float = 1.0

# Maximum active triggers
MAX_ACTIVE_TRIGGERS: int = 32

# =============================================================================
# Default Aspect Ratio
# =============================================================================

# Default screen aspect ratio (16:9)
DEFAULT_ASPECT_RATIO: float = 16.0 / 9.0

# =============================================================================
# Interpolation Speed Constants
# =============================================================================

# Head bob decay factor (per frame multiplier)
HEAD_BOB_DECAY_FACTOR: float = 0.9

# Eye height interpolation speed multiplier
EYE_HEIGHT_INTERP_SPEED: float = 10.0

# Boom arm length interpolation speed multiplier
BOOM_LENGTH_INTERP_SPEED: float = 5.0

# Orbit/isometric distance interpolation speed multiplier
DISTANCE_INTERP_SPEED: float = 5.0

# Default auto-rotate speed for orbit camera (degrees per second)
DEFAULT_AUTO_ROTATE_SPEED: float = 10.0

# Rotation transition speed multiplier for isometric camera
ISOMETRIC_ROTATION_TRANSITION_SPEED: float = 3.0

# =============================================================================
# Collision Interpolation Constants
# =============================================================================

# Collision fade distance default
DEFAULT_FADE_DISTANCE: float = 50.0

# Collision interpolation factor for pull-in/push-out
COLLISION_INTERP_FACTOR: float = 0.01

# Minimum fade alpha for blend response
BLEND_RESPONSE_MIN_ALPHA: float = 0.3

# Blend response alpha range
BLEND_RESPONSE_ALPHA_RANGE: float = 0.7

# Pull-in weight threshold
PULL_IN_WEIGHT_THRESHOLD: float = 0.01

# =============================================================================
# Occlusion Detector Constants
# =============================================================================

# Default occlusion fade-in time (seconds)
DEFAULT_OCCLUSION_FADE_IN_TIME: float = 0.2

# Default occlusion fade-out time (seconds)
DEFAULT_OCCLUSION_FADE_OUT_TIME: float = 0.1

# Minimum occlusion alpha (never fully invisible)
MIN_OCCLUSION_ALPHA: float = 0.3

# =============================================================================
# Camera Shake Effect Constants
# =============================================================================

# Explosion shake default duration
EXPLOSION_SHAKE_DURATION: float = 0.5

# Damage shake default duration
DAMAGE_SHAKE_DURATION: float = 0.2

# Footstep shake default duration
FOOTSTEP_SHAKE_DURATION: float = 0.1

# Footstep shake frequency override
FOOTSTEP_SHAKE_FREQUENCY: float = 50.0

# Shake instance rotation amplitude multiplier (relative to position)
SHAKE_ROTATION_AMPLITUDE_MULTIPLIER: float = 0.5

# Explosion shake base frequency
EXPLOSION_SHAKE_BASE_FREQUENCY: float = 30.0

# Explosion shake frequency variation
EXPLOSION_SHAKE_FREQUENCY_VARIATION: float = 10.0

# Explosion shake decay rate
EXPLOSION_SHAKE_DECAY_RATE: float = 5.0

# Explosion shake Z-axis amplitude multiplier
EXPLOSION_SHAKE_Z_MULTIPLIER: float = 0.3

# Impact shake rotation multiplier
IMPACT_SHAKE_ROTATION_MULTIPLIER: float = 2.0

# Continuous shake frequency X
CONTINUOUS_SHAKE_FREQ_X: float = 8.0

# Continuous shake frequency Y
CONTINUOUS_SHAKE_FREQ_Y: float = 12.0

# Continuous shake frequency Z
CONTINUOUS_SHAKE_FREQ_Z: float = 6.0

# Continuous shake amplitude X multiplier
CONTINUOUS_SHAKE_AMP_X: float = 0.3

# Continuous shake amplitude Y multiplier
CONTINUOUS_SHAKE_AMP_Y: float = 0.2

# Continuous shake amplitude Z multiplier
CONTINUOUS_SHAKE_AMP_Z: float = 0.1

# Continuous shake rotation frequencies
CONTINUOUS_SHAKE_ROT_FREQ_X: float = 10.0
CONTINUOUS_SHAKE_ROT_FREQ_Y: float = 7.0
CONTINUOUS_SHAKE_ROT_FREQ_Z: float = 9.0

# Continuous shake rotation amplitudes
CONTINUOUS_SHAKE_ROT_AMP_XY: float = 0.2
CONTINUOUS_SHAKE_ROT_AMP_Z: float = 0.1

# =============================================================================
# FOV Effect Constants
# =============================================================================

# Default FOV punch decay rate
DEFAULT_PUNCH_DECAY: float = 5.0

# FOV punch threshold (below this, snap to zero)
FOV_PUNCH_THRESHOLD: float = 0.1

# =============================================================================
# Tilt Effect Constants
# =============================================================================

# Default maximum tilt angle (degrees)
DEFAULT_MAX_TILT: float = 15.0

# Default tilt transition speed (degrees per second)
DEFAULT_TILT_TRANSITION_SPEED: float = 90.0

# Default auto-level speed (degrees per second)
DEFAULT_AUTO_LEVEL_SPEED: float = 30.0

# =============================================================================
# DOF Constants (Additional)
# =============================================================================

# Default bokeh blade count
DEFAULT_BOKEH_BLADES: int = 6

# Minimum aperture (f-stop)
MIN_APERTURE: float = 1.0

# Maximum aperture (f-stop)
MAX_APERTURE: float = 22.0

# =============================================================================
# Motion Blur Constants (Additional)
# =============================================================================

# Motion blur velocity normalization factor
MOTION_BLUR_VELOCITY_NORMALIZATION: float = 100.0

# =============================================================================
# Spline Constants (Additional)
# =============================================================================

# Tangent scale factor for Bezier splines
BEZIER_TANGENT_SCALE: float = 0.25

# Tangent calculation delta for numerical derivative
TANGENT_CALC_DELTA: float = 0.001

# =============================================================================
# Blend Easing Constants
# =============================================================================

# Elastic easing period
ELASTIC_EASING_PERIOD: float = 0.3

# Bounce easing coefficient
BOUNCE_EASING_COEFFICIENT: float = 7.5625

# Bounce easing divisor
BOUNCE_EASING_DIVISOR: float = 2.75
