"""
Motion preferences system for UI accessibility.

Provides support for reduced motion preferences:
- System preference detection (prefers-reduced-motion)
- Animation enabling/disabling
- Alternative static states for animations
- Transition duration multiplier
- Parallax effect disabling

Reference (ARCHITECTURE_UI.md):
- Reduce Motion: Animation preferences
- Visual Accessibility: Motion sensitivity support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class MotionPreference(Enum):
    """User motion preference levels."""
    NO_PREFERENCE = auto()  # Use default animations
    REDUCE = auto()         # Reduce non-essential motion
    NONE = auto()           # Disable all motion/animation


class ReducedMotionLevel(Enum):
    """Levels of motion reduction."""
    OFF = auto()           # No reduction - full animations
    SUBTLE = auto()        # Reduce amplitude and distance
    MINIMAL = auto()       # Only essential animations
    STATIC = auto()        # No animations - instant state changes


class AnimationCategory(Enum):
    """Categories of animations for selective control."""
    ESSENTIAL = auto()     # Required for understanding (loading indicators)
    DECORATIVE = auto()    # Purely visual enhancement
    TRANSITION = auto()    # State transitions (page changes, modals)
    FEEDBACK = auto()      # User interaction feedback (button press)
    PARALLAX = auto()      # Scrolling parallax effects
    BACKGROUND = auto()    # Background animations
    HOVER = auto()         # Hover state animations
    ATTENTION = auto()     # Attention-grabbing animations


# Motion reduction multipliers for different levels
# These values control how much animations are scaled down
MOTION_MULTIPLIER_MINIMAL_DURATION = 0.3   # Duration multiplier for minimal motion
MOTION_MULTIPLIER_SUBTLE_DURATION = 0.6    # Duration multiplier for subtle motion
MOTION_MULTIPLIER_MINIMAL_DISTANCE = 0.2   # Distance multiplier for minimal motion
MOTION_MULTIPLIER_SUBTLE_DISTANCE = 0.5    # Distance multiplier for subtle motion

# Default animation timing values (in seconds)
DEFAULT_ANIMATION_DURATION = 0.3
DEFAULT_REDUCED_DURATION = 0.1
DEFAULT_MINIMUM_DURATION = 0.0

# Default motion distance multipliers
DEFAULT_MOTION_DISTANCE = 1.0
DEFAULT_REDUCED_MOTION_DISTANCE = 0.3

# Duration multiplier limits
MIN_DURATION_MULTIPLIER = 0.0
MAX_DURATION_MULTIPLIER = 3.0


@dataclass
class AnimationPreference:
    """
    Configuration for a specific animation.

    Defines how an animation should behave under different
    motion preference settings.
    """
    animation_id: str
    category: AnimationCategory = AnimationCategory.DECORATIVE

    # Duration settings
    default_duration: float = DEFAULT_ANIMATION_DURATION  # seconds
    reduced_duration: float = DEFAULT_REDUCED_DURATION  # seconds when reduced motion
    minimum_duration: float = DEFAULT_MINIMUM_DURATION  # minimum allowed duration

    # Motion settings
    default_distance: float = DEFAULT_MOTION_DISTANCE  # multiplier for movement distance
    reduced_distance: float = DEFAULT_REDUCED_MOTION_DISTANCE  # multiplier when reduced motion

    # Alternative behaviors
    can_be_disabled: bool = True
    static_alternative: bool = True  # Has a static fallback

    # Timing function adjustments
    use_reduced_easing: bool = True  # Use simpler easing when reduced

    def get_duration(self, level: ReducedMotionLevel) -> float:
        """Get the appropriate duration for a motion level."""
        if level == ReducedMotionLevel.OFF:
            return self.default_duration
        if level == ReducedMotionLevel.STATIC:
            return 0.0
        if level in (ReducedMotionLevel.SUBTLE, ReducedMotionLevel.MINIMAL):
            return max(self.minimum_duration, self.reduced_duration)
        return self.default_duration

    def get_distance_multiplier(self, level: ReducedMotionLevel) -> float:
        """Get the distance multiplier for a motion level."""
        if level == ReducedMotionLevel.OFF:
            return self.default_distance
        if level == ReducedMotionLevel.STATIC:
            return 0.0
        if level in (ReducedMotionLevel.SUBTLE, ReducedMotionLevel.MINIMAL):
            return self.reduced_distance
        return self.default_distance

    def should_animate(self, level: ReducedMotionLevel) -> bool:
        """Check if this animation should play at the given level."""
        if level == ReducedMotionLevel.OFF:
            return True
        if level == ReducedMotionLevel.STATIC:
            return not self.can_be_disabled
        if level == ReducedMotionLevel.MINIMAL:
            return self.category == AnimationCategory.ESSENTIAL
        if level == ReducedMotionLevel.SUBTLE:
            return self.category in (
                AnimationCategory.ESSENTIAL,
                AnimationCategory.FEEDBACK,
                AnimationCategory.TRANSITION,
            )
        return True


@dataclass
@dataclass
class MotionConfig:
    """
    Configuration for motion preferences.

    Defines global motion settings and category-level controls.
    """
    # Global settings
    motion_level: ReducedMotionLevel = ReducedMotionLevel.OFF
    system_preference: MotionPreference = MotionPreference.NO_PREFERENCE

    # Duration multiplier (applied to all animations)
    duration_multiplier: float = 1.0
    min_duration_multiplier: float = MIN_DURATION_MULTIPLIER
    max_duration_multiplier: float = MAX_DURATION_MULTIPLIER
    reduced_duration_multiplier: float = 0.0

    # Essential animations setting
    allow_essential_animations: bool = True

    # Category toggles
    enable_decorative: bool = True
    enable_parallax: bool = True
    enable_background: bool = True
    enable_hover: bool = True
    enable_attention: bool = True

    # Transition settings
    instant_transitions: bool = False  # Skip all transitions
    crossfade_only: bool = False       # Only allow opacity transitions

    # Easing
    use_simplified_easing: bool = False  # Linear instead of complex easing

    def clamp_duration_multiplier(self, multiplier: float) -> float:
        """Clamp a duration multiplier to valid range."""
        return max(
            self.min_duration_multiplier,
            min(self.max_duration_multiplier, multiplier),
        )

    def is_category_enabled(self, category: AnimationCategory) -> bool:
        """Check if a category of animation is enabled."""
        if category == AnimationCategory.ESSENTIAL:
            return True  # Always allow essential
        if category == AnimationCategory.DECORATIVE:
            return self.enable_decorative
        if category == AnimationCategory.PARALLAX:
            return self.enable_parallax
        if category == AnimationCategory.BACKGROUND:
            return self.enable_background
        if category == AnimationCategory.HOVER:
            return self.enable_hover
        if category == AnimationCategory.ATTENTION:
            return self.enable_attention
        return True


@dataclass
class MotionChangeEvent:
    """
    Event fired when motion settings change.

    Contains old and new settings for comparison.
    """
    old_level: ReducedMotionLevel
    new_level: ReducedMotionLevel
    old_multiplier: float
    new_multiplier: float
    source: str = "user"  # "user", "system", "auto"

    @property
    def level_changed(self) -> bool:
        """Check if motion level changed."""
        return self.old_level != self.new_level

    @property
    def multiplier_changed(self) -> bool:
        """Check if duration multiplier changed."""
        return self.old_multiplier != self.new_multiplier

    @property
    def became_more_restrictive(self) -> bool:
        """Check if motion became more restricted."""
        levels = list(ReducedMotionLevel)
        return levels.index(self.new_level) > levels.index(self.old_level)


class MotionManager:
    """
    Singleton manager for motion preferences.

    Coordinates animation settings, system preference detection,
    and motion reduction across the UI system.
    """

    _instance: Optional["MotionManager"] = None
    _singleton_mode: bool = False  # Set to True for production singleton behavior

    def __new__(cls, preference: MotionPreference = MotionPreference.NO_PREFERENCE, config: Optional[MotionConfig] = None) -> "MotionManager":
        if cls._singleton_mode and cls._instance is not None:
            return cls._instance
        cls._instance = super().__new__(cls)
        cls._instance._initialized = False
        return cls._instance

    def __init__(self, preference: MotionPreference = MotionPreference.NO_PREFERENCE, config: Optional[MotionConfig] = None) -> None:
        if self._initialized and type(self)._singleton_mode:
            # Allow re-setting preference on existing instance
            if preference != MotionPreference.NO_PREFERENCE:
                self._preference = preference
            return

        self._initialized = True

        # User preference
        self._preference = preference

        # Configuration
        self._config = config if config is not None else MotionConfig()

        # Animation preferences registry
        self._animations: dict[str, AnimationPreference] = {}

        # Callbacks
        self._motion_callbacks: list[Callable[[MotionChangeEvent], None]] = []
        self._preference_callbacks: list[Callable[[MotionPreference], None]] = []

        # Enabled state
        self._enabled: bool = True

        # System preference cache
        self._system_reduced_motion: bool = False

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def preference(self) -> MotionPreference:
        """Get the motion preference."""
        return self._preference

    @preference.setter
    def preference(self, value: MotionPreference) -> None:
        """Set the motion preference."""
        if self._preference != value:
            old_pref = self._preference
            self._preference = value
            for callback in self._preference_callbacks:
                callback(old_pref, value)

    @property
    def prefers_reduced_motion(self) -> bool:
        """Check if reduced motion is preferred."""
        return self._preference in (MotionPreference.REDUCE, MotionPreference.NONE)

    @property
    def enabled(self) -> bool:
        """Check if motion management is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable motion management."""
        self._enabled = value

    @property
    def config(self) -> MotionConfig:
        """Get the motion configuration."""
        return self._config

    @property
    def motion_level(self) -> ReducedMotionLevel:
        """Get the current motion reduction level."""
        return self._config.motion_level

    @property
    def duration_multiplier(self) -> float:
        """Get the current duration multiplier."""
        return self._config.duration_multiplier if self._enabled else 1.0

    @property
    def is_reduced(self) -> bool:
        """Check if motion is currently reduced."""
        return self._config.motion_level != ReducedMotionLevel.OFF

    @property
    def is_static(self) -> bool:
        """Check if motion is fully disabled (static mode)."""
        return self._config.motion_level == ReducedMotionLevel.STATIC

    @property
    def parallax_enabled(self) -> bool:
        """Check if parallax effects are enabled."""
        return (
            self._config.enable_parallax and
            self._config.motion_level in (
                ReducedMotionLevel.OFF,
                ReducedMotionLevel.SUBTLE,
            )
        )

    @property
    def animations_allowed(self) -> bool:
        """Check if animations are allowed."""
        return self._preference == MotionPreference.NO_PREFERENCE

    @property
    def essential_animations_allowed(self) -> bool:
        """Check if essential animations are allowed (always True)."""
        return True

    @property
    def transitions_allowed(self) -> bool:
        """Check if transitions are allowed."""
        return self._preference == MotionPreference.NO_PREFERENCE

    @property
    def parallax_allowed(self) -> bool:
        """Check if parallax effects are allowed."""
        return self._preference == MotionPreference.NO_PREFERENCE

    @property
    def autoplay_allowed(self) -> bool:
        """Check if video autoplay is allowed."""
        return self._preference == MotionPreference.NO_PREFERENCE

    @property
    def blinking_allowed(self) -> bool:
        """Check if blinking/flashing elements are allowed."""
        return self._preference == MotionPreference.NO_PREFERENCE

    def get_animation_duration(self, duration: float) -> float:
        """Get animation duration based on preference."""
        if self._preference == MotionPreference.NO_PREFERENCE:
            return duration
        return duration * self._config.reduced_duration_multiplier

    def get_transition_duration(self, duration: float) -> float:
        """Get transition duration based on preference."""
        if self._preference == MotionPreference.NO_PREFERENCE:
            return duration
        return 0.0  # Instant

    def get_parallax_factor(self, factor: float) -> float:
        """Get parallax factor based on preference."""
        if self._preference == MotionPreference.NO_PREFERENCE:
            return factor
        return 0.0  # No parallax

    def on_preference_changed(self, callback: Callable) -> Callable[[], None]:
        """Register a callback for preference changes."""
        self._preference_callbacks.append(callback)
        return lambda: self._preference_callbacks.remove(callback)

    def remove_preference_callback(self, callback: Callable) -> None:
        """Remove a preference change callback."""
        if callback in self._preference_callbacks:
            self._preference_callbacks.remove(callback)

    # Motion level management
    def set_motion_level(
        self,
        level: ReducedMotionLevel,
        source: str = "user",
    ) -> None:
        """Set the motion reduction level."""
        old_level = self._config.motion_level
        if old_level == level:
            return

        self._config.motion_level = level
        self._fire_motion_change(old_level, level, source=source)

    def set_from_preference(
        self,
        preference: MotionPreference,
        source: str = "user",
    ) -> None:
        """Set motion level from a preference enum."""
        level_map = {
            MotionPreference.NO_PREFERENCE: ReducedMotionLevel.OFF,
            MotionPreference.REDUCE: ReducedMotionLevel.MINIMAL,
            MotionPreference.NONE: ReducedMotionLevel.STATIC,
        }
        level = level_map.get(preference, ReducedMotionLevel.OFF)
        self.set_motion_level(level, source)

    # Duration multiplier
    def set_duration_multiplier(
        self,
        multiplier: float,
        source: str = "user",
    ) -> None:
        """Set the global duration multiplier."""
        old_multiplier = self._config.duration_multiplier
        new_multiplier = self._config.clamp_duration_multiplier(multiplier)

        if old_multiplier == new_multiplier:
            return

        self._config.duration_multiplier = new_multiplier
        self._fire_motion_change(
            self._config.motion_level,
            self._config.motion_level,
            old_multiplier,
            new_multiplier,
            source,
        )

    # System preference detection
    def detect_system_preference(self) -> MotionPreference:
        """
        Detect system reduced motion preference.

        Checks for prefers-reduced-motion media query equivalent.

        Returns:
            The detected motion preference.
        """
        # Platform-specific detection would go here
        # Windows: SystemParametersInfo SPI_GETCLIENTAREAANIMATION
        # macOS: NSWorkspace.accessibilityDisplayShouldReduceMotion
        # Linux: GTK settings / GNOME accessibility
        # Web: window.matchMedia('(prefers-reduced-motion: reduce)')
        if self._system_reduced_motion:
            return MotionPreference.REDUCE
        return MotionPreference.NO_PREFERENCE

    def set_system_reduced_motion(self, reduced: bool) -> None:
        """Set system reduced motion state (for testing or manual override)."""
        self._system_reduced_motion = reduced
        self._config.system_preference = (
            MotionPreference.REDUCE if reduced else MotionPreference.NO_PREFERENCE
        )

    def apply_system_preference(self) -> bool:
        """
        Apply the detected system preference.

        Returns True if a preference was applied.
        """
        preference = self.detect_system_preference()
        if preference != MotionPreference.NO_PREFERENCE:
            self.set_from_preference(preference, source="system")
            return True
        return False

    # Category controls
    def set_category_enabled(
        self,
        category: AnimationCategory,
        enabled: bool,
    ) -> None:
        """Enable or disable a category of animations."""
        if category == AnimationCategory.DECORATIVE:
            self._config.enable_decorative = enabled
        elif category == AnimationCategory.PARALLAX:
            self._config.enable_parallax = enabled
        elif category == AnimationCategory.BACKGROUND:
            self._config.enable_background = enabled
        elif category == AnimationCategory.HOVER:
            self._config.enable_hover = enabled
        elif category == AnimationCategory.ATTENTION:
            self._config.enable_attention = enabled

    def is_category_enabled(self, category: AnimationCategory) -> bool:
        """Check if a category of animations is enabled."""
        if not self._enabled:
            return True
        return self._config.is_category_enabled(category)

    def disable_parallax(self) -> None:
        """Convenience method to disable parallax effects."""
        self._config.enable_parallax = False

    def enable_parallax(self) -> None:
        """Convenience method to enable parallax effects."""
        self._config.enable_parallax = True

    # Animation registration
    def register_animation(self, animation: AnimationPreference) -> None:
        """Register an animation preference."""
        self._animations[animation.animation_id] = animation

    def unregister_animation(self, animation_id: str) -> None:
        """Unregister an animation preference."""
        self._animations.pop(animation_id, None)

    def get_animation(self, animation_id: str) -> Optional[AnimationPreference]:
        """Get an animation preference by ID."""
        return self._animations.get(animation_id)

    # Animation control
    def should_animate(
        self,
        animation_id: Optional[str] = None,
        category: Optional[AnimationCategory] = None,
    ) -> bool:
        """
        Check if an animation should play.

        Args:
            animation_id: Specific animation ID to check
            category: Category to check if no ID specified

        Returns:
            True if the animation should play.
        """
        if not self._enabled:
            return True

        level = self._config.motion_level

        # Check specific animation
        if animation_id:
            anim = self._animations.get(animation_id)
            if anim:
                if not self._config.is_category_enabled(anim.category):
                    return False
                return anim.should_animate(level)

        # Check category
        if category:
            if not self._config.is_category_enabled(category):
                return False

            # Check level restrictions
            if level == ReducedMotionLevel.STATIC:
                return category == AnimationCategory.ESSENTIAL
            if level == ReducedMotionLevel.MINIMAL:
                return category in (
                    AnimationCategory.ESSENTIAL,
                    AnimationCategory.FEEDBACK,
                )

        return level != ReducedMotionLevel.STATIC

    def get_duration(
        self,
        base_duration: float,
        animation_id: Optional[str] = None,
    ) -> float:
        """
        Get the effective duration for an animation.

        Args:
            base_duration: The base duration in seconds
            animation_id: Optional animation ID for specific settings

        Returns:
            The effective duration in seconds.
        """
        if not self._enabled:
            return base_duration

        level = self._config.motion_level
        multiplier = self._config.duration_multiplier

        # Check for static mode
        if level == ReducedMotionLevel.STATIC:
            return 0.0

        # Check specific animation
        if animation_id:
            anim = self._animations.get(animation_id)
            if anim:
                return anim.get_duration(level) * multiplier

        # Apply level reduction
        if level == ReducedMotionLevel.MINIMAL:
            return base_duration * MOTION_MULTIPLIER_MINIMAL_DURATION * multiplier
        if level == ReducedMotionLevel.SUBTLE:
            return base_duration * MOTION_MULTIPLIER_SUBTLE_DURATION * multiplier

        return base_duration * multiplier

    def get_distance_multiplier(
        self,
        animation_id: Optional[str] = None,
    ) -> float:
        """
        Get the distance multiplier for motion.

        Args:
            animation_id: Optional animation ID for specific settings

        Returns:
            A multiplier for movement distance (0.0 to 1.0).
        """
        if not self._enabled:
            return 1.0

        level = self._config.motion_level

        # Check specific animation
        if animation_id:
            anim = self._animations.get(animation_id)
            if anim:
                return anim.get_distance_multiplier(level)

        # Default multipliers
        if level == ReducedMotionLevel.STATIC:
            return 0.0
        if level == ReducedMotionLevel.MINIMAL:
            return MOTION_MULTIPLIER_MINIMAL_DISTANCE
        if level == ReducedMotionLevel.SUBTLE:
            return MOTION_MULTIPLIER_SUBTLE_DISTANCE

        return DEFAULT_MOTION_DISTANCE

    def get_easing(
        self,
        default_easing: str,
        animation_id: Optional[str] = None,
    ) -> str:
        """
        Get the appropriate easing function.

        Args:
            default_easing: The default easing function name
            animation_id: Optional animation ID for specific settings

        Returns:
            The easing function name to use.
        """
        if not self._enabled:
            return default_easing

        if self._config.use_simplified_easing:
            return "linear"

        # Check specific animation
        if animation_id:
            anim = self._animations.get(animation_id)
            if anim and anim.use_reduced_easing:
                if self._config.motion_level != ReducedMotionLevel.OFF:
                    return "ease-out"  # Simple, quick easing

        return default_easing

    # Transition controls
    def should_use_transitions(self) -> bool:
        """Check if transitions should be used."""
        if not self._enabled:
            return True
        return not self._config.instant_transitions

    def should_use_crossfade_only(self) -> bool:
        """Check if only crossfade transitions should be used."""
        if not self._enabled:
            return False
        return self._config.crossfade_only or self.is_reduced

    def set_instant_transitions(self, instant: bool) -> None:
        """Enable or disable instant transitions."""
        self._config.instant_transitions = instant

    def set_crossfade_only(self, crossfade_only: bool) -> None:
        """Enable or disable crossfade-only mode."""
        self._config.crossfade_only = crossfade_only

    # Motion change callbacks
    def add_motion_callback(
        self,
        callback: Callable[[MotionChangeEvent], None],
    ) -> None:
        """Add a callback for motion setting changes."""
        self._motion_callbacks.append(callback)

    def remove_motion_callback(
        self,
        callback: Callable[[MotionChangeEvent], None],
    ) -> None:
        """Remove a motion change callback."""
        if callback in self._motion_callbacks:
            self._motion_callbacks.remove(callback)

    def _fire_motion_change(
        self,
        old_level: ReducedMotionLevel,
        new_level: ReducedMotionLevel,
        old_multiplier: Optional[float] = None,
        new_multiplier: Optional[float] = None,
        source: str = "user",
    ) -> None:
        """Fire motion change event."""
        event = MotionChangeEvent(
            old_level=old_level,
            new_level=new_level,
            old_multiplier=old_multiplier or self._config.duration_multiplier,
            new_multiplier=new_multiplier or self._config.duration_multiplier,
            source=source,
        )

        for callback in self._motion_callbacks:
            callback(event)

    # Utility
    def reset(self) -> None:
        """Reset all motion settings to defaults."""
        old_level = self._config.motion_level
        old_multiplier = self._config.duration_multiplier

        self._config = MotionConfig()
        self._preference = MotionPreference.NO_PREFERENCE

        self._fire_motion_change(
            old_level,
            self._config.motion_level,
            old_multiplier,
            self._config.duration_multiplier,
            "reset",
        )

    def clear(self) -> None:
        """Clear all custom data."""
        self._animations.clear()
        self._motion_callbacks.clear()
