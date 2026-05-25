"""
UI Accessibility System.

Provides comprehensive accessibility features for the game engine UI:
- Screen reader support with ARIA roles, properties, and states
- Keyboard navigation with tab order and arrow key support
- High contrast mode for visual accessibility
- UI scaling for DPI awareness and font scaling
- Motion preferences for reduced motion support

Architecture Reference (ARCHITECTURE_UI.md):
- Visual Accessibility: High contrast, text scaling, colorblind modes, reduce motion
- Audio Accessibility: Subtitles, closed captions, visual cues
- Input Accessibility: Input remapping, hold to press, one-handed
- Screen Reader Support: Announcements, navigation, landmarks

Integration with Trinity Pattern:
- Uses TrackedDescriptor for state change tracking
- Uses ObservableDescriptor for accessibility announcements
- Uses ValidatedDescriptor for accessibility property validation
"""

from .screen_reader import (
    AccessibilityManager,
    AriaLiveRegion,
    AriaProperty,
    AriaRole,
    AriaState,
    FocusAnnouncement,
    LiveRegionPoliteness,
)
from .keyboard_nav import (
    FocusDirection,
    KeyboardNavigator,
    NavigationGroup,
    SkipLink,
    TabOrder,
    TabStop,
)
from .high_contrast import (
    ContrastLevel,
    ContrastMode,
    FocusIndicator,
    FocusIndicatorStyle,
    HighContrastManager,
    IconAlternative,
)
from .scale import (
    DPIAwareness,
    ScaleManager,
    ScaleMode,
    TouchTargetSize,
)
from .motion import (
    AnimationPreference,
    MotionManager,
    MotionPreference,
    ReducedMotionLevel,
)

__all__ = [
    # Screen Reader
    "AccessibilityManager",
    "AriaRole",
    "AriaProperty",
    "AriaState",
    "AriaLiveRegion",
    "LiveRegionPoliteness",
    "FocusAnnouncement",
    # Keyboard Navigation
    "KeyboardNavigator",
    "TabOrder",
    "TabStop",
    "NavigationGroup",
    "FocusDirection",
    "SkipLink",
    # High Contrast
    "HighContrastManager",
    "ContrastMode",
    "ContrastLevel",
    "FocusIndicator",
    "FocusIndicatorStyle",
    "IconAlternative",
    # Scaling
    "ScaleManager",
    "ScaleMode",
    "DPIAwareness",
    "TouchTargetSize",
    # Motion
    "MotionManager",
    "MotionPreference",
    "ReducedMotionLevel",
    "AnimationPreference",
]
