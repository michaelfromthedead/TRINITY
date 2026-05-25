"""
Shared constants for UI widgets.

This module provides centralized constants to avoid magic numbers scattered
across widget implementations, ensuring consistency and maintainability.
"""

from __future__ import annotations


# =============================================================================
# COLOR CONSTANTS
# =============================================================================

class Colors:
    """Standard color palette for UI widgets."""

    # Primary colors
    PRIMARY = "#4A90D9"
    PRIMARY_HOVER = "#5BA0E9"
    PRIMARY_PRESSED = "#3A80C9"
    PRIMARY_BORDER = "#2A70B9"

    # Neutral colors
    WHITE = "#FFFFFF"
    BLACK = "#000000"
    TRANSPARENT = "#00000000"

    # Gray scale
    GRAY_50 = "#F9FAFB"
    GRAY_100 = "#F3F4F6"
    GRAY_200 = "#E5E7EB"
    GRAY_300 = "#D1D5DB"
    GRAY_400 = "#9CA3AF"
    GRAY_500 = "#6B7280"
    GRAY_600 = "#4B5563"
    GRAY_700 = "#374151"
    GRAY_800 = "#1F2937"
    GRAY_900 = "#111827"

    # Disabled state
    DISABLED_BG = "#CCCCCC"
    DISABLED_TEXT = "#888888"
    DISABLED_BORDER = "#AAAAAA"

    # Semantic colors
    SUCCESS = "#22C55E"
    SUCCESS_DARK = "#16A34A"
    WARNING = "#F59E0B"
    WARNING_DARK = "#D97706"
    ERROR = "#EF4444"
    ERROR_DARK = "#DC2626"
    INFO = "#3B82F6"
    INFO_DARK = "#2563EB"

    # Game-specific colors
    HEALTH = "#22C55E"
    HEALTH_BG = "#1F2937"
    MANA = "#3B82F6"
    STAMINA = "#F59E0B"
    SHIELD = "#3B82F6"
    EXPERIENCE = "#8B5CF6"

    # Rarity colors
    RARITY_COMMON = "#9CA3AF"
    RARITY_UNCOMMON = "#22C55E"
    RARITY_RARE = "#3B82F6"
    RARITY_EPIC = "#A855F7"
    RARITY_LEGENDARY = "#F59E0B"
    RARITY_MYTHIC = "#EF4444"

    # Damage type colors
    DAMAGE_PHYSICAL = "#FFFFFF"
    DAMAGE_MAGIC = "#A855F7"
    DAMAGE_FIRE = "#EF4444"
    DAMAGE_ICE = "#06B6D4"
    DAMAGE_LIGHTNING = "#FACC15"
    DAMAGE_POISON = "#22C55E"
    DAMAGE_HEAL = "#10B981"


# =============================================================================
# DIMENSION CONSTANTS
# =============================================================================

class Dimensions:
    """Standard dimensions for UI widgets."""

    # Border widths
    BORDER_NONE = 0.0
    BORDER_THIN = 1.0
    BORDER_NORMAL = 2.0
    BORDER_THICK = 3.0

    # Corner radius
    RADIUS_NONE = 0.0
    RADIUS_SMALL = 2.0
    RADIUS_NORMAL = 4.0
    RADIUS_MEDIUM = 6.0
    RADIUS_LARGE = 8.0
    RADIUS_PILL = 9999.0  # Large value for pill shape

    # Padding
    PADDING_NONE = 0.0
    PADDING_XS = 2.0
    PADDING_SM = 4.0
    PADDING_MD = 8.0
    PADDING_LG = 16.0
    PADDING_XL = 24.0

    # Spacing
    SPACING_XS = 2.0
    SPACING_SM = 4.0
    SPACING_MD = 8.0
    SPACING_LG = 16.0
    SPACING_XL = 24.0

    # Icon sizes
    ICON_XS = 12.0
    ICON_SM = 16.0
    ICON_MD = 20.0
    ICON_LG = 24.0
    ICON_XL = 32.0

    # Default widget sizes
    BUTTON_HEIGHT = 40.0
    BUTTON_WIDTH = 100.0
    CHECKBOX_SIZE = 20.0
    SLIDER_HEIGHT = 30.0
    SLIDER_WIDTH = 200.0
    SLIDER_TRACK_HEIGHT = 6.0
    SLIDER_THUMB_SIZE = 20.0
    PROGRESS_BAR_HEIGHT = 20.0
    PROGRESS_BAR_WIDTH = 200.0
    INVENTORY_SLOT_SIZE = 64.0
    MINIMAP_SIZE = 200.0
    HEALTH_BAR_HEIGHT = 20.0
    HEALTH_BAR_WIDTH = 200.0


# =============================================================================
# TYPOGRAPHY CONSTANTS
# =============================================================================

class Typography:
    """Standard typography settings for UI widgets."""

    # Font sizes
    FONT_XS = 10.0
    FONT_SM = 12.0
    FONT_MD = 14.0
    FONT_LG = 16.0
    FONT_XL = 20.0
    FONT_2XL = 24.0
    FONT_3XL = 30.0

    # Line heights
    LINE_HEIGHT_TIGHT = 1.1
    LINE_HEIGHT_NORMAL = 1.2
    LINE_HEIGHT_RELAXED = 1.5

    # Font weights
    WEIGHT_LIGHT = "light"
    WEIGHT_NORMAL = "normal"
    WEIGHT_BOLD = "bold"

    # Character width estimate (for rough sizing)
    CHAR_WIDTH_FACTOR = 0.6  # Approximate char width = font_size * 0.6


# =============================================================================
# ANIMATION CONSTANTS
# =============================================================================

class Animation:
    """Standard animation timings for UI widgets."""

    # Durations (in seconds)
    DURATION_INSTANT = 0.0
    DURATION_FAST = 0.1
    DURATION_NORMAL = 0.2
    DURATION_SLOW = 0.3
    DURATION_SLOWER = 0.5

    # Specific animations
    HOVER_TRANSITION = 0.15
    PRESS_TRANSITION = 0.1
    FOCUS_TRANSITION = 0.2
    VALUE_TRANSITION = 0.2
    TOOLTIP_DELAY = 0.5
    CURSOR_BLINK_INTERVAL = 0.5
    DAMAGE_PREVIEW_DURATION = 1.0
    DAMAGE_NUMBER_DURATION = 1.5

    # Easing factors
    DECELERATION_FACTOR = 0.95


# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================

class Thresholds:
    """Standard thresholds for UI widgets."""

    # Health bar
    LOW_HEALTH_PERCENT = 0.25

    # Damage numbers
    STACK_TIME_WINDOW = 0.2
    STACK_DISTANCE = 30.0
    MAX_STACK_COUNT = 5

    # Text
    MIN_FONT_SIZE = 6.0
    MAX_FONT_SIZE = 200.0

    # Zoom
    MIN_ZOOM = 0.5
    MAX_ZOOM = 4.0
    ZOOM_STEP = 0.25


# =============================================================================
# LIMITS
# =============================================================================

class Limits:
    """Maximum limits for UI widgets."""

    MAX_ACTIVE_DAMAGE_NUMBERS = 100
    MAX_MINIMAP_MARKERS = 100
    MAX_STACK_SIZE = 99
    MAX_TEXT_LINES = 1000
