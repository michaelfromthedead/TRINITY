"""
Dialogue System Constants.

Contains configuration constants for the dialogue system including
typing speeds, timeouts, presentation settings, and validation limits.
"""

from enum import Enum, auto
from typing import Final

# =============================================================================
# Typing and Animation Constants
# =============================================================================

# Characters per second for text display
DEFAULT_TYPING_SPEED: Final[float] = 30.0
SLOW_TYPING_SPEED: Final[float] = 15.0
FAST_TYPING_SPEED: Final[float] = 60.0
INSTANT_TYPING_SPEED: Final[float] = float("inf")

# Minimum and maximum typing speeds
MIN_TYPING_SPEED: Final[float] = 1.0
MAX_TYPING_SPEED: Final[float] = 200.0

# =============================================================================
# Choice and Timeout Constants
# =============================================================================

# Default timeout for player choices (0 = no timeout)
DEFAULT_CHOICE_TIMEOUT: Final[float] = 0.0

# Minimum choice timeout when enabled
MIN_CHOICE_TIMEOUT: Final[float] = 1.0

# Maximum choice timeout
MAX_CHOICE_TIMEOUT: Final[float] = 300.0  # 5 minutes

# Default auto-advance delay in seconds
DEFAULT_AUTO_ADVANCE_DELAY: Final[float] = 2.0

# Default choice when timeout expires (-1 = first choice, -2 = exit dialogue)
TIMEOUT_DEFAULT_CHOICE: Final[int] = -1
TIMEOUT_EXIT_DIALOGUE: Final[int] = -2

# Maximum number of choices per node
MAX_CHOICES_PER_NODE: Final[int] = 10

# =============================================================================
# Presentation Constants
# =============================================================================

# Default text box dimensions (percentage of screen)
DEFAULT_TEXT_BOX_WIDTH: Final[float] = 0.8
DEFAULT_TEXT_BOX_HEIGHT: Final[float] = 0.25

# Portrait position options
class PortraitPosition(Enum):
    """Portrait display position."""
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()
    NONE = auto()

# Default portrait size (pixels)
DEFAULT_PORTRAIT_SIZE: Final[tuple[int, int]] = (128, 128)

# Text alignment options
class TextAlignment(Enum):
    """Text alignment in dialogue box."""
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    JUSTIFIED = auto()

# =============================================================================
# Node Type Constants
# =============================================================================

class NodeType(Enum):
    """Types of dialogue nodes."""
    TEXT = auto()       # NPC speech/narration
    CHOICE = auto()     # Player choice options
    BRANCH = auto()     # Conditional branching
    EVENT = auto()      # Trigger game events
    RANDOM = auto()     # Random variation selection
    ENTRY = auto()      # Dialogue entry point
    EXIT = auto()       # Dialogue exit point

# =============================================================================
# Variable Scope Constants
# =============================================================================

class VariableScope(Enum):
    """Variable storage scope."""
    LOCAL = auto()      # Per-conversation (cleared on exit)
    GLOBAL = auto()     # Persistent across all dialogues
    QUEST = auto()      # Linked to quest state

# Default namespace for variables
DEFAULT_VARIABLE_NAMESPACE: Final[str] = "dialogue"

# Maximum variable name length
MAX_VARIABLE_NAME_LENGTH: Final[int] = 64

# Maximum variable value length (for strings)
MAX_VARIABLE_VALUE_LENGTH: Final[int] = 1024

# =============================================================================
# Condition Operator Constants
# =============================================================================

class ComparisonOperator(Enum):
    """Comparison operators for conditions."""
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    CONTAINS = "in"
    NOT_CONTAINS = "not in"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex match

class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = auto()
    OR = auto()
    NOT = auto()
    XOR = auto()

# =============================================================================
# Effect Type Constants
# =============================================================================

class EffectType(Enum):
    """Types of dialogue effects."""
    SET_VARIABLE = auto()
    INCREMENT_VARIABLE = auto()
    DECREMENT_VARIABLE = auto()
    GIVE_ITEM = auto()
    TAKE_ITEM = auto()
    UPDATE_QUEST = auto()
    CHANGE_REPUTATION = auto()
    PLAY_SOUND = auto()
    PLAY_ANIMATION = auto()
    TRIGGER_EVENT = auto()
    START_DIALOGUE = auto()
    END_DIALOGUE = auto()

# =============================================================================
# Quest State Constants
# =============================================================================

class QuestState(Enum):
    """Quest progression states."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()

# =============================================================================
# Reputation Constants
# =============================================================================

# Default reputation range
MIN_REPUTATION: Final[int] = -100
MAX_REPUTATION: Final[int] = 100
DEFAULT_REPUTATION: Final[int] = 0

# Reputation threshold names
class ReputationLevel(Enum):
    """Named reputation levels."""
    HATED = auto()      # -100 to -75
    HOSTILE = auto()    # -74 to -50
    UNFRIENDLY = auto() # -49 to -25
    NEUTRAL = auto()    # -24 to 24
    FRIENDLY = auto()   # 25 to 49
    HONORED = auto()    # 50 to 74
    EXALTED = auto()    # 75 to 100

# =============================================================================
# Localization Constants
# =============================================================================

# Default language code
DEFAULT_LANGUAGE: Final[str] = "en"

# Supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES: Final[frozenset[str]] = frozenset({
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "ru",  # Russian
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
    "ar",  # Arabic
    "pl",  # Polish
})

# String table key format
STRING_KEY_SEPARATOR: Final[str] = "."
STRING_KEY_PREFIX: Final[str] = "dialogue"

# =============================================================================
# Graph Validation Constants
# =============================================================================

# Maximum nodes per dialogue graph
MAX_NODES_PER_GRAPH: Final[int] = 1000

# Maximum connections per node
MAX_CONNECTIONS_PER_NODE: Final[int] = 50

# Maximum depth for graph traversal (prevent infinite loops)
MAX_TRAVERSAL_DEPTH: Final[int] = 100

# =============================================================================
# Serialization Constants
# =============================================================================

# File extension for dialogue files
DIALOGUE_FILE_EXTENSION: Final[str] = ".dialogue"

# JSON schema version
DIALOGUE_SCHEMA_VERSION: Final[str] = "1.0.0"

# =============================================================================
# Priority Constants
# =============================================================================

# Effect execution priority (lower = executes first)
class EffectPriority(Enum):
    """Effect execution priority levels."""
    IMMEDIATE = 0
    HIGH = 10
    NORMAL = 50
    LOW = 90
    DEFERRED = 100


# =============================================================================
# Quest Objective Constants
# =============================================================================

# Default detection radius for reach objectives (units)
DEFAULT_REACH_RADIUS: Final[float] = 5.0

# Health percentage defaults
DEFAULT_HEALTH_PERCENT: Final[float] = 100.0
MIN_HEALTH_PERCENT_DEFAULT: Final[float] = 0.0

# Escort objective defaults
DEFAULT_ESCORT_DISTANCE_THRESHOLD: Final[float] = 20.0

# Defend objective defaults
DEFAULT_DEFEND_DURATION: Final[float] = 60.0
DEFAULT_TARGET_HEALTH_PERCENT: Final[float] = 100.0

# Time limit defaults
DEFAULT_TIMED_OBJECTIVE_LIMIT: Final[float] = 60.0

# =============================================================================
# Quest Journal Constants
# =============================================================================

# Maximum quests that can be tracked on HUD
DEFAULT_MAX_HUD_TRACKED_QUESTS: Final[int] = 3

# Maximum objectives displayed per quest on HUD
MAX_HUD_OBJECTIVES_DISPLAYED: Final[int] = 3
