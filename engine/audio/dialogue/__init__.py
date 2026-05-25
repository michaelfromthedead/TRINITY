"""
Dialogue Systems Module.

Comprehensive voice-over and dialogue management system for game audio.

Components:
- DialogueManager: Central coordination for all dialogue functionality
- VOQueue: Priority-based voice-over queue management
- VOLine: Individual VO line with metadata
- VOStreamManager: Audio streaming and caching
- VOProcessor: Audio effects and processing
- SubtitleManager: Subtitle timing and display
- LocalizationManager: Language switching and audio banks
- ConversationManager: Multi-character conversations
- BarkSystem: Short reaction barks
- AmbientVOSystem: Background ambient VO
"""

from .config import (
    # Priority levels
    VOPriority,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    PRIORITY_LOW,
    PRIORITY_BARK,
    PRIORITY_AMBIENT,
    DEFAULT_INTERRUPT_PRIORITY,
    # Queue settings
    MAX_QUEUE_SIZE,
    QUEUE_TIMEOUT_MS,
    # Cooldowns
    BARK_COOLDOWN_MS,
    SAME_LINE_COOLDOWN_MS,
    SAME_SPEAKER_COOLDOWN_MS,
    # Overlap
    MAX_SIMULTANEOUS_VO,
    OVERLAP_DUCK_DB,
    CONVERSATION_GAP_MS,
    # Streaming
    VO_PRELOAD_TIME_MS,
    VO_CACHE_SIZE_MB,
    VO_STREAM_BUFFER_MS,
    # Processing
    RADIO_BAND_LOW,
    RADIO_BAND_HIGH,
    RADIO_DISTORTION,
    DISTANCE_FILTER_START,
    DISTANCE_FILTER_MAX,
    VO_REVERB_SEND_DEFAULT,
    # Subtitles
    SUBTITLE_FADE_TIME_MS,
    SUBTITLE_MIN_DISPLAY_MS,
    SUBTITLE_CHARS_PER_SECOND,
    MAX_SUBTITLE_LINES,
    # Localization
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    # Context types
    ContextType,
    CONTEXT_BARK,
    CONTEXT_CONVERSATION,
    CONTEXT_AMBIENT,
    CONTEXT_NARRATION,
    CONTEXT_TUTORIAL,
    # Selection modes
    SelectionMode,
    SELECTION_RANDOM,
    SELECTION_SEQUENTIAL,
    SELECTION_WEIGHTED,
    SELECTION_CONDITIONAL,
    # States
    DialogueState,
)

from .vo_line import (
    VOLine,
    VOLineState,
    LipSyncData,
    SubtitleData,
    create_vo_line,
)

from .vo_queue import (
    VOQueue,
    VOQueueManager,
    QueueEntry,
)

from .contextual_dialogue import (
    CooldownTracker,
    LinePool,
    ContextualDialogueManager,
    BarkSystem,
    AmbientVOSystem,
    create_bark_lines,
)

from .conversation import (
    Conversation,
    ConversationNode,
    ConversationState,
    ConversationManager,
    create_linear_conversation,
    create_branching_conversation,
)

from .localization import (
    LocalizedAsset,
    AudioBank,
    LocalizationManager,
    create_localized_asset,
    create_audio_bank,
)

from .subtitle_sync import (
    SubtitlePosition,
    SubtitleState,
    SubtitleStyle,
    ActiveSubtitle,
    SubtitleCue,
    SubtitleTrack,
    SubtitleManager,
    create_subtitle_data,
)

from .vo_streaming import (
    StreamState,
    CachedAudio,
    StreamHandle,
    VOCache,
    VOStreamManager,
)

from .vo_processing import (
    EffectType,
    RadioEffect,
    DistanceFilter,
    ReverbSettings,
    SpatialSettings,
    VOProcessingState,
    VOProcessor,
    create_radio_preset,
    create_telephone_preset,
    create_megaphone_preset,
)

from .dialogue_manager import (
    DialogueEvent,
    DialogueManager,
)


__all__ = [
    # Config
    "VOPriority",
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
    "PRIORITY_BARK",
    "PRIORITY_AMBIENT",
    "DEFAULT_INTERRUPT_PRIORITY",
    "MAX_QUEUE_SIZE",
    "QUEUE_TIMEOUT_MS",
    "BARK_COOLDOWN_MS",
    "SAME_LINE_COOLDOWN_MS",
    "SAME_SPEAKER_COOLDOWN_MS",
    "MAX_SIMULTANEOUS_VO",
    "OVERLAP_DUCK_DB",
    "CONVERSATION_GAP_MS",
    "VO_PRELOAD_TIME_MS",
    "VO_CACHE_SIZE_MB",
    "VO_STREAM_BUFFER_MS",
    "RADIO_BAND_LOW",
    "RADIO_BAND_HIGH",
    "RADIO_DISTORTION",
    "DISTANCE_FILTER_START",
    "DISTANCE_FILTER_MAX",
    "VO_REVERB_SEND_DEFAULT",
    "SUBTITLE_FADE_TIME_MS",
    "SUBTITLE_MIN_DISPLAY_MS",
    "SUBTITLE_CHARS_PER_SECOND",
    "MAX_SUBTITLE_LINES",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "ContextType",
    "CONTEXT_BARK",
    "CONTEXT_CONVERSATION",
    "CONTEXT_AMBIENT",
    "CONTEXT_NARRATION",
    "CONTEXT_TUTORIAL",
    "SelectionMode",
    "SELECTION_RANDOM",
    "SELECTION_SEQUENTIAL",
    "SELECTION_WEIGHTED",
    "SELECTION_CONDITIONAL",
    "DialogueState",
    # VO Line
    "VOLine",
    "VOLineState",
    "LipSyncData",
    "SubtitleData",
    "create_vo_line",
    # VO Queue
    "VOQueue",
    "VOQueueManager",
    "QueueEntry",
    # Contextual Dialogue
    "CooldownTracker",
    "LinePool",
    "ContextualDialogueManager",
    "BarkSystem",
    "AmbientVOSystem",
    "create_bark_lines",
    # Conversation
    "Conversation",
    "ConversationNode",
    "ConversationState",
    "ConversationManager",
    "create_linear_conversation",
    "create_branching_conversation",
    # Localization
    "LocalizedAsset",
    "AudioBank",
    "LocalizationManager",
    "create_localized_asset",
    "create_audio_bank",
    # Subtitles
    "SubtitlePosition",
    "SubtitleState",
    "SubtitleStyle",
    "ActiveSubtitle",
    "SubtitleCue",
    "SubtitleTrack",
    "SubtitleManager",
    "create_subtitle_data",
    # Streaming
    "StreamState",
    "CachedAudio",
    "StreamHandle",
    "VOCache",
    "VOStreamManager",
    # Processing
    "EffectType",
    "RadioEffect",
    "DistanceFilter",
    "ReverbSettings",
    "SpatialSettings",
    "VOProcessingState",
    "VOProcessor",
    "create_radio_preset",
    "create_telephone_preset",
    "create_megaphone_preset",
    # Main Manager
    "DialogueEvent",
    "DialogueManager",
]
