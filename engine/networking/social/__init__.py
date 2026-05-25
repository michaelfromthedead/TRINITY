"""
Social Systems Module for AI Game Engine.

Provides comprehensive multiplayer social features including matchmaking,
skill rating systems, lobbies, parties, voice chat, and text chat.

Example usage:
    from engine.networking.social import (
        MatchmakingQueue, MatchCriteria, MatchmakingState,
        MMRManager, SkillRating, EloCalculator,
        LobbyManager, LobbySettings, LobbyState,
        PartyManager, PartyRole,
        VoiceChatManager, VoiceChannel,
        ChatManager, ChatChannel, ProfanityFilter
    )

    # Matchmaking
    queue = MatchmakingQueue(min_players=2, max_players=10)
    criteria = MatchCriteria(mode="ranked", region="us-west", skill_range=(1400, 1600))
    queue.join("player1", criteria, skill=1500)

    # Skill Rating
    mmr = MMRManager(use_glicko=True)
    rating = mmr.get_rating("player1")

    # Lobbies
    lobby_mgr = LobbyManager()
    lobby = lobby_mgr.create_lobby("host_player", "HostName")

    # Parties
    party_mgr = PartyManager()
    party = party_mgr.create_party("leader_id", "LeaderName")

    # Voice Chat
    voice = VoiceChatManager()
    voice.join_channel("player1", "Player1", VoiceChannel.TEAM)

    # Text Chat
    chat = ChatManager()
    chat.send_message("player1", "Player1", ChatChannel.GLOBAL, "Hello!")
"""

# Matchmaking
from .matchmaking import (
    MatchmakingState,
    MatchCriteria,
    QueueEntry,
    MatchResult,
    MatchmakingQueue,
    MatchmakingService,
)

# Skill Rating
from .skill_rating import (
    SkillRating,
    MatchOutcome,
    EloCalculator,
    Glicko2Calculator,
    MMRManager,
)

# Lobby
from .lobby import (
    LobbyState,
    LobbySettings,
    LobbyPlayer,
    Lobby,
    LobbyManager,
)

# Party
from .party import (
    PartyRole,
    PartyState,
    PartyMember,
    PartyInvite,
    Party,
    PartyManager,
)

# Voice Chat
from .voice_chat import (
    VoiceChannel,
    VoiceQuality,
    VoiceState,
    VoiceParticipant,
    VoiceChannelInfo,
    ProximityVoice,
    VoiceChatManager,
)

# Text Chat
from .text_chat import (
    ChatChannel,
    MessageType,
    ChatMessage,
    ProfanityFilter,
    RateLimiter,
    ChatManager,
)

# Configuration
from .config import (
    SOCIAL_CONFIG,
    SocialConfig,
    MatchmakingConfig,
    SkillRatingConfig,
    LobbyConfig,
    PartyConfig,
    VoiceChatConfig,
    TextChatConfig,
)


__all__ = [
    # Matchmaking
    "MatchmakingState",
    "MatchCriteria",
    "QueueEntry",
    "MatchResult",
    "MatchmakingQueue",
    "MatchmakingService",
    # Skill Rating
    "SkillRating",
    "MatchOutcome",
    "EloCalculator",
    "Glicko2Calculator",
    "MMRManager",
    # Lobby
    "LobbyState",
    "LobbySettings",
    "LobbyPlayer",
    "Lobby",
    "LobbyManager",
    # Party
    "PartyRole",
    "PartyState",
    "PartyMember",
    "PartyInvite",
    "Party",
    "PartyManager",
    # Voice Chat
    "VoiceChannel",
    "VoiceQuality",
    "VoiceState",
    "VoiceParticipant",
    "VoiceChannelInfo",
    "ProximityVoice",
    "VoiceChatManager",
    # Text Chat
    "ChatChannel",
    "MessageType",
    "ChatMessage",
    "ProfanityFilter",
    "RateLimiter",
    "ChatManager",
    # Configuration
    "SOCIAL_CONFIG",
    "SocialConfig",
    "MatchmakingConfig",
    "SkillRatingConfig",
    "LobbyConfig",
    "PartyConfig",
    "VoiceChatConfig",
    "TextChatConfig",
]
