"""
Social Systems Configuration Module.

Centralizes all magic numbers and configuration constants for the social
systems including matchmaking, skill rating, lobbies, parties, voice chat,
and text chat.

Usage:
    from engine.networking.social.config import SocialConfig

    # Access configuration values
    max_party_size = SocialConfig.Party.MAX_SIZE
    default_rating = SocialConfig.SkillRating.DEFAULT_RATING
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MatchmakingConfig:
    """Configuration for matchmaking system."""

    # Player limits
    MIN_PLAYERS_DEFAULT: Final[int] = 2
    MAX_PLAYERS_DEFAULT: Final[int] = 10

    # Skill range settings
    BASE_SKILL_RANGE: Final[float] = 100.0
    EXPANSION_RATE: Final[float] = 10.0
    EXPANSION_INTERVAL_SECONDS: Final[float] = 5.0

    # Wait time estimation
    INSTANT_MATCH_THRESHOLD_SECONDS: Final[float] = 5.0
    ESTIMATED_SECONDS_PER_MISSING_PLAYER: Final[float] = 10.0

    # Server allocation
    DEFAULT_SERVER_PORT_BASE: Final[int] = 5000
    DEFAULT_SERVER_PORT_RANGE: Final[int] = 1000


@dataclass(frozen=True)
class SkillRatingConfig:
    """Configuration for skill rating systems."""

    # Default rating values
    DEFAULT_RATING: Final[float] = 1500.0
    DEFAULT_UNCERTAINTY: Final[float] = 350.0
    MIN_RATING: Final[float] = 100.0
    MIN_UNCERTAINTY: Final[float] = 30.0

    # Elo-specific settings
    ELO_K_FACTOR_DEFAULT: Final[int] = 32
    ELO_K_FACTOR_NEW_PLAYER: Final[int] = 40
    ELO_K_FACTOR_HIGH_RATED: Final[int] = 16
    ELO_HIGH_RATING_THRESHOLD: Final[float] = 2400.0
    ELO_PROVISIONAL_GAMES: Final[int] = 30
    ELO_SCALE_DIVISOR: Final[float] = 400.0

    # Glicko-2 specific settings
    GLICKO2_TAU: Final[float] = 0.5
    GLICKO2_EPSILON: Final[float] = 0.000001
    GLICKO2_MAX_ITERATIONS: Final[int] = 100
    GLICKO2_SCALE_FACTOR: Final[float] = 173.7178
    GLICKO2_DEFAULT_VOLATILITY: Final[float] = 0.06
    GLICKO2_INACTIVITY_DECAY_DAYS: Final[float] = 30.0

    # Leaderboard settings
    LEADERBOARD_DEFAULT_LIMIT: Final[int] = 100
    LEADERBOARD_MIN_GAMES: Final[int] = 10

    # Decay settings
    MAX_INACTIVE_DAYS: Final[float] = 90.0
    SECONDS_PER_DAY: Final[int] = 86400


@dataclass(frozen=True)
class LobbyConfig:
    """Configuration for lobby system."""

    # Player limits
    MAX_PLAYERS_DEFAULT: Final[int] = 8
    MIN_PLAYERS_DEFAULT: Final[int] = 2
    MAX_SPECTATORS_DEFAULT: Final[int] = 10

    # Countdown settings
    COUNTDOWN_SECONDS_DEFAULT: Final[int] = 10

    # Search settings
    FIND_LOBBIES_MAX_RESULTS: Final[int] = 50


@dataclass(frozen=True)
class PartyConfig:
    """Configuration for party system."""

    # Size limits
    MAX_SIZE_DEFAULT: Final[int] = 4
    MAX_SIZE_ABSOLUTE: Final[int] = 10
    MIN_SIZE: Final[int] = 1

    # Invite settings
    INVITE_EXPIRE_SECONDS_DEFAULT: Final[float] = 60.0


@dataclass(frozen=True)
class VoiceChatConfig:
    """Configuration for voice chat system."""

    # Proximity voice settings
    PROXIMITY_MAX_DISTANCE: Final[float] = 50.0
    PROXIMITY_MIN_DISTANCE: Final[float] = 1.0
    PROXIMITY_FALLOFF_EXPONENT: Final[float] = 2.0

    # Volume settings
    VOLUME_MIN: Final[float] = 0.0
    VOLUME_MAX: Final[float] = 2.0
    VOLUME_DEFAULT: Final[float] = 1.0
    INPUT_VOLUME_DEFAULT: Final[float] = 1.0

    # Voice activation
    NOISE_GATE_DEFAULT: Final[float] = 0.02
    NOISE_GATE_MIN: Final[float] = 0.0
    NOISE_GATE_MAX: Final[float] = 1.0

    # Speaker tracking
    SPEAKER_TIMEOUT_SECONDS: Final[float] = 0.5

    # Channel limits
    CHANNEL_MAX_PARTICIPANTS: Final[int] = 100


@dataclass(frozen=True)
class TextChatConfig:
    """Configuration for text chat system."""

    # Rate limiting
    RATE_MESSAGES_PER_SECOND: Final[float] = 2.0
    RATE_BURST_LIMIT: Final[int] = 5
    RATE_COOLDOWN_SECONDS: Final[float] = 5.0

    # Message limits
    MESSAGE_MAX_LENGTH: Final[int] = 1000
    HISTORY_LIMIT_DEFAULT: Final[int] = 100

    # Profanity filter
    PROFANITY_REPLACEMENT_CHAR: Final[str] = "*"


@dataclass(frozen=True)
class SocialConfig:
    """
    Central configuration for all social systems.

    Usage:
        from engine.networking.social.config import SocialConfig

        # Access nested configs
        max_party = SocialConfig.Party.MAX_SIZE_DEFAULT
        default_rating = SocialConfig.SkillRating.DEFAULT_RATING
    """

    Matchmaking: Final = MatchmakingConfig()
    SkillRating: Final = SkillRatingConfig()
    Lobby: Final = LobbyConfig()
    Party: Final = PartyConfig()
    VoiceChat: Final = VoiceChatConfig()
    TextChat: Final = TextChatConfig()


# Create singleton instance for easy access
SOCIAL_CONFIG = SocialConfig()
