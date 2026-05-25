"""
Text Chat System Module.

Provides text-based communication including channel management,
profanity filtering, rate limiting, and chat history.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
from threading import Lock
from collections import deque
import logging
import time
import re
import uuid

from .config import SOCIAL_CONFIG

logger = logging.getLogger(__name__)


class ChatChannel(Enum):
    """Text chat channel types."""
    GLOBAL = auto()     # Everyone in the server
    TEAM = auto()       # Team-only chat
    PARTY = auto()      # Party/squad chat
    WHISPER = auto()    # Direct player-to-player
    SYSTEM = auto()     # System messages
    LOBBY = auto()      # Lobby chat
    MATCH = auto()      # In-match all chat


class MessageType(Enum):
    """Type of chat message."""
    TEXT = auto()       # Normal text message
    EMOTE = auto()      # Emote/action (/me)
    SYSTEM = auto()     # System announcement
    COMMAND = auto()    # Command response
    ALERT = auto()      # Important alert


@dataclass
class ChatMessage:
    """A chat message."""
    id: str
    sender_id: str
    sender_name: str
    channel: ChatChannel
    content: str
    timestamp: float
    target_id: Optional[str] = None  # For whisper
    target_name: Optional[str] = None
    message_type: MessageType = MessageType.TEXT
    is_filtered: bool = False  # Was profanity filtered
    original_content: Optional[str] = None  # Original before filtering
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        sender_id: str,
        sender_name: str,
        channel: ChatChannel,
        content: str,
        **kwargs: Any
    ) -> 'ChatMessage':
        """Create a new chat message with auto-generated ID and timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            sender_name=sender_name,
            channel=channel,
            content=content,
            timestamp=time.time(),
            **kwargs
        )


class ProfanityFilter:
    """
    Filters profanity and inappropriate content from messages.

    Supports word blocking, pattern matching, and bypass detection.
    """

    def __init__(self) -> None:
        """Initialize the profanity filter with default words."""
        self._blocked_words: set[str] = set()
        self._blocked_patterns: list[re.Pattern[str]] = []
        self._whitelist: set[str] = set()
        self._replacement_char = SOCIAL_CONFIG.TextChat.PROFANITY_REPLACEMENT_CHAR
        self._enabled = True
        self._lock = Lock()

        # Add some common blocked words (kept minimal for example)
        # In production, load from a comprehensive word list
        # self._blocked_words starts as empty set - add words via add_blocked_word()

    @property
    def enabled(self) -> bool:
        """Check if filter is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable the filter."""
        self._enabled = value

    @property
    def blocked_words(self) -> set[str]:
        """Get the set of blocked words."""
        with self._lock:
            return self._blocked_words.copy()

    def add_blocked_word(self, word: str) -> None:
        """Add a word to the block list."""
        with self._lock:
            self._blocked_words.add(word.lower())

    def remove_blocked_word(self, word: str) -> None:
        """Remove a word from the block list."""
        with self._lock:
            self._blocked_words.discard(word.lower())

    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a regex pattern to block."""
        with self._lock:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._blocked_patterns.append(compiled)

    def add_whitelist_word(self, word: str) -> None:
        """Add a word to the whitelist (won't be filtered)."""
        with self._lock:
            self._whitelist.add(word.lower())

    def set_replacement_char(self, char: str) -> None:
        """Set the character used to replace profanity."""
        self._replacement_char = char[0] if char else "*"

    def contains_profanity(self, text: str) -> bool:
        """
        Check if text contains profanity.

        Args:
            text: The text to check.

        Returns:
            True if profanity is detected, False otherwise.
        """
        if not self._enabled:
            return False

        with self._lock:
            text_lower = text.lower()

            # Check blocked words
            words = re.findall(r'\b\w+\b', text_lower)
            for word in words:
                if word in self._whitelist:
                    continue
                if word in self._blocked_words:
                    return True

            # Check patterns
            for pattern in self._blocked_patterns:
                if pattern.search(text):
                    return True

            # Check for obfuscation attempts (l33t speak, etc.)
            normalized = self._normalize_text(text_lower)
            for word in self._blocked_words:
                if word in normalized:
                    return True

            return False

    def filter(self, text: str) -> str:
        """
        Filter profanity from text.

        Args:
            text: The text to filter.

        Returns:
            Filtered text with profanity replaced.
        """
        if not self._enabled:
            return text

        with self._lock:
            result = text

            # Filter blocked words
            for word in self._blocked_words:
                pattern = re.compile(
                    r'\b' + re.escape(word) + r'\b',
                    re.IGNORECASE
                )
                replacement = self._replacement_char * len(word)
                result = pattern.sub(replacement, result)

            # Filter patterns
            for pattern in self._blocked_patterns:
                def replacer(match: re.Match[str]) -> str:
                    return self._replacement_char * len(match.group())
                result = pattern.sub(replacer, result)

            # Handle obfuscation
            result = self._filter_obfuscated(result)

            return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text to detect obfuscation attempts."""
        # Common l33t speak substitutions
        substitutions = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a',
            '5': 's', '7': 't', '@': 'a', '$': 's',
            '!': 'i', '+': 't'
        }

        result = text.lower()
        for old, new in substitutions.items():
            result = result.replace(old, new)

        # Remove repeated characters
        result = re.sub(r'(.)\1+', r'\1', result)

        # Remove spaces and special chars
        result = re.sub(r'[^a-z]', '', result)

        return result

    def _filter_obfuscated(self, text: str) -> str:
        """Filter obfuscated profanity."""
        # Check normalized version
        normalized = self._normalize_text(text)

        for word in self._blocked_words:
            if word in normalized:
                # Find and replace the obfuscated version
                # This is a simplified approach
                pattern = self._build_obfuscation_pattern(word)
                replacement = self._replacement_char * len(word)
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _build_obfuscation_pattern(self, word: str) -> str:
        """Build a regex pattern that matches obfuscated versions of a word."""
        # Map each character to possible obfuscations
        char_patterns: dict[str, str] = {
            'a': '[a@4]',
            'e': '[e3]',
            'i': '[i1!]',
            'o': '[o0]',
            's': '[s5$]',
            't': '[t7+]',
        }

        pattern_parts = []
        for char in word.lower():
            if char in char_patterns:
                pattern_parts.append(char_patterns[char])
            else:
                pattern_parts.append(re.escape(char))

            # Allow optional separators between characters
            pattern_parts.append(r'[\s\-_.]*')

        return r'\b' + ''.join(pattern_parts[:-1]) + r'\b'


class RateLimiter:
    """
    Rate limiter for chat messages.

    Prevents spam by limiting message frequency per player.
    """

    def __init__(
        self,
        messages_per_second: Optional[float] = None,
        burst_limit: Optional[int] = None,
        cooldown_seconds: Optional[float] = None
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            messages_per_second: Maximum sustained message rate.
            burst_limit: Maximum messages in a burst.
            cooldown_seconds: Cooldown after hitting burst limit.
        """
        self.messages_per_second = messages_per_second if messages_per_second is not None else SOCIAL_CONFIG.TextChat.RATE_MESSAGES_PER_SECOND
        self.burst_limit = burst_limit if burst_limit is not None else SOCIAL_CONFIG.TextChat.RATE_BURST_LIMIT
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else SOCIAL_CONFIG.TextChat.RATE_COOLDOWN_SECONDS

        self._player_state: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def _get_player_state(self, player_id: str) -> dict[str, Any]:
        """Get or create rate limit state for a player."""
        if player_id not in self._player_state:
            self._player_state[player_id] = {
                'tokens': self.burst_limit,
                'last_update': time.time(),
                'cooldown_until': 0.0,
                'message_count': 0
            }
        return self._player_state[player_id]

    def can_send(self, player_id: str) -> bool:
        """
        Check if a player can send a message.

        Args:
            player_id: The player ID.

        Returns:
            True if allowed to send, False if rate limited.
        """
        with self._lock:
            state = self._get_player_state(player_id)
            current_time = time.time()

            # Check cooldown
            if current_time < state['cooldown_until']:
                return False

            # Refill tokens
            time_passed = current_time - state['last_update']
            tokens_to_add = time_passed * self.messages_per_second
            state['tokens'] = min(self.burst_limit, state['tokens'] + tokens_to_add)
            state['last_update'] = current_time

            return state['tokens'] >= 1.0

    def consume(self, player_id: str) -> bool:
        """
        Consume a rate limit token for sending.

        Args:
            player_id: The player ID.

        Returns:
            True if consumed successfully, False if rate limited.
        """
        with self._lock:
            state = self._get_player_state(player_id)
            current_time = time.time()

            # Check cooldown
            if current_time < state['cooldown_until']:
                return False

            # Refill tokens
            time_passed = current_time - state['last_update']
            tokens_to_add = time_passed * self.messages_per_second
            state['tokens'] = min(self.burst_limit, state['tokens'] + tokens_to_add)
            state['last_update'] = current_time

            if state['tokens'] >= 1.0:
                state['tokens'] -= 1.0
                state['message_count'] += 1

                # Apply cooldown if burst depleted
                if state['tokens'] < 1.0:
                    state['cooldown_until'] = current_time + self.cooldown_seconds

                return True

            return False

    def get_wait_time(self, player_id: str) -> float:
        """
        Get time until player can send another message.

        Args:
            player_id: The player ID.

        Returns:
            Seconds until allowed to send (0 if allowed now).
        """
        with self._lock:
            state = self._get_player_state(player_id)
            current_time = time.time()

            # Check cooldown
            if current_time < state['cooldown_until']:
                return state['cooldown_until'] - current_time

            # Check token availability
            time_passed = current_time - state['last_update']
            tokens_to_add = time_passed * self.messages_per_second
            current_tokens = min(self.burst_limit, state['tokens'] + tokens_to_add)

            if current_tokens >= 1.0:
                return 0.0

            # Calculate time until 1 token
            tokens_needed = 1.0 - current_tokens
            return tokens_needed / self.messages_per_second

    def reset(self, player_id: str) -> None:
        """Reset rate limit state for a player."""
        with self._lock:
            if player_id in self._player_state:
                del self._player_state[player_id]

    def get_stats(self, player_id: str) -> dict[str, Any]:
        """Get rate limit stats for a player."""
        with self._lock:
            state = self._get_player_state(player_id)
            current_time = time.time()

            return {
                'tokens': state['tokens'],
                'in_cooldown': current_time < state['cooldown_until'],
                'cooldown_remaining': max(0.0, state['cooldown_until'] - current_time),
                'message_count': state['message_count']
            }


class ChatManager:
    """
    Manages text chat for all players.

    Handles message sending, filtering, rate limiting, and history.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        history_limit: Optional[int] = None,
        enable_profanity_filter: bool = True,
        enable_rate_limiting: bool = True
    ) -> None:
        """
        Initialize the chat manager.

        Args:
            history_limit: Maximum messages to keep in history per channel.
            enable_profanity_filter: Whether to filter profanity.
            enable_rate_limiting: Whether to enforce rate limits.
        """
        self._history_limit = history_limit if history_limit is not None else SOCIAL_CONFIG.TextChat.HISTORY_LIMIT_DEFAULT

        # Chat history per channel (channel_id -> deque of messages)
        self._history: dict[str, deque[ChatMessage]] = {}

        # Muted players (player_id -> mute expiry time, or None for permanent)
        self._muted_players: dict[str, Optional[float]] = {}

        # Per-player mutes (player_id -> set of muted player_ids)
        self._player_mutes: dict[str, set[str]] = {}

        # Components
        self._profanity_filter = ProfanityFilter()
        self._profanity_filter.enabled = enable_profanity_filter

        self._rate_limiter = RateLimiter()
        self._rate_limit_enabled = enable_rate_limiting

        self._lock = Lock()

        # Callbacks
        self._on_message: Optional[Callable[[ChatMessage], None]] = None
        self._on_message_blocked: Optional[Callable[[str, str], None]] = None

    def set_on_message(self, callback: Callable[[ChatMessage], None]) -> None:
        """Set callback for new messages."""
        self._on_message = callback

    def set_on_message_blocked(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for blocked messages (player_id, reason)."""
        self._on_message_blocked = callback

    @property
    def profanity_filter(self) -> ProfanityFilter:
        """Get the profanity filter."""
        return self._profanity_filter

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get the rate limiter."""
        return self._rate_limiter

    def _get_channel_id(
        self,
        channel: ChatChannel,
        context_id: Optional[str] = None
    ) -> str:
        """Generate a unique channel ID."""
        if channel in (ChatChannel.GLOBAL, ChatChannel.SYSTEM):
            return channel.name
        elif context_id:
            return f"{channel.name}:{context_id}"
        else:
            return channel.name

    def _get_history(self, channel_id: str) -> deque[ChatMessage]:
        """Get or create history deque for a channel."""
        if channel_id not in self._history:
            self._history[channel_id] = deque(maxlen=self._history_limit)
        return self._history[channel_id]

    def _is_muted_internal(self, player_id: str) -> bool:
        """Check if a player is server-muted (caller must hold lock)."""
        if player_id not in self._muted_players:
            return False

        expiry = self._muted_players[player_id]
        if expiry is None:
            return True  # Permanent mute

        if time.time() < expiry:
            return True

        # Mute expired
        del self._muted_players[player_id]
        return False

    def is_muted(self, player_id: str) -> bool:
        """Check if a player is server-muted."""
        with self._lock:
            return self._is_muted_internal(player_id)

    def send_message(
        self,
        sender_id: str,
        sender_name: str,
        channel: ChatChannel,
        content: str,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        context_id: Optional[str] = None,
        message_type: MessageType = MessageType.TEXT
    ) -> Optional[ChatMessage]:
        """
        Send a chat message.

        Args:
            sender_id: The sender's player ID.
            sender_name: The sender's display name.
            channel: The channel to send to.
            content: The message content.
            target_id: Target player ID (for whisper).
            target_name: Target player name (for whisper).
            context_id: Context ID (team, lobby, etc.).
            message_type: Type of message.

        Returns:
            The created ChatMessage, or None if blocked.
        """
        with self._lock:
            # Check mute status
            if self._is_muted_internal(sender_id):
                if self._on_message_blocked:
                    self._on_message_blocked(sender_id, "muted")
                return None

            # Check rate limit
            if self._rate_limit_enabled and not self._rate_limiter.consume(sender_id):
                if self._on_message_blocked:
                    self._on_message_blocked(sender_id, "rate_limited")
                return None

            # Validate content
            if not content or not content.strip():
                return None

            max_length = SOCIAL_CONFIG.TextChat.MESSAGE_MAX_LENGTH
            if len(content) > max_length:
                content = content[:max_length]

            # Filter profanity
            original_content = None
            is_filtered = False

            if self._profanity_filter.enabled:
                if self._profanity_filter.contains_profanity(content):
                    original_content = content
                    content = self._profanity_filter.filter(content)
                    is_filtered = True

            # Create message
            message = ChatMessage.create(
                sender_id=sender_id,
                sender_name=sender_name,
                channel=channel,
                content=content.strip(),
                target_id=target_id,
                target_name=target_name,
                message_type=message_type,
                is_filtered=is_filtered,
                original_content=original_content
            )

            # Add to history
            channel_id = self._get_channel_id(channel, context_id)
            history = self._get_history(channel_id)
            history.append(message)

            # Notify listeners
            if self._on_message:
                self._on_message(message)

            return message

    def send_system_message(
        self,
        content: str,
        channel: ChatChannel = ChatChannel.SYSTEM,
        context_id: Optional[str] = None
    ) -> ChatMessage:
        """
        Send a system message.

        Args:
            content: The message content.
            channel: The channel to send to.
            context_id: Context ID for targeted channels.

        Returns:
            The created ChatMessage.
        """
        with self._lock:
            message = ChatMessage.create(
                sender_id="SYSTEM",
                sender_name="System",
                channel=channel,
                content=content,
                message_type=MessageType.SYSTEM
            )

            channel_id = self._get_channel_id(channel, context_id)
            history = self._get_history(channel_id)
            history.append(message)

            if self._on_message:
                self._on_message(message)

            return message

    def get_history(
        self,
        channel: ChatChannel,
        context_id: Optional[str] = None,
        limit: int = 50,
        before_id: Optional[str] = None
    ) -> list[ChatMessage]:
        """
        Get chat history for a channel.

        Args:
            channel: The channel.
            context_id: Context ID for targeted channels.
            limit: Maximum messages to return.
            before_id: Get messages before this message ID.

        Returns:
            List of ChatMessages, newest first.
        """
        with self._lock:
            channel_id = self._get_channel_id(channel, context_id)
            history = self._get_history(channel_id)

            if not history:
                return []

            messages = list(history)

            # Filter by before_id
            if before_id:
                try:
                    idx = next(
                        i for i, m in enumerate(messages)
                        if m.id == before_id
                    )
                    messages = messages[:idx]
                except StopIteration:
                    pass

            # Return newest first, limited
            return list(reversed(messages[-limit:]))

    def mute_player(
        self,
        player_id: str,
        muted_by: str,
        duration_seconds: Optional[float] = None
    ) -> bool:
        """
        Server-mute a player.

        Args:
            player_id: The player to mute.
            muted_by: Who is muting (for logging).
            duration_seconds: Mute duration (None = permanent).

        Returns:
            True if muted successfully.
        """
        with self._lock:
            expiry = None
            if duration_seconds is not None:
                expiry = time.time() + duration_seconds

            self._muted_players[player_id] = expiry
            return True

    def unmute_player(self, player_id: str, unmuted_by: str) -> bool:
        """
        Remove server mute from a player.

        Args:
            player_id: The player to unmute.
            unmuted_by: Who is unmuting (for logging).

        Returns:
            True if unmuted, False if wasn't muted.
        """
        with self._lock:
            if player_id in self._muted_players:
                del self._muted_players[player_id]
                return True
            return False

    def personal_mute(self, player_id: str, target_id: str) -> bool:
        """
        Mute another player for yourself only.

        Args:
            player_id: The player doing the muting.
            target_id: The player being muted.

        Returns:
            True if muted successfully.
        """
        with self._lock:
            if player_id not in self._player_mutes:
                self._player_mutes[player_id] = set()
            self._player_mutes[player_id].add(target_id)
            return True

    def personal_unmute(self, player_id: str, target_id: str) -> bool:
        """
        Unmute another player for yourself.

        Args:
            player_id: The player doing the unmuting.
            target_id: The player being unmuted.

        Returns:
            True if unmuted, False if wasn't muted.
        """
        with self._lock:
            if player_id not in self._player_mutes:
                return False
            if target_id in self._player_mutes[player_id]:
                self._player_mutes[player_id].discard(target_id)
                return True
            return False

    def is_personally_muted(self, player_id: str, target_id: str) -> bool:
        """Check if a player has personally muted another."""
        with self._lock:
            mutes = self._player_mutes.get(player_id, set())
            return target_id in mutes

    def get_visible_messages(
        self,
        player_id: str,
        channel: ChatChannel,
        context_id: Optional[str] = None,
        limit: int = 50
    ) -> list[ChatMessage]:
        """
        Get messages visible to a specific player.

        Filters out messages from personally muted players.

        Args:
            player_id: The viewing player.
            channel: The channel.
            context_id: Context ID for targeted channels.
            limit: Maximum messages to return.

        Returns:
            List of visible ChatMessages.
        """
        with self._lock:
            messages = self.get_history(channel, context_id, limit * 2)

            # Filter personally muted
            mutes = self._player_mutes.get(player_id, set())

            visible = [
                m for m in messages
                if m.sender_id not in mutes and m.sender_id != player_id
                   or m.sender_id == player_id
            ]

            return visible[:limit]

    def clear_history(
        self,
        channel: ChatChannel,
        context_id: Optional[str] = None
    ) -> int:
        """
        Clear chat history for a channel.

        Args:
            channel: The channel.
            context_id: Context ID for targeted channels.

        Returns:
            Number of messages cleared.
        """
        with self._lock:
            channel_id = self._get_channel_id(channel, context_id)
            if channel_id in self._history:
                count = len(self._history[channel_id])
                self._history[channel_id].clear()
                return count
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get chat manager statistics."""
        with self._lock:
            total_messages = sum(len(h) for h in self._history.values())
            muted_count = len([
                p for p, exp in self._muted_players.items()
                if exp is None or time.time() < exp
            ])

            return {
                "total_messages_in_history": total_messages,
                "channels_with_history": len(self._history),
                "server_muted_players": muted_count,
                "profanity_filter_enabled": self._profanity_filter.enabled,
                "rate_limiting_enabled": self._rate_limit_enabled
            }
