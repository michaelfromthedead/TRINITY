"""
Matchmaking System Module.

Provides skill-based matchmaking with queue management, match finding algorithms,
and dynamic search expansion for optimal player matching in multiplayer games.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
from threading import Lock
import logging
import time
import uuid

from .config import SOCIAL_CONFIG

logger = logging.getLogger(__name__)


class MatchmakingState(Enum):
    """Current state of a player in the matchmaking system."""
    IDLE = auto()
    SEARCHING = auto()
    FOUND = auto()
    CONNECTING = auto()
    FAILED = auto()


@dataclass
class MatchCriteria:
    """Criteria for matchmaking a player."""
    mode: str
    region: str
    skill_range: tuple[float, float]
    party_size: int = 1

    def __post_init__(self) -> None:
        """Validate criteria after initialization."""
        if self.skill_range[0] > self.skill_range[1]:
            raise ValueError("skill_range min must be <= max")
        if self.party_size < 1:
            raise ValueError("party_size must be at least 1")

    def matches(self, other: 'MatchCriteria') -> bool:
        """Check if this criteria is compatible with another."""
        if self.mode != other.mode:
            return False
        if self.region != other.region:
            return False
        # Check for skill range overlap
        return (self.skill_range[0] <= other.skill_range[1] and
                self.skill_range[1] >= other.skill_range[0])


@dataclass
class QueueEntry:
    """An entry in the matchmaking queue."""
    player_id: str
    criteria: MatchCriteria
    queue_time: float
    skill: float
    party_id: Optional[str] = None
    state: MatchmakingState = MatchmakingState.SEARCHING

    @property
    def wait_time(self) -> float:
        """Get time spent waiting in queue."""
        return time.time() - self.queue_time


@dataclass
class MatchResult:
    """Result of a successful matchmaking operation."""
    players: list[str]
    server_address: str
    match_id: str
    mode: str
    region: str
    average_skill: float


class MatchmakingQueue:
    """
    Manages player queues for matchmaking.

    Supports skill-based matching with dynamic search expansion
    to reduce wait times for players who have been waiting longer.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        min_players: int = 2,
        max_players: int = 10,
        base_skill_range: float = 100.0,
        expansion_rate: float = 10.0,
        expansion_interval: float = 5.0,
        server_allocator: Optional[Callable[[str, str], str]] = None
    ) -> None:
        """
        Initialize the matchmaking queue.

        Args:
            min_players: Minimum players needed for a match.
            max_players: Maximum players in a match.
            base_skill_range: Base skill range for matching.
            expansion_rate: How much to expand skill range per interval.
            expansion_interval: Seconds between range expansions.
            server_allocator: Callback to allocate server address given mode and region.
        """
        self._entries: dict[str, QueueEntry] = {}
        self._lock = Lock()
        self.min_players = min_players
        self.max_players = max_players
        self.base_skill_range = base_skill_range
        self.expansion_rate = expansion_rate
        self.expansion_interval = expansion_interval
        self._server_allocator = server_allocator or self._default_server_allocator

        # Callbacks
        self._on_match_found: Optional[Callable[[MatchResult], None]] = None
        self._on_state_change: Optional[Callable[[str, MatchmakingState], None]] = None

    def _default_server_allocator(self, mode: str, region: str) -> str:
        """Default server address allocator."""
        port_base = SOCIAL_CONFIG.Matchmaking.DEFAULT_SERVER_PORT_BASE
        port_range = SOCIAL_CONFIG.Matchmaking.DEFAULT_SERVER_PORT_RANGE
        return f"game://{region}.server.example.com:{port_base + hash(mode) % port_range}"

    def set_on_match_found(self, callback: Callable[[MatchResult], None]) -> None:
        """Set callback for when a match is found."""
        self._on_match_found = callback

    def set_on_state_change(self, callback: Callable[[str, MatchmakingState], None]) -> None:
        """Set callback for player state changes."""
        self._on_state_change = callback

    def _notify_state_change(self, player_id: str, state: MatchmakingState) -> None:
        """Notify listeners of state change."""
        if self._on_state_change:
            self._on_state_change(player_id, state)

    def join(
        self,
        player_id: str,
        criteria: MatchCriteria,
        skill: Optional[float] = None,
        party_id: Optional[str] = None
    ) -> bool:
        """
        Add a player to the matchmaking queue.

        Args:
            player_id: Unique identifier for the player.
            criteria: Matchmaking criteria.
            skill: Player's skill rating (uses midpoint of range if not provided).
            party_id: Optional party ID for group matching.

        Returns:
            True if successfully joined, False if already in queue.
        """
        with self._lock:
            if player_id in self._entries:
                return False

            player_skill = skill if skill is not None else (
                (criteria.skill_range[0] + criteria.skill_range[1]) / 2
            )

            entry = QueueEntry(
                player_id=player_id,
                criteria=criteria,
                queue_time=time.time(),
                skill=player_skill,
                party_id=party_id,
                state=MatchmakingState.SEARCHING
            )
            self._entries[player_id] = entry
            self._notify_state_change(player_id, MatchmakingState.SEARCHING)
            return True

    def leave(self, player_id: str) -> bool:
        """
        Remove a player from the matchmaking queue.

        Args:
            player_id: Unique identifier for the player.

        Returns:
            True if successfully removed, False if not in queue.
        """
        with self._lock:
            if player_id not in self._entries:
                return False

            del self._entries[player_id]
            self._notify_state_change(player_id, MatchmakingState.IDLE)
            return True

    def get_state(self, player_id: str) -> Optional[MatchmakingState]:
        """Get the current matchmaking state for a player."""
        with self._lock:
            entry = self._entries.get(player_id)
            return entry.state if entry else None

    def get_entry(self, player_id: str) -> Optional[QueueEntry]:
        """Get the queue entry for a player."""
        with self._lock:
            return self._entries.get(player_id)

    def get_queue_size(self, mode: Optional[str] = None, region: Optional[str] = None) -> int:
        """
        Get the number of players in queue.

        Args:
            mode: Filter by game mode (optional).
            region: Filter by region (optional).

        Returns:
            Number of players matching the filters.
        """
        with self._lock:
            count = 0
            for entry in self._entries.values():
                if entry.state != MatchmakingState.SEARCHING:
                    continue
                if mode and entry.criteria.mode != mode:
                    continue
                if region and entry.criteria.region != region:
                    continue
                count += 1
            return count

    def expand_search_over_time(self) -> None:
        """
        Expand skill search ranges for players waiting in queue.

        Should be called periodically (e.g., every second).
        Expands the skill range based on wait time to find matches faster.
        """
        with self._lock:
            current_time = time.time()

            # Iterate over a snapshot of entries to avoid modification during iteration
            entries_snapshot = list(self._entries.values())

            for entry in entries_snapshot:
                # Re-check entry still exists (could have been removed)
                if entry.player_id not in self._entries:
                    continue

                if entry.state != MatchmakingState.SEARCHING:
                    continue

                wait_time = current_time - entry.queue_time
                expansions = int(wait_time / self.expansion_interval)

                if expansions > 0:
                    expansion = expansions * self.expansion_rate
                    original = entry.criteria.skill_range

                    # Expand symmetrically but don't go below 0
                    new_min = max(0, original[0] - expansion)
                    new_max = original[1] + expansion

                    entry.criteria = MatchCriteria(
                        mode=entry.criteria.mode,
                        region=entry.criteria.region,
                        skill_range=(new_min, new_max),
                        party_size=entry.criteria.party_size
                    )

    def find_match(self) -> Optional[MatchResult]:
        """
        Attempt to find a match from the current queue.

        Returns:
            MatchResult if a valid match is found, None otherwise.
        """
        with self._lock:
            # Get all searching players
            searching = [
                entry for entry in self._entries.values()
                if entry.state == MatchmakingState.SEARCHING
            ]

            if len(searching) < self.min_players:
                return None

            # Group by mode and region
            groups: dict[tuple[str, str], list[QueueEntry]] = {}
            for entry in searching:
                key = (entry.criteria.mode, entry.criteria.region)
                if key not in groups:
                    groups[key] = []
                groups[key].append(entry)

            # Try to find a match in each group
            for (mode, region), entries in groups.items():
                if len(entries) < self.min_players:
                    continue

                # Sort by skill for better matching
                entries.sort(key=lambda e: e.skill)

                # Try to form a match with compatible skill ranges
                match_players = self._find_compatible_players(entries)

                if match_players:
                    return self._create_match(match_players, mode, region)

            return None

    def _find_compatible_players(
        self,
        entries: list[QueueEntry]
    ) -> Optional[list[QueueEntry]]:
        """
        Find a group of players with compatible skill ranges.

        Uses a sliding window approach on skill-sorted players.
        """
        if len(entries) < self.min_players:
            return None

        best_match: Optional[list[QueueEntry]] = None
        best_variance = float('inf')

        # Sliding window to find best skill-matched group
        for i in range(len(entries) - self.min_players + 1):
            window_end = min(i + self.max_players, len(entries))

            for size in range(self.min_players, window_end - i + 1):
                candidates = entries[i:i + size]

                # Check if all candidates have compatible skill ranges
                if self._are_compatible(candidates):
                    # Calculate skill variance
                    skills = [e.skill for e in candidates]
                    avg = sum(skills) / len(skills)
                    variance = sum((s - avg) ** 2 for s in skills) / len(skills)

                    # Prefer larger groups with lower variance
                    score = variance - (len(candidates) * 10)

                    if score < best_variance:
                        best_variance = score
                        best_match = candidates

        return best_match

    def _are_compatible(self, entries: list[QueueEntry]) -> bool:
        """Check if all entries have overlapping skill ranges."""
        if not entries:
            return False

        # Find the intersection of all skill ranges
        min_skill = max(e.criteria.skill_range[0] for e in entries)
        max_skill = min(e.criteria.skill_range[1] for e in entries)

        return min_skill <= max_skill

    def _create_match(
        self,
        matched_entries: list[QueueEntry],
        mode: str,
        region: str
    ) -> MatchResult:
        """Create a match result and update player states."""
        match_id = str(uuid.uuid4())
        player_ids = [entry.player_id for entry in matched_entries]

        # Update states
        for entry in matched_entries:
            entry.state = MatchmakingState.FOUND
            self._notify_state_change(entry.player_id, MatchmakingState.FOUND)

        # Calculate average skill
        total_skill = sum(entry.skill for entry in matched_entries)
        average_skill = total_skill / len(matched_entries)

        # Allocate server
        server_address = self._server_allocator(mode, region)

        result = MatchResult(
            players=player_ids,
            server_address=server_address,
            match_id=match_id,
            mode=mode,
            region=region,
            average_skill=average_skill
        )

        # Notify listeners
        if self._on_match_found:
            self._on_match_found(result)

        # Remove matched players from queue
        for player_id in player_ids:
            del self._entries[player_id]

        return result

    def update_player_state(
        self,
        player_id: str,
        state: MatchmakingState
    ) -> bool:
        """
        Update a player's matchmaking state.

        Args:
            player_id: The player to update.
            state: New state.

        Returns:
            True if updated, False if player not found.
        """
        with self._lock:
            entry = self._entries.get(player_id)
            if not entry:
                return False

            entry.state = state
            self._notify_state_change(player_id, state)

            # Remove if failed or idle
            if state in (MatchmakingState.FAILED, MatchmakingState.IDLE):
                del self._entries[player_id]

            return True

    def get_estimated_wait_time(
        self,
        mode: str,
        region: str,
        skill: float
    ) -> float:
        """
        Estimate wait time for a player with given parameters.

        Returns estimated seconds until match found.
        """
        with self._lock:
            # Count players in range
            skill_range = self.base_skill_range
            compatible_count = 0

            for entry in self._entries.values():
                if entry.criteria.mode != mode or entry.criteria.region != region:
                    continue
                if abs(entry.skill - skill) <= skill_range:
                    compatible_count += 1

            # Estimate based on queue size and min players
            instant_threshold = SOCIAL_CONFIG.Matchmaking.INSTANT_MATCH_THRESHOLD_SECONDS
            seconds_per_player = SOCIAL_CONFIG.Matchmaking.ESTIMATED_SECONDS_PER_MISSING_PLAYER

            if compatible_count >= self.min_players - 1:
                return instant_threshold  # Nearly instant match

            needed = self.min_players - 1 - compatible_count
            return max(instant_threshold, needed * seconds_per_player)


class MatchmakingService:
    """
    High-level matchmaking service that manages multiple queues.

    Provides a unified interface for different game modes and regions.
    """

    def __init__(self) -> None:
        """Initialize the matchmaking service."""
        self._queues: dict[str, MatchmakingQueue] = {}
        self._player_queues: dict[str, str] = {}  # player_id -> queue_key
        self._lock = Lock()

    def get_or_create_queue(
        self,
        mode: str,
        region: str,
        **queue_kwargs: Any
    ) -> MatchmakingQueue:
        """Get or create a queue for a specific mode/region combination."""
        key = f"{mode}:{region}"

        with self._lock:
            if key not in self._queues:
                self._queues[key] = MatchmakingQueue(**queue_kwargs)
            return self._queues[key]

    def join_queue(
        self,
        player_id: str,
        criteria: MatchCriteria,
        skill: Optional[float] = None
    ) -> bool:
        """Join the appropriate queue for the given criteria."""
        queue = self.get_or_create_queue(criteria.mode, criteria.region)

        with self._lock:
            if player_id in self._player_queues:
                return False  # Already in a queue

            result = queue.join(player_id, criteria, skill)
            if result:
                self._player_queues[player_id] = f"{criteria.mode}:{criteria.region}"
            return result

    def leave_queue(self, player_id: str) -> bool:
        """Leave the current queue."""
        with self._lock:
            queue_key = self._player_queues.get(player_id)
            if not queue_key:
                return False

            queue = self._queues.get(queue_key)
            if queue:
                queue.leave(player_id)

            del self._player_queues[player_id]
            return True

    def tick(self) -> list[MatchResult]:
        """
        Process all queues and return any matches found.

        Should be called periodically (e.g., every second).
        """
        matches: list[MatchResult] = []

        with self._lock:
            for queue in self._queues.values():
                queue.expand_search_over_time()
                match = queue.find_match()
                if match:
                    matches.append(match)
                    # Clean up player tracking
                    for player_id in match.players:
                        if player_id in self._player_queues:
                            del self._player_queues[player_id]

        return matches
