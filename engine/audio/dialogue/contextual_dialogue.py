"""
Contextual Dialogue Module.

Handles barks, ambient VO, conditional lines, and context-based dialogue selection.
Supports line pools, cooldown tracking, weighted selection, and game state conditions.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from .config import (
    AMBIENT_MAX_INTERVAL_MS,
    AMBIENT_MIN_INTERVAL_MS,
    BARK_COOLDOWN_MS,
    CONTEXT_AMBIENT,
    CONTEXT_BARK,
    CONTEXT_COMBAT,
    CONTEXT_EXPLORATION,
    CONTEXT_NARRATION,
    CONTEXT_TUTORIAL,
    ContextType,
    PRIORITY_AMBIENT,
    PRIORITY_BARK,
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    SAME_LINE_COOLDOWN_MS,
    SAME_SPEAKER_COOLDOWN_MS,
    SelectionMode,
)
from .vo_line import VOLine, create_vo_line


@dataclass
class CooldownTracker:
    """Tracks cooldowns for lines, speakers, and categories."""

    _line_cooldowns: dict[str, float] = field(default_factory=dict)
    _speaker_cooldowns: dict[str, float] = field(default_factory=dict)
    _category_cooldowns: dict[str, float] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def record_play(
        self,
        line_id: str,
        speaker_id: str,
        category: str,
        current_time: float,
    ) -> None:
        """Record that a line was played."""
        with self._lock:
            self._line_cooldowns[line_id] = current_time
            if speaker_id:
                self._speaker_cooldowns[speaker_id] = current_time
            if category:
                self._category_cooldowns[category] = current_time

    def is_line_on_cooldown(
        self,
        line_id: str,
        current_time: float,
        cooldown_ms: float = SAME_LINE_COOLDOWN_MS,
    ) -> bool:
        """Check if a specific line is on cooldown."""
        with self._lock:
            last_played = self._line_cooldowns.get(line_id, 0.0)
            elapsed = (current_time - last_played) * 1000
            return elapsed < cooldown_ms

    def is_speaker_on_cooldown(
        self,
        speaker_id: str,
        current_time: float,
        cooldown_ms: float = SAME_SPEAKER_COOLDOWN_MS,
    ) -> bool:
        """Check if a speaker is on cooldown."""
        with self._lock:
            last_played = self._speaker_cooldowns.get(speaker_id, 0.0)
            elapsed = (current_time - last_played) * 1000
            return elapsed < cooldown_ms

    def is_category_on_cooldown(
        self,
        category: str,
        current_time: float,
        cooldown_ms: float,
    ) -> bool:
        """Check if a category is on cooldown."""
        with self._lock:
            last_played = self._category_cooldowns.get(category, 0.0)
            elapsed = (current_time - last_played) * 1000
            return elapsed < cooldown_ms

    def clear_cooldowns(self) -> None:
        """Clear all cooldown tracking."""
        with self._lock:
            self._line_cooldowns.clear()
            self._speaker_cooldowns.clear()
            self._category_cooldowns.clear()

    def clear_speaker_cooldown(self, speaker_id: str) -> None:
        """Clear cooldown for a specific speaker."""
        with self._lock:
            self._speaker_cooldowns.pop(speaker_id, None)

    def get_cooldown_remaining(
        self,
        line_id: str,
        current_time: float,
        cooldown_ms: float = SAME_LINE_COOLDOWN_MS,
    ) -> float:
        """Get remaining cooldown time in milliseconds."""
        with self._lock:
            last_played = self._line_cooldowns.get(line_id, 0.0)
            elapsed = (current_time - last_played) * 1000
            remaining = cooldown_ms - elapsed
            return max(0.0, remaining)


@dataclass
class LinePool:
    """
    A pool of lines for selection.

    Supports various selection modes: random, sequential, weighted, shuffle.
    """

    pool_id: str
    lines: list[VOLine] = field(default_factory=list)
    selection_mode: str = SelectionMode.RANDOM.value
    cooldown_ms: float = SAME_LINE_COOLDOWN_MS
    category: str = ""

    # Runtime state
    _sequential_index: int = field(default=0, init=False)
    _shuffle_order: list[int] = field(default_factory=list, init=False)
    _shuffle_index: int = field(default=0, init=False)

    def add_line(self, line: VOLine) -> None:
        """Add a line to the pool."""
        self.lines.append(line)
        self._reset_shuffle()

    def remove_line(self, line_id: str) -> bool:
        """Remove a line from the pool by ID."""
        for i, line in enumerate(self.lines):
            if line.line_id == line_id:
                self.lines.pop(i)
                self._reset_shuffle()
                return True
        return False

    def select_line(
        self,
        current_time: float,
        cooldown_tracker: Optional[CooldownTracker] = None,
        game_state: Optional[dict[str, Any]] = None,
    ) -> Optional[VOLine]:
        """
        Select a line from the pool based on selection mode.

        Args:
            current_time: Current game time
            cooldown_tracker: Optional cooldown tracker
            game_state: Game state for conditional selection

        Returns:
            Selected line or None if no valid lines
        """
        available = self._get_available_lines(
            current_time, cooldown_tracker, game_state
        )

        if not available:
            return None

        if self.selection_mode == SelectionMode.RANDOM.value:
            return random.choice(available)

        elif self.selection_mode == SelectionMode.SEQUENTIAL.value:
            # Find next available in sequence
            for _ in range(len(self.lines)):
                line = self.lines[self._sequential_index]
                self._sequential_index = (self._sequential_index + 1) % len(self.lines)
                if line in available:
                    return line
            return available[0] if available else None

        elif self.selection_mode == SelectionMode.WEIGHTED.value:
            weights = [line.weight for line in available]
            total = sum(weights)
            if total <= 0:
                return random.choice(available)
            r = random.random() * total
            cumulative = 0.0
            for line in available:
                cumulative += line.weight
                if r <= cumulative:
                    return line
            return available[-1]

        elif self.selection_mode == SelectionMode.SHUFFLE.value:
            # Ensure shuffle order is valid
            if not self._shuffle_order or len(self._shuffle_order) != len(self.lines):
                self._reset_shuffle()

            # Find next available in shuffle order
            for _ in range(len(self.lines)):
                idx = self._shuffle_order[self._shuffle_index]
                self._shuffle_index = (self._shuffle_index + 1) % len(self.lines)

                # Reshuffle at end
                if self._shuffle_index == 0:
                    self._reset_shuffle()

                line = self.lines[idx]
                if line in available:
                    return line
            return available[0] if available else None

        elif self.selection_mode == SelectionMode.CONDITIONAL.value:
            # All lines already filtered by conditions
            return available[0] if available else None

        return random.choice(available)

    def _get_available_lines(
        self,
        current_time: float,
        cooldown_tracker: Optional[CooldownTracker],
        game_state: Optional[dict[str, Any]],
    ) -> list[VOLine]:
        """Get lines that are available for selection."""
        available = []

        for line in self.lines:
            # Check cooldown
            if cooldown_tracker:
                if cooldown_tracker.is_line_on_cooldown(
                    line.line_id, current_time, self.cooldown_ms
                ):
                    continue

            # Check conditions
            if game_state and line.conditions:
                if not line.matches_conditions(game_state):
                    continue

            available.append(line)

        return available

    def _reset_shuffle(self) -> None:
        """Reset shuffle order."""
        self._shuffle_order = list(range(len(self.lines)))
        random.shuffle(self._shuffle_order)
        self._shuffle_index = 0

    def reset_sequential(self) -> None:
        """Reset sequential index."""
        self._sequential_index = 0

    @property
    def size(self) -> int:
        """Get pool size."""
        return len(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[VOLine]:
        return iter(self.lines)


class ContextualDialogueManager:
    """
    Manages contextual dialogue including barks, ambient VO, and conditional lines.
    """

    def __init__(
        self,
        on_line_selected: Optional[Callable[[VOLine, str], None]] = None,
    ) -> None:
        """
        Initialize the contextual dialogue manager.

        Args:
            on_line_selected: Callback when a line is selected (line, pool_id)
        """
        self._pools: dict[str, LinePool] = {}
        self._cooldown_tracker = CooldownTracker()
        self._lock = threading.RLock()
        self._on_line_selected = on_line_selected
        self._current_game_state: dict[str, Any] = {}

    def create_pool(
        self,
        pool_id: str,
        selection_mode: str = SelectionMode.RANDOM.value,
        cooldown_ms: float = SAME_LINE_COOLDOWN_MS,
        category: str = "",
    ) -> LinePool:
        """Create a new line pool."""
        with self._lock:
            if pool_id in self._pools:
                raise ValueError(f"Pool '{pool_id}' already exists")

            pool = LinePool(
                pool_id=pool_id,
                selection_mode=selection_mode,
                cooldown_ms=cooldown_ms,
                category=category,
            )
            self._pools[pool_id] = pool
            return pool

    def get_pool(self, pool_id: str) -> Optional[LinePool]:
        """Get a pool by ID."""
        with self._lock:
            return self._pools.get(pool_id)

    def get_or_create_pool(
        self,
        pool_id: str,
        selection_mode: str = SelectionMode.RANDOM.value,
        cooldown_ms: float = SAME_LINE_COOLDOWN_MS,
        category: str = "",
    ) -> LinePool:
        """Get existing pool or create new one."""
        with self._lock:
            if pool_id not in self._pools:
                return self.create_pool(pool_id, selection_mode, cooldown_ms, category)
            return self._pools[pool_id]

    def add_line_to_pool(self, pool_id: str, line: VOLine) -> bool:
        """Add a line to an existing pool."""
        with self._lock:
            pool = self._pools.get(pool_id)
            if pool:
                pool.add_line(line)
                return True
            return False

    def remove_pool(self, pool_id: str) -> bool:
        """Remove a pool."""
        with self._lock:
            if pool_id in self._pools:
                del self._pools[pool_id]
                return True
            return False

    def select_from_pool(
        self,
        pool_id: str,
        current_time: Optional[float] = None,
    ) -> Optional[VOLine]:
        """Select a line from a specific pool."""
        if current_time is None:
            current_time = time.time()

        with self._lock:
            pool = self._pools.get(pool_id)
            if not pool:
                return None

            line = pool.select_line(
                current_time,
                self._cooldown_tracker,
                self._current_game_state,
            )

            if line and self._on_line_selected:
                self._on_line_selected(line, pool_id)

            return line

    def record_play(
        self,
        line: VOLine,
        pool_id: str,
        current_time: Optional[float] = None,
    ) -> None:
        """Record that a line was played (for cooldown tracking)."""
        if current_time is None:
            current_time = time.time()

        pool = self._pools.get(pool_id)
        category = pool.category if pool else ""

        self._cooldown_tracker.record_play(
            line.line_id,
            line.speaker_id,
            category,
            current_time,
        )

    def update_game_state(self, state: dict[str, Any]) -> None:
        """Update the current game state for conditional selection."""
        with self._lock:
            self._current_game_state.update(state)

    def set_game_state(self, state: dict[str, Any]) -> None:
        """Replace the current game state."""
        with self._lock:
            self._current_game_state = dict(state)

    def clear_game_state(self) -> None:
        """Clear the game state."""
        with self._lock:
            self._current_game_state.clear()

    def clear_cooldowns(self) -> None:
        """Clear all cooldown tracking."""
        self._cooldown_tracker.clear_cooldowns()

    def is_line_available(
        self,
        pool_id: str,
        line_id: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """Check if a specific line is available for selection."""
        if current_time is None:
            current_time = time.time()

        with self._lock:
            pool = self._pools.get(pool_id)
            if not pool:
                return False

            for line in pool.lines:
                if line.line_id == line_id:
                    # Check cooldown
                    if self._cooldown_tracker.is_line_on_cooldown(
                        line.line_id, current_time, pool.cooldown_ms
                    ):
                        return False

                    # Check conditions
                    if line.conditions:
                        if not line.matches_conditions(self._current_game_state):
                            return False

                    return True

            return False

    @property
    def pool_ids(self) -> list[str]:
        """Get list of pool IDs."""
        with self._lock:
            return list(self._pools.keys())

    @property
    def cooldown_tracker(self) -> CooldownTracker:
        """Get the cooldown tracker."""
        return self._cooldown_tracker


# =============================================================================
# Bark System
# =============================================================================


class BarkSystem:
    """
    System for managing short reaction barks.

    Barks are context-sensitive short voice lines like "Reloading!",
    "Enemy down!", "Taking fire!", etc.
    """

    def __init__(
        self,
        cooldown_ms: float = BARK_COOLDOWN_MS,
        on_bark_triggered: Optional[Callable[[VOLine, str], None]] = None,
    ) -> None:
        """
        Initialize the bark system.

        Args:
            cooldown_ms: Default cooldown between barks
            on_bark_triggered: Callback when a bark is triggered
        """
        self._manager = ContextualDialogueManager(on_line_selected=on_bark_triggered)
        self._default_cooldown = cooldown_ms
        self._enabled = True

    def register_bark_pool(
        self,
        bark_type: str,
        lines: list[VOLine],
        selection_mode: str = SelectionMode.RANDOM.value,
        cooldown_ms: Optional[float] = None,
    ) -> LinePool:
        """
        Register a pool of barks for a specific type.

        Args:
            bark_type: Type of bark (e.g., "reload", "enemy_spotted")
            lines: List of VO lines for this bark type
            selection_mode: How to select lines from pool
            cooldown_ms: Override cooldown for this bark type
        """
        pool = self._manager.create_pool(
            pool_id=bark_type,
            selection_mode=selection_mode,
            cooldown_ms=cooldown_ms or self._default_cooldown,
            category=CONTEXT_BARK,
        )

        for line in lines:
            line.context_type = CONTEXT_BARK
            if line.priority == PRIORITY_NORMAL:
                line.priority = PRIORITY_BARK
            pool.add_line(line)

        return pool

    def trigger_bark(
        self,
        bark_type: str,
        speaker_id: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> Optional[VOLine]:
        """
        Trigger a bark of the specified type.

        Args:
            bark_type: Type of bark to trigger
            speaker_id: Optionally filter by speaker
            current_time: Current game time

        Returns:
            The selected bark line, or None if unavailable
        """
        if not self._enabled:
            return None

        if current_time is None:
            current_time = time.time()

        # Check speaker cooldown
        if speaker_id and self._manager.cooldown_tracker.is_speaker_on_cooldown(
            speaker_id, current_time, SAME_SPEAKER_COOLDOWN_MS
        ):
            return None

        line = self._manager.select_from_pool(bark_type, current_time)

        if line:
            # Filter by speaker if specified
            if speaker_id and line.speaker_id != speaker_id:
                # Try to find a line from the right speaker
                pool = self._manager.get_pool(bark_type)
                if pool:
                    for pool_line in pool.lines:
                        if pool_line.speaker_id == speaker_id:
                            if self._manager.is_line_available(
                                bark_type, pool_line.line_id, current_time
                            ):
                                line = pool_line
                                break

            self._manager.record_play(line, bark_type, current_time)

        return line

    def enable(self) -> None:
        """Enable the bark system."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the bark system."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if bark system is enabled."""
        return self._enabled

    @property
    def bark_types(self) -> list[str]:
        """Get list of registered bark types."""
        return self._manager.pool_ids


# =============================================================================
# Ambient VO System
# =============================================================================


class AmbientVOSystem:
    """
    System for managing ambient/background voice-over.

    Ambient VO includes background chatter, environmental dialogue,
    and atmospheric voice content.
    """

    def __init__(
        self,
        min_interval_ms: float = AMBIENT_MIN_INTERVAL_MS,
        max_interval_ms: float = AMBIENT_MAX_INTERVAL_MS,
        on_ambient_triggered: Optional[Callable[[VOLine], None]] = None,
    ) -> None:
        """
        Initialize the ambient VO system.

        Args:
            min_interval_ms: Minimum time between ambient lines
            max_interval_ms: Maximum time between ambient lines
            on_ambient_triggered: Callback when ambient VO triggers
        """
        self._manager = ContextualDialogueManager()
        self._min_interval = min_interval_ms
        self._max_interval = max_interval_ms
        self._on_ambient_triggered = on_ambient_triggered
        self._enabled = True
        self._last_play_time = 0.0
        self._next_interval = self._calculate_interval()
        self._active_zones: set[str] = set()

    def _calculate_interval(self) -> float:
        """Calculate next random interval."""
        return random.uniform(self._min_interval, self._max_interval)

    def register_zone(
        self,
        zone_id: str,
        lines: list[VOLine],
        selection_mode: str = SelectionMode.SHUFFLE.value,
    ) -> LinePool:
        """
        Register ambient VO for a zone.

        Args:
            zone_id: Unique zone identifier
            lines: Lines for this zone
            selection_mode: Selection mode for the pool
        """
        pool = self._manager.create_pool(
            pool_id=zone_id,
            selection_mode=selection_mode,
            cooldown_ms=SAME_LINE_COOLDOWN_MS,
            category=CONTEXT_AMBIENT,
        )

        for line in lines:
            line.context_type = CONTEXT_AMBIENT
            line.priority = PRIORITY_AMBIENT
            pool.add_line(line)

        return pool

    def enter_zone(self, zone_id: str) -> None:
        """Notify that player entered a zone."""
        self._active_zones.add(zone_id)

    def exit_zone(self, zone_id: str) -> None:
        """Notify that player exited a zone."""
        self._active_zones.discard(zone_id)

    def update(self, current_time: float) -> Optional[VOLine]:
        """
        Update ambient system and potentially trigger a line.

        Args:
            current_time: Current game time

        Returns:
            Triggered line or None
        """
        if not self._enabled or not self._active_zones:
            return None

        elapsed = (current_time - self._last_play_time) * 1000

        if elapsed >= self._next_interval:
            # Select random active zone
            zone_id = random.choice(list(self._active_zones))
            line = self._manager.select_from_pool(zone_id, current_time)

            if line:
                self._last_play_time = current_time
                self._next_interval = self._calculate_interval()
                self._manager.record_play(line, zone_id, current_time)

                if self._on_ambient_triggered:
                    self._on_ambient_triggered(line)

                return line

        return None

    def force_trigger(self, zone_id: Optional[str] = None) -> Optional[VOLine]:
        """Force trigger an ambient line immediately."""
        current_time = time.time()

        if zone_id:
            target_zone = zone_id
        elif self._active_zones:
            target_zone = random.choice(list(self._active_zones))
        else:
            return None

        line = self._manager.select_from_pool(target_zone, current_time)

        if line:
            self._last_play_time = current_time
            self._next_interval = self._calculate_interval()

            if self._on_ambient_triggered:
                self._on_ambient_triggered(line)

        return line

    def enable(self) -> None:
        """Enable ambient VO."""
        self._enabled = True

    def disable(self) -> None:
        """Disable ambient VO."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if ambient VO is enabled."""
        return self._enabled

    @property
    def active_zones(self) -> set[str]:
        """Get currently active zones."""
        return set(self._active_zones)


# =============================================================================
# Helper Functions
# =============================================================================


def create_bark_lines(
    bark_data: list[dict[str, Any]],
    speaker_id: str = "",
) -> list[VOLine]:
    """
    Create bark VO lines from data.

    Args:
        bark_data: List of dicts with audio_asset, text, and optional fields
        speaker_id: Default speaker ID

    Returns:
        List of VOLine objects
    """
    lines = []
    for data in bark_data:
        line = create_vo_line(
            audio_asset=data.get("audio_asset", ""),
            text=data.get("text", ""),
            speaker_id=data.get("speaker_id", speaker_id),
            duration_ms=data.get("duration_ms", 0.0),
            priority=data.get("priority", PRIORITY_BARK),
            interruptible=data.get("interruptible", True),
            context_type=CONTEXT_BARK,
            tags=set(data.get("tags", [])),
            weight=data.get("weight", 1.0),
        )
        lines.append(line)
    return lines
