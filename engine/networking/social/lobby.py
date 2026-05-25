"""
Game Lobby System Module.

Provides lobby management for pre-game gathering, player ready systems,
countdown handling, and game session initialization.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
from threading import Lock, Timer
import logging
import time
import uuid

from .config import SOCIAL_CONFIG

logger = logging.getLogger(__name__)


class LobbyState(Enum):
    """Current state of a game lobby."""
    WAITING = auto()     # Waiting for players to join
    COUNTDOWN = auto()   # Countdown to game start
    STARTING = auto()    # Game is being initialized
    IN_GAME = auto()     # Game is in progress
    CLOSED = auto()      # Lobby has been closed


@dataclass
class LobbySettings:
    """Configuration settings for a lobby."""
    max_players: int = SOCIAL_CONFIG.Lobby.MAX_PLAYERS_DEFAULT
    min_players: int = SOCIAL_CONFIG.Lobby.MIN_PLAYERS_DEFAULT
    game_mode: str = "default"
    map_name: str = "default"
    is_private: bool = False
    password: Optional[str] = None
    countdown_seconds: int = SOCIAL_CONFIG.Lobby.COUNTDOWN_SECONDS_DEFAULT
    auto_start: bool = True  # Auto-start when min_players ready
    allow_spectators: bool = False
    max_spectators: int = SOCIAL_CONFIG.Lobby.MAX_SPECTATORS_DEFAULT

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.max_players < self.min_players:
            raise ValueError("max_players must be >= min_players")
        if self.min_players < 1:
            raise ValueError("min_players must be at least 1")
        if self.countdown_seconds < 0:
            raise ValueError("countdown_seconds cannot be negative")


@dataclass
class LobbyPlayer:
    """A player in a lobby."""
    player_id: str
    display_name: str
    is_ready: bool = False
    is_host: bool = False
    team: Optional[str] = None
    slot: int = 0
    join_time: float = field(default_factory=time.time)
    is_spectator: bool = False


class Lobby:
    """
    Represents a game lobby where players gather before a match.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        host_id: str,
        host_name: str,
        settings: Optional[LobbySettings] = None
    ) -> None:
        """
        Initialize a new lobby.

        Args:
            host_id: The host player's ID.
            host_name: The host player's display name.
            settings: Lobby configuration settings.
        """
        self.id = str(uuid.uuid4())
        self.host_id = host_id
        self.settings = settings or LobbySettings()
        self.state = LobbyState.WAITING
        self.created_at = time.time()

        self._players: dict[str, LobbyPlayer] = {}
        self._spectators: dict[str, LobbyPlayer] = {}
        self._lock = Lock()

        # Countdown management
        self._countdown_timer: Optional[Timer] = None
        self._countdown_start: Optional[float] = None

        # Callbacks
        self._on_all_ready: Optional[Callable[['Lobby'], None]] = None
        self._on_state_change: Optional[Callable[['Lobby', LobbyState], None]] = None
        self._on_player_join: Optional[Callable[['Lobby', str], None]] = None
        self._on_player_leave: Optional[Callable[['Lobby', str], None]] = None
        self._on_countdown_complete: Optional[Callable[['Lobby'], None]] = None

        # Add host as first player
        self._add_player(host_id, host_name, is_host=True)

    def _add_player(
        self,
        player_id: str,
        display_name: str,
        is_host: bool = False,
        team: Optional[str] = None,
        is_spectator: bool = False
    ) -> bool:
        """Internal method to add a player."""
        slot = len(self._players)
        player = LobbyPlayer(
            player_id=player_id,
            display_name=display_name,
            is_host=is_host,
            team=team,
            slot=slot,
            is_spectator=is_spectator
        )

        if is_spectator:
            self._spectators[player_id] = player
        else:
            self._players[player_id] = player

        return True

    def set_on_all_ready(self, callback: Callable[['Lobby'], None]) -> None:
        """Set callback for when all players are ready."""
        self._on_all_ready = callback

    def set_on_state_change(
        self,
        callback: Callable[['Lobby', LobbyState], None]
    ) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def set_on_player_join(self, callback: Callable[['Lobby', str], None]) -> None:
        """Set callback for player joins."""
        self._on_player_join = callback

    def set_on_player_leave(self, callback: Callable[['Lobby', str], None]) -> None:
        """Set callback for player leaves."""
        self._on_player_leave = callback

    def set_on_countdown_complete(self, callback: Callable[['Lobby'], None]) -> None:
        """Set callback for countdown completion."""
        self._on_countdown_complete = callback

    def _change_state(self, new_state: LobbyState) -> None:
        """Change lobby state and notify listeners."""
        old_state = self.state
        self.state = new_state

        if self._on_state_change and old_state != new_state:
            self._on_state_change(self, new_state)

    @property
    def player_count(self) -> int:
        """Get the number of players in the lobby."""
        with self._lock:
            return len(self._players)

    @property
    def spectator_count(self) -> int:
        """Get the number of spectators in the lobby."""
        with self._lock:
            return len(self._spectators)

    @property
    def is_full(self) -> bool:
        """Check if the lobby is full."""
        with self._lock:
            return len(self._players) >= self.settings.max_players

    def _can_start_internal(self) -> bool:
        """Check if the lobby can start (caller must hold lock)."""
        if self.state != LobbyState.WAITING:
            return False

        ready_count = sum(1 for p in self._players.values() if p.is_ready)
        return ready_count >= self.settings.min_players

    @property
    def can_start(self) -> bool:
        """Check if the lobby can start (enough ready players)."""
        with self._lock:
            return self._can_start_internal()

    def _all_ready_internal(self) -> bool:
        """Check if all players are ready (caller must hold lock)."""
        if not self._players:
            return False
        return all(p.is_ready for p in self._players.values())

    @property
    def all_ready(self) -> bool:
        """Check if all players are ready."""
        with self._lock:
            return self._all_ready_internal()

    @property
    def ready_count(self) -> int:
        """Get the number of ready players."""
        with self._lock:
            return sum(1 for p in self._players.values() if p.is_ready)

    def get_players(self) -> list[LobbyPlayer]:
        """Get a list of all players in the lobby."""
        with self._lock:
            return list(self._players.values())

    def get_spectators(self) -> list[LobbyPlayer]:
        """Get a list of all spectators."""
        with self._lock:
            return list(self._spectators.values())

    def get_player(self, player_id: str) -> Optional[LobbyPlayer]:
        """Get a specific player by ID."""
        with self._lock:
            return self._players.get(player_id) or self._spectators.get(player_id)

    def join(
        self,
        player_id: str,
        display_name: str,
        password: Optional[str] = None,
        as_spectator: bool = False
    ) -> bool:
        """
        Attempt to join the lobby.

        Args:
            player_id: The joining player's ID.
            display_name: The joining player's display name.
            password: Password if lobby is private.
            as_spectator: Whether to join as a spectator.

        Returns:
            True if successfully joined, False otherwise.
        """
        with self._lock:
            # Check if lobby accepts new players
            if self.state not in (LobbyState.WAITING, LobbyState.COUNTDOWN):
                return False

            # Check password
            if self.settings.is_private and self.settings.password:
                if password != self.settings.password:
                    return False

            # Check if already in lobby
            if player_id in self._players or player_id in self._spectators:
                return False

            if as_spectator:
                if not self.settings.allow_spectators:
                    return False
                if len(self._spectators) >= self.settings.max_spectators:
                    return False
                self._add_player(player_id, display_name, is_spectator=True)
            else:
                if len(self._players) >= self.settings.max_players:
                    return False
                self._add_player(player_id, display_name)

            if self._on_player_join:
                self._on_player_join(self, player_id)

            return True

    def leave(self, player_id: str) -> bool:
        """
        Remove a player from the lobby.

        Args:
            player_id: The leaving player's ID.

        Returns:
            True if successfully left, False if not in lobby.
        """
        with self._lock:
            return self._leave_internal(player_id)

    def kick(self, player_id: str, kicked_by: str) -> bool:
        """
        Kick a player from the lobby (host only).

        Args:
            player_id: The player to kick.
            kicked_by: The player doing the kicking.

        Returns:
            True if successfully kicked, False otherwise.
        """
        with self._lock:
            # Only host can kick
            if kicked_by != self.host_id:
                return False

            # Can't kick yourself
            if player_id == kicked_by:
                return False

            # Use internal leave logic to avoid deadlock (we already hold the lock)
            return self._leave_internal(player_id)

    def _leave_internal(self, player_id: str) -> bool:
        """
        Internal leave method (caller must hold lock).

        Args:
            player_id: The leaving player's ID.

        Returns:
            True if successfully left, False if not in lobby.
        """
        if player_id in self._spectators:
            del self._spectators[player_id]
            if self._on_player_leave:
                self._on_player_leave(self, player_id)
            return True

        if player_id not in self._players:
            return False

        was_host = self._players[player_id].is_host
        del self._players[player_id]

        # Handle host leaving
        if was_host and self._players:
            # Promote first player to host
            new_host = next(iter(self._players.values()))
            new_host.is_host = True
            self.host_id = new_host.player_id

        # Cancel countdown if not enough players
        if len(self._players) < self.settings.min_players:
            if self.state == LobbyState.COUNTDOWN:
                self._cancel_countdown()

        if self._on_player_leave:
            self._on_player_leave(self, player_id)

        # Close lobby if empty
        if not self._players:
            self._change_state(LobbyState.CLOSED)

        return True

    def set_ready(self, player_id: str, ready: bool = True) -> bool:
        """
        Set a player's ready status.

        Args:
            player_id: The player's ID.
            ready: The ready status to set.

        Returns:
            True if successfully updated, False if player not found.
        """
        with self._lock:
            if player_id not in self._players:
                return False

            self._players[player_id].is_ready = ready

            # Check if all ready
            if ready and self._all_ready_internal():
                if self._on_all_ready:
                    self._on_all_ready(self)

                # Auto-start countdown if enabled
                if self.settings.auto_start and self.state == LobbyState.WAITING:
                    self._start_countdown_internal()

            # Cancel countdown if someone unreadies
            if not ready and self.state == LobbyState.COUNTDOWN:
                self._cancel_countdown()

            return True

    def set_team(self, player_id: str, team: Optional[str]) -> bool:
        """
        Set a player's team.

        Args:
            player_id: The player's ID.
            team: The team name (or None for no team).

        Returns:
            True if successfully updated, False if player not found.
        """
        with self._lock:
            if player_id not in self._players:
                return False

            self._players[player_id].team = team
            return True

    def start_countdown(self, requested_by: Optional[str] = None) -> bool:
        """
        Start the game countdown.

        Args:
            requested_by: The player requesting start (must be host if specified).

        Returns:
            True if countdown started, False otherwise.
        """
        with self._lock:
            if requested_by and requested_by != self.host_id:
                return False

            if not self._can_start_internal():
                return False

            return self._start_countdown_internal()

    def _start_countdown_internal(self) -> bool:
        """Internal method to start countdown (caller must hold lock)."""
        if self.state != LobbyState.WAITING:
            return False

        self._change_state(LobbyState.COUNTDOWN)
        self._countdown_start = time.time()

        # Create countdown timer
        self._countdown_timer = Timer(
            self.settings.countdown_seconds,
            self._on_countdown_finished
        )
        self._countdown_timer.daemon = True
        self._countdown_timer.start()

        return True

    def _cancel_countdown(self) -> None:
        """Cancel the current countdown (caller must hold lock)."""
        if self._countdown_timer:
            self._countdown_timer.cancel()
            self._countdown_timer = None

        self._countdown_start = None
        self._change_state(LobbyState.WAITING)

    def cancel_countdown(self, requested_by: Optional[str] = None) -> bool:
        """
        Cancel the current countdown.

        Args:
            requested_by: The player requesting cancel (must be host if specified).

        Returns:
            True if cancelled, False otherwise.
        """
        with self._lock:
            if requested_by and requested_by != self.host_id:
                return False

            if self.state != LobbyState.COUNTDOWN:
                return False

            self._cancel_countdown()
            return True

    def get_countdown_remaining(self) -> Optional[float]:
        """Get seconds remaining in countdown, or None if not counting down."""
        with self._lock:
            if self.state != LobbyState.COUNTDOWN or not self._countdown_start:
                return None

            elapsed = time.time() - self._countdown_start
            remaining = self.settings.countdown_seconds - elapsed
            return max(0.0, remaining)

    def _on_countdown_finished(self) -> None:
        """Called when countdown completes."""
        with self._lock:
            if self.state != LobbyState.COUNTDOWN:
                return

            self._countdown_timer = None
            self._countdown_start = None
            self._change_state(LobbyState.STARTING)

            if self._on_countdown_complete:
                self._on_countdown_complete(self)

    def update_settings(
        self,
        settings: LobbySettings,
        updated_by: str
    ) -> bool:
        """
        Update lobby settings (host only).

        Args:
            settings: New settings.
            updated_by: Player making the update.

        Returns:
            True if updated, False otherwise.
        """
        with self._lock:
            if updated_by != self.host_id:
                return False

            if self.state != LobbyState.WAITING:
                return False

            # Validate new settings against current state
            if settings.max_players < len(self._players):
                return False

            self.settings = settings
            return True

    def transfer_host(self, new_host_id: str, transferred_by: str) -> bool:
        """
        Transfer host to another player.

        Args:
            new_host_id: The new host's player ID.
            transferred_by: The current host making the transfer.

        Returns:
            True if transferred, False otherwise.
        """
        with self._lock:
            if transferred_by != self.host_id:
                return False

            if new_host_id not in self._players:
                return False

            # Update host status
            self._players[transferred_by].is_host = False
            self._players[new_host_id].is_host = True
            self.host_id = new_host_id

            return True

    def close(self, closed_by: Optional[str] = None) -> bool:
        """
        Close the lobby.

        Args:
            closed_by: The player closing (must be host if specified).

        Returns:
            True if closed, False otherwise.
        """
        with self._lock:
            if closed_by and closed_by != self.host_id:
                return False

            if self.state == LobbyState.CLOSED:
                return False

            self._cancel_countdown()
            self._change_state(LobbyState.CLOSED)
            self._players.clear()
            self._spectators.clear()

            return True

    def to_dict(self) -> dict[str, Any]:
        """Convert lobby to dictionary for serialization."""
        with self._lock:
            return {
                "id": self.id,
                "host_id": self.host_id,
                "state": self.state.name,
                "player_count": len(self._players),
                "max_players": self.settings.max_players,
                "game_mode": self.settings.game_mode,
                "map_name": self.settings.map_name,
                "is_private": self.settings.is_private,
                "players": [
                    {
                        "id": p.player_id,
                        "name": p.display_name,
                        "is_ready": p.is_ready,
                        "is_host": p.is_host,
                        "team": p.team
                    }
                    for p in self._players.values()
                ],
                "created_at": self.created_at,
                "countdown_remaining": self.get_countdown_remaining()
            }


class LobbyManager:
    """
    Manages multiple game lobbies.

    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the lobby manager."""
        self._lobbies: dict[str, Lobby] = {}
        self._player_lobbies: dict[str, str] = {}  # player_id -> lobby_id
        self._lock = Lock()

        # Callbacks
        self._on_lobby_created: Optional[Callable[[Lobby], None]] = None
        self._on_lobby_closed: Optional[Callable[[str], None]] = None

    def set_on_lobby_created(self, callback: Callable[[Lobby], None]) -> None:
        """Set callback for lobby creation."""
        self._on_lobby_created = callback

    def set_on_lobby_closed(self, callback: Callable[[str], None]) -> None:
        """Set callback for lobby closure."""
        self._on_lobby_closed = callback

    def create_lobby(
        self,
        host_id: str,
        host_name: str,
        settings: Optional[LobbySettings] = None
    ) -> Optional[Lobby]:
        """
        Create a new lobby.

        Args:
            host_id: The host player's ID.
            host_name: The host player's display name.
            settings: Lobby configuration.

        Returns:
            The created Lobby, or None if player already in a lobby.
        """
        with self._lock:
            # Check if player is already in a lobby
            if host_id in self._player_lobbies:
                return None

            lobby = Lobby(host_id, host_name, settings)

            # Set up internal callbacks
            original_leave = lobby.leave

            def wrapped_leave(player_id: str) -> bool:
                result = original_leave(player_id)
                if result:
                    with self._lock:
                        if player_id in self._player_lobbies:
                            del self._player_lobbies[player_id]
                        if lobby.state == LobbyState.CLOSED:
                            if lobby.id in self._lobbies:
                                del self._lobbies[lobby.id]
                            if self._on_lobby_closed:
                                self._on_lobby_closed(lobby.id)
                return result

            lobby.leave = wrapped_leave  # type: ignore

            self._lobbies[lobby.id] = lobby
            self._player_lobbies[host_id] = lobby.id

            if self._on_lobby_created:
                self._on_lobby_created(lobby)

            return lobby

    def get_lobby(self, lobby_id: str) -> Optional[Lobby]:
        """Get a lobby by ID."""
        with self._lock:
            return self._lobbies.get(lobby_id)

    def get_player_lobby(self, player_id: str) -> Optional[Lobby]:
        """Get the lobby a player is currently in."""
        with self._lock:
            lobby_id = self._player_lobbies.get(player_id)
            if lobby_id:
                return self._lobbies.get(lobby_id)
            return None

    def join_lobby(
        self,
        lobby_id: str,
        player_id: str,
        player_name: str,
        password: Optional[str] = None,
        as_spectator: bool = False
    ) -> bool:
        """
        Join an existing lobby.

        Args:
            lobby_id: The lobby to join.
            player_id: The joining player's ID.
            player_name: The joining player's display name.
            password: Password if lobby is private.
            as_spectator: Whether to join as spectator.

        Returns:
            True if successfully joined, False otherwise.
        """
        with self._lock:
            # Check if player is already in a lobby
            if player_id in self._player_lobbies:
                return False

            lobby = self._lobbies.get(lobby_id)
            if not lobby:
                return False

            if lobby.join(player_id, player_name, password, as_spectator):
                self._player_lobbies[player_id] = lobby_id
                return True

            return False

    def leave_lobby(self, player_id: str) -> bool:
        """
        Leave the current lobby.

        Args:
            player_id: The leaving player's ID.

        Returns:
            True if successfully left, False otherwise.
        """
        with self._lock:
            lobby_id = self._player_lobbies.get(player_id)
            if not lobby_id:
                return False

            lobby = self._lobbies.get(lobby_id)
            if not lobby:
                del self._player_lobbies[player_id]
                return False

            return lobby.leave(player_id)

    def find_lobbies(
        self,
        game_mode: Optional[str] = None,
        map_name: Optional[str] = None,
        include_full: bool = False,
        include_private: bool = False,
        max_results: Optional[int] = None
    ) -> list[Lobby]:
        """
        Find lobbies matching the given filters.

        Args:
            game_mode: Filter by game mode.
            map_name: Filter by map name.
            include_full: Include full lobbies.
            include_private: Include private lobbies.
            max_results: Maximum number of results.

        Returns:
            List of matching lobbies.
        """
        if max_results is None:
            max_results = SOCIAL_CONFIG.Lobby.FIND_LOBBIES_MAX_RESULTS

        with self._lock:
            results: list[Lobby] = []

            for lobby in self._lobbies.values():
                # Only show waiting/countdown lobbies
                if lobby.state not in (LobbyState.WAITING, LobbyState.COUNTDOWN):
                    continue

                # Apply filters
                if game_mode and lobby.settings.game_mode != game_mode:
                    continue

                if map_name and lobby.settings.map_name != map_name:
                    continue

                if not include_full and lobby.is_full:
                    continue

                if not include_private and lobby.settings.is_private:
                    continue

                results.append(lobby)

                if len(results) >= max_results:
                    break

            # Sort by player count (more players = more likely to start)
            results.sort(key=lambda l: l.player_count, reverse=True)

            return results

    def get_lobby_count(self) -> int:
        """Get the total number of active lobbies."""
        with self._lock:
            return len(self._lobbies)

    def cleanup_closed_lobbies(self) -> int:
        """
        Remove closed lobbies from the manager.

        Returns:
            Number of lobbies removed.
        """
        with self._lock:
            closed_ids = [
                lid for lid, lobby in self._lobbies.items()
                if lobby.state == LobbyState.CLOSED
            ]

            for lid in closed_ids:
                del self._lobbies[lid]
                if self._on_lobby_closed:
                    self._on_lobby_closed(lid)

            return len(closed_ids)

    def get_stats(self) -> dict[str, Any]:
        """Get lobby manager statistics."""
        with self._lock:
            total_players = len(self._player_lobbies)
            total_lobbies = len(self._lobbies)

            state_counts = {state.name: 0 for state in LobbyState}
            for lobby in self._lobbies.values():
                state_counts[lobby.state.name] += 1

            return {
                "total_lobbies": total_lobbies,
                "total_players": total_players,
                "state_counts": state_counts
            }
