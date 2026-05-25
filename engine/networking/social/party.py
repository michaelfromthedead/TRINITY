"""
Party System Module.

Provides group management for players who want to play together,
including party creation, invitations, leadership, and group queueing.
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


class PartyRole(Enum):
    """Role of a member within a party."""
    LEADER = auto()
    MEMBER = auto()


class PartyState(Enum):
    """Current state of a party."""
    IDLE = auto()           # Not doing anything
    SEARCHING = auto()      # In matchmaking queue
    IN_LOBBY = auto()       # In a game lobby
    IN_GAME = auto()        # Playing a game
    DISBANDED = auto()      # Party has been disbanded


@dataclass
class PartyMember:
    """A member of a party."""
    player_id: str
    display_name: str
    role: PartyRole
    is_ready: bool = False
    join_time: float = field(default_factory=time.time)


@dataclass
class PartyInvite:
    """An invitation to join a party."""
    id: str
    party_id: str
    from_player: str
    from_player_name: str
    to_player: str
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if the invite has expired."""
        return time.time() > self.expires_at


class Party:
    """
    Represents a group of players who want to play together.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        leader_id: str,
        leader_name: str,
        max_size: Optional[int] = None
    ) -> None:
        """
        Initialize a new party.

        Args:
            leader_id: The party leader's player ID.
            leader_name: The party leader's display name.
            max_size: Maximum party size (1-10).
        """
        if max_size is None:
            max_size = SOCIAL_CONFIG.Party.MAX_SIZE_DEFAULT

        if max_size < SOCIAL_CONFIG.Party.MIN_SIZE or max_size > SOCIAL_CONFIG.Party.MAX_SIZE_ABSOLUTE:
            raise ValueError(
                f"max_size must be between {SOCIAL_CONFIG.Party.MIN_SIZE} and "
                f"{SOCIAL_CONFIG.Party.MAX_SIZE_ABSOLUTE}"
            )

        self.id = str(uuid.uuid4())
        self.leader_id = leader_id
        self.max_size = max_size
        self.state = PartyState.IDLE
        self.created_at = time.time()

        self._members: dict[str, PartyMember] = {}
        self._pending_invites: dict[str, PartyInvite] = {}  # invite_id -> invite
        self._lock = Lock()

        # Callbacks
        self._on_member_join: Optional[Callable[['Party', str], None]] = None
        self._on_member_leave: Optional[Callable[['Party', str], None]] = None
        self._on_leader_change: Optional[Callable[['Party', str], None]] = None
        self._on_all_ready: Optional[Callable[['Party'], None]] = None
        self._on_state_change: Optional[Callable[['Party', PartyState], None]] = None

        # Add leader as first member
        self._add_member(leader_id, leader_name, PartyRole.LEADER)

    def _add_member(
        self,
        player_id: str,
        display_name: str,
        role: PartyRole
    ) -> bool:
        """Internal method to add a member."""
        member = PartyMember(
            player_id=player_id,
            display_name=display_name,
            role=role
        )
        self._members[player_id] = member
        return True

    def set_on_member_join(self, callback: Callable[['Party', str], None]) -> None:
        """Set callback for member joins."""
        self._on_member_join = callback

    def set_on_member_leave(self, callback: Callable[['Party', str], None]) -> None:
        """Set callback for member leaves."""
        self._on_member_leave = callback

    def set_on_leader_change(self, callback: Callable[['Party', str], None]) -> None:
        """Set callback for leader changes."""
        self._on_leader_change = callback

    def set_on_all_ready(self, callback: Callable[['Party'], None]) -> None:
        """Set callback for when all members are ready."""
        self._on_all_ready = callback

    def set_on_state_change(
        self,
        callback: Callable[['Party', PartyState], None]
    ) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def _change_state(self, new_state: PartyState) -> None:
        """Change party state and notify listeners."""
        old_state = self.state
        self.state = new_state

        if self._on_state_change and old_state != new_state:
            self._on_state_change(self, new_state)

    @property
    def size(self) -> int:
        """Get the current number of members."""
        with self._lock:
            return len(self._members)

    @property
    def is_full(self) -> bool:
        """Check if the party is full."""
        with self._lock:
            return len(self._members) >= self.max_size

    def _all_ready_internal(self) -> bool:
        """Check if all members are ready (caller must hold lock)."""
        if not self._members:
            return False
        return all(m.is_ready for m in self._members.values())

    @property
    def all_ready(self) -> bool:
        """Check if all members are ready."""
        with self._lock:
            return self._all_ready_internal()

    @property
    def ready_count(self) -> int:
        """Get the number of ready members."""
        with self._lock:
            return sum(1 for m in self._members.values() if m.is_ready)

    def get_members(self) -> list[PartyMember]:
        """Get a list of all party members."""
        with self._lock:
            return list(self._members.values())

    def get_member(self, player_id: str) -> Optional[PartyMember]:
        """Get a specific member by ID."""
        with self._lock:
            return self._members.get(player_id)

    def get_member_ids(self) -> list[str]:
        """Get all member player IDs."""
        with self._lock:
            return list(self._members.keys())

    def is_member(self, player_id: str) -> bool:
        """Check if a player is a member of this party."""
        with self._lock:
            return player_id in self._members

    def is_leader(self, player_id: str) -> bool:
        """Check if a player is the party leader."""
        with self._lock:
            return player_id == self.leader_id

    def invite(
        self,
        to_player_id: str,
        invited_by: str,
        inviter_name: str = "Unknown",
        expire_seconds: Optional[float] = None
    ) -> Optional[PartyInvite]:
        """
        Create an invitation to join the party.

        Args:
            to_player_id: The player to invite.
            invited_by: The player sending the invite.
            inviter_name: Display name of the inviter.
            expire_seconds: Seconds until invite expires.

        Returns:
            The PartyInvite if created, None if conditions not met.
        """
        if expire_seconds is None:
            expire_seconds = SOCIAL_CONFIG.Party.INVITE_EXPIRE_SECONDS_DEFAULT

        with self._lock:
            # Only members can invite
            if invited_by not in self._members:
                return None

            # Can't invite someone already in party
            if to_player_id in self._members:
                return None

            # Can't invite if full
            if len(self._members) >= self.max_size:
                return None

            # Check if already has pending invite for this player
            for inv in self._pending_invites.values():
                if inv.to_player == to_player_id and not inv.is_expired:
                    return inv  # Return existing invite

            invite = PartyInvite(
                id=str(uuid.uuid4()),
                party_id=self.id,
                from_player=invited_by,
                from_player_name=inviter_name,
                to_player=to_player_id,
                expires_at=time.time() + expire_seconds
            )
            self._pending_invites[invite.id] = invite
            return invite

    def cancel_invite(self, invite_id: str, cancelled_by: str) -> bool:
        """
        Cancel a pending invite.

        Args:
            invite_id: The invite to cancel.
            cancelled_by: The player cancelling (must be inviter or leader).

        Returns:
            True if cancelled, False otherwise.
        """
        with self._lock:
            invite = self._pending_invites.get(invite_id)
            if not invite:
                return False

            # Only inviter or leader can cancel
            if cancelled_by not in (invite.from_player, self.leader_id):
                return False

            del self._pending_invites[invite_id]
            return True

    def get_invite(self, invite_id: str) -> Optional[PartyInvite]:
        """Get an invite by ID."""
        with self._lock:
            return self._pending_invites.get(invite_id)

    def get_pending_invites(self) -> list[PartyInvite]:
        """Get all non-expired pending invites."""
        with self._lock:
            return [inv for inv in self._pending_invites.values() if not inv.is_expired]

    def accept_invite(
        self,
        invite_id: str,
        player_id: str,
        player_name: str
    ) -> bool:
        """
        Accept an invitation and join the party.

        Args:
            invite_id: The invite to accept.
            player_id: The accepting player's ID.
            player_name: The accepting player's display name.

        Returns:
            True if successfully joined, False otherwise.
        """
        with self._lock:
            invite = self._pending_invites.get(invite_id)

            if not invite:
                return False

            if invite.is_expired:
                del self._pending_invites[invite_id]
                return False

            if invite.to_player != player_id:
                return False

            if len(self._members) >= self.max_size:
                return False

            # Remove invite and add member
            del self._pending_invites[invite_id]
            self._add_member(player_id, player_name, PartyRole.MEMBER)

            if self._on_member_join:
                self._on_member_join(self, player_id)

            return True

    def decline_invite(self, invite_id: str, player_id: str) -> bool:
        """
        Decline an invitation.

        Args:
            invite_id: The invite to decline.
            player_id: The declining player's ID.

        Returns:
            True if declined, False otherwise.
        """
        with self._lock:
            invite = self._pending_invites.get(invite_id)

            if not invite:
                return False

            if invite.to_player != player_id:
                return False

            del self._pending_invites[invite_id]
            return True

    def kick(self, player_id: str, kicked_by: str) -> bool:
        """
        Kick a member from the party.

        Args:
            player_id: The player to kick.
            kicked_by: The player doing the kicking (must be leader).

        Returns:
            True if kicked, False otherwise.
        """
        with self._lock:
            # Only leader can kick
            if kicked_by != self.leader_id:
                return False

            # Can't kick yourself (use leave instead)
            if player_id == kicked_by:
                return False

            if player_id not in self._members:
                return False

            del self._members[player_id]

            if self._on_member_leave:
                self._on_member_leave(self, player_id)

            return True

    def leave(self, player_id: str) -> bool:
        """
        Leave the party.

        Args:
            player_id: The leaving player's ID.

        Returns:
            True if successfully left, False if not in party.
        """
        with self._lock:
            if player_id not in self._members:
                return False

            was_leader = player_id == self.leader_id
            del self._members[player_id]

            if self._on_member_leave:
                self._on_member_leave(self, player_id)

            # Handle leader leaving
            if was_leader and self._members:
                # Promote longest-standing member
                oldest_member = min(
                    self._members.values(),
                    key=lambda m: m.join_time
                )
                oldest_member.role = PartyRole.LEADER
                self.leader_id = oldest_member.player_id

                if self._on_leader_change:
                    self._on_leader_change(self, self.leader_id)

            # Disband if empty
            if not self._members:
                self._change_state(PartyState.DISBANDED)

            return True

    def promote_leader(self, new_leader_id: str, promoted_by: str) -> bool:
        """
        Promote a member to party leader.

        Args:
            new_leader_id: The player to promote.
            promoted_by: The current leader making the promotion.

        Returns:
            True if promoted, False otherwise.
        """
        with self._lock:
            # Only leader can promote
            if promoted_by != self.leader_id:
                return False

            if new_leader_id not in self._members:
                return False

            if new_leader_id == self.leader_id:
                return False  # Already leader

            # Demote current leader
            self._members[self.leader_id].role = PartyRole.MEMBER

            # Promote new leader
            self._members[new_leader_id].role = PartyRole.LEADER
            self.leader_id = new_leader_id

            if self._on_leader_change:
                self._on_leader_change(self, new_leader_id)

            return True

    def set_ready(self, player_id: str, ready: bool = True) -> bool:
        """
        Set a member's ready status.

        Args:
            player_id: The player's ID.
            ready: The ready status to set.

        Returns:
            True if updated, False if player not in party.
        """
        with self._lock:
            if player_id not in self._members:
                return False

            self._members[player_id].is_ready = ready

            # Check if all ready
            if ready and self._all_ready_internal() and self._on_all_ready:
                self._on_all_ready(self)

            return True

    def set_all_not_ready(self) -> None:
        """Reset all members to not ready."""
        with self._lock:
            for member in self._members.values():
                member.is_ready = False

    def set_state(self, new_state: PartyState, changed_by: str) -> bool:
        """
        Change the party state (leader only).

        Args:
            new_state: The new state.
            changed_by: The player changing state (must be leader).

        Returns:
            True if changed, False otherwise.
        """
        with self._lock:
            if changed_by != self.leader_id:
                return False

            self._change_state(new_state)
            return True

    def disband(self, disbanded_by: str) -> bool:
        """
        Disband the party.

        Args:
            disbanded_by: The player disbanding (must be leader).

        Returns:
            True if disbanded, False otherwise.
        """
        with self._lock:
            if disbanded_by != self.leader_id:
                return False

            if self.state == PartyState.DISBANDED:
                return False

            self._pending_invites.clear()
            self._members.clear()
            self._change_state(PartyState.DISBANDED)

            return True

    def cleanup_expired_invites(self) -> int:
        """
        Remove expired invites.

        Returns:
            Number of invites removed.
        """
        with self._lock:
            expired = [
                inv_id for inv_id, inv in self._pending_invites.items()
                if inv.is_expired
            ]
            for inv_id in expired:
                del self._pending_invites[inv_id]
            return len(expired)

    def to_dict(self) -> dict[str, Any]:
        """Convert party to dictionary for serialization."""
        with self._lock:
            return {
                "id": self.id,
                "leader_id": self.leader_id,
                "state": self.state.name,
                "size": len(self._members),
                "max_size": self.max_size,
                "members": [
                    {
                        "id": m.player_id,
                        "name": m.display_name,
                        "role": m.role.name,
                        "is_ready": m.is_ready
                    }
                    for m in self._members.values()
                ],
                "pending_invites": len([
                    i for i in self._pending_invites.values()
                    if not i.is_expired
                ]),
                "created_at": self.created_at
            }


class PartyManager:
    """
    Manages multiple parties across the system.

    Thread-safe for concurrent access.
    """

    def __init__(self, default_max_size: Optional[int] = None) -> None:
        """
        Initialize the party manager.

        Args:
            default_max_size: Default maximum party size.
        """
        self._parties: dict[str, Party] = {}
        self._player_parties: dict[str, str] = {}  # player_id -> party_id
        self._player_invites: dict[str, list[PartyInvite]] = {}  # player_id -> invites
        self._lock = Lock()
        self.default_max_size = default_max_size if default_max_size is not None else SOCIAL_CONFIG.Party.MAX_SIZE_DEFAULT

        # Callbacks
        self._on_party_created: Optional[Callable[[Party], None]] = None
        self._on_party_disbanded: Optional[Callable[[str], None]] = None

    def set_on_party_created(self, callback: Callable[[Party], None]) -> None:
        """Set callback for party creation."""
        self._on_party_created = callback

    def set_on_party_disbanded(self, callback: Callable[[str], None]) -> None:
        """Set callback for party disbanding."""
        self._on_party_disbanded = callback

    def create_party(
        self,
        leader_id: str,
        leader_name: str,
        max_size: Optional[int] = None
    ) -> Optional[Party]:
        """
        Create a new party.

        Args:
            leader_id: The party leader's player ID.
            leader_name: The party leader's display name.
            max_size: Maximum party size (uses default if not specified).

        Returns:
            The created Party, or None if player already in a party.
        """
        with self._lock:
            # Check if player is already in a party
            if leader_id in self._player_parties:
                return None

            party = Party(
                leader_id,
                leader_name,
                max_size or self.default_max_size
            )

            self._parties[party.id] = party
            self._player_parties[leader_id] = party.id

            if self._on_party_created:
                self._on_party_created(party)

            return party

    def get_party(self, party_id: str) -> Optional[Party]:
        """Get a party by ID."""
        with self._lock:
            return self._parties.get(party_id)

    def get_player_party(self, player_id: str) -> Optional[Party]:
        """Get the party a player is currently in."""
        with self._lock:
            party_id = self._player_parties.get(player_id)
            if party_id:
                return self._parties.get(party_id)
            return None

    def send_invite(
        self,
        party_id: str,
        to_player_id: str,
        invited_by: str,
        inviter_name: str = "Unknown",
        expire_seconds: Optional[float] = None
    ) -> Optional[PartyInvite]:
        """
        Send a party invitation.

        Args:
            party_id: The party to invite to.
            to_player_id: The player to invite.
            invited_by: The player sending the invite.
            inviter_name: Display name of the inviter.
            expire_seconds: Seconds until invite expires.

        Returns:
            The PartyInvite if created, None otherwise.
        """
        if expire_seconds is None:
            expire_seconds = SOCIAL_CONFIG.Party.INVITE_EXPIRE_SECONDS_DEFAULT

        with self._lock:
            # Can't invite someone already in a party
            if to_player_id in self._player_parties:
                return None

            party = self._parties.get(party_id)
            if not party:
                return None

            invite = party.invite(
                to_player_id,
                invited_by,
                inviter_name,
                expire_seconds
            )

            if invite:
                # Track invite for the target player
                if to_player_id not in self._player_invites:
                    self._player_invites[to_player_id] = []
                self._player_invites[to_player_id].append(invite)

            return invite

    def get_player_invites(self, player_id: str) -> list[PartyInvite]:
        """Get all pending invites for a player."""
        with self._lock:
            invites = self._player_invites.get(player_id, [])
            # Filter out expired
            return [inv for inv in invites if not inv.is_expired]

    def accept_invite(
        self,
        invite_id: str,
        player_id: str,
        player_name: str
    ) -> bool:
        """
        Accept a party invitation.

        Args:
            invite_id: The invite to accept.
            player_id: The accepting player's ID.
            player_name: The accepting player's display name.

        Returns:
            True if successfully joined, False otherwise.
        """
        with self._lock:
            # Find the invite
            invites = self._player_invites.get(player_id, [])
            invite = next((i for i in invites if i.id == invite_id), None)

            if not invite:
                return False

            # Can't join if already in a party
            if player_id in self._player_parties:
                return False

            party = self._parties.get(invite.party_id)
            if not party:
                return False

            if party.accept_invite(invite_id, player_id, player_name):
                self._player_parties[player_id] = party.id
                # Remove all invites for this player
                self._player_invites[player_id] = []
                return True

            return False

    def decline_invite(self, invite_id: str, player_id: str) -> bool:
        """
        Decline a party invitation.

        Args:
            invite_id: The invite to decline.
            player_id: The declining player's ID.

        Returns:
            True if declined, False otherwise.
        """
        with self._lock:
            invites = self._player_invites.get(player_id, [])
            invite = next((i for i in invites if i.id == invite_id), None)

            if not invite:
                return False

            party = self._parties.get(invite.party_id)
            if party:
                party.decline_invite(invite_id, player_id)

            # Remove from player's invites
            self._player_invites[player_id] = [
                i for i in invites if i.id != invite_id
            ]
            return True

    def leave_party(self, player_id: str) -> bool:
        """
        Leave the current party.

        Args:
            player_id: The leaving player's ID.

        Returns:
            True if successfully left, False otherwise.
        """
        with self._lock:
            party_id = self._player_parties.get(player_id)
            if not party_id:
                return False

            party = self._parties.get(party_id)
            if not party:
                del self._player_parties[player_id]
                return False

            if party.leave(player_id):
                del self._player_parties[player_id]

                # Check if party was disbanded
                if party.state == PartyState.DISBANDED:
                    del self._parties[party_id]
                    if self._on_party_disbanded:
                        self._on_party_disbanded(party_id)

                return True

            return False

    def dissolve_party(self, party_id: str, dissolved_by: str) -> bool:
        """
        Dissolve/disband a party.

        Args:
            party_id: The party to dissolve.
            dissolved_by: The player dissolving (must be leader).

        Returns:
            True if dissolved, False otherwise.
        """
        with self._lock:
            party = self._parties.get(party_id)
            if not party:
                return False

            member_ids = party.get_member_ids()

            if party.disband(dissolved_by):
                # Remove all members from tracking
                for member_id in member_ids:
                    if member_id in self._player_parties:
                        del self._player_parties[member_id]

                del self._parties[party_id]

                if self._on_party_disbanded:
                    self._on_party_disbanded(party_id)

                return True

            return False

    def get_party_count(self) -> int:
        """Get the total number of active parties."""
        with self._lock:
            return len(self._parties)

    def cleanup_disbanded_parties(self) -> int:
        """
        Remove disbanded parties from the manager.

        Returns:
            Number of parties removed.
        """
        with self._lock:
            disbanded_ids = [
                pid for pid, party in self._parties.items()
                if party.state == PartyState.DISBANDED
            ]

            for pid in disbanded_ids:
                del self._parties[pid]
                if self._on_party_disbanded:
                    self._on_party_disbanded(pid)

            return len(disbanded_ids)

    def cleanup_expired_invites(self) -> int:
        """
        Remove expired invites from all parties and tracking.

        Returns:
            Total number of invites removed.
        """
        with self._lock:
            total_removed = 0

            # Clean party invites
            for party in self._parties.values():
                total_removed += party.cleanup_expired_invites()

            # Clean player invite tracking
            for player_id in list(self._player_invites.keys()):
                invites = self._player_invites[player_id]
                valid = [i for i in invites if not i.is_expired]
                removed = len(invites) - len(valid)
                total_removed += removed

                if valid:
                    self._player_invites[player_id] = valid
                else:
                    del self._player_invites[player_id]

            return total_removed

    def get_stats(self) -> dict[str, Any]:
        """Get party manager statistics."""
        with self._lock:
            total_players = len(self._player_parties)
            total_parties = len(self._parties)
            total_invites = sum(
                len([i for i in invs if not i.is_expired])
                for invs in self._player_invites.values()
            )

            sizes = [p.size for p in self._parties.values()]
            avg_size = sum(sizes) / len(sizes) if sizes else 0

            return {
                "total_parties": total_parties,
                "total_players_in_parties": total_players,
                "total_pending_invites": total_invites,
                "average_party_size": avg_size
            }
