"""XR Social Services - Platform social integration for multiplayer XR.

This module provides social services integration for XR platforms:
- Friends list and presence
- Party/group management
- Invites and join requests
- Voice chat integration
- Avatar identity and customization
- Cross-platform social features
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Callable, Any
import logging
from datetime import datetime

from engine.xr.config import XR_CONFIG

logger = logging.getLogger(__name__)


class UserPresence(Enum):
    """User online presence status."""

    OFFLINE = auto()
    ONLINE = auto()
    AWAY = auto()
    BUSY = auto()
    IN_GAME = auto()
    IN_VR = auto()
    INVISIBLE = auto()


class FriendRelationship(Enum):
    """Relationship status with another user."""

    NONE = auto()
    PENDING_SENT = auto()  # Request sent, awaiting response
    PENDING_RECEIVED = auto()  # Request received, awaiting action
    FRIEND = auto()
    BLOCKED = auto()


class PartyState(Enum):
    """Party/group state."""

    IDLE = auto()
    FORMING = auto()
    READY = auto()
    IN_SESSION = auto()
    DISBANDED = auto()


class InviteType(Enum):
    """Types of invitations."""

    PARTY = auto()
    GAME_SESSION = auto()
    FRIEND_REQUEST = auto()
    VOICE_CHANNEL = auto()


class VoiceChatState(Enum):
    """Voice chat connection state."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    MUTED = auto()
    DEAFENED = auto()


@dataclass
class UserProfile:
    """User profile information."""

    user_id: str = ""
    display_name: str = ""
    avatar_url: str = ""
    avatar_id: str = ""  # For custom 3D avatars

    # Status
    presence: UserPresence = UserPresence.OFFLINE
    status_message: str = ""
    current_app_id: str = ""
    current_session_id: str = ""

    # Stats
    last_online: Optional[datetime] = None
    friend_count: int = 0
    vr_hours: float = 0.0

    # Platform-specific
    platform_id: str = ""  # Platform-specific ID
    platform_name: str = ""  # Steam, Oculus, PSN, etc.

    # XR-specific
    preferred_hand: str = "right"
    height: float = XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M
    ipd: float = XR_CONFIG.runtime.DEFAULT_IPD_MM / 1000.0


@dataclass
class Friend:
    """Friend entry with relationship info."""

    profile: UserProfile = field(default_factory=UserProfile)
    relationship: FriendRelationship = FriendRelationship.NONE
    nickname: str = ""  # Custom nickname
    favorite: bool = False
    added_at: Optional[datetime] = None
    last_played_together: Optional[datetime] = None


@dataclass
class PartyMember:
    """Party member information."""

    profile: UserProfile = field(default_factory=UserProfile)
    is_leader: bool = False
    is_ready: bool = False
    joined_at: Optional[datetime] = None
    voice_state: VoiceChatState = VoiceChatState.DISCONNECTED
    is_speaking: bool = False


@dataclass
class Party:
    """Party/group information."""

    party_id: str = ""
    name: str = ""
    state: PartyState = PartyState.IDLE

    # Members
    members: List[PartyMember] = field(default_factory=list)
    max_members: int = 8
    leader_id: str = ""

    # Privacy
    is_public: bool = False
    invite_only: bool = True
    join_in_progress: bool = True

    # Session info
    current_app_id: str = ""
    current_session_id: str = ""

    # Voice
    voice_channel_id: str = ""
    voice_enabled: bool = True


@dataclass
class Invite:
    """Invitation information."""

    invite_id: str = ""
    invite_type: InviteType = InviteType.GAME_SESSION
    sender: UserProfile = field(default_factory=UserProfile)
    recipient_id: str = ""

    # Target
    party_id: str = ""
    session_id: str = ""
    app_id: str = ""

    # Metadata
    message: str = ""
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # State
    accepted: bool = False
    declined: bool = False


@dataclass
class VoiceChannel:
    """Voice chat channel information."""

    channel_id: str = ""
    name: str = ""
    participants: List[str] = field(default_factory=list)  # User IDs

    # Settings
    max_participants: int = 16
    is_positional: bool = True  # Spatial audio
    falloff_distance: float = XR_CONFIG.platform.NAME_TAG_FADE_DISTANCE_M
    max_distance: float = XR_CONFIG.platform.AVATAR_VISIBLE_DISTANCE_M

    # State
    state: VoiceChatState = VoiceChatState.DISCONNECTED


class SocialServices(ABC):
    """Abstract base class for social services.

    Provides unified interface for platform-specific social features
    including friends, parties, invites, and voice chat.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._current_user: Optional[UserProfile] = None
        self._friends: Dict[str, Friend] = {}
        self._party: Optional[Party] = None
        self._pending_invites: Dict[str, Invite] = {}

        # Event callbacks
        self._on_presence_changed: List[Callable[[str, UserPresence], None]] = []
        self._on_friend_added: List[Callable[[Friend], None]] = []
        self._on_friend_removed: List[Callable[[str], None]] = []
        self._on_invite_received: List[Callable[[Invite], None]] = []
        self._on_party_updated: List[Callable[[Party], None]] = []
        self._on_voice_state_changed: List[Callable[[str, VoiceChatState], None]] = []

    @property
    def current_user(self) -> Optional[UserProfile]:
        """Get current user profile."""
        return self._current_user

    @property
    def friends(self) -> Dict[str, Friend]:
        """Get friends dictionary keyed by user ID."""
        return self._friends.copy()

    @property
    def party(self) -> Optional[Party]:
        """Get current party."""
        return self._party

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize social services.

        Returns:
            True if initialization succeeded.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown social services."""
        pass

    # User Profile
    @abstractmethod
    def get_current_user(self) -> Optional[UserProfile]:
        """Get current user's profile.

        Returns:
            Current user profile or None if not signed in.
        """
        pass

    @abstractmethod
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user's profile by ID.

        Args:
            user_id: User identifier.

        Returns:
            User profile or None if not found.
        """
        pass

    @abstractmethod
    def set_presence(self, presence: UserPresence, status: str = "") -> bool:
        """Set current user's presence status.

        Args:
            presence: Presence status.
            status: Optional status message.

        Returns:
            True if presence was set.
        """
        pass

    # Friends
    @abstractmethod
    def get_friends_list(self) -> List[Friend]:
        """Get list of friends.

        Returns:
            List of friends.
        """
        pass

    @abstractmethod
    def send_friend_request(self, user_id: str) -> bool:
        """Send friend request to a user.

        Args:
            user_id: Target user ID.

        Returns:
            True if request was sent.
        """
        pass

    @abstractmethod
    def accept_friend_request(self, user_id: str) -> bool:
        """Accept a friend request.

        Args:
            user_id: Requesting user's ID.

        Returns:
            True if request was accepted.
        """
        pass

    @abstractmethod
    def decline_friend_request(self, user_id: str) -> bool:
        """Decline a friend request.

        Args:
            user_id: Requesting user's ID.

        Returns:
            True if request was declined.
        """
        pass

    @abstractmethod
    def remove_friend(self, user_id: str) -> bool:
        """Remove a friend.

        Args:
            user_id: Friend's user ID.

        Returns:
            True if friend was removed.
        """
        pass

    @abstractmethod
    def block_user(self, user_id: str) -> bool:
        """Block a user.

        Args:
            user_id: User to block.

        Returns:
            True if user was blocked.
        """
        pass

    # Party
    @abstractmethod
    def create_party(self, max_members: int = 8) -> Optional[Party]:
        """Create a new party.

        Args:
            max_members: Maximum party size.

        Returns:
            Created party or None if failed.
        """
        pass

    @abstractmethod
    def join_party(self, party_id: str) -> bool:
        """Join an existing party.

        Args:
            party_id: Party identifier.

        Returns:
            True if joined successfully.
        """
        pass

    @abstractmethod
    def leave_party(self) -> bool:
        """Leave current party.

        Returns:
            True if left successfully.
        """
        pass

    @abstractmethod
    def kick_party_member(self, user_id: str) -> bool:
        """Kick a member from the party (leader only).

        Args:
            user_id: Member to kick.

        Returns:
            True if member was kicked.
        """
        pass

    @abstractmethod
    def promote_party_leader(self, user_id: str) -> bool:
        """Promote a member to party leader.

        Args:
            user_id: Member to promote.

        Returns:
            True if promoted.
        """
        pass

    @abstractmethod
    def set_party_ready(self, is_ready: bool) -> bool:
        """Set ready status in party.

        Args:
            is_ready: Whether user is ready.

        Returns:
            True if status was set.
        """
        pass

    # Invites
    @abstractmethod
    def send_invite(
        self,
        user_id: str,
        invite_type: InviteType,
        message: str = ""
    ) -> Optional[Invite]:
        """Send an invitation.

        Args:
            user_id: Target user ID.
            invite_type: Type of invitation.
            message: Optional message.

        Returns:
            Created invite or None if failed.
        """
        pass

    @abstractmethod
    def accept_invite(self, invite_id: str) -> bool:
        """Accept an invitation.

        Args:
            invite_id: Invite identifier.

        Returns:
            True if accepted.
        """
        pass

    @abstractmethod
    def decline_invite(self, invite_id: str) -> bool:
        """Decline an invitation.

        Args:
            invite_id: Invite identifier.

        Returns:
            True if declined.
        """
        pass

    @abstractmethod
    def get_pending_invites(self) -> List[Invite]:
        """Get pending invites.

        Returns:
            List of pending invites.
        """
        pass

    # Voice Chat
    @abstractmethod
    def join_voice_channel(self, channel_id: str) -> bool:
        """Join a voice channel.

        Args:
            channel_id: Channel to join.

        Returns:
            True if joined.
        """
        pass

    @abstractmethod
    def leave_voice_channel(self) -> bool:
        """Leave current voice channel.

        Returns:
            True if left.
        """
        pass

    @abstractmethod
    def set_voice_muted(self, muted: bool) -> bool:
        """Set voice mute state.

        Args:
            muted: Whether to mute.

        Returns:
            True if state was set.
        """
        pass

    @abstractmethod
    def set_voice_deafened(self, deafened: bool) -> bool:
        """Set voice deafen state.

        Args:
            deafened: Whether to deafen.

        Returns:
            True if state was set.
        """
        pass

    # Event subscriptions
    def on_presence_changed(
        self,
        callback: Callable[[str, UserPresence], None]
    ) -> None:
        """Subscribe to presence changes.

        Args:
            callback: Callback(user_id, new_presence).
        """
        self._on_presence_changed.append(callback)

    def on_friend_added(
        self,
        callback: Callable[[Friend], None]
    ) -> None:
        """Subscribe to friend additions.

        Args:
            callback: Callback(new_friend).
        """
        self._on_friend_added.append(callback)

    def on_friend_removed(
        self,
        callback: Callable[[str], None]
    ) -> None:
        """Subscribe to friend removals.

        Args:
            callback: Callback(removed_user_id).
        """
        self._on_friend_removed.append(callback)

    def on_invite_received(
        self,
        callback: Callable[[Invite], None]
    ) -> None:
        """Subscribe to invite receipts.

        Args:
            callback: Callback(invite).
        """
        self._on_invite_received.append(callback)

    def on_party_updated(
        self,
        callback: Callable[[Party], None]
    ) -> None:
        """Subscribe to party updates.

        Args:
            callback: Callback(updated_party).
        """
        self._on_party_updated.append(callback)

    def on_voice_state_changed(
        self,
        callback: Callable[[str, VoiceChatState], None]
    ) -> None:
        """Subscribe to voice state changes.

        Args:
            callback: Callback(user_id, new_state).
        """
        self._on_voice_state_changed.append(callback)

    # Helper methods
    def _emit_presence_changed(
        self,
        user_id: str,
        presence: UserPresence
    ) -> None:
        """Emit presence changed event."""
        for callback in self._on_presence_changed:
            try:
                callback(user_id, presence)
            except Exception as e:
                logger.error(f"Error in presence callback: {e}")

    def _emit_friend_added(self, friend: Friend) -> None:
        """Emit friend added event."""
        for callback in self._on_friend_added:
            try:
                callback(friend)
            except Exception as e:
                logger.error(f"Error in friend added callback: {e}")

    def _emit_invite_received(self, invite: Invite) -> None:
        """Emit invite received event."""
        for callback in self._on_invite_received:
            try:
                callback(invite)
            except Exception as e:
                logger.error(f"Error in invite callback: {e}")

    def _emit_party_updated(self, party: Party) -> None:
        """Emit party updated event."""
        for callback in self._on_party_updated:
            try:
                callback(party)
            except Exception as e:
                logger.error(f"Error in party callback: {e}")


class MetaSocialServices(SocialServices):
    """Meta/Oculus social services implementation."""

    def initialize(self) -> bool:
        """Initialize Meta Platform SDK."""
        try:
            logger.info("Initializing Meta social services")
            self._current_user = self.get_current_user()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Meta social: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown Meta social services."""
        self._initialized = False
        logger.info("Meta social services shutdown")

    def get_current_user(self) -> Optional[UserProfile]:
        """Get current Meta user."""
        # TODO: Query Oculus Platform SDK
        return UserProfile(
            user_id="meta_user_1",
            display_name="VR Player",
            platform_name="Meta",
            presence=UserPresence.IN_VR,
        )

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get Meta user profile."""
        # TODO: ovr_User_Get
        return None

    def set_presence(self, presence: UserPresence, status: str = "") -> bool:
        """Set Meta presence."""
        # TODO: ovr_RichPresence_Set
        return True

    def get_friends_list(self) -> List[Friend]:
        """Get Meta friends list."""
        # TODO: ovr_User_GetLoggedInUserFriends
        return []

    def send_friend_request(self, user_id: str) -> bool:
        """Send Meta friend request."""
        return True

    def accept_friend_request(self, user_id: str) -> bool:
        """Accept Meta friend request."""
        return True

    def decline_friend_request(self, user_id: str) -> bool:
        """Decline Meta friend request."""
        return True

    def remove_friend(self, user_id: str) -> bool:
        """Remove Meta friend."""
        return True

    def block_user(self, user_id: str) -> bool:
        """Block Meta user."""
        return True

    def create_party(self, max_members: int = 8) -> Optional[Party]:
        """Create Meta party."""
        # TODO: ovr_Party_Create
        party = Party(
            party_id="meta_party_1",
            max_members=max_members,
            leader_id=self._current_user.user_id if self._current_user else "",
        )
        self._party = party
        return party

    def join_party(self, party_id: str) -> bool:
        """Join Meta party."""
        # TODO: ovr_Party_Join
        return True

    def leave_party(self) -> bool:
        """Leave Meta party."""
        # TODO: ovr_Party_Leave
        self._party = None
        return True

    def kick_party_member(self, user_id: str) -> bool:
        """Kick from Meta party."""
        return True

    def promote_party_leader(self, user_id: str) -> bool:
        """Promote Meta party leader."""
        return True

    def set_party_ready(self, is_ready: bool) -> bool:
        """Set ready in Meta party."""
        return True

    def send_invite(
        self,
        user_id: str,
        invite_type: InviteType,
        message: str = ""
    ) -> Optional[Invite]:
        """Send Meta invite."""
        # TODO: ovr_User_LaunchInvitePanel or ovr_Notification_Send
        invite = Invite(
            invite_id=f"meta_invite_{user_id}",
            invite_type=invite_type,
            sender=self._current_user or UserProfile(),
            recipient_id=user_id,
            message=message,
            created_at=datetime.now(),
        )
        return invite

    def accept_invite(self, invite_id: str) -> bool:
        """Accept Meta invite."""
        return True

    def decline_invite(self, invite_id: str) -> bool:
        """Decline Meta invite."""
        return True

    def get_pending_invites(self) -> List[Invite]:
        """Get pending Meta invites."""
        return list(self._pending_invites.values())

    def join_voice_channel(self, channel_id: str) -> bool:
        """Join Meta voice channel."""
        # TODO: ovr_Party_Join (includes voice)
        return True

    def leave_voice_channel(self) -> bool:
        """Leave Meta voice channel."""
        return True

    def set_voice_muted(self, muted: bool) -> bool:
        """Set Meta voice mute."""
        return True

    def set_voice_deafened(self, deafened: bool) -> bool:
        """Set Meta voice deafen."""
        return True


class SteamSocialServices(SocialServices):
    """Steam social services implementation."""

    def initialize(self) -> bool:
        """Initialize Steam API."""
        try:
            logger.info("Initializing Steam social services")
            self._current_user = self.get_current_user()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Steam social: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown Steam social services."""
        self._initialized = False
        logger.info("Steam social services shutdown")

    def get_current_user(self) -> Optional[UserProfile]:
        """Get current Steam user."""
        # TODO: ISteamUser::GetSteamID, ISteamFriends::GetPersonaName
        return UserProfile(
            user_id="steam_user_1",
            display_name="Steam VR Player",
            platform_name="Steam",
            presence=UserPresence.IN_VR,
        )

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get Steam user profile."""
        # TODO: ISteamFriends::GetFriendPersonaName
        return None

    def set_presence(self, presence: UserPresence, status: str = "") -> bool:
        """Set Steam rich presence."""
        # TODO: ISteamFriends::SetRichPresence
        return True

    def get_friends_list(self) -> List[Friend]:
        """Get Steam friends list."""
        # TODO: ISteamFriends::GetFriendCount, GetFriendByIndex
        return []

    def send_friend_request(self, user_id: str) -> bool:
        """Steam doesn't support programmatic friend requests."""
        logger.warning("Steam friend requests require Steam overlay")
        return False

    def accept_friend_request(self, user_id: str) -> bool:
        """Steam friend requests handled via overlay."""
        return False

    def decline_friend_request(self, user_id: str) -> bool:
        """Steam friend requests handled via overlay."""
        return False

    def remove_friend(self, user_id: str) -> bool:
        """Steam friend removal via overlay."""
        return False

    def block_user(self, user_id: str) -> bool:
        """Block Steam user."""
        return True

    def create_party(self, max_members: int = 8) -> Optional[Party]:
        """Create Steam lobby."""
        # TODO: ISteamMatchmaking::CreateLobby
        party = Party(
            party_id="steam_lobby_1",
            max_members=max_members,
            leader_id=self._current_user.user_id if self._current_user else "",
        )
        self._party = party
        return party

    def join_party(self, party_id: str) -> bool:
        """Join Steam lobby."""
        # TODO: ISteamMatchmaking::JoinLobby
        return True

    def leave_party(self) -> bool:
        """Leave Steam lobby."""
        # TODO: ISteamMatchmaking::LeaveLobby
        self._party = None
        return True

    def kick_party_member(self, user_id: str) -> bool:
        """Kick from Steam lobby (if owner)."""
        return True

    def promote_party_leader(self, user_id: str) -> bool:
        """Transfer Steam lobby ownership."""
        # TODO: ISteamMatchmaking::SetLobbyOwner
        return True

    def set_party_ready(self, is_ready: bool) -> bool:
        """Set ready in Steam lobby."""
        # TODO: ISteamMatchmaking::SetLobbyMemberData
        return True

    def send_invite(
        self,
        user_id: str,
        invite_type: InviteType,
        message: str = ""
    ) -> Optional[Invite]:
        """Send Steam invite."""
        # TODO: ISteamFriends::InviteUserToGame
        invite = Invite(
            invite_id=f"steam_invite_{user_id}",
            invite_type=invite_type,
            sender=self._current_user or UserProfile(),
            recipient_id=user_id,
            message=message,
            created_at=datetime.now(),
        )
        return invite

    def accept_invite(self, invite_id: str) -> bool:
        """Accept Steam invite (handled via overlay)."""
        return True

    def decline_invite(self, invite_id: str) -> bool:
        """Decline Steam invite."""
        return True

    def get_pending_invites(self) -> List[Invite]:
        """Get pending Steam invites."""
        return list(self._pending_invites.values())

    def join_voice_channel(self, channel_id: str) -> bool:
        """Join Steam voice channel."""
        # TODO: ISteamNetworkingSockets for voice
        return True

    def leave_voice_channel(self) -> bool:
        """Leave Steam voice channel."""
        return True

    def set_voice_muted(self, muted: bool) -> bool:
        """Set Steam voice mute."""
        return True

    def set_voice_deafened(self, deafened: bool) -> bool:
        """Set Steam voice deafen."""
        return True


class PlayStationSocialServices(SocialServices):
    """PlayStation Network social services for PSVR2."""

    def initialize(self) -> bool:
        """Initialize PSN services."""
        try:
            logger.info("Initializing PlayStation social services")
            self._current_user = self.get_current_user()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PSN social: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown PSN services."""
        self._initialized = False
        logger.info("PlayStation social services shutdown")

    def get_current_user(self) -> Optional[UserProfile]:
        """Get current PSN user."""
        return UserProfile(
            user_id="psn_user_1",
            display_name="PS VR Player",
            platform_name="PlayStation",
            presence=UserPresence.IN_VR,
        )

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get PSN user profile."""
        return None

    def set_presence(self, presence: UserPresence, status: str = "") -> bool:
        """Set PSN presence."""
        return True

    def get_friends_list(self) -> List[Friend]:
        """Get PSN friends list."""
        return []

    def send_friend_request(self, user_id: str) -> bool:
        """Send PSN friend request."""
        return True

    def accept_friend_request(self, user_id: str) -> bool:
        """Accept PSN friend request."""
        return True

    def decline_friend_request(self, user_id: str) -> bool:
        """Decline PSN friend request."""
        return True

    def remove_friend(self, user_id: str) -> bool:
        """Remove PSN friend."""
        return True

    def block_user(self, user_id: str) -> bool:
        """Block PSN user."""
        return True

    def create_party(self, max_members: int = 8) -> Optional[Party]:
        """Create PSN party."""
        party = Party(
            party_id="psn_party_1",
            max_members=min(max_members, 16),  # PSN party limit
            leader_id=self._current_user.user_id if self._current_user else "",
            voice_enabled=True,
        )
        self._party = party
        return party

    def join_party(self, party_id: str) -> bool:
        """Join PSN party."""
        return True

    def leave_party(self) -> bool:
        """Leave PSN party."""
        self._party = None
        return True

    def kick_party_member(self, user_id: str) -> bool:
        """Kick from PSN party."""
        return True

    def promote_party_leader(self, user_id: str) -> bool:
        """Promote PSN party leader."""
        return True

    def set_party_ready(self, is_ready: bool) -> bool:
        """Set ready in PSN party."""
        return True

    def send_invite(
        self,
        user_id: str,
        invite_type: InviteType,
        message: str = ""
    ) -> Optional[Invite]:
        """Send PSN invite."""
        invite = Invite(
            invite_id=f"psn_invite_{user_id}",
            invite_type=invite_type,
            sender=self._current_user or UserProfile(),
            recipient_id=user_id,
            message=message,
            created_at=datetime.now(),
        )
        return invite

    def accept_invite(self, invite_id: str) -> bool:
        """Accept PSN invite."""
        return True

    def decline_invite(self, invite_id: str) -> bool:
        """Decline PSN invite."""
        return True

    def get_pending_invites(self) -> List[Invite]:
        """Get pending PSN invites."""
        return list(self._pending_invites.values())

    def join_voice_channel(self, channel_id: str) -> bool:
        """Join PSN voice channel (via party)."""
        return True

    def leave_voice_channel(self) -> bool:
        """Leave PSN voice channel."""
        return True

    def set_voice_muted(self, muted: bool) -> bool:
        """Set PSN voice mute."""
        return True

    def set_voice_deafened(self, deafened: bool) -> bool:
        """Set PSN voice deafen."""
        return True


def create_social_services(platform: str = "meta") -> SocialServices:
    """Create appropriate social services for platform.

    Args:
        platform: Platform name (meta, steam, playstation).

    Returns:
        Social services instance.
    """
    platform_lower = platform.lower()

    if platform_lower in ["steam", "steamvr"]:
        return SteamSocialServices()
    elif platform_lower in ["playstation", "psn", "psvr2"]:
        return PlayStationSocialServices()
    else:
        return MetaSocialServices()


__all__ = [
    # Enums
    "UserPresence",
    "FriendRelationship",
    "PartyState",
    "InviteType",
    "VoiceChatState",
    # Data classes
    "UserProfile",
    "Friend",
    "PartyMember",
    "Party",
    "Invite",
    "VoiceChannel",
    # Services
    "SocialServices",
    "MetaSocialServices",
    "SteamSocialServices",
    "PlayStationSocialServices",
    # Factory
    "create_social_services",
]
