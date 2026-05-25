"""XR Platform Integration - Platform abstraction for VR/AR/MR devices.

This module provides platform-specific integration including:
- Device detection and capabilities
- Guardian/boundary system
- Social services integration
- Store/entitlements (future)

Supported platforms:
- PC VR: Valve Index, HTC Vive, Oculus Rift, Windows Mixed Reality
- Standalone: Meta Quest, Pico
- AR: Phone AR (ARCore/ARKit), HoloLens, Magic Leap, Apple Vision Pro
- Console: PlayStation VR2
"""

from engine.xr.platform.platform_integration import (
    # Enums
    XRDevice,
    XRPlatformType,
    XRRuntime,
    # Data classes
    XRDeviceCapabilities,
    XRPlatformInfo,
    # Abstract base
    XRPlatform,
    # Platform implementations
    OpenXRPlatform,
    SteamVRPlatform,
    MetaQuestPlatform,
    AppleVisionProPlatform,
    PSVR2Platform,
    # Functions
    detect_xr_platform,
    get_device_capabilities,
)

from engine.xr.platform.guardian import (
    # Enums
    GuardianMode,
    BoundaryType,
    ProximityLevel,
    # Data classes
    BoundaryVertex,
    PlayAreaBounds,
    GuardianConfig,
    ProximityInfo,
    # Systems
    GuardianSystem,
    OpenXRGuardian,
    SteamVRGuardian,
    QuestGuardian,
    # Factory
    create_guardian_system,
)

from engine.xr.platform.social import (
    # Enums
    UserPresence,
    FriendRelationship,
    PartyState,
    InviteType,
    VoiceChatState,
    # Data classes
    UserProfile,
    Friend,
    PartyMember,
    Party,
    Invite,
    VoiceChannel,
    # Services
    SocialServices,
    MetaSocialServices,
    SteamSocialServices,
    PlayStationSocialServices,
    # Factory
    create_social_services,
)

__all__ = [
    # Platform Integration
    "XRDevice",
    "XRPlatformType",
    "XRRuntime",
    "XRDeviceCapabilities",
    "XRPlatformInfo",
    "XRPlatform",
    "OpenXRPlatform",
    "SteamVRPlatform",
    "MetaQuestPlatform",
    "AppleVisionProPlatform",
    "PSVR2Platform",
    "detect_xr_platform",
    "get_device_capabilities",
    # Guardian System
    "GuardianMode",
    "BoundaryType",
    "ProximityLevel",
    "BoundaryVertex",
    "PlayAreaBounds",
    "GuardianConfig",
    "ProximityInfo",
    "GuardianSystem",
    "OpenXRGuardian",
    "SteamVRGuardian",
    "QuestGuardian",
    "create_guardian_system",
    # Social Services
    "UserPresence",
    "FriendRelationship",
    "PartyState",
    "InviteType",
    "VoiceChatState",
    "UserProfile",
    "Friend",
    "PartyMember",
    "Party",
    "Invite",
    "VoiceChannel",
    "SocialServices",
    "MetaSocialServices",
    "SteamSocialServices",
    "PlayStationSocialServices",
    "create_social_services",
]
