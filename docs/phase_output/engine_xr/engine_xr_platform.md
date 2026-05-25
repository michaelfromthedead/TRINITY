# XR Platform Investigation Report

**Module:** `engine/xr/platform/`
**Total Lines:** 2,813
**Date:** 2026-05-22
**Classification:** STUB (Well-Architected)

---

## Executive Summary

The XR platform module provides comprehensive hardware abstraction for VR/AR/MR devices, guardian/boundary safety systems, and social services integration. All implementations are **architectural stubs** with well-defined interfaces and data structures but no actual SDK bindings. The design demonstrates thorough domain knowledge of XR platform requirements.

---

## File Analysis

### 1. `__init__.py` (124 lines) - STUB

**Purpose:** Package exports consolidating all platform, guardian, and social APIs.

**Exports:**
- Platform integration: 12 symbols
- Guardian system: 12 symbols
- Social services: 16 symbols

**Classification:** STUB - Re-exports only, no implementation logic.

---

### 2. `platform_integration.py` (774 lines) - STUB

**Purpose:** Platform abstraction for VR/AR/MR device detection and capability reporting.

#### Device Enumeration (`XRDevice` enum - 35 devices)

| Category | Devices |
|----------|---------|
| **PC VR** | Valve Index, HTC Vive (4 variants), Oculus Rift/Rift S, HP Reverb G2, Samsung Odyssey, WMR Generic, Pimax 8KX, Bigscreen Beyond, Varjo Aero/XR3 |
| **Standalone VR** | Meta Quest 2/3/Pro, Pico 4/4 Enterprise/Neo 3, HTC Vive Focus 3 |
| **Mobile AR** | ARCore Phone/Tablet, ARKit Phone/Tablet |
| **AR Headsets** | HoloLens 2, Magic Leap 2, Apple Vision Pro, Nreal Air, XReal Air 2 |
| **Console** | PSVR2 |

#### Platform Types (`XRPlatformType` enum)

- `PC_VR`, `STANDALONE_VR`, `MOBILE_AR`, `AR_HEADSET`, `CONSOLE_VR`, `UNKNOWN`

#### Runtime APIs (`XRRuntime` enum)

- `OPENXR`, `STEAMVR`, `OCULUS_PC`, `OCULUS_MOBILE`, `WEBXR`, `ARCORE`, `ARKIT`, `WINDOWS_MR`, `PSVR`, `VISIONOS`

#### Device Capabilities (`XRDeviceCapabilities` dataclass)

| Category | Capabilities |
|----------|--------------|
| **Tracking** | 6DOF head/controllers, hand tracking, eye tracking, face tracking, body tracking |
| **Display** | Resolution, refresh rates, FOV, HDR, local dimming |
| **Rendering** | Foveated rendering, dynamic foveation, multiview, space warp |
| **Spatial** | Passthrough (mono/color), depth sensing, plane detection, mesh detection, spatial anchors, cloud anchors, scene understanding |
| **Input** | Controller type, haptics (channels), finger tracking |
| **Audio** | Integrated audio, spatial audio |
| **Power** | Tethered flag, battery capacity |
| **Performance** | Tier (low/medium/high/ultra) |

#### Platform Implementations

| Class | Runtime | Stub Methods | TODO Comments |
|-------|---------|--------------|---------------|
| `OpenXRPlatform` | OpenXR | `initialize()`, `detect_device()` | "Actual OpenXR initialization", "xrGetSystemProperties" |
| `SteamVRPlatform` | SteamVR | `initialize()`, `detect_device()`, `get_chaperone_bounds()`, `trigger_haptic_pulse()` | "Initialize OpenVR", "TrackedDeviceProperty", "IVRChaperone" |
| `MetaQuestPlatform` | Oculus Mobile | `initialize()`, `detect_device()`, `request_passthrough()`, `get_guardian_bounds()` | "Request passthrough via Meta SDK", "Query Guardian bounds" |
| `AppleVisionProPlatform` | visionOS | `initialize()`, `detect_device()`, `is_available()` | "Check for visionOS platform" |
| `PSVR2Platform` | PSVR | `initialize()`, `detect_device()`, `is_available()` | "Check PlayStation platform" |

#### Factory Functions

- `detect_xr_platform()` - Priority-ordered detection: Quest > Vision Pro > PSVR2 > SteamVR > OpenXR
- `get_device_capabilities(device)` - Static capability database for known devices

**Classification:** STUB - All `initialize()` and `detect_device()` return hardcoded defaults with TODO comments for SDK integration.

---

### 3. `guardian.py` (814 lines) - PARTIAL (Math Real, SDK Stub)

**Purpose:** Play area boundary detection, proximity warnings, and safety visualization.

#### Boundary Types (`BoundaryType` enum)

- `RECTANGLE`, `POLYGON`, `CYLINDER`, `CUSTOM_MESH`

#### Guardian Modes (`GuardianMode` enum)

- `DISABLED`, `STATIONARY`, `ROOM_SCALE`, `CUSTOM`, `PASS_THROUGH`

#### Proximity Levels (`ProximityLevel` enum)

- `SAFE`, `APPROACHING`, `NEAR`, `AT_BOUNDARY`, `OUTSIDE`

#### Data Structures

| Dataclass | Fields |
|-----------|--------|
| `BoundaryVertex` | x, y, z |
| `PlayAreaBounds` | vertices, boundary_type, width/depth/height, center, rotation, floor_height |
| `GuardianConfig` | mode, distance thresholds, colors (wall/grid/warning/danger), wall height, grid settings, passthrough settings, audio/haptic flags, fade settings |
| `ProximityInfo` | level, distance, nearest_point, normal |

#### Real Implementations (Math/Geometry)

| Method | Lines | Description |
|--------|-------|-------------|
| `PlayAreaBounds.get_area()` | ~20 | Shoelace formula for polygon area |
| `PlayAreaBounds.contains_point()` | ~30 | Ray casting point-in-polygon test |
| `GuardianSystem._calculate_proximity()` | ~115 | Distance to boundary for rectangle/polygon/cylinder |
| `GuardianSystem._point_to_segment_distance()` | ~20 | Point-to-segment projection |
| `GuardianSystem._nearest_point_on_segment()` | ~18 | Nearest point on line segment |
| `GuardianSystem.update()` | ~45 | Proximity state machine with event emission |
| `GuardianSystem.get_passthrough_blend()` | ~15 | Linear blend calculation |
| `GuardianSystem.get_warning_intensity()` | ~20 | Multi-level warning intensity |

#### Guardian System Implementations

| Class | Runtime | `request_bounds()` | Notes |
|-------|---------|-------------------|-------|
| `OpenXRGuardian` | OpenXR | Returns 2.5m x 2.5m rectangle | TODO: `xrGetReferenceSpaceBoundsRect` |
| `SteamVRGuardian` | SteamVR | Returns 2.5m x 2.5m rectangle | TODO: `IVRChaperone.GetPlayAreaRect` |
| `QuestGuardian` | Quest | Returns 3.0m x 3.0m polygon (4 vertices) | TODO: `OVR_Guardian.OvrBoundary_GetGeometry` |

#### Factory

- `create_guardian_system(runtime)` - Creates appropriate guardian for `steamvr`, `quest`, or `openxr`

**Classification:** PARTIAL
- **REAL:** All geometry/proximity math is implemented and functional
- **STUB:** SDK calls (`request_bounds()`, `set_custom_bounds()`, `recenter()`) return hardcoded defaults

---

### 4. `social.py` (1,101 lines) - STUB

**Purpose:** Cross-platform social services for multiplayer XR (friends, parties, invites, voice).

#### Social Enums

| Enum | Values |
|------|--------|
| `UserPresence` | OFFLINE, ONLINE, AWAY, BUSY, IN_GAME, IN_VR, INVISIBLE |
| `FriendRelationship` | NONE, PENDING_SENT, PENDING_RECEIVED, FRIEND, BLOCKED |
| `PartyState` | IDLE, FORMING, READY, IN_SESSION, DISBANDED |
| `InviteType` | PARTY, GAME_SESSION, FRIEND_REQUEST, VOICE_CHANNEL |
| `VoiceChatState` | DISCONNECTED, CONNECTING, CONNECTED, MUTED, DEAFENED |

#### Social Data Structures

| Dataclass | Key Fields |
|-----------|------------|
| `UserProfile` | user_id, display_name, avatar_url/id, presence, status_message, platform_id/name, XR settings (preferred_hand, height, IPD) |
| `Friend` | profile, relationship, nickname, favorite, timestamps |
| `PartyMember` | profile, is_leader, is_ready, voice_state, is_speaking |
| `Party` | party_id, name, state, members, max_members, leader_id, privacy settings, session info, voice settings |
| `Invite` | invite_id, type, sender, recipient_id, target (party/session/app), message, timestamps, accepted/declined |
| `VoiceChannel` | channel_id, name, participants, spatial audio settings (positional, falloff, max distance) |

#### SocialServices Abstract Interface (28 abstract methods)

| Category | Methods |
|----------|---------|
| **User Profile** | `get_current_user()`, `get_user_profile()`, `set_presence()` |
| **Friends** | `get_friends_list()`, `send_friend_request()`, `accept_friend_request()`, `decline_friend_request()`, `remove_friend()`, `block_user()` |
| **Party** | `create_party()`, `join_party()`, `leave_party()`, `kick_party_member()`, `promote_party_leader()`, `set_party_ready()` |
| **Invites** | `send_invite()`, `accept_invite()`, `decline_invite()`, `get_pending_invites()` |
| **Voice** | `join_voice_channel()`, `leave_voice_channel()`, `set_voice_muted()`, `set_voice_deafened()` |

#### Event Callbacks (Implemented in base class)

- `on_presence_changed(user_id, presence)`
- `on_friend_added(friend)`
- `on_friend_removed(user_id)`
- `on_invite_received(invite)`
- `on_party_updated(party)`
- `on_voice_state_changed(user_id, state)`

#### Platform Implementations

| Class | Platform | SDK TODO Comments |
|-------|----------|-------------------|
| `MetaSocialServices` | Meta/Oculus | `ovr_User_Get`, `ovr_RichPresence_Set`, `ovr_User_GetLoggedInUserFriends`, `ovr_Party_Create/Join/Leave`, `ovr_User_LaunchInvitePanel` |
| `SteamSocialServices` | Steam | `ISteamUser::GetSteamID`, `ISteamFriends::GetPersonaName/SetRichPresence/GetFriendCount`, `ISteamMatchmaking::CreateLobby/JoinLobby` |
| `PlayStationSocialServices` | PlayStation Network | No specific SDK comments (generic stub) |

#### Factory

- `create_social_services(platform)` - Creates services for `steam`/`steamvr`, `playstation`/`psn`/`psvr2`, or defaults to Meta

**Classification:** STUB - All methods return `True`, empty lists, or placeholder data. No actual SDK integration.

---

## Architecture Assessment

### Strengths

1. **Comprehensive Device Coverage:** 35 devices across 5 platform categories
2. **Well-Defined Abstractions:** Clear ABC interfaces with proper type hints
3. **Event-Driven Design:** Callback registration for async platform events
4. **Spatial Audio Support:** Voice channels include positional audio parameters
5. **XR-Specific User Profiles:** IPD, height, preferred hand stored per-user
6. **Guardian Math Complete:** All proximity/boundary geometry is implemented

### Implementation Gaps

| Area | Missing |
|------|---------|
| **OpenXR** | `pyopenxr` bindings, `xrGetSystemProperties`, `xrGetReferenceSpaceBoundsRect` |
| **SteamVR** | `openvr` package, `IVRSystem`, `IVRChaperone`, `ISteamMatchmaking` |
| **Meta Quest** | Oculus Platform SDK bindings, Guardian API, Passthrough API |
| **visionOS** | Swift/Objective-C bridge for ARKit, RealityKit |
| **PSVR2** | PlayStation SDK (requires devkit access) |
| **Voice Chat** | WebRTC or platform-specific voice implementations |

### Dependencies on Other Modules

- `engine.xr.config.XR_CONFIG` - Used for defaults (avatar height, IPD, guardian distances, grid size)

---

## Recommendations

1. **Priority SDK Integration:** OpenXR > SteamVR > Meta Quest (covers most desktop/standalone)
2. **Use `pyopenxr`:** Python OpenXR bindings exist and are mature
3. **Voice Integration:** Consider Vivox, Photon Voice, or WebRTC for cross-platform voice
4. **Guardian Testing:** Geometry math should be unit tested (no SDK needed)
5. **Capability Detection:** Runtime capability probing vs static database

---

## Summary Table

| File | Lines | Classification | Real Code | Stub Code |
|------|-------|----------------|-----------|-----------|
| `__init__.py` | 124 | STUB | 0% | 100% (re-exports) |
| `platform_integration.py` | 774 | STUB | 5% (enums/dataclasses) | 95% |
| `guardian.py` | 814 | PARTIAL | 40% (geometry math) | 60% (SDK calls) |
| `social.py` | 1,101 | STUB | 5% (enums/dataclasses) | 95% |
| **Total** | **2,813** | **STUB** | **~15%** | **~85%** |

---

## Conclusion

The XR platform module is a **well-architected stub layer** providing abstraction over VR/AR/MR hardware, safety systems, and social services. The guardian geometry code is the only substantial implementation. Platform-specific SDK bindings (OpenXR, SteamVR, Meta SDK, PSN) are documented in TODO comments but not implemented. The design is production-ready for integration once SDK bindings are added.
