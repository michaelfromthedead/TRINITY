# Phase 10 Architecture -- Social Services

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/social/`

---

## Overview

The social module provides multiplayer social features: matchmaking with skill-based ranking, lobby and party management, voice chat with proximity, text chat with moderation, and skill rating systems (Elo, Glicko-2, MMR).

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `matchmaking.py` | 547 | Queue with skill-based expansion and match formation |
| `skill_rating.py` | 662 | Elo (K-factor), Glicko-2, MMRManager |
| `lobby.py` | 847 | Lobby state machine with configurable slots and privacy |
| `party.py` | 921 | Party system with invite, join, leave, role management |
| `voice_chat.py` | 764 | 5 channel types, positional audio, proximity voice |
| `text_chat.py` | 848 | 7 channel types, l33t-aware profanity filter, rate limits |
| `config.py` | 173 | SocialConfig with 6 nested frozen dataclass configs |
| `__init__.py` | ~60 | Module exports |

---

## Architecture

### Matchmaking (matchmaking.py)

**MatchmakingQueue**: Per-region, per-playlist queue with skill-based expansion:

```
1. Player enters queue with skill rating + preferences
2. Queue periodically attempts to form matches:
   a. Sort by skill, expand search range over time
   b. Group into teams of appropriate size
   c. Validate team balance (avg skill per team)
3. On match found: create lobby, notify players
```

**Skill Expansion**: Search range starts narrow (e.g., +/- 50 MMR) and expands over time (e.g., +25 MMR per 10 seconds) to balance quality vs. wait time.

**MatchmakingService**: Top-level service managing multiple queues, handling player enter/leave/cancel, and match formation callbacks.

### Skill Rating (skill_rating.py)

**Three Rating Systems**:

| System | Description | Tunable Parameters |
|--------|-------------|-------------------|
| Elo | Classic zero-sum with K-factor | K-factor (default 32), initial rating (1500) |
| Glicko-2 | Rating + deviation + volatility | Tau, initial rating/deviation/volatility, convergence speed |
| MMRManager | Wraps both with persistence | Configurable system selection |

**MMRManager**: Unified interface that delegates to the configured system. Supports:
- Record match outcome (win/loss/draw)
- Get player rating and deviation
- Calculate match quality
- Adjust ratings for team games

### Lobby (lobby.py)

**Lobby**: Full state machine with slots, teams, privacy settings, and configuration.

**Key Features**:
- Configurable slot count and team assignment
- Privacy levels (PUBLIC, PRIVATE, FRIENDS_ONLY, INVITE_ONLY)
- Lobby owner with transferable ownership
- Ready-state tracking per player
- Game launch when all ready
- Matchmake -> Lobby -> Game flow integration

**LobbyManager**: Global lobby registry with CRUD operations, player membership tracking, and matchmaking integration.

### Party (party.py)

**Party**: Persistent group across game sessions with full lifecycle management.

**State Machine**:
```
DISBANDED --create--> ACTIVE --disband--> DISBANDED
ACTIVE --join--> ACTIVE --leave--> ACTIVE (or DISBANDED if last)
ACTIVE --invite--> ACTIVE --kick--> ACTIVE
```

**Key Features**:
- Party leader with transferable ownership
- Role management (leader, member)
- Invite/accept/decline flow with expiry
- Join-by-ID with optional password
- Cross-game session persistence
- Max member limit (configurable, default 8)

**PartyManager**: Global party registry handling CRUD plus invite lifecycle and auto-disband on last member leave.

### Voice Chat (voice_chat.py)

**VoiceChatManager**: Manages 5 audio channel types:
| Channel | Visibility | Use Case |
|---------|-----------|----------|
| GLOBAL | All players | Lobby chat, announcements |
| TEAM | Team members | Team coordination |
| LOCAL | Proximity range | In-world voice |
| PARTY | Party members | Cross-game party chat |
| CUSTOM | Whitelist | Custom groups |

**Proximity Voice (ProximityVoice)**: Spatial voice with:
- Distance-based volume falloff (configurable min/max range)
- 3D positional audio metadata
- Occlusion support (line-of-sight check)
- Configurable rolloff curve (linear, logarithmic, custom)

**VoiceConfig**: Per-channel mute, volume, and priority settings per player.

### Text Chat (text_chat.py)

**ChatManager**: Manages 7 channel types:
| Channel | Scope | Persistence |
|---------|-------|-------------|
| GLOBAL | All players | No |
| TEAM | Team members | No |
| PRIVATE | 1-on-1 | Yes (DM history) |
| PARTY | Party members | Yes |
| LOBBY | Lobby members | No |
| SYSTEM | System messages | N/A |
| CUSTOM | Custom scope | Configurable |

**Key Features**:
- Message history per channel (configurable limit)
- Per-channel mute (player/channel)
- Rate limiting per player/channel
- Typing indicators
- Emoji/rich text parsing

**ProfanityFilter**: Multi-strategy content filtering:
- Exact match against word list
- Levenshtein distance for fuzzy matching
- L33t-speak normalization (e.g., "h3ll0" -> "hello") before filter
- Configurable strictness levels
- Replacement character (default "*")

### Config (config.py)

**SocialConfig**: Singleton with 6 nested frozen dataclass configs:
- `MatchmakingConfig`: Queue params, expansion rate, team size
- `SkillConfig`: Rating system selection and parameters
- `LobbyConfig`: Max lobbies, slot limits, timeouts
- `PartyConfig`: Max members, invite expiry
- `VoiceConfig`: Channel settings, proximity ranges
- `ChatConfig`: History limits, rate limits, profanity list

---

## Missing Components

1. **Server Browser** (`social/server_browser.py`): Not implemented. Would provide server discovery via UDP broadcast query/list/response protocol. Required for LAN parties and community server lists.

---

## Reality Status

- Matchmaking (queue with skill expansion): **[x]** Complete
- Skill Rating (Elo, Glicko-2, MMR): **[x]** Complete
- Lobby (state machine, slots, privacy): **[x]** Complete
- Party (invite, join, leave, roles): **[x]** Complete
- Voice Chat (5 channels, proximity): **[x]** Complete
- Text Chat (7 channels, profanity filter): **[x]** Complete
- Config (6 nested configs): **[x]** Complete
- Server Browser: **[-]** Not implemented

---

*End of PHASE_10_ARCH.md*
