# PHASE 5 TODO: Social and Matchmaking Systems

## Overview

Phase 5 implements the social layer. All tasks assume the existing implementation is production-ready; these TODOs focus on testing, verification, and identified gaps.

---

## 1. Matchmaking Tasks

### 1.1 Unit Tests: Queue Entry

**File**: `tests/blackbox_matchmaking.py`

**Acceptance Criteria**:
- [ ] `enter_queue()` creates QueueEntry
- [ ] Entry starts in SEARCHING state
- [ ] Queue time tracked from entry
- [ ] `leave_queue()` removes entry
- [ ] Duplicate entry rejected

---

### 1.2 Unit Tests: Skill Range Expansion

**File**: `tests/blackbox_skill_expansion.py`

**Acceptance Criteria**:
- [ ] Initial range matches criteria
- [ ] Range expands after interval
- [ ] Expansion rate applied correctly
- [ ] Expansion continues over time

---

### 1.3 Unit Tests: Match Formation

**File**: `tests/blackbox_match_formation.py`

**Acceptance Criteria**:
- [ ] Compatible players matched
- [ ] Mutual compatibility required
- [ ] Closest skill players selected
- [ ] Match callback invoked
- [ ] Players removed from queue

---

### 1.4 Unit Tests: Estimated Wait Time

**File**: `tests/blackbox_wait_time.py`

**Acceptance Criteria**:
- [ ] Estimate based on recent matches
- [ ] Multiplier applied
- [ ] Skill range affects estimate

---

## 2. Skill Rating Tasks

### 2.1 Unit Tests: Elo Calculator

**File**: `tests/blackbox_elo.py`

**Acceptance Criteria**:
- [ ] Expected score calculated correctly
- [ ] Winner gains points, loser loses
- [ ] Draw splits points
- [ ] K-factor varies by games/rating
- [ ] Floor rating enforced

---

### 2.2 Unit Tests: Glicko-2 Calculator

**File**: `tests/blackbox_glicko2.py`

**Acceptance Criteria**:
- [ ] Rating updates with uncertainty
- [ ] Deviation increases with inactivity
- [ ] Multi-opponent rating periods work
- [ ] Scale conversion accurate

---

### 2.3 Unit Tests: Leaderboard

**File**: `tests/blackbox_leaderboard.py`

**Acceptance Criteria**:
- [ ] Sorted by rating descending
- [ ] Minimum games filter applied
- [ ] Limit parameter respected
- [ ] Rank numbers correct

---

### 2.4 Unit Tests: Percentile

**File**: `tests/blackbox_percentile.py`

**Acceptance Criteria**:
- [ ] Percentile calculated correctly
- [ ] Edge cases (top/bottom) handled
- [ ] Unknown player returns 0

---

## 3. Party Tasks

### 3.1 Unit Tests: Party Creation

**File**: `tests/blackbox_party.py`

**Acceptance Criteria**:
- [ ] Creator becomes leader
- [ ] Party ID generated
- [ ] Initial state is IDLE
- [ ] Max size enforced

---

### 3.2 Unit Tests: Member Management

**File**: `tests/blackbox_party_members.py`

**Acceptance Criteria**:
- [ ] `add_member()` adds player
- [ ] `remove_member()` removes player
- [ ] Leader leaving promotes oldest
- [ ] Empty party disbanded
- [ ] Callbacks invoked

---

### 3.3 Unit Tests: Invitations

**File**: `tests/blackbox_party_invites.py`

**Acceptance Criteria**:
- [ ] Only leader can invite
- [ ] Invite expires after timeout
- [ ] Accept joins party
- [ ] Decline removes invite
- [ ] Accepting leaves current party

---

### 3.4 Unit Tests: Ready Check

**File**: `tests/blackbox_party_ready.py`

**Acceptance Criteria**:
- [ ] `set_ready()` updates member state
- [ ] `all_ready()` returns correct status
- [ ] Ready callback invoked when all ready
- [ ] New member resets all-ready

---

## 4. Lobby Tasks

### 4.1 Unit Tests: Lobby Creation

**File**: `tests/blackbox_lobby.py`

**Acceptance Criteria**:
- [ ] Creator becomes host
- [ ] Settings applied
- [ ] Initial state is WAITING
- [ ] Private lobby requires password

---

### 4.2 Unit Tests: Player Management

**File**: `tests/blackbox_lobby_players.py`

**Acceptance Criteria**:
- [ ] `add_player()` adds to lobby
- [ ] `remove_player()` removes from lobby
- [ ] Max players enforced
- [ ] Spectator limit enforced
- [ ] Team assignment works

---

### 4.3 Unit Tests: Countdown

**File**: `tests/blackbox_lobby_countdown.py`

**Acceptance Criteria**:
- [ ] `start_countdown()` changes state
- [ ] Countdown duration from settings
- [ ] `cancel_countdown()` returns to WAITING
- [ ] `update()` returns True when expired
- [ ] Auto-start when all ready

---

### 4.4 Unit Tests: Lobby Discovery

**File**: `tests/blackbox_lobby_discovery.py`

**Acceptance Criteria**:
- [ ] Private lobbies excluded
- [ ] Full lobbies excluded
- [ ] Game mode filter works
- [ ] Map filter works
- [ ] Slot filter works

---

### 4.5 Unit Tests: Password Protection

**File**: `tests/blackbox_lobby_password.py`

**Acceptance Criteria**:
- [ ] Wrong password rejected
- [ ] Correct password accepts
- [ ] Missing password rejected

---

## 5. Voice Chat Tasks

### 5.1 Unit Tests: Proximity Attenuation

**File**: `tests/blackbox_proximity_voice.py`

**Acceptance Criteria**:
- [ ] Distance 0: attenuation 1.0
- [ ] Distance < min: attenuation 1.0
- [ ] Distance > max: attenuation 0.0
- [ ] Falloff follows power law
- [ ] Occlusion callback applied

---

### 5.2 Unit Tests: Channel Management

**File**: `tests/blackbox_voice_channels.py`

**Acceptance Criteria**:
- [ ] Channel created with type
- [ ] Member added to channel
- [ ] Member removed from channel
- [ ] Multiple channels per player

---

### 5.3 Unit Tests: Mute/Deafen

**File**: `tests/blackbox_voice_mute.py`

**Acceptance Criteria**:
- [ ] Self-mute stops own audio
- [ ] Self-deafen stops hearing
- [ ] Per-player volume override
- [ ] Server mute enforced
- [ ] Server mute duration expires

---

### 5.4 Unit Tests: Push-to-Talk

**File**: `tests/blackbox_voice_ptt.py`

**Acceptance Criteria**:
- [ ] PTT mode requires activation
- [ ] Voice only when PTT active
- [ ] PTT state tracked per player

---

## 6. Text Chat Tasks

### 6.1 Unit Tests: Message Sending

**File**: `tests/blackbox_text_chat.py`

**Acceptance Criteria**:
- [ ] Message sent to channel
- [ ] Message stored in history
- [ ] Timestamp recorded
- [ ] Sender recorded

---

### 6.2 Unit Tests: Channel Scoping

**File**: `tests/blackbox_chat_channels.py`

**Acceptance Criteria**:
- [ ] TEAM message reaches team only
- [ ] PARTY message reaches party only
- [ ] WHISPER reaches recipient only
- [ ] GLOBAL reaches all

---

### 6.3 Unit Tests: Profanity Filter

**File**: `tests/blackbox_profanity_filter.py`

**Acceptance Criteria**:
- [ ] Blocked words replaced
- [ ] Regex patterns matched
- [ ] L33t speak detected
- [ ] Whitelist bypasses filter
- [ ] Original preserved in metadata

---

### 6.4 Unit Tests: Rate Limiting

**File**: `tests/blackbox_chat_rate_limit.py`

**Acceptance Criteria**:
- [ ] Messages within limit allowed
- [ ] Messages exceeding limit rejected
- [ ] Burst consumed first
- [ ] Cooldown enforced after burst

---

### 6.5 Unit Tests: Server Mute

**File**: `tests/blackbox_chat_mute.py`

**Acceptance Criteria**:
- [ ] Muted player cannot send
- [ ] Duration mute expires
- [ ] Per-player mute (local) works

---

## 7. Integration Tests

### 7.1 Matchmaking End-to-End

**File**: `tests/integration_matchmaking.py`

**Acceptance Criteria**:
- [ ] Players queue for match
- [ ] Compatible players matched
- [ ] Match callback invoked
- [ ] Players removed from queue
- [ ] Server allocated

---

### 7.2 Party-Lobby Integration

**File**: `tests/integration_party_lobby.py`

**Acceptance Criteria**:
- [ ] Party joins lobby together
- [ ] Party members on same team
- [ ] Party leader actions affect party

---

### 7.3 Matchmaking with Parties

**File**: `tests/integration_party_matchmaking.py`

**Acceptance Criteria**:
- [ ] Party queues as unit
- [ ] Party matched together
- [ ] Party skill averaged for matching

---

### 7.4 Chat in Lobby

**File**: `tests/integration_lobby_chat.py`

**Acceptance Criteria**:
- [ ] LOBBY channel scoped to lobby
- [ ] Messages reach lobby members
- [ ] Non-members don't receive

---

## 8. Gap Tasks

### 8.1 Gap: Persistence Layer

**File**: `engine/networking/social/persistence.py` (new)

**Background**: All data is in-memory only.

**Acceptance Criteria**:
- [ ] Skill ratings persisted
- [ ] Leaderboard persisted
- [ ] Ban records persisted
- [ ] Chat history optionally persisted
- [ ] Database abstraction layer

---

### 8.2 Gap: Voice Transport

**File**: `engine/networking/social/voice_transport.py` (new)

**Background**: Voice chat has no actual audio transport.

**Acceptance Criteria**:
- [ ] Audio codec integration (Opus)
- [ ] UDP voice packet format
- [ ] Jitter buffer
- [ ] Voice activity detection

---

### 8.3 Gap: Observability

**File**: `engine/networking/social/metrics.py` (new)

**Background**: No metrics instrumentation.

**Acceptance Criteria**:
- [ ] Queue time histogram
- [ ] Match formation rate
- [ ] Chat message rate
- [ ] Voice participant count
- [ ] Export to Prometheus/StatsD

---

### 8.4 Gap: Skill Rating Seasons

**File**: `engine/networking/social/skill_rating.py` (modify)

**Background**: No season/reset system.

**Acceptance Criteria**:
- [ ] Season start/end dates
- [ ] Rating soft reset between seasons
- [ ] Season rewards based on final rating
- [ ] Historical season data

---

## 9. Performance Tasks

### 9.1 Benchmark: Matchmaking Throughput

**File**: `benchmarks/matchmaking_throughput.py`

**Acceptance Criteria**:
- [ ] 1000 players queued: < 100ms find_match
- [ ] 10000 players queued: < 1s find_match
- [ ] Match formation: > 100 matches/second

---

### 9.2 Benchmark: Chat Processing

**File**: `benchmarks/chat_processing.py`

**Acceptance Criteria**:
- [ ] Profanity filter: > 50000 messages/second
- [ ] Rate limit check: > 100000 checks/second
- [ ] Message routing: > 10000 messages/second

---

### 9.3 Benchmark: Voice Attenuation

**File**: `benchmarks/voice_attenuation.py`

**Acceptance Criteria**:
- [ ] Proximity calculation: > 1000000/second
- [ ] 64 players, all audible: < 1ms per frame

---

## 10. Documentation Tasks

### 10.1 Matchmaking Configuration Guide

**Acceptance Criteria**:
- [ ] Skill range tuning
- [ ] Expansion rate optimization
- [ ] Queue time vs match quality tradeoff
- [ ] Server allocation integration

---

### 10.2 Skill Rating Guide

**Acceptance Criteria**:
- [ ] Elo vs Glicko-2 comparison
- [ ] K-factor selection
- [ ] Leaderboard configuration
- [ ] Season implementation

---

### 10.3 Party/Lobby Integration Guide

**Acceptance Criteria**:
- [ ] Party-lobby workflow
- [ ] Ready check implementation
- [ ] Countdown customization
- [ ] Private lobby setup

---

### 10.4 Voice Chat Setup Guide

**Acceptance Criteria**:
- [ ] Proximity parameters
- [ ] Channel type selection
- [ ] Moderation tools
- [ ] Quality level configuration

---

### 10.5 Text Chat Moderation Guide

**Acceptance Criteria**:
- [ ] Profanity list management
- [ ] Rate limit tuning
- [ ] Mute/ban workflow
- [ ] Appeals process
