# PHASE 5 ARCHITECTURE: Social and Matchmaking Systems

## Phase Overview

Phase 5 implements the social layer including matchmaking, skill rating, lobbies, parties, and communication systems. These components provide the player-facing multiplayer experience on top of the networking infrastructure.

---

## 1. Matchmaking System Architecture

### 1.1 Component Overview

```
Matchmaking System
    |
    +-- MatchmakingService (high-level coordinator)
    |       - Multiple queues by mode/region
    |       - Server allocation callback
    |       - Match found notification
    |
    +-- MatchmakingQueue (per-mode/region queue)
    |       - Skill-based matching
    |       - Dynamic range expansion
    |       - Estimated wait time
    |
    +-- QueueEntry (player in queue)
    |       - player_id, skill, criteria
    |       - queue_time, search_state
    |
    +-- MatchCriteria (search parameters)
            - game_mode, region
            - skill_min, skill_max
```

### 1.2 Queue Entry Lifecycle

```
Player requests match
        |
        v
MatchmakingService.enter_queue(player_id, criteria)
        |
        v
QueueEntry created (state: IDLE -> SEARCHING)
        |
        v
MatchmakingQueue.find_match() [each tick]
        |
        +-- Expand skill range based on wait time
        |
        +-- Find players within range
        |
        +-- If enough players: create match
        |
        v
Match found (state: SEARCHING -> FOUND)
        |
        v
on_match_found callback invoked
        |
        v
Server allocated (state: FOUND -> CONNECTING)
        |
        v
Players connect to server
```

### 1.3 Skill Range Expansion

```python
def _get_expanded_range(self, entry: QueueEntry) -> tuple[float, float]:
    """Expand skill range based on wait time to reduce queue times."""
    wait_time = time.time() - entry.queue_time
    expansion = (wait_time / self._expansion_interval) * self._expansion_rate
    
    return (
        entry.criteria.skill_min - expansion,
        entry.criteria.skill_max + expansion
    )
```

**Expansion Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| expansion_interval | 30s | Time between expansion steps |
| expansion_rate | 50 | Skill points per step |
| initial_range | 100 | Starting skill range |

### 1.4 Match Formation Algorithm

```python
def find_match(self) -> Optional[MatchResult]:
    """Find compatible players for a match."""
    entries = sorted(self._queue.values(), key=lambda e: e.queue_time)
    
    for entry in entries:
        expanded_min, expanded_max = self._get_expanded_range(entry)
        
        # Find compatible players using sliding window
        compatible = []
        for other in self._queue.values():
            if other.player_id == entry.player_id:
                continue
            if expanded_min <= other.skill <= expanded_max:
                # Check mutual compatibility
                other_min, other_max = self._get_expanded_range(other)
                if other_min <= entry.skill <= other_max:
                    compatible.append(other)
        
        if len(compatible) >= self._players_per_match - 1:
            # Form match with closest skill players
            compatible.sort(key=lambda e: abs(e.skill - entry.skill))
            match_players = [entry] + compatible[:self._players_per_match - 1]
            return self._create_match(match_players)
    
    return None
```

---

## 2. Skill Rating Architecture

### 2.1 Component Overview

```
Skill Rating System
    |
    +-- EloCalculator
    |       - Classic Elo with dynamic K-factors
    |       - Expected score: E = 1 / (1 + 10^((Rb-Ra)/400))
    |       - New rating: R' = R + K * (actual - expected)
    |
    +-- Glicko2Calculator
    |       - Rating + deviation (uncertainty)
    |       - Volatility tracking
    |       - Inactivity decay
    |
    +-- MMRManager (high-level API)
    |       - Player rating storage
    |       - Leaderboard generation
    |       - Percentile calculation
    |
    +-- SkillRating (player rating data)
            - rating, uncertainty, games_played
            - last_played timestamp
```

### 2.2 Elo Calculation

```python
class EloCalculator:
    def calculate_new_ratings(self, winner_rating: float, loser_rating: float, draw: bool = False) -> tuple[float, float]:
        # Expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loser = 1 - expected_winner
        
        # Actual scores
        if draw:
            actual_winner = 0.5
            actual_loser = 0.5
        else:
            actual_winner = 1.0
            actual_loser = 0.0
        
        # K-factors
        k_winner = self._get_k_factor(winner_rating, winner_games)
        k_loser = self._get_k_factor(loser_rating, loser_games)
        
        # New ratings
        new_winner = winner_rating + k_winner * (actual_winner - expected_winner)
        new_loser = loser_rating + k_loser * (actual_loser - expected_loser)
        
        # Floor enforcement
        new_winner = max(new_winner, MIN_RATING)
        new_loser = max(new_loser, MIN_RATING)
        
        return (new_winner, new_loser)
    
    def _get_k_factor(self, rating: float, games: int) -> int:
        if games < 30:
            return 40  # Provisional: faster adjustment
        elif rating >= 2400:
            return 16  # Expert: slower adjustment
        else:
            return 32  # Standard
```

### 2.3 Glicko-2 Calculation

```python
class Glicko2Calculator:
    # Scale factors (Glicko-2 internal scale)
    SCALE = 173.7178  # Converts between Elo and Glicko-2 scale
    
    def calculate_new_rating(self, player: Glicko2Rating, opponents: list[tuple[Glicko2Rating, float]]) -> Glicko2Rating:
        # Convert to Glicko-2 scale
        mu = (player.rating - 1500) / self.SCALE
        phi = player.rd / self.SCALE
        
        # Calculate new values using Glicko-2 algorithm
        # (See paper: http://www.glicko.net/glicko/glicko2.pdf)
        
        # Step 3: Compute variance
        v = self._compute_variance(mu, opponents)
        
        # Step 4: Compute delta
        delta = self._compute_delta(mu, opponents, v)
        
        # Step 5: Update volatility (simplified: fixed)
        sigma_new = player.volatility
        
        # Step 6: Update rating deviation
        phi_star = sqrt(phi**2 + sigma_new**2)
        phi_new = 1 / sqrt(1/phi_star**2 + 1/v)
        
        # Step 7: Update rating
        mu_new = mu + phi_new**2 * sum(
            g(op.rd) * (score - E(mu, op.mu, op.rd))
            for op, score in opponents
        )
        
        # Convert back to Elo scale
        return Glicko2Rating(
            rating=mu_new * self.SCALE + 1500,
            rd=phi_new * self.SCALE,
            volatility=sigma_new
        )
    
    def apply_inactivity_decay(self, rating: Glicko2Rating, days_inactive: int) -> Glicko2Rating:
        """Increase uncertainty for inactive players."""
        if days_inactive < 30:
            return rating
        
        # RD increases over time toward initial value
        new_rd = min(350, sqrt(rating.rd**2 + rating.volatility**2 * days_inactive))
        return Glicko2Rating(rating.rating, new_rd, rating.volatility)
```

### 2.4 Leaderboard

```python
class MMRManager:
    def get_leaderboard(self, limit: int = 100, min_games: int = 10) -> list[LeaderboardEntry]:
        """Get top players by rating."""
        eligible = [
            (player_id, rating)
            for player_id, rating in self._ratings.items()
            if rating.games_played >= min_games
        ]
        
        # Sort by rating descending
        eligible.sort(key=lambda x: -x[1].rating)
        
        return [
            LeaderboardEntry(rank=i+1, player_id=pid, rating=r)
            for i, (pid, r) in enumerate(eligible[:limit])
        ]
    
    def get_percentile(self, player_id: int) -> float:
        """Get player's percentile rank."""
        if player_id not in self._ratings:
            return 0.0
        
        player_rating = self._ratings[player_id].rating
        total = len(self._ratings)
        below = sum(1 for r in self._ratings.values() if r.rating < player_rating)
        
        return below / total * 100
```

---

## 3. Party System Architecture

### 3.1 Component Overview

```
Party System
    |
    +-- PartyManager (system-wide coordination)
    |       - Party creation/lookup
    |       - Member tracking
    |       - Event callbacks
    |
    +-- Party (individual party)
    |       - Members list
    |       - Leader tracking
    |       - State machine
    |       - Ready check
    |
    +-- PartyMember
    |       - player_id, role
    |       - ready_state
    |       - join_time
    |
    +-- PartyInvite
            - from_player, to_player
            - party_id, expires_at
```

### 3.2 Party State Machine

```
IDLE ──────────> SEARCHING ──────────> IN_LOBBY
  ^                  |                     |
  |                  |                     |
  |                  v                     v
  +──────────── DISBANDED <─────────── IN_GAME
```

### 3.3 Party Operations

```python
class Party:
    def add_member(self, player_id: int) -> bool:
        if len(self._members) >= self._max_size:
            return False
        
        member = PartyMember(
            player_id=player_id,
            role=PartyRole.MEMBER,
            ready_state=False,
            join_time=time.time()
        )
        self._members[player_id] = member
        
        if self._on_member_join:
            self._on_member_join(player_id, self)
        
        return True
    
    def remove_member(self, player_id: int) -> bool:
        if player_id not in self._members:
            return False
        
        member = self._members.pop(player_id)
        
        # Handle leader leaving
        if member.role == PartyRole.LEADER:
            self._promote_new_leader()
        
        if self._on_member_leave:
            self._on_member_leave(player_id, self)
        
        # Disband if empty
        if not self._members:
            self._state = PartyState.DISBANDED
            if self._on_party_disbanded:
                self._on_party_disbanded(self)
        
        return True
    
    def _promote_new_leader(self):
        """Promote oldest member to leader."""
        if not self._members:
            return
        
        oldest = min(self._members.values(), key=lambda m: m.join_time)
        oldest.role = PartyRole.LEADER
        self._leader_id = oldest.player_id
        
        if self._on_leader_change:
            self._on_leader_change(oldest.player_id, self)
```

### 3.4 Invitation System

```python
class PartyManager:
    def invite_to_party(self, party_id: str, inviter_id: int, invitee_id: int) -> bool:
        party = self._parties.get(party_id)
        if not party or party.leader_id != inviter_id:
            return False
        
        invite = PartyInvite(
            party_id=party_id,
            from_player=inviter_id,
            to_player=invitee_id,
            expires_at=time.time() + INVITE_EXPIRY_SECONDS
        )
        
        self._pending_invites[invitee_id].append(invite)
        return True
    
    def accept_invite(self, player_id: int, party_id: str) -> bool:
        invites = self._pending_invites.get(player_id, [])
        invite = next((i for i in invites if i.party_id == party_id), None)
        
        if not invite or time.time() > invite.expires_at:
            return False
        
        party = self._parties.get(party_id)
        if not party:
            return False
        
        # Leave current party if in one
        current_party = self._player_parties.get(player_id)
        if current_party:
            current_party.remove_member(player_id)
        
        # Join new party
        success = party.add_member(player_id)
        if success:
            self._player_parties[player_id] = party
            invites.remove(invite)
        
        return success
```

---

## 4. Lobby System Architecture

### 4.1 Component Overview

```
Lobby System
    |
    +-- LobbyManager (system-wide coordination)
    |       - Lobby creation/lookup
    |       - Discovery/search
    |       - Event callbacks
    |
    +-- Lobby (individual lobby)
    |       - Host and players
    |       - Settings
    |       - Countdown timer
    |       - State machine
    |
    +-- LobbySettings
    |       - max/min players
    |       - game_mode, map_name
    |       - private, password
    |       - auto_start
    |
    +-- LobbyPlayer
            - player_id, team
            - ready_state
            - is_spectator
```

### 4.2 Lobby State Machine

```
WAITING ────────> COUNTDOWN ────────> STARTING ────────> IN_GAME
    ^                 |                                      |
    |                 | (cancel)                             |
    +-----------------+                                      v
    ^                                                    CLOSED
    |                                                       |
    +-------------------------------------------------------+
```

### 4.3 Lobby Operations

```python
class Lobby:
    def set_ready(self, player_id: int, ready: bool) -> bool:
        player = self._players.get(player_id)
        if not player or player.is_spectator:
            return False
        
        player.ready_state = ready
        
        # Check auto-start condition
        if self._settings.auto_start and self._all_ready():
            self.start_countdown()
        
        return True
    
    def start_countdown(self) -> bool:
        if self._state != LobbyState.WAITING:
            return False
        
        if len(self._get_active_players()) < self._settings.min_players:
            return False
        
        self._state = LobbyState.COUNTDOWN
        self._countdown_start = time.time()
        self._countdown_duration = self._settings.countdown_duration
        
        return True
    
    def cancel_countdown(self) -> bool:
        if self._state != LobbyState.COUNTDOWN:
            return False
        
        self._state = LobbyState.WAITING
        return True
    
    def update(self) -> bool:
        """Called each tick. Returns True if game should start."""
        if self._state != LobbyState.COUNTDOWN:
            return False
        
        elapsed = time.time() - self._countdown_start
        if elapsed >= self._countdown_duration:
            self._state = LobbyState.STARTING
            return True
        
        return False
```

### 4.4 Lobby Discovery

```python
class LobbyManager:
    def search_lobbies(self, filters: LobbyFilters) -> list[LobbyInfo]:
        """Search for public lobbies matching filters."""
        results = []
        
        for lobby in self._lobbies.values():
            if lobby.settings.private:
                continue
            if lobby.state != LobbyState.WAITING:
                continue
            if lobby.is_full():
                continue
            
            # Apply filters
            if filters.game_mode and lobby.settings.game_mode != filters.game_mode:
                continue
            if filters.map_name and lobby.settings.map_name != filters.map_name:
                continue
            if filters.has_slots and lobby.available_slots < filters.has_slots:
                continue
            
            results.append(lobby.get_info())
        
        return results
    
    def join_lobby(self, lobby_id: str, player_id: int, password: Optional[str] = None) -> bool:
        lobby = self._lobbies.get(lobby_id)
        if not lobby:
            return False
        
        if lobby.settings.private:
            if not password or lobby.settings.password != password:
                return False
        
        return lobby.add_player(player_id)
```

---

## 5. Voice Chat Architecture

### 5.1 Component Overview

```
Voice Chat System
    |
    +-- VoiceChatManager (central coordination)
    |       - Channel management
    |       - Participant tracking
    |       - Volume/mute controls
    |
    +-- ProximityVoice (distance-based attenuation)
    |       - 3D position tracking
    |       - Falloff calculation
    |       - Occlusion support
    |
    +-- VoiceParticipant
    |       - player_id, position
    |       - state (mute, deafen, ptt)
    |
    +-- VoiceChannel
            - type (TEAM, SQUAD, PROXIMITY, GLOBAL, PRIVATE)
            - members
```

### 5.2 Channel Types

| Type | Description |
|------|-------------|
| TEAM | All players on same team |
| SQUAD | Party/squad members only |
| PROXIMITY | Distance-based (3D) |
| GLOBAL | Everyone in match |
| PRIVATE | Direct player-to-player |

### 5.3 Proximity Voice Attenuation

```python
class ProximityVoice:
    def __init__(self, max_distance: float = 50.0, min_distance: float = 1.0, falloff_exponent: float = 2.0):
        self._max_distance = max_distance
        self._min_distance = min_distance
        self._falloff_exponent = falloff_exponent
        self._occlusion_callback: Optional[Callable] = None
    
    def calculate_attenuation(self, listener_pos: Vector3, speaker_pos: Vector3) -> float:
        distance = euclidean_distance(listener_pos, speaker_pos)
        
        if distance <= self._min_distance:
            return 1.0
        
        if distance >= self._max_distance:
            return 0.0
        
        # Inverse power law falloff
        normalized = (distance - self._min_distance) / (self._max_distance - self._min_distance)
        attenuation = 1.0 - pow(normalized, 1.0 / self._falloff_exponent)
        
        # Apply occlusion if callback set
        if self._occlusion_callback:
            occlusion = self._occlusion_callback(listener_pos, speaker_pos)
            attenuation *= (1.0 - occlusion)
        
        return attenuation
```

### 5.4 Voice State Management

```python
class VoiceChatManager:
    def set_mute(self, player_id: int, muted: bool) -> bool:
        """Self-mute (player mutes their own microphone)."""
        participant = self._participants.get(player_id)
        if not participant:
            return False
        participant.state.muted = muted
        return True
    
    def set_deafen(self, player_id: int, deafened: bool) -> bool:
        """Self-deafen (player doesn't hear others)."""
        participant = self._participants.get(player_id)
        if not participant:
            return False
        participant.state.deafened = deafened
        return True
    
    def set_player_volume(self, listener_id: int, speaker_id: int, volume: float) -> bool:
        """Per-player volume override (0.0 - 2.0)."""
        participant = self._participants.get(listener_id)
        if not participant:
            return False
        volume = clamp(volume, 0.0, 2.0)
        participant.volume_overrides[speaker_id] = volume
        return True
    
    def server_mute(self, player_id: int, muted: bool, duration: Optional[float] = None) -> bool:
        """Server-enforced mute (moderation)."""
        participant = self._participants.get(player_id)
        if not participant:
            return False
        participant.state.server_muted = muted
        if duration:
            participant.server_mute_expires = time.time() + duration
        return True
```

---

## 6. Text Chat Architecture

### 6.1 Component Overview

```
Text Chat System
    |
    +-- ChatManager (central coordination)
    |       - Channel management
    |       - Message history
    |       - Rate limiting
    |
    +-- ProfanityFilter
    |       - Word blocking
    |       - Regex patterns
    |       - L33t speak detection
    |       - Whitelist support
    |
    +-- RateLimiter
    |       - Token bucket
    |       - Per-player tracking
    |
    +-- ChatMessage
            - sender, channel
            - content (original/filtered)
            - timestamp
```

### 6.2 Channel Types

| Type | Scope |
|------|-------|
| GLOBAL | All players in match |
| TEAM | Same team only |
| PARTY | Party members only |
| WHISPER | Direct message |
| SYSTEM | Server announcements |
| LOBBY | Lobby participants |
| MATCH | Current match participants |

### 6.3 Profanity Filter

```python
class ProfanityFilter:
    def __init__(self):
        self._blocked_words: set[str] = set()
        self._blocked_patterns: list[re.Pattern] = []
        self._whitelist: set[str] = set()
        self._leet_map = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a',
            '5': 's', '7': 't', '@': 'a', '$': 's'
        }
    
    def filter(self, text: str) -> tuple[str, bool]:
        """Filter text. Returns (filtered_text, was_modified)."""
        # Normalize for detection
        normalized = self._normalize(text)
        
        # Check against blocked words
        words = normalized.split()
        filtered_words = []
        modified = False
        
        for word in words:
            if word in self._whitelist:
                filtered_words.append(word)
            elif word in self._blocked_words or self._matches_pattern(word):
                filtered_words.append('*' * len(word))
                modified = True
            else:
                filtered_words.append(word)
        
        return (' '.join(filtered_words), modified)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for detection (lowercase, de-leet)."""
        result = text.lower()
        for leet, normal in self._leet_map.items():
            result = result.replace(leet, normal)
        return result
```

### 6.4 Rate Limiting

```python
class ChatRateLimiter:
    def __init__(self, rate: float = 2.0, burst: int = 5, cooldown: float = 5.0):
        self._buckets: dict[int, TokenBucket] = {}
        self._rate = rate
        self._burst = burst
        self._cooldown = cooldown
    
    def check(self, player_id: int) -> bool:
        bucket = self._buckets.get(player_id)
        if not bucket:
            bucket = TokenBucket(self._rate, self._burst)
            self._buckets[player_id] = bucket
        
        return bucket.consume()
    
    def get_cooldown_remaining(self, player_id: int) -> float:
        bucket = self._buckets.get(player_id)
        if not bucket or bucket.tokens > 0:
            return 0.0
        
        # Calculate time until next token
        return (1.0 - bucket.tokens) / self._rate
```

---

## 7. Configuration

**All social system constants centralized in `config.py`**:

```python
SOCIAL_CONFIG = SocialConfig(
    matchmaking=MatchmakingConfig(
        initial_skill_range=100,
        expansion_interval=30.0,
        expansion_rate=50,
        players_per_match=10,
        estimated_wait_multiplier=1.5,
    ),
    skill_rating=SkillRatingConfig(
        initial_rating=1500,
        min_rating=100,
        k_factor_new=40,
        k_factor_standard=32,
        k_factor_expert=16,
        expert_threshold=2400,
        games_for_full_k=30,
    ),
    lobby=LobbyConfig(
        max_players=16,
        min_players=2,
        countdown_duration=10.0,
        spectator_limit=4,
        invite_expiry=60.0,
    ),
    party=PartyConfig(
        max_size=10,
        invite_expiry=60.0,
    ),
    voice_chat=VoiceChatConfig(
        max_distance=50.0,
        min_distance=1.0,
        falloff_exponent=2.0,
        default_volume=1.0,
        quality_levels=['LOW', 'MEDIUM', 'HIGH', 'ULTRA'],
    ),
    text_chat=TextChatConfig(
        rate_limit=2.0,
        burst_limit=5,
        cooldown=5.0,
        history_size=100,
    ),
)
```

---

## 8. Thread Safety

All managers use `threading.Lock` for concurrent access:

```python
class PartyManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._parties: dict[str, Party] = {}
    
    def create_party(self, player_id: int) -> Party:
        with self._lock:
            party = Party(player_id)
            self._parties[party.id] = party
            return party
```

---

## 9. Integration Points

### Internal Dependencies

- All modules depend on `config.py` for constants
- Party and Lobby systems can integrate (party enters lobby together)
- Matchmaking can use MMRManager for skill values
- Voice and Text chat can be scoped to lobbies/parties

### External Integration Needed

| Component | External Integration |
|-----------|---------------------|
| Matchmaking | Server allocation callback |
| Voice Chat | Audio codec/transport |
| Text Chat | Network message delivery |
| Skill Rating | Persistence layer |
| All | Player authentication |
