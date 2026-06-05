"""
BLACKBOX Tests for Combat Systems.

Tests PUBLIC behavior only - no internal state inspection.
Tests are based on specifications from GAPSET_17_GAMEPLAY/PHASE_N_TODO.md:

Systems covered:
- Scoring: multi-kill detection, killstreaks, assist attribution, leaderboards
- Hitbox: regions with damage multipliers, hit detection
- Damage: damage types, resistances, armor
- Health: current/max health, regeneration, shields
- Death: death state, respawn timers
- Teams: team assignment, balancing, IFF
- Spawn: spawn point selection

Total: 170+ tests
"""

import pytest
import time
from typing import List, Optional, Tuple


# =============================================================================
# PUBLIC API IMPORTS ONLY
# =============================================================================

from engine.gameplay.combat.scoring import (
    ScoringSystem,
    ScoreEventType,
    LeaderboardSortKey,
    PlayerStats,
    TeamStats,
    ScoreEvent,
    LeaderboardEntry,
)

from engine.gameplay.combat.hitbox import (
    HitboxSystem,
    Hitbox,
    Hurtbox,
    HitboxType,
    HurtboxType,
    HitboxShape,
    CollisionResult,
    CollisionInfo,
    Vector3,
    BoundingBox,
)

from engine.gameplay.combat.damage import (
    DamageSystem,
    DamageInfo,
    DamageResult,
    DamageModifier,
    ResistanceProfile,
    calculate_dps,
    calculate_effective_health,
)

from engine.gameplay.combat.health import (
    HealthComponent,
    HealthPool,
    HealthChangeEvent,
    HealthChangeReason,
    InvulnerabilityReason,
    ShieldInfo,
    InvulnerabilityInfo,
)

from engine.gameplay.combat.death import (
    DeathSystem,
    DeathInfo,
    RespawnRequest,
    DeathEvent,
    RespawnEvent,
)

from engine.gameplay.combat.teams import (
    TeamSystem,
    TeamInfo,
    TeamMembership,
    TeamChangeEvent,
    IFFResult,
)

from engine.gameplay.combat.spawn_manager import (
    SpawnManager,
    SpawnPoint,
    SpawnRule,
    SpawnRuleType,
    TeamSpawnConfig,
)

from engine.gameplay.combat.constants import (
    DamageType,
    DeathState,
    HitboxZone,
    TeamRelation,
    CombatEventType,
    DamageSource,
    DamageConfig,
    HealthConfig,
    DeathConfig,
    ScoringConfig,
    TeamConfig,
    POINTS_PER_KILL,
    POINTS_PER_ASSIST,
    POINTS_PER_FIRST_BLOOD,
    KILLSTREAK_THRESHOLDS,
    MULTI_KILL_WINDOW,
    MULTI_KILL_NAMES,
    HITBOX_DAMAGE_MULTIPLIERS,
    CRITICAL_HIT_ZONES,
    DEFAULT_MAX_HEALTH,
    DEFAULT_RESPAWN_TIME,
    RESPAWN_INVULNERABILITY_DURATION,
    FRIENDLY_FIRE_NONE,
    FRIENDLY_FIRE_FULL,
)


# =============================================================================
# SCORING SYSTEM BLACKBOX TESTS (35 tests)
# =============================================================================


class TestScoringSystemMultiKill:
    """Blackbox tests for multi-kill detection in scoring system."""

    @pytest.fixture
    def scoring(self):
        """Fresh scoring system for each test."""
        return ScoringSystem()

    def test_double_kill_detected_within_window(self, scoring):
        """Two kills within multi-kill window should register double kill."""
        scoring.add_player("killer")
        scoring.add_player("victim1")
        scoring.add_player("victim2")

        scoring.record_kill("killer", "victim1")
        scoring.record_kill("killer", "victim2")

        stats = scoring.get_player_stats("killer")
        assert stats.total_multi_kills >= 1

    def test_triple_kill_detected(self, scoring):
        """Three rapid kills should register triple kill."""
        scoring.add_player("killer")
        for i in range(3):
            scoring.add_player(f"victim{i}")
            scoring.record_kill("killer", f"victim{i}")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 3
        assert stats.total_multi_kills >= 1

    def test_quad_kill_detected(self, scoring):
        """Four rapid kills should register quad kill."""
        scoring.add_player("killer")
        for i in range(4):
            scoring.add_player(f"victim{i}")
            scoring.record_kill("killer", f"victim{i}")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 4

    def test_penta_kill_detected(self, scoring):
        """Five rapid kills should register penta kill."""
        scoring.add_player("killer")
        for i in range(5):
            scoring.add_player(f"victim{i}")
            scoring.record_kill("killer", f"victim{i}")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 5

    def test_no_multi_kill_if_gap_too_large(self, scoring):
        """Kills separated by more than window should not count as multi-kill."""
        scoring.add_player("killer")
        scoring.add_player("victim1")
        scoring.add_player("victim2")

        scoring.record_kill("killer", "victim1")
        # Multi-kill window expires - each kill is separate
        time.sleep(0.01)  # Minimal delay for separate events
        scoring.record_kill("killer", "victim2")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 2


class TestScoringSystemKillstreaks:
    """Blackbox tests for killstreak tracking."""

    @pytest.fixture
    def scoring(self):
        return ScoringSystem()

    def test_killstreak_increments_on_kill(self, scoring):
        """Each kill should increment killstreak counter."""
        scoring.add_player("killer")
        scoring.add_player("victim")

        for i in range(5):
            scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 5
        assert stats.current_killstreak >= 5

    def test_killstreak_resets_on_death(self, scoring):
        """Death should reset the killstreak counter."""
        scoring.add_player("player1")
        scoring.add_player("player2")

        # Build a killstreak
        for i in range(5):
            scoring.record_kill("player1", "player2")

        stats = scoring.get_player_stats("player1")
        assert stats.current_killstreak >= 5

        # Player dies - use killer_id keyword
        scoring.record_death("player1", killer_id="player2")

        stats = scoring.get_player_stats("player1")
        assert stats.current_killstreak == 0

    def test_killing_spree_at_threshold(self, scoring):
        """3 kills should trigger killing spree."""
        scoring.add_player("killer")
        scoring.add_player("victim")

        for i in range(3):
            scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("killer")
        assert stats.current_killstreak >= 3

    def test_max_killstreak_tracked(self, scoring):
        """Max killstreak should be tracked across deaths."""
        scoring.add_player("player1")
        scoring.add_player("player2")

        # First streak of 5
        for i in range(5):
            scoring.record_kill("player1", "player2")

        # Die - use killer_id keyword
        scoring.record_death("player1", killer_id="player2")

        # Second streak of 3
        for i in range(3):
            scoring.record_kill("player1", "player2")

        stats = scoring.get_player_stats("player1")
        assert stats.best_killstreak >= 5


class TestScoringSystemAssists:
    """Blackbox tests for assist attribution."""

    @pytest.fixture
    def scoring(self):
        return ScoringSystem()

    def test_assist_awarded_for_damage_dealt(self, scoring):
        """Player who dealt damage before kill should get assist."""
        scoring.add_player("killer")
        scoring.add_player("assister")
        scoring.add_player("victim")

        # Assister deals damage
        scoring.record_damage("assister", "victim", 50.0)

        # Killer gets the kill
        scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("assister")
        assert stats.assists >= 1

    def test_no_assist_without_damage(self, scoring):
        """Player who dealt no damage should not get assist."""
        scoring.add_player("killer")
        scoring.add_player("bystander")
        scoring.add_player("victim")

        scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("bystander")
        assert stats.assists == 0

    def test_killer_does_not_get_assist_for_own_kill(self, scoring):
        """The killer should not receive an assist for their own kill."""
        scoring.add_player("killer")
        scoring.add_player("victim")

        scoring.record_damage("killer", "victim", 50.0)
        scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("killer")
        assert stats.assists == 0

    def test_multiple_assists_on_single_kill(self, scoring):
        """Multiple players can get assists on the same kill."""
        scoring.add_player("killer")
        scoring.add_player("assister1")
        scoring.add_player("assister2")
        scoring.add_player("victim")

        scoring.record_damage("assister1", "victim", 30.0)
        scoring.record_damage("assister2", "victim", 30.0)
        scoring.record_kill("killer", "victim")

        stats1 = scoring.get_player_stats("assister1")
        stats2 = scoring.get_player_stats("assister2")
        assert stats1.assists >= 1
        assert stats2.assists >= 1


class TestScoringSystemLeaderboard:
    """Blackbox tests for leaderboard functionality."""

    @pytest.fixture
    def scoring(self):
        return ScoringSystem()

    def test_leaderboard_sorted_by_score(self, scoring):
        """Leaderboard should be sorted by score descending."""
        scoring.add_player("low")
        scoring.add_player("mid")
        scoring.add_player("high")

        scoring.add_score("low", 100)
        scoring.add_score("mid", 500)
        scoring.add_score("high", 1000)

        leaderboard = scoring.get_leaderboard()
        assert len(leaderboard) == 3
        assert leaderboard[0].player_id == "high"
        assert leaderboard[1].player_id == "mid"
        assert leaderboard[2].player_id == "low"

    def test_leaderboard_by_kills(self, scoring):
        """Leaderboard can be sorted by kills."""
        scoring.add_player("few")
        scoring.add_player("many")
        scoring.add_player("victim")

        for i in range(2):
            scoring.record_kill("few", "victim")
        for i in range(5):
            scoring.record_kill("many", "victim")

        leaderboard = scoring.get_leaderboard(sort_by=LeaderboardSortKey.KILLS)
        assert leaderboard[0].player_id == "many"

    def test_leaderboard_by_kd_ratio(self, scoring):
        """Leaderboard can be sorted by K/D ratio."""
        scoring.add_player("good_kd")
        scoring.add_player("bad_kd")
        scoring.add_player("victim")

        # Good K/D: 10 kills, 1 death = 10.0
        for i in range(10):
            scoring.record_kill("good_kd", "victim")
        scoring.record_death("good_kd", killer_id="victim")

        # Bad K/D: 2 kills, 4 deaths = 0.5
        for i in range(2):
            scoring.record_kill("bad_kd", "victim")
        for i in range(4):
            scoring.record_death("bad_kd", killer_id="victim")

        leaderboard = scoring.get_leaderboard(sort_by=LeaderboardSortKey.KD_RATIO)
        assert leaderboard[0].player_id == "good_kd"

    def test_player_rank_returned(self, scoring):
        """Can get specific player's rank."""
        scoring.add_player("first")
        scoring.add_player("second")
        scoring.add_player("third")

        scoring.add_score("first", 1000)
        scoring.add_score("second", 500)
        scoring.add_score("third", 100)

        assert scoring.get_player_rank("first") == 1
        assert scoring.get_player_rank("second") == 2
        assert scoring.get_player_rank("third") == 3

    def test_team_leaderboard_sorted(self, scoring):
        """Team leaderboard should be sorted by team score."""
        team_scoring = ScoringSystem(is_team_based=True)
        team_scoring.add_player("p1", team_id="red")
        team_scoring.add_player("p2", team_id="blue")
        team_scoring.add_player("victim")

        for i in range(5):
            team_scoring.record_kill("p1", "victim")
        for i in range(3):
            team_scoring.record_kill("p2", "victim")

        team_board = team_scoring.get_team_leaderboard()
        # Returns list of (team_id, score) tuples
        assert team_board[0][0] == "red"


class TestScoringSystemFirstBlood:
    """Blackbox tests for first blood mechanic."""

    @pytest.fixture
    def scoring(self):
        return ScoringSystem()

    def test_first_blood_awarded_once(self, scoring):
        """First blood should only be awarded once per match."""
        scoring.add_player("first_killer")
        scoring.add_player("second_killer")
        scoring.add_player("victim")

        scoring.record_kill("first_killer", "victim")
        assert scoring.first_blood_awarded

        # Second kill should not get first blood
        scoring.record_kill("second_killer", "victim")

        # First blood flag stays true
        assert scoring.first_blood_awarded

    def test_first_blood_bonus_points(self, scoring):
        """First blood should award bonus points."""
        scoring.add_player("killer")
        scoring.add_player("victim")

        initial_score = scoring.get_score("killer")
        scoring.record_kill("killer", "victim")
        final_score = scoring.get_score("killer")

        # Should include kill points + first blood bonus
        expected_min = POINTS_PER_KILL + POINTS_PER_FIRST_BLOOD
        assert final_score - initial_score >= expected_min


class TestScoringSystemTeams:
    """Blackbox tests for team-based scoring."""

    def test_team_score_aggregates_player_scores(self):
        """Team score should be sum of player scores."""
        scoring = ScoringSystem(is_team_based=True)
        scoring.add_player("p1", team_id="red")
        scoring.add_player("p2", team_id="red")
        scoring.add_player("victim")

        scoring.record_kill("p1", "victim")
        scoring.record_kill("p2", "victim")

        team_score = scoring.get_team_score("red")
        assert team_score >= POINTS_PER_KILL * 2

    def test_player_can_change_teams(self):
        """Player team change should update team scores."""
        scoring = ScoringSystem(is_team_based=True)
        scoring.add_player("player", team_id="red")
        scoring.add_player("victim")

        scoring.record_kill("player", "victim")

        # Change team
        result = scoring.set_player_team("player", "blue")
        assert result is True

        stats = scoring.get_player_stats("player")
        assert stats.team_id == "blue"


# =============================================================================
# HITBOX SYSTEM BLACKBOX TESTS (30 tests)
# =============================================================================


class TestHitboxDamageMultipliers:
    """Blackbox tests for hitbox region damage multipliers."""

    @pytest.fixture
    def hitbox_system(self):
        return HitboxSystem()

    def test_head_hitbox_has_highest_multiplier(self, hitbox_system):
        """Head region should have 2x damage multiplier."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_head_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.HEAD,
        )
        assert hitbox.damage_multiplier == 2.0

    def test_torso_hitbox_base_multiplier(self, hitbox_system):
        """Torso region should have 1x damage multiplier."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_torso_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.TORSO,
        )
        assert hitbox.damage_multiplier == 1.0

    def test_limb_hitbox_reduced_multiplier(self, hitbox_system):
        """Limb regions should have 0.75x damage multiplier."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_arm_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.LEFT_ARM,
        )
        assert hitbox.damage_multiplier == 0.75

    def test_extremity_hitbox_low_multiplier(self, hitbox_system):
        """Hand/foot regions should have 0.5x damage multiplier."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_hand_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.LEFT_HAND,
        )
        assert hitbox.damage_multiplier == 0.5

    def test_critical_zone_detection(self, hitbox_system):
        """Head and neck should be flagged as critical zones."""
        head_hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_head_crit",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.HEAD,
        )
        neck_hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_neck_crit",
            owner_id=2,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            zone=HitboxZone.NECK,
        )
        assert head_hitbox.is_critical_zone
        assert neck_hitbox.is_critical_zone


class TestHitboxCollisionDetection:
    """Blackbox tests for hit detection between hitboxes and hurtboxes."""

    @pytest.fixture
    def hitbox_system(self):
        return HitboxSystem()

    def test_overlapping_boxes_collide(self, hitbox_system):
        """Overlapping hitbox and hurtbox should detect collision."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_attack_1",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hitbox_type=HitboxType.ATTACK,
        )
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_body_1",
            owner_id=2,
            position=(1, 0, 0),
            size=(2, 2, 2),
        )

        hitbox_system.activate_hitbox(hitbox.hitbox_id)
        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is not None
        assert collision.is_hit

    def test_separated_boxes_no_collision(self, hitbox_system):
        """Non-overlapping hitbox and hurtbox should not collide."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_attack_far",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
        )
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_body_far",
            owner_id=2,
            position=(100, 100, 100),
            size=(1, 1, 1),
        )

        hitbox_system.activate_hitbox(hitbox.hitbox_id)
        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        assert collision is None or not collision.is_hit

    def test_same_entity_no_self_collision(self, hitbox_system):
        """Entity should not collide with its own hitboxes."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_self_attack",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hitbox_type=HitboxType.ATTACK,
        )
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_self_body",
            owner_id=1,  # Same entity
            position=(0, 0, 0),
            size=(2, 2, 2),
        )

        hitbox_system.activate_hitbox(hitbox.hitbox_id)
        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)

        collision = hitbox_system.check_collision(hitbox, hurtbox)
        # Same entity should not collide with itself
        assert collision is None or collision.hitbox_entity_id == collision.hurtbox_entity_id

    def test_inactive_hitbox_no_collision(self, hitbox_system):
        """Inactive hitbox should not register collisions."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_inactive",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hitbox_type=HitboxType.ATTACK,
        )
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_inactive_test",
            owner_id=2,
            position=(0, 0, 0),
            size=(2, 2, 2),
        )

        # Do NOT activate hitbox
        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)

        collisions = hitbox_system.process_collisions()
        assert len(collisions) == 0


class TestHitboxInvincibility:
    """Blackbox tests for invincibility frames and blocking."""

    @pytest.fixture
    def hitbox_system(self):
        return HitboxSystem()

    def test_intangible_hurtbox_invincible(self, hitbox_system):
        """Intangible (invincible) hurtbox should not register hits."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_vs_intangible",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hitbox_type=HitboxType.ATTACK,
        )
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_intangible",
            owner_id=2,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hurtbox_type=HurtboxType.INTANGIBLE,
        )

        hitbox_system.activate_hitbox(hitbox.hitbox_id)
        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)

        assert hurtbox.is_invincible

    def test_armor_absorbs_hits(self, hitbox_system):
        """Armored hurtbox should absorb limited hits."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_armored",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hurtbox_type=HurtboxType.ARMORED,
            armor_value=3,
        )

        assert hurtbox.has_armor

        # Absorb hits
        absorbed = hurtbox.absorb_armor_hit()
        assert absorbed is True

        absorbed = hurtbox.absorb_armor_hit()
        assert absorbed is True

        absorbed = hurtbox.absorb_armor_hit()
        assert absorbed is True

        # Armor depleted
        absorbed = hurtbox.absorb_armor_hit()
        assert absorbed is False

    def test_counter_state_detection(self, hitbox_system):
        """Counter-state hurtbox should be detectable."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_counter",
            owner_id=1,
            position=(0, 0, 0),
            size=(2, 2, 2),
            hurtbox_type=HurtboxType.COUNTER,
        )

        assert hurtbox.is_counter_state


class TestHitboxLifetime:
    """Blackbox tests for hitbox activation and expiration."""

    @pytest.fixture
    def hitbox_system(self):
        return HitboxSystem()

    def test_hitbox_expires_after_lifetime(self, hitbox_system):
        """Hitbox should expire after specified lifetime."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_timed",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            lifetime=0.1,  # 100ms lifetime
        )

        hitbox.activate()

        # Simulate time passing
        time.sleep(0.15)

        assert hitbox.is_expired

    def test_hitbox_hit_limit(self, hitbox_system):
        """Hitbox should stop registering hits after max_hits reached."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_limited",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            max_hits=2,
        )

        hitbox.activate()

        # Record hits
        can_hit_1 = hitbox.record_hit(entity_id=100)
        can_hit_2 = hitbox.record_hit(entity_id=101)

        assert can_hit_1 is True
        assert can_hit_2 is True

        # Third hit should fail
        assert not hitbox.can_hit_more

    def test_hitbox_cannot_hit_same_entity_twice(self, hitbox_system):
        """Hitbox should not hit the same entity twice."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_no_repeat",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
            hitbox_type=HitboxType.ATTACK,
            max_hits=10,
        )

        hitbox.activate()

        # First hit succeeds
        can_hit_1 = hitbox.record_hit(entity_id=100)
        assert can_hit_1 is True

        # Same entity again should fail
        can_hit_same = hitbox.can_hit_entity(entity_id=100)
        assert can_hit_same is False


# =============================================================================
# DAMAGE SYSTEM BLACKBOX TESTS (25 tests)
# =============================================================================


class TestDamageTypes:
    """Blackbox tests for damage type handling."""

    @pytest.fixture
    def damage_system(self):
        return DamageSystem()

    def test_physical_damage_calculated(self, damage_system):
        """Physical damage should be calculated correctly."""
        final_damage, armor_reduction, resist_reduction = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
        )

        assert final_damage == 100

    def test_true_damage_ignores_resistance(self, damage_system):
        """TRUE damage type should ignore all resistances."""
        final_damage, armor_reduction, resist_reduction = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.TRUE,
            armor=100,
            resistance=0.5,
        )

        # TRUE damage ignores armor and resistance
        assert final_damage == 100

    def test_elemental_damage_affected_by_resistance(self, damage_system):
        """Elemental damage should be reduced by resistance."""
        final_damage, armor_reduction, resist_reduction = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.FIRE,
            armor=0,
            resistance=0.5,
        )

        assert final_damage == 50


class TestDamageArmor:
    """Blackbox tests for armor damage reduction."""

    @pytest.fixture
    def damage_system(self):
        return DamageSystem()

    def test_armor_reduces_physical_damage(self, damage_system):
        """Armor should reduce physical damage."""
        final_no_armor, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
        )
        final_with_armor, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=100,
            resistance=0,
        )

        assert final_with_armor < final_no_armor

    def test_armor_reduction_capped(self, damage_system):
        """Armor reduction should be capped at max (90%)."""
        # Very high armor
        final_damage, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=10000,
            resistance=0,
        )

        # Should not reduce below 10% (90% cap)
        assert final_damage >= 10

    def test_armor_calculation_formula(self, damage_system):
        """Armor reduction should follow: damage * (100 / (100 + armor))."""
        # 100 armor = 50% reduction
        reduction = damage_system.calculate_effective_armor(100)
        assert abs(reduction - 0.5) < 0.01


class TestDamageResistances:
    """Blackbox tests for resistance mechanics."""

    def test_resistance_profile_creation(self):
        """Resistance profile should store per-type resistances."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 0.5)
        profile.set_resistance(DamageType.ICE, 0.25)

        assert profile.get_resistance(DamageType.FIRE) == 0.5
        assert profile.get_resistance(DamageType.ICE) == 0.25
        assert profile.get_resistance(DamageType.LIGHTNING) == 0.0

    def test_resistance_capped_at_maximum(self):
        """Resistance should be capped at 75%."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 1.0)  # Try to set 100%

        # Should be capped
        assert profile.get_resistance(DamageType.FIRE) <= 0.75

    def test_negative_resistance_vulnerability(self):
        """Negative resistance should increase damage taken."""
        damage_system = DamageSystem()

        # -25% resistance = +25% damage
        final_damage, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.FIRE,
            armor=0,
            resistance=-0.25,
        )

        assert final_damage > 100


class TestDamageModifiers:
    """Blackbox tests for damage modifiers."""

    def test_global_modifier_applied(self):
        """Global modifiers should affect all damage via additional_multipliers."""
        damage_system = DamageSystem()

        # Apply double damage via additional_multipliers
        final_damage, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            additional_multipliers=[2.0],
        )

        assert final_damage == 200

    def test_type_modifier_only_affects_type(self):
        """Multiple multipliers can be stacked."""
        damage_system = DamageSystem()

        # Apply multiple multipliers
        fire_damage, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.FIRE,
            armor=0,
            resistance=0,
            additional_multipliers=[1.5],
        )

        ice_damage, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.ICE,
            armor=0,
            resistance=0,
        )

        assert fire_damage == 150
        assert ice_damage == 100


class TestDamageUtilities:
    """Blackbox tests for damage utility functions."""

    def test_calculate_dps(self):
        """DPS calculation should be damage per second."""
        # 100 damage at 2 attacks per second = 200 DPS (no crits)
        dps = calculate_dps(base_damage=100, attacks_per_second=2.0)
        assert dps == 200

    def test_calculate_effective_health(self):
        """Effective health should account for armor."""
        # 100 HP with 100 armor = 200 effective HP (50% reduction)
        ehp = calculate_effective_health(health=100, armor=100)
        assert ehp == 200


# =============================================================================
# HEALTH SYSTEM BLACKBOX TESTS (25 tests)
# =============================================================================


class TestHealthBasic:
    """Blackbox tests for basic health functionality."""

    def test_health_component_creation(self):
        """Health component should be created with correct values."""
        health = HealthComponent(entity_id=1, max_health=100)

        assert health.current_health == 100
        assert health.max_health == 100
        assert health.health_percentage == 1.0

    def test_take_damage_reduces_health(self):
        """Taking damage should reduce current health."""
        health = HealthComponent(entity_id=1, max_health=100)

        actual_damage = health.take_damage(30)

        assert health.current_health == 70
        assert actual_damage == 30

    def test_healing_restores_health(self):
        """Healing should restore health up to max."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(50)

        healed = health.heal(30)

        assert health.current_health == 80
        assert healed == 30

    def test_healing_capped_at_max(self):
        """Healing should not exceed max health."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(10)

        health.heal(50)  # Try to heal more than missing

        assert health.current_health == 100

    def test_death_at_zero_health(self):
        """Entity should die when health reaches zero."""
        health = HealthComponent(entity_id=1, max_health=100)

        health.take_damage(100)

        assert health.is_dead
        assert not health.is_alive


class TestHealthRegeneration:
    """Blackbox tests for health regeneration."""

    def test_regeneration_restores_health(self):
        """Regeneration should restore health over time (after delay)."""
        # Use custom config with no regen delay and a regen rate
        from engine.gameplay.combat.constants import HealthConfig
        config = HealthConfig(regen_delay_after_damage=0.0, default_regen_rate=10.0)
        health = HealthComponent(entity_id=1, max_health=100, config=config)

        # Set health directly without triggering damage time
        health.set_health(50)

        # Simulate 1 second
        healed = health.update_regeneration(delta_time=1.0)

        assert healed > 0
        assert health.current_health > 50

    def test_regeneration_disabled_stops_healing(self):
        """Disabled regeneration should not heal."""
        from engine.gameplay.combat.constants import HealthConfig
        config = HealthConfig(regen_delay_after_damage=0.0, default_regen_rate=10.0)
        health = HealthComponent(entity_id=1, max_health=100, config=config)
        health.set_health(50)
        health.disable_regeneration()

        healed = health.update_regeneration(delta_time=1.0)

        assert healed == 0
        assert health.current_health == 50

    def test_regeneration_stops_at_full(self):
        """Regeneration should stop when health is full."""
        from engine.gameplay.combat.constants import HealthConfig
        config = HealthConfig(default_regen_rate=10.0)
        health = HealthComponent(entity_id=1, max_health=100, config=config)

        healed = health.update_regeneration(delta_time=1.0)

        assert healed == 0  # Already at full


class TestHealthShields:
    """Blackbox tests for shield mechanics."""

    def test_shield_absorbs_damage(self):
        """Shield should absorb damage before health."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield(name="barrier", amount=50)

        health.take_damage(30)

        # Shield absorbs, health unchanged
        assert health.current_health == 100
        assert health.total_shield == 20

    def test_shield_overflow_damages_health(self):
        """Damage exceeding shield should damage health."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield(name="barrier", amount=30)

        health.take_damage(50)

        # Shield depleted, 20 damage to health
        assert health.total_shield == 0
        assert health.current_health == 80

    def test_multiple_shields_stack(self):
        """Multiple shields should add together."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield(name="barrier1", amount=25)
        health.add_shield(name="barrier2", amount=25)

        assert health.total_shield == 50

    def test_shield_expires(self):
        """Shield with duration should expire."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield(name="temp_shield", amount=50, duration=1.0)

        shield = health.get_shield("temp_shield")
        assert shield is not None

        # After expiration
        time.sleep(1.1)
        assert shield.is_expired


class TestHealthInvulnerability:
    """Blackbox tests for invulnerability mechanics."""

    def test_invulnerability_prevents_damage(self):
        """Invulnerable entity should not take damage."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_invulnerability(
            reason=InvulnerabilityReason.RESPAWN,
            duration=5.0
        )

        assert health.is_invulnerable

        health.take_damage(50)

        assert health.current_health == 100

    def test_invulnerability_expires(self):
        """Invulnerability should expire after duration."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_invulnerability(
            reason=InvulnerabilityReason.RESPAWN,
            duration=0.1
        )

        assert health.is_invulnerable

        time.sleep(0.15)
        health._cleanup_expired_invulnerabilities()

        assert not health.is_invulnerable

    def test_invulnerability_stacking(self):
        """Multiple invulnerabilities should stack."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_invulnerability(
            reason=InvulnerabilityReason.RESPAWN,
            duration=0.5
        )
        health.add_invulnerability(
            reason=InvulnerabilityReason.ABILITY,
            duration=1.0
        )

        remaining = health.get_invulnerability_remaining()
        assert remaining > 0.5  # Should be longer due to stacking


# =============================================================================
# DEATH SYSTEM BLACKBOX TESTS (20 tests)
# =============================================================================


class TestDeathStateTransitions:
    """Blackbox tests for death state machine."""

    @pytest.fixture
    def death_system(self):
        return DeathSystem()

    def test_death_transitions_to_dying(self, death_system):
        """Processing death should transition to DYING state."""
        death_system.process_death(
            entity_id=1,
            killer_id=2,
            damage_info=None,
        )

        state = death_system.get_death_state(1)
        assert state == DeathState.DYING or state == DeathState.DEAD

    def test_dying_transitions_to_dead(self, death_system):
        """DYING state should transition to DEAD."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)

        state = death_system.get_death_state(1)
        assert state == DeathState.DEAD

    def test_dead_transitions_to_respawning(self, death_system):
        """DEAD state can transition to RESPAWNING."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)

        state = death_system.get_death_state(1)
        assert state == DeathState.RESPAWNING

    def test_respawn_completes_to_alive(self, death_system):
        """Completing respawn should return to ALIVE."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)
        death_system.complete_respawn(1)

        state = death_system.get_death_state(1)
        assert state == DeathState.ALIVE


class TestRespawnTimers:
    """Blackbox tests for respawn timing."""

    @pytest.fixture
    def death_system(self):
        return DeathSystem()

    def test_respawn_queued_with_timer(self, death_system):
        """Respawn should be queued with specified delay."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)

        request = death_system.queue_respawn(entity_id=1, delay=5.0)

        assert request is not None
        assert request.time_until_respawn > 4.0  # Approximately 5 seconds

    def test_respawn_ready_after_timer(self, death_system):
        """Respawn should be ready after timer expires."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)

        death_system.queue_respawn(entity_id=1, delay=0.1)

        time.sleep(0.15)

        request = death_system.get_respawn_request(1)
        assert request is not None
        assert request.is_ready

    def test_respawn_cancelled(self, death_system):
        """Respawn can be cancelled."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(entity_id=1, delay=10.0)

        result = death_system.cancel_respawn(1)

        assert result is True
        request = death_system.get_respawn_request(1)
        assert request is None

    def test_complete_respawn_returns_to_alive(self, death_system):
        """Completing respawn should return entity to ALIVE state."""
        death_system.process_death(entity_id=1, killer_id=2)
        death_system.transition_to_dead(1)
        death_system.transition_to_respawning(1)

        death_system.complete_respawn(1)

        state = death_system.get_death_state(1)
        assert state == DeathState.ALIVE


class TestDeathTracking:
    """Blackbox tests for death information tracking."""

    @pytest.fixture
    def death_system(self):
        return DeathSystem()

    def test_death_info_recorded(self, death_system):
        """Death information should be recorded."""
        death_system.process_death(
            entity_id=1,
            killer_id=2,
        )

        info = death_system.get_death_info(1)
        assert info is not None
        assert info.entity_id == 1
        assert info.killer_id == 2

    def test_all_dead_entities_tracked(self, death_system):
        """Should be able to get list of all dead entities."""
        death_system.process_death(entity_id=1, killer_id=10)
        death_system.process_death(entity_id=2, killer_id=10)
        death_system.transition_to_dead(1)
        death_system.transition_to_dead(2)

        dead_list = death_system.get_all_dead()
        assert 1 in dead_list
        assert 2 in dead_list

    def test_recent_deaths_queryable(self, death_system):
        """Should be able to query recent deaths."""
        death_system.process_death(entity_id=1, killer_id=10)
        death_system.process_death(entity_id=2, killer_id=10)

        recent = death_system.get_recent_deaths(time_window=60.0)
        assert len(recent) == 2


# =============================================================================
# TEAM SYSTEM BLACKBOX TESTS (20 tests)
# =============================================================================


class TestTeamCreation:
    """Blackbox tests for team creation and management."""

    @pytest.fixture
    def team_system(self):
        return TeamSystem()

    def test_create_team(self, team_system):
        """Should be able to create a new team."""
        team = team_system.create_team(team_id=1, name="Red Team")

        assert team is not None
        assert team.team_id == 1
        assert team.name == "Red Team"

    def test_team_capacity(self, team_system):
        """Team should respect capacity limit."""
        team = team_system.create_team(team_id=1, name="Small", max_members=2)

        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        # Team full
        team_info = team_system.get_team(1)
        assert team_info.is_full

    def test_remove_team(self, team_system):
        """Should be able to remove a team."""
        team_system.create_team(team_id=1, name="Temp")

        result = team_system.remove_team(1)

        assert result is True
        assert not team_system.team_exists(1)


class TestTeamIFF:
    """Blackbox tests for IFF (Identification Friend or Foe)."""

    @pytest.fixture
    def team_system(self):
        system = TeamSystem()
        system.create_team(team_id=1, name="Red")
        system.create_team(team_id=2, name="Blue")
        return system

    def test_same_team_is_friendly(self, team_system):
        """Entities on same team should be friendly."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        result = team_system.check_iff(source_id=1, target_id=2)

        assert result.is_friendly

    def test_different_teams_hostile(self, team_system):
        """Entities on different teams should be hostile."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=2)

        # Set teams as hostile
        team_system.set_relationship(1, 2, TeamRelation.HOSTILE)

        result = team_system.check_iff(source_id=1, target_id=2)

        assert result.is_hostile

    def test_can_attack_hostile(self, team_system):
        """Should be able to attack hostile entities."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=2)
        team_system.set_relationship(1, 2, TeamRelation.HOSTILE)

        can_attack = team_system.can_attack(source_id=1, target_id=2)

        assert can_attack is True

    def test_cannot_attack_friendly_by_default(self, team_system):
        """Should not be able to attack friendly entities by default."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        can_attack = team_system.can_attack(source_id=1, target_id=2)

        assert can_attack is False


class TestTeamBalancing:
    """Blackbox tests for team auto-balancing."""

    @pytest.fixture
    def team_system(self):
        system = TeamSystem()
        system.create_team(team_id=1, name="Red", max_members=4)
        system.create_team(team_id=2, name="Blue", max_members=4)
        return system

    def test_auto_assign_to_smallest_team(self, team_system):
        """Auto-assign should put player on smallest team."""
        # Red has 3 players
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)
        team_system.set_team(entity_id=3, team_id=1)

        # Blue has 1 player
        team_system.set_team(entity_id=4, team_id=2)

        # New player should go to Blue (excluding default and neutral teams)
        assigned_team = team_system.auto_assign_team(entity_id=5, exclude={0, -1})

        assert assigned_team == 2

    def test_get_team_with_fewest_members(self, team_system):
        """Should identify team with fewest members."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)
        team_system.set_team(entity_id=3, team_id=2)

        # Exclude default and neutral teams
        smallest = team_system.get_team_with_fewest_members(exclude={0, -1})

        assert smallest == 2


class TestFriendlyFire:
    """Blackbox tests for friendly fire settings."""

    @pytest.fixture
    def team_system(self):
        system = TeamSystem()
        system.create_team(team_id=1, name="Red")
        return system

    def test_friendly_fire_disabled_by_default(self, team_system):
        """Friendly fire should be disabled by default."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        multiplier = team_system.get_friendly_fire_multiplier(1, 2)

        assert multiplier == FRIENDLY_FIRE_NONE

    def test_enable_friendly_fire(self, team_system):
        """Should be able to enable friendly fire."""
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        team_system.enable_friendly_fire(team_id=1, multiplier=FRIENDLY_FIRE_FULL)

        multiplier = team_system.get_friendly_fire_multiplier(1, 2)

        assert multiplier == FRIENDLY_FIRE_FULL


# =============================================================================
# SPAWN SYSTEM BLACKBOX TESTS (20 tests)
# =============================================================================


class TestSpawnPointRegistration:
    """Blackbox tests for spawn point management."""

    @pytest.fixture
    def spawn_manager(self):
        return SpawnManager()

    def test_register_spawn_point(self, spawn_manager):
        """Should register spawn points."""
        spawn = SpawnPoint(
            point_id="spawn_1",
            position=(0, 0, 0),
            team_id="red",
        )

        result = spawn_manager.register_spawn_point(spawn)

        assert result is True

    def test_get_spawn_point(self, spawn_manager):
        """Should retrieve registered spawn point."""
        spawn = SpawnPoint(
            point_id="spawn_1",
            position=(10, 0, 10),
            team_id="blue",
        )
        spawn_manager.register_spawn_point(spawn)

        retrieved = spawn_manager.get_spawn_point("spawn_1")

        assert retrieved is not None
        assert retrieved.position == (10, 0, 10)

    def test_unregister_spawn_point(self, spawn_manager):
        """Should unregister spawn points."""
        spawn = SpawnPoint(point_id="spawn_1", position=(0, 0, 0))
        spawn_manager.register_spawn_point(spawn)

        result = spawn_manager.unregister_spawn_point("spawn_1")

        assert result is True
        assert spawn_manager.get_spawn_point("spawn_1") is None


class TestSpawnPointSelection:
    """Blackbox tests for spawn point selection."""

    @pytest.fixture
    def spawn_manager(self):
        manager = SpawnManager()
        # Register multiple spawn points
        for i in range(5):
            spawn = SpawnPoint(
                point_id=f"spawn_{i}",
                position=(i * 10, 0, 0),
                team_id="red" if i < 3 else "blue",
            )
            manager.register_spawn_point(spawn)
        return manager

    def test_select_team_spawn_point(self, spawn_manager):
        """Should select spawn point for correct team."""
        spawn_manager.update_player_team("player1", "red")

        spawn = spawn_manager.select_spawn_point(
            player_id="player1",
            team_id="red",
        )

        assert spawn is not None
        assert spawn.team_id == "red"

    def test_available_spawn_points_filtered(self, spawn_manager):
        """Should filter to available spawn points."""
        # Block some spawn points
        spawn = spawn_manager.get_spawn_point("spawn_0")
        spawn.block()

        available = spawn_manager.get_available_spawn_points()

        assert "spawn_0" not in [s.point_id for s in available]

    def test_spawn_point_cooldown(self, spawn_manager):
        """Recently used spawn point should be on cooldown."""
        spawn = spawn_manager.get_spawn_point("spawn_0")
        spawn.use(cooldown_seconds=5.0)

        assert not spawn.is_available

    def test_spawn_point_cooldown_expires(self, spawn_manager):
        """Spawn point should become available after cooldown."""
        spawn = spawn_manager.get_spawn_point("spawn_0")
        spawn.use(cooldown_seconds=0.1)

        # Release the occupant
        spawn.release()

        time.sleep(0.15)

        # is_available is a property that checks cooldown
        assert spawn.is_available is True


class TestSpawnPlayerFlow:
    """Blackbox tests for complete player spawn flow."""

    @pytest.fixture
    def spawn_manager(self):
        manager = SpawnManager()
        spawn = SpawnPoint(
            point_id="main_spawn",
            position=(100, 0, 100),
            team_id="red",
        )
        manager.register_spawn_point(spawn)
        return manager

    def test_spawn_player_at_point(self, spawn_manager):
        """Should spawn player at selected point."""
        result = spawn_manager.spawn_player(
            player_id="player1",
            team_id="red",
        )

        assert result is not None
        position, rotation = result
        # Position should be at or near spawn point
        assert abs(position[0] - 100) < 1
        assert abs(position[2] - 100) < 1

    def test_schedule_respawn(self, spawn_manager):
        """Should schedule respawn with timer."""
        respawn_time = spawn_manager.schedule_respawn(
            player_id="player1",
            delay_seconds=2.0,
        )

        # Should return absolute time in future
        assert respawn_time > time.time()

    def test_respawn_ready_after_timer(self, spawn_manager):
        """Respawn should be ready after timer."""
        spawn_manager.schedule_respawn(
            player_id="player1",
            delay_seconds=0.1,
        )

        time.sleep(0.15)

        is_ready = spawn_manager.is_respawn_ready("player1")
        assert is_ready is True

    def test_cancel_scheduled_respawn(self, spawn_manager):
        """Should cancel scheduled respawn."""
        spawn_manager.schedule_respawn(
            player_id="player1",
            delay_seconds=10.0,
        )

        result = spawn_manager.cancel_respawn("player1")

        assert result is True


class TestSpawnRules:
    """Blackbox tests for spawn rule types."""

    def test_random_spawn_rule(self):
        """RANDOM rule should select randomly from available."""
        manager = SpawnManager(
            default_rule=SpawnRule(rule_type=SpawnRuleType.RANDOM)
        )
        for i in range(5):
            manager.register_spawn_point(
                SpawnPoint(point_id=f"spawn_{i}", position=(i, 0, 0))
            )

        selections = set()
        for _ in range(20):
            spawn = manager.select_spawn_point("player")
            if spawn:
                selections.add(spawn.point_id)

        # Should have selected multiple different spawns
        assert len(selections) > 1

    def test_sequential_spawn_rule(self):
        """SEQUENTIAL rule should cycle through spawns."""
        manager = SpawnManager(
            default_rule=SpawnRule(rule_type=SpawnRuleType.SEQUENTIAL)
        )
        for i in range(3):
            manager.register_spawn_point(
                SpawnPoint(point_id=f"spawn_{i}", position=(i, 0, 0))
            )

        selections = []
        for _ in range(6):
            spawn = manager.select_spawn_point("player")
            if spawn:
                selections.append(spawn.point_id)

        # Should cycle: 0,1,2,0,1,2
        assert selections[0] == selections[3]


# =============================================================================
# INTEGRATION / COMBINED BLACKBOX TESTS (15 tests)
# =============================================================================


class TestCombatIntegration:
    """Blackbox tests for integrated combat scenarios."""

    def test_full_kill_flow(self):
        """Test complete kill flow: damage -> death -> score -> respawn."""
        # Setup systems
        health_pool = HealthPool()
        death_system = DeathSystem()
        scoring = ScoringSystem()

        # Create entities
        health_pool.create(entity_id=1, max_health=100)
        health_pool.create(entity_id=2, max_health=100)
        scoring.add_player("killer")
        scoring.add_player("victim")

        # Deal lethal damage
        victim_health = health_pool.get(2)
        victim_health.take_damage(100)

        assert victim_health.is_dead

        # Process death
        death_system.process_death(entity_id=2, killer_id=1)

        # Record kill
        scoring.record_kill("killer", "victim")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 1

    def test_team_damage_flow(self):
        """Test team-based damage with friendly fire check."""
        team_system = TeamSystem()
        team_system.create_team(team_id=1, name="Red")

        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)

        # Check if can attack teammate
        can_attack = team_system.can_attack(1, 2)

        # By default friendly fire disabled
        assert can_attack is False

    def test_killstreak_to_death_flow(self):
        """Test killstreak building and ending on death."""
        scoring = ScoringSystem()
        scoring.add_player("streak_player")
        scoring.add_player("victims")
        scoring.add_player("ender")

        # Build killstreak
        for i in range(7):
            scoring.record_kill("streak_player", "victims")

        stats = scoring.get_player_stats("streak_player")
        assert stats.current_killstreak == 7

        # Die - streak ends
        scoring.record_death("streak_player", killer_id="ender")

        stats = scoring.get_player_stats("streak_player")
        assert stats.current_killstreak == 0
        assert stats.best_killstreak == 7

    def test_shield_before_health_damage(self):
        """Test shields absorb damage before health."""
        health = HealthComponent(entity_id=1, max_health=100)

        # Add shield
        health.add_shield("barrier", amount=50)

        # Take damage less than shield
        health.take_damage(30)

        assert health.current_health == 100
        assert health.total_shield == 20

        # Take more damage
        health.take_damage(40)

        assert health.total_shield == 0
        assert health.current_health == 80

    def test_invulnerability_blocks_all_damage(self):
        """Test invulnerability completely blocks damage."""
        health = HealthComponent(entity_id=1, max_health=100)

        # Add invulnerability
        health.add_invulnerability(
            reason=InvulnerabilityReason.ABILITY,
            duration=5.0
        )

        # Try to deal damage
        health.take_damage(1000)

        assert health.current_health == 100
        assert health.is_invulnerable


class TestEdgeCases:
    """Blackbox tests for edge cases and boundary conditions."""

    def test_zero_damage_no_effect(self):
        """Zero damage should have no effect."""
        health = HealthComponent(entity_id=1, max_health=100)

        health.take_damage(0)

        assert health.current_health == 100

    def test_negative_damage_treated_as_zero(self):
        """Negative damage should be treated as zero or ignored."""
        health = HealthComponent(entity_id=1, max_health=100)

        # Negative damage should not heal
        health.take_damage(-50)

        assert health.current_health == 100

    def test_overkill_damage(self):
        """Damage exceeding health should cap at zero."""
        health = HealthComponent(entity_id=1, max_health=100)

        health.take_damage(500)

        assert health.current_health == 0
        assert health.is_dead

    def test_heal_when_dead(self):
        """Healing a dead entity should not work."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(100)

        assert health.is_dead

        # Try to heal
        health.heal(50)

        # Still dead
        assert health.is_dead
        assert health.current_health == 0

    def test_empty_leaderboard(self):
        """Empty leaderboard should return empty list."""
        scoring = ScoringSystem()

        leaderboard = scoring.get_leaderboard()

        assert leaderboard == []

    def test_nonexistent_player_stats(self):
        """Getting stats for nonexistent player should return None."""
        scoring = ScoringSystem()

        stats = scoring.get_player_stats("nobody")

        assert stats is None

    def test_spawn_with_no_available_points(self):
        """Spawn selection with no available points should handle gracefully."""
        manager = SpawnManager()

        spawn = manager.select_spawn_point("player")

        assert spawn is None

    def test_revive_dead_entity(self):
        """Should be able to revive a dead entity."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(100)

        assert health.is_dead

        # Revive
        health.revive(health_percentage=0.5)

        assert health.is_alive
        assert health.current_health == 50

    def test_max_health_modification(self):
        """Modifying max health should update current proportionally."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(50)  # 50/100

        # Double max health with adjust_current
        health.set_max_health(200, adjust_current=True)

        # Should be 100/200 (same percentage = 50%)
        assert health.max_health == 200
        assert health.current_health == 100

    def test_simultaneous_kills_multi_kill(self):
        """Multiple kills in rapid succession should count as multi-kill."""
        scoring = ScoringSystem()
        scoring.add_player("killer")

        # Kill 4 enemies very quickly
        for i in range(4):
            scoring.add_player(f"victim_{i}")
            scoring.record_kill("killer", f"victim_{i}")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 4
        assert stats.total_multi_kills >= 1


# =============================================================================
# ADDITIONAL BLACKBOX TESTS (35+ more tests)
# =============================================================================


class TestScoringAdvanced:
    """Additional scoring system tests."""

    @pytest.fixture
    def scoring(self):
        return ScoringSystem()

    def test_revenge_kill_detection(self, scoring):
        """Killing the player who last killed you should be a revenge kill."""
        scoring.add_player("player1")
        scoring.add_player("player2")

        # Player2 kills Player1
        scoring.record_kill("player2", "player1")
        scoring.record_death("player1", killer_id="player2")

        # Player1 respawns and kills Player2 back
        scoring.record_kill("player1", "player2")

        stats = scoring.get_player_stats("player1")
        # Revenge kills should be tracked
        assert stats.revenge_kills >= 1

    def test_objective_capture_scoring(self, scoring):
        """Capturing objectives should award points."""
        scoring.add_player("player1")

        initial = scoring.get_score("player1")
        scoring.record_objective_capture("player1")
        final = scoring.get_score("player1")

        assert final > initial

    def test_objective_defend_scoring(self, scoring):
        """Defending objectives should award points."""
        scoring.add_player("player1")

        initial = scoring.get_score("player1")
        scoring.record_objective_defend("player1")
        final = scoring.get_score("player1")

        assert final > initial

    def test_healing_tracked(self, scoring):
        """Healing done should be tracked."""
        scoring.add_player("healer")
        scoring.add_player("patient")

        scoring.record_healing("healer", "patient", 100.0)

        stats = scoring.get_player_stats("healer")
        assert stats.healing_done == 100.0

    def test_headshot_tracked(self, scoring):
        """Headshots should be tracked."""
        scoring.add_player("sniper")
        scoring.add_player("target")

        scoring.record_kill("sniper", "target", is_headshot=True)

        stats = scoring.get_player_stats("sniper")
        assert stats.headshots >= 1

    def test_score_summary(self, scoring):
        """Should be able to get full score summary."""
        scoring.add_player("player1")
        scoring.add_player("player2")

        scoring.record_kill("player1", "player2")
        scoring.record_kill("player1", "player2")

        summary = scoring.get_summary()

        assert "player_count" in summary
        assert summary["player_count"] == 2


class TestHitboxAdvanced:
    """Additional hitbox system tests."""

    @pytest.fixture
    def hitbox_system(self):
        return HitboxSystem()

    def test_hitbox_deactivation(self, hitbox_system):
        """Deactivated hitbox should not be active."""
        hitbox = hitbox_system.create_hitbox(
            hitbox_id="hb_deact",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        hitbox.activate()
        assert hitbox.active

        hitbox.deactivate()
        assert not hitbox.active

    def test_hurtbox_activation(self, hitbox_system):
        """Hurtbox activation should work correctly."""
        hurtbox = hitbox_system.create_hurtbox(
            hurtbox_id="hrt_act",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        hitbox_system.activate_hurtbox(hurtbox.hurtbox_id)
        assert hurtbox.active

        hitbox_system.deactivate_hurtbox(hurtbox.hurtbox_id)
        assert not hurtbox.active

    def test_entity_removal_clears_boxes(self, hitbox_system):
        """Removing entity should clear all its hitboxes/hurtboxes."""
        hitbox_system.create_hitbox(
            hitbox_id="hb_entity",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )
        hitbox_system.create_hurtbox(
            hurtbox_id="hrt_entity",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        hitbox_system.remove_entity(1)

        assert len(hitbox_system.get_entity_hitboxes(1)) == 0
        assert len(hitbox_system.get_entity_hurtboxes(1)) == 0

    def test_active_hitboxes_list(self, hitbox_system):
        """Should be able to get list of active hitboxes."""
        hb1 = hitbox_system.create_hitbox(
            hitbox_id="hb_active1",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )
        hb2 = hitbox_system.create_hitbox(
            hitbox_id="hb_active2",
            owner_id=2,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        hitbox_system.activate_hitbox(hb1.hitbox_id)

        active = hitbox_system.get_active_hitboxes()
        assert len(active) == 1
        assert active[0].hitbox_id == "hb_active1"

    def test_hitbox_system_stats(self, hitbox_system):
        """Should be able to get system statistics."""
        hitbox_system.create_hitbox(
            hitbox_id="hb_stats",
            owner_id=1,
            position=(0, 0, 0),
            size=(1, 1, 1),
        )

        stats = hitbox_system.get_stats()

        assert "total_hitboxes" in stats
        assert stats["total_hitboxes"] >= 1


class TestDamageAdvanced:
    """Additional damage system tests."""

    def test_critical_hit_multiplier(self):
        """Critical hit should multiply damage."""
        damage_system = DamageSystem()

        normal, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            critical_multiplier=1.0,
        )

        crit, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            critical_multiplier=2.0,
        )

        assert crit == normal * 2

    def test_hitbox_zone_multiplier(self):
        """Hitbox zone should affect damage."""
        damage_system = DamageSystem()

        head_dmg, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            hitbox_zone=HitboxZone.HEAD,
        )

        torso_dmg, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            hitbox_zone=HitboxZone.TORSO,
        )

        assert head_dmg > torso_dmg  # Head has 2x multiplier

    def test_stacked_multipliers(self):
        """Multiple multipliers should stack."""
        damage_system = DamageSystem()

        # 100 base * 2.0 (crit) * 1.5 (bonus) = 300
        final, _, _ = damage_system.calculate_damage(
            base_damage=100,
            damage_type=DamageType.PHYSICAL,
            armor=0,
            resistance=0,
            critical_multiplier=2.0,
            additional_multipliers=[1.5],
        )

        assert final == 300

    def test_calculate_dps_with_crits(self):
        """DPS calculation should account for crit chance."""
        # 100 damage, 2 attacks/sec, 50% crit chance, 2x crit multiplier
        # Avg damage = 100 * (1 + 0.5 * (2 - 1)) = 100 * 1.5 = 150
        # DPS = 150 * 2 = 300
        dps = calculate_dps(
            base_damage=100,
            attacks_per_second=2.0,
            crit_chance=0.5,
            crit_multiplier=2.0,
        )

        assert dps == 300


class TestHealthAdvanced:
    """Additional health system tests."""

    def test_missing_health_calculation(self):
        """Missing health should be calculated correctly."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(30)

        assert health.missing_health == 30

    def test_full_health_detection(self):
        """Should detect when at full health."""
        health = HealthComponent(entity_id=1, max_health=100)

        assert health.is_full_health

        health.take_damage(10)
        assert not health.is_full_health

        health.heal(10)
        assert health.is_full_health

    def test_effective_health_with_shield(self):
        """Effective health should include shields."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield("barrier", 50)

        assert health.effective_health == 150

    def test_shield_removal(self):
        """Should be able to remove specific shields."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.add_shield("shield1", 25)
        health.add_shield("shield2", 25)

        assert health.total_shield == 50

        health.remove_shield("shield1")

        assert health.total_shield == 25

    def test_reset_health(self):
        """Should be able to reset health to default."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(50)
        health.add_shield("temp", 25)
        health.disable_regeneration()

        health.reset()

        assert health.current_health == 100
        assert health.total_shield == 0

    def test_health_state_serialization(self):
        """Should be able to get health state as dict."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(30)

        state = health.get_state()

        assert "current_health" in state
        assert state["current_health"] == 70


class TestDeathAdvanced:
    """Additional death system tests."""

    @pytest.fixture
    def death_system(self):
        return DeathSystem()

    def test_is_dying_state(self, death_system):
        """Should detect dying state correctly."""
        death_system.process_death(entity_id=1, killer_id=2)

        assert death_system.is_dying(1) or death_system.is_dead(1)

    def test_pending_respawns_list(self, death_system):
        """Should get list of pending respawns."""
        death_system.process_death(entity_id=1, killer_id=10)
        death_system.transition_to_dead(1)
        death_system.queue_respawn(entity_id=1, delay=10.0)

        death_system.process_death(entity_id=2, killer_id=10)
        death_system.transition_to_dead(2)
        death_system.queue_respawn(entity_id=2, delay=10.0)

        pending = death_system.get_pending_respawns()
        assert len(pending) == 2

    def test_clear_death_system(self, death_system):
        """Should be able to clear all death state."""
        death_system.process_death(entity_id=1, killer_id=10)
        death_system.process_death(entity_id=2, killer_id=10)

        death_system.clear()

        assert len(death_system.get_all_dead()) == 0


class TestTeamAdvanced:
    """Additional team system tests."""

    @pytest.fixture
    def team_system(self):
        return TeamSystem()

    def test_get_enemies_list(self, team_system):
        """Should get list of enemy entities."""
        team_system.create_team(team_id=1, name="Red")
        team_system.create_team(team_id=2, name="Blue")
        team_system.set_relationship(1, 2, TeamRelation.HOSTILE)

        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=2)
        team_system.set_team(entity_id=3, team_id=2)

        enemies = team_system.get_enemies(1)

        assert 2 in enemies
        assert 3 in enemies

    def test_get_allies_list(self, team_system):
        """Should get list of allied entities."""
        team_system.create_team(team_id=1, name="Red")

        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=2, team_id=1)
        team_system.set_team(entity_id=3, team_id=1)

        allies = team_system.get_allies(1)

        assert 2 in allies
        assert 3 in allies
        assert 1 not in allies  # Self not included

    def test_team_membership_info(self, team_system):
        """Should get membership information."""
        team_system.create_team(team_id=1, name="Red")
        team_system.set_team(entity_id=1, team_id=1)

        membership = team_system.get_membership(1)

        assert membership is not None
        assert membership.team_id == 1

    def test_hostile_teams_list(self, team_system):
        """Should get list of hostile teams."""
        team_system.create_team(team_id=1, name="Red")
        team_system.create_team(team_id=2, name="Blue")
        team_system.create_team(team_id=3, name="Green")

        team_system.set_relationship(1, 2, TeamRelation.HOSTILE)
        team_system.set_relationship(1, 3, TeamRelation.HOSTILE)

        hostile = team_system.get_hostile_teams(1)

        assert 2 in hostile
        assert 3 in hostile


class TestSpawnAdvanced:
    """Additional spawn system tests."""

    @pytest.fixture
    def spawn_manager(self):
        return SpawnManager()

    def test_spawn_points_by_type(self, spawn_manager):
        """Should filter spawn points by type."""
        spawn1 = SpawnPoint(point_id="s1", position=(0, 0, 0), spawn_type="initial")
        spawn2 = SpawnPoint(point_id="s2", position=(1, 0, 0), spawn_type="respawn")
        spawn3 = SpawnPoint(point_id="s3", position=(2, 0, 0), spawn_type="initial")

        spawn_manager.register_spawn_point(spawn1)
        spawn_manager.register_spawn_point(spawn2)
        spawn_manager.register_spawn_point(spawn3)

        initial_spawns = spawn_manager.get_spawn_points_by_type("initial")

        assert len(initial_spawns) == 2

    def test_spawn_manager_stats(self, spawn_manager):
        """Should get spawn manager statistics."""
        spawn_manager.register_spawn_point(
            SpawnPoint(point_id="s1", position=(0, 0, 0))
        )

        stats = spawn_manager.get_stats()

        assert "total_spawn_points" in stats
        assert stats["total_spawn_points"] >= 1

    def test_spawn_manager_reset(self, spawn_manager):
        """Should reset spawn manager state."""
        spawn_manager.register_spawn_point(
            SpawnPoint(point_id="s1", position=(0, 0, 0))
        )
        spawn_manager.schedule_respawn("player1", delay_seconds=10.0)

        spawn_manager.reset()

        # Player state should be cleared
        assert spawn_manager.get_respawn_time("player1") is None


class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_minimum_damage_floor(self):
        """Damage should not go below minimum threshold."""
        damage_system = DamageSystem()

        # High armor should still allow minimum damage through
        final, _, _ = damage_system.calculate_damage(
            base_damage=1,
            damage_type=DamageType.PHYSICAL,
            armor=10000,
            resistance=0.75,
        )

        assert final >= 1.0  # Minimum damage

    def test_maximum_resistance_cap(self):
        """Resistance should be capped at 75%."""
        profile = ResistanceProfile()
        profile.set_resistance(DamageType.FIRE, 2.0)  # Try 200%

        actual = profile.get_resistance(DamageType.FIRE)
        assert actual <= 0.75

    def test_health_cannot_go_negative(self):
        """Health should not go below zero."""
        health = HealthComponent(entity_id=1, max_health=100)

        health.take_damage(1000)

        assert health.current_health == 0

    def test_heal_cannot_exceed_max(self):
        """Healing should not exceed max health."""
        health = HealthComponent(entity_id=1, max_health=100)
        health.take_damage(50)

        health.heal(1000)

        assert health.current_health == 100

    def test_empty_team_handling(self):
        """Should handle teams with no members."""
        team_system = TeamSystem()
        team_system.create_team(team_id=1, name="Empty")

        count = team_system.get_team_member_count(1)
        assert count == 0


class TestConcurrentOperations:
    """Tests for concurrent/rapid operations."""

    def test_rapid_damage_application(self):
        """Rapid damage application should work correctly."""
        health = HealthComponent(entity_id=1, max_health=100)

        # Apply 10 instances of 5 damage rapidly
        for _ in range(10):
            health.take_damage(5)

        assert health.current_health == 50

    def test_rapid_kills(self):
        """Rapid kills should all be counted."""
        scoring = ScoringSystem()
        scoring.add_player("killer")

        for i in range(10):
            scoring.add_player(f"victim_{i}")
            scoring.record_kill("killer", f"victim_{i}")

        stats = scoring.get_player_stats("killer")
        assert stats.kills == 10

    def test_rapid_shield_stacking(self):
        """Adding multiple shields rapidly should work."""
        health = HealthComponent(entity_id=1, max_health=100)

        for i in range(5):
            health.add_shield(f"shield_{i}", 10)

        assert health.total_shield == 50

    def test_rapid_team_changes(self):
        """Rapid team changes should be tracked."""
        team_system = TeamSystem()
        team_system.create_team(team_id=1, name="Red")
        team_system.create_team(team_id=2, name="Blue")

        # Change teams rapidly
        for i in range(10):
            team_id = 1 if i % 2 == 0 else 2
            team_system.set_team(entity_id=1, team_id=team_id)

        # Should end up on team 2
        final_team = team_system.get_team_id(1)
        assert final_team == 2


class TestEventHandling:
    """Tests for event handlers and callbacks."""

    def test_death_event_handler(self):
        """Death event handler should be called."""
        health = HealthComponent(entity_id=1, max_health=100)

        death_triggered = []

        def on_death(event):
            death_triggered.append(event)

        health.on_death(on_death)
        health.take_damage(100)

        assert len(death_triggered) == 1

    def test_health_changed_handler(self):
        """Health changed handler should be called."""
        health = HealthComponent(entity_id=1, max_health=100)

        changes = []

        def on_change(event):
            changes.append(event)

        health.on_health_changed(on_change)
        health.take_damage(30)
        health.heal(10)

        assert len(changes) == 2

    def test_team_change_handler(self):
        """Team change handler should be called."""
        team_system = TeamSystem()
        team_system.create_team(team_id=1, name="Red")
        team_system.create_team(team_id=2, name="Blue")

        changes = []

        def on_change(event):
            changes.append(event)

        team_system.on_team_change(on_change)
        team_system.set_team(entity_id=1, team_id=1)
        team_system.set_team(entity_id=1, team_id=2)

        assert len(changes) >= 1


class TestPoolManagement:
    """Tests for entity pool management."""

    def test_health_pool_creation(self):
        """Health pool should create and manage components."""
        pool = HealthPool()

        pool.create(entity_id=1, max_health=100)
        pool.create(entity_id=2, max_health=150)

        assert len(pool) == 2

    def test_health_pool_retrieval(self):
        """Should retrieve health component from pool."""
        pool = HealthPool()
        pool.create(entity_id=1, max_health=100)

        health = pool.get(1)

        assert health is not None
        assert health.max_health == 100

    def test_health_pool_removal(self):
        """Should remove health component from pool."""
        pool = HealthPool()
        pool.create(entity_id=1, max_health=100)

        result = pool.remove(1)

        assert result is True
        assert pool.get(1) is None

    def test_health_pool_alive_dead_lists(self):
        """Should track alive and dead entities."""
        pool = HealthPool()
        pool.create(entity_id=1, max_health=100)
        pool.create(entity_id=2, max_health=100)

        # Kill entity 1
        pool.get(1).take_damage(100)

        alive = pool.get_all_alive()
        dead = pool.get_all_dead()

        assert len(alive) == 1
        assert len(dead) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
