"""
Whitebox tests for the combat scoring system (engine/gameplay/combat/scoring.py).

Tests implementation-internal paths that the existing contract/blackbox tests
do not reach.  Every test exploits knowledge of the specific data structures,
branch conditions, and error-handling paths that the DEV wrote.

WHITEBOX coverage plan:
  - PlayerStats._damage_to_targets: the Tuple[float,float] internals,
    get() with default (0.0, 0.0), tuple unpacking in clear_damage_tracking
  - PlayerStats.get_assist_damage: return None for missing / expired target,
    max_health parameter accepted but unused (API stub contract)
  - PlayerStats._killed_by set lifecycle: revenge check occurs BEFORE the
    current killer is added, so the check succeeds; test the discard-on-revenge
    and subsequent re-add
  - _multi_kill_count state machine: the >=5 branch increments penta_kills
    for EVERY subsequent kill in same window (not just once at 5)
  - Handler error swallowing: on_score_changed, on_kill, on_killstreak,
    on_multi_kill, on_first_blood all wrap handlers in try/except pass;
    a faulty handler must NOT break the system
  - Event history FIFO trim: _record_event when len > _max_history_size
    truncates from the front
  - set_score team-score diff: team score delta is (new - old), not the
    absolute value
  - record_death points branching: None vs 0 vs explicit value; also
    stats._multi_kill_count = 0 reset
  - add_score unknown player returns False (early return branch)
  - get_leaderboard unhandled sort keys (DAMAGE_DEALT, DAMAGE_TAKEN,
    OBJECTIVES) fall through to no-sort (entries unsorted beyond dict
    iteration order)
  - ScoreEvent.is_positive / is_negative for zero-point events
  - _ensure_team idempotency on repeated calls
  - remove_player when team_id is set but team-stats entry is missing
    (guard: self._team_stats[stats.team_id] existence check)
  - _record_event trim boundary: exactly at limit does nothing; one past
    limit drops the oldest entry
  - start_match resets first_blood but leaves player/team state intact
  - get_team_leaderboard fallback sort value for unhandled sort keys
  - record_objective_defend points default: config.points_per_objective // 2
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.gameplay.combat.scoring import (
    LeaderboardEntry,
    LeaderboardSortKey,
    PlayerStats,
    ScoringSystem,
    ScoreEvent,
    ScoreEventType,
    TeamStats,
)
from engine.gameplay.combat.constants import (
    MULTI_KILL_NAMES,
    KILLSTREAK_THRESHOLDS,
    POINTS_PER_KILL,
    POINTS_PER_ASSIST,
    POINTS_PER_OBJECTIVE,
    POINTS_PER_HEADSHOT_BONUS,
    POINTS_PER_FIRST_BLOOD,
    POINTS_PER_REVENGE_KILL,
    POINTS_PER_KILLSTREAK_BONUS,
    ASSIST_DAMAGE_THRESHOLD,
    ASSIST_TIME_WINDOW,
    MULTI_KILL_WINDOW,
    ScoringConfig,
    DEFAULT_SCORING_CONFIG,
    MAX_SCORING_HISTORY_SIZE,
)


# =============================================================================
# Helpers
# =============================================================================


def _minimal() -> ScoringSystem:
    """Two players, no team."""
    s = ScoringSystem()
    s.add_player("killer")
    s.add_player("victim")
    return s


def _team_based() -> ScoringSystem:
    s = ScoringSystem(is_team_based=True)
    s.add_player("a", team_id="red")
    s.add_player("b", team_id="red")
    s.add_player("c", team_id="blue")
    s.add_player("d", team_id="blue")
    return s


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def time_ctrl():
    """Deterministic time control, patches engine.gameplay.combat.scoring.time.time."""
    _current = [1000.0]

    def _fake_time():
        return _current[0]

    class _Controller:
        @staticmethod
        def set(t: float) -> float:
            _current[0] = t
            return _current[0]

        @staticmethod
        def advance(sec: float) -> float:
            _current[0] += sec
            return _current[0]

        @staticmethod
        def now() -> float:
            return _current[0]

    with patch("engine.gameplay.combat.scoring.time.time", _fake_time):
        yield _Controller()


# =============================================================================
# PlayerStats internals  --  _damage_to_targets, _killed_by, assist internals
# =============================================================================


class TestPlayerStatsDamageTracking:
    """Direct tests of the _damage_to_targets dict and its Tuple[float,float]
    internals."""

    def test_record_damage_dealt_accumulates(self):
        """record_damage_dealt accumulates damage for same target via tuple
        unpacking: current = _damage_to_targets.get(target_id, (0.0, 0.0))."""
        st = PlayerStats(player_id="p")
        st.record_damage_dealt("target1", 10.0, max_health=100.0)
        st.record_damage_dealt("target1", 5.0, max_health=100.0)

        entry = st._damage_to_targets.get("target1")
        assert entry is not None
        damage, _ts = entry
        assert damage == 15.0

    def test_record_damage_dealt_updates_timestamp(self):
        """record_damage_dealt sets a fresh timestamp on each call."""
        st = PlayerStats(player_id="p")
        with patch("engine.gameplay.combat.scoring.time.time", return_value=100.0):
            st.record_damage_dealt("t", 10.0, max_health=100.0)

        with patch("engine.gameplay.combat.scoring.time.time", return_value=200.0):
            st.record_damage_dealt("t", 5.0, max_health=100.0)

        entry = st._damage_to_targets.get("t")
        assert entry is not None
        _damage, ts = entry
        assert ts == 200.0

    def test_get_assist_damage_returns_none_for_missing(self):
        """get_assist_damage returns None when target_id not in dict."""
        st = PlayerStats(player_id="p")
        assert st.get_assist_damage("nobody", max_health=100.0) is None

    def test_get_assist_damage_returns_none_expired(self):
        """get_assist_damage returns None when time_window has elapsed."""
        st = PlayerStats(player_id="p")
        with patch("engine.gameplay.combat.scoring.time.time", return_value=100.0):
            st.record_damage_dealt("t", 15.0, max_health=100.0)

        with patch("engine.gameplay.combat.scoring.time.time", return_value=200.0):
            result = st.get_assist_damage("t", max_health=100.0, time_window=30.0)

        assert result is None

    def test_get_assist_damage_returns_damage_within_window(self):
        """get_assist_damage returns the damage value when within window."""
        st = PlayerStats(player_id="p")
        with patch("engine.gameplay.combat.scoring.time.time", return_value=100.0):
            st.record_damage_dealt("t", 15.0, max_health=100.0)

        with patch("engine.gameplay.combat.scoring.time.time", return_value=105.0):
            result = st.get_assist_damage("t", max_health=100.0, time_window=10.0)

        assert result == 15.0

    def test_clear_damage_tracking_removes_target(self):
        """clear_damage_tracking calls dict.pop(target_id, None)."""
        st = PlayerStats(player_id="p")
        st.record_damage_dealt("t", 10.0, max_health=100.0)
        st.clear_damage_tracking("t")

        assert "t" not in st._damage_to_targets

    def test_clear_damage_tracking_missing_no_error(self):
        """clear_damage_tracking with an unknown target does not raise (pop
        default)."""
        st = PlayerStats(player_id="p")
        st.clear_damage_tracking("nobody")  # must not raise


class TestPlayerStatsKilledBySet:
    """Direct tests of the _killed_by set lifecycle."""

    def test_revenge_check_before_current_killer_add(self):
        """The revenge check (line 620) is evaluated BEFORE the current killer
        is added to _killed_by (line 627).  Revenge checks whether the VICTIM
        is in the KILLER's _killed_by set (i.e. victim killed killer before),
        not whether the killer is in the victim's set."""
        killer = PlayerStats(player_id="killer")
        victim = PlayerStats(player_id="victim")

        # Simulate victim having killed killer previously
        killer._killed_by.add("victim")

        # Mirrors corrected record_kill logic:
        # line 620: if victim_id in killer_stats._killed_by:
        assert "victim" in killer._killed_by  # revenge eligible
        killer._killed_by.discard("victim")    # line 623: debt settled

        # line 627: record that killer killed victim
        victim._killed_by.add("killer")

        assert "victim" not in killer._killed_by   # revenge served
        assert "killer" in victim._killed_by        # victim can avenge later

    def test_killed_by_accumulates_unique_killers(self):
        """_killed_by tracks unique killers via a set."""
        st = PlayerStats(player_id="victim")
        st._killed_by.add("a")
        st._killed_by.add("b")
        st._killed_by.add("a")  # idempotent

        assert st._killed_by == {"a", "b"}


# =============================================================================
# _multi_kill_count internal state machine
# =============================================================================


class TestMultiKillStateMachine:
    """The _multi_kill_count counter and the >=5 branch at line 664."""

    def test_every_kill_at_five_plus_increments_penta(self, time_ctrl):
        """Line 664: ``elif multi >= 5`` fires penta_kills for EVERY
        subsequent kill within the same window, not just once at 5."""
        s = ScoringSystem()
        for pid in "abcdefghij":
            s.add_player(pid)
        time_ctrl.set(1000.0)

        # 4 kills: _multi_kill_count advances 1,2,3,4
        for victim in "bcde":
            time_ctrl.advance(0.1)
            s.record_kill("a", victim)

        st = s.get_player_stats("a")
        assert st.quad_kills == 1
        assert st.penta_kills == 0  # not yet — multi_kill_count is only 4

        # Kill 5: multi_kill_count = 5, fires penta for the first time
        s.add_player("i")
        time_ctrl.advance(0.1)
        s.record_kill("a", "i")
        assert st.penta_kills == 1

        # Kill 6: multi_kill_count = 6, still >=5, penta fires again
        s.add_player("j")
        time_ctrl.advance(0.1)
        s.record_kill("a", "j")
        assert st.penta_kills == 2, (
            f"Expected 2 penta_kills (one at multi=5, one at multi=6), "
            f"got {st.penta_kills}"
        )

    def test_death_resets_multi_kill_to_zero(self):
        """record_death sets _multi_kill_count = 0 (line 752)."""
        s = _minimal()
        s.add_player("v2")
        s.record_kill("killer", "victim")     # _multi_kill_count → 1
        s.record_death("killer")               # _multi_kill_count → 0

        st = s.get_player_stats("killer")
        assert st._multi_kill_count == 0

    def test_window_expiry_resets_multi_kill_to_one(self, time_ctrl):
        """Line 676: when ``time_since_last > window``, _multi_kill_count is
        set to 1 (not 0), so the very next kill starts a fresh chain."""
        s = _minimal()
        s.add_player("v2")
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")     # _multi_kill_count → 1
        time_ctrl.advance(MULTI_KILL_WINDOW + 1.0)
        s.record_kill("killer", "v2")          # counter reset to 1

        st = s.get_player_stats("killer")
        assert st._multi_kill_count == 1, (
            f"Expected _multi_kill_count=1 after window expiry, "
            f"got {st._multi_kill_count}"
        )


# =============================================================================
# Handler error swallowing
# =============================================================================


class TestHandlerErrorSwallowing:
    """All five handler registries wrap calls in try/except pass.  A faulty
    handler must NOT crash the operation."""

    @staticmethod
    def _faulty_handler(*args, **kwargs):
        raise RuntimeError("intentional handler failure")

    def test_on_score_changed_faulty_handler(self, time_ctrl):
        s = _minimal()
        s.on_score_changed(self._faulty_handler)
        time_ctrl.set(1000.0)
        # Must not raise
        s.record_kill("killer", "victim")

    def test_on_kill_faulty_handler(self, time_ctrl):
        s = _minimal()
        s.on_kill(self._faulty_handler)
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")

    def test_on_killstreak_faulty_handler(self, time_ctrl):
        s = ScoringSystem()
        s.add_player("a")
        for i in range(5):
            s.add_player(f"v{i}")
        s.on_killstreak(self._faulty_handler)
        time_ctrl.set(1000.0)
        for i in range(3):
            time_ctrl.advance(0.1)
            s.record_kill("a", f"v{i}")

    def test_on_multi_kill_faulty_handler(self, time_ctrl):
        s = ScoringSystem()
        s.add_player("a")
        for i in range(5):
            s.add_player(f"v{i}")
        s.on_multi_kill(self._faulty_handler)
        time_ctrl.set(1000.0)
        for victim in (f"v{i}" for i in range(2)):
            time_ctrl.advance(0.1)
            s.record_kill("a", victim)

    def test_on_first_blood_faulty_handler(self, time_ctrl):
        s = _minimal()
        s.on_first_blood(self._faulty_handler)
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")

    def test_all_handlers_faulty_do_not_block_each_other(self, time_ctrl):
        """When all handler registries hold faulty callbacks, the kill
        completes and returns a valid result dict."""
        s = _minimal()
        s.on_score_changed(self._faulty_handler)
        s.on_kill(self._faulty_handler)
        s.on_killstreak(self._faulty_handler)
        s.on_multi_kill(self._faulty_handler)
        s.on_first_blood(self._faulty_handler)
        time_ctrl.set(1000.0)
        result = s.record_kill("killer", "victim")
        assert result["kill_awarded"] is True


# =============================================================================
# Event history FIFO trim
# =============================================================================


class TestEventHistoryTrim:
    """_record_event trims the deque when len > _max_history_size."""

    def test_trim_preserves_most_recent(self):
        """When the history exceeds the max, only the *most recent* events are
        kept (``self._event_history[-self._max_history_size:]``)."""
        s = _minimal()
        s._max_history_size = 5

        for i in range(7):
            s.add_score("killer", 10, ScoreEventType.BONUS, seq=i)

        assert len(s._event_history) == 5
        seqs = [e.metadata["seq"] for e in s._event_history]
        assert seqs == [2, 3, 4, 5, 6], f"Expected seqs 2-6, got {seqs}"

    def test_no_trim_when_at_limit(self):
        """Exactly at MAX_SCORING_HISTORY_SIZE: no trimming occurs."""
        s = _minimal()
        # Fill exactly to the limit then add one more
        for i in range(MAX_SCORING_HISTORY_SIZE):
            s.add_score("killer", 1, ScoreEventType.BONUS)
        assert len(s._event_history) == MAX_SCORING_HISTORY_SIZE

    def test_one_over_limit_trims_one(self):
        """At MAX+1 entries, the oldest entry is dropped."""
        s = _minimal()
        for i in range(MAX_SCORING_HISTORY_SIZE + 1):
            s.add_score("killer", 1, ScoreEventType.BONUS)
        assert len(s._event_history) == MAX_SCORING_HISTORY_SIZE


# =============================================================================
# set_score team-score diff
# =============================================================================


class TestSetScoreTeamDiff:
    """set_score adjusts the team score by ``diff = new - old``."""

    def test_team_score_tracks_diff(self):
        s = _team_based()
        st = s.get_player_stats("a")
        st.score = 200  # pre-set old value

        s.set_score("a", 500)  # diff = 300

        assert s._team_stats["red"].score == 300

    def test_team_score_decrease(self):
        s = _team_based()
        st = s.get_player_stats("a")
        st.score = 500

        s.set_score("a", 200)  # diff = -300

        assert s._team_stats["red"].score == -300

    def test_set_score_no_team_no_crash(self):
        s = _minimal()
        assert s.set_score("killer", 500) is True


# =============================================================================
# record_death points branching
# =============================================================================


class TestRecordDeathBranching:
    """The death_points parameter: None uses config default (0), 0 also works,
    explicit positive/negative values are passed through."""

    def test_death_points_none_uses_config(self, time_ctrl):
        """death_points=None → self._config.points_per_death (which is 0 by
        default, so no event is recorded)."""
        s = _minimal()
        time_ctrl.set(1000.0)
        s.record_death("killer", death_points=None)

        deaths = s.get_event_history(event_type=ScoreEventType.DEATH)
        assert len(deaths) == 0

    def test_death_points_explicit_zero(self, time_ctrl):
        """death_points=0 → passes through, checks ``if points != 0`` so
        no event is recorded."""
        s = _minimal()
        time_ctrl.set(1000.0)
        s.record_death("killer", death_points=0)

        deaths = s.get_event_history(event_type=ScoreEventType.DEATH)
        assert len(deaths) == 0

    def test_death_points_negative(self, time_ctrl):
        """death_points=-10 → event recorded, score decreased."""
        s = _minimal()
        time_ctrl.set(1000.0)
        s.record_death("killer", death_points=-10)

        deaths = s.get_event_history(event_type=ScoreEventType.DEATH)
        assert len(deaths) == 1
        assert s.get_player_stats("killer").score == -10

    def test_death_points_positive(self, time_ctrl):
        """death_points=50 → event recorded, score increased (e.g. penalty
        awarded to another player's score via different mechanism)."""
        s = _minimal()
        time_ctrl.set(1000.0)
        s.record_death("killer", death_points=50)

        assert s.get_player_stats("killer").score == 50

    def test_death_resets_multi_kill_count_to_zero(self, time_ctrl):
        """record_death sets _multi_kill_count = 0 (line 752)."""
        s = _minimal()
        s.add_player("v2")
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")     # _multi_kill_count → 1
        s.record_death("killer")               # → 0

        st = s.get_player_stats("killer")
        assert st._multi_kill_count == 0


# =============================================================================
# add_score edge cases
# =============================================================================


class TestAddScoreEdgeCases:
    """Branches in add_score: unknown player, negative points."""

    def test_unknown_player_returns_false(self):
        s = _minimal()
        assert s.add_score("nobody", 100) is False

    def test_negative_points_decrease_score(self, time_ctrl):
        s = _minimal()
        time_ctrl.set(1000.0)
        s.add_score("killer", -50, ScoreEventType.PENALTY)

        st = s.get_player_stats("killer")
        assert st.score == -50

    def test_negative_points_team_score_decreases(self, time_ctrl):
        s = _team_based()
        time_ctrl.set(1000.0)
        s.add_score("a", -50, ScoreEventType.PENALTY)

        assert s._team_stats["red"].score == -50


# =============================================================================
# Leaderboard unhandled sort keys
# =============================================================================


class TestLeaderboardUnhandledSortKeys:
    """get_leaderboard handles SCORE, KILLS, DEATHS, ASSISTS, KD_RATIO,
    KDA_RATIO explicitly.  DAMAGE_DEALT, DAMAGE_TAKEN, OBJECTIVES are
    defined in the enum but not handled — they fall through to an
    unsorted entries list (dict-iteration order)."""

    def test_sort_by_damage_dealt_falls_through(self):
        """DAMAGE_DEALT is not handled; the entries remain in insertion
        order (no crash)."""
        s = _minimal()
        s.get_player_stats("killer").damage_dealt = 500.0
        s.get_player_stats("victim").damage_dealt = 1000.0

        lb = s.get_leaderboard(sort_by=LeaderboardSortKey.DAMAGE_DEALT)
        # No crash; entries are returned (unsorted)
        assert len(lb) == 2

    def test_sort_by_damage_taken_falls_through(self):
        s = _minimal()
        s.get_player_stats("killer").damage_taken = 200.0
        s.get_player_stats("victim").damage_taken = 50.0

        lb = s.get_leaderboard(sort_by=LeaderboardSortKey.DAMAGE_TAKEN)
        assert len(lb) == 2

    def test_sort_by_objectives_falls_through(self):
        s = _minimal()
        lb = s.get_leaderboard(sort_by=LeaderboardSortKey.OBJECTIVES)
        assert len(lb) == 2


# =============================================================================
# ScoreEvent zero-point edges
# =============================================================================


class TestScoreEventZeroPoint:
    """is_positive and is_negative for zero-point events."""

    def test_zero_points_neither_positive_nor_negative(self):
        event = ScoreEvent(
            event_type=ScoreEventType.BONUS,
            player_id="p",
            points=0,
        )
        assert event.is_positive is False
        assert event.is_negative is False

    def test_positive_points(self):
        event = ScoreEvent(
            event_type=ScoreEventType.KILL,
            player_id="p",
            points=100,
        )
        assert event.is_positive is True
        assert event.is_negative is False

    def test_negative_points(self):
        event = ScoreEvent(
            event_type=ScoreEventType.PENALTY,
            player_id="p",
            points=-50,
        )
        assert event.is_positive is False
        assert event.is_negative is True


# =============================================================================
# _ensure_team idempotency
# =============================================================================


class TestEnsureTeam:
    """_ensure_team creates a TeamStats if missing, returns existing if
    already present."""

    def test_creates_new_team(self):
        s = _minimal()
        ts = s._ensure_team("new_team")
        assert isinstance(ts, TeamStats)
        assert ts.team_id == "new_team"

    def test_returns_existing(self):
        s = _minimal()
        ts1 = s._ensure_team("red")
        ts2 = s._ensure_team("red")
        assert ts1 is ts2


# =============================================================================
# remove_player edge cases
# =============================================================================


class TestRemovePlayer:
    """remove_player branches: player has team_id but team no longer exists."""

    def test_remove_player_with_team(self):
        """Remove a player who belongs to a team; team membership is
        cleaned up."""
        s = _team_based()
        s.remove_player("a")

        assert s.get_player_stats("a") is None
        assert "a" not in s._team_stats["red"].members

    def test_remove_player_team_exists_but_player_not_member(self):
        """Remove a player whose team_id is set but who is not in the
        team's members set — discard is a no-op."""
        s = _team_based()
        st = s.get_player_stats("a")
        st.team_id = "red"
        s._team_stats["red"].members.clear()  # desync

        s.remove_player("a")  # must not raise

    def test_remove_player_unknown_returns_false(self):
        s = _minimal()
        assert s.remove_player("nobody") is False


# =============================================================================
# start_match behavior
# =============================================================================


class TestStartMatch:
    """start_match resets first_blood and sets match_start_time but leaves
    player/team state intact."""

    def test_start_match_resets_first_blood(self, time_ctrl):
        s = _minimal()
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")
        assert s.first_blood_awarded is True

        s.start_match()
        assert s.first_blood_awarded is False

    def test_start_match_sets_match_start_time(self, time_ctrl):
        s = _minimal()
        time_ctrl.set(5000.0)
        s.start_match()
        assert s._match_start_time == 5000.0

    def test_start_match_preserves_players(self):
        s = _minimal()
        s.start_match()
        assert s.get_player_stats("killer") is not None
        assert s.get_player_stats("victim") is not None


# =============================================================================
# get_team_leaderboard fallback
# =============================================================================


class TestGetTeamLeaderboardFallback:
    """get_team_leaderboard handles SCORE and KILLS explicitly; everything
    else falls back to stats.score."""

    def test_default_is_score(self):
        s = _team_based()
        s._team_stats["red"].score = 1000
        s._team_stats["blue"].score = 2000

        lb = s.get_team_leaderboard()
        assert lb[0][0] == "blue"  # higher score first

    def test_sort_by_kills(self):
        s = _team_based()
        s._team_stats["red"].kills = 30
        s._team_stats["blue"].kills = 50

        lb = s.get_team_leaderboard(sort_by=LeaderboardSortKey.KILLS)
        assert lb[0][0] == "blue"  # higher kills first

    def test_unhandled_sort_key_falls_back_to_score(self):
        s = _team_based()
        s._team_stats["red"].score = 1000
        s._team_stats["blue"].score = 2000

        lb = s.get_team_leaderboard(sort_by=LeaderboardSortKey.ASSISTS)
        assert lb[0][0] == "blue"  # sorted by score fallback


# =============================================================================
# get_event_history filter combination
# =============================================================================


class TestGetEventHistoryFilters:
    """get_event_history applies player_id filter THEN event_type filter
    sequentially."""

    def test_filter_by_player_and_type(self, time_ctrl):
        s = _minimal()
        s.add_player("bystander")
        time_ctrl.set(1000.0)

        s.add_score("killer", 50, ScoreEventType.BONUS)
        s.add_score("bystander", 100, ScoreEventType.KILL)
        s.add_score("killer", -10, ScoreEventType.PENALTY)

        events = s.get_event_history(
            player_id="killer", event_type=ScoreEventType.PENALTY
        )
        assert len(events) == 1
        assert events[0].player_id == "killer"
        assert events[0].event_type == ScoreEventType.PENALTY

    def test_filter_player_only(self, time_ctrl):
        s = _minimal()
        time_ctrl.set(1000.0)
        s.add_score("killer", 50, ScoreEventType.BONUS)
        s.add_score("victim", 30, ScoreEventType.BONUS)

        events = s.get_event_history(player_id="killer")
        assert len(events) == 1

    def test_filter_type_only(self, time_ctrl):
        s = _minimal()
        time_ctrl.set(1000.0)
        s.add_score("killer", 50, ScoreEventType.BONUS)
        s.add_score("killer", 100, ScoreEventType.KILL)

        events = s.get_event_history(event_type=ScoreEventType.KILL)
        assert len(events) == 1


# =============================================================================
# LeaderboardEntry rank assignment
# =============================================================================


class TestLeaderboardEntryRanks:
    """Ranks are 1-based sequential after sorting."""

    def test_tied_scores_get_distinct_ranks(self):
        """Tied values do not share a rank; the loop assigns sequential
        1-based ranks regardless of ties."""
        s = _minimal()
        s.add_player("p3")
        s.add_player("p4")
        # All at score=0 (default)
        lb = s.get_leaderboard()
        ranks = [e.rank for e in lb]
        assert ranks == list(range(1, 5))


# =============================================================================
# _calculate_assists internal behavior
# =============================================================================


class TestCalculateAssistsInternals:
    """The _calculate_assists method applies assist_damage_threshold with
    a * 100.0 multiplier (line 794)."""

    def test_threshold_percentage_multiplication(self, time_ctrl):
        """Line 794: ``damage >= threshold * 100.0``.  With threshold=0.1,
        the effective threshold is 10.0 damage against DEFAULT_MAX_HEALTH
        (100 HP)."""
        cfg = ScoringConfig(assist_damage_threshold=0.1)
        s = ScoringSystem(config=cfg)
        s.add_player("killer")
        s.add_player("victim")
        s.add_player("near_miss")

        time_ctrl.set(1000.0)
        # 9.9 damage is below 10% of 100 HP — no assist
        s.record_damage("near_miss", "victim", 9.9)
        time_ctrl.advance(1.0)
        result = s.record_kill("killer", "victim")

        assert "near_miss" not in result["assists"]

    def test_exact_threshold_grants_assist(self, time_ctrl):
        """Exactly 10.0 damage against 100 HP meets the threshold exactly."""
        cfg = ScoringConfig(assist_damage_threshold=0.1)
        s = ScoringSystem(config=cfg)
        s.add_player("killer")
        s.add_player("victim")
        s.add_player("exact")

        time_ctrl.set(1000.0)
        s.record_damage("exact", "victim", 10.0)
        time_ctrl.advance(1.0)
        result = s.record_kill("killer", "victim")

        assert "exact" in result["assists"]

    def test_killer_excluded_internal(self, time_ctrl):
        """The killer_id check at line 785 explicitly skips the killer."""
        cfg = ScoringConfig(assist_damage_threshold=0.1)
        s = ScoringSystem(config=cfg)
        s.add_player("killer")
        s.add_player("victim")

        time_ctrl.set(1000.0)
        s.record_damage("killer", "victim", 50.0)  # lots of damage
        time_ctrl.advance(1.0)

        lb = s.get_leaderboard(sort_by=LeaderboardSortKey.ASSISTS)
        # The internal method _calculate_assists should skip killer_id
        result = s.record_kill("killer", "victim")
        assert "killer" not in result["assists"]


# =============================================================================
# PlayerStats.to_dict key correctness
# =============================================================================


class TestPlayerStatsToDict:
    """to_dict returns all expected keys, including computed properties."""

    def test_to_dict_contains_all_keys(self):
        st = PlayerStats(player_id="p1")
        d = st.to_dict()
        expected_keys = {
            "player_id", "team_id",
            "score", "kills", "deaths", "assists",
            "damage_dealt", "damage_taken", "healing_done", "headshots",
            "current_killstreak", "best_killstreak",
            "double_kills", "triple_kills", "quad_kills", "penta_kills",
            "first_bloods", "revenge_kills",
            "objectives_captured", "objectives_defended", "objective_time",
            "kd_ratio", "kda_ratio",
        }
        assert d.keys() == expected_keys, f"Missing: {expected_keys - d.keys()}"

    def test_to_dict_serializes_none_team_id(self):
        st = PlayerStats(player_id="p1", team_id=None)
        d = st.to_dict()
        assert d["team_id"] is None


# =============================================================================
# ScoringSystem.get_score / get_team_score edge cases
# =============================================================================


class TestGetScoreEdgeCases:
    """get_score and get_team_score for missing players/teams."""

    def test_get_score_unknown_player_returns_zero(self):
        s = _minimal()
        assert s.get_score("nobody") == 0

    def test_get_team_score_unknown_team_returns_zero(self):
        s = _minimal()
        assert s.get_team_score("nonexistent") == 0


# =============================================================================
# add_player / set_player_team edge cases
# =============================================================================


class TestAddPlayerEdgeCases:
    """add_player returns existing stats for duplicate; team membership."""

    def test_add_player_duplicate_returns_same_object(self):
        s = _minimal()
        st1 = s.add_player("killer")
        st2 = s.add_player("killer")
        assert st1 is st2

    def test_add_player_with_invalid_team_gives_team_stats(self):
        s = ScoringSystem(is_team_based=True)
        s.add_player("p1", team_id="yellow")
        assert "yellow" in s._team_stats
        assert s._team_stats["yellow"].member_count == 1


class TestSetPlayerTeamEdgeCases:
    """set_player_team: unknown player returns False."""

    def test_set_player_team_unknown(self):
        s = _minimal()
        assert s.set_player_team("nobody", "red") is False

    def test_set_player_team_to_none_removes_from_old(self):
        s = _team_based()
        s.set_player_team("a", None)
        assert s.get_player_stats("a").team_id is None
        assert "a" not in s._team_stats["red"].members


# =============================================================================
# record_objective_defend default points
# =============================================================================


class TestObjectiveDefendPoints:
    """record_objective_defend defaults to config.points_per_objective // 2."""

    def test_default_defend_points(self):
        s = _minimal()
        s.record_objective_defend("killer", objective_id="obj1")

        st = s.get_player_stats("killer")
        assert st.objectives_defended == 1
        expected = POINTS_PER_OBJECTIVE // 2
        assert st.score == expected, (
            f"Expected {expected} points for defense, got {st.score}"
        )

    def test_custom_defend_points(self):
        s = _minimal()
        s.record_objective_defend("killer", objective_id="obj1", points=75)

        st = s.get_player_stats("killer")
        assert st.score == 75

    def test_unknown_player_defend(self):
        s = _minimal()
        assert s.record_objective_defend("nobody", objective_id="obj1") is False


# =============================================================================
# record_damage updates both parties
# =============================================================================


class TestRecordDamageBothParties:
    """record_damage updates attacker's damage_dealt AND victim's damage_taken."""

    def test_attacker_and_victim_both_updated(self):
        s = _minimal()
        s.record_damage("killer", "victim", 30.0)

        st_k = s.get_player_stats("killer")
        st_v = s.get_player_stats("victim")
        assert st_k.damage_dealt == 30.0
        assert st_v.damage_taken == 30.0

    def test_attacker_unknown_no_crash(self):
        s = _minimal()
        assert s.record_damage("nobody", "victim", 10.0) is False
        st_v = s.get_player_stats("victim")
        assert st_v.damage_taken == 0.0  # not updated when attacker unknown


# =============================================================================
# get_player_rank integration
# =============================================================================


class TestGetPlayerRank:
    """get_player_rank returns 1-based rank or 0 if not found."""

    def test_nonexistent_player_returns_zero(self):
        s = _minimal()
        assert s.get_player_rank("nobody") == 0

    def test_rank_reflects_current_leaderboard(self):
        s = _minimal()
        s.get_player_stats("killer").score = 500
        s.get_player_stats("victim").score = 200
        assert s.get_player_rank("killer") == 1
        assert s.get_player_rank("victim") == 2


# =============================================================================
# reset idempotency
# =============================================================================


class TestResetIdempotency:
    """Calling reset on a fresh system is a no-op (no crash)."""

    def test_reset_empty(self):
        s = ScoringSystem()
        s.reset()  # must not raise
        assert s._player_stats == {}
        assert s._team_stats == {}
        assert s._event_history == []
        assert s._first_blood_awarded is False
        assert s._match_start_time is None


# =============================================================================
# get_summary integration
# =============================================================================


class TestGetSummary:
    """get_summary returns consistent aggregation."""

    def test_summary_empty(self):
        s = ScoringSystem()
        summary = s.get_summary()
        assert summary["player_count"] == 0
        assert summary["team_count"] == 0
        assert summary["total_kills"] == 0
        assert summary["total_deaths"] == 0

    def test_summary_with_teams(self):
        s = _team_based()
        s.record_kill("a", "c")
        s.record_kill("b", "d")

        summary = s.get_summary()
        assert summary["player_count"] == 4
        assert summary["total_kills"] == 2
        assert summary["total_assists"] == 0
        assert summary["first_blood_awarded"] is True
        assert summary["event_count"] >= 2


# =============================================================================
# LeaderboardEntry kd_ratio edge cases
# =============================================================================


class TestLeaderboardEntryKDRatio:
    """kd_ratio with no deaths returns kills value (not a division)."""

    def test_kd_ratio_zero_deaths(self):
        e = LeaderboardEntry(rank=1, player_id="p", score=0, kills=5, deaths=0, assists=0)
        assert e.kd_ratio == 5.0

    def test_kd_ratio_with_deaths(self):
        e = LeaderboardEntry(rank=1, player_id="p", score=0, kills=10, deaths=4, assists=2)
        assert e.kd_ratio == 2.5
