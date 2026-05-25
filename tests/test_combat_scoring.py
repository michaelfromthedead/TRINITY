"""
Tests for the combat scoring system (engine/gameplay/combat/scoring.py).

Phase 1 T1.1 Scoring System Validation:
  - T-1.1.1: Multi-kill detection (double/triple kill windows, counter reset)
  - T-1.1.2: Killstreak detection
  - T-1.1.3: Assist attribution
  - T-1.1.4: Leaderboard sorting

All tests use deterministic time via patching time.time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.gameplay.combat.scoring import (
    LeaderboardEntry,
    LeaderboardSortKey,
    PlayerStats,
    ScoringSystem,
    ScoreEventType,
    TeamStats,
)
from engine.gameplay.combat.constants import (
    MULTI_KILL_WINDOW,
    MULTI_KILL_NAMES,
    KILLSTREAK_THRESHOLDS,
    ScoringConfig,
)


# =============================================================================
# Helpers
# =============================================================================


def _minimal_scoring() -> ScoringSystem:
    """Return a ScoringSystem with two players. Caller adds more as needed."""
    s = ScoringSystem()
    s.add_player("killer")
    s.add_player("victim")
    return s


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def time_ctrl():
    """Deterministic time control.

    Patches ``engine.gameplay.combat.scoring.time.time`` so all timing
    inside the scoring module returns the value set by the controller.

    Usage::

        def test_something(time_ctrl):
            time_ctrl.set(1000.0)   # baseline
            time_ctrl.advance(2.0)  # +2 seconds
            assert time_ctrl.now() == 1002.0
    """
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
# T-1.1.1  Multi-Kill Detection
# =============================================================================


class TestMultiKillDetection:
    """Double/triple/quad/penta kill windows and counter reset."""

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _three_player_system() -> tuple[ScoringSystem, str, str, str]:
        s = ScoringSystem()
        s.add_player("a")
        s.add_player("b")
        s.add_player("c")
        return s, "a", "b", "c"

    @staticmethod
    def _five_player_system() -> tuple[ScoringSystem, str, str, str, str, str]:
        s = ScoringSystem()
        for pid in "abcde":
            s.add_player(pid)
        return s, "a", "b", "c", "d", "e"

    # -- double kill -------------------------------------------------------

    def test_double_kill_within_window(self, time_ctrl):
        """Two kills within the multi_kill window produce a double kill."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        s.record_kill(a, c)

        st = s.get_player_stats(a)
        assert st.double_kills == 1
        assert st.triple_kills == 0

    def test_double_kill_achievement_in_result(self, time_ctrl):
        """The result dict contains 'double_kill' for the second kill."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        result = s.record_kill(a, c)

        assert "double_kill" in result["achievements"], result["achievements"]

    # -- triple kill -------------------------------------------------------

    def test_triple_kill_within_window(self, time_ctrl):
        """Three fast kills produce a double *and* a triple kill."""
        s, a, b, c = self._three_player_system()
        s.add_player("d")
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        s.record_kill(a, c)
        time_ctrl.advance(1.0)
        s.record_kill(a, "d")

        st = s.get_player_stats(a)
        assert st.double_kills == 1
        assert st.triple_kills == 1

    def test_triple_kill_achievement_in_result(self, time_ctrl):
        """Third kill result contains 'triple_kill'."""
        s, a, b, c = self._three_player_system()
        s.add_player("d")
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(0.5)
        s.record_kill(a, c)
        time_ctrl.advance(0.5)
        result = s.record_kill(a, "d")

        assert "triple_kill" in result["achievements"], result["achievements"]

    # -- quad kill ---------------------------------------------------------

    def test_quad_kill_within_window(self, time_ctrl):
        """Four fast kills produce all tiers up to quad."""
        s, a, b, c, d, e = self._five_player_system()
        time_ctrl.set(1000.0)
        for victim in (b, c, d, e):
            time_ctrl.advance(0.4)
            s.record_kill(a, victim)

        st = s.get_player_stats(a)
        assert st.double_kills == 1
        assert st.triple_kills == 1
        assert st.quad_kills == 1
        assert st.penta_kills == 0

    # -- penta kill --------------------------------------------------------

    def test_penta_kill_within_window(self, time_ctrl):
        """Five fast kills produce all tiers up to penta."""
        s = ScoringSystem()
        for pid in "abcdef":
            s.add_player(pid)
        time_ctrl.set(1000.0)
        for victim in "bcdef":
            time_ctrl.advance(0.3)
            s.record_kill("a", victim)

        st = s.get_player_stats("a")
        assert st.double_kills == 1
        assert st.triple_kills == 1
        assert st.quad_kills == 1
        assert st.penta_kills == 1

    def test_sixth_kill_still_counts_penta(self, time_ctrl):
        """Six or more kills in the same window: penta_kills fire at each
        multi >= 5 (current implementation behaviour)."""
        s = ScoringSystem()
        for pid in "abcdefg":
            s.add_player(pid)
        time_ctrl.set(1000.0)
        for victim in "bcdefg":  # 6 kills
            time_ctrl.advance(0.2)
            s.record_kill("a", victim)

        st = s.get_player_stats("a")
        assert st.double_kills == 1
        assert st.triple_kills == 1
        assert st.quad_kills == 1
        # FIXME: code uses `multi >= 5` so every kill at 5+ increments
        # penta_kills. This is probably a bug (should be `== 5`).
        assert st.penta_kills == 2

    # -- counter reset -----------------------------------------------------

    def test_counter_reset_after_window_expires(self, time_ctrl):
        """After the window expires, the multi-kill counter resets to 1 so the
        next fast pair produces a *new* double kill."""
        s, a, b, c = self._three_player_system()
        s.add_player("d")
        s.add_player("e")
        time_ctrl.set(1000.0)

        # Chain 1: double kill
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        s.record_kill(a, c)

        # Window expires
        time_ctrl.advance(MULTI_KILL_WINDOW + 1.0)

        # Chain 2: another double kill (counter was reset at the third kill's
        # record_kill call because time_since_last > window)
        s.record_kill(a, "d")
        time_ctrl.advance(1.0)
        s.record_kill(a, "e")

        st = s.get_player_stats(a)
        assert st.double_kills == 2, (
            f"Expected 2 double kills, got {st.double_kills}"
        )
        assert st.triple_kills == 0

    def test_just_beyond_window_boundary_no_multi(self, time_ctrl):
        """A kill *just* past the window boundary does NOT count as multi."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(MULTI_KILL_WINDOW + 0.001)
        s.record_kill(a, c)

        st = s.get_player_stats(a)
        assert st.double_kills == 0

    def test_exact_window_boundary_counts_as_multi(self, time_ctrl):
        """A kill *exactly* at the window boundary DOES count as multi
        (``<=`` comparison)."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(MULTI_KILL_WINDOW)
        s.record_kill(a, c)

        st = s.get_player_stats(a)
        assert st.double_kills == 1

    # -- multi-kill handler ------------------------------------------------

    def test_multi_kill_handler_fires(self, time_ctrl):
        """The ``on_multi_kill`` callback fires with (player_id, count)."""
        s, a, b, c = self._three_player_system()
        calls: list[tuple[str, int]] = []

        s.on_multi_kill(lambda pid, cnt: calls.append((pid, cnt)))
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        s.record_kill(a, c)

        assert len(calls) >= 1, "Handler was not called"
        assert calls[0] == (a, 2), f"Expected ({a}, 2), got {calls[0]}"

    # -- killstreak still increments even when no multi --------------------

    def test_killstreak_increments_outside_multi_window(self, time_ctrl):
        """Killstreak increases even when multi-kill window has expired."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(MULTI_KILL_WINDOW + 1.0)
        s.record_kill(a, c)

        st = s.get_player_stats(a)
        assert st.current_killstreak == 2
        assert st.double_kills == 0  # no multi-kill, but streak still counts

    # -- isolation ---------------------------------------------------------

    def test_bystander_unaffected(self, time_ctrl):
        """A different player's kills do not affect the first player's multi."""
        s, a, b, c = self._three_player_system()
        s.add_player("d")
        time_ctrl.set(1000.0)
        s.record_kill(a, b)
        time_ctrl.advance(1.0)
        s.record_kill(a, c)

        s.record_kill(b, "d")  # bystander kill, well after first pair

        st_bystander = s.get_player_stats(b)
        assert st_bystander.double_kills == 0

    # -- death resets multi-kill count -------------------------------------

    def test_death_resets_multi_kill_count(self, time_ctrl):
        """When the killer dies before the next kill, the multi-kill counter
        is reset (death calls ``_multi_kill_count = 0``)."""
        s, a, b, c = self._three_player_system()
        time_ctrl.set(1000.0)
        s.record_kill(a, b)       # _multi_kill_count = 1
        s.record_death(a)          # _multi_kill_count = 0
        time_ctrl.advance(1.0)
        s.record_kill(a, c)       # time_since_last > window → set to 1, no multi

        st = s.get_player_stats(a)
        assert st.double_kills == 0


# =============================================================================
# T-1.1.2  Killstreak Detection
# =============================================================================


class TestKillstreakDetection:
    """Streak increment on kill, streak reset on death, configurable thresholds."""

    def test_streak_increments_on_kill(self):
        """Each kill increments current killstreak by 1."""
        s = _minimal_scoring()
        s.add_player("v2")
        s.record_kill("killer", "victim")
        s.record_kill("killer", "v2")

        st = s.get_player_stats("killer")
        assert st.current_killstreak == 2

    def test_best_killstreak_tracks_peak(self):
        """best_killstreak records the highest streak achieved."""
        s = _minimal_scoring()
        s.add_player("v2")
        s.add_player("v3")
        s.record_kill("killer", "victim")
        s.record_kill("killer", "v2")
        s.record_death("killer")
        s.record_kill("killer", "v3")

        st = s.get_player_stats("killer")
        assert st.best_killstreak == 2
        assert st.current_killstreak == 1  # new streak after death

    def test_streak_resets_on_death(self):
        """Death resets current_killstreak to 0."""
        s = _minimal_scoring()
        s.record_kill("killer", "victim")
        s.record_death("killer")

        st = s.get_player_stats("killer")
        assert st.current_killstreak == 0
        assert st.best_killstreak == 1

    def test_deathstreak_increments_on_death(self):
        """current_deathstreak increments each time the player dies."""
        s = _minimal_scoring()
        s.record_death("killer")
        s.record_death("killer")

        st = s.get_player_stats("killer")
        assert st.current_deathstreak == 2

    def test_kill_resets_deathstreak(self):
        """A kill sets current_deathstreak back to 0."""
        s = _minimal_scoring()
        s.record_death("killer")
        s.record_death("killer")
        s.record_kill("killer", "victim")

        st = s.get_player_stats("killer")
        assert st.current_deathstreak == 0

    def test_killstreak_threshold_achievement(self, time_ctrl):
        """The achievement name for reaching a threshold appears in the
        result dict."""
        s = ScoringSystem()
        s.add_player("a")
        for i in range(10):
            s.add_player(f"v{i}")

        time_ctrl.set(1000.0)
        result = None
        for i in range(3):          # 3 kills → "killing_spree"
            time_ctrl.advance(0.1)
            result = s.record_kill("a", f"v{i}")

        assert result is not None
        assert "killing_spree" in result["achievements"], result["achievements"]

    def test_killstreak_handler_fires(self, time_ctrl):
        """The on_killstreak handler fires when a threshold is reached."""
        s = ScoringSystem()
        s.add_player("a")
        for i in range(10):
            s.add_player(f"v{i}")
        calls: list[tuple[str, int]] = []

        s.on_killstreak(lambda pid, cnt: calls.append((pid, cnt)))
        time_ctrl.set(1000.0)
        for i in range(3):
            time_ctrl.advance(0.1)
            s.record_kill("a", f"v{i}")

        assert len(calls) >= 1
        assert calls[0] == ("a", 3)

    def test_loss_of_killstreak_does_not_affect_other_players(self):
        """One player's death does not reset another player's streak."""
        s = ScoringSystem()
        s.add_player("a")
        s.add_player("b")
        s.add_player("c")
        s.record_kill("a", "b")
        s.record_death("b")          # b loses streak, but they had none

        st_a = s.get_player_stats("a")
        assert st_a.current_killstreak == 1  # unaffected


# =============================================================================
# T-1.1.3  Assist Attribution
# =============================================================================


class TestAssistAttribution:
    """Damage within time window grants assist; threshold check; multi-assist."""

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _system_with_assist_config(
        threshold: float = 0.1,
        window: float = 10.0,
    ) -> ScoringSystem:
        cfg = ScoringConfig(
            assist_damage_threshold=threshold,
            assist_time_window=window,
        )
        s = ScoringSystem(config=cfg)
        s.add_player("killer")
        s.add_player("victim")
        s.add_player("assister")
        s.add_player("low_dmg")
        return s

    # -- basic -------------------------------------------------------------

    def test_damage_within_window_grants_assist(self, time_ctrl):
        """Player dealing >= threshold damage within the assist window gets
        an assist."""
        s = self._system_with_assist_config(threshold=0.1, window=10.0)

        time_ctrl.set(1000.0)
        s.record_damage("assister", "victim", 15.0)  # 15 >= 10
        time_ctrl.advance(1.0)
        result = s.record_kill("killer", "victim")

        assert "assister" in result["assists"], result["assists"]

    def test_assist_points_awarded(self, time_ctrl):
        """Assist player receives points and assist count increments."""
        s = self._system_with_assist_config()
        time_ctrl.set(1000.0)
        s.record_damage("assister", "victim", 15.0)
        time_ctrl.advance(1.0)
        s.record_kill("killer", "victim")

        st = s.get_player_stats("assister")
        assert st.assists == 1, f"Expected 1 assist, got {st.assists}"
        assert st.score > 0, "Assist should award points"

    # -- below threshold ---------------------------------------------------

    def test_damage_below_threshold_no_assist(self, time_ctrl):
        """Player dealing below-threshold damage does NOT get an assist."""
        s = self._system_with_assist_config(threshold=0.1, window=10.0)

        time_ctrl.set(1000.0)
        s.record_damage("low_dmg", "victim", 5.0)  # 5 < 10
        time_ctrl.advance(1.0)
        result = s.record_kill("killer", "victim")

        assert "low_dmg" not in result["assists"], result["assists"]

    # -- outside time window -----------------------------------------------

    def test_damage_outside_window_no_assist(self, time_ctrl):
        """Damage dealt outside the assist window does NOT grant an assist."""
        s = self._system_with_assist_config(threshold=0.1, window=4.0)

        time_ctrl.set(1000.0)
        s.record_damage("assister", "victim", 15.0)
        time_ctrl.advance(5.0)  # past the 4 s window
        result = s.record_kill("killer", "victim")

        assert "assister" not in result["assists"], result["assists"]

    # -- multi-assist ------------------------------------------------------

    def test_multiple_assisters_on_one_kill(self, time_ctrl):
        """Multiple players dealing sufficient damage all get assists."""
        s = ScoringSystem(config=ScoringConfig(assist_damage_threshold=0.1))
        s.add_player("killer")
        s.add_player("victim")
        s.add_player("c")
        s.add_player("d")

        time_ctrl.set(1000.0)
        s.record_damage("c", "victim", 20.0)
        time_ctrl.advance(0.5)
        s.record_damage("d", "victim", 15.0)
        time_ctrl.advance(0.5)
        result = s.record_kill("killer", "victim")

        assert "c" in result["assists"], result["assists"]
        assert "d" in result["assists"], result["assists"]
        assert len(result["assists"]) == 2

    # -- killer exclusion --------------------------------------------------

    def test_killer_not_in_assists(self, time_ctrl):
        """The killer is excluded from the assist list even if they dealt
        damage to the victim."""
        s = self._system_with_assist_config()

        time_ctrl.set(1000.0)
        s.record_damage("killer", "victim", 50.0)
        time_ctrl.advance(1.0)
        result = s.record_kill("killer", "victim")

        assert "killer" not in result["assists"], result["assists"]

    # -- tracking cleared after kill ---------------------------------------

    def test_damage_tracking_cleared_after_kill(self, time_ctrl):
        """After a kill, all damage tracking for the victim is cleared."""
        s = self._system_with_assist_config()
        time_ctrl.set(1000.0)
        s.record_damage("assister", "victim", 15.0)
        time_ctrl.advance(1.0)
        s.record_kill("killer", "victim")

        st = s.get_player_stats("assister")
        assert "victim" not in st._damage_to_targets


# =============================================================================
# T-1.1.4  Leaderboard Sorting
# =============================================================================


class TestLeaderboardSorting:
    """Sort by kills (desc), deaths (asc), score (desc), kd_ratio, assists."""

    @pytest.fixture(autouse=True)
    def _populate(self):
        self.s = ScoringSystem()
        self.s.add_player("pk")   # killer  — most kills
        self.s.add_player("ps")   # survivor — fewest deaths
        self.s.add_player("psc")  # scorer   — highest score
        self.s.add_player("avg")  # average  — middle stats

    def _set(self, pid: str, kills=0, deaths=0, assists=0, score=0):
        st = self.s.get_player_stats(pid)
        st.kills = kills
        st.deaths = deaths
        st.assists = assists
        st.score = score

    def test_sort_by_kills_desc(self):
        self._set("pk", kills=20, score=100)
        self._set("ps", kills=10, score=100)
        self._set("psc", kills=15, score=100)
        self._set("avg", kills=8, score=100)

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.KILLS)
        vals = [e.kills for e in lb]
        assert vals == sorted(vals, reverse=True), vals

    def test_sort_by_deaths_asc(self):
        self._set("pk", deaths=5)
        self._set("ps", deaths=2)
        self._set("psc", deaths=10)
        self._set("avg", deaths=8)

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.DEATHS)
        vals = [e.deaths for e in lb]
        assert vals == sorted(vals), vals            # ascending
        assert lb[0].player_id == "ps", lb[0]

    def test_sort_by_score_desc(self):
        self._set("pk", score=2000)
        self._set("ps", score=1000)
        self._set("psc", score=3000)
        self._set("avg", score=800)

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.SCORE)
        vals = [e.score for e in lb]
        assert vals == sorted(vals, reverse=True), vals
        assert lb[0].player_id == "psc", lb[0]

    def test_sort_by_kd_ratio(self):
        self._set("pk", kills=20, deaths=5)        # 4.0
        self._set("ps", kills=10, deaths=2)        # 5.0
        self._set("psc", kills=15, deaths=10)       # 1.5
        self._set("avg", kills=8, deaths=8)         # 1.0

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.KD_RATIO)
        vals = [e.kd_ratio for e in lb]
        assert vals == sorted(vals, reverse=True), vals
        assert lb[0].player_id == "ps", lb[0]

    def test_sort_by_assists(self):
        self._set("pk", assists=5)
        self._set("ps", assists=2)
        self._set("psc", assists=10)
        self._set("avg", assists=3)

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.ASSISTS)
        vals = [e.assists for e in lb]
        assert vals == sorted(vals, reverse=True), vals

    def test_team_filter(self):
        s = ScoringSystem(is_team_based=True)
        s.add_player("p1", team_id="red")
        s.add_player("p2", team_id="red")
        s.add_player("p3", team_id="blue")
        s.add_player("p4", team_id="blue")

        assert len(s.get_leaderboard()) == 4
        assert len(s.get_leaderboard(team_id="red")) == 2
        assert len(s.get_leaderboard(team_id="blue")) == 2

    def test_ranks_sequential(self):
        self._set("pk", score=200)
        self._set("psc", score=300)

        lb = self.s.get_leaderboard()
        for i, e in enumerate(lb):
            assert e.rank == i + 1, f"Entry {i} rank {e.rank} != {i+1}"

    def test_limit_clips_entries(self):
        self._set("pk", score=100)
        self._set("psc", score=200)

        lb = self.s.get_leaderboard(limit=1)
        assert len(lb) == 1

    def test_kda_ratio_sort(self):
        self._set("pk", kills=20, assists=5, deaths=5)     # 5.0
        self._set("ps", kills=10, assists=10, deaths=2)    # 10.0

        lb = self.s.get_leaderboard(sort_by=LeaderboardSortKey.KDA_RATIO)
        assert lb[0].player_id == "ps", lb[0]

    def test_empty_leaderboard(self):
        s = ScoringSystem()
        assert s.get_leaderboard() == []


# =============================================================================
# PlayerStats unit tests
# =============================================================================


class TestPlayerStats:
    """PlayerStats computed-property and serialisation tests."""

    def test_kd_ratio_no_deaths(self):
        st = PlayerStats(player_id="p", kills=10, deaths=0)
        assert st.kd_ratio == 10.0

    def test_kd_ratio_with_deaths(self):
        st = PlayerStats(player_id="p", kills=10, deaths=5)
        assert st.kd_ratio == 2.0

    def test_kda_ratio_no_deaths(self):
        st = PlayerStats(player_id="p", kills=10, deaths=0, assists=5)
        assert st.kda_ratio == 15.0

    def test_kda_ratio_with_deaths(self):
        st = PlayerStats(player_id="p", kills=10, deaths=5, assists=5)
        assert st.kda_ratio == 3.0

    def test_total_multi_kills(self):
        st = PlayerStats(
            player_id="p",
            double_kills=2,
            triple_kills=1,
            quad_kills=0,
            penta_kills=1,
        )
        assert st.total_multi_kills == 4

    def test_to_dict_round_trip(self):
        st = PlayerStats(
            player_id="p1",
            team_id="red",
            kills=5, deaths=3, assists=2, score=500,
            damage_dealt=100.0, damage_taken=50.0,
            headshots=2,
            current_killstreak=3, best_killstreak=5,
            double_kills=1, triple_kills=0, quad_kills=0, penta_kills=0,
        )
        d = st.to_dict()
        assert d["player_id"] == "p1"
        assert d["team_id"] == "red"
        assert d["kills"] == 5
        assert d["deaths"] == 3
        assert d["score"] == 500
        assert d["kd_ratio"] == 5.0 / 3.0
        assert d["kda_ratio"] == (5 + 2) / 3.0


class TestTeamStats:
    """TeamStats computed-property tests."""

    def test_kd_ratio_no_deaths(self):
        ts = TeamStats(team_id="red", kills=10, deaths=0)
        assert ts.kd_ratio == 10.0

    def test_kd_ratio_with_deaths(self):
        ts = TeamStats(team_id="red", kills=10, deaths=5)
        assert ts.kd_ratio == 2.0

    def test_member_count(self):
        ts = TeamStats(team_id="red", members={"a", "b", "c"})
        assert ts.member_count == 3


class TestLeaderboardEntry:
    """LeaderboardEntry computed-property tests."""

    def test_kd_ratio_no_deaths(self):
        e = LeaderboardEntry(rank=1, player_id="p", score=100, kills=10, deaths=0, assists=0)
        assert e.kd_ratio == 10.0

    def test_kd_ratio_with_deaths(self):
        e = LeaderboardEntry(rank=2, player_id="p", score=100, kills=10, deaths=5, assists=3)
        assert e.kd_ratio == 2.0


# =============================================================================
# Integration tests  (ScoringSystem-level)
# =============================================================================


class TestScoringSystemIntegration:
    """Broader integration checks for the scoring system."""

    def test_kill_event_emitted(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")

        kills = s.get_event_history(event_type=ScoreEventType.KILL)
        assert len(kills) >= 1
        assert kills[-1].player_id == "killer"
        assert kills[-1].target_id == "victim"
        assert kills[-1].points > 0

    def test_death_updates_stats(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        s.record_death("killer")

        st = s.get_player_stats("killer")
        assert st.deaths == 1
        assert st.current_killstreak == 0

    def test_death_with_points_emits_event(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        s.record_death("killer", death_points=-10)

        deaths = s.get_event_history(event_type=ScoreEventType.DEATH)
        assert len(deaths) >= 1

    def test_first_blood_awarded(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")

        assert s.first_blood_awarded is True
        st = s.get_player_stats("killer")
        assert st.first_bloods == 1

    def test_first_blood_only_once(self, time_ctrl):
        s = _minimal_scoring()
        s.add_player("v2")
        time_ctrl.set(1000.0)
        s.record_kill("killer", "victim")
        time_ctrl.advance(1.0)
        s.record_kill("killer", "v2")

        st = s.get_player_stats("killer")
        assert st.first_bloods == 1  # not 2

    def test_revenge_kill(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        # victim kills killer first — populates killer._killed_by
        s.record_kill("victim", "killer")
        time_ctrl.advance(1.0)
        # now killer kills victim = revenge (victim had killed killer before)
        result = s.record_kill("killer", "victim")

        assert "revenge" in result["achievements"], result["achievements"]
        st = s.get_player_stats("killer")
        assert st.revenge_kills == 1

    def test_headshot_bonus(self, time_ctrl):
        s = _minimal_scoring()
        time_ctrl.set(1000.0)
        result = s.record_kill("killer", "victim", is_headshot=True)

        assert "headshot" in result["achievements"], result["achievements"]
        st = s.get_player_stats("killer")
        assert st.headshots == 1

    def test_team_stats_aggregate(self):
        """TeamStats correctly aggregates kills for team members."""
        s = ScoringSystem(is_team_based=True)
        s.add_player("a", team_id="red")
        s.add_player("b", team_id="red")
        s.add_player("c", team_id="blue")
        s.add_player("d", team_id="blue")

        s.record_kill("a", "c")
        s.record_kill("b", "d")

        assert s._team_stats["red"].kills == 2
        assert s._team_stats["blue"].deaths == 2

    def test_add_player_duplicate_id_no_error(self):
        s = _minimal_scoring()
        st1 = s.add_player("killer")
        st2 = s.add_player("killer")  # same id
        assert st1 is st2  # returns existing stats

    def test_remove_player_unknown(self):
        s = _minimal_scoring()
        assert s.remove_player("nonexistent") is False

    def test_record_kill_unknown_killer(self):
        s = _minimal_scoring()
        result = s.record_kill("unknown", "victim")
        assert result["kill_awarded"] is False

    def test_record_kill_unknown_victim(self):
        s = _minimal_scoring()
        result = s.record_kill("killer", "unknown")
        assert result["kill_awarded"] is False

    def test_record_death_unknown(self):
        s = _minimal_scoring()
        assert s.record_death("unknown") is False

    def test_record_damage_unknown(self):
        s = _minimal_scoring()
        assert s.record_damage("unknown", "victim", 10.0) is False

    def test_reset_clears_everything(self):
        s = _minimal_scoring()
        s.record_kill("killer", "victim")
        s.reset()

        assert s.get_player_stats("killer") is None
        assert len(s.get_leaderboard()) == 0
        assert s.first_blood_awarded is False

    def test_summary_after_kill(self):
        s = _minimal_scoring()
        s.record_kill("killer", "victim")
        summary = s.get_summary()

        assert summary["player_count"] == 2
        assert summary["total_kills"] == 1
        assert summary["total_deaths"] == 1

    def test_get_player_rank(self):
        s = _minimal_scoring()
        s.get_player_stats("killer").score = 200
        s.get_player_stats("victim").score = 100

        assert s.get_player_rank("killer") == 1
        assert s.get_player_rank("victim") == 2
        assert s.get_player_rank("nobody") == 0

    def test_get_team_leaderboard(self):
        s = ScoringSystem(is_team_based=True)
        s.add_player("a", team_id="red")
        s.add_player("b", team_id="blue")
        s.record_kill("a", "b")

        lb = s.get_team_leaderboard()
        assert len(lb) == 2
        assert lb[0][0] == "red"  # winning team first

    def test_objective_capture_awards_points(self):
        s = _minimal_scoring()
        s.record_objective_capture("killer", objective_id="obj1")
        st = s.get_player_stats("killer")
        assert st.objectives_captured == 1
        assert st.score > 0

    def test_objective_defend_awards_points(self):
        s = _minimal_scoring()
        s.record_objective_defend("killer", objective_id="obj1")
        st = s.get_player_stats("killer")
        assert st.objectives_defended == 1
        assert st.score > 0

    def test_healing_tracked(self):
        s = _minimal_scoring()
        s.record_healing("killer", "victim", 30.0)
        st = s.get_player_stats("killer")
        assert st.healing_done == 30.0

    def test_damage_taken_tracked(self):
        s = _minimal_scoring()
        s.record_damage("killer", "victim", 25.0)
        st_v = s.get_player_stats("victim")
        assert st_v.damage_taken == 25.0

    def test_set_score(self):
        s = _minimal_scoring()
        assert s.set_score("killer", 500) is True
        assert s.get_player_stats("killer").score == 500

    def test_set_score_unknown(self):
        s = _minimal_scoring()
        assert s.set_score("unknown", 500) is False

    def test_set_player_team_moves_player(self):
        s = ScoringSystem(is_team_based=True)
        s.add_player("p1", team_id="red")
        s.set_player_team("p1", "blue")
        assert s.get_player_stats("p1").team_id == "blue"

    def test_get_all_player_stats(self):
        s = _minimal_scoring()
        all_stats = s.get_all_player_stats()
        assert "killer" in all_stats
        assert "victim" in all_stats
