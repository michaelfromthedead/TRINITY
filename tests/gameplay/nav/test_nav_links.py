"""
Comprehensive tests for off-mesh navigation links.

Tests cover:
- Off-mesh link creation
- Jump links (up, down, across)
- Ladder links
- Door links (open/close state)
- Link traversal costs
- Bidirectional vs unidirectional
- Link validity checks
"""

import math
import pytest
from typing import List, Optional

from engine.gameplay.nav.nav_links import (
    DoorLink,
    DoorState,
    LadderLink,
    LadderParams,
    NavLink,
    NavLinkManager,
    NavLinkParams,
    NavLinkTraversal,
)
from engine.gameplay.nav.navmesh import NavMesh, Vector3
from engine.gameplay.nav.constants import (
    DEFAULT_CLIMB_DURATION,
    DEFAULT_CLIMB_HEIGHT,
    DEFAULT_DROP_DURATION,
    DEFAULT_DROP_HEIGHT,
    DEFAULT_JUMP_DISTANCE,
    DEFAULT_JUMP_DURATION,
    DEFAULT_JUMP_HEIGHT,
    DEFAULT_TELEPORT_COOLDOWN,
    DEFAULT_TELEPORT_DURATION,
    MAX_CLIMB_HEIGHT,
    MAX_DROP_HEIGHT,
    MAX_JUMP_HEIGHT,
    MIN_CLIMB_HEIGHT,
    MIN_DROP_HEIGHT,
    MIN_JUMP_HEIGHT,
    NavLinkDirection,
    NavLinkType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_params():
    """Create default link parameters."""
    return NavLinkParams()


@pytest.fixture
def link_manager():
    """Create link manager without navmesh."""
    return NavLinkManager()


@pytest.fixture
def link_manager_with_navmesh():
    """Create link manager with navmesh."""
    navmesh = NavMesh()
    navmesh.add_polygon([
        Vector3(0, 0, 0),
        Vector3(10, 0, 0),
        Vector3(10, 0, 10),
        Vector3(0, 0, 10)
    ])
    navmesh.add_polygon([
        Vector3(0, 5, 15),
        Vector3(10, 5, 15),
        Vector3(10, 5, 25),
        Vector3(0, 5, 25)
    ])
    return NavLinkManager(navmesh)


# =============================================================================
# NavLinkParams Tests
# =============================================================================


class TestNavLinkParams:
    """Tests for NavLinkParams class."""

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.cost_modifier == 1.0
        assert default_params.traversal_time == 0.5
        assert default_params.enabled
        assert default_params.jump_height == DEFAULT_JUMP_HEIGHT
        assert default_params.drop_height == DEFAULT_DROP_HEIGHT
        assert default_params.climb_height == DEFAULT_CLIMB_HEIGHT

    def test_custom_values(self):
        """Test custom parameter values."""
        params = NavLinkParams(
            cost_modifier=2.0,
            traversal_time=1.0,
            jump_height=3.0
        )
        assert params.cost_modifier == 2.0
        assert params.traversal_time == 1.0
        assert params.jump_height == 3.0

    def test_validate_valid_params(self, default_params):
        """Test validation of valid parameters."""
        assert default_params.validate()

    def test_validate_negative_cost(self):
        """Test validation rejects negative cost."""
        params = NavLinkParams(cost_modifier=-1.0)
        assert not params.validate()

    def test_validate_negative_traversal_time(self):
        """Test validation rejects negative traversal time."""
        params = NavLinkParams(traversal_time=-1.0)
        assert not params.validate()

    def test_validate_jump_height_too_small(self):
        """Test validation rejects too small jump height."""
        params = NavLinkParams(jump_height=MIN_JUMP_HEIGHT - 0.1)
        assert not params.validate()

    def test_validate_jump_height_too_large(self):
        """Test validation rejects too large jump height."""
        params = NavLinkParams(jump_height=MAX_JUMP_HEIGHT + 1.0)
        assert not params.validate()

    def test_validate_drop_height_too_small(self):
        """Test validation rejects too small drop height."""
        params = NavLinkParams(drop_height=MIN_DROP_HEIGHT - 0.1)
        assert not params.validate()

    def test_validate_drop_height_too_large(self):
        """Test validation rejects too large drop height."""
        params = NavLinkParams(drop_height=MAX_DROP_HEIGHT + 1.0)
        assert not params.validate()

    def test_validate_climb_height_too_small(self):
        """Test validation rejects too small climb height."""
        params = NavLinkParams(climb_height=MIN_CLIMB_HEIGHT - 0.1)
        assert not params.validate()

    def test_validate_climb_height_too_large(self):
        """Test validation rejects too large climb height."""
        params = NavLinkParams(climb_height=MAX_CLIMB_HEIGHT + 1.0)
        assert not params.validate()

    def test_required_flags(self):
        """Test required flags."""
        params = NavLinkParams(required_flags=0b0101)
        assert params.required_flags == 5

    def test_excluded_flags(self):
        """Test excluded flags."""
        params = NavLinkParams(excluded_flags=0b1010)
        assert params.excluded_flags == 10

    def test_agent_radius_constraints(self):
        """Test agent radius constraints."""
        params = NavLinkParams(min_agent_radius=0.3, max_agent_radius=1.0)
        assert params.min_agent_radius == 0.3
        assert params.max_agent_radius == 1.0


# =============================================================================
# NavLink Tests
# =============================================================================


class TestNavLink:
    """Tests for NavLink class."""

    def test_construction(self):
        """Test link construction."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 2, 0)
        )
        assert link.id == 1
        assert link.link_type == NavLinkType.JUMP
        assert link.direction == NavLinkDirection.ONE_WAY

    def test_is_bidirectional_one_way(self):
        """Test is_bidirectional for one-way link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.DROP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 5, 0),
            end_position=Vector3(0, 0, 0)
        )
        assert not link.is_bidirectional

    def test_is_bidirectional_two_way(self):
        """Test is_bidirectional for two-way link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        assert link.is_bidirectional

    def test_length(self):
        """Test link length calculation."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(3, 4, 0)
        )
        assert link.length == pytest.approx(5.0)

    def test_height_difference_up(self):
        """Test height difference going up."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        assert link.height_difference == 5.0

    def test_height_difference_down(self):
        """Test height difference going down."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.DROP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 5, 0),
            end_position=Vector3(0, 0, 0)
        )
        assert link.height_difference == -5.0

    def test_get_cost(self):
        """Test cost calculation."""
        params = NavLinkParams(cost_modifier=2.0)
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0),
            params=params
        )
        # cost = base_cost * modifier * length
        assert link.get_cost(1.0) == pytest.approx(10.0)

    def test_can_traverse_enabled(self):
        """Test can_traverse for enabled link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0)
        )
        assert link.can_traverse()

    def test_can_traverse_disabled(self):
        """Test can_traverse for disabled link."""
        params = NavLinkParams(enabled=False)
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0),
            params=params
        )
        assert not link.can_traverse()

    def test_can_traverse_backward_one_way(self):
        """Test can_traverse backward on one-way link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.DROP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 5, 0),
            end_position=Vector3(0, 0, 0)
        )
        assert not link.can_traverse(forward=False)

    def test_can_traverse_backward_two_way(self):
        """Test can_traverse backward on two-way link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        assert link.can_traverse(forward=False)

    def test_can_traverse_required_flags(self):
        """Test can_traverse with required flags."""
        params = NavLinkParams(required_flags=0b0011)
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0),
            params=params
        )
        # Agent has required flags
        assert link.can_traverse(agent_flags=0b0011)
        # Agent missing some flags
        assert not link.can_traverse(agent_flags=0b0001)

    def test_can_traverse_excluded_flags(self):
        """Test can_traverse with excluded flags."""
        params = NavLinkParams(excluded_flags=0b1000)
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0),
            params=params
        )
        # Agent without excluded flag
        assert link.can_traverse(agent_flags=0b0011)
        # Agent with excluded flag
        assert not link.can_traverse(agent_flags=0b1011)

    def test_can_traverse_radius_constraints(self):
        """Test can_traverse with radius constraints."""
        params = NavLinkParams(min_agent_radius=0.3, max_agent_radius=1.0)
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0),
            params=params
        )
        assert link.can_traverse(agent_radius=0.5)
        assert not link.can_traverse(agent_radius=0.2)
        assert not link.can_traverse(agent_radius=1.5)

    def test_get_traversal_position_jump(self):
        """Test traversal position for jump link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(10, 0, 0),
            params=NavLinkParams(jump_height=3.0)
        )
        # At midpoint, should be at max height
        mid_pos = link.get_traversal_position(0.5)
        assert mid_pos.x == pytest.approx(5.0)
        assert mid_pos.y > 0  # Parabolic arc

    def test_get_traversal_position_drop(self):
        """Test traversal position for drop link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.DROP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 10, 0),
            end_position=Vector3(0, 0, 0)
        )
        pos = link.get_traversal_position(0.5)
        assert pos.y > 0 and pos.y < 10

    def test_get_traversal_position_climb(self):
        """Test traversal position for climb link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        mid_pos = link.get_traversal_position(0.5)
        assert mid_pos.y == pytest.approx(2.5)

    def test_get_traversal_position_teleport(self):
        """Test traversal position for teleport link."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.TELEPORT,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(100, 0, 100)
        )
        mid_pos = link.get_traversal_position(0.5)
        assert mid_pos.x == pytest.approx(50.0)
        assert mid_pos.z == pytest.approx(50.0)

    def test_get_traversal_position_backward(self):
        """Test traversal position going backward."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        # Going backward at t=0.5 should be at same position
        pos = link.get_traversal_position(0.5, forward=False)
        assert pos.y == pytest.approx(2.5)

    def test_get_traversal_position_clamped(self):
        """Test traversal position is clamped to 0-1."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        pos_under = link.get_traversal_position(-0.5)
        pos_over = link.get_traversal_position(1.5)

        assert pos_under.y == pytest.approx(0.0)
        assert pos_over.y == pytest.approx(5.0)

    def test_tags(self):
        """Test link tags."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.JUMP,
            direction=NavLinkDirection.ONE_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(5, 0, 0)
        )
        link.tags.add("special")
        link.tags.add("parkour")
        assert "special" in link.tags
        assert "parkour" in link.tags


# =============================================================================
# NavLinkTraversal Tests
# =============================================================================


class TestNavLinkTraversal:
    """Tests for NavLinkTraversal class."""

    def test_construction(self):
        """Test traversal construction."""
        traversal = NavLinkTraversal(
            link_id=1,
            agent_id=10,
            start_time=0.0
        )
        assert traversal.link_id == 1
        assert traversal.agent_id == 10
        assert traversal.progress == 0.0
        assert not traversal.completed

    def test_update_progress(self):
        """Test updating traversal progress."""
        traversal = NavLinkTraversal(link_id=1, agent_id=10, start_time=0.0)

        completed = traversal.update(0.25, duration=1.0)
        assert not completed
        assert traversal.progress == pytest.approx(0.25)

    def test_update_completes(self):
        """Test traversal completion."""
        traversal = NavLinkTraversal(link_id=1, agent_id=10, start_time=0.0)

        completed = traversal.update(1.0, duration=1.0)
        assert completed
        assert traversal.completed
        assert traversal.progress == 1.0

    def test_update_already_completed(self):
        """Test update on already completed traversal."""
        traversal = NavLinkTraversal(link_id=1, agent_id=10, start_time=0.0)
        traversal.update(1.0, duration=1.0)

        # Second update
        completed = traversal.update(0.5, duration=1.0)
        assert completed

    def test_update_zero_duration(self):
        """Test update with zero duration."""
        traversal = NavLinkTraversal(link_id=1, agent_id=10, start_time=0.0)

        completed = traversal.update(0.1, duration=0.0)
        assert completed

    def test_forward_direction(self):
        """Test forward traversal direction."""
        traversal = NavLinkTraversal(
            link_id=1, agent_id=10, start_time=0.0, forward=True
        )
        assert traversal.forward

    def test_backward_direction(self):
        """Test backward traversal direction."""
        traversal = NavLinkTraversal(
            link_id=1, agent_id=10, start_time=0.0, forward=False
        )
        assert not traversal.forward


# =============================================================================
# DoorState Tests
# =============================================================================


class TestDoorState:
    """Tests for DoorState class."""

    def test_default_state(self):
        """Test default door state."""
        state = DoorState()
        assert not state.is_open
        assert not state.is_locked
        assert state.open_progress == 0.0

    def test_open_state(self):
        """Test open door state."""
        state = DoorState(is_open=True, open_progress=1.0)
        assert state.is_open
        assert state.open_progress == 1.0

    def test_locked_state(self):
        """Test locked door state."""
        state = DoorState(is_locked=True, required_key="gold_key")
        assert state.is_locked
        assert state.required_key == "gold_key"

    def test_auto_close_time(self):
        """Test auto close time."""
        state = DoorState(auto_close_time=5.0)
        assert state.auto_close_time == 5.0


# =============================================================================
# DoorLink Tests
# =============================================================================


class TestDoorLink:
    """Tests for DoorLink class."""

    def test_construction(self):
        """Test door link construction."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)
        assert not door.is_open
        assert not door.is_locked

    def test_initial_open(self):
        """Test door created open."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, initial_open=True)
        assert door.is_open

    def test_open_door(self):
        """Test opening door."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)

        result = door.open()
        assert result
        assert door.state.is_open

    def test_open_locked_door_no_key(self):
        """Test opening locked door without key."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, locked=True, required_key="gold_key")

        result = door.open()
        assert not result
        assert door.is_locked

    def test_open_locked_door_with_key(self):
        """Test opening locked door with correct key."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, locked=True, required_key="gold_key")

        result = door.open(key="gold_key")
        assert result
        assert not door.is_locked
        assert door.state.is_open

    def test_open_locked_door_wrong_key(self):
        """Test opening locked door with wrong key."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, locked=True, required_key="gold_key")

        result = door.open(key="silver_key")
        assert not result

    def test_close_door(self):
        """Test closing door."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, initial_open=True)

        result = door.close()
        assert result
        assert not door.state.is_open

    def test_lock_closed_door(self):
        """Test locking closed door."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)

        result = door.lock()
        assert result
        assert door.is_locked

    def test_lock_with_key(self):
        """Test locking door with key."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)

        door.lock(key="new_key")
        assert door.state.required_key == "new_key"

    def test_unlock_door(self):
        """Test unlocking door."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, locked=True)

        result = door.unlock()
        assert result
        assert not door.is_locked

    def test_unlock_with_required_key(self):
        """Test unlocking door that requires key."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, locked=True, required_key="gold_key")

        # Wrong key
        result = door.unlock(key="silver_key")
        assert not result

        # Correct key
        result = door.unlock(key="gold_key")
        assert result

    def test_update_opens_door(self):
        """Test update opens door gradually."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)
        door.open()

        door.update(0.1, open_speed=5.0)
        assert door.state.open_progress > 0

    def test_update_closes_door(self):
        """Test update closes door gradually."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, initial_open=True)
        door.close()

        door.update(0.1, open_speed=5.0)
        assert door.state.open_progress < 1.0

    def test_update_auto_close(self):
        """Test update triggers auto close."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, initial_open=True, auto_close_time=1.0)

        # Simulate time passing
        for _ in range(20):
            door.update(0.1)

        assert not door.state.is_open

    def test_can_traverse_open(self):
        """Test can_traverse when door is open."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link, initial_open=True)

        assert door.can_traverse()

    def test_can_traverse_closed(self):
        """Test can_traverse when door is closed."""
        link = NavLink(
            id=1, link_type=NavLinkType.CUSTOM,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(2, 0, 0)
        )
        door = DoorLink(link)

        assert not door.can_traverse()


# =============================================================================
# LadderParams Tests
# =============================================================================


class TestLadderParams:
    """Tests for LadderParams class."""

    def test_default_values(self):
        """Test default ladder parameters."""
        params = LadderParams()
        assert params.climb_speed == 2.0
        assert params.dismount_height == 0.5
        assert params.requires_hands_free
        assert params.max_agent_width == 1.0

    def test_custom_values(self):
        """Test custom ladder parameters."""
        params = LadderParams(
            climb_speed=3.0,
            dismount_height=0.3,
            requires_hands_free=False
        )
        assert params.climb_speed == 3.0
        assert params.dismount_height == 0.3
        assert not params.requires_hands_free


# =============================================================================
# LadderLink Tests
# =============================================================================


class TestLadderLink:
    """Tests for LadderLink class."""

    def test_construction(self):
        """Test ladder link construction."""
        link = NavLink(
            id=1,
            link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        ladder = LadderLink(link)
        assert ladder.height == 5.0

    def test_height(self):
        """Test ladder height calculation."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 2, 0),
            end_position=Vector3(0, 8, 0)
        )
        ladder = LadderLink(link)
        assert ladder.height == 6.0

    def test_rung_count(self):
        """Test rung count calculation."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 3, 0)
        )
        ladder = LadderLink(link)
        # 3m / 0.3m per rung = 10 rungs
        assert ladder.rung_count == 10

    def test_climb_time(self):
        """Test climb time calculation."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 4, 0)
        )
        params = LadderParams(climb_speed=2.0)
        ladder = LadderLink(link, params)
        assert ladder.climb_time == pytest.approx(2.0)

    def test_get_rung_position(self):
        """Test getting rung position."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 3, 0)
        )
        ladder = LadderLink(link)

        pos = ladder.get_rung_position(5)  # Middle rung
        assert pos.y > 0 and pos.y < 3

    def test_can_mount_from_bottom(self):
        """Test can mount ladder from bottom."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        ladder = LadderLink(link)

        position = Vector3(0.5, 0, 0)
        assert ladder.can_mount(position)

    def test_can_mount_from_top(self):
        """Test can mount ladder from top."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        ladder = LadderLink(link)

        position = Vector3(0.5, 5, 0)
        assert ladder.can_mount(position)

    def test_cannot_mount_too_far(self):
        """Test cannot mount ladder when too far."""
        link = NavLink(
            id=1, link_type=NavLinkType.CLIMB,
            direction=NavLinkDirection.TWO_WAY,
            start_position=Vector3(0, 0, 0),
            end_position=Vector3(0, 5, 0)
        )
        ladder = LadderLink(link)

        position = Vector3(5, 0, 0)  # Too far
        assert not ladder.can_mount(position)


# =============================================================================
# NavLinkManager Tests
# =============================================================================


class TestNavLinkManager:
    """Tests for NavLinkManager class."""

    def test_construction(self, link_manager):
        """Test manager construction."""
        assert link_manager.link_count == 0

    def test_add_link(self, link_manager):
        """Test adding generic link."""
        link_id = link_manager.add_link(
            NavLinkType.JUMP,
            Vector3(0, 0, 0),
            Vector3(5, 2, 0)
        )
        assert link_id > 0
        assert link_manager.link_count == 1

    def test_add_jump_link(self, link_manager):
        """Test adding jump link."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 2, 0),
            jump_height=3.0
        )
        link = link_manager.get_link(link_id)
        assert link.link_type == NavLinkType.JUMP
        assert link.params.jump_height == 3.0

    def test_add_jump_link_bidirectional(self, link_manager):
        """Test adding bidirectional jump link."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 0, 0),
            bidirectional=True
        )
        link = link_manager.get_link(link_id)
        assert link.is_bidirectional

    def test_add_drop_link(self, link_manager):
        """Test adding drop link."""
        link_id = link_manager.add_drop_link(
            Vector3(0, 5, 0),
            Vector3(0, 0, 0)
        )
        link = link_manager.get_link(link_id)
        assert link.link_type == NavLinkType.DROP
        assert not link.is_bidirectional

    def test_add_ladder_link(self, link_manager):
        """Test adding ladder link."""
        link_id = link_manager.add_ladder_link(
            Vector3(0, 0, 0),
            Vector3(0, 5, 0)
        )
        link = link_manager.get_link(link_id)
        assert link.link_type == NavLinkType.CLIMB

        ladder = link_manager.get_ladder(link_id)
        assert ladder is not None

    def test_add_door_link(self, link_manager):
        """Test adding door link."""
        link_id = link_manager.add_door_link(
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            initial_open=False,
            locked=True,
            required_key="gold_key"
        )

        door = link_manager.get_door(link_id)
        assert door is not None
        assert not door.is_open
        assert door.is_locked

    def test_add_teleport_link(self, link_manager):
        """Test adding teleport link."""
        link_id = link_manager.add_teleport_link(
            Vector3(0, 0, 0),
            Vector3(100, 0, 100),
            cooldown=2.0
        )
        link = link_manager.get_link(link_id)
        assert link.link_type == NavLinkType.TELEPORT
        assert link.params.teleport_cooldown == 2.0

    def test_remove_link(self, link_manager):
        """Test removing link."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 0, 0)
        )

        result = link_manager.remove_link(link_id)
        assert result
        assert link_manager.link_count == 0

    def test_remove_nonexistent_link(self, link_manager):
        """Test removing nonexistent link."""
        result = link_manager.remove_link(999)
        assert not result

    def test_get_link(self, link_manager):
        """Test getting link by ID."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 0, 0)
        )

        link = link_manager.get_link(link_id)
        assert link is not None
        assert link.id == link_id

    def test_get_nonexistent_link(self, link_manager):
        """Test getting nonexistent link."""
        link = link_manager.get_link(999)
        assert link is None

    def test_enable_link(self, link_manager):
        """Test enabling/disabling link."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 0, 0)
        )

        link_manager.enable_link(link_id, False)
        link = link_manager.get_link(link_id)
        assert not link.params.enabled

        link_manager.enable_link(link_id, True)
        link = link_manager.get_link(link_id)
        assert link.params.enabled

    def test_find_links_at_position(self, link_manager):
        """Test finding links at position."""
        link_manager.add_jump_link(Vector3(0, 0, 0), Vector3(5, 0, 0))
        link_manager.add_jump_link(Vector3(0, 0, 5), Vector3(5, 0, 5))
        link_manager.add_jump_link(Vector3(20, 0, 20), Vector3(25, 0, 20))

        links = link_manager.find_links_at_position(Vector3(0, 0, 2.5), radius=5.0)
        assert len(links) == 2

    def test_find_links_by_type(self, link_manager):
        """Test finding links by type."""
        link_manager.add_jump_link(Vector3(0, 0, 0), Vector3(5, 0, 0))
        link_manager.add_drop_link(Vector3(0, 5, 5), Vector3(0, 0, 5))
        link_manager.add_jump_link(Vector3(10, 0, 0), Vector3(15, 0, 0))

        jumps = link_manager.find_links_by_type(NavLinkType.JUMP)
        drops = link_manager.find_links_by_type(NavLinkType.DROP)

        assert len(jumps) == 2
        assert len(drops) == 1

    def test_find_links_with_tag(self, link_manager):
        """Test finding links by tag."""
        link_id1 = link_manager.add_jump_link(Vector3(0, 0, 0), Vector3(5, 0, 0))
        link_id2 = link_manager.add_jump_link(Vector3(10, 0, 0), Vector3(15, 0, 0))

        link1 = link_manager.get_link(link_id1)
        link1.tags.add("parkour")

        links = link_manager.find_links_with_tag("parkour")
        assert len(links) == 1
        assert links[0].id == link_id1

    def test_begin_traversal(self, link_manager):
        """Test beginning link traversal."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 0, 0)
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)
        assert traversal_id is not None
        assert link_manager.active_traversal_count == 1

    def test_begin_traversal_closed_door(self, link_manager):
        """Test cannot begin traversal through closed door."""
        link_id = link_manager.add_door_link(
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            initial_open=False
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)
        assert traversal_id is None

    def test_update_traversal(self, link_manager):
        """Test updating traversal."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0)
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)

        completed, position = link_manager.update_traversal(traversal_id, dt=0.25)
        assert not completed
        assert position is not None

    def test_update_traversal_completion(self, link_manager):
        """Test traversal completion."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0)
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)

        # Complete the traversal
        for _ in range(20):
            completed, _ = link_manager.update_traversal(traversal_id, dt=0.1)
            if completed:
                break

        assert link_manager.active_traversal_count == 0

    def test_cancel_traversal(self, link_manager):
        """Test cancelling traversal."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0)
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)

        result = link_manager.cancel_traversal(traversal_id)
        assert result
        assert link_manager.active_traversal_count == 0

    def test_get_traversal(self, link_manager):
        """Test getting active traversal."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(10, 0, 0)
        )

        traversal_id = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)

        traversal = link_manager.get_traversal(traversal_id)
        assert traversal is not None
        assert traversal.link_id == link_id

    def test_update_doors(self, link_manager):
        """Test updating door states."""
        link_id = link_manager.add_door_link(
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            initial_open=True,
            auto_close_time=0.5
        )

        # Simulate time passing
        for _ in range(10):
            link_manager.update(0.1)

        door = link_manager.get_door(link_id)
        assert not door.state.is_open

    def test_validate_link_valid_jump(self, link_manager):
        """Test validating valid jump link."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(5, 1, 0),
            jump_height=2.0
        )

        assert link_manager.validate_link(link_id)

    def test_validate_link_drop_wrong_direction(self, link_manager):
        """Test validating drop link going up."""
        link_id = link_manager.add_link(
            NavLinkType.DROP,
            Vector3(0, 0, 0),
            Vector3(0, 5, 0)  # Going up!
        )

        assert not link_manager.validate_link(link_id)

    def test_validate_nonexistent_link(self, link_manager):
        """Test validating nonexistent link."""
        assert not link_manager.validate_link(999)

    def test_get_links_between_polygons(self, link_manager_with_navmesh):
        """Test getting links between navmesh polygons."""
        link_manager_with_navmesh.add_jump_link(
            Vector3(5, 0, 5),  # In first polygon
            Vector3(5, 5, 20)  # In second polygon
        )

        # Would need polygon IDs to test properly


# =============================================================================
# Integration Tests
# =============================================================================


class TestNavLinksIntegration:
    """Integration tests for nav links."""

    def test_full_traversal_cycle(self, link_manager):
        """Test full link traversal cycle."""
        link_id = link_manager.add_jump_link(
            Vector3(0, 0, 0),
            Vector3(10, 2, 0),
            jump_height=3.0
        )

        # Begin traversal
        traversal_id = link_manager.begin_traversal(
            link_id, agent_id=1, current_time=0.0
        )
        assert traversal_id is not None

        # Update until complete
        positions = []
        for i in range(100):
            completed, pos = link_manager.update_traversal(traversal_id, dt=0.02)
            if pos:
                positions.append(pos)
            if completed:
                break

        # Should have traversed path
        assert len(positions) > 0

        # Final position should be near end
        assert positions[-1].x > 5.0

    def test_door_traversal(self, link_manager):
        """Test door opening and traversal."""
        link_id = link_manager.add_door_link(
            Vector3(0, 0, 0),
            Vector3(2, 0, 0),
            initial_open=False
        )

        # Cannot traverse while closed
        traversal = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)
        assert traversal is None

        # Open door
        door = link_manager.get_door(link_id)
        door.open()

        # Update door to fully open
        for _ in range(20):
            link_manager.update(0.1)

        # Now can traverse
        traversal = link_manager.begin_traversal(link_id, agent_id=1, current_time=0.0)
        assert traversal is not None

    def test_ladder_traversal(self, link_manager):
        """Test ladder traversal."""
        link_id = link_manager.add_ladder_link(
            Vector3(0, 0, 0),
            Vector3(0, 6, 0),
            ladder_params=LadderParams(climb_speed=2.0)
        )

        ladder = link_manager.get_ladder(link_id)
        assert ladder.height == 6.0
        assert ladder.climb_time == pytest.approx(3.0)

        traversal_id = link_manager.begin_traversal(
            link_id, agent_id=1, current_time=0.0
        )
        assert traversal_id is not None
