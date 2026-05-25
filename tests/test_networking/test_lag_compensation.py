"""
Tests for lag compensation systems.

Tests cover:
- Rewind manager operations
- World state history
- Hitbox history tracking
- View time calculations
- Hit detection at historical positions
"""

import pytest
import math
from engine.networking.lag_compensation.rewind_manager import (
    HistoryFrame,
    RewindManager,
    WorldState,
    EntityState,
)
from engine.networking.lag_compensation.hitbox_history import (
    HitboxSnapshot,
    HitboxHistory,
    Bounds,
)
from engine.networking.lag_compensation.view_time import (
    ViewTimeCalculator,
    ViewTimeConfig,
    calculate_client_view_time,
    LagCompensationValidator,
)


class TestBounds:
    """Tests for Bounds class."""

    def test_bounds_creation(self):
        """Test bounds initialization."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(1.0, 1.0, 1.0),
        )
        assert bounds.min_point == (0.0, 0.0, 0.0)
        assert bounds.max_point == (1.0, 1.0, 1.0)

    def test_center(self):
        """Test center calculation."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(10.0, 10.0, 10.0),
        )
        center = bounds.center
        assert center == (5.0, 5.0, 5.0)

    def test_size(self):
        """Test size calculation."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(2.0, 4.0, 6.0),
        )
        size = bounds.size
        assert size == (2.0, 4.0, 6.0)

    def test_extents(self):
        """Test extents (half-size) calculation."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(2.0, 4.0, 6.0),
        )
        extents = bounds.extents
        assert extents == (1.0, 2.0, 3.0)

    def test_contains_point(self):
        """Test point containment."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(10.0, 10.0, 10.0),
        )

        assert bounds.contains_point((5.0, 5.0, 5.0))
        assert bounds.contains_point((0.0, 0.0, 0.0))
        assert bounds.contains_point((10.0, 10.0, 10.0))
        assert not bounds.contains_point((11.0, 5.0, 5.0))
        assert not bounds.contains_point((-1.0, 5.0, 5.0))

    def test_intersects(self):
        """Test bounds intersection."""
        bounds1 = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(10.0, 10.0, 10.0),
        )
        bounds2 = Bounds(
            min_point=(5.0, 5.0, 5.0),
            max_point=(15.0, 15.0, 15.0),
        )
        bounds3 = Bounds(
            min_point=(20.0, 20.0, 20.0),
            max_point=(30.0, 30.0, 30.0),
        )

        assert bounds1.intersects(bounds2)
        assert not bounds1.intersects(bounds3)

    def test_translated(self):
        """Test bounds translation."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(1.0, 1.0, 1.0),
        )

        translated = bounds.translated((10.0, 5.0, 0.0))
        assert translated.min_point == (10.0, 5.0, 0.0)
        assert translated.max_point == (11.0, 6.0, 1.0)

    def test_from_center_extents(self):
        """Test creating bounds from center and extents."""
        bounds = Bounds.from_center_extents(
            center=(5.0, 5.0, 5.0),
            extents=(2.0, 2.0, 2.0),
        )

        assert bounds.min_point == (3.0, 3.0, 3.0)
        assert bounds.max_point == (7.0, 7.0, 7.0)


class TestEntityState:
    """Tests for EntityState class."""

    def test_entity_state_creation(self):
        """Test entity state initialization."""
        state = EntityState(
            entity_id=1,
            position=(10.0, 0.0, 5.0),
            velocity=(1.0, 0.0, 0.0),
        )
        assert state.entity_id == 1
        assert state.position == (10.0, 0.0, 5.0)
        assert state.velocity == (1.0, 0.0, 0.0)

    def test_entity_state_copy(self):
        """Test entity state copying."""
        state = EntityState(
            entity_id=1,
            position=(10.0, 0.0, 5.0),
            custom_data={"health": 100},
        )

        copied = state.copy()
        assert copied.entity_id == state.entity_id
        assert copied.position == state.position
        assert copied.custom_data == state.custom_data
        assert copied.custom_data is not state.custom_data  # Deep copy


class TestWorldState:
    """Tests for WorldState class."""

    def test_world_state_creation(self):
        """Test world state initialization."""
        world = WorldState(timestamp=1.0, tick=60)
        assert world.timestamp == 1.0
        assert world.tick == 60
        assert len(world.entities) == 0

    def test_add_entity(self):
        """Test adding entities."""
        world = WorldState()

        state = EntityState(entity_id=1, position=(0.0, 0.0, 0.0))
        world.add_entity(state)

        assert len(world.entities) == 1
        assert world.get_entity(1) is not None

    def test_get_entity(self):
        """Test entity lookup."""
        world = WorldState()

        state = EntityState(entity_id=42, position=(10.0, 20.0, 30.0))
        world.add_entity(state)

        retrieved = world.get_entity(42)
        assert retrieved is not None
        assert retrieved.position == (10.0, 20.0, 30.0)

        assert world.get_entity(999) is None

    def test_remove_entity(self):
        """Test entity removal."""
        world = WorldState()

        state = EntityState(entity_id=1, position=(0.0, 0.0, 0.0))
        world.add_entity(state)

        assert world.remove_entity(1)
        assert world.get_entity(1) is None
        assert not world.remove_entity(1)  # Already removed

    def test_world_state_copy(self):
        """Test world state deep copy."""
        world = WorldState(timestamp=1.0, tick=60)
        world.add_entity(EntityState(entity_id=1, position=(1.0, 0.0, 0.0)))
        world.add_entity(EntityState(entity_id=2, position=(2.0, 0.0, 0.0)))

        copied = world.copy()
        assert copied.timestamp == world.timestamp
        assert copied.tick == world.tick
        assert len(copied.entities) == 2
        assert copied.entities is not world.entities


class TestRewindManager:
    """Tests for RewindManager class."""

    def test_manager_creation(self):
        """Test manager initialization."""
        manager = RewindManager(max_history_ms=200.0, tick_rate=60.0)
        assert manager.max_history_ms == 200.0
        assert manager.frame_count == 0
        assert not manager.is_rewound

    def test_record_frame(self):
        """Test recording world state frames."""
        manager = RewindManager()

        world = WorldState(timestamp=0.0, tick=0)
        world.add_entity(EntityState(entity_id=1, position=(0.0, 0.0, 0.0)))

        manager.record_frame(tick=0, world_state=world)
        assert manager.frame_count == 1

    def test_get_frame_at_time(self):
        """Test retrieving frame by timestamp."""
        manager = RewindManager()

        # Record several frames
        for i in range(10):
            world = WorldState(timestamp=float(i) * 0.016, tick=i)
            world.add_entity(EntityState(
                entity_id=1,
                position=(float(i), 0.0, 0.0),
            ))
            manager.record_frame(tick=i, world_state=world)

        # Get frame at specific time
        frame = manager.get_frame_at_time(0.08)
        assert frame is not None
        assert 4 <= frame.tick <= 6  # Should be close to tick 5

    def test_get_frame_at_tick(self):
        """Test retrieving frame by tick number."""
        manager = RewindManager()

        for i in range(10):
            world = WorldState(timestamp=float(i) * 0.016, tick=i)
            manager.record_frame(tick=i, world_state=world)

        frame = manager.get_frame_at_tick(5)
        assert frame is not None
        assert frame.tick == 5

        assert manager.get_frame_at_tick(999) is None

    def test_rewind_to(self):
        """Test rewinding to a timestamp."""
        manager = RewindManager()

        for i in range(10):
            world = WorldState(timestamp=float(i) * 0.016, tick=i)
            world.add_entity(EntityState(
                entity_id=1,
                position=(float(i), 0.0, 0.0),
            ))
            manager.record_frame(tick=i, world_state=world)

        # Rewind to middle
        rewound_state = manager.rewind_to(0.08)
        assert rewound_state is not None
        assert manager.is_rewound

        # Check position is historical
        entity = rewound_state.get_entity(1)
        assert entity is not None
        assert entity.position[0] < 9.0  # Should be earlier position

    def test_restore_to_current(self):
        """Test restoring after rewind."""
        manager = RewindManager()

        world = WorldState(timestamp=1.0, tick=60)
        world.add_entity(EntityState(entity_id=1, position=(100.0, 0.0, 0.0)))
        manager.record_frame(tick=60, world_state=world)

        manager.rewind_to(0.5)
        assert manager.is_rewound

        manager.restore_to_current()
        assert not manager.is_rewound

    def test_double_rewind_error(self):
        """Test that double rewind raises error."""
        manager = RewindManager()

        world = WorldState(timestamp=1.0, tick=60)
        manager.record_frame(tick=60, world_state=world)

        manager.rewind_to(0.5)

        with pytest.raises(RuntimeError):
            manager.rewind_to(0.3)

    def test_get_interpolated_frame(self):
        """Test interpolated frame retrieval."""
        manager = RewindManager()

        # Frame at tick 0
        world0 = WorldState(timestamp=0.0, tick=0)
        world0.add_entity(EntityState(entity_id=1, position=(0.0, 0.0, 0.0)))
        manager.record_frame(tick=0, world_state=world0)

        # Frame at tick 1
        world1 = WorldState(timestamp=1.0, tick=1)
        world1.add_entity(EntityState(entity_id=1, position=(10.0, 0.0, 0.0)))
        manager.record_frame(tick=1, world_state=world1)

        # Get interpolated frame at middle
        frame = manager.get_interpolated_frame(0.5)
        assert frame is not None

        entity = frame.world_state.get_entity(1)
        assert entity is not None
        # Should be interpolated to around x=5
        assert 4.0 < entity.position[0] < 6.0

    def test_get_entity_at_time(self):
        """Test single entity lookup at time."""
        manager = RewindManager()

        for i in range(10):
            world = WorldState(timestamp=float(i), tick=i)
            world.add_entity(EntityState(
                entity_id=1,
                position=(float(i * 10), 0.0, 0.0),
            ))
            manager.record_frame(tick=i, world_state=world)

        entity = manager.get_entity_at_time(entity_id=1, timestamp=5.0)
        assert entity is not None
        assert abs(entity.position[0] - 50.0) < 1.0

    def test_history_time_range(self):
        """Test getting history time range."""
        manager = RewindManager()

        for i in range(10):
            world = WorldState(timestamp=float(i), tick=i)
            manager.record_frame(tick=i, world_state=world)

        oldest, newest = manager.get_history_time_range()
        assert oldest == 0.0
        assert newest == 9.0

    def test_can_rewind_to(self):
        """Test checking if rewind is possible."""
        manager = RewindManager()

        for i in range(10):
            world = WorldState(timestamp=float(i), tick=i)
            manager.record_frame(tick=i, world_state=world)

        assert manager.can_rewind_to(5.0)
        assert manager.can_rewind_to(0.0)
        assert manager.can_rewind_to(9.0)
        assert not manager.can_rewind_to(-10.0)
        assert not manager.can_rewind_to(100.0)


class TestHitboxSnapshot:
    """Tests for HitboxSnapshot class."""

    def test_snapshot_creation(self):
        """Test snapshot initialization."""
        snapshot = HitboxSnapshot(
            entity_id=1,
            position=(10.0, 0.0, 5.0),
            bounds=Bounds(min_point=(-1.0, 0.0, -1.0), max_point=(1.0, 2.0, 1.0)),
            timestamp=1.0,
        )

        assert snapshot.entity_id == 1
        assert snapshot.position == (10.0, 0.0, 5.0)
        assert snapshot.is_active

    def test_get_world_bounds(self):
        """Test world space bounds calculation."""
        snapshot = HitboxSnapshot(
            entity_id=1,
            position=(10.0, 0.0, 0.0),
            bounds=Bounds(min_point=(-1.0, 0.0, -1.0), max_point=(1.0, 2.0, 1.0)),
            timestamp=1.0,
        )

        world_bounds = snapshot.get_world_bounds()
        assert world_bounds.min_point == (9.0, 0.0, -1.0)
        assert world_bounds.max_point == (11.0, 2.0, 1.0)


class TestHitboxHistory:
    """Tests for HitboxHistory class."""

    def test_history_creation(self):
        """Test history initialization."""
        history = HitboxHistory(max_frames=60)
        assert history.max_frames == 60
        assert history.entity_count == 0

    def test_record(self):
        """Test recording hitbox snapshots."""
        history = HitboxHistory()

        bounds = Bounds(min_point=(-1.0, 0.0, -1.0), max_point=(1.0, 2.0, 1.0))
        history.record(
            entity_id=1,
            position=(0.0, 0.0, 0.0),
            bounds=bounds,
            timestamp=0.0,
        )

        assert history.entity_count == 1
        assert history.has_entity(1)

    def test_get_hitbox_at_time(self):
        """Test hitbox lookup by time."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        # Record several snapshots
        for i in range(10):
            history.record(
                entity_id=1,
                position=(float(i * 10), 0.0, 0.0),
                bounds=bounds,
                timestamp=float(i),
            )

        # Lookup at specific time
        snapshot = history.get_hitbox_at_time(entity_id=1, timestamp=5.0)
        assert snapshot is not None
        assert abs(snapshot.position[0] - 50.0) < 1.0

    def test_get_hitbox_at_tick(self):
        """Test hitbox lookup by tick."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        for i in range(10):
            history.set_tick(i, float(i))
            history.record(
                entity_id=1,
                position=(float(i * 10), 0.0, 0.0),
                bounds=bounds,
            )

        snapshot = history.get_hitbox_at_tick(entity_id=1, tick=5)
        assert snapshot is not None
        assert snapshot.tick == 5

    def test_get_all_hitboxes_at_time(self):
        """Test getting all hitboxes at a time."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        # Record multiple entities
        for i in range(5):
            history.record(
                entity_id=i,
                position=(float(i * 10), 0.0, 0.0),
                bounds=bounds,
                timestamp=1.0,
            )

        snapshots = history.get_all_hitboxes_at_time(timestamp=1.0)
        assert len(snapshots) == 5

    def test_active_only_filter(self):
        """Test filtering for active hitboxes."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        history.record(entity_id=1, position=(0.0, 0.0, 0.0), bounds=bounds,
                      timestamp=1.0, is_active=True)
        history.record(entity_id=2, position=(10.0, 0.0, 0.0), bounds=bounds,
                      timestamp=1.0, is_active=False)

        active = history.get_all_hitboxes_at_time(timestamp=1.0, active_only=True)
        assert len(active) == 1
        assert active[0].entity_id == 1

        all_boxes = history.get_all_hitboxes_at_time(timestamp=1.0, active_only=False)
        assert len(all_boxes) == 2

    def test_get_interpolated_hitbox(self):
        """Test interpolated hitbox retrieval."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        history.record(entity_id=1, position=(0.0, 0.0, 0.0), bounds=bounds, timestamp=0.0)
        history.record(entity_id=1, position=(10.0, 0.0, 0.0), bounds=bounds, timestamp=1.0)

        # Get interpolated at middle
        snapshot = history.get_interpolated_hitbox(entity_id=1, timestamp=0.5)
        assert snapshot is not None
        assert 4.0 < snapshot.position[0] < 6.0

    def test_remove_entity(self):
        """Test entity removal."""
        history = HitboxHistory()
        bounds = Bounds()

        history.record(entity_id=1, position=(0.0, 0.0, 0.0), bounds=bounds, timestamp=0.0)
        assert history.has_entity(1)

        assert history.remove_entity(1)
        assert not history.has_entity(1)
        assert not history.remove_entity(1)

    def test_max_frames_respected(self):
        """Test that max frames is enforced."""
        history = HitboxHistory(max_frames=5)
        bounds = Bounds()

        for i in range(20):
            history.record(
                entity_id=1,
                position=(float(i), 0.0, 0.0),
                bounds=bounds,
                timestamp=float(i),
            )

        # Should only keep last 5 snapshots
        snapshot = history.get_hitbox_at_time(entity_id=1, timestamp=0.0)
        # First snapshots should be gone, so closest should be around 15
        assert snapshot.timestamp >= 14.0


class TestViewTimeCalculation:
    """Tests for view time calculation."""

    def test_calculate_client_view_time(self):
        """Test basic view time calculation."""
        server_time = 10.0
        rtt = 0.1  # 100ms
        interp_delay = 0.1  # 100ms

        view_time = calculate_client_view_time(server_time, rtt, interp_delay)

        # Should be server_time - rtt/2 - interp_delay
        expected = 10.0 - 0.05 - 0.1
        assert abs(view_time - expected) < 0.001


class TestViewTimeCalculator:
    """Tests for ViewTimeCalculator class."""

    def test_calculator_creation(self):
        """Test calculator initialization."""
        calc = ViewTimeCalculator(client_id=1)
        assert calc.client_id == 1
        assert calc.average_rtt == 0.0

    def test_add_rtt_sample(self):
        """Test adding RTT samples."""
        calc = ViewTimeCalculator()

        calc.add_rtt_sample(100.0)
        calc.add_rtt_sample(120.0)
        calc.add_rtt_sample(110.0)

        assert abs(calc.average_rtt - 110.0) < 0.1
        assert calc.min_rtt == 100.0
        assert calc.max_rtt == 120.0

    def test_jitter_calculation(self):
        """Test jitter (RTT variance) calculation."""
        calc = ViewTimeCalculator()

        # Add samples with varying RTT
        for rtt in [100.0, 110.0, 90.0, 120.0, 80.0]:
            calc.add_rtt_sample(rtt)

        assert calc.jitter > 0  # Should have some variance
        assert calc.rtt_variance > 0

    def test_get_interpolated_view_time(self):
        """Test interpolated view time calculation."""
        calc = ViewTimeCalculator()

        for _ in range(10):
            calc.add_rtt_sample(100.0)  # 100ms RTT

        server_time = 10.0
        view_time = calc.get_interpolated_view_time(server_time)

        # Should account for RTT/2 + interpolation delay
        assert view_time < server_time
        assert view_time > server_time - 0.5  # Not too far back

    def test_conservative_vs_liberal_view_time(self):
        """Test conservative and liberal view time estimates."""
        calc = ViewTimeCalculator()

        # Add samples with variance
        for rtt in [80.0, 100.0, 120.0, 100.0, 90.0]:
            calc.add_rtt_sample(rtt)

        server_time = 10.0

        conservative = calc.get_conservative_view_time(server_time)
        liberal = calc.get_liberal_view_time(server_time)

        # Liberal should be further in the past (more compensation)
        assert liberal < conservative

    def test_view_time_range(self):
        """Test view time range calculation."""
        calc = ViewTimeCalculator()

        for rtt in [80.0, 100.0, 120.0]:
            calc.add_rtt_sample(rtt)

        server_time = 10.0
        conservative, liberal = calc.get_view_time_range(server_time)

        assert conservative > liberal
        assert conservative < server_time
        assert liberal < server_time

    def test_compensation_limit(self):
        """Test max lag compensation limit."""
        config = ViewTimeConfig(max_lag_compensation_ms=100.0)
        calc = ViewTimeCalculator(config=config)

        # Add very high RTT
        for _ in range(10):
            calc.add_rtt_sample(500.0)  # 500ms RTT

        server_time = 10.0
        view_time = calc.get_interpolated_view_time(server_time)

        # Should be clamped
        assert view_time >= server_time - 0.1  # Max 100ms compensation

    def test_reset(self):
        """Test resetting the calculator."""
        calc = ViewTimeCalculator()

        for _ in range(10):
            calc.add_rtt_sample(100.0)

        calc.reset()

        assert calc.average_rtt == 0.0
        assert len(calc.rtt_history) == 0


class TestLagCompensationValidator:
    """Tests for LagCompensationValidator class."""

    def test_validator_creation(self):
        """Test validator initialization."""
        validator = LagCompensationValidator()
        assert not validator.is_suspicious(1)

    def test_register_client(self):
        """Test client registration."""
        validator = LagCompensationValidator()

        calc = validator.register_client(client_id=1)
        assert calc is not None
        assert calc.client_id == 1

    def test_validate_good_claim(self):
        """Test validating a reasonable view time claim."""
        validator = LagCompensationValidator()
        calc = validator.register_client(client_id=1)

        # Add RTT samples
        for _ in range(10):
            calc.add_rtt_sample(100.0)

        server_time = 10.0
        expected_view_time = calc.get_interpolated_view_time(server_time)

        is_valid, _ = validator.validate_view_time_claim(
            client_id=1,
            claimed_view_time=expected_view_time,
            server_time=server_time,
        )

        assert is_valid

    def test_validate_suspicious_claim(self):
        """Test validating a suspicious view time claim."""
        validator = LagCompensationValidator(max_deviation_ms=50.0)
        calc = validator.register_client(client_id=1)

        for _ in range(10):
            calc.add_rtt_sample(100.0)

        server_time = 10.0

        # Claim a view time way in the past
        is_valid, corrected = validator.validate_view_time_claim(
            client_id=1,
            claimed_view_time=server_time - 1.0,  # 1 second back
            server_time=server_time,
        )

        assert not is_valid
        assert corrected > server_time - 1.0  # Should be corrected

    def test_suspicious_threshold(self):
        """Test suspicious client detection."""
        validator = LagCompensationValidator(
            max_deviation_ms=10.0,
            suspicious_threshold=3,
        )
        calc = validator.register_client(client_id=1)

        for _ in range(5):
            calc.add_rtt_sample(100.0)

        server_time = 10.0

        # Make multiple bad claims
        for _ in range(3):
            validator.validate_view_time_claim(
                client_id=1,
                claimed_view_time=server_time - 5.0,  # Very bad claim
                server_time=server_time,
            )

        assert validator.is_suspicious(1)


class TestEdgeCases:
    """Edge case tests for lag compensation."""

    def test_empty_rewind_manager(self):
        """Test operations on empty rewind manager."""
        manager = RewindManager()

        assert manager.get_frame_at_time(1.0) is None
        assert manager.get_frame_at_tick(0) is None
        assert manager.oldest_timestamp is None
        assert manager.newest_timestamp is None

    def test_empty_hitbox_history(self):
        """Test operations on empty hitbox history."""
        history = HitboxHistory()

        assert history.get_hitbox_at_time(1, 1.0) is None
        assert history.get_all_hitboxes_at_time(1.0) == []

    def test_view_time_no_samples(self):
        """Test view time calculation with no RTT samples."""
        calc = ViewTimeCalculator()

        view_time = calc.get_interpolated_view_time(10.0)
        # Should still work, assuming minimal latency
        assert view_time < 10.0

    def test_single_frame_rewind(self):
        """Test rewind with only one frame."""
        manager = RewindManager()

        world = WorldState(timestamp=5.0, tick=300)
        world.add_entity(EntityState(entity_id=1, position=(50.0, 0.0, 0.0)))
        manager.record_frame(tick=300, world_state=world)

        state = manager.rewind_to(5.0)
        assert state is not None
        entity = state.get_entity(1)
        assert entity.position == (50.0, 0.0, 0.0)

    def test_bounds_edge_intersection(self):
        """Test bounds intersection at edges."""
        bounds1 = Bounds(min_point=(0.0, 0.0, 0.0), max_point=(10.0, 10.0, 10.0))
        bounds2 = Bounds(min_point=(10.0, 0.0, 0.0), max_point=(20.0, 10.0, 10.0))

        # Touching at edge should intersect
        assert bounds1.intersects(bounds2)


class TestMathCorrectness:
    """Tests to verify mathematical correctness of lag compensation."""

    def test_view_time_formula(self):
        """Verify view time calculation follows correct formula."""
        server_time = 10.0
        rtt = 0.2  # 200ms
        interp_delay = 0.1  # 100ms

        view_time = calculate_client_view_time(server_time, rtt, interp_delay)

        # Expected: server_time - rtt/2 - interp_delay = 10 - 0.1 - 0.1 = 9.8
        expected = server_time - (rtt / 2.0) - interp_delay
        assert abs(view_time - expected) < 0.0001

    def test_rtt_statistics_accuracy(self):
        """Verify RTT statistics are calculated correctly."""
        calc = ViewTimeCalculator()

        samples = [100.0, 110.0, 120.0, 130.0, 140.0]
        for s in samples:
            calc.add_rtt_sample(s)

        # Mean should be 120
        assert abs(calc.average_rtt - 120.0) < 0.001

        # Min/max
        assert calc.min_rtt == 100.0
        assert calc.max_rtt == 140.0

        # Variance calculation (sample variance)
        expected_variance = sum((s - 120.0)**2 for s in samples) / (len(samples) - 1)
        assert abs(calc.rtt_variance - expected_variance) < 0.001

    def test_interpolation_position_accuracy(self):
        """Verify interpolated position is mathematically correct."""
        manager = RewindManager()

        world0 = WorldState(timestamp=0.0, tick=0)
        world0.add_entity(EntityState(entity_id=1, position=(0.0, 0.0, 0.0)))
        manager.record_frame(tick=0, world_state=world0)

        world1 = WorldState(timestamp=1.0, tick=1)
        world1.add_entity(EntityState(entity_id=1, position=(100.0, 0.0, 0.0)))
        manager.record_frame(tick=1, world_state=world1)

        # At t=0.5, position should be exactly 50.0
        frame = manager.get_interpolated_frame(0.5)
        entity = frame.world_state.get_entity(1)
        assert abs(entity.position[0] - 50.0) < 0.001

    def test_bounds_center_calculation(self):
        """Verify bounds center is calculated correctly."""
        bounds = Bounds(
            min_point=(10.0, 20.0, 30.0),
            max_point=(20.0, 40.0, 60.0),
        )

        center = bounds.center
        assert center == (15.0, 30.0, 45.0)

    def test_bounds_size_and_extents(self):
        """Verify bounds size and extents are correct."""
        bounds = Bounds(
            min_point=(0.0, 0.0, 0.0),
            max_point=(10.0, 20.0, 30.0),
        )

        assert bounds.size == (10.0, 20.0, 30.0)
        assert bounds.extents == (5.0, 10.0, 15.0)


class TestLagCompensationBehavior:
    """Tests to verify correct lag compensation behavior."""

    def test_rewind_preserves_entity_order(self):
        """Verify rewind returns entities in correct state."""
        manager = RewindManager()

        # Record moving entity
        for i in range(10):
            world = WorldState(timestamp=float(i), tick=i)
            world.add_entity(EntityState(
                entity_id=1,
                position=(float(i * 10), 0.0, 0.0),
            ))
            manager.record_frame(tick=i, world_state=world)

        # Rewind to timestamp 5 should give position ~50
        state = manager.rewind_to(5.0)
        entity = state.get_entity(1)
        assert 45.0 <= entity.position[0] <= 55.0

        # Must restore before another rewind
        manager.restore_to_current()

    def test_hitbox_interpolation_accuracy(self):
        """Verify hitbox interpolation is accurate."""
        history = HitboxHistory()
        bounds = Bounds.from_center_extents((0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

        history.record(entity_id=1, position=(0.0, 0.0, 0.0), bounds=bounds, timestamp=0.0)
        history.record(entity_id=1, position=(100.0, 0.0, 0.0), bounds=bounds, timestamp=1.0)

        # Interpolate to middle
        snapshot = history.get_interpolated_hitbox(entity_id=1, timestamp=0.5)
        assert snapshot is not None
        assert abs(snapshot.position[0] - 50.0) < 0.001

    def test_max_lag_compensation_enforced(self):
        """Verify max lag compensation limit is enforced."""
        config = ViewTimeConfig(max_lag_compensation_ms=100.0)
        calc = ViewTimeCalculator(config=config)

        # Add very high RTT (500ms)
        for _ in range(10):
            calc.add_rtt_sample(500.0)

        server_time = 10.0
        view_time = calc.get_interpolated_view_time(server_time)

        # Compensation should be clamped to 100ms, not 250ms+ (rtt/2 + interp)
        compensation = server_time - view_time
        assert compensation <= 0.101  # 100ms + small tolerance

    def test_validator_detects_cheating(self):
        """Verify validator detects suspicious view time claims."""
        validator = LagCompensationValidator(
            max_deviation_ms=50.0,
            suspicious_threshold=3,
        )
        calc = validator.register_client(client_id=1)

        # Setup normal RTT
        for _ in range(10):
            calc.add_rtt_sample(100.0)

        server_time = 10.0

        # Make 3 bad claims (claiming to see 1 second in past)
        for _ in range(3):
            is_valid, _ = validator.validate_view_time_claim(
                client_id=1,
                claimed_view_time=server_time - 1.0,  # Way too far back
                server_time=server_time,
            )
            assert not is_valid

        # Should now be flagged as suspicious
        assert validator.is_suspicious(1)

    def test_history_frame_limits(self):
        """Verify history respects frame limits."""
        manager = RewindManager(max_history_ms=100.0, tick_rate=60.0)

        # Record more frames than should fit
        for i in range(20):
            world = WorldState(timestamp=float(i) / 60.0, tick=i)
            manager.record_frame(tick=i, world_state=world)

        # Should have limited frames (100ms at 60fps = ~6 frames + 1)
        assert manager.frame_count <= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
