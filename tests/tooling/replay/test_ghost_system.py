"""
Tests for ghost_system.py - Ghost replay for racing/speedrun comparisons.
"""

import pytest

from engine.tooling.replay.ghost_system import (
    GhostSystem,
    Ghost,
    GhostConfig,
    GhostRenderMode,
    GhostComparison,
    GhostFrame,
)


class TestGhostRenderMode:
    """Tests for GhostRenderMode enum."""

    def test_render_modes_exist(self):
        """Test all render modes exist."""
        assert GhostRenderMode.SOLID
        assert GhostRenderMode.TRANSPARENT
        assert GhostRenderMode.OUTLINE
        assert GhostRenderMode.SILHOUETTE
        assert GhostRenderMode.TRAIL
        assert GhostRenderMode.HIDDEN


class TestGhostConfig:
    """Tests for GhostConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = GhostConfig()
        assert config.render_mode == GhostRenderMode.TRANSPARENT
        assert config.opacity == 0.5
        assert config.time_offset == 0.0
        assert config.visible is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GhostConfig(
            render_mode=GhostRenderMode.OUTLINE,
            opacity=0.7,
            color=(255, 0, 0)
        )
        assert config.render_mode == GhostRenderMode.OUTLINE
        assert config.opacity == 0.7
        assert config.color == (255, 0, 0)


class TestGhostFrame:
    """Tests for GhostFrame dataclass."""

    def test_create_frame(self):
        """Test creating a ghost frame."""
        frame = GhostFrame(
            frame=100,
            timestamp=1.67,
            position=(10.0, 20.0, 30.0),
            rotation=(0.0, 0.0, 0.0, 1.0)
        )
        assert frame.frame == 100
        assert frame.position == (10.0, 20.0, 30.0)

    def test_frame_with_velocity(self):
        """Test frame with velocity."""
        frame = GhostFrame(
            frame=50,
            timestamp=0.83,
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            velocity=(5.0, 0.0, 0.0)
        )
        assert frame.velocity == (5.0, 0.0, 0.0)

    def test_frame_with_animation(self):
        """Test frame with animation state."""
        frame = GhostFrame(
            frame=75,
            timestamp=1.25,
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            animation_state="running"
        )
        assert frame.animation_state == "running"


class TestGhostComparison:
    """Tests for GhostComparison dataclass."""

    def test_create_comparison(self):
        """Test creating a comparison."""
        comparison = GhostComparison(
            ghost_id="ghost_1",
            current_time_difference=2.5,
            total_time_difference=10.0,
            current_distance=5.0,
            closest_approach=1.0,
            furthest_separation=20.0,
            lead_changes=3
        )
        assert comparison.ghost_id == "ghost_1"
        assert comparison.current_time_difference == 2.5
        assert comparison.lead_changes == 3


class TestGhost:
    """Tests for Ghost dataclass."""

    def create_test_frames(self, count: int = 100) -> list[GhostFrame]:
        """Create test ghost frames."""
        return [
            GhostFrame(
                frame=i,
                timestamp=i * 0.016,
                position=(float(i), 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0)
            )
            for i in range(count)
        ]

    def test_create_ghost(self):
        """Test creating a ghost."""
        frames = self.create_test_frames(10)
        ghost = Ghost(
            id="ghost_1",
            name="Test Ghost",
            frames=frames,
            config=GhostConfig()
        )
        assert ghost.id == "ghost_1"
        assert ghost.name == "Test Ghost"
        assert ghost.frame_count == 10

    def test_ghost_duration(self):
        """Test ghost duration property."""
        frames = self.create_test_frames(100)
        ghost = Ghost(
            id="test",
            name="Test",
            frames=frames,
            config=GhostConfig()
        )
        assert ghost.duration == frames[-1].timestamp

    def test_get_frame(self):
        """Test getting frame by number."""
        frames = self.create_test_frames(100)
        ghost = Ghost(
            id="test",
            name="Test",
            frames=frames,
            config=GhostConfig()
        )

        frame = ghost.get_frame(50)
        assert frame is not None
        assert frame.frame == 50

        # Out of range
        frame = ghost.get_frame(200)
        assert frame is None

    def test_get_frame_at_time(self):
        """Test getting frame at specific time."""
        frames = self.create_test_frames(100)
        ghost = Ghost(
            id="test",
            name="Test",
            frames=frames,
            config=GhostConfig()
        )

        frame = ghost.get_frame_at_time(0.5)
        assert frame is not None

    def test_get_interpolated_state(self):
        """Test getting interpolated state."""
        frames = [
            GhostFrame(frame=0, timestamp=0.0, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            GhostFrame(frame=1, timestamp=1.0, position=(10.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        ]
        ghost = Ghost(
            id="test",
            name="Test",
            frames=frames,
            config=GhostConfig()
        )

        state = ghost.get_interpolated_state(0.5)
        assert state is not None
        # Position should be interpolated to (5.0, 0.0, 0.0)
        assert state['position'][0] == pytest.approx(5.0)

    def test_add_checkpoint(self):
        """Test adding checkpoint."""
        frames = self.create_test_frames(100)
        ghost = Ghost(
            id="test",
            name="Test",
            frames=frames,
            config=GhostConfig()
        )

        ghost.add_checkpoint(50, 0.83)
        assert len(ghost.checkpoints) == 1
        assert ghost.checkpoints[0] == (50, 0.83)


class TestGhostSystem:
    """Tests for GhostSystem class."""

    def create_test_frames(self, count: int = 100) -> list[GhostFrame]:
        """Create test ghost frames."""
        return [
            GhostFrame(
                frame=i,
                timestamp=i * 0.016,
                position=(float(i), 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0)
            )
            for i in range(count)
        ]

    def test_create_system(self):
        """Test creating a ghost system."""
        system = GhostSystem()
        assert system.ghost_count == 0
        assert system.active_ghost_count == 0
        assert not system.is_recording

    def test_add_ghost(self):
        """Test adding a ghost."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        ghost = system.add_ghost(frames, name="Test Ghost")

        assert ghost is not None
        assert system.ghost_count == 1

    def test_remove_ghost(self):
        """Test removing a ghost."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        ghost = system.add_ghost(frames)
        assert system.ghost_count == 1

        removed = system.remove_ghost(ghost.id)
        assert removed
        assert system.ghost_count == 0

    def test_get_ghost(self):
        """Test getting a ghost by ID."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        added = system.add_ghost(frames, name="Test")
        found = system.get_ghost(added.id)

        assert found is not None
        assert found.name == "Test"

    def test_activate_deactivate_ghost(self):
        """Test activating and deactivating ghost."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        ghost = system.add_ghost(frames)

        activated = system.activate_ghost(ghost.id)
        assert activated
        assert system.active_ghost_count == 1

        deactivated = system.deactivate_ghost(ghost.id)
        assert deactivated
        assert system.active_ghost_count == 0

    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        system = GhostSystem()

        system.start_recording()
        assert system.is_recording

        # Record some frames
        for i in range(10):
            system.record_frame(
                position=(float(i), 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0)
            )
            system.update(0.016, (float(i), 0.0, 0.0))

        ghost = system.stop_recording()
        assert not system.is_recording
        assert ghost.frame_count > 0

    def test_record_frame(self):
        """Test recording a frame."""
        system = GhostSystem()
        system.start_recording()

        system.record_frame(
            position=(10.0, 20.0, 30.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            velocity=(1.0, 0.0, 0.0),
            animation_state="running"
        )

        ghost = system.stop_recording()
        assert ghost.frame_count == 1

    def test_update_returns_ghost_states(self):
        """Test that update returns ghost states."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.activate_ghost(ghost.id)

        states = system.update(0.016, (0.0, 0.0, 0.0))

        assert ghost.id in states
        assert 'position' in states[ghost.id]

    def test_get_ghost_state(self):
        """Test getting ghost state at time."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.activate_ghost(ghost.id)

        state = system.get_ghost_state(ghost.id)
        assert state is not None

    def test_get_comparison(self):
        """Test getting comparison data."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.activate_ghost(ghost.id)

        # Update to generate comparison
        system.update(0.5, (50.0, 0.0, 0.0))

        comparison = system.get_comparison(ghost.id)
        assert comparison is not None
        assert comparison.ghost_id == ghost.id

    def test_set_ghost_time_offset(self):
        """Test setting ghost time offset."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.set_ghost_time_offset(ghost.id, 1.0)

        assert ghost.config.time_offset == 1.0

    def test_set_ghost_render_mode(self):
        """Test setting ghost render mode."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.set_ghost_render_mode(ghost.id, GhostRenderMode.OUTLINE)

        assert ghost.config.render_mode == GhostRenderMode.OUTLINE

    def test_set_ghost_opacity(self):
        """Test setting ghost opacity."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost = system.add_ghost(frames)
        system.set_ghost_opacity(ghost.id, 0.8)

        assert ghost.config.opacity == 0.8

        # Test clamping
        system.set_ghost_opacity(ghost.id, 1.5)
        assert ghost.config.opacity == 1.0

        system.set_ghost_opacity(ghost.id, -0.5)
        assert ghost.config.opacity == 0.0

    def test_get_best_ghost(self):
        """Test getting best (fastest) ghost."""
        system = GhostSystem()

        # Add ghosts with different times
        fast_frames = [
            GhostFrame(frame=0, timestamp=0.0, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            GhostFrame(frame=60, timestamp=1.0, position=(100.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        ]
        slow_frames = [
            GhostFrame(frame=0, timestamp=0.0, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            GhostFrame(frame=120, timestamp=2.0, position=(100.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        ]

        system.add_ghost(slow_frames, name="Slow")
        system.add_ghost(fast_frames, name="Fast")

        best = system.get_best_ghost()
        assert best is not None
        assert best.name == "Fast"

    def test_iter_ghosts(self):
        """Test iterating over ghosts."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        system.add_ghost(frames, name="Ghost 1")
        system.add_ghost(frames, name="Ghost 2")

        ghosts = list(system.iter_ghosts())
        assert len(ghosts) == 2

    def test_iter_active_ghosts(self):
        """Test iterating over active ghosts."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        ghost1 = system.add_ghost(frames, name="Ghost 1")
        ghost2 = system.add_ghost(frames, name="Ghost 2")

        system.activate_ghost(ghost1.id)

        active = list(system.iter_active_ghosts())
        assert len(active) == 1
        assert active[0].name == "Ghost 1"

    def test_clear_ghosts(self):
        """Test clearing all ghosts."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        system.add_ghost(frames)
        system.add_ghost(frames)

        assert system.ghost_count == 2

        system.clear_ghosts()
        assert system.ghost_count == 0

    def test_serialize_deserialize_ghost(self):
        """Test ghost serialization."""
        system = GhostSystem()
        frames = self.create_test_frames(50)

        ghost = system.add_ghost(frames, name="Serializable")
        ghost.add_checkpoint(25, 0.42)

        serialized = system.serialize_ghost(ghost.id)
        assert serialized is not None

        # Deserialize into new system
        system2 = GhostSystem()
        restored = system2.deserialize_ghost(serialized)

        assert restored.name == "Serializable"
        assert restored.frame_count == 50
        assert len(restored.checkpoints) == 1

    def test_event_callbacks(self):
        """Test event callbacks."""
        added_ghosts = []
        lead_changes = []

        def on_ghost_added(ghost):
            added_ghosts.append(ghost)

        def on_lead_change(data):
            lead_changes.append(data)

        system = GhostSystem()
        system.on('ghost_added', on_ghost_added)
        system.on('lead_change', on_lead_change)

        frames = self.create_test_frames(50)
        system.add_ghost(frames)

        assert len(added_ghosts) == 1

    def test_off_callback(self):
        """Test removing callback."""
        added_ghosts = []

        def on_ghost_added(ghost):
            added_ghosts.append(ghost)

        system = GhostSystem()
        system.on('ghost_added', on_ghost_added)

        frames = self.create_test_frames(50)
        system.add_ghost(frames)
        assert len(added_ghosts) == 1

        system.off('ghost_added', on_ghost_added)
        system.add_ghost(frames)
        assert len(added_ghosts) == 1  # Should not have increased

    def test_multiple_active_ghosts(self):
        """Test multiple active ghosts."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost1 = system.add_ghost(frames, name="Ghost 1")
        ghost2 = system.add_ghost(frames, name="Ghost 2")
        ghost3 = system.add_ghost(frames, name="Ghost 3")

        system.activate_ghost(ghost1.id)
        system.activate_ghost(ghost2.id)
        system.activate_ghost(ghost3.id)

        states = system.update(0.5, (50.0, 0.0, 0.0))

        assert len(states) == 3
        assert ghost1.id in states
        assert ghost2.id in states
        assert ghost3.id in states

    def test_get_all_comparisons(self):
        """Test getting all comparisons."""
        system = GhostSystem()
        frames = self.create_test_frames(100)

        ghost1 = system.add_ghost(frames, name="Ghost 1")
        ghost2 = system.add_ghost(frames, name="Ghost 2")

        system.activate_ghost(ghost1.id)
        system.activate_ghost(ghost2.id)

        system.update(0.5, (50.0, 0.0, 0.0))

        comparisons = system.get_all_comparisons()
        assert len(comparisons) == 2
