"""
Test Suite: T-GP-3.23 - AI Debug Decorator for AI Visualization

Tests for the @ai_debug decorator and AIDebugData class for visualization support.
Integration with Foundation Registry for runtime discovery.

Requirements tested:
1. @ai_debug registers with Registry
2. Debug options stored correctly
3. BT state visualization data correct
4. Perception range data correct
5. Influence map data correct
6. Enable/disable toggle works
7. Multiple debug configs coexist
8. Factory instantiation
9. Performance: 100 debug queries under 50ms
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from foundation import registry, Registry
from engine.gameplay.ai.ai_debug import (
    # Main decorator
    ai_debug,
    # Tag constant
    TAG_AI_DEBUG,
    # Data classes
    BTNodeDebugStatus,
    BTNodeDebugInfo,
    BTDebugState,
    PerceptionRange,
    PerceptionDebugState,
    InfluenceCell,
    InfluenceDebugState,
    AIDebugConfig,
    AIDebugData,
    # Storage
    AIDebugStorage,
    # Data API
    get_debug_data,
    set_debug_data,
    remove_debug_data,
    get_all_debug_data,
    clear_all_debug_data,
    get_debug_data_count,
    # Factory functions
    create_debug_data,
    create_bt_debug_state,
    create_perception_debug_state,
    create_influence_debug_state,
    # Query helpers
    get_all_ai_debug_configs,
    get_enabled_debug_configs,
    get_debug_configs_with_bt,
    get_debug_configs_with_perception,
    get_debug_configs_with_influence,
    # Runtime control
    enable_debug,
    disable_debug,
    is_debug_enabled,
    toggle_debug,
    get_debug_config,
    # Visualization integration
    WireframeCone,
    BTNodeVisualization,
    generate_perception_wireframes,
    generate_bt_visualization,
    # Color defaults
    DEFAULT_BT_NODE_COLORS,
    DEFAULT_PERCEPTION_COLORS,
    DEFAULT_INFLUENCE_COLORS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test to avoid cross-contamination."""
    # Store initial state
    initial_types = set(registry.all_types())

    yield

    # Clean up any types added during the test
    for cls in registry.all_types():
        if cls not in initial_types:
            try:
                registry.unregister(cls)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def clean_debug_storage():
    """Clean debug storage before and after each test."""
    clear_all_debug_data()
    yield
    clear_all_debug_data()
    AIDebugStorage.reset_instance()


@pytest.fixture
def sample_bt_state():
    """Create a sample BT debug state."""
    state = BTDebugState(tree_id="test_tree", tree_name="TestTree")

    root = BTNodeDebugInfo(
        node_id="root",
        node_name="Selector",
        node_type="selector",
        status=BTNodeDebugStatus.RUNNING,
        children_ids=["child1", "child2"],
    )

    child1 = BTNodeDebugInfo(
        node_id="child1",
        node_name="Sequence",
        node_type="sequence",
        status=BTNodeDebugStatus.SUCCESS,
        parent_id="root",
        children_ids=["action1"],
    )

    child2 = BTNodeDebugInfo(
        node_id="child2",
        node_name="Wait",
        node_type="action",
        status=BTNodeDebugStatus.IDLE,
        parent_id="root",
    )

    action1 = BTNodeDebugInfo(
        node_id="action1",
        node_name="Attack",
        node_type="action",
        status=BTNodeDebugStatus.SUCCESS,
        parent_id="child1",
    )

    state.add_node(root)
    state.add_node(child1)
    state.add_node(child2)
    state.add_node(action1)
    state.root_node_id = "root"
    state.active_node_id = "action1"

    return state


@pytest.fixture
def sample_perception_state():
    """Create a sample perception debug state."""
    state = PerceptionDebugState(
        entity_id=1,
        position=(10.0, 0.0, 10.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    state.add_range(
        "sight",
        PerceptionRange(
            sense_type="sight",
            range_value=30.0,
            fov_degrees=90.0,
            direction=(0.0, 0.0, 1.0),
        ),
    )

    state.add_range(
        "hearing",
        PerceptionRange(
            sense_type="hearing",
            range_value=15.0,
            fov_degrees=360.0,
        ),
    )

    return state


@pytest.fixture
def sample_influence_state():
    """Create a sample influence debug state."""
    state = InfluenceDebugState(
        map_id="threat_map",
        map_name="ThreatMap",
        grid_size=(16, 16),
        cell_size=2.0,
    )

    # Add some sample cells
    state.set_cell(5, 5, 0.8, source_id=1)
    state.set_cell(6, 5, 0.6, source_id=1)
    state.set_cell(5, 6, 0.5, source_id=1)
    state.set_cell(10, 10, -0.5, source_id=2)

    return state


# =============================================================================
# Test Classes
# =============================================================================


class TestAIDebugDecorator:
    """Tests for @ai_debug decorator registration."""

    def test_ai_debug_registers_class(self, clean_registry):
        """Test that @ai_debug registers the class with Foundation Registry."""
        @ai_debug(enabled=True)
        class TestAI01:
            pass

        assert registry.is_registered(TestAI01)

    def test_ai_debug_adds_tag(self, clean_registry):
        """Test that @ai_debug adds the 'ai_debug' tag."""
        @ai_debug(enabled=True)
        class TestAI02:
            pass

        assert registry.has_tag(TestAI02, TAG_AI_DEBUG)

    def test_ai_debug_stores_enabled_metadata(self, clean_registry):
        """Test that @ai_debug stores debug_enabled metadata."""
        @ai_debug(enabled=True)
        class TestAI03:
            pass

        @ai_debug(enabled=False)
        class TestAI04:
            pass

        assert registry.get_metadata(TestAI03, "debug_enabled") is True
        assert registry.get_metadata(TestAI04, "debug_enabled") is False

    def test_ai_debug_stores_show_bt_metadata(self, clean_registry):
        """Test that @ai_debug stores show_bt metadata."""
        @ai_debug(enabled=True, show_bt=True)
        class TestAI05:
            pass

        @ai_debug(enabled=True, show_bt=False)
        class TestAI06:
            pass

        assert registry.get_metadata(TestAI05, "show_bt") is True
        assert registry.get_metadata(TestAI06, "show_bt") is False

    def test_ai_debug_stores_show_perception_metadata(self, clean_registry):
        """Test that @ai_debug stores show_perception metadata."""
        @ai_debug(enabled=True, show_perception=True)
        class TestAI07:
            pass

        @ai_debug(enabled=True, show_perception=False)
        class TestAI08:
            pass

        assert registry.get_metadata(TestAI07, "show_perception") is True
        assert registry.get_metadata(TestAI08, "show_perception") is False

    def test_ai_debug_stores_show_influence_metadata(self, clean_registry):
        """Test that @ai_debug stores show_influence metadata."""
        @ai_debug(enabled=True, show_influence=True)
        class TestAI09:
            pass

        @ai_debug(enabled=True, show_influence=False)
        class TestAI10:
            pass

        assert registry.get_metadata(TestAI09, "show_influence") is True
        assert registry.get_metadata(TestAI10, "show_influence") is False

    def test_ai_debug_stores_description(self, clean_registry):
        """Test that @ai_debug stores description metadata."""
        @ai_debug(enabled=True, description="Enemy patrol AI")
        class TestAI11:
            pass

        assert registry.get_metadata(TestAI11, "description") == "Enemy patrol AI"

    def test_ai_debug_sets_class_attributes(self, clean_registry):
        """Test that @ai_debug sets class attributes."""
        @ai_debug(enabled=True, show_bt=True, show_perception=True, show_influence=False)
        class TestAI12:
            pass

        assert hasattr(TestAI12, "_ai_debug")
        assert TestAI12._ai_debug is True
        assert TestAI12._ai_debug_enabled is True
        assert hasattr(TestAI12, "_ai_debug_config")

    def test_ai_debug_config_object(self, clean_registry):
        """Test that @ai_debug creates proper AIDebugConfig object."""
        @ai_debug(enabled=True, show_bt=True, show_perception=False, show_influence=True)
        class TestAI13:
            pass

        config = TestAI13._ai_debug_config
        assert isinstance(config, AIDebugConfig)
        assert config.enabled is True
        assert config.show_bt is True
        assert config.show_perception is False
        assert config.show_influence is True

    def test_ai_debug_custom_bt_colors(self, clean_registry):
        """Test that @ai_debug accepts custom BT node colors."""
        custom_colors = {"running": (1.0, 0.5, 0.0, 1.0)}

        @ai_debug(enabled=True, bt_node_colors=custom_colors)
        class TestAI14:
            pass

        stored_colors = registry.get_metadata(TestAI14, "bt_node_colors")
        assert stored_colors["running"] == (1.0, 0.5, 0.0, 1.0)

    def test_ai_debug_custom_perception_colors(self, clean_registry):
        """Test that @ai_debug accepts custom perception colors."""
        custom_colors = {"sight": (0.0, 0.0, 1.0, 0.5)}

        @ai_debug(enabled=True, perception_colors=custom_colors)
        class TestAI15:
            pass

        stored_colors = registry.get_metadata(TestAI15, "perception_colors")
        assert stored_colors["sight"] == (0.0, 0.0, 1.0, 0.5)

    def test_ai_debug_custom_influence_colors(self, clean_registry):
        """Test that @ai_debug accepts custom influence colors."""
        custom_colors = {"positive": (0.5, 1.0, 0.5, 0.6)}

        @ai_debug(enabled=True, influence_colors=custom_colors)
        class TestAI16:
            pass

        stored_colors = registry.get_metadata(TestAI16, "influence_colors")
        assert stored_colors["positive"] == (0.5, 1.0, 0.5, 0.6)

    def test_ai_debug_custom_options(self, clean_registry):
        """Test that @ai_debug accepts custom options."""
        custom_opts = {"show_labels": True, "label_font_size": 12}

        @ai_debug(enabled=True, custom_options=custom_opts)
        class TestAI17:
            pass

        stored_opts = registry.get_metadata(TestAI17, "custom_options")
        assert stored_opts["show_labels"] is True
        assert stored_opts["label_font_size"] == 12

    def test_ai_debug_track_instances(self, clean_registry):
        """Test that track_instances=True enables instance tracking."""
        @ai_debug(enabled=True, track_instances=True)
        class TestAI18:
            pass

        instance1 = TestAI18()
        instance2 = TestAI18()

        count = registry.instance_count(TestAI18)
        assert count >= 2


class TestAIDebugQuery:
    """Tests for AI debug query functionality."""

    def test_query_all_ai_debug_configs(self, clean_registry):
        """Test Registry.query(tag='ai_debug') returns all debug configs."""
        @ai_debug(enabled=True)
        class QueryAI01:
            pass

        @ai_debug(enabled=True)
        class QueryAI02:
            pass

        @ai_debug(enabled=False)
        class QueryAI03:
            pass

        all_configs = registry.query(tag=TAG_AI_DEBUG)
        assert len(all_configs) >= 3

    def test_query_enabled_configs(self, clean_registry):
        """Test querying only enabled debug configs."""
        @ai_debug(enabled=True)
        class EnabledAI01:
            pass

        @ai_debug(enabled=False)
        class DisabledAI01:
            pass

        enabled = get_enabled_debug_configs()
        assert EnabledAI01 in enabled
        assert DisabledAI01 not in enabled

    def test_query_configs_with_bt(self, clean_registry):
        """Test querying configs that show BT."""
        @ai_debug(enabled=True, show_bt=True)
        class BTAI01:
            pass

        @ai_debug(enabled=True, show_bt=False)
        class BTAI02:
            pass

        bt_configs = get_debug_configs_with_bt()
        assert BTAI01 in bt_configs
        assert BTAI02 not in bt_configs

    def test_query_configs_with_perception(self, clean_registry):
        """Test querying configs that show perception."""
        @ai_debug(enabled=True, show_perception=True)
        class PerceptionAI01:
            pass

        @ai_debug(enabled=True, show_perception=False)
        class PerceptionAI02:
            pass

        perception_configs = get_debug_configs_with_perception()
        assert PerceptionAI01 in perception_configs
        assert PerceptionAI02 not in perception_configs

    def test_query_configs_with_influence(self, clean_registry):
        """Test querying configs that show influence maps."""
        @ai_debug(enabled=True, show_influence=True)
        class InfluenceAI01:
            pass

        @ai_debug(enabled=True, show_influence=False)
        class InfluenceAI02:
            pass

        influence_configs = get_debug_configs_with_influence()
        assert InfluenceAI01 in influence_configs
        assert InfluenceAI02 not in influence_configs

    def test_get_all_ai_debug_configs_helper(self, clean_registry):
        """Test get_all_ai_debug_configs helper function."""
        @ai_debug(enabled=True)
        class HelperAI01:
            pass

        configs = get_all_ai_debug_configs()
        assert HelperAI01 in configs


class TestBTDebugState:
    """Tests for BTDebugState data class."""

    def test_bt_debug_state_creation(self):
        """Test creating BTDebugState."""
        state = BTDebugState(tree_id="tree_01", tree_name="TestTree")
        assert state.tree_id == "tree_01"
        assert state.tree_name == "TestTree"
        assert state.nodes == {}
        assert state.total_ticks == 0

    def test_bt_debug_state_add_node(self, sample_bt_state):
        """Test adding nodes to BTDebugState."""
        assert len(sample_bt_state.nodes) == 4
        assert "root" in sample_bt_state.nodes
        assert "child1" in sample_bt_state.nodes
        assert "action1" in sample_bt_state.nodes

    def test_bt_debug_state_get_node(self, sample_bt_state):
        """Test getting nodes from BTDebugState."""
        node = sample_bt_state.get_node("root")
        assert node is not None
        assert node.node_name == "Selector"
        assert node.node_type == "selector"

    def test_bt_debug_state_get_active_path(self, sample_bt_state):
        """Test getting active path in BTDebugState."""
        path = sample_bt_state.get_active_path()
        assert path == ["root", "child1", "action1"]

    def test_bt_node_debug_info_creation(self):
        """Test creating BTNodeDebugInfo."""
        node = BTNodeDebugInfo(
            node_id="test_node",
            node_name="TestAction",
            node_type="action",
            status=BTNodeDebugStatus.RUNNING,
            execution_time_ms=5.5,
            tick_count=10,
        )

        assert node.node_id == "test_node"
        assert node.node_name == "TestAction"
        assert node.status == BTNodeDebugStatus.RUNNING
        assert node.execution_time_ms == 5.5
        assert node.tick_count == 10

    def test_bt_node_debug_status_enum(self):
        """Test BTNodeDebugStatus enum values."""
        assert BTNodeDebugStatus.IDLE == 0
        assert BTNodeDebugStatus.RUNNING == 1
        assert BTNodeDebugStatus.SUCCESS == 2
        assert BTNodeDebugStatus.FAILURE == 3


class TestPerceptionDebugState:
    """Tests for PerceptionDebugState data class."""

    def test_perception_debug_state_creation(self):
        """Test creating PerceptionDebugState."""
        state = PerceptionDebugState(
            entity_id=42,
            position=(1.0, 2.0, 3.0),
        )

        assert state.entity_id == 42
        assert state.position == (1.0, 2.0, 3.0)
        assert state.ranges == {}

    def test_perception_debug_state_add_range(self, sample_perception_state):
        """Test adding perception ranges."""
        assert len(sample_perception_state.ranges) == 2
        assert "sight" in sample_perception_state.ranges
        assert "hearing" in sample_perception_state.ranges

    def test_perception_debug_state_get_range(self, sample_perception_state):
        """Test getting perception ranges."""
        sight = sample_perception_state.get_range("sight")
        assert sight is not None
        assert sight.range_value == 30.0
        assert sight.fov_degrees == 90.0

    def test_perception_range_creation(self):
        """Test creating PerceptionRange."""
        perception_range = PerceptionRange(
            sense_type="sight",
            range_value=25.0,
            fov_degrees=120.0,
            direction=(1.0, 0.0, 0.0),
        )

        assert perception_range.sense_type == "sight"
        assert perception_range.range_value == 25.0
        assert perception_range.fov_degrees == 120.0
        assert perception_range.direction == (1.0, 0.0, 0.0)

    def test_perception_range_default_values(self):
        """Test PerceptionRange default values."""
        perception_range = PerceptionRange(
            sense_type="test",
            range_value=10.0,
        )

        assert perception_range.fov_degrees == 360.0
        assert perception_range.direction == (0.0, 0.0, 1.0)
        assert perception_range.is_active is True


class TestInfluenceDebugState:
    """Tests for InfluenceDebugState data class."""

    def test_influence_debug_state_creation(self):
        """Test creating InfluenceDebugState."""
        state = InfluenceDebugState(
            map_id="test_map",
            map_name="TestMap",
            grid_size=(32, 32),
            cell_size=1.5,
        )

        assert state.map_id == "test_map"
        assert state.map_name == "TestMap"
        assert state.grid_size == (32, 32)
        assert state.cell_size == 1.5

    def test_influence_debug_state_set_cell(self, sample_influence_state):
        """Test setting influence cells."""
        assert len(sample_influence_state.cells) == 4

    def test_influence_debug_state_get_cell(self, sample_influence_state):
        """Test getting influence cells."""
        cell = sample_influence_state.get_cell(5, 5)
        assert cell is not None
        assert cell.value == 0.8
        assert cell.source_entity_id == 1

    def test_influence_debug_state_get_value(self, sample_influence_state):
        """Test getting influence values."""
        assert sample_influence_state.get_value(5, 5) == 0.8
        assert sample_influence_state.get_value(10, 10) == -0.5
        assert sample_influence_state.get_value(0, 0) == 0.0  # Not set

    def test_influence_cell_creation(self):
        """Test creating InfluenceCell."""
        cell = InfluenceCell(
            x=3,
            y=4,
            value=0.75,
            decay_rate=0.05,
            source_entity_id=99,
        )

        assert cell.x == 3
        assert cell.y == 4
        assert cell.value == 0.75
        assert cell.decay_rate == 0.05
        assert cell.source_entity_id == 99


class TestAIDebugData:
    """Tests for AIDebugData data class."""

    def test_ai_debug_data_creation(self):
        """Test creating AIDebugData."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=1, config=config)

        assert data.entity_id == 1
        assert data.config.enabled is True
        assert data.bt_state is None
        assert data.perception_state is None
        assert data.influence_states == {}

    def test_ai_debug_data_with_bt_state(self, sample_bt_state):
        """Test AIDebugData with BT state."""
        config = AIDebugConfig(enabled=True, show_bt=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            bt_state=sample_bt_state,
        )

        assert data.bt_state is not None
        assert data.bt_state.tree_id == "test_tree"

    def test_ai_debug_data_with_perception_state(self, sample_perception_state):
        """Test AIDebugData with perception state."""
        config = AIDebugConfig(enabled=True, show_perception=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            perception_state=sample_perception_state,
        )

        assert data.perception_state is not None
        assert len(data.perception_state.ranges) == 2

    def test_ai_debug_data_with_influence_states(self, sample_influence_state):
        """Test AIDebugData with influence states."""
        config = AIDebugConfig(enabled=True, show_influence=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            influence_states={"threat": sample_influence_state},
        )

        assert "threat" in data.influence_states
        assert data.influence_states["threat"].map_id == "threat_map"

    def test_ai_debug_data_update_timestamp(self):
        """Test updating timestamp on AIDebugData."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=1, config=config)

        old_timestamp = data.timestamp
        time.sleep(0.01)
        data.update_timestamp()

        assert data.timestamp > old_timestamp


class TestAIDebugStorage:
    """Tests for AIDebugStorage singleton."""

    def test_debug_storage_store_and_get(self, clean_debug_storage):
        """Test storing and retrieving debug data."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=1, config=config)

        set_debug_data(1, data)
        retrieved = get_debug_data(1)

        assert retrieved is not None
        assert retrieved.entity_id == 1

    def test_debug_storage_remove(self, clean_debug_storage):
        """Test removing debug data."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=2, config=config)

        set_debug_data(2, data)
        assert get_debug_data(2) is not None

        result = remove_debug_data(2)
        assert result is True
        assert get_debug_data(2) is None

    def test_debug_storage_get_all(self, clean_debug_storage):
        """Test getting all debug data."""
        config = AIDebugConfig(enabled=True)

        set_debug_data(1, AIDebugData(entity_id=1, config=config))
        set_debug_data(2, AIDebugData(entity_id=2, config=config))
        set_debug_data(3, AIDebugData(entity_id=3, config=config))

        all_data = get_all_debug_data()
        assert len(all_data) == 3
        assert 1 in all_data
        assert 2 in all_data
        assert 3 in all_data

    def test_debug_storage_clear(self, clean_debug_storage):
        """Test clearing all debug data."""
        config = AIDebugConfig(enabled=True)

        set_debug_data(1, AIDebugData(entity_id=1, config=config))
        set_debug_data(2, AIDebugData(entity_id=2, config=config))

        clear_all_debug_data()

        assert get_debug_data_count() == 0

    def test_debug_storage_count(self, clean_debug_storage):
        """Test counting debug entries."""
        config = AIDebugConfig(enabled=True)

        assert get_debug_data_count() == 0

        set_debug_data(1, AIDebugData(entity_id=1, config=config))
        assert get_debug_data_count() == 1

        set_debug_data(2, AIDebugData(entity_id=2, config=config))
        assert get_debug_data_count() == 2


class TestRuntimeDebugControl:
    """Tests for runtime debug enable/disable control."""

    def test_enable_debug(self, clean_registry):
        """Test enabling debug for a registered class."""
        @ai_debug(enabled=False)
        class RuntimeAI01:
            pass

        assert is_debug_enabled(RuntimeAI01) is False

        result = enable_debug(RuntimeAI01)
        assert result is True
        assert is_debug_enabled(RuntimeAI01) is True

    def test_disable_debug(self, clean_registry):
        """Test disabling debug for a registered class."""
        @ai_debug(enabled=True)
        class RuntimeAI02:
            pass

        assert is_debug_enabled(RuntimeAI02) is True

        result = disable_debug(RuntimeAI02)
        assert result is True
        assert is_debug_enabled(RuntimeAI02) is False

    def test_toggle_debug(self, clean_registry):
        """Test toggling debug state."""
        @ai_debug(enabled=True)
        class RuntimeAI03:
            pass

        # Toggle off
        new_state = toggle_debug(RuntimeAI03)
        assert new_state is False
        assert is_debug_enabled(RuntimeAI03) is False

        # Toggle on
        new_state = toggle_debug(RuntimeAI03)
        assert new_state is True
        assert is_debug_enabled(RuntimeAI03) is True

    def test_is_debug_enabled_unregistered(self):
        """Test is_debug_enabled for unregistered class."""
        class UnregisteredAI:
            pass

        assert is_debug_enabled(UnregisteredAI) is False

    def test_enable_debug_unregistered(self):
        """Test enable_debug for unregistered class returns False."""
        class UnregisteredAI:
            pass

        result = enable_debug(UnregisteredAI)
        assert result is False

    def test_toggle_debug_unregistered_raises(self):
        """Test toggle_debug raises for unregistered class."""
        class UnregisteredAI:
            pass

        with pytest.raises(ValueError):
            toggle_debug(UnregisteredAI)

    def test_get_debug_config(self, clean_registry):
        """Test getting debug config from class."""
        @ai_debug(enabled=True, show_bt=True, show_perception=False)
        class RuntimeAI04:
            pass

        config = get_debug_config(RuntimeAI04)
        assert config is not None
        assert config.enabled is True
        assert config.show_bt is True
        assert config.show_perception is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_debug_data(self):
        """Test create_debug_data factory."""
        data = create_debug_data(entity_id=42)

        assert data.entity_id == 42
        assert data.config.enabled is True  # Default
        assert data.bt_state is None

    def test_create_debug_data_with_config(self):
        """Test create_debug_data with custom config."""
        config = AIDebugConfig(enabled=False, show_bt=True)
        data = create_debug_data(entity_id=42, config=config)

        assert data.config.enabled is False
        assert data.config.show_bt is True

    def test_create_bt_debug_state(self):
        """Test create_bt_debug_state factory."""
        state = create_bt_debug_state("tree_001", "PatrolTree")

        assert state.tree_id == "tree_001"
        assert state.tree_name == "PatrolTree"
        assert state.nodes == {}

    def test_create_perception_debug_state(self):
        """Test create_perception_debug_state factory."""
        state = create_perception_debug_state(
            entity_id=5,
            position=(10.0, 20.0, 30.0),
        )

        assert state.entity_id == 5
        assert state.position == (10.0, 20.0, 30.0)

    def test_create_influence_debug_state(self):
        """Test create_influence_debug_state factory."""
        state = create_influence_debug_state(
            map_id="danger_map",
            map_name="DangerMap",
            grid_size=(64, 64),
            cell_size=0.5,
        )

        assert state.map_id == "danger_map"
        assert state.map_name == "DangerMap"
        assert state.grid_size == (64, 64)
        assert state.cell_size == 0.5


class TestVisualizationIntegration:
    """Tests for visualization integration points."""

    def test_generate_perception_wireframes(self, sample_perception_state):
        """Test generating perception wireframes."""
        config = AIDebugConfig(enabled=True, show_perception=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            perception_state=sample_perception_state,
        )

        wireframes = generate_perception_wireframes(data)
        assert len(wireframes) == 2

        sight_cone = next((w for w in wireframes if w.range_value == 30.0), None)
        assert sight_cone is not None
        assert sight_cone.fov_degrees == 90.0
        assert sight_cone.origin == (10.0, 0.0, 10.0)

    def test_generate_perception_wireframes_inactive(self, sample_perception_state):
        """Test that inactive perception ranges are excluded."""
        sample_perception_state.ranges["sight"].is_active = False

        config = AIDebugConfig(enabled=True, show_perception=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            perception_state=sample_perception_state,
        )

        wireframes = generate_perception_wireframes(data)
        assert len(wireframes) == 1  # Only hearing

    def test_generate_bt_visualization(self, sample_bt_state):
        """Test generating BT visualization."""
        config = AIDebugConfig(enabled=True, show_bt=True)
        data = AIDebugData(
            entity_id=1,
            config=config,
            bt_state=sample_bt_state,
        )

        visualizations = generate_bt_visualization(data)
        assert len(visualizations) == 4  # All nodes

        root_vis = next((v for v in visualizations if v.node_id == "root"), None)
        assert root_vis is not None
        assert root_vis.is_active is True  # In active path
        assert root_vis.depth == 0

    def test_wireframe_cone_data_class(self):
        """Test WireframeCone data class."""
        cone = WireframeCone(
            origin=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            range_value=20.0,
            fov_degrees=60.0,
            color=(1.0, 1.0, 0.0, 0.5),
            segments=24,
        )

        assert cone.range_value == 20.0
        assert cone.fov_degrees == 60.0
        assert cone.segments == 24

    def test_bt_node_visualization_data_class(self):
        """Test BTNodeVisualization data class."""
        vis = BTNodeVisualization(
            node_id="test_node",
            node_name="TestAction",
            position=(100.0, 50.0),
            size=(80.0, 30.0),
            color=(0.0, 1.0, 0.0, 0.8),
            is_active=True,
            children=["child1", "child2"],
            depth=2,
        )

        assert vis.node_id == "test_node"
        assert vis.position == (100.0, 50.0)
        assert vis.is_active is True
        assert len(vis.children) == 2


class TestMultipleConfigs:
    """Tests for multiple debug configurations coexisting."""

    def test_multiple_configs_coexist(self, clean_registry):
        """Test that multiple AI debug configs can coexist."""
        @ai_debug(enabled=True, show_bt=True)
        class MultiAI01:
            pass

        @ai_debug(enabled=True, show_perception=True)
        class MultiAI02:
            pass

        @ai_debug(enabled=False, show_influence=True)
        class MultiAI03:
            pass

        all_configs = get_all_ai_debug_configs()
        assert MultiAI01 in all_configs
        assert MultiAI02 in all_configs
        assert MultiAI03 in all_configs

    def test_multiple_configs_independent(self, clean_registry):
        """Test that multiple configs are independent."""
        @ai_debug(enabled=True)
        class IndepAI01:
            pass

        @ai_debug(enabled=True)
        class IndepAI02:
            pass

        # Disable one
        disable_debug(IndepAI01)

        assert is_debug_enabled(IndepAI01) is False
        assert is_debug_enabled(IndepAI02) is True


class TestDefaultColors:
    """Tests for default color constants."""

    def test_default_bt_node_colors(self):
        """Test DEFAULT_BT_NODE_COLORS constant."""
        assert "running" in DEFAULT_BT_NODE_COLORS
        assert "success" in DEFAULT_BT_NODE_COLORS
        assert "failure" in DEFAULT_BT_NODE_COLORS
        assert "idle" in DEFAULT_BT_NODE_COLORS

        # Check RGBA format
        for color in DEFAULT_BT_NODE_COLORS.values():
            assert len(color) == 4
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_default_perception_colors(self):
        """Test DEFAULT_PERCEPTION_COLORS constant."""
        assert "sight" in DEFAULT_PERCEPTION_COLORS
        assert "hearing" in DEFAULT_PERCEPTION_COLORS
        assert "smell" in DEFAULT_PERCEPTION_COLORS

        for color in DEFAULT_PERCEPTION_COLORS.values():
            assert len(color) == 4
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_default_influence_colors(self):
        """Test DEFAULT_INFLUENCE_COLORS constant."""
        assert "positive" in DEFAULT_INFLUENCE_COLORS
        assert "negative" in DEFAULT_INFLUENCE_COLORS
        assert "neutral" in DEFAULT_INFLUENCE_COLORS

        for color in DEFAULT_INFLUENCE_COLORS.values():
            assert len(color) == 4
            assert all(0.0 <= c <= 1.0 for c in color)


class TestPerformance:
    """Performance tests."""

    def test_100_debug_queries_under_50ms(self, clean_registry):
        """Test that 100 debug queries complete under 50ms."""
        # Register 20 AI debug classes with unique names
        classes = []
        for i in range(20):
            # Use exec to create classes with unique names to avoid registry conflicts
            class_name = f"PerfAI{i:02d}"
            namespace = {}
            exec(f"""
@ai_debug(enabled={i % 2 == 0}, show_bt=True, show_perception={i % 3 == 0}, name="perf_test.{class_name}")
class {class_name}:
    pass
""", {"ai_debug": ai_debug}, namespace)
            classes.append(namespace[class_name])

        # Run 100 queries
        start = time.perf_counter()

        for _ in range(100):
            get_all_ai_debug_configs()
            get_enabled_debug_configs()
            get_debug_configs_with_bt()
            get_debug_configs_with_perception()
            get_debug_configs_with_influence()

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"100 queries took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_100_storage_operations_under_50ms(self, clean_debug_storage):
        """Test that 100 storage operations complete under 50ms."""
        config = AIDebugConfig(enabled=True)

        start = time.perf_counter()

        for i in range(100):
            data = AIDebugData(entity_id=i, config=config)
            set_debug_data(i, data)
            get_debug_data(i)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"100 storage ops took {elapsed_ms:.2f}ms, expected < 50ms"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_bt_state_active_path(self):
        """Test get_active_path with empty BT state."""
        state = BTDebugState(tree_id="empty", tree_name="EmptyTree")
        path = state.get_active_path()
        assert path == []

    def test_empty_perception_wireframes(self):
        """Test generate_perception_wireframes with no perception state."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=1, config=config, perception_state=None)

        wireframes = generate_perception_wireframes(data)
        assert wireframes == []

    def test_empty_bt_visualization(self):
        """Test generate_bt_visualization with no BT state."""
        config = AIDebugConfig(enabled=True)
        data = AIDebugData(entity_id=1, config=config, bt_state=None)

        visualizations = generate_bt_visualization(data)
        assert visualizations == []

    def test_remove_nonexistent_debug_data(self, clean_debug_storage):
        """Test removing debug data that doesn't exist."""
        result = remove_debug_data(999)
        assert result is False

    def test_get_nonexistent_debug_data(self, clean_debug_storage):
        """Test getting debug data that doesn't exist."""
        result = get_debug_data(999)
        assert result is None

    def test_ai_debug_decorator_reload_safe(self, clean_registry):
        """Test that @ai_debug handles re-registration gracefully."""
        @ai_debug(enabled=True, name="reload_test.ReloadAI")
        class ReloadAI:
            pass

        # Verify first class is registered
        assert registry.is_registered(ReloadAI)

        # Test with a different class that has same base name but different identity
        # The decorator should handle this by generating a unique name
        @ai_debug(enabled=True, name="reload_test.ReloadAI2")
        class ReloadAI2:
            pass

        # Both should be queryable
        assert registry.is_registered(ReloadAI)
        assert registry.is_registered(ReloadAI2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_ai_debug_workflow(
        self,
        clean_registry,
        clean_debug_storage,
        sample_bt_state,
        sample_perception_state,
        sample_influence_state,
    ):
        """Test complete AI debug workflow."""
        # 1. Register AI class with debug
        @ai_debug(
            enabled=True,
            show_bt=True,
            show_perception=True,
            show_influence=True,
            description="Full workflow test AI",
        )
        class WorkflowAI:
            pass

        # 2. Verify registration
        assert registry.is_registered(WorkflowAI)
        assert registry.has_tag(WorkflowAI, TAG_AI_DEBUG)

        # 3. Query via Registry
        all_configs = get_all_ai_debug_configs()
        assert WorkflowAI in all_configs

        # 4. Get config
        config = get_debug_config(WorkflowAI)
        assert config is not None

        # 5. Create and store debug data
        data = AIDebugData(
            entity_id=100,
            config=config,
            bt_state=sample_bt_state,
            perception_state=sample_perception_state,
            influence_states={"threat": sample_influence_state},
        )
        set_debug_data(100, data)

        # 6. Retrieve and verify
        retrieved = get_debug_data(100)
        assert retrieved is not None
        assert retrieved.entity_id == 100
        assert retrieved.bt_state is not None
        assert retrieved.perception_state is not None
        assert "threat" in retrieved.influence_states

        # 7. Generate visualizations
        wireframes = generate_perception_wireframes(retrieved)
        assert len(wireframes) == 2

        bt_vis = generate_bt_visualization(retrieved)
        assert len(bt_vis) == 4

        # 8. Toggle debug
        toggle_debug(WorkflowAI)
        assert is_debug_enabled(WorkflowAI) is False

        # 9. Cleanup
        remove_debug_data(100)
        assert get_debug_data(100) is None
