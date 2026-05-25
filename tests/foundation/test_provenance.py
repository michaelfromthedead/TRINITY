"""
Tests for Computed Provenance tracking.

Verifies:
- ComputedProvenance dataclass
- @track_provenance decorator
- record_input function
- record_read function (automatic read tracking)
- provenance() query function
- clear_provenance() cleanup
- derivation_tree() query function
- ProvenanceView inspector integration
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.provenance import (
    ReadRecord,
    DerivationNode,
    ComputedProvenance,
    track_provenance,
    record_input,
    record_read,
    get_current_reads_collector,
    provenance,
    clear_provenance,
    all_provenance,
    derivation_tree,
)
from foundation.eventlog import set_current_tick


@pytest.fixture(autouse=True)
def reset_provenance():
    """Reset provenance and tick between tests."""
    clear_provenance()
    set_current_tick(0)
    yield
    clear_provenance()
    set_current_tick(0)


class TestComputedProvenanceDataclass:
    """Test ComputedProvenance dataclass."""

    def test_provenance_creation_minimal(self):
        """ComputedProvenance with minimal fields."""
        prov = ComputedProvenance(
            value=42,
            computed_by="Entity.computed_value",
            tick=0
        )

        assert prov.value == 42
        assert prov.computed_by == "Entity.computed_value"
        assert prov.tick == 0
        assert prov.input_summary == {}

    def test_provenance_creation_with_inputs(self):
        """ComputedProvenance with input summary."""
        prov = ComputedProvenance(
            value=100,
            computed_by="Player.threat_level",
            tick=50,
            input_summary={"enemies": [1, 2, 3], "damage": 100}
        )

        assert prov.value == 100
        assert prov.computed_by == "Player.threat_level"
        assert prov.tick == 50
        assert prov.input_summary == {"enemies": [1, 2, 3], "damage": 100}

    def test_provenance_with_none_value(self):
        """ComputedProvenance can store None as value."""
        prov = ComputedProvenance(
            value=None,
            computed_by="Entity.optional_value",
            tick=10
        )

        assert prov.value is None


class TestTrackProvenanceDecorator:
    """Test @track_provenance decorator."""

    def test_provenance_recorded(self):
        """@track_provenance records provenance."""
        class Entity:
            @track_provenance
            def computed_value(self):
                return 42

        e = Entity()
        result = e.computed_value()

        assert result == 42
        prov = provenance(e, "computed_value")
        assert prov is not None
        assert prov.value == 42
        assert prov.computed_by == "Entity.computed_value"

    def test_provenance_tick(self):
        """Provenance records the current tick."""
        set_current_tick(100)

        class Entity:
            @track_provenance
            def value(self):
                return 1

        e = Entity()
        e.value()

        prov = provenance(e, "value")
        assert prov is not None
        assert prov.tick == 100

    def test_provenance_different_ticks(self):
        """Provenance updates with different ticks on recompute."""
        class Entity:
            def __init__(self):
                self.n = 0

            @track_provenance
            def value(self):
                self.n += 1
                return self.n

        e = Entity()

        set_current_tick(10)
        e.value()
        assert provenance(e, "value").tick == 10

        set_current_tick(20)
        e.value()
        assert provenance(e, "value").tick == 20

    def test_provenance_with_arguments(self):
        """@track_provenance works with method arguments."""
        class Calculator:
            @track_provenance
            def add(self, a, b):
                return a + b

        calc = Calculator()
        result = calc.add(3, 5)

        assert result == 8
        prov = provenance(calc, "add")
        assert prov.value == 8

    def test_provenance_with_kwargs(self):
        """@track_provenance works with keyword arguments."""
        class Entity:
            @track_provenance
            def compute(self, multiplier=1):
                return 10 * multiplier

        e = Entity()
        result = e.compute(multiplier=3)

        assert result == 30
        prov = provenance(e, "compute")
        assert prov.value == 30

    def test_provenance_preserves_return_value(self):
        """@track_provenance passes through return values."""
        class Entity:
            @track_provenance
            def get_list(self):
                return [1, 2, 3]

        e = Entity()
        result = e.get_list()

        assert result == [1, 2, 3]
        assert provenance(e, "get_list").value == [1, 2, 3]

    def test_provenance_handles_exceptions(self):
        """@track_provenance re-raises exceptions."""
        class Entity:
            @track_provenance
            def fail(self):
                raise ValueError("test error")

        e = Entity()
        with pytest.raises(ValueError) as exc_info:
            e.fail()

        assert "test error" in str(exc_info.value)
        # Provenance should not be recorded for failed computations
        # because result = fn(...) raises before storing
        assert provenance(e, "fail") is None


class TestRecordInput:
    """Test record_input function."""

    def test_provenance_inputs(self):
        """Input summary captures recorded inputs."""
        class Entity:
            @track_provenance
            def threat(self):
                record_input("enemies", [1, 2, 3])
                record_input("total_damage", 75)
                return 75

        e = Entity()
        e.threat()

        prov = provenance(e, "threat")
        assert prov is not None
        assert prov.input_summary["enemies"] == [1, 2, 3]
        assert prov.input_summary["total_damage"] == 75

    def test_record_input_overwrites_same_key(self):
        """Recording same input key overwrites previous value."""
        class Entity:
            @track_provenance
            def compute(self):
                record_input("value", 1)
                record_input("value", 2)
                return 2

        e = Entity()
        e.compute()

        prov = provenance(e, "compute")
        assert prov.input_summary["value"] == 2

    def test_record_input_outside_provenance_is_noop(self):
        """record_input outside @track_provenance is a no-op."""
        # Should not raise or have any effect
        record_input("key", "value")
        # No assertion needed - just verify no exception

    def test_record_input_various_types(self):
        """record_input handles various value types."""
        class Entity:
            @track_provenance
            def compute(self):
                record_input("string", "hello")
                record_input("number", 42)
                record_input("float", 3.14)
                record_input("list", [1, 2, 3])
                record_input("dict", {"a": 1})
                record_input("none", None)
                record_input("bool", True)
                return "done"

        e = Entity()
        e.compute()

        prov = provenance(e, "compute")
        assert prov.input_summary["string"] == "hello"
        assert prov.input_summary["number"] == 42
        assert prov.input_summary["float"] == 3.14
        assert prov.input_summary["list"] == [1, 2, 3]
        assert prov.input_summary["dict"] == {"a": 1}
        assert prov.input_summary["none"] is None
        assert prov.input_summary["bool"] is True


class TestProvenanceQuery:
    """Test provenance() query function."""

    def test_provenance_query_returns_none_for_untracked(self):
        """provenance() returns None for non-computed fields."""
        class Entity:
            def regular_method(self):
                return 1

        e = Entity()
        e.regular_method()

        assert provenance(e, "regular_method") is None
        assert provenance(e, "nonexistent") is None

    def test_provenance_query_different_instances(self):
        """Different instances have separate provenance."""
        class Entity:
            def __init__(self, value):
                self._value = value

            @track_provenance
            def get_value(self):
                return self._value

        e1 = Entity(10)
        e2 = Entity(20)

        e1.get_value()
        e2.get_value()

        assert provenance(e1, "get_value").value == 10
        assert provenance(e2, "get_value").value == 20

    def test_provenance_updates_on_recompute(self):
        """New computation updates provenance."""
        class Counter:
            def __init__(self):
                self.n = 0

            @track_provenance
            def count(self):
                self.n += 1
                return self.n

        c = Counter()
        c.count()
        assert provenance(c, "count").value == 1

        c.count()
        assert provenance(c, "count").value == 2

        c.count()
        assert provenance(c, "count").value == 3

    def test_provenance_multiple_fields_same_object(self):
        """Multiple computed fields on same object tracked separately."""
        class Entity:
            def __init__(self):
                self.x = 10
                self.y = 20

            @track_provenance
            def sum(self):
                return self.x + self.y

            @track_provenance
            def product(self):
                return self.x * self.y

        e = Entity()
        e.sum()
        e.product()

        sum_prov = provenance(e, "sum")
        product_prov = provenance(e, "product")

        assert sum_prov.value == 30
        assert sum_prov.computed_by == "Entity.sum"

        assert product_prov.value == 200
        assert product_prov.computed_by == "Entity.product"


class TestClearProvenance:
    """Test clear_provenance function."""

    def test_clear_provenance(self):
        """clear_provenance removes all data."""
        class E:
            @track_provenance
            def v(self):
                return 1

        e = E()
        e.v()
        assert provenance(e, "v") is not None

        clear_provenance()
        assert provenance(e, "v") is None

    def test_clear_provenance_multiple_objects(self):
        """clear_provenance removes data for all objects."""
        class E:
            @track_provenance
            def v(self):
                return 1

        e1 = E()
        e2 = E()
        e3 = E()

        e1.v()
        e2.v()
        e3.v()

        assert len(all_provenance()) == 3

        clear_provenance()

        assert len(all_provenance()) == 0
        assert provenance(e1, "v") is None
        assert provenance(e2, "v") is None
        assert provenance(e3, "v") is None


class TestAllProvenance:
    """Test all_provenance function."""

    def test_all_provenance_returns_copy(self):
        """all_provenance returns a copy of the registry."""
        class E:
            @track_provenance
            def v(self):
                return 1

        e = E()
        e.v()

        prov_dict = all_provenance()
        assert len(prov_dict) == 1

        # Modifying returned dict should not affect internal state
        prov_dict.clear()

        assert len(all_provenance()) == 1

    def test_all_provenance_contains_all_data(self):
        """all_provenance includes all tracked computations."""
        class Entity:
            @track_provenance
            def a(self):
                return 1

            @track_provenance
            def b(self):
                return 2

        e1 = Entity()
        e2 = Entity()

        e1.a()
        e1.b()
        e2.a()

        prov_dict = all_provenance()
        assert len(prov_dict) == 3


class TestNestedComputation:
    """Test nested @track_provenance computations."""

    def test_nested_provenance_separate(self):
        """Nested computations track separately."""
        class Entity:
            @track_provenance
            def outer(self):
                record_input("outer_input", "from_outer")
                inner_result = self.inner()
                return inner_result + 10

            @track_provenance
            def inner(self):
                record_input("inner_input", "from_inner")
                return 5

        e = Entity()
        result = e.outer()

        assert result == 15

        outer_prov = provenance(e, "outer")
        inner_prov = provenance(e, "inner")

        assert outer_prov.value == 15
        assert outer_prov.input_summary == {"outer_input": "from_outer"}

        assert inner_prov.value == 5
        assert inner_prov.input_summary == {"inner_input": "from_inner"}

    def test_nested_inputs_isolated(self):
        """Inputs in nested computations don't leak to outer."""
        class Entity:
            @track_provenance
            def outer(self):
                record_input("level", "outer")
                self.inner()
                return "outer_result"

            @track_provenance
            def inner(self):
                record_input("level", "inner")
                return "inner_result"

        e = Entity()
        e.outer()

        outer_prov = provenance(e, "outer")
        inner_prov = provenance(e, "inner")

        # Each should only have its own inputs
        assert outer_prov.input_summary == {"level": "outer"}
        assert inner_prov.input_summary == {"level": "inner"}


class TestReadRecord:
    """Test ReadRecord dataclass."""

    def test_read_record_creation(self):
        """ReadRecord stores all fields correctly."""
        record = ReadRecord(
            obj_id=12345,
            obj_type="Entity",
            field="health",
            value=100
        )

        assert record.obj_id == 12345
        assert record.obj_type == "Entity"
        assert record.field == "health"
        assert record.value == 100

    def test_read_record_with_complex_value(self):
        """ReadRecord can store complex values."""
        record = ReadRecord(
            obj_id=1,
            obj_type="Entity",
            field="items",
            value={"sword": 1, "shield": 2}
        )

        assert record.value == {"sword": 1, "shield": 2}


class TestDerivationNode:
    """Test DerivationNode dataclass."""

    def test_derivation_node_minimal(self):
        """DerivationNode with minimal fields."""
        node = DerivationNode(
            field="threat_level",
            value=75
        )

        assert node.field == "threat_level"
        assert node.value == 75
        assert node.source_obj_id is None
        assert node.source_obj_type is None
        assert node.children == []

    def test_derivation_node_full(self):
        """DerivationNode with all fields."""
        child = DerivationNode(field="damage", value=50)
        node = DerivationNode(
            field="threat_level",
            value=75,
            source_obj_id=12345,
            source_obj_type="Player",
            children=[child]
        )

        assert node.field == "threat_level"
        assert node.value == 75
        assert node.source_obj_id == 12345
        assert node.source_obj_type == "Player"
        assert len(node.children) == 1
        assert node.children[0].field == "damage"


class TestAutoReadTracking:
    """Test automatic read tracking via record_read."""

    def test_record_read_captured(self):
        """record_read captures reads during provenance tracking."""
        class Entity:
            def __init__(self, damage: int):
                self._damage = damage

            @track_provenance
            def threat(self):
                # Simulate what a descriptor would do
                record_read(self, "damage", self._damage)
                return self._damage * 2

        e = Entity(50)
        result = e.threat()

        assert result == 100
        prov = provenance(e, "threat")
        assert prov is not None
        assert len(prov.reads) == 1
        assert prov.reads[0].field == "damage"
        assert prov.reads[0].value == 50
        assert prov.reads[0].obj_type == "Entity"

    def test_record_read_multiple(self):
        """Multiple reads are captured in order."""
        class Player:
            def __init__(self, health: int, armor: int):
                self._health = health
                self._armor = armor

            @track_provenance
            def effective_health(self):
                record_read(self, "health", self._health)
                record_read(self, "armor", self._armor)
                return self._health + self._armor * 2

        p = Player(100, 50)
        result = p.effective_health()

        assert result == 200
        prov = provenance(p, "effective_health")
        assert len(prov.reads) == 2
        assert prov.reads[0].field == "health"
        assert prov.reads[0].value == 100
        assert prov.reads[1].field == "armor"
        assert prov.reads[1].value == 50

    def test_record_read_outside_provenance_is_noop(self):
        """record_read outside @track_provenance is a no-op."""
        # Should not raise or have any effect
        record_read(object(), "field", "value")
        # No assertion needed - just verify no exception

    def test_record_read_from_different_objects(self):
        """Reads from multiple objects are tracked."""
        class Enemy:
            def __init__(self, damage: int):
                self._damage = damage

        class Player:
            def __init__(self, enemies: list):
                self._enemies = enemies

            @track_provenance
            def threat_level(self):
                total = 0
                for enemy in self._enemies:
                    record_read(enemy, "damage", enemy._damage)
                    total += enemy._damage
                return total

        e1 = Enemy(25)
        e2 = Enemy(50)
        p = Player([e1, e2])
        result = p.threat_level()

        assert result == 75
        prov = provenance(p, "threat_level")
        assert len(prov.reads) == 2
        # First read from e1
        assert prov.reads[0].obj_id == id(e1)
        assert prov.reads[0].value == 25
        # Second read from e2
        assert prov.reads[1].obj_id == id(e2)
        assert prov.reads[1].value == 50

    def test_get_current_reads_collector(self):
        """get_current_reads_collector returns None outside provenance."""
        assert get_current_reads_collector() is None

    def test_get_current_reads_collector_inside_provenance(self):
        """get_current_reads_collector returns list inside provenance."""
        collector_inside = None

        class Entity:
            @track_provenance
            def compute(self):
                nonlocal collector_inside
                collector_inside = get_current_reads_collector()
                return 42

        e = Entity()
        e.compute()

        assert collector_inside is not None
        assert isinstance(collector_inside, list)


class TestDerivationTree:
    """Test derivation_tree function."""

    def test_derivation_tree_simple(self):
        """derivation_tree builds tree from single reads."""
        class Entity:
            def __init__(self, x: int, y: int):
                self._x = x
                self._y = y

            @track_provenance
            def sum(self):
                record_read(self, "x", self._x)
                record_read(self, "y", self._y)
                return self._x + self._y

        e = Entity(10, 20)
        e.sum()

        tree = derivation_tree(e, "sum")
        assert tree is not None
        assert tree.field == "sum"
        assert tree.value == 30
        assert len(tree.children) == 2
        assert tree.children[0].field == "x"
        assert tree.children[0].value == 10
        assert tree.children[1].field == "y"
        assert tree.children[1].value == 20

    def test_derivation_tree_returns_none_for_untracked(self):
        """derivation_tree returns None for non-computed fields."""
        class Entity:
            pass

        e = Entity()
        tree = derivation_tree(e, "nonexistent")
        assert tree is None

    def test_derivation_tree_structure(self):
        """derivation_tree has correct structure."""
        class Entity:
            @track_provenance
            def compute(self):
                record_read(self, "value", 42)
                return 42

        e = Entity()
        e.compute()

        tree = derivation_tree(e, "compute")
        assert tree is not None
        assert tree.source_obj_id == id(e)
        assert tree.source_obj_type == "Entity"
        assert len(tree.children) == 1
        assert tree.children[0].source_obj_id == id(e)
        assert tree.children[0].source_obj_type == "Entity"

    def test_derivation_tree_no_reads(self):
        """derivation_tree with no recorded reads."""
        class Entity:
            @track_provenance
            def constant(self):
                return 42

        e = Entity()
        e.constant()

        tree = derivation_tree(e, "constant")
        assert tree is not None
        assert tree.field == "constant"
        assert tree.value == 42
        assert tree.children == []


class TestNestedReadsTracking:
    """Test nested computations track reads separately."""

    def test_nested_reads_isolated(self):
        """Reads in nested computations don't leak to outer."""
        class Entity:
            def __init__(self):
                self._outer_val = 10
                self._inner_val = 5

            @track_provenance
            def outer(self):
                record_read(self, "outer_val", self._outer_val)
                inner_result = self.inner()
                return self._outer_val + inner_result

            @track_provenance
            def inner(self):
                record_read(self, "inner_val", self._inner_val)
                return self._inner_val

        e = Entity()
        result = e.outer()

        assert result == 15

        outer_prov = provenance(e, "outer")
        inner_prov = provenance(e, "inner")

        # Each should only have its own reads
        assert len(outer_prov.reads) == 1
        assert outer_prov.reads[0].field == "outer_val"

        assert len(inner_prov.reads) == 1
        assert inner_prov.reads[0].field == "inner_val"

    def test_deeply_nested_reads(self):
        """Deeply nested computations maintain isolation."""
        class Entity:
            @track_provenance
            def level1(self):
                record_read(self, "a", 1)
                return self.level2() + 1

            @track_provenance
            def level2(self):
                record_read(self, "b", 2)
                return self.level3() + 2

            @track_provenance
            def level3(self):
                record_read(self, "c", 3)
                return 3

        e = Entity()
        result = e.level1()

        assert result == 6  # 3 + 2 + 1

        assert len(provenance(e, "level1").reads) == 1
        assert provenance(e, "level1").reads[0].field == "a"

        assert len(provenance(e, "level2").reads) == 1
        assert provenance(e, "level2").reads[0].field == "b"

        assert len(provenance(e, "level3").reads) == 1
        assert provenance(e, "level3").reads[0].field == "c"


class TestProvenanceView:
    """Test ProvenanceView inspector integration."""

    def test_provenance_view_can_render(self):
        """ProvenanceView.can_render returns True for objects with provenance."""
        from foundation.inspector_views import ProvenanceView

        class Entity:
            @track_provenance
            def value(self):
                return 42

        e = Entity()
        view = ProvenanceView()

        # Before computing, no provenance exists
        assert view.can_render(e) is False

        # After computing, provenance exists
        e.value()
        assert view.can_render(e) is True

    def test_provenance_view_render(self):
        """ProvenanceView.render produces correct output."""
        from foundation.inspector_views import ProvenanceView
        from foundation.inspector import TextUIContext

        set_current_tick(100)

        class Player:
            @track_provenance
            def threat_level(self):
                record_input("enemies", 3)
                record_read(self, "damage", 50)
                return 150

        p = Player()
        p.threat_level()

        view = ProvenanceView()
        ctx = TextUIContext()
        output = view.render(p, ctx)

        assert "Provenance for Player" in output
        assert "threat_level = 150" in output
        assert "computed by: Player.threat_level" in output
        assert "at tick: 100" in output
        assert "enemies: 3" in output
        assert "damage = 50" in output

    def test_provenance_view_empty(self):
        """ProvenanceView.render handles objects without provenance."""
        from foundation.inspector_views import ProvenanceView
        from foundation.inspector import TextUIContext

        class Entity:
            pass

        e = Entity()
        view = ProvenanceView()
        ctx = TextUIContext()
        output = view.render(e, ctx)

        assert "No provenance recorded" in output

    def test_provenance_view_multiple_fields(self):
        """ProvenanceView.render shows all computed fields."""
        from foundation.inspector_views import ProvenanceView
        from foundation.inspector import TextUIContext

        class Entity:
            @track_provenance
            def field_a(self):
                return 1

            @track_provenance
            def field_b(self):
                return 2

        e = Entity()
        e.field_a()
        e.field_b()

        view = ProvenanceView()
        ctx = TextUIContext()
        output = view.render(e, ctx)

        assert "field_a = 1" in output
        assert "field_b = 2" in output

    def test_provenance_view_name(self):
        """ProvenanceView has correct name."""
        from foundation.inspector_views import ProvenanceView

        view = ProvenanceView()
        assert view.name == "Provenance"


class TestRegisterInspectorViews:
    """Test register_inspector_views includes ProvenanceView."""

    def test_provenance_view_registered(self):
        """register_inspector_views registers ProvenanceView."""
        from foundation.inspector import inspector
        from foundation.inspector_views import register_inspector_views

        # Clear existing views to ensure clean test
        original_views = inspector._views.copy()
        inspector._views = []

        try:
            register_inspector_views()
            view_names = [v.name for v in inspector._views]
            assert "Provenance" in view_names
            assert "History" in view_names
            assert "Causality" in view_names
        finally:
            # Restore original views
            inspector._views = original_views
