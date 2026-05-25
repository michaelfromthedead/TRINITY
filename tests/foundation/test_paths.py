"""
Comprehensive unit tests for the Path Utilities module.
Tests parsing, getting, and setting values at dotted paths.
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.paths import parse_path, get_path, set_path, PathError
from foundation.mirror import mirror


class TestParsePath:
    """Test parse_path() function."""

    def test_parse_simple_path(self):
        """Simple dotted path should split into segments."""
        result = parse_path("a.b.c")
        assert result == ['a', 'b', 'c']

    def test_parse_array_index(self):
        """Path with array index should return int for index."""
        result = parse_path("items[0]")
        assert result == ['items', 0]

    def test_parse_nested_arrays(self):
        """Multiple consecutive array indices should be parsed."""
        result = parse_path("a[0][1]")
        assert result == ['a', 0, 1]

    def test_parse_mixed(self):
        """Mixed dotted paths and array indices should parse correctly."""
        result = parse_path("data[0].name[1].value")
        assert result == ['data', 0, 'name', 1, 'value']

    def test_parse_empty_path(self):
        """Empty path should return empty list."""
        result = parse_path("")
        assert result == []

    def test_parse_single_segment(self):
        """Single segment path should return single-element list."""
        result = parse_path("foo")
        assert result == ['foo']

    def test_parse_single_index(self):
        """Path with only index should work."""
        result = parse_path("[0]")
        assert result == [0]

    def test_parse_large_index(self):
        """Large array indices should parse correctly."""
        result = parse_path("items[999]")
        assert result == ['items', 999]

    def test_parse_complex_path(self):
        """Complex path with multiple patterns should parse correctly."""
        result = parse_path("x[2].y[3].z")
        assert result == ['x', 2, 'y', 3, 'z']

    def test_parse_deeply_nested(self):
        """Deeply nested path should parse correctly."""
        result = parse_path("a.b.c.d.e.f.g")
        assert result == ['a', 'b', 'c', 'd', 'e', 'f', 'g']


class TestGetPathDict:
    """Test get_path() with dictionary objects."""

    def test_get_path_dict(self):
        """Simple dict access should work."""
        data = {"a": {"b": 1}}
        result = get_path(data, "a.b")
        assert result == 1

    def test_get_path_dict_list(self):
        """Dict containing list should be navigable."""
        data = {"items": [10, 20, 30]}
        result = get_path(data, "items[1]")
        assert result == 20

    def test_get_path_nested_dict_list(self):
        """Nested dicts and lists should navigate correctly."""
        data = {"players": [{"name": "Alice", "score": 100}]}
        result = get_path(data, "players[0].name")
        assert result == "Alice"

    def test_get_path_empty_path(self):
        """Empty path should return the object itself."""
        data = {"a": 1}
        result = get_path(data, "")
        assert result == data


class TestGetPathObject:
    """Test get_path() with object attributes."""

    def test_get_path_object(self):
        """Object attribute access should work."""
        class Player:
            def __init__(self):
                self.health = 100
                self.name = "Hero"

        player = Player()
        assert get_path(player, "health") == 100
        assert get_path(player, "name") == "Hero"

    def test_get_path_nested_object(self):
        """Nested object attributes should navigate correctly."""
        class Weapon:
            damage = 25

        class Player:
            def __init__(self):
                self.weapon = Weapon()

        player = Player()
        assert get_path(player, "weapon.damage") == 25


class TestGetPathMixed:
    """Test get_path() with mixed dicts, objects, and lists."""

    def test_get_path_mixed(self):
        """Mixed dict + object + list navigation should work."""
        class Item:
            def __init__(self, name, damage):
                self.name = name
                self.damage = damage

        class Inventory:
            def __init__(self):
                self.items = [Item("sword", 10), Item("axe", 15)]

        class Player:
            def __init__(self):
                self.inventory = Inventory()

        player = Player()
        result = get_path(player, "inventory.items[0].damage")
        assert result == 10
        result = get_path(player, "inventory.items[1].name")
        assert result == "axe"

    def test_get_path_dict_with_object(self):
        """Dict containing object should navigate correctly."""
        class Stats:
            strength = 10
            agility = 15

        data = {"player": {"stats": Stats()}}
        result = get_path(data, "player.stats.strength")
        assert result == 10


class TestGetPathError:
    """Test get_path() error handling."""

    def test_get_path_missing_raises(self):
        """Missing path should raise PathError."""
        data = {"a": 1}
        with pytest.raises(PathError):
            get_path(data, "missing.path")

    def test_get_path_missing_key(self):
        """Missing dict key should raise PathError."""
        data = {"a": {"b": 1}}
        with pytest.raises(PathError):
            get_path(data, "a.c")

    def test_get_path_missing_attribute(self):
        """Missing object attribute should raise PathError."""
        class Simple:
            x = 1

        obj = Simple()
        with pytest.raises(PathError):
            get_path(obj, "y")

    def test_get_path_index_out_of_range(self):
        """Out of range index should raise PathError."""
        data = {"items": [1, 2]}
        with pytest.raises(PathError):
            get_path(data, "items[99]")


class TestGetPathDefault:
    """Test get_path() with default values."""

    def test_get_path_default(self):
        """Missing path with default should return default."""
        data = {"a": 1}
        result = get_path(data, "missing.path", default=None)
        assert result is None

    def test_get_path_default_value(self):
        """Default value should be returned when path not found."""
        data = {"a": 1}
        result = get_path(data, "b.c", default="fallback")
        assert result == "fallback"

    def test_get_path_default_none_explicit(self):
        """None as default should work (not confused with missing)."""
        data = {}
        result = get_path(data, "missing", default=None)
        assert result is None

    def test_get_path_default_zero(self):
        """Zero as default should work (falsy value)."""
        data = {}
        result = get_path(data, "missing", default=0)
        assert result == 0

    def test_get_path_found_returns_value_not_default(self):
        """When path exists, actual value is returned not default."""
        data = {"a": 42}
        result = get_path(data, "a", default=999)
        assert result == 42


class TestSetPathDict:
    """Test set_path() with dictionaries."""

    def test_set_path_dict(self):
        """Setting dict value at path should work."""
        data = {"a": {"b": 1}}
        set_path(data, "a.b", 2)
        assert data["a"]["b"] == 2

    def test_set_path_dict_new_key(self):
        """Setting new key in existing dict should work."""
        data = {"a": {}}
        set_path(data, "a.b", 99)
        assert data["a"]["b"] == 99

    def test_set_path_list_item(self):
        """Setting list item by index should work."""
        data = {"items": [1, 2, 3]}
        set_path(data, "items[1]", 20)
        assert data["items"][1] == 20


class TestSetPathObject:
    """Test set_path() with objects."""

    def test_set_path_object(self):
        """Setting object attribute should work."""
        class Player:
            health = 100

        player = Player()
        set_path(player, "health", 50)
        assert player.health == 50

    def test_set_path_nested_object(self):
        """Setting nested object attribute should work."""
        class Weapon:
            damage = 10

        class Player:
            def __init__(self):
                self.weapon = Weapon()

        player = Player()
        set_path(player, "weapon.damage", 25)
        assert player.weapon.damage == 25


class TestSetPathNested:
    """Test set_path() with deep mutation."""

    def test_set_path_nested(self):
        """Deep nested mutation should work."""
        data = {"level1": {"level2": {"level3": 0}}}
        set_path(data, "level1.level2.level3", 999)
        assert data["level1"]["level2"]["level3"] == 999

    def test_set_path_create_intermediate(self):
        """create_intermediate=True should create missing dicts."""
        data = {}
        set_path(data, "x.y.z", 5, create_intermediate=True)
        assert data["x"]["y"]["z"] == 5

    def test_set_path_create_intermediate_partial(self):
        """create_intermediate should work with partially existing path."""
        data = {"a": {}}
        set_path(data, "a.b.c", 10, create_intermediate=True)
        assert data["a"]["b"]["c"] == 10

    def test_set_path_without_create_intermediate_raises(self):
        """Without create_intermediate, missing path should raise."""
        data = {}
        with pytest.raises(PathError):
            set_path(data, "x.y.z", 5)


class TestSetPathError:
    """Test set_path() error handling."""

    def test_set_path_empty_raises(self):
        """Empty path should raise PathError."""
        data = {"a": 1}
        with pytest.raises(PathError):
            set_path(data, "", 5)

    def test_set_path_index_type_error(self):
        """Setting index on non-indexable should raise PathError."""
        data = {"a": 42}  # int is not subscriptable
        with pytest.raises(PathError):
            set_path(data, "a[0]", 5)


class TestMirrorGetPath:
    """Test ObjectMirror.get_path() method."""

    def test_mirror_get_path(self):
        """Mirror.get_path() should navigate from mirrored object."""
        class Inventory:
            def __init__(self):
                self.items = [{"name": "sword", "damage": 10}]

        class Player:
            def __init__(self):
                self.inventory = Inventory()
                self.health = 100

        player = Player()
        m = mirror(player)

        assert m.get_path("health") == 100
        assert m.get_path("inventory.items[0].name") == "sword"
        assert m.get_path("inventory.items[0].damage") == 10

    def test_mirror_get_path_missing_raises(self):
        """Mirror.get_path() should raise PathError for missing path."""
        class Simple:
            x = 1

        obj = Simple()
        m = mirror(obj)
        with pytest.raises(PathError):
            m.get_path("missing.path")


class TestMirrorSetPath:
    """Test ObjectMirror.set_path() method."""

    def test_mirror_set_path(self):
        """Mirror.set_path() should mutate the mirrored object."""
        class Stats:
            strength = 10
            agility = 15

        class Player:
            def __init__(self):
                self.stats = Stats()
                self.name = "Hero"

        player = Player()
        m = mirror(player)

        m.set_path("name", "Champion")
        assert player.name == "Champion"

        m.set_path("stats.strength", 20)
        assert player.stats.strength == 20

    def test_mirror_set_path_list_item(self):
        """Mirror.set_path() should work with list indices."""
        class Container:
            def __init__(self):
                self.items = [1, 2, 3]

        obj = Container()
        m = mirror(obj)

        m.set_path("items[0]", 100)
        assert obj.items[0] == 100

        m.set_path("items[2]", 300)
        assert obj.items[2] == 300


class TestPathEdgeCases:
    """Test edge cases and special scenarios."""

    def test_path_with_underscore(self):
        """Paths with underscores should work."""
        data = {"my_key": {"sub_key": 42}}
        assert get_path(data, "my_key.sub_key") == 42

    def test_path_with_numbers_in_names(self):
        """Paths with numbers in names should work."""
        data = {"item1": {"value2": 10}}
        assert get_path(data, "item1.value2") == 10

    def test_multiple_set_operations(self):
        """Multiple set operations should all work."""
        data = {"a": 0, "b": 0, "c": {"d": 0}}
        set_path(data, "a", 1)
        set_path(data, "b", 2)
        set_path(data, "c.d", 3)
        assert data == {"a": 1, "b": 2, "c": {"d": 3}}

    def test_get_path_none_value(self):
        """Getting None value should return None (not raise)."""
        data = {"a": None}
        result = get_path(data, "a")
        assert result is None

    def test_set_path_none_value(self):
        """Setting None value should work."""
        data = {"a": 1}
        set_path(data, "a", None)
        assert data["a"] is None
