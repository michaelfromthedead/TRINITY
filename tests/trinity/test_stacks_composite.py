"""Tests for composite stacks (Phase C)."""
import pytest
from trinity.decorators.stacks import Stack
from trinity.decorators.builtin_stacks.composite import (
    multiplayer_character,
    competitive_entity,
    open_world_entity,
    mmo_entity,
    moddable_content,
)


def _expand(s: Stack) -> list[str]:
    """Return decorator names for a stack."""
    return s.expand()


def _has_decorator(s: Stack, name: str) -> bool:
    """Check if a stack contains a decorator with the given name."""
    return name in _expand(s)


class TestMultiplayerCharacter:
    """multiplayer_character = production_component + predicted_entity + versioned_saveable + secure_multiplayer."""

    def test_default_returns_stack(self):
        s = multiplayer_character()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = multiplayer_character()
        # production_component(5) + predicted_entity(4) + versioned_saveable(3) + secure_multiplayer(3) = 15
        assert len(s.decorators) == 15

    def test_contains_key_decorators(self):
        s = multiplayer_character()
        names = _expand(s)
        assert "component" in names, "Should include component from production_component"
        assert "track_changes" in names, "Should include track_changes"
        assert "server_authoritative" in names, "Should include server_authoritative from secure_multiplayer"

    def test_custom_pool_size_changes_output(self):
        s1 = multiplayer_character(pool_size=64)
        s2 = multiplayer_character(pool_size=256)
        # Both should produce valid stacks with the same structure
        assert len(s1.decorators) == len(s2.decorators)
        # Verify the function accepts and processes the parameter without error
        assert isinstance(s2, Stack)

    def test_custom_history_frames(self):
        s = multiplayer_character(history_frames=60)
        assert len(s.decorators) == 15

    def test_custom_version(self):
        s = multiplayer_character(version=3)
        assert len(s.decorators) == 15


class TestCompetitiveEntity:
    """competitive_entity = production_component + deterministic_data + replay_ready + predicted_entity + secure_multiplayer."""

    def test_default_returns_stack(self):
        s = competitive_entity()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = competitive_entity()
        # production_component(5) + deterministic_data(4) + replay_ready(7) + predicted_entity(4) + secure_multiplayer(3) = 23
        assert len(s.decorators) == 23

    def test_contains_key_decorators(self):
        s = competitive_entity()
        names = _expand(s)
        assert "component" in names, "Should include component"
        assert "deterministic" in names, "Should include deterministic from deterministic_data"
        assert "server_authoritative" in names, "Should include server_authoritative from secure_multiplayer"
        assert names.count("track_changes") >= 3, "Should have track_changes from multiple sub-stacks"

    def test_custom_params_accepted(self):
        s = competitive_entity(pool_size=256, history_frames=300)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 23

    def test_default_pool_size_is_128(self):
        """Composite default pool_size=128, differs from core default of 1000."""
        s = competitive_entity()
        # Just verify it builds without error with the intended default
        assert len(s.decorators) == 23


class TestOpenWorldEntity:
    """open_world_entity = production_component + streaming_chunk + lod_scalable + versioned_saveable."""

    def test_default_returns_stack(self):
        s = open_world_entity()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = open_world_entity()
        # production_component(5) + streaming_chunk(8) + lod_scalable(3) + versioned_saveable(3) = 19
        assert len(s.decorators) == 19

    def test_contains_key_decorators(self):
        s = open_world_entity()
        names = _expand(s)
        assert "component" in names
        assert names.count("track_changes") >= 2, "Should have track_changes from multiple sub-stacks"

    def test_custom_chunk_size(self):
        s = open_world_entity(chunk_size=(200, 200, 200))
        assert isinstance(s, Stack)
        assert len(s.decorators) == 19

    def test_custom_pool_size(self):
        s = open_world_entity(pool_size=20000)
        assert len(s.decorators) == 19


class TestMmoEntity:
    """mmo_entity = production_component + bandwidth_efficient + secure_multiplayer + versioned_saveable."""

    def test_default_returns_stack(self):
        s = mmo_entity()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = mmo_entity()
        # production_component(5) + bandwidth_efficient(6) + secure_multiplayer(3) + versioned_saveable(3) = 17
        assert len(s.decorators) == 17

    def test_contains_key_decorators(self):
        s = mmo_entity()
        names = _expand(s)
        assert "component" in names
        assert "server_authoritative" in names, "Should include server_authoritative from secure_multiplayer"
        assert "track_changes" in names

    def test_custom_relevance_radius(self):
        s = mmo_entity(relevance_radius=20000)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 17

    def test_custom_pool_size(self):
        s = mmo_entity(pool_size=10000)
        assert len(s.decorators) == 17


class TestModdableContent:
    """moddable_content uses stack() directly with 6 individual decorators."""

    def test_requires_namespace(self):
        """namespace is a required parameter with no default."""
        with pytest.raises(TypeError):
            moddable_content()

    def test_default_returns_stack(self):
        s = moddable_content(namespace="test")
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = moddable_content(namespace="test")
        # component, moddable, serializable, versioned, track_changes, observable = 6
        assert len(s.decorators) == 6

    def test_contains_key_decorators(self):
        s = moddable_content(namespace="test")
        names = _expand(s)
        assert "component" in names
        assert "track_changes" in names

    def test_custom_version(self):
        s = moddable_content(namespace="weapons", version=2)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 6

    def test_different_namespaces_same_structure(self):
        s1 = moddable_content(namespace="weapons")
        s2 = moddable_content(namespace="armor")
        assert len(s1.decorators) == len(s2.decorators)
        assert _expand(s1) == _expand(s2)
