"""
Tests for Trinity Pattern Tier 33: SAVE_SYSTEM decorators.
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.save_system import (
    VALID_CONFLICT_RESOLUTIONS,
    atomic_save,
    cloud_sync,
    save_migration,
    save_slot,
)


# =============================================================================
# @save_slot tests
# =============================================================================


def test_save_slot_basic():
    """Test basic @save_slot application."""

    @save_slot(max_slots=5, auto_save=True, auto_save_interval=60.0)
    class SaveManager:
        pass

    assert hasattr(SaveManager, "_save_slot")
    assert SaveManager._save_slot is True
    assert SaveManager._save_max_slots == 5
    assert SaveManager._save_auto_save is True
    assert SaveManager._save_auto_save_interval == 60.0
    assert "save_slot" in SaveManager._applied_decorators


def test_save_slot_defaults():
    """Test @save_slot with default parameters."""

    @save_slot
    class SaveManager:
        pass

    assert SaveManager._save_max_slots == 10
    assert SaveManager._save_auto_save is True
    assert SaveManager._save_auto_save_interval == 300.0


def test_save_slot_auto_save_disabled():
    """Test @save_slot with auto_save disabled."""

    @save_slot(max_slots=3, auto_save=False)
    class SaveManager:
        pass

    assert SaveManager._save_auto_save is False


def test_save_slot_invalid_max_slots():
    """Test @save_slot with invalid max_slots raises ValueError."""
    with pytest.raises(ValueError, match="max_slots must be a positive integer"):

        @save_slot(max_slots=0)
        class SaveManager:
            pass

    with pytest.raises(ValueError, match="max_slots must be a positive integer"):

        @save_slot(max_slots=-5)
        class SaveManager:
            pass


def test_save_slot_invalid_interval():
    """Test @save_slot with invalid auto_save_interval raises ValueError."""
    with pytest.raises(ValueError, match="auto_save_interval must be a positive number"):

        @save_slot(auto_save_interval=0)
        class SaveManager:
            pass

    with pytest.raises(ValueError, match="auto_save_interval must be a positive number"):

        @save_slot(auto_save_interval=-10.0)
        class SaveManager:
            pass


def test_save_slot_registry():
    """Test that @save_slot registers properly."""

    @save_slot
    class SaveManager:
        pass

    assert "save_system" in SaveManager._registries


def test_save_slot_steps():
    """Test that @save_slot generates correct steps."""
    steps = decompose(save_slot)

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops
    # Should have 5 steps: 4 TAGs + 1 REGISTER
    assert len(steps) == 5


def test_save_slot_registry_spec():
    """Test that @save_slot is registered in the decorator registry."""
    spec = registry.get("save_slot")
    assert spec is not None
    assert spec.name == "save_slot"
    assert spec.tier == Tier.SAVE_SYSTEM
    assert spec.unique is True
    assert spec.foundation is False


# =============================================================================
# @atomic_save tests
# =============================================================================


def test_atomic_save_basic():
    """Test basic @atomic_save application."""

    @atomic_save
    class SaveHandler:
        pass

    assert hasattr(SaveHandler, "_atomic_save")
    assert SaveHandler._atomic_save is True
    assert "atomic_save" in SaveHandler._applied_decorators


def test_atomic_save_tags():
    """Test that @atomic_save sets proper tags."""

    @atomic_save
    class SaveHandler:
        pass

    assert hasattr(SaveHandler, "_tags")
    assert SaveHandler._tags.get("atomic_save") is True


def test_atomic_save_registry():
    """Test that @atomic_save registers properly."""

    @atomic_save
    class SaveHandler:
        pass

    assert "save_system" in SaveHandler._registries


def test_atomic_save_steps():
    """Test that @atomic_save generates correct steps."""
    steps = decompose(atomic_save)

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops
    # Should have 2 steps: TAG + REGISTER
    assert len(steps) == 2


def test_atomic_save_registry_spec():
    """Test that @atomic_save is registered in the decorator registry."""
    spec = registry.get("atomic_save")
    assert spec is not None
    assert spec.name == "atomic_save"
    assert spec.tier == Tier.SAVE_SYSTEM
    assert spec.unique is True


# =============================================================================
# @cloud_sync tests
# =============================================================================


def test_cloud_sync_basic():
    """Test basic @cloud_sync application."""

    @cloud_sync(platform="steam", conflict_resolution="newest")
    class CloudSaveManager:
        pass

    assert hasattr(CloudSaveManager, "_cloud_sync")
    assert CloudSaveManager._cloud_sync is True
    assert CloudSaveManager._cloud_platform == "steam"
    assert CloudSaveManager._cloud_conflict_resolution == "newest"
    assert "cloud_sync" in CloudSaveManager._applied_decorators


def test_cloud_sync_all_resolutions():
    """Test all valid conflict resolution strategies."""
    for resolution in VALID_CONFLICT_RESOLUTIONS:

        @cloud_sync(platform="test", conflict_resolution=resolution)
        class Manager:
            pass

        assert Manager._cloud_conflict_resolution == resolution


def test_cloud_sync_invalid_platform():
    """Test @cloud_sync with invalid platform raises ValueError."""
    with pytest.raises(ValueError, match="platform must be a non-empty string"):

        @cloud_sync(platform="", conflict_resolution="newest")
        class Manager:
            pass


def test_cloud_sync_invalid_resolution():
    """Test @cloud_sync with invalid conflict_resolution raises ValueError."""
    with pytest.raises(ValueError, match="Invalid conflict_resolution"):

        @cloud_sync(platform="steam", conflict_resolution="invalid")
        class Manager:
            pass


def test_cloud_sync_tags():
    """Test that @cloud_sync sets proper tags."""

    @cloud_sync(platform="epic", conflict_resolution="merge")
    class Manager:
        pass

    assert hasattr(Manager, "_tags")
    assert Manager._tags.get("cloud_sync") is True
    assert Manager._tags.get("cloud_platform") == "epic"
    assert Manager._tags.get("cloud_conflict_resolution") == "merge"


def test_cloud_sync_registry():
    """Test that @cloud_sync registers properly."""

    @cloud_sync(platform="gog", conflict_resolution="ask_player")
    class Manager:
        pass

    assert "save_system" in Manager._registries


def test_cloud_sync_steps():
    """Test that @cloud_sync generates correct steps."""
    steps = decompose(cloud_sync)

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops
    # Should have 4 steps: 3 TAGs + 1 REGISTER
    assert len(steps) == 4


def test_cloud_sync_registry_spec():
    """Test that @cloud_sync is registered in the decorator registry."""
    spec = registry.get("cloud_sync")
    assert spec is not None
    assert spec.name == "cloud_sync"
    assert spec.tier == Tier.SAVE_SYSTEM
    assert spec.unique is True


# =============================================================================
# @save_migration tests
# =============================================================================


def test_save_migration_basic():
    """Test basic @save_migration application."""

    @save_migration(from_version=1, to_version=2)
    class MigrationHandler:
        pass

    assert hasattr(MigrationHandler, "_save_migration")
    assert MigrationHandler._save_migration is True
    assert MigrationHandler._migration_from_version == 1
    assert MigrationHandler._migration_to_version == 2
    assert "save_migration" in MigrationHandler._applied_decorators


def test_save_migration_multiple():
    """Test multiple @save_migration decorators can be applied."""

    @save_migration(from_version=2, to_version=3)
    @save_migration(from_version=1, to_version=2)
    class MigrationChain:
        pass

    # The decorator name appears once in _applied_decorators (framework behavior)
    # but the decorator can be applied multiple times (unique=False in registry)
    assert "save_migration" in MigrationChain._applied_decorators
    # The last application's values are retained (decorators applied bottom-up)
    assert MigrationChain._migration_from_version == 2
    assert MigrationChain._migration_to_version == 3


def test_save_migration_invalid_from_version():
    """Test @save_migration with invalid from_version raises ValueError."""
    with pytest.raises(ValueError, match="from_version must be >= 0"):

        @save_migration(from_version=-1, to_version=1)
        class Handler:
            pass


def test_save_migration_invalid_to_version():
    """Test @save_migration with to_version <= from_version raises ValueError."""
    with pytest.raises(ValueError, match="to_version must be > from_version"):

        @save_migration(from_version=5, to_version=5)
        class Handler:
            pass

    with pytest.raises(ValueError, match="to_version must be > from_version"):

        @save_migration(from_version=5, to_version=3)
        class Handler:
            pass


def test_save_migration_tags():
    """Test that @save_migration sets proper tags."""

    @save_migration(from_version=0, to_version=1)
    class Handler:
        pass

    assert hasattr(Handler, "_tags")
    assert Handler._tags.get("save_migration") is True
    assert Handler._tags.get("migration_from_version") == 0
    assert Handler._tags.get("migration_to_version") == 1


def test_save_migration_registry():
    """Test that @save_migration registers properly."""

    @save_migration(from_version=0, to_version=1)
    class Handler:
        pass

    assert "save_system" in Handler._registries


def test_save_migration_steps():
    """Test that @save_migration generates correct steps."""
    steps = decompose(save_migration)

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops
    # Should have 4 steps: 3 TAGs + 1 REGISTER
    assert len(steps) == 4


def test_save_migration_registry_spec():
    """Test that @save_migration is registered in the decorator registry."""
    spec = registry.get("save_migration")
    assert spec is not None
    assert spec.name == "save_migration"
    assert spec.tier == Tier.SAVE_SYSTEM
    assert spec.unique is False  # Multiple migrations allowed


# =============================================================================
# Composition tests
# =============================================================================


def test_save_system_composition():
    """Test combining multiple save system decorators."""

    @save_slot(max_slots=5)
    @atomic_save
    @cloud_sync(platform="steam", conflict_resolution="newest")
    class FullSaveSystem:
        pass

    assert FullSaveSystem._save_slot is True
    assert FullSaveSystem._atomic_save is True
    assert FullSaveSystem._cloud_sync is True
    assert "save_slot" in FullSaveSystem._applied_decorators
    assert "atomic_save" in FullSaveSystem._applied_decorators
    assert "cloud_sync" in FullSaveSystem._applied_decorators


def test_save_with_other_tiers():
    """Test save system decorators with other tier decorators."""
    from trinity.decorators.compilation import native

    @save_slot
    @native(backend="cython")
    class OptimizedSaveSystem:
        pass

    assert OptimizedSaveSystem._save_slot is True
    assert OptimizedSaveSystem._native is True


# =============================================================================
# Module exports tests
# =============================================================================


def test_module_exports():
    """Test that module exports expected symbols."""
    from trinity.decorators import save_system

    assert hasattr(save_system, "save_slot")
    assert hasattr(save_system, "atomic_save")
    assert hasattr(save_system, "cloud_sync")
    assert hasattr(save_system, "save_migration")
    assert hasattr(save_system, "VALID_CONFLICT_RESOLUTIONS")

    assert "save_slot" in save_system.__all__
    assert "atomic_save" in save_system.__all__
    assert "cloud_sync" in save_system.__all__
    assert "save_migration" in save_system.__all__
    assert "VALID_CONFLICT_RESOLUTIONS" in save_system.__all__


# =============================================================================
# Integration tests
# =============================================================================


def test_save_system_introspection():
    """Test introspection of save system decorated classes."""
    from trinity.decorators.registry import get_decorator_chain, inspect_decorated

    @save_slot(max_slots=3)
    @atomic_save
    class SaveSystem:
        pass

    chain = get_decorator_chain(SaveSystem)
    assert "save_slot" in chain
    assert "atomic_save" in chain

    info = inspect_decorated(SaveSystem)
    assert "save_slot" in info.decorators
    assert "atomic_save" in info.decorators
    assert info.attributes.get("_save_slot") is True
    assert info.attributes.get("_atomic_save") is True


def test_save_system_preserves_existing_tags():
    """Test that save system decorators preserve existing tags."""

    class SaveManager:
        _tags = {"custom": "value"}

    save_slot()(SaveManager)

    assert SaveManager._tags.get("custom") == "value"
    assert SaveManager._tags.get("save_slot") is True
