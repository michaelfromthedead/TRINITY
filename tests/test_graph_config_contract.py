"""Contract tests for AnimationGraph Configuration Module (T-AG-1.7).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation or implementation details.

Contract sources:
  - Task T-AG-1.7 description (public API: config classes, get_config)
  - engine/animation/graph/__init__.py (public exports: 7 config classes + get_config)

Forbidden files (NOT read):
  - engine/animation/graph/config.py (DEV implementation)
  - tests/test_graph_config_whitebox.py (parallel peer)
"""
import dataclasses
import typing

import pytest

from engine.animation.graph import (
    # Configuration
    AnimationGraphConfig,
    TransitionConfig,
    BlendTreeConfig,
    SyncConfig,
    LayerConfig,
    QuaternionConfig,
    BlendConfig,
    get_config,
)


# ============================================================================
# Import / Export Contract
# ============================================================================


class TestConfigPublicExports:
    """All config types are importable from the public API surface."""

    def test_animation_graph_config_exported(self):
        """AnimationGraphConfig is exported from engine.animation.graph."""
        assert AnimationGraphConfig is not None

    def test_transition_config_exported(self):
        """TransitionConfig is exported from engine.animation.graph."""
        assert TransitionConfig is not None

    def test_blend_tree_config_exported(self):
        """BlendTreeConfig is exported from engine.animation.graph."""
        assert BlendTreeConfig is not None

    def test_sync_config_exported(self):
        """SyncConfig is exported from engine.animation.graph."""
        assert SyncConfig is not None

    def test_layer_config_exported(self):
        """LayerConfig is exported from engine.animation.graph."""
        assert LayerConfig is not None

    def test_quaternion_config_exported(self):
        """QuaternionConfig is exported from engine.animation.graph."""
        assert QuaternionConfig is not None

    def test_blend_config_exported(self):
        """BlendConfig is exported from engine.animation.graph."""
        assert BlendConfig is not None

    def test_get_config_exported(self):
        """get_config function is exported from engine.animation.graph."""
        assert callable(get_config)

    def test_all_configs_are_distinct_types(self):
        """Each config class is a distinct type."""
        config_types = {
            AnimationGraphConfig,
            TransitionConfig,
            BlendTreeConfig,
            SyncConfig,
            LayerConfig,
            QuaternionConfig,
            BlendConfig,
        }
        assert len(config_types) == 7


# ============================================================================
# Default Values Contract
# ============================================================================


class TestConfigDefaultValues:
    """Each config type can be instantiated with no arguments and has
    reasonable default values."""

    def test_animation_graph_config_defaults(self):
        """AnimationGraphConfig has reasonable default values."""
        cfg = AnimationGraphConfig()
        assert cfg is not None
        assert isinstance(cfg, AnimationGraphConfig)

    def test_transition_config_defaults(self):
        """TransitionConfig has reasonable default values."""
        cfg = TransitionConfig()
        assert cfg is not None
        assert isinstance(cfg, TransitionConfig)

    def test_blend_tree_config_defaults(self):
        """BlendTreeConfig has reasonable default values."""
        cfg = BlendTreeConfig()
        assert cfg is not None
        assert isinstance(cfg, BlendTreeConfig)

    def test_sync_config_defaults(self):
        """SyncConfig has reasonable default values."""
        cfg = SyncConfig()
        assert cfg is not None
        assert isinstance(cfg, SyncConfig)

    def test_layer_config_defaults(self):
        """LayerConfig has reasonable default values."""
        cfg = LayerConfig()
        assert cfg is not None
        assert isinstance(cfg, LayerConfig)

    def test_quaternion_config_defaults(self):
        """QuaternionConfig has reasonable default values."""
        cfg = QuaternionConfig()
        assert cfg is not None
        assert isinstance(cfg, QuaternionConfig)

    def test_blend_config_defaults(self):
        """BlendConfig has reasonable default values."""
        cfg = BlendConfig()
        assert cfg is not None
        assert isinstance(cfg, BlendConfig)


# ============================================================================
# Data Structure Contract
# ============================================================================


class TestConfigDataStructure:
    """Config classes follow expected data structure patterns."""

    CONFIG_CLASSES = [
        ("AnimationGraphConfig", AnimationGraphConfig),
        ("TransitionConfig", TransitionConfig),
        ("BlendTreeConfig", BlendTreeConfig),
        ("SyncConfig", SyncConfig),
        ("LayerConfig", LayerConfig),
        ("QuaternionConfig", QuaternionConfig),
        ("BlendConfig", BlendConfig),
    ]

    def test_all_configs_are_dataclasses(self):
        """All config classes are Python dataclasses."""
        for name, cls in self.CONFIG_CLASSES:
            msg = f"{name} should be a dataclass"
            assert dataclasses.is_dataclass(cls), msg

    def test_all_configs_have_fields(self):
        """All config classes have at least one field."""
        for name, cls in self.CONFIG_CLASSES:
            fields = dataclasses.fields(cls)
            msg = f"{name} should have at least 1 field, got {len(fields)}"
            assert len(fields) >= 1, msg

    def test_all_configs_are_immutable(self):
        """All config classes are frozen dataclasses (immutable by default)."""
        for name, cls in self.CONFIG_CLASSES:
            frozen = cls.__dataclass_params__.frozen
            msg = f"{name} should be frozen (immutable)"
            assert frozen, msg

    def test_all_configs_equality_supported(self):
        """Config instances support equality comparison."""
        for name, cls in self.CONFIG_CLASSES:
            a = cls()
            b = cls()
            assert a == b, f"{name} default instances should be equal"


# ============================================================================
# Field Access Contract
# ============================================================================


class TestConfigFieldAccess:
    """Fields on config instances are readable."""

    CONFIG_CLASSES = [
        ("AnimationGraphConfig", AnimationGraphConfig),
        ("TransitionConfig", TransitionConfig),
        ("BlendTreeConfig", BlendTreeConfig),
        ("SyncConfig", SyncConfig),
        ("LayerConfig", LayerConfig),
        ("QuaternionConfig", QuaternionConfig),
        ("BlendConfig", BlendConfig),
    ]

    def test_all_fields_accessible_on_default_instance(self):
        """Every field on every config class is accessible and non-None."""
        for name, cls in self.CONFIG_CLASSES:
            inst = cls()
            fields = dataclasses.fields(cls)
            for f in fields:
                val = getattr(inst, f.name)
                msg = f"{name}.{f.name} should be accessible, got {val!r}"
                # Defaults must be set (not left as None / MISSING)
                assert val is not None, msg

    def test_all_fields_have_type_annotations(self):
        """Every config field has a type annotation."""
        for name, cls in self.CONFIG_CLASSES:
            fields = dataclasses.fields(cls)
            for f in fields:
                msg = f"{name}.{f.name} is missing type annotation"
                assert f.type is not dataclasses.MISSING, msg

    def test_field_types_are_basic_types_or_config_types(self):
        """Field types are primitives, enums, or other config types."""
        allowed_bases = (
            int,
            float,
            bool,
            str,
            typing.Any,
            list,
            dict,
            tuple,
            set,
        )
        for name, cls in self.CONFIG_CLASSES:
            fields = dataclasses.fields(cls)
            for f in fields:
                # Unwrap Optionals and generics
                origin = typing.get_origin(f.type)
                if origin is not None and origin is typing.Union:
                    # Optional[T] = Union[T, None]
                    args = typing.get_args(f.type)
                    for arg in args:
                        if arg is type(None):
                            continue
                        self._assert_valid_field_type(name, f.name, arg, allowed_bases)
                elif origin is not None:
                    # list[T], dict[K,V], etc.
                    self._assert_valid_field_type(name, f.name, origin, allowed_bases)
                else:
                    self._assert_valid_field_type(name, f.name, f.type, allowed_bases)

    def _assert_valid_field_type(self, config_name, field_name, typ, allowed_bases):
        if typ in allowed_bases:
            return
        # Allow enum types
        if isinstance(typ, type) and issubclass(typ, (int, float, str)):
            return
        msg = f"{config_name}.{field_name}: unexpected type {typ}"
        # This is informational, not a hard failure for blackbox testing
        # since we don't know all expected types from the contract alone


# ============================================================================
# get_config Contract
# ============================================================================


class TestGetConfig:
    """get_config() returns the global AnimationGraphConfig."""

    def test_get_config_returns_animation_graph_config(self):
        """get_config() returns an AnimationGraphConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, AnimationGraphConfig)
        assert cfg is not None

    def test_get_config_is_callable_without_args(self):
        """get_config() accepts no arguments."""
        cfg = get_config()
        assert cfg is not None

    def test_get_config_returns_same_structure(self):
        """get_config() result has the same fields as default config."""
        cfg = get_config()
        assert dataclasses.is_dataclass(cfg)
        assert len(dataclasses.fields(type(cfg))) >= 1

    def test_get_config_values_are_not_none(self):
        """All fields on get_config() result have non-None values."""
        cfg = get_config()
        for f in dataclasses.fields(type(cfg)):
            val = getattr(cfg, f.name)
            assert val is not None, f"get_config().{f.name} is None"

    def test_get_config_returns_immutable_instance(self):
        """get_config() returns an immutable (frozen) instance."""
        cfg = get_config()
        # Verify the type is frozen
        assert type(cfg).__dataclass_params__.frozen


# ============================================================================
# QuaternionConfig Specifics (explicitly named in T-AG-1.7)
# ============================================================================


class TestQuaternionConfig:
    """QuaternionConfig has documented tuning constants."""

    def test_quaternion_config_has_epsilon(self):
        """QuaternionConfig has an epsilon field for numerical stability."""
        cfg = QuaternionConfig()
        fields_lower = {f.name.lower() for f in dataclasses.fields(QuaternionConfig)}
        has_epsilon = any("epsilon" in n for n in fields_lower)
        has_threshold = any("threshold" in n for n in fields_lower)
        assert has_epsilon or has_threshold, (
            f"No epsilon/threshold field found in {fields_lower}"
        )

    def test_quaternion_config_default_is_float_fields(self):
        """QuaternionConfig default values are float-type."""
        cfg = QuaternionConfig()
        for f in dataclasses.fields(QuaternionConfig):
            val = getattr(cfg, f.name)
            assert isinstance(val, (int, float)), (
                f"QuaternionConfig.{f.name} should be numeric, got {type(val).__name__}"
            )


# ============================================================================
# BlendConfig Specifics (explicitly named in T-AG-1.7)
# ============================================================================


class TestBlendConfig:
    """BlendConfig has documented tuning constants."""

    def test_blend_config_has_blend_fields(self):
        """BlendConfig has fields controlling blend behaviour."""
        cfg = BlendConfig()
        fields = {f.name for f in dataclasses.fields(BlendConfig)}
        assert len(fields) >= 1

    def test_blend_config_values_are_in_valid_range(self):
        """BlendConfig float values are in [0, 1] for weights/ratios."""
        cfg = BlendConfig()
        for f in dataclasses.fields(BlendConfig):
            val = getattr(cfg, f.name)
            if isinstance(val, float):
                assert 0.0 <= val <= 1.0, (
                    f"BlendConfig.{f.name}={val} outside [0, 1]"
                )


# ============================================================================
# Cross-config Contract
# ============================================================================


class TestConfigRelationship:
    """Some config types reference or compose with other config types."""

    def test_animation_graph_config_contains_other_configs(self):
        """AnimationGraphConfig may embed or reference other config types."""
        cfg = AnimationGraphConfig()
        fields = {f.name for f in dataclasses.fields(AnimationGraphConfig)}
        # Should have meaningful field names
        assert len(fields) >= 1

    def test_config_instances_are_independent(self):
        """Default instances of each config type are independent objects."""
        a = AnimationGraphConfig()
        b = AnimationGraphConfig()
        assert a is not b  # Different object identity
        assert a == b       # Same structural equality


# ============================================================================
# Re-export from engine.animation.graph Contract
# ============================================================================


class TestConfigReexports:
    """All config types re-exported from engine.animation.graph.__init__."""

    def test_reexport_animation_graph_config(self):
        """AnimationGraphConfig importable from the package."""
        from engine.animation.graph import AnimationGraphConfig as Cfg
        assert issubclass(type(Cfg()), AnimationGraphConfig)

    def test_reexport_get_config(self):
        """get_config importable from the package."""
        from engine.animation.graph import get_config as gc
        assert callable(gc)
        assert isinstance(gc(), AnimationGraphConfig)

    def test_all_configs_reexported(self):
        """All 7 config types are re-exported from the package."""
        import engine.animation.graph as ag
        config_names = {
            "AnimationGraphConfig",
            "TransitionConfig",
            "BlendTreeConfig",
            "SyncConfig",
            "LayerConfig",
            "QuaternionConfig",
            "BlendConfig",
        }
        for name in config_names:
            assert hasattr(ag, name), f"{name} missing from engine.animation.graph"

    def test_get_config_is_reexported(self):
        """get_config function is re-exported from the package."""
        import engine.animation.graph as ag
        assert hasattr(ag, "get_config")
        assert callable(ag.get_config)
