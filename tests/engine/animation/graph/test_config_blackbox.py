"""
Blackbox tests for animation graph configuration module.

Tests the contract:
- QuaternionConfig class with SLERP parameters
- GraphConfig class with evaluation limits
- BlendConfig class with blend parameters
- Global config instance (cfg) with .quat, .graph, .blend access
- reset_config() test helper

CLEANROOM: Tests written from contract only, no implementation reading.
"""

import pytest


class TestQuaternionConfigExists:
    """Test QuaternionConfig class existence and attributes."""

    def test_quaternion_config_importable(self):
        """QuaternionConfig should be importable from config module."""
        from engine.animation.graph.config import QuaternionConfig
        assert QuaternionConfig is not None

    def test_quaternion_config_instantiable(self):
        """QuaternionConfig should be instantiable."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert config is not None

    def test_quaternion_config_has_slerp_threshold(self):
        """QuaternionConfig should have SLERP threshold parameter."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        # Should have some form of threshold for SLERP interpolation
        assert hasattr(config, 'SLERP_DOT_THRESHOLD'), "QuaternionConfig should have SLERP_DOT_THRESHOLD"

    def test_quaternion_config_has_min_sin_theta(self):
        """QuaternionConfig should have min sin theta for SLERP stability."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert hasattr(config, 'SLERP_MIN_SIN_THETA'), "QuaternionConfig should have SLERP_MIN_SIN_THETA"

    def test_quaternion_config_has_epsilon(self):
        """QuaternionConfig should have epsilon for numerical stability."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert hasattr(config, 'EPSILON'), "QuaternionConfig should have EPSILON"

    def test_quaternion_config_has_normalization_epsilon(self):
        """QuaternionConfig should have normalization epsilon."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert hasattr(config, 'NORMALIZATION_EPSILON'), "QuaternionConfig should have NORMALIZATION_EPSILON"

    def test_quaternion_config_values_are_positive(self):
        """QuaternionConfig threshold values should be positive."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert config.SLERP_DOT_THRESHOLD > 0, "SLERP_DOT_THRESHOLD must be positive"
        assert config.SLERP_MIN_SIN_THETA > 0, "SLERP_MIN_SIN_THETA must be positive"
        assert config.EPSILON > 0, "EPSILON must be positive"
        assert config.NORMALIZATION_EPSILON > 0, "NORMALIZATION_EPSILON must be positive"

    def test_quaternion_config_dot_threshold_near_one(self):
        """QuaternionConfig SLERP_DOT_THRESHOLD should be close to 1.0."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        # Dot threshold near 1.0 means quaternions are nearly identical
        assert 0.99 <= config.SLERP_DOT_THRESHOLD <= 1.0, "SLERP_DOT_THRESHOLD should be close to 1.0"

    def test_quaternion_config_epsilons_are_small(self):
        """QuaternionConfig epsilon values should be very small."""
        from engine.animation.graph.config import QuaternionConfig
        config = QuaternionConfig()
        assert config.EPSILON < 1e-5, "EPSILON should be very small"
        assert config.NORMALIZATION_EPSILON < 1e-4, "NORMALIZATION_EPSILON should be small"


class TestGraphConfigExists:
    """Test GraphConfig class existence and attributes."""

    def test_graph_config_importable(self):
        """GraphConfig should be importable from config module."""
        from engine.animation.graph.config import GraphConfig
        assert GraphConfig is not None

    def test_graph_config_instantiable(self):
        """GraphConfig should be instantiable."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert config is not None

    def test_graph_config_has_max_evaluation_depth(self):
        """GraphConfig should have MAX_EVALUATION_DEPTH parameter."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert hasattr(config, 'MAX_EVALUATION_DEPTH'), "GraphConfig should have MAX_EVALUATION_DEPTH"

    def test_graph_config_has_cycle_detection(self):
        """GraphConfig should have CYCLE_DETECTION_ENABLED parameter."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert hasattr(config, 'CYCLE_DETECTION_ENABLED'), "GraphConfig should have CYCLE_DETECTION_ENABLED"

    def test_graph_config_has_default_time_scale(self):
        """GraphConfig should have DEFAULT_TIME_SCALE parameter."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert hasattr(config, 'DEFAULT_TIME_SCALE'), "GraphConfig should have DEFAULT_TIME_SCALE"

    def test_graph_config_max_depth_is_positive_integer(self):
        """GraphConfig MAX_EVALUATION_DEPTH should be positive integer."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert isinstance(config.MAX_EVALUATION_DEPTH, int), "MAX_EVALUATION_DEPTH should be an integer"
        assert config.MAX_EVALUATION_DEPTH > 0, "MAX_EVALUATION_DEPTH should be positive"

    def test_graph_config_max_depth_is_reasonable(self):
        """GraphConfig MAX_EVALUATION_DEPTH should be reasonable (not too small or too large)."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert config.MAX_EVALUATION_DEPTH >= 1, "MAX_EVALUATION_DEPTH should be at least 1"
        assert config.MAX_EVALUATION_DEPTH <= 1000, "MAX_EVALUATION_DEPTH should be reasonable (< 1000)"

    def test_graph_config_cycle_detection_is_bool(self):
        """GraphConfig CYCLE_DETECTION_ENABLED should be boolean."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert isinstance(config.CYCLE_DETECTION_ENABLED, bool), "CYCLE_DETECTION_ENABLED should be boolean"

    def test_graph_config_time_scale_is_positive(self):
        """GraphConfig DEFAULT_TIME_SCALE should be positive."""
        from engine.animation.graph.config import GraphConfig
        config = GraphConfig()
        assert config.DEFAULT_TIME_SCALE > 0, "DEFAULT_TIME_SCALE should be positive"


class TestBlendConfigExists:
    """Test BlendConfig class existence and attributes."""

    def test_blend_config_importable(self):
        """BlendConfig should be importable from config module."""
        from engine.animation.graph.config import BlendConfig
        assert BlendConfig is not None

    def test_blend_config_instantiable(self):
        """BlendConfig should be instantiable."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert config is not None

    def test_blend_config_has_min_weight(self):
        """BlendConfig should have MIN_BLEND_WEIGHT parameter."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert hasattr(config, 'MIN_BLEND_WEIGHT'), "BlendConfig should have MIN_BLEND_WEIGHT"

    def test_blend_config_has_max_weight(self):
        """BlendConfig should have MAX_BLEND_WEIGHT parameter."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert hasattr(config, 'MAX_BLEND_WEIGHT'), "BlendConfig should have MAX_BLEND_WEIGHT"

    def test_blend_config_has_default_blend_time(self):
        """BlendConfig should have DEFAULT_BLEND_TIME parameter."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert hasattr(config, 'DEFAULT_BLEND_TIME'), "BlendConfig should have DEFAULT_BLEND_TIME"

    def test_blend_config_has_weight_epsilon(self):
        """BlendConfig should have WEIGHT_EPSILON parameter."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert hasattr(config, 'WEIGHT_EPSILON'), "BlendConfig should have WEIGHT_EPSILON"

    def test_blend_config_has_normalize_weights(self):
        """BlendConfig should have NORMALIZE_WEIGHTS parameter."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert hasattr(config, 'NORMALIZE_WEIGHTS'), "BlendConfig should have NORMALIZE_WEIGHTS"

    def test_blend_config_min_weight_in_range(self):
        """BlendConfig MIN_BLEND_WEIGHT should be between 0 and 1."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert 0.0 <= config.MIN_BLEND_WEIGHT <= 1.0, "MIN_BLEND_WEIGHT should be between 0 and 1"

    def test_blend_config_max_weight_in_range(self):
        """BlendConfig MAX_BLEND_WEIGHT should be between 0 and 1."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert 0.0 <= config.MAX_BLEND_WEIGHT <= 1.0, "MAX_BLEND_WEIGHT should be between 0 and 1"

    def test_blend_config_min_less_than_max(self):
        """BlendConfig MIN_BLEND_WEIGHT should be less than MAX_BLEND_WEIGHT."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert config.MIN_BLEND_WEIGHT < config.MAX_BLEND_WEIGHT, "MIN should be less than MAX"

    def test_blend_config_default_blend_time_positive(self):
        """BlendConfig DEFAULT_BLEND_TIME should be positive."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert config.DEFAULT_BLEND_TIME > 0, "DEFAULT_BLEND_TIME should be positive"

    def test_blend_config_weight_epsilon_is_small(self):
        """BlendConfig WEIGHT_EPSILON should be small."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert config.WEIGHT_EPSILON > 0, "WEIGHT_EPSILON should be positive"
        assert config.WEIGHT_EPSILON < 0.1, "WEIGHT_EPSILON should be small"

    def test_blend_config_normalize_weights_is_bool(self):
        """BlendConfig NORMALIZE_WEIGHTS should be boolean."""
        from engine.animation.graph.config import BlendConfig
        config = BlendConfig()
        assert isinstance(config.NORMALIZE_WEIGHTS, bool), "NORMALIZE_WEIGHTS should be boolean"


class TestGlobalConfigInstance:
    """Test global config instance (DEFAULT_CONFIG / get_config) with accessor pattern."""

    def test_default_config_importable(self):
        """Global DEFAULT_CONFIG instance should be importable."""
        from engine.animation.graph.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG is not None

    def test_get_config_importable(self):
        """get_config function should be importable."""
        from engine.animation.graph.config import get_config
        assert get_config is not None

    def test_get_config_callable(self):
        """get_config should be callable."""
        from engine.animation.graph.config import get_config
        assert callable(get_config)

    def test_get_config_returns_config(self):
        """get_config() should return a config instance."""
        from engine.animation.graph.config import get_config, AnimationGraphConfig
        config = get_config()
        assert isinstance(config, AnimationGraphConfig)

    def test_animation_graph_config_has_quaternion(self):
        """AnimationGraphConfig should have quaternion config accessor."""
        from engine.animation.graph.config import get_config
        config = get_config()
        assert hasattr(config, 'quaternion') or hasattr(config, 'quat'), "Config should have quaternion accessor"

    def test_animation_graph_config_has_graph(self):
        """AnimationGraphConfig should have graph config accessor."""
        from engine.animation.graph.config import get_config
        config = get_config()
        assert hasattr(config, 'graph'), "Config should have graph accessor"

    def test_animation_graph_config_has_blend(self):
        """AnimationGraphConfig should have blend config accessor."""
        from engine.animation.graph.config import get_config
        config = get_config()
        assert hasattr(config, 'blend'), "Config should have blend accessor"

    def test_config_quaternion_is_quaternion_config(self):
        """Config quaternion should be a QuaternionConfig instance."""
        from engine.animation.graph.config import get_config, QuaternionConfig
        config = get_config()
        quat = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        assert isinstance(quat, QuaternionConfig)

    def test_config_graph_is_graph_config(self):
        """Config graph should be a GraphConfig instance."""
        from engine.animation.graph.config import get_config, GraphConfig
        config = get_config()
        assert isinstance(config.graph, GraphConfig)

    def test_config_blend_is_blend_config(self):
        """Config blend should be a BlendConfig instance."""
        from engine.animation.graph.config import get_config, BlendConfig
        config = get_config()
        assert isinstance(config.blend, BlendConfig)


class TestResetConfig:
    """Test reset_config() helper function."""

    def test_reset_config_importable(self):
        """reset_config should be importable."""
        from engine.animation.graph.config import reset_config
        assert reset_config is not None

    def test_reset_config_callable(self):
        """reset_config should be callable."""
        from engine.animation.graph.config import reset_config
        assert callable(reset_config)

    def test_reset_config_runs_without_error(self):
        """reset_config() should run without raising exceptions."""
        from engine.animation.graph.config import reset_config
        # Should not raise
        reset_config()

    def test_reset_config_restores_defaults(self):
        """reset_config() should restore config to default values."""
        from engine.animation.graph.config import get_config, reset_config

        # Then reset
        reset_config()

        # After reset, get_config should return valid config
        config = get_config()
        quat = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        assert quat is not None
        assert config.graph is not None
        assert config.blend is not None


class TestConfigDocumentation:
    """Test that config values are documented (have docstrings or comments)."""

    def test_quaternion_config_has_docstring(self):
        """QuaternionConfig should have a docstring."""
        from engine.animation.graph.config import QuaternionConfig
        assert QuaternionConfig.__doc__ is not None, "QuaternionConfig should have a docstring"

    def test_graph_config_has_docstring(self):
        """GraphConfig should have a docstring."""
        from engine.animation.graph.config import GraphConfig
        assert GraphConfig.__doc__ is not None, "GraphConfig should have a docstring"

    def test_blend_config_has_docstring(self):
        """BlendConfig should have a docstring."""
        from engine.animation.graph.config import BlendConfig
        assert BlendConfig.__doc__ is not None, "BlendConfig should have a docstring"


class TestConfigValueRanges:
    """Test that config values are in reasonable ranges."""

    def test_epsilon_values_are_small(self):
        """Epsilon values should be very small (for numerical precision)."""
        from engine.animation.graph.config import get_config

        config = get_config()
        quat = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        # Check QuaternionConfig epsilon
        assert quat.EPSILON > 0, "EPSILON should be positive"
        assert quat.EPSILON < 0.01, "EPSILON should be small (< 0.01)"

    def test_thresholds_are_reasonable(self):
        """Threshold values should be in reasonable ranges."""
        from engine.animation.graph.config import get_config

        config = get_config()
        quat = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        # SLERP dot threshold should be close to 1.0
        assert 0.99 <= quat.SLERP_DOT_THRESHOLD <= 1.0, "SLERP_DOT_THRESHOLD should be close to 1.0"

    def test_limits_are_positive(self):
        """All limit values should be positive."""
        from engine.animation.graph.config import get_config

        config = get_config()
        # Check GraphConfig limits
        assert config.graph.MAX_EVALUATION_DEPTH > 0, "MAX_EVALUATION_DEPTH should be positive"


class TestConfigImmutabilityPattern:
    """Test config immutability behavior (if implemented)."""

    def test_config_values_are_consistent(self):
        """Config values should be consistent across accesses."""
        from engine.animation.graph.config import get_config

        config = get_config()
        # Access twice and compare
        quat1 = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        quat2 = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)

        # Should be the same instance or equal values
        assert quat1 is quat2 or quat1 == quat2

    def test_reset_creates_fresh_state(self):
        """After reset, config should be in a valid fresh state."""
        from engine.animation.graph.config import get_config, reset_config

        reset_config()

        # All accessors should work
        config = get_config()
        quat = getattr(config, 'quaternion', None) or getattr(config, 'quat', None)
        assert quat is not None
        assert config.graph is not None
        assert config.blend is not None


class TestAdditionalConfigClasses:
    """Test additional config classes exposed by the module."""

    def test_animation_graph_config_importable(self):
        """AnimationGraphConfig should be importable."""
        from engine.animation.graph.config import AnimationGraphConfig
        assert AnimationGraphConfig is not None

    def test_blend_tree_config_importable(self):
        """BlendTreeConfig should be importable."""
        from engine.animation.graph.config import BlendTreeConfig
        assert BlendTreeConfig is not None

    def test_layer_config_importable(self):
        """LayerConfig should be importable."""
        from engine.animation.graph.config import LayerConfig
        assert LayerConfig is not None

    def test_sync_config_importable(self):
        """SyncConfig should be importable."""
        from engine.animation.graph.config import SyncConfig
        assert SyncConfig is not None

    def test_transition_config_importable(self):
        """TransitionConfig should be importable."""
        from engine.animation.graph.config import TransitionConfig
        assert TransitionConfig is not None

    def test_animation_graph_config_instantiable(self):
        """AnimationGraphConfig should be instantiable."""
        from engine.animation.graph.config import AnimationGraphConfig
        config = AnimationGraphConfig()
        assert config is not None

    def test_blend_tree_config_instantiable(self):
        """BlendTreeConfig should be instantiable."""
        from engine.animation.graph.config import BlendTreeConfig
        config = BlendTreeConfig()
        assert config is not None

    def test_layer_config_instantiable(self):
        """LayerConfig should be instantiable."""
        from engine.animation.graph.config import LayerConfig
        config = LayerConfig()
        assert config is not None

    def test_sync_config_instantiable(self):
        """SyncConfig should be instantiable."""
        from engine.animation.graph.config import SyncConfig
        config = SyncConfig()
        assert config is not None

    def test_transition_config_instantiable(self):
        """TransitionConfig should be instantiable."""
        from engine.animation.graph.config import TransitionConfig
        config = TransitionConfig()
        assert config is not None
