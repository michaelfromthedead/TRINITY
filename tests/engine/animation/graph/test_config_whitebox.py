"""
Whitebox tests for animation graph configuration module.

Tests internal structure, default values, environment variable overrides,
and documentation presence for all configuration classes.

Task: T-AG-1.7 Configuration Module Whitebox Testing
"""

import os
import pytest
from dataclasses import fields, is_dataclass
from typing import get_type_hints


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up animation config environment variables before/after each test."""
    env_vars = [
        "TRINITY_ANIM_MAX_EVAL_DEPTH",
        "TRINITY_ANIM_CYCLE_DETECTION",
        "TRINITY_ANIM_DEFAULT_BLEND_TIME",
        "TRINITY_ANIM_DEFAULT_TIME_SCALE",
        "TRINITY_ANIM_SLERP_THRESHOLD",
    ]
    # Save original values
    original = {k: os.environ.get(k) for k in env_vars}

    # Clear env vars before test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values after test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
        if original[var] is not None:
            os.environ[var] = original[var]

    # Reset config to clean state
    from engine.animation.graph.config import reset_config
    reset_config()


@pytest.fixture
def fresh_config():
    """Provide a fresh config instance after resetting globals."""
    from engine.animation.graph.config import reset_config, get_config
    reset_config()
    return get_config()


# =============================================================================
# QuaternionConfig Tests
# =============================================================================


class TestQuaternionConfigWhitebox:
    """Whitebox tests for QuaternionConfig class."""

    def test_slerp_dot_threshold_default_value(self, fresh_config):
        """SLERP_DOT_THRESHOLD should default to 0.9995."""
        quat_config = fresh_config.quaternion
        assert quat_config.SLERP_DOT_THRESHOLD == 0.9995

    def test_slerp_dot_threshold_type(self, fresh_config):
        """SLERP_DOT_THRESHOLD should be a float."""
        quat_config = fresh_config.quaternion
        assert isinstance(quat_config.SLERP_DOT_THRESHOLD, float)

    def test_epsilon_default_value(self, fresh_config):
        """EPSILON should default to 1e-7 for numerical stability."""
        quat_config = fresh_config.quaternion
        assert quat_config.EPSILON == 1e-7

    def test_epsilon_type(self, fresh_config):
        """EPSILON should be a float."""
        quat_config = fresh_config.quaternion
        assert isinstance(quat_config.EPSILON, float)

    def test_slerp_min_sin_theta_default_value(self, fresh_config):
        """SLERP_MIN_SIN_THETA should default to 0.0001."""
        quat_config = fresh_config.quaternion
        assert quat_config.SLERP_MIN_SIN_THETA == 0.0001

    def test_slerp_min_sin_theta_type(self, fresh_config):
        """SLERP_MIN_SIN_THETA should be a float."""
        quat_config = fresh_config.quaternion
        assert isinstance(quat_config.SLERP_MIN_SIN_THETA, float)

    def test_normalization_epsilon_default_value(self, fresh_config):
        """NORMALIZATION_EPSILON should default to 1e-6."""
        quat_config = fresh_config.quaternion
        assert quat_config.NORMALIZATION_EPSILON == 1e-6

    def test_normalization_epsilon_type(self, fresh_config):
        """NORMALIZATION_EPSILON should be a float."""
        quat_config = fresh_config.quaternion
        assert isinstance(quat_config.NORMALIZATION_EPSILON, float)

    def test_slerp_dot_threshold_is_less_than_one(self, fresh_config):
        """SLERP_DOT_THRESHOLD must be less than 1.0 (cos of 0 angle)."""
        quat_config = fresh_config.quaternion
        assert quat_config.SLERP_DOT_THRESHOLD < 1.0

    def test_slerp_dot_threshold_is_close_to_one(self, fresh_config):
        """SLERP_DOT_THRESHOLD should be close to 1.0 for near-parallel quats."""
        quat_config = fresh_config.quaternion
        assert quat_config.SLERP_DOT_THRESHOLD > 0.99

    def test_epsilon_is_positive(self, fresh_config):
        """EPSILON must be a positive small value."""
        quat_config = fresh_config.quaternion
        assert quat_config.EPSILON > 0.0
        assert quat_config.EPSILON < 1e-5

    def test_normalization_epsilon_greater_than_epsilon(self, fresh_config):
        """NORMALIZATION_EPSILON should be larger than EPSILON (less strict)."""
        quat_config = fresh_config.quaternion
        assert quat_config.NORMALIZATION_EPSILON > quat_config.EPSILON

    def test_slerp_min_sin_theta_is_small_positive(self, fresh_config):
        """SLERP_MIN_SIN_THETA should be a small positive value."""
        quat_config = fresh_config.quaternion
        assert quat_config.SLERP_MIN_SIN_THETA > 0.0
        assert quat_config.SLERP_MIN_SIN_THETA < 0.01


# =============================================================================
# GraphConfig Tests
# =============================================================================


class TestGraphConfigWhitebox:
    """Whitebox tests for GraphConfig class."""

    def test_max_evaluation_depth_default_value(self, fresh_config):
        """MAX_EVALUATION_DEPTH should default to 100."""
        graph_config = fresh_config.graph
        assert graph_config.MAX_EVALUATION_DEPTH == 100

    def test_max_evaluation_depth_type(self, fresh_config):
        """MAX_EVALUATION_DEPTH should be an integer."""
        graph_config = fresh_config.graph
        assert isinstance(graph_config.MAX_EVALUATION_DEPTH, int)

    def test_cycle_detection_enabled_default_value(self, fresh_config):
        """CYCLE_DETECTION_ENABLED should default to True."""
        graph_config = fresh_config.graph
        assert graph_config.CYCLE_DETECTION_ENABLED is True

    def test_cycle_detection_enabled_type(self, fresh_config):
        """CYCLE_DETECTION_ENABLED should be a boolean."""
        graph_config = fresh_config.graph
        assert isinstance(graph_config.CYCLE_DETECTION_ENABLED, bool)

    def test_default_time_scale_default_value(self, fresh_config):
        """DEFAULT_TIME_SCALE should default to 1.0."""
        graph_config = fresh_config.graph
        assert graph_config.DEFAULT_TIME_SCALE == 1.0

    def test_default_time_scale_type(self, fresh_config):
        """DEFAULT_TIME_SCALE should be a float."""
        graph_config = fresh_config.graph
        assert isinstance(graph_config.DEFAULT_TIME_SCALE, float)

    def test_max_evaluation_depth_is_positive(self, fresh_config):
        """MAX_EVALUATION_DEPTH must be positive to allow any recursion."""
        graph_config = fresh_config.graph
        assert graph_config.MAX_EVALUATION_DEPTH > 0

    def test_max_evaluation_depth_is_reasonable(self, fresh_config):
        """MAX_EVALUATION_DEPTH should be reasonable (not too deep)."""
        graph_config = fresh_config.graph
        assert graph_config.MAX_EVALUATION_DEPTH <= 1000

    def test_default_time_scale_is_positive(self, fresh_config):
        """DEFAULT_TIME_SCALE must be positive for forward playback."""
        graph_config = fresh_config.graph
        assert graph_config.DEFAULT_TIME_SCALE > 0.0


# =============================================================================
# BlendConfig Tests
# =============================================================================


class TestBlendConfigWhitebox:
    """Whitebox tests for BlendConfig class."""

    def test_default_blend_time_default_value(self, fresh_config):
        """DEFAULT_BLEND_TIME should default to 0.25 seconds."""
        blend_config = fresh_config.blend
        assert blend_config.DEFAULT_BLEND_TIME == 0.25

    def test_default_blend_time_type(self, fresh_config):
        """DEFAULT_BLEND_TIME should be a float."""
        blend_config = fresh_config.blend
        assert isinstance(blend_config.DEFAULT_BLEND_TIME, float)

    def test_min_blend_weight_default_value(self, fresh_config):
        """MIN_BLEND_WEIGHT should equal 0.0."""
        blend_config = fresh_config.blend
        assert blend_config.MIN_BLEND_WEIGHT == 0.0

    def test_min_blend_weight_type(self, fresh_config):
        """MIN_BLEND_WEIGHT should be a float."""
        blend_config = fresh_config.blend
        assert isinstance(blend_config.MIN_BLEND_WEIGHT, float)

    def test_max_blend_weight_default_value(self, fresh_config):
        """MAX_BLEND_WEIGHT should equal 1.0."""
        blend_config = fresh_config.blend
        assert blend_config.MAX_BLEND_WEIGHT == 1.0

    def test_max_blend_weight_type(self, fresh_config):
        """MAX_BLEND_WEIGHT should be a float."""
        blend_config = fresh_config.blend
        assert isinstance(blend_config.MAX_BLEND_WEIGHT, float)

    def test_weight_epsilon_default_value(self, fresh_config):
        """WEIGHT_EPSILON should default to 0.001."""
        blend_config = fresh_config.blend
        assert blend_config.WEIGHT_EPSILON == 0.001

    def test_weight_epsilon_type(self, fresh_config):
        """WEIGHT_EPSILON should be a float."""
        blend_config = fresh_config.blend
        assert isinstance(blend_config.WEIGHT_EPSILON, float)

    def test_min_less_than_max_weight(self, fresh_config):
        """MIN_BLEND_WEIGHT must be less than MAX_BLEND_WEIGHT."""
        blend_config = fresh_config.blend
        assert blend_config.MIN_BLEND_WEIGHT < blend_config.MAX_BLEND_WEIGHT

    def test_weight_epsilon_is_positive(self, fresh_config):
        """WEIGHT_EPSILON must be positive."""
        blend_config = fresh_config.blend
        assert blend_config.WEIGHT_EPSILON > 0.0

    def test_weight_epsilon_is_small(self, fresh_config):
        """WEIGHT_EPSILON should be a small value."""
        blend_config = fresh_config.blend
        assert blend_config.WEIGHT_EPSILON < 0.1

    def test_default_blend_time_is_positive(self, fresh_config):
        """DEFAULT_BLEND_TIME must be positive."""
        blend_config = fresh_config.blend
        assert blend_config.DEFAULT_BLEND_TIME > 0.0

    def test_normalize_weights_default(self, fresh_config):
        """NORMALIZE_WEIGHTS should default to True."""
        blend_config = fresh_config.blend
        assert blend_config.NORMALIZE_WEIGHTS is True


# =============================================================================
# Environment Variable Override Tests
# =============================================================================


class TestEnvironmentVariableOverrides:
    """Test that TRINITY_ANIM_* environment variables override config values."""

    def test_max_eval_depth_env_override(self):
        """TRINITY_ANIM_MAX_EVAL_DEPTH should override MAX_EVALUATION_DEPTH."""
        os.environ["TRINITY_ANIM_MAX_EVAL_DEPTH"] = "50"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.MAX_EVALUATION_DEPTH == 50

    def test_max_eval_depth_env_invalid_value(self):
        """Invalid TRINITY_ANIM_MAX_EVAL_DEPTH should fall back to default."""
        os.environ["TRINITY_ANIM_MAX_EVAL_DEPTH"] = "not_a_number"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.MAX_EVALUATION_DEPTH == 100

    def test_cycle_detection_env_override_enabled(self):
        """TRINITY_ANIM_CYCLE_DETECTION=1 should enable cycle detection."""
        os.environ["TRINITY_ANIM_CYCLE_DETECTION"] = "1"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.CYCLE_DETECTION_ENABLED is True

    def test_cycle_detection_env_override_disabled(self):
        """TRINITY_ANIM_CYCLE_DETECTION=0 should disable cycle detection."""
        os.environ["TRINITY_ANIM_CYCLE_DETECTION"] = "0"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.CYCLE_DETECTION_ENABLED is False

    def test_cycle_detection_env_override_true_string(self):
        """TRINITY_ANIM_CYCLE_DETECTION=true should enable cycle detection."""
        os.environ["TRINITY_ANIM_CYCLE_DETECTION"] = "true"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.CYCLE_DETECTION_ENABLED is True

    def test_cycle_detection_env_override_yes_string(self):
        """TRINITY_ANIM_CYCLE_DETECTION=yes should enable cycle detection."""
        os.environ["TRINITY_ANIM_CYCLE_DETECTION"] = "yes"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.CYCLE_DETECTION_ENABLED is True

    def test_cycle_detection_env_override_on_string(self):
        """TRINITY_ANIM_CYCLE_DETECTION=on should enable cycle detection."""
        os.environ["TRINITY_ANIM_CYCLE_DETECTION"] = "on"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.CYCLE_DETECTION_ENABLED is True

    def test_default_blend_time_env_override(self):
        """TRINITY_ANIM_DEFAULT_BLEND_TIME should override DEFAULT_BLEND_TIME."""
        os.environ["TRINITY_ANIM_DEFAULT_BLEND_TIME"] = "0.5"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.blend.DEFAULT_BLEND_TIME == 0.5

    def test_default_blend_time_env_invalid_value(self):
        """Invalid TRINITY_ANIM_DEFAULT_BLEND_TIME should fall back to default."""
        os.environ["TRINITY_ANIM_DEFAULT_BLEND_TIME"] = "invalid"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.blend.DEFAULT_BLEND_TIME == 0.25

    def test_default_time_scale_env_override(self):
        """TRINITY_ANIM_DEFAULT_TIME_SCALE should override DEFAULT_TIME_SCALE."""
        os.environ["TRINITY_ANIM_DEFAULT_TIME_SCALE"] = "2.0"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.DEFAULT_TIME_SCALE == 2.0

    def test_default_time_scale_env_invalid_value(self):
        """Invalid TRINITY_ANIM_DEFAULT_TIME_SCALE should fall back to default."""
        os.environ["TRINITY_ANIM_DEFAULT_TIME_SCALE"] = "bad"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.graph.DEFAULT_TIME_SCALE == 1.0

    def test_slerp_threshold_env_override(self):
        """TRINITY_ANIM_SLERP_THRESHOLD should override SLERP_DOT_THRESHOLD."""
        os.environ["TRINITY_ANIM_SLERP_THRESHOLD"] = "0.999"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.quaternion.SLERP_DOT_THRESHOLD == 0.999

    def test_slerp_threshold_env_invalid_value(self):
        """Invalid TRINITY_ANIM_SLERP_THRESHOLD should fall back to default."""
        os.environ["TRINITY_ANIM_SLERP_THRESHOLD"] = "notfloat"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()

        assert config.quaternion.SLERP_DOT_THRESHOLD == 0.9995


# =============================================================================
# reset_config() Tests
# =============================================================================


class TestResetConfigFunction:
    """Test that reset_config() restores defaults properly."""

    def test_reset_config_restores_default_after_env_override(self):
        """reset_config() should restore defaults when env vars are cleared."""
        # Set env var and create config
        os.environ["TRINITY_ANIM_MAX_EVAL_DEPTH"] = "25"

        from engine.animation.graph.config import reset_config, get_config
        reset_config()
        config = get_config()
        assert config.graph.MAX_EVALUATION_DEPTH == 25

        # Clear env var and reset
        del os.environ["TRINITY_ANIM_MAX_EVAL_DEPTH"]
        reset_config()
        config = get_config()

        assert config.graph.MAX_EVALUATION_DEPTH == 100

    def test_reset_config_creates_new_instance(self):
        """reset_config() should create a new config instance."""
        from engine.animation.graph.config import (
            reset_config,
            get_config,
            DEFAULT_CONFIG,
        )

        reset_config()
        config1 = get_config()
        original_id = id(config1)

        reset_config()
        config2 = get_config()

        assert id(config2) != original_id

    def test_reset_config_re_reads_env_vars(self):
        """reset_config() should re-read environment variables."""
        from engine.animation.graph.config import reset_config, get_config

        # Start clean
        reset_config()
        config = get_config()
        assert config.blend.DEFAULT_BLEND_TIME == 0.25

        # Set env var and reset
        os.environ["TRINITY_ANIM_DEFAULT_BLEND_TIME"] = "1.5"
        reset_config()
        config = get_config()

        assert config.blend.DEFAULT_BLEND_TIME == 1.5

    def test_reset_config_affects_global_default_config(self):
        """reset_config() should update the global DEFAULT_CONFIG."""
        from engine.animation.graph import config as config_module

        os.environ["TRINITY_ANIM_DEFAULT_TIME_SCALE"] = "3.0"
        config_module.reset_config()

        assert config_module.DEFAULT_CONFIG.graph.DEFAULT_TIME_SCALE == 3.0


# =============================================================================
# Documentation Tests
# =============================================================================


class TestDocumentation:
    """Verify docstrings exist for all configuration classes."""

    def test_quaternion_config_has_docstring(self):
        """QuaternionConfig should have a docstring."""
        from engine.animation.graph.config import QuaternionConfig

        assert QuaternionConfig.__doc__ is not None
        assert len(QuaternionConfig.__doc__) > 50

    def test_quaternion_config_docstring_mentions_slerp(self):
        """QuaternionConfig docstring should mention SLERP."""
        from engine.animation.graph.config import QuaternionConfig

        assert "SLERP" in QuaternionConfig.__doc__ or "slerp" in QuaternionConfig.__doc__

    def test_graph_config_has_docstring(self):
        """GraphConfig should have a docstring."""
        from engine.animation.graph.config import GraphConfig

        assert GraphConfig.__doc__ is not None
        assert len(GraphConfig.__doc__) > 50

    def test_graph_config_docstring_mentions_evaluation(self):
        """GraphConfig docstring should mention evaluation depth."""
        from engine.animation.graph.config import GraphConfig

        doc = GraphConfig.__doc__.lower()
        assert "evaluation" in doc or "depth" in doc

    def test_blend_config_has_docstring(self):
        """BlendConfig should have a docstring."""
        from engine.animation.graph.config import BlendConfig

        assert BlendConfig.__doc__ is not None
        assert len(BlendConfig.__doc__) > 50

    def test_blend_config_docstring_mentions_weight(self):
        """BlendConfig docstring should mention weight."""
        from engine.animation.graph.config import BlendConfig

        assert "weight" in BlendConfig.__doc__.lower()

    def test_transition_config_has_docstring(self):
        """TransitionConfig should have a docstring."""
        from engine.animation.graph.config import TransitionConfig

        assert TransitionConfig.__doc__ is not None
        assert len(TransitionConfig.__doc__) > 20

    def test_blend_tree_config_has_docstring(self):
        """BlendTreeConfig should have a docstring."""
        from engine.animation.graph.config import BlendTreeConfig

        assert BlendTreeConfig.__doc__ is not None
        assert len(BlendTreeConfig.__doc__) > 20

    def test_sync_config_has_docstring(self):
        """SyncConfig should have a docstring."""
        from engine.animation.graph.config import SyncConfig

        assert SyncConfig.__doc__ is not None
        assert len(SyncConfig.__doc__) > 20

    def test_layer_config_has_docstring(self):
        """LayerConfig should have a docstring."""
        from engine.animation.graph.config import LayerConfig

        assert LayerConfig.__doc__ is not None
        assert len(LayerConfig.__doc__) > 20

    def test_animation_graph_config_has_docstring(self):
        """AnimationGraphConfig should have a docstring."""
        from engine.animation.graph.config import AnimationGraphConfig

        assert AnimationGraphConfig.__doc__ is not None

    def test_module_has_docstring(self):
        """The config module should have a module-level docstring."""
        from engine.animation.graph import config as config_module

        assert config_module.__doc__ is not None
        assert len(config_module.__doc__) > 100

    def test_module_docstring_mentions_environment_variables(self):
        """Module docstring should document environment variable overrides."""
        from engine.animation.graph import config as config_module

        doc = config_module.__doc__
        assert "TRINITY_ANIM" in doc

    def test_get_config_has_docstring(self):
        """get_config() should have a docstring."""
        from engine.animation.graph.config import get_config

        assert get_config.__doc__ is not None

    def test_reset_config_has_docstring(self):
        """reset_config() should have a docstring."""
        from engine.animation.graph.config import reset_config

        assert reset_config.__doc__ is not None


# =============================================================================
# Dataclass Structure Tests
# =============================================================================


class TestDataclassStructure:
    """Test that config classes are proper dataclasses."""

    def test_quaternion_config_is_dataclass(self):
        """QuaternionConfig should be a dataclass."""
        from engine.animation.graph.config import QuaternionConfig

        assert is_dataclass(QuaternionConfig)

    def test_graph_config_is_dataclass(self):
        """GraphConfig should be a dataclass."""
        from engine.animation.graph.config import GraphConfig

        assert is_dataclass(GraphConfig)

    def test_blend_config_is_dataclass(self):
        """BlendConfig should be a dataclass."""
        from engine.animation.graph.config import BlendConfig

        assert is_dataclass(BlendConfig)

    def test_animation_graph_config_is_dataclass(self):
        """AnimationGraphConfig should be a dataclass."""
        from engine.animation.graph.config import AnimationGraphConfig

        assert is_dataclass(AnimationGraphConfig)

    def test_quaternion_config_has_expected_fields(self):
        """QuaternionConfig should have the expected fields."""
        from engine.animation.graph.config import QuaternionConfig

        field_names = {f.name for f in fields(QuaternionConfig)}
        expected = {
            "SLERP_DOT_THRESHOLD",
            "SLERP_MIN_SIN_THETA",
            "NORMALIZATION_EPSILON",
            "EPSILON",
        }
        assert expected.issubset(field_names)

    def test_graph_config_has_expected_fields(self):
        """GraphConfig should have the expected fields."""
        from engine.animation.graph.config import GraphConfig

        field_names = {f.name for f in fields(GraphConfig)}
        expected = {
            "MAX_EVALUATION_DEPTH",
            "CYCLE_DETECTION_ENABLED",
            "DEFAULT_TIME_SCALE",
        }
        assert expected.issubset(field_names)

    def test_blend_config_has_expected_fields(self):
        """BlendConfig should have the expected fields."""
        from engine.animation.graph.config import BlendConfig

        field_names = {f.name for f in fields(BlendConfig)}
        expected = {
            "DEFAULT_BLEND_TIME",
            "MIN_BLEND_WEIGHT",
            "MAX_BLEND_WEIGHT",
            "WEIGHT_EPSILON",
            "NORMALIZE_WEIGHTS",
        }
        assert expected.issubset(field_names)

    def test_animation_graph_config_has_all_sub_configs(self):
        """AnimationGraphConfig should aggregate all sub-configs."""
        from engine.animation.graph.config import AnimationGraphConfig

        field_names = {f.name for f in fields(AnimationGraphConfig)}
        expected = {
            "transition",
            "blend_tree",
            "sync",
            "layer",
            "quaternion",
            "graph",
            "blend",
        }
        assert expected == field_names


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test that __all__ exports the expected symbols."""

    def test_all_exports_config_classes(self):
        """__all__ should export all config classes."""
        from engine.animation.graph.config import __all__

        expected_classes = [
            "TransitionConfig",
            "BlendTreeConfig",
            "SyncConfig",
            "LayerConfig",
            "QuaternionConfig",
            "GraphConfig",
            "BlendConfig",
            "AnimationGraphConfig",
        ]
        for cls in expected_classes:
            assert cls in __all__, f"{cls} not in __all__"

    def test_all_exports_default_config(self):
        """__all__ should export DEFAULT_CONFIG."""
        from engine.animation.graph.config import __all__

        assert "DEFAULT_CONFIG" in __all__

    def test_all_exports_get_config(self):
        """__all__ should export get_config function."""
        from engine.animation.graph.config import __all__

        assert "get_config" in __all__

    def test_all_exports_reset_config(self):
        """__all__ should export reset_config function."""
        from engine.animation.graph.config import __all__

        assert "reset_config" in __all__


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test internal helper functions for environment variable parsing."""

    def test_env_float_returns_default_when_unset(self):
        """_env_float should return default when env var is not set."""
        from engine.animation.graph.config import _env_float

        # Ensure var is not set
        if "TEST_FLOAT_VAR" in os.environ:
            del os.environ["TEST_FLOAT_VAR"]

        result = _env_float("TEST_FLOAT_VAR", 42.5)
        assert result == 42.5

    def test_env_float_parses_valid_float(self):
        """_env_float should parse valid float strings."""
        from engine.animation.graph.config import _env_float

        os.environ["TEST_FLOAT_VAR"] = "3.14159"
        try:
            result = _env_float("TEST_FLOAT_VAR", 0.0)
            assert result == 3.14159
        finally:
            del os.environ["TEST_FLOAT_VAR"]

    def test_env_float_returns_default_on_invalid(self):
        """_env_float should return default for invalid float strings."""
        from engine.animation.graph.config import _env_float

        os.environ["TEST_FLOAT_VAR"] = "not_a_float"
        try:
            result = _env_float("TEST_FLOAT_VAR", 99.9)
            assert result == 99.9
        finally:
            del os.environ["TEST_FLOAT_VAR"]

    def test_env_int_returns_default_when_unset(self):
        """_env_int should return default when env var is not set."""
        from engine.animation.graph.config import _env_int

        if "TEST_INT_VAR" in os.environ:
            del os.environ["TEST_INT_VAR"]

        result = _env_int("TEST_INT_VAR", 42)
        assert result == 42

    def test_env_int_parses_valid_int(self):
        """_env_int should parse valid integer strings."""
        from engine.animation.graph.config import _env_int

        os.environ["TEST_INT_VAR"] = "123"
        try:
            result = _env_int("TEST_INT_VAR", 0)
            assert result == 123
        finally:
            del os.environ["TEST_INT_VAR"]

    def test_env_int_returns_default_on_invalid(self):
        """_env_int should return default for invalid integer strings."""
        from engine.animation.graph.config import _env_int

        os.environ["TEST_INT_VAR"] = "12.5"  # Float not valid int
        try:
            result = _env_int("TEST_INT_VAR", 99)
            assert result == 99
        finally:
            del os.environ["TEST_INT_VAR"]

    def test_env_bool_returns_default_when_unset(self):
        """_env_bool should return default when env var is not set."""
        from engine.animation.graph.config import _env_bool

        if "TEST_BOOL_VAR" in os.environ:
            del os.environ["TEST_BOOL_VAR"]

        assert _env_bool("TEST_BOOL_VAR", True) is True
        assert _env_bool("TEST_BOOL_VAR", False) is False

    def test_env_bool_parses_true_values(self):
        """_env_bool should recognize various true values."""
        from engine.animation.graph.config import _env_bool

        true_values = ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]
        for val in true_values:
            os.environ["TEST_BOOL_VAR"] = val
            try:
                result = _env_bool("TEST_BOOL_VAR", False)
                assert result is True, f"Failed for value: {val}"
            finally:
                del os.environ["TEST_BOOL_VAR"]

    def test_env_bool_parses_false_values(self):
        """_env_bool should treat other values as false."""
        from engine.animation.graph.config import _env_bool

        false_values = ["0", "false", "False", "no", "off", "anything_else"]
        for val in false_values:
            os.environ["TEST_BOOL_VAR"] = val
            try:
                result = _env_bool("TEST_BOOL_VAR", True)
                assert result is False, f"Should be False for value: {val}"
            finally:
                del os.environ["TEST_BOOL_VAR"]


# =============================================================================
# Additional Config Class Tests
# =============================================================================


class TestTransitionConfigWhitebox:
    """Whitebox tests for TransitionConfig class."""

    def test_default_transition_duration_value(self, fresh_config):
        """DEFAULT_TRANSITION_DURATION should be 0.2 seconds."""
        trans_config = fresh_config.transition
        assert trans_config.DEFAULT_TRANSITION_DURATION == 0.2

    def test_forced_transition_duration_value(self, fresh_config):
        """FORCED_TRANSITION_DURATION should be 0.2 seconds."""
        trans_config = fresh_config.transition
        assert trans_config.FORCED_TRANSITION_DURATION == 0.2

    def test_any_state_priority_value(self, fresh_config):
        """ANY_STATE_PRIORITY should be 100."""
        trans_config = fresh_config.transition
        assert trans_config.ANY_STATE_PRIORITY == 100

    def test_any_state_priority_is_int(self, fresh_config):
        """ANY_STATE_PRIORITY should be an integer."""
        trans_config = fresh_config.transition
        assert isinstance(trans_config.ANY_STATE_PRIORITY, int)


class TestBlendTreeConfigWhitebox:
    """Whitebox tests for BlendTreeConfig class."""

    def test_default_gradient_band_width_value(self, fresh_config):
        """DEFAULT_GRADIENT_BAND_WIDTH should be 0.1."""
        bt_config = fresh_config.blend_tree
        assert bt_config.DEFAULT_GRADIENT_BAND_WIDTH == 0.1

    def test_inverse_distance_power_value(self, fresh_config):
        """INVERSE_DISTANCE_POWER should be 2.0."""
        bt_config = fresh_config.blend_tree
        assert bt_config.INVERSE_DISTANCE_POWER == 2.0

    def test_distance_epsilon_value(self, fresh_config):
        """DISTANCE_EPSILON should be 1e-10."""
        bt_config = fresh_config.blend_tree
        assert bt_config.DISTANCE_EPSILON == 1e-10


class TestSyncConfigWhitebox:
    """Whitebox tests for SyncConfig class."""

    def test_sync_tolerance_value(self, fresh_config):
        """SYNC_TOLERANCE should be 0.01."""
        sync_config = fresh_config.sync
        assert sync_config.SYNC_TOLERANCE == 0.01

    def test_event_dedup_precision_value(self, fresh_config):
        """EVENT_DEDUP_PRECISION should be 2."""
        sync_config = fresh_config.sync
        assert sync_config.EVENT_DEDUP_PRECISION == 2

    def test_event_dedup_precision_is_int(self, fresh_config):
        """EVENT_DEDUP_PRECISION should be an integer."""
        sync_config = fresh_config.sync
        assert isinstance(sync_config.EVENT_DEDUP_PRECISION, int)


class TestLayerConfigWhitebox:
    """Whitebox tests for LayerConfig class."""

    def test_default_layer_weight_value(self, fresh_config):
        """DEFAULT_LAYER_WEIGHT should be 1.0."""
        layer_config = fresh_config.layer
        assert layer_config.DEFAULT_LAYER_WEIGHT == 1.0

    def test_default_layer_weight_is_float(self, fresh_config):
        """DEFAULT_LAYER_WEIGHT should be a float."""
        layer_config = fresh_config.layer
        assert isinstance(layer_config.DEFAULT_LAYER_WEIGHT, float)
