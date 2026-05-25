"""
WHITEBOX tests for engine/animation/graph/config.py.

WHITEBOX coverage plan:
  [TransitionConfig]
    Path A1:  default-initialised all fields
    Path A2:  all fields overridden with custom values
    Path A3:  individual positional/keyword override

  [BlendTreeConfig]
    Path B1:  default-initialised all fields
    Path B2:  all fields overridden with custom values
    Path B3:  DISTANCE_EPSILON boundary — fractionally above/below zero

  [SyncConfig]
    Path C1:  default-initialised all fields
    Path C2:  all fields overridden with custom values
    Path C3:  EVENT_DEDUP_PRECISION negative integer

  [LayerConfig]
    Path D1:  default-initialised
    Path D2:  DEFAULT_LAYER_WEIGHT boundary values (0.0, 0.5)

  [QuaternionConfig]
    Path E1:  default-initialised all fields
    Path E2:  all fields overridden with custom values
    Path E3:  SLERP_DOT_THRESHOLD boundary — exactly 1.0, exactly -1.0
    Path E4:  SLERP_MIN_SIN_THETA boundary — 0.0, negative value
    Path E5:  NORMALIZATION_EPSILON boundary — 0.0, negative value

  [GraphConfig]
    Path F1:  default-initialised all fields
    Path F2:  MAX_EVALUATION_DEPTH custom values (0, 1, negative)
    Path F3:  CYCLE_DETECTION_ENABLED toggle True/False

  [BlendConfig]
    Path G1:  default-initialised all fields
    Path G2:  WEIGHT_EPSILON custom values (0.0, near-zero)
    Path G3:  NORMALIZE_WEIGHTS toggle True/False

  [AnimationGraphConfig composite]
    Path H1:  default-initialised — all seven sub-configs created via default_factory
    Path H2:  single sub-config overridden
    Path H3:  all sub-configs overridden
    Path H4:  nested field access — config.transition.DEFAULT_TRANSITION_DURATION

  [DEFAULT_CONFIG singleton and get_config()]
    Path I1:  DEFAULT_CONFIG is an AnimationGraphConfig
    Path I2:  get_config() returns DEFAULT_CONFIG (same object identity)
    Path I3:  DEFAULT_CONFIG sub-configs are populated (not None)
    Path I4:  DEFAULT_CONFIG field values match the source defaults

  [Edge cases — numeric and type]
    Path J1:  negative values for all float fields accepted
    Path J2:  zero values for all float fields accepted
    Path J3:  large int for MAX_EVALUATION_DEPTH
    Path J4:  boolean fields accept both True and False
    Path J5:  configs are independent — mutating one does not affect another
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field, fields as dataclass_fields

from engine.animation.graph.config import (
    TransitionConfig,
    BlendTreeConfig,
    SyncConfig,
    LayerConfig,
    QuaternionConfig,
    GraphConfig,
    BlendConfig,
    AnimationGraphConfig,
    DEFAULT_CONFIG,
    get_config,
)


# ===========================================================================
# Path A — TransitionConfig
# ===========================================================================

class TestTransitionConfig:
    """WHITEBOX: TransitionConfig — defaults, overrides, boundaries."""

    def test_A1_defaults(self):
        """Default values match source constants."""
        cfg = TransitionConfig()
        assert cfg.DEFAULT_TRANSITION_DURATION == 0.2
        assert cfg.FORCED_TRANSITION_DURATION == 0.2
        assert cfg.ANY_STATE_PRIORITY == 100

    def test_A2_all_overridden(self):
        """All fields overridden via keyword args."""
        cfg = TransitionConfig(
            DEFAULT_TRANSITION_DURATION=0.5,
            FORCED_TRANSITION_DURATION=0.1,
            ANY_STATE_PRIORITY=200,
        )
        assert cfg.DEFAULT_TRANSITION_DURATION == 0.5
        assert cfg.FORCED_TRANSITION_DURATION == 0.1
        assert cfg.ANY_STATE_PRIORITY == 200

    def test_A3_single_field_override(self):
        """Single field override leaves others at default."""
        cfg = TransitionConfig(DEFAULT_TRANSITION_DURATION=0.8)
        assert cfg.DEFAULT_TRANSITION_DURATION == 0.8
        assert cfg.FORCED_TRANSITION_DURATION == 0.2  # unchanged default
        assert cfg.ANY_STATE_PRIORITY == 100  # unchanged default


# ===========================================================================
# Path B — BlendTreeConfig
# ===========================================================================

class TestBlendTreeConfig:
    """WHITEBOX: BlendTreeConfig — defaults, overrides, epsilon boundary."""

    def test_B1_defaults(self):
        """Default values match source constants."""
        cfg = BlendTreeConfig()
        assert cfg.DEFAULT_GRADIENT_BAND_WIDTH == 0.1
        assert cfg.INVERSE_DISTANCE_POWER == 2.0
        assert cfg.DISTANCE_EPSILON == 1e-10

    def test_B2_all_overridden(self):
        """All fields overridden via keyword args."""
        cfg = BlendTreeConfig(
            DEFAULT_GRADIENT_BAND_WIDTH=0.05,
            INVERSE_DISTANCE_POWER=3.0,
            DISTANCE_EPSILON=1e-12,
        )
        assert cfg.DEFAULT_GRADIENT_BAND_WIDTH == 0.05
        assert cfg.INVERSE_DISTANCE_POWER == 3.0
        assert cfg.DISTANCE_EPSILON == 1e-12

    def test_B3_epsilon_boundary_positive(self):
        """DISTANCE_EPSILON accepts a value just above zero (the intended regime)."""
        cfg = BlendTreeConfig(DISTANCE_EPSILON=1e-20)
        assert cfg.DISTANCE_EPSILON > 0.0


# ===========================================================================
# Path C — SyncConfig
# ===========================================================================

class TestSyncConfig:
    """WHITEBOX: SyncConfig — defaults, overrides, int boundary."""

    def test_C1_defaults(self):
        """Default values match source constants."""
        cfg = SyncConfig()
        assert cfg.SYNC_TOLERANCE == 0.01
        assert cfg.EVENT_DEDUP_PRECISION == 2

    def test_C2_all_overridden(self):
        """All fields overridden via keyword args."""
        cfg = SyncConfig(SYNC_TOLERANCE=0.02, EVENT_DEDUP_PRECISION=3)
        assert cfg.SYNC_TOLERANCE == 0.02
        assert cfg.EVENT_DEDUP_PRECISION == 3

    def test_C3_negative_precision(self):
        """EVENT_DEDUP_PRECISION accepts a negative integer (valid round(..., ndigits=-1))."""
        cfg = SyncConfig(EVENT_DEDUP_PRECISION=-1)
        assert cfg.EVENT_DEDUP_PRECISION == -1


# ===========================================================================
# Path D — LayerConfig
# ===========================================================================

class TestLayerConfig:
    """WHITEBOX: LayerConfig — defaults, weight boundaries."""

    def test_D1_defaults(self):
        """Default value matches source constant."""
        cfg = LayerConfig()
        assert cfg.DEFAULT_LAYER_WEIGHT == 1.0

    def test_D2_weight_boundary_values(self):
        """DEFAULT_LAYER_WEIGHT accepts boundary values 0.0 and 0.5."""
        cfg_zero = LayerConfig(DEFAULT_LAYER_WEIGHT=0.0)
        assert cfg_zero.DEFAULT_LAYER_WEIGHT == 0.0

        cfg_mid = LayerConfig(DEFAULT_LAYER_WEIGHT=0.5)
        assert cfg_mid.DEFAULT_LAYER_WEIGHT == 0.5


# ===========================================================================
# Path E — QuaternionConfig
# ===========================================================================

class TestQuaternionConfig:
    """WHITEBOX: QuaternionConfig — defaults, overrides, numeric boundaries."""

    def test_E1_defaults(self):
        """Default values match source constants."""
        cfg = QuaternionConfig()
        assert cfg.SLERP_DOT_THRESHOLD == 0.9995
        assert cfg.SLERP_MIN_SIN_THETA == 0.0001
        assert cfg.NORMALIZATION_EPSILON == 1e-6

    def test_E2_all_overridden(self):
        """All fields overridden via keyword args."""
        cfg = QuaternionConfig(
            SLERP_DOT_THRESHOLD=0.99,
            SLERP_MIN_SIN_THETA=0.001,
            NORMALIZATION_EPSILON=1e-8,
        )
        assert cfg.SLERP_DOT_THRESHOLD == 0.99
        assert cfg.SLERP_MIN_SIN_THETA == 0.001
        assert cfg.NORMALIZATION_EPSILON == 1e-8

    def test_E3_dot_threshold_extremes(self):
        """SLERP_DOT_THRESHOLD accepts extreme valid values (1.0 and -1.0)."""
        cfg_one = QuaternionConfig(SLERP_DOT_THRESHOLD=1.0)
        assert cfg_one.SLERP_DOT_THRESHOLD == 1.0

        cfg_neg = QuaternionConfig(SLERP_DOT_THRESHOLD=-1.0)
        assert cfg_neg.SLERP_DOT_THRESHOLD == -1.0

    def test_E4_min_sin_theta_boundaries(self):
        """SLERP_MIN_SIN_THETA accepts 0.0 and negative (physically odd but valid python)."""
        cfg_zero = QuaternionConfig(SLERP_MIN_SIN_THETA=0.0)
        assert cfg_zero.SLERP_MIN_SIN_THETA == 0.0

        cfg_neg = QuaternionConfig(SLERP_MIN_SIN_THETA=-0.001)
        assert cfg_neg.SLERP_MIN_SIN_THETA == -0.001

    def test_E5_normalization_epsilon_boundaries(self):
        """NORMALIZATION_EPSILON accepts 0.0 and negative (edge-of-domain values)."""
        cfg_zero = QuaternionConfig(NORMALIZATION_EPSILON=0.0)
        assert cfg_zero.NORMALIZATION_EPSILON == 0.0

        cfg_neg = QuaternionConfig(NORMALIZATION_EPSILON=-1e-6)
        assert cfg_neg.NORMALIZATION_EPSILON == -1e-6


# ===========================================================================
# Path F — GraphConfig
# ===========================================================================

class TestGraphConfig:
    """WHITEBOX: GraphConfig — defaults, depth boundaries, cycle toggle."""

    def test_F1_defaults(self):
        """Default values match source constants."""
        cfg = GraphConfig()
        assert cfg.MAX_EVALUATION_DEPTH == 100
        assert cfg.CYCLE_DETECTION_ENABLED is True

    def test_F2_depth_boundaries(self):
        """MAX_EVALUATION_DEPTH accepts boundary values."""
        cfg_zero = GraphConfig(MAX_EVALUATION_DEPTH=0)
        assert cfg_zero.MAX_EVALUATION_DEPTH == 0

        cfg_one = GraphConfig(MAX_EVALUATION_DEPTH=1)
        assert cfg_one.MAX_EVALUATION_DEPTH == 1

        cfg_neg = GraphConfig(MAX_EVALUATION_DEPTH=-1)
        assert cfg_neg.MAX_EVALUATION_DEPTH == -1

    def test_F3_cycle_detection_toggle(self):
        """CYCLE_DETECTION_ENABLED toggles True/False."""
        cfg_on = GraphConfig(CYCLE_DETECTION_ENABLED=True)
        assert cfg_on.CYCLE_DETECTION_ENABLED is True

        cfg_off = GraphConfig(CYCLE_DETECTION_ENABLED=False)
        assert cfg_off.CYCLE_DETECTION_ENABLED is False


# ===========================================================================
# Path G — BlendConfig
# ===========================================================================

class TestBlendConfig:
    """WHITEBOX: BlendConfig — defaults, epsilon boundaries, normalize toggle."""

    def test_G1_defaults(self):
        """Default values match source constants."""
        cfg = BlendConfig()
        assert cfg.WEIGHT_EPSILON == 0.001
        assert cfg.NORMALIZE_WEIGHTS is True

    def test_G2_weight_epsilon_boundaries(self):
        """WEIGHT_EPSILON accepts 0.0 and other near-zero values."""
        cfg_zero = BlendConfig(WEIGHT_EPSILON=0.0)
        assert cfg_zero.WEIGHT_EPSILON == 0.0

        cfg_small = BlendConfig(WEIGHT_EPSILON=1e-10)
        assert cfg_small.WEIGHT_EPSILON == 1e-10

    def test_G3_normalize_weights_toggle(self):
        """NORMALIZE_WEIGHTS toggles True/False."""
        cfg_on = BlendConfig(NORMALIZE_WEIGHTS=True)
        assert cfg_on.NORMALIZE_WEIGHTS is True

        cfg_off = BlendConfig(NORMALIZE_WEIGHTS=False)
        assert cfg_off.NORMALIZE_WEIGHTS is False


# ===========================================================================
# Path H — AnimationGraphConfig composite
# ===========================================================================

class TestAnimationGraphConfig:
    """WHITEBOX: AnimationGraphConfig — composite construction, overrides."""

    def test_H1_defaults_all_subconfigs_created(self):
        """Default AnimationGraphConfig creates all seven sub-configs via default_factory."""
        cfg = AnimationGraphConfig()
        assert isinstance(cfg.transition, TransitionConfig)
        assert isinstance(cfg.blend_tree, BlendTreeConfig)
        assert isinstance(cfg.sync, SyncConfig)
        assert isinstance(cfg.layer, LayerConfig)
        assert isinstance(cfg.quaternion, QuaternionConfig)
        assert isinstance(cfg.graph, GraphConfig)
        assert isinstance(cfg.blend, BlendConfig)

    def test_H2_single_subconfig_override(self):
        """Single sub-config overridden; others remain default."""
        custom = TransitionConfig(DEFAULT_TRANSITION_DURATION=0.5)
        cfg = AnimationGraphConfig(transition=custom)
        assert cfg.transition.DEFAULT_TRANSITION_DURATION == 0.5
        assert cfg.transition is custom
        # Others are independent default instances
        assert isinstance(cfg.blend_tree, BlendTreeConfig)
        assert cfg.blend_tree is not None

    def test_H3_all_subconfigs_overridden(self):
        """All seven sub-configs overridden with custom values."""
        t = TransitionConfig(DEFAULT_TRANSITION_DURATION=0.5, FORCED_TRANSITION_DURATION=0.1)
        b = BlendTreeConfig(DEFAULT_GRADIENT_BAND_WIDTH=0.2)
        s = SyncConfig(SYNC_TOLERANCE=0.05)
        l = LayerConfig(DEFAULT_LAYER_WEIGHT=0.8)
        q = QuaternionConfig(SLERP_DOT_THRESHOLD=0.999)
        g = GraphConfig(MAX_EVALUATION_DEPTH=50)
        bl = BlendConfig(WEIGHT_EPSILON=0.0005)

        cfg = AnimationGraphConfig(
            transition=t,
            blend_tree=b,
            sync=s,
            layer=l,
            quaternion=q,
            graph=g,
            blend=bl,
        )
        assert cfg.transition.DEFAULT_TRANSITION_DURATION == 0.5
        assert cfg.blend_tree.DEFAULT_GRADIENT_BAND_WIDTH == 0.2
        assert cfg.sync.SYNC_TOLERANCE == 0.05
        assert cfg.layer.DEFAULT_LAYER_WEIGHT == 0.8
        assert cfg.quaternion.SLERP_DOT_THRESHOLD == 0.999
        assert cfg.graph.MAX_EVALUATION_DEPTH == 50
        assert cfg.blend.WEIGHT_EPSILON == 0.0005

    def test_H4_nested_field_access(self):
        """Nested field path through composite config works."""
        cfg = AnimationGraphConfig()
        assert cfg.transition.DEFAULT_TRANSITION_DURATION == 0.2
        assert cfg.quaternion.SLERP_DOT_THRESHOLD == 0.9995
        assert cfg.blend.NORMALIZE_WEIGHTS is True


# ===========================================================================
# Path I — DEFAULT_CONFIG singleton and get_config()
# ===========================================================================

class TestDefaultConfig:
    """WHITEBOX: DEFAULT_CONFIG global singleton and get_config() accessor."""

    def test_I1_default_config_is_animation_graph_config(self):
        """DEFAULT_CONFIG is an instance of AnimationGraphConfig."""
        assert isinstance(DEFAULT_CONFIG, AnimationGraphConfig)

    def test_I2_get_config_returns_default_config(self):
        """get_config() returns the same object as DEFAULT_CONFIG."""
        result = get_config()
        assert result is DEFAULT_CONFIG

    def test_I3_default_config_subconfigs_populated(self):
        """DEFAULT_CONFIG sub-configs are not None."""
        assert DEFAULT_CONFIG.transition is not None
        assert DEFAULT_CONFIG.blend_tree is not None
        assert DEFAULT_CONFIG.sync is not None
        assert DEFAULT_CONFIG.layer is not None
        assert DEFAULT_CONFIG.quaternion is not None
        assert DEFAULT_CONFIG.graph is not None
        assert DEFAULT_CONFIG.blend is not None

    def test_I4_default_config_field_values(self):
        """DEFAULT_CONFIG field values match the source defaults."""
        cfg = DEFAULT_CONFIG
        assert cfg.transition.DEFAULT_TRANSITION_DURATION == 0.2
        assert cfg.transition.FORCED_TRANSITION_DURATION == 0.2
        assert cfg.transition.ANY_STATE_PRIORITY == 100
        assert cfg.blend_tree.DEFAULT_GRADIENT_BAND_WIDTH == 0.1
        assert cfg.blend_tree.INVERSE_DISTANCE_POWER == 2.0
        assert cfg.blend_tree.DISTANCE_EPSILON == 1e-10
        assert cfg.sync.SYNC_TOLERANCE == 0.01
        assert cfg.sync.EVENT_DEDUP_PRECISION == 2
        assert cfg.layer.DEFAULT_LAYER_WEIGHT == 1.0
        assert cfg.quaternion.SLERP_DOT_THRESHOLD == 0.9995
        assert cfg.quaternion.SLERP_MIN_SIN_THETA == 0.0001
        assert cfg.quaternion.NORMALIZATION_EPSILON == 1e-6
        assert cfg.graph.MAX_EVALUATION_DEPTH == 100
        assert cfg.graph.CYCLE_DETECTION_ENABLED is True
        assert cfg.blend.WEIGHT_EPSILON == 0.001
        assert cfg.blend.NORMALIZE_WEIGHTS is True


# ===========================================================================
# Path J — Edge cases: numeric boundaries, type, config independence
# ===========================================================================

class TestConfigEdgeCases:
    """WHITEBOX: config edge cases — numeric extremes, independence."""

    # -- J1: negative float values --

    def test_J1_negative_transition_duration(self):
        """Negative DEFAULT_TRANSITION_DURATION accepted (Python dataclass does not clamp)."""
        cfg = TransitionConfig(DEFAULT_TRANSITION_DURATION=-0.5)
        assert cfg.DEFAULT_TRANSITION_DURATION == -0.5

    def test_J1_negative_gradient_band(self):
        """Negative DEFAULT_GRADIENT_BAND_WIDTH accepted."""
        cfg = BlendTreeConfig(DEFAULT_GRADIENT_BAND_WIDTH=-0.1)
        assert cfg.DEFAULT_GRADIENT_BAND_WIDTH == -0.1

    def test_J1_negative_layer_weight(self):
        """Negative DEFAULT_LAYER_WEIGHT accepted."""
        cfg = LayerConfig(DEFAULT_LAYER_WEIGHT=-1.0)
        assert cfg.DEFAULT_LAYER_WEIGHT == -1.0

    def test_J1_negative_slerp_dot(self):
        """Negative SLERP_DOT_THRESHOLD accepted (allows -1.0)."""
        cfg = QuaternionConfig(SLERP_DOT_THRESHOLD=-1.5)
        assert cfg.SLERP_DOT_THRESHOLD == -1.5

    # -- J2: zero float values --

    def test_J2_zero_sync_tolerance(self):
        """Zero SYNC_TOLERANCE accepted."""
        cfg = SyncConfig(SYNC_TOLERANCE=0.0)
        assert cfg.SYNC_TOLERANCE == 0.0

    def test_J2_zero_weight_epsilon(self):
        """Zero WEIGHT_EPSILON accepted."""
        cfg = BlendConfig(WEIGHT_EPSILON=0.0)
        assert cfg.WEIGHT_EPSILON == 0.0

    # -- J3: large int values --

    def test_J3_large_max_evaluation_depth(self):
        """MAX_EVALUATION_DEPTH accepts large values (e.g. sys.maxsize)."""
        import sys
        cfg = GraphConfig(MAX_EVALUATION_DEPTH=sys.maxsize)
        assert cfg.MAX_EVALUATION_DEPTH == sys.maxsize

    def test_J3_large_any_state_priority(self):
        """ANY_STATE_PRIORITY accepts very large int."""
        cfg = TransitionConfig(ANY_STATE_PRIORITY=2**31)
        assert cfg.ANY_STATE_PRIORITY == 2**31

    def test_J3_zero_event_dedup_precision(self):
        """EVENT_DEDUP_PRECISION = 0 is a valid round(..., ndigits=0) argument."""
        cfg = SyncConfig(EVENT_DEDUP_PRECISION=0)
        assert cfg.EVENT_DEDUP_PRECISION == 0

    # -- J4: boolean toggles --

    def test_J4_cycle_detection_accepts_bool(self):
        """CYCLE_DETECTION_ENABLED is a proper bool, not an int."""
        cfg_true = GraphConfig(CYCLE_DETECTION_ENABLED=True)
        cfg_false = GraphConfig(CYCLE_DETECTION_ENABLED=False)
        assert cfg_true.CYCLE_DETECTION_ENABLED is True
        assert cfg_false.CYCLE_DETECTION_ENABLED is False
        assert isinstance(cfg_true.CYCLE_DETECTION_ENABLED, bool)

    def test_J4_normalize_weights_accepts_bool(self):
        """NORMALIZE_WEIGHTS is a proper bool, not an int."""
        cfg_true = BlendConfig(NORMALIZE_WEIGHTS=True)
        cfg_false = BlendConfig(NORMALIZE_WEIGHTS=False)
        assert cfg_true.NORMALIZE_WEIGHTS is True
        assert cfg_false.NORMALIZE_WEIGHTS is False
        assert isinstance(cfg_true.NORMALIZE_WEIGHTS, bool)

    # -- J5: config independence --

    def test_J5_default_config_immutable_by_custom(self):
        """Custom AnimationGraphConfig does not mutate DEFAULT_CONFIG."""
        custom = AnimationGraphConfig(
            transition=TransitionConfig(DEFAULT_TRANSITION_DURATION=99.0)
        )
        assert custom.transition.DEFAULT_TRANSITION_DURATION == 99.0
        assert DEFAULT_CONFIG.transition.DEFAULT_TRANSITION_DURATION == 0.2

    def test_J5_custom_configs_are_independent(self):
        """Two custom configs are independent objects (identity comparison)."""
        a = AnimationGraphConfig()
        b = AnimationGraphConfig()
        # Each subconfig is a distinct object (default_factory creates new instances)
        assert a.transition is not b.transition
        # Same structural equality despite different object identity
        assert a.transition == b.transition

    def test_J5_subconfigs_are_independent_defaults(self):
        """Each AnimationGraphConfig default_factory creates distinct sub-config objects."""
        a = AnimationGraphConfig()
        b = AnimationGraphConfig()
        assert a.transition is not b.transition
        assert a.blend_tree is not b.blend_tree
        assert a.sync is not b.sync
        assert a.layer is not b.layer
        assert a.quaternion is not b.quaternion
        assert a.graph is not b.graph
        assert a.blend is not b.blend


# ===========================================================================
# Fixture / import sanity
# ===========================================================================

class TestModuleExports:
    """Verify __all__ exports match the module's public API."""

    def test_all_exports_present(self):
        """All expected names are exported in __all__."""
        from engine.animation.graph import config as mod
        expected = {
            "TransitionConfig",
            "BlendTreeConfig",
            "SyncConfig",
            "LayerConfig",
            "QuaternionConfig",
            "GraphConfig",
            "BlendConfig",
            "AnimationGraphConfig",
            "DEFAULT_CONFIG",
            "get_config",
        }
        assert set(mod.__all__) == expected
