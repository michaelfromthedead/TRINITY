"""
Tests for Trinity Pattern - Tier 41: BUILD_DEPLOY Decorators
"""

import pytest

from trinity.decorators.build_deploy import (
    asset_bundle,
    build_only,
    feature_flag,
    strip_in_release,
)
from trinity.decorators.ops import decompose
from trinity.decorators.registry import registry


class TestBuildOnly:
    """Test @build_only decorator."""

    def test_basic_application(self):
        """Test basic @build_only decorator application."""

        @build_only(configurations={"debug"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_build_only")
        assert TestClass._build_only is True
        assert hasattr(TestClass, "_build_configurations")
        assert TestClass._build_configurations == frozenset({"debug"})

    def test_multiple_configurations(self):
        """Test @build_only with multiple configurations."""

        @build_only(configurations={"debug", "development", "test"})
        class TestClass:
            pass

        assert TestClass._build_configurations == frozenset(
            {"debug", "development", "test"}
        )

    def test_default_configurations(self):
        """Test @build_only with default configurations."""

        @build_only()
        class TestClass:
            pass

        assert TestClass._build_configurations == frozenset({"debug"})

    def test_empty_configurations_raises_error(self):
        """Test empty configurations raises ValueError."""
        with pytest.raises(ValueError, match="configurations must be a non-empty set"):

            @build_only(configurations=set())
            class TestClass:
                pass

    def test_invalid_configurations_type(self):
        """Test invalid configurations type raises TypeError."""
        with pytest.raises(TypeError, match="configurations must be a set"):

            @build_only(configurations=["debug", "release"])
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @build_only(configurations={"debug", "test"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("build_only") is True
        assert TestClass._tags.get("build_configurations") == frozenset(
            {"debug", "test"}
        )
        assert hasattr(TestClass, "_registries")
        assert "build_deploy" in TestClass._registries

    def test_decorator_tracking(self):
        """Test decorator application tracking."""

        @build_only(configurations={"debug"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "build_only" in TestClass._applied_decorators

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(build_only)
        assert len(steps) == 3
        assert any(s.op.value == "tag" for s in steps)
        assert any(s.op.value == "register" for s in steps)

    def test_on_function(self):
        """Test @build_only on a function."""

        @build_only(configurations={"debug", "profile"})
        def test_func():
            pass

        assert test_func._build_only is True
        assert test_func._build_configurations == frozenset({"debug", "profile"})


class TestStripInRelease:
    """Test @strip_in_release decorator."""

    def test_basic_application(self):
        """Test basic @strip_in_release decorator application."""

        @strip_in_release
        class TestClass:
            pass

        assert hasattr(TestClass, "_strip_in_release")
        assert TestClass._strip_in_release is True

    def test_with_parentheses(self):
        """Test @strip_in_release with parentheses."""

        @strip_in_release()
        class TestClass:
            pass

        assert TestClass._strip_in_release is True

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @strip_in_release
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("strip_in_release") is True
        assert hasattr(TestClass, "_registries")
        assert "build_deploy" in TestClass._registries

    def test_decorator_tracking(self):
        """Test decorator application tracking."""

        @strip_in_release()
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "strip_in_release" in TestClass._applied_decorators

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(strip_in_release)
        assert len(steps) == 2
        assert any(s.op.value == "tag" for s in steps)
        assert any(s.op.value == "register" for s in steps)

    def test_on_function(self):
        """Test @strip_in_release on a function."""

        @strip_in_release
        def test_func():
            pass

        assert test_func._strip_in_release is True


class TestAssetBundle:
    """Test @asset_bundle decorator."""

    def test_basic_application(self):
        """Test basic @asset_bundle decorator application."""

        @asset_bundle(name="ui_assets")
        class TestClass:
            pass

        assert hasattr(TestClass, "_asset_bundle")
        assert TestClass._asset_bundle is True
        assert hasattr(TestClass, "_asset_bundle_name")
        assert TestClass._asset_bundle_name == "ui_assets"
        assert hasattr(TestClass, "_asset_bundle_platforms")
        assert TestClass._asset_bundle_platforms is None

    def test_with_platforms(self):
        """Test @asset_bundle with platform specification."""

        @asset_bundle(name="textures", platforms={"windows", "linux", "macos"})
        class TestClass:
            pass

        assert TestClass._asset_bundle_name == "textures"
        assert TestClass._asset_bundle_platforms == frozenset(
            {"windows", "linux", "macos"}
        )

    def test_empty_name_raises_error(self):
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be a non-empty string"):

            @asset_bundle(name="")
            class TestClass:
                pass

    def test_invalid_name_type(self):
        """Test invalid name type raises TypeError."""
        with pytest.raises(TypeError, match="name must be a string"):

            @asset_bundle(name=123)
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @asset_bundle(name="sounds", platforms={"android", "ios"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("asset_bundle") is True
        assert TestClass._tags.get("asset_bundle_name") == "sounds"
        assert TestClass._tags.get("asset_bundle_platforms") == frozenset(
            {"android", "ios"}
        )
        assert hasattr(TestClass, "_registries")
        assert "build_deploy" in TestClass._registries

    def test_decorator_tracking(self):
        """Test decorator application tracking."""

        @asset_bundle(name="models")
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "asset_bundle" in TestClass._applied_decorators

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(asset_bundle)
        assert len(steps) == 4
        assert any(s.op.value == "tag" for s in steps)
        assert any(s.op.value == "register" for s in steps)


class TestFeatureFlag:
    """Test @feature_flag decorator."""

    def test_basic_application(self):
        """Test basic @feature_flag decorator application."""

        @feature_flag(id="new_ui")
        class TestClass:
            pass

        assert hasattr(TestClass, "_feature_flag")
        assert TestClass._feature_flag is True
        assert hasattr(TestClass, "_feature_flag_id")
        assert TestClass._feature_flag_id == "new_ui"
        assert hasattr(TestClass, "_feature_flag_default")
        assert TestClass._feature_flag_default is False

    def test_with_default_true(self):
        """Test @feature_flag with default=True."""

        @feature_flag(id="experimental_feature", default=True)
        class TestClass:
            pass

        assert TestClass._feature_flag_id == "experimental_feature"
        assert TestClass._feature_flag_default is True

    def test_empty_id_raises_error(self):
        """Test empty id raises ValueError."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @feature_flag(id="")
            class TestClass:
                pass

    def test_invalid_id_type(self):
        """Test invalid id type raises TypeError."""
        with pytest.raises(TypeError, match="id must be a string"):

            @feature_flag(id=456)
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @feature_flag(id="beta_mode", default=True)
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("feature_flag") is True
        assert TestClass._tags.get("feature_flag_id") == "beta_mode"
        assert TestClass._tags.get("feature_flag_default") is True
        assert hasattr(TestClass, "_registries")
        assert "build_deploy" in TestClass._registries

    def test_decorator_tracking(self):
        """Test decorator application tracking."""

        @feature_flag(id="advanced_settings")
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "feature_flag" in TestClass._applied_decorators

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(feature_flag)
        assert len(steps) == 4
        assert any(s.op.value == "tag" for s in steps)
        assert any(s.op.value == "register" for s in steps)

    def test_on_function(self):
        """Test @feature_flag on a function."""

        @feature_flag(id="experimental_api", default=True)
        def test_func():
            pass

        assert test_func._feature_flag is True
        assert test_func._feature_flag_id == "experimental_api"
        assert test_func._feature_flag_default is True


class TestComposition:
    """Test decorator composition and stacking."""

    def test_multiple_build_deploy_decorators(self):
        """Test stacking multiple build/deploy decorators."""

        @build_only(configurations={"debug"})
        @strip_in_release
        @feature_flag(id="test_feature")
        class TestClass:
            pass

        assert TestClass._build_only is True
        assert TestClass._strip_in_release is True
        assert TestClass._feature_flag is True

    def test_all_four_decorators(self):
        """Test stacking all four decorators."""

        @build_only(configurations={"debug", "test"})
        @strip_in_release()
        @asset_bundle(name="debug_assets", platforms={"windows"})
        @feature_flag(id="debug_mode", default=True)
        class TestClass:
            pass

        assert TestClass._build_only is True
        assert TestClass._strip_in_release is True
        assert TestClass._asset_bundle is True
        assert TestClass._feature_flag is True
        assert len(TestClass._applied_decorators) == 4


class TestRegistry:
    """Test registry integration."""

    def test_decorators_registered(self):
        """Test that all decorators are registered."""
        assert registry.get("build_only") is not None
        assert registry.get("strip_in_release") is not None
        assert registry.get("asset_bundle") is not None
        assert registry.get("feature_flag") is not None

    def test_tier_assignment(self):
        """Test that decorators are assigned to correct tier."""
        from trinity.decorators.registry import Tier

        tier_decorators = registry.by_tier(Tier.BUILD_DEPLOY)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "build_only" in decorator_names
        assert "strip_in_release" in decorator_names
        assert "asset_bundle" in decorator_names
        assert "feature_flag" in decorator_names

    def test_decorator_specs(self):
        """Test decorator specifications."""
        build_only_spec = registry.get("build_only")
        assert build_only_spec is not None
        assert build_only_spec.tier.value == 41
        assert not build_only_spec.foundation
        assert ("class", "function") == build_only_spec.target_types or (
            "function", "class"
        ) == build_only_spec.target_types


class TestValidation:
    """Test parameter validation."""

    def test_build_only_validation(self):
        """Test @build_only parameter validation."""
        with pytest.raises(ValueError):

            @build_only(configurations=set())
            class TestClass:
                pass

        with pytest.raises(TypeError):

            @build_only(configurations=["debug"])
            class TestClass:
                pass

    def test_asset_bundle_validation(self):
        """Test @asset_bundle parameter validation."""
        with pytest.raises(ValueError):

            @asset_bundle(name="")
            class TestClass:
                pass

        with pytest.raises(TypeError):

            @asset_bundle(name=999)
            class TestClass:
                pass

    def test_feature_flag_validation(self):
        """Test @feature_flag parameter validation."""
        with pytest.raises(ValueError):

            @feature_flag(id="")
            class TestClass:
                pass

        with pytest.raises(TypeError):

            @feature_flag(id=123)
            class TestClass:
                pass
