"""Tests for descriptor_steps property across all Phase 2 descriptors."""
import pytest
from trinity.decorators.ops import Step, Op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_steps_list(steps):
    """Every descriptor_steps must return a list of Step objects."""
    assert isinstance(steps, list)
    for s in steps:
        assert isinstance(s, Step)
        assert isinstance(s.op, Op)
        assert isinstance(s.args, dict)


# ===========================================================================
# base.py
# ===========================================================================

class TestBaseDescriptorSteps:
    def test_returns_empty_list(self):
        from trinity.descriptors.base import BaseDescriptor
        d = BaseDescriptor()
        assert d.descriptor_steps == []

    def test_is_list(self):
        from trinity.descriptors.base import BaseDescriptor
        d = BaseDescriptor()
        _assert_steps_list(d.descriptor_steps)


# ===========================================================================
# storage.py
# ===========================================================================

class TestStorageDescriptorSteps:
    def test_returns_empty_list(self):
        from trinity.descriptors.storage import StorageDescriptor
        d = StorageDescriptor()
        assert d.descriptor_steps == []

    def test_is_list(self):
        from trinity.descriptors.storage import StorageDescriptor
        d = StorageDescriptor()
        _assert_steps_list(d.descriptor_steps)


# ===========================================================================
# tracking.py
# ===========================================================================

class TestTrackedDescriptorSteps:
    def test_without_bitmask(self):
        from trinity.descriptors.tracking import TrackedDescriptor
        d = TrackedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.TRACK
        assert "field" in steps[0].args

    def test_with_bitmask(self):
        from trinity.descriptors.tracking import TrackedDescriptor
        d = TrackedDescriptor(use_bitmask=True)
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0].op == Op.TRACK
        assert steps[1].op == Op.TAG
        assert steps[1].args["key"] == "track_bitmask"
        assert steps[1].args["value"] is True


class TestVersionedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.tracking import VersionedDescriptor
        d = VersionedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.TRACK
        assert steps[0].args.get("strategy") == "versioned"


class TestDiffDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.tracking import DiffDescriptor
        d = DiffDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.TRACK
        assert steps[0].args.get("strategy") == "shallow"


# ===========================================================================
# validation.py
# ===========================================================================

class TestValidatedDescriptorSteps:
    def test_no_validators(self):
        from trinity.descriptors.validation import ValidatedDescriptor
        d = ValidatedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.VALIDATE
        assert steps[0].args["constraint"] == "custom"
        assert steps[0].args["validator_count"] == 0

    def test_with_validators(self):
        from trinity.descriptors.validation import ValidatedDescriptor
        d = ValidatedDescriptor(validators=[lambda v: True, lambda v: True])
        steps = d.descriptor_steps
        assert steps[0].args["validator_count"] == 2


class TestRangeDescriptorSteps:
    def test_default_range(self):
        from trinity.descriptors.validation import RangeDescriptor
        d = RangeDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.VALIDATE
        assert steps[0].args["constraint"] == "range"
        assert steps[0].args["clamp"] is True

    def test_custom_range(self):
        from trinity.descriptors.validation import RangeDescriptor
        d = RangeDescriptor(min_val=0.0, max_val=100.0, clamp=False)
        steps = d.descriptor_steps
        assert steps[0].args["min"] == 0.0
        assert steps[0].args["max"] == 100.0
        assert steps[0].args["clamp"] is False


class TestTypeDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.validation import TypeDescriptor
        d = TypeDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.VALIDATE
        assert steps[0].args.get("rule") == "type"

    def test_expected_type(self):
        from trinity.descriptors.validation import TypeDescriptor
        d = TypeDescriptor(expected_type=int)
        steps = d.descriptor_steps
        assert steps[0].args.get("expected_type") == "<class 'int'>"


class TestChoiceDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.validation import ChoiceDescriptor
        d = ChoiceDescriptor(choices=["a", "b"])
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.VALIDATE
        assert steps[0].args.get("rule") == "choice"
        assert steps[0].args.get("choices") == ["a", "b"]


class TestPatternDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.validation import PatternDescriptor
        d = PatternDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.VALIDATE
        assert steps[0].args.get("rule") == "pattern"
        assert steps[0].args.get("pattern") == ".*"

    def test_custom_pattern(self):
        from trinity.descriptors.validation import PatternDescriptor
        d = PatternDescriptor(pattern=r"\d+")
        steps = d.descriptor_steps
        assert steps[0].args.get("pattern") == r"\d+"


# ===========================================================================
# observable.py
# ===========================================================================

class TestObservableDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.observable import ObservableDescriptor
        d = ObservableDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.HOOK
        assert steps[0].args["event"] == "on_change"
        assert steps[0].args["callback"] == "observer_dispatch"


class TestBoundDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.observable import BoundDescriptor
        d = BoundDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.INTERCEPT
        assert steps[0].args.get("strategy") == "bound"


# ===========================================================================
# networking.py
# ===========================================================================

class TestNetworkedDescriptorSteps:
    def test_default_steps(self):
        from trinity.descriptors.networking import NetworkedDescriptor
        d = NetworkedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 4
        assert steps[0] == Step(Op.TAG, {"key": "networked", "value": True})
        assert steps[1] == Step(Op.TAG, {"key": "authority", "value": "server"})
        assert steps[2] == Step(Op.TAG, {"key": "interpolated", "value": False})
        assert steps[3] == Step(Op.INTERCEPT, {"set": "network_queue"})

    def test_custom_authority(self):
        from trinity.descriptors.networking import NetworkedDescriptor
        d = NetworkedDescriptor(authority="client", interpolated=True)
        steps = d.descriptor_steps
        assert steps[1].args["value"] == "client"
        assert steps[2].args["value"] is True


class TestInterpolatedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.networking import InterpolatedDescriptor
        d = InterpolatedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.INTERCEPT
        assert steps[0].args.get("strategy") == "interpolation"
        assert steps[0].args.get("method") == "linear"

    def test_hermite_mode(self):
        from trinity.descriptors.networking import InterpolatedDescriptor
        d = InterpolatedDescriptor(mode="hermite")
        steps = d.descriptor_steps
        assert steps[0].args.get("method") == "hermite"


class TestPredictedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.networking import PredictedDescriptor
        d = PredictedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.INTERCEPT
        assert steps[0].args.get("strategy") == "prediction"


class TestThrottledNetworkDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.networking import ThrottledNetworkDescriptor
        d = ThrottledNetworkDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0].op == Op.INTERCEPT
        assert steps[0].args.get("strategy") == "throttle"


# ===========================================================================
# persistence.py
# ===========================================================================

class TestSerializableDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.persistence import SerializableDescriptor
        d = SerializableDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 3
        assert steps[0] == Step(Op.HOOK, {"event": "on_serialize", "callback": "encode"})
        assert steps[1] == Step(Op.HOOK, {"event": "on_deserialize", "callback": "decode"})
        assert steps[2].op == Op.TAG
        assert steps[2].args["key"] == "serialization_format"

    def test_custom_format(self):
        from trinity.descriptors.persistence import SerializableDescriptor
        d = SerializableDescriptor(format="binary")
        steps = d.descriptor_steps
        assert steps[2].args["value"] == "binary"


class TestTransientDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.persistence import TransientDescriptor
        d = TransientDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 1
        assert steps[0] == Step(Op.TAG, {"key": "transient", "value": True})


class TestMigratedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.persistence import MigratedDescriptor
        d = MigratedDescriptor(from_name="old_field")
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.TAG, {"key": "migrated_from", "value": "old_field"})
        assert steps[1].op == Op.TAG
        assert steps[1].args["key"] == "version_added"


class TestEncryptedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.persistence import EncryptedDescriptor
        d = EncryptedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"get": "decrypt", "set": "encrypt"})
        assert steps[1] == Step(Op.TAG, {"key": "encrypted", "value": True})


# ===========================================================================
# caching.py
# ===========================================================================

class TestCachedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.caching import CachedDescriptor
        d = CachedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"get": "cache_check"})
        assert steps[1].op == Op.TAG
        assert steps[1].args["key"] == "ttl"

    def test_ttl_value(self):
        from trinity.descriptors.caching import CachedDescriptor
        d = CachedDescriptor(ttl=5.0)
        steps = d.descriptor_steps
        assert steps[1].args["value"] == 5.0

    def test_ttl_none(self):
        from trinity.descriptors.caching import CachedDescriptor
        d = CachedDescriptor()
        steps = d.descriptor_steps
        assert steps[1].args["value"] is None


class TestComputedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.caching import ComputedDescriptor
        d = ComputedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 3
        assert steps[0] == Step(Op.INTERCEPT, {"get": "compute", "set": "deny", "delete": "deny"})
        assert steps[1] == Step(Op.TAG, {"key": "computed", "value": True})
        assert steps[2] == Step(Op.TAG, {"key": "transient", "value": True})


# ===========================================================================
# debug.py
# ===========================================================================

class TestProfiledDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.debug import ProfiledDescriptor
        d = ProfiledDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"get": "profile_get", "set": "profile_set"})
        assert steps[1] == Step(Op.TAG, {"key": "profiled", "value": True})


class TestLoggedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.debug import LoggedDescriptor
        d = LoggedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"get": "log_get", "set": "log_set"})
        assert steps[1] == Step(Op.TAG, {"key": "logged", "value": True})


class TestWatchedDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.debug import WatchedDescriptor
        d = WatchedDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"set": "watch_condition"})
        assert steps[1] == Step(Op.TAG, {"key": "watched", "value": True})


# ===========================================================================
# async_descriptors.py
# ===========================================================================

class TestLazyDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.async_descriptors import LazyDescriptor
        d = LazyDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 3
        assert steps[0] == Step(Op.INTERCEPT, {"get": "lazy_init"})
        assert steps[1] == Step(Op.TAG, {"key": "lazy", "value": True})
        assert steps[2] == Step(Op.TAG, {"key": "init_mode", "value": "first_access"})

    def test_explicit_mode(self):
        from trinity.descriptors.async_descriptors import LazyDescriptor
        d = LazyDescriptor(init_mode="explicit")
        steps = d.descriptor_steps
        assert steps[2].args["value"] == "explicit"


class TestAsyncLoadDescriptorSteps:
    def test_steps(self):
        from trinity.descriptors.async_descriptors import AsyncLoadDescriptor
        d = AsyncLoadDescriptor()
        steps = d.descriptor_steps
        _assert_steps_list(steps)
        assert len(steps) == 2
        assert steps[0] == Step(Op.INTERCEPT, {"get": "async_load"})
        assert steps[1] == Step(Op.TAG, {"key": "async_load", "value": True})


# ===========================================================================
# composer.py - collect_steps
# ===========================================================================

class TestComposerCollectSteps:
    def test_single_descriptor_no_steps(self):
        from trinity.descriptors.base import BaseDescriptor
        from trinity.descriptors.composer import DescriptorComposer
        d = BaseDescriptor()
        assert DescriptorComposer.collect_steps(d) == []

    def test_single_descriptor_with_steps(self):
        from trinity.descriptors.observable import ObservableDescriptor
        from trinity.descriptors.composer import DescriptorComposer
        d = ObservableDescriptor()
        steps = DescriptorComposer.collect_steps(d)
        assert len(steps) == 1
        assert steps[0].op == Op.HOOK

    def test_chain_aggregates_steps(self):
        from trinity.descriptors.tracking import TrackedDescriptor
        from trinity.descriptors.storage import StorageDescriptor
        from trinity.descriptors.composer import DescriptorComposer

        storage = StorageDescriptor(field_type=float, default=0.0)
        tracked = TrackedDescriptor(field_type=float, inner=storage)

        steps = DescriptorComposer.collect_steps(tracked)
        # tracked has 1 TRACK step, storage has 0
        assert len(steps) == 1
        assert steps[0].op == Op.TRACK

    def test_three_layer_chain(self):
        from trinity.descriptors.networking import NetworkedDescriptor
        from trinity.descriptors.tracking import TrackedDescriptor
        from trinity.descriptors.storage import StorageDescriptor
        from trinity.descriptors.composer import DescriptorComposer

        storage = StorageDescriptor(field_type=float)
        tracked = TrackedDescriptor(field_type=float, inner=storage)
        networked = NetworkedDescriptor(field_type=float, inner=tracked)

        steps = DescriptorComposer.collect_steps(networked)
        # networked: 4 steps, tracked: 1, storage: 0
        assert len(steps) == 5
        ops = [s.op for s in steps]
        assert ops.count(Op.TAG) == 3
        assert ops.count(Op.INTERCEPT) == 1
        assert ops.count(Op.TRACK) == 1

    def test_compose_then_collect(self):
        from trinity.descriptors.networking import NetworkedDescriptor
        from trinity.descriptors.tracking import TrackedDescriptor
        from trinity.descriptors.storage import StorageDescriptor
        from trinity.descriptors.composer import DescriptorComposer

        chain = DescriptorComposer.compose(
            NetworkedDescriptor(field_type=float),
            TrackedDescriptor(field_type=float),
            StorageDescriptor(field_type=float, default=0.0),
        )
        steps = DescriptorComposer.collect_steps(chain)
        assert len(steps) == 5
        # First steps come from outermost (networked)
        assert steps[0].op == Op.TAG
        assert steps[0].args["key"] == "networked"
