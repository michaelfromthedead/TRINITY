"""
Tests for the Resource Manager.

Tests resource allocation, lifetime tracking, and memory aliasing
as specified in RENDERING_CONTEXT.md Section 6.1.
"""

import pytest

from engine.rendering.framegraph import (
    ExternalResource,
    HistoryResource,
    ResourceDescriptor,
    ResourceFormat,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
    TransientResource,
)


# =============================================================================
# ResourceHandle
# =============================================================================


class TestResourceHandle:
    """Test ResourceHandle functionality."""

    def test_handle_creation(self):
        """Test creating a resource handle."""
        handle = ResourceHandle()

        assert handle.id is not None
        assert handle.version == 0
        assert handle.descriptor is None

    def test_handle_with_descriptor(self):
        """Test handle with descriptor."""
        descriptor = ResourceDescriptor(
            name="test_texture",
            format=ResourceFormat.R8G8B8A8_UNORM,
            width=1920,
            height=1080,
        )
        handle = ResourceHandle(descriptor=descriptor)

        assert handle.name == "test_texture"
        assert handle.resource_type == ResourceType.TRANSIENT

    def test_handle_name_fallback(self):
        """Test handle name fallback when no descriptor."""
        handle = ResourceHandle()

        assert handle.name.startswith("unnamed_")

    def test_handle_equality(self):
        """Test handle equality based on ID."""
        handle1 = ResourceHandle()
        handle2 = ResourceHandle()

        assert handle1 != handle2

        # Same ID means equal
        handle3 = ResourceHandle()
        handle3.id = handle1.id
        assert handle1 == handle3

    def test_handle_hash(self):
        """Test handle hashing."""
        handle = ResourceHandle()
        handles_set = {handle}

        assert handle in handles_set


# =============================================================================
# TransientResource
# =============================================================================


class TestTransientResource:
    """Test TransientResource functionality."""

    def test_transient_creation(self):
        """Test creating a transient resource."""
        handle = ResourceHandle()
        transient = TransientResource(handle=handle)

        assert transient.handle is handle
        assert transient.first_use_pass == -1
        assert transient.last_use_pass == -1
        assert transient.alias_group == -1
        assert transient.current_state == ResourceState.UNDEFINED

    def test_lifetime_overlap_check(self):
        """Test lifetime overlap detection."""
        handle1 = ResourceHandle()
        handle2 = ResourceHandle()

        t1 = TransientResource(handle=handle1, first_use_pass=0, last_use_pass=2)
        t2 = TransientResource(handle=handle2, first_use_pass=1, last_use_pass=3)
        t3 = TransientResource(handle=handle2, first_use_pass=3, last_use_pass=5)

        # t1 and t2 overlap (passes 1-2)
        assert t1.overlaps_with(t2) is True

        # t1 and t3 don't overlap
        assert t1.overlaps_with(t3) is False

    def test_lifetime_overlap_uninitialized(self):
        """Test that uninitialized resources don't overlap."""
        handle1 = ResourceHandle()
        handle2 = ResourceHandle()

        t1 = TransientResource(handle=handle1)  # first_use = -1
        t2 = TransientResource(handle=handle2, first_use_pass=0, last_use_pass=1)

        assert t1.overlaps_with(t2) is False


# =============================================================================
# HistoryResource
# =============================================================================


class TestHistoryResource:
    """Test HistoryResource functionality."""

    def test_history_creation(self):
        """Test creating a history resource."""
        handle = ResourceHandle()
        history = HistoryResource(handle=handle)

        assert history.handle is handle
        assert history.frame_count == 0
        assert history.double_buffered is True
        assert history.current_index == 0

    def test_buffer_swap(self):
        """Test double buffer swapping."""
        handle = ResourceHandle()
        history = HistoryResource(handle=handle, double_buffered=True)

        assert history.current_index == 0
        assert history.frame_count == 0

        history.swap_buffers()

        assert history.current_index == 1
        assert history.frame_count == 1

        history.swap_buffers()

        assert history.current_index == 0
        assert history.frame_count == 2

    def test_no_swap_when_single_buffered(self):
        """Test that single-buffered resources don't swap."""
        handle = ResourceHandle()
        history = HistoryResource(handle=handle, double_buffered=False)

        history.swap_buffers()
        history.swap_buffers()

        assert history.current_index == 0


# =============================================================================
# ResourceManager
# =============================================================================


class TestResourceManager:
    """Test ResourceManager functionality."""

    def test_create_transient_texture(self):
        """Test creating a transient texture resource."""
        rm = ResourceManager()

        handle = rm.create_transient(
            "albedo",
            format=ResourceFormat.R8G8B8A8_UNORM,
            width=1920,
            height=1080,
        )

        assert handle is not None
        assert handle.name == "albedo"

        transient = rm.get_transient("albedo")
        assert transient is not None
        assert transient.handle is handle

    def test_create_transient_buffer(self):
        """Test creating a transient buffer."""
        rm = ResourceManager()

        handle = rm.create_buffer(
            "indirect_args",
            size_bytes=1024,
            resource_type=ResourceType.TRANSIENT,
        )

        assert handle is not None
        transient = rm.get_transient("indirect_args")
        assert transient is not None
        assert transient.size_bytes == 1024

    def test_create_history(self):
        """Test creating a history resource."""
        rm = ResourceManager()

        handle = rm.create_history(
            "taa_history",
            format=ResourceFormat.R16G16B16A16_FLOAT,
            width=1920,
            height=1080,
            double_buffered=True,
        )

        assert handle is not None
        assert handle.name == "taa_history"

        history = rm.get_history("taa_history")
        assert history is not None
        assert history.double_buffered is True

    def test_register_external(self):
        """Test registering an external resource."""
        rm = ResourceManager()
        fake_resource = object()

        handle = rm.register_external(
            "backbuffer",
            gpu_resource=fake_resource,
            is_backbuffer=True,
        )

        assert handle is not None
        assert handle.name == "backbuffer"

        external = rm.get_external("backbuffer")
        assert external is not None
        assert external.gpu_resource is fake_resource
        assert external.is_backbuffer is True

    def test_duplicate_name_fails(self):
        """Test that duplicate resource names are rejected."""
        rm = ResourceManager()
        rm.create_transient("duplicate")

        with pytest.raises(ValueError, match="already exists"):
            rm.create_transient("duplicate")

        with pytest.raises(ValueError, match="already exists"):
            rm.create_history("duplicate")

        with pytest.raises(ValueError, match="already exists"):
            rm.register_external("duplicate", None)

    def test_get_handle(self):
        """Test getting a handle by name."""
        rm = ResourceManager()
        created = rm.create_transient("test")

        retrieved = rm.get_handle("test")
        assert retrieved is created

    def test_get_nonexistent_handle(self):
        """Test getting a handle that doesn't exist."""
        rm = ResourceManager()

        result = rm.get_handle("nonexistent")
        assert result is None


# =============================================================================
# Resource Aliasing
# =============================================================================


class TestResourceAliasing:
    """Test resource memory aliasing."""

    def test_update_lifetime(self):
        """Test updating resource lifetimes."""
        rm = ResourceManager()

        handle = rm.create_transient("test")
        transient = rm.get_transient("test")

        rm.update_lifetime(handle, pass_index=0)
        assert transient.first_use_pass == 0
        assert transient.last_use_pass == 0

        rm.update_lifetime(handle, pass_index=3)
        assert transient.first_use_pass == 0
        assert transient.last_use_pass == 3

    def test_compute_aliasing_non_overlapping(self):
        """Test aliasing for non-overlapping resources."""
        rm = ResourceManager()

        # Create resources used at different times
        h1 = rm.create_transient("early")
        h2 = rm.create_transient("late")

        # Set lifetimes (non-overlapping)
        rm.update_lifetime(h1, pass_index=0)
        rm.update_lifetime(h1, pass_index=1)
        rm.update_lifetime(h2, pass_index=2)
        rm.update_lifetime(h2, pass_index=3)

        rm.compute_aliasing()

        t1 = rm.get_transient("early")
        t2 = rm.get_transient("late")

        # Should be in same alias group (can share memory)
        assert t1.alias_group == t2.alias_group
        assert rm.get_alias_group_count() == 1

    def test_compute_aliasing_overlapping(self):
        """Test aliasing for overlapping resources."""
        rm = ResourceManager()

        # Create resources used at overlapping times
        h1 = rm.create_transient("a")
        h2 = rm.create_transient("b")

        # Set overlapping lifetimes
        rm.update_lifetime(h1, pass_index=0)
        rm.update_lifetime(h1, pass_index=2)
        rm.update_lifetime(h2, pass_index=1)
        rm.update_lifetime(h2, pass_index=3)

        rm.compute_aliasing()

        t1 = rm.get_transient("a")
        t2 = rm.get_transient("b")

        # Should be in different alias groups (cannot share memory)
        assert t1.alias_group != t2.alias_group
        assert rm.get_alias_group_count() == 2

    def test_compute_aliasing_unused_resources(self):
        """Test that unused resources are skipped."""
        rm = ResourceManager()

        # Create a resource but never set its lifetime
        rm.create_transient("unused")

        rm.compute_aliasing()

        t = rm.get_transient("unused")
        assert t.alias_group == -1


# =============================================================================
# Frame Lifecycle
# =============================================================================


class TestFrameLifecycle:
    """Test per-frame resource management."""

    def test_begin_frame_resets_transients(self):
        """Test that begin_frame resets transient resources."""
        rm = ResourceManager()

        handle = rm.create_transient("test")
        transient = rm.get_transient("test")

        # Simulate usage during a frame
        rm.update_lifetime(handle, 0)
        rm.update_lifetime(handle, 2)
        rm.compute_aliasing()

        assert transient.first_use_pass == 0
        assert transient.alias_group >= 0

        # Begin new frame
        rm.begin_frame()

        # Should be reset
        assert transient.first_use_pass == -1
        assert transient.last_use_pass == -1
        assert transient.alias_group == -1

    def test_begin_frame_swaps_history_buffers(self):
        """Test that begin_frame swaps history buffers."""
        rm = ResourceManager()
        rm.create_history("history", double_buffered=True)

        history = rm.get_history("history")
        assert history.current_index == 0
        assert history.frame_count == 0

        rm.begin_frame()

        assert history.current_index == 1
        assert history.frame_count == 1

    def test_clear(self):
        """Test clearing all resources."""
        rm = ResourceManager()
        rm.create_transient("t1")
        rm.create_history("h1")
        rm.register_external("e1", None)

        rm.clear()

        assert rm.get_transient("t1") is None
        assert rm.get_history("h1") is None
        assert rm.get_external("e1") is None
        assert rm.get_handle("t1") is None


# =============================================================================
# ResourceDescriptor
# =============================================================================


class TestResourceDescriptor:
    """Test ResourceDescriptor functionality."""

    def test_texture_descriptor(self):
        """Test creating a texture descriptor."""
        desc = ResourceDescriptor(
            name="gbuffer_albedo",
            resource_type=ResourceType.TRANSIENT,
            format=ResourceFormat.R8G8B8A8_UNORM,
            width=1920,
            height=1080,
            mip_levels=1,
            sample_count=4,
            is_texture=True,
        )

        assert desc.name == "gbuffer_albedo"
        assert desc.is_texture is True
        assert desc.sample_count == 4

    def test_buffer_descriptor(self):
        """Test creating a buffer descriptor."""
        desc = ResourceDescriptor(
            name="vertex_buffer",
            resource_type=ResourceType.TRANSIENT,
            is_texture=False,
            buffer_size=65536,
        )

        assert desc.name == "vertex_buffer"
        assert desc.is_texture is False
        assert desc.buffer_size == 65536


# =============================================================================
# Input Validation
# =============================================================================


class TestInputValidation:
    """Test input validation for resource creation."""

    def test_create_buffer_negative_size_fails(self):
        """Test that negative buffer size is rejected."""
        rm = ResourceManager()

        with pytest.raises(ValueError, match="must be positive"):
            rm.create_buffer("bad_buffer", size_bytes=-100)

    def test_create_buffer_zero_size_fails(self):
        """Test that zero buffer size is rejected."""
        rm = ResourceManager()

        with pytest.raises(ValueError, match="must be positive"):
            rm.create_buffer("bad_buffer", size_bytes=0)

    def test_create_buffer_positive_size_succeeds(self):
        """Test that positive buffer size is accepted."""
        rm = ResourceManager()

        handle = rm.create_buffer("good_buffer", size_bytes=1)
        assert handle is not None


# =============================================================================
# ResourceDescriptor Continued
# =============================================================================


class TestResourceDescriptorContinued:
    """Additional ResourceDescriptor tests."""

    def test_descriptor_defaults(self):
        """Test descriptor default values."""
        desc = ResourceDescriptor(name="test")

        assert desc.resource_type == ResourceType.TRANSIENT
        assert desc.format == ResourceFormat.R8G8B8A8_UNORM
        assert desc.width == 0
        assert desc.height == 0
        assert desc.depth == 1
        assert desc.mip_levels == 1
        assert desc.sample_count == 1
        assert desc.is_texture is True
        assert desc.clear_value is None
