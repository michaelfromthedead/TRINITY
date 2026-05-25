//! Swapchain and descriptor heap tests.
//!
//! Mirrors tests/platform/rhi/test_swapchain.py and adds binding tests.

mod common;

use common::*;

// =========================================================================
// Swapchain tests
// =========================================================================

fn make_swapchain_desc() -> SwapchainDesc {
    SwapchainDesc {
        width: 1920,
        height: 1080,
        format: Format::RGBA8Unorm,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    }
}

#[test]
fn test_swapchain_creation() {
    let desc = make_swapchain_desc();
    let sc = MockSwapchain::new(desc);

    assert_eq!(sc.buffer_count(), 2);
}

#[test]
fn test_swapchain_current_texture() {
    let desc = SwapchainDesc {
        width: 800,
        height: 600,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);
    let texture = sc.current_texture();

    assert!(texture.is_valid());
    assert_eq!(texture.desc().width, 800);
    assert_eq!(texture.desc().height, 600);
}

#[test]
fn test_swapchain_current_index() {
    let desc = make_swapchain_desc();
    let sc = MockSwapchain::new(desc);
    let index = sc.current_index();

    assert_eq!(index, 0);
}

#[test]
fn test_swapchain_present_advances_index() {
    let desc = SwapchainDesc {
        width: 1920,
        height: 1080,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);
    let initial = sc.current_index();

    sc.present();
    let new_index = sc.current_index();

    assert_eq!(new_index, (initial + 1) % 3);
}

#[test]
fn test_swapchain_present_wraps_around() {
    let desc = SwapchainDesc {
        width: 800,
        height: 600,
        format: Format::RGBA8Unorm,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);

    for _ in 0..5 {
        sc.present();
    }

    let index = sc.current_index();
    assert!(index < 2);
}

#[test]
fn test_swapchain_resize() {
    let desc = SwapchainDesc {
        width: 800,
        height: 600,
        format: Format::RGBA8Unorm,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);
    let old_texture = sc.current_texture();
    let old_handle = old_texture.handle();

    // Resize
    sc.resize(1920, 1080);

    let new_texture = sc.current_texture();
    assert_eq!(new_texture.desc().width, 1920);
    assert_eq!(new_texture.desc().height, 1080);
    assert_ne!(new_texture.handle(), old_handle);
}

#[test]
fn test_swapchain_resize_resets_index() {
    let desc = SwapchainDesc {
        width: 640,
        height: 480,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);

    // Advance to index 2
    sc.present();
    sc.present();
    assert_eq!(sc.current_index(), 2);

    // Resize should reset to 0
    sc.resize(1024, 768);
    assert_eq!(sc.current_index(), 0);
}

#[test]
fn test_swapchain_triple_buffering() {
    let desc = SwapchainDesc {
        width: 1920,
        height: 1080,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Mailbox,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);

    // All three buffers should be visited across three presents
    let mut indices = std::collections::HashSet::new();
    for _ in 0..3 {
        indices.insert(sc.current_index());
        sc.present();
    }

    assert_eq!(indices.len(), 3);
}

#[test]
fn test_swapchain_different_buffer_counts() {
    for count in &[2u32, 3u32, 4u32] {
        let desc = SwapchainDesc {
            width: 800,
            height: 600,
            format: Format::RGBA8Unorm,
            buffer_count: *count,
            present_mode: PresentMode::Vsync,
            color_space: ColorSpace::Srgb,
        };
        let sc = MockSwapchain::new(desc);
        assert_eq!(sc.buffer_count(), *count);
    }
}

#[test]
fn test_swapchain_present_modes() {
    let base = SwapchainDesc {
        width: 800,
        height: 600,
        format: Format::RGBA8Unorm,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };

    for mode in &[PresentMode::Immediate, PresentMode::Vsync, PresentMode::Mailbox] {
        let sc = MockSwapchain::new(SwapchainDesc {
            present_mode: *mode,
            ..base.clone()
        });
        // Index starts at 0 regardless of mode
        assert_eq!(sc.current_index(), 0);
    }
}

#[test]
fn test_swapchain_color_spaces() {
    let base = SwapchainDesc {
        width: 1920,
        height: 1080,
        format: Format::RGBA8Unorm,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };

    let sc_srgb = MockSwapchain::new(SwapchainDesc {
        color_space: ColorSpace::Srgb,
        ..base.clone()
    });
    assert_eq!(sc_srgb.current_texture().desc().format, Format::RGBA8Unorm);

    let sc_hdr = MockSwapchain::new(SwapchainDesc {
        format: Format::RGBA16Float,
        color_space: ColorSpace::Hdr10,
        ..base
    });
    assert_eq!(sc_hdr.current_texture().desc().format, Format::RGBA16Float);
}

#[test]
fn test_swapchain_hdr_format() {
    let desc = SwapchainDesc {
        width: 3840,
        height: 2160,
        format: Format::RGBA16Float,
        buffer_count: 2,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Hdr10,
    };
    let sc = MockSwapchain::new(desc);
    let texture = sc.current_texture();

    assert_eq!(texture.desc().format, Format::RGBA16Float);
}

#[test]
fn test_swapchain_present_cycle_consistency() {
    let desc = SwapchainDesc {
        width: 1024,
        height: 768,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);

    // Full cycle through all buffers
    for i in 0..6 {
        let idx = sc.current_index();
        assert_eq!(idx, (i % 3) as u32);
        sc.present();
    }
}

// =========================================================================
// Binding / Descriptor Heap tests
// =========================================================================

#[test]
fn test_descriptor_heap_creation() {
    let heap = MockDescriptorHeap::new(DescriptorType::Srv, 64);
    assert_eq!(heap.descriptor_type(), DescriptorType::Srv);
}

#[test]
fn test_descriptor_heap_allocate() {
    let heap = MockDescriptorHeap::new(DescriptorType::Cbv, 16);
    let handle = heap.allocate();

    assert!(handle.is_some());
    assert_eq!(handle.as_ref().unwrap().offset, 0);
}

#[test]
fn test_descriptor_heap_allocate_multiple() {
    let heap = MockDescriptorHeap::new(DescriptorType::Srv, 64);
    let mut offsets = Vec::new();

    for _ in 0..5 {
        let h = heap.allocate().expect("Should allocate");
        offsets.push(h.offset);
    }

    assert_eq!(offsets, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_descriptor_heap_free_and_reuse() {
    let heap = MockDescriptorHeap::new(DescriptorType::Uav, 16);
    let h1 = heap.allocate().unwrap();
    let _h2 = heap.allocate().unwrap();

    heap.free(h1);
    let h3 = heap.allocate().expect("Should reuse freed slot");

    // Should reuse the freed offset
    assert_eq!(h3.offset, h1.offset);
    assert_eq!(h3.heap_index, h1.heap_index);
}

#[test]
fn test_descriptor_heap_allocate_all() {
    let heap = MockDescriptorHeap::new(DescriptorType::Sampler, 3);

    for _ in 0..3 {
        assert!(heap.allocate().is_some());
    }

    // Should be full now
    assert!(heap.allocate().is_none());
}

#[test]
fn test_descriptor_heap_free_wrong_heap_noop() {
    let heap = MockDescriptorHeap::new(DescriptorType::Srv, 8);
    let h1 = heap.allocate().unwrap();

    // Freeing a handle from a different heap index should be a noop
    let wrong_handle = DescriptorHandle {
        heap_index: 9999,
        offset: h1.offset,
    };
    heap.free(wrong_handle);

    // Allocate should return the next sequential offset (1), not reuse offset 0
    let h2 = heap.allocate().unwrap();
    assert_eq!(h2.offset, 1); // noop free did not add offset 0 to free list
}

#[test]
fn test_descriptor_heap_type_preservation() {
    let heap_srv = MockDescriptorHeap::new(DescriptorType::Srv, 32);
    assert_eq!(heap_srv.descriptor_type(), DescriptorType::Srv);

    let heap_uav = MockDescriptorHeap::new(DescriptorType::Uav, 32);
    assert_eq!(heap_uav.descriptor_type(), DescriptorType::Uav);

    let heap_cbv = MockDescriptorHeap::new(DescriptorType::Cbv, 32);
    assert_eq!(heap_cbv.descriptor_type(), DescriptorType::Cbv);
}

#[test]
fn test_descriptor_heap_large_allocation() {
    let heap = MockDescriptorHeap::new(DescriptorType::Srv, 1024);
    for i in 0..1024 {
        let h = heap.allocate();
        assert!(h.is_some(), "Failed to allocate slot {}", i);
        assert_eq!(h.unwrap().offset, i);
    }
    assert!(heap.allocate().is_none());
}

#[test]
fn test_descriptor_heap_free_chain() {
    let heap = MockDescriptorHeap::new(DescriptorType::Cbv, 32);
    let mut handles = Vec::new();

    // Allocate 5
    for _ in 0..5 {
        handles.push(heap.allocate().unwrap());
    }

    // Free 2, 4 (indices 1 and 3)
    heap.free(handles.remove(3)); // offset 3
    heap.free(handles.remove(1)); // offset 1

    // Next allocations should reuse freed slots (LIFO from free list)
    // Freed order: offset 3 then offset 1. LIFO pop returns offset 1 first, then offset 3.
    let r1 = heap.allocate().unwrap();
    let r2 = heap.allocate().unwrap();
    assert_eq!(r1.offset, 1); // Last freed (LIFO: offset 1 popped first)
    assert_eq!(r2.offset, 3); // First freed (LIFO: offset 3 popped second)
}
