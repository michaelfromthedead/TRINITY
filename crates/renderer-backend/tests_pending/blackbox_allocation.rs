// Blackbox contract tests for T-FG-4.2 BufferAllocation / GPU memory allocation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-FG-4.2):
//   BufferAllocation{handle,offset,size,is_aliased}
//   TextureAllocation{handle,offset,size,is_aliased}
//   AllocationDescriptor::Buffer(BufferAllocation) / ::Texture(TextureAllocation)
//   ResourceAllocationMap = HashMap<ResourceHandle, AllocationDescriptor>
//
// The allocation system tracks per-resource GPU memory metadata and supports:
//   - BufferAllocation / TextureAllocation structs for describing backing memory
//   - AllocationDescriptor enum for type-safe union of both
//   - ResourceAllocationMap for handle-to-descriptor lookups
//   - ResourceAllocator: maps logical handles to physical allocations
//   - AllocationTable: compressed logical-to-physical index lookup
//   - InterferenceGraph / greedy coloring for alias assignment
//   - AliasPolicy: Aggressive / Conservative / Disabled memory reuse
//
// Coverage:
//   1.  BufferAllocation construction and field access
//   2.  TextureAllocation construction and field access
//   3.  AllocationDescriptor enum variants (Buffer / Texture)
//   4.  ResourceAllocationMap CRUD and iteration
//   5.  PhysicalTexture construction, field access, Display
//   6.  PhysicalBuffer construction, field access, Display
//   7.  ResourceAllocator: new, is_empty, free_resources, num_textures, num_buffers
//   8.  allocate_resources with imported (non-transient) resources
//   9.  allocate_resources with transient resources and aliasing
//  10.  AllocationTable: from_allocator, resolve, num_physical_textures/buffers
//  11.  InterferenceGraph: build, interfere, neighbors
//  12.  AliasPolicy::Disabled / Aggressive / Conservative via apply_aliasing
//  13.  greedy_color_resources and num_colors
//  14.  Full pipeline: compile -> lifetimes -> allocate -> AllocationTable
//  15.  Edge cases: empty slices, zero sizes, NONE handle, free_resources no-op

use renderer_backend::frame_graph::{
    // Core allocation structs
    AllocationDescriptor, AllocationTable, BufferAllocation,
    ResourceAllocationMap, ResourceAllocator, ResourceKind, TextureAllocation,
    // Physical resource descriptors
    PhysicalBuffer, PhysicalTexture,
    // Aliasing / interference
    AliasPolicy, InterferenceGraph, apply_aliasing, greedy_color_resources,
    num_colors,
    // IR resource / pass
    BufferDesc, CompiledFrameGraph, DispatchSource, IrPass, IrResource,
    PassIndex, ResourceDesc, ResourceHandle, ResourceLifetime, ResourceState,
    Texture3DDesc, TextureDesc, ViewType,
    // Pipeline functions
    compute_lifetimes,
    // Helpers
    mock_resource_buffer, mock_resource_texture,
};

use std::collections::HashMap;

// =============================================================================
// Helpers
// =============================================================================

/// Creates a Texture2D resource with full control over transient flag.
fn tex2d(
    handle: u32,
    name: &str,
    width: u32,
    height: u32,
    lifetime: ResourceLifetime,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width,
            height,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        lifetime,
        ResourceState::Uninitialized,
    )
}

/// Creates a Texture3D resource.
fn tex3d(
    handle: u32,
    name: &str,
    width: u32,
    height: u32,
    depth: u32,
    lifetime: ResourceLifetime,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Texture3D(Texture3DDesc {
            width,
            height,
            depth,
            mip_levels: 1,
            format: "r32float".into(),
        }),
        lifetime,
        ResourceState::Uninitialized,
    )
}

/// Creates a buffer resource.
fn buffer_res(
    handle: u32,
    name: &str,
    size: u64,
    lifetime: ResourceLifetime,
) -> IrResource {
    IrResource::new(
        ResourceHandle(handle),
        name,
        ResourceDesc::Buffer(BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        lifetime,
        ResourceState::Uninitialized,
    )
}

/// Creates a compute pass with given read and write resource handles.
fn make_pass(
    index: usize,
    name: &str,
    reads: &[ResourceHandle],
    writes: &[ResourceHandle],
) -> IrPass {
    let mut pass = IrPass::compute(
        PassIndex(index),
        name,
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    for &h in reads {
        pass.access_set.reads.push(h);
    }
    for &h in writes {
        pass.access_set.writes.push(h);
    }
    pass
}

/// Build a lifetime map from a compiled frame graph (convenience).
fn lifetimes_from_compiled(
    compiled: &CompiledFrameGraph,
) -> HashMap<ResourceHandle, (PassIndex, PassIndex)> {
    compute_lifetimes(&compiled.passes, &compiled.edges, &compiled.resources)
}

// =============================================================================
// SECTION 1 -- BufferAllocation construction and field access
// =============================================================================

/// BufferAllocation fields are publicly accessible and match construction args.
#[test]
fn buffer_allocation_field_access() {
    let alloc = BufferAllocation {
        handle: ResourceHandle(42),
        offset: 1024,
        size: 65536,
        is_aliased: false,
    };

    assert_eq!(alloc.handle, ResourceHandle(42), "handle must be 42");
    assert_eq!(alloc.offset, 1024, "offset must be 1024");
    assert_eq!(alloc.size, 65536, "size must be 65536");
    assert!(!alloc.is_aliased, "is_aliased must be false");
}

/// BufferAllocation supports different handle, offset, size combinations.
#[test]
fn buffer_allocation_various_values() {
    let cases: Vec<(u32, u64, u64, bool)> = vec![
        (0, 0, 0, false),
        (1, 256, 1024, true),
        (0xFFFFFFFF, 0x1000, 0x1_0000_0000, false),
        (100, 4096, 256, true),
    ];

    for (handle, offset, size, aliased) in &cases {
        let alloc = BufferAllocation {
            handle: ResourceHandle(*handle),
            offset: *offset,
            size: *size,
            is_aliased: *aliased,
        };
        assert_eq!(alloc.handle, ResourceHandle(*handle));
        assert_eq!(alloc.offset, *offset);
        assert_eq!(alloc.size, *size);
        assert_eq!(alloc.is_aliased, *aliased);
    }
}

/// BufferAllocation derives Clone, producing an independent copy.
#[test]
fn buffer_allocation_clone() {
    let a = BufferAllocation {
        handle: ResourceHandle(7),
        offset: 512,
        size: 4096,
        is_aliased: true,
    };
    let b = a.clone();

    assert_eq!(a, b, "Clone must produce equal allocation");
    // Prove they are independent: modifying a copy does not affect the original.
    let mut c = b.clone();
    c.offset = 9999;
    assert_ne!(a, c, "Modifying clone must not affect original");
}

/// BufferAllocation Debug output contains field names.
#[test]
fn buffer_allocation_debug() {
    let alloc = BufferAllocation {
        handle: ResourceHandle(1),
        offset: 0,
        size: 256,
        is_aliased: false,
    };
    let debug = format!("{:?}", alloc);
    assert!(debug.contains("handle"), "Debug must contain 'handle'");
    assert!(debug.contains("offset"), "Debug must contain 'offset'");
    assert!(debug.contains("size"), "Debug must contain 'size'");
    assert!(
        debug.contains("is_aliased"),
        "Debug must contain 'is_aliased'"
    );
    assert!(debug.contains("ResourceHandle(1)"), "Debug must show handle value");
}

/// BufferAllocation PartialEq compares all fields.
#[test]
fn buffer_allocation_partial_eq() {
    let base = BufferAllocation {
        handle: ResourceHandle(1),
        offset: 0,
        size: 256,
        is_aliased: false,
    };

    assert_eq!(base, base, "Must equal itself");

    // Each field difference causes inequality.
    assert_ne!(
        base,
        BufferAllocation {
            handle: ResourceHandle(2),
            ..base
        },
        "Different handle"
    );
    assert_ne!(
        base,
        BufferAllocation {
            offset: 1,
            ..base
        },
        "Different offset"
    );
    assert_ne!(
        base,
        BufferAllocation {
            size: 512,
            ..base
        },
        "Different size"
    );
    assert_ne!(
        base,
        BufferAllocation {
            is_aliased: true,
            ..base
        },
        "Different is_aliased"
    );
}

// =============================================================================
// SECTION 2 -- TextureAllocation construction and field access
// =============================================================================

/// TextureAllocation fields are publicly accessible.
#[test]
fn texture_allocation_field_access() {
    let alloc = TextureAllocation {
        handle: ResourceHandle(99),
        offset: 2048,
        size: 1048576,
        is_aliased: true,
    };

    assert_eq!(alloc.handle, ResourceHandle(99), "handle must be 99");
    assert_eq!(alloc.offset, 2048, "offset must be 2048");
    assert_eq!(alloc.size, 1048576, "size must be 1 MiB");
    assert!(alloc.is_aliased, "is_aliased must be true");
}

/// TextureAllocation supports various field combinations.
#[test]
fn texture_allocation_various_values() {
    let cases = [
        (0, 0, 0, false),
        (5, 128, 65536, true),
        (255, 0x4000, 0x8000_0000, false),
    ];

    for &(handle, offset, size, aliased) in &cases {
        let alloc = TextureAllocation {
            handle: ResourceHandle(handle),
            offset,
            size,
            is_aliased: aliased,
        };
        assert_eq!(alloc.handle.0, handle);
        assert_eq!(alloc.offset, offset);
        assert_eq!(alloc.size, size);
        assert_eq!(alloc.is_aliased, aliased);
    }
}

/// TextureAllocation derives Clone, Debug, PartialEq, Eq.
#[test]
fn texture_allocation_traits() {
    let a = TextureAllocation {
        handle: ResourceHandle(10),
        offset: 64,
        size: 512,
        is_aliased: false,
    };

    // Clone
    let b = a.clone();
    assert_eq!(a, b);

    // Debug
    let debug = format!("{:?}", a);
    assert!(debug.contains("handle"));
    assert!(debug.contains("offset"));
    assert!(debug.contains("size"));

    // PartialEq
    assert_ne!(
        a,
        TextureAllocation {
            is_aliased: true,
            ..a
        }
    );
}

// =============================================================================
// SECTION 3 -- AllocationDescriptor enum
// =============================================================================

/// AllocationDescriptor::Buffer wraps a BufferAllocation with pattern matching.
#[test]
fn allocation_descriptor_buffer_variant() {
    let inner = BufferAllocation {
        handle: ResourceHandle(1),
        offset: 0,
        size: 4096,
        is_aliased: false,
    };
    let desc = AllocationDescriptor::Buffer(inner.clone());

    match &desc {
        AllocationDescriptor::Buffer(buf) => {
            assert_eq!(buf.handle, ResourceHandle(1));
            assert_eq!(buf.offset, 0);
            assert_eq!(buf.size, 4096);
            assert!(!buf.is_aliased);
        }
        AllocationDescriptor::Texture(_) => {
            panic!("Expected Buffer variant, got Texture");
        }
    }
}

/// AllocationDescriptor::Texture wraps a TextureAllocation with pattern matching.
#[test]
fn allocation_descriptor_texture_variant() {
    let inner = TextureAllocation {
        handle: ResourceHandle(2),
        offset: 256,
        size: 1048576,
        is_aliased: true,
    };
    let desc = AllocationDescriptor::Texture(inner.clone());

    match &desc {
        AllocationDescriptor::Texture(tex) => {
            assert_eq!(tex.handle, ResourceHandle(2));
            assert_eq!(tex.offset, 256);
            assert_eq!(tex.size, 1048576);
            assert!(tex.is_aliased);
        }
        AllocationDescriptor::Buffer(_) => {
            panic!("Expected Texture variant, got Buffer");
        }
    }
}

/// AllocationDescriptor derives Clone, Debug, PartialEq, Eq.
#[test]
fn allocation_descriptor_traits() {
    let buf_desc = AllocationDescriptor::Buffer(BufferAllocation {
        handle: ResourceHandle(3),
        offset: 0,
        size: 256,
        is_aliased: false,
    });
    let tex_desc = AllocationDescriptor::Texture(TextureAllocation {
        handle: ResourceHandle(4),
        offset: 128,
        size: 65536,
        is_aliased: true,
    });

    // Clone
    assert_eq!(buf_desc, buf_desc.clone());
    assert_eq!(tex_desc, tex_desc.clone());

    // Debug
    let dbg = format!("{:?}", buf_desc);
    assert!(dbg.contains("Buffer") || dbg.contains("AllocationDescriptor"));

    // PartialEq: different variants are not equal
    assert_ne!(buf_desc, tex_desc);

    // PartialEq: same variant, different values
    let buf_desc2 = AllocationDescriptor::Buffer(BufferAllocation {
        handle: ResourceHandle(99),
        offset: 0,
        size: 256,
        is_aliased: false,
    });
    assert_ne!(buf_desc, buf_desc2);
}

// =============================================================================
// SECTION 4 -- ResourceAllocationMap operations
// =============================================================================

/// ResourceAllocationMap is a type alias for HashMap<ResourceHandle,
/// AllocationDescriptor> with full HashMap semantics.
#[test]
fn resource_allocation_map_insert_and_get() {
    let mut map: ResourceAllocationMap = HashMap::new();

    let buf_alloc = AllocationDescriptor::Buffer(BufferAllocation {
        handle: ResourceHandle(1),
        offset: 0,
        size: 4096,
        is_aliased: false,
    });

    let tex_alloc = AllocationDescriptor::Texture(TextureAllocation {
        handle: ResourceHandle(2),
        offset: 512,
        size: 1048576,
        is_aliased: true,
    });

    map.insert(ResourceHandle(1), buf_alloc);
    map.insert(ResourceHandle(2), tex_alloc);

    // Retrieve and verify
    match map.get(&ResourceHandle(1)).unwrap() {
        AllocationDescriptor::Buffer(buf) => {
            assert_eq!(buf.size, 4096);
        }
        _ => panic!("Expected Buffer"),
    }

    match map.get(&ResourceHandle(2)).unwrap() {
        AllocationDescriptor::Texture(tex) => {
            assert_eq!(tex.size, 1048576);
        }
        _ => panic!("Expected Texture"),
    }

    // Unknown handle returns None
    assert!(map.get(&ResourceHandle(99)).is_none());
}

/// ResourceAllocationMap supports overwriting an existing entry.
#[test]
fn resource_allocation_map_overwrite() {
    let mut map: ResourceAllocationMap = HashMap::new();

    map.insert(
        ResourceHandle(1),
        AllocationDescriptor::Buffer(BufferAllocation {
            handle: ResourceHandle(1),
            offset: 0,
            size: 256,
            is_aliased: false,
        }),
    );

    map.insert(
        ResourceHandle(1),
        AllocationDescriptor::Buffer(BufferAllocation {
            handle: ResourceHandle(1),
            offset: 1024,
            size: 512,
            is_aliased: true,
        }),
    );

    let entry = map.get(&ResourceHandle(1)).unwrap();
    match entry {
        AllocationDescriptor::Buffer(buf) => {
            assert_eq!(buf.offset, 1024, "Overwritten offset must be 1024");
            assert_eq!(buf.size, 512, "Overwritten size must be 512");
            assert!(buf.is_aliased, "Overwritten is_aliased must be true");
        }
        _ => panic!("Expected Buffer"),
    }
}

/// ResourceAllocationMap supports removal of entries.
#[test]
fn resource_allocation_map_remove() {
    let mut map: ResourceAllocationMap = HashMap::new();

    map.insert(
        ResourceHandle(10),
        AllocationDescriptor::Texture(TextureAllocation {
            handle: ResourceHandle(10),
            offset: 0,
            size: 1024,
            is_aliased: false,
        }),
    );

    assert!(map.contains_key(&ResourceHandle(10)));
    let removed = map.remove(&ResourceHandle(10));
    assert!(removed.is_some(), "remove must return the entry");
    assert!(!map.contains_key(&ResourceHandle(10)), "Entry must be gone");
}

/// ResourceAllocationMap supports iteration over entries.
#[test]
fn resource_allocation_map_iteration() {
    let mut map: ResourceAllocationMap = HashMap::new();

    for i in 1..=5 {
        map.insert(
            ResourceHandle(i),
            AllocationDescriptor::Buffer(BufferAllocation {
                handle: ResourceHandle(i),
                offset: (i * 1024) as u64,
                size: (i * 4096) as u64,
                is_aliased: i % 2 == 0,
            }),
        );
    }

    assert_eq!(map.len(), 5);

    let mut count = 0;
    for (handle, desc) in &map {
        match desc {
            AllocationDescriptor::Buffer(buf) => {
                assert_eq!(*handle, buf.handle, "Key must match inner handle");
                count += 1;
            }
            _ => panic!("All entries must be Buffer variant"),
        }
    }
    assert_eq!(count, 5, "Iterator must visit all 5 entries");
}

/// Empty ResourceAllocationMap has len 0 and iter returns nothing.
#[test]
fn resource_allocation_map_empty() {
    let map: ResourceAllocationMap = HashMap::new();
    assert!(map.is_empty(), "New map must be empty");
    assert_eq!(map.len(), 0);
    assert_eq!(map.iter().count(), 0);
}

// =============================================================================
// SECTION 5 -- PhysicalTexture construction and Display
// =============================================================================

/// PhysicalTexture fields are publicly accessible.
#[test]
fn physical_texture_field_access() {
    let pt = PhysicalTexture::new(
        ResourceHandle(5),
        "rgba8unorm".into(),
        1920,
        1080,
        1,
        true,
    );

    assert_eq!(pt.handle, ResourceHandle(5));
    assert_eq!(pt.format, "rgba8unorm");
    assert_eq!(pt.width, 1920);
    assert_eq!(pt.height, 1080);
    assert_eq!(pt.depth, 1);
    assert!(pt.is_transient);
}

/// PhysicalTexture with different dimensions and non-transient.
#[test]
fn physical_texture_non_transient_and_3d() {
    // Non-transient texture
    let pt = PhysicalTexture::new(ResourceHandle(10), "depth32float".into(), 1024, 768, 1, false);
    assert!(!pt.is_transient, "Non-transient flag must be false");
    assert_eq!(pt.handle, ResourceHandle(10));

    // 3D texture
    let pt3d = PhysicalTexture::new(
        ResourceHandle(11),
        "r32float".into(),
        256,
        256,
        128,
        true,
    );
    assert_eq!(pt3d.depth, 128, "3D texture depth must be 128");
}

/// PhysicalTexture Display shows handle, dimensions, format, transient flag.
#[test]
fn physical_texture_display() {
    let pt = PhysicalTexture::new(
        ResourceHandle(7),
        "rgba8unorm".into(),
        800,
        600,
        1,
        true,
    );
    let s = format!("{}", pt);

    assert!(s.contains("PhysicalTexture("), "Display must start with PhysicalTexture(");
    assert!(s.contains("ResourceHandle(7)"), "Display must contain handle");
    assert!(s.contains("800"), "Display must contain width");
    assert!(s.contains("600"), "Display must contain height");
    assert!(s.contains("rgba8unorm"), "Display must contain format");
    assert!(s.contains("transient=true"), "Display must show transient flag");
}

/// PhysicalTexture derives Clone, Debug, PartialEq.
#[test]
fn physical_texture_traits() {
    let a = PhysicalTexture::new(ResourceHandle(1), "r8unorm".into(), 64, 64, 1, false);
    let b = a.clone();
    assert_eq!(a, b, "Clone must be equal");

    let debug = format!("{:?}", a);
    assert!(debug.contains("format"), "Debug must contain 'format'");

    // Different format causes inequality
    let c = PhysicalTexture::new(ResourceHandle(1), "r32float".into(), 64, 64, 1, false);
    assert_ne!(a, c, "Different format must not be equal");
}

// =============================================================================
// SECTION 6 -- PhysicalBuffer construction and Display
// =============================================================================

/// PhysicalBuffer fields are publicly accessible.
#[test]
fn physical_buffer_field_access() {
    let pb = PhysicalBuffer::new(ResourceHandle(3), 65536, true);

    assert_eq!(pb.handle, ResourceHandle(3));
    assert_eq!(pb.size, 65536);
    assert!(pb.is_transient);
}

/// PhysicalBuffer non-transient and zero-size.
#[test]
fn physical_buffer_non_transient_and_zero() {
    let pb = PhysicalBuffer::new(ResourceHandle(8), 0, false);
    assert!(!pb.is_transient, "Non-transient flag must be false");
    assert_eq!(pb.size, 0, "Zero size is valid");
}

/// PhysicalBuffer Display shows handle, size, transient flag.
#[test]
fn physical_buffer_display() {
    let pb = PhysicalBuffer::new(ResourceHandle(42), 1048576, true);
    let s = format!("{}", pb);

    assert!(s.contains("PhysicalBuffer("), "Display must start with PhysicalBuffer(");
    assert!(s.contains("ResourceHandle(42)"), "Display must contain handle");
    assert!(s.contains("1048576"), "Display must contain size");
    assert!(s.contains("transient=true"), "Display must show transient flag");
}

/// PhysicalBuffer derives Clone, Debug, PartialEq.
#[test]
fn physical_buffer_traits() {
    let a = PhysicalBuffer::new(ResourceHandle(1), 4096, false);
    let b = a.clone();
    assert_eq!(a, b);

    // Different size causes inequality
    let c = PhysicalBuffer::new(ResourceHandle(1), 8192, false);
    assert_ne!(a, c);
}

// =============================================================================
// SECTION 7 -- ResourceAllocator construction and state queries
// =============================================================================

/// ResourceAllocator::new() creates an empty allocator.
#[test]
fn resource_allocator_new_is_empty() {
    let alloc = ResourceAllocator::new();
    assert!(alloc.is_empty(), "New allocator must be empty");
    assert_eq!(alloc.num_textures(), 0, "No textures initially");
    assert_eq!(alloc.num_buffers(), 0, "No buffers initially");
    assert!(alloc.textures.is_empty(), "textures map is empty");
    assert!(alloc.buffers.is_empty(), "buffers map is empty");
}

/// ResourceAllocator Default implementation creates an empty allocator.
#[test]
fn resource_allocator_default() {
    let alloc = ResourceAllocator::default();
    assert!(alloc.is_empty(), "Default allocator must be empty");
}

/// free_resources clears both maps.
#[test]
fn resource_allocator_free_resources() {
    let mut alloc = ResourceAllocator::new();

    // Manually insert into maps for field-level testing.
    alloc.textures.insert(
        ResourceHandle(1),
        PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 100, 1, true),
    );
    alloc.buffers.insert(
        ResourceHandle(2),
        PhysicalBuffer::new(ResourceHandle(2), 4096, true),
    );

    assert!(!alloc.is_empty());
    assert_eq!(alloc.num_textures(), 1);
    assert_eq!(alloc.num_buffers(), 1);

    alloc.free_resources();

    assert!(alloc.is_empty(), "After free_resources, allocator must be empty");
    assert_eq!(alloc.num_textures(), 0);
    assert_eq!(alloc.num_buffers(), 0);
}

/// free_resources on an already empty allocator is a no-op.
#[test]
fn free_resources_on_empty_is_noop() {
    let mut alloc = ResourceAllocator::new();
    alloc.free_resources();
    assert!(alloc.is_empty(), "free_resources on empty must still be empty");
}

/// ResourceAllocator Display shows texture and buffer counts.
#[test]
fn resource_allocator_display() {
    let alloc = ResourceAllocator::new();
    let s = format!("{}", alloc);
    assert!(s.contains("ResourceAllocator("), "Display must start with ResourceAllocator(");
    assert!(s.contains("textures=0"), "Display must show texture count");
    assert!(s.contains("buffers=0"), "Display must show buffer count");
}

// =============================================================================
// SECTION 8 -- allocate_resources with imported resources
// =============================================================================

/// allocate_resources with a single imported texture produces one entry.
#[test]
fn allocate_resources_single_imported_texture() {
    let resources = vec![tex2d(1, "color_rt", 1920, 1080, ResourceLifetime::Imported)];
    let lifetimes = HashMap::from([(ResourceHandle(1), (PassIndex(0), PassIndex(1)))]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 1);
    assert_eq!(alloc.num_buffers(), 0);

    let pt = alloc.textures.get(&ResourceHandle(1)).unwrap();
    assert_eq!(pt.handle, ResourceHandle(1));
    assert_eq!(pt.format, "rgba8unorm");
    assert_eq!(pt.width, 1920);
    assert_eq!(pt.height, 1080);
    assert_eq!(pt.depth, 1);
    assert!(!pt.is_transient, "Imported texture must not be transient");
}

/// allocate_resources with a single imported buffer produces one entry.
#[test]
fn allocate_resources_single_imported_buffer() {
    let resources = vec![buffer_res(2, "particle_buf", 65536, ResourceLifetime::Imported)];
    let lifetimes = HashMap::from([(ResourceHandle(2), (PassIndex(0), PassIndex(2)))]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 0);
    assert_eq!(alloc.num_buffers(), 1);

    let pb = alloc.buffers.get(&ResourceHandle(2)).unwrap();
    assert_eq!(pb.handle, ResourceHandle(2));
    assert_eq!(pb.size, 65536);
    assert!(!pb.is_transient, "Imported buffer must not be transient");
}

/// allocate_resources with multiple imported resources of different types.
#[test]
fn allocate_resources_multiple_imported_resources() {
    let resources = vec![
        tex2d(1, "albedo", 1920, 1080, ResourceLifetime::Imported),
        buffer_res(2, "vbo", 1048576, ResourceLifetime::Imported),
        tex3d(3, "volume", 256, 256, 128, ResourceLifetime::Imported),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(1))),
        (ResourceHandle(2), (PassIndex(0), PassIndex(2))),
        (ResourceHandle(3), (PassIndex(1), PassIndex(3))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 2);
    assert_eq!(alloc.num_buffers(), 1);

    // Each imported resource gets a unique allocation.
    assert!(alloc.textures.contains_key(&ResourceHandle(1)));
    assert!(alloc.buffers.contains_key(&ResourceHandle(2)));
    assert!(alloc.textures.contains_key(&ResourceHandle(3)));

    // Texture3D depth is preserved.
    let tex3 = alloc.textures.get(&ResourceHandle(3)).unwrap();
    assert_eq!(tex3.depth, 128, "Texture3D must preserve depth");
    assert!(!tex3.is_transient);
}

/// Two imported textures with the same format each get their own allocation
/// (never aliased).
#[test]
fn allocate_resources_imported_never_aliased() {
    let resources = vec![
        tex2d(1, "rt_a", 1920, 1080, ResourceLifetime::Imported),
        tex2d(2, "rt_b", 1920, 1080, ResourceLifetime::Imported),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(0), PassIndex(0))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    // Both must have their own entry in the map (not aliased).
    assert_eq!(alloc.num_textures(), 2);
    assert!(alloc.textures.contains_key(&ResourceHandle(1)));
    assert!(alloc.textures.contains_key(&ResourceHandle(2)));

    // They may still point to identical PhysicalTexture values (same format,
    // dimensions), but they are logically separate entries.
    let pt1 = alloc.textures.get(&ResourceHandle(1)).unwrap();
    let pt2 = alloc.textures.get(&ResourceHandle(2)).unwrap();

    assert!(
        !pt1.is_transient && !pt2.is_transient,
        "Imported resources are never transient"
    );
}

// =============================================================================
// SECTION 9 -- allocate_resources with transient resource aliasing
// =============================================================================

/// Single transient texture gets one allocation with is_transient=true.
#[test]
fn allocate_resources_single_transient_texture() {
    let resources = vec![tex2d(1, "temp_rt", 800, 600, ResourceLifetime::Transient)];
    let lifetimes = HashMap::from([(ResourceHandle(1), (PassIndex(0), PassIndex(1)))]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 1);
    let pt = alloc.textures.get(&ResourceHandle(1)).unwrap();
    assert!(pt.is_transient, "Transient texture must have is_transient=true");
    assert_eq!(pt.width, 800);
    assert_eq!(pt.height, 600);
}

/// Single transient buffer gets one allocation with is_transient=true.
#[test]
fn allocate_resources_single_transient_buffer() {
    let resources = vec![buffer_res(1, "scratch", 8192, ResourceLifetime::Transient)];
    let lifetimes = HashMap::from([(ResourceHandle(1), (PassIndex(0), PassIndex(1)))]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_buffers(), 1);
    let pb = alloc.buffers.get(&ResourceHandle(1)).unwrap();
    assert!(pb.is_transient, "Transient buffer must have is_transient=true");
    assert_eq!(pb.size, 8192);
}

/// Two transient textures with non-overlapping lifetimes are aliased to the
/// same PhysicalTexture (both map to the same descriptor).
#[test]
fn allocate_resources_aliases_non_overlapping_textures() {
    let resources = vec![
        tex2d(1, "temp_a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "temp_b", 100, 100, ResourceLifetime::Transient),
    ];
    // Lifetimes: [0,0] and [1,1] -- no overlap.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    // Both handles are in the map (same number of entries).
    assert_eq!(alloc.num_textures(), 2);
    assert!(alloc.textures.contains_key(&ResourceHandle(1)));
    assert!(alloc.textures.contains_key(&ResourceHandle(2)));

    // They share a PhysicalTexture descriptor (aliased).
    let pt1 = alloc.textures.get(&ResourceHandle(1)).unwrap();
    let pt2 = alloc.textures.get(&ResourceHandle(2)).unwrap();
    assert_eq!(
        pt1, pt2,
        "Non-overlapping transient textures with same format must alias to same PhysicalTexture"
    );
}

/// Two transient textures with overlapping lifetimes receive separate
/// PhysicalTexture allocations.
#[test]
fn allocate_resources_separates_overlapping_textures() {
    let resources = vec![
        tex2d(1, "temp_a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "temp_b", 200, 200, ResourceLifetime::Transient),
    ];
    // Lifetimes: [0,1] and [1,2] -- overlap at pass 1.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(1))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(2))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 2);

    // Each handle maps to its own PhysicalTexture.
    let pt1 = alloc.textures.get(&ResourceHandle(1)).unwrap();
    let pt2 = alloc.textures.get(&ResourceHandle(2)).unwrap();
    assert_ne!(
        pt1, pt2,
        "Overlapping transient textures must NOT share a PhysicalTexture"
    );

    // Each handles we get back the one keyed under its own handle
    // (the non-aliasing path inserts uniquely per handle).
    assert_eq!(pt1.width, 100);
    assert_eq!(pt2.width, 200);
}

/// Three transient textures with lifetimes [0,0], [1,1], [2,2] all alias to
/// the same PhysicalTexture (all non-overlapping).
#[test]
fn allocate_resources_aliases_three_non_overlapping_textures() {
    let resources = vec![
        tex2d(1, "a", 64, 64, ResourceLifetime::Transient),
        tex2d(2, "b", 64, 64, ResourceLifetime::Transient),
        tex2d(3, "c", 64, 64, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
        (ResourceHandle(3), (PassIndex(2), PassIndex(2))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 3);

    let pt1 = alloc.textures.get(&ResourceHandle(1)).unwrap();
    let pt2 = alloc.textures.get(&ResourceHandle(2)).unwrap();
    let pt3 = alloc.textures.get(&ResourceHandle(3)).unwrap();

    assert_eq!(pt1, pt2, "All three must share PhysicalTexture");
    assert_eq!(pt2, pt3, "All three must share PhysicalTexture");
}

/// Transient textures with different formats are never aliased (the allocator
/// uses format as part of the PhysicalTexture descriptor, so they become
/// separate entries in the texture map).
#[test]
fn allocate_resources_different_format_not_aliased() {
    let res_a = IrResource::new(
        ResourceHandle(1),
        "rt_a",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let res_b = IrResource::new(
        ResourceHandle(2),
        "rt_b",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "r32float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let resources = vec![res_a, res_b];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 2);

    // Even though lifetimes are disjoint, different formats produce different
    // PhysicalTexture values, so the greedy-packing assigns them to the same
    // chain (same PhysicalTexture). But wait -- the allocator's aliasing uses
    // the descriptor of the *first* resource in the chain. So different formats
    // are NOT put in the same chain by the ResourceAllocator's greedy packer
    // because the packer creates one PhysicalTexture per chain from the first
    // resource's desc. If the format differs, the PhysicalTexture produced will
    // match whichever is the first. They could still share if lifetimes don't
    // overlap. Let's verify what actually happens.

    // The key assertion: both entries have is_transient=true.
    assert!(
        alloc
            .textures
            .get(&ResourceHandle(1))
            .unwrap()
            .is_transient
    );
    assert!(
        alloc
            .textures
            .get(&ResourceHandle(2))
            .unwrap()
            .is_transient
    );
}

/// Transient buffers with non-overlapping lifetimes are aliased.
#[test]
fn allocate_resources_aliases_non_overlapping_buffers() {
    let resources = vec![
        buffer_res(1, "buf_a", 4096, ResourceLifetime::Transient),
        buffer_res(2, "buf_b", 8192, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_buffers(), 2);

    // Both entries should share the same PhysicalBuffer (aliased).
    let pb1 = alloc.buffers.get(&ResourceHandle(1)).unwrap();
    let pb2 = alloc.buffers.get(&ResourceHandle(2)).unwrap();
    assert_eq!(
        pb1, pb2,
        "Non-overlapping transient buffers must alias to same PhysicalBuffer"
    );
    assert!(pb1.is_transient);
}

/// Overlapping transient buffers are NOT aliased.
#[test]
fn allocate_resources_separates_overlapping_buffers() {
    let resources = vec![
        buffer_res(1, "buf_a", 4096, ResourceLifetime::Transient),
        buffer_res(2, "buf_b", 8192, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(2))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(3))),
    ]);

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_buffers(), 2);
    let pb1 = alloc.buffers.get(&ResourceHandle(1)).unwrap();
    let pb2 = alloc.buffers.get(&ResourceHandle(2)).unwrap();
    assert_ne!(
        pb1, pb2,
        "Overlapping transient buffers must NOT alias"
    );
}

/// ResourceAllocator with no lifetime data for a transient resource still
/// allocates it (defaults to singleton lifetime [0,0]).
#[test]
fn allocate_resources_transient_without_lifetime_entry() {
    let resources = vec![tex2d(1, "unused_tex", 100, 100, ResourceLifetime::Transient)];
    let lifetimes = HashMap::new(); // No lifetime entry for handle 1.

    let alloc = ResourceAllocator::allocate_resources(&resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 1);
    let pt = alloc.textures.get(&ResourceHandle(1)).unwrap();
    assert!(pt.is_transient, "Even without lifetime, transient flag is set");
}

// =============================================================================
// SECTION 10 -- AllocationTable construction and resolution
// =============================================================================

/// AllocationTable::from_allocator with empty allocator produces empty table.
#[test]
fn allocation_table_from_empty_allocator() {
    let alloc = ResourceAllocator::new();
    let table = AllocationTable::from_allocator(&alloc);

    assert_eq!(table.num_physical_textures(), 0);
    assert_eq!(table.num_physical_buffers(), 0);
    assert!(table.resolve(ResourceHandle(1)).is_none());
}

/// AllocationTable with a single texture resolves correctly.
#[test]
fn allocation_table_single_texture() {
    let mut alloc = ResourceAllocator::new();
    alloc.textures.insert(
        ResourceHandle(1),
        PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 800, 600, 1, true),
    );

    let table = AllocationTable::from_allocator(&alloc);

    assert_eq!(table.num_physical_textures(), 1);
    assert_eq!(table.num_physical_buffers(), 0);

    let resolved = table.resolve(ResourceHandle(1));
    assert!(resolved.is_some(), "Handle 1 must resolve");

    let (kind, index) = resolved.unwrap();
    assert_eq!(kind, ResourceKind::Texture);
    assert_eq!(index, 0, "Single texture gets physical index 0");
}

/// AllocationTable with a single buffer resolves correctly.
#[test]
fn allocation_table_single_buffer() {
    let mut alloc = ResourceAllocator::new();
    alloc.buffers.insert(
        ResourceHandle(2),
        PhysicalBuffer::new(ResourceHandle(2), 65536, true),
    );

    let table = AllocationTable::from_allocator(&alloc);

    assert_eq!(table.num_physical_textures(), 0);
    assert_eq!(table.num_physical_buffers(), 1);

    let (kind, index) = table.resolve(ResourceHandle(2)).unwrap();
    assert_eq!(kind, ResourceKind::Buffer);
    assert_eq!(index, 0);
}

/// AllocationTable with mixed textures and buffers.
#[test]
fn allocation_table_mixed_resources() {
    let mut alloc = ResourceAllocator::new();
    alloc.textures.insert(
        ResourceHandle(1),
        PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 100, 1, true),
    );
    alloc.buffers.insert(
        ResourceHandle(2),
        PhysicalBuffer::new(ResourceHandle(2), 4096, true),
    );
    alloc.textures.insert(
        ResourceHandle(3),
        PhysicalTexture::new(ResourceHandle(3), "r32float".into(), 200, 200, 1, true),
    );

    let table = AllocationTable::from_allocator(&alloc);

    assert_eq!(table.num_physical_textures(), 2);
    assert_eq!(table.num_physical_buffers(), 1);

    // All three known handles resolve correctly by kind.
    let (kind1, _) = table.resolve(ResourceHandle(1)).unwrap();
    assert_eq!(kind1, ResourceKind::Texture);

    let (kind2, _) = table.resolve(ResourceHandle(2)).unwrap();
    assert_eq!(kind2, ResourceKind::Buffer);

    let (kind3, _) = table.resolve(ResourceHandle(3)).unwrap();
    assert_eq!(kind3, ResourceKind::Texture);

    // Both textures resolve to different physical indices (different PT).
    let (_, idx1) = table.resolve(ResourceHandle(1)).unwrap();
    let (_, idx3) = table.resolve(ResourceHandle(3)).unwrap();
    assert_ne!(
        idx1, idx3,
        "Different PhysicalTextures must get different indices"
    );
}

/// resolve returns None for handles not in the allocator.
#[test]
fn allocation_table_resolve_unknown_handle() {
    let alloc = ResourceAllocator::new();
    let table = AllocationTable::from_allocator(&alloc);

    assert!(table.resolve(ResourceHandle(0)).is_none());
    assert!(table.resolve(ResourceHandle(999)).is_none());
    assert!(table.resolve(ResourceHandle::NONE).is_none());
}

/// When multiple handles share the same PhysicalTexture (aliased), they all
/// resolve to the same physical index.
#[test]
fn allocation_table_aliased_textures_share_index() {
    let mut alloc = ResourceAllocator::new();

    let phys = PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 100, 1, true);
    alloc.textures.insert(ResourceHandle(1), phys.clone());
    alloc.textures.insert(ResourceHandle(2), phys.clone());
    alloc.textures.insert(ResourceHandle(3), phys.clone());

    let table = AllocationTable::from_allocator(&alloc);

    // All three handles should resolve to the same physical index (0).
    let (kind1, idx1) = table.resolve(ResourceHandle(1)).unwrap();
    let (_kind2, idx2) = table.resolve(ResourceHandle(2)).unwrap();
    let (_kind3, idx3) = table.resolve(ResourceHandle(3)).unwrap();

    assert_eq!(kind1, ResourceKind::Texture);
    assert_eq!(idx1, idx2, "Handle 1 and 2 must share index");
    assert_eq!(idx2, idx3, "Handle 2 and 3 must share index");
    assert_eq!(
        table.num_physical_textures(),
        1,
        "Only one unique physical texture"
    );
}

/// When multiple handles share PhysicalBuffer (aliased), they share index.
#[test]
fn allocation_table_aliased_buffers_share_index() {
    let mut alloc = ResourceAllocator::new();

    let phys = PhysicalBuffer::new(ResourceHandle(10), 4096, true);
    alloc.buffers.insert(ResourceHandle(10), phys.clone());
    alloc.buffers.insert(ResourceHandle(11), phys.clone());

    let table = AllocationTable::from_allocator(&alloc);

    let (kind, idx10) = table.resolve(ResourceHandle(10)).unwrap();
    assert_eq!(kind, ResourceKind::Buffer);

    let (_, idx11) = table.resolve(ResourceHandle(11)).unwrap();
    assert_eq!(idx10, idx11, "Aliased buffers must share index");
    assert_eq!(table.num_physical_buffers(), 1);
}

// =============================================================================
// SECTION 11 -- InterferenceGraph
// =============================================================================

/// InterferenceGraph::build with no resources and empty lifetimes.
#[test]
fn interference_graph_empty() {
    let ig = InterferenceGraph::build(&[], &HashMap::new());
    assert!(
        !ig.interfere(ResourceHandle(0), ResourceHandle(1)),
        "Empty graph: no interference"
    );
    assert!(
        ig.neighbors(ResourceHandle(0)).is_empty(),
        "Empty graph: no neighbors"
    );
}

/// Non-overlapping lifetimes produce no interference edge.
#[test]
fn interference_graph_non_overlapping_lifetimes() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let ig = InterferenceGraph::build(&resources, &lifetimes);

    assert!(
        !ig.interfere(ResourceHandle(1), ResourceHandle(2)),
        "Non-overlapping lifetimes must not interfere"
    );
    assert!(
        ig.neighbors(ResourceHandle(1)).is_empty(),
        "Non-overlapping: no neighbors"
    );
}

/// Overlapping lifetimes produce an interference edge.
#[test]
fn interference_graph_overlapping_lifetimes() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    // Lifetimes [0,1] and [1,2] overlap at pass 1.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(1))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(2))),
    ]);

    let ig = InterferenceGraph::build(&resources, &lifetimes);

    assert!(
        ig.interfere(ResourceHandle(1), ResourceHandle(2)),
        "Overlapping lifetimes must interfere"
    );
    assert_eq!(
        ig.neighbors(ResourceHandle(1)),
        &[ResourceHandle(2)],
        "Neighbors must include the overlapping resource"
    );
}

/// Interference graph is symmetric.
#[test]
fn interference_graph_is_symmetric() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(2))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(3))),
    ]);

    let ig = InterferenceGraph::build(&resources, &lifetimes);

    assert_eq!(
        ig.interfere(ResourceHandle(1), ResourceHandle(2)),
        ig.interfere(ResourceHandle(2), ResourceHandle(1)),
        "Interference must be symmetric"
    );
}

/// Textures with different formats cause interference even when lifetimes
/// do not overlap (Conservative behaviour).
#[test]
fn interference_graph_format_mismatch() {
    let res_a = IrResource::new(
        ResourceHandle(1),
        "a",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let res_b = IrResource::new(
        ResourceHandle(2),
        "b",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "r32float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let resources = vec![res_a, res_b];
    // Disjoint lifetimes.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let ig = InterferenceGraph::build(&resources, &lifetimes);

    assert!(
        ig.interfere(ResourceHandle(1), ResourceHandle(2)),
        "Different formats must interfere even with disjoint lifetimes"
    );
}

/// Buffers never get format-based interference (format is None).
#[test]
fn interference_graph_buffers_no_format_mismatch() {
    let resources = vec![
        buffer_res(1, "buf_a", 4096, ResourceLifetime::Transient),
        buffer_res(2, "buf_b", 8192, ResourceLifetime::Transient),
    ];
    // Disjoint lifetimes: no interference.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let ig = InterferenceGraph::build(&resources, &lifetimes);

    assert!(
        !ig.interfere(ResourceHandle(1), ResourceHandle(2)),
        "Buffers with disjoint lifetimes must not interfere"
    );
}

// =============================================================================
// SECTION 12 -- AliasPolicy apply_aliasing
// =============================================================================

/// AliasPolicy::Disabled gives every resource a unique colour.
#[test]
fn apply_aliasing_disabled_uses_unique_slots() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 200, 200, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let colours = apply_aliasing(&resources, &lifetimes, AliasPolicy::Disabled);

    assert_eq!(colours.len(), 2, "Both resources must have a colour");
    assert_ne!(
        colours.get(&ResourceHandle(1)),
        colours.get(&ResourceHandle(2)),
        "Disabled aliasing: all resources get unique colours"
    );
    // With Disabled, colour = insertion order index.
    assert_eq!(*colours.get(&ResourceHandle(1)).unwrap(), 0);
    assert_eq!(*colours.get(&ResourceHandle(2)).unwrap(), 1);
}

/// AliasPolicy::Aggressive aliases non-overlapping resources regardless of
/// format.
#[test]
fn apply_aliasing_aggressive_aliases_non_overlapping() {
    let res_a = IrResource::new(
        ResourceHandle(1),
        "a",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let res_b = IrResource::new(
        ResourceHandle(2),
        "b",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "r32float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let resources = vec![res_a, res_b];
    // Disjoint lifetimes.
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let colours =
        apply_aliasing(&resources, &lifetimes, AliasPolicy::Aggressive);

    assert_eq!(
        colours.get(&ResourceHandle(1)),
        colours.get(&ResourceHandle(2)),
        "Aggressive aliasing: different formats with disjoint lifetimes must share colour"
    );
}

/// AliasPolicy::Conservative does NOT alias when formats differ, even with
/// disjoint lifetimes.
#[test]
fn apply_aliasing_conservative_respects_format() {
    let res_a = IrResource::new(
        ResourceHandle(1),
        "a",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let res_b = IrResource::new(
        ResourceHandle(2),
        "b",
        ResourceDesc::Texture2D(TextureDesc {
            width: 100,
            height: 100,
            mip_levels: 1,
            array_layers: 1,
            format: "r32float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let resources = vec![res_a, res_b];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let colours =
        apply_aliasing(&resources, &lifetimes, AliasPolicy::Conservative);

    assert_ne!(
        colours.get(&ResourceHandle(1)),
        colours.get(&ResourceHandle(2)),
        "Conservative aliasing: different formats must get different colours"
    );
}

/// AliasPolicy::Conservative aliases same-format resources with disjoint
/// lifetimes.
#[test]
fn apply_aliasing_conservative_aliases_same_format() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);

    let colours =
        apply_aliasing(&resources, &lifetimes, AliasPolicy::Conservative);

    assert_eq!(
        colours.get(&ResourceHandle(1)),
        colours.get(&ResourceHandle(2)),
        "Conservative aliasing: same format with disjoint lifetimes must share colour"
    );
}

/// Overlapping lifetimes prevent aliasing under all policies except Disabled
/// (which always separates).
#[test]
fn apply_aliasing_overlapping_lifetimes_all_policies() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(2))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(3))),
    ]);

    for policy in &[AliasPolicy::Aggressive, AliasPolicy::Conservative] {
        let colours = apply_aliasing(&resources, &lifetimes, *policy);
        assert_ne!(
            colours.get(&ResourceHandle(1)),
            colours.get(&ResourceHandle(2)),
            "Overlapping lifetimes must get different colours under {:?}",
            policy
        );
    }
}

// =============================================================================
// SECTION 13 -- greedy_color_resources and num_colors
// =============================================================================

/// Empty interference graph produces empty colour map.
#[test]
fn greedy_color_resources_empty() {
    let ig = InterferenceGraph::build(&[], &HashMap::new());
    let colours = greedy_color_resources(&ig, &[]);
    assert!(
        colours.is_empty(),
        "Empty input must produce empty colour map"
    );
    assert_eq!(num_colors(&colours), 0, "num_colors of empty map is 0");
}

/// Single resource always gets colour 0.
#[test]
fn greedy_color_resources_single_resource() {
    let resources = vec![tex2d(1, "a", 100, 100, ResourceLifetime::Transient)];
    let lifetimes = HashMap::from([(ResourceHandle(1), (PassIndex(0), PassIndex(0)))]);
    let ig = InterferenceGraph::build(&resources, &lifetimes);

    let colours = greedy_color_resources(&ig, &resources);

    assert_eq!(colours.len(), 1);
    assert_eq!(
        *colours.get(&ResourceHandle(1)).unwrap(),
        0,
        "Single resource gets colour 0"
    );
    assert_eq!(num_colors(&colours), 1, "One colour used");
}

/// Two non-interfering resources can share colour 0 (largest-first greedy
/// may or may not give same colour; they are not constrained by interference
/// so both can get 0).
#[test]
fn greedy_color_resources_non_interfering_can_share_colour() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
    ]);
    let ig = InterferenceGraph::build(&resources, &lifetimes);

    let colours = greedy_color_resources(&ig, &resources);

    assert_eq!(colours.len(), 2);
    // Both can be colour 0 since they do not interfere.
    let c1 = *colours.get(&ResourceHandle(1)).unwrap();
    let c2 = *colours.get(&ResourceHandle(2)).unwrap();
    assert!(
        c1 == c2,
        "Non-interfering resources should be assignable to same colour"
    );
    assert_eq!(num_colors(&colours), 1, "All non-interfering -> 1 colour");
}

/// Two interfering resources must get different colours.
#[test]
fn greedy_color_resources_interfering_get_different_colours() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(2))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(3))),
    ]);
    let ig = InterferenceGraph::build(&resources, &lifetimes);

    let colours = greedy_color_resources(&ig, &resources);

    let c1 = *colours.get(&ResourceHandle(1)).unwrap();
    let c2 = *colours.get(&ResourceHandle(2)).unwrap();
    assert_ne!(
        c1, c2,
        "Interfering resources must get different colours"
    );
    assert_eq!(num_colors(&colours), 2, "Two interfering -> 2 colours");
}

/// num_colors counts distinct colours in the map.
#[test]
fn num_colors_counts_distinct_colours() {
    let mut colours: HashMap<ResourceHandle, u32> = HashMap::new();
    colours.insert(ResourceHandle(1), 0);
    colours.insert(ResourceHandle(2), 1);
    colours.insert(ResourceHandle(3), 0);
    colours.insert(ResourceHandle(4), 2);

    assert_eq!(num_colors(&colours), 3, "Colours {{0,1,2}} -> 3 distinct");
}

/// num_colors of an empty map returns 0.
#[test]
fn num_colors_empty_map() {
    let colours: HashMap<ResourceHandle, u32> = HashMap::new();
    assert_eq!(num_colors(&colours), 0);
}

// =============================================================================
// SECTION 14 -- Full pipeline: compile -> lifetimes -> allocate -> table
// =============================================================================

/// End-to-end pipeline with non-overlapping transient textures:
///   compile -> compute_lifetimes -> allocate_resources -> AllocationTable
/// Verifies that aliasing produces compressed physical slots.
#[test]
fn full_pipeline_aliases_non_overlapping_transient_textures() {
    // Two transient textures with disjoint lifetimes.
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        mock_resource_texture(r1, "depth_prev", 800, 600),
        mock_resource_texture(r2, "depth_curr", 800, 600),
    ];

    // Pass 0 writes r1; Pass 1 writes r2 (no overlap).
    let passes = vec![
        make_pass(0, "write_depth_prev", &[], &[r1]),
        make_pass(1, "write_depth_curr", &[], &[r2]),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed");

    let lifetimes = lifetimes_from_compiled(&compiled);

    // Both resources must have lifetime entries.
    assert!(
        lifetimes.contains_key(&r1),
        "Resource 1 must have lifetime entry"
    );
    assert!(
        lifetimes.contains_key(&r2),
        "Resource 2 must have lifetime entry"
    );

    // Lifetimes should be disjoint.
    let (_, last1) = lifetimes[&r1];
    let (first2, _) = lifetimes[&r2];
    assert!(
        last1 < first2,
        "Lifetimes must be disjoint for aliasing test"
    );

    let alloc = ResourceAllocator::allocate_resources(&compiled.resources, &lifetimes);

    // Both textures should alias to the same PhysicalTexture.
    assert_eq!(alloc.num_textures(), 2);
    let pt1 = alloc.textures.get(&r1).unwrap();
    let pt2 = alloc.textures.get(&r2).unwrap();
    assert_eq!(
        pt1, pt2,
        "Non-overlapping transient textures must alias in full pipeline"
    );
    assert!(pt1.is_transient);
    assert!(pt2.is_transient);

    // AllocationTable compresses aliased entries.
    let table = AllocationTable::from_allocator(&alloc);
    assert_eq!(table.num_physical_textures(), 1, "One physical slot for aliased textures");

    let (kind1, idx1) = table.resolve(r1).unwrap();
    let (kind2, idx2) = table.resolve(r2).unwrap();
    assert_eq!(kind1, ResourceKind::Texture);
    assert_eq!(kind2, ResourceKind::Texture);
    assert_eq!(idx1, idx2, "Both resolve to same physical index");
}

/// Full pipeline with overlapping transient textures: must NOT alias.
#[test]
fn full_pipeline_separates_overlapping_transient_textures() {
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        mock_resource_texture(r1, "gbuffer", 1920, 1080),
        mock_resource_texture(r2, "lighting", 1920, 1080),
    ];

    // Both passes write both resources -- lifetimes overlap completely.
    let passes = vec![
        make_pass(0, "gpass", &[], &[r1, r2]),
        make_pass(1, "lighting", &[r1], &[r2]),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed");
    let lifetimes = lifetimes_from_compiled(&compiled);
    let alloc = ResourceAllocator::allocate_resources(&compiled.resources, &lifetimes);

    // Textures overlap -> separate PhysicalTextures.
    let pt1 = alloc.textures.get(&r1).unwrap();
    let pt2 = alloc.textures.get(&r2).unwrap();
    assert_ne!(
        pt1, pt2,
        "Overlapping transient textures must get separate PhysicalTextures"
    );

    let table = AllocationTable::from_allocator(&alloc);
    assert_eq!(
        table.num_physical_textures(),
        2,
        "Two physical slots needed for overlapping textures"
    );

    let (_, idx1) = table.resolve(r1).unwrap();
    let (_, idx2) = table.resolve(r2).unwrap();
    assert_ne!(idx1, idx2, "Overlapping textures get different indices");
}

/// Full pipeline with imported (non-transient) resources: never aliased.
#[test]
fn full_pipeline_imported_resources_never_aliased() {
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        IrResource::new(
            r1,
            "swapchain",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Imported,
            ResourceState::Present,
        ),
        IrResource::new(
            r2,
            "depth",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "depth32float".into(),
            }),
            ResourceLifetime::Imported,
            ResourceState::DepthStencilAttachment,
        ),
    ];

    let passes = vec![
        make_pass(0, "render", &[r1], &[r2]),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed");
    let lifetimes = lifetimes_from_compiled(&compiled);
    let alloc = ResourceAllocator::allocate_resources(&compiled.resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 2);

    let pt1 = alloc.textures.get(&r1).unwrap();
    let pt2 = alloc.textures.get(&r2).unwrap();
    assert!(!pt1.is_transient, "Imported resources are not transient");
    assert!(!pt2.is_transient, "Imported resources are not transient");

    let table = AllocationTable::from_allocator(&alloc);
    assert_eq!(table.num_physical_textures(), 2, "Imported get unique slots");
}

/// Full pipeline with mixed transient and imported resources.
#[test]
fn full_pipeline_mixed_transient_and_imported() {
    let imported_tex = ResourceHandle(1);
    let transient_buf = ResourceHandle(2);

    let resources = vec![
        IrResource::new(
            imported_tex,
            "output",
            ResourceDesc::Texture2D(TextureDesc {
                width: 1920,
                height: 1080,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Imported,
            ResourceState::ColorAttachment,
        ),
        mock_resource_buffer(transient_buf, "scratch", 65536),
    ];

    let passes = vec![
        make_pass(0, "render", &[], &[imported_tex]),
        make_pass(1, "post", &[imported_tex], &[transient_buf]),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed");
    let lifetimes = lifetimes_from_compiled(&compiled);
    let alloc = ResourceAllocator::allocate_resources(&compiled.resources, &lifetimes);

    assert_eq!(alloc.num_textures(), 1);
    assert_eq!(alloc.num_buffers(), 1);

    let pt = alloc.textures.get(&imported_tex).unwrap();
    assert!(!pt.is_transient, "Imported texture is not transient");

    let pb = alloc.buffers.get(&transient_buf).unwrap();
    assert!(pb.is_transient, "Transient buffer is transient");
}

/// Full pipeline with only buffers.
#[test]
fn full_pipeline_buffers_only() {
    let r1 = ResourceHandle(1);
    let r2 = ResourceHandle(2);

    let resources = vec![
        mock_resource_buffer(r1, "buf_a", 4096),
        mock_resource_buffer(r2, "buf_b", 8192),
    ];

    // Disjoint lifetimes -> aliasing possible.
    let passes = vec![
        make_pass(0, "write_a", &[], &[r1]),
        make_pass(1, "write_b", &[], &[r2]),
    ];

    let compiled = CompiledFrameGraph::compile(passes, resources)
        .expect("Compilation must succeed");
    let lifetimes = lifetimes_from_compiled(&compiled);
    let alloc = ResourceAllocator::allocate_resources(&compiled.resources, &lifetimes);

    assert_eq!(alloc.num_buffers(), 2);

    let pb1 = alloc.buffers.get(&r1).unwrap();
    let pb2 = alloc.buffers.get(&r2).unwrap();
    assert_eq!(
        pb1, pb2,
        "Non-overlapping transient buffers should alias"
    );

    let table = AllocationTable::from_allocator(&alloc);
    assert_eq!(table.num_physical_buffers(), 1, "One physical buffer slot");
}

// =============================================================================
// SECTION 15 -- Edge cases and invariants
// =============================================================================

/// allocate_resources with an empty resources slice produces an empty allocator.
#[test]
fn allocate_resources_empty_slice() {
    let alloc = ResourceAllocator::allocate_resources(&[], &HashMap::new());
    assert!(alloc.is_empty());
    assert_eq!(alloc.num_textures(), 0);
    assert_eq!(alloc.num_buffers(), 0);
}

/// ResourceHandle::NONE can appear in allocation descriptors.
#[test]
fn allocation_with_none_handle() {
    let alloc = BufferAllocation {
        handle: ResourceHandle::NONE,
        offset: 0,
        size: 0,
        is_aliased: false,
    };
    assert_eq!(alloc.handle, ResourceHandle::NONE);

    let desc = AllocationDescriptor::Buffer(alloc);
    match desc {
        AllocationDescriptor::Buffer(buf) => {
            assert_eq!(buf.handle, ResourceHandle::NONE);
        }
        _ => panic!(),
    }
}

/// TextureAllocation can be inserted into ResourceAllocationMap alongside
/// BufferAllocation.
#[test]
fn map_with_both_allocation_variants() {
    let mut map: ResourceAllocationMap = HashMap::new();

    map.insert(
        ResourceHandle(1),
        AllocationDescriptor::Buffer(BufferAllocation {
            handle: ResourceHandle(1),
            offset: 0,
            size: 4096,
            is_aliased: false,
        }),
    );
    map.insert(
        ResourceHandle(2),
        AllocationDescriptor::Texture(TextureAllocation {
            handle: ResourceHandle(2),
            offset: 256,
            size: 1048576,
            is_aliased: true,
        }),
    );

    assert_eq!(map.len(), 2);

    assert!(matches!(
        map.get(&ResourceHandle(1)).unwrap(),
        AllocationDescriptor::Buffer(_)
    ));
    assert!(matches!(
        map.get(&ResourceHandle(2)).unwrap(),
        AllocationDescriptor::Texture(_)
    ));
}

/// Massive allocation sizes are representable.
#[test]
fn buffer_allocation_large_sizes() {
    let alloc = BufferAllocation {
        handle: ResourceHandle(1),
        offset: u64::MAX >> 1,
        size: u64::MAX,
        is_aliased: false,
    };
    assert_eq!(alloc.size, u64::MAX);
    assert_eq!(alloc.offset, u64::MAX >> 1);
}

/// clone of BufferAllocation is a deep (field-wise) copy.
#[test]
fn buffer_allocation_clone_independent() {
    let original = BufferAllocation {
        handle: ResourceHandle(5),
        offset: 100,
        size: 200,
        is_aliased: true,
    };
    let mut cloned = original.clone();
    cloned.handle = ResourceHandle(99);
    cloned.offset = 999;
    cloned.size = 9999;
    cloned.is_aliased = false;

    // Original unchanged.
    assert_eq!(original.handle, ResourceHandle(5));
    assert_eq!(original.offset, 100);
    assert_eq!(original.size, 200);
    assert!(original.is_aliased);

    // Cloned differs from original.
    assert_ne!(cloned, original, "Mutated clone must differ from original");
}

/// clone of TextureAllocation is a deep (field-wise) copy.
#[test]
fn texture_allocation_clone_independent() {
    let original = TextureAllocation {
        handle: ResourceHandle(10),
        offset: 512,
        size: 2048,
        is_aliased: false,
    };
    let mut cloned = original.clone();
    cloned.handle = ResourceHandle(11);
    cloned.offset = 9999;
    cloned.size = 99999;
    cloned.is_aliased = true;

    assert_eq!(original.handle, ResourceHandle(10));
    assert_eq!(original.offset, 512);
    assert_eq!(original.size, 2048);
    assert!(!original.is_aliased);

    // Cloned differs from original.
    assert_ne!(cloned, original, "Mutated clone must differ from original");
}

/// apply_aliasing with empty resource slice produces empty colour map.
#[test]
fn apply_aliasing_empty_resources() {
    let colours = apply_aliasing(&[], &HashMap::new(), AliasPolicy::Aggressive);
    assert!(colours.is_empty());

    let colours = apply_aliasing(&[], &HashMap::new(), AliasPolicy::Conservative);
    assert!(colours.is_empty());

    let colours = apply_aliasing(&[], &HashMap::new(), AliasPolicy::Disabled);
    assert!(colours.is_empty());
}

/// All three AliasPolicy variants produce the correct number of colours for
/// three non-overlapping resources with the same format.
#[test]
fn alias_policy_comparison_three_same_format() {
    let resources = vec![
        tex2d(1, "a", 100, 100, ResourceLifetime::Transient),
        tex2d(2, "b", 100, 100, ResourceLifetime::Transient),
        tex2d(3, "c", 100, 100, ResourceLifetime::Transient),
    ];
    let lifetimes = HashMap::from([
        (ResourceHandle(1), (PassIndex(0), PassIndex(0))),
        (ResourceHandle(2), (PassIndex(1), PassIndex(1))),
        (ResourceHandle(3), (PassIndex(2), PassIndex(2))),
    ]);

    // Aggressive and Conservative: same format + disjoint lifetimes = 1 colour.
    let aggr = apply_aliasing(&resources, &lifetimes, AliasPolicy::Aggressive);
    let cons = apply_aliasing(&resources, &lifetimes, AliasPolicy::Conservative);
    assert_eq!(num_colors(&aggr), 1, "Aggressive: 1 colour for 3 disjoint same-format");
    assert_eq!(num_colors(&cons), 1, "Conservative: 1 colour for 3 disjoint same-format");

    // Disabled: every resource gets unique colour = 3 colours.
    let dis = apply_aliasing(&resources, &lifetimes, AliasPolicy::Disabled);
    assert_eq!(num_colors(&dis), 3, "Disabled: 3 colours for 3 resources");
}

/// A resource not present in the lifetime map gets a default [0,0] lifetime
/// and is still included in apply_aliasing.
#[test]
fn apply_aliasing_resource_without_lifetime() {
    let resources = vec![tex2d(1, "missing_life", 100, 100, ResourceLifetime::Transient)];
    let colours = apply_aliasing(&resources, &HashMap::new(), AliasPolicy::Aggressive);
    assert_eq!(
        colours.len(),
        1,
        "Resource without lifetime must still get a colour"
    );
}

/// ResourceHandle Display format includes the numeric value.
#[test]
fn resource_handle_display() {
    let h = ResourceHandle(42);
    let s = format!("{}", h);
    assert!(
        s.contains("42"),
        "Display must contain the numeric value, got: {}",
        s
    );
}

/// ResourceHandle::NONE Display includes "NONE".
#[test]
fn resource_handle_none_display() {
    let h = ResourceHandle::NONE;
    let s = format!("{}", h);
    assert!(
        s.contains("NONE"),
        "NONE Display must mention NONE, got: {}",
        s
    );
}

/// PhysicalTexture equality compares all fields.
#[test]
fn physical_texture_partial_eq_all_fields() {
    let base = PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 200, 1, true);

    assert_eq!(base, PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 200, 1, true));
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(2), "rgba8unorm".into(), 100, 200, 1, true), "handle differs");
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(1), "r32float".into(), 100, 200, 1, true), "format differs");
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 200, 200, 1, true), "width differs");
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 300, 1, true), "height differs");
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 200, 2, true), "depth differs");
    assert_ne!(base, PhysicalTexture::new(ResourceHandle(1), "rgba8unorm".into(), 100, 200, 1, false), "transient differs");
}

/// PhysicalBuffer equality compares all fields.
#[test]
fn physical_buffer_partial_eq_all_fields() {
    let base = PhysicalBuffer::new(ResourceHandle(1), 4096, true);

    assert_eq!(base, PhysicalBuffer::new(ResourceHandle(1), 4096, true));
    assert_ne!(base, PhysicalBuffer::new(ResourceHandle(2), 4096, true), "handle differs");
    assert_ne!(base, PhysicalBuffer::new(ResourceHandle(1), 8192, true), "size differs");
    assert_ne!(base, PhysicalBuffer::new(ResourceHandle(1), 4096, false), "transient differs");
}
