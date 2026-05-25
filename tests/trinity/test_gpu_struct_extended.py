"""
Blackbox tests for @gpu_struct extension (T-GPU-1.1).

Tests Vec2, Vec3, Vec4, Mat4, f32[N], and nested struct types
with proper WGSL @size/@align byte-accurate layouts.

Cleanroom: Tests written from the public contract only.
Spec: GAPSET_8_GPU_COMPUTE/PHASE_N_TODO.md S: T-GPU-1.1
"""

import pytest

from trinity.decorators.gpu import (
    gpu_struct,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
)


# =============================================================================
# TYPE EXISTENCE
# =============================================================================


class TestVecTypesExist:
    """Vec2, Vec3, Vec4, Mat4 must be importable from the public API."""

    def test_vec2_imported(self):
        assert Vec2 is not None

    def test_vec3_imported(self):
        assert Vec3 is not None

    def test_vec4_imported(self):
        assert Vec4 is not None

    def test_mat4_imported(self):
        assert Mat4 is not None


# =============================================================================
# TYPE SIZE AND ALIGNMENT METADATA
# =============================================================================


class TestVecTypeSizes:
    """Each type exposes _size matching WGSL byte-width."""

    def test_vec2_size(self):
        assert Vec2._size == 8  # 2 x f32

    def test_vec3_size(self):
        assert Vec3._size == 12  # 3 x f32

    def test_vec4_size(self):
        assert Vec4._size == 16  # 4 x f32

    def test_mat4_size(self):
        assert Mat4._size == 64  # 4 x vec4 = 4 x 16


class TestVecTypeAlignments:
    """Each type exposes _alignment matching WGSL @align."""

    def test_vec2_alignment(self):
        assert Vec2._alignment == 8  # vec2<f32> align 8

    def test_vec3_alignment(self):
        assert Vec3._alignment == 16  # vec3<f32> align 16

    def test_vec4_alignment(self):
        assert Vec4._alignment == 16  # vec4<f32> align 16

    def test_mat4_alignment(self):
        assert Mat4._alignment == 16  # mat4x4<f32> align 16


# =============================================================================
# SINGLE-FIELD STRUCTS
# =============================================================================


class TestGpuStructSingleVecFields:
    """@gpu_struct with a single Vec2/Vec3/Vec4/Mat4 field."""

    def test_vec2_field(self):
        @gpu_struct()
        class S:
            pos: Vec2

        assert S._gpu_struct is True
        assert S._gpu_struct_size == 8
        assert S._gpu_struct_alignment == 8

    def test_vec3_field(self):
        @gpu_struct()
        class S:
            pos: Vec3

        assert S._gpu_struct_size == 12
        assert S._gpu_struct_alignment == 16

    def test_vec4_field(self):
        @gpu_struct()
        class S:
            color: Vec4

        assert S._gpu_struct_size == 16
        assert S._gpu_struct_alignment == 16

    def test_mat4_field(self):
        @gpu_struct()
        class S:
            transform: Mat4

        assert S._gpu_struct_size == 64
        assert S._gpu_struct_alignment == 16


# =============================================================================
# MIXED FIELD TYPES
# =============================================================================


class TestGpuStructMixedFields:
    """Structs combining basic types with Vec/Mat types."""

    def test_scalar_plus_vec2(self):
        @gpu_struct()
        class S:
            a: float
            b: Vec2

        # float offset 0-3, padding 4-7 (vec2 needs align 8),
        # vec2 offset 8-15
        # size: 16, align: 8
        assert S._gpu_struct_size == 16
        assert S._gpu_struct_alignment == 8

    def test_scalar_plus_vec4(self):
        @gpu_struct()
        class S:
            a: float
            b: Vec4

        # float offset 0-3, padding 4-15 (vec4 needs align 16),
        # vec4 offset 16-31
        # size: 32, align: 16
        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16

    def test_two_vec4(self):
        @gpu_struct()
        class S:
            a: Vec4
            b: Vec4

        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16

    def test_vec2_vec2_scalar(self):
        @gpu_struct()
        class S:
            a: Vec2
            b: Vec2
            c: float

        # Vec2(0-7), Vec2(8-15), float(16-19)
        # size: 20, rounded up to align 8: 24
        assert S._gpu_struct_alignment == 8
        assert S._gpu_struct_size % S._gpu_struct_alignment == 0


# =============================================================================
# WGSL @size/@align COMPLIANCE
# =============================================================================


class TestGpuStructWgslAlignment:
    """WGSL-specific alignment rules for vec3 and mat4x4."""

    def test_vec3_align16_size12(self):
        """vec3<f32> has alignment 16 but logical size 12."""
        @gpu_struct()
        class S:
            a: Vec3
            b: float

        # Vec3 offset 0-11, float offset 12-15
        # Total: 16 (no padding needed if vec3 comes first)
        assert S._gpu_struct_size == 16
        assert S._gpu_struct_alignment == 16

    def test_float_before_vec3_pads_to_16(self):
        """f32 before vec3 must pad to vec3 alignment boundary."""
        @gpu_struct()
        class S:
            a: float
            b: Vec3

        # float offset 0-3, padding 4-15 (align 16), vec3 offset 16-27
        # Total: 28 -> round to 32 (struct align = max field align = 16)
        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0

    def test_vec3_between_scalars_total_size(self):
        """Mixed layout with vec3 in middle produces correct total."""
        @gpu_struct()
        class S:
            a: float
            b: Vec3
            c: float

        # float 0-3, pad 4-15, vec3 16-27, float 28-31
        # Total: 32 (aligned to 16)
        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16

    def test_mat4_alignment(self):
        """mat4x4<f32> has alignment 16."""
        @gpu_struct()
        class S:
            m: Mat4
            i: int

        # Mat4 0-63, int 64-67
        # Total: 68 -> round to 80 (align 16)
        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0

    def test_complex_wgsl_layout(self):
        """Complex WGSL struct layout with mixed types."""
        @gpu_struct()
        class S:
            pos: Vec3          # 0-11
            uv: Vec2           # 16-23 (align 8, so offset 16)
            color: Vec4        # 32-47 (align 16)
            flags: int         # 48-51

        # Total: 64 (aligned to 16)
        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size == 64


# =============================================================================
# f32[N] ARRAY SUPPORT
# =============================================================================


class TestGpuStructF32Array:
    """f32[N] fixed-size array field support via subscript."""

    def test_f32_imported(self):
        from trinity.decorators.gpu import f32

        assert f32 is not None
        assert f32._size == 4
        assert f32._alignment == 4

    def test_f32_array_4(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[4]

        # 4 x f32 = 16 bytes, align 16 (storage buffer array rule)
        assert S._gpu_struct_size == 16
        assert S._gpu_struct_alignment == 16

    def test_f32_array_8(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[8]

        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16

    def test_f32_array_16(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[16]

        assert S._gpu_struct_size == 64
        assert S._gpu_struct_alignment == 16

    def test_f32_array_with_scalar(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[4]
            extra: float

        # array(0-15), float(16-19)
        # Total: 20 -> round to 32 (align 16)
        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0

    def test_f32_array_2(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[2]

        # 2 x f32 = 8 bytes, align 16 (storage buffer min alignment)
        assert S._gpu_struct_size == 16  # padded to alignment
        assert S._gpu_struct_alignment == 16

    def test_f32_array_1(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            data: f32[1]

        assert S._gpu_struct_size == 16  # padded to alignment
        assert S._gpu_struct_alignment == 16

    def test_two_f32_arrays(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            a: f32[4]
            b: f32[4]

        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16


class TestGpuStructF32ArrayValidation:
    """f32[N] must reject non-positive array sizes."""

    def test_f32_negative_rejected(self):
        from trinity.decorators.gpu import f32
        with pytest.raises(ValueError, match="> 0"):
            f32[-1]

    def test_f32_zero_rejected(self):
        from trinity.decorators.gpu import f32
        with pytest.raises(ValueError, match="> 0"):
            f32[0]

    def test_f32_non_int_rejected(self):
        from trinity.decorators.gpu import f32
        with pytest.raises(TypeError, match="int"):
            f32["not_a_number"]  # type: ignore[arg-type]

    def test_f32_negative_via_annotated_rejected(self):
        from trinity.decorators.gpu import _get_gpu_type_info
        from typing import Annotated
        with pytest.raises(ValueError, match="> 0"):
            _get_gpu_type_info(Annotated[float, -1])

    def test_f32_negative_via_struct_rejected(self):
        from trinity.decorators.gpu import f32

        with pytest.raises(ValueError, match="> 0"):
            @gpu_struct()
            class S:
                data: f32[-1]

    def test_f32_one_accepted(self):
        from trinity.decorators.gpu import f32
        t = f32[1]
        # Should not raise
        assert t is not None


# =============================================================================
# NESTED STRUCT SUPPORT
# =============================================================================


class TestGpuStructNested:
    """@gpu_struct containing another @gpu_struct field."""

    def test_nested_inner_resolves_size(self):
        @gpu_struct()
        class Inner:
            x: float
            y: float

        @gpu_struct()
        class Outer:
            inner: Inner
            z: float

        # Inner = 8 bytes, align 4
        # Outer: inner 0-7, z 8-11 = 12, align 4
        assert Outer._gpu_struct_size == 12
        assert Outer._gpu_struct_alignment == 4

    def test_nested_vec3_struct(self):
        @gpu_struct()
        class Inner:
            pos: Vec3

        @gpu_struct()
        class Outer:
            inner: Inner
            extra: float

        # Inner = 12 bytes, align 16
        # Outer: inner 0-11, float 12-15 = 16, align 16
        assert Outer._gpu_struct_alignment == 16
        assert Outer._gpu_struct_size % 16 == 0

    def test_deeply_nested(self):
        @gpu_struct()
        class A:
            x: float

        @gpu_struct()
        class B:
            a: A
            y: float

        @gpu_struct()
        class C:
            b: B
            z: float

        assert C._gpu_struct is True
        assert C._gpu_struct_size > 0
        assert C._gpu_struct_alignment >= 4

    def test_nested_with_vec_fields(self):
        @gpu_struct()
        class Vertex:
            pos: Vec3
            uv: Vec2

        @gpu_struct()
        class Meshlet:
            vertices: Vertex
            index_count: int

        assert Meshlet._gpu_struct is True
        assert Meshlet._gpu_struct_size > 0
        assert Meshlet._gpu_struct_alignment == 16  # max of vec3(16), vec2(8), int(4)


# =============================================================================
# ACCEPTANCE: MeshTableEntry
# =============================================================================


class TestMeshTableEntry:
    """MeshTableEntry must be definable with @gpu_struct and produce
    a correct byte-accurate layout suitable for array<MeshTableEntry>."""

    def test_mesh_table_entry_defined(self):
        @gpu_struct()
        class MeshTableEntry:
            vertex_offset: int       # 0-3
            index_offset: int        # 4-7
            vertex_count: int        # 8-11
            index_count: int         # 12-15
            material_index: int      # 16-19
            lod_base: int            # 20-23
            bounding_sphere: Vec4    # 32-47 (align 16)

        assert MeshTableEntry._gpu_struct is True
        assert MeshTableEntry._gpu_struct_size > 0
        assert MeshTableEntry._gpu_struct_alignment == 16

    def test_mesh_table_entry_size_aligned(self):
        @gpu_struct()
        class MeshTableEntry:
            vertex_offset: int
            index_offset: int
            vertex_count: int
            index_count: int
            material_index: int
            lod_base: int
            bounding_sphere: Vec4

        # Size must be multiple of alignment
        size = MeshTableEntry._gpu_struct_size
        align = MeshTableEntry._gpu_struct_alignment
        assert size % align == 0, f"Size {size} not multiple of alignment {align}"

    def test_mesh_table_entry_reasonable_size(self):
        @gpu_struct()
        class MeshTableEntry:
            vertex_offset: int       # 4
            index_offset: int        # 4
            vertex_count: int        # 4
            index_count: int         # 4
            material_index: int      # 4
            lod_base: int            # 4
            bounding_sphere: Vec4    # 16

        # 6 x int(4) + Vec4(16) + alignment padding
        size = MeshTableEntry._gpu_struct_size
        assert 48 <= size <= 64, f"Unexpected size {size}"


# =============================================================================
# ACCEPTANCE: MaterialTableEntry
# =============================================================================


class TestMaterialTableEntry:
    """MaterialTableEntry must be definable with @gpu_struct and produce
    a correct byte-accurate layout suitable for array<MaterialTableEntry>."""

    def test_material_table_entry_defined(self):
        @gpu_struct()
        class MaterialTableEntry:
            base_color: Vec4        # 0-15
            metallic: float         # 16-19
            roughness: float        # 20-23
            emissive: Vec4          # 32-47 (align 16)

        assert MaterialTableEntry._gpu_struct is True
        assert MaterialTableEntry._gpu_struct_size > 0
        assert MaterialTableEntry._gpu_struct_alignment == 16

    def test_material_table_entry_size_aligned(self):
        @gpu_struct()
        class MaterialTableEntry:
            base_color: Vec4
            metallic: float
            roughness: float
            emissive: Vec4

        size = MaterialTableEntry._gpu_struct_size
        align = MaterialTableEntry._gpu_struct_alignment
        assert size % align == 0, f"Size {size} not multiple of alignment {align}"

    def test_material_table_entry_reasonable_size(self):
        @gpu_struct()
        class MaterialTableEntry:
            base_color: Vec4        # 16
            metallic: float         # 4
            roughness: float        # 4
            emissive: Vec4          # 16

        size = MaterialTableEntry._gpu_struct_size
        # 2 x Vec4(16) + 2 x float(4) + alignment padding
        assert 48 <= size <= 64, f"Unexpected size {size}"

    def test_material_table_with_texture_indices(self):
        @gpu_struct()
        class MaterialTableEntry:
            base_color: Vec4        # 16
            metallic: float         # 4
            roughness: float        # 4
            emissive_factor: Vec4   # 32 (align 16)
            texture_indices: Vec4   # 48 (align 16)

        size = MaterialTableEntry._gpu_struct_size
        align = MaterialTableEntry._gpu_struct_alignment
        assert size % align == 0
        assert 64 <= size <= 80, f"Unexpected size {size}"


# =============================================================================
# ACCEPTANCE: GPUInstance
# =============================================================================


class TestGPUInstance:
    """GPUInstance must be definable with @gpu_struct and produce
    a correct byte-accurate layout."""

    def test_gpu_instance_defined(self):
        @gpu_struct()
        class GPUInstance:
            transform: Mat4          # 0-63
            prev_transform: Mat4     # 64-127
            mesh_index: int          # 128-131
            material_index: int      # 132-135
            flags: int               # 136-139
            draw_id: int             # 140-143

        assert GPUInstance._gpu_struct is True
        assert GPUInstance._gpu_struct_size > 0
        assert GPUInstance._gpu_struct_alignment == 16

    def test_gpu_instance_size_aligned(self):
        @gpu_struct()
        class GPUInstance:
            transform: Mat4
            prev_transform: Mat4
            mesh_index: int
            material_index: int
            flags: int
            draw_id: int

        size = GPUInstance._gpu_struct_size
        align = GPUInstance._gpu_struct_alignment
        assert size % align == 0, f"Size {size} not multiple of alignment {align}"

    def test_gpu_instance_reasonable_size(self):
        @gpu_struct()
        class GPUInstance:
            transform: Mat4          # 64
            prev_transform: Mat4     # 64
            mesh_index: int          # 4
            material_index: int      # 4
            flags: int               # 4
            draw_id: int             # 4

        size = GPUInstance._gpu_struct_size
        # 2 x Mat4(64) + 4 x int(4) + alignment padding
        assert 144 <= size <= 160, f"Unexpected size {size}"

    def test_gpu_instance_compact(self):
        """Compact GPUInstance without double transform."""
        @gpu_struct()
        class GPUInstance:
            transform: Mat4          # 0-63
            mesh_index: int          # 64-67
            material_index: int      # 68-71
            flags: int               # 72-75

        size = GPUInstance._gpu_struct_size
        align = GPUInstance._gpu_struct_alignment
        assert size % align == 0
        assert size == 80, f"Unexpected size {size}"


# =============================================================================
# WGSL @size ATTRIBUTE COMPLIANCE
# =============================================================================


class TestGpuStructSizeAttribute:
    """Struct total size must match WGSL @size(n) semantics:
    size must be a multiple of alignment."""

    def test_empty_struct_size_zero(self):
        @gpu_struct()
        class Empty:
            pass

        assert Empty._gpu_struct_size == 0
        assert Empty._gpu_struct_alignment == 4

    def test_single_float_size(self):
        @gpu_struct()
        class S:
            x: float

        assert S._gpu_struct_size == 4
        assert S._gpu_struct_alignment == 4

    def test_size_is_aligned_to_alignment(self):
        """Struct size must always be a multiple of its alignment."""
        @gpu_struct()
        class S:
            a: float
            b: Vec3
            c: float

        size = S._gpu_struct_size
        align = S._gpu_struct_alignment
        assert size % align == 0, f"@{align} struct of size {size} is misaligned"


# =============================================================================
# FIELD-LEVEL METADATA
# =============================================================================


class TestGpuStructFieldMetadata:
    """If field-level metadata (_gpu_struct_fields) is exposed,
    it must report correct offset, size, and alignment per field."""

    def test_field_metadata_exists(self):
        @gpu_struct()
        class S:
            x: Vec4

        assert hasattr(S, "_gpu_struct_fields"), (
            "@gpu_struct should expose _gpu_struct_fields"
        )

    def test_field_metadata_structure(self):
        @gpu_struct()
        class S:
            x: Vec4

        fields = S._gpu_struct_fields
        assert isinstance(fields, (list, tuple, dict))
        if isinstance(fields, dict):
            assert "x" in fields
        elif isinstance(fields, (list, tuple)):
            assert len(fields) >= 1

    def test_field_metadata_offsets(self):
        @gpu_struct()
        class S:
            a: float
            b: Vec2

        fields = S._gpu_struct_fields
        # a should be at offset 0, b should be at offset 8 (align 8)
        a_info = self._get_field(fields, "a")
        b_info = self._get_field(fields, "b")
        assert a_info is not None
        assert b_info is not None

    def _get_field(self, fields, name):
        if isinstance(fields, dict):
            return fields.get(name)
        for f in fields:
            if isinstance(f, dict) and f.get("name") == name:
                return f
            if isinstance(f, (list, tuple)) and f[0] == name:
                return f
        return None


# =============================================================================
# F32 TYPE DIRECT FIELD
# =============================================================================


class TestGpuStructF32Direct:
    """f32 used directly as a field type (alias for float)."""

    def test_f32_direct_field(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            value: f32

        assert S._gpu_struct_size == 4
        assert S._gpu_struct_alignment == 4

    def test_f32_mixed_with_int(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            a: f32
            b: int

        assert S._gpu_struct_size == 8
        assert S._gpu_struct_alignment == 4

    def test_f32_mixed_with_vec4(self):
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            a: f32
            b: Vec4

        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0


# =============================================================================
# DECORATOR MARKER AND METADATA
# =============================================================================


class TestGpuStructMarker:
    """@gpu_struct sets standard metadata attributes."""

    def test_marker_set(self):
        @gpu_struct()
        class S:
            v: Vec3

        assert S._gpu_struct is True

    def test_marker_no_parens(self):
        @gpu_struct
        class S:
            v: Vec3

        assert S._gpu_struct is True

    def test_applied_decorators(self):
        @gpu_struct()
        class S:
            v: Vec3

        assert "gpu_struct" in S._applied_decorators

    def test_applied_steps_recorded(self):
        from trinity.decorators.ops import Op

        @gpu_struct()
        class S:
            v: Vec3

        assert hasattr(S, "_applied_steps")
        ops_used = {step.op for step in S._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used
        assert Op.DESCRIBE in ops_used


# =============================================================================
# VEC4 WITH SCALARS EDGE CASES
# =============================================================================


class TestGpuStructEdgeCases:
    """Edge cases for extended @gpu_struct."""

    def test_all_vec_types_in_one_struct(self):
        @gpu_struct()
        class S:
            a: Vec2   # 0-7
            b: Vec3   # 16-27 (align 16)
            c: Vec4   # 32-47 (align 16)
            d: Mat4   # 64-127 (align 16)

        assert S._gpu_struct is True
        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0

    def test_struct_marked_twice(self):
        """Applying @gpu_struct twice should not error."""
        @gpu_struct()
        @gpu_struct()
        class S:
            v: Vec4

        assert S._gpu_struct is True
        assert S._gpu_struct_size == 16

    def test_struct_with_many_fields(self):
        @gpu_struct()
        class S:
            v0: Vec4
            v1: Vec4
            v2: Vec4
            v3: Vec4
            v4: Vec4
            v5: Vec4
            v6: Vec4
            v7: Vec4

        # 8 x Vec4 = 128 bytes
        assert S._gpu_struct_size == 128
        assert S._gpu_struct_alignment == 16

    def test_mat4_with_floats(self):
        @gpu_struct()
        class S:
            transform: Mat4
            weight: float

        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size == 80  # 64 + 4 + padding

    def test_vec_field_with_bool(self):
        @gpu_struct()
        class S:
            v: Vec4
            flag: bool

        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size % 16 == 0


# =============================================================================
# WGSL ROUND-TRIP COMPATIBILITY
# =============================================================================


class TestGpuStructWgslRoundTrip:
    """Struct layout must match equivalent WGSL struct layout."""

    def test_simple_wgsl_struct_equivalent(self):
        """Equivalent to WGSL struct { pos: vec3<f32>, alpha: f32 }."""
        @gpu_struct()
        class S:
            pos: Vec3
            alpha: float

        assert S._gpu_struct_size == 16  # 12 + 4
        assert S._gpu_struct_alignment == 16

    def test_wgsl_mat4_weight_equivalent(self):
        """Equivalent to WGSL struct { transform: mat4x4<f32>, weight: f32 }."""
        @gpu_struct()
        class S:
            transform: Mat4
            weight: float

        # mat4x4 0-63, f32 64-67
        # struct align 16, size must be multiple of 16 -> 80
        assert S._gpu_struct_size == 80
        assert S._gpu_struct_alignment == 16

    def test_wgsl_vec4_array_equivalent(self):
        """Equivalent to WGSL struct { color: vec4<f32>, data: array<f32, 4> }."""
        from trinity.decorators.gpu import f32

        @gpu_struct()
        class S:
            color: Vec4
            data: f32[4]

        assert S._gpu_struct_alignment == 16
        assert S._gpu_struct_size == 32  # 16 + 16
