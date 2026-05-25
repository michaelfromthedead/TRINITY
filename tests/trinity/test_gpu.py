"""
Tests for GPU decorators (gpu.py).

Tests the 8 GPU decorators built on Ops:
    @gpu_buffer, @gpu_kernel, @gpu_struct, @bind_group,
    @dispatch, @shader, @render_pass, @async_compute

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.gpu import (
    VALID_BUFFER_USAGE,
    VALID_GPU_BACKENDS,
    VALID_MSAA_SAMPLES,
    VALID_SHADER_STAGES,
    GpuBufferConfig,
    GpuKernelConfig,
    RenderPassConfig,
    ShaderConfig,
    async_compute,
    bind_group,
    dispatch,
    gpu_buffer,
    gpu_kernel,
    gpu_struct,
    render_pass,
    shader,
)
from trinity.decorators.ops import Op, decompose, expand


# =============================================================================
# @gpu_buffer
# =============================================================================


class TestGpuBuffer:
    def test_default_params(self):
        @gpu_buffer()
        class VertexBuffer:
            position: float
            normal: float

        assert VertexBuffer._gpu_buffer is True
        assert isinstance(VertexBuffer._gpu_usage, frozenset)
        assert "storage" in VertexBuffer._gpu_usage
        assert VertexBuffer._gpu_mapped is False

    def test_custom_usage_single(self):
        @gpu_buffer(usage={"vertex"})
        class VB:
            pos: float

        assert "vertex" in VB._gpu_usage

    def test_custom_usage_multiple(self):
        @gpu_buffer(usage={"vertex", "storage"})
        class VB:
            pass

        assert "vertex" in VB._gpu_usage
        assert "storage" in VB._gpu_usage

    def test_mapped_true(self):
        @gpu_buffer(usage={"uniform"}, mapped=True)
        class UB:
            pass

        assert UB._gpu_mapped is True

    def test_invalid_usage(self):
        with pytest.raises(ValueError, match="invalid usage flag"):

            @gpu_buffer(usage={"quantum"})
            class Bad:
                pass

    def test_buffer_fields_extracted(self):
        @gpu_buffer()
        class BufferData:
            x: float
            y: int
            z: bool

        assert hasattr(BufferData, "_gpu_buffer_fields")
        assert "x" in BufferData._gpu_buffer_fields
        assert "y" in BufferData._gpu_buffer_fields
        assert "z" in BufferData._gpu_buffer_fields

    def test_applied_decorators(self):
        @gpu_buffer()
        class C:
            pass

        assert "gpu_buffer" in C._applied_decorators

    def test_steps_recorded(self):
        @gpu_buffer()
        class C:
            pass

        assert hasattr(C, "_applied_steps")
        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used
        assert Op.DESCRIBE in ops_used

    def test_config_object(self):
        @gpu_buffer(usage={"index"}, mapped=True)
        class C:
            pass

        config = C._tags.get("gpu_buffer_config")
        assert isinstance(config, GpuBufferConfig)
        assert "index" in config.usage
        assert config.mapped is True


# =============================================================================
# @gpu_kernel
# =============================================================================


class TestGpuKernel:
    def test_default_params(self):
        @gpu_kernel()
        class MyKernel:
            pass

        assert MyKernel._gpu_kernel is True
        assert MyKernel._workgroup_size == (64, 1, 1)
        assert MyKernel._gpu_backend == "wgpu"

    def test_custom_workgroup_size(self):
        @gpu_kernel(workgroup_size=(8, 8, 1))
        class K:
            pass

        assert K._workgroup_size == (8, 8, 1)

    def test_custom_backend(self):
        @gpu_kernel(backend="cuda")
        class K:
            pass

        assert K._gpu_backend == "cuda"

    def test_metal_backend(self):
        @gpu_kernel(backend="metal")
        class K:
            pass

        assert K._gpu_backend == "metal"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="invalid backend"):

            @gpu_kernel(backend="opencl")
            class Bad:
                pass

    def test_applied_decorators(self):
        @gpu_kernel()
        class C:
            pass

        assert "gpu_kernel" in C._applied_decorators

    def test_steps_recorded(self):
        @gpu_kernel()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_config_object(self):
        @gpu_kernel(workgroup_size=(16, 16, 1), backend="metal")
        class C:
            pass

        config = C._tags.get("gpu_kernel_config")
        assert isinstance(config, GpuKernelConfig)
        assert config.workgroup_size == (16, 16, 1)
        assert config.backend == "metal"


# =============================================================================
# @gpu_struct
# =============================================================================


class TestGpuStruct:
    def test_marker(self):
        @gpu_struct
        class Transform:
            matrix: float

        assert Transform._gpu_struct is True

    def test_with_parens(self):
        @gpu_struct()
        class T:
            pass

        assert T._gpu_struct is True

    def test_size_computation(self):
        @gpu_struct()
        class Vec3:
            x: float
            y: float
            z: float

        assert hasattr(Vec3, "_gpu_struct_size")
        assert Vec3._gpu_struct_size == 12  # 3 floats * 4 bytes

    def test_alignment(self):
        @gpu_struct()
        class S:
            a: int

        assert hasattr(S, "_gpu_struct_alignment")
        assert S._gpu_struct_alignment == 4

    def test_mixed_types(self):
        @gpu_struct()
        class Mixed:
            x: float  # 4
            flag: bool  # 4
            count: int  # 4

        assert Mixed._gpu_struct_size == 12

    def test_applied_decorators(self):
        @gpu_struct
        class C:
            pass

        assert "gpu_struct" in C._applied_decorators

    def test_steps_recorded(self):
        @gpu_struct
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used
        assert Op.DESCRIBE in ops_used


# =============================================================================
# @bind_group
# =============================================================================


class TestBindGroup:
    def test_default_index(self):
        @bind_group()
        class Resources:
            pass

        assert Resources._bind_group is True
        assert Resources._bind_group_index == 0

    def test_custom_index(self):
        @bind_group(index=2)
        class R:
            pass

        assert R._bind_group_index == 2

    def test_index_zero_explicit(self):
        @bind_group(index=0)
        class R:
            pass

        assert R._bind_group_index == 0

    def test_negative_index_fails(self):
        with pytest.raises(ValueError, match="index must be >= 0"):

            @bind_group(index=-1)
            class Bad:
                pass

    def test_applied_decorators(self):
        @bind_group(index=1)
        class C:
            pass

        assert "bind_group" in C._applied_decorators

    def test_steps_recorded(self):
        @bind_group(index=3)
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used


# =============================================================================
# @dispatch
# =============================================================================


class TestDispatch:
    def test_default_params(self):
        @dispatch()
        def compute():
            pass

        assert compute._dispatch is True
        assert compute._dispatch_indirect is False

    def test_indirect_true(self):
        @dispatch(indirect=True)
        def comp():
            pass

        assert comp._dispatch_indirect is True

    def test_on_class(self):
        @dispatch()
        class DispatchConfig:
            pass

        assert DispatchConfig._dispatch is True

    def test_applied_decorators(self):
        @dispatch()
        def fn():
            pass

        assert "dispatch" in fn._applied_decorators

    def test_steps_recorded(self):
        @dispatch(indirect=True)
        def fn():
            pass

        ops_used = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used


# =============================================================================
# @shader
# =============================================================================


class TestShader:
    def test_default_params(self):
        @shader()
        class ComputeShader:
            pass

        assert ComputeShader._shader is True
        assert ComputeShader._shader_stage == "compute"
        assert ComputeShader._shader_entry == "main"

    def test_vertex_stage(self):
        @shader(stage="vertex")
        class VS:
            pass

        assert VS._shader_stage == "vertex"

    def test_fragment_stage(self):
        @shader(stage="fragment", entry="frag_main")
        class FS:
            pass

        assert FS._shader_stage == "fragment"
        assert FS._shader_entry == "frag_main"

    def test_custom_entry_point(self):
        @shader(stage="compute", entry="cs_main")
        class CS:
            pass

        assert CS._shader_entry == "cs_main"

    def test_invalid_stage(self):
        with pytest.raises(ValueError, match="invalid stage"):

            @shader(stage="geometry")
            class Bad:
                pass

    def test_on_function(self):
        @shader(stage="vertex")
        def vertex_main():
            pass

        assert vertex_main._shader is True
        assert vertex_main._shader_stage == "vertex"

    def test_applied_decorators(self):
        @shader(stage="fragment")
        class C:
            pass

        assert "shader" in C._applied_decorators

    def test_config_object(self):
        @shader(stage="vertex", entry="vs_main")
        class C:
            pass

        config = C._tags.get("shader_config")
        assert isinstance(config, ShaderConfig)
        assert config.stage == "vertex"
        assert config.entry == "vs_main"


# =============================================================================
# @render_pass
# =============================================================================


class TestRenderPass:
    def test_default_params(self):
        @render_pass()
        class MainPass:
            pass

        assert MainPass._render_pass is True
        assert MainPass._render_pass_colors == 1
        assert MainPass._render_pass_depth is True
        assert MainPass._render_pass_msaa == 1

    def test_multiple_color_attachments(self):
        @render_pass(color_attachments=4)
        class GBufferPass:
            pass

        assert GBufferPass._render_pass_colors == 4

    def test_no_depth(self):
        @render_pass(depth=False)
        class P:
            pass

        assert P._render_pass_depth is False

    def test_msaa_4x(self):
        @render_pass(msaa=4)
        class P:
            pass

        assert P._render_pass_msaa == 4

    def test_msaa_8x(self):
        @render_pass(msaa=8)
        class P:
            pass

        assert P._render_pass_msaa == 8

    def test_invalid_msaa(self):
        with pytest.raises(ValueError, match="msaa must be power of 2"):

            @render_pass(msaa=3)
            class Bad:
                pass

    def test_invalid_msaa_too_large(self):
        with pytest.raises(ValueError, match="msaa must be power of 2"):

            @render_pass(msaa=32)
            class Bad:
                pass

    def test_zero_color_attachments_fails(self):
        with pytest.raises(ValueError, match="color_attachments must be >= 1"):

            @render_pass(color_attachments=0)
            class Bad:
                pass

    def test_applied_decorators(self):
        @render_pass()
        class C:
            pass

        assert "render_pass" in C._applied_decorators

    def test_config_object(self):
        @render_pass(color_attachments=2, depth=False, msaa=4)
        class C:
            pass

        config = C._tags.get("render_pass_config")
        assert isinstance(config, RenderPassConfig)
        assert config.color_attachments == 2
        assert config.depth is False
        assert config.msaa == 4


# =============================================================================
# @async_compute
# =============================================================================


class TestAsyncCompute:
    def test_marker(self):
        @async_compute
        class AsyncTask:
            pass

        assert AsyncTask._async_compute is True

    def test_with_parens(self):
        @async_compute()
        class T:
            pass

        assert T._async_compute is True

    def test_applied_decorators(self):
        @async_compute
        class C:
            pass

        assert "async_compute" in C._applied_decorators

    def test_steps_recorded(self):
        @async_compute
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used


# =============================================================================
# STACKING
# =============================================================================


class TestGpuStacking:
    def test_kernel_with_async(self):
        @gpu_kernel(backend="cuda")
        @async_compute
        class K:
            pass

        assert K._gpu_kernel is True
        assert K._async_compute is True
        assert "gpu_kernel" in K._applied_decorators
        assert "async_compute" in K._applied_decorators

    def test_buffer_with_bind_group(self):
        @gpu_buffer(usage={"uniform"})
        @bind_group(index=1)
        class UBO:
            data: float

        assert UBO._gpu_buffer is True
        assert UBO._bind_group is True
        assert UBO._bind_group_index == 1

    def test_shader_with_dispatch(self):
        @shader(stage="compute")
        @dispatch(indirect=True)
        class ComputePass:
            pass

        assert ComputePass._shader is True
        assert ComputePass._dispatch is True
        assert ComputePass._dispatch_indirect is True


# =============================================================================
# INTROSPECTION (all decorators decompose)
# =============================================================================


class TestGpuIntrospection:
    @pytest.mark.parametrize(
        "dec",
        [
            gpu_buffer,
            gpu_kernel,
            gpu_struct,
            bind_group,
            dispatch,
            shader,
            render_pass,
            async_compute,
        ],
    )
    def test_decompose_returns_steps(self, dec):
        steps = decompose(dec)
        assert isinstance(steps, list)

    @pytest.mark.parametrize(
        "dec",
        [
            gpu_buffer,
            gpu_kernel,
            gpu_struct,
            bind_group,
            dispatch,
            shader,
            render_pass,
            async_compute,
        ],
    )
    def test_expand_returns_string(self, dec):
        result = expand(dec)
        assert isinstance(result, str)

    def test_all_register_gpu(self):
        """Every GPU decorator should have a REGISTER step for 'gpu'."""
        for dec in [
            gpu_buffer,
            gpu_kernel,
            gpu_struct,
            bind_group,
            dispatch,
            shader,
            render_pass,
            async_compute,
        ]:
            steps = decompose(dec)
            reg_steps = [s for s in steps if s.op is Op.REGISTER]
            assert any(
                s.args.get("registry") == "gpu" for s in reg_steps
            ), f"{dec.__name__} missing REGISTER(gpu) step"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestGpuEdgeCases:
    def test_buffer_usage_as_list(self):
        @gpu_buffer(usage=["vertex", "index"])
        class VB:
            pass

        assert "vertex" in VB._gpu_usage
        assert "index" in VB._gpu_usage

    def test_buffer_usage_as_tuple(self):
        @gpu_buffer(usage=("storage", "indirect"))
        class VB:
            pass

        assert "storage" in VB._gpu_usage
        assert "indirect" in VB._gpu_usage

    def test_struct_no_fields(self):
        @gpu_struct()
        class Empty:
            pass

        assert Empty._gpu_struct_size == 0

    def test_kernel_3d_workgroup(self):
        @gpu_kernel(workgroup_size=(4, 4, 4))
        class K:
            pass

        assert K._workgroup_size == (4, 4, 4)

    def test_render_pass_16x_msaa(self):
        @render_pass(msaa=16)
        class P:
            pass

        assert P._render_pass_msaa == 16

    def test_render_pass_2x_msaa(self):
        @render_pass(msaa=2)
        class P:
            pass

        assert P._render_pass_msaa == 2

    def test_shader_all_stages(self):
        @shader(stage="vertex")
        class VS:
            pass

        @shader(stage="fragment")
        class FS:
            pass

        @shader(stage="compute")
        class CS:
            pass

        assert VS._shader_stage == "vertex"
        assert FS._shader_stage == "fragment"
        assert CS._shader_stage == "compute"

    def test_buffer_usage_single_string(self):
        """Test single string usage (not in collection)."""
        @gpu_buffer(usage="vertex")
        class VB:
            pass

        assert "vertex" in VB._gpu_usage
        assert isinstance(VB._gpu_usage, frozenset)

    def test_struct_unknown_type(self):
        """Test struct with unknown type defaults to 4 bytes."""
        @gpu_struct()
        class Custom:
            custom_field: "UnknownType"

        # Unknown types default to 4 bytes
        assert Custom._gpu_struct_size == 4

    def test_kernel_invalid_workgroup_size_type(self):
        """Test that invalid workgroup_size types are handled."""
        # This doesn't fail validation, but ensures tuple is preserved
        @gpu_kernel(workgroup_size=(1, 2, 3))
        class K:
            pass

        assert K._workgroup_size == (1, 2, 3)
        assert isinstance(K._workgroup_size, tuple)

    def test_render_pass_zero_msaa_fails(self):
        """Test that MSAA=0 is rejected."""
        with pytest.raises(ValueError, match="msaa must be power of 2"):

            @render_pass(msaa=0)
            class Bad:
                pass

    def test_bind_group_large_index(self):
        """Test that large bind group indices are accepted."""
        @bind_group(index=15)
        class R:
            pass

        assert R._bind_group_index == 15

    def test_buffer_empty_usage_set(self):
        """Test that empty usage set defaults correctly."""
        @gpu_buffer(usage=set())
        class B:
            pass

        # Empty set should be converted to frozenset
        assert isinstance(B._gpu_usage, frozenset)
        assert len(B._gpu_usage) == 0


class TestRoundUp:
    def test_align_4_value_0(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(4, 0) == 0
    def test_align_4_value_8(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(4, 8) == 8
    def test_align_4_value_1(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(4, 1) == 4
    def test_align_16_value_12(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(16, 12) == 16
    def test_align_256_value_100(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(256, 100) == 256
    def test_align_1_is_noop(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(1, 42) == 42
    def test_exact_multiples(self):
        from trinity.decorators.gpu import _round_up; assert _round_up(8, 16) == 16 and _round_up(4, 12) == 12

class TestWgslTypeMarkers:
    def test_vec2(self):
        from trinity.decorators.gpu import Vec2; assert Vec2._size == 8 and Vec2._alignment == 8
    def test_vec3(self):
        from trinity.decorators.gpu import Vec3; assert Vec3._size == 12 and Vec3._alignment == 16
    def test_vec4(self):
        from trinity.decorators.gpu import Vec4; assert Vec4._size == 16 and Vec4._alignment == 16
    def test_mat4(self):
        from trinity.decorators.gpu import Mat4; assert Mat4._size == 64 and Mat4._alignment == 16

class TestGetGpuTypeInfo:
    def test_float(self):
        from trinity.decorators.gpu import _get_gpu_type_info; i=_get_gpu_type_info(float); assert i=={'name':'f32','size':4,'align':4}
    def test_int(self):
        from trinity.decorators.gpu import _get_gpu_type_info; i=_get_gpu_type_info(int); assert i=={'name':'i32','size':4,'align':4}
    def test_bool(self):
        from trinity.decorators.gpu import _get_gpu_type_info; i=_get_gpu_type_info(bool); assert i=={'name':'bool','size':4,'align':4}
    def test_vec2(self):
        from trinity.decorators.gpu import Vec2,_get_gpu_type_info; i=_get_gpu_type_info(Vec2); assert i=={'name':'vec2<f32>','size':8,'align':8}
    def test_vec3(self):
        from trinity.decorators.gpu import Vec3,_get_gpu_type_info; i=_get_gpu_type_info(Vec3); assert i=={'name':'vec3<f32>','size':12,'align':16}
    def test_vec4(self):
        from trinity.decorators.gpu import Vec4,_get_gpu_type_info; i=_get_gpu_type_info(Vec4); assert i=={'name':'vec4<f32>','size':16,'align':16}
    def test_mat4(self):
        from trinity.decorators.gpu import Mat4,_get_gpu_type_info; i=_get_gpu_type_info(Mat4); assert i=={'name':'mat4x4<f32>','size':64,'align':16}
    def test_annotated_float_4(self):
        from typing import Annotated; from trinity.decorators.gpu import _get_gpu_type_info; i=_get_gpu_type_info(Annotated[float,4])
        assert i['name']=='array<f32,4>' and i['size']==16 and i['align']==16
    def test_annotated_vec3_8(self):
        from typing import Annotated; from trinity.decorators.gpu import Vec3,_get_gpu_type_info; i=_get_gpu_type_info(Annotated[Vec3,8])
        assert i['name']=='array<vec3<f32>,8>' and i['stride']==16 and i['size']==128
    def test_annotated_vec2_3(self):
        from typing import Annotated; from trinity.decorators.gpu import Vec2,_get_gpu_type_info; i=_get_gpu_type_info(Annotated[Vec2,3])
        assert i['stride']==8 and i['size']==24
    def test_unknown(self):
        from trinity.decorators.gpu import _get_gpu_type_info; i=_get_gpu_type_info('UnknownType'); assert i=={'name':'UnknownType','size':4,'align':4}
    def test_nested_struct(self):
        from trinity.decorators.gpu import gpu_struct,_get_gpu_type_info
        @gpu_struct()
        class Inner: x:float; y:float
        i=_get_gpu_type_info(Inner); assert i['size']==8 and i['align']==4

class TestArrayTypeInfo:
    def test_f32_4(self):
        from trinity.decorators.gpu import _array_type_info; i=_array_type_info({'name':'f32','size':4,'align':4},4)
        assert i['stride']==4 and i['size']==16 and i['align']==16
    def test_vec3_8(self):
        from trinity.decorators.gpu import _array_type_info; i=_array_type_info({'name':'vec3<f32>','size':12,'align':16},8)
        assert i['stride']==16 and i['size']==128
    def test_vec2_3(self):
        from trinity.decorators.gpu import _array_type_info; i=_array_type_info({'name':'vec2<f32>','size':8,'align':8},3)
        assert i['stride']==8 and i['size']==24
    def test_single(self):
        from trinity.decorators.gpu import _array_type_info; i=_array_type_info({'name':'f32','size':4,'align':4},1)
        assert i['align']==16
    def test_large(self):
        from trinity.decorators.gpu import _array_type_info; i=_array_type_info({'name':'f32','size':4,'align':4},1024)
        assert i['size']==4096

class TestComputeGpuStructLayout:
    def test_empty(self):
        from trinity.decorators.gpu import _compute_gpu_struct_layout as c; r=c({}); assert r['size']==0 and r['align']==4
    def test_float(self):
        from trinity.decorators.gpu import _compute_gpu_struct_layout as c; r=c({'x':float}); assert r['size']==4
    def test_three_floats(self):
        from trinity.decorators.gpu import _compute_gpu_struct_layout as c; r=c({'x':float,'y':float,'z':float})
        assert r['size']==12 and [f['offset'] for f in r['fields']]==[0,4,8]
    def test_mesh_vertex(self):
        from trinity.decorators.gpu import Vec3,Vec2,_compute_gpu_struct_layout as c
        r=c({'position':Vec3,'normal':Vec3,'uv':Vec2})
        assert r['size']==48 and r['align']==16
        off={f['name']:f['offset'] for f in r['fields']}
        assert off=={'position':0,'normal':16,'uv':32}
    def test_gpu_instance(self):
        from trinity.decorators.gpu import Mat4,Vec4,_compute_gpu_struct_layout as c
        r=c({'model':Mat4,'color':Vec4,'layer':int})
        assert r['size']==96 and r['align']==16
        off={f['name']:f['offset'] for f in r['fields']}
        assert off=={'model':0,'color':64,'layer':80}
    def test_material_entry(self):
        from typing import Annotated; from trinity.decorators.gpu import Vec4,_compute_gpu_struct_layout as c
        r=c({'bc':Vec4,'r':float,'m':float,'e':Annotated[float,2]})
        assert r['size']==48 and r['align']==16
    def test_single_vec3(self):
        from trinity.decorators.gpu import Vec3,_compute_gpu_struct_layout as c; r=c({'pos':Vec3})
        assert r['size']==12

class TestGpuStructWhitebox:
    def test_mesh_vertex(self):
        from trinity.decorators.gpu import gpu_struct,Vec3,Vec2
        @gpu_struct()
        class MV: position:Vec3; normal:Vec3; uv:Vec2
        assert MV._gpu_struct_size==48 and MV._gpu_struct_alignment==16
    def test_material_entry(self):
        from typing import Annotated; from trinity.decorators.gpu import gpu_struct,Vec4
        @gpu_struct()
        class ME: bc:Vec4; r:float; m:float; e:Annotated[float,2]
        assert ME._gpu_struct_size==48
    def test_gpu_instance(self):
        from trinity.decorators.gpu import gpu_struct,Mat4,Vec4
        @gpu_struct()
        class GI: model:Mat4; color:Vec4; layer:int
        assert GI._gpu_struct_size==96
    def test_nested(self):
        from trinity.decorators.gpu import gpu_struct
        @gpu_struct()
        class I: x:float; y:float
        @gpu_struct()
        class O: inner:I; z:float
        assert O._gpu_struct_size==12
    def test_nested_aligned(self):
        from trinity.decorators.gpu import gpu_struct,Vec4
        @gpu_struct()
        class I: v:Vec4
        @gpu_struct()
        class O: data:I; extra:float
        assert O._gpu_struct_size==32
    def test_annotated_vec3_8(self):
        from typing import Annotated; from trinity.decorators.gpu import gpu_struct,Vec3
        @gpu_struct()
        class B: w:Annotated[Vec3,8]
        assert B._gpu_struct_size==128
    def test_empty(self):
        from trinity.decorators.gpu import gpu_struct
        @gpu_struct()
        class E: pass
        assert E._gpu_struct_size==0 and E._gpu_struct_alignment==4
    def test_all_wgsl(self):
        from trinity.decorators.gpu import gpu_struct,Vec2,Vec3,Vec4,Mat4
        @gpu_struct()
        class A: a:Vec2; b:Vec3; c:Vec4; d:Mat4
        assert A._gpu_struct_size==112
    def test_unknown(self):
        from trinity.decorators.gpu import gpu_struct
        @gpu_struct()
        class C: data:'UnknownType'
        assert C._gpu_struct_size==4


# =============================================================================
# T-GPU-1.1 VERIFICATION TESTS
#
# Three fixes verified:
#   1. f32 class added (_size=4, _alignment=4, __class_getitem__)
#   2. _wgsl_size/_wgsl_align renamed to _size/_alignment
#   3. Struct layout: align=4 default, single-field no trailing pad,
#      stride propagation in fields
# =============================================================================


class TestTgpu11F32Class:
    """Verify fix 1: f32 class with size, alignment, subscript syntax."""

    def test_f32_has_size_and_alignment(self):
        from trinity.decorators.gpu import f32
        assert f32._size == 4
        assert f32._alignment == 4

    def test_f32_getitem_returns_annotated(self):
        from trinity.decorators.gpu import f32
        from typing import Annotated
        result = f32[4]
        assert result is Annotated[float, 4]

    def test_f32_getitem_resolves_via_type_info(self):
        from trinity.decorators.gpu import f32, _get_gpu_type_info
        arr = f32[4]
        info = _get_gpu_type_info(arr)
        assert info["name"] == "array<f32,4>"
        assert info["size"] == 16
        assert info["align"] == 16
        assert info["stride"] == 4

    def test_f32_single_via_type_info(self):
        from trinity.decorators.gpu import f32, _get_gpu_type_info
        info = _get_gpu_type_info(f32)
        assert info == {"name": "f32", "size": 4, "align": 4}

    def test_f32_array_in_gpu_struct(self):
        from trinity.decorators.gpu import gpu_struct, f32
        @gpu_struct()
        class S:
            data: f32[4]
        assert S._gpu_struct_size == 16
        assert S._gpu_struct_alignment == 16
        field = S._gpu_struct_fields[0]
        assert field["type"] == "array<f32,4>"
        assert field["stride"] == 4

    def test_f32_array_in_multi_field_struct(self):
        from trinity.decorators.gpu import gpu_struct, f32, Vec3
        @gpu_struct()
        class S:
            pos: Vec3
            weights: f32[4]
        assert S._gpu_struct_size == 32
        assert S._gpu_struct_alignment == 16

    def test_f32_is_exported(self):
        from trinity.decorators.gpu import __all__
        assert "f32" in __all__


class TestTgpu11RenameAttributes:
    """Verify fix 2: _wgsl_size/_wgsl_align renamed to _size/_alignment."""

    def test_vec2_old_attrs_gone(self):
        from trinity.decorators.gpu import Vec2
        assert not hasattr(Vec2, "_wgsl_size")
        assert not hasattr(Vec2, "_wgsl_align")

    def test_vec3_old_attrs_gone(self):
        from trinity.decorators.gpu import Vec3
        assert not hasattr(Vec3, "_wgsl_size")
        assert not hasattr(Vec3, "_wgsl_align")

    def test_vec4_old_attrs_gone(self):
        from trinity.decorators.gpu import Vec4
        assert not hasattr(Vec4, "_wgsl_size")
        assert not hasattr(Vec4, "_wgsl_align")

    def test_mat4_old_attrs_gone(self):
        from trinity.decorators.gpu import Mat4
        assert not hasattr(Mat4, "_wgsl_size")
        assert not hasattr(Mat4, "_wgsl_align")

    def test_f32_old_attrs_gone(self):
        from trinity.decorators.gpu import f32
        assert not hasattr(f32, "_wgsl_size")
        assert not hasattr(f32, "_wgsl_align")

    def test_duck_reader_rejects_old_style(self):
        """Classes with only _wgsl_size/_wgsl_align are NOT recognized."""
        from trinity.decorators.gpu import _get_gpu_type_info
        class OldStyle:
            _wgsl_size = 8
            _wgsl_align = 8
        info = _get_gpu_type_info(OldStyle)
        assert info["size"] == 4
        assert info["align"] == 4

    def test_duck_reader_accepts_new_style(self):
        """Classes with _size/_alignment ARE recognized (custom types)."""
        from trinity.decorators.gpu import _get_gpu_type_info
        class CustomVec:
            _size = 12
            _alignment = 8
            _wgsl_name = "custom<f32>"
        info = _get_gpu_type_info(CustomVec)
        assert info["name"] == "custom<f32>"
        assert info["size"] == 12
        assert info["align"] == 8


class TestTgpu11StructLayoutFixes:
    """Verify fix 3: struct layout improvements."""

    def test_empty_struct_align_is_4(self):
        """Empty struct alignment default changed from 1 to 4."""
        from trinity.decorators.gpu import _compute_gpu_struct_layout as c
        r = c({})
        assert r["align"] == 4

    def test_single_vec3_no_trailing_pad(self):
        """Single-field non-array struct: content size, no round-up."""
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"pos": Vec3})
        assert r["size"] == 12

    def test_single_float_no_trailing_pad(self):
        from trinity.decorators.gpu import _compute_gpu_struct_layout as c
        r = c({"x": float})
        assert r["size"] == 4

    def test_array_field_rounds_up(self):
        """Array fields always round up (WGSL storage-buffer rule)."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"arr": Annotated[Vec3, 8]})
        assert r["size"] == 128
        assert r["fields"][0]["stride"] == 16

    def test_multi_field_rounds_up(self):
        """Multi-field struct rounds up to alignment per WGSL."""
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"a": float, "b": Vec3})
        assert r["size"] == 32
        assert r["align"] == 16
        off = {f["name"]: f["offset"] for f in r["fields"]}
        assert off["a"] == 0
        assert off["b"] == 16

    def test_stride_propagated_in_fields(self):
        """Array fields carry 'stride' in their field entry."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"arr": Annotated[Vec3, 3]})
        assert "stride" in r["fields"][0]
        assert r["fields"][0]["stride"] == 16

    def test_non_array_fields_no_stride(self):
        """Non-array fields should NOT have 'stride'."""
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"pos": Vec3})
        assert "stride" not in r["fields"][0]

    def test_struct_align_derived_from_fields(self):
        """struct_align is max of field aligns."""
        from trinity.decorators.gpu import Vec4, Vec2, _compute_gpu_struct_layout as c
        r = c({"a": Vec2, "b": Vec4})
        assert r["align"] == 16

    def test_material_entry_with_stride(self):
        """Material entry with mixed scalar + array fields."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec4, _compute_gpu_struct_layout as c
        r = c({"bc": Vec4, "r": float, "m": float, "e": Annotated[float, 2]})
        assert r["size"] == 48
        assert r["align"] == 16
        e_field = [f for f in r["fields"] if f["name"] == "e"][0]
        assert e_field["stride"] == 4
        bc_field = [f for f in r["fields"] if f["name"] == "bc"][0]
        assert "stride" not in bc_field

    def test_vec3_array_with_mixed_offsets(self):
        """Verify offsets with array field after Vec3."""
        from typing import Annotated
        from trinity.decorators.gpu import Vec3, _compute_gpu_struct_layout as c
        r = c({"pos": Vec3, "weights": Annotated[float, 4]})
        off = {f["name"]: f["offset"] for f in r["fields"]}
        assert off["pos"] == 0
        assert off["weights"] == 16
        assert r["size"] == 32
