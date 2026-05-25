//! Command list and queue tests.
//!
//! Mirrors tests/platform/rhi/test_commands.py plus Rust-native
//! multi-buffer barrier and render pass depth-target tests.

mod common;

use common::*;

// =========================================================================
// Command list tests
// =========================================================================

#[test]
fn test_command_list_begin_end() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.end();

    assert_eq!(cmd.recorded_commands().len(), 0);
}

#[test]
fn test_command_list_not_recording_outside_begin_end() {
    let mut cmd = MockCommandList::new();
    // Commands recorded outside begin/end should be ignored
    cmd.draw(3, 1, 0, 0);
    assert_eq!(cmd.recorded_commands().len(), 0);

    cmd.begin();
    cmd.draw(3, 1, 0, 0);
    cmd.end();
    assert_eq!(cmd.recorded_commands().len(), 1);
}

#[test]
fn test_command_list_barrier() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.barrier(42, ResourceState::Common, ResourceState::UnorderedAccess);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].cmd_type, "barrier");
}

#[test]
fn test_command_list_render_pass() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.begin_render_pass(&[100, 101], Some(200), Some((0.0, 0.0, 0.0, 1.0)), Some(1.0));
    cmd.end_render_pass();
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 2);
    assert_eq!(cmds[0].cmd_type, "begin_render_pass");
    assert_eq!(cmds[1].cmd_type, "end_render_pass");
}

#[test]
fn test_command_list_render_pass_multi_rt() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.begin_render_pass(&[100, 101, 102], None, None, None);
    cmd.end_render_pass();
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds[0].cmd_type, "begin_render_pass");
}

#[test]
fn test_command_list_draw() {
    let device = create_test_device();
    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    });
    let vb = device.create_buffer(BufferDesc {
        size: 4096,
        usage: BufferUsage::VERTEX,
        memory_type: MemoryType::Default,
        stride: 32,
    });

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.set_pipeline(pipeline.handle());
    cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
    cmd.set_scissor(0, 0, 1920, 1080);
    cmd.set_vertex_buffer(0, vb.handle(), 0, 32);
    cmd.draw(3, 1, 0, 0);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 5);
    assert_eq!(cmds[0].cmd_type, "set_pipeline");
    assert_eq!(cmds[1].cmd_type, "set_viewport");
    assert_eq!(cmds[2].cmd_type, "set_scissor");
    assert_eq!(cmds[3].cmd_type, "set_vertex_buffer");
    assert_eq!(cmds[4].cmd_type, "draw");
}

#[test]
fn test_command_list_draw_indexed() {
    let device = create_test_device();
    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    });
    let vb = device.create_buffer(BufferDesc {
        size: 4096,
        usage: BufferUsage::VERTEX,
        memory_type: MemoryType::Default,
        stride: 32,
    });
    let ib = device.create_buffer(BufferDesc {
        size: 2048,
        usage: BufferUsage::INDEX,
        memory_type: MemoryType::Default,
        stride: 4,
    });

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.set_pipeline(pipeline.handle());
    cmd.set_vertex_buffer(0, vb.handle(), 0, 32);
    cmd.set_index_buffer(ib.handle(), 0, Format::R32Uint);
    cmd.draw_indexed(36, 1, 0, 0, 0);
    cmd.end();

    let cmds = cmd.recorded_commands();
    let draw_cmd = &cmds[cmds.len() - 1];
    assert_eq!(draw_cmd.cmd_type, "draw_indexed");
}

#[test]
fn test_command_list_dispatch() {
    let device = create_test_device();
    let pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_shader: Some(ShaderDesc {
            stage: ShaderStage::Compute,
            source: b"cs".to_vec(),
            entry_point: "main".into(),
        }),
    });

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.set_pipeline(pipeline.handle());
    cmd.dispatch(16, 16, 1);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 2);
    assert_eq!(cmds[1].cmd_type, "dispatch");
}

#[test]
fn test_command_list_dispatch_1d() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.dispatch(64, 1, 1);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].cmd_type, "dispatch");
}

#[test]
fn test_command_list_copy_buffer() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.copy_buffer(200, 0, 100, 0, 1024);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].cmd_type, "copy_buffer");
}

#[test]
fn test_command_list_multi_barrier() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.barrier(1, ResourceState::Undefined, ResourceState::CopyDst);
    cmd.barrier(1, ResourceState::CopyDst, ResourceState::ShaderResource);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 2);
    assert_eq!(cmds[0].cmd_type, "barrier");
    assert_eq!(cmds[1].cmd_type, "barrier");
}

#[test]
fn test_command_list_clear_records_on_begin() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.draw(3, 1, 0, 0);
    cmd.end();
    assert_eq!(cmd.recorded_commands().len(), 1);

    // Second begin should clear previous commands
    cmd.begin();
    cmd.end();
    assert_eq!(cmd.recorded_commands().len(), 0);
}

// =========================================================================
// Queue tests
// =========================================================================

#[test]
fn test_queue_submit() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Graphics);

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.dispatch(1, 1, 1);
    cmd.end();

    queue.submit(&[cmd]);
    assert_eq!(queue.submitted_count(), 1);
}

#[test]
fn test_queue_submit_with_fence() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Graphics);
    let fence = MockFence::new(0);
    let initial = fence.value();

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.draw(3, 1, 0, 0);
    cmd.end();

    queue.submit_with_fence(&[cmd], &fence);
    assert!(fence.value() > initial);
}

#[test]
fn test_queue_wait_and_signal() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Graphics);
    let fence = MockFence::new(0);

    queue.signal(&fence);
    assert!(fence.value() > 0);

    queue.wait(&fence); // Should not hang
}

#[test]
fn test_multiple_command_lists_submission() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Graphics);

    let mut cmd1 = MockCommandList::new();
    cmd1.begin();
    cmd1.draw(3, 1, 0, 0);
    cmd1.end();

    let mut cmd2 = MockCommandList::new();
    cmd2.begin();
    cmd2.dispatch(1, 1, 1);
    cmd2.end();

    queue.submit(&[cmd1, cmd2]);
    assert_eq!(queue.submitted_count(), 2);
}

#[test]
fn test_queue_type_in_queue() {
    let device = create_test_device();
    let gfx = device.get_queue(QueueType::Graphics);
    let compute = device.get_queue(QueueType::Compute);

    assert_eq!(gfx.queue_type(), QueueType::Graphics);
    assert_eq!(compute.queue_type(), QueueType::Compute);
}

#[test]
fn test_queue_transfer_operations() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Transfer);
    assert_eq!(queue.queue_type(), QueueType::Transfer);
}

#[test]
fn test_queue_shutdown() {
    let device = create_test_device();
    let queue = device.get_queue(QueueType::Graphics);
    assert!(!queue.is_shutdown());
    queue.shutdown();
    assert!(queue.is_shutdown());
}

// =========================================================================
// Barrier integration tests
// =========================================================================

#[test]
fn test_resource_state_transitions() {
    let mut cmd = MockCommandList::new();
    cmd.begin();

    // Upload → ShaderRead
    cmd.barrier(1, ResourceState::CopyDst, ResourceState::ShaderResource);

    // RenderTarget → Present
    cmd.barrier(2, ResourceState::RenderTarget, ResourceState::Present);

    // Undefined → CopyDst
    cmd.barrier(3, ResourceState::Undefined, ResourceState::CopyDst);

    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 3);
}

#[test]
fn test_uav_barrier() {
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.barrier(1, ResourceState::UnorderedAccess, ResourceState::UnorderedAccess);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds[0].cmd_type, "barrier");
}
