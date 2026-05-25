// SPDX-License-Identifier: MIT
//
// DISABLED -- blackbox_profile.rs
//
// These tests targeted the old `CompilerProfile` enum (Debug / Default /
// Performance) and `FrameGraphCompiler` builder, which were removed during
// the frame graph compiler rewrite (octo-merge).  The current API uses
// `CompiledFrameGraph::compile(passes, resources)` directly with no profile
// or compiler-object layer.
//
// See these files for equivalent coverage using the current API:
//   - blackbox_ffi_roundtrip.rs        — basic compilation + JSON emission
//   - blackbox_frame_graph_conv.rs     — consistency, multi-pass topologies
//   - blackbox_cull_stats.rs           — cull statistics from compilation
