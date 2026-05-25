// SPDX-License-Identifier: MIT
//
// DISABLED -- blackbox_validate_compile.rs
//
// These tests targeted the old `compile_with_config()` API and `CompilerConfig`
// with `validate_edges` field, which were removed during the frame graph
// compiler rewrite (octo-merge).  The current API uses
// `CompiledFrameGraph::compile(passes, resources)` directly and does not
// expose edge validation as a configurable option.
//
// See these files for equivalent coverage using the current API:
//   - blackbox_ffi_roundtrip.rs        — basic compilation + JSON emission
//   - blackbox_frame_graph_conv.rs     — consistency, multi-pass topologies
//   - blackbox_cull_stats.rs           — cull statistics from compilation
