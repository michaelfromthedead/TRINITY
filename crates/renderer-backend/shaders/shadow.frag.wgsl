// SPDX-License-Identifier: MIT
//
// shadow.frag.wgsl — Shadow map depth-only fragment shader (T-BRG-6.2).
//
// Minimal fragment entry point. Depth is written automatically by the
// hardware depth attachment; no color output needed for depth-only passes.

@fragment
fn fs_main() {
    // Depth is written automatically to the depth attachment.
    // No color output — this is a depth-only render pass.
}
