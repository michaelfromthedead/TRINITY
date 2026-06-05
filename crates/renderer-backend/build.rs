//! Build-time DSL compilation for TRINITY demoscene shaders (T-DEMO-5.5).
//!
//! This build script:
//! 1. Invokes the Python DSL compiler (`scripts/compile_demo.py`)
//! 2. Compiles scene definitions from `scenes/` to WGSL
//! 3. Embeds the generated WGSL as `const` strings for runtime use
//!
//! # Rerun Triggers
//!
//! The build script reruns when:
//! - `scenes/demo.py` changes
//! - `scripts/compile_demo.py` changes
//! - Any file in `engine/rendering/demoscene/` changes
//!
//! # Output
//!
//! Generated files are placed in `$OUT_DIR/generated/`:
//! - `demo.wgsl` - Compiled demoscene shader
//!
//! # Usage in Rust
//!
//! ```rust,ignore
//! const DEMO_SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/generated/demo.wgsl"));
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Exit codes from compile_demo.py
mod exit_codes {
    pub const SUCCESS: i32 = 0;
    pub const MISSING_SCENE: i32 = 1;
    pub const IMPORT_ERROR: i32 = 2;
    pub const MISSING_SCENE_VAR: i32 = 3;
    pub const INVALID_SCENE_TYPE: i32 = 4;
    pub const COMPILATION_ERROR: i32 = 5;
    pub const WGSL_VALIDATION_ERROR: i32 = 6;
    pub const OUTPUT_WRITE_ERROR: i32 = 7;
    pub const INVALID_ARGUMENTS: i32 = 8;
}

/// Error type for build script operations.
#[derive(Debug)]
enum BuildError {
    /// Scene file not found
    MissingScene(PathBuf),
    /// Python import error
    ImportError(String),
    /// Missing SCENE variable in module
    MissingSceneVar,
    /// Invalid scene type
    InvalidSceneType,
    /// WGSL compilation failed
    CompilationFailed(String),
    /// WGSL validation failed
    ValidationFailed,
    /// Failed to write output
    OutputError(String),
    /// Python/uv not found
    PythonNotFound,
    /// Other error
    Other(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingScene(path) => write!(f, "Scene file not found: {}", path.display()),
            Self::ImportError(msg) => write!(f, "Python import error: {}", msg),
            Self::MissingSceneVar => write!(f, "Missing SCENE variable in scene module"),
            Self::InvalidSceneType => write!(f, "SCENE must be a FullSceneNode"),
            Self::CompilationFailed(msg) => write!(f, "WGSL compilation failed: {}", msg),
            Self::ValidationFailed => write!(f, "WGSL validation failed"),
            Self::OutputError(msg) => write!(f, "Failed to write output: {}", msg),
            Self::PythonNotFound => write!(f, "Python/uv not found - install uv"),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

/// Convert exit code to BuildError.
fn exit_code_to_error(code: i32, stderr: &str) -> BuildError {
    match code {
        exit_codes::MISSING_SCENE => BuildError::MissingScene(PathBuf::new()),
        exit_codes::IMPORT_ERROR => BuildError::ImportError(stderr.to_string()),
        exit_codes::MISSING_SCENE_VAR => BuildError::MissingSceneVar,
        exit_codes::INVALID_SCENE_TYPE => BuildError::InvalidSceneType,
        exit_codes::COMPILATION_ERROR => BuildError::CompilationFailed(stderr.to_string()),
        exit_codes::WGSL_VALIDATION_ERROR => BuildError::ValidationFailed,
        exit_codes::OUTPUT_WRITE_ERROR => BuildError::OutputError(stderr.to_string()),
        _ => BuildError::Other(format!("Unknown error (code {}): {}", code, stderr)),
    }
}

/// Find the project root directory.
fn find_project_root() -> Option<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let manifest_path = PathBuf::from(manifest_dir);

    // Navigate up from crates/renderer-backend to project root
    manifest_path.parent()?.parent().map(PathBuf::from)
}

/// Run the Python DSL compiler.
fn compile_scene(
    project_root: &Path,
    scene_path: &Path,
    output_path: &Path,
) -> Result<(), BuildError> {
    let script_path = project_root.join("scripts/compile_demo.py");

    // Check if scene file exists
    if !scene_path.exists() {
        return Err(BuildError::MissingScene(scene_path.to_path_buf()));
    }

    // Check if script exists
    if !script_path.exists() {
        return Err(BuildError::Other(format!(
            "Compiler script not found: {}",
            script_path.display()
        )));
    }

    // Create output directory
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| BuildError::OutputError(e.to_string()))?;
    }

    // Try uv first, then fall back to python3
    let result = Command::new("uv")
        .args(["run", "python"])
        .arg(&script_path)
        .arg(scene_path)
        .arg(output_path)
        .arg("--verbose")
        .current_dir(project_root)
        .output();

    let output = match result {
        Ok(output) => output,
        Err(_) => {
            // Fall back to python3
            Command::new("python3")
                .arg(&script_path)
                .arg(scene_path)
                .arg(output_path)
                .arg("--verbose")
                .current_dir(project_root)
                .output()
                .map_err(|_| BuildError::PythonNotFound)?
        }
    };

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);

        eprintln!("compile_demo.py stdout: {}", stdout);
        eprintln!("compile_demo.py stderr: {}", stderr);

        return Err(exit_code_to_error(code, &stderr));
    }

    // Print compiler output for cargo
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        println!("cargo:warning=compile_demo: {}", stdout.trim());
    }

    Ok(())
}

/// Emit rerun-if-changed directives for the build script.
fn emit_rerun_directives(project_root: &Path) {
    // Scene file
    println!(
        "cargo:rerun-if-changed={}",
        project_root.join("scenes/demo.py").display()
    );

    // Compiler script
    println!(
        "cargo:rerun-if-changed={}",
        project_root.join("scripts/compile_demo.py").display()
    );

    // Demoscene module (track all Python files)
    let demoscene_dir = project_root.join("engine/rendering/demoscene");
    if demoscene_dir.exists() {
        println!("cargo:rerun-if-changed={}", demoscene_dir.display());

        // Also track individual Python files for more precise rerun
        if let Ok(entries) = fs::read_dir(&demoscene_dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "py") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}

/// Generate a fallback WGSL shader when compilation fails.
fn generate_fallback_shader(output_path: &Path, error: &BuildError) -> Result<(), std::io::Error> {
    let fallback = format!(
        r#"// SPDX-License-Identifier: MIT
//
// FALLBACK SHADER - Compilation failed
// Error: {}
//
// This shader displays a solid color to indicate the build failed.

@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {{
    camera_origin: vec3<f32>,
    camera_fov: f32,
    camera_target: vec3<f32>,
    camera_aspect: f32,
    camera_up: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    max_steps: u32,
    max_distance: f32,
    epsilon: f32,
    _padding: vec3<f32>,
}}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

struct Material {{
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    ambient_occlusion: f32,
}}

fn scene_sdf(p: vec3<f32>) -> vec2<f32> {{
    // Simple sphere fallback
    let d = length(p) - 1.0;
    return vec2<f32>(d, 0.0);
}}

fn scene_material(id: u32) -> Material {{
    return Material(
        vec3<f32>(1.0, 0.0, 1.0), // Magenta = error
        0.5,
        0.0,
        vec3<f32>(0.0),
        1.0,
    );
}}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let dims = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {{
        return;
    }}

    // Error indicator: magenta/black checkerboard
    let checker = (global_id.x / 32u + global_id.y / 32u) % 2u;
    let color = select(vec3<f32>(0.0), vec3<f32>(1.0, 0.0, 1.0), checker == 0u);

    textureStore(output_texture, coords, vec4<f32>(color, 1.0));
}}
"#,
        error
    );

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(output_path, fallback)
}

fn main() {
    // Find project root
    let project_root = match find_project_root() {
        Some(root) => root,
        None => {
            eprintln!("cargo:warning=Could not find project root");
            return;
        }
    };

    // Get OUT_DIR
    let out_dir = match env::var("OUT_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            eprintln!("cargo:warning=OUT_DIR not set");
            return;
        }
    };

    // Define paths
    let scene_path = project_root.join("scenes/demo.py");
    let output_path = out_dir.join("generated/demo.wgsl");

    // Emit rerun directives
    emit_rerun_directives(&project_root);

    // Compile the scene
    match compile_scene(&project_root, &scene_path, &output_path) {
        Ok(()) => {
            println!(
                "cargo:warning=Compiled demoscene shader: {}",
                output_path.display()
            );
        }
        Err(ref e @ BuildError::MissingScene(_)) => {
            // Scene file doesn't exist yet - generate fallback
            println!(
                "cargo:warning=Scene file not found, generating fallback: {}",
                e
            );
            if let Err(write_err) = generate_fallback_shader(&output_path, e) {
                eprintln!("cargo:warning=Failed to write fallback shader: {}", write_err);
            }
        }
        Err(ref e @ BuildError::PythonNotFound) => {
            // Python not available - generate fallback
            println!("cargo:warning=Python not found, generating fallback: {}", e);
            if let Err(write_err) = generate_fallback_shader(&output_path, e) {
                eprintln!("cargo:warning=Failed to write fallback shader: {}", write_err);
            }
        }
        Err(ref e) => {
            // Compilation failed - generate fallback and warn
            println!("cargo:warning=Scene compilation failed: {}", e);
            if let Err(write_err) = generate_fallback_shader(&output_path, e) {
                eprintln!("cargo:warning=Failed to write fallback shader: {}", write_err);
            }
        }
    }

    // Export the output path as an environment variable for tests
    println!("cargo:rustc-env=DEMO_WGSL_PATH={}", output_path.display());
}
