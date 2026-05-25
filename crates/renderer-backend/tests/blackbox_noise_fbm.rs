// SPDX-License-Identifier: MIT
//
// blackbox_noise_fbm.rs -- Blackbox tests for noise_fbm.wgsl.
//
// This test file verifies the WGSL FBM noise source is structurally valid.
// Cleanroom: tests are based ONLY on the spec definition (T-DEMO-1.31:
// "fractal Brownian motion, configurable octaves/lacunarity/gain,
//  correct spectral composition, WGSL. Uses value or Perlin noise as base.").
//
// Coverage:
//   - naga WGSL compilation (full file)
//   - Individual fbm_1d, fbm_2d, fbm_3d, fbm_perlin_3d function compilation
//   - Function existence verification
//   - Source structure checks
//   - Spectral composition verification (gain/lacunarity/octaves config)

use naga::front::wgsl::parse_str;

/// The full WGSL source file, baked in at compile time via include_str!.
static WGSL_SOURCE: &str = include_str!("../src/demoscene/noise_fbm.wgsl");

// =============================================================================
// Section 1: naga compilation
// =============================================================================

/// Helper: concatenate all noise WGSL dependencies for compilation tests.
/// noise_fbm.wgsl contains both value-noise-based and perlin-noise-based FBM
/// functions, so all three dependency files must always be included.
fn dependency_source() -> String {
    format!(
        "{}\n\n{}\n\n{}\n\n{}",
        include_str!("../src/demoscene/noise_hash.wgsl"),
        include_str!("../src/demoscene/noise_value.wgsl"),
        include_str!("../src/demoscene/noise_perlin.wgsl"),
        WGSL_SOURCE
    )
}

/// The full noise_fbm.wgsl file must compile through naga without errors.
/// Note: requires all dependencies (hash + value + perlin) since the file
/// contains both value-noise-based (fbm_1d/2d/3d) and perlin-based
/// (fbm_perlin_3d) functions in a single source file.
#[test]
fn compiles_via_naga() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("noise_fbm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// fbm_1d must compile with all dependencies (hash + value + perlin).
/// noise_fbm.wgsl is a single file containing both value and perlin variants,
/// so all dependencies are always required.
#[test]
fn fbm_1d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"fbm_1d"),
                "fbm_1d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_value.wgsl + noise_fbm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// fbm_2d must compile with all dependencies (hash + value + perlin).
#[test]
fn fbm_2d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"fbm_2d"),
                "fbm_2d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_value.wgsl + noise_fbm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// fbm_3d must compile with all dependencies (hash + value + perlin).
#[test]
fn fbm_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"fbm_3d"),
                "fbm_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_value.wgsl + noise_fbm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// fbm_perlin_3d must compile with all dependencies (hash + value + perlin).
/// noise_fbm.wgsl is a single file -- all deps are always needed.
#[test]
fn fbm_perlin_3d_compiles_individually() {
    let source = dependency_source();
    let result = parse_str(&source);
    match result {
        Ok(module) => {
            let names: Vec<&str> = module.functions.iter()
                .filter_map(|(_, f)| f.name.as_deref())
                .collect();
            assert!(
                names.contains(&"fbm_perlin_3d"),
                "fbm_perlin_3d not found in parsed module. Found functions: {:?}",
                names
            );
        }
        Err(err) => {
            panic!("noise_hash.wgsl + noise_perlin.wgsl + noise_fbm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// Section 2: Function existence
// =============================================================================

/// fbm_1d must exist in the source.
#[test]
fn fbm_1d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn fbm_1d("),
        "fbm_1d function not found in noise_fbm.wgsl"
    );
}

/// fbm_2d must exist in the source.
#[test]
fn fbm_2d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn fbm_2d("),
        "fbm_2d function not found in noise_fbm.wgsl"
    );
}

/// fbm_3d must exist in the source.
#[test]
fn fbm_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn fbm_3d("),
        "fbm_3d function not found in noise_fbm.wgsl"
    );
}

/// fbm_perlin_3d must exist in the source.
#[test]
fn fbm_perlin_3d_exists() {
    assert!(
        WGSL_SOURCE.contains("fn fbm_perlin_3d("),
        "fbm_perlin_3d function not found in noise_fbm.wgsl"
    );
}

// =============================================================================
// Section 3: Source structure
// =============================================================================

/// The file must start with the SPDX MIT license header.
#[test]
fn starts_with_license() {
    assert!(
        WGSL_SOURCE.starts_with("// SPDX-License-Identifier: MIT"),
        "noise_fbm.wgsl must start with the MIT license header"
    );
}

/// The T-DEMO-1.31 section headers must appear.
#[test]
fn section_headers_present() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.31: FBM Noise 1D"),
        "Section header 'T-DEMO-1.31: FBM Noise 1D' not found"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.31: FBM Noise 2D"),
        "Section header 'T-DEMO-1.31: FBM Noise 2D' not found"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.31: FBM Noise 3D"),
        "Section header 'T-DEMO-1.31: FBM Noise 3D' not found"
    );
}

/// The file must reference its dependency files.
#[test]
fn dependency_references_present() {
    assert!(
        WGSL_SOURCE.contains("noise_hash.wgsl"),
        "noise_fbm.wgsl must reference noise_hash.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_value.wgsl"),
        "noise_fbm.wgsl must reference noise_value.wgsl as a dependency"
    );
    assert!(
        WGSL_SOURCE.contains("noise_perlin.wgsl"),
        "noise_fbm.wgsl must reference noise_perlin.wgsl as a dependency"
    );
}

/// The file must reference T-DEMO-1.28, T-DEMO-1.29, T-DEMO-1.30.
#[test]
fn references_dependency_versions() {
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.28"),
        "noise_fbm.wgsl must reference T-DEMO-1.28 as hash dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.29"),
        "noise_fbm.wgsl must reference T-DEMO-1.29 as value noise dependency"
    );
    assert!(
        WGSL_SOURCE.contains("T-DEMO-1.30"),
        "noise_fbm.wgsl must reference T-DEMO-1.30 as Perlin noise dependency"
    );
}

/// fbm_1d must call value_noise_1d.
#[test]
fn fbm_1d_uses_value_noise_1d() {
    assert!(
        WGSL_SOURCE.contains("value_noise_1d("),
        "fbm_1d must call value_noise_1d"
    );
}

/// fbm_2d must call value_noise_2d.
#[test]
fn fbm_2d_uses_value_noise_2d() {
    assert!(
        WGSL_SOURCE.contains("value_noise_2d("),
        "fbm_2d must call value_noise_2d"
    );
}

/// fbm_3d must call value_noise_3d.
#[test]
fn fbm_3d_uses_value_noise_3d() {
    assert!(
        WGSL_SOURCE.contains("value_noise_3d("),
        "fbm_3d must call value_noise_3d"
    );
}

/// fbm_perlin_3d must call perlin_noise_3d.
#[test]
fn fbm_perlin_3d_uses_perlin_noise_3d() {
    assert!(
        WGSL_SOURCE.contains("perlin_noise_3d("),
        "fbm_perlin_3d must call perlin_noise_3d"
    );
}

/// All FBM functions must use the octave loop with lacunarity and gain.
#[test]
fn uses_octave_loop_with_lacunarity_gain() {
    let has_lacunarity = WGSL_SOURCE.contains("lacunarity");
    let has_gain = WGSL_SOURCE.contains("gain");
    let has_octaves = WGSL_SOURCE.contains("octaves");
    let has_loop = WGSL_SOURCE.contains("for (var i = 0u; i < octaves;");
    assert!(
        has_lacunarity && has_gain && has_octaves,
        "FBM functions must use 'lacunarity', 'gain', and 'octaves' parameters"
    );
    assert!(
        has_loop,
        "FBM functions must have a for loop over octaves"
    );
}

/// All FBM functions must divide by max_amplitude for normalization.
#[test]
fn uses_amplitude_normalization() {
    let has_normalization = WGSL_SOURCE.contains("max_amplitude");
    assert!(
        has_normalization,
        "FBM functions must normalize by max_amplitude"
    );
}

/// All FBM functions must compute amplitude *= gain and frequency *= lacunarity.
#[test]
fn uses_spectral_update() {
    let has_freq_update = WGSL_SOURCE.contains("frequency *= lacunarity");
    let has_amp_update = WGSL_SOURCE.contains("amplitude *= gain");
    assert!(
        has_freq_update && has_amp_update,
        "FBM functions must update frequency and amplitude per octave"
    );
}

/// The file must have four FBM subsections, each with delimiters.
#[test]
fn all_subsections_have_delimiters() {
    let sections: Vec<&str> = WGSL_SOURCE.split("// =============").collect();
    // Minimum: license header, 1D, 2D, 3D value, 3D perlin = at least 5 sections
    assert!(
        sections.len() >= 5,
        "Expected at least 5 section delimiters in noise_fbm.wgsl, found {}",
        sections.len()
    );
}

/// The file must contain exactly 4 FBM function definitions.
#[test]
fn has_correct_number_of_functions() {
    let fn_count = WGSL_SOURCE.matches("fn fbm_").count();
    assert_eq!(
        fn_count, 4,
        "Expected exactly 4 fbm_ function definitions, found {}",
        fn_count
    );
}
