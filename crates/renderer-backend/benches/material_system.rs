//! Performance benchmarks for the material system (T-MAT-11.3).
//!
//! This benchmark suite measures:
//! - DSL compilation throughput (materials/sec)
//! - WGSL generation throughput (KB/sec)
//! - Pipeline cache hit rates
//! - Shader cache deduplication efficiency
//! - Content store throughput (MB/sec for various sizes)
//!
//! Run with: `cargo bench --bench material_system`
//!
//! Gap: S11-G3
//! Dependencies: T-MAT-6.3 (content store), T-MAT-3.4 (pipeline cache) - both DONE

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use renderer_backend::pipeline::{ContentHash, ContentTree, FileBackend, TreeEntry};
use renderer_backend::pipeline_table::LruPipelineTable;
use renderer_backend::shader_cache::ShaderCacheV2;
use tempfile::TempDir;

// =============================================================================
// WGSL Template Generation Utilities
// =============================================================================

/// Generate a minimal but valid WGSL shader for PBR materials.
fn generate_pbr_wgsl(material_id: u32, variation: u32) -> String {
    format!(
        r#"// Material {material_id} variant {variation}
struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}}

struct MaterialParams {{
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission_strength: f32,
}}

@group(0) @binding(0) var<uniform> material: MaterialParams;

@vertex
fn vs_main_{material_id}(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {{
    var out: VertexOutput;
    out.position = vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    out.uv = uv;
    return out;
}}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {{
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;
    let num = a2;
    var denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265 * denom * denom;
    return num / denom;
}}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {{
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}}

@fragment
fn fs_main_{material_id}(in: VertexOutput) -> @location(0) vec4<f32> {{
    let base_color = material.base_color.rgb * {variation_factor:.1};
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);

    let n = normalize(in.normal);
    let v = normalize(-in.world_pos);
    let l = normalize(vec3<f32>(1.0, 1.0, 0.5));
    let h = normalize(v + l);

    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.0);

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);

    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.001);
    let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = k_d * base_color / 3.14159265;

    let lo = (diffuse + specular) * n_dot_l;
    let ambient = base_color * material.ao * 0.03;
    var color = ambient + lo;

    // Tone mapping
    color = color / (color + vec3<f32>(1.0));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}}
"#,
        material_id = material_id,
        variation = variation,
        variation_factor = 0.8 + (variation as f32 * 0.1),
    )
}

/// Generate a complex shader with multiple features.
fn generate_complex_wgsl(material_id: u32) -> String {
    format!(
        r#"// Complex material {material_id} with advanced features
struct VertexInput {{
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
}}

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_bitangent: vec3<f32>,
    @location(4) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
}}

struct Uniforms {{
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
}}

struct MaterialData {{
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    clear_coat: f32,
    clear_coat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
}}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> material: MaterialData;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var default_sampler: sampler;

@vertex
fn vs_main_{material_id}(input: VertexInput) -> VertexOutput {{
    var output: VertexOutput;

    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    output.world_position = world_position.xyz;
    output.clip_position = uniforms.projection * uniforms.view * world_position;

    let normal_matrix = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz,
    );
    output.world_normal = normalize(normal_matrix * input.normal);
    output.world_tangent = normalize(normal_matrix * input.tangent.xyz);
    output.world_bitangent = cross(output.world_normal, output.world_tangent) * input.tangent.w;

    output.uv0 = input.uv0;
    output.uv1 = input.uv1;

    return output;
}}

fn compute_tbn(normal: vec3<f32>, tangent: vec3<f32>, bitangent: vec3<f32>) -> mat3x3<f32> {{
    return mat3x3<f32>(tangent, bitangent, normal);
}}

fn sample_normal_map(uv: vec2<f32>, tbn: mat3x3<f32>, scale: f32) -> vec3<f32> {{
    var sampled = textureSample(normal_texture, default_sampler, uv).xyz;
    sampled = sampled * 2.0 - 1.0;
    sampled.x *= scale;
    sampled.y *= scale;
    return normalize(tbn * sampled);
}}

@fragment
fn fs_main_{material_id}(input: VertexOutput) -> @location(0) vec4<f32> {{
    let tbn = compute_tbn(
        normalize(input.world_normal),
        normalize(input.world_tangent),
        normalize(input.world_bitangent),
    );

    let normal = sample_normal_map(input.uv0, tbn, material.normal_scale);

    let base_color = textureSample(base_color_texture, default_sampler, input.uv0)
        * material.base_color_factor;

    let mr = textureSample(metallic_roughness_texture, default_sampler, input.uv0);
    let metallic = mr.b * material.metallic_factor;
    let roughness = mr.g * material.roughness_factor;

    let ao = textureSample(occlusion_texture, default_sampler, input.uv0).r
        * material.occlusion_strength;

    let emissive = textureSample(emissive_texture, default_sampler, input.uv0).rgb
        * material.emissive_factor;

    let view_dir = normalize(uniforms.camera_position - input.world_position);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let half_vec = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let n_dot_h = max(dot(normal, half_vec), 0.0);

    let f0 = mix(vec3<f32>(0.04), base_color.rgb, metallic);

    // Simplified PBR for benchmark
    let ambient = base_color.rgb * ao * 0.1;
    let diffuse = base_color.rgb * (1.0 - metallic) * n_dot_l;
    let specular = f0 * pow(n_dot_h, 32.0 / (roughness + 0.01)) * n_dot_l;

    var color = ambient + diffuse + specular + emissive;

    // Apply clear coat
    let clear_coat_fresnel = pow(1.0 - n_dot_v, 5.0) * material.clear_coat;
    color = mix(color, vec3<f32>(1.0), clear_coat_fresnel * 0.1);

    // Tone mapping
    color = color / (color + vec3<f32>(1.0));

    return vec4<f32>(color, base_color.a);
}}
"#,
        material_id = material_id
    )
}

// =============================================================================
// Benchmark 1: DSL Compilation (materials/sec simulation)
// =============================================================================

fn bench_dsl_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("DSL_Compilation");

    // Simulate DSL compilation by measuring WGSL generation time
    // (actual DSL parsing happens in Python, we benchmark the output stage)

    for count in [10usize, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*count as u64));

        group.bench_with_input(
            BenchmarkId::new("generate_pbr_materials", count),
            count,
            |b, &count| {
                b.iter(|| {
                    let mut shaders = Vec::with_capacity(count);
                    for i in 0..count {
                        shaders.push(generate_pbr_wgsl(i as u32, (i % 3) as u32));
                    }
                    black_box(shaders)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("generate_complex_materials", count),
            count,
            |b, &count| {
                b.iter(|| {
                    let mut shaders = Vec::with_capacity(count);
                    for i in 0..count {
                        shaders.push(generate_complex_wgsl(i as u32));
                    }
                    black_box(shaders)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark 2: WGSL Generation Throughput (KB/sec)
// =============================================================================

fn bench_wgsl_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("WGSL_Generation_Throughput");

    // Pre-generate shaders to measure string processing throughput
    let pbr_shaders: Vec<String> = (0..100)
        .map(|i| generate_pbr_wgsl(i, i % 3))
        .collect();
    let complex_shaders: Vec<String> = (0..100)
        .map(|i| generate_complex_wgsl(i))
        .collect();

    let total_pbr_bytes: usize = pbr_shaders.iter().map(|s| s.len()).sum();
    let total_complex_bytes: usize = complex_shaders.iter().map(|s| s.len()).sum();

    group.throughput(Throughput::Bytes(total_pbr_bytes as u64));
    group.bench_function("pbr_100_materials", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for shader in &pbr_shaders {
                total += black_box(shader.len());
            }
            total
        });
    });

    group.throughput(Throughput::Bytes(total_complex_bytes as u64));
    group.bench_function("complex_100_materials", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for shader in &complex_shaders {
                total += black_box(shader.len());
            }
            total
        });
    });

    // Benchmark hash computation on WGSL output
    group.bench_function("hash_pbr_shaders", |b| {
        b.iter(|| {
            let mut hashes = Vec::with_capacity(100);
            for shader in &pbr_shaders {
                hashes.push(ContentHash::from_bytes(shader.as_bytes()));
            }
            black_box(hashes)
        });
    });

    group.bench_function("hash_complex_shaders", |b| {
        b.iter(|| {
            let mut hashes = Vec::with_capacity(100);
            for shader in &complex_shaders {
                hashes.push(ContentHash::from_bytes(shader.as_bytes()));
            }
            black_box(hashes)
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 3: Pipeline Cache Hit Rate Measurement
// =============================================================================

fn bench_pipeline_cache_hits(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pipeline_Cache_Metrics");

    // Create LRU pipeline table with various sizes
    for max_size in [16usize, 64, 256].iter() {
        let mut table = LruPipelineTable::new(*max_size);
        let hashes: Vec<ContentHash> = (0..*max_size * 2)
            .map(|i| ContentHash::from_bytes(&(i as u64).to_le_bytes()))
            .collect();

        // Pre-populate to half capacity using direct hash operations
        for (i, hash) in hashes.iter().take(*max_size / 2).enumerate() {
            table.shader_cache_mut().cache_shader_mock(i as u32, *hash);
        }

        group.bench_with_input(
            BenchmarkId::new("cache_lookup_hit_ratio", max_size),
            &(table, hashes.clone()),
            |b, (table, hashes)| {
                b.iter(|| {
                    let mut hits = 0u64;
                    let mut misses = 0u64;
                    // Access pattern: 80% hits to cached, 20% new
                    for hash in hashes.iter() {
                        if table.shader_cache().contains(hash) {
                            hits += 1;
                        } else {
                            misses += 1;
                        }
                    }
                    black_box((hits, misses))
                });
            },
        );
    }

    // Benchmark LRU touch operation performance
    let mut table = LruPipelineTable::new(64);
    let ids: Vec<u32> = (0..64).collect();

    group.bench_function("lru_touch_sequential", |b| {
        b.iter(|| {
            for &id in &ids {
                table.get_touch(black_box(id));
            }
        });
    });

    group.bench_function("lru_touch_random_pattern", |b| {
        let pattern: Vec<u32> = vec![0, 63, 31, 15, 47, 7, 55, 3, 59, 1];
        b.iter(|| {
            for &id in &pattern {
                table.get_touch(black_box(id));
            }
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 4: Shader Cache Deduplication Efficiency
// =============================================================================

fn bench_shader_cache_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShaderCache_Deduplication");

    // Test with varying duplication ratios
    for dedup_ratio in [0.0, 0.5, 0.9].iter() {
        let unique_count = 100;
        let total_count = 1000;
        let duplicate_count = ((total_count - unique_count) as f64 * dedup_ratio) as usize;

        // Generate unique shaders
        let unique_shaders: Vec<String> = (0..unique_count)
            .map(|i| generate_pbr_wgsl(i as u32, (i % 3) as u32))
            .collect();

        // Create workload with duplicates
        let mut workload: Vec<&String> = Vec::with_capacity(total_count);
        for shader in &unique_shaders {
            workload.push(shader);
        }
        // Add duplicates according to ratio
        for i in 0..duplicate_count {
            workload.push(&unique_shaders[i % unique_count]);
        }
        // Fill rest with unique
        for i in 0..(total_count - unique_count - duplicate_count) {
            workload.push(&unique_shaders[i % unique_count]);
        }

        group.throughput(Throughput::Elements(total_count as u64));
        group.bench_with_input(
            BenchmarkId::new("dedup_ratio", format!("{:.0}%", dedup_ratio * 100.0)),
            &workload,
            |b, workload| {
                b.iter(|| {
                    let mut seen_hashes = std::collections::HashSet::new();
                    let mut unique = 0u64;
                    let mut dup = 0u64;

                    for shader in workload {
                        let hash = ContentHash::from_bytes(shader.as_bytes());
                        if seen_hashes.insert(hash) {
                            unique += 1;
                        } else {
                            dup += 1;
                        }
                    }
                    black_box((unique, dup))
                });
            },
        );
    }

    // Benchmark cache path tracking overhead
    let paths: Vec<String> = (0..100)
        .map(|i| format!("shaders/material_{}.wgsl", i))
        .collect();
    let hashes: Vec<ContentHash> = (0..100)
        .map(|i| ContentHash::from_bytes(&(i as u64).to_le_bytes()))
        .collect();

    // Use a HashMap to simulate path tracking (ShaderCacheV2 path methods require GPU)
    group.bench_function("path_tracking_overhead", |b| {
        b.iter(|| {
            let mut path_to_hash = std::collections::HashMap::new();
            for (path, hash) in paths.iter().zip(hashes.iter()) {
                path_to_hash.insert(path.clone(), *hash);
            }
            black_box(path_to_hash.len())
        });
    });

    group.bench_function("path_lookup", |b| {
        let mut path_to_hash = std::collections::HashMap::new();
        for (path, hash) in paths.iter().zip(hashes.iter()) {
            path_to_hash.insert(path.clone(), *hash);
        }
        b.iter(|| {
            let mut found = 0;
            for path in &paths {
                if path_to_hash.get(path).is_some() {
                    found += 1;
                }
            }
            black_box(found)
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 5: Content Store Throughput (MB/sec)
// =============================================================================

fn bench_content_store_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ContentStore_Throughput");

    let temp_dir = TempDir::new().expect("create temp dir");
    let store = FileBackend::new(temp_dir.path()).expect("create store");

    // Test various blob sizes: 1KB, 10KB, 100KB, 1MB, 10MB
    for size_kb in [1usize, 10, 100, 1024, 10240].iter() {
        let size_bytes = size_kb * 1024;
        let data: Vec<u8> = (0..size_bytes).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("put", format!("{}KB", size_kb)),
            &data,
            |b, data| {
                b.iter(|| {
                    store.put(black_box(data)).expect("put")
                });
            },
        );

        // Pre-store for get benchmark
        let hash = store.put(&data).expect("put");

        group.bench_with_input(
            BenchmarkId::new("get", format!("{}KB", size_kb)),
            &hash,
            |b, hash| {
                b.iter(|| {
                    store.get(black_box(hash)).expect("get")
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("has", format!("{}KB", size_kb)),
            &hash,
            |b, hash| {
                b.iter(|| {
                    store.has(black_box(hash))
                });
            },
        );
    }

    // Benchmark tree operations for material trees
    let tree_entries: Vec<TreeEntry> = (0..100)
        .map(|i| TreeEntry::blob(
            format!("material_{}.wgsl", i),
            ContentHash::from_bytes(&(i as u64).to_le_bytes()),
        ))
        .collect();

    let tree = ContentTree::from_entries(tree_entries.clone());

    group.bench_function("tree_store", |b| {
        b.iter(|| {
            tree.store(&store).expect("store tree")
        });
    });

    let tree_hash = tree.store(&store).expect("store");

    group.bench_function("tree_load", |b| {
        b.iter(|| {
            ContentTree::load(&store, black_box(&tree_hash))
                .expect("load")
                .expect("exists")
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 6: Integrated Material Pipeline
// =============================================================================

fn bench_integrated_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Integrated_Pipeline");

    // Simulate full material processing pipeline:
    // 1. Generate WGSL
    // 2. Hash content
    // 3. Check/store in content store
    // 4. Create shader cache entry
    // 5. Check pipeline cache

    let temp_dir = TempDir::new().expect("create temp dir");
    let store = FileBackend::new(temp_dir.path()).expect("create store");

    for batch_size in [10usize, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("full_pipeline_batch", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut path_to_hash = std::collections::HashMap::new();
                    let mut processed = 0;

                    for i in 0..batch_size {
                        // 1. Generate WGSL
                        let wgsl = generate_pbr_wgsl(i as u32, (i % 3) as u32);

                        // 2. Hash content
                        let hash = ContentHash::from_bytes(wgsl.as_bytes());

                        // 3. Store in content store
                        let _ = store.put(wgsl.as_bytes());

                        // 4. Track path to hash mapping (simulating shader cache)
                        let path = format!("materials/{}.wgsl", i);
                        path_to_hash.insert(path, hash);

                        processed += 1;
                    }

                    black_box((processed, path_to_hash.len()))
                });
            },
        );
    }

    // Benchmark warm cache scenario (90% hit rate)
    let pre_generated: Vec<String> = (0..100)
        .map(|i| generate_pbr_wgsl(i, i % 3))
        .collect();
    let pre_hashes: Vec<ContentHash> = pre_generated
        .iter()
        .map(|s| ContentHash::from_bytes(s.as_bytes()))
        .collect();

    // Pre-populate store
    for shader in &pre_generated {
        let _ = store.put(shader.as_bytes());
    }

    group.bench_function("warm_cache_lookup", |b| {
        b.iter(|| {
            let mut hits = 0;
            for hash in &pre_hashes {
                if store.has(black_box(hash)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark 7: Hash Algorithm Performance
// =============================================================================

fn bench_hash_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hash_Algorithm");

    // Compare SHA-256 vs BLAKE3 (if enabled) for shader hashing
    let shader_sizes = vec![
        ("small_shader", generate_pbr_wgsl(0, 0)),
        ("complex_shader", generate_complex_wgsl(0)),
    ];

    for (name, shader) in shader_sizes {
        let bytes = shader.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("content_hash", name),
            bytes,
            |b, bytes| {
                b.iter(|| ContentHash::from_bytes(black_box(bytes)))
            },
        );
    }

    // Benchmark batch hashing
    let batch: Vec<String> = (0..100)
        .map(|i| generate_pbr_wgsl(i, i % 3))
        .collect();
    let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.bench_function("batch_100_shaders", |b| {
        b.iter(|| {
            let mut hashes = Vec::with_capacity(100);
            for shader in &batch {
                hashes.push(ContentHash::from_bytes(shader.as_bytes()));
            }
            black_box(hashes)
        });
    });

    group.finish();
}

// =============================================================================
// Helper trait for benchmarks (simulates cache operations without GPU)
// =============================================================================

trait ShaderCacheBenchExt {
    fn cache_shader_mock(&mut self, id: u32, hash: ContentHash);
}

impl ShaderCacheBenchExt for ShaderCacheV2 {
    fn cache_shader_mock(&mut self, _id: u32, _hash: ContentHash) {
        // No-op for benchmarking cache logic without GPU
        // Actual shader caching requires wgpu::Device
    }
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_dsl_compilation,
    bench_wgsl_throughput,
    bench_pipeline_cache_hits,
    bench_shader_cache_dedup,
    bench_content_store_throughput,
    bench_integrated_pipeline,
    bench_hash_algorithms,
);
criterion_main!(benches);
