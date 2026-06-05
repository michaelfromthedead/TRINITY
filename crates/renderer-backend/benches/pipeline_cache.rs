//! Benchmarks for the pipeline cache and sharding system.
//!
//! Run with: `cargo bench --bench pipeline_cache`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renderer_backend::pipeline::ShardedPipelineTable;

fn bench_sharded_table_shard_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShardedPipelineTable");

    // Benchmark shard index computation for different shard counts
    for shard_count in [4, 8, 16, 32, 64].iter() {
        let table = ShardedPipelineTable::new(*shard_count);

        group.bench_with_input(
            BenchmarkId::new("shard_index", shard_count),
            &table,
            |b, table| {
                let mut id = 0u32;
                b.iter(|| {
                    id = id.wrapping_add(1);
                    table.contains(black_box(id))
                })
            },
        );
    }

    group.finish();
}

fn bench_sharded_table_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShardedPipelineTable_Lookup");

    // We can't easily insert real pipelines without a GPU, so we benchmark
    // the lookup path for non-existent IDs (which exercises shard selection)
    for shard_count in [4, 16, 64].iter() {
        let table = ShardedPipelineTable::new(*shard_count);

        group.bench_with_input(
            BenchmarkId::new("contains_miss", shard_count),
            &table,
            |b, table| {
                let mut id = 0u32;
                b.iter(|| {
                    id = id.wrapping_add(1);
                    table.contains(black_box(id))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("with_pipeline_miss", shard_count),
            &table,
            |b, table| {
                let mut id = 0u32;
                b.iter(|| {
                    id = id.wrapping_add(1);
                    table.with_pipeline(black_box(id), |_| ())
                })
            },
        );
    }

    group.finish();
}

fn bench_sharded_table_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShardedPipelineTable_Stats");

    for shard_count in [4, 16, 64].iter() {
        let table = ShardedPipelineTable::new(*shard_count);

        group.bench_with_input(
            BenchmarkId::new("len", shard_count),
            &table,
            |b, table| b.iter(|| black_box(table).len()),
        );

        group.bench_with_input(
            BenchmarkId::new("is_empty", shard_count),
            &table,
            |b, table| b.iter(|| black_box(table).is_empty()),
        );

        group.bench_with_input(
            BenchmarkId::new("shard_stats", shard_count),
            &table,
            |b, table| b.iter(|| black_box(table).shard_stats()),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sharded_table_shard_index,
    bench_sharded_table_lookup,
    bench_sharded_table_stats,
);
criterion_main!(benches);
