//! Benchmarks for the content-addressed storage system.
//!
//! Run with: `cargo bench --bench content_store`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use renderer_backend::pipeline::{ContentHash, ContentTree, FileBackend, TreeEntry};
use tempfile::TempDir;

fn bench_content_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("ContentHash");

    // Benchmark hashing different data sizes
    for size in [64, 256, 1024, 4096, 16384, 65536].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("from_bytes", size), &data, |b, data| {
            b.iter(|| ContentHash::from_bytes(black_box(data)))
        });
    }

    group.finish();
}

fn bench_file_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("FileBackend");

    // Setup temporary store
    let temp_dir = TempDir::new().expect("create temp dir");
    let store = FileBackend::new(temp_dir.path()).expect("create store");

    // Benchmark put/get for different sizes
    for size in [256, 1024, 4096, 16384].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("put", size), &data, |b, data| {
            b.iter(|| store.put(black_box(data)).expect("put"))
        });

        // Put the data once so we can benchmark get
        let hash = store.put(&data).expect("put");
        group.bench_with_input(BenchmarkId::new("get", size), &hash, |b, hash| {
            b.iter(|| store.get(black_box(hash)).expect("get"))
        });

        group.bench_with_input(BenchmarkId::new("has", size), &hash, |b, hash| {
            b.iter(|| store.has(black_box(hash)))
        });
    }

    group.finish();
}

fn bench_content_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("ContentTree");

    // Create trees of different sizes
    for size in [10usize, 100, 1000].iter() {
        let entries: Vec<TreeEntry> = (0..*size)
            .map(|i: usize| {
                let hash = ContentHash::from_bytes(&i.to_le_bytes());
                TreeEntry::blob(format!("file_{}.txt", i), hash)
            })
            .collect();

        let tree = ContentTree::from_entries(entries.clone());

        group.bench_with_input(
            BenchmarkId::new("from_entries", size),
            &entries,
            |b, entries| b.iter(|| ContentTree::from_entries(black_box(entries.clone()))),
        );

        group.bench_with_input(BenchmarkId::new("compute_hash", size), &tree, |b, tree| {
            b.iter(|| tree.compute_hash())
        });

        let lookup_name = format!("file_{}.txt", size / 2);
        group.bench_with_input(
            BenchmarkId::new("get", size),
            &(tree, lookup_name),
            |b, (tree, name)| b.iter(|| tree.get(name)),
        );
    }

    // Benchmark diff
    let small_tree1 = ContentTree::from_entries(
        (0..100)
            .map(|i| TreeEntry::blob(format!("f{}", i), ContentHash::from_bytes(&[i as u8])))
            .collect(),
    );
    let small_tree2 = ContentTree::from_entries(
        (0..100)
            .map(|i| {
                let data = if i % 10 == 0 { [i as u8, 1] } else { [i as u8, 0] };
                TreeEntry::blob(format!("f{}", i), ContentHash::from_bytes(&data))
            })
            .collect(),
    );

    group.bench_function("diff_100_entries", |b| {
        b.iter(|| small_tree1.diff(&small_tree2))
    });

    group.finish();
}

fn bench_structural_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("StructuralSharing");

    // Benchmark with_entry (immutable update)
    for size in [10usize, 100, 1000].iter() {
        let entries: Vec<TreeEntry> = (0..*size)
            .map(|i: usize| {
                let hash = ContentHash::from_bytes(&i.to_le_bytes());
                TreeEntry::blob(format!("file_{}.txt", i), hash)
            })
            .collect();
        let tree = ContentTree::from_entries(entries);
        let new_entry = TreeEntry::blob("new_file.txt", ContentHash::from_bytes(b"new"));

        group.bench_with_input(
            BenchmarkId::new("with_entry", size),
            &(tree.clone(), new_entry.clone()),
            |b, (tree, entry)| b.iter(|| tree.with_entry(entry.clone())),
        );

        let remove_name = "file_0.txt".to_string();
        group.bench_with_input(
            BenchmarkId::new("without_entry", size),
            &(tree, remove_name),
            |b, (tree, name)| b.iter(|| tree.without_entry(name)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_content_hash,
    bench_file_backend,
    bench_content_tree,
    bench_structural_sharing,
);
criterion_main!(benches);
