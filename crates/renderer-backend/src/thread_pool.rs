// SPDX-License-Identifier: MIT
//
// thread_pool.rs — Priority-aware thread pool (T-CORE-3.1)

use crossbeam::deque::{Injector, Steal};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    Critical = 0,
    High = 1,
    AboveNormal = 2,
    Normal = 3,
    BelowNormal = 4,
    Background = 5,
}

impl Default for Priority {
    fn default() -> Self { Priority::Normal }
}

type Task = Box<dyn FnOnce() + Send + 'static>;

struct Inner {
    running: AtomicBool,
    pending: AtomicUsize,
    num_workers: usize,
    injectors: [Injector<Task>; 6],
    cv: Condvar,
    mu: Mutex<()>,
    handles: Mutex<Option<Vec<thread::JoinHandle<()>>>>,
}

pub struct ThreadPool {
    inner: Arc<Inner>,
}

impl ThreadPool {
    pub fn new(num_workers: usize) -> Self {
        let n = if num_workers == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get()).unwrap_or(4)
        } else {
            num_workers
        };

        let inner = Arc::new(Inner {
            running: AtomicBool::new(true),
            pending: AtomicUsize::new(0),
            num_workers: n,
            injectors: [
                Injector::new(), Injector::new(), Injector::new(),
                Injector::new(), Injector::new(), Injector::new(),
            ],
            cv: Condvar::new(),
            mu: Mutex::new(()),
            handles: Mutex::new(None),
        });

        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let inner = Arc::clone(&inner);
            handles.push(thread::spawn(move || worker_loop(inner)));
        }
        *inner.handles.lock() = Some(handles);

        ThreadPool { inner }
    }

    pub fn new_auto() -> Self { Self::new(0) }

    pub fn spawn<F>(&self, priority: Priority, task: F)
    where F: FnOnce() + Send + 'static
    {
        self.inner.pending.fetch_add(1, Ordering::Relaxed);
        self.inner.injectors[priority as usize].push(Box::new(task));
        self.inner.cv.notify_one();
    }

    pub fn pending(&self) -> usize {
        self.inner.pending.load(Ordering::Relaxed)
    }

    pub fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    pub fn shutdown(&self) {
        self.inner.running.store(false, Ordering::Relaxed);
        self.inner.cv.notify_all();
        if let Some(handles) = self.inner.handles.lock().take() {
            for h in handles { let _ = h.join(); }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) { self.shutdown(); }
}

fn worker_loop(inner: Arc<Inner>) {
    loop {
        // Try all injectors, highest priority first.
        let mut found = false;
        for prio in 0..6 {
            loop {
                match inner.injectors[prio].steal() {
                    Steal::Success(task) => {
                        inner.pending.fetch_sub(1, Ordering::Relaxed);
                        task();
                        found = true;
                        break;
                    }
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        if found { continue; }

        // Nothing available.
        if !inner.running.load(Ordering::Relaxed) {
            // Final drain.
            for prio in 0..6 {
                loop {
                    match inner.injectors[prio].steal() {
                        Steal::Success(task) => {
                            inner.pending.fetch_sub(1, Ordering::Relaxed); task();
                        }
                        Steal::Retry => continue,
                        Steal::Empty => break,
                    }
                }
            }
            return;
        }

        let mut guard = inner.mu.lock();
        if inner.running.load(Ordering::Relaxed) {
            let _ = inner.cv.wait_for(&mut guard, std::time::Duration::from_millis(100));
        }
    }
}

/// Splits `range` into chunks and distributes them as Normal-priority tasks to `pool`.
///
/// `chunk_size` controls the number of indices per chunk:
///   - `0` → auto-sizes to `ceil(range.len() / pool.num_workers())`
///   - `> 0` → uses the given size (clamped to at least 1)
///
/// Blocks the caller until every chunk has completed.
pub fn parallel_for<F: Fn(usize) + Send + Sync + 'static>(
    pool: &ThreadPool,
    range: std::ops::Range<usize>,
    chunk_size: usize,
    f: F,
) {
    let len = range.len();
    if len == 0 {
        return;
    }

    let cs = if chunk_size == 0 {
        let workers = pool.num_workers().max(1);
        (len + workers - 1) / workers
    } else {
        chunk_size.max(1)
    };

    let num_chunks = (len + cs - 1) / cs;
    let completed = Arc::new(AtomicUsize::new(0));
    let cv = Arc::new(Condvar::new());
    let done = Arc::new(Mutex::new(false));
    let f = Arc::new(f);

    for chunk_idx in 0..num_chunks {
        let start = range.start + chunk_idx * cs;
        let end = std::cmp::min(range.start + (chunk_idx + 1) * cs, range.end);
        let f = Arc::clone(&f);
        let completed = Arc::clone(&completed);
        let cv = Arc::clone(&cv);
        let done = Arc::clone(&done);

        pool.spawn(Priority::Normal, move || {
            for i in start..end {
                f(i);
            }
            let prev = completed.fetch_add(1, Ordering::Relaxed);
            if prev + 1 == num_chunks {
                *done.lock() = true;
                cv.notify_all();
            }
        });
    }

    // Wait until all chunks signal completion.
    let mut guard = done.lock();
    let _ = cv.wait_while(&mut guard, |d: &mut bool| !*d);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_spawn() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..10 {
            let c = Arc::clone(&counter);
            pool.spawn(Priority::Normal, move || { c.fetch_add(1, Ordering::Relaxed); });
        }
        pool.shutdown();
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn auto_detect() {
        let pool = ThreadPool::new_auto();
        pool.shutdown();
    }

    // =========================================================================
    // parallel_for tests (25+)
    // =========================================================================

    #[test]
    fn parallel_for_single_element() {
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..1, 0, move |i| {
            r.lock().push(i);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, vec![0]);
    }

    #[test]
    fn parallel_for_zero_elements() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..0, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn parallel_for_large_range() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..10000, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 10000);
    }

    #[test]
    fn parallel_for_chunk_size_1() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..100, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn parallel_for_chunk_size_larger_than_range() {
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..5, 100, move |i| {
            r.lock().push(i);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn parallel_for_explicit_chunk_size() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..100, 10, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn parallel_for_auto_chunk_size() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..100, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn parallel_for_closure_captures_values() {
        let pool = ThreadPool::new(4);
        let multiplier = 3;
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..10, 0, move |i| {
            r.lock().push(i * multiplier);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, vec![0, 3, 6, 9, 12, 15, 18, 21, 24, 27]);
    }

    #[test]
    fn parallel_for_non_zero_start() {
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 50..60, 0, move |i| {
            r.lock().push(i);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, (50..60).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_for_thread_safety_no_data_race() {
        let pool = ThreadPool::new(8);
        let sum = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sum);
        parallel_for(&pool, 0..1000, 0, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        // Sum of 0..1000 = 999*1000/2 = 499500
        assert_eq!(sum.load(Ordering::Relaxed), 499500);
    }

    #[test]
    fn parallel_for_all_indices_visited_exactly_once() {
        let pool = ThreadPool::new(4);
        let visited = Arc::new(Mutex::new(vec![false; 100]));
        let v = Arc::clone(&visited);
        parallel_for(&pool, 0..100, 0, move |i| {
            let mut guard = v.lock();
            assert!(!guard[i], "Index {} visited more than once", i);
            guard[i] = true;
        });
        let guard = visited.lock();
        for (i, &b) in guard.iter().enumerate() {
            assert!(b, "Index {} was never visited", i);
        }
    }

    #[test]
    fn parallel_for_single_worker_pool() {
        let pool = ThreadPool::new(1);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..50, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn parallel_for_many_workers_few_items() {
        let pool = ThreadPool::new(16);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..3, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn parallel_for_uneven_chunk_distribution() {
        // 7 items with chunk_size=3 => 3 chunks: [0,1,2], [3,4,5], [6]
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..7, 3, move |i| {
            r.lock().push(i);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn parallel_for_very_small_chunk_size() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..1000, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn parallel_for_accumulate_into_vec() {
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::with_capacity(100)));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..100, 0, move |i| {
            r.lock().push(i * i);
        });
        let mut v = results.lock().clone();
        v.sort();
        let expected: Vec<_> = (0..100).map(|i| i * i).collect();
        assert_eq!(v, expected);
    }

    #[test]
    fn parallel_for_sequential_pools() {
        // Use separate pools for nested-style work to avoid deadlock
        let pool1 = ThreadPool::new(4);
        let pool2 = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        // First level
        let c1 = Arc::clone(&counter);
        parallel_for(&pool1, 0..4, 0, move |i| {
            c1.fetch_add(i * 10, Ordering::Relaxed);
        });

        // Second level (sequential, not nested)
        let c2 = Arc::clone(&counter);
        parallel_for(&pool2, 0..10, 0, move |_| {
            c2.fetch_add(1, Ordering::Relaxed);
        });

        // 0+10+20+30 + 10 = 60 + 10 = 70
        assert_eq!(counter.load(Ordering::Relaxed), 70);
    }

    #[test]
    fn parallel_for_preserves_order_within_chunk() {
        // Each chunk processes indices sequentially
        let pool = ThreadPool::new(1); // Single worker to guarantee chunk order
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..10, 5, move |i| {
            r.lock().push(i);
        });
        let v = results.lock().clone();
        // With 1 worker and chunk_size=5, order should be sequential
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn parallel_for_empty_range_with_offset() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 100..100, 0, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn parallel_for_stress_concurrent_access() {
        let pool = ThreadPool::new(8);
        let data = Arc::new(Mutex::new(vec![0i64; 1000]));
        let d = Arc::clone(&data);
        parallel_for(&pool, 0..1000, 10, move |i| {
            let mut guard = d.lock();
            guard[i] = i as i64 + 1;
        });
        let guard = data.lock();
        for (i, &val) in guard.iter().enumerate() {
            assert_eq!(val, i as i64 + 1);
        }
    }

    #[test]
    fn parallel_for_modifies_shared_state_atomically() {
        let pool = ThreadPool::new(4);
        let max_seen = Arc::new(AtomicUsize::new(0));
        let m = Arc::clone(&max_seen);
        parallel_for(&pool, 1..101, 0, move |i| {
            loop {
                let current = m.load(Ordering::Relaxed);
                if i <= current {
                    break;
                }
                if m.compare_exchange_weak(current, i, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    break;
                }
            }
        });
        assert_eq!(max_seen.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn parallel_for_with_two_elements() {
        let pool = ThreadPool::new(4);
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for(&pool, 0..2, 0, move |i| {
            r.lock().push(i);
        });
        let mut v = results.lock().clone();
        v.sort();
        assert_eq!(v, vec![0, 1]);
    }

    #[test]
    fn parallel_for_chunk_size_equals_range_length() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..50, 50, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn parallel_for_prime_number_elements() {
        // 97 elements with various chunk sizes
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(&pool, 0..97, 7, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 97);
    }

    #[test]
    fn parallel_for_multiple_sequential_calls() {
        let pool = ThreadPool::new(4);

        for round in 0..5 {
            let counter = Arc::new(AtomicUsize::new(0));
            let c = Arc::clone(&counter);
            parallel_for(&pool, 0..100, 0, move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(counter.load(Ordering::Relaxed), 100, "Round {} failed", round);
        }
    }

    #[test]
    fn parallel_for_high_contention() {
        let pool = ThreadPool::new(8);
        let shared = Arc::new(Mutex::new(0u64));
        let s = Arc::clone(&shared);
        parallel_for(&pool, 0..10000, 1, move |i| {
            let mut guard = s.lock();
            *guard += i as u64;
        });
        // Sum of 0..10000 = 9999*10000/2 = 49995000
        assert_eq!(*shared.lock(), 49995000);
    }
}
