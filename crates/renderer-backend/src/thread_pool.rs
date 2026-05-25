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
}
