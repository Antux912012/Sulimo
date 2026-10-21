//! Physical-core pinned thread pool with low-power bounded spinning ($BATT, A10).
//!
//! Spawns exactly one worker thread per physical core (excluding SMT/HyperThreading siblings),
//! pins each worker using `sched_setaffinity(2)`, and executes parallel matrix-vector
//! multiplications using a low-latency barrier that spins for ≤ 2 µs before parking
//! to prevent L3 cache thrashing and preserve battery life.

use crate::topology::CpuTopology;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Task assigned to a worker thread.
type WorkFn = Box<dyn Fn(usize, usize) + Send + Sync>;

/// State shared between the pool master and worker threads.
struct WorkerState {
    work: Mutex<Option<WorkFn>>,
    job_id: AtomicUsize,
    completed: AtomicUsize,
    shutdown: AtomicBool,
}

/// A worker thread handle.
struct Worker {
    handle: Option<JoinHandle<()>>,
    thread: thread::Thread,
}

/// Thread pool pinned to physical execution cores.
pub struct ThreadPool {
    workers: Vec<Worker>,
    state: Arc<WorkerState>,
    num_threads: usize,
}

impl ThreadPool {
    /// Creates a thread pool with one worker per physical core, pinned to hardware CPUs.
    pub fn new(threads: Option<usize>) -> Self {
        let topology = CpuTopology::detect().ok();
        let pin_targets = match &topology {
            Some(topo) => topo.decode_pin_targets(),
            None => (0..num_cpus::get_physical().max(1)).collect(),
        };

        let num_threads = threads
            .unwrap_or(pin_targets.len())
            .min(pin_targets.len())
            .max(1);

        let state = Arc::new(WorkerState {
            work: Mutex::new(None),
            job_id: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });

        let mut workers = Vec::with_capacity(num_threads);

        for thread_idx in 0..num_threads {
            let state_clone = Arc::clone(&state);
            let cpu_target = pin_targets[thread_idx % pin_targets.len()];

            let builder = thread::Builder::new().name(format!("sulimo-cpu-{thread_idx}"));
            let (tx, rx) = std::sync::mpsc::sync_channel::<thread::Thread>(1);

            let handle = builder
                .spawn(move || {
                    // Pin current worker thread to the target physical CPU (A10)
                    pin_current_thread(cpu_target);

                    tx.send(thread::current()).unwrap();

                    worker_loop(thread_idx, num_threads, state_clone);
                })
                .expect("failed to spawn worker thread");

            let thread = rx.recv().unwrap();
            workers.push(Worker {
                handle: Some(handle),
                thread,
            });
        }

        Self {
            workers,
            state,
            num_threads,
        }
    }

    /// Number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Executes a parallel loop over `total_items`, splitting work evenly across workers.
    pub fn execute_parallel<F>(&mut self, total_items: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if total_items == 0 {
            return;
        }

        if self.num_threads <= 1 || total_items == 1 {
            f(0, total_items);
            return;
        }

        {
            let mut guard = self.state.work.lock().unwrap();
            *guard = Some(Box::new(f));
        }

        let _current_job = self.state.job_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.state.completed.store(0, Ordering::SeqCst);

        // Unpark all workers
        for w in &self.workers {
            w.thread.unpark();
        }

        // Master thread assists by waiting with bounded spin then park
        wait_bounded_spin(&self.state.completed, self.num_threads);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.state.shutdown.store(true, Ordering::SeqCst);
        for w in &self.workers {
            w.thread.unpark();
        }
        for w in &mut self.workers {
            if let Some(handle) = w.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Worker loop: bounded spin for ≤ 2 µs, then park until unparked by master.
fn worker_loop(thread_idx: usize, num_threads: usize, state: Arc<WorkerState>) {
    let mut last_seen_job = 0;

    while !state.shutdown.load(Ordering::Relaxed) {
        let current_job = state.job_id.load(Ordering::Acquire);
        if current_job > last_seen_job {
            last_seen_job = current_job;

            {
                let guard = state.work.lock().unwrap();
                if let Some(work) = guard.as_ref() {
                    work(thread_idx, num_threads);
                }
            }

            state.completed.fetch_add(1, Ordering::Release);
        }

        // Bounded spin loop (≤ ~2 µs = ~500-1000 spin iterations)
        let mut spins = 0;
        while state.job_id.load(Ordering::Relaxed) == last_seen_job
            && !state.shutdown.load(Ordering::Relaxed)
            && spins < 800
        {
            std::hint::spin_loop();
            spins += 1;
        }

        // If no work arrived during the bounded spin, park the OS thread to enter deep idle (C-states)
        if state.job_id.load(Ordering::Relaxed) == last_seen_job
            && !state.shutdown.load(Ordering::Relaxed)
        {
            thread::park();
        }
    }
}

/// Waits for `target` completions using bounded spinning before yielding/parking.
fn wait_bounded_spin(counter: &AtomicUsize, target: usize) {
    let mut spins = 0;
    while counter.load(Ordering::Acquire) < target {
        if spins < 800 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            thread::yield_now();
        }
    }
}

/// Pins the calling thread to a single logical CPU target using Linux `sched_setaffinity`.
pub fn pin_current_thread(cpu_id: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu_id, &mut set);
        libc::sched_setaffinity(
            0, // Current thread
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu_id;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_parallel_sum() {
        let mut pool = ThreadPool::new(Some(2));
        assert_eq!(pool.num_threads(), 2);

        let data = Arc::new(vec![1u32; 1000]);
        let sum_atomic = Arc::new(AtomicUsize::new(0));

        let data_ref = Arc::clone(&data);
        let sum_ref = Arc::clone(&sum_atomic);

        pool.execute_parallel(1000, move |idx, num_workers| {
            let chunk_size = 1000 / num_workers;
            let start = idx * chunk_size;
            let end = if idx == num_workers - 1 {
                1000
            } else {
                start + chunk_size
            };

            let mut local_sum = 0;
            for i in start..end {
                local_sum += data_ref[i] as usize;
            }
            sum_ref.fetch_add(local_sum, Ordering::SeqCst);
        });

        assert_eq!(sum_atomic.load(Ordering::SeqCst), 1000);
    }
}
