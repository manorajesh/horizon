//! Fork-join for the render thread's per-frame work: a few helper threads that run
//! alongside it at its own priority, asleep between frames.
//!
//! The chunk workers (`jobs`) keep every core busy while terrain loads and run below the
//! render thread, so splitting a frame's work over them would make the frame wait behind
//! generation. These helpers are favoured as the render thread is (user-interactive quality
//! of service on macOS, where they then stay on the performance cores) and sleep on a
//! condition variable when there is no frame work, so an idle game spends nothing on them.
//!
//! [`for_each_part`] runs `f(part)` for every part in `0..parts`, on the calling thread and
//! the helpers, and returns once every part is done; `f` may borrow from the caller's stack,
//! because no helper touches it after that. A panic in `f` is raised on the caller.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

/// One call's work, shared with the helpers while it runs.
#[derive(Clone, Copy)]
struct Job {
    /// `&(dyn Fn(usize) + Sync)` with its lifetime erased; valid until the call that set it
    /// sees `busy` back at zero.
    f: *const (dyn Fn(usize) + Sync + 'static),
    parts: usize,
    /// The next part to claim, on the caller's stack.
    next: *const AtomicUsize,
}
// SAFETY: the pointers are only dereferenced while `for_each_part` blocks on `busy`, and they
// point to a `Sync` closure and an atomic.
unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    /// Bumped for every job, so a helper runs each one at most once.
    generation: u64,
    /// Helpers still inside the current job.
    busy: usize,
    /// A helper's panic in the current job, raised on the caller.
    panic: Option<Box<dyn std::any::Any + Send>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

struct Helpers {
    shared: Arc<Shared>,
    threads: usize,
    /// One job at a time: a second caller (another window's renderer) waits its turn.
    turn: Mutex<()>,
}

/// How many helpers to start: on Apple silicon the performance cores less the render thread
/// (`hw.perflevel0.physicalcpu`: 10 on an M4 Pro, 4 on an M1), elsewhere half the threads
/// less one; at most 4, as the work split is a few milliseconds a frame. None on 4 threads or
/// fewer, where the chunk workers and the render thread already share every core.
fn helper_count() -> usize {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads <= 4 {
        return 0;
    }
    let cores = performance_cores().unwrap_or(threads / 2);
    cores.saturating_sub(1).clamp(1, 4)
}

#[cfg(target_vendor = "apple")]
fn performance_cores() -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    // SAFETY: reads one integer into `value`, whose size is passed.
    let ok = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut value).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } == 0;
    (ok && value > 0).then_some(value as usize)
}

#[cfg(not(target_vendor = "apple"))]
fn performance_cores() -> Option<usize> {
    None
}

fn helpers() -> &'static Helpers {
    static HELPERS: OnceLock<Helpers> = OnceLock::new();
    HELPERS.get_or_init(|| {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                job: None,
                generation: 0,
                busy: 0,
                panic: None,
            }),
            wake: Condvar::new(),
            done: Condvar::new(),
        });
        let mut threads = 0;
        for i in 0..helper_count() {
            let shared = Arc::clone(&shared);
            let started = std::thread::Builder::new()
                .name(format!("Frame helper {i}"))
                .spawn(move || helper(&shared));
            if started.is_ok() {
                threads += 1;
            }
        }
        Helpers {
            shared,
            threads,
            turn: Mutex::new(()),
        }
    })
}

fn helper(shared: &Shared) {
    crate::favour_current();
    let mut seen = 0;
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        state = shared
            .wake
            .wait_while(state, |s| s.generation == seen || s.job.is_none())
            .unwrap_or_else(|e| e.into_inner());
        seen = state.generation;
        let Some(job) = state.job else { continue };
        state.busy += 1;
        drop(state);
        // SAFETY: the caller keeps `f` and `next` alive until `busy` is back at zero, which
        // this helper's decrement below comes before.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe { claim(job) }));
        state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(panic) = result {
            state.panic.get_or_insert(panic);
        }
        state.busy -= 1;
        if state.busy == 0 {
            shared.done.notify_all();
        }
    }
}

/// Runs parts of `job` until none are left.
///
/// # Safety
/// `job`'s pointers must be valid for the whole call.
unsafe fn claim(job: Job) {
    // SAFETY: valid for this call (the caller's contract).
    let (f, next) = unsafe { (&*job.f, &*job.next) };
    loop {
        let part = next.fetch_add(1, Ordering::Relaxed);
        if part >= job.parts {
            return;
        }
        f(part);
    }
}

/// Runs `f(part)` for each part in `0..parts`, spread over the calling thread and the frame
/// helpers, and returns when all are done. Each part runs exactly once, in no set order and on
/// no set thread; with no helpers (4 threads or fewer) the caller runs them all in order.
pub fn for_each_part(parts: usize, f: &(dyn Fn(usize) + Sync)) {
    if parts == 0 {
        return;
    }
    let helpers = helpers();
    if parts == 1 || helpers.threads == 0 {
        (0..parts).for_each(f);
        return;
    }
    let _turn = helpers.turn.lock().unwrap_or_else(|e| e.into_inner());
    let next = AtomicUsize::new(0);
    // SAFETY: only the lifetime is erased; `f` outlives every use, as this function waits
    // below for every helper to leave the job before it returns or unwinds.
    let erased: *const (dyn Fn(usize) + Sync + 'static) =
        unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), _>(f as *const _) };
    let shared = &helpers.shared;
    {
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.job = Some(Job {
            f: erased,
            parts,
            next: &next,
        });
        state.generation = state.generation.wrapping_add(1);
        state.panic = None;
        shared.wake.notify_all();
    }
    // The caller works too; a panic here still waits for the helpers before it unwinds.
    // SAFETY: `f` and `next` live on this frame until the wait below.
    let own = catch_unwind(AssertUnwindSafe(|| unsafe {
        claim(Job {
            f: erased,
            parts,
            next: &next,
        })
    }));
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    // No helper may start this job once the caller has seen every part claimed; those
    // still in it finish their part. Taking the job away first closes that window.
    state.job = None;
    state = shared
        .done
        .wait_while(state, |s| s.busy > 0)
        .unwrap_or_else(|e| e.into_inner());
    let panic = state.panic.take();
    drop(state);
    if let Err(panic) = own {
        std::panic::resume_unwind(panic);
    }
    if let Some(panic) = panic {
        std::panic::resume_unwind(panic);
    }
}

/// Splits `items` into at most `parts` runs of about equal length, in order: run `i` is
/// `ranges(len, parts)[i]`. With the outputs of [`for_each_part`] kept per run and joined in
/// run order, a split loop gives what the plain loop gives.
pub fn ranges(len: usize, parts: usize) -> Vec<std::ops::Range<usize>> {
    let parts = parts.clamp(1, len.max(1));
    (0..parts)
        .map(|i| (len * i / parts)..(len * (i + 1) / parts))
        .collect()
}

/// How many runs to split `len` items into: enough for the helpers and the caller to share
/// them with some slack, none smaller than `min_run` items.
pub fn runs_for(len: usize, min_run: usize) -> usize {
    let workers = helpers().threads + 1;
    (len / min_run.max(1)).clamp(1, workers * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_part_runs_once() {
        for parts in [0, 1, 2, 7, 64, 1000] {
            let hits: Vec<AtomicU64> = (0..parts).map(|_| AtomicU64::new(0)).collect();
            for_each_part(parts, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "{parts}"
            );
        }
    }

    #[test]
    fn borrows_the_callers_data_and_joins_in_order() {
        let input: Vec<u64> = (0..10_000).collect();
        let runs = ranges(input.len(), 13);
        let outputs: Vec<Mutex<Vec<u64>>> = runs.iter().map(|_| Mutex::new(Vec::new())).collect();
        for_each_part(runs.len(), &|i| {
            let out: Vec<u64> = input[runs[i].clone()].iter().map(|v| v * 3).collect();
            *outputs[i].lock().unwrap() = out;
        });
        let joined: Vec<u64> = outputs
            .into_iter()
            .flat_map(|m| m.into_inner().unwrap())
            .collect();
        let plain: Vec<u64> = input.iter().map(|v| v * 3).collect();
        assert_eq!(joined, plain);
    }

    #[test]
    fn a_panic_reaches_the_caller_and_the_helpers_stay_usable() {
        let caught = catch_unwind(|| {
            for_each_part(32, &|i| {
                if i == 17 {
                    panic!("part 17");
                }
            })
        });
        assert!(caught.is_err());
        let count = AtomicU64::new(0);
        for_each_part(100, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn ranges_cover_everything_in_order() {
        for (len, parts) in [(0, 4), (1, 4), (10, 3), (10, 10), (10, 20), (1000, 7)] {
            let r = ranges(len, parts);
            assert_eq!(r.first().map(|r| r.start), Some(0));
            assert_eq!(r.last().map(|r| r.end), Some(len));
            assert!(
                r.windows(2).all(|w| w[0].end == w[1].start),
                "{len} {parts}"
            );
        }
    }
}
