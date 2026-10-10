//! Dedicated worker threads that run queries off the JS event loop.
//!
//! Why not libuv's `AsyncTask` pool: the engine needs `QUERY_THREAD_STACK_SIZE`
//! of stack under every query ("a Rust stack overflow aborts the process"), and
//! a libuv worker's stack depends on the platform and `RLIMIT_STACK`. Owning the
//! threads makes that a guarantee instead of an assumption.
//!
//! Two lanes. Reads run on a small pool (`min(4, cores)` threads). Everything
//! that takes the graph's write lock (auto-commit writes, commits, checkpoints,
//! `close`) runs on one dedicated writer thread. The lock already serialises
//! those jobs, so a single thread costs no parallelism, and it keeps each
//! commit's copy-on-write fork and the drop of the superseded graph on one
//! thread's allocator heap. Rotating them across four workers made every
//! commit free memory another thread's heap owned, which measured 25-30 %
//! slower on a 1M-node delete and 20-30 % on a 100k-node relationship write.
//!
//! A third lane runs the auto-commit writes the engine's group-commit queue
//! takes (`Session::auto_commit_is_grouped`: plain data writes on a graph
//! durable at `full`). Those serialise inside the session and share one log
//! barrier per batch, but only when several are in flight at once, which one
//! writer thread cannot offer. The lane grows a thread per queued job up to
//! [`GROUP_THREADS`], so a lone writer costs one thread and a burst of 64 can
//! share a barrier. The writer thread still runs every other write; the two
//! are kept apart by `Inner::exclusive` / `Inner::grouped`.
//!
//! Each lane's queue holds [`QUEUE_CAPACITY`] jobs. A call that finds it full is
//! rejected `QueueFull` ([`FullPolicy::Reject`], the default) or parked in a FIFO
//! waiting list ([`FullPolicy::Wait`], `onQueueFull: 'wait'`) that feeds the
//! queue as workers take jobs. A parked call holds its promise, its closure and
//! the caller's arguments and nothing else; how many there are is the caller's
//! to bound by awaiting.
//!
//! A job returns a [`Settle`] closure; the worker hands it to a napi `JsDeferred`,
//! which runs it back on the JS thread (building JS values needs the `Env`) and
//! resolves the promise with its result.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread;

use kglite::api::cypher::{with_query_warning_sink, QueryWarningSink};
use kglite::api::session::QUERY_THREAD_STACK_SIZE;
use napi::bindgen_prelude::ToNapiValue;
use napi::{sys, Env, JsDeferred};

use crate::errors::{rejected_promise, JsErr, JsRes, CODE_QUEUE_FULL};

/// A napi value handed back to a promise, resolved or already-rejected.
pub struct RawJs(pub sys::napi_value);

impl ToNapiValue for RawJs {
    unsafe fn to_napi_value(_env: sys::napi_env, val: Self) -> napi::Result<sys::napi_value> {
        Ok(val.0)
    }
}

/// Runs on the JS thread and produces the promise's settlement: `Ok` resolves it
/// with the value, `Err` rejects it with a coded `Error`.
pub type Settle = Box<dyn FnOnce(Env) -> JsRes<sys::napi_value> + Send>;

type Deferred = JsDeferred<RawJs, Box<dyn FnOnce(Env) -> napi::Result<RawJs> + Send>>;

/// Per-lane queue ceiling. A burst past it is refused rather than buffered without bound.
const QUEUE_CAPACITY: usize = 4096;

/// What a call does when its lane's queue is full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FullPolicy {
    /// Reject the call with `QueueFull`.
    #[default]
    Reject,
    /// Park the call until a slot frees, in arrival order.
    Wait,
}

/// Which thread set a job runs on.
#[derive(Clone, Copy)]
enum Lane {
    /// Queries that only read, spread over the worker pool.
    Read,
    /// Jobs that take the write lock, on the single writer thread.
    Write,
    /// Grouped auto-commit writes, on a pool that grows with the burst.
    Group,
}

type Job = Box<dyn FnOnce() + Send>;

struct Queue {
    state: Mutex<LaneState>,
    ready: Condvar,
    /// Thread ceiling and name prefix, for a lane that starts threads on demand.
    grow: Option<(usize, &'static str)>,
}

struct LaneState {
    /// Jobs a worker may take, at most [`QUEUE_CAPACITY`].
    jobs: VecDeque<Job>,
    /// Calls parked by [`FullPolicy::Wait`]. Non-empty only while `jobs` is at
    /// capacity: a worker moves the oldest one in under the same lock that
    /// frees the slot, so arrival order is preserved.
    waiting: VecDeque<Job>,
    /// Workers started, and how many of them wait for a job.
    threads: usize,
    idle: usize,
}

fn spawn_worker(queue: &Arc<Queue>, name: &str, n: usize) -> std::io::Result<()> {
    let q = Arc::clone(queue);
    thread::Builder::new()
        .name(format!("{name}-{n}"))
        .stack_size(worker_stack_size())
        .spawn(move || worker(&q))
        .map(|_| ())
}

fn start_lane(threads: usize, name: &'static str, grow: Option<usize>) -> Arc<Queue> {
    let queue = Arc::new(Queue {
        state: Mutex::new(LaneState {
            jobs: VecDeque::new(),
            waiting: VecDeque::new(),
            threads,
            idle: 0,
        }),
        ready: Condvar::new(),
        grow: grow.map(|max| (max, name)),
    });
    for n in 0..threads {
        spawn_worker(&queue, name, n).expect("spawn kglite-node worker");
    }
    queue
}

/// Ceiling on the grouped-write lane: `KGLITE_NODE_GROUP_WRITERS`, else 64.
///
/// Each thread parks inside the engine's queue until its batch's barrier
/// completes, so the ceiling is the largest batch the binding can offer.
fn group_threads() -> usize {
    std::env::var("KGLITE_NODE_GROUP_WRITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(GROUP_THREADS)
}

const GROUP_THREADS: usize = 64;

fn lane_queue(lane: Lane) -> &'static Arc<Queue> {
    static READ: OnceLock<Arc<Queue>> = OnceLock::new();
    static WRITE: OnceLock<Arc<Queue>> = OnceLock::new();
    static GROUP: OnceLock<Arc<Queue>> = OnceLock::new();
    match lane {
        Lane::Read => READ.get_or_init(|| start_lane(worker_count(), "kglite-node", None)),
        Lane::Write => WRITE.get_or_init(|| start_lane(1, "kglite-node-writer", None)),
        Lane::Group => {
            GROUP.get_or_init(|| start_lane(0, "kglite-node-group", Some(group_threads())))
        }
    }
}

/// Stack for every worker: the engine's `QUERY_THREAD_STACK_SIZE`, in every profile.
///
/// The parser's deepest accepted nesting (511 levels) fits it in a debug build
/// because planning runs under the engine's stack guard.
pub const fn worker_stack_size() -> usize {
    QUERY_THREAD_STACK_SIZE
}

/// `KGLITE_NODE_THREADS`, else `min(4, available cores)`.
fn worker_count() -> usize {
    std::env::var("KGLITE_NODE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| thread::available_parallelism().map_or(2, |n| n.get().min(4)))
}

fn worker(queue: &Queue) {
    // The structured `warnings` array on every result is the only channel: the
    // engine's stderr echo would write into the host process's stderr.
    with_query_warning_sink(QueryWarningSink::Silent, || worker_loop(queue));
}

fn worker_loop(queue: &Queue) {
    loop {
        let job = {
            let mut state = queue.state.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    if let Some(parked) = state.waiting.pop_front() {
                        state.jobs.push_back(parked);
                    }
                    break job;
                }
                state.idle += 1;
                state = queue
                    .ready
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                state.idle -= 1;
            }
        };
        job();
    }
}

fn enqueue(lane: Lane, policy: FullPolicy, job: Job) -> Result<(), Job> {
    let queue = lane_queue(lane);
    let queue: &Arc<Queue> = queue;
    let mut state = queue.state.lock().unwrap_or_else(PoisonError::into_inner);
    if state.jobs.len() >= QUEUE_CAPACITY {
        if policy == FullPolicy::Reject {
            return Err(job);
        }
        state.waiting.push_back(job);
        return Ok(());
    }
    state.jobs.push_back(job);
    // A growing lane starts a worker when the jobs outnumber the idle ones.
    if let Some((max, name)) = queue.grow {
        if state.jobs.len() > state.idle && state.threads < max {
            let n = state.threads;
            if spawn_worker(queue, name, n).is_ok() {
                state.threads += 1;
            }
        }
    }
    drop(state);
    queue.ready.notify_one();
    Ok(())
}

/// Wrap a settlement so a panic or napi failure while building the JS value still
/// settles the promise: a panic escaping a threadsafe-function callback aborts Node.
fn guarded(settle: Settle) -> Box<dyn FnOnce(Env) -> napi::Result<RawJs> + Send> {
    Box::new(move |env: Env| {
        let raw = env.raw();
        let outcome = catch_unwind(AssertUnwindSafe(|| settle(env)));
        let result = match outcome {
            Ok(Ok(value)) => return Ok(RawJs(value)),
            Ok(Err(e)) => e,
            Err(p) => JsErr::from_panic(p.as_ref()),
        };
        match rejected_promise(raw, &result) {
            Ok(promise) => Ok(RawJs(promise)),
            Err(e) => Err(napi::Error::from_reason(e.message)),
        }
    })
}

fn settle_now(deferred: Deferred, settle: Settle) {
    deferred.resolve(guarded(settle));
}

/// Run `work` on a read-pool thread and return the promise it settles.
///
/// `work` runs inside `catch_unwind`; a panic rejects with `Internal`. A full
/// queue rejects `QueueFull`; [`spawn_for`] takes the graph's policy instead.
pub fn spawn<'e>(
    env: &'e Env,
    work: impl FnOnce() -> Settle + Send + 'static,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    spawn_on(Lane::Read, FullPolicy::Reject, env, work)
}

/// [`spawn`] on the writer thread, for jobs that take the graph's write lock
/// or mutate through a transaction.
pub fn spawn_write<'e>(
    policy: FullPolicy,
    env: &'e Env,
    work: impl FnOnce() -> Settle + Send + 'static,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    spawn_on(Lane::Write, policy, env, work)
}

/// [`spawn`] on the grouped-write lane: auto-commit writes the engine batches.
pub fn spawn_group<'e>(
    policy: FullPolicy,
    env: &'e Env,
    work: impl FnOnce() -> Settle + Send + 'static,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    spawn_on(Lane::Group, policy, env, work)
}

/// [`spawn_write`] when `write`, else [`spawn`] with the same full-queue `policy`.
pub fn spawn_for<'e>(
    write: bool,
    policy: FullPolicy,
    env: &'e Env,
    work: impl FnOnce() -> Settle + Send + 'static,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    spawn_on(
        if write { Lane::Write } else { Lane::Read },
        policy,
        env,
        work,
    )
}

fn spawn_on<'e>(
    lane: Lane,
    policy: FullPolicy,
    env: &'e Env,
    work: impl FnOnce() -> Settle + Send + 'static,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    let (deferred, promise) =
        env.create_deferred::<RawJs, Box<dyn FnOnce(Env) -> napi::Result<RawJs> + Send>>()?;
    // The deferred is needed by both the job and the full-queue path, so it
    // lives in a slot the first to run takes.
    let slot = Arc::new(Mutex::new(Some(deferred)));
    let job_slot = Arc::clone(&slot);
    let job: Job = Box::new(move || {
        let settle: Settle = match catch_unwind(AssertUnwindSafe(work)) {
            Ok(s) => s,
            Err(p) => {
                let e = JsErr::from_panic(p.as_ref());
                Box::new(move |_| Err(e))
            }
        };
        if let Some(d) = job_slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            settle_now(d, settle);
        }
    });
    if enqueue(lane, policy, job).is_err() {
        if let Some(d) = slot.lock().unwrap_or_else(PoisonError::into_inner).take() {
            let e = JsErr::new(CODE_QUEUE_FULL, "the query queue is full; retry later");
            settle_now(d, Box::new(move |_| Err(e)));
        }
    }
    Ok(promise)
}

/// A promise that is already settled with `settle`'s outcome (argument errors).
pub fn settled<'e>(
    env: &'e Env,
    settle: Settle,
) -> napi::Result<napi::bindgen_prelude::Object<'e>> {
    let (deferred, promise) =
        env.create_deferred::<RawJs, Box<dyn FnOnce(Env) -> napi::Result<RawJs> + Send>>()?;
    settle_now(deferred, settle);
    Ok(promise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kglite::api::session::{execute_read, open_path, ExecuteOptions, OpenSpec};
    use kglite::api::Value;
    use std::collections::HashMap;

    /// The deepest list literal the parser accepts runs end to end through the
    /// public entry point on a thread of exactly the pool's stack size.
    #[test]
    fn deepest_accepted_query_fits_the_worker_stack() {
        thread::Builder::new()
            .stack_size(worker_stack_size())
            .spawn(|| {
                let spec = OpenSpec {
                    lease_timeout: None,
                    ..OpenSpec::writer()
                };
                let opened = open_path(std::path::Path::new("kglite-node-stack-test.kgl"), &spec)
                    .expect("in-memory open");
                let params: HashMap<String, Value> = HashMap::new();
                let mut opts = ExecuteOptions::eager(&params);
                opts.streaming = true;
                let depth = 511;
                let q = format!("RETURN {}1{} AS v", "[".repeat(depth), "]".repeat(depth));
                execute_read(&opened.session.snapshot(), &q, &opts).expect("query within budget");
            })
            .expect("spawn")
            .join()
            .expect("deep query overflowed the worker stack");
    }
}
