//! Tests the bytes of old allocations of moving reallocations that jemalloc
//! passes to the `experimental.hooks.purge_before_grow` hook.

use std::cell::Cell;
use std::ffi::{c_void, CString};
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use libc::{c_uint, size_t};
use tikv_jemalloc_sys::{
    dallocx, extent_hooks_t, mallctl, mallocx, rallocx, sallocx, MALLOCX_ALIGN, MALLOCX_ARENA,
    MALLOCX_TCACHE_NONE,
};

const MIB: usize = 1 << 20;

type Hook = extern "C" fn(size_t, size_t, size_t) -> size_t;

const HOOK_NAME: &str = "experimental.hooks.purge_before_grow_v2";

static CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_MOVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static MAX_MOVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static MIN_MOVE_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
static BLOCKED_MOVERS: AtomicUsize = AtomicUsize::new(0);
static RELEASE_MOVERS: AtomicBool = AtomicBool::new(false);
static FAIL_EXTENT_ALLOC: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Movers wait in the hook until released, so their old allocations stay counted.
    static IS_MOVER: Cell<bool> = const { Cell::new(false) };
}

extern "C" fn record_hook(_size: size_t, _dirty_bytes: size_t, move_bytes: size_t) -> size_t {
    if IS_MOVER.with(|m| m.get()) {
        BLOCKED_MOVERS.fetch_add(1, Ordering::SeqCst);
        let start = Instant::now();
        while !RELEASE_MOVERS.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        return 0;
    }
    CALLS.fetch_add(1, Ordering::SeqCst);
    LAST_HOOK_SEQUENCE.store(SEQUENCE.fetch_add(1, Ordering::SeqCst), Ordering::SeqCst);
    LAST_MOVE_BYTES.store(move_bytes, Ordering::SeqCst);
    MAX_MOVE_BYTES.fetch_max(move_bytes, Ordering::SeqCst);
    MIN_MOVE_BYTES.fetch_min(move_bytes, Ordering::SeqCst);
    0
}

fn reset_records() {
    CALLS.store(0, Ordering::SeqCst);
    LAST_MOVE_BYTES.store(0, Ordering::SeqCst);
    MAX_MOVE_BYTES.store(0, Ordering::SeqCst);
    MIN_MOVE_BYTES.store(usize::MAX, Ordering::SeqCst);
}

/// The hook is global, so the tests run one at a time.
fn setup() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    write_ctl(HOOK_NAME, record_hook as Hook);
    let installed: usize = read_ctl(HOOK_NAME);
    assert_eq!(installed, record_hook as Hook as usize);
    reset_records();
    guard
}

fn read_ctl<T: Default>(name: &str) -> T {
    let c_name = CString::new(name).unwrap();
    let mut out = T::default();
    let mut len = size_of::<T>();
    let ret = unsafe {
        mallctl(
            c_name.as_ptr(),
            &mut out as *mut T as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(ret, 0, "{name:?}");
    out
}

fn write_ctl<T>(name: &str, mut value: T) {
    let c_name = CString::new(name).unwrap();
    let ret = unsafe {
        mallctl(
            c_name.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut value as *mut T as *mut c_void,
            size_of::<T>(),
        )
    };
    assert_eq!(ret, 0, "{name:?}");
}

/// Returns the unused pages of the arena to the OS.
fn purge(arena: c_uint) {
    let c_name = CString::new(arena_ctl(arena, "purge")).unwrap();
    let ret = unsafe {
        mallctl(
            c_name.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(ret, 0);
}

fn arena_ctl(arena: c_uint, name: &str) -> String {
    format!("arena.{arena}.{name}")
}

static DEFAULT_ALLOC: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn failing_alloc(
    hooks: *mut extent_hooks_t,
    new_addr: *mut c_void,
    size: size_t,
    alignment: size_t,
    zero: *mut bool,
    commit: *mut bool,
    arena_ind: c_uint,
) -> *mut c_void {
    if FAIL_EXTENT_ALLOC.load(Ordering::SeqCst) {
        return std::ptr::null_mut();
    }
    type Alloc = unsafe extern "C" fn(
        *mut extent_hooks_t,
        *mut c_void,
        size_t,
        size_t,
        *mut bool,
        *mut bool,
        c_uint,
    ) -> *mut c_void;
    let default_alloc: Alloc = std::mem::transmute(DEFAULT_ALLOC.load(Ordering::SeqCst));
    default_alloc(hooks, new_addr, size, alignment, zero, commit, arena_ind)
}

/// Creates an arena that returns unused pages to the OS right away, so that
/// every allocation needs new pages and calls the hook. Without merge, large
/// allocations of `no_merge` arenas can't grow in place.
fn new_arena(no_merge: bool) -> c_uint {
    let arena = if no_merge {
        let mut hooks = default_hooks();
        DEFAULT_ALLOC.store(hooks.alloc.unwrap() as usize, Ordering::SeqCst);
        hooks.alloc = Some(failing_alloc);
        hooks.merge = None;
        create_arena(hooks)
    } else {
        read_ctl("arenas.create")
    };
    write_ctl(&arena_ctl(arena, "dirty_decay_ms"), 0isize);
    write_ctl(&arena_ctl(arena, "muzzy_decay_ms"), 0isize);
    arena
}

fn default_hooks() -> extent_hooks_t {
    let default_hooks: *mut extent_hooks_t = read_ctl("arena.0.extent_hooks");
    unsafe { *default_hooks }
}

fn create_arena(hooks: extent_hooks_t) -> c_uint {
    let mut hooks_ptr: *mut extent_hooks_t = Box::leak(Box::new(hooks));
    let mut arena: c_uint = 0;
    let mut len = size_of::<c_uint>();
    let ret = unsafe {
        mallctl(
            b"arenas.create\0".as_ptr().cast(),
            &mut arena as *mut c_uint as *mut c_void,
            &mut len,
            &mut hooks_ptr as *mut *mut extent_hooks_t as *mut c_void,
            size_of::<*mut extent_hooks_t>(),
        )
    };
    assert_eq!(ret, 0);
    arena
}

/// Orders the calls of the purge hook and the commit hook.
static SEQUENCE: AtomicUsize = AtomicUsize::new(1);
static LAST_HOOK_SEQUENCE: AtomicUsize = AtomicUsize::new(0);
static FIRST_COMMIT_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn recording_commit(
    _hooks: *mut extent_hooks_t,
    _addr: *mut c_void,
    _size: size_t,
    _offset: size_t,
    _length: size_t,
    _arena_ind: c_uint,
) -> bool {
    let seq = SEQUENCE.fetch_add(1, Ordering::SeqCst);
    let _ = FIRST_COMMIT_SEQUENCE.compare_exchange(0, seq, Ordering::SeqCst, Ordering::SeqCst);
    false
}

/// Reports success without decommitting, so the pages stay usable.
unsafe extern "C" fn fake_decommit(
    _hooks: *mut extent_hooks_t,
    _addr: *mut c_void,
    _size: size_t,
    _offset: size_t,
    _length: size_t,
    _arena_ind: c_uint,
) -> bool {
    false
}

fn flags(arena: c_uint) -> i32 {
    MALLOCX_ARENA(arena as usize) | MALLOCX_TCACHE_NONE
}

/// The bytes of old allocations that moving reallocations hold right now.
fn current_move_bytes(probe_arena: c_uint) -> usize {
    purge(probe_arena);
    reset_records();
    unsafe {
        let p = mallocx(MIB, flags(probe_arena));
        assert!(!p.is_null());
        dallocx(p, flags(probe_arena));
    }
    assert!(
        CALLS.load(Ordering::SeqCst) > 0,
        "the probe did not call the hook"
    );
    LAST_MOVE_BYTES.load(Ordering::SeqCst)
}

fn fill(p: *mut c_void, len: usize, seed: u8) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(p.cast::<u8>(), len) };
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(seed);
    }
}

fn check(p: *const c_void, len: usize, seed: u8) {
    let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) };
    for (i, b) in bytes.iter().enumerate() {
        assert_eq!(
            *b,
            (i as u8).wrapping_mul(31).wrapping_add(seed),
            "byte {i}"
        );
    }
}

/// Grows `old_size` to `new_size` with a move, returning the largest bytes of
/// moving reallocations the hook saw.
fn moved_growth(arena: c_uint, old_size: usize, new_size: usize) -> usize {
    unsafe {
        let old = mallocx(old_size, flags(arena));
        assert!(!old.is_null());
        fill(old, old_size, 1);
        let old_usize = sallocx(old, 0);
        purge(arena);
        reset_records();
        let new = rallocx(old, new_size, flags(arena));
        assert!(!new.is_null());
        assert_ne!(new, old, "the reallocation did not move");
        assert!(
            CALLS.load(Ordering::SeqCst) > 0,
            "the replacement did not call the hook: {} -> {}",
            old_size,
            new_size
        );
        check(new, old_size.min(new_size), 1);
        dallocx(new, flags(arena));
        let seen = MAX_MOVE_BYTES.load(Ordering::SeqCst);
        assert!(seen == 0 || seen == old_usize, "{} != {}", seen, old_usize);
        seen
    }
}

#[test]
fn moved_growth_counts_old_allocation() {
    let _guard = setup();
    let arena = new_arena(true);
    let probe_arena = new_arena(false);

    let large_min: usize = read_ctl("arenas.lextent.0.size");

    let seen = moved_growth(arena, 3 * MIB, 8 * MIB);
    assert!(seen >= 3 * MIB);
    // Shrinking can also move, e.g. out of the large size classes.
    let seen = moved_growth(arena, large_min, 1 << 10);
    assert_eq!(seen, large_min);
    // Small allocations are not counted.
    assert_eq!(moved_growth(arena, large_min / 2, 4 * MIB), 0);

    assert_eq!(current_move_bytes(probe_arena), 0);
}

#[test]
fn realignment_counts_old_allocation() {
    let _guard = setup();
    let arena = new_arena(false);
    let probe_arena = new_arena(false);
    unsafe {
        let old = mallocx(2 * MIB, flags(arena));
        assert!(!old.is_null());
        let old_usize = sallocx(old, 0);
        // Ask for an alignment that the old allocation does not have.
        let alignment = 2usize << (old as usize).trailing_zeros();
        reset_records();
        let new = rallocx(old, 2 * MIB, flags(arena) | MALLOCX_ALIGN(alignment));
        assert!(!new.is_null());
        assert_eq!(new as usize % alignment, 0);
        assert_eq!(MAX_MOVE_BYTES.load(Ordering::SeqCst), old_usize);
        dallocx(new, flags(arena));
    }
    assert_eq!(current_move_bytes(probe_arena), 0);
}

#[test]
fn in_place_growth_is_not_counted() {
    let _guard = setup();
    let arena = new_arena(false);
    let probe_arena = new_arena(false);
    unsafe {
        // Shrinking returns the tail to the OS, so growing again takes new
        // pages right behind the allocation.
        let p = mallocx(8 * MIB, flags(arena));
        assert!(!p.is_null());
        assert_eq!(rallocx(p, 2 * MIB, flags(arena)), p);
        purge(arena);
        reset_records();
        assert_eq!(rallocx(p, 6 * MIB, flags(arena)), p);
        assert!(CALLS.load(Ordering::SeqCst) > 0);
        assert_eq!(MAX_MOVE_BYTES.load(Ordering::SeqCst), 0);
        dallocx(p, flags(arena));
    }
    assert_eq!(current_move_bytes(probe_arena), 0);
}

#[test]
fn failed_expansion_does_not_call_hook() {
    let _guard = setup();
    let arena = new_arena(false);
    unsafe {
        let p = mallocx(2 * MIB, flags(arena));
        assert!(!p.is_null());
        // Takes the pages right behind `p`, so `p` can't grow in place.
        let neighbor = mallocx(2 * MIB, flags(arena));
        assert!(!neighbor.is_null());
        let old_usize = sallocx(p, 0);
        purge(arena);
        reset_records();
        let new = rallocx(p, 4 * MIB, flags(arena));
        assert!(!new.is_null());
        assert_ne!(new, p, "the reallocation did not move");
        // Only the replacement asks to purge, with the old allocation counted.
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(MIN_MOVE_BYTES.load(Ordering::SeqCst), old_usize);
        dallocx(new, flags(arena));
        dallocx(neighbor, flags(arena));
    }
}

#[test]
fn purge_comes_before_committing_growth() {
    let _guard = setup();
    // Purged pages of this arena are decommitted, so growing into them commits.
    let mut hooks = default_hooks();
    hooks.commit = Some(recording_commit);
    hooks.decommit = Some(fake_decommit);
    let arena = create_arena(hooks);
    write_ctl(&arena_ctl(arena, "dirty_decay_ms"), 0isize);
    write_ctl(&arena_ctl(arena, "muzzy_decay_ms"), 0isize);
    unsafe {
        let p = mallocx(8 * MIB, flags(arena));
        assert!(!p.is_null());
        assert_eq!(rallocx(p, 2 * MIB, flags(arena)), p);
        purge(arena);
        reset_records();
        FIRST_COMMIT_SEQUENCE.store(0, Ordering::SeqCst);
        assert_eq!(rallocx(p, 6 * MIB, flags(arena)), p);
        let commit = FIRST_COMMIT_SEQUENCE.load(Ordering::SeqCst);
        assert_ne!(commit, 0, "growing did not commit");
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        assert!(LAST_HOOK_SEQUENCE.load(Ordering::SeqCst) < commit);
        dallocx(p, flags(arena));
    }
}

#[test]
fn failed_move_stops_counting() {
    let _guard = setup();
    let arena = new_arena(true);
    let probe_arena = new_arena(false);
    unsafe {
        let old = mallocx(MIB, flags(arena));
        assert!(!old.is_null());
        fill(old, MIB, 7);
        let old_usize = sallocx(old, 0);
        reset_records();
        FAIL_EXTENT_ALLOC.store(true, Ordering::SeqCst);
        let new = rallocx(old, 64 * MIB, flags(arena));
        FAIL_EXTENT_ALLOC.store(false, Ordering::SeqCst);
        assert!(new.is_null());
        assert_eq!(MAX_MOVE_BYTES.load(Ordering::SeqCst), old_usize);
        check(old, MIB, 7);
        assert_eq!(current_move_bytes(probe_arena), 0);
        dallocx(old, flags(arena));
    }
}

#[test]
fn concurrent_moves_are_all_counted() {
    const THREADS: usize = 4;
    let _guard = setup();
    let probe_arena = new_arena(false);
    BLOCKED_MOVERS.store(0, Ordering::SeqCst);
    RELEASE_MOVERS.store(false, Ordering::SeqCst);
    let start = Barrier::new(THREADS);

    let old_usizes: usize = std::thread::scope(|s| {
        let movers: Vec<_> = (0..THREADS)
            .map(|i| {
                let start = &start;
                s.spawn(move || unsafe {
                    // Bound to different arenas, the movers count in different shards.
                    let arena = new_arena(true);
                    write_ctl("thread.arena", arena);
                    let old_size = (i + 1) * MIB;
                    let old = mallocx(old_size, flags(arena));
                    assert!(!old.is_null());
                    fill(old, old_size, i as u8);
                    let old_usize = sallocx(old, 0);
                    start.wait();
                    IS_MOVER.with(|m| m.set(true));
                    let new = rallocx(old, 8 * MIB, flags(arena));
                    IS_MOVER.with(|m| m.set(false));
                    assert!(!new.is_null() && new != old);
                    check(new, old_size, i as u8);
                    dallocx(new, flags(arena));
                    old_usize
                })
            })
            .collect();

        let wait_start = Instant::now();
        while BLOCKED_MOVERS.load(Ordering::SeqCst) < THREADS {
            assert!(wait_start.elapsed() < Duration::from_secs(10));
            std::thread::yield_now();
        }
        let during = current_move_bytes(probe_arena);
        RELEASE_MOVERS.store(true, Ordering::SeqCst);
        let old_usizes = movers.into_iter().map(|m| m.join().unwrap()).sum();
        assert_eq!(during, old_usizes);
        old_usizes
    });
    assert!(old_usizes >= 10 * MIB);
    assert_eq!(current_move_bytes(probe_arena), 0);
}

#[test]
fn unversioned_hook_name_is_rejected() {
    let _guard = setup();
    // A two-argument hook from before the third argument must not install.
    let name = CString::new("experimental.hooks.purge_before_grow").unwrap();
    let mut hook = record_hook as Hook;
    let ret = unsafe {
        mallctl(
            name.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut hook as *mut Hook as *mut c_void,
            size_of::<Hook>(),
        )
    };
    assert_eq!(ret, libc::ENOENT);
}
