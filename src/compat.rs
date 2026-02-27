// Platform abstractions for native vs WASM builds.

// ── Time ────────────────────────────────────────────────────────────────────
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
pub use web_time::{Duration, Instant};

// ── Parallel iteration ──────────────────────────────────────────────────────
// On native, delegates to rayon's `into_par_iter()`.
// On WASM (no threads), falls back to sequential `into_iter()`.

#[cfg(not(target_arch = "wasm32"))]
macro_rules! maybe_par_iter {
    ($expr:expr) => {
        $expr.into_par_iter()
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! maybe_par_iter {
    ($expr:expr) => {
        $expr.into_iter()
    };
}

pub(crate) use maybe_par_iter;
