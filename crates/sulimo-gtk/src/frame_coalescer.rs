//! Frame-budget token coalescing for 60 FPS responsive UI rendering (§7 M5, A14).
//!
//! Accumulates high-speed streaming tokens and flushes chunks to the GTK text buffer
//! at most once per display frame (~16 ms budget), preventing widget layout thrashing
//! and maintaining smooth scrolling even at 100+ tok/s.

use std::time::{Duration, Instant};

/// Coalesces stream tokens to maintain a target 60 FPS frame rate.
pub struct FrameCoalescer {
    pending_buffer: String,
    last_flush: Instant,
    target_frame_budget: Duration,
    total_tokens_coalesced: usize,
    flushes_count: usize,
}

impl Default for FrameCoalescer {
    fn default() -> Self {
        Self::new(Duration::from_millis(16))
    }
}

impl FrameCoalescer {
    /// Creates a coalescer with the specified frame budget (default ~16.6 ms for 60 FPS).
    pub fn new(target_frame_budget: Duration) -> Self {
        Self {
            pending_buffer: String::with_capacity(1024),
            last_flush: Instant::now(),
            target_frame_budget,
            total_tokens_coalesced: 0,
            flushes_count: 0,
        }
    }

    /// Pushes a token to the coalescing buffer. Returns text to flush if the frame budget expired.
    pub fn push_token(&mut self, token: &str) -> Option<String> {
        self.pending_buffer.push_str(token);
        self.total_tokens_coalesced += 1;

        if self.last_flush.elapsed() >= self.target_frame_budget {
            self.flush()
        } else {
            None
        }
    }

    /// Flushes any pending tokens if the frame budget duration has elapsed.
    pub fn flush_if_needed(&mut self) -> Option<String> {
        if !self.pending_buffer.is_empty() && self.last_flush.elapsed() >= self.target_frame_budget
        {
            self.flush()
        } else {
            None
        }
    }

    /// Forces a flush of all remaining pending tokens (e.g. at end of generation).
    pub fn flush(&mut self) -> Option<String> {
        if self.pending_buffer.is_empty() {
            None
        } else {
            let out = std::mem::take(&mut self.pending_buffer);
            self.last_flush = Instant::now();
            self.flushes_count += 1;
            Some(out)
        }
    }

    /// Whether there are pending tokens waiting in the buffer.
    pub fn has_pending(&self) -> bool {
        !self.pending_buffer.is_empty()
    }

    /// Statistics on tokens and flushes.
    pub fn stats(&self) -> (usize, usize) {
        (self.total_tokens_coalesced, self.flushes_count)
    }

    /// Resets the coalescer state for a new conversation turn.
    pub fn reset(&mut self) {
        self.pending_buffer.clear();
        self.last_flush = Instant::now();
        self.total_tokens_coalesced = 0;
        self.flushes_count = 0;
    }
}
