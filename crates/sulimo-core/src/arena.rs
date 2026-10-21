//! Page-aligned static arena allocator with transparent hugepage backing (@C:MMAP, $ZCOPY).
//!
//! Provides a single contiguous anonymous memory mapping with 2 MiB base alignment
//! and `MADV_HUGEPAGE` advice. Tensors are allocated via bump pointer with strict
//! 4096-byte page alignment and 64-byte size padding, enabling zero-copy Intel OpenCL
//! `CL_MEM_USE_HOST_PTR` buffers without page faults (A4, A5).

use std::ptr::NonNull;
use thiserror::Error;

/// Page alignment enforced for all arena allocations (Intel OpenCL zero-copy requirement).
pub const ARENA_ALLOC_ALIGN: usize = 4096;

/// Size padding multiple enforced for all arena allocations (Intel cacheline / sub-group requirement).
pub const ARENA_SIZE_ALIGN: usize = 64;

/// Base alignment for Transparent Huge Pages (2 MiB).
pub const HUGEPAGE_ALIGN: usize = 2 * 1024 * 1024;

/// Errors returned during arena creation or allocation.
#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("failed to allocate anonymous mmap of {size} bytes: {source}")]
    MmapFailed {
        size: usize,
        #[source]
        source: std::io::Error,
    },
    #[error("out of arena memory: requested {needed} bytes (padded {padded}), but only {available} bytes remain")]
    OutOfMemory {
        needed: usize,
        padded: usize,
        available: usize,
    },
}

/// Logical memory regions within the static arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaRegion {
    /// Static model weights (Q4_0 SoA, F16, F32).
    Weights,
    /// Persistent circular ring KV cache (Q8_0).
    KvCache,
    /// Ephemeral activations and scratch workspace.
    Scratch,
}

/// A sub-allocated slice within the static arena.
#[derive(Debug)]
pub struct ArenaSlice {
    ptr: NonNull<u8>,
    offset: usize,
    size: usize,
    padded_size: usize,
    region: ArenaRegion,
}

// SAFETY: ArenaSlice points to exclusive memory allocated inside the thread-safe Arena.
unsafe impl Send for ArenaSlice {}
unsafe impl Sync for ArenaSlice {}

impl ArenaSlice {
    /// The user-requested size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The padded size (multiple of 64 bytes) suitable for GPU buffer creation.
    #[inline]
    pub fn padded_size(&self) -> usize {
        self.padded_size
    }

    /// Offset from the start of the arena.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Raw pointer to the allocated memory (guaranteed 4096-byte aligned).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer to the allocated memory.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Immutable slice view of the requested byte range.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            // SAFETY: Memory is valid and exclusively managed by this slice.
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
        }
    }

    /// Mutable slice view of the requested byte range.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            // SAFETY: Memory is valid and exclusively managed by this slice.
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
        }
    }

    /// Immutable slice view of the full padded byte range.
    #[inline]
    pub fn as_padded_slice(&self) -> &[u8] {
        unsafe {
            // SAFETY: Padded size is fully backed by arena mapping.
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.padded_size)
        }
    }

    /// Mutable slice view of the full padded byte range.
    #[inline]
    pub fn as_padded_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            // SAFETY: Padded size is fully backed by arena mapping.
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.padded_size)
        }
    }

    #[inline]
    pub fn region(&self) -> ArenaRegion {
        self.region
    }
}

/// Static contiguous memory arena owning the unified host allocation.
pub struct Arena {
    raw_mmap_ptr: *mut libc::c_void,
    raw_mmap_len: usize,
    base_aligned_ptr: NonNull<u8>,
    capacity: usize,
    bump_offset: usize,
    allocated_weights_bytes: usize,
    allocated_kv_bytes: usize,
    allocated_scratch_bytes: usize,
}

// SAFETY: Arena owns its anonymous memory mapping and bump allocation logic is single-owner.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Creates a new static arena of the given capacity with 2 MiB alignment and `MADV_HUGEPAGE`.
    pub fn new(capacity: usize) -> Result<Self, ArenaError> {
        assert!(capacity > 0, "arena capacity must be greater than zero");

        // Over-allocate by 2 MiB to ensure we can find a 2 MiB-aligned base pointer
        let raw_len = capacity + HUGEPAGE_ALIGN;

        let raw_ptr = unsafe {
            // SAFETY: Creating an anonymous, private, read/write memory mapping.
            libc::mmap(
                std::ptr::null_mut(),
                raw_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        if raw_ptr == libc::MAP_FAILED {
            return Err(ArenaError::MmapFailed {
                size: raw_len,
                source: std::io::Error::last_os_error(),
            });
        }

        // Align the usable base pointer to a 2 MiB boundary for Transparent Huge Pages (A5)
        let raw_addr = raw_ptr as usize;
        let aligned_addr = (raw_addr + HUGEPAGE_ALIGN - 1) & !(HUGEPAGE_ALIGN - 1);
        let base_aligned_ptr = match NonNull::new(aligned_addr as *mut u8) {
            Some(p) => p,
            None => {
                unsafe {
                    // SAFETY: Releasing mmap on failure.
                    libc::munmap(raw_ptr, raw_len);
                }
                return Err(ArenaError::MmapFailed {
                    size: raw_len,
                    source: std::io::Error::other("unexpected null pointer after alignment"),
                });
            }
        };

        // Enable transparent hugepages on the aligned range
        #[cfg(target_os = "linux")]
        unsafe {
            // SAFETY: base_aligned_ptr is within the allocated mmap of raw_len.
            libc::madvise(
                base_aligned_ptr.as_ptr() as *mut libc::c_void,
                capacity,
                libc::MADV_HUGEPAGE,
            );
        }

        Ok(Self {
            raw_mmap_ptr: raw_ptr,
            raw_mmap_len: raw_len,
            base_aligned_ptr,
            capacity,
            bump_offset: 0,
            allocated_weights_bytes: 0,
            allocated_kv_bytes: 0,
            allocated_scratch_bytes: 0,
        })
    }

    /// Allocates a contiguous chunk of memory with 4096-B alignment and 64-B size padding.
    pub fn alloc(&mut self, size: usize, region: ArenaRegion) -> Result<ArenaSlice, ArenaError> {
        let align = ARENA_ALLOC_ALIGN;
        let aligned_offset = (self.bump_offset + align - 1) & !(align - 1);
        let padded_size = (size + ARENA_SIZE_ALIGN - 1) & !(ARENA_SIZE_ALIGN - 1);

        let new_offset = aligned_offset.saturating_add(padded_size);
        if new_offset > self.capacity {
            let available = self.capacity.saturating_sub(aligned_offset);
            return Err(ArenaError::OutOfMemory {
                needed: size,
                padded: padded_size,
                available,
            });
        }

        self.bump_offset = new_offset;

        match region {
            ArenaRegion::Weights => self.allocated_weights_bytes += padded_size,
            ArenaRegion::KvCache => self.allocated_kv_bytes += padded_size,
            ArenaRegion::Scratch => self.allocated_scratch_bytes += padded_size,
        }

        let slice_ptr = unsafe {
            // SAFETY: aligned_offset + padded_size <= capacity.
            NonNull::new_unchecked(self.base_aligned_ptr.as_ptr().add(aligned_offset))
        };

        // Assert strictly the invariants required by $ZCOPY (§7 M1 Acceptance Criterion)
        debug_assert_eq!(
            (slice_ptr.as_ptr() as usize) % ARENA_ALLOC_ALIGN,
            0,
            "allocated pointer must be 4096-byte page aligned"
        );
        debug_assert_eq!(
            padded_size % ARENA_SIZE_ALIGN,
            0,
            "padded size must be a multiple of 64 bytes"
        );

        Ok(ArenaSlice {
            ptr: slice_ptr,
            offset: aligned_offset,
            size,
            padded_size,
            region,
        })
    }

    /// Total arena capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current allocated/committed byte offset.
    pub fn used_bytes(&self) -> usize {
        self.bump_offset
    }

    /// Remaining unallocated byte count.
    pub fn remaining_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.bump_offset)
    }

    pub fn weights_bytes(&self) -> usize {
        self.allocated_weights_bytes
    }

    pub fn kv_bytes(&self) -> usize {
        self.allocated_kv_bytes
    }

    pub fn scratch_bytes(&self) -> usize {
        self.allocated_scratch_bytes
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if !self.raw_mmap_ptr.is_null() && self.raw_mmap_ptr != libc::MAP_FAILED {
            unsafe {
                // SAFETY: Unmapping the original allocation created in Arena::new.
                libc::munmap(self.raw_mmap_ptr, self.raw_mmap_len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation_alignments() {
        let mut arena = Arena::new(10 * 1024 * 1024).expect("arena creation");
        assert_eq!(arena.capacity(), 10 * 1024 * 1024);

        // Allocate 3 slices of various non-aligned sizes
        let s1 = arena.alloc(100, ArenaRegion::Weights).unwrap();
        let s2 = arena.alloc(4097, ArenaRegion::KvCache).unwrap();
        let s3 = arena.alloc(64, ArenaRegion::Scratch).unwrap();

        // Check 4096-B alignment
        assert_eq!((s1.as_ptr() as usize) % 4096, 0);
        assert_eq!((s2.as_ptr() as usize) % 4096, 0);
        assert_eq!((s3.as_ptr() as usize) % 4096, 0);

        // Check 64-B size padding
        assert_eq!(s1.padded_size() % 64, 0);
        assert_eq!(s2.padded_size() % 64, 0);
        assert_eq!(s3.padded_size() % 64, 0);

        assert_eq!(s1.size(), 100);
        assert_eq!(s1.padded_size(), 128);

        assert_eq!(s2.size(), 4097);
        assert_eq!(s2.padded_size(), 4160);
    }

    #[test]
    fn test_arena_oom() {
        let mut arena = Arena::new(4096).unwrap();
        let _ = arena.alloc(100, ArenaRegion::Weights).unwrap();
        let err = arena.alloc(4096, ArenaRegion::Weights).unwrap_err();
        assert!(matches!(err, ArenaError::OutOfMemory { .. }));
    }
}
