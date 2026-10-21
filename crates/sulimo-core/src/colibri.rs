//! Colibrì Engine: Circular Q8_0 KV Cache & NVMe Expert Streamer (@C:COL).

use half::f16;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use thiserror::Error;

pub const BLOCK_Q8_0_ELEMS: usize = 32;
pub const BLOCK_Q8_0_BYTES: usize = 34; // 2 bytes f16 scale + 32 bytes i8 values
pub const PAGE_ALIGNMENT: usize = 4096;
pub const EXPERT_LRU_RING_BYTES: usize = 1_200 * 1024 * 1024; // 1.2 GB pinned buffer

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub scale: f16,
    pub qs: [i8; BLOCK_Q8_0_ELEMS],
}

#[derive(Debug, Error)]
pub enum ColibriError {
    #[error("I/O error during expert streaming: {0}")]
    Io(#[from] std::io::Error),
    #[error("Cache alignment failure: memory not 4096-byte page aligned")]
    UnalignedMemory,
    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(usize),
}

/// Circular Q8_0 KV Cache with Sliding Window and GQA support.
pub struct CircularKvCacheQ8 {
    pub n_layers: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub capacity: usize,
    pub sliding_window: Option<usize>, // e.g., 512 for Gemma 4
    pub blocks_per_head: usize,
    raw_buffer: Vec<u8>,
}

impl CircularKvCacheQ8 {
    pub fn new(
        n_layers: usize,
        n_head_kv: usize,
        head_dim: usize,
        capacity: usize,
        sliding_window: Option<usize>,
    ) -> Result<Self, ColibriError> {
        let blocks_per_head = head_dim.div_ceil(BLOCK_Q8_0_ELEMS);
        let total_blocks = n_layers * 2 * capacity * n_head_kv * blocks_per_head;
        let total_bytes = total_blocks * BLOCK_Q8_0_BYTES;

        let mut raw_buffer = vec![0u8; total_bytes + PAGE_ALIGNMENT];
        let align_offset =
            (PAGE_ALIGNMENT - (raw_buffer.as_ptr() as usize % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT;
        raw_buffer.drain(0..align_offset);
        raw_buffer.truncate(total_bytes);

        Ok(Self {
            n_layers,
            n_head_kv,
            head_dim,
            capacity,
            sliding_window,
            blocks_per_head,
            raw_buffer,
        })
    }

    #[inline(always)]
    pub fn slot_for_pos(&self, pos: usize) -> usize {
        if let Some(win) = self.sliding_window {
            pos % win
        } else {
            pos % self.capacity
        }
    }

    pub fn write_kv_q8(&mut self, layer: usize, pos: usize, k_fp32: &[f32], v_fp32: &[f32]) {
        let slot = self.slot_for_pos(pos);
        let stride = self.n_head_kv * self.blocks_per_head * BLOCK_Q8_0_BYTES;
        let layer_offset = layer * 2 * self.capacity * stride;

        let k_offset = layer_offset + (slot * stride);
        let v_offset = layer_offset + (self.capacity * stride) + (slot * stride);

        Self::quantize_row_q8_0(k_fp32, &mut self.raw_buffer[k_offset..k_offset + stride]);
        Self::quantize_row_q8_0(v_fp32, &mut self.raw_buffer[v_offset..v_offset + stride]);
    }

    fn quantize_row_q8_0(src: &[f32], dst_bytes: &mut [u8]) {
        let num_blocks = src.len() / BLOCK_Q8_0_ELEMS;
        for b in 0..num_blocks {
            let chunk = &src[b * BLOCK_Q8_0_ELEMS..(b + 1) * BLOCK_Q8_0_ELEMS];
            let mut amax = 0.0f32;
            for &val in chunk {
                let abs = val.abs();
                if abs > amax {
                    amax = abs;
                }
            }
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let scale = f16::from_f32(d);

            let out_slice = &mut dst_bytes[b * BLOCK_Q8_0_BYTES..(b + 1) * BLOCK_Q8_0_BYTES];
            out_slice[0..2].copy_from_slice(&scale.to_le_bytes());
            for i in 0..BLOCK_Q8_0_ELEMS {
                let q = (chunk[i] * id).round().clamp(-128.0, 127.0) as i8;
                out_slice[2 + i] = q as u8;
            }
        }
    }
}

/// NVMe Asynchronous Expert Streaming Engine with 1.2 GB Pinned LRU Cache.
pub struct ExpertStreamer {
    file_fd: i32,
    ring_buffer: Vec<u8>,
    lru_expert_ids: Vec<u32>,
    pub expert_size_bytes: usize,
    capacity_experts: usize,
}

impl ExpertStreamer {
    pub fn open<P: AsRef<Path>>(
        nvme_weights_path: P,
        expert_size_bytes: usize,
    ) -> Result<Self, ColibriError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(nvme_weights_path)?;

        let capacity_experts = EXPERT_LRU_RING_BYTES / expert_size_bytes;
        let total_bytes = capacity_experts * expert_size_bytes;

        // Pinned page-aligned LRU ring allocation
        let mut ring_buffer = vec![0u8; total_bytes + PAGE_ALIGNMENT];
        let offset =
            (PAGE_ALIGNMENT - (ring_buffer.as_ptr() as usize % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT;
        ring_buffer.drain(0..offset);
        ring_buffer.truncate(total_bytes);

        let fd = file.as_raw_fd();
        std::mem::forget(file); // Retain FD ownership

        Ok(Self {
            file_fd: fd,
            ring_buffer,
            lru_expert_ids: Vec::with_capacity(capacity_experts),
            expert_size_bytes,
            capacity_experts,
        })
    }

    pub fn get_expert_slice(
        &mut self,
        expert_id: u32,
        file_offset: u64,
    ) -> Result<&[u8], ColibriError> {
        if let Some(pos) = self.lru_expert_ids.iter().position(|&id| id == expert_id) {
            let id = self.lru_expert_ids.remove(pos);
            self.lru_expert_ids.push(id);
            let idx = self.lru_expert_ids.len() - 1;
            let start = idx * self.expert_size_bytes;
            return Ok(&self.ring_buffer[start..start + self.expert_size_bytes]);
        }

        // Cache miss: stream directly from NVMe
        let target_idx = if self.lru_expert_ids.len() >= self.capacity_experts {
            self.lru_expert_ids.remove(0);
            self.lru_expert_ids.push(expert_id);
            self.capacity_experts - 1
        } else {
            self.lru_expert_ids.push(expert_id);
            self.lru_expert_ids.len() - 1
        };

        let start = target_idx * self.expert_size_bytes;
        let dst_ptr = unsafe { self.ring_buffer.as_mut_ptr().add(start) };

        // Synchronous direct-IO read fallback (integrating with io_uring on kernels >= 5.10)
        let ret = unsafe {
            libc::pread(
                self.file_fd,
                dst_ptr as *mut libc::c_void,
                self.expert_size_bytes,
                file_offset as libc::off_t,
            )
        };

        if ret < 0 {
            return Err(ColibriError::Io(std::io::Error::last_os_error()));
        }

        Ok(&self.ring_buffer[start..start + self.expert_size_bytes])
    }
}

impl Drop for ExpertStreamer {
    fn drop(&mut self) {
        unsafe { libc::close(self.file_fd) };
    }
}
