//! Zero-copy OpenCL buffer management ($ZCOPY, §7 M3a).
//!
//! Wraps page-aligned arena allocations using `CL_MEM_USE_HOST_PTR`, asserts pointer identity
//! on mapping, and ensures memory coherency between GPU kernel execution and host CPU reads.

use crate::program::OclError;
use opencl3::command_queue::{enqueue_map_buffer, enqueue_unmap_mem_object, CommandQueue};
use opencl3::context::Context;
use opencl3::memory::{
    cl_mem, create_buffer, release_mem_object, CL_MAP_READ, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE,
    CL_MEM_USE_HOST_PTR,
};
use opencl3::types::CL_TRUE;
use std::ffi::c_void;
use std::ptr;

/// A zero-copy OpenCL buffer mapped directly over page-aligned arena memory ($ZCOPY).
pub struct OclBuffer {
    mem: cl_mem,
    host_ptr: *const u8,
    size: usize,
    pub read_only: bool,
}

unsafe impl Send for OclBuffer {}
unsafe impl Sync for OclBuffer {}

impl OclBuffer {
    /// Creates a zero-copy buffer from an existing page-aligned host pointer ($ZCOPY).
    ///
    /// # Safety
    /// `host_ptr` must point to valid memory for at least `size` bytes, must be 4096-byte aligned,
    /// and `size` must be a multiple of 64 bytes.
    pub unsafe fn from_host_ptr(
        context: &Context,
        queue: &CommandQueue,
        host_ptr: *const u8,
        size: usize,
        read_only: bool,
    ) -> Result<Self, OclError> {
        assert_eq!(
            (host_ptr as usize) % 4096,
            0,
            "host_ptr must be 4096-byte page aligned ($ZCOPY)"
        );
        assert_eq!(size % 64, 0, "buffer size must be 64-byte padded ($ZCOPY)");

        let flags = if read_only {
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR
        } else {
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR
        };

        let mem = create_buffer(context.get(), flags, size, host_ptr as *mut c_void)
            .map_err(|e| OclError::Driver(format!("create_buffer error: {e}")))?;

        // Verify zero-copy: map buffer and assert mapped_ptr == host_ptr
        let mut mapped_ptr: cl_mem = ptr::null_mut();
        let map_event = enqueue_map_buffer(
            queue.get(),
            mem,
            CL_TRUE,
            CL_MAP_READ,
            0,
            size,
            &mut mapped_ptr,
            0,
            ptr::null(),
        );

        if let Err(e) = map_event {
            let _ = release_mem_object(mem);
            return Err(OclError::Driver(format!(
                "enqueue_map_buffer verification error: {e}"
            )));
        }

        let is_zero_copy = (mapped_ptr as *const u8) == host_ptr;

        let unmap_res = enqueue_unmap_mem_object(queue.get(), mem, mapped_ptr, 0, ptr::null());

        if let Err(e) = unmap_res {
            let _ = release_mem_object(mem);
            return Err(OclError::Driver(format!(
                "enqueue_unmap_mem_object error: {e}"
            )));
        }

        if !is_zero_copy {
            let _ = release_mem_object(mem);
            return Err(OclError::ZeroCopyViolation(
                mapped_ptr as *const u8,
                host_ptr,
            ));
        }

        Ok(Self {
            mem,
            host_ptr,
            size,
            read_only,
        })
    }

    /// Returns the underlying raw `cl_mem` handle for kernel arguments.
    pub fn cl_mem(&self) -> cl_mem {
        self.mem
    }

    /// Returns the host pointer for this buffer.
    pub fn host_ptr(&self) -> *const u8 {
        self.host_ptr
    }

    /// Maps buffer for reading, synchronizing GPU updates to host memory.
    pub fn map_read(&self, queue: &CommandQueue) -> Result<*const u8, OclError> {
        let mut mapped_ptr: cl_mem = ptr::null_mut();
        unsafe {
            enqueue_map_buffer(
                queue.get(),
                self.mem,
                CL_TRUE,
                CL_MAP_READ,
                0,
                self.size,
                &mut mapped_ptr,
                0,
                ptr::null(),
            )
            .map_err(|e| OclError::Driver(format!("enqueue_map_buffer error: {e}")))?;
        }
        Ok(mapped_ptr as *const u8)
    }

    /// Unmaps a previously mapped buffer.
    pub fn unmap(&self, queue: &CommandQueue, mapped_ptr: *const u8) -> Result<(), OclError> {
        unsafe {
            enqueue_unmap_mem_object(
                queue.get(),
                self.mem,
                mapped_ptr as *mut c_void,
                0,
                ptr::null(),
            )
            .map_err(|e| OclError::Driver(format!("enqueue_unmap_mem_object error: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for OclBuffer {
    fn drop(&mut self) {
        if !self.mem.is_null() {
            unsafe {
                let _ = release_mem_object(self.mem);
            }
        }
    }
}
