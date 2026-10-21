//! Zero-spin OpenCL event completion waiter ($BATT, A9, §7 M3a).
//!
//! Replaces polling loops (`clWaitForEvents` spin or sleep) with OS-level futex parking.
//! Registers `clSetEventCallback(CL_COMPLETE)`, flushes the queue, and calls `thread::park()`,
//! ensuring 0% CPU core package power draw while the iGPU computes.

use crate::program::OclError;
use opencl3::command_queue::CommandQueue;
use opencl3::event::Event;
use opencl3::types::{cl_event, cl_int};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::thread::{self, Thread};

const CL_COMPLETE: cl_int = 0;
const CL_SUCCESS: cl_int = 0;

struct WaiterState {
    thread: Thread,
    done: AtomicBool,
    status: AtomicI32,
}

extern "C" fn event_callback(
    _event: cl_event,
    event_command_status: cl_int,
    user_data: *mut c_void,
) {
    if !user_data.is_null() {
        let state = unsafe { &*(user_data as *const WaiterState) };
        state.status.store(event_command_status, Ordering::Release);
        state.done.store(true, Ordering::Release);
        state.thread.unpark();
    }
}

/// Waits for an OpenCL event to complete by parking the calling thread ($BATT, A9).
pub fn wait_event_zero_spin(queue: &CommandQueue, event: &Event) -> Result<(), OclError> {
    // 1. Flush command queue to ensure GPU execution has started
    queue
        .flush()
        .map_err(|e| OclError::Driver(format!("clFlush: {e}")))?;

    // 2. Register callback on CL_COMPLETE
    let state = Arc::new(WaiterState {
        thread: thread::current(),
        done: AtomicBool::new(false),
        status: AtomicI32::new(CL_SUCCESS),
    });

    let raw_ptr = Arc::into_raw(state);

    let callback_res = opencl3::event::set_event_callback(
        event.get(),
        CL_COMPLETE,
        event_callback,
        raw_ptr as *mut c_void,
    );

    if let Err(e) = callback_res {
        // Reconstruct Arc to avoid leak if registration fails
        unsafe {
            let _ = Arc::from_raw(raw_ptr);
        }
        return Err(OclError::Driver(format!("clSetEventCallback error: {e}")));
    }

    // 3. Zero-spin parking: thread is put to sleep in the OS scheduler
    while !unsafe { &*raw_ptr }.done.load(Ordering::Acquire) {
        thread::park();
    }

    // 4. Clean up Arc and verify execution status
    let state = unsafe { Arc::from_raw(raw_ptr) };
    let status = state.status.load(Ordering::Acquire);
    if status < 0 {
        return Err(OclError::ExecutionFailed(status));
    }

    Ok(())
}
