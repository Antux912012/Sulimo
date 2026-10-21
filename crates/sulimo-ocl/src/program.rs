//! OpenCL program building, caching, and kernel dispatch management (@C:OCL-GEN9, §7 M3a).
//!
//! Embeds OpenCL C kernel source (`kernels/gemm_q4_0.cl`), compiles with fast FP16 flags,
//! caches binaries in `~/.cache/sulimo/ocl/<device-hash>.bin`, and selects between
//! Intel SIMD8 sub-group and portable variants at runtime.

use crate::probe::{ensure_icd_env, OclDeviceInfo};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel;
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use thiserror::Error;

const GEMM_KERNEL_SOURCE: &str = include_str!("../../../kernels/gemm_q4_0.cl");
const PREFILL_KERNEL_SOURCE: &str = include_str!("../../../kernels/prefill.cl");
const BUILD_OPTIONS: &str = "-DGEN8_PLUS_FP16 -cl-mad-enable -cl-fast-relaxed-math";

/// Errors returned by OpenCL operations.
#[derive(Debug, Error)]
pub enum OclError {
    #[error("OpenCL platform or device not found: {0}")]
    DeviceNotFound(String),
    #[error("OpenCL driver error: {0}")]
    Driver(String),
    #[error("OpenCL compilation or build error: {0}\nLog: {1}")]
    BuildFailed(String, String),
    #[error("Zero-copy violation: mapped pointer {0:p} does not match host pointer {1:p}")]
    ZeroCopyViolation(*const u8, *const u8),
    #[error("OpenCL command execution failed with status: {0}")]
    ExecutionFailed(i32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// The kernel variant selected at runtime based on device capabilities (A16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// Intel Gen8–11 SIMD8 sub-group specialized kernel with half4 FMA.
    Intel,
    /// Standard OpenCL FP16 portable kernel.
    Portable,
}

/// Manages the initialized OpenCL context, command queue, and precompiled kernels.
pub struct OclContext {
    pub context: Context,
    pub device: Device,
    pub queue: CommandQueue,
    pub program: Program,
    pub kernel_intel: Option<Kernel>,
    pub kernel_portable: Kernel,
    pub kernel_rmsnorm: Kernel,
    pub kernel_rope: Kernel,
    pub kernel_kv_store: Kernel,
    pub kernel_attention: Kernel,
    pub kernel_silu_mul: Kernel,
    pub kernel_add: Kernel,
    pub kernel_logits: Kernel,
    pub active_variant: KernelVariant,
    pub device_info: OclDeviceInfo,
}

impl OclContext {
    /// Initializes OpenCL context on the preferred Intel GPU device.
    pub fn new() -> Result<Self, OclError> {
        ensure_icd_env();

        let platforms = get_platforms().map_err(|e| OclError::DeviceNotFound(e.to_string()))?;
        if platforms.is_empty() {
            return Err(OclError::DeviceNotFound(
                "No OpenCL platforms available".into(),
            ));
        }

        let mut chosen_dev = None;
        let mut chosen_plat = None;

        for plat in &platforms {
            if let Ok(devs) = plat.get_devices(CL_DEVICE_TYPE_GPU) {
                for dev_id in devs {
                    let d = Device::new(dev_id);
                    if d.name().unwrap_or_default().contains("Intel") {
                        chosen_dev = Some(d);
                        chosen_plat = Some(plat);
                        break;
                    }
                }
            }
            if chosen_dev.is_some() {
                break;
            }
        }

        // Fallback to first available GPU device if no Intel device matched
        if chosen_dev.is_none() {
            for plat in &platforms {
                if let Ok(devs) = plat.get_devices(CL_DEVICE_TYPE_GPU) {
                    if let Some(&first) = devs.first() {
                        chosen_dev = Some(Device::new(first));
                        chosen_plat = Some(plat);
                        break;
                    }
                }
            }
        }

        let device =
            chosen_dev.ok_or_else(|| OclError::DeviceNotFound("No GPU device found".into()))?;
        let _platform = chosen_plat.unwrap();

        let devices = [device.id()];
        let context = Context::from_devices(&devices, &[], None, std::ptr::null_mut())
            .map_err(|e| OclError::Driver(format!("clCreateContext: {e}")))?;

        // In-order queue with profiling support
        let queue = unsafe {
            CommandQueue::create_with_properties(
                &context,
                device.id(),
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .map_err(|e| OclError::Driver(format!("clCreateCommandQueue: {e}")))?
        };

        let device_name = device.name().unwrap_or_default();
        let driver_version = device.driver_version().unwrap_or_default();
        let extensions = device.extensions().unwrap_or_default();
        let has_fp16 = extensions.contains("cl_khr_fp16");
        let has_intel_subgroups = extensions.contains("cl_intel_subgroups");
        let has_intel_reqd_subgroup_size = extensions.contains("cl_intel_required_subgroup_size");
        let host_unified_memory = device.host_unified_memory().unwrap_or(false);

        let device_info = OclDeviceInfo {
            platform_name: _platform.name().unwrap_or_default(),
            platform_version: _platform.version().unwrap_or_default(),
            device_name: device_name.clone(),
            device_vendor: device.vendor().unwrap_or_default(),
            driver_version: driver_version.clone(),
            compute_units: device.max_compute_units().unwrap_or(0),
            base_addr_align_bits: device.min_data_type_align_size().unwrap_or(0) * 8,
            host_unified_memory,
            has_fp16,
            has_intel_subgroups,
            has_intel_reqd_subgroup_size,
            extensions: extensions
                .split_whitespace()
                .map(|s| s.to_string())
                .collect(),
        };

        let full_source = format!("{GEMM_KERNEL_SOURCE}\n{PREFILL_KERNEL_SOURCE}");
        let cache_hash =
            compute_device_hash(&device_name, &driver_version, &full_source, BUILD_OPTIONS);
        let program =
            load_or_build_program(&context, &device, &full_source, BUILD_OPTIONS, cache_hash)?;

        let kernel_portable = Kernel::create(&program, "gemm_q4_0_f16_portable")
            .map_err(|e| OclError::Driver(format!("clCreateKernel portable: {e}")))?;

        let kernel_intel = if has_intel_subgroups && has_intel_reqd_subgroup_size {
            Kernel::create(&program, "gemm_q4_0_f16_intel").ok()
        } else {
            None
        };

        let kernel_rmsnorm = Kernel::create(&program, "rmsnorm_f16_f32w")
            .map_err(|e| OclError::Driver(format!("clCreateKernel rmsnorm: {e}")))?;
        let kernel_rope = Kernel::create(&program, "rope_f16")
            .map_err(|e| OclError::Driver(format!("clCreateKernel rope: {e}")))?;
        let kernel_kv_store = Kernel::create(&program, "kv_store_q8_0")
            .map_err(|e| OclError::Driver(format!("clCreateKernel kv_store: {e}")))?;
        let kernel_attention = Kernel::create(&program, "causal_gqa_attention_prefill")
            .map_err(|e| OclError::Driver(format!("clCreateKernel attention: {e}")))?;
        let kernel_silu_mul = Kernel::create(&program, "silu_mul_f16")
            .map_err(|e| OclError::Driver(format!("clCreateKernel silu_mul: {e}")))?;
        let kernel_add = Kernel::create(&program, "add_f16")
            .map_err(|e| OclError::Driver(format!("clCreateKernel add: {e}")))?;
        let kernel_logits = Kernel::create(&program, "logits_f16_output")
            .map_err(|e| OclError::Driver(format!("clCreateKernel logits: {e}")))?;

        let active_variant = if kernel_intel.is_some() {
            KernelVariant::Intel
        } else {
            KernelVariant::Portable
        };

        Ok(Self {
            context,
            device,
            queue,
            program,
            kernel_intel,
            kernel_portable,
            kernel_rmsnorm,
            kernel_rope,
            kernel_kv_store,
            kernel_attention,
            kernel_silu_mul,
            kernel_add,
            kernel_logits,
            active_variant,
            device_info,
        })
    }
}

fn compute_device_hash(device_name: &str, driver_ver: &str, source: &str, options: &str) -> String {
    let mut hasher = DefaultHasher::new();
    device_name.hash(&mut hasher);
    driver_ver.hash(&mut hasher);
    source.hash(&mut hasher);
    options.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn cache_dir_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/sulimo/ocl"))
}

fn load_or_build_program(
    context: &Context,
    device: &Device,
    source: &str,
    options: &str,
    cache_hash: String,
) -> Result<Program, OclError> {
    let cache_dir = cache_dir_path();
    let cache_file = cache_dir
        .as_ref()
        .map(|d| d.join(format!("{cache_hash}.bin")));

    // 1. Try loading cached binary
    if let Some(ref path) = cache_file {
        if path.exists() {
            if let Ok(bin) = fs::read(path) {
                let bins = vec![bin.as_slice()];
                let devices = [device.id()];
                let res = unsafe { Program::create_from_binary(context, &devices, &bins) };
                if let Ok(mut prog) = res {
                    if prog.build(&devices, options).is_ok() {
                        return Ok(prog);
                    }
                }
            }
        }
    }

    // 2. Build from source
    let mut program = Program::create_from_source(context, source)
        .map_err(|e| OclError::Driver(format!("clCreateProgramWithSource: {e}")))?;

    let devices = [device.id()];
    if let Err(e) = program.build(&devices, options) {
        let log = program
            .get_build_log(device.id())
            .unwrap_or_else(|_| "No build log".into());
        return Err(OclError::BuildFailed(e.to_string(), log));
    }

    // 3. Save binary to cache
    if let Some(ref dir) = cache_dir {
        let _ = fs::create_dir_all(dir);
        if let Some(ref file_path) = cache_file {
            if let Ok(binaries) = program.get_binaries() {
                if let Some(first_bin) = binaries.first() {
                    let _ = fs::write(file_path, first_bin);
                }
            }
        }
    }

    Ok(program)
}
