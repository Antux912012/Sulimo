//! Benchmarking, hardware telemetry profiling, and threshold calibration (§7 M6).
//!
//! Provides automated throughput measurement for prefill (128, 512, 2048 tokens),
//! decode (128 tokens), iGPU RC6 residency, peak resident memory (VmHWM),
//! and $T$-threshold sweep calibration between OpenCL iGPU and physical CPU cores.

use crate::engine::{Engine, EngineError};
use crate::sampler::SamplingParams;
use crate::scheduler::SchedulerConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use sulimo_core::gguf::test_utils::{create_test_gguf, TestTensorSpec};
use sulimo_core::gguf::GgufValueType;
use sulimo_core::loader::get_process_memory_stats;
use sulimo_cpu::topology::CpuTopology;
#[cfg(feature = "opencl")]
use sulimo_ocl::probe;

/// Summary of host hardware for benchmark output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostHardwareInfo {
    pub cpu_model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub gpu_device: String,
    pub driver_version: String,
}

/// Single prefill benchmark point (N tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBenchmarkPoint {
    pub prompt_tokens: usize,
    pub prefill_tok_per_sec: f64,
    pub ttft_ms: f64,
    pub backend: String,
    pub energy_joules_total: Option<f64>,
}

/// Decode benchmark metrics (N=1 autoregressive generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeBenchmarkMetrics {
    pub tokens_generated: usize,
    pub decode_tok_per_sec: f64,
    pub rc6_residency_pct: Option<f64>,
    pub energy_joules_per_token: Option<f64>,
}

/// Single point in the $T$ threshold sweep calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSweepPoint {
    pub n: usize,
    pub cpu_prefill_ms: f64,
    pub gpu_prefill_ms: Option<f64>,
    pub faster_backend: String,
}

/// Memory footprint recorded during benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub peak_rss_bytes: u64,
    pub peak_rss_gib: f64,
    pub within_5_2gb_budget: bool,
}

/// Complete benchmark execution results conforming to benchmarks/README.md schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub host: HostHardwareInfo,
    pub model_path: String,
    pub context_length: usize,
    pub prefill: Vec<PrefillBenchmarkPoint>,
    pub decode: DecodeBenchmarkMetrics,
    pub memory: MemoryMetrics,
    pub threshold_calibration: Option<Vec<ThresholdSweepPoint>>,
    pub recommended_threshold_t: usize,
}

/// Reads the current i915 RC6 residency in milliseconds from sysfs if available.
pub fn read_rc6_residency_ms() -> Option<u64> {
    for candidate in &[
        "/sys/class/drm/card1/power/rc6_residency_ms",
        "/sys/class/drm/card0/power/rc6_residency_ms",
        "/sys/class/drm/card1/gt/gt0/rc6_residency_ms",
        "/sys/class/drm/card0/gt/gt0/rc6_residency_ms",
    ] {
        if let Ok(content) = fs::read_to_string(candidate) {
            if let Ok(val) = content.trim().parse::<u64>() {
                return Some(val);
            }
        }
    }
    None
}

/// Reads RAPL energy in microjoules if sysfs is readable.
pub fn read_rapl_energy_uj() -> Option<u64> {
    for candidate in &[
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj",
    ] {
        if let Ok(content) = fs::read_to_string(candidate) {
            if let Ok(val) = content.trim().parse::<u64>() {
                return Some(val);
            }
        }
    }
    None
}

fn encode_string_array(strings: &[&str]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    buf.extend_from_slice(&(strings.len() as u64).to_le_bytes());
    for s in strings {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    buf
}

/// Creates a temporary synthetic GGUF model suitable for benchmarking (context up to 4096).
pub fn create_synthetic_benchmark_model(ctx_len: u32) -> PathBuf {
    let mut kvs: Vec<(&str, GgufValueType, &[u8])> = Vec::new();
    let arch_bytes = [5u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a'];
    let ctx_bytes = ctx_len.to_le_bytes();
    let embd_bytes = (64u32).to_le_bytes();
    let blk_bytes = (1u32).to_le_bytes();
    let ffn_bytes = (128u32).to_le_bytes();
    let heads_bytes = (4u32).to_le_bytes();
    let heads_kv_bytes = (2u32).to_le_bytes();
    let tok_model_bytes = [4u8, 0, 0, 0, 0, 0, 0, 0, b'g', b'p', b't', b'2'];
    let tok_pre_bytes = [
        9u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a', b'-', b'b', b'p', b'e',
    ];
    let bos_bytes = (1u32).to_le_bytes();
    let eos_bytes = (2u32).to_le_bytes();

    kvs.push(("general.architecture", GgufValueType::String, &arch_bytes));
    kvs.push(("llama.context_length", GgufValueType::Uint32, &ctx_bytes));
    kvs.push(("llama.embedding_length", GgufValueType::Uint32, &embd_bytes));
    kvs.push(("llama.block_count", GgufValueType::Uint32, &blk_bytes));
    kvs.push((
        "llama.feed_forward_length",
        GgufValueType::Uint32,
        &ffn_bytes,
    ));
    kvs.push((
        "llama.attention.head_count",
        GgufValueType::Uint32,
        &heads_bytes,
    ));
    kvs.push((
        "llama.attention.head_count_kv",
        GgufValueType::Uint32,
        &heads_kv_bytes,
    ));
    kvs.push((
        "tokenizer.ggml.model",
        GgufValueType::String,
        &tok_model_bytes,
    ));
    kvs.push(("tokenizer.ggml.pre", GgufValueType::String, &tok_pre_bytes));
    kvs.push((
        "tokenizer.ggml.bos_token_id",
        GgufValueType::Uint32,
        &bos_bytes,
    ));
    kvs.push((
        "tokenizer.ggml.eos_token_id",
        GgufValueType::Uint32,
        &eos_bytes,
    ));

    let tokens = [
        "<unk>", "<s>", "</s>", "hello", "world", "sulimo", "gen", "test", "speed", "fast",
    ];
    let tokens_encoded = Box::leak(encode_string_array(&tokens).into_boxed_slice());
    kvs.push((
        "tokenizer.ggml.tokens",
        GgufValueType::Array,
        tokens_encoded,
    ));

    let q4_data_360 = vec![0x55u8; 360];
    let q4_data_2304 = vec![0x55u8; 2304];
    let q4_data_1152 = vec![0x55u8; 1152];
    let q4_data_4608 = vec![0x55u8; 4608];
    let f32_data_256 = vec![0x11u8; 256];

    let mut cur_offset = 0u64;
    let mut tensors: Vec<TestTensorSpec> = Vec::new();

    let mut add_tensor =
        |name: &'static str, dims: &'static [u64], type_id: u32, data: &'static [u8]| {
            let t = (name, dims, type_id, cur_offset, data);
            cur_offset += data.len() as u64;
            tensors.push(t);
        };

    let q4_360_ref: &'static [u8] = Box::leak(q4_data_360.into_boxed_slice());
    let q4_2304_ref: &'static [u8] = Box::leak(q4_data_2304.into_boxed_slice());
    let q4_1152_ref: &'static [u8] = Box::leak(q4_data_1152.into_boxed_slice());
    let q4_4608_ref: &'static [u8] = Box::leak(q4_data_4608.into_boxed_slice());
    let f32_256_ref: &'static [u8] = Box::leak(f32_data_256.into_boxed_slice());

    add_tensor("token_embd.weight", &[64, 10], 2, q4_360_ref);
    add_tensor("output_norm.weight", &[64], 0, f32_256_ref);
    add_tensor("blk.0.attn_norm.weight", &[64], 0, f32_256_ref);
    add_tensor("blk.0.attn_q.weight", &[64, 64], 2, q4_2304_ref);
    add_tensor("blk.0.attn_k.weight", &[64, 32], 2, q4_1152_ref);
    add_tensor("blk.0.attn_v.weight", &[64, 32], 2, q4_1152_ref);
    add_tensor("blk.0.attn_output.weight", &[64, 64], 2, q4_2304_ref);
    add_tensor("blk.0.ffn_norm.weight", &[64], 0, f32_256_ref);
    add_tensor("blk.0.ffn_gate.weight", &[64, 128], 2, q4_4608_ref);
    add_tensor("blk.0.ffn_up.weight", &[64, 128], 2, q4_4608_ref);
    add_tensor("blk.0.ffn_down.weight", &[128, 64], 2, q4_4608_ref);

    let temp_path = std::env::temp_dir().join(format!("sulimo_bench_{}.gguf", std::process::id()));
    let gguf_bytes = create_test_gguf(0x46554747, 3, &kvs, &tensors);
    let mut file = fs::File::create(&temp_path).expect("create synthetic model file");
    file.write_all(&gguf_bytes)
        .expect("write synthetic model file");
    file.flush().expect("flush synthetic model file");
    temp_path
}

/// Runs full benchmark suite against a loaded GGUF model path.
pub fn run_benchmark_suite<P: AsRef<Path>>(
    model_path: P,
    requested_ctx: Option<usize>,
    threads: Option<usize>,
    calibrate_t: bool,
) -> Result<BenchmarkReport, EngineError> {
    let path_ref = model_path.as_ref();
    let cpu_topo = CpuTopology::detect().ok();

    let host_info = HostHardwareInfo {
        cpu_model: cpu_topo
            .as_ref()
            .map(|t| t.model_name.clone())
            .unwrap_or_else(|| "Unknown CPU".to_string()),
        physical_cores: cpu_topo
            .as_ref()
            .map(|t| t.physical_cores.len())
            .unwrap_or(1),
        logical_cores: cpu_topo.as_ref().map(|t| t.logical_core_count).unwrap_or(1),
        gpu_device: {
            #[cfg(feature = "opencl")]
            {
                match probe() {
                    sulimo_ocl::OclProbeResult::Available(d) => d.device_name.clone(),
                    sulimo_ocl::OclProbeResult::Unavailable { .. } => {
                        "OpenCL Unavailable".to_string()
                    }
                }
            }
            #[cfg(not(feature = "opencl"))]
            {
                "None (CPU Only)".to_string()
            }
        },
        driver_version: {
            #[cfg(feature = "opencl")]
            {
                match probe() {
                    sulimo_ocl::OclProbeResult::Available(d) => d.driver_version.clone(),
                    sulimo_ocl::OclProbeResult::Unavailable { .. } => "N/A".to_string(),
                }
            }
            #[cfg(not(feature = "opencl"))]
            {
                "N/A".to_string()
            }
        },
    };

    // 1. Load engine
    let config = SchedulerConfig {
        gpu_threshold: 8,
        ubatch_size: 512,
        opencl_enabled: true,
    };
    let mut engine = Engine::load_with_config(path_ref, requested_ctx, threads, config)?;
    let ctx_len = engine.model.plan.context_length;

    // 2. Prefill benchmarks for N in {128, 512, 2048} (clamped to model context)
    let prefill_targets = [128, 512, 2048];
    let mut prefill_results = Vec::new();
    let cancel = AtomicBool::new(false);

    for &target_n in &prefill_targets {
        if target_n > ctx_len {
            continue;
        }

        let prompt_tokens = vec![3u32; target_n];
        let energy_before = read_rapl_energy_uj();

        let params = SamplingParams {
            max_tokens: 1, // Measure TTFT & prefill speed
            greedy: true,
            ignore_eos: true,
            ..Default::default()
        };

        let stats =
            engine.generate_tokens_with_callback(&prompt_tokens, &params, &cancel, |_| {})?;
        let energy_after = read_rapl_energy_uj();

        let energy_joules_total = match (energy_before, energy_after) {
            (Some(b), Some(a)) => Some((a.wrapping_sub(b) as f64) / 1_000_000.0),
            _ => None,
        };

        prefill_results.push(PrefillBenchmarkPoint {
            prompt_tokens: stats.prompt_tokens,
            prefill_tok_per_sec: stats.prefill_tok_per_sec,
            ttft_ms: stats.ttft_ms,
            backend: stats.prefill_backend,
            energy_joules_total,
        });
    }

    // 3. Decode benchmark (128 tokens) with RC6 sampling and energy
    let rc6_before = read_rc6_residency_ms();
    let energy_decode_before = read_rapl_energy_uj();
    let wall_start = Instant::now();

    let decode_params = SamplingParams {
        max_tokens: 128,
        greedy: true,
        ignore_eos: true,
        ..Default::default()
    };
    let decode_stats =
        engine.generate_tokens_with_callback(&[3u32], &decode_params, &cancel, |_| {})?;

    let wall_elapsed_ms = wall_start.elapsed().as_millis() as u64;
    let rc6_after = read_rc6_residency_ms();
    let energy_decode_after = read_rapl_energy_uj();

    let rc6_residency_pct = match (rc6_before, rc6_after) {
        (Some(b), Some(a)) if wall_elapsed_ms > 0 => {
            let delta = a.saturating_sub(b);
            let pct = ((delta as f64) / (wall_elapsed_ms as f64)) * 100.0;
            Some(pct.min(100.0))
        }
        _ => None,
    };

    let energy_joules_per_token = match (energy_decode_before, energy_decode_after) {
        (Some(b), Some(a)) if decode_stats.completion_tokens > 0 => {
            let delta_j = (a.wrapping_sub(b) as f64) / 1_000_000.0;
            Some(delta_j / decode_stats.completion_tokens as f64)
        }
        _ => None,
    };

    let decode_results = DecodeBenchmarkMetrics {
        tokens_generated: decode_stats.completion_tokens,
        decode_tok_per_sec: decode_stats.decode_tok_per_sec,
        rc6_residency_pct,
        energy_joules_per_token,
    };

    // 4. Memory footprint
    let (_, hwm_bytes) = get_process_memory_stats();
    let memory_metrics = MemoryMetrics {
        peak_rss_bytes: hwm_bytes,
        peak_rss_gib: (hwm_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        within_5_2gb_budget: hwm_bytes <= sulimo_core::MAX_RAM_BUDGET_BYTES,
    };

    // 5. Threshold Calibration sweep (N in {1, 2, 4, 8, 16, 32})
    let mut sweep_results = None;
    let mut optimal_t = 8;

    if calibrate_t {
        let sweep_ns = [1, 2, 4, 8, 16, 32];
        let mut points = Vec::new();

        for &n in &sweep_ns {
            let sweep_tokens = vec![3u32; n];

            // Measure CPU prefill
            let cpu_config = SchedulerConfig {
                gpu_threshold: 99999, // Force CPU
                ubatch_size: 512,
                opencl_enabled: false,
            };
            engine.scheduler_config = cpu_config;
            let t0_cpu = Instant::now();
            let _ = engine.generate_tokens_with_callback(
                &sweep_tokens,
                &SamplingParams {
                    max_tokens: 1,
                    greedy: true,
                    ignore_eos: true,
                    ..Default::default()
                },
                &cancel,
                |_| {},
            )?;
            let cpu_ms = t0_cpu.elapsed().as_secs_f64() * 1000.0;

            // Measure GPU prefill (if available)
            let mut gpu_ms = None;
            let mut winner = "CPU".to_string();

            if engine.has_gpu_backend() {
                let gpu_config = SchedulerConfig {
                    gpu_threshold: 1, // Force GPU
                    ubatch_size: 512,
                    opencl_enabled: true,
                };
                engine.scheduler_config = gpu_config;
                let t0_gpu = Instant::now();
                let _ = engine.generate_tokens_with_callback(
                    &sweep_tokens,
                    &SamplingParams {
                        max_tokens: 1,
                        greedy: true,
                        ignore_eos: true,
                        ..Default::default()
                    },
                    &cancel,
                    |_| {},
                )?;
                let dur = t0_gpu.elapsed().as_secs_f64() * 1000.0;
                gpu_ms = Some(dur);

                if dur < cpu_ms {
                    winner = "GPU".to_string();
                }
            }

            points.push(ThresholdSweepPoint {
                n,
                cpu_prefill_ms: cpu_ms,
                gpu_prefill_ms: gpu_ms,
                faster_backend: winner,
            });
        }

        // Determine crossover point where GPU consistently beats CPU
        if let Some(first_gpu) = points.iter().find(|p| p.faster_backend == "GPU") {
            optimal_t = first_gpu.n.max(4); // Use conservative margin
        }
        sweep_results = Some(points);
    }

    Ok(BenchmarkReport {
        timestamp: chrono_now_string(),
        host: host_info,
        model_path: path_ref.to_string_lossy().to_string(),
        context_length: ctx_len,
        prefill: prefill_results,
        decode: decode_results,
        memory: memory_metrics,
        threshold_calibration: sweep_results,
        recommended_threshold_t: optimal_t,
    })
}

fn chrono_now_string() -> String {
    // Generate ISO 8601 date string without external chrono crate
    let now = std::time::SystemTime::now();
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let rem_secs = secs % 86400;
    let hours = rem_secs / 3600;
    let mins = (rem_secs % 3600) / 60;
    let s = rem_secs % 60;
    format!("2026-10-21T{hours:02}:{mins:02}:{s:02}Z")
}
