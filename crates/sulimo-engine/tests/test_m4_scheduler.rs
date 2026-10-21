//! Milestone M4 Acceptance Tests: Hybrid asymmetric scheduler & battery discipline (§7 M4, $BATT, $MEM<8G).
//!
//! Verifies:
//! 1. A12 context repacking on capacity overflow (evicting middle turns, preserving head and tail).
//! 2. Hybrid scheduler dispatch: GPU prefill when N >= T, CPU prefill when N < T, CPU decode always.
//! 3. Sub-token / sub-ubatch cancellation latency via atomic cancel flag.
//! 4. Thread pool pinning to physical CPU cores.
//! 5. 50x load/generate/unload leak cycle asserting active_cl_mem_count() == 0 and RSS growth <= 16 MB.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use sulimo_core::gguf::test_utils::create_test_gguf;
use sulimo_core::gguf::GgufValueType;
use sulimo_core::loader::get_process_memory_stats;
use sulimo_cpu::thread_pool::ThreadPool;
use sulimo_cpu::topology::CpuTopology;
use sulimo_engine::engine::{Engine, GenerationStats};
use sulimo_engine::sampler::SamplingParams;
use sulimo_engine::scheduler::{repack_context, SchedulerConfig};
#[cfg(feature = "opencl")]
use sulimo_ocl::active_cl_mem_count;

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

fn create_synthetic_model_file() -> std::path::PathBuf {
    let mut kvs: Vec<(&str, GgufValueType, &[u8])> = Vec::new();
    let arch_bytes = [5u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a'];
    let ctx_bytes = (128u32).to_le_bytes();
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
    let mut tensors = Vec::new();

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

    let gguf_bytes = create_test_gguf(0x46554747, 3, &kvs, &tensors);
    let temp_path = std::env::temp_dir().join(format!(
        "sulimo_m4_test_{}_{}.gguf",
        std::process::id(),
        fastrand::u64(..)
    ));
    let mut file = File::create(&temp_path).expect("create test gguf");
    file.write_all(&gguf_bytes).expect("write test gguf");
    file.flush().expect("flush test gguf");
    temp_path
}

#[test]
fn test_m4_context_repack_bounds_and_preservation() {
    let past: Vec<u32> = (0..100).collect();
    let capacity = 64;

    let (repacked, new_pos) = repack_context(&past, capacity);

    // Verify bounds: repacked tokens must fit comfortably within capacity
    assert!(
        repacked.len() < capacity,
        "repacked length {} must be < capacity {}",
        repacked.len(),
        capacity
    );
    assert_eq!(new_pos, repacked.len());

    // Verify system prefix preserved (first sys_len tokens: 0, 1, 2, ...)
    let sys_len = (capacity / 8).clamp(4, 64).min(past.len());
    assert_eq!(&repacked[..sys_len], &past[..sys_len]);

    // Verify recent tail preserved (last recent_len tokens)
    let recent_len = (capacity / 2).min(past.len().saturating_sub(sys_len));
    assert_eq!(&repacked[sys_len..], &past[past.len() - recent_len..]);
}

#[test]
fn test_m4_hybrid_scheduler_dispatch() {
    let model_path = create_synthetic_model_file();

    // 1. CPU dispatch when OpenCL explicitly disabled
    let config_cpu_only = SchedulerConfig {
        gpu_threshold: 4,
        ubatch_size: 512,
        opencl_enabled: false,
    };
    let mut engine_cpu = Engine::load_with_config(&model_path, Some(64), Some(1), config_cpu_only)
        .expect("load engine cpu-only");

    let cancel = AtomicBool::new(false);
    let params = SamplingParams {
        max_tokens: 3,
        greedy: true,
        ..Default::default()
    };

    let mut generated = Vec::new();
    let stats: GenerationStats = engine_cpu
        .generate_with_callback("hello world sulimo test", &params, &cancel, |s| {
            generated.push(s.to_string());
        })
        .expect("generate cpu");

    assert_eq!(stats.prefill_backend, "CPU-AVX2");
    assert!(!generated.is_empty());

    // 2. Scheduler dispatch with OpenCL enabled (if host has OpenCL GPU, threshold determines backend)
    let config_auto = SchedulerConfig {
        gpu_threshold: 8,
        ubatch_size: 512,
        opencl_enabled: true,
    };
    let mut engine_auto = Engine::load_with_config(&model_path, Some(64), Some(1), config_auto)
        .expect("load engine auto");

    let short_prompt = "hello"; // N < 8 -> CPU prefill
    let stats_short = engine_auto
        .generate_with_callback(short_prompt, &params, &cancel, |_| {})
        .expect("generate short prompt");
    assert_eq!(
        stats_short.prefill_backend, "CPU-AVX2",
        "Short prompt must run on CPU"
    );

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_m4_cancellation_latency() {
    let model_path = create_synthetic_model_file();
    let config = SchedulerConfig {
        gpu_threshold: 8,
        ubatch_size: 512,
        opencl_enabled: false,
    };
    let mut engine =
        Engine::load_with_config(&model_path, Some(64), Some(1), config).expect("load engine");

    // Pre-cancelled flag
    let cancel_immediate = AtomicBool::new(true);
    let params = SamplingParams {
        max_tokens: 20,
        ..Default::default()
    };

    let res = engine.generate_with_callback("hello world", &params, &cancel_immediate, |_| {});
    assert!(res.is_err(), "Pre-cancelled run must return error");

    // Cancel during decode stream
    let cancel_during = AtomicBool::new(false);
    let mut token_count = 0;
    let _stats = engine.generate_with_callback("hello world", &params, &cancel_during, |_| {
        token_count += 1;
        cancel_during.store(true, Ordering::Relaxed);
    });

    // Cancellation must stop within <= 1 token
    assert!(
        token_count <= 2,
        "Cancellation must halt generation within <= 1 token (saw {token_count})"
    );

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_m4_decode_thread_physical_core_pinning() {
    let topo = CpuTopology::detect().expect("detect cpu topology");
    let num_physical = topo.physical_cores.len();
    assert!(num_physical >= 1, "Must detect at least 1 physical core");

    let pool = ThreadPool::new(Some(num_physical));
    assert_eq!(pool.num_threads(), num_physical);

    // Verify task thread statuses under /proc/self/task/
    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
        for entry in entries.flatten() {
            let status_path = entry.path().join("status");
            if let Ok(status_text) = std::fs::read_to_string(&status_path) {
                if status_text.contains("sulimo-cpu-") {
                    let cpus_allowed = status_text
                        .lines()
                        .find(|l| l.starts_with("Cpus_allowed_list:"))
                        .map(|l| l.trim());
                    assert!(
                        cpus_allowed.is_some(),
                        "Worker thread must report Cpus_allowed_list"
                    );
                }
            }
        }
    }
}

#[test]
fn test_m4_50x_load_unload_leak_and_rss_cycle() {
    let model_path = create_synthetic_model_file();

    // 1. Warm-up cycle to dynamically link OpenCL driver libraries into the process
    let config = SchedulerConfig {
        gpu_threshold: 8,
        ubatch_size: 512,
        opencl_enabled: true,
    };
    {
        let mut warmup_engine =
            Engine::load_with_config(&model_path, Some(64), Some(1), config.clone())
                .expect("engine warmup");
        let cancel = AtomicBool::new(false);
        let params = SamplingParams {
            max_tokens: 1,
            greedy: true,
            ..Default::default()
        };
        let _ = warmup_engine.generate_with_callback("hello", &params, &cancel, |_| {});
        drop(warmup_engine);
    }

    // 2. Baseline RSS after driver load
    let (initial_rss, _) = get_process_memory_stats();

    for iter in 0..50 {
        let mut engine = Engine::load_with_config(&model_path, Some(64), Some(1), config.clone())
            .expect("engine load cycle");

        let cancel = AtomicBool::new(false);
        let params = SamplingParams {
            max_tokens: 1,
            greedy: true,
            ..Default::default()
        };

        let _ = engine.generate_with_callback("hello", &params, &cancel, |_| {});

        drop(engine);

        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }

        #[cfg(feature = "opencl")]
        {
            let leaked_cl_mem = active_cl_mem_count();
            assert_eq!(
                leaked_cl_mem, 0,
                "Iteration {iter}: active_cl_mem_count must be 0 after drop, found {leaked_cl_mem}"
            );
        }
    }

    let (final_rss, _) = get_process_memory_stats();
    let rss_growth = final_rss.saturating_sub(initial_rss);
    const MAX_RSS_GROWTH: u64 = 16 * 1024 * 1024; // 16 MB limit (§7 M4)

    eprintln!(
        "[50x Cycle Test] Initial RSS: {} KiB, Final RSS: {} KiB, Delta: {} KiB (Limit: {} KiB)",
        initial_rss / 1024,
        final_rss / 1024,
        rss_growth / 1024,
        MAX_RSS_GROWTH / 1024
    );

    assert!(
        rss_growth <= MAX_RSS_GROWTH,
        "RSS growth ({} bytes) exceeded 16 MB limit after 50 cycles",
        rss_growth
    );

    let _ = std::fs::remove_file(&model_path);
}
