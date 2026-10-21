//! Zero-allocation verification test on per-token hot path ($NOALLOC).
//!
//! Employs a custom counting #[global_allocator] to assert that no heap allocations
//! occur during autoregressive decode forward passes after initialization and warmup.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use sulimo_core::gguf::test_utils::create_test_gguf;
use sulimo_core::gguf::{GgufValueType, GGUF_MAGIC, GGUF_VERSION_V3};
use sulimo_core::kv_cache::KvCache;
use sulimo_core::loader::LoadedModel;
use sulimo_cpu::thread_pool::ThreadPool;
use sulimo_engine::forward_pass::{forward_token, ModelTensors, ScratchBuffers};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNTER: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Delegating directly to System allocator.
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Delegating directly to System allocator.
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[test]
fn test_no_heap_allocation_during_token_decode() {
    let mut kvs: Vec<(&str, GgufValueType, &[u8])> = Vec::new();
    let arch_bytes = [5u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a'];
    let ctx_bytes = (256u32).to_le_bytes();
    let embd_bytes = (64u32).to_le_bytes();
    let blk_bytes = (1u32).to_le_bytes();
    let ffn_bytes = (128u32).to_le_bytes();
    let heads_bytes = (4u32).to_le_bytes();
    let heads_kv_bytes = (2u32).to_le_bytes();

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

    let q4_data_144 = vec![0x55u8; 144];
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

    let q4_144_ref: &'static [u8] = Box::leak(q4_data_144.into_boxed_slice());
    let q4_2304_ref: &'static [u8] = Box::leak(q4_data_2304.into_boxed_slice());
    let q4_1152_ref: &'static [u8] = Box::leak(q4_data_1152.into_boxed_slice());
    let q4_4608_ref: &'static [u8] = Box::leak(q4_data_4608.into_boxed_slice());
    let f32_256_ref: &'static [u8] = Box::leak(f32_data_256.into_boxed_slice());

    add_tensor("token_embd.weight", &[64, 4], 2, q4_144_ref);
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

    let gguf_bytes = create_test_gguf(GGUF_MAGIC, GGUF_VERSION_V3, &kvs, &tensors);

    let temp_path =
        std::env::temp_dir().join(format!("sulimo_noalloc_test_{}.gguf", std::process::id()));
    let mut file = File::create(&temp_path).unwrap();
    file.write_all(&gguf_bytes).unwrap();
    file.flush().unwrap();
    drop(file);

    let mut model = LoadedModel::load(&temp_path, Some(256)).expect("load model");
    let mut scratch = ScratchBuffers::new(&model.config);
    let mut kv_cache = KvCache::allocate(&mut model.arena, &model.config, 256).expect("kv cache");
    let mut pool = ThreadPool::new(Some(1));

    let model_tensors = ModelTensors::build(&model);

    // Warmup pass (allows any one-time initializers to run)
    let _ = forward_token(
        &model.config,
        &model_tensors,
        &mut kv_cache,
        0,
        0,
        &mut scratch,
        &mut pool,
    );

    // Reset allocation counter and activate tracking on the decode hot path
    ALLOCATION_COUNTER.store(0, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);

    // Execute 3 subsequent token decode steps
    for step in 1..=3 {
        let _ = forward_token(
            &model.config,
            &model_tensors,
            &mut kv_cache,
            1,
            step,
            &mut scratch,
            &mut pool,
        );
    }

    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    let allocations = ALLOCATION_COUNTER.load(Ordering::SeqCst);

    let _ = std::fs::remove_file(&temp_path);

    println!("Total heap allocations on decode hot path: {allocations}");
    assert_eq!(
        allocations, 0,
        "violation of $NOALLOC: {allocations} heap allocations occurred during token decode!"
    );
}
