//! High-level inference engine coordinating model, tokenizer, and sampler (@C:ENG, §7 M4).
//!
//! Provides the primary application programming interface (`Engine::load`, `Engine::generate_with_callback`)
//! for headless CLI runners and reactive desktop user interfaces, coordinating asymmetric
//! iGPU prefill and physical-core CPU decode.

use crate::forward_pass::{forward_token, ModelTensors, ScratchBuffers};
use crate::sampler::{sample_token, SamplingParams};
#[cfg(feature = "opencl")]
use crate::scheduler::OclEngineState;
use crate::scheduler::{lookup_token_embedding_f16, repack_context, SchedulerConfig};
use half::f16;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use sulimo_core::gguf::GgufMmap;
use sulimo_core::kv_cache::KvCache;
use sulimo_core::loader::{LoadedModel, LoaderError};
use sulimo_core::tokenizer::{Tokenizer, TokenizerError};
use sulimo_cpu::thread_pool::ThreadPool;
#[cfg(feature = "opencl")]
use sulimo_ocl::execute_prefill_ubatch;
use thiserror::Error;

/// Errors returned during engine initialization or generation.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("loader error: {0}")]
    Loader(#[from] LoaderError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("GGUF mmap error: {0}")]
    Gguf(#[from] sulimo_core::GgufError),
    #[cfg(feature = "opencl")]
    #[error("OpenCL error: {0}")]
    Ocl(#[from] sulimo_ocl::OclError),
    #[error("scheduler error: {0}")]
    Scheduler(#[from] crate::scheduler::SchedulerError),
    #[error("generation cancelled by user")]
    Cancelled,
}

/// Generation performance and telemetry metrics.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: f64,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
    pub total_time_ms: f64,
    pub prefill_backend: String,
    pub context_repacks: usize,
}

/// The local inference engine instance.
pub struct Engine {
    pub model: LoadedModel,
    pub tokenizer: Tokenizer,
    pub kv_cache: KvCache,
    pub scratch: ScratchBuffers,
    pub thread_pool: ThreadPool,
    pub scheduler_config: SchedulerConfig,
    #[cfg(feature = "opencl")]
    pub ocl_state: Option<OclEngineState>,
}

impl Engine {
    /// Loads a model from disk with default scheduler configuration.
    pub fn load<P: AsRef<Path>>(
        path: P,
        requested_ctx: Option<usize>,
        threads: Option<usize>,
    ) -> Result<Self, EngineError> {
        Self::load_with_config(path, requested_ctx, threads, SchedulerConfig::default())
    }

    /// Loads a model from disk with customized scheduler configuration.
    pub fn load_with_config<P: AsRef<Path>>(
        path: P,
        requested_ctx: Option<usize>,
        threads: Option<usize>,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, EngineError> {
        let path_ref = path.as_ref();
        let gguf_mmap = GgufMmap::open(path_ref)?;
        let tokenizer = Tokenizer::from_gguf(gguf_mmap.file())?;

        let mut model = LoadedModel::load(path_ref, requested_ctx)?;
        let scratch = ScratchBuffers::new(&model.config);
        let kv_cache =
            KvCache::allocate(&mut model.arena, &model.config, model.plan.context_length)
                .map_err(LoaderError::from)?;

        let thread_pool = ThreadPool::new(threads);

        #[cfg(feature = "opencl")]
        let ocl_state = if scheduler_config.opencl_enabled {
            match OclEngineState::try_init(&mut model, &kv_cache, scheduler_config.ubatch_size) {
                Ok(state) => {
                    eprintln!(
                        "[Scheduler] OpenCL initialization: active on {} ({}, {:?}). GPU prompt prefill enabled.",
                        state.ctx.device_info.device_name,
                        state.ctx.device_info.driver_version,
                        state.ctx.active_variant
                    );
                    Some(state)
                }
                Err(e) => {
                    eprintln!(
                        "[Scheduler] OpenCL initialization: unavailable ({e}). Running in CPU-only mode."
                    );
                    None
                }
            }
        } else {
            eprintln!("[Scheduler] OpenCL explicitly disabled by configuration. Running in CPU-only mode.");
            None
        };

        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            scratch,
            thread_pool,
            scheduler_config,
            #[cfg(feature = "opencl")]
            ocl_state,
        })
    }

    /// Generates tokens from a prompt text, streaming strings to a callback function.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        params: &SamplingParams,
        cancel: &AtomicBool,
        mut on_token: F,
    ) -> Result<GenerationStats, EngineError>
    where
        F: FnMut(&str),
    {
        let prompt_tokens = self.tokenizer.encode(prompt, true);
        if prompt_tokens.is_empty() {
            return Ok(GenerationStats::default());
        }

        let mut rng = fastrand::Rng::with_seed(params.seed);
        let mut past_tokens = Vec::with_capacity(prompt_tokens.len() + params.max_tokens);
        past_tokens.extend_from_slice(&prompt_tokens);

        let model_tensors = ModelTensors::build(&self.model);

        let start_time = Instant::now();
        let mut ttft = 0.0;
        let mut context_repacks = 0;

        let n = prompt_tokens.len();
        let use_gpu = self.scheduler_config.opencl_enabled
            && n >= self.scheduler_config.gpu_threshold
            && self.has_gpu_backend();

        let prefill_backend_str = if use_gpu {
            #[cfg(feature = "opencl")]
            if let Some(ref ocl) = self.ocl_state {
                eprintln!(
                    "[Scheduler] Prefill: using GPU (prompt tokens N={n} >= threshold T={}, ubatch_size={}) on {}",
                    self.scheduler_config.gpu_threshold,
                    self.scheduler_config.ubatch_size,
                    ocl.ctx.device_info.device_name
                );
            }
            "GPU-OpenCL"
        } else {
            if !self.has_gpu_backend() {
                eprintln!(
                    "[Scheduler] Prefill: using CPU (OpenCL unavailable or disabled; executing on physical cores)"
                );
            } else {
                eprintln!(
                    "[Scheduler] Prefill: using CPU (prompt tokens N={n} < threshold T={})",
                    self.scheduler_config.gpu_threshold
                );
            }
            "CPU-AVX2"
        };

        let mut current_pos = 0;
        let mut current_logits = vec![0.0f32; self.model.config.vocab_size];

        // 1. Prompt Prefill Phase (GPU when N >= T, CPU otherwise)
        if use_gpu {
            #[cfg(feature = "opencl")]
            if let Some(ref mut ocl) = self.ocl_state {
                let ubatch_size = self.scheduler_config.ubatch_size;
                let embd = self.model.config.embedding_length;
                let mut token_offset = 0;

                while token_offset < n {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(EngineError::Cancelled);
                    }

                    let chunk_len = (n - token_offset).min(ubatch_size);
                    let chunk_tokens = &prompt_tokens[token_offset..token_offset + chunk_len];

                    // Check context overflow before executing ubatch
                    if current_pos + chunk_len > self.model.plan.context_length {
                        let (new_past, _new_pos) =
                            repack_context(&past_tokens, self.model.plan.context_length);
                        eprintln!(
                            "[Context Repack] Context capacity ({}) reached at pos {current_pos}. Repacking context to {} tokens, re-prefilling on GPU.",
                            self.model.plan.context_length,
                            new_past.len()
                        );
                        context_repacks += 1;
                        past_tokens = new_past;
                        current_pos = 0;
                        token_offset = 0;
                        continue;
                    }

                    // Populate scratch.x with F16 token embeddings
                    {
                        let x_f16: &mut [f16] = unsafe {
                            std::slice::from_raw_parts_mut(
                                ocl.scratch_x_slice.as_ptr() as *mut f16,
                                chunk_len * embd,
                            )
                        };
                        for (i, &tok) in chunk_tokens.iter().enumerate() {
                            lookup_token_embedding_f16(
                                model_tensors.token_embd,
                                embd,
                                tok,
                                &mut x_f16[i * embd..(i + 1) * embd],
                            );
                        }
                    }

                    // Execute full-layer prefill for this ubatch on GPU ($BATT, A9)
                    execute_prefill_ubatch(
                        &ocl.ctx,
                        &self.model.config,
                        &ocl.model_tensors,
                        &mut ocl.kv_cache,
                        &mut ocl.scratch,
                        current_pos,
                        chunk_len,
                    )?;

                    current_pos += chunk_len;
                    token_offset += chunk_len;

                    // If final chunk of the prompt, read output logits for sampling
                    if token_offset == n {
                        let map_ptr = ocl.scratch.logits.map_read(&ocl.ctx.queue)?;
                        let logits_f32: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                map_ptr as *const f32,
                                self.model.config.vocab_size,
                            )
                        };
                        current_logits.copy_from_slice(logits_f32);
                        let _ = ocl.scratch.logits.unmap(&ocl.ctx.queue, map_ptr);
                    }
                }
            }
        } else {
            // CPU Prefill path
            for &token_id in &prompt_tokens {
                if cancel.load(Ordering::Relaxed) {
                    return Err(EngineError::Cancelled);
                }

                if current_pos >= self.model.plan.context_length {
                    let (new_past, _) =
                        repack_context(&past_tokens, self.model.plan.context_length);
                    eprintln!(
                        "[Context Repack] Context capacity ({}) reached at pos {current_pos}. Repacking context to {} tokens, re-prefilling on CPU.",
                        self.model.plan.context_length,
                        new_past.len()
                    );
                    context_repacks += 1;
                    past_tokens = new_past;
                    current_pos = 0;
                }

                let logits_ref = forward_token(
                    &self.model.config,
                    &model_tensors,
                    &mut self.kv_cache,
                    token_id,
                    current_pos,
                    &mut self.scratch,
                    &mut self.thread_pool,
                );
                current_logits.copy_from_slice(logits_ref);
                current_pos += 1;
            }
        }

        let prefill_duration = start_time.elapsed().as_secs_f64();
        let prefill_tok_s = if prefill_duration > 0.0 {
            prompt_tokens.len() as f64 / prefill_duration
        } else {
            0.0
        };

        // 2. Autoregressive Token Decode Phase (Always CPU physical cores, GPU in RC6 deep sleep)
        eprintln!(
            "[Scheduler] Decode: using CPU physical cores pinned (N=1), GPU idle -> entering RC6 deep sleep"
        );

        let mut decode_start = Instant::now();
        let mut completion_tokens = 0;

        while completion_tokens < params.max_tokens {
            if cancel.load(Ordering::Relaxed) {
                break;
            }

            let next_tok = sample_token(&mut current_logits, &past_tokens, params, &mut rng);

            if completion_tokens == 0 {
                ttft = start_time.elapsed().as_secs_f64() * 1000.0;
                decode_start = Instant::now();
            }

            // Check EOS / EOT stop condition
            if Some(next_tok) == self.tokenizer.eos_token_id
                || Some(next_tok) == self.tokenizer.eot_token_id
            {
                break;
            }

            let tok_str = self.tokenizer.decode(&[next_tok]).unwrap_or_default();
            on_token(&tok_str);

            past_tokens.push(next_tok);
            completion_tokens += 1;

            if current_pos >= self.model.plan.context_length {
                let (new_past, _) = repack_context(&past_tokens, self.model.plan.context_length);
                eprintln!(
                    "[Context Repack] Context capacity ({}) reached during decode at pos {current_pos}. Repacking context to {} tokens.",
                    self.model.plan.context_length,
                    new_past.len()
                );
                context_repacks += 1;
                past_tokens = new_past;
                current_pos = 0;

                // Re-prefill past context on CPU
                for &p_tok in &past_tokens {
                    let logits_ref = forward_token(
                        &self.model.config,
                        &model_tensors,
                        &mut self.kv_cache,
                        p_tok,
                        current_pos,
                        &mut self.scratch,
                        &mut self.thread_pool,
                    );
                    current_logits.copy_from_slice(logits_ref);
                    current_pos += 1;
                }
                continue;
            }

            let logits_ref = forward_token(
                &self.model.config,
                &model_tensors,
                &mut self.kv_cache,
                next_tok,
                current_pos,
                &mut self.scratch,
                &mut self.thread_pool,
            );
            current_logits.copy_from_slice(logits_ref);
            current_pos += 1;
        }

        let decode_duration = decode_start.elapsed().as_secs_f64();
        let decode_tok_s = if completion_tokens > 1 && decode_duration > 0.0 {
            (completion_tokens - 1) as f64 / decode_duration
        } else {
            0.0
        };

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            ttft_ms: ttft,
            prefill_tok_per_sec: prefill_tok_s,
            decode_tok_per_sec: decode_tok_s,
            total_time_ms,
            prefill_backend: prefill_backend_str.to_string(),
            context_repacks,
        })
    }

    /// Returns whether the engine has an active GPU OpenCL backend initialized.
    pub fn has_gpu_backend(&self) -> bool {
        #[cfg(feature = "opencl")]
        {
            self.ocl_state.is_some()
        }
        #[cfg(not(feature = "opencl"))]
        {
            false
        }
    }
}
