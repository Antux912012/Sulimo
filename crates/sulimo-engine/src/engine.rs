//! High-level inference engine coordinating model, tokenizer, and sampler (@C:ENG).
//!
//! Provides the primary application programming interface (`Engine::load`, `Engine::generate`)
//! for headless CLI runners and reactive desktop user interfaces.

use crate::forward_pass::{forward_token, ModelTensors, ScratchBuffers};
use crate::sampler::{sample_token, SamplingParams};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use sulimo_core::gguf::GgufMmap;
use sulimo_core::kv_cache::KvCache;
use sulimo_core::loader::{LoadedModel, LoaderError};
use sulimo_core::tokenizer::{Tokenizer, TokenizerError};
use sulimo_cpu::thread_pool::ThreadPool;
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
}

/// The local inference engine instance.
pub struct Engine {
    pub model: LoadedModel,
    pub tokenizer: Tokenizer,
    pub kv_cache: KvCache,
    pub scratch: ScratchBuffers,
    pub thread_pool: ThreadPool,
}

impl Engine {
    /// Loads a model from disk, sets up tokenizer, allocates KV cache and scratch memory.
    pub fn load<P: AsRef<Path>>(
        path: P,
        requested_ctx: Option<usize>,
        threads: Option<usize>,
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

        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            scratch,
            thread_pool,
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

        // 1. Prompt Ingestion / Prefill phase
        let mut next_token_logits = &[][..];
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            next_token_logits = forward_token(
                &self.model.config,
                &model_tensors,
                &mut self.kv_cache,
                token_id,
                pos,
                &mut self.scratch,
                &mut self.thread_pool,
            );
        }

        let prefill_duration = start_time.elapsed().as_secs_f64();
        let prefill_tok_s = if prefill_duration > 0.0 {
            prompt_tokens.len() as f64 / prefill_duration
        } else {
            0.0
        };

        let mut decode_start = Instant::now();
        let mut completion_tokens = 0;

        // 2. Autoregressive token decode phase
        let mut current_pos = prompt_tokens.len();
        let mut current_logits = next_token_logits.to_vec();

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
                // Circular ring buffer wraps around (A12)
                current_pos %= self.model.plan.context_length;
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
        })
    }
}
