//! Memory budget planning and 5.2 GB unified RAM ceiling enforcement (@C:MMAP, $MEM<8G).
//!
//! Pre-calculates post-repack weight footprints, dynamic Q8_0 KV cache allocations (A11),
//! and execution scratch buffers to guarantee the inference engine stays strictly within
//! the 5.2 GB unified memory envelope.

use crate::gguf::{GgmlType, GgufFile};
use crate::model_config::ModelConfig;
use crate::repack::Q4_0_BLOCK_SIZE;
use crate::sysmem::MAX_RAM_BUDGET_BYTES;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors returned when memory budget planning fails.
#[derive(Debug, Error)]
pub enum BudgetError {
    #[error(
        "model exceeds 5.2 GB RAM ceiling: total required {required_gib:.2} GiB > budget {budget_gib:.2} GiB\n\
         Breakdown:\n\
         - Weights (repacked SoA): {weights_gib:.2} GiB\n\
         - KV Cache (ctx {context_len}): {kv_gib:.2} GiB\n\
         - Scratch workspace:     {scratch_gib:.2} GiB\n\
         - Safety headroom (5%):  {slack_gib:.2} GiB"
    )]
    ExceedsBudget {
        required_gib: f64,
        budget_gib: f64,
        weights_gib: f64,
        kv_gib: f64,
        scratch_gib: f64,
        slack_gib: f64,
        context_len: usize,
    },
}

/// Detailed itemized memory plan for model loading and execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudgetPlan {
    /// Total bytes required by all model weights post-repacking to SoA.
    pub weights_bytes: usize,
    /// Bytes required by the Q8_0 KV cache at selected context length.
    pub kv_cache_bytes: usize,
    /// Scratch memory for activations, attention scores, and intermediate GEMM buffers.
    pub scratch_bytes: usize,
    /// 5% safety margin to guard against allocator padding and transient allocations.
    pub slack_bytes: usize,
    /// Total planned memory footprint (`weights + kv + scratch + slack`).
    pub total_planned_bytes: usize,
    /// The context length selected to satisfy the 5.2 GB budget constraint.
    pub context_length: usize,
    /// Whether this plan fits within the 5.2 GB budget ceiling.
    pub fits_budget: bool,
}

impl MemoryBudgetPlan {
    /// Format human-readable summary of the memory plan.
    pub fn summary(&self) -> String {
        format!(
            "Weights: {:.2} GiB | KV Cache ({} ctx): {:.2} GiB | Scratch: {:.2} MiB | Total: {:.2} GiB / {:.2} GiB",
            self.weights_bytes as f64 / 1_073_741_824.0,
            self.context_length,
            self.kv_cache_bytes as f64 / 1_073_741_824.0,
            self.scratch_bytes as f64 / 1_048_576.0,
            self.total_planned_bytes as f64 / 1_073_741_824.0,
            MAX_RAM_BUDGET_BYTES as f64 / 1_073_741_824.0,
        )
    }
}

pub struct MemoryBudget;

impl MemoryBudget {
    /// Computes the post-repack weight footprint of all tensors in the GGUF file.
    pub fn calculate_weights_footprint(gguf: &GgufFile<'_>) -> usize {
        let mut total = 0;
        for t in &gguf.tensors {
            let padded_size = match t.ggml_type {
                GgmlType::Q4_0 => {
                    let num_elements = t.num_elements();
                    let num_blocks = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
                    // SoA layout: scales (f16) + quants (16 bytes per block), each padded to 64 bytes
                    let scales_padded = (num_blocks * 2 + 63) & !63;
                    let quants_padded = (num_blocks * 16 + 63) & !63;
                    scales_padded + quants_padded
                }
                _ => (t.raw_size + 63) & !63,
            };
            // Each tensor allocation in the arena is page-aligned to 4096 bytes
            let page_aligned = (padded_size + 4095) & !4095;
            total += page_aligned;
        }
        total
    }

    /// Estimates scratch activation memory for forward pass execution (ubatch = 512).
    pub fn estimate_scratch_bytes(config: &ModelConfig, context_length: usize) -> usize {
        let ubatch = 512.min(context_length);
        // Activation vectors: ubatch * embedding_length * 4 (float)
        let act = ubatch * config.embedding_length * 4;
        // Feed-forward hidden activations: ubatch * feed_forward_length * 4
        let ffn = ubatch * config.feed_forward_length * 4;
        // Attention score matrix: head_count * ubatch * context_length * 4
        let attn = config.head_count * ubatch * context_length * 4;
        // OpenCL / CPU working buffer slack
        let extra = 64 * 1024 * 1024; // 64 MB

        let total = act + ffn + attn + extra;
        // Align to 4096-byte page
        (total + 4095) & !4095
    }

    /// Generates a pre-flight memory plan, clamping context length if needed to fit 5.2 GB.
    pub fn plan(
        gguf: &GgufFile<'_>,
        config: &ModelConfig,
        requested_ctx: Option<usize>,
    ) -> Result<MemoryBudgetPlan, BudgetError> {
        let weights_bytes = Self::calculate_weights_footprint(gguf);
        let max_budget = MAX_RAM_BUDGET_BYTES as usize;

        let initial_ctx = requested_ctx
            .unwrap_or(config.context_length)
            .min(config.context_length);

        // Candidate context lengths to try in order
        let mut candidate_contexts = vec![initial_ctx];
        for &step in &[8192, 4096, 2048, 1024, 512] {
            if step < initial_ctx && !candidate_contexts.contains(&step) {
                candidate_contexts.push(step);
            }
        }

        let bytes_per_tok = config.kv_cache_bytes_per_token();

        let mut best_plan = None;

        for &ctx in &candidate_contexts {
            let kv_cache_bytes = ctx * bytes_per_tok;
            let scratch_bytes = Self::estimate_scratch_bytes(config, ctx);
            let subtotal = weights_bytes + kv_cache_bytes + scratch_bytes;
            let slack_bytes = subtotal / 20; // 5% safety margin
            let total_planned_bytes = subtotal + slack_bytes;

            let plan = MemoryBudgetPlan {
                weights_bytes,
                kv_cache_bytes,
                scratch_bytes,
                slack_bytes,
                total_planned_bytes,
                context_length: ctx,
                fits_budget: total_planned_bytes <= max_budget,
            };

            if plan.fits_budget {
                return Ok(plan);
            }

            if best_plan.is_none() {
                best_plan = Some(plan);
            }
        }

        let failed = best_plan.unwrap();
        Err(BudgetError::ExceedsBudget {
            required_gib: failed.total_planned_bytes as f64 / 1_073_741_824.0,
            budget_gib: max_budget as f64 / 1_073_741_824.0,
            weights_gib: failed.weights_bytes as f64 / 1_073_741_824.0,
            kv_gib: failed.kv_cache_bytes as f64 / 1_073_741_824.0,
            scratch_gib: failed.scratch_bytes as f64 / 1_073_741_824.0,
            slack_gib: failed.slack_bytes as f64 / 1_073_741_824.0,
            context_len: failed.context_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{create_test_gguf, GGUF_MAGIC, GGUF_VERSION_V3};
    use crate::model_config::ModelArch;

    #[test]
    fn test_budget_planning_within_budget() {
        let dummy_q4 = vec![0u8; 18 * 100]; // 100 blocks = 1800 bytes
        let test_data = create_test_gguf(
            GGUF_MAGIC,
            GGUF_VERSION_V3,
            &[],
            &[(
                "blk.0.attn_q.weight",
                &[3200],
                2, /* Q4_0 */
                0,
                &dummy_q4,
            )],
        );
        let gguf = GgufFile::parse(&test_data).unwrap();

        let config = ModelConfig {
            arch: ModelArch::Llama,
            context_length: 2048,
            embedding_length: 2048,
            block_count: 16,
            feed_forward_length: 8192,
            head_count: 32,
            head_count_kv: 8,
            head_dim: 64,
            rms_epsilon: 1e-5,
            rope_freq_base: 10000.0,
            rope_dimension_count: 64,
            vocab_size: 32000,
            bos_token_id: None,
            eos_token_id: None,
            pre_tokenizer: "default".to_string(),
            sliding_window: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            scale_emb: None,
        };

        let plan = MemoryBudget::plan(&gguf, &config, Some(2048)).expect("plan must fit budget");
        assert!(plan.fits_budget);
        assert!(plan.total_planned_bytes <= MAX_RAM_BUDGET_BYTES as usize);
        assert_eq!(plan.context_length, 2048);
    }
}
