//! Token sampling strategies with repetition penalty and seeded RNG (@C:ENG).
//!
//! Supports greedy, temperature, top-K, and top-P (nucleus) sampling without heap allocations.

use serde::{Deserialize, Serialize};

/// Hyperparameters controlling token generation and sampling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature scaling (0.0 = greedy).
    pub temperature: f32,
    /// Top-K filtering threshold (0 = disabled).
    pub top_k: usize,
    /// Top-P (nucleus) probability mass threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty factor for recently emitted tokens (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Random number generator seed.
    pub seed: u64,
    /// Forced greedy selection (ignores temperature and top-k/top-p).
    pub greedy: bool,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 42,
            greedy: false,
            max_tokens: 128,
        }
    }
}

/// Selects the next token ID from an unnormalized logit distribution.
pub fn sample_token(
    logits: &mut [f32],
    past_tokens: &[u32],
    params: &SamplingParams,
    rng: &mut fastrand::Rng,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // 1. Greedy selection bypass
    if params.greedy || params.temperature <= 0.0 {
        return argmax(logits);
    }

    // 2. Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        for &tok in past_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= params.repetition_penalty;
                } else {
                    logits[idx] *= params.repetition_penalty;
                }
            }
        }
    }

    // 3. Apply temperature scaling
    let inv_temp = 1.0 / params.temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }

    // 4. Softmax normalization
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for logit in logits.iter_mut() {
        let e = (*logit - max_logit).exp();
        *logit = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-9);
    for prob in logits.iter_mut() {
        *prob *= inv_sum;
    }

    // 5. Top-K filtering
    if params.top_k > 0 && params.top_k < logits.len() {
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, &(idx, _)) in indexed.iter().enumerate() {
            if rank >= params.top_k {
                logits[idx] = 0.0;
            }
        }

        // Re-normalize after top-k
        let top_sum: f32 = logits.iter().sum();
        if top_sum > 0.0 {
            let inv_top_sum = 1.0 / top_sum;
            for p in logits.iter_mut() {
                *p *= inv_top_sum;
            }
        }
    }

    // 6. Top-P (nucleus) filtering and roulette wheel sampling
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum_sum = 0.0f32;
    let mut cut_off = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cum_sum += p;
        if cum_sum >= params.top_p {
            cut_off = i + 1;
            break;
        }
    }

    // Sample from top-P subset using seeded RNG
    let r = rng.f32() * cum_sum;
    let mut current_sum = 0.0f32;
    for &(idx, prob) in &indexed[..cut_off] {
        current_sum += prob;
        if r <= current_sum {
            return idx as u32;
        }
    }

    // Fallback to top token
    indexed.first().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

fn argmax(slice: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampler() {
        let mut logits = [-1.0f32, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            greedy: true,
            ..Default::default()
        };
        let mut rng = fastrand::Rng::with_seed(42);

        let tok = sample_token(&mut logits, &[], &params, &mut rng);
        assert_eq!(tok, 1);
    }
}
