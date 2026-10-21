//! Model architecture configuration and shape validation (@C:MMAP, §6.3).
//!
//! Validates GGUF metadata and tensor shapes for Llama and Qwen2 decoder-only models.

use crate::gguf::{GgufFile, GgufValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Supported decoder-only transformer architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArch {
    Llama,
    Qwen2,
    Gemma2,
    MiniCPM3,
    LFM,
}

impl ModelArch {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Gemma2 => "gemma2",
            Self::MiniCPM3 => "minicpm3",
            Self::LFM => "lfm",
        }
    }
}

/// Errors returned during model configuration extraction and tensor shape validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing 'general.architecture' metadata in GGUF")]
    MissingArchitecture,
    #[error(
        "unsupported architecture '{0}' (supported: 'llama', 'qwen2', 'gemma2', 'minicpm3', 'lfm')"
    )]
    UnsupportedArchitecture(String),
    #[error("missing required metadata field '{0}' for architecture '{1}'")]
    MissingField(String, &'static str),
    #[error(
        "invalid configuration: embedding length {embedding} not divisible by head count {heads}"
    )]
    InvalidHeadDimension { embedding: usize, heads: usize },
    #[error("missing required tensor '{0}'")]
    MissingTensor(String),
    #[error("tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
}

/// Validated hyperparameters and architecture details for a transformer model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: ModelArch,
    pub context_length: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub rope_dimension_count: usize,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pre_tokenizer: String,
    pub sliding_window: Option<usize>,
    pub attn_logit_softcapping: Option<f32>,
    pub final_logit_softcapping: Option<f32>,
    pub scale_emb: Option<f32>,
}

impl ModelConfig {
    /// Extracts and validates model hyperparameters from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, ConfigError> {
        let arch_str = gguf
            .get_str("general.architecture")
            .ok_or(ConfigError::MissingArchitecture)?;

        let arch = match arch_str {
            "llama" => ModelArch::Llama,
            "qwen2" => ModelArch::Qwen2,
            "gemma2" => ModelArch::Gemma2,
            "minicpm3" | "minicpm" => ModelArch::MiniCPM3,
            "lfm" | "lfm2" => ModelArch::LFM,
            other => return Err(ConfigError::UnsupportedArchitecture(other.to_string())),
        };

        let prefix = arch.as_str();

        let req_u32 = |suffix: &'static str| -> Result<usize, ConfigError> {
            let key = format!("{prefix}.{suffix}");
            gguf.get_u32(&key)
                .map(|v| v as usize)
                .ok_or(ConfigError::MissingField(key, prefix))
        };

        let context_length = req_u32("context_length")?;
        let embedding_length = req_u32("embedding_length")?;
        let block_count = req_u32("block_count")?;
        let feed_forward_length = req_u32("feed_forward_length")?;
        let head_count = req_u32("attention.head_count")?;

        let head_count_kv = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(head_count);

        if embedding_length % head_count != 0 {
            return Err(ConfigError::InvalidHeadDimension {
                embedding: embedding_length,
                heads: head_count,
            });
        }
        let head_dim = embedding_length / head_count;

        let rms_epsilon = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let rope_freq_base = gguf
            .get_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);

        let rope_dimension_count = gguf
            .get_u32(&format!("{prefix}.rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        // Vocab size from tokenizer.ggml.tokens array length, or embedding tensor dimension
        let vocab_size = if let Some(GgufValue::Array(tokens)) = gguf.get("tokenizer.ggml.tokens") {
            tokens.len()
        } else if let Some(embd) = gguf.tensor("token_embd.weight") {
            if embd.dims.len() >= 2 {
                embd.dims[1] as usize
            } else {
                32000
            }
        } else {
            32000
        };

        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id");
        let pre_tokenizer = gguf
            .get_str("tokenizer.ggml.pre")
            .unwrap_or("default")
            .to_string();

        let sliding_window = gguf
            .get_u32(&format!("{prefix}.attention.sliding_window"))
            .or_else(|| gguf.get_u32(&format!("{prefix}.attention.sliding_window_size")))
            .map(|v| v as usize);

        let attn_logit_softcapping = gguf
            .get_f32(&format!("{prefix}.attention.attn_logit_softcapping"))
            .or_else(|| {
                if arch == ModelArch::Gemma2 {
                    Some(50.0)
                } else {
                    None
                }
            });

        let final_logit_softcapping = gguf
            .get_f32(&format!("{prefix}.final_logit_softcapping"))
            .or_else(|| {
                if arch == ModelArch::Gemma2 {
                    Some(30.0)
                } else {
                    None
                }
            });

        let scale_emb = gguf.get_f32(&format!("{prefix}.scale_emb")).or_else(|| {
            if arch == ModelArch::MiniCPM3 {
                Some(12.0)
            } else {
                None
            }
        });

        Ok(Self {
            arch,
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dim,
            rms_epsilon,
            rope_freq_base,
            rope_dimension_count,
            vocab_size,
            bos_token_id,
            eos_token_id,
            pre_tokenizer,
            sliding_window,
            attn_logit_softcapping,
            final_logit_softcapping,
            scale_emb,
        })
    }

    /// Validates that critical tensor shapes in GGUF match the expected hyperparameters.
    pub fn validate_tensor_shapes(&self, gguf: &GgufFile<'_>) -> Result<(), ConfigError> {
        // Token embedding: [embedding_length, vocab_size]
        if let Some(t) = gguf.tensor("token_embd.weight") {
            let expected = vec![self.embedding_length as u64, self.vocab_size as u64];
            if t.dims != expected {
                return Err(ConfigError::ShapeMismatch {
                    name: "token_embd.weight".to_string(),
                    expected,
                    actual: t.dims.clone(),
                });
            }
        }

        // Output norm: [embedding_length]
        if let Some(t) = gguf.tensor("output_norm.weight") {
            let expected = vec![self.embedding_length as u64];
            if t.dims != expected {
                return Err(ConfigError::ShapeMismatch {
                    name: "output_norm.weight".to_string(),
                    expected,
                    actual: t.dims.clone(),
                });
            }
        }

        // Check layer 0 attention and feed-forward shapes if present
        if let Some(t) = gguf.tensor("blk.0.attn_q.weight") {
            let expected = vec![
                self.embedding_length as u64,
                (self.head_count * self.head_dim) as u64,
            ];
            if t.dims != expected {
                return Err(ConfigError::ShapeMismatch {
                    name: "blk.0.attn_q.weight".to_string(),
                    expected,
                    actual: t.dims.clone(),
                });
            }
        }

        Ok(())
    }

    /// Computes the exact KV cache size in bytes per token (A11).
    ///
    /// `bytes/token = 2 · n_layer · n_head_kv · head_dim · (34 / 32)`
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        // Q8_0 quant block = 34 bytes per 32 values
        let values_per_token = 2 * self.block_count * self.head_count_kv * self.head_dim;
        let blocks = values_per_token.div_ceil(32);
        blocks * 34
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{create_test_gguf, GgufValueType, GGUF_MAGIC, GGUF_VERSION_V3};

    #[test]
    fn test_model_config_extraction() {
        let mut kvs: Vec<(&str, GgufValueType, &[u8])> = Vec::new();

        let arch_bytes = [5u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a'];
        let ctx_bytes = (2048u32).to_le_bytes();
        let embd_bytes = (2048u32).to_le_bytes();
        let blk_bytes = (16u32).to_le_bytes();
        let ffn_bytes = (8192u32).to_le_bytes();
        let heads_bytes = (32u32).to_le_bytes();
        let heads_kv_bytes = (8u32).to_le_bytes();

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

        let test_data = create_test_gguf(GGUF_MAGIC, GGUF_VERSION_V3, &kvs, &[]);
        let gguf = GgufFile::parse(&test_data).unwrap();

        let cfg = ModelConfig::from_gguf(&gguf).expect("config parsing");
        assert_eq!(cfg.arch, ModelArch::Llama);
        assert_eq!(cfg.context_length, 2048);
        assert_eq!(cfg.embedding_length, 2048);
        assert_eq!(cfg.head_count, 32);
        assert_eq!(cfg.head_count_kv, 8);
        assert_eq!(cfg.head_dim, 64);

        // KV cache bytes per token: 2 * 16 layers * 8 kv_heads * 64 dim = 16384 values = 512 blocks of 34 bytes = 17,408 bytes
        assert_eq!(cfg.kv_cache_bytes_per_token(), 17408);
    }
}
