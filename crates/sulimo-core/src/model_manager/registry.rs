//! Curated model registry with verified GGUF endpoints, hashes, and hardware budgets.
//!
//! Provides specifications for approved models tested for the 8 GB RAM / 5.20 GiB
//! unified memory budget ceiling and 30 GB SSD storage constraint.

use crate::model_config::ModelArch;
use crate::sysmem::MAX_RAM_BUDGET_BYTES;
use serde::{Deserialize, Serialize};

/// Supported chat prompt template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptTemplateType {
    /// ChatML format used by Qwen2, Liquid AI LFM, and MiniCPM:
    /// `<|im_start|>role\ncontent<|im_end|>`
    ChatMl,
    /// Llama 3 format:
    /// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// Gemma format:
    /// `<start_of_turn>role\ncontent<end_of_turn>`
    Gemma,
}

impl PromptTemplateType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::Llama3 => "llama3",
            Self::Gemma => "gemma",
        }
    }
}

/// Curated model entry with hardware requirements and download metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique identifier (e.g. "qwen2.5-1.5b-instruct")
    pub id: String,
    /// Human-readable display title
    pub display_name: String,
    /// Model family / creator brand ("Qwen", "Google Gemma", "Liquid AI", "OpenBMB", "Meta")
    pub family: String,
    /// Underlying transformer architecture
    pub architecture: ModelArch,
    /// Nominal parameter count string (e.g. "1.54B")
    pub param_count: String,
    /// Quantization format (e.g. "Q4_0", "Q4_K_M")
    pub quantization: String,
    /// Direct HTTPS download URL (Hugging Face resolve endpoint)
    pub download_url: String,
    /// Target file basename (e.g. "qwen2.5-1.5b-instruct-q4_0.gguf")
    pub filename: String,
    /// Exact file size in bytes
    pub file_size_bytes: u64,
    /// Expected SHA-256 hex digest for cryptographic integrity verification
    pub sha256: String,
    /// Native context length window
    pub default_context_length: usize,
    /// Estimated RAM footprint in bytes when loaded with 4096 context window
    pub estimated_ram_bytes_4k: u64,
    /// Default prompt formatting template
    pub prompt_template: PromptTemplateType,
    /// Brief description and recommended use case
    pub description: String,
}

impl ModelMetadata {
    /// Formatted file size string (e.g. "1.07 GB", "428 MB").
    pub fn formatted_size(&self) -> String {
        format_bytes(self.file_size_bytes)
    }

    /// Formatted RAM requirement string (e.g. "1.45 GiB").
    pub fn formatted_ram_requirement(&self) -> String {
        format!(
            "{:.2} GiB",
            self.estimated_ram_bytes_4k as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }

    /// Returns true if this model fits within the strict 5.20 GiB unified RAM budget.
    pub fn fits_in_ram_budget(&self) -> bool {
        self.estimated_ram_bytes_4k <= MAX_RAM_BUDGET_BYTES
    }
}

/// Formats a byte count into a human-readable string (MB or GB).
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    }
}

/// Returns the full list of curated models supported by Súlimo.
pub fn default_registry() -> Vec<ModelMetadata> {
    vec![
        ModelMetadata {
            id: "qwen2.5-1.5b-instruct".to_string(),
            display_name: "Qwen 2.5 1.5B Instruct".to_string(),
            family: "Qwen".to_string(),
            architecture: ModelArch::Qwen2,
            param_count: "1.54B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf".to_string(),
            filename: "qwen2.5-1.5b-instruct-q4_0.gguf".to_string(),
            file_size_bytes: 1_066_227_232,
            sha256: "dcd819ff094852c38faba6873d8ff0c9d51eadb2844539e52042ae5d647bbfdb".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 1_556_480_000,
            prompt_template: PromptTemplateType::ChatMl,
            description: "Top coding and reasoning workhorse in a compact ~1 GB footprint.".to_string(),
        },
        ModelMetadata {
            id: "qwen2.5-0.5b-instruct".to_string(),
            display_name: "Qwen 2.5 0.5B Instruct".to_string(),
            family: "Qwen".to_string(),
            architecture: ModelArch::Qwen2,
            param_count: "0.49B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf".to_string(),
            filename: "qwen2.5-0.5b-instruct-q4_0.gguf".to_string(),
            file_size_bytes: 428_730_208,
            sha256: "7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 697_932_800,
            prompt_template: PromptTemplateType::ChatMl,
            description: "Ultra-low resource edge model with blazing fast inference speed (>25 tok/s).".to_string(),
        },
        ModelMetadata {
            id: "gemma-2-2b-it".to_string(),
            display_name: "Gemma 2 2B Instruct".to_string(),
            family: "Google Gemma".to_string(),
            architecture: ModelArch::Gemma2,
            param_count: "2.61B".to_string(),
            quantization: "Q4_K_M".to_string(),
            download_url: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf".to_string(),
            filename: "gemma-2-2b-it-Q4_K_M.gguf".to_string(),
            file_size_bytes: 1_708_582_752,
            sha256: "e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787".to_string(),
            default_context_length: 8192,
            estimated_ram_bytes_4k: 2_415_919_104,
            prompt_template: PromptTemplateType::Gemma,
            description: "High-caliber conversational and factual quality, competitive with 7B models.".to_string(),
        },
        ModelMetadata {
            id: "lfm2.5-2.6b".to_string(),
            display_name: "Liquid LFM 2.5 2.6B".to_string(),
            family: "Liquid AI".to_string(),
            architecture: ModelArch::LFM,
            param_count: "2.60B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_0.gguf".to_string(),
            filename: "LFM2.5-2.6B-Q4_0.gguf".to_string(),
            file_size_bytes: 1_593_894_912,
            sha256: "e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 2_254_857_830,
            prompt_template: PromptTemplateType::ChatMl,
            description: "Hybrid Conv/RNN/Attention foundation model from Liquid AI.".to_string(),
        },
        ModelMetadata {
            id: "lfm2.5-230m".to_string(),
            display_name: "Liquid LFM 2.5 230M".to_string(),
            family: "Liquid AI".to_string(),
            architecture: ModelArch::LFM,
            param_count: "0.23B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-Q4_0.gguf".to_string(),
            filename: "LFM2.5-230M-Q4_0.gguf".to_string(),
            file_size_bytes: 149_080_928,
            sha256: "430fbec5b1b355e9bb12cd0638c9f2a8f21fedd6eafb4103e42c7e88887daa73".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 268_435_456,
            prompt_template: PromptTemplateType::ChatMl,
            description: "Ultra-compact sub-150 MB liquid model for battery-first edge tasks.".to_string(),
        },
        ModelMetadata {
            id: "minicpm5-1b".to_string(),
            display_name: "MiniCPM 5 1B".to_string(),
            family: "OpenBMB".to_string(),
            architecture: ModelArch::MiniCPM3,
            param_count: "1.10B".to_string(),
            quantization: "Q4_K_M".to_string(),
            download_url: "https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf".to_string(),
            filename: "MiniCPM5-1B-Q4_K_M.gguf".to_string(),
            file_size_bytes: 688_065_920,
            sha256: "81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 1_181_116_006,
            prompt_template: PromptTemplateType::ChatMl,
            description: "OpenBMB lightweight on-device bilingual powerhouse model.".to_string(),
        },
        ModelMetadata {
            id: "minicpm3-4b".to_string(),
            display_name: "MiniCPM 3 4B".to_string(),
            family: "OpenBMB".to_string(),
            architecture: ModelArch::MiniCPM3,
            param_count: "3.90B".to_string(),
            quantization: "Q4_K_M".to_string(),
            download_url: "https://huggingface.co/openbmb/MiniCPM3-4B-GGUF/resolve/main/minicpm3-4b-q4_k_m.gguf".to_string(),
            filename: "minicpm3-4b-q4_k_m.gguf".to_string(),
            file_size_bytes: 2_469_791_584,
            sha256: "64913247e927414ecf47fd3e9ea8e3f0c9acae293f583dfa7e24b8872e20fa4c".to_string(),
            default_context_length: 32768,
            estimated_ram_bytes_4k: 3_650_722_201,
            prompt_template: PromptTemplateType::ChatMl,
            description: "Near GPT-3.5 quality in a 2.4 GB package, within the 5.2 GB budget ceiling.".to_string(),
        },
        ModelMetadata {
            id: "llama-3.2-1b-instruct".to_string(),
            display_name: "Llama 3.2 1B Instruct".to_string(),
            family: "Meta".to_string(),
            architecture: ModelArch::Llama,
            param_count: "1.23B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf".to_string(),
            filename: "Llama-3.2-1B-Instruct-Q4_0.gguf".to_string(),
            file_size_bytes: 773_025_920,
            sha256: "fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8".to_string(),
            default_context_length: 131072,
            estimated_ram_bytes_4k: 1_288_490_188,
            prompt_template: PromptTemplateType::Llama3,
            description: "Meta reference lightweight instruct model, fast and general-purpose.".to_string(),
        },
    ]
}

/// Finds a model in the default registry by its unique ID.
pub fn find_model_by_id(id: &str) -> Option<ModelMetadata> {
    default_registry().into_iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_registry_entries_valid() {
        let registry = default_registry();
        assert!(
            !registry.is_empty(),
            "registry should contain curated models"
        );

        for model in &registry {
            assert!(!model.id.is_empty(), "model id should not be empty");
            assert!(!model.display_name.is_empty());
            assert!(!model.download_url.is_empty());
            assert!(
                model.download_url.starts_with("https://"),
                "URL should be https: {}",
                model.download_url
            );
            assert!(
                model.filename.ends_with(".gguf"),
                "filename must end with .gguf: {}",
                model.filename
            );
            assert_eq!(
                model.sha256.len(),
                64,
                "sha256 must be 64 hex characters for {}",
                model.id
            );
            assert!(
                model.file_size_bytes > 0,
                "file size must be positive for {}",
                model.id
            );

            // Verify strict budget compliance
            assert!(
                model.fits_in_ram_budget(),
                "Model {} exceeds 5.20 GiB RAM budget ({} bytes)",
                model.id,
                model.estimated_ram_bytes_4k
            );

            // Verify disk size is suitable for 30 GB SSD (< 3.0 GB)
            assert!(
                model.file_size_bytes < 3_000_000_000,
                "Model {} exceeds 3.0 GB disk size limit: {} bytes",
                model.id,
                model.file_size_bytes
            );
        }
    }

    #[test]
    fn test_find_model_by_id() {
        assert!(find_model_by_id("qwen2.5-1.5b-instruct").is_some());
        assert!(find_model_by_id("gemma-2-2b-it").is_some());
        assert!(find_model_by_id("lfm2.5-2.6b").is_some());
        assert!(find_model_by_id("minicpm3-4b").is_some());
        assert!(find_model_by_id("non-existent-model").is_none());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1_066_227_232), "1.07 GB");
        assert_eq!(format_bytes(428_730_208), "428.7 MB");
    }
}
