//! Byte-level BPE tokenizer and chat template formatting (@C:TOK).
//!
//! Implements GPT-2 style byte-level Byte-Pair Encoding (BPE) from GGUF metadata
//! (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`), supporting Llama-3 and ChatML
//! prompt formatting without external template engines.

use crate::gguf::{GgufFile, GgufValue};
use std::collections::HashMap;
use thiserror::Error;

/// Errors returned during tokenization or decoding.
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("missing 'tokenizer.ggml.tokens' in GGUF metadata")]
    MissingTokens,
    #[error("missing 'tokenizer.ggml.merges' in GGUF metadata")]
    MissingMerges,
    #[error("unknown token ID {0}")]
    UnknownTokenId(u32),
    #[error("failed to decode UTF-8 bytes from tokens: {0}")]
    Utf8Decode(#[from] std::string::FromUtf8Error),
}

/// Byte-level BPE Tokenizer for transformer models.
pub struct Tokenizer {
    /// Token text to token ID mapping.
    pub token_to_id: HashMap<String, u32>,
    /// Token ID to token bytes mapping.
    pub id_to_token: Vec<String>,
    /// BPE merge pairs mapped to their priority rank (lower rank = higher priority).
    pub merge_ranks: HashMap<(String, String), usize>,
    /// Special token IDs.
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub eot_token_id: Option<u32>,
    /// Byte to BPE unicode character map (GPT-2 byte encoder).
    pub byte_to_char: [char; 256],
    /// Reverse mapping from BPE character to raw byte.
    pub char_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Builds a byte-level BPE tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, TokenizerError> {
        let tokens_val = gguf
            .get("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingTokens)?;
        let token_items = match tokens_val {
            GgufValue::Array(arr) => arr,
            _ => return Err(TokenizerError::MissingTokens),
        };

        let mut token_to_id = HashMap::with_capacity(token_items.len());
        let mut id_to_token = Vec::with_capacity(token_items.len());

        for (id, val) in token_items.iter().enumerate() {
            if let GgufValue::String(s) = val {
                let token_str = s.to_string();
                token_to_id.insert(token_str.clone(), id as u32);
                id_to_token.push(token_str);
            }
        }

        let mut merge_ranks = HashMap::new();
        if let Some(GgufValue::Array(merges)) = gguf.get("tokenizer.ggml.merges") {
            for (rank, val) in merges.iter().enumerate() {
                if let GgufValue::String(merge_str) = val {
                    if let Some((left, right)) = merge_str.split_once(' ') {
                        merge_ranks.insert((left.to_string(), right.to_string()), rank);
                    }
                }
            }
        }

        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id");
        let eot_token_id = token_to_id
            .get("<|eot_id|>")
            .or_else(|| token_to_id.get("<|im_end|>"))
            .copied()
            .or(eos_token_id);

        let (byte_to_char, char_to_byte) = build_gpt2_byte_encoder();

        Ok(Self {
            token_to_id,
            id_to_token,
            merge_ranks,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Pre-tokenizes and encodes text into token IDs.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                tokens.push(bos);
            }
        }

        if text.is_empty() {
            return tokens;
        }

        // Split text into pre-token chunks (words, punctuation, whitespace runs)
        let chunks = pre_tokenize_regex_split(text);

        for chunk in chunks {
            // Check if chunk is an exact special token (e.g. <|start_header_id|>)
            if let Some(&id) = self.token_to_id.get(chunk) {
                tokens.push(id);
                continue;
            }

            // Convert UTF-8 bytes to GPT-2 byte-level unicode characters
            let mut word_pieces: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();

            // Iteratively apply BPE merges
            while word_pieces.len() >= 2 {
                let mut best_pair = None;
                let mut min_rank = usize::MAX;

                for i in 0..word_pieces.len() - 1 {
                    let pair = (&word_pieces[i], &word_pieces[i + 1]);
                    if let Some(&rank) = self.merge_ranks.get(&(pair.0.clone(), pair.1.clone())) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some((i, rank));
                        }
                    }
                }

                match best_pair {
                    Some((idx, _)) => {
                        let merged = format!("{}{}", word_pieces[idx], word_pieces[idx + 1]);
                        word_pieces[idx] = merged;
                        word_pieces.remove(idx + 1);
                    }
                    None => break,
                }
            }

            // Map merged pieces to vocabulary token IDs
            for piece in word_pieces {
                if let Some(&id) = self.token_to_id.get(&piece) {
                    tokens.push(id);
                } else {
                    // Fallback to individual byte tokens
                    for ch in piece.chars() {
                        let ch_str = ch.to_string();
                        if let Some(&id) = self.token_to_id.get(&ch_str) {
                            tokens.push(id);
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut raw_bytes = Vec::new();

        for &tok_id in tokens {
            let tok_str = self
                .id_to_token
                .get(tok_id as usize)
                .ok_or(TokenizerError::UnknownTokenId(tok_id))?;

            // Check for byte-fallback token pattern: <0xXX>
            if tok_str.starts_with("<0x") && tok_str.ends_with('>') && tok_str.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&tok_str[3..5], 16) {
                    raw_bytes.push(byte_val);
                    continue;
                }
            }

            // Skip special marker tokens when decoding text stream
            if tok_str.starts_with("<|") && tok_str.ends_with("|>") {
                continue;
            }

            for ch in tok_str.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    raw_bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    raw_bytes.extend_from_slice(s.as_bytes());
                }
            }
        }

        Ok(String::from_utf8_lossy(&raw_bytes).to_string())
    }

    /// Formats a chat prompt using the model's standard chat template.
    pub fn format_chat_prompt(
        &self,
        arch_or_pre: &str,
        system_prompt: Option<&str>,
        user_message: &str,
    ) -> String {
        let is_chatml = arch_or_pre.contains("qwen") || arch_or_pre.contains("chatml");

        if is_chatml {
            // ChatML template
            let mut prompt = String::new();
            if let Some(sys) = system_prompt {
                prompt.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
            }
            prompt.push_str(&format!(
                "<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            ));
            prompt
        } else {
            // Llama 3 template
            let mut prompt = String::from("<|begin_of_text|>");
            if let Some(sys) = system_prompt {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
                ));
            }
            prompt.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ));
            prompt
        }
    }
}

/// Splits text into pre-tokenization chunks matching the standard GPT-2 / Llama-3 regex pattern.
pub fn pre_tokenize_regex_split(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Special tokens: check if starting with <|
        if bytes[i..].starts_with(b"<|") {
            if let Some(end) = text[i..].find("|>") {
                let end_idx = i + end + 2;
                chunks.push(&text[i..end_idx]);
                i = end_idx;
                continue;
            }
        }

        let ch = text[i..].chars().next().unwrap();
        let ch_len = ch.len_utf8();

        // 1. Whitespace sequence: \s*[\r\n]+ or whitespace runs
        if ch == '\r' || ch == '\n' {
            let mut j = i + ch_len;
            while j < bytes.len() {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch == '\r' || next_ch == '\n' || next_ch == ' ' || next_ch == '\t' {
                    j += next_ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..j]);
            i = j;
            continue;
        }

        // 2. Spaces run
        if ch == ' ' || ch == '\t' {
            let mut j = i + ch_len;
            while j < bytes.len() {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch == ' ' || next_ch == '\t' {
                    j += next_ch.len_utf8();
                } else {
                    break;
                }
            }
            // Check if next character is a letter or number (e.g. " hello" vs " ")
            if j < bytes.len() {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch.is_alphabetic() || next_ch.is_numeric() {
                    // Include the leading space with the word per GPT-2 convention
                    let word_start = i;
                    let mut k = j;
                    while k < bytes.len() {
                        let word_ch = text[k..].chars().next().unwrap();
                        if word_ch.is_alphabetic() || word_ch.is_numeric() {
                            k += word_ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    chunks.push(&text[word_start..k]);
                    i = k;
                    continue;
                }
            }
            chunks.push(&text[i..j]);
            i = j;
            continue;
        }

        // 3. Contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if ch == '\'' {
            let rem = &text[i..];
            let lower = rem.to_ascii_lowercase();
            let mut matched = false;
            for &c in &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
                if lower.starts_with(c) {
                    chunks.push(&text[i..i + c.len()]);
                    i += c.len();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
        }

        // 4. Letters (alphabetic run)
        if ch.is_alphabetic() {
            let mut j = i + ch_len;
            while j < bytes.len() {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch.is_alphabetic() {
                    j += next_ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..j]);
            i = j;
            continue;
        }

        // 5. Numbers: runs of 1 to 3 digits
        if ch.is_numeric() {
            let mut count = 0;
            let mut j = i;
            while j < bytes.len() && count < 3 {
                let next_ch = text[j..].chars().next().unwrap();
                if next_ch.is_numeric() {
                    j += next_ch.len_utf8();
                    count += 1;
                } else {
                    break;
                }
            }
            chunks.push(&text[i..j]);
            i = j;
            continue;
        }

        // 6. Punctuation / symbols / other unicode codepoints
        let mut j = i + ch_len;
        while j < bytes.len() {
            let next_ch = text[j..].chars().next().unwrap();
            if !next_ch.is_alphabetic()
                && !next_ch.is_numeric()
                && !next_ch.is_whitespace()
                && !text[j..].starts_with("<|")
            {
                j += next_ch.len_utf8();
            } else {
                break;
            }
        }
        chunks.push(&text[i..j]);
        i = j;
    }

    chunks
}

/// Builds the bijective GPT-2 byte-to-character encoder table.
pub fn build_gpt2_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);

    let mut n = 0u32;
    for b in 0u8..=255 {
        // Printable ASCII and select Latin-1 characters map to themselves
        let ch =
            if (b'!'..=b'~').contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                b as char
            } else {
                // Other bytes map to code points starting at 256 (0x100)
                let c = char::from_u32(256 + n).unwrap();
                n += 1;
                c
            };
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b);
    }

    (byte_to_char, char_to_byte)
}

/// In-place SIMD Fast Sampler operating directly on static arena logits buffer (@C:TOK).
pub struct SamplerChain {
    pub temperature: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.1,
        }
    }
}

impl SamplerChain {
    pub fn sample_in_place(&self, logits: &mut [f32], history: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // 1. Repetition penalty in-place
        if self.repetition_penalty != 1.0 {
            for &tok in history {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.repetition_penalty;
                    } else {
                        logits[idx] *= self.repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature scaling
        if self.temperature > 0.0 && (self.temperature - 1.0).abs() > 1e-4 {
            let inv_t = 1.0 / self.temperature;
            for val in logits.iter_mut() {
                *val *= inv_t;
            }
        }

        // 3. Max logit for numerical stability
        let mut max_l = f32::NEG_INFINITY;
        for &val in logits.iter() {
            if val > max_l {
                max_l = val;
            }
        }

        // 4. Softmax exponentiation
        let mut sum_exp = 0.0f32;
        for val in logits.iter_mut() {
            *val = (*val - max_l).exp();
            sum_exp += *val;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-9);
        for val in logits.iter_mut() {
            *val *= inv_sum;
        }

        // 5. Min-P filtering
        let max_prob = logits.iter().cloned().fold(0.0f32, f32::max);
        let min_p_threshold = max_prob * self.min_p;
        for val in logits.iter_mut() {
            if *val < min_p_threshold {
                *val = 0.0;
            }
        }

        // 6. Greedy pick from surviving candidates
        let mut best_id = 0;
        let mut best_p = 0.0;
        for (i, &p) in logits.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_id = i;
            }
        }
        best_id as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_byte_encoder_bijective() {
        let (b2c, c2b) = build_gpt2_byte_encoder();
        assert_eq!(c2b.len(), 256);
        for b in 0u8..=255 {
            let ch = b2c[b as usize];
            assert_eq!(c2b.get(&ch), Some(&b));
        }
    }

    #[test]
    fn test_pre_tokenize_split() {
        let text = "Hello world! 12345 're\n\n<|end|>";
        let chunks = pre_tokenize_regex_split(text);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0], "Hello");
        assert!(chunks.contains(&" world"));
        assert!(chunks.contains(&"!"));
        assert!(chunks.contains(&"'re"));
        assert!(chunks.contains(&"<|end|>"));
    }

    #[test]
    fn test_chat_template_formatting() {
        let llama_prompt = Tokenizer::format_chat_prompt(
            &Tokenizer {
                token_to_id: HashMap::new(),
                id_to_token: Vec::new(),
                merge_ranks: HashMap::new(),
                bos_token_id: None,
                eos_token_id: None,
                eot_token_id: None,
                byte_to_char: ['\0'; 256],
                char_to_byte: HashMap::new(),
            },
            "llama-bpe",
            Some("You are a helpful assistant."),
            "What is Arda's Breath?",
        );

        assert!(llama_prompt.contains("<|begin_of_text|>"));
        assert!(llama_prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(llama_prompt.contains("You are a helpful assistant."));
        assert!(llama_prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(llama_prompt.contains("What is Arda's Breath?"));
        assert!(llama_prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }
}
