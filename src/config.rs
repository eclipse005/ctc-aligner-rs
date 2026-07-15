//! Model + align runtime configuration.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Hugging Face `config.json` subset for Wav2Vec2ForCTC / MMS.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_conv_dim")]
    pub conv_dim: Vec<usize>,
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: Vec<usize>,
    #[serde(default = "default_conv_stride")]
    pub conv_stride: Vec<usize>,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_true")]
    pub do_stable_layer_norm: bool,
    #[serde(default = "default_feat_extract_norm")]
    pub feat_extract_norm: String,
    #[serde(default = "default_num_conv_pos_embeddings")]
    pub num_conv_pos_embeddings: usize,
    #[serde(default = "default_num_conv_pos_embedding_groups")]
    pub num_conv_pos_embedding_groups: usize,
    /// CTC blank is usually pad_token_id / vocab `<blank>`.
    #[serde(default)]
    pub pad_token_id: Option<usize>,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_vocab_size() -> usize {
    37
}
fn default_conv_dim() -> Vec<usize> {
    vec![512; 7]
}
fn default_conv_kernel() -> Vec<usize> {
    vec![10, 3, 3, 3, 3, 2, 2]
}
fn default_conv_stride() -> Vec<usize> {
    vec![5, 2, 2, 2, 2, 2, 2]
}
fn default_true() -> bool {
    true
}
fn default_feat_extract_norm() -> String {
    "layer".into()
}
fn default_num_conv_pos_embeddings() -> usize {
    128
}
fn default_num_conv_pos_embedding_groups() -> usize {
    16
}

impl ModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read model config {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&raw)
            .with_context(|| format!("parse model config {}", path.display()))?;
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Align-time knobs matching Python CLI defaults.
#[derive(Debug, Clone)]
pub struct AlignOptions {
    /// Sliding window length in seconds (Python `--window_size`, default 30).
    pub window_size_sec: f32,
    /// Context on each side of a window in seconds (Python `--context_size`, default 2).
    pub context_size_sec: f32,
    pub batch_size: usize,
    /// Language ISO code for text normalization (e.g. `eng`, `cmn`, `jpn`).
    pub language: String,
    /// `word` | `char` | `auto`.
    pub split_size: String,
    /// `segment` | `once` | `none` — star token injection frequency.
    pub star_frequency: String,
    pub merge_threshold: f32,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            window_size_sec: 30.0,
            context_size_sec: 2.0,
            batch_size: 4,
            language: "eng".into(),
            split_size: "auto".into(),
            star_frequency: "segment".into(),
            merge_threshold: 0.12,
        }
    }
}

/// Native model sample rate (Wav2Vec2 / MMS).
pub const SAMPLE_RATE: u32 = 16_000;
