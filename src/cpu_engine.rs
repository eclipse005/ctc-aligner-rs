//! CPU Wav2Vec2ForCTC engine (gemm + rayon).
//!
//! M1: implement feature extractor + encoder + CTC head in f32.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};

use crate::config::ModelConfig;
use crate::raw_tensor::RawTensor;

/// Loaded CPU weights + config (forward not yet implemented).
pub struct CpuEngine {
    pub config: ModelConfig,
    /// Weight name → host f32 (or raw until materialised).
    #[allow(dead_code)]
    pub weights: HashMap<String, RawTensor>,
}

impl CpuEngine {
    pub fn load(model_dir: &Path, config: ModelConfig) -> Result<Self> {
        let weights = crate::weights::load_weights(model_dir)?;
        log::info!(
            "CPU engine: loaded {} tensors from {}",
            weights.len(),
            model_dir.display()
        );
        Ok(Self { config, weights })
    }

    /// Forward waveform → CTC logits `(T, vocab)`.
    ///
    /// Not implemented yet (M1).
    pub fn forward_logits(&self, _waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        bail!(
            "CPU Wav2Vec2 forward not implemented yet (M1). \
             config: layers={}, hidden={}",
            self.config.num_hidden_layers,
            self.config.hidden_size
        );
    }
}
