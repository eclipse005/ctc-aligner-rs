//! CUDA Wav2Vec2ForCTC engine (cudarc + hand-written kernels + cuBLAS).
//!
//! M2: full GPU path. For now only device probe / CudaState scaffolding.

use std::path::Path;

use anyhow::{bail, Result};

use crate::config::ModelConfig;

/// Minimal CUDA device handle used by `DeviceRequest::resolve`.
pub struct CudaState {
    pub ordinal: usize,
}

impl CudaState {
    pub fn new(ordinal: usize) -> Result<Self> {
        // Probe that cudarc can open a context. Full engine init lands with M2.
        let _ctx = cudarc::driver::CudaContext::new(ordinal).map_err(|e| {
            anyhow::anyhow!("cudarc CudaContext::new({ordinal}) failed: {e}")
        })?;
        log::info!("CUDA device {ordinal} available");
        Ok(Self { ordinal })
    }
}

pub struct CudaEngine {
    pub config: ModelConfig,
    #[allow(dead_code)]
    pub state: std::sync::Arc<CudaState>,
}

impl CudaEngine {
    pub fn load(
        model_dir: &Path,
        config: ModelConfig,
        state: std::sync::Arc<CudaState>,
    ) -> Result<Self> {
        // Ensure weights file exists early; full upload is M2.
        let st = model_dir.join("model.safetensors");
        let index = model_dir.join("model.safetensors.index.json");
        if !st.exists() && !index.exists() {
            bail!(
                "CUDA engine: no model.safetensors under {}",
                model_dir.display()
            );
        }
        log::info!(
            "CUDA engine stub ready (device {}); forward lands in M2",
            state.ordinal
        );
        Ok(Self { config, state })
    }

    pub fn forward_logits(&self, _waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        bail!(
            "CUDA Wav2Vec2 forward not implemented yet (M2). device={}",
            self.state.ordinal
        );
    }
}
