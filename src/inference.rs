//! Shared align pipeline: audio + text → emissions → CTC → timestamps.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::audio_io::load_wav_mono_f32;
use crate::backend::{DeviceRequest, ResolvedBackend};
use crate::config::{AlignOptions, ModelConfig, SAMPLE_RATE};
use crate::ctc::{forced_align, log_softmax_rows_inplace};
use crate::postprocess::{get_spans, merge_repeats, spans_to_items};
use crate::text::{load_vocab, split_text_placeholder, tokens_to_indices};

#[cfg(feature = "cpu")]
use crate::cpu_engine::CpuEngine;
#[cfg(feature = "cuda")]
use crate::cudarc_engine::CudaEngine;

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub device: DeviceRequest,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: DeviceRequest::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignRequest {
    pub audio_path: PathBuf,
    pub text_path: PathBuf,
    pub options: AlignOptions,
}

impl AlignRequest {
    pub fn from_paths(
        audio: impl Into<PathBuf>,
        text: impl Into<PathBuf>,
        language: impl Into<String>,
    ) -> Self {
        let mut options = AlignOptions::default();
        options.language = language.into();
        Self {
            audio_path: audio.into(),
            text_path: text.into(),
            options,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ForcedAlignItem {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
    /// Frame stride in milliseconds (Python `stride`).
    pub stride_ms: f32,
    pub backend: String,
}

enum Engine {
    #[cfg(feature = "cpu")]
    Cpu(CpuEngine),
    #[cfg(feature = "cuda")]
    Cuda(CudaEngine),
}

pub struct Aligner {
    engine: Engine,
    vocab: std::collections::HashMap<String, usize>,
    blank_id: usize,
    backend_tag: String,
}

pub fn load_model(model_dir: impl AsRef<Path>, opts: ModelOptions) -> Result<Aligner> {
    let model_dir = model_dir.as_ref();
    let config = ModelConfig::load(model_dir)?;
    let vocab = load_vocab(model_dir)?;
    let blank_id = vocab
        .get("<blank>")
        .copied()
        .or(config.pad_token_id)
        .unwrap_or(0);

    let resolved = opts.device.resolve()?;
    let (engine, backend_tag) = match resolved {
        ResolvedBackend::Cpu => {
            #[cfg(feature = "cpu")]
            {
                (
                    Engine::Cpu(CpuEngine::load(model_dir, config)?),
                    "cpu".to_string(),
                )
            }
            #[cfg(not(feature = "cpu"))]
            {
                bail!("CPU backend requested but not compiled (enable feature `cpu`)");
            }
        }
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(state) => (
            Engine::Cuda(CudaEngine::load(model_dir, config, state)?),
            "cuda".to_string(),
        ),
    };

    Ok(Aligner {
        engine,
        vocab,
        blank_id,
        backend_tag,
    })
}

impl Aligner {
    pub fn align(&self, req: AlignRequest) -> Result<ForcedAlignResult> {
        let waveform = load_wav_mono_f32(&req.audio_path)?;
        let raw_text = std::fs::read_to_string(&req.text_path)
            .with_context(|| format!("read text {}", req.text_path.display()))?;
        let raw_text = raw_text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        // M1: full normalize + uroman. Placeholder split for API wiring only.
        let split = if req.options.split_size == "auto" {
            if raw_text.chars().any(|c| c as u32 >= 0x4E00) {
                "char"
            } else {
                "word"
            }
        } else {
            req.options.split_size.as_str()
        };
        let text_pieces = split_text_placeholder(&raw_text, split);
        // Starred form not yet implemented — use pieces as both text & tokens.
        let tokens_starred = text_pieces.clone();
        let token_indices = tokens_to_indices(&tokens_starred, &self.vocab)?;

        let (mut logits, t, c) = match &self.engine {
            #[cfg(feature = "cpu")]
            Engine::Cpu(eng) => eng.forward_logits(&waveform)?,
            #[cfg(feature = "cuda")]
            Engine::Cuda(eng) => eng.forward_logits(&waveform)?,
        };

        if t == 0 || c == 0 || logits.len() != t * c {
            bail!("invalid logits shape T={t} C={c} len={}", logits.len());
        }

        log_softmax_rows_inplace(&mut logits, t, c);
        // Python appends a star column of zeros.
        let star_col = c;
        let mut emissions = Vec::with_capacity(t * (c + 1));
        for row in 0..t {
            emissions.extend_from_slice(&logits[row * c..(row + 1) * c]);
            emissions.push(0.0);
        }
        let c_em = c + 1;
        let _ = star_col;

        let (paths, _scores) =
            forced_align(&emissions, t, c_em, &token_indices, self.blank_id)?;

        let idx_to_token: std::collections::HashMap<usize, String> =
            self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let blank_label = idx_to_token
            .get(&self.blank_id)
            .cloned()
            .unwrap_or_else(|| "<blank>".into());

        let segments = merge_repeats(&paths, &idx_to_token);
        let spans = get_spans(&tokens_starred, &segments, &blank_label)?;
        let stride_ms =
            waveform.len() as f32 * 1000.0 / t as f32 / SAMPLE_RATE as f32;
        let items = spans_to_items(&text_pieces, &spans, stride_ms);

        Ok(ForcedAlignResult {
            items,
            stride_ms,
            backend: self.backend_tag.clone(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.backend_tag
    }
}
