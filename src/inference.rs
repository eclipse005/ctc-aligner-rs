//! Shared align pipeline: audio + text → emissions → CTC → timestamps.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::audio_io::load_wav_mono_f32;
use crate::backend::{DeviceRequest, ResolvedBackend};
use crate::config::{AlignOptions, ModelConfig, SAMPLE_RATE};
use crate::ctc::{forced_align, log_softmax_rows_inplace};
use crate::postprocess::{get_spans, merge_repeats, spans_to_items};
use crate::text::{load_vocab, preprocess_text, tokens_to_indices};

#[cfg(feature = "cpu")]
use crate::cpu_engine::CpuEngine;
#[cfg(feature = "cuda")]
use crate::cudarc_engine::CudaEngine;

/// Python `time_to_frame` uses a fixed 20 ms stride for context cropping.
const TIME_TO_FRAME_MS: f32 = 20.0;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
    /// Frame stride in milliseconds (Python `math.ceil(stride)`).
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

        let split = if req.options.split_size == "auto" {
            if req.options.language == "jpn" || req.options.language == "chi" {
                "char"
            } else {
                "word"
            }
        } else {
            req.options.split_size.as_str()
        };

        let (tokens_starred, text_starred) = preprocess_text(
            &raw_text,
            req.options.romanize,
            &req.options.language,
            split,
            &req.options.star_frequency,
        )?;
        let token_indices = tokens_to_indices(&tokens_starred, &self.vocab)?;

        let (emissions, t, c, stride_ms) = self.generate_emissions(&waveform, &req.options)?;

        let (paths, scores) = forced_align(&emissions, t, c, &token_indices, self.blank_id)?;

        let idx_to_token: std::collections::HashMap<usize, String> =
            self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let blank_label = idx_to_token
            .get(&self.blank_id)
            .cloned()
            .unwrap_or_else(|| "<blank>".into());

        let segments = merge_repeats(&paths, &idx_to_token);
        let spans = get_spans(&tokens_starred, &segments, &blank_label)?;
        let items = spans_to_items(
            &text_starred,
            &spans,
            stride_ms,
            &scores,
            req.options.merge_threshold,
        );

        Ok(ForcedAlignResult {
            items,
            stride_ms,
            backend: self.backend_tag.clone(),
        })
    }

    /// Python `generate_emissions`: windowed logits → log_softmax → star column.
    fn generate_emissions(
        &self,
        waveform: &[f32],
        opts: &AlignOptions,
    ) -> Result<(Vec<f32>, usize, usize, f32)> {
        let (mut logits, t, c) = self.generate_emissions_logits(waveform, opts)?;
        log_softmax_rows_inplace(&mut logits, t, c);

        // Python appends a star column of zeros.
        let mut emissions = Vec::with_capacity(t * (c + 1));
        for row in 0..t {
            emissions.extend_from_slice(&logits[row * c..(row + 1) * c]);
            emissions.push(0.0);
        }
        let c_em = c + 1;

        let stride = waveform.len() as f32 * 1000.0 / t as f32 / SAMPLE_RATE as f32;
        let stride_ms = stride.ceil();
        Ok((emissions, t, c_em, stride_ms))
    }

    fn generate_emissions_logits(
        &self,
        waveform: &[f32],
        opts: &AlignOptions,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let window = (opts.window_size_sec * SAMPLE_RATE as f32) as usize;
        let context = (opts.context_size_sec * SAMPLE_RATE as f32) as usize;
        let ctx_frames = time_to_frame(opts.context_size_sec);

        if waveform.len() < window {
            return self.forward_logits(waveform);
        }

        let n = waveform.len();
        let n_windows = (n + window - 1) / window;
        let extension = n_windows * window - n;
        let mut padded = vec![0.0f32; context + n + context + extension];
        padded[context..context + n].copy_from_slice(waveform);
        let chunk_len = window + 2 * context;

        let mut all = Vec::new();
        let mut c = 0usize;
        for w in 0..n_windows {
            let start = w * window;
            let slice = &padded[start..start + chunk_len];
            let (logits, t, vocab) = self.forward_logits(slice)?;
            c = vocab;
            // Python: emissions[:, ctx : -ctx+1]
            let start_f = ctx_frames.min(t);
            let end_f = if t > ctx_frames {
                t - ctx_frames + 1
            } else {
                t
            };
            let end_f = end_f.min(t);
            for i in start_f..end_f {
                all.extend_from_slice(&logits[i * c..(i + 1) * c]);
            }
        }

        let ext_frames = time_to_frame(extension as f32 / SAMPLE_RATE as f32);
        if c == 0 {
            bail!("no frames produced");
        }
        let total = all.len() / c;
        let keep = total.saturating_sub(ext_frames);
        all.truncate(keep * c);
        Ok((all, keep, c))
    }

    fn forward_logits(&self, waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        match &self.engine {
            #[cfg(feature = "cpu")]
            Engine::Cpu(eng) => eng.forward_logits(waveform),
            #[cfg(feature = "cuda")]
            Engine::Cuda(eng) => eng.forward_logits(waveform),
        }
    }

    pub fn backend(&self) -> &str {
        &self.backend_tag
    }
}

fn time_to_frame(time_sec: f32) -> usize {
    (time_sec * (1000.0 / TIME_TO_FRAME_MS)) as usize
}
