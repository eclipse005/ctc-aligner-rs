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
    /// Exact frame stride in milliseconds (`waveform_len / t / 16`).
    /// Python upstream ceils this (~21 ms for a nominal 20 ms model); we keep
    /// it exact — the ceil inflates all timestamps ~5% and makes a chunk's
    /// last word overshoot its window (cross-chunk cue overlap downstream).
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
    /// Cache last decoded waveform (same path → skip re-decode / ffmpeg).
    audio_cache: std::sync::Mutex<Option<(PathBuf, std::sync::Arc<Vec<f32>>)>>,
    /// Cache last text preprocess (same path + options → skip uroman).
    text_cache: std::sync::Mutex<
        Option<(
            PathBuf,
            String, // language|split|star|romanize key
            Vec<String>,
            Vec<String>,
            Vec<usize>,
        )>,
    >,
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
        audio_cache: std::sync::Mutex::new(None),
        text_cache: std::sync::Mutex::new(None),
    })
}

impl Aligner {
    pub fn align(&self, req: AlignRequest) -> Result<ForcedAlignResult> {
        let profile = std::env::var("CTC_PROFILE").ok().as_deref() == Some("1");
        let t_all = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let waveform = {
            let mut cache = self.audio_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((p, w)) = cache.as_ref() {
                if p == &req.audio_path {
                    std::sync::Arc::clone(w)
                } else {
                    let w = std::sync::Arc::new(load_wav_mono_f32(&req.audio_path)?);
                    *cache = Some((req.audio_path.clone(), std::sync::Arc::clone(&w)));
                    w
                }
            } else {
                let w = std::sync::Arc::new(load_wav_mono_f32(&req.audio_path)?);
                *cache = Some((req.audio_path.clone(), std::sync::Arc::clone(&w)));
                w
            }
        };
        let t_audio = t0.elapsed();

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

        // Overlap CPU text prep with GPU emissions (text only needed for CTC).
        let t0 = std::time::Instant::now();
        let text_key = format!(
            "{}|{}|{}|{}",
            req.options.language,
            split,
            req.options.star_frequency,
            req.options.romanize
        );
        let romanize = req.options.romanize;
        let language = req.options.language.clone();
        let star_frequency = req.options.star_frequency.clone();
        let split_owned = split.to_string();
        let opts = req.options.clone();
        let text_path = req.text_path.clone();
        let (text_out, em_out) = std::thread::scope(|scope| {
            let text_h = scope.spawn(|| {
                {
                    let cache = self.text_cache.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some((p, k, tok, txt, idx)) = cache.as_ref() {
                        if p == &text_path && k == &text_key {
                            return Ok::<_, anyhow::Error>((
                                tok.clone(),
                                txt.clone(),
                                idx.clone(),
                            ));
                        }
                    }
                }
                let (tokens_starred, text_starred) = preprocess_text(
                    &raw_text,
                    romanize,
                    &language,
                    &split_owned,
                    &star_frequency,
                )?;
                let token_indices = tokens_to_indices(&tokens_starred, &self.vocab)?;
                {
                    let mut cache = self.text_cache.lock().unwrap_or_else(|e| e.into_inner());
                    *cache = Some((
                        text_path.clone(),
                        text_key.clone(),
                        tokens_starred.clone(),
                        text_starred.clone(),
                        token_indices.clone(),
                    ));
                }
                Ok((tokens_starred, text_starred, token_indices))
            });
            let em_h = scope.spawn(|| self.generate_emissions(waveform.as_slice(), &opts));
            (text_h.join().expect("text thread"), em_h.join().expect("em thread"))
        });
        let (tokens_starred, text_starred, token_indices) = text_out?;
        let (emissions, t, c, stride_ms) = em_out?;
        let t_parallel = t0.elapsed();
        let t_text = t_parallel; // overlapped
        let t_em = t_parallel;

        let t0 = std::time::Instant::now();
        let (paths, scores) = forced_align(&emissions, t, c, &token_indices, self.blank_id)?;
        let t_ctc = t0.elapsed();

        let idx_to_token: std::collections::HashMap<usize, String> =
            self.vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let blank_label = idx_to_token
            .get(&self.blank_id)
            .cloned()
            .unwrap_or_else(|| "<blank>".into());

        let t0 = std::time::Instant::now();
        let segments = merge_repeats(&paths, &idx_to_token);
        let spans = get_spans(&tokens_starred, &segments, &blank_label)?;
        let items = spans_to_items(
            &text_starred,
            &spans,
            stride_ms,
            &scores,
            req.options.merge_threshold,
        );
        let t_post = t0.elapsed();

        if profile {
            eprintln!(
                "[CTC_PROFILE align] audio={:.3}s text={:.3}s emissions={:.3}s ctc={:.3}s post={:.3}s total={:.3}s T={t} L={} items={}",
                t_audio.as_secs_f64(),
                t_text.as_secs_f64(),
                t_em.as_secs_f64(),
                t_ctc.as_secs_f64(),
                t_post.as_secs_f64(),
                t_all.elapsed().as_secs_f64(),
                token_indices.len(),
                items.len()
            );
        }

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

        // Python appends a star column of zeros — single pass pack.
        let c_em = c + 1;
        let mut emissions = vec![0.0f32; t * c_em];
        for row in 0..t {
            let src = row * c;
            let dst = row * c_em;
            emissions[dst..dst + c].copy_from_slice(&logits[src..src + c]);
            // emissions[dst + c] already 0
        }

        // Python upstream does `math.ceil(stride)` (alignment_utils.py:165).
        // We deliberately do NOT: the true frame stride is ~20.013 ms for the
        // MMS conv stack, and ceiling it to 21 ms inflates every timestamp by
        // ~5% (a 30 s window's last word lands at ~31.5 s). Besides drifting
        // all times, the inflated tail of a chunked alignment overshoots the
        // chunk window and overlaps the next chunk's first word (observed as
        // overlapping SRT cues at every chunk seam in voxtrans). Keeping the
        // exact stride is both more accurate and provably bounded by the
        // waveform duration: max end = (t-1) * stride / 1000 < len/rate.
        let stride_ms = waveform.len() as f32 * 1000.0 / t as f32 / SAMPLE_RATE as f32;
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

        // Gather equal-length window slices and run batched CPU forward (fat FFN gemms).
        let slices: Vec<&[f32]> = (0..n_windows)
            .map(|w| {
                let start = w * window;
                &padded[start..start + chunk_len]
            })
            .collect();

        let batch_out = self.forward_logits_batch(&slices)?;
        let mut all = Vec::new();
        let mut c = 0usize;
        for (logits, t, vocab) in batch_out {
            c = vocab;
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

    fn forward_logits_batch(
        &self,
        waveforms: &[&[f32]],
    ) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        match &self.engine {
            #[cfg(feature = "cpu")]
            Engine::Cpu(eng) => eng.forward_logits_batch(waveforms),
            #[cfg(feature = "cuda")]
            Engine::Cuda(eng) => eng.forward_logits_batch(waveforms),
        }
    }

    pub fn backend(&self) -> &str {
        &self.backend_tag
    }
}

fn time_to_frame(time_sec: f32) -> usize {
    (time_sec * (1000.0 / TIME_TO_FRAME_MS)) as usize
}
