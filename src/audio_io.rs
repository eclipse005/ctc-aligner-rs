//! Audio loading — always normalize to 16 kHz mono s16le via ffmpeg, then decode.
//!
//! Design (match production practice + avoid silent sample-rate bugs):
//! - **All** inputs go through ffmpeg → 16 kHz, mono, pcm_s16le.
//! - Decode with a single scale (`i16 as f32 / 32768.0`), same for every file.
//! - Finds ffmpeg on PATH, `FFMPEG` env, or common relative locations.
//!
//! This is intentionally *not* a pure-Rust resampler: ffmpeg is the portable
//! standard and matches how operators pre-convert media in the wild.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader};

use crate::config::SAMPLE_RATE;

/// Load any ffmpeg-readable media as mono f32 @ 16 kHz for the aligner.
pub fn load_audio_mono_16k(path: &Path) -> Result<Vec<f32>> {
    if !path.exists() {
        bail!("audio not found: {}", path.display());
    }
    // Fast path: already 16 kHz mono WAV — skip ffmpeg round-trip.
    if let Ok(samples) = try_load_native_16k_mono_wav(path) {
        return Ok(samples);
    }

    let ffmpeg = find_ffmpeg()?;
    let tmp = tempfile_wav_path(path)?;

    // -ar 16000 -ac 1 -c:a pcm_s16le : deterministic model-native format
    // -y overwrite; -hide_banner -loglevel error keep logs clean
    let status = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
        ])
        .arg(path)
        .args([
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
        ])
        .arg(&tmp)
        .status()
        .with_context(|| format!("spawn ffmpeg ({})", ffmpeg.display()))?;

    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!(
            "ffmpeg failed converting {} → 16 kHz mono (status {status})",
            path.display()
        );
    }

    let result = load_wav_s16le_mono_f32(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result.with_context(|| {
        format!(
            "decode ffmpeg output for {} (temp {})",
            path.display(),
            tmp.display()
        )
    })
}

/// If `path` is already 16 kHz mono WAV, decode in-process (no ffmpeg).
fn try_load_native_16k_mono_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        bail!("not native 16k mono");
    }
    match spec.sample_format {
        SampleFormat::Int if spec.bits_per_sample == 16 => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0).map_err(Into::into))
            .collect(),
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.map_err(Into::into)).collect(),
        _ => bail!("unsupported native wav format"),
    }
}

/// Back-compat name used by inference.
pub fn load_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    load_audio_mono_16k(path)
}

/// Decode a 16 kHz mono s16le WAV → f32 in approximately [-1, 1].
/// Scale matches common torchaudio int16 path: `sample / 32768.0`.
fn load_wav_s16le_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "internal: expected {} Hz after ffmpeg, got {}",
            SAMPLE_RATE,
            spec.sample_rate
        );
    }
    if spec.channels != 1 {
        bail!(
            "internal: expected mono after ffmpeg, got {} ch",
            spec.channels
        );
    }

    match spec.sample_format {
        SampleFormat::Int if spec.bits_per_sample == 16 => {
            // Prefer exact i16 path (ffmpeg pcm_s16le).
            let samples: Result<Vec<f32>> = reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0).map_err(Into::into))
                .collect();
            samples
        }
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max = (1i64 << (bits.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).map_err(Into::into))
                .collect()
        }
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(Into::into))
            .collect(),
    }
}

fn find_ffmpeg() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FFMPEG") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
    }
    // Next to common monorepo layout: <repo>/ffmpeg/bin/ffmpeg.exe
    let candidates = [
        PathBuf::from(r"D:\ctc-forced-aligner\ffmpeg\bin\ffmpeg.exe"),
        PathBuf::from("ffmpeg/bin/ffmpeg.exe"),
        PathBuf::from("ffmpeg.exe"),
        PathBuf::from("ffmpeg"),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    // PATH
    if let Ok(out) = Command::new("where").arg("ffmpeg").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().next() {
                let p = PathBuf::from(line.trim());
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    if let Ok(out) = Command::new("which").arg("ffmpeg").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let p = PathBuf::from(s.trim());
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    bail!(
        "ffmpeg not found. Set FFMPEG=path/to/ffmpeg or install ffmpeg on PATH. \
         Required to normalize audio to 16 kHz mono."
    )
}

fn tempfile_wav_path(src: &Path) -> Result<PathBuf> {
    let mut dir = std::env::temp_dir();
    dir.push("ctc-aligner-rs");
    std::fs::create_dir_all(&dir).ok();
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    // unique-ish name
    let name = format!(
        "{stem}_{}_{}.wav",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    );
    dir.push(name);
    Ok(dir)
}
