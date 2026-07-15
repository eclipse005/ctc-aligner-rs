//! Audio loading — mono f32 at model sample rate.

use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader};

use crate::config::SAMPLE_RATE;

/// Load a WAV file as mono f32 samples, resampling to 16 kHz if needed.
///
/// Currently only exact 16 kHz mono/stereo PCM is accepted without resampling.
/// Non-16k inputs return an error (ffmpeg pre-convert recommended, same as
/// qwen-aligner-rs best practice). A proper resampler lands in M1.
pub fn load_wav_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "expected {} Hz audio, got {} Hz (pre-convert with ffmpeg -ar 16000 -ac 1)",
            SAMPLE_RATE,
            spec.sample_rate
        );
    }
    if spec.channels == 0 {
        bail!("wav has zero channels");
    }

    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(Into::into))
            .collect::<Result<Vec<_>>>()?,
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max = (1i64 << (bits.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?
        }
    };

    if channels == 1 {
        return Ok(samples);
    }

    // Average channels → mono.
    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut acc = 0f32;
        for ch in 0..channels {
            acc += samples[i * channels + ch];
        }
        mono.push(acc / channels as f32);
    }
    Ok(mono)
}
