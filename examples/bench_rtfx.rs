//! Load model once, run N aligns — true RTFx (no process reload).
//!
//! Args: [model] [audio] [text] [n_runs] [device=cpu|cuda]
fn main() -> anyhow::Result<()> {
    env_logger::init();
    let model = std::env::args().nth(1).unwrap_or_else(|| {
        r"D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner".into()
    });
    let audio = std::env::args().nth(2).unwrap_or_else(|| r"D:\ctc-forced-aligner\tests\3m.wav".into());
    let text = std::env::args().nth(3).unwrap_or_else(|| r"D:\ctc-forced-aligner\tests\3m.txt".into());
    let n: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let device_s = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "cpu".into())
        .to_ascii_lowercase();

    let device = match device_s.as_str() {
        "cpu" => ctc_forced_aligner_rs::DeviceRequest::Cpu,
        "cuda" => ctc_forced_aligner_rs::DeviceRequest::Cuda(0),
        other if other.starts_with("cuda:") => {
            let n: usize = other[5..].parse()?;
            ctc_forced_aligner_rs::DeviceRequest::Cuda(n)
        }
        other => anyhow::bail!("unknown device {other:?} (cpu|cuda|cuda:N)"),
    };
    let opts = ctc_forced_aligner_rs::ModelOptions { device };
    let t0 = std::time::Instant::now();
    let aligner = ctc_forced_aligner_rs::load_model(&model, opts)?;
    println!("model_load_s={:.3}", t0.elapsed().as_secs_f64());

    let req = ctc_forced_aligner_rs::AlignRequest::from_paths(&audio, &text, "eng");
    // Warmup modes (CTC_BENCH_WARMUP):
    //   full (default) — one full align (heats GPU; matches heavy production)
    //   light — short 2s prefix align to prime CUDA without full thermal load
    //   0 / off — skip
    let warmup = std::env::var("CTC_BENCH_WARMUP").unwrap_or_else(|_| "full".into());
    match warmup.to_ascii_lowercase().as_str() {
        "0" | "off" | "none" | "false" => {}
        "light" => {
            // Prime kernels/cublas with a short real align if fixtures allow.
            if let Ok(short) = make_short_warmup_request(&audio, &text, 2.0) {
                let _ = aligner.align(short);
            } else {
                let _ = aligner.align(req.clone());
            }
        }
        _ => {
            let _ = aligner.align(req.clone())?;
        }
    }
    let mut times = Vec::new();
    for i in 0..n {
        let t = std::time::Instant::now();
        let r = aligner.align(req.clone())?;
        let s = t.elapsed().as_secs_f64();
        times.push(s);
        println!("run{} total_s={:.3} items={}", i + 1, s, r.items.len());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let best = times[0];
    // duration from wav header via hound
    let reader = hound::WavReader::open(&audio)?;
    let dur = reader.duration() as f64 / reader.spec().sample_rate as f64;
    println!(
        "audio_s={:.3} median_s={:.3} mean_s={:.3} best_s={:.3} RTFx_median={:.2}x RTFx_best={:.2}x",
        dur,
        med,
        mean,
        best,
        dur / med,
        dur / best
    );
    Ok(())
}

/// Build a short warmup request: first `secs` of audio + first line of text.
fn make_short_warmup_request(
    audio: &str,
    text: &str,
    secs: f32,
) -> anyhow::Result<ctc_forced_aligner_rs::AlignRequest> {
    use std::io::Write;
    let reader = hound::WavReader::open(audio)?;
    let spec = reader.spec();
    let n = ((secs * spec.sample_rate as f32) as usize).min(reader.duration() as usize);
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .take(n)
        .collect::<Result<_, _>>()?;
    let dir = std::env::temp_dir().join("ctc-aligner-bench");
    std::fs::create_dir_all(&dir)?;
    let wav_path = dir.join("warmup.wav");
    let txt_path = dir.join("warmup.txt");
    {
        let mut w = hound::WavWriter::create(
            &wav_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: spec.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for s in samples {
            w.write_sample(s)?;
        }
        w.finalize()?;
    }
    let head = std::fs::read_to_string(text)?
        .split_whitespace()
        .take(20)
        .collect::<Vec<_>>()
        .join(" ");
    let mut f = std::fs::File::create(&txt_path)?;
    writeln!(f, "{head}")?;
    Ok(ctc_forced_aligner_rs::AlignRequest::from_paths(
        wav_path, txt_path, "eng",
    ))
}
