//! Load model once, run N aligns — true RTFx (no process reload).
fn main() -> anyhow::Result<()> {
    env_logger::init();
    let model = std::env::args().nth(1).unwrap_or_else(|| {
        r"D:\ctc-forced-aligner\_model\models--MahmoudAshraf--mms-300m-1130-forced-aligner\snapshots\2d856eb340893e274480dfb15a7b2a94d7ab7f84".into()
    });
    let audio = std::env::args().nth(2).unwrap_or_else(|| r"D:\ctc-forced-aligner\tests\3m.wav".into());
    let text = std::env::args().nth(3).unwrap_or_else(|| r"D:\ctc-forced-aligner\tests\3m.txt".into());
    let n: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let opts = ctc_forced_aligner_rs::ModelOptions {
        device: ctc_forced_aligner_rs::DeviceRequest::Cpu,
    };
    let t0 = std::time::Instant::now();
    let aligner = ctc_forced_aligner_rs::load_model(&model, opts)?;
    println!("model_load_s={:.3}", t0.elapsed().as_secs_f64());

    let req = ctc_forced_aligner_rs::AlignRequest::from_paths(&audio, &text, "eng");
    // warmup
    let _ = aligner.align(req.clone())?;
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
    // duration from wav header via hound
    let reader = hound::WavReader::open(&audio)?;
    let dur = reader.duration() as f64 / reader.spec().sample_rate as f64;
    println!("audio_s={:.3} median_s={:.3} RTFx={:.2}x", dur, med, dur / med);
    Ok(())
}
