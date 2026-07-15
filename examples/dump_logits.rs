fn main() -> anyhow::Result<()> {
    env_logger::init();
    let model = std::path::PathBuf::from(r"D:\ctc-forced-aligner\_model\models--MahmoudAshraf--mms-300m-1130-forced-aligner\snapshots\2d856eb340893e274480dfb15a7b2a94d7ab7f84");
    let cfg = ctc_forced_aligner_rs::config::ModelConfig::load(&model)?;
    let eng = ctc_forced_aligner_rs::cpu_engine::CpuEngine::load(&model, cfg)?;
    let wave = ctc_forced_aligner_rs::audio_io::load_wav_mono_f32(std::path::Path::new(r"D:\ctc-forced-aligner\tests\en15s.wav"))?;
    let (logits, t, c) = eng.forward_logits(&wave)?;
    println!("logits {}x{}", t, c);
    print!("row0: ");
    for j in 0..5.min(c) { print!("{} ", logits[j]); }
    println!();
    if t > 1 {
        print!("row1: ");
        for j in 0..5.min(c) { print!("{} ", logits[c+j]); }
        println!();
    }
    let bytes: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\rust_logits_en15s.bin", &bytes)?;
    std::fs::write(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\rust_logits_shape.txt", format!("{t} {c}"))?;
    println!("max {} min {}", logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max), logits.iter().cloned().fold(f32::INFINITY, f32::min));
    Ok(())
}
