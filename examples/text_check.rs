fn main() -> anyhow::Result<()> {
    let text = std::fs::read_to_string(r"D:\ctc-forced-aligner\tests\en15s.txt")?;
    let text = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    let (ts, xs) = ctc_forced_aligner_rs::text::preprocess_text(&text, true, "eng", "word", "edges")?;
    println!("n tokens {}", ts.len());
    for (a,b) in ts.iter().zip(xs.iter()).take(12) {
        println!("{a:?} <- {b:?}");
    }
    let model = std::path::PathBuf::from(r"D:\ctc-forced-aligner\_model\models--MahmoudAshraf--mms-300m-1130-forced-aligner\snapshots\2d856eb340893e274480dfb15a7b2a94d7ab7f84");
    let vocab = ctc_forced_aligner_rs::text::load_vocab(&model)?;
    let ids = ctc_forced_aligner_rs::text::tokens_to_indices(&ts, &vocab)?;
    println!("n ids {}", ids.len());
    println!("ids[:30] {:?}", &ids[..30.min(ids.len())]);
    // save
    let s = ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    std::fs::write(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\rust_ids.txt", s)?;
    std::fs::write(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\rust_tokens.json", serde_json::to_string_pretty(&ts)?)?;
    Ok(())
}
