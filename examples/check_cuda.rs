//! Smoke-test CUDA init + weight upload (no full forward yet).
//!
//! ```
//! cargo run --release --features cuda --example check_cuda -- path/to/model
//! ```

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let model = std::env::args().nth(1).unwrap_or_else(|| {
        r"D:\ctc-forced-aligner\_model\models--MahmoudAshraf--mms-300m-1130-forced-aligner\snapshots\2d856eb340893e274480dfb15a7b2a94d7ab7f84".into()
    });

    let opts = ctc_forced_aligner_rs::ModelOptions {
        device: ctc_forced_aligner_rs::DeviceRequest::Cuda(0),
    };
    let t0 = std::time::Instant::now();
    let aligner = ctc_forced_aligner_rs::load_model(&model, opts)?;
    println!(
        "CUDA model load ok in {:.2}s backend={}",
        t0.elapsed().as_secs_f64(),
        aligner.backend()
    );
    Ok(())
}
