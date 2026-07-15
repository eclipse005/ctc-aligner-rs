use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use ctc_forced_aligner_rs::{
    load_model, write_forced_align_items_json, AlignRequest, DeviceRequest, ModelOptions,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "CTC forced aligner (hand-written CUDA / CPU). Golden: original Python."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Align one audio file with one text file.
    Align {
        #[arg(long, value_name = "WAV")]
        audio: PathBuf,
        #[arg(long, value_name = "TXT")]
        text: PathBuf,
        #[arg(long, value_name = "DIR", help = "Model directory (HF layout).")]
        model: PathBuf,
        #[arg(long, default_value = "eng", help = "Language ISO code.")]
        language: String,
        #[arg(long, value_name = "JSON")]
        output: PathBuf,
        #[arg(
            long,
            default_value = "auto",
            help = "Backend: auto | cuda | cuda:<n> | cpu"
        )]
        device: String,
    },
}

fn parse_device(s: &str) -> Result<DeviceRequest> {
    let s = s.trim().to_ascii_lowercase();
    if s == "auto" {
        return Ok(DeviceRequest::Auto);
    }
    if s == "cpu" {
        return Ok(DeviceRequest::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if s == "cuda" {
            return Ok(DeviceRequest::Cuda(0));
        }
        if let Some(rest) = s.strip_prefix("cuda:") {
            let n: usize = rest
                .parse()
                .map_err(|e| anyhow::anyhow!("invalid CUDA ordinal {rest:?}: {e}"))?;
            return Ok(DeviceRequest::Cuda(n));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if s.starts_with("cuda") {
            anyhow::bail!("binary built without `cuda` feature");
        }
    }
    anyhow::bail!("unknown --device {s:?} (expected: auto | cuda | cuda:<n> | cpu)")
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Command::Align {
            audio,
            text,
            model,
            language,
            output,
            device,
        } => {
            let opts = ModelOptions {
                device: parse_device(&device)?,
            };
            let aligner = load_model(&model, opts)?;
            let result = aligner.align(AlignRequest::from_paths(audio, text, language))?;
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            write_forced_align_items_json(&output, &result.items)?;
            println!(
                "align completed: items={}, backend={}, stride_ms={:.2}, output={}",
                result.items.len(),
                result.backend,
                result.stride_ms,
                output.display()
            );
            println!(
                "(note: Wav2Vec2 forward is not implemented yet — this path errors before write if engine is stub)"
            );
        }
    }
    Ok(())
}
