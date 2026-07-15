//! Text normalization + tokenization stubs.
//!
//! Full Python parity (uroman, norm_config, star injection, word/char split)
//! lands in M1. For now we only provide light helpers used by the scaffold.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Load `vocab.json` as lowercase token → id map (matches Python get_alignments).
pub fn load_vocab(model_dir: &Path) -> Result<HashMap<String, usize>> {
    let path = model_dir.join("vocab.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read vocab {}", path.display()))?;
    let map: HashMap<String, usize> = serde_json::from_str(&raw)
        .with_context(|| format!("parse vocab {}", path.display()))?;
    let mut lower: HashMap<String, usize> = HashMap::with_capacity(map.len() + 1);
    for (k, v) in map {
        lower.insert(k.to_lowercase(), v);
    }
    let star_id = lower.len();
    lower.insert("<star>".into(), star_id);
    Ok(lower)
}

/// Map space-joined romanized tokens to vocab indices (Python get_alignments).
pub fn tokens_to_indices(tokens: &[String], vocab: &HashMap<String, usize>) -> Result<Vec<usize>> {
    let joined = tokens.join(" ");
    let mut out = Vec::new();
    for part in joined.split(' ') {
        if part.is_empty() {
            continue;
        }
        let key = part.to_lowercase();
        match vocab.get(&key) {
            Some(&id) => out.push(id),
            None => bail!("token {:?} not in vocab", part),
        }
    }
    if out.is_empty() {
        bail!("no tokens mapped into vocab");
    }
    Ok(out)
}

/// Minimal whitespace / CJK-ish split placeholder.
///
/// Real path must mirror Python `get_uroman_tokens` + `text_normalize`.
pub fn split_text_placeholder(text: &str, split_size: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    match split_size {
        "char" => text
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_string())
            .collect(),
        _ => text.split_whitespace().map(|s| s.to_string()).collect(),
    }
}
