//! Text normalization + tokenization matching Python `text_utils.py`.
//!
//! Golden: `ctc_forced_aligner.text_utils.preprocess_text` with romanize=True
//! (MMS character vocab path).

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use uroman::{rom_format, Uroman};

thread_local! {
    static UROMAN: Uroman = Uroman::new();
}

/// Load `vocab.json` as lowercase token → id, then append `<star>`.
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

/// Python `text_normalize` for the default / English config subset.
pub fn text_normalize(text: &str, _iso_code: &str) -> String {
    // unicode NFKC-ish: Python uses config unicode_norm (often NFKC)
    let mut text: String = text.nfkc().collect();
    text = text.to_lowercase();

    // always strip brackets containing digits, e.g. "(Sam 23:17)"
    static RE_BRACKETS_NUM: OnceLock<Regex> = OnceLock::new();
    let re_bn = RE_BRACKETS_NUM.get_or_init(|| Regex::new(r"\([^\)]*\d[^\)]*\)").unwrap());
    text = re_bn.replace_all(&text, " ").into_owned();

    // basic punctuation → space (period ? , : ! { } ")
    static RE_PUNC: OnceLock<Regex> = OnceLock::new();
    let re_punc = RE_PUNC.get_or_init(|| Regex::new(r#"[.?!,;:{}"]+"#).unwrap());
    text = re_punc.replace_all(&text, " ").into_owned();

    // remove pure digit words
    static RE_DIGITS: OnceLock<Regex> = OnceLock::new();
    let re_digits = RE_DIGITS.get_or_init(|| {
        Regex::new(r"(^|\s)\d+(\s|$)").unwrap()
    });
    // apply repeatedly for overlapping
    for _ in 0..8 {
        let next = re_digits.replace_all(&text, " ").into_owned();
        if next == text {
            break;
        }
        text = next;
    }

    // collapse spaces
    static RE_SPACE: OnceLock<Regex> = OnceLock::new();
    let re_space = RE_SPACE.get_or_init(|| Regex::new(r"\s+").unwrap());
    re_space.replace_all(text.trim(), " ").into_owned()
}

fn normalize_uroman(text: &str) -> String {
    let text = text.to_lowercase();
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"([^a-z' ])").unwrap());
    let text = re.replace_all(&text, " ");
    static RE_SPACE: OnceLock<Regex> = OnceLock::new();
    let re_space = RE_SPACE.get_or_init(|| Regex::new(r" +").unwrap());
    re_space.replace_all(text.trim(), " ").into_owned()
}

/// Python `get_uroman_tokens`.
pub fn get_uroman_tokens(norm_transcripts: &[String], iso: Option<&str>) -> Vec<String> {
    UROMAN.with(|u| {
        let mut out = Vec::with_capacity(norm_transcripts.len());
        for t in norm_transcripts {
            let romanized: String = u
                .romanize_string::<rom_format::Str>(t, iso)
                .to_output_string();
            // Python: ot = " ".join(ot.strip())  → space-separate characters
            let chars_spaced: String = romanized
                .trim()
                .chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            static RE_SPACE: OnceLock<Regex> = OnceLock::new();
            let re_space = RE_SPACE.get_or_init(|| Regex::new(r"\s+").unwrap());
            let ot = re_space.replace_all(chars_spaced.trim(), " ");
            out.push(normalize_uroman(&ot));
        }
        out
    })
}

/// Python `preprocess_text`.
///
/// Returns `(tokens_starred, text_starred)`.
pub fn preprocess_text(
    text: &str,
    romanize: bool,
    language: &str,
    mut split_size: &str,
    star_frequency: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    if language == "jpn" || language == "chi" {
        split_size = "char";
    }
    let text_split = split_text(text, split_size);
    let norm_text: Vec<String> = text_split
        .iter()
        .map(|line| text_normalize(line.trim(), language))
        .collect();

    let tokens: Vec<String> = if romanize {
        get_uroman_tokens(&norm_text, Some(language))
    } else {
        norm_text
            .iter()
            .map(|w| w.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" "))
            .collect()
    };

    let (tokens_starred, text_starred) = match star_frequency {
        "segment" => {
            let mut ts = Vec::new();
            let mut xs = Vec::new();
            for (tok, chunk) in tokens.iter().zip(text_split.iter()) {
                ts.push("<star>".into());
                ts.push(tok.clone());
                xs.push("<star>".into());
                xs.push(chunk.clone());
            }
            (ts, xs)
        }
        "edges" | _ => {
            let mut ts = vec!["<star>".into()];
            ts.extend(tokens);
            ts.push("<star>".into());
            let mut xs = vec!["<star>".into()];
            xs.extend(text_split);
            xs.push("<star>".into());
            (ts, xs)
        }
    };
    Ok((tokens_starred, text_starred))
}

fn split_text(text: &str, split_size: &str) -> Vec<String> {
    match split_size {
        "char" => text.chars().map(|c| c.to_string()).collect(),
        "sentence" => {
            // fallback: split on . ! ?
            let mut out = Vec::new();
            let mut cur = String::new();
            for ch in text.chars() {
                cur.push(ch);
                if matches!(ch, '.' | '!' | '?') {
                    let t = cur.trim().to_string();
                    if !t.is_empty() {
                        out.push(t);
                    }
                    cur.clear();
                }
            }
            let t = cur.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            if out.is_empty() && !text.trim().is_empty() {
                out.push(text.trim().to_string());
            }
            out
        }
        _ => text.split_whitespace().map(|s| s.to_string()).collect(),
    }
}

/// Map space-joined romanized tokens to vocab indices (Python `get_alignments`).
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
            None => {
                // skip unknown (Python filters `if c in dictionary`)
                log::warn!("token {part:?} not in vocab — skipped");
            }
        }
    }
    if out.is_empty() {
        bail!("no tokens mapped into vocab");
    }
    Ok(out)
}
