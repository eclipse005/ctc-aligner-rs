//! Path → segments → timestamps (Python merge_repeats / get_spans / postprocess).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::inference::ForcedAlignItem;

#[derive(Debug, Clone)]
pub struct Segment {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// Collapse consecutive identical path labels (Python `merge_repeats`).
pub fn merge_repeats(path: &[usize], idx_to_token: &HashMap<usize, String>) -> Vec<Segment> {
    let mut segments = Vec::new();
    if path.is_empty() {
        return segments;
    }
    let mut i1 = 0usize;
    while i1 < path.len() {
        let mut i2 = i1 + 1;
        while i2 < path.len() && path[i1] == path[i2] {
            i2 += 1;
        }
        let label = idx_to_token
            .get(&path[i1])
            .cloned()
            .unwrap_or_else(|| format!("id={}", path[i1]));
        segments.push(Segment {
            label,
            start: i1,
            end: i2 - 1,
        });
        i1 = i2;
    }
    segments
}

/// Map starred tokens onto collapsed segments → span ranges (Python `get_spans`).
pub fn get_spans(
    tokens: &[String],
    segments: &[Segment],
    blank: &str,
) -> Result<Vec<Vec<Segment>>> {
    let mut ltr_idx = 0usize;
    let mut tokens_idx = 0usize;
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;

    for (seg_idx, seg) in segments.iter().enumerate() {
        if tokens_idx == tokens.len() {
            if seg_idx != segments.len() - 1 {
                bail!("extra segments after tokens exhausted");
            }
            if seg.label != blank {
                bail!("expected trailing blank, got {:?}", seg.label);
            }
            continue;
        }
        let cur_token: Vec<&str> = tokens[tokens_idx].split(' ').collect();
        if cur_token.is_empty() {
            bail!("empty token at index {tokens_idx}");
        }
        let ltr = cur_token[ltr_idx];
        if seg.label == blank {
            continue;
        }
        if seg.label != ltr {
            bail!("segment label {:?} != expected {:?}", seg.label, ltr);
        }
        if ltr_idx == 0 {
            start = seg_idx;
        }
        if ltr_idx == cur_token.len() - 1 {
            ltr_idx = 0;
            tokens_idx += 1;
            intervals.push((start, seg_idx));
            while tokens_idx < tokens.len() && tokens[tokens_idx].is_empty() {
                intervals.push((seg_idx, seg_idx));
                tokens_idx += 1;
            }
        } else {
            ltr_idx += 1;
        }
    }

    let mut spans = Vec::with_capacity(intervals.len());
    for (s, e) in intervals {
        spans.push(segments[s..=e].to_vec());
    }
    Ok(spans)
}

/// Convert spans to timestamps in seconds.
pub fn spans_to_items(
    text_pieces: &[String],
    spans: &[Vec<Segment>],
    stride_ms: f32,
) -> Vec<ForcedAlignItem> {
    let mut items = Vec::new();
    for (i, span) in spans.iter().enumerate() {
        if span.is_empty() {
            continue;
        }
        let start_frame = span.first().map(|s| s.start).unwrap_or(0);
        let end_frame = span.last().map(|s| s.end).unwrap_or(0);
        let start = (start_frame as f32 * stride_ms) / 1000.0;
        // Python uses inclusive end frame → (end+1) * stride
        let end = ((end_frame + 1) as f32 * stride_ms) / 1000.0;
        let text = text_pieces.get(i).cloned().unwrap_or_default();
        if text == "<star>" {
            continue;
        }
        items.push(ForcedAlignItem {
            start: round2(start as f64),
            end: round2(end as f64),
            text,
        });
    }
    items
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[derive(Serialize)]
struct JsonItem {
    start: f64,
    end: f64,
    text: String,
}

pub fn write_forced_align_items_json(path: &Path, items: &[ForcedAlignItem]) -> Result<()> {
    let rows: Vec<JsonItem> = items
        .iter()
        .map(|i| JsonItem {
            start: i.start,
            end: i.end,
            text: i.text.clone(),
        })
        .collect();
    let s = serde_json::to_string_pretty(&rows)?;
    std::fs::write(path, s)?;
    Ok(())
}
