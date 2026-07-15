//! Path → segments → timestamps (Python merge_repeats / get_spans / postprocess_results).

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
///
/// Includes blank-padding into neighbouring silence (half-split), which is critical
/// for timestamp boundaries matching the original Python package.
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
                bail!("extra segments after tokens exhausted at seg {seg_idx}");
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
            bail!(
                "segment label {:?} != expected {:?} (token_idx={tokens_idx} ltr={ltr_idx})",
                seg.label,
                ltr
            );
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
    for (idx, &(s, e)) in intervals.iter().enumerate() {
        let mut span = segments[s..=e].to_vec();
        // Pad with half of neighbouring blank segments (Python get_spans).
        if s > 0 {
            let prev_seg = &segments[s - 1];
            if prev_seg.label == blank {
                let pad_start = if idx == 0 {
                    prev_seg.start
                } else {
                    (prev_seg.start + prev_seg.end) / 2
                };
                span.insert(
                    0,
                    Segment {
                        label: blank.to_string(),
                        start: pad_start,
                        end: span[0].start,
                    },
                );
            }
        }
        if e + 1 < segments.len() {
            let next_seg = &segments[e + 1];
            if next_seg.label == blank {
                let pad_end = if idx == intervals.len() - 1 {
                    next_seg.end
                } else {
                    // math.floor((start+end)/2) — integer division for non-neg
                    (next_seg.start + next_seg.end) / 2
                };
                let last_end = span.last().map(|x| x.end).unwrap_or(0);
                span.push(Segment {
                    label: blank.to_string(),
                    start: last_end,
                    end: pad_end,
                });
            }
        }
        spans.push(span);
    }
    Ok(spans)
}

/// Python `postprocess_results` (score optional on public items).
///
/// End time uses inclusive `seg_end_idx * stride_ms / 1000` — **no +1**.
pub fn spans_to_items(
    text_pieces: &[String],
    spans: &[Vec<Segment>],
    stride_ms: f32,
    scores: &[f32],
    merge_threshold: f32,
) -> Vec<ForcedAlignItem> {
    let mut items = Vec::new();
    for (i, t) in text_pieces.iter().enumerate() {
        if t == "<star>" {
            continue;
        }
        if i >= spans.len() {
            break;
        }
        let span = &spans[i];
        if span.is_empty() {
            continue;
        }
        let seg_start_idx = span.first().map(|s| s.start).unwrap_or(0);
        let seg_end_idx = span.last().map(|s| s.end).unwrap_or(0);
        let start = seg_start_idx as f32 * stride_ms / 1000.0;
        let end = seg_end_idx as f32 * stride_ms / 1000.0;
        // Python: scores[seg_start_idx:seg_end_idx]  (end exclusive)
        let score = if seg_end_idx > seg_start_idx && seg_end_idx <= scores.len() {
            scores[seg_start_idx..seg_end_idx].iter().sum::<f32>()
        } else {
            0.0
        };
        items.push(ForcedAlignItem {
            start: start as f64,
            end: end as f64,
            text: t.clone(),
            score: Some(score as f64),
        });
    }
    merge_segments(&mut items, merge_threshold);
    items
}

fn merge_segments(segments: &mut [ForcedAlignItem], threshold: f32) {
    if segments.is_empty() {
        return;
    }
    for i in 0..segments.len() - 1 {
        if (segments[i + 1].start - segments[i].end) < threshold as f64 {
            segments[i + 1].start = segments[i].end;
        }
    }
}

#[derive(Serialize)]
struct JsonItem {
    start: f64,
    end: f64,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
}

pub fn write_forced_align_items_json(path: &Path, items: &[ForcedAlignItem]) -> Result<()> {
    let rows: Vec<JsonItem> = items
        .iter()
        .map(|i| JsonItem {
            start: i.start,
            end: i.end,
            text: i.text.clone(),
            score: i.score,
        })
        .collect();
    let s = serde_json::to_string_pretty(&rows)?;
    std::fs::write(path, s)?;
    Ok(())
}
