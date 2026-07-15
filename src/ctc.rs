//! CTC forced alignment (Viterbi).
//!
//! Port of Python `ctc_forced_aligner/forced_align_impl.py` — bit-compatible
//! control flow with the numba/pure-Python reference.

use anyhow::{bail, Result};

/// Align a CTC label sequence to frame-wise log-probabilities.
///
/// * `log_probs`: shape `(T, C)`, f32, row-major
/// * `targets`: non-blank token indices length `L`
/// * `blank`: blank token id (usually 0)
///
/// Returns `(paths, scores)` where `paths` has length `T` (token id per frame)
/// and `scores[t] = log_probs[t, paths[t]]`.
pub fn forced_align(
    log_probs: &[f32],
    t: usize,
    c: usize,
    targets: &[usize],
    blank: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    if log_probs.len() != t.saturating_mul(c) {
        bail!(
            "log_probs length {} != T*C ({}*{})",
            log_probs.len(),
            t,
            c
        );
    }
    if blank >= c {
        bail!("blank {blank} out of range for C={c}");
    }
    if targets.iter().any(|&v| v == blank) {
        bail!("targets must not contain blank index {blank}");
    }
    if targets.iter().any(|&v| v >= c) {
        bail!("targets values must be within [0, C)");
    }

    let l = targets.len();
    if l == 0 {
        bail!("empty targets");
    }
    if t == 0 {
        bail!("empty log_probs (T=0)");
    }

    let mut repeats = 0usize;
    for i in 1..l {
        if targets[i] == targets[i - 1] {
            repeats += 1;
        }
    }
    if t < l + repeats {
        bail!("targets length too long for CTC (T={t}, L={l}, R={repeats})");
    }

    let s = 2 * l + 1;
    let neg_inf = f32::NEG_INFINITY;
    let mut alphas = vec![neg_inf; 2 * s];
    // back_ptr[t * s + i] ∈ {0,1,2}
    let mut back_ptr = vec![0u8; t * s];

    let mut start = if t > l + repeats { 0usize } else { 1usize };
    let mut end = if s == 1 { 1usize } else { 2usize };

    for i in start..end {
        let label_idx = if i % 2 == 0 {
            blank
        } else {
            targets[i / 2]
        };
        alphas[i] = log_probs[label_idx];
    }

    for time in 1..t {
        if t - time <= l + repeats {
            if start % 2 == 1 && targets[start / 2] != targets[start / 2 + 1] {
                start += 1;
            }
            start += 1;
        }

        if time <= l + repeats {
            if end % 2 == 0 && end < 2 * l && targets[end / 2 - 1] != targets[end / 2] {
                end += 1;
            }
            end += 1;
        }

        let cur_idx = time % 2;
        let prev_idx = (time - 1) % 2;
        let cur_base = cur_idx * s;
        let prev_base = prev_idx * s;

        for i in 0..s {
            alphas[cur_base + i] = neg_inf;
        }

        let mut start_loop = start;
        if start == 0 {
            alphas[cur_base] = alphas[prev_base] + log_probs[time * c + blank];
            back_ptr[time * s] = 0;
            start_loop += 1;
        }

        for i in start_loop..end {
            let x0 = alphas[prev_base + i];
            let x1 = alphas[prev_base + i - 1];
            let mut x2 = neg_inf;
            let label_idx = if i % 2 == 0 {
                blank
            } else {
                targets[i / 2]
            };
            if i % 2 != 0 && i != 1 && targets[i / 2] != targets[i / 2 - 1] {
                x2 = alphas[prev_base + i - 2];
            }

            let (result, step) = if x2 > x1 && x2 > x0 {
                (x2, 2u8)
            } else if x1 > x0 && x1 > x2 {
                (x1, 1u8)
            } else {
                (x0, 0u8)
            };
            alphas[cur_base + i] = result + log_probs[time * c + label_idx];
            back_ptr[time * s + i] = step;
        }
    }

    // Backtrack
    let mut paths = vec![0usize; t];
    let idx_last = (t - 1) % 2;
    let last_base = idx_last * s;
    let mut ltr_idx = if alphas[last_base + s - 1] > alphas[last_base + s - 2] {
        s - 1
    } else {
        s - 2
    };

    for time in (0..t).rev() {
        let lbl_idx = if ltr_idx % 2 == 0 {
            blank
        } else {
            targets[ltr_idx / 2]
        };
        paths[time] = lbl_idx;
        let step = back_ptr[time * s + ltr_idx] as usize;
        if ltr_idx < step {
            bail!("backtracking failed (index < 0) at t={time}");
        }
        ltr_idx -= step;
    }

    let mut scores = vec![0f32; t];
    for time in 0..t {
        scores[time] = log_probs[time * c + paths[time]];
    }

    Ok((paths, scores))
}

/// Row-wise log-softmax: `out[t,c] = x[t,c] - logsumexp(x[t,:])`.
pub fn log_softmax_rows_into(src: &[f32], t: usize, c: usize, dst: &mut [f32]) {
    assert_eq!(src.len(), t * c);
    assert_eq!(dst.len(), t * c);
    for row in 0..t {
        let base = row * c;
        let row_src = &src[base..base + c];
        let max = row_src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for &v in row_src {
            sum += (v - max).exp();
        }
        let log_z = max + sum.ln();
        for j in 0..c {
            dst[base + j] = row_src[j] - log_z;
        }
    }
}

/// In-place log-softmax over rows of a dense `(T, C)` matrix.
pub fn log_softmax_rows_inplace(logits: &mut [f32], t: usize, c: usize) {
    assert_eq!(logits.len(), t * c);
    // Parallel over frames when rayon is available (default builds).
    #[cfg(feature = "cpu")]
    {
        use rayon::prelude::*;
        logits.par_chunks_mut(c).for_each(|row_slice| {
            let max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in row_slice.iter() {
                sum += (*v - max).exp();
            }
            let log_z = max + sum.ln();
            for v in row_slice.iter_mut() {
                *v -= log_z;
            }
        });
        return;
    }
    #[cfg(not(feature = "cpu"))]
    for row in 0..t {
        let base = row * c;
        let row_slice = &mut logits[base..base + c];
        let max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for &v in row_slice.iter() {
            sum += (v - max).exp();
        }
        let log_z = max + sum.ln();
        for v in row_slice.iter_mut() {
            *v -= log_z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_path_no_repeats() {
        // T=3, C=3 (blank=0, a=1, b=2), targets = [a, b]
        // Make path blank-a-b or a-b-blank clearly best.
        let t = 3usize;
        let c = 3usize;
        let mut log_probs = vec![f32::NEG_INFINITY; t * c];
        // t0 prefers blank, t1 prefers a, t2 prefers b
        log_probs[0 * c + 0] = 0.0;
        log_probs[1 * c + 1] = 0.0;
        log_probs[2 * c + 2] = 0.0;
        // weak alternatives
        for time in 0..t {
            for cls in 0..c {
                if log_probs[time * c + cls].is_infinite() {
                    log_probs[time * c + cls] = -10.0;
                }
            }
        }
        let (paths, _scores) = forced_align(&log_probs, t, c, &[1, 2], 0).unwrap();
        assert_eq!(paths.len(), 3);
        // Must cover targets a,b somewhere
        assert!(paths.contains(&1));
        assert!(paths.contains(&2));
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let t = 2;
        let c = 3;
        let mut x = vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        log_softmax_rows_inplace(&mut x, t, c);
        for row in 0..t {
            let sum: f32 = x[row * c..(row + 1) * c].iter().map(|v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum={sum}");
        }
    }
}
