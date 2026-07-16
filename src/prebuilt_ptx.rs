//! Precompiled multi-arch PTX selection (scheme B).
//!
//! Same design as `qwen-aligner-rs` / `cohere-transcribe-native`:
//! - Prebuilt PTX lives in `ptx/kernels_smXX.ptx` (rebuild offline with nvcc if kernels change)
//! - Embed at compile time via `include_str!`
//! - Runtime: pick highest prebuilt SM ≤ device compute capability
//! - End users need **no** CUDA Toolkit / NVRTC (only driver + cudart/cublas DLL)

/// Supported prebuilt compute capabilities, ascending.
pub const PREBUILT_SMS: &[u32] = &[61, 70, 75, 80, 86, 89, 90];

/// Minimum supported device CC (must match first entry of [`PREBUILT_SMS`]).
pub const MIN_SM: u32 = 61;

/// Encode `(major, minor)` as a single integer, e.g. `(8, 6) -> 86`.
#[inline]
pub fn sm_code(major: i32, minor: i32) -> u32 {
    (major as u32) * 10 + (minor as u32)
}

/// Choose the highest prebuilt SM ≤ `device_sm`.
pub fn select_prebuilt_sm(device_sm: u32) -> Option<u32> {
    let mut best = None;
    for &sm in PREBUILT_SMS {
        if sm <= device_sm {
            best = Some(sm);
        } else {
            break;
        }
    }
    best
}

/// Embedded PTX source for the given prebuilt SM code.
pub fn ptx_for_sm(sm: u32) -> Option<&'static str> {
    match sm {
        61 => Some(include_str!("../ptx/kernels_sm61.ptx")),
        70 => Some(include_str!("../ptx/kernels_sm70.ptx")),
        75 => Some(include_str!("../ptx/kernels_sm75.ptx")),
        80 => Some(include_str!("../ptx/kernels_sm80.ptx")),
        86 => Some(include_str!("../ptx/kernels_sm86.ptx")),
        89 => Some(include_str!("../ptx/kernels_sm89.ptx")),
        90 => Some(include_str!("../ptx/kernels_sm90.ptx")),
        _ => None,
    }
}

/// Resolve PTX text for a live device `(major, minor)`.
pub fn resolve_ptx_for_device(major: i32, minor: i32) -> Result<(&'static str, u32), String> {
    let device_sm = sm_code(major, minor);
    let selected = select_prebuilt_sm(device_sm).ok_or_else(|| {
        format!(
            "GPU compute capability sm_{device_sm} is below the minimum supported sm_{MIN_SM}"
        )
    })?;
    let ptx = ptx_for_sm(selected)
        .ok_or_else(|| format!("internal error: missing embedded PTX for sm_{selected}"))?;
    Ok((ptx, selected))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kernel entry points loaded by `CudaKernels::load_all` (f32 path).
    /// Keep in sync with `cudarc_engine.rs`. After editing `kernels.cu`, re-run
    /// `scripts/compile-ptx.ps1` so every name below is present in `ptx/`.
    const REQUIRED_F32_KERNELS: &[&str] = &[
        "layer_norm_f32",
        "gelu_inplace_f32",
        "add_inplace_f32",
        "add_bias_inplace_f32",
        "bias_residual_f32",
        "softmax_inplace_last_dim_f32",
        "merge_heads_f32",
        "split_qkv_to_heads_f32",
        "im2col_1d_f32",
        "im2col_1d_ch_f32",
        "pad_time_f32",
        "add_bias_gelu_inplace_f32",
        "scatter_group_bias_gelu_f32",
        "split_qkv_to_heads_batched_f32",
        "merge_heads_batched_f32",
    ];

    #[test]
    fn selects_exact_and_floor() {
        assert_eq!(select_prebuilt_sm(61), Some(61));
        assert_eq!(select_prebuilt_sm(86), Some(86));
        assert_eq!(select_prebuilt_sm(87), Some(86));
        assert_eq!(select_prebuilt_sm(100), Some(90));
    }

    #[test]
    fn rejects_below_minimum() {
        assert_eq!(select_prebuilt_sm(60), None);
        assert!(resolve_ptx_for_device(5, 0).is_err());
    }

    #[test]
    fn embeds_nonempty_ptx() {
        for &sm in PREBUILT_SMS {
            let ptx = ptx_for_sm(sm).expect("missing PTX");
            assert!(
                ptx.len() > 100 && ptx.contains(".version"),
                "PTX sm_{sm} invalid (len={})",
                ptx.len()
            );
        }
    }

    /// Guards the v1.3.0 regression: kernels.cu advanced past stale prebuilt PTX,
    /// so `CudaKernels::load_all` failed at runtime with a useless outer message.
    #[test]
    fn prebuilt_ptx_contains_all_loaded_f32_kernels() {
        for &sm in PREBUILT_SMS {
            let ptx = ptx_for_sm(sm).expect("missing PTX");
            for &name in REQUIRED_F32_KERNELS {
                let entry = format!(".visible .entry {name}(");
                assert!(
                    ptx.contains(&entry),
                    "PTX sm_{sm} missing kernel {name}; re-run scripts/compile-ptx.ps1"
                );
            }
        }
    }
}
