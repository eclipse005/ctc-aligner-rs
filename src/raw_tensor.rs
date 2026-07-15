//! Raw on-disk tensor view from safetensors (deserialization intermediate).

use anyhow::{anyhow, Result};
use bytes::Bytes;
use safetensors::Dtype;

/// One tensor as it sits in the safetensors file: raw bytes + shape + dtype.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Little-endian payload; backed by a refcounted mmap slice.
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Ok(self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            Dtype::F16 => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]).to_f32())
                .collect()),
            Dtype::BF16 => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    f32::from_bits((b as u32) << 16)
                })
                .collect()),
            other => Err(anyhow!("unsupported dtype {:?} for to_f32_vec", other)),
        }
    }

    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>> {
        match self.dtype {
            Dtype::F16 => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                .collect()),
            Dtype::F32 => Ok(self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .map(half::f16::from_f32)
                .collect()),
            Dtype::BF16 => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| {
                    let b = u16::from_ne_bytes([c[0], c[1]]);
                    half::f16::from_f32(f32::from_bits((b as u32) << 16))
                })
                .collect()),
            other => Err(anyhow!("unsupported dtype {:?} for to_f16_vec", other)),
        }
    }
}
