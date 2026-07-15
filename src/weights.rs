//! Safetensors weight loading via mmap + zero-copy `Bytes` slices.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::anyhow;
use bytes::Bytes;
use memmap2::Mmap;

use crate::raw_tensor::RawTensor;

pub(crate) fn load_weights(model_dir: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let wm = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid model.safetensors.index.json"))?;
        let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();
        for v in wm.values() {
            if let Some(s) = v.as_str() {
                files.insert(s.to_string());
            }
        }
        let mut all = HashMap::new();
        for s in files {
            all.extend(load_safetensors_file(&model_dir.join(&s))?);
        }
        return Ok(all);
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return load_safetensors_file(&single);
    }

    Err(anyhow!(
        "no model.safetensors (or sharded index) under {}",
        model_dir.display()
    ))
}

fn load_safetensors_file(path: &Path) -> anyhow::Result<HashMap<String, RawTensor>> {
    let file = File::open(path).map_err(|e| anyhow!("open {:?}: {}", path, e))?;
    // SAFETY: read-only checkpoint; we never mutate the file while mapped.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| anyhow!("mmap {:?}: {}", path, e))?;

    let buf: Bytes = Bytes::from_owner(mmap);
    let st = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| anyhow!("safetensors {:?}: {}", path, e))?;

    let base = buf.as_ptr() as usize;
    let mut weights = HashMap::with_capacity(st.len());
    for (name, view) in st.iter() {
        let view_data = view.data();
        let offset = view_data.as_ptr() as usize - base;
        let len = view_data.len();
        weights.insert(
            name.to_string(),
            RawTensor {
                data: buf.slice(offset..offset + len),
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            },
        );
    }
    Ok(weights)
}
