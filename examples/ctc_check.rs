fn main() -> anyhow::Result<()> {
    let shape = std::fs::read_to_string(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\py_em_shape.txt")?;
    let mut it = shape.split_whitespace();
    let t: usize = it.next().unwrap().parse()?;
    let c: usize = it.next().unwrap().parse()?;
    let bytes = std::fs::read(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\py_em_en15s.bin")?;
    let em: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
    let targets_bytes = std::fs::read(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\targets.npy")?;
    // npy header skip - use raw from json instead
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\meta.json")?)?;
    let blank = meta["blank"].as_u64().unwrap() as usize;
    // load targets from npy with simple parse: skip header to data
    let targets = {
        // np.save int64 - find magic
        let data_offset = {
            assert_eq!(&targets_bytes[0..6], b"\x93NUMPY");
            let header_len = u16::from_le_bytes([targets_bytes[8], targets_bytes[9]]) as usize;
            10 + header_len
        };
        targets_bytes[data_offset..].chunks_exact(8).map(|b| {
            i64::from_le_bytes(b.try_into().unwrap()) as usize
        }).collect::<Vec<_>>()
    };
    println!("T={} C={} targets={} blank={}", t, c, targets.len(), blank);
    let (paths, _scores) = ctc_forced_aligner_rs::ctc::forced_align(&em, t, c, &targets, blank)?;
    // save
    let pb: Vec<u8> = paths.iter().flat_map(|p| (*p as u32).to_le_bytes()).collect();
    std::fs::write(r"D:\ctc-forced-aligner\ctc-aligner-rs\tmp\rust_path_from_py_em.bin", pb)?;
    println!("path first 40: {:?}", &paths[..40.min(paths.len())]);
    Ok(())
}
