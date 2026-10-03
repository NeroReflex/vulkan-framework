//! Rewrite a legacy object tar without texture sidecar text files.

use std::fs::File;
use std::io::Read;
use std::path::Path;

const SIDECAR_SUFFIXES: [&str; 3] = ["width.txt", "height.txt", "miplevel.txt"];

pub fn is_texture_sidecar(path: &str) -> bool {
    let trimmed = path.strip_prefix("./").unwrap_or(path);
    if !trimmed.starts_with("textures/") {
        return false;
    }
    SIDECAR_SUFFIXES.iter().any(|suffix| trimmed.ends_with(suffix))
}

pub fn repack(input: &Path, output: &Path) -> Result<(), String> {
    let input_file = File::open(input).map_err(|err| err.to_string())?;
    let mut archive = tar::Archive::new(input_file);
    let output_file = File::create(output).map_err(|err| err.to_string())?;
    let mut builder = tar::Builder::new(output_file);

    let mut copied = 0usize;
    let mut skipped = 0usize;
    for entry in archive.entries().map_err(|err| err.to_string())? {
        let mut entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path().map_err(|err| err.to_string())?;
        let path_str = path.to_string_lossy();
        if is_texture_sidecar(&path_str) {
            skipped += 1;
            continue;
        }

        let mut header = tar::Header::new_ustar();
        header
            .set_path(path.as_os_str())
            .map_err(|err| err.to_string())?;
        header.set_mode(entry.header().mode().unwrap_or(0o644));
        header.set_entry_type(entry.header().entry_type());

        if entry.header().entry_type().is_symlink() {
            let link = entry
                .link_name()
                .map_err(|err| err.to_string())?
                .ok_or_else(|| format!("symlink without target: {path_str}"))?;
            header.set_size(0);
            builder
                .append_link(&mut header, path.as_os_str(), link)
                .map_err(|err| err.to_string())?;
        } else {
            let size = entry.header().size().map_err(|err| err.to_string())?;
            let mut payload = Vec::with_capacity(size as usize);
            entry
                .read_to_end(&mut payload)
                .map_err(|err| err.to_string())?;
            header.set_size(size);
            header.set_cksum();
            builder
                .append(&header, payload.as_slice())
                .map_err(|err| err.to_string())?;
        }
        copied += 1;
    }

    builder.finish().map_err(|err| err.to_string())?;
    eprintln!("repack: copied {copied} entries, dropped {skipped} texture sidecars");
    Ok(())
}

pub fn validate_textures(input: &Path) -> Result<(), String> {
    let file = File::open(input).map_err(|err| err.to_string())?;
    let mut archive = tar::Archive::new(file);
    let mut count = 0usize;
    for entry in archive.entries().map_err(|err| err.to_string())? {
        let mut entry = entry.map_err(|err| err.to_string())?;
        let path_str = entry
            .path()
            .map_err(|err| err.to_string())?
            .to_string_lossy()
            .into_owned();
        let is_texture_data = path_str.ends_with("/data")
            && (path_str.contains("/textures/") || path_str.starts_with("textures/"));
        if !is_texture_data {
            continue;
        }
        let size = entry.header().size().map_err(|err| err.to_string())?;
        let mut data = vec![0u8; size as usize];
        entry
            .read_exact(&mut data)
            .map_err(|err| format!("{path_str}: {err}"))?;
        validate_texture_blob(&data).map_err(|err| format!("{path_str}: {err}"))?;
        count += 1;
    }
    eprintln!("validate: {count} texture data entries ok");
    Ok(())
}

fn validate_texture_blob(data: &[u8]) -> Result<(), String> {
    if data.len() < 4 {
        return Err("texture data too small".into());
    }
    if &data[0..4] == b"DDS " {
        return validate_dds(data);
    }
    if data.len() >= 12 && &data[0..12] == KTX2_IDENT {
        return Ok(());
    }
    Err("expected DDS or KTX2".into())
}

const KTX2_IDENT: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

fn validate_dds(data: &[u8]) -> Result<(), String> {
    if data.len() < 4 + 124 {
        return Err("truncated DDS header".into());
    }
    let header = &data[4..4 + 124];
    let width = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let height = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let mip_map_count = u32::from_le_bytes(header[24..28].try_into().unwrap());
    let mip_levels = if mip_map_count == 0 { 1 } else { mip_map_count };
    let four_cc = u32::from_le_bytes(header[80..84].try_into().unwrap());
    let prefix = if four_cc == 0x3031_5844 {
        if data.len() < 4 + 124 + 20 {
            return Err("truncated DDS DX10 header".into());
        }
        4 + 124 + 20
    } else {
        4 + 124
    };
    let payload = data.len() - prefix;
    let expected = bc7_mip_payload_bytes(width, height, mip_levels);
    if payload != expected {
        return Err(format!(
            "payload is {payload} bytes, expected {expected} for {width}x{height} with {mip_levels} mips"
        ));
    }
    Ok(())
}

fn bc7_mip_payload_bytes(width: u32, height: u32, levels: u32) -> usize {
    let mut total = 0usize;
    let mut w = width.max(1);
    let mut h = height.max(1);
    for _ in 0..levels {
        let blocks_x = (w + 3) / 4;
        let blocks_y = (h + 3) / 4;
        total += (blocks_x * blocks_y * 16) as usize;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn drops_texture_sidecars() {
        assert!(is_texture_sidecar("./textures/foo/width.txt"));
        assert!(!is_texture_sidecar("./textures/foo/data"));
    }

    #[test]
    fn repack_strips_sidecars_from_minimal_tar() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut builder = tar::Builder::new(&mut cursor);
            for (path, body) in [
                ("textures/brick/data", vec![0u8; 8]),
                ("textures/brick/width.txt", b"0".to_vec()),
            ] {
                let mut header = tar::Header::new_ustar();
                header.set_path(path).unwrap();
                header.set_size(body.len() as u64);
                header.set_cksum();
                builder.append(&header, body.as_slice()).unwrap();
            }
            builder.finish().unwrap();
        }

        let in_path = std::env::temp_dir().join(format!("artrtic-repack-in-{}", std::process::id()));
        let out_path = std::env::temp_dir().join(format!("artrtic-repack-out-{}", std::process::id()));
        std::fs::write(&in_path, &cursor.into_inner()).unwrap();
        repack(&in_path, &out_path).unwrap();

        let file = File::open(&out_path).unwrap();
        let mut archive = tar::Archive::new(file);
        let names: Vec<_> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["textures/brick/data".to_string()]);
        let _ = std::fs::remove_file(in_path);
        let _ = std::fs::remove_file(out_path);
    }
}
