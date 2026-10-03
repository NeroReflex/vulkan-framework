//! Metadata for `textures/<name>/data` payloads inside an object tar.
//! Width, height, mip count, and format come from the DDS or KTX2 header only.

use std::io::{Cursor, Read, Result};
use std::mem::MaybeUninit;

use vulkan_framework::ash::vk;

use super::block;
use super::directdraw_surface::{DDSHeader, DDSHeaderDXT10, DirectDrawSurface};
use super::ktx2;

pub struct CookedTextureHeader {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub prefix_bytes: u64,
    pub ktx_placements: Option<Vec<ktx2::Placement>>,
}

impl CookedTextureHeader {
    pub fn payload_bytes(&self, archive_entry_size: u64) -> u64 {
        match &self.ktx_placements {
            Some(placements) => ktx2::payload_bytes(placements),
            None => archive_entry_size - self.prefix_bytes,
        }
    }

    pub fn expected_payload_bytes(&self) -> Option<u64> {
        if self.ktx_placements.is_some() {
            return None;
        }
        let chain = block::mip_chain(self.format, self.width, self.height, self.mip_levels)?;
        Some(chain.iter().map(|level| level.bytes).sum())
    }
}

/// Read the file prefix (magic + headers) from a tar `data` entry.
pub fn read_header<R: Read>(mut reader: R, _total_size: u64) -> Result<CookedTextureHeader> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    let mut prefix_bytes = 4u64;

    if magic == *b"DDS " {
        let mut dds_uninitialized = MaybeUninit::<DDSHeader>::uninit();
        let dds_header_slice = unsafe {
            std::slice::from_raw_parts_mut(
                dds_uninitialized.as_mut_ptr() as *mut std::ffi::c_void as *mut u8,
                std::mem::size_of::<DDSHeader>(),
            )
        };
        reader.read_exact(dds_header_slice)?;
        prefix_bytes += std::mem::size_of::<DDSHeader>() as u64;
        let dds_header = unsafe { dds_uninitialized.assume_init() };

        let dds_dxt10_header = if dds_header.is_followed_by_dxt10_header() {
            let mut dxt10_uninitialized = MaybeUninit::<DDSHeaderDXT10>::uninit();
            let dx10_header_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dxt10_uninitialized.as_mut_ptr() as *mut std::ffi::c_void as *mut u8,
                    std::mem::size_of::<DDSHeaderDXT10>(),
                )
            };
            reader.read_exact(dx10_header_slice)?;
            prefix_bytes += std::mem::size_of::<DDSHeaderDXT10>() as u64;
            Some(unsafe { dxt10_uninitialized.assume_init() })
        } else {
            None
        };

        let surface = DirectDrawSurface::new(dds_header, dds_dxt10_header);
        let format = surface.vulkan_format();
        if format == vk::Format::UNDEFINED {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported DDS pixel format",
            ));
        }

        return Ok(CookedTextureHeader {
            format,
            width: surface.width(),
            height: surface.height(),
            mip_levels: surface.mip_level_count(),
            prefix_bytes,
            ktx_placements: None,
        });
    }

    let mut magic_rest = [0u8; 8];
    reader.read_exact(&mut magic_rest)?;
    prefix_bytes += 8;
    let mut ident = [0u8; 12];
    ident[0..4].copy_from_slice(&magic);
    ident[4..12].copy_from_slice(&magic_rest);
    if ident != ktx2::IDENTIFIER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "texture data must be DDS or KTX2",
        ));
    }

    let mut rest = vec![0u8; ktx2::HEADER_BYTES - 12];
    reader.read_exact(&mut rest)?;
    prefix_bytes += rest.len() as u64;
    let mut full = ident.to_vec();
    full.extend_from_slice(&rest);
    let level_count = u32::from_le_bytes(full[48..52].try_into().unwrap());
    let index_len = level_count as usize * ktx2::LEVEL_INDEX_BYTES;
    let mut index = vec![0u8; index_len];
    reader.read_exact(&mut index)?;
    prefix_bytes += index_len as u64;
    full.extend_from_slice(&index);
    let parsed = ktx2::parse(&full).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "KTX2 header is missing an uncompressed level index",
        )
    })?;
    let placements = ktx2::vulkan_placements(&parsed);

    Ok(CookedTextureHeader {
        format: parsed.format,
        width: parsed.width,
        height: parsed.height,
        mip_levels: parsed.levels.len() as u32,
        prefix_bytes,
        ktx_placements: Some(placements),
    })
}

/// Validate a full `data` blob (header + payload) from memory.
pub fn validate_data_blob(data: &[u8]) -> std::result::Result<(), String> {
    let total = data.len() as u64;
    let header = read_header(Cursor::new(data), total).map_err(|err| err.to_string())?;
    let payload_len = header.payload_bytes(total);
    if header.ktx_placements.is_none() {
        let expected = header
            .expected_payload_bytes()
            .ok_or_else(|| "could not size mip chain".to_string())?;
        if payload_len != expected {
            return Err(format!(
                "DDS payload is {payload_len} bytes, expected {expected} for {}x{} {} mips",
                header.width,
                header.height,
                header.mip_levels
            ));
        }
    }
    let consumed = header.prefix_bytes + payload_len;
    if consumed != total {
        return Err(format!(
            "texture data entry is {total} bytes but header + payload need {consumed}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::PathBuf;
    use tar::Archive;

    #[test]
    fn sponza_tar_bc7_textures_match_dds_headers() {
        let tar_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../crytek_sponza.tar");
        if !tar_path.is_file() {
            eprintln!("skipping: crytek_sponza.tar not present");
            return;
        }

        let file = File::open(&tar_path).unwrap();
        let mut archive = Archive::new(file);
        let mut validated = 0usize;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().to_string_lossy().into_owned();
            if !path.contains("/textures/") || !path.ends_with("/data") {
                continue;
            }
            let size = entry.header().size().unwrap();
            let mut data = vec![0u8; size as usize];
            entry.read_exact(&mut data).unwrap();
            validate_data_blob(&data).unwrap_or_else(|err| {
                panic!("{path}: {err}");
            });
            validated += 1;
        }
        assert_eq!(validated, 42, "expected 42 BC7 textures in crytek_sponza.tar");
    }
}
