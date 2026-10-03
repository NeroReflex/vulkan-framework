//! KTX2 header parse for ASTC payloads.
//!
//! The engine does not link KTX-Software. A cooked file already contains the
//! compressed blocks. This module only reads the header so the loader can
//! scatter those blocks into a mapped GPU buffer, largest mip first.

use vulkan_framework::ash::vk;

pub const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

pub const HEADER_BYTES: usize = 80;
pub const LEVEL_INDEX_BYTES: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelIndex {
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ktx2Header {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<LevelIndex>,
}

/// `file_offset` is from the start of the KTX2 file. `dest_offset` is where
/// that level sits in the mapped staging buffer (largest mip at offset 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub file_offset: u64,
    pub dest_offset: u64,
    pub length: u64,
}

pub fn vulkan_placements(header: &Ktx2Header) -> Vec<Placement> {
    let mut dest_offset = 0u64;
    vulkan_level_order(header)
        .into_iter()
        .map(|level| {
            let placement = Placement {
                file_offset: level.byte_offset,
                dest_offset,
                length: level.byte_length,
            };
            dest_offset += level.byte_length;
            placement
        })
        .collect()
}

pub fn payload_bytes(placements: &[Placement]) -> u64 {
    placements.iter().map(|placement| placement.length).sum()
}

/// Copy one sequential read straight into the mapped staging buffer.
pub fn scatter(dest: &mut [u8], file_pos: u64, chunk: &[u8], placements: &[Placement]) {
    let chunk_end = file_pos + chunk.len() as u64;
    for place in placements {
        let place_end = place.file_offset + place.length;
        let start = file_pos.max(place.file_offset);
        let end = chunk_end.min(place_end);
        if start >= end {
            continue;
        }
        let src_off = (start - file_pos) as usize;
        let dst_off = (place.dest_offset + (start - place.file_offset)) as usize;
        let len = (end - start) as usize;
        dest[dst_off..dst_off + len].copy_from_slice(&chunk[src_off..src_off + len]);
    }
}

/// `levels[0]` in the file is the smallest mip. Vulkan mip 0 is the largest,
/// which is the last index entry.
pub fn vulkan_level_order(header: &Ktx2Header) -> Vec<LevelIndex> {
    header.levels.iter().rev().copied().collect()
}

pub fn parse(bytes: &[u8]) -> Option<Ktx2Header> {
    if bytes.len() < HEADER_BYTES || bytes[0..12] != IDENTIFIER {
        return None;
    }
    let vk_format = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    let pixel_width = u32::from_le_bytes(bytes[20..24].try_into().ok()?);
    let pixel_height = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
    let level_count = u32::from_le_bytes(bytes[48..52].try_into().ok()?);
    let supercompression = u32::from_le_bytes(bytes[52..56].try_into().ok()?);
    if supercompression != 0 || level_count == 0 {
        return None;
    }
    let index_bytes = level_count as usize * LEVEL_INDEX_BYTES;
    if bytes.len() < HEADER_BYTES + index_bytes {
        return None;
    }
    let mut levels = Vec::with_capacity(level_count as usize);
    for level in 0..level_count as usize {
        let at = HEADER_BYTES + level * LEVEL_INDEX_BYTES;
        let byte_offset = u64::from_le_bytes(bytes[at..at + 8].try_into().ok()?);
        let byte_length = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().ok()?);
        levels.push(LevelIndex {
            byte_offset,
            byte_length,
        });
    }
    Some(Ktx2Header {
        format: vk::Format::from_raw(vk_format as i32),
        width: pixel_width,
        height: pixel_height,
        levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(format: vk::Format, width: u32, height: u32, levels: &[(u64, u64)]) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_BYTES + levels.len() * LEVEL_INDEX_BYTES];
        bytes[0..12].copy_from_slice(&IDENTIFIER);
        bytes[12..16].copy_from_slice(&(format.as_raw() as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&width.to_le_bytes());
        bytes[24..28].copy_from_slice(&height.to_le_bytes());
        bytes[28..32].copy_from_slice(&1u32.to_le_bytes());
        bytes[48..52].copy_from_slice(&(levels.len() as u32).to_le_bytes());
        for (index, (offset, length)) in levels.iter().enumerate() {
            let at = HEADER_BYTES + index * LEVEL_INDEX_BYTES;
            bytes[at..at + 8].copy_from_slice(&offset.to_le_bytes());
            bytes[at + 8..at + 16].copy_from_slice(&length.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn astc_header_round_trip_puts_base_level_first_for_vulkan() {
        let format = vk::Format::ASTC_4X4_SRGB_BLOCK;
        let bytes = fixture(format, 8, 8, &[(80, 16), (96, 64)]);
        let header = parse(&bytes).unwrap();
        assert_eq!(header.format, format);
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
        let vulkan = vulkan_level_order(&header);
        assert_eq!(vulkan[0].byte_offset, 96);
        assert_eq!(vulkan[0].byte_length, 64);
        assert_eq!(vulkan[1].byte_length, 16);
    }
}
