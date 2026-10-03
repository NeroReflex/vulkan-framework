//! Byte layout of a compressed mip chain inside a GPU-visible staging buffer.
//!
//! The loader writes these bytes straight into a mapped device buffer. The
//! upload then copies each level with a buffer offset. There is no second
//! host copy of the image.

use vulkan_framework::ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFootprint {
    pub block_width: u32,
    pub block_height: u32,
    pub bytes_per_block: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLevel {
    pub offset: u64,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
}

pub fn block_footprint(format: vk::Format) -> Option<BlockFootprint> {
    let (block_width, block_height, bytes_per_block) = match format {
        vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK
        | vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK => (4, 4, 8),
        vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK => (4, 4, 16),
        vk::Format::ASTC_4X4_UNORM_BLOCK | vk::Format::ASTC_4X4_SRGB_BLOCK => (4, 4, 16),
        vk::Format::ASTC_5X4_UNORM_BLOCK | vk::Format::ASTC_5X4_SRGB_BLOCK => (5, 4, 16),
        vk::Format::ASTC_5X5_UNORM_BLOCK | vk::Format::ASTC_5X5_SRGB_BLOCK => (5, 5, 16),
        vk::Format::ASTC_6X5_UNORM_BLOCK | vk::Format::ASTC_6X5_SRGB_BLOCK => (6, 5, 16),
        vk::Format::ASTC_6X6_UNORM_BLOCK | vk::Format::ASTC_6X6_SRGB_BLOCK => (6, 6, 16),
        vk::Format::ASTC_8X5_UNORM_BLOCK | vk::Format::ASTC_8X5_SRGB_BLOCK => (8, 5, 16),
        vk::Format::ASTC_8X6_UNORM_BLOCK | vk::Format::ASTC_8X6_SRGB_BLOCK => (8, 6, 16),
        vk::Format::ASTC_8X8_UNORM_BLOCK | vk::Format::ASTC_8X8_SRGB_BLOCK => (8, 8, 16),
        vk::Format::ASTC_10X5_UNORM_BLOCK | vk::Format::ASTC_10X5_SRGB_BLOCK => (10, 5, 16),
        vk::Format::ASTC_10X6_UNORM_BLOCK | vk::Format::ASTC_10X6_SRGB_BLOCK => (10, 6, 16),
        vk::Format::ASTC_10X8_UNORM_BLOCK | vk::Format::ASTC_10X8_SRGB_BLOCK => (10, 8, 16),
        vk::Format::ASTC_10X10_UNORM_BLOCK | vk::Format::ASTC_10X10_SRGB_BLOCK => (10, 10, 16),
        vk::Format::ASTC_12X10_UNORM_BLOCK | vk::Format::ASTC_12X10_SRGB_BLOCK => (12, 10, 16),
        vk::Format::ASTC_12X12_UNORM_BLOCK | vk::Format::ASTC_12X12_SRGB_BLOCK => (12, 12, 16),
        _ => return None,
    };
    Some(BlockFootprint {
        block_width,
        block_height,
        bytes_per_block,
    })
}

pub fn level_byte_size(format: vk::Format, width: u32, height: u32) -> Option<u64> {
    let footprint = block_footprint(format)?;
    let blocks_x = (width.max(1) + footprint.block_width - 1) / footprint.block_width;
    let blocks_y = (height.max(1) + footprint.block_height - 1) / footprint.block_height;
    Some(blocks_x as u64 * blocks_y as u64 * footprint.bytes_per_block as u64)
}

/// Tightly packed levels, largest first. This is the DDS payload order and the
/// order the mapped staging buffer uses before `copy_buffer_to_image`.
pub fn mip_chain(format: vk::Format, width: u32, height: u32, levels: u32) -> Option<Vec<MipLevel>> {
    if levels == 0 {
        return None;
    }
    let mut chain = Vec::with_capacity(levels as usize);
    let mut offset = 0u64;
    let mut level_width = width.max(1);
    let mut level_height = height.max(1);
    for _ in 0..levels {
        let bytes = level_byte_size(format, level_width, level_height)?;
        chain.push(MipLevel {
            offset,
            width: level_width,
            height: level_height,
            bytes,
        });
        offset += bytes;
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    Some(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc7_two_level_chain_matches_direct_buffer_sizes() {
        let chain = mip_chain(vk::Format::BC7_SRGB_BLOCK, 8, 8, 2).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].offset, 0);
        assert_eq!(chain[0].bytes, 64);
        assert_eq!(chain[0].width, 8);
        assert_eq!(chain[1].offset, 64);
        assert_eq!(chain[1].bytes, 16);
        assert_eq!(chain[1].width, 4);
        assert_eq!(chain[0].bytes + chain[1].bytes, 80);
    }

    #[test]
    fn astc_4x4_block_is_sixteen_bytes() {
        let footprint = block_footprint(vk::Format::ASTC_4X4_SRGB_BLOCK).unwrap();
        assert_eq!(footprint.bytes_per_block, 16);
        assert_eq!(level_byte_size(vk::Format::ASTC_4X4_SRGB_BLOCK, 4, 4), Some(16));
    }
}
