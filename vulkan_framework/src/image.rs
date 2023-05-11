use ash::vk::{Extent3D, ImageType, SampleCountFlags};

use crate::{
    device::{Device, DeviceOwned},
    instance::{InstanceAPIVersion, InstanceOwned},
    memory_allocator::{AllocationResult, MemoryAllocator},
    memory_heap::MemoryHeapOwned,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

use std::sync::Arc;

#[repr(u32)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum ImageAspect {
    Color = 0x00000001u32,
    Depth = 0x00000002u32,
    Stencil = 0x00000004u32,
    Metadata = 0x00000008u32,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct ImageAspects {
    color: bool,
    depth: bool,
    stencil: bool,
    metadata: bool,
}

impl ImageAspects {
    pub(crate) fn ash_flags(&self) -> ash::vk::ImageAspectFlags {
        (match self.color {
            true => ash::vk::ImageAspectFlags::COLOR,
            false => ash::vk::ImageAspectFlags::empty(),
        }) |
        (match self.depth {
            true => ash::vk::ImageAspectFlags::DEPTH,
            false => ash::vk::ImageAspectFlags::empty(),
        }) |
        (match self.stencil {
            true => ash::vk::ImageAspectFlags::STENCIL,
            false => ash::vk::ImageAspectFlags::empty(),
        }) |
        (match self.metadata {
            true => ash::vk::ImageAspectFlags::METADATA,
            false => ash::vk::ImageAspectFlags::empty(),
        })
    }
    
    pub fn new(
        color: bool,
        depth: bool,
        stencil: bool,
        metadata: bool,
    ) -> Self {
        Self {
            color,
            depth,
            stencil,
            metadata,
        }
    }

    pub fn from(aspects: &[ImageAspect]) -> Self {
        Self {
            color: aspects.contains(&ImageAspect::Color),
            depth: aspects.contains(&ImageAspect::Depth),
            stencil: aspects.contains(&ImageAspect::Stencil),
            metadata: aspects.contains(&ImageAspect::Metadata),
        }
    }

    pub fn all_from_format(format: &ImageFormat) -> Self {
        match format {
            ImageFormat::undefined => todo!(),
            ImageFormat::s8_uint => Self::from(&[]),
            ImageFormat::d16_unorm => Self::from(&[ImageAspect::Depth]),
            ImageFormat::x8_d24_unorm_pack32 => Self::from(&[ImageAspect::Depth]),
            ImageFormat::d32_sfloat => Self::from(&[ImageAspect::Depth]),
            ImageFormat::d32_sfloat_s8_uint => Self::from(&[ImageAspect::Stencil, ImageAspect::Depth]),
            ImageFormat::d16_unorm_s8_uint => Self::from(&[ImageAspect::Stencil, ImageAspect::Depth]),
            ImageFormat::d24_unorm_s8_uint => Self::from(&[ImageAspect::Stencil, ImageAspect::Depth]),
            _ => Self::from(&[ImageAspect::Color])
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum ImageLayout {
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
    Preinitialized = 8,
    Other(u32)
}

impl ImageLayout {
    pub(crate) fn ash_layout(&self) -> ash::vk::ImageLayout {
        ash::vk::ImageLayout::from_raw(match self {
            ImageLayout::Other(fmt) => *fmt,
            fmt => unsafe { std::mem::transmute_copy::<ImageLayout, u32>(fmt) },
        } as i32)
    }
}

pub trait Image1DTrait {
    fn width(&self) -> u32;
}

pub trait Image2DTrait: Image1DTrait {
    fn height(&self) -> u32;
}

pub trait Image3DTrait: Image2DTrait {
    fn depth(&self) -> u32;
}

#[derive(Clone)]
pub struct Image1DDimensions {
    width: u32,
}

impl Image1DDimensions {
    pub fn new(width: u32) -> Self {
        Self { width }
    }
}

impl Image1DTrait for Image1DDimensions {
    fn width(&self) -> u32 {
        self.width
    }
}

#[derive(Clone)]
pub struct Image2DDimensions {
    width: u32,
    height: u32,
}

impl Image2DDimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Image1DTrait for Image2DDimensions {
    fn width(&self) -> u32 {
        self.width
    }
}

impl Image2DTrait for Image2DDimensions {
    fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Clone)]
pub struct Image3DDimensions {
    width: u32,
    height: u32,
    depth: u32,
}

impl Image3DDimensions {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

impl Image1DTrait for Image3DDimensions {
    fn width(&self) -> u32 {
        self.width
    }
}

impl Image2DTrait for Image3DDimensions {
    fn height(&self) -> u32 {
        self.height
    }
}

impl Image3DTrait for Image3DDimensions {
    fn depth(&self) -> u32 {
        self.depth
    }
}

#[derive(Clone)]
pub enum ImageDimensions {
    Image1D { extent: Image1DDimensions },
    Image2D { extent: Image2DDimensions },
    Image3D { extent: Image3DDimensions },
}

#[derive(Clone)]
pub enum ImageMultisampling {
    SamplesPerPixel2,
    SamplesPerPixel4,
    SamplesPerPixel8,
    SamplesPerPixel16,
    SamplesPerPixel32,
    SamplesPerPixel64,
}

/**
 *
 */
#[derive(PartialEq, Eq, Copy, Clone)]
pub struct ImageUsageSpecifier {
    transfer_src: bool,
    transfer_dst: bool,
    sampled: bool,
    storage: bool,
    color_attachment: bool,
    depth_stencil_attachment: bool,
    transient_attachment: bool,
    input_attachment: bool,
}

impl ImageUsageSpecifier {
    pub fn transfer_src(&self) -> bool {
        self.transfer_src
    }

    pub fn transfer_dst(&self) -> bool {
        self.transfer_dst
    }

    pub fn sampled(&self) -> bool {
        self.sampled
    }

    pub fn storage(&self) -> bool {
        self.storage
    }

    pub fn color_attachment(&self) -> bool {
        self.color_attachment
    }

    pub fn depth_stencil_attachment(&self) -> bool {
        self.depth_stencil_attachment
    }

    pub fn transient_attachment(&self) -> bool {
        self.transient_attachment
    }

    pub fn input_attachment(&self) -> bool {
        self.input_attachment
    }

    pub fn new(
        transfer_src: bool,
        transfer_dst: bool,
        sampled: bool,
        storage: bool,
        color_attachment: bool,
        depth_stencil_attachment: bool,
        transient_attachment: bool,
        input_attachment: bool,
    ) -> Self {
        Self {
            transfer_src,
            transfer_dst,
            sampled,
            storage,
            color_attachment,
            depth_stencil_attachment,
            transient_attachment,
            input_attachment,
        }
    }
}

#[derive(Clone)]
pub enum ImageUsage {
    Managed(ImageUsageSpecifier),
    Unmanaged(u32),
}

impl ImageUsage {
    pub(crate) fn ash_usage(&self) -> ash::vk::ImageUsageFlags {
        match &self {
            ImageUsage::Managed(flags) => {
                let raw_flags = (match flags.transfer_src() {
                    true => {
                        ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::TRANSFER_SRC)
                    }
                    false => 0x00000000u32,
                }) | (match flags.transfer_dst() {
                    true => {
                        ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::TRANSFER_DST)
                    }
                    false => 0x00000000u32,
                }) | (match flags.sampled() {
                    true => ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::SAMPLED),
                    false => 0x00000000u32,
                }) | (match flags.storage() {
                    true => ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::STORAGE),
                    false => 0x00000000u32,
                }) | (match flags.color_attachment() {
                    true => {
                        ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    }
                    false => 0x00000000u32,
                }) | (match flags.depth_stencil_attachment() {
                    true => ash::vk::ImageUsageFlags::as_raw(
                        ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    ),
                    false => 0x00000000u32,
                }) | (match flags.transient_attachment() {
                    true => ash::vk::ImageUsageFlags::as_raw(
                        ash::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                    ),
                    false => 0x00000000u32,
                }) | (match flags.input_attachment() {
                    true => {
                        ash::vk::ImageUsageFlags::as_raw(ash::vk::ImageUsageFlags::INPUT_ATTACHMENT)
                    }
                    false => 0x00000000u32,
                });

                ash::vk::ImageUsageFlags::from_raw(raw_flags)
            }
            ImageUsage::Unmanaged(raw_flags) => ash::vk::ImageUsageFlags::from_raw(*raw_flags),
        }
    }
}

/**
 * Specify the image format for an image.
 *
 * Common image formats has a name for convenience,
 * but it is always possible to specify Other(VkFormat).
 */
#[allow(non_camel_case_types)]
#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(i32)]
pub enum ImageFormat {
    undefined = 0,
    r4g4_unorm_pack8 = 1,
    r4g4b4a4_unorm_pack16 = 2,
    b4g4r4a4_unorm_pack16 = 3,
    r5g6b5_unorm_pack16 = 4,
    b5g6r5_unorm_pack16 = 5,
    r5g5b5a1_unorm_pack16 = 6,
    b5g5r5a1_unorm_pack16 = 7,
    a1r5g5b5_unorm_pack16 = 8,
    r8_unorm = 9,
    r8_snorm = 10,
    r8_uscaled = 11,
    r8_sscaled = 12,
    r8_uint = 13,
    r8_sint = 14,
    r8_srgb = 15,
    r8g8_unorm = 16,
    r8g8_snorm = 17,
    r8g8_uscaled = 18,
    r8g8_sscaled = 19,
    r8g8_uint = 20,
    r8g8_sint = 21,
    r8g8_srgb = 22,
    r8g8b8_unorm = 23,
    r8g8b8_snorm = 24,
    r8g8b8_uscaled = 25,
    r8g8b8_sscaled = 26,
    r8g8b8_uint = 27,
    r8g8b8_sint = 28,
    r8g8b8_srgb = 29,
    b8g8r8_unorm = 30,
    b8g8r8_snorm = 31,
    b8g8r8_uscaled = 32,
    b8g8r8_sscaled = 33,
    b8g8r8_uint = 34,
    b8g8r8_sint = 35,
    b8g8r8_srgb = 36,
    r8g8b8a8_unorm = 37,
    r8g8b8a8_snorm = 38,
    r8g8b8a8_uscaled = 39,
    r8g8b8a8_sscaled = 40,
    r8g8b8a8_uint = 41,
    r8g8b8a8_sint = 42,
    r8g8b8a8_srgb = 43,
    b8g8r8a8_unorm = 44,
    b8g8r8a8_snorm = 45,
    b8g8r8a8_uscaled = 46,
    b8g8r8a8_sscaled = 47,
    b8g8r8a8_uint = 48,
    b8g8r8a8_sint = 49,
    b8g8r8a8_srgb = 50,
    a8b8g8r8_unorm_pack32 = 51,
    a8b8g8r8_snorm_pack32 = 52,
    a8b8g8r8_uscaled_pack32 = 53,
    a8b8g8r8_sscaled_pack32 = 54,
    a8b8g8r8_uint_pack32 = 55,
    a8b8g8r8_sint_pack32 = 56,
    a8b8g8r8_srgb_pack32 = 57,
    a2r10g10b10_unorm_pack32 = 58,
    a2r10g10b10_snorm_pack32 = 59,
    a2r10g10b10_uscaled_pack32 = 60,
    a2r10g10b10_sscaled_pack32 = 61,
    a2r10g10b10_uint_pack32 = 62,
    a2r10g10b10_sint_pack32 = 63,
    a2b10g10r10_unorm_pack32 = 64,
    a2b10g10r10_snorm_pack32 = 65,
    a2b10g10r10_uscaled_pack32 = 66,
    a2b10g10r10_sscaled_pack32 = 67,
    a2b10g10r10_uint_pack32 = 68,
    a2b10g10r10_sint_pack32 = 69,
    r16_unorm = 70,
    r16_snorm = 71,
    r16_uscaled = 72,
    r16_sscaled = 73,
    r16_uint = 74,
    r16_sint = 75,
    r16_sfloat = 76,
    r16g16_unorm = 77,
    r16g16_snorm = 78,
    r16g16_uscaled = 79,
    r16g16_sscaled = 80,
    r16g16_uint = 81,
    r16g16_sint = 82,
    r16g16_sfloat = 83,
    r16g16b16_unorm = 84,
    r16g16b16_snorm = 85,
    r16g16b16_uscaled = 86,
    r16g16b16_sscaled = 87,
    r16g16b16_uint = 88,
    r16g16b16_sint = 89,
    r16g16b16_sfloat = 90,
    r16g16b16a16_unorm = 91,
    r16g16b16a16_snorm = 92,
    r16g16b16a16_uscaled = 93,
    r16g16b16a16_sscaled = 94,
    r16g16b16a16_uint = 95,
    r16g16b16a16_sint = 96,
    r16g16b16a16_sfloat = 97,
    r32_uint = 98,
    r32_sint = 99,
    r32_sfloat = 100,
    r32g32_uint = 101,
    r32g32_sint = 102,
    r32g32_sfloat = 103,
    r32g32b32_uint = 104,
    r32g32b32_sint = 105,
    r32g32b32_sfloat = 106,
    r32g32b32a32_uint = 107,
    r32g32b32a32_sint = 108,
    r32g32b32a32_sfloat = 109,
    r64_uint = 110,
    r64_sint = 111,
    r64_sfloat = 112,
    r64g64_uint = 113,
    r64g64_sint = 114,
    r64g64_sfloat = 115,
    r64g64b64_uint = 116,
    r64g64b64_sint = 117,
    r64g64b64_sfloat = 118,
    r64g64b64a64_uint = 119,
    r64g64b64a64_sint = 120,
    r64g64b64a64_sfloat = 121,
    b10g11r11_ufloat_pack32 = 122,
    e5b9g9r9_ufloat_pack32 = 123,
    d16_unorm = 124,
    x8_d24_unorm_pack32 = 125,
    d32_sfloat = 126,
    s8_uint = 127,
    d16_unorm_s8_uint = 128,
    d24_unorm_s8_uint = 129,
    d32_sfloat_s8_uint = 130,
    bc1_rgb_unorm_block = 131,
    bc1_rgb_srgb_block = 132,
    bc1_rgba_unorm_block = 133,
    bc1_rgba_srgb_block = 134,
    bc2_unorm_block = 135,
    bc2_srgb_block = 136,
    bc3_unorm_block = 137,
    bc3_srgb_block = 138,
    bc4_unorm_block = 139,
    bc4_snorm_block = 140,
    bc5_unorm_block = 141,
    bc5_snorm_block = 142,
    bc6h_ufloat_block = 143,
    bc6h_sfloat_block = 144,
    bc7_unorm_block = 145,
    bc7_srgb_block = 146,
    etc2_r8g8b8_unorm_block = 147,
    etc2_r8g8b8_srgb_block = 148,
    etc2_r8g8b8a1_unorm_block = 149,
    etc2_r8g8b8a1_srgb_block = 150,
    etc2_r8g8b8a8_unorm_block = 151,
    etc2_r8g8b8a8_srgb_block = 152,
    eac_r11_unorm_block = 153,
    eac_r11_snorm_block = 154,
    eac_r11g11_unorm_block = 155,
    eac_r11g11_snorm_block = 156,
    astc_4x4_unorm_block = 157,
    astc_4x4_srgb_block = 158,
    astc_5x4_unorm_block = 159,
    astc_5x4_srgb_block = 160,
    astc_5x5_unorm_block = 161,
    astc_5x5_srgb_block = 162,
    astc_6x5_unorm_block = 163,
    astc_6x5_srgb_block = 164,
    astc_6x6_unorm_block = 165,
    astc_6x6_srgb_block = 166,
    astc_8x5_unorm_block = 167,
    astc_8x5_srgb_block = 168,
    astc_8x6_unorm_block = 169,
    astc_8x6_srgb_block = 170,
    astc_8x8_unorm_block = 171,
    astc_8x8_srgb_block = 172,
    astc_10x5_unorm_block = 173,
    astc_10x5_srgb_block = 174,
    astc_10x6_unorm_block = 175,
    astc_10x6_srgb_block = 176,
    astc_10x8_unorm_block = 177,
    astc_10x8_srgb_block = 178,
    astc_10x10_unorm_block = 179,
    astc_10x10_srgb_block = 180,
    astc_12x10_unorm_block = 181,
    astc_12x10_srgb_block = 182,
    astc_12x12_unorm_block = 183,
    astc_12x12_srgb_block = 184,
    Other(u32),
}

impl ImageFormat {
    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        ash::vk::Format::from_raw(match self {
            ImageFormat::Other(fmt) => *fmt,
            fmt => unsafe { std::mem::transmute_copy::<ImageFormat, u32>(fmt) },
        } as i32)
    }
}

#[derive(Clone)]
pub struct ImageFlagsCollection {
    mutable_format: bool,
    cube_compatible: bool,
}

impl ImageFlagsCollection {
    pub(crate) fn ash_flags(&self) -> ash::vk::ImageCreateFlags {
        (match self.mutable_format {
            true => ash::vk::ImageCreateFlags::MUTABLE_FORMAT,
            false => ash::vk::ImageCreateFlags::from_raw(0),
        }) | (match self.cube_compatible {
            true => ash::vk::ImageCreateFlags::CUBE_COMPATIBLE,
            false => ash::vk::ImageCreateFlags::from_raw(0),
        })
    }

    pub fn new(mutable_format: bool, cube_compatible: bool) -> Self {
        Self {
            mutable_format,
            cube_compatible,
        }
    }
}

#[derive(Clone)]
pub enum ImageFlags {
    Recognised(ImageFlagsCollection),
    Other(u32),
    Empty,
}

impl ImageFlags {
    pub(crate) fn ash_flags(&self) -> ash::vk::ImageCreateFlags {
        match self {
            Self::Other(raw) => ash::vk::ImageCreateFlags::from_raw(*raw),
            Self::Recognised(r) => r.ash_flags(),
            Self::Empty => ash::vk::ImageCreateFlags::from_raw(0),
        }
    }

    pub fn empty() -> Self {
        Self::Empty
    }

    pub fn from(flags: ImageFlagsCollection) -> Self {
        Self::Recognised(flags)
    }

    pub fn from_raw(flags: u32) -> Self {
        Self::Other(flags)
    }
}

#[derive(Clone)]
pub enum ImageTiling {
    Optimal,
    Linear,
    Other(i32),
}

impl ImageTiling {
    pub(crate) fn ash_tiling(&self) -> ash::vk::ImageTiling {
        match self {
            Self::Optimal => ash::vk::ImageTiling::OPTIMAL,
            Self::Linear => ash::vk::ImageTiling::LINEAR,
            Self::Other(raw) => ash::vk::ImageTiling::from_raw(*raw),
        }
    }
}

#[derive(Clone)]
pub struct ConcreteImageDescriptor {
    img_dimensions: ImageDimensions,
    img_usage: ImageUsage,
    img_multisampling: Option<ImageMultisampling>,
    img_layers: u32,
    img_mip_levels: u32,
    img_format: ImageFormat,
    img_flags: ImageFlags,
    img_tiling: ImageTiling,
}

impl ConcreteImageDescriptor {
    pub(crate) fn ash_tiling(&self) -> ash::vk::ImageTiling {
        self.img_tiling.ash_tiling()
    }

    pub(crate) fn ash_flags(&self) -> ash::vk::ImageCreateFlags {
        self.img_flags.ash_flags()
    }

    pub(crate) fn ash_usage(&self) -> ash::vk::ImageUsageFlags {
        self.img_usage.ash_usage()
    }

    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        self.img_format.ash_format()
    }

    pub(crate) fn ash_image_type(&self) -> ImageType {
        match &self.img_dimensions {
            ImageDimensions::Image1D { extent: _ } => ImageType::TYPE_1D,
            ImageDimensions::Image2D { extent: _ } => ImageType::TYPE_2D,
            ImageDimensions::Image3D { extent: _ } => ImageType::TYPE_3D,
        }
    }

    pub(crate) fn ash_sample_count(&self) -> SampleCountFlags {
        match &self.img_multisampling.clone() {
            Some(ms) => match ms {
                ImageMultisampling::SamplesPerPixel2 => SampleCountFlags::from_raw(0x00000002u32),
                ImageMultisampling::SamplesPerPixel4 => SampleCountFlags::from_raw(0x00000004u32),
                ImageMultisampling::SamplesPerPixel8 => SampleCountFlags::from_raw(0x00000008u32),
                ImageMultisampling::SamplesPerPixel16 => SampleCountFlags::from_raw(0x00000010u32),
                ImageMultisampling::SamplesPerPixel32 => SampleCountFlags::from_raw(0x00000020u32),
                ImageMultisampling::SamplesPerPixel64 => SampleCountFlags::from_raw(0x00000040u32),
            },
            None => SampleCountFlags::from_raw(0x00000001u32),
        }
    }

    pub(crate) fn ash_extent_3d(&self) -> Extent3D {
        match &self.img_dimensions {
            ImageDimensions::Image1D { extent } => Extent3D {
                width: extent.width(),
                height: 1,
                depth: 1,
            },
            ImageDimensions::Image2D { extent } => Extent3D {
                width: extent.width(),
                height: extent.height(),
                depth: 1,
            },
            ImageDimensions::Image3D { extent } => Extent3D {
                width: extent.width(),
                height: extent.height(),
                depth: extent.depth(),
            },
        }
    }

    pub fn new(
        img_dimensions: ImageDimensions,
        img_usage: ImageUsage,
        img_multisampling: Option<ImageMultisampling>,
        img_layers: u32,
        img_mip_levels: u32,
        img_format: ImageFormat,
        img_flags: ImageFlags,
        img_tiling: ImageTiling,
    ) -> Self {
        Self {
            img_dimensions,
            img_usage,
            img_multisampling,
            img_layers,
            img_mip_levels,
            img_format,
            img_flags,
            img_tiling,
        }
    }
}

pub trait ImageTrait: Send + Sync + DeviceOwned {
    fn native_handle(&self) -> u64;

    fn format(&self) -> ImageFormat;

    fn dimensions(&self) -> ImageDimensions;

    fn layers_count(&self) -> u32;

    fn mip_levels_count(&self) -> u32;
}

pub(crate) trait ImageOwned {
    fn get_parent_image(&self) -> Arc<dyn ImageTrait>;
}

pub struct Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    memory_pool: Arc<MemoryPool<Allocator>>,
    reserved_memory_from_pool: AllocationResult,
    image: ash::vk::Image,
    descriptor: ConcreteImageDescriptor,
}

impl<Allocator> DeviceOwned for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn get_parent_device(&self) -> Arc<Device> {
        self.get_backing_memory_pool()
            .get_parent_memory_heap()
            .get_parent_device()
    }
}

impl<Allocator> ImageTrait for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image)
    }

    fn format(&self) -> ImageFormat {
        self.descriptor.img_format
    }

    fn dimensions(&self) -> ImageDimensions {
        self.descriptor.img_dimensions.clone()
    }

    fn layers_count(&self) -> u32 {
        self.descriptor.img_layers
    }

    fn mip_levels_count(&self) -> u32 {
        self.descriptor.img_mip_levels
    }
}

impl<Allocator> Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    pub(crate) fn ash_native(&self) -> ash::vk::Image {
        self.image
    }

    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        self.descriptor.ash_format()
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image)
    }

    /**
     * Create a new Image from the provided memory pool.
     *
     * The creation process is expected to fail if the memory pool does not allow allocation for such resource,
     * either because the allocation fails (for example due to lack of memory) or because such image
     * has memory requirements that cannot be met.
     *
     * The resulting image has the initial layout of VK_IMAGE_LAYOUT_UNDEFINED.
     *
     *
     */
    pub fn new(
        memory_pool: Arc<MemoryPool<Allocator>>,
        descriptor: ConcreteImageDescriptor,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        if descriptor.img_layers == 0 {
            return Err(VulkanError::Unspecified);
        }

        if descriptor.img_mip_levels == 0 {
            return Err(VulkanError::Unspecified);
        }

        let mut queue_family_indices = Vec::<u32>::new();
        if let Some(weak_queue_family_iter) = sharing {
            for allowed_queue_family in weak_queue_family_iter {
                if let Some(queue_family) = allowed_queue_family.upgrade() {
                    let family_index = queue_family.get_family_index();
                    if !queue_family_indices.contains(&family_index) {
                        queue_family_indices.push(family_index);
                    }
                }
            }
        }

        let create_info = ash::vk::ImageCreateInfo::builder()
            .flags(descriptor.ash_flags())
            .image_type(descriptor.ash_image_type())
            .extent(descriptor.ash_extent_3d())
            .samples(descriptor.ash_sample_count())
            .mip_levels(descriptor.img_mip_levels)
            .array_layers(descriptor.img_layers)
            .initial_layout(ash::vk::ImageLayout::UNDEFINED)
            .format(descriptor.ash_format())
            .usage(descriptor.ash_usage())
            .sharing_mode(match queue_family_indices.len() <= 1 {
                true => ash::vk::SharingMode::EXCLUSIVE,
                false => ash::vk::SharingMode::CONCURRENT,
            })
            .queue_family_indices(queue_family_indices.as_ref())
            .tiling(descriptor.ash_tiling())
            .build();

        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        let image = unsafe {
            match device.ash_handle().create_image(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            ) {
                Ok(image) => image,
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        panic!("Error creating the image: {}", err)
                    }

                    return Err(VulkanError::Unspecified);
                }
            }
        };

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.get_parent_instance().get_debug_ext_extension() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                    // set device name for debugging
                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                        .object_type(ash::vk::ObjectType::IMAGE)
                        .object_handle(ash::vk::Handle::as_raw(image))
                        .object_name(object_name)
                        .build();

                    match ext.set_debug_utils_object_name(device.ash_handle().handle(), &dbg_info) {
                        Ok(_) => {
                            #[cfg(debug_assertions)]
                            {
                                println!("Queue Debug object name changed");
                            }
                        }
                        Err(err) => {
                            #[cfg(debug_assertions)]
                            {
                                panic!("Error setting the Debug name for the newly created Image, will use handle. Error: {}", err)
                            }
                        }
                    }
                }
            }
        }

        unsafe {
            let requirements = if device.get_parent_instance().instance_vulkan_version()
                == InstanceAPIVersion::Version1_0
            {
                device.ash_handle().get_image_memory_requirements(image)
            } else {
                let requirements_info = ash::vk::ImageMemoryRequirementsInfo2::builder()
                    .image(image)
                    .build();

                let mut requirements = ash::vk::MemoryRequirements2::default();

                device
                    .ash_handle()
                    .get_image_memory_requirements2(&requirements_info, &mut requirements);

                requirements.memory_requirements
            };

            match memory_pool.alloc(requirements) {
                Some(reserved_memory_from_pool) => {
                    match device.ash_handle().bind_image_memory(
                        image,
                        memory_pool.ash_handle(),
                        reserved_memory_from_pool.offset_in_pool(),
                    ) {
                        Ok(_) => Ok(Arc::new(Self {
                            memory_pool,
                            reserved_memory_from_pool,
                            image,
                            descriptor,
                        })),
                        Err(err) => {
                            // the image will not let this function, destroy it or it will leak
                            device.ash_handle().destroy_image(
                                image,
                                device.get_parent_instance().get_alloc_callbacks(),
                            );

                            #[cfg(debug_assertions)]
                            {
                                panic!("Error allocating memory on the device: {}, probably this is due to an incorrect implementation of the memory allocation algorithm", err)
                            }

                            Err(VulkanError::Unspecified)
                        }
                    }
                }
                None => {
                    // the image will not let this function, destroy it or it will leak
                    device
                        .ash_handle()
                        .destroy_image(image, device.get_parent_instance().get_alloc_callbacks());

                    Err(VulkanError::Unspecified)
                }
            }
        }
    }
}

impl<Allocator> Drop for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn drop(&mut self) {
        let device = self
            .memory_pool
            .get_parent_memory_heap()
            .get_parent_device();

        unsafe {
            device.ash_handle().destroy_image(
                self.image,
                device.get_parent_instance().get_alloc_callbacks(),
            );
        }

        self.memory_pool
            .dealloc(&mut self.reserved_memory_from_pool)
    }
}

impl<Allocator> MemoryPoolBacked<Allocator> for Image<Allocator>
where
    Allocator: MemoryAllocator + Send + Sync,
{
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool<Allocator>> {
        self.memory_pool.clone()
    }

    fn allocation_offset(&self) -> u64 {
        self.reserved_memory_from_pool.offset_in_pool()
    }

    fn allocation_size(&self) -> u64 {
        self.reserved_memory_from_pool.size()
    }
}
