use ash::vk::{Extent3D, Handle, ImageType};

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_allocator::SuccessfulAllocation,
    memory_heap::MemoryHeapOwned,
    memory_management::UnallocatedResource,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    memory_requiring::{AllocationRequirements, AllocationRequiring},
    prelude::{FrameworkError, VulkanError, VulkanResult},
    queue_family::QueueFamily,
};

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Represents commonly used image formats
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[repr(i32)]
pub enum CommonImageFormat {
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ImageAspect {
    Color,
    Depth,
    Stencil,
    Metadata,
}

impl From<&ImageAspect> for ash::vk::ImageAspectFlags {
    fn from(val: &ImageAspect) -> Self {
        type FlagType = ash::vk::ImageAspectFlags;
        match val {
            ImageAspect::Color => FlagType::COLOR,
            ImageAspect::Depth => FlagType::DEPTH,
            ImageAspect::Stencil => FlagType::STENCIL,
            ImageAspect::Metadata => FlagType::METADATA,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[repr(transparent)]
pub struct ImageAspects(ash::vk::ImageAspectFlags);

impl From<u32> for ImageAspects {
    fn from(value: u32) -> Self {
        Self(ash::vk::ImageAspectFlags::from_raw(value))
    }
}

impl From<crate::ash::vk::ImageAspectFlags> for ImageAspects {
    fn from(usage: crate::ash::vk::ImageAspectFlags) -> Self {
        Self(usage)
    }
}

impl From<ImageAspects> for crate::ash::vk::ImageAspectFlags {
    fn from(val: ImageAspects) -> Self {
        val.0.to_owned()
    }
}

impl From<&[ImageAspect]> for ImageAspects {
    fn from(value: &[ImageAspect]) -> Self {
        let mut usage = ash::vk::ImageAspectFlags::empty();
        for flag in value.iter() {
            usage |= flag.into()
        }

        Self(usage)
    }
}

impl From<ImageFormat> for ImageAspects {
    fn from(value: ImageFormat) -> Self {
        type FormatType = crate::ash::vk::Format;
        match value.into() {
            FormatType::UNDEFINED => panic!("Undefined image format is not allowed"),
            FormatType::S8_UINT => Self::from([].as_ref()),
            FormatType::D16_UNORM => Self::from([ImageAspect::Depth].as_ref()),
            FormatType::X8_D24_UNORM_PACK32 => Self::from([ImageAspect::Depth].as_ref()),
            FormatType::D32_SFLOAT => Self::from([ImageAspect::Depth].as_ref()),
            FormatType::D32_SFLOAT_S8_UINT => {
                Self::from([ImageAspect::Stencil, ImageAspect::Depth].as_ref())
            }
            FormatType::D16_UNORM_S8_UINT => {
                Self::from([ImageAspect::Stencil, ImageAspect::Depth].as_ref())
            }
            FormatType::D24_UNORM_S8_UINT => {
                Self::from([ImageAspect::Stencil, ImageAspect::Depth].as_ref())
            }
            FormatType::R32G32B32A32_SFLOAT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::R32G32B32A32_SINT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::R32G32B32A32_UINT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::R32G32B32_SFLOAT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::R32G32B32_SINT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::R32G32B32_UINT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::B8G8R8A8_SRGB => Self::from([ImageAspect::Color].as_ref()),
            FormatType::B8G8R8A8_UINT => Self::from([ImageAspect::Color].as_ref()),
            FormatType::B8G8R8A8_UNORM => Self::from([ImageAspect::Color].as_ref()),
            FormatType::BC7_SRGB_BLOCK => Self::from([ImageAspect::Color].as_ref()),
            fmt => {
                println!(
                    "Check available image aspects for this format: {}",
                    fmt.as_raw()
                );
                Self::from([ImageAspect::Color].as_ref())
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ImageSubresourceLayers {
    image_aspect: ImageAspects,
    mip_level: u32,
    base_array_layer: u32,
    array_layers_count: u32,
}

impl ImageSubresourceLayers {
    pub(crate) fn ash_subresource_layers(&self) -> ash::vk::ImageSubresourceLayers {
        ash::vk::ImageSubresourceLayers::default()
            .aspect_mask(self.image_aspect.into())
            .base_array_layer(self.base_array_layer)
            .layer_count(self.array_layers_count)
            .mip_level(self.mip_level)
    }

    pub fn new(
        image_aspect: ImageAspects,
        mip_level: u32,
        base_array_layer: u32,
        array_layers_count: u32,
    ) -> Self {
        Self {
            image_aspect,
            mip_level,
            base_array_layer,
            array_layers_count,
        }
    }
}

#[derive(Clone)]
pub struct ImageSubresourceRange {
    image: Arc<dyn ImageTrait>,
    image_aspects: ImageAspects,
    base_mip_level: u32,
    mip_levels_count: u32,
    base_array_layer: u32,
    array_layers_count: u32,
}

impl From<Arc<dyn ImageTrait>> for ImageSubresourceRange {
    fn from(image: Arc<dyn ImageTrait>) -> Self {
        Self::new(
            image.clone(),
            ImageAspects::from(image.as_ref().format()),
            0,
            image.mip_levels_count(),
            0,
            image.layers_count(),
        )
    }
}

impl ImageSubresourceRange {
    #[inline]
    pub fn image(&self) -> Arc<dyn ImageTrait> {
        self.image.clone()
    }

    #[inline]
    pub fn new(
        image: Arc<dyn ImageTrait>,
        image_aspects: ImageAspects,
        base_mip_level: u32,
        mip_levels_count: u32,
        base_array_layer: u32,
        array_layers_count: u32,
    ) -> Self {
        assert!(base_mip_level + mip_levels_count <= image.mip_levels_count());
        assert!(base_array_layer + array_layers_count <= image.layers_count());

        Self {
            image,
            image_aspects,
            base_mip_level,
            mip_levels_count,
            base_array_layer,
            array_layers_count,
        }
    }

    pub(crate) fn ash_subresource_range(&self) -> ash::vk::ImageSubresourceRange {
        ash::vk::ImageSubresourceRange::default()
            .aspect_mask(self.image_aspects.into())
            .base_array_layer(self.base_array_layer)
            .layer_count(self.array_layers_count)
            .base_mip_level(self.base_mip_level)
            .level_count(self.mip_levels_count)
    }
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ImageLayoutSwapchainKHR {
    PresentSrc = 1000001002u32,
}

impl From<ImageLayoutSwapchainKHR> for u32 {
    fn from(val: ImageLayoutSwapchainKHR) -> Self {
        (&val).into()
    }
}

impl From<&ImageLayoutSwapchainKHR> for u32 {
    fn from(val: &ImageLayoutSwapchainKHR) -> Self {
        match &val {
            fmt => unsafe { std::mem::transmute_copy::<ImageLayoutSwapchainKHR, u32>(fmt) },
        }
    }
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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
    SwapchainKHR(ImageLayoutSwapchainKHR),
    Other(u32),
}

impl From<&ImageLayout> for ash::vk::ImageLayout {
    fn from(val: &ImageLayout) -> Self {
        ash::vk::ImageLayout::from_raw(match val {
            ImageLayout::Other(fmt) => *fmt,
            ImageLayout::SwapchainKHR(swapchain_fmt) => swapchain_fmt.into(),
            fmt => unsafe { std::mem::transmute_copy::<ImageLayout, u32>(fmt) },
        } as i32)
    }
}

impl From<ImageLayout> for ash::vk::ImageLayout {
    fn from(val: ImageLayout) -> Self {
        ash::vk::ImageLayout::from_raw(match &val {
            ImageLayout::Other(fmt) => *fmt,
            ImageLayout::SwapchainKHR(swapchain_fmt) => swapchain_fmt.into(),
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Image2DDimensions {
    width: u32,
    height: u32,
}

impl Image2DDimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl TryFrom<ImageDimensions> for Image2DDimensions {
    type Error = VulkanError;

    fn try_from(value: ImageDimensions) -> Result<Self, Self::Error> {
        match &value {
            ImageDimensions::Image1D { extent: _ } => Err(VulkanError::Framework(
                FrameworkError::UnsuitableImageDimensions,
            )),
            ImageDimensions::Image2D { extent } => {
                Ok(Image2DDimensions::new(extent.width(), extent.height()))
            }
            ImageDimensions::Image3D { extent: _ } => Err(VulkanError::Framework(
                FrameworkError::UnsuitableImageDimensions,
            )),
        }
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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Image3DDimensions {
    width: u32,
    height: u32,
    depth: u32,
}

impl From<Image2DDimensions> for Image3DDimensions {
    fn from(value: Image2DDimensions) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: 1,
        }
    }
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageDimensions {
    Image1D { extent: Image1DDimensions },
    Image2D { extent: Image2DDimensions },
    Image3D { extent: Image3DDimensions },
}

impl From<Image1DDimensions> for ImageDimensions {
    fn from(extent: Image1DDimensions) -> Self {
        Self::Image1D { extent }
    }
}

impl From<Image2DDimensions> for ImageDimensions {
    fn from(extent: Image2DDimensions) -> Self {
        Self::Image2D { extent }
    }
}

impl From<Image3DDimensions> for ImageDimensions {
    fn from(extent: Image3DDimensions) -> Self {
        Self::Image3D { extent }
    }
}

impl ImageDimensions {
    pub(crate) fn ash_extent_3d(&self) -> Extent3D {
        match &self {
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
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageMultisampling {
    SamplesPerPixel1,
    SamplesPerPixel2,
    SamplesPerPixel4,
    SamplesPerPixel8,
    SamplesPerPixel16,
    SamplesPerPixel32,
    SamplesPerPixel64,
}

impl ImageMultisampling {
    pub(crate) fn ash_samples(&self) -> ash::vk::SampleCountFlags {
        match self {
            ImageMultisampling::SamplesPerPixel1 => ash::vk::SampleCountFlags::TYPE_1,
            ImageMultisampling::SamplesPerPixel2 => ash::vk::SampleCountFlags::TYPE_2,
            ImageMultisampling::SamplesPerPixel4 => ash::vk::SampleCountFlags::TYPE_4,
            ImageMultisampling::SamplesPerPixel8 => ash::vk::SampleCountFlags::TYPE_8,
            ImageMultisampling::SamplesPerPixel16 => ash::vk::SampleCountFlags::TYPE_16,
            ImageMultisampling::SamplesPerPixel32 => ash::vk::SampleCountFlags::TYPE_32,
            ImageMultisampling::SamplesPerPixel64 => ash::vk::SampleCountFlags::TYPE_64,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageUseAs {
    TransferSrc,
    TransferDst,
    Sampled,
    Storage,
    ColorAttachment,
    DepthStencilAttachment,
    TransientAttachment,
    InputAttachment,
}

impl From<&ImageUseAs> for ash::vk::ImageUsageFlags {
    fn from(val: &ImageUseAs) -> Self {
        type FlagType = ash::vk::ImageUsageFlags;
        match val {
            ImageUseAs::TransferSrc => FlagType::TRANSFER_SRC,
            ImageUseAs::TransferDst => FlagType::TRANSFER_DST,
            ImageUseAs::Sampled => FlagType::SAMPLED,
            ImageUseAs::Storage => FlagType::STORAGE,
            ImageUseAs::ColorAttachment => FlagType::COLOR_ATTACHMENT,
            ImageUseAs::DepthStencilAttachment => FlagType::DEPTH_STENCIL_ATTACHMENT,
            ImageUseAs::TransientAttachment => FlagType::TRANSIENT_ATTACHMENT,
            ImageUseAs::InputAttachment => FlagType::INPUT_ATTACHMENT,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct ImageUsage(ash::vk::ImageUsageFlags);

impl From<u32> for ImageUsage {
    fn from(value: u32) -> Self {
        Self(ash::vk::ImageUsageFlags::from_raw(value))
    }
}

impl From<ImageUsage> for ash::vk::ImageUsageFlags {
    fn from(val: ImageUsage) -> Self {
        val.0.to_owned()
    }
}

impl From<&[ImageUseAs]> for ImageUsage {
    fn from(value: &[ImageUseAs]) -> Self {
        let mut usage = ash::vk::ImageUsageFlags::empty();
        for flag in value.iter() {
            usage |= flag.into()
        }

        Self(usage)
    }
}

/// Represents the format of an Image
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[repr(transparent)]
pub struct ImageFormat(ash::vk::Format);

impl Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_raw())
    }
}

impl From<CommonImageFormat> for ImageFormat {
    fn from(value: CommonImageFormat) -> Self {
        unsafe { std::mem::transmute_copy::<CommonImageFormat, i32>(&value) }.into()
    }
}

impl From<i32> for ImageFormat {
    fn from(value: i32) -> Self {
        Self(crate::ash::vk::Format::from_raw(value))
    }
}

impl From<&crate::ash::vk::Format> for ImageFormat {
    fn from(value: &crate::ash::vk::Format) -> Self {
        Self(*value)
    }
}

impl From<crate::ash::vk::Format> for ImageFormat {
    fn from(value: crate::ash::vk::Format) -> Self {
        Self(value)
    }
}

impl From<ImageFormat> for crate::ash::vk::Format {
    fn from(val: ImageFormat) -> Self {
        (&val).into()
    }
}

impl From<&ImageFormat> for crate::ash::vk::Format {
    fn from(val: &ImageFormat) -> Self {
        val.0.to_owned()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageFlags {
    Managed(ImageFlagsCollection),
    Unmanaged(u32),
}

impl ImageFlags {
    pub(crate) fn ash_flags(&self) -> ash::vk::ImageCreateFlags {
        match self {
            Self::Unmanaged(raw) => ash::vk::ImageCreateFlags::from_raw(*raw),
            Self::Managed(r) => r.ash_flags(),
            //Self::Empty => ash::vk::ImageCreateFlags::from_raw(0),
        }
    }

    pub fn empty() -> Self {
        Self::Unmanaged(0u32)
    }

    pub fn from(flags: ImageFlagsCollection) -> Self {
        Self::Managed(flags)
    }

    pub fn from_raw(flags: u32) -> Self {
        Self::Unmanaged(flags)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ConcreteImageDescriptor {
    img_dimensions: ImageDimensions,
    img_usage: ImageUsage,
    img_multisampling: ImageMultisampling,
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
        self.img_usage.into()
    }

    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        self.img_format.to_owned().into()
    }

    pub(crate) fn ash_image_type(&self) -> ImageType {
        match &self.img_dimensions {
            ImageDimensions::Image1D { extent: _ } => ImageType::TYPE_1D,
            ImageDimensions::Image2D { extent: _ } => ImageType::TYPE_2D,
            ImageDimensions::Image3D { extent: _ } => ImageType::TYPE_3D,
        }
    }

    pub(crate) fn ash_sample_count(&self) -> ash::vk::SampleCountFlags {
        self.img_multisampling.ash_samples()
    }

    pub(crate) fn ash_extent_3d(&self) -> Extent3D {
        self.img_dimensions.ash_extent_3d()
    }

    pub fn new(
        img_dimensions: ImageDimensions,
        img_usage: ImageUsage,
        img_multisampling: ImageMultisampling,
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

    fn flags(&self) -> ImageFlags;

    fn usage(&self) -> ImageUsage;

    fn format(&self) -> ImageFormat;

    fn dimensions(&self) -> ImageDimensions;

    fn layers_count(&self) -> u32;

    fn mip_levels_count(&self) -> u32;
}

pub struct Image {
    device: Arc<Device>,
    image: ash::vk::Image,
    descriptor: ConcreteImageDescriptor,
}

impl Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image {}", ash::vk::Image::as_raw(self.image))
    }
}

impl DeviceOwned for Image {
    fn get_parent_device(&self) -> Arc<Device> {
        self.device.clone()
    }
}

impl Image {
    pub(crate) fn ash_native(&self) -> ash::vk::Image {
        self.image
    }

    //pub(crate) fn ash_format(&self) -> ash::vk::Format {
    //    self.descriptor.ash_format()
    //}

    pub(crate) fn descriptor(&self) -> &ConcreteImageDescriptor {
        &self.descriptor
    }

    /**
     * Create a new Image on the specified device.
     *
     * The resulting image has the initial layout of VK_IMAGE_LAYOUT_UNDEFINED.
     */
    pub fn new(
        device: Arc<Device>,
        descriptor: ConcreteImageDescriptor,
        sharing: Option<&[std::sync::Weak<QueueFamily>]>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Self> {
        if descriptor.img_layers == 0 {
            return Err(VulkanError::Framework(
                FrameworkError::NoImageLayersSpecified,
            ));
        }

        if descriptor.img_mip_levels == 0 {
            return Err(VulkanError::Framework(
                FrameworkError::NoMipMapLevelsSpecified,
            ));
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

        let create_info = ash::vk::ImageCreateInfo::default()
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
            .tiling(descriptor.ash_tiling());

        let image = unsafe {
            device.ash_handle().create_image(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }?;

        let mut obj_name_bytes = vec![];
        if let Some(ext) = device.ash_ext_debug_utils_ext() {
            if let Some(name) = debug_name {
                for name_ch in name.as_bytes().iter() {
                    obj_name_bytes.push(*name_ch);
                }
                obj_name_bytes.push(0x00);

                unsafe {
                    let object_name =
                        std::ffi::CStr::from_bytes_with_nul_unchecked(obj_name_bytes.as_slice());
                    // set device name for debugging
                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(image)
                        .object_name(object_name);

                    if let Err(err) = ext.set_debug_utils_object_name(&dbg_info) {
                        #[cfg(debug_assertions)]
                        {
                            println!("Error setting the Debug name for the newly created Image, will use handle. Error: {}", err)
                        }
                    }
                }
            }
        }

        Ok(Self {
            device,
            image,
            descriptor,
        })
    }
}

impl AllocationRequiring for Image {
    fn allocation_requirements(&self) -> AllocationRequirements {
        let requirements_info =
            ash::vk::ImageMemoryRequirementsInfo2::default().image(self.ash_native());

        let mut requirements = ash::vk::MemoryRequirements2::default();

        unsafe {
            self.get_parent_device()
                .ash_handle()
                .get_image_memory_requirements2(&requirements_info, &mut requirements);
        };

        AllocationRequirements::new(
            requirements.memory_requirements.memory_type_bits,
            requirements.memory_requirements.size,
            requirements.memory_requirements.alignment,
        )
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.get_parent_device();

        unsafe {
            device.ash_handle().destroy_image(
                self.image,
                device.get_parent_instance().get_alloc_callbacks(),
            );
        }
    }
}

pub struct AllocatedImage {
    memory_pool: Arc<MemoryPool>,
    reserved_memory_from_pool: SuccessfulAllocation,
    image: Image,
}

impl DeviceOwned for AllocatedImage {
    fn get_parent_device(&self) -> Arc<Device> {
        self.get_backing_memory_pool()
            .get_parent_memory_heap()
            .get_parent_device()
    }
}

impl AllocatedImage {
    /**
     * Allocate the given Image on the provided memory pool.
     *
     * The creation process is expected to fail if the memory pool does not allow allocation for such resource,
     * either because the allocation fails (for example due to lack of memory) or because such image
     * has memory requirements that cannot be met.
     */
    pub fn new(memory_pool: Arc<MemoryPool>, image: Image) -> VulkanResult<Arc<Self>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();

        if image.get_parent_device() != memory_pool.get_parent_memory_heap().get_parent_device() {
            return Err(VulkanError::Framework(
                FrameworkError::MemoryHeapAndResourceNotFromTheSameDevice,
            ));
        }

        let requirments = image.allocation_requirements();

        if !memory_pool
            .get_parent_memory_heap()
            .check_memory_type_bits_are_satified(requirments.memory_type_bits())
        {
            return Err(VulkanError::Framework(
                FrameworkError::IncompatibleMemoryHeapType,
            ));
        }

        match memory_pool
            .get_memory_allocator()
            .alloc(requirments.size(), requirments.alignment())
        {
            Some(reserved_memory_from_pool) => {
                unsafe {
                    device.ash_handle().bind_image_memory(
                        image.ash_native(),
                        memory_pool.ash_handle(),
                        reserved_memory_from_pool.offset_in_pool(),
                    )
                }.inspect_err(|&err| {
                    println!("ERROR: Error allocating memory on the device: {}, probably this is due to an incorrect implementation of the memory allocation algorithm", err);})?;

                Ok(Arc::new(Self {
                    memory_pool,
                    reserved_memory_from_pool,
                    image,
                }))
            }
            None => Err(VulkanError::Framework(FrameworkError::MallocFail(
                UnallocatedResource::Image(image),
            ))),
        }
    }
}

impl MemoryPoolBacked for AllocatedImage {
    fn get_backing_memory_pool(&self) -> Arc<MemoryPool> {
        self.memory_pool.clone()
    }

    fn allocation_offset(&self) -> u64 {
        self.reserved_memory_from_pool.offset_in_pool()
    }

    fn allocation_size(&self) -> u64 {
        self.reserved_memory_from_pool.size()
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        self.memory_pool
            .get_memory_allocator()
            .dealloc(&mut self.reserved_memory_from_pool)
    }
}

impl ImageTrait for AllocatedImage {
    #[inline]
    fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image.ash_native())
    }

    #[inline]
    fn flags(&self) -> ImageFlags {
        self.image.descriptor().img_flags
    }

    #[inline]
    fn usage(&self) -> ImageUsage {
        self.image.descriptor().img_usage
    }

    #[inline]
    fn format(&self) -> ImageFormat {
        self.image.descriptor().img_format
    }

    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        self.image.descriptor().img_dimensions
    }

    #[inline]
    fn layers_count(&self) -> u32 {
        self.image.descriptor().img_layers
    }

    #[inline]
    fn mip_levels_count(&self) -> u32 {
        self.image.descriptor().img_mip_levels
    }
}
