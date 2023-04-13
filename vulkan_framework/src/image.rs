use ash::vk::{Extent3D, ImageLayout, ImageType, ImageUsageFlags, SampleCountFlags};

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwned,
    memory_allocator::{AllocationResult, MemoryAllocator},
    memory_heap::MemoryHeapOwned,
    memory_pool::{MemoryPool, MemoryPoolBacked},
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

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

#[derive(Clone)]
pub enum ImageUsage {}

/**
 * Specify the image format for an image.
 *
 * Common image formats has a name for convenience,
 * but it is always possible to specify Other(VkFormat).
 */
#[derive(Copy, Clone, PartialEq)]
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

#[derive(Clone)]
pub struct ConcreteImageDescriptor {
    img_dimensions: ImageDimensions,
    img_multisampling: Option<ImageMultisampling>,
    img_layers: u32,
    img_mip_levels: u32,
    img_format: ImageFormat,
}

impl ConcreteImageDescriptor {
    pub(crate) fn ash_format(&self) -> ash::vk::Format {
        ash::vk::Format::from_raw(match &self.img_format {
            ImageFormat::Other(fmt) => *fmt,
            fmt => unsafe { std::mem::transmute_copy::<ImageFormat, u32>(fmt) },
        } as i32)
    }

    pub(crate) fn ash_image_type(&self) -> ImageType {
        match &self.img_dimensions {
            ImageDimensions::Image1D { extent } => ImageType::TYPE_1D,
            ImageDimensions::Image2D { extent } => ImageType::TYPE_2D,
            ImageDimensions::Image3D { extent } => ImageType::TYPE_3D,
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
        img_multisampling: Option<ImageMultisampling>,
        img_layers: u32,
        img_mip_levels: u32,
        img_format: ImageFormat,
    ) -> Self {
        Self {
            img_dimensions,
            img_multisampling,
            img_layers,
            img_mip_levels,
            img_format,
        }
    }
}

pub trait ImageTrait {
    fn dimensions(&self) -> ImageDimensions;

    fn layers_count(&self) -> u32;

    fn mip_levels_count(&self) -> u32;
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
    ) -> VulkanResult<Arc<Self>> {

        if descriptor.img_layers == 0 {
            return Err(VulkanError::Unspecified);
        }

        if descriptor.img_mip_levels == 0 {
            return Err(VulkanError::Unspecified);
        }

        let create_info = ash::vk::ImageCreateInfo::builder()
            .image_type(descriptor.ash_image_type())
            .extent(descriptor.ash_extent_3d())
            .samples(descriptor.ash_sample_count())
            .mip_levels(descriptor.img_mip_levels)
            .array_layers(descriptor.img_layers)
            .initial_layout(ImageLayout::UNDEFINED)
            .format(descriptor.ash_format())
            .usage(ImageUsageFlags::INPUT_ATTACHMENT)
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
                        println!("Error creating the image: {}", err);
                        assert_eq!(true, false)
                    }

                    return Err(VulkanError::Unspecified);
                }
            }
        };

        unsafe {
            let requirements = device
                .ash_handle()
                .get_image_memory_requirements(image.clone());
            match memory_pool.alloc(requirements) {
                Some(reserved_memory_from_pool) => {
                    match device.ash_handle().bind_image_memory(
                        image.clone(),
                        memory_pool.native_handle(),
                        reserved_memory_from_pool.offset_in_pool(),
                    ) {
                        Ok(_) => Ok(Arc::new(Self {
                            memory_pool,
                            reserved_memory_from_pool,
                            image,
                            descriptor,
                        })),
                        Err(err) => {
                            #[cfg(debug_assertions)]
                            {
                                println!("Error allocating memory on the device: {}, probably this is due to an incorrect implementation of the memory allocation algorithm", err);
                                assert_eq!(true, false)
                            }

                            // the image will not let this function, destroy it or it will leak
                            device.ash_handle().destroy_image(
                                image,
                                device.get_parent_instance().get_alloc_callbacks(),
                            );

                            return Err(VulkanError::Unspecified);
                        }
                    }
                }
                None => {
                    // the image will not let this function, destroy it or it will leak
                    device
                        .ash_handle()
                        .destroy_image(image, device.get_parent_instance().get_alloc_callbacks());

                    return Err(VulkanError::Unspecified);
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
}
