use ash::vk::{Extent3D, SampleCountFlags};

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
pub struct ConcreteImageDescriptor {
    img_dimensions: ImageDimensions,
    img_multisampling: Option<ImageMultisampling>,
    img_layers: u32,
    img_mip_levels: u32,
}

impl ConcreteImageDescriptor {
    pub(crate) fn ash_sample_count(&self) -> SampleCountFlags {
        match self.img_multisampling.clone() {
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
        match self.img_dimensions.clone() {
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
    ) -> Self {
        Self {
            img_dimensions,
            img_multisampling,
            img_layers,
            img_mip_levels,
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
    pub fn new(
        memory_pool: Arc<MemoryPool<Allocator>>,
        descriptor: ConcreteImageDescriptor,
    ) -> VulkanResult<Arc<Self>> {
        let create_info = ash::vk::ImageCreateInfo::builder()
            .extent(descriptor.ash_extent_3d())
            .samples(descriptor.ash_sample_count())
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
