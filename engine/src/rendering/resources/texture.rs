use std::sync::Arc;

use vulkan_framework::{
    buffer::BufferTrait,
    command_buffer::{CommandBufferRecorder, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions,
        ImageAspect, ImageAspects, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_allocator::DefaultAllocator,
    memory_heap::{
        ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHeapOwned, MemoryRequirements, MemoryType,
    },
    memory_pool::{MemoryPool, MemoryPoolFeature, MemoryPoolFeatures},
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingError, RenderingResult,
    resources::{ResourceError, collection::LoadableResourcesCollection},
};

type DescriptorSetsType = smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct TextureManager {
    queue: Arc<Queue>,

    _descriptor_pool: Arc<DescriptorPool>,
    binding_descriptor: Arc<BindingDescriptor>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_sets: DescriptorSetsType,

    memory_pool: Arc<MemoryPool>,

    stub_image: u32,
    textures: LoadableResourcesCollection<Arc<ImageView>>,
    sampler: Arc<Sampler>,
}

impl TextureManager {
    #[inline]
    fn texture_memory_pool_size(max_textures: u32, frames_in_flight: u32) -> u64 {
        (1024u64 * 1024u64 * 128u64)
            + ((1024u64 * 512u64 * (max_textures as u64)) * (frames_in_flight as u64))
    }

    #[inline]
    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        self.textures.wait_load_nonblock()
    }

    #[inline]
    pub fn stub_texture_index(&self) -> u32 {
        self.stub_image.to_owned()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        stub_image_data: Arc<dyn BufferTrait>,
        max_textures: u32,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let queue = Queue::new(queue_family.clone(), Some("texture_manager.queue"))?;

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    max_textures * frames_in_flight,
                    0,
                    max_textures * frames_in_flight,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("texture_manager_descriptor_pool"),
        )?;

        let binding_descriptor = BindingDescriptor::new(
            ShaderStagesAccess::graphics(),
            BindingType::Native(NativeBindingType::CombinedImageSampler),
            0,
            max_textures,
        );

        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), &[binding_descriptor.clone()])?;

        let mut descriptor_sets: DescriptorSetsType = smallvec::smallvec![];
        for _ in 0..frames_in_flight as usize {
            let descriptor_set =
                DescriptorSet::new(descriptor_pool.clone(), descriptor_set_layout.clone())?;

            descriptor_sets.push(descriptor_set);
        }

        let stub_image = Self::create_image(
            device.clone(),
            ImageFormat::from(CommonImageFormat::bc7_srgb_block),
            Image2DDimensions::new(400, 400),
            1,
            String::from("texture_empty_image"),
        )?;

        let total_size = Self::texture_memory_pool_size(max_textures, frames_in_flight);

        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(MemoryType::DeviceLocal(None), total_size),
            MemoryRequirements::from(&stub_image),
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(total_size)),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let mut textures = LoadableResourcesCollection::new(
            queue_family,
            max_textures,
            String::from("texture_manager"),
        )?;

        let sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeLinear,
            1.0,
        )?;

        let Some(stub_image) = textures.load(
            queue.clone(),
            || Self::allocate_image(memory_pool.clone(), stub_image),
            |recorder, image_view| {
                Self::setup_load_image_operation(
                    recorder,
                    stub_image_data,
                    image_view,
                    queue.clone(),
                )
            },
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoTextureSlotAvailable,
            ));
        };

        Ok(Self {
            queue,

            _descriptor_pool: descriptor_pool,
            binding_descriptor,
            descriptor_set_layout,
            descriptor_sets,

            memory_pool,

            stub_image,
            textures,
            sampler,
        })
    }

    fn create_and_allocate_image(
        memory_pool: Arc<MemoryPool>,
        format: ImageFormat,
        dimensions: Image2DDimensions,
        mip_levels: u32,
        debug_name: String,
    ) -> RenderingResult<Arc<ImageView>> {
        let device = memory_pool.get_parent_memory_heap().get_parent_device();
        let texture = Self::create_image(device, format, dimensions, mip_levels, debug_name)?;
        Self::allocate_image(memory_pool, texture)
    }

    fn create_image(
        device: Arc<Device>,
        format: ImageFormat,
        dimensions: Image2DDimensions,
        mip_levels: u32,
        debug_name: String,
    ) -> RenderingResult<Image> {
        let texture = Image::new(
            device.clone(),
            ConcreteImageDescriptor::new(
                dimensions.into(),
                ImageUsage::from([ImageUseAs::TransferDst, ImageUseAs::Sampled].as_slice()),
                ImageMultisampling::SamplesPerPixel1,
                1,
                mip_levels,
                format,
                ImageFlags::empty(),
                ImageTiling::Optimal,
            ),
            None,
            Some(debug_name.as_str()),
        )?;

        Ok(texture)
    }

    fn allocate_image(
        memory_pool: Arc<MemoryPool>,
        texture: Image,
    ) -> RenderingResult<Arc<ImageView>> {
        let texture = AllocatedImage::new(memory_pool, texture)?;
        let texture_imageview_name = "texture_manager.texture[...].imageview".to_string();
        let texture_imageview = ImageView::new(
            texture.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(texture_imageview_name.as_str()),
        )?;

        Ok(texture_imageview)
    }

    fn setup_load_image_operation(
        recorder: &mut CommandBufferRecorder,
        image_data: Arc<dyn BufferTrait>,
        image_view: Arc<ImageView>,
        queue: Arc<Queue>,
    ) -> RenderingResult<()> {
        let queue_family = queue.get_parent_queue_family();
        let image = image_view.image();

        let before_transfer_barrier = ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::TopOfPipe].as_ref()),
            MemoryAccess::default(),
            PipelineStages::from([PipelineStage::Transfer].as_ref()),
            MemoryAccess::from([MemoryAccessAs::MemoryWrite].as_slice()),
            ImageSubresourceRange::from(image.clone() as Arc<dyn ImageTrait>),
            ImageLayout::Undefined,
            ImageLayout::TransferDstOptimal,
            queue_family.clone(),
            queue_family.clone(),
        );

        recorder.image_barrier(before_transfer_barrier);

        recorder.copy_buffer_to_image(
            image_data,
            ImageLayout::TransferDstOptimal,
            ImageSubresourceLayers::new(ImageAspects::from([ImageAspect::Color].as_ref()), 0, 0, 1),
            image.clone(),
            image.dimensions(),
        );

        let after_transfer_barrier = ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::Transfer].as_ref()),
            MemoryAccess::from([MemoryAccessAs::MemoryWrite].as_slice()),
            PipelineStages::from([PipelineStage::BottomOfPipe].as_ref()),
            MemoryAccess::default(),
            ImageSubresourceRange::from(image),
            ImageLayout::TransferDstOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_family.clone(),
            queue_family.clone(),
        );

        recorder.image_barrier(after_transfer_barrier);

        Ok(())
    }

    pub fn load(
        &mut self,
        image_format: ImageFormat,
        image_dimenstions: Image2DDimensions,
        image_mip_levels: u32,
        image_data: Arc<dyn BufferTrait>,
    ) -> RenderingResult<u32> {
        let queue = self.queue.clone();
        match self.textures.load(
            queue.clone(),
            || {
                Self::create_and_allocate_image(
                    self.memory_pool.clone(),
                    image_format,
                    image_dimenstions,
                    image_mip_levels,
                    String::from("texture_empty_image"),
                )
            },
            |recorder, image_view| {
                Self::setup_load_image_operation(recorder, image_data, image_view, queue.clone())
            },
        )? {
            Some(texture_index) => Ok(texture_index),
            None => Err(RenderingError::ResourceError(
                super::ResourceError::NoTextureSlotAvailable,
            )),
        }
    }

    #[inline]
    pub fn remove(&mut self, index: u32) -> RenderingResult<()> {
        // Avoid removing the default texture
        if index == 0 {
            return Err(RenderingError::ResourceError(
                ResourceError::AttemptedRemovalOfEmptyTexture,
            ));
        }

        self.textures.remove(index)
    }
}
