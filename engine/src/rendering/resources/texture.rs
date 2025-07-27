use std::sync::Arc;

use vulkan_framework::{
    buffer::BufferTrait,
    command_buffer::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs, PrimaryCommandBuffer},
    command_pool::CommandPool,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    fence::{Fence, FenceWaiter},
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions,
        ImageAspect, ImageAspects, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_allocator::DefaultAllocator,
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeature, MemoryPoolFeatures},
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
};

use crate::rendering::{MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingError, RenderingResult};

type DescriptorSetsType = smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type TexturesMapType = smallvec::SmallVec<[Option<Arc<ImageView>>; 128]>;

pub struct TextureManager {
    device: Arc<Device>,
    queue: Arc<Queue>,

    command_pool: Arc<CommandPool>,
    command_buffer: Arc<PrimaryCommandBuffer>,

    _descriptor_pool: Arc<DescriptorPool>,
    binding_descriptor: Arc<BindingDescriptor>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_sets: DescriptorSetsType,

    memory_pool: Arc<MemoryPool>,

    stub_image: Arc<AllocatedImage>,
    textures: TexturesMapType,
    sampler: Arc<Sampler>,

    load_fence: Arc<Fence>,
    load_fence_waiter: Option<FenceWaiter>,
}

impl TextureManager {
    fn texture_memory_pool_size(max_textures: u32, frames_in_flight: u32) -> u64 {
        (1024u64 * 1024u64 * 128u64)
            + ((1024u64 * 512u64 * (max_textures as u64)) * (frames_in_flight as u64))
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        stub_image_data: Arc<dyn BufferTrait>,
        max_textures: u32,
        frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let queue = Queue::new(queue_family.clone(), Some("texture_manager.queue"))?;

        let command_pool =
            CommandPool::new(queue_family.clone(), Some("texture_manager.command_pool"))?;
        let command_buffer =
            PrimaryCommandBuffer::new(command_pool.clone(), Some("texture_manager.command_pool"))?;

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
            &[&stub_image],
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(total_size)),
            MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable {}].as_slice()),
        )?;

        let stub_image = AllocatedImage::new(memory_pool.clone(), stub_image)?;

        let textures: TexturesMapType = (0..max_textures as usize).map(|_| Option::None).collect();

        let sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeLinear,
            1.0,
        )?;

        let load_fence = Fence::new(device.clone(), false, Some("texture_manager.load_fence"))?;

        let load_fence_waiter = Self::setup_load_image_operation(
            command_buffer.clone(),
            stub_image_data,
            stub_image.clone(),
            load_fence.clone(),
            queue.clone(),
        )?;

        // this will wait for the GPU to finish the resource copy
        // and the fence will be resetted back in unsignaled state
        drop(load_fence_waiter);

        Ok(Self {
            device,
            queue,

            command_pool,
            command_buffer,

            _descriptor_pool: descriptor_pool,
            binding_descriptor,
            descriptor_set_layout,
            descriptor_sets,

            memory_pool,

            stub_image,
            textures,
            sampler,

            load_fence,
            load_fence_waiter: Option::None,
        })
    }

    /// Sets up an image loading operation in a command buffer and submits it for execution.
    ///
    /// This function records a one-time submit operation to load image data from a buffer into a Vulkan image.
    /// It performs the necessary image memory barriers to ensure proper synchronization and layout transitions
    /// during the transfer operation.
    ///
    /// # Parameters
    /// - `command_buffer`: An `Arc<PrimaryCommandBuffer>` that will be used to record the image loading commands.
    /// - `image_data`: An `Arc<dyn BufferTrait>` representing the buffer containing the image data to be loaded.
    /// - `image`: An `Arc<dyn ImageTrait>` representing the Vulkan image that will receive the loaded data.
    /// - `load_fence`: An `Arc<Fence>` used to synchronize the completion of the image loading operation.
    /// - `queue`: An `Arc<Queue>` representing the Vulkan queue to which the command buffer will be submitted.
    ///
    /// # Returns
    /// A `RenderingResult<FenceWaiter>` that contains a `FenceWaiter` for waiting on the completion of the operation.
    ///
    /// # Errors
    /// This function may return an error if the command buffer recording or queue submission fails.
    ///
    /// # Operation Overview
    /// 1. **Image Memory Barrier (Before Transfer)**:
    ///    - Sets up a barrier to transition the image layout from `Undefined` to `TransferDstOptimal`
    ///      before the transfer operation begins. This ensures that the image is ready to receive data.
    ///
    /// 2. **Copy Buffer to Image**:
    ///    - Copies the image data from the provided buffer to the specified image using the `copy_buffer_to_image` method.
    ///
    /// 3. **Image Memory Barrier (After Transfer)**:
    ///    - Sets up another barrier to transition the image layout from `TransferDstOptimal` to `ShaderReadOnlyOptimal`
    ///      after the transfer operation is complete, making the image ready for shader access.
    ///
    /// The function encapsulates the entire process of preparing the command buffer, recording the necessary operations,
    /// and submitting the command buffer to the queue for execution.
    fn setup_load_image_operation(
        command_buffer: Arc<PrimaryCommandBuffer>,
        image_data: Arc<dyn BufferTrait>,
        image: Arc<dyn ImageTrait>,
        load_fence: Arc<Fence>,
        queue: Arc<Queue>,
    ) -> RenderingResult<FenceWaiter> {
        let queue_family = queue.get_parent_queue_family();

        command_buffer.record_one_time_submit(|recorder| {
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
                ImageSubresourceLayers::new(
                    ImageAspects::from([ImageAspect::Color].as_ref()),
                    0,
                    0,
                    1,
                ),
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
        })?;

        let load_fence_waiter =
            queue.submit(&[command_buffer.clone()], &[], &[], load_fence.clone())?;

        Ok(load_fence_waiter)
    }

    fn create_image(
        device: Arc<Device>,
        format: ImageFormat,
        dimensions: Image2DDimensions,
        mip_levels: u32,
        debug_name: String,
    ) -> RenderingResult<Image> {
        let image = Image::new(
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

        Ok(image)
    }

    pub fn load(
        &mut self,
        image_format: ImageFormat,
        image_dimenstions: Image2DDimensions,
        image_mip_levels: u32,
        image_data: Arc<dyn BufferTrait>,
    ) -> RenderingResult<u32> {
        for index in 0..self.textures.len() {
            // ensure this is an avaialble slot for the new texture
            if !self.textures[index].is_none() {
                continue;
            }

            let texture_name = format!("texture_manager.texture[{index}].image");
            let texture = Self::create_image(
                self.device.clone(),
                image_format,
                image_dimenstions,
                image_mip_levels,
                texture_name,
            )?;

            let texture = AllocatedImage::new(self.memory_pool.clone(), texture)?;
            let texture_imageview_name = format!("texture_manager.texture[{index}].imageview");
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

            self.textures[index] = Some(texture_imageview);

            // wait for the previous image to be loaded
            drop(self.load_fence_waiter.take());

            let fence_waiter = Self::setup_load_image_operation(
                self.command_buffer.clone(),
                image_data,
                texture,
                self.load_fence.clone(),
                self.queue.clone(),
            )?;

            self.load_fence_waiter.replace(fence_waiter);

            // the image has been created
            return Ok(index as u32);
        }

        return Err(RenderingError::ResourceError(
            super::ResourceError::NoTextureSlotAvailable,
        ));
    }
}
