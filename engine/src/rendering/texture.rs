use std::sync::Arc;

use vulkan_framework::{
    buffer::BufferTrait,
    command_buffer::PrimaryCommandBuffer,
    command_pool::CommandPool,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    fence::{Fence, FenceWaiter},
    image::{
        AllocatedImage, ConcreteImageDescriptor, Image, Image2DDimensions, ImageAspects,
        ImageDimensions, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceLayers, ImageTiling, ImageTrait, ImageUsage, ImageUsageSpecifier,
    },
    image_view::ImageView,
    memory_allocator::DefaultAllocator,
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryType},
    memory_pool::{MemoryPool, MemoryPoolFeature, MemoryPoolFeatures},
    queue::Queue,
    queue_family::QueueFamily,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStagesAccess,
};

use crate::rendering::{MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingResult};

type DescriptorSetsType = smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

type TexturesMapType = smallvec::SmallVec<[Option<(Arc<AllocatedImage>, Arc<ImageView>)>; 128]>;

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

        let command_pool = CommandPool::new(queue_family, Some("texture_manager.command_pool"))?;
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

        let stub_image = Image::new(
            device.clone(),
            ConcreteImageDescriptor::new(
                ImageDimensions::Image2D {
                    extent: Image2DDimensions::new(400, 400),
                },
                ImageUsage::Managed(ImageUsageSpecifier::new(
                    false, true, true, false, false, false, false, false,
                )),
                ImageMultisampling::SamplesPerPixel1,
                1,
                1,
                ImageFormat::bc7_srgb_block,
                ImageFlags::empty(),
                ImageTiling::Optimal,
            ),
            None,
            Some("texture_empty_image"),
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

        command_buffer.record_commands(|recorder| {
            recorder.copy_buffer_to_image(
                stub_image_data,
                ImageLayout::TransferDstOptimal,
                ImageSubresourceLayers::new(ImageAspects::new(true, true, false, false), 1, 0, 1),
                stub_image.clone(),
                stub_image.dimensions(),
            );
            /*
            recorder.copy_image(
                src_layout,
                src_subresource,
                src,
                dst_layout,
                dst_subresource,
                dst,
                extent,
            );
            */
        })?;

        let load_fence_waiter =
            queue.submit(&[command_buffer.clone()], &[], &[], load_fence.clone());

        // this will wait for the GPU to finish the resource copy
        // and the fence will be resetted back in unsignaled state
        drop(load_fence_waiter.unwrap());

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
}
