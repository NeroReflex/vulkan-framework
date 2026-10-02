use std::{
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use vulkan_framework::{
    buffer::{BufferSubresourceRange, BufferTrait},
    command_buffer::CommandBufferRecorder,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::DescriptorSet,
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    image::{
        CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions, ImageAspect,
        ImageAspects, ImageFlags, ImageFormat, ImageLayout, ImageMultisampling,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage,
        ImageUseAs,
    },
    image_view::ImageView,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue::{Queue, SemaphoreWaitOp},
    queue_family::QueueFamilyOwned,
    sampler::{Filtering, MipmapMode, Sampler},
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, MAX_TEXTURES, RenderingError, RenderingResult,
    resources::{ResourceError, collection::LoadableResourcesCollection},
};

type DescriptorSetsType =
    smallvec::SmallVec<[Mutex<(u64, Option<Arc<DescriptorSet>>)>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct TextureManager {
    debug_name: String,

    queue: Arc<Queue>,

    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_sets: DescriptorSetsType,

    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

    stub_image: u32,
    textures: LoadableResourcesCollection<Arc<ImageView>>,
    sampler: Arc<Sampler>,
}

impl TextureManager {
    #[inline]
    pub fn descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    #[inline]
    pub fn is_loaded(&self, index: usize) -> bool {
        self.textures.fetch_loaded(index).is_some()
    }

    #[inline]
    pub(crate) fn wait_load_nonblock(&mut self) -> RenderingResult<usize> {
        self.textures.wait_load_nonblock()
    }

    #[inline]
    pub(crate) fn wait_load_blocking(&mut self) -> RenderingResult<usize> {
        self.textures.wait_load_blocking()
    }

    pub(crate) fn upload_wait(&self) -> Option<SemaphoreWaitOp> {
        self.textures.upload_wait()
    }

    pub fn texture_descriptor_set(&self, current_frame: usize) -> Arc<DescriptorSet> {
        let mut cache = self.descriptor_sets[current_frame].lock().unwrap();
        let status = self.textures.status();
        if cache.0 == status {
            if let Some(set) = &cache.1 {
                return set.clone();
            }
        }

        // Replace rather than update: pending command buffers keep the old set
        // and its referenced images alive, including textures removed meanwhile.
        let pool = DescriptorPool::new(
            self.descriptor_set_layout.get_parent_device(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    MAX_TEXTURES,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                ),
                1,
            ),
            Some("texture_manager_descriptor_pool"),
        )
        .unwrap();
        let set = DescriptorSet::new(pool, self.descriptor_set_layout.clone()).unwrap();
        let stub = self
            .textures
            .fetch_loaded(self.stub_texture_index() as usize)
            .unwrap();
        let images: Vec<_> = (0..self.textures.size())
            .map(|index| {
                (
                    ImageLayout::ShaderReadOnlyOptimal,
                    self.textures.fetch_loaded(index).unwrap_or(stub).clone(),
                    self.sampler.clone(),
                )
            })
            .collect();
        set.bind_resources(|binder| {
            binder.bind_combined_images_samplers(0, &images).unwrap();
        })
        .unwrap();
        *cache = (status, Some(set.clone()));
        set
    }

    #[inline]
    pub fn stub_texture_index(&self) -> u32 {
        self.stub_image.to_owned()
    }

    pub fn new(
        queue: Arc<Queue>,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        stub_image_data: Arc<dyn BufferTrait>,
        frames_in_flight: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let queue_family = queue.get_parent_queue_family();
        let device = queue_family.get_parent_device();

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [
                    ShaderStageAccessIn::Fragment,
                    ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::ClosestHit),
                ]
                .as_slice()
                .into(),
                BindingType::Native(NativeBindingType::CombinedImageSampler),
                0,
                MAX_TEXTURES,
            )]
            .as_slice(),
        )?;

        let mut descriptor_sets: DescriptorSetsType = smallvec::smallvec![];
        for _ in 0..frames_in_flight {
            descriptor_sets.push(Mutex::new((0, None)));
        }

        let stub_image = Self::create_image(
            device.clone(),
            ImageFormat::from(CommonImageFormat::bc7_srgb_block),
            Image2DDimensions::new(400, 400),
            1,
            String::from("texture_empty_image"),
            0,
        )?;

        let mut textures = LoadableResourcesCollection::new(
            queue_family,
            MAX_TEXTURES,
            String::from("texture_manager"),
        )?;

        let sampler = Sampler::new(
            device.clone(),
            Filtering::Linear,
            Filtering::Linear,
            MipmapMode::ModeLinear,
            1.0,
        )?;

        let mut allocator = memory_manager.lock().unwrap();
        let Some(stub_image) = textures.load(
            queue.clone(),
            |_| Self::allocate_image(allocator.deref_mut(), 0, stub_image, debug_name.clone()),
            |recorder, _, image_view| {
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
        drop(allocator);
        // Every unpopulated descriptor points here, including the first frame.
        textures.wait_load_blocking()?;

        Ok(Self {
            debug_name,

            queue,

            descriptor_set_layout,
            descriptor_sets,

            memory_manager,

            stub_image,
            textures,
            sampler,
        })
    }

    fn create_and_allocate_image(
        index: usize,
        memory_manager: &mut dyn MemoryManagerTrait,
        format: ImageFormat,
        dimensions: Image2DDimensions,
        mip_levels: u32,
        debug_name: String,
    ) -> RenderingResult<Arc<ImageView>> {
        let texture = Self::create_image(
            memory_manager.get_parent_device(),
            format,
            dimensions,
            mip_levels,
            debug_name.clone(),
            index,
        )?;
        Self::allocate_image(memory_manager, index, texture, debug_name)
    }

    fn create_image(
        device: Arc<Device>,
        format: ImageFormat,
        dimensions: Image2DDimensions,
        mip_levels: u32,
        debug_name: String,
        index: usize,
    ) -> RenderingResult<Image> {
        let texture_name = format!("{debug_name}.texture[{index}].imageview");
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
            Some(texture_name.as_str()),
        )?;

        Ok(texture)
    }

    fn allocate_image(
        allocator: &mut dyn MemoryManagerTrait,
        index: usize,
        texture: Image,
        debug_name: String,
    ) -> RenderingResult<Arc<ImageView>> {
        let texture = {
            let allocation_result = allocator.allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::from([].as_slice()),
                vec![texture.into()],
                MemoryManagementTags::default()
                    .with_name("images".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;
            assert_eq!(allocation_result.len(), 1);
            allocation_result[0].image()
        };

        let texture_imageview_name = format!("{debug_name}.texture[{index}].imageview");
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

        // Wait for host to finish writing the buffer
        recorder.pipeline_barriers([BufferMemoryBarrier::new(
            [PipelineStage::Host].as_slice().into(),
            [MemoryAccessAs::HostWrite].as_slice().into(),
            [PipelineStage::Transfer].as_slice().into(),
            [MemoryAccessAs::TransferRead].as_slice().into(),
            BufferSubresourceRange::new(image_data.clone(), 0u64, image_data.size()),
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::TopOfPipe].as_ref()),
            MemoryAccess::default(),
            PipelineStages::from([PipelineStage::Transfer].as_ref()),
            MemoryAccess::from([MemoryAccessAs::TransferWrite].as_slice()),
            ImageSubresourceRange::from(image.clone() as Arc<dyn ImageTrait>),
            ImageLayout::Undefined,
            ImageLayout::TransferDstOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

        recorder.copy_buffer_to_image(
            image_data,
            ImageLayout::TransferDstOptimal,
            ImageSubresourceLayers::new(ImageAspects::from([ImageAspect::Color].as_ref()), 0, 0, 1),
            image.clone(),
            image.dimensions(),
        );

        recorder.pipeline_barriers([ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::Transfer].as_ref()),
            MemoryAccess::from([MemoryAccessAs::TransferWrite].as_slice()),
            [
                PipelineStage::FragmentShader,
                PipelineStage::RayTracingPipelineKHR(
                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                ),
            ]
            .as_slice()
            .into(),
            [MemoryAccessAs::ShaderRead].as_slice().into(),
            ImageSubresourceRange::from(image),
            ImageLayout::TransferDstOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
            queue_family.clone(),
            queue_family.clone(),
        )
        .into()]);

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
        let mut allocator = self.memory_manager.lock().unwrap();
        let Some(texture_index) = self.textures.load(
            queue.clone(),
            |index| {
                Self::create_and_allocate_image(
                    index,
                    allocator.deref_mut(),
                    image_format,
                    image_dimenstions,
                    image_mip_levels,
                    self.debug_name.clone(),
                )
            },
            |recorder, _, image_view| {
                Self::setup_load_image_operation(recorder, image_data, image_view, queue.clone())
            },
        )?
        else {
            return Err(RenderingError::ResourceError(
                super::ResourceError::NoTextureSlotAvailable,
            ));
        };

        Ok(texture_index)
    }

    #[inline]
    pub fn remove(&mut self, index: u32) -> RenderingResult<()> {
        // Avoid removing the default texture
        if index == self.stub_image {
            return Err(RenderingError::ResourceError(
                ResourceError::AttemptedRemovalOfEmptyTexture,
            ));
        }

        self.textures.remove(index)
    }
}
