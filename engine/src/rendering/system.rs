use std::{
    ops::Deref,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use sdl2::VideoSubsystem;
use vulkan_framework::{
    acceleration_structure::bottom_level::IDENTITY_MATRIX,
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    command_buffer::PrimaryCommandBuffer,
    command_pool::CommandPool,
    descriptor_pool::{
        DescriptorPool, DescriptorPoolConcreteDescriptor,
        DescriptorPoolSizesAcceletarionStructureKHR, DescriptorPoolSizesConcreteDescriptor,
    },
    descriptor_set::{DescriptorSet, DescriptorSetWriter},
    descriptor_set_layout::DescriptorSetLayout,
    device::{Device, DeviceOwned},
    fence::{Fence, FenceWaiter},
    image::{
        Image1DTrait, Image2DDimensions, Image2DTrait, ImageLayout, ImageLayoutSwapchainKHR,
        ImageUsage, ImageUseAs,
    },
    image_view::ImageView,
    instance::InstanceOwned,
    memory_barriers::{BufferMemoryBarrier, ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::MemoryType,
    memory_management::{DefaultMemoryManager, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue::Queue,
    queue_family::{ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilySupportedOperationType},
    semaphore::Semaphore,
    shader_layout_binding::{
        AccelerationStructureBindingType, BindingDescriptor, BindingType, NativeBindingType,
    },
    shader_stage_access::{ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR},
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
};

use crate::{
    core::{camera::CameraTrait, hdr::HDR, lights::directional::DirectionalLight},
    rendering::{
        MAX_DIRECTIONAL_LIGHTS, MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingError, RenderingResult,
        pipeline::{
            directional_lighting::DirectionalLighting, final_rendering::FinalRendering,
            global_illumination::GILighting, hdr_transform::HDRTransform,
            mesh_rendering::MeshRendering, renderquad::RenderQuad,
        },
        rendering_dimensions::RenderingDimensions,
        resources::{
            directional_lights::DirectionalLights,
            object::{Manager as ResourceManager, TLASRebuildDevice},
        },
        surface::SurfaceHelper,
    },
};

type SwapchainImagesType =
    smallvec::SmallVec<[Arc<ImageSwapchainKHR>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;
type SwapchainImageViewsType = smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct System {
    swapchain: Option<(Arc<SwapchainKHR>, SwapchainImageViewsType)>,

    queue_family: Arc<QueueFamily>,
    queues: smallvec::SmallVec<[Arc<Queue>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    rendering_fences: smallvec::SmallVec<[Arc<Fence>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    current_frame: AtomicUsize,

    image_available_semaphores:
        smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    _command_pool: Arc<CommandPool>,
    present_command_buffers:
        smallvec::SmallVec<[Arc<PrimaryCommandBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    present_ready: smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    surface: SurfaceHelper,

    view_projection_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    view_projection_buffers:
        smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
    rt_descriptor_pool: Arc<DescriptorPool>,
    rt_descriptor_set: Option<Arc<DescriptorSet>>,

    mesh_rendering: Arc<MeshRendering>,
    directional_lighting: Arc<DirectionalLighting>,
    global_illumination_lighting: Arc<GILighting>,
    final_rendering: Arc<FinalRendering>,
    hdr: Arc<HDRTransform>,
    renderquad: Arc<RenderQuad>,

    resources_manager: Arc<Mutex<ResourceManager>>,
    lights_manager: Arc<Mutex<DirectionalLights>>,

    frames_in_flight: smallvec::SmallVec<[Option<FenceWaiter>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    // It is VERY important that the window is dropped early
    // Otherwise it will be impossible to destroy the swapchain
    window: sdl2::video::Window,
}

impl Drop for System {
    fn drop(&mut self) {
        // wait for all fences to be signaled: meaning execution has ended
        for w in 0..self.frames_in_flight.len() {
            if let Some(fence_waiter) = self.frames_in_flight[w].take() {
                drop(fence_waiter)
            }
        }

        // wait for every other device operation to terminate
        self.device().wait_idle().unwrap();
    }
}

impl System {
    fn required_device_extensions() -> Vec<String> {
        vec![
            String::from("VK_KHR_swapchain"),
            String::from("VK_KHR_acceleration_structure"),
            String::from("VK_KHR_ray_tracing_maintenance1"),
            String::from("VK_KHR_ray_tracing_pipeline"),
            String::from("VK_KHR_buffer_device_address"),
            String::from("VK_KHR_deferred_host_operations"),
            String::from("VK_EXT_descriptor_indexing"),
            String::from("VK_KHR_spirv_1_4"),
            String::from("VK_KHR_shader_float_controls"),
        ]
    }

    pub fn resources_manager(&self) -> Arc<Mutex<ResourceManager>> {
        self.resources_manager.clone()
    }

    pub fn device(&self) -> Arc<vulkan_framework::device::Device> {
        self.queue_family().get_parent_device()
    }

    pub fn queue_family(&self) -> Arc<vulkan_framework::queue_family::QueueFamily> {
        self.queue_family.clone()
    }

    pub fn test(&mut self) {
        let mut manager = self.resources_manager.lock().unwrap();

        let sponza_object_id = manager
            .load_object(PathBuf::from("crytek_sponza.tar"), IDENTITY_MATRIX)
            .unwrap();

        manager
            .add_instance(sponza_object_id, IDENTITY_MATRIX, TLASRebuildDevice::GPU)
            .unwrap();

        /*
        scene->addDirectionalLight(
            NeroReflex::PBRenderer::Core::Lighting::DirectionalLight(
                glm::vec3(0.0f, -1.0, 0.0f),
                glm::vec3(1.0, 1.0, 1.0),
                glm::float32(10.2f)
            )
        );
        scene->addDirectionalLight(
            NeroReflex::PBRenderer::Core::Lighting::DirectionalLight(
                glm::vec3(0, +0.947768, 0.318959),
                glm::vec3(1.0, 1.0, 1.0),
                glm::float32(10.2f)
            )
        );
        scene->addDirectionalLight(
            NeroReflex::PBRenderer::Core::Lighting::DirectionalLight(
                glm::vec3(0.0, -0.98, 0.6),
                glm::vec3(1.0, 1.0, 0.90),
                glm::float32(10.2f)
            )
        );
        */

        let mut lights = self.lights_manager.lock().unwrap();
        lights
            .load(DirectionalLight::new(
                glm::Vec3::new(-0.6, -0.98, 0.00000001),
                glm::Vec3::new(10.2, 10.2, 10.2),
            ))
            .unwrap();

        // Update the TLAS and create a descriptor set for it:
        // this is very important as it define the geometry of the whole scene
        {
            let (tlas, tlas_data) = manager.tlas();

            // create the new descriptor set for RT pipelines
            let rt_descriptor_set = DescriptorSet::new(
                self.rt_descriptor_pool.clone(),
                self.rt_descriptor_set_layout.clone(),
            )
            .unwrap();

            // bind TLAS data to the new descriptor set
            rt_descriptor_set
                .bind_resources(|binder| {
                    binder
                        .bind_storage_buffers(0, [(tlas_data, None, None)].as_slice())
                        .unwrap();
                    binder.bind_tlas(1, [tlas].as_slice()).unwrap();
                })
                .unwrap();

            self.rt_descriptor_set = Some(rt_descriptor_set);
        }
    }

    pub fn new(
        app_name: String,
        video_subsystem: VideoSubsystem,
        initial_width: u32,
        initial_height: u32,
        preferred_frames_in_flight: u32,
    ) -> RenderingResult<Self> {
        let mut instance_extensions = vec![];
        let mut instance_layers = vec![];

        // Enable vulkan debug utils on debug builds
        #[cfg(debug_assertions)]
        {
            println!("Running with debugging features enabled...");
            instance_extensions.push(String::from("VK_EXT_debug_utils"));
            instance_layers.push(String::from("VK_LAYER_KHRONOS_validation"));
            //instance_layers.push(String::from("VK_LAYER_RENDERDOC_Capture"));
        }

        let engine_name = String::from("ArtRTic");

        let window = video_subsystem
            .window("Window", initial_width, initial_height)
            .vulkan()
            .build()
            .map_err(RenderingError::Window)?;

        let required_extensions = window
            .vulkan_instance_extensions()
            .map_err(RenderingError::Unknown)?;
        let instance_extensions = instance_extensions
            .into_iter()
            .chain(
                required_extensions
                    .iter()
                    .map(|ext_name| String::from(*ext_name)),
            )
            .collect::<Vec<_>>();

        let instance = vulkan_framework::instance::Instance::new(
            instance_layers.as_slice(),
            instance_extensions.as_slice(),
            &engine_name,
            &app_name,
        )?;

        let surface = vulkan_framework::surface::Surface::from_raw(
            instance.clone(),
            window
                .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
                .unwrap(),
        )?;

        let device = Device::new(
            surface.get_parent_instance(),
            [ConcreteQueueFamilyDescriptor::new(
                vec![
                    QueueFamilySupportedOperationType::Graphics,
                    QueueFamilySupportedOperationType::Transfer,
                    QueueFamilySupportedOperationType::Present(surface.clone()),
                ]
                .as_ref(),
                [1.0f32].as_slice(),
            )]
            .as_slice(),
            Self::required_device_extensions().as_slice(),
            Some("Device"),
        )?;

        let queue_family = QueueFamily::new(device.clone(), 0)?;

        let device_swapchain_info = DeviceSurfaceInfo::new(device.clone(), surface)?;

        let (frames_in_flight, swapchain_images_count) =
            SurfaceHelper::frames_in_flight(preferred_frames_in_flight, &device_swapchain_info)
                .ok_or(RenderingError::Unknown(String::from(
                    "Could not detect a compatible amount of swapchain images",
                )))?;

        let mut queues = smallvec::smallvec![];
        for index in 0..frames_in_flight {
            queues.push(Queue::new(
                queue_family.clone(),
                Some(format!("queues[{index}]").as_str()),
            )?);
        }

        let rendering_fences = (0..swapchain_images_count)
            .map(|idx| {
                Fence::new(
                    device.clone(),
                    false,
                    Some(format!("rendering_fences[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect();

        let image_available_semaphores = (0..frames_in_flight)
            .map(|idx| {
                Semaphore::new(
                    device.clone(),
                    Some(format!("image_available_semaphores[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect();

        let command_pool = CommandPool::new(queue_family.clone(), Some("My command pool")).unwrap();

        let present_command_buffers = (0..frames_in_flight)
            .map(|idx| {
                PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some(format!("present_command_buffers[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect();

        // this tells me when the present operation can start
        let present_ready = (0..swapchain_images_count)
            .map(|idx| {
                Semaphore::new(
                    device.clone(),
                    Some(format!("present_ready[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect();

        // Follow-up code creates uniform buffers to hold view and projection matrix:
        // shader expects those two to be a certain size and have no padding between them:
        // check for the layout to be correct.
        assert_eq!(std::mem::size_of::<glm::Mat4>(), 4 * 4 * 4);
        assert_eq!(std::mem::size_of::<[glm::Mat4; 2]>(), 4 * 4 * 4 * 2);

        let mut view_projection_unallocated_buffers = vec![];
        for index in 0..frames_in_flight {
            view_projection_unallocated_buffers.push(
                Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        // vkCmdUpdateBuffer counts as a trasfer operation, therefore set TrasferDst
                        [BufferUseAs::UniformBuffer, BufferUseAs::TransferDst]
                            .as_slice()
                            .into(),
                        4u64 * 4u64 * 4u64 * 2u64,
                    ),
                    None,
                    Some(format!("view_projection_buffers[{index}]").as_str()),
                )?
                .into(),
            );
        }

        let mut memory_manager = DefaultMemoryManager::new(device.clone());
        let view_projection_buffers: smallvec::SmallVec<
            [Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = memory_manager
            .allocate_resources(
                // I don't care if the memory is visible or not to the host:
                // I will use vkCmdUpdateBuffer to change memory content
                &MemoryType::device_local_and_host_visible(),
                &MemoryPoolFeatures::default(),
                view_projection_unallocated_buffers,
                MemoryManagementTags::default().with_exclusivity(true),
            )?
            .into_iter()
            .map(|r| r.buffer())
            .collect();

        let memory_manager = Arc::new(Mutex::new(memory_manager));

        let obj_manager = ResourceManager::new(
            queue_family.clone(),
            memory_manager.clone(),
            frames_in_flight,
            String::from("resource_manager"),
        )?;

        let surface = SurfaceHelper::new(swapchain_images_count, device_swapchain_info)?;

        let render_area = RenderingDimensions::new(1920, 1080);

        let view_projection_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [BindingDescriptor::new(
                [
                    ShaderStageAccessIn::Vertex,
                    ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                ]
                .as_slice()
                .into(),
                BindingType::Native(NativeBindingType::UniformBuffer),
                0,
                1,
            )]
            .as_slice(),
        )?;

        let view_projection_descriptors_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    frames_in_flight,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("view_projection_descriptors_pool"),
        )?;

        let mut view_projection_descriptor_sets = smallvec::SmallVec::<
            [Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        >::with_capacity(
            frames_in_flight as usize
        );
        for index in 0..(frames_in_flight as usize) {
            let view_proj_descriptor_set = DescriptorSet::new(
                view_projection_descriptors_pool.clone(),
                view_projection_descriptor_set_layout.clone(),
            )?;

            view_proj_descriptor_set.bind_resources(|binder: &mut DescriptorSetWriter<'_>| {
                binder
                    .bind_uniform_buffer(
                        0,
                        [(
                            view_projection_buffers[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();
            })?;

            view_projection_descriptor_sets.push(view_proj_descriptor_set);
        }

        let rt_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                // Descriptor for the whole TLAS
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                        ShaderStageAccessIn::RayTracing(
                            ShaderStageAccessInRayTracingKHR::ClosestHit,
                        ),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    0,
                    1,
                ),
                // The TLAS itself
                BindingDescriptor::new(
                    [ShaderStageAccessIn::RayTracing(
                        ShaderStageAccessInRayTracingKHR::RayGen,
                    )]
                    .as_slice()
                    .into(),
                    BindingType::AccelerationStructure(
                        AccelerationStructureBindingType::AccelerationStructure,
                    ),
                    1,
                    1,
                ),
            ]
            .as_slice(),
        )?;

        let rt_descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    MAX_DIRECTIONAL_LIGHTS * (frames_in_flight + 1),
                    0,
                    0,
                    frames_in_flight + 1,
                    0,
                    0,
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(
                        frames_in_flight + 1,
                    )),
                ),
                frames_in_flight + 1,
            ),
            Some("rt_descriptor_pool"),
        )?;

        let rt_descriptor_set = None;

        let mesh_rendering = Arc::new(MeshRendering::new(
            memory_manager.clone(),
            obj_manager.textures_descriptor_set_layout(),
            obj_manager.materials_descriptor_set_layout(),
            view_projection_descriptor_set_layout.clone(),
            &render_area,
            frames_in_flight,
        )?);

        let directional_lighting = Arc::new(DirectionalLighting::new(
            queue_family.clone(),
            &render_area,
            memory_manager.clone(),
            rt_descriptor_set_layout.clone(),
            mesh_rendering.descriptor_set_layout(),
            view_projection_descriptor_set_layout.clone(),
            frames_in_flight,
        )?);

        let global_illumination_lighting = Arc::new(GILighting::new(
            queue_family.clone(),
            &render_area,
            memory_manager.clone(),
            rt_descriptor_set_layout.clone(),
            mesh_rendering.descriptor_set_layout(),
            directional_lighting.descriptor_set_layout(),
            frames_in_flight,
        )?);

        let final_rendering = Arc::new(FinalRendering::new(
            memory_manager.clone(),
            mesh_rendering.descriptor_set_layout(),
            directional_lighting.descriptor_set_layout(),
            global_illumination_lighting.descriptor_set_layout(),
            &render_area,
            frames_in_flight,
        )?);

        let hdr = Arc::new(HDRTransform::new(
            memory_manager.clone(),
            frames_in_flight,
            &render_area,
        )?);

        let renderquad = Arc::new(RenderQuad::new(
            device.clone(),
            frames_in_flight,
            surface.final_format(),
            initial_width,
            initial_height,
        )?);

        let resources_manager = Arc::new(Mutex::new(obj_manager));
        let lights_manager = Arc::new(Mutex::new(DirectionalLights::new(
            queue_family.clone(),
            memory_manager.clone(),
            String::from("directional_lights"),
        )?));

        let frames_in_flight = (0..frames_in_flight).map(|_| Option::None).collect();

        Ok(Self {
            queue_family,

            frames_in_flight,
            window,
            queues,
            rendering_fences,

            image_available_semaphores,
            _command_pool: command_pool,
            present_command_buffers,

            current_frame: AtomicUsize::new(0),

            swapchain: None,
            present_ready,

            surface,

            view_projection_descriptor_sets,

            view_projection_buffers,

            rt_descriptor_set_layout,
            rt_descriptor_pool,
            rt_descriptor_set,

            mesh_rendering,
            directional_lighting,
            global_illumination_lighting,
            final_rendering,
            hdr,
            renderquad,

            resources_manager,
            lights_manager,
        })
    }

    pub fn recreate_swapchain(&mut self) -> RenderingResult<()> {
        let (new_width, new_height) = self.window.drawable_size();
        let new_dimensions = Image2DDimensions::new(new_width, new_height);
        let render_queue_families = [self.queue_family()];

        // create the new swapchain if none is present
        let swapchain = match self.swapchain.take() {
            Some((mut swapchain, images)) => {
                drop(images);

                match Arc::get_mut(&mut swapchain) {
                    Some(_swapchain) => {
                        // TODO: regenerate the swapchain with new dimensions
                    }
                    None => {
                        // the swapchain (or one of its images) is currently in use
                        // so warn the user about it.
                        println!("The swapchain is not ready to be regenerated.");
                    }
                }

                swapchain
            }
            None => SwapchainKHR::new(
                self.surface.device_swapchain_info(),
                render_queue_families.as_slice(),
                PresentModeSwapchainKHR::FIFO,
                self.surface.color_space(),
                CompositeAlphaSwapchainKHR::Opaque,
                SurfaceTransformSwapchainKHR::Identity,
                true,
                self.surface.final_format(),
                ImageUsage::from([ImageUseAs::TransferDst, ImageUseAs::ColorAttachment].as_slice()),
                new_dimensions,
                self.surface.images_count(),
                1,
            )?,
        };

        let mut images = SwapchainImagesType::default();
        for index in 0..self.surface.images_count() {
            images.push(SwapchainKHR::image(swapchain.clone(), index)?);
        }

        let mut image_views = SwapchainImageViewsType::default();
        for (index, image) in images.iter().enumerate() {
            let image_view_name = format!("swapchain_image_views[{index}]");
            image_views.push(ImageView::new(
                image.clone(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(image_view_name.as_str()),
            )?);
        }

        self.swapchain = Some((swapchain, image_views));

        Ok(())
    }

    pub fn render(&mut self, camera: &dyn CameraTrait, hdr: &HDR) -> RenderingResult<()> {
        // Ensure the swapchain is available and evey resource tied to is is usable
        // create the new swapchain if none is present
        if self.swapchain.is_none() {
            Self::recreate_swapchain(self)?;
        }

        // if there is still no swapchain then somethign has gone horribly wrong
        let Some((swapchain, swapchain_imageviews)) = &self.swapchain else {
            return Err(RenderingError::NotEnoughSwapchainImages);
        };

        let current_frame = self.current_frame.fetch_add(1, Ordering::SeqCst);
        let current_frame = current_frame % self.frames_in_flight.len();

        // this will ensure the previous frame in flight (relative to the same swapchain image) has completed its execution
        drop(self.frames_in_flight[current_frame].take());

        // swapchain_index is the index of the swapchain image relative to the specified swapchain
        let (swapchain_index, swapchain_optimal) = swapchain.acquire_next_image_index(
            Duration::from_nanos(u64::MAX),
            Some(self.image_available_semaphores[current_frame].clone()),
            None,
        )?;

        let camera_matrices = [
            camera.view_matrix(),
            camera.projection_matrix(
                swapchain.images_extent().width(),
                swapchain.images_extent().height(),
            ),
        ];

        {
            let mut static_meshes_resources = self.resources_manager.lock().unwrap();
            let mut directional_lighting_resources = self.lights_manager.lock().unwrap();

            static_meshes_resources.wait_nonblocking()?;
            directional_lighting_resources.wait_nonblocking()?;

            // if there is no descriptor set then no element is on the scene, therefore there is
            // simply nothing to be rendered.
            let Some(rt_descriptor_set) = &self.rt_descriptor_set else {
                return Ok(());
            };

            // here register the command buffer: command buffer at index i is associated with rendering_fences[i],
            // that I just awaited above, so thecommand buffer is surely NOT currently in use
            self.present_command_buffers[current_frame].record_one_time_submit(|recorder| {
                // Write view and projection matrices to GPU memory and wait for completion before using them to render the scene
                recorder.update_buffer(self.view_projection_buffers[current_frame].clone(), 0, camera_matrices.as_slice());
                recorder.buffer_barriers(
                    [BufferMemoryBarrier::new(
                        [PipelineStage::Transfer].as_slice().into(),
                        [MemoryAccessAs::TransferWrite].as_slice().into(),
                        [PipelineStage::AllCommands].as_slice().into(),
                        [MemoryAccessAs::ShaderRead, MemoryAccessAs::MemoryRead].as_slice().into(),
                        BufferSubresourceRange::new(self.view_projection_buffers[current_frame].clone(), 0u64, self.view_projection_buffers[current_frame].size()),
                        self.queue_family(),
                        self.queue_family(),
                    )]
                    .as_slice(),
                );

                // Record rendering commands to generate the gbuffer (position, normal and texture) for each
                // pixel in the final image: this solves the visibility problem and provides data for later stager
                // along the GPU pipeline
                let gbuffer_descriptor_set = self.mesh_rendering.record_rendering_commands(
                    self.view_projection_descriptor_sets[current_frame].clone(),
                    self.queue_family(),
                    [PipelineStage::AllGraphics, PipelineStage::RayTracingPipelineKHR(PipelineStageRayTracingPipelineKHR::RayTracingShader)].as_slice().into(),
                    [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead].as_slice().into(),
                    current_frame,
                    static_meshes_resources,
                    recorder
                );

                let dlbuffer_descriptor_set = self.directional_lighting.record_rendering_commands(
                    rt_descriptor_set.clone(),
                    gbuffer_descriptor_set.clone(),
                    self.view_projection_descriptor_sets[current_frame].clone(),
                    directional_lighting_resources.deref(),
                    current_frame,
                    [PipelineStage::AllGraphics].as_slice().into(),
                    [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead].as_slice().into(),
                    recorder
                );

                let gibuffer_descriptor_set = self.global_illumination_lighting.record_rendering_commands(
                    rt_descriptor_set.clone(),
                    gbuffer_descriptor_set.clone(),
                    dlbuffer_descriptor_set.clone(),
                    current_frame,
                    [PipelineStage::AllGraphics].as_slice().into(),
                    [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead].as_slice().into(),
                    recorder
                );

                // Record rendering commands to assemble the gbuffer and other resources into a an image
                // ready to be post-processed to add effects
                let (
                    final_rendering_output_image,
                    final_rendering_output_image_subresource_range,
                    final_rendering_output_image_layout,
                ) = self.final_rendering.record_rendering_commands(
                    self.queue_family(),
                    gbuffer_descriptor_set,
                    dlbuffer_descriptor_set,
                    gibuffer_descriptor_set,
                    current_frame,
                    recorder,
                );

                // Insert a barrier to transition image layout from the final rendering output to renderquad input
                // while also ensuring the rendering operation of final rendering pipeline has completed before initiating
                // the final renderquad step.
                recorder.image_barriers(
                    [ImageMemoryBarrier::new(
                        PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                        PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ShaderRead].as_slice()),
                        final_rendering_output_image_subresource_range,
                        final_rendering_output_image_layout,
                        RenderQuad::image_input_layout(),
                        self.queue_family(),
                        self.queue_family(),
                    )]
                    .as_slice(),
                );

                let (hdr_output_image, hdr_output_image_subresource_range, hdr_output_image_layout) =
                    self.hdr.record_rendering_commands(
                        self.queue_family(),
                        hdr,
                        final_rendering_output_image,
                        current_frame,
                        recorder,
                    );

                // Insert a barrier to transition image layout from the final rendering output to renderquad input
                // while also ensuring the rendering operation of final rendering pipeline has completed before initiating
                // the final renderquad step.
                recorder.image_barriers(
                    [ImageMemoryBarrier::new(
                        PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                        PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ShaderRead].as_slice()),
                        hdr_output_image_subresource_range,
                        hdr_output_image_layout,
                        RenderQuad::image_input_layout(),
                        self.queue_family(),
                        self.queue_family(),
                    )]
                    .as_slice(),
                );

                // Transition the final swapchain image into color attachment optimal layout,
                // so that the graphics pipeline has it in the best format, and the final barrier (*1)
                // can transition it from that layout to the one suitable for presentation on the
                // swapchain
                recorder.image_barriers(
                    [ImageMemoryBarrier::new(
                        PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                        MemoryAccess::from([].as_slice()),
                        PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                        swapchain_imageviews[swapchain_index as usize]
                            .image()
                            .into(),
                        ImageLayout::Undefined,
                        ImageLayout::ColorAttachmentOptimal,
                        self.queue_family(),
                        self.queue_family(),
                    )]
                    .as_slice(),
                );

                // record commands to finalize the rendering image
                self.renderquad.record_rendering_commands(
                    swapchain.images_extent(),
                    hdr_output_image,
                    swapchain_imageviews[swapchain_index as usize].clone(),
                    current_frame,
                    recorder,
                );

                // Final barrier (*1) for presentation:
                // wait for the renderquad to complete the rendering so that we can then transition
                // the swapchain image in a layout that is suitable for presentation on the swapchain.
                recorder.image_barriers(
                    [ImageMemoryBarrier::new(
                        PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                        MemoryAccess::from([MemoryAccessAs::ShaderWrite].as_slice()),
                        PipelineStages::from([PipelineStage::BottomOfPipe].as_slice()),
                        MemoryAccess::from([].as_slice()),
                        swapchain_imageviews[swapchain_index as usize]
                            .image()
                            .into(),
                        ImageLayout::ColorAttachmentOptimal,
                        ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                        self.queue_family(),
                        self.queue_family(),
                    )]
                    .as_slice(),
                );
            })?
        };

        let frame_queue = self.queues[current_frame].clone();

        let present_semaphore = self.present_ready[swapchain_index as usize].clone();
        let signal_semaphores = [present_semaphore.clone()];
        self.frames_in_flight[current_frame] = Some(frame_queue.submit(
            &[self.present_command_buffers[current_frame].clone()],
            &[(
                PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                self.image_available_semaphores[current_frame].clone(),
            )],
            signal_semaphores.as_slice(),
            self.rendering_fences[current_frame].clone(),
        )?);

        swapchain.queue_present(frame_queue, swapchain_index, &[present_semaphore])?;

        // the swapchain is suboptimal: recreate a new one from the currently existing one!
        if !swapchain_optimal {
            // wait for all rendering work to terminate before recreating the swapchain
            for frame_in_flight in 0..self.frames_in_flight.len() {
                drop(self.frames_in_flight[frame_in_flight].take())
            }

            // Regenerate the swapchain
            Self::recreate_swapchain(self)?;
        }

        Ok(())
    }
}
