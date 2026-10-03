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
    ash::vk,
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
    image::{Image1DTrait, Image2DDimensions, Image2DTrait, ImageUsage, ImageUseAs},
    image_view::ImageView,
    instance::InstanceOwned,
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs, MemoryBarrier},
    memory_heap::MemoryType,
    memory_management::{DefaultMemoryManager, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::MemoryPoolFeatures,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    prelude::{FrameworkError, VulkanError},
    queue::{Queue, SemaphoreSignalOp, SemaphoreWaitOp},
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
            final_rendering::FinalRendering, global_illumination::GILighting,
            hdr_transform::HDRTransform, mesh_rendering::MeshRendering, renderquad::RenderQuad,
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

/// The stage after which the per-frame command buffer recording must stop.
///
/// This is used to bisect which rendering pass hangs the GPU by setting the
/// ART_RTIC_STOP_AFTER environment variable to one of: none|mesh|gi|final|hdr|all.
/// When the recording stops before the end of the pipeline the frame is neither
/// presented nor acquired from the swapchain: only the workload and its fence
/// are involved.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum StopAfter {
    None,
    Mesh,
    Gi,
    Final,
    Hdr,
    All,
}

pub struct System {
    swapchain: Option<(Arc<SwapchainKHR>, SwapchainImageViewsType)>,

    queue_family: Arc<QueueFamily>,
    queues: smallvec::SmallVec<[Arc<Queue>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    rendering_fences: smallvec::SmallVec<[Arc<Fence>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    current_frame: AtomicUsize,

    debug_stop_after: StopAfter,
    debug_no_present: bool,

    image_available_semaphores:
        smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    _command_pool: Arc<CommandPool>,
    present_command_buffers:
        smallvec::SmallVec<[Arc<PrimaryCommandBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    present_ready: smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    surface: SurfaceHelper,

    status_descriptor_sets:
        smallvec::SmallVec<[Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    view_projection_buffers:
        smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    directional_light_buffers:
        smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    status_buffers: smallvec::SmallVec<[Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    rt_descriptor_set_layout: Arc<DescriptorSetLayout>,
    rt_descriptor_pool: Arc<DescriptorPool>,
    rt_descriptor_set: Option<Arc<DescriptorSet>>,

    mesh_rendering: Arc<MeshRendering>,
    global_illumination_lighting: Arc<GILighting>,
    final_rendering: Arc<FinalRendering>,
    hdr: Arc<HDRTransform>,
    renderquad: Arc<RenderQuad>,

    active_camera: Option<Arc<dyn CameraTrait>>,
    resources_manager: Arc<Mutex<ResourceManager>>,
    lights_manager: Arc<Mutex<DirectionalLights>>,

    // if the previous frame used GI and can be reused for the current frame
    // this value will be non-zero.
    //
    // this happens when the camera has not been moved and no other object
    // has been moved/added/removed from the scene in addition to when
    // no light has been changed/added/removed.
    prev_frame_gi_reuse: u32,

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
            // Required by VK_LAYER_PRINTF_ENABLE / debugPrintfEXT instrumentation.
            String::from("VK_KHR_shader_non_semantic_info"),
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
        {
            lights
                .load(DirectionalLight::new(
                    glm::Vec3::new(-0.6, -0.98, 0.00000001),
                    glm::Vec3::new(80.2, 80.2, 80.2),
                ))
                .unwrap();

            lights
                .load(DirectionalLight::new(
                    glm::Vec3::new(0.0, -0.98, 0.6),
                    glm::Vec3::new(80.0, 80.0, 80.0),
                ))
                .unwrap();

            self.prev_frame_gi_reuse = 0;
        }

        // Update the TLAS and create a descriptor set for it:
        // this is very important as it define the geometry of the whole scene
        {
            manager.wait_blocking().unwrap();
            let (tlas, tlas_data) = manager.tlas_ready().unwrap();

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
                        .bind_storage_buffers(0, [(tlas_data.clone(), None, None)].as_slice())
                        .unwrap();
                    binder.bind_tlas(1, [tlas.clone()].as_slice()).unwrap();
                })
                .unwrap();

            self.rt_descriptor_set = Some(rt_descriptor_set);
            self.prev_frame_gi_reuse = 0;
        }
    }

    pub fn change_camera(&mut self, camera: Arc<dyn CameraTrait>) {
        self.active_camera = Some(camera);
        self.prev_frame_gi_reuse = 0;
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

        // Enable vulkan debug utils on debug builds (unless ART_RTIC_NO_VALIDATION is set)
        #[cfg(debug_assertions)]
        {
            match std::env::var("ART_RTIC_NO_VALIDATION") {
                Ok(value) if value != "0" => {
                    println!(
                        "Running WITHOUT the validation layer (ART_RTIC_NO_VALIDATION is set)"
                    );
                }
                _ => {
                    println!("Running with debugging features enabled...");
                    instance_extensions.push(String::from("VK_EXT_debug_utils"));
                    instance_layers.push(String::from("VK_LAYER_KHRONOS_validation"));
                    //instance_layers.push(String::from("VK_LAYER_RENDERDOC_Capture"));
                }
            }
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

        // Request distinct queues when available; a single-queue device uses aliases.
        // Frame synchronization must work independently of the queue count.
        let requested_queues = std::env::var("ART_RTIC_QUEUE_COUNT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(preferred_frames_in_flight.max(1));
        let queue_priorities = (0..requested_queues).map(|_| 1.0f32).collect::<Vec<_>>();

        let device = Device::new(
            surface.get_parent_instance(),
            [ConcreteQueueFamilyDescriptor::new(
                vec![
                    QueueFamilySupportedOperationType::Graphics,
                    QueueFamilySupportedOperationType::Compute,
                    QueueFamilySupportedOperationType::Transfer,
                    QueueFamilySupportedOperationType::Present(surface.clone()),
                ]
                .as_ref(),
                queue_priorities.as_slice(),
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

        // one queue per frame in flight: the driver hands out distinct queues as
        // long as the queue family has enough of them; when the family is
        // exhausted its first queue is shared instead (which is always legal:
        // the frames sharing it are simply executed in submission order)
        let mut queues: smallvec::SmallVec<[Arc<Queue>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]> =
            smallvec::smallvec![];
        for index in 0..frames_in_flight {
            match Queue::new(
                queue_family.clone(),
                Some(format!("queues[{index}]").as_str()),
            ) {
                Ok(queue) => queues.push(queue),
                Err(VulkanError::Framework(FrameworkError::TooManyQueues(_, _))) => {
                    queues.push(queues[0].clone())
                }
                Err(err) => return Err(err.into()),
            }
        }

        // the queue used for resource loading and one-time initialization work
        let main_queue = queues[0].clone();

        println!(
            "Renderer configuration: {} queue(s), {frames_in_flight} frames in flight, at least {swapchain_images_count} swapchain images",
            queue_family.max_queues().min(frames_in_flight as usize)
        );

        let rendering_fences = (0..frames_in_flight)
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
        let mut directional_lights_unallocated = vec![];
        let mut status_buffer_unallocated = vec![];
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

            directional_lights_unallocated.push(
                Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        [BufferUseAs::StorageBuffer, BufferUseAs::TransferDst]
                            .as_slice()
                            .into(),
                        4u64 * 6u64 * (MAX_DIRECTIONAL_LIGHTS as u64),
                    ),
                    None,
                    Some(format!("directional_lights[{index}]").as_str()),
                )?
                .into(),
            );

            status_buffer_unallocated.push(
                Buffer::new(
                    device.clone(),
                    ConcreteBufferDescriptor::new(
                        // vkCmdUpdateBuffer counts as a trasfer operation, therefore set TrasferDst
                        [BufferUseAs::UniformBuffer, BufferUseAs::TransferDst]
                            .as_slice()
                            .into(),
                        4u64 * 8u64,
                    ),
                    None,
                    Some(format!("status_buffers[{index}]").as_str()),
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

        let directional_light_buffers: smallvec::SmallVec<
            [Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = memory_manager
            .allocate_resources(
                // I don't care if the memory is visible or not to the host:
                // I will use vkCmdUpdateBuffer to change memory content
                &MemoryType::device_local_and_host_visible(),
                &MemoryPoolFeatures::default(),
                directional_lights_unallocated,
                MemoryManagementTags::default().with_exclusivity(true),
            )?
            .into_iter()
            .map(|r| r.buffer())
            .collect();

        let status_buffers: smallvec::SmallVec<
            [Arc<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = memory_manager
            .allocate_resources(
                // I don't care if the memory is visible or not to the host:
                // I will use vkCmdUpdateBuffer to change memory content
                &MemoryType::device_local_and_host_visible(),
                &MemoryPoolFeatures::default(),
                status_buffer_unallocated,
                MemoryManagementTags::default().with_exclusivity(true),
            )?
            .into_iter()
            .map(|r| r.buffer())
            .collect();

        let memory_manager = Arc::new(Mutex::new(memory_manager));

        let obj_manager = ResourceManager::new(
            main_queue.clone(),
            memory_manager.clone(),
            frames_in_flight,
            String::from("resource_manager"),
        )?;

        let surface = SurfaceHelper::new(swapchain_images_count, device_swapchain_info)?;

        let render_area = RenderingDimensions::new(1920, 1080);

        let status_descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            [
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Compute,
                        // view-projection is used in vertex shaders
                        ShaderStageAccessIn::Vertex,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::UniformBuffer),
                    0,
                    1,
                ),
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Compute,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::StorageBuffer),
                    1,
                    1,
                ),
                BindingDescriptor::new(
                    [
                        ShaderStageAccessIn::Fragment,
                        ShaderStageAccessIn::RayTracing(ShaderStageAccessInRayTracingKHR::RayGen),
                    ]
                    .as_slice()
                    .into(),
                    BindingType::Native(NativeBindingType::UniformBuffer),
                    2,
                    1,
                ),
            ]
            .as_slice(),
        )?;

        let status_descriptors_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolConcreteDescriptor::new(
                DescriptorPoolSizesConcreteDescriptor::new(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    frames_in_flight,
                    2 * frames_in_flight,
                    0,
                    None,
                ),
                frames_in_flight,
            ),
            Some("view_projection_descriptors_pool"),
        )?;

        let mut status_descriptor_sets = smallvec::SmallVec::<
            [Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        >::with_capacity(frames_in_flight as usize);
        for index in 0..(frames_in_flight as usize) {
            let status_descriptor_set = DescriptorSet::new(
                status_descriptors_pool.clone(),
                status_descriptor_set_layout.clone(),
            )?;

            status_descriptor_set.bind_resources(|binder: &mut DescriptorSetWriter<'_>| {
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

                // bind the directional lights buffer
                binder
                    .bind_storage_buffers(
                        1,
                        [(
                            directional_light_buffers[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();

                // bind the status buffer
                binder
                    .bind_uniform_buffer(
                        2,
                        [(
                            status_buffers[index].clone() as Arc<dyn BufferTrait>,
                            None,
                            None,
                        )]
                        .as_slice(),
                    )
                    .unwrap();
            })?;

            status_descriptor_sets.push(status_descriptor_set);
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
            status_descriptor_set_layout.clone(),
            &render_area,
        )?);

        let global_illumination_lighting = Arc::new(GILighting::new(
            queue_family.clone(),
            &render_area,
            memory_manager.clone(),
            rt_descriptor_set_layout.clone(),
            mesh_rendering.descriptor_set_layout(),
            status_descriptor_set_layout.clone(),
            obj_manager.textures_descriptor_set_layout(),
            obj_manager.materials_descriptor_set_layout(),
        )?);

        let final_rendering = Arc::new(FinalRendering::new(
            memory_manager.clone(),
            mesh_rendering.descriptor_set_layout(),
            global_illumination_lighting.descriptor_set_layout(),
            &render_area,
            frames_in_flight,
        )?);

        let hdr = Arc::new(HDRTransform::new(memory_manager.clone(), &render_area)?);

        let renderquad = Arc::new(RenderQuad::new(
            device.clone(),
            surface.final_format(),
            initial_width,
            initial_height,
        )?);

        let resources_manager = Arc::new(Mutex::new(obj_manager));
        let lights_manager = Arc::new(Mutex::new(DirectionalLights::new(
            main_queue.clone(),
            memory_manager.clone(),
            String::from("directional_lights"),
        )?));

        let init_fence = Fence::new(device.clone(), false, Some("init_fence")).unwrap();

        let init_command_buffer =
            PrimaryCommandBuffer::new(command_pool.clone(), Some("init_command_buffer"))?;

        init_command_buffer.record_one_time_submit(|recorder| {
            mesh_rendering.record_init_commands(recorder);
            global_illumination_lighting.record_init_commands(recorder);
            hdr.record_init_commands(recorder);
            renderquad.record_init_commands(recorder);
        })?;

        // the init commands do not require any synchronization: the wait for
        // this submission is what orders the resource loading and the first
        // frame against the images/buffers initialization performed here
        let init_waiter =
            main_queue.submit(&[init_command_buffer.clone()], &[], &[], init_fence.clone())?;

        let frames_in_flight = (0..frames_in_flight).map(|_| Option::None).collect();
        let prev_frame_gi_reuse = 0;

        // ART_RTIC_STOP_AFTER=none|mesh|gi|final|hdr|all is used to bisect
        // which pass of the rendering pipeline hangs the GPU
        let debug_stop_after = match std::env::var("ART_RTIC_STOP_AFTER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "none" | "empty" => StopAfter::None,
            "mesh" => StopAfter::Mesh,
            "gi" => StopAfter::Gi,
            "final" => StopAfter::Final,
            "hdr" => StopAfter::Hdr,
            _ => StopAfter::All,
        };
        println!("Renderer stage bisection stops after: {debug_stop_after:?}");

        // ART_RTIC_NO_PRESENT=1 records and submits the whole frame (acquire and
        // renderquad included) but never calls vkQueuePresentKHR: used to
        // discriminate a device lost caused by the frame workload from one
        // caused by the present operation itself. Expect the run to block on
        // acquire after all the swapchain images have been acquired.
        let debug_no_present = match std::env::var("ART_RTIC_NO_PRESENT") {
            Ok(value) if value != "0" => true,
            _ => false,
        };
        println!("Renderer no-present mode: {debug_no_present}");

        let active_camera = None;

        drop(init_waiter);

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

            debug_stop_after,
            debug_no_present,

            swapchain: None,
            present_ready,

            surface,

            status_descriptor_sets,

            view_projection_buffers,
            directional_light_buffers,
            status_buffers,

            rt_descriptor_set_layout,
            rt_descriptor_pool,
            rt_descriptor_set,

            mesh_rendering,
            global_illumination_lighting,
            final_rendering,
            hdr,
            renderquad,

            resources_manager,
            lights_manager,
            active_camera,

            prev_frame_gi_reuse,
        })
    }

    pub fn recreate_swapchain(&mut self) -> RenderingResult<()> {
        let (new_width, new_height) = self.window.drawable_size();
        if new_width == 0 || new_height == 0 {
            return Ok(());
        }
        let new_dimensions = Image2DDimensions::new(new_width, new_height);
        let render_queue_families = [self.queue_family()];

        // Render fences do not cover presentation. Finish both before releasing
        // swapchain views or recycling presentation semaphores.
        for frame in &mut self.frames_in_flight {
            drop(frame.take());
        }
        self.device().wait_idle()?;

        let swapchain = match self.swapchain.take() {
            Some((mut swapchain, image_views)) => {
                drop(image_views);
                let Some(exclusive) = Arc::get_mut(&mut swapchain) else {
                    return Err(RenderingError::Unknown(String::from(
                        "Swapchain resources are still referenced after device idle",
                    )));
                };
                exclusive.recreate_with_extent(new_dimensions)?;
                swapchain
            }
            None => {
                let info = self.surface.device_swapchain_info();
                let transform = if info.transform_supported(&SurfaceTransformSwapchainKHR::Identity)
                {
                    SurfaceTransformSwapchainKHR::Identity
                } else {
                    info.current_transform()
                };
                let alpha = [
                    CompositeAlphaSwapchainKHR::Opaque,
                    CompositeAlphaSwapchainKHR::PreMultiplied,
                    CompositeAlphaSwapchainKHR::PostMultiplied,
                    CompositeAlphaSwapchainKHR::Inherit,
                ]
                .into_iter()
                .find(|alpha| info.composite_alpha_supported(alpha))
                .ok_or_else(|| {
                    RenderingError::Unknown(String::from("No supported composite alpha mode"))
                })?;
                SwapchainKHR::new(
                    info,
                    render_queue_families.as_slice(),
                    PresentModeSwapchainKHR::FIFO,
                    self.surface.color_space(),
                    alpha,
                    transform,
                    true,
                    self.surface.final_format(),
                    ImageUsage::from([ImageUseAs::ColorAttachment].as_slice()),
                    info.image_extent(new_dimensions),
                    self.surface.images_count(),
                    1,
                )?
            }
        };

        // minImageCount is a request, not the number returned by the driver.
        self.present_ready = (0..swapchain.images_count())
            .map(|index| Semaphore::new(self.device(), Some(&format!("present_ready[{index}]"))))
            .collect::<Result<_, _>>()?;
        let mut images = SwapchainImagesType::default();
        for index in 0..swapchain.images_count() {
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

    pub fn render(&mut self, hdr: &HDR) -> RenderingResult<()> {
        let (width, height) = self.window.drawable_size();
        if width == 0 || height == 0 {
            return Ok(());
        }
        // Ensure the swapchain is available and evey resource tied to is is usable
        // create the new swapchain if none is present
        if self.swapchain.is_none() {
            Self::recreate_swapchain(self)?;
        }

        // if there is still no swapchain then somethign has gone horribly wrong
        let Some((swapchain, swapchain_imageviews)) = &self.swapchain else {
            return Err(RenderingError::NotEnoughSwapchainImages);
        };

        // if there is no camera (active viewport) then there is nothing to be rendered:
        // returning before acquiring a swapchain image (and before consuming a frame
        // counter value) avoids leaving signaled semaphores and holes in the timeline
        // semaphore value chain behind
        let Some(camera) = &self.active_camera else {
            return Ok(());
        };

        // if there is no raytracing descriptor set then there is no scene to be rendered
        let Some(rt_descriptor_set) = &self.rt_descriptor_set else {
            return Ok(());
        };

        // Only successful submissions advance the timeline; acquisition/recording
        // errors must not leave an unsignaled value for the next frame to await.
        let frame_counter = self.current_frame.load(Ordering::SeqCst);
        let current_frame = frame_counter % self.frames_in_flight.len();

        // this will ensure the previous frame in flight (relative to the same swapchain image) has completed its execution
        drop(self.frames_in_flight[current_frame].take());

        // When bisecting a GPU hang (ART_RTIC_STOP_AFTER != all) neither the
        // acquire nor the present are performed: this way any number of frames
        // can be submitted and waited on without ever involving the swapchain
        // and its semaphores.
        let full_frame = self.debug_stop_after == StopAfter::All;

        // swapchain_index is the index of the swapchain image relative to the specified swapchain
        let (swapchain_index, _swapchain_optimal) = if full_frame {
            if self.debug_no_present && frame_counter >= swapchain.images_count() as usize {
                return Err(RenderingError::Unknown(String::from(
                    "No-present diagnostic exhausted its swapchain images; refusing to block on acquire",
                )));
            }
            match swapchain.acquire_next_image_index(
                Duration::from_secs(1),
                Some(self.image_available_semaphores[current_frame].clone()),
                None,
            ) {
                Ok(image) => image,
                Err(VulkanError::Vulkan(vk::Result::TIMEOUT)) => return Ok(()),
                Err(VulkanError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR)) => {
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(err) => return Err(err.into()),
            }
        } else {
            (0, true)
        };

        let camera_matrices = [
            camera.view_matrix(),
            camera.projection_matrix(
                swapchain.images_extent().width(),
                swapchain.images_extent().height(),
            ),
        ];

        let mut frame_loading_waits = Vec::new();
        {
            let mut static_meshes_resources = self.resources_manager.lock().unwrap();
            let mut directional_lighting_resources = self.lights_manager.lock().unwrap();

            static_meshes_resources.wait_blocking()?;
            directional_lighting_resources.wait_blocking()?;
            frame_loading_waits.extend(static_meshes_resources.loading_waits());
            frame_loading_waits.extend(directional_lighting_resources.loading_waits());

            let (texture_descriptor_set, material_descriptor_set) =
                static_meshes_resources.static_mesh_descriptor_sets(current_frame);

            let directional_lights = directional_lighting_resources.deref();

            // get the number of directional lights to compute and transfer into the buffer theirs directions
            assert!(directional_lights.count() <= MAX_DIRECTIONAL_LIGHTS);
            let size_of_light = 4u64 * 6u64;

            // here register the command buffer: command buffer at index i is associated with rendering_fences[i],
            // that I just awaited above, so thecommand buffer is surely NOT currently in use
            self.present_command_buffers[current_frame].record_one_time_submit(|recorder| {
                // bisecting helper: when ART_RTIC_STOP_AFTER is "none" an empty
                // command buffer is submitted to test the submission machinery
                if self.debug_stop_after == StopAfter::None {
                    return;
                }

                // Write status (view*projection matrix and directional lights) to GPU memory and
                // wait for completion before using them to render the scene
                {
                    recorder.pipeline_barriers([
                        BufferMemoryBarrier::new(
                            [PipelineStage::TopOfPipe].as_slice().into(),
                            [].as_slice().into(),
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            BufferSubresourceRange::new(
                                self.view_projection_buffers[current_frame].clone(),
                                0u64,
                                self.view_projection_buffers[current_frame].size(),
                            ),
                            self.queue_family(),
                            self.queue_family(),
                        )
                        .into(),
                        BufferMemoryBarrier::new(
                            [PipelineStage::TopOfPipe].as_slice().into(),
                            [].as_slice().into(),
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            BufferSubresourceRange::new(
                                self.directional_light_buffers[current_frame].clone(),
                                0,
                                self.directional_light_buffers[current_frame].size(),
                            ),
                            self.queue_family.clone(),
                            self.queue_family.clone(),
                        )
                        .into(),
                        BufferMemoryBarrier::new(
                            [PipelineStage::TopOfPipe].as_slice().into(),
                            [].as_slice().into(),
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            BufferSubresourceRange::new(
                                self.status_buffers[current_frame].clone(),
                                0,
                                self.status_buffers[current_frame].size(),
                            ),
                            self.queue_family.clone(),
                            self.queue_family.clone(),
                        )
                        .into(),
                    ]);

                    recorder.update_buffer(
                        self.view_projection_buffers[current_frame].clone(),
                        0,
                        camera_matrices.as_slice(),
                    );

                    recorder.update_buffer(
                        self.status_buffers[current_frame].clone(),
                        0,
                        [self.prev_frame_gi_reuse].as_slice(),
                    );

                    let mut light_index = 0u64;
                    directional_lights.foreach(|dir_light| {
                        recorder.copy_buffer(
                            dir_light.clone(),
                            self.directional_light_buffers[current_frame].clone(),
                            [(0u64, light_index * size_of_light, size_of_light)].as_slice(),
                        );

                        light_index += 1u64;
                    });

                    let stub_buffer = (0..size_of_light).map(|_| 0u8).collect::<Vec<_>>();
                    for light_unused_index in light_index..(MAX_DIRECTIONAL_LIGHTS as u64) {
                        recorder.update_buffer(
                            self.directional_light_buffers[current_frame].clone(),
                            light_unused_index * size_of_light,
                            stub_buffer.as_slice(),
                        );
                    }

                    recorder.pipeline_barriers([
                        BufferMemoryBarrier::new(
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            [
                                PipelineStage::AllGraphics,
                                PipelineStage::ComputeShader,
                                PipelineStage::RayTracingPipelineKHR(
                                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                                ),
                            ]
                            .as_slice()
                            .into(),
                            [MemoryAccessAs::UniformRead].as_slice().into(),
                            BufferSubresourceRange::new(
                                self.view_projection_buffers[current_frame].clone(),
                                0u64,
                                self.view_projection_buffers[current_frame].size(),
                            ),
                            self.queue_family(),
                            self.queue_family(),
                        )
                        .into(),
                        BufferMemoryBarrier::new(
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            [
                                PipelineStage::AllGraphics,
                                PipelineStage::RayTracingPipelineKHR(
                                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                                ),
                            ]
                            .as_slice()
                            .into(),
                            [
                                MemoryAccessAs::MemoryRead,
                                MemoryAccessAs::ShaderRead,
                                MemoryAccessAs::UniformRead,
                            ]
                            .as_slice()
                            .into(),
                            BufferSubresourceRange::new(
                                self.directional_light_buffers[current_frame].clone(),
                                0,
                                self.directional_light_buffers[current_frame].size(),
                            ),
                            self.queue_family.clone(),
                            self.queue_family.clone(),
                        )
                        .into(),
                        BufferMemoryBarrier::new(
                            [PipelineStage::Transfer].as_slice().into(),
                            [MemoryAccessAs::TransferWrite].as_slice().into(),
                            [
                                PipelineStage::AllGraphics,
                                PipelineStage::RayTracingPipelineKHR(
                                    PipelineStageRayTracingPipelineKHR::RayTracingShader,
                                ),
                            ]
                            .as_slice()
                            .into(),
                            [MemoryAccessAs::UniformRead].as_slice().into(),
                            BufferSubresourceRange::new(
                                self.status_buffers[current_frame].clone(),
                                0,
                                self.status_buffers[current_frame].size(),
                            ),
                            self.queue_family.clone(),
                            self.queue_family.clone(),
                        )
                        .into(),
                    ]);
                }

                // Record rendering commands to generate the gbuffer (position, normal and texture) for each
                // pixel in the final image: this solves the visibility problem and provides data for later stager
                // along the GPU pipeline
                let gbuffer_descriptor_set = self.mesh_rendering.record_rendering_commands(
                    self.status_descriptor_sets[current_frame].clone(),
                    self.queue_family(),
                    [
                        PipelineStage::FragmentShader,
                        PipelineStage::ComputeShader,
                        PipelineStage::RayTracingPipelineKHR(
                            PipelineStageRayTracingPipelineKHR::RayTracingShader,
                        ),
                    ]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::ShaderRead].as_slice().into(),
                    current_frame,
                    static_meshes_resources,
                    recorder,
                );

                if self.debug_stop_after == StopAfter::Mesh {
                    return;
                }

                // Upload semaphore waits provide the cross-queue dependency;
                // include actual producers for any work recorded on this queue.
                recorder.pipeline_barriers([MemoryBarrier::new(
                    [PipelineStage::AllCommands].as_slice().into(),
                    [MemoryAccessAs::MemoryWrite].as_slice().into(),
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [
                        MemoryAccessAs::MemoryRead,
                        MemoryAccessAs::ShaderRead,
                        MemoryAccessAs::AccelerationStructureRead,
                    ]
                    .as_slice()
                    .into(),
                )
                .into()]);

                let gibuffer_descriptor_set =
                    self.global_illumination_lighting.record_rendering_commands(
                        self.prev_frame_gi_reuse,
                        rt_descriptor_set.clone(),
                        gbuffer_descriptor_set.clone(),
                        self.status_descriptor_sets[current_frame].clone(),
                        texture_descriptor_set,
                        material_descriptor_set,
                        [PipelineStage::FragmentShader].as_slice().into(),
                        [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead]
                            .as_slice()
                            .into(),
                        recorder,
                    );

                if self.debug_stop_after == StopAfter::Gi {
                    return;
                }

                // make resources available for ray tracing pipeline(s)
                recorder.pipeline_barriers([MemoryBarrier::new(
                    [PipelineStage::RayTracingPipelineKHR(
                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::MemoryWrite].as_slice().into(),
                    [PipelineStage::AllGraphics].as_slice().into(),
                    [MemoryAccessAs::MemoryRead, MemoryAccessAs::ShaderRead]
                        .as_slice()
                        .into(),
                )
                .into()]);

                // Record rendering commands to assemble the gbuffer and other resources into a an image
                // ready to be post-processed to add effects
                let final_rendering_output_image = self.final_rendering.record_rendering_commands(
                    self.queue_family(),
                    gbuffer_descriptor_set,
                    gibuffer_descriptor_set,
                    current_frame,
                    recorder,
                );

                if self.debug_stop_after == StopAfter::Final {
                    return;
                }

                let hdr_output_image = self.hdr.record_rendering_commands(
                    self.queue_family(),
                    hdr,
                    final_rendering_output_image,
                    recorder,
                );

                if self.debug_stop_after == StopAfter::Hdr {
                    return;
                }

                // record commands to finalize the rendering image
                self.renderquad.record_rendering_commands(
                    self.queue_family(),
                    swapchain.images_extent(),
                    hdr_output_image,
                    swapchain_imageviews[swapchain_index as usize].clone(),
                    recorder,
                );
            })?
        };

        let frame_queue = self.queues[current_frame].clone();

        let present_semaphore = self.present_ready[swapchain_index as usize].clone();

        // The ordering between the passes of this frame (mesh rendering, global
        // illumination, final rendering, HDR transform, renderquad) is already
        // enforced by the order of the commands in the command buffer together
        // with the pipeline barriers that have been recorded: semaphores are
        // only required to synchronize operations that are external to this
        // single submission, namely the swapchain image acquire/present and
        // the previous frame (which shares the gbuffer, the global
        // illumination and the HDR images with this one).
        //
        // Frame N waits on the timeline payload value N (signaled when the
        // commands of frame N-1 terminated) and signals the payload value N+1
        // for the frame that follows: a timeline semaphore is what makes
        // waiting and signaling in the same submission legal, no matter how
        // many queues the frames are submitted to.
        let gi_reuse_timeline = self.global_illumination_lighting.reuse_timeline();
        let timeline_wait_value = frame_counter as u64;

        let mut wait_semaphores = frame_loading_waits;
        if full_frame {
            // the swapchain image has to be acquired before it can be used
            wait_semaphores.push(SemaphoreWaitOp::Binary(
                PipelineStages::from([PipelineStage::AllCommands].as_slice()),
                self.image_available_semaphores[current_frame].clone(),
            ));
        }

        // wait for the whole previous frame to be over: its commands are
        // the producers of the data this frame reuses
        wait_semaphores.push(SemaphoreWaitOp::Timeline(
            PipelineStages::from([PipelineStage::AllCommands].as_slice()),
            gi_reuse_timeline.clone(),
            timeline_wait_value,
        ));

        let mut signal_semaphores = vec![SemaphoreSignalOp::Timeline(
            gi_reuse_timeline,
            timeline_wait_value + 1u64,
        )];
        if full_frame && !self.debug_no_present {
            signal_semaphores.insert(0, SemaphoreSignalOp::Binary(present_semaphore.clone()));
        }

        self.frames_in_flight[current_frame] = Some(frame_queue.submit_mixed(
            &[self.present_command_buffers[current_frame].clone()],
            wait_semaphores.as_slice(),
            signal_semaphores.as_slice(),
            self.rendering_fences[current_frame].clone(),
        )?);

        self.current_frame.fetch_add(1, Ordering::SeqCst);

        if full_frame && !self.debug_no_present {
            let present_out_of_date = match swapchain.queue_present(
                frame_queue,
                swapchain_index,
                &[present_semaphore],
            ) {
                Ok(_) => false,
                Err(VulkanError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR)) => true,
                Err(err) => return Err(err.into()),
            };
            let (drawable_width, drawable_height) = self.window.drawable_size();
            let extent_changed = swapchain.images_extent()
                != Image2DDimensions::new(drawable_width, drawable_height);
            // Suboptimal images remain presentable. Recreate only when the
            // drawable size changed or the swapchain is actually out of date.
            if present_out_of_date || extent_changed {
                self.recreate_swapchain()?;
            }
        }

        // The GI has been calculated for this frame: try to reuse it for the next frame
        self.prev_frame_gi_reuse = self.prev_frame_gi_reuse.saturating_add(1);

        Ok(())
    }
}
