use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use sdl2::VideoSubsystem;
use vulkan_framework::{
    command_buffer::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::{Device, DeviceOwned},
    fence::{Fence, FenceWaiter},
    framebuffer::Framebuffer,
    image::{Image2DDimensions, ImageUsage, ImageUseAs},
    image_view::ImageView,
    instance::InstanceOwned,
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::Queue,
    queue_family::{
        ConcreteQueueFamilyDescriptor, QueueFamily, QueueFamilyOwned,
        QueueFamilySupportedOperationType,
    },
    semaphore::Semaphore,
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
};

use crate::rendering::{
    MAX_FRAMES_IN_FLIGHT_NO_MALLOC, RenderingError, RenderingResult,
    pipeline::{final_rendering::FinalRendering, renderquad::RenderQuad},
    resources::ResourceManager,
    surface::SurfaceHelper,
};

type SwapchainImagesType =
    smallvec::SmallVec<[Arc<ImageSwapchainKHR>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;
type SwapchainImageViewsType = smallvec::SmallVec<[Arc<ImageView>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;
type FramebuffersType = smallvec::SmallVec<[Arc<Framebuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct System {
    swapchain: Option<(Arc<SwapchainKHR>, SwapchainImagesType, FramebuffersType)>,

    queue: Arc<vulkan_framework::queue::Queue>,
    rendering_fences: smallvec::SmallVec<[Arc<Fence>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    current_frame: AtomicUsize,

    image_available_semaphores:
        smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    _command_pool: Arc<CommandPool>,
    present_command_buffers:
        smallvec::SmallVec<[Arc<PrimaryCommandBuffer>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,
    present_ready: smallvec::SmallVec<[Arc<Semaphore>; MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>,

    surface: SurfaceHelper,

    renderquad: Arc<RenderQuad>,
    final_rendering: Arc<FinalRendering>,

    resources_manager: Arc<Mutex<ResourceManager>>,

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
        self.queue().get_parent_queue_family()
    }

    pub fn queue(&self) -> Arc<vulkan_framework::queue::Queue> {
        self.queue.clone()
    }

    pub fn new(
        app_name: String,
        video_subsystem: VideoSubsystem,
        initial_width: u32,
        initial_height: u32,
    ) -> RenderingResult<Self> {
        let mut instance_extensions = vec![];
        let mut instance_layers = vec![];

        // Enable vulkan debug utils on debug builds
        #[cfg(debug_assertions)]
        {
            instance_extensions.push(String::from("VK_EXT_debug_utils"));
            instance_layers.push(String::from("VK_LAYER_KHRONOS_validation"));
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

        let queue = Queue::new(queue_family.clone(), Some("Queue"))?;

        let device_swapchain_info = DeviceSurfaceInfo::new(device.clone(), surface)?;

        let (frames_in_flight, swapchain_images_count) =
            SurfaceHelper::frames_in_flight(2, &device_swapchain_info).ok_or(
                RenderingError::Unknown(String::from(
                    "Could not detect a compatible amount of swapchain images",
                )),
            )?;

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

        let surface = SurfaceHelper::new(swapchain_images_count, device_swapchain_info)?;

        let renderquad = Arc::new(RenderQuad::new(
            device.clone(),
            frames_in_flight,
            surface.final_format(),
            initial_width,
            initial_height,
        )?);

        let final_rendering = Arc::new(FinalRendering::new(
            device.clone(),
            1920,
            1080,
            frames_in_flight,
        )?);

        let resources_manager = Arc::new(Mutex::new(ResourceManager::new(
            queue_family.clone(),
            frames_in_flight,
        )?));

        let frames_in_flight = (0..frames_in_flight).map(|_| Option::None).collect();

        Ok(Self {
            frames_in_flight,
            window,
            queue,
            rendering_fences,

            image_available_semaphores,
            _command_pool: command_pool,
            present_command_buffers,

            current_frame: AtomicUsize::new(0),

            swapchain: None,
            present_ready,

            surface,

            renderquad,
            final_rendering,

            resources_manager,
        })
    }

    pub fn render(&mut self) -> RenderingResult<()> {
        let (new_width, new_height) = self.window.drawable_size();
        let new_dimensions = Image2DDimensions::new(new_width, new_height);
        let render_queue_families = [self.queue_family()];

        // create the new swapchain if none is present
        if self.swapchain.is_none() {
            let swapchain = SwapchainKHR::new(
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
            )?;

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

            let mut framebuffers: FramebuffersType = Default::default();
            for index in 0..self.surface.images_count() {
                let framebuffer = Framebuffer::new(
                    self.renderquad.renderpass(),
                    [image_views[index as usize].clone()].as_ref(),
                    swapchain.images_extent(),
                    swapchain.images_layers_count(),
                )?;
                framebuffers.push(framebuffer);
            }

            self.swapchain = Some((swapchain, images, framebuffers))
        }

        // if there is still no swapchain then somethign has gone horribly wrong
        let Some((swapchain, _, framebuffers)) = &self.swapchain else {
            return Err(RenderingError::NotEnoughSwapchainImages);
        };

        let current_frame = self.current_frame.fetch_add(1, Ordering::SeqCst);
        let current_frame = current_frame % self.frames_in_flight.len();

        // this will ensure the previous frame in flight has completed execution
        drop(self.frames_in_flight[current_frame].take());

        // swapchain_index is the index of the swapchain image relative to the specified swapchain
        let (swapchain_index, swapchain_optimal) = swapchain.acquire_next_image_index(
            Duration::from_nanos(u64::MAX),
            Some(self.image_available_semaphores[current_frame].clone()),
            None,
        )?;

        // here register the command buffer: command buffer at index i is associated with rendering_fences[i],
        // that I just awaited above, so thecommand buffer is surely NOT currently in use
        self.present_command_buffers[current_frame].record_commands(|recorder| {
            let (rendered_image, subresource_range, final_rendering_output_image_layout) = self
                .final_rendering
                .record_rendering_commands(current_frame, recorder);

            // Insert a barrier to transition image layout from the final rendering output to renderquad input
            // while also ensuring the rendering operation of final rendering pipeline has completed before initiating
            // the final renderquad step
            let renderquad_input_image_layout = RenderQuad::image_input_format();

            recorder.image_barrier(ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::ColorAttachmentOutput].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
                PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ShaderRead].as_slice()),
                subresource_range,
                final_rendering_output_image_layout,
                renderquad_input_image_layout,
                self.queue_family(),
                self.queue_family(),
            ));

            // record commands to finalize the rendering image
            self.renderquad.record_rendering_commands(
                rendered_image,
                framebuffers[swapchain_index as usize].clone(),
                current_frame,
                recorder,
            );
        })?;

        let present_semaphore = self.present_ready[swapchain_index as usize].clone();
        let signal_semaphores = [present_semaphore.clone()];
        self.frames_in_flight[current_frame] = Some(self.queue().submit(
            &[self.present_command_buffers[current_frame].clone()],
            &[(
                PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                self.image_available_semaphores[current_frame].clone(),
            )],
            signal_semaphores.as_slice(),
            self.rendering_fences[current_frame].clone(),
        )?);

        swapchain.queue_present(self.queue().clone(), swapchain_index, &[present_semaphore])?;

        // the swapchain is suboptimal: recreate a new one from the currently existing one!
        if !swapchain_optimal {
            // wait for all rendering work to terminate before recreating the swapchain
            for frame_in_flight in 0..self.frames_in_flight.len() {
                drop(self.frames_in_flight[frame_in_flight].take())
            }

            // TODO: recreate the swapchain
            //todo!();
        }

        Ok(())
    }
}
