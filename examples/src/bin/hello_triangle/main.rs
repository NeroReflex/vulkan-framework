use std::sync::Arc;

use inline_spirv::*;

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::*,
    fence::{Fence, FenceWaiter},
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer,
    },
    image::{
        Image2DDimensions, ImageFormat, ImageLayout, ImageLayoutSwapchainKHR, ImageMultisampling,
        ImageUsage, ImageUsageSpecifier,
    },
    image_view::{ImageView, ImageViewType},
    instance::*,
    memory_allocator::*,
    memory_heap::*,
    memory_pool::{MemoryPool, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::*,
    queue_family::*,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass,
        RenderSubPassDescription,
    },
    semaphore::Semaphore,
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
    shaders::{
        vertex_shader::VertexShader,
        fragment_shader::FragmentShader,
    }
};

const VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
"#,
    vert
);

const FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"#,
    frag
);

fn main() {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_window");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];
    let device_layers: Vec<String> = vec![];

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 800;

    {
        // this parenthesis contains the window handle and closes before calling deinitialize(). This is important as Window::drop() MUST be called BEFORE calling deinitialize()!!!
        // create a sdl2 window
        match video_subsystem
            .window("Window", WIDTH, HEIGHT)
            .vulkan()
            .build()
        {
            Ok(window) => {
                println!("SDL2 window created");

                match window.vulkan_instance_extensions() {
                    Ok(required_extensions) => {
                        println!("To present frames in this window the following vulkan instance extensions must be enabled: ");

                        for (required_instance_extension_index, required_instance_extension_name) in
                            required_extensions.iter().enumerate()
                        {
                            instance_extensions
                                .push(String::from(*required_instance_extension_name));
                            println!(
                                "    {}) {}",
                                required_instance_extension_index,
                                *required_instance_extension_name
                            );
                        }
                    }
                    Err(err) => {
                        println!(
                            "Error fetching required vulkan instance extensions: {}",
                            err
                        );
                    }
                }

                let instance = Instance::new(
                    [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
                    instance_extensions.as_slice(),
                    &engine_name,
                    &app_name,
                    &api_version,
                )
                .unwrap();
                println!("Vulkan instance created");

                let sfc = vulkan_framework::surface::Surface::from_raw(
                    instance.clone(),
                    window
                        .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
                        .unwrap(),
                )
                .unwrap();
                println!("Vulkan rendering surface created and registered successfully");

                let dev = Device::new(
                    instance,
                    [ConcreteQueueFamilyDescriptor::new(
                        vec![
                            QueueFamilySupportedOperationType::Graphics,
                            QueueFamilySupportedOperationType::Transfer,
                            QueueFamilySupportedOperationType::Present(sfc.clone()),
                        ]
                        .as_ref(),
                        [1.0f32].as_slice(),
                    )]
                    .as_slice(),
                    device_extensions.as_slice(),
                    device_layers.as_slice(),
                    Some("Opened Device"),
                )
                .unwrap();
                println!("Device opened successfully");

                let queue_family = QueueFamily::new(dev.clone(), 0).unwrap();

                println!("Base queue family obtained successfully from Device");

                let queue = Queue::new(queue_family.clone(), Some("best queua evah")).unwrap();
                println!("Queue created successfully");

                let memory_heap = MemoryHeap::new(
                    dev.clone(),
                    ConcreteMemoryHeapDescriptor::new(
                        MemoryType::DeviceLocal(None),
                        1024 * 1024 * 128, // 128MiB of memory!
                    ),
                )
                .unwrap();
                println!("Memory heap created! <3");
                let memory_heap_size = memory_heap.total_size();

                let _default_allocator = MemoryPool::new(
                    memory_heap,
                    Arc::new(StackAllocator::new(memory_heap_size)),
                    MemoryPoolFeatures::from(&[]),
                )
                .unwrap();

                let device_swapchain_info = DeviceSurfaceInfo::new(dev.clone(), sfc).unwrap();

                if !device_swapchain_info.present_mode_supported(&PresentModeSwapchainKHR::FIFO) {
                    panic!("Device does not support the most common present mode. LOL.");
                }

                let final_format = ImageFormat::b8g8r8a8_srgb;
                let color_space = SurfaceColorspaceSwapchainKHR::SRGBNonlinear;
                if !device_swapchain_info.format_supported(&color_space, &final_format) {
                    panic!("Device does not support the most common format. LOL.");
                }

                let mut swapchain_images_count = device_swapchain_info.min_image_count() + 2;

                if !device_swapchain_info.image_count_supported(swapchain_images_count) {
                    println!("Image count {} not supported (the maximum is {}), sticking with the minimum one: {}", swapchain_images_count, device_swapchain_info.max_image_count(), device_swapchain_info.min_image_count());
                    swapchain_images_count = device_swapchain_info.min_image_count();
                }

                let swapchain = SwapchainKHR::new(
                    &device_swapchain_info,
                    &[queue_family.clone()],
                    None,
                    PresentModeSwapchainKHR::FIFO,
                    color_space,
                    CompositeAlphaSwapchainKHR::Opaque,
                    SurfaceTransformSwapchainKHR::Identity,
                    true,
                    final_format,
                    ImageUsage::Managed(ImageUsageSpecifier::new(
                        false, true, false, false, true, false, false, false,
                    )),
                    Image2DDimensions::new(WIDTH, HEIGHT),
                    swapchain_images_count,
                    1,
                )
                .unwrap();
                println!("Swapchain created!");

                let swapchain_images = ImageSwapchainKHR::extract(swapchain.clone()).unwrap();
                println!("Swapchain images extracted!");

                let image_available_semaphores = (0..swapchain_images_count)
                    .map(|_idx| {
                        Semaphore::new(dev.clone(), Some("image_available_semaphores[...]"))
                            .unwrap()
                    })
                    .collect::<Vec<Arc<Semaphore>>>();

                let image_rendered_semaphores = (0..swapchain_images_count)
                    .map(|_idx| {
                        Semaphore::new(dev.clone(), Some("image_rendered_semaphores[...]")).unwrap()
                    })
                    .collect::<Vec<Arc<Semaphore>>>();

                let swapchain_fences = (0..swapchain_images_count)
                    .map(|_idx| {
                        Fence::new(dev.clone(), true, Some("swapchain_fences[...]")).unwrap()
                    })
                    .collect::<Vec<Arc<Fence>>>();

                let mut swapchain_fence_waiters = (0..swapchain_images_count)
                    .map(|idx| FenceWaiter::from_fence(swapchain_fences[idx as usize].clone()))
                    .collect::<Vec<FenceWaiter>>();

                let command_pool = CommandPool::new(queue_family, Some("My command pool")).unwrap();

                let _command_buffer =
                    PrimaryCommandBuffer::new(command_pool.clone(), Some("my command buffer <3"))
                        .unwrap();

                let present_command_buffers = (0..swapchain_images_count)
                    .map(|_idx| {
                        PrimaryCommandBuffer::new(
                            command_pool.clone(),
                            Some("present_command_buffers[0]"),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<PrimaryCommandBuffer>>>();

                let renderpass = RenderPass::new(
                    dev.clone(),
                    &[
                        AttachmentDescription::new(
                            final_format,
                            ImageMultisampling::SamplesPerPixel1,
                            ImageLayout::Undefined,
                            ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                            AttachmentLoadOp::Clear,
                            AttachmentStoreOp::Store,
                            AttachmentLoadOp::Clear,
                            AttachmentStoreOp::Store,
                        ),
                        /*
                        // depth
                        AttachmentDescription::new(
                            final_format,
                            ImageMultisampling::SamplesPerPixel1,
                            ImageLayout::Undefined,
                            ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                            AttachmentLoadOp::Clear,
                            AttachmentStoreOp::Store,
                            AttachmentLoadOp::Clear,
                            AttachmentStoreOp::Store,
                        )*/
                    ],
                    &[RenderSubPassDescription::new(&[], &[0], None)],
                )
                .unwrap();
                println!("Renderpass created!");

                let pipeline_layout = PipelineLayout::new(
                    dev.clone(),
                    &[
                        /*BindingDescriptor::new(
                            shader_access,
                            binding_type,
                            binding_point,
                            binding_count
                        )*/
                    ],
                    &[],
                    Some("pipeline_layout"),
                )
                .unwrap();
                println!("Pipeline layout created!");

                let vertex_shader = VertexShader::new(dev.clone(), &[], &[], VERTEX_SPV).unwrap();

                let fragment_shader = FragmentShader::new(dev, &[], &[], FRAGMENT_SPV).unwrap();

                let graphics_pipeline = GraphicsPipeline::new(
                    renderpass.clone(),
                    0,
                    ImageMultisampling::SamplesPerPixel1,
                    Some(DepthConfiguration::new(
                        true,
                        DepthCompareOp::Always,
                        Some((0.0, 1.0)),
                    )),
                    Image2DDimensions::new(WIDTH, HEIGHT),
                    pipeline_layout,
                    &[
                        /*VertexInputBinding::new(
                        VertexInputRate::PerVertex,
                        0,
                        &[VertexInputAttribute::new(0, 0, AttributeType::Vec4)],
                        )*/
                    ],
                    Rasterizer::new(
                        PolygonMode::Fill,
                        FrontFace::CounterClockwise,
                        CullMode::None,
                        None,
                    ),
                    (vertex_shader, None),
                    (fragment_shader, None),
                    Some("triangle_pipeline_<3"),
                )
                .unwrap();
                println!("Graphics pipeline created!");

                let swapchain_image_views = swapchain_images
                    .into_iter()
                    .map(|image_swapchain| {
                        ImageView::new(
                            image_swapchain,
                            ImageViewType::Image2D,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            Some("swapchain_image_views[...]"),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<ImageView>>>();

                let swapchain_framebuffers = swapchain_image_views
                    .iter()
                    .map(|iv| {
                        Framebuffer::new(
                            renderpass.clone(),
                            &[iv.clone()],
                            swapchain.images_extent(),
                            1,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<Framebuffer>>>();

                let mut current_frame: usize = 0;

                let mut event_pump = sdl_context.event_pump().unwrap();
                'running: loop {
                    for event in event_pump.poll_iter() {
                        match event {
                            sdl2::event::Event::Quit { .. }
                            | sdl2::event::Event::KeyDown {
                                keycode: Some(sdl2::keyboard::Keycode::Escape),
                                ..
                            } => {
                                for fence_waiter in swapchain_fence_waiters.iter_mut() {
                                    fence_waiter.wait(u64::MAX).unwrap()
                                }
                                break 'running;
                            }
                            _ => {}
                        }
                    }

                    let swapchain_index = swapchain
                        .acquire_next_image_index(
                            None,
                            Some(
                                image_available_semaphores
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                            ),
                            None,
                        )
                        .unwrap();

                    // wait for fence
                    swapchain_fence_waiters[current_frame % (swapchain_images_count as usize)]
                        .wait(u64::MAX)
                        .unwrap();

                    present_command_buffers[current_frame % (swapchain_images_count as usize)]
                        .record_commands(|recorder: &mut CommandBufferRecorder| {
                            recorder.begin_renderpass(
                                swapchain_framebuffers
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                                &[ClearValues::new(Some(ColorClearValues::Vec4(
                                    0.0, 0.0, 0.0, 1.0,
                                )))],
                            );

                            recorder.bind_graphics_pipeline(graphics_pipeline.clone());

                            recorder.draw(0, 3, 0, 1);

                            recorder.end_renderpass();
                        })
                        .unwrap();

                    swapchain_fence_waiters[current_frame % (swapchain_images_count as usize)] =
                        queue
                            .submit(
                                &[present_command_buffers
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone()],
                                &[(
                                    PipelineStages::from(
                                        &[PipelineStage::FragmentShader],
                                        None,
                                        None,
                                        None,
                                    ),
                                    image_available_semaphores
                                        [current_frame % (swapchain_images_count as usize)]
                                        .clone(),
                                )],
                                &[image_rendered_semaphores
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone()],
                                swapchain_fences[current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                            )
                            .unwrap();

                    swapchain
                        .queue_present(
                            queue.clone(),
                            swapchain_index,
                            &[image_rendered_semaphores
                                [current_frame % (swapchain_images_count as usize)]
                                .clone()],
                        )
                        .unwrap();

                    current_frame = swapchain_index as usize;
                }
            }
            Err(err) => {
                println!("Error creating sdl2 window: {}", err);
            }
        }
    }
}
