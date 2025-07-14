use std::{sync::Arc, time::Duration};

use inline_spirv::*;

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::*,
    fence::{Fence, FenceWaitFor},
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, Scissor, Viewport,
    },
    image::{
        Image2DDimensions, ImageFormat, ImageLayout, ImageLayoutSwapchainKHR, ImageMultisampling,
        ImageUsage, ImageUsageSpecifier,
    },
    image_view::{ImageView, ImageViewType},
    instance::*,
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::*,
    queue_family::*,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass,
        RenderSubPassDescription,
    },
    semaphore::Semaphore,
    shaders::{fragment_shader::FragmentShader, vertex_shader::VertexShader},
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
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

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];
    let device_layers: Vec<String> = vec![];

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 800;

    {
        // this parenthesis contains the window handle and closes before calling deinitialize(). This is important as Window::drop() MUST be called BEFORE calling deinitialize()!!!
        // create a sdl2 window
        let window = match video_subsystem
            .window("Window", WIDTH, HEIGHT)
            .vulkan()
            .build()
        {
            Ok(window) => {
                println!("SDL2 window created");
                window
            }
            Err(err) => panic!("Error creating sdl2 window: {err}"),
        };
        match window.vulkan_instance_extensions() {
            Ok(required_extensions) => {
                println!("To present frames in this window the following vulkan instance extensions must be enabled: ");

                for (required_instance_extension_index, required_instance_extension_name) in
                    required_extensions.iter().enumerate()
                {
                    instance_extensions.push(String::from(*required_instance_extension_name));
                    println!(
                        "    {}) {}",
                        required_instance_extension_index, *required_instance_extension_name
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
        let frames_in_flight = device_swapchain_info.min_image_count() as usize;
        if !device_swapchain_info.image_count_supported(swapchain_images_count) {
            println!("Image count {swapchain_images_count} not supported (the maximum is {}), sticking with the minimum one: {}", device_swapchain_info.max_image_count(), device_swapchain_info.min_image_count());
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

        let image_available_semaphores = (0..frames_in_flight)
            .map(|idx| {
                Semaphore::new(
                    dev.clone(),
                    Some(format!("image_available_semaphores[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Semaphore>>>();

        // this tells me when the present operation can start
        let present_ready = (0..swapchain_images_count)
            .map(|idx| {
                Semaphore::new(dev.clone(), Some(format!("present_ready[{idx}]").as_str())).unwrap()
            })
            .collect::<Vec<Arc<Semaphore>>>();

        let swapchain_fences = (0..swapchain_images_count)
            .map(|idx| {
                Fence::new(
                    dev.clone(),
                    true,
                    Some(format!("swapchain_fences[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Fence>>>();

        let command_pool = CommandPool::new(queue_family, Some("My command pool")).unwrap();

        let present_command_buffers = (0..swapchain_images_count)
            .map(|idx| {
                PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some(format!("present_command_buffers[{idx}]").as_str()),
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

        let fragment_shader = FragmentShader::new(dev.clone(), &[], &[], FRAGMENT_SPV).unwrap();

        let graphics_pipeline = GraphicsPipeline::new(
            None,
            renderpass.clone(),
            0,
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            None,
            None,
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
            .enumerate()
            .map(|(idx, image_swapchain)| {
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
                    Some(format!("swapchain_image_views[{idx}]").as_str()),
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
                        for fence_waiter in swapchain_fences.iter() {
                            Fence::wait_for_fences(
                                &[fence_waiter.clone()],
                                FenceWaitFor::All,
                                Duration::from_nanos(u64::MAX),
                            )
                            .unwrap();
                            fence_waiter.reset().unwrap();
                        }
                        dev.clone().wait_idle().unwrap();
                        break 'running;
                    }
                    _ => {}
                }
            }

            Fence::wait_for_fences(
                &[swapchain_fences[current_frame].clone()],
                FenceWaitFor::All,
                Duration::from_nanos(u64::MAX),
            )
            .unwrap();

            let (swapchain_index, _swapchain_optimal) = swapchain
                .acquire_next_image_index(
                    Duration::from_nanos(u64::MAX),
                    Some(image_available_semaphores[current_frame].clone()),
                    None,
                )
                .unwrap();

            swapchain_fences[current_frame].reset().unwrap();

            present_command_buffers[swapchain_index as usize]
                .record_commands(|recorder: &mut CommandBufferRecorder| {
                    recorder.begin_renderpass(
                        swapchain_framebuffers[swapchain_index as usize].clone(),
                        &[ClearValues::new(Some(ColorClearValues::Vec4(
                            0.0, 0.0, 0.0, 1.0,
                        )))],
                    );

                    recorder.bind_graphics_pipeline(
                        graphics_pipeline.clone(),
                        Some(Viewport::new(
                            0.0f32,
                            0.0f32,
                            WIDTH as f32,
                            HEIGHT as f32,
                            0.0f32,
                            0.0f32,
                        )),
                        Some(Scissor::new(0, 0, Image2DDimensions::new(WIDTH, HEIGHT))),
                    );

                    recorder.draw(0, 3, 0, 1);

                    recorder.end_renderpass();
                })
                .unwrap();

            queue
                .submit(
                    &[present_command_buffers[swapchain_index as usize].clone()],
                    &[(
                        PipelineStages::from(&[PipelineStage::FragmentShader], None, None, None),
                        image_available_semaphores[current_frame].clone(),
                    )],
                    &[present_ready[swapchain_index as usize].clone()],
                    swapchain_fences[current_frame].clone(),
                )
                .unwrap();

            swapchain
                .queue_present(
                    queue.clone(),
                    swapchain_index,
                    &[present_ready[swapchain_index as usize].clone()],
                )
                .unwrap();

            current_frame = (current_frame + 1) % frames_in_flight;
        }
    }
}
