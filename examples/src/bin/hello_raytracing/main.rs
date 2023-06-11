use std::sync::Arc;

use inline_spirv::*;

use vulkan_framework::{
    command_buffer::{ClearValues, ColorClearValues, CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    device::*,
    fence::{Fence, FenceWaiter},
    fragment_shader::FragmentShader,
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer,
    },
    image::{
        Image2DDimensions, ImageFormat, ImageLayout, ImageLayoutSwapchainKHR, ImageMultisampling,
        ImageUsage, ImageUsageSpecifier, Image3DDimensions,
    },
    image_view::{ImageView, ImageViewType},
    instance::*,
    memory_allocator::*,
    memory_heap::*,
    memory_pool::MemoryPool,
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStages},
    queue::*,
    queue_family::*,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass, RenderSubPassDescription,
    },
    semaphore::Semaphore,
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
    vertex_shader::VertexShader, raytracing_pipeline::RaytracingPipeline, raygen_shader::RaygenShader, miss_shader::MissShader, intersection_shader::IntersectionShader, any_hit_shader::AnyHitShader, closest_hit_shader::ClosestHitShader, callable_shader::CallableShader, binding_tables::{required_memory_type, RaytracingBindingTables},
};

const RAYGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    /*
    const vec2 resolution = vec2(imageSize(outputImage));

    ivec2 pixelCoords = ivec2(glLaunchIDEXT.xy);
    vec2 uv = (vec2(pixelCoords) + vec2(0.5)) / resolution;

    // Calculate the ray direction based on the UV coordinates
    vec3 rayDir = vec3(uv * 2.0 - 1.0, -1.0);
    rayDir.y *= -1.0; // Flip the Y-axis if needed

    // Normalize the ray direction
    rayDir = normalize(rayDir);

    // Create a ray with the origin at the camera position and the direction calculated above
    Ray ray;
    ray.origin = vec4(0.0, 0.0, 0.0, 1.0);
    ray.direction = vec4(rayDir, 0.0);

    // Trace the ray
    TraceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, 0xFF, 0, 1, 0, ray);

    // Store the output color to the image
    imageStore(outputImage, pixelCoords, vec4(ray.color.xyz, 1.0));
    */
}
"#,
    glsl, rgen, vulkan1_2,
    entry = "main"
);

const MISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    
}
"#,
    glsl, rmiss, vulkan1_2,
    entry = "main"
);

const AHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    
}
"#,
    glsl, rahit, vulkan1_2,
    entry = "main"
);

const CHIT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    
}
"#,
    glsl, rchit, vulkan1_2,
    entry = "main"
);

const CALLABLE_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    
}
"#,
    glsl, rcall, vulkan1_2,
    entry = "main"
);

const INTERSECTION_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    
}
"#,
    glsl, rint, vulkan1_2,
    entry = "main"
);

fn main() {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_window");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_3;

    let device_extensions: Vec<String> = vec![
        String::from("VK_KHR_swapchain"),
        String::from("VK_KHR_acceleration_structure"),
        String::from("VK_KHR_ray_tracing_maintenance1"),
        String::from("VK_KHR_ray_tracing_pipeline"),

        String::from("VK_KHR_buffer_device_address"),
        String::from("VK_KHR_deferred_host_operations"),
        String::from("VK_EXT_descriptor_indexing"),
        String::from("VK_KHR_spirv_1_4"),
        String::from("VK_KHR_shader_float_controls"),
    ];
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

                let device_local_memory_heap = MemoryHeap::new(
                    dev.clone(),
                    ConcreteMemoryHeapDescriptor::new(
                        MemoryType::DeviceLocal(None),
                        1024 * 1024 * 128, // 128MiB of memory!
                    ),
                )
                .unwrap();
                println!("Memory heap created! <3");
                let main_heap_size = device_local_memory_heap.total_size();

                let device_local_default_allocator = MemoryPool::new(
                    device_local_memory_heap,
                    Arc::new(StackAllocator::new(main_heap_size)),
                )
                .unwrap();

                let sbt_memory_heap = MemoryHeap::new(
                    dev.clone(),
                    ConcreteMemoryHeapDescriptor::new(
                        required_memory_type(),
                        1024 * 1024 * 32,
                    ),
                )
                .unwrap();
                println!("Memory heap created! <3");
                let main_heap_size = sbt_memory_heap.total_size();

                let sbt_default_allocator = MemoryPool::new_with_device_address(
                    sbt_memory_heap,
                    Arc::new(StackAllocator::new(main_heap_size)),
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

                let swapchain_extent = Image2DDimensions::new(WIDTH, HEIGHT);

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
                    swapchain_extent,
                    swapchain_images_count,
                    1,
                )
                .unwrap();
                println!("Swapchain created!");

                let swapchain_images = ImageSwapchainKHR::extract(swapchain.clone()).unwrap();
                println!("Swapchain images extracted!");

                let image_available_semaphores = (0..swapchain_images_count)
                    .map(|_idx| {
                        Semaphore::new(dev.clone(), false, Some("image_available_semaphores[...]"))
                            .unwrap()
                    })
                    .collect::<Vec<Arc<Semaphore>>>();

                let image_rendered_semaphores = (0..swapchain_images_count)
                    .map(|_idx| {
                        Semaphore::new(dev.clone(), false, Some("image_rendered_semaphores[...]"))
                            .unwrap()
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

                let swapchain_image_views = swapchain_images.clone()
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

                let raygen_shader = RaygenShader::new(dev.clone(), RAYGEN_SPV).unwrap();
                //let intersection_shader = IntersectionShader::new(dev.clone(), INTERSECTION_SPV).unwrap();
                let miss_shader = MissShader::new(dev.clone(), MISS_SPV).unwrap();
                //let anyhit_shader = AnyHitShader::new(dev.clone(), AHIT_SPV).unwrap();
                let closesthit_shader = ClosestHitShader::new(dev.clone(), CHIT_SPV).unwrap();
                //let callable_shader = CallableShader::new(dev.clone(), CALLABLE_SPV).unwrap();

                let pipeline = RaytracingPipeline::new(
                    pipeline_layout.clone(),
                    16,
                    raygen_shader,
                    None,
                    miss_shader,
                    None,
                    closesthit_shader,
                    None,
                    Some("raytracing_pipeline!")
                ).unwrap();

                let shader_binding_tables = swapchain_images.clone()
                    .into_iter()
                    .map(|_image_swapchain| {
                        RaytracingBindingTables::new(pipeline.clone(), sbt_default_allocator.clone()).unwrap()
                    })
                    .collect::<Vec<Arc<RaytracingBindingTables>>>();

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
                                for i in 0..(swapchain_images_count as usize) {
                                    swapchain_fence_waiters[i].wait(u64::MAX).unwrap()
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
                            recorder.bind_ray_tracing_pipeline(pipeline.clone());

                            recorder.trace_rays(
                                shader_binding_tables[current_frame % (swapchain_images_count as usize)].clone(),
                                Image3DDimensions::from(swapchain_extent)
                            );
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
