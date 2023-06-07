use inline_spirv::*;

use vulkan_framework::{
    device::*,
    fragment_shader::FragmentShader,
    graphics_pipeline::{
        AttributeType, CullMode, FrontFace, GraphicsPipeline, PolygonMode, Rasterizer,
        VertexInputAttribute, VertexInputBinding, VertexInputRate,
    },
    image::{
        ConcreteImageDescriptor, Image, Image2DDimensions, ImageDimensions, ImageFlags,
        ImageFormat, ImageLayout, ImageLayoutSwapchainKHR, ImageMultisampling, ImageTiling,
        ImageUsage, ImageUsageSpecifier,
    },
    instance::*,
    memory_allocator::*,
    memory_heap::*,
    memory_pool::MemoryPool,
    pipeline_layout::PipelineLayout,
    queue::*,
    queue_family::*,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderPass, RenderSubPass,
    },
    shader_layout_binding::BindingDescriptor,
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
    vertex_shader::VertexShader,
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
                    instance.clone(),
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
                        1024 * 1024 * 1024 * 2, // 2GB of memory!
                    ),
                )
                .unwrap();
                println!("Memory heap created! <3");

                let default_allocator = MemoryPool::new(
                    memory_heap,
                    StackAllocator::new(
                        1024 * 1024 * 128, // 128MiB
                    ),
                )
                .unwrap();

                let device_swapchain_info =
                    DeviceSurfaceInfo::new(dev.clone(), sfc.clone()).unwrap();

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
                    &[RenderSubPass::from(&[], &[0], None)],
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

                let fragment_shader =
                    FragmentShader::new(dev.clone(), &[], &[], FRAGMENT_SPV).unwrap();

                let graphics_pipeline = GraphicsPipeline::new(
                    renderpass,
                    0,
                    ImageMultisampling::SamplesPerPixel1,
                    Image2DDimensions::new(WIDTH, HEIGHT),
                    pipeline_layout,
                    &[VertexInputBinding::new(
                        VertexInputRate::PerVertex,
                        0,
                        &[VertexInputAttribute::new(0, 0, AttributeType::Vec4)],
                    )],
                    Rasterizer::new(
                        PolygonMode::Fill,
                        FrontFace::CounterClockwise,
                        CullMode::None,
                        false,
                        0.0,
                        None,
                        0.0,
                    ),
                    (vertex_shader, None),
                    (fragment_shader, None),
                    Some("triangle_pipeline_<3"),
                )
                .unwrap();
                println!("Graphics pipeline created!");
            }
            Err(err) => {
                println!("Error creating sdl2 window: {}", err);
            }
        }
    }
}
