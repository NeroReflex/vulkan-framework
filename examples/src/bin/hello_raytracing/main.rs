use std::sync::Arc;

use inline_spirv::*;

use vulkan_framework::{
    buffer::{Buffer, ConcreteBufferDescriptor, BufferUsage},
    acceleration_structure::{
        BottomLevelAccelerationStructure,
        DeviceScratchBuffer, HostScratchBuffer, AllowedBuildingDevice, BottomLevelTrianglesGroupDecl, BottomLevelTrianglesGroupData,
    },
    binding_tables::{required_memory_type, RaytracingBindingTables},
    closest_hit_shader::ClosestHitShader,
    command_buffer::{
        AccessFlag, AccessFlags, AccessFlagsSpecifier, ClearValues, ColorClearValues,
        CommandBufferRecorder, ImageMemoryBarrier, PrimaryCommandBuffer,
    },
    command_pool::CommandPool,
    descriptor_set_layout::DescriptorSetLayout,
    device::*,
    fence::{Fence, FenceWaiter},
    fragment_shader::FragmentShader,
    framebuffer::Framebuffer,
    graphics_pipeline::{
        CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
        Rasterizer, AttributeType,
    },
    image::{
        ConcreteImageDescriptor, Image2DDimensions, Image3DDimensions, ImageDimensions, ImageFlags,
        ImageFormat, ImageLayout, ImageLayoutSwapchainKHR, ImageMultisampling, ImageTiling,
        ImageUsage, ImageUsageSpecifier,
    },
    image_view::{ImageView, ImageViewType},
    instance::*,
    memory_allocator::*,
    memory_heap::*,
    memory_pool::{MemoryPool, MemoryPoolFeature, MemoryPoolFeatures, MemoryPoolBacked},
    miss_shader::MissShader,
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue::*,
    queue_family::*,
    raygen_shader::RaygenShader,
    raytracing_pipeline::RaytracingPipeline,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderSubPassDescription,
    },
    semaphore::Semaphore,
    shader_layout_binding::{BindingDescriptor, BindingType, NativeBindingType},
    shader_stage_access::ShaderStageAccess,
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
    vertex_shader::VertexShader, prelude::VulkanError,
};

use vulkan_framework::descriptor_pool::DescriptorPool;
use vulkan_framework::descriptor_pool::DescriptorPoolConcreteDescriptor;
use vulkan_framework::descriptor_pool::DescriptorPoolSizesConcreteDescriptor;
use vulkan_framework::descriptor_set::DescriptorSet;

const RAYGEN_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

uniform layout(binding=0, set = 0, rgba32f) writeonly image2D outputImage;

//layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;

void main() {
    const vec2 resolution = vec2(imageSize(outputImage));

    ivec2 pixelCoords = ivec2(gl_LaunchIDEXT.xy);

    vec4 output_color = vec4(1.0, 0.0, 0.0, 0.0);

    // Store the output color to the image
    imageStore(outputImage, pixelCoords, vec4(output_color.xyz, 1.0));

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
    glsl,
    rgen,
    vulkan1_2,
    entry = "main"
);

const MISS_SPV: &[u32] = inline_spirv!(
    r#"
#version 460
#extension GL_EXT_ray_tracing : require

uniform layout(binding=0, set = 0, rgba32f) writeonly image2D someImage;

//layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;

void main() {
    
}
"#,
    glsl,
    rmiss,
    vulkan1_2,
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
    glsl,
    rchit,
    vulkan1_2,
    entry = "main"
);

const RENDERQUAD_VERTEX_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

layout (location = 0) out vec2 out_vTextureUV;

const vec2 vQuadPosition[6] = {
	vec2(-1, -1),
	vec2(+1, -1),
	vec2(-1, +1),
	vec2(-1, +1),
	vec2(+1, +1),
	vec2(+1, -1),
};

const vec2 vUVCoordinates[6] = {
    vec2(0.0, 0.0),
	vec2(1.0, 0.0),
	vec2(0.0, 1.0),
	vec2(0.0, 1.0),
	vec2(1.0, 1.0),
	vec2(1.0, 0.0),
};

void main() {
    out_vTextureUV = vUVCoordinates[gl_VertexIndex];
    gl_Position = vec4(vQuadPosition[gl_VertexIndex], 0.0, 1.0);
}
"#,
    glsl,
    vert,
    vulkan1_0,
    entry = "main"
);

const RENDERQUAD_FRAGMENT_SPV: &[u32] = inline_spirv!(
    r#"
#version 460

layout (location = 0) in vec2 in_vTextureUV;

layout(binding = 0, set = 0) uniform sampler2D src;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(src, in_vTextureUV);
}
"#,
    glsl,
    frag,
    vulkan1_0,
    entry = "main"
);

const INSTANCE_DATA: [f32; 12] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
];
const VERTEX_INDEX: [u32; 3] = [0, 1, 2, ];
const VERTEX_DATA: [f32; 9] = [
    0.0, 0.0, 0.0,
    0.8, 0.0, 0.0,
    0.0, 0.8, 0.0,
];

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

                let device_local_default_allocator = MemoryPool::new(
                    device_local_memory_heap,
                    Arc::new(DefaultAllocator::new(1024 * 1024 * 128)),
                    MemoryPoolFeatures::from(&[]),
                )
                .unwrap();

                let raytracing_memory_heap = MemoryHeap::new(
                    dev.clone(),
                    ConcreteMemoryHeapDescriptor::new(required_memory_type(), 1024 * 1024 * 128),
                )
                .unwrap();
                println!("Memory heap created! <3");

                let raytracing_allocator = MemoryPool::new(
                    raytracing_memory_heap.clone(),
                    Arc::new(DefaultAllocator::new(1024 * 1024 * 128)),
                    MemoryPoolFeatures::from(&[MemoryPoolFeature::DeviceAddressable]),
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

                let rt_images = (0..swapchain_images_count)
                    .map(|_idx| {
                        vulkan_framework::image::Image::new(
                            device_local_default_allocator.clone(),
                            ConcreteImageDescriptor::new(
                                ImageDimensions::Image2D {
                                    extent: swapchain_extent,
                                },
                                ImageUsage::Managed(ImageUsageSpecifier::new(
                                    false, false, true, true, false, false, false, false,
                                )),
                                ImageMultisampling::SamplesPerPixel1,
                                1,
                                1,
                                ImageFormat::r32g32b32a32_sfloat,
                                ImageFlags::empty(),
                                ImageTiling::Optimal,
                            ),
                            None,
                            Some("Test Image"),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<vulkan_framework::image::Image>>>();

                let rt_image_views = rt_images
                    .iter()
                    .map(|rt_img| {
                        ImageView::new(
                            rt_img.clone(),
                            ImageViewType::Image2D,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<ImageView>>>();

                let swapchain_images = ImageSwapchainKHR::extract(swapchain.clone()).unwrap();
                println!("Swapchain images extracted!");

                let descriptor_pool = DescriptorPool::new(
                    dev.clone(),
                    DescriptorPoolConcreteDescriptor::new(
                        DescriptorPoolSizesConcreteDescriptor::new(
                            0,
                            swapchain_images_count,
                            0,
                            swapchain_images_count,
                            0,
                            0,
                            0,
                            0,
                            0,
                            None,
                        ),
                        2 * swapchain_images_count, // one for descriptor_set and one for renderquad_descriptor_set
                    ),
                    Some("My descriptor pool"),
                )
                .unwrap();

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

                let command_pool =
                    CommandPool::new(queue_family.clone(), Some("My command pool")).unwrap();

                let present_command_buffers = (0..swapchain_images_count)
                    .map(|_idx| {
                        PrimaryCommandBuffer::new(
                            command_pool.clone(),
                            Some("present_command_buffers[...]"),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<PrimaryCommandBuffer>>>();

                let rt_output_image_descriptor = BindingDescriptor::new(
                    ShaderStageAccess::raytracing(),
                    BindingType::Native(NativeBindingType::StorageImage),
                    0,
                    1,
                );

                let rt_writer_img_layout = ImageLayout::General;

                let rt_descriptor_set_layout =
                    DescriptorSetLayout::new(dev.clone(), &[rt_output_image_descriptor]).unwrap();

                let rt_descriptor_sets = (0..swapchain_images_count)
                    .map(|idx| {
                        let result = DescriptorSet::new(
                            descriptor_pool.clone(),
                            rt_descriptor_set_layout.clone(),
                        )
                        .unwrap();

                        result
                            .bind_resources(|binder| {
                                binder.bind_storage_images(
                                    0,
                                    &[(rt_writer_img_layout, rt_image_views[idx as usize].clone())],
                                );
                            })
                            .unwrap();

                        result
                    })
                    .collect::<Vec<Arc<DescriptorSet>>>();

                let pipeline_layout = PipelineLayout::new(
                    dev.clone(),
                    &[rt_descriptor_set_layout],
                    &[],
                    Some("pipeline_layout"),
                )
                .unwrap();
                println!("Pipeline layout created!");

                let swapchain_image_views = swapchain_images
                    .clone()
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

                let renderquad_sampler = vulkan_framework::sampler::Sampler::new(
                    dev.clone(),
                    vulkan_framework::sampler::Filtering::Nearest,
                    vulkan_framework::sampler::Filtering::Nearest,
                    vulkan_framework::sampler::MipmapMode::ModeNearest,
                    0.0,
                )
                .unwrap();

                let renderquad_renderpass = vulkan_framework::renderpass::RenderPass::new(
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

                let renderquad_image_input_format = ImageLayout::ShaderReadOnlyOptimal;

                let renderquad_texture_binding_descriptor = BindingDescriptor::new(
                    ShaderStageAccess::graphics(),
                    BindingType::Native(NativeBindingType::CombinedImageSampler),
                    0,
                    1,
                );

                let renderquad_descriptor_set_layout = DescriptorSetLayout::new(
                    dev.clone(),
                    &[renderquad_texture_binding_descriptor.clone()],
                )
                .unwrap();

                let renderquad_pipeline_layout = PipelineLayout::new(
                    dev.clone(),
                    &[renderquad_descriptor_set_layout.clone()],
                    &[],
                    Some("pipeline_layout"),
                )
                .unwrap();
                println!("Pipeline layout created!");

                let renderquad_descriptor_sets = (0..swapchain_images_count)
                    .map(|idx| {
                        let result = DescriptorSet::new(
                            descriptor_pool.clone(),
                            renderquad_descriptor_set_layout.clone(),
                        )
                        .unwrap();

                        result
                            .bind_resources(|binder| {
                                binder.bind_combined_images_samplers(
                                    0,
                                    &[(
                                        renderquad_image_input_format,
                                        rt_image_views[idx as usize].clone(),
                                        renderquad_sampler.clone(),
                                    )],
                                )
                            })
                            .unwrap();

                        result
                    })
                    .collect::<Vec<Arc<DescriptorSet>>>();

                let renderquad_vertex_shader = VertexShader::new(
                    dev.clone(),
                    &[],
                    &[renderquad_texture_binding_descriptor.clone()],
                    RENDERQUAD_VERTEX_SPV,
                )
                .unwrap();

                let renderquad_fragment_shader = FragmentShader::new(
                    dev.clone(),
                    &[],
                    &[renderquad_texture_binding_descriptor],
                    RENDERQUAD_FRAGMENT_SPV,
                )
                .unwrap();

                let renderquad_graphics_pipeline = GraphicsPipeline::new(
                    renderquad_renderpass.clone(),
                    0,
                    ImageMultisampling::SamplesPerPixel1,
                    Some(DepthConfiguration::new(
                        true,
                        DepthCompareOp::Always,
                        Some((0.0, 1.0)),
                    )),
                    Image2DDimensions::new(WIDTH, HEIGHT),
                    renderquad_pipeline_layout.clone(),
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
                    (renderquad_vertex_shader, None),
                    (renderquad_fragment_shader, None),
                    Some("renderquad_pipeline"),
                )
                .unwrap();
                println!("Graphics pipeline created!");

                let rendequad_framebuffers = (0..(swapchain_images_count))
                    .map(|idx| {
                        Framebuffer::new(
                            renderquad_renderpass.clone(),
                            &[swapchain_image_views[idx as usize].clone()],
                            swapchain_extent,
                            1,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<Arc<Framebuffer>>>();

                let raygen_shader = RaygenShader::new(dev.clone(), RAYGEN_SPV).unwrap();
                //let intersection_shader = IntersectionShader::new(dev.clone(), INTERSECTION_SPV).unwrap();
                let miss_shader = MissShader::new(dev.clone(), MISS_SPV).unwrap();
                //let anyhit_shader = AnyHitShader::new(dev.clone(), AHIT_SPV).unwrap();
                let closesthit_shader = ClosestHitShader::new(dev.clone(), CHIT_SPV).unwrap();
                //let callable_shader = CallableShader::new(dev.clone(), CALLABLE_SPV).unwrap();

                let pipeline = RaytracingPipeline::new(
                    pipeline_layout.clone(),
                    1,
                    raygen_shader,
                    None,
                    miss_shader,
                    None,
                    closesthit_shader,
                    None,
                    Some("raytracing_pipeline!"),
                )
                .unwrap();

                let shader_binding_tables = swapchain_images
                    .into_iter()
                    .map(|_image_swapchain| {
                        RaytracingBindingTables::new(pipeline.clone(), raytracing_allocator.clone())
                            .unwrap()
                    })
                    .collect::<Vec<Arc<RaytracingBindingTables>>>();

                let triangle_decl = BottomLevelTrianglesGroupDecl::new(
                    1,
                    (std::mem::size_of::<f32>() as u64) * 3u64,
                    AttributeType::Vec3
                );

                let blas_estimated_sizes = BottomLevelAccelerationStructure::query_minimum_sizes(
                    dev.clone(),
                    AllowedBuildingDevice::DeviceOnly,
                    &[triangle_decl]
                ).unwrap();

                let scratch_buffer = DeviceScratchBuffer::new(
                    raytracing_allocator.clone(),
                    blas_estimated_sizes.1,
                ).unwrap();

                let vertex_buffer = Buffer::new(
                    raytracing_allocator.clone(),
                    ConcreteBufferDescriptor::new(
                        BufferUsage::Unmanaged(
                                (
                                    ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR |
                                    ash::vk::BufferUsageFlags::VERTEX_BUFFER |
                                    ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                ).as_raw()
                            ),
                            (core::mem::size_of::<[f32; 3]>() as u64) * 3u64
                        ),
                        None,
                        None
                ).unwrap();

                let index_buffer = Buffer::new(
                    raytracing_allocator.clone(),
                    ConcreteBufferDescriptor::new(
                        BufferUsage::Unmanaged(
                                (
                                    ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR |
                                    ash::vk::BufferUsageFlags::INDEX_BUFFER |
                                    ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                ).as_raw()
                            ),
                            (core::mem::size_of::<u32>() as u64) * 3u64
                        ),
                        None,
                        None
                ).unwrap();

                let transform_buffer = Buffer::new(
                    raytracing_allocator.clone(),
                    ConcreteBufferDescriptor::new(
                        BufferUsage::Unmanaged(
                                (
                                    ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR |
                                    ash::vk::BufferUsageFlags::INDEX_BUFFER |
                                    ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                ).as_raw()
                            ),
                            (core::mem::size_of::<[f32; 12]>() as u64) * 1u64
                        ),
                        None,
                        None
                ).unwrap();

                raytracing_allocator.write_raw_data(index_buffer.allocation_offset(), VERTEX_INDEX.as_slice()).unwrap();
                raytracing_allocator.write_raw_data(vertex_buffer.allocation_offset(), VERTEX_DATA.as_slice()).unwrap();
                raytracing_allocator.write_raw_data(transform_buffer.allocation_offset(), INSTANCE_DATA.as_slice()).unwrap();

                let blas = BottomLevelAccelerationStructure::new(
                    raytracing_allocator.clone(),
                    blas_estimated_sizes.0,
                )
                .unwrap();

                // PUNTO DI INTERESE
                let tlas_building = PrimaryCommandBuffer::new(command_pool.clone(), Some("AS_Builder")).unwrap();
                tlas_building.record_commands(|cmd|
                    {
                        cmd.build_blas(
                            blas.clone(),
                            scratch_buffer.clone(),
                            &[
                                BottomLevelTrianglesGroupData::new(
                                    triangle_decl,
                                    index_buffer.clone(),
                                    vertex_buffer.clone(),
                                    transform_buffer.clone(),
                                    0,
                                    1,
                                    0,
                                    0
                                )
                            ],
                        );
                    }).unwrap();
                let tlas_building_fence = Fence::new(dev.clone(), false, Some("tlas_building_fence")).unwrap();
                let mut waiter = queue.submit(
                    &[tlas_building.clone()],
                    &[],
                    &[],
                    tlas_building_fence
                ).unwrap();

                loop {
                    match waiter.wait(100) {
                        Ok(_) => {
                            break;
                        },
                        Err(err) => {
                            if err.is_timeout() {
                                //println!("TIMEOUT");
                                continue;
                            }

                            panic!("{}", err)
                        }
                    }
                }
                

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
                            

                            // TODO: HERE transition the image layout from UNDEFINED to GENERAL so that ray tracing pipeline can write to it
                            recorder.image_barrier(ImageMemoryBarrier::new(
                                PipelineStages::from(&[PipelineStage::TopOfPipe], None, None, None),
                                AccessFlags::Unmanaged(0),
                                PipelineStages::from(
                                    &[],
                                    None,
                                    None,
                                    Some(&[PipelineStageRayTracingPipelineKHR::RayTracingShader]),
                                ),
                                AccessFlags::Managed(AccessFlagsSpecifier::from(
                                    &[AccessFlag::ShaderWrite],
                                    None,
                                )),
                                rt_images[current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                                None,
                                None,
                                None,
                                None,
                                None,
                                ImageLayout::Undefined,
                                rt_writer_img_layout,
                                queue_family.clone(),
                                queue_family.clone(),
                            ));

                            recorder.bind_ray_tracing_pipeline(pipeline.clone());
                            recorder.bind_descriptor_sets_for_ray_tracing_pipeline(
                                pipeline_layout.clone(),
                                0,
                                &[rt_descriptor_sets
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone()],
                            );
                            recorder.trace_rays(
                                shader_binding_tables
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                                Image3DDimensions::from(swapchain_extent),
                            );

                            // HERE wait for the ray tracing pipeline to transition image layout from GENERAL to renderquad_texture_layout
                            recorder.image_barrier(ImageMemoryBarrier::new(
                                PipelineStages::from(
                                    &[],
                                    None,
                                    None,
                                    Some(&[PipelineStageRayTracingPipelineKHR::RayTracingShader]),
                                ),
                                AccessFlags::Managed(AccessFlagsSpecifier::from(
                                    &[AccessFlag::ShaderWrite],
                                    None,
                                )),
                                PipelineStages::from(
                                    &[PipelineStage::FragmentShader],
                                    None,
                                    None,
                                    None,
                                ),
                                AccessFlags::Managed(AccessFlagsSpecifier::from(
                                    &[AccessFlag::ShaderRead],
                                    None,
                                )),
                                rt_images[current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                                None,
                                None,
                                None,
                                None,
                                None,
                                rt_writer_img_layout,
                                renderquad_image_input_format,
                                queue_family.clone(),
                                queue_family.clone(),
                            ));

                            recorder.begin_renderpass(
                                rendequad_framebuffers
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone(),
                                &[ClearValues::new(Some(ColorClearValues::Vec4(
                                    1.0, 1.0, 1.0, 1.0,
                                )))],
                            );
                            recorder.bind_graphics_pipeline(renderquad_graphics_pipeline.clone());
                            recorder.bind_descriptor_sets_for_graphics_pipeline(
                                renderquad_pipeline_layout.clone(),
                                0,
                                &[renderquad_descriptor_sets
                                    [current_frame % (swapchain_images_count as usize)]
                                    .clone()],
                            );
                            recorder.draw(0, 6, 0, 1);

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
