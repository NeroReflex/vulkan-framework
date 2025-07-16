use std::{sync::Arc, time::Duration};

use inline_spirv::*;

use vulkan_framework::{
    acceleration_structure::{
        AllowedBuildingDevice, BottomLevelAccelerationStructure, BottomLevelTrianglesGroupData,
        BottomLevelTrianglesGroupDecl, DeviceScratchBuffer, TopLevelAccelerationStructure,
        TopLevelBLASGroupData, TopLevelBLASGroupDecl, VertexIndexing,
    },
    binding_tables::{required_memory_type, RaytracingBindingTables},
    buffer::{AllocatedBuffer, Buffer, BufferUsage, ConcreteBufferDescriptor},
    command_buffer::{
        AccessFlag, AccessFlags, AccessFlagsSpecifier, ClearValues, ColorClearValues,
        CommandBufferRecorder, ImageMemoryBarrier, PrimaryCommandBuffer,
    },
    command_pool::CommandPool,
    descriptor_pool::DescriptorPoolSizesAcceletarionStructureKHR,
    descriptor_set_layout::DescriptorSetLayout,
    device::*,
    fence::{Fence, FenceWaitFor},
    framebuffer::Framebuffer,
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, Viewport,
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
    memory_pool::{MemoryPool, MemoryPoolBacked, MemoryPoolFeature, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue::*,
    queue_family::*,
    raytracing_pipeline::RaytracingPipeline,
    renderpass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, RenderSubPassDescription,
    },
    semaphore::Semaphore,
    shader_layout_binding::{
        AccelerationStructureBindingType, BindingDescriptor, BindingType, NativeBindingType,
    },
    shader_stage_access::{ShaderStageRayTracingKHR, ShaderStagesAccess},
    shaders::{
        closest_hit_shader::ClosestHitShader, fragment_shader::FragmentShader,
        miss_shader::MissShader, raygen_shader::RaygenShader, vertex_shader::VertexShader,
    },
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
    swapchain_image::ImageSwapchainKHR,
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

layout(binding = 1, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadEXT vec3 hitValue;

void main() {
    const vec2 resolution = vec2(imageSize(outputImage));

    const ivec2 pixelCoords = ivec2(gl_LaunchIDEXT.xy);

    const vec2 position_xy = vec2((float(pixelCoords.x) + 0.5) / resolution.x, (float(pixelCoords.y) + 0.5) / resolution.y);
    const vec3 origin = vec3(position_xy, -0.5);
    const vec3 direction = vec3(0.0, 0.0, 1.0);

    vec4 output_color = vec4(1.0, 0.0, 0.0, 0.0);

    hitValue = vec3(0.0, 0.0, 0.1);

    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, 0.001, direction.xyz, 10.0, 0);
    //                      gl_RayFlagsNoneEXT

    // Store the output color to the image
    imageStore(outputImage, pixelCoords, vec4(hitValue.xyz, 1.0));
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

//uniform layout(binding=0, set = 0, rgba32f) writeonly image2D someImage;

//layout(binding = 0, set = 1) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main() {
    hitValue = vec3(0.0, 0.0, 0.2);
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
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;

//layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
//layout(binding = 1, set = 0) uniform image2D outputImage;

void main() {
    const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    hitValue = barycentricCoords;
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

const INSTANCE_DATA: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
const VERTEX_INDEX: [u32; 3] = [0, 1, 2];
const VERTEX_DATA: [f32; 9] = [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_window");

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
        let window = match video_subsystem
            .window("Window", WIDTH, HEIGHT)
            .vulkan()
            .build()
        {
            Ok(window) => {
                println!("SDL2 window created");
                window
            }
            Err(err) => {
                println!("Error creating sdl2 window: {}", err);
                return Err(Box::new(err));
            }
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
                panic!(
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

        let rt_image_handles = (0..swapchain_images_count)
            .map(|_idx| {
                vulkan_framework::image::Image::new(
                    dev.clone(),
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
            .collect::<Vec<_>>();

        let device_local_memory_heap = MemoryHeap::new(
            dev.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(None),
                1024 * 1024 * 128, // 128MiB of memory!
            ),
            rt_image_handles
                .iter()
                .map(|h| h as &dyn MemoryRequiring)
                .collect::<Vec<_>>()
                .as_slice(),
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
            &[],
        )
        .unwrap();
        println!("Memory heap created! <3");

        let raytracing_allocator = MemoryPool::new(
            raytracing_memory_heap,
            Arc::new(DefaultAllocator::new(1024 * 1024 * 128)),
            MemoryPoolFeatures::from(&[MemoryPoolFeature::DeviceAddressable]),
        )
        .unwrap();

        let rt_images = rt_image_handles
            .into_iter()
            .map(|rt_image_handle| {
                vulkan_framework::image::AllocatedImage::new(
                    device_local_default_allocator.clone(),
                    rt_image_handle,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

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
            .collect::<Vec<_>>();

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
                    Some(DescriptorPoolSizesAcceletarionStructureKHR::new(
                        swapchain_images_count,
                    )),
                ),
                2 * swapchain_images_count, // one for descriptor_set and one for renderquad_descriptor_set
            ),
            Some("My descriptor pool"),
        )
        .unwrap();

        let image_available_semaphores = (0..frames_in_flight)
            .map(|idx| {
                Semaphore::new(
                    dev.clone(),
                    Some(format!("image_available_semaphores[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // this tells me when the present operation can start
        let present_ready = (0..swapchain_images_count)
            .map(|idx| {
                Semaphore::new(dev.clone(), Some(format!("present_ready[{idx}]").as_str())).unwrap()
            })
            .collect::<Vec<_>>();

        let swapchain_fences = (0..swapchain_images_count)
            .map(|idx| {
                Fence::new(
                    dev.clone(),
                    true,
                    Some(format!("swapchain_fences[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let command_pool = CommandPool::new(queue_family.clone(), Some("My command pool")).unwrap();

        let present_command_buffers = (0..swapchain_images_count)
            .map(|idx| {
                PrimaryCommandBuffer::new(
                    command_pool.clone(),
                    Some(format!("present_command_buffers[{idx}]").as_str()),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let rt_output_image_descriptor = BindingDescriptor::new(
            ShaderStagesAccess::from(&[], &[ShaderStageRayTracingKHR::RayGen]),
            BindingType::Native(NativeBindingType::StorageImage),
            0,
            1,
        );

        let rt_acceleration_structure_descriptor = BindingDescriptor::new(
            ShaderStagesAccess::from(&[], &[ShaderStageRayTracingKHR::RayGen]),
            BindingType::AccelerationStructure(
                AccelerationStructureBindingType::AccelerationStructure,
            ),
            1,
            1,
        );

        let rt_writer_img_layout = ImageLayout::General;

        let rt_descriptor_set_layout = DescriptorSetLayout::new(
            dev.clone(),
            &[
                rt_output_image_descriptor,
                rt_acceleration_structure_descriptor,
            ],
        )
        .unwrap();

        let pipeline_layout = PipelineLayout::new(
            dev.clone(),
            &[rt_descriptor_set_layout.clone()],
            &[],
            Some("pipeline_layout"),
        )
        .unwrap();
        println!("Pipeline layout created!");

        let swapchain_image_views = swapchain_images
            .clone()
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
            .collect::<Vec<_>>();

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
            ShaderStagesAccess::graphics(),
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
                        binder
                            .bind_combined_images_samplers(
                                0,
                                &[(
                                    renderquad_image_input_format,
                                    rt_image_views[idx as usize].clone(),
                                    renderquad_sampler.clone(),
                                )],
                            )
                            .unwrap()
                    })
                    .unwrap();

                result
            })
            .collect::<Vec<_>>();

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
            None,
            renderquad_renderpass.clone(),
            0,
            ImageMultisampling::SamplesPerPixel1,
            Some(DepthConfiguration::new(
                true,
                DepthCompareOp::Always,
                Some((0.0, 1.0)),
            )),
            Some(Viewport::new(
                0.0f32,
                0.0f32,
                WIDTH as f32,
                HEIGHT as f32,
                0.0f32,
                0.0f32,
            )),
            Some(Scissor::new(0, 0, Image2DDimensions::new(WIDTH, HEIGHT))),
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
            .collect::<Vec<_>>();

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
            .collect::<Vec<_>>();

        let triangle_decl = BottomLevelTrianglesGroupDecl::new(
            VertexIndexing::UInt32,
            1,
            (std::mem::size_of::<f32>() as u64) * 3u64,
            AttributeType::Vec3,
        );

        let blas_decl = TopLevelBLASGroupDecl::new();

        let blas_estimated_sizes = BottomLevelAccelerationStructure::query_minimum_sizes(
            dev.clone(),
            AllowedBuildingDevice::DeviceOnly,
            &[triangle_decl],
        )
        .unwrap();

        let blas = BottomLevelAccelerationStructure::new(
            raytracing_allocator.clone(),
            blas_estimated_sizes.0,
        )
        .unwrap();

        let blas_scratch_buffer = DeviceScratchBuffer::new(
            raytracing_allocator.clone(),
            blas_estimated_sizes.1,
        )
        .unwrap();

        let tlas_estimated_sizes = TopLevelAccelerationStructure::query_minimum_sizes(
            dev.clone(),
            AllowedBuildingDevice::DeviceOnly,
            &[blas_decl],
        )
        .unwrap();

        let tlas = TopLevelAccelerationStructure::new(
            raytracing_allocator.clone(),
            tlas_estimated_sizes.0,
        )
        .unwrap();

        let vertex_buffer = Buffer::new(
            dev.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::VERTEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                (core::mem::size_of::<[f32; 3]>() as u64) * 3u64,
            ),
            None,
            None,
        )
        .unwrap();

        let vertex_buffer =
            AllocatedBuffer::new(raytracing_allocator.clone(), vertex_buffer).unwrap();

        let index_buffer = Buffer::new(
            dev.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                (core::mem::size_of::<u32>() as u64) * 3u64,
            ),
            None,
            None,
        )
        .unwrap();

        let index_buffer =
            AllocatedBuffer::new(raytracing_allocator.clone(), index_buffer).unwrap();

        let transform_buffer = Buffer::new(
            dev.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                core::mem::size_of::<[f32; 12]>() as u64,
            ),
            None,
            None,
        )
        .unwrap();

        let transform_buffer =
            AllocatedBuffer::new(raytracing_allocator.clone(), transform_buffer).unwrap();

        raytracing_allocator
            .write_raw_data(index_buffer.allocation_offset(), VERTEX_INDEX.as_slice())
            .unwrap();
        raytracing_allocator
            .write_raw_data(vertex_buffer.allocation_offset(), VERTEX_DATA.as_slice())
            .unwrap();
        raytracing_allocator
            .write_raw_data(
                transform_buffer.allocation_offset(),
                INSTANCE_DATA.as_slice(),
            )
            .unwrap();

        // PUNTO DI INTERESE
        let blas_building =
            PrimaryCommandBuffer::new(command_pool.clone(), Some("BLAS_Builder")).unwrap();
        blas_building
            .record_commands(|cmd| {
                cmd.build_blas(
                    blas.clone(),
                    blas_scratch_buffer.clone(),
                    &[BottomLevelTrianglesGroupData::new(
                        triangle_decl,
                        Some(index_buffer.clone()),
                        vertex_buffer.clone(),
                        transform_buffer.clone(),
                        0,
                        1,
                        0,
                        0,
                    )],
                );
            })
            .unwrap();
        let blas_building_fence =
            Fence::new(dev.clone(), false, Some("blas_building_fence")).unwrap();

        queue
            .submit(&[blas_building], &[], &[], blas_building_fence.clone())
            .unwrap();

        loop {
            match Fence::wait_for_fences(
                &[blas_building_fence.clone()],
                FenceWaitFor::All,
                Duration::from_nanos(100),
            ) {
                Ok(_) => {
                    blas_building_fence.reset().unwrap();
                    break;
                }
                Err(err) => {
                    if err.is_timeout() {
                        //println!("TIMEOUT");
                        continue;
                    }

                    panic!("Error in waiting for fence: {err}")
                }
            }
        }

        let blas_instances_buffer = Buffer::new(
            dev.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::Unmanaged(
                    (ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | ash::vk::BufferUsageFlags::INDEX_BUFFER
                        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .as_raw(),
                ),
                core::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() as u64,
            ),
            None,
            None,
        )
        .unwrap();

        let blas_instances_buffer =
            AllocatedBuffer::new(raytracing_allocator.clone(), blas_instances_buffer).unwrap();

        let accel_structure_instance = ash::vk::AccelerationStructureInstanceKHR {
            transform: ash::vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            instance_shader_binding_table_record_offset_and_flags: ash::vk::Packed24_8::new(
                0, 0x01,
            ), // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
            instance_custom_index_and_mask: ash::vk::Packed24_8::new(0x00, 0xFF),
            acceleration_structure_reference: ash::vk::AccelerationStructureReferenceKHR {
                device_handle: blas.device_addr(),
            },
        };

        raytracing_allocator
            .write_raw_data(
                blas_instances_buffer.allocation_offset(),
                &[accel_structure_instance],
            )
            .unwrap();

        let tlas_scratch_buffer = DeviceScratchBuffer::new(
            raytracing_allocator.clone(),
            tlas_estimated_sizes.1,
        )
        .unwrap();

        let tlas_building = PrimaryCommandBuffer::new(command_pool, Some("TLAS_Builder")).unwrap();
        tlas_building
            .record_commands(|cmd| {
                cmd.build_tlas(
                    tlas.clone(),
                    tlas_scratch_buffer.clone(),
                    &[TopLevelBLASGroupData::new(
                        blas_decl,
                        blas_instances_buffer.clone(),
                        0,
                        1,
                        0,
                        0,
                    )],
                )
            })
            .unwrap();
        let tlas_building_fence =
            Fence::new(dev.clone(), false, Some("tlas_building_fence")).unwrap();

        queue
            .submit(&[tlas_building], &[], &[], tlas_building_fence.clone())
            .unwrap();

        loop {
            match Fence::wait_for_fences(
                &[tlas_building_fence.clone()],
                FenceWaitFor::All,
                Duration::from_nanos(100),
            ) {
                Ok(_) => {
                    tlas_building_fence.reset().unwrap();
                    break;
                }
                Err(err) => {
                    if err.is_timeout() {
                        //println!("TIMEOUT");
                        continue;
                    }

                    panic!("error in waiting for fence: {err}")
                }
            }
        }

        let rt_descriptor_sets = (0..swapchain_images_count)
            .map(|idx| {
                let result =
                    DescriptorSet::new(descriptor_pool.clone(), rt_descriptor_set_layout.clone())
                        .unwrap();

                result
                    .bind_resources(|binder| {
                        binder
                            .bind_storage_images(
                                0,
                                &[(rt_writer_img_layout, rt_image_views[idx as usize].clone())],
                            )
                            .unwrap();

                        binder.bind_tlas(1, &[tlas.clone()]).unwrap();
                    })
                    .unwrap();

                result
            })
            .collect::<Vec<_>>();

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
                        for fence in swapchain_fences.iter() {
                            Fence::wait_for_fences(
                                &[fence.clone()],
                                FenceWaitFor::All,
                                Duration::from_nanos(u64::MAX),
                            )
                            .unwrap();
                            fence.reset().unwrap();
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
                        rt_images[swapchain_index as usize].clone(),
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
                        &[rt_descriptor_sets[swapchain_index as usize].clone()],
                    );
                    recorder.trace_rays(
                        shader_binding_tables[swapchain_index as usize].clone(),
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
                        PipelineStages::from(&[PipelineStage::FragmentShader], None, None, None),
                        AccessFlags::Managed(AccessFlagsSpecifier::from(
                            &[AccessFlag::ShaderRead],
                            None,
                        )),
                        rt_images[swapchain_index as usize].clone(),
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
                        rendequad_framebuffers[swapchain_index as usize].clone(),
                        &[ClearValues::new(Some(ColorClearValues::Vec4(
                            1.0, 1.0, 1.0, 1.0,
                        )))],
                    );
                    recorder.bind_graphics_pipeline(
                        renderquad_graphics_pipeline.clone(),
                        None,
                        None,
                    );
                    recorder.bind_descriptor_sets_for_graphics_pipeline(
                        renderquad_pipeline_layout.clone(),
                        0,
                        &[renderquad_descriptor_sets[swapchain_index as usize].clone()],
                    );
                    recorder.draw(0, 6, 0, 1);

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

    Ok(())
}
