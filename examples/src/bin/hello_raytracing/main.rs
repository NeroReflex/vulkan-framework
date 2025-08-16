use std::{sync::Arc, time::Duration};

use inline_spirv::*;

use vulkan_framework::{
    acceleration_structure::{
        bottom_level::{
            BottomLevelAccelerationStructure, BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl,
        },
        top_level::{
            TopLevelAccelerationStructure, TopLevelAccelerationStructureInstanceBuffer,
            TopLevelBLASGroupDecl,
        },
        AllowedBuildingDevice, VertexIndexing,
    },
    ash::vk::{AccelerationStructureInstanceKHR, TransformMatrixKHR},
    binding_tables::RaytracingBindingTables,
    buffer::{Buffer, BufferUsage},
    clear_values::ColorClearValues,
    command_buffer::{CommandBufferRecorder, PrimaryCommandBuffer},
    command_pool::CommandPool,
    descriptor_pool::DescriptorPoolSizesAcceletarionStructureKHR,
    descriptor_set_layout::DescriptorSetLayout,
    device::*,
    dynamic_rendering::{
        AttachmentStoreOp, DynamicRendering, DynamicRenderingColorAttachment,
        RenderingAttachmentSetup,
    },
    fence::Fence,
    graphics_pipeline::{
        AttributeType, CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline,
        PolygonMode, Rasterizer, Scissor, Viewport,
    },
    image::{
        AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image2DDimensions,
        Image3DDimensions, ImageDimensions, ImageFlags, ImageFormat, ImageLayout,
        ImageLayoutSwapchainKHR, ImageMultisampling, ImageSubresourceRange, ImageTiling,
        ImageTrait, ImageUsage, ImageUseAs,
    },
    image_view::ImageView,
    instance::*,
    memory_barriers::{ImageMemoryBarrier, MemoryAccess, MemoryAccessAs},
    memory_heap::*,
    memory_management::{
        DefaultMemoryManager, MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait,
    },
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeature, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStageRayTracingPipelineKHR, PipelineStages},
    queue::*,
    queue_family::*,
    raytracing_pipeline::RaytracingPipeline,
    semaphore::Semaphore,
    shader_layout_binding::{
        AccelerationStructureBindingType, BindingDescriptor, BindingType, NativeBindingType,
    },
    shader_stage_access::{
        ShaderStageAccessIn, ShaderStageAccessInRayTracingKHR, ShaderStagesAccess,
    },
    shaders::{
        closest_hit_shader::ClosestHitShader, fragment_shader::FragmentShader,
        miss_shader::MissShader, raygen_shader::RaygenShader, vertex_shader::VertexShader,
    },
    swapchain::{
        CompositeAlphaSwapchainKHR, DeviceSurfaceInfo, PresentModeSwapchainKHR,
        SurfaceColorspaceSwapchainKHR, SurfaceTransformSwapchainKHR, SwapchainKHR,
    },
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

const TRANSFORM_DATA: [[f32; 12]; 2] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
];
const VERTEX_INDEX: [[u32; 3]; 2] = [[0, 1, 2], [0, 1, 2]];
const VERTEX_DATA: [[f32; 9]; 2] = [
    [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0],
];

const INSTANCES_DATA: [[f32; 12]; 2] = [
    [1.0, -2.0, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
];

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

        let final_format = ImageFormat::from(CommonImageFormat::b8g8r8a8_srgb);
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
            PresentModeSwapchainKHR::FIFO,
            color_space,
            CompositeAlphaSwapchainKHR::Opaque,
            SurfaceTransformSwapchainKHR::Identity,
            true,
            final_format,
            ImageUsage::from([ImageUseAs::TransferDst, ImageUseAs::ColorAttachment].as_slice()),
            swapchain_extent,
            swapchain_images_count,
            1,
        )
        .unwrap();
        println!("Swapchain created!");

        let mut mem_manager = DefaultMemoryManager::new(dev.clone());

        let rt_image_handles = (0..swapchain_images_count)
            .map(|_idx| {
                vulkan_framework::image::Image::new(
                    dev.clone(),
                    ConcreteImageDescriptor::new(
                        ImageDimensions::Image2D {
                            extent: swapchain_extent,
                        },
                        ImageUsage::from([ImageUseAs::Sampled, ImageUseAs::Storage].as_slice()),
                        ImageMultisampling::SamplesPerPixel1,
                        1,
                        1,
                        ImageFormat::from(CommonImageFormat::r32g32b32a32_sfloat),
                        ImageFlags::empty(),
                        ImageTiling::Optimal,
                    ),
                    None,
                    Some("Test Image"),
                )
                .unwrap()
                .into()
            })
            .collect::<Vec<_>>();

        let rt_images: Vec<Arc<AllocatedImage>> = mem_manager
            .allocate_resources(
                &MemoryType::device_local(),
                &MemoryPoolFeatures::from([].as_slice()),
                rt_image_handles,
                MemoryManagementTags::default()
                    .with_exclusivity(false)
                    .with_name("non_raytracing".to_string())
                    .with_size(MemoryManagementTagSize::Small),
            )?
            .into_iter()
            .map(|r| r.image())
            .collect();

        let rt_image_views = rt_images
            .iter()
            .map(|rt_img| {
                ImageView::new(
                    rt_img.clone(),
                    None,
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
                    false,
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
            [ShaderStageAccessIn::RayTracing(
                ShaderStageAccessInRayTracingKHR::RayGen,
            )]
            .as_slice()
            .into(),
            BindingType::Native(NativeBindingType::StorageImage),
            0,
            1,
        );

        let rt_acceleration_structure_descriptor = BindingDescriptor::new(
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

        let mut swapchain_images = vec![];
        for index in 0..swapchain_images_count {
            swapchain_images.push(SwapchainKHR::image(swapchain.clone(), index).unwrap());
        }

        let mut swapchain_image_views = vec![];
        for (idx, img) in swapchain_images.iter().enumerate() {
            let swapchain_image_name = format!("swapchain_image_views[{idx}]");
            swapchain_image_views.push(
                ImageView::new(
                    img.clone(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(swapchain_image_name.as_str()),
                )
                .unwrap(),
            );
        }

        let renderquad_sampler = vulkan_framework::sampler::Sampler::new(
            dev.clone(),
            vulkan_framework::sampler::Filtering::Nearest,
            vulkan_framework::sampler::Filtering::Nearest,
            vulkan_framework::sampler::MipmapMode::ModeNearest,
            0.0,
        )
        .unwrap();

        let renderquad_image_input_layout = ImageLayout::ShaderReadOnlyOptimal;

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
                                    renderquad_image_input_layout,
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
            DynamicRendering::new([swapchain.images_format()].as_slice(), None, None),
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

        let raytracing_allocation_tags = MemoryManagementTags::default()
            .with_exclusivity(false)
            .with_name("raytracing".to_string())
            .with_size(MemoryManagementTagSize::MediumSmall);

        let shader_binding_tables = (0..swapchain_images_count)
            .map(|_| {
                RaytracingBindingTables::new(
                    pipeline.clone(),
                    &mut mem_manager,
                    raytracing_allocation_tags.clone(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let vertex_stride = 0u64;
        let vertex_count = 3u64;

        let vertices_topology = BottomLevelVerticesTopologyDecl::new(
            vertex_count as u32 * 3u32,
            AttributeType::Vec3,
            vertex_stride,
        );

        let triangles_decl = BottomLevelTrianglesGroupDecl::new(
            VertexIndexing::UInt32,
            (vertex_count / 3u64) as u32,
        );

        let vertex_buffer_backing = Buffer::new(
            dev.clone(),
            BottomLevelAccelerationStructureVertexBuffer::template(
                &vertices_topology,
                BufferUsage::default(),
            ),
            None,
            Some("blas_vertex_buffer"),
        )?;
        let index_buffer_backing = Buffer::new(
            dev.clone(),
            BottomLevelAccelerationStructureIndexBuffer::template(
                &triangles_decl,
                BufferUsage::default(),
            ),
            None,
            Some("blas_index_buffer"),
        )?;
        let transform_buffer_backing = Buffer::new(
            dev.clone(),
            BottomLevelAccelerationStructureTransformBuffer::template(BufferUsage::default()),
            None,
            Some("blas_transform_buffer"),
        )?;

        let memory_type = MemoryType::device_local_and_host_visible();

        let allocated_buffers = mem_manager.allocate_resources(
            &memory_type,
            &MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable].as_slice()),
            vec![
                vertex_buffer_backing.into(),
                index_buffer_backing.into(),
                transform_buffer_backing.into(),
            ],
            raytracing_allocation_tags.clone(),
        )?;
        assert_eq!(allocated_buffers.len(), 3);

        let vertex_buffer = BottomLevelAccelerationStructureVertexBuffer::new(
            vertices_topology,
            allocated_buffers[0].buffer(),
        )?;
        let index_buffer = BottomLevelAccelerationStructureIndexBuffer::new(
            triangles_decl,
            allocated_buffers[1].buffer(),
        )?;
        let transform_buffer =
            BottomLevelAccelerationStructureTransformBuffer::new(allocated_buffers[2].buffer())?;

        let blas = BottomLevelAccelerationStructure::new(
            &mut mem_manager,
            AllowedBuildingDevice::DeviceOnly,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            raytracing_allocation_tags.clone(),
            None,
            Some("my_blas"),
        )
        .unwrap();

        // fill buffers with data
        {
            {
                let buffer = blas.index_buffer().buffer();
                let mem_map = MemoryMap::new(buffer.get_backing_memory_pool())?;
                let mut range =
                    mem_map.range::<[u32; 3]>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
                *range = VERTEX_INDEX[0];
            }

            {
                let buffer = blas.vertex_buffer().buffer();
                let mem_map = MemoryMap::new(buffer.get_backing_memory_pool())?;
                let mut range =
                    mem_map.range::<[f32; 9]>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
                *range = VERTEX_DATA[0];
            }

            {
                let buffer = blas.transform_buffer().buffer();
                let mem_map = MemoryMap::new(buffer.get_backing_memory_pool())?;
                let mut range = mem_map
                    .range::<TransformMatrixKHR>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
                *range = vulkan_framework::ash::vk::TransformMatrixKHR {
                    matrix: TRANSFORM_DATA[0],
                };
            }
        }

        let blas_decl = TopLevelBLASGroupDecl::new();
        let max_instances = 2;

        let instances_buffer_backing = Buffer::new(
            dev.clone(),
            TopLevelAccelerationStructureInstanceBuffer::template(
                &blas_decl,
                max_instances,
                BufferUsage::default(),
            ),
            None,
            Some("tlas_instance_buffer"),
        )?;

        let allocated_buffers = mem_manager.allocate_resources(
            &memory_type,
            &MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable].as_slice()),
            vec![instances_buffer_backing.into()],
            MemoryManagementTags::default()
                .with_exclusivity(false)
                .with_name("raytracing".to_string())
                .with_size(MemoryManagementTagSize::MediumSmall),
        )?;
        assert_eq!(allocated_buffers.len(), 1);

        let instance_buffer = TopLevelAccelerationStructureInstanceBuffer::new(
            blas_decl,
            max_instances,
            allocated_buffers[0].buffer(),
        )?;

        let tlas = TopLevelAccelerationStructure::new(
            &mut mem_manager,
            AllowedBuildingDevice::DeviceOnly,
            instance_buffer,
            raytracing_allocation_tags.clone(),
            None,
            Some("tlas"),
        )
        .unwrap();

        {
            let accel_structure_instances: smallvec::SmallVec<
                [vulkan_framework::ash::vk::AccelerationStructureInstanceKHR; 2],
            > = smallvec::smallvec![
                vulkan_framework::ash::vk::AccelerationStructureInstanceKHR {
                    transform: vulkan_framework::ash::vk::TransformMatrixKHR {
                        matrix: INSTANCES_DATA[0],
                    },
                    instance_shader_binding_table_record_offset_and_flags:
                        vulkan_framework::ash::vk::Packed24_8::new(0, 0x01,), // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
                    instance_custom_index_and_mask: vulkan_framework::ash::vk::Packed24_8::new(
                        0x00, 0xFF
                    ),
                    acceleration_structure_reference:
                        vulkan_framework::ash::vk::AccelerationStructureReferenceKHR {
                            device_handle: blas.device_addr(),
                        },
                },
                vulkan_framework::ash::vk::AccelerationStructureInstanceKHR {
                    transform: vulkan_framework::ash::vk::TransformMatrixKHR {
                        matrix: INSTANCES_DATA[1],
                    },
                    instance_shader_binding_table_record_offset_and_flags:
                        vulkan_framework::ash::vk::Packed24_8::new(0, 0x01,), // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
                    instance_custom_index_and_mask: vulkan_framework::ash::vk::Packed24_8::new(
                        0x00, 0xFF
                    ),
                    acceleration_structure_reference:
                        vulkan_framework::ash::vk::AccelerationStructureReferenceKHR {
                            device_handle: blas.device_addr(),
                        },
                }
            ];

            {
                let buffer = tlas.instance_buffer().buffer();
                let mem_map = MemoryMap::new(buffer.get_backing_memory_pool())?;
                let mut range = mem_map.range::<AccelerationStructureInstanceKHR>(
                    buffer.clone() as Arc<dyn MemoryPoolBacked>,
                )?;
                let instances = range.as_mut_slice();
                instances.copy_from_slice(accel_structure_instances.as_slice());
            }
        }

        let blas_building =
            PrimaryCommandBuffer::new(command_pool.clone(), Some("BLAS_Builder")).unwrap();
        blas_building
            .record_one_time_submit(|cmd| {
                cmd.build_blas(blas.clone(), 0, 1, 0, 0);
            })
            .unwrap();
        let blas_building_fence =
            Fence::new(dev.clone(), false, Some("blas_building_fence")).unwrap();

        // dropping the fence waiter makes the fence being waited for
        drop(
            queue
                .submit(&[blas_building], &[], &[], blas_building_fence.clone())
                .unwrap(),
        );

        let tlas_building = PrimaryCommandBuffer::new(command_pool, Some("TLAS_Builder")).unwrap();
        tlas_building
            .record_one_time_submit(|cmd| cmd.build_tlas(tlas.clone(), 0, 2))
            .unwrap();
        let tlas_building_fence =
            Fence::new(dev.clone(), false, Some("tlas_building_fence")).unwrap();

        // dropping the fence waiter makes the fence being waited for
        drop(
            queue
                .submit(&[tlas_building], &[], &[], tlas_building_fence.clone())
                .unwrap(),
        );

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

        {
            let mut fence_waiters: smallvec::SmallVec<[_; 4]> = (0..(frames_in_flight as usize))
                .map(|_| Option::None)
                .collect();

            let mut event_pump = sdl_context.event_pump().unwrap();
            'running: loop {
                for event in event_pump.poll_iter() {
                    match event {
                        sdl2::event::Event::Quit { .. }
                        | sdl2::event::Event::KeyDown {
                            keycode: Some(sdl2::keyboard::Keycode::Escape),
                            ..
                        } => {
                            dev.clone().wait_idle().unwrap();
                            break 'running;
                        }
                        _ => {}
                    }
                }

                drop(fence_waiters[current_frame].take());

                let (swapchain_index, _swapchain_optimal) = swapchain
                    .acquire_next_image_index(
                        Duration::from_nanos(u64::MAX),
                        Some(image_available_semaphores[current_frame].clone()),
                        None,
                    )
                    .unwrap();

                present_command_buffers[swapchain_index as usize]
                    .record_one_time_submit(|recorder: &mut CommandBufferRecorder| {
                        // TODO: HERE transition the image layout from UNDEFINED to GENERAL so that ray tracing pipeline can write to it
                        recorder.image_barriers(
                            [ImageMemoryBarrier::new(
                                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                                MemoryAccess::from([].as_slice()),
                                PipelineStages::from(
                                    [PipelineStage::RayTracingPipelineKHR(
                                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                                    )]
                                    .as_slice(),
                                ),
                                MemoryAccess::from([MemoryAccessAs::ShaderWrite].as_slice()),
                                ImageSubresourceRange::from(
                                    rt_images[swapchain_index as usize].clone()
                                        as Arc<dyn ImageTrait>,
                                ),
                                ImageLayout::Undefined,
                                rt_writer_img_layout,
                                queue_family.clone(),
                                queue_family.clone(),
                            )]
                            .as_slice(),
                        );

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

                        // Wait for the ray tracing pipeline to transition the resulting image's layout
                        // from rt_writer_img_layout to renderquad_image_input_layout, so that the image can be used in a descriptor set,
                        // and sampled from there to write to the image in the swapchain (that is a different image)
                        recorder.image_barriers(
                            [ImageMemoryBarrier::new(
                                PipelineStages::from(
                                    [PipelineStage::RayTracingPipelineKHR(
                                        PipelineStageRayTracingPipelineKHR::RayTracingShader,
                                    )]
                                    .as_slice(),
                                ),
                                MemoryAccess::from([MemoryAccessAs::ShaderWrite].as_slice()),
                                PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                                MemoryAccess::from([MemoryAccessAs::ShaderRead].as_slice()),
                                ImageSubresourceRange::from(
                                    rt_images[swapchain_index as usize].clone()
                                        as Arc<dyn ImageTrait>,
                                ),
                                rt_writer_img_layout,
                                renderquad_image_input_layout,
                                queue_family.clone(),
                                queue_family.clone(),
                            )]
                            .as_slice(),
                        );

                        // Transition the final swapchain image into color attachment optimal layout,
                        // so that the graphics pipeline has it in the best format, and the final barrier
                        // can transition it from that layout to the one suitable for presentation on the
                        // swapchain
                        recorder.image_barriers(
                            [ImageMemoryBarrier::new(
                                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                                MemoryAccess::from([].as_slice()),
                                PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                                MemoryAccess::from(
                                    [MemoryAccessAs::ColorAttachmentWrite].as_slice(),
                                ),
                                ImageSubresourceRange::from(
                                    swapchain_images[swapchain_index as usize].clone()
                                        as Arc<dyn ImageTrait>,
                                ),
                                ImageLayout::Undefined,
                                ImageLayout::ColorAttachmentOptimal,
                                queue_family.clone(),
                                queue_family.clone(),
                            )]
                            .as_slice(),
                        );

                        let rendering_color_attachments = [DynamicRenderingColorAttachment::new(
                            swapchain_image_views[swapchain_index as usize].clone(),
                            RenderingAttachmentSetup::clear(ColorClearValues::Vec4(
                                1.0, 1.0, 1.0, 1.0,
                            )),
                            AttachmentStoreOp::Store,
                        )];
                        recorder.graphics_rendering(
                            swapchain_extent,
                            rendering_color_attachments.as_slice(),
                            None,
                            None,
                            |recorder| {
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
                            },
                        );

                        // Wait for the renderquad to complete the rendering so that we can then transition the image
                        // in a layout that is suitable for presentation on the swapchain.
                        recorder.image_barriers(
                            [ImageMemoryBarrier::new(
                                PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                                MemoryAccess::from([MemoryAccessAs::ShaderWrite].as_slice()),
                                PipelineStages::from([PipelineStage::BottomOfPipe].as_slice()),
                                MemoryAccess::from([].as_slice()),
                                ImageSubresourceRange::from(
                                    swapchain_images[swapchain_index as usize].clone()
                                        as Arc<dyn ImageTrait>,
                                ),
                                ImageLayout::ColorAttachmentOptimal,
                                ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                                queue_family.clone(),
                                queue_family.clone(),
                            )]
                            .as_slice(),
                        );
                    })
                    .unwrap();

                fence_waiters[current_frame] = Some(
                    queue
                        .submit(
                            &[present_command_buffers[swapchain_index as usize].clone()],
                            &[(
                                PipelineStages::from([PipelineStage::FragmentShader].as_slice()),
                                image_available_semaphores[current_frame].clone(),
                            )],
                            &[present_ready[swapchain_index as usize].clone()],
                            swapchain_fences[current_frame].clone(),
                        )
                        .unwrap(),
                );

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

    Ok(())
}
