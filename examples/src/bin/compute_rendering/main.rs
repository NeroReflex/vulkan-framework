use std::sync::Arc;
use std::time::Duration;

use inline_spirv::*;
use vulkan_framework::command_buffer::AccessFlag;
use vulkan_framework::command_buffer::AccessFlags;
use vulkan_framework::command_buffer::AccessFlagsSpecifier;
use vulkan_framework::command_buffer::ClearValues;
use vulkan_framework::command_buffer::ColorClearValues;
use vulkan_framework::command_buffer::CommandBufferRecorder;
use vulkan_framework::command_buffer::ImageMemoryBarrier;
use vulkan_framework::command_buffer::PrimaryCommandBuffer;
use vulkan_framework::command_pool::CommandPool;
use vulkan_framework::compute_pipeline::ComputePipeline;
use vulkan_framework::descriptor_pool::DescriptorPool;
use vulkan_framework::descriptor_pool::DescriptorPoolConcreteDescriptor;
use vulkan_framework::descriptor_pool::DescriptorPoolSizesConcreteDescriptor;
use vulkan_framework::descriptor_set::DescriptorSet;
use vulkan_framework::descriptor_set_layout::DescriptorSetLayout;
use vulkan_framework::device::*;
use vulkan_framework::fence::Fence;
use vulkan_framework::fence::FenceWaitFor;
use vulkan_framework::framebuffer::Framebuffer;
use vulkan_framework::graphics_pipeline::CullMode;
use vulkan_framework::graphics_pipeline::DepthCompareOp;
use vulkan_framework::graphics_pipeline::DepthConfiguration;
use vulkan_framework::graphics_pipeline::FrontFace;
use vulkan_framework::graphics_pipeline::GraphicsPipeline;
use vulkan_framework::graphics_pipeline::PolygonMode;
use vulkan_framework::graphics_pipeline::Rasterizer;
use vulkan_framework::graphics_pipeline::Scissor;
use vulkan_framework::graphics_pipeline::Viewport;
use vulkan_framework::image::AllocatedImage;
use vulkan_framework::image::ConcreteImageDescriptor;
use vulkan_framework::image::Image;
use vulkan_framework::image::Image2DDimensions;
use vulkan_framework::image::ImageDimensions;
use vulkan_framework::image::ImageFlags;
use vulkan_framework::image::ImageFormat;
use vulkan_framework::image::ImageLayout;
use vulkan_framework::image::ImageLayoutSwapchainKHR;
use vulkan_framework::image::ImageMultisampling;
use vulkan_framework::image::ImageTiling;
use vulkan_framework::image::ImageUsage;
use vulkan_framework::image::ImageUsageSpecifier;
use vulkan_framework::image_view::ImageView;
use vulkan_framework::image_view::ImageViewType;
use vulkan_framework::instance::*;
use vulkan_framework::memory_allocator::StackAllocator;
use vulkan_framework::memory_heap::ConcreteMemoryHeapDescriptor;
use vulkan_framework::memory_heap::MemoryHeap;
use vulkan_framework::memory_heap::MemoryHostVisibility;
use vulkan_framework::memory_heap::MemoryType;
use vulkan_framework::memory_pool::MemoryPool;
use vulkan_framework::memory_pool::MemoryPoolFeatures;
use vulkan_framework::pipeline_layout::PipelineLayout;
use vulkan_framework::pipeline_stage::PipelineStage;
use vulkan_framework::pipeline_stage::PipelineStages;
use vulkan_framework::push_constant_range::PushConstanRange;
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;
use vulkan_framework::renderpass::AttachmentDescription;
use vulkan_framework::renderpass::AttachmentLoadOp;
use vulkan_framework::renderpass::AttachmentStoreOp;
use vulkan_framework::renderpass::RenderSubPassDescription;
use vulkan_framework::semaphore::Semaphore;
use vulkan_framework::shader_layout_binding::BindingDescriptor;
use vulkan_framework::shader_layout_binding::BindingType;
use vulkan_framework::shader_layout_binding::NativeBindingType;
use vulkan_framework::shader_stage_access::ShaderStagesAccess;
use vulkan_framework::shaders::{
    compute_shader::ComputeShader, fragment_shader::FragmentShader, vertex_shader::VertexShader,
};
use vulkan_framework::swapchain::CompositeAlphaSwapchainKHR;
use vulkan_framework::swapchain::DeviceSurfaceInfo;
use vulkan_framework::swapchain::PresentModeSwapchainKHR;
use vulkan_framework::swapchain::SurfaceColorspaceSwapchainKHR;
use vulkan_framework::swapchain::SurfaceTransformSwapchainKHR;
use vulkan_framework::swapchain::SwapchainKHR;
use vulkan_framework::swapchain_image::ImageSwapchainKHR;

const COMPUTE_SPV: &[u32] = inline_spirv!(
    r#"
#version 450 core
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform layout(binding=0,rgba32f) writeonly image2D someImage;

layout(push_constant) uniform pushConstants {
    uint width;
    uint height;
} u_pushConstants;

void main() {
    if ((gl_GlobalInvocationID.x < u_pushConstants.width) && (gl_GlobalInvocationID.y < u_pushConstants.height)) {
        const float red_content = mix(0.0, 1.0, float(gl_LocalInvocationID.x) / float(gl_WorkGroupSize.x));
        const float blu_content = mix(0.0, 1.0, float(gl_LocalInvocationID.y) / float(gl_WorkGroupSize.y));
        const vec4 interpolation = vec4(red_content, 0, blu_content, 1.0);
        imageStore(someImage, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), interpolation );
    }
}
"#,
    comp
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

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

fn main() {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_compute");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];
    let device_layers: Vec<String> = vec![];

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 1024;

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
            panic!("Error creating sdl2 window: {err}");
        }
    };

    match window.vulkan_instance_extensions() {
        Ok(required_extensions) => {
            println!("To present frames in this window the following vulkan instance extensions must be enabled: ");

            for (required_instance_extension_index, required_instance_extension_name) in
                required_extensions.iter().enumerate()
            {
                let ext_name = String::from(*required_instance_extension_name);
                instance_extensions.push(ext_name.clone());
                println!("    {required_instance_extension_index}) {ext_name}");
            }
        }
        Err(err) => panic!("Cannot query vulkan instance extensions: {err}"),
    }

    let Ok(instance) = Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
    ) else {
        panic!("Error creating vulkan instance")
    };

    println!("Vulkan instance created");

    let sfc = vulkan_framework::surface::Surface::from_raw(
        instance.clone(),
        window
            .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
            .unwrap(),
    )
    .unwrap();
    println!("Vulkan rendering surface created and registered successfully");

    let Ok(device) = Device::new(
        instance,
        [ConcreteQueueFamilyDescriptor::new(
            vec![
                QueueFamilySupportedOperationType::Compute,
                QueueFamilySupportedOperationType::Present(sfc.clone()),
                QueueFamilySupportedOperationType::Transfer,
            ]
            .as_ref(),
            [1.0f32].as_slice(),
        )]
        .as_slice(),
        device_extensions.as_slice(),
        device_layers.as_slice(),
        Some("Opened Device"),
    ) else {
        panic!("Error opening a suitable device");
    };

    println!("Device opened successfully");

    let queue_family = match QueueFamily::new(device.clone(), 0) {
        Ok(queue_family) => {
            println!("Base queue family obtained successfully from Device");
            queue_family
        }
        Err(err) => panic!("Error opening the base queue family: {err}"),
    };

    let queue = match Queue::new(queue_family.clone(), Some("best queua evah")) {
        Ok(queue) => {
            println!("Queue created successfully");
            queue
        }
        Err(err) => panic!("Error opening a queue from the given QueueFamily: {err}"),
    };

    let memory_heap = match MemoryHeap::new(
        device.clone(),
        ConcreteMemoryHeapDescriptor::new(
            MemoryType::DeviceLocal(Some(MemoryHostVisibility::new(false))),
            1024 * 1024 * 512,
        ),
    ) {
        Ok(memory_heap) => memory_heap,
        Err(err) => {
            panic!("Error creating the memory heap: {err}");
        }
    };

    let descriptor_pool = DescriptorPool::new(
        device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 1, 0, 1, 0, 0, 0, 0, 0, None),
            2, // one for descriptor_set and one for renderquad_descriptor_set
        ),
        Some("My descriptor pool"),
    )
    .unwrap();

    println!("Memory heap created! <3");
    let memory_heap_size = memory_heap.total_size();

    let stack_allocator = match MemoryPool::new(
        memory_heap,
        Arc::new(StackAllocator::new(memory_heap_size)),
        MemoryPoolFeatures::from(&[]),
    ) {
        Ok(mem_pool) => {
            println!("Stack allocator created");
            mem_pool
        }
        Err(err) => {
            println!("Error creating the memory pool: {err}");
            return;
        }
    };

    let swapchain_extent = Image2DDimensions::new(WIDTH, HEIGHT);

    let device_swapchain_info = DeviceSurfaceInfo::new(device.clone(), sfc).unwrap();

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
        println!(
            "Image count {} not supported (the maximum is {}), sticking with the minimum one: {}",
            swapchain_images_count,
            device_swapchain_info.max_image_count(),
            device_swapchain_info.min_image_count()
        );
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
            false, false, false, false, true, false, false, false,
        )),
        swapchain_extent,
        swapchain_images_count,
        1,
    )
    .unwrap();
    println!("Swapchain created!");

    let image_handle = Image::new(
        device.clone(),
        ConcreteImageDescriptor::new(
            ImageDimensions::Image2D {
                extent: Image2DDimensions::new(1024, 1024),
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
    .unwrap();

    let image = AllocatedImage::new(stack_allocator, image_handle).unwrap();

    let image_view = match ImageView::new(
        image.clone(),
        ImageViewType::Image2D,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("ImageView"),
    ) {
        Ok(image_view) => image_view,
        Err(_err) => {
            println!("Error creating image view...");
            return;
        }
    };

    let renderquad_sampler = vulkan_framework::sampler::Sampler::new(
        device.clone(),
        vulkan_framework::sampler::Filtering::Nearest,
        vulkan_framework::sampler::Filtering::Nearest,
        vulkan_framework::sampler::MipmapMode::ModeNearest,
        0.0,
    )
    .unwrap();

    let renderquad_renderpass = vulkan_framework::renderpass::RenderPass::new(
        device.clone(),
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

    let renderquad_image_imput_format = ImageLayout::ShaderReadOnlyOptimal;

    let renderquad_texture_binding_descriptor = BindingDescriptor::new(
        ShaderStagesAccess::graphics(),
        BindingType::Native(NativeBindingType::CombinedImageSampler),
        0,
        1,
    );

    let renderquad_descriptor_set_layout = DescriptorSetLayout::new(
        device.clone(),
        &[renderquad_texture_binding_descriptor.clone()],
    )
    .unwrap();

    let renderquad_pipeline_layout = PipelineLayout::new(
        device.clone(),
        &[renderquad_descriptor_set_layout.clone()],
        &[],
        Some("pipeline_layout"),
    )
    .unwrap();
    println!("Pipeline layout created!");

    let renderquad_descriptor_set =
        DescriptorSet::new(descriptor_pool.clone(), renderquad_descriptor_set_layout).unwrap();

    renderquad_descriptor_set
        .bind_resources(|binder| {
            binder
                .bind_combined_images_samplers(
                    0,
                    &[(
                        ImageLayout::General,
                        image_view.clone(),
                        renderquad_sampler.clone(),
                    )],
                )
                .unwrap()
        })
        .unwrap();

    let renderquad_vertex_shader = VertexShader::new(
        device.clone(),
        &[],
        &[renderquad_texture_binding_descriptor.clone()],
        RENDERQUAD_VERTEX_SPV,
    )
    .unwrap();

    let renderquad_fragment_shader = FragmentShader::new(
        device.clone(),
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

    let resulting_image_shader_binding = BindingDescriptor::new(
        ShaderStagesAccess::compute(),
        BindingType::Native(NativeBindingType::StorageImage),
        0,
        1,
    );

    let image_dimensions_shader_push_constant =
        PushConstanRange::new(0, 8, ShaderStagesAccess::compute());

    let compute_shader = match ComputeShader::new(
        device.clone(),
        //&[image_dimensions_shader_push_constant.clone()],
        //&[resulting_image_shader_binding.clone()],
        COMPUTE_SPV,
    ) {
        Ok(res) => {
            println!("Shader module created");
            res
        }
        Err(_err) => {
            println!("Error creating the compute shader...");
            return;
        }
    };

    let descriptor_set_layout =
        match DescriptorSetLayout::new(device.clone(), &[resulting_image_shader_binding]) {
            Ok(res) => {
                println!("Descriptor set layout created");
                res
            }
            Err(_err) => {
                println!("Error creating the descriptor set layout...");
                return;
            }
        };

    let compute_pipeline_layout = PipelineLayout::new(
        device.clone(),
        &[descriptor_set_layout.clone()],
        &[image_dimensions_shader_push_constant],
        Some("Layout of Example pipeline"),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        None,
        compute_pipeline_layout.clone(),
        (compute_shader, None),
        Some("Example pipeline"),
    )
    .unwrap();

    let command_pool = CommandPool::new(queue_family.clone(), Some("My command pool")).unwrap();

    let descriptor_set = DescriptorSet::new(descriptor_pool, descriptor_set_layout).unwrap();

    descriptor_set
        .bind_resources(|binder| {
            binder
                .bind_storage_images(0, &[(ImageLayout::General, image_view.clone())])
                .unwrap()
        })
        .unwrap();

    let command_buffer =
        PrimaryCommandBuffer::new(command_pool.clone(), Some("my command buffer <3")).unwrap();

    match command_buffer.record_commands(|recorder| {
        recorder.image_barrier(ImageMemoryBarrier::new(
            PipelineStages::from(&[PipelineStage::TopOfPipe], None, None, None),
            AccessFlags::from(AccessFlagsSpecifier::from(&[], None)),
            PipelineStages::from(&[PipelineStage::ComputeShader], None, None, None),
            AccessFlags::from(AccessFlagsSpecifier::from(&[AccessFlag::ShaderWrite], None)),
            image.clone(),
            None,
            None,
            None,
            None,
            None,
            ImageLayout::Undefined,
            ImageLayout::General,
            queue_family.clone(),
            queue_family.clone(),
        ));

        recorder.bind_compute_pipeline(compute_pipeline.clone());

        recorder.bind_descriptor_sets_for_compute_pipeline(
            compute_pipeline_layout.clone(),
            0,
            &[descriptor_set.clone()],
        );

        let data = [unsafe { any_as_u8_slice(&1024u32) }, unsafe {
            any_as_u8_slice(&1024u32)
        }]
        .concat();

        recorder.push_constant_for_compute_shader(
            compute_pipeline_layout.clone(),
            0,
            data.as_slice(),
        );

        recorder.dispatch(32, 32, 1);

        let storage_image_transition_mem_barrier = ImageMemoryBarrier::new(
            PipelineStages::from(&[PipelineStage::TopOfPipe], None, None, None),
            AccessFlags::from(AccessFlagsSpecifier::from(&[], None)),
            PipelineStages::from(&[PipelineStage::Transfer], None, None, None),
            AccessFlags::from(AccessFlagsSpecifier::from(
                &[AccessFlag::TransferRead],
                None,
            )),
            image.clone(),
            None,
            None,
            None,
            None,
            None,
            ImageLayout::General,
            renderquad_image_imput_format,
            queue_family.clone(),
            queue_family.clone(),
        );

        recorder.image_barrier(storage_image_transition_mem_barrier);
    }) {
        Ok(res) => {
            println!("Commands written in the command buffer, there are resources used in that.");
            res
        }
        Err(_err) => {
            println!("Error writing the Command Buffer...");
            return;
        }
    };

    let semaphore = Semaphore::new(device.clone(), Some("MySemaphore")).unwrap();
    println!("Semaphore created");

    let swapchain_images = ImageSwapchainKHR::extract(swapchain.clone()).unwrap();

    let image_available_semaphores = (0..(swapchain_images_count))
        .map(|idx| {
            Semaphore::new(
                device.clone(),
                Some(format!("image_available_semaphores[{idx}]").as_str()),
            )
            .unwrap()
        })
        .collect::<Vec<Arc<Semaphore>>>();

    // this tells me when the present operation can start
    let present_ready = (0..swapchain_images_count)
        .map(|idx| {
            Semaphore::new(
                device.clone(),
                Some(format!("present_ready[{idx}]").as_str()),
            )
            .unwrap()
        })
        .collect::<Vec<Arc<Semaphore>>>();

    let swapchain_fences = (0..swapchain_images_count)
        .map(|idx| {
            Fence::new(
                device.clone(),
                true,
                Some(format!("swapchain_fences[{idx}]").as_str()),
            )
            .unwrap()
        })
        .collect::<Vec<Arc<Fence>>>();

    let present_command_buffers = (0..swapchain_images_count)
        .map(|idx| {
            PrimaryCommandBuffer::new(
                command_pool.clone(),
                Some(format!("present_command_buffers[{idx}]").as_str()),
            )
            .unwrap()
        })
        .collect::<Vec<Arc<PrimaryCommandBuffer>>>();

    let swapchain_images_imageview = (0..(swapchain_images_count))
        .map(|idx| {
            ImageView::new(
                swapchain_images[idx as usize].clone(),
                ImageViewType::Image2D,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(format!("swapchain_images_imageview[{idx}]").as_str()),
            )
            .unwrap()
        })
        .collect::<Vec<Arc<ImageView>>>();

    let rendequad_framebuffers = (0..(swapchain_images_count))
        .map(|idx| {
            Framebuffer::new(
                renderquad_renderpass.clone(),
                &[swapchain_images_imageview[idx as usize].clone()],
                swapchain_extent,
                1,
            )
            .unwrap()
        })
        .collect::<Vec<Arc<Framebuffer>>>();

    let mut current_frame: usize = 0;

    // perform the rendering on the initial image
    {
        let fence = Fence::new(device.clone(), false, Some("MyFence")).unwrap();
        println!("Fence created");
        match queue.submit(&[command_buffer], &[], &[semaphore], fence.clone()) {
            Ok(()) => {
                println!("Command buffer submitted! GPU will work on that!");

                'wait_for_fence: loop {
                    match Fence::wait_for_fences(
                        &[fence.clone()],
                        FenceWaitFor::All,
                        Duration::from_nanos(100),
                    ) {
                        Ok(_) => {
                            fence.reset().unwrap();
                            break 'wait_for_fence;
                        }
                        Err(err) => {
                            if err.is_timeout() {
                                continue 'wait_for_fence;
                            }

                            panic!("Error waiting for device to complete the task. Don't know what to do... Panic!");
                        }
                    }
                }
            }
            Err(err) => {
                panic!("Error submitting the command buffer to the queue: {err} -- No work will be done :(");
            }
        };
    }

    renderquad_descriptor_set
        .bind_resources(|renderquad_binder| {
            renderquad_binder
                .bind_combined_images_samplers(
                    0,
                    &[(
                        renderquad_image_imput_format,
                        image_view.clone(),
                        renderquad_sampler.clone(),
                    )],
                )
                .unwrap()
        })
        .unwrap();

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
                    device.wait_idle().unwrap();
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
                // when submitting wait for the image available semaphore before beginning transfer

                recorder.begin_renderpass(
                    rendequad_framebuffers[swapchain_index as usize].clone(),
                    &[ClearValues::new(Some(ColorClearValues::Vec4(
                        0.0, 0.0, 0.0, 0.0,
                    )))],
                );
                recorder.bind_graphics_pipeline(renderquad_graphics_pipeline.clone(), None, None);
                recorder.bind_descriptor_sets_for_graphics_pipeline(
                    renderquad_pipeline_layout.clone(),
                    0,
                    &[renderquad_descriptor_set.clone()],
                );
                recorder.draw(0, 6, 0, 1);
                recorder.end_renderpass();
            })
            .unwrap();

        queue
            .submit(
                &[present_command_buffers[swapchain_index as usize].clone()],
                &[(
                    PipelineStages::from(&[PipelineStage::Transfer], None, None, None),
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
