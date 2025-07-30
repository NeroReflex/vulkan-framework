use std::sync::Arc;
use std::time::Duration;

use inline_spirv::*;
use vulkan_framework::command_buffer::{
    ClearValues, ColorClearValues, CommandBufferRecorder, ImageMemoryBarrier, MemoryAccess,
    MemoryAccessAs, PrimaryCommandBuffer,
};
use vulkan_framework::command_pool::CommandPool;
use vulkan_framework::compute_pipeline::ComputePipeline;
use vulkan_framework::descriptor_pool::{
    DescriptorPool, DescriptorPoolConcreteDescriptor, DescriptorPoolSizesConcreteDescriptor,
};
use vulkan_framework::descriptor_set::DescriptorSet;
use vulkan_framework::descriptor_set_layout::DescriptorSetLayout;
use vulkan_framework::device::*;
use vulkan_framework::dynamic_rendering::{
    AttachmentLoadOp, AttachmentStoreOp, DynamicRendering, DynamicRenderingAttachment,
};
use vulkan_framework::fence::Fence;
use vulkan_framework::graphics_pipeline::{
    CullMode, DepthCompareOp, DepthConfiguration, FrontFace, GraphicsPipeline, PolygonMode,
    Rasterizer, Scissor, Viewport,
};
use vulkan_framework::image::{
    AllocatedImage, CommonImageFormat, ConcreteImageDescriptor, Image, Image2DDimensions,
    ImageDimensions, ImageFlags, ImageFormat, ImageLayout, ImageLayoutSwapchainKHR,
    ImageMultisampling, ImageSubresourceRange, ImageTiling, ImageTrait, ImageUsage, ImageUseAs,
};
use vulkan_framework::image_view::{ImageView, ImageViewType};
use vulkan_framework::instance::*;
use vulkan_framework::memory_allocator::StackAllocator;
use vulkan_framework::memory_heap::{ConcreteMemoryHeapDescriptor, MemoryRequirements};
use vulkan_framework::memory_heap::{MemoryHeap, MemoryType};
use vulkan_framework::memory_pool::{MemoryPool, MemoryPoolFeatures};
use vulkan_framework::memory_requiring::MemoryRequiring;
use vulkan_framework::pipeline_layout::PipelineLayout;
use vulkan_framework::pipeline_stage::{PipelineStage, PipelineStages};
use vulkan_framework::push_constant_range::PushConstanRange;
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;
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

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];

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

    let swapchain_extent = Image2DDimensions::new(WIDTH, HEIGHT);

    let device_swapchain_info = DeviceSurfaceInfo::new(device.clone(), sfc).unwrap();

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
        PresentModeSwapchainKHR::FIFO,
        color_space,
        CompositeAlphaSwapchainKHR::Opaque,
        SurfaceTransformSwapchainKHR::Identity,
        true,
        final_format,
        ImageUsage::from([ImageUseAs::ColorAttachment].as_slice()),
        swapchain_extent,
        swapchain_images_count,
        1,
    )
    .unwrap();
    println!("Swapchain created!");

    let image = Image::new(
        device.clone(),
        ConcreteImageDescriptor::new(
            ImageDimensions::Image2D {
                extent: Image2DDimensions::new(1024, 1024),
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
    .unwrap();

    let memory_heap = match MemoryHeap::new(
        device.clone(),
        ConcreteMemoryHeapDescriptor::new(MemoryType::DeviceLocal(None), 1024 * 1024 * 512),
        MemoryRequirements::try_from([&image as &dyn MemoryRequiring].as_slice()).unwrap(),
    ) {
        Ok(memory_heap) => memory_heap,
        Err(err) => {
            panic!("Error creating the memory heap: {err}");
        }
    };

    let stack_allocator = match MemoryPool::new(
        memory_heap.clone(),
        Arc::new(StackAllocator::new(memory_heap.total_size())),
        MemoryPoolFeatures::from([].as_slice()),
    ) {
        Ok(mem_pool) => {
            println!("Stack allocator created");
            mem_pool
        }
        Err(err) => {
            panic!("Error creating the memory pool: {err}");
        }
    };

    let image = AllocatedImage::new(stack_allocator, image).unwrap();

    let image_view = match ImageView::new(
        image.clone(),
        Some(ImageViewType::Image2D),
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
        Err(err) => {
            panic!("Error creating image view: {err}");
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

    //&[swapchain_images_imageview[idx as usize].clone()],
    //            swapchain_extent,

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

    match command_buffer.record_one_time_submit(|recorder| {
        recorder.image_barriers(
            [ImageMemoryBarrier::new(
                PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                MemoryAccess::from([].as_slice()),
                PipelineStages::from([PipelineStage::ComputeShader].as_slice()),
                MemoryAccess::from([MemoryAccessAs::ShaderWrite].as_slice()),
                ImageSubresourceRange::from(image.clone() as Arc<dyn ImageTrait>),
                ImageLayout::Undefined,
                ImageLayout::General,
                queue_family.clone(),
                queue_family.clone(),
            )]
            .as_slice(),
        );

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

        recorder.push_constant_for_compute_pipeline(
            compute_pipeline_layout.clone(),
            0,
            data.as_slice(),
        );

        recorder.dispatch(32, 32, 1);

        let storage_image_transition_mem_barrier = ImageMemoryBarrier::new(
            PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
            MemoryAccess::from([].as_slice()),
            PipelineStages::from([PipelineStage::Transfer].as_slice()),
            MemoryAccess::from([MemoryAccessAs::TransferRead].as_slice()),
            ImageSubresourceRange::from(image.clone() as Arc<dyn ImageTrait>),
            ImageLayout::General,
            renderquad_image_imput_format,
            queue_family.clone(),
            queue_family.clone(),
        );

        recorder.image_barriers([storage_image_transition_mem_barrier].as_slice());
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
                false,
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

    let mut swapchain_images = vec![];
    for index in 0..swapchain_images_count {
        swapchain_images.push(SwapchainKHR::image(swapchain.clone(), index).unwrap());
    }

    let mut swapchain_images_imageview = vec![];
    for (idx, img) in swapchain_images.iter().enumerate() {
        let swapchain_image_name = format!("swapchain_images_imageview[{idx}]");
        swapchain_images_imageview.push(
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

    let mut current_frame: usize = 0;

    // perform the rendering on the initial image
    {
        let fence = Fence::new(device.clone(), false, Some("MyFence")).unwrap();
        println!("Fence created");
        match queue.submit(&[command_buffer], &[], &[semaphore], fence.clone()) {
            Ok(_) => {
                println!("Command buffer submitted! GPU will work on that!");
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
                        // exiting the loop will destroy fence_waiter
                        device.wait_idle().unwrap();
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
                    // when submitting wait for the image available semaphore before beginning transfer

                    // Transition the final swapchain image into color attachment optimal layout,
                    // so that the graphics pipeline has it in the best format, and the final barrier
                    // can transition it from that layout to the one suitable for presentation on the
                    // swapchain
                    recorder.image_barriers(
                        [ImageMemoryBarrier::new(
                            PipelineStages::from([PipelineStage::TopOfPipe].as_slice()),
                            MemoryAccess::from([].as_slice()),
                            PipelineStages::from([PipelineStage::AllGraphics].as_slice()),
                            MemoryAccess::from([MemoryAccessAs::ColorAttachmentWrite].as_slice()),
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

                    let rendering_color_attachments = [DynamicRenderingAttachment::new(
                        swapchain_images_imageview[swapchain_index as usize].clone(),
                        ImageLayout::ColorAttachmentOptimal,
                        ClearValues::new(Some(ColorClearValues::Vec4(0.0, 0.0, 0.0, 0.0))),
                        AttachmentLoadOp::Clear,
                        AttachmentStoreOp::Store,
                    )];
                    recorder.graphics_rendering(
                        swapchain.images_extent(),
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
                                &[renderquad_descriptor_set.clone()],
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
                            PipelineStages::from([PipelineStage::Transfer].as_slice()),
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
