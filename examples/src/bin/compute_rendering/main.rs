use inline_spirv::*;
use vulkan_framework::command_buffer::AccessFlag;
use vulkan_framework::command_buffer::AccessFlags;
use vulkan_framework::command_buffer::AccessFlagsSpecifier;
use vulkan_framework::command_buffer::CommandBufferRecorder;
use vulkan_framework::command_buffer::ImageMemoryBarrier;
use vulkan_framework::command_buffer::PrimaryCommandBuffer;
use vulkan_framework::command_pool::CommandPool;
use vulkan_framework::compute_pipeline::ComputePipeline;
use vulkan_framework::compute_shader::ComputeShader;
use vulkan_framework::descriptor_pool::DescriptorPool;
use vulkan_framework::descriptor_pool::DescriptorPoolConcreteDescriptor;
use vulkan_framework::descriptor_pool::DescriptorPoolSizesConcreteDescriptor;
use vulkan_framework::descriptor_set::DescriptorSet;
use vulkan_framework::descriptor_set_layout::DescriptorSetLayout;
use vulkan_framework::device::*;
use vulkan_framework::fence::Fence;
use vulkan_framework::fence::FenceWaiter;
use vulkan_framework::image::ConcreteImageDescriptor;
use vulkan_framework::image::Image;
use vulkan_framework::image::Image2DDimensions;
use vulkan_framework::image::ImageAspect;
use vulkan_framework::image::ImageAspects;
use vulkan_framework::image::ImageDimensions;
use vulkan_framework::image::ImageFlags;
use vulkan_framework::image::ImageFormat;
use vulkan_framework::image::ImageLayout;
use vulkan_framework::image::ImageLayoutSwapchainKHR;
use vulkan_framework::image::ImageMultisampling;
use vulkan_framework::image::ImageSubresourceLayers;
use vulkan_framework::image::ImageTiling;
use vulkan_framework::image::ImageTrait;
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
use vulkan_framework::pipeline_layout::PipelineLayout;
use vulkan_framework::pipeline_stage::PipelineStage;
use vulkan_framework::pipeline_stage::PipelineStages;
use vulkan_framework::push_constant_range::PushConstanRange;
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;
use vulkan_framework::semaphore::Semaphore;
use vulkan_framework::shader_layout_binding::BindingDescriptor;
use vulkan_framework::shader_layout_binding::BindingType;
use vulkan_framework::shader_layout_binding::NativeBindingType;
use vulkan_framework::shader_stage_access::ShaderStageAccess;
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

uniform layout(binding=0,rgba8_snorm) writeonly image2D someImage;

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

            if let Ok(instance) = Instance::new(
                [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
                instance_extensions.as_slice(),
                &engine_name,
                &app_name,
                &api_version,
            ) {
                println!("Vulkan instance created");

                let sfc = vulkan_framework::surface::Surface::from_raw(
                    instance.clone(),
                    window
                        .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
                        .unwrap(),
                )
                .unwrap();
                println!("Vulkan rendering surface created and registered successfully");

                if let Ok(device) = Device::new(
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
                ) {
                    println!("Device opened successfully");

                    match QueueFamily::new(device.clone(), 0) {
                        Ok(queue_family) => {
                            println!("Base queue family obtained successfully from Device");

                            match Queue::new(queue_family.clone(), Some("best queua evah")) {
                                Ok(queue) => {
                                    println!("Queue created successfully");

                                    match MemoryHeap::new(
                                        device.clone(),
                                        ConcreteMemoryHeapDescriptor::new(
                                            MemoryType::DeviceLocal(Some(
                                                MemoryHostVisibility::new(false),
                                            )),
                                            1024 * 1024 * 512,
                                        ),
                                    ) {
                                        Ok(memory_heap) => {
                                            println!("Memory heap created! <3");

                                            let stack_allocator = match MemoryPool::new(
                                                memory_heap,
                                                StackAllocator::new(1024 * 1024 * 1024),
                                            ) {
                                                Ok(mem_pool) => {
                                                    println!("Stack allocator created");
                                                    mem_pool
                                                }
                                                Err(_err) => {
                                                    println!("Error creating the memory pool");
                                                    return;
                                                }
                                            };

                                            let device_swapchain_info =
                                                DeviceSurfaceInfo::new(device.clone(), sfc)
                                                    .unwrap();

                                            let final_format = ImageFormat::b8g8r8a8_srgb;
                                            let color_space =
                                                SurfaceColorspaceSwapchainKHR::SRGBNonlinear;
                                            if !device_swapchain_info
                                                .format_supported(&color_space, &final_format)
                                            {
                                                panic!("Device does not support the most common format. LOL.");
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
                                                    false, true, false, false, true, false, false,
                                                    false,
                                                )),
                                                Image2DDimensions::new(WIDTH, HEIGHT),
                                                4,
                                                1,
                                            )
                                            .unwrap();
                                            println!("Swapchain created!");

                                            let image = Image::new(
                                                stack_allocator,
                                                ConcreteImageDescriptor::new(
                                                    ImageDimensions::Image2D {
                                                        extent: Image2DDimensions::new(1024, 1024),
                                                    },
                                                    ImageUsage::Managed(ImageUsageSpecifier::new(
                                                        true, false, false, true, false, false,
                                                        false, false,
                                                    )),
                                                    ImageMultisampling::SamplesPerPixel1,
                                                    1,
                                                    1,
                                                    final_format,
                                                    ImageFlags::empty(),
                                                    ImageTiling::Optimal,
                                                ),
                                                None,
                                                Some("Test Image"),
                                            ).unwrap();

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

                                            let resulting_image_shader_binding =
                                                BindingDescriptor::new(
                                                    ShaderStageAccess::compute(),
                                                    BindingType::Native(
                                                        NativeBindingType::StorageImage,
                                                    ),
                                                    0,
                                                    1,
                                                );

                                            let image_dimensions_shader_push_constant =
                                                PushConstanRange::new(
                                                    0,
                                                    8,
                                                    ShaderStageAccess::compute(),
                                                );

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
                                                    println!(
                                                        "Error creating the compute shader..."
                                                    );
                                                    return;
                                                }
                                            };

                                            let descriptor_set_layout =
                                                match DescriptorSetLayout::new(
                                                    device.clone(),
                                                    &[resulting_image_shader_binding],
                                                ) {
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
                                            ).unwrap();

                                            let compute_pipeline = ComputePipeline::new(
                                                compute_pipeline_layout.clone(),
                                                (compute_shader, None),
                                                Some("Example pipeline"),
                                            ).unwrap();

                                            let command_pool = CommandPool::new(
                                                queue_family.clone(),
                                                Some("My command pool"),
                                            ).unwrap();

                                            let descriptor_pool = DescriptorPool::new(
                                                device.clone(),
                                                DescriptorPoolConcreteDescriptor::new(
                                                    DescriptorPoolSizesConcreteDescriptor::new(
                                                        0, 0, 0, 1, 0, 0, 0, 0, 0, None,
                                                    ),
                                                    1,
                                                ),
                                                Some("My descriptor pool"),
                                            ).unwrap();

                                            let descriptor_set = DescriptorSet::new(
                                                descriptor_pool,
                                                descriptor_set_layout,
                                            ).unwrap();

                                            let command_buffer = PrimaryCommandBuffer::new(
                                                command_pool.clone(),
                                                Some("my command buffer <3"),
                                            ).unwrap();

                                            if let Err(_error) =
                                                descriptor_set.bind_resources(|binder| {
                                                    binder.bind_storage_images(
                                                        0,
                                                        &[(
                                                            ImageLayout::General,
                                                            image_view.clone(),
                                                        )],
                                                    )
                                                })
                                            {
                                                panic!("error in binding resources");
                                            }

                                            match command_buffer.record_commands(|recorder| {
                                                recorder.image_barrier(ImageMemoryBarrier::new(
                                                    PipelineStages::from(
                                                        &[PipelineStage::TopOfPipe],
                                                        None,
                                                        None,
                                                        None,
                                                    ),
                                                    AccessFlags::from(AccessFlagsSpecifier::from(
                                                        &[],
                                                        None,
                                                    )),
                                                    PipelineStages::from(
                                                        &[PipelineStage::ComputeShader],
                                                        None,
                                                        None,
                                                        None,
                                                    ),
                                                    AccessFlags::from(AccessFlagsSpecifier::from(
                                                        &[AccessFlag::ShaderWrite],
                                                        None,
                                                    )),
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

                                                let descriptor_sets = vec![descriptor_set.clone()];
                                                recorder.bind_compute_pipeline(
                                                    compute_pipeline.clone(),
                                                );

                                                recorder.bind_descriptor_sets(
                                                    compute_pipeline_layout.clone(),
                                                    0,
                                                    descriptor_sets.as_slice(),
                                                );

                                                let data = [
                                                    unsafe { any_as_u8_slice(&1024u32) },
                                                    unsafe { any_as_u8_slice(&1024u32) },
                                                ]
                                                .concat();

                                                recorder.push_constant_for_compute_shader(
                                                    compute_pipeline_layout.clone(),
                                                    0,
                                                    data.as_slice(),
                                                );

                                                recorder.dispatch(32, 32, 1);

                                                let storage_image_transition_mem_barrier =
                                                    ImageMemoryBarrier::new(
                                                        PipelineStages::from(
                                                            &[PipelineStage::TopOfPipe],
                                                            None,
                                                            None,
                                                            None,
                                                        ),
                                                        AccessFlags::from(
                                                            AccessFlagsSpecifier::from(&[], None),
                                                        ),
                                                        PipelineStages::from(
                                                            &[PipelineStage::Transfer],
                                                            None,
                                                            None,
                                                            None,
                                                        ),
                                                        AccessFlags::from(
                                                            AccessFlagsSpecifier::from(
                                                                &[AccessFlag::TransferRead],
                                                                None,
                                                            ),
                                                        ),
                                                        image.clone(),
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        ImageLayout::General,
                                                        ImageLayout::TransferSrcOptimal,
                                                        queue_family.clone(),
                                                        queue_family.clone(),
                                                    );

                                                recorder.image_barrier(
                                                    storage_image_transition_mem_barrier,
                                                );
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

                                            let fence =
                                                Fence::new(device.clone(), false, Some("MyFence"))
                                                    .unwrap();
                                            println!("Fence created");

                                            let semaphore = Semaphore::new(
                                                device.clone(),
                                                false,
                                                Some("MySemaphore"),
                                            )
                                            .unwrap();
                                            println!("Semaphore created");

                                            let swapchain_images =
                                                ImageSwapchainKHR::extract(swapchain.clone())
                                                    .unwrap();

                                            let image_available_semaphores = vec![
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_available_semaphores[0]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_available_semaphores[1]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_available_semaphores[2]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_available_semaphores[3]"),
                                                )
                                                .unwrap(),
                                            ];

                                            let image_rendered_semaphores = vec![
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_rendered_semaphores[0]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_rendered_semaphores[1]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_rendered_semaphores[2]"),
                                                )
                                                .unwrap(),
                                                Semaphore::new(
                                                    device.clone(),
                                                    false,
                                                    Some("image_rendered_semaphores[3]"),
                                                )
                                                .unwrap(),
                                            ];

                                            let swapchain_fences = vec![
                                                Fence::new(
                                                    device.clone(),
                                                    true,
                                                    Some("swapchain_fences[0]"),
                                                )
                                                .unwrap(),
                                                Fence::new(
                                                    device.clone(),
                                                    true,
                                                    Some("swapchain_fences[1]"),
                                                )
                                                .unwrap(),
                                                Fence::new(
                                                    device.clone(),
                                                    true,
                                                    Some("swapchain_fences[2]"),
                                                )
                                                .unwrap(),
                                                Fence::new(
                                                    device.clone(),
                                                    true,
                                                    Some("swapchain_fences[3]"),
                                                )
                                                .unwrap(),
                                            ];

                                            let mut swapchain_fence_waiters = vec![
                                                FenceWaiter::from_fence(
                                                    swapchain_fences[0].clone(),
                                                ),
                                                FenceWaiter::from_fence(
                                                    swapchain_fences[1].clone(),
                                                ),
                                                FenceWaiter::from_fence(
                                                    swapchain_fences[2].clone(),
                                                ),
                                                FenceWaiter::from_fence(
                                                    swapchain_fences[3].clone(),
                                                ),
                                            ];

                                            let present_command_buffers = vec![
                                                PrimaryCommandBuffer::new(
                                                    command_pool.clone(),
                                                    Some("present_command_buffers[0]"),
                                                )
                                                .unwrap(),
                                                PrimaryCommandBuffer::new(
                                                    command_pool.clone(),
                                                    Some("present_command_buffers[1]"),
                                                )
                                                .unwrap(),
                                                PrimaryCommandBuffer::new(
                                                    command_pool.clone(),
                                                    Some("present_command_buffers[2]"),
                                                )
                                                .unwrap(),
                                                PrimaryCommandBuffer::new(
                                                    command_pool,
                                                    Some("present_command_buffers[3]"),
                                                )
                                                .unwrap(),
                                            ];

                                            let mut current_frame: usize = 0;

                                            match queue.submit(
                                                &[command_buffer],
                                                &[],
                                                &[semaphore],
                                                fence,
                                            ) {
                                                Ok(mut fence_waiter) => {
                                                    println!("Command buffer submitted! GPU will work on that!");

                                                    'wait_for_fence: loop {
                                                        match fence_waiter.wait(100u64) {
                                                            Ok(_) => {
                                                                device.wait_idle().unwrap();
                                                                break 'wait_for_fence;
                                                            }
                                                            Err(err) => {
                                                                if err.timeout() {
                                                                    continue 'wait_for_fence;
                                                                }

                                                                panic!("Error waiting for device to complete the task. Don't know what to do... Panic!");
                                                            }
                                                        }
                                                    }

                                                    let mut event_pump =
                                                        sdl_context.event_pump().unwrap();
                                                    'running: loop {
                                                        for event in event_pump.poll_iter() {
                                                            match event {
                                                                sdl2::event::Event::Quit {..} | sdl2::event::Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Escape), .. } => {
                                                                    for i in 0..4 {
                                                                        swapchain_fence_waiters[i]
                                                                            .wait(u64::MAX)
                                                                            .unwrap()
                                                                    }
                                                                    break 'running
                                                                },
                                                                _ => {}
                                                            }
                                                        }

                                                        let swapchain_index = swapchain
                                                            .acquire_next_image_index(
                                                                None,
                                                                Some(
                                                                    image_available_semaphores
                                                                        [current_frame % 4]
                                                                        .clone(),
                                                                ),
                                                                None,
                                                            )
                                                            .unwrap();

                                                        // wait for fence
                                                        swapchain_fence_waiters[current_frame % 4]
                                                            .wait(u64::MAX)
                                                            .unwrap();

                                                        present_command_buffers[current_frame % 4].record_commands(|recorder: &mut CommandBufferRecorder| {
                                                            // when submitting wait for the image available semaphore before beginning transfer

                                                            recorder.image_barrier(
                                                                ImageMemoryBarrier::new(
                                                                    PipelineStages::from(
                                                                        &[PipelineStage::TopOfPipe],
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),
                                                                    AccessFlags::from(AccessFlagsSpecifier::from(&[], None)),
                                                                    PipelineStages::from(
                                                                        &[PipelineStage::Transfer],
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),
                                                                    AccessFlags::from(AccessFlagsSpecifier::from(&[AccessFlag::TransferWrite], None)),
                                                                    swapchain_images[current_frame % 4].clone(),
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    ImageLayout::Undefined,
                                                                    ImageLayout::TransferDstOptimal,
                                                                    queue_family.clone(),
                                                                    queue_family.clone(),
                                                                )
                                                            );

                                                            recorder.copy_image(
                                                                ImageLayout::TransferSrcOptimal,
                                                                ImageSubresourceLayers::new(
                                                                    ImageAspects::from(&[ImageAspect::Color]),
                                                                    0,
                                                                    0,
                                                                    1
                                                                ),
                                                                image.clone(),
                                                                ImageLayout::TransferDstOptimal,
                                                                ImageSubresourceLayers::new(
                                                                    ImageAspects::from(&[ImageAspect::Color]),
                                                                    0,
                                                                    0,
                                                                    1
                                                                ),
                                                                swapchain_images[current_frame % 4].clone(),
                                                                image.dimensions(),
                                                            );

                                                            recorder.image_barrier(
                                                                ImageMemoryBarrier::new(
                                                                    PipelineStages::from(
                                                                        &[PipelineStage::Transfer],
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),
                                                                    AccessFlags::from(AccessFlagsSpecifier::from(&[AccessFlag::TransferWrite], None)),
                                                                    PipelineStages::from(
                                                                        &[PipelineStage::BottomOfPipe],
                                                                        None,
                                                                        None,
                                                                        None
                                                                    ),
                                                                    AccessFlags::from(AccessFlagsSpecifier::from(&[], None)),
                                                                    swapchain_images[current_frame % 4].clone(),
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    None,
                                                                    ImageLayout::TransferDstOptimal,
                                                                    ImageLayout::SwapchainKHR(ImageLayoutSwapchainKHR::PresentSrc),
                                                                    queue_family.clone(),
                                                                    queue_family.clone(),
                                                                )
                                                            );
                                                        }).unwrap();

                                                        swapchain_fence_waiters
                                                            [current_frame % 4] = queue
                                                            .submit(
                                                                &[present_command_buffers
                                                                    [current_frame % 4]
                                                                    .clone()],
                                                                &[(
                                                                    PipelineStages::from(
                                                                        &[PipelineStage::Transfer],
                                                                        None,
                                                                        None,
                                                                        None,
                                                                    ),
                                                                    image_available_semaphores
                                                                        [current_frame % 4]
                                                                        .clone(),
                                                                )],
                                                                &[image_rendered_semaphores
                                                                    [current_frame % 4]
                                                                    .clone()],
                                                                swapchain_fences[current_frame % 4]
                                                                    .clone(),
                                                            )
                                                            .unwrap();

                                                        swapchain
                                                            .queue_present(
                                                                queue.clone(),
                                                                swapchain_index,
                                                                &[image_rendered_semaphores
                                                                    [current_frame % 4]
                                                                    .clone()],
                                                            )
                                                            .unwrap();

                                                        current_frame = swapchain_index as usize;
                                                    }
                                                }
                                                Err(_) => {
                                                    println!("Error submitting the command buffer to the queue. No work will be done :(");
                                                }
                                            }
                                        }
                                        Err(_err) => {
                                            println!("Error creating the memory heap :(");
                                        }
                                    }
                                }
                                Err(_err) => {
                                    println!("Error opening a queue from the given QueueFamily");
                                }
                            }
                        }
                        Err(_err) => {
                            println!("Error opening the base queue family");
                        }
                    }
                } else {
                    println!("Error opening a suitable device");
                }
            } else {
                println!("Error creating vulkan instance");
            }
        }
        Err(err) => {
            println!("Error creating sdl2 window: {}", err);
        }
    }
}
