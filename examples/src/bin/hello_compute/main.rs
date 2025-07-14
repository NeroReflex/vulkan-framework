use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use inline_spirv::*;
use vulkan_framework::command_buffer::AccessFlag;
use vulkan_framework::command_buffer::AccessFlags;
use vulkan_framework::command_buffer::AccessFlagsSpecifier;
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
use vulkan_framework::image::AllocatedImage;
use vulkan_framework::image::ConcreteImageDescriptor;
use vulkan_framework::image::Image;
use vulkan_framework::image::Image2DDimensions;
use vulkan_framework::image::ImageDimensions;
use vulkan_framework::image::ImageFlags;
use vulkan_framework::image::ImageLayout;
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
use vulkan_framework::memory_pool::MemoryPoolBacked;
use vulkan_framework::memory_pool::MemoryPoolFeatures;
use vulkan_framework::pipeline_layout::PipelineLayout;
use vulkan_framework::pipeline_stage::PipelineStage;
use vulkan_framework::pipeline_stage::PipelineStages;
use vulkan_framework::push_constant_range::PushConstanRange;
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;
use vulkan_framework::shader_layout_binding::BindingDescriptor;
use vulkan_framework::shader_layout_binding::BindingType;
use vulkan_framework::shader_layout_binding::NativeBindingType;
use vulkan_framework::shader_stage_access::ShaderStagesAccess;
use vulkan_framework::shaders::compute_shader::ComputeShader;

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

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

fn main() {
    let instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_compute");

    let device_extensions: Vec<String> = vec![];
    let device_layers: Vec<String> = vec![];

    let Ok(instance) = Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
    ) else {
        panic!("Error creating vulkan instance");
    };

    println!("Vulkan instance created");

    let Ok(device) = Device::new(
        instance,
        [ConcreteQueueFamilyDescriptor::new(
            vec![
                QueueFamilySupportedOperationType::Compute,
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

    let Ok(queue_family) = QueueFamily::new(device.clone(), 0) else {
        panic!("Error creating the basic queue family");
    };

    println!("Base queue family obtained successfully from Device");

    let Ok(queue) = Queue::new(queue_family.clone(), Some("best queua evah")) else {
        panic!("Error opening a queue from the given QueueFamily");
    };

    println!("Queue created successfully");

    let image_handle = match Image::new(
        device.clone(),
        ConcreteImageDescriptor::new(
            ImageDimensions::Image2D {
                extent: Image2DDimensions::new(1024, 1024),
            },
            ImageUsage::Managed(ImageUsageSpecifier::new(
                true, false, false, true, false, false, false, false,
            )),
            ImageMultisampling::SamplesPerPixel1,
            1,
            1,
            vulkan_framework::image::ImageFormat::r32g32b32a32_sfloat,
            ImageFlags::empty(),
            ImageTiling::Linear,
        ),
        None,
        Some("Test Image"),
    ) {
        Ok(img) => {
            println!("Image created");
            img
        }
        Err(err) => {
            panic!("Error creating image: {err}")
        }
    };

    let Ok(memory_heap) = MemoryHeap::new(
        device.clone(),
        ConcreteMemoryHeapDescriptor::new(
            MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                cached: false,
            })),
            1024 * 1024 * 512, // Memory heap with at least 512MiB of memory
        ),
        &[&image_handle],
    ) else {
        panic!("Error creating the memory heap :(");
    };
    println!("Memory heap created! <3");

    let stack_allocator = match MemoryPool::new(
        memory_heap,
        Arc::new(StackAllocator::new(1024 * 1024 * 128)), // of the 512MiB of the heap this memory pool will manage 128MiB
        MemoryPoolFeatures::from(&[]),
    ) {
        Ok(mem_pool) => {
            println!("Stack allocator created");
            mem_pool
        }
        Err(err) => {
            panic!("Error creating the memory pool: {err}");
        }
    };

    let image = match AllocatedImage::new(stack_allocator.clone(), image_handle) {
        Ok(img) => img,
        Err(err) => panic!("Error allocating the image: {err}"),
    };

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

    let compute_pipeline_layout = match PipelineLayout::new(
        device.clone(),
        &[descriptor_set_layout.clone()],
        &[image_dimensions_shader_push_constant],
        Some("Layout of Example pipeline"),
    ) {
        Ok(res) => {
            println!("Pipeline layout created");
            res
        }
        Err(_err) => {
            println!("Error creating the pipeline layout...");
            return;
        }
    };

    let compute_pipeline = match ComputePipeline::new(
        None,
        compute_pipeline_layout.clone(),
        (compute_shader, None),
        Some("Example pipeline"),
    ) {
        Ok(res) => {
            println!("Compute pipeline created");
            res
        }
        Err(err) => {
            panic!("Error creating the pipeline: {err}");
        }
    };

    let command_pool = match CommandPool::new(queue_family.clone(), Some("My command pool")) {
        Ok(res) => {
            println!("Command Pool created");
            res
        }
        Err(err) => {
            panic!("Error creating the Command Pool: {err}");
        }
    };

    let descriptor_pool = match DescriptorPool::new(
        device.clone(),
        DescriptorPoolConcreteDescriptor::new(
            DescriptorPoolSizesConcreteDescriptor::new(0, 0, 0, 1, 0, 0, 0, 0, 0, None),
            1,
        ),
        Some("My descriptor pool"),
    ) {
        Ok(res) => {
            println!("Descriptor Pool created");
            res
        }
        Err(err) => {
            panic!("Error creating the Descriptor Pool: {err}");
        }
    };

    let descriptor_set = match DescriptorSet::new(descriptor_pool, descriptor_set_layout) {
        Ok(res) => {
            println!("Descriptor Set created");
            res
        }
        Err(_err) => {
            panic!("Error creating the Descriptor Set...");
        }
    };

    let command_buffer = match PrimaryCommandBuffer::new(command_pool, Some("my command buffer <3"))
    {
        Ok(res) => {
            println!("Primary Command Buffer created");
            res
        }
        Err(err) => {
            panic!("Error creating the Primary Command Buffer: {err}");
        }
    };

    if let Err(_error) = descriptor_set.bind_resources(|binder| {
        binder
            .bind_storage_images(0, &[(ImageLayout::General, image_view.clone())])
            .unwrap()
    }) {
        panic!("error in binding resources");
    }

    match command_buffer.record_commands(|recorder| {
        recorder.image_barrier(ImageMemoryBarrier::new(
            PipelineStages::from(&[PipelineStage::TopOfPipe], None, None, None),
            AccessFlags::from(AccessFlagsSpecifier::from(&[AccessFlag::MemoryRead], None)),
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

        let descriptor_sets = vec![descriptor_set.clone()];
        recorder.bind_compute_pipeline(compute_pipeline.clone());

        recorder.bind_descriptor_sets_for_compute_pipeline(
            compute_pipeline_layout.clone(),
            0,
            descriptor_sets.as_slice(),
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

        recorder.dispatch(32, 32, 1)
    }) {
        Ok(res) => {
            println!("Commands written in the command buffer, there are resources used in that.");
            res
        }
        Err(err) => {
            panic!("Error writing the Command Buffer: {err}");
        }
    };

    let fence = match Fence::new(device, false, Some("MyFence")) {
        Ok(res) => {
            println!("Fence created");
            res
        }
        Err(err) => {
            panic!("Error creating the Primary Command Buffer: {err}");
        }
    };

    let Ok(_) = queue.submit(&[command_buffer], &[], &[], fence.clone()) else {
        panic!("Error submitting the command buffer to the queue. No work will be done :(");
    };

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

    let Ok(image_raw_data) = stack_allocator
        .read_raw_data::<[f32; 4]>(image.allocation_offset(), image.allocation_size())
    else {
        panic!("Error copying data from the GPU memory :(");
    };

    println!(
        "Image in GPU memory is {} bytes long, {} pixels in rgba32f were retrieved!",
        image.allocation_size(),
        image_raw_data.len()
    );

    let path = std::path::Path::new("image.pfm");
    let display = path.display();

    let mut file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(why) => panic!("couldn't open {}: {}", display, why),
    };

    let rgb_data = image_raw_data
        .iter()
        .map(|f| [f[0], f[1], f[2]])
        .collect::<Vec<[f32; 3]>>();

    if let Err(err) = write!(file, "PF\n1024 1024\n-1.0\n") {
        panic!("Unexpected error while writing the resulting image header: {err}");
    }

    for rgb in rgb_data.iter() {
        let slice = unsafe {
            std::slice::from_raw_parts(
                rgb.as_ptr() as *const u8,
                rgb.len() * std::mem::size_of::<[f32; 3]>(),
            )
        };

        if let Err(err) = file.write(slice) {
            panic!(
                "Unexpected error while writing the resulting image data: {}",
                err
            );
        }
    }
}
