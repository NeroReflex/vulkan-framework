use inline_spirv::*;
use vulkan_framework::command_buffer::PrimaryCommandBuffer;
use vulkan_framework::command_pool::CommandPool;
use vulkan_framework::compute_pipeline::ComputePipeline;
use vulkan_framework::compute_shader::ComputeShader;
use vulkan_framework::descriptor_set_layout::DescriptorSetLayout;
use vulkan_framework::device::*;
use vulkan_framework::image::ConcreteImageDescriptor;
use vulkan_framework::image::Image;
use vulkan_framework::image::Image2DDimensions;
use vulkan_framework::image::ImageDimensions;
use vulkan_framework::image::ImageFlags;
use vulkan_framework::image::ImageTiling;
use vulkan_framework::image::ImageUsage;
use vulkan_framework::image::ImageUsageSpecifier;
use vulkan_framework::image_view::ImageView;
use vulkan_framework::image_view::ImageViewType;
use vulkan_framework::instance::*;
use vulkan_framework::memory_allocator::StackAllocator;
use vulkan_framework::memory_heap::ConcreteMemoryHeapDescriptor;
use vulkan_framework::memory_heap::MemoryHeap;
use vulkan_framework::memory_heap::MemoryType;
use vulkan_framework::memory_pool::MemoryPool;
use vulkan_framework::pipeline_layout::PipelineLayout;
use vulkan_framework::push_constant_range::PushConstanRange;
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;
use vulkan_framework::shader_layout_binding::BindingDescriptor;
use vulkan_framework::shader_layout_binding::BindingType;
use vulkan_framework::shader_layout_binding::NativeBindingType;
use vulkan_framework::shader_stage_access::ShaderStageAccess;

fn main() {
    let instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_compute");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![];
    let device_layers: Vec<String> = vec![];

    if let Ok(instance) = Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
    ) {
        println!("Vulkan instance created");

        if let Ok(device) = Device::new(
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
        ) {
            println!("Device opened successfully");

            match QueueFamily::new(device.clone(), 0) {
                Ok(queue_family) => {
                    println!("Base queue family obtained successfully from Device");

                    match Queue::new(queue_family.clone(), Some("best queua evah")) {
                        Ok(_queue) => {
                            println!("Queue created successfully");

                            match MemoryHeap::new(
                                device.clone(),
                                ConcreteMemoryHeapDescriptor::new(
                                    MemoryType::DeviceLocal(None),
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

                                    let image = match Image::new(
                                                    stack_allocator,
                                                    ConcreteImageDescriptor::new(
                                                        ImageDimensions::Image2D {extent: Image2DDimensions::new(100, 100)},
                                                        ImageUsage::Managed(
                                                            ImageUsageSpecifier::new(
                                                                true,
                                                                false,
                                                                false,
                                                                true,
                                                                false,
                                                                false,
                                                                false,
                                                                false
                                                            )
                                                        ),
                                                        None,
                                                        1,
                                                        1,
                                                        vulkan_framework::image::ImageFormat::r32g32b32a32_sfloat,
                                                        ImageFlags::empty(),
                                                        ImageTiling::Optimal
                                                    ),
                                                    None,
                                                    Some("Test Image")
                                                ) {
                                                    Ok(img) => {
                                                        println!("Image created");
                                                        img
                                                    },
                                                    Err(_err) => {
                                                        println!("Error creating image...");
                                                        return
                                                    }
                                                };

                                    let _image_view = match ImageView::new(
                                        image,
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
                                        ShaderStageAccess::compute(),
                                        BindingType::Native(NativeBindingType::StorageImage),
                                        0,
                                        1,
                                    );

                                    let image_dimensions_shader_push_constant =
                                        PushConstanRange::new(0, 8, ShaderStageAccess::compute());

                                    let spv: &'static [u32] = inline_spirv!(
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
                                            imageStore(someImage, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), vec4(1.0, 1.0, 1.0, 1.0));
                                        }
                                    }
                                    "#,
                                        comp
                                    );

                                    let compute_shader = match ComputeShader::new(
                                        device.clone(),
                                        &[image_dimensions_shader_push_constant.clone()],
                                        &[resulting_image_shader_binding.clone()],
                                        spv,
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

                                    let descriptor_set_layout = match DescriptorSetLayout::new(
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

                                    let compute_pipeline_layout = match PipelineLayout::new(
                                        device,
                                        &[descriptor_set_layout],
                                        &[image_dimensions_shader_push_constant],
                                        Some("Layout of Example pipeline")
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
                                        compute_pipeline_layout,
                                        compute_shader,
                                        None,
                                        Some("Example pipeline")
                                    ) {
                                        Ok(res) => {
                                            println!("Compute pipeline created");
                                            res
                                        }
                                        Err(_err) => {
                                            println!("Error creating the pipeline...");
                                            return;
                                        }
                                    };

                                    let command_pool = match CommandPool::new(queue_family.clone(), Some("My command pool")) {
                                        Ok(res) => {
                                            println!("Command Pool created");
                                            res
                                        }
                                        Err(_err) => {
                                            println!("Error creating the Command Pool...");
                                            return;
                                        }
                                    };

                                    let command_buffer = match PrimaryCommandBuffer::new(command_pool.clone(), Some("my command buffer <3")) {
                                        Ok(res) => {
                                            println!("Primary Command Buffer created");
                                            res
                                        }
                                        Err(_err) => {
                                            println!("Error creating the Primary Command Buffer...");
                                            return;
                                        }
                                    };


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
