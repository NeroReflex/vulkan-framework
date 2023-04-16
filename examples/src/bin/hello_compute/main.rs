use vulkan_framework::device::*;
use vulkan_framework::image::ConcreteImageDescriptor;
use vulkan_framework::image::Image;
use vulkan_framework::image::Image2DDimensions;
use vulkan_framework::image::ImageDimensions;
use vulkan_framework::image::ImageFlags;
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
use vulkan_framework::queue::Queue;
use vulkan_framework::queue_family::*;

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

                    match Queue::new(queue_family, Some("best queua evah")) {
                        Ok(_queue) => {
                            println!("Queue created successfully");

                            match MemoryHeap::new(
                                device,
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
                                            return ;
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
                                                        ImageFlags::empty()
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
                                    ) {
                                        Ok(image_view) => image_view,
                                        Err(_err) => {
                                            println!("Error creating image view...");
                                            return ;
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
