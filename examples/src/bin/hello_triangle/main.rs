use vulkan_framework::{
    device::*, instance::*, memory_allocator::*, memory_heap::*, memory_pool::MemoryPool, queue::*,
    queue_family::*,
};

fn main() {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];
    let engine_name = String::from("None");
    let app_name = String::from("hello_window");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];
    let device_layers: Vec<String> = vec![];

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    {
        // this parenthesis contains the window handle and closes before calling deinitialize(). This is important as Window::drop() MUST be called BEFORE calling deinitialize()!!!
        // create a sdl2 window
        match video_subsystem.window("Window", 800, 600).vulkan().build() {
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

                if let Ok(instance) = Instance::new(
                    [
                        String::from("VK_LAYER_KHRONOS_validation")
                    ].as_slice(),
                    instance_extensions.as_slice(),
                    &engine_name,
                    &app_name,
                    &api_version,
                ) {
                    println!("Vulkan instance created");

                    match window
                        .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
                    {
                        Ok(surface_handle) => {
                            println!("Vulkan rendering surface created successfully");

                            match vulkan_framework::surface::Surface::from_raw(
                                instance.clone(),
                                surface_handle,
                            ) {
                                Ok(sfc) => {
                                    //let supported_ops = ;
                                    let required_queues = vec![ConcreteQueueFamilyDescriptor::new(
                                        vec![
                                            QueueFamilySupportedOperationType::Graphics,
                                            QueueFamilySupportedOperationType::Transfer,
                                            QueueFamilySupportedOperationType::Present(sfc.clone()),
                                        ]
                                        .as_ref(),
                                        [1.0f32].as_slice(),
                                    )];

                                    println!("Surface registered");

                                    {
                                        let device_result = Device::new(
                                            instance.clone(),
                                            [
                                                ConcreteQueueFamilyDescriptor::new(
                                                    vec![
                                                        QueueFamilySupportedOperationType::Graphics,
                                                        QueueFamilySupportedOperationType::Transfer,
                                                        QueueFamilySupportedOperationType::Present(sfc.clone()),
                                                    ]
                                                    .as_ref(),
                                                    [1.0f32].as_slice(),
                                                )
                                            ].as_slice(),
                                            [ConcreteMemoryHeapDescriptor::new(
                                                MemoryType::DeviceLocal(None),
                                                1024 * 1024 * 1024 * 2, // 2GB of memory!
                                            )].as_ref(),
                                            device_extensions.as_slice().as_ref(),
                                            device_layers.as_slice().as_ref(),
                                            Some("Opened Device"),
                                        );

                                        match device_result {
                                            Ok(dev) => {
                                                println!("Device opened successfully");

                                                match QueueFamily::new(dev.clone(), 0) {
                                                    Ok(queue_family) => {
                                                        println!("Base queue family obtained successfully from Device");

                                                        match Queue::new(
                                                            queue_family.clone(),
                                                            Some("best queua evah"),
                                                        ) {
                                                            Ok(queue) => {
                                                                println!(
                                                                    "Queue created successfully"
                                                                );

                                                                match MemoryHeap::new(
                                                                    dev.clone(),
                                                                    0,
                                                                ) {
                                                                    Ok(memory_heap) => {
                                                                        println!("Memory heap created! <3");

                                                                        let stack_allocator = match MemoryPool::new(
                                                                            memory_heap.clone(),
                                                                            StackAllocator::new(
                                                                                1024 * 1024 * 1024 * 1,
                                                                            ),
                                                                        ) {
                                                                            Ok(mem_pool) => {
                                                                                println!("Stack allocator created");
                                                                                mem_pool
                                                                            }
                                                                            Err(_err) => {
                                                                                println!("Error creating the memory pool");
                                                                                return ()
                                                                            }
                                                                        };

                                                                        let default_allocator = match MemoryPool::new(
                                                                            memory_heap.clone(),
                                                                            StackAllocator::new(
                                                                                1024 * 1024 * 1024 * 1,
                                                                            ),
                                                                        ) {
                                                                            Ok(mem_pool) => {
                                                                                println!("Default allocator created");
                                                                                mem_pool
                                                                            }
                                                                            Err(_err) => {
                                                                                println!("Error creating the memory pool");
                                                                                return ()
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
                                                        println!(
                                                            "Error opening the base queue family"
                                                        );
                                                    }
                                                }
                                            }
                                            Err(_err) => {
                                                println!("Error opening a suitable device");
                                            }
                                        };
                                    }
                                }
                                Err(_err) => {
                                    println!("Error registering the given surface");
                                }
                            }
                        }
                        Err(err) => {
                            println!("Error creating vulkan rendering surface: {}", err);
                        }
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
}
