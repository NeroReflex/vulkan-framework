use std::rc::Rc;

use vulkan_framework;
use vulkan_framework_sdl2_glue;

fn main() {
    let mut instance_extensions = Vec::<String>::new();
    let engine_name = String::from("None");
    let app_name = String::from("hello_window");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![];
    let device_layers: Vec<String> = vec![];
    let required_queues: Vec<vulkan_framework::queue_family::ConcreteQueueFamilyDescriptor> = vec![
        vulkan_framework::queue_family::ConcreteQueueFamilyDescriptor::new(
            [vulkan_framework::queue_family::QueueFamilySupportedOperationType::Compute].as_slice(),
            [1.0f32].as_slice(),
        ),
    ];

    // initialize sdl2 context
    vulkan_framework_sdl2_glue::init();

    {
        // this parenthesis contains the window handle and closes before calling deinitialize(). This is important as Window::drop() MUST be called BEFORE calling deinitialize()!!!
        // create a sdl2 window
        match vulkan_framework_sdl2_glue::window::Window::new(&app_name, 640, 480, None, None) {
            Ok(mut window) => {
                println!("SDL2 window created");

                match window.get_vulkan_instance_extensions() {
                    Ok(required_extensions) => {
                        println!("To present frames in this window the following vulkan instance extensions must be enabled: ");

                        for (required_instance_extension_index, required_instance_extension_name) in
                            required_extensions.iter().enumerate()
                        {
                            instance_extensions.push(required_instance_extension_name.clone());
                            println!(
                                "    {}) {}",
                                required_instance_extension_index,
                                *required_instance_extension_name
                            );
                        }
                    }
                    Err(err) => {
                        println!("Error fetching required vulkan instance extensions!");
                    }
                }

                if let Ok(instance) = vulkan_framework::instance::Instance::new(
                    instance_extensions.as_slice(),
                    &engine_name,
                    &app_name,
                    &api_version,
                    true,
                ) {
                    println!("Vulkan instance created");

                    match window.create_surface(instance.native_handle()) {
                        Ok(surface) => {
                            println!("Vulkan rendering surface created successfully");

                            if let Ok(_device) = vulkan_framework::device::Device::new(
                                Rc::downgrade(&instance),
                                required_queues.as_slice().as_ref(),
                                device_extensions.as_slice().as_ref(),
                                device_layers.as_slice().as_ref(),
                                |instance, phy_dev, queue_family| -> bool { true },
                            ) {
                                println!("Device opened successfully");
                            } else {
                                println!("Error opening a suitable device");
                            }

                            // TODO: destroy surface
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

    // deinitialize sdl2 context
    vulkan_framework_sdl2_glue::deinit();
}
