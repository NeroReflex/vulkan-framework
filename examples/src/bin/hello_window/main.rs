use vulkan_framework::device::*;
use vulkan_framework::instance::*;
use vulkan_framework::queue_family::*;

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
                    instance_extensions.as_slice(),
                    &engine_name,
                    &app_name,
                    &api_version,
                    true,
                ) {
                    println!("Vulkan instance created");

                    match window
                        .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
                    {
                        Ok(surface_handle) => {
                            println!("Vulkan rendering surface created successfully");

                            match vulkan_framework::surface::Surface::from_raw(
                                &instance,
                                surface_handle,
                            ) {
                                Ok(sfc) => {
                                    //let supported_ops = ;
                                    let required_queues = [ConcreteQueueFamilyDescriptor::new(
                                        vec![
                                                QueueFamilySupportedOperationType::Graphics,
                                                QueueFamilySupportedOperationType::Transfer,
                                                QueueFamilySupportedOperationType::Present(&sfc),
                                            ],
                                            [1.0f32].as_slice(),
                                        )];

                                    println!("Surface registered");

                                    if let Ok(_device) = Device::new(
                                        &instance,
                                        &required_queues /* .as_slice()*/,
                                        device_extensions.as_slice().as_ref(),
                                        device_layers.as_slice().as_ref(),
                                        Some("Opened Device"),
                                    ) {
                                        println!("Device opened successfully");
                                    } else {
                                        println!("Error opening a suitable device");
                                    }

                                    drop(required_queues);
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
