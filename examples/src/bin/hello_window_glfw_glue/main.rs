use vulkan_framework::device::*;
use vulkan_framework::instance::*;
use vulkan_framework::queue_family::*;

use vulkan_framework_glfw_glue;

fn main() {
    let mut instance_extensions = vec![String::from("VK_EXT_debug_utils")];

    let engine_name = String::from("None");
    let app_name = String::from("hello_window");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![String::from("VK_KHR_swapchain")];
    let device_layers: Vec<String> = vec![];

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

                    match window.create_surface(instance.clone()) {
                        Ok(surface) => {
                            println!("Vulkan rendering surface created successfully");

                            if let Ok(_device) = Device::new(
                                instance.clone(),
                                [ConcreteQueueFamilyDescriptor::new(
                                    vec![
                                        QueueFamilySupportedOperationType::Graphics,
                                        QueueFamilySupportedOperationType::Transfer,
                                        QueueFamilySupportedOperationType::Present(surface.clone()),
                                    ]
                                    .as_ref(),
                                    [1.0f32].as_slice(),
                                )]
                                .as_slice(),
                                device_extensions.as_slice().as_ref(),
                                device_layers.as_slice().as_ref(),
                                Some("Opened Device"),
                            ) {
                                println!("Device opened successfully");
                            } else {
                                println!("Error opening a suitable device");
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

    // deinitialize sdl2 context
    vulkan_framework_sdl2_glue::deinit();
}
